import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Any, Optional
import torch.nn.functional as nnf
import time
from datetime import datetime

from envs.trading_environment import TradingEnvironment
from ai.transformer import MultiBranchTransformer
from agent.utils import ReplayBuffer, convert_state_dict_to_tensors
from agent.base_callbacks import TrainingCallback
from agent.callbacks import CallbackManager


class PPOTrainer:
    def _safe_date_format(self, date_obj) -> str:
        """Safely format a date object to YYYY-MM-DD string"""
        if isinstance(date_obj, str):
            return date_obj  # Already a string
        elif hasattr(date_obj, 'strftime'):
            return date_obj.strftime('%Y-%m-%d')
        elif hasattr(date_obj, 'date'):
            return date_obj.date().strftime('%Y-%m-%d')
        else:
            return str(date_obj)

    def __init__(
        self,
        env: TradingEnvironment,
        model: MultiBranchTransformer,
        callback_manager: CallbackManager,
        config: Any,
        device: Optional[Union[str, torch.device]] = None,
        output_dir: str = "./ppo_output",
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        self.env = env
        self.model = model
        self.config = config  # Store full config for curriculum access
        self.callback_manager = callback_manager

        self.logger = logging.getLogger(__name__)

        # Extract training parameters from config
        training_config = config.training
        self.lr = training_config.learning_rate
        self.gamma = training_config.gamma
        self.gae_lambda = training_config.gae_lambda
        self.clip_eps = training_config.clip_epsilon
        self.critic_coef = training_config.value_coef
        self.entropy_coef = training_config.entropy_coef
        self.max_grad_norm = training_config.max_grad_norm
        self.ppo_epochs = training_config.n_epochs
        self.batch_size = training_config.batch_size
        self.rollout_steps = training_config.rollout_steps

        # Store model config - convert to dict if it's a Pydantic object
        model_config = config.model
        if model_config is not None:
            if hasattr(model_config, "model_dump"):
                # It's a Pydantic model, convert to dict for storage
                self.model_config = model_config.model_dump()
            else:
                # Already a dict or other type
                self.model_config = model_config
        else:
            self.model_config = {}

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=self.rollout_steps, device=self.device)

        # Output directories
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Callbacks
        self.callbacks = callbacks if callbacks else []
        
        # Callbacks registered

        # Training state
        self.global_step_counter = 0
        self.global_episode_counter = 0
        self.global_update_counter = 0

        # Initialize episode rewards for use in PPO metrics
        self.recent_episode_rewards = []

        # Performance tracking
        self.is_evaluating = False
        self.training_start_time = 0.0

        # Stage timing
        self.stage_timers = {}

        # Timing metrics tracking
        self.last_update_time = None
        self.update_times = []
        self.episode_times = []

        # Momentum-based training configuration (managed by TrainingManager)
        self.episode_selection_mode = "momentum_days"  # Enable momentum-based training

        # Day selection configuration - extract from config or use defaults
        day_selection_config = getattr(config, "day_selection", None)
        if day_selection_config:
            self.episodes_per_day = day_selection_config.episodes_per_day
            self.reset_point_quality_range = (
                day_selection_config.reset_point_quality_range or [0.0, 1.0]
            )
            self.day_switching_strategy = day_selection_config.day_switching_strategy
        else:
            # Default values if day_selection config not present
            self.episodes_per_day = 10
            self.reset_point_quality_range = [0.0, 1.0]
            self.day_switching_strategy = "exhaustive"

        # Momentum training state
        self.current_momentum_day = None
        self.used_momentum_days = set()
        self.current_reset_points = []
        self.used_reset_point_indices = set()
        # Data quality filtering (managed by TrainingManager)
        self.quality_range = [0.7, 1.0]  # Default quality range

        # Day episode tracking
        self.episodes_completed_on_current_day = 0
        self.reset_point_cycles_completed = 0

        # Training Manager integration for data lifecycle
        self.training_manager = None  # Will be set by TrainingManager

        # Training control
        self.stop_training = False

        self.logger.info(
            f"🤖 PPOTrainer initialized with callback system. Device: {self.device}"
        )

    def set_training_manager(self, training_manager):
        """Set the training manager for data lifecycle integration."""
        self.training_manager = training_manager

    def get_current_training_data(self) -> Optional[Dict[str, Any]]:
        """Get current training data from TrainingManager."""
        if self.training_manager:
            return self.training_manager.get_current_training_data()
        return None

    def _update_current_momentum_day(self) -> bool:
        """Update current momentum day from TrainingManager data lifecycle."""
        if self.episode_selection_mode != "momentum_days":
            return False

        # Get current training data from TrainingManager
        training_data = self.get_current_training_data()
        if not training_data:
            self.logger.warning("🔄 No current training data from TrainingManager")
            return False

        # Update current momentum day from training data
        if 'day_info' in training_data:
            prev_day = getattr(self, 'current_momentum_day', {})
            prev_date = prev_day.get('date') if prev_day else None
            
            self.current_momentum_day = training_data['day_info']
            day_date = self.current_momentum_day.get('date')
            quality = self.current_momentum_day.get('quality_score', 0)
            
            # Only log when day actually changes
            if day_date and day_date != prev_date:
                self.logger.info(
                    f"📅 Using training day: {self._safe_date_format(day_date)} "
                    f"(quality: {quality:.3f})"
                )

            # Trigger momentum day change callback
            enhanced_info = self.current_momentum_day.copy()
            enhanced_info.update({
                "day_date": self._safe_date_format(day_date) if day_date else "unknown",
                "day_quality": quality,
                "episodes_on_day": self.episodes_completed_on_current_day,
                "cycles_completed": self.reset_point_cycles_completed,
            })

            momentum_event_data = {
                "day_info": enhanced_info,
                "reset_points": training_data.get('reset_points', []),
            }
            self.callback_manager.trigger("on_momentum_day_change", momentum_event_data)
            
            return True
        else:
            self.logger.warning("🔄 Training data missing day_info")
            return False

    def _get_filtered_reset_points_for_dashboard(
        self, reset_points_data: List[Dict]
    ) -> List[Dict]:
        """Filter reset points for dashboard display."""
        if not reset_points_data:
            return reset_points_data

        # Simple filtering - just return the data as-is since TrainingManager handles filtering
        return reset_points_data

    def _get_current_reset_point(self) -> Optional[int]:
        """Get current reset point from TrainingManager."""
        if not self.training_manager:
            # Fallback for when TrainingManager is not set - use first reset point
            return 0

        training_data = self.training_manager.get_current_training_data()
        if training_data and 'reset_point_index' in training_data:
            return training_data['reset_point_index']
        
        # Fallback to first reset point
        return 0

    # Curriculum methods removed - now handled by TrainingManager with DataLifecycleManager

    def _should_switch_day(self) -> bool:
        """Get momentum days filtered by stage criteria."""
        if not hasattr(self.env, "data_manager"):
            return []

        all_momentum_days = self.env.data_manager.get_all_momentum_days()
        if not all_momentum_days:
            return []

        filtered_days = []
        for day_info in all_momentum_days:
            # Filter by symbol
            if stage.symbols and day_info.get("symbol") not in stage.symbols:
                continue

            # Filter by date range
            if stage.date_range[0] is not None:
                from datetime import datetime

                start_date = datetime.strptime(stage.date_range[0], "%Y-%m-%d").date()
                # Convert pd.Timestamp to date for comparison
                day_date = (
                    day_info["date"].date()
                    if hasattr(day_info["date"], "date")
                    else day_info["date"]
                )
                if day_date < start_date:
                    continue

            if stage.date_range[1] is not None:
                from datetime import datetime

                end_date = datetime.strptime(stage.date_range[1], "%Y-%m-%d").date()
                # Convert pd.Timestamp to date for comparison
                day_date = (
                    day_info["date"].date()
                    if hasattr(day_info["date"], "date")
                    else day_info["date"]
                )
                if day_date > end_date:
                    continue

            # Filter by day score
            day_score = day_info.get("quality_score", 0)
            if not (stage.day_score_range[0] <= day_score <= stage.day_score_range[1]):
                continue

            filtered_days.append(day_info)

        return filtered_days

    def _get_current_curriculum_stage(self):
        """Placeholder for legacy curriculum stage - returns None since we use adaptive data lifecycle."""
        return None

    def _get_curriculum_stage_info(self):
        """Placeholder for legacy curriculum stage info - returns empty dict since we use adaptive data lifecycle."""
        return {}

    def _on_stage_change(self):
        """Called when curriculum stage changes."""
        self.stage_start_update = self.global_update_counter
        self.stage_start_episode = self.global_episode_counter
        self.stage_cycles_completed = 0
        self.used_momentum_days.clear()
        self.used_reset_point_indices.clear()

        stage = self._get_current_curriculum_stage()
        if stage:
            self.logger.info(
                f"📚 Curriculum stage changed to {self.current_stage_idx + 1}"
            )
            self.logger.info(f"   Day score range: {stage.day_score_range}")
            self.logger.info(f"   ROC range: {stage.roc_range}")
            self.logger.info(f"   Activity range: {stage.activity_range}")

    def _check_stage_completion(self):
        """Check if current stage should be completed."""
        stage = self._get_current_curriculum_stage()
        if not stage:
            return

        # Check max_updates condition
        if stage.max_updates is not None:
            updates_in_stage = self.global_update_counter - self.stage_start_update
            if updates_in_stage >= stage.max_updates:
                self.logger.info(
                    f"Stage completed: max_updates ({stage.max_updates}) reached"
                )
                self._advance_to_next_stage()
                return

        # Check max_episodes condition
        if stage.max_episodes is not None:
            episodes_in_stage = self.global_episode_counter - self.stage_start_episode
            if episodes_in_stage >= stage.max_episodes:
                self.logger.info(
                    f"Stage completed: max_episodes ({stage.max_episodes}) reached"
                )
                self._advance_to_next_stage()
                return

        # Check max_cycles condition
        if stage.max_cycles is not None:
            if self.stage_cycles_completed >= stage.max_cycles:
                self.logger.info(
                    f"Stage completed: max_cycles ({stage.max_cycles}) reached"
                )
                self._advance_to_next_stage()
                return

    def _advance_to_next_stage(self):
        """Advance to next enabled curriculum stage."""
        stages = [
            self.config.env.curriculum.stage_1,
            self.config.env.curriculum.stage_2,
            self.config.env.curriculum.stage_3,
        ]

        # Find next enabled stage
        for i in range(self.current_stage_idx + 1, len(stages)):
            if stages[i].enabled:
                self.current_stage_idx = i
                self._on_stage_change()
                return

        # No more stages - training complete
        self.logger.info("🎉 All curriculum stages completed!")
        # Set a flag to indicate training should stop
        self.stop_training = True
        if hasattr(self, "callbacks"):
            for callback in self.callbacks:
                if hasattr(callback, "on_curriculum_complete"):
                    callback.on_curriculum_complete(self)  # type: ignore

    def _emit_curriculum_progress(self):
        """Trigger curriculum progress callback."""
        stage = self._get_curriculum_stage_info()

        # Calculate stage progress percentage
        stage_progress = 0.0
        stage_name = "unknown"
        # Calculate stage progress percentage based on actual curriculum config
        current_stage = self._get_current_curriculum_stage()
        
        if current_stage:
            stage_name = self._get_stage_name(current_stage)
            total_episodes = self.global_episode_counter
            total_updates = self.global_update_counter
            total_cycles = self.reset_point_cycles_completed
            
            # Calculate progress based on the primary completion criteria
            max_progress = 0.0
            
            if current_stage.max_episodes is not None:
                episode_progress = min(100.0, (total_episodes / current_stage.max_episodes) * 100)
                max_progress = max(max_progress, episode_progress)
                
            if current_stage.max_updates is not None:
                update_progress = min(100.0, (total_updates / current_stage.max_updates) * 100)
                max_progress = max(max_progress, update_progress)
                
            if current_stage.max_cycles is not None:
                cycle_progress = min(100.0, (total_cycles / current_stage.max_cycles) * 100)
                max_progress = max(max_progress, cycle_progress)
                
            stage_progress = max_progress
        else:
            stage_progress = 100.0
            stage_name = "stage_complete"

        curriculum_data = {
            "progress": self.curriculum_progress,
            "stage": stage_name,
            "roc_range": stage.roc_range if stage else [0.0, 1.0],
            "activity_range": stage.activity_range if stage else [0.0, 1.0],
            "total_episodes": self.global_episode_counter,
            "stage_progress": stage_progress,
        }
        self.callback_manager.trigger(
            "on_custom_event", "curriculum_progress", curriculum_data
        )

        # Also emit the detailed curriculum tracking
        self._emit_curriculum_detail()

    def _emit_curriculum_detail(self):
        """Trigger detailed curriculum tracking callback."""
        stage = self._get_curriculum_stage_info()

        # Calculate episodes needed for next stage
        episodes_to_next_stage = 0
        next_stage_name = ""
        current_stage_name = ""
        total_episodes = self.global_episode_counter

        # Calculate episodes needed for next stage based on actual curriculum config
        current_stage = self._get_current_curriculum_stage()
        if current_stage:
            current_stage_name = self._get_stage_name(current_stage)
            
            # Calculate remaining based on configured completion criteria
            remaining = float('inf')
            completion_type = "none"
            
            if current_stage.max_episodes is not None:
                episodes_remaining = current_stage.max_episodes - total_episodes
                if episodes_remaining < remaining:
                    remaining = episodes_remaining
                    completion_type = "episodes"
                    
            if current_stage.max_updates is not None:
                updates_remaining = current_stage.max_updates - self.global_update_counter
                if updates_remaining < remaining:
                    remaining = updates_remaining
                    completion_type = "updates"
                    
            if current_stage.max_cycles is not None:
                cycles_remaining = current_stage.max_cycles - self.reset_point_cycles_completed
                if cycles_remaining < remaining:
                    remaining = cycles_remaining
                    completion_type = "cycles"
            
            # Set episodes_to_next_stage based on completion type
            if completion_type == "episodes":
                episodes_to_next_stage = max(0, int(remaining))
            elif completion_type == "updates":
                episodes_to_next_stage = max(0, int(remaining))  # Show updates as count
            elif completion_type == "cycles":
                episodes_to_next_stage = max(0, int(remaining))  # Show cycles as count
            else:
                episodes_to_next_stage = 0  # No completion criteria set
                
            # Get next stage name
            next_stage = self._get_next_curriculum_stage()
            next_stage_name = self._get_stage_name(next_stage) if next_stage else "Complete"
        else:
            current_stage_name = "Unknown"
            next_stage_name = "Unknown"
            episodes_to_next_stage = 0

        curriculum_detail = {
            "current_stage": current_stage_name,
            "roc_range": stage.roc_range if stage else [0.0, 1.0],
            "activity_range": stage.activity_range if stage else [0.0, 1.0],
            "total_episodes": self.global_episode_counter,
            "episodes_to_next_stage": episodes_to_next_stage,
            "next_stage_name": next_stage_name,
            "episodes_per_day_config": self.episodes_per_day,
            "curriculum_method": self.curriculum_method,
        }
        self.callback_manager.trigger(
            "on_custom_event", "curriculum_detail", curriculum_detail
        )

    def _emit_initial_curriculum_detail(self):
        """Emit initial curriculum detail at training start."""
        self._emit_curriculum_detail()

    def _emit_initial_cycle_tracking(self):
        """Trigger initial cycle tracking after day setup."""
        # Initial cycle tracking with zero values
        cycle_tracking = {
            "cycles_completed": self.reset_point_cycles_completed,
            "target_cycles_per_day": self.episodes_per_day,
            "cycles_remaining_for_day_switch": self.episodes_per_day
            - self.reset_point_cycles_completed,
            "episodes_on_current_day": self.episodes_completed_on_current_day,
            "day_switch_progress_pct": 0.0,
            "current_day_date": self._safe_date_format(self.current_momentum_day["date"])
            if self.current_momentum_day
            else "unknown",
        }
        self.callback_manager.trigger(
            "on_custom_event", "cycle_completion", cycle_tracking
        )

    def _update_curriculum_progress(self):
        """Update curriculum progress based on training performance."""
        # Always emit curriculum progress, even with limited data
        if len(self.recent_episode_rewards) < 5:
            self._emit_curriculum_progress()
            return

        # Calculate recent performance stability
        recent_rewards = self.recent_episode_rewards[-20:]
        if len(recent_rewards) >= 10:
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            # If performance is stable and positive, increase difficulty
            if std_reward < abs(mean_reward) * 0.3 and mean_reward > 0:
                self.curriculum_progress = min(
                    1.0, self.curriculum_progress + 0.05
                )  # Faster progression
            # If performance is poor, decrease difficulty slightly
            elif mean_reward < -1.0:
                self.curriculum_progress = max(0.0, self.curriculum_progress - 0.02)

            # Emit curriculum progress event
            self._emit_curriculum_progress()

    def _should_switch_day(self) -> bool:
        """Determine if we should switch to a new day based on episodes per day configuration."""
        if self.reset_point_cycles_completed >= self.episodes_per_day:
            return True
        return False

    def _reset_environment_with_momentum(self):
        """Reset environment using momentum-based training with configurable day switching."""
        if self.episode_selection_mode == "momentum_days":
            # If TrainingManager is in control, use its data
            if self.training_manager:
                # TrainingManager controls day/reset point selection
                # Just update our local state with current data
                if not self._update_current_momentum_day():
                    self.logger.warning("Failed to get current training data from TrainingManager")
                    return self.env.reset()
                
                # Set up environment if day changed
                current_day = self.current_momentum_day
                if current_day is not None:
                    # Only set up session if it's different from current
                    current_symbol = getattr(self.env, 'primary_asset', None)
                    current_date = getattr(self.env, 'current_session_date', None)
                    
                    # Normalize dates for comparison (handle both string and datetime)
                    new_date_str = self._safe_date_format(current_day["date"])
                    current_date_str = self._safe_date_format(current_date) if current_date else None
                    
                    if (current_symbol != current_day["symbol"] or 
                        current_date_str != new_date_str):
                        self.logger.info(
                            f"📅 Setting up NEW session: {current_day['symbol']} on {new_date_str} "
                            f"(quality: {current_day.get('quality_score', 0):.3f}) "
                            f"[previous: {current_symbol} {current_date_str}]"
                        )
                        self.env.setup_session(
                            symbol=current_day["symbol"], date=current_day["date"]
                        )
                    else:
                        self.logger.debug(
                            f"📅 Reusing session: {current_day['symbol']} on {new_date_str} "
                            f"(no session setup needed)"
                        )
            else:
                # Legacy mode: PPOTrainer manages days independently
                should_switch_day = False

                if self.current_momentum_day is None:
                    should_switch_day = True
                    self.logger.info("🔄 No current momentum day, selecting new day")
                elif self._should_switch_day():
                    should_switch_day = True
                    date_str = self._safe_date_format(self.current_momentum_day["date"])
                    self.logger.info(
                        f"🔄 Completed {self.episodes_completed_on_current_day} episodes "
                        f"({self.reset_point_cycles_completed} cycles) on {date_str}, switching day"
                    )

                # Switch to new momentum day if needed
                if should_switch_day:
                    if not self._update_current_momentum_day():
                        self.logger.warning(
                            "No more momentum days available, reusing current day"
                        )
                        if self.current_momentum_day is None:
                            return self.env.reset()
                    else:
                        # Set up environment with new momentum day
                        current_day = self.current_momentum_day
                        if current_day is not None:
                            self.logger.info(
                                f"📅 Switching to momentum day: {self._safe_date_format(current_day['date'])} "
                                f"(quality: {current_day.get('quality_score', 0):.3f})"
                            )

                            # Only setup session if actually different
                            current_symbol = getattr(self.env, 'primary_asset', None)
                            current_date = getattr(self.env, 'current_session_date', None)
                            new_date_str = self._safe_date_format(current_day["date"])
                            current_date_str = self._safe_date_format(current_date) if current_date else None
                            
                            if (current_symbol != current_day["symbol"] or 
                                current_date_str != new_date_str):
                                self.env.setup_session(
                                    symbol=current_day["symbol"], date=current_day["date"]
                                )
                            else:
                                self.logger.debug(f"📅 Reusing existing session for {current_day['symbol']} {new_date_str}")

                        # Reset day tracking
                        self.episodes_completed_on_current_day = 0
                        self.reset_point_cycles_completed = 0
                        self.used_reset_point_indices.clear()

                        # Trigger initial cycle tracking after day setup
                        self._emit_initial_cycle_tracking()

            # Select reset point and reset environment
            reset_point_idx = self._get_current_reset_point()
            if reset_point_idx is None:
                reset_point_idx = 0  # Safety fallback

            # Track episode completion
            self.episodes_completed_on_current_day += 1

            # Check if we completed a cycle through all reset points
            if not self.env.has_more_reset_points():
                self.reset_point_cycles_completed += 1
                self.used_reset_point_indices.clear()
                self.logger.info(
                    f"🔄 Completed cycle {self.reset_point_cycles_completed} through reset points"
                )

            # Always trigger cycle tracking callback after each episode for real-time updates
            cycles_remaining = self.episodes_per_day - self.reset_point_cycles_completed
            progress_pct = (
                self.reset_point_cycles_completed / max(1, self.episodes_per_day)
            ) * 100

            # Calculate more detailed progress within current cycle
            total_available_points = (
                len(getattr(self.env, "available_reset_points", []))
                if hasattr(self.env, "available_reset_points")
                else 0
            )
            points_used_in_cycle = len(self.used_reset_point_indices)
            points_remaining_in_cycle = max(
                0, total_available_points - points_used_in_cycle
            )

            cycle_tracking = {
                "cycles_completed": self.reset_point_cycles_completed,
                "target_cycles_per_day": self.episodes_per_day,
                "cycles_remaining_for_day_switch": cycles_remaining,
                "episodes_on_current_day": self.episodes_completed_on_current_day,
                "day_switch_progress_pct": progress_pct,
                "current_day_date": self._safe_date_format(self.current_momentum_day["date"])
                if self.current_momentum_day
                else "unknown",
                "total_available_points": total_available_points,
                "points_used_in_cycle": points_used_in_cycle,
                "points_remaining_in_cycle": points_remaining_in_cycle,
            }
            self.callback_manager.trigger(
                "on_custom_event", "cycle_completion", cycle_tracking
            )

            # Note: momentum day progress tracking is done via metrics,
            # reset points data is only sent on actual day changes

            return self.env.reset_at_point(reset_point_idx)
        else:
            # Use standard reset
            return self.env.reset()

    def _start_timer(self, stage: str):
        """Start timing for a stage."""
        self.stage_timers[stage] = time.time()

    def _end_timer(self, stage: str) -> float:
        """End timing for a stage and return duration."""
        if stage in self.stage_timers:
            duration = time.time() - self.stage_timers[stage]
            del self.stage_timers[stage]
            return duration
        return 0.0

    def _convert_action_for_env(self, action_tensor: torch.Tensor) -> Any:
        """Converts model's action tensor to environment-compatible format."""
        # System only uses discrete actions
        if action_tensor.ndim > 0 and action_tensor.shape[-1] == 2:
            return action_tensor.cpu().numpy().squeeze().astype(int)
        else:
            return action_tensor.cpu().numpy().item()

    def collect_rollout_data(self) -> Dict[str, Any]:
        """Collect rollout data with comprehensive logging.

        PPO (Proximal Policy Optimization) collects a fixed number of steps (rollout_steps)
        across potentially multiple episodes before performing a training update. This is
        different from episodic algorithms that wait for episode completion.

        Why PPO works this way:
        1. More stable training - uses large batches of experience
        2. Better sample efficiency - can learn from partial episodes
        3. Consistent compute - predictable training iterations

        Episodes that complete during rollout are automatically reset to continue
        collecting data until rollout_steps is reached.
        """
        self._start_timer("rollout")

        self.logger.info(f"🎯 ROLLOUT START: Collecting {self.rollout_steps} steps")
        self.logger.info(
            "   ℹ️  PPO collects data across multiple episodes before training"
        )
        self.logger.info("   ℹ️  Episodes will reset automatically when they complete")
        self.buffer.clear()

        # Trigger training update callback that we're in rollout phase
        training_data = {
            "mode": "Training",
            "stage": "Collecting Rollout",
            "updates": self.global_update_counter,
            "global_steps": self.global_step_counter,
            "total_episodes": self.global_episode_counter,
            "stage_status": f"Collecting {self.rollout_steps} steps...",
            "time_per_update": np.mean(self.update_times) if self.update_times else 0.0,
            "time_per_episode": np.mean(self.episode_times)
            if self.episode_times
            else 0.0,
        }
        self.callback_manager.trigger(
            "on_custom_event", "training_update", training_data
        )

        current_env_state_np, _ = self._reset_environment_with_momentum()

        # Trigger initial episode start
        reset_info = {
            "symbol": getattr(self.env, "symbol", "UNKNOWN"),
            "date": getattr(self.env, "date", None),
            "momentum_day": self.current_momentum_day,
            "reset_point_idx": 0,
            "max_steps": self.rollout_steps,  # Expected max steps per episode
        }
        self.callback_manager.trigger(
            "on_episode_start", self.global_episode_counter + 1, reset_info
        )

        for callback in self.callbacks:
            callback.on_rollout_start(self)

        # Rollout tracking
        collected_steps = 0
        episode_rewards_in_rollout = []
        episode_lengths_in_rollout = []
        episode_details = []
        current_episode_reward = 0.0
        current_episode_length = 0
        episode_start_time = time.time()
        total_invalid_actions = 0

        while collected_steps < self.rollout_steps:
            # Check for training interruption EVERY step for immediate response
            if self.stop_training or (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning(
                    f"Training interrupted during rollout collection at step {collected_steps}"
                )
                self.stop_training = True
                break

            # Trigger rollout progress callback periodically
            if collected_steps % 100 == 0:
                training_data = {
                    "mode": "Training",
                    "stage": "Collecting Rollouts",
                    "updates": self.global_update_counter,
                    "global_steps": self.global_step_counter,
                    "total_episodes": self.global_episode_counter,
                    "rollout_steps": collected_steps,
                    "rollout_total": self.rollout_steps,
                    "stage_status": f"Collecting: {collected_steps}/{self.rollout_steps} steps",
                    "time_per_update": np.mean(self.update_times)
                    if self.update_times
                    else 0.0,
                    "time_per_episode": np.mean(self.episode_times)
                    if self.episode_times
                    else 0.0,
                }
                self.callback_manager.trigger(
                    "on_custom_event", "training_update", training_data
                )
            single_step_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in current_env_state_np.items()
            }

            current_model_state_torch_batched = {}
            for key, tensor_val in single_step_tensors.items():
                if key in ["hf", "mf", "lf", "portfolio"]:
                    if tensor_val.ndim == 2:
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)
                    elif tensor_val.ndim == 3 and tensor_val.shape[0] == 1:
                        current_model_state_torch_batched[key] = tensor_val
                    else:
                        current_model_state_torch_batched[key] = tensor_val
                else:
                    current_model_state_torch_batched[key] = tensor_val

            with torch.no_grad():
                action_tensor, action_info = self.model.get_action(
                    current_model_state_torch_batched, deterministic=False
                )

            env_action = self._convert_action_for_env(action_tensor)

            # Trigger model forward callback with internals
            forward_data = {
                "features": current_model_state_torch_batched,
                "action": action_tensor,
                "action_info": action_info,
                "step_num": collected_steps,
            }

            # Add attention weights if available
            if hasattr(self.model, "get_last_attention_weights"):
                attention_weights = self.model.get_last_attention_weights()
                if attention_weights is not None:
                    forward_data["attention_weights"] = attention_weights

            # Add action probabilities if available
            if hasattr(self.model, "get_last_action_probabilities"):
                action_probs = self.model.get_last_action_probabilities()
                if action_probs is not None:
                    forward_data["action_probabilities"] = action_probs

            self.callback_manager.trigger("on_model_forward", forward_data)

            try:
                next_env_state_np, reward, terminated, truncated, info = self.env.step(
                    env_action
                )
                done = terminated or truncated

                # Check for interruption immediately after step
                if self.stop_training or (
                    hasattr(__import__("main"), "training_interrupted")
                    and __import__("main").training_interrupted
                ):
                    self.logger.warning(
                        f"Training interrupted during step execution at step {collected_steps}"
                    )
                    self.stop_training = True
                    # Return incomplete rollout data immediately
                    break

                # Track invalid actions
                if info.get("invalid_action_in_step", False):
                    total_invalid_actions += 1

            except Exception as e:
                import traceback

                self.logger.error(f"Error during environment step: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                break

            self.buffer.add(
                current_env_state_np,
                action_tensor,
                reward,
                next_env_state_np,
                done,
                action_info,
            )

            current_env_state_np = next_env_state_np
            collected_steps += 1
            current_episode_reward += reward
            current_episode_length += 1

            # Update step tracking
            self.global_step_counter += 1

            # Trigger episode step callback with all necessary data
            step_data = {
                "action": env_action,
                "reward": reward,
                "info": info,
                "step": collected_steps,
                "max_steps": getattr(
                    self.env, "max_steps", 256
                ),  # Use environment's max steps
            }
            self.callback_manager.trigger("on_episode_step", step_data)

            for callback in self.callbacks:
                callback.on_step(
                    self,
                    current_model_state_torch_batched,
                    action_tensor,
                    reward,
                    next_env_state_np,
                    info,
                )

            if done:
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time

                self.global_episode_counter += 1
                episode_rewards_in_rollout.append(current_episode_reward)
                episode_lengths_in_rollout.append(current_episode_length)

                # Store episode details for summary
                episode_details.append(
                    {
                        "reward": current_episode_reward,
                        "length": current_episode_length,
                        "duration": episode_duration,
                        "final_equity": info.get("portfolio_equity", 0),
                        "termination_reason": info.get("termination_reason", "UNKNOWN"),
                        "truncated": info.get("truncated", False),
                        "pnl": info.get("total_pnl", 0),
                        "win_rate": info.get("win_rate", 0),
                        "trades": info.get("total_trades", 0),
                    }
                )

                # Trigger episode end callback
                episode_data = {
                    "episode_num": self.global_episode_counter,
                    "reward": current_episode_reward,
                    "length": current_episode_length,
                    "final_equity": info.get("portfolio_equity", 0),
                    "termination_reason": info.get("termination_reason", "UNKNOWN"),
                    "truncated": info.get("truncated", False),
                    "pnl": info.get("total_pnl", 0),
                    "win_rate": info.get("win_rate", 0),
                    "trades": info.get("total_trades", 0),
                    "trainer": self,  # Add trainer reference for Captum callback
                }
                self.callback_manager.trigger(
                    "on_episode_end", self.global_episode_counter, episode_data
                )

                # Update recent episode rewards for dashboard
                self.recent_episode_rewards.append(current_episode_reward)

                if len(self.recent_episode_rewards) > 10:  # Keep only last 10 episodes
                    self.recent_episode_rewards.pop(0)

                # Track episode timing
                self.episode_times.append(episode_duration)
                if len(self.episode_times) > 20:  # Keep last 20 episodes
                    self.episode_times.pop(0)

                # Log key episode metrics for interpretation
                if self.global_episode_counter % 10 == 0:  # Log every 10th episode
                    pnl = info.get("total_pnl", 0)
                    win_rate = info.get("win_rate", 0)
                    trades = info.get("total_trades", 0)
                    hold_ratio = info.get("hold_ratio", 0)

                    self.logger.info(
                        f"📊 Episode {self.global_episode_counter} Summary:"
                    )
                    self.logger.info(
                        f"   💵 PnL: ${pnl:.2f} | Reward: {current_episode_reward:.3f}"
                    )
                    self.logger.info(
                        f"   📈 Win Rate: {win_rate:.1f}% | Trades: {trades}"
                    )
                    self.logger.info(
                        f"   ⏸️  Hold Ratio: {hold_ratio:.1f}% | Steps: {current_episode_length}"
                    )
                    self.logger.info(
                        f"   🏁 Reason: {info.get('termination_reason', 'UNKNOWN')}"
                    )

                for callback in self.callbacks:
                    callback.on_episode_end(
                        self, current_episode_reward, current_episode_length, info
                    )

                # Update environment training info to sync episode numbers
                self.env.set_training_info(
                    episode_num=self.global_episode_counter,
                    total_episodes=self.global_episode_counter,
                    total_steps=self.global_step_counter,
                    update_count=self.global_update_counter,
                )

                current_env_state_np, _ = self._reset_environment_with_momentum()
                current_episode_reward = 0.0
                current_episode_length = 0
                episode_start_time = time.time()

                # Log that we're starting a new episode within the same rollout
                if collected_steps < self.rollout_steps:
                    remaining_steps = self.rollout_steps - collected_steps
                    self.logger.info(
                        f"🔄 Starting new episode within rollout | "
                        f"Steps collected: {collected_steps}/{self.rollout_steps} | "
                        f"Remaining: {remaining_steps}"
                    )

                # Trigger new episode start callback
                reset_info = {
                    "symbol": getattr(self.env, "symbol", "UNKNOWN"),
                    "date": getattr(self.env, "date", None),
                    "momentum_day": self.current_momentum_day,
                    "reset_point_idx": len(self.used_reset_point_indices) - 1,
                    "max_steps": self.rollout_steps,  # Expected max steps per episode
                }
                self.callback_manager.trigger(
                    "on_episode_start", self.global_episode_counter + 1, reset_info
                )

                if collected_steps >= self.rollout_steps:
                    break

        # Calculate comprehensive rollout metrics
        rollout_duration = self._end_timer("rollout")
        steps_per_second = (
            collected_steps / rollout_duration if rollout_duration > 0 else 0
        )
        mean_episode_reward = (
            np.mean(episode_rewards_in_rollout) if episode_rewards_in_rollout else 0
        )
        std_episode_reward = (
            np.std(episode_rewards_in_rollout)
            if len(episode_rewards_in_rollout) > 1
            else 0
        )
        mean_episode_length = (
            np.mean(episode_lengths_in_rollout) if episode_lengths_in_rollout else 0
        )

        # Prepare rollout data for callbacks
        rollout_data = {
            "collected_steps": collected_steps,
            "num_episodes": len(episode_rewards_in_rollout),
            "episode_rewards": episode_rewards_in_rollout,
            "episode_lengths": episode_lengths_in_rollout,
            "mean_reward": mean_episode_reward,
            "std_reward": std_episode_reward,
            "rollout_time": rollout_duration,
        }

        # Trigger rollout end callback
        self.callback_manager.trigger("on_rollout_end", rollout_data)

        for callback in self.callbacks:
            callback.on_rollout_end(self)

        self.buffer.prepare_data_for_training()

        # Termination reason analysis
        termination_counts = {}
        if episode_details:
            for ep in episode_details:
                reason = ep["termination_reason"]
                termination_counts[reason] = termination_counts.get(reason, 0) + 1

        rollout_stats = {
            "collected_steps": collected_steps,
            "mean_reward": mean_episode_reward,
            "std_reward": std_episode_reward,
            "mean_episode_length": mean_episode_length,
            "num_episodes_in_rollout": len(episode_rewards_in_rollout),
            "rollout_time": rollout_duration,
            "steps_per_second": steps_per_second,
            "global_step_counter": self.global_step_counter,
            "global_episode_counter": self.global_episode_counter,
            "invalid_actions": total_invalid_actions,
        }

        # Calculate aggregate metrics for interpretation
        if episode_details:
            avg_pnl = np.mean([ep["pnl"] for ep in episode_details])
            avg_win_rate = np.mean([ep["win_rate"] for ep in episode_details])
            avg_trades = np.mean([ep["trades"] for ep in episode_details])
        else:
            # No episodes completed in this rollout - use zeros
            avg_pnl = 0.0
            avg_win_rate = 0.0
            avg_trades = 0.0

        # Comprehensive rollout summary
        self.logger.info("🎯 ROLLOUT COMPLETE:")
        self.logger.info(
            f"   ⏱️  Duration: {rollout_duration:.1f}s ({steps_per_second:.1f} steps/s)"
        )
        self.logger.info(
            f"   📊 Episodes: {len(episode_rewards_in_rollout)} | Steps: {collected_steps:,}"
        )
        self.logger.info(
            f"   💰 Rewards: μ={mean_episode_reward:.3f} σ={std_episode_reward:.3f}"
        )
        self.logger.info(
            f"   💵 Avg PnL: ${avg_pnl:.2f} | Win Rate: {avg_win_rate:.1f}%"
        )
        self.logger.info(
            f"   📈 Avg Trades: {avg_trades:.1f} | Avg Length: {mean_episode_length:.1f} steps"
        )

        if total_invalid_actions > 0:
            invalid_rate = (total_invalid_actions / collected_steps) * 100
            self.logger.info(
                f"   ⚠️  Invalid Actions: {total_invalid_actions} ({invalid_rate:.1f}%)"
            )

        if termination_counts:
            top_reasons = sorted(
                termination_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            reasons_str = " | ".join(
                [f"{reason}: {count}" for reason, count in top_reasons]
            )
            self.logger.info(f"   🏁 Terminations: {reasons_str}")

        return rollout_stats

    def _compute_advantages_and_returns(self):
        """Computes GAE advantages and returns, storing them in the buffer."""
        if (
            self.buffer.rewards is None
            or self.buffer.values is None
            or self.buffer.dones is None
        ):
            self.logger.error("Cannot compute advantages: buffer data not prepared.")
            return

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        num_steps = len(rewards)

        advantages = torch.zeros_like(values, device=self.device)
        last_gae_lam = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                if dones[t]:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    next_value = values[t].clone().detach()
            else:
                if dones[t]:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    next_value = values[t + 1]

            if next_value.ndim > 1 and next_value.size(1) > 1:
                next_value = next_value[:, 0]

            delta = (
                rewards[t]
                + self.gamma * next_value * (1.0 - dones[t].float())
                - values[t]
            )
            advantages[t] = last_gae_lam = (
                delta
                + self.gamma * self.gae_lambda * (1.0 - dones[t].float()) * last_gae_lam
            )

        self.buffer.advantages = advantages

        if values.ndim > 1 and values.shape[1] > 1:
            values = values[:, 0:1]

        returns = advantages + values

        if returns.ndim > 1 and returns.shape[1] > 1:
            returns = returns[:, 0:1]

        self.buffer.returns = returns

    def update_policy(self) -> Dict[str, float]:
        """PPO policy update with detailed logging."""
        # Check for interruption before starting update
        if self.stop_training or (
            hasattr(__import__("main"), "training_interrupted")
            and __import__("main").training_interrupted
        ):
            self.logger.warning("Training interrupted before policy update")
            return {"interrupted": True}

        self._start_timer("update")

        self.logger.info(f"🔄 UPDATE START: Update #{self.global_update_counter + 1}")

        # Trigger update start callback
        self.callback_manager.trigger("on_update_start", self.global_update_counter)


        # Trigger training update callback that we're in update phase
        # Calculate current performance metrics
        current_time = time.time()
        elapsed_time = current_time - self.training_start_time
        steps_per_second = (
            self.global_step_counter / elapsed_time if elapsed_time > 0 else 0
        )
        episodes_per_hour = (
            (self.global_episode_counter / elapsed_time) * 3600
            if elapsed_time > 0
            else 0
        )
        updates_per_second = (
            self.global_update_counter / elapsed_time if elapsed_time > 0 else 0
        )

        training_data = {
            "mode": "Training",
            "stage": "Updating Policy",
            "updates": self.global_update_counter,
            "global_steps": self.global_step_counter,
            "total_episodes": self.global_episode_counter,
            "stage_status": f"PPO Update {self.global_update_counter + 1}...",
            "steps_per_second": steps_per_second,
            "episodes_per_hour": episodes_per_hour,
            "updates_per_second": updates_per_second,
            "time_per_update": np.mean(self.update_times) if self.update_times else 0.0,
            "time_per_episode": np.mean(self.episode_times)
            if self.episode_times
            else 0.0,
        }
        self.callback_manager.trigger(
            "on_custom_event", "training_update", training_data
        )

        self._compute_advantages_and_returns()

        training_data = self.buffer.get_training_data()
        if training_data is None:
            self.logger.error(
                "Skipping policy update due to missing training data in buffer."
            )
            return {}

        states_dict = training_data["states"]
        actions = training_data["actions"]
        old_log_probs = training_data["old_log_probs"]
        advantages = training_data["advantages"]
        returns = training_data["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = actions.size(0)
        if num_samples == 0:
            self.logger.warning(
                "No samples in buffer to update policy. Skipping update."
            )
            return {}

        indices = np.arange(num_samples)

        for callback in self.callbacks:
            callback.on_update_start(self)

        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        num_updates_in_epoch = 0
        total_batches = (num_samples + self.batch_size - 1) // self.batch_size

        # Log update details
        self.logger.info(
            f"   📊 Processing {num_samples} samples in {total_batches} batches"
        )
        self.logger.info(
            f"   🔁 Running {self.ppo_epochs} PPO epochs with batch size {self.batch_size}"
        )

        # PPO-specific metrics tracking
        total_clipfrac = 0
        total_approx_kl = 0
        total_explained_variance = 0
        total_gradient_norm = 0

        for epoch in range(self.ppo_epochs):
            # Check for training interruption before each epoch
            if self.stop_training or (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning(
                    f"Training interrupted during policy update at epoch {epoch}"
                )
                # Return partial update results immediately
                avg_actor_loss = total_actor_loss / max(1, num_updates_in_epoch)
                avg_critic_loss = total_critic_loss / max(1, num_updates_in_epoch)
                return {
                    "policy_loss": avg_actor_loss,
                    "value_loss": avg_critic_loss,
                    "interrupted": True
                }

            np.random.shuffle(indices)

            # Log epoch start
            self.logger.info(f"📚 PPO Epoch {epoch + 1}/{self.ppo_epochs} starting...")

            current_batch = 0
            epoch_start_time = time.time()

            for start_idx in range(0, num_samples, self.batch_size):
                current_batch += 1

                # Log batch progress periodically (every 10 batches or for small batch counts)
                if current_batch % 10 == 0 or total_batches < 20:
                    batch_progress = (current_batch / total_batches) * 100
                    self.logger.info(
                        f"   📦 Batch {current_batch}/{total_batches} ({batch_progress:.1f}%)"
                    )

                # Trigger training update callback with epoch/batch progress
                training_data = {
                    "mode": "Training",
                    "stage": "PPO Update",
                    "updates": self.global_update_counter,
                    "global_steps": self.global_step_counter,
                    "total_episodes": self.global_episode_counter,
                    "current_epoch": epoch + 1,
                    "total_epochs": self.ppo_epochs,
                    "current_batch": current_batch,
                    "total_batches": total_batches,
                    "batch_size": self.batch_size,
                    "stage_status": f"Epoch {epoch + 1}/{self.ppo_epochs}, Batch {current_batch}/{total_batches}",
                }
                self.callback_manager.trigger(
                    "on_custom_event", "training_update", training_data
                )
                # Ensure batch indices don't exceed available samples
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Safely extract batch data with bounds checking
                try:
                    batch_states = {
                        key: tensor_val[batch_indices]
                        for key, tensor_val in states_dict.items()
                    }
                    batch_actions = actions[batch_indices]

                    # Store last batch for monitoring
                    self._last_batch_states = batch_states
                    self._last_batch_actions = batch_actions
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                except IndexError as e:
                    self.logger.error(f"Index error in batch extraction: {e}")
                    self.logger.error(
                        f"Batch indices: {batch_indices}, num_samples: {num_samples}"
                    )
                    continue

                action_params, current_values = self.model(batch_states)

                if batch_returns.ndim > 1 and batch_returns.shape[1] > 1:
                    batch_returns = batch_returns[:, 0:1]
                elif batch_returns.ndim == 1:
                    batch_returns = batch_returns.unsqueeze(1)

                if batch_advantages.ndim > 1 and batch_advantages.shape[1] > 1:
                    batch_advantages = batch_advantages[:, 0:1]
                elif batch_advantages.ndim == 1:
                    batch_advantages = batch_advantages.unsqueeze(1)

                # System only uses discrete actions
                action_type_logits, action_size_logits = action_params

                action_types_taken = batch_actions[:, 0].long()
                action_sizes_taken = batch_actions[:, 1].long()

                type_dist = torch.distributions.Categorical(logits=action_type_logits)
                size_dist = torch.distributions.Categorical(logits=action_size_logits)

                new_type_log_probs = type_dist.log_prob(action_types_taken)
                new_size_log_probs = size_dist.log_prob(action_sizes_taken)
                new_log_probs = (new_type_log_probs + new_size_log_probs).unsqueeze(1)

                entropy = (type_dist.entropy() + size_dist.entropy()).unsqueeze(1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate PPO metrics
                with torch.no_grad():
                    clipfrac = torch.mean(
                        (torch.abs(ratio - 1.0) > self.clip_eps).float()
                    ).item()
                    total_clipfrac += clipfrac

                    approx_kl = torch.mean(batch_old_log_probs - new_log_probs).item()
                    total_approx_kl += approx_kl

                    var_y = torch.var(batch_returns)
                    explained_var = 1 - torch.var(
                        batch_returns - current_values.view(-1, 1)
                    ) / (var_y + 1e-8)
                    total_explained_variance += explained_var.item()

                current_values_shaped = current_values.view(-1, 1)
                batch_returns_shaped = batch_returns.view(-1, 1)

                if current_values_shaped.size(0) != batch_returns_shaped.size(0):
                    min_size = min(
                        current_values_shaped.size(0), batch_returns_shaped.size(0)
                    )
                    current_values_shaped = current_values_shaped[:min_size]
                    batch_returns_shaped = batch_returns_shaped[:min_size]

                critic_loss = nnf.mse_loss(current_values_shaped, batch_returns_shaped)
                entropy_loss = -entropy.mean()
                loss = (
                    actor_loss
                    + self.critic_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                # Track gradient norm
                grad_norm = 0
                if self.max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    total_gradient_norm += float(grad_norm)

                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.mean().item()
                num_updates_in_epoch += 1

            # Log epoch completion
            epoch_duration = time.time() - epoch_start_time
            batches_per_second = (
                total_batches / epoch_duration if epoch_duration > 0 else 0
            )
            self.logger.info(
                f"   ✅ Epoch {epoch + 1}/{self.ppo_epochs} complete | "
                f"Time: {epoch_duration:.1f}s | "
                f"Batches/s: {batches_per_second:.1f}"
            )

        self.global_update_counter += 1


        # Calculate averages
        avg_actor_loss = (
            total_actor_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        )
        avg_critic_loss = (
            total_critic_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        )
        avg_entropy = (
            total_entropy_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        )
        avg_clipfrac = (
            total_clipfrac / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        )
        avg_approx_kl = (
            total_approx_kl / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        )
        avg_explained_variance = (
            total_explained_variance / num_updates_in_epoch
            if num_updates_in_epoch > 0
            else 0
        )
        avg_gradient_norm = (
            total_gradient_norm / num_updates_in_epoch
            if num_updates_in_epoch > 0
            else 0
        )

        # Get actual learning rate from optimizer in case it has been modified by scheduler
        current_lr = self.optimizer.param_groups[0]["lr"]

        # All metrics are now included in update_metrics dict for callbacks

        # Trigger PPO metrics callback
        mean_reward = (
            np.mean(self.recent_episode_rewards)
            if len(self.recent_episode_rewards) > 0
            else 0
        )
        ppo_data = {
            "learning_rate": current_lr,
            "mean_episode_reward": mean_reward,
            "policy_loss": avg_actor_loss,
            "value_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "total_loss": avg_actor_loss + avg_critic_loss,
            "clip_fraction": avg_clipfrac,
            "kl_divergence": avg_approx_kl,
            "explained_variance": avg_explained_variance,
        }
        self.callback_manager.trigger("on_custom_event", "ppo_metrics", ppo_data)

        # End update timing
        update_duration = self._end_timer("update")

        # Track update timing
        self.update_times.append(update_duration)
        if len(self.update_times) > 20:  # Keep last 20 updates
            self.update_times.pop(0)

        # Calculate performance metrics for dashboard
        current_time = time.time()
        elapsed_time = current_time - self.training_start_time
        steps_per_second = (
            self.global_step_counter / elapsed_time if elapsed_time > 0 else 0
        )
        episodes_per_hour = (
            (self.global_episode_counter / elapsed_time) * 3600
            if elapsed_time > 0
            else 0
        )
        updates_per_second = (
            self.global_update_counter / elapsed_time if elapsed_time > 0 else 0
        )
        updates_per_hour = updates_per_second * 3600

        update_metrics = {
            # Use consistent naming with ppo_data
            "policy_loss": avg_actor_loss,
            "value_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "clip_fraction": avg_clipfrac,
            "kl_divergence": avg_approx_kl,
            "explained_variance": avg_explained_variance,
            "gradient_norm": avg_gradient_norm,
            "total_loss": avg_actor_loss + avg_critic_loss,
            "mean_episode_reward": mean_reward,
            "global_step_counter": self.global_step_counter,
            "global_episode_counter": self.global_episode_counter,
            "global_update_counter": self.global_update_counter,
            "update_time": update_duration,
            "learning_rate": current_lr,
            "total_steps": self.global_step_counter,  # Add for performance metrics
            # Add performance metrics for dashboard
            "steps_per_second": steps_per_second,
            "episodes_per_hour": episodes_per_hour,
            "updates_per_second": updates_per_second,
            "updates_per_hour": updates_per_hour,
            # Add batch data for monitoring
            "batch_data": {
                "states": getattr(self, "_last_batch_states", None),
                "actions": getattr(self, "_last_batch_actions", None),
                "buffer_size": self.buffer.get_size(),
            },
            "trainer": self,  # Add trainer reference for Captum callback
        }

        # Comprehensive update summary with interpretation hints
        self.logger.info("🔄 UPDATE COMPLETE:")
        self.logger.info(
            f"   ⏱️  Duration: {update_duration:.1f}s | Batches: {total_batches}"
        )
        self.logger.info(
            f"   🎭 Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}"
        )
        self.logger.info(
            f"   📊 Entropy: {avg_entropy:.4f} (↓=converging) | Clip Rate: {avg_clipfrac * 100:.1f}% (target<30%)"
        )
        self.logger.info(
            f"   🧠 KL Div: {avg_approx_kl:.4f} (<0.01 stable) | Explained Var: {avg_explained_variance * 100:.1f}% (>80% good)"
        )
        self.logger.info(f"   📈 Grad Norm: {avg_gradient_norm:.4f}")

        # Add interpretation warnings
        if avg_clipfrac > 0.3:
            self.logger.warning(
                "   ⚠️  High clip rate - consider reducing learning rate"
            )
        if avg_approx_kl > 0.02:
            self.logger.warning(
                "   ⚠️  High KL divergence - updates may be too aggressive"
            )
        if avg_explained_variance < 0.5:
            self.logger.warning(
                "   ⚠️  Low explained variance - value function may need tuning"
            )

        # Trigger update end callback
        self.callback_manager.trigger(
            "on_update_end", self.global_update_counter, update_metrics
        )

        for callback in self.callbacks:
            callback.on_update_end(self, update_metrics)

        # Trigger training update callback after update completes
        training_data = {
            "mode": "Training",
            "stage": "Preparing Next Rollout",
            "updates": self.global_update_counter,
            "global_steps": self.global_step_counter,
            "total_episodes": self.global_episode_counter,
            "stage_progress": 0.0,  # Reset stage progress
            "stage_status": "Update completed, preparing next rollout...",
            "time_per_update": np.mean(self.update_times) if self.update_times else 0.0,
            "time_per_episode": np.mean(self.episode_times)
            if self.episode_times
            else 0.0,
        }
        self.callback_manager.trigger(
            "on_custom_event", "training_update", training_data
        )

        return update_metrics

    def train_with_manager(self) -> Dict[str, Any]:
        """
        Train using the new TrainingManager system
        This replaces the old curriculum-based train method
        """
        from training.training_manager import TrainingManager
        
        # Get training manager configuration
        training_manager_config = self.config.env.training_manager
        mode = training_manager_config.mode
        
        # Get available momentum days for data lifecycle with adaptive data filtering
        available_days = []
        if hasattr(self.env, 'data_manager') and hasattr(self.env.data_manager, 'get_all_momentum_days'):
            # Extract date range and symbols from adaptive data configuration
            adaptive_data_config = training_manager_config.data_lifecycle.adaptive_data
            symbols = adaptive_data_config.symbols if adaptive_data_config.symbols else None
            
            # Parse date range from config
            start_date = None
            end_date = None
            start_date_str = None
            end_date_str = None
            
            if adaptive_data_config.date_range and len(adaptive_data_config.date_range) >= 2:
                start_date_str = adaptive_data_config.date_range[0]
                end_date_str = adaptive_data_config.date_range[1]
                
                if start_date_str:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                if end_date_str:
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            self.logger.info(f"🔍 Loading momentum days with filters:")
            self.logger.info(f"   📊 Symbols: {symbols}")
            self.logger.info(f"   📅 Date range: {start_date_str or 'None'} to {end_date_str or 'None'}")
            
            momentum_days_dicts = self.env.data_manager.get_all_momentum_days(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                min_activity=0.0  # No activity filtering here, let data lifecycle handle it
            )
            
            self.logger.info(f"   📊 Found {len(momentum_days_dicts)} momentum days after filtering")
            
            # Convert dictionary format to DayInfo objects for DataLifecycleManager
            from training.data_lifecycle_manager import DayInfo, ResetPointInfo
            for day_dict in momentum_days_dicts:
                # Get reset points for this day
                reset_points = []
                if hasattr(self.env.data_manager, 'get_reset_points'):
                    reset_points_df = self.env.data_manager.get_reset_points(
                        day_dict['symbol'], day_dict['date']
                    )
                    # Convert reset points to ResetPointInfo objects
                    for _, rp_row in reset_points_df.iterrows():
                        reset_point = ResetPointInfo(
                            timestamp=rp_row['timestamp'],
                            quality_score=rp_row.get('combined_score', 0.5),
                            roc_score=rp_row.get('roc_score', 0.0),
                            activity_score=rp_row.get('activity_score', 0.5),
                            price=rp_row.get('price', 0.0)
                        )
                        reset_points.append(reset_point)
                
                # Convert date to string format for DayInfo
                date_str = self._safe_date_format(day_dict['date'])
                
                day_info = DayInfo(
                    date=date_str,
                    symbol=day_dict['symbol'],
                    day_score=day_dict.get('quality_score', 0.5),
                    reset_points=reset_points
                )
                available_days.append(day_info)
            
        # Initialize TrainingManager with available days
        training_manager = TrainingManager(training_manager_config.__dict__, mode, available_days)
        
        self.logger.info(f"🎯 Starting training with TrainingManager in {mode} mode")
        
        # Episode configuration is now handled directly via env.max_steps config
        self.logger.info(f"🎯 Using episode configuration: max_steps={self.env.max_steps}")
        
        # The training manager will call our training step methods
        # We need to implement the interface it expects
        self.training_manager = training_manager
        
        # Start training with manager (it will control the lifecycle)
        final_stats = training_manager.start_training(self)
        
        return final_stats
    
    def run_training_step(self) -> bool:
        """
        Run one training step for TrainingManager integration
        Returns True if training should continue, False to stop
        """
        try:
            # Check for training interruption
            if (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning("Training interrupted during training step")
                return False
            
            # Collect rollout data
            rollout_info = self.collect_rollout_data()
            
            # Check for interruption after rollout
            if (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning("Training interrupted after rollout collection")
                return False
            
            # Check buffer size
            if (
                self.buffer.get_size() < self.rollout_steps
                and self.buffer.get_size() < self.batch_size
            ):
                if self.buffer.get_size() < self.batch_size:
                    return True  # Continue training, just skip this update
                    
            # Update policy
            update_metrics = self.update_policy()
            
            # Check if update was interrupted
            if update_metrics.get("interrupted", False):
                self.logger.warning("Policy update was interrupted")
                return False
            
            # Check for interruption after update
            if (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning("Training interrupted after policy update")
                return False
            
            # Trigger callbacks
            for callback in self.callbacks:
                callback.on_update_iteration_end(
                    self, self.global_update_counter, update_metrics, rollout_info
                )
            
            # Check if TrainingManager wants us to stop
            if hasattr(self, 'training_manager') and self.training_manager is not None:
                return not self.training_manager.should_stop
            
            return True  # Continue training
            
        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            return False  # Stop training on error
    
    def apply_data_difficulty_change(self, change: Dict[str, Any]):
        """Apply data difficulty changes from TrainingManager"""
        if 'quality_range' in change:
            quality_range = change['quality_range']
            self.logger.info(f"📊 Applying data difficulty change: {quality_range}")
            
            # Update data filtering if we have momentum-based training
            if self.episode_selection_mode == "momentum_days" and hasattr(self.env, 'data_manager'):
                # Apply quality range to data manager
                if hasattr(self.env.data_manager, 'set_quality_filter'):
                    self.env.data_manager.set_quality_filter(quality_range)

    def train(self, eval_freq_steps: Optional[int] = None):
        """Main training loop with curriculum-driven stopping."""
        self.logger.info("🚀 TRAINING START: Curriculum-driven training")
        self.logger.info(
            f"   🎯 Rollout size: {self.rollout_steps} | Batch size: {self.batch_size}"
        )
        self.logger.info(
            f"   🔄 PPO epochs: {self.ppo_epochs} | Learning rate: {self.lr}"
        )

        # Log curriculum plan
        stage = self._get_current_curriculum_stage()
        if stage:
            self.logger.info(
                f"   📚 Starting curriculum stage with max_updates: {stage.max_updates}, max_cycles: {stage.max_cycles}"
            )
        else:
            self.logger.warning("   ⚠️ No active curriculum stage found!")

        # Start training
        self.training_start_time = time.time()

        # Trigger training start callback with comprehensive config
        training_config = {
            # PPO specific parameters
            "rollout_steps": self.rollout_steps,
            "batch_size": self.batch_size,
            "ppo_epochs": self.ppo_epochs,
            "learning_rate": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_eps,
            "value_coef": self.critic_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            # Training configuration
            "curriculum_stage": self._get_current_curriculum_stage(),
            "momentum_training": self.episode_selection_mode == "momentum_days",
            "device": str(self.device),
            # Include full config for WandB and other callbacks that need it
            "full_config": self.config,
            "experiment_name": getattr(self.config, "experiment_name", "training"),
            # Model info
            "model_config": self.model_config,
            "model": self.model,  # Add model for attribution callback
            "trainer": self,  # Add trainer reference for callbacks that need it
        }
        self.callback_manager.trigger("on_training_start", training_config)

        for callback in self.callbacks:
            callback.on_training_start(self)

        # Initialize training state
        training_data = {
            "mode": "Training",
            "stage": "Initializing",
            "updates": 0,
            "global_steps": 0,
            "total_episodes": 0,
            "overall_progress": 0.0,
            "stage_progress": 0.0,
            "stage_status": "Starting training...",
            "steps_per_second": 0.0,
            "time_per_update": 0.0,
            "time_per_episode": 0.0,
        }
        self.callback_manager.trigger(
            "on_custom_event", "training_update", training_data
        )

        # Initial training setup (curriculum system removed)

        # Trigger initial reset point tracking (with default values)
        initial_reset_tracking = {
            "selected_index": 0,
            "selected_timestamp": "Initial Training Start",
            "total_available_points": 0,
            "points_used_in_cycle": 0,
            "points_remaining_in_cycle": 0,
            "roc_score": 0.0,
            "activity_score": 0.0,
            "roc_range": [0.8, 1.0],
            "activity_range": [0.5, 1.0],
            "curriculum_stage": "stage_1",
        }
        self.callback_manager.trigger("on_reset_point_selected", initial_reset_tracking)

        best_eval_reward = -float("inf")

        while not self.stop_training:
            # Check for training interruption
            if (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning("Training interrupted during training loop")
                break

            rollout_info = self.collect_rollout_data()

            # Check if rollout was interrupted
            if self.stop_training or (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning("Training interrupted during or after rollout collection")
                break

            if (
                self.buffer.get_size() < self.rollout_steps
                and self.buffer.get_size() < self.batch_size
            ):
                self.logger.warning(
                    f"Buffer size {self.buffer.get_size()} too small. Skipping update."
                )
                if self.buffer.get_size() < self.batch_size:
                    continue

            update_metrics = self.update_policy()

            # Check if update was interrupted
            if update_metrics.get("interrupted", False) or self.stop_training or (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning("Training interrupted during or after policy update")
                break

            for callback in self.callbacks:
                callback.on_update_iteration_end(
                    self, self.global_update_counter, update_metrics, rollout_info
                )

            # Curriculum-driven progress logging
            elapsed_time = time.time() - self.training_start_time
            steps_per_second = (
                self.global_step_counter / elapsed_time if elapsed_time > 0 else 0
            )
            episodes_per_hour = (
                (self.global_episode_counter / elapsed_time) * 3600
                if elapsed_time > 0
                else 0
            )
            updates_per_second = (
                self.global_update_counter / elapsed_time if elapsed_time > 0 else 0
            )

            # Calculate curriculum stage progress
            stage = self._get_current_curriculum_stage()
            stage_progress = 0.0
            stage_name = "No Active Stage"

            if stage:
                stage_name = f"Stage {self.current_stage_idx + 1}"

                # Calculate progress based on stage limits
                if stage.max_updates is not None:
                    updates_in_stage = (
                        self.global_update_counter - self.stage_start_update
                    )
                    stage_progress = min(
                        100.0, (updates_in_stage / stage.max_updates) * 100
                    )
                elif stage.max_episodes is not None:
                    episodes_in_stage = (
                        self.global_episode_counter - self.stage_start_episode
                    )
                    stage_progress = min(
                        100.0, (episodes_in_stage / stage.max_episodes) * 100
                    )
                elif stage.max_cycles is not None:
                    stage_progress = min(
                        100.0, (self.stage_cycles_completed / stage.max_cycles) * 100
                    )

            # Always trigger training progress callback (not just every 5 updates)
            training_data = {
                "mode": "Training",
                "stage": "Active Training",
                "updates": self.global_update_counter,
                "global_steps": self.global_step_counter,
                "total_episodes": self.global_episode_counter,
                "overall_progress": stage_progress,
                "stage_progress": stage_progress,
                "stage_status": f"{stage_name} - Update {self.global_update_counter}",
                "steps_per_second": steps_per_second,
                "episodes_per_hour": episodes_per_hour,
                "updates_per_second": updates_per_second,
                "time_per_update": update_metrics.get("update_time", 0)
                if "update_metrics" in locals()
                else 0,
                "time_per_episode": rollout_info.get("rollout_time", 0)
                / max(1, rollout_info.get("num_episodes_in_rollout", 1))
                if "rollout_info" in locals()
                else 0,
            }
            self.callback_manager.trigger(
                "on_custom_event", "training_update", training_data
            )

            if self.global_update_counter % 5 == 0:  # Log every 5 updates
                # Calculate recent performance trends
                recent_rewards = (
                    self.recent_episode_rewards[-10:]
                    if len(self.recent_episode_rewards) > 0
                    else []
                )
                recent_mean = np.mean(recent_rewards) if recent_rewards else 0
                recent_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0

                steps_per_hour = getattr(self, "steps_per_second", 0) * 3600
                self.logger.info(f"📈 PROGRESS: {stage_name} ({stage_progress:.1f}%)")
                self.logger.info(
                    f"   ⏱️  Rate: {steps_per_hour:.0f} steps/hr | Steps: {self.global_step_counter:,}"
                )
                self.logger.info(
                    f"   🏆 Episodes: {self.global_episode_counter} | Updates: {self.global_update_counter}"
                )
                self.logger.info(
                    f"   📊 Recent Performance: μ={recent_mean:.3f} σ={recent_std:.3f}"
                )

            # Periodic training analysis every 25 updates
            if self.global_update_counter % 25 == 0 and self.global_update_counter > 0:
                self._log_training_analysis(update_metrics)

            # Evaluation
            eval_freq_updates = (
                max(1, eval_freq_steps // self.rollout_steps) if eval_freq_steps else 0
            )
            if eval_freq_steps and self.global_update_counter % eval_freq_updates == 0:
                eval_stats = self.evaluate(n_episodes=10)

                for callback in self.callbacks:
                    eval_metrics = {
                        f"eval/{k}": v
                        for k, v in eval_stats.items()
                        if k not in ["episode_rewards", "episode_lengths"]
                    }
                    eval_metrics["global_step"] = self.global_step_counter
                    callback.on_update_iteration_end(
                        self, self.global_update_counter, eval_metrics, {}
                    )

                if eval_stats["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_stats["mean_reward"]
                    best_model_path = os.path.join(
                        self.model_dir,
                        f"best_model_update_{self.global_update_counter}.pt",
                    )
                    self.save_model(best_model_path)

                latest_model_path = os.path.join(self.model_dir, "latest_model.pt")
                self.save_model(latest_model_path)

        # Training completion
        total_time = time.time() - self.training_start_time
        final_stats = {
            "total_steps_trained": self.global_step_counter,
            "total_updates": self.global_update_counter,
            "total_episodes": self.global_episode_counter,
            "training_time_hours": total_time / 3600,
        }

        self.logger.info("🎉 TRAINING COMPLETE!")
        self.logger.info(f"   ⏱️  Total time: {total_time / 3600:.2f} hours")
        self.logger.info(
            f"   📊 Final stats: {self.global_step_counter:,} steps | {self.global_episode_counter} episodes | {self.global_update_counter} updates"
        )

        # Trigger training end callback
        self.callback_manager.trigger("on_training_end", final_stats)

        for callback in self.callbacks:
            callback.on_training_end(self, final_stats)

        return final_stats

    def evaluate(
        self, n_episodes: int = 10, deterministic: bool = True
    ) -> Dict[str, Any]:
        """Evaluation with detailed logging."""
        self.logger.info(f"🔍 EVALUATION START: {n_episodes} episodes")
        self._start_timer("evaluation")

        # Trigger evaluation start callback and training update
        self.callback_manager.trigger("on_evaluation_start")
        
        # Update training stage to show evaluation in progress
        eval_training_data = {
            "mode": "Evaluation",
            "stage": "Running Evaluation",
            "updates": self.global_update_counter,
            "global_steps": self.global_step_counter,
            "total_episodes": self.global_episode_counter,
            "stage_status": f"Evaluating model with {n_episodes} episodes...",
            "is_evaluating": True,
        }
        self.callback_manager.trigger("on_custom_event", "training_update", eval_training_data)

        self.model.eval()
        self.is_evaluating = True

        episode_rewards = []
        episode_lengths = []
        episode_details = []

        for i in range(n_episodes):
            # Check for training interruption during evaluation
            if (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                self.logger.warning(
                    f"Training interrupted during evaluation at episode {i}"
                )
                break
            
            # Update evaluation progress
            eval_progress = (i / n_episodes) * 100
            eval_training_data = {
                "mode": "Evaluation",
                "stage": "Running Evaluation",
                "updates": self.global_update_counter,
                "global_steps": self.global_step_counter,
                "total_episodes": self.global_episode_counter,
                "stage_status": f"Evaluating episode {i+1}/{n_episodes} ({eval_progress:.1f}%)",
                "stage_progress": eval_progress,
                "is_evaluating": True,
            }
            self.callback_manager.trigger("on_custom_event", "training_update", eval_training_data)

            env_state_np, _ = self._reset_environment_with_momentum()
            current_episode_reward = 0.0
            current_episode_length = 0
            done = False

            while not done:
                model_state_torch = convert_state_dict_to_tensors(
                    env_state_np, self.device
                )
                with torch.no_grad():
                    action_tensor, _ = self.model.get_action(
                        model_state_torch, deterministic=deterministic
                    )

                env_action = self._convert_action_for_env(action_tensor)

                try:
                    next_env_state_np, reward, terminated, truncated, info = (
                        self.env.step(env_action)
                    )
                    done = terminated or truncated
                except Exception as e:
                    self.logger.error(f"Error during evaluation step: {e}")
                    done = True
                    reward = 0
                    next_env_state_np = env_state_np  # Keep current state on error
                    info = {}

                env_state_np = next_env_state_np
                current_episode_reward += reward
                current_episode_length += 1

            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            episode_details.append(
                {
                    "reward": current_episode_reward,
                    "length": current_episode_length,
                    "final_equity": info.get("portfolio_equity", 0),
                }
            )

        # Trigger evaluation end callback
        eval_results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0,
            "n_episodes": len(episode_rewards),
        }
        self.callback_manager.trigger("on_evaluation_end", eval_results)

        self.model.train()
        self.is_evaluating = False

        eval_duration = self._end_timer("evaluation")

        eval_results = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0,
            "min_reward": np.min(episode_rewards) if episode_rewards else 0,
            "max_reward": np.max(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        # Comprehensive evaluation summary
        self.logger.info("🔍 EVALUATION COMPLETE:")
        self.logger.info(f"   ⏱️  Duration: {eval_duration:.1f}s")
        self.logger.info(
            f"   💰 evaluation_mean_reward={eval_results['mean_reward']:.3f} evaluation_std_reward={eval_results['std_reward']:.3f}"
        )
        self.logger.info(
            f"   📊 Range: [{eval_results['min_reward']:.3f}, {eval_results['max_reward']:.3f}]"
        )
        self.logger.info(f"   📏 Avg Length: {eval_results['mean_length']:.1f} steps")

        return eval_results

    def save_model(self, path: str) -> None:
        """Saves the model and optimizer state."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "global_step_counter": self.global_step_counter,
                    "global_episode_counter": self.global_episode_counter,
                    "global_update_counter": self.global_update_counter,
                    "model_config": self.model_config,
                },
                path,
            )
        except Exception as e:
            self.logger.error(f"Error saving model to {path}: {e}")

    def load_model(self, path: str) -> None:
        """Loads the model and optimizer state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.global_step_counter = checkpoint.get("global_step_counter", 0)
            self.global_episode_counter = checkpoint.get("global_episode_counter", 0)
            self.global_update_counter = checkpoint.get("global_update_counter", 0)

            self.model.to(self.device)
            self.logger.info(
                f"Model loaded from {path}. Resuming from step {self.global_step_counter}"
            )
        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {e}")

    def _log_training_analysis(self, update_metrics: Dict[str, float]) -> None:
        """Log comprehensive training analysis for interpretation."""
        self.logger.info("=" * 80)
        self.logger.info(
            "🔬 TRAINING ANALYSIS - Update {}".format(self.global_update_counter)
        )
        self.logger.info("=" * 80)

        # Performance trends
        recent_rewards = (
            self.recent_episode_rewards[-30:]
            if len(self.recent_episode_rewards) > 0
            else []
        )
        if len(recent_rewards) >= 10:
            first_10 = np.mean(recent_rewards[:10])
            last_10 = np.mean(recent_rewards[-10:])
            trend = last_10 - first_10
            trend_pct = (trend / abs(first_10) * 100) if first_10 != 0 else 0.0

            self.logger.info("📈 PERFORMANCE TREND:")
            self.logger.info(f"   First 10 episodes: {first_10:.3f}")
            self.logger.info(f"   Last 10 episodes: {last_10:.3f}")
            self.logger.info(f"   Trend: {trend:+.3f} ({trend_pct:+.1f}%)")

            # Diagnose performance issues
            if trend < -0.1:
                self.logger.warning("   ⚠️  Performance declining - check for:")
                self.logger.warning("      - Overfitting to recent data")
                self.logger.warning("      - Learning rate too high")
                self.logger.warning("      - Reward component imbalance")
            elif abs(trend) < 0.01:
                self.logger.warning("   ⚠️  Performance plateaued - consider:")
                self.logger.warning(
                    "      - Increasing exploration (entropy coefficient)"
                )
                self.logger.warning("      - Adjusting reward weights")
                self.logger.warning("      - Checking for data diversity")

        # Learning stability
        self.logger.info("\n🧠 LEARNING STABILITY:")
        kl = update_metrics.get("approx_kl", 0)
        clipfrac = update_metrics.get("clipfrac", 0)
        entropy = update_metrics.get("entropy", 0)

        stability_score = "STABLE"
        if kl > 0.02 or clipfrac > 0.3:
            stability_score = "UNSTABLE"
        elif kl > 0.01 or clipfrac > 0.2:
            stability_score = "BORDERLINE"

        self.logger.info(f"   Status: {stability_score}")
        self.logger.info(f"   KL Divergence: {kl:.4f}")
        self.logger.info(f"   Clip Fraction: {clipfrac * 100:.1f}%")
        self.logger.info(f"   Entropy: {entropy:.4f}")

        # Value function quality
        explained_var = update_metrics.get("value_function_explained_variance", 0)
        critic_loss = update_metrics.get("critic_loss", 0)

        self.logger.info("\n📊 VALUE FUNCTION:")
        self.logger.info(f"   Explained Variance: {explained_var * 100:.1f}%")
        self.logger.info(f"   Critic Loss: {critic_loss:.4f}")

        if explained_var < 0.7:
            self.logger.warning("   ⚠️  Poor value estimation - actions:")
            self.logger.warning("      - Increase critic coefficient")
            self.logger.warning("      - Check feature quality")
            self.logger.warning("      - Verify reward normalization")

        # Action recommendations
        self.logger.info("\n💡 RECOMMENDATIONS:")

        if stability_score == "UNSTABLE":
            self.logger.info("   1. Reduce learning rate by 50%")
            self.logger.info("   2. Decrease PPO clip range")
            self.logger.info("   3. Increase batch size")
        elif stability_score == "BORDERLINE":
            self.logger.info("   1. Monitor next few updates closely")
            self.logger.info("   2. Consider small learning rate reduction")

        if entropy < 0.01:
            self.logger.info(
                "   - Increase entropy coefficient to encourage exploration"
            )
        elif entropy > 0.1:
            self.logger.info("   - Decrease entropy coefficient to focus learning")

        self.logger.info("=" * 80)
