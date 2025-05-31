import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Any, Optional
import torch.nn.functional as nnf
import time

from config.schemas import ModelConfig
from envs.trading_environment import TradingEnvironment
from ai.transformer import MultiBranchTransformer
from agent.utils import ReplayBuffer, convert_state_dict_to_tensors
from agent.base_callbacks import TrainingCallback
from metrics.factory import MetricsIntegrator


class PPOTrainer:
    def __init__(
            self,
            env: TradingEnvironment,
            model: MultiBranchTransformer,
            metrics_integrator: MetricsIntegrator,
            model_config: ModelConfig = None,
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            critic_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            ppo_epochs: int = 10,
            batch_size: int = 64,
            rollout_steps: int = 2048,
            device: Optional[Union[str, torch.device]] = None,
            output_dir: str = "./ppo_output",
            callbacks: Optional[List[TrainingCallback]] = None,
            curriculum_strategy: str = "quality_based",
            min_quality_threshold: float = 0.3,
            episode_selection_mode: str = "momentum_days",
            episodes_per_day: int = 10,
            reset_point_quality_range: List[float] = None,
            day_switching_strategy: str = "exhaustive",
    ):
        self.env = env
        self.model = model
        # Store model config - convert to dict if it's a Pydantic object
        if model_config is not None:
            if hasattr(model_config, 'model_dump'):
                # It's a Pydantic model, convert to dict for storage
                self.model_config = model_config.model_dump()
            else:
                # Already a dict or other type
                self.model_config = model_config
        else:
            self.model_config = {}
        self.metrics = metrics_integrator

        self.logger = logging.getLogger(__name__)

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps

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

        # Momentum-based training configuration
        self.curriculum_strategy = curriculum_strategy
        self.min_quality_threshold = min_quality_threshold
        self.episode_selection_mode = episode_selection_mode
        
        # Day selection configuration
        self.episodes_per_day = episodes_per_day
        self.reset_point_quality_range = reset_point_quality_range or [0.0, 1.0]
        self.day_switching_strategy = day_switching_strategy
        
        # Momentum training state
        self.current_momentum_day = None
        self.used_momentum_days = set()
        self.current_reset_points = []
        self.used_reset_point_indices = set()
        self.curriculum_progress = 0.0  # 0.0 = easy episodes, 1.0 = hard episodes
        
        # Day episode tracking
        self.episodes_completed_on_current_day = 0
        self.reset_point_cycles_completed = 0

        self.logger.info(f"🤖 PPOTrainer initialized with metrics integration. Device: {self.device}")

    def _select_next_momentum_day(self) -> bool:
        """Select next momentum day based on curriculum strategy."""
        if self.episode_selection_mode != "momentum_days":
            return False
            
        # Try to get a new momentum day from environment
        momentum_day_info = self.env.select_next_momentum_day(
            exclude_dates=list(self.used_momentum_days)
        )
        
        if momentum_day_info is None:
            self.logger.warning("No more momentum days available, resetting used days")
            self.used_momentum_days.clear()
            momentum_day_info = self.env.select_next_momentum_day()
            
        if momentum_day_info is None:
            self.logger.error("No momentum days available in environment")
            return False
            
        self.current_momentum_day = momentum_day_info
        self.used_momentum_days.add(momentum_day_info['date'])
        self.used_reset_point_indices.clear()
        
        self.logger.info(f"📅 Selected momentum day: {momentum_day_info['date'].strftime('%Y-%m-%d')} "
                        f"(quality: {momentum_day_info.get('quality_score', 0):.3f})")
        
        # Emit momentum day change event with tracking information and reset points
        if hasattr(self.metrics, 'metrics_manager'):
            # Get reset points data from environment for this day
            reset_points_data = []
            if hasattr(self.env, 'data_manager'):
                reset_points_df = self.env.data_manager.get_reset_points(
                    momentum_day_info.get('symbol', 'MLGO'), 
                    momentum_day_info['date']
                )
                if not reset_points_df.empty:
                    reset_points_data = reset_points_df.to_dict('records')
            
            # Enhance momentum_day_info with tracking data
            enhanced_info = momentum_day_info.copy()
            enhanced_info.update({
                'day_date': momentum_day_info['date'].strftime('%Y-%m-%d'),
                'day_quality': momentum_day_info.get('quality_score', 0.0),
                'episodes_on_day': self.episodes_completed_on_current_day,
                'cycles_completed': self.reset_point_cycles_completed,
                'total_days_used': len(self.used_momentum_days)
            })
            
            # Emit event with both day info and reset points
            self.metrics.metrics_manager.emit_event('momentum_day_change', {
                'day_info': enhanced_info,
                'reset_points': reset_points_data
            })
        
        return True

    def _select_reset_point(self) -> int:
        """Select next reset point based on curriculum strategy and quality range filtering."""
        # Get current reset points from environment
        if not hasattr(self.env, 'reset_points') or not self.env.reset_points:
            return 0
            
        reset_points = self.env.reset_points
        
        # Filter reset points by quality range
        min_quality, max_quality = self.reset_point_quality_range
        quality_filtered_indices = []
        for i, reset_point in enumerate(reset_points):
            quality_score = reset_point.get('activity_score', reset_point.get('quality_score', 0.5))
            if min_quality <= quality_score <= max_quality:
                quality_filtered_indices.append(i)
        
        if not quality_filtered_indices:
            # If no reset points match quality range, use all
            self.logger.warning(f"No reset points match quality range {self.reset_point_quality_range}, using all")
            quality_filtered_indices = list(range(len(reset_points)))
        
        # Filter by used indices
        available_indices = [i for i in quality_filtered_indices 
                            if i not in self.used_reset_point_indices]
        
        if not available_indices:
            # Reset used indices when all quality-filtered points are exhausted
            self.used_reset_point_indices.clear()
            available_indices = quality_filtered_indices
            
        if self.curriculum_strategy == "quality_based":
            # Start with easier (lower activity) reset points, progress to harder ones
            sorted_indices = sorted(available_indices, 
                                  key=lambda i: reset_points[i].get('activity_score', 0))
            
            # Select based on curriculum progress
            progress_idx = int(self.curriculum_progress * len(sorted_indices))
            progress_idx = min(progress_idx, len(sorted_indices) - 1)
            selected_idx = sorted_indices[progress_idx]
            
        elif self.curriculum_strategy == "random":
            selected_idx = np.random.choice(available_indices)
        else:
            # Sequential
            selected_idx = available_indices[0]
            
        self.used_reset_point_indices.add(selected_idx)
        
        reset_point = reset_points[selected_idx]
        self.logger.debug(f"🎯 Selected reset point {selected_idx}: "
                         f"{reset_point.get('timestamp', 'unknown')} "
                         f"(activity: {reset_point.get('activity_score', 0):.3f})")
        
        return selected_idx

    def _update_curriculum_progress(self):
        """Update curriculum progress based on training performance."""
        if len(self.recent_episode_rewards) < 10:
            return
            
        # Calculate recent performance stability
        recent_rewards = self.recent_episode_rewards[-20:]
        if len(recent_rewards) >= 10:
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            
            # If performance is stable and positive, increase difficulty
            if std_reward < abs(mean_reward) * 0.3 and mean_reward > 0:
                self.curriculum_progress = min(1.0, self.curriculum_progress + 0.02)
            # If performance is poor, decrease difficulty slightly
            elif mean_reward < -1.0:
                self.curriculum_progress = max(0.0, self.curriculum_progress - 0.01)
                
            # Emit curriculum progress event
            if hasattr(self.metrics, 'metrics_manager'):
                self.metrics.metrics_manager.emit_event('curriculum_progress', {
                    'progress': self.curriculum_progress,
                    'strategy': self.curriculum_strategy
                })

    def _should_switch_day(self) -> bool:
        """Determine if we should switch to a new day based on episodes per day configuration."""
        if self.reset_point_cycles_completed >= self.episodes_per_day:
            return True
        return False

    def _reset_environment_with_momentum(self):
        """Reset environment using momentum-based training with configurable day switching."""
        if self.episode_selection_mode == "momentum_days":
            # Check if we need to switch to a new momentum day
            should_switch_day = False
            
            if self.current_momentum_day is None:
                should_switch_day = True
                self.logger.info("🔄 No current momentum day, selecting new day")
            elif self._should_switch_day():
                should_switch_day = True
                date_str = self.current_momentum_day['date'].strftime('%Y-%m-%d')
                self.logger.info(f"🔄 Completed {self.episodes_completed_on_current_day} episodes "
                               f"({self.reset_point_cycles_completed} cycles) on {date_str}, switching day")
            
            # Switch to new momentum day if needed
            if should_switch_day:
                if not self._select_next_momentum_day():
                    self.logger.warning("No more momentum days available, reusing current day")
                    if self.current_momentum_day is None:
                        return self.env.reset()
                else:
                    # Set up environment with new momentum day
                    current_day = self.current_momentum_day
                    self.logger.info(f"📅 Switching to momentum day: {current_day['date'].strftime('%Y-%m-%d')} "
                                   f"(quality: {current_day.get('quality_score', 0):.3f})")
                    
                    self.env.setup_session(
                        symbol=current_day['symbol'], 
                        date=current_day['date']
                    )
                    
                    # Reset day tracking
                    self.episodes_completed_on_current_day = 0
                    self.reset_point_cycles_completed = 0
                    self.used_reset_point_indices.clear()
                    
            # Select reset point and reset environment
            reset_point_idx = self._select_reset_point()
            
            # Track episode completion
            self.episodes_completed_on_current_day += 1
            
            # Check if we completed a cycle through all reset points
            if not self.env.has_more_reset_points():
                self.reset_point_cycles_completed += 1
                self.used_reset_point_indices.clear()
                self.logger.info(f"🔄 Completed cycle {self.reset_point_cycles_completed} through reset points")
            
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
        if self.model.continuous_action:
            action_np = action_tensor.cpu().numpy().squeeze()
            return np.array([action_np], dtype=np.float32) if np.isscalar(action_np) else action_np.astype(np.float32)
        else:
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
        self.logger.info(f"   ℹ️  PPO collects data across multiple episodes before training")
        self.logger.info(f"   ℹ️  Episodes will reset automatically when they complete")
        self.buffer.clear()
        
        # Update dashboard that we're in rollout phase
        if hasattr(self.metrics, "metrics_manager"):
            training_data = {
                'mode': 'Training',
                'stage': 'Collecting Rollout',
                'updates': self.global_update_counter,
                'global_steps': self.global_step_counter,
                'total_episodes': self.global_episode_counter,
                'stage_status': f"Collecting {self.rollout_steps} steps...",
                'time_per_update': np.mean(self.update_times) if self.update_times else 0.0,
                'time_per_episode': np.mean(self.episode_times) if self.episode_times else 0.0
            }
            self.metrics.metrics_manager.emit_event("training_update", training_data)

        current_env_state_np, _ = self._reset_environment_with_momentum()

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
            # Update rollout progress periodically
            if collected_steps % 100 == 0 and hasattr(self.metrics, "metrics_manager"):
                training_data = {
                    'mode': 'Training',
                    'stage': 'Collecting Rollouts',
                    'updates': self.global_update_counter,
                    'global_steps': self.global_step_counter,
                    'total_episodes': self.global_episode_counter,
                    'rollout_steps': collected_steps,
                    'rollout_total': self.rollout_steps,
                    'stage_status': f"Collecting: {collected_steps}/{self.rollout_steps} steps",
                    'time_per_update': np.mean(self.update_times) if self.update_times else 0.0,
                    'time_per_episode': np.mean(self.episode_times) if self.episode_times else 0.0
                }
                self.metrics.metrics_manager.emit_event("training_update", training_data)
            single_step_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in current_env_state_np.items()
            }

            current_model_state_torch_batched = {}
            for key, tensor_val in single_step_tensors.items():
                if key in ['hf', 'mf', 'lf', 'portfolio']:
                    if tensor_val.ndim == 2:
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)
                    elif tensor_val.ndim == 3 and tensor_val.shape[0] == 1:
                        current_model_state_torch_batched[key] = tensor_val
                    else:
                        current_model_state_torch_batched[key] = tensor_val
                else:
                    current_model_state_torch_batched[key] = tensor_val

            with torch.no_grad():
                action_tensor, action_info = self.model.get_action(current_model_state_torch_batched,
                                                                   deterministic=False)

            env_action = self._convert_action_for_env(action_tensor)
            
            # Track model internals
            if hasattr(self.model, 'get_last_attention_weights'):
                attention_weights = self.model.get_last_attention_weights()
                if attention_weights is not None:
                    self.metrics.update_attention_weights(attention_weights)
            
            if hasattr(self.model, 'get_last_action_probabilities'):
                action_probs = self.model.get_last_action_probabilities()
                if action_probs is not None:
                    self.metrics.update_action_probabilities(action_probs)
            
            # Track feature statistics periodically
            if collected_steps % 100 == 0:
                self.metrics.update_feature_statistics(current_env_state_np)

            try:
                next_env_state_np, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated

                # Track invalid actions
                if info.get('invalid_action_in_step', False):
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
                action_info
            )

            current_env_state_np = next_env_state_np
            collected_steps += 1
            current_episode_reward += reward
            current_episode_length += 1

            # Update step tracking
            self.global_step_counter += 1
            self.metrics.update_step(self.global_step_counter)

            for callback in self.callbacks:
                callback.on_step(self, current_model_state_torch_batched, action_tensor, reward, next_env_state_np, info)

            if done:
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time

                self.global_episode_counter += 1
                episode_rewards_in_rollout.append(current_episode_reward)
                episode_lengths_in_rollout.append(current_episode_length)

                # Store episode details for summary
                episode_details.append({
                    'reward': current_episode_reward,
                    'length': current_episode_length,
                    'duration': episode_duration,
                    'final_equity': info.get('portfolio_equity', 0),
                    'termination_reason': info.get('termination_reason', 'UNKNOWN'),
                    'truncated': info.get('truncated', False),
                    'pnl': info.get('total_pnl', 0),
                    'win_rate': info.get('win_rate', 0),
                    'trades': info.get('total_trades', 0)
                })

                # Record episode metrics
                self.metrics.end_episode(current_episode_reward, current_episode_length)
                
                # Update recent episode rewards for dashboard
                self.recent_episode_rewards.append(current_episode_reward)
                
                # Update curriculum progress based on performance
                self._update_curriculum_progress()
                if len(self.recent_episode_rewards) > 10:  # Keep only last 10 episodes
                    self.recent_episode_rewards.pop(0)
                    
                # Track episode timing
                self.episode_times.append(episode_duration)
                if len(self.episode_times) > 20:  # Keep last 20 episodes
                    self.episode_times.pop(0)

                # Log key episode metrics for interpretation
                if self.global_episode_counter % 10 == 0:  # Log every 10th episode
                    pnl = info.get('total_pnl', 0)
                    win_rate = info.get('win_rate', 0)
                    trades = info.get('total_trades', 0)
                    hold_ratio = info.get('hold_ratio', 0)
                    
                    self.logger.info(f"📊 Episode {self.global_episode_counter} Summary:")
                    self.logger.info(f"   💵 PnL: ${pnl:.2f} | Reward: {current_episode_reward:.3f}")
                    self.logger.info(f"   📈 Win Rate: {win_rate:.1f}% | Trades: {trades}")
                    self.logger.info(f"   ⏸️  Hold Ratio: {hold_ratio:.1f}% | Steps: {current_episode_length}")
                    self.logger.info(f"   🏁 Reason: {info.get('termination_reason', 'UNKNOWN')}")
                
                for callback in self.callbacks:
                    callback.on_episode_end(self, current_episode_reward, current_episode_length, info)

                # Update environment training info to sync episode numbers
                self.env.set_training_info(
                    episode_num=self.global_episode_counter,
                    total_episodes=self.global_episode_counter,
                    total_steps=self.global_step_counter,
                    update_count=self.global_update_counter
                )

                current_env_state_np, _ = self._reset_environment_with_momentum()
                current_episode_reward = 0.0
                current_episode_length = 0
                episode_start_time = time.time()
                
                # Log that we're starting a new episode within the same rollout
                if collected_steps < self.rollout_steps:
                    remaining_steps = self.rollout_steps - collected_steps
                    self.logger.info(f"🔄 Starting new episode within rollout | "
                                   f"Steps collected: {collected_steps}/{self.rollout_steps} | "
                                   f"Remaining: {remaining_steps}")

                # Start new episode tracking
                self.metrics.start_episode()

                if collected_steps >= self.rollout_steps:
                    break

        for callback in self.callbacks:
            callback.on_rollout_end(self)

        self.buffer.prepare_data_for_training()

        # Calculate comprehensive rollout metrics
        rollout_duration = self._end_timer("rollout")
        self.metrics.record_rollout_time(rollout_duration)

        steps_per_second = collected_steps / rollout_duration if rollout_duration > 0 else 0
        mean_episode_reward = np.mean(episode_rewards_in_rollout) if episode_rewards_in_rollout else 0
        std_episode_reward = np.std(episode_rewards_in_rollout) if len(episode_rewards_in_rollout) > 1 else 0
        mean_episode_length = np.mean(episode_lengths_in_rollout) if episode_lengths_in_rollout else 0

        # Termination reason analysis
        termination_counts = {}
        if episode_details:
            for ep in episode_details:
                reason = ep['termination_reason']
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
            "invalid_actions": total_invalid_actions
        }

        # Calculate aggregate metrics for interpretation
        if episode_details:
            avg_pnl = np.mean([ep['pnl'] for ep in episode_details])
            avg_win_rate = np.mean([ep['win_rate'] for ep in episode_details]) 
            avg_trades = np.mean([ep['trades'] for ep in episode_details])
        else:
            # No episodes completed in this rollout - use zeros
            avg_pnl = 0.0
            avg_win_rate = 0.0
            avg_trades = 0.0
        
        # Comprehensive rollout summary
        self.logger.info(f"🎯 ROLLOUT COMPLETE:")
        self.logger.info(f"   ⏱️  Duration: {rollout_duration:.1f}s ({steps_per_second:.1f} steps/s)")
        self.logger.info(f"   📊 Episodes: {len(episode_rewards_in_rollout)} | Steps: {collected_steps:,}")
        self.logger.info(f"   💰 Rewards: μ={mean_episode_reward:.3f} σ={std_episode_reward:.3f}")
        self.logger.info(f"   💵 Avg PnL: ${avg_pnl:.2f} | Win Rate: {avg_win_rate:.1f}%")
        self.logger.info(f"   📈 Avg Trades: {avg_trades:.1f} | Avg Length: {mean_episode_length:.1f} steps")

        if total_invalid_actions > 0:
            invalid_rate = (total_invalid_actions / collected_steps) * 100
            self.logger.info(f"   ⚠️  Invalid Actions: {total_invalid_actions} ({invalid_rate:.1f}%)")

        if termination_counts:
            top_reasons = sorted(termination_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            reasons_str = " | ".join([f"{reason}: {count}" for reason, count in top_reasons])
            self.logger.info(f"   🏁 Terminations: {reasons_str}")

        return rollout_stats

    def _compute_advantages_and_returns(self):
        """Computes GAE advantages and returns, storing them in the buffer."""
        if self.buffer.rewards is None or self.buffer.values is None or self.buffer.dones is None:
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

            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t].float()) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (
                    1.0 - dones[t].float()) * last_gae_lam

        self.buffer.advantages = advantages

        if values.ndim > 1 and values.shape[1] > 1:
            values = values[:, 0:1]

        returns = advantages + values

        if returns.ndim > 1 and returns.shape[1] > 1:
            returns = returns[:, 0:1]

        self.buffer.returns = returns

    def update_policy(self) -> Dict[str, float]:
        """PPO policy update with detailed logging."""
        self._start_timer("update")

        self.logger.info(f"🔄 UPDATE START: Update #{self.global_update_counter + 1}")

        # Start update timing
        self.metrics.start_update()
        
        # Update dashboard that we're in update phase
        if hasattr(self.metrics, "metrics_manager"):
            # Calculate current performance metrics
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            steps_per_second = self.global_step_counter / elapsed_time if elapsed_time > 0 else 0
            episodes_per_hour = (self.global_episode_counter / elapsed_time) * 3600 if elapsed_time > 0 else 0
            
            training_data = {
                'mode': 'Training',
                'stage': 'Updating Policy',
                'updates': self.global_update_counter,
                'global_steps': self.global_step_counter,
                'total_episodes': self.global_episode_counter,
                'stage_status': f"PPO Update {self.global_update_counter + 1}...",
                'steps_per_second': steps_per_second,
                'episodes_per_hour': episodes_per_hour,
                'time_per_update': np.mean(self.update_times) if self.update_times else 0.0,
                'time_per_episode': np.mean(self.episode_times) if self.episode_times else 0.0
            }
            self.metrics.metrics_manager.emit_event("training_update", training_data)

        self._compute_advantages_and_returns()

        training_data = self.buffer.get_training_data()
        if training_data is None:
            self.logger.error("Skipping policy update due to missing training data in buffer.")
            return {}

        states_dict = training_data["states"]
        actions = training_data["actions"]
        old_log_probs = training_data["old_log_probs"]
        advantages = training_data["advantages"]
        returns = training_data["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = actions.size(0)
        if num_samples == 0:
            self.logger.warning("No samples in buffer to update policy. Skipping update.")
            return {}

        indices = np.arange(num_samples)

        for callback in self.callbacks:
            callback.on_update_start(self)

        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        num_updates_in_epoch = 0
        total_batches = (num_samples + self.batch_size - 1) // self.batch_size

        # Log update details
        self.logger.info(f"   📊 Processing {num_samples} samples in {total_batches} batches")
        self.logger.info(f"   🔁 Running {self.ppo_epochs} PPO epochs with batch size {self.batch_size}")

        # PPO-specific metrics tracking
        total_clipfrac = 0
        total_approx_kl = 0
        total_explained_variance = 0
        total_gradient_norm = 0

        for epoch in range(self.ppo_epochs):
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
                    self.logger.info(f"   📦 Batch {current_batch}/{total_batches} ({batch_progress:.1f}%)")
                
                # Update dashboard with epoch/batch progress
                if hasattr(self.metrics, "metrics_manager"):
                    training_data = {
                        'mode': 'Training',
                        'stage': 'PPO Update',
                        'updates': self.global_update_counter,
                        'global_steps': self.global_step_counter,
                        'total_episodes': self.global_episode_counter,
                        'current_epoch': epoch + 1,
                        'total_epochs': self.ppo_epochs,
                        'current_batch': current_batch,
                        'total_batches': total_batches,
                        'stage_status': f"Epoch {epoch + 1}/{self.ppo_epochs}, Batch {current_batch}/{total_batches}"
                    }
                    self.metrics.metrics_manager.emit_event("training_update", training_data)
                # Ensure batch indices don't exceed available samples
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Safely extract batch data with bounds checking
                try:
                    batch_states = {key: tensor_val[batch_indices] for key, tensor_val in states_dict.items()}
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                except IndexError as e:
                    self.logger.error(f"Index error in batch extraction: {e}")
                    self.logger.error(f"Batch indices: {batch_indices}, num_samples: {num_samples}")
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

                if self.model.continuous_action:
                    pass  # Handle continuous actions if needed
                else:
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
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate PPO metrics
                with torch.no_grad():
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()
                    total_clipfrac += clipfrac

                    approx_kl = torch.mean(batch_old_log_probs - new_log_probs).item()
                    total_approx_kl += approx_kl

                    var_y = torch.var(batch_returns)
                    explained_var = 1 - torch.var(batch_returns - current_values.view(-1, 1)) / (var_y + 1e-8)
                    total_explained_variance += explained_var.item()

                current_values_shaped = current_values.view(-1, 1)
                batch_returns_shaped = batch_returns.view(-1, 1)

                if current_values_shaped.size(0) != batch_returns_shaped.size(0):
                    min_size = min(current_values_shaped.size(0), batch_returns_shaped.size(0))
                    current_values_shaped = current_values_shaped[:min_size]
                    batch_returns_shaped = batch_returns_shaped[:min_size]

                critic_loss = nnf.mse_loss(current_values_shaped, batch_returns_shaped)
                entropy_loss = -entropy.mean()
                loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()

                # Track gradient norm
                grad_norm = 0
                if self.max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    total_gradient_norm += float(grad_norm)

                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.mean().item()
                num_updates_in_epoch += 1
            
            # Log epoch completion
            epoch_duration = time.time() - epoch_start_time
            batches_per_second = total_batches / epoch_duration if epoch_duration > 0 else 0
            self.logger.info(f"   ✅ Epoch {epoch + 1}/{self.ppo_epochs} complete | "
                           f"Time: {epoch_duration:.1f}s | "
                           f"Batches/s: {batches_per_second:.1f}")

        self.global_update_counter += 1

        # Calculate averages
        avg_actor_loss = total_actor_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_entropy = total_entropy_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_clipfrac = total_clipfrac / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_approx_kl = total_approx_kl / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_explained_variance = total_explained_variance / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_gradient_norm = total_gradient_norm / num_updates_in_epoch if num_updates_in_epoch > 0 else 0

        # Record metrics
        self.metrics.record_model_losses(avg_actor_loss, avg_critic_loss, avg_entropy)
        self.metrics.record_ppo_metrics(avg_clipfrac, avg_approx_kl, avg_explained_variance)
        self.metrics.record_learning_rate(self.lr)

        # Update dashboard with PPO metrics
        if hasattr(self.metrics, "metrics_manager"):
            mean_reward = np.mean(self.recent_episode_rewards) if len(self.recent_episode_rewards) > 0 else 0
            ppo_data = {
                'lr': self.lr,
                'mean_reward': mean_reward,
                'policy_loss': avg_actor_loss,
                'value_loss': avg_critic_loss,
                'entropy': avg_entropy,
                'total_loss': avg_actor_loss + avg_critic_loss,
                'clip_fraction': avg_clipfrac,
                'approx_kl': avg_approx_kl,
                'explained_variance': avg_explained_variance
            }
            self.metrics.metrics_manager.emit_event("ppo_metrics", ppo_data)

        # End update timing
        self.metrics.end_update()
        update_duration = self._end_timer("update")
        
        # Track update timing
        self.update_times.append(update_duration)
        if len(self.update_times) > 20:  # Keep last 20 updates
            self.update_times.pop(0)

        update_metrics = {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "clipfrac": avg_clipfrac,
            "approx_kl": avg_approx_kl,
            "value_function_explained_variance": avg_explained_variance,
            "gradient_norm": avg_gradient_norm,
            "global_step_counter": self.global_step_counter,
            "global_episode_counter": self.global_episode_counter,
            "global_update_counter": self.global_update_counter
        }

        # Comprehensive update summary with interpretation hints
        self.logger.info(f"🔄 UPDATE COMPLETE:")
        self.logger.info(f"   ⏱️  Duration: {update_duration:.1f}s | Batches: {total_batches}")
        self.logger.info(f"   🎭 Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")
        self.logger.info(f"   📊 Entropy: {avg_entropy:.4f} (↓=converging) | Clip Rate: {avg_clipfrac * 100:.1f}% (target<30%)")
        self.logger.info(f"   🧠 KL Div: {avg_approx_kl:.4f} (<0.01 stable) | Explained Var: {avg_explained_variance * 100:.1f}% (>80% good)")
        self.logger.info(f"   📈 Grad Norm: {avg_gradient_norm:.4f}")
        
        # Add interpretation warnings
        if avg_clipfrac > 0.3:
            self.logger.warning("   ⚠️  High clip rate - consider reducing learning rate")
        if avg_approx_kl > 0.02:
            self.logger.warning("   ⚠️  High KL divergence - updates may be too aggressive")
        if avg_explained_variance < 0.5:
            self.logger.warning("   ⚠️  Low explained variance - value function may need tuning")

        for callback in self.callbacks:
            callback.on_update_end(self, update_metrics)
        
        # Reset dashboard stage progress after update completes
        # Emit training update event
        training_data = {
            'mode': 'Training',
            'stage': 'Preparing Next Rollout',
            'updates': self.global_update_counter,
            'global_steps': self.global_step_counter,
            'total_episodes': self.global_episode_counter,
            'stage_progress': 0.0,  # Reset stage progress
            'stage_status': 'Update completed, preparing next rollout...',
            'time_per_update': np.mean(self.update_times) if self.update_times else 0.0,
            'time_per_episode': np.mean(self.episode_times) if self.episode_times else 0.0
        }
        self.metrics.metrics_manager.emit_event('training_update', training_data)

        return update_metrics

    def train(self, total_training_steps: int, eval_freq_steps: Optional[int] = None):
        """Main training loop with comprehensive stage logging."""
        self.logger.info(f"🚀 TRAINING START: {total_training_steps:,} steps planned")
        self.logger.info(f"   🎯 Rollout size: {self.rollout_steps} | Batch size: {self.batch_size}")
        self.logger.info(f"   🔄 PPO epochs: {self.ppo_epochs} | Learning rate: {self.lr}")

        # Start training metrics
        self.metrics.start_training()
        self.training_start_time = time.time()

        for callback in self.callbacks:
            callback.on_training_start(self)
            
        # Initialize dashboard training state
        # Emit training update event
        training_data = {
            'mode': 'Training',
            'stage': 'Initializing',
            'updates': 0,
            'global_steps': 0,
            'total_episodes': 0,
            'overall_progress': 0.0,
            'stage_progress': 0.0,
            'stage_status': 'Starting training...',
            'steps_per_second': 0.0,
            'time_per_update': 0.0,
            'time_per_episode': 0.0
        }
        self.metrics.metrics_manager.emit_event('training_update', training_data)

        best_eval_reward = -float('inf')

        while self.global_step_counter < total_training_steps:
            rollout_info = self.collect_rollout_data()

            if self.buffer.get_size() < self.rollout_steps and self.buffer.get_size() < self.batch_size:
                self.logger.warning(f"Buffer size {self.buffer.get_size()} too small. Skipping update.")
                if self.buffer.get_size() < self.batch_size:
                    continue

            update_metrics = self.update_policy()

            for callback in self.callbacks:
                callback.on_update_iteration_end(self, self.global_update_counter, update_metrics, rollout_info)

            # Progress logging
            progress = (self.global_step_counter / total_training_steps) * 100
            remaining_steps = total_training_steps - self.global_step_counter
            elapsed_time = time.time() - self.training_start_time
            steps_per_hour = (self.global_step_counter / elapsed_time) * 3600 if elapsed_time > 0 else 0
            eta_hours = remaining_steps / steps_per_hour if steps_per_hour > 0 else 0

            # Always update dashboard with training progress (not just every 5 updates)
            if hasattr(self.metrics, "metrics_manager"):
                training_data = {
                    'mode': 'Training',
                    'stage': 'Active Training',
                    'updates': self.global_update_counter,
                    'global_steps': self.global_step_counter,
                    'total_episodes': self.global_episode_counter,
                    'overall_progress': progress,
                    'stage_progress': progress,
                    'stage_status': f"Update {self.global_update_counter}/{total_training_steps // self.rollout_steps}",
                    'steps_per_second': steps_per_hour / 3600 if steps_per_hour > 0 else 0,
                    'time_per_update': update_metrics.get('update_time', 0) if 'update_metrics' in locals() else 0,
                    'time_per_episode': rollout_info.get('rollout_time', 0) / max(1, rollout_info.get('num_episodes_in_rollout', 1)) if 'rollout_info' in locals() else 0
                }
                self.metrics.metrics_manager.emit_event("training_update", training_data)

            if self.global_update_counter % 5 == 0:  # Log every 5 updates
                # Calculate recent performance trends
                recent_rewards = self.recent_episode_rewards[-10:] if len(self.recent_episode_rewards) > 0 else []
                recent_mean = np.mean(recent_rewards) if recent_rewards else 0
                recent_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0
                
                self.logger.info(f"📈 PROGRESS: {progress:.1f}% | Steps: {self.global_step_counter:,}/{total_training_steps:,}")
                self.logger.info(f"   ⏱️  Rate: {steps_per_hour:.0f} steps/hr | ETA: {eta_hours:.1f}h")
                self.logger.info(f"   🏆 Episodes: {self.global_episode_counter} | Updates: {self.global_update_counter}")
                self.logger.info(f"   📊 Recent Performance: μ={recent_mean:.3f} σ={recent_std:.3f}")
                
            # Periodic training analysis every 25 updates
            if self.global_update_counter % 25 == 0 and self.global_update_counter > 0:
                self._log_training_analysis(update_metrics)

            # Evaluation
            eval_freq_updates = max(1, eval_freq_steps // self.rollout_steps) if eval_freq_steps else 0
            if eval_freq_steps and (self.global_update_counter % eval_freq_updates == 0
                                    or self.global_step_counter >= total_training_steps):

                eval_stats = self.evaluate(n_episodes=10)

                for callback in self.callbacks:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_stats.items() if
                                    k not in ['episode_rewards', 'episode_lengths']}
                    eval_metrics["global_step"] = self.global_step_counter
                    callback.on_update_iteration_end(self, self.global_update_counter, eval_metrics, {})

                if eval_stats['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_stats['mean_reward']
                    best_model_path = os.path.join(self.model_dir, f"best_model_update_{self.global_update_counter}.pt")
                    self.save_model(best_model_path)

                latest_model_path = os.path.join(self.model_dir, "latest_model.pt")
                self.save_model(latest_model_path)

        # Training completion
        total_time = time.time() - self.training_start_time
        final_stats = {
            "total_steps_trained": self.global_step_counter,
            "total_updates": self.global_update_counter,
            "total_episodes": self.global_episode_counter,
            "training_time_hours": total_time / 3600
        }

        self.logger.info(f"🎉 TRAINING COMPLETE!")
        self.logger.info(f"   ⏱️  Total time: {total_time / 3600:.2f} hours")
        self.logger.info(
            f"   📊 Final stats: {self.global_step_counter:,} steps | {self.global_episode_counter} episodes | {self.global_update_counter} updates")

        for callback in self.callbacks:
            callback.on_training_end(self, final_stats)

        return final_stats

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """Evaluation with detailed logging."""
        self.logger.info(f"🔍 EVALUATION START: {n_episodes} episodes")
        self._start_timer("evaluation")

        # Start evaluation metrics
        self.metrics.start_evaluation()

        self.model.eval()
        self.is_evaluating = True

        episode_rewards = []
        episode_lengths = []
        episode_details = []

        for i in range(n_episodes):
            env_state_np, _ = self._reset_environment_with_momentum()
            current_episode_reward = 0.0
            current_episode_length = 0
            done = False

            while not done:
                model_state_torch = convert_state_dict_to_tensors(env_state_np, self.device)
                with torch.no_grad():
                    action_tensor, _ = self.model.get_action(model_state_torch, deterministic=deterministic)

                env_action = self._convert_action_for_env(action_tensor)

                try:
                    next_env_state_np, reward, terminated, truncated, info = self.env.step(env_action)
                    done = terminated or truncated
                except Exception as e:
                    self.logger.error(f"Error during evaluation step: {e}")
                    done = True
                    reward = 0

                env_state_np = next_env_state_np
                current_episode_reward += reward
                current_episode_length += 1

            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            episode_details.append({
                'reward': current_episode_reward,
                'length': current_episode_length,
                'final_equity': info.get('portfolio_equity', 0)
            })

        # End evaluation metrics
        self.metrics.end_evaluation(episode_rewards, episode_lengths)

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
            "episode_lengths": episode_lengths
        }

        # Comprehensive evaluation summary
        self.logger.info(f"🔍 EVALUATION COMPLETE:")
        self.logger.info(f"   ⏱️  Duration: {eval_duration:.1f}s")
        self.logger.info(f"   💰 Rewards: μ={eval_results['mean_reward']:.3f} σ={eval_results['std_reward']:.3f}")
        self.logger.info(f"   📊 Range: [{eval_results['min_reward']:.3f}, {eval_results['max_reward']:.3f}]")
        self.logger.info(f"   📏 Avg Length: {eval_results['mean_length']:.1f} steps")

        return eval_results

    def save_model(self, path: str) -> None:
        """Saves the model and optimizer state."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step_counter': self.global_step_counter,
                'global_episode_counter': self.global_episode_counter,
                'global_update_counter': self.global_update_counter,
                'model_config': self.model_config
            }, path)
        except Exception as e:
            self.logger.error(f"Error saving model to {path}: {e}")

    def load_model(self, path: str) -> None:
        """Loads the model and optimizer state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.global_step_counter = checkpoint.get('global_step_counter', 0)
            self.global_episode_counter = checkpoint.get('global_episode_counter', 0)
            self.global_update_counter = checkpoint.get('global_update_counter', 0)

            self.model.to(self.device)
            self.logger.info(f"Model loaded from {path}. Resuming from step {self.global_step_counter}")
        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {e}")
    
    def _log_training_analysis(self, update_metrics: Dict[str, float]) -> None:
        """Log comprehensive training analysis for interpretation."""
        self.logger.info("=" * 80)
        self.logger.info("🔬 TRAINING ANALYSIS - Update {}".format(self.global_update_counter))
        self.logger.info("=" * 80)
        
        # Performance trends
        recent_rewards = self.recent_episode_rewards[-30:] if len(self.recent_episode_rewards) > 0 else []
        if len(recent_rewards) >= 10:
            first_10 = np.mean(recent_rewards[:10])
            last_10 = np.mean(recent_rewards[-10:])
            trend = last_10 - first_10
            trend_pct = (trend / abs(first_10) * 100) if first_10 != 0 else 0
            
            self.logger.info(f"📈 PERFORMANCE TREND:")
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
                self.logger.warning("      - Increasing exploration (entropy coefficient)")
                self.logger.warning("      - Adjusting reward weights")
                self.logger.warning("      - Checking for data diversity")
        
        # Learning stability
        self.logger.info(f"\n🧠 LEARNING STABILITY:")
        kl = update_metrics.get('approx_kl', 0)
        clipfrac = update_metrics.get('clipfrac', 0)
        entropy = update_metrics.get('entropy', 0)
        
        stability_score = "STABLE"
        if kl > 0.02 or clipfrac > 0.3:
            stability_score = "UNSTABLE"
        elif kl > 0.01 or clipfrac > 0.2:
            stability_score = "BORDERLINE"
            
        self.logger.info(f"   Status: {stability_score}")
        self.logger.info(f"   KL Divergence: {kl:.4f}")
        self.logger.info(f"   Clip Fraction: {clipfrac*100:.1f}%")
        self.logger.info(f"   Entropy: {entropy:.4f}")
        
        # Value function quality
        explained_var = update_metrics.get('value_function_explained_variance', 0)
        critic_loss = update_metrics.get('critic_loss', 0)
        
        self.logger.info(f"\n📊 VALUE FUNCTION:")
        self.logger.info(f"   Explained Variance: {explained_var*100:.1f}%")
        self.logger.info(f"   Critic Loss: {critic_loss:.4f}")
        
        if explained_var < 0.7:
            self.logger.warning("   ⚠️  Poor value estimation - actions:")
            self.logger.warning("      - Increase critic coefficient")
            self.logger.warning("      - Check feature quality")
            self.logger.warning("      - Verify reward normalization")
        
        # Action recommendations
        self.logger.info(f"\n💡 RECOMMENDATIONS:")
        
        if stability_score == "UNSTABLE":
            self.logger.info("   1. Reduce learning rate by 50%")
            self.logger.info("   2. Decrease PPO clip range")
            self.logger.info("   3. Increase batch size")
        elif stability_score == "BORDERLINE":
            self.logger.info("   1. Monitor next few updates closely")
            self.logger.info("   2. Consider small learning rate reduction")
        
        if entropy < 0.01:
            self.logger.info("   - Increase entropy coefficient to encourage exploration")
        elif entropy > 0.1:
            self.logger.info("   - Decrease entropy coefficient to focus learning")
            
        self.logger.info("=" * 80)