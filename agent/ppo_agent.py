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
from agent.replay_buffer import ReplayBuffer, convert_state_dict_to_tensors
from agent.base_callbacks import V1TrainingCallback
from agent.callbacks import CallbackManager


class V1PPOTrainer:


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
            
            # Notify DataLifecycleManager of episode completion - this advances to next reset point
            # and handles cycle completion tracking internally
            if self.training_manager and hasattr(self.training_manager, 'data_lifecycle_manager'):
                if self.training_manager.data_lifecycle_manager:
                    try:
                        self.training_manager.data_lifecycle_manager.advance_cycle_on_episode_completion()
                        self.logger.debug("🔄 Notified DataLifecycleManager of episode completion")
                    except Exception as e:
                        self.logger.debug(f"DataLifecycleManager episode notification failed: {e}")

            # Check if we completed a cycle through all reset points (for PPO agent's own tracking)
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

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
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