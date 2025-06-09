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

    def _convert_action_for_env(self, action_tensor: torch.Tensor) -> Any:
        """Converts model's action tensor to environment-compatible format."""
        # System only uses discrete actions
        if action_tensor.ndim > 0 and action_tensor.shape[-1] == 2:
            return action_tensor.cpu().numpy().squeeze().astype(int)
        else:
            return action_tensor.cpu().numpy().item()

    def collect_rollout_data(self) -> Dict[str, Any]:
        """Collect rollout data for PPO training.
        
        Collects fixed number of steps across potentially multiple episodes.
        Episodes are automatically reset when they complete.
        """
        self.buffer.clear()
        current_env_state_np, _ = self._reset_environment_with_momentum()
        
        collected_steps = 0
        current_episode_reward = 0.0
        current_episode_length = 0

        while collected_steps < self.rollout_steps:
            # Convert state to tensors
            single_step_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in current_env_state_np.items()
            }

            # Batch state for model
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

            # Get action from model
            with torch.no_grad():
                action_tensor, action_info = self.model.get_action(
                    current_model_state_torch_batched, deterministic=False
                )

            env_action = self._convert_action_for_env(action_tensor)

            # Step environment
            try:
                next_env_state_np, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated

                # Check for training interruption
                if self.stop_training or (hasattr(__import__("main"), "training_interrupted") and __import__("main").training_interrupted):
                    self.stop_training = True
                    break

            except Exception as e:
                self.logger.error(f"Error during environment step: {e}")
                break

            # Store experience in buffer
            self.buffer.add(
                current_env_state_np,
                action_tensor,
                reward,
                next_env_state_np,
                done,
                action_info,
            )

            # Update state and counters
            current_env_state_np = next_env_state_np
            collected_steps += 1
            current_episode_reward += reward
            current_episode_length += 1
            self.global_step_counter += 1

            # Handle episode completion
            if done:
                self.global_episode_counter += 1
                current_env_state_np, _ = self._reset_environment_with_momentum()
                current_episode_reward = 0.0
                current_episode_length = 0

        # Prepare buffer for training
        self.buffer.prepare_data_for_training()
        
        # Return basic rollout statistics
        return {
            "collected_steps": collected_steps,
            "global_step_counter": self.global_step_counter,
            "global_episode_counter": self.global_episode_counter,
        }

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
        """PPO policy update."""
        # Check for interruption
        if self.stop_training or (
            hasattr(__import__("main"), "training_interrupted")
            and __import__("main").training_interrupted
        ):
            return {"interrupted": True}

        # Compute advantages and returns
        self._compute_advantages_and_returns()

        training_data = self.buffer.get_training_data()
        if training_data is None:
            return {}

        states_dict = training_data["states"]
        actions = training_data["actions"]
        old_log_probs = training_data["old_log_probs"]
        advantages = training_data["advantages"]
        returns = training_data["returns"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = actions.size(0)
        if num_samples == 0:
            return {}

        indices = np.arange(num_samples)
        total_actor_loss, total_critic_loss = 0, 0
        num_updates_in_epoch = 0

        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Check for interruption
            if self.stop_training or (
                hasattr(__import__("main"), "training_interrupted")
                and __import__("main").training_interrupted
            ):
                avg_actor_loss = total_actor_loss / max(1, num_updates_in_epoch)
                avg_critic_loss = total_critic_loss / max(1, num_updates_in_epoch)
                return {
                    "policy_loss": avg_actor_loss,
                    "value_loss": avg_critic_loss,
                    "interrupted": True
                }

            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Extract batch data
                try:
                    batch_states = {
                        key: tensor_val[batch_indices]
                        for key, tensor_val in states_dict.items()
                    }
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                except IndexError:
                    continue

                # Forward pass
                action_params, current_values = self.model(batch_states)

                # Shape tensors correctly
                if batch_returns.ndim > 1 and batch_returns.shape[1] > 1:
                    batch_returns = batch_returns[:, 0:1]
                elif batch_returns.ndim == 1:
                    batch_returns = batch_returns.unsqueeze(1)

                if batch_advantages.ndim > 1 and batch_advantages.shape[1] > 1:
                    batch_advantages = batch_advantages[:, 0:1]
                elif batch_advantages.ndim == 1:
                    batch_advantages = batch_advantages.unsqueeze(1)

                # Discrete action distributions
                action_type_logits, action_size_logits = action_params
                action_types_taken = batch_actions[:, 0].long()
                action_sizes_taken = batch_actions[:, 1].long()

                type_dist = torch.distributions.Categorical(logits=action_type_logits)
                size_dist = torch.distributions.Categorical(logits=action_size_logits)

                new_type_log_probs = type_dist.log_prob(action_types_taken)
                new_size_log_probs = size_dist.log_prob(action_sizes_taken)
                new_log_probs = (new_type_log_probs + new_size_log_probs).unsqueeze(1)

                entropy = (type_dist.entropy() + size_dist.entropy()).unsqueeze(1)

                # PPO loss calculation
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                current_values_shaped = current_values.view(-1, 1)
                batch_returns_shaped = batch_returns.view(-1, 1)
                if current_values_shaped.size(0) != batch_returns_shaped.size(0):
                    min_size = min(current_values_shaped.size(0), batch_returns_shaped.size(0))
                    current_values_shaped = current_values_shaped[:min_size]
                    batch_returns_shaped = batch_returns_shaped[:min_size]

                critic_loss = nnf.mse_loss(current_values_shaped, batch_returns_shaped)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_updates_in_epoch += 1

        # Update counter
        self.global_update_counter += 1
        
        # Calculate averages
        avg_actor_loss = total_actor_loss / max(1, num_updates_in_epoch)
        avg_critic_loss = total_critic_loss / max(1, num_updates_in_epoch)
        
        # Return basic metrics
        return {
            "policy_loss": avg_actor_loss,
            "value_loss": avg_critic_loss,
            "global_update_counter": self.global_update_counter,
        }

    def train_with_manager(self) -> Dict[str, Any]:
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

        self.training_manager = training_manager

        # Start training with manager (it will control the lifecycle)
        final_stats = training_manager.start_training(self)

        return final_stats



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