# agent/ppo_agent.py - UPDATED: Full integration with comprehensive dashboard

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Any, Optional
import wandb
import torch.nn.functional as nnf
import time

from config.config import ModelConfig
from envs.env_dashboard import TrainingStage  # Updated import
from envs.trading_env import TradingEnvironment
from ai.transformer import MultiBranchTransformer
from agent.utils import ReplayBuffer, convert_state_dict_to_tensors
from agent.callbacks import TrainingCallback


class PPOTrainer:
    def __init__(
            self,
            env: TradingEnvironment,
            model: MultiBranchTransformer,
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
            use_wandb: bool = True,
            callbacks: Optional[List[TrainingCallback]] = None,
            dashboard: Optional[Any] = None,
    ):
        self.env = env
        self.model = model
        self.model_config = model_config if model_config else {}
        self.dashboard = dashboard

        # Use standard logging with Rich formatting
        self.logger = logging.getLogger(f"{__name__}.PPOTrainer")

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

        # Logging and W&B
        self.use_wandb = use_wandb
        if self.use_wandb and wandb.run is None:
            logging.warning("W&B not initialized. Consider initializing wandb.init() before trainer.")

        # Callbacks
        self.callbacks = callbacks if callbacks else []

        # Training state
        self.global_step_counter = 0
        self.global_episode_counter = 0
        self.global_update_counter = 0

        # UPDATED: Comprehensive dashboard integration
        self.is_evaluating = False
        self.training_start_time = 0.0

        # Enhanced dashboard update tracking
        self.last_update_start_time = 0.0
        self.last_rollout_start_time = 0.0
        self.last_dashboard_sync = 0.0
        self.dashboard_sync_interval = 0.3  # More frequent updates for comprehensive dashboard

        # Performance tracking for comprehensive dashboard
        self.steps_collected_current_rollout = 0
        self.evaluation_episode_count = 0
        self.evaluation_total_episodes = 0
        self.recent_episode_times = []
        self.recent_update_times = []

        # UPDATED: Advanced training metrics tracking for comprehensive dashboard
        self.training_metrics_history = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'clipfrac': 0.0,
            'approx_kl': 0.0,
            'value_explained_variance': 0.0,
            'gradient_norm': 0.0,
            'policy_kl_divergence': 0.0,
            'total_loss': 0.0,
            'mean_episode_reward': 0.0,
            'mean_episode_length': 0.0,
            'steps_per_second': 0.0,
            'time_per_update': 0.0,
            'episodes_per_update': 0
        }

        logging.info(
            f"🤖 PPOTrainer initialized with comprehensive dashboard integration. "
            f"Device: {self.device}, LR: {self.lr}, Rollout Steps: {self.rollout_steps}")

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

    def _update_comprehensive_dashboard_training_info(self, is_training: bool = True, is_evaluating: bool = False):
        """Update comprehensive dashboard with current training information"""
        if hasattr(self.env, 'set_training_info'):
            self.env.set_training_info(
                episode_num=self.global_episode_counter,
                total_episodes=0,
                total_steps=self.global_step_counter,
                update_count=self.global_update_counter,
                buffer_size=self.rollout_steps,
                is_training=is_training,
                is_evaluating=is_evaluating,
                learning_rate=self.lr
            )

    def _sync_comprehensive_dashboard_realtime(self, collected_steps: int, episode_rewards: List[float],
                                               force_update: bool = False):
        """UPDATED: Enhanced real-time dashboard sync with comprehensive metrics"""
        current_time = time.time()

        # Update global step counter immediately for real-time display
        actual_global_steps = self.global_step_counter + collected_steps

        should_update = (
                force_update or
                current_time - self.last_dashboard_sync >= self.dashboard_sync_interval or
                collected_steps % 3 == 0  # Update every 3 steps for smooth progress
        )

        if should_update:
            # Update dashboard with current progress
            if self.dashboard:
                rollout_progress = collected_steps / self.rollout_steps
                self.dashboard.set_training_stage(
                    TrainingStage.COLLECTING_ROLLOUT,
                    None,
                    f"Collecting step {collected_steps}/{self.rollout_steps}",
                    rollout_progress
                )

                # Calculate comprehensive performance metrics
                rollout_time = current_time - self.last_rollout_start_time
                steps_per_second = collected_steps / rollout_time if rollout_time > 0 else 0
                mean_reward = np.mean(episode_rewards) if episode_rewards else 0

                # UPDATED: Comprehensive metrics update for advanced dashboard
                comprehensive_metrics = {
                    # Basic metrics
                    'collected_steps': collected_steps,
                    'rollout_steps': self.rollout_steps,
                    'mean_episode_reward': mean_reward,
                    'steps_per_second': steps_per_second,
                    'global_step_counter': actual_global_steps,
                    'total_steps': actual_global_steps,
                    'episode_number': self.global_episode_counter,
                    'step': actual_global_steps,
                    'update_count': self.global_update_counter,

                    # Training performance metrics
                    'episodes_per_update': len(episode_rewards),
                    'time_per_update': rollout_time,
                    'batch_size': self.batch_size,
                    'buffer_size': self.rollout_steps,
                    'learning_rate': self.lr,

                    # Advanced training metrics (from history)
                    **self.training_metrics_history
                }

                # Add reward components from environment's reward calculator if available
                if hasattr(self.env, 'reward_calculator') and hasattr(self.env.reward_calculator,
                                                                      'get_last_reward_components'):
                    try:
                        reward_components = self.env.reward_calculator.get_last_reward_components()
                        if reward_components:
                            comprehensive_metrics['reward_components'] = reward_components
                    except Exception as e:
                        self.logger.debug(f"Could not get reward components: {e}")

                self.dashboard.update_training_metrics(comprehensive_metrics)

            # Force update environment's training info with real-time steps
            if hasattr(self.env, 'set_training_info'):
                self.env.set_training_info(
                    episode_num=self.global_episode_counter,
                    total_episodes=0,
                    total_steps=actual_global_steps,
                    update_count=self.global_update_counter,
                    buffer_size=self.rollout_steps,
                    is_training=not self.is_evaluating,
                    is_evaluating=self.is_evaluating,
                    learning_rate=self.lr
                )

            self.last_dashboard_sync = current_time

    def collect_rollout_data(self) -> Dict[str, Any]:
        """UPDATED: Enhanced rollout collection with comprehensive dashboard integration."""
        logging.info(f"🎲 Starting rollout data collection for {self.rollout_steps} steps...")
        self.buffer.clear()

        # Record start time for performance tracking
        self.last_rollout_start_time = time.time()
        self.last_dashboard_sync = 0.0
        self.steps_collected_current_rollout = 0

        # Update comprehensive dashboard that we're collecting rollouts
        self._update_comprehensive_dashboard_training_info(is_training=True, is_evaluating=False)

        current_env_state_np, _ = self.env.reset()

        for callback in self.callbacks:
            callback.on_rollout_start(self)

        collected_steps = 0
        episode_rewards_in_rollout = []
        episode_lengths_in_rollout = []
        current_episode_reward = 0.0
        current_episode_length = 0
        episode_start_time = time.time()

        while collected_steps < self.rollout_steps:
            single_step_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in current_env_state_np.items()
            }

            # UPDATED: More frequent real-time dashboard updates during rollout
            if collected_steps % 2 == 0 or collected_steps == self.rollout_steps - 1:  # Update every 2 steps
                self._sync_comprehensive_dashboard_realtime(collected_steps, episode_rewards_in_rollout)

            current_model_state_torch_batched = {}
            for key, tensor_val in single_step_tensors.items():
                if key in ['hf', 'mf', 'lf', 'portfolio']:
                    if tensor_val.ndim == 2:
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)
                    elif tensor_val.ndim == 3 and tensor_val.shape[0] == 1:
                        current_model_state_torch_batched[key] = tensor_val
                    else:
                        logging.error(
                            f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}'. Shape: {tensor_val.shape}")
                        current_model_state_torch_batched[key] = tensor_val
                elif key == 'static':
                    if tensor_val.ndim == 2 and tensor_val.shape[0] == 1:
                        current_model_state_torch_batched[key] = tensor_val
                    elif tensor_val.ndim == 1:
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)
                    else:
                        logging.error(
                            f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}' (static). Shape: {tensor_val.shape}")
                        current_model_state_torch_batched[key] = tensor_val
                else:
                    current_model_state_torch_batched[key] = tensor_val

            with torch.no_grad():
                action_tensor, action_info = self.model.get_action(current_model_state_torch_batched,
                                                                   deterministic=False)

            env_action = self._convert_action_for_env(action_tensor)

            try:
                next_env_state_np, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
            except Exception as e:
                logging.error(f"Error during environment step: {e}")
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

            # Update global step counter immediately for real-time tracking
            self.global_step_counter += 1

            for callback in self.callbacks:
                callback.on_step(self, current_model_state_torch_batched, action_tensor, reward, next_env_state_np,
                                 info)

            if done:
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                self.recent_episode_times.append(episode_duration)
                if len(self.recent_episode_times) > 10:
                    self.recent_episode_times.pop(0)

                self.global_episode_counter += 1
                episode_rewards_in_rollout.append(current_episode_reward)
                episode_lengths_in_rollout.append(current_episode_length)

                logging.info(f"🏁 Episode {self.global_episode_counter} finished. "
                             f"Reward: {current_episode_reward:.2f}, "
                             f"Length: {current_episode_length}, "
                             f"Duration: {episode_duration:.1f}s, "
                             f"Global Steps: {self.global_step_counter}")

                for callback in self.callbacks:
                    callback.on_episode_end(self, current_episode_reward, current_episode_length, info)

                current_env_state_np, _ = self.env.reset()
                current_episode_reward = 0.0
                current_episode_length = 0
                episode_start_time = time.time()

                # Update comprehensive dashboard after episode completion
                self._sync_comprehensive_dashboard_realtime(collected_steps, episode_rewards_in_rollout, force_update=True)

                if collected_steps >= self.rollout_steps:
                    break

        # Final comprehensive dashboard update after rollout completion
        self._sync_comprehensive_dashboard_realtime(collected_steps, episode_rewards_in_rollout, force_update=True)

        for callback in self.callbacks:
            callback.on_rollout_end(self)

        self.buffer.prepare_data_for_training()

        # Calculate comprehensive performance metrics
        rollout_time = time.time() - self.last_rollout_start_time
        steps_per_second = collected_steps / rollout_time if rollout_time > 0 else 0
        mean_episode_reward = np.mean(episode_rewards_in_rollout) if episode_rewards_in_rollout else 0
        mean_episode_length = np.mean(episode_lengths_in_rollout) if episode_lengths_in_rollout else 0

        # Update training metrics history
        self.training_metrics_history.update({
            'mean_episode_reward': mean_episode_reward,
            'mean_episode_length': mean_episode_length,
            'steps_per_second': steps_per_second,
            'time_per_update': rollout_time,
            'episodes_per_update': len(episode_rewards_in_rollout)
        })

        rollout_stats = {
            "collected_steps": collected_steps,
            "mean_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "num_episodes_in_rollout": len(episode_rewards_in_rollout),
            "rollout_time": rollout_time,
            "steps_per_second": steps_per_second,
            "global_step_counter": self.global_step_counter,
            "global_episode_counter": self.global_episode_counter
        }

        # Final comprehensive dashboard update
        if self.dashboard:
            final_metrics = {
                'collected_steps': collected_steps,
                'mean_episode_reward': rollout_stats["mean_reward"],
                'mean_episode_length': rollout_stats["mean_episode_length"],
                'episodes_per_update': rollout_stats["num_episodes_in_rollout"],
                'steps_per_second': steps_per_second,
                'global_step_counter': self.global_step_counter,
                'total_steps': self.global_step_counter,
                'episode_number': self.global_episode_counter,
                'time_per_update': rollout_time
            }
            self.dashboard.update_training_metrics(final_metrics)

        logging.info(
            f"📊 Rollout finished. Global Steps: {self.global_step_counter}, Episodes: {self.global_episode_counter}")
        return rollout_stats

    def _compute_advantages_and_returns(self):
        """Computes GAE advantages and returns, storing them in the buffer."""
        if self.buffer.rewards is None or self.buffer.values is None or self.buffer.dones is None:
            logging.error("Cannot compute advantages: buffer data not prepared.")
            return

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        num_steps = len(rewards)

        logging.debug(
            f"Computing advantages with shapes - rewards: {rewards.shape}, values: {values.shape}, dones: {dones.shape}")

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
            logging.warning(f"Values has multiple columns: {values.shape}. Taking first column.")
            values = values[:, 0:1]

        returns = advantages + values

        if returns.ndim > 1 and returns.shape[1] > 1:
            logging.warning(f"Returns has unexpected shape: {returns.shape}. Taking first column.")
            returns = returns[:, 0:1]

        self.buffer.returns = returns
        logging.debug(
            f"Final shapes - advantages: {self.buffer.advantages.shape}, returns: {self.buffer.returns.shape}")

    def update_policy(self) -> Dict[str, float]:
        """UPDATED: Enhanced PPO updates with comprehensive metrics tracking for dashboard."""
        # Record start time for performance tracking
        self.last_update_start_time = time.time()

        # Update comprehensive dashboard that we're updating policy
        self._update_comprehensive_dashboard_training_info(is_training=True, is_evaluating=False)

        self._compute_advantages_and_returns()

        training_data = self.buffer.get_training_data()
        if training_data is None:
            logging.error("Skipping policy update due to missing training data in buffer.")
            return {}

        states_dict = training_data["states"]
        actions = training_data["actions"]
        old_log_probs = training_data["old_log_probs"]
        advantages = training_data["advantages"]
        returns = training_data["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = actions.size(0)
        if num_samples == 0:
            logging.warning("No samples in buffer to update policy. Skipping update.")
            return {}

        indices = np.arange(num_samples)

        for callback in self.callbacks:
            callback.on_update_start(self)

        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        num_updates_in_epoch = 0
        total_batches = (num_samples + self.batch_size - 1) // self.batch_size
        total_updates = self.ppo_epochs * total_batches

        # UPDATED: Advanced metrics tracking for comprehensive dashboard
        total_clipfrac = 0
        total_approx_kl = 0
        total_explained_variance = 0
        total_gradient_norm = 0

        logging.info(f"🔄 Starting PPO update for {self.ppo_epochs} epochs with {num_samples} samples")

        update_idx = 0
        for epoch in range(self.ppo_epochs):
            # Update comprehensive dashboard with epoch progress
            if self.dashboard:
                epoch_progress = epoch / self.ppo_epochs
                self.dashboard.set_training_stage(
                    TrainingStage.UPDATING_POLICY,
                    None,
                    f"Policy update epoch {epoch + 1}/{self.ppo_epochs}",
                    epoch_progress
                )

            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx: start_idx + self.batch_size]

                # Real-time batch progress updates for comprehensive dashboard
                if self.dashboard and update_idx % 2 == 0:  # Update every 2 batches
                    batch_progress = update_idx / total_updates
                    batch_num = (start_idx // self.batch_size) + 1
                    epoch_batches = total_batches
                    self.dashboard.set_training_stage(
                        TrainingStage.UPDATING_POLICY,
                        None,
                        f"Epoch {epoch + 1}/{self.ppo_epochs}, Batch {batch_num}/{epoch_batches}",
                        batch_progress
                    )

                batch_states = {key: tensor_val[batch_indices] for key, tensor_val in states_dict.items()}
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

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

                # UPDATED: Calculate advanced metrics for comprehensive dashboard
                with torch.no_grad():
                    # Clip fraction
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()
                    total_clipfrac += clipfrac

                    # Approximate KL divergence
                    approx_kl = torch.mean(batch_old_log_probs - new_log_probs).item()
                    total_approx_kl += approx_kl

                    # Value function explained variance
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

                # Track gradient norm for comprehensive dashboard
                grad_norm = 0
                if self.max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    total_gradient_norm += float(grad_norm)

                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.mean().item()
                num_updates_in_epoch += 1
                update_idx += 1

        self.global_update_counter += 1

        # Calculate comprehensive performance metrics
        update_time = time.time() - self.last_update_start_time
        self.recent_update_times.append(update_time)
        if len(self.recent_update_times) > 10:
            self.recent_update_times.pop(0)

        avg_actor_loss = total_actor_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_entropy = total_entropy_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0

        # UPDATED: Calculate advanced metrics averages for comprehensive dashboard
        avg_clipfrac = total_clipfrac / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_approx_kl = total_approx_kl / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_explained_variance = total_explained_variance / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_gradient_norm = total_gradient_norm / num_updates_in_epoch if num_updates_in_epoch > 0 else 0

        # Calculate total loss
        total_loss_value = avg_actor_loss + self.critic_coef * avg_critic_loss + self.entropy_coef * (-avg_entropy)

        # UPDATED: Store comprehensive advanced metrics for dashboard
        self.training_metrics_history.update({
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'total_loss': total_loss_value,
            'clipfrac': avg_clipfrac,
            'approx_kl': avg_approx_kl,
            'value_explained_variance': avg_explained_variance,
            'gradient_norm': avg_gradient_norm,
            'policy_kl_divergence': avg_approx_kl,
            'time_per_update': update_time
        })

        update_metrics = {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "total_loss": total_loss_value,
            "time_per_update": update_time,
            "global_step_counter": self.global_step_counter,
            "global_episode_counter": self.global_episode_counter,
            "global_update_counter": self.global_update_counter,
            # Advanced metrics for comprehensive dashboard
            "clipfrac": avg_clipfrac,
            "approx_kl": avg_approx_kl,
            "value_function_explained_variance": avg_explained_variance,
            "gradient_norm": avg_gradient_norm
        }

        # UPDATED: Comprehensive dashboard update with all metrics
        if self.dashboard:
            comprehensive_update = {
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'entropy': avg_entropy,
                'total_loss': total_loss_value,
                'time_per_update': update_time,
                'update_count': self.global_update_counter,
                'global_step_counter': self.global_step_counter,
                'total_steps': self.global_step_counter,
                'episode_number': self.global_episode_counter,
                # Advanced metrics
                'clipfrac': avg_clipfrac,
                'approx_kl': avg_approx_kl,
                'value_function_explained_variance': avg_explained_variance,
                'policy_kl_divergence': avg_approx_kl,
                'gradient_norm': avg_gradient_norm,
                # Performance metrics
                'batch_size': self.batch_size,
                'buffer_size': self.rollout_steps,
                'learning_rate': self.lr
            }
            self.dashboard.update_training_metrics(comprehensive_update)

        for callback in self.callbacks:
            callback.on_update_end(self, update_metrics)

        logging.info(f"📈 PPO Update {self.global_update_counter}: "
                     f"Actor: {avg_actor_loss:.4f}, Critic: {avg_critic_loss:.4f}, "
                     f"Entropy: {avg_entropy:.4f}, ClipFrac: {avg_clipfrac:.3f}, "
                     f"Time: {update_time:.1f}s, Global Steps: {self.global_step_counter}")

        return update_metrics

    def train(self, total_training_steps: int, eval_freq_steps: Optional[int] = None):
        """Main training loop with comprehensive dashboard integration."""
        logging.info(f"🚀 Starting PPO training for a total of {total_training_steps} environment steps.")

        # Record training start time
        self.training_start_time = time.time()

        for callback in self.callbacks:
            callback.on_training_start(self)

        best_eval_reward = -float('inf')

        while self.global_step_counter < total_training_steps:
            logging.info(f"🔄 Update {self.global_update_counter + 1} - Collecting rollout data...")
            rollout_info = self.collect_rollout_data()

            if self.buffer.get_size() < self.rollout_steps and self.buffer.get_size() < self.batch_size:
                logging.warning(f"Buffer size {self.buffer.get_size()} too small. Skipping update.")
                if self.buffer.get_size() < self.batch_size:
                    continue

            logging.info(f"🧠 Updating policy with {self.buffer.get_size()} samples...")
            update_metrics = self.update_policy()

            for callback in self.callbacks:
                callback.on_update_iteration_end(self, self.global_update_counter, update_metrics, rollout_info)

            # Evaluation with comprehensive dashboard update
            if eval_freq_steps and (self.global_update_counter % (
                    eval_freq_steps // self.rollout_steps) == 0 or self.global_step_counter >= total_training_steps):
                # Update comprehensive dashboard that we're evaluating
                self._update_comprehensive_dashboard_training_info(is_training=False, is_evaluating=True)

                logging.info(f"🔍 Running evaluation at step {self.global_step_counter}...")
                eval_stats = self.evaluate(n_episodes=10)
                logging.info(f"📊 Evaluation: Mean Reward: {eval_stats['mean_reward']:.2f}")

                # Reset comprehensive dashboard back to training mode
                self._update_comprehensive_dashboard_training_info(is_training=True, is_evaluating=False)

                for callback in self.callbacks:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_stats.items() if
                                    k not in ['episode_rewards', 'episode_lengths']}
                    eval_metrics["global_step"] = self.global_step_counter
                    callback.on_update_iteration_end(self, self.global_update_counter, eval_metrics, {})

                if eval_stats['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_stats['mean_reward']
                    best_model_path = os.path.join(self.model_dir, f"best_model_update_{self.global_update_counter}.pt")
                    self.save_model(best_model_path)
                    logging.info(f"🏆 New best model saved: {best_eval_reward:.2f} at {best_model_path}")

                latest_model_path = os.path.join(self.model_dir, "latest_model.pt")
                self.save_model(latest_model_path)

        final_stats = {"total_steps_trained": self.global_step_counter, "total_updates": self.global_update_counter}

        for callback in self.callbacks:
            callback.on_training_end(self, final_stats)

        logging.info(
            f"🎉 Training finished! Total steps: {self.global_step_counter}, Updates: {self.global_update_counter}")
        return final_stats

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """UPDATED: Enhanced evaluation with comprehensive dashboard updates."""
        logging.info(f"🔍 Starting evaluation for {n_episodes} episodes (deterministic: {deterministic})...")

        # Set model to evaluation mode
        self.model.eval()
        self.is_evaluating = True
        self.evaluation_episode_count = 0
        self.evaluation_total_episodes = n_episodes

        # Update comprehensive dashboard
        self._update_comprehensive_dashboard_training_info(is_training=False, is_evaluating=True)

        episode_rewards = []
        episode_lengths = []

        for i in range(n_episodes):
            # Update comprehensive dashboard with evaluation progress
            if self.dashboard:
                eval_progress = i / n_episodes
                self.dashboard.set_training_stage(
                    TrainingStage.EVALUATING,
                    None,
                    f"Evaluation episode {i + 1}/{n_episodes}",
                    eval_progress
                )

                # Update evaluation metrics
                eval_metrics_update = {
                    'global_step_counter': self.global_step_counter,
                    'total_steps': self.global_step_counter,
                    'episode_number': self.global_episode_counter,
                    'evaluation_episode': i + 1,
                    'evaluation_total': n_episodes,
                    'is_evaluating': True,
                    'is_training': False
                }
                self.dashboard.update_training_metrics(eval_metrics_update)

            env_state_np, _ = self.env.reset()
            current_episode_reward = 0.0
            current_episode_length = 0
            done = False
            step_count = 0

            while not done:
                model_state_torch = convert_state_dict_to_tensors(env_state_np, self.device)
                with torch.no_grad():
                    action_tensor, _ = self.model.get_action(model_state_torch, deterministic=deterministic)

                env_action = self._convert_action_for_env(action_tensor)

                try:
                    next_env_state_np, reward, terminated, truncated, info = self.env.step(env_action)
                    done = terminated or truncated
                except Exception as e:
                    logging.error(f"Error during evaluation step: {e}")
                    done = True
                    reward = 0

                env_state_np = next_env_state_np
                current_episode_reward += reward
                current_episode_length += 1
                step_count += 1

                # Real-time evaluation progress updates
                if step_count % 20 == 0 and self.dashboard:
                    eval_step_update = {
                        'evaluation_episode_reward': current_episode_reward,
                        'evaluation_episode_length': current_episode_length,
                        'evaluation_step': step_count
                    }
                    self.dashboard.update_training_metrics(eval_step_update)

            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            self.evaluation_episode_count += 1

            logging.info(
                f"🔍 Eval Episode {i + 1}/{n_episodes}: Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")

            # Update comprehensive dashboard after each evaluation episode
            if self.dashboard:
                mean_reward_so_far = np.mean(episode_rewards) if episode_rewards else 0
                eval_episode_update = {
                    'evaluation_mean_reward': mean_reward_so_far,
                    'evaluation_episodes_completed': self.evaluation_episode_count
                }
                self.dashboard.update_training_metrics(eval_episode_update)

        # Reset model to training mode
        self.model.train()
        self.is_evaluating = False

        eval_results = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }

        # Final evaluation comprehensive dashboard update
        if self.dashboard:
            final_eval_update = {
                'evaluation_final_mean_reward': eval_results["mean_reward"],
                'evaluation_final_std_reward': eval_results["std_reward"],
                'evaluation_completed': True,
                'is_evaluating': False,
                'is_training': True  # Back to training mode
            }
            self.dashboard.update_training_metrics(final_eval_update)

        logging.info(
            f"📊 Evaluation complete! Mean: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
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
                'model_config': self.model_config,
                'training_metrics_history': self.training_metrics_history
            }, path)
            logging.info(f"💾 Model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving model to {path}: {e}")

    def load_model(self, path: str) -> None:
        """Loads the model and optimizer state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.global_step_counter = checkpoint.get('global_step_counter', 0)
            self.global_episode_counter = checkpoint.get('global_episode_counter', 0)
            self.global_update_counter = checkpoint.get('global_update_counter', 0)

            # Load training metrics history if available
            if 'training_metrics_history' in checkpoint:
                self.training_metrics_history.update(checkpoint['training_metrics_history'])

            self.model.to(self.device)
            logging.info(f"📂 Model loaded from {path}. "
                         f"Resuming from step {self.global_step_counter}, "
                         f"update {self.global_update_counter}.")
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}")