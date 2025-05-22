# agent/ppo_agent.py - Updated with proper dashboard progress tracking
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
from envs.env_dashboard import TrainingStage
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

        # Dashboard integration
        self.is_evaluating = False

        # Performance tracking for dashboard
        self.last_update_start_time = 0.0
        self.last_rollout_start_time = 0.0

        logging.info(f"🤖 PPOTrainer initialized. Device: {self.device}, LR: {self.lr}, Rollout Steps: {self.rollout_steps}")

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

    def _update_dashboard_training_info(self, is_training: bool = True, is_evaluating: bool = False):
        """Update the environment's dashboard with current training information"""
        if hasattr(self.env, 'dashboard') and self.env.dashboard:
            self.env.set_training_info(
                episode_num=self.global_episode_counter,
                total_episodes=0,  # We don't track total episodes in advance
                total_steps=self.global_step_counter,
                update_count=self.global_update_counter,
                buffer_size=self.rollout_steps,
                is_training=is_training,
                is_evaluating=is_evaluating,
                learning_rate=self.lr
            )

    def collect_rollout_data(self) -> Dict[str, Any]:
        """Collects data for a fixed number of steps with proper dashboard progress tracking."""
        logging.info(f"🎲 Starting rollout data collection for {self.rollout_steps} steps...")
        self.buffer.clear()

        # Record start time for performance tracking
        self.last_rollout_start_time = time.time()

        # Update dashboard that we're collecting rollouts (training mode)
        self._update_dashboard_training_info(is_training=True, is_evaluating=False)

        current_env_state_np, _ = self.env.reset()

        for callback in self.callbacks:
            callback.on_rollout_start(self)

        collected_steps = 0
        episode_rewards_in_rollout = []
        episode_lengths_in_rollout = []
        current_episode_reward = 0.0
        current_episode_length = 0

        while collected_steps < self.rollout_steps:
            single_step_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in current_env_state_np.items()
            }

            # Update dashboard with rollout progress
            if self.dashboard:
                rollout_progress = collected_steps / self.rollout_steps
                self.dashboard.set_training_stage(
                    TrainingStage.COLLECTING_ROLLOUT,
                    None,  # Don't change overall progress
                    f"Collecting step {collected_steps}/{self.rollout_steps}",
                    rollout_progress  # Set substage progress
                )

                # Update training metrics with current collection status
                self.dashboard.update_training_metrics({
                    'collected_steps': collected_steps,
                    'rollout_steps': self.rollout_steps,
                })

            current_model_state_torch_batched = {}
            for key, tensor_val in single_step_tensors.items():
                if key in ['hf', 'mf', 'lf', 'portfolio']:
                    if tensor_val.ndim == 2:
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)
                    elif tensor_val.ndim == 3 and tensor_val.shape[0] == 1:
                        current_model_state_torch_batched[key] = tensor_val
                    else:
                        logging.error(f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}'. Expected 2D. Shape: {tensor_val.shape}")
                        current_model_state_torch_batched[key] = tensor_val
                elif key == 'static':
                    if tensor_val.ndim == 2 and tensor_val.shape[0] == 1:
                        current_model_state_torch_batched[key] = tensor_val
                    elif tensor_val.ndim == 1:
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)
                    else:
                        logging.error(
                            f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}' (static). Expected 1D or 2D (1,F). Shape: {tensor_val.shape}")
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
            self.global_step_counter += 1
            collected_steps += 1
            current_episode_reward += reward
            current_episode_length += 1

            for callback in self.callbacks:
                callback.on_step(self, current_model_state_torch_batched, action_tensor, reward, next_env_state_np,
                                 info)

            if done:
                self.global_episode_counter += 1
                episode_rewards_in_rollout.append(current_episode_reward)
                episode_lengths_in_rollout.append(current_episode_length)

                logging.info(f"🏁 Episode {self.global_episode_counter} finished. "
                             f"Reward: {current_episode_reward:.2f}, "
                             f"Length: {current_episode_length}, "
                             f"Global Steps: {self.global_step_counter}")

                for callback in self.callbacks:
                    callback.on_episode_end(self, current_episode_reward, current_episode_length, info)

                current_env_state_np, _ = self.env.reset()
                current_episode_reward = 0.0
                current_episode_length = 0

                if collected_steps >= self.rollout_steps:
                    break

        for callback in self.callbacks:
            callback.on_rollout_end(self)

        self.buffer.prepare_data_for_training()

        # Calculate performance metrics
        rollout_time = time.time() - self.last_rollout_start_time
        steps_per_second = collected_steps / rollout_time if rollout_time > 0 else 0

        rollout_stats = {
            "collected_steps": collected_steps,
            "mean_reward": np.mean(episode_rewards_in_rollout) if episode_rewards_in_rollout else 0,
            "mean_episode_length": np.mean(episode_lengths_in_rollout) if episode_lengths_in_rollout else 0,
            "num_episodes_in_rollout": len(episode_rewards_in_rollout),
            "rollout_time": rollout_time,
            "steps_per_second": steps_per_second
        }

        # Update dashboard with rollout completion
        if self.dashboard:
            self.dashboard.update_training_metrics({
                'collected_steps': collected_steps,
                'mean_episode_reward': rollout_stats["mean_reward"],
                'mean_episode_length': rollout_stats["mean_episode_length"],
                'episodes_per_update': rollout_stats["num_episodes_in_rollout"],
                'steps_per_second': steps_per_second,
            })

        logging.info(f"📊 Rollout finished. {rollout_stats}")
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

        logging.debug(f"Computing advantages with shapes - rewards: {rewards.shape}, values: {values.shape}, dones: {dones.shape}")

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
        logging.debug(f"Final shapes - advantages: {self.buffer.advantages.shape}, returns: {self.buffer.returns.shape}")

    def update_policy(self) -> Dict[str, float]:
        """Performs PPO updates for ppo_epochs using data in the buffer with proper dashboard progress tracking."""
        # Record start time for performance tracking
        self.last_update_start_time = time.time()

        # Update dashboard that we're updating policy
        self._update_dashboard_training_info(is_training=True, is_evaluating=False)

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

        logging.info(f"🔄 Starting PPO update for {self.ppo_epochs} epochs with {num_samples} samples")

        update_idx = 0
        for epoch in range(self.ppo_epochs):
            # Update dashboard with epoch progress
            if self.dashboard:
                epoch_progress = epoch / self.ppo_epochs
                self.dashboard.set_training_stage(
                    TrainingStage.UPDATING_POLICY,
                    None,  # Don't change overall progress
                    f"Policy update epoch {epoch + 1}/{self.ppo_epochs}",
                    epoch_progress  # Set substage progress
                )

            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx: start_idx + self.batch_size]

                # Update dashboard with batch progress within epoch
                if self.dashboard:
                    batch_progress = update_idx / total_updates
                    batch_num = (start_idx // self.batch_size) + 1
                    epoch_batches = (num_samples + self.batch_size - 1) // self.batch_size
                    self.dashboard.set_training_stage(
                        TrainingStage.UPDATING_POLICY,
                        None,  # Don't change overall progress
                        f"Epoch {epoch + 1}/{self.ppo_epochs}, Batch {batch_num}/{epoch_batches}",
                        batch_progress  # Set substage progress
                    )

                batch_states = {key: tensor_val[batch_indices] for key, tensor_val in states_dict.items()}
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                action_params, current_values = self.model(batch_states)

                if epoch == 0 and start_idx == 0:
                    logging.debug(f"Batch shapes - actions: {batch_actions.shape}, old_log_probs: {batch_old_log_probs.shape}, "
                                  f"advantages: {batch_advantages.shape}, returns: {batch_returns.shape}")

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

                logging.debug(f"Shape check - current_values: {current_values.shape}, batch_returns: {batch_returns.shape}")

                if batch_returns.ndim > 1 and batch_returns.size(1) > 1:
                    logging.warning(f"Unexpected batch_returns shape: {batch_returns.shape}. Taking first column.")
                    batch_returns = batch_returns[:, 0:1]

                current_values_shaped = current_values.view(-1, 1)
                batch_returns_shaped = batch_returns.view(-1, 1)

                if current_values_shaped.size(0) != batch_returns_shaped.size(0):
                    logging.error(f"Shape mismatch after reshaping: {current_values_shaped.shape} vs {batch_returns_shaped.shape}")
                    min_size = min(current_values_shaped.size(0), batch_returns_shaped.size(0))
                    current_values_shaped = current_values_shaped[:min_size]
                    batch_returns_shaped = batch_returns_shaped[:min_size]

                critic_loss = nnf.mse_loss(current_values_shaped, batch_returns_shaped)
                entropy_loss = -entropy.mean()
                loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.mean().item()
                num_updates_in_epoch += 1
                update_idx += 1

        self.global_update_counter += 1

        # Calculate performance metrics
        update_time = time.time() - self.last_update_start_time

        avg_actor_loss = total_actor_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_entropy = total_entropy_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0

        update_metrics = {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "total_loss": avg_actor_loss + self.critic_coef * avg_critic_loss + self.entropy_coef * (-avg_entropy),
            "time_per_update": update_time
        }

        # Update dashboard with final metrics
        if self.dashboard:
            self.dashboard.update_training_metrics({
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'entropy': avg_entropy,
                'time_per_update': update_time,
                'update_count': self.global_update_counter
            })

        for callback in self.callbacks:
            callback.on_update_end(self, update_metrics)

        logging.info(f"📈 PPO Update {self.global_update_counter}: "
                     f"Actor Loss: {avg_actor_loss:.4f}, "
                     f"Critic Loss: {avg_critic_loss:.4f}, "
                     f"Entropy: {avg_entropy:.4f}, "
                     f"Time: {update_time:.1f}s")

        return update_metrics

    def train(self, total_training_steps: int, eval_freq_steps: Optional[int] = None):
        """Main training loop with enhanced logging."""
        logging.info(f"🚀 Starting PPO training for a total of {total_training_steps} environment steps.")

        for callback in self.callbacks:
            callback.on_training_start(self)

        best_eval_reward = -float('inf')

        while self.global_step_counter < total_training_steps:
            logging.info(f"🔄 Update {self.global_update_counter + 1} - Collecting rollout data...")
            rollout_info = self.collect_rollout_data()

            if self.buffer.get_size() < self.rollout_steps and self.buffer.get_size() < self.batch_size:
                logging.warning(f"Buffer size {self.buffer.get_size()} is less than rollout_steps {self.rollout_steps} "
                                f"or batch_size {self.batch_size} after collection. Might be due to early episode ends. Skipping update if too small.")
                if self.buffer.get_size() < self.batch_size:
                    continue

            logging.info(f"🧠 Updating policy with {self.buffer.get_size()} samples...")
            update_metrics = self.update_policy()

            for callback in self.callbacks:
                callback.on_update_iteration_end(self, self.global_update_counter, update_metrics, rollout_info)

            # Evaluation with dashboard update
            if eval_freq_steps and (self.global_update_counter % (
                    eval_freq_steps // self.rollout_steps) == 0 or self.global_step_counter >= total_training_steps):

                # Update dashboard that we're evaluating
                self._update_dashboard_training_info(is_training=False, is_evaluating=True)

                logging.info(f"🔍 Running evaluation at step {self.global_step_counter}...")
                eval_stats = self.evaluate(n_episodes=10)
                logging.info(f"📊 Evaluation at step {self.global_step_counter}: Mean Reward: {eval_stats['mean_reward']:.2f}")

                # Reset dashboard back to training mode
                self._update_dashboard_training_info(is_training=True, is_evaluating=False)

                for callback in self.callbacks:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_stats.items() if
                                    k != 'episode_rewards' and k != 'episode_lengths'}
                    eval_metrics["global_step"] = self.global_step_counter
                    callback.on_update_iteration_end(self, self.global_update_counter, eval_metrics, {})

                if eval_stats['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_stats['mean_reward']
                    best_model_path = os.path.join(self.model_dir, f"best_model_update_{self.global_update_counter}.pt")
                    self.save_model(best_model_path)
                    logging.info(f"🏆 New best model saved with eval reward: {best_eval_reward:.2f} at {best_model_path}")

                latest_model_path = os.path.join(self.model_dir, "latest_model.pt")
                self.save_model(latest_model_path)

        final_stats = {"total_steps_trained": self.global_step_counter, "total_updates": self.global_update_counter}

        for callback in self.callbacks:
            callback.on_training_end(self, final_stats)

        logging.info(f"🎉 Training finished! Total steps: {self.global_step_counter}, Total PPO updates: {self.global_update_counter}")
        return final_stats

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """Evaluates the current model policy with enhanced logging and dashboard updates."""
        logging.info(f"🔍 Starting evaluation for {n_episodes} episodes (deterministic: {deterministic})...")

        # Set model to evaluation mode
        self.model.eval()
        self.is_evaluating = True

        # Update dashboard
        self._update_dashboard_training_info(is_training=False, is_evaluating=True)

        episode_rewards = []
        episode_lengths = []

        for i in range(n_episodes):
            # Update dashboard with evaluation progress
            if self.dashboard:
                eval_progress = i / n_episodes
                self.dashboard.set_training_stage(
                    TrainingStage.EVALUATING,
                    None,  # Don't change overall progress
                    f"Evaluation episode {i + 1}/{n_episodes}",
                    eval_progress  # Set substage progress
                )

            env_state_np, _ = self.env.reset()
            current_episode_reward = 0.0
            current_episode_length = 0
            done = False

            while not done:
                model_state_torch = convert_state_dict_to_tensors(env_state_np, self.device)
                with torch.no_grad():
                    action_tensor, _ = self.model.get_action(model_state_torch, deterministic=deterministic)

                env_action = self._convert_action_for_env(action_tensor)
                next_env_state_np = None

                try:
                    next_env_state_np, reward, terminated, truncated, _ = self.env.step(env_action)
                    done = terminated or truncated
                except Exception as e:
                    logging.error(f"Error during evaluation step: {e}")
                    done = True
                    reward = 0

                env_state_np = next_env_state_np
                current_episode_reward += reward
                current_episode_length += 1

            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            logging.debug(f"Eval Episode {i + 1}/{n_episodes} finished. "
                          f"Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")

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

        logging.info(f"📊 Evaluation complete! Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        return eval_results

    def save_model(self, path: str) -> None:
        """Saves the model and optimizer state with enhanced logging."""
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
            logging.info(f"💾 Model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving model to {path}: {e}")

    def load_model(self, path: str) -> None:
        """Loads the model and optimizer state with enhanced logging."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.global_step_counter = checkpoint.get('global_step_counter', 0)
            self.global_episode_counter = checkpoint.get('global_episode_counter', 0)
            self.global_update_counter = checkpoint.get('global_update_counter', 0)

            self.model.to(self.device)
            logging.info(f"📂 Model loaded from {path}. "
                         f"Resuming from step {self.global_step_counter}, "
                         f"update {self.global_update_counter}.")
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}")