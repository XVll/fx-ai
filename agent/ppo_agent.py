# agent/ppo_agent.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Any, Optional
import logging
import wandb  # Assuming wandb is used
from omegaconf import DictConfig
import torch.nn.functional as nnf

from config.config import ModelConfig
# Assuming these are your custom module imports
from envs.trading_env import TradingEnvironment  # Your environment class
from ai.transformer import MultiBranchTransformer  # Your model class
from agent.utils import ReplayBuffer, convert_state_dict_to_tensors  # From the rewritten utils
from agent.callbacks import TrainingCallback  # Your callback base class

logger = logging.getLogger(__name__)


class PPOTrainer:
    def __init__(
            self,
            env: TradingEnvironment,
            model: MultiBranchTransformer,  # Expects MultiBranchTransformer specifically
            model_config: ModelConfig = None,  # For model details if needed elsewhere
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            critic_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            ppo_epochs: int = 10,  # Number of epochs to train on the collected data
            batch_size: int = 64,  # Minibatch size for PPO updates
            rollout_steps: int = 2048,  # Number of steps to collect per rollout before updating
            device: Optional[Union[str, torch.device]] = None,
            output_dir: str = "./ppo_output",
            use_wandb: bool = True,
            callbacks: Optional[List[TrainingCallback]] = None,
    ):
        self.env = env
        self.model = model
        self.model_config = model_config if model_config else {}  # Store for reference

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
        self.rollout_steps = rollout_steps  # Buffer capacity will be this

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.model.to(self.device)  # Ensure the model is on the correct device

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=self.rollout_steps, device=self.device)

        # Output directories
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, "ai")
        os.makedirs(self.model_dir, exist_ok=True)

        # Logging and W&B
        self.use_wandb = use_wandb
        if self.use_wandb and wandb.run is None:  # Check if wandb is already initialized
            logger.warning("W&B not initialized. Consider initializing wandb.init() before trainer.")
            # Or initialize here: wandb.init(project="your_project_name")

        # Callbacks
        self.callbacks = callbacks if callbacks else []

        # Training state
        self.global_step_counter = 0  # Total environment steps taken
        self.global_episode_counter = 0
        self.global_update_counter = 0

        logger.info(f"PPOTrainer initialized. Device: {self.device}, LR: {self.lr}, Rollout Steps: {self.rollout_steps}")
        logger.info(f"Model: {type(self.model).__name__}")

    def _convert_action_for_env(self, action_tensor: torch.Tensor) -> Any:
        """Converts model's action tensor to environment-compatible format."""
        if self.model.continuous_action:
            action_np = action_tensor.cpu().numpy().squeeze()
            # Ensure it's a 1D array if squeezed to scalar
            return np.array([action_np], dtype=np.float32) if np.isscalar(action_np) else action_np.astype(np.float32)
        else:  # Discrete actions
            # For MultiDiscrete actions (which we're using)
            if action_tensor.ndim > 0 and action_tensor.shape[-1] == 2:
                # Convert to NumPy array rather than tuple for consistency
                return action_tensor.cpu().numpy().squeeze().astype(int)
            else:  # Single discrete action
                return action_tensor.cpu().numpy().item()

    def collect_rollout_data(self) -> Dict[str, Any]:
        """Collects data for a fixed number of steps (self.rollout_steps)."""
        logger.info(f"Starting rollout data collection for {self.rollout_steps} steps...")
        self.buffer.clear()  # Clear buffer before a new rollout collection

        # Get initial state from environment
        # The environment should return Dict[str, np.ndarray]
        current_env_state_np, _ = self.env.reset()

        # For callback on_rollout_start
        for callback in self.callbacks: callback.on_rollout_start(self)

        collected_steps = 0
        episode_rewards_in_rollout = []
        episode_lengths_in_rollout = []
        current_episode_reward = 0.0
        current_episode_length = 0

        while collected_steps < self.rollout_steps:
            # Convert the current environment state (NumPy dict) to PyTorch tensor dict for the model
            # Convert NumPy dict from environment to PyTorch tensor dict
            # (This step might be handled by your `convert_state_dict_to_tensors` utility.
            # If that utility doesn't add the batch dimension, you do it here.)
            single_step_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in current_env_state_np.items()
            }

            # Add the batch dimension to each tensor as expected by the model.
            # The model's forward pass (and thus get_action) expects:
            # - hf, mf, lf, portfolio: [batch_size, sequence_length, feature_dimension]
            # - static: [batch_size, feature_dimension]
            current_model_state_torch_batched = {}
            for key, tensor_val in single_step_tensors.items():
                if key in ['hf', 'mf', 'lf', 'portfolio']:  # Sequential features
                    if tensor_val.ndim == 2:  # Expected shape from env: [sequence_length, feature_dimension]
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)  # Add batch dim: [1, sequence_length, feature_dimension]
                    elif tensor_val.ndim == 3 and tensor_val.shape[0] == 1:  # Already correctly batched
                        current_model_state_torch_batched[key] = tensor_val
                    else:
                        # Log an error if the tensor shape is unexpected for sequential data
                        logger.error(f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}'. Expected 2D. Shape: {tensor_val.shape}")
                        current_model_state_torch_batched[key] = tensor_val  # Fallback, may cause issues
                elif key == 'static':  # Static features
                    if tensor_val.ndim == 2 and tensor_val.shape[0] == 1:  # Expected shape from env: [1, feature_dimension]
                        current_model_state_torch_batched[key] = tensor_val  # Already correctly batched: [1, feature_dimension]
                    elif tensor_val.ndim == 1:  # Shape from env might be [feature_dimension]
                        current_model_state_torch_batched[key] = tensor_val.unsqueeze(0)  # Add batch dim: [1, feature_dimension]
                    else:
                        # Log an error if the tensor shape is unexpected for static data
                        logger.error(f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}' (static). Expected 1D or 2D (1,F). Shape: {tensor_val.shape}")
                        current_model_state_torch_batched[key] = tensor_val  # Fallback
                else:
                    current_model_state_torch_batched[key] = tensor_val  # Handle any other keys as they are

            with torch.no_grad():
                # Model's get_action should return action tensor and action_info dict (with 'value', 'log_prob')
                action_tensor, action_info = self.model.get_action(current_model_state_torch_batched, deterministic=False)

            # Convert action tensor to environment-compatible format
            env_action = self._convert_action_for_env(action_tensor)

            try:
                next_env_state_np, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
            except Exception as e:
                logger.error(f"Error during environment step: {e}", exc_info=True)
                break  # End rollout if an environment step fails

            self.buffer.add(
                current_env_state_np,
                action_tensor,  # Store model's tensor action
                reward,
                next_env_state_np,
                done,
                action_info  # Contains value and log_prob tensors
            )

            current_env_state_np = next_env_state_np
            self.global_step_counter += 1
            collected_steps += 1
            current_episode_reward += reward
            current_episode_length += 1

            # For callback on_step
            for callback in self.callbacks: callback.on_step(self, current_model_state_torch_batched, action_tensor, reward, next_env_state_np, info)

            if done:
                self.global_episode_counter += 1
                episode_rewards_in_rollout.append(current_episode_reward)
                episode_lengths_in_rollout.append(current_episode_length)
                logger.info(
                    f"Episode {self.global_episode_counter} finished. Reward: {current_episode_reward:.2f}, Length: {current_episode_length}, Global Steps: {self.global_step_counter}")
                if self.use_wandb:
                    wandb.log({
                        "rollout/episode_reward": current_episode_reward,
                        "rollout/episode_length": current_episode_length,
                        "global_step": self.global_step_counter
                    }, step=self.global_update_counter)  # Log against PPO updates

                # For callback on_episode_end
                for callback in self.callbacks: callback.on_episode_end(self, current_episode_reward, current_episode_length, info)

                current_env_state_np, _ = self.env.reset()  # Reset for the next episode within the rollout
                current_episode_reward = 0.0
                current_episode_length = 0

                if collected_steps >= self.rollout_steps:  # Ensure we don't overshoot if an episode ends exactly at rollout_steps
                    break

        # For callback on_rollout_end
        for callback in self.callbacks: callback.on_rollout_end(self)

        # Prepare buffer data (stacks all collected experiences into large tensors)
        self.buffer.prepare_data_for_training()

        rollout_stats = {
            "collected_steps": collected_steps,
            "mean_episode_reward": np.mean(episode_rewards_in_rollout) if episode_rewards_in_rollout else 0,
            "mean_episode_length": np.mean(episode_lengths_in_rollout) if episode_lengths_in_rollout else 0,
            "num_episodes_in_rollout": len(episode_rewards_in_rollout)
        }
        logger.info(f"Rollout finished. {rollout_stats}")
        return rollout_stats

    def _compute_advantages_and_returns(self):
        """Computes GAE advantages and returns, storing them in the buffer."""
        if self.buffer.rewards is None or self.buffer.values is None or self.buffer.dones is None:
            logger.error("Cannot compute advantages: buffer data (rewards, values, dones) not prepared.")
            return

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        num_steps = len(rewards)

        # Debug shapes to help diagnose issues
        logger.debug(
            f"Computing advantages - shapes: rewards:{rewards.shape}, values:{values.shape}, dones:{dones.shape}")

        # Initialize advantages to zeros with the same shape as rewards
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae_lam = 0

        for t in reversed(range(num_steps)):
            # Determine next value based on whether episode is done or not
            if t == num_steps - 1:  # Last step in the buffer
                if dones[t]:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    # Bootstrap with the value of the state *after* the last action of the rollout
                    # This requires proper tensor shape handling to avoid dimension errors
                    last_exp_next_state_dict = self.buffer.buffer[-1]['next_state']

                    # Create properly batched state dict for the model
                    batched_state_dict = {}
                    for key, tensor_val in last_exp_next_state_dict.items():
                        if key in ['hf', 'mf', 'lf', 'portfolio']:  # Sequential features
                            if tensor_val.ndim == 2:  # [sequence_length, feature_dimension]
                                batched_state_dict[key] = tensor_val.unsqueeze(0)
                            elif tensor_val.ndim == 3:  # Already has batch dim
                                batched_state_dict[key] = tensor_val
                            else:
                                logger.warning(f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}'")
                                batched_state_dict[key] = tensor_val
                        elif key == 'static':  # Static features
                            if tensor_val.ndim == 1:  # [feature_dimension]
                                batched_state_dict[key] = tensor_val.unsqueeze(0)
                            elif tensor_val.ndim == 2:  # Already has batch dim
                                batched_state_dict[key] = tensor_val
                            else:
                                logger.warning(f"Unexpected tensor ndim ({tensor_val.ndim}) for key '{key}'")
                                batched_state_dict[key] = tensor_val
                        else:
                            batched_state_dict[key] = tensor_val

                    try:
                        # Get value from model
                        with torch.no_grad():
                            _, next_value = self.model(batched_state_dict)
                            # Make next_value a scalar-like tensor
                            if next_value.ndim > 0:
                                next_value = next_value.squeeze()
                                if next_value.ndim > 0:  # Still has dimensions
                                    next_value = next_value[0]  # Take first element if multi-dimensional
                    except Exception as e:
                        logger.error(f"Error computing bootstrap value: {e}")
                        next_value = torch.tensor([0.0], device=self.device)
            else:
                if dones[t]:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    next_value = values[t + 1]
                    # Handle next_value dimensions
                    if next_value.ndim > 0:
                        next_value = next_value.squeeze()
                        if next_value.ndim > 0:  # Still has dimensions
                            next_value = next_value[0]  # Take first element

            # Calculate TD error (delta)
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t].float()) - values[t]

            # Update advantage using GAE formula
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (
                        1.0 - dones[t].float()) * last_gae_lam

        # Ensure advantages has the correct shape before setting
        if advantages.ndim > 1 and advantages.shape[1] > 1:
            # If advantages has multiple columns, aggregate (take mean or first column)
            logger.warning(f"Advantages has multiple columns: {advantages.shape}. Taking first column.")
            advantages = advantages[:, 0:1]  # Keep as 2D with shape [num_steps, 1]

        # Ensure values has correct shape for returns calculation
        if values.ndim > 1 and values.shape[1] > 1:
            logger.warning(f"Values has multiple columns: {values.shape}. Taking first column.")
            values = values[:, 0:1]  # Keep as 2D with shape [num_steps, 1]

        # Compute returns as advantages + value estimates
        self.buffer.advantages = advantages
        self.buffer.returns = advantages + values  # Rt = At + Vt

        # Log final shapes
        logger.debug(f"Computed advantages shape: {self.buffer.advantages.shape}")
        logger.debug(f"Computed returns shape: {self.buffer.returns.shape}")

    def update_policy(self) -> Dict[str, float]:
        """Performs PPO updates for ppo_epochs using data in the buffer."""
        # Compute advantages and returns first
        self._compute_advantages_and_returns()

        training_data = self.buffer.get_training_data()
        if training_data is None:
            logger.error("Skipping policy update due to missing training data in buffer.")
            return {}

        states_dict = training_data["states"]
        actions = training_data["actions"]
        old_log_probs = training_data["old_log_probs"]
        advantages = training_data["advantages"]
        returns = training_data["returns"]

        # Normalize advantages (important for PPO stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = actions.size(0)
        if num_samples == 0:
            logger.warning("No samples in buffer to update policy. Skipping update.")
            return {}

        # Create indices for batching
        indices = np.arange(num_samples)

        # For callback on_update_start
        for callback in self.callbacks: callback.on_update_start(self)

        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        num_updates_in_epoch = 0

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)  # Shuffle for each epoch

            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx: start_idx + self.batch_size]

                # Create a batch states dictionary
                batch_states = {key: tensor_val[batch_indices] for key, tensor_val in states_dict.items()}
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new log probs, entropy, and values from current policy
                action_params, current_values = self.model(batch_states)

                # FIXED: Ensure current_values is the right shape
                if current_values.ndim > 1 and current_values.size(-1) == 1:
                    current_values = current_values.squeeze(-1)  # Remove last dimension if it's 1

                # Calculate new log probabilities and entropy
                if self.model.continuous_action:
                    # [continuous action handling code...]
                    pass
                else:  # Discrete actions
                    action_type_logits, action_size_logits = action_params  # Assuming tuple action

                    # batch_actions should be shape [batch_size, 2] for (type, size)
                    action_types_taken = batch_actions[:, 0].long()
                    action_sizes_taken = batch_actions[:, 1].long()

                    type_dist = torch.distributions.Categorical(logits=action_type_logits)
                    size_dist = torch.distributions.Categorical(logits=action_size_logits)

                    new_type_log_probs = type_dist.log_prob(action_types_taken)
                    new_size_log_probs = size_dist.log_prob(action_sizes_taken)
                    new_log_probs = (new_type_log_probs + new_size_log_probs).unsqueeze(1)  # Ensure [batch, 1]

                    entropy = (type_dist.entropy() + size_dist.entropy()).unsqueeze(1)  # Ensure [batch, 1]

                # PPO Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped Surrogate Objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # CRITICAL FIX: Ensure returns and values have the same shape before loss calculation
                # Debug the shapes
                logger.debug(
                    f"Shape check - current_values: {current_values.shape}, batch_returns: {batch_returns.shape}")

                # Ensure both are of compatible shapes for MSE loss
                # First, make sure they're both 1D or both 2D with shape [batch_size, 1]
                if batch_returns.ndim > 1 and batch_returns.size(1) > 1:
                    logger.warning(f"Unexpected batch_returns shape: {batch_returns.shape}. Taking first column.")
                    batch_returns = batch_returns[:, 0]  # Take just the first column

                # Now reshape both to [batch_size, 1] for consistent calculation
                current_values_shaped = current_values.view(-1, 1)
                batch_returns_shaped = batch_returns.view(-1, 1)

                # One final check to ensure shapes match
                if current_values_shaped.size(0) != batch_returns_shaped.size(0):
                    logger.error(
                        f"Shape mismatch after reshaping: {current_values_shaped.shape} vs {batch_returns_shaped.shape}")
                    # Use the smaller size to avoid errors
                    min_size = min(current_values_shaped.size(0), batch_returns_shaped.size(0))
                    current_values_shaped = current_values_shaped[:min_size]
                    batch_returns_shaped = batch_returns_shaped[:min_size]

                # Now the shapes should match for the MSE loss
                critic_loss = nnf.mse_loss(current_values_shaped, batch_returns_shaped)

                # Entropy Bonus
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.mean().item()  # Use actual entropy value, not loss
                num_updates_in_epoch += 1

            # End of minibatch loop
        # End of epoch loop
        self.global_update_counter += 1

        avg_actor_loss = total_actor_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0
        avg_entropy = total_entropy_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0

        update_metrics = {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "total_loss": avg_actor_loss + self.critic_coef * avg_critic_loss + self.entropy_coef * (-avg_entropy)
        }

        # For callback on_update_end
        for callback in self.callbacks: callback.on_update_end(self, update_metrics)

        logger.info(f"PPO Update {self.global_update_counter}: {update_metrics}")
        if self.use_wandb:
            wandb.log({f"update/{k}": v for k, v in update_metrics.items()}, step=self.global_update_counter)
            wandb.log({"global_step": self.global_step_counter}, step=self.global_update_counter)

        return update_metrics

        # For callback on_update_end
        for callback in self.callbacks: callback.on_update_end(self, update_metrics)

        logger.info(f"PPO Update {self.global_update_counter}: {update_metrics}")
        if self.use_wandb:
            wandb.log({f"update/{k}": v for k, v in update_metrics.items()}, step=self.global_update_counter)
            wandb.log({"global_step": self.global_step_counter}, step=self.global_update_counter)

        return update_metrics

    def train(self, total_training_steps: int, eval_freq_steps: Optional[int] = None):
        """Main training loop."""
        logger.info(f"Starting PPO training for a total of {total_training_steps} environment steps.")
        # For callback on_training_start
        for callback in self.callbacks: callback.on_training_start(self)

        best_eval_reward = -float('inf')

        while self.global_step_counter < total_training_steps:
            rollout_info = self.collect_rollout_data()  # Collects self.rollout_steps

            if self.buffer.get_size() < self.rollout_steps and self.buffer.get_size() < self.batch_size:  # Or some minimum
                logger.warning(f"Buffer size {self.buffer.get_size()} is less than rollout_steps {self.rollout_steps} "
                               f"or batch_size {self.batch_size} after collection. Might be due to early episode ends. Skipping update if too small.")
                if self.buffer.get_size() < self.batch_size:  # Ensure enough data for at least one batch
                    continue

            update_metrics = self.update_policy()

            # For callback on_update_iteration_end
            for callback in self.callbacks:
                callback.on_update_iteration_end(self, self.global_update_counter, update_metrics, rollout_info)

            # Evaluation (optional)
            if eval_freq_steps and (self.global_update_counter % (
                    eval_freq_steps // self.rollout_steps) == 0 or self.global_step_counter >= total_training_steps):  # Approx eval freq
                eval_stats = self.evaluate(n_episodes=10)  # Number of eval episodes
                logger.info(f"Evaluation at step {self.global_step_counter}: Mean Reward: {eval_stats['mean_reward']:.2f}")
                if self.use_wandb:
                    wandb.log({
                        "eval/mean_reward": eval_stats['mean_reward'],
                        "eval/mean_length": eval_stats['mean_length'],
                        "global_step": self.global_step_counter
                    }, step=self.global_update_counter)

                if eval_stats['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_stats['mean_reward']
                    self.save_model(os.path.join(self.model_dir, f"best_model_update_{self.global_update_counter}.pt"))
                    logger.info(f"New best model saved with eval reward: {best_eval_reward:.2f}")

                self.save_model(os.path.join(self.model_dir, "latest_model.pt"))

        final_stats = {"total_steps_trained": self.global_step_counter, "total_updates": self.global_update_counter}
        # For callback on_training_end
        for callback in self.callbacks: callback.on_training_end(self, final_stats)
        logger.info(f"Training finished. Total steps: {self.global_step_counter}, Total PPO updates: {self.global_update_counter}")
        return final_stats

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """Evaluates the current model policy."""
        logger.info(f"Starting evaluation for {n_episodes} episodes (deterministic: {deterministic})...")
        self.model.eval()  # Set the model to evaluation mode

        episode_rewards = []
        episode_lengths = []

        for i in range(n_episodes):
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
                    logger.error(f"Error during evaluation step: {e}", exc_info=True)
                    done = True  # Terminate episode on error
                    reward = 0  # Or some penalty

                env_state_np = next_env_state_np
                current_episode_reward += reward
                current_episode_length += 1

            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            logger.debug(f"Eval Episode {i + 1}/{n_episodes} finished. Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")

        self.model.train()  # Set the model back to training mode

        eval_results = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
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
                'model_config': self.model_config  # Save model config for reproducibility
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}", exc_info=True)

    def load_model(self, path: str) -> None:
        """Loads the model and optimizer state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.global_step_counter = checkpoint.get('global_step_counter', 0)
            self.global_episode_counter = checkpoint.get('global_episode_counter', 0)
            self.global_update_counter = checkpoint.get('global_update_counter', 0)
            # self.model_config = checkpoint.get('model_config', self.model_config) # Optionally load if needed

            self.model.to(self.device)  # Ensure the model is on the correct device after loading
            logger.info(f"Model loaded from {path}. Resuming from step {self.global_step_counter}, update {self.global_update_counter}.")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}", exc_info=True)
