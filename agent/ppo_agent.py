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

# Assuming these are your custom module imports
from envs.trading_env import TradingEnvironment  # Your environment class
from models.transformer import MultiBranchTransformer  # Your model class
from agent.utils import ReplayBuffer, convert_state_dict_to_tensors  # From the rewritten utils
from agent.callbacks import TrainingCallback  # Your callback base class

logger = logging.getLogger(__name__)


class PPOTrainer:
    def __init__(
            self,
            env: TradingEnvironment,
            model: MultiBranchTransformer,  # Expects MultiBranchTransformer specifically
            model_config: Optional[Union[Dict[str, Any], DictConfig]] = None,  # For model details if needed elsewhere
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
        self.model_dir = os.path.join(output_dir, "models")
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
            # Ensure it's a 1D array if squeezed to scalar, matching common env expectations
            return np.array([action_np], dtype=np.float32) if np.isscalar(action_np) else action_np.astype(np.float32)
        else:  # Discrete actions
            # For MultiDiscrete, model.get_action should return a tensor like [action_type_idx, action_size_idx]
            if action_tensor.ndim > 0 and action_tensor.shape[-1] == 2:  # Assuming (..., 2) for MultiDiscrete
                return tuple(action_tensor.cpu().numpy().squeeze().astype(int))
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
            current_model_state_torch = convert_state_dict_to_tensors(current_env_state_np, self.device)

            with torch.no_grad():
                # Model's get_action should return action tensor and action_info dict (with 'value', 'log_prob')
                action_tensor, action_info = self.model.get_action(current_model_state_torch, deterministic=False)

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
            for callback in self.callbacks: callback.on_step(self, current_model_state_torch, action_tensor, reward, next_env_state_np, info)

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

        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae_lam = 0

        # Estimate the value of the last state if not done, otherwise 0
        # This requires getting the value of the *very last* next_state encountered in rollout
        # For simplicity, if the last step was 'done', next_value is 0.
        # If not, we would ideally compute model.value(last_next_state).
        # Common practice: if rollout ends not due to 'done', bootstrap from the last value.
        # Here, we assume the value tensor already includes V(s_t) for all t in rollout.
        # So V(s_{t+1}) is values[t+1] effectively.

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:  # Last step in the buffer
                # If the episode was not 'done' at this last step, we might need V(s_T+1)
                # For simplicity, if it's not done, the advantage calculation can use V(s_T) as if it were terminal,
                # or we'd need to have stored V(s_T+1) from the model.
                # Assuming values contains V(s_0)...V(s_T-1).
                # So, next_value for rewards[T-1] would be 0 if dones[T-1] is true, or V(s_T) if we had it.
                # A common simplification: if dones[t] is true, next_val_for_delta is 0.
                # If not, it's values[t+1] if t+1 is in buffer, or bootstrap V(s_last_next_state)

                # Let's assume values are V(s_t).
                # For the last element rewards[num_steps-1], delta uses V(s_{num_steps-1})
                # and needs a next_value. If dones[num_steps-1] is true, next_value = 0.
                # If dones[num_steps-1] is false, we'd ideally use the value of the *actual next state* after the rollout.
                # For PPO rollouts, the 'values' are V(s_t).
                # The 'next_value' for delta at step 't' is V(s_{t+1}).
                # So for the last step 'num_steps-1', the next_value for delta would be from V(s_{num_steps}).
                # We use the value of the state *after* the last action in the buffer.
                # This is usually handled by getting the value of `last_next_env_state_np` if the episode didn't end.

                # A practical way: `values` stores V(s_0)...V(s_N-1)
                # For GAE at step N-1, we need V(s_N).
                # If done[N-1] is true, V(s_N) = 0.
                # If done[N-1] is false, V(s_N) is estimated by the model on the actual next state s_N.
                # Since we are operating on a fixed buffer, if not done, the next_value is values[t+1]
                # For the last step in buffer:
                if dones[t]:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    # Bootstrap with the value of the state *after* the last action of the rollout
                    # This requires `self.buffer.buffer[-1]['next_state']` to be evaluated by the critic
                    # For simplicity, let's assume it if not done, we use the value estimate of the last *next_state*
                    # that was stored alongside the last experience in the buffer before `prepare_data_for_training`.
                    # This value is NOT in self.buffer.values (which are V(s_t) for t in buffer).
                    # We need to compute it.
                    last_exp_next_state_torch = self.buffer.buffer[-1]['next_state']  # This is Dict[str, Tensor]
                    with torch.no_grad():
                        _, next_value_container = self.model(last_exp_next_state_torch)  # Get value V(s_N)
                        next_value = next_value_container
            else:
                if dones[t]:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    next_value = values[t + 1]  # V(s_{t+1})

            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t].float()) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - dones[t].float()) * last_gae_lam

        self.buffer.advantages = advantages
        self.buffer.returns = advantages + values  # Rt = At + Vt
        logger.info("Advantages and returns computed and stored in buffer.")

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
        # old_values = training_data["values"] # V(s_t) from rollout

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
                # batch_old_values = old_values[batch_indices] # For value clipping if used

                # Get new log probs, entropy, and values from current policy
                # model.forward returns (action_params, value_estimate)
                action_params, current_values = self.model(batch_states)
                current_values = current_values.squeeze(-1)  # Ensure the correct shape if [batch, 1]

                # Calculate new log probabilities and entropy
                if self.model.continuous_action:
                    mean, log_std = action_params
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)

                    # For tanh-squashed actions, PPO typically works with the distribution *before* squashing
                    # The action_tensor in buffer is the squashed action. We need to "unsquash" it
                    # or re-evaluate the log_prob of the pre-squashed sample if that was stored.
                    # Assuming log_probs in buffer are for the squashed action WITH correction,
                    # or the PPO update should use the distribution of the pre-squashed action.
                    # For simplicity, let's assume log_probs in the buffer are from `dist.log_prob(pre_squashed_action) - correction`.
                    # When evaluating new log_probs for PPO, we need log_prob of batch_actions.
                    # If batch_actions are squashed, we need to apply the same logic.
                    # The model's get_action already provides log_prob including Tanh correction.
                    # So, batch_old_log_probs already has this.
                    # For new_log_probs, we need to re-evaluate based on batch_actions (which are squashed).
                    # To get log_prob of squashed action `a` from Normal(mu, sigma) then tanh:
                    #   x = atanh(a)
                    #   log_prob = Normal(mu, sigma).log_prob(x) - log(1 - a^2 + eps)
                    # This should match how old_log_probs were computed.

                    # Let's ensure `batch_actions` are correctly shaped for `dist.log_prob`
                    # If action_dim > 1, sum log_probs over action dimensions
                    # Model's `get_action` log_prob sum is `sum(1, keepdim=True)`

                    # Re-evaluate log_prob for the current policy based on actions taken
                    # inverse_tanh_actions = torch.atanh(batch_actions.clamp(-0.9999, 0.9999))
                    # new_log_probs = dist.log_prob(inverse_tanh_actions).sum(dim=-1, keepdim=True)
                    # entropy = dist.entropy().sum(dim=-1, keepdim=True)

                    # Alternative & often simpler: A PPO ratio is on the distribution directly.
                    # The critical part is that old_log_probs and new_log_probs are comparable.
                    # If model.get_action stores log_prob of the pre-squashed sample, then:
                    # new_log_probs = dist.log_prob(dist.rsample()).sum(-1, keepdim=True) - this isn't right for PPO.
                    # We need log_prob of *taken actions* under *current policy*.

                    # Let's assume 'batch_actions' are the Tanh-squashed actions.
                    # To get log_prob(batch_actions | current_policy):
                    # 1. Inverse Tanh: x_t = atanh(batch_actions)
                    # 2. Log prob of x_t undercurrent Gaussian: log_pi(x_t)
                    # 3. Tanh correction: log(1 - batch_actions^2)
                    # new_log_probs = log_pi(x_t) - log(1 - batch_actions^2)
                    # This must match how old_log_probs was computed.

                    # Let's follow the model's get_action structure for log_prob:
                    x_t_for_actions = torch.atanh(batch_actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6))  # Inverse of action
                    new_log_probs = dist.log_prob(x_t_for_actions)
                    # Tanh correction for log_prob
                    new_log_probs -= torch.log(1.0 - batch_actions.pow(2) + 1e-6)
                    new_log_probs = new_log_probs.sum(1, keepdim=True)
                    entropy = dist.entropy().sum(1, keepdim=True)

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

                # Critic Loss (Value Loss)
                # Optional: Value clipping (as in original PPO paper)
                # values_clipped = batch_old_values + torch.clamp(current_values - batch_old_values, -self.clip_eps, self.clip_eps)
                # critic_loss1 = F.mse_loss(current_values, batch_returns)
                # critic_loss2 = F.mse_loss(values_clipped, batch_returns)
                # critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean() # Or just simple MSE
                critic_loss = nnf.mse_loss(current_values.unsqueeze(1), batch_returns)

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
        avg_entropy = total_entropy_loss / num_updates_in_epoch if num_updates_in_epoch > 0 else 0  # This is actually avg entropy value

        update_metrics = {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "total_loss": avg_actor_loss + self.critic_coef * avg_critic_loss + self.entropy_coef * (-avg_entropy)
            # Reconstruct total loss with the correct entropy sign
        }

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
