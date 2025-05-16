import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Union, Any
import logging
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig

from models.transformer import MultiBranchTransformer
from envs.trading_env import TradingEnv
from agent.utils import ReplayBuffer, normalize_state_dict, preprocess_state_to_dict
from agent.callbacks import TrainingCallback


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for the Multi-Branch Transformer model.

    Implements PPO with clipped surrogate objective, critic loss, and entropy bonus.
    """

    def __init__(
            self,
            env: TradingEnv,
            model: MultiBranchTransformer,
            model_config: Union[Dict[str, Any], DictConfig] = None,
            # Training hyperparameters
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            critic_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            n_epochs: int = 10,
            batch_size: int = 64,
            # Buffer and rollout settings
            buffer_size: int = 2048,
            n_episodes_per_update: int = 8,
            # Other settings
            device: Union[str, torch.device] = None,
            output_dir: str = "./output",
            logger: logging.Logger = None,
            callbacks: List[TrainingCallback] = None
    ):
        """
        Initialize the PPO trainer.

        Args:
            env: Trading environment instance
            model: Multi-Branch Transformer model instance
            model_config: Configuration dictionary for model architecture (can be Hydra DictConfig)
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_eps: PPO clip epsilon
            critic_coef: Weight of critic loss
            entropy_coef: Weight of entropy bonus
            max_grad_norm: Max gradient norm for clipping
            n_epochs: Number of optimization epochs per update
            batch_size: Minibatch size for optimization
            buffer_size: Size of replay buffer
            n_episodes_per_update: Number of episodes to collect before updating
            device: Device to use (default: cuda if available, otherwise cpu)
            output_dir: Directory for saving models and logs
            logger: Logger object
            callbacks: List of callbacks for training hooks
        """
        # Environment and model
        self.env = env
        self.model = model

        # Handle Hydra config
        if model_config is not None and hasattr(model_config, "_to_dict"):
            self.model_config = model_config._to_dict()
        else:
            self.model_config = model_config or {}

        # Training parameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Rollout parameters
        self.buffer_size = buffer_size
        self.n_episodes_per_update = n_episodes_per_update

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Set up buffer
        self.buffer = ReplayBuffer(buffer_size, self.device)

        # Setup directories
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, "models")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logging
        self.logger = logger or logging.getLogger(__name__)

        # Callbacks
        self.callbacks = callbacks or []

        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.updates = 0

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []

        # Log info
        self.logger.info(f"PPO Trainer initialized on device: {self.device}")
        self.logger.info(f"Model: {type(self.model).__name__}")
        self.logger.info(f"Learning rate: {self.lr}")

    def collect_rollouts(self, n_episodes: int = None) -> Dict[str, Any]:
        """
        Collect rollouts from the environment.

        Args:
            n_episodes: Number of episodes to collect (if None, use n_episodes_per_update)

        Returns:
            Dictionary with collection statistics
        """
        n_episodes = n_episodes or self.n_episodes_per_update

        total_reward = 0.0
        total_steps = 0
        episode_rewards = []
        episode_lengths = []

        self.logger.info(f"Collecting {n_episodes} episodes...")

        # Call on_rollout_start for callbacks
        for callback in self.callbacks:
            callback.on_rollout_start(self)

        for episode in range(n_episodes):
            # Reset environment
            state, _ = self.env.reset()
            state_dict = preprocess_state_to_dict(state, self.model_config)

            # Track episode data
            episode_reward = 0
            episode_length = 0
            episode_done = False

            while not episode_done:
                # Get action from policy
                with torch.no_grad():
                    action, action_info = self.model.get_action(state_dict)

                # Convert action to the format expected by the environment
                if self.model.continuous_action:
                    # For continuous actions - ensure it's a 1D numpy array
                    action_np = action.cpu().numpy().squeeze()
                    if np.isscalar(action_np):
                        action_np = np.array([action_np], dtype=np.float32)
                else:
                    # Discrete actions - could be single or tuple
                    if isinstance(action, torch.Tensor) and action.shape[-1] == 2:
                        # We have a tuple action [action_type, action_size]
                        action_np = tuple(action.cpu().numpy().squeeze().astype(int))
                    else:
                        # Single discrete action
                        action_np = action.cpu().numpy().item()

                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action_np)
                episode_done = terminated or truncated

                # Process next state
                next_state_dict = preprocess_state_to_dict(next_state, self.model_config)

                # Add to buffer
                self.buffer.add(
                    state_dict,
                    action,
                    reward,
                    next_state_dict,
                    episode_done,
                    action_info
                )

                # Update state
                state_dict = next_state_dict

                # Update counters
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                # Call on_step for callbacks
                for callback in self.callbacks:
                    callback.on_step(self, state_dict, action, reward, next_state_dict, info)

            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_reward += episode_reward

            # Log episode results
            self.logger.info(f"Episode {self.total_episodes + 1}: reward={episode_reward:.2f}, length={episode_length}")

            # Increment episode counter
            self.total_episodes += 1

            # Call on_episode_end for callbacks
            for callback in self.callbacks:
                callback.on_episode_end(self, episode_reward, episode_length, info)

        # Update global counters
        self.total_steps += total_steps

        # Call on_rollout_end for callbacks
        for callback in self.callbacks:
            callback.on_rollout_end(self)

        # Return statistics
        stats = {
            "episodes": n_episodes,
            "total_steps": total_steps,
            "mean_reward": total_reward / max(1, n_episodes),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        return stats

    # Simplified compute_advantages method for agent/ppo_agent.py

    def compute_advantages(self) -> None:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        """
        # Make sure buffer data is prepared
        if isinstance(self.buffer.rewards, list):
            self.buffer.prepare_data()

        # Get data from buffer
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        # Log shapes for debugging
        self.logger.info(
            f"Computing advantages - rewards: {rewards.shape}, values: {values.shape}, dones: {dones.shape}")

        # Ensure consistent shapes - reshape if needed
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)  # Convert [batch_size] to [batch_size, 1]
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(1)  # Convert [batch_size] to [batch_size, 1]
        if len(values.shape) == 1:
            values = values.unsqueeze(1)  # Convert [batch_size] to [batch_size, 1]

        # Convert to NumPy arrays for easier processing
        rewards_np = rewards.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy()

        # Create NumPy array for advantages
        advantages_np = np.zeros_like(rewards_np)

        # Log after reshaping
        self.logger.info(
            f"After reshaping - rewards: {rewards_np.shape}, values: {values_np.shape}, dones: {dones_np.shape}")

        # Compute GAE
        last_gae = 0
        for t in reversed(range(len(rewards_np))):
            # End of episode or end of buffer
            if t == len(rewards_np) - 1:
                # For the last step, use 0 as the next value
                next_value = 0
                next_non_terminal = 1.0 - float(dones_np[t, 0])
            else:
                next_value = values_np[t + 1, 0]
                next_non_terminal = 1.0 - float(dones_np[t, 0])

            # TD error
            delta = rewards_np[t, 0] + self.gamma * next_value * next_non_terminal - values_np[t, 0]

            # Recursive GAE formula
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

            # Store advantage in NumPy array
            advantages_np[t, 0] = last_gae

        # Convert back to PyTorch tensors
        advantages = torch.FloatTensor(advantages_np).to(self.device)
        returns = advantages + values

        # Store in buffer
        self.buffer.advantages = advantages
        self.buffer.returns = returns

        # Log output shapes
        self.logger.info(f"Computed - advantages: {advantages.shape}, returns: {returns.shape}")

    def update_policy(self) -> Dict[str, float]:
        """
        Update the policy using PPO with a safer implementation that avoids
        backward graph issues.

        Returns:
            Dictionary with training metrics
        """
        # Compute advantages and returns
        self.compute_advantages()

        # Get data from buffer and convert to CPU numpy arrays to reset any computation history
        states_np = {k: v.detach().cpu().numpy() for k, v in self.buffer.states.items()}
        actions_np = self.buffer.actions.detach().cpu().numpy()
        old_log_probs_np = self.buffer.log_probs.detach().cpu().numpy()
        advantages_np = self.buffer.advantages.detach().cpu().numpy()
        returns_np = self.buffer.returns.detach().cpu().numpy()

        # Get total buffer size
        buffer_size = advantages_np.shape[0]

        # Log buffer sizes for debugging
        self.logger.info(f"Buffer sizes - buffer: {buffer_size}, advantages: {advantages_np.shape}")

        # Tracking metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        n_updates = 0

        # Main training loop
        for epoch in range(self.n_epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(buffer_size)

            # Process in batches
            for start_idx in range(0, buffer_size, self.batch_size):
                # Get batch indices
                end_idx = min(start_idx + self.batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]

                if len(batch_indices) == 0:
                    continue

                # Explicitly convert numpy arrays back to fresh PyTorch tensors
                # This ensures no gradient history is carried over
                batch_states = {k: torch.FloatTensor(v[batch_indices]).to(self.device)
                                for k, v in states_np.items()}
                batch_actions = torch.FloatTensor(actions_np[batch_indices]).to(self.device)
                batch_old_log_probs = torch.FloatTensor(old_log_probs_np[batch_indices]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages_np[batch_indices]).to(self.device)
                batch_returns = torch.FloatTensor(returns_np[batch_indices]).to(self.device)

                # Ensure consistent shapes
                if len(batch_actions.shape) == 1:
                    batch_actions = batch_actions.unsqueeze(1)
                if len(batch_old_log_probs.shape) == 1:
                    batch_old_log_probs = batch_old_log_probs.unsqueeze(1)
                if len(batch_advantages.shape) == 1:
                    batch_advantages = batch_advantages.unsqueeze(1)
                if len(batch_returns.shape) == 1:
                    batch_returns = batch_returns.unsqueeze(1)

                # Normalize advantages (just for this batch)
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # STEP 1: UPDATE ACTOR (POLICY)
                # =============================
                # Zero gradients for actor update
                self.optimizer.zero_grad()

                # Get action distribution parameters and values
                action_params, values = self.model(batch_states)

                # Compute log probabilities of actions
                if self.model.continuous_action:
                    mean, log_std = action_params
                    std = torch.exp(log_std)
                    normal_dist = torch.distributions.Normal(mean, std)

                    # Handle squashed actions (tanh)
                    action_unsquashed = torch.atanh(torch.clamp(batch_actions, -0.999, 0.999))
                    log_probs = normal_dist.log_prob(action_unsquashed).sum(1, keepdim=True)
                    entropy = normal_dist.entropy().sum(1, keepdim=True)
                else:
                    logits = action_params
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(batch_actions.squeeze()).unsqueeze(1)
                    entropy = dist.entropy().unsqueeze(1)

                # Calculate PPO ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages

                # Actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Entropy loss (encouraged exploration)
                entropy_loss = -entropy.mean()

                # STEP 2: UPDATE CRITIC (VALUE FUNCTION)
                # ======================================
                # Compute critic loss
                critic_loss = F.mse_loss(values, batch_returns)

                # STEP 3: COMBINED LOSS
                # ====================
                # Combine losses with coefficients
                loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

                # Backward pass and optimize
                loss.backward()

                # Optional gradient clipping
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update model parameters
                self.optimizer.step()

                # Compute metrics with no gradients
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = ((batch_old_log_probs - log_probs) ** 2).mean().item()

                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                # Update tracking metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_loss.item()
                total_kl += approx_kl
                total_clip_fraction += clip_fraction
                n_updates += 1

        # Compute average metrics
        metrics = {
            "actor_loss": total_actor_loss / max(1, n_updates),
            "critic_loss": total_critic_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": total_kl / max(1, n_updates),
            "clip_fraction": total_clip_fraction / max(1, n_updates),
            "loss": (total_actor_loss + self.critic_coef * total_critic_loss +
                     self.entropy_coef * total_entropy) / max(1, n_updates)
        }

        # Clear buffer after update
        self.buffer.clear()

        # Call callbacks
        for callback in self.callbacks:
            callback.on_update_end(self, metrics)

        return metrics

    def train(self, total_updates: int, eval_freq: int = 5) -> Dict[str, Any]:
        """
        Train the model for a specified number of updates.

        Args:
            total_updates: Total number of policy updates to perform
            eval_freq: Frequency of evaluation (in updates)

        Returns:
            Dictionary with final training statistics
        """
        self.logger.info(f"Starting training for {total_updates} updates")

        # Call on_training_start for callbacks
        for callback in self.callbacks:
            callback.on_training_start(self)

        # Track best model
        best_mean_reward = -float('inf')
        best_model_path = None

        # Training loop
        start_time = time.time()
        for update in range(1, total_updates + 1):
            self.logger.info(f"Update {update}/{total_updates}")

            # Collect rollouts
            rollout_stats = self.collect_rollouts()

            # Log rollout statistics
            for k, v in rollout_stats.items():
                if k != "episode_rewards" and k != "episode_lengths":
                    wandb.log({f"rollout/{k}":v},step=self.updates)

            # Update policy
            update_metrics = self.update_policy()
            self.updates += 1

            # Log update metrics
            for k, v in update_metrics.items():
                wandb.log({f"train/{k}":v},step=self.updates)

            # Log mean episode reward
            mean_reward = rollout_stats["mean_reward"]
            wandb.log({"rollout/mean_reward": mean_reward}, step=self.updates)

            # Save model
            if update % eval_freq == 0:
                # Evaluate model
                eval_stats = self.evaluate(10)
                eval_mean_reward = eval_stats["mean_reward"]

                # Log evaluation statistics
                for k, v in eval_stats.items():
                    if k != "episode_rewards" and k != "episode_lengths":
                        wandb.log({f"eval/{k}": v},step=self.updates)

                # Save model if it's the best so far
                if eval_mean_reward > best_mean_reward:
                    best_mean_reward = eval_mean_reward
                    best_model_path = os.path.join(self.model_dir, f"best_model_{self.updates}.pt")
                    self.save_model(best_model_path)
                    self.logger.info(f"New best model with mean reward: {best_mean_reward:.4f}")

                # Also save latest model
                latest_model_path = os.path.join(self.model_dir, "latest_model.pt")
                self.save_model(latest_model_path)

            # Log elapsed time
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time: {elapsed_time:.2f}s")

            # Call on_update_iteration_end for callbacks
            for callback in self.callbacks:
                callback.on_update_iteration_end(self, update, update_metrics, rollout_stats)

        # Call on_training_end for callbacks
        final_stats = {
            "total_updates": self.updates,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "best_mean_reward": best_mean_reward,
            "best_model_path": best_model_path,
            "elapsed_time": time.time() - start_time
        }
        for callback in self.callbacks:
            callback.on_training_end(self, final_stats)

        self.logger.info(f"Training completed. Total updates: {self.updates}, episodes: {self.total_episodes}")
        self.logger.info(f"Best mean reward: {best_mean_reward:.4f}")

        return final_stats

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the current policy.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with evaluation statistics
        """
        self.logger.info(f"Evaluating for {n_episodes} episodes...")

        # Set model to evaluation mode
        self.model.eval()

        total_reward = 0.0
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            # Reset environment
            state, _ = self.env.reset()
            state_dict = preprocess_state_to_dict(state, self.model_config)

            # Track episode data
            episode_reward = 0
            episode_length = 0
            episode_done = False

            while not episode_done:
                # Get action from policy (deterministic)
                with torch.no_grad():
                    action, _ = self.model.get_action(state_dict, deterministic=True)

                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                episode_done = terminated or truncated

                # Process next state
                next_state_dict = preprocess_state_to_dict(next_state, self.model_config)

                # Update state
                state_dict = next_state_dict

                # Update counters
                episode_reward += reward
                episode_length += 1

            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_reward += episode_reward

            self.logger.info(f"Eval episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")

        # Set model back to training mode
        self.model.train()

        # Return statistics
        stats = {
            "episodes": n_episodes,
            "mean_reward": total_reward / n_episodes,
            "mean_length": sum(episode_lengths) / n_episodes,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        return stats

    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "updates": self.updates,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }, path)

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a model from a file.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.updates = checkpoint.get("updates", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        self.logger.info(f"Model loaded from {path}")