import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import torch.nn.functional as nnf

from agent.replay_buffer import ReplayBuffer
from config import TrainingConfig
from model.transformer import MultiBranchTransformer
from core.types import RolloutResult, UpdateResult




class PPOTrainer:
    """
    Responsibilities:
    - PPO algorithm implementation (rollout collection and policy updates)
    - Model training and optimization
    - Buffer management
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: MultiBranchTransformer,
        device: Optional[torch.device] = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.PPOTrainer")
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")

        # PPO hyperparameters
        self.lr = config.learning_rate
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_eps = config.clip_epsilon
        self.critic_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.ppo_epochs = config.n_epochs
        self.batch_size = config.batch_size
        self.rollout_steps = config.training_manager.rollout_steps

        # Optimizer and buffer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(capacity=self.rollout_steps, device=self.device)


        self.logger.info(f"🤖 V2 PPOTrainer initialized. Device: {self.device}")

    def collect_rollout(self, environment, num_steps: Optional[int] = None) -> RolloutResult:
        """
        Collect rollout data from environment.
        
        Args:
            environment: Environment to collect data from  
            num_steps: Number of steps to collect (defaults to config.rollout_steps)
            
        Returns:
            RolloutResult with rollout statistics and buffer readiness
        """
        if num_steps is None:
            num_steps = self.rollout_steps
        self.buffer.clear()
        
        # Get initial state from environment (TrainingManager should have set this up)
        current_state = environment.get_current_state()
        if current_state is None:
            return RolloutResult(
                steps_collected=0,
                episodes_completed=0,
                buffer_ready=False,
                interrupted=True
            )
        
        collected_steps = 0
        episodes_completed = 0
        
        while num_steps is not None and collected_steps < num_steps:
            # Convert state to tensors for a model
            state_tensors = self._convert_state_to_tensors(current_state)
            
            # Get action from a model
            with torch.no_grad():
                action_tensor, action_info = self.model.get_action(
                    state_tensors, deterministic=False
                )
            
            # Convert action for environment
            env_action = self._convert_action_for_env(action_tensor)
            
            # Step environment
            try:
                next_state, reward, terminated, truncated, info = environment.step(env_action)
                done = terminated or truncated
                
            except Exception as e:
                self.logger.error(f"Error during environment step: {e}")
                break
            
            # Store experience in buffer
            self.buffer.add(
                current_state,
                action_tensor,
                reward,
                next_state,
                done,
                action_info,
            )
            
            # Update state and counters
            current_state = next_state
            collected_steps += 1
            
            # Handle episode completion
            if done:
                episodes_completed += 1
                
                # Get next episode state from environment 
                # (TrainingManager handles episode advancement)
                current_state = environment.get_current_state()
                if current_state is None:
                    break
        
        # Prepare buffer for training
        self.buffer.prepare_data_for_training()
        
        return RolloutResult(
            steps_collected=collected_steps,
            episodes_completed=episodes_completed,
            buffer_ready=self.buffer.get_size() >= self.batch_size,
        )

    def update_policy(self) -> UpdateResult:
        """
        Perform PPO policy update using collected rollout data.
        
        Returns:
            UpdateResult with loss metrics and update status
        """
        if not self.buffer.is_ready_for_training():
            return UpdateResult(
                policy_loss=0.0,
                value_loss=0.0,
                entropy_loss=0.0,
                total_loss=0.0,
                interrupted=True
            )
        
        # Compute advantages and returns
        self._compute_advantages_and_returns()
        
        training_data = self.buffer.get_training_data()
        if training_data is None:
            return UpdateResult(
                policy_loss=0.0,
                value_loss=0.0,
                entropy_loss=0.0,
                total_loss=0.0,
                interrupted=True
            )
        
        states_dict = training_data["states"]
        actions = training_data["actions"]
        old_log_probs = training_data["old_log_probs"]
        advantages = training_data["advantages"]
        returns = training_data["returns"]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        num_samples = actions.size(0)
        if num_samples == 0:
            return UpdateResult(
                policy_loss=0.0,
                value_loss=0.0,
                entropy_loss=0.0,
                total_loss=0.0,
                interrupted=True
            )
        
        indices = np.arange(num_samples)
        actor_loss = critic_loss = entropy_loss = 0.0
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
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
                
        
        # Return last computed losses (already calculated for training)
        return UpdateResult(
            policy_loss=actor_loss.item(),
            value_loss=critic_loss.item(),
            entropy_loss=entropy_loss.item(),
            total_loss=(actor_loss + critic_loss + entropy_loss).item(),
        )


    def _convert_state_to_tensors(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert state dictionary to tensors for a model."""
        state_tensors = {}
        for key, np_array in state_dict.items():
            # Convert to tensor and add batch dimension if needed
            tensor = torch.as_tensor(np_array, dtype=torch.float32).to(self.device)
            
            # Add batch dimension for a model
            if key in ["hf", "mf", "lf", "portfolio"]:
                if tensor.ndim == 2:  # [seq_len, feat_dim] -> [1, seq_len, feat_dim]
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim == 3 and tensor.shape[0] != 1:
                    # Assume first dim is batch, reshape if needed
                    pass
            
            state_tensors[key] = tensor
        
        return state_tensors

    @staticmethod
    def _convert_action_for_env(action_tensor: torch.Tensor) -> Any:
        """Convert model action tensor to environment format."""
        if action_tensor.ndim > 0 and action_tensor.shape[-1] == 2:
            return action_tensor.cpu().numpy().squeeze().astype(int)
        else:
            return action_tensor.cpu().numpy().item()

    def _compute_advantages_and_returns(self):
        """Compute GAE advantages and returns."""
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

    def evaluate(self, environment, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate model performance over multiple episodes.
        
        Args:
            environment: TradingEnvironment instance (v2 compatible)
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"🔍 Starting evaluation: {n_episodes} episodes (deterministic={deterministic})")
        
        # Set model to evaluation mode
        was_training = self.model.training
        self.model.eval()
        
        episode_rewards = []
        episode_lengths = []
        episode_details = []
        
        try:
            for episode_idx in range(n_episodes):
                self.logger.debug(f"Evaluation episode {episode_idx + 1}/{n_episodes}")
                
                # Get initial state from environment (should already be setup by caller)
                current_state = environment.get_current_state()
                if current_state is None:
                    self.logger.warning(f"No initial state available for evaluation episode {episode_idx + 1}")
                    break
                
                episode_reward = 0.0
                episode_length = 0
                done = False
                info = {}  # Initialize info to handle edge cases
                
                while not done:
                    # Convert state to tensors for a model
                    state_tensors = self._convert_state_to_tensors(current_state)
                    
                    # Get action from a model (deterministic or stochastic)
                    with torch.no_grad():
                        action_tensor, action_info = self.model.get_action(
                            state_tensors, deterministic=deterministic
                        )
                    
                    # Convert action for environment
                    env_action = self._convert_action_for_env(action_tensor)
                    
                    # Take an environment step
                    try:
                        next_state, reward, terminated, truncated, info = environment.step(env_action)
                        done = terminated or truncated
                        
                    except Exception as e:
                        self.logger.error(f"Error during evaluation step: {e}")
                        # Graceful degradation - end episode on error
                        next_state = current_state  # Keep the current state
                        done = True
                        reward = 0.0
                        info = {}
                    
                    # Update episode tracking
                    current_state = next_state
                    episode_reward += reward
                    episode_length += 1
                    
                    # Safety check for runaway episodes
                    if episode_length > 10000:  # Configurable limit
                        self.logger.warning(f"Episode {episode_idx + 1} exceeded length limit, terminating")
                        done = True
                
                # Store episode results
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_details.append({
                    "episode": episode_idx + 1,
                    "reward": episode_reward,
                    "length": episode_length,
                    "final_info": info
                })
                
                self.logger.debug(
                    f"Episode {episode_idx + 1} complete: reward={episode_reward:.3f}, length={episode_length}"
                )
        
        finally:
            # Restore model training mode
            if was_training:
                self.model.train()
        
        # Calculate evaluation metrics
        if episode_rewards:
            eval_results = {
                "n_episodes": len(episode_rewards),
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "min_reward": float(np.min(episode_rewards)),
                "max_reward": float(np.max(episode_rewards)),
                "mean_length": float(np.mean(episode_lengths)),
                "std_length": float(np.std(episode_lengths)),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "episode_details": episode_details
            }
        else:
            # No episodes completed
            eval_results = {
                "n_episodes": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
                "episode_rewards": [],
                "episode_lengths": [],
                "episode_details": []
            }
        
        # Log comprehensive evaluation summary
        self.logger.info("🔍 EVALUATION COMPLETE:")
        self.logger.info(f"   📊 Episodes: {eval_results['n_episodes']}")
        self.logger.info(f"   💰 Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        self.logger.info(f"   📈 Range: [{eval_results['min_reward']:.3f}, {eval_results['max_reward']:.3f}]")
        self.logger.info(f"   📏 Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
        
        return eval_results
