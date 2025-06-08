"""
PPO Agent implementation following the unified IAgent interface.

Implements the Proximal Policy Optimization algorithm
for trading with clean separation of concerns.
"""

from typing import Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from ...core.agent.interfaces_improved import (
    IAgent, Experience, ExperienceBatch,
    ActionArray, ObservationArray, ProbabilityArray
)


class PPOAgent(IAgent):
    """PPO agent implementation following unified IAgent interface.
    
    This class implements the full PPO algorithm with:
    - Actor-critic architecture with shared backbone
    - GAE for advantage estimation
    - Clipped surrogate objective
    - Value function clipping
    - Entropy regularization
    - Experience collection and batching
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        config: dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            observation_space: Environment observation space (Dict with hf, mf, lf, portfolio)
            action_space: Environment action space (MultiDiscrete [3, 4])
            config: Agent configuration containing:
                - learning_rate: Learning rate for optimizer (default: 3e-4)
                - gamma: Discount factor (default: 0.99)
                - gae_lambda: GAE lambda parameter (default: 0.95)
                - clip_epsilon: PPO clipping parameter (default: 0.2)
                - value_clip: Value function clipping (default: 0.2)
                - entropy_coef: Entropy regularization weight (default: 0.01)
                - value_coef: Value loss coefficient (default: 0.5)
                - max_grad_norm: Gradient clipping threshold (default: 0.5)
                - n_epochs: Training epochs per batch (default: 4)
                - batch_size: Minibatch size (default: 64)
            device: Computation device (auto-detect if None)
            
        Implementation steps:
        1. Store configuration and validate required parameters
        2. Set up device and ensure deterministic behavior
        3. Initialize multi-branch transformer network (actor-critic)
        4. Set up optimizers with learning rate scheduling
        5. Initialize experience collection buffers
        6. Set up normalization statistics tracking
        """
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Device setup
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
            
        # PPO hyperparameters
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_clip = config.get("value_clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.n_epochs = config.get("n_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        
        # Training state
        self._training_mode = True
        
        # TODO: Initialize networks
        # self.network = MultiBranchTransformer(...)
        # self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # TODO: Initialize experience collection
        # self.experience_buffer = []
        
        # TODO: Initialize normalization statistics
        # self.obs_normalizer = ...
        
        print(f"PPO Agent initialized on {self._device}")
        print(f"Config: lr={self.lr}, gamma={self.gamma}, clip={self.clip_epsilon}")
    
    # === BASIC PROPERTIES ===
    @property
    def device(self) -> torch.device:
        """Current computation device."""
        return self._device
    
    @property
    def is_training_mode(self) -> bool:
        """Whether agent is in training mode (affects exploration)."""
        return self._training_mode
    
    @property
    def algorithm_type(self) -> str:
        """Algorithm type."""
        return "PPO"
    
    # === POLICY EXECUTION ===
    def predict(
        self,
        observation: ObservationArray,
        deterministic: bool = False,
        return_extras: bool = False
    ) -> tuple[ActionArray, Optional[dict[str, Any]]]:
        """Predict action given observation.
        
        Args:
            observation: Current environment observation (dict with hf, mf, lf, portfolio)
            deterministic: If True, use mode of distribution (no exploration)
            return_extras: If True, return additional info (values, log_probs)
            
        Returns:
            Tuple of (action, extras if requested)
            
        Implementation steps:
        1. Convert observation dict to tensors and move to device
        2. Forward pass through multi-branch transformer
        3. Sample action from policy distribution (or take mode if deterministic)
        4. If return_extras, also compute value estimate and log_prob
        5. Convert action back to numpy and return
        """
        # TODO: Implement prediction
        # 1. Process observation dict (hf, mf, lf, portfolio features)
        # 2. Forward pass through network
        # 3. Sample action from policy
        # 4. Optionally return value and log_prob for training
        
        # Placeholder: return dummy action
        action = np.array([0, 0])  # [HOLD, SIZE_25]
        extras = None
        if return_extras:
            extras = {
                "value": 0.0,
                "log_prob": 0.0
            }
        return action, extras
    
    def predict_batch(
        self,
        observations: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Predict batch of actions efficiently.
        
        Args:
            observations: Batch of states [batch_size, ...]
            deterministic: If True, no exploration
            
        Returns:
            Actions [batch_size, action_dim]
            
        Implementation steps:
        1. Process batch of observations efficiently
        2. Single forward pass for entire batch
        3. Sample actions for all observations
        4. Return batch of actions
        """
        # TODO: Implement batch prediction
        batch_size = observations.shape[0]
        return np.zeros((batch_size, 2))  # Placeholder
    
    def get_action_probabilities(
        self,
        observation: ObservationArray
    ) -> ProbabilityArray:
        """Get action probability distribution.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Probability distribution over actions
            
        Implementation steps:
        1. Forward pass through policy network
        2. Apply softmax for discrete actions
        3. Handle multi-discrete action space (action_type, position_size)
        4. Return flattened probability distribution
        """
        # TODO: Implement probability calculation
        # For multi-discrete [3, 4] -> flatten to 12 actions
        return np.ones(12) / 12  # Placeholder uniform distribution
    
    def set_training_mode(self, training: bool) -> None:
        """Set training/evaluation mode.
        
        Args:
            training: True for training, False for evaluation
            
        Implementation steps:
        1. Update internal training flag
        2. Call train() or eval() on all networks
        3. Affects dropout, batch norm, and exploration
        """
        self._training_mode = training
        # TODO: Set mode on networks
        # if hasattr(self, 'network'):
        #     self.network.train(training)
    
    # === EXPERIENCE COLLECTION ===
    def collect_experience(
        self,
        observation: ObservationArray,
        action: ActionArray,
        reward: float,
        next_observation: ObservationArray,
        done: bool,
        info: dict[str, Any]
    ) -> Experience:
        """Collect single experience with algorithm-specific data.
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Episode termination
            info: Additional info
            
        Returns:
            Complete experience record with algorithm-specific fields
            
        Implementation steps:
        1. Get value estimate for current observation
        2. Get log probability of taken action
        3. Create Experience with PPO-specific fields
        """
        # TODO: Get value and log_prob from networks
        value = 0.0  # Placeholder
        log_prob = 0.0  # Placeholder
        
        return Experience(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            info=info,
            value=value,
            log_prob=log_prob,
            advantage=None  # Will be computed during GAE
        )
    
    def batch_experiences(
        self,
        experiences: list[Experience]
    ) -> ExperienceBatch:
        """Convert experience list to efficient batch format.
        
        Args:
            experiences: List of individual experiences
            
        Returns:
            Batched experience data ready for learning
            
        Implementation steps:
        1. Stack all observations, actions, rewards, etc.
        2. Compute advantages using GAE
        3. Compute returns from advantages
        4. Create ExperienceBatch with all data
        """
        if not experiences:
            raise ValueError("Cannot batch empty experience list")
        
        # Stack basic data
        observations = np.stack([exp.observation for exp in experiences])
        actions = np.stack([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_observations = np.stack([exp.next_observation for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        # Stack PPO-specific data
        values = np.array([exp.value for exp in experiences])
        log_probs = np.array([exp.log_prob for exp in experiences])
        
        # Compute GAE advantages and returns
        next_values = np.zeros_like(values)  # TODO: Compute next values
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            values=values,
            log_probs=log_probs,
            advantages=advantages,
            returns=returns
        )
    
    # === LEARNING ===
    def learn_from_batch(
        self,
        batch: ExperienceBatch,
        step: int
    ) -> dict[str, float]:
        """Update policy from experience batch.
        
        Args:
            batch: Batched experience data
            step: Current training step
            
        Returns:
            Training metrics (loss, entropy, etc.)
            
        Implementation steps:
        1. For each epoch:
           a. Shuffle batch data
           b. Create minibatches
           c. For each minibatch:
              - Forward pass through networks
              - Compute PPO clipped surrogate loss
              - Compute value function loss with clipping
              - Add entropy regularization
              - Backward pass and optimizer step
        2. Track metrics: policy_loss, value_loss, entropy, kl_div, clip_fraction
        """
        # TODO: Implement PPO learning algorithm
        # Placeholder metrics
        return {
            "policy_loss": 0.1,
            "value_loss": 0.05,
            "entropy_loss": 0.01,
            "total_loss": 0.16,
            "kl_divergence": 0.001,
            "clip_fraction": 0.3
        }
    
    # === ALGORITHM-SPECIFIC METHODS ===
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward sequence (T,)
            values: Value estimates (T,)
            dones: Episode termination flags (T,)
            next_values: Bootstrap values (T,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
            
        Implementation steps:
        1. Compute TD errors: delta = r + gamma * V(s') * (1-done) - V(s)
        2. Compute GAE advantages: A = sum(gamma * lambda)^k * delta_{t+k}
        3. Compute returns: R = A + V(s)
        4. Normalize advantages for stability
        """
        # Use config values if not provided
        gamma = gamma or self.gamma
        gae_lambda = gae_lambda or self.gae_lambda
        
        # TODO: Implement proper GAE calculation
        # For now, return placeholder values
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        return advantages, returns
    
    def get_value_estimates(
        self,
        observations: np.ndarray
    ) -> np.ndarray:
        """Get value function estimates for batch.
        
        Args:
            observations: Batch of observations
            
        Returns:
            Value estimates for each observation
        """
        # TODO: Implement value estimation
        batch_size = observations.shape[0]
        return np.zeros(batch_size)
    
    def get_action_log_probs(
        self,
        observations: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """Get log probabilities for taken actions.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions
            
        Returns:
            Log probabilities for each action
        """
        # TODO: Implement log probability calculation
        batch_size = observations.shape[0]
        return np.zeros(batch_size)
    
    # === MODEL STATE MANAGEMENT ===
    def get_model_state(self) -> dict[str, Any]:
        """Get complete model state for checkpointing.
        
        Returns:
            Dictionary containing:
            - model_state_dict: Network weights
            - optimizer_state_dict: Optimizer state
            - config: Agent configuration
            - training_state: Additional training info
        """
        # TODO: Implement state collection
        return {
            "model_state_dict": {},  # Placeholder
            "optimizer_state_dict": {},
            "config": self.config,
            "training_state": {
                "training_mode": self._training_mode
            }
        }
    
    def set_model_state(self, state: dict[str, Any]) -> None:
        """Restore model from state.
        
        Args:
            state: State dictionary from get_model_state()
        """
        # TODO: Implement state restoration
        # self.network.load_state_dict(state["model_state_dict"])
        # self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._training_mode = state.get("training_state", {}).get("training_mode", True)
    
    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate for all optimizers.
        
        Args:
            lr: New learning rate
        """
        self.lr = lr
        # TODO: Update optimizer learning rates
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr
    
    def save(
        self,
        path: Path,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Save agent state to disk.
        
        Args:
            path: Directory to save to
            metadata: Additional metadata to save
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Get complete state
        state = self.get_model_state()
        if metadata:
            state["metadata"] = metadata
        
        # TODO: Save state to files
        # torch.save(state, path / "agent_state.pth")
        # json.dump(state["config"], open(path / "config.json", "w"))
        
        print(f"Agent saved to {path}")
    
    def load(
        self,
        path: Path,
        load_optimizer: bool = True
    ) -> dict[str, Any]:
        """Load agent state from disk.
        
        Args:
            path: Directory to load from
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Metadata dictionary
        """
        # TODO: Load state from files
        # state = torch.load(path / "agent_state.pth")
        # if not load_optimizer:
        #     state.pop("optimizer_state_dict", None)
        # self.set_model_state(state)
        
        print(f"Agent loaded from {path}")
        return {"metadata": {}}  # Placeholder
    
    # === CONFIGURABLE INTERFACE ===
    def get_config(self) -> dict[str, Any]:
        """Return current configuration."""
        return self.config.copy()
    
    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration dynamically."""
        self.config.update(config)
        # Update relevant hyperparameters
        self.lr = config.get("learning_rate", self.lr)
        self.gamma = config.get("gamma", self.gamma)
        self.gae_lambda = config.get("gae_lambda", self.gae_lambda)
        # ... update other parameters as needed
    
    # === SERIALIZABLE INTERFACE ===
    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return self.get_model_state()
    
    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from dictionary."""
        self.set_model_state(data)
