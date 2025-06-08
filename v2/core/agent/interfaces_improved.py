"""
Simplified Agent interface - Single unified interface without factories.

Key improvements:
- Single IAgent interface that handles everything
- Clear model state management
- Batch processing support
- No factory methods
"""

from abc import abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass

from ..types.common import (
    ActionArray, ObservationArray, ProbabilityArray, 
    ModelVersion, EpisodeMetrics, Configurable, Serializable
)


@dataclass
class Experience:
    """Single step experience for RL training."""
    observation: ObservationArray
    action: ActionArray
    reward: float
    next_observation: ObservationArray
    done: bool
    info: dict[str, Any]
    
    # Algorithm-specific fields (optional)
    value: Optional[float] = None
    log_prob: Optional[float] = None
    advantage: Optional[float] = None


@dataclass
class ExperienceBatch:
    """Batch of experiences for efficient training."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    
    # Algorithm-specific batched data
    values: Optional[np.ndarray] = None
    log_probs: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None


@runtime_checkable
class IAgent(Protocol, Configurable, Serializable):
    """Unified agent interface for RL trading agents.
    
    Combines policy execution, learning, and experience collection
    in a single clean interface.
    """
    
    # === BASIC PROPERTIES ===
    @property
    def device(self) -> torch.device:
        """Current computation device."""
        ...
    
    @property
    def is_training_mode(self) -> bool:
        """Whether agent is in training mode (affects exploration)."""
        ...
    
    @property
    def algorithm_type(self) -> str:
        """Algorithm type (e.g., 'PPO', 'SAC', 'DQN')."""
        ...
    
    # === POLICY EXECUTION ===
    def predict(
        self, 
        observation: ObservationArray,
        deterministic: bool = False,
        return_extras: bool = False
    ) -> tuple[ActionArray, Optional[dict[str, Any]]]:
        """Predict action given observation.
        
        Args:
            observation: Current state
            deterministic: If True, no exploration
            return_extras: If True, return additional info (values, log_probs)
            
        Returns:
            Tuple of (action, extras if requested)
            
        Design notes:
        - Primary method for action selection
        - extras used for experience collection
        """
        ...
    
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
        """
        ...
    
    def get_action_probabilities(
        self,
        observation: ObservationArray
    ) -> ProbabilityArray:
        """Get action probability distribution."""
        ...
    
    def set_training_mode(self, training: bool) -> None:
        """Set training/evaluation mode."""
        ...
    
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
        """
        ...
    
    def batch_experiences(
        self,
        experiences: list[Experience]
    ) -> ExperienceBatch:
        """Convert experience list to efficient batch format.
        
        Args:
            experiences: List of individual experiences
            
        Returns:
            Batched experience data ready for learning
        """
        ...
    
    # === LEARNING ===
    @abstractmethod
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
            
        Design notes:
        - Core learning algorithm implementation
        - Algorithm-specific (PPO, SAC, etc.)
        - Returns metrics for monitoring
        """
        ...
    
    @abstractmethod
    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate for all optimizers."""
        ...
    
    # === MODEL STATE MANAGEMENT ===
    @abstractmethod
    def get_model_state(self) -> dict[str, Any]:
        """Get complete model state for checkpointing."""
        ...
    
    @abstractmethod
    def set_model_state(self, state: dict[str, Any]) -> None:
        """Restore model from state."""
        ...
    
    @abstractmethod
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
        ...
    
    @abstractmethod
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
        ...
    
    # === ALGORITHM-SPECIFIC METHODS (Optional) ===
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (PPO-specific).
        
        Returns:
            Tuple of (advantages, returns)
            
        Note: Only relevant for value-based algorithms like PPO
        """
        raise NotImplementedError("GAE not implemented for this algorithm")
    
    def get_value_estimates(
        self,
        observations: np.ndarray
    ) -> np.ndarray:
        """Get value function estimates for batch (PPO-specific).
        
        Note: Only relevant for actor-critic algorithms
        """
        raise NotImplementedError("Value estimates not available for this algorithm")
    
    def get_action_log_probs(
        self,
        observations: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """Get log probabilities for taken actions (PPO-specific).
        
        Note: Only relevant for policy gradient algorithms
        """
        raise NotImplementedError("Log probs not available for this algorithm")


# Simplified callback interface
class IAgentCallback(Protocol):
    """Callback interface for agent events."""
    
    def on_episode_start(self, episode: int) -> None:
        """Called at episode start."""
        ...
    
    def on_episode_end(
        self,
        episode: int,
        metrics: EpisodeMetrics
    ) -> None:
        """Called at episode end."""
        ...
    
    def on_learning_step(
        self,
        step: int,
        metrics: dict[str, float]
    ) -> None:
        """Called after learning update."""
        ...
    
    def on_model_save(
        self,
        path: Path,
        metadata: dict[str, Any]
    ) -> None:
        """Called when model is saved."""
        ...