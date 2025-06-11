"""
Enhanced context system for callbacks with comprehensive data access.

Provides rich, strongly-typed contexts for all events with lazy loading
and direct access to all training components.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import torch
import numpy as np

from .events import EventType, EventMetadata

# Avoid circular imports
if TYPE_CHECKING:
    from agent.ppo_agent import PPOTrainer
    from agent.replay_buffer import ReplayBuffer
    from envs import TradingEnvironment
    from data.data_manager import DataManager
    from simulators.market_simulator import MarketSimulator
    from simulators.portfolio_simulator import PortfolioSimulator
    from simulators.execution_simulator import ExecutionSimulator
    from training.episode_manager import EpisodeManager
    from training.training_manager import TrainingState
    from core.model_manager import ModelManager


@dataclass
class BaseContext:
    """Base context with common fields for all events."""
    event_metadata: EventMetadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core component references (available in all contexts)
    trainer: Optional['PPOTrainer'] = None
    environment: Optional['TradingEnvironment'] = None
    data_manager: Optional['DataManager'] = None
    episode_manager: Optional['EpisodeManager'] = None
    training_state: Optional['TrainingState'] = None
    model_manager: Optional['ModelManager'] = None
    
    # Simulator references for deep access
    market_simulator: Optional['MarketSimulator'] = None
    portfolio_simulator: Optional['PortfolioSimulator'] = None
    execution_simulator: Optional['ExecutionSimulator'] = None
    
    # Replay buffer for direct access
    replay_buffer: Optional['ReplayBuffer'] = None
    
    def get_model(self) -> Optional[torch.nn.Module]:
        """Get the current model."""
        return self.trainer.model if self.trainer else None
    
    def get_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Get the current optimizer."""
        return self.trainer.optimizer if self.trainer else None
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        if self.trainer and hasattr(self.trainer, 'device'):
            return self.trainer.device
        return torch.device('cpu')


@dataclass
class StepContext(BaseContext):
    """Context for step-level events."""
    step_num: int
    episode_step: int                              # Step within episode
    global_step: int                               # Global step count
    
    # State information
    observation: Dict[str, np.ndarray]             # Current observation
    previous_observation: Optional[Dict[str, np.ndarray]] = None
    
    # Action information
    action: Optional[Union[int, np.ndarray]] = None
    action_probs: Optional[np.ndarray] = None
    action_logprob: Optional[float] = None
    value_estimate: Optional[float] = None
    
    # Step results
    reward: Optional[float] = None
    next_observation: Optional[Dict[str, np.ndarray]] = None
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    
    # Market state at step
    current_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    spread: Optional[float] = None
    volume: Optional[float] = None
    
    # Portfolio state at step
    position: Optional[int] = None
    cash: Optional[float] = None
    portfolio_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    
    # Features at step (lazy loaded)
    _features: Optional[Dict[str, np.ndarray]] = None
    
    def get_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Lazy load features if available."""
        if self._features is None and self.environment:
            # Extract features from environment if available
            if hasattr(self.environment, 'get_current_features'):
                self._features = self.environment.get_current_features()
        return self._features


@dataclass
class EpisodeContext(BaseContext):
    """Context for episode-level events."""
    episode_num: int
    global_episode: int
    
    # Episode configuration
    symbol: str
    date: datetime
    reset_point_idx: int
    reset_timestamp: datetime
    
    # Episode results
    episode_reward: float = 0.0
    episode_length: int = 0
    terminated: bool = False
    truncated: bool = False
    termination_reason: Optional[str] = None
    
    # Episode timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Trading metrics
    trades: List[Dict[str, Any]] = field(default_factory=list)
    num_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    
    # Portfolio state
    starting_cash: float = 0.0
    ending_cash: float = 0.0
    starting_portfolio_value: float = 0.0
    ending_portfolio_value: float = 0.0
    max_position: int = 0
    
    # Episode trajectory (for analysis)
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    
    # Step-by-step data (lazy loaded)
    _step_data: Optional[List[StepContext]] = None
    
    def get_step_data(self) -> Optional[List[StepContext]]:
        """Get detailed step data if available."""
        return self._step_data


@dataclass
class RolloutContext(BaseContext):
    """Context for rollout collection events."""
    rollout_num: int
    num_steps_requested: int
    num_steps_collected: int = 0
    num_episodes_completed: int = 0
    
    # Rollout timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Rollout statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    total_reward: float = 0.0
    
    # Buffer state
    buffer_size_before: int = 0
    buffer_size_after: int = 0
    buffer_ready: bool = False
    
    # Episodes in rollout
    episodes: List[EpisodeContext] = field(default_factory=list)
    
    # Performance metrics
    steps_per_second: float = 0.0
    episodes_per_second: float = 0.0


@dataclass
class UpdateContext(BaseContext):
    """Context for policy update events."""
    update_num: int
    global_update: int
    
    # Update configuration
    batch_size: int
    num_epochs: int
    learning_rate: float
    clip_epsilon: float
    
    # Losses
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    total_loss: float = 0.0
    
    # Gradients
    gradient_norm: float = 0.0
    gradient_max: float = 0.0
    clipped_gradients: bool = False
    
    # PPO specific metrics
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    entropy: float = 0.0
    
    # Advantages
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_max: float = 0.0
    advantage_min: float = 0.0
    
    # Update timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Batch/Epoch details
    current_epoch: int = 0
    current_batch: int = 0
    total_batches: int = 0
    
    # Model state snapshots (before/after)
    model_state_before: Optional[Dict[str, Any]] = None
    model_state_after: Optional[Dict[str, Any]] = None
    
    # Detailed gradient information (lazy loaded)
    _gradient_info: Optional[Dict[str, Any]] = None
    
    def get_gradient_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed gradient information if available."""
        if self._gradient_info is None and self.trainer:
            # Extract gradient information
            self._gradient_info = self._compute_gradient_info()
        return self._gradient_info
    
    def _compute_gradient_info(self) -> Dict[str, Any]:
        """Compute detailed gradient statistics."""
        if not self.trainer or not self.get_model():
            return {}
        
        grad_info = {
            'layer_gradients': {},
            'parameter_updates': {},
            'gradient_histogram': {}
        }
        
        # Analyze gradients by layer
        for name, param in self.get_model().named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_info['layer_gradients'][name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item()
                }
        
        return grad_info


@dataclass
class BatchContext(BaseContext):
    """Context for batch processing events."""
    batch_idx: int
    epoch_idx: int
    update_num: int
    
    # Batch data
    batch_size: int
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_logprobs: torch.Tensor
    
    # Batch results
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    
    # Predictions
    action_logprobs: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    entropy: Optional[float] = None


@dataclass
class ModelContext(BaseContext):
    """Context for model-related events."""
    model_version: int
    checkpoint_path: Optional[Path] = None
    
    # Model metrics
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Performance metrics
    best_reward: float = 0.0
    current_reward: float = 0.0
    improvement: float = 0.0
    
    # Checkpoint info
    saved_at_step: int = 0
    saved_at_episode: int = 0
    saved_at_update: int = 0
    
    # Model state
    optimizer_state: Optional[Dict[str, Any]] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationContext(BaseContext):
    """Context for evaluation events."""
    eval_num: int
    num_episodes: int
    
    # Evaluation results
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    
    # Evaluation episodes
    episodes: List[EpisodeContext] = field(default_factory=list)
    
    # Aggregated metrics
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None


@dataclass
class DataContext(BaseContext):
    """Context for data-related events."""
    symbol: str
    date: datetime
    
    # Data info
    data_points: int = 0
    time_range_start: datetime = None
    time_range_end: datetime = None
    
    # Data quality
    missing_data_points: int = 0
    data_quality_score: float = 1.0
    
    # Market info
    momentum_score: Optional[float] = None
    volatility: Optional[float] = None
    volume_profile: Optional[Dict[str, float]] = None


@dataclass
class ErrorContext(BaseContext):
    """Context for error events."""
    error_type: str
    error_message: str
    error_traceback: Optional[str] = None
    
    # Error location
    component: str
    method: Optional[str] = None
    
    # Recovery info
    recoverable: bool = False
    recovery_action: Optional[str] = None
    
    # State at error
    step: int = 0
    episode: int = 0
    update: int = 0


@dataclass
class CustomContext(BaseContext):
    """Context for custom events."""
    event_name: str
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Allow arbitrary data storage
    def __setitem__(self, key: str, value: Any):
        self.event_data[key] = value
    
    def __getitem__(self, key: str) -> Any:
        return self.event_data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.event_data.get(key, default)


# Context type mapping
CONTEXT_MAPPING = {
    EventType.TRAINING_START: BaseContext,
    EventType.TRAINING_END: BaseContext,
    EventType.TRAINING_ERROR: ErrorContext,
    
    EventType.EPISODE_START: EpisodeContext,
    EventType.EPISODE_END: EpisodeContext,
    EventType.EPISODE_RESET: EpisodeContext,
    EventType.EPISODE_TERMINATED: EpisodeContext,
    EventType.EPISODE_TRUNCATED: EpisodeContext,
    
    EventType.STEP_START: StepContext,
    EventType.STEP_END: StepContext,
    EventType.ACTION_SELECTED: StepContext,
    EventType.REWARD_COMPUTED: StepContext,
    
    EventType.ROLLOUT_START: RolloutContext,
    EventType.ROLLOUT_END: RolloutContext,
    EventType.BUFFER_ADD: BaseContext,
    EventType.BUFFER_READY: BaseContext,
    
    EventType.UPDATE_START: UpdateContext,
    EventType.UPDATE_END: UpdateContext,
    EventType.GRADIENT_COMPUTED: UpdateContext,
    EventType.OPTIMIZER_STEP: UpdateContext,
    EventType.BATCH_START: BatchContext,
    EventType.BATCH_END: BatchContext,
    EventType.EPOCH_START: UpdateContext,
    EventType.EPOCH_END: UpdateContext,
    
    EventType.MODEL_SAVED: ModelContext,
    EventType.MODEL_LOADED: ModelContext,
    EventType.MODEL_IMPROVED: ModelContext,
    EventType.LEARNING_RATE_UPDATED: BaseContext,
    
    EventType.EVALUATION_START: EvaluationContext,
    EventType.EVALUATION_END: EvaluationContext,
    EventType.EVALUATION_EPISODE: EpisodeContext,
    
    EventType.DATA_LOADED: DataContext,
    EventType.DAY_SWITCHED: DataContext,
    EventType.SYMBOL_SWITCHED: DataContext,
    
    EventType.MEMORY_WARNING: BaseContext,
    EventType.PERFORMANCE_LOG: BaseContext,
    
    EventType.CUSTOM: CustomContext,
}


def get_context_class(event_type: EventType) -> type:
    """Get the appropriate context class for an event type."""
    return CONTEXT_MAPPING.get(event_type, BaseContext)