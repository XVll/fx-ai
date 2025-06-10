"""
Training configuration for PPO and training management.
"""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field

class TrainingManagerConfig(BaseModel):
    """Training Manager - Central authority for training lifecycle"""

    # Core mode selection
    mode: str = Field("training", description="Training mode: training, optuna, benchmark")

    # Training Termination (always enforced) - for ending entire training
    termination_max_episodes: Optional[int] = Field(None, description="Maximum total episodes before training termination")
    termination_max_updates: Optional[int] = Field(None, description="Maximum total updates before training termination")
    termination_max_cycles: Optional[int] = Field(None, description="Maximum total data cycles before training termination")

    intelligent_termination: bool = Field(True, description="Enable intelligent termination")
    plateau_patience: Optional[int] = Field(50, description="Updates without improvement before plateau termination (null to disable)")
    degradation_threshold: Optional[float] = Field(0.05, description="Performance degradation threshold (5%) (null to disable)")

    # Episode limits (When to reset environment and start new episode)
    episode_max_steps: Optional[int] = Field(None, description="Max steps per episode")

    # Daily limits (when to switch to next day)
    daily_max_episodes: Optional[int] = Field(None, description="Max episodes per day before switching to next day")
    daily_max_updates: Optional[int] = Field(None, description="Max updates per day before switching to next day")
    daily_max_cycles: Optional[int] = Field(3, description="How many times to cycle through ALL reset points before switching to next day")

    # Date, symbol and reset point selection
    symbols: List[str] = Field(default_factory=lambda: ["MLGO"], description="Trading symbols")
    date_range: List[Optional[str]] = Field(default_factory=lambda: [None, None], description="Date range [start, end]")

    day_score_range: List[float] = Field(default_factory=lambda: [0.7, 1.0], description="Day quality score range")
    reset_roc_range: List[float] = Field(default_factory=lambda: [0.05, 1.0], description="ROC score range")
    reset_activity_range: List[float] = Field(default_factory=lambda: [0.0, 1.0], description="Activity score range")

    day_selection_mode: Literal["sequential", "quality", "random"] = Field("sequential", description="How to order reset points: sequential (by time), quality (by combined score) or random")
    reset_point_selection_mode: Literal["sequential", "quality", "random"] = Field("sequential", description="How to order reset points: sequential (by time), quality (by combined score) or random")

    # Preloading next training data
    preload_enabled: bool = Field(True, description="Enable data preloading")

    # Evaluation configuration
    eval_frequency: Optional[int] = Field(5, description="Updates between evaluations (null to disable)")
    eval_episodes: int = Field(10, description="Episodes for evaluation")

    # Continuous training configuration
    performance_window: Optional[int] = Field(50, description="Performance history window size (null to disable)")
    recommendation_frequency: Optional[int] = Field(10, description="Episodes between recommendations (null to disable)")
    adaptation_enabled: bool = Field(True, description="Enable adaptive data difficulty")
    early_stop_patience: int = Field(300, description="Updates without improvement")
    early_stop_min_delta: float = Field(0.01, description="Minimum improvement")

    # Model management
    best_model_metric: str = Field("mean_reward", description="Model selection metric")
    continue_training: bool = Field(False, description="Continue from best model")
    keep_best_n_models: int = Field(5, description="Number of best models to keep")
    checkpoint_frequency: Optional[int] = Field(25, description="Updates between checkpoints (null to disable)")

    rollout_steps: int = Field(2048, gt=0, description="Steps per rollout")


class TrainingConfig(BaseModel):
    """PPO training configuration"""
    seed: int = Field(42, description="Random seed")
    shutdown_timeout: int = Field(10, description="Shutdown timeout in seconds")
    device: str = Field("mps", description="Training device")

    # PPO hyperparameters
    learning_rate: float = Field(1.5e-4, gt=0.0, description="Learning rate")
    batch_size: int = Field(64, gt=0, description="Batch size")
    n_epochs: int = Field(8, gt=0, description="Training epochs per update")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor")
    gae_lambda: float = Field(0.95, ge=0.0, le=1.0, description="GAE lambda")
    clip_epsilon: float = Field(0.15, gt=0.0, description="PPO clip range")
    value_coef: float = Field(0.5, ge=0.0, description="Value function coefficient")
    entropy_coef: float = Field(0.01, ge=0.0, description="Entropy coefficient")
    max_grad_norm: float = Field(0.3, gt=0.0, description="Gradient clipping")

    training_manager: Optional[TrainingManagerConfig] = Field(default_factory=TrainingManagerConfig)
