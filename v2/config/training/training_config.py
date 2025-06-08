"""
Training configuration for PPO and training management.
"""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field



class TerminationConfig(BaseModel):
    """Training termination configuration"""

    # Hard limits (always enforced) - for ending entire training
    training_max_episodes: Optional[int] = Field(None, description="Maximum total episodes before training termination")
    training_max_updates: Optional[int] = Field(None, description="Maximum total updates before training termination")
    training_max_cycles: Optional[int] = Field(None, description="Maximum total data cycles before training termination")

    # Intelligent termination (production mode only)
    intelligent_termination: bool = Field(True, description="Enable intelligent termination")
    plateau_patience: Optional[int] = Field(50, description="Updates without improvement before plateau termination (null to disable)")
    degradation_threshold: Optional[float] = Field(0.05, description="Performance degradation threshold (5%) (null to disable)")


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    frequency: Optional[int] = Field(50, description="Updates between evaluations (null to disable)")
    episodes: int = Field(10, description="Episodes per evaluation")


class ContinuousTrainingConfig(BaseModel):
    """Continuous training advisor and model management configuration"""

    # Performance analysis
    performance_window: Optional[int] = Field(50, description="Performance history window size (null to disable)")
    recommendation_frequency: Optional[int] = Field(10, description="Episodes between recommendations (null to disable)")

    # Model management
    checkpoint_frequency: Optional[int] = Field(25, description="Updates between checkpoints (null to disable)")

    # Evaluation settings (centralized)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Data difficulty adaptation
    adaptation_enabled: bool = Field(True, description="Enable adaptive data difficulty")


class DataCycleConfig(BaseModel):
    """Data cycle management - when to switch days/reset points"""

    # Day switching conditions (when ANY is met, switch to next day)
    day_max_episodes: Optional[int] = Field(None, description="Max episodes per day before switching to next day")
    day_max_updates: Optional[int] = Field(None, description="Max updates per day before switching to next day")
    day_max_cycles: Optional[int] = Field(3, description="How many times to cycle through ALL reset points before switching to next day")


class ResetPointConfig(BaseModel):
    """Reset point management configuration"""
    selection_mode: Literal["sequential", "random"] = Field("sequential", description="How to order reset points: sequential (by quality) or random")


class DaySelectionConfig(BaseModel):
    """Day selection configuration"""
    selection_mode: Literal["sequential", "random", "quality_weighted", "curriculum_ordered"] = Field("quality_weighted", description="Selection mode")
    randomize_order: bool = Field(False, description="Randomize selection order")


class PreloadingConfig(BaseModel):
    """Data preloading configuration"""
    preload_enabled: bool = Field(True, description="Enable data preloading")


class DataLifecycleConfig(BaseModel):
    """Data lifecycle management configuration"""


class AdaptiveDataConfig(BaseModel):
    """Adaptive data selection configuration"""

    # Static constraints (never change)
    symbols: List[str] = Field(default_factory=lambda: ["MLGO"], description="Trading symbols")
    date_range: List[Optional[str]] = Field(default_factory=lambda: [None, None], description="Date range [start, end]")

    # Adaptive ranges (ContinuousTraining can modify these)
    day_score_range: List[float] = Field(default_factory=lambda: [0.7, 1.0], description="Day quality score range")
    roc_range: List[float] = Field(default_factory=lambda: [0.05, 1.0], description="ROC score range")
    activity_range: List[float] = Field(default_factory=lambda: [0.0, 1.0], description="Activity score range")

    # SIMPLIFIED SELECTION MODES
    # DAY SELECTION: How to pick which trading day to use next
    day_selection_mode: Literal["sequential", "quality_weighted", "random"] = Field(
        "sequential",
        description="Day selection: 'sequential' (chronological), 'quality_weighted' (by score), 'random'"
    )

    # RESET POINT SELECTION: How to cycle through reset points within each day
    reset_point_selection_mode: Literal["sequential", "random"] = Field(
        "sequential",
        description="Reset point selection: 'sequential' (by quality), 'random'"
    )

    # BACKWARD COMPATIBILITY: Keep old fields for migration
    selection_mode: Optional[Literal["sequential", "random", "quality_weighted"]] = Field(
        None, description="DEPRECATED: Use day_selection_mode instead"
    )
    randomize_order: Optional[bool] = Field(None, description="DEPRECATED: Removed for simplicity")


class TrainingManagerConfig(BaseModel):
    """Training Manager - Central authority for training lifecycle"""

    # Core mode selection
    mode: str = Field("production", description="Training mode: sweep, production, optuna")

    # Core configuration sections
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    continuous: ContinuousTrainingConfig = Field(default_factory=ContinuousTrainingConfig)

    # Enable/disable data lifecycle management
    enabled: bool = Field(True, description="Enable data lifecycle management")

    # Data cycle management (when to switch days/reset points)
    cycles: DataCycleConfig = Field(default_factory=DataCycleConfig)

    # Adaptive data selection (replaces stages, includes all selection modes)
    adaptive_data: AdaptiveDataConfig = Field(default_factory=AdaptiveDataConfig)

    # Preloading
    preloading: PreloadingConfig = Field(default_factory=PreloadingConfig)

    # BACKWARD COMPATIBILITY: Keep old configs for migration
    reset_points: Optional[ResetPointConfig] = Field(None, description="DEPRECATED: Use adaptive_data instead")
    day_selection: Optional[DaySelectionConfig] = Field(None, description="DEPRECATED: Use adaptive_data instead")


class TrainingConfig(BaseModel):
    """PPO training configuration"""
    seed: int = Field(42, description="Random seed")
    # -----------------------------------------------------------------------------#

    # Core settings
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

    # Rollout settings
    rollout_steps: int = Field(2048, gt=0, description="Steps per rollout")

    # Model management
    continue_training: bool = Field(False, description="Continue from best model")
    checkpoint_interval: Optional[int] = Field(50, description="Updates between checkpoints (null to disable)")
    keep_best_n_models: int = Field(5, description="Number of best models to keep")

    # Early stopping
    early_stop_patience: int = Field(300, description="Updates without improvement")
    early_stop_min_delta: float = Field(0.01, description="Minimum improvement")

    # Evaluation
    eval_frequency: Optional[int] = Field(5, description="Updates between evaluations (null to disable)")
    eval_episodes: int = Field(10, description="Episodes for evaluation")
    best_model_metric: str = Field("mean_reward", description="Model selection metric")

    training_manager: Optional[str] = Field(default_factory=TrainingManagerConfig)


