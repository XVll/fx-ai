"""
Typed callback configurations using Pydantic.

Replaces dictionary-based callback configurations with strongly typed
Pydantic models for better validation and IDE support.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MetricsCallbackConfig(BaseModel):
    """Configuration for MetricsCallback."""
    enabled: bool = Field(default=True, description="Whether callback is active")
    log_freq: int = Field(default=10, description="Episode frequency for logging metrics")
    console_output: bool = Field(default=True, description="Whether to output to console")


class CheckpointCallbackConfig(BaseModel):
    """Configuration for CheckpointCallback."""
    enabled: bool = Field(default=True, description="Whether callback is active")
    save_freq: int = Field(default=100, description="Episode frequency for saving checkpoints")
    keep_best: int = Field(default=5, description="Number of best models to keep")
    save_optimizer: bool = Field(default=True, description="Whether to save optimizer state")


class WandBCallbackConfig(BaseModel):
    """Configuration for WandBCallback."""
    enabled: bool = Field(default=False, description="Whether callback is active")
    project: str = Field(default="fxai-v2", description="WandB project name")
    entity: Optional[str] = Field(default=None, description="WandB entity/team name")
    tags: List[str] = Field(default_factory=list, description="List of tags for the run")
    log_freq: int = Field(default=10, description="Episode frequency for logging metrics")
    log_gradients: bool = Field(default=False, description="Whether to log gradient norms")
    log_parameters: bool = Field(default=True, description="Whether to log model parameters")
    log_artifacts: bool = Field(default=True, description="Whether to log model artifacts")


class AttributionCallbackConfig(BaseModel):
    """Configuration for AttributionCallback."""
    enabled: bool = Field(default=False, description="Whether callback is active")
    analysis_freq: int = Field(default=1000, description="Episode frequency for attribution analysis")
    methods: List[str] = Field(
        default_factory=lambda: ["integrated_gradients"],
        description="List of attribution methods to use"
    )
    sample_size: int = Field(default=100, description="Number of samples for attribution analysis")
    save_visualizations: bool = Field(default=True, description="Whether to save attribution plots")


class PerformanceCallbackConfig(BaseModel):
    """Configuration for PerformanceCallback."""
    enabled: bool = Field(default=True, description="Whether callback is active")
    analysis_freq: int = Field(default=100, description="Episode frequency for performance analysis")
    metrics: List[str] = Field(
        default_factory=lambda: ["sharpe_ratio", "max_drawdown", "win_rate"],
        description="List of performance metrics to calculate"
    )
    rolling_window: int = Field(default=100, description="Rolling window for metric calculations")
    save_reports: bool = Field(default=True, description="Whether to save performance reports")


class OptunaCallbackConfig(BaseModel):
    """Configuration for OptunaCallback."""
    enabled: bool = Field(default=False, description="Whether callback is active")
    study_name: Optional[str] = Field(default=None, description="Name of Optuna study")
    storage_url: Optional[str] = Field(default=None, description="Storage URL for study persistence")
    pruning_enabled: bool = Field(default=True, description="Whether to enable trial pruning")
    report_freq: int = Field(default=10, description="Episode frequency for reporting to Optuna")


class EarlyStoppingCallbackConfig(BaseModel):
    """Configuration for EarlyStoppingCallback."""
    enabled: bool = Field(default=False, description="Whether callback is active")
    patience: int = Field(default=100, description="Number of episodes to wait for improvement")
    min_delta: float = Field(default=0.001, description="Minimum improvement to consider as progress")
    metric: str = Field(default="episode_reward", description="Metric to monitor for early stopping")
    mode: str = Field(default="max", description="Whether to maximize or minimize metric")


class CallbackConfig(BaseModel):
    """Main callback configuration containing all callback configs."""
    
    metrics: MetricsCallbackConfig = Field(default_factory=MetricsCallbackConfig)
    checkpoint: CheckpointCallbackConfig = Field(default_factory=CheckpointCallbackConfig)
    wandb: WandBCallbackConfig = Field(default_factory=WandBCallbackConfig)
    attribution: AttributionCallbackConfig = Field(default_factory=AttributionCallbackConfig)
    performance: PerformanceCallbackConfig = Field(default_factory=PerformanceCallbackConfig)
    optuna: OptunaCallbackConfig = Field(default_factory=OptunaCallbackConfig)
    early_stopping: EarlyStoppingCallbackConfig = Field(default_factory=EarlyStoppingCallbackConfig)
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True  # Validate on assignment