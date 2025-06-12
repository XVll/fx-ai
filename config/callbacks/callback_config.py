"""
Typed callback configurations using Hydra dataclasses.

Replaces dictionary-based callback configurations with strongly typed
dataclasses for better validation and IDE support.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ContinuousCallbackConfig:
    enabled: bool = True                    # Whether callback is active
    metric_name: str = "mean_reward"
    metric_mode: str = "max"
    checkpoint_frequency: int = 100
    checkpoint_time_interval: float = 300.0  # 5 minutes
    save_initial_model: bool = True


@dataclass
class EvaluationCallbackConfig:
    """Configuration for EvaluationCallback."""
    enabled: bool = True                    # Whether callback is active
    update_frequency: Optional[int] = None  # Updates between evaluations
    episode_frequency: Optional[int] = 100  # Episodes between evaluations
    time_frequency_minutes: Optional[float] = None  # Minutes between evaluations


@dataclass
class PPOMetricsCallbackConfig:
    """Configuration for PPOMetricsCallback."""
    enabled: bool = True                    # Whether callback is active
    buffer_size: int = 1000                 # Buffer size for rolling metrics


@dataclass
class ExecutionMetricsCallbackConfig:
    """Configuration for ExecutionMetricsCallback."""
    enabled: bool = True                    # Whether callback is active
    buffer_size: int = 2000                 # Buffer size for execution metrics


@dataclass
class PortfolioMetricsCallbackConfig:
    """Configuration for PortfolioMetricsCallback."""
    enabled: bool = True                    # Whether callback is active
    buffer_size: int = 1000                 # Buffer size for portfolio metrics
    log_frequency: int = 10                 # Episode frequency for logging


@dataclass
class ModelMetricsCallbackConfig:
    """Configuration for ModelMetricsCallback."""
    enabled: bool = True                    # Whether callback is active
    buffer_size: int = 500                  # Buffer size for model metrics


@dataclass
class SessionMetricsCallbackConfig:
    """Configuration for SessionMetricsCallback."""
    enabled: bool = True                    # Whether callback is active
    log_frequency: int = 100                # Episode frequency for logging
    track_system_resources: bool = True     # Whether to track system resources


@dataclass 
class CaptumAttributionCallbackConfig:
    """Configuration for CaptumAttributionCallback."""
    enabled: bool = True                    # Whether callback is active
    analyze_every_n_episodes: Optional[int] = 10    # Episodes between analyses
    analyze_every_n_updates: Optional[int] = 5      # Updates between analyses  
    save_to_wandb: bool = True              # Log results to WandB


@dataclass
class OptunaCallbackConfig:
    """Configuration for OptunaCallback."""
    enabled: bool = False                   # Whether callback is active
    metric_name: str = "mean_reward"        # Metric to optimize
    report_interval: int = 1                # Report every N evaluations
    use_best_value: bool = False            # Report best value seen so far




@dataclass
class CallbackConfig:
    continuous: ContinuousCallbackConfig = field(default_factory=ContinuousCallbackConfig)
    evaluation: EvaluationCallbackConfig = field(default_factory=EvaluationCallbackConfig)
    ppo_metrics: PPOMetricsCallbackConfig = field(default_factory=PPOMetricsCallbackConfig)
    execution_metrics: ExecutionMetricsCallbackConfig = field(default_factory=ExecutionMetricsCallbackConfig)
    portfolio_metrics: PortfolioMetricsCallbackConfig = field(default_factory=PortfolioMetricsCallbackConfig)
    model_metrics: ModelMetricsCallbackConfig = field(default_factory=ModelMetricsCallbackConfig)
    session_metrics: SessionMetricsCallbackConfig = field(default_factory=SessionMetricsCallbackConfig)
    captum_attribution: CaptumAttributionCallbackConfig = field(default_factory=CaptumAttributionCallbackConfig)
    optuna: OptunaCallbackConfig = field(default_factory=OptunaCallbackConfig)