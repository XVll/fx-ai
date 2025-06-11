"""
Typed callback configurations using Hydra dataclasses.

Replaces dictionary-based callback configurations with strongly typed
dataclasses for better validation and IDE support.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MetricsCallbackConfig:
    """Configuration for MetricsCallback."""
    enabled: bool = True                    # Whether callback is active
    log_freq: int = 10                      # Episode frequency for logging metrics
    console_output: bool = True             # Whether to output to console
    shutdown_timeout: float = 5.0           # Shutdown timeout in seconds


@dataclass
class CheckpointCallbackConfig:
    """Configuration for CheckpointCallback."""
    enabled: bool = True                    # Whether callback is active
    save_freq: int = 100                    # Episode frequency for saving checkpoints
    keep_best: int = 5                      # Number of best models to keep
    save_optimizer: bool = True             # Whether to save optimizer state
    shutdown_timeout: float = 30.0          # Shutdown timeout in seconds


@dataclass
class WandBCallbackConfig:
    """Configuration for WandBCallback."""
    enabled: bool = False                   # Whether callback is active
    project: str = "fx-ai"                  # WandB project name
    entity: Optional[str] = None            # WandB entity/team name
    name: Optional[str] = None              # Run name
    tags: List[str] = field(default_factory=list)  # List of tags for the run
    notes: Optional[str] = None             # Run notes
    log_freq: int = 10                      # Episode frequency for logging metrics
    log_gradients: bool = False             # Whether to log gradient norms
    log_parameters: bool = True             # Whether to log model parameters
    log_artifacts: bool = True              # Whether to log model artifacts
    shutdown_timeout: float = 60.0          # Shutdown timeout in seconds


@dataclass
class AttributionCallbackConfig:
    """Configuration for AttributionCallback."""
    enabled: bool = False                   # Whether callback is active
    analysis_freq: int = 1000               # Episode frequency for attribution analysis
    methods: List[str] = field(default_factory=lambda: ["integrated_gradients"])  # Attribution methods
    sample_size: int = 100                  # Number of samples for attribution analysis
    save_visualizations: bool = True        # Whether to save attribution plots
    shutdown_timeout: float = 15.0          # Shutdown timeout in seconds


@dataclass
class PerformanceCallbackConfig:
    """Configuration for PerformanceCallback."""
    enabled: bool = True                    # Whether callback is active
    analysis_freq: int = 100                # Episode frequency for performance analysis
    metrics: List[str] = field(default_factory=lambda: ["sharpe_ratio", "max_drawdown", "win_rate"])  # Performance metrics
    rolling_window: int = 100               # Rolling window for metric calculations
    save_reports: bool = True               # Whether to save performance reports
    shutdown_timeout: float = 10.0          # Shutdown timeout in seconds


@dataclass
class OptunaCallbackConfig:
    """Configuration for OptunaCallback."""
    enabled: bool = False                   # Whether callback is active
    study_name: Optional[str] = None        # Name of Optuna study
    storage_url: Optional[str] = None       # Storage URL for study persistence
    pruning_enabled: bool = True            # Whether to enable trial pruning
    report_freq: int = 10                   # Episode frequency for reporting to Optuna
    shutdown_timeout: float = 5.0           # Shutdown timeout in seconds


@dataclass
class EarlyStoppingCallbackConfig:
    """Configuration for EarlyStoppingCallback."""
    enabled: bool = False                   # Whether callback is active
    patience: int = 100                     # Number of episodes to wait for improvement
    min_delta: float = 0.001                # Minimum improvement to consider as progress
    metric: str = "episode_reward"          # Metric to monitor for early stopping
    mode: str = "max"                       # Whether to maximize or minimize metric
    shutdown_timeout: float = 5.0           # Shutdown timeout in seconds


@dataclass
class CallbackConfig:
    """Main callback configuration containing all callback configs."""
    
    # System settings
    shutdown_timeout: float = 10.0          # Shutdown timeout in seconds
    
    # Individual callback configs
    metrics: MetricsCallbackConfig = field(default_factory=MetricsCallbackConfig)
    checkpoint: CheckpointCallbackConfig = field(default_factory=CheckpointCallbackConfig)
    wandb: WandBCallbackConfig = field(default_factory=WandBCallbackConfig)
    attribution: AttributionCallbackConfig = field(default_factory=AttributionCallbackConfig)
    performance: PerformanceCallbackConfig = field(default_factory=PerformanceCallbackConfig)
    optuna: OptunaCallbackConfig = field(default_factory=OptunaCallbackConfig)
    early_stopping: EarlyStoppingCallbackConfig = field(default_factory=EarlyStoppingCallbackConfig)