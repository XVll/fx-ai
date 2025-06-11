"""Optuna (Hyperparameter Optimization) structured configuration for Hydra."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum


class SamplerType(str, Enum):
    """Available optuna samplers."""
    TPE = "TPESampler"                    # Tree-structured Parzen Estimator
    GRID = "GridSampler"                  # Grid search sampler
    RANDOM = "RandomSampler"              # Random sampling
    CMA_ES = "CmaEsSampler"               # Covariance Matrix Adaptation Evolution Strategy
    NSGA2 = "NSGAIISampler"               # Non-dominated Sorting Genetic Algorithm II
    QMC = "QMCSampler"                    # Quasi-Monte Carlo sampler


class PrunerType(str, Enum):
    """Available optuna pruners."""
    MEDIAN = "MedianPruner"               # Prune if below median of completed trials
    PERCENTILE = "PercentilePruner"       # Prune if below specified percentile
    SUCCESSIVE_HALVING = "SuccessiveHalvingPruner"  # Successive halving algorithm
    HYPERBAND = "HyperbandPruner"         # Hyperband algorithm
    THRESHOLD = "ThresholdPruner"         # Prune if below/above threshold
    PATIENT = "PatientPruner"             # Wait for patience steps before pruning


class DistributionType(str, Enum):
    """Types of parameter distributions."""
    FLOAT = "float"                       # Continuous float parameter
    INT = "int"                           # Discrete integer parameter
    CATEGORICAL = "categorical"           # Categorical choice parameter
    FLOAT_LOG = "float_log"               # Log-scale float parameter
    INT_LOG = "int_log"                   # Log-scale integer parameter


@dataclass
class ParameterConfig:
    """Configuration for a single hyperparameter."""
    name: str                                          # Parameter name (dot notation for nested)
    type: DistributionType                             # Distribution type
    low: Optional[Union[float, int]] = None            # Lower bound for numeric types
    high: Optional[Union[float, int]] = None           # Upper bound for numeric types
    choices: Optional[List[Any]] = None                # Choices for categorical type
    step: Optional[Union[float, int]] = None           # Step size for discrete params


@dataclass
class SamplerConfig:
    """Configuration for Optuna sampler."""
    type: SamplerType = SamplerType.TPE                # Sampler algorithm type
    n_startup_trials: int = 10                         # Number of random startup trials
    n_ei_candidates: int = 24                          # Number of EI candidates (TPE)
    seed: Optional[int] = None                         # Random seed for reproducibility
    multivariate: bool = True                          # Consider parameter correlations
    warn_independent_sampling: bool = False            # Warn about independent sampling
    
    # TPE-specific settings
    consider_prior: bool = True                        # Consider prior distributions
    prior_weight: float = 1.0                          # Weight of prior distribution
    consider_magic_clip: bool = True                   # Use magic clip technique
    consider_endpoints: bool = False                   # Consider parameter endpoints


@dataclass
class PrunerConfig:
    """Configuration for Optuna pruner."""
    type: Optional[PrunerType] = PrunerType.MEDIAN     # Pruner algorithm type
    n_startup_trials: int = 5                          # Startup trials before pruning
    n_warmup_steps: int = 10                           # Warmup steps before pruning
    interval_steps: int = 1                            # Pruning check interval
    
    # Median/Percentile pruner settings
    percentile: float = 25.0                           # Percentile threshold for pruning
    n_min_trials: int = 5                              # Minimum trials for statistics
    
    # Successive halving settings
    min_resource: int = 1                              # Minimum resource allocation
    reduction_factor: int = 4                          # Resource reduction factor
    
    # Hyperband settings
    max_resource: int = 100                            # Maximum resource allocation
    
    # Threshold settings
    lower: Optional[float] = None                      # Lower threshold for pruning
    upper: Optional[float] = None                      # Upper threshold for pruning
    
    # Patient pruner settings
    patience: int = 10                                 # Patience steps before pruning
    min_delta: float = 0.0                             # Minimum improvement delta


@dataclass
class StudyConfig:
    """Configuration for an Optuna study."""
    study_name: str                                    # Unique name for the study
    direction: Literal["minimize", "maximize"] = "maximize"  # Optimization direction
    metric_name: str = "mean_reward"                   # Metric to optimize
    
    # Storage settings
    storage: str = "sqlite:///sweep_studies.db"        # Database storage URL
    load_if_exists: bool = True                        # Load existing study if found
    
    # Sampler and pruner
    sampler: SamplerConfig = field(default_factory=SamplerConfig)  # Sampling strategy
    pruner: Optional[PrunerConfig] = field(default_factory=PrunerConfig)  # Pruning strategy
    
    # Parameters to optimize
    parameters: List[ParameterConfig] = field(default_factory=list)  # Hyperparameters to tune
    
    # Trial settings
    n_trials: int = 100                                # Number of trials to run
    timeout: Optional[int] = None                      # Timeout in seconds
    n_jobs: int = 1                                    # Number of parallel jobs
    catch_exceptions: bool = True                      # Catch and log exceptions
    
    # Base config reference system
    base_config: Optional[str] = None                  # Base config name to inherit from
    trial_overrides: Dict[str, Any] = field(default_factory=dict)  # Trial-specific overrides
    
    # Legacy full training config (for backward compatibility)
    training_config: Dict[str, Any] = field(default_factory=dict)  # Complete training config
    
    # Trial settings
    episodes_per_trial: int = 1000                     # Episodes per trial
    eval_frequency: int = 100                          # Evaluation frequency
    eval_episodes: int = 10                            # Episodes for evaluation
    
    # Early stopping
    early_stop_patience: int = 5                       # Trials without improvement
    early_stop_min_delta: float = 0.01                 # Minimum improvement delta
    
    # Checkpointing
    save_checkpoints: bool = True                      # Save trial checkpoints
    checkpoint_dir: str = "sweep_checkpoints"          # Checkpoint directory
    keep_best_n: int = 5                               # Keep best N models


@dataclass
class OptunaStudySpec:
    """Complete specification for an Optuna optimization study."""
    # Basic metadata
    name: str                                          # Study specification name
    description: str = ""                              # Study description
    version: str = "1.0"                               # Specification version
    
    # Multiple study configurations
    studies: List[StudyConfig] = field(default_factory=list)  # Study configurations
    
    # Global settings
    dashboard_port: int = 8052                         # Optuna dashboard port
    log_level: str = "INFO"                            # Logging level
    results_dir: str = "sweep_results"                 # Results directory
    
    # Notification settings
    notify_on_complete: bool = False                   # Send notification on completion
    save_study_plots: bool = True                      # Save optimization plots
    
    # Resource limits
    max_concurrent_trials: int = 4                     # Max concurrent trials
    gpu_per_trial: float = 1.0                         # GPU allocation per trial