"""Optuna configuration schemas and types."""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class SamplerType(str, Enum):
    """Available Optuna samplers."""
    
    TPE = "TPESampler"
    GRID = "GridSampler"
    RANDOM = "RandomSampler"
    CMA_ES = "CmaEsSampler"
    NSGA2 = "NSGAIISampler"
    QMC = "QMCSampler"


class PrunerType(str, Enum):
    """Available Optuna pruners."""
    
    MEDIAN = "MedianPruner"
    PERCENTILE = "PercentilePruner"
    SUCCESSIVE_HALVING = "SuccessiveHalvingPruner"
    HYPERBAND = "HyperbandPruner"
    THRESHOLD = "ThresholdPruner"
    PATIENT = "PatientPruner"


class DistributionType(str, Enum):
    """Types of parameter distributions."""
    
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    FLOAT_LOG = "float_log"
    INT_LOG = "int_log"


class ParameterConfig(BaseModel):
    """Configuration for a single hyperparameter."""
    
    name: str = Field(..., description="Parameter name (dot notation for nested)")
    type: DistributionType = Field(..., description="Distribution type")
    low: Optional[Union[float, int]] = Field(None, description="Lower bound")
    high: Optional[Union[float, int]] = Field(None, description="Upper bound")
    choices: Optional[List[Any]] = Field(None, description="Categorical choices")
    step: Optional[Union[float, int]] = Field(None, description="Step size for discrete params")
    
    @validator("low", "high")
    def validate_bounds(cls, v, values):
        """Validate bounds are provided for numeric types."""
        if "type" in values and values["type"] in [
            DistributionType.FLOAT,
            DistributionType.INT,
            DistributionType.FLOAT_LOG,
            DistributionType.INT_LOG,
        ]:
            if v is None:
                raise ValueError(f"Bounds required for {values['type']} distribution")
        return v
    
    @validator("choices")
    def validate_choices(cls, v, values):
        """Validate choices are provided for categorical type."""
        if "type" in values and values["type"] == DistributionType.CATEGORICAL:
            if not v:
                raise ValueError("Choices required for categorical distribution")
        return v


class SamplerConfig(BaseModel):
    """Configuration for Optuna sampler."""
    
    type: SamplerType = Field(SamplerType.TPE, description="Sampler type")
    n_startup_trials: int = Field(10, description="Number of random startup trials")
    n_ei_candidates: int = Field(24, description="Number of EI candidates (TPE)")
    seed: Optional[int] = Field(None, description="Random seed")
    multivariate: bool = Field(True, description="Consider parameter correlations")
    warn_independent_sampling: bool = Field(False, description="Warn about independent sampling")
    
    # Additional sampler-specific settings
    consider_prior: bool = Field(True, description="Consider prior (TPE)")
    prior_weight: float = Field(1.0, description="Prior weight (TPE)")
    consider_magic_clip: bool = Field(True, description="Use magic clip (TPE)")
    consider_endpoints: bool = Field(False, description="Consider endpoints (TPE)")


class PrunerConfig(BaseModel):
    """Configuration for Optuna pruner."""
    
    type: Optional[PrunerType] = Field(PrunerType.MEDIAN, description="Pruner type")
    n_startup_trials: int = Field(5, description="Startup trials before pruning")
    n_warmup_steps: int = Field(10, description="Warmup steps before pruning")
    interval_steps: int = Field(1, description="Pruning check interval")
    
    # Median/Percentile pruner settings
    percentile: float = Field(25.0, description="Percentile for PercentilePruner")
    n_min_trials: int = Field(5, description="Minimum trials for statistics")
    
    # Successive halving settings
    min_resource: int = Field(1, description="Minimum resource (SuccessiveHalving)")
    reduction_factor: int = Field(4, description="Reduction factor (SuccessiveHalving)")
    
    # Hyperband settings
    max_resource: int = Field(100, description="Maximum resource (Hyperband)")
    
    # Threshold settings
    lower: Optional[float] = Field(None, description="Lower threshold")
    upper: Optional[float] = Field(None, description="Upper threshold")
    
    # Patient pruner settings
    patience: int = Field(10, description="Patience steps (Patient)")
    min_delta: float = Field(0.0, description="Minimum delta (Patient)")


class StudyConfig(BaseModel):
    """Configuration for an Optuna study."""
    
    study_name: str = Field(..., description="Name of the study")
    direction: Literal["minimize", "maximize"] = Field("maximize", description="Optimization direction")
    metric_name: str = Field("mean_reward", description="Metric to optimize")
    
    # Storage settings
    storage: str = Field("sqlite:///optuna_studies.db", description="Storage URL")
    load_if_exists: bool = Field(True, description="Load existing study")
    
    # Sampler and pruner
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    pruner: Optional[PrunerConfig] = Field(default_factory=PrunerConfig)
    
    # Parameters to optimize
    parameters: List[ParameterConfig] = Field(..., description="Parameters to optimize")
    
    # Trial settings
    n_trials: int = Field(100, description="Number of trials to run")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    n_jobs: int = Field(1, description="Number of parallel jobs")
    catch_exceptions: bool = Field(True, description="Catch and log exceptions")
    
    # Training settings
    training_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Base training configuration"
    )
    episodes_per_trial: int = Field(1000, description="Episodes per trial")
    eval_frequency: int = Field(100, description="Evaluation frequency")
    eval_episodes: int = Field(10, description="Episodes for evaluation")
    
    # Early stopping
    early_stop_patience: int = Field(5, description="Trials without improvement")
    early_stop_min_delta: float = Field(0.01, description="Minimum improvement")
    
    # Checkpointing
    save_checkpoints: bool = Field(True, description="Save trial checkpoints")
    checkpoint_dir: str = Field("optuna_checkpoints", description="Checkpoint directory")
    keep_best_n: int = Field(5, description="Keep best N models")


class OptunaStudySpec(BaseModel):
    """Complete specification for an Optuna optimization study."""
    
    # Basic metadata
    name: str = Field(..., description="Study specification name")
    description: str = Field("", description="Study description")
    version: str = Field("1.0", description="Specification version")
    
    # Multiple study configurations
    studies: List[StudyConfig] = Field(..., description="Study configurations")
    
    # Global settings
    dashboard_port: int = Field(8052, description="Optuna dashboard port")
    log_level: str = Field("INFO", description="Logging level")
    results_dir: str = Field("optuna_results", description="Results directory")
    
    # Notification settings
    notify_on_complete: bool = Field(False, description="Send notification on completion")
    save_study_plots: bool = Field(True, description="Save optimization plots")
    
    # Resource limits
    max_concurrent_trials: int = Field(4, description="Max concurrent trials")
    gpu_per_trial: float = Field(1.0, description="GPU allocation per trial")
    
    class Config:
        """Pydantic config."""
        extra = "forbid"