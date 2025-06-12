"""
Centralized path configuration for FxAI trading system.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PathConfig(BaseModel):
    """
    Centralized path configuration that works with PathManager.
    
    This replaces hardcoded path strings scattered across individual config files.
    All paths are relative to the base directory managed by PathManager.
    """
    
    # Base configuration
    base_dir: Optional[str] = Field(
        default=None,
        description="Base directory for the project. If None, uses current working directory."
    )
    
    use_hydra_output: bool = Field(
        default=False,
        description="Whether to use Hydra's output directory as base for experiments"
    )
    
    experiment_dir: Optional[str] = Field(
        default=None,
        description="Explicit experiment directory path"
    )
    
    # Persistent directories (survive across runs)
    data_subdir: str = Field(
        default="data",
        description="Data directory name under base_dir"
    )
    
    cache_subdir: str = Field(
        default="cache", 
        description="Cache directory name under base_dir"
    )
    
    models_subdir: str = Field(
        default="models",
        description="Models directory name under base_dir"
    )
    
    logs_subdir: str = Field(
        default="logs",
        description="Global logs directory name under base_dir"
    )
    
    # Experiment directories (per-run)
    experiments_subdir: str = Field(
        default="experiments",
        description="Experiments directory name under base_dir"
    )
    
    # Specific data subdirectories
    databento_subdir: str = Field(
        default="dnb",
        description="Databento data subdirectory under data/"
    )
    
    # Specific cache subdirectories  
    features_cache_subdir: str = Field(
        default="features",
        description="Features cache subdirectory under cache/"
    )
    
    indices_cache_subdir: str = Field(
        default="indices", 
        description="Indices cache subdirectory under cache/"
    )
    
    scanner_cache_subdir: str = Field(
        default="scanner",
        description="Scanner cache subdirectory under cache/"
    )
    
    # Specific model subdirectories
    checkpoints_subdir: str = Field(
        default="checkpoints",
        description="Checkpoints subdirectory under models/"
    )
    
    best_models_subdir: str = Field(
        default="best",
        description="Best models subdirectory under models/"
    )
    
    temp_models_subdir: str = Field(
        default="temp",
        description="Temporary models subdirectory under models/"
    )
    
    # Experiment subdirectories
    experiment_logs_subdir: str = Field(
        default="logs",
        description="Logs subdirectory under current experiment/"
    )
    
    experiment_analysis_subdir: str = Field(
        default="analysis",
        description="Analysis subdirectory under current experiment/"
    )
    
    experiment_artifacts_subdir: str = Field(
        default="artifacts", 
        description="Artifacts subdirectory under current experiment/"
    )
    
    # W&B configuration
    wandb_subdir: str = Field(
        default="wandb",
        description="W&B subdirectory under experiments/"
    )
    
    # Legacy compatibility paths (to be deprecated)
    legacy_best_models_dir: str = Field(
        default="best_models",
        description="Legacy best models directory (deprecated, use models/best/)"
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True