"""
Centralized path configuration for FxAI trading system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PathConfig:
    """
    Centralized path configuration that works with PathManager.
    
    This replaces hardcoded path strings scattered across individual config files.
    All paths are relative to the base directory managed by PathManager.
    """
    
    # Base configuration
    base_dir: Optional[str] = None  # Base directory for the project. If None, uses current working directory.
    
    use_hydra_output: bool = False  # Whether to use Hydra's output directory as base for experiments
    
    experiment_dir: Optional[str] = None  # Explicit experiment directory path
    
    # Persistent directories (survive across runs)
    data_subdir: str = "data"  # Data directory name under base_dir
    
    cache_subdir: str = "cache"  # Cache directory name under base_dir
    
    models_subdir: str = "models"  # Models directory name under base_dir
    
    logs_subdir: str = "logs"  # Global logs directory name under base_dir
    
    # Experiment directories (per-run)
    experiments_subdir: str = "experiments"  # Experiments directory name under base_dir
    
    # Specific data subdirectories
    databento_subdir: str = "dnb"  # Databento data subdirectory under data/
    
    # Specific cache subdirectories  
    features_cache_subdir: str = "features"  # Features cache subdirectory under cache/
    
    indices_cache_subdir: str = "indices"  # Indices cache subdirectory under cache/
    
    scanner_cache_subdir: str = "scanner"  # Scanner cache subdirectory under cache/
    
    # Specific model subdirectories
    checkpoints_subdir: str = "checkpoints"  # Checkpoints subdirectory under models/
    
    best_models_subdir: str = "best"  # Best models subdirectory under models/
    
    temp_models_subdir: str = "temp"  # Temporary models subdirectory under models/
    
    # Experiment subdirectories
    experiment_logs_subdir: str = "logs"  # Logs subdirectory under current experiment/
    
    experiment_analysis_subdir: str = "analysis"  # Analysis subdirectory under current experiment/
    
    experiment_artifacts_subdir: str = "artifacts"  # Artifacts subdirectory under current experiment/
    
    # W&B configuration
    wandb_subdir: str = "wandb"  # W&B subdirectory under experiments/
    
    # Legacy compatibility paths (to be deprecated)
    legacy_best_models_dir: str = "best_models"  # Legacy best models directory (deprecated, use models/best/)