"""
Simplified model storage configuration.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelStorageConfig:
    """Simple configuration for model storage."""
    
    # NOTE: Paths now managed by PathManager - these fields deprecated
    # Use PathManager.checkpoints_dir, PathManager.best_models_dir, PathManager.temp_models_dir instead
    
    # Model naming
    model_prefix: str = "model"
    checkpoint_name: str = "checkpoint.pt"
    
    # Best model management
    max_best_models: int = 5
    
    # File management
    save_metadata: bool = True
    
    # Versioning
    version_format: str = "v{version:04d}"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # NOTE: Path methods deprecated - use PathManager instead
    # def get_checkpoint_path(self, base_dir: Path) -> Path:
    #     return base_dir / self.checkpoint_dir / self.checkpoint_name
    # 
    # def get_best_models_path(self, base_dir: Path) -> Path:
    #     return base_dir / self.best_models_dir
    # 
    # def get_temp_path(self, base_dir: Path) -> Path:
    #     return base_dir / self.temp_dir