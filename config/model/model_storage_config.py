"""
Simplified model storage configuration.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelStorageConfig:
    """Simple configuration for model storage."""
    
    # Directory paths (relative to base directory)
    checkpoint_dir: str = "checkpoints"
    best_models_dir: str = "best_models"
    temp_dir: str = "temp"
    
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
    
    def get_checkpoint_path(self, base_dir: Path) -> Path:
        return base_dir / self.checkpoint_dir / self.checkpoint_name
    
    def get_best_models_path(self, base_dir: Path) -> Path:
        return base_dir / self.best_models_dir
    
    def get_temp_path(self, base_dir: Path) -> Path:
        return base_dir / self.temp_dir