"""
Model storage configuration for managing model checkpoints and best models.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass
class ModelStorageConfig:
    """Configuration for model storage and management."""
    
    # Directory paths (relative to Hydra output directory)
    checkpoint_dir: str = "checkpoints"              # Directory for regular checkpoints
    best_models_dir: str = "best_models"             # Directory for best model storage
    temp_dir: str = "temp"                           # Temporary directory for atomic operations
    
    # Model naming
    model_prefix: str = "model"                      # Prefix for model files
    checkpoint_name: str = "checkpoint.pt"           # Name for regular checkpoint file
    
    # Best model management
    max_best_models: int = 5                         # Maximum number of best models to keep
    best_model_selection_metric: str = "mean_reward" # Metric for selecting best models
    best_model_selection_mode: Literal["max", "min"] = "max"  # Whether higher or lower is better
    
    # File management
    save_metadata: bool = True                       # Whether to save metadata JSON files
    compression: Optional[str] = None                # Compression type (None, "gzip", etc.)
    atomic_saves: bool = True                        # Use atomic file operations for safety
    
    # Versioning
    version_format: str = "v{version:04d}"          # Format string for version numbers
    timestamp_format: str = "%Y%m%d_%H%M%S"         # Timestamp format for filenames
    
    # Validation
    validate_on_load: bool = True                   # Validate model files when loading
    require_metadata: bool = False                  # Require metadata files to exist
    strict_mode: bool = False                       # Fail on any validation errors
    
    # Cleanup
    cleanup_on_start: bool = False                  # Clean up old files on initialization
    cleanup_temp_on_exit: bool = True              # Clean up temp files on exit
    
    # Backup
    backup_before_overwrite: bool = True           # Create backup before overwriting files
    backup_suffix: str = ".backup"                 # Suffix for backup files
    
    def get_checkpoint_path(self, base_dir: Path) -> Path:
        """Get full checkpoint file path."""
        return base_dir / self.checkpoint_dir / self.checkpoint_name
    
    def get_best_models_path(self, base_dir: Path) -> Path:
        """Get best models directory path."""
        return base_dir / self.best_models_dir
    
    def get_temp_path(self, base_dir: Path) -> Path:
        """Get temporary directory path."""
        return base_dir / self.temp_dir