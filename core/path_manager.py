"""
Centralized path management for FxAI trading system.

This module provides a unified approach to managing file paths across the entire system,
with support for Hydra integration and environment awareness.
"""

from pathlib import Path
from typing import Optional
from hydra.core.hydra_config import HydraConfig


class PathManager:
    """
    Centralized path management system for FxAI.
    
    Provides consistent path resolution across all components with support for:
    - Environment-aware base directory configuration
    - Hydra integration for experiment outputs
    - Separation of persistent data vs ephemeral experiment outputs
    """
    
    def __init__(
        self, 
        base_dir: Optional[Path] = None,
        use_hydra_output: bool = False,
        experiment_dir: Optional[Path] = None
    ):
        """
        Initialize PathManager.
        
        Args:
            base_dir: Base directory for the project (defaults to current working directory)
            use_hydra_output: Whether to use Hydra's output directory as base for experiments
            experiment_dir: Explicit experiment directory (overrides Hydra detection)
        """
        self._base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._use_hydra_output = use_hydra_output
        self._experiment_dir = Path(experiment_dir) if experiment_dir else None
        
        # Ensure base directory exists
        self._base_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def base_dir(self) -> Path:
        """Project base directory."""
        return self._base_dir
    
    @property
    def experiment_dir(self) -> Optional[Path]:
        """Current experiment directory (from Hydra or explicit)."""
        if self._experiment_dir:
            return self._experiment_dir
            
        if self._use_hydra_output:
            try:
                hydra_cfg = HydraConfig.get()
                return Path(hydra_cfg.runtime.output_dir)
            except Exception:
                # Hydra not initialized, fall back to experiments directory
                pass
                
        return None
    
    # Persistent directories (survive across runs)
    @property
    def data_dir(self) -> Path:
        """Raw data directory."""
        path = self._base_dir / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory for persistent cached data."""
        path = self._base_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def models_dir(self) -> Path:
        """Models directory for persistent model storage."""
        path = self._base_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def global_logs_dir(self) -> Path:
        """Global logs directory."""
        path = self._base_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Experiment directories (per-run, can be cleaned up)
    @property
    def experiments_dir(self) -> Path:
        """Base experiments directory."""
        path = self._base_dir / "experiments"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def current_experiment_dir(self) -> Path:
        """Current experiment directory."""
        if self.experiment_dir:
            return self.experiment_dir
        
        # Fallback to experiments directory
        return self.experiments_dir
    
    # Specific subdirectories
    def get_data_subdir(self, subdir: str) -> Path:
        """Get subdirectory under data directory."""
        path = self.data_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_cache_subdir(self, subdir: str) -> Path:
        """Get subdirectory under cache directory."""
        path = self.cache_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_models_subdir(self, subdir: str) -> Path:
        """Get subdirectory under models directory."""
        path = self.models_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_experiment_subdir(self, subdir: str) -> Path:
        """Get subdirectory under current experiment directory."""
        path = self.current_experiment_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Commonly used specific paths
    @property
    def databento_dir(self) -> Path:
        """Databento data files directory."""
        return self.get_data_subdir("dnb")
    
    @property
    def features_cache_dir(self) -> Path:
        """Feature cache directory."""
        return self.get_cache_subdir("features")
    
    @property
    def indices_cache_dir(self) -> Path:
        """Indices cache directory."""
        return self.get_cache_subdir("indices")
    
    @property
    def scanner_cache_dir(self) -> Path:
        """Scanner cache directory."""
        return self.get_cache_subdir("scanner")
    
    @property
    def checkpoints_dir(self) -> Path:
        """Model checkpoints directory."""
        return self.get_models_subdir("checkpoints")
    
    @property
    def best_models_dir(self) -> Path:
        """Best models directory."""
        return self.get_models_subdir("best")
    
    @property
    def temp_models_dir(self) -> Path:
        """Temporary models directory."""
        return self.get_models_subdir("temp")
    
    @property
    def experiment_logs_dir(self) -> Path:
        """Current experiment logs directory."""
        return self.get_experiment_subdir("logs")
    
    @property
    def experiment_analysis_dir(self) -> Path:
        """Current experiment analysis directory."""
        return self.get_experiment_subdir("analysis")
    
    @property
    def experiment_artifacts_dir(self) -> Path:
        """Current experiment artifacts directory."""
        return self.get_experiment_subdir("artifacts")
    
    @property
    def wandb_dir(self) -> Path:
        """W&B directory under experiments."""
        path = self.experiments_dir / "wandb"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def cleanup_experiment(self, experiment_path: Optional[Path] = None) -> bool:
        """
        Clean up experiment directory.
        
        Args:
            experiment_path: Specific experiment path to clean up. 
                           If None, cleans current experiment.
        
        Returns:
            True if cleanup successful, False otherwise.
        """
        target_path = experiment_path or self.current_experiment_dir
        
        if not target_path.exists():
            return True
            
        try:
            import shutil
            shutil.rmtree(target_path)
            return True
        except Exception as e:
            print(f"Failed to cleanup experiment directory {target_path}: {e}")
            return False
    
    def get_relative_path(self, path: Path, relative_to: Optional[Path] = None) -> Path:
        """
        Get path relative to a base directory.
        
        Args:
            path: The path to make relative
            relative_to: Base directory (defaults to base_dir)
        
        Returns:
            Relative path
        """
        base = relative_to or self.base_dir
        try:
            return path.relative_to(base)
        except ValueError:
            # Path is not relative to base, return as-is
            return path
    
    def __str__(self) -> str:
        """String representation showing key paths."""
        return (
            f"PathManager(\n"
            f"  base_dir={self.base_dir}\n"
            f"  experiment_dir={self.experiment_dir}\n"
            f"  data_dir={self.data_dir}\n"
            f"  cache_dir={self.cache_dir}\n"
            f"  models_dir={self.models_dir}\n"
            f")"
        )


# Global instance - can be overridden by applications
_global_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """Get the global PathManager instance."""
    global _global_path_manager
    if _global_path_manager is None:
        _global_path_manager = PathManager()
    return _global_path_manager


def set_path_manager(path_manager: PathManager) -> None:
    """Set the global PathManager instance."""
    global _global_path_manager
    _global_path_manager = path_manager


def initialize_paths(
    base_dir: Optional[Path] = None,
    use_hydra_output: bool = False,
    experiment_dir: Optional[Path] = None
) -> PathManager:
    """
    Initialize the global PathManager with specific configuration.
    
    Args:
        base_dir: Base directory for the project
        use_hydra_output: Whether to use Hydra's output directory for experiments
        experiment_dir: Explicit experiment directory
    
    Returns:
        Configured PathManager instance
    """
    path_manager = PathManager(
        base_dir=base_dir,
        use_hydra_output=use_hydra_output,
        experiment_dir=experiment_dir
    )
    set_path_manager(path_manager)
    return path_manager