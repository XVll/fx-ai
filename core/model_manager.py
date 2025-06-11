"""
Simplified model manager with configuration-based directories and better error handling.
"""

import os
import json
import shutil
import logging
import torch
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

from config.model.model_storage_config import ModelStorageConfig

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Base exception for model manager errors."""
    pass


class ModelNotFoundError(ModelManagerError):
    """Raised when a requested model is not found."""
    pass


class ModelManager:
    """Simplified model manager with configuration-based storage and robust error handling."""

    def __init__(
        self,
        config: Optional[ModelStorageConfig] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize model manager.
        
        Args:
            config: Model storage configuration (defaults to ModelStorageConfig())
            base_dir: Base directory for model storage (defaults to current directory)
        """
        self.config = config or ModelStorageConfig()
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        # Create directories
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.base_dir / self.config.checkpoint_dir,
            self.base_dir / self.config.best_models_dir,
            self.base_dir / self.config.temp_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
                    
    def get_model_version(self) -> int:
        """Get the next model version number."""
        pattern = str(self.base_dir / self.config.best_models_dir / f"{self.config.model_prefix}_v*.pt")
        existing_models = glob.glob(pattern)
        
        versions = []
        for model_path in existing_models:
            try:
                filename = os.path.basename(model_path)
                version_part = filename.split("_v")[1].split("_")[0]
                versions.append(int(version_part))
            except (IndexError, ValueError) as e:
                logger.debug(f"Could not parse version from {model_path}: {e}")
                continue
                
        return max(versions) + 1 if versions else 1
    
    def find_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Find the best model for continued training.
        
        Returns:
            Dictionary with model info, or None if no models found
        """
        pattern = str(self.base_dir / self.config.best_models_dir / f"{self.config.model_prefix}_v*.pt")
        model_files = glob.glob(pattern)
        
        if not model_files:
            return None
            
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_model_path = model_files[0]
        
        # Load metadata if available
        metadata_path = Path(latest_model_path).with_suffix(".json")
        metadata = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        else:
            # Extract basic info from filename
            metadata = self._extract_metadata_from_filename(Path(latest_model_path))
                
        return {
            "path": latest_model_path,
            "metadata": metadata,
            "version": metadata.get("version", 0),
        }
    
    def _extract_metadata_from_filename(self, path: Path) -> Dict[str, Any]:
        """Extract metadata from filename when metadata file is missing."""
        filename = path.stem
        metadata = {
            "version": 0,
            "timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "reward": 0.0,
        }
        
        # Try to extract version
        if "_v" in filename:
            try:
                version_str = filename.split("_v")[1].split("_")[0]
                metadata["version"] = int(version_str)
            except (IndexError, ValueError):
                pass
                
        # Try to extract reward
        if "_reward" in filename:
            try:
                reward_part = filename.split("_reward")[1].split("_")[0]
                if reward_part.startswith("n"):
                    metadata["reward"] = -float(reward_part[1:])
                else:
                    metadata["reward"] = float(reward_part)
            except (IndexError, ValueError):
                pass
                
        return metadata
    
    def save_best_model(
        self,
        model_path: str,
        metrics: Dict[str, Any],
        target_reward: float,
        version: Optional[int] = None,
    ) -> str:
        """
        Save a model as one of the best models.
        
        Args:
            model_path: Path to the model file to save
            metrics: Training metrics dictionary
            target_reward: Reward value for this model
            version: Optional version number (auto-generated if None)
            
        Returns:
            Path to saved model file
        """
        if not os.path.exists(model_path):
            raise ModelManagerError(f"Model file not found: {model_path}")

        # Determine version
        version = version or self.get_model_version()

        # Create filename
        reward_str = f"{target_reward:.4f}".replace("-", "n")
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        
        version_str = self.config.version_format.format(version=version)
        filename = f"{self.config.model_prefix}_{version_str}_reward{reward_str}_{timestamp}.pt"
        target_path = self.base_dir / self.config.best_models_dir / filename

        try:
            # Copy model file
            shutil.copy2(model_path, target_path)

            # Create metadata file if configured
            if self.config.save_metadata:
                metadata = {
                    "version": version,
                    "timestamp": timestamp,
                    "reward": target_reward,
                    "metrics": metrics,
                    "source": model_path,
                    "file_size": target_path.stat().st_size,
                }

                metadata_path = target_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            # Cleanup old models
            self._cleanup_old_models()

            logger.info(f"Saved best model v{version} (reward: {target_reward:.4f}) to {target_path}")
            return str(target_path)

        except Exception as e:
            raise ModelManagerError(f"Failed to save best model: {e}") from e
            
    def _cleanup_old_models(self) -> None:
        """Remove old models if exceeding max limit."""
        model_files = glob.glob(
            str(self.base_dir / self.config.best_models_dir / f"{self.config.model_prefix}_v*.pt")
        )

        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Keep only the max_best_models newest files
        if len(model_files) > self.config.max_best_models:
            for old_file in model_files[self.config.max_best_models:]:
                try:
                    os.remove(old_file)
                    # Remove metadata file too
                    metadata_file = Path(old_file).with_suffix(".json")
                    if metadata_file.exists():
                        metadata_file.unlink()
                        
                    logger.debug(f"Removed old model: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove old model: {e}")

    def load_model(
        self, 
        model, 
        optimizer=None, 
        model_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            model_path: Optional specific model path (uses best model if None)
            
        Returns:
            Tuple of (model, training_state_dict)
        """
        # Find model path
        if not model_path:
            best_model_info = self.find_best_model()
            if not best_model_info:
                raise ModelNotFoundError("No best model found")
            model_path = best_model_info["path"]
            
        # Convert to string for os.path.exists
        model_path_str = str(model_path)
        if not os.path.exists(model_path_str):
            raise ModelNotFoundError(f"Model file not found: {model_path_str}")

        try:
            # Load checkpoint
            device = next(model.parameters()).device
            checkpoint = torch.load(model_path_str, map_location=device, weights_only=False)

            # Load model weights
            model.load_state_dict(checkpoint["model_state_dict"])
            
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Return training state
            training_state = {
                "global_step": checkpoint.get("global_step_counter", 0),
                "global_episode": checkpoint.get("global_episode_counter", 0),
                "global_update": checkpoint.get("global_update_counter", 0),
                "global_cycle": checkpoint.get("global_cycle_counter", 0),
                "metadata": checkpoint.get("metadata", {}),
            }

            logger.info(f"Loaded model from {model_path_str}")
            return model, training_state

        except Exception as e:
            raise ModelManagerError(f"Failed to load model from {model_path_str}: {e}") from e

    def save_checkpoint(
        self,
        model,
        optimizer,
        global_step_counter: int,
        global_episode_counter: int,
        global_update_counter: int,
        global_cycle_counter: int,
        metadata: Dict[str, Any],
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Save a checkpoint with full training state."""
        if checkpoint_path is None:
            checkpoint_path = self.config.get_checkpoint_path(self.base_dir)
        else:
            checkpoint_path = Path(checkpoint_path)

        try:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "global_step_counter": global_step_counter,
                "global_episode_counter": global_episode_counter,
                "global_update_counter": global_update_counter,
                "global_cycle_counter": global_cycle_counter,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
            }

            # Ensure directory exists
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            raise ModelManagerError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(
        self, 
        model, 
        optimizer=None, 
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a checkpoint with full training state."""
        if checkpoint_path is None:
            checkpoint_path_obj = self.config.get_checkpoint_path(self.base_dir)
        else:
            checkpoint_path_obj = Path(checkpoint_path)

        if not checkpoint_path_obj.exists():
            logger.debug(f"Checkpoint not found: {checkpoint_path_obj}")
            return model, {}

        try:
            return self.load_model(model, optimizer, checkpoint_path_obj)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return model, {}
    
    # Backward compatibility methods
    
    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """Get best model info - backward compatibility method."""
        best_model = self.find_best_model()
        if not best_model:
            return None
            
        return {
            'path': best_model['path'],
            'reward': best_model['metadata'].get('reward', 0.0),
            'version': best_model.get('version', 0),
            'metadata': best_model['metadata'],
        }