"""
Model manager with typed state, better error handling, and configuration-based directories.
"""

import os
import json
import shutil
import hashlib
import logging
import torch
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager

from .model_state import ModelState, ModelMetadata
from config.model.model_storage_config import ModelStorageConfig

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Base exception for model manager errors."""
    pass


class ModelNotFoundError(ModelManagerError):
    """Raised when a requested model is not found."""
    pass


class ModelValidationError(ModelManagerError):
    """Raised when model validation fails."""
    pass


class ModelManager:
    """Enhanced model manager with typed state and robust error handling."""

    def __init__(
        self,
        config: ModelStorageConfig,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize model manager.
        
        Args:
            config: Model storage configuration
            base_dir: Base directory for model storage (defaults to current directory)
        """
        self.config = config
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        # Create directories
        self._setup_directories()
        
        # Track active operations for cleanup
        self._active_operations: List[Path] = []
        
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.base_dir / self.config.checkpoint_dir,
            self.base_dir / self.config.best_models_dir,
            self.base_dir / self.config.temp_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Clean up temp directory if configured
        if self.config.cleanup_temp_on_exit:
            temp_dir = self.base_dir / self.config.temp_dir
            for temp_file in temp_dir.glob("*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean temp file {temp_file}: {e}")
                    
    def get_model_version(self) -> int:
        """Get the next model version number."""
        pattern = str(self.base_dir / self.config.best_models_dir / f"{self.config.model_prefix}_v*.pt")
        existing_models = glob.glob(pattern)
        
        versions = []
        for model_path in existing_models:
            try:
                filename = os.path.basename(model_path)
                # Extract version from format: model_v0001_...
                version_part = filename.split("_v")[1].split("_")[0]
                versions.append(int(version_part))
            except (IndexError, ValueError) as e:
                logger.debug(f"Could not parse version from {model_path}: {e}")
                continue
                
        return max(versions) + 1 if versions else 1
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @contextmanager
    def _atomic_write(self, target_path: Path):
        """Context manager for atomic file writes."""
        temp_path = self.base_dir / self.config.temp_dir / f"{target_path.name}.tmp"
        
        try:
            yield temp_path
            
            # Create backup if configured
            if self.config.backup_before_overwrite and target_path.exists():
                backup_path = target_path.with_suffix(target_path.suffix + self.config.backup_suffix)
                shutil.copy2(target_path, backup_path)
                
            # Move temp file to target
            shutil.move(str(temp_path), str(target_path))
            
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
            
    def find_best_model(self) -> Optional[ModelState]:
        """
        Find the best model for continued training.
        
        Returns:
            ModelState with loaded metadata, or None if no models found
        """
        pattern = str(self.base_dir / self.config.best_models_dir / f"{self.config.model_prefix}_v*.pt")
        model_files = glob.glob(pattern)
        
        if not model_files:
            return None
            
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Try to load the most recent valid model
        for model_path in model_files:
            try:
                model_path = Path(model_path)
                
                # Load metadata
                metadata_path = model_path.with_suffix(".json")
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata_dict = json.load(f)
                    metadata = ModelMetadata.from_dict(metadata_dict)
                else:
                    # Create basic metadata from filename
                    metadata = self._metadata_from_filename(model_path)
                    
                # Create model state with metadata only (no weights loaded yet)
                model_state = ModelState(metadata=metadata)
                model_state.metadata.file_path = model_path
                
                if model_path.exists():
                    model_state.metadata.file_size_bytes = model_path.stat().st_size
                    
                return model_state
                
            except Exception as e:
                logger.warning(f"Failed to load model info from {model_path}: {e}")
                continue
                
        return None
    
    def _metadata_from_filename(self, path: Path) -> ModelMetadata:
        """Extract metadata from filename when metadata file is missing."""
        filename = path.stem
        
        # Default metadata
        metadata = ModelMetadata(
            version=0,
            timestamp=datetime.fromtimestamp(path.stat().st_mtime),
            reward=0.0,
        )
        
        # Try to extract version
        if "_v" in filename:
            try:
                version_str = filename.split("_v")[1].split("_")[0]
                metadata.version = int(version_str)
            except (IndexError, ValueError):
                pass
                
        # Try to extract reward
        if "_reward" in filename:
            try:
                reward_part = filename.split("_reward")[1].split("_")[0]
                if reward_part.startswith("n"):
                    metadata.reward = -float(reward_part[1:])
                else:
                    metadata.reward = float(reward_part)
            except (IndexError, ValueError):
                pass
                
        return metadata
    
    def get_best_models(self, top_n: Optional[int] = None) -> List[ModelState]:
        """
        Get list of best models sorted by reward.
        
        Args:
            top_n: Number of top models to return (None for all)
            
        Returns:
            List of ModelState objects sorted by reward (descending)
        """
        pattern = str(self.base_dir / self.config.best_models_dir / f"{self.config.model_prefix}_v*.pt")
        model_files = glob.glob(pattern)
        
        models = []
        for model_path in model_files:
            try:
                model_path = Path(model_path)
                
                # Load metadata
                metadata_path = model_path.with_suffix(".json")
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata_dict = json.load(f)
                    metadata = ModelMetadata.from_dict(metadata_dict)
                else:
                    metadata = self._metadata_from_filename(model_path)
                    
                model_state = ModelState(metadata=metadata)
                model_state.metadata.file_path = model_path
                models.append(model_state)
                
            except Exception as e:
                logger.warning(f"Failed to load model info from {model_path}: {e}")
                
        # Sort by reward
        models.sort(key=lambda x: x.metadata.reward, reverse=True)
        
        if top_n:
            models = models[:top_n]
            
        return models
    
    def save_best_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        model_state: ModelState,
    ) -> Path:
        """
        Save a model as one of the best models.
        
        Args:
            model: PyTorch model to save
            optimizer: Optional optimizer to save
            model_state: Complete model state including metadata
            
        Returns:
            Path to saved model file
            
        Raises:
            ModelManagerError: If save fails
        """
        # Update model state with current weights
        model_state.model_state_dict = model.state_dict()
        if optimizer:
            model_state.optimizer_state_dict = optimizer.state_dict()
            
        # Validate model state
        if not model_state.validate():
            raise ModelValidationError(f"Model validation failed: {model_state.validation_errors}")
            
        # Determine version
        if model_state.metadata.version == 0:
            model_state.metadata.version = self.get_model_version()
            
        # Create filename
        version_str = self.config.version_format.format(version=model_state.metadata.version)
        reward_str = f"{model_state.metadata.reward:.4f}".replace("-", "n")
        timestamp_str = model_state.metadata.timestamp.strftime(self.config.timestamp_format)
        
        filename = f"{self.config.model_prefix}_{version_str}_reward{reward_str}_{timestamp_str}.pt"
        target_path = self.base_dir / self.config.best_models_dir / filename
        
        try:
            # Save model with atomic write
            if self.config.atomic_saves:
                with self._atomic_write(target_path) as temp_path:
                    torch.save(model_state.to_checkpoint(), temp_path)
                    
                    # Calculate checksum after save
                    model_state.metadata.checksum = self._calculate_checksum(temp_path)
                    model_state.metadata.file_size_bytes = temp_path.stat().st_size
            else:
                torch.save(model_state.to_checkpoint(), target_path)
                model_state.metadata.checksum = self._calculate_checksum(target_path)
                model_state.metadata.file_size_bytes = target_path.stat().st_size
                
            # Save metadata file
            if self.config.save_metadata:
                metadata_path = target_path.with_suffix(".json")
                model_state.metadata.file_path = target_path
                
                with open(metadata_path, "w") as f:
                    json.dump(model_state.metadata.to_dict(), f, indent=2)
                    
            # Cleanup old models
            self._cleanup_old_models()
            
            logger.info(
                f"Saved best model v{model_state.metadata.version} "
                f"(reward: {model_state.metadata.reward:.4f}) to {target_path}"
            )
            
            return target_path
            
        except Exception as e:
            raise ModelManagerError(f"Failed to save best model: {e}") from e
            
    def _cleanup_old_models(self) -> None:
        """Remove old models if exceeding max limit."""
        models = self.get_best_models()
        
        if len(models) > self.config.max_best_models:
            # Remove oldest models
            for model_state in models[self.config.max_best_models:]:
                if model_state.metadata.file_path:
                    try:
                        # Remove model file
                        model_state.metadata.file_path.unlink()
                        
                        # Remove metadata file
                        metadata_path = model_state.metadata.file_path.with_suffix(".json")
                        if metadata_path.exists():
                            metadata_path.unlink()
                            
                        # Remove backup if exists
                        backup_path = model_state.metadata.file_path.with_suffix(
                            model_state.metadata.file_path.suffix + self.config.backup_suffix
                        )
                        if backup_path.exists():
                            backup_path.unlink()
                            
                        logger.debug(f"Removed old model: {model_state.metadata.file_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove old model: {e}")
                        
    def load_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> ModelState:
        """
        Load a model from checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            model_path: Optional specific model path (uses best model if None)
            
        Returns:
            Loaded model state
            
        Raises:
            ModelNotFoundError: If no model found
            ModelValidationError: If model validation fails
        """
        # Find model path
        if model_path:
            model_path = Path(model_path)
            if not model_path.exists():
                raise ModelNotFoundError(f"Model file not found: {model_path}")
        else:
            best_model = self.find_best_model()
            if not best_model:
                raise ModelNotFoundError("No best model found")
            model_path = best_model.metadata.file_path
            
        try:
            # Load checkpoint
            device = next(model.parameters()).device
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Load into model state
            model_state = ModelState.from_checkpoint(checkpoint)
            
            # Validate if configured
            if self.config.validate_on_load and not model_state.validate():
                if self.config.strict_mode:
                    raise ModelValidationError(
                        f"Model validation failed: {model_state.validation_errors}"
                    )
                else:
                    logger.warning(f"Model validation warnings: {model_state.validation_errors}")
                    
            # Load weights
            model.load_state_dict(model_state.model_state_dict)
            
            if optimizer and model_state.optimizer_state_dict:
                optimizer.load_state_dict(model_state.optimizer_state_dict)
                
            # Verify checksum if available
            if model_state.metadata.checksum:
                actual_checksum = self._calculate_checksum(model_path)
                if actual_checksum != model_state.metadata.checksum:
                    logger.warning(
                        f"Checksum mismatch for {model_path}: "
                        f"expected {model_state.metadata.checksum}, got {actual_checksum}"
                    )
                    
            logger.info(
                f"Loaded model v{model_state.metadata.version} from {model_path} "
                f"(reward: {model_state.metadata.reward:.4f}, "
                f"episodes: {model_state.metadata.episode_count})"
            )
            
            return model_state
            
        except Exception as e:
            raise ModelManagerError(f"Failed to load model from {model_path}: {e}") from e
            
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        model_state: ModelState,
        checkpoint_name: Optional[str] = None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optional optimizer to save
            model_state: Complete model state
            checkpoint_name: Optional checkpoint filename
            
        Returns:
            Path to saved checkpoint
        """
        # Update model state with current weights
        model_state.model_state_dict = model.state_dict()
        if optimizer:
            model_state.optimizer_state_dict = optimizer.state_dict()
            
        # Update timestamp
        model_state.metadata.timestamp = datetime.now()
        
        # Determine checkpoint path
        if checkpoint_name:
            checkpoint_path = self.base_dir / self.config.checkpoint_dir / checkpoint_name
        else:
            checkpoint_path = self.config.get_checkpoint_path(self.base_dir)
            
        try:
            # Ensure directory exists
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with atomic write
            if self.config.atomic_saves:
                with self._atomic_write(checkpoint_path) as temp_path:
                    torch.save(model_state.to_checkpoint(), temp_path)
            else:
                torch.save(model_state.to_checkpoint(), checkpoint_path)
                
            logger.debug(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            raise ModelManagerError(f"Failed to save checkpoint: {e}") from e
            
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: Optional[str] = None,
    ) -> Optional[ModelState]:
        """
        Load a checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            checkpoint_name: Optional checkpoint filename
            
        Returns:
            Loaded model state or None if not found
        """
        # Determine checkpoint path
        if checkpoint_name:
            checkpoint_path = self.base_dir / self.config.checkpoint_dir / checkpoint_name
        else:
            checkpoint_path = self.config.get_checkpoint_path(self.base_dir)
            
        if not checkpoint_path.exists():
            logger.debug(f"Checkpoint not found: {checkpoint_path}")
            return None
            
        try:
            return self.load_model(model, optimizer, checkpoint_path)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
            
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        if self.config.cleanup_temp_on_exit:
            temp_dir = self.base_dir / self.config.temp_dir
            for temp_file in temp_dir.glob("*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean temp file {temp_file}: {e}")
                    
    # Backward compatibility methods
    
    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """Get best model info - backward compatibility method."""
        best_model = self.find_best_model()
        if not best_model:
            return None
            
        return {
            'path': str(best_model.metadata.file_path),
            'reward': best_model.metadata.reward,
            'version': best_model.metadata.version,
            'metadata': best_model.metadata.to_dict(),
        }