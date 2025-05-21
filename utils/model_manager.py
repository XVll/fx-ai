# utils/model_manager.py
import os
import json
import shutil
import logging
import torch
import glob
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model saving, loading, and versioning for continuous training.

    Features:
    - Maintains a dedicated best_models directory
    - Tracks model versions and performance metrics
    - Supports automatic discovery of best ai for continued training
    - Stores training metadata alongside ai
    """

    def __init__(
            self,
            base_dir: str = "./ai",
            best_models_dir: str = "./best_models",
            model_prefix: str = "model",
            max_best_models: int = 5,
            symbol: str = None
    ):
        """
        Initialize the model manager.

        Args:
            base_dir: Base directory for regular model checkpoints
            best_models_dir: Directory to store best ai
            model_prefix: Prefix for model filenames
            max_best_models: Maximum number of best ai to keep
            symbol: Trading symbol for model specialization
        """
        self.base_dir = base_dir
        self.best_models_dir = best_models_dir
        self.model_prefix = model_prefix
        self.max_best_models = max_best_models
        self.symbol = symbol

        # Create directories if they don't exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)

        # Symbol-specific subdirectories for multi-symbol training
        if self.symbol:
            self.symbol_base_dir = os.path.join(self.base_dir, self.symbol)
            self.symbol_best_dir = os.path.join(self.best_models_dir, self.symbol)
            os.makedirs(self.symbol_base_dir, exist_ok=True)
            os.makedirs(self.symbol_best_dir, exist_ok=True)

        logger.info(f"ModelManager initialized with base_dir={self.base_dir}, best_models_dir={self.best_models_dir}")

    def get_model_version(self) -> int:
        """Get the next model version by checking existing files."""
        # Get all model files in the best ai directory
        pattern = os.path.join(
            self.symbol_best_dir if self.symbol else self.best_models_dir,
            f"{self.model_prefix}_v*.pt"
        )
        existing_models = glob.glob(pattern)

        # Extract version numbers
        versions = []
        for model_path in existing_models:
            try:
                # Extract v{number} from filename
                filename = os.path.basename(model_path)
                version_str = filename.split('_v')[1].split('_')[0]
                versions.append(int(version_str))
            except (IndexError, ValueError):
                continue

        # Return next version (or 1 if none exist)
        return max(versions) + 1 if versions else 1

    def find_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Find the best model for continued training.

        Returns:
            Dictionary with model info or None if no ai found
        """
        model_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        pattern = os.path.join(model_dir, f"{self.model_prefix}_v*.pt")
        model_files = glob.glob(pattern)

        if not model_files:
            logger.info("No existing ai found for continued training")
            return None

        # Find the latest model by sorting based on modified time
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_model_path = model_files[0]

        # Try to load metadata
        metadata_path = latest_model_path.replace('.pt', '_meta.json')
        metadata = {}

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {latest_model_path}: {e}")

        return {
            'path': latest_model_path,
            'metadata': metadata,
            'version': metadata.get('version', 0)
        }

    def save_best_model(
            self,
            model_path: str,
            metrics: Dict[str, Any],
            target_reward: float,
            version: Optional[int] = None
    ) -> str:
        """
        Save a model as one of the best ai.

        Args:
            model_path: Path to the model checkpoint
            metrics: Performance metrics
            target_reward: The reward value for ranking
            version: Optional version override

        Returns:
            Path to the saved best model
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return ""

        # Determine the version
        version = version or self.get_model_version()

        # Format the filename with version and reward
        reward_str = f"{target_reward:.4f}".replace('-', 'n')  # Handle negative rewards
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{self.model_prefix}_v{version}_reward{reward_str}_{timestamp}.pt"
        target_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        target_path = os.path.join(target_dir, filename)

        # Copy the model file
        try:
            shutil.copy2(model_path, target_path)
            logger.info(f"Saved best model to {target_path}")

            # Create metadata file
            metadata = {
                'version': version,
                'timestamp': timestamp,
                'reward': target_reward,
                'metrics': metrics,
                'source': model_path,
                'symbol': self.symbol
            }

            metadata_path = target_path.replace('.pt', '_meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Cleanup old ai if we exceed the limit
            self._cleanup_old_models()

            return target_path

        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            return ""

    def _cleanup_old_models(self):
        """Remove old ai if we exceed the maximum number to keep."""
        target_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        model_files = glob.glob(os.path.join(target_dir, f"{self.model_prefix}_v*.pt"))

        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Keep only the max_best_models newest files
        if len(model_files) > self.max_best_models:
            for old_file in model_files[self.max_best_models:]:
                try:
                    os.remove(old_file)
                    # Also remove metadata file if it exists
                    metadata_file = old_file.replace('.pt', '_meta.json')
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                    logger.info(f"Removed old model: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove old model {old_file}: {e}")

    def load_model(
            self,
            model,
            optimizer=None,
            model_path: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from a checkpoint.

        Args:
            model: The model instance to load weights into
            optimizer: Optional optimizer to load state
            model_path: Path to the specific model to load,
                        if None, loads the latest best model

        Returns:
            Tuple of (model, metadata)
        """
        # Find the best model if not specified
        if not model_path:
            best_model_info = self.find_best_model()
            if not best_model_info:
                logger.warning("No best model found to load")
                return model, {}

            model_path = best_model_info['path']
            metadata = best_model_info['metadata']
        else:
            metadata = {}
            # Try to load metadata
            metadata_path = model_path.replace('.pt', '_meta.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_path}: {e}")

        # Load the model
        try:
            device = next(model.parameters()).device
            checkpoint = torch.load(model_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Loaded model from {model_path}")

            # Return additional training state if needed
            training_state = {
                'global_step': checkpoint.get('global_step_counter', 0),
                'global_episode': checkpoint.get('global_episode_counter', 0),
                'global_update': checkpoint.get('global_update_counter', 0),
                'metadata': metadata
            }

            return model, training_state
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return model, {}