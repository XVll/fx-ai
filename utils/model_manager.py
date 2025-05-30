# utils/model_manager.py - CLEAN: Minimal logging, focus on model management

import os
import json
import shutil
import logging
import torch
import glob
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """Clean model manager with minimal logging"""

    def __init__(
            self,
            base_dir: str = "./models",
            best_models_dir: str = "./best_models",
            model_prefix: str = "model",
            max_best_models: int = 5,
            symbol: str = None
    ):
        self.base_dir = base_dir
        self.best_models_dir = best_models_dir
        self.model_prefix = model_prefix
        self.max_best_models = max_best_models
        self.symbol = symbol

        # Create directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)

        # Symbol-specific subdirectories
        if self.symbol:
            self.symbol_base_dir = os.path.join(self.base_dir, self.symbol)
            self.symbol_best_dir = os.path.join(self.best_models_dir, self.symbol)
            os.makedirs(self.symbol_base_dir, exist_ok=True)
            os.makedirs(self.symbol_best_dir, exist_ok=True)

    def get_model_version(self) -> int:
        """Get the next model version number"""
        model_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        pattern = os.path.join(model_dir, f"{self.model_prefix}_v*.pt")
        existing_models = glob.glob(pattern)

        versions = []
        for model_path in existing_models:
            try:
                filename = os.path.basename(model_path)
                version_str = filename.split('_v')[1].split('_')[0]
                versions.append(int(version_str))
            except (IndexError, ValueError):
                continue

        return max(versions) + 1 if versions else 1

    def find_best_model(self) -> Optional[Dict[str, Any]]:
        """Find the best model for continued training"""
        model_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        pattern = os.path.join(model_dir, f"{self.model_prefix}_v*.pt")
        model_files = glob.glob(pattern)

        if not model_files:
            return None

        # Get latest model by modification time
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_model_path = model_files[0]

        # Load metadata if available
        metadata_path = latest_model_path.replace('.pt', '_meta.json')
        metadata = {}

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

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
        """Save a model as one of the best models"""
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return ""

        # Determine version
        version = version or self.get_model_version()

        # Create filename
        reward_str = f"{target_reward:.4f}".replace('-', 'n')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_prefix}_v{version}_reward{reward_str}_{timestamp}.pt"

        target_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        target_path = os.path.join(target_dir, filename)

        try:
            # Copy model file
            shutil.copy2(model_path, target_path)

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

            # Cleanup old models
            self._cleanup_old_models()

            logger.info(f"Saved best model v{version} (reward: {target_reward:.4f})")
            return target_path

        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            return ""

    def _cleanup_old_models(self):
        """Remove old models if exceeding max limit"""
        target_dir = self.symbol_best_dir if self.symbol else self.best_models_dir
        model_files = glob.glob(os.path.join(target_dir, f"{self.model_prefix}_v*.pt"))

        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Keep only the max_best_models newest files
        if len(model_files) > self.max_best_models:
            for old_file in model_files[self.max_best_models:]:
                try:
                    os.remove(old_file)
                    # Remove metadata file too
                    metadata_file = old_file.replace('.pt', '_meta.json')
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                except Exception as e:
                    logger.warning(f"Failed to remove old model: {e}")

    def load_model(
            self,
            model,
            optimizer=None,
            model_path: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a model from checkpoint"""
        # Find best model if no path specified
        if not model_path:
            best_model_info = self.find_best_model()
            if not best_model_info:
                logger.warning("No best model found")
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
                    logger.warning(f"Failed to load metadata: {e}")

        # Load model
        try:
            device = next(model.parameters()).device
            checkpoint = torch.load(model_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Loaded model: {os.path.basename(model_path)}")

            # Return training state
            training_state = {
                'global_step': checkpoint.get('global_step_counter', 0),
                'global_episode': checkpoint.get('global_episode_counter', 0),
                'global_update': checkpoint.get('global_update_counter', 0),
                'metadata': metadata
            }

            return model, training_state

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return model, {}