"""
Model checkpointing callback.

Handles saving and loading of model checkpoints during training
with configurable frequency and best model tracking.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch
from datetime import datetime
from .base import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Model checkpointing callback.
    
    Saves model checkpoints at specified intervals and keeps track
    of the best performing models based on episode rewards.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        save_freq: int = 100,
        keep_best: int = 5,
        output_path: Optional[str] = None,
        trainer: Optional[Any] = None,
        name: Optional[str] = None
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            enabled: Whether callback is active
            save_freq: Episode frequency for saving checkpoints
            keep_best: Number of best models to keep
            output_path: Directory to save checkpoints
            trainer: PPO trainer instance
            name: Optional custom name
        """
        super().__init__(enabled, name)
        self.save_freq = save_freq
        self.keep_best = keep_best
        self.trainer = trainer
        
        # Setup paths
        self.output_path = Path(output_path) if output_path else Path("./checkpoints")
        self.checkpoint_dir = self.output_path / "models"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best models tracking
        self.best_models = []  # List of (reward, episode, path) tuples
        self.last_save_episode = 0
        self.checkpoint_count = 0
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize checkpointing."""
        super().on_training_start(context)
        
        self.logger.info(f"💾 Checkpoint callback initialized")
        self.logger.info(f"  Save frequency: {self.save_freq} episodes")
        self.logger.info(f"  Keep best: {self.keep_best} models")
        self.logger.info(f"  Output path: {self.checkpoint_dir}")
        
        # Save initial configuration
        config = context.get("config", {})
        if config:
            config_path = self.output_path / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.info(f"  Saved training config to {config_path}")
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Check if checkpoint should be saved."""
        super().on_episode_end(context)
        
        episode_info = context.get("episode", {})
        episode_num = episode_info.get("num", self.episode_count)
        reward = episode_info.get("reward", 0.0)
        
        # Save checkpoint at specified frequency
        if episode_num - self.last_save_episode >= self.save_freq:
            self._save_checkpoint(episode_num, reward, context, is_scheduled=True)
            self.last_save_episode = episode_num
        
        # Check if this is a best model
        self._check_and_save_best_model(episode_num, reward, context)
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Save final checkpoint."""
        super().on_training_end(context)
        
        final_metrics = context.get("final_metrics", {})
        total_episodes = context.get("total_episodes", self.episode_count)
        best_reward = final_metrics.get("best_reward", 0.0)
        
        # Save final checkpoint
        self._save_checkpoint(
            total_episodes, 
            best_reward, 
            context, 
            is_final=True
        )
        
        self.logger.info(f"💾 Checkpointing completed")
        self.logger.info(f"  Total checkpoints saved: {self.checkpoint_count}")
        self.logger.info(f"  Best models tracked: {len(self.best_models)}")
    
    def _save_checkpoint(
        self, 
        episode_num: int, 
        reward: float, 
        context: Dict[str, Any],
        is_scheduled: bool = False,
        is_best: bool = False,
        is_final: bool = False
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            episode_num: Current episode number
            reward: Episode reward
            context: Training context
            is_scheduled: Whether this is a scheduled save
            is_best: Whether this is a best model save
            is_final: Whether this is the final save
            
        Returns:
            Path to saved checkpoint
        """
        if not self.trainer:
            self.logger.warning("No trainer provided, cannot save checkpoint")
            return ""
        
        try:
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_final:
                filename = f"final_model_ep{episode_num}_r{reward:.3f}_{timestamp}.pth"
            elif is_best:
                filename = f"best_model_ep{episode_num}_r{reward:.3f}_{timestamp}.pth"
            else:
                filename = f"checkpoint_ep{episode_num}_r{reward:.3f}_{timestamp}.pth"
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Get model state from trainer
            if hasattr(self.trainer, 'get_model_state'):
                model_state = self.trainer.get_model_state()
            else:
                self.logger.warning("Trainer does not have get_model_state method")
                return ""
            
            # Create checkpoint data
            checkpoint_data = {
                "model_state": model_state,
                "episode_num": episode_num,
                "reward": reward,
                "timestamp": timestamp,
                "checkpoint_type": "final" if is_final else "best" if is_best else "scheduled",
                "training_context": {
                    "total_episodes": episode_num,
                    "total_updates": context.get("update", {}).get("num", 0),
                    "config": context.get("config", {})
                }
            }
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            self.checkpoint_count += 1
            
            # Create metadata file
            metadata_path = checkpoint_path.with_suffix('.json')
            metadata = {
                "episode_num": episode_num,
                "reward": reward,
                "timestamp": timestamp,
                "checkpoint_type": checkpoint_data["checkpoint_type"],
                "model_path": str(checkpoint_path),
                "training_context": checkpoint_data["training_context"]
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            save_type = "FINAL" if is_final else "BEST" if is_best else "SCHEDULED"
            self.logger.info(f"💾 {save_type} checkpoint saved: {checkpoint_path.name}")
            self.logger.info(f"   Episode: {episode_num}, Reward: {reward:.3f}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return ""
    
    def _check_and_save_best_model(
        self, 
        episode_num: int, 
        reward: float, 
        context: Dict[str, Any]
    ) -> None:
        """Check if current model should be saved as best model."""
        # Check if this reward qualifies as a best model
        should_save = False
        
        if len(self.best_models) < self.keep_best:
            # We haven't reached the limit yet
            should_save = True
        else:
            # Check if this reward is better than the worst best model
            worst_best_reward = min(self.best_models, key=lambda x: x[0])[0]
            if reward > worst_best_reward:
                should_save = True
        
        if should_save:
            # Save as best model
            checkpoint_path = self._save_checkpoint(
                episode_num, reward, context, is_best=True
            )
            
            if checkpoint_path:
                # Add to best models list
                self.best_models.append((reward, episode_num, checkpoint_path))
                
                # Sort by reward (descending) and keep only the best ones
                self.best_models.sort(key=lambda x: x[0], reverse=True)
                
                # Remove worst models if we exceed the limit
                if len(self.best_models) > self.keep_best:
                    removed_models = self.best_models[self.keep_best:]
                    self.best_models = self.best_models[:self.keep_best]
                    
                    # Delete removed model files
                    for _, _, path in removed_models:
                        try:
                            Path(path).unlink(missing_ok=True)
                            Path(path).with_suffix('.json').unlink(missing_ok=True)
                            self.logger.info(f"Removed old best model: {Path(path).name}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove old model {path}: {e}")
                
                self.logger.info(f"🏆 New best model saved! Reward: {reward:.3f}")
                self.logger.info(f"   Best models count: {len(self.best_models)}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from path.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Load state into trainer if available
            if self.trainer and hasattr(self.trainer, 'set_model_state'):
                self.trainer.set_model_state(checkpoint_data["model_state"])
                self.logger.info(f"💾 Checkpoint loaded: {checkpoint_path.name}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return {}
    
    def get_best_models(self) -> List[Dict[str, Any]]:
        """Get list of best models with metadata."""
        best_models_info = []
        
        for reward, episode, path in self.best_models:
            metadata_path = Path(path).with_suffix('.json')
            metadata = {}
            
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load metadata from {metadata_path}: {e}")
            
            best_models_info.append({
                "reward": reward,
                "episode": episode,
                "path": path,
                "metadata": metadata
            })
        
        return best_models_info
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest_checkpoint)