# agent/continuous_training_callback.py
import os
import time
import json
import logging
from typing import Dict, Any, Optional
import numpy as np

from agent.callbacks import TrainingCallback
from utils.medel_manager import ModelManager


class ContinuousTrainingCallback(TrainingCallback):
    """
    Callback for managing continuous training between sessions.

    Features:
    - Tracks best models across training sessions
    - Synchronizes checkpoints to the best_models directory
    - Manages model versioning and metadata
    - Handles learning rate scheduling for continued training
    """

    def __init__(
            self,
            model_manager: ModelManager,
            reward_metric: str = "mean_reward",
            checkpoint_sync_frequency: int = 2,
            lr_annealing: Optional[Dict[str, Any]] = None,
            best_model_criterion: str = "mean_reward",
            best_model_mode: str = "max",
            load_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the continuous training callback.

        Args:
            model_manager: ModelManager instance
            reward_metric: Metric to track for determining best model
            checkpoint_sync_frequency: How often to sync to best_models dir
            lr_annealing: Learning rate decay settings
            best_model_criterion: Which metric to use for best model
            best_model_mode: "max" or "min" for best model selection
            load_metadata: Metadata from loaded model (for resuming)
        """
        self.model_manager = model_manager
        self.reward_metric = reward_metric
        self.checkpoint_sync_frequency = checkpoint_sync_frequency
        self.lr_annealing = lr_annealing or {}
        self.best_model_criterion = best_model_criterion
        self.best_model_mode = best_model_mode  # "max" or "min"
        self.load_metadata = load_metadata or {}

        # Create a logger
        self.logger = logging.getLogger(__name__)

        # Initialize tracking variables
        self.best_reward = float('-inf') if best_model_mode == "max" else float('inf')
        self.best_model_path = None
        self.session_start_time = None
        self.last_sync_time = None
        self.last_sync_update = 0

        # Load previous best reward if continuing
        if load_metadata and 'metrics' in load_metadata:
            prev_metrics = load_metadata.get('metrics', {})
            prev_reward = prev_metrics.get(self.reward_metric)
            if prev_reward is not None:
                self.best_reward = prev_reward
                self.logger.info(f"Loaded previous best {self.reward_metric}: {self.best_reward}")

    def on_training_start(self, trainer):
        """Called when training starts."""
        self.session_start_time = time.time()
        self.last_sync_time = self.session_start_time

        # Apply learning rate annealing if this is a continuation
        if self.lr_annealing.get('enabled', False) and self.load_metadata:
            self._apply_lr_annealing(trainer)

        self.logger.info(f"Continuous training session started with best {self.reward_metric}: {self.best_reward}")

    def _apply_lr_annealing(self, trainer):
        """Apply learning rate decay for continued training."""
        if not hasattr(trainer, 'optimizer'):
            self.logger.warning("Trainer has no optimizer attribute, cannot apply LR annealing")
            return

        current_lr = trainer.optimizer.param_groups[0]['lr']
        min_lr = self.lr_annealing.get('min_lr', 1e-5)
        decay_factor = self.lr_annealing.get('decay_factor', 0.7)

        # Calculate new learning rate
        new_lr = max(current_lr * decay_factor, min_lr)

        # Apply new learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.logger.info(f"Applied LR annealing: {current_lr:.6f} -> {new_lr:.6f}")

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Called at the end of each update iteration (rollout + update)."""
        # Check if this is a new best model
        current_reward = rollout_stats.get(self.reward_metric, -float('inf'))

        is_best = False
        if self.best_model_mode == "max":
            if current_reward > self.best_reward:
                is_best = True
        else:  # "min"
            if current_reward < self.best_reward:
                is_best = True

        if is_best:
            self.best_reward = current_reward

            # Save to regular checkpoint first
            checkpoint_path = os.path.join(
                trainer.model_dir,
                f"checkpoint_iter{update_iter}_{self.reward_metric}{current_reward:.4f}.pt"
            )
            trainer.save_model(checkpoint_path)
            self.best_model_path = checkpoint_path

            # Save to best_models directory
            metrics = {
                **update_metrics,
                **rollout_stats,
                "update_iter": update_iter,
                "timestamp": time.time()
            }

            self.model_manager.save_best_model(
                checkpoint_path,
                metrics,
                current_reward
            )

            self.logger.info(f"New best model at iteration {update_iter} with {self.reward_metric}={current_reward:.4f}")

        # Periodically sync to best_models directory even if not the best
        if (update_iter - self.last_sync_update) >= self.checkpoint_sync_frequency:
            if self.best_model_path:
                current_time = time.time()
                # Only sync if it's been more than 5 minutes since last sync
                if current_time - self.last_sync_time > 300:
                    self.last_sync_time = current_time
                    self.last_sync_update = update_iter

                    # Create metadata with current metrics
                    metrics = {
                        **update_metrics,
                        **rollout_stats,
                        "update_iter": update_iter,
                        "timestamp": time.time(),
                        "is_periodic_sync": True
                    }

                    # Save latest checkpoint to best_models with lower version priority
                    latest_checkpoint_path = os.path.join(
                        trainer.model_dir,
                        f"latest_checkpoint_iter{update_iter}.pt"
                    )
                    trainer.save_model(latest_checkpoint_path)

                    self.model_manager.save_best_model(
                        latest_checkpoint_path,
                        metrics,
                        current_reward
                    )

                    self.logger.info(f"Periodic sync to best_models at iteration {update_iter}")

    def on_training_end(self, trainer, stats):
        """Called when training ends."""
        session_duration = time.time() - self.session_start_time

        # Save final model
        final_stats = {
            **stats,
            "session_duration": session_duration,
            "best_reward": self.best_reward,
            "is_final": True,
            "timestamp": time.time()
        }

        # Save final checkpoint
        final_checkpoint_path = os.path.join(
            trainer.model_dir,
            "final_model.pt"
        )
        trainer.save_model(final_checkpoint_path)

        # Save to best_models
        if self.best_model_path != final_checkpoint_path:
            self.model_manager.save_best_model(
                final_checkpoint_path,
                final_stats,
                stats.get(self.reward_metric, self.best_reward)
            )

        self.logger.info(f"Continuous training session ended. Duration: {session_duration:.2f}s. "
                         f"Best {self.reward_metric}: {self.best_reward:.4f}")

        # Create a session summary file
        summary_path = os.path.join(trainer.output_dir, "session_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(final_stats, f, indent=2)