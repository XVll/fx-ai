
import os
import time
import json
import logging
from typing import Dict, Any, Optional

from agent.base_callbacks import TrainingCallback
from utils.model_manager import ModelManager


class ContinuousTrainingCallback(TrainingCallback):
    """Clean continuous training callback with minimal logging"""

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
        self.model_manager = model_manager
        self.reward_metric = reward_metric
        self.checkpoint_sync_frequency = checkpoint_sync_frequency
        self.lr_annealing = lr_annealing or {}
        self.best_model_criterion = best_model_criterion
        self.best_model_mode = best_model_mode
        self.load_metadata = load_metadata or {}

        self.logger = logging.getLogger(__name__)

        # Initialize tracking
        self.best_reward = float('-inf') if best_model_mode == "max" else float('inf')
        self.best_model_path = None
        self.session_start_time = None
        self.last_sync_time = None
        self.last_sync_update = 0

        # Load the previous best reward if continuing
        if load_metadata and 'metrics' in load_metadata:
            prev_metrics = load_metadata.get('metrics', {})
            prev_reward = prev_metrics.get(self.reward_metric)
            if prev_reward is not None:
                self.best_reward = prev_reward
                self.logger.info(f"Continuing from previous best {self.reward_metric}: {self.best_reward:.4f}")

    def on_training_start(self, trainer):
        """Initialize continuous training session"""
        self.session_start_time = time.time()
        self.last_sync_time = self.session_start_time

        # Apply learning rate annealing if this is a continuation
        if self.lr_annealing.get('enabled', False) and self.load_metadata:
            self._apply_lr_annealing(trainer)

    def _apply_lr_annealing(self, trainer):
        """Apply learning rate decay for continued training"""
        if not hasattr(trainer, 'optimizer'):
            self.logger.warning("No optimizer found for LR annealing")
            return

        current_lr = trainer.optimizer.param_groups[0]['lr']
        min_lr = self.lr_annealing.get('min_lr', 1e-5)
        decay_factor = self.lr_annealing.get('decay_factor', 0.7)

        new_lr = max(current_lr * decay_factor, min_lr)

        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.logger.info(f"LR annealing applied: {current_lr:.6f} → {new_lr:.6f}")

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Check for the new best model and periodic sync"""
        current_reward = rollout_stats.get(self.reward_metric, -float('inf'))

        # Check if this is the new best model
        is_best = False
        if self.best_model_mode == "max":
            if current_reward > self.best_reward:
                is_best = True
        else:  # "min"
            if current_reward < self.best_reward:
                is_best = True

        if is_best:
            self.best_reward = current_reward

            # Save checkpoint
            checkpoint_path = os.path.join(
                trainer.model_dir,
                f"checkpoint_iter{update_iter}_{self.reward_metric}{current_reward:.4f}.pt"
            )
            trainer.save_model(checkpoint_path)
            self.best_model_path = checkpoint_path

            # Save to the best_models directory
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

            self.logger.info(f"New best model: iter {update_iter}, {self.reward_metric}={current_reward:.4f}")

        # Periodic sync
        if (update_iter - self.last_sync_update) >= self.checkpoint_sync_frequency:
            current_time = time.time()
            if current_time - self.last_sync_time > 300:  # 5 minutes
                self.last_sync_time = current_time
                self.last_sync_update = update_iter

                # Save periodic checkpoint
                latest_checkpoint_path = os.path.join(
                    trainer.model_dir,
                    f"latest_checkpoint_iter{update_iter}.pt"
                )
                trainer.save_model(latest_checkpoint_path)

                metrics = {
                    **update_metrics,
                    **rollout_stats,
                    "update_iter": update_iter,
                    "timestamp": time.time(),
                    "is_periodic_sync": True
                }

                self.model_manager.save_best_model(
                    latest_checkpoint_path,
                    metrics,
                    current_reward
                )

    def on_training_end(self, trainer, stats):
        """Save final model and session summary"""
        session_duration = time.time() - self.session_start_time

        # Final statistics
        final_stats = {
            **stats,
            "session_duration": session_duration,
            "best_reward": self.best_reward,
            "is_final": True,
            "timestamp": time.time()
        }

        # Save the final model
        final_checkpoint_path = os.path.join(trainer.model_dir, "final_model.pt")
        trainer.save_model(final_checkpoint_path)

        # Save to best_models if different from best
        if self.best_model_path != final_checkpoint_path:
            self.model_manager.save_best_model(
                final_checkpoint_path,
                final_stats,
                stats.get(self.reward_metric, self.best_reward)
            )

        self.logger.info(f"Training session ended. Duration: {session_duration:.1f}s, "
                         f"Best {self.reward_metric}: {self.best_reward:.4f}")

        # Save session summary
        summary_path = os.path.join(trainer.output_dir, "session_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(final_stats, f, indent=2)