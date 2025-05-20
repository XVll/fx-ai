from typing import Dict, List, Any
import os
import torch
import numpy as np
import time


class TrainingCallback:
    """
    Base callback class for training events.
    Implement specific callbacks by inheriting from this class.
    """

    def on_training_start(self, trainer):
        """Called when training starts."""
        pass

    def on_training_end(self, trainer, stats):
        """Called when training ends."""
        pass

    def on_rollout_start(self, trainer):
        """Called before collecting rollouts."""
        pass

    def on_rollout_end(self, trainer):
        """Called after collecting rollouts."""
        pass

    def on_step(self, trainer, state, action, reward, next_state, info):
        """Called after each environment step."""
        pass

    def on_episode_end(self, trainer, episode_reward, episode_length, info):
        """Called at the end of an episode."""
        pass

    def on_update_start(self, trainer):
        """Called before policy update."""
        pass

    def on_update_end(self, trainer, metrics):
        """Called after policy update."""
        pass

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Called at the end of each update iteration (rollout + update)."""
        pass


class ModelCheckpointCallback(TrainingCallback):
    """
    Callback to save model checkpoints during training.
    """

    def __init__(self, save_dir, save_freq=5, prefix="model", save_best_only=False):
        """
        Initialize the callback.

        Args:
            save_dir: Directory to save checkpoints
            save_freq: Save frequency in update iterations
            prefix: Prefix for checkpoint filenames
            save_best_only: If True, only save models that improve mean reward
        """
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.prefix = prefix
        self.save_best_only = save_best_only
        self.best_mean_reward = -float('inf')

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Save model checkpoint at specified frequency."""
        if update_iter % self.save_freq == 0:
            mean_reward = rollout_stats.get("mean_reward", -float('inf'))

            # Only save if reward improved (if save_best_only is True)
            if not self.save_best_only or mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                filename = f"{self.prefix}_iter{update_iter}_reward{mean_reward:.2f}.pt"
                trainer.save_model(os.path.join(self.save_dir, filename))


class EarlyStoppingCallback(TrainingCallback):
    """
    Early stopping callback to prevent overfitting.
    """

    def __init__(self, patience=5, min_delta=0.0):
        """
        Initialize the callback.

        Args:
            patience: Number of updates with no improvement after which training will be stopped
            min_delta: Minimum change in mean reward to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
        self.should_stop = False

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Check if training should be stopped due to no improvement."""
        mean_reward = rollout_stats.get("mean_reward", -float('inf'))

        if mean_reward > self.best_mean_reward + self.min_delta:
            # Reward improved
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
        else:
            # No improvement
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.should_stop = True
                trainer.logger.info(f"Early stopping triggered after {update_iter} updates. "
                                    f"No improvement for {self.patience} consecutive updates.")