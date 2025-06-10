import os
import logging
import numpy as np


def safe_format_date(date_obj) -> str:
    """Safely format a date object to YYYY-MM-DD string."""
    if isinstance(date_obj, str):
        return date_obj
    elif hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%Y-%m-%d')
    elif hasattr(date_obj, 'date'):
        return date_obj.date().strftime('%Y-%m-%d')
    else:
        return str(date_obj)


class V1TrainingCallback:
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

    def on_attribution_analysis(self, attribution_data):
        """Called when attribution analysis should be performed."""
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

    def on_update_iteration_end(
        self, trainer, update_iter, update_metrics, rollout_stats
    ):
        """Called at the end of each update iteration (rollout + update)."""
        pass


class ModelCheckpointCallbackV1(V1TrainingCallback):
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
            save_best_only: If True, only save ai that improve mean reward
        """
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.prefix = prefix
        self.save_best_only = save_best_only
        self.best_mean_reward = -float("inf")

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def on_update_iteration_end(
        self, trainer, update_iter, update_metrics, rollout_stats
    ):
        """Save model checkpoint at specified frequency."""
        if update_iter % self.save_freq == 0:
            # Get reward value with robust validation
            raw_reward = rollout_stats.get("mean_reward")
            if (
                raw_reward is None
                or not isinstance(raw_reward, (int, float, np.number))
                or not np.isfinite(raw_reward)
            ):
                mean_reward = -float("inf")
            else:
                mean_reward = float(raw_reward)

            # Only save if reward improved (if save_best_only is True)
            if not self.save_best_only or mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                filename = f"{self.prefix}_iter{update_iter}_reward{mean_reward:.2f}.pt"
                trainer.save_model(os.path.join(self.save_dir, filename))


class EarlyStoppingCallbackV1(V1TrainingCallback):
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
        self.best_mean_reward = -float("inf")
        self.no_improvement_count = 0
        self.should_stop = False

    def on_update_iteration_end(
        self, trainer, update_iter, update_metrics, rollout_stats
    ):
        """Check if training should be stopped due to no improvement."""
        # Get reward value with robust validation
        raw_reward = rollout_stats.get("mean_reward")
        if (
            raw_reward is None
            or not isinstance(raw_reward, (int, float, np.number))
            or not np.isfinite(raw_reward)
        ):
            mean_reward = -float("inf")
        else:
            mean_reward = float(raw_reward)

        if mean_reward > self.best_mean_reward + self.min_delta:
            # Reward improved
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
        else:
            # No improvement
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.should_stop = True
                trainer.logger.info(
                    f"Early stopping triggered after {update_iter} updates. "
                    f"No improvement for {self.patience} iterations."
                )


class MomentumTrackingCallbackV1(V1TrainingCallback):
    """Callback to track momentum-specific training metrics."""

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(__name__)

        # Momentum tracking
        self.momentum_day_switches = 0
        self.reset_point_usage = {}
        self.curriculum_progress_history = []
        self.day_performance_stats = {}

    def on_training_start(self, trainer):
        """Initialize momentum tracking."""
        self.logger.info("🎯 Momentum tracking callback initialized")

    def on_update_iteration_end(
        self, trainer, update_iter, update_metrics, rollout_stats
    ):
        """Track momentum-specific metrics."""

        # Track curriculum progress
        if hasattr(trainer, "curriculum_progress"):
            self.curriculum_progress_history.append(trainer.curriculum_progress)

        # Track current momentum day performance
        if hasattr(trainer, "current_momentum_day") and trainer.current_momentum_day:
            day_key = safe_format_date(trainer.current_momentum_day["date"])
            if day_key not in self.day_performance_stats:
                self.day_performance_stats[day_key] = {
                    "episodes": 0,
                    "total_reward": 0,
                    "best_reward": float("-inf"),
                    "activity_score": trainer.current_momentum_day.get(
                        "activity_score", 0
                    ),
                }

            day_stats = self.day_performance_stats[day_key]
            mean_reward = rollout_stats.get("mean_reward", 0)
            num_episodes = rollout_stats.get("num_episodes_in_rollout", 0)

            day_stats["episodes"] += num_episodes
            day_stats["total_reward"] += mean_reward * num_episodes
            day_stats["best_reward"] = max(day_stats["best_reward"], mean_reward)

        # Track reset point usage
        if (
            hasattr(trainer, "used_reset_point_indices")
            and trainer.used_reset_point_indices
        ):
            for idx in trainer.used_reset_point_indices:
                self.reset_point_usage[idx] = self.reset_point_usage.get(idx, 0) + 1

        # Periodic logging
        if update_iter % self.log_frequency == 0:
            self._log_momentum_stats(trainer, update_iter)

    def _log_momentum_stats(self, trainer, update_iter):
        """Log momentum-specific statistics."""

        # Curriculum progress
        if hasattr(trainer, "curriculum_progress"):
            self.logger.info(
                f"📚 Curriculum Progress: {trainer.curriculum_progress:.1%}"
            )

        # Current momentum day info
        if hasattr(trainer, "current_momentum_day") and trainer.current_momentum_day:
            day_info = trainer.current_momentum_day
            day_key = safe_format_date(day_info["date"])

            if day_key in self.day_performance_stats:
                stats = self.day_performance_stats[day_key]
                avg_reward = stats["total_reward"] / max(1, stats["episodes"])

                self.logger.info(
                    f"📅 Current Day: {day_key} "
                    f"(quality: {day_info.get('activity_score', 0):.3f}, "
                    f"episodes: {stats['episodes']}, "
                    f"avg_reward: {avg_reward:.3f})"
                )

        # Reset point distribution
        if self.reset_point_usage and update_iter % (self.log_frequency * 5) == 0:
            total_uses = sum(self.reset_point_usage.values())
            most_used = max(self.reset_point_usage.items(), key=lambda x: x[1])
            self.logger.info(
                f"🎯 Reset Points: {len(self.reset_point_usage)} used, "
                f"most frequent: #{most_used[0]} ({most_used[1]}/{total_uses} uses)"
            )

    def on_training_end(self, trainer, stats):
        """Log final momentum training summary."""
        self.logger.info("🎯 Momentum Training Summary:")
        self.logger.info(f"   Days trained: {len(self.day_performance_stats)}")
        self.logger.info(f"   Reset points used: {len(self.reset_point_usage)}")

        if self.curriculum_progress_history:
            final_progress = self.curriculum_progress_history[-1]
            self.logger.info(f"   Final curriculum progress: {final_progress:.1%}")

        # Best performing day
        if self.day_performance_stats:
            best_day = max(
                self.day_performance_stats.items(), key=lambda x: x[1]["best_reward"]
            )
            self.logger.info(
                f"   Best day: {best_day[0]} (reward: {best_day[1]['best_reward']:.3f})"
            )
