"""Optuna callback for hyperparameter optimization tracking and pruning."""

from typing import Dict, Any, Optional, List
import numpy as np

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from agent.callbacks import V1BaseCallback


class OptunaCallbackV1(V1BaseCallback):
    """Callback for Optuna hyperparameter optimization.

    This callback:
    - Reports metrics to Optuna for pruning decisions
    - Tracks optimization-specific metrics
    - Handles trial pruning when performance is poor
    - Only performs calculations when enabled
    """

    def __init__(
        self,
        trial: Optional["optuna.Trial"] = None,
        metric_name: str = "mean_reward",
        report_frequency: int = 1,
        pruning_warmup_steps: int = 5,
        enabled: bool = True,
    ):
        """Initialize Optuna callback.

        Args:
            trial: Optuna trial object
            metric_name: Name of the metric to optimize (default: mean_reward)
            report_frequency: How often to report metrics (every N episodes)
            pruning_warmup_steps: Number of steps before pruning is allowed
            enabled: Whether this callback is active
        """
        super().__init__(enabled)

        if not OPTUNA_AVAILABLE and enabled:
            self.logger.warning("Optuna not installed. OptunaCallback disabled.")
            self.enabled = False
            return

        self.trial = trial
        self.metric_name = metric_name
        self.report_frequency = report_frequency
        self.pruning_warmup_steps = pruning_warmup_steps

        # Check if this is a subprocess trial (no real trial object)
        self.is_subprocess = trial is None
        if self.is_subprocess:
            self.logger.info(
                "Optuna callback running in subprocess mode (no trial object)"
            )

        # Tracking
        self.step = 0
        self.episode_rewards: List[float] = []
        self.update_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.best_reward = float("-inf")

        # Performance tracking
        self.training_start_time = None
        self.episodes_completed = 0
        self.updates_completed = 0

    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Initialize tracking at training start."""
        if not self.enabled or not self.trial:
            return

        import time

        self.training_start_time = time.time()
        self.logger.info(
            f"Optuna trial {self.trial.number} started - optimizing {self.metric_name}"
        )

    def on_episode_end(self, episode_num: int, episode_data: Dict[str, Any]) -> None:
        """Track episode metrics and report to Optuna."""
        if not self.enabled:
            return

        self.episodes_completed += 1

        # Extract reward
        episode_reward = episode_data.get("episode_reward", 0.0)
        self.episode_rewards.append(episode_reward)

        # Report at specified frequency
        if self.episodes_completed % self.report_frequency == 0:
            # Calculate mean reward over recent episodes
            recent_rewards = self.episode_rewards[-self.report_frequency :]
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0

            # Update best reward
            self.best_reward = max(self.best_reward, mean_reward)

            # Report to Optuna (only if we have a real trial object)
            if not self.is_subprocess and self.trial:
                self.step += 1
                self.trial.report(mean_reward, self.step)

                # Check for pruning (after warmup)
                if self.step > self.pruning_warmup_steps:
                    if self.trial.should_prune():
                        self.logger.info(
                            f"Trial {self.trial.number} pruned at step {self.step} "
                            f"(reward: {mean_reward:.4f})"
                        )
                        raise optuna.TrialPruned()
            else:
                # In subprocess mode, just track the step
                self.step += 1

    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Track update metrics for more granular reporting."""
        if not self.enabled:
            return

        self.updates_completed += 1

        # Track update-level rewards if available
        if "mean_reward" in update_metrics:
            self.update_rewards.append(update_metrics["mean_reward"])

            # Also report update-level metrics to Optuna (only if real trial)
            if (
                self.metric_name == "mean_reward"
                and not self.is_subprocess
                and self.trial
            ):
                self.trial.report(update_metrics["mean_reward"], self.step)
                self.step += 1

                # Check for pruning
                if self.step > self.pruning_warmup_steps and self.trial.should_prune():
                    self.logger.info(
                        f"Trial {self.trial.number} pruned at update {update_num} "
                        f"(reward: {update_metrics['mean_reward']:.4f})"
                    )
                    raise optuna.TrialPruned()
            elif self.is_subprocess:
                self.step += 1

    def on_evaluation_end(self, eval_results: Dict[str, Any]) -> None:
        """Track evaluation metrics - these are often most reliable for optimization."""
        if not self.enabled:
            return

        eval_reward = eval_results.get("mean_reward", 0.0)
        self.eval_rewards.append(eval_reward)

        # Evaluation metrics are high quality - always report them (if real trial)
        if not self.is_subprocess and self.trial:
            self.trial.report(eval_reward, self.step)
            self.step += 1

            # Check for pruning
            if self.step > self.pruning_warmup_steps and self.trial.should_prune():
                self.logger.info(
                    f"Trial {self.trial.number} pruned after evaluation "
                    f"(reward: {eval_reward:.4f})"
                )
                raise optuna.TrialPruned()

            # Log progress
            self.logger.info(
                f"Trial {self.trial.number} - Eval reward: {eval_reward:.4f} "
                f"(best: {self.best_reward:.4f})"
            )
        else:
            # In subprocess mode, just track
            self.step += 1
            self.logger.info(
                f"Subprocess - Eval reward: {eval_reward:.4f} "
                f"(best: {self.best_reward:.4f})"
            )

        # Update best regardless of mode
        self.best_reward = max(self.best_reward, eval_reward)

    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Report final metrics and compute optimization objective."""
        if not self.enabled:
            return

        import time

        training_duration = (
            time.time() - self.training_start_time if self.training_start_time else 0
        )

        # Compute final metric based on metric_name
        if self.metric_name == "mean_reward":
            if self.eval_rewards:
                # Prefer evaluation rewards as they're most reliable
                final_metric = np.mean(self.eval_rewards[-5:])  # Last 5 evaluations
            elif self.update_rewards:
                # Fall back to update rewards
                final_metric = np.mean(self.update_rewards[-10:])  # Last 10 updates
            else:
                # Last resort: episode rewards
                final_metric = (
                    np.mean(self.episode_rewards[-50:])
                    if self.episode_rewards
                    else float("-inf")
                )
        else:
            # Custom metric from final_stats
            final_metric = final_stats.get(self.metric_name, float("-inf"))

        # Set trial user attributes for analysis (only if real trial)
        if not self.is_subprocess and self.trial:
            self.trial.set_user_attr("episodes_completed", self.episodes_completed)
            self.trial.set_user_attr("updates_completed", self.updates_completed)
            self.trial.set_user_attr("training_duration", training_duration)
            self.trial.set_user_attr("best_reward", self.best_reward)
            self.trial.set_user_attr("final_metric", final_metric)

            # Log summary
            self.logger.info(
                f"Trial {self.trial.number} completed - "
                f"Final {self.metric_name}: {final_metric:.4f}, "
                f"Episodes: {self.episodes_completed}, "
                f"Duration: {training_duration:.1f}s"
            )
        else:
            # In subprocess mode, save metrics for parent process to read
            subprocess_results = {
                "final_metric": final_metric,
                "metric_name": self.metric_name,
                "episodes_completed": self.episodes_completed,
                "updates_completed": self.updates_completed,
                "training_duration": training_duration,
                "best_reward": self.best_reward,
                "eval_rewards": self.eval_rewards,
                "update_rewards": self.update_rewards,
                "episode_rewards": self.episode_rewards[-50:],  # Last 50 episodes
            }

            # Save to a file that parent process can read
            import os

            results_file = os.environ.get(
                "OPTUNA_RESULTS_FILE", "optuna_subprocess_results.json"
            )
            try:
                import json

                with open(results_file, "w") as f:
                    json.dump(subprocess_results, f)
                self.logger.info(f"Subprocess results saved to {results_file}")
            except Exception as e:
                self.logger.error(f"Failed to save subprocess results: {e}")

            # Log summary
            self.logger.info(
                f"Subprocess completed - "
                f"Final {self.metric_name}: {final_metric:.4f}, "
                f"Episodes: {self.episodes_completed}, "
                f"Duration: {training_duration:.1f}s"
            )

    def on_custom_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Handle custom optimization events."""
        if not self.enabled:
            return

        # Track any custom metrics that might be relevant for optimization
        if event_name == "custom_metric" and self.metric_name in event_data:
            metric_value = event_data[self.metric_name]
            if not self.is_subprocess and self.trial:
                self.trial.report(metric_value, self.step)
            self.step += 1
