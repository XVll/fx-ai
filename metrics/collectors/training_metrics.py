# metrics/collectors/training_metrics.py - Training process metrics collector

import logging
import time
from typing import Dict, Optional, Any, List
from collections import deque
import numpy as np

from ..core import MetricCollector, MetricValue, MetricCategory, MetricType, MetricMetadata


class TrainingMetricsCollector(MetricCollector):
    """Collector for training process metrics"""

    def __init__(self, buffer_size: int = 100):
        super().__init__("process", MetricCategory.TRAINING)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size

        # State tracking
        self.current_episode = 0
        self.current_step = 0
        self.current_update = 0
        self.training_start_time = None
        self.episode_start_time = None
        self.update_start_time = None

        # Episode tracking
        self.episode_rewards = deque(maxlen=buffer_size)
        self.episode_lengths = deque(maxlen=buffer_size)
        self.episode_times = deque(maxlen=buffer_size)

        # Update tracking
        self.update_times = deque(maxlen=buffer_size)
        self.rollout_times = deque(maxlen=buffer_size)

        # Performance tracking
        self.steps_per_second_history = deque(maxlen=buffer_size)

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register all training metrics"""

        # Episode metrics
        self.register_metric("episode_count", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.COUNTER,
            description="Total number of completed episodes",
            unit="episodes",
            frequency="episode"
        ))

        self.register_metric("episode_reward_mean", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Mean episode reward (recent)",
            unit="reward",
            aggregation="mean",
            frequency="episode"
        ))

        self.register_metric("episode_reward_std", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Standard deviation of episode rewards",
            unit="reward",
            frequency="episode"
        ))

        self.register_metric("episode_length_mean", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Mean episode length (recent)",
            unit="steps",
            aggregation="mean",
            frequency="episode"
        ))

        self.register_metric("episode_duration_mean", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.TIME,
            description="Mean episode duration",
            unit="seconds",
            aggregation="mean",
            frequency="episode"
        ))

        # Step metrics
        self.register_metric("global_step", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.COUNTER,
            description="Global step counter",
            unit="steps",
            frequency="step"
        ))

        self.register_metric("steps_per_second", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.RATE,
            description="Training steps per second",
            unit="steps/sec",
            frequency="step"
        ))

        # Update metrics
        self.register_metric("update_count", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.COUNTER,
            description="Total number of policy updates",
            unit="updates",
            frequency="update"
        ))

        self.register_metric("update_duration", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.TIME,
            description="Time taken for policy update",
            unit="seconds",
            frequency="update"
        ))

        self.register_metric("rollout_duration", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.TIME,
            description="Time taken for rollout collection",
            unit="seconds",
            frequency="update"
        ))

        # Training session metrics
        self.register_metric("training_duration", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.TIME,
            description="Total training time",
            unit="seconds",
            frequency="step"
        ))

        # Performance metrics
        self.register_metric("episodes_per_hour", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.RATE,
            description="Episodes completed per hour",
            unit="episodes/hour",
            frequency="episode"
        ))

        self.register_metric("updates_per_hour", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.RATE,
            description="Updates completed per hour",
            unit="updates/hour",
            frequency="update"
        ))

    def collect(self) -> Dict[str, MetricValue]:
        """Collect current training metrics"""
        metrics = {}
        current_time = time.time()

        try:
            # Episode metrics
            metrics[self.register_metric("episode_count", self._get_metadata("episode_count"))] = MetricValue(self.current_episode)

            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards)
                std_reward = np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0.0
                metrics[self.register_metric("episode_reward_mean", self._get_metadata("episode_reward_mean"))] = MetricValue(mean_reward)
                metrics[self.register_metric("episode_reward_std", self._get_metadata("episode_reward_std"))] = MetricValue(std_reward)

            if self.episode_lengths:
                mean_length = np.mean(self.episode_lengths)
                metrics[self.register_metric("episode_length_mean", self._get_metadata("episode_length_mean"))] = MetricValue(mean_length)

            if self.episode_times:
                mean_duration = np.mean(self.episode_times)
                metrics[self.register_metric("episode_duration_mean", self._get_metadata("episode_duration_mean"))] = MetricValue(mean_duration)

            # Step metrics
            metrics[self.register_metric("global_step", self._get_metadata("global_step"))] = MetricValue(self.current_step)

            if self.steps_per_second_history:
                sps = np.mean(self.steps_per_second_history)
                metrics[self.register_metric("steps_per_second", self._get_metadata("steps_per_second"))] = MetricValue(sps)

            # Update metrics
            metrics[self.register_metric("update_count", self._get_metadata("update_count"))] = MetricValue(self.current_update)

            if self.update_times:
                mean_update_time = np.mean(self.update_times)
                metrics[self.register_metric("update_duration", self._get_metadata("update_duration"))] = MetricValue(mean_update_time)

            if self.rollout_times:
                mean_rollout_time = np.mean(self.rollout_times)
                metrics[self.register_metric("rollout_duration", self._get_metadata("rollout_duration"))] = MetricValue(mean_rollout_time)

            # Training session metrics
            if self.training_start_time:
                training_duration = current_time - self.training_start_time
                metrics[self.register_metric("training_duration", self._get_metadata("training_duration"))] = MetricValue(training_duration)

                # Performance rates
                if training_duration > 0:
                    episodes_per_hour = (self.current_episode / training_duration) * 3600
                    metrics[self.register_metric("episodes_per_hour", self._get_metadata("episodes_per_hour"))] = MetricValue(episodes_per_hour)

                    updates_per_hour = (self.current_update / training_duration) * 3600
                    metrics[self.register_metric("updates_per_hour", self._get_metadata("updates_per_hour"))] = MetricValue(updates_per_hour)

        except Exception as e:
            self.logger.debug(f"Error collecting training metrics: {e}")

        return metrics

    def start_training(self):
        """Mark the start of training"""
        self.training_start_time = time.time()
        self.logger.info("Training started")

    def start_episode(self):
        """Mark the start of an episode"""
        self.episode_start_time = time.time()

    def end_episode(self, reward: float, length: int):
        """Mark the end of an episode"""
        self.current_episode += 1
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if self.episode_start_time:
            duration = time.time() - self.episode_start_time
            self.episode_times.append(duration)
            self.episode_start_time = None

    def start_update(self):
        """Mark the start of a policy update"""
        self.update_start_time = time.time()

    def end_update(self):
        """Mark the end of a policy update"""
        self.current_update += 1
        if self.update_start_time:
            duration = time.time() - self.update_start_time
            self.update_times.append(duration)
            self.update_start_time = None

    def record_rollout_time(self, duration: float):
        """Record time taken for rollout collection"""
        self.rollout_times.append(duration)

    def update_step(self, step: int):
        """Update the current step count"""
        if self.current_step > 0:
            # Calculate steps per second
            current_time = time.time()
            if hasattr(self, '_last_step_time'):
                time_diff = current_time - self._last_step_time
                step_diff = step - self.current_step
                if time_diff > 0:
                    sps = step_diff / time_diff
                    self.steps_per_second_history.append(sps)
            self._last_step_time = current_time

        self.current_step = step

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress"""
        summary = {
            "episodes": self.current_episode,
            "steps": self.current_step,
            "updates": self.current_update,
        }

        if self.episode_rewards:
            summary["mean_reward"] = np.mean(self.episode_rewards)
            summary["reward_trend"] = "improving" if len(self.episode_rewards) > 10 and \
                                                     np.mean(list(self.episode_rewards)[-5:]) > np.mean(list(self.episode_rewards)[-10:-5]) else "stable"

        if self.training_start_time:
            summary["training_time"] = time.time() - self.training_start_time

        return summary

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)


class EvaluationMetricsCollector(MetricCollector):
    """Collector for evaluation metrics"""

    def __init__(self, buffer_size: int = 50):
        super().__init__("evaluation", MetricCategory.TRAINING)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size

        # Evaluation tracking
        self.eval_rewards = deque(maxlen=buffer_size)
        self.eval_lengths = deque(maxlen=buffer_size)
        self.eval_count = 0
        self.is_evaluating = False

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register evaluation metrics"""

        self.register_metric("eval_reward_mean", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Mean evaluation reward",
            unit="reward",
            frequency="manual"
        ))

        self.register_metric("eval_reward_std", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Standard deviation of evaluation rewards",
            unit="reward",
            frequency="manual"
        ))

        self.register_metric("eval_length_mean", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Mean evaluation episode length",
            unit="steps",
            frequency="manual"
        ))

        self.register_metric("eval_count", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.COUNTER,
            description="Number of evaluation runs",
            unit="evaluations",
            frequency="manual"
        ))

    def collect(self) -> Dict[str, MetricValue]:
        """Collect evaluation metrics"""
        metrics = {}

        if not self.eval_rewards:
            return metrics

        try:
            mean_reward = np.mean(self.eval_rewards)
            std_reward = np.std(self.eval_rewards) if len(self.eval_rewards) > 1 else 0.0
            mean_length = np.mean(self.eval_lengths) if self.eval_lengths else 0.0

            metrics[self.register_metric("eval_reward_mean", self._get_metadata("eval_reward_mean"))] = MetricValue(mean_reward)
            metrics[self.register_metric("eval_reward_std", self._get_metadata("eval_reward_std"))] = MetricValue(std_reward)
            metrics[self.register_metric("eval_length_mean", self._get_metadata("eval_length_mean"))] = MetricValue(mean_length)
            metrics[self.register_metric("eval_count", self._get_metadata("eval_count"))] = MetricValue(self.eval_count)

        except Exception as e:
            self.logger.debug(f"Error collecting evaluation metrics: {e}")

        return metrics

    def start_evaluation(self):
        """Mark the start of evaluation"""
        self.is_evaluating = True

    def end_evaluation(self, rewards: List[float], lengths: List[int]):
        """Mark the end of evaluation with results"""
        self.is_evaluating = False
        self.eval_count += 1

        for reward in rewards:
            self.eval_rewards.append(reward)

        for length in lengths:
            self.eval_lengths.append(length)

        self.logger.info(f"Evaluation {self.eval_count} completed: "
                         f"Mean reward: {np.mean(rewards):.2f}, "
                         f"Episodes: {len(rewards)}")

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)