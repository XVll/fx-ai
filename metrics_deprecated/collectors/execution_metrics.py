# metrics/collectors/execution_metrics.py - Execution and environment metrics collectors

import logging
from typing import Dict, Optional, Any
from collections import deque, defaultdict
import numpy as np

from ..core import (
    MetricCollector,
    MetricValue,
    MetricCategory,
    MetricType,
    MetricMetadata,
)


class ExecutionMetricsCollector(MetricCollector):
    """Collector for execution-related metrics"""

    def __init__(self, buffer_size: int = 1000):
        super().__init__("execution", MetricCategory.EXECUTION)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size

        # Execution tracking
        self.total_fills = 0
        self.total_volume = 0.0
        self.total_turnover = 0.0
        self.total_commission = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0

        # Recent execution data
        self.recent_fills = deque(maxlen=buffer_size)
        self.commission_history = deque(maxlen=buffer_size)
        self.slippage_history = deque(maxlen=buffer_size)
        self.fill_sizes = deque(maxlen=buffer_size)

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register execution metrics"""

        self.register_metric(
            "total_fills",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.COUNTER,
                description="Total number of fills executed",
                unit="fills",
                frequency="step",
            ),
        )

        self.register_metric(
            "total_volume",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.GAUGE,
                description="Total volume traded",
                unit="shares",
                frequency="step",
            ),
        )

        self.register_metric(
            "total_turnover",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.CURRENCY,
                description="Total dollar turnover",
                unit="USD",
                frequency="step",
            ),
        )

        self.register_metric(
            "total_commission",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.CURRENCY,
                description="Total commission paid",
                unit="USD",
                frequency="step",
            ),
        )

        self.register_metric(
            "total_fees",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.CURRENCY,
                description="Total fees paid",
                unit="USD",
                frequency="step",
            ),
        )

        self.register_metric(
            "total_slippage",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.CURRENCY,
                description="Total slippage cost",
                unit="USD",
                frequency="step",
            ),
        )

        self.register_metric(
            "avg_commission_per_share",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.CURRENCY,
                description="Average commission per share",
                unit="USD/share",
                frequency="step",
            ),
        )

        self.register_metric(
            "avg_slippage_bps",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.GAUGE,
                description="Average slippage in basis points",
                unit="bps",
                frequency="step",
            ),
        )

        self.register_metric(
            "avg_fill_size",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.GAUGE,
                description="Average fill size",
                unit="shares",
                frequency="step",
            ),
        )

        self.register_metric(
            "total_transaction_costs",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.CURRENCY,
                description="Total transaction costs (commission + fees + slippage)",
                unit="USD",
                frequency="step",
            ),
        )

        self.register_metric(
            "transaction_cost_bps",
            MetricMetadata(
                category=MetricCategory.EXECUTION,
                metric_type=MetricType.GAUGE,
                description="Transaction costs as basis points of turnover",
                unit="bps",
                frequency="step",
            ),
        )

    def collect(self) -> Dict[str, MetricValue]:
        """Collect execution metrics"""
        metrics = {}

        try:
            # Basic counters
            metrics[f"{self.category.value}.{self.name}.total_fills"] = MetricValue(
                self.total_fills
            )
            metrics[f"{self.category.value}.{self.name}.total_volume"] = MetricValue(
                self.total_volume
            )
            metrics[f"{self.category.value}.{self.name}.total_turnover"] = MetricValue(
                self.total_turnover
            )
            metrics[f"{self.category.value}.{self.name}.total_commission"] = (
                MetricValue(self.total_commission)
            )
            metrics[f"{self.category.value}.{self.name}.total_fees"] = MetricValue(
                self.total_fees
            )
            metrics[f"{self.category.value}.{self.name}.total_slippage"] = MetricValue(
                self.total_slippage
            )

            # Calculated metrics
            total_costs = self.total_commission + self.total_fees + self.total_slippage
            metrics[f"{self.category.value}.{self.name}.total_transaction_costs"] = (
                MetricValue(total_costs)
            )

            if self.total_volume > 0:
                avg_commission_per_share = self.total_commission / self.total_volume
                metrics[
                    f"{self.category.value}.{self.name}.avg_commission_per_share"
                ] = MetricValue(avg_commission_per_share)

            if self.total_turnover > 0:
                cost_bps = (total_costs / self.total_turnover) * 10000
                metrics[f"{self.category.value}.{self.name}.transaction_cost_bps"] = (
                    MetricValue(cost_bps)
                )

            if self.slippage_history and self.total_turnover > 0:
                avg_slippage_bps = np.mean(self.slippage_history)
                metrics[f"{self.category.value}.{self.name}.avg_slippage_bps"] = (
                    MetricValue(avg_slippage_bps)
                )

            if self.fill_sizes:
                avg_fill_size = np.mean(self.fill_sizes)
                metrics[f"{self.category.value}.{self.name}.avg_fill_size"] = (
                    MetricValue(avg_fill_size)
                )

        except Exception as e:
            self.logger.debug(f"Error collecting execution metrics: {e}")

        return metrics

    def record_fill(self, fill_data: Dict[str, Any]):
        """Record a fill execution"""
        try:
            quantity = fill_data.get("executed_quantity", 0)
            price = fill_data.get("executed_price", 0)
            commission = fill_data.get("commission", 0)
            fees = fill_data.get("fees", 0)
            slippage = fill_data.get("slippage_cost_total", 0)

            # Update totals
            self.total_fills += 1
            self.total_volume += quantity
            turnover = quantity * price
            self.total_turnover += turnover
            self.total_commission += commission
            self.total_fees += fees
            self.total_slippage += slippage

            # Add to history
            self.recent_fills.append(fill_data)
            self.commission_history.append(commission)
            self.fill_sizes.append(quantity)

            # Calculate slippage in bps if possible
            if turnover > 0:
                slippage_bps = (slippage / turnover) * 10000
                self.slippage_history.append(slippage_bps)

        except Exception as e:
            self.logger.debug(f"Error recording fill: {e}")

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)


class EnvironmentMetricsCollector(MetricCollector):
    """Collector for environment-related metrics"""

    def __init__(self, buffer_size: int = 1000):
        super().__init__("environment", MetricCategory.ENVIRONMENT)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size

        # State tracking
        self.total_steps = 0
        self.total_episodes = 0
        # self.invalid_actions = 0  # Removed - action masking prevents invalid actions

        # Current episode state
        self.current_step = 0
        self.max_steps = 0
        self.episode_number = 0
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0

        # Reward tracking
        self.step_rewards = deque(maxlen=buffer_size)
        self.episode_rewards = deque(maxlen=buffer_size)
        self.reward_components = defaultdict(list)

        # Action tracking
        self.action_counts = defaultdict(int)
        self.action_rewards = defaultdict(list)

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register environment metrics"""

        self.register_metric(
            "total_env_steps",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.COUNTER,
                description="Total environment steps",
                unit="steps",
                frequency="step",
            ),
        )

        # Episode state metrics
        self.register_metric(
            "current_step",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Current step in episode",
                unit="steps",
                frequency="step",
            ),
        )

        self.register_metric(
            "max_steps",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Maximum steps per episode",
                unit="steps",
                frequency="episode",
            ),
        )

        self.register_metric(
            "episode_number",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.COUNTER,
                description="Current episode number",
                unit="episodes",
                frequency="episode",
            ),
        )

        self.register_metric(
            "cumulative_reward",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Cumulative episode reward",
                unit="reward",
                frequency="step",
            ),
        )

        self.register_metric(
            "step_reward",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Current step reward",
                unit="reward",
                frequency="step",
            ),
        )

        self.register_metric(
            "step_reward_mean",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Mean step reward (recent)",
                unit="reward",
                frequency="step",
            ),
        )

        self.register_metric(
            "episode_reward_current",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Current episode cumulative reward",
                unit="reward",
                frequency="step",
            ),
        )

        # Invalid action tracking removed - action masking prevents invalid actions
        # self.register_metric("invalid_action_rate", MetricMetadata(
        #     category=MetricCategory.ENVIRONMENT,
        #     metric_type=MetricType.PERCENTAGE,
        #     description="Rate of invalid actions",
        #     unit="%",
        #     frequency="step"
        # ))

        # Action distribution metrics
        self.register_metric(
            "action_hold_pct",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.PERCENTAGE,
                description="Percentage of HOLD actions",
                unit="%",
                frequency="episode",
            ),
        )

        self.register_metric(
            "action_buy_pct",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.PERCENTAGE,
                description="Percentage of BUY actions",
                unit="%",
                frequency="episode",
            ),
        )

        self.register_metric(
            "action_sell_pct",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.PERCENTAGE,
                description="Percentage of SELL actions",
                unit="%",
                frequency="episode",
            ),
        )

        # Action count metrics
        self.register_metric(
            "action_hold_count",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.COUNTER,
                description="Total number of HOLD actions",
                unit="count",
                frequency="session",
            ),
        )

        self.register_metric(
            "action_buy_count",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.COUNTER,
                description="Total number of BUY actions",
                unit="count",
                frequency="session",
            ),
        )

        self.register_metric(
            "action_sell_count",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.COUNTER,
                description="Total number of SELL actions",
                unit="count",
                frequency="session",
            ),
        )

        # Reward component metrics (will be registered dynamically)
        self.register_metric(
            "reward_equity_change",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Reward component: equity change",
                unit="reward",
                frequency="step",
            ),
        )

        self.register_metric(
            "reward_realized_pnl",
            MetricMetadata(
                category=MetricCategory.ENVIRONMENT,
                metric_type=MetricType.GAUGE,
                description="Reward component: realized P&L",
                unit="reward",
                frequency="step",
            ),
        )

    def collect(self) -> Dict[str, MetricValue]:
        """Collect environment metrics"""
        metrics = {}

        try:
            # Basic counters
            metrics[f"{self.category.value}.{self.name}.total_env_steps"] = MetricValue(
                self.total_steps
            )

            # Episode state metrics
            metrics[f"{self.category.value}.{self.name}.current_step"] = MetricValue(
                self.current_step
            )
            metrics[f"{self.category.value}.{self.name}.max_steps"] = MetricValue(
                self.max_steps
            )
            metrics[f"{self.category.value}.{self.name}.episode_number"] = MetricValue(
                self.episode_number
            )
            metrics[f"{self.category.value}.{self.name}.cumulative_reward"] = (
                MetricValue(self.cumulative_reward)
            )

            # Step rewards
            if self.step_rewards:
                current_reward = self.step_rewards[-1]
                mean_reward = np.mean(self.step_rewards)
                metrics[f"{self.category.value}.{self.name}.step_reward"] = MetricValue(
                    current_reward
                )
                metrics[f"{self.category.value}.{self.name}.step_reward_mean"] = (
                    MetricValue(mean_reward)
                )
            elif self.last_step_reward != 0.0:
                # Fallback to last step reward if no deque data
                metrics[f"{self.category.value}.{self.name}.step_reward"] = MetricValue(
                    self.last_step_reward
                )

            # Invalid action tracking removed - action masking prevents invalid actions
            # if self.total_steps > 0:
            #     invalid_rate = (self.invalid_actions / self.total_steps) * 100
            #     metrics[f"{self.category.value}.{self.name}.invalid_action_rate"] = MetricValue(invalid_rate)

            # Action distribution
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                hold_pct = (self.action_counts.get("HOLD", 0) / total_actions) * 100
                buy_pct = (self.action_counts.get("BUY", 0) / total_actions) * 100
                sell_pct = (self.action_counts.get("SELL", 0) / total_actions) * 100

                metrics[f"{self.category.value}.{self.name}.action_hold_pct"] = (
                    MetricValue(hold_pct)
                )
                metrics[f"{self.category.value}.{self.name}.action_buy_pct"] = (
                    MetricValue(buy_pct)
                )
                metrics[f"{self.category.value}.{self.name}.action_sell_pct"] = (
                    MetricValue(sell_pct)
                )

                # Raw action counts
                metrics[f"{self.category.value}.{self.name}.action_hold_count"] = (
                    MetricValue(self.action_counts.get("HOLD", 0))
                )
                metrics[f"{self.category.value}.{self.name}.action_buy_count"] = (
                    MetricValue(self.action_counts.get("BUY", 0))
                )
                metrics[f"{self.category.value}.{self.name}.action_sell_count"] = (
                    MetricValue(self.action_counts.get("SELL", 0))
                )

        except Exception as e:
            self.logger.debug(f"Error collecting environment metrics: {e}")

        return metrics

    def record_step(
        self,
        reward: float,
        action: str,
        is_invalid: bool = False,
        reward_components: Optional[Dict[str, float]] = None,
        episode_reward: Optional[float] = None,
    ):
        """Record a step in the environment"""
        self.total_steps += 1
        self.step_rewards.append(reward)

        if action:
            self.action_counts[action] += 1
            self.action_rewards[action].append(reward)

        # Invalid action tracking removed - action masking prevents invalid actions
        # if is_invalid:
        #     self.invalid_actions += 1

        # Record reward components
        if reward_components:
            for component, value in reward_components.items():
                self.reward_components[component].append(value)

        # Update current episode reward if provided
        if episode_reward is not None:
            self._current_episode_reward = episode_reward

    def record_episode_end(self, episode_reward: float):
        """Record the end of an episode"""
        self.total_episodes += 1
        self.episode_rewards.append(episode_reward)

        # Reset episode-level tracking
        self._current_episode_reward = 0.0

    def get_reward_component_metrics(self) -> Dict[str, MetricValue]:
        """Get metrics for reward components"""
        metrics = {}

        for component, values in self.reward_components.items():
            if values:
                # Use recent value
                recent_value = values[-1] if values else 0
                metric_name = f"reward_{component}"

                # Register metric if not exists
                if not hasattr(self, f"_registered_{metric_name}"):
                    self.register_metric(
                        metric_name,
                        MetricMetadata(
                            category=MetricCategory.ENVIRONMENT,
                            metric_type=MetricType.GAUGE,
                            description=f"Reward component: {component}",
                            unit="reward",
                            frequency="step",
                        ),
                    )
                    setattr(self, f"_registered_{metric_name}", True)

                full_name = f"{self.category.value}.{self.name}.{metric_name}"
                metrics[full_name] = MetricValue(recent_value)

        return metrics

    def get_action_efficiency_metrics(self) -> Dict[str, MetricValue]:
        """Get action efficiency metrics"""
        metrics = {}

        for action, rewards in self.action_rewards.items():
            if rewards:
                mean_reward = np.mean(rewards)
                positive_rate = (sum(1 for r in rewards if r > 0) / len(rewards)) * 100

                # Register metrics
                for metric_type, value in [
                    ("efficiency", mean_reward),
                    ("success_rate", positive_rate),
                ]:
                    metric_name = f"action_{action.lower()}_{metric_type}"

                    if not hasattr(self, f"_registered_{metric_name}"):
                        self.register_metric(
                            metric_name,
                            MetricMetadata(
                                category=MetricCategory.ENVIRONMENT,
                                metric_type=MetricType.GAUGE
                                if metric_type == "efficiency"
                                else MetricType.PERCENTAGE,
                                description=f"{action} action {metric_type}",
                                unit="reward" if metric_type == "efficiency" else "%",
                                frequency="episode",
                            ),
                        )
                        setattr(self, f"_registered_{metric_name}", True)

                    full_name = f"{self.category.value}.{self.name}.{metric_name}"
                    metrics[full_name] = MetricValue(value)

        return metrics

    def update_episode_state(
        self,
        current_step: int,
        max_steps: int,
        episode_number: int,
        cumulative_reward: float,
        step_reward: float = 0.0,
    ):
        """Update episode state tracking"""
        self.current_step = current_step
        self.max_steps = max_steps
        self.episode_number = episode_number
        self.cumulative_reward = cumulative_reward
        self.last_step_reward = step_reward

        # Update step rewards deque
        if step_reward != 0.0:
            self.step_rewards.append(step_reward)

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)


class SystemMetricsCollector(MetricCollector):
    """Collector for system performance metrics"""

    def __init__(self):
        super().__init__("system", MetricCategory.SYSTEM)
        self.logger = logging.getLogger(__name__)

        # System tracking
        self.start_time = None
        self.memory_usage_history = deque(maxlen=100)

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register system metrics"""

        self.register_metric(
            "uptime_seconds",
            MetricMetadata(
                category=MetricCategory.SYSTEM,
                metric_type=MetricType.TIME,
                description="System uptime in seconds",
                unit="seconds",
                frequency="manual",
            ),
        )

        self.register_metric(
            "memory_usage_mb",
            MetricMetadata(
                category=MetricCategory.SYSTEM,
                metric_type=MetricType.GAUGE,
                description="Memory usage in MB",
                unit="MB",
                frequency="manual",
            ),
        )

        self.register_metric(
            "cpu_usage_pct",
            MetricMetadata(
                category=MetricCategory.SYSTEM,
                metric_type=MetricType.PERCENTAGE,
                description="CPU usage percentage",
                unit="%",
                frequency="manual",
            ),
        )

    def collect(self) -> Dict[str, MetricValue]:
        """Collect system metrics"""
        metrics = {}

        try:
            import psutil
            import time

            # Uptime
            if self.start_time:
                uptime = time.time() - self.start_time
                metrics[f"{self.category.value}.{self.name}.uptime_seconds"] = (
                    MetricValue(uptime)
                )

            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            metrics[f"{self.category.value}.{self.name}.memory_usage_mb"] = MetricValue(
                memory_mb
            )
            self.memory_usage_history.append(memory_mb)

            # CPU usage
            cpu_pct = psutil.cpu_percent(interval=None)
            metrics[f"{self.category.value}.{self.name}.cpu_usage_pct"] = MetricValue(
                cpu_pct
            )

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.debug(f"Error collecting system metrics: {e}")

        return metrics

    def start_tracking(self):
        """Start system tracking"""
        import time

        self.start_time = time.time()

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)
