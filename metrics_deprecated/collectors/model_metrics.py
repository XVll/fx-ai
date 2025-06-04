import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..core import (
    MetricCollector,
    MetricValue,
    MetricCategory,
    MetricType,
    MetricMetadata,
)


class ModelMetricsCollector(MetricCollector):
    """Collector for model-related metrics"""

    def __init__(self, model: Optional[nn.Module] = None):
        super().__init__("model", MetricCategory.MODEL)
        self.model = model
        self.logger = logging.getLogger(__name__)

        # Register metrics
        self._register_metrics()

        # Track previous values for rate calculations
        self._previous_values = {}

    def _register_metrics(self):
        """Register all model metrics"""

        # Loss metrics
        self.register_metric(
            "actor_loss",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Policy/Actor loss value",
                unit="loss",
                frequency="update",
            ),
        )

        self.register_metric(
            "critic_loss",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Value/Critic loss value",
                unit="loss",
                frequency="update",
            ),
        )

        self.register_metric(
            "total_loss",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Combined total loss",
                unit="loss",
                frequency="update",
            ),
        )

        self.register_metric(
            "entropy",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Policy entropy",
                unit="nats",
                frequency="update",
            ),
        )

        # Gradient metrics
        self.register_metric(
            "gradient_norm",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Global gradient norm",
                unit="norm",
                frequency="update",
            ),
        )

        self.register_metric(
            "gradient_max",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Maximum gradient value",
                unit="grad",
                frequency="update",
            ),
        )

        # Parameter metrics
        self.register_metric(
            "param_norm",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Total parameter norm",
                unit="norm",
                frequency="update",
            ),
        )

        self.register_metric(
            "param_count",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.COUNTER,
                description="Total number of parameters",
                unit="params",
                frequency="manual",
            ),
        )

        # PPO-specific metrics
        self.register_metric(
            "clip_fraction",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.PERCENTAGE,
                description="Fraction of samples that were clipped",
                unit="%",
                frequency="update",
            ),
        )

        self.register_metric(
            "approx_kl",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Approximate KL divergence",
                unit="kl",
                frequency="update",
            ),
        )

        self.register_metric(
            "explained_variance",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.PERCENTAGE,
                description="Value function explained variance",
                unit="%",
                frequency="update",
            ),
        )

        # Learning rate
        self.register_metric(
            "learning_rate",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Current learning rate",
                unit="lr",
                frequency="update",
            ),
        )

    def set_model(self, model: nn.Module):
        """Set the model to track"""
        self.model = model
        if model is not None:
            self.logger.info(
                f"Model set for metrics collection: {type(model).__name__}"
            )

    def collect(self) -> Dict[str, MetricValue]:
        """Collect current model metrics"""
        metrics = {}

        if self.model is None:
            return metrics

        try:
            # Parameter count (only calculate once)
            if "param_count" not in self._previous_values:
                param_count = sum(p.numel() for p in self.model.parameters())
                metrics[f"{self.category.value}.{self.name}.param_count"] = MetricValue(
                    param_count
                )
                self._previous_values["param_count"] = param_count

            # Parameter norm
            param_norm = self._calculate_parameter_norm()
            if param_norm is not None:
                metrics[f"{self.category.value}.{self.name}.param_norm"] = MetricValue(
                    param_norm
                )

            # Gradient metrics (if gradients are available)
            grad_norm, grad_max = self._calculate_gradient_metrics()
            if grad_norm is not None:
                metrics[f"{self.category.value}.{self.name}.gradient_norm"] = (
                    MetricValue(grad_norm)
                )
            if grad_max is not None:
                metrics[f"{self.category.value}.{self.name}.gradient_max"] = (
                    MetricValue(grad_max)
                )

        except Exception as e:
            self.logger.debug(f"Error collecting model metrics: {e}")

        return metrics

    def record_loss_metrics(
        self,
        actor_loss: float,
        critic_loss: float,
        entropy: float,
        total_loss: Optional[float] = None,
    ):
        """Record loss metrics manually"""
        metrics = {
            f"{self.category.value}.{self.name}.actor_loss": MetricValue(actor_loss),
            f"{self.category.value}.{self.name}.critic_loss": MetricValue(critic_loss),
            f"{self.category.value}.{self.name}.entropy": MetricValue(entropy),
        }

        if total_loss is None:
            total_loss = actor_loss + critic_loss
        metrics[f"{self.category.value}.{self.name}.total_loss"] = MetricValue(
            total_loss
        )

        return metrics

    def record_ppo_metrics(
        self, clip_fraction: float, approx_kl: float, explained_variance: float
    ):
        """Record PPO-specific metrics"""
        metrics = {
            f"{self.category.value}.{self.name}.clip_fraction": MetricValue(
                clip_fraction * 100
            ),
            f"{self.category.value}.{self.name}.approx_kl": MetricValue(approx_kl),
            f"{self.category.value}.{self.name}.explained_variance": MetricValue(
                explained_variance * 100
            ),
        }

        return metrics

    def record_learning_rate(self, learning_rate: float):
        """Record current learning rate"""
        return {
            f"{self.category.value}.{self.name}.learning_rate": MetricValue(
                learning_rate
            )
        }

    def _calculate_parameter_norm(self) -> Optional[float]:
        """Calculate the norm of all model parameters"""
        if self.model is None:
            return None

        try:
            total_norm = 0.0
            for param in self.model.parameters():
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            return total_norm
        except Exception as e:
            self.logger.debug(f"Error calculating parameter norm: {e}")
            return None

    def _calculate_gradient_metrics(self) -> tuple[Optional[float], Optional[float]]:
        """Calculate gradient norm and max gradient"""
        if self.model is None:
            return None, None

        try:
            total_norm = 0.0
            max_grad = 0.0
            param_count = 0

            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    max_grad = max(max_grad, param.grad.data.abs().max().item())
                    param_count += 1

            if param_count == 0:
                return None, None

            total_norm = total_norm ** (1.0 / 2)
            return total_norm, max_grad

        except Exception as e:
            self.logger.debug(f"Error calculating gradient metrics: {e}")
            return None, None

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)


class OptimizerMetricsCollector(MetricCollector):
    """Collector for optimizer-related metrics"""

    def __init__(self, optimizer: Optional[torch.optim.Optimizer] = None):
        super().__init__("optimizer", MetricCategory.MODEL)
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register optimizer metrics"""

        self.register_metric(
            "learning_rate",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Current learning rate",
                unit="lr",
                frequency="update",
            ),
        )

        self.register_metric(
            "momentum",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Momentum parameter",
                unit="momentum",
                frequency="update",
            ),
        )

        self.register_metric(
            "weight_decay",
            MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Weight decay parameter",
                unit="decay",
                frequency="update",
            ),
        )

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer to track"""
        self.optimizer = optimizer
        self.logger.info(
            f"Optimizer set for metrics collection: {type(optimizer).__name__}"
        )

    def collect(self) -> Dict[str, MetricValue]:
        """Collect optimizer metrics"""
        metrics = {}

        if self.optimizer is None:
            return metrics

        try:
            # Get the first parameter group (assuming uniform settings)
            if self.optimizer.param_groups:
                param_group = self.optimizer.param_groups[0]

                # Learning rate
                lr = param_group.get("lr", 0.0)
                metrics[f"{self.category.value}.{self.name}.learning_rate"] = (
                    MetricValue(lr)
                )

                # Momentum (if available)
                momentum = param_group.get("momentum")
                if momentum is not None:
                    metrics[f"{self.category.value}.{self.name}.momentum"] = (
                        MetricValue(momentum)
                    )

                # Weight decay
                weight_decay = param_group.get("weight_decay", 0.0)
                metrics[f"{self.category.value}.{self.name}.weight_decay"] = (
                    MetricValue(weight_decay)
                )

        except Exception as e:
            self.logger.debug(f"Error collecting optimizer metrics: {e}")

        return metrics

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)
