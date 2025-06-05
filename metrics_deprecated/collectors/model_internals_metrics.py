"""Model Internals Metrics Collector - Simplified version without SHAP attribution"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from ..core import (
    MetricCollector,
    MetricCategory,
    MetricType,
    MetricMetadata,
)


class ModelInternalsCollector(MetricCollector):
    """Collector for model internals and diagnostic metrics"""

    def __init__(
        self,
        buffer_size: int = 100,
        model: Optional[torch.nn.Module] = None,
        feature_names: Optional[Dict[str, List[str]]] = None,
        model_config: Optional[Any] = None,
    ):
        super().__init__("internals", MetricCategory.MODEL)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size
        self.model = model
        self.feature_names = feature_names or {}

        # Action probabilities tracking
        self.action_probs_history = deque(maxlen=buffer_size)
        self.last_action_probs = None

        # Feature statistics tracking
        self.feature_stats = {}

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register model internals metrics"""

        # Attention metrics
        self.register_metric(
            "attention_entropy",
            MetricMetadata(
                category=MetricCategory.MODEL,
                type=MetricType.GAUGE,
                description="Entropy of attention weights",
                tags=["attention", "entropy"],
            ),
        )

        # Gradient metrics
        self.register_metric(
            "gradient_norm",
            MetricMetadata(
                category=MetricCategory.MODEL,
                type=MetricType.GAUGE,
                description="L2 norm of model gradients",
                tags=["gradients", "norm"],
            ),
        )

        # Action probability metrics
        self.register_metric(
            "action_prob_entropy",
            MetricMetadata(
                category=MetricCategory.MODEL,
                type=MetricType.GAUGE,
                description="Entropy of action probabilities",
                tags=["actions", "entropy"],
            ),
        )

        # Feature statistics
        self.register_metric(
            "feature_mean",
            MetricMetadata(
                category=MetricCategory.MODEL,
                type=MetricType.GAUGE,
                description="Mean of input features",
                tags=["features", "statistics"],
            ),
        )

        self.register_metric(
            "feature_std",
            MetricMetadata(
                category=MetricCategory.MODEL,
                type=MetricType.GAUGE,
                description="Standard deviation of input features",
                tags=["features", "statistics"],
            ),
        )

    def collect(self) -> Dict[str, float]:
        """Collect current model internals metrics"""
        metrics = {}

        # Add any tracked metrics
        if self.last_action_probs is not None:
            # Calculate action probability entropy
            probs = torch.softmax(self.last_action_probs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
            metrics["action_prob_entropy"] = entropy

        # Add feature statistics if available
        for branch, stats in self.feature_stats.items():
            if "mean" in stats:
                metrics[f"feature_mean_{branch}"] = stats["mean"]
            if "std" in stats:
                metrics[f"feature_std_{branch}"] = stats["std"]

        return metrics

    def on_model_forward(self, inputs: Dict[str, torch.Tensor], outputs: Any):
        """Called after model forward pass"""
        try:
            # Store action probabilities if available
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                action_logits = outputs[0]
                if torch.is_tensor(action_logits):
                    self.last_action_probs = action_logits.detach()
                    self.action_probs_history.append(action_logits.detach().cpu())

            # Update feature statistics
            self._update_feature_statistics(inputs)

        except Exception as e:
            self.logger.warning(f"Error in model forward tracking: {e}")

    def on_gradient_update(self, model: torch.nn.Module):
        """Called after gradient update"""
        try:
            # Calculate gradient norm
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            
            # Store as metric
            self.update_metric("gradient_norm", total_norm)

        except Exception as e:
            self.logger.warning(f"Error in gradient tracking: {e}")

    def _update_feature_statistics(self, features: Dict[str, torch.Tensor]):
        """Update feature statistics for monitoring"""
        try:
            for branch, tensor in features.items():
                if torch.is_tensor(tensor) and tensor.numel() > 0:
                    mean_val = tensor.mean().item()
                    std_val = tensor.std().item()
                    
                    self.feature_stats[branch] = {
                        "mean": mean_val,
                        "std": std_val,
                        "last_updated": time.time()
                    }
        except Exception as e:
            self.logger.warning(f"Error updating feature statistics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model internals"""
        summary = {
            "action_probs_history_length": len(self.action_probs_history),
            "feature_branches_tracked": list(self.feature_stats.keys()),
            "last_gradient_norm": self.get_metric_value("gradient_norm"),
            "last_action_entropy": self.get_metric_value("action_prob_entropy"),
        }
        
        # Add feature statistics summary
        if self.feature_stats:
            summary["feature_statistics"] = {}
            for branch, stats in self.feature_stats.items():
                summary["feature_statistics"][branch] = {
                    "mean": stats.get("mean", 0.0),
                    "std": stats.get("std", 0.0)
                }
        
        return summary