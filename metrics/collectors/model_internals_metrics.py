# metrics/collectors/model_internals_metrics.py - Model internals and diagnostics collector

import logging
from typing import Dict, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from ..core import MetricCollector, MetricValue, MetricCategory, MetricType, MetricMetadata


class ModelInternalsCollector(MetricCollector):
    """Collector for model internals and diagnostic metrics"""
    
    def __init__(self, buffer_size: int = 100):
        super().__init__("internals", MetricCategory.MODEL)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size
        
        # Attention weights tracking
        self.attention_weights_history = deque(maxlen=buffer_size)
        self.last_attention_weights = None
        
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
        self.register_metric("attention_entropy", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.GAUGE,
            description="Entropy of attention weights (measure of focus)",
            unit="nats",
            frequency="step"
        ))
        
        self.register_metric("attention_max_weight", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.GAUGE,
            description="Maximum attention weight value",
            unit="weight",
            frequency="step"
        ))
        
        self.register_metric("attention_focus_branch", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.GAUGE,
            description="Branch with highest attention (0=HF, 1=MF, 2=LF, 3=Portfolio, 4=Static)",
            unit="branch_id",
            frequency="step"
        ))
        
        # Action probability metrics
        self.register_metric("action_entropy", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.GAUGE,
            description="Entropy of action probability distribution",
            unit="nats",
            frequency="step"
        ))
        
        self.register_metric("action_confidence", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.PERCENTAGE,
            description="Confidence in selected action (max probability)",
            unit="%",
            frequency="step"
        ))
        
        self.register_metric("action_type_entropy", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.GAUGE,
            description="Entropy of action type distribution",
            unit="nats",
            frequency="step"
        ))
        
        self.register_metric("action_size_entropy", MetricMetadata(
            category=MetricCategory.MODEL,
            metric_type=MetricType.GAUGE,
            description="Entropy of action size distribution",
            unit="nats",
            frequency="step"
        ))
        
        # Feature distribution metrics for each branch
        for branch in ['hf', 'mf', 'lf', 'portfolio', 'static']:
            self.register_metric(f"feature_{branch}_mean", MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description=f"Mean value of {branch} features",
                unit="value",
                frequency="step"
            ))
            
            self.register_metric(f"feature_{branch}_std", MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description=f"Standard deviation of {branch} features",
                unit="value",
                frequency="step"
            ))
            
            self.register_metric(f"feature_{branch}_sparsity", MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.PERCENTAGE,
                description=f"Sparsity of {branch} features (% zeros)",
                unit="%",
                frequency="step"
            ))
    
    def collect(self) -> Dict[str, MetricValue]:
        """Collect model internals metrics"""
        metrics = {}
        
        try:
            # Attention metrics
            if self.last_attention_weights is not None:
                # Calculate attention entropy
                attention_probs = self.last_attention_weights
                entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8))
                metrics[f"{self.category.value}.{self.name}.attention_entropy"] = MetricValue(entropy)
                
                # Max weight and focus branch
                max_weight = np.max(attention_probs)
                focus_branch = np.argmax(attention_probs)
                metrics[f"{self.category.value}.{self.name}.attention_max_weight"] = MetricValue(max_weight)
                metrics[f"{self.category.value}.{self.name}.attention_focus_branch"] = MetricValue(focus_branch)
            
            # Action probability metrics
            if self.last_action_probs is not None:
                if isinstance(self.last_action_probs, tuple) and len(self.last_action_probs) == 2:
                    # Discrete action with type and size
                    type_probs, size_probs = self.last_action_probs
                    
                    # Type entropy
                    type_entropy = -np.sum(type_probs * np.log(type_probs + 1e-8))
                    metrics[f"{self.category.value}.{self.name}.action_type_entropy"] = MetricValue(type_entropy)
                    
                    # Size entropy
                    size_entropy = -np.sum(size_probs * np.log(size_probs + 1e-8))
                    metrics[f"{self.category.value}.{self.name}.action_size_entropy"] = MetricValue(size_entropy)
                    
                    # Combined entropy (average)
                    combined_entropy = (type_entropy + size_entropy) / 2
                    metrics[f"{self.category.value}.{self.name}.action_entropy"] = MetricValue(combined_entropy)
                    
                    # Confidence (max probability)
                    confidence = max(np.max(type_probs), np.max(size_probs)) * 100
                    metrics[f"{self.category.value}.{self.name}.action_confidence"] = MetricValue(confidence)
                else:
                    # Single action distribution
                    probs = self.last_action_probs
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    metrics[f"{self.category.value}.{self.name}.action_entropy"] = MetricValue(entropy)
                    
                    confidence = np.max(probs) * 100
                    metrics[f"{self.category.value}.{self.name}.action_confidence"] = MetricValue(confidence)
            
            # Feature statistics
            for branch, stats in self.feature_stats.items():
                if stats:
                    metrics[f"{self.category.value}.{self.name}.feature_{branch}_mean"] = MetricValue(stats['mean'])
                    metrics[f"{self.category.value}.{self.name}.feature_{branch}_std"] = MetricValue(stats['std'])
                    metrics[f"{self.category.value}.{self.name}.feature_{branch}_sparsity"] = MetricValue(stats['sparsity'])
                    
        except Exception as e:
            self.logger.debug(f"Error collecting model internals metrics: {e}")
            
        return metrics
    
    def update_attention_weights(self, weights: np.ndarray):
        """Update attention weights from the fusion layer"""
        self.last_attention_weights = weights
        self.attention_weights_history.append(weights)
    
    def update_action_probabilities(self, action_probs: Any):
        """Update action probability distribution"""
        if torch.is_tensor(action_probs):
            action_probs = action_probs.detach().cpu().numpy()
        elif isinstance(action_probs, tuple):
            # Handle tuple of tensors
            action_probs = tuple(
                p.detach().cpu().numpy() if torch.is_tensor(p) else p 
                for p in action_probs
            )
        
        self.last_action_probs = action_probs
        self.action_probs_history.append(action_probs)
    
    def update_feature_statistics(self, features: Dict[str, torch.Tensor]):
        """Update feature distribution statistics"""
        for branch, feat_tensor in features.items():
            if torch.is_tensor(feat_tensor):
                feat_array = feat_tensor.detach().cpu().numpy().flatten()
                
                self.feature_stats[branch] = {
                    'mean': float(np.mean(feat_array)),
                    'std': float(np.std(feat_array)),
                    'sparsity': float(np.sum(feat_array == 0) / len(feat_array) * 100)
                }
    
    def get_attention_weights_for_visualization(self) -> Optional[np.ndarray]:
        """Get recent attention weights for visualization"""
        if len(self.attention_weights_history) > 0:
            # Return average of recent weights
            return np.mean(list(self.attention_weights_history), axis=0)
        return None
    
    def get_action_distribution_history(self) -> List[Any]:
        """Get action probability history for analysis"""
        return list(self.action_probs_history)
    
    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)