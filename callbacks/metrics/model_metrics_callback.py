"""
Model internals metrics callback for WandB integration.

Tracks model-specific metrics including attention patterns, gradients,
activations, and internal model behavior.
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None

from callbacks.core.base import BaseCallback

logger = logging.getLogger(__name__)


class ModelMetricsCallback(BaseCallback):
    """
    Specialized callback for model internal metrics.
    
    Tracks:
    - Attention patterns: entropy, weights, focus
    - Gradient statistics: norms, distributions
    - Activation patterns: magnitudes, sparsity
    - Feature utilization: usage, importance
    - Model behavior: prediction confidence, uncertainty
    """
    
    def __init__(self, buffer_size: int = 500, enabled: bool = True):
        """
        Initialize model metrics callback.
        
        Args:
            buffer_size: Size of local buffers for rolling calculations
            enabled: Whether callback is active
        """
        super().__init__(name="ModelMetrics", enabled=enabled)
        
        self.buffer_size = buffer_size
        
        # Attention tracking buffers
        self.attention_entropies = deque(maxlen=buffer_size)
        self.attention_max_weights = deque(maxlen=buffer_size)
        self.attention_focus_scores = deque(maxlen=buffer_size)
        
        # Gradient tracking buffers
        self.gradient_norms = deque(maxlen=buffer_size)
        self.gradient_variances = deque(maxlen=buffer_size)
        self.gradient_max_values = deque(maxlen=buffer_size)
        
        # Activation tracking buffers
        self.activation_magnitudes = deque(maxlen=buffer_size)
        self.activation_sparsities = deque(maxlen=buffer_size)
        self.layer_activations = {}  # Dict of layer_name -> deque
        
        # Feature utilization tracking
        self.feature_importances = deque(maxlen=buffer_size)
        self.feature_correlations = deque(maxlen=buffer_size)
        self.unused_features_count = deque(maxlen=buffer_size)
        
        # Model behavior tracking
        self.prediction_confidences = deque(maxlen=buffer_size)
        self.action_entropies = deque(maxlen=buffer_size)
        self.value_function_errors = deque(maxlen=buffer_size)
        
        # Weight statistics
        self.weight_norms = {}  # Dict of layer_name -> deque
        self.weight_updates = {}  # Dict of layer_name -> deque
        
        # Performance tracking
        self.forward_passes_logged = 0
        self.backward_passes_logged = 0
        self.weight_updates_logged = 0
        
        if wandb is None:
            self.logger.warning("wandb not installed - model metrics will not be logged")
        
        self.logger.info(f"🧠 Model metrics callback initialized (buffer_size={buffer_size})")
    
    def on_custom_event(self, event_name: str, context: Dict[str, Any]) -> None:
        """Handle model-related custom events."""
        if not wandb or not wandb.run:
            return
        
        if event_name == 'model_forward':
            self._handle_model_forward(context)
        elif event_name == 'model_backward':
            self._handle_model_backward(context)
        elif event_name == 'weight_update':
            self._handle_weight_update(context)
        elif event_name == 'attention_analysis':
            self._handle_attention_analysis(context)
        elif event_name == 'feature_analysis':
            self._handle_feature_analysis(context)
    
    def _handle_model_forward(self, context: Dict[str, Any]) -> None:
        """Handle model forward pass events."""
        raw = context.get('raw', {})
        
        # Extract forward pass metrics
        attention_entropy = raw.get('attention_entropy', 0)
        attention_max_weight = raw.get('attention_max_weight', 0)
        prediction_confidence = raw.get('prediction_confidence', 0)
        action_entropy = raw.get('action_entropy', 0)
        
        # Extract activation statistics
        activation_magnitude = raw.get('activation_magnitude', 0)
        activation_sparsity = raw.get('activation_sparsity', 0)
        
        # Extract layer-specific activations
        layer_activations = raw.get('layer_activations', {})
        
        # Add to buffers
        self.attention_entropies.append(attention_entropy)
        self.attention_max_weights.append(attention_max_weight)
        self.prediction_confidences.append(prediction_confidence)
        self.action_entropies.append(action_entropy)
        self.activation_magnitudes.append(activation_magnitude)
        self.activation_sparsities.append(activation_sparsity)
        
        # Handle layer-specific activations
        for layer_name, activation_value in layer_activations.items():
            if layer_name not in self.layer_activations:
                self.layer_activations[layer_name] = deque(maxlen=self.buffer_size)
            self.layer_activations[layer_name].append(activation_value)
        
        # Calculate attention focus (inverse of entropy)
        attention_focus = 1 / (1 + attention_entropy) if attention_entropy >= 0 else 0
        self.attention_focus_scores.append(attention_focus)
        
        # Prepare metrics
        metrics = {
            # Raw model metrics
            'model/attention_entropy': attention_entropy,
            'model/attention_max_weight': attention_max_weight,
            'model/attention_focus_score': attention_focus,
            'model/prediction_confidence': prediction_confidence,
            'model/action_entropy': action_entropy,
            'model/activation_magnitude': activation_magnitude,
            'model/activation_sparsity': activation_sparsity,
        }
        
        # Add layer-specific metrics
        for layer_name, activation_value in layer_activations.items():
            metrics[f'model/layer_{layer_name}_activation'] = activation_value
        
        # Add rolling metrics
        self._add_rolling_forward_metrics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.forward_passes_logged += 1
        
        if self.forward_passes_logged % 100 == 0:
            self.logger.debug(f"🧠 Logged {self.forward_passes_logged} forward passes to WandB")
    
    def _handle_model_backward(self, context: Dict[str, Any]) -> None:
        """Handle model backward pass events."""
        raw = context.get('raw', {})
        
        # Extract gradient statistics
        gradient_norm = raw.get('gradient_norm', 0)
        gradient_variance = raw.get('gradient_variance', 0)
        gradient_max_value = raw.get('gradient_max_value', 0)
        gradient_clip_ratio = raw.get('gradient_clip_ratio', 0)
        
        # Extract layer-specific gradients
        layer_gradients = raw.get('layer_gradients', {})
        
        # Add to buffers
        self.gradient_norms.append(gradient_norm)
        self.gradient_variances.append(gradient_variance)
        self.gradient_max_values.append(gradient_max_value)
        
        # Prepare metrics
        metrics = {
            # Raw gradient metrics
            'model/gradient_norm': gradient_norm,
            'model/gradient_variance': gradient_variance,
            'model/gradient_max_value': gradient_max_value,
            'model/gradient_clip_ratio': gradient_clip_ratio,
        }
        
        # Add layer-specific gradient metrics
        for layer_name, grad_norm in layer_gradients.items():
            metrics[f'model/layer_{layer_name}_grad_norm'] = grad_norm
        
        # Add rolling gradient metrics
        self._add_rolling_backward_metrics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.backward_passes_logged += 1
    
    def _handle_weight_update(self, context: Dict[str, Any]) -> None:
        """Handle weight update events."""
        raw = context.get('raw', {})
        
        # Extract weight statistics
        weight_norms = raw.get('weight_norms', {})
        weight_updates = raw.get('weight_updates', {})
        learning_rate = raw.get('learning_rate', 0)
        
        # Track weight norms and updates per layer
        for layer_name, weight_norm in weight_norms.items():
            if layer_name not in self.weight_norms:
                self.weight_norms[layer_name] = deque(maxlen=self.buffer_size)
            self.weight_norms[layer_name].append(weight_norm)
        
        for layer_name, weight_update in weight_updates.items():
            if layer_name not in self.weight_updates:
                self.weight_updates[layer_name] = deque(maxlen=self.buffer_size)
            self.weight_updates[layer_name].append(weight_update)
        
        # Prepare metrics
        metrics = {
            'model/learning_rate': learning_rate,
        }
        
        # Add layer-specific weight metrics
        for layer_name, weight_norm in weight_norms.items():
            metrics[f'model/weight_norm_{layer_name}'] = weight_norm
            
        for layer_name, weight_update in weight_updates.items():
            metrics[f'model/weight_update_{layer_name}'] = weight_update
        
        # Add rolling weight metrics
        self._add_rolling_weight_metrics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.weight_updates_logged += 1
    
    def _handle_attention_analysis(self, context: Dict[str, Any]) -> None:
        """Handle detailed attention analysis events."""
        raw = context.get('raw', {})
        
        # Extract attention pattern analysis
        attention_diversity = raw.get('attention_diversity', 0)
        attention_locality = raw.get('attention_locality', 0)
        attention_stability = raw.get('attention_stability', 0)
        head_specialization = raw.get('head_specialization', 0)
        
        metrics = {
            'attention/diversity': attention_diversity,
            'attention/locality': attention_locality,
            'attention/stability': attention_stability,
            'attention/head_specialization': head_specialization
        }
        
        # Extract per-head attention statistics
        head_entropies = raw.get('head_entropies', {})
        for head_idx, entropy in head_entropies.items():
            metrics[f'attention/head_{head_idx}_entropy'] = entropy
        
        wandb.log(metrics)
    
    def _handle_feature_analysis(self, context: Dict[str, Any]) -> None:
        """Handle feature utilization analysis events."""
        raw = context.get('raw', {})
        
        # Extract feature utilization statistics
        feature_importance = raw.get('feature_importance', 0)
        feature_correlation = raw.get('feature_correlation', 0)
        unused_features = raw.get('unused_features_count', 0)
        feature_variance = raw.get('feature_variance', 0)
        
        self.feature_importances.append(feature_importance)
        self.feature_correlations.append(feature_correlation)
        self.unused_features_count.append(unused_features)
        
        metrics = {
            'features/importance_score': feature_importance,
            'features/correlation_score': feature_correlation,
            'features/unused_count': unused_features,
            'features/variance_score': feature_variance
        }
        
        # Add rolling feature metrics
        if len(self.feature_importances) >= 20:
            metrics.update({
                'features/avg_importance_20': np.mean(list(self.feature_importances)[-20:]),
                'features/avg_correlation_20': np.mean(list(self.feature_correlations)[-20:]),
                'features/avg_unused_20': np.mean(list(self.unused_features_count)[-20:])
            })
        
        wandb.log(metrics)
    
    def _add_rolling_forward_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling metrics for forward pass data."""
        # 20-step rolling metrics
        if len(self.attention_entropies) >= 20:
            recent_20_attention = list(self.attention_entropies)[-20:]
            recent_20_confidence = list(self.prediction_confidences)[-20:]
            recent_20_action_entropy = list(self.action_entropies)[-20:]
            
            metrics.update({
                'rolling_20/avg_attention_entropy': np.mean(recent_20_attention),
                'rolling_20/attention_entropy_std': np.std(recent_20_attention),
                'rolling_20/avg_prediction_confidence': np.mean(recent_20_confidence),
                'rolling_20/avg_action_entropy': np.mean(recent_20_action_entropy),
                'rolling_20/attention_stability': 1 / (1 + np.std(recent_20_attention))
            })
        
        # 100-step rolling metrics
        if len(self.activation_magnitudes) >= 100:
            recent_100_mag = list(self.activation_magnitudes)[-100:]
            recent_100_sparsity = list(self.activation_sparsities)[-100:]
            recent_100_focus = list(self.attention_focus_scores)[-100:]
            
            metrics.update({
                'rolling_100/avg_activation_magnitude': np.mean(recent_100_mag),
                'rolling_100/avg_activation_sparsity': np.mean(recent_100_sparsity),
                'rolling_100/avg_attention_focus': np.mean(recent_100_focus),
                'rolling_100/activation_consistency': 1 / (1 + np.std(recent_100_mag))
            })
    
    def _add_rolling_backward_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling metrics for backward pass data."""
        if len(self.gradient_norms) >= 10:
            recent_10_norms = list(self.gradient_norms)[-10:]
            recent_10_vars = list(self.gradient_variances)[-10:]
            
            metrics.update({
                'rolling_10/avg_gradient_norm': np.mean(recent_10_norms),
                'rolling_10/gradient_norm_std': np.std(recent_10_norms),
                'rolling_10/avg_gradient_variance': np.mean(recent_10_vars),
                'rolling_10/gradient_stability': 1 / (1 + np.std(recent_10_norms))
            })
        
        if len(self.gradient_norms) >= 50:
            recent_50_norms = list(self.gradient_norms)[-50:]
            recent_50_max = list(self.gradient_max_values)[-50:]
            
            metrics.update({
                'rolling_50/avg_gradient_norm': np.mean(recent_50_norms),
                'rolling_50/max_gradient_norm': np.max(recent_50_norms),
                'rolling_50/avg_gradient_max': np.mean(recent_50_max),
                'rolling_50/gradient_explosion_risk': np.mean([g > 10.0 for g in recent_50_norms])
            })
    
    def _add_rolling_weight_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling metrics for weight updates."""
        # Calculate average weight norms across layers
        if self.weight_norms:
            all_recent_norms = []
            for layer_name, norms in self.weight_norms.items():
                if len(norms) >= 10:
                    recent_norms = list(norms)[-10:]
                    all_recent_norms.extend(recent_norms)
                    
                    # Layer-specific rolling metrics
                    metrics[f'rolling_10/avg_weight_norm_{layer_name}'] = np.mean(recent_norms)
            
            if all_recent_norms:
                metrics.update({
                    'rolling_10/avg_weight_norm_all_layers': np.mean(all_recent_norms),
                    'rolling_10/weight_norm_std_all_layers': np.std(all_recent_norms)
                })
        
        # Calculate average weight updates across layers
        if self.weight_updates:
            all_recent_updates = []
            for layer_name, updates in self.weight_updates.items():
                if len(updates) >= 10:
                    recent_updates = list(updates)[-10:]
                    all_recent_updates.extend(recent_updates)
                    
                    # Layer-specific rolling metrics
                    metrics[f'rolling_10/avg_weight_update_{layer_name}'] = np.mean(recent_updates)
            
            if all_recent_updates:
                metrics.update({
                    'rolling_10/avg_weight_update_all_layers': np.mean(all_recent_updates),
                    'rolling_10/weight_update_magnitude': np.mean([abs(u) for u in all_recent_updates])
                })
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Log update-level model summaries."""
        if not wandb or not wandb.run:
            return
        
        # Extract model state from context
        model_confidence = context.get('model_confidence', 0)
        value_function_error = context.get('value_function_error', 0)
        
        if value_function_error != 0:
            self.value_function_errors.append(value_function_error)
        
        # Calculate model health metrics
        metrics = {}
        
        if len(self.attention_entropies) >= 10:
            recent_attention = list(self.attention_entropies)[-10:]
            attention_health = np.mean(recent_attention)  # Higher entropy = more exploration
            metrics['model_health/attention_exploration'] = attention_health
        
        if len(self.gradient_norms) >= 10:
            recent_grads = list(self.gradient_norms)[-10:]
            gradient_health = 1 / (1 + np.std(recent_grads))  # Lower variance = more stable
            metrics['model_health/gradient_stability'] = gradient_health
        
        if len(self.activation_sparsities) >= 10:
            recent_sparsity = list(self.activation_sparsities)[-10:]
            sparsity_health = np.mean(recent_sparsity)  # Healthy sparsity indicates feature specialization
            metrics['model_health/activation_sparsity'] = sparsity_health
        
        if metrics:
            wandb.log(metrics)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics."""
        stats = {
            'buffer_size': self.buffer_size,
            'forward_passes_logged': self.forward_passes_logged,
            'backward_passes_logged': self.backward_passes_logged,
            'weight_updates_logged': self.weight_updates_logged,
            'attention_entropies_buffer_size': len(self.attention_entropies),
            'gradient_norms_buffer_size': len(self.gradient_norms),
            'layer_activations_tracked': len(self.layer_activations),
            'weight_layers_tracked': len(self.weight_norms)
        }
        
        # Add current values
        if self.attention_entropies:
            stats['current_attention_entropy'] = self.attention_entropies[-1]
        if self.gradient_norms:
            stats['current_gradient_norm'] = self.gradient_norms[-1]
        if self.prediction_confidences:
            stats['current_prediction_confidence'] = self.prediction_confidences[-1]
        
        return stats