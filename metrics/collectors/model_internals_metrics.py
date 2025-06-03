# metrics/collectors/model_internals_metrics.py - Model internals and diagnostics collector

import logging
import time
from typing import Dict, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from ..core import MetricCollector, MetricValue, MetricCategory, MetricType, MetricMetadata

try:
    from feature.attribution import CaptumFeatureAnalyzer, AttributionConfig
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False


class ModelInternalsCollector(MetricCollector):
    """Collector for model internals and diagnostic metrics with feature attribution"""
    
    def __init__(
        self, 
        buffer_size: int = 100,
        model: Optional[torch.nn.Module] = None,
        feature_names: Optional[Dict[str, List[str]]] = None,
        enable_attribution: bool = True
    ):
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
        
        # Feature attribution tracking
        self.attribution_analyzer = None
        self.enable_attribution = enable_attribution and ATTRIBUTION_AVAILABLE
        self.state_buffer = deque(maxlen=50)  # Store states for attribution analysis
        self.last_attribution_analysis_time = 0
        self.attribution_analysis_interval = 300  # 5 minutes between attribution analyses
        
        # Initialize feature attribution analyzer if available
        if self.enable_attribution and model is not None and feature_names is not None:
            try:
                # Configure Captum with sensible defaults
                config = AttributionConfig(
                    primary_method=AttributionConfig.primary_method,
                    n_steps=25,  # Faster for real-time analysis
                    n_samples=10,
                    use_noise_tunnel=True,
                    track_gradients=True,
                    track_activations=True
                )
                self.attribution_analyzer = CaptumFeatureAnalyzer(
                    model=model,
                    feature_names=feature_names,
                    config=config,
                    logger=self.logger
                )
                self.logger.info("Feature attribution analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize attribution analyzer: {e}")
                self.enable_attribution = False
        
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
            description="Branch with highest attention (0=HF, 1=MF, 2=LF, 3=Portfolio)",
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
        for branch in ['hf', 'mf', 'lf', 'portfolio']:
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
        
        # Feature attribution metrics (if enabled)
        if self.enable_attribution:
            self.register_metric("attribution_analysis_count", MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.COUNTER,
                description="Number of feature attribution analyses performed",
                unit="count",
                frequency="episode"
            ))
            
            # Top feature importance scores
            for branch in ['hf', 'mf', 'lf', 'portfolio']:
                self.register_metric(f"top_feature_importance_{branch}", MetricMetadata(
                    category=MetricCategory.MODEL,
                    metric_type=MetricType.GAUGE,
                    description=f"Importance score of top feature in {branch} branch",
                    unit="importance",
                    frequency="episode"
                ))
            
            # Dead feature detection
            self.register_metric("dead_features_count", MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Number of detected dead/unused features",
                unit="count",
                frequency="episode"
            ))
            
            # Attention stability metrics
            self.register_metric("attention_stability", MetricMetadata(
                category=MetricCategory.MODEL,
                metric_type=MetricType.GAUGE,
                description="Stability of attention patterns over time",
                unit="stability",
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
            
            # Feature attribution metrics
            if self.enable_attribution and self.attribution_analyzer:
                # Track attention patterns if available
                if self.last_attention_weights is not None:
                    attention_analysis = self.attribution_analyzer.track_attention_patterns(self.last_attention_weights)
                    if 'attention_stability' in attention_analysis:
                        metrics[f"{self.category.value}.{self.name}.attention_stability"] = MetricValue(attention_analysis['attention_stability'])
                
                # Get top feature importance scores
                top_features = self.attribution_analyzer.get_top_features(n=1)
                for branch, features in top_features.items():
                    if features:
                        top_importance = features[0][1]  # (name, importance)
                        metrics[f"{self.category.value}.{self.name}.top_feature_importance_{branch}"] = MetricValue(top_importance)
                
                # Dead feature detection
                dead_features = self.attribution_analyzer.detect_dead_features()
                total_dead = sum(len(features) for features in dead_features.values())
                metrics[f"{self.category.value}.{self.name}.dead_features_count"] = MetricValue(total_dead)
                    
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
    
    def update_state_for_attribution(self, state_dict: Dict[str, torch.Tensor]):
        """Store state for feature attribution analysis"""
        if self.enable_attribution:
            # Store a copy of the state for later analysis
            state_copy = {}
            for key, tensor in state_dict.items():
                if torch.is_tensor(tensor):
                    state_copy[key] = tensor.detach().cpu().clone()
                else:
                    state_copy[key] = tensor
            self.state_buffer.append(state_copy)
    
    def run_periodic_shap_analysis(self) -> Optional[Dict[str, Any]]:
        """Run attribution analysis periodically using Captum"""
        if not self.enable_attribution or not self.attribution_analyzer:
            return None
        
        current_time = time.time()
        if (current_time - self.last_attribution_analysis_time < self.attribution_analysis_interval or 
            len(self.state_buffer) < 20):
            return None
        
        try:
            # Convert state buffer to list of dicts
            states_for_analysis = []
            for state in list(self.state_buffer)[-30:]:  # Use last 30 states
                # Ensure tensors are properly formatted
                formatted_state = {}
                for key, tensor in state.items():
                    if torch.is_tensor(tensor):
                        formatted_state[key] = tensor.to(self.attribution_analyzer.device)
                    else:
                        formatted_state[key] = tensor
                states_for_analysis.append(formatted_state)
            
            # Run Captum attribution analysis
            attribution_results = self.attribution_analyzer.calculate_attributions(
                sample_states=states_for_analysis,
                return_convergence_delta=False  # Faster without convergence check
            )
            
            if attribution_results:
                self.last_attribution_analysis_time = current_time
                self.logger.info("Completed periodic Captum attribution analysis")
                
                # Log consensus metrics if available
                if 'consensus' in attribution_results:
                    consensus = attribution_results['consensus']
                    self.logger.info(f"Attribution consensus: {consensus.get('mean_correlation', 0):.3f}")
                
                # Log attribution metrics
                if 'metrics' in attribution_results:
                    metrics = attribution_results['metrics']
                    self.logger.debug(f"Attribution quality - Sparsity: {metrics.get('sparsity', 0):.3f}, "
                                     f"Concentration: {metrics.get('concentration', 0):.3f}")
                
                return attribution_results
            
        except Exception as e:
            self.logger.warning(f"Captum attribution analysis failed: {e}")
        
        return None
    
    def run_permutation_importance_analysis(self, environment, n_episodes: int = 20) -> Optional[Dict[str, Dict[str, float]]]:
        """Run permutation importance analysis"""
        if not self.enable_attribution or not self.attribution_analyzer:
            return None
        
        try:
            # Get the model from the attribution analyzer
            model = self.attribution_analyzer.model
            
            # Permutation importance is less relevant with gradient-based attribution
            # Return feature rankings instead
            rankings = self.attribution_analyzer.get_feature_importance_ranking(
                n_top=5,
                method="historical"
            )
            
            if rankings:
                self.logger.info("Retrieved feature importance rankings from Captum")
            
            return rankings
            
        except Exception as e:
            self.logger.warning(f"Permutation importance analysis failed: {e}")
            return None
    
    def generate_feature_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive feature importance report"""
        if not self.enable_attribution or not self.attribution_analyzer:
            return {}
        
        try:
            # Prepare states for analysis
            states_for_analysis = []
            if len(self.state_buffer) >= 10:
                for state in list(self.state_buffer)[-10:]:
                    formatted_state = {}
                    for key, tensor in state.items():
                        if torch.is_tensor(tensor):
                            formatted_state[key] = tensor.to(self.attribution_analyzer.device)
                        else:
                            formatted_state[key] = tensor
                    states_for_analysis.append(formatted_state)
            
            # Generate comprehensive report using Captum
            report = self.attribution_analyzer.generate_attribution_report(
                sample_states=states_for_analysis if states_for_analysis else [],
                save_path=save_path
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate feature report: {e}")
            return {}
    
    def get_feature_attribution_summary(self) -> Dict[str, Any]:
        """Get a summary of current feature attribution analysis"""
        if not self.enable_attribution or not self.attribution_analyzer:
            return {"attribution_enabled": False}
        
        summary = {
            "attribution_enabled": True,
            "states_buffered": len(self.state_buffer),
            "last_attribution_analysis": self.last_attribution_analysis_time,
            "total_features": self.attribution_analyzer.total_features
        }
        
        # Add top features by branch using Captum rankings
        top_features = self.attribution_analyzer.get_feature_importance_ranking(n_top=3)
        summary["top_features_by_branch"] = {}
        for branch, features in top_features.items():
            summary["top_features_by_branch"][branch] = [
                {"name": name, "importance": importance} 
                for name, importance in features
            ]
        
        # Add dead features info
        dead_features = self.attribution_analyzer.get_dead_features()
        summary["dead_features"] = {
            "total_count": sum(len(features) for features in dead_features.values()),
            "by_branch": {branch: features for branch, features in dead_features.items()}
        }
        
        # Add gradient flow analysis if available
        gradient_analysis = self.attribution_analyzer.analyze_gradient_flow()
        if gradient_analysis and 'error' not in gradient_analysis:
            summary["gradient_flow"] = {
                "vanishing_layers": gradient_analysis.get('vanishing_gradients', []),
                "exploding_layers": gradient_analysis.get('exploding_gradients', [])
            }
        
        return summary
    
    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)