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
    from feature.attribution.comprehensive_shap_analyzer import ComprehensiveSHAPAnalyzer, AttributionConfig
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
        enable_attribution: bool = True,
        model_config: Optional[Any] = None
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
        self.attribution_analysis_interval = 0  # No time-based limits (controlled by PPO agent)
        self.last_attribution_summary = None  # Store summary for dashboard
        
        # Initialize feature attribution analyzer if available
        if self.enable_attribution and model is not None and feature_names is not None:
            try:
                # Extract branch configs from model_config
                branch_configs = {}
                if model_config:
                    for branch in feature_names.keys():
                        seq_len_attr = f"{branch}_seq_len"
                        feat_dim_attr = f"{branch}_feat_dim"
                        
                        if hasattr(model_config, seq_len_attr) and hasattr(model_config, feat_dim_attr):
                            seq_len = getattr(model_config, seq_len_attr)
                            feat_dim = getattr(model_config, feat_dim_attr)
                            branch_configs[branch] = (seq_len, feat_dim)
                else:
                    # Fallback defaults
                    default_configs = {
                        'hf': (60, 7),
                        'mf': (30, 43), 
                        'lf': (30, 19),
                        'portfolio': (5, 10)
                    }
                    for branch in feature_names.keys():
                        branch_configs[branch] = default_configs.get(branch, (30, 10))
                
                # Create attribution config
                attribution_config = AttributionConfig(
                    enabled=True,
                    update_frequency=10,  # Default, will be controlled by PPO agent
                    methods=["gradient_shap", "integrated_gradients"],
                    primary_method="gradient_shap",
                    max_samples_per_analysis=5,
                    background_samples=10,
                    track_interactions=True,
                    save_plots=True,
                    log_to_wandb=True,
                    dashboard_update=True
                )
                
                self.attribution_analyzer = ComprehensiveSHAPAnalyzer(
                    model=model,
                    feature_names=feature_names,
                    branch_configs=branch_configs,
                    config=attribution_config,
                    logger=self.logger
                )
                self.logger.info("✅ Comprehensive SHAP feature attribution analyzer initialized")
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
            
            # SHAP attribution metrics (from stored results)
            # Note: Actual SHAP analysis happens in run_periodic_shap_analysis()
            # Here we just collect the stored metrics for dashboard/wandb
                
                # Simple attention stability calculation (since track_attention_patterns doesn't exist)
                if self.last_attention_weights is not None and len(self.attention_weights_history) > 1:
                    try:
                        # Calculate stability as variance across recent attention weights
                        recent_weights = list(self.attention_weights_history)[-5:]
                        if len(recent_weights) > 1:
                            weight_variance = np.var(recent_weights, axis=0).mean()
                            stability = 1.0 / (1.0 + weight_variance)  # Higher stability = lower variance
                            metrics[f"{self.category.value}.{self.name}.attention_stability"] = MetricValue(stability)
                    except Exception as e:
                        self.logger.debug(f"Error calculating attention stability: {e}")
            
            # Add stored attribution metrics to collection
            if hasattr(self, 'last_attribution_metrics') and self.last_attribution_metrics:
                for metric_name, value in self.last_attribution_metrics.items():
                    metrics[f"{self.category.value}.{self.name}.{metric_name}"] = MetricValue(value)
                    
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
            
            # Log buffer status every 10 states for debugging
            if len(self.state_buffer) % 10 == 0:
                self.logger.debug(f"Attribution state buffer: {len(self.state_buffer)} states stored")
    
    def run_periodic_shap_analysis(self) -> Optional[Dict[str, Any]]:
        """Run comprehensive SHAP attribution analysis periodically"""
        if not self.enable_attribution or not self.attribution_analyzer:
            return None
        
        # Check if attribution is enabled in config
        if hasattr(self.attribution_analyzer, 'config') and not self.attribution_analyzer.config.enabled:
            return None
            
        current_time = time.time()
        time_since_last = current_time - self.last_attribution_analysis_time
        states_available = len(self.state_buffer)
        
        self.logger.debug(f"SHAP check: {time_since_last:.1f}s since last, {states_available} states available")
        
        if states_available < 2:  # Need minimal states for analysis
            self.logger.debug(f"Attribution skipped: only {states_available} states (need 2)")
            return None
        
        # Setup background data if needed
        if not hasattr(self.attribution_analyzer, 'background_cache') or self.attribution_analyzer.background_cache is None:
            background_states = list(self.state_buffer)[:10]  # Use first 10 states as background
            self.attribution_analyzer.setup_background(background_states)
            
        # Run comprehensive SHAP analysis
        try:
            self.logger.info(f"🔍 Starting comprehensive SHAP analysis with {states_available} states")
            
            # Convert states to list
            analysis_states = list(self.state_buffer)
            
            # Run analysis
            results = self.attribution_analyzer.analyze_features(analysis_states)
            
            if results and 'error' not in results:
                self.last_attribution_analysis_time = current_time
                
                # Store results for metrics collection
                self._store_attribution_results(results)
                
                # Log to WandB
                if hasattr(self.attribution_analyzer, 'log_to_wandb'):
                    self.attribution_analyzer.log_to_wandb(results)
                    
                # Get dashboard summary
                if hasattr(self.attribution_analyzer, 'get_summary_for_logging'):
                    dashboard_summary = self.attribution_analyzer.get_summary_for_logging()
                    # Store for dashboard access
                    self.last_attribution_summary = dashboard_summary
                    
                return results
            else:
                self.logger.warning(f"SHAP analysis returned error or None: {results}")
                return None
                
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")
            # Fallback to fast gradient attribution
            self.logger.info("Falling back to fast gradient attribution")
            return self._run_fast_gradient_attribution()
    
    def configure_shap_frequency(self, update_frequency: int):
        """Configure SHAP update frequency (e.g., 10 or 50 updates)"""
        if self.attribution_analyzer and hasattr(self.attribution_analyzer, 'config'):
            old_freq = self.attribution_analyzer.config.update_frequency
            self.attribution_analyzer.config.update_frequency = update_frequency
            self.logger.info(f"Updated SHAP frequency from {old_freq} to {update_frequency} updates")
            
    def disable_shap_analysis(self):
        """Completely disable SHAP analysis"""
        if self.attribution_analyzer and hasattr(self.attribution_analyzer, 'config'):
            self.attribution_analyzer.config.enabled = False
            self.logger.info("SHAP analysis has been disabled")
            
    def enable_shap_analysis(self):
        """Enable SHAP analysis"""
        if self.attribution_analyzer and hasattr(self.attribution_analyzer, 'config'):
            self.attribution_analyzer.config.enabled = True
            self.logger.info("SHAP analysis has been enabled")
    
    def _run_fast_gradient_attribution(self) -> Optional[Dict[str, Any]]:
        """Fast gradient-based attribution as fallback when SHAP is too slow"""
        if not self.enable_attribution or len(self.state_buffer) < 2:
            return None
            
        try:
            from feature.attribution.simple_attribution import SimpleFeatureAttribution
            
            # Create simple attribution analyzer if needed
            if not hasattr(self, '_simple_analyzer'):
                self._simple_analyzer = SimpleFeatureAttribution(
                    model=self.attribution_analyzer.model if self.attribution_analyzer else None,
                    feature_names=self.attribution_analyzer.feature_names if self.attribution_analyzer else {},
                    branch_configs=self.attribution_analyzer.branch_configs if self.attribution_analyzer else {},
                    logger=self.logger
                )
            
            # Use recent states
            analysis_states = list(self.state_buffer)[-3:]  # Use 3 states for gradient analysis
            
            self.logger.info(f"🚀 Running fast gradient attribution on {len(analysis_states)} states")
            start_time = time.time()
            
            # Run simple attribution
            attribution_results = self._simple_analyzer.analyze_features(analysis_states)
            analysis_time = time.time() - start_time
            
            if attribution_results and 'error' not in attribution_results:
                self.logger.info(f"✅ Fast gradient attribution completed in {analysis_time:.2f}s")
                
                # Convert to SHAP-compatible format for dashboard
                if 'input_gradients' in attribution_results:
                    branch_importance = attribution_results['input_gradients']
                    
                    # Create simplified results in SHAP format
                    simplified_results = {
                        'method': 'fast_gradient',
                        'analysis_time': analysis_time,
                        'n_samples': len(analysis_states),
                        'branch_importance': branch_importance,
                        'summary_stats': {
                            'mean_absolute_importance': float(np.mean([abs(v) for v in branch_importance.values()])),
                            'explanation_quality': 'fast_approximation'
                        },
                        'top_features': [
                            {'branch': branch, 'importance': importance, 'rank': i+1} 
                            for i, (branch, importance) in enumerate(
                                sorted(branch_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                            )
                        ][:5]
                    }
                    
                    # Store results for metrics collection
                    self._store_attribution_results(simplified_results)
                    
                    # Log key results to console (like SHAP did)
                    branch_ranking = sorted(branch_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                    self.logger.info(f"📊 Branch ranking: {[(b, f'{s:.3f}') for b, s in branch_ranking]}")
                    
                    top_5 = simplified_results['top_features'][:5]
                    feature_list = [(f['branch'], f"{f['importance']:.4f}") for f in top_5]
                    self.logger.info(f"🔥 Top branches: {feature_list}")
                    
                    # Log to WandB using native plots (no local files!)
                    try:
                        import wandb
                        if wandb.run:
                            log_dict = {}
                            
                            # Branch importance metrics
                            for branch, importance in branch_importance.items():
                                log_dict[f"attribution/branch_{branch}_importance"] = importance
                            
                            # Summary stats
                            for stat, value in simplified_results['summary_stats'].items():
                                log_dict[f"attribution/{stat}"] = value
                            
                            # Top features
                            for i, feature_info in enumerate(simplified_results['top_features'][:5]):
                                log_dict[f"attribution/top_branch_{i+1}_importance"] = feature_info['importance']
                                log_dict[f"attribution/top_branch_{i+1}_name"] = feature_info['branch']
                            
                            # Analysis metadata
                            log_dict["attribution/analysis_time"] = analysis_time
                            log_dict["attribution/n_samples_analyzed"] = len(analysis_states)
                            log_dict["attribution/method"] = "fast_gradient"
                            
                            # Create native WandB visualizations (no local files!)
                            branches = list(branch_importance.keys())
                            importances = list(branch_importance.values())
                            
                            # 1. Bar chart using wandb.plot.bar
                            bar_data = [[branch, importance] for branch, importance in zip(branches, importances)]
                            bar_table = wandb.Table(data=bar_data, columns=["Branch", "Importance"])
                            log_dict["attribution/branch_importance_chart"] = wandb.plot.bar(
                                bar_table, "Branch", "Importance",
                                title=f"Branch Importance (Gradient Attribution) - {analysis_time:.2f}s"
                            )
                            
                            # 2. Line plot for trends
                            log_dict["attribution/branch_importance_line"] = wandb.plot.line(
                                bar_table, "Branch", "Importance",
                                title="Attribution Scores by Branch"
                            )
                            
                            # 3. Custom HTML plot for pie chart using plotly
                            try:
                                import plotly.graph_objects as go
                                import plotly.io as pio
                                
                                # Create pie chart
                                fig = go.Figure(data=[go.Pie(
                                    labels=branches,
                                    values=[abs(v) for v in importances],
                                    title=f"Branch Importance Distribution<br>(Analysis: {analysis_time:.2f}s)"
                                )])
                                
                                fig.update_traces(
                                    textposition='inside', 
                                    textinfo='percent+label',
                                    textfont_size=12
                                )
                                
                                fig.update_layout(
                                    showlegend=True,
                                    width=600, height=500,
                                    font=dict(size=12)
                                )
                                
                                # Log as HTML (wandb supports this!)
                                log_dict["attribution/branch_pie_chart"] = wandb.Html(pio.to_html(fig, include_plotlyjs='inline'))
                                
                            except ImportError:
                                self.logger.debug("Plotly not available for pie chart")
                            
                            # 4. Create a summary table
                            summary_data = []
                            for i, (branch, importance) in enumerate(zip(branches, importances)):
                                summary_data.append([i+1, branch, f"{importance:.4f}", f"{abs(importance)/sum(abs(v) for v in importances)*100:.1f}%"])
                            
                            summary_table = wandb.Table(
                                data=summary_data,
                                columns=["Rank", "Branch", "Importance", "Percentage"]
                            )
                            log_dict["attribution/importance_table"] = summary_table
                            
                            wandb.log(log_dict)
                            self.logger.info(f"📈 Logged {len(log_dict)} gradient attribution metrics + native WandB plots")
                    except Exception as e:
                        self.logger.debug(f"WandB logging failed: {e}")
                    
                    return simplified_results
            
            return attribution_results
            
        except Exception as e:
            self.logger.error(f"Fast gradient attribution failed: {e}")
            return None
    
    def _store_attribution_results(self, attribution_results: Dict[str, Any]):
        """Store SHAP attribution results for later retrieval by metrics collection"""
        self.last_attribution_metrics = {}
        
        # Store branch importance scores
        if 'branch_importance' in attribution_results:
            for branch, importance in attribution_results['branch_importance'].items():
                self.last_attribution_metrics[f"shap_branch_{branch}_importance"] = importance
        
        # Store summary statistics
        if 'summary_stats' in attribution_results:
            for metric, value in attribution_results['summary_stats'].items():
                self.last_attribution_metrics[f"shap_{metric}"] = value
        
        # Store top feature scores
        if 'top_features' in attribution_results:
            for i, feature_info in enumerate(attribution_results['top_features'][:5]):  # Top 5
                self.last_attribution_metrics[f"shap_top_feature_{i+1}_importance"] = feature_info['importance']
                # Handle both SHAP and gradient attribution formats
                if 'feature_name' in feature_info:
                    self.last_attribution_metrics[f"shap_top_feature_{i+1}_name"] = f"{feature_info['branch']}.{feature_info['feature_name']}"
                else:
                    # For gradient attribution, just use branch name
                    self.last_attribution_metrics[f"shap_top_feature_{i+1}_name"] = feature_info['branch']
        
        # Store dead features count
        if 'dead_features' in attribution_results:
            total_dead = sum(len(features) for features in attribution_results['dead_features'].values())
            self.last_attribution_metrics['shap_dead_features_total'] = total_dead
            
            for branch, dead_list in attribution_results['dead_features'].items():
                self.last_attribution_metrics[f"shap_dead_features_{branch}"] = len(dead_list)
        
        # Store analysis metadata
        self.last_attribution_metrics['shap_analysis_time'] = attribution_results.get('analysis_time', 0)
        self.last_attribution_metrics['shap_samples_analyzed'] = attribution_results.get('n_samples', 0)
        
        self.logger.debug(f"Stored {len(self.last_attribution_metrics)} SHAP attribution metrics for dashboard/wandb")
    
    def run_permutation_importance_analysis(self, environment, n_episodes: int = 20) -> Optional[Dict[str, Dict[str, float]]]:
        """Run permutation importance analysis"""
        if not self.enable_attribution or not self.attribution_analyzer:
            return None
        
        try:
            # Get the model from the attribution analyzer
            model = self.attribution_analyzer.model
            
            # Use recent attribution results if available
            rankings = {}
            if hasattr(self, 'last_attribution_metrics') and self.last_attribution_metrics:
                # Convert SHAP metrics to permutation importance format
                for key, value in self.last_attribution_metrics.items():
                    if 'branch_' in key and '_importance' in key:
                        branch = key.replace('shap_branch_', '').replace('_importance', '')
                        rankings[branch] = {'importance': float(value)}
                
                if rankings:
                    self.logger.info("Retrieved feature importance rankings from SHAP")
                    return rankings
            
            # Fallback: run new SHAP analysis if no recent results
            if len(self.state_buffer) >= 5:
                attribution_results = self.run_periodic_shap_analysis()
                if attribution_results and 'branch_importance' in attribution_results:
                    for branch, importance in attribution_results['branch_importance'].items():
                        rankings[branch] = {'importance': float(importance)}
                    if rankings:
                        self.logger.info("Generated new SHAP rankings")
                        return rankings
            
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
            
            # Generate comprehensive report using SHAP
            if states_for_analysis:
                attribution_results = self.attribution_analyzer.analyze_features(states_for_analysis, max_samples=10)
                
                if attribution_results and 'error' not in attribution_results:
                    # Build comprehensive report from SHAP results
                    report = {
                        "method": "SHAP",
                        "analysis_summary": attribution_results.get('summary_stats', {}),
                        "branch_importance": attribution_results.get('branch_importance', {}),
                        "top_features": attribution_results.get('top_features', []),
                        "dead_features": attribution_results.get('dead_features', {}),
                        "feature_interactions": attribution_results.get('feature_interactions', {}),
                        "individual_explanations": attribution_results.get('individual_explanations', []),
                        "analysis_time": attribution_results.get('analysis_time', 0),
                        "samples_analyzed": attribution_results.get('n_samples', 0)
                    }
                    
                    # Save plot paths if available
                    if 'plot_paths' in attribution_results:
                        report['visualizations'] = attribution_results['plot_paths']
                        
                    self.logger.info(f"Generated comprehensive SHAP feature report with {len(report)} sections")
                    
                    # Save to file if requested
                    if save_path:
                        import json
                        try:
                            with open(save_path, 'w') as f:
                                # Convert non-serializable objects for JSON
                                serializable_report = {}
                                for key, value in report.items():
                                    try:
                                        json.dumps(value)  # Test if serializable
                                        serializable_report[key] = value
                                    except (TypeError, ValueError):
                                        serializable_report[key] = str(value)
                                json.dump(serializable_report, f, indent=2)
                            self.logger.info(f"SHAP feature report saved to {save_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to save report to {save_path}: {e}")
                            
                    return report
                else:
                    return {"error": "SHAP analysis failed"}
            else:
                return {"error": "No states available for analysis"}
            
        except Exception as e:
            self.logger.error(f"Failed to generate feature report: {e}")
            return {}
    
    def get_feature_attribution_summary(self) -> Dict[str, Any]:
        """Get a summary of current gradient feature attribution analysis"""
        if not self.enable_attribution:
            return {"attribution_enabled": False}
        
        # Use stored attribution results from gradient analysis
        if hasattr(self, 'last_attribution_metrics') and self.last_attribution_metrics:
            # Convert gradient attribution results to dashboard format
            summary = {
                "attribution_enabled": True,
                "method": "fast_gradient",
                "total_features_tracked": len(self.last_attribution_metrics),
                "branches_tracked": len([k for k in self.last_attribution_metrics.keys() if 'branch_' in k and '_importance' in k]),
            }
            
            # Top features for dashboard
            top_features = []
            for i in range(1, 6):  # Top 5
                importance_key = f"shap_top_feature_{i}_importance"
                name_key = f"shap_top_feature_{i}_name" 
                if importance_key in self.last_attribution_metrics and name_key in self.last_attribution_metrics:
                    top_features.append({
                        "name": self.last_attribution_metrics[name_key],
                        "current": float(self.last_attribution_metrics[importance_key]),
                        "average": float(self.last_attribution_metrics[importance_key]),  # No history yet
                        "trend": "stable"
                    })
            
            summary["top_features"] = top_features
            
            # Branch importance for dashboard
            branch_importance = {}
            for key, value in self.last_attribution_metrics.items():
                if key.startswith('shap_branch_') and key.endswith('_importance'):
                    branch = key.replace('shap_branch_', '').replace('_importance', '')
                    branch_importance[branch] = {
                        "current": float(value),
                        "average": float(value),  # No history yet
                        "trend": "stable"
                    }
            
            summary["branch_importance"] = branch_importance
            
            # Convert branch importance to top_features_by_branch format for dashboard
            top_features_by_branch = {}
            for branch, importance_data in branch_importance.items():
                top_features_by_branch[branch] = [{
                    "name": branch,
                    "importance": importance_data["current"],
                    "rank": 1
                }]
            
            summary["top_features_by_branch"] = top_features_by_branch
            
            # Add quality metrics
            summary["explanation_quality"] = "fast_approximation"
            summary["dead_features_count"] = self.last_attribution_metrics.get('shap_dead_features_total', 0)
            
            self.logger.info(f"📊 Generated attribution summary with {len(summary)} keys for dashboard")
            self.logger.info(f"📊 Top branches: {list(branch_importance.keys())}")
            return summary
        
        return {"attribution_enabled": False, "reason": "No attribution data available"}
    
    # Local file visualization method removed - using WandB native plots instead!
    
    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)