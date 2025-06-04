"""SHAP Attribution Callback for real-time feature analysis."""

import logging
from typing import Dict, Any, Optional
import time

from agent.callbacks import BaseCallback

try:
    from feature.attribution.comprehensive_shap_analyzer import ComprehensiveSHAPAnalyzer, AttributionConfig
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False


class AttributionCallback(BaseCallback):
    """Callback for SHAP feature attribution analysis.
    
    This callback:
    - Runs SHAP analysis on model at specified intervals
    - Updates dashboard with attribution results
    - Logs feature importance to W&B
    - Tracks dead features and attribution quality
    - Only performs calculations when enabled
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        enabled: bool = True
    ):
        """Initialize attribution callback.
        
        Args:
            config: Attribution configuration
            enabled: Whether this callback is active
        """
        super().__init__(enabled)
        
        if not ATTRIBUTION_AVAILABLE and enabled:
            self.logger.warning("SHAP attribution dependencies not available. AttributionCallback disabled.")
            self.enabled = False
            return
        
        self.config = config
        self.shap_analyzer: Optional[ComprehensiveSHAPAnalyzer] = None
        self.model = None
        self.last_attribution_time = 0
        self.attribution_count = 0
        
        # Configuration
        self.update_frequency = config.get('update_frequency', 10)
        self.max_samples = config.get('max_samples_per_analysis', 5)
        self.enabled_analysis = config.get('enabled', True)
        
        self.logger.info(f"Attribution callback initialized - frequency: {self.update_frequency}, max_samples: {self.max_samples}")
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Initialize SHAP analyzer with model."""
        if not self.enabled or not self.enabled_analysis:
            return
        
        # Get model from config if available
        model = config.get('model')
        if model is None:
            self.logger.warning("No model provided to attribution callback")
            return
        
        self.model = model
        
        try:
            # Create SHAP analyzer configuration
            shap_config = self.config.copy()
            shap_config.update({
                'max_samples_per_analysis': self.max_samples,
                'update_frequency': self.update_frequency,
                'dashboard_update': True,
                'log_to_wandb': True
            })
            
            attribution_config = AttributionConfig(**shap_config)
            
            # Initialize SHAP analyzer
            self.shap_analyzer = ComprehensiveSHAPAnalyzer(
                model=self.model,
                config=attribution_config,
                logger=self.logger
            )
            
            self.logger.info("✅ SHAP analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP analyzer: {e}")
            self.enabled = False
    
    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Run SHAP analysis at specified intervals."""
        if not self.enabled or not self.shap_analyzer or not self.enabled_analysis:
            return
        
        # Check if it's time for attribution analysis
        if update_num % self.update_frequency != 0:
            return
        
        current_time = time.time()
        self.logger.info(f"🔍 Running SHAP attribution analysis at update {update_num}")
        
        try:
            # Get recent batch data for analysis
            batch_data = update_metrics.get('batch_data')
            if batch_data is None:
                self.logger.warning("No batch data available for attribution analysis")
                return
            
            # Run SHAP analysis
            analysis_start = time.time()
            
            # Extract states from batch data
            batch_states = batch_data.get('states')
            if batch_states is None or not batch_states:
                self.logger.warning("No batch states available for attribution analysis")
                return
            
            # Convert batch states to list format expected by analyzer
            if isinstance(batch_states, dict):
                # Convert from batched dict to list of individual state dicts
                batch_size = len(next(iter(batch_states.values())))
                states_list = []
                for i in range(min(batch_size, self.max_samples)):
                    state_dict = {key: value[i:i+1] for key, value in batch_states.items()}
                    states_list.append(state_dict)
            else:
                states_list = batch_states[:self.max_samples]
            
            batch_actions = batch_data.get('actions')
            actions_list = batch_actions[:self.max_samples].tolist() if batch_actions is not None else None
            
            attribution_results = self.shap_analyzer.analyze_features(
                states=states_list,
                actions=actions_list
            )
            analysis_time = time.time() - analysis_start
            
            if attribution_results:
                self.attribution_count += 1
                self.last_attribution_time = current_time
                
                self.logger.info(f"✅ SHAP analysis completed in {analysis_time:.2f}s")
                self.logger.info(f"📊 Found {len(attribution_results.get('top_features', []))} top features")
                self.logger.info(f"💀 Detected {attribution_results.get('dead_features_count', 0)} dead features")
                
                # Trigger dashboard update with attribution results
                self._update_dashboard_attribution(attribution_results)
                
                # Log to W&B if enabled
                self._log_attribution_to_wandb(attribution_results, update_num)
                
            else:
                self.logger.warning("SHAP analysis returned no results")
                
        except Exception as e:
            self.logger.error(f"Error during SHAP attribution analysis: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_dashboard_attribution(self, attribution_results: Dict[str, Any]) -> None:
        """Update dashboard with attribution results."""
        try:
            from dashboard.shared_state import dashboard_state
            
            # Extract key metrics for dashboard
            attribution_summary = {
                'top_features_by_branch': attribution_results.get('top_features_by_branch', {}),
                'dead_features_count': attribution_results.get('dead_features_count', 0),
                'quality_sparsity': attribution_results.get('quality_metrics', {}).get('sparsity', 0.0),
                'consensus_mean_correlation': attribution_results.get('quality_metrics', {}).get('consensus_correlation', 0.0),
                'branch_hf_max_attribution': attribution_results.get('branch_max_attributions', {}).get('hf', 0.0),
                'branch_mf_max_attribution': attribution_results.get('branch_max_attributions', {}).get('mf', 0.0),
                'branch_lf_max_attribution': attribution_results.get('branch_max_attributions', {}).get('lf', 0.0),
                'branch_portfolio_max_attribution': attribution_results.get('branch_max_attributions', {}).get('portfolio', 0.0),
            }
            
            dashboard_state.attribution_summary = attribution_summary
            self.logger.debug("📊 Updated dashboard with attribution results")
            
        except Exception as e:
            self.logger.warning(f"Failed to update dashboard with attribution: {e}")
    
    def _log_attribution_to_wandb(self, attribution_results: Dict[str, Any], update_num: int) -> None:
        """Log attribution results to W&B."""
        try:
            import wandb
            
            if wandb.run is None:
                return
            
            # Log key attribution metrics
            metrics = {
                'attribution/dead_features_count': attribution_results.get('dead_features_count', 0),
                'attribution/total_features_analyzed': attribution_results.get('total_features', 0),
                'attribution/sparsity': attribution_results.get('quality_metrics', {}).get('sparsity', 0.0),
                'attribution/consensus_correlation': attribution_results.get('quality_metrics', {}).get('consensus_correlation', 0.0),
                'update': update_num
            }
            
            # Log branch-specific max attributions
            branch_max = attribution_results.get('branch_max_attributions', {})
            for branch, max_attr in branch_max.items():
                metrics[f'attribution/max_{branch}'] = max_attr
            
            # Log top features by importance
            top_features = attribution_results.get('top_features', [])
            for i, feature_info in enumerate(top_features[:10]):  # Top 10
                metrics[f'attribution/top_feature_{i+1}'] = feature_info.get('importance', 0.0)
            
            wandb.log(metrics)
            self.logger.debug(f"📈 Logged attribution metrics to W&B")
            
        except Exception as e:
            self.logger.warning(f"Failed to log attribution to W&B: {e}")
    
    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Clean up attribution resources."""
        if not self.enabled:
            return
        
        if self.shap_analyzer:
            try:
                # Save final attribution report if configured
                if hasattr(self.shap_analyzer, 'save_final_report'):
                    self.shap_analyzer.save_final_report()
                self.logger.info(f"🔍 Attribution analysis completed: {self.attribution_count} analyses performed")
            except Exception as e:
                self.logger.warning(f"Error during attribution cleanup: {e}")
        
        self.shap_analyzer = None
        self.model = None