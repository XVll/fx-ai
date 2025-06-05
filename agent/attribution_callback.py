"""SHAP Attribution Callback for real-time feature analysis."""

import logging
from typing import Dict, Any, Optional, List
import time
from collections import deque

from agent.base_callbacks import TrainingCallback

try:
    from feature.attribution.comprehensive_shap_analyzer import (
        ComprehensiveSHAPAnalyzer,
        AttributionConfig,
    )
    ATTRIBUTION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Attribution import error: {e}")
    ATTRIBUTION_AVAILABLE = False


class AttributionCallback(TrainingCallback):
    """Callback for SHAP feature attribution analysis.

    This callback:
    - Runs SHAP analysis on model at specified intervals
    - Updates dashboard with attribution results
    - Logs feature importance to W&B
    - Tracks dead features and attribution quality
    - Only performs calculations when enabled
    """

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """Initialize attribution callback.

        Args:
            config: Attribution configuration
            enabled: Whether this callback is active
        """
        self.config = config
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)

        if not ATTRIBUTION_AVAILABLE:
            self.logger.error(
                "❌ SHAP attribution dependencies not available. AttributionCallback disabled."
            )
            self.enabled = False
            return

        self.shap_analyzer: Optional[ComprehensiveSHAPAnalyzer] = None
        self.model = None
        self.last_attribution_time = 0
        self.attribution_count = 0
        self.state_cache = deque(maxlen=50)  # Cache states for background

        # Configuration
        self.update_frequency = config.get("update_frequency", 10)
        self.max_samples = config.get("max_samples_per_analysis", 5)
        self.background_samples = config.get("background_samples", 10)
        self.enabled_analysis = config.get("enabled", True)

        self.logger.info(
            f"🔍 Attribution callback initialized - frequency: {self.update_frequency}, max_samples: {self.max_samples}"
        )

    def on_training_start(self, trainer):
        """Initialize SHAP analyzer with model."""
        if not self.enabled or not self.enabled_analysis:
            return

        # Get model from trainer
        model = getattr(trainer, 'model', None)
        if model is None:
            self.logger.warning("No model available in trainer for attribution callback")
            return

        self.model = model
        device = getattr(trainer, 'device', None)

        try:
            # Extract feature info from model architecture
            feature_names = self._get_feature_names_from_config()
            branch_configs = self._get_branch_configs_from_config()

            # Create SHAP analyzer configuration from existing config
            attribution_config = AttributionConfig(**self.config)

            # Initialize SHAP analyzer
            self.shap_analyzer = ComprehensiveSHAPAnalyzer(
                model=self.model,
                feature_names=feature_names,
                branch_configs=branch_configs,
                config=attribution_config,
                device=device,
                logger=self.logger
            )

            self.logger.info("✅ SHAP analyzer initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SHAP analyzer: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self.enabled = False

    def on_attribution_analysis(self, attribution_data: Dict[str, Any]):
        """Handle attribution analysis trigger."""
        update_num = attribution_data.get('update_num', 0)
        self.logger.info(f"🔍 Attribution analysis triggered at update {update_num}")
        
        if not self.enabled:
            self.logger.info("🔕 Attribution callback is disabled")
            return
            
        if not self.enabled_analysis:
            self.logger.info("🔕 Attribution analysis is disabled in config")
            return
            
        if not self.shap_analyzer:
            self.logger.warning("❌ SHAP analyzer not initialized")
            return
        
        try:
            # Check if we have enough cached states for background
            if len(self.state_cache) < self.background_samples:
                self.logger.info(f"⏳ Not enough cached states for SHAP analysis: {len(self.state_cache)}/{self.background_samples} needed")
                return

            self.logger.info(f"🔍 Running SHAP analysis at update {update_num} with {len(self.state_cache)} cached states")
            start_time = time.time()

            # Setup background if needed
            if not hasattr(self.shap_analyzer, '_background_ready'):
                background_states = list(self.state_cache)[-self.background_samples:]
                self.shap_analyzer.setup_background(background_states)
                self.shap_analyzer._background_ready = True
                self.logger.info("🔧 SHAP background setup completed")

            # Get recent states for analysis
            analysis_states = list(self.state_cache)[-self.max_samples:]
            
            # Run SHAP analysis
            results = self.shap_analyzer.analyze_features(
                states=analysis_states,
                actions=None,
                rewards=None
            )

            analysis_time = time.time() - start_time
            self.attribution_count += 1
            
            if results and 'error' not in results:
                self.logger.info(f"✅ SHAP analysis completed in {analysis_time:.2f}s")
                
                # Log key results
                if 'top_features' in results:
                    self.logger.info(f"📊 Top features identified: {len(results['top_features'])}")
                if 'dead_features_count' in results:
                    self.logger.info(f"💀 Dead features detected: {results['dead_features_count']}")
                
                # Process results
                self._process_attribution_results(results, update_num)
            else:
                self.logger.warning(f"⚠️ SHAP analysis returned no valid results")

        except Exception as e:
            self.logger.error(f"❌ Error during SHAP analysis: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def on_step(self, trainer, state, action, reward, next_state, info):
        """Cache states for attribution analysis."""
        if not self.enabled:
            return

        # Cache states occasionally to avoid memory issues
        if len(self.state_cache) == 0 or len(self.state_cache) % 3 == 0:
            if isinstance(state, dict):
                try:
                    # Clone tensors to avoid reference issues
                    cached_state = {}
                    for key, value in state.items():
                        if hasattr(value, 'clone'):
                            cached_state[key] = value.clone().detach()
                        else:
                            cached_state[key] = value
                    self.state_cache.append(cached_state)
                    
                    # Log progress towards having enough states
                    if len(self.state_cache) in [1, 5, 10] or len(self.state_cache) % 20 == 0:
                        self.logger.debug(f"📦 Cached {len(self.state_cache)} states for attribution analysis")
                except Exception as e:
                    self.logger.warning(f"Failed to cache state for attribution: {e}")

    def _get_feature_names_from_config(self) -> Dict[str, List[str]]:
        """Extract feature names from model configuration."""
        # This should match your actual model architecture
        return {
            'hf': [f'hf_feature_{i}' for i in range(60 * 9)],    # High-frequency: 60 timesteps * 9 features
            'mf': [f'mf_feature_{i}' for i in range(30 * 43)],   # Medium-frequency: 30 timesteps * 43 features  
            'lf': [f'lf_feature_{i}' for i in range(30 * 19)],   # Low-frequency: 30 timesteps * 19 features
            'portfolio': [f'portfolio_feature_{i}' for i in range(5 * 10)],  # Portfolio: 5 timesteps * 10 features
        }

    def _get_branch_configs_from_config(self) -> Dict[str, tuple]:
        """Extract branch configurations from model."""
        return {
            'hf': (60, 9),      # (seq_len, feat_dim)
            'mf': (30, 43),
            'lf': (30, 19),
            'portfolio': (5, 10)
        }

    def _process_attribution_results(self, results: Dict[str, Any], update_num: int):
        """Process and distribute attribution results."""
        try:
            # Update dashboard
            self._update_dashboard_attribution(results)
            
            # Log to WandB
            self._log_attribution_to_wandb(results, update_num)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process attribution results: {e}")

    def _update_dashboard_attribution(
        self, attribution_results: Dict[str, Any]
    ) -> None:
        """Update dashboard with attribution results."""
        try:
            from dashboard.shared_state import dashboard_state

            # Extract key metrics for dashboard
            attribution_summary = {
                "top_features_by_branch": attribution_results.get(
                    "top_features_by_branch", {}
                ),
                "dead_features_count": attribution_results.get(
                    "dead_features_count", 0
                ),
                "quality_sparsity": attribution_results.get("quality_metrics", {}).get(
                    "sparsity", 0.0
                ),
                "consensus_mean_correlation": attribution_results.get(
                    "quality_metrics", {}
                ).get("consensus_correlation", 0.0),
                "branch_hf_max_attribution": attribution_results.get(
                    "branch_max_attributions", {}
                ).get("hf", 0.0),
                "branch_mf_max_attribution": attribution_results.get(
                    "branch_max_attributions", {}
                ).get("mf", 0.0),
                "branch_lf_max_attribution": attribution_results.get(
                    "branch_max_attributions", {}
                ).get("lf", 0.0),
                "branch_portfolio_max_attribution": attribution_results.get(
                    "branch_max_attributions", {}
                ).get("portfolio", 0.0),
            }

            dashboard_state.attribution_summary = attribution_summary
            self.logger.debug("📊 Updated dashboard with attribution results")

        except Exception as e:
            self.logger.warning(f"Failed to update dashboard with attribution: {e}")

    def _log_attribution_to_wandb(
        self, attribution_results: Dict[str, Any], update_num: int
    ) -> None:
        """Log attribution results to W&B."""
        try:
            import wandb

            if wandb.run is None:
                return

            # Log key attribution metrics
            metrics = {
                "attribution/dead_features_count": attribution_results.get(
                    "dead_features_count", 0
                ),
                "attribution/total_features_analyzed": attribution_results.get(
                    "total_features", 0
                ),
                "attribution/sparsity": attribution_results.get(
                    "quality_metrics", {}
                ).get("sparsity", 0.0),
                "attribution/consensus_correlation": attribution_results.get(
                    "quality_metrics", {}
                ).get("consensus_correlation", 0.0),
                "update": update_num,
            }

            # Log branch-specific max attributions
            branch_max = attribution_results.get("branch_max_attributions", {})
            for branch, max_attr in branch_max.items():
                metrics[f"attribution/max_{branch}"] = max_attr

            # Log top features by importance
            top_features = attribution_results.get("top_features", [])
            for i, feature_info in enumerate(top_features[:10]):  # Top 10
                metrics[f"attribution/top_feature_{i + 1}"] = feature_info.get(
                    "importance", 0.0
                )

            wandb.log(metrics)
            self.logger.debug("📈 Logged attribution metrics to W&B")

        except Exception as e:
            self.logger.warning(f"Failed to log attribution to W&B: {e}")

    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Clean up attribution resources."""
        if not self.enabled:
            return

        if self.shap_analyzer:
            try:
                # Save final attribution report if configured
                if hasattr(self.shap_analyzer, "save_final_report"):
                    self.shap_analyzer.save_final_report()
                self.logger.info(
                    f"🔍 Attribution analysis completed: {self.attribution_count} analyses performed"
                )
            except Exception as e:
                self.logger.warning(f"Error during attribution cleanup: {e}")

        self.shap_analyzer = None
        self.model = None
