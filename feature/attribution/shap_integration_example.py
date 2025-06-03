"""
Example integration of Comprehensive SHAP Analyzer into training pipeline

This shows how to integrate the analyzer with PPO training callbacks.
"""

import logging
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from feature.attribution import ComprehensiveSHAPAnalyzer, AttributionConfig


class SHAPAnalysisCallback:
    """Callback to run SHAP analysis during training"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Dict[str, List[str]],
        branch_configs: Dict[str, tuple],
        update_frequency: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        
        # Configure SHAP analyzer
        config = AttributionConfig(
            update_frequency=update_frequency,
            max_samples_per_analysis=5,  # Limited for 2,330 features
            background_samples=10,
            methods=["gradient_shap", "integrated_gradients"],
            primary_method="gradient_shap",
            top_k_features=50,
            dead_feature_threshold=0.001,
            use_gpu=torch.cuda.is_available(),
            save_plots=True,
            plot_dir="outputs/shap_plots",
            dashboard_update_freq=5
        )
        
        # Initialize analyzer
        self.analyzer = ComprehensiveSHAPAnalyzer(
            model=model,
            feature_names=feature_names,
            branch_configs=branch_configs,
            config=config,
            logger=logger
        )
        
        self.update_count = 0
        self.background_setup = False
        
    def on_episode_end(self, episode_data: Dict[str, Any]):
        """Collect background data from episodes"""
        if not self.background_setup and len(episode_data.get("states", [])) > 0:
            # Use first few episodes for background
            states = episode_data["states"][:20]  # Limit background samples
            self.analyzer.setup_background(states)
            self.background_setup = True
            self.logger.info("✅ SHAP background data collected")
            
    def on_update_end(self, update_data: Dict[str, Any]):
        """Run SHAP analysis after PPO updates"""
        self.update_count += 1
        
        # Check if it's time to run analysis
        if self.update_count % self.analyzer.config.update_frequency != 0:
            return
            
        if not self.background_setup:
            self.logger.warning("SHAP analysis skipped - no background data yet")
            return
            
        # Get recent rollout data
        states = update_data.get("states", [])
        actions = update_data.get("actions", [])
        rewards = update_data.get("rewards", [])
        
        if not states:
            return
            
        self.logger.info(f"🔍 Running SHAP analysis (update {self.update_count})")
        
        # Run comprehensive analysis
        results = self.analyzer.analyze_features(
            states=states[-50:],  # Last 50 states
            actions=actions[-50:] if actions else None,
            rewards=rewards[-50:] if rewards else None
        )
        
        # Log results
        if "error" not in results:
            # Log to WandB
            if update_data.get("wandb_step"):
                self.analyzer.log_to_wandb(results, step=update_data["wandb_step"])
                
            # Send to dashboard
            if update_data.get("dashboard_queue"):
                dashboard_data = {
                    "type": "shap_analysis",
                    "data": self.analyzer.get_summary_for_logging()
                }
                update_data["dashboard_queue"].put(dashboard_data)
                
            # Log summary
            self._log_analysis_summary(results)
            
    def _log_analysis_summary(self, results: Dict[str, Any]):
        """Log a summary of SHAP analysis results"""
        # Top features
        if "top_features" in results:
            self.logger.info("📊 Top 5 Important Features:")
            for feat in results["top_features"][:5]:
                self.logger.info(
                    f"  {feat['rank']}. {feat['full_name']}: "
                    f"{feat['importance']:.4f} (±{feat['importance_std']:.4f})"
                )
                
        # Branch importance
        if "branch_importance" in results:
            self.logger.info("🌳 Branch Importance:")
            sorted_branches = sorted(
                results["branch_importance"].items(),
                key=lambda x: x[1]["mean"],
                reverse=True
            )
            for branch, stats in sorted_branches:
                self.logger.info(
                    f"  {branch}: {stats['proportion']*100:.1f}% "
                    f"(μ={stats['mean']:.4f})"
                )
                
        # Dead features
        if "dead_features" in results:
            dead_info = results["dead_features"]
            self.logger.info(
                f"💀 Dead Features: {dead_info['total_dead']}/{self.analyzer.total_features} "
                f"({dead_info['dead_percentage']:.1f}%)"
            )
            
        # Feature interactions
        if results.get("feature_interactions") and "top_interactions" in results["feature_interactions"]:
            interactions = results["feature_interactions"]["top_interactions"]
            if interactions:
                self.logger.info(f"🔗 Top Feature Interactions: {len(interactions)} strong correlations found")
                
        # Performance
        self.logger.info(
            f"⏱️ Analysis completed in {results['analysis_time']:.2f}s "
            f"for {results['num_samples']} samples"
        )


# Example usage in training script:
"""
# In your training script (e.g., main.py or ppo_agent.py):

# 1. Create the SHAP callback
shap_callback = SHAPAnalysisCallback(
    model=agent.policy,
    feature_names=feature_manager.feature_names,
    branch_configs={
        'hf': (config.model.hf_seq_len, config.model.hf_feat_dim),
        'mf': (config.model.mf_seq_len, config.model.mf_feat_dim),
        'lf': (config.model.lf_seq_len, config.model.lf_feat_dim),
        'portfolio': (config.model.portfolio_seq_len, config.model.portfolio_feat_dim)
    },
    update_frequency=10,  # Run every 10 updates
    logger=logger
)

# 2. Add to callbacks
callbacks = [
    shap_callback,
    # ... other callbacks
]

# 3. In your training loop:
for episode in range(num_episodes):
    # Collect episode data
    episode_data = run_episode(env, agent)
    
    # Trigger episode callbacks
    for callback in callbacks:
        if hasattr(callback, 'on_episode_end'):
            callback.on_episode_end(episode_data)
    
    # After PPO update
    if episode % update_frequency == 0:
        update_data = agent.update()
        update_data['wandb_step'] = episode
        update_data['dashboard_queue'] = dashboard_queue
        
        # Trigger update callbacks
        for callback in callbacks:
            if hasattr(callback, 'on_update_end'):
                callback.on_update_end(update_data)

# 4. For dashboard integration:
# The callback automatically sends data to the dashboard queue
# Dashboard can display:
# - Top features with importance scores
# - Branch importance breakdown
# - Dead feature counts
# - Feature interaction networks
# - Importance trends over time
# - Sample-level explanations

# 5. For WandB integration:
# All metrics are automatically logged with shap/ prefix
# Visualizations are logged as images
# Can create custom SHAP dashboard in WandB
"""