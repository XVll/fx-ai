import torch
import numpy as np
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

from agent.callbacks import BaseCallback
from feature.attribution.captum_attribution import (
    CaptumAttributionAnalyzer,
    AttributionConfig,
)


class CaptumCallback(BaseCallback):
    """Callback for feature attribution analysis using Captum during training.
    
    This callback integrates Captum attribution analysis into the training loop,
    providing insights into feature importance for both training and evaluation.
    """
    
    def __init__(
        self,
        config: AttributionConfig,
        analyze_every_n_episodes: int = 10,
        analyze_every_n_updates: int = 5,
        save_to_wandb: bool = True,
        save_to_dashboard: bool = True,
        feature_names: Optional[Dict[str, List[str]]] = None,
        output_dir: str = "outputs/captum",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.config = config
        self.analyze_every_n_episodes = analyze_every_n_episodes
        self.analyze_every_n_updates = analyze_every_n_updates
        self.save_to_wandb = save_to_wandb
        self.save_to_dashboard = save_to_dashboard
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.analyzer = None
        self.wandb_run = None
        
        # Track analysis history
        self.analysis_count = 0
        self.episode_analyses = []
        self.update_analyses = []
        
        # Performance tracking
        self.analysis_times = []
        
        # State caching to avoid env.reset() issues
        self.cached_state = None
        
    def on_training_start(self, config: Dict[str, Any]):
        """Initialize Captum analyzer with the model."""
        self.logger.info("🔍 Initializing Captum attribution analyzer")
        
        # Store trainer reference if available
        self.trainer = config.get("trainer")
        
        # Get model from config
        model = config.get("model")
        if model is None:
            self.logger.error("No model found in training config, disabling Captum callback")
            self.enabled = False
            return
        
        # Create analyzer
        self.analyzer = CaptumAttributionAnalyzer(
            model=model,
            config=self.config,
            feature_names=self.feature_names,
            logger=self.logger,
        )
        
        # Get WandB run if available
        if self.save_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    self.wandb_run = wandb.run
                    self.logger.info("📁 WandB integration enabled for Captum")
            except ImportError:
                self.logger.warning("WandB not available, skipping WandB logging")
                self.save_to_wandb = False
    
    def on_episode_end(self, episode_num: int, episode_data: Dict[str, Any]):
        """Analyze attributions at episode end if scheduled."""
        if self.analyze_every_n_episodes is None:
            self.logger.debug(f"Episode {episode_num} completed - episode analysis disabled")
            return
            
        self.logger.info(f"Episode {episode_num} completed, checking for Captum analysis (every {self.analyze_every_n_episodes})")
        if episode_num % self.analyze_every_n_episodes == 0:
            self.logger.info(f"Triggering Captum analysis for episode {episode_num}")
            self._perform_analysis_from_episode_data(episode_num, episode_data)
        else:
            self.logger.info(f"Skipping Captum analysis for episode {episode_num}")
    
    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Analyze attributions after PPO update if scheduled."""
        if self.analyze_every_n_updates is None:
            self.logger.debug(f"Update {update_num} completed - update analysis disabled")
            return
            
        self.logger.info(f"Update {update_num} completed, checking for Captum analysis (every {self.analyze_every_n_updates})")
        if update_num % self.analyze_every_n_updates == 0:
            self.logger.info(f"Triggering Captum analysis for update {update_num}")
            self._perform_analysis_from_update_data(update_num, update_metrics)
        else:
            self.logger.info(f"Skipping Captum analysis for update {update_num}")
    
    def _perform_analysis_from_episode_data(self, episode_num: int, episode_data: Dict[str, Any]):
        """Perform attribution analysis from episode data (new callback system)."""
        try:
            # Try to get trainer from episode data or use stored reference
            trainer = episode_data.get('trainer') or getattr(self, 'trainer', None)
            if trainer is None:
                self.logger.warning("No trainer available for Captum analysis")
                return
            
            self._perform_analysis(trainer, "episode", episode_num)
            
        except Exception as e:
            self.logger.error(f"Error in episode-based Captum analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _perform_analysis_from_update_data(self, update_num: int, update_metrics: Dict[str, Any]):
        """Perform attribution analysis from update data (new callback system)."""
        try:
            # Try to get trainer from update data or use stored reference
            trainer = update_metrics.get('trainer') or getattr(self, 'trainer', None)
            if trainer is None:
                self.logger.warning("No trainer available for Captum analysis")
                return
            
            self._perform_analysis(trainer, "update", update_num)
            
        except Exception as e:
            self.logger.error(f"Error in update-based Captum analysis: {str(e)}")
            import traceback
            traceback.print_exc()

    def _perform_analysis(self, trainer, trigger: str, count: int):
        """Perform attribution analysis on current model state."""
        try:
            start_time = datetime.now()
            
            # Get a sample from the environment
            state_dict = self._get_sample_state(trainer)
            if state_dict is None:
                return
            
            # Run attribution analysis
            self.logger.info(f"🔍 Running Captum analysis (trigger: {trigger} {count})")
            results = self.analyzer.analyze_sample(state_dict)
            
            # Track analysis
            self.analysis_count += 1
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_times.append(analysis_time)
            
            # Store results
            if trigger == "episode":
                self.episode_analyses.append(results)
            else:
                self.update_analyses.append(results)
            
            # Log to WandB
            if self.save_to_wandb and self.wandb_run:
                self._log_to_wandb(results, trigger, count)
            
            # Send to dashboard
            if self.save_to_dashboard and hasattr(trainer, "callback_manager"):
                self._send_to_dashboard(trainer, results, trigger, count)
            
            # Save periodic reports
            if self.analysis_count % 20 == 0:
                self._save_analysis_report(trainer)
            
            self.logger.info(
                f"✅ Captum analysis complete in {analysis_time:.2f}s "
                f"(total analyses: {self.analysis_count})"
            )
            
        except Exception as e:
            self.logger.error(f"Error in Captum analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _get_sample_state(self, trainer) -> Optional[Dict[str, torch.Tensor]]:
        """Get a sample state from buffer with caching fallback (no env.reset)."""
        try:
            # Try to get from replay buffer first
            if hasattr(trainer, "buffer") and trainer.buffer.get_size() > 0:
                # Get the most recent state from buffer
                recent_experience = trainer.buffer.buffer[-1]  # Get last experience
                if "state" in recent_experience:
                    state_dict = {}
                    state_data = recent_experience["state"]
                    for key in ["hf", "mf", "lf", "portfolio"]:
                        if key in state_data:
                            if isinstance(state_data[key], torch.Tensor):
                                tensor = state_data[key].to(trainer.device)
                            else:
                                tensor = torch.as_tensor(
                                    state_data[key], dtype=torch.float32
                                ).to(trainer.device)
                            # Ensure proper dimensions
                            if tensor.dim() == 2:  # [seq_len, feat_dim]
                                tensor = tensor.unsqueeze(0)  # [1, seq_len, feat_dim]
                            state_dict[key] = tensor
                    
                    # Cache this good state for future use
                    self.cached_state = state_dict
                    self.logger.debug("Cached fresh state from buffer for Captum analysis")
                    return state_dict
            
            # Use cached state if buffer is empty
            if self.cached_state is not None:
                self.logger.info("Using cached state for Captum analysis (buffer empty)")
                return self.cached_state
            
            # No data available - skip analysis
            self.logger.warning("No state available for Captum analysis - no buffer data and no cached state")
            return None
                
        except Exception as e:
            self.logger.error(f"Error getting sample state: {str(e)}")
            return None
    
    def _log_to_wandb(self, results: Dict, trigger: str, count: int):
        """Log attribution results to WandB."""
        try:
            import wandb
            
            # Log scalar metrics
            log_dict = {
                f"captum/{trigger}_count": count,
                f"captum/analysis_count": self.analysis_count,
            }
            
            # Log branch importance
            if "branch_importance" in results:
                for method_name, branches in results["branch_importance"].items():
                    for branch, importance in branches.items():
                        log_dict[f"captum/{method_name}/{branch}_importance"] = importance
            
            # Log top attributions as tables
            if "top_attributions" in results:
                for method_name, branches in results["top_attributions"].items():
                    for branch, features in branches.items():
                        if features:
                            # Create table data
                            table_data = []
                            for feat in features:
                                table_data.append([
                                    feat["name"],
                                    feat["index"],
                                    feat["importance"],
                                ])
                            
                            # Log as WandB table
                            table = wandb.Table(
                                columns=["Feature", "Index", "Importance"],
                                data=table_data,
                            )
                            log_dict[f"captum/{method_name}/{branch}_top_features"] = table
            
            # Log visualizations with better naming
            if "visualizations" in results:
                for viz_path in results["visualizations"]:
                    try:
                        # Parse visualization type from filename
                        filename = Path(viz_path).stem
                        if "branches" in filename:
                            viz_type = "branch_heatmap"
                        elif "timeseries" in filename:
                            if "hf" in filename:
                                viz_type = "hf_timeseries"
                            elif "mf" in filename:
                                viz_type = "mf_timeseries"
                            else:
                                viz_type = "timeseries"
                        elif "aggregated" in filename:
                            viz_type = "aggregated_importance"
                        else:
                            viz_type = "unknown"
                        
                        # Extract method name if present
                        method_part = filename.split('_')[0]
                        if method_part in ['saliency', 'deep', 'integrated', 'gradient']:
                            log_key = f"captum/{method_part}/{viz_type}"
                        else:
                            log_key = f"captum/{viz_type}"
                            
                        # Upload image to WandB
                        log_dict[log_key] = wandb.Image(viz_path)
                        self.logger.debug(f"Uploaded {viz_type} to W&B: {log_key}")
                    except Exception as e:
                        self.logger.error(f"Error uploading visualization {viz_path}: {str(e)}")
                
                self.logger.info(f"Uploaded {len(results['visualizations'])} visualizations to W&B")
            
            # Log predictions if available
            if "predictions" in results:
                if "action_probs" in results["predictions"]:
                    probs = results["predictions"]["action_probs"]
                    if isinstance(probs, np.ndarray):
                        for i, p in enumerate(probs.flatten()[:10]):  # First 10 actions
                            log_dict[f"captum/action_prob_{i}"] = p
                
                if "value" in results["predictions"]:
                    log_dict["captum/value_estimate"] = float(
                        results["predictions"]["value"]
                    )
            
            # Log summary statistics periodically
            if self.analysis_count % 10 == 0:
                summary = self.analyzer.get_summary_statistics()
                if "branch_importance_mean" in summary:
                    for branch, importance in summary["branch_importance_mean"].items():
                        log_dict[f"captum/summary/{branch}_mean_importance"] = importance
            
            wandb.log(log_dict)
            
        except Exception as e:
            self.logger.error(f"Error logging to WandB: {str(e)}")
    
    def _send_to_dashboard(self, trainer, results: Dict, trigger: str, count: int):
        """Send attribution results to the dashboard."""
        try:
            # Prepare dashboard data
            dashboard_data = {
                "analysis_count": self.analysis_count,
                "trigger": trigger,
                "trigger_count": count,
                "timestamp": results.get("timestamp", datetime.now().isoformat()),
            }
            
            # Add branch importance
            if "branch_importance" in results:
                # Get the first method's results for simplicity
                method_data = next(iter(results["branch_importance"].values()))
                dashboard_data["branch_importance"] = method_data
            
            # Add top features per branch
            if "top_attributions" in results:
                # Get the first method's results
                method_data = next(iter(results["top_attributions"].values()))
                top_features = {}
                for branch, features in method_data.items():
                    if features:
                        # Get top 3 features
                        top_features[branch] = [
                            {"name": f["name"], "score": f["importance"]}
                            for f in features[:3]
                        ]
                dashboard_data["top_features"] = top_features
            
            # Convert visualizations to base64 for dashboard
            if "visualizations" in results and results["visualizations"]:
                try:
                    # Use the first visualization
                    viz_path = results["visualizations"][0]
                    with open(viz_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        dashboard_data["visualization"] = f"data:image/png;base64,{img_data}"
                except Exception as e:
                    self.logger.error(f"Error encoding visualization: {str(e)}")
            
            # Add performance metrics
            dashboard_data["avg_analysis_time"] = (
                np.mean(self.analysis_times) if self.analysis_times else 0
            )
            
            # Send to dashboard via callback manager
            trainer.callback_manager.trigger(
                "on_custom_event", "captum_analysis", dashboard_data
            )
            
        except Exception as e:
            self.logger.error(f"Error sending to dashboard: {str(e)}")
    
    def _save_analysis_report(self, trainer):
        """Save periodic analysis report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"captum_report_{timestamp}.json"
            
            # Create comprehensive report
            report = {
                "metadata": {
                    "timestamp": timestamp,
                    "analysis_count": self.analysis_count,
                    "training_episodes": trainer.global_episode_counter,
                    "training_updates": trainer.global_update_counter,
                },
                "config": self.config.__dict__,
                "summary_statistics": self.analyzer.get_summary_statistics(),
                "performance": {
                    "avg_analysis_time": np.mean(self.analysis_times),
                    "total_analyses": self.analysis_count,
                },
            }
            
            # Add recent analyses
            if self.episode_analyses:
                report["recent_episode_analyses"] = [
                    self._format_analysis_for_report(a)
                    for a in self.episode_analyses[-5:]
                ]
            
            if self.update_analyses:
                report["recent_update_analyses"] = [
                    self._format_analysis_for_report(a)
                    for a in self.update_analyses[-5:]
                ]
            
            # Save report
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"💾 Saved Captum analysis report to {report_path}")
            
            # Also save the full analyzer report
            full_report_path = self.output_dir / f"captum_full_report_{timestamp}.json"
            self.analyzer.save_analysis_report(str(full_report_path))
            
        except Exception as e:
            self.logger.error(f"Error saving analysis report: {str(e)}")
    
    def _format_analysis_for_report(self, analysis: Dict) -> Dict:
        """Format analysis results for JSON report."""
        formatted = {
            "timestamp": analysis.get("timestamp"),
            "predictions": analysis.get("predictions", {}),
        }
        
        # Add branch importance
        if "branch_importance" in analysis:
            formatted["branch_importance"] = analysis["branch_importance"]
        
        # Add top features (simplified)
        if "top_attributions" in analysis:
            formatted["top_features"] = {}
            for method, branches in analysis["top_attributions"].items():
                formatted["top_features"][method] = {}
                for branch, features in branches.items():
                    if features:
                        formatted["top_features"][method][branch] = [
                            {"name": f["name"], "importance": f["importance"]}
                            for f in features[:2]
                        ]
        
        return formatted
    
    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Save final analysis report at training end."""
        self.logger.info("📊 Saving final Captum analysis report")
        
        # Use stored trainer reference
        if hasattr(self, 'trainer') and self.trainer is not None:
            self._save_analysis_report(self.trainer)
        
        # Log summary
        self.logger.info(
            f"🔍 Captum Analysis Summary:\n"
            f"   Total analyses: {self.analysis_count}\n"
            f"   Episode analyses: {len(self.episode_analyses)}\n"
            f"   Update analyses: {len(self.update_analyses)}\n"
            f"   Avg analysis time: {np.mean(self.analysis_times):.2f}s" if self.analysis_times else "   No analyses performed"
        )
        
        # Get final summary statistics
        if self.analyzer:
            summary = self.analyzer.get_summary_statistics()
            if summary:
                self.logger.info("📈 Feature Importance Summary:")
                if "branch_importance_mean" in summary:
                    for branch, importance in summary["branch_importance_mean"].items():
                        self.logger.info(f"   {branch.upper()}: {importance:.4f}")
                
                if "most_frequent_features" in summary:
                    self.logger.info("\n🏆 Most Important Features:")
                    for feature, count in summary["most_frequent_features"][:5]:
                        self.logger.info(f"   {feature}: appeared {count} times")
