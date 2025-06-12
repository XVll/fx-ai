"""
Captum Feature Attribution Callback for the new callback system.

Integrates Captum attribution analysis into the training loop with proper
lifecycle management and context-based component access.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import json

from callbacks.core.base import BaseCallback
from config.attribution.attribution_config import AttributionConfig

# Import feature registry for feature names
from feature.feature_registry import FeatureRegistry

# Import Captum components with graceful fallback
try:
    from core.attribution import (
        CaptumAttributionAnalyzer,
        MultiBranchTransformerWrapper,
        CAPTUM_AVAILABLE,
    )
except ImportError:
    CAPTUM_AVAILABLE = False
    CaptumAttributionAnalyzer = None
    MultiBranchTransformerWrapper = None

try:
    import wandb
except ImportError:
    wandb = None


class CaptumAttributionCallback(BaseCallback):
    """
    Feature attribution analysis callback using Captum.
    
    Provides insights into feature importance for both training and evaluation
    using state-of-the-art attribution methods.
    """
    
    def __init__(self, config: AttributionConfig, enabled: bool = True):
        """
        Initialize Captum attribution callback.
        
        Args:
            config: Attribution configuration with all settings
            enabled: Whether callback is active
        """
        super().__init__(name="CaptumAttribution", enabled=enabled and config.enabled, config=config)
        
        # Check if Captum is available
        if not CAPTUM_AVAILABLE:
            self.logger.warning("Captum not available, disabling attribution analysis")
            self.enabled = False
            return
        
        self.config: AttributionConfig = config
        self.analyzer: Optional[CaptumAttributionAnalyzer] = None
        
        # Analysis tracking
        self.analysis_count = 0
        self.episode_analyses = []
        self.update_analyses = []
        self.analysis_times = []
        
        # State caching to avoid env.reset() issues
        self.cached_state = None
        self.cached_action = None
        
        # Create output directory using PathManager
        from core.path_manager import get_path_manager
        path_manager = get_path_manager()
        self.output_dir = path_manager.experiment_analysis_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized Captum attribution callback (enabled={enabled})")
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize Captum analyzer with model from context."""
        if not self.enabled:
            return
            
        self.logger.info("🔍 Initializing Captum attribution analyzer")
        
        # Get model from context
        model = context.get("model")
        if model is None:
            self.logger.error("No model found in training context, disabling Captum")
            self.enabled = False
            return
        
        # Store component references from context
        self.trainer = context.get("trainer")
        self.environment = context.get("environment")
        self.data_manager = context.get("data_manager")
        
        # Get feature names from registry
        feature_names = self._get_feature_names_from_registry()
        
        # Create attribution analyzer with Hydra config directly
        try:
            self.analyzer = CaptumAttributionAnalyzer(
                model=model,
                config=self.config,
                feature_names=feature_names,
                logger=self.logger,
            )
            
            self.logger.info("✅ Captum attribution analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Captum analyzer: {e}")
            self.enabled = False
            return
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Analyze attributions at episode end if scheduled."""
        if not self.enabled or not self.analyzer:
            return
            
        episode_num = context.get("episode_num", 0)
        
        if (self.config.analyze_every_n_episodes is not None and 
            episode_num % self.config.analyze_every_n_episodes == 0):
            self.logger.info(f"Triggering Captum analysis for episode {episode_num}")
            self._perform_analysis(context, "episode", episode_num)
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Analyze attributions after PPO update if scheduled.""" 
        if not self.enabled or not self.analyzer:
            return
            
        update_num = context.get("update_num", 0)
        
        if (self.config.analyze_every_n_updates is not None and
            update_num % self.config.analyze_every_n_updates == 0):
            self.logger.info(f"Triggering Captum analysis for update {update_num}")
            self._perform_analysis(context, "update", update_num)
    
    def _perform_analysis(self, context: Dict[str, Any], trigger: str, count: int) -> None:
        """Perform attribution analysis on current model state."""
        try:
            start_time = datetime.now()
            
            # Get sample state from context
            result = self._get_sample_state(context)
            if result is None:
                return
                
            state_dict, target_action = result
            if state_dict is None:
                return
                
            # Log analysis details
            state_shapes = {k: v.shape for k, v in state_dict.items()}
            if target_action is not None:
                self.logger.info(f"🔍 Running Captum analysis (trigger: {trigger} {count}, target_action: {target_action}, state_shapes: {state_shapes})")
            else:
                self.logger.info(f"🔍 Running Captum analysis (trigger: {trigger} {count}, no target action, state_shapes: {state_shapes})")
            
            # Run attribution analysis with error handling
            try:
                results = self.analyzer.analyze_sample(state_dict, target_action=target_action)
            except Exception as e:
                self.logger.error(f"Captum analysis failed: {str(e)}")
                # Try again without target action for debugging
                if target_action is not None:
                    self.logger.info("Retrying Captum analysis without target action")
                    try:
                        results = self.analyzer.analyze_sample(state_dict, target_action=None)
                        self.logger.info("Captum analysis succeeded without target action")
                    except Exception as e2:
                        self.logger.error(f"Captum analysis failed even without target action: {str(e2)}")
                        return
                else:
                    return
            
            # Track analysis
            self.analysis_count += 1
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_times.append(analysis_time)
            
            # Store results
            if trigger == "episode":
                self.episode_analyses.append(results)
            else:
                self.update_analyses.append(results)
            
            # Log to WandB if enabled
            if self.config.save_to_wandb and wandb is not None:
                self._log_to_wandb(results, trigger, count)
            
            # Save periodic reports
            if self.analysis_count % 20 == 0:
                self._save_analysis_report(context)
            
            self.logger.info(
                f"✅ Captum analysis complete in {analysis_time:.2f}s "
                f"(total analyses: {self.analysis_count})"
            )
            
        except Exception as e:
            self.logger.error(f"Error in Captum analysis: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    
    def _get_sample_state(self, context: Dict[str, Any]) -> Optional[Tuple[Dict[str, torch.Tensor], Optional[int]]]:
        """Get sample state and action from buffer for analysis."""
        try:
            # Get trainer from context
            trainer = context.get("trainer") or self.trainer
            if trainer is None:
                self.logger.warning("No trainer available for Captum analysis")
                return None, None
            
            # Strategy 1: Use prepared data from buffer (most reliable)
            if hasattr(trainer, "buffer") and hasattr(trainer.buffer, "states") and trainer.buffer.states is not None:
                batch_size = trainer.buffer.states["hf"].shape[0]
                if batch_size > 0:
                    # Get sample from middle of batch
                    idx = min(batch_size // 2, batch_size - 1)
                    
                    state_dict = {}
                    for key in ["hf", "mf", "lf", "portfolio"]:
                        if key in trainer.buffer.states:
                            tensor = trainer.buffer.states[key][idx:idx+1].to(trainer.device)
                            state_dict[key] = tensor
                    
                    # Validate state dictionary
                    if len(state_dict) < 4:
                        self.logger.warning(f"Incomplete state dict: {list(state_dict.keys())}")
                        return None, None
                    
                    # Get corresponding action
                    target_action = None
                    if hasattr(trainer.buffer, "actions") and trainer.buffer.actions is not None:
                        if idx < trainer.buffer.actions.shape[0]:
                            action_tensor = trainer.buffer.actions[idx]
                            target_action = self._convert_action_to_linear_index(action_tensor)
                            if target_action is not None:
                                self.logger.debug(f"Extracted action: tensor={action_tensor}, linear_idx={target_action}")
                    
                    # Cache for future use
                    self.cached_state = state_dict
                    self.cached_action = target_action
                    return state_dict, target_action
            
            # Strategy 2: Use raw buffer experiences
            elif hasattr(trainer, "buffer") and hasattr(trainer.buffer, "buffer") and len(trainer.buffer.buffer) > 0:
                self.logger.debug(f"Using raw buffer with {len(trainer.buffer.buffer)} experiences")
                
                # Try multiple recent experiences
                for i in range(min(5, len(trainer.buffer.buffer))):
                    experience_idx = -(i + 1)
                    experience = trainer.buffer.buffer[experience_idx]
                    
                    state_dict = self._extract_state_from_experience(experience, trainer.device)
                    target_action = self._extract_action_from_experience(experience)
                    
                    if state_dict and len(state_dict) >= 4:
                        self.logger.debug(f"Successfully extracted state from experience {experience_idx}")
                        self.cached_state = state_dict
                        self.cached_action = target_action
                        return state_dict, target_action
            
            # Strategy 3: Use cached state
            elif self.cached_state is not None:
                self.logger.debug("Using cached state for Captum analysis")
                return self.cached_state, self.cached_action
            
            # No data available
            self.logger.warning("No valid state data available for Captum analysis")
            return None, None
                
        except Exception as e:
            self.logger.error(f"Error getting sample state: {str(e)}")
            return None, None
    
    def _extract_state_from_experience(self, experience: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """Extract and format state from buffer experience."""
        state_dict = {}
        
        if "state" not in experience:
            return state_dict
            
        state_data = experience["state"]
        
        for key in ["hf", "mf", "lf", "portfolio"]:
            if key in state_data:
                try:
                    if isinstance(state_data[key], torch.Tensor):
                        tensor = state_data[key].to(device)
                    elif isinstance(state_data[key], np.ndarray):
                        tensor = torch.from_numpy(state_data[key]).to(device, dtype=torch.float32)
                    else:
                        tensor = torch.as_tensor(state_data[key], dtype=torch.float32).to(device)
                    
                    # Ensure proper dimensions [batch, seq_len, feat_dim]
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                    elif tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() > 3:
                        continue
                    
                    if tensor.numel() == 0:
                        continue
                        
                    state_dict[key] = tensor
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {key} in experience state: {e}")
                    
        return state_dict
    
    def _extract_action_from_experience(self, experience: Dict) -> Optional[int]:
        """Extract action from buffer experience and convert to linear index."""
        if "action" not in experience:
            return None
            
        action = experience["action"]
        return self._convert_action_to_linear_index(action)
    
    def _convert_action_to_linear_index(self, action: Any) -> Optional[int]:
        """Convert various action formats to linear index (0-11)."""
        try:
            # Convert to numpy for consistent handling
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy()
            elif isinstance(action, np.ndarray):
                action_np = action
            elif isinstance(action, (list, tuple)):
                action_np = np.array(action)
            elif isinstance(action, (int, float)):
                # Already linear index
                linear_idx = int(action)
                if 0 <= linear_idx <= 11:
                    return linear_idx
                else:
                    return None
            else:
                return None
            
            # Handle different array shapes
            if action_np.size == 1:
                # Single element - treat as linear index
                linear_idx = int(action_np.item())
                if 0 <= linear_idx <= 11:
                    return linear_idx
                else:
                    return None
                    
            elif action_np.size == 2:
                # Two elements - [action_type, position_size]
                if action_np.ndim == 1:
                    action_type, position_size = int(action_np[0]), int(action_np[1])
                else:
                    action_flat = action_np.flatten()
                    if len(action_flat) == 2:
                        action_type, position_size = int(action_flat[0]), int(action_flat[1])
                    else:
                        return None
                
                # Validate ranges
                if not (0 <= action_type <= 2) or not (0 <= position_size <= 3):
                    return None
                
                # Convert to linear index
                return action_type * 4 + position_size
                
            elif action_np.shape[-1] == 2:  # Batched format
                # Take first sample if batched
                if action_np.ndim == 2:
                    action_type, position_size = int(action_np[0, 0]), int(action_np[0, 1])
                else:
                    action_flat = action_np.flatten()
                    action_type, position_size = int(action_flat[0]), int(action_flat[1])
                
                # Validate ranges
                if not (0 <= action_type <= 2) or not (0 <= position_size <= 3):
                    return None
                
                # Convert to linear index
                return action_type * 4 + position_size
                
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting action to linear index: {e}")
            return None
    
    def _get_feature_names_from_registry(self) -> Dict[str, List[str]]:
        """Get feature names from FeatureRegistry."""
        return {
            "hf": FeatureRegistry.get_feature_names("hf"),
            "mf": FeatureRegistry.get_feature_names("mf"), 
            "lf": FeatureRegistry.get_feature_names("lf"),
            "portfolio": FeatureRegistry.get_feature_names("portfolio"),
        }
    
    
    def _log_to_wandb(self, results: Dict, trigger: str, count: int) -> None:
        """Log attribution results to WandB."""
        if wandb is None or wandb.run is None:
            return
            
        try:
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
                            table_data = []
                            for feat in features:
                                table_data.append([
                                    feat["name"],
                                    feat["index"], 
                                    feat["importance"],
                                ])
                            
                            table = wandb.Table(
                                columns=["Feature", "Index", "Importance"],
                                data=table_data,
                            )
                            log_dict[f"captum/{method_name}/{branch}_top_features"] = table
            
            # Log visualizations
            if "visualizations" in results:
                for viz_path in results["visualizations"]:
                    try:
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
                        
                        method_part = filename.split('_')[0]
                        if method_part in ['saliency', 'deep', 'integrated', 'gradient']:
                            log_key = f"captum/{method_part}/{viz_type}"
                        else:
                            log_key = f"captum/{viz_type}"
                            
                        log_dict[log_key] = wandb.Image(viz_path)
                        
                    except Exception as e:
                        self.logger.error(f"Error uploading visualization {viz_path}: {str(e)}")
            
            # Log predictions
            if "predictions" in results:
                if "action_probs" in results["predictions"]:
                    probs = results["predictions"]["action_probs"]
                    if isinstance(probs, np.ndarray):
                        for i, p in enumerate(probs.flatten()[:10]):
                            log_dict[f"captum/action_prob_{i}"] = p
                
                if "value" in results["predictions"]:
                    value = results["predictions"]["value"]
                    if isinstance(value, np.ndarray):
                        value = value.item() if value.size == 1 else float(value.flatten()[0])
                    log_dict["captum/value_estimate"] = float(value)
            
            # Log summary statistics periodically
            if self.analysis_count % 10 == 0:
                summary = self.analyzer.get_summary_statistics()
                if "branch_importance_mean" in summary:
                    for branch, importance in summary["branch_importance_mean"].items():
                        log_dict[f"captum/summary/{branch}_mean_importance"] = importance
            
            wandb.log(log_dict)
            
        except Exception as e:
            self.logger.error(f"Error logging to WandB: {str(e)}")
    
    def _save_analysis_report(self, context: Dict[str, Any]) -> None:
        """Save periodic analysis report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"captum_report_{timestamp}.json"
            
            trainer = context.get("trainer") or self.trainer
            
            # Create comprehensive report
            report = {
                "metadata": {
                    "timestamp": timestamp,
                    "analysis_count": self.analysis_count,
                    "training_episodes": getattr(trainer, "global_episode_counter", 0),
                    "training_updates": getattr(trainer, "global_update_counter", 0),
                },
                "config": {
                    "methods": self.config.methods,
                    "n_steps": self.config.n_steps,
                    "baseline_type": self.config.baseline_type,
                    "analyze_branches": self.config.analyze_branches,
                },
                "summary_statistics": self.analyzer.get_summary_statistics() if self.analyzer else {},
                "performance": {
                    "avg_analysis_time": np.mean(self.analysis_times) if self.analysis_times else 0,
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
            
            # Also save full analyzer report
            if self.analyzer:
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
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Save final analysis report and summary statistics."""
        if not self.enabled:
            return
            
        self.logger.info("📊 Saving final Captum analysis report")
        
        # Save final report
        self._save_analysis_report(context)
        
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