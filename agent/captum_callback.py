import torch
import numpy as np
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import io

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
        save_to_dashboard: bool = False,
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
        self.cached_action = None
        
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
            result = self._get_sample_state(trainer)
            if result is None:
                return
            
            state_dict, target_action = result
            if state_dict is None:
                return
                
            # Log what we're analyzing
            state_shapes = {k: v.shape for k, v in state_dict.items()}
            if target_action is not None:
                self.logger.info(f"🔍 Running Captum analysis (trigger: {trigger} {count}, target_action: {target_action}, state_shapes: {state_shapes})")
            else:
                self.logger.info(f"🔍 Running Captum analysis (trigger: {trigger} {count}, no target action - will use model predictions, state_shapes: {state_shapes})")
            
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
            
            # Log to WandB
            if self.save_to_wandb and self.wandb_run:
                self._log_to_wandb(results, trigger, count)
            
            
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
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            # Don't re-raise, just log and continue
    
    def _get_sample_state(self, trainer) -> Optional[Tuple[Dict[str, torch.Tensor], Optional[int]]]:
        """Get a sample state and action from buffer for Captum analysis.
        
        Returns:
            Tuple of (state_dict, target_action) or (None, None) if unavailable
        """
        try:
            # Strategy 1: Use prepared data from buffer (most reliable)
            if hasattr(trainer, "buffer") and hasattr(trainer.buffer, "states") and trainer.buffer.states is not None:
                batch_size = trainer.buffer.states["hf"].shape[0]
                if batch_size > 0:
                    # Get a sample from the middle of the batch (more representative than edges)
                    idx = min(batch_size // 2, batch_size - 1)
                    
                    state_dict = {}
                    for key in ["hf", "mf", "lf", "portfolio"]:
                        if key in trainer.buffer.states:
                            # Get single sample with proper batch dimension
                            tensor = trainer.buffer.states[key][idx:idx+1].to(trainer.device)
                            state_dict[key] = tensor
                    
                    # Validate state dictionary
                    if len(state_dict) < 4:
                        self.logger.warning(f"Incomplete state dict: {list(state_dict.keys())}")
                        return None, None
                    
                    # Get the corresponding action
                    target_action = None
                    if hasattr(trainer.buffer, "actions") and trainer.buffer.actions is not None:
                        if idx < trainer.buffer.actions.shape[0]:
                            action_tensor = trainer.buffer.actions[idx]
                            target_action = self._convert_action_to_linear_index(action_tensor)
                            if target_action is not None:
                                self.logger.debug(f"Extracted action from prepared buffer: action_tensor={action_tensor}, linear_idx={target_action}")
                            else:
                                self.logger.warning(f"Failed to convert action tensor: {action_tensor} (shape: {action_tensor.shape if hasattr(action_tensor, 'shape') else 'N/A'})")
                    
                    # Cache for future use
                    self.cached_state = state_dict
                    self.cached_action = target_action
                    return state_dict, target_action
            
            # Strategy 2: Use raw buffer experiences
            elif hasattr(trainer, "buffer") and hasattr(trainer.buffer, "buffer") and len(trainer.buffer.buffer) > 0:
                self.logger.debug(f"Using raw buffer with {len(trainer.buffer.buffer)} experiences")
                
                # Try multiple recent experiences to find a valid one
                for i in range(min(5, len(trainer.buffer.buffer))):
                    experience_idx = -(i + 1)  # -1, -2, -3, -4, -5
                    experience = trainer.buffer.buffer[experience_idx]
                    
                    state_dict = self._extract_state_from_experience(experience, trainer.device)
                    target_action = self._extract_action_from_experience(experience)
                    
                    if state_dict and len(state_dict) >= 4:
                        self.logger.debug(f"Successfully extracted state from experience {experience_idx}")
                        # Cache for future use
                        self.cached_state = state_dict
                        self.cached_action = target_action
                        return state_dict, target_action
                    else:
                        self.logger.debug(f"Experience {experience_idx} has incomplete state: {list(state_dict.keys()) if state_dict else 'None'}")
            
            # Strategy 3: Use cached state
            elif self.cached_state is not None:
                self.logger.debug("Using cached state for Captum analysis")
                return self.cached_state, getattr(self, 'cached_action', None)
            
            # No data available
            self.logger.warning("No valid state data available for Captum analysis")
            return None, None
                
        except Exception as e:
            self.logger.error(f"Error getting sample state: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None, None
    
    def _extract_state_from_experience(self, experience: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """Extract and format state from a buffer experience."""
        state_dict = {}
        
        if "state" not in experience:
            self.logger.debug("No 'state' key in experience")
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
                    if tensor.dim() == 1:  # [feat_dim] -> [1, 1, feat_dim]
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                    elif tensor.dim() == 2:  # [seq_len, feat_dim] -> [1, seq_len, feat_dim]
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() > 3:
                        self.logger.warning(f"Unexpected tensor dimension for {key}: {tensor.shape}")
                        continue
                    
                    # Basic sanity check
                    if tensor.numel() == 0:
                        self.logger.warning(f"Empty tensor for {key}")
                        continue
                        
                    state_dict[key] = tensor
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {key} in experience state: {e}")
            else:
                self.logger.debug(f"Missing {key} in state data")
                
        return state_dict
    
    def _extract_action_from_experience(self, experience: Dict) -> Optional[int]:
        """Extract action from a buffer experience and convert to linear index."""
        if "action" not in experience:
            self.logger.debug("No 'action' key in experience")
            return None
            
        action = experience["action"]
        linear_action = self._convert_action_to_linear_index(action)
        
        if linear_action is not None:
            self.logger.debug(f"Converted action {action} -> linear index {linear_action}")
        else:
            self.logger.warning(f"Failed to convert action from experience: {action} (type: {type(action)})")
            
        return linear_action
    
    @staticmethod
    def test_action_conversion():
        """Test the action conversion logic with various input formats."""
        import torch
        import numpy as np
        from feature.attribution.captum_attribution import AttributionConfig
        
        callback = CaptumCallback(
            config=AttributionConfig(),
            enabled=False
        )
        
        # Test cases: (input, expected_output)
        test_cases = [
            # Tensor formats
            (torch.tensor([0, 0]), 0),  # HOLD, 25%
            (torch.tensor([1, 1]), 5),  # BUY, 50%
            (torch.tensor([2, 3]), 11), # SELL, 100%
            (torch.tensor([[1, 2]]), 6), # BUY, 75% (batched)
            (torch.tensor([7]), 7),      # Linear index
            
            # NumPy formats
            (np.array([0, 1]), 1),       # HOLD, 50%
            (np.array([2, 2]), 10),      # SELL, 75%
            (np.array([[1, 0]]), 4),     # BUY, 25% (batched)
            (np.array([9]), 9),          # Linear index
            
            # List/tuple formats
            ([0, 2], 2),                 # HOLD, 75%
            ((1, 3), 7),                 # BUY, 100%
            
            # Scalar formats
            (5, 5),                      # Linear index
            (11, 11),                    # Max linear index
        ]
        
        print("Testing action conversion logic:")
        for i, (input_action, expected) in enumerate(test_cases):
            result = callback._convert_action_to_linear_index(input_action)
            status = "✓" if result == expected else "✗"
            print(f"  {status} Test {i+1}: {input_action} -> {result} (expected {expected})")
            
        # Test invalid cases
        invalid_cases = [
            torch.tensor([3, 0]),  # Invalid action type
            torch.tensor([1, 4]),  # Invalid position size
            torch.tensor([12]),    # Invalid linear index
            "invalid",             # Invalid type
        ]
        
        print("\nTesting invalid inputs:")
        for i, invalid_input in enumerate(invalid_cases):
            result = callback._convert_action_to_linear_index(invalid_input)
            status = "✓" if result is None else "✗"
            print(f"  {status} Invalid {i+1}: {invalid_input} -> {result} (expected None)")
        
        print("\nAction space mapping (for reference):")
        for action_type in range(3):
            for position_size in range(4):
                linear_idx = action_type * 4 + position_size
                action_names = ["HOLD", "BUY", "SELL"]
                size_names = ["25%", "50%", "75%", "100%"]
                print(f"  [{action_type}, {position_size}] -> {linear_idx} ({action_names[action_type]}, {size_names[position_size]})")
    
    def _convert_action_to_linear_index(self, action: Any) -> Optional[int]:
        """Convert various action formats to linear index (0-11).
        
        The trading environment uses MultiDiscrete([3, 4]) action space:
        - 3 action types: HOLD=0, BUY=1, SELL=2
        - 4 position sizes: 25%=0, 50%=1, 75%=2, 100%=3
        - Linear index = action_type * 4 + position_size
        
        Handles all formats found in the buffer:
        - Tensor with shape [2] -> [action_type, position_size]
        - Tensor with shape [1, 2] -> batched format
        - NumPy array with same shapes
        - List/tuple with 2 elements
        - Single integer (already linear)
        """
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
                if 0 <= linear_idx <= 11:  # Valid range check
                    return linear_idx
                else:
                    self.logger.warning(f"Linear action index {linear_idx} out of range [0, 11]")
                    return None
            else:
                self.logger.warning(f"Unsupported action type: {type(action)}")
                return None
            
            # Handle different array shapes
            if action_np.size == 1:
                # Single element - treat as linear index
                linear_idx = int(action_np.item())
                if 0 <= linear_idx <= 11:
                    return linear_idx
                else:
                    self.logger.warning(f"Linear action index {linear_idx} out of range [0, 11]")
                    return None
                    
            elif action_np.size == 2:
                # Two elements - could be flat [action_type, position_size] or [batch=1, linear_idx]
                if action_np.ndim == 1:
                    # [action_type, position_size]
                    action_type, position_size = int(action_np[0]), int(action_np[1])
                else:
                    # Could be [[action_type, position_size]] or [batch, linear_idx]
                    action_flat = action_np.flatten()
                    if len(action_flat) == 2:
                        action_type, position_size = int(action_flat[0]), int(action_flat[1])
                    else:
                        self.logger.warning(f"Unexpected action shape: {action_np.shape}")
                        return None
                
                # Validate ranges
                if not (0 <= action_type <= 2):
                    self.logger.warning(f"Action type {action_type} out of range [0, 2]")
                    return None
                if not (0 <= position_size <= 3):
                    self.logger.warning(f"Position size {position_size} out of range [0, 3]")
                    return None
                
                # Convert to linear index
                linear_idx = action_type * 4 + position_size
                return linear_idx
                
            elif action_np.shape[-1] == 2:  # Batched format [..., 2]
                # Take the first sample if batched
                if action_np.ndim == 2:
                    action_type, position_size = int(action_np[0, 0]), int(action_np[0, 1])
                else:
                    # Flatten and take first two
                    action_flat = action_np.flatten()
                    action_type, position_size = int(action_flat[0]), int(action_flat[1])
                
                # Validate ranges
                if not (0 <= action_type <= 2):
                    self.logger.warning(f"Action type {action_type} out of range [0, 2]")
                    return None
                if not (0 <= position_size <= 3):
                    self.logger.warning(f"Position size {position_size} out of range [0, 3]")
                    return None
                
                # Convert to linear index
                linear_idx = action_type * 4 + position_size
                return linear_idx
                
            else:
                self.logger.warning(f"Unexpected action shape: {action_np.shape}, size: {action_np.size}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting action to linear index: {e}")
            self.logger.debug(f"Action details: type={type(action)}, value={action}")
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
                
                if results.get('visualizations'):
                    self.logger.info(f"Uploaded {len(results['visualizations'])} visualizations to W&B")
                else:
                    self.logger.debug("No visualizations to upload to W&B")
            
            # Log predictions if available
            if "predictions" in results:
                if "action_probs" in results["predictions"]:
                    probs = results["predictions"]["action_probs"]
                    if isinstance(probs, np.ndarray):
                        for i, p in enumerate(probs.flatten()[:10]):  # First 10 actions
                            log_dict[f"captum/action_prob_{i}"] = p
                
                if "value" in results["predictions"]:
                    value = results["predictions"]["value"]
                    # Handle numpy arrays properly
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
