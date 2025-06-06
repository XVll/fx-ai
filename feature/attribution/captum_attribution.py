import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json

# Captum imports
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    LayerConductance,
    LayerIntegratedGradients,
    NeuronConductance,
    Saliency,
    InputXGradient,
    FeatureAblation,
    ShapleyValueSampling,
)
from captum.attr._utils.visualization import visualize_image_attr_multiple


@dataclass
class AttributionConfig:
    """Configuration for Captum attribution methods."""
    
    # Attribution methods to use
    methods: List[str] = None  # ["integrated_gradients", "deep_lift", "gradient_shap"]
    
    # Method-specific parameters
    n_steps: int = 50  # Steps for integrated gradients
    n_samples: int = 25  # Samples for gradient SHAP
    
    # Analysis settings
    analyze_branches: bool = True  # Analyze each transformer branch separately
    analyze_fusion: bool = True  # Analyze attention fusion layer
    analyze_actions: bool = True  # Analyze action head attributions
    
    # Baseline configuration
    baseline_type: str = "zero"  # "zero", "mean", "random"
    
    # Visualization settings
    save_visualizations: bool = True
    visualization_dir: str = "outputs/captum"
    heatmap_threshold: float = 0.01  # Min attribution value to show
    
    # Performance settings
    batch_analysis: bool = False  # Analyze multiple samples at once
    max_batch_size: int = 32
    
    # Feature importance aggregation
    aggregate_features: bool = True  # Aggregate attributions by feature groups
    feature_groups: Dict[str, List[str]] = None  # Feature grouping definitions
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["integrated_gradients", "deep_lift"]
        
        if self.feature_groups is None:
            # Default feature groups based on your system
            self.feature_groups = {
                "price_action": ["price", "returns", "volatility"],
                "volume": ["volume", "vwap", "relative_volume"],
                "microstructure": ["spread", "imbalance", "order_flow"],
                "technical": ["ema", "rsi", "patterns"],
                "portfolio": ["position", "pnl", "risk"],
            }


class MultiBranchTransformerWrapper(nn.Module):
    """Wrapper to make MultiBranchTransformer compatible with Captum.
    
    Captum expects models with simple forward signatures, so we wrap
    the transformer to handle the complex input structure.
    """
    
    def __init__(self, model, target_output: str = "action"):
        super().__init__()
        self.model = model
        self.target_output = target_output  # "action", "value", or "both"
        
    def forward(self, hf_features, mf_features, lf_features, portfolio_features):
        """Forward pass with separated inputs for Captum compatibility."""
        # Reconstruct state dict
        state_dict = {
            "hf": hf_features,
            "mf": mf_features,
            "lf": lf_features,
            "portfolio": portfolio_features,
        }
        
        # Get model outputs
        action_params, value = self.model(state_dict)
        
        if self.target_output == "action":
            # For discrete actions, return action logits
            if len(action_params) == 2:
                # Combined logits for action type and size
                return torch.cat(action_params, dim=-1)
            else:
                return action_params[0]
        elif self.target_output == "value":
            return value
        else:  # "both"
            if len(action_params) == 2:
                return torch.cat([action_params[0], action_params[1], value], dim=-1)
            else:
                return torch.cat([action_params[0], value], dim=-1)


class CaptumAttributionAnalyzer:
    """Main class for analyzing feature attributions using Captum."""
    
    def __init__(
        self,
        model: nn.Module,
        config: AttributionConfig,
        feature_names: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.config = config
        self.feature_names = feature_names or self._get_default_feature_names()
        self.logger = logger or logging.getLogger(__name__)
        
        # Create visualization directory
        self.viz_dir = Path(self.config.visualization_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attribution methods
        self._init_attribution_methods()
        
        # Attribution history for analysis
        self.attribution_history = []
        
    def _get_default_feature_names(self) -> Dict[str, List[str]]:
        """Get default feature names based on actual FxAIv2 feature implementation."""
        # Real feature names from FxAIv2 SimpleFeatureManager
        hf_features = [
            "price_velocity", "price_acceleration", "tape_imbalance", "tape_aggression_ratio",
            "spread_compression", "quote_velocity", "quote_imbalance", 
            "volume_velocity", "volume_acceleration"
        ]
        
        mf_features = [
            # Candle features
            "1m_position_in_current_candle", "5m_position_in_current_candle",
            "1m_body_size_relative", "5m_body_size_relative",
            "1m_position_in_previous_candle", "5m_position_in_previous_candle",
            "1m_upper_wick_relative", "1m_lower_wick_relative",
            "5m_upper_wick_relative", "5m_lower_wick_relative",
            # EMA features
            "distance_to_ema9_1m", "distance_to_ema20_1m", "distance_to_ema9_5m", "distance_to_ema20_5m",
            "ema_interaction_pattern", "ema_crossover_dynamics", "ema_trend_alignment",
            # Swing features
            "swing_high_distance_1m", "swing_low_distance_1m", "swing_high_distance_5m", "swing_low_distance_5m",
            # Velocity/acceleration features
            "price_velocity_1m", "price_velocity_5m", "volume_velocity_1m", "volume_velocity_5m",
            "price_acceleration_1m", "price_acceleration_5m", "volume_acceleration_1m", "volume_acceleration_5m",
            # VWAP features
            "distance_to_vwap", "vwap_slope", "price_vwap_divergence",
            "vwap_interaction_dynamics", "vwap_breakout_quality", "vwap_mean_reversion_tendency",
            # Volume features
            "relative_volume", "volume_surge", "cumulative_volume_delta", "volume_momentum",
            # Professional indicators
            "professional_ema_system", "professional_vwap_analysis", 
            "professional_momentum_quality", "professional_volatility_regime",
            # Sequence patterns
            "trend_acceleration", "volume_pattern_evolution", "momentum_quality", "pattern_maturation",
            # Aggregated signals
            "mf_trend_consistency", "mf_volume_price_divergence", "mf_momentum_persistence",
            # Adaptive features
            "volatility_adjusted_momentum", "regime_relative_volume"
        ]
        
        lf_features = [
            # Range features
            "daily_range_position", "prev_day_range_position", "price_change_from_prev_close",
            # Level features
            "support_distance", "resistance_distance", "whole_dollar_proximity", "half_dollar_proximity",
            # Time features
            "market_session_type", "time_of_day_sin", "time_of_day_cos",
            # Market structure
            "halt_state", "time_since_halt", "distance_to_luld_up", "distance_to_luld_down", "luld_band_width",
            # Context features
            "session_progress", "market_stress", "session_volume_profile",
            # Adaptive features
            "adaptive_support_resistance",
            # HF summary features
            "hf_momentum_summary", "hf_volume_dynamics", "hf_microstructure_quality"
        ]
        
        portfolio_features = [
            "position_size_normalized",    # Feature 0: -1 to 1
            "unrealized_pnl_normalized",   # Feature 1: -2 to 2
            "time_in_position",            # Feature 2: 0 to 2
            "cash_ratio",                  # Feature 3: 0 to 2
            "session_pnl_percentage",      # Feature 4: -1 to 1
            "max_favorable_excursion",     # Feature 5: -2 to 2 (MFE)
            "max_adverse_excursion",       # Feature 6: -2 to 2 (MAE)
            "profit_giveback_ratio",       # Feature 7: -1 to 1
            "recovery_ratio",              # Feature 8: -1 to 1
            "trade_quality_score"          # Feature 9: -1 to 1
        ]
        
        # Get model config for dimensions if available
        if hasattr(self.model, "model_config"):
            config = self.model.model_config
            # Handle both dict and Pydantic model
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump()
            elif hasattr(config, "__dict__"):
                config_dict = config.__dict__
            else:
                config_dict = config
            
            # Pad or truncate feature lists to match model dimensions
            hf_dim = config_dict.get("hf_feat_dim", len(hf_features))
            mf_dim = config_dict.get("mf_feat_dim", len(mf_features))
            lf_dim = config_dict.get("lf_feat_dim", len(lf_features))
            portfolio_dim = config_dict.get("portfolio_feat_dim", len(portfolio_features))
            
            # Pad with generic names if needed
            while len(hf_features) < hf_dim:
                hf_features.append(f"hf_feat_{len(hf_features)}")
            while len(mf_features) < mf_dim:
                mf_features.append(f"mf_feat_{len(mf_features)}")
            while len(lf_features) < lf_dim:
                lf_features.append(f"lf_feat_{len(lf_features)}")
            while len(portfolio_features) < portfolio_dim:
                portfolio_features.append(f"portfolio_feat_{len(portfolio_features)}")
            
            # Truncate if needed
            hf_features = hf_features[:hf_dim]
            mf_features = mf_features[:mf_dim]
            lf_features = lf_features[:lf_dim]
            portfolio_features = portfolio_features[:portfolio_dim]
        
        return {
            "hf": hf_features,
            "mf": mf_features,
            "lf": lf_features,
            "portfolio": portfolio_features,
        }
    
    def _init_attribution_methods(self):
        """Initialize Captum attribution methods."""
        self.methods = {}
        
        # Create wrapped models for different targets
        action_model = MultiBranchTransformerWrapper(self.model, "action")
        value_model = MultiBranchTransformerWrapper(self.model, "value")
        
        # Initialize requested methods
        if "integrated_gradients" in self.config.methods:
            self.methods["integrated_gradients_action"] = IntegratedGradients(action_model)
            self.methods["integrated_gradients_value"] = IntegratedGradients(value_model)
            
        if "deep_lift" in self.config.methods:
            self.methods["deep_lift_action"] = DeepLift(action_model)
            self.methods["deep_lift_value"] = DeepLift(value_model)
            
        if "gradient_shap" in self.config.methods:
            self.methods["gradient_shap_action"] = GradientShap(action_model)
            self.methods["gradient_shap_value"] = GradientShap(value_model)
            
        if "saliency" in self.config.methods:
            self.methods["saliency_action"] = Saliency(action_model)
            self.methods["saliency_value"] = Saliency(value_model)
            
        if "input_x_gradient" in self.config.methods:
            self.methods["input_x_gradient_action"] = InputXGradient(action_model)
            self.methods["input_x_gradient_value"] = InputXGradient(value_model)
            
        # Layer-specific methods for branch analysis
        if self.config.analyze_branches:
            self._init_layer_methods()
            
        self.logger.info(f"🔍 Initialized {len(self.methods)} attribution methods")
    
    def _init_layer_methods(self):
        """Initialize layer-specific attribution methods."""
        # Layer conductance for each transformer branch
        action_model = MultiBranchTransformerWrapper(self.model, "action")
        
        if hasattr(self.model, "hf_encoder"):
            self.methods["layer_conductance_hf"] = LayerConductance(
                action_model, self.model.hf_encoder
            )
            
        if hasattr(self.model, "mf_encoder"):
            self.methods["layer_conductance_mf"] = LayerConductance(
                action_model, self.model.mf_encoder
            )
            
        if hasattr(self.model, "lf_encoder"):
            self.methods["layer_conductance_lf"] = LayerConductance(
                action_model, self.model.lf_encoder
            )
            
        if hasattr(self.model, "fusion") and self.config.analyze_fusion:
            self.methods["layer_conductance_fusion"] = LayerConductance(
                action_model, self.model.fusion
            )
    
    def _create_baseline(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create baseline inputs for attribution calculation."""
        baselines = {}
        
        for key, tensor in inputs.items():
            if self.config.baseline_type == "zero":
                baselines[key] = torch.zeros_like(tensor)
            elif self.config.baseline_type == "mean":
                # Use mean values from training data if available
                baselines[key] = tensor.mean(dim=0, keepdim=True).expand_as(tensor)
            elif self.config.baseline_type == "random":
                baselines[key] = torch.randn_like(tensor) * 0.1
            else:
                baselines[key] = torch.zeros_like(tensor)
                
        return baselines
    
    def analyze_sample(
        self,
        state_dict: Dict[str, torch.Tensor],
        target_action: Optional[torch.Tensor] = None,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """Analyze feature attributions for a single sample.
        
        Args:
            state_dict: Model input state dictionary
            target_action: Optional target action for attribution
            return_raw: Return raw attribution tensors
            
        Returns:
            Dictionary containing attribution results and visualizations
        """
        self.logger.info("🔍 Analyzing feature attributions for sample")
        
        # Prepare inputs
        hf_features = state_dict["hf"]
        mf_features = state_dict["mf"]
        lf_features = state_dict["lf"]
        portfolio_features = state_dict["portfolio"]
        
        # Ensure batch dimension
        if hf_features.dim() == 2:
            hf_features = hf_features.unsqueeze(0)
        if mf_features.dim() == 2:
            mf_features = mf_features.unsqueeze(0)
        if lf_features.dim() == 2:
            lf_features = lf_features.unsqueeze(0)
        if portfolio_features.dim() == 2:
            portfolio_features = portfolio_features.unsqueeze(0)
        
        # Create baselines
        baselines = self._create_baseline({
            "hf": hf_features,
            "mf": mf_features,
            "lf": lf_features,
            "portfolio": portfolio_features,
        })
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "attributions": {},
            "aggregated": {},
            "visualizations": [],
        }
        
        # Run each attribution method
        for method_name, method in self.methods.items():
            try:
                if "integrated_gradients" in method_name:
                    attributions = self._run_integrated_gradients(
                        method, 
                        (hf_features, mf_features, lf_features, portfolio_features),
                        baselines,
                        target_action,
                    )
                elif "deep_lift" in method_name:
                    attributions = self._run_deep_lift(
                        method,
                        (hf_features, mf_features, lf_features, portfolio_features),
                        baselines,
                        target_action,
                    )
                elif "gradient_shap" in method_name:
                    attributions = self._run_gradient_shap(
                        method,
                        (hf_features, mf_features, lf_features, portfolio_features),
                        baselines,
                        target_action,
                    )
                elif "layer_conductance" in method_name:
                    attributions = self._run_layer_conductance(
                        method,
                        (hf_features, mf_features, lf_features, portfolio_features),
                        baselines,
                        target_action,
                    )
                else:
                    # Generic attribution
                    attributions = method.attribute(
                        inputs=(hf_features, mf_features, lf_features, portfolio_features),
                        target=target_action,
                    )
                
                # Store results
                results["attributions"][method_name] = {
                    "hf": attributions[0].cpu().numpy() if isinstance(attributions, tuple) else None,
                    "mf": attributions[1].cpu().numpy() if isinstance(attributions, tuple) and len(attributions) > 1 else None,
                    "lf": attributions[2].cpu().numpy() if isinstance(attributions, tuple) and len(attributions) > 2 else None,
                    "portfolio": attributions[3].cpu().numpy() if isinstance(attributions, tuple) and len(attributions) > 3 else None,
                }
                
            except Exception as e:
                self.logger.error(f"Error in {method_name}: {str(e)}")
                continue
        
        # Aggregate attributions
        if self.config.aggregate_features:
            results["aggregated"] = self._aggregate_attributions(results["attributions"])
        
        # Create visualizations
        if self.config.save_visualizations:
            viz_paths = self._create_visualizations(results, state_dict)
            results["visualizations"] = viz_paths
        
        # Store in history
        self.attribution_history.append(results)
        
        # Get model predictions for context
        with torch.no_grad():
            if isinstance(self.model, MultiBranchTransformerWrapper):
                # For wrapper, pass arguments separately
                action_output = self.model(
                    state_dict["hf"], 
                    state_dict["mf"], 
                    state_dict["lf"], 
                    state_dict["portfolio"]
                )
                # Since wrapper can return different outputs, we need to get raw model output
                action_params, value = self.model.model(state_dict)
            else:
                action_params, value = self.model(state_dict)
            if len(action_params) == 2:
                action_probs = (
                    torch.softmax(action_params[0], dim=-1),
                    torch.softmax(action_params[1], dim=-1),
                )
            else:
                action_probs = torch.softmax(action_params[0], dim=-1)
            
            results["predictions"] = {
                "action_probs": action_probs[0].cpu().numpy() if isinstance(action_probs, tuple) else action_probs.cpu().numpy(),
                "value": value.cpu().numpy(),
            }
        
        if not return_raw:
            # Convert to more compact format
            results = self._format_results(results)
        
        return results
    
    def _run_integrated_gradients(
        self, 
        method, 
        inputs: Tuple[torch.Tensor, ...], 
        baselines: Dict[str, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Run integrated gradients attribution."""
        baseline_tuple = (
            baselines["hf"],
            baselines["mf"],
            baselines["lf"],
            baselines["portfolio"],
        )
        
        return method.attribute(
            inputs=inputs,
            baselines=baseline_tuple,
            target=target,
            n_steps=self.config.n_steps,
        )
    
    def _run_deep_lift(self, method, inputs, baselines, target):
        """Run DeepLift attribution."""
        baseline_tuple = (
            baselines["hf"],
            baselines["mf"],
            baselines["lf"],
            baselines["portfolio"],
        )
        
        return method.attribute(
            inputs=inputs,
            baselines=baseline_tuple,
            target=target,
        )
    
    def _run_gradient_shap(self, method, inputs, baselines, target):
        """Run Gradient SHAP attribution."""
        # Create multiple baseline samples
        baseline_samples = []
        for _ in range(self.config.n_samples):
            baseline_samples.append((
                torch.randn_like(inputs[0]) * 0.1,
                torch.randn_like(inputs[1]) * 0.1,
                torch.randn_like(inputs[2]) * 0.1,
                torch.randn_like(inputs[3]) * 0.1,
            ))
        
        # Stack baselines
        stacked_baselines = (
            torch.cat([b[0] for b in baseline_samples], dim=0),
            torch.cat([b[1] for b in baseline_samples], dim=0),
            torch.cat([b[2] for b in baseline_samples], dim=0),
            torch.cat([b[3] for b in baseline_samples], dim=0),
        )
        
        return method.attribute(
            inputs=inputs,
            baselines=stacked_baselines,
            target=target,
            n_samples=self.config.n_samples,
        )
    
    def _run_layer_conductance(self, method, inputs, baselines, target):
        """Run layer conductance attribution."""
        baseline_tuple = (
            baselines["hf"],
            baselines["mf"],
            baselines["lf"],
            baselines["portfolio"],
        )
        
        return method.attribute(
            inputs=inputs,
            baselines=baseline_tuple,
            target=target,
        )
    
    def _aggregate_attributions(
        self, attributions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate attributions by feature groups."""
        aggregated = {}
        
        for method_name, method_attrs in attributions.items():
            aggregated[method_name] = {}
            
            # Aggregate each branch
            for branch in ["hf", "mf", "lf", "portfolio"]:
                if method_attrs.get(branch) is None:
                    continue
                    
                branch_attrs = method_attrs[branch]
                if branch_attrs.ndim > 2:
                    # Average over sequence dimension
                    branch_attrs = branch_attrs.mean(axis=1)
                
                # Sum absolute attributions for each feature
                feature_importance = np.abs(branch_attrs).sum(axis=0)
                
                # Store top features
                top_indices = np.argsort(feature_importance)[-10:][::-1]
                aggregated[method_name][branch] = {
                    "top_features": [
                        {
                            "index": int(idx),
                            "name": self.feature_names[branch][idx] if idx < len(self.feature_names[branch]) else f"{branch}_feat_{idx}",
                            "importance": float(feature_importance[idx]),
                        }
                        for idx in top_indices
                    ],
                    "total_importance": float(feature_importance.sum()),
                }
        
        return aggregated
    
    def _create_visualizations(self, results: Dict, state_dict: Dict) -> List[str]:
        """Create and save attribution visualizations."""
        viz_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create heatmaps for each method
        for method_name, attributions in results["attributions"].items():
            if not any(v is not None for v in attributions.values()):
                continue
                
            # Branch comparison heatmap
            fig = self._create_branch_heatmap(method_name, attributions)
            if fig:
                path = self.viz_dir / f"{method_name}_branches_{timestamp}.png"
                # Ensure directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                viz_paths.append(str(path))
            
            # Time series attribution plot for HF features
            if attributions.get("hf") is not None:
                fig = self._create_timeseries_plot(method_name, attributions["hf"], "hf")
                if fig:
                    path = self.viz_dir / f"{method_name}_hf_timeseries_{timestamp}.png"
                    # Ensure directory exists
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    viz_paths.append(str(path))
        
        # Create aggregated importance plot
        if results.get("aggregated"):
            fig = self._create_aggregated_importance_plot(results["aggregated"])
            if fig:
                path = self.viz_dir / f"aggregated_importance_{timestamp}.png"
                # Ensure directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                viz_paths.append(str(path))
        
        return viz_paths
    
    def _create_branch_heatmap(
        self, method_name: str, attributions: Dict[str, np.ndarray]
    ) -> Optional[plt.Figure]:
        """Create heatmap comparing attributions across branches."""
        try:
            # Prepare data
            branch_data = []
            branch_labels = []
            
            for branch in ["hf", "mf", "lf", "portfolio"]:
                if attributions.get(branch) is not None:
                    attrs = attributions[branch]
                    if attrs.ndim > 2:
                        # Average over batch and sequence dimensions
                        attrs = attrs.mean(axis=(0, 1))
                    elif attrs.ndim == 2:
                        attrs = attrs.mean(axis=0)
                    
                    # Normalize and take top features
                    attrs = np.abs(attrs)
                    top_indices = np.argsort(attrs)[-20:][::-1]
                    branch_data.append(attrs[top_indices])
                    branch_labels.append(branch.upper())
            
            if not branch_data:
                return None
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Stack data
            data_matrix = np.array(branch_data)
            
            # Create heatmap
            sns.heatmap(
                data_matrix,
                xticklabels=[f"F{i}" for i in range(data_matrix.shape[1])],
                yticklabels=branch_labels,
                cmap="RdBu_r",
                center=0,
                cbar_kws={"label": "Attribution Score"},
                ax=ax,
            )
            
            ax.set_title(f"Feature Attributions - {method_name}")
            ax.set_xlabel("Top Features")
            ax.set_ylabel("Branch")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating branch heatmap: {str(e)}")
            return None
    
    def _create_timeseries_plot(
        self, method_name: str, attributions: np.ndarray, branch: str
    ) -> Optional[plt.Figure]:
        """Create time series plot of attributions."""
        try:
            if attributions.ndim < 3:
                return None
            
            # Get sequence attributions
            seq_attrs = attributions[0]  # First sample
            
            # Select top features
            feature_importance = np.abs(seq_attrs).mean(axis=0)
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for idx in top_features:
                feature_name = (
                    self.feature_names[branch][idx]
                    if idx < len(self.feature_names[branch])
                    else f"{branch}_feat_{idx}"
                )
                ax.plot(seq_attrs[:, idx], label=feature_name, linewidth=2)
            
            ax.set_title(f"Time Series Attributions - {method_name} ({branch.upper()})")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Attribution Score")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating timeseries plot: {str(e)}")
            return None
    
    def _create_aggregated_importance_plot(
        self, aggregated: Dict[str, Dict[str, Any]]
    ) -> Optional[plt.Figure]:
        """Create aggregated feature importance plot."""
        try:
            # Collect importance scores across methods
            importance_data = {}
            
            for method_name, method_data in aggregated.items():
                for branch, branch_data in method_data.items():
                    if "top_features" not in branch_data:
                        continue
                        
                    for feature in branch_data["top_features"][:5]:  # Top 5 per branch
                        feature_key = f"{branch}_{feature['name']}"
                        if feature_key not in importance_data:
                            importance_data[feature_key] = []
                        importance_data[feature_key].append(feature["importance"])
            
            if not importance_data:
                return None
            
            # Average importance across methods
            avg_importance = {
                k: np.mean(v) for k, v in importance_data.items()
            }
            
            # Sort by importance
            sorted_features = sorted(
                avg_importance.items(), key=lambda x: x[1], reverse=True
            )[:20]  # Top 20 overall
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            features, scores = zip(*sorted_features)
            y_pos = np.arange(len(features))
            
            # Color by branch
            colors = []
            for f in features:
                if f.startswith("hf_"):
                    colors.append("#1f77b4")
                elif f.startswith("mf_"):
                    colors.append("#ff7f0e")
                elif f.startswith("lf_"):
                    colors.append("#2ca02c")
                else:
                    colors.append("#d62728")
            
            ax.barh(y_pos, scores, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel("Average Importance Score")
            ax.set_title("Top Feature Importances (Aggregated Across Methods)")
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#1f77b4", label="HF"),
                Patch(facecolor="#ff7f0e", label="MF"),
                Patch(facecolor="#2ca02c", label="LF"),
                Patch(facecolor="#d62728", label="Portfolio"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating aggregated plot: {str(e)}")
            return None
    
    def _format_results(self, results: Dict) -> Dict:
        """Format results for more compact representation."""
        formatted = {
            "timestamp": results["timestamp"],
            "predictions": results.get("predictions", {}),
            "top_attributions": {},
            "branch_importance": {},
            "visualizations": results.get("visualizations", []),
        }
        
        # Extract top attributions per method
        if "aggregated" in results:
            for method_name, method_data in results["aggregated"].items():
                formatted["top_attributions"][method_name] = {}
                formatted["branch_importance"][method_name] = {}
                
                for branch, branch_data in method_data.items():
                    # Top 3 features per branch
                    if "top_features" in branch_data:
                        formatted["top_attributions"][method_name][branch] = (
                            branch_data["top_features"][:3]
                        )
                    
                    # Branch total importance
                    if "total_importance" in branch_data:
                        formatted["branch_importance"][method_name][branch] = (
                            branch_data["total_importance"]
                        )
        
        return formatted
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from attribution history."""
        if not self.attribution_history:
            return {}
        
        # Aggregate statistics across history
        branch_importance_history = {branch: [] for branch in ["hf", "mf", "lf", "portfolio"]}
        feature_frequency = {}
        
        for record in self.attribution_history:
            if "aggregated" not in record:
                continue
                
            for method_data in record["aggregated"].values():
                for branch, branch_data in method_data.items():
                    if "total_importance" in branch_data:
                        branch_importance_history[branch].append(
                            branch_data["total_importance"]
                        )
                    
                    if "top_features" in branch_data:
                        for feature in branch_data["top_features"][:5]:
                            feature_key = f"{branch}_{feature['name']}"
                            if feature_key not in feature_frequency:
                                feature_frequency[feature_key] = 0
                            feature_frequency[feature_key] += 1
        
        # Calculate statistics
        summary = {
            "samples_analyzed": len(self.attribution_history),
            "branch_importance_mean": {
                branch: np.mean(scores) if scores else 0
                for branch, scores in branch_importance_history.items()
            },
            "branch_importance_std": {
                branch: np.std(scores) if scores else 0
                for branch, scores in branch_importance_history.items()
            },
            "most_frequent_features": sorted(
                feature_frequency.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }
        
        return summary
    
    def save_analysis_report(self, filepath: str):
        """Save comprehensive analysis report."""
        report = {
            "config": self.config.__dict__,
            "summary_statistics": self.get_summary_statistics(),
            "attribution_history": [
                self._format_results(record) for record in self.attribution_history[-10:]
            ],  # Last 10 samples
        }
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Saved analysis report to {filepath}")
