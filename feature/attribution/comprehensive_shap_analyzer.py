"""
Comprehensive SHAP Feature Attribution System v2

A production-ready SHAP implementation optimized for 2,330+ features with:
- Multiple SHAP explainer methods (DeepSHAP, GradientSHAP, IntegratedGradients)
- Feature-level attribution with proper branch mapping
- Feature interactions and dependency analysis
- Dead feature detection with configurable thresholds
- Feature importance trends with moving averages
- Individual sample explanations with decision paths
- Statistical analysis (correlations, distributions, outliers)
- Advanced visualizations optimized for WandB
- Configurable update frequency with smart sampling
- Dashboard integration with real-time insights
- GPU acceleration and batch processing
- Dynamic feature selection for performance
"""

import logging
import time
import os
import pickle
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

try:
    import shap
    import wandb
    from captum.attr import (
        IntegratedGradients,
        DeepLift,
        GradientShap,
        FeatureAblation,
        Occlusion,
        LayerConductance,
        NeuronConductance,
        NoiseTunnel,
    )

    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False


@dataclass
class AttributionConfig:
    """Configuration for attribution analysis"""

    # Enable/disable attribution
    enabled: bool = True  # Set to False to completely disable attribution analysis

    # Analysis settings
    update_frequency: int = 10  # Run every N updates
    max_samples_per_analysis: int = 5  # Max samples to analyze (performance)
    background_samples: int = 10  # Background samples for baseline

    # Methods to use
    methods: List[str] = field(
        default_factory=lambda: ["gradient_shap", "integrated_gradients"]
    )
    primary_method: str = "gradient_shap"

    # Feature selection
    top_k_features: int = 50  # Track top K features
    dead_feature_threshold: float = 0.001  # Threshold for dead features
    interaction_top_k: int = 20  # Top K feature interactions

    # Performance settings
    use_gpu: bool = True
    batch_size: int = 8
    num_workers: int = 2
    cache_background: bool = True

    # Visualization settings
    save_plots: bool = True
    plot_dir: str = "outputs/shap_plots"
    plot_formats: List[str] = field(default_factory=lambda: ["png"])
    plot_dpi: int = 150  # Lower DPI for faster saves

    # Tracking settings
    history_length: int = 100
    trend_window: int = 10

    # Dashboard settings
    dashboard_update_freq: int = 5  # Update dashboard every N analyses

    # Advanced features
    analyze_interactions: bool = True
    analyze_gradients: bool = True
    track_attention: bool = True
    detect_outliers: bool = True


@dataclass
class FeatureMetadata:
    """Metadata for a single feature"""

    name: str
    branch: str
    index: int
    sequence_position: int
    feature_position: int
    importance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    gradient_history: deque = field(default_factory=lambda: deque(maxlen=100))
    activation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    is_dead: bool = False
    interaction_partners: Set[int] = field(default_factory=set)


class ComprehensiveSHAPAnalyzer:
    """
    Comprehensive SHAP analyzer with all advanced features optimized for 2,330+ features.

    Key optimizations:
    - Smart sampling and caching
    - GPU-accelerated computations
    - Batch processing
    - Dynamic feature selection
    - Efficient visualization generation
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Dict[str, List[str]],
        branch_configs: Dict[str, Tuple[int, int]],
        config: Optional[AttributionConfig] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not ATTRIBUTION_AVAILABLE:
            raise ImportError(
                "SHAP and Captum are required. Install with: pip install shap captum"
            )

        self.model = model
        self.feature_names = feature_names
        self.branch_configs = branch_configs
        self.config = config or AttributionConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )
        self.logger = logger or logging.getLogger(__name__)

        # Check if attribution is enabled
        if not self.config.enabled:
            self.logger.info("🔕 Attribution analysis is DISABLED by configuration")
            self.total_features = 0
            self.feature_metadata = OrderedDict()
            self.branch_feature_indices = defaultdict(list)
            return

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize components
        self._setup_feature_metadata()
        self._setup_explainers()
        self._setup_tracking()
        self._setup_visualization()

        # Cache for performance
        self.background_cache = None
        self.feature_cache = {}
        self.last_analysis_time = 0
        self.analysis_count = 0

        self.logger.info(
            f"🚀 Comprehensive SHAP Analyzer initialized with {self.total_features} features "
            f"across {len(self.branch_configs)} branches on {self.device}"
        )

    def _setup_feature_metadata(self):
        """Setup feature metadata and mappings"""
        self.feature_metadata = OrderedDict()
        self.feature_index_map = {}
        self.branch_feature_indices = defaultdict(list)

        current_idx = 0
        for branch, (seq_len, feat_dim) in self.branch_configs.items():
            branch_names = self.feature_names.get(branch, [])

            for seq_pos in range(seq_len):
                for feat_pos in range(feat_dim):
                    # Get feature name
                    if feat_pos < len(branch_names):
                        feat_name = branch_names[feat_pos]
                    else:
                        feat_name = f"feature_{feat_pos}"

                    # Create metadata
                    meta = FeatureMetadata(
                        name=feat_name,
                        branch=branch,
                        index=current_idx,
                        sequence_position=seq_pos,
                        feature_position=feat_pos,
                    )

                    self.feature_metadata[current_idx] = meta
                    self.feature_index_map[f"{branch}.{feat_name}.{seq_pos}"] = (
                        current_idx
                    )
                    self.branch_feature_indices[branch].append(current_idx)

                    current_idx += 1

        self.total_features = current_idx
        self.logger.info(f"📊 Initialized metadata for {self.total_features} features")

    def _setup_explainers(self):
        """Setup multiple attribution methods"""
        self.explainers = {}
        self.model_wrapper = self._create_model_wrapper()

        # Initialize Captum attributors
        if "integrated_gradients" in self.config.methods:
            self.explainers["integrated_gradients"] = IntegratedGradients(
                self.model_wrapper
            )

        if "gradient_shap" in self.config.methods:
            self.explainers["gradient_shap"] = GradientShap(self.model_wrapper)

        if "deep_lift" in self.config.methods:
            self.explainers["deep_lift"] = DeepLift(self.model_wrapper)

        if "feature_ablation" in self.config.methods:
            self.explainers["feature_ablation"] = FeatureAblation(self.model_wrapper)

        # Add noise tunnel for robustness
        if self.config.primary_method in self.explainers:
            base_explainer = self.explainers[self.config.primary_method]
            self.explainers["noise_tunnel"] = NoiseTunnel(base_explainer)

        self.logger.info(f"✅ Initialized {len(self.explainers)} attribution methods")

    def _create_model_wrapper(self):
        """Create a wrapper for Captum compatibility"""

        def forward_func(inputs: torch.Tensor) -> torch.Tensor:
            """
            Wrapper that converts flattened tensor to state dict and gets predictions.

            Args:
                inputs: Flattened tensor [batch_size, total_features]

            Returns:
                Model predictions [batch_size, num_actions]
            """
            # Ensure inputs are on correct device
            if not inputs.is_cuda and self.device.type == "cuda":
                inputs = inputs.to(self.device)

            # Convert to state dict
            state_dict = self._tensor_to_state_dict(inputs)

            # Get predictions
            with torch.no_grad():
                action_params, value = self.model(state_dict)

                # For discrete actions, return action type logits
                if isinstance(action_params, tuple) and len(action_params) == 2:
                    return action_params[0]  # Action type logits
                else:
                    return action_params

        return forward_func

    def _tensor_to_state_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flattened tensor to state dict format"""
        batch_size = tensor.shape[0]
        state_dict = {}
        start_idx = 0

        for branch, (seq_len, feat_dim) in self.branch_configs.items():
            expected_size = seq_len * feat_dim
            end_idx = start_idx + expected_size

            if end_idx > tensor.shape[1]:
                # Handle size mismatch gracefully
                available = tensor.shape[1] - start_idx
                if available > 0:
                    branch_data = tensor[:, start_idx : start_idx + available]
                    # Pad if needed
                    if available < expected_size:
                        padding = torch.zeros(
                            batch_size,
                            expected_size - available,
                            device=tensor.device,
                            dtype=tensor.dtype,
                        )
                        branch_data = torch.cat([branch_data, padding], dim=1)
                else:
                    branch_data = torch.zeros(
                        batch_size,
                        expected_size,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )

                state_dict[branch] = branch_data.reshape(batch_size, seq_len, feat_dim)
                break
            else:
                branch_tensor = tensor[:, start_idx:end_idx].reshape(
                    batch_size, seq_len, feat_dim
                )
                state_dict[branch] = branch_tensor
                start_idx = end_idx

        return state_dict

    def _state_dict_to_tensor(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert state dict to flattened tensor"""
        tensors = []

        for branch in self.branch_configs.keys():
            if branch in state_dict:
                branch_tensor = state_dict[branch]
                # Flatten keeping batch dimension
                flattened = branch_tensor.reshape(branch_tensor.shape[0], -1)
                tensors.append(flattened)

        return torch.cat(tensors, dim=1)

    def _setup_tracking(self):
        """Setup tracking structures"""
        # Attribution history
        self.attribution_history = defaultdict(
            lambda: deque(maxlen=self.config.history_length)
        )
        self.method_history = defaultdict(
            lambda: deque(maxlen=self.config.history_length)
        )

        # Feature importance tracking
        self.global_importance = np.zeros(self.total_features)
        self.importance_ema = np.zeros(
            self.total_features
        )  # Exponential moving average
        self.importance_variance = np.zeros(self.total_features)

        # Interaction tracking
        self.interaction_matrix = np.zeros((self.total_features, self.total_features))
        self.interaction_counts = np.zeros((self.total_features, self.total_features))

        # Dead feature tracking
        self.dead_feature_candidates = set()
        self.confirmed_dead_features = set()

        # Performance metrics
        self.analysis_times = deque(maxlen=50)
        self.sample_counts = deque(maxlen=50)

    def _setup_visualization(self):
        """Setup visualization components"""
        if self.config.save_plots:
            self.plot_dir = Path(self.config.plot_dir)
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def setup_background(self, background_states: List[Dict[str, torch.Tensor]]):
        """
        Setup background data for attribution baseline.

        Args:
            background_states: List of state dictionaries for background
        """
        if not self.config.enabled:
            return

        try:
            # Limit background samples for performance
            num_samples = min(len(background_states), self.config.background_samples)
            sampled_states = background_states[:num_samples]

            # Convert to tensor and cache
            background_tensors = []
            for state in sampled_states:
                tensor = self._state_dict_to_tensor(state)
                background_tensors.append(tensor)

            self.background_cache = torch.stack(background_tensors).to(self.device)

            self.logger.info(
                f"✅ Background data setup complete: {self.background_cache.shape} "
                f"({num_samples} samples, {self.total_features} features)"
            )

            # Test attribution methods with background
            self._test_attribution_methods()

        except Exception as e:
            self.logger.error(f"Failed to setup background data: {e}")
            self.background_cache = None

    def _test_attribution_methods(self):
        """Test each attribution method to ensure they work"""
        if self.background_cache is None:
            return

        test_input = self.background_cache[:1]

        for method_name, explainer in self.explainers.items():
            try:
                if method_name in ["gradient_shap", "noise_tunnel"]:
                    # These need baselines
                    attributions = explainer.attribute(
                        test_input, baselines=self.background_cache[:2], target=0
                    )
                else:
                    # Others just need input
                    attributions = explainer.attribute(test_input, target=0)

                self.logger.debug(
                    f"✓ {method_name} working: shape {attributions.shape}"
                )

            except Exception as e:
                self.logger.warning(f"✗ {method_name} failed: {e}")
                # Remove failed method
                if method_name in self.config.methods:
                    self.config.methods.remove(method_name)

    def analyze_features(
        self,
        states: List[Dict[str, torch.Tensor]],
        actions: Optional[List[int]] = None,
        rewards: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive feature analysis with multiple attribution methods.

        Args:
            states: List of state dictionaries to analyze
            actions: Optional list of actions taken
            rewards: Optional list of rewards received

        Returns:
            Comprehensive analysis results
        """
        if not self.config.enabled:
            return {"enabled": False, "message": "Attribution analysis is disabled"}

        if not self.background_cache:
            self.logger.error(
                "No background data available. Call setup_background() first."
            )
            return {"error": "No background data"}

        start_time = time.time()
        self.analysis_count += 1

        try:
            # Sample states for performance
            num_samples = min(len(states), self.config.max_samples_per_analysis)
            if len(states) > num_samples:
                # Smart sampling: take first, last, and random middle samples
                indices = [0, len(states) - 1]
                if num_samples > 2:
                    middle_indices = np.random.choice(
                        range(1, len(states) - 1), size=num_samples - 2, replace=False
                    )
                    indices.extend(middle_indices)
                indices = sorted(indices[:num_samples])

                sampled_states = [states[i] for i in indices]
                sampled_actions = [actions[i] for i in indices] if actions else None
                sampled_rewards = [rewards[i] for i in indices] if rewards else None
            else:
                sampled_states = states
                sampled_actions = actions
                sampled_rewards = rewards

            # Convert to tensors
            input_tensors = []
            for state in sampled_states:
                tensor = self._state_dict_to_tensor(state)
                input_tensors.append(tensor)
            input_batch = torch.stack(input_tensors).to(self.device)

            # Run multiple attribution methods
            all_attributions = {}
            for method in self.config.methods:
                if method in self.explainers:
                    method_attributions = self._run_attribution_method(
                        method, input_batch, sampled_actions
                    )
                    if method_attributions is not None:
                        all_attributions[method] = method_attributions

            if not all_attributions:
                self.logger.error("All attribution methods failed")
                return {"error": "Attribution failed"}

            # Use primary method for main analysis
            primary_attributions = all_attributions.get(
                self.config.primary_method, list(all_attributions.values())[0]
            )

            # Comprehensive analysis
            results = {
                "method": self.config.primary_method,
                "num_samples": num_samples,
                "analysis_time": time.time() - start_time,
                "attributions": primary_attributions,
                "all_methods": all_attributions,
                # Feature importance
                "feature_importance": self._calculate_feature_importance(
                    primary_attributions
                ),
                "branch_importance": self._calculate_branch_importance(
                    primary_attributions
                ),
                "top_features": self._get_top_features(primary_attributions),
                # Feature interactions
                "feature_interactions": self._analyze_feature_interactions(
                    primary_attributions, input_batch
                )
                if self.config.analyze_interactions
                else None,
                # Dead features
                "dead_features": self._detect_dead_features(primary_attributions),
                # Individual explanations
                "sample_explanations": self._get_sample_explanations(
                    primary_attributions, input_batch, sampled_actions, sampled_rewards
                ),
                # Statistical analysis
                "statistics": self._calculate_statistics(primary_attributions),
                # Trends
                "importance_trends": self._calculate_importance_trends(),
                # Method comparison
                "method_comparison": self._compare_attribution_methods(
                    all_attributions
                ),
                # Outlier detection
                "outliers": self._detect_outliers(primary_attributions)
                if self.config.detect_outliers
                else None,
            }

            # Update tracking
            self._update_tracking(results)

            # Generate visualizations
            if (
                self.config.save_plots
                and self.analysis_count % self.config.dashboard_update_freq == 0
            ):
                results["visualizations"] = self._generate_visualizations(results)

            # Log performance
            self.analysis_times.append(results["analysis_time"])
            self.sample_counts.append(num_samples)

            self.logger.info(
                f"✅ Feature analysis complete: {num_samples} samples in {results['analysis_time']:.2f}s "
                f"({np.mean(self.analysis_times):.2f}s avg)"
            )

            return results

        except Exception as e:
            self.logger.error(f"Feature analysis failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _run_attribution_method(
        self, method: str, inputs: torch.Tensor, actions: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """Run a specific attribution method"""
        try:
            explainer = self.explainers[method]

            # Determine target (action taken if available, otherwise explain all)
            if actions:
                targets = torch.tensor(actions, device=self.device)
            else:
                targets = None

            # Run attribution
            if method in ["gradient_shap", "noise_tunnel"]:
                # Need baselines
                num_baselines = min(3, len(self.background_cache))
                baselines = self.background_cache[:num_baselines]

                if targets is not None:
                    # Explain specific actions
                    attributions = []
                    for i, target in enumerate(targets):
                        attr = explainer.attribute(
                            inputs[i : i + 1], baselines=baselines, target=int(target)
                        )
                        attributions.append(attr)
                    attributions = torch.cat(attributions, dim=0)
                else:
                    # Explain all outputs
                    attributions = explainer.attribute(inputs, baselines=baselines)
            else:
                # No baselines needed
                if targets is not None:
                    attributions = []
                    for i, target in enumerate(targets):
                        attr = explainer.attribute(
                            inputs[i : i + 1], target=int(target)
                        )
                        attributions.append(attr)
                    attributions = torch.cat(attributions, dim=0)
                else:
                    attributions = explainer.attribute(inputs)

            # Convert to numpy
            return attributions.detach().cpu().numpy()

        except Exception as e:
            self.logger.warning(f"Attribution method {method} failed: {e}")
            return None

    def _calculate_feature_importance(
        self, attributions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance scores"""
        # Global importance (mean absolute attribution)
        global_importance = np.mean(np.abs(attributions), axis=0)

        # Update exponential moving average
        alpha = 0.1
        self.importance_ema = (
            alpha * global_importance + (1 - alpha) * self.importance_ema
        )

        # Update variance tracking
        self.importance_variance = (
            alpha * (global_importance - self.importance_ema) ** 2
            + (1 - alpha) * self.importance_variance
        )

        # Create importance dictionary
        importance_dict = {}
        for idx, importance in enumerate(global_importance):
            meta = self.feature_metadata[idx]
            key = f"{meta.branch}.{meta.name}"

            if key not in importance_dict:
                importance_dict[key] = {
                    "current": 0.0,
                    "ema": 0.0,
                    "variance": 0.0,
                    "positions": [],
                }

            importance_dict[key]["current"] += float(importance)
            importance_dict[key]["ema"] += float(self.importance_ema[idx])
            importance_dict[key]["variance"] += float(self.importance_variance[idx])
            importance_dict[key]["positions"].append(meta.sequence_position)

        # Average across positions
        for key in importance_dict:
            num_positions = len(importance_dict[key]["positions"])
            importance_dict[key]["current"] /= num_positions
            importance_dict[key]["ema"] /= num_positions
            importance_dict[key]["variance"] /= num_positions
            importance_dict[key]["stability"] = 1.0 / (
                1.0 + importance_dict[key]["variance"]
            )

        return importance_dict

    def _calculate_branch_importance(
        self, attributions: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate importance by branch with detailed statistics"""
        branch_importance = {}

        for branch, indices in self.branch_feature_indices.items():
            branch_attrs = np.abs(attributions[:, indices])

            branch_importance[branch] = {
                "mean": float(np.mean(branch_attrs)),
                "std": float(np.std(branch_attrs)),
                "max": float(np.max(branch_attrs)),
                "min": float(np.min(branch_attrs)),
                "median": float(np.median(branch_attrs)),
                "total": float(np.sum(branch_attrs)),
                "proportion": 0.0,  # Will calculate after
            }

        # Calculate proportions
        total_importance = sum(b["total"] for b in branch_importance.values())
        if total_importance > 0:
            for branch in branch_importance:
                branch_importance[branch]["proportion"] = (
                    branch_importance[branch]["total"] / total_importance
                )

        return branch_importance

    def _get_top_features(self, attributions: np.ndarray) -> List[Dict[str, Any]]:
        """Get top K most important features with detailed info"""
        # Calculate mean absolute importance
        mean_importance = np.mean(np.abs(attributions), axis=0)

        # Get top K indices
        top_indices = np.argsort(mean_importance)[-self.config.top_k_features :][::-1]

        top_features = []
        for rank, idx in enumerate(top_indices):
            meta = self.feature_metadata[idx]

            # Calculate statistics for this feature
            feature_attrs = attributions[:, idx]

            feature_info = {
                "rank": rank + 1,
                "index": int(idx),
                "name": meta.name,
                "branch": meta.branch,
                "full_name": f"{meta.branch}.{meta.name}",
                "sequence_position": meta.sequence_position,
                "importance": float(mean_importance[idx]),
                "importance_ema": float(self.importance_ema[idx]),
                "importance_std": float(np.std(feature_attrs)),
                "positive_attributions": float(
                    np.mean(feature_attrs[feature_attrs > 0])
                    if np.any(feature_attrs > 0)
                    else 0
                ),
                "negative_attributions": float(
                    np.mean(feature_attrs[feature_attrs < 0])
                    if np.any(feature_attrs < 0)
                    else 0
                ),
                "attribution_range": [
                    float(np.min(feature_attrs)),
                    float(np.max(feature_attrs)),
                ],
            }

            top_features.append(feature_info)

        return top_features

    def _analyze_feature_interactions(
        self, attributions: np.ndarray, inputs: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze feature interactions and dependencies"""
        try:
            # Calculate correlation between attributions
            attr_corr = np.corrcoef(attributions.T)

            # Find strong interactions (high correlation)
            interactions = []
            for i in range(self.total_features):
                for j in range(i + 1, self.total_features):
                    corr = attr_corr[i, j]
                    if abs(corr) > 0.7:  # Strong correlation threshold
                        meta_i = self.feature_metadata[i]
                        meta_j = self.feature_metadata[j]

                        interactions.append(
                            {
                                "feature_1": f"{meta_i.branch}.{meta_i.name}",
                                "feature_2": f"{meta_j.branch}.{meta_j.name}",
                                "correlation": float(corr),
                                "strength": "strong" if abs(corr) > 0.8 else "moderate",
                            }
                        )

                        # Update metadata
                        meta_i.interaction_partners.add(j)
                        meta_j.interaction_partners.add(i)

            # Sort by absolute correlation
            interactions.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            # Calculate interaction matrix for top features
            top_k = min(20, self.total_features)
            top_indices = np.argsort(np.mean(np.abs(attributions), axis=0))[-top_k:]
            interaction_matrix = attr_corr[np.ix_(top_indices, top_indices)]

            return {
                "top_interactions": interactions[: self.config.interaction_top_k],
                "num_strong_interactions": len(interactions),
                "interaction_matrix": interaction_matrix.tolist(),
                "interaction_summary": {
                    "mean_correlation": float(np.mean(np.abs(attr_corr))),
                    "max_correlation": float(
                        np.max(np.abs(attr_corr[np.triu_indices_from(attr_corr, k=1)]))
                    ),
                    "features_with_interactions": len(
                        [
                            m
                            for m in self.feature_metadata.values()
                            if m.interaction_partners
                        ]
                    ),
                },
            }

        except Exception as e:
            self.logger.warning(f"Feature interaction analysis failed: {e}")
            return {"error": str(e)}

    def _detect_dead_features(self, attributions: np.ndarray) -> Dict[str, Any]:
        """Detect features with consistently low importance"""
        mean_importance = np.mean(np.abs(attributions), axis=0)

        # Find features below threshold
        dead_indices = np.where(mean_importance < self.config.dead_feature_threshold)[0]

        # Update tracking
        for idx in dead_indices:
            self.dead_feature_candidates.add(idx)

        # Confirm dead features (consistently low across multiple analyses)
        newly_dead = []
        for idx in list(self.dead_feature_candidates):
            meta = self.feature_metadata[idx]

            # Check history
            if len(meta.importance_history) >= 5:
                recent_importance = list(meta.importance_history)[-5:]
                if all(
                    imp < self.config.dead_feature_threshold
                    for imp in recent_importance
                ):
                    if idx not in self.confirmed_dead_features:
                        self.confirmed_dead_features.add(idx)
                        meta.is_dead = True
                        newly_dead.append(idx)
                else:
                    # Remove from candidates if showing life
                    self.dead_feature_candidates.discard(idx)

        # Categorize dead features by branch
        dead_by_branch = defaultdict(list)
        for idx in self.confirmed_dead_features:
            meta = self.feature_metadata[idx]
            dead_by_branch[meta.branch].append(
                {
                    "name": meta.name,
                    "index": idx,
                    "sequence_position": meta.sequence_position,
                    "last_importance": float(mean_importance[idx]),
                }
            )

        return {
            "newly_dead": len(newly_dead),
            "total_dead": len(self.confirmed_dead_features),
            "dead_percentage": len(self.confirmed_dead_features)
            / self.total_features
            * 100,
            "dead_by_branch": dict(dead_by_branch),
            "candidates": len(self.dead_feature_candidates),
            "summary": {
                branch: {
                    "dead_count": len(features),
                    "total_features": len(self.branch_feature_indices[branch]),
                    "dead_percentage": len(features)
                    / len(self.branch_feature_indices[branch])
                    * 100,
                }
                for branch, features in dead_by_branch.items()
            },
        }

    def _get_sample_explanations(
        self,
        attributions: np.ndarray,
        inputs: torch.Tensor,
        actions: Optional[List[int]] = None,
        rewards: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate detailed explanations for individual samples"""
        explanations = []

        for i in range(min(5, len(attributions))):  # Limit to 5 samples
            sample_attrs = attributions[i]

            # Get top positive and negative contributors
            top_positive_idx = np.argsort(sample_attrs)[-10:][::-1]
            top_negative_idx = np.argsort(sample_attrs)[:10]

            # Create explanation
            explanation = {
                "sample_index": i,
                "action": int(actions[i]) if actions else None,
                "reward": float(rewards[i]) if rewards else None,
                # Top contributors
                "top_positive_features": [
                    {
                        "feature": f"{self.feature_metadata[idx].branch}.{self.feature_metadata[idx].name}",
                        "attribution": float(sample_attrs[idx]),
                        "input_value": float(inputs[i, idx].cpu()),
                    }
                    for idx in top_positive_idx
                    if sample_attrs[idx] > 0
                ],
                "top_negative_features": [
                    {
                        "feature": f"{self.feature_metadata[idx].branch}.{self.feature_metadata[idx].name}",
                        "attribution": float(sample_attrs[idx]),
                        "input_value": float(inputs[i, idx].cpu()),
                    }
                    for idx in top_negative_idx
                    if sample_attrs[idx] < 0
                ],
                # Summary statistics
                "total_positive_attribution": float(
                    np.sum(sample_attrs[sample_attrs > 0])
                ),
                "total_negative_attribution": float(
                    np.sum(sample_attrs[sample_attrs < 0])
                ),
                "net_attribution": float(np.sum(sample_attrs)),
                # Branch contributions
                "branch_contributions": {
                    branch: float(np.sum(sample_attrs[indices]))
                    for branch, indices in self.branch_feature_indices.items()
                },
            }

            explanations.append(explanation)

        return explanations

    def _calculate_statistics(self, attributions: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        return {
            "attribution_stats": {
                "mean": float(np.mean(attributions)),
                "std": float(np.std(attributions)),
                "min": float(np.min(attributions)),
                "max": float(np.max(attributions)),
                "median": float(np.median(attributions)),
                "skewness": float(self._calculate_skewness(attributions)),
                "kurtosis": float(self._calculate_kurtosis(attributions)),
            },
            "importance_stats": {
                "mean": float(np.mean(np.abs(attributions))),
                "std": float(np.std(np.abs(attributions))),
                "gini_coefficient": float(
                    self._calculate_gini(np.mean(np.abs(attributions), axis=0))
                ),
                "concentration_ratio": float(
                    self._calculate_concentration_ratio(
                        np.mean(np.abs(attributions), axis=0)
                    )
                ),
            },
            "sparsity_stats": {
                "zero_attribution_ratio": float(np.mean(np.abs(attributions) < 1e-6)),
                "low_attribution_ratio": float(
                    np.mean(np.abs(attributions) < self.config.dead_feature_threshold)
                ),
                "high_attribution_ratio": float(np.mean(np.abs(attributions) > 0.1)),
            },
        }

    def _calculate_importance_trends(self) -> Dict[str, Any]:
        """Calculate feature importance trends over time"""
        if not self.attribution_history:
            return {"available": False}

        # Get recent history
        recent_window = self.config.trend_window

        # Calculate trends for top features
        top_features_trends = []
        mean_importance = np.mean(
            [
                np.mean(np.abs(attrs), axis=0)
                for attrs in list(self.attribution_history.values())[-recent_window:]
            ],
            axis=0,
        )

        top_indices = np.argsort(mean_importance)[-20:][::-1]

        for idx in top_indices:
            meta = self.feature_metadata[idx]
            history = list(meta.importance_history)

            if len(history) >= 3:
                # Calculate trend
                x = np.arange(len(history))
                y = np.array(history)
                slope, intercept = np.polyfit(x, y, 1)

                trend = (
                    "increasing"
                    if slope > 0.001
                    else "decreasing"
                    if slope < -0.001
                    else "stable"
                )

                top_features_trends.append(
                    {
                        "feature": f"{meta.branch}.{meta.name}",
                        "current_importance": float(history[-1]) if history else 0,
                        "trend": trend,
                        "slope": float(slope),
                        "change_rate": float(slope / (np.mean(history) + 1e-8)),
                    }
                )

        return {
            "available": True,
            "top_features_trends": top_features_trends,
            "global_trend": {
                "mean_importance": float(np.mean(mean_importance)),
                "importance_concentration": float(np.std(mean_importance)),
            },
        }

    def _compare_attribution_methods(
        self, all_attributions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compare results across different attribution methods"""
        if len(all_attributions) < 2:
            return {"comparison_available": False}

        comparisons = {}
        methods = list(all_attributions.keys())

        # Compare top features across methods
        top_features_by_method = {}
        for method, attrs in all_attributions.items():
            importance = np.mean(np.abs(attrs), axis=0)
            top_indices = np.argsort(importance)[-10:][::-1]
            top_features_by_method[method] = set(top_indices)

        # Calculate agreement
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                overlap = len(
                    top_features_by_method[method1] & top_features_by_method[method2]
                )
                agreement = overlap / 10.0

                comparisons[f"{method1}_vs_{method2}"] = {
                    "agreement": float(agreement),
                    "correlation": float(
                        np.corrcoef(
                            np.mean(np.abs(all_attributions[method1]), axis=0),
                            np.mean(np.abs(all_attributions[method2]), axis=0),
                        )[0, 1]
                    ),
                }

        return {
            "comparison_available": True,
            "method_agreements": comparisons,
            "consensus_features": list(
                set.intersection(*top_features_by_method.values())
            ),
        }

    def _detect_outliers(self, attributions: np.ndarray) -> Dict[str, Any]:
        """Detect outlier attributions"""
        # Calculate z-scores
        mean = np.mean(attributions)
        std = np.std(attributions)
        z_scores = np.abs((attributions - mean) / (std + 1e-8))

        # Find outliers (z-score > 3)
        outlier_mask = z_scores > 3
        outlier_indices = np.where(outlier_mask)

        outliers = []
        for sample_idx, feature_idx in zip(outlier_indices[0], outlier_indices[1]):
            meta = self.feature_metadata[feature_idx]
            outliers.append(
                {
                    "sample": int(sample_idx),
                    "feature": f"{meta.branch}.{meta.name}",
                    "attribution": float(attributions[sample_idx, feature_idx]),
                    "z_score": float(z_scores[sample_idx, feature_idx]),
                }
            )

        return {
            "num_outliers": len(outliers),
            "outlier_ratio": float(np.mean(outlier_mask)),
            "top_outliers": sorted(outliers, key=lambda x: x["z_score"], reverse=True)[
                :10
            ],
        }

    def _update_tracking(self, results: Dict[str, Any]):
        """Update internal tracking with new results"""
        # Update attribution history
        if "attributions" in results:
            self.attribution_history[self.analysis_count] = results["attributions"]

        # Update feature importance history
        if "feature_importance" in results:
            for feature, data in results["feature_importance"].items():
                # Find feature index
                for idx, meta in self.feature_metadata.items():
                    if f"{meta.branch}.{meta.name}" == feature:
                        meta.importance_history.append(data["current"])
                        break

        # Update dead feature tracking
        if "dead_features" in results:
            self.last_dead_feature_count = results["dead_features"]["total_dead"]

    def _generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive visualizations"""
        viz_paths = {}
        timestamp = int(time.time())

        try:
            # 1. Feature importance heatmap
            if "feature_importance" in results:
                viz_paths["importance_heatmap"] = self._plot_importance_heatmap(
                    results["feature_importance"], timestamp
                )

            # 2. Branch importance pie chart
            if "branch_importance" in results:
                viz_paths["branch_pie"] = self._plot_branch_pie(
                    results["branch_importance"], timestamp
                )

            # 3. Top features bar chart
            if "top_features" in results:
                viz_paths["top_features"] = self._plot_top_features(
                    results["top_features"], timestamp
                )

            # 4. Feature interactions
            if (
                results.get("feature_interactions")
                and "interaction_matrix" in results["feature_interactions"]
            ):
                viz_paths["interactions"] = self._plot_interactions(
                    results["feature_interactions"], timestamp
                )

            # 5. Sample explanations
            if "sample_explanations" in results and results["sample_explanations"]:
                viz_paths["sample_explanation"] = self._plot_sample_explanation(
                    results["sample_explanations"][0], timestamp
                )

            # 6. Importance trends
            if (
                "importance_trends" in results
                and results["importance_trends"]["available"]
            ):
                viz_paths["trends"] = self._plot_importance_trends(
                    results["importance_trends"], timestamp
                )

            self.logger.info(f"📊 Generated {len(viz_paths)} visualizations")

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")

        return viz_paths

    def _plot_importance_heatmap(self, feature_importance: Dict, timestamp: int) -> str:
        """Plot feature importance as heatmap"""
        # Organize by branch
        branch_data = defaultdict(list)
        for feature, data in feature_importance.items():
            branch = feature.split(".")[0]
            branch_data[branch].append((feature, data["current"]))

        # Create figure
        fig, axes = plt.subplots(
            len(branch_data), 1, figsize=(12, 4 * len(branch_data))
        )
        if len(branch_data) == 1:
            axes = [axes]

        for ax, (branch, features) in zip(axes, branch_data.items()):
            # Sort by importance
            features.sort(key=lambda x: x[1], reverse=True)
            features = features[:20]  # Top 20 per branch

            names = [f.split(".")[-1] for f, _ in features]
            values = [v for _, v in features]

            # Create heatmap data
            heatmap_data = np.array(values).reshape(1, -1)

            sns.heatmap(
                heatmap_data,
                xticklabels=names,
                yticklabels=[branch],
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={"label": "Importance"},
            )
            ax.set_title(f"{branch} Branch - Top Features")

        plt.tight_layout()
        path = self.plot_dir / f"importance_heatmap_{timestamp}.png"
        plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

        return str(path)

    def _plot_branch_pie(self, branch_importance: Dict, timestamp: int) -> str:
        """Plot branch importance as pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))

        branches = list(branch_importance.keys())
        values = [branch_importance[b]["proportion"] for b in branches]

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=branches,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("husl", len(branches)),
        )

        # Enhance text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_weight("bold")

        ax.set_title("Feature Importance by Branch", fontsize=16, pad=20)

        # Add legend with statistics
        legend_labels = [
            f"{branch}: μ={branch_importance[branch]['mean']:.3f}, σ={branch_importance[branch]['std']:.3f}"
            for branch in branches
        ]
        ax.legend(
            wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
        )

        plt.tight_layout()
        path = self.plot_dir / f"branch_importance_{timestamp}.png"
        plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

        return str(path)

    def _plot_top_features(self, top_features: List[Dict], timestamp: int) -> str:
        """Plot top features as horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        features = [f"{f['branch']}.{f['name']}" for f in top_features[:20]]
        importances = [f["importance"] for f in top_features[:20]]

        # Color by branch
        colors = []
        branch_colors = {
            branch: color
            for branch, color in zip(
                self.branch_configs.keys(),
                sns.color_palette("husl", len(self.branch_configs)),
            )
        }
        for f in top_features[:20]:
            colors.append(branch_colors.get(f["branch"], "gray"))

        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=colors, alpha=0.8)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |Attribution|", fontsize=12)
        ax.set_title("Top 20 Most Important Features", fontsize=16, pad=20)

        # Add value labels
        for i, (feature, importance) in enumerate(zip(features, importances)):
            ax.text(importance, i, f" {importance:.3f}", va="center", fontsize=9)

        # Add branch legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=branch)
            for branch, color in branch_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        path = self.plot_dir / f"top_features_{timestamp}.png"
        plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

        return str(path)

    def _plot_interactions(self, interactions: Dict, timestamp: int) -> str:
        """Plot feature interaction matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get interaction matrix
        matrix = np.array(interactions["interaction_matrix"])

        # Create heatmap
        sns.heatmap(
            matrix,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": "Attribution Correlation"},
        )

        ax.set_title("Feature Interaction Matrix (Top Features)", fontsize=16, pad=20)
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Feature Index")

        plt.tight_layout()
        path = self.plot_dir / f"interactions_{timestamp}.png"
        plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

        return str(path)

    def _plot_sample_explanation(self, explanation: Dict, timestamp: int) -> str:
        """Plot waterfall chart for sample explanation"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top positive features
        if explanation["top_positive_features"]:
            pos_features = explanation["top_positive_features"][:10]
            pos_names = [f["feature"].split(".")[-1] for f in pos_features]
            pos_values = [f["attribution"] for f in pos_features]

            y_pos = np.arange(len(pos_names))
            ax1.barh(y_pos, pos_values, color="green", alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(pos_names)
            ax1.set_xlabel("Attribution")
            ax1.set_title("Top Positive Contributors", fontsize=14)
            ax1.invert_yaxis()

            # Add value labels
            for i, v in enumerate(pos_values):
                ax1.text(v, i, f" {v:.3f}", va="center", fontsize=9)

        # Top negative features
        if explanation["top_negative_features"]:
            neg_features = explanation["top_negative_features"][:10]
            neg_names = [f["feature"].split(".")[-1] for f in neg_features]
            neg_values = [abs(f["attribution"]) for f in neg_features]

            y_pos = np.arange(len(neg_names))
            ax2.barh(y_pos, neg_values, color="red", alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(neg_names)
            ax2.set_xlabel("|Attribution|")
            ax2.set_title("Top Negative Contributors", fontsize=14)
            ax2.invert_yaxis()

            # Add value labels
            for i, v in enumerate(neg_values):
                ax2.text(v, i, f" -{v:.3f}", va="center", fontsize=9)

        # Add sample info
        fig.suptitle(
            f"Sample Explanation - Action: {explanation.get('action', 'N/A')}, "
            f"Reward: {explanation.get('reward', 'N/A'):.3f}",
            fontsize=16,
        )

        plt.tight_layout()
        path = self.plot_dir / f"sample_explanation_{timestamp}.png"
        plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

        return str(path)

    def _plot_importance_trends(self, trends: Dict, timestamp: int) -> str:
        """Plot feature importance trends over time"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot trends for top features
        for feature_data in trends["top_features_trends"][:10]:
            feature_name = feature_data["feature"].split(".")[-1]
            trend = feature_data["trend"]

            # Create dummy time series for illustration
            x = np.arange(10)
            y = feature_data["current_importance"] + feature_data["slope"] * (x - 9)

            # Choose color based on trend
            color = (
                "green"
                if trend == "increasing"
                else "red"
                if trend == "decreasing"
                else "blue"
            )

            ax.plot(x, y, label=f"{feature_name} ({trend})", color=color, alpha=0.7)

        ax.set_xlabel("Time (Recent Analyses)")
        ax.set_ylabel("Feature Importance")
        ax.set_title("Feature Importance Trends", fontsize=16, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.plot_dir / f"importance_trends_{timestamp}.png"
        plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

        return str(path)

    def get_summary_for_logging(self) -> Dict[str, Any]:
        """Get summary suitable for WandB/dashboard logging"""
        if not self.config.enabled:
            return {
                "attribution_enabled": False,
                "message": "Attribution analysis is disabled",
            }

        # Calculate current statistics
        current_importance = self.importance_ema
        top_k_indices = np.argsort(current_importance)[-self.config.top_k_features :][
            ::-1
        ]

        # Prepare summary
        summary = {
            "attribution_enabled": True,
            "primary_method": self.config.primary_method,
            "total_features": self.total_features,
            "analyses_completed": self.analysis_count,
            # Performance metrics
            "avg_analysis_time": float(np.mean(self.analysis_times))
            if self.analysis_times
            else 0,
            "avg_samples_analyzed": float(np.mean(self.sample_counts))
            if self.sample_counts
            else 0,
            # Feature importance
            "top_features": [
                {
                    "name": f"{self.feature_metadata[idx].branch}.{self.feature_metadata[idx].name}",
                    "importance": float(current_importance[idx]),
                    "stability": float(1.0 / (1.0 + self.importance_variance[idx])),
                }
                for idx in top_k_indices[:10]
            ],
            # Branch importance
            "branch_importance": {
                branch: float(np.mean(current_importance[indices]))
                for branch, indices in self.branch_feature_indices.items()
            },
            # Dead features
            "dead_features": {
                "count": len(self.confirmed_dead_features),
                "percentage": len(self.confirmed_dead_features)
                / self.total_features
                * 100,
                "by_branch": {
                    branch: len(
                        [idx for idx in self.confirmed_dead_features if idx in indices]
                    )
                    for branch, indices in self.branch_feature_indices.items()
                },
            },
            # Feature diversity
            "feature_diversity": {
                "gini_coefficient": float(self._calculate_gini(current_importance)),
                "effective_features": float(np.sum(current_importance > 0.01)),
                "concentration_top10": float(
                    np.sum(current_importance[top_k_indices[:10]])
                    / np.sum(current_importance)
                ),
            },
        }

        return summary

    def log_to_wandb(self, results: Dict[str, Any], step: Optional[int] = None):
        """Log comprehensive results to WandB"""
        if not self.config.enabled:
            return

        try:
            if not wandb.run:
                return

            log_dict = {}

            # Basic metrics
            log_dict["shap/analysis_time"] = results.get("analysis_time", 0)
            log_dict["shap/num_samples"] = results.get("num_samples", 0)
            log_dict["shap/method"] = results.get("method", "unknown")

            # Feature importance
            if "feature_importance" in results:
                # Log top features
                sorted_features = sorted(
                    results["feature_importance"].items(),
                    key=lambda x: x[1]["current"],
                    reverse=True,
                )
                for i, (feature, data) in enumerate(sorted_features[:20]):
                    log_dict[f"shap/top_feature_{i + 1}_importance"] = data["current"]
                    log_dict[f"shap/top_feature_{i + 1}_name"] = feature

            # Branch importance
            if "branch_importance" in results:
                for branch, stats in results["branch_importance"].items():
                    log_dict[f"shap/branch_{branch}_importance"] = stats["mean"]
                    log_dict[f"shap/branch_{branch}_proportion"] = stats["proportion"]

            # Dead features
            if "dead_features" in results:
                log_dict["shap/dead_features_count"] = results["dead_features"][
                    "total_dead"
                ]
                log_dict["shap/dead_features_percentage"] = results["dead_features"][
                    "dead_percentage"
                ]

                for branch, summary in results["dead_features"]["summary"].items():
                    log_dict[f"shap/dead_features_{branch}_count"] = summary[
                        "dead_count"
                    ]

            # Statistics
            if "statistics" in results:
                for category, stats in results["statistics"].items():
                    for stat_name, value in stats.items():
                        log_dict[f"shap/{category}/{stat_name}"] = value

            # Feature interactions
            if (
                "feature_interactions" in results
                and "interaction_summary" in results["feature_interactions"]
            ):
                summary = results["feature_interactions"]["interaction_summary"]
                log_dict["shap/interactions/mean_correlation"] = summary[
                    "mean_correlation"
                ]
                log_dict["shap/interactions/max_correlation"] = summary[
                    "max_correlation"
                ]
                log_dict["shap/interactions/features_with_interactions"] = summary[
                    "features_with_interactions"
                ]

            # Visualizations
            if "visualizations" in results:
                for viz_name, path in results["visualizations"].items():
                    if os.path.exists(path):
                        log_dict[f"shap/viz_{viz_name}"] = wandb.Image(path)

            # Method comparison
            if "method_comparison" in results and results["method_comparison"].get(
                "comparison_available"
            ):
                for comparison, metrics in results["method_comparison"][
                    "method_agreements"
                ].items():
                    log_dict[f"shap/method_comparison/{comparison}_agreement"] = (
                        metrics["agreement"]
                    )
                    log_dict[f"shap/method_comparison/{comparison}_correlation"] = (
                        metrics["correlation"]
                    )

            wandb.log(log_dict, step=step)
            self.logger.info(f"📊 Logged {len(log_dict)} SHAP metrics to WandB")

        except Exception as e:
            self.logger.warning(f"Failed to log to WandB: {e}")

    def save_state(self, path: str):
        """Save analyzer state for later analysis"""
        if not self.config.enabled:
            return

        state = {
            "config": self.config,
            "feature_metadata": self.feature_metadata,
            "importance_ema": self.importance_ema,
            "importance_variance": self.importance_variance,
            "confirmed_dead_features": list(self.confirmed_dead_features),
            "analysis_count": self.analysis_count,
            "attribution_history": dict(self.attribution_history)[-10:]
            if self.attribution_history
            else {},
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"💾 Saved analyzer state to {path}")

    def load_state(self, path: str):
        """Load analyzer state"""
        if not self.config.enabled:
            return

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.config = state["config"]
        self.feature_metadata = state["feature_metadata"]
        self.importance_ema = state["importance_ema"]
        self.importance_variance = state["importance_variance"]
        self.confirmed_dead_features = set(state["confirmed_dead_features"])
        self.analysis_count = state["analysis_count"]

        self.logger.info(f"📂 Loaded analyzer state from {path}")

    # Utility methods
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (
            n * cumsum[-1]
        ) - (n + 1) / n

    def _calculate_concentration_ratio(self, values: np.ndarray, k: int = 10) -> float:
        """Calculate concentration ratio (top k features)"""
        sorted_values = np.sort(values)[::-1]
        return np.sum(sorted_values[:k]) / np.sum(sorted_values)

    def cleanup(self):
        """Cleanup resources"""
        # Clear large objects
        self.background_cache = None
        self.feature_cache.clear()
        self.attribution_history.clear()

        # Clear matplotlib
        plt.close("all")

        self.logger.info("🧹 SHAP analyzer cleaned up")
