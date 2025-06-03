"""Captum-based Feature Attribution Analyzer for Trading Models

Provides comprehensive gradient-based feature attribution using Captum's
state-of-the-art interpretability methods including Integrated Gradients,
GradientShap, DeepLift, and more.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum

try:
    import captum
    from captum.attr import (
        IntegratedGradients,
        GradientShap,
        DeepLift,
        DeepLiftShap,
        InputXGradient,
        Saliency,
        NoiseTunnel,
        FeatureAblation,
        FeaturePermutation,
        LayerGradientXActivation,
        LayerIntegratedGradients,
        LayerConductance,
        NeuronConductance,
        visualization as viz
    )
    from captum.metrics import infidelity, sensitivity_max
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    viz = None
    logging.warning("Captum not available. Install with: poetry add captum")


class AttributionMethod(Enum):
    """Available attribution methods in Captum"""
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEP_LIFT = "deep_lift"
    DEEP_LIFT_SHAP = "deep_lift_shap"
    INPUT_X_GRADIENT = "input_x_gradient"
    SALIENCY = "saliency"
    FEATURE_ABLATION = "feature_ablation"
    FEATURE_PERMUTATION = "feature_permutation"


@dataclass
class AttributionConfig:
    """Configuration for attribution analysis"""
    primary_method: AttributionMethod = AttributionMethod.INTEGRATED_GRADIENTS
    secondary_methods: List[AttributionMethod] = None
    n_steps: int = 50  # For integrated gradients
    n_samples: int = 25  # For gradient shap
    use_noise_tunnel: bool = True  # Add noise for smoothing
    noise_tunnel_samples: int = 10
    track_gradients: bool = True
    track_activations: bool = True
    compute_convergence_delta: bool = True
    baseline_type: str = "zero"  # "zero", "mean", "gaussian"
    
    def __post_init__(self):
        if self.secondary_methods is None:
            self.secondary_methods = [
                AttributionMethod.GRADIENT_SHAP,
                AttributionMethod.DEEP_LIFT
            ]


class CaptumFeatureAnalyzer:
    """
    Comprehensive feature attribution analysis using Captum for multi-branch transformer models.
    
    Key features:
    - Multiple attribution methods (Integrated Gradients, GradientShap, DeepLift, etc.)
    - Gradient flow analysis and visualization
    - Layer-wise attribution analysis
    - Neuron importance tracking
    - Real-time attribution tracking during training
    - Attribution uncertainty quantification
    - Cross-method consensus analysis
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Dict[str, List[str]],
        device: Union[str, torch.device] = None,
        config: AttributionConfig = None,
        logger: Optional[logging.Logger] = None,
        model_config: Optional[Any] = None
    ):
        """
        Initialize Captum feature analyzer.
        
        Args:
            model: The transformer model to analyze
            feature_names: Dict mapping branch names to feature name lists
            device: Torch device for computations
            config: Attribution configuration
            logger: Optional logger for debugging
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required for feature attribution. Install with: poetry add captum")
            
        self.model = model
        self.feature_names = feature_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or AttributionConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = model_config
        
        # Branch information
        self.branch_names = ['hf', 'mf', 'lf', 'portfolio']
        self.total_features = sum(len(names) for names in feature_names.values())
        
        # Set branch configurations from model config if available
        if model_config and hasattr(model_config, 'hf_seq_len'):
            self.branch_configs = {
                'hf': (model_config.hf_seq_len, model_config.hf_feat_dim),
                'mf': (model_config.mf_seq_len, model_config.mf_feat_dim),
                'lf': (model_config.lf_seq_len, model_config.lf_feat_dim),
                'portfolio': (model_config.portfolio_seq_len, model_config.portfolio_feat_dim)
            }
            self.logger.info(f"Using model config dimensions: hf=({model_config.hf_seq_len},{model_config.hf_feat_dim}), "
                           f"mf=({model_config.mf_seq_len},{model_config.mf_feat_dim}), "
                           f"lf=({model_config.lf_seq_len},{model_config.lf_feat_dim}), "
                           f"portfolio=({model_config.portfolio_seq_len},{model_config.portfolio_feat_dim})")
        else:
            # Fallback to updated defaults
            self.branch_configs = {
                'hf': (60, 7),
                'mf': (30, 43),
                'lf': (30, 19),
                'portfolio': (5, 10)
            }
            self.logger.warning(f"No model_config provided, using fallback dimensions: {self.branch_configs}")
        
        # Attribution tracking
        self.attribution_history = defaultdict(lambda: deque(maxlen=1000))
        self.gradient_history = defaultdict(lambda: deque(maxlen=1000))
        self.activation_history = defaultdict(lambda: deque(maxlen=1000))
        self.consensus_scores = defaultdict(float)
        
        # Initialize attribution methods
        self._initialize_attribution_methods()
        
        # Gradient tracking
        self.gradient_hooks = []
        self.activation_hooks = []
        self.gradients = {}
        self.activations = {}
        
        # Setup gradient tracking if enabled
        if self.config.track_gradients:
            self._setup_gradient_hooks()
            
        self.logger.info(f"CaptumFeatureAnalyzer initialized with {len(self.attribution_methods)} methods")
    
    def _initialize_attribution_methods(self):
        """Initialize Captum attribution methods"""
        self.attribution_methods = {}
        
        # Create a wrapper for the model that handles dict inputs
        self.wrapped_model = self._create_model_wrapper()
        
        # Primary method
        if self.config.primary_method == AttributionMethod.INTEGRATED_GRADIENTS:
            self.attribution_methods['primary'] = IntegratedGradients(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.GRADIENT_SHAP:
            self.attribution_methods['primary'] = GradientShap(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.DEEP_LIFT:
            self.attribution_methods['primary'] = DeepLift(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.DEEP_LIFT_SHAP:
            self.attribution_methods['primary'] = DeepLiftShap(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.INPUT_X_GRADIENT:
            self.attribution_methods['primary'] = InputXGradient(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.SALIENCY:
            self.attribution_methods['primary'] = Saliency(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.FEATURE_ABLATION:
            self.attribution_methods['primary'] = FeatureAblation(self.wrapped_model)
        elif self.config.primary_method == AttributionMethod.FEATURE_PERMUTATION:
            self.attribution_methods['primary'] = FeaturePermutation(self.wrapped_model)
            
        # Add noise tunnel if configured
        if self.config.use_noise_tunnel:
            self.attribution_methods['primary'] = NoiseTunnel(
                self.attribution_methods['primary']
            )
        
        # Secondary methods for consensus
        for method in self.config.secondary_methods:
            if method == AttributionMethod.INTEGRATED_GRADIENTS:
                attr_method = IntegratedGradients(self.wrapped_model)
            elif method == AttributionMethod.GRADIENT_SHAP:
                attr_method = GradientShap(self.wrapped_model)
            elif method == AttributionMethod.DEEP_LIFT:
                attr_method = DeepLift(self.wrapped_model)
            elif method == AttributionMethod.INPUT_X_GRADIENT:
                attr_method = InputXGradient(self.wrapped_model)
            elif method == AttributionMethod.SALIENCY:
                attr_method = Saliency(self.wrapped_model)
            else:
                continue
                
            self.attribution_methods[method.value] = attr_method
            
    def _create_model_wrapper(self) -> Callable:
        """Create a wrapper function that converts tensor inputs to model dict format"""
        def model_wrapper(inputs: torch.Tensor) -> torch.Tensor:
            """
            Wrapper that takes flattened tensor input and returns action probabilities.
            
            Args:
                inputs: Flattened tensor of shape [batch, total_features]
                
            Returns:
                Action probabilities or logits
            """
            # Convert flattened tensor to state dict
            state_dict = self._tensor_to_state_dict(inputs)
            
            # Get model output
            with torch.enable_grad():
                action_params, _ = self.model(state_dict)
                
                if isinstance(action_params, tuple) and len(action_params) == 2:
                    # For discrete actions, combine type and size logits
                    type_logits, size_logits = action_params
                    # Return the maximum probability action's logit as target
                    type_probs = F.softmax(type_logits, dim=-1)
                    size_probs = F.softmax(size_logits, dim=-1)
                    
                    # Combine probabilities and get max
                    max_type_prob, _ = type_probs.max(dim=-1, keepdim=True)
                    max_size_prob, _ = size_probs.max(dim=-1, keepdim=True)
                    combined = max_type_prob * max_size_prob
                    
                    return combined
                else:
                    return action_params[0] if isinstance(action_params, tuple) else action_params
                    
        return model_wrapper
    
    def _tensor_to_state_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flattened tensor back to state dict format"""
        batch_size = tensor.shape[0]
        total_expected_size = sum(seq_len * feat_dim for seq_len, feat_dim in self.branch_configs.values())
        
        if tensor.shape[1] != total_expected_size:
            self.logger.error(f"Tensor size mismatch: expected {total_expected_size}, got {tensor.shape[1]}")
            self.logger.error(f"Branch configs: {self.branch_configs}")
            self.logger.error(f"Tensor shape: {tensor.shape}")
            raise ValueError(f"Input tensor size {tensor.shape[1]} does not match expected size {total_expected_size}")
        
        state_dict = {}
        start_idx = 0
        
        for branch, (seq_len, feat_dim) in self.branch_configs.items():
            end_idx = start_idx + (seq_len * feat_dim)
            branch_tensor = tensor[:, start_idx:end_idx].reshape(batch_size, seq_len, feat_dim)
            state_dict[branch] = branch_tensor.to(self.device)
            start_idx = end_idx
            
        return state_dict
    
    def _state_dict_to_tensor(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert state dict to flattened tensor for attribution"""
        tensors = []
        
        # Process branches in consistent order
        for branch in self.branch_names:
            if branch in state_dict:
                branch_tensor = state_dict[branch]
                expected_seq_len, expected_feat_dim = self.branch_configs[branch]
                
                # Log current tensor shape for debugging
                self.logger.debug(f"Processing {branch}: shape={branch_tensor.shape}, expected=({expected_seq_len}, {expected_feat_dim})")
                
                # Ensure tensor is 2D (seq_len, feat_dim) before flattening
                if branch_tensor.dim() == 4:  # [batch=1, 1, seq, feat]
                    branch_tensor = branch_tensor.squeeze(0).squeeze(0)  # Remove batch dims
                elif branch_tensor.dim() == 3:  # [batch=1, seq, feat] or [1, seq, feat]
                    if branch_tensor.shape[0] == 1:
                        branch_tensor = branch_tensor.squeeze(0)  # Remove batch dim
                    else:
                        # Multiple samples in batch - take first one
                        branch_tensor = branch_tensor[0]
                elif branch_tensor.dim() == 2:  # [seq, feat] - already correct
                    pass
                elif branch_tensor.dim() == 1:  # Already flattened
                    # Reshape to expected dimensions
                    expected_size = expected_seq_len * expected_feat_dim
                    if branch_tensor.shape[0] == expected_size:
                        branch_tensor = branch_tensor.reshape(expected_seq_len, expected_feat_dim)
                    else:
                        self.logger.warning(f"Tensor size mismatch for {branch}: got {branch_tensor.shape[0]}, expected {expected_size}")
                        # Pad or truncate to expected size
                        if branch_tensor.shape[0] < expected_size:
                            padding = torch.zeros(expected_size - branch_tensor.shape[0], device=branch_tensor.device)
                            branch_tensor = torch.cat([branch_tensor, padding])
                        else:
                            branch_tensor = branch_tensor[:expected_size]
                        branch_tensor = branch_tensor.reshape(expected_seq_len, expected_feat_dim)
                
                # Validate final shape
                if branch_tensor.shape != (expected_seq_len, expected_feat_dim):
                    self.logger.error(f"Shape mismatch for {branch}: got {branch_tensor.shape}, expected ({expected_seq_len}, {expected_feat_dim})")
                    # Force reshape if possible
                    try:
                        total_elements = branch_tensor.numel()
                        expected_elements = expected_seq_len * expected_feat_dim
                        if total_elements == expected_elements:
                            branch_tensor = branch_tensor.reshape(expected_seq_len, expected_feat_dim)
                        else:
                            # Create tensor of correct shape filled with zeros, copy what we can
                            correct_tensor = torch.zeros(expected_seq_len, expected_feat_dim, device=branch_tensor.device)
                            flat_input = branch_tensor.flatten()
                            copy_size = min(flat_input.shape[0], expected_elements)
                            correct_tensor.flatten()[:copy_size] = flat_input[:copy_size]
                            branch_tensor = correct_tensor
                    except Exception as e:
                        self.logger.error(f"Failed to fix shape for {branch}: {e}")
                        # Create zero tensor of correct shape as fallback
                        branch_tensor = torch.zeros(expected_seq_len, expected_feat_dim, device=self.device)
                
                # Flatten to 1D for concatenation
                flattened = branch_tensor.flatten()
                tensors.append(flattened)
            else:
                # Branch not present - create zero tensor
                expected_seq_len, expected_feat_dim = self.branch_configs[branch]
                zero_tensor = torch.zeros(expected_seq_len * expected_feat_dim, device=self.device)
                tensors.append(zero_tensor)
        
        # Concatenate all branch tensors
        full_tensor = torch.cat(tensors, dim=0)
        
        # Add batch dimension
        return full_tensor.unsqueeze(0)
    
    def _setup_gradient_hooks(self):
        """Setup hooks to track gradients and activations"""
        def gradient_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
            
        def activation_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook important layers
        if hasattr(self.model, 'branch_processors'):
            for branch_name, processor in self.model.branch_processors.items():
                # Hook the transformer layers
                if hasattr(processor, 'transformer_layers'):
                    for i, layer in enumerate(processor.transformer_layers):
                        layer_name = f"{branch_name}_transformer_{i}"
                        self.gradient_hooks.append(
                            layer.register_backward_hook(gradient_hook(layer_name))
                        )
                        self.activation_hooks.append(
                            layer.register_forward_hook(activation_hook(layer_name))
                        )
                        
        # Hook fusion layer
        if hasattr(self.model, 'fusion_layer'):
            self.gradient_hooks.append(
                self.model.fusion_layer.register_backward_hook(gradient_hook('fusion'))
            )
            self.activation_hooks.append(
                self.model.fusion_layer.register_forward_hook(activation_hook('fusion'))
            )
    
    def calculate_attributions(
        self,
        sample_states: List[Dict[str, torch.Tensor]],
        target_actions: Optional[torch.Tensor] = None,
        baseline_states: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_convergence_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate feature attributions using configured methods.
        
        Args:
            sample_states: List of state dicts to analyze
            target_actions: Target actions for attribution (if None, uses model's choice)
            baseline_states: Baseline states for comparison
            return_convergence_delta: Whether to compute convergence metrics
            
        Returns:
            Dict containing attribution results and metrics
        """
        # Convert states to tensors
        sample_tensor = torch.stack([
            self._state_dict_to_tensor(state) for state in sample_states
        ])
        
        # Create baseline
        if baseline_states is None:
            baseline = self._create_baseline(sample_tensor)
        else:
            baseline = torch.stack([
                self._state_dict_to_tensor(state) for state in baseline_states
            ])
        
        # Get target if not provided
        if target_actions is None:
            with torch.no_grad():
                outputs = self.wrapped_model(sample_tensor)
                target_actions = outputs.argmax(dim=-1) if outputs.dim() > 1 else None
        
        results = {
            'attributions': {},
            'gradients': {},
            'metrics': {},
            'consensus': {}
        }
        
        # Calculate attributions with primary method
        primary_attrs = self._calculate_primary_attribution(
            sample_tensor, baseline, target_actions, return_convergence_delta
        )
        results['attributions'][self.config.primary_method.value] = primary_attrs
        
        # Calculate with secondary methods for consensus
        if self.config.secondary_methods:
            secondary_results = self._calculate_secondary_attributions(
                sample_tensor, baseline, target_actions
            )
            results['attributions'].update(secondary_results)
            
            # Calculate consensus
            results['consensus'] = self._calculate_consensus(results['attributions'])
        
        # Extract gradients if tracked
        if self.config.track_gradients and self.gradients:
            results['gradients'] = self._extract_gradient_info()
            
        # Convert attributions back to branch structure
        results['branch_attributions'] = self._convert_to_branch_structure(
            primary_attrs['attributions']
        )
        
        # Calculate metrics
        results['metrics'] = self._calculate_attribution_metrics(
            sample_tensor, primary_attrs['attributions'], baseline
        )
        
        # Update history
        self._update_attribution_history(results)
        
        return results
    
    def _calculate_primary_attribution(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        target: Optional[torch.Tensor],
        return_convergence_delta: bool
    ) -> Dict[str, torch.Tensor]:
        """Calculate attribution with primary method"""
        method = self.attribution_methods['primary']
        
        kwargs = {
            'inputs': inputs,
            'target': target,
            'baselines': baseline
        }
        
        # Add method-specific parameters
        if isinstance(method, IntegratedGradients) or (
            isinstance(method, NoiseTunnel) and isinstance(method.attribution_method, IntegratedGradients)
        ):
            kwargs['n_steps'] = self.config.n_steps
            kwargs['return_convergence_delta'] = return_convergence_delta
        elif isinstance(method, GradientShap) or (
            isinstance(method, NoiseTunnel) and isinstance(method.attribution_method, GradientShap)
        ):
            kwargs['n_samples'] = self.config.n_samples
            kwargs['stdevs'] = 0.1
            
        # Handle NoiseTunnel
        if isinstance(method, NoiseTunnel):
            kwargs['nt_samples'] = self.config.noise_tunnel_samples
            kwargs['nt_type'] = 'smoothgrad'
            
        # Calculate attributions
        if return_convergence_delta and 'return_convergence_delta' in kwargs:
            attributions, delta = method.attribute(**kwargs)
            return {
                'attributions': attributions,
                'convergence_delta': delta
            }
        else:
            attributions = method.attribute(**kwargs)
            return {'attributions': attributions}
    
    def _calculate_secondary_attributions(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        target: Optional[torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Calculate attributions with secondary methods"""
        results = {}
        
        for method_name, method in self.attribution_methods.items():
            if method_name == 'primary':
                continue
                
            try:
                kwargs = {
                    'inputs': inputs,
                    'target': target,
                    'baselines': baseline
                }
                
                # Add method-specific parameters
                if isinstance(method, IntegratedGradients):
                    kwargs['n_steps'] = self.config.n_steps // 2  # Faster for secondary
                elif isinstance(method, GradientShap):
                    kwargs['n_samples'] = self.config.n_samples // 2
                    kwargs['stdevs'] = 0.1
                    
                attributions = method.attribute(**kwargs)
                results[method_name] = {'attributions': attributions}
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate {method_name} attributions: {e}")
                
        return results
    
    def _create_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create baseline for attribution based on configuration"""
        if self.config.baseline_type == "zero":
            return torch.zeros_like(inputs)
        elif self.config.baseline_type == "mean":
            return inputs.mean(dim=0, keepdim=True).expand_as(inputs)
        elif self.config.baseline_type == "gaussian":
            return torch.randn_like(inputs) * inputs.std() + inputs.mean()
        else:
            return torch.zeros_like(inputs)
    
    def _calculate_consensus(self, attributions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate consensus scores across different attribution methods"""
        # Extract attribution tensors
        attr_tensors = []
        for method_results in attributions.values():
            if 'attributions' in method_results:
                attr_tensors.append(method_results['attributions'].abs())
        
        if len(attr_tensors) < 2:
            return {}
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(attr_tensors)):
            for j in range(i + 1, len(attr_tensors)):
                # Flatten and correlate
                attr_i = attr_tensors[i].flatten()
                attr_j = attr_tensors[j].flatten()
                corr = torch.corrcoef(torch.stack([attr_i, attr_j]))[0, 1].item()
                correlations.append(corr)
        
        # Calculate consensus metrics
        consensus = {
            'mean_correlation': np.mean(correlations),
            'min_correlation': np.min(correlations),
            'std_correlation': np.std(correlations),
            'agreement_score': np.mean([c > 0.7 for c in correlations])  # % with high correlation
        }
        
        return consensus
    
    def _convert_to_branch_structure(self, attributions: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert flattened attributions back to branch structure"""
        batch_size = attributions.shape[0]
        branch_attrs = {}
        start_idx = 0
        
        for branch, (seq_len, feat_dim) in self.branch_configs.items():
            end_idx = start_idx + (seq_len * feat_dim)
            branch_attr = attributions[:, start_idx:end_idx]
            
            # Reshape to [batch, seq_len, feat_dim]
            branch_attr = branch_attr.reshape(batch_size, seq_len, feat_dim)
            
            # Average over sequence dimension for feature importance
            branch_attrs[branch] = branch_attr.abs().mean(dim=1).cpu().numpy()
            
            start_idx = end_idx
        
        return branch_attrs
    
    def _extract_gradient_info(self) -> Dict[str, Any]:
        """Extract gradient information from hooks"""
        grad_info = {}
        
        for layer_name, gradients in self.gradients.items():
            grad_info[layer_name] = {
                'mean': gradients.abs().mean().item(),
                'std': gradients.std().item(),
                'max': gradients.abs().max().item(),
                'norm': gradients.norm().item()
            }
            
        return grad_info
    
    def _calculate_attribution_metrics(
        self,
        inputs: torch.Tensor,
        attributions: torch.Tensor,
        baseline: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate quality metrics for attributions"""
        metrics = {}
        
        # Infidelity metric - measures attribution quality
        try:
            infidelity_score = infidelity(
                self.wrapped_model,
                inputs,
                attributions,
                baselines=baseline,
                n_perturb_samples=10
            )
            metrics['infidelity'] = infidelity_score.mean().item()
        except Exception as e:
            self.logger.debug(f"Failed to calculate infidelity: {e}")
            
        # Sensitivity metric - measures attribution stability
        try:
            sensitivity = sensitivity_max(
                self.wrapped_model,
                inputs,
                attributions,
                baselines=baseline
            )
            metrics['sensitivity'] = sensitivity.mean().item()
        except Exception as e:
            self.logger.debug(f"Failed to calculate sensitivity: {e}")
            
        # Sparsity - percentage of features with low attribution
        attr_abs = attributions.abs()
        threshold = attr_abs.mean() * 0.1  # 10% of mean
        metrics['sparsity'] = (attr_abs < threshold).float().mean().item()
        
        # Concentration - how concentrated attributions are
        attr_normalized = attr_abs / (attr_abs.sum(dim=-1, keepdim=True) + 1e-8)
        metrics['concentration'] = attr_normalized.max(dim=-1)[0].mean().item()
        
        return metrics
    
    def _update_attribution_history(self, results: Dict[str, Any]):
        """Update attribution history for tracking"""
        # Store branch attributions
        if 'branch_attributions' in results:
            for branch, attrs in results['branch_attributions'].items():
                mean_attrs = attrs.mean(axis=0)  # Average over batch
                for i, feature_name in enumerate(self.feature_names.get(branch, [])):
                    self.attribution_history[f"{branch}.{feature_name}"].append(mean_attrs[i])
        
        # Store consensus scores
        if 'consensus' in results:
            for metric, value in results['consensus'].items():
                self.consensus_scores[metric] = value
    
    def get_feature_importance_ranking(
        self,
        n_top: int = 10,
        method: str = "recent"
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features by importance.
        
        Args:
            n_top: Number of top features per branch
            method: "recent" for latest attributions, "historical" for averaged
            
        Returns:
            Dict mapping branch names to ranked features
        """
        rankings = {}
        
        for branch in self.branch_names:
            if branch not in self.feature_names:
                continue
                
            branch_features = []
            
            for i, feature_name in enumerate(self.feature_names[branch]):
                key = f"{branch}.{feature_name}"
                
                if key in self.attribution_history and self.attribution_history[key]:
                    if method == "recent":
                        importance = abs(self.attribution_history[key][-1])
                    else:  # historical
                        importance = np.mean([abs(x) for x in self.attribution_history[key]])
                    
                    branch_features.append((feature_name, float(importance)))
            
            # Sort by importance
            branch_features.sort(key=lambda x: x[1], reverse=True)
            rankings[branch] = branch_features[:n_top]
        
        return rankings
    
    def analyze_gradient_flow(self) -> Dict[str, Any]:
        """Analyze gradient flow through the model"""
        if not self.gradients:
            return {"error": "No gradients tracked. Enable gradient tracking."}
        
        flow_analysis = {
            'layer_gradients': {},
            'gradient_norms': {},
            'vanishing_gradients': [],
            'exploding_gradients': []
        }
        
        for layer_name, grad_info in self._extract_gradient_info().items():
            flow_analysis['layer_gradients'][layer_name] = grad_info
            flow_analysis['gradient_norms'][layer_name] = grad_info['norm']
            
            # Detect gradient issues
            if grad_info['norm'] < 1e-7:
                flow_analysis['vanishing_gradients'].append(layer_name)
            elif grad_info['norm'] > 1e3:
                flow_analysis['exploding_gradients'].append(layer_name)
        
        return flow_analysis
    
    def generate_attribution_report(
        self,
        sample_states: List[Dict[str, torch.Tensor]],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive attribution report with visualizations.
        
        Args:
            sample_states: States to analyze
            save_path: Optional path to save visualizations
            
        Returns:
            Comprehensive attribution report
        """
        self.logger.info("Generating comprehensive attribution report...")
        
        # Calculate attributions
        results = self.calculate_attributions(sample_states)
        
        # Get rankings
        rankings = self.get_feature_importance_ranking(n_top=10)
        
        # Analyze gradient flow
        gradient_analysis = self.analyze_gradient_flow()
        
        report = {
            'timestamp': time.time(),
            'n_samples': len(sample_states),
            'primary_method': self.config.primary_method.value,
            'feature_rankings': rankings,
            'attribution_metrics': results.get('metrics', {}),
            'consensus_analysis': results.get('consensus', {}),
            'gradient_flow': gradient_analysis,
            'branches': {}
        }
        
        # Add branch-specific analysis
        for branch in self.branch_names:
            if branch not in self.feature_names:
                continue
                
            branch_data = {
                'n_features': len(self.feature_names[branch]),
                'top_features': rankings.get(branch, []),
                'feature_names': self.feature_names[branch]
            }
            
            # Add attribution statistics
            if branch in results.get('branch_attributions', {}):
                attrs = results['branch_attributions'][branch]
                branch_data['attribution_stats'] = {
                    'mean': float(np.mean(attrs)),
                    'std': float(np.std(attrs)),
                    'max': float(np.max(attrs)),
                    'min': float(np.min(attrs))
                }
            
            report['branches'][branch] = branch_data
        
        # Generate visualizations if requested
        if save_path:
            self._generate_visualizations(results, report, save_path)
        
        return report
    
    def _generate_visualizations(
        self,
        results: Dict[str, Any],
        report: Dict[str, Any],
        save_path: str
    ):
        """Generate and save attribution visualizations using both custom plots and Captum's visualization tools"""
        try:
            # Create comprehensive visualization figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            fig.suptitle('Captum Feature Attribution Analysis', fontsize=16)
            
            # Plot 1: Feature importance by branch (custom)
            ax1 = fig.add_subplot(gs[0, 0])
            branch_colors = {'hf': 'red', 'mf': 'blue', 'lf': 'green', 'portfolio': 'orange'}
            for i, (branch, rankings) in enumerate(report['feature_rankings'].items()):
                if rankings:
                    features, importances = zip(*rankings[:5])
                    y_pos = np.arange(len(features)) + i * 0.2
                    ax1.barh(y_pos, importances, height=0.15, 
                            label=branch, alpha=0.7, color=branch_colors.get(branch, 'gray'))
            ax1.set_xlabel('Attribution Score')
            ax1.set_title('Top Features by Branch')
            ax1.legend()
            
            # Plot 2: Attribution distribution (custom)
            ax2 = fig.add_subplot(gs[0, 1])
            all_attributions = []
            for branch_attrs in results.get('branch_attributions', {}).values():
                all_attributions.extend(branch_attrs.flatten())
            if all_attributions:
                ax2.hist(all_attributions, bins=50, alpha=0.7, color='blue', edgecolor='black')
                ax2.set_xlabel('Attribution Value')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Attribution Distribution')
                ax2.axvline(np.mean(all_attributions), color='red', linestyle='--', label=f'Mean: {np.mean(all_attributions):.4f}')
                ax2.legend()
            
            # Plot 3: Consensus analysis (custom)
            ax3 = fig.add_subplot(gs[0, 2])
            if report.get('consensus_analysis'):
                metrics = list(report['consensus_analysis'].keys())
                values = list(report['consensus_analysis'].values())
                colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
                bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
                ax3.set_ylabel('Score')
                ax3.set_title('Attribution Method Consensus')
                ax3.set_ylim(0, 1)
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Gradient flow (custom)
            ax4 = fig.add_subplot(gs[1, 0])
            if report.get('gradient_flow', {}).get('gradient_norms'):
                layers = list(report['gradient_flow']['gradient_norms'].keys())
                norms = list(report['gradient_flow']['gradient_norms'].values())
                ax4.semilogy(range(len(layers)), norms, 'o-', markersize=8, linewidth=2)
                ax4.set_xticks(range(len(layers)))
                ax4.set_xticklabels(layers, rotation=45, ha='right')
                ax4.set_ylabel('Gradient Norm (log scale)')
                ax4.set_title('Gradient Flow Analysis')
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Branch-wise attribution heatmap (custom)
            ax5 = fig.add_subplot(gs[1, 1])
            if 'branch_attributions' in results:
                branch_data = []
                branch_labels = []
                for branch, attrs in results['branch_attributions'].items():
                    # Take mean across samples and select top features
                    mean_attrs = attrs.mean(axis=0)
                    top_indices = np.argsort(np.abs(mean_attrs))[-10:]  # Top 10 features
                    branch_data.append(mean_attrs[top_indices])
                    branch_labels.append(branch)
                
                if branch_data:
                    # Create heatmap
                    import seaborn as sns
                    heatmap_data = np.array([data for data in branch_data if len(data) > 0])
                    if heatmap_data.size > 0:
                        sns.heatmap(heatmap_data, yticklabels=branch_labels, 
                                  cmap='RdBu_r', center=0, ax=ax5, cbar_kws={'label': 'Attribution Score'})
                        ax5.set_title('Branch Attribution Heatmap (Top Features)')
                        ax5.set_xlabel('Feature Index (Top 10)')
            
            # Plot 6: Attribution ranking comparison (custom)
            ax6 = fig.add_subplot(gs[1, 2])
            if len(results.get('attributions', {})) > 1:
                # Compare different attribution methods
                methods = list(results['attributions'].keys())
                method_scores = []
                for method in methods:
                    if 'attributions' in results['attributions'][method]:
                        attrs = results['attributions'][method]['attributions']
                        # Calculate total attribution magnitude per method
                        total_magnitude = torch.abs(attrs).sum().item()
                        method_scores.append(total_magnitude)
                    else:
                        method_scores.append(0)
                
                if method_scores:
                    bars = ax6.bar(methods, method_scores, alpha=0.7, color='skyblue', edgecolor='black')
                    ax6.set_ylabel('Total Attribution Magnitude')
                    ax6.set_title('Attribution Method Comparison')
                    ax6.tick_params(axis='x', rotation=45)
                    # Add value labels
                    for bar, score in zip(bars, method_scores):
                        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(method_scores)*0.01,
                               f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 7: Feature stability over time (if we have historical data)
            ax7 = fig.add_subplot(gs[2, :])
            if hasattr(self, 'attribution_history') and self.attribution_history:
                # Plot top 5 most important features over time
                stability_data = {}
                for feature_key, history in self.attribution_history.items():
                    if len(history) >= 5:  # Need some history
                        # Calculate recent average importance
                        recent_avg = np.mean([abs(x) for x in list(history)[-10:]])
                        stability_data[feature_key] = (list(history), recent_avg)
                
                # Sort by importance and take top 5
                top_features = sorted(stability_data.items(), key=lambda x: x[1][1], reverse=True)[:5]
                
                for feature_key, (history, _) in top_features:
                    x_vals = range(len(history))
                    y_vals = [abs(val) for val in history]
                    ax7.plot(x_vals, y_vals, label=feature_key.split('.')[-1], marker='o', alpha=0.7)
                
                ax7.set_xlabel('Analysis Step')
                ax7.set_ylabel('Attribution Magnitude')
                ax7.set_title('Feature Importance Stability Over Time (Top 5 Features)')
                ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax7.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate additional Captum-specific visualizations if available
            self._generate_captum_native_visualizations(results, save_path)
            
            self.logger.info(f"Comprehensive attribution visualizations saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")
    
    def _generate_captum_native_visualizations(self, results: Dict[str, Any], base_save_path: str):
        """Generate visualizations using Captum's native visualization tools"""
        try:
            from pathlib import Path
            base_path = Path(base_save_path)
            captum_dir = base_path.parent / f"{base_path.stem}_captum_viz"
            captum_dir.mkdir(exist_ok=True)
            
            # Get primary attribution results
            primary_attrs = results.get('attributions', {}).get(self.config.primary_method.value, {})
            if 'attributions' not in primary_attrs:
                self.logger.debug("No primary attribution data for Captum visualizations")
                return
            
            attributions = primary_attrs['attributions']
            
            # Generate feature attribution bar chart using Captum's visualization
            if 'branch_attributions' in results:
                for branch, branch_attrs in results['branch_attributions'].items():
                    if branch in self.feature_names:
                        try:
                            # Prepare data for Captum visualization
                            feature_names = self.feature_names[branch]
                            mean_attrs = branch_attrs.mean(axis=0)  # Average across samples
                            
                            # Use Captum's visualization function
                            if viz is not None:
                                try:
                                    fig, ax = viz.visualize_importances_as_bar_chart(
                                        importances=mean_attrs,
                                        feature_names=feature_names,
                                        title=f'{branch.upper()} Branch Feature Attributions ({self.config.primary_method.value})',
                                        x_label='Attribution Score'
                                    )
                                except AttributeError:
                                    # Fallback to manual visualization if Captum function not available
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    y_pos = np.arange(len(feature_names))
                                    ax.barh(y_pos, mean_attrs, alpha=0.7)
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels(feature_names)
                                    ax.set_xlabel('Attribution Score')
                                    ax.set_title(f'{branch.upper()} Branch Feature Attributions ({self.config.primary_method.value})')
                                    ax.grid(True, alpha=0.3)
                            else:
                                # Fallback to manual visualization
                                fig, ax = plt.subplots(figsize=(12, 8))
                                y_pos = np.arange(len(feature_names))
                                ax.barh(y_pos, mean_attrs, alpha=0.7)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(feature_names)
                                ax.set_xlabel('Attribution Score')
                                ax.set_title(f'{branch.upper()} Branch Feature Attributions ({self.config.primary_method.value})')
                                ax.grid(True, alpha=0.3)
                            
                            # Save the figure
                            captum_path = captum_dir / f"{branch}_captum_bar_chart.png"
                            fig.savefig(captum_path, dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            
                            self.logger.debug(f"Saved Captum bar chart for {branch} to {captum_path}")
                            
                        except Exception as e:
                            self.logger.debug(f"Failed to generate Captum bar chart for {branch}: {e}")
            
            self.logger.info(f"Captum native visualizations saved to {captum_dir}")
            
        except Exception as e:
            self.logger.debug(f"Failed to generate Captum native visualizations: {e}")
    
    def get_dead_features(
        self,
        threshold: float = 0.01,
        min_history: int = 100
    ) -> Dict[str, List[str]]:
        """
        Identify features with consistently low attributions.
        
        Args:
            threshold: Attribution threshold for considering a feature "dead"
            min_history: Minimum history length required for analysis
            
        Returns:
            Dict mapping branches to lists of dead features
        """
        dead_features = {}
        
        for branch in self.branch_names:
            if branch not in self.feature_names:
                continue
                
            branch_dead = []
            
            for i, feature_name in enumerate(self.feature_names[branch]):
                key = f"{branch}.{feature_name}"
                
                if key in self.attribution_history:
                    history = list(self.attribution_history[key])
                    
                    if len(history) >= min_history:
                        # Check if consistently low attribution
                        mean_attr = np.mean([abs(x) for x in history[-min_history:]])
                        if mean_attr < threshold:
                            branch_dead.append(feature_name)
            
            if branch_dead:
                dead_features[branch] = branch_dead
        
        return dead_features
    
    def compare_model_decisions(
        self,
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Compare attributions between two different states to understand
        what drives different decisions.
        
        Args:
            state_a: First state
            state_b: Second state
            
        Returns:
            Comparison analysis
        """
        # Calculate attributions for both states
        attrs_a = self.calculate_attributions([state_a])
        attrs_b = self.calculate_attributions([state_b])
        
        # Get branch attributions
        branch_attrs_a = attrs_a['branch_attributions']
        branch_attrs_b = attrs_b['branch_attributions']
        
        comparison = {
            'attribution_differences': {},
            'key_discriminative_features': {},
            'decision_drivers': []
        }
        
        # Compare branch by branch
        for branch in self.branch_names:
            if branch in branch_attrs_a and branch in branch_attrs_b:
                diff = branch_attrs_a[branch][0] - branch_attrs_b[branch][0]
                comparison['attribution_differences'][branch] = diff
                
                # Find most discriminative features
                abs_diff = np.abs(diff)
                top_indices = np.argsort(abs_diff)[-5:][::-1]
                
                discriminative = []
                for idx in top_indices:
                    if idx < len(self.feature_names[branch]):
                        feature_name = self.feature_names[branch][idx]
                        discriminative.append({
                            'feature': feature_name,
                            'difference': float(diff[idx]),
                            'attr_a': float(branch_attrs_a[branch][0][idx]),
                            'attr_b': float(branch_attrs_b[branch][0][idx])
                        })
                
                comparison['key_discriminative_features'][branch] = discriminative
        
        return comparison
    
    def cleanup(self):
        """Remove gradient hooks and clean up resources"""
        for hook in self.gradient_hooks:
            hook.remove()
        for hook in self.activation_hooks:
            hook.remove()
        
        self.gradient_hooks.clear()
        self.activation_hooks.clear()
        self.gradients.clear()
        self.activations.clear()