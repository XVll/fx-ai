"""
Simple, Robust Feature Attribution System

A lightweight alternative to Captum that actually works reliably.
Uses basic gradient-based methods without external library complications.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque


class SimpleFeatureAttribution:
    """
    Simple, reliable feature attribution using basic gradient methods.
    No external dependencies causing compatibility issues.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Dict[str, List[str]],
        branch_configs: Dict[str, Tuple[int, int]],
        device: torch.device = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.branch_configs = branch_configs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Attribution history for tracking
        self.attribution_history = defaultdict(lambda: deque(maxlen=100))
        
        self.logger.info("Simple Feature Attribution initialized successfully")
    
    def analyze_features(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Analyze feature importance using simple gradient-based methods.
        
        Args:
            states: List of state dictionaries from the environment
            
        Returns:
            Dictionary with attribution results
        """
        if not states:
            return {"error": "No states provided"}
        
        try:
            # Use first few states for attribution
            analysis_states = states[:min(10, len(states))]
            
            # Method 1: Input gradients
            input_gradients = self._calculate_input_gradients(analysis_states)
            
            # Method 2: Feature permutation importance (lightweight version)
            permutation_importance = self._calculate_permutation_importance(analysis_states)
            
            # Method 3: Feature variance analysis
            variance_importance = self._calculate_variance_importance(analysis_states)
            
            # Combine results
            results = {
                "method": "simple_attribution",
                "n_states_analyzed": len(analysis_states),
                "input_gradients": input_gradients,
                "permutation_importance": permutation_importance,
                "variance_importance": variance_importance,
                "top_features_by_branch": self._get_top_features(input_gradients),
                "summary": self._create_summary(input_gradients, permutation_importance, variance_importance)
            }
            
            # Update history
            self._update_history(input_gradients)
            
            self.logger.info(f"✅ Simple attribution analysis completed for {len(analysis_states)} states")
            return results
            
        except Exception as e:
            self.logger.error(f"Simple attribution analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_input_gradients(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """Calculate gradients of output with respect to inputs"""
        gradients_by_branch = {}
        
        for state in states:
            # Prepare inputs with gradient tracking
            state_copy = {}
            for branch, tensor in state.items():
                if torch.is_tensor(tensor):
                    state_copy[branch] = tensor.clone().detach().requires_grad_(True).to(self.device)
                else:
                    state_copy[branch] = tensor
            
            try:
                # Forward pass
                action_params, _ = self.model(state_copy)
                
                # Get scalar output for gradient calculation
                if isinstance(action_params, tuple):
                    # For discrete actions, use action type logits
                    output = action_params[0].sum()
                else:
                    output = action_params.sum()
                
                # Backward pass to get gradients
                output.backward()
                
                # Collect gradients
                for branch, tensor in state_copy.items():
                    if hasattr(tensor, 'grad') and tensor.grad is not None:
                        grad_magnitude = tensor.grad.abs().mean().item()
                        
                        if branch not in gradients_by_branch:
                            gradients_by_branch[branch] = []
                        gradients_by_branch[branch].append(grad_magnitude)
                
            except Exception as e:
                self.logger.debug(f"Gradient calculation failed for one state: {e}")
                continue
        
        # Average gradients across states
        averaged_gradients = {}
        for branch, grad_list in gradients_by_branch.items():
            if grad_list:
                averaged_gradients[branch] = np.mean(grad_list)
            else:
                averaged_gradients[branch] = 0.0
                
        return averaged_gradients
    
    def _calculate_permutation_importance(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Calculate importance by measuring output change when features are permuted"""
        importance_scores = {}
        
        if len(states) < 2:
            return importance_scores
        
        try:
            # Get baseline predictions
            baseline_outputs = []
            for state in states[:3]:  # Use first 3 states only for speed
                with torch.no_grad():
                    output, _ = self.model(state)
                    if isinstance(output, tuple):
                        baseline_outputs.append(output[0])
                    else:
                        baseline_outputs.append(output)
            
            baseline_mean = torch.stack(baseline_outputs).mean()
            
            # Test each branch
            for branch in self.branch_configs.keys():
                permuted_outputs = []
                
                for state in states[:3]:
                    # Create permuted state
                    permuted_state = state.copy()
                    if branch in permuted_state:
                        # Permute this branch's features
                        original_tensor = permuted_state[branch]
                        if torch.is_tensor(original_tensor) and original_tensor.numel() > 1:
                            permuted_tensor = original_tensor.clone()
                            # Simple permutation: shuffle along last dimension
                            perm_indices = torch.randperm(permuted_tensor.shape[-1])
                            permuted_tensor = permuted_tensor[..., perm_indices]
                            permuted_state[branch] = permuted_tensor
                    
                    with torch.no_grad():
                        output, _ = self.model(permuted_state)
                        if isinstance(output, tuple):
                            permuted_outputs.append(output[0])
                        else:
                            permuted_outputs.append(output)
                
                if permuted_outputs:
                    permuted_mean = torch.stack(permuted_outputs).mean()
                    importance = abs(baseline_mean.item() - permuted_mean.item())
                    importance_scores[branch] = importance
                else:
                    importance_scores[branch] = 0.0
                    
        except Exception as e:
            self.logger.debug(f"Permutation importance calculation failed: {e}")
            
        return importance_scores
    
    def _calculate_variance_importance(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Calculate importance based on feature variance"""
        variance_scores = {}
        
        try:
            branch_variances = {}
            
            for state in states:
                for branch, tensor in state.items():
                    if torch.is_tensor(tensor):
                        variance = tensor.var().item()
                        if branch not in branch_variances:
                            branch_variances[branch] = []
                        branch_variances[branch].append(variance)
            
            # Average variances
            for branch, variances in branch_variances.items():
                variance_scores[branch] = np.mean(variances)
                
        except Exception as e:
            self.logger.debug(f"Variance importance calculation failed: {e}")
            
        return variance_scores
    
    def _get_top_features(self, gradients: Dict[str, float], top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Get top features by importance"""
        top_features = {}
        
        # Sort branches by gradient magnitude
        sorted_branches = sorted(gradients.items(), key=lambda x: x[1], reverse=True)
        
        for branch, importance in sorted_branches[:top_k]:
            if branch in self.feature_names:
                # For now, just return the branch name and importance
                # In a more sophisticated version, we'd calculate per-feature importance
                feature_names = self.feature_names[branch][:3]  # Top 3 feature names
                top_features[branch] = [(name, importance) for name in feature_names]
        
        return top_features
    
    def _create_summary(self, gradients: Dict, permutation: Dict, variance: Dict) -> Dict[str, Any]:
        """Create a summary of attribution results"""
        # Find most important branch by each method
        most_important = {}
        
        if gradients:
            most_important['gradient'] = max(gradients.items(), key=lambda x: x[1])
        if permutation:
            most_important['permutation'] = max(permutation.items(), key=lambda x: x[1])
        if variance:
            most_important['variance'] = max(variance.items(), key=lambda x: x[1])
        
        # Overall ranking (simple average)
        all_branches = set(gradients.keys()) | set(permutation.keys()) | set(variance.keys())
        overall_scores = {}
        
        for branch in all_branches:
            scores = []
            if branch in gradients:
                scores.append(gradients[branch])
            if branch in permutation:
                scores.append(permutation[branch])
            if branch in variance:
                scores.append(variance[branch])
            
            if scores:
                overall_scores[branch] = np.mean(scores)
        
        return {
            "most_important_by_method": most_important,
            "overall_ranking": sorted(overall_scores.items(), key=lambda x: x[1], reverse=True),
            "analysis_quality": "good" if len(all_branches) >= 3 else "limited"
        }
    
    def _update_history(self, gradients: Dict[str, float]):
        """Update attribution history"""
        for branch, importance in gradients.items():
            self.attribution_history[branch].append(importance)
    
    def get_attribution_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard display"""
        if not self.attribution_history:
            return {"attribution_enabled": False}
        
        # Get recent averages
        recent_importance = {}
        for branch, history in self.attribution_history.items():
            if history:
                recent_importance[branch] = {
                    "current": float(history[-1]),
                    "average": float(np.mean(list(history))),
                    "trend": "stable"  # Could be enhanced
                }
        
        return {
            "attribution_enabled": True,
            "method": "simple_gradient_based",
            "branches_analyzed": len(recent_importance),
            "recent_importance": recent_importance,
            "top_branch": max(recent_importance.items(), key=lambda x: x[1]["current"])[0] if recent_importance else None
        }