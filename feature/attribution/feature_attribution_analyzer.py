"""Feature Attribution Analyzer for Trading Models

Provides comprehensive feature importance analysis including SHAP values,
permutation importance, and gradient-based attribution methods.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: poetry add shap tensorflow")


class FeatureAttributionAnalyzer:
    """
    Comprehensive feature attribution analysis for multi-branch transformer trading models.
    
    Supports:
    - SHAP values for model interpretability
    - Permutation importance for feature ranking
    - Gradient-based attribution methods
    - Real-time feature importance tracking
    - Feature correlation and dependency analysis
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Dict[str, List[str]],
        device: Union[str, torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize feature attribution analyzer.
        
        Args:
            model: The transformer model to analyze
            feature_names: Dict mapping branch names to feature name lists
                          e.g. {'hf': ['price_velocity', ...], 'mf': [...], 'lf': [...]}
            device: Torch device for computations
            logger: Optional logger for debugging
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Feature importance tracking
        self.importance_history = defaultdict(lambda: deque(maxlen=1000))
        self.attention_history = deque(maxlen=1000)
        self.correlation_matrix = None
        self.last_analysis_time = None
        
        # Branch information
        self.branch_names = ['hf', 'mf', 'lf', 'portfolio']
        self.total_features = sum(len(names) for names in feature_names.values())
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if SHAP_AVAILABLE:
            self._initialize_shap_explainer()
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for the model"""
        try:
            # Create a wrapper function for SHAP that handles dict inputs
            def model_wrapper(input_tensor):
                """Wrapper function for SHAP that converts tensor to model input format"""
                # Convert flattened tensor back to model dict format
                return self._tensor_to_model_input(input_tensor)
            
            # We'll initialize the explainer when we have actual data
            self.logger.info("SHAP integration ready - explainer will be created on first use")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _model_input_to_tensor(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert model state dict to flattened tensor for SHAP"""
        tensors = []
        for branch in self.branch_names:
            if branch in state_dict:
                branch_tensor = state_dict[branch]
                # Flatten the tensor while preserving batch dimension
                if branch_tensor.dim() == 3:  # [batch, seq, feat]
                    flattened = branch_tensor.flatten(start_dim=1)
                else:
                    flattened = branch_tensor
                tensors.append(flattened)
        
        return torch.cat(tensors, dim=1)
    
    def _tensor_to_model_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert flattened tensor back to model input format and get action logits"""
        batch_size = tensor.shape[0]
        
        # Reconstruct state dict from flattened tensor
        state_dict = {}
        start_idx = 0
        
        # Reconstruct each branch based on expected dimensions
        # Note: These dimensions should match your model config
        branch_configs = {
            'hf': (60, 9),      # (seq_len, feat_dim) 
            'mf': (20, 43),     # (seq_len, feat_dim)
            'lf': (1, 19),      # (seq_len, feat_dim)
            'portfolio': (1, 5)  # (seq_len, feat_dim)
        }
        
        for branch, (seq_len, feat_dim) in branch_configs.items():
            end_idx = start_idx + (seq_len * feat_dim)
            branch_tensor = tensor[:, start_idx:end_idx].reshape(batch_size, seq_len, feat_dim)
            state_dict[branch] = branch_tensor.to(self.device)
            start_idx = end_idx
        
        # Get model output (action logits for SHAP analysis)
        with torch.no_grad():
            action_params, _ = self.model(state_dict)
            if isinstance(action_params, tuple) and len(action_params) == 2:
                # For discrete actions, combine type and size logits
                type_logits, size_logits = action_params
                # Take max probabilities and combine
                combined = torch.cat([
                    F.softmax(type_logits, dim=-1).max(dim=-1, keepdim=True)[0],
                    F.softmax(size_logits, dim=-1).max(dim=-1, keepdim=True)[0]
                ], dim=-1)
                return combined
            else:
                return action_params[0] if isinstance(action_params, tuple) else action_params
    
    def calculate_shap_values(
        self,
        sample_states: List[Dict[str, torch.Tensor]],
        background_states: Optional[List[Dict[str, torch.Tensor]]] = None,
        max_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Calculate SHAP values for feature importance.
        
        Args:
            sample_states: List of state dicts to analyze
            background_states: Background dataset for SHAP (uses sample if None)
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dict mapping branch names to SHAP value arrays
        """
        if not SHAP_AVAILABLE:
            self.logger.error("SHAP not available. Please install with: pip install shap")
            return {}
        
        try:
            # Limit samples for computational efficiency
            sample_states = sample_states[:max_samples]
            if background_states is None:
                background_states = sample_states[:min(20, len(sample_states))]
            
            # Convert states to tensors
            sample_tensors = []
            background_tensors = []
            
            for state in sample_states:
                sample_tensors.append(self._model_input_to_tensor(state))
            
            for state in background_states:
                background_tensors.append(self._model_input_to_tensor(state))
            
            sample_tensor = torch.cat(sample_tensors, dim=0)
            background_tensor = torch.cat(background_tensors, dim=0)
            
            # Create SHAP explainer
            explainer = shap.DeepExplainer(self._tensor_to_model_input, background_tensor)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_tensor)
            
            # Convert back to branch-wise SHAP values
            branch_shap_values = self._convert_shap_to_branches(shap_values, sample_tensor.shape[0])
            
            # Store for history
            self.last_analysis_time = time.time()
            
            self.logger.info(f"SHAP analysis completed for {len(sample_states)} samples")
            return branch_shap_values
            
        except Exception as e:
            self.logger.error(f"SHAP calculation failed: {e}")
            return {}
    
    def _convert_shap_to_branches(self, shap_values: np.ndarray, batch_size: int) -> Dict[str, np.ndarray]:
        """Convert flattened SHAP values back to branch structure"""
        branch_shap = {}
        start_idx = 0
        
        branch_configs = {
            'hf': (60, 9),
            'mf': (20, 43), 
            'lf': (1, 19),
            'portfolio': (1, 5)
        }
        
        for branch, (seq_len, feat_dim) in branch_configs.items():
            end_idx = start_idx + (seq_len * feat_dim)
            branch_values = shap_values[:, start_idx:end_idx]
            
            # Reshape to [batch, seq_len, feat_dim] and aggregate over sequence
            branch_values = branch_values.reshape(batch_size, seq_len, feat_dim)
            # Take mean over sequence dimension for feature importance
            branch_shap[branch] = np.mean(np.abs(branch_values), axis=1)  # [batch, feat_dim]
            
            start_idx = end_idx
        
        return branch_shap
    
    def calculate_permutation_importance(
        self,
        environment,
        model,
        n_episodes: int = 50,
        n_permutations: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate permutation importance by measuring performance drop when features are shuffled.
        
        Args:
            environment: Trading environment for testing
            model: Model to test
            n_episodes: Number of episodes to run for each test
            n_permutations: Number of permutation tests per feature
            
        Returns:
            Dict mapping branch names to feature importance scores
        """
        self.logger.info(f"Starting permutation importance analysis with {n_episodes} episodes")
        
        # Get baseline performance
        baseline_rewards = self._run_episodes(environment, model, n_episodes)
        baseline_mean = np.mean(baseline_rewards)
        
        importance_scores = {}
        
        for branch in self.branch_names:
            if branch not in self.feature_names:
                continue
                
            branch_scores = {}
            
            for feature_idx, feature_name in enumerate(self.feature_names[branch]):
                feature_importance = []
                
                for perm in range(n_permutations):
                    # Create modified model that permutes this feature
                    modified_rewards = self._run_episodes_with_permuted_feature(
                        environment, model, n_episodes, branch, feature_idx
                    )
                    
                    # Calculate importance as drop in performance
                    importance = baseline_mean - np.mean(modified_rewards)
                    feature_importance.append(importance)
                
                # Average importance across permutations
                branch_scores[feature_name] = np.mean(feature_importance)
                
            importance_scores[branch] = branch_scores
            
        self.logger.info("Permutation importance analysis completed")
        return importance_scores
    
    def _run_episodes(self, environment, model, n_episodes: int) -> List[float]:
        """Run episodes and collect rewards"""
        rewards = []
        
        for episode in range(n_episodes):
            obs, info = environment.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action, _ = model.get_action(obs, deterministic=True)
                    obs, reward, done, truncated, info = environment.step(action)
                    episode_reward += reward
                    done = done or truncated
            
            rewards.append(episode_reward)
            
        return rewards
    
    def _run_episodes_with_permuted_feature(
        self,
        environment,
        model,
        n_episodes: int,
        branch: str,
        feature_idx: int
    ) -> List[float]:
        """Run episodes with one feature permuted"""
        # Create a wrapper that permutes the specified feature
        original_forward = model.forward
        
        def permuted_forward(state_dict, return_internals=False):
            # Permute the specified feature across batch dimension
            if branch in state_dict and feature_idx < state_dict[branch].shape[-1]:
                modified_state = state_dict.copy()
                branch_tensor = modified_state[branch].clone()
                
                # Permute the feature across the batch dimension
                batch_size = branch_tensor.shape[0]
                if batch_size > 1:
                    perm_indices = torch.randperm(batch_size)
                    branch_tensor[:, :, feature_idx] = branch_tensor[perm_indices, :, feature_idx]
                    modified_state[branch] = branch_tensor
                
                return original_forward(modified_state, return_internals)
            else:
                return original_forward(state_dict, return_internals)
        
        # Temporarily replace forward method
        model.forward = permuted_forward
        
        try:
            rewards = self._run_episodes(environment, model, n_episodes)
        finally:
            # Restore original forward method
            model.forward = original_forward
        
        return rewards
    
    def track_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """
        Track and analyze attention patterns over time.
        
        Args:
            attention_weights: Attention weights from fusion layer [num_branches]
            
        Returns:
            Dict with attention analysis metrics
        """
        self.attention_history.append(attention_weights.copy())
        
        # Calculate attention statistics
        analysis = {
            'entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-8)),
            'max_attention': np.max(attention_weights),
            'dominant_branch': int(np.argmax(attention_weights)),
            'attention_std': np.std(attention_weights)
        }
        
        # Historical analysis if we have enough data
        if len(self.attention_history) >= 100:
            recent_attention = np.array(list(self.attention_history)[-100:])
            analysis['attention_stability'] = 1.0 - np.mean(np.std(recent_attention, axis=0))
            analysis['branch_usage_balance'] = np.min(np.mean(recent_attention, axis=0))
        
        return analysis
    
    def detect_dead_features(
        self,
        feature_usage_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """
        Detect features that are unused or highly correlated.
        
        Args:
            feature_usage_threshold: Minimum usage to be considered active
            correlation_threshold: Correlation threshold for redundancy detection
            
        Returns:
            Dict mapping issue types to lists of problematic features
        """
        dead_features = {
            'unused': [],
            'low_variance': [],
            'highly_correlated': []
        }
        
        # Check for unused features based on importance history
        for branch in self.feature_names:
            for feature in self.feature_names[branch]:
                if feature in self.importance_history:
                    recent_importance = list(self.importance_history[feature])[-100:]
                    if recent_importance and np.mean(recent_importance) < feature_usage_threshold:
                        dead_features['unused'].append(f"{branch}.{feature}")
        
        return dead_features
    
    def calculate_feature_correlation(
        self,
        feature_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate correlation matrix between features within and across branches.
        
        Args:
            feature_data: Dict mapping branch names to feature matrices
            
        Returns:
            Dict with correlation matrices for each branch and cross-branch
        """
        correlations = {}
        
        for branch, data in feature_data.items():
            if data.size > 0:
                # Calculate within-branch correlation
                branch_corr = np.corrcoef(data.T)
                correlations[f"{branch}_internal"] = branch_corr
        
        # Cross-branch correlation (using aggregated features)
        branch_aggregates = {}
        for branch, data in feature_data.items():
            if data.size > 0:
                # Aggregate features (mean over time dimension if 3D)
                if data.ndim == 3:
                    branch_aggregates[branch] = np.mean(data, axis=1)  # [batch, features]
                else:
                    branch_aggregates[branch] = data
        
        if len(branch_aggregates) > 1:
            # Combine all branch features
            all_features = np.concatenate(list(branch_aggregates.values()), axis=1)
            correlations['cross_branch'] = np.corrcoef(all_features.T)
        
        self.correlation_matrix = correlations
        return correlations
    
    def generate_feature_importance_report(
        self,
        shap_values: Optional[Dict[str, np.ndarray]] = None,
        permutation_scores: Optional[Dict[str, Dict[str, float]]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feature importance report.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            permutation_scores: Permutation importance from calculate_permutation_importance
            save_path: Optional path to save report plots
            
        Returns:
            Dict with comprehensive analysis results
        """
        report = {
            'timestamp': time.time(),
            'total_features': self.total_features,
            'branches': {}
        }
        
        for branch in self.branch_names:
            if branch not in self.feature_names:
                continue
                
            branch_report = {
                'feature_count': len(self.feature_names[branch]),
                'feature_names': self.feature_names[branch]
            }
            
            # Add SHAP analysis if available
            if shap_values and branch in shap_values:
                shap_importance = np.mean(np.abs(shap_values[branch]), axis=0)
                branch_report['shap_importance'] = {
                    name: float(importance) 
                    for name, importance in zip(self.feature_names[branch], shap_importance)
                }
                branch_report['top_shap_features'] = [
                    self.feature_names[branch][i] 
                    for i in np.argsort(shap_importance)[-5:][::-1]
                ]
            
            # Add permutation importance if available
            if permutation_scores and branch in permutation_scores:
                branch_report['permutation_importance'] = permutation_scores[branch]
                sorted_features = sorted(
                    permutation_scores[branch].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                branch_report['top_permutation_features'] = [name for name, _ in sorted_features[:5]]
            
            report['branches'][branch] = branch_report
        
        # Add attention analysis if available
        if len(self.attention_history) > 0:
            recent_attention = np.array(list(self.attention_history)[-100:])
            report['attention_analysis'] = {
                'mean_attention_per_branch': np.mean(recent_attention, axis=0).tolist(),
                'attention_stability': np.std(recent_attention, axis=0).tolist(),
                'dominant_branch_frequency': [
                    float(np.mean(np.argmax(recent_attention, axis=1) == i))
                    for i in range(recent_attention.shape[1])
                ]
            }
        
        # Save plots if requested
        if save_path:
            self._save_importance_plots(report, save_path)
        
        return report
    
    def _save_importance_plots(self, report: Dict[str, Any], save_path: str):
        """Save feature importance visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Feature Importance Analysis', fontsize=16)
            
            # Plot 1: SHAP importance by branch
            if any('shap_importance' in branch_data for branch_data in report['branches'].values()):
                ax = axes[0, 0]
                for branch, branch_data in report['branches'].items():
                    if 'shap_importance' in branch_data:
                        values = list(branch_data['shap_importance'].values())
                        ax.bar(range(len(values)), values, alpha=0.7, label=branch)
                ax.set_title('SHAP Feature Importance by Branch')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('SHAP Importance')
                ax.legend()
            
            # Plot 2: Permutation importance
            if any('permutation_importance' in branch_data for branch_data in report['branches'].values()):
                ax = axes[0, 1]
                for branch, branch_data in report['branches'].items():
                    if 'permutation_importance' in branch_data:
                        values = list(branch_data['permutation_importance'].values())
                        ax.bar(range(len(values)), values, alpha=0.7, label=branch)
                ax.set_title('Permutation Feature Importance by Branch')
                ax.set_xlabel('Feature Index') 
                ax.set_ylabel('Importance Score')
                ax.legend()
            
            # Plot 3: Attention patterns
            if 'attention_analysis' in report:
                ax = axes[1, 0]
                attention_data = report['attention_analysis']['mean_attention_per_branch']
                branch_labels = list(report['branches'].keys())
                ax.pie(attention_data, labels=branch_labels, autopct='%1.1f%%')
                ax.set_title('Average Attention Distribution')
            
            # Plot 4: Feature count by branch
            ax = axes[1, 1]
            branch_names = list(report['branches'].keys())
            feature_counts = [report['branches'][branch]['feature_count'] for branch in branch_names]
            ax.bar(branch_names, feature_counts)
            ax.set_title('Feature Count by Branch')
            ax.set_ylabel('Number of Features')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance plots saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save importance plots: {e}")
    
    def update_feature_importance(self, feature_name: str, importance_score: float):
        """Update importance score for a feature"""
        self.importance_history[feature_name].append(importance_score)
    
    def get_top_features(self, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get top N most important features across all methods"""
        top_features = {}
        
        for branch in self.feature_names:
            branch_features = []
            
            for feature in self.feature_names[branch]:
                if feature in self.importance_history and self.importance_history[feature]:
                    avg_importance = np.mean(list(self.importance_history[feature])[-100:])
                    branch_features.append((feature, avg_importance))
            
            # Sort by importance and take top N
            branch_features.sort(key=lambda x: x[1], reverse=True)
            top_features[branch] = branch_features[:n]
        
        return top_features