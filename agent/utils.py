# training/utils.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import os


class ReplayBuffer:
    """
    Replay buffer for PPO training, storing transitions and episode information.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.clear()

    # In agent/utils.py - ReplayBuffer class
    def add(self, state: Dict[str, torch.Tensor], action: torch.Tensor,
            reward: float, next_state: Dict[str, torch.Tensor],
            done: bool, action_info: Dict[str, torch.Tensor]) -> None:
        """
        Add a transition to the buffer with improved log probability handling.
        """
        # Convert scalar to tensor
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([done], dtype=torch.bool, device=self.device)

        state_tensors = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                # Add batch dimension if missing
                if len(v.shape) == 1 and k == 'static_features':
                    v = v[np.newaxis, :]
                elif len(v.shape) == 2 and k in ['hf_features', 'mf_features', 'lf_features']:
                    v = v[np.newaxis, :, :]
                state_tensors[k] = torch.FloatTensor(v).to(self.device)
            elif isinstance(v, torch.Tensor):
                state_tensors[k] = v.to(self.device)

        # Initialize state dictionary on first call
        if not self.states:
            self.states = {k: [] for k in state_tensors.keys()}

        # Add to buffers
        for k, v in state_tensors.items():
            self.states[k].append(v)

        # Add to buffers
        for k, v in state.items():
            self.states[k].append(v)

        self.actions.append(action)
        self.rewards.append(reward_tensor)
        self.dones.append(done_tensor)

        # Extract value
        if 'value' in action_info:
            self.values.append(action_info['value'])

        # Extract log probability directly if available
        if 'log_prob' in action_info:
            self.log_probs.append(action_info['log_prob'])
        else:
            # Fallback calculations if log_prob not provided directly
            if 'mean' in action_info and 'log_std' in action_info:
                # For continuous actions
                mean = action_info['mean']
                log_std = action_info['log_std']
                std = torch.exp(log_std)

                # Compute log_prob of the action
                normal_dist = torch.distributions.Normal(mean, std)
                # Assuming actions are squashed with tanh
                action_unsquashed = torch.atanh(torch.clamp(action, -0.999, 0.999))
                log_prob = normal_dist.log_prob(action_unsquashed).sum(1, keepdim=True)
                self.log_probs.append(log_prob)
            elif 'logits' in action_info:
                # For discrete actions
                logits = action_info['logits']
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action.squeeze()).unsqueeze(1)
                self.log_probs.append(log_prob)
            elif len(action.shape) > 1 and action.shape[1] == 2:
                # For tuple actions, warn but create dummy
                self.logger.warning("Creating dummy log_prob for tuple action without distribution parameters")
                dummy_log_prob = torch.zeros((action.shape[0], 1), device=self.device)
                self.log_probs.append(dummy_log_prob)

    def prepare_data(self) -> None:
        """
        Prepare data for training by converting lists to tensors.
        """
        # States - dictionary of tensors
        self.states = {k: torch.cat(v, dim=0) for k, v in self.states.items()}

        # Actions, rewards, etc.
        self.actions = torch.cat(self.actions, dim=0)
        self.rewards = torch.cat(self.rewards, dim=0)
        self.dones = torch.cat(self.dones, dim=0)

        # Add safety checks for potentially empty lists
        if self.values:
            self.values = torch.cat(self.values, dim=0)
        else:
            # Create empty values tensor with matching batch size
            self.values = torch.zeros_like(self.rewards)

        if self.log_probs:
            self.log_probs = torch.cat(self.log_probs, dim=0)
        else:
            # Create empty log_probs tensor with matching batch size
            self.log_probs = torch.zeros_like(self.rewards)

    def get_size(self) -> int:
        """Get current size of buffer."""
        return len(self.rewards)

    def clear(self) -> None:
        """Clear the buffer."""
        self.states = {}
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        # Will be computed during training
        self.advantages = None
        self.returns = None

    def is_ready(self) -> bool:
        """Check if buffer is ready for training."""
        return self.get_size() > 0


def normalize_state_dict(state_dict: Dict[str, torch.Tensor],
                         running_stats: Dict[str, Dict[str, torch.Tensor]] = None,
                         update_stats: bool = False) -> Tuple[
    Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Normalize the state dictionary. If running_stats is provided, use it for normalization.
    If update_stats is True, update the running statistics.

    Args:
        state_dict: Dictionary of state tensors to normalize
        running_stats: Dictionary of running statistics (mean, std) for each component
        update_stats: Whether to update the running statistics

    Returns:
        Tuple of (normalized state dict, updated running stats)
    """
    if running_stats is None:
        running_stats = {k: {'mean': None, 'std': None} for k in state_dict.keys()}

    normalized_state = {}

    for k, v in state_dict.items():
        if update_stats or running_stats[k]['mean'] is None:
            # Update or initialize running stats
            mean = v.mean(dim=0, keepdim=True)
            std = v.std(dim=0, keepdim=True) + 1e-8

            if running_stats[k]['mean'] is None:
                running_stats[k]['mean'] = mean
                running_stats[k]['std'] = std
            else:
                # Exponential moving average (momentum=0.99)
                running_stats[k]['mean'] = 0.99 * running_stats[k]['mean'] + 0.01 * mean
                running_stats[k]['std'] = 0.99 * running_stats[k]['std'] + 0.01 * std

        # Normalize using running stats
        normalized_state[k] = (v - running_stats[k]['mean']) / running_stats[k]['std']

    return normalized_state, running_stats


def preprocess_state_to_dict(state: np.ndarray, model_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Preprocess a raw state vector from the environment into a structured dictionary
    that can be fed into the multi-branch transformer model.

    Args:
        state: Raw state vector from environment
        model_config: Model configuration with feature dimensions

    Returns:
        Dictionary with state components as tensors
    """
    # Extract feature dimensions from model config
    hf_feat_dim = model_config.get('hf_feat_dim', 20)
    mf_feat_dim = model_config.get('mf_feat_dim', 15)
    lf_feat_dim = model_config.get('lf_feat_dim', 10)
    static_feat_dim = model_config.get('static_feat_dim', 15)

    hf_seq_len = model_config.get('hf_seq_len', 60)
    mf_seq_len = model_config.get('mf_seq_len', 30)
    lf_seq_len = model_config.get('lf_seq_len', 30)

    # Convert state array to tensor
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Create state dictionary with expected shapes
    # This is a simplified version that assumes the state vector is already structured
    # in a specific way. In practice, you'd have custom logic to extract features
    # from your environment's state.

    # For demonstration, we'll just split the state into segments
    state_dict = {}

    total_hf_size = hf_seq_len * hf_feat_dim
    total_mf_size = mf_seq_len * mf_feat_dim
    total_lf_size = lf_seq_len * lf_feat_dim

    # Check if state is large enough
    if len(state_tensor) >= total_hf_size + total_mf_size + total_lf_size + static_feat_dim:
        # Extract features (this is simplified and would need to be adapted to actual state structure)
        hf_flat = state_tensor[:total_hf_size]
        mf_flat = state_tensor[total_hf_size:total_hf_size + total_mf_size]
        lf_flat = state_tensor[total_hf_size + total_mf_size:total_hf_size + total_mf_size + total_lf_size]
        static_features = state_tensor[total_hf_size + total_mf_size + total_lf_size:
                                       total_hf_size + total_mf_size + total_lf_size + static_feat_dim]

        # Reshape sequences
        hf_features = hf_flat.reshape(1, hf_seq_len, hf_feat_dim)  # Add batch dimension
        mf_features = mf_flat.reshape(1, mf_seq_len, mf_feat_dim)
        lf_features = lf_flat.reshape(1, lf_seq_len, lf_feat_dim)
        static_features = static_features.reshape(1, static_feat_dim)
    else:
        # If state is not large enough, create dummy tensors with zeros
        # This is just a fallback - in practice, ensure your state vector is properly structured
        hf_features = torch.zeros((1, hf_seq_len, hf_feat_dim))
        mf_features = torch.zeros((1, mf_seq_len, mf_feat_dim))
        lf_features = torch.zeros((1, lf_seq_len, lf_feat_dim))
        static_features = torch.zeros((1, static_feat_dim))

    # Build state dict
    state_dict = {
        'hf_features': hf_features,
        'mf_features': mf_features,
        'lf_features': lf_features,
        'static_features': static_features
    }

    return state_dict