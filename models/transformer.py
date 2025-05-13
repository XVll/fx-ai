# models/transformer.py (updated for Hydra)
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from omegaconf import DictConfig

from models.layers import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerEncoder,
    AttentionFusion
)
from models.networks import ActorNetwork, CriticNetwork


class MultiBranchTransformer(nn.Module):
    """
    Multi-Branch Transformer with Attention Fusion for financial trading.

    Processes high, medium, and low-frequency features through separate transformer
    branches, plus a static branch for position and S/R features, then fuses them
    with an attention mechanism.
    """

    def __init__(
            self,
            # Feature dimensions
            hf_seq_len: int = 60,
            hf_feat_dim: int = 20,
            mf_seq_len: int = 30,
            mf_feat_dim: int = 15,
            lf_seq_len: int = 30,
            lf_feat_dim: int = 10,
            static_feat_dim: int = 15,

            # Model dimensions
            d_model: int = 64,
            d_fused: int = 256,

            # Transformer parameters
            hf_layers: int = 2,
            mf_layers: int = 2,
            lf_layers: int = 2,
            hf_heads: int = 4,
            mf_heads: int = 4,
            lf_heads: int = 4,

            # Output parameters
            action_dim: int = 1,
            continuous_action: bool = True,

            # Other parameters
            dropout: float = 0.1,
            device: Union[str, torch.device] = None,

            # For Hydra compatibility
            _target_: Optional[str] = None,  # Hydra instantiation target
            **kwargs  # Capture any additional Hydra parameters
    ):
        super().__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.continuous_action = continuous_action
        self.action_dim = action_dim

        # Store dimensions for reference
        self.hf_seq_len = hf_seq_len
        self.mf_seq_len = mf_seq_len
        self.lf_seq_len = lf_seq_len
        self.static_feat_dim = static_feat_dim

        # Feature dimensions
        self.hf_feat_dim = hf_feat_dim
        self.mf_feat_dim = mf_feat_dim
        self.lf_feat_dim = lf_feat_dim

        # HF Branch (High Frequency)
        self.hf_proj = nn.Linear(hf_feat_dim, d_model)
        self.hf_pos_enc = PositionalEncoding(d_model, dropout, max_len=hf_seq_len)
        encoder_layer_hf = TransformerEncoderLayer(d_model, hf_heads, dim_feedforward=2 * d_model, dropout=dropout)
        self.hf_encoder = TransformerEncoder(encoder_layer_hf, hf_layers)

        # MF Branch (Medium Frequency)
        self.mf_proj = nn.Linear(mf_feat_dim, d_model)
        self.mf_pos_enc = PositionalEncoding(d_model, dropout, max_len=mf_seq_len)
        encoder_layer_mf = TransformerEncoderLayer(d_model, mf_heads, dim_feedforward=2 * d_model, dropout=dropout)
        self.mf_encoder = TransformerEncoder(encoder_layer_mf, mf_layers)

        # LF Branch (Low Frequency)
        self.lf_proj = nn.Linear(lf_feat_dim, d_model)
        self.lf_pos_enc = PositionalEncoding(d_model, dropout, max_len=lf_seq_len)
        encoder_layer_lf = TransformerEncoderLayer(d_model, lf_heads, dim_feedforward=2 * d_model, dropout=dropout)
        self.lf_encoder = TransformerEncoder(encoder_layer_lf, lf_layers)

        # Static Branch (Position, S/R, etc.)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Fusion Layer
        self.fusion = AttentionFusion(d_model, 4, d_fused, 4)

        # Actor-Critic Networks
        self.actor = ActorNetwork(d_fused, action_dim, continuous_action)
        self.critic = CriticNetwork(d_fused)

        self.to(self.device)

    # Rest of the class remains the same
    # ...

    def forward(
            self,
            state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            state_dict: Dictionary containing:
                - hf_features: (batch_size, hf_seq_len, hf_feat_dim)
                - mf_features: (batch_size, mf_seq_len, mf_feat_dim)
                - lf_features: (batch_size, lf_seq_len, lf_feat_dim)
                - static_features: (batch_size, static_feat_dim)

        Returns:
            Tuple of (action_distribution_params, value)
            - For continuous actions: (mean, log_std), value
            - For discrete actions: (logits), value
        """
        # Extract features from dict, ensuring they're on the right device
        hf_features = state_dict['hf_features'].to(self.device)
        mf_features = state_dict['mf_features'].to(self.device)
        lf_features = state_dict['lf_features'].to(self.device)
        static_features = state_dict['static_features'].to(self.device)

        # Process HF Branch
        # Shape: (batch_size, hf_seq_len, d_model)
        hf_x = self.hf_proj(hf_features)
        hf_x = self.hf_pos_enc(hf_x)
        # Shape: (batch_size, hf_seq_len, d_model)
        hf_x = self.hf_encoder(hf_x)
        # Take last timestep's representation
        # Shape: (batch_size, d_model)
        hf_rep = hf_x[:, -1, :]

        # Process MF Branch
        # Shape: (batch_size, mf_seq_len, d_model)
        mf_x = self.mf_proj(mf_features)
        mf_x = self.mf_pos_enc(mf_x)
        # Shape: (batch_size, mf_seq_len, d_model)
        mf_x = self.mf_encoder(mf_x)
        # Take last timestep's representation
        # Shape: (batch_size, d_model)
        mf_rep = mf_x[:, -1, :]

        # Process LF Branch
        # Shape: (batch_size, lf_seq_len, d_model)
        lf_x = self.lf_proj(lf_features)
        lf_x = self.lf_pos_enc(lf_x)
        # Shape: (batch_size, lf_seq_len, d_model)
        lf_x = self.lf_encoder(lf_x)
        # Take last timestep's representation
        # Shape: (batch_size, d_model)
        lf_rep = lf_x[:, -1, :]

        # Process Static Branch
        # Shape: (batch_size, d_model)
        static_rep = self.static_encoder(static_features)

        # Fusion via attention
        # Prepare all branches for fusion
        # Shape: (batch_size, 4, d_model)
        features_to_fuse = torch.stack([hf_rep, mf_rep, lf_rep, static_rep], dim=1)

        # Fuse all branches
        # Shape: (batch_size, d_fused)
        fused = self.fusion(features_to_fuse)

        # Get policy outputs and value estimate
        action_params = self.actor(fused)
        value = self.critic(fused)

        return action_params, value

    def get_action(
            self,
            state_dict: Dict[str, torch.Tensor],
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an action from the policy.

        Args:
            state_dict: Dictionary of state features
            deterministic: If True, returns the mean action instead of sampling

        Returns:
            action: The selected action
            action_info: Dictionary with action distribution parameters
        """
        with torch.no_grad():
            action_params, value = self.forward(state_dict)

            if self.continuous_action:
                # For continuous action space (e.g., PPO with Gaussian policy)
                mean, log_std = action_params

                if deterministic:
                    # During evaluation, use the mean action
                    action = torch.tanh(mean)
                else:
                    # During training, sample from the distribution
                    std = torch.exp(log_std)
                    normal = torch.distributions.Normal(mean, std)
                    # Reparameterization trick
                    x_t = normal.rsample()
                    # Squash using tanh to bound between -1 and 1
                    action = torch.tanh(x_t)

                action_info = {
                    'mean': mean,
                    'log_std': log_std,
                    'value': value
                }
            else:
                # For discrete action space
                logits = action_params

                if deterministic:
                    # During evaluation, select the most probable action
                    action = torch.argmax(logits, dim=-1)
                else:
                    # During training, sample from the categorical distribution
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()

                action_info = {
                    'logits': logits,
                    'value': value
                }

            return action, action_info