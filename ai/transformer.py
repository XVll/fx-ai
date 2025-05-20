import logging

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union

from config.config import ModelConfig
from ai.layers import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerEncoder,
    AttentionFusion
)


class MultiBranchTransformer(nn.Module):
    """
    Multi-Branch Transformer with Attention Fusion for financial trading.

    Processes high, medium, and low-frequency features through separate transformer
    branches, plus a static branch for position and S/R features, then fuses them
    with an attention mechanism.
    """

    def __init__(
            self,
            hf_seq_len: int,
            hf_feat_dim: int,
            mf_seq_len: int,
            mf_feat_dim: int,
            lf_seq_len: int,
            lf_feat_dim: int,
            static_feat_dim,
            portfolio_feat_dim,
            portfolio_seq_len: int,

            hf_layers: int,
            mf_layers: int,
            lf_layers: int,
            portfolio_layers: int,
            hf_heads: int,
            mf_heads: int,
            lf_heads: int,
            portfolio_heads: int,

            action_dim: list,
            continuous_action: bool,

            d_model: int,
            d_fused: int,

            dropout: float = 0.1,
            device: Union[str, torch.device] = None,

            # For Hydra compatibility
            _target_: Optional[str] = None,  # Hydra instantiation target
            logger: Optional[object] = None  # Logger for debugging
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Store continuous/discrete action selection
        self.continuous_action = continuous_action

        # Handle action dimensions (support both single int and tuple/list)
        if continuous_action:
            if isinstance(action_dim, (list, tuple)):
                # For continuous actions, use the first dimension
                self.action_dim = action_dim[0]
            else:
                self.action_dim = action_dim
        else:
            # For discrete actions, we need to handle tuples
            if isinstance(action_dim, (list, tuple)):
                self.action_types = action_dim[0]
                self.action_sizes = action_dim[1]
            else:
                self.action_types = action_dim
                self.action_sizes = None

        # Store dimensions for reference
        self.hf_seq_len = hf_seq_len
        self.mf_seq_len = mf_seq_len
        self.lf_seq_len = lf_seq_len
        self.portfolio_seq_len = portfolio_seq_len
        self.static_feat_dim = static_feat_dim

        # Feature dimensions
        self.hf_feat_dim = hf_feat_dim
        self.mf_feat_dim = mf_feat_dim
        self.lf_feat_dim = lf_feat_dim
        self.portfolio_feat_dim = portfolio_feat_dim

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

        # Portfolio Branch (Static features)
        self.portfolio_proj = nn.Linear(portfolio_feat_dim, d_model)
        self.portfolio_pos_enc = PositionalEncoding(d_model, dropout, max_len=portfolio_seq_len)
        encoder_layer_portfolio = TransformerEncoderLayer(d_model, portfolio_heads, dim_feedforward=2 * d_model,
                                                          dropout=dropout)
        self.portfolio_encoder = TransformerEncoder(encoder_layer_portfolio, portfolio_layers)

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
        self.fusion = AttentionFusion(d_model, 5, d_fused, 4)

        # Output layers - different depending on continuous vs. discrete
        if continuous_action:
            # For continuous actions
            self.actor_mean = nn.Linear(d_fused, self.action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))
        else:
            # For discrete tuple actions
            if self.action_sizes is not None:
                # Separate outputs for action type and size
                self.action_type_head = nn.Linear(d_fused, self.action_types)
                self.action_size_head = nn.Linear(d_fused, self.action_sizes)
            else:
                # Single discrete output
                self.action_head = nn.Linear(d_fused, self.action_types)

        # Critic Network (value function)
        self.critic = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2),
            nn.LayerNorm(d_fused // 2),
            nn.GELU(),
            nn.Linear(d_fused // 2, 1)
        )

        self.to(self.device)

    def forward(
            self,
            state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...]], torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            state_dict: Dictionary containing:
                - hf: (batch_size, hf_seq_len, hf_feat_dim)
                - mf: (batch_size, mf_seq_len, mf_feat_dim)
                - lf: (batch_size, lf_seq_len, lf_feat_dim)
                - static: (batch_size, static_feat_dim)
                - portfolio: (batch_size, portfolio_seq_len, portfolio_feat_dim)

        Returns:
            Tuple of (action_distribution_params, value)
            - For continuous actions: (mean, log_std)
            - For discrete tuple actions: (action_type_logits, action_size_logits)
            - For discrete single actions: (logits)
            - value: Value estimate tensor
        """

        # Todo Delete me for debugs
        for key, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                self.logger.warning(f"NaN detected in input tensor '{key}': {nan_count} values")

        # Extract features from dict, ensuring they're on the right device
        hf_features = state_dict['hf'].to(self.device)
        mf_features = state_dict['mf'].to(self.device)
        lf_features = state_dict['lf'].to(self.device)
        static_features = state_dict['static'].to(self.device)
        portfolio_features = state_dict['portfolio'].to(self.device)

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

        # Process Portfolio Branch
        # Shape: (batch_size, portfolio_seq_len, d_model)
        portfolio_x = self.portfolio_proj(portfolio_features)
        portfolio_x = self.portfolio_pos_enc(portfolio_x)
        # Shape: (batch_size, portfolio_seq_len, d_model)
        portfolio_x = self.portfolio_encoder(portfolio_x)
        # Take last timestep's representation
        # Shape: (batch_size, d_model)
        portfolio_rep = portfolio_x[:, -1, :]

        # Process Static Branch
        # Shape: (batch_size, d_model)
        static_rep = self.static_encoder(static_features)

        # Fusion via attention
        # Prepares all branches for fusion
        # Shape: (batch_size, 4, d_model)
        features_to_fuse = torch.stack([hf_rep, mf_rep, lf_rep, portfolio_rep, static_rep], dim=1)

        # Fuse all-branches
        # Shape: (batch_size, d_fused)
        fused = self.fusion(features_to_fuse)

        # Output action parameters based on an action space type
        if self.continuous_action:
            # For continuous action space, output mean and log_std
            mean = self.actor_mean(fused)
            # Expand log_std to match the batch size
            batch_size = fused.size(0)
            log_std = self.actor_log_std.expand(batch_size, -1)
            action_params = (mean, log_std)
        else:
            # For discrete action space
            if self.action_sizes is not None:
                # Output separate logits for action type and size
                action_type_logits = self.action_type_head(fused)
                action_size_logits = self.action_size_head(fused)
                action_params = (action_type_logits, action_size_logits)
            else:
                # Single discrete output
                logits = self.action_head(fused)
                action_params = (logits,)

        # Get value estimate
        value = self.critic(fused)

        return action_params, value

    # In ai/transformer.py - MultiBranchTransformer class
    def get_action(
            self,
            state_dict: Dict[str, torch.Tensor],
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an action from the policy with guaranteed log probability calculation.
        """
        with torch.no_grad():
            action_params, value = self.forward(state_dict)

            if self.continuous_action:
                mean, log_std = action_params
                std = torch.exp(log_std)
                normal = torch.distributions.Normal(mean, std)

                if deterministic:
                    action = torch.tanh(mean)
                    # Still calculate log_prob
                    x_t = mean
                else:
                    x_t = normal.rsample()
                    action = torch.tanh(x_t)

                # Calculate log_prob including tanh squashing correction
                log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(1, keepdim=True)

                action_info = {
                    'mean': mean,
                    'log_std': log_std,
                    'value': value,
                    'log_prob': log_prob  # Always include log_prob
                }
            else:
                # For discrete actions (tuple in your case)
                if len(action_params) == 2:
                    action_type_logits, action_size_logits = action_params

                    if deterministic:
                        action_type = torch.argmax(action_type_logits, dim=-1)
                        action_size = torch.argmax(action_size_logits, dim=-1)
                    else:
                        action_type_dist = torch.distributions.Categorical(logits=action_type_logits)
                        action_size_dist = torch.distributions.Categorical(logits=action_size_logits)
                        action_type = action_type_dist.sample()
                        action_size = action_size_dist.sample()

                    action = torch.stack([action_type, action_size], dim=-1)

                    # Calculate combined log probability
                    type_log_prob = torch.distributions.Categorical(logits=action_type_logits).log_prob(action_type)
                    size_log_prob = torch.distributions.Categorical(logits=action_size_logits).log_prob(action_size)
                    log_prob = (type_log_prob + size_log_prob).unsqueeze(1)

                    action_info = {
                        'action_type_logits': action_type_logits,
                        'action_size_logits': action_size_logits,
                        'value': value,
                        'log_prob': log_prob  # Always include log_prob
                    }
                else:
                    # For a single discrete action
                    logits = action_params[0]
                    dist = torch.distributions.Categorical(logits=logits)

                    if deterministic:
                        action = torch.argmax(logits, dim=-1)
                    else:
                        action = dist.sample()

                    log_prob = dist.log_prob(action).unsqueeze(1)

                    action_info = {
                        'logits': logits,
                        'value': value,
                        'log_prob': log_prob  # Always include log_prob
                    }

            return action, action_info
