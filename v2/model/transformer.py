import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union

from v2.model.layers import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerEncoder,
    AttentionFusion,
)
from v2.config import Config


class MultiBranchTransformer(nn.Module):
    """
    Multi-Branch Transformer with Attention Fusion for financial trading.

    Processes high, medium, and low-frequency features through separate transformer
    branches, plus portfolio features, then fuses them with an attention mechanism.
    """

    def __init__(
        self,
        model_config,  # Config.model
        device: Union[str, torch.device] = None,
        logger: Optional[object] = None,  # Logger for debugging
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

        # System only uses discrete actions
        self.continuous_action = False

        # Handle discrete action dimensions
        if isinstance(model_config.action_dim, (list, tuple)):
            self.action_types = int(model_config.action_dim[0])
            self.action_sizes = int(model_config.action_dim[1])
        else:
            self.action_types = model_config.action_dim
            self.action_sizes = None

        # Store model config and dimensions for reference
        self.model_config = model_config
        self.hf_seq_len = model_config.hf_seq_len
        self.mf_seq_len = model_config.mf_seq_len
        self.lf_seq_len = model_config.lf_seq_len
        self.portfolio_seq_len = model_config.portfolio_seq_len

        # Feature dimensions
        self.hf_feat_dim = model_config.hf_feat_dim
        self.mf_feat_dim = model_config.mf_feat_dim
        self.lf_feat_dim = model_config.lf_feat_dim
        self.portfolio_feat_dim = model_config.portfolio_feat_dim

        # HF Branch (High Frequency)
        self.hf_proj = nn.Linear(model_config.hf_feat_dim, model_config.d_model)
        self.hf_pos_enc = PositionalEncoding(
            model_config.d_model, model_config.dropout, max_len=model_config.hf_seq_len
        )
        encoder_layer_hf = TransformerEncoderLayer(
            model_config.d_model,
            model_config.hf_heads,
            dim_feedforward=2 * model_config.d_model,
            dropout=model_config.dropout,
        )
        self.hf_encoder = TransformerEncoder(encoder_layer_hf, model_config.hf_layers)

        # MF Branch (Medium Frequency)
        self.mf_proj = nn.Linear(model_config.mf_feat_dim, model_config.d_model)
        self.mf_pos_enc = PositionalEncoding(
            model_config.d_model, model_config.dropout, max_len=model_config.mf_seq_len
        )
        encoder_layer_mf = TransformerEncoderLayer(
            model_config.d_model,
            model_config.mf_heads,
            dim_feedforward=2 * model_config.d_model,
            dropout=model_config.dropout,
        )
        self.mf_encoder = TransformerEncoder(encoder_layer_mf, model_config.mf_layers)

        # LF Branch (Low Frequency)
        self.lf_proj = nn.Linear(model_config.lf_feat_dim, model_config.d_model)
        self.lf_pos_enc = PositionalEncoding(
            model_config.d_model, model_config.dropout, max_len=model_config.lf_seq_len
        )
        encoder_layer_lf = TransformerEncoderLayer(
            model_config.d_model,
            model_config.lf_heads,
            dim_feedforward=2 * model_config.d_model,
            dropout=model_config.dropout,
        )
        self.lf_encoder = TransformerEncoder(encoder_layer_lf, model_config.lf_layers)

        # Portfolio Branch (Static features)
        self.portfolio_proj = nn.Linear(
            model_config.portfolio_feat_dim, model_config.d_model
        )
        self.portfolio_pos_enc = PositionalEncoding(
            model_config.d_model,
            model_config.dropout,
            max_len=model_config.portfolio_seq_len,
        )
        encoder_layer_portfolio = TransformerEncoderLayer(
            model_config.d_model,
            model_config.portfolio_heads,
            dim_feedforward=2 * model_config.d_model,
            dropout=model_config.dropout,
        )
        self.portfolio_encoder = TransformerEncoder(
            encoder_layer_portfolio, model_config.portfolio_layers
        )

        # Cross-timeframe attention for pattern recognition
        # This allows HF features to attend to MF/LF patterns
        self.cross_timeframe_attention = nn.MultiheadAttention(
            embed_dim=model_config.d_model,
            num_heads=4,
            dropout=model_config.dropout,
            batch_first=True,
        )

        # Pattern extraction layer - identifies key points in sequences
        self.pattern_extractor = nn.Sequential(
            nn.Conv1d(
                model_config.d_model,
                model_config.d_model // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                model_config.d_model // 2,
                model_config.d_model // 4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # Extract most prominent pattern
        )

        # Fusion Layer - 5 inputs: HF, MF, LF, Portfolio, Cross-attention
        self.fusion = AttentionFusion(model_config.d_model, 5, model_config.d_fused, 4)

        # Output layers for discrete actions only
        if self.action_sizes is not None:
            # Separate outputs for action type and size
            self.action_type_head = nn.Linear(model_config.d_fused, self.action_types)
            self.action_size_head = nn.Linear(model_config.d_fused, self.action_sizes)
        else:
            # Single discrete output
            self.action_head = nn.Linear(model_config.d_fused, self.action_types)

        # Critic Network (value function)
        self.critic = nn.Sequential(
            nn.Linear(model_config.d_fused, model_config.d_fused // 2),
            nn.LayerNorm(model_config.d_fused // 2),
            nn.GELU(),
            nn.Linear(model_config.d_fused // 2, 1),
        )

        self.to(self.device)

    def forward(
        self, state_dict: Dict[str, torch.Tensor], return_internals: bool = False
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...]], torch.Tensor
    ]:
        """
        Forward pass through the model.

        Args:
            state_dict: Dictionary containing:
                - hf: (batch_size, hf_seq_len, hf_feat_dim)
                - mf: (batch_size, mf_seq_len, mf_feat_dim)
                - lf: (batch_size, lf_seq_len, lf_feat_dim)
                - portfolio: (batch_size, portfolio_seq_len, portfolio_feat_dim)

        Returns:
            Tuple of (action_distribution_params, value)
            - For continuous actions: (mean, log_std)
            - For discrete tuple actions: (action_type_logits, action_size_logits)
            - For discrete single actions: (logits)
            - value: Value estimate tensor
        """
        # Check for NaNs in input tensors
        for key, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                self.logger.warning(
                    f"NaN detected in input tensor '{key}': {nan_count} values"
                )

        # Extract features from dict, ensuring they're on the right device
        hf_features = state_dict["hf"].to(self.device)
        mf_features = state_dict["mf"].to(self.device)
        lf_features = state_dict["lf"].to(self.device)
        portfolio_features = state_dict["portfolio"].to(self.device)

        # HF features: should be [batch_size, hf_seq_len, hf_feat_dim]
        if hf_features.ndim == 2:  # [seq_len, feat_dim]
            hf_features = hf_features.unsqueeze(0)  # Add batch dim

        # MF features: should be [batch_size, mf_seq_len, mf_feat_dim]
        if mf_features.ndim == 2:  # [seq_len, feat_dim]
            mf_features = mf_features.unsqueeze(0)  # Add batch dim

        # LF features: should be [batch_size, lf_seq_len, lf_feat_dim]
        if lf_features.ndim == 2:  # [seq_len, feat_dim]
            lf_features = lf_features.unsqueeze(0)  # Add batch dim

        # Portfolio features: should be [batch_size, portfolio_seq_len, portfolio_feat_dim]
        if portfolio_features.ndim == 2:  # [seq_len, feat_dim]
            portfolio_features = portfolio_features.unsqueeze(0)  # Add batch dim

        # Add missing dimensions if needed - this makes the model more robust
        # HF features
        if hf_features.ndim < 3:
            self.logger.warning(
                f"hf_features has unexpected shape: {hf_features.shape}. Adding missing dimensions."
            )
            if hf_features.ndim == 1:  # [feat_dim]
                hf_features = hf_features.unsqueeze(0).unsqueeze(0)  # [1, 1, feat_dim]
            elif hf_features.ndim == 2:  # [seq_len, feat_dim]
                hf_features = hf_features.unsqueeze(0)  # [1, seq_len, feat_dim]

        # MF features
        if mf_features.ndim < 3:
            self.logger.warning(
                f"mf_features has unexpected shape: {mf_features.shape}. Adding missing dimensions."
            )
            if mf_features.ndim == 1:  # [feat_dim]
                mf_features = mf_features.unsqueeze(0).unsqueeze(0)  # [1, 1, feat_dim]
            elif mf_features.ndim == 2:  # [seq_len, feat_dim]
                mf_features = mf_features.unsqueeze(0)  # [1, seq_len, feat_dim]

        # LF features
        if lf_features.ndim < 3:
            self.logger.warning(
                f"lf_features has unexpected shape: {lf_features.shape}. Adding missing dimensions."
            )
            if lf_features.ndim == 1:  # [feat_dim]
                lf_features = lf_features.unsqueeze(0).unsqueeze(0)  # [1, 1, feat_dim]
            elif lf_features.ndim == 2:  # [seq_len, feat_dim]
                lf_features = lf_features.unsqueeze(0)  # [1, seq_len, feat_dim]

        # Portfolio features
        if portfolio_features.ndim < 3:
            self.logger.warning(
                f"portfolio_features has unexpected shape: {portfolio_features.shape}. Adding missing dimensions."
            )
            if portfolio_features.ndim == 1:  # [feat_dim]
                portfolio_features = portfolio_features.unsqueeze(0).unsqueeze(
                    0
                )  # [1, 1, feat_dim]
            elif portfolio_features.ndim == 2:  # [seq_len, feat_dim]
                portfolio_features = portfolio_features.unsqueeze(
                    0
                )  # [1, seq_len, feat_dim]

        # Process HF Branch
        # Shape: (batch_size, hf_seq_len, d_model)
        hf_x = self.hf_proj(hf_features)
        hf_x = self.hf_pos_enc(hf_x)
        # Shape: (batch_size, hf_seq_len, d_model)
        hf_x = self.hf_encoder(hf_x)
        # Use temporal attention pooling to preserve pattern information
        # Recent timesteps get more weight (exponential decay)
        seq_len = hf_x.size(1)
        time_weights = torch.exp(torch.linspace(-2, 0, seq_len, device=hf_x.device))
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        time_weights = time_weights / time_weights.sum(dim=1, keepdim=True)  # Normalize

        # Weighted temporal pooling
        hf_weighted = hf_x * time_weights
        hf_rep = hf_weighted.sum(dim=1)  # (batch_size, d_model)

        # Process MF Branch
        # Shape: (batch_size, mf_seq_len, d_model)
        mf_x = self.mf_proj(mf_features)
        mf_x = self.mf_pos_enc(mf_x)
        # Shape: (batch_size, mf_seq_len, d_model)
        mf_x = self.mf_encoder(mf_x)
        # Use temporal attention pooling for medium frequency
        # Pattern recognition needs to see the whole sequence
        seq_len = mf_x.size(1)
        time_weights = torch.exp(torch.linspace(-1.5, 0, seq_len, device=mf_x.device))
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)
        time_weights = time_weights / time_weights.sum(dim=1, keepdim=True)

        # Weighted temporal pooling
        mf_weighted = mf_x * time_weights
        mf_rep = mf_weighted.sum(dim=1)  # (batch_size, d_model)

        # Process LF Branch
        # Shape: (batch_size, lf_seq_len, d_model)
        lf_x = self.lf_proj(lf_features)
        lf_x = self.lf_pos_enc(lf_x)
        # Shape: (batch_size, lf_seq_len, d_model)
        lf_x = self.lf_encoder(lf_x)
        # Use temporal attention pooling for low frequency
        # Swing points and patterns across the full window
        seq_len = lf_x.size(1)
        time_weights = torch.exp(torch.linspace(-1, 0, seq_len, device=lf_x.device))
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)
        time_weights = time_weights / time_weights.sum(dim=1, keepdim=True)

        # Weighted temporal pooling
        lf_weighted = lf_x * time_weights
        lf_rep = lf_weighted.sum(dim=1)  # (batch_size, d_model)

        # Process Portfolio Branch
        # Shape: (batch_size, portfolio_seq_len, d_model)
        portfolio_x = self.portfolio_proj(portfolio_features)
        portfolio_x = self.portfolio_pos_enc(portfolio_x)
        # Shape: (batch_size, portfolio_seq_len, d_model)
        portfolio_x = self.portfolio_encoder(portfolio_x)
        # Portfolio uses recent history with decay
        seq_len = portfolio_x.size(1)
        time_weights = torch.exp(
            torch.linspace(-1, 0, seq_len, device=portfolio_x.device)
        )
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)
        time_weights = time_weights / time_weights.sum(dim=1, keepdim=True)

        # Weighted temporal pooling
        portfolio_weighted = portfolio_x * time_weights
        portfolio_rep = portfolio_weighted.sum(dim=1)  # (batch_size, d_model)

        # Cross-timeframe attention - HF attends to MF/LF patterns
        # This helps identify entry points within larger patterns
        # Concatenate MF and LF sequences for context
        pattern_context = torch.cat(
            [mf_x, lf_x], dim=1
        )  # (batch, mf_len + lf_len, d_model)

        # Use recent HF data as query to find relevant patterns
        hf_recent = hf_x[:, -10:, :]  # Last 10 seconds for entry timing
        cross_attn_output, cross_attn_weights = self.cross_timeframe_attention(
            query=hf_recent, key=pattern_context, value=pattern_context
        )
        # Average the cross-attention output
        cross_attn_rep = cross_attn_output.mean(dim=1)  # (batch, d_model)

        # Extract key patterns from LF data (swing points, consolidations)
        # Transpose for Conv1d: (batch, d_model, seq_len)
        lf_patterns = lf_x.transpose(1, 2)
        pattern_features = self.pattern_extractor(lf_patterns).squeeze(
            -1
        )  # (batch, d_model//4)

        # Pad pattern features to match d_model
        pattern_features_padded = F.pad(
            pattern_features, (0, self.model_config.d_model - pattern_features.size(-1))
        )

        # Fusion via attention
        # Prepares all branches for fusion
        # Shape: (batch_size, 5, d_model) - HF, MF, LF, Portfolio, Cross-attention
        features_to_fuse = torch.stack(
            [hf_rep, mf_rep, lf_rep, portfolio_rep, cross_attn_rep], dim=1
        )

        # Fuse all-branches
        # Shape: (batch_size, d_fused)
        fused = self.fusion(features_to_fuse)

        # Output action parameters for discrete action space
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

        # Get attention weights if available
        if hasattr(self.fusion, "get_branch_importance"):
            self._last_branch_importance = self.fusion.get_branch_importance()

        # Store action probabilities for analysis (discrete actions only)
        if len(action_params) == 2:
            # Convert logits to probabilities
            type_probs = torch.softmax(action_params[0], dim=-1)
            size_probs = torch.softmax(action_params[1], dim=-1)
            self._last_action_probs = (type_probs.detach(), size_probs.detach())

        return action_params, value

    # In ai/transformer.py - MultiBranchTransformer class
    def get_action(
        self, state_dict: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an action from the policy with guaranteed log probability calculation.
        """
        with torch.no_grad():
            action_params, value = self.forward(state_dict)

            # For discrete actions only
            if len(action_params) == 2:
                action_type_logits, action_size_logits = action_params

                if deterministic:
                    action_type = torch.argmax(action_type_logits, dim=-1)
                    action_size = torch.argmax(action_size_logits, dim=-1)
                else:
                    action_type_dist = torch.distributions.Categorical(
                        logits=action_type_logits
                    )
                    action_size_dist = torch.distributions.Categorical(
                        logits=action_size_logits
                    )
                    action_type = action_type_dist.sample()
                    action_size = action_size_dist.sample()

                action = torch.stack([action_type, action_size], dim=-1)

                # Calculate combined log probability
                type_log_prob = torch.distributions.Categorical(
                    logits=action_type_logits
                ).log_prob(action_type)
                size_log_prob = torch.distributions.Categorical(
                    logits=action_size_logits
                ).log_prob(action_size)
                log_prob = (type_log_prob + size_log_prob).unsqueeze(1)

                action_info = {
                    "action_type_logits": action_type_logits,
                    "action_size_logits": action_size_logits,
                    "value": value,
                    "log_prob": log_prob,  # Always include log_prob
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
                    "logits": logits,
                    "value": value,
                    "log_prob": log_prob,  # Always include log_prob
                }

            return action, action_info

    def get_last_attention_weights(self) -> Optional[np.ndarray]:
        """Get the last computed attention weights from fusion layer"""
        if hasattr(self, "_last_branch_importance"):
            return self._last_branch_importance
        return None

    def get_last_action_probabilities(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the last computed action probabilities"""
        if hasattr(self, "_last_action_probs"):
            return self._last_action_probs
        return None
