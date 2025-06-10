import torch
import torch.nn as nn
import math
import copy


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer ai.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with pre-norm architecture for improved training stability.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pre-norm transformer architecture.
        """
        # Self-attention block
        src_norm = self.norm1(src)
        src2, _ = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)

        # Feed-forward block
        src_norm = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    """
    Stack of N transformer encoder layers.
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Create independent copies of the encoder layer
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape[0])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        return self.norm(output)


class AttentionFusion(nn.Module):
    """
    Self-attention fusion layer to combine representations from different branches.
    """

    def __init__(self, in_dim, num_branches=4, out_dim=None, num_heads=4, dropout=0.1):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim * num_branches

        # Self-attention for fusion
        self.self_attention = nn.MultiheadAttention(
            in_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(in_dim * num_branches, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Store last attention weights for analysis
        self.last_attention_weights = None

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Tensor of shape [batch_size, num_branches, in_dim]
            return_attention: Whether to return attention weights

        Returns:
            output: Tensor of shape [batch_size, out_dim]
            attention_weights: Optional tensor of shape [batch_size, num_branches, num_branches] if return_attention=True
        """
        # Apply self-attention across branches
        attn_output, attention_weights = self.self_attention(x, x, x)

        # Store attention weights for analysis
        if attention_weights is not None:
            self.last_attention_weights = attention_weights.detach()

        # Flatten the branches dimension
        batch_size = x.size(0)
        flattened = attn_output.reshape(batch_size, -1)

        # Project to output dimension
        output = self.proj(flattened)

        if return_attention:
            return output, attention_weights
        return output

    def get_branch_importance(self):
        """Get the average attention each branch receives"""
        if self.last_attention_weights is None:
            return None

        # Average over batch and heads, then sum attention received by each branch
        # Shape: [batch, num_heads, seq_len, seq_len] -> [seq_len]
        avg_attention = self.last_attention_weights.mean(dim=0).mean(dim=0).sum(dim=0)
        return avg_attention.cpu().numpy()
