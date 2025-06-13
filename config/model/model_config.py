"""
Model configuration for transformer architecture - Hydra version.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Multi-branch transformer model configuration"""

    # Core architecture
    d_model: int = 128                               # Model dimension (typical: 64, 128, 256, 512)
    d_fused: int = 512                              # Fused representation dimension
    n_heads: int = 8                                # Attention heads (typical: 4, 8, 16)
    n_layers: int = 6                               # Transformer layers (typical: 3-12)
    d_ff: int = 2048                                # Feed-forward dimension (usually 4x d_model)
    dropout: float = 0.1                            # Dropout rate (0.0-0.5)

    # Branch-specific configuration
    hf_layers: int = 3                              # High-frequency branch layers
    mf_layers: int = 3                              # Medium-frequency branch layers  
    lf_layers: int = 2                              # Low-frequency branch layers
    portfolio_layers: int = 2                       # Portfolio branch layers

    hf_heads: int = 8                               # High-frequency attention heads
    mf_heads: int = 8                               # Medium-frequency attention heads
    lf_heads: int = 4                               # Low-frequency attention heads
    portfolio_heads: int = 4                        # Portfolio attention heads

    # Feature dimensions (set from FeatureRegistry)
    hf_seq_len: int = 60                            # High-frequency sequence length (1s data)
    hf_feat_dim: int = 9                            # High-frequency features count
    mf_seq_len: int = 30                            # Medium-frequency sequence length (1m data)
    mf_feat_dim: int = 43                           # Medium-frequency features count
    lf_seq_len: int = 30                            # Low-frequency sequence length (5m+ data)
    lf_feat_dim: int = 19                           # Low-frequency features count
    portfolio_seq_len: int = 5                      # Portfolio sequence length
    portfolio_feat_dim: int = 10                    # Portfolio features count

    # Action space - single index for clean 7 actions
    action_count: int = 7  # HOLD, BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dropout range
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout {self.dropout} must be in range [0.0, 1.0]")
        
        # Validate action count
        if self.action_count != 7:
            raise ValueError("action_count must be 7 (HOLD, BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100)")
        
        # Validate attention heads divide evenly into d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}")

    @classmethod
    def from_registry(cls):
        """Create ModelConfig with dimensions from FeatureRegistry."""
        from feature.feature_registry import FeatureRegistry
        dims = FeatureRegistry.get_feature_dimensions(active_only=True)
        
        return cls(
            hf_feat_dim=dims['hf'],
            mf_feat_dim=dims['mf'],
            lf_feat_dim=dims['lf'],
            portfolio_feat_dim=dims['portfolio']
        )