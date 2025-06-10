"""
Model configuration for transformer architecture.
"""

from typing import List, Any
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Multi-branch transformer model configuration"""

    model_config = {"protected_namespaces": ()}

    # Core architecture
    d_model: int = Field(128, description="Model dimension")
    d_fused: int = Field(512, description="Fused representation dimension")
    n_heads: int = Field(8, description="Attention heads")
    n_layers: int = Field(6, description="Transformer layers")
    d_ff: int = Field(2048, description="Feed-forward dimension")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")

    # Branch-specific configuration
    hf_layers: int = Field(3, description="High-frequency branch layers")
    mf_layers: int = Field(3, description="Medium-frequency branch layers")
    lf_layers: int = Field(2, description="Low-frequency branch layers")
    portfolio_layers: int = Field(2, description="Portfolio branch layers")

    hf_heads: int = Field(8, description="High-frequency attention heads")
    mf_heads: int = Field(8, description="Medium-frequency attention heads")
    lf_heads: int = Field(4, description="Low-frequency attention heads")
    portfolio_heads: int = Field(4, description="Portfolio attention heads")

    # Feature dimensions - dynamically set from FeatureRegistry
    hf_seq_len: int = Field(60, description="High-frequency sequence length")
    hf_feat_dim: int = Field(9, description="High-frequency features")
    mf_seq_len: int = Field(30, description="Medium-frequency sequence length")
    mf_feat_dim: int = Field(43, description="Medium-frequency features")
    lf_seq_len: int = Field(30, description="Low-frequency sequence length")
    lf_feat_dim: int = Field(19, description="Low-frequency features")
    portfolio_seq_len: int = Field(5, description="Portfolio sequence length")
    portfolio_feat_dim: int = Field(10, description="Portfolio features")

    # Todo : action_space_size
    
    @field_validator("hf_feat_dim", "mf_feat_dim", "lf_feat_dim", "portfolio_feat_dim", mode="before")
    @classmethod
    def validate_dimensions(cls, v, info):
        """Validate feature dimensions match FeatureRegistry."""
        from feature.feature_registry import FeatureRegistry
        
        # Get expected dimensions from registry
        dims = FeatureRegistry.get_feature_dimensions(active_only=True)
        
        # Map field names to category names
        field_to_category = {
            "hf_feat_dim": "hf",
            "mf_feat_dim": "mf", 
            "lf_feat_dim": "lf",
            "portfolio_feat_dim": "portfolio"
        }
        
        field_name = info.field_name
        category = field_to_category.get(field_name)
        
        if category and v != dims[category]:
            # Log warning but allow the value for backward compatibility
            import logging
            logging.getLogger(__name__).warning(
                f"Feature dimension mismatch: {field_name}={v} but FeatureRegistry has {dims[category]} active features"
            )
        
        return v
    
    @classmethod
    def from_registry(cls):
        """Create ModelConfig with dimensions from FeatureRegistry."""
        from feature.feature_registry import FeatureRegistry
        dims = FeatureRegistry.get_feature_dimensions(active_only=True)
        
        # Create with registry dimensions
        return cls(
            hf_feat_dim=dims['hf'],
            mf_feat_dim=dims['mf'],
            lf_feat_dim=dims['lf'],
            portfolio_feat_dim=dims['portfolio']
        )

    # Action space
    action_dim: List[int] = Field([3, 4], description="[action_types, position_sizes]")

    @field_validator("action_dim", mode="before")
    @classmethod
    def validate_action_dim(cls, v: Any) -> Any:
        if len(v) != 2:
            raise ValueError("action_dim must have exactly 2 elements")
        if v[0] != 3:
            raise ValueError("action_types must be 3 (BUY, SELL, HOLD)")
        return v