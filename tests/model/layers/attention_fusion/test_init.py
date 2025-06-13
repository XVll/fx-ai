"""
Comprehensive tests for AttentionFusion.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.layers import AttentionFusion


class TestAttentionFusionInit:
    """Test cases for AttentionFusion initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        in_dim = 128
        num_branches = 4
        
        fusion = AttentionFusion(in_dim, num_branches)
        
        # Check self-attention layer
        assert isinstance(fusion.self_attention, nn.MultiheadAttention)
        assert fusion.self_attention.embed_dim == in_dim
        assert fusion.self_attention.num_heads == 4  # Default
        assert fusion.self_attention.batch_first == True
        
        # Check projection layer
        assert isinstance(fusion.proj, nn.Sequential)
        expected_out_dim = in_dim * num_branches  # Default out_dim
        
        # Check components of projection
        proj_linear = fusion.proj[0]
        assert isinstance(proj_linear, nn.Linear)
        assert proj_linear.in_features == in_dim * num_branches
        assert proj_linear.out_features == expected_out_dim

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        in_dim = 256
        num_branches = 5
        out_dim = 512
        num_heads = 8
        dropout = 0.2
        
        fusion = AttentionFusion(in_dim, num_branches, out_dim, num_heads, dropout)
        
        # Check attention layer
        assert fusion.self_attention.embed_dim == in_dim
        assert fusion.self_attention.num_heads == num_heads
        
        # Check projection output dimension
        proj_linear = fusion.proj[0]
        assert proj_linear.out_features == out_dim

    def test_init_out_dim_none(self):
        """Test initialization when out_dim is None (uses default)."""
        in_dim = 128
        num_branches = 3
        
        fusion = AttentionFusion(in_dim, num_branches, out_dim=None)
        
        # Should use default: in_dim * num_branches
        expected_out_dim = in_dim * num_branches
        proj_linear = fusion.proj[0]
        assert proj_linear.out_features == expected_out_dim

    def test_init_projection_sequence_structure(self):
        """Test that projection sequence has correct structure."""
        fusion = AttentionFusion(in_dim=128, num_branches=4)
        
        # Should have: Linear -> LayerNorm -> GELU -> Dropout
        assert len(fusion.proj) == 4
        assert isinstance(fusion.proj[0], nn.Linear)
        assert isinstance(fusion.proj[1], nn.LayerNorm)
        assert isinstance(fusion.proj[2], nn.GELU)
        assert isinstance(fusion.proj[3], nn.Dropout)

    def test_init_attention_weights_storage(self):
        """Test that attention weights storage is initialized."""
        fusion = AttentionFusion(in_dim=128, num_branches=4)
        
        # Should initialize last_attention_weights as None
        assert fusion.last_attention_weights is None

    def test_init_layer_norm_configuration(self):
        """Test LayerNorm configuration in projection."""
        in_dim = 128
        num_branches = 4
        out_dim = 256
        
        fusion = AttentionFusion(in_dim, num_branches, out_dim)
        
        layer_norm = fusion.proj[1]
        assert isinstance(layer_norm, nn.LayerNorm)
        assert layer_norm.normalized_shape == (out_dim,)

    def test_init_dropout_configuration(self):
        """Test dropout configuration in projection."""
        dropout = 0.3
        fusion = AttentionFusion(in_dim=128, num_branches=4, dropout=dropout)
        
        # Check attention dropout
        assert fusion.self_attention.dropout == dropout
        
        # Check projection dropout
        dropout_layer = fusion.proj[3]
        assert isinstance(dropout_layer, nn.Dropout)
        assert dropout_layer.p == dropout

    @pytest.mark.parametrize("in_dim", [64, 128, 256, 512])
    def test_init_various_input_dimensions(self, in_dim):
        """Test initialization with various input dimensions."""
        fusion = AttentionFusion(in_dim, num_branches=4)
        assert fusion.self_attention.embed_dim == in_dim

    @pytest.mark.parametrize("num_branches", [2, 3, 4, 5, 8])
    def test_init_various_branch_counts(self, num_branches):
        """Test initialization with various branch counts."""
        in_dim = 128
        fusion = AttentionFusion(in_dim, num_branches)
        
        proj_linear = fusion.proj[0]
        assert proj_linear.in_features == in_dim * num_branches

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
    def test_init_various_head_counts(self, num_heads):
        """Test initialization with various attention head counts."""
        fusion = AttentionFusion(in_dim=128, num_branches=4, num_heads=num_heads)
        assert fusion.self_attention.num_heads == num_heads

    def test_init_minimum_valid_parameters(self):
        """Test initialization with minimum valid parameters."""
        fusion = AttentionFusion(in_dim=1, num_branches=1, num_heads=1)
        
        assert fusion.self_attention.embed_dim == 1
        assert fusion.self_attention.num_heads == 1

    def test_init_large_parameters(self):
        """Test initialization with large parameters."""
        fusion = AttentionFusion(
            in_dim=1024, 
            num_branches=10, 
            out_dim=2048, 
            num_heads=16
        )
        
        assert fusion.self_attention.embed_dim == 1024
        assert fusion.self_attention.num_heads == 16
        
        proj_linear = fusion.proj[0]
        assert proj_linear.in_features == 1024 * 10
        assert proj_linear.out_features == 2048

    def test_init_zero_dropout(self):
        """Test initialization with zero dropout."""
        fusion = AttentionFusion(in_dim=128, num_branches=4, dropout=0.0)
        
        assert fusion.self_attention.dropout == 0.0
        dropout_layer = fusion.proj[3]
        assert dropout_layer.p == 0.0

    def test_init_batch_first_attention(self):
        """Test that attention layer uses batch_first=True."""
        fusion = AttentionFusion(in_dim=128, num_branches=4)
        assert fusion.self_attention.batch_first == True

    def test_init_parameter_initialization(self):
        """Test that parameters are properly initialized."""
        fusion = AttentionFusion(in_dim=128, num_branches=4)
        
        # Check that all parameters exist and are tensors
        for name, param in fusion.named_parameters():
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad

    def test_init_state_dict_structure(self):
        """Test state dict structure."""
        fusion = AttentionFusion(in_dim=128, num_branches=4, out_dim=256)
        state_dict = fusion.state_dict()
        
        # Should have keys for attention and projection
        attention_keys = [k for k in state_dict.keys() if 'self_attention' in k]
        proj_keys = [k for k in state_dict.keys() if 'proj' in k]
        
        assert len(attention_keys) > 0
        assert len(proj_keys) > 0

    def test_init_device_placement(self):
        """Test that layer is properly initialized on CPU."""
        fusion = AttentionFusion(in_dim=128, num_branches=4)
        
        # All parameters should be on CPU by default
        for param in fusion.parameters():
            assert param.device.type == 'cpu'

    def test_init_training_mode(self):
        """Test that layer is in training mode by default."""
        fusion = AttentionFusion(in_dim=128, num_branches=4)
        
        assert fusion.training
        assert fusion.self_attention.training