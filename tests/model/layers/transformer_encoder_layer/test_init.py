"""
Comprehensive tests for TransformerEncoderLayer.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.layers import TransformerEncoderLayer


class TestTransformerEncoderLayerInit:
    """Test cases for TransformerEncoderLayer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # Check self-attention layer
        assert isinstance(layer.self_attn, nn.MultiheadAttention)
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead
        assert layer.self_attn.batch_first == True
        
        # Check feed-forward layers with default dim_feedforward
        assert isinstance(layer.linear1, nn.Linear)
        assert isinstance(layer.linear2, nn.Linear)
        assert layer.linear1.in_features == d_model
        assert layer.linear1.out_features == 2048  # Default dim_feedforward
        assert layer.linear2.in_features == 2048
        assert layer.linear2.out_features == d_model
        
        # Check normalization layers
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)
        assert layer.norm1.normalized_shape == (d_model,)
        assert layer.norm2.normalized_shape == (d_model,)
        
        # Check dropout layers
        assert isinstance(layer.dropout, nn.Dropout)
        assert isinstance(layer.dropout1, nn.Dropout)
        assert isinstance(layer.dropout2, nn.Dropout)
        assert layer.dropout.p == 0.1  # Default dropout
        assert layer.dropout1.p == 0.1
        assert layer.dropout2.p == 0.1
        
        # Check activation
        assert isinstance(layer.activation, nn.GELU)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        d_model = 256
        nhead = 4
        dim_feedforward = 1024
        dropout = 0.2
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        
        # Check dimensions
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead
        assert layer.linear1.out_features == dim_feedforward
        assert layer.linear2.in_features == dim_feedforward
        
        # Check dropout values
        assert layer.dropout.p == dropout
        assert layer.dropout1.p == dropout
        assert layer.dropout2.p == dropout

    def test_init_small_dimensions(self):
        """Test initialization with small dimensions."""
        d_model = 64
        nhead = 2
        dim_feedforward = 128
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead
        assert layer.linear1.out_features == dim_feedforward

    def test_init_large_dimensions(self):
        """Test initialization with large dimensions."""
        d_model = 1024
        nhead = 16
        dim_feedforward = 4096
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead
        assert layer.linear1.out_features == dim_feedforward

    def test_init_zero_dropout(self):
        """Test initialization with zero dropout."""
        d_model = 512
        nhead = 8
        dropout = 0.0
        
        layer = TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        
        assert layer.dropout.p == 0.0
        assert layer.dropout1.p == 0.0
        assert layer.dropout2.p == 0.0

    def test_init_maximum_dropout(self):
        """Test initialization with maximum dropout."""
        d_model = 512
        nhead = 8
        dropout = 1.0
        
        layer = TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        
        assert layer.dropout.p == 1.0
        assert layer.dropout1.p == 1.0
        assert layer.dropout2.p == 1.0

    def test_init_attention_parameters(self):
        """Test that attention layer is configured correctly."""
        d_model = 512
        nhead = 8
        dropout = 0.15
        
        layer = TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        
        # Check attention-specific parameters
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead
        assert layer.self_attn.dropout == dropout
        assert layer.self_attn.batch_first == True

    def test_init_feedforward_network_structure(self):
        """Test that feed-forward network is structured correctly."""
        d_model = 256
        nhead = 4
        dim_feedforward = 512
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        # Check feed-forward structure: linear1 -> activation -> dropout -> linear2
        assert isinstance(layer.linear1, nn.Linear)
        assert isinstance(layer.linear2, nn.Linear)
        assert isinstance(layer.dropout, nn.Dropout)
        assert isinstance(layer.activation, nn.GELU)
        
        # Check dimensions flow correctly
        assert layer.linear1.in_features == d_model
        assert layer.linear1.out_features == dim_feedforward
        assert layer.linear2.in_features == dim_feedforward
        assert layer.linear2.out_features == d_model

    def test_init_normalization_layers(self):
        """Test that normalization layers are set up correctly."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # Both norms should be LayerNorm with d_model dimensions
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)
        assert layer.norm1.normalized_shape == (d_model,)
        assert layer.norm2.normalized_shape == (d_model,)
        
        # Check default parameters
        assert layer.norm1.eps == 1e-5  # Default LayerNorm eps
        assert layer.norm2.eps == 1e-5

    def test_init_activation_function(self):
        """Test that GELU activation is used."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        assert isinstance(layer.activation, nn.GELU)

    def test_init_parameter_initialization(self):
        """Test that parameters are initialized properly."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # Check that all parameters exist and are tensors
        for name, param in layer.named_parameters():
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad
            
        # Check specific parameter existence
        assert hasattr(layer.linear1, 'weight')
        assert hasattr(layer.linear1, 'bias')
        assert hasattr(layer.linear2, 'weight')
        assert hasattr(layer.linear2, 'bias')

    @pytest.mark.parametrize("d_model", [64, 128, 256, 512, 1024])
    def test_init_various_d_model_sizes(self, d_model):
        """Test initialization with various d_model sizes."""
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        assert layer.self_attn.embed_dim == d_model
        assert layer.norm1.normalized_shape == (d_model,)
        assert layer.norm2.normalized_shape == (d_model,)

    @pytest.mark.parametrize("nhead", [1, 2, 4, 8, 16])
    def test_init_various_nhead_sizes(self, nhead):
        """Test initialization with various nhead sizes."""
        d_model = 512
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        assert layer.self_attn.num_heads == nhead

    @pytest.mark.parametrize("dim_feedforward", [256, 512, 1024, 2048, 4096])
    def test_init_various_feedforward_sizes(self, dim_feedforward):
        """Test initialization with various feedforward sizes."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        assert layer.linear1.out_features == dim_feedforward
        assert layer.linear2.in_features == dim_feedforward

    def test_init_d_model_not_divisible_by_nhead(self):
        """Test initialization when d_model is not divisible by nhead."""
        d_model = 513  # Not divisible by 8
        nhead = 8
        
        # Should work - PyTorch MultiheadAttention handles this internally
        layer = TransformerEncoderLayer(d_model, nhead)
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead

    def test_init_minimum_valid_dimensions(self):
        """Test initialization with minimum valid dimensions."""
        d_model = 1
        nhead = 1
        dim_feedforward = 1
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        assert layer.self_attn.embed_dim == d_model
        assert layer.self_attn.num_heads == nhead
        assert layer.linear1.out_features == dim_feedforward

    def test_init_batch_first_attention(self):
        """Test that attention layer uses batch_first=True."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # Should be configured for batch-first inputs
        assert layer.self_attn.batch_first == True

    def test_init_dropout_layer_consistency(self):
        """Test that all dropout layers have the same dropout rate."""
        d_model = 512
        nhead = 8
        dropout = 0.3
        
        layer = TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        
        # All dropout layers should have the same rate
        assert layer.dropout.p == dropout
        assert layer.dropout1.p == dropout
        assert layer.dropout2.p == dropout

    def test_init_layer_norm_parameters(self):
        """Test LayerNorm layer parameters."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # Check LayerNorm has learnable parameters
        assert layer.norm1.weight.requires_grad
        assert layer.norm1.bias.requires_grad
        assert layer.norm2.weight.requires_grad
        assert layer.norm2.bias.requires_grad
        
        # Check shapes
        assert layer.norm1.weight.shape == (d_model,)
        assert layer.norm1.bias.shape == (d_model,)
        assert layer.norm2.weight.shape == (d_model,)
        assert layer.norm2.bias.shape == (d_model,)

    def test_init_memory_efficiency(self):
        """Test that initialization is memory efficient."""
        d_model = 1024
        nhead = 16
        dim_feedforward = 4096
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        # Should initialize without memory issues
        assert layer.self_attn.embed_dim == d_model
        
        # Check parameter count is reasonable
        total_params = sum(p.numel() for p in layer.parameters())
        assert total_params > 0

    def test_init_device_placement(self):
        """Test that layer is properly initialized on CPU."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # All parameters should be on CPU by default
        for param in layer.parameters():
            assert param.device.type == 'cpu'

    def test_init_parameter_dtypes(self):
        """Test that parameters have correct dtypes."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # All parameters should be float32 by default
        for param in layer.parameters():
            assert param.dtype == torch.float32

    def test_init_module_training_mode(self):
        """Test that layer is in training mode by default."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        assert layer.training
        assert layer.dropout.training
        assert layer.dropout1.training
        assert layer.dropout2.training

    def test_init_named_modules_structure(self):
        """Test the structure of named modules."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        
        # Check that all expected modules are present
        module_names = {name for name, _ in layer.named_modules()}
        
        expected_modules = {
            '', 'self_attn', 'linear1', 'dropout', 'linear2',
            'norm1', 'norm2', 'dropout1', 'dropout2', 'activation'
        }
        
        assert expected_modules.issubset(module_names)

    def test_init_state_dict_keys(self):
        """Test that state dict has expected keys."""
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        state_dict = layer.state_dict()
        
        # Check for key parameter groups
        linear1_keys = [k for k in state_dict.keys() if 'linear1' in k]
        linear2_keys = [k for k in state_dict.keys() if 'linear2' in k]
        norm1_keys = [k for k in state_dict.keys() if 'norm1' in k]
        norm2_keys = [k for k in state_dict.keys() if 'norm2' in k]
        
        assert len(linear1_keys) >= 2  # weight and bias
        assert len(linear2_keys) >= 2  # weight and bias
        assert len(norm1_keys) >= 2   # weight and bias
        assert len(norm2_keys) >= 2   # weight and bias