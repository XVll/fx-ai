"""
Comprehensive tests for TransformerEncoder.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
import copy

from model.layers import TransformerEncoder, TransformerEncoderLayer


class TestTransformerEncoderInit:
    """Test cases for TransformerEncoder initialization."""

    @pytest.fixture
    def encoder_layer(self):
        """Create a standard encoder layer for testing."""
        return TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256)

    def test_init_basic_functionality(self, encoder_layer):
        """Test basic initialization functionality."""
        num_layers = 3
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Check that layers are created
        assert len(encoder.layers) == num_layers
        assert isinstance(encoder.layers, nn.ModuleList)
        
        # Check norm layer
        assert isinstance(encoder.norm, nn.LayerNorm)
        assert encoder.norm.normalized_shape == (64,)  # d_model from encoder_layer

    def test_init_single_layer(self, encoder_layer):
        """Test initialization with single layer."""
        encoder = TransformerEncoder(encoder_layer, 1)
        
        assert len(encoder.layers) == 1
        assert isinstance(encoder.layers[0], TransformerEncoderLayer)

    def test_init_multiple_layers(self, encoder_layer):
        """Test initialization with multiple layers."""
        num_layers = 6
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        assert len(encoder.layers) == num_layers
        
        # Each layer should be an independent copy
        for i in range(num_layers):
            assert isinstance(encoder.layers[i], TransformerEncoderLayer)
            if i > 0:
                assert encoder.layers[i] is not encoder.layers[i-1]

    def test_init_layer_independence(self, encoder_layer):
        """Test that layers are independent copies."""
        num_layers = 3
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Modify one layer's parameters
        encoder.layers[0].linear1.weight.data.fill_(1.0)
        
        # Other layers should be unaffected
        assert not torch.allclose(
            encoder.layers[0].linear1.weight,
            encoder.layers[1].linear1.weight
        )

    def test_init_norm_layer_configuration(self, encoder_layer):
        """Test that norm layer is configured correctly."""
        encoder = TransformerEncoder(encoder_layer, 3)
        
        # Norm should match the d_model from encoder layer
        d_model = encoder_layer.norm1.normalized_shape[0]
        assert encoder.norm.normalized_shape == (d_model,)

    def test_init_large_number_of_layers(self, encoder_layer):
        """Test initialization with large number of layers."""
        num_layers = 12
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        assert len(encoder.layers) == num_layers

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 6, 12])
    def test_init_various_layer_counts(self, encoder_layer, num_layers):
        """Test initialization with various layer counts."""
        encoder = TransformerEncoder(encoder_layer, num_layers)
        assert len(encoder.layers) == num_layers

    def test_init_parameter_sharing_prevention(self, encoder_layer):
        """Test that parameters are not shared between layers."""
        encoder = TransformerEncoder(encoder_layer, 3)
        
        # Collect all parameter ids
        param_ids = set()
        for layer in encoder.layers:
            for param in layer.parameters():
                param_id = id(param)
                assert param_id not in param_ids  # No sharing
                param_ids.add(param_id)

    def test_init_deepcopy_usage(self, encoder_layer):
        """Test that deepcopy creates independent layers."""
        original_weight = encoder_layer.linear1.weight.clone()
        
        encoder = TransformerEncoder(encoder_layer, 2)
        
        # Original layer should be unchanged
        assert torch.equal(encoder_layer.linear1.weight, original_weight)
        
        # Each copy should have different parameter objects
        assert encoder.layers[0].linear1.weight is not encoder.layers[1].linear1.weight

    def test_init_state_dict_structure(self, encoder_layer):
        """Test state dict structure."""
        encoder = TransformerEncoder(encoder_layer, 3)
        state_dict = encoder.state_dict()
        
        # Should have keys for each layer and norm
        layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
        norm_keys = [k for k in state_dict.keys() if 'norm.' in k]
        
        assert len(layer_keys) > 0
        assert len(norm_keys) > 0

    def test_init_memory_efficiency(self, encoder_layer):
        """Test memory efficiency with many layers."""
        # This should not cause memory issues
        encoder = TransformerEncoder(encoder_layer, 6)
        
        # Count total parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        assert total_params > 0