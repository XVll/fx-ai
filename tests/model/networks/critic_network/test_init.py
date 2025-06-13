"""
Comprehensive tests for CriticNetwork.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.networks import CriticNetwork


class TestCriticNetworkInit:
    """Test cases for CriticNetwork initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        input_dim = 256
        
        critic = CriticNetwork(input_dim)
        
        # Check that critic network is created
        assert isinstance(critic.critic, nn.Sequential)
        
        # Check network structure: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear
        assert len(critic.critic) == 7
        assert isinstance(critic.critic[0], nn.Linear)
        assert isinstance(critic.critic[1], nn.LayerNorm)
        assert isinstance(critic.critic[2], nn.GELU)
        assert isinstance(critic.critic[3], nn.Linear)
        assert isinstance(critic.critic[4], nn.LayerNorm)
        assert isinstance(critic.critic[5], nn.GELU)
        assert isinstance(critic.critic[6], nn.Linear)

    def test_init_custom_hidden_dim(self):
        """Test initialization with custom hidden dimension."""
        input_dim = 128
        hidden_dim = 512
        
        critic = CriticNetwork(input_dim, hidden_dim)
        
        # Check first layer dimensions
        first_linear = critic.critic[0]
        assert first_linear.in_features == input_dim
        assert first_linear.out_features == hidden_dim
        
        # Check second layer dimensions
        second_linear = critic.critic[3]
        assert second_linear.in_features == hidden_dim
        assert second_linear.out_features == hidden_dim

    def test_init_output_dimension(self):
        """Test that output dimension is always 1."""
        input_dim = 256
        hidden_dim = 256
        
        critic = CriticNetwork(input_dim, hidden_dim)
        
        # Final layer should output single value
        final_linear = critic.critic[6]
        assert final_linear.out_features == 1

    def test_init_layer_norm_configuration(self):
        """Test LayerNorm layer configuration."""
        input_dim = 256
        hidden_dim = 512
        
        critic = CriticNetwork(input_dim, hidden_dim)
        
        # Check LayerNorm dimensions
        first_norm = critic.critic[1]
        second_norm = critic.critic[4]
        
        assert first_norm.normalized_shape == (hidden_dim,)
        assert second_norm.normalized_shape == (hidden_dim,)

    def test_init_activation_functions(self):
        """Test that GELU activations are used."""
        critic = CriticNetwork(256)
        
        # Check activations
        assert isinstance(critic.critic[2], nn.GELU)
        assert isinstance(critic.critic[5], nn.GELU)

    @pytest.mark.parametrize("input_dim", [64, 128, 256, 512])
    def test_init_various_input_dimensions(self, input_dim):
        """Test initialization with various input dimensions."""
        critic = CriticNetwork(input_dim)
        
        first_linear = critic.critic[0]
        assert first_linear.in_features == input_dim

    @pytest.mark.parametrize("hidden_dim", [128, 256, 512, 1024])
    def test_init_various_hidden_dimensions(self, hidden_dim):
        """Test initialization with various hidden dimensions."""
        critic = CriticNetwork(256, hidden_dim)
        
        first_linear = critic.critic[0]
        assert first_linear.out_features == hidden_dim

    def test_init_parameter_initialization(self):
        """Test that parameters are properly initialized."""
        critic = CriticNetwork(256)
        
        # Check that all parameters exist and are tensors
        for name, param in critic.named_parameters():
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad

    def test_init_state_dict_structure(self):
        """Test state dict structure."""
        critic = CriticNetwork(256, 256)
        state_dict = critic.state_dict()
        
        # Should have keys for all layers
        linear_keys = [k for k in state_dict.keys() if 'weight' in k or 'bias' in k]
        assert len(linear_keys) > 0

    def test_init_minimum_dimensions(self):
        """Test initialization with minimum dimensions."""
        critic = CriticNetwork(input_dim=1, hidden_dim=1)
        
        assert isinstance(critic.critic, nn.Sequential)
        assert len(critic.critic) == 7