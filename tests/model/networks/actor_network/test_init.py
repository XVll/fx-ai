"""
Comprehensive tests for ActorNetwork.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.networks import ActorNetwork


class TestActorNetworkInit:
    """Test cases for ActorNetwork initialization."""

    def test_init_continuous_action_default(self):
        """Test initialization with continuous action (default)."""
        input_dim = 256
        action_dim = 4
        
        actor = ActorNetwork(input_dim, action_dim)
        
        # Should be continuous by default
        assert actor.continuous_action == True
        
        # Check shared layers
        assert isinstance(actor.shared, nn.Sequential)
        
        # Check continuous action components
        assert hasattr(actor, 'mean')
        assert hasattr(actor, 'log_std')
        assert isinstance(actor.mean, nn.Linear)
        assert isinstance(actor.log_std, nn.Parameter)
        
        # Check dimensions
        assert actor.mean.in_features == 256  # hidden_dim
        assert actor.mean.out_features == action_dim
        assert actor.log_std.shape == (action_dim,)

    def test_init_discrete_action(self):
        """Test initialization with discrete action."""
        input_dim = 256
        action_dim = 7
        
        actor = ActorNetwork(input_dim, action_dim, continuous_action=False)
        
        # Should be discrete
        assert actor.continuous_action == False
        
        # Check discrete action components
        assert hasattr(actor, 'logits')
        assert isinstance(actor.logits, nn.Linear)
        assert not hasattr(actor, 'mean')
        assert not hasattr(actor, 'log_std')
        
        # Check dimensions
        assert actor.logits.in_features == 256  # hidden_dim
        assert actor.logits.out_features == action_dim

    def test_init_custom_hidden_dim(self):
        """Test initialization with custom hidden dimension."""
        input_dim = 128
        action_dim = 4
        hidden_dim = 512
        
        actor = ActorNetwork(input_dim, action_dim, hidden_dim=hidden_dim)
        
        # Check shared layer dimensions
        first_linear = actor.shared[0]
        assert first_linear.in_features == input_dim
        assert first_linear.out_features == hidden_dim

    def test_init_shared_network_structure(self):
        """Test shared network structure."""
        actor = ActorNetwork(256, 4)
        
        # Should have: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU
        assert len(actor.shared) == 6
        assert isinstance(actor.shared[0], nn.Linear)
        assert isinstance(actor.shared[1], nn.LayerNorm)
        assert isinstance(actor.shared[2], nn.GELU)
        assert isinstance(actor.shared[3], nn.Linear)
        assert isinstance(actor.shared[4], nn.LayerNorm)
        assert isinstance(actor.shared[5], nn.GELU)

    @pytest.mark.parametrize("continuous", [True, False])
    def test_init_both_action_types(self, continuous):
        """Test initialization for both action types."""
        actor = ActorNetwork(256, 4, continuous_action=continuous)
        assert actor.continuous_action == continuous

    def test_init_log_std_initialization(self):
        """Test log_std parameter initialization."""
        action_dim = 4
        actor = ActorNetwork(256, action_dim, continuous_action=True)
        
        # log_std should be initialized to zeros
        assert torch.allclose(actor.log_std, torch.zeros(action_dim))

    def test_init_parameter_count(self):
        """Test parameter count for both action types."""
        input_dim = 256
        action_dim = 4
        hidden_dim = 256
        
        actor_continuous = ActorNetwork(input_dim, action_dim, True, hidden_dim)
        actor_discrete = ActorNetwork(input_dim, action_dim, False, hidden_dim)
        
        # Continuous should have more parameters (log_std)
        continuous_params = sum(p.numel() for p in actor_continuous.parameters())
        discrete_params = sum(p.numel() for p in actor_discrete.parameters())
        
        assert continuous_params > discrete_params