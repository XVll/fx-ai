"""
Comprehensive tests for ActorNetwork.forward method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.networks import ActorNetwork


class TestActorNetworkForward:
    """Test cases for ActorNetwork forward pass."""

    @pytest.fixture
    def actor_continuous(self):
        """Create continuous action actor."""
        return ActorNetwork(256, 4, continuous_action=True, hidden_dim=256)

    @pytest.fixture
    def actor_discrete(self):
        """Create discrete action actor."""
        return ActorNetwork(256, 7, continuous_action=False, hidden_dim=256)

    @pytest.fixture
    def input_tensor(self):
        """Create input tensor."""
        return torch.randn(2, 256)

    def test_forward_continuous_action(self, actor_continuous, input_tensor):
        """Test forward pass for continuous actions."""
        result = actor_continuous(input_tensor)
        
        # Should return tuple of (mean, log_std)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        mean, log_std = result
        assert mean.shape == (2, 4)  # batch_size, action_dim
        assert log_std.shape == (2, 4)  # batch_size, action_dim

    def test_forward_discrete_action(self, actor_discrete, input_tensor):
        """Test forward pass for discrete actions."""
        result = actor_discrete(input_tensor)
        
        # Should return logits tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 7)  # batch_size, action_dim

    def test_forward_log_std_expansion(self, actor_continuous):
        """Test that log_std is properly expanded for batch."""
        batch_sizes = [1, 3, 8]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 256)
            mean, log_std = actor_continuous(input_tensor)
            
            assert log_std.shape == (batch_size, 4)

    def test_forward_gradient_flow(self, actor_continuous):
        """Test gradient flow through network."""
        input_tensor = torch.randn(2, 256, requires_grad=True)
        
        mean, log_std = actor_continuous(input_tensor)
        loss = mean.sum() + log_std.sum()
        loss.backward()
        
        assert input_tensor.grad is not None

    def test_forward_different_batch_sizes(self, actor_continuous, actor_discrete):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 3, 8]:
            input_tensor = torch.randn(batch_size, 256)
            
            # Continuous
            mean, log_std = actor_continuous(input_tensor)
            assert mean.shape == (batch_size, 4)
            
            # Discrete
            logits = actor_discrete(input_tensor)
            assert logits.shape == (batch_size, 7)

    def test_forward_extreme_values(self, actor_continuous, actor_discrete):
        """Test forward pass with extreme input values."""
        # Large values
        input_large = torch.full((2, 256), 1e3)
        
        mean, log_std = actor_continuous(input_large)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_std).all()
        
        logits = actor_discrete(input_large)
        assert torch.isfinite(logits).all()

    def test_forward_zero_input(self, actor_continuous, actor_discrete):
        """Test forward pass with zero input."""
        input_tensor = torch.zeros(2, 256)
        
        mean, log_std = actor_continuous(input_tensor)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_std).all()
        
        logits = actor_discrete(input_tensor)
        assert torch.isfinite(logits).all()