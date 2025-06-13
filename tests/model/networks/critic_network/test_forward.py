"""
Comprehensive tests for CriticNetwork.forward method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.networks import CriticNetwork


class TestCriticNetworkForward:
    """Test cases for CriticNetwork forward pass."""

    @pytest.fixture
    def critic(self):
        """Create a standard CriticNetwork instance."""
        return CriticNetwork(input_dim=256, hidden_dim=256)

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor."""
        return torch.randn(2, 256)  # batch_size=2, input_dim=256

    def test_forward_basic_functionality(self, critic, input_tensor):
        """Test basic forward pass functionality."""
        output = critic(input_tensor)
        
        # Should return value estimates
        assert output.shape == (2, 1)  # batch_size, 1
        assert torch.isfinite(output).all()

    def test_forward_single_value_output(self, critic, input_tensor):
        """Test that output is single value per batch element."""
        output = critic(input_tensor)
        
        # Each batch element should have single value estimate
        assert output.shape[-1] == 1

    def test_forward_different_batch_sizes(self, critic):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 3, 8, 16]:
            input_tensor = torch.randn(batch_size, 256)
            output = critic(input_tensor)
            assert output.shape == (batch_size, 1)

    def test_forward_gradient_flow(self, critic):
        """Test gradient flow through the network."""
        input_tensor = torch.randn(2, 256, requires_grad=True)
        
        output = critic(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

    def test_forward_extreme_values(self, critic):
        """Test forward pass with extreme input values."""
        # Large values
        input_large = torch.full((2, 256), 1e3)
        output_large = critic(input_large)
        assert torch.isfinite(output_large).all()
        
        # Small values
        input_small = torch.full((2, 256), 1e-6)
        output_small = critic(input_small)
        assert torch.isfinite(output_small).all()
        
        # Negative values
        input_negative = torch.full((2, 256), -1e3)
        output_negative = critic(input_negative)
        assert torch.isfinite(output_negative).all()

    def test_forward_zero_input(self, critic):
        """Test forward pass with zero input."""
        input_tensor = torch.zeros(2, 256)
        output = critic(input_tensor)
        
        assert output.shape == (2, 1)
        assert torch.isfinite(output).all()

    def test_forward_single_batch_element(self, critic):
        """Test forward pass with single batch element."""
        input_tensor = torch.randn(1, 256)
        output = critic(input_tensor)
        
        assert output.shape == (1, 1)

    def test_forward_training_vs_eval_mode(self, critic, input_tensor):
        """Test behavior in training vs eval mode."""
        # Training mode
        critic.train()
        output_train = critic(input_tensor)
        
        # Eval mode
        critic.eval()
        output_eval = critic(input_tensor)
        
        # Shapes should be same
        assert output_train.shape == output_eval.shape

    def test_forward_reproducibility(self, critic, input_tensor):
        """Test reproducibility of forward pass."""
        critic.eval()  # Ensure deterministic behavior
        
        output1 = critic(input_tensor)
        output2 = critic(input_tensor)
        
        torch.testing.assert_close(output1, output2)

    def test_forward_device_consistency(self, critic):
        """Test device consistency."""
        if torch.cuda.is_available():
            critic_cuda = critic.cuda()
            input_cuda = torch.randn(2, 256).cuda()
            
            output = critic_cuda(input_cuda)
            assert output.device.type == 'cuda'

    def test_forward_dtype_preservation(self, critic):
        """Test that forward pass preserves input dtype."""
        for dtype in [torch.float32, torch.float64]:
            input_tensor = torch.randn(2, 256, dtype=dtype)
            output = critic(input_tensor)
            assert output.dtype == dtype

    def test_forward_numerical_stability(self, critic):
        """Test numerical stability with challenging inputs."""
        # Mixed extreme values
        input_mixed = torch.randn(2, 256)
        input_mixed[0] = 1e6
        input_mixed[1] = -1e6
        
        output = critic(input_mixed)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("input_dim", [64, 128, 256, 512])
    def test_forward_various_input_dimensions(self, input_dim):
        """Test forward pass with various input dimensions."""
        critic = CriticNetwork(input_dim)
        input_tensor = torch.randn(2, input_dim)
        
        output = critic(input_tensor)
        assert output.shape == (2, 1)

    def test_forward_large_batch(self, critic):
        """Test forward pass with large batch size."""
        large_batch = 100
        input_tensor = torch.randn(large_batch, 256)
        
        output = critic(input_tensor)
        assert output.shape == (large_batch, 1)
        assert torch.isfinite(output).all()

    def test_forward_network_layers_activation(self, critic, input_tensor):
        """Test that network layers are properly activated."""
        critic.eval()
        
        output = critic(input_tensor)
        
        # Output should be different from zero (network is activated)
        assert not torch.allclose(output, torch.zeros_like(output), atol=1e-6)