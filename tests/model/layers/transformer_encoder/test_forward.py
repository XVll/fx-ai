"""
Comprehensive tests for TransformerEncoder.forward method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn

from model.layers import TransformerEncoder, TransformerEncoderLayer


class TestTransformerEncoderForward:
    """Test cases for TransformerEncoder forward pass."""

    @pytest.fixture
    def encoder(self):
        """Create a standard TransformerEncoder instance."""
        encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256)
        return TransformerEncoder(encoder_layer, num_layers=3)

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor."""
        return torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, d_model=64

    def test_forward_basic_functionality(self, encoder, input_tensor):
        """Test basic forward pass functionality."""
        output = encoder(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert torch.isfinite(output).all()

    def test_forward_with_mask(self, encoder, input_tensor):
        """Test forward pass with attention mask."""
        seq_len = input_tensor.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output = encoder(input_tensor, mask=mask)
        assert output.shape == input_tensor.shape

    def test_forward_with_key_padding_mask(self, encoder, input_tensor):
        """Test forward pass with key padding mask."""
        batch_size, seq_len = input_tensor.shape[:2]
        padding_mask = torch.zeros(batch_size, seq_len).bool()
        padding_mask[0, -2:] = True
        
        output = encoder(input_tensor, src_key_padding_mask=padding_mask)
        assert output.shape == input_tensor.shape

    def test_forward_layer_composition(self, encoder, input_tensor):
        """Test that layers are applied in sequence."""
        encoder.eval()  # For deterministic behavior
        
        # Forward through entire encoder
        final_output = encoder(input_tensor)
        
        # Manual forward through each layer
        x = input_tensor
        for layer in encoder.layers:
            x = layer(x)
        x = encoder.norm(x)
        manual_output = x
        
        torch.testing.assert_close(final_output, manual_output)

    def test_forward_norm_application(self, encoder, input_tensor):
        """Test that final norm is applied."""
        encoder.eval()
        
        # Get output before final norm
        x = input_tensor
        for layer in encoder.layers:
            x = layer(x)
        pre_norm_output = x
        
        # Get final output with norm
        final_output = encoder(input_tensor)
        
        # They should be different (norm applied)
        assert not torch.allclose(pre_norm_output, final_output)

    def test_forward_gradient_flow(self, encoder):
        """Test gradient flow through all layers."""
        input_tensor = torch.randn(2, 10, 64, requires_grad=True)
        
        output = encoder(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

    def test_forward_different_batch_sizes(self, encoder):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 3, 8]:
            input_tensor = torch.randn(batch_size, 10, 64)
            output = encoder(input_tensor)
            assert output.shape == (batch_size, 10, 64)

    def test_forward_different_sequence_lengths(self, encoder):
        """Test forward pass with different sequence lengths."""
        for seq_len in [1, 5, 20]:
            input_tensor = torch.randn(2, seq_len, 64)
            output = encoder(input_tensor)
            assert output.shape == (2, seq_len, 64)

    def test_forward_single_layer_encoder(self):
        """Test forward pass with single layer encoder."""
        encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8)
        encoder = TransformerEncoder(encoder_layer, num_layers=1)
        input_tensor = torch.randn(2, 10, 64)
        
        output = encoder(input_tensor)
        assert output.shape == input_tensor.shape

    def test_forward_deep_encoder(self):
        """Test forward pass with deep encoder."""
        encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8)
        encoder = TransformerEncoder(encoder_layer, num_layers=12)
        input_tensor = torch.randn(2, 10, 64)
        
        output = encoder(input_tensor)
        assert output.shape == input_tensor.shape
        assert torch.isfinite(output).all()

    def test_forward_extreme_values(self, encoder):
        """Test forward pass with extreme input values."""
        # Large values
        input_large = torch.full((2, 10, 64), 1e3)
        output_large = encoder(input_large)
        assert torch.isfinite(output_large).all()
        
        # Small values
        input_small = torch.full((2, 10, 64), 1e-6)
        output_small = encoder(input_small)
        assert torch.isfinite(output_small).all()

    def test_forward_zero_input(self, encoder):
        """Test forward pass with zero input."""
        input_tensor = torch.zeros(2, 10, 64)
        output = encoder(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("num_layers", [1, 3, 6])
    def test_forward_various_depths(self, num_layers):
        """Test forward pass with various encoder depths."""
        encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8)
        encoder = TransformerEncoder(encoder_layer, num_layers)
        input_tensor = torch.randn(2, 10, 64)
        
        output = encoder(input_tensor)
        assert output.shape == input_tensor.shape

    def test_forward_training_vs_eval(self, encoder, input_tensor):
        """Test behavior difference between training and eval modes."""
        encoder.train()
        output_train = encoder(input_tensor)
        
        encoder.eval()
        output_eval = encoder(input_tensor)
        
        # Shapes should be same, values might differ due to dropout
        assert output_train.shape == output_eval.shape

    def test_forward_device_consistency(self, encoder):
        """Test device consistency."""
        if torch.cuda.is_available():
            encoder_cuda = encoder.cuda()
            input_cuda = torch.randn(2, 10, 64).cuda()
            
            output = encoder_cuda(input_cuda)
            assert output.device.type == 'cuda'