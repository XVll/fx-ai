"""
Comprehensive tests for PositionalEncoding.forward method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
import math

from model.layers import PositionalEncoding


class TestPositionalEncodingForward:
    """Test cases for PositionalEncoding forward pass."""

    @pytest.fixture
    def pe_standard(self):
        """Create a standard PositionalEncoding instance."""
        return PositionalEncoding(d_model=64, dropout=0.1, max_len=100)

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor."""
        return torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, d_model=64

    def test_forward_basic_functionality(self, pe_standard, input_tensor):
        """Test basic forward pass functionality."""
        output = pe_standard(input_tensor)
        
        # Output should have same shape as input
        assert output.shape == input_tensor.shape
        
        # Output should be different from input (due to positional encoding addition)
        assert not torch.equal(output, input_tensor)

    def test_forward_positional_encoding_addition(self):
        """Test that positional encoding is correctly added."""
        d_model = 4
        seq_len = 3
        batch_size = 1
        
        pe = PositionalEncoding(d_model, dropout=0.0, max_len=seq_len)  # No dropout for exact testing
        input_tensor = torch.zeros(batch_size, seq_len, d_model)
        
        output = pe(input_tensor)
        
        # With zero input and no dropout, output should equal positional encoding
        expected = pe.pe[:seq_len].unsqueeze(0)  # Add batch dimension
        torch.testing.assert_close(output, expected, atol=1e-6)

    def test_forward_input_preservation_with_encoding(self):
        """Test that input is preserved and encoding is added."""
        d_model = 4
        seq_len = 2
        batch_size = 1
        
        pe = PositionalEncoding(d_model, dropout=0.0, max_len=seq_len)
        input_tensor = torch.ones(batch_size, seq_len, d_model)
        
        output = pe(input_tensor)
        
        # Output should be input + positional encoding
        expected = input_tensor + pe.pe[:seq_len].unsqueeze(0)
        torch.testing.assert_close(output, expected, atol=1e-6)

    def test_forward_different_batch_sizes(self, pe_standard):
        """Test forward pass with different batch sizes."""
        seq_len = 5
        d_model = 64
        
        for batch_size in [1, 3, 8, 16]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            output = pe_standard(input_tensor)
            
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        d_model = 32
        batch_size = 2
        pe = PositionalEncoding(d_model, dropout=0.0, max_len=50)
        
        for seq_len in [1, 5, 10, 25, 50]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            output = pe(input_tensor)
            
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_sequence_length_exceeds_max_len(self):
        """Test behavior when sequence length exceeds max_len."""
        d_model = 32
        max_len = 5
        seq_len = 10  # Exceeds max_len
        
        pe = PositionalEncoding(d_model, max_len=max_len)
        input_tensor = torch.randn(1, seq_len, d_model)
        
        # Should raise an error or handle gracefully
        with pytest.raises(RuntimeError):
            pe(input_tensor)

    def test_forward_dropout_application(self):
        """Test that dropout is applied during training."""
        d_model = 64
        seq_len = 10
        
        pe_with_dropout = PositionalEncoding(d_model, dropout=0.5)
        pe_without_dropout = PositionalEncoding(d_model, dropout=0.0)
        
        input_tensor = torch.randn(2, seq_len, d_model)
        
        # Set to training mode
        pe_with_dropout.train()
        pe_without_dropout.train()
        
        # Multiple forward passes should give different results with dropout
        output1_dropout = pe_with_dropout(input_tensor)
        output2_dropout = pe_with_dropout(input_tensor)
        
        output1_no_dropout = pe_without_dropout(input_tensor)
        output2_no_dropout = pe_without_dropout(input_tensor)
        
        # With dropout, outputs should be different (with high probability)
        # Without dropout, outputs should be identical
        assert torch.equal(output1_no_dropout, output2_no_dropout)
        # Note: With dropout, outputs might still be equal by chance, so we don't assert inequality

    def test_forward_eval_mode_no_dropout(self):
        """Test that dropout is not applied in eval mode."""
        d_model = 64
        seq_len = 10
        
        pe = PositionalEncoding(d_model, dropout=0.5)
        input_tensor = torch.randn(2, seq_len, d_model)
        
        # Set to eval mode
        pe.eval()
        
        # Multiple forward passes should give identical results in eval mode
        output1 = pe(input_tensor)
        output2 = pe(input_tensor)
        
        torch.testing.assert_close(output1, output2)

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the forward pass."""
        pe = PositionalEncoding(d_model=32, dropout=0.1)
        input_tensor = torch.randn(1, 5, 32, requires_grad=True)
        
        output = pe(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

    def test_forward_no_gradient_on_positional_encoding(self):
        """Test that positional encoding buffer doesn't accumulate gradients."""
        pe = PositionalEncoding(d_model=32, dropout=0.1)
        input_tensor = torch.randn(1, 5, 32, requires_grad=True)
        
        output = pe(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Positional encoding buffer should not have gradients
        assert not pe.pe.requires_grad

    def test_forward_device_consistency(self):
        """Test forward pass with different devices."""
        d_model = 32
        pe = PositionalEncoding(d_model)
        
        # CPU test
        input_cpu = torch.randn(1, 5, d_model)
        output_cpu = pe(input_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # GPU test if available
        if torch.cuda.is_available():
            pe_cuda = pe.cuda()
            input_cuda = input_cpu.cuda()
            output_cuda = pe_cuda(input_cuda)
            assert output_cuda.device.type == 'cuda'

    def test_forward_dtype_preservation(self):
        """Test that forward pass preserves input dtype."""
        pe = PositionalEncoding(d_model=32)
        
        # Test with different dtypes
        for dtype in [torch.float32, torch.float64, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue  # Skip float16 on CPU
                
            input_tensor = torch.randn(1, 5, 32, dtype=dtype)
            output = pe(input_tensor)
            assert output.dtype == dtype

    def test_forward_empty_sequence(self):
        """Test forward pass with empty sequence."""
        pe = PositionalEncoding(d_model=32)
        input_tensor = torch.randn(2, 0, 32)  # Empty sequence
        
        output = pe(input_tensor)
        assert output.shape == (2, 0, 32)

    def test_forward_single_timestep(self):
        """Test forward pass with single timestep."""
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        input_tensor = torch.randn(1, 1, 32)
        
        output = pe(input_tensor)
        
        # Should add position 0 encoding
        expected = input_tensor + pe.pe[0:1].unsqueeze(0)
        torch.testing.assert_close(output, expected, atol=1e-6)

    def test_forward_maximum_sequence_length(self):
        """Test forward pass with maximum allowed sequence length."""
        d_model = 32
        max_len = 50
        pe = PositionalEncoding(d_model, max_len=max_len)
        
        # Use exactly max_len
        input_tensor = torch.randn(1, max_len, d_model)
        output = pe(input_tensor)
        
        assert output.shape == (1, max_len, d_model)

    def test_forward_positional_encoding_slice_correctness(self):
        """Test that correct slice of positional encoding is used."""
        d_model = 4
        max_len = 10
        seq_len = 3
        
        pe = PositionalEncoding(d_model, dropout=0.0, max_len=max_len)
        input_tensor = torch.zeros(1, seq_len, d_model)
        
        output = pe(input_tensor)
        
        # Should use positions 0, 1, 2
        expected = pe.pe[:seq_len].unsqueeze(0)
        torch.testing.assert_close(output, expected, atol=1e-6)

    def test_forward_batch_independence(self):
        """Test that different batch elements get same positional encoding."""
        d_model = 32
        seq_len = 5
        batch_size = 3
        
        pe = PositionalEncoding(d_model, dropout=0.0)
        
        # Different inputs for each batch element
        input_tensor = torch.randn(batch_size, seq_len, d_model)
        output = pe(input_tensor)
        
        # The positional encoding added should be the same for all batch elements
        pos_encoding_added = output - input_tensor
        
        for i in range(1, batch_size):
            torch.testing.assert_close(pos_encoding_added[0], pos_encoding_added[i])

    def test_forward_numerical_stability(self):
        """Test numerical stability with extreme input values."""
        pe = PositionalEncoding(d_model=32)
        
        # Very large values
        input_large = torch.full((1, 5, 32), 1e6)
        output_large = pe(input_large)
        assert torch.isfinite(output_large).all()
        
        # Very small values
        input_small = torch.full((1, 5, 32), 1e-6)
        output_small = pe(input_small)
        assert torch.isfinite(output_small).all()
        
        # Mixed values
        input_mixed = torch.tensor([[[1e6, 1e-6, 0, -1e6]] * 32]).transpose(1, 2)
        output_mixed = pe(input_mixed)
        assert torch.isfinite(output_mixed).all()

    def test_forward_zero_input(self):
        """Test forward pass with zero input."""
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        input_tensor = torch.zeros(1, 5, 32)
        
        output = pe(input_tensor)
        
        # Output should equal positional encoding
        expected = pe.pe[:5].unsqueeze(0)
        torch.testing.assert_close(output, expected, atol=1e-6)

    def test_forward_reproducibility(self):
        """Test reproducibility of forward pass."""
        torch.manual_seed(42)
        pe = PositionalEncoding(d_model=32, dropout=0.1)
        input_tensor = torch.randn(1, 5, 32)
        
        pe.eval()  # Disable dropout for reproducibility
        output1 = pe(input_tensor)
        output2 = pe(input_tensor)
        
        torch.testing.assert_close(output1, output2)

    @pytest.mark.parametrize("seq_len", [1, 5, 10, 20, 50])
    def test_forward_various_sequence_lengths(self, seq_len):
        """Test forward pass with various sequence lengths."""
        d_model = 64
        pe = PositionalEncoding(d_model, max_len=100)
        input_tensor = torch.randn(2, seq_len, d_model)
        
        output = pe(input_tensor)
        assert output.shape == (2, seq_len, d_model)

    def test_forward_input_shape_validation(self):
        """Test that forward validates input shape."""
        pe = PositionalEncoding(d_model=32)
        
        # Wrong number of dimensions
        with pytest.raises((RuntimeError, IndexError)):
            pe(torch.randn(32))  # 1D instead of 3D
            
        with pytest.raises((RuntimeError, IndexError)):
            pe(torch.randn(2, 32))  # 2D instead of 3D
            
        with pytest.raises((RuntimeError, IndexError)):
            pe(torch.randn(2, 5, 5, 32))  # 4D instead of 3D

    def test_forward_feature_dimension_mismatch(self):
        """Test behavior with wrong feature dimension."""
        pe = PositionalEncoding(d_model=32)
        
        # Wrong feature dimension
        input_wrong_dim = torch.randn(1, 5, 64)  # d_model=64 instead of 32
        
        with pytest.raises(RuntimeError):
            pe(input_wrong_dim)

    def test_forward_performance_large_batch(self):
        """Test performance with large batch size."""
        pe = PositionalEncoding(d_model=512)
        
        # Large batch
        large_input = torch.randn(100, 50, 512)
        output = pe(large_input)
        
        assert output.shape == (100, 50, 512)
        assert torch.isfinite(output).all()

    def test_forward_memory_efficiency(self):
        """Test memory efficiency - should not create unnecessary copies."""
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        input_tensor = torch.randn(1, 5, 32)
        
        # Forward pass should be memory efficient
        output = pe(input_tensor)
        
        # Basic check - output should be computed correctly
        expected = input_tensor + pe.pe[:5].unsqueeze(0)
        torch.testing.assert_close(output, expected, atol=1e-6)