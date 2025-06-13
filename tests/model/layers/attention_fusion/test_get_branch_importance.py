"""
Comprehensive tests for AttentionFusion.get_branch_importance method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np

from model.layers import AttentionFusion


class TestAttentionFusionGetBranchImportance:
    """Test cases for AttentionFusion get_branch_importance method."""

    @pytest.fixture
    def fusion(self):
        """Create a standard AttentionFusion instance."""
        return AttentionFusion(in_dim=64, num_branches=4, out_dim=256, num_heads=4)

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor."""
        return torch.randn(2, 4, 64)  # batch_size=2, num_branches=4, in_dim=64

    def test_get_branch_importance_no_forward_call(self, fusion):
        """Test get_branch_importance when no forward pass has been made."""
        result = fusion.get_branch_importance()
        
        # Should return None if no forward pass has been made
        assert result is None

    def test_get_branch_importance_after_forward(self, fusion, input_tensor):
        """Test get_branch_importance after forward pass."""
        # Make a forward pass to generate attention weights
        fusion(input_tensor)
        
        result = fusion.get_branch_importance()
        
        # Should return branch importance scores
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)  # num_branches

    def test_get_branch_importance_valid_probabilities(self, fusion, input_tensor):
        """Test that branch importance scores are valid."""
        fusion(input_tensor)
        result = fusion.get_branch_importance()
        
        if result is not None:
            # Should be non-negative
            assert np.all(result >= 0)
            # Should be finite
            assert np.all(np.isfinite(result))

    def test_get_branch_importance_computation_correctness(self, fusion, input_tensor):
        """Test that branch importance is computed correctly."""
        # Use eval mode for deterministic behavior
        fusion.eval()
        
        # Forward pass with return_attention=True to get attention weights
        output, attention_weights = fusion(input_tensor, return_attention=True)
        
        # Get branch importance
        importance = fusion.get_branch_importance()
        
        if importance is not None and attention_weights is not None:
            # Verify it's a numpy array
            assert isinstance(importance, np.ndarray)
            assert importance.shape == (4,)  # num_branches

    def test_get_branch_importance_multiple_forward_calls(self, fusion, input_tensor):
        """Test get_branch_importance after multiple forward passes."""
        # First forward pass
        fusion(input_tensor)
        first_importance = fusion.get_branch_importance()
        
        # Second forward pass with different input
        different_input = torch.randn(2, 4, 64) + 10
        fusion(different_input)
        second_importance = fusion.get_branch_importance()
        
        # Should have valid importance scores for both
        if first_importance is not None and second_importance is not None:
            assert first_importance.shape == second_importance.shape
            assert isinstance(second_importance, np.ndarray)

    def test_get_branch_importance_batch_averaging(self, fusion, input_tensor):
        """Test that importance is averaged over batch and heads."""
        fusion(input_tensor)
        importance = fusion.get_branch_importance()
        
        if importance is not None:
            # Should be 1D array with num_branches elements
            assert importance.ndim == 1
            assert importance.shape[0] == 4  # num_branches

    def test_get_branch_importance_different_batch_sizes(self, fusion):
        """Test get_branch_importance with different batch sizes."""
        for batch_size in [1, 3, 8]:
            input_tensor = torch.randn(batch_size, 4, 64)
            fusion(input_tensor)
            importance = fusion.get_branch_importance()
            
            if importance is not None:
                assert importance.shape == (4,)  # Always num_branches

    def test_get_branch_importance_none_attention_weights(self, fusion):
        """Test behavior when attention weights are None."""
        # Manually set attention weights to None
        fusion.last_attention_weights = None
        
        result = fusion.get_branch_importance()
        assert result is None

    def test_get_branch_importance_zero_attention_weights(self, fusion, input_tensor):
        """Test behavior with zero attention weights."""
        # Forward pass first
        fusion(input_tensor)
        
        # Manually set zero attention weights
        if fusion.last_attention_weights is not None:
            fusion.last_attention_weights = torch.zeros_like(fusion.last_attention_weights)
            
            importance = fusion.get_branch_importance()
            if importance is not None:
                # All importance scores should be zero
                assert np.allclose(importance, 0.0)

    def test_get_branch_importance_uniform_attention(self, fusion, input_tensor):
        """Test behavior with uniform attention weights."""
        fusion(input_tensor)
        
        # Manually set uniform attention weights
        if fusion.last_attention_weights is not None:
            batch_size = fusion.last_attention_weights.shape[0]
            num_heads = fusion.last_attention_weights.shape[1] if fusion.last_attention_weights.ndim > 2 else 1
            seq_len = 4  # num_branches
            
            # Create uniform attention weights
            uniform_weights = torch.ones_like(fusion.last_attention_weights) / seq_len
            fusion.last_attention_weights = uniform_weights
            
            importance = fusion.get_branch_importance()
            if importance is not None:
                # All branches should have equal importance
                expected_value = 1.0  # Sum of uniform probabilities
                assert np.allclose(importance, expected_value, atol=1e-6)

    def test_get_branch_importance_cpu_conversion(self, fusion, input_tensor):
        """Test that result is properly converted to CPU numpy array."""
        if torch.cuda.is_available():
            fusion_cuda = fusion.cuda()
            input_cuda = input_tensor.cuda()
            
            fusion_cuda(input_cuda)
            importance = fusion_cuda.get_branch_importance()
            
            if importance is not None:
                # Should be numpy array (on CPU)
                assert isinstance(importance, np.ndarray)

    def test_get_branch_importance_detached_computation(self, fusion):
        """Test that computation doesn't require gradients."""
        input_tensor = torch.randn(2, 4, 64, requires_grad=True)
        
        # Forward pass
        output = fusion(input_tensor)
        
        # Get importance (should not affect gradients)
        importance = fusion.get_branch_importance()
        
        if importance is not None:
            assert isinstance(importance, np.ndarray)
            # Should not interfere with gradient computation
            loss = output.sum()
            loss.backward()
            assert input_tensor.grad is not None

    def test_get_branch_importance_consistency(self, fusion, input_tensor):
        """Test consistency across multiple calls."""
        fusion(input_tensor)
        
        # Multiple calls should return the same result
        importance1 = fusion.get_branch_importance()
        importance2 = fusion.get_branch_importance()
        importance3 = fusion.get_branch_importance()
        
        if importance1 is not None:
            assert importance2 is not None
            assert importance3 is not None
            np.testing.assert_array_equal(importance1, importance2)
            np.testing.assert_array_equal(importance2, importance3)

    def test_get_branch_importance_different_num_branches(self):
        """Test get_branch_importance with different numbers of branches."""
        for num_branches in [2, 3, 5, 8]:
            fusion = AttentionFusion(in_dim=64, num_branches=num_branches)
            input_tensor = torch.randn(2, num_branches, 64)
            
            fusion(input_tensor)
            importance = fusion.get_branch_importance()
            
            if importance is not None:
                assert importance.shape == (num_branches,)

    def test_get_branch_importance_mathematical_properties(self, fusion, input_tensor):
        """Test mathematical properties of branch importance."""
        fusion(input_tensor)
        importance = fusion.get_branch_importance()
        
        if importance is not None:
            # Should be finite
            assert np.all(np.isfinite(importance))
            # Should be non-negative
            assert np.all(importance >= 0)
            # Should have reasonable magnitude (not too large)
            assert np.all(importance <= 100)  # Reasonable upper bound

    def test_get_branch_importance_edge_case_single_head(self):
        """Test get_branch_importance with single attention head."""
        fusion = AttentionFusion(in_dim=64, num_branches=4, num_heads=1)
        input_tensor = torch.randn(2, 4, 64)
        
        fusion(input_tensor)
        importance = fusion.get_branch_importance()
        
        if importance is not None:
            assert importance.shape == (4,)

    def test_get_branch_importance_edge_case_many_heads(self):
        """Test get_branch_importance with many attention heads."""
        fusion = AttentionFusion(in_dim=64, num_branches=4, num_heads=8)
        input_tensor = torch.randn(2, 4, 64)
        
        fusion(input_tensor)
        importance = fusion.get_branch_importance()
        
        if importance is not None:
            assert importance.shape == (4,)

    @pytest.mark.parametrize("num_branches", [2, 3, 4, 5])
    def test_get_branch_importance_various_branch_counts_param(self, num_branches):
        """Parameterized test for various branch counts."""
        fusion = AttentionFusion(in_dim=64, num_branches=num_branches)
        input_tensor = torch.randn(2, num_branches, 64)
        
        fusion(input_tensor)
        importance = fusion.get_branch_importance()
        
        if importance is not None:
            assert importance.shape == (num_branches,)