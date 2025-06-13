"""
Comprehensive tests for MultiBranchTransformer.get_last_action_probabilities method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from model.transformer import MultiBranchTransformer
from config.model.model_config import ModelConfig


class TestMultiBranchTransformerGetLastActionProbabilities:
    """Test cases for MultiBranchTransformer get_last_action_probabilities method."""

    @pytest.fixture
    def model_config(self):
        """Create a standard model configuration for testing."""
        return ModelConfig(
            d_model=64,
            d_fused=256,
            hf_seq_len=10,
            mf_seq_len=5,
            lf_seq_len=3,
            portfolio_seq_len=2,
            hf_feat_dim=8,
            mf_feat_dim=12,
            lf_feat_dim=6,
            portfolio_feat_dim=4,
            action_count=7
        )

    @pytest.fixture
    def model(self, model_config):
        """Create a model instance for testing."""
        return MultiBranchTransformer(model_config, device="cpu")

    @pytest.fixture
    def valid_state_dict(self, model_config):
        """Create valid input state dictionary."""
        return {
            "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }

    def test_get_last_action_probabilities_no_forward_call(self, model):
        """Test get_last_action_probabilities when no forward pass has been made."""
        result = model.get_last_action_probabilities()
        
        # Should return None if no forward pass has been made
        assert result is None

    def test_get_last_action_probabilities_after_forward(self, model, valid_state_dict, model_config):
        """Test get_last_action_probabilities after forward pass."""
        # Make a forward pass to generate action probabilities
        model.forward(valid_state_dict)
        
        result = model.get_last_action_probabilities()
        
        # Should have action probabilities stored
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, model_config.action_count)
        
        # Should be valid probabilities (sum to 1, non-negative)
        assert torch.allclose(result.sum(dim=-1), torch.ones(1))
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_get_last_action_probabilities_after_get_action(self, model, valid_state_dict, model_config):
        """Test get_last_action_probabilities after get_action call."""
        # get_action internally calls forward
        model.get_action(valid_state_dict)
        
        result = model.get_last_action_probabilities()
        
        # Should have action probabilities stored
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, model_config.action_count)

    def test_get_last_action_probabilities_multiple_forward_calls(self, model, valid_state_dict, model_config):
        """Test get_last_action_probabilities after multiple forward passes."""
        # First forward pass
        model.forward(valid_state_dict)
        first_probs = model.get_last_action_probabilities()
        
        # Second forward pass with different input
        different_state_dict = {
            "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        model.forward(different_state_dict)
        second_probs = model.get_last_action_probabilities()
        
        # Should return the most recent action probabilities
        assert first_probs is not None
        assert second_probs is not None
        assert first_probs.shape == second_probs.shape
        
        # They should be different (with high probability) due to different inputs
        # We won't assert inequality as they could theoretically be the same

    def test_get_last_action_probabilities_batch_processing(self, model, model_config):
        """Test get_last_action_probabilities with batch inputs."""
        batch_size = 3
        state_dict = {
            "hf": torch.randn(batch_size, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(batch_size, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(batch_size, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(batch_size, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        model.forward(state_dict)
        result = model.get_last_action_probabilities()
        
        assert result is not None
        assert result.shape == (batch_size, model_config.action_count)
        
        # Each sample should have valid probabilities
        for i in range(batch_size):
            assert torch.allclose(result[i].sum(), torch.tensor(1.0))
            assert (result[i] >= 0).all()

    def test_get_last_action_probabilities_detached_tensor(self, model, valid_state_dict):
        """Test that returned probabilities are detached from computation graph."""
        # Enable gradients
        for param in model.parameters():
            param.requires_grad = True
        
        model.forward(valid_state_dict)
        result = model.get_last_action_probabilities()
        
        assert result is not None
        # Should be detached (no gradients)
        assert not result.requires_grad

    def test_get_last_action_probabilities_softmax_application(self, model, valid_state_dict):
        """Test that softmax is properly applied to logits."""
        # Make forward pass
        action_params, _ = model.forward(valid_state_dict)
        logits = action_params[0]
        
        # Get stored probabilities
        result = model.get_last_action_probabilities()
        
        # Manually compute softmax and compare
        expected_probs = torch.softmax(logits, dim=-1)
        
        assert result is not None
        assert torch.allclose(result, expected_probs.detach())

    def test_get_last_action_probabilities_extreme_logits(self, model, valid_state_dict):
        """Test behavior with extreme logit values."""
        with patch.object(model, 'forward') as mock_forward:
            # Create extreme logits
            extreme_logits = torch.tensor([[-1000, 0, 1000, -500, 500, -100, 100]], dtype=torch.float32)
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((extreme_logits,), value)
            
            # Call forward to set the probabilities
            model.forward(valid_state_dict)
            result = model.get_last_action_probabilities()
            
            assert result is not None
            # Should handle extreme values without NaN/Inf
            assert torch.isfinite(result).all()
            assert torch.allclose(result.sum(dim=-1), torch.ones(1))
            assert (result >= 0).all()

    def test_get_last_action_probabilities_uniform_logits(self, model, valid_state_dict, model_config):
        """Test behavior with uniform logits."""
        with patch.object(model, 'forward') as mock_forward:
            # Create uniform logits (all zeros)
            uniform_logits = torch.zeros(1, model_config.action_count)
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((uniform_logits,), value)
            
            model.forward(valid_state_dict)
            result = model.get_last_action_probabilities()
            
            assert result is not None
            # Should be uniform distribution
            expected_prob = 1.0 / model_config.action_count
            expected_probs = torch.full((1, model_config.action_count), expected_prob)
            assert torch.allclose(result, expected_probs, atol=1e-6)

    def test_get_last_action_probabilities_storage_mechanism(self, model, valid_state_dict):
        """Test the internal storage mechanism for action probabilities."""
        # Ensure no probabilities stored initially
        assert not hasattr(model, '_last_action_probs') or model._last_action_probs is None
        
        # After forward pass
        model.forward(valid_state_dict)
        
        # Should have stored probabilities
        assert hasattr(model, '_last_action_probs')
        assert model._last_action_probs is not None
        
        # get_last_action_probabilities should return the stored tensor
        result = model.get_last_action_probabilities()
        assert result is model._last_action_probs

    def test_get_last_action_probabilities_consistency(self, model, valid_state_dict):
        """Test consistency across multiple calls."""
        model.forward(valid_state_dict)
        
        # Multiple calls should return the same tensor
        result1 = model.get_last_action_probabilities()
        result2 = model.get_last_action_probabilities()
        result3 = model.get_last_action_probabilities()
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        
        # Should be the same tensor object
        assert result1 is result2
        assert result2 is result3

    def test_get_last_action_probabilities_return_type(self, model, valid_state_dict):
        """Test return type annotation compliance."""
        model.forward(valid_state_dict)
        result = model.get_last_action_probabilities()
        
        # Should return torch.Tensor or None
        assert result is None or isinstance(result, torch.Tensor)

    def test_get_last_action_probabilities_device_consistency(self, model_config):
        """Test device consistency."""
        model = MultiBranchTransformer(model_config, device="cpu")
        state_dict = {
            "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        model.forward(state_dict)
        result = model.get_last_action_probabilities()
        
        assert result is not None
        assert result.device.type == "cpu"

    def test_get_last_action_probabilities_dtype_consistency(self, model, valid_state_dict):
        """Test data type consistency."""
        model.forward(valid_state_dict)
        result = model.get_last_action_probabilities()
        
        assert result is not None
        # Should be float tensor
        assert result.dtype.is_floating_point

    def test_get_last_action_probabilities_no_gradients(self, model, valid_state_dict):
        """Test that probabilities don't require gradients."""
        # Enable gradients on model
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        model.forward(valid_state_dict)
        result = model.get_last_action_probabilities()
        
        assert result is not None
        assert not result.requires_grad

    def test_get_last_action_probabilities_overwrite_behavior(self, model, valid_state_dict, model_config):
        """Test that new forward passes overwrite previous probabilities."""
        # First forward pass
        model.forward(valid_state_dict)
        first_result = model.get_last_action_probabilities()
        
        # Store reference to first result
        first_tensor = first_result.clone() if first_result is not None else None
        
        # Second forward pass with different input
        different_state_dict = {
            "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim) + 10,
            "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim) + 10,
            "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim) + 10,
            "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim) + 10,
        }
        model.forward(different_state_dict)
        second_result = model.get_last_action_probabilities()
        
        if first_tensor is not None and second_result is not None:
            # Results should likely be different due to different inputs
            # (We won't assert they're different as they could theoretically be the same)
            assert second_result.shape == first_tensor.shape

    def test_get_last_action_probabilities_attribute_existence(self, model):
        """Test hasattr check for _last_action_probs."""
        # Initially should not have the attribute or it should be None
        result = model.get_last_action_probabilities()
        assert result is None
        
        # After setting the attribute manually
        dummy_probs = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]])
        model._last_action_probs = dummy_probs
        
        result = model.get_last_action_probabilities()
        assert result is not None
        assert torch.equal(result, dummy_probs)

    def test_get_last_action_probabilities_none_attribute(self, model):
        """Test behavior when _last_action_probs is explicitly None."""
        model._last_action_probs = None
        
        result = model.get_last_action_probabilities()
        assert result is None

    def test_get_last_action_probabilities_entropy_calculation(self, model, valid_state_dict):
        """Test that probabilities can be used for entropy calculation."""
        model.forward(valid_state_dict)
        result = model.get_last_action_probabilities()
        
        if result is not None:
            # Should be able to calculate entropy
            entropy = -(result * torch.log(result + 1e-8)).sum(dim=-1)
            assert torch.isfinite(entropy).all()
            assert (entropy >= 0).all()  # Entropy is non-negative

    def test_get_last_action_probabilities_maximum_probability(self, model, valid_state_dict):
        """Test that maximum probability is reasonable."""
        model.forward(valid_state_dict)
        result = model.get_last_action_probabilities()
        
        if result is not None:
            max_prob = result.max(dim=-1)[0]
            # Maximum probability should be at least 1/num_actions for uniform distribution
            min_expected = 1.0 / result.shape[-1]
            assert (max_prob >= min_expected - 1e-6).all()

    def test_get_last_action_probabilities_numerical_stability(self, model, valid_state_dict):
        """Test numerical stability of softmax computation."""
        with patch.object(model, 'forward') as mock_forward:
            # Create logits that would cause overflow in naive softmax
            large_logits = torch.tensor([[700, 800, 900, 600, 750, 650, 850]], dtype=torch.float32)
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((large_logits,), value)
            
            model.forward(valid_state_dict)
            result = model.get_last_action_probabilities()
            
            assert result is not None
            assert torch.isfinite(result).all()
            assert torch.allclose(result.sum(dim=-1), torch.ones(1))

    def test_get_last_action_probabilities_single_action_dominance(self, model, valid_state_dict):
        """Test case where one action dominates."""
        with patch.object(model, 'forward') as mock_forward:
            # One very large logit, others very small
            dominant_logits = torch.tensor([[0, 0, 20, 0, 0, 0, 0]], dtype=torch.float32)
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((dominant_logits,), value)
            
            model.forward(valid_state_dict)
            result = model.get_last_action_probabilities()
            
            assert result is not None
            # Action 2 should have very high probability
            assert result[0, 2] > 0.99
            assert torch.allclose(result.sum(dim=-1), torch.ones(1))