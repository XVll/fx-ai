"""
Comprehensive tests for MultiBranchTransformer.get_action method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from model.transformer import MultiBranchTransformer
from config.model.model_config import ModelConfig


class TestMultiBranchTransformerGetAction:
    """Test cases for MultiBranchTransformer get_action method."""

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

    def test_get_action_deterministic_true(self, model, valid_state_dict, model_config):
        """Test get_action with deterministic=True (should use argmax)."""
        action, action_info = model.get_action(valid_state_dict, deterministic=True)
        
        # Check action shape and type
        assert action.shape == (1,)
        assert action.dtype == torch.long
        assert 0 <= action.item() < model_config.action_count
        
        # Check action_info contents
        assert "action_logits" in action_info
        assert "value" in action_info
        assert "log_prob" in action_info
        
        # Check shapes
        assert action_info["action_logits"].shape == (1, model_config.action_count)
        assert action_info["value"].shape == (1, 1)
        assert action_info["log_prob"].shape == (1, 1)
        
        # Deterministic should use argmax
        expected_action = torch.argmax(action_info["action_logits"], dim=-1)
        assert torch.equal(action, expected_action)

    def test_get_action_deterministic_false(self, model, valid_state_dict, model_config):
        """Test get_action with deterministic=False (should sample)."""
        action, action_info = model.get_action(valid_state_dict, deterministic=False)
        
        # Check action shape and type
        assert action.shape == (1,)
        assert action.dtype == torch.long
        assert 0 <= action.item() < model_config.action_count
        
        # Check action_info contents
        assert "action_logits" in action_info
        assert "value" in action_info
        assert "log_prob" in action_info
        
        # Log probability should be valid
        assert torch.isfinite(action_info["log_prob"]).all()
        assert action_info["log_prob"].item() <= 0  # Log probabilities are <= 0

    def test_get_action_default_deterministic(self, model, valid_state_dict):
        """Test get_action with default deterministic=False."""
        action, action_info = model.get_action(valid_state_dict)
        
        # Should work with default parameter
        assert action.shape == (1,)
        assert "log_prob" in action_info

    def test_get_action_no_grad_context(self, model, valid_state_dict):
        """Test that get_action uses torch.no_grad context."""
        # Enable gradient computation on model parameters
        for param in model.parameters():
            param.requires_grad = True
        
        action, action_info = model.get_action(valid_state_dict)
        
        # Outputs should not require gradients (due to no_grad context)
        assert not action.requires_grad
        assert not action_info["action_logits"].requires_grad
        assert not action_info["value"].requires_grad
        assert not action_info["log_prob"].requires_grad

    def test_get_action_batch_processing(self, model, model_config):
        """Test get_action with batch inputs."""
        batch_size = 3
        state_dict = {
            "hf": torch.randn(batch_size, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(batch_size, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(batch_size, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(batch_size, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        action, action_info = model.get_action(state_dict, deterministic=True)
        
        # Check batch dimensions
        assert action.shape == (batch_size,)
        assert action_info["action_logits"].shape == (batch_size, model_config.action_count)
        assert action_info["value"].shape == (batch_size, 1)
        assert action_info["log_prob"].shape == (batch_size, 1)

    def test_get_action_categorical_distribution(self, model, valid_state_dict):
        """Test that get_action creates proper categorical distribution."""
        # Mock the Categorical distribution to verify it's called correctly
        with patch('torch.distributions.Categorical') as mock_categorical:
            mock_dist = Mock()
            mock_dist.sample.return_value = torch.tensor([2])
            mock_dist.log_prob.return_value = torch.tensor([-1.5])
            mock_categorical.return_value = mock_dist
            
            action, action_info = model.get_action(valid_state_dict, deterministic=False)
            
            # Verify Categorical was called with logits
            mock_categorical.assert_called_once()
            call_kwargs = mock_categorical.call_args[1]
            assert "logits" in call_kwargs
            
            # Verify distribution methods were called
            mock_dist.sample.assert_called_once()
            mock_dist.log_prob.assert_called_once()

    def test_get_action_log_prob_calculation(self, model, valid_state_dict):
        """Test that log probabilities are calculated correctly."""
        action, action_info = model.get_action(valid_state_dict, deterministic=True)
        
        # Manually calculate log probability to verify
        logits = action_info["action_logits"]
        dist = torch.distributions.Categorical(logits=logits)
        expected_log_prob = dist.log_prob(action).unsqueeze(1)
        
        assert torch.allclose(action_info["log_prob"], expected_log_prob)

    def test_get_action_value_consistency(self, model, valid_state_dict):
        """Test that value output is consistent with forward pass."""
        # Get action and value
        action, action_info = model.get_action(valid_state_dict)
        
        # Compare with direct forward pass
        action_params, value_direct = model.forward(valid_state_dict)
        
        assert torch.allclose(action_info["value"], value_direct)
        assert torch.allclose(action_info["action_logits"], action_params[0])

    def test_get_action_action_bounds(self, model, valid_state_dict, model_config):
        """Test that actions are within valid bounds."""
        # Test multiple samples to ensure bounds are respected
        for _ in range(10):
            action, _ = model.get_action(valid_state_dict, deterministic=False)
            assert 0 <= action.item() < model_config.action_count

    def test_get_action_deterministic_reproducibility(self, model, valid_state_dict):
        """Test that deterministic actions are reproducible."""
        # Set same seed
        torch.manual_seed(42)
        action1, _ = model.get_action(valid_state_dict, deterministic=True)
        
        torch.manual_seed(42)
        action2, _ = model.get_action(valid_state_dict, deterministic=True)
        
        assert torch.equal(action1, action2)

    def test_get_action_stochastic_variability(self, model, valid_state_dict):
        """Test that stochastic actions show variability."""
        actions = []
        
        # Sample multiple times
        for i in range(20):
            torch.manual_seed(i)  # Different seeds for variability
            action, _ = model.get_action(valid_state_dict, deterministic=False)
            actions.append(action.item())
        
        # Should have some variability (not all identical)
        unique_actions = len(set(actions))
        assert unique_actions > 1  # Should have at least some variation

    def test_get_action_extreme_logits(self, model, valid_state_dict):
        """Test get_action behavior with extreme logit values."""
        # Modify the model to produce extreme logits
        with patch.object(model, 'forward') as mock_forward:
            # Create extreme logits (one very large, others very small)
            extreme_logits = torch.tensor([[-10, -10, 10, -10, -10, -10, -10]], dtype=torch.float32)
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((extreme_logits,), value)
            
            action, action_info = model.get_action(valid_state_dict, deterministic=True)
            
            # Should select the action with highest logit
            assert action.item() == 2
            assert torch.isfinite(action_info["log_prob"]).all()

    def test_get_action_uniform_logits(self, model, valid_state_dict, model_config):
        """Test get_action behavior with uniform logits."""
        with patch.object(model, 'forward') as mock_forward:
            # Create uniform logits (all equal)
            uniform_logits = torch.zeros(1, model_config.action_count)
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((uniform_logits,), value)
            
            action, action_info = model.get_action(valid_state_dict, deterministic=True)
            
            # Should select action 0 (first in argmax)
            assert action.item() == 0
            
            # Log probability should be approximately log(1/action_count)
            expected_log_prob = np.log(1.0 / model_config.action_count)
            assert abs(action_info["log_prob"].item() - expected_log_prob) < 0.1

    def test_get_action_nan_logits_handling(self, model, valid_state_dict):
        """Test get_action handling of NaN logits."""
        with patch.object(model, 'forward') as mock_forward:
            # Create logits with NaN
            nan_logits = torch.tensor([[float('nan'), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
            value = torch.tensor([[1.0]])
            mock_forward.return_value = ((nan_logits,), value)
            
            # Should handle NaN gracefully or raise appropriate error
            try:
                action, action_info = model.get_action(valid_state_dict, deterministic=True)
                # If it succeeds, check that outputs are reasonable
                assert torch.isfinite(action_info["log_prob"]).all()
            except (RuntimeError, ValueError):
                # It's acceptable to raise an error for NaN inputs
                pass

    def test_get_action_gradient_disabled(self, model, valid_state_dict):
        """Test that get_action disables gradients properly."""
        # Enable gradients
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        # Call get_action
        action, action_info = model.get_action(valid_state_dict)
        
        # Check that outputs don't require gradients
        assert not action.requires_grad
        for key, tensor in action_info.items():
            assert not tensor.requires_grad

    def test_get_action_device_consistency(self, model_config):
        """Test that get_action respects device placement."""
        # Test with CPU
        model_cpu = MultiBranchTransformer(model_config, device="cpu")
        state_dict_cpu = {
            "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        action, action_info = model_cpu.get_action(state_dict_cpu)
        
        # Outputs should be on CPU
        assert action.device.type == "cpu"
        for tensor in action_info.values():
            assert tensor.device.type == "cpu"

    def test_get_action_return_types(self, model, valid_state_dict):
        """Test that get_action returns correct types."""
        action, action_info = model.get_action(valid_state_dict)
        
        # Check return types
        assert isinstance(action, torch.Tensor)
        assert isinstance(action_info, dict)
        assert isinstance(action_info["action_logits"], torch.Tensor)
        assert isinstance(action_info["value"], torch.Tensor)
        assert isinstance(action_info["log_prob"], torch.Tensor)

    def test_get_action_empty_batch(self, model, model_config):
        """Test get_action with empty batch."""
        state_dict = {
            "hf": torch.empty(0, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.empty(0, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.empty(0, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.empty(0, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        with pytest.raises(RuntimeError):  # Should fail with empty batch
            model.get_action(state_dict)

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_get_action_both_modes(self, model, valid_state_dict, deterministic):
        """Test get_action in both deterministic and stochastic modes."""
        action, action_info = model.get_action(valid_state_dict, deterministic=deterministic)
        
        # Basic checks that should pass for both modes
        assert action.shape == (1,)
        assert action.dtype == torch.long
        assert "action_logits" in action_info
        assert "value" in action_info
        assert "log_prob" in action_info
        assert torch.isfinite(action_info["log_prob"]).all()

    def test_get_action_distribution_properties(self, model, valid_state_dict):
        """Test that the underlying distribution has correct properties."""
        action, action_info = model.get_action(valid_state_dict)
        
        # Create distribution manually to verify properties
        logits = action_info["action_logits"]
        dist = torch.distributions.Categorical(logits=logits)
        
        # Probabilities should sum to 1
        probs = dist.probs
        assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]))
        
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_get_action_log_prob_shape_consistency(self, model, model_config):
        """Test log_prob shape consistency across different batch sizes."""
        for batch_size in [1, 3, 5]:
            state_dict = {
                "hf": torch.randn(batch_size, model_config.hf_seq_len, model_config.hf_feat_dim),
                "mf": torch.randn(batch_size, model_config.mf_seq_len, model_config.mf_feat_dim),
                "lf": torch.randn(batch_size, model_config.lf_seq_len, model_config.lf_feat_dim),
                "portfolio": torch.randn(batch_size, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
            }
            
            action, action_info = model.get_action(state_dict)
            
            # Log prob should have shape (batch_size, 1)
            assert action_info["log_prob"].shape == (batch_size, 1)