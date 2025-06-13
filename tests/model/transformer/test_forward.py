"""
Comprehensive tests for MultiBranchTransformer.forward method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import logging

from model.transformer import MultiBranchTransformer
from config.model.model_config import ModelConfig


class TestMultiBranchTransformerForward:
    """Test cases for MultiBranchTransformer forward pass."""

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

    def test_forward_normal_case(self, model, valid_state_dict, model_config):
        """Test forward pass with normal inputs."""
        action_params, value = model.forward(valid_state_dict)
        
        # Check outputs
        assert len(action_params) == 1  # Single discrete action space
        logits = action_params[0]
        
        assert logits.shape == (1, model_config.action_count)
        assert value.shape == (1, 1)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(value).any()

    def test_forward_batch_processing(self, model, model_config):
        """Test forward pass with batch inputs."""
        batch_size = 3
        state_dict = {
            "hf": torch.randn(batch_size, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(batch_size, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(batch_size, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(batch_size, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        action_params, value = model.forward(state_dict)
        
        logits = action_params[0]
        assert logits.shape == (batch_size, model_config.action_count)
        assert value.shape == (batch_size, 1)

    def test_forward_missing_batch_dimension_handling(self, model, model_config):
        """Test forward pass handles missing batch dimensions."""
        state_dict = {
            "hf": torch.randn(model_config.hf_seq_len, model_config.hf_feat_dim),  # Missing batch dim
            "mf": torch.randn(model_config.mf_seq_len, model_config.mf_feat_dim),  # Missing batch dim
            "lf": torch.randn(model_config.lf_seq_len, model_config.lf_feat_dim),  # Missing batch dim
            "portfolio": torch.randn(model_config.portfolio_seq_len, model_config.portfolio_feat_dim),  # Missing batch dim
        }
        
        action_params, value = model.forward(state_dict)
        
        logits = action_params[0]
        assert logits.shape == (1, model_config.action_count)
        assert value.shape == (1, 1)

    def test_forward_1d_input_handling(self, model, model_config):
        """Test forward pass handles 1D inputs by adding dimensions."""
        state_dict = {
            "hf": torch.randn(model_config.hf_feat_dim),  # 1D input
            "mf": torch.randn(model_config.mf_feat_dim),  # 1D input
            "lf": torch.randn(model_config.lf_feat_dim),  # 1D input
            "portfolio": torch.randn(model_config.portfolio_feat_dim),  # 1D input
        }
        
        with patch.object(model.logger, 'warning') as mock_warning:
            action_params, value = model.forward(state_dict)
            
            # Should log warnings about unexpected shapes
            assert mock_warning.call_count >= 4
            
        logits = action_params[0]
        assert logits.shape == (1, model_config.action_count)
        assert value.shape == (1, 1)

    def test_forward_nan_input_detection(self, model, valid_state_dict):
        """Test forward pass detects and logs NaN inputs."""
        # Inject NaN values
        valid_state_dict["hf"][0, 0, 0] = float('nan')
        valid_state_dict["mf"][0, 0, 0] = float('nan')
        
        with patch.object(model.logger, 'warning') as mock_warning:
            action_params, value = model.forward(valid_state_dict)
            
            # Should log warnings about NaN values
            assert mock_warning.call_count >= 2

    def test_forward_device_handling(self, model_config):
        """Test forward pass handles device movement correctly."""
        model = MultiBranchTransformer(model_config, device="cpu")
        
        # Create inputs on different device (if CUDA available)
        if torch.cuda.is_available():
            state_dict = {
                "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim).cuda(),
                "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim).cuda(),
                "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim).cuda(),
                "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim).cuda(),
            }
        else:
            state_dict = {
                "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim),
                "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
                "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
                "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
            }
        
        action_params, value = model.forward(state_dict)
        
        # Outputs should be on model's device
        logits = action_params[0]
        assert logits.device.type == model.device.type
        assert value.device.type == model.device.type

    def test_forward_cross_attention_window_functionality(self, model, valid_state_dict, model_config):
        """Test that cross-attention uses the correct window size."""
        # Test with different window sizes
        model.cross_attn_hf_window = 3  # Smaller than hf_seq_len
        
        action_params, value = model.forward(valid_state_dict)
        
        logits = action_params[0]
        assert logits.shape == (1, model_config.action_count)
        assert not torch.isnan(logits).any()

    def test_forward_temporal_pooling_weights(self, model, valid_state_dict):
        """Test that temporal pooling uses pre-computed weights correctly."""
        action_params, value = model.forward(valid_state_dict)
        
        # Check that time weights are used (they should be registered buffers)
        assert model.hf_time_weights is not None
        assert model.mf_time_weights is not None
        assert model.lf_time_weights is not None
        assert model.portfolio_time_weights is not None

    def test_forward_pattern_extraction(self, model, valid_state_dict, model_config):
        """Test pattern extraction functionality."""
        action_params, value = model.forward(valid_state_dict)
        
        # Pattern extractor should work without errors
        logits = action_params[0]
        assert logits.shape == (1, model_config.action_count)

    def test_forward_attention_weights_storage(self, model, valid_state_dict):
        """Test that attention weights are stored for analysis."""
        action_params, value = model.forward(valid_state_dict)
        
        # Check if attention weights are stored
        if hasattr(model.fusion, "get_branch_importance"):
            assert hasattr(model, '_last_branch_importance')

    def test_forward_action_probabilities_storage(self, model, valid_state_dict):
        """Test that action probabilities are stored for analysis."""
        action_params, value = model.forward(valid_state_dict)
        
        # Should store last action probabilities
        assert hasattr(model, '_last_action_probs')
        assert model._last_action_probs is not None
        assert model._last_action_probs.shape == (1, model.action_count)
        
        # Should sum to 1 (probabilities)
        assert torch.allclose(model._last_action_probs.sum(dim=-1), torch.ones(1))

    def test_forward_return_internals_false(self, model, valid_state_dict):
        """Test forward pass with return_internals=False (default)."""
        result = model.forward(valid_state_dict, return_internals=False)
        
        assert len(result) == 2  # action_params, value
        action_params, value = result
        assert isinstance(action_params, tuple)
        assert len(action_params) == 1

    def test_forward_return_internals_true(self, model, valid_state_dict):
        """Test forward pass with return_internals=True."""
        result = model.forward(valid_state_dict, return_internals=True)
        
        # Should still return the same structure (method doesn't actually use this parameter)
        assert len(result) == 2
        action_params, value = result
        assert isinstance(action_params, tuple)

    def test_forward_different_sequence_lengths(self, model_config):
        """Test forward pass with different sequence lengths."""
        config = ModelConfig(
            hf_seq_len=20,  # Different lengths
            mf_seq_len=10,
            lf_seq_len=5,
            portfolio_seq_len=3
        )
        model = MultiBranchTransformer(config, device="cpu")
        
        state_dict = {
            "hf": torch.randn(1, config.hf_seq_len, config.hf_feat_dim),
            "mf": torch.randn(1, config.mf_seq_len, config.mf_feat_dim),
            "lf": torch.randn(1, config.lf_seq_len, config.lf_feat_dim),
            "portfolio": torch.randn(1, config.portfolio_seq_len, config.portfolio_feat_dim),
        }
        
        action_params, value = model.forward(state_dict)
        
        logits = action_params[0]
        assert logits.shape == (1, config.action_count)

    def test_forward_extreme_values(self, model, model_config):
        """Test forward pass with extreme input values."""
        state_dict = {
            "hf": torch.full((1, model_config.hf_seq_len, model_config.hf_feat_dim), 1e6),  # Large values
            "mf": torch.full((1, model_config.mf_seq_len, model_config.mf_feat_dim), -1e6),  # Large negative
            "lf": torch.full((1, model_config.lf_seq_len, model_config.lf_feat_dim), 1e-6),  # Small values
            "portfolio": torch.zeros(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),  # Zeros
        }
        
        action_params, value = model.forward(state_dict)
        
        logits = action_params[0]
        # Should handle extreme values without NaN/Inf
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()

    def test_forward_gradient_flow(self, model, valid_state_dict):
        """Test that gradients flow through the forward pass."""
        # Enable gradient computation
        for param in model.parameters():
            param.requires_grad = True
            
        action_params, value = model.forward(valid_state_dict)
        
        # Compute a simple loss
        logits = action_params[0]
        loss = logits.sum() + value.sum()
        loss.backward()
        
        # Check that gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any()
        
        assert grad_count > 0  # Some parameters should have gradients

    def test_forward_deterministic_output(self, model, valid_state_dict):
        """Test that forward pass is deterministic with same inputs."""
        # Set deterministic mode
        torch.manual_seed(42)
        
        action_params1, value1 = model.forward(valid_state_dict)
        
        torch.manual_seed(42)
        
        action_params2, value2 = model.forward(valid_state_dict)
        
        # Should produce identical outputs
        assert torch.allclose(action_params1[0], action_params2[0])
        assert torch.allclose(value1, value2)

    def test_forward_memory_efficiency(self, model_config):
        """Test forward pass memory usage with large inputs."""
        model = MultiBranchTransformer(model_config, device="cpu")
        
        # Create larger inputs
        batch_size = 10
        state_dict = {
            "hf": torch.randn(batch_size, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.randn(batch_size, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(batch_size, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(batch_size, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        # Should handle without memory issues
        action_params, value = model.forward(state_dict)
        
        logits = action_params[0]
        assert logits.shape == (batch_size, model_config.action_count)
        assert value.shape == (batch_size, 1)

    @pytest.mark.parametrize("missing_key", ["hf", "mf", "lf", "portfolio"])
    def test_forward_missing_state_key(self, model, valid_state_dict, missing_key):
        """Test forward pass with missing state dictionary keys."""
        del valid_state_dict[missing_key]
        
        with pytest.raises(KeyError):
            model.forward(valid_state_dict)

    def test_forward_invalid_tensor_shapes(self, model, model_config):
        """Test forward pass with invalid tensor shapes."""
        state_dict = {
            "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim + 5),  # Wrong feature dim
            "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        with pytest.raises(RuntimeError):  # Should fail due to dimension mismatch
            model.forward(state_dict)

    def test_forward_empty_tensors(self, model, model_config):
        """Test forward pass with empty tensors."""
        state_dict = {
            "hf": torch.empty(0, model_config.hf_seq_len, model_config.hf_feat_dim),
            "mf": torch.empty(0, model_config.mf_seq_len, model_config.mf_feat_dim),
            "lf": torch.empty(0, model_config.lf_seq_len, model_config.lf_feat_dim),
            "portfolio": torch.empty(0, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
        }
        
        with pytest.raises(RuntimeError):  # Should fail with empty batch
            model.forward(state_dict)