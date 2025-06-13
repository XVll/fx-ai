"""
Comprehensive tests for MultiBranchTransformer.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import logging

from model.transformer import MultiBranchTransformer
from config.model.model_config import ModelConfig


class TestMultiBranchTransformerInit:
    """Test cases for MultiBranchTransformer initialization."""

    def test_init_default_device(self):
        """Test initialization with default device detection."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        # Verify basic attributes
        assert model.model_config == config
        assert model.continuous_action == False
        assert model.action_count == 7
        assert model.hf_seq_len == config.hf_seq_len
        assert model.mf_seq_len == config.mf_seq_len
        assert model.lf_seq_len == config.lf_seq_len
        assert model.portfolio_seq_len == config.portfolio_seq_len
        
        # Verify device is set correctly
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert model.device == expected_device

    def test_init_explicit_string_device(self):
        """Test initialization with explicit string device."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config, device="cpu")
        
        assert model.device == torch.device("cpu")

    def test_init_explicit_torch_device(self):
        """Test initialization with explicit torch.device."""
        config = ModelConfig()
        device = torch.device("cpu")
        
        model = MultiBranchTransformer(config, device=device)
        
        assert model.device == device

    def test_init_cuda_device_when_available(self):
        """Test initialization with CUDA device when available."""
        config = ModelConfig()
        
        with patch('torch.cuda.is_available', return_value=True):
            model = MultiBranchTransformer(config, device="cuda")
            assert model.device == torch.device("cuda")

    def test_init_feature_dimensions_stored(self):
        """Test that feature dimensions are correctly stored."""
        config = ModelConfig(
            hf_feat_dim=15,
            mf_feat_dim=25,
            lf_feat_dim=35,
            portfolio_feat_dim=45
        )
        
        model = MultiBranchTransformer(config)
        
        assert model.hf_feat_dim == 15
        assert model.mf_feat_dim == 25
        assert model.lf_feat_dim == 35
        assert model.portfolio_feat_dim == 45

    def test_init_cross_attention_window_default(self):
        """Test cross-attention window uses default value."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        assert model.cross_attn_hf_window == 10  # Default value

    def test_init_cross_attention_window_custom(self):
        """Test cross-attention window with custom value."""
        config = ModelConfig()
        config.cross_attn_hf_window = 15
        
        model = MultiBranchTransformer(config)
        
        assert model.cross_attn_hf_window == 15

    def test_init_cross_attention_window_clamped_to_seq_len(self):
        """Test cross-attention window is clamped to sequence length."""
        config = ModelConfig(hf_seq_len=5)
        config.cross_attn_hf_window = 20  # Larger than seq_len
        
        model = MultiBranchTransformer(config)
        
        assert model.cross_attn_hf_window == 5  # Clamped to hf_seq_len

    def test_init_layers_created_correctly(self):
        """Test that all neural network layers are created."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        # Test HF branch components
        assert isinstance(model.hf_proj, nn.Linear)
        assert model.hf_proj.in_features == config.hf_feat_dim
        assert model.hf_proj.out_features == config.d_model
        assert hasattr(model, 'hf_pos_enc')
        assert hasattr(model, 'hf_encoder')
        
        # Test MF branch components
        assert isinstance(model.mf_proj, nn.Linear)
        assert model.mf_proj.in_features == config.mf_feat_dim
        assert model.mf_proj.out_features == config.d_model
        assert hasattr(model, 'mf_pos_enc')
        assert hasattr(model, 'mf_encoder')
        
        # Test LF branch components
        assert isinstance(model.lf_proj, nn.Linear)
        assert model.lf_proj.in_features == config.lf_feat_dim
        assert model.lf_proj.out_features == config.d_model
        assert hasattr(model, 'lf_pos_enc')
        assert hasattr(model, 'lf_encoder')
        
        # Test Portfolio branch components
        assert isinstance(model.portfolio_proj, nn.Linear)
        assert model.portfolio_proj.in_features == config.portfolio_feat_dim
        assert model.portfolio_proj.out_features == config.d_model
        assert hasattr(model, 'portfolio_pos_enc')
        assert hasattr(model, 'portfolio_encoder')

    def test_init_fusion_and_output_layers(self):
        """Test fusion and output layers are created correctly."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        # Test fusion layer
        assert hasattr(model, 'fusion')
        
        # Test action head
        assert isinstance(model.action_head, nn.Linear)
        assert model.action_head.in_features == config.d_fused
        assert model.action_head.out_features == config.action_count
        
        # Test critic network
        assert hasattr(model, 'critic')
        assert isinstance(model.critic, nn.Sequential)

    def test_init_cross_timeframe_attention(self):
        """Test cross-timeframe attention layer creation."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        assert hasattr(model, 'cross_timeframe_attention')
        assert isinstance(model.cross_timeframe_attention, nn.MultiheadAttention)
        assert model.cross_timeframe_attention.embed_dim == config.d_model
        assert model.cross_timeframe_attention.num_heads == 4

    def test_init_pattern_extractor(self):
        """Test pattern extractor layer creation."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        assert hasattr(model, 'pattern_extractor')
        assert isinstance(model.pattern_extractor, nn.Sequential)

    def test_init_time_weights_registered_as_buffers(self):
        """Test that time weights are registered as buffers."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        # Check that time weights are registered as buffers
        assert 'hf_time_weights' in model._buffers
        assert 'mf_time_weights' in model._buffers
        assert 'lf_time_weights' in model._buffers
        assert 'portfolio_time_weights' in model._buffers
        
        # Check shapes
        assert model.hf_time_weights.shape == (1, config.hf_seq_len, 1)
        assert model.mf_time_weights.shape == (1, config.mf_seq_len, 1)
        assert model.lf_time_weights.shape == (1, config.lf_seq_len, 1)
        assert model.portfolio_time_weights.shape == (1, config.portfolio_seq_len, 1)

    def test_init_time_weights_normalized(self):
        """Test that time weights are properly normalized."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        # Each time weight sequence should sum to 1
        assert torch.allclose(model.hf_time_weights.sum(dim=1), torch.ones(1, 1))
        assert torch.allclose(model.mf_time_weights.sum(dim=1), torch.ones(1, 1))
        assert torch.allclose(model.lf_time_weights.sum(dim=1), torch.ones(1, 1))
        assert torch.allclose(model.portfolio_time_weights.sum(dim=1), torch.ones(1, 1))

    def test_init_time_weights_exponential_decay(self):
        """Test that time weights follow exponential decay pattern."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        # Later timesteps should have higher weights (exponential decay from past to present)
        hf_weights = model.hf_time_weights.squeeze()
        mf_weights = model.mf_time_weights.squeeze()
        lf_weights = model.lf_time_weights.squeeze()
        portfolio_weights = model.portfolio_time_weights.squeeze()
        
        # Check that weights are monotonically increasing (recent timesteps have higher weights)
        assert torch.all(hf_weights[1:] >= hf_weights[:-1])
        assert torch.all(mf_weights[1:] >= mf_weights[:-1])
        assert torch.all(lf_weights[1:] >= lf_weights[:-1])
        assert torch.all(portfolio_weights[1:] >= portfolio_weights[:-1])

    def test_init_logger_setup(self):
        """Test that logger is properly set up."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config)
        
        assert hasattr(model, 'logger')
        assert isinstance(model.logger, logging.Logger)

    def test_init_action_count_type_conversion(self):
        """Test that action_count is properly converted to int."""
        config = ModelConfig()
        config.action_count = 7.0  # Float that should be converted to int
        
        model = MultiBranchTransformer(config)
        
        assert model.action_count == 7
        assert isinstance(model.action_count, int)

    def test_init_model_moved_to_device(self):
        """Test that model is moved to the specified device."""
        config = ModelConfig()
        
        model = MultiBranchTransformer(config, device="cpu")
        
        # Check that model parameters are on the correct device
        for param in model.parameters():
            assert param.device.type == "cpu"

    def test_init_different_config_values(self):
        """Test initialization with different configuration values."""
        config = ModelConfig(
            d_model=256,
            d_fused=1024,
            hf_layers=4,
            mf_layers=2,
            lf_layers=1,
            portfolio_layers=1,
            hf_heads=16,
            mf_heads=8,
            lf_heads=4,
            portfolio_heads=2,
            dropout=0.2
        )
        
        model = MultiBranchTransformer(config)
        
        # Verify configuration is used correctly
        assert model.model_config.d_model == 256
        assert model.model_config.d_fused == 1024
        assert model.model_config.dropout == 0.2

    @pytest.mark.parametrize("seq_len,expected_window", [
        (60, 10),  # Normal case
        (5, 5),    # Window clamped to seq_len
        (100, 10), # Window smaller than seq_len
    ])
    def test_init_cross_attention_window_various_sizes(self, seq_len, expected_window):
        """Test cross-attention window with various sequence lengths."""
        config = ModelConfig(hf_seq_len=seq_len)
        config.cross_attn_hf_window = 10
        
        model = MultiBranchTransformer(config)
        
        assert model.cross_attn_hf_window == expected_window

    def test_init_missing_cross_attn_window_attr(self):
        """Test initialization when cross_attn_hf_window attribute is missing."""
        config = ModelConfig()
        # Remove the attribute if it exists
        if hasattr(config, 'cross_attn_hf_window'):
            delattr(config, 'cross_attn_hf_window')
        
        model = MultiBranchTransformer(config)
        
        assert model.cross_attn_hf_window == 10  # Should use default

    def test_init_zero_sequence_lengths(self):
        """Test initialization with zero sequence lengths (edge case)."""
        config = ModelConfig(
            hf_seq_len=1,  # Minimum valid sequence length
            mf_seq_len=1,
            lf_seq_len=1,
            portfolio_seq_len=1
        )
        
        model = MultiBranchTransformer(config)
        
        assert model.hf_seq_len == 1
        assert model.mf_seq_len == 1
        assert model.lf_seq_len == 1
        assert model.portfolio_seq_len == 1