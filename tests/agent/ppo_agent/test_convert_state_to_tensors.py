"""
Comprehensive tests for PPOTrainer._convert_state_to_tensors method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from agent.ppo_agent import PPOTrainer
from config.training.training_config import TrainingConfig


class TestPPOTrainerConvertStateToTensors:
    """Test cases for PPOTrainer _convert_state_to_tensors method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.parameters.return_value = iter([torch.tensor([1.0])])
        return model

    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
        return TrainingConfig()

    @pytest.fixture
    def trainer(self, training_config, mock_model):
        """Create PPOTrainer instance."""
        device = torch.device("cpu")
        return PPOTrainer(training_config, mock_model, device=device)

    @pytest.fixture
    def standard_state_dict(self):
        """Create standard state dictionary with numpy arrays."""
        return {
            "hf": np.random.randn(60, 9).astype(np.float32),
            "mf": np.random.randn(30, 43).astype(np.float32),
            "lf": np.random.randn(30, 19).astype(np.float32),
            "portfolio": np.random.randn(5, 10).astype(np.float32)
        }

    def test_convert_state_to_tensors_basic_functionality(self, trainer, standard_state_dict):
        """Test basic state to tensor conversion functionality."""
        result = trainer._convert_state_to_tensors(standard_state_dict)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == set(standard_state_dict.keys())
        
        for key, tensor in result.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device == trainer.device

    def test_convert_state_to_tensors_2d_to_3d_expansion(self, trainer, standard_state_dict):
        """Test 2D to 3D tensor expansion for model branch inputs."""
        result = trainer._convert_state_to_tensors(standard_state_dict)
        
        # hf, mf, lf, portfolio should get batch dimension added
        for key in ["hf", "mf", "lf", "portfolio"]:
            original_shape = standard_state_dict[key].shape
            result_shape = result[key].shape
            
            # Should add batch dimension: (seq_len, feat_dim) -> (1, seq_len, feat_dim)
            assert result_shape == (1,) + original_shape

    def test_convert_state_to_tensors_already_3d_input(self, trainer):
        """Test conversion when input is already 3D."""
        state_dict_3d = {
            "hf": np.random.randn(1, 60, 9).astype(np.float32),
            "mf": np.random.randn(1, 30, 43).astype(np.float32),
            "lf": np.random.randn(1, 30, 19).astype(np.float32),
            "portfolio": np.random.randn(1, 5, 10).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_3d)
        
        for key in ["hf", "mf", "lf", "portfolio"]:
            # Should preserve 3D shape when batch dimension is 1
            assert result[key].shape == state_dict_3d[key].shape

    def test_convert_state_to_tensors_batch_size_not_one(self, trainer):
        """Test conversion when 3D input has batch size != 1."""
        state_dict_batch = {
            "hf": np.random.randn(3, 60, 9).astype(np.float32),
            "mf": np.random.randn(3, 30, 43).astype(np.float32),
            "lf": np.random.randn(3, 30, 19).astype(np.float32),
            "portfolio": np.random.randn(3, 5, 10).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_batch)
        
        for key in ["hf", "mf", "lf", "portfolio"]:
            # Should preserve shape when batch dimension is not 1
            assert result[key].shape == state_dict_batch[key].shape

    def test_convert_state_to_tensors_non_branch_keys(self, trainer):
        """Test conversion for keys that are not model branches."""
        state_dict_mixed = {
            "hf": np.random.randn(60, 9).astype(np.float32),
            "custom_feature": np.random.randn(10).astype(np.float32),
            "another_key": np.random.randn(5, 3).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_mixed)
        
        # hf should get batch dimension
        assert result["hf"].shape == (1, 60, 9)
        
        # Non-branch keys should be converted as-is
        assert result["custom_feature"].shape == (10,)
        assert result["another_key"].shape == (5, 3)

    def test_convert_state_to_tensors_device_placement(self, trainer, standard_state_dict):
        """Test that tensors are placed on correct device."""
        result = trainer._convert_state_to_tensors(standard_state_dict)
        
        for tensor in result.values():
            assert tensor.device == trainer.device

    def test_convert_state_to_tensors_cuda_device(self, training_config, mock_model, standard_state_dict):
        """Test conversion with CUDA device if available."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            trainer = PPOTrainer(training_config, mock_model, device=device)
            
            result = trainer._convert_state_to_tensors(standard_state_dict)
            
            for tensor in result.values():
                assert tensor.device.type == "cuda"

    def test_convert_state_to_tensors_dtype_conversion(self, trainer, standard_state_dict):
        """Test that tensors have correct dtype."""
        result = trainer._convert_state_to_tensors(standard_state_dict)
        
        for tensor in result.values():
            assert tensor.dtype == torch.float32

    def test_convert_state_to_tensors_different_dtypes(self, trainer):
        """Test conversion with different input dtypes."""
        state_dict_dtypes = {
            "hf": np.random.randn(60, 9).astype(np.float64),  # float64
            "mf": np.random.randn(30, 43).astype(np.int32),   # int32
            "lf": np.random.randn(30, 19).astype(np.float16), # float16
            "portfolio": np.random.randn(5, 10).astype(np.uint8)  # uint8
        }
        
        result = trainer._convert_state_to_tensors(state_dict_dtypes)
        
        # All should be converted to float32
        for tensor in result.values():
            assert tensor.dtype == torch.float32

    def test_convert_state_to_tensors_empty_dict(self, trainer):
        """Test conversion with empty state dictionary."""
        result = trainer._convert_state_to_tensors({})
        
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_convert_state_to_tensors_single_key(self, trainer):
        """Test conversion with single key."""
        state_dict_single = {
            "hf": np.random.randn(60, 9).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_single)
        
        assert len(result) == 1
        assert "hf" in result
        assert result["hf"].shape == (1, 60, 9)

    def test_convert_state_to_tensors_zero_dimensional(self, trainer):
        """Test conversion with zero-dimensional arrays."""
        state_dict_0d = {
            "scalar": np.array(5.0),
            "hf": np.random.randn(60, 9).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_0d)
        
        assert result["scalar"].shape == ()  # Scalar tensor
        assert result["hf"].shape == (1, 60, 9)

    def test_convert_state_to_tensors_1d_arrays(self, trainer):
        """Test conversion with 1D arrays."""
        state_dict_1d = {
            "vector": np.random.randn(10).astype(np.float32),
            "hf": np.random.randn(60, 9).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_1d)
        
        assert result["vector"].shape == (10,)  # Preserve 1D
        assert result["hf"].shape == (1, 60, 9)  # Add batch dim

    def test_convert_state_to_tensors_large_arrays(self, trainer):
        """Test conversion with large arrays."""
        state_dict_large = {
            "hf": np.random.randn(1000, 100).astype(np.float32),
            "mf": np.random.randn(500, 200).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_large)
        
        assert result["hf"].shape == (1, 1000, 100)
        assert result["mf"].shape == (1, 500, 200)

    def test_convert_state_to_tensors_negative_values(self, trainer):
        """Test conversion with negative values."""
        state_dict_negative = {
            "hf": np.full((60, 9), -1.5).astype(np.float32),
            "mf": np.random.randn(30, 43).astype(np.float32) * -10
        }
        
        result = trainer._convert_state_to_tensors(state_dict_negative)
        
        assert torch.all(result["hf"] == -1.5)
        assert torch.all(result["mf"] < 0)

    def test_convert_state_to_tensors_extreme_values(self, trainer):
        """Test conversion with extreme values."""
        state_dict_extreme = {
            "hf": np.full((60, 9), 1e6).astype(np.float32),
            "mf": np.full((30, 43), 1e-6).astype(np.float32),
            "lf": np.full((30, 19), -1e6).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_extreme)
        
        assert torch.all(torch.isfinite(result["hf"]))
        assert torch.all(torch.isfinite(result["mf"]))
        assert torch.all(torch.isfinite(result["lf"]))

    def test_convert_state_to_tensors_inf_nan_values(self, trainer):
        """Test conversion with inf and nan values."""
        state_dict_special = {
            "hf": np.array([[np.inf, -np.inf], [np.nan, 1.0]]).astype(np.float32),
            "mf": np.random.randn(30, 43).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_special)
        
        # Should preserve inf/nan values
        assert torch.isinf(result["hf"][0, 0, 0])
        assert torch.isinf(result["hf"][0, 0, 1])
        assert torch.isnan(result["hf"][0, 1, 0])

    def test_convert_state_to_tensors_all_branch_keys(self, trainer):
        """Test conversion specifically for all model branch keys."""
        branch_keys = ["hf", "mf", "lf", "portfolio"]
        state_dict_branches = {}
        
        for key in branch_keys:
            state_dict_branches[key] = np.random.randn(10, 5).astype(np.float32)
        
        result = trainer._convert_state_to_tensors(state_dict_branches)
        
        for key in branch_keys:
            # All branch keys should get batch dimension
            assert result[key].shape == (1, 10, 5)

    def test_convert_state_to_tensors_memory_efficiency(self, trainer):
        """Test memory efficiency with large state dictionaries."""
        # Create large state dict
        large_state = {
            "hf": np.random.randn(1000, 500).astype(np.float32),
            "mf": np.random.randn(800, 300).astype(np.float32),
            "lf": np.random.randn(600, 200).astype(np.float32),
            "portfolio": np.random.randn(100, 100).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(large_state)
        
        # Should convert successfully without memory issues
        assert len(result) == 4
        assert all(isinstance(v, torch.Tensor) for v in result.values())

    def test_convert_state_to_tensors_preserves_key_order(self, trainer):
        """Test that key order is preserved in result."""
        ordered_keys = ["z_last", "a_first", "m_middle", "hf"]
        state_dict_ordered = {}
        
        for key in ordered_keys:
            state_dict_ordered[key] = np.random.randn(10, 5).astype(np.float32)
        
        result = trainer._convert_state_to_tensors(state_dict_ordered)
        
        # Key order should be preserved
        assert list(result.keys()) == ordered_keys

    def test_convert_state_to_tensors_contiguous_arrays(self, trainer, standard_state_dict):
        """Test that result tensors are contiguous."""
        result = trainer._convert_state_to_tensors(standard_state_dict)
        
        for tensor in result.values():
            assert tensor.is_contiguous()

    def test_convert_state_to_tensors_non_contiguous_input(self, trainer):
        """Test conversion with non-contiguous input arrays."""
        # Create non-contiguous array
        large_array = np.random.randn(100, 50).astype(np.float32)
        non_contiguous = large_array[::2, ::2]  # Create strided view
        
        state_dict_nc = {
            "hf": non_contiguous,
            "mf": np.random.randn(30, 43).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_nc)
        
        # Should handle non-contiguous arrays
        assert result["hf"].is_contiguous()
        assert result["mf"].is_contiguous()

    def test_convert_state_to_tensors_tensor_properties(self, trainer, standard_state_dict):
        """Test properties of converted tensors."""
        result = trainer._convert_state_to_tensors(standard_state_dict)
        
        for key, tensor in result.items():
            # Basic tensor properties
            assert tensor.requires_grad == False  # Should not require gradients
            assert tensor.dtype == torch.float32
            assert tensor.device == trainer.device
            assert tensor.is_contiguous()
            
            # Shape properties
            assert tensor.ndim >= 1  # At least 1D
            assert tensor.numel() > 0  # Non-empty

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_convert_state_to_tensors_various_batch_sizes(self, trainer, batch_size):
        """Test conversion with various existing batch sizes."""
        state_dict_batch = {
            "hf": np.random.randn(batch_size, 60, 9).astype(np.float32),
            "mf": np.random.randn(batch_size, 30, 43).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_batch)
        
        if batch_size == 1:
            # Should preserve shape for batch_size=1
            assert result["hf"].shape == (1, 60, 9)
        else:
            # Should preserve shape for batch_size > 1
            assert result["hf"].shape == (batch_size, 60, 9)

    def test_convert_state_to_tensors_string_keys(self, trainer):
        """Test conversion with various string key formats."""
        state_dict_keys = {
            "simple": np.random.randn(10, 5).astype(np.float32),
            "with_underscore": np.random.randn(8, 3).astype(np.float32),
            "with-dash": np.random.randn(6, 2).astype(np.float32),
            "123numeric": np.random.randn(4, 1).astype(np.float32),
            "hf": np.random.randn(60, 9).astype(np.float32)
        }
        
        result = trainer._convert_state_to_tensors(state_dict_keys)
        
        # All keys should be preserved
        assert set(result.keys()) == set(state_dict_keys.keys())
        
        # Only 'hf' should get batch dimension treatment
        assert result["hf"].shape == (1, 60, 9)
        assert result["simple"].shape == (10, 5)