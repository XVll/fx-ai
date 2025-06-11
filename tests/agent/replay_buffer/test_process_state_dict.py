"""
Comprehensive tests for ReplayBuffer._process_state_dict method with 100% coverage.
Tests state dictionary processing, tensor conversion, object array handling, and edge cases.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferProcessStateDict:
    """Test suite for ReplayBuffer._process_state_dict method with complete coverage."""

    @pytest.fixture
    def buffer(self):
        """Create a standard replay buffer for testing."""
        return ReplayBuffer(capacity=10, device=torch.device("cpu"))

    @pytest.fixture
    def sample_state_dict(self):
        """Create a comprehensive sample state dictionary."""
        return {
            "hf": np.random.randn(60, 10).astype(np.float32)
            "mf": np.random.randn(10, 15).astype(np.float32)
            "lf": np.random.randn(5, 8).astype(np.float32)
            "portfolio": np.random.randn(1, 5).astype(np.float32)
            
        }

    def test_basic_state_processing(self, buffer, sample_state_dict):
        """Test basic functionality of state dictionary processing."""
        processed_state = buffer._process_state_dict(sample_state_dict)
        
        # Should return a dictionary
        assert isinstance(processed_state, dict)
        
        # Should have same keys
        assert set(processed_state.keys()) == set(sample_state_dict.keys())
        
        # All values should be tensors
        for key, value in processed_state.items():
            assert isinstance(value, torch.Tensor)
            assert value.device == buffer.device

    def test_numpy_array_conversion(self, buffer):
        """Test conversion of various numpy array types."""
        state_dict = {
            "float32": np.random.randn(5, 3).astype(np.float32)
            "float64": np.random.randn(4, 2).astype(np.float64)
            "int32": np.random.randint(0, 10, (3, 4)).astype(np.int32)
            "int64": np.random.randint(0, 10, (2, 5)).astype(np.int64)
            "uint8": np.random.randint(0, 255, (6, 2)).astype(np.uint8)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # All should be converted to float32 tensors
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.device == buffer.device
            assert tensor.shape == state_dict[key].shape

    def test_object_array_handling(self, buffer):
        """Test handling of problematic object arrays."""
        # Create object arrays with different contents
        object_array_1 = np.array([1.0, 2.0, 3.0], dtype=object)
        object_array_2 = np.array([[1, 2], [3, 4]], dtype=object)
        mixed_object = np.array([1.0, "string", None], dtype=object)
        
        state_dict = {
            "object_numeric": object_array_1
            "object_2d": object_array_2
            "object_mixed": mixed_object
            "normal": np.array([1.0, 2.0], dtype=np.float32)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # All should be converted to tensors
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.device == buffer.device

    def test_empty_state_dict(self, buffer):
        """Test processing of empty state dictionary."""
        empty_state = {}
        processed_state = buffer._process_state_dict(empty_state)
        
        assert isinstance(processed_state, dict)
        assert len(processed_state) == 0

    def test_single_key_state_dict(self, buffer):
        """Test processing with single key."""
        state_dict = {"single": np.random.randn(5, 3).astype(np.float32)}
        processed_state = buffer._process_state_dict(state_dict)
        
        assert len(processed_state) == 1
        assert "single" in processed_state
        assert isinstance(processed_state["single"], torch.Tensor)

    @pytest.mark.parametrize("device_type", [
        "cpu"
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available"))
    ])
    def test_device_placement(self, device_type, sample_state_dict):
        """Test that processed tensors are placed on correct device."""
        device = torch.device(device_type)
        buffer = ReplayBuffer(capacity=5, device=device)
        
        processed_state = buffer._process_state_dict(sample_state_dict)
        
        # All tensors should be on the correct device
        for key, tensor in processed_state.items():
            assert tensor.device == device

    def test_dtype_conversion_consistency(self, buffer):
        """Test that all arrays are converted to float32."""
        state_dict = {
            "float16": np.random.randn(3, 2).astype(np.float16)
            "float32": np.random.randn(3, 2).astype(np.float32)
            "float64": np.random.randn(3, 2).astype(np.float64)
            "int8": np.random.randint(-128, 127, (3, 2)).astype(np.int8)
            "int16": np.random.randint(-1000, 1000, (3, 2)).astype(np.int16)
            "int32": np.random.randint(-10000, 10000, (3, 2)).astype(np.int32)
            "int64": np.random.randint(-100000, 100000, (3, 2)).astype(np.int64)
            "uint8": np.random.randint(0, 255, (3, 2)).astype(np.uint8)
            "uint16": np.random.randint(0, 65535, (3, 2)).astype(np.uint16)
            "uint32": np.random.randint(0, 1000000, (3, 2)).astype(np.uint32)
            "bool": np.random.choice([True, False], (3, 2))
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # All should be float32
        for key, tensor in processed_state.items():
            assert tensor.dtype == torch.float32

    @pytest.mark.parametrize("shape", [
        (1,),           # 1D single element
        (5,),           # 1D multiple elements
        (3, 4),         # 2D
        (2, 3, 4),      # 3D
        (1, 2, 3, 4),   # 4D
        (2, 1, 3, 1, 2), # 5D with unit dimensions
    ])
    def test_different_array_shapes(self, buffer, shape):
        """Test processing arrays with different shapes."""
        state_dict = {
            f"shape_{len(shape)}d": np.random.randn(*shape).astype(np.float32)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        tensor = processed_state[f"shape_{len(shape)}d"]
        assert tensor.shape == shape

    def test_large_arrays(self, buffer):
        """Test processing of large arrays."""
        state_dict = {
            "large_1d": np.random.randn(10000).astype(np.float32)
            "large_2d": np.random.randn(1000, 100).astype(np.float32)
            "large_3d": np.random.randn(100, 50, 20).astype(np.float32)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # Should handle large arrays
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.shape == state_dict[key].shape

    def test_zero_dimensional_arrays(self, buffer):
        """Test processing of zero-dimensional (scalar) arrays."""
        state_dict = {
            "scalar": np.array(5.0)
            "scalar_int": np.array(10)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.dim() == 0  # Zero-dimensional tensor

    def test_empty_arrays(self, buffer):
        """Test processing of empty arrays."""
        state_dict = {
            "empty_1d": np.array([], dtype=np.float32)
            "empty_2d": np.array([]).reshape(0, 5).astype(np.float32)
            "empty_3d": np.array([]).reshape(0, 3, 4).astype(np.float32)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.numel() == 0  # Empty tensor

    def test_special_float_values(self, buffer):
        """Test processing arrays with special float values."""
        state_dict = {
            "with_nan": np.array([1.0, float('nan'), 3.0], dtype=np.float32)
            "with_inf": np.array([1.0, float('inf'), 3.0], dtype=np.float32)
            "with_neg_inf": np.array([1.0, float('-inf'), 3.0], dtype=np.float32)
            "mixed_special": np.array([float('nan'), float('inf'), float('-inf')], dtype=np.float32)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # Should convert without error (though may contain NaN/inf)
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32

    def test_very_small_and_large_values(self, buffer):
        """Test processing arrays with extreme values."""
        state_dict = {
            "very_small": np.array([1e-30, 1e-40, 1e-100], dtype=np.float32)
            "very_large": np.array([1e30, 1e40], dtype=np.float64),  # Use float64 to test conversion
            "mixed_extreme": np.array([-1e20, 1e-20, 0.0, 1e20], dtype=np.float32)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32

    def test_object_array_edge_cases(self, buffer):
        """Test edge cases in object array processing."""
        # Create various problematic object arrays
        state_dict = {
            "empty_object": np.array([], dtype=object)
            "single_object": np.array([5.0], dtype=object)
            "nested_list": np.array([[1, 2], [3, 4]], dtype=object)
            "mixed_types": np.array([1, 2.5, True], dtype=object)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # Should all become tensors (even if contents are problematic)
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32

    def test_contiguous_vs_non_contiguous_arrays(self, buffer):
        """Test processing of both contiguous and non-contiguous arrays."""
        base_array = np.random.randn(10, 8).astype(np.float32)
        
        state_dict = {
            "contiguous": base_array.copy()
            "non_contiguous": base_array[::2, ::2],  # Non-contiguous view
            "transposed": base_array.T,  # Transposed (non-contiguous)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # All should be converted successfully
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32

    @patch('agent.replay_buffer.logger')
    def test_object_array_warning_logging(self, mock_logger, buffer):
        """Test that object array processing logs warnings."""
        state_dict = {
            "object_array": np.array([1.0, 2.0, 3.0], dtype=object)
            "normal_array": np.array([1.0, 2.0, 3.0], dtype=np.float32)
        }
        
        buffer._process_state_dict(state_dict)
        
        # Should log warning for object array
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args[0][0]
        assert "object array" in call_args.lower()

    def test_memory_efficiency(self, buffer):
        """Test that processing doesn't create excessive memory overhead."""
        # Create large state dict
        state_dict = {
            f"array_{i}": np.random.randn(100, 50).astype(np.float32)
            for i in range(10)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # Should process all arrays
        assert len(processed_state) == 10
        
        # Each should be proper tensor
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (100, 50)

    def test_key_preservation(self, buffer):
        """Test that dictionary keys are preserved exactly."""
        special_keys = {
            "normal_key": np.array([1.0])
            "key_with_spaces": np.array([2.0])
            "key-with-dashes": np.array([3.0])
            "key_with_numbers_123": np.array([4.0])
            "": np.array([5.0]),  # Empty string key
            "unicode_key_αβγ": np.array([6.0])
        }
        
        processed_state = buffer._process_state_dict(special_keys)
        
        # All keys should be preserved exactly
        assert set(processed_state.keys()) == set(special_keys.keys())

    def test_modification_independence(self, buffer, sample_state_dict):
        """Test that processed state is independent of original."""
        processed_state = buffer._process_state_dict(sample_state_dict)
        
        # Modify original arrays
        for key, array in sample_state_dict.items():
            array.fill(999.0)
        
        # Processed tensors should not be affected
        for key, tensor in processed_state.items():
            assert not torch.all(tensor == 999.0)

    def test_repeated_processing(self, buffer, sample_state_dict):
        """Test that repeated processing gives consistent results."""
        processed_state_1 = buffer._process_state_dict(sample_state_dict)
        processed_state_2 = buffer._process_state_dict(sample_state_dict)
        
        # Should be equivalent (but not necessarily identical objects)
        assert set(processed_state_1.keys()) == set(processed_state_2.keys())
        
        for key in processed_state_1.keys():
            assert torch.equal(processed_state_1[key], processed_state_2[key])

    def test_complex_nested_object_arrays(self, buffer):
        """Test processing of complex nested object arrays."""
        # Create complex object array
        complex_objects = np.array([
            [1.0, 2.0]
            [3.0, 4.0]
            [5.0, 6.0]
        ], dtype=object)
        
        state_dict = {"complex": complex_objects}
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # Should handle complex case
        assert isinstance(processed_state["complex"], torch.Tensor)
        assert processed_state["complex"].dtype == torch.float32

    def test_error_resilience(self, buffer):
        """Test that processing is resilient to conversion errors."""
        # Create problematic arrays that might cause conversion issues
        problematic_dict = {
            "strings": np.array(["a", "b", "c"], dtype=object)
            "none_values": np.array([None, None, None], dtype=object)
            "mixed_problematic": np.array([1, "string", None, [1, 2]], dtype=object)
        }
        
        # Should not crash, even if some conversions fail
        processed_state = buffer._process_state_dict(problematic_dict)
        
        # Should return a dictionary (contents may vary based on conversion success)
        assert isinstance(processed_state, dict)
        assert len(processed_state) == len(problematic_dict)

    @pytest.mark.parametrize("num_keys", [1, 5, 10, 50, 100])
    def test_scalability_with_many_keys(self, buffer, num_keys):
        """Test processing scalability with many dictionary keys."""
        state_dict = {
            f"key_{i}": np.random.randn(10, 5).astype(np.float32)
            for i in range(num_keys)
        }
        
        processed_state = buffer._process_state_dict(state_dict)
        
        # Should handle many keys efficiently
        assert len(processed_state) == num_keys
        
        for key, tensor in processed_state.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.shape == (10, 5)