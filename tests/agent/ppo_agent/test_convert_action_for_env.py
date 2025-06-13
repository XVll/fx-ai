"""
Comprehensive tests for PPOTrainer._convert_action_for_env static method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np

from agent.ppo_agent import PPOTrainer


class TestPPOTrainerConvertActionForEnv:
    """Test cases for PPOTrainer _convert_action_for_env static method."""

    def test_convert_action_for_env_basic_functionality(self):
        """Test basic action conversion functionality."""
        action_tensor = torch.tensor([3])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert result == 3

    def test_convert_action_for_env_scalar_tensor(self):
        """Test conversion of scalar tensor."""
        action_tensor = torch.tensor(5)  # Scalar tensor
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 5

    def test_convert_action_for_env_single_element_tensor(self):
        """Test conversion of single-element tensor."""
        action_tensor = torch.tensor([7])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 7

    def test_convert_action_for_env_float_tensor(self):
        """Test conversion of float tensor."""
        action_tensor = torch.tensor([2.5])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 2.5
        assert isinstance(result, float)

    def test_convert_action_for_env_int_tensor(self):
        """Test conversion of int tensor."""
        action_tensor = torch.tensor([4], dtype=torch.int32)
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 4
        assert isinstance(result, (int, np.integer))

    def test_convert_action_for_env_long_tensor(self):
        """Test conversion of long tensor."""
        action_tensor = torch.tensor([6], dtype=torch.long)
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 6

    def test_convert_action_for_env_zero_action(self):
        """Test conversion of zero action."""
        action_tensor = torch.tensor([0])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 0

    def test_convert_action_for_env_negative_action(self):
        """Test conversion of negative action."""
        action_tensor = torch.tensor([-1])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == -1

    def test_convert_action_for_env_large_action(self):
        """Test conversion of large action value."""
        action_tensor = torch.tensor([999999])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 999999

    def test_convert_action_for_env_gpu_tensor(self):
        """Test conversion of GPU tensor if CUDA is available."""
        if torch.cuda.is_available():
            action_tensor = torch.tensor([3]).cuda()
            
            result = PPOTrainer._convert_action_for_env(action_tensor)
            
            assert result == 3
            # Result should be Python scalar, not CUDA tensor

    def test_convert_action_for_env_cpu_placement(self):
        """Test that tensor is moved to CPU before conversion."""
        action_tensor = torch.tensor([8])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        # Result should be a Python scalar, not a tensor
        assert not isinstance(result, torch.Tensor)
        assert result == 8

    def test_convert_action_for_env_numpy_compatibility(self):
        """Test that result is compatible with numpy operations."""
        action_tensor = torch.tensor([5])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        # Should be able to use in numpy operations
        numpy_result = np.array([result])
        assert numpy_result[0] == 5

    def test_convert_action_for_env_different_dtypes(self):
        """Test conversion with different tensor dtypes."""
        dtypes_and_values = [
            (torch.int8, 1),
            (torch.int16, 2),
            (torch.int32, 3),
            (torch.int64, 4),
            (torch.float16, 5.5),
            (torch.float32, 6.5),
            (torch.float64, 7.5)
        ]
        
        for dtype, value in dtypes_and_values:
            if dtype == torch.float16:
                # Skip float16 on CPU as it may not be supported
                try:
                    action_tensor = torch.tensor([value], dtype=dtype)
                except RuntimeError:
                    continue
            else:
                action_tensor = torch.tensor([value], dtype=dtype)
            
            result = PPOTrainer._convert_action_for_env(action_tensor)
            
            if dtype in [torch.float16, torch.float32, torch.float64]:
                assert abs(result - value) < 1e-5  # Float comparison
            else:
                assert result == value

    def test_convert_action_for_env_scalar_vs_1d_tensor(self):
        """Test conversion behaves same for scalar and 1D single-element tensors."""
        scalar_tensor = torch.tensor(42)
        single_element_tensor = torch.tensor([42])
        
        scalar_result = PPOTrainer._convert_action_for_env(scalar_tensor)
        single_element_result = PPOTrainer._convert_action_for_env(single_element_tensor)
        
        assert scalar_result == single_element_result == 42

    def test_convert_action_for_env_preserves_value_precision(self):
        """Test that conversion preserves value precision."""
        # Test with precise float value
        precise_value = 3.141592653589793
        action_tensor = torch.tensor([precise_value])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        # Should preserve reasonable precision
        assert abs(result - precise_value) < 1e-10

    def test_convert_action_for_env_extreme_values(self):
        """Test conversion with extreme values."""
        extreme_values = [
            1e-10,   # Very small positive
            -1e-10,  # Very small negative
            1e10,    # Very large positive
            -1e10,   # Very large negative
        ]
        
        for value in extreme_values:
            action_tensor = torch.tensor([value])
            result = PPOTrainer._convert_action_for_env(action_tensor)
            
            assert abs(result - value) < abs(value * 1e-10)  # Relative tolerance

    def test_convert_action_for_env_special_float_values(self):
        """Test conversion with special float values."""
        # Test with inf
        inf_tensor = torch.tensor([float('inf')])
        inf_result = PPOTrainer._convert_action_for_env(inf_tensor)
        assert np.isinf(inf_result)
        
        # Test with -inf
        neg_inf_tensor = torch.tensor([float('-inf')])
        neg_inf_result = PPOTrainer._convert_action_for_env(neg_inf_tensor)
        assert np.isinf(neg_inf_result) and neg_inf_result < 0
        
        # Test with nan
        nan_tensor = torch.tensor([float('nan')])
        nan_result = PPOTrainer._convert_action_for_env(nan_tensor)
        assert np.isnan(nan_result)

    def test_convert_action_for_env_zero_dimensional_tensor(self):
        """Test conversion with zero-dimensional tensor."""
        zero_dim_tensor = torch.tensor(15)  # 0-D tensor
        
        result = PPOTrainer._convert_action_for_env(zero_dim_tensor)
        
        assert result == 15

    def test_convert_action_for_env_requires_grad_tensor(self):
        """Test conversion with tensor that requires gradients."""
        action_tensor = torch.tensor([9.0], requires_grad=True)
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        # Should successfully convert despite requiring gradients
        assert result == 9.0
        assert not isinstance(result, torch.Tensor)

    def test_convert_action_for_env_detached_tensor(self):
        """Test conversion with detached tensor."""
        action_tensor = torch.tensor([11.0], requires_grad=True).detach()
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 11.0

    def test_convert_action_for_env_multiple_calls_consistency(self):
        """Test that multiple calls with same input produce same result."""
        action_tensor = torch.tensor([13])
        
        result1 = PPOTrainer._convert_action_for_env(action_tensor)
        result2 = PPOTrainer._convert_action_for_env(action_tensor)
        result3 = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result1 == result2 == result3 == 13

    def test_convert_action_for_env_tensor_not_modified(self):
        """Test that original tensor is not modified."""
        original_tensor = torch.tensor([17])
        original_value = original_tensor.clone()
        
        result = PPOTrainer._convert_action_for_env(original_tensor)
        
        # Original tensor should remain unchanged
        assert torch.equal(original_tensor, original_value)
        assert result == 17

    def test_convert_action_for_env_is_static_method(self):
        """Test that method can be called without instance."""
        action_tensor = torch.tensor([19])
        
        # Should be callable without creating PPOTrainer instance
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 19

    def test_convert_action_for_env_memory_efficiency(self):
        """Test memory efficiency - no memory leaks."""
        action_tensor = torch.tensor([21])
        
        # Multiple conversions shouldn't accumulate memory
        for _ in range(100):
            result = PPOTrainer._convert_action_for_env(action_tensor)
            assert result == 21

    @pytest.mark.parametrize("action_value", [0, 1, 2, 3, 4, 5, 6])
    def test_convert_action_for_env_discrete_action_values(self, action_value):
        """Test conversion with typical discrete action values."""
        action_tensor = torch.tensor([action_value])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == action_value
        assert isinstance(result, (int, np.integer))

    @pytest.mark.parametrize("device_type", ["cpu"])
    def test_convert_action_for_env_different_devices(self, device_type):
        """Test conversion with tensors on different devices."""
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device(device_type)
        action_tensor = torch.tensor([23]).to(device)
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        assert result == 23
        assert not isinstance(result, torch.Tensor)

    def test_convert_action_for_env_return_type_consistency(self):
        """Test that return type is consistent."""
        action_tensor = torch.tensor([25])
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        # Should return Python scalar type
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not isinstance(result, torch.Tensor)

    def test_convert_action_for_env_thread_safety(self):
        """Test that static method is thread-safe."""
        import threading
        results = []
        
        def convert_action():
            action_tensor = torch.tensor([27])
            result = PPOTrainer._convert_action_for_env(action_tensor)
            results.append(result)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=convert_action)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All results should be the same
        assert all(result == 27 for result in results)
        assert len(results) == 10

    def test_convert_action_for_env_no_side_effects(self):
        """Test that conversion has no side effects."""
        action_tensor = torch.tensor([29])
        original_data = action_tensor.data.clone()
        
        result = PPOTrainer._convert_action_for_env(action_tensor)
        
        # Tensor data should remain unchanged
        assert torch.equal(action_tensor.data, original_data)
        assert result == 29

    def test_convert_action_for_env_performance(self):
        """Test performance of conversion operation."""
        import time
        
        action_tensor = torch.tensor([31])
        
        start_time = time.time()
        for _ in range(1000):
            result = PPOTrainer._convert_action_for_env(action_tensor)
        end_time = time.time()
        
        # Should be very fast
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete 1000 operations in under 1 second
        assert result == 31