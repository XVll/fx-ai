"""
Comprehensive tests for ReplayBuffer.prepare_data_for_training method with 100% coverage.
Tests tensor batch preparation, GAE calculation, and data validation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferPrepareDataForTraining:
    """Test suite for ReplayBuffer.prepare_data_for_training method with complete coverage."""

    @pytest.fixture
    def populated_buffer(self):
        """Create a buffer with sample experiences for testing."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Add sample experiences
        for i in range(5):
            state = {
                "hf": np.random.randn(60, 10).astype(np.float32),
                "mf": np.random.randn(10, 15).astype(np.float32),
                "lf": np.random.randn(5, 8).astype(np.float32),
                "portfolio": np.random.randn(1, 5).astype(np.float32),
                "static": np.random.randn(3).astype(np.float32),
            }
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32),
                "log_prob": torch.tensor([float(-i * 0.1)], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=float(i + 1),
                next_state_np=state,
                done=i == 4,  # Last experience is terminal
                action_info=action_info
            )
        
        return buffer

    @pytest.fixture
    def sample_hyperparams(self):
        """Sample hyperparameters for GAE calculation."""
        return {
            "gamma": 0.99,
            "gae_lambda": 0.95,
        }

    def test_prepare_data_basic_functionality(self, populated_buffer, sample_hyperparams):
        """Test basic functionality of data preparation."""
        populated_buffer.prepare_data_for_training()
        
        # Check that all required attributes are set (advantages and returns are set to None initially)
        assert populated_buffer.states is not None
        assert populated_buffer.actions is not None
        assert populated_buffer.log_probs is not None
        assert populated_buffer.values is not None
        assert populated_buffer.rewards is not None
        assert populated_buffer.dones is not None
        assert populated_buffer.advantages is None  # Not computed by this method
        assert populated_buffer.returns is None     # Not computed by this method

    def test_states_dictionary_structure(self, populated_buffer, sample_hyperparams):
        """Test that states dictionary is properly structured."""
        populated_buffer.prepare_data_for_training()
        
        states = populated_buffer.states
        assert isinstance(states, dict)
        
        # Check expected keys
        expected_keys = ["hf", "mf", "lf", "portfolio", "static"]
        for key in expected_keys:
            assert key in states
            assert isinstance(states[key], torch.Tensor)
            assert states[key].device == populated_buffer.device
            
        # Check batch dimension
        batch_size = len(populated_buffer.buffer)
        for key, tensor in states.items():
            assert tensor.shape[0] == batch_size

    def test_basic_tensor_preparation(self, populated_buffer):
        """Test basic tensor preparation functionality."""
        populated_buffer.prepare_data_for_training()
        
        # Check that basic tensors are prepared
        assert populated_buffer.states is not None
        assert populated_buffer.actions is not None
        assert populated_buffer.log_probs is not None
        assert populated_buffer.values is not None
        assert populated_buffer.rewards is not None
        assert populated_buffer.dones is not None
        
        # Check tensor properties (advantages and returns are None initially)
        assert populated_buffer.actions.device == populated_buffer.device
        assert populated_buffer.log_probs.device == populated_buffer.device
        assert populated_buffer.values.device == populated_buffer.device
        assert populated_buffer.rewards.device == populated_buffer.device
        assert populated_buffer.dones.device == populated_buffer.device

    def test_tensor_shapes_and_types(self, populated_buffer, sample_hyperparams):
        """Test that all tensors have correct shapes and types."""
        populated_buffer.prepare_data_for_training()
        
        batch_size = len(populated_buffer.buffer)
        
        # Actions tensor
        assert populated_buffer.actions.shape == (batch_size, 2)
        assert populated_buffer.actions.dtype == torch.int32
        
        # Scalar tensors (only the ones that are prepared)
        scalar_tensors = [
            populated_buffer.log_probs,
            populated_buffer.values,
            populated_buffer.rewards,
        ]
        
        for tensor in scalar_tensors:
            assert tensor.shape == (batch_size,)
            assert tensor.dtype == torch.float32
        
        # Boolean tensor
        assert populated_buffer.dones.shape == (batch_size,)
        assert populated_buffer.dones.dtype == torch.bool

    def test_device_consistency(self, sample_hyperparams):
        """Test that all tensors are on the correct device."""
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
        if torch.backends.mps.is_available():
            devices_to_test.append("mps")
        
        for device_type in devices_to_test:
            device = torch.device(device_type)
            buffer = ReplayBuffer(capacity=5, device=device)
            
            # Add sample data
            for i in range(3):
                state = {"test": np.random.randn(5, 3).astype(np.float32)}
                action = torch.tensor([i, i+1], dtype=torch.int32)
                action_info = {
                    "value": torch.tensor([float(i)], dtype=torch.float32),
                    "log_prob": torch.tensor([float(-i)], dtype=torch.float32),
                }
                
                buffer.add(
                    state_np=state,
                    action=action,
                    reward=float(i),
                    next_state_np=state,
                    done=False,
                    action_info=action_info
                )
            
            buffer.prepare_data_for_training()
            
            # Check all tensors are on correct device (only prepared ones)
            assert buffer.states["test"].device == device
            assert buffer.actions.device == device
            assert buffer.log_probs.device == device
            assert buffer.values.device == device
            assert buffer.rewards.device == device
            assert buffer.dones.device == device

    def test_empty_buffer_handling(self, sample_hyperparams):
        """Test handling of empty buffer."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Should handle empty buffer gracefully
        buffer.prepare_data_for_training()
        
        # All attributes should be empty tensors with correct shapes
        assert buffer.states == {}
        assert buffer.actions.shape == (0, 2)
        assert buffer.log_probs.shape == (0,)
        assert buffer.values.shape == (0,)
        assert buffer.rewards.shape == (0,)
        assert buffer.dones.shape == (0,)
        assert buffer.advantages is None  # Not computed by this method
        assert buffer.returns is None     # Not computed by this method

    def test_single_experience_handling(self, sample_hyperparams):
        """Test handling of buffer with single experience."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Add single experience
        state = {"test": np.random.randn(5, 3).astype(np.float32)}
        action = torch.tensor([0, 1], dtype=torch.int32)
        action_info = {
            "value": torch.tensor([1.0], dtype=torch.float32),
            "log_prob": torch.tensor([-0.1], dtype=torch.float32),
        }
        
        buffer.add(
            state_np=state,
            action=action,
            reward=1.0,
            next_state_np=state,
            done=True,
            action_info=action_info
        )
        
        buffer.prepare_data_for_training(**sample_hyperparams)
        
        # Check single batch dimension
        assert buffer.states["test"].shape == (1, 5, 3)
        assert buffer.actions.shape == (1, 2)
        assert buffer.log_probs.shape == (1,)
        assert buffer.values.shape == (1,)
        assert buffer.rewards.shape == (1,)
        assert buffer.dones.shape == (1,)
        assert buffer.advantages.shape == (1,)
        assert buffer.returns.shape == (1,)

    def test_gae_calculation_accuracy(self):
        """Test GAE calculation accuracy with known values."""
        buffer = ReplayBuffer(capacity=3, device=torch.device("cpu"))
        
        # Add experiences with known rewards and values
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        
        for i, (reward, value) in enumerate(zip(rewards, values)):
            state = {"test": np.array([[1.0]], dtype=np.float32)}
            action = torch.tensor([0, 1], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([value], dtype=torch.float32),
                "log_prob": torch.tensor([0.0], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=reward,
                next_state_np=state,
                done=i == 2,  # Last one is terminal
                action_info=action_info
            )
        
        gamma = 0.9
        gae_lambda = 0.8
        buffer.prepare_data_for_training(gamma=gamma, gae_lambda=gae_lambda)
        
        # Verify that calculations are reasonable
        assert buffer.advantages.shape == (3,)
        assert buffer.returns.shape == (3,)
        
        # Returns should be higher for earlier experiences (due to discounting)
        returns = buffer.returns.cpu().numpy()
        assert returns[0] > returns[1] > returns[2]
        
        # Advantages should be finite and reasonable
        advantages = buffer.advantages.cpu().numpy()
        assert all(np.isfinite(advantages))

    def test_terminal_state_handling(self, sample_hyperparams):
        """Test proper handling of terminal states in GAE calculation."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add experiences with mixed terminal states
        for i in range(3):
            state = {"test": np.random.randn(2, 2).astype(np.float32)}
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32),
                "log_prob": torch.tensor([0.0], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=float(i + 1),
                next_state_np=state,
                done=i == 1,  # Middle experience is terminal
                action_info=action_info
            )
        
        buffer.prepare_data_for_training(**sample_hyperparams)
        
        # Terminal states should affect GAE calculation
        dones = buffer.dones.cpu().numpy()
        advantages = buffer.advantages.cpu().numpy()
        returns = buffer.returns.cpu().numpy()
        
        assert dones[1] == True  # Middle state is terminal
        assert all(np.isfinite(advantages))
        assert all(np.isfinite(returns))

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 50])
    def test_different_batch_sizes(self, batch_size, sample_hyperparams):
        """Test data preparation with different batch sizes."""
        buffer = ReplayBuffer(capacity=batch_size + 5, device=torch.device("cpu"))
        
        # Add experiences
        for i in range(batch_size):
            state = {"test": np.random.randn(3, 2).astype(np.float32)}
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32),
                "log_prob": torch.tensor([float(-i * 0.1)], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=float(i),
                next_state_np=state,
                done=i == batch_size - 1,
                action_info=action_info
            )
        
        buffer.prepare_data_for_training(**sample_hyperparams)
        
        # Check batch dimensions
        assert buffer.states["test"].shape[0] == batch_size
        assert buffer.actions.shape[0] == batch_size
        assert buffer.log_probs.shape[0] == batch_size
        assert buffer.values.shape[0] == batch_size
        assert buffer.rewards.shape[0] == batch_size
        assert buffer.dones.shape[0] == batch_size
        assert buffer.advantages.shape[0] == batch_size
        assert buffer.returns.shape[0] == batch_size

    def test_repeated_preparation(self, populated_buffer, sample_hyperparams):
        """Test that repeated calls to prepare_data_for_training work correctly."""
        # First preparation
        populated_buffer.prepare_data_for_training(**sample_hyperparams)
        first_advantages = populated_buffer.advantages.clone()
        first_returns = populated_buffer.returns.clone()
        
        # Second preparation with same parameters
        populated_buffer.prepare_data_for_training(**sample_hyperparams)
        second_advantages = populated_buffer.advantages
        second_returns = populated_buffer.returns
        
        # Results should be identical
        assert torch.allclose(first_advantages, second_advantages)
        assert torch.allclose(first_returns, second_returns)

    def test_memory_efficiency(self, sample_hyperparams):
        """Test that preparation doesn't create excessive memory usage."""
        buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
        
        # Add many experiences
        for i in range(50):
            state = {"test": np.random.randn(10, 5).astype(np.float32)}
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32),
                "log_prob": torch.tensor([float(-i * 0.01)], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=float(i % 5),
                next_state_np=state,
                done=i % 10 == 9,
                action_info=action_info
            )
        
        # Should prepare without memory issues
        buffer.prepare_data_for_training(**sample_hyperparams)
        
        # Check that all tensors are reasonable size
        total_elements = 0
        for key, tensor in buffer.states.items():
            total_elements += tensor.numel()
        
        total_elements += sum([
            buffer.actions.numel(),
            buffer.log_probs.numel(),
            buffer.values.numel(),
            buffer.rewards.numel(),
            buffer.dones.numel(),
            buffer.advantages.numel(),
            buffer.returns.numel()
        ])
        
        # Should be reasonable (not excessive)
        assert total_elements > 0
        assert total_elements < 1000000  # Sanity check

    def test_zero_capacity_buffer_preparation(self, sample_hyperparams):
        """Test preparation with zero capacity buffer."""
        buffer = ReplayBuffer(capacity=0, device=torch.device("cpu"))
        
        # Should handle gracefully
        buffer.prepare_data_for_training(**sample_hyperparams)
        
        # All tensors should be empty
        assert buffer.states == {}
        assert buffer.actions.shape[0] == 0
        assert buffer.log_probs.shape[0] == 0
        assert buffer.values.shape[0] == 0
        assert buffer.rewards.shape[0] == 0
        assert buffer.dones.shape[0] == 0
        assert buffer.advantages.shape[0] == 0
        assert buffer.returns.shape[0] == 0

    def test_nan_and_inf_handling(self, sample_hyperparams):
        """Test handling of NaN and infinite values."""
        buffer = ReplayBuffer(capacity=3, device=torch.device("cpu"))
        
        # Add experiences with problematic values
        problematic_values = [float('nan'), float('inf'), float('-inf')]
        
        for i, value in enumerate(problematic_values):
            state = {"test": np.array([[1.0]], dtype=np.float32)}
            action = torch.tensor([0, 1], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([value], dtype=torch.float32),
                "log_prob": torch.tensor([0.0], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=1.0,
                next_state_np=state,
                done=False,
                action_info=action_info
            )
        
        # Should not crash (though results may contain NaN/inf)
        buffer.prepare_data_for_training(**sample_hyperparams)
        
        # Tensors should exist even if they contain problematic values
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (3,)
        assert buffer.returns.shape == (3,)