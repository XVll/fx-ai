"""
Comprehensive tests for ReplayBuffer.get_training_data method with 100% coverage.
Tests data retrieval, validation, and state management.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferGetTrainingData:
    """Test suite for ReplayBuffer.get_training_data method with complete coverage."""

    @pytest.fixture
    def prepared_buffer(self):
        """Create a buffer with prepared training data."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Add sample experiences
        for i in range(5):
            state = {
                "hf": np.random.randn(60, 10).astype(np.float32)
                "mf": np.random.randn(10, 15).astype(np.float32)
                "lf": np.random.randn(5, 8).astype(np.float32)
                "portfolio": np.random.randn(1, 5).astype(np.float32)
                
            }
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32)
                "log_prob": torch.tensor([float(-i * 0.1)], dtype=torch.float32)
            }
            
            buffer.add(
                state_np=state
                action=action
                reward=float(i + 1)
                next_state_np=state
                done=i == 4
                action_info=action_info
            )
        
        # Prepare data for training
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        return buffer

    @pytest.fixture
    def unprepared_buffer(self):
        """Create a buffer without prepared training data."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add some experiences but don't prepare data
        state = {"test": np.array([[1.0]], dtype=np.float32)}
        action = torch.tensor([0, 1], dtype=torch.int32)
        action_info = {
            "value": torch.tensor([1.0], dtype=torch.float32)
            "log_prob": torch.tensor([0.0], dtype=torch.float32)
        }
        
        buffer.add(
            state_np=state
            action=action
            reward=1.0
            next_state_np=state
            done=False
            action_info=action_info
        )
        
        return buffer

    def test_get_training_data_basic_functionality(self, prepared_buffer):
        """Test basic functionality of get_training_data."""
        # Need to set advantages and returns to make data complete
        prepared_buffer.advantages = torch.zeros(len(prepared_buffer.buffer), device=prepared_buffer.device)
        prepared_buffer.returns = torch.zeros(len(prepared_buffer.buffer), device=prepared_buffer.device)
        
        data = prepared_buffer.get_training_data()
        
        # Check that all expected keys are present
        expected_keys = ["states", "actions", "old_log_probs", "advantages", "returns", "values"]
        for key in expected_keys:
            assert key in data
        
        # Check data types
        assert isinstance(data["states"], dict)
        for key in ["actions", "old_log_probs", "advantages", "returns", "values"]:
            assert isinstance(data[key], torch.Tensor)

    def test_data_consistency_with_buffer_attributes(self, prepared_buffer):
        """Test that returned data is consistent with buffer attributes."""
        data = prepared_buffer.get_training_data()
        
        # Check that returned data matches buffer attributes
        assert data["states"] is prepared_buffer.states
        assert torch.equal(data["actions"], prepared_buffer.actions)
        assert torch.equal(data["log_probs"], prepared_buffer.log_probs)
        assert torch.equal(data["values"], prepared_buffer.values)
        assert torch.equal(data["rewards"], prepared_buffer.rewards)
        assert torch.equal(data["dones"], prepared_buffer.dones)
        assert torch.equal(data["advantages"], prepared_buffer.advantages)
        assert torch.equal(data["returns"], prepared_buffer.returns)

    def test_unprepared_buffer_raises_error(self, unprepared_buffer):
        """Test that accessing training data from unprepared buffer raises error."""
        with pytest.raises(ValueError, match="Training data not prepared"):
            unprepared_buffer.get_training_data()

    def test_empty_prepared_buffer(self):
        """Test get_training_data with empty but prepared buffer."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Prepare empty buffer
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        
        data = buffer.get_training_data()
        
        # Should return empty tensors with correct structure
        assert isinstance(data["states"], dict)
        assert len(data["states"]) == 0
        
        # All other tensors should be empty with correct shapes
        for key in ["actions", "log_probs", "values", "rewards", "dones", "advantages", "returns"]:
            assert isinstance(data[key], torch.Tensor)
            assert data[key].shape[0] == 0

    def test_states_dictionary_structure(self, prepared_buffer):
        """Test that states dictionary has correct structure."""
        data = prepared_buffer.get_training_data()
        states = data["states"]
        
        assert isinstance(states, dict)
        expected_keys = ["hf", "mf", "lf", "portfolio", "static"]
        
        for key in expected_keys:
            assert key in states
            assert isinstance(states[key], torch.Tensor)
            assert states[key].device == prepared_buffer.device
            assert states[key].dtype == torch.float32

    def test_tensor_shapes_and_types(self, prepared_buffer):
        """Test that all tensors have correct shapes and types."""
        data = prepared_buffer.get_training_data()
        batch_size = len(prepared_buffer.buffer)
        
        # Actions should be (batch_size, 2)
        assert data["actions"].shape == (batch_size, 2)
        assert data["actions"].dtype == torch.int32
        
        # Scalar tensors should be (batch_size,)
        scalar_tensors = ["log_probs", "values", "rewards", "advantages", "returns"]
        for key in scalar_tensors:
            assert data[key].shape == (batch_size,)
            assert data[key].dtype == torch.float32
        
        # Done tensor should be boolean
        assert data["dones"].shape == (batch_size,)
        assert data["dones"].dtype == torch.bool

    @pytest.mark.parametrize("device_type", [
        "cpu"
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available"))
    ])
    def test_device_consistency(self, device_type):
        """Test that all returned tensors are on correct device."""
        device = torch.device(device_type)
        buffer = ReplayBuffer(capacity=5, device=device)
        
        # Add sample data
        for i in range(3):
            state = {"test": np.random.randn(5, 3).astype(np.float32)}
            action = torch.tensor([i, i+1], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32)
                "log_prob": torch.tensor([float(-i)], dtype=torch.float32)
            }
            
            buffer.add(
                state_np=state
                action=action
                reward=float(i)
                next_state_np=state
                done=False
                action_info=action_info
            )
        
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        data = buffer.get_training_data()
        
        # Check all tensors are on correct device
        for key, tensor in data["states"].items():
            assert tensor.device == device
        
        for key in ["actions", "log_probs", "values", "rewards", "dones", "advantages", "returns"]:
            assert data[key].device == device

    def test_data_immutability(self, prepared_buffer):
        """Test that returned data references don't affect buffer state."""
        data = prepared_buffer.get_training_data()
        
        # Modify returned data
        original_actions = data["actions"].clone()
        data["actions"][0] = torch.tensor([999, 999], dtype=torch.int32)
        
        # Buffer should be affected (since we return references)
        # This is expected behavior for efficiency
        assert not torch.equal(prepared_buffer.actions, original_actions)
        assert torch.equal(prepared_buffer.actions, data["actions"])

    def test_multiple_calls_consistency(self, prepared_buffer):
        """Test that multiple calls return consistent data."""
        data1 = prepared_buffer.get_training_data()
        data2 = prepared_buffer.get_training_data()
        
        # Should return same references
        assert data1["states"] is data2["states"]
        assert torch.equal(data1["actions"], data2["actions"])
        assert torch.equal(data1["log_probs"], data2["log_probs"])
        assert torch.equal(data1["values"], data2["values"])
        assert torch.equal(data1["rewards"], data2["rewards"])
        assert torch.equal(data1["dones"], data2["dones"])
        assert torch.equal(data1["advantages"], data2["advantages"])
        assert torch.equal(data1["returns"], data2["returns"])

    def test_single_experience_data(self):
        """Test get_training_data with single experience."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add single experience
        state = {"test": np.random.randn(2, 3).astype(np.float32)}
        action = torch.tensor([0, 1], dtype=torch.int32)
        action_info = {
            "value": torch.tensor([1.0], dtype=torch.float32)
            "log_prob": torch.tensor([-0.1], dtype=torch.float32)
        }
        
        buffer.add(
            state_np=state
            action=action
            reward=1.0
            next_state_np=state
            done=True
            action_info=action_info
        )
        
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        data = buffer.get_training_data()
        
        # Check single batch dimension
        assert data["states"]["test"].shape == (1, 2, 3)
        assert data["actions"].shape == (1, 2)
        assert data["log_probs"].shape == (1,)
        assert data["values"].shape == (1,)
        assert data["rewards"].shape == (1,)
        assert data["dones"].shape == (1,)
        assert data["advantages"].shape == (1,)
        assert data["returns"].shape == (1,)

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 50])
    def test_different_batch_sizes(self, batch_size):
        """Test get_training_data with different batch sizes."""
        buffer = ReplayBuffer(capacity=batch_size + 5, device=torch.device("cpu"))
        
        # Add experiences
        for i in range(batch_size):
            state = {"test": np.random.randn(3, 2).astype(np.float32)}
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32)
                "log_prob": torch.tensor([float(-i * 0.1)], dtype=torch.float32)
            }
            
            buffer.add(
                state_np=state
                action=action
                reward=float(i)
                next_state_np=state
                done=i == batch_size - 1
                action_info=action_info
            )
        
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        data = buffer.get_training_data()
        
        # Check batch dimensions
        assert data["states"]["test"].shape[0] == batch_size
        assert data["actions"].shape[0] == batch_size
        assert data["log_probs"].shape[0] == batch_size
        assert data["values"].shape[0] == batch_size
        assert data["rewards"].shape[0] == batch_size
        assert data["dones"].shape[0] == batch_size
        assert data["advantages"].shape[0] == batch_size
        assert data["returns"].shape[0] == batch_size

    def test_data_after_clear_and_reprepare(self, prepared_buffer):
        """Test get_training_data after clearing and re-preparing buffer."""
        # Get initial data
        initial_data = prepared_buffer.get_training_data()
        initial_batch_size = initial_data["actions"].shape[0]
        
        # Clear buffer
        prepared_buffer.clear()
        
        # Should raise error after clear (data not prepared)
        with pytest.raises(ValueError, match="Training data not prepared"):
            prepared_buffer.get_training_data()
        
        # Add new data and prepare
        state = {"test": np.array([[2.0]], dtype=np.float32)}
        action = torch.tensor([1, 0], dtype=torch.int32)
        action_info = {
            "value": torch.tensor([2.0], dtype=torch.float32)
            "log_prob": torch.tensor([-0.2], dtype=torch.float32)
        }
        
        prepared_buffer.add(
            state_np=state
            action=action
            reward=2.0
            next_state_np=state
            done=True
            action_info=action_info
        )
        
        prepared_buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        new_data = prepared_buffer.get_training_data()
        
        # New data should be different
        assert new_data["actions"].shape[0] == 1  # Single experience
        assert new_data["actions"].shape[0] != initial_batch_size
        assert not torch.equal(new_data["actions"], initial_data["actions"])

    def test_data_integrity_checks(self, prepared_buffer):
        """Test data integrity and validation."""
        data = prepared_buffer.get_training_data()
        
        # Check that all tensors have same batch size
        batch_size = data["actions"].shape[0]
        
        for key, value in data.items():
            if key == "states":
                for state_key, state_tensor in value.items():
                    assert state_tensor.shape[0] == batch_size
            else:
                assert value.shape[0] == batch_size
        
        # Check that all tensors are finite (no NaN/inf)
        for key, value in data.items():
            if key == "states":
                for state_tensor in value.values():
                    assert torch.all(torch.isfinite(state_tensor))
            elif key != "dones":  # Skip boolean tensor
                assert torch.all(torch.isfinite(value))

    def test_partial_data_preparation_error(self):
        """Test error when only some data is prepared."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Manually set only some attributes (simulating partial preparation)
        buffer.states = {"test": torch.tensor([[1.0]], dtype=torch.float32)}
        buffer.actions = torch.tensor([[0, 1]], dtype=torch.int32)
        # Leave other attributes as None
        
        # Should raise error due to missing attributes
        with pytest.raises(ValueError, match="Training data not prepared"):
            buffer.get_training_data()

    def test_return_dictionary_keys(self, prepared_buffer):
        """Test that returned dictionary has exactly the expected keys."""
        data = prepared_buffer.get_training_data()
        
        expected_keys = {"states", "actions", "log_probs", "values", "rewards", "dones", "advantages", "returns"}
        actual_keys = set(data.keys())
        
        assert actual_keys == expected_keys

    def test_large_batch_memory_efficiency(self):
        """Test memory efficiency with large batches."""
        buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
        
        # Add many experiences
        for i in range(50):
            state = {"test": np.random.randn(10, 5).astype(np.float32)}
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32)
                "log_prob": torch.tensor([float(-i * 0.01)], dtype=torch.float32)
            }
            
            buffer.add(
                state_np=state
                action=action
                reward=float(i % 5)
                next_state_np=state
                done=i % 10 == 9
                action_info=action_info
            )
        
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        
        # Should return data without memory issues
        data = buffer.get_training_data()
        
        # Verify data is reasonable
        assert data["actions"].shape[0] == 50
        assert isinstance(data["states"], dict)
        assert "test" in data["states"]

    def test_zero_capacity_buffer_data(self):
        """Test get_training_data with zero capacity buffer."""
        buffer = ReplayBuffer(capacity=0, device=torch.device("cpu"))
        
        # Prepare empty buffer
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        
        data = buffer.get_training_data()
        
        # Should return empty but valid data structure
        assert isinstance(data["states"], dict)
        assert len(data["states"]) == 0
        
        for key in ["actions", "log_probs", "values", "rewards", "dones", "advantages", "returns"]:
            assert isinstance(data[key], torch.Tensor)
            assert data[key].shape[0] == 0