"""
Comprehensive tests for ReplayBuffer.clear method with 100% coverage.
Tests buffer clearing, memory management, and state reset functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferClear:
    """Test suite for ReplayBuffer.clear method with complete coverage."""

    @pytest.fixture
    def populated_buffer(self):
        """Create a buffer with multiple experiences and prepared data."""
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
                done=i == 4,
                action_info=action_info
            )
        
        # Prepare training data
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        return buffer

    def test_clear_empty_buffer(self):
        """Test clearing an already empty buffer."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Verify initially empty
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
        
        # Clear empty buffer should work without issues
        buffer.clear()
        
        # Should remain empty
        assert len(buffer.buffer) == 0
        assert buffer.position == 0

    def test_clear_populated_buffer_basic(self, populated_buffer):
        """Test basic clearing functionality."""
        # Verify buffer is populated
        assert len(populated_buffer.buffer) > 0
        assert populated_buffer.position > 0
        
        # Clear buffer
        populated_buffer.clear()
        
        # Check basic clearing
        assert len(populated_buffer.buffer) == 0
        assert populated_buffer.position == 0

    def test_clear_resets_all_training_data(self, populated_buffer):
        """Test that clear resets all training data attributes to None."""
        # Verify training data exists
        assert populated_buffer.states is not None
        assert populated_buffer.actions is not None
        assert populated_buffer.log_probs is not None
        assert populated_buffer.values is not None
        assert populated_buffer.rewards is not None
        assert populated_buffer.dones is not None
        assert populated_buffer.advantages is not None
        assert populated_buffer.returns is not None
        
        # Clear buffer
        populated_buffer.clear()
        
        # Check all training data is reset to None
        assert populated_buffer.states is None
        assert populated_buffer.actions is None
        assert populated_buffer.log_probs is None
        assert populated_buffer.values is None
        assert populated_buffer.rewards is None
        assert populated_buffer.dones is None
        assert populated_buffer.advantages is None
        assert populated_buffer.returns is None

    def test_clear_preserves_capacity_and_device(self, populated_buffer):
        """Test that clear preserves capacity and device settings."""
        original_capacity = populated_buffer.capacity
        original_device = populated_buffer.device
        
        # Clear buffer
        populated_buffer.clear()
        
        # Capacity and device should be unchanged
        assert populated_buffer.capacity == original_capacity
        assert populated_buffer.device == original_device

    @pytest.mark.parametrize("capacity", [0, 1, 5, 10, 100, 1000])
    def test_clear_different_capacities(self, capacity):
        """Test clearing buffers with different capacities."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        # Add experiences up to capacity (if capacity > 0)
        if capacity > 0:
            for i in range(min(capacity, 3)):
                state = {"test": np.array([[1.0]], dtype=np.float32)}
                action = torch.tensor([0, 1], dtype=torch.int32)
                action_info = {
                    "value": torch.tensor([1.0], dtype=torch.float32),
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
        
        # Clear buffer
        buffer.clear()
        
        # Should be empty regardless of original capacity
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
        assert buffer.capacity == capacity  # Capacity preserved

    @pytest.mark.parametrize("device_type", [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")),
    ])
    def test_clear_different_devices(self, device_type):
        """Test clearing buffers on different devices."""
        device = torch.device(device_type)
        buffer = ReplayBuffer(capacity=5, device=device)
        
        # Add some experiences
        for i in range(3):
            state = {"test": np.random.randn(2, 2).astype(np.float32)}
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
        
        # Prepare data to have tensors on device
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        
        # Clear buffer
        buffer.clear()
        
        # Device should be preserved
        assert buffer.device == device
        assert len(buffer.buffer) == 0
        assert buffer.position == 0

    def test_clear_after_overflow(self):
        """Test clearing buffer after it has overflowed."""
        capacity = 3
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        # Add more experiences than capacity
        for i in range(7):
            state = {"test": np.array([[float(i)]], dtype=np.float32)}
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
                done=False,
                action_info=action_info
            )
        
        # Buffer should be at capacity with wrapped position
        assert len(buffer.buffer) == capacity
        assert buffer.position == 7 % capacity
        
        # Clear buffer
        buffer.clear()
        
        # Should be completely empty
        assert len(buffer.buffer) == 0
        assert buffer.position == 0

    def test_clear_multiple_times(self, populated_buffer):
        """Test calling clear multiple times."""
        # Clear first time
        populated_buffer.clear()
        assert len(populated_buffer.buffer) == 0
        assert populated_buffer.position == 0
        
        # Clear second time (should not cause issues)
        populated_buffer.clear()
        assert len(populated_buffer.buffer) == 0
        assert populated_buffer.position == 0
        
        # Clear third time
        populated_buffer.clear()
        assert len(populated_buffer.buffer) == 0
        assert populated_buffer.position == 0

    def test_clear_and_reuse_buffer(self):
        """Test that buffer can be reused after clearing."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add initial experiences
        for i in range(3):
            state = {"test": np.array([[float(i)]], dtype=np.float32)}
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
        
        assert len(buffer.buffer) == 3
        
        # Clear buffer
        buffer.clear()
        assert len(buffer.buffer) == 0
        
        # Add new experiences
        for i in range(2):
            state = {"test": np.array([[float(i + 10)]], dtype=np.float32)}
            action = torch.tensor([i + 5, i + 6], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i + 10)], dtype=torch.float32),
                "log_prob": torch.tensor([float(-(i + 10))], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=float(i + 10),
                next_state_np=state,
                done=False,
                action_info=action_info
            )
        
        # Should have new experiences
        assert len(buffer.buffer) == 2
        assert buffer.position == 2

    def test_clear_memory_deallocation(self, populated_buffer):
        """Test that clear properly deallocates memory."""
        # Get initial memory references
        initial_buffer_id = id(populated_buffer.buffer)
        initial_states_id = id(populated_buffer.states) if populated_buffer.states else None
        
        # Clear buffer
        populated_buffer.clear()
        
        # Buffer list should be new empty list
        assert id(populated_buffer.buffer) != initial_buffer_id
        assert populated_buffer.buffer == []
        
        # Training data should be None (deallocated)
        assert populated_buffer.states is None
        assert populated_buffer.actions is None

    def test_clear_with_large_data(self):
        """Test clearing buffer with large amounts of data."""
        buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
        
        # Add many large experiences
        for i in range(50):
            state = {
                "large_array": np.random.randn(100, 50).astype(np.float32),
                "another_large": np.random.randn(200, 25).astype(np.float32),
            }
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([float(i)], dtype=torch.float32),
                "log_prob": torch.tensor([float(-i * 0.01)], dtype=torch.float32),
            }
            
            buffer.add(
                state_np=state,
                action=action,
                reward=float(i),
                next_state_np=state,
                done=i % 10 == 9,
                action_info=action_info
            )
        
        # Prepare large training data
        buffer.prepare_data_for_training(gamma=0.99, gae_lambda=0.95)
        
        # Should have large tensors
        assert len(buffer.buffer) == 50
        assert buffer.states is not None
        
        # Clear should handle large data without issues
        buffer.clear()
        
        # Should be completely clear
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
        assert buffer.states is None

    def test_clear_preserves_get_size_method(self):
        """Test that get_size returns 0 after clear."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add some experiences
        for i in range(3):
            state = {"test": np.array([[1.0]], dtype=np.float32)}
            action = torch.tensor([0, 1], dtype=torch.int32)
            action_info = {
                "value": torch.tensor([1.0], dtype=torch.float32),
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
        
        assert buffer.get_size() == 3
        
        # Clear buffer
        buffer.clear()
        
        # Size should be 0
        assert buffer.get_size() == 0

    def test_clear_affects_is_ready_for_training(self):
        """Test that is_ready_for_training returns False after clear."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add experience to make ready
        state = {"test": np.array([[1.0]], dtype=np.float32)}
        action = torch.tensor([0, 1], dtype=torch.int32)
        action_info = {
            "value": torch.tensor([1.0], dtype=torch.float32),
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
        
        assert buffer.is_ready_for_training() == True
        
        # Clear buffer
        buffer.clear()
        
        # Should not be ready for training
        assert buffer.is_ready_for_training() == False

    def test_clear_invalidates_training_data_access(self, populated_buffer):
        """Test that get_training_data raises error after clear."""
        # Should work before clear
        data = populated_buffer.get_training_data()
        assert data is not None
        
        # Clear buffer
        populated_buffer.clear()
        
        # Should raise error after clear
        with pytest.raises(ValueError, match="Training data not prepared"):
            populated_buffer.get_training_data()

    def test_clear_with_zero_capacity(self):
        """Test clearing zero capacity buffer."""
        buffer = ReplayBuffer(capacity=0, device=torch.device("cpu"))
        
        # Should start empty
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
        
        # Clear should work without issues
        buffer.clear()
        
        # Should remain empty
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
        assert buffer.capacity == 0

    def test_clear_logging_message(self, caplog):
        """Test that clear logs appropriate message."""
        import logging
        
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add some experiences
        state = {"test": np.array([[1.0]], dtype=np.float32)}
        action = torch.tensor([0, 1], dtype=torch.int32)
        action_info = {
            "value": torch.tensor([1.0], dtype=torch.float32),
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
        
        with caplog.at_level(logging.INFO):
            buffer.clear()
        
        # Should log clearing message
        log_messages = [record.message for record in caplog.records]
        found_log = any("cleared" in msg.lower() for msg in log_messages)
        assert found_log, f"Expected clear log message not found. Messages: {log_messages}"

    @pytest.mark.parametrize("num_experiences", [1, 5, 10, 50, 100])
    def test_clear_with_different_fill_levels(self, num_experiences):
        """Test clearing buffers with different fill levels."""
        capacity = max(num_experiences + 10, 20)
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        # Add specified number of experiences
        for i in range(num_experiences):
            state = {"test": np.array([[float(i)]], dtype=np.float32)}
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
                done=False,
                action_info=action_info
            )
        
        # Verify fill level
        assert len(buffer.buffer) == num_experiences
        
        # Clear buffer
        buffer.clear()
        
        # Should be empty regardless of original fill level
        assert len(buffer.buffer) == 0
        assert buffer.position == 0

    def test_clear_state_consistency(self, populated_buffer):
        """Test that buffer state is fully consistent after clear."""
        # Clear buffer
        populated_buffer.clear()
        
        # All state should be consistent with empty buffer
        assert len(populated_buffer.buffer) == 0
        assert populated_buffer.position == 0
        assert populated_buffer.get_size() == 0
        assert populated_buffer.is_ready_for_training() == False
        
        # All training data attributes should be None
        training_attrs = ['states', 'actions', 'log_probs', 'values', 
                         'rewards', 'dones', 'advantages', 'returns']
        for attr in training_attrs:
            assert getattr(populated_buffer, attr) is None