"""
Comprehensive tests for ReplayBuffer.get_size method with 100% coverage.
Tests buffer size tracking with various operations.
"""

import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferGetSize:
    """Test suite for ReplayBuffer.get_size method with complete coverage."""

    @pytest.fixture
    def sample_experience_data(self):
        """Create sample data for adding experiences."""
        return {
            "state": {"test": np.array([[1.0]], dtype=np.float32)}
            "action": torch.tensor([0, 1], dtype=torch.int32)
            "reward": 1.0
            "done": False
            "action_info": {
                "value": torch.tensor([0.0], dtype=torch.float32)
                "log_prob": torch.tensor([0.0], dtype=torch.float32)
            }
        }

    @pytest.mark.parametrize("capacity", [1, 5, 10, 100, 1000])
    def test_empty_buffer_size(self, capacity):
        """Test that newly created buffer has size 0."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        assert buffer.get_size() == 0

    @pytest.mark.parametrize("capacity,num_adds", [
        (5, 1),    # Single add
        (5, 3),    # Multiple adds within capacity
        (5, 5),    # Fill to capacity
        (10, 7),   # Partial fill
        (100, 50), # Large buffer, partial fill
    ])
    def test_size_within_capacity(self, capacity, num_adds, sample_experience_data):
        """Test buffer size when adding experiences within capacity."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        for i in range(num_adds):
            buffer.add(
                state_np=sample_experience_data["state"]
                action=sample_experience_data["action"]
                reward=sample_experience_data["reward"]
                next_state_np=sample_experience_data["state"]
                done=sample_experience_data["done"]
                action_info=sample_experience_data["action_info"]
            )
            
            # Size should increment with each add
            assert buffer.get_size() == i + 1

    @pytest.mark.parametrize("capacity,num_adds", [
        (3, 5),    # Exceed capacity
        (5, 10),   # Double capacity
        (2, 7),    # Multiple wraps
        (1, 5),    # Single slot, multiple overwrites
        (4, 20),   # Many overwrites
    ])
    def test_size_with_overflow(self, capacity, num_adds, sample_experience_data):
        """Test buffer size when adding more experiences than capacity."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        for i in range(num_adds):
            buffer.add(
                state_np=sample_experience_data["state"]
                action=sample_experience_data["action"]
                reward=sample_experience_data["reward"]
                next_state_np=sample_experience_data["state"]
                done=sample_experience_data["done"]
                action_info=sample_experience_data["action_info"]
            )
            
            # Size should not exceed capacity
            expected_size = min(i + 1, capacity)
            assert buffer.get_size() == expected_size

    def test_size_after_clear(self, sample_experience_data):
        """Test buffer size after clearing."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Add some experiences
        for i in range(5):
            buffer.add(
                state_np=sample_experience_data["state"]
                action=sample_experience_data["action"]
                reward=sample_experience_data["reward"]
                next_state_np=sample_experience_data["state"]
                done=sample_experience_data["done"]
                action_info=sample_experience_data["action_info"]
            )
        
        assert buffer.get_size() == 5
        
        # Clear buffer
        buffer.clear()
        
        # Size should be 0 after clear
        assert buffer.get_size() == 0

    def test_size_consistency_with_buffer_length(self, sample_experience_data):
        """Test that get_size() is consistent with len(buffer.buffer)."""
        buffer = ReplayBuffer(capacity=7, device=torch.device("cpu"))
        
        # Test at various fill levels
        for i in range(10):  # Go beyond capacity
            if i > 0:
                buffer.add(
                    state_np=sample_experience_data["state"]
                    action=sample_experience_data["action"]
                    reward=sample_experience_data["reward"]
                    next_state_np=sample_experience_data["state"]
                    done=sample_experience_data["done"]
                    action_info=sample_experience_data["action_info"]
                )
            
            # get_size() should always equal len(buffer.buffer)
            assert buffer.get_size() == len(buffer.buffer)

    def test_zero_capacity_buffer_size(self, sample_experience_data):
        """Test size behavior with zero capacity buffer."""
        buffer = ReplayBuffer(capacity=0, device=torch.device("cpu"))
        
        # Should start with size 0
        assert buffer.get_size() == 0
        
        # Zero capacity buffer should handle gracefully now
        buffer.add(
            state_np=sample_experience_data["state"]
            action=sample_experience_data["action"]
            reward=sample_experience_data["reward"]
            next_state_np=sample_experience_data["state"]
            done=sample_experience_data["done"]
            action_info=sample_experience_data["action_info"]
        )
        
        # Size should remain 0 (nothing added)
        assert buffer.get_size() == 0

    def test_size_with_repeated_clear_and_add(self, sample_experience_data):
        """Test size behavior with repeated clear and add operations."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Repeat clear and add cycle multiple times
        for cycle in range(3):
            # Add experiences
            for i in range(3):
                buffer.add(
                    state_np=sample_experience_data["state"]
                    action=sample_experience_data["action"]
                    reward=sample_experience_data["reward"]
                    next_state_np=sample_experience_data["state"]
                    done=sample_experience_data["done"]
                    action_info=sample_experience_data["action_info"]
                )
            
            assert buffer.get_size() == 3
            
            # Clear buffer
            buffer.clear()
            assert buffer.get_size() == 0

    @pytest.mark.parametrize("capacity", [1, 2, 3, 5, 8, 13, 21])
    def test_size_progression_to_capacity(self, capacity, sample_experience_data):
        """Test size progression from 0 to capacity."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        sizes = []
        
        # Add experiences up to capacity
        for i in range(capacity):
            sizes.append(buffer.get_size())
            
            buffer.add(
                state_np=sample_experience_data["state"]
                action=sample_experience_data["action"]
                reward=sample_experience_data["reward"]
                next_state_np=sample_experience_data["state"]
                done=sample_experience_data["done"]
                action_info=sample_experience_data["action_info"]
            )
        
        # Final size
        sizes.append(buffer.get_size())
        
        # Should progress from 0 to capacity
        expected_sizes = list(range(capacity + 1))
        assert sizes == expected_sizes

    def test_size_return_type(self):
        """Test that get_size returns an integer."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        size = buffer.get_size()
        assert isinstance(size, int)
        assert size >= 0

    def test_size_immutable_operation(self, sample_experience_data):
        """Test that calling get_size doesn't modify buffer state."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add some experiences
        for i in range(3):
            buffer.add(
                state_np=sample_experience_data["state"]
                action=sample_experience_data["action"]
                reward=sample_experience_data["reward"]
                next_state_np=sample_experience_data["state"]
                done=sample_experience_data["done"]
                action_info=sample_experience_data["action_info"]
            )
        
        # Record state before get_size calls
        initial_buffer_len = len(buffer.buffer)
        initial_position = buffer.position
        initial_capacity = buffer.capacity
        
        # Call get_size multiple times
        for _ in range(10):
            size = buffer.get_size()
            assert size == 3
        
        # State should be unchanged
        assert len(buffer.buffer) == initial_buffer_len
        assert buffer.position == initial_position
        assert buffer.capacity == initial_capacity