"""
Comprehensive tests for ReplayBuffer.is_ready_for_training method with 100% coverage.
Tests training readiness conditions with various buffer states.
"""

import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferIsReadyForTraining:
    """Test suite for ReplayBuffer.is_ready_for_training method with complete coverage."""

    @pytest.fixture
    def sample_experience_data(self):
        """Create sample data for adding experiences."""
        return {
            "state": {"test": np.array([[1.0]], dtype=np.float32)},
            "action": torch.tensor([0, 1], dtype=torch.int32),
            "reward": 1.0,
            "done": False,
            "action_info": {
                "value": torch.tensor([0.0], dtype=torch.float32),
                "log_prob": torch.tensor([0.0], dtype=torch.float32),
            }
        }

    @pytest.mark.parametrize("capacity", [1, 5, 10, 100, 1000])
    def test_empty_buffer_not_ready(self, capacity):
        """Test that empty buffer is not ready for training."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        assert buffer.is_ready_for_training() == False

    @pytest.mark.parametrize("capacity,num_adds", [
        (5, 1),    # Single experience
        (5, 3),    # Multiple experiences
        (5, 5),    # Full buffer
        (10, 7),   # Partial fill
        (100, 50), # Large buffer, partial fill
    ])
    def test_buffer_ready_with_experiences(self, capacity, num_adds, sample_experience_data):
        """Test that buffer with experiences is ready for training."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        # Add experiences
        for i in range(num_adds):
            buffer.add(
                state_np=sample_experience_data["state"],
                action=sample_experience_data["action"],
                reward=sample_experience_data["reward"],
                next_state_np=sample_experience_data["state"],
                done=sample_experience_data["done"],
                action_info=sample_experience_data["action_info"]
            )
        
        # Should be ready as soon as we have any experiences
        assert buffer.is_ready_for_training() == True

    def test_buffer_ready_after_overflow(self, sample_experience_data):
        """Test buffer readiness after capacity overflow."""
        capacity = 3
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        # Add more experiences than capacity
        for i in range(7):
            buffer.add(
                state_np=sample_experience_data["state"],
                action=sample_experience_data["action"],
                reward=sample_experience_data["reward"],
                next_state_np=sample_experience_data["state"],
                done=sample_experience_data["done"],
                action_info=sample_experience_data["action_info"]
            )
        
        # Should still be ready (has experiences)
        assert buffer.is_ready_for_training() == True
        assert buffer.get_size() == capacity

    def test_buffer_not_ready_after_clear(self, sample_experience_data):
        """Test that buffer is not ready after clearing."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add experiences to make it ready
        for i in range(3):
            buffer.add(
                state_np=sample_experience_data["state"],
                action=sample_experience_data["action"],
                reward=sample_experience_data["reward"],
                next_state_np=sample_experience_data["state"],
                done=sample_experience_data["done"],
                action_info=sample_experience_data["action_info"]
            )
        
        assert buffer.is_ready_for_training() == True
        
        # Clear buffer
        buffer.clear()
        
        # Should not be ready after clear
        assert buffer.is_ready_for_training() == False

    def test_zero_capacity_buffer_readiness(self, sample_experience_data):
        """Test readiness of zero capacity buffer."""
        buffer = ReplayBuffer(capacity=0, device=torch.device("cpu"))
        
        # Should not be ready initially
        assert buffer.is_ready_for_training() == False
        
        # Try to add experience (won't actually be stored due to zero capacity)
        buffer.add(
            state_np=sample_experience_data["state"],
            action=sample_experience_data["action"],
            reward=sample_experience_data["reward"],
            next_state_np=sample_experience_data["state"],
            done=sample_experience_data["done"],
            action_info=sample_experience_data["action_info"]
        )
        
        # Should still not be ready (no experiences stored)
        assert buffer.is_ready_for_training() == False

    def test_single_capacity_buffer_readiness(self, sample_experience_data):
        """Test readiness of single capacity buffer."""
        buffer = ReplayBuffer(capacity=1, device=torch.device("cpu"))
        
        # Should not be ready initially
        assert buffer.is_ready_for_training() == False
        
        # Add one experience
        buffer.add(
            state_np=sample_experience_data["state"],
            action=sample_experience_data["action"],
            reward=sample_experience_data["reward"],
            next_state_np=sample_experience_data["state"],
            done=sample_experience_data["done"],
            action_info=sample_experience_data["action_info"]
        )
        
        # Should be ready with one experience
        assert buffer.is_ready_for_training() == True
        
        # Add another experience (overwrites first)
        buffer.add(
            state_np=sample_experience_data["state"],
            action=sample_experience_data["action"],
            reward=sample_experience_data["reward"],
            next_state_np=sample_experience_data["state"],
            done=sample_experience_data["done"],
            action_info=sample_experience_data["action_info"]
        )
        
        # Should still be ready
        assert buffer.is_ready_for_training() == True

    def test_readiness_consistency_with_size(self, sample_experience_data):
        """Test that readiness is consistent with buffer size > 0."""
        buffer = ReplayBuffer(capacity=10, device=torch.device("cpu"))
        
        # Test at various fill levels
        for i in range(12):  # Go beyond capacity
            # Check consistency before adding
            size = buffer.get_size()
            ready = buffer.is_ready_for_training()
            assert ready == (size > 0)
            
            # Add experience
            buffer.add(
                state_np=sample_experience_data["state"],
                action=sample_experience_data["action"],
                reward=sample_experience_data["reward"],
                next_state_np=sample_experience_data["state"],
                done=sample_experience_data["done"],
                action_info=sample_experience_data["action_info"]
            )
            
            # Check consistency after adding
            size = buffer.get_size()
            ready = buffer.is_ready_for_training()
            assert ready == (size > 0)

    def test_readiness_return_type(self):
        """Test that is_ready_for_training returns a boolean."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        result = buffer.is_ready_for_training()
        assert isinstance(result, bool)

    def test_readiness_immutable_operation(self, sample_experience_data):
        """Test that calling is_ready_for_training doesn't modify buffer state."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        # Add some experiences
        for i in range(3):
            buffer.add(
                state_np=sample_experience_data["state"],
                action=sample_experience_data["action"],
                reward=sample_experience_data["reward"],
                next_state_np=sample_experience_data["state"],
                done=sample_experience_data["done"],
                action_info=sample_experience_data["action_info"]
            )
        
        # Record state before readiness checks
        initial_buffer_len = len(buffer.buffer)
        initial_position = buffer.position
        initial_capacity = buffer.capacity
        
        # Call is_ready_for_training multiple times
        for _ in range(10):
            ready = buffer.is_ready_for_training()
            assert ready == True
        
        # State should be unchanged
        assert len(buffer.buffer) == initial_buffer_len
        assert buffer.position == initial_position
        assert buffer.capacity == initial_capacity

    def test_readiness_with_repeated_clear_and_add(self, sample_experience_data):
        """Test readiness with repeated clear and add operations."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        
        for cycle in range(3):
            # Should not be ready initially
            assert buffer.is_ready_for_training() == False
            
            # Add experiences
            for i in range(3):
                # Check readiness progression
                if i == 0:
                    assert buffer.is_ready_for_training() == False
                
                buffer.add(
                    state_np=sample_experience_data["state"],
                    action=sample_experience_data["action"],
                    reward=sample_experience_data["reward"],
                    next_state_np=sample_experience_data["state"],
                    done=sample_experience_data["done"],
                    action_info=sample_experience_data["action_info"]
                )
                
                # Should be ready after first add
                assert buffer.is_ready_for_training() == True
            
            # Clear buffer
            buffer.clear()
            
            # Should not be ready after clear
            assert buffer.is_ready_for_training() == False

    @pytest.mark.parametrize("capacity", [1, 2, 3, 5, 8, 13, 21, 100])
    def test_readiness_transition_points(self, capacity, sample_experience_data):
        """Test exact transition points from not ready to ready."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        
        # Should not be ready initially
        assert buffer.is_ready_for_training() == False
        
        # Add first experience
        buffer.add(
            state_np=sample_experience_data["state"],
            action=sample_experience_data["action"],
            reward=sample_experience_data["reward"],
            next_state_np=sample_experience_data["state"],
            done=sample_experience_data["done"],
            action_info=sample_experience_data["action_info"]
        )
        
        # Should be ready after first experience
        assert buffer.is_ready_for_training() == True
        
        # Should remain ready as we add more
        for i in range(1, min(capacity * 2, 20)):  # Test some beyond capacity
            buffer.add(
                state_np=sample_experience_data["state"],
                action=sample_experience_data["action"],
                reward=sample_experience_data["reward"],
                next_state_np=sample_experience_data["state"],
                done=sample_experience_data["done"],
                action_info=sample_experience_data["action_info"]
            )
            
            assert buffer.is_ready_for_training() == True

    @pytest.mark.parametrize("device_type", [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")),
    ])
    def test_readiness_with_different_devices(self, device_type, sample_experience_data):
        """Test readiness behavior is consistent across devices."""
        device = torch.device(device_type)
        buffer = ReplayBuffer(capacity=5, device=device)
        
        # Should not be ready initially
        assert buffer.is_ready_for_training() == False
        
        # Add experience
        buffer.add(
            state_np=sample_experience_data["state"],
            action=sample_experience_data["action"],
            reward=sample_experience_data["reward"],
            next_state_np=sample_experience_data["state"],
            done=sample_experience_data["done"],
            action_info=sample_experience_data["action_info"]
        )
        
        # Should be ready regardless of device
        assert buffer.is_ready_for_training() == True