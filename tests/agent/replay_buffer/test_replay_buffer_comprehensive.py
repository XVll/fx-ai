"""
Comprehensive tests for ReplayBuffer with corrected state structure.
Tests all major functionality with the current FxAI system architecture.
"""

import pytest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferComprehensive:
    """Comprehensive test suite for ReplayBuffer with corrected state structure."""

    @pytest.fixture
    def buffer(self):
        """Create a standard replay buffer for testing."""
        return ReplayBuffer(capacity=10, device=torch.device("cpu"))

    @pytest.fixture
    def sample_state(self):
        """Create sample state dictionary matching current model architecture."""
        return {
            "hf": np.random.randn(60, 10).astype(np.float32),
            "mf": np.random.randn(10, 15).astype(np.float32),
            "lf": np.random.randn(5, 8).astype(np.float32),
            "portfolio": np.random.randn(1, 5).astype(np.float32),
        }

    @pytest.fixture
    def sample_action_info(self):
        """Create sample action info dictionary."""
        return {
            "value": torch.tensor([0.5], dtype=torch.float32),
            "log_prob": torch.tensor([-0.2], dtype=torch.float32),
        }

    # ===== INITIALIZATION TESTS =====
    def test_init_basic(self):
        """Test basic initialization."""
        buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
        assert buffer.capacity == 5
        assert buffer.device == torch.device("cpu")
        assert len(buffer.buffer) == 0
        assert buffer.position == 0

    @pytest.mark.parametrize("capacity", [0, 1, 5, 100, 1000])
    def test_init_different_capacities(self, capacity):
        """Test initialization with different capacities."""
        buffer = ReplayBuffer(capacity=capacity, device=torch.device("cpu"))
        assert buffer.capacity == capacity
        assert len(buffer.buffer) == 0

    # ===== ADD TESTS =====
    def test_add_single_experience(self, buffer, sample_state, sample_action_info):
        """Test adding a single experience."""
        action = torch.tensor([1, 0], dtype=torch.int32)
        
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        assert len(buffer.buffer) == 1
        assert buffer.position == 1

    def test_add_multiple_experiences(self, buffer, sample_state, sample_action_info):
        """Test adding multiple experiences."""
        for i in range(5):
            action = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.int32)
            
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=float(i),
                next_state_np=sample_state,
                done=i == 4,
                action_info=sample_action_info
            )
        
        assert len(buffer.buffer) == 5
        assert buffer.position == 5

    def test_add_buffer_overflow(self, sample_state, sample_action_info):
        """Test buffer overflow and wrapping."""
        buffer = ReplayBuffer(capacity=3, device=torch.device("cpu"))
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        # Add more than capacity
        for i in range(5):
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=float(i),
                next_state_np=sample_state,
                done=False,
                action_info=sample_action_info
            )
        
        assert len(buffer.buffer) == 3  # Should not exceed capacity
        assert buffer.position == 2  # 5 % 3 = 2

    def test_add_zero_capacity(self, sample_state, sample_action_info):
        """Test adding to zero capacity buffer."""
        buffer = ReplayBuffer(capacity=0, device=torch.device("cpu"))
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        # Should handle gracefully
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        assert len(buffer.buffer) == 0  # Nothing should be added

    # ===== SIZE TESTS =====
    def test_get_size_empty(self, buffer):
        """Test get_size on empty buffer."""
        assert buffer.get_size() == 0

    def test_get_size_with_experiences(self, buffer, sample_state, sample_action_info):
        """Test get_size with experiences."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        for i in range(3):
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=float(i),
                next_state_np=sample_state,
                done=False,
                action_info=sample_action_info
            )
            assert buffer.get_size() == i + 1

    # ===== READY FOR TRAINING TESTS =====
    def test_is_ready_for_training_empty(self, buffer):
        """Test is_ready_for_training on empty buffer."""
        assert buffer.is_ready_for_training() == False

    def test_is_ready_for_training_with_experiences(self, buffer, sample_state, sample_action_info):
        """Test is_ready_for_training with experiences."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        assert buffer.is_ready_for_training() == True

    # ===== PREPARE DATA TESTS =====
    def test_prepare_data_empty_buffer(self, buffer):
        """Test prepare_data_for_training on empty buffer."""
        buffer.prepare_data_for_training()
        
        # Should create empty tensors
        assert buffer.states == {}
        assert buffer.actions.shape == (0, 2)
        assert buffer.log_probs.shape == (0,)
        assert buffer.values.shape == (0,)
        assert buffer.rewards.shape == (0,)
        assert buffer.dones.shape == (0,)

    def test_prepare_data_with_experiences(self, buffer, sample_state, sample_action_info):
        """Test prepare_data_for_training with experiences."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        # Add 3 experiences
        for i in range(3):
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=float(i + 1),
                next_state_np=sample_state,
                done=i == 2,
                action_info=sample_action_info
            )
        
        buffer.prepare_data_for_training()
        
        # Check state shapes match model expectations
        expected_shapes = {
            "hf": (3, 60, 10),
            "mf": (3, 10, 15), 
            "lf": (3, 5, 8),
            "portfolio": (3, 1, 5),
        }
        
        for key, expected_shape in expected_shapes.items():
            assert key in buffer.states
            assert tuple(buffer.states[key].shape) == expected_shape
        
        # Check other tensor shapes
        assert buffer.actions.shape == (3, 2)
        assert buffer.log_probs.shape == (3,)
        assert buffer.values.shape == (3,)
        assert buffer.rewards.shape == (3,)
        assert buffer.dones.shape == (3,)

    # ===== GET TRAINING DATA TESTS =====
    def test_get_training_data_unprepared(self, buffer, sample_state, sample_action_info):
        """Test get_training_data on unprepared buffer."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        # Should raise error
        with pytest.raises(ValueError, match="Training data not prepared"):
            buffer.get_training_data()

    def test_get_training_data_no_advantages(self, buffer, sample_state, sample_action_info):
        """Test get_training_data with prepared data but no advantages."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        buffer.prepare_data_for_training()
        
        # Should return None (advantages not computed)
        data = buffer.get_training_data()
        assert data is None

    def test_get_training_data_complete(self, buffer, sample_state, sample_action_info):
        """Test get_training_data with complete data."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        buffer.prepare_data_for_training()
        
        # Set advantages and returns manually (normally done by PPO agent)
        buffer.advantages = torch.tensor([0.5], dtype=torch.float32)
        buffer.returns = torch.tensor([1.5], dtype=torch.float32)
        
        data = buffer.get_training_data()
        assert data is not None
        assert "states" in data
        assert "actions" in data
        assert "old_log_probs" in data
        assert "advantages" in data
        assert "returns" in data
        assert "values" in data

    # ===== CLEAR TESTS =====
    def test_clear_empty_buffer(self, buffer):
        """Test clearing empty buffer."""
        buffer.clear()
        assert len(buffer.buffer) == 0
        assert buffer.position == 0

    def test_clear_populated_buffer(self, buffer, sample_state, sample_action_info):
        """Test clearing populated buffer."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        # Add experiences and prepare data
        for i in range(3):
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=float(i),
                next_state_np=sample_state,
                done=False,
                action_info=sample_action_info
            )
        
        buffer.prepare_data_for_training()
        
        # Clear buffer
        buffer.clear()
        
        assert len(buffer.buffer) == 0
        assert buffer.position == 0
        assert buffer.states is None
        assert buffer.actions is None
        assert buffer.log_probs is None
        assert buffer.values is None
        assert buffer.rewards is None
        assert buffer.dones is None

    # ===== DEVICE TESTS =====
    @pytest.mark.parametrize("device_type", [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")),
    ])
    def test_device_consistency(self, device_type, sample_state, sample_action_info):
        """Test device consistency across different devices."""
        device = torch.device(device_type)
        buffer = ReplayBuffer(capacity=5, device=device)
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        buffer.add(
            state_np=sample_state,
            action=action,
            reward=1.0,
            next_state_np=sample_state,
            done=False,
            action_info=sample_action_info
        )
        
        buffer.prepare_data_for_training()
        
        # Check all tensors are on correct device (handle device index differences)
        for key, tensor in buffer.states.items():
            assert tensor.device.type == device.type
        
        assert buffer.actions.device.type == device.type
        assert buffer.log_probs.device.type == device.type
        assert buffer.values.device.type == device.type
        assert buffer.rewards.device.type == device.type
        assert buffer.dones.device.type == device.type

    # ===== EDGE CASE TESTS =====
    def test_different_reward_values(self, buffer, sample_state, sample_action_info):
        """Test various reward values."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        rewards = [0.0, 1.0, -1.0, 0.5, -0.75, 100.0, -100.0]
        
        for reward in rewards:
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=reward,
                next_state_np=sample_state,
                done=False,
                action_info=sample_action_info
            )
        
        buffer.prepare_data_for_training()
        
        # Check rewards are stored correctly
        stored_rewards = buffer.rewards.cpu().numpy()
        for i, expected_reward in enumerate(rewards):
            assert abs(stored_rewards[i] - expected_reward) < 1e-6

    def test_different_done_flags(self, buffer, sample_state, sample_action_info):
        """Test different done flags."""
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        done_flags = [True, False, True, False]
        
        for done in done_flags:
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=1.0,
                next_state_np=sample_state,
                done=done,
                action_info=sample_action_info
            )
        
        buffer.prepare_data_for_training()
        
        # Check done flags are stored correctly
        stored_dones = buffer.dones.cpu().numpy()
        for i, expected_done in enumerate(done_flags):
            assert stored_dones[i] == expected_done

    def test_object_array_handling(self, buffer, sample_action_info):
        """Test handling of object arrays in state."""
        # Create state with object array
        problematic_state = {
            "hf": np.random.randn(60, 10).astype(np.float32),
            "mf": np.array([1.0, 2.0, 3.0], dtype=object),  # Object array
            "lf": np.random.randn(5, 8).astype(np.float32),
            "portfolio": np.random.randn(1, 5).astype(np.float32),
        }
        
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        # Should handle gracefully
        buffer.add(
            state_np=problematic_state,
            action=action,
            reward=1.0,
            next_state_np=problematic_state,
            done=False,
            action_info=sample_action_info
        )
        
        buffer.prepare_data_for_training()
        
        # Should convert object array to tensor
        assert isinstance(buffer.states["mf"], torch.Tensor)
        assert buffer.states["mf"].dtype == torch.float32

    def test_large_batch_size(self, sample_state, sample_action_info):
        """Test with large batch size."""
        buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
        action = torch.tensor([0, 1], dtype=torch.int32)
        
        # Add many experiences
        for i in range(50):
            buffer.add(
                state_np=sample_state,
                action=action,
                reward=float(i),
                next_state_np=sample_state,
                done=i % 10 == 9,
                action_info=sample_action_info
            )
        
        buffer.prepare_data_for_training()
        
        # Check shapes
        assert buffer.actions.shape == (50, 2)
        assert buffer.log_probs.shape == (50,)
        for key, tensor in buffer.states.items():
            assert tensor.shape[0] == 50