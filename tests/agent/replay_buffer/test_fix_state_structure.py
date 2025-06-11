"""
Quick test to verify the corrected state structure works
"""

import pytest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer

def test_corrected_state_structure():
    """Test that the corrected state structure (no static) works properly."""
    buffer = ReplayBuffer(capacity=5, device=torch.device("cpu"))
    
    # Use correct state structure matching the model
    state = {
        "hf": np.random.randn(60, 10).astype(np.float32),
        "mf": np.random.randn(10, 15).astype(np.float32),
        "lf": np.random.randn(5, 8).astype(np.float32),
        "portfolio": np.random.randn(1, 5).astype(np.float32),
    }
    
    action = torch.tensor([0, 1], dtype=torch.int32)
    action_info = {
        "value": torch.tensor([1.0], dtype=torch.float32),
        "log_prob": torch.tensor([-0.1], dtype=torch.float32),
    }
    
    # Add experiences
    for i in range(3):
        buffer.add(
            state_np=state,
            action=action,
            reward=float(i + 1),
            next_state_np=state,
            done=i == 2,
            action_info=action_info
        )
    
    # Prepare data
    buffer.prepare_data_for_training()
    
    # Verify shapes match model expectations
    expected_shapes = {
        "hf": (3, 60, 10),    # [batch_size, seq_len, feat_dim]
        "mf": (3, 10, 15),
        "lf": (3, 5, 8),
        "portfolio": (3, 1, 5),
    }
    
    assert len(buffer.states) == 4  # No static
    for key, expected_shape in expected_shapes.items():
        assert key in buffer.states
        actual_shape = tuple(buffer.states[key].shape)
        assert actual_shape == expected_shape, f"{key}: got {actual_shape}, expected {expected_shape}"
    
    # Verify other tensors
    assert buffer.actions.shape == (3, 2)
    assert buffer.log_probs.shape == (3,)
    assert buffer.values.shape == (3,)
    assert buffer.rewards.shape == (3,)
    assert buffer.dones.shape == (3,)
    
    print("✅ All tests passed! State structure is correct.")

if __name__ == "__main__":
    test_corrected_state_structure()