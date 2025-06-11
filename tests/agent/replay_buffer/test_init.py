"""
Comprehensive tests for ReplayBuffer.__init__ method with 100% coverage.
Tests initialization, parameter validation, and initial state setup.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agent.replay_buffer import ReplayBuffer


class TestReplayBufferInit:
    """Test suite for ReplayBuffer.__init__ method with complete coverage."""

    @pytest.mark.parametrize("capacity,expected_capacity", [
        (0, 0),                    # Edge case: zero capacity
        (1, 1),                    # Edge case: minimum useful capacity
        (10, 10),                  # Small capacity
        (100, 100),                # Normal capacity
        (1000, 1000),              # Large capacity
        (2048, 2048),              # Power of 2 capacity
        (10000, 10000),            # Very large capacity
        (-1, -1),                  # Edge case: negative capacity
        (-100, -100),              # Edge case: large negative capacity
    ])
    def test_capacity_initialization(self, capacity, expected_capacity):
        """Test initialization with various capacity values."""
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity, device)
        
        assert buffer.capacity == expected_capacity
        assert buffer.device == device
        assert buffer.buffer == []
        assert buffer.position == 0

    @pytest.mark.parametrize("device_spec", [
        "cpu"
        torch.device("cpu")
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))
        pytest.param(torch.device("cuda"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available"))
        pytest.param(torch.device("mps"), marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available"))
    ])
    def test_device_initialization(self, device_spec):
        """Test initialization with various device specifications."""
        capacity = 100
        if isinstance(device_spec, str):
            device = torch.device(device_spec)
        else:
            device = device_spec
            
        buffer = ReplayBuffer(capacity, device)
        
        assert buffer.capacity == capacity
        assert buffer.device == device
        assert len(buffer.buffer) == 0

    def test_all_attributes_initialized_correctly(self):
        """Test that all attributes are properly initialized."""
        capacity = 512
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(capacity, device)
        
        # Core attributes
        assert buffer.capacity == capacity
        assert buffer.device == device
        assert isinstance(buffer.buffer, list)
        assert buffer.buffer == []
        assert buffer.position == 0
        
        # Training data attributes (should all be None initially)
        training_attributes = [
            'states', 'actions', 'log_probs', 'values', 
            'rewards', 'dones', 'advantages', 'returns'
        ]
        
        for attr in training_attributes:
            assert hasattr(buffer, attr)
            assert getattr(buffer, attr) is None

    @pytest.mark.parametrize("capacity,device_type", [
        (1, "cpu")
        (100, "cpu")
        (2048, "cpu")
        pytest.param(1000, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))
        pytest.param(512, "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available"))
    ])
    def test_attribute_types(self, capacity, device_type):
        """Test that all attributes have correct types after initialization."""
        device = torch.device(device_type)
        buffer = ReplayBuffer(capacity, device)
        
        # Type checking
        assert isinstance(buffer.capacity, int)
        assert isinstance(buffer.device, torch.device)
        assert isinstance(buffer.buffer, list)
        assert isinstance(buffer.position, int)
        
        # Tensor attributes should be None or proper type
        tensor_attributes = ['states', 'actions', 'log_probs', 'values', 'rewards', 'dones', 'advantages', 'returns']
        for attr in tensor_attributes:
            value = getattr(buffer, attr)
            if attr == 'states':
                assert value is None or isinstance(value, dict)
            else:
                assert value is None or isinstance(value, torch.Tensor)

    @pytest.mark.parametrize("capacity", [0, 1, 10, 100, 1000])
    def test_position_starts_at_zero(self, capacity):
        """Test that position counter always starts at zero regardless of capacity."""
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity, device)
        
        assert buffer.position == 0

    @pytest.mark.parametrize("capacity", [0, 1, 5, 50, 500, 5000])
    def test_buffer_starts_empty(self, capacity):
        """Test that buffer list always starts empty regardless of capacity."""
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity, device)
        
        assert len(buffer.buffer) == 0
        assert buffer.buffer == []

    def test_initialization_logging(self, caplog):
        """Test that initialization logs appropriate message with correct details."""
        import logging
        
        capacity = 100
        device = torch.device("cpu")
        
        with caplog.at_level(logging.INFO):
            buffer = ReplayBuffer(capacity, device)
        
        log_messages = [record.message for record in caplog.records]
        expected_parts = [
            f"ReplayBuffer initialized with capacity {capacity}"
            f"device {device}"
        ]
        
        # Check that log message contains all expected parts
        found_log = False
        for msg in log_messages:
            if all(part in msg for part in expected_parts):
                found_log = True
                break
        
        assert found_log, f"Expected log message not found. Messages: {log_messages}"

    @pytest.mark.parametrize("capacity,device_str", [
        (64, "cpu")
        (128, "cpu")
        (256, "cpu")
        pytest.param(512, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))
    ])
    def test_device_string_conversion(self, capacity, device_str):
        """Test device string representation is correct."""
        device = torch.device(device_str)
        buffer = ReplayBuffer(capacity, device)
        
        assert str(buffer.device) == device_str
        assert buffer.device.type == device_str

    @pytest.mark.parametrize("extreme_capacity", [
        -2**31,      # Very negative
        -1000000,    # Large negative
        -1,          # Just negative
        2**20,       # 1 Million (large but reasonable)
        2**30,       # 1 Billion (very large)
    ])
    def test_extreme_capacity_values(self, extreme_capacity):
        """Test initialization with extreme capacity values."""
        device = torch.device("cpu")
        
        # Should not crash, even with extreme values
        buffer = ReplayBuffer(extreme_capacity, device)
        
        assert buffer.capacity == extreme_capacity
        assert buffer.device == device
        assert buffer.buffer == []
        assert buffer.position == 0

    def test_multiple_initialization_independence(self):
        """Test that multiple buffer instances are independent."""
        device = torch.device("cpu")
        
        buffer1 = ReplayBuffer(100, device)
        buffer2 = ReplayBuffer(200, device)
        buffer3 = ReplayBuffer(300, device)
        
        # Each should have its own capacity
        assert buffer1.capacity == 100
        assert buffer2.capacity == 200
        assert buffer3.capacity == 300
        
        # Each should have its own buffer list
        assert buffer1.buffer is not buffer2.buffer
        assert buffer2.buffer is not buffer3.buffer
        assert buffer1.buffer is not buffer3.buffer
        
        # Each should have its own position
        assert buffer1.position == 0
        assert buffer2.position == 0
        assert buffer3.position == 0