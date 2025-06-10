"""
Comprehensive tests for EpisodeManager.on_update_completed method with 100% coverage.
Tests update counting, state validation, and edge cases.
"""

import pytest
from unittest.mock import Mock
import logging
from typing import Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManager
from config.training.training_config import TrainingManagerConfig


class TestOnUpdateCompleted:
    """Test suite for on_update_completed method with complete coverage."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=TrainingManagerConfig)
        config.day_selection_mode = "sequential"
        config.reset_point_selection_mode = "sequential"
        config.symbols = ["AAPL", "TSLA"]
        config.date_range = ["2024-01-01", "2024-01-31"]
        config.day_score_range = [0.3, 0.9]
        config.reset_roc_range = [0.0, 1.0]
        config.reset_activity_range = [0.0, 1.0]
        config.daily_max_episodes = 10
        config.daily_max_updates = 5
        config.daily_max_cycles = 3
        return config

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock data manager."""
        return Mock()

    @pytest.fixture
    def episode_manager(self, mock_config, mock_data_manager):
        """Create EpisodeManager instance for testing."""
        return EpisodeManager(mock_config, mock_data_manager)

    def test_increments_update_count(self, episode_manager):
        """Test that update count is incremented by 1."""
        initial_count = episode_manager.state.current_day_updates
        
        episode_manager.on_update_completed(None)
        
        assert episode_manager.state.current_day_updates == initial_count + 1

    def test_accepts_any_update_info(self, episode_manager):
        """Test that method accepts any type of update_info parameter."""
        # Test with various types
        update_infos = [
            None,
            {},
            {"loss": 0.5, "accuracy": 0.95},
            "update_string",
            123,
            [1, 2, 3],
            Mock(),
        ]
        
        for info in update_infos:
            initial_count = episode_manager.state.current_day_updates
            episode_manager.on_update_completed(info)
            assert episode_manager.state.current_day_updates == initial_count + 1

    def test_logging_message(self, episode_manager, caplog):
        """Test that debug log message is generated correctly."""
        episode_manager.state.current_day_updates = 5
        
        with caplog.at_level(logging.DEBUG):
            episode_manager.on_update_completed(None)
        
        assert "Update completed, day total: 6" in caplog.text

    def test_negative_count_detection_and_reset(self, episode_manager, caplog):
        """Test that negative update count is detected and reset to 0."""
        # Artificially set negative count
        episode_manager.state.current_day_updates = -5
        
        with caplog.at_level(logging.WARNING):
            episode_manager.on_update_completed(None)
        
        assert "Negative update count detected, resetting" in caplog.text
        assert episode_manager.state.current_day_updates == 0

    def test_multiple_calls_accumulate(self, episode_manager):
        """Test that multiple calls correctly accumulate update count."""
        episode_manager.state.current_day_updates = 0
        
        for i in range(5):
            episode_manager.on_update_completed({"iteration": i})
        
        assert episode_manager.state.current_day_updates == 5

    def test_large_update_count(self, episode_manager):
        """Test handling of large update counts."""
        episode_manager.state.current_day_updates = 999999
        
        episode_manager.on_update_completed(None)
        
        assert episode_manager.state.current_day_updates == 1000000

    def test_state_validation_happens_after_increment(self, episode_manager):
        """Test that state validation happens after incrementing the count."""
        # Set count to -1, which will become 0 after increment
        episode_manager.state.current_day_updates = -1
        
        episode_manager.on_update_completed(None)
        
        # After increment, it becomes 0, which is valid
        assert episode_manager.state.current_day_updates == 0

    def test_extreme_negative_value_reset(self, episode_manager, caplog):
        """Test that extreme negative values are also reset to 0."""
        episode_manager.state.current_day_updates = -999999
        
        with caplog.at_level(logging.WARNING):
            episode_manager.on_update_completed(None)
        
        assert episode_manager.state.current_day_updates == 0

    def test_boundary_case_minus_one(self, episode_manager):
        """Test boundary case where count is -1 before increment."""
        episode_manager.state.current_day_updates = -1
        
        # After increment, becomes 0, no warning should be logged
        episode_manager.on_update_completed(None)
        
        assert episode_manager.state.current_day_updates == 0

    def test_boundary_case_minus_two(self, episode_manager, caplog):
        """Test boundary case where count is -2 before increment."""
        episode_manager.state.current_day_updates = -2
        
        with caplog.at_level(logging.WARNING):
            episode_manager.on_update_completed(None)
        
        # -2 + 1 = -1, which is negative, so reset to 0
        assert episode_manager.state.current_day_updates == 0
        assert "Negative update count detected" in caplog.text

    def test_update_info_not_used_in_logic(self, episode_manager):
        """Test that update_info parameter doesn't affect the counting logic."""
        results = []
        
        # Test with different update_info values
        for info in [None, {"important": True}, {"important": False}]:
            episode_manager.state.current_day_updates = 0
            episode_manager.on_update_completed(info)
            results.append(episode_manager.state.current_day_updates)
        
        # All should result in count of 1
        assert all(r == 1 for r in results)

    def test_no_side_effects_on_other_state(self, episode_manager):
        """Test that only current_day_updates is modified, no other state changes."""
        # Set initial state
        episode_manager.state.current_day_episodes = 10
        episode_manager.state.episodes_in_current_cycle = 5
        episode_manager.state.cycle_count = 3
        
        episode_manager.on_update_completed(None)
        
        # Only current_day_updates should change
        assert episode_manager.state.current_day_episodes == 10
        assert episode_manager.state.episodes_in_current_cycle == 5
        assert episode_manager.state.cycle_count == 3

    @pytest.mark.parametrize("initial_count,expected_after", [
        (0, 1),      # Normal case
        (10, 11),    # Positive count
        (-1, 0),     # Becomes 0 after increment
        (-2, 0),     # Negative after increment, reset to 0
        (-100, 0),   # Large negative, reset to 0
    ])
    def test_various_initial_counts(self, episode_manager, initial_count, expected_after):
        """Test behavior with various initial count values."""
        episode_manager.state.current_day_updates = initial_count
        
        episode_manager.on_update_completed(None)
        
        assert episode_manager.state.current_day_updates == expected_after

    def test_called_with_complex_update_info(self, episode_manager):
        """Test that complex update_info objects are handled gracefully."""
        complex_info = {
            "loss": 0.123,
            "accuracy": 0.95,
            "gradients": [[1, 2, 3], [4, 5, 6]],
            "metadata": {
                "epoch": 10,
                "batch": 100,
                "lr": 0.001
            },
            "model": Mock()
        }
        
        initial_count = episode_manager.state.current_day_updates
        episode_manager.on_update_completed(complex_info)
        
        assert episode_manager.state.current_day_updates == initial_count + 1

    def test_consistency_after_reset(self, episode_manager):
        """Test that after reset, normal increment behavior continues."""
        # Force a reset
        episode_manager.state.current_day_updates = -10
        episode_manager.on_update_completed(None)
        assert episode_manager.state.current_day_updates == 0
        
        # Continue with normal updates
        episode_manager.on_update_completed(None)
        assert episode_manager.state.current_day_updates == 1
        
        episode_manager.on_update_completed(None)
        assert episode_manager.state.current_day_updates == 2

    def test_float_values_in_counter(self, episode_manager):
        """Test behavior when counter has float value (shouldn't happen but test robustness)."""
        episode_manager.state.current_day_updates = 5.5
        
        episode_manager.on_update_completed(None)
        
        assert episode_manager.state.current_day_updates == 6.5

    def test_logging_with_zero_initial_count(self, episode_manager, caplog):
        """Test logging message when starting from zero."""
        episode_manager.state.current_day_updates = 0
        
        with caplog.at_level(logging.DEBUG):
            episode_manager.on_update_completed({"first": True})
        
        assert "Update completed, day total: 1" in caplog.text