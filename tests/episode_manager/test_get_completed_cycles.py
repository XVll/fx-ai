"""
Tests for EpisodeManager.get_completed_cycles method.
Tests cycle counting and tracking functionality.
"""

import pytest
from unittest.mock import Mock
import pendulum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManager, DayInfo, ResetPointInfo
from config.training.training_config import TrainingManagerConfig


class TestGetCompletedCycles:
    """Test suite for get_completed_cycles method."""

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

    def test_initial_completed_cycles_is_zero(self, episode_manager):
        """Test that initial completed cycles count is zero."""
        assert episode_manager.get_completed_cycles() == 0

    def test_completed_cycles_increases_with_cycle_completion(self, episode_manager):
        """Test that completed cycles count increases when cycles are completed."""
        # Simulate cycle completions by calling reset_for_new_cycle
        assert episode_manager.get_completed_cycles() == 0
        
        episode_manager.state.reset_for_new_cycle()
        assert episode_manager.get_completed_cycles() == 1
        
        episode_manager.state.reset_for_new_cycle()
        assert episode_manager.get_completed_cycles() == 2
        
        episode_manager.state.reset_for_new_cycle()
        assert episode_manager.get_completed_cycles() == 3

    def test_completed_cycles_persists_across_day_resets(self, episode_manager):
        """Test that completed cycles count persists when days are reset."""
        # Complete some cycles
        episode_manager.state.reset_for_new_cycle()
        episode_manager.state.reset_for_new_cycle()
        assert episode_manager.get_completed_cycles() == 2
        
        # Reset for new day (should not affect total cycles)
        episode_manager.state.reset_for_new_day()
        assert episode_manager.get_completed_cycles() == 2
        
        # Complete more cycles
        episode_manager.state.reset_for_new_cycle()
        assert episode_manager.get_completed_cycles() == 3

    def test_completed_cycles_matches_total_cycles_completed(self, episode_manager):
        """Test that get_completed_cycles returns the same as total_cycles_completed."""
        # Verify they start the same
        assert episode_manager.get_completed_cycles() == episode_manager.state.total_cycles_completed
        
        # Complete cycles and verify they stay in sync
        for i in range(1, 6):
            episode_manager.state.reset_for_new_cycle()
            assert episode_manager.get_completed_cycles() == episode_manager.state.total_cycles_completed
            assert episode_manager.get_completed_cycles() == i

    def test_completed_cycles_is_session_wide_counter(self, episode_manager):
        """Test that completed cycles represents session-wide (not daily) counting."""
        # Day 1: 3 cycles
        episode_manager.state.reset_for_new_cycle()  # 1
        episode_manager.state.reset_for_new_cycle()  # 2
        episode_manager.state.reset_for_new_cycle()  # 3
        assert episode_manager.get_completed_cycles() == 3
        
        # Switch to Day 2 (daily cycle counter resets, but total doesn't)
        episode_manager.state.reset_for_new_day()
        episode_manager.state.current_reset_point_cycle = 0  # Day-level reset
        assert episode_manager.get_completed_cycles() == 3  # Session-wide preserved
        
        # Day 2: 2 more cycles
        episode_manager.state.reset_for_new_cycle()  # 4
        episode_manager.state.reset_for_new_cycle()  # 5
        assert episode_manager.get_completed_cycles() == 5

    def test_completed_cycles_return_type(self, episode_manager):
        """Test that get_completed_cycles returns an integer."""
        result = episode_manager.get_completed_cycles()
        assert isinstance(result, int)
        
        episode_manager.state.reset_for_new_cycle()
        result = episode_manager.get_completed_cycles()
        assert isinstance(result, int)

    def test_completed_cycles_never_decreases(self, episode_manager):
        """Test that completed cycles count never decreases."""
        previous_count = episode_manager.get_completed_cycles()
        
        for _ in range(10):
            episode_manager.state.reset_for_new_cycle()
            current_count = episode_manager.get_completed_cycles()
            assert current_count > previous_count
            previous_count = current_count

    def test_completed_cycles_different_from_daily_cycle_count(self, episode_manager):
        """Test that completed cycles is different from daily cycle counter."""
        # Complete some cycles
        episode_manager.state.reset_for_new_cycle()
        episode_manager.state.reset_for_new_cycle()
        
        # Set daily cycle counter to different value
        episode_manager.state.current_reset_point_cycle = 1
        
        # get_completed_cycles should return total, not daily count
        assert episode_manager.get_completed_cycles() == 2
        assert episode_manager.state.current_reset_point_cycle == 1
        assert episode_manager.get_completed_cycles() != episode_manager.state.current_reset_point_cycle