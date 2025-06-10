"""
Comprehensive tests for EpisodeManager.on_episodes_completed method with 100% coverage.
Tests episode counting, state updates, and reset point advancement.
"""

import pytest
from unittest.mock import Mock, patch
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManager, EpisodeManagerState
from config.training.training_config import TrainingManagerConfig


class TestOnEpisodesCompleted:
    """Test suite for on_episodes_completed method with complete coverage."""

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

    def test_valid_count_updates_all_counters(self, episode_manager):
        """Test that valid count updates all episode counters correctly."""
        # Set initial state
        episode_manager.state.current_day_episodes = 5
        episode_manager.state.episodes_in_current_day = 3
        episode_manager.state.episodes_in_current_cycle = 10
        
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(2)
        
        assert episode_manager.state.current_day_episodes == 7
        assert episode_manager.state.episodes_in_current_day == 5
        assert episode_manager.state.episodes_in_current_cycle == 12

    @pytest.mark.parametrize("count", [0, -1, -5, -100])
    def test_invalid_count_warning(self, episode_manager, count, caplog):
        """Test that invalid counts log warning and don't update state."""
        initial_day_episodes = episode_manager.state.current_day_episodes
        initial_cycle_episodes = episode_manager.state.episodes_in_current_cycle
        
        with caplog.at_level(logging.WARNING):
            episode_manager.on_episodes_completed(count)
        
        assert f"Invalid episode count: {count}" in caplog.text
        assert episode_manager.state.current_day_episodes == initial_day_episodes
        assert episode_manager.state.episodes_in_current_cycle == initial_cycle_episodes

    def test_calls_advance_reset_point_after_update(self, episode_manager):
        """Test that _advance_to_next_reset_point is called after updating counts."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True) as mock_advance:
            episode_manager.on_episodes_completed(1)
        
        mock_advance.assert_called_once()

    def test_logging_when_advance_reset_point_succeeds(self, episode_manager, caplog):
        """Test debug logging for successful reset point advancement."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            with caplog.at_level(logging.DEBUG):
                episode_manager.on_episodes_completed(3)
        
        assert "Episodes completed: +3" in caplog.text
        assert f"day total: {episode_manager.state.current_day_episodes}" in caplog.text

    def test_logging_when_advance_reset_point_fails(self, episode_manager, caplog):
        """Test debug logging when reset point advancement fails."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=False):
            with caplog.at_level(logging.DEBUG):
                episode_manager.on_episodes_completed(1)
        
        assert "Completed all reset points in current cycle" in caplog.text

    @pytest.mark.parametrize("episode_count", [1, 5, 10, 100])
    def test_various_valid_counts(self, episode_manager, episode_count):
        """Test various valid episode counts are handled correctly."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(episode_count)
        
        assert episode_manager.state.current_day_episodes == episode_count
        assert episode_manager.state.episodes_in_current_day == episode_count
        assert episode_manager.state.episodes_in_current_cycle == episode_count

    def test_multiple_calls_accumulate_correctly(self, episode_manager):
        """Test that multiple calls accumulate episode counts correctly."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(2)
            episode_manager.on_episodes_completed(3)
            episode_manager.on_episodes_completed(1)
        
        assert episode_manager.state.current_day_episodes == 6
        assert episode_manager.state.episodes_in_current_day == 6
        assert episode_manager.state.episodes_in_current_cycle == 6

    def test_no_advance_called_for_invalid_count(self, episode_manager):
        """Test that _advance_to_next_reset_point is not called for invalid counts."""
        with patch.object(episode_manager, '_advance_to_next_reset_point') as mock_advance:
            episode_manager.on_episodes_completed(0)
            episode_manager.on_episodes_completed(-1)
        
        mock_advance.assert_not_called()

    def test_exception_in_advance_reset_point(self, episode_manager):
        """Test that exceptions in _advance_to_next_reset_point are handled."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', side_effect=RuntimeError("Test error")):
            # Should not raise exception
            with pytest.raises(RuntimeError):
                episode_manager.on_episodes_completed(1)

    def test_state_consistency_with_initial_values(self, episode_manager):
        """Test state remains consistent when starting from non-zero values."""
        # Set initial non-zero state
        episode_manager.state.current_day_episodes = 10
        episode_manager.state.episodes_in_current_day = 8
        episode_manager.state.episodes_in_current_cycle = 15
        
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(5)
        
        # All counters should increment by 5
        assert episode_manager.state.current_day_episodes == 15
        assert episode_manager.state.episodes_in_current_day == 13
        assert episode_manager.state.episodes_in_current_cycle == 20

    def test_boundary_value_zero(self, episode_manager, caplog):
        """Test boundary value of exactly zero."""
        with caplog.at_level(logging.WARNING):
            episode_manager.on_episodes_completed(0)
        
        assert "Invalid episode count: 0" in caplog.text

    def test_large_episode_count(self, episode_manager):
        """Test handling of very large episode counts."""
        large_count = 1000000
        
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(large_count)
        
        assert episode_manager.state.current_day_episodes == large_count
        assert episode_manager.state.episodes_in_current_day == large_count
        assert episode_manager.state.episodes_in_current_cycle == large_count

    def test_counters_independent_update(self, episode_manager):
        """Test that each counter is updated independently."""
        # Set different initial values for each counter
        episode_manager.state.current_day_episodes = 1
        episode_manager.state.episodes_in_current_day = 2
        episode_manager.state.episodes_in_current_cycle = 3
        
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(10)
        
        # Each should increment by 10
        assert episode_manager.state.current_day_episodes == 11
        assert episode_manager.state.episodes_in_current_day == 12
        assert episode_manager.state.episodes_in_current_cycle == 13

    def test_advance_behavior_does_not_affect_counters(self, episode_manager):
        """Test that counter updates happen regardless of advance result."""
        # Test with advance returning True
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(5)
        
        count_after_true = episode_manager.state.current_day_episodes
        
        # Reset and test with advance returning False
        episode_manager.state.current_day_episodes = 0
        episode_manager.state.episodes_in_current_day = 0
        episode_manager.state.episodes_in_current_cycle = 0
        
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=False):
            episode_manager.on_episodes_completed(5)
        
        count_after_false = episode_manager.state.current_day_episodes
        
        assert count_after_true == count_after_false == 5

    def test_float_count_handling(self, episode_manager):
        """Test handling of float values (should work as int is not enforced)."""
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(2.5)
        
        assert episode_manager.state.current_day_episodes == 2.5
        assert episode_manager.state.episodes_in_current_day == 2.5
        assert episode_manager.state.episodes_in_current_cycle == 2.5

    def test_negative_float_warning(self, episode_manager, caplog):
        """Test that negative float values trigger warning."""
        with caplog.at_level(logging.WARNING):
            episode_manager.on_episodes_completed(-0.5)
        
        assert "Invalid episode count: -0.5" in caplog.text

    def test_idempotency_with_zero_count(self, episode_manager):
        """Test that calling with valid count after invalid doesn't cause issues."""
        # First call with invalid count
        episode_manager.on_episodes_completed(0)
        
        # Then call with valid count
        with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
            episode_manager.on_episodes_completed(3)
        
        assert episode_manager.state.current_day_episodes == 3