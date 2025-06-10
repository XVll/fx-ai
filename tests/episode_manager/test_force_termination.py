"""
Comprehensive tests for EpisodeManager.force_termination method with 100% coverage.
Tests termination state setting and all termination reasons.
"""

import pytest
from unittest.mock import Mock
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManager, EpisodeTerminationReason
from config.training.training_config import TrainingManagerConfig


class TestForceTermination:
    """Test suite for force_termination method with complete coverage."""

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

    def test_sets_termination_state(self, episode_manager):
        """Test that force_termination sets should_terminate to True."""
        assert episode_manager.state.should_terminate is False
        
        episode_manager.force_termination(EpisodeTerminationReason.CYCLE_LIMIT_REACHED)
        
        assert episode_manager.state.should_terminate is True

    def test_sets_termination_reason(self, episode_manager):
        """Test that force_termination sets the termination reason correctly."""
        assert episode_manager.state.termination_reason is None
        
        episode_manager.force_termination(EpisodeTerminationReason.UPDATE_LIMIT_REACHED)
        
        assert episode_manager.state.termination_reason == EpisodeTerminationReason.UPDATE_LIMIT_REACHED

    @pytest.mark.parametrize("reason", [
        EpisodeTerminationReason.CYCLE_LIMIT_REACHED,
        EpisodeTerminationReason.EPISODE_LIMIT_REACHED,
        EpisodeTerminationReason.UPDATE_LIMIT_REACHED,
        EpisodeTerminationReason.NO_MORE_RESET_POINTS,
        EpisodeTerminationReason.NO_MORE_DAYS,
        EpisodeTerminationReason.DATE_RANGE_EXHAUSTED,
        EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET,
        EpisodeTerminationReason.PRELOAD_FAILED,
    ])
    def test_all_termination_reasons(self, episode_manager, reason):
        """Test that all termination reasons can be set."""
        episode_manager.force_termination(reason)
        
        assert episode_manager.state.should_terminate is True
        assert episode_manager.state.termination_reason == reason

    def test_overwrites_existing_termination(self, episode_manager):
        """Test that force_termination overwrites existing termination state."""
        # Set initial termination
        episode_manager.force_termination(EpisodeTerminationReason.CYCLE_LIMIT_REACHED)
        
        # Force different termination
        episode_manager.force_termination(EpisodeTerminationReason.NO_MORE_DAYS)
        
        assert episode_manager.state.should_terminate is True
        assert episode_manager.state.termination_reason == EpisodeTerminationReason.NO_MORE_DAYS

    def test_logging_message(self, episode_manager, caplog):
        """Test that appropriate log message is generated."""
        with caplog.at_level(logging.INFO):
            episode_manager.force_termination(EpisodeTerminationReason.PRELOAD_FAILED)
        
        assert "🛑 Force terminating episode manager: preload_failed" in caplog.text

    def test_multiple_calls_maintain_state(self, episode_manager):
        """Test that multiple calls maintain terminated state."""
        episode_manager.force_termination(EpisodeTerminationReason.UPDATE_LIMIT_REACHED)
        
        # Call again with different reason
        episode_manager.force_termination(EpisodeTerminationReason.EPISODE_LIMIT_REACHED)
        
        # Should still be terminated with latest reason
        assert episode_manager.state.should_terminate is True
        assert episode_manager.state.termination_reason == EpisodeTerminationReason.EPISODE_LIMIT_REACHED

    def test_no_side_effects_on_other_state(self, episode_manager):
        """Test that only termination-related state is modified."""
        # Set some state
        episode_manager.state.current_day_updates = 5
        episode_manager.state.current_day_episodes = 10
        episode_manager.state.cycle_count = 3
        
        episode_manager.force_termination(EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET)
        
        # Other state should remain unchanged
        assert episode_manager.state.current_day_updates == 5
        assert episode_manager.state.current_day_episodes == 10
        assert episode_manager.state.cycle_count == 3

    def test_termination_affects_should_terminate_method(self, episode_manager):
        """Test that force_termination affects should_terminate() method output."""
        # Initially should return None
        assert episode_manager.should_terminate() is None
        
        # Force termination
        episode_manager.force_termination(EpisodeTerminationReason.DATE_RANGE_EXHAUSTED)
        
        # Now should return the termination reason
        assert episode_manager.should_terminate() == EpisodeTerminationReason.DATE_RANGE_EXHAUSTED

    def test_termination_reason_enum_value(self, episode_manager):
        """Test that termination reason value property is accessible."""
        episode_manager.force_termination(EpisodeTerminationReason.NO_MORE_RESET_POINTS)
        
        reason = episode_manager.state.termination_reason
        assert reason.value == "no_more_reset_points"

    def test_force_termination_from_non_terminated_state(self, episode_manager):
        """Test force termination when starting from non-terminated state."""
        # Ensure clean state
        assert episode_manager.state.should_terminate is False
        assert episode_manager.state.termination_reason is None
        
        episode_manager.force_termination(EpisodeTerminationReason.CYCLE_LIMIT_REACHED)
        
        assert episode_manager.state.should_terminate is True
        assert episode_manager.state.termination_reason == EpisodeTerminationReason.CYCLE_LIMIT_REACHED

    def test_idempotency_same_reason(self, episode_manager):
        """Test that calling with same reason multiple times is idempotent."""
        reason = EpisodeTerminationReason.EPISODE_LIMIT_REACHED
        
        # Call multiple times with same reason
        for _ in range(3):
            episode_manager.force_termination(reason)
        
        assert episode_manager.state.should_terminate is True
        assert episode_manager.state.termination_reason == reason

    def test_termination_persists_across_other_operations(self, episode_manager):
        """Test that termination state persists when other methods are called."""
        episode_manager.force_termination(EpisodeTerminationReason.UPDATE_LIMIT_REACHED)
        
        # Call other methods that might affect state
        episode_manager.on_update_completed(None)
        episode_manager.on_episodes_completed(1)
        
        # Termination should persist
        assert episode_manager.state.should_terminate is True
        assert episode_manager.state.termination_reason == EpisodeTerminationReason.UPDATE_LIMIT_REACHED

    def test_each_termination_reason_has_unique_value(self):
        """Test that all termination reasons have unique string values."""
        values = [reason.value for reason in EpisodeTerminationReason]
        assert len(values) == len(set(values))  # All values should be unique

    def test_termination_reason_string_representation(self, episode_manager):
        """Test string representation of termination reasons."""
        episode_manager.force_termination(EpisodeTerminationReason.PRELOAD_FAILED)
        
        reason_str = str(episode_manager.state.termination_reason)
        assert "PRELOAD_FAILED" in reason_str

    def test_force_termination_with_logging_levels(self, episode_manager, caplog):
        """Test logging at different levels to ensure INFO level is used."""
        # Test that nothing is logged at WARNING level
        with caplog.at_level(logging.WARNING):
            episode_manager.force_termination(EpisodeTerminationReason.NO_MORE_DAYS)
            warning_logs = caplog.text
        
        # Test that message is logged at INFO level
        caplog.clear()
        with caplog.at_level(logging.INFO):
            episode_manager.force_termination(EpisodeTerminationReason.NO_MORE_DAYS)
            info_logs = caplog.text
        
        assert len(warning_logs) == 0 or "Force terminating" not in warning_logs
        assert "Force terminating" in info_logs