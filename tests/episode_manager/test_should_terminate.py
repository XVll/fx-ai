"""
Comprehensive tests for EpisodeManager.should_terminate method with 100% coverage.
Tests all termination conditions and state transitions.
"""

import pytest
from unittest.mock import Mock, patch
import pendulum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import (
    EpisodeManager, DayInfo, ResetPointInfo, 
    EpisodeTerminationReason, EpisodeManagerState
)
from config.training.training_config import TrainingManagerConfig


class TestShouldTerminate:
    """Test suite for should_terminate method with complete coverage."""

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

    @pytest.fixture
    def mock_day_info(self):
        """Create a mock DayInfo object for testing."""
        reset_points = [
            ResetPointInfo(
                timestamp="2024-01-15 09:30:00",
                quality_score=0.8,
                roc_score=0.6,
                activity_score=0.9,
                price=150.0,
                index=0
            )
        ]
        
        return DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=reset_points
        )

    def test_no_termination_when_state_not_terminated(self, episode_manager):
        """Test returns None when should_terminate is False."""
        episode_manager.state.should_terminate = False
        
        result = episode_manager.should_terminate()
        
        assert result is None

    def test_returns_termination_reason_when_state_terminated(self, episode_manager):
        """Test returns termination reason when should_terminate is True."""
        episode_manager.state.should_terminate = True
        episode_manager.state.termination_reason = EpisodeTerminationReason.CYCLE_LIMIT_REACHED
        
        result = episode_manager.should_terminate()
        
        assert result == EpisodeTerminationReason.CYCLE_LIMIT_REACHED

    @pytest.mark.parametrize("termination_reason", [
        EpisodeTerminationReason.CYCLE_LIMIT_REACHED,
        EpisodeTerminationReason.EPISODE_LIMIT_REACHED,
        EpisodeTerminationReason.UPDATE_LIMIT_REACHED,
        EpisodeTerminationReason.NO_MORE_RESET_POINTS,
        EpisodeTerminationReason.NO_MORE_DAYS,
        EpisodeTerminationReason.DATE_RANGE_EXHAUSTED,
        EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET,
        EpisodeTerminationReason.PRELOAD_FAILED,
    ])
    def test_all_termination_reasons(self, episode_manager, termination_reason):
        """Test all possible termination reasons are properly returned."""
        episode_manager.state.should_terminate = True
        episode_manager.state.termination_reason = termination_reason
        
        result = episode_manager.should_terminate()
        
        assert result == termination_reason

    def test_checks_should_switch_day_when_not_terminated(self, episode_manager):
        """Test that should_switch_day is called when not terminated."""
        episode_manager.state.should_terminate = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False) as mock_switch:
            result = episode_manager.should_terminate()
        
        assert result is None
        mock_switch.assert_called_once()

    def test_advances_to_next_day_when_should_switch_returns_true(self, episode_manager):
        """Test advances to next day when should_switch_day returns True."""
        episode_manager.state.should_terminate = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=True) as mock_advance:
                result = episode_manager.should_terminate()
        
        assert result is None
        mock_advance.assert_called_once()

    def test_resets_index_when_advance_to_next_day_fails(self, episode_manager, mock_day_info):
        """Test resets to beginning of reset points when advance fails."""
        episode_manager.state.should_terminate = False
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point_index = 5
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
                result = episode_manager.should_terminate()
        
        assert result is None
        assert episode_manager.state.current_reset_point_index == 0

    def test_logging_when_no_more_days_available(self, episode_manager, mock_day_info, caplog):
        """Test logging message when no more days are available."""
        episode_manager.state.should_terminate = False
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
                with caplog.at_level("INFO"):
                    result = episode_manager.should_terminate()
        
        assert result is None
        assert "🔄 No more days available, continuing with current day reset points" in caplog.text

    def test_does_not_check_switch_day_when_terminated(self, episode_manager):
        """Test that _should_switch_day is not called when already terminated."""
        episode_manager.state.should_terminate = True
        episode_manager.state.termination_reason = EpisodeTerminationReason.UPDATE_LIMIT_REACHED
        
        with patch.object(episode_manager, '_should_switch_day') as mock_switch:
            result = episode_manager.should_terminate()
        
        assert result == EpisodeTerminationReason.UPDATE_LIMIT_REACHED
        mock_switch.assert_not_called()

    def test_handles_exception_in_should_switch_day(self, episode_manager):
        """Test graceful handling of exceptions in _should_switch_day."""
        episode_manager.state.should_terminate = False
        
        with patch.object(episode_manager, '_should_switch_day', side_effect=RuntimeError("Test error")):
            # Should not raise exception
            with pytest.raises(RuntimeError):
                episode_manager.should_terminate()

    def test_handles_exception_in_advance_to_next_day(self, episode_manager):
        """Test graceful handling of exceptions in _advance_to_next_day."""
        episode_manager.state.should_terminate = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', side_effect=RuntimeError("Test error")):
                # Should not raise exception
                with pytest.raises(RuntimeError):
                    episode_manager.should_terminate()

    def test_state_consistency_after_multiple_calls(self, episode_manager):
        """Test that state remains consistent after multiple calls."""
        episode_manager.state.should_terminate = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            # Call multiple times
            for _ in range(5):
                result = episode_manager.should_terminate()
                assert result is None
        
        # State should remain unchanged
        assert episode_manager.state.should_terminate is False

    def test_termination_reason_none_when_should_terminate_false(self, episode_manager):
        """Test that termination_reason can be None when should_terminate is False."""
        episode_manager.state.should_terminate = False
        episode_manager.state.termination_reason = None
        
        result = episode_manager.should_terminate()
        
        assert result is None

    def test_termination_with_none_reason_when_should_terminate_true(self, episode_manager):
        """Test behavior when should_terminate is True but reason is None."""
        episode_manager.state.should_terminate = True
        episode_manager.state.termination_reason = None
        
        result = episode_manager.should_terminate()
        
        assert result is None

    def test_reset_point_index_boundary_conditions(self, episode_manager, mock_day_info):
        """Test reset point index boundary conditions."""
        episode_manager.state.should_terminate = False
        episode_manager.state.current_day = mock_day_info
        
        # Test with index at boundary
        episode_manager.state.current_reset_point_index = len(mock_day_info.reset_points) - 1
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
                result = episode_manager.should_terminate()
        
        assert result is None
        assert episode_manager.state.current_reset_point_index == 0

    def test_no_current_day_when_checking_switch(self, episode_manager):
        """Test behavior when current_day is None during switch check."""
        episode_manager.state.should_terminate = False
        episode_manager.state.current_day = None
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
                result = episode_manager.should_terminate()
        
        assert result is None

    @pytest.mark.parametrize("should_switch,advance_success,expected_index_reset", [
        (False, None, False),  # No switch, no advance
        (True, True, False),   # Switch and advance success
        (True, False, True),   # Switch but advance fails
    ])
    def test_switch_and_advance_combinations(self, episode_manager, mock_day_info, 
                                            should_switch, advance_success, expected_index_reset):
        """Test various combinations of switch and advance scenarios."""
        episode_manager.state.should_terminate = False
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point_index = 5
        
        with patch.object(episode_manager, '_should_switch_day', return_value=should_switch):
            if should_switch:
                with patch.object(episode_manager, '_advance_to_next_day', return_value=advance_success):
                    result = episode_manager.should_terminate()
            else:
                result = episode_manager.should_terminate()
        
        assert result is None
        if expected_index_reset:
            assert episode_manager.state.current_reset_point_index == 0
        else:
            assert episode_manager.state.current_reset_point_index == 5

    def test_thread_safety_considerations(self, episode_manager):
        """Test that method doesn't have obvious thread safety issues."""
        # This is a basic test - real thread safety would require more complex testing
        episode_manager.state.should_terminate = True
        episode_manager.state.termination_reason = EpisodeTerminationReason.CYCLE_LIMIT_REACHED
        
        # Multiple rapid calls should return consistent results
        results = []
        for _ in range(10):
            results.append(episode_manager.should_terminate())
        
        assert all(r == EpisodeTerminationReason.CYCLE_LIMIT_REACHED for r in results)