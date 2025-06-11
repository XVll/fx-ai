"""
Comprehensive tests for EpisodeManager.get_next_episode method with 100% coverage.
Tests episode context creation, day/reset point transitions, and exception conditions.
"""

import pytest
from unittest.mock import Mock, patch
import pendulum
from pendulum import Date

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import (
    EpisodeManager, EpisodeManagerException, EpisodeTerminationReason,
    DayInfo, ResetPointInfo, EpisodeContext, EpisodeManagerState
)
from config.training.training_config import TrainingManagerConfig


class TestGetNextEpisode:
    """Test suite for get_next_episode method with complete coverage."""

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
    def mock_reset_point(self):
        """Create a mock ResetPointInfo object."""
        return ResetPointInfo(
            timestamp="2024-01-15 09:30:00",
            quality_score=0.8,
            roc_score=0.6,
            activity_score=0.9,
            price=150.0,
            index=0
        )

    @pytest.fixture
    def mock_day_info(self, mock_reset_point):
        """Create a mock DayInfo object with reset points."""
        return DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=[mock_reset_point]
        )

    def test_successful_episode_creation(self, episode_manager, mock_day_info, mock_reset_point):
        """Test successful creation of episode context."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = mock_reset_point
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            result = episode_manager.get_next_episode()
        
        assert isinstance(result, EpisodeContext)
        assert result.symbol == "AAPL"
        assert result.date == pendulum.parse("2024-01-15").date()
        assert result.reset_point == mock_reset_point
        assert result.day_info == mock_day_info

    def test_shutdown_requested_raises_exception(self, episode_manager):
        """Test that shutdown request raises EpisodeManagerException."""
        episode_manager._shutdown_requested = True
        
        with pytest.raises(EpisodeManagerException) as exc_info:
            episode_manager.get_next_episode()
        
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_DAYS
        assert "Shutdown requested" in str(exc_info.value)

    def test_switch_day_when_should_switch_returns_true(self, episode_manager, mock_day_info, mock_reset_point):
        """Test that day switch is attempted when _should_switch_day returns True."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = mock_reset_point
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=True) as mock_advance:
                result = episode_manager.get_next_episode()
        
        mock_advance.assert_called_once()
        assert isinstance(result, EpisodeContext)

    def test_raises_exception_when_advance_day_fails(self, episode_manager):
        """Test raises EpisodeManagerException when _advance_to_next_day fails."""
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_DAYS

    def test_advances_reset_point_when_current_is_none(self, episode_manager, mock_day_info):
        """Test advances to next reset point when current_reset_point is None."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = None
        episode_manager._shutdown_requested = False
        
        mock_new_reset_point = ResetPointInfo(
            timestamp="2024-01-15 10:00:00",
            quality_score=0.7,
            roc_score=0.5,
            activity_score=0.8,
            price=152.0,
            index=1
        )
        
        def side_effect():
            # Simulate the effect of advancing to next reset point
            episode_manager.state.current_reset_point = mock_new_reset_point
            return True
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            with patch.object(episode_manager, '_advance_to_next_reset_point', side_effect=side_effect) as mock_advance:
                result = episode_manager.get_next_episode()
        
        mock_advance.assert_called_once()
        assert isinstance(result, EpisodeContext)

    def test_raises_exception_when_advance_reset_point_fails(self, episode_manager, mock_day_info):
        """Test raises EpisodeManagerException when _advance_to_next_reset_point fails."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = None
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=False):
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_RESET_POINTS

    def test_raises_exception_when_current_day_is_none(self, episode_manager, mock_reset_point):
        """Test raises EpisodeManagerException when current_day is None after checks."""
        episode_manager.state.current_day = None
        episode_manager.state.current_reset_point = mock_reset_point
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            with pytest.raises(EpisodeManagerException) as exc_info:
                episode_manager.get_next_episode()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED
        assert "Missing current day or reset point" in str(exc_info.value)

    def test_raises_exception_when_current_reset_point_is_none_after_checks(self, episode_manager, mock_day_info):
        """Test raises EpisodeManagerException when current_reset_point is None after advance attempt."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = None
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
                # Simulate advance succeeded but reset point still None
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED

    def test_both_day_and_reset_point_none(self, episode_manager):
        """Test raises EpisodeManagerException when both current_day and current_reset_point are None."""
        episode_manager.state.current_day = None
        episode_manager.state.current_reset_point = None
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED

    def test_multiple_calls_without_state_change(self, episode_manager, mock_day_info, mock_reset_point):
        """Test multiple calls return same episode context when state doesn't change."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = mock_reset_point
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            results = []
            for _ in range(3):
                result = episode_manager.get_next_episode()
                results.append(result)
        
        # All results should be valid and have same values
        assert all(isinstance(r, EpisodeContext) for r in results)
        assert all(r.symbol == "AAPL" for r in results)
        assert all(r.reset_point == mock_reset_point for r in results)

    def test_switch_day_then_advance_reset_point(self, episode_manager):
        """Test scenario where both day switch and reset point advance happen."""
        episode_manager._shutdown_requested = False
        
        # Initial state
        old_day = DayInfo(
            date=pendulum.parse("2024-01-14").date(),
            symbol="AAPL",
            day_score=0.7,
            reset_points=[Mock()]
        )
        episode_manager.state.current_day = old_day
        episode_manager.state.current_reset_point = None
        
        # New day after switch
        new_day = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="TSLA",
            day_score=0.8,
            reset_points=[Mock()]
        )
        
        new_reset_point = ResetPointInfo(
            timestamp="2024-01-15 09:30:00",
            quality_score=0.8,
            roc_score=0.6,
            activity_score=0.9,
            price=200.0,
            index=0
        )
        
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=True):
                with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
                    # Simulate state changes from the advances
                    episode_manager.state.current_day = new_day
                    episode_manager.state.current_reset_point = new_reset_point
                    
                    result = episode_manager.get_next_episode()
        
        assert isinstance(result, EpisodeContext)
        assert result.symbol == "TSLA"
        assert result.date == pendulum.parse("2024-01-15").date()

    def test_exception_propagation_from_internal_methods(self, episode_manager):
        """Test that exceptions from internal methods are propagated."""
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError):
                episode_manager.get_next_episode()

    def test_episode_context_attributes(self, episode_manager, mock_day_info, mock_reset_point):
        """Test that EpisodeContext has all expected attributes correctly set."""
        episode_manager.state.current_day = mock_day_info
        episode_manager.state.current_reset_point = mock_reset_point
        episode_manager._shutdown_requested = False
        
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            result = episode_manager.get_next_episode()
        
        # Verify all attributes
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'date')
        assert hasattr(result, 'reset_point')
        assert hasattr(result, 'day_info')
        
        # Verify types
        assert isinstance(result.symbol, str)
        assert isinstance(result.date, Date)
        assert isinstance(result.reset_point, ResetPointInfo)
        assert isinstance(result.day_info, DayInfo)
        
        # Verify values match state
        assert result.symbol == episode_manager.state.current_day.symbol
        assert result.date == episode_manager.state.current_day.date
        assert result.reset_point == episode_manager.state.current_reset_point
        assert result.day_info == episode_manager.state.current_day

    def test_all_termination_reasons_covered(self, episode_manager):
        """Test that all relevant termination reasons can be raised."""
        # Test NO_MORE_DAYS (shutdown)
        episode_manager._shutdown_requested = True
        with pytest.raises(EpisodeManagerException) as exc_info:
            episode_manager.get_next_episode()
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_DAYS
        
        # Reset for next test
        episode_manager._shutdown_requested = False
        
        # Test NO_MORE_DAYS (advance day fails)
        with patch.object(episode_manager, '_should_switch_day', return_value=True):
            with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_DAYS
        
        # Test NO_MORE_RESET_POINTS
        episode_manager.state.current_reset_point = None
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=False):
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_RESET_POINTS
        
        # Test PRELOAD_FAILED - when both day and reset point are None
        episode_manager.state.current_day = None
        episode_manager.state.current_reset_point = None
        with patch.object(episode_manager, '_should_switch_day', return_value=False):
            # Since current_reset_point is None, it will try to advance and then hit the final check
            with patch.object(episode_manager, '_advance_to_next_reset_point', return_value=True):
                with pytest.raises(EpisodeManagerException) as exc_info:
                    episode_manager.get_next_episode()
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED