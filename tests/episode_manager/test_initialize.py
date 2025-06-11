"""
Comprehensive tests for EpisodeManager.initialize method with 100% coverage.
Tests initialization logic, error handling, and exception conditions.
"""

import pytest
import logging
from unittest.mock import Mock, patch
import pendulum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import (
    EpisodeManager, EpisodeManagerException, EpisodeTerminationReason,
    DayInfo, ResetPointInfo
)
from config.training.training_config import TrainingManagerConfig


class TestInitialize:
    """Test suite for initialize method with complete coverage."""

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
        config.daily_max_episodes = 100
        config.daily_max_updates = 50
        config.daily_max_cycles = 10
        return config

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock data manager with test data."""
        data_manager = Mock()
        return data_manager

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
            ),
            ResetPointInfo(
                timestamp="2024-01-15 10:00:00",
                quality_score=0.7,
                roc_score=0.5,
                activity_score=0.8,
                price=152.0,
                index=1
            )
        ]
        
        return DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=reset_points
        )

    @pytest.fixture
    def episode_manager_with_days(self, mock_config, mock_data_manager, mock_day_info):
        """Create episode manager with available days."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        episode_manager.available_days = [mock_day_info]
        return episode_manager

    def test_successful_initialization(self, episode_manager_with_days, mock_day_info):
        """Test successful initialization with valid day selection."""
        episode_manager_with_days.initialize()
        
        assert episode_manager_with_days.state.current_day == mock_day_info
        assert len(episode_manager_with_days.state.ordered_reset_points) == 2
        assert episode_manager_with_days.state.current_reset_point_index == 1  # After initialization, it advances to first reset point

    def test_initialization_no_available_days_raises_exception(self, mock_config, mock_data_manager):
        """Test initialization raises exception when no days are available."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        episode_manager.available_days = []  # No days available
        
        with pytest.raises(EpisodeManagerException) as exc_info:
            episode_manager.initialize()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED
        assert "Failed to select initial day" in str(exc_info.value)

    def test_initialization_advance_day_fails_raises_exception(self, mock_config, mock_data_manager):
        """Test initialization raises exception when _advance_to_next_day fails."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Mock _advance_to_next_day to return False
        with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
            with pytest.raises(EpisodeManagerException) as exc_info:
                episode_manager.initialize()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED

    def test_initialization_with_exception_in_advance_day(self, episode_manager_with_days):
        """Test initialization handles exceptions in _advance_to_next_day."""
        test_error = RuntimeError("Test error")
        with patch.object(episode_manager_with_days, '_advance_to_next_day', side_effect=test_error):
            with pytest.raises(EpisodeManagerException) as exc_info:
                episode_manager_with_days.initialize()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED
        assert "Initialization failed: Test error" in str(exc_info.value)

    def test_initialization_logging_with_valid_day(self, episode_manager_with_days, caplog):
        """Test that appropriate log messages are generated on successful initialization."""
        with caplog.at_level(logging.INFO):
            episode_manager_with_days.initialize()
        
        assert "📅 Selected: AAPL 2024-01-15" in caplog.text
        assert "(quality: 0.750)" in caplog.text
        assert "🔄 Reset points: 2 available" in caplog.text

    def test_initialization_logging_with_no_days(self, mock_config, mock_data_manager, caplog):
        """Test error logging when no days are available."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        episode_manager.available_days = []
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(EpisodeManagerException):
                episode_manager.initialize()
        
        assert "Failed to select initial day" in caplog.text

    def test_initialization_logging_with_exception(self, episode_manager_with_days, caplog):
        """Test error logging when exception occurs during initialization."""
        with patch.object(episode_manager_with_days, '_advance_to_next_day', side_effect=RuntimeError("Test runtime error")):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(EpisodeManagerException):
                    episode_manager_with_days.initialize()
        
        assert "Failed to initialize episode manager: Test runtime error" in caplog.text

    def test_initialization_state_after_success(self, episode_manager_with_days):
        """Test that state is properly configured after successful initialization."""
        episode_manager_with_days.initialize()
        
        state = episode_manager_with_days.state
        
        # Check state is properly initialized
        assert state.current_day is not None
        assert state.current_day.symbol == "AAPL"
        assert state.current_day.date == pendulum.parse("2024-01-15").date()
        assert len(state.ordered_reset_points) == 2
        assert state.current_reset_point_index == 1  # After initialization, it advances to first reset point
        assert len(state.used_days) == 1  # Current day should be marked as used

    def test_initialization_with_empty_reset_points_raises_exception(self, mock_config, mock_data_manager):
        """Test initialization raises exception when day has no reset points."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Create day with empty reset points
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=[]  # Empty reset points
        )
        episode_manager.available_days = [day_info]
        
        with pytest.raises(EpisodeManagerException) as exc_info:
            episode_manager.initialize()
        
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED

    def test_initialization_preserves_existing_state(self, episode_manager_with_days):
        """Test that initialization preserves certain state values."""
        # Set some existing state
        episode_manager_with_days.state.cycle_count = 5
        episode_manager_with_days.state.total_cycles_completed = 10
        
        episode_manager_with_days.initialize()
        
        # These values should be preserved (not reset)
        assert episode_manager_with_days.state.cycle_count == 5
        assert episode_manager_with_days.state.total_cycles_completed == 10

    def test_initialization_idempotency(self, episode_manager_with_days):
        """Test that multiple initializations don't break state."""
        # First initialization
        episode_manager_with_days.initialize()
        first_day = episode_manager_with_days.state.current_day
        
        # Second initialization
        episode_manager_with_days.initialize()
        second_day = episode_manager_with_days.state.current_day
        
        # Should successfully initialize both times
        assert first_day == second_day
        assert episode_manager_with_days.state.current_day is not None

    def test_initialization_exception_types(self, mock_config, mock_data_manager):
        """Test that different failure scenarios raise appropriate exception reasons."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Test PRELOAD_FAILED for no days
        episode_manager.available_days = []
        with pytest.raises(EpisodeManagerException) as exc_info:
            episode_manager.initialize()
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED
        
        # Test PRELOAD_FAILED for advance failure
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL", 
            day_score=0.75,
            reset_points=[ResetPointInfo(
                timestamp="2024-01-15 09:30:00",
                quality_score=0.8,
                roc_score=0.6,
                activity_score=0.9,
                price=150.0,
                index=0
            )]
        )
        episode_manager.available_days = [day_info]
        
        with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
            with pytest.raises(EpisodeManagerException) as exc_info:
                episode_manager.initialize()
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED
        
        # Test PRELOAD_FAILED for exception during advance
        with patch.object(episode_manager, '_advance_to_next_day', side_effect=ValueError("Test value error")):
            with pytest.raises(EpisodeManagerException) as exc_info:
                episode_manager.initialize()
        assert exc_info.value.reason == EpisodeTerminationReason.PRELOAD_FAILED
        assert "Initialization failed: Test value error" in str(exc_info.value)

    def test_initialization_cleanup_on_failure(self, mock_config, mock_data_manager):
        """Test that state is not partially modified on initialization failure."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Ensure clean initial state
        assert episode_manager.state.current_day is None
        assert len(episode_manager.state.ordered_reset_points) == 0
        
        # Try to initialize with no days (should fail)
        episode_manager.available_days = []
        with pytest.raises(EpisodeManagerException):
            episode_manager.initialize()
        
        # State should remain clean after failure
        assert episode_manager.state.current_day is None
        assert len(episode_manager.state.ordered_reset_points) == 0