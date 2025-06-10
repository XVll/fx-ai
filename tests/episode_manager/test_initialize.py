"""
Comprehensive tests for EpisodeManager.initialize method with 100% coverage.
Tests initialization logic, error handling, and state setup.
"""

import pytest
import logging
from unittest.mock import Mock, patch
import pendulum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManager, DayInfo, ResetPointInfo
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
        """Create EpisodeManager with pre-loaded days."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        # Inject available days directly
        episode_manager.available_days = [mock_day_info]
        return episode_manager

    def test_successful_initialization(self, episode_manager_with_days, mock_day_info):
        """Test successful initialization with valid day selection."""
        result = episode_manager_with_days.initialize()
        
        assert result is True
        assert episode_manager_with_days.state.current_day == mock_day_info
        assert len(episode_manager_with_days.state.ordered_reset_points) == 2
        assert episode_manager_with_days.state.current_reset_point_index == 1  # After initialization, it advances to first reset point

    def test_initialization_no_available_days(self, mock_config, mock_data_manager):
        """Test initialization fails when no days are available."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        episode_manager.available_days = []  # No days available
        
        result = episode_manager.initialize()
        
        assert result is False
        assert episode_manager.state.current_day is None

    def test_initialization_advance_day_fails(self, mock_config, mock_data_manager):
        """Test initialization fails when _advance_to_next_day fails."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Mock _advance_to_next_day to return False
        with patch.object(episode_manager, '_advance_to_next_day', return_value=False):
            result = episode_manager.initialize()
        
        assert result is False

    def test_initialization_with_exception_in_advance_day(self, episode_manager_with_days):
        """Test initialization handles exceptions in _advance_to_next_day."""
        with patch.object(episode_manager_with_days, '_advance_to_next_day', side_effect=Exception("Test error")):
            result = episode_manager_with_days.initialize()
        
        assert result is False

    def test_initialization_logging_with_valid_day(self, episode_manager_with_days, caplog):
        """Test that appropriate log messages are generated on successful initialization."""
        with caplog.at_level(logging.INFO):
            result = episode_manager_with_days.initialize()
        
        assert result is True
        assert "📅 Selected: AAPL 2024-01-15" in caplog.text
        assert "(quality: 0.750)" in caplog.text
        assert "🔄 Reset points: 2 available" in caplog.text

    def test_initialization_logging_with_no_days(self, mock_config, mock_data_manager, caplog):
        """Test error logging when no days are available."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        episode_manager.available_days = []
        
        with caplog.at_level(logging.ERROR):
            result = episode_manager.initialize()
        
        assert result is False
        assert "Failed to select initial day" in caplog.text

    def test_initialization_logging_with_exception(self, episode_manager_with_days, caplog):
        """Test error logging when exception occurs during initialization."""
        with patch.object(episode_manager_with_days, '_advance_to_next_day', side_effect=RuntimeError("Test runtime error")):
            with caplog.at_level(logging.ERROR):
                result = episode_manager_with_days.initialize()
        
        assert result is False
        assert "Failed to initialize episode manager: Test runtime error" in caplog.text

    def test_initialization_state_after_success(self, episode_manager_with_days):
        """Test that state is properly configured after successful initialization."""
        result = episode_manager_with_days.initialize()
        
        assert result is True
        state = episode_manager_with_days.state
        
        # Check state is properly initialized
        assert state.current_day is not None
        assert state.current_day.symbol == "AAPL"
        assert state.current_day.date == pendulum.parse("2024-01-15").date()
        assert len(state.ordered_reset_points) == 2
        assert state.current_reset_point_index == 1  # After initialization, it advances to first reset point
        assert len(state.used_days) == 1  # Current day should be marked as used

    def test_initialization_with_empty_reset_points(self, mock_config, mock_data_manager):
        """Test initialization when day has no reset points."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Create day with empty reset points
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=[]  # Empty reset points
        )
        episode_manager.available_days = [day_info]
        
        result = episode_manager.initialize()
        
        # Should fail as day has no reset points
        assert result is False

    def test_initialization_multiple_days_sequential_selection(self, mock_config, mock_data_manager):
        """Test initialization with multiple days in sequential mode."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Create multiple days
        days = []
        for i in range(3):
            reset_points = [
                ResetPointInfo(
                    timestamp=f"2024-01-{15+i} 09:30:00",
                    quality_score=0.8,
                    roc_score=0.6,
                    activity_score=0.9,
                    price=150.0 + i,
                    index=0
                )
            ]
            day = DayInfo(
                date=pendulum.parse(f"2024-01-{15+i}").date(),
                symbol="AAPL",
                day_score=0.75 + i * 0.05,
                reset_points=reset_points
            )
            days.append(day)
        
        episode_manager.available_days = days
        result = episode_manager.initialize()
        
        assert result is True
        # In sequential mode, should select first day
        assert episode_manager.state.current_day == days[0]

    def test_initialization_random_selection_mode(self, mock_config, mock_data_manager):
        """Test initialization with random day selection mode."""
        mock_config.day_selection_mode = "random"
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Create multiple days
        days = []
        for i in range(5):
            reset_points = [
                ResetPointInfo(
                    timestamp=f"2024-01-{10+i} 09:30:00",
                    quality_score=0.8,
                    roc_score=0.6,
                    activity_score=0.9,
                    price=150.0,
                    index=0
                )
            ]
            day = DayInfo(
                date=pendulum.parse(f"2024-01-{10+i}").date(),
                symbol="AAPL",
                day_score=0.75,
                reset_points=reset_points
            )
            days.append(day)
        
        episode_manager.available_days = days
        
        # Test multiple times to ensure randomness
        selected_days = set()
        for _ in range(10):
            # Reset state for each test
            episode_manager.state.current_day = None
            episode_manager.state.used_days.clear()
            
            result = episode_manager.initialize()
            assert result is True
            if episode_manager.state.current_day:
                selected_days.add(episode_manager.state.current_day.date.format('YYYY-MM-DD'))
        
        # Should have selected different days (probabilistically)
        # With 5 days and 10 tries, very likely to see more than 1 day
        assert len(selected_days) > 1

    def test_initialization_quality_selection_mode(self, mock_config, mock_data_manager):
        """Test initialization with quality-based day selection mode."""
        mock_config.day_selection_mode = "quality"
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Create days with different quality scores
        days = []
        quality_scores = [0.5, 0.9, 0.3, 0.7, 0.6]  # 0.9 is highest
        for i, score in enumerate(quality_scores):
            reset_points = [
                ResetPointInfo(
                    timestamp=f"2024-01-{10+i} 09:30:00",
                    quality_score=0.8,
                    roc_score=0.6,
                    activity_score=0.9,
                    price=150.0,
                    index=0
                )
            ]
            day = DayInfo(
                date=pendulum.parse(f"2024-01-{10+i}").date(),
                symbol="AAPL",
                day_score=score,
                reset_points=reset_points
            )
            days.append(day)
        
        episode_manager.available_days = days
        result = episode_manager.initialize()
        
        assert result is True
        # In quality mode, should select day with highest score (0.9)
        assert episode_manager.state.current_day.day_score == 0.9

    def test_initialization_idempotency(self, episode_manager_with_days):
        """Test that calling initialize multiple times is safe."""
        # First initialization
        result1 = episode_manager_with_days.initialize()
        assert result1 is True
        
        initial_day = episode_manager_with_days.state.current_day
        initial_reset_points = len(episode_manager_with_days.state.ordered_reset_points)
        
        # Second initialization
        result2 = episode_manager_with_days.initialize()
        assert result2 is True
        
        # Should have same or different day (depending on selection logic)
        # But should not corrupt state
        assert episode_manager_with_days.state.current_day is not None
        assert len(episode_manager_with_days.state.ordered_reset_points) > 0

    @pytest.mark.parametrize("exception_type,exception_message", [
        (ValueError, "Invalid value"),
        (RuntimeError, "Runtime error"),
        (AttributeError, "Attribute missing"),
        (KeyError, "Key not found"),
    ])
    def test_initialization_various_exceptions(self, episode_manager_with_days, exception_type, exception_message):
        """Test initialization handles various exception types gracefully."""
        with patch.object(episode_manager_with_days, '_advance_to_next_day', side_effect=exception_type(exception_message)):
            result = episode_manager_with_days.initialize()
        
        assert result is False

    def test_initialization_with_none_current_day(self, episode_manager_with_days):
        """Test initialization when _advance_to_next_day succeeds but current_day is None."""
        with patch.object(episode_manager_with_days, '_advance_to_next_day', return_value=True):
            # Force current_day to be None
            episode_manager_with_days.state.current_day = None
            
            result = episode_manager_with_days.initialize()
        
        # Should still succeed, just won't log the summary
        assert result is True

    def test_initialization_preserves_existing_state(self, episode_manager_with_days):
        """Test that initialization preserves certain state properties."""
        # Set some existing state
        episode_manager_with_days.state.cycle_count = 5
        episode_manager_with_days.state.total_cycles_completed = 10
        
        result = episode_manager_with_days.initialize()
        
        assert result is True
        # These values should be preserved (not reset)
        assert episode_manager_with_days.state.cycle_count == 5
        assert episode_manager_with_days.state.total_cycles_completed == 10