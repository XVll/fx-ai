"""
Comprehensive tests for EpisodeManager._load_available_days method with 100% coverage.
Tests all scenarios including edge cases, error handling, and data validation.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock
import pendulum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManager, DayInfo, ResetPointInfo
from config.training.training_config import TrainingManagerConfig


class TestLoadAvailableDays:
    """Test suite for _load_available_days method with complete coverage."""

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
        """Create mock data manager with realistic data."""
        data_manager = Mock()
        
        # Mock momentum days response
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'quality_score': 0.75
            },
            {
                'symbol': 'TSLA', 
                'date': '2024-01-20',
                'quality_score': 0.65
            }
        ]
        data_manager.get_all_momentum_days.return_value = momentum_days
        
        # Mock reset points response
        reset_points_df = pd.DataFrame({
            'timestamp': ['2024-01-15 09:30:00', '2024-01-15 10:00:00'],
            'combined_score': [0.8, 0.7],
            'roc_score': [0.6, 0.5],
            'activity_score': [0.9, 0.8],
            'price': [150.0, 152.0]
        })
        data_manager.get_reset_points.return_value = reset_points_df
        
        return data_manager

    @pytest.fixture
    def episode_manager(self, mock_config, mock_data_manager):
        """Create EpisodeManager instance for testing."""
        return EpisodeManager(mock_config, mock_data_manager)

    def test_no_data_manager_returns_empty_list(self, mock_config):
        """Test that missing data manager returns empty list."""
        episode_manager = EpisodeManager(mock_config, None)
        result = episode_manager._load_available_days()
        
        assert result == []

    @pytest.mark.parametrize("invalid_date_range", [
        ["invalid-date", "2024-01-31"],
        ["2024-01-01", "invalid-date"], 
        ["", "2024-01-31"],
        ["2024-01-01", ""],
        [None, "2024-01-31"],
        ["2024-01-01", None]
    ])
    def test_invalid_date_range_returns_empty_list(self, mock_config, mock_data_manager, invalid_date_range):
        """Test that invalid date ranges return empty list."""
        mock_config.date_range = invalid_date_range
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert result == []

    def test_data_manager_exception_returns_empty_list(self, mock_config, mock_data_manager):
        """Test that data manager exceptions are handled gracefully."""
        mock_data_manager.get_all_momentum_days.side_effect = Exception("Database connection failed")
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert result == []

    def test_successful_loading_with_valid_data(self, episode_manager, mock_data_manager):
        """Test successful loading of momentum days with valid data."""
        result = episode_manager._load_available_days()
        
        assert len(result) == 2
        
        # Check first day
        day1 = result[0]
        assert isinstance(day1, DayInfo)
        assert day1.symbol == "AAPL"
        assert day1.day_score == 0.75
        assert isinstance(day1.date, pendulum.Date)
        assert day1.date.format('YYYY-MM-DD') == "2024-01-15"
        assert len(day1.reset_points) == 2
        
        # Check reset points
        rp1 = day1.reset_points[0]
        assert isinstance(rp1, ResetPointInfo)
        assert rp1.quality_score == 0.8
        assert rp1.roc_score == 0.6
        assert rp1.activity_score == 0.9
        assert rp1.price == 150.0
        assert rp1.index == 0

    def test_day_without_quality_score_is_skipped(self, mock_config, mock_data_manager):
        """Test that days without quality_score are skipped."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                # Missing quality_score
            },
            {
                'symbol': 'TSLA',
                'date': '2024-01-20', 
                'quality_score': None  # Null quality_score
            },
            {
                'symbol': 'NVDA',
                'date': '2024-01-25',
                'quality_score': 0.8  # Valid
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        # Only NVDA should be included
        assert len(result) == 1
        assert result[0].symbol == "NVDA"

    def test_day_with_invalid_date_is_skipped(self, mock_config, mock_data_manager):
        """Test that days with invalid date formats are skipped."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': 'not-a-real-date-at-all',  # Truly invalid date
                'quality_score': 0.75
            },
            {
                'symbol': 'TSLA',
                'date': '2024-01-20',
                'quality_score': 0.65
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        # Invalid dates should be skipped
        assert len(result) == 1
        assert result[0].symbol == "TSLA"

    def test_day_without_reset_points_is_skipped(self, mock_config, mock_data_manager):
        """Test that days without valid reset points are skipped."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        # Return empty DataFrame for reset points
        mock_data_manager.get_reset_points.return_value = pd.DataFrame()
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert len(result) == 0

    def test_reset_point_missing_required_fields_are_skipped(self, mock_config, mock_data_manager):
        """Test that reset points missing required fields are skipped, and day is skipped if no valid reset points."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        
        # Reset points with all missing required fields
        reset_points_df = pd.DataFrame({
            'timestamp': ['2024-01-15 09:30:00', '2024-01-15 10:00:00'],
            'combined_score': [None, None],  # All missing
            'roc_score': [None, None],  # All missing
            'activity_score': [None, None],  # All missing
            'price': [150.0, 152.0]
        })
        mock_data_manager.get_reset_points.return_value = reset_points_df
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        # Day should be skipped because no valid reset points
        assert len(result) == 0

    def test_reset_point_exception_handling(self, mock_config, mock_data_manager):
        """Test that reset point loading exceptions are handled gracefully."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        mock_data_manager.get_reset_points.side_effect = Exception("Reset points failed")
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        # Day should be skipped due to reset points exception
        assert len(result) == 0

    @pytest.mark.parametrize("date_format", [
        "2024-01-15",           # ISO format
        "2024-01-15T00:00:00Z"  # ISO with time
    ])
    def test_supported_date_formats_are_handled(self, mock_config, mock_data_manager, date_format):
        """Test that supported date formats are properly parsed."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': date_format,
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert len(result) == 1
        assert isinstance(result[0].date, pendulum.Date)

    @pytest.mark.parametrize("invalid_date_format", [
        "01/15/2024",           # US format - not supported by pendulum.parse
        "15-01-2024",           # European format - not supported by pendulum.parse
    ])
    def test_unsupported_date_formats_are_skipped(self, mock_config, mock_data_manager, invalid_date_format):
        """Test that unsupported date formats are skipped."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': invalid_date_format,
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert len(result) == 0  # Should be skipped due to parsing error

    def test_price_defaults_to_zero_when_missing(self, mock_config, mock_data_manager):
        """Test that price defaults to 0.0 when missing from reset points."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        
        reset_points_df = pd.DataFrame({
            'timestamp': ['2024-01-15 09:30:00'],
            'combined_score': [0.8],
            'roc_score': [0.6],
            'activity_score': [0.9]
            # Missing 'price' column
        })
        mock_data_manager.get_reset_points.return_value = reset_points_df
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert len(result) == 1
        assert result[0].reset_points[0].price == 0.0

    def test_data_manager_call_parameters(self, mock_config, mock_data_manager):
        """Test that data manager is called with correct parameters."""
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        # Clear the call from __init__ and call the method directly
        mock_data_manager.reset_mock()
        episode_manager._load_available_days()
        
        # Verify get_all_momentum_days was called with correct parameters
        mock_data_manager.get_all_momentum_days.assert_called_once_with(
            symbols=["AAPL", "TSLA"],
            start_date=pendulum.parse("2024-01-01").date(),
            end_date=pendulum.parse("2024-01-31").date()
        )
        
        # Verify get_reset_points was called for each day
        assert mock_data_manager.get_reset_points.call_count == 2

    def test_logging_output(self, episode_manager, mock_data_manager, caplog):
        """Test that appropriate log messages are generated."""
        with caplog.at_level("INFO"):
            result = episode_manager._load_available_days()
        
        # Check for info logs
        assert "🔍 Loading momentum days with filters:" in caplog.text
        assert "📊 Symbols: ['AAPL', 'TSLA']" in caplog.text
        assert "📊 Found 2 momentum days after filtering" in caplog.text
        assert f"✅ Loaded {len(result)} valid days with reset points" in caplog.text

    def test_empty_momentum_days_returns_empty_list(self, mock_config, mock_data_manager):
        """Test handling of empty momentum days from data manager."""
        mock_data_manager.get_all_momentum_days.return_value = []
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert result == []

    def test_datetime_objects_in_date_field_are_handled(self, mock_config, mock_data_manager):
        """Test that datetime objects in date field are properly handled by to_date()."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': datetime(2024, 1, 15),  # datetime object instead of string
                'quality_score': 0.75
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        # Should be handled correctly by to_date() which converts datetime to date
        assert len(result) == 1
        assert result[0].date == pendulum.date(2024, 1, 15)

    def test_large_dataset_performance(self, mock_config, mock_data_manager):
        """Test performance with larger dataset."""
        # Create 100 momentum days
        momentum_days = []
        for i in range(100):
            momentum_days.append({
                'symbol': f'STOCK{i:03d}',
                'date': f'2024-01-{(i % 30) + 1:02d}',
                'quality_score': 0.5 + (i % 50) / 100
            })
        
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert len(result) == 100
        # Verify all are DayInfo objects
        assert all(isinstance(day, DayInfo) for day in result)

    @pytest.mark.parametrize("quality_score", [0.0, 0.5, 1.0, 0.999999])
    def test_quality_score_edge_values(self, mock_config, mock_data_manager, quality_score):
        """Test handling of edge values for quality scores."""
        momentum_days = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'quality_score': quality_score
            }
        ]
        mock_data_manager.get_all_momentum_days.return_value = momentum_days
        episode_manager = EpisodeManager(mock_config, mock_data_manager)
        
        result = episode_manager._load_available_days()
        
        assert len(result) == 1
        assert result[0].day_score == quality_score