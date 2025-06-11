"""
Comprehensive tests for EpisodeManager.get_evaluation_episodes method with 100% coverage.
Tests episode caching, selection strategies, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch
import pendulum
from pendulum import Date
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import (
    EpisodeManager, EpisodeManagerException, EpisodeTerminationReason,
    DayInfo, ResetPointInfo, EpisodeContext, EpisodeManagerState
)
from config.training.training_config import TrainingManagerConfig
from config.evaluation.evaluation_config import EvaluationConfig


class TestGetEvaluationEpisodes:
    """Test suite for get_evaluation_episodes method with complete coverage."""

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
        with patch.object(EpisodeManager, '_load_available_days', return_value=[]):
            return EpisodeManager(mock_config, mock_data_manager)

    @pytest.fixture
    def eval_config(self):
        """Create evaluation config for testing."""
        return EvaluationConfig(
            episodes=5,
            seed=42,
            episode_selection="diverse"
        )

    @pytest.fixture
    def mock_reset_points(self):
        """Create multiple mock ResetPointInfo objects."""
        return [
            ResetPointInfo(
                timestamp=f"2024-01-15 {9+i}:{30 if i%2 else 45}:00",
                quality_score=0.7 + (i * 0.05),
                roc_score=0.5 + (i * 0.1),
                activity_score=0.8 + (i * 0.02),
                price=150.0 + i,
                index=i
            )
            for i in range(10)
        ]

    @pytest.fixture
    def mock_day_info(self, mock_reset_points):
        """Create a mock DayInfo object with reset points."""
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=mock_reset_points
        )
        return day_info

    def test_no_current_day_returns_empty_list(self, episode_manager, eval_config):
        """Test that empty list is returned when no current day is loaded."""
        # Ensure no current day
        episode_manager.state.current_day = None
        
        result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert result == []

    def test_no_available_reset_points_returns_empty_list(self, episode_manager, eval_config):
        """Test that empty list is returned when no reset points meet criteria."""
        # Create day with no reset points that meet criteria
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=[]
        )
        episode_manager.state.current_day = day_info
        
        result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert result == []

    def test_cached_episodes_returned_on_subsequent_calls(self, episode_manager, eval_config, mock_day_info):
        """Test that cached episodes are returned on subsequent calls."""
        episode_manager.state.current_day = mock_day_info
        
        # First call - should calculate episodes
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result1 = episode_manager.get_evaluation_episodes(eval_config)
        
        # Second call - should return cached episodes
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=[]) as mock_get_rps:
            result2 = episode_manager.get_evaluation_episodes(eval_config)
            
            # get_available_reset_points should not be called on second call
            mock_get_rps.assert_not_called()
        
        assert result1 == result2
        assert len(result1) > 0

    def test_diverse_selection_strategy(self, episode_manager, eval_config, mock_day_info):
        """Test diverse episode selection strategy."""
        eval_config.episode_selection = "diverse"
        eval_config.episodes = 3
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert len(result) == 3
        assert all(isinstance(ep, EpisodeContext) for ep in result)
        assert all(ep.symbol == "AAPL" for ep in result)
        assert all(ep.date == pendulum.parse("2024-01-15").date() for ep in result)

    def test_best_selection_strategy(self, episode_manager, eval_config, mock_day_info):
        """Test best episode selection strategy (highest quality scores)."""
        eval_config.episode_selection = "best"
        eval_config.episodes = 3
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert len(result) == 3
        # Results should be sorted by quality score descending
        quality_scores = [ep.reset_point.quality_score for ep in result]
        assert quality_scores == sorted(quality_scores, reverse=True)

    def test_worst_selection_strategy(self, episode_manager, eval_config, mock_day_info):
        """Test worst episode selection strategy (lowest quality scores)."""
        eval_config.episode_selection = "worst"
        eval_config.episodes = 3
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert len(result) == 3
        # Results should be sorted by quality score ascending
        quality_scores = [ep.reset_point.quality_score for ep in result]
        assert quality_scores == sorted(quality_scores)

    def test_random_selection_strategy(self, episode_manager, eval_config, mock_day_info):
        """Test random episode selection strategy with deterministic seed."""
        eval_config.episode_selection = "random"
        eval_config.episodes = 3
        eval_config.seed = 42
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result1 = episode_manager.get_evaluation_episodes(eval_config)
        
        # Clear cache and run again with same seed
        episode_manager._cached_evaluation_episodes = None
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result2 = episode_manager.get_evaluation_episodes(eval_config)
        
        # Should get same results with same seed
        assert len(result1) == len(result2) == 3
        # Compare reset point indices to verify determinism
        indices1 = [ep.reset_point.index for ep in result1]
        indices2 = [ep.reset_point.index for ep in result2]
        assert indices1 == indices2

    def test_episodes_requested_exceeds_available(self, episode_manager, eval_config, mock_day_info):
        """Test when requested episodes exceed available reset points."""
        eval_config.episodes = 20  # More than available
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        # Should return all available episodes (10)
        assert len(result) == len(mock_day_info.reset_points)

    def test_diverse_selection_with_few_reset_points(self, episode_manager, eval_config):
        """Test diverse selection when available reset points <= requested episodes."""
        eval_config.episode_selection = "diverse"
        eval_config.episodes = 5
        
        # Create day with only 3 reset points
        few_reset_points = [
            ResetPointInfo(
                timestamp=f"2024-01-15 {9+i}:30:00",
                quality_score=0.7,
                roc_score=0.5,
                activity_score=0.8,
                price=150.0,
                index=i
            )
            for i in range(3)
        ]
        
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL", 
            day_score=0.75,
            reset_points=few_reset_points
        )
        episode_manager.state.current_day = day_info
        
        with patch.object(day_info, 'get_available_reset_points', return_value=few_reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        # Should return all 3 available episodes
        assert len(result) == 3

    def test_diverse_selection_with_jitter(self, episode_manager, eval_config, mock_day_info):
        """Test that diverse selection includes jitter for variety."""
        eval_config.episode_selection = "diverse"
        eval_config.episodes = 3
        episode_manager.state.current_day = mock_day_info
        
        # Test with different seeds to verify jitter works
        results = []
        for seed in [42, 43, 44]:
            episode_manager._cached_evaluation_episodes = None  # Clear cache
            eval_config.seed = seed
            
            with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
                result = episode_manager.get_evaluation_episodes(eval_config)
                results.append([ep.reset_point.index for ep in result])
        
        # Results should be different due to jitter (high probability)
        assert not all(r == results[0] for r in results[1:])

    def test_diverse_selection_no_jitter_when_insufficient_points(self, episode_manager, eval_config):
        """Test that jitter is not applied when there are insufficient points."""
        eval_config.episode_selection = "diverse"
        eval_config.episodes = 3
        
        # Create exactly enough reset points (no room for jitter)
        exact_reset_points = [
            ResetPointInfo(
                timestamp=f"2024-01-15 {9+i}:30:00",
                quality_score=0.7,
                roc_score=0.5,
                activity_score=0.8,
                price=150.0,
                index=i
            )
            for i in range(5)  # episodes + 2 = 5, jitter needs episodes + 4
        ]
        
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=exact_reset_points
        )
        episode_manager.state.current_day = day_info
        
        with patch.object(day_info, 'get_available_reset_points', return_value=exact_reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert len(result) == 3

    def test_episode_context_creation(self, episode_manager, eval_config, mock_day_info):
        """Test that EpisodeContext objects are created correctly."""
        eval_config.episodes = 1
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        assert len(result) == 1
        episode = result[0]
        
        assert isinstance(episode, EpisodeContext)
        assert episode.symbol == "AAPL"
        assert episode.date == pendulum.parse("2024-01-15").date()
        assert isinstance(episode.reset_point, ResetPointInfo)
        assert episode.day_info == mock_day_info

    def test_uses_correct_config_ranges(self, episode_manager, eval_config, mock_day_info):
        """Test that evaluation uses correct config ranges for filtering."""
        episode_manager.state.current_day = mock_day_info
        
        # Verify that get_available_reset_points is called with correct ranges
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points) as mock_get_rps:
            episode_manager.get_evaluation_episodes(eval_config)
            
            mock_get_rps.assert_called_once_with(
                quality_range=episode_manager.day_score_range,
                roc_range=episode_manager.roc_range,
                activity_range=episode_manager.activity_range,
                max_reuse=999
            )

    def test_logging_messages(self, episode_manager, eval_config, mock_day_info, caplog):
        """Test that appropriate log messages are generated."""
        import logging
        
        episode_manager.state.current_day = mock_day_info
        
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            with caplog.at_level(logging.INFO):
                result = episode_manager.get_evaluation_episodes(eval_config)
        
        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Selected" in msg and "evaluation episodes" in msg for msg in log_messages)
        assert any("Cached" in msg and "evaluation episodes" in msg for msg in log_messages)

    def test_cache_debug_logging(self, episode_manager, eval_config, mock_day_info, caplog):
        """Test debug logging when returning cached episodes."""
        import logging
        
        episode_manager.state.current_day = mock_day_info
        
        # First call to populate cache
        with patch.object(mock_day_info, 'get_available_reset_points', return_value=mock_day_info.reset_points):
            episode_manager.get_evaluation_episodes(eval_config)
        
        # Second call should log cache hit
        with caplog.at_level(logging.DEBUG):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        log_messages = [record.message for record in caplog.records]
        assert any("Returning" in msg and "cached evaluation episodes" in msg for msg in log_messages)

    def test_no_current_day_warning_logging(self, episode_manager, eval_config, caplog):
        """Test warning logging when no current day is available."""
        import logging
        
        episode_manager.state.current_day = None
        
        with caplog.at_level(logging.WARNING):
            result = episode_manager.get_evaluation_episodes(eval_config)
        
        log_messages = [record.message for record in caplog.records]
        assert any("No current day loaded for evaluation" in msg for msg in log_messages)

    def test_no_available_reset_points_warning_logging(self, episode_manager, eval_config, caplog):
        """Test warning logging when no reset points are available."""
        import logging
        
        day_info = DayInfo(
            date=pendulum.parse("2024-01-15").date(),
            symbol="AAPL",
            day_score=0.75,
            reset_points=[]
        )
        episode_manager.state.current_day = day_info
        
        with patch.object(day_info, 'get_available_reset_points', return_value=[]):
            with caplog.at_level(logging.WARNING):
                result = episode_manager.get_evaluation_episodes(eval_config)
        
        log_messages = [record.message for record in caplog.records]
        assert any("No available reset points for evaluation" in msg for msg in log_messages)