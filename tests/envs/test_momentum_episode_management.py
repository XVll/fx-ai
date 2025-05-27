"""Test suite for momentum-based episode management.

This covers:
- Integration with pre-computed momentum indices
- Episode selection based on curriculum
- Reset point quality scoring
- Pattern-based episode initialization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
from pathlib import Path

from envs.momentum_episode_manager import (
    MomentumEpisodeManager,
    EpisodeSelector,
    CurriculumStage,
    MomentumContext,
    PatternType,
    PhaseType
)
from data.utils.index_utils import IndexManager, MomentumDay, ResetPoint


class TestMomentumEpisodeManager:
    """Test the MomentumEpisodeManager that handles episode selection from indices."""
    
    @pytest.fixture
    def config(self):
        """Episode manager configuration."""
        return {
            'indices': {
                'path': 'data/indices/momentum',
                'symbols': ['MLGO', 'GOVX', 'VVPR'],
                'min_quality_score': 0.5,
                'max_days_to_load': 100
            },
            'curriculum': {
                'enabled': True,
                'stages': {
                    'beginner': {
                        'episodes': [0, 1000],
                        'min_quality': 0.8,
                        'patterns': ['breakout', 'market_open'],
                        'avoid_patterns': ['flush', 'parabolic']
                    },
                    'intermediate': {
                        'episodes': [1000, 5000],
                        'min_quality': 0.7,
                        'patterns': None,  # All patterns
                        'avoid_patterns': ['dead']
                    },
                    'advanced': {
                        'episodes': [5000, None],
                        'min_quality': 0.5,
                        'patterns': None,
                        'avoid_patterns': None
                    }
                }
            },
            'selection': {
                'strategy': 'adaptive',  # adaptive, random, sequential
                'performance_weight': 0.3,
                'diversity_weight': 0.2,
                'recency_weight': 0.1,
                'quality_weight': 0.4
            },
            'pattern_preferences': {
                'breakout': 1.2,
                'flush': 0.9,
                'bounce': 1.1,
                'consolidation': 0.7,
                'parabolic': 0.8,
                'market_open': 1.0,
                'power_hour': 1.15
            }
        }
    
    @pytest.fixture
    def mock_index_manager(self):
        """Mock IndexManager with pre-computed indices."""
        manager = Mock(spec=IndexManager)
        
        # Create diverse momentum days
        momentum_days = []
        
        # High quality breakout days
        for i in range(5):
            date = datetime(2025, 1, 15) + timedelta(days=i)
            momentum_days.append(MomentumDay(
                symbol='MLGO',
                date=date,
                quality_score=0.85 + (i % 3) * 0.05,
                max_intraday_move=0.15 + (i % 3) * 0.05,
                volume_multiplier=3.0 + i * 0.5,
                reset_points=[
                    ResetPoint(
                        timestamp=date.replace(hour=9, minute=30),
                        pattern='market_open',
                        phase='neutral',
                        quality_score=0.7
                    ),
                    ResetPoint(
                        timestamp=date.replace(hour=10, minute=15 + i * 5),
                        pattern='breakout',
                        phase='front_side',
                        quality_score=0.9 + (i % 2) * 0.05
                    ),
                    ResetPoint(
                        timestamp=date.replace(hour=14, minute=30),
                        pattern='power_hour',
                        phase='accumulation',
                        quality_score=0.8
                    )
                ]
            ))
        
        # Medium quality mixed days
        for i in range(3):
            date = datetime(2025, 1, 20) + timedelta(days=i)
            momentum_days.append(MomentumDay(
                symbol='MLGO',
                date=date,
                quality_score=0.7 + (i % 2) * 0.05,
                max_intraday_move=0.12,
                volume_multiplier=2.0,
                reset_points=[
                    ResetPoint(
                        timestamp=date.replace(hour=10, minute=0),
                        pattern='bounce',
                        phase='recovery',
                        quality_score=0.75
                    ),
                    ResetPoint(
                        timestamp=date.replace(hour=13, minute=0),
                        pattern='consolidation',
                        phase='neutral',
                        quality_score=0.65
                    )
                ]
            ))
        
        # Low quality/risky days
        for i in range(2):
            date = datetime(2025, 1, 25) + timedelta(days=i)
            momentum_days.append(MomentumDay(
                symbol='MLGO',
                date=date,
                quality_score=0.6,
                max_intraday_move=0.25,  # High volatility
                volume_multiplier=5.0,
                reset_points=[
                    ResetPoint(
                        timestamp=date.replace(hour=11, minute=0),
                        pattern='flush',
                        phase='back_side',
                        quality_score=0.6
                    ),
                    ResetPoint(
                        timestamp=date.replace(hour=15, minute=0),
                        pattern='parabolic',
                        phase='exhaustion',
                        quality_score=0.55
                    )
                ]
            ))
        
        manager.load_index.return_value = {'MLGO': momentum_days}
        manager.get_days_by_quality.side_effect = lambda symbol, min_q: [
            d for d in momentum_days if d.quality_score >= min_q
        ]
        manager.get_days_by_pattern.side_effect = lambda symbol, patterns: [
            d for d in momentum_days 
            if any(rp.pattern in patterns for rp in d.reset_points)
        ]
        
        return manager
    
    @pytest.fixture
    def episode_manager(self, config, mock_index_manager):
        """Create MomentumEpisodeManager instance."""
        return MomentumEpisodeManager(
            config=config,
            index_manager=mock_index_manager,
            logger=Mock()
        )
    
    def test_initialization(self, episode_manager, config):
        """Test episode manager initialization."""
        assert episode_manager.curriculum_enabled == config['curriculum']['enabled']
        assert episode_manager.selection_strategy == config['selection']['strategy']
        assert len(episode_manager.curriculum_stages) == 3
        
        # Check index loading
        episode_manager.index_manager.load_index.assert_called_once_with('MLGO')
    
    def test_curriculum_stage_determination(self, episode_manager):
        """Test determining current curriculum stage."""
        # Beginner stage
        episode_manager.episode_count = 500
        stage = episode_manager.get_current_stage()
        assert stage.name == 'beginner'
        assert stage.min_quality == 0.8
        
        # Intermediate stage
        episode_manager.episode_count = 2500
        stage = episode_manager.get_current_stage()
        assert stage.name == 'intermediate'
        assert stage.min_quality == 0.7
        
        # Advanced stage
        episode_manager.episode_count = 6000
        stage = episode_manager.get_current_stage()
        assert stage.name == 'advanced'
        assert stage.min_quality == 0.5
    
    def test_stage_appropriate_day_selection(self, episode_manager):
        """Test selecting days appropriate for curriculum stage."""
        # Beginner stage - should avoid difficult patterns
        episode_manager.episode_count = 100
        available_days = episode_manager.get_available_days()
        
        # Should filter by quality
        assert all(day.quality_score >= 0.8 for day in available_days)
        
        # Should not include days with flush/parabolic patterns
        for day in available_days:
            patterns = [rp.pattern for rp in day.reset_points]
            assert 'flush' not in patterns
            assert 'parabolic' not in patterns
    
    def test_adaptive_selection_strategy(self, episode_manager):
        """Test adaptive episode selection based on performance."""
        # Simulate performance history
        episode_manager.performance_tracker.update({
            'recent_returns': [-0.02, -0.03, -0.01],  # Struggling
            'win_rate': 0.3,
            'avg_episode_duration': 1800  # 30 minutes
        })
        
        # Select next episode
        next_episode = episode_manager.select_next_episode()
        
        # Should select easier episode due to poor performance
        assert next_episode.reset_point.quality_score >= 0.85
        assert next_episode.reset_point.pattern in ['breakout', 'market_open']
    
    def test_diversity_in_episode_selection(self, episode_manager):
        """Test that episode selection promotes diversity."""
        selected_episodes = []
        
        # Select multiple episodes
        for _ in range(10):
            episode = episode_manager.select_next_episode()
            selected_episodes.append(episode)
            
            # Mark as used
            episode_manager.mark_episode_used(episode)
        
        # Check diversity
        unique_days = set(ep.day.date for ep in selected_episodes)
        unique_patterns = set(ep.reset_point.pattern for ep in selected_episodes)
        
        # Should have selected from multiple days
        assert len(unique_days) >= 3
        
        # Should have variety in patterns
        assert len(unique_patterns) >= 2
    
    def test_pattern_preference_weighting(self, episode_manager):
        """Test pattern preference affects selection."""
        # Get selection weights for episodes
        available_episodes = episode_manager._get_candidate_episodes()
        
        weights = episode_manager._calculate_selection_weights(available_episodes)
        
        # Power hour should have higher weight (1.15x)
        power_hour_episodes = [
            i for i, ep in enumerate(available_episodes)
            if ep.reset_point.pattern == 'power_hour'
        ]
        
        consolidation_episodes = [
            i for i, ep in enumerate(available_episodes)
            if ep.reset_point.pattern == 'consolidation'
        ]
        
        if power_hour_episodes and consolidation_episodes:
            avg_power_hour_weight = np.mean([weights[i] for i in power_hour_episodes])
            avg_consolidation_weight = np.mean([weights[i] for i in consolidation_episodes])
            
            # Power hour should have higher average weight
            assert avg_power_hour_weight > avg_consolidation_weight
    
    def test_performance_based_difficulty_adjustment(self, episode_manager):
        """Test difficulty adjustment based on recent performance."""
        # Good performance - increase difficulty
        episode_manager.performance_tracker.update({
            'recent_returns': [0.02, 0.03, 0.025],
            'win_rate': 0.7,
            'consecutive_wins': 5
        })
        
        difficulty = episode_manager._calculate_target_difficulty()
        assert difficulty > 0.5  # Should target harder episodes
        
        # Poor performance - decrease difficulty  
        episode_manager.performance_tracker.update({
            'recent_returns': [-0.02, -0.01, -0.025],
            'win_rate': 0.3,
            'consecutive_losses': 4
        })
        
        difficulty = episode_manager._calculate_target_difficulty()
        assert difficulty < 0.5  # Should target easier episodes
    
    def test_momentum_context_creation(self, episode_manager):
        """Test creation of momentum context for episodes."""
        # Select an episode
        episode = episode_manager.select_next_episode()
        
        # Create context
        context = episode_manager.create_momentum_context(episode)
        
        assert isinstance(context, MomentumContext)
        assert context.pattern == PatternType(episode.reset_point.pattern)
        assert context.phase == PhaseType(episode.reset_point.phase)
        assert context.quality_score == episode.reset_point.quality_score
        assert context.day_quality == episode.day.quality_score
        assert context.intraday_move == episode.day.max_intraday_move
        assert context.volume_multiplier == episode.day.volume_multiplier
    
    def test_episode_completion_tracking(self, episode_manager):
        """Test tracking of completed episodes."""
        episode = episode_manager.select_next_episode()
        
        # Complete episode with metrics
        completion_metrics = {
            'total_return': 0.015,
            'duration_seconds': 1200,
            'trades_executed': 3,
            'max_drawdown': -0.005,
            'win': True
        }
        
        episode_manager.complete_episode(episode, completion_metrics)
        
        # Should update tracking
        assert episode_manager.episodes_completed == 1
        assert episode in episode_manager._completed_episodes
        
        # Should update performance tracker
        assert episode_manager.performance_tracker.episode_count == 1
    
    def test_index_refresh_handling(self, episode_manager):
        """Test handling of index updates/refreshes."""
        # Simulate index update
        new_momentum_days = [
            MomentumDay(
                symbol='MLGO',
                date=datetime(2025, 2, 1),
                quality_score=0.9,
                max_intraday_move=0.18,
                volume_multiplier=4.0,
                reset_points=[
                    ResetPoint(
                        timestamp=datetime(2025, 2, 1, 10, 0),
                        pattern='breakout',
                        phase='front_side',
                        quality_score=0.95
                    )
                ]
            )
        ]
        
        # Refresh index
        episode_manager.index_manager.load_index.return_value = {'MLGO': new_momentum_days}
        episode_manager.refresh_indices()
        
        # Should have new days available
        available_days = episode_manager.get_available_days()
        assert any(day.date == datetime(2025, 2, 1) for day in available_days)
    
    def test_sequential_mode(self, episode_manager):
        """Test sequential episode selection mode."""
        episode_manager.selection_strategy = 'sequential'
        
        # Get all episodes in order
        all_episodes = episode_manager._get_all_episodes_sorted()
        
        # Select episodes sequentially
        selected = []
        for i in range(min(5, len(all_episodes))):
            episode = episode_manager.select_next_episode()
            selected.append(episode)
            episode_manager.mark_episode_used(episode)
        
        # Should be in chronological order
        for i in range(1, len(selected)):
            assert selected[i].reset_point.timestamp > selected[i-1].reset_point.timestamp
    
    def test_episode_filtering_by_time(self, episode_manager):
        """Test filtering episodes by time of day."""
        # Filter for morning episodes only
        morning_episodes = episode_manager.get_episodes_by_time_range(
            start_time=time(9, 0),
            end_time=time(11, 30)
        )
        
        # All should be within time range
        for episode in morning_episodes:
            hour = episode.reset_point.timestamp.hour
            minute = episode.reset_point.timestamp.minute
            
            time_minutes = hour * 60 + minute
            assert 9 * 60 <= time_minutes <= 11 * 60 + 30
    
    def test_save_and_load_state(self, episode_manager):
        """Test saving and loading episode manager state."""
        # Set some state
        episode_manager.episode_count = 150
        episode_manager.episodes_completed = 145
        
        # Select and complete some episodes
        for _ in range(3):
            episode = episode_manager.select_next_episode()
            episode_manager.complete_episode(episode, {'total_return': 0.01})
        
        # Save state
        state = episode_manager.save_state()
        
        assert state['episode_count'] == 150
        assert state['episodes_completed'] == 148  # 145 + 3
        assert 'completed_episodes' in state
        assert 'performance_history' in state
        
        # Create new manager and load state
        new_manager = MomentumEpisodeManager(
            config=episode_manager.config,
            index_manager=episode_manager.index_manager
        )
        new_manager.load_state(state)
        
        assert new_manager.episode_count == 150
        assert new_manager.episodes_completed == 148


class TestEpisodeSelector:
    """Test the EpisodeSelector utility for different selection strategies."""
    
    @pytest.fixture
    def selector_config(self):
        return {
            'weights': {
                'quality': 0.4,
                'diversity': 0.2,
                'recency': 0.1,
                'performance': 0.3
            },
            'diversity_bonus_per_use': 0.05,
            'recency_penalty_days': 3,
            'performance_adjustment_range': 0.2
        }
    
    @pytest.fixture
    def selector(self, selector_config):
        return EpisodeSelector(selector_config)
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for selection."""
        episodes = []
        
        for i in range(10):
            day = Mock(
                date=datetime(2025, 1, 15) + timedelta(days=i // 3),
                quality_score=0.7 + (i % 3) * 0.1
            )
            
            reset_point = Mock(
                timestamp=day.date.replace(hour=9 + i % 8, minute=30),
                pattern=['breakout', 'bounce', 'flush'][i % 3],
                quality_score=0.6 + (i % 4) * 0.1
            )
            
            episode = Mock(day=day, reset_point=reset_point, id=f'ep_{i}')
            episodes.append(episode)
            
        return episodes
    
    def test_quality_based_weights(self, selector, sample_episodes):
        """Test weight calculation based on quality."""
        weights = selector.calculate_weights(
            sample_episodes,
            used_episodes=[],
            performance_score=0.5
        )
        
        # Higher quality episodes should have higher weights
        high_quality_idx = [
            i for i, ep in enumerate(sample_episodes)
            if ep.reset_point.quality_score >= 0.8
        ]
        
        low_quality_idx = [
            i for i, ep in enumerate(sample_episodes)
            if ep.reset_point.quality_score <= 0.6
        ]
        
        if high_quality_idx and low_quality_idx:
            avg_high = np.mean([weights[i] for i in high_quality_idx])
            avg_low = np.mean([weights[i] for i in low_quality_idx])
            assert avg_high > avg_low
    
    def test_diversity_bonus_application(self, selector, sample_episodes):
        """Test diversity bonus for unused episodes."""
        # Mark some as used
        used_episodes = sample_episodes[:3]
        
        weights_with_used = selector.calculate_weights(
            sample_episodes,
            used_episodes=used_episodes,
            performance_score=0.5
        )
        
        weights_no_used = selector.calculate_weights(
            sample_episodes,
            used_episodes=[],
            performance_score=0.5
        )
        
        # Unused episodes should have higher weights when some are used
        for i in range(3, len(sample_episodes)):
            assert weights_with_used[i] > weights_no_used[i]
    
    def test_recency_penalty(self, selector, sample_episodes):
        """Test recency penalty for recently used episodes."""
        # Mark recent usage
        for i, ep in enumerate(sample_episodes[:3]):
            ep.last_used = datetime.now() - timedelta(hours=i)
        
        weights = selector.calculate_weights(
            sample_episodes,
            used_episodes=[],
            performance_score=0.5
        )
        
        # Recently used should have lower weights
        assert weights[0] < weights[5]  # Most recent vs not used
    
    def test_performance_based_adjustment(self, selector, sample_episodes):
        """Test weight adjustment based on performance."""
        # High performance - prefer harder episodes
        weights_high_perf = selector.calculate_weights(
            sample_episodes,
            used_episodes=[],
            performance_score=0.8
        )
        
        # Low performance - prefer easier episodes  
        weights_low_perf = selector.calculate_weights(
            sample_episodes,
            used_episodes=[],
            performance_score=0.2
        )
        
        # Weights should differ based on performance
        assert not np.allclose(weights_high_perf, weights_low_perf)
    
    def test_weighted_random_selection(self, selector, sample_episodes):
        """Test weighted random selection."""
        weights = selector.calculate_weights(
            sample_episodes,
            used_episodes=[],
            performance_score=0.5
        )
        
        # Select multiple times to test distribution
        selections = []
        for _ in range(100):
            selected = selector.weighted_random_select(sample_episodes, weights)
            selections.append(selected)
        
        # Should have selected multiple different episodes
        unique_selections = set(ep.id for ep in selections)
        assert len(unique_selections) > 1
        
        # Higher weight episodes should be selected more often
        selection_counts = {}
        for ep in selections:
            selection_counts[ep.id] = selection_counts.get(ep.id, 0) + 1
        
        # This is probabilistic, so we just check trend
        high_weight_episodes = sorted(
            enumerate(sample_episodes), 
            key=lambda x: weights[x[0]], 
            reverse=True
        )[:3]
        
        high_weight_count = sum(
            selection_counts.get(ep.id, 0) 
            for _, ep in high_weight_episodes
        )
        
        assert high_weight_count > 30  # Should be selected more than average


class TestMomentumContext:
    """Test the MomentumContext data structure."""
    
    def test_context_creation(self):
        """Test creating momentum context."""
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9,
            day_quality=0.85,
            intraday_move=0.15,
            volume_multiplier=3.5,
            time_of_day='market_open',
            metadata={
                'pre_consolidation': 300,
                'volume_surge': 4.5
            }
        )
        
        assert context.pattern == PatternType.BREAKOUT
        assert context.phase == PhaseType.FRONT_SIDE
        assert context.quality_score == 0.9
        assert context.metadata['pre_consolidation'] == 300
    
    def test_context_momentum_strength(self):
        """Test calculating momentum strength from context."""
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9,
            day_quality=0.85,
            intraday_move=0.15,
            volume_multiplier=3.5
        )
        
        strength = context.calculate_momentum_strength()
        
        # Should be high for strong breakout
        assert strength > 0.8
        
        # Weak context
        weak_context = MomentumContext(
            pattern=PatternType.CONSOLIDATION,
            phase=PhaseType.NEUTRAL,
            quality_score=0.5,
            day_quality=0.6,
            intraday_move=0.05,
            volume_multiplier=1.2
        )
        
        weak_strength = weak_context.calculate_momentum_strength()
        assert weak_strength < 0.5
    
    def test_context_action_hints(self):
        """Test getting action hints from context."""
        # Breakout context
        breakout_context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9
        )
        
        hints = breakout_context.get_action_hints()
        assert hints['preferred_side'] == 'buy'
        assert hints['urgency'] == 'high'
        assert hints['risk_level'] == 'medium'
        
        # Flush context
        flush_context = MomentumContext(
            pattern=PatternType.FLUSH,
            phase=PhaseType.BACK_SIDE,
            quality_score=0.7
        )
        
        hints = flush_context.get_action_hints()
        assert hints['preferred_side'] == 'sell'
        assert hints['urgency'] == 'high'
        assert hints['risk_level'] == 'high'
    
    def test_context_serialization(self):
        """Test context serialization for storage."""
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9,
            day_quality=0.85,
            intraday_move=0.15,
            volume_multiplier=3.5,
            time_of_day='market_open',
            metadata={'test': 'value'}
        )
        
        # Serialize
        data = context.to_dict()
        assert data['pattern'] == 'breakout'
        assert data['phase'] == 'front_side'
        assert data['quality_score'] == 0.9
        
        # Deserialize
        loaded = MomentumContext.from_dict(data)
        assert loaded.pattern == context.pattern
        assert loaded.phase == context.phase
        assert loaded.metadata == context.metadata