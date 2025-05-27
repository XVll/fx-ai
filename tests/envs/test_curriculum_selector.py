import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any
import random

from envs.curriculum_selector import (
    CurriculumSelector,
    CurriculumStage,
    SelectionStrategy,
    PerformanceTracker,
    CategoryWeights,
    ResetPointCategory,
    CurriculumConfig
)


class TestCurriculumSelector:
    """Test suite for progressive curriculum-based episode selection."""
    
    @pytest.fixture
    def curriculum_config(self):
        """Configuration for curriculum stages."""
        return CurriculumConfig(
            stages=[
                CurriculumStage(
                    name="stage_1_prime_focus",
                    episode_range=(0, 1000),
                    category_weights={
                        'prime_momentum': 0.8,
                        'risk_scenarios': 0.2,
                        'secondary_momentum': 0.0,
                        'dead_zones': 0.0,
                        'educational': 0.0
                    },
                    min_quality_score=0.7,
                    focus_time_periods=['market_open', 'power_hour']
                ),
                CurriculumStage(
                    name="stage_2_expanded",
                    episode_range=(1000, 3000),
                    category_weights={
                        'prime_momentum': 0.4,
                        'secondary_momentum': 0.3,
                        'risk_scenarios': 0.2,
                        'educational': 0.1,
                        'dead_zones': 0.0
                    },
                    min_quality_score=0.5,
                    focus_time_periods=None  # All times
                ),
                CurriculumStage(
                    name="stage_3_patience",
                    episode_range=(3000, 5000),
                    category_weights={
                        'prime_momentum': 0.3,
                        'secondary_momentum': 0.3,
                        'risk_scenarios': 0.2,
                        'educational': 0.1,
                        'dead_zones': 0.1
                    },
                    min_quality_score=0.3,
                    include_dead_zones=True
                ),
                CurriculumStage(
                    name="stage_4_full_market",
                    episode_range=(5000, None),
                    category_weights={
                        'prime_momentum': 0.2,
                        'secondary_momentum': 0.2,
                        'risk_scenarios': 0.2,
                        'educational': 0.2,
                        'dead_zones': 0.2
                    },
                    min_quality_score=0.0,
                    all_categories_equal=True
                )
            ],
            performance_based_adjustment=True,
            adjustment_threshold=0.6,  # 60% win rate triggers adjustment
            exploration_rate=0.1,      # 10% random selection
            prioritize_unseen_patterns=True
        )
    
    @pytest.fixture
    def mock_performance_tracker(self):
        """Mock performance tracker."""
        tracker = Mock(spec=PerformanceTracker)
        tracker.get_category_performance.return_value = {
            'prime_momentum': {'win_rate': 0.7, 'avg_pnl': 1500, 'count': 100},
            'secondary_momentum': {'win_rate': 0.6, 'avg_pnl': 800, 'count': 50},
            'risk_scenarios': {'win_rate': 0.4, 'avg_pnl': -500, 'count': 75},
            'educational': {'win_rate': 0.5, 'avg_pnl': 200, 'count': 25},
            'dead_zones': {'win_rate': 0.45, 'avg_pnl': -100, 'count': 10}
        }
        tracker.get_pattern_performance.return_value = {
            'breakout': {'win_rate': 0.75, 'count': 80},
            'flush': {'win_rate': 0.35, 'count': 40},
            'bounce': {'win_rate': 0.55, 'count': 30}
        }
        tracker.get_total_episodes.return_value = 0
        return tracker
    
    @pytest.fixture
    def curriculum_selector(self, curriculum_config, mock_performance_tracker):
        """Create curriculum selector instance."""
        return CurriculumSelector(curriculum_config, mock_performance_tracker)
    
    @pytest.fixture
    def sample_reset_points(self):
        """Generate sample categorized reset points."""
        points = {}
        
        # Prime momentum points
        points['prime_momentum'] = [
            Mock(
                timestamp=datetime(2025, 1, 15, 9, 35),
                quality_score=0.92,
                momentum_phase='front_breakout',
                pattern_type='breakout',
                metadata={'volume_surge': 4.5}
            ) for _ in range(20)
        ]
        
        # Secondary momentum
        points['secondary_momentum'] = [
            Mock(
                timestamp=datetime(2025, 1, 15, 11, 0),
                quality_score=0.65,
                momentum_phase='consolidation',
                pattern_type='accumulation',
                metadata={'bid_ask_ratio': 1.3}
            ) for _ in range(15)
        ]
        
        # Risk scenarios
        points['risk_scenarios'] = [
            Mock(
                timestamp=datetime(2025, 1, 15, 10, 15),
                quality_score=0.5,
                momentum_phase='back_flush',
                pattern_type='flush',
                metadata={'decline_velocity': -0.003}
            ) for _ in range(10)
        ]
        
        # Dead zones
        points['dead_zones'] = [
            Mock(
                timestamp=datetime(2025, 1, 15, 12, 30),
                quality_score=0.2,
                momentum_phase='dead',
                pattern_type=None,
                metadata={}
            ) for _ in range(5)
        ]
        
        # Educational
        points['educational'] = [
            Mock(
                timestamp=datetime(2025, 1, 15, 13, 0),
                quality_score=0.4,
                momentum_phase='accumulation',
                pattern_type='failed_breakout',
                metadata={'lesson': 'false_breakout'}
            ) for _ in range(8)
        ]
        
        return points
    
    def test_curriculum_selector_initialization(self, curriculum_selector, curriculum_config):
        """Test curriculum selector initialization."""
        assert curriculum_selector.config == curriculum_config
        assert curriculum_selector.current_stage_index == 0
        assert len(curriculum_selector.stages) == 4
        assert curriculum_selector.selection_history == []
    
    def test_stage_selection_based_on_episode_count(self, curriculum_selector, mock_performance_tracker):
        """Test stage selection based on episode count."""
        # Stage 1 (0-1000)
        mock_performance_tracker.get_total_episodes.return_value = 500
        stage = curriculum_selector.get_current_stage()
        assert stage.name == "stage_1_prime_focus"
        
        # Stage 2 (1000-3000)
        mock_performance_tracker.get_total_episodes.return_value = 2000
        stage = curriculum_selector.get_current_stage()
        assert stage.name == "stage_2_expanded"
        
        # Stage 3 (3000-5000)
        mock_performance_tracker.get_total_episodes.return_value = 4000
        stage = curriculum_selector.get_current_stage()
        assert stage.name == "stage_3_patience"
        
        # Stage 4 (5000+)
        mock_performance_tracker.get_total_episodes.return_value = 6000
        stage = curriculum_selector.get_current_stage()
        assert stage.name == "stage_4_full_market"
    
    def test_reset_point_selection_stage1(self, curriculum_selector, sample_reset_points, mock_performance_tracker):
        """Test reset point selection in stage 1 - prime focus."""
        mock_performance_tracker.get_total_episodes.return_value = 100
        
        # Select multiple points to test distribution
        selections = []
        for _ in range(100):
            point = curriculum_selector.select_reset_point(sample_reset_points)
            selections.append(point)
        
        # Count categories
        category_counts = {'prime_momentum': 0, 'risk_scenarios': 0, 'other': 0}
        for point in selections:
            if point in sample_reset_points['prime_momentum']:
                category_counts['prime_momentum'] += 1
            elif point in sample_reset_points['risk_scenarios']:
                category_counts['risk_scenarios'] += 1
            else:
                category_counts['other'] += 1
        
        # Should be roughly 80% prime, 20% risk (with some exploration)
        assert category_counts['prime_momentum'] > 70
        assert category_counts['risk_scenarios'] > 10
        assert category_counts['other'] < 10  # Only from exploration
    
    def test_performance_based_adjustment(self, curriculum_selector, sample_reset_points, mock_performance_tracker):
        """Test performance-based weight adjustment."""
        mock_performance_tracker.get_total_episodes.return_value = 2000  # Stage 2
        
        # Set high performance for risk scenarios
        mock_performance_tracker.get_category_performance.return_value = {
            'prime_momentum': {'win_rate': 0.5, 'avg_pnl': 500, 'count': 100},
            'risk_scenarios': {'win_rate': 0.8, 'avg_pnl': 2000, 'count': 50},  # Outperforming
            'secondary_momentum': {'win_rate': 0.4, 'avg_pnl': -200, 'count': 75},
            'educational': {'win_rate': 0.5, 'avg_pnl': 0, 'count': 25},
            'dead_zones': {'win_rate': 0.3, 'avg_pnl': -500, 'count': 10}
        }
        
        # Select with adjustment
        adjusted_weights = curriculum_selector._adjust_weights_by_performance(
            curriculum_selector.get_current_stage().category_weights
        )
        
        # Risk scenarios should have increased weight
        base_weight = curriculum_selector.get_current_stage().category_weights['risk_scenarios']
        assert adjusted_weights['risk_scenarios'] > base_weight
        
        # Underperforming categories should have decreased weight
        assert adjusted_weights['secondary_momentum'] < \
               curriculum_selector.get_current_stage().category_weights['secondary_momentum']
    
    def test_exploration_rate(self, curriculum_selector, sample_reset_points, mock_performance_tracker):
        """Test exploration rate for random selection."""
        mock_performance_tracker.get_total_episodes.return_value = 1500  # Stage 2
        
        # Track if we get selections outside expected categories
        exploration_selections = 0
        total_selections = 1000
        
        for _ in range(total_selections):
            point = curriculum_selector.select_reset_point(sample_reset_points)
            
            # In stage 2, dead zones have 0 weight, so any selection from there is exploration
            if point in sample_reset_points['dead_zones']:
                exploration_selections += 1
        
        # Should be roughly 10% (exploration rate)
        exploration_rate = exploration_selections / total_selections
        assert 0.05 < exploration_rate < 0.15  # Allow some variance
    
    def test_prioritize_unseen_patterns(self, curriculum_selector, mock_performance_tracker):
        """Test prioritization of unseen patterns."""
        mock_performance_tracker.get_total_episodes.return_value = 2500
        
        # Create reset points with seen/unseen patterns
        seen_patterns = ['breakout', 'flush', 'bounce']
        unseen_patterns = ['failed_breakout', 'accumulation', 'distribution']
        
        reset_points = {
            'prime_momentum': [
                Mock(
                    timestamp=datetime.now(),
                    quality_score=0.8,
                    pattern_type=pattern,
                    metadata={'seen_count': 50 if pattern in seen_patterns else 0}
                ) for pattern in seen_patterns + unseen_patterns
            ]
        }
        
        # Track selections
        pattern_counts = {p: 0 for p in seen_patterns + unseen_patterns}
        
        for _ in range(100):
            point = curriculum_selector.select_reset_point(reset_points)
            pattern_counts[point.pattern_type] += 1
        
        # Unseen patterns should be selected more often
        avg_seen = sum(pattern_counts[p] for p in seen_patterns) / len(seen_patterns)
        avg_unseen = sum(pattern_counts[p] for p in unseen_patterns) / len(unseen_patterns)
        
        assert avg_unseen > avg_seen
    
    def test_time_period_focus(self, curriculum_selector, mock_performance_tracker):
        """Test focus on specific time periods in early stages."""
        mock_performance_tracker.get_total_episodes.return_value = 500  # Stage 1
        
        # Create points at different times
        reset_points = {
            'prime_momentum': [
                Mock(
                    timestamp=datetime(2025, 1, 15, 9, 30),  # Market open
                    quality_score=0.8,
                    pattern_type='breakout'
                ),
                Mock(
                    timestamp=datetime(2025, 1, 15, 12, 0),  # Midday
                    quality_score=0.8,
                    pattern_type='breakout'
                ),
                Mock(
                    timestamp=datetime(2025, 1, 15, 15, 0),  # Power hour
                    quality_score=0.8,
                    pattern_type='breakout'
                ),
            ]
        }
        
        # Select many times
        time_counts = {9: 0, 12: 0, 15: 0}
        for _ in range(100):
            point = curriculum_selector.select_reset_point(reset_points)
            time_counts[point.timestamp.hour] += 1
        
        # Should prefer market open and power hour
        assert time_counts[9] > time_counts[12]
        assert time_counts[15] > time_counts[12]
    
    def test_quality_threshold_filtering(self, curriculum_selector, mock_performance_tracker):
        """Test quality score threshold filtering."""
        mock_performance_tracker.get_total_episodes.return_value = 100  # Stage 1
        
        # Stage 1 has min quality of 0.7
        reset_points = {
            'prime_momentum': [
                Mock(timestamp=datetime.now(), quality_score=0.9, pattern_type='breakout'),
                Mock(timestamp=datetime.now(), quality_score=0.6, pattern_type='breakout'),  # Below threshold
                Mock(timestamp=datetime.now(), quality_score=0.8, pattern_type='breakout'),
            ]
        }
        
        # Should only select high quality points
        for _ in range(20):
            point = curriculum_selector.select_reset_point(reset_points)
            assert point.quality_score >= 0.7
    
    def test_stage_transition_tracking(self, curriculum_selector, mock_performance_tracker):
        """Test tracking of stage transitions."""
        # Progress through stages
        episode_counts = [100, 1100, 3100, 5100]
        
        for count in episode_counts:
            mock_performance_tracker.get_total_episodes.return_value = count
            stage = curriculum_selector.get_current_stage()
            
            # Check transition logging
            if count > 100:
                assert curriculum_selector.stage_transition_history
                last_transition = curriculum_selector.stage_transition_history[-1]
                assert 'from_stage' in last_transition
                assert 'to_stage' in last_transition
                assert 'episode_count' in last_transition
    
    def test_category_weight_validation(self, curriculum_selector):
        """Test that category weights sum to 1.0."""
        for stage in curriculum_selector.stages:
            total_weight = sum(stage.category_weights.values())
            assert abs(total_weight - 1.0) < 0.01  # Allow small floating point error
    
    def test_selection_history_tracking(self, curriculum_selector, sample_reset_points, mock_performance_tracker):
        """Test tracking of selection history."""
        mock_performance_tracker.get_total_episodes.return_value = 1000
        
        # Make several selections
        for _ in range(10):
            point = curriculum_selector.select_reset_point(sample_reset_points)
        
        # Check history
        assert len(curriculum_selector.selection_history) == 10
        
        history_entry = curriculum_selector.selection_history[0]
        assert 'timestamp' in history_entry
        assert 'reset_point' in history_entry
        assert 'category' in history_entry
        assert 'stage' in history_entry
        assert 'episode_number' in history_entry
    
    def test_get_selection_statistics(self, curriculum_selector, sample_reset_points, mock_performance_tracker):
        """Test generation of selection statistics."""
        mock_performance_tracker.get_total_episodes.return_value = 2000
        
        # Make many selections
        for _ in range(100):
            curriculum_selector.select_reset_point(sample_reset_points)
        
        stats = curriculum_selector.get_selection_statistics()
        
        assert 'total_selections' in stats
        assert stats['total_selections'] == 100
        assert 'category_distribution' in stats
        assert 'pattern_distribution' in stats
        assert 'quality_distribution' in stats
        assert 'current_stage' in stats
    
    def test_reset_point_weighting_function(self, curriculum_selector):
        """Test custom weighting function for reset points."""
        # Test quality-based weighting
        points = [
            Mock(quality_score=0.9, pattern_type='breakout'),
            Mock(quality_score=0.7, pattern_type='flush'),
            Mock(quality_score=0.5, pattern_type='bounce')
        ]
        
        weights = curriculum_selector._calculate_point_weights(points)
        
        # Higher quality should have higher weight
        assert weights[0] > weights[1] > weights[2]
        assert sum(weights) == 1.0  # Normalized
    
    def test_forced_category_selection(self, curriculum_selector, sample_reset_points):
        """Test forced selection from specific category."""
        # Force selection from risk scenarios
        point = curriculum_selector.select_from_category(
            sample_reset_points, 
            'risk_scenarios'
        )
        
        assert point in sample_reset_points['risk_scenarios']
        assert point.momentum_phase == 'back_flush'
    
    def test_empty_category_handling(self, curriculum_selector):
        """Test handling of empty categories."""
        reset_points = {
            'prime_momentum': [],
            'secondary_momentum': [Mock(quality_score=0.7, pattern_type='breakout')],
            'risk_scenarios': [],
            'dead_zones': [],
            'educational': []
        }
        
        # Should handle empty categories gracefully
        point = curriculum_selector.select_reset_point(reset_points)
        assert point is not None
        assert point in reset_points['secondary_momentum']
    
    def test_deterministic_selection_mode(self, curriculum_selector, sample_reset_points):
        """Test deterministic selection for reproducibility."""
        # Set seed for reproducibility
        curriculum_selector.set_random_seed(42)
        
        # Make selections
        selections1 = [
            curriculum_selector.select_reset_point(sample_reset_points)
            for _ in range(10)
        ]
        
        # Reset seed and select again
        curriculum_selector.set_random_seed(42)
        selections2 = [
            curriculum_selector.select_reset_point(sample_reset_points)
            for _ in range(10)
        ]
        
        # Should be identical
        assert selections1 == selections2


class TestPerformanceTracker:
    """Test the PerformanceTracker component."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Create performance tracker instance."""
        return PerformanceTracker()
    
    def test_record_episode_result(self, performance_tracker):
        """Test recording episode results."""
        result = {
            'category': 'prime_momentum',
            'pattern_type': 'breakout',
            'pnl': 1500,
            'win': True,
            'duration': 3600,
            'quality_score': 0.9
        }
        
        performance_tracker.record_episode(result)
        
        assert performance_tracker.get_total_episodes() == 1
        
        category_perf = performance_tracker.get_category_performance()
        assert 'prime_momentum' in category_perf
        assert category_perf['prime_momentum']['count'] == 1
        assert category_perf['prime_momentum']['win_rate'] == 1.0
    
    def test_performance_aggregation(self, performance_tracker):
        """Test performance aggregation by category and pattern."""
        # Record multiple episodes
        episodes = [
            {'category': 'prime_momentum', 'pattern_type': 'breakout', 'pnl': 1000, 'win': True},
            {'category': 'prime_momentum', 'pattern_type': 'breakout', 'pnl': 1500, 'win': True},
            {'category': 'prime_momentum', 'pattern_type': 'flush', 'pnl': -500, 'win': False},
            {'category': 'risk_scenarios', 'pattern_type': 'flush', 'pnl': -800, 'win': False},
            {'category': 'risk_scenarios', 'pattern_type': 'bounce', 'pnl': 400, 'win': True},
        ]
        
        for ep in episodes:
            performance_tracker.record_episode(ep)
        
        # Check category performance
        cat_perf = performance_tracker.get_category_performance()
        assert cat_perf['prime_momentum']['count'] == 3
        assert cat_perf['prime_momentum']['win_rate'] == 2/3
        assert cat_perf['prime_momentum']['avg_pnl'] == (1000 + 1500 - 500) / 3
        
        # Check pattern performance
        pattern_perf = performance_tracker.get_pattern_performance()
        assert pattern_perf['breakout']['count'] == 2
        assert pattern_perf['breakout']['win_rate'] == 1.0
        assert pattern_perf['flush']['win_rate'] == 0.0
    
    def test_moving_window_performance(self, performance_tracker):
        """Test performance tracking over moving window."""
        # Record many episodes
        for i in range(200):
            result = {
                'category': 'prime_momentum',
                'pattern_type': 'breakout',
                'pnl': 1000 if i < 100 else -500,  # Performance degrades
                'win': i < 100,
                'episode_number': i
            }
            performance_tracker.record_episode(result)
        
        # Get recent performance (last 50 episodes)
        recent_perf = performance_tracker.get_recent_performance(window=50)
        
        assert recent_perf['prime_momentum']['win_rate'] == 0.0  # All losses in recent
        assert recent_perf['prime_momentum']['avg_pnl'] == -500