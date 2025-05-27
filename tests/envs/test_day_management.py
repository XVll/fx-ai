"""Test suite for day-based data management in the environment simulator.

This covers:
- Loading full trading days
- Managing multiple reset points within a day
- Transitioning between days
- Background preloading of next day
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Optional
from concurrent.futures import Future

from envs.day_manager import (
    DayManager,
    TradingDay,
    ResetPoint,
    ResetPointSelector,
    DayTransitionHandler
)
from data.data_manager import DataManager
from data.utils.index_utils import IndexManager, MomentumDay
from simulators.market_simulator_v2 import MarketSimulatorV2


class TestDayManager:
    """Test the DayManager component responsible for day-level data management."""
    
    @pytest.fixture
    def config(self):
        """Day manager configuration."""
        return {
            'trading_hours': {
                'session_start': time(4, 0),      # 4 AM
                'session_end': time(20, 0),       # 8 PM
                'market_open': time(9, 30),       # 9:30 AM
                'market_close': time(16, 0),      # 4 PM
            },
            'preloading': {
                'enabled': True,
                'background_thread': True,
                'cache_size': 3  # Keep 3 days in memory
            },
            'reset_points': {
                'fixed_times': [
                    time(9, 30),   # Market open
                    time(10, 30),  # Post-open
                    time(14, 0),   # Afternoon
                    time(15, 30),  # Power hour
                ],
                'min_spacing_minutes': 30,
                'max_per_day': 10
            }
        }
    
    @pytest.fixture
    def mock_data_manager(self):
        """Mock DataManager."""
        manager = Mock(spec=DataManager)
        
        # Sample day data
        def create_day_data(date):
            timestamps = pd.date_range(
                start=date.replace(hour=4),
                end=date.replace(hour=20),
                freq='1s'
            )
            
            return {
                'ohlcv_1s': pd.DataFrame({
                    'open': 10.0,
                    'high': 10.1,
                    'low': 9.9,
                    'close': 10.05,
                    'volume': 10000
                }, index=timestamps),
                'quotes': pd.DataFrame({
                    'bid_price': 10.0,
                    'ask_price': 10.05,
                    'bid_size': 5000,
                    'ask_size': 5000
                }, index=timestamps),
                'trades': pd.DataFrame({
                    'price': 10.02,
                    'size': 100
                }, index=timestamps[::10])
            }
        
        manager.load_day.return_value = True
        manager.get_day_data.side_effect = lambda symbol, date: create_day_data(date)
        
        # Async preloading
        future = Future()
        future.set_result(True)
        manager.preload_day_async.return_value = future
        
        return manager
    
    @pytest.fixture
    def mock_index_manager(self):
        """Mock IndexManager with momentum days."""
        manager = Mock(spec=IndexManager)
        
        # Create sample momentum days
        def create_momentum_day(date, quality):
            return MomentumDay(
                symbol='MLGO',
                date=date,
                quality_score=quality,
                max_intraday_move=0.10 + quality * 0.10,
                volume_multiplier=2.0 + quality * 2.0,
                reset_points=[
                    ResetPoint(
                        timestamp=date.replace(hour=9, minute=30),
                        pattern='market_open',
                        phase='neutral',
                        quality_score=0.7
                    ),
                    ResetPoint(
                        timestamp=date.replace(hour=10, minute=15),
                        pattern='breakout',
                        phase='front_side',
                        quality_score=0.9
                    ),
                    ResetPoint(
                        timestamp=date.replace(hour=14, minute=30),
                        pattern='power_hour_setup',
                        phase='accumulation',
                        quality_score=0.8
                    ),
                ]
            )
        
        momentum_days = [
            create_momentum_day(datetime(2025, 1, 15), 0.9),
            create_momentum_day(datetime(2025, 1, 16), 0.85),
            create_momentum_day(datetime(2025, 1, 17), 0.75),
            create_momentum_day(datetime(2025, 1, 20), 0.95),
            create_momentum_day(datetime(2025, 1, 21), 0.7),
        ]
        
        manager.get_momentum_days.return_value = momentum_days
        manager.get_next_day.side_effect = lambda current: momentum_days[1] if current else momentum_days[0]
        
        return manager
    
    @pytest.fixture
    def day_manager(self, config, mock_data_manager, mock_index_manager):
        """Create DayManager instance."""
        return DayManager(
            config=config,
            data_manager=mock_data_manager,
            index_manager=mock_index_manager,
            symbol='MLGO',
            logger=Mock()
        )
    
    def test_initialization(self, day_manager, config):
        """Test DayManager initialization."""
        assert day_manager.symbol == 'MLGO'
        assert day_manager.session_start == config['trading_hours']['session_start']
        assert day_manager.session_end == config['trading_hours']['session_end']
        assert day_manager.preloading_enabled == config['preloading']['enabled']
        assert day_manager.cache_size == config['preloading']['cache_size']
    
    def test_load_trading_day(self, day_manager, mock_data_manager):
        """Test loading a full trading day."""
        momentum_day = MomentumDay(
            symbol='MLGO',
            date=datetime(2025, 1, 15),
            quality_score=0.9,
            max_intraday_move=0.15,
            volume_multiplier=3.5,
            reset_points=[
                ResetPoint(
                    timestamp=datetime(2025, 1, 15, 9, 30),
                    pattern='breakout',
                    phase='front_side',
                    quality_score=0.95
                )
            ]
        )
        
        # Load the day
        trading_day = day_manager.load_day(momentum_day)
        
        # Check data manager was called
        mock_data_manager.load_day.assert_called_once_with(
            symbol='MLGO',
            date=momentum_day.date,
            with_lookback=True
        )
        
        # Check trading day structure
        assert isinstance(trading_day, TradingDay)
        assert trading_day.date == momentum_day.date
        assert trading_day.symbol == 'MLGO'
        assert trading_day.quality_score == 0.9
        assert len(trading_day.reset_points) > 0
        assert trading_day.data is not None
    
    def test_reset_point_augmentation(self, day_manager):
        """Test augmentation of reset points with fixed times."""
        momentum_day = MomentumDay(
            symbol='MLGO',
            date=datetime(2025, 1, 15),
            quality_score=0.9,
            max_intraday_move=0.15,
            volume_multiplier=3.5,
            reset_points=[
                ResetPoint(
                    timestamp=datetime(2025, 1, 15, 10, 15),
                    pattern='breakout',
                    phase='front_side',
                    quality_score=0.95
                )
            ]
        )
        
        # Load day which should augment reset points
        trading_day = day_manager.load_day(momentum_day)
        
        # Should have momentum reset points plus fixed times
        reset_times = [rp.timestamp.time() for rp in trading_day.reset_points]
        
        # Check fixed times are included
        assert time(9, 30) in reset_times    # Market open
        assert time(10, 30) in reset_times   # Post-open
        assert time(14, 0) in reset_times    # Afternoon
        assert time(15, 30) in reset_times   # Power hour
        
        # Original momentum point should be there too
        assert any(rp.pattern == 'breakout' for rp in trading_day.reset_points)
    
    def test_reset_point_spacing(self, day_manager):
        """Test minimum spacing between reset points."""
        momentum_day = MomentumDay(
            symbol='MLGO',
            date=datetime(2025, 1, 15),
            quality_score=0.9,
            max_intraday_move=0.15,
            volume_multiplier=3.5,
            reset_points=[
                ResetPoint(
                    timestamp=datetime(2025, 1, 15, 9, 30),
                    pattern='open',
                    phase='neutral',
                    quality_score=0.7
                ),
                ResetPoint(
                    timestamp=datetime(2025, 1, 15, 9, 35),  # Too close to previous
                    pattern='breakout',
                    phase='front_side',
                    quality_score=0.9
                ),
                ResetPoint(
                    timestamp=datetime(2025, 1, 15, 10, 30),
                    pattern='continuation',
                    phase='front_side',
                    quality_score=0.85
                ),
            ]
        )
        
        trading_day = day_manager.load_day(momentum_day)
        
        # Check spacing between consecutive reset points
        for i in range(1, len(trading_day.reset_points)):
            time_diff = (trading_day.reset_points[i].timestamp - 
                        trading_day.reset_points[i-1].timestamp)
            minutes_diff = time_diff.total_seconds() / 60
            
            # Should respect minimum spacing
            assert minutes_diff >= day_manager.min_reset_spacing_minutes
    
    def test_current_day_management(self, day_manager):
        """Test management of current trading day."""
        # Initially no day loaded
        assert day_manager.current_day is None
        
        # Load a day
        momentum_day = Mock(date=datetime(2025, 1, 15))
        trading_day = day_manager.load_day(momentum_day)
        
        # Should be set as current
        assert day_manager.current_day == trading_day
        
        # Get current should return it
        assert day_manager.get_current_day() == trading_day
        
        # Load another day
        momentum_day2 = Mock(date=datetime(2025, 1, 16))
        trading_day2 = day_manager.load_day(momentum_day2)
        
        # Should update current
        assert day_manager.current_day == trading_day2
    
    def test_day_cache_management(self, day_manager):
        """Test caching of loaded days."""
        # Load multiple days
        days_to_load = [
            Mock(date=datetime(2025, 1, 15)),
            Mock(date=datetime(2025, 1, 16)),
            Mock(date=datetime(2025, 1, 17)),
            Mock(date=datetime(2025, 1, 20)),  # This should evict oldest
        ]
        
        loaded_days = []
        for momentum_day in days_to_load:
            trading_day = day_manager.load_day(momentum_day)
            loaded_days.append(trading_day)
        
        # Cache should have last 3 days
        assert len(day_manager._day_cache) == 3
        
        # Oldest should be evicted
        assert days_to_load[0].date not in day_manager._day_cache
        
        # Recent ones should be cached
        assert days_to_load[1].date in day_manager._day_cache
        assert days_to_load[2].date in day_manager._day_cache
        assert days_to_load[3].date in day_manager._day_cache
    
    def test_async_preloading(self, day_manager, mock_data_manager, mock_index_manager):
        """Test asynchronous preloading of next day."""
        # Load current day
        current_momentum_day = Mock(date=datetime(2025, 1, 15))
        day_manager.load_day(current_momentum_day)
        
        # Trigger preload of next day
        future = day_manager.preload_next_day()
        
        # Should query index for next day
        mock_index_manager.get_next_day.assert_called()
        
        # Should trigger async load
        mock_data_manager.preload_day_async.assert_called()
        
        # Future should complete
        assert isinstance(future, Future)
        assert future.done()
        assert future.result() is True
    
    def test_preload_with_curriculum(self, day_manager, mock_index_manager):
        """Test preloading respects curriculum constraints."""
        # Set curriculum stage
        day_manager.set_curriculum_constraints(min_quality=0.8, max_difficulty=0.5)
        
        # Trigger preload
        day_manager.preload_next_day()
        
        # Should pass constraints to index manager
        call_args = mock_index_manager.get_next_day.call_args
        assert 'min_quality' in call_args.kwargs
        assert call_args.kwargs['min_quality'] == 0.8
    
    def test_day_transition_handling(self, day_manager):
        """Test smooth transition between days."""
        # Load first day
        day1 = Mock(date=datetime(2025, 1, 15))
        trading_day1 = day_manager.load_day(day1)
        
        # Mark day as completed
        day_manager.mark_day_completed(trading_day1)
        
        # Should be in completed set
        assert trading_day1.date in day_manager._completed_days
        
        # Get next day (should use preloaded if available)
        next_day = day_manager.get_next_trading_day()
        
        # Should be different day
        assert next_day.date != trading_day1.date
        
        # Previous day should still be in cache for reference
        assert trading_day1.date in day_manager._day_cache
    
    def test_reset_point_iterator(self, day_manager):
        """Test iteration through reset points within a day."""
        momentum_day = Mock(
            date=datetime(2025, 1, 15),
            reset_points=[
                ResetPoint(timestamp=datetime(2025, 1, 15, 9, 30)),
                ResetPoint(timestamp=datetime(2025, 1, 15, 10, 30)),
                ResetPoint(timestamp=datetime(2025, 1, 15, 14, 0)),
            ]
        )
        
        trading_day = day_manager.load_day(momentum_day)
        
        # Create iterator
        reset_iterator = day_manager.create_reset_iterator(trading_day)
        
        # Iterate through points
        points = list(reset_iterator)
        assert len(points) >= 3  # At least the momentum points
        
        # Should be in chronological order
        for i in range(1, len(points)):
            assert points[i].timestamp > points[i-1].timestamp
    
    def test_data_validation(self, day_manager, mock_data_manager):
        """Test validation of loaded day data."""
        # Set up data manager to return incomplete data
        mock_data_manager.get_day_data.return_value = {
            'ohlcv_1s': pd.DataFrame(),  # Empty
            'quotes': None,  # Missing
            'trades': pd.DataFrame()
        }
        
        momentum_day = Mock(date=datetime(2025, 1, 15))
        
        # Should raise or handle gracefully
        with pytest.raises(ValueError, match="Invalid or incomplete day data"):
            day_manager.load_day(momentum_day)
    
    def test_memory_cleanup(self, day_manager):
        """Test memory cleanup of old days."""
        # Load many days to test cleanup
        for i in range(10):
            momentum_day = Mock(date=datetime(2025, 1, 15) + timedelta(days=i))
            day_manager.load_day(momentum_day)
        
        # Trigger cleanup
        day_manager.cleanup_old_days()
        
        # Should only keep recent days in cache
        assert len(day_manager._day_cache) <= day_manager.cache_size
        
        # Completed days older than threshold should be removed
        old_date = datetime(2025, 1, 15)
        assert old_date not in day_manager._completed_days


class TestResetPointSelector:
    """Test the ResetPointSelector for choosing episode start points."""
    
    @pytest.fixture
    def selector_config(self):
        """Reset point selector configuration."""
        return {
            'selection_strategy': 'quality_weighted',
            'quality_threshold': 0.6,
            'diversity_bonus': 0.1,
            'recency_penalty': 0.05,
            'pattern_weights': {
                'breakout': 1.2,
                'flush': 1.0,
                'bounce': 1.1,
                'consolidation': 0.8,
                'market_open': 0.9,
                'power_hour': 1.15
            }
        }
    
    @pytest.fixture
    def selector(self, selector_config):
        """Create ResetPointSelector."""
        return ResetPointSelector(selector_config)
    
    @pytest.fixture
    def sample_reset_points(self):
        """Create sample reset points for testing."""
        base_time = datetime(2025, 1, 15, 9, 30)
        
        return [
            ResetPoint(
                timestamp=base_time,
                pattern='market_open',
                phase='neutral',
                quality_score=0.7,
                metadata={'fixed_time': True}
            ),
            ResetPoint(
                timestamp=base_time + timedelta(minutes=45),
                pattern='breakout',
                phase='front_side',
                quality_score=0.95,
                metadata={'volume_surge': 4.5}
            ),
            ResetPoint(
                timestamp=base_time + timedelta(hours=2),
                pattern='consolidation',
                phase='neutral',
                quality_score=0.5,
                metadata={'low_volatility': True}
            ),
            ResetPoint(
                timestamp=base_time + timedelta(hours=5),
                pattern='power_hour',
                phase='accumulation',
                quality_score=0.85,
                metadata={'eod_momentum': True}
            ),
        ]
    
    def test_quality_based_selection(self, selector, sample_reset_points):
        """Test selection based on quality scores."""
        # Select with high quality preference
        selected = selector.select_next(
            sample_reset_points,
            used_indices=[],
            prefer_high_quality=True
        )
        
        # Should select the breakout (highest quality)
        assert selected.pattern == 'breakout'
        assert selected.quality_score == 0.95
    
    def test_pattern_weighting(self, selector, sample_reset_points):
        """Test pattern-based weighting in selection."""
        # Adjust weights to prefer power hour
        selector.pattern_weights['power_hour'] = 2.0
        
        selected = selector.select_next(
            sample_reset_points,
            used_indices=[]
        )
        
        # Despite lower quality, power hour might be selected due to weight
        # This is probabilistic, so we test the weighting calculation
        weights = selector._calculate_selection_weights(sample_reset_points, [])
        
        power_hour_idx = 3
        assert weights[power_hour_idx] > weights[2]  # Higher than consolidation
    
    def test_diversity_bonus(self, selector, sample_reset_points):
        """Test diversity bonus for unused reset points."""
        # Mark first two as used
        used_indices = [0, 1]
        
        weights_before = selector._calculate_selection_weights(
            sample_reset_points, []
        )
        weights_after = selector._calculate_selection_weights(
            sample_reset_points, used_indices
        )
        
        # Unused points should have higher weights
        assert weights_after[2] > weights_before[2]
        assert weights_after[3] > weights_before[3]
    
    def test_sequential_selection(self, selector, sample_reset_points):
        """Test sequential selection doesn't repeat."""
        selected_patterns = []
        used_indices = []
        
        # Select all points
        for _ in range(len(sample_reset_points)):
            selected = selector.select_next(
                sample_reset_points,
                used_indices=used_indices
            )
            
            idx = sample_reset_points.index(selected)
            used_indices.append(idx)
            selected_patterns.append(selected.pattern)
        
        # All should be selected exactly once
        assert len(set(selected_patterns)) == len(sample_reset_points)
        assert len(selected_patterns) == len(sample_reset_points)
    
    def test_minimum_quality_filtering(self, selector, sample_reset_points):
        """Test filtering by minimum quality."""
        selector.quality_threshold = 0.8
        
        # Get valid points
        valid_points = selector.filter_by_quality(sample_reset_points)
        
        # Should only include high quality points
        assert len(valid_points) == 2  # breakout and power_hour
        assert all(rp.quality_score >= 0.8 for rp in valid_points)
    
    def test_time_based_preferences(self, selector, sample_reset_points):
        """Test time-of-day preferences."""
        # Add time preference
        selector.time_preferences = {
            'morning': 1.5,    # 9:30-11:30
            'midday': 0.7,     # 11:30-14:00
            'afternoon': 1.2,  # 14:00-16:00
        }
        
        weights = selector._calculate_selection_weights(sample_reset_points, [])
        
        # Morning points should have higher weights
        assert weights[0] > weights[2]  # market_open > consolidation
        assert weights[1] > weights[2]  # breakout > consolidation


class TestDayTransitionHandler:
    """Test smooth transitions between trading days."""
    
    @pytest.fixture
    def transition_handler(self):
        """Create DayTransitionHandler."""
        return DayTransitionHandler(logger=Mock())
    
    def test_position_carryover_check(self, transition_handler):
        """Test checking for position carryover between days."""
        # Portfolio with position
        portfolio_state = Mock(
            positions={'MLGO': {'quantity': 1000, 'side': 'long'}},
            cash=90000,
            total_value=100000
        )
        
        # Check carryover
        result = transition_handler.check_position_carryover(
            portfolio_state,
            allow_carryover=False
        )
        
        assert result['has_position'] is True
        assert result['requires_closure'] is True
        assert result['position_details']['symbol'] == 'MLGO'
        assert result['position_details']['quantity'] == 1000
    
    def test_day_transition_with_position(self, transition_handler):
        """Test handling day transition with open position."""
        current_day = TradingDay(
            date=datetime(2025, 1, 15),
            symbol='MLGO',
            quality_score=0.9,
            reset_points=[],
            data={}
        )
        
        next_day = TradingDay(
            date=datetime(2025, 1, 16),
            symbol='MLGO',
            quality_score=0.85,
            reset_points=[],
            data={}
        )
        
        portfolio_state = Mock(
            positions={'MLGO': {'quantity': 1000}},
            total_value=102000,
            initial_value=100000
        )
        
        # Handle transition
        transition_info = transition_handler.handle_day_transition(
            current_day=current_day,
            next_day=next_day,
            portfolio_state=portfolio_state,
            force_position_close=True
        )
        
        assert transition_info['from_date'] == current_day.date
        assert transition_info['to_date'] == next_day.date
        assert transition_info['position_closed'] is True
        assert transition_info['day_return'] == 0.02  # 2%
    
    def test_performance_summary_generation(self, transition_handler):
        """Test generation of day performance summary."""
        day_metrics = {
            'total_trades': 15,
            'winning_trades': 9,
            'total_pnl': 2500,
            'max_drawdown': -0.015,
            'episodes_completed': 4,
            'average_episode_return': 0.625
        }
        
        summary = transition_handler.generate_day_summary(
            date=datetime(2025, 1, 15),
            metrics=day_metrics
        )
        
        assert summary['date'] == datetime(2025, 1, 15)
        assert summary['win_rate'] == 0.6  # 9/15
        assert summary['total_pnl'] == 2500
        assert summary['episodes_completed'] == 4
        assert 'performance_grade' in summary  # A/B/C/D/F rating
    
    def test_continuity_preservation(self, transition_handler):
        """Test preservation of important state across days."""
        current_state = {
            'episode_count': 150,
            'total_steps': 45000,
            'curriculum_stage': 'stage_2',
            'performance_history': [0.02, 0.01, -0.005, 0.03]
        }
        
        # Preserve state
        preserved = transition_handler.preserve_training_state(current_state)
        
        # All important fields should be preserved
        assert preserved['episode_count'] == 150
        assert preserved['curriculum_stage'] == 'stage_2'
        assert len(preserved['performance_history']) == 4
        
        # Restore state
        restored = transition_handler.restore_training_state(preserved)
        assert restored == current_state