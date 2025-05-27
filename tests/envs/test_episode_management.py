import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any

from envs.episode_manager import (
    EpisodeManager,
    Episode,
    EpisodeState,
    TerminationReason,
    EpisodeConfig,
    EpisodeMetrics,
    ResetPointInfo
)


class TestEpisodeManager:
    """Test suite for episode lifecycle and management."""
    
    @pytest.fixture
    def episode_config(self):
        """Configuration for episode management."""
        return EpisodeConfig(
            max_duration_seconds=14400,  # 4 hours
            max_loss_threshold=0.05,     # 5%
            single_day_only=True,
            force_close_at_market_close=True,
            market_close_time="20:00:00",
            min_episode_duration=60,     # 1 minute minimum
            allow_multiple_per_day=True,
            fixed_reset_times=["09:30:00", "10:30:00", "14:00:00", "15:30:00"]
        )
    
    @pytest.fixture
    def mock_market_simulator(self):
        """Mock market simulator for testing."""
        simulator = Mock()
        simulator.current_time = datetime(2025, 1, 15, 9, 30)
        simulator.get_current_market_state.return_value = {
            'timestamp': datetime(2025, 1, 15, 9, 30),
            'bid': 10.0,
            'ask': 10.01,
            'last': 10.005,
            'volume': 100000
        }
        return simulator
    
    @pytest.fixture
    def mock_portfolio_manager(self):
        """Mock portfolio manager for testing."""
        manager = Mock()
        manager.get_portfolio_state.return_value = {
            'cash': 100000,
            'positions': {},
            'total_value': 100000,
            'unrealized_pnl': 0,
            'realized_pnl': 0
        }
        return manager
    
    @pytest.fixture
    def episode_manager(self, episode_config, mock_market_simulator, mock_portfolio_manager):
        """Create episode manager instance."""
        return EpisodeManager(
            config=episode_config,
            market_simulator=mock_market_simulator,
            portfolio_manager=mock_portfolio_manager
        )
    
    def test_episode_initialization(self, episode_manager, episode_config):
        """Test episode manager initialization."""
        assert episode_manager.config == episode_config
        assert episode_manager.current_episode is None
        assert episode_manager.episode_history == []
        assert episode_manager.episodes_today == 0
    
    def test_start_new_episode(self, episode_manager):
        """Test starting a new episode."""
        reset_point = ResetPointInfo(
            timestamp=datetime(2025, 1, 15, 9, 30),
            momentum_phase="front_breakout",
            quality_score=0.9,
            pattern_type="breakout"
        )
        
        episode = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=reset_point
        )
        
        assert isinstance(episode, Episode)
        assert episode.symbol == "MLGO"
        assert episode.start_time == reset_point.timestamp
        assert episode.state == EpisodeState.ACTIVE
        assert episode.reset_point_info == reset_point
        assert episode_manager.current_episode == episode
        assert episode_manager.episodes_today == 1
    
    def test_episode_termination_conditions(self, episode_manager):
        """Test various episode termination conditions."""
        # Start episode
        episode = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 9, 30),
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        # Test max duration termination
        episode_manager.market_simulator.current_time = episode.start_time + timedelta(hours=5)
        terminated, reason = episode_manager.check_termination_conditions()
        assert terminated is True
        assert reason == TerminationReason.MAX_DURATION
        
        # Reset
        episode_manager.current_episode.state = EpisodeState.ACTIVE
        episode_manager.market_simulator.current_time = episode.start_time + timedelta(minutes=30)
        
        # Test max loss termination
        episode_manager.portfolio_manager.get_portfolio_state.return_value = {
            'cash': 94000,
            'positions': {},
            'total_value': 94000,  # 6% loss
            'unrealized_pnl': -6000,
            'realized_pnl': 0
        }
        
        terminated, reason = episode_manager.check_termination_conditions()
        assert terminated is True
        assert reason == TerminationReason.MAX_LOSS
        
        # Test market close termination
        episode_manager.current_episode.state = EpisodeState.ACTIVE
        episode_manager.market_simulator.current_time = datetime(2025, 1, 15, 20, 0, 5)
        
        terminated, reason = episode_manager.check_termination_conditions()
        assert terminated is True
        assert reason == TerminationReason.MARKET_CLOSE
    
    def test_episode_end_handling(self, episode_manager):
        """Test episode end handling and metrics collection."""
        # Start episode
        start_time = datetime(2025, 1, 15, 9, 30)
        episode = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=start_time,
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        # Simulate some trading
        episode_manager.market_simulator.current_time = start_time + timedelta(hours=1)
        episode_manager.portfolio_manager.get_portfolio_state.return_value = {
            'cash': 95000,
            'positions': {
                'MLGO': {
                    'quantity': 500,
                    'avg_price': 10.0,
                    'current_price': 10.2,
                    'unrealized_pnl': 100,
                    'side': 'long'
                }
            },
            'total_value': 100100,
            'unrealized_pnl': 100,
            'realized_pnl': 0
        }
        
        # End episode
        metrics = episode_manager.end_episode(TerminationReason.MAX_DURATION)
        
        assert episode.state == EpisodeState.COMPLETED
        assert episode.end_time == episode_manager.market_simulator.current_time
        assert episode.termination_reason == TerminationReason.MAX_DURATION
        
        # Check metrics
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.duration_seconds == 3600
        assert metrics.total_pnl == 100
        assert metrics.had_open_position is True
        assert metrics.final_position_info['quantity'] == 500
        
        # Check history
        assert len(episode_manager.episode_history) == 1
        assert episode_manager.episode_history[0] == episode
    
    def test_multiple_episodes_same_day(self, episode_manager):
        """Test handling multiple episodes in the same trading day."""
        date = datetime(2025, 1, 15)
        
        # Episode 1: Morning
        episode1 = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=date.replace(hour=9, minute=30),
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        episode_manager.market_simulator.current_time = date.replace(hour=10, minute=30)
        episode_manager.end_episode(TerminationReason.MAX_DURATION)
        
        # Episode 2: Midday
        episode2 = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=date.replace(hour=11, minute=0),
                momentum_phase="consolidation",
                quality_score=0.6
            )
        )
        
        assert episode_manager.episodes_today == 2
        assert episode2.episode_number == 2
        assert episode_manager.current_episode == episode2
        
        # Episode 3: Power hour
        episode_manager.market_simulator.current_time = date.replace(hour=14, minute=0)
        episode_manager.end_episode(TerminationReason.MANUAL)
        
        episode3 = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=date.replace(hour=14, minute=0),
                momentum_phase="front_momentum",
                quality_score=0.85
            )
        )
        
        assert episode_manager.episodes_today == 3
        assert len(episode_manager.episode_history) == 2  # First two completed
    
    def test_episode_validation(self, episode_manager):
        """Test episode validation rules."""
        # Test starting episode when one is active
        episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 9, 30),
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        with pytest.raises(RuntimeError, match="Episode already active"):
            episode_manager.start_episode(
                symbol="MLGO",
                reset_point=ResetPointInfo(
                    timestamp=datetime(2025, 1, 15, 10, 0),
                    momentum_phase="front_momentum",
                    quality_score=0.8
                )
            )
        
        # Test ending episode when none active
        episode_manager.current_episode = None
        with pytest.raises(RuntimeError, match="No active episode"):
            episode_manager.end_episode(TerminationReason.MANUAL)
    
    def test_episode_state_transitions(self, episode_manager):
        """Test episode state transition logic."""
        episode = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 9, 30),
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        # Active -> Completed
        assert episode.state == EpisodeState.ACTIVE
        episode_manager.end_episode(TerminationReason.MAX_LOSS)
        assert episode.state == EpisodeState.COMPLETED
        
        # Test invalid transition
        with pytest.raises(ValueError):
            episode.state = EpisodeState.ACTIVE  # Can't go back to active
    
    def test_episode_metrics_calculation(self, episode_manager):
        """Test detailed episode metrics calculation."""
        start_time = datetime(2025, 1, 15, 9, 30)
        episode = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=start_time,
                momentum_phase="front_breakout",
                quality_score=0.9,
                metadata={'volume_surge': 3.5}
            )
        )
        
        # Simulate trading activity
        episode.num_trades = 5
        episode.win_trades = 3
        episode.total_volume = 2500
        episode.max_drawdown = -0.02
        
        # End with profit
        episode_manager.market_simulator.current_time = start_time + timedelta(minutes=45)
        episode_manager.portfolio_manager.get_portfolio_state.return_value = {
            'cash': 100000,
            'positions': {},
            'total_value': 102000,
            'unrealized_pnl': 0,
            'realized_pnl': 2000
        }
        
        metrics = episode_manager.end_episode(TerminationReason.MANUAL)
        
        assert metrics.duration_seconds == 2700  # 45 minutes
        assert metrics.total_pnl == 2000
        assert metrics.pnl_percent == 0.02  # 2%
        assert metrics.num_trades == 5
        assert metrics.win_rate == 0.6  # 3/5
        assert metrics.avg_trade_size == 500  # 2500/5
        assert metrics.momentum_phase == "front_breakout"
        assert metrics.reset_quality == 0.9
        assert metrics.metadata['volume_surge'] == 3.5
    
    def test_episode_with_bankruptcy(self, episode_manager):
        """Test episode termination due to bankruptcy."""
        episode = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 9, 30),
                momentum_phase="back_flush",
                quality_score=0.7
            )
        )
        
        # Simulate bankruptcy
        episode_manager.portfolio_manager.get_portfolio_state.return_value = {
            'cash': 100,
            'positions': {},
            'total_value': 100,  # Nearly zero
            'unrealized_pnl': -99900,
            'realized_pnl': 0
        }
        
        terminated, reason = episode_manager.check_termination_conditions()
        assert terminated is True
        assert reason == TerminationReason.BANKRUPTCY
    
    def test_fixed_reset_time_handling(self, episode_manager):
        """Test handling of fixed reset times."""
        # Check if current time matches fixed reset
        test_times = [
            (datetime(2025, 1, 15, 9, 30), True),   # Market open
            (datetime(2025, 1, 15, 10, 30), True),  # Post-open
            (datetime(2025, 1, 15, 11, 45), False), # Random time
            (datetime(2025, 1, 15, 14, 0), True),   # Afternoon
            (datetime(2025, 1, 15, 15, 30), True),  # Power hour
        ]
        
        for test_time, expected in test_times:
            is_fixed = episode_manager.is_fixed_reset_time(test_time)
            assert is_fixed == expected
    
    def test_episode_continuation_tracking(self, episode_manager):
        """Test tracking of position continuation across episodes."""
        # Episode 1 ends with position
        episode1 = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 9, 30),
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        # End with open position
        episode_manager.portfolio_manager.get_portfolio_state.return_value = {
            'cash': 95000,
            'positions': {
                'MLGO': {
                    'quantity': 500,
                    'avg_price': 10.0,
                    'current_price': 10.1,
                    'unrealized_pnl': 50,
                    'side': 'long',
                    'entry_time': datetime(2025, 1, 15, 9, 35)
                }
            },
            'total_value': 100050,
            'unrealized_pnl': 50,
            'realized_pnl': 0
        }
        
        episode_manager.market_simulator.current_time = datetime(2025, 1, 15, 10, 30)
        metrics1 = episode_manager.end_episode(TerminationReason.MAX_DURATION)
        
        assert metrics1.had_open_position is True
        assert metrics1.final_position_info['quantity'] == 500
        
        # Episode 2 starts with inherited position
        episode2 = episode_manager.start_episode(
            symbol="MLGO",
            reset_point=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 10, 30),
                momentum_phase="front_momentum",
                quality_score=0.8
            )
        )
        
        # Check inherited position tracking
        assert episode2.inherited_position is not None
        assert episode2.inherited_position['quantity'] == 500
        assert episode2.inherited_position['entry_price'] == 10.0
        assert episode2.inherited_position['entry_time'] == datetime(2025, 1, 15, 9, 35)
    
    def test_episode_summary_generation(self, episode_manager):
        """Test episode summary generation for analysis."""
        # Create several episodes
        episodes_data = [
            ("front_breakout", 0.9, 1500, 0.03),
            ("back_flush", 0.7, -800, -0.008),
            ("consolidation", 0.4, 200, 0.002),
            ("front_momentum", 0.85, 2200, 0.022)
        ]
        
        for phase, quality, pnl, pnl_pct in episodes_data:
            episode = episode_manager.start_episode(
                symbol="MLGO",
                reset_point=ResetPointInfo(
                    timestamp=datetime(2025, 1, 15, 9, 30),
                    momentum_phase=phase,
                    quality_score=quality
                )
            )
            
            # Simulate results
            episode_manager.portfolio_manager.get_portfolio_state.return_value = {
                'total_value': 100000 + pnl,
                'realized_pnl': pnl
            }
            
            episode_manager.market_simulator.current_time += timedelta(hours=1)
            episode_manager.end_episode(TerminationReason.MANUAL)
        
        # Generate summary
        summary = episode_manager.generate_day_summary()
        
        assert summary['total_episodes'] == 4
        assert summary['total_pnl'] == 3100
        assert summary['win_rate'] == 0.75  # 3/4
        assert summary['avg_pnl_per_episode'] == 775
        assert summary['best_episode_pnl'] == 2200
        assert summary['worst_episode_pnl'] == -800
        assert 'momentum_phase_performance' in summary
        assert 'front_breakout' in summary['momentum_phase_performance']


class TestEpisodeDataStructures:
    """Test episode-related data structures."""
    
    def test_episode_creation(self):
        """Test Episode data class creation."""
        episode = Episode(
            episode_id="ep_001",
            symbol="MLGO",
            start_time=datetime(2025, 1, 15, 9, 30),
            episode_number=1,
            reset_point_info=ResetPointInfo(
                timestamp=datetime(2025, 1, 15, 9, 30),
                momentum_phase="front_breakout",
                quality_score=0.9
            )
        )
        
        assert episode.episode_id == "ep_001"
        assert episode.symbol == "MLGO"
        assert episode.state == EpisodeState.ACTIVE
        assert episode.end_time is None
        assert episode.termination_reason is None
    
    def test_episode_state_enum(self):
        """Test EpisodeState enum values."""
        assert EpisodeState.ACTIVE.value == "active"
        assert EpisodeState.COMPLETED.value == "completed"
        assert EpisodeState.FAILED.value == "failed"
        assert EpisodeState.TERMINATED.value == "terminated"
    
    def test_termination_reason_enum(self):
        """Test TerminationReason enum values."""
        reasons = [
            TerminationReason.MAX_DURATION,
            TerminationReason.MAX_LOSS,
            TerminationReason.BANKRUPTCY,
            TerminationReason.MARKET_CLOSE,
            TerminationReason.DATA_END,
            TerminationReason.INVALID_ACTION,
            TerminationReason.MANUAL
        ]
        
        for reason in reasons:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0
    
    def test_episode_metrics_validation(self):
        """Test EpisodeMetrics validation."""
        # Valid metrics
        metrics = EpisodeMetrics(
            episode_id="ep_001",
            duration_seconds=3600,
            total_pnl=1500,
            pnl_percent=0.015,
            num_trades=10,
            win_rate=0.6,
            max_drawdown=-0.02,
            sharpe_ratio=1.5,
            momentum_phase="front_breakout",
            reset_quality=0.9
        )
        
        assert metrics.duration_seconds == 3600
        assert metrics.total_pnl == 1500
        
        # Test invalid metrics
        with pytest.raises(ValueError):
            EpisodeMetrics(
                episode_id="ep_001",
                duration_seconds=-100,  # Invalid
                total_pnl=1500,
                pnl_percent=0.015
            )
    
    def test_reset_point_info_serialization(self):
        """Test ResetPointInfo serialization."""
        reset_info = ResetPointInfo(
            timestamp=datetime(2025, 1, 15, 9, 30),
            momentum_phase="front_breakout",
            quality_score=0.92,
            pattern_type="breakout",
            volume_ratio=3.5,
            metadata={
                'consolidation_duration': 300,
                'volume_surge': 4.2
            }
        )
        
        # Serialize
        data = reset_info.to_dict()
        assert data['timestamp'] == '2025-01-15T09:30:00'
        assert data['momentum_phase'] == 'front_breakout'
        assert data['quality_score'] == 0.92
        
        # Deserialize
        loaded = ResetPointInfo.from_dict(data)
        assert loaded.timestamp == reset_info.timestamp
        assert loaded.metadata['consolidation_duration'] == 300