"""Comprehensive integration tests for the complete environment simulator system.

This covers end-to-end workflows including:
- Full episode lifecycle
- Multi-day training sequences
- Curriculum progression
- Performance tracking
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from envs.environment_simulator import EnvironmentSimulator
from envs.day_manager import DayManager
from envs.momentum_episode_manager import MomentumEpisodeManager
from data.data_manager import DataManager
from data.utils.index_utils import IndexManager
from simulators.market_simulator_v2 import MarketSimulatorV2
from simulators.execution_simulator import ExecutionSimulator
from simulators.portfolio_simulator import PortfolioSimulator
from feature.feature_extractor import FeatureExtractor
from rewards.calculator import RewardSystemV2


class TestFullIntegration:
    """Test complete integration of all environment components."""
    
    @pytest.fixture
    def full_config(self):
        """Complete configuration for all components."""
        return {
            'env': {
                'symbol': 'MLGO',
                'mode': 'momentum_trading',
                'single_day_only': True,
                'max_episode_duration': 14400,  # 4 hours
                'max_episode_loss_percent': 0.05,
                'invalid_action_limit': 10
            },
            'data': {
                'provider': 'databento',
                'cache_dir': 'cache/data',
                'lookback_days': 2,
                'preload_next_day': True
            },
            'indices': {
                'path': 'data/indices/momentum',
                'min_quality_score': 0.6,
                'refresh_interval_days': 7
            },
            'curriculum': {
                'enabled': True,
                'adaptive': True,
                'stages': {
                    'beginner': {'episodes': [0, 1000], 'min_quality': 0.8},
                    'intermediate': {'episodes': [1000, 5000], 'min_quality': 0.7},
                    'advanced': {'episodes': [5000, None], 'min_quality': 0.5}
                }
            },
            'simulation': {
                'market_hours': {
                    'session_start': '04:00',
                    'session_end': '20:00',
                    'market_open': '09:30',
                    'market_close': '16:00'
                },
                'execution': {
                    'latency_ms': 100,
                    'slippage_model': 'linear',
                    'commission_per_share': 0.005
                }
            },
            'features': {
                'categories': ['hf', 'mf', 'lf', 'static', 'portfolio'],
                'lookback_windows': {
                    'hf': 60,     # 60 seconds
                    'mf': 300,    # 5 minutes
                    'lf': 3600    # 1 hour
                }
            },
            'reward': {
                'system_version': 'v2',
                'components': ['pnl', 'momentum_alignment', 'time_efficiency', 'risk_management']
            }
        }
    
    @pytest.fixture
    def mock_dependencies(self, full_config):
        """Create all mocked dependencies."""
        # Data Manager
        data_manager = Mock(spec=DataManager)
        data_manager.load_day.return_value = True
        data_manager.get_day_data.return_value = self._create_sample_day_data()
        
        # Index Manager
        index_manager = Mock(spec=IndexManager)
        index_manager.get_momentum_days.return_value = self._create_sample_momentum_days()
        
        # Market Simulator
        market_sim = Mock(spec=MarketSimulatorV2)
        market_sim.get_market_state.return_value = self._create_sample_market_state()
        
        # Execution Simulator
        exec_sim = Mock(spec=ExecutionSimulator)
        exec_sim.simulate_execution.return_value = self._create_sample_execution()
        
        # Portfolio Simulator
        portfolio_sim = Mock(spec=PortfolioSimulator)
        portfolio_sim.get_state.return_value = self._create_sample_portfolio_state()
        
        # Feature Extractor
        feature_ext = Mock(spec=FeatureExtractor)
        feature_ext.extract_features.return_value = np.random.randn(150)
        
        # Reward System
        reward_sys = Mock(spec=RewardSystemV2)
        reward_sys.calculate.return_value = {
            'total_reward': 0.5,
            'components': {'pnl': 0.3, 'momentum': 0.2}
        }
        
        return {
            'data_manager': data_manager,
            'index_manager': index_manager,
            'market_simulator': market_sim,
            'execution_simulator': exec_sim,
            'portfolio_simulator': portfolio_sim,
            'feature_extractor': feature_ext,
            'reward_system': reward_sys
        }
    
    def test_complete_episode_lifecycle(self, full_config, mock_dependencies):
        """Test a complete episode from reset to termination."""
        # Create environment
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Inject mocked components
        for name, component in mock_dependencies.items():
            if hasattr(env, name):
                setattr(env, name, component)
        
        # Reset environment
        obs, info = env.reset()
        
        # Verify initialization
        assert isinstance(obs, np.ndarray)
        assert 'day_date' in info
        assert 'reset_point' in info
        
        # Verify day was loaded
        mock_dependencies['data_manager'].load_day.assert_called_once()
        
        # Run episode
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < 100:
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # Verify step outputs
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(info, dict)
        
        # Verify episode completed
        assert step_count > 0
        assert 'termination_reason' in info
        
        # Verify components were called
        assert mock_dependencies['market_simulator'].get_market_state.called
        assert mock_dependencies['feature_extractor'].extract_features.called
        assert mock_dependencies['reward_system'].calculate.called
    
    def test_multi_episode_single_day(self, full_config, mock_dependencies):
        """Test running multiple episodes within a single trading day."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Set up day with multiple reset points
        momentum_days = mock_dependencies['index_manager'].get_momentum_days.return_value
        momentum_days[0].reset_points = [
            Mock(timestamp=datetime(2025, 1, 15, 9, 30)),
            Mock(timestamp=datetime(2025, 1, 15, 11, 0)),
            Mock(timestamp=datetime(2025, 1, 15, 14, 0)),
        ]
        
        # Run multiple episodes
        episodes_completed = 0
        current_day = None
        
        for _ in range(3):
            obs, info = env.reset()
            
            # Track day
            if current_day is None:
                current_day = info['day_date']
            else:
                # Should be same day
                assert info['day_date'] == current_day
            
            # Quick episode
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            episodes_completed += 1
        
        # Should have completed multiple episodes on same day
        assert episodes_completed == 3
        
        # Data should only be loaded once
        assert mock_dependencies['data_manager'].load_day.call_count == 1
    
    def test_curriculum_progression(self, full_config, mock_dependencies):
        """Test curriculum progression through stages."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Track quality scores requested
        quality_scores_requested = []
        
        def track_quality(symbol, min_quality):
            quality_scores_requested.append(min_quality)
            return mock_dependencies['index_manager'].get_momentum_days.return_value
        
        mock_dependencies['index_manager'].get_momentum_days.side_effect = track_quality
        
        # Simulate progression through stages
        episode_counts = [0, 500, 1500, 6000]  # Different stages
        
        for episode_count in episode_counts:
            env.episode_count = episode_count
            env.reset()
        
        # Should request different quality thresholds
        assert quality_scores_requested[0] == 0.8  # Beginner
        assert quality_scores_requested[1] == 0.8  # Still beginner
        assert quality_scores_requested[2] == 0.7  # Intermediate
        assert quality_scores_requested[3] == 0.5  # Advanced
    
    def test_performance_adaptive_selection(self, full_config, mock_dependencies):
        """Test adaptive episode selection based on performance."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Set up performance tracking
        env.performance_tracker = Mock()
        
        # Simulate poor performance
        env.performance_tracker.get_recent_performance.return_value = {
            'avg_return': -0.02,
            'win_rate': 0.3,
            'consecutive_losses': 5
        }
        
        # Reset should select easier episode
        obs, info = env.reset()
        
        # Check that easier episode was selected
        assert info['reset_point']['quality_score'] >= 0.85
        
        # Simulate good performance
        env.performance_tracker.get_recent_performance.return_value = {
            'avg_return': 0.03,
            'win_rate': 0.7,
            'consecutive_wins': 5
        }
        
        # Reset should allow harder episodes
        obs, info = env.reset()
        
        # Quality requirement should be relaxed
        assert info['reset_point']['quality_score'] >= 0.6
    
    def test_async_day_preloading(self, full_config, mock_dependencies):
        """Test asynchronous preloading of next day."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Set up async preloading
        preload_future = asyncio.Future()
        preload_future.set_result(True)
        mock_dependencies['data_manager'].preload_day_async.return_value = preload_future
        
        # Load first day
        env.reset()
        
        # Complete all reset points on current day
        for _ in range(len(env.current_reset_points)):
            env.current_reset_idx += 1
        
        # Should trigger preload of next day
        assert mock_dependencies['data_manager'].preload_day_async.called
        
        # Next reset should use preloaded day
        env.reset()
        
        # Should not load again (already preloaded)
        assert mock_dependencies['data_manager'].load_day.call_count == 1
    
    def test_position_handling_across_episodes(self, full_config, mock_dependencies):
        """Test position handling when transitioning between episodes."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Set up portfolio with position
        portfolio_with_position = self._create_sample_portfolio_state()
        portfolio_with_position.positions = {
            'MLGO': Mock(
                quantity=1000,
                side='long',
                unrealized_pnl=500
            )
        }
        mock_dependencies['portfolio_simulator'].get_state.return_value = portfolio_with_position
        
        # Complete episode with position
        env.reset()
        
        # Force episode end
        env.episode_state.start_time = pd.Timestamp.now() - pd.Timedelta(hours=5)
        obs, reward, terminated, truncated, info = env.step((0, 0))  # Hold
        
        assert terminated is True
        assert 'final_position' in info
        assert info['final_position']['has_position'] is True
        
        # Next episode should handle position appropriately
        if env.single_day_only and env._is_end_of_day():
            assert info['final_position']['will_force_close'] is True
    
    def test_error_recovery(self, full_config, mock_dependencies):
        """Test environment recovery from various error conditions."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Test 1: Data loading failure
        mock_dependencies['data_manager'].load_day.return_value = False
        
        with pytest.raises(RuntimeError, match="Failed to load day data"):
            env.reset()
        
        # Reset mock
        mock_dependencies['data_manager'].load_day.return_value = True
        
        # Test 2: Execution failure
        mock_dependencies['execution_simulator'].simulate_execution.side_effect = Exception("Network error")
        
        env.reset()
        obs, reward, terminated, truncated, info = env.step((1, 1))  # Buy
        
        # Should handle gracefully
        assert 'execution_error' in info
        assert not (terminated or truncated)  # Don't terminate on execution error
        
        # Test 3: Feature extraction failure
        mock_dependencies['feature_extractor'].extract_features.side_effect = Exception("Feature error")
        
        # Should provide default observation
        obs = env._get_observation()
        assert isinstance(obs, np.ndarray)
        assert np.all(obs == 0)  # Default zeros
    
    def test_metrics_collection(self, full_config, mock_dependencies):
        """Test comprehensive metrics collection during episodes."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Set up metrics collector
        env.metrics_collector = Mock()
        
        # Run episode
        obs, info = env.reset()
        
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Check metrics were collected
        if hasattr(env, 'metrics_collector'):
            assert env.metrics_collector.record_step.called
            assert env.metrics_collector.record_episode.called
    
    def test_training_state_persistence(self, full_config, mock_dependencies):
        """Test saving and loading training state."""
        env1 = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Run some episodes
        for _ in range(5):
            env1.reset()
            for _ in range(20):
                action = env1.action_space.sample()
                obs, reward, terminated, truncated, info = env1.step(action)
                if terminated or truncated:
                    break
        
        # Save state
        state = env1.save_state()
        
        assert 'episode_count' in state
        assert 'curriculum_stage' in state
        assert 'performance_history' in state
        
        # Create new environment and load state
        env2 = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        env2.load_state(state)
        
        # Should have same state
        assert env2.episode_count == env1.episode_count
        assert env2._get_curriculum_stage() == env1._get_curriculum_stage()
    
    def test_render_modes(self, full_config, mock_dependencies):
        """Test different rendering modes."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        env.reset()
        
        # Human-readable render
        env.render_mode = 'human'
        output = env.render()
        assert isinstance(output, str) or output is None
        
        # Structured data render
        env.render_mode = 'data'
        output = env.render()
        if output is not None:
            assert isinstance(output, dict)
            assert 'market_state' in output
            assert 'portfolio_state' in output
            assert 'episode_info' in output
    
    # Helper methods
    def _create_sample_day_data(self):
        """Create sample day data."""
        timestamps = pd.date_range(
            start=datetime(2025, 1, 15, 4),
            end=datetime(2025, 1, 15, 20),
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
                'ask_price': 10.02,
                'bid_size': 5000,
                'ask_size': 5000
            }, index=timestamps)
        }
    
    def _create_sample_momentum_days(self):
        """Create sample momentum days."""
        return [
            Mock(
                symbol='MLGO',
                date=datetime(2025, 1, 15),
                quality_score=0.9,
                reset_points=[
                    Mock(
                        timestamp=datetime(2025, 1, 15, 9, 30),
                        pattern='breakout',
                        phase='front_side',
                        quality_score=0.95
                    )
                ]
            )
        ]
    
    def _create_sample_market_state(self):
        """Create sample market state."""
        return Mock(
            timestamp=pd.Timestamp.now(),
            bid_price=10.0,
            ask_price=10.02,
            bid_size=5000,
            ask_size=5000,
            last_price=10.01,
            volume=100000,
            is_halted=False
        )
    
    def _create_sample_portfolio_state(self):
        """Create sample portfolio state."""
        return Mock(
            timestamp=pd.Timestamp.now(),
            cash=100000,
            positions={},
            total_value=100000,
            buying_power=100000,
            realized_pnl=0,
            unrealized_pnl=0
        )
    
    def _create_sample_execution(self):
        """Create sample execution result."""
        return Mock(
            order_id='TEST_001',
            executed=True,
            executed_size=1000,
            executed_price=10.02,
            commission=5.0,
            slippage=0.002
        )


class TestEndToEndScenarios:
    """Test specific end-to-end trading scenarios."""
    
    @pytest.fixture
    def scenario_env(self, full_config, mock_dependencies):
        """Create environment for scenario testing."""
        env = EnvironmentSimulator(
            config=full_config,
            data_manager=mock_dependencies['data_manager'],
            index_manager=mock_dependencies['index_manager']
        )
        
        # Inject components
        for name, component in mock_dependencies.items():
            if hasattr(env, name):
                setattr(env, name, component)
        
        return env
    
    def test_momentum_breakout_scenario(self, scenario_env):
        """Test typical momentum breakout trading scenario."""
        # Set up breakout context
        scenario_env.reset()
        
        # Simulate breakout detection
        scenario_env.momentum_context = Mock(
            pattern='breakout',
            phase='front_side',
            quality_score=0.9
        )
        
        # Buy on breakout
        obs, reward, terminated, truncated, info = scenario_env.step((1, 2))  # Buy 75%
        
        # Should execute buy
        assert info['action_taken']['type'] == 'buy'
        assert info['action_taken']['size_fraction'] == 0.75
        
        # Simulate price movement up
        scenario_env.portfolio_simulator.get_state.return_value.unrealized_pnl = 500
        
        # Hold for a bit
        for _ in range(30):  # 30 seconds
            obs, reward, terminated, truncated, info = scenario_env.step((0, 0))  # Hold
            if terminated:
                break
        
        # Take profit
        obs, reward, terminated, truncated, info = scenario_env.step((2, 3))  # Sell 100%
        
        # Should have positive reward
        assert reward > 0
    
    def test_risk_management_scenario(self, scenario_env):
        """Test risk management in adverse conditions."""
        scenario_env.reset()
        
        # Enter position
        scenario_env.step((1, 1))  # Buy 50%
        
        # Simulate adverse movement
        scenario_env.portfolio_simulator.get_state.return_value.unrealized_pnl = -2000
        scenario_env.portfolio_simulator.get_state.return_value.total_value = 98000
        
        # Continue trading
        obs, reward, terminated, truncated, info = scenario_env.step((0, 0))  # Hold
        
        # Check if stop loss triggered
        if scenario_env.portfolio_simulator.get_state.return_value.total_value < 95000:
            assert terminated is True
            assert info['termination_reason'] == 'MAX_LOSS_REACHED'
    
    def test_end_of_day_scenario(self, scenario_env):
        """Test end of day position handling."""
        scenario_env.reset()
        
        # Enter position late in day
        scenario_env.market_simulator.get_time_until_close.return_value = 300  # 5 minutes
        
        scenario_env.step((1, 0))  # Buy 25%
        
        # Simulate time passing
        scenario_env.market_simulator.get_time_until_close.return_value = 0
        scenario_env.market_simulator.is_market_open.return_value = False
        
        # Next step should terminate
        obs, reward, terminated, truncated, info = scenario_env.step((0, 0))
        
        assert terminated is True
        assert info['termination_reason'] == 'END_OF_DAY'
        assert info['final_position']['will_force_close'] is True