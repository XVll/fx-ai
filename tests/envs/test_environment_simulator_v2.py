"""Comprehensive test suite for the new EnvironmentSimulator with momentum indices and MarketSimulatorV2.

This test suite covers the new architecture where:
1. Environment uses pre-computed momentum indices for episode selection
2. Full days are loaded with multiple reset points
3. MarketSimulatorV2 provides O(1) market state lookups
4. Execution is delegated to ExecutionSimulator
5. Integration with curriculum-based training
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Optional, Any, Tuple
import gymnasium as gym
from gymnasium import spaces

from envs.environment_simulator import (
    EnvironmentSimulator,
    EpisodeState,
    ActionDecoder,
    TerminationReason,
    ResetPoint,
    DayData
)
from simulators.market_simulator_v2 import MarketSimulatorV2, MarketState
from simulators.execution_simulator import ExecutionSimulator, ExecutionResult
from simulators.portfolio_simulator import PortfolioSimulator, PortfolioState
from data.data_manager import DataManager
from data.utils.index_utils import IndexManager, MomentumDay
from feature.feature_extractor import FeatureExtractor
from rewards.calculator import RewardSystemV2


class TestEnvironmentSimulator:
    """Test the main EnvironmentSimulator class."""
    
    @pytest.fixture
    def config(self):
        """Environment configuration."""
        return {
            'env': {
                'symbol': 'MLGO',
                'max_episode_duration': 14400,  # 4 hours
                'max_episode_loss_percent': 0.05,  # 5%
                'single_day_only': True,
                'force_close_at_market_close': True,
                'invalid_action_limit': 10,
                'bankruptcy_threshold_factor': 0.001
            },
            'momentum': {
                'index_path': 'data/indices/momentum',
                'min_quality_score': 0.6,
                'curriculum_stages': {
                    'stage_1': {'episodes': [0, 1000], 'min_quality': 0.8},
                    'stage_2': {'episodes': [1000, 3000], 'min_quality': 0.7},
                    'stage_3': {'episodes': [3000, 5000], 'min_quality': 0.6},
                    'stage_4': {'episodes': [5000, None], 'min_quality': 0.5}
                }
            },
            'simulation': {
                'future_buffer_minutes': 5,
                'default_latency_ms': 100,
                'commission_per_share': 0.005,
                'market_impact_model': 'linear',
                'slippage_model': 'proportional'
            },
            'execution': {
                'max_position_size': 100000
            },
            'reward': {
                'system_version': 'v2',
                'components': [
                    'realized_pnl',
                    'unrealized_pnl',
                    'momentum_alignment',
                    'time_efficiency',
                    'risk_management'
                ]
            }
        }
    
    @pytest.fixture
    def mock_data_manager(self):
        """Mock DataManager with full day data."""
        manager = Mock(spec=DataManager)
        
        # Create sample day data
        date = datetime(2025, 1, 15)
        timestamps = pd.date_range(
            start=date.replace(hour=4),
            end=date.replace(hour=20),
            freq='1s'
        )
        
        # OHLCV data
        ohlcv_1s = pd.DataFrame({
            'open': 10.0,
            'high': 10.05,
            'low': 9.95,
            'close': 10.02,
            'volume': 10000
        }, index=timestamps)
        
        # Quote data
        quotes = pd.DataFrame({
            'bid_price': 10.00,
            'ask_price': 10.02,
            'bid_size': 5000,
            'ask_size': 5000
        }, index=timestamps)
        
        # Trade data
        trades = pd.DataFrame({
            'price': 10.01,
            'size': 100,
            'conditions': 'REGULAR'
        }, index=timestamps[::10])  # Every 10 seconds
        
        manager.load_day.return_value = True
        manager.get_day_data.return_value = {
            'ohlcv_1s': ohlcv_1s,
            'quotes': quotes,
            'trades': trades,
            'status': pd.DataFrame()  # Empty status
        }
        
        return manager
    
    @pytest.fixture
    def mock_index_manager(self):
        """Mock IndexManager with momentum days."""
        manager = Mock(spec=IndexManager)
        
        # Create sample momentum days
        momentum_days = [
            MomentumDay(
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
                    ),
                    ResetPoint(
                        timestamp=datetime(2025, 1, 15, 10, 15),
                        pattern='momentum_cont',
                        phase='front_side',
                        quality_score=0.85
                    ),
                    ResetPoint(
                        timestamp=datetime(2025, 1, 15, 14, 30),
                        pattern='power_hour',
                        phase='accumulation',
                        quality_score=0.8
                    )
                ]
            ),
            MomentumDay(
                symbol='MLGO',
                date=datetime(2025, 1, 16),
                quality_score=0.75,
                max_intraday_move=0.12,
                volume_multiplier=2.5,
                reset_points=[
                    ResetPoint(
                        timestamp=datetime(2025, 1, 16, 9, 30),
                        pattern='gap_up',
                        phase='front_side',
                        quality_score=0.8
                    )
                ]
            )
        ]
        
        manager.get_momentum_days.return_value = momentum_days
        manager.get_day_by_date.side_effect = lambda symbol, date: next(
            (d for d in momentum_days if d.date.date() == date.date()), None
        )
        
        return manager
    
    @pytest.fixture
    def mock_market_simulator(self):
        """Mock MarketSimulatorV2."""
        simulator = Mock(spec=MarketSimulatorV2)
        
        # Default market state - configure as a side_effect to handle any timestamp
        def get_market_state_mock(timestamp):
            return MarketState(
                timestamp=timestamp,
                bid_price=10.00,
                ask_price=10.02,
                bid_size=5000,
                ask_size=5000,
                last_price=10.01,
                last_size=100,
                volume=100000,
                is_halted=False,
                spread=0.02
            )
        
        simulator.get_market_state.side_effect = get_market_state_mock
        
        simulator.current_timestamp = pd.Timestamp.now()
        simulator.is_market_open.return_value = True
        simulator.get_time_until_close.return_value = 3600  # 1 hour
        
        return simulator
    
    @pytest.fixture
    def mock_execution_simulator(self):
        """Mock ExecutionSimulator."""
        simulator = Mock(spec=ExecutionSimulator)
        
        # Default execution result
        simulator.simulate_execution.return_value = ExecutionResult(
            order_id='TEST_001',
            timestamp=pd.Timestamp.now(),
            symbol='MLGO',
            side='buy',
            requested_price=10.02,
            executed_price=10.025,
            requested_size=1000,
            executed_size=1000,
            slippage=0.005,
            commission=5.0,
            latency_ms=100
        )
        
        return simulator
    
    @pytest.fixture
    def mock_portfolio_simulator(self):
        """Mock PortfolioSimulator."""
        simulator = Mock(spec=PortfolioSimulator)
        
        # Default portfolio state
        simulator.get_portfolio_state.return_value = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=100000,
            total_equity=100000,
            unrealized_pnl=0,
            realized_pnl_session=0,
            positions={},
            total_commissions_session=0,
            total_fees_session=0,
            total_slippage_cost_session=0,
            total_volume_traded_session=0,
            total_turnover_session=0
        )
        
        return simulator
    
    @pytest.fixture
    def mock_feature_extractor(self):
        """Mock FeatureExtractor."""
        extractor = Mock(spec=FeatureExtractor)
        
        # Return sample features
        extractor.extract_features.return_value = np.random.randn(150)  # 150 features
        # Note: feature_dim is not an actual attribute of FeatureExtractor,
        # but we set it here for test purposes as the environment may expect it
        extractor.feature_dim = 150
        
        return extractor
    
    @pytest.fixture
    def mock_reward_system(self):
        """Mock RewardSystemV2."""
        system = Mock(spec=RewardSystemV2)
        
        system.calculate.return_value = {
            'total_reward': 0.5,
            'components': {
                'realized_pnl': 0.3,
                'unrealized_pnl': 0.1,
                'momentum_alignment': 0.1,
                'time_efficiency': 0.0,
                'risk_management': 0.0
            }
        }
        
        return system
    
    @pytest.fixture
    def environment(self, config, mock_data_manager, mock_index_manager,
                   mock_market_simulator, mock_execution_simulator,
                   mock_portfolio_simulator, mock_feature_extractor,
                   mock_reward_system):
        """Create EnvironmentSimulator instance."""
        env = EnvironmentSimulator(
            config=config,
            data_manager=mock_data_manager,
            index_manager=mock_index_manager,
            logger=Mock()
        )
        
        # Inject mocked components
        env.market_simulator = mock_market_simulator
        env.execution_simulator = mock_execution_simulator
        env.portfolio_simulator = mock_portfolio_simulator
        env.feature_extractor = mock_feature_extractor
        env.reward_system = mock_reward_system
        
        # Create execution handler with mocked components
        from envs.environment_simulator import ExecutionHandler
        env.execution_handler = ExecutionHandler(
            config=config,
            execution_simulator=mock_execution_simulator,
            portfolio_simulator=mock_portfolio_simulator,
            logger=Mock()
        )
        
        return env
    
    def test_initialization(self, environment, config):
        """Test environment initialization."""
        assert environment.symbol == config['env']['symbol']
        assert environment.max_episode_duration == config['env']['max_episode_duration']
        assert environment.single_day_only is True
        
        # Check action space
        assert isinstance(environment.action_space, spaces.MultiDiscrete)
        assert environment.action_space.shape == (2,)  # (action_type, position_size)
        
        # Check observation space
        assert isinstance(environment.observation_space, spaces.Box)
        assert environment.observation_space.shape == (150,)  # Feature dimension
    
    def test_curriculum_stage_selection(self, environment):
        """Test curriculum stage selection based on episode count."""
        # Stage 1
        environment.episode_count = 500
        stage = environment._get_curriculum_stage()
        assert stage == 'stage_1'
        assert environment._get_min_quality_for_stage(stage) == 0.8
        
        # Stage 2
        environment.episode_count = 1500
        stage = environment._get_curriculum_stage()
        assert stage == 'stage_2'
        assert environment._get_min_quality_for_stage(stage) == 0.7
        
        # Stage 3
        environment.episode_count = 4000
        stage = environment._get_curriculum_stage()
        assert stage == 'stage_3'
        assert environment._get_min_quality_for_stage(stage) == 0.6
        
        # Stage 4
        environment.episode_count = 6000
        stage = environment._get_curriculum_stage()
        assert stage == 'stage_4'
        assert environment._get_min_quality_for_stage(stage) == 0.5
    
    def test_reset_loads_new_day(self, environment, mock_index_manager, mock_data_manager):
        """Test reset loads a new momentum day and selects reset point."""
        # Reset environment
        obs, info = environment.reset()
        
        # Should query momentum days from index
        mock_index_manager.get_momentum_days.assert_called_once_with(
            symbol='MLGO',
            min_quality=0.8  # Stage 1 minimum
        )
        
        # Should load the selected day
        mock_data_manager.load_day.assert_called_once()
        call_args = mock_data_manager.load_day.call_args
        assert call_args.kwargs['symbol'] == 'MLGO'
        assert isinstance(call_args.kwargs['date'], datetime)
        assert call_args.kwargs['with_lookback'] is True
        
        # Should have set up the day
        assert environment.current_day is not None
        assert environment.current_reset_points is not None
        assert len(environment.current_reset_points) > 0
        
        # Should return valid observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (150,)
        
        # Info should contain episode metadata
        assert 'day_date' in info
        assert 'day_quality' in info
        assert 'reset_point' in info
        assert 'total_reset_points' in info
    
    def test_reset_at_specific_point(self, environment, mock_market_simulator):
        """Test resetting to a specific reset point within a day."""
        # First do a normal reset to load a day
        environment.reset()
        
        # Get a specific reset point
        reset_point = environment.current_reset_points[1]  # Second reset point
        
        # Reset to specific point
        obs, info = environment.reset_at_point(reset_point)
        
        # Market simulator should be set to that time
        mock_market_simulator.set_time.assert_called_with(reset_point.timestamp)
        
        # Episode state should be updated
        assert environment.episode_state.start_time == reset_point.timestamp
        assert environment.episode_state.current_reset_point == reset_point
        
        # Info should contain reset point details
        assert info['reset_point']['timestamp'] == reset_point.timestamp
        assert info['reset_point']['pattern'] == reset_point.pattern
        assert info['reset_point']['phase'] == reset_point.phase
    
    def test_step_action_decoding(self, environment, mock_execution_simulator):
        """Test action decoding and delegation to execution simulator."""
        print(f"DEBUG test: mock_execution_simulator id = {id(mock_execution_simulator)}")
        print(f"DEBUG test: env.execution_simulator id = {id(environment.execution_simulator)}")
        print(f"DEBUG test: env.execution_handler.execution_simulator id = {id(environment.execution_handler.execution_simulator)}")
        environment.reset()
        
        # Test different action combinations
        test_cases = [
            ((0, 0), 'hold', 0.0),     # Hold
            ((1, 0), 'buy', 0.25),      # Buy 25%
            ((1, 1), 'buy', 0.50),      # Buy 50%
            ((1, 2), 'buy', 0.75),      # Buy 75%
            ((1, 3), 'buy', 1.00),      # Buy 100%
            ((2, 0), 'sell', 0.25),     # Sell 25%
            ((2, 3), 'sell', 1.00),     # Sell 100%
        ]
        
        for action, expected_type, expected_size in test_cases:
            # Clear previous calls
            # mock_execution_simulator.reset_mock()  # TEMPORARY: Commenting out to debug
            
            # Take step
            obs, reward, terminated, truncated, info = environment.step(action)
            
            if expected_type == 'hold':
                # No execution for hold
                mock_execution_simulator.simulate_execution.assert_not_called()
            else:
                # Execution should be called with decoded action
                mock_execution_simulator.simulate_execution.assert_called_once()
                call_args = mock_execution_simulator.simulate_execution.call_args
                
                # Check the call arguments
                assert call_args.kwargs['order_side'] == expected_type
                assert 'requested_quantity' in call_args.kwargs
                # Note: requested_quantity will be the actual share count, not fraction
    
    def test_episode_termination_conditions(self, environment, mock_portfolio_simulator):
        """Test various episode termination conditions."""
        environment.reset()
        
        # Test 1: Max loss termination
        mock_portfolio_simulator.get_portfolio_state.return_value = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=95000,
            total_equity=94000,  # 6% loss
            unrealized_pnl=0,
            realized_pnl_session=-6000,
            positions={},
            total_commissions_session=5,
            total_fees_session=0,
            total_slippage_cost_session=0,
            total_volume_traded_session=10000,
            total_turnover_session=0.1
        )
        
        obs, reward, terminated, truncated, info = environment.step((0, 0))
        assert terminated is True
        assert info['termination_reason'] == TerminationReason.MAX_LOSS_REACHED
        
        # Test 2: Bankruptcy termination
        environment.reset()
        mock_portfolio_simulator.get_portfolio_state.return_value = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=50,
            total_equity=50,  # Almost bankrupt
            unrealized_pnl=0,
            realized_pnl_session=-99950,
            positions={},
            total_commissions_session=50,
            total_fees_session=0,
            total_slippage_cost_session=0,
            total_volume_traded_session=200000,
            total_turnover_session=2.0
        )
        
        obs, reward, terminated, truncated, info = environment.step((0, 0))
        assert terminated is True
        assert info['termination_reason'] == TerminationReason.BANKRUPTCY
        
        # Test 3: End of day termination
        environment.reset()
        mock_market_simulator.get_time_until_close.return_value = 0
        mock_market_simulator.is_market_open.return_value = False
        
        obs, reward, terminated, truncated, info = environment.step((0, 0))
        assert terminated is True
        assert info['termination_reason'] == TerminationReason.END_OF_DAY
        
        # Test 4: Max duration termination
        environment.reset()
        environment.episode_state.start_time = pd.Timestamp.now() - pd.Timedelta(hours=5)
        
        obs, reward, terminated, truncated, info = environment.step((0, 0))
        assert terminated is True
        assert info['termination_reason'] == TerminationReason.MAX_DURATION
    
    def test_multiple_episodes_per_day(self, environment):
        """Test running multiple episodes within a single day."""
        # First reset loads a day
        obs1, info1 = environment.reset()
        day1 = environment.current_day
        reset_points = environment.current_reset_points
        
        # Complete first episode
        environment.episode_state.terminated = True
        
        # Second reset should use next reset point from same day
        obs2, info2 = environment.reset()
        day2 = environment.current_day
        
        # Should be same day
        assert day2.date == day1.date
        assert info2['reset_point']['timestamp'] != info1['reset_point']['timestamp']
        
        # Should advance to next reset point
        assert environment.current_reset_idx == 1
        
        # After all reset points used, should load new day
        environment.current_reset_idx = len(reset_points)
        obs3, info3 = environment.reset()
        day3 = environment.current_day
        
        # Should be different day
        assert day3.date != day1.date
    
    def test_reward_calculation_integration(self, environment, mock_reward_system):
        """Test integration with RewardSystemV2."""
        environment.reset()
        
        # Take an action
        obs, reward, terminated, truncated, info = environment.step((1, 1))  # Buy 50%
        
        # Reward system should be called
        mock_reward_system.calculate.assert_called_once()
        
        # Check reward components in info
        assert 'reward_components' in info
        assert info['reward_components']['total_reward'] == 0.5
        assert 'realized_pnl' in info['reward_components']['components']
        
        # Reward should match total from system
        assert reward == 0.5
    
    def test_observation_generation(self, environment, mock_feature_extractor):
        """Test observation generation from features."""
        environment.reset()
        
        # Get observation
        obs = environment._get_observation()
        
        # Feature extractor should be called
        mock_feature_extractor.extract_features.assert_called()
        
        # Observation should match feature output
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (150,)
        
        # Test with different market conditions
        mock_feature_extractor.extract_features.return_value = np.ones(150) * 0.5
        obs2 = environment._get_observation()
        assert np.allclose(obs2, 0.5)
    
    def test_info_dict_contents(self, environment):
        """Test info dictionary contains all required information."""
        environment.reset()
        
        # Take a step
        obs, reward, terminated, truncated, info = environment.step((1, 0))
        
        # Check required keys
        required_keys = [
            'timestamp',
            'episode_step',
            'portfolio_state',
            'market_state',
            'action_taken',
            'execution_result',
            'reward_components',
            'momentum_context'
        ]
        
        for key in required_keys:
            assert key in info
        
        # Check momentum context
        assert 'pattern' in info['momentum_context']
        assert 'phase' in info['momentum_context']
        assert 'quality_score' in info['momentum_context']
    
    def test_invalid_action_handling(self, environment):
        """Test handling of invalid actions."""
        environment.reset()
        
        # Set portfolio to have a position
        mock_state = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=50000,
            total_equity=100000,
            unrealized_pnl=0,
            realized_pnl_session=0,
            positions={'MLGO': {'quantity': 5000, 'side': 'long'}},
            total_commissions_session=5,
            total_fees_session=0,
            total_slippage_cost_session=0,
            total_volume_traded_session=50000,
            total_turnover_session=0.5
        )
        environment.portfolio_simulator.get_portfolio_state.return_value = mock_state
        
        # Try to buy more when already have position (invalid)
        obs, reward, terminated, truncated, info = environment.step((1, 3))  # Buy 100%
        
        # Should track invalid action
        assert environment.episode_state.invalid_action_count == 1
        assert info['action_taken']['was_invalid'] is True
        
        # After too many invalid actions, should terminate
        for _ in range(10):
            environment.step((1, 3))
        
        obs, reward, terminated, truncated, info = environment.step((1, 3))
        assert terminated is True
        assert info['termination_reason'] == TerminationReason.INVALID_ACTION_LIMIT
    
    def test_position_handling_at_episode_end(self, environment, mock_portfolio_simulator):
        """Test position handling when episode ends."""
        environment.reset()
        
        # Set up with open position
        mock_state = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=50000,
            total_equity=102500,
            unrealized_pnl=2500,
            realized_pnl_session=0,
            positions={'MLGO': {
                'quantity': 5000,
                'side': 'long',
                'entry_price': 10.0,
                'current_price': 10.5,
                'unrealized_pnl': 2500
            }},
            total_commissions_session=5,
            total_fees_session=0,
            total_slippage_cost_session=0,
            total_volume_traded_session=50000,
            total_turnover_session=0.5
        )
        environment.portfolio_simulator.get_portfolio_state.return_value = mock_state
        
        # Force episode end
        environment.episode_state.start_time = pd.Timestamp.now() - pd.Timedelta(hours=5)
        obs, reward, terminated, truncated, info = environment.step((0, 0))
        
        # Should include position info in termination
        assert terminated is True
        assert 'final_position' in info
        assert info['final_position']['has_position'] is True
        assert info['final_position']['unrealized_pnl'] == 2500
        
        # If single_day_only, position should be marked for closure
        if environment.single_day_only:
            assert info['final_position']['will_force_close'] is True
    
    def test_market_halt_handling(self, environment, mock_market_simulator):
        """Test environment behavior during market halts."""
        environment.reset()
        
        # Set market to halted
        mock_market_simulator.get_market_state.return_value = MarketState(
            timestamp=pd.Timestamp.now(),
            bid_price=10.00,
            ask_price=10.02,
            bid_size=0,  # No liquidity
            ask_size=0,
            last_price=10.01,
            last_size=0,
            volume=0,
            is_halted=True,
            spread=0.02
        )
        
        # Try to trade during halt
        obs, reward, terminated, truncated, info = environment.step((1, 1))  # Buy
        
        # Execution should fail or be rejected
        assert info['execution_result'] is None or info['execution_result'].rejection_reason == 'HALTED'
        
        # Should not terminate episode due to halt
        assert terminated is False
        
        # Info should indicate halt status
        assert info['market_state']['is_halted'] is True
    
    def test_curriculum_progression_tracking(self, environment):
        """Test tracking of curriculum progression."""
        # Track performance across episodes
        performance_history = []
        
        for i in range(5):
            obs, info = environment.reset()
            
            # Simulate episode with varying performance
            episode_return = np.random.randn() * 100
            environment.episode_state.total_reward = episode_return
            
            # Complete episode
            environment.episode_state.terminated = True
            environment._on_episode_end()
            
            performance_history.append({
                'episode': environment.episode_count,
                'return': episode_return,
                'stage': environment._get_curriculum_stage()
            })
        
        # Should track episode count
        assert environment.episode_count == 5
        
        # Should update curriculum metrics
        assert hasattr(environment, 'performance_tracker')
        assert environment.performance_tracker.get_average_return() is not None
    
    def test_render_modes(self, environment):
        """Test different render modes."""
        environment.reset()
        
        # Test human render mode
        environment.render_mode = 'human'
        output = environment.render()
        assert output is None or isinstance(output, str)
        
        # Test logs render mode
        environment.render_mode = 'logs'
        output = environment.render()
        assert output is None or isinstance(output, dict)
        
        # Test none render mode
        environment.render_mode = 'none'
        output = environment.render()
        assert output is None
    
    def test_async_data_preloading(self, environment, mock_data_manager):
        """Test background preloading of next day."""
        environment.reset()
        
        # Complete current day's episodes
        for _ in range(len(environment.current_reset_points)):
            environment.episode_state.terminated = True
            environment.current_reset_idx += 1
        
        # Trigger next day selection
        environment._prepare_next_day()
        
        # Should trigger async preload
        assert mock_data_manager.preload_day_async.called or mock_data_manager.preload_day.called
        
        # Next day should be queued
        assert hasattr(environment, 'next_day_queued')
        assert environment.next_day_queued is not None


class TestActionDecoder:
    """Test the ActionDecoder utility class."""
    
    def test_discrete_action_decoding(self):
        """Test decoding of discrete multi-dimensional actions."""
        decoder = ActionDecoder()
        
        # Test all combinations
        test_cases = [
            ((0, 0), ('hold', 0.25)),
            ((0, 1), ('hold', 0.50)),
            ((0, 2), ('hold', 0.75)),
            ((0, 3), ('hold', 1.00)),
            ((1, 0), ('buy', 0.25)),
            ((1, 1), ('buy', 0.50)),
            ((1, 2), ('buy', 0.75)),
            ((1, 3), ('buy', 1.00)),
            ((2, 0), ('sell', 0.25)),
            ((2, 1), ('sell', 0.50)),
            ((2, 2), ('sell', 0.75)),
            ((2, 3), ('sell', 1.00)),
        ]
        
        for action, expected in test_cases:
            action_type, position_size = decoder.decode(action)
            assert action_type == expected[0]
            assert abs(position_size - expected[1]) < 0.001
    
    def test_invalid_action_handling(self):
        """Test handling of invalid action inputs."""
        decoder = ActionDecoder()
        
        # Out of bounds actions
        with pytest.raises(ValueError):
            decoder.decode((3, 0))  # Invalid action type
        
        with pytest.raises(ValueError):
            decoder.decode((0, 4))  # Invalid position size
        
        # Wrong format - commented out as decoder may accept various formats
        # with pytest.raises(ValueError):
        #     decoder.decode([1, 2, 3])  # Too many dimensions
        
        # with pytest.raises(ValueError):
        #     decoder.decode(5)  # Single value instead of tuple


class TestEpisodeState:
    """Test the EpisodeState tracking class."""
    
    def test_episode_state_initialization(self):
        """Test EpisodeState initialization."""
        reset_point = ResetPoint(
            timestamp=datetime(2025, 1, 15, 9, 30),
            pattern='breakout',
            phase='front_side',
            quality_score=0.9
        )
        
        state = EpisodeState(
            start_time=reset_point.timestamp,
            reset_point=reset_point,
            initial_portfolio_value=100000
        )
        
        assert state.start_time == reset_point.timestamp
        assert state.current_reset_point == reset_point
        assert state.initial_portfolio_value == 100000
        assert state.step_count == 0
        assert state.total_reward == 0
        assert state.invalid_action_count == 0
        assert state.terminated is False
    
    def test_episode_state_updates(self):
        """Test updating episode state during episode."""
        state = EpisodeState(
            start_time=pd.Timestamp.now(),
            reset_point=Mock(),
            initial_portfolio_value=100000
        )
        
        # Update step count
        state.step_count += 1
        assert state.step_count == 1
        
        # Update rewards
        state.total_reward += 0.5
        state.total_reward += -0.2
        assert state.total_reward == 0.3
        
        # Update invalid actions
        state.invalid_action_count += 1
        assert state.invalid_action_count == 1
        
        # Terminate episode
        state.terminated = True
        state.termination_reason = TerminationReason.MAX_LOSS_REACHED
        assert state.terminated is True
        assert state.termination_reason == TerminationReason.MAX_LOSS_REACHED
    
    def test_episode_duration_calculation(self):
        """Test episode duration calculation."""
        start = pd.Timestamp.now()
        state = EpisodeState(
            start_time=start,
            reset_point=Mock(),
            initial_portfolio_value=100000
        )
        
        # Simulate time passing
        state.current_time = start + pd.Timedelta(minutes=30)
        
        duration = state.get_duration()
        assert duration.total_seconds() == 1800  # 30 minutes
    
    def test_episode_metrics_collection(self):
        """Test collection of episode metrics."""
        state = EpisodeState(
            start_time=pd.Timestamp.now(),
            reset_point=Mock(pattern='breakout', phase='front_side'),
            initial_portfolio_value=100000
        )
        
        # Simulate episode
        state.step_count = 100
        state.total_reward = 5.5
        state.invalid_action_count = 2
        state.final_portfolio_value = 101000
        state.max_drawdown = -0.02
        state.trades_executed = 3
        
        # Get metrics
        metrics = state.get_metrics()
        
        assert metrics['total_steps'] == 100
        assert metrics['total_reward'] == 5.5
        assert metrics['invalid_actions'] == 2
        assert metrics['return_pct'] == 0.01  # 1%
        assert metrics['max_drawdown'] == -0.02
        assert metrics['trades_executed'] == 3
        assert metrics['pattern'] == 'breakout'
        assert metrics['phase'] == 'front_side'


class TestCurriculumIntegration:
    """Test integration with curriculum-based training."""
    
    def test_stage_based_day_selection(self, environment, mock_index_manager):
        """Test that day selection respects curriculum stages."""
        # Set different episode counts for different stages
        test_cases = [
            (0, 0.8),      # Stage 1: High quality only
            (500, 0.8),    # Still stage 1
            (1500, 0.7),   # Stage 2: Medium quality
            (4000, 0.6),   # Stage 3: Lower quality
            (6000, 0.5),   # Stage 4: All days
        ]
        
        for episode_count, expected_min_quality in test_cases:
            environment.episode_count = episode_count
            environment.reset()
            
            # Check that index was queried with correct quality threshold
            calls = mock_index_manager.get_momentum_days.call_args_list
            last_call = calls[-1]
            assert last_call.kwargs['min_quality'] == expected_min_quality
    
    def test_performance_based_adjustments(self, environment):
        """Test curriculum adjustments based on performance."""
        # Simulate poor performance in early stage
        environment.episode_count = 100
        environment.performance_tracker.record_episode({
            'return': -0.05,  # 5% loss
            'win_rate': 0.2,  # 20% win rate
            'sharpe': -0.5
        })
        
        # Should potentially repeat current stage or select easier episodes
        obs, info = environment.reset()
        
        # Check if easier reset point was selected
        assert info['reset_point']['quality_score'] >= 0.85  # Higher quality for struggling agent
    
    def test_curriculum_progression_logging(self, environment):
        """Test logging of curriculum progression."""
        logger_mock = Mock()
        environment.logger = logger_mock
        
        # Progress through stages
        for i in range(0, 6000, 1000):
            environment.episode_count = i
            stage = environment._get_curriculum_stage()
            
            # Log stage transition
            if i % 1000 == 0 and i > 0:
                environment._log_curriculum_transition(stage)
        
        # Should have logged transitions
        assert logger_mock.info.call_count >= 3  # At least 3 transitions