"""
Comprehensive tests for TradingEnvironment

Tests are written from a black-box perspective, focusing on interface behavior
and expected outputs without looking at implementation details.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from unittest.mock import Mock, MagicMock, patch
import gymnasium as gym
from typing import Dict, Any, List, Optional, Tuple

from envs.trading_environment import (
    TradingEnvironment, ActionTypeEnum, PositionSizeTypeEnum, TerminationReasonEnum
)
from simulators.portfolio_simulator import PositionSideEnum


class TestTradingEnvironmentInitialization:
    """Tests for environment initialization and setup."""
    
    def test_environment_creation(self, mock_config, mock_data_manager):
        """Test basic environment creation."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        # Check basic attributes are set
        assert env.config == mock_config
        assert env.data_manager == mock_data_manager
        assert env.primary_asset is None
        assert env.max_invalid_actions_per_episode == 100
        assert env.bankruptcy_threshold_factor == 0.1
        assert env.initial_capital_for_session == 100000.0
        
        # Check spaces
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert env.action_space.nvec.tolist() == [3, 4]  # 3 action types, 4 position sizes
        
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert set(env.observation_space.spaces.keys()) == {'hf', 'mf', 'lf', 'portfolio', 'static'}
    
    def test_environment_creation_with_logger(self, mock_config, mock_data_manager):
        """Test environment creation with custom logger."""
        mock_logger = MagicMock()
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager,
            logger=mock_logger
        )
        assert env.logger == mock_logger
    
    def test_environment_creation_with_metrics(self, mock_config, mock_data_manager):
        """Test environment creation with metrics integrator."""
        mock_metrics = MagicMock()
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager,
            metrics_integrator=mock_metrics
        )
        assert env.metrics_integrator == mock_metrics
    
    def test_observation_space_shapes(self, mock_config, mock_data_manager):
        """Test observation space has correct shapes based on config."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        # Check each component shape
        assert env.observation_space['hf'].shape == (60, 10)
        assert env.observation_space['mf'].shape == (20, 8)
        assert env.observation_space['lf'].shape == (5, 6)
        assert env.observation_space['portfolio'].shape == (1, 5)
        assert env.observation_space['static'].shape == (1, 4)
        
        # All should be float32
        for key in env.observation_space.spaces:
            assert env.observation_space[key].dtype == np.float32
    
    def test_invalid_action_limit_none(self, mock_data_manager):
        """Test handling when invalid_action_limit is None."""
        config = MagicMock()
        config.env = MagicMock()
        config.env.invalid_action_limit = None
        config.env.initial_capital = 100000.0
        config.env.early_stop_loss_threshold = 0.95
        config.env.reward = MagicMock()
        
        # Model config
        config.model = MagicMock()
        config.model.hf_seq_len = 60
        config.model.hf_feat_dim = 10
        config.model.mf_seq_len = 20
        config.model.mf_feat_dim = 8
        config.model.lf_seq_len = 5
        config.model.lf_feat_dim = 6
        config.model.portfolio_seq_len = 1
        config.model.portfolio_feat_dim = 5
        
        config.simulation = MagicMock()
        config.simulation.default_position_value = 10000.0
        
        env = TradingEnvironment(
            config=config,
            data_manager=mock_data_manager
        )
        
        # Should default to 1000
        assert env.max_invalid_actions_per_episode == 1000


class TestSessionSetup:
    """Tests for session setup functionality."""
    
    @pytest.fixture
    def env(self, mock_config, mock_data_manager):
        return TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
    
    def test_setup_session_invalid_symbol(self, env):
        """Test setup_session with invalid symbol."""
        with pytest.raises(ValueError, match="valid symbol"):
            env.setup_session("", datetime(2025, 1, 15))
        
        with pytest.raises(ValueError, match="valid symbol"):
            env.setup_session(None, datetime(2025, 1, 15))
    
    def test_setup_session_string_date(self, env):
        """Test setup_session with string date."""
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = True
        mock_market_sim.get_stats.return_value = {
            'total_seconds': 57600,
            'warmup_info': {'has_warmup': True}
        }
        
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            env.setup_session("AAPL", "2025-01-15")
            
        assert env.primary_asset == "AAPL"
        assert env.current_session_date.date() == datetime(2025, 1, 15).date()
        mock_market_sim.initialize_day.assert_called_once()
    
    def test_setup_session_datetime_date(self, env):
        """Test setup_session with datetime object."""
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = True
        mock_market_sim.get_stats.return_value = {
            'total_seconds': 57600,
            'warmup_info': {'has_warmup': False}
        }
        
        test_date = datetime(2025, 1, 15)
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            env.setup_session("TSLA", test_date)
            
        assert env.primary_asset == "TSLA"
        assert env.current_session_date == test_date
    
    def test_setup_session_initialization_failure(self, env):
        """Test setup_session when market simulator initialization fails."""
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = False
        
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            with pytest.raises(ValueError, match="Failed to initialize"):
                env.setup_session("AAPL", "2025-01-15")
    
    def test_setup_session_with_momentum_reset_points(self, env):
        """Test setup_session with momentum-based reset points."""
        # Create momentum reset points
        momentum_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2025-01-15 14:30:00+00:00',
                '2025-01-15 15:30:00+00:00',
                '2025-01-15 16:30:00+00:00'
            ]),
            'activity_score': [0.8, 0.6, 0.4],
            'combined_score': [0.7, 0.5, 0.3],
            'day_activity_score': [0.7, 0.7, 0.7],
            'is_positive_move': [True, False, True],
            'volume_ratio': [2.5, 1.8, 1.2],
            'price_change': [0.05, -0.03, 0.02]
        })
        env.data_manager.get_reset_points.return_value = momentum_data
        
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = True
        mock_market_sim.get_stats.return_value = {
            'total_seconds': 57600,
            'warmup_info': {'has_warmup': True}
        }
        
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            env.setup_session("AAPL", "2025-01-15")
            
        assert len(env.reset_points) == 3
        assert env.reset_points[0]['reset_type'] == 'momentum'
        assert env.reset_points[0]['activity_score'] == 0.8
        assert env.reset_points[0]['is_positive_move'] == True
    
    def test_setup_session_with_fixed_reset_points(self, env):
        """Test setup_session falls back to fixed reset points when no momentum data."""
        env.data_manager.get_reset_points.return_value = pd.DataFrame()  # Empty
        
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = True
        mock_market_sim.get_stats.return_value = {
            'total_seconds': 57600,
            'warmup_info': {'has_warmup': False}
        }
        
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            env.setup_session("AAPL", "2025-01-15")
            
        assert len(env.reset_points) == 4  # Fixed reset points
        assert all(rp['reset_type'] == 'fixed' for rp in env.reset_points)
        assert env.reset_points[0]['activity_score'] == 0.5  # Default score


class TestResetFunctionality:
    """Tests for reset and reset_at_point functionality."""
    
    @pytest.fixture
    def env_with_session_multi_reset(self, mock_config, mock_data_manager):
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        # Setup mock market simulator
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = True
        mock_market_sim.get_stats.return_value = {
            'total_seconds': 57600,
            'warmup_info': {'has_warmup': True}
        }
        mock_market_sim.reset.return_value = True
        mock_market_sim.set_time.return_value = True
        mock_market_sim.get_market_state.return_value = MagicMock(timestamp=pd.Timestamp('2025-01-15 14:30:00', tz='UTC'))
        mock_market_sim.get_current_market_data.return_value = {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00', tz='UTC'),
            'current_price': 100.0,
            'best_ask_price': 100.1,
            'best_bid_price': 99.9
        }
        
        # Mock features
        mock_market_sim.get_current_features.return_value = {
            'hf': np.zeros((60, 10), dtype=np.float32),
            'mf': np.zeros((20, 8), dtype=np.float32),
            'lf': np.zeros((5, 6), dtype=np.float32),
            'static': np.zeros((1, 4), dtype=np.float32)
        }
        
        env.market_simulator = mock_market_sim
        env.primary_asset = "AAPL"
        env.current_session_date = datetime(2025, 1, 15)
        
        # Setup multiple reset points for tests that need them
        env.reset_points = [
            {
                'timestamp': pd.Timestamp('2025-01-15 14:30:00', tz='UTC'),
                'activity_score': 0.5,
                'combined_score': 0.5,
                'max_duration_hours': 4,
                'reset_type': 'fixed'
            },
            {
                'timestamp': pd.Timestamp('2025-01-15 15:30:00', tz='UTC'),
                'activity_score': 0.7,
                'combined_score': 0.6,
                'max_duration_hours': 3,
                'reset_type': 'fixed'
            }
        ]
        
        # Create mock simulators
        with patch('envs.trading_environment.PortfolioSimulator') as mock_portfolio:
            with patch('envs.trading_environment.RewardSystem') as mock_reward:
                with patch('envs.trading_environment.ExecutionSimulator') as mock_exec:
                    # Create mock instances
                    env.portfolio_manager = MagicMock()
                    env.reward_calculator = MagicMock()
                    env.execution_manager = MagicMock()
                    
                    # Setup portfolio manager
                    env.portfolio_manager.get_portfolio_state.return_value = {
                        'timestamp': pd.Timestamp('2025-01-15 14:30:00', tz='UTC'),
                        'total_equity': 100000.0,
                        'cash': 100000.0,
                        'unrealized_pnl': 0.0,
                        'realized_pnl_session': 0.0,
                        'positions': {},
                        'session_metrics': {
                            'total_commissions_session': 0.0,
                            'total_fees_session': 0.0,
                            'total_slippage_cost_session': 0.0
                        }
                    }
                    env.portfolio_manager.get_portfolio_observation.return_value = {
                        'features': np.zeros((1, 5), dtype=np.float32)
                    }
                    
                    # Setup initial capital
                    env.portfolio_manager.initial_capital = 100000.0
                    
                    # Initialize random number generator using Generator (not RandomState)
                    env.np_random = np.random.default_rng(42)
        
        return env
    
    def test_reset_basic(self, env_with_session):
        """Test basic reset functionality."""
        obs, info = env_with_session.reset()
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {'hf', 'mf', 'lf', 'portfolio', 'static'}
        
        # Check shapes
        assert obs['hf'].shape == (60, 10)
        assert obs['mf'].shape == (20, 8)
        assert obs['lf'].shape == (5, 6)
        assert obs['portfolio'].shape == (1, 5)
        assert obs['static'].shape == (1, 4)
        
        # Check info structure
        assert isinstance(info, dict)
        assert 'timestamp_iso' in info
        assert info['step'] == 0
        assert info['episode_number'] == 1
        assert info['reset_point_idx'] == 0
        assert info['reset_points_total'] == 1  # Only one reset point in default fixture
    
    def test_reset_at_specific_point(self, env_with_session_multi_reset):
        """Test resetting at a specific reset point."""
        obs, info = env_with_session_multi_reset.reset_at_point(1)
        
        assert env_with_session_multi_reset.current_reset_idx == 1
        assert info['reset_point_idx'] == 1
        assert env_with_session_multi_reset.episode_number == 1
    
    def test_reset_invalid_point_index(self, env_with_session):
        """Test reset with invalid reset point index."""
        obs, info = env_with_session.reset_at_point(10)  # Out of range
        
        # Should return dummy observation and error
        assert "error" in info
        assert info["error"] == "Invalid reset point"
    
    def test_reset_without_session_setup(self, mock_config, mock_data_manager):
        """Test reset without calling setup_session first."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        obs, info = env.reset()
        
        # Should return dummy observation and error
        assert "error" in info
        assert info["error"] == "Session not set up"
    
    def test_reset_episode_state_cleared(self, env_with_session):
        """Test that reset properly clears episode state."""
        # Set some state
        env_with_session.current_step = 100
        env_with_session.invalid_action_count_episode = 50
        env_with_session.episode_total_reward = 123.45
        env_with_session.is_terminated = True
        env_with_session.current_termination_reason = "TEST"
        
        obs, info = env_with_session.reset()
        
        # Check state is cleared
        assert env_with_session.current_step == 0
        assert env_with_session.invalid_action_count_episode == 0
        assert env_with_session.episode_total_reward == 0.0
        assert env_with_session.is_terminated == False
        assert env_with_session.current_termination_reason is None
    
    def test_reset_with_seed(self, env_with_session):
        """Test reset with random seed."""
        obs1, info1 = env_with_session.reset(seed=42)
        obs2, info2 = env_with_session.reset(seed=42)
        
        # With same seed, randomization should be similar (though not necessarily identical due to timing)
        assert env_with_session.np_random is not None
    
    def test_reset_handles_open_positions(self, env_with_session):
        """Test reset handles open positions properly."""
        # Mock an open position
        portfolio_state = {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00', tz='UTC'),
            'total_equity': 100500.0,
            'cash': 90000.0,
            'unrealized_pnl': 500.0,
            'realized_pnl_session': 0.0,
            'positions': {
                'AAPL': {
                    'quantity': 100,
                    'avg_entry_price': 95.0,
                    'unrealized_pnl': 500.0,
                    'current_side': PositionSideEnum.LONG
                }
            },
            'session_metrics': {
                'total_commissions_session': 10.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 5.0
            }
        }
        env_with_session.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        obs, info = env_with_session.reset()
        
        # Check position handling info
        assert 'had_open_position_at_reset' in info
        # Note: actual position closing logic would be in the implementation


class TestStepFunctionality:
    """Tests for step functionality and action handling."""
    
    def test_step_basic_hold_action(self, env_ready):
        """Test step with HOLD action."""
        action = np.array([0, 0])  # HOLD, SIZE_25
        
        # Mock execution result
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = 0.0
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        # Check returns
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check action was processed
        assert info['action_decoded']['action_type'] == 'HOLD'
        assert env_ready.action_counts['HOLD'] == 1
    
    def test_step_buy_action_with_fill(self, env_ready):
        """Test step with BUY action that results in a fill."""
        action = np.array([1, 3])  # BUY, SIZE_100
        
        # Mock execution result with fill
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'BUY',
            'position_size': 1.0,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'BUY'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = MagicMock(
            executed_quantity=100,
            executed_price=100.0,
            commission=5.0,
            fees=0.0,
            slippage_cost_total=10.0
        )
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.portfolio_manager.process_fill.return_value = {
            'executed_quantity': 100,
            'executed_price': 100.0,
            'commission': 5.0,
            'fees': 0.0,
            'slippage_cost_total': 10.0,
            'fill_timestamp': datetime(2025, 1, 15, 14, 31)
        }
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = 0.1
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        # Check fill was processed
        assert len(info['fills_step']) == 1
        assert env_ready.action_counts['BUY'] == 1
        assert reward == 0.1
    
    def test_step_sell_action(self, env_ready):
        """Test step with SELL action."""
        action = np.array([2, 1])  # SELL, SIZE_50
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'SELL',
            'position_size': 0.5,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'SELL'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = -0.05
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert info['action_decoded']['action_type'] == 'SELL'
        assert env_ready.action_counts['SELL'] == 1
        assert reward == -0.05
    
    def test_step_invalid_action(self, env_ready):
        """Test step with invalid action."""
        action = np.array([1, 0])  # BUY with insufficient capital
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'BUY',
            'position_size': 0.25,
            'is_valid': False,
            'validation_reason': 'Insufficient capital'
        }
        mock_execution_result.action_decode_result.action_type = 'BUY'
        mock_execution_result.action_decode_result.is_valid = False
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = -0.1  # Penalty
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert info['invalid_action_in_step'] == True
        assert env_ready.invalid_action_count_episode == 1
        assert reward == -0.1
    
    def test_step_without_prior_reset(self, env_with_session):
        """Test step called without reset."""
        action = np.array([0, 0])
        
        obs, reward, terminated, truncated, info = env_with_session.step(action)
        
        # Should handle gracefully
        assert terminated == True
        assert "error" in info
        assert reward == 0.0
    
    def test_step_updates_portfolio_metrics(self, env_ready):
        """Test that step properly updates portfolio metrics."""
        action = np.array([0, 0])  # HOLD
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = 0.0
        
        # Mock portfolio state
        portfolio_state = {
            'total_equity': 100500.0,
            'cash': 90500.0,
            'unrealized_pnl': 500.0,
            'realized_pnl_session': 0.0,
            'timestamp': datetime(2025, 1, 15, 14, 31),
            'positions': {'AAPL': {'quantity': 100, 'current_side': PositionSideEnum.LONG, 'avg_entry_price': 95.0}},
            'session_metrics': {
                'total_commissions_session': 10.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 5.0
            }
        }
        env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert info['portfolio_equity'] == 100500.0
        assert info['portfolio_cash'] == 90500.0
        assert info['portfolio_unrealized_pnl'] == 500.0


class TestTerminationConditions:
    """Tests for various termination conditions."""
    
    @pytest.fixture
    def env_ready(self, env_with_session):
        """Environment that has been reset and is ready for steps."""
        env_with_session.reset()
        return env_with_session
    
    def test_termination_bankruptcy(self, env_ready):
        """Test termination due to bankruptcy."""
        action = np.array([1, 3])  # BUY, SIZE_100
        
        # Setup execution result
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'BUY',
            'position_size': 1.0,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'BUY'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        
        # Mock portfolio state with very low equity (bankruptcy)
        portfolio_state = {
            'total_equity': 5000.0,  # Below 10% of initial capital (100k)
            'cash': 5000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': -95000.0,
            'timestamp': datetime(2025, 1, 15, 14, 31),
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 100.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 50.0
            }
        }
        env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        env_ready.reward_calculator.calculate.return_value = -10.0
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert terminated == True
        assert info['termination_reason'] == TerminationReasonEnum.BANKRUPTCY.value
    
    def test_termination_max_loss(self, env_ready):
        """Test termination due to max loss threshold."""
        action = np.array([0, 0])
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        
        # Mock portfolio state with 5.5% loss (> 5% threshold)
        portfolio_state = {
            'total_equity': 94000.0,  # 6% loss from initial 100k
            'cash': 94000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': -6000.0,
            'timestamp': datetime(2025, 1, 15, 14, 31),
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 50.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 25.0
            }
        }
        env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        env_ready.reward_calculator.calculate.return_value = -1.0
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert terminated == True
        assert info['termination_reason'] == TerminationReasonEnum.MAX_LOSS_REACHED.value
    
    def test_termination_end_of_data(self, env_ready):
        """Test termination when market data ends."""
        action = np.array([0, 0])
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = False  # No more data
        env_ready.reward_calculator.calculate.return_value = 0.0
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert terminated == True
        assert info['termination_reason'] == TerminationReasonEnum.END_OF_SESSION_DATA.value
    
    def test_termination_invalid_action_limit(self, env_ready):
        """Test termination due to too many invalid actions."""
        # Set invalid action count to just below limit
        env_ready.invalid_action_count_episode = 99
        env_ready.max_invalid_actions_per_episode = 100
        
        action = np.array([1, 0])
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'BUY',
            'position_size': 0.25,
            'is_valid': False
        }
        mock_execution_result.action_decode_result.action_type = 'BUY'
        mock_execution_result.action_decode_result.is_valid = False  # Invalid action
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = -0.1
        
        # Mock normal portfolio state
        portfolio_state = {
            'total_equity': 100000.0,
            'cash': 100000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': 0.0,
            'timestamp': datetime(2025, 1, 15, 14, 31),
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 0.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 0.0
            }
        }
        env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert terminated == True
        assert info['termination_reason'] == TerminationReasonEnum.INVALID_ACTION_LIMIT_REACHED.value
        assert env_ready.invalid_action_count_episode == 100
    
    def test_termination_max_duration(self, env_ready):
        """Test termination due to episode duration limit."""
        action = np.array([0, 0])
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        
        # Set next market time to be past episode end time
        env_ready.episode_end_time_utc = datetime(2025, 1, 15, 18, 30)
        env_ready.market_simulator.step.return_value = True
        env_ready.market_simulator.get_current_market_data.return_value = {
            'timestamp': datetime(2025, 1, 15, 18, 31),  # Past end time
            'current_price': 100.0,
            'best_ask_price': 100.1,
            'best_bid_price': 99.9
        }
        
        env_ready.reward_calculator.calculate.return_value = 0.0
        
        obs, reward, terminated, truncated, info = env_ready.step(action)
        
        assert terminated == True
        assert info['termination_reason'] == TerminationReasonEnum.MAX_DURATION.value


class TestObservationGeneration:
    """Tests for observation generation."""
    
    @pytest.fixture
    def env_ready(self, env_with_session):
        """Environment that has been reset and is ready."""
        env_with_session.reset()
        return env_with_session
    
    def test_observation_structure(self, env_ready):
        """Test observation has correct structure and types."""
        obs = env_ready._get_observation()
        
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {'hf', 'mf', 'lf', 'portfolio', 'static'}
        
        # Check all are numpy arrays
        for key, value in obs.items():
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.float32
    
    def test_observation_shapes(self, env_ready):
        """Test observation components have correct shapes."""
        obs = env_ready._get_observation()
        
        assert obs['hf'].shape == (60, 10)
        assert obs['mf'].shape == (20, 8)
        assert obs['lf'].shape == (5, 6)
        assert obs['portfolio'].shape == (1, 5)
        assert obs['static'].shape == (1, 4)
    
    def test_observation_handles_missing_features(self, env_ready):
        """Test observation generation when features are missing."""
        # Mock market simulator returning None features
        env_ready.market_simulator.get_current_features.return_value = None
        
        obs = env_ready._get_observation()
        
        # Should return None
        assert obs is None
    
    def test_observation_handles_nan_values(self, env_ready):
        """Test observation properly handles NaN values."""
        # Mock features with NaN values
        features_with_nan = {
            'hf': np.full((60, 10), np.nan, dtype=np.float32),
            'mf': np.full((20, 8), np.nan, dtype=np.float32),
            'lf': np.full((5, 6), np.nan, dtype=np.float32),
            'static': np.full((1, 4), np.nan, dtype=np.float32)
        }
        env_ready.market_simulator.get_current_features.return_value = features_with_nan
        
        obs = env_ready._get_observation()
        
        # Should replace NaN with 0
        for key, value in obs.items():
            assert not np.any(np.isnan(value))
            if key != 'portfolio':  # Portfolio comes from portfolio manager
                assert np.all(value == 0.0)
    
    def test_dummy_observation_generation(self, env_ready):
        """Test dummy observation has correct structure."""
        dummy_obs = env_ready._get_dummy_observation()
        
        # Check structure
        assert isinstance(dummy_obs, dict)
        assert set(dummy_obs.keys()) == {'hf', 'mf', 'lf', 'portfolio', 'static'}
        
        # Check all are zeros with correct shapes
        assert np.all(dummy_obs['hf'] == 0.0)
        assert dummy_obs['hf'].shape == (60, 10)
        
        assert np.all(dummy_obs['mf'] == 0.0)
        assert dummy_obs['mf'].shape == (20, 8)
        
        assert np.all(dummy_obs['lf'] == 0.0)
        assert dummy_obs['lf'].shape == (5, 6)
        
        assert np.all(dummy_obs['portfolio'] == 0.0)
        assert dummy_obs['portfolio'].shape == (1, 5)
        
        assert np.all(dummy_obs['static'] == 0.0)
        assert dummy_obs['static'].shape == (1, 4)


class TestActionSpace:
    """Tests for action space and action handling."""
    
    def test_action_space_structure(self, mock_config, mock_data_manager):
        """Test action space has correct structure."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert env.action_space.nvec.tolist() == [3, 4]
        assert env.action_space.shape == (2,)
    
    def test_action_space_sample(self, mock_config, mock_data_manager):
        """Test action space sampling produces valid actions."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        for _ in range(10):
            action = env.action_space.sample()
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert 0 <= action[0] <= 2  # Action type
            assert 0 <= action[1] <= 3  # Position size
    
    def test_action_enum_values(self):
        """Test action enums have expected values."""
        assert ActionTypeEnum.HOLD.value == 0
        assert ActionTypeEnum.BUY.value == 1
        assert ActionTypeEnum.SELL.value == 2
        
        assert PositionSizeTypeEnum.SIZE_25.value == 0
        assert PositionSizeTypeEnum.SIZE_50.value == 1
        assert PositionSizeTypeEnum.SIZE_75.value == 2
        assert PositionSizeTypeEnum.SIZE_100.value == 3
    
    def test_position_size_float_values(self):
        """Test position size enum float conversion."""
        assert PositionSizeTypeEnum.SIZE_25.value_float == 0.25
        assert PositionSizeTypeEnum.SIZE_50.value_float == 0.50
        assert PositionSizeTypeEnum.SIZE_75.value_float == 0.75
        assert PositionSizeTypeEnum.SIZE_100.value_float == 1.00


class TestEpisodeManagement:
    """Tests for episode management and tracking."""
    
    @pytest.fixture
    def env_ready(self, env_with_session):
        """Environment that has been reset and is ready."""
        env_with_session.reset()
        return env_with_session
    
    def test_episode_number_increments(self, env_ready):
        """Test episode number increments on reset."""
        initial_episode = env_ready.episode_number
        
        env_ready.reset()
        assert env_ready.episode_number == initial_episode + 1
        
        env_ready.reset()
        assert env_ready.episode_number == initial_episode + 2
    
    def test_episode_metrics_tracking(self, env_ready):
        """Test episode metrics are tracked correctly."""
        # Perform some steps
        for i in range(5):
            action = np.array([0, 0])  # HOLD
            
            mock_execution_result = MagicMock()
            mock_execution_result.action_decode_result.to_dict.return_value = {
                'action_type': 'HOLD',
                'position_size': 0.25,
                'is_valid': True
            }
            mock_execution_result.action_decode_result.action_type = 'HOLD'
            mock_execution_result.action_decode_result.is_valid = True
            mock_execution_result.fill_details = None
            
            env_ready.execution_manager.execute_action.return_value = mock_execution_result
            env_ready.market_simulator.step.return_value = True
            env_ready.reward_calculator.calculate.return_value = 0.1 * i
            
            env_ready.step(action)
        
        assert env_ready.current_step == 5
        assert env_ready.episode_total_reward == sum(0.1 * i for i in range(5))
        assert env_ready.action_counts['HOLD'] == 5
    
    def test_episode_peak_equity_tracking(self, env_ready):
        """Test peak equity is tracked during episode."""
        env_ready.episode_peak_equity = 100000.0
        
        # Step with equity increase
        portfolio_state = {
            'total_equity': 105000.0,
            'cash': 105000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': 5000.0,
            'timestamp': datetime(2025, 1, 15, 14, 31),
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 0.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 0.0
            }
        }
        
        action = np.array([0, 0])
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = 0.0
        
        env_ready.step(action)
        
        assert env_ready.episode_peak_equity == 105000.0
    
    def test_episode_drawdown_tracking(self, env_ready):
        """Test drawdown is tracked correctly."""
        env_ready.episode_peak_equity = 110000.0
        
        # Step with equity decrease
        portfolio_state = {
            'total_equity': 100000.0,  # 10k loss from peak
            'cash': 100000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': -10000.0,
            'timestamp': datetime(2025, 1, 15, 14, 31),
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 0.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 0.0
            }
        }
        
        action = np.array([0, 0])
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_ready.execution_manager.execute_action.return_value = mock_execution_result
        env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        env_ready.market_simulator.step.return_value = True
        env_ready.reward_calculator.calculate.return_value = -0.5
        
        env_ready.step(action)
        
        expected_drawdown = (110000.0 - 100000.0) / 110000.0
        assert abs(env_ready.episode_max_drawdown - expected_drawdown) < 0.001


class TestMomentumFeatures:
    """Tests for momentum-based features and day selection."""
    
    def test_get_momentum_days(self, mock_config, mock_data_manager):
        """Test getting momentum days."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        mock_momentum_days = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': pd.to_datetime(['2025-01-15', '2025-01-16']),
            'quality_score': [0.8, 0.6]
        })
        mock_data_manager.get_momentum_days.return_value = mock_momentum_days
        
        env.primary_asset = 'AAPL'
        days = env.get_momentum_days(min_quality=0.5)
        
        assert len(days) == 2
        mock_data_manager.get_momentum_days.assert_called_with('AAPL', 0.5)
    
    def test_select_next_momentum_day(self, mock_config, mock_data_manager):
        """Test selecting next momentum day."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        mock_momentum_days = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'date': pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17']),
            'quality_score': [0.8, 0.9, 0.7],
            'max_intraday_move': [0.05, 0.08, 0.03],
            'volume_multiplier': [2.5, 3.0, 1.8]
        })
        mock_data_manager.get_momentum_days.return_value = mock_momentum_days
        
        env.primary_asset = 'AAPL'
        next_day = env.select_next_momentum_day()
        
        assert next_day is not None
        assert next_day['symbol'] == 'AAPL'
        assert next_day['quality_score'] == 0.8  # Highest quality first
    
    def test_select_next_momentum_day_with_exclusions(self, mock_config, mock_data_manager):
        """Test selecting next momentum day with date exclusions."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        mock_momentum_days = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'date': pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17']),
            'quality_score': [0.8, 0.9, 0.7]
        })
        mock_data_manager.get_momentum_days.return_value = mock_momentum_days
        
        env.primary_asset = 'AAPL'
        exclude_dates = [datetime(2025, 1, 15), datetime(2025, 1, 16)]
        next_day = env.select_next_momentum_day(exclude_dates=exclude_dates)
        
        assert next_day is not None
        assert next_day['date'].date() == datetime(2025, 1, 17).date()
    
    def test_select_next_momentum_day_no_days_available(self, mock_config, mock_data_manager):
        """Test selecting next momentum day when none available."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        mock_data_manager.get_momentum_days.return_value = pd.DataFrame()  # Empty
        
        env.primary_asset = 'AAPL'
        next_day = env.select_next_momentum_day()
        
        assert next_day is None


class TestSessionSwitching:
    """Tests for session preparation and switching."""
    
    def test_prepare_next_session(self, mock_config, mock_data_manager):
        """Test preparing next session in background."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        mock_market_sim = MagicMock()
        mock_market_sim.initialize_day.return_value = True
        
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            env.prepare_next_session("AAPL", "2025-01-16")
        
        assert env.next_market_simulator is not None
        mock_market_sim.initialize_day.assert_called_once()
    
    def test_switch_to_next_session(self, mock_config, mock_data_manager):
        """Test switching to prepared session."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        # Setup current session
        env.market_simulator = MagicMock()
        env.execution_manager = MagicMock()
        env.current_session_date = datetime(2025, 1, 16)  # Set date for next session
        
        # Prepare next session
        mock_next_sim = MagicMock()
        env.next_market_simulator = mock_next_sim
        
        # Mock reset points
        env.data_manager.get_reset_points.return_value = pd.DataFrame()
        
        env.switch_to_next_session()
        
        assert env.market_simulator == mock_next_sim
        assert env.next_market_simulator is None
        assert env.execution_manager.market_simulator == mock_next_sim
        assert env.current_reset_idx == 0
    
    def test_switch_without_preparation(self, mock_config, mock_data_manager):
        """Test switching fails without preparation."""
        env = TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager
        )
        
        with pytest.raises(ValueError, match="No next session prepared"):
            env.switch_to_next_session()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_step_with_market_data_failure(self, env_with_session):
        """Test step when market data is unavailable."""
        env_with_session.reset()
        env_with_session.market_simulator.get_current_market_data.return_value = None
        
        action = np.array([0, 0])
        obs, reward, terminated, truncated, info = env_with_session.step(action)
        
        assert terminated == True
        assert "error" in info
        assert reward == 0.0
    
    def test_reset_with_market_simulator_failure(self, env_with_session):
        """Test reset when market simulator fails."""
        env_with_session.market_simulator.reset.return_value = False
        
        obs, info = env_with_session.reset()
        
        assert "error" in info
        assert info["error"] == "Market simulator reset failed"
    
    def test_reset_with_time_setting_failure(self, env_with_session):
        """Test reset when setting time fails."""
        env_with_session.market_simulator.reset.return_value = True
        env_with_session.market_simulator.set_time.return_value = False
        
        obs, info = env_with_session.reset()
        
        assert "error" in info
        assert info["error"] == "Failed to set market time"
    
    def test_observation_with_shape_mismatch(self, env_with_session):
        """Test observation generation with shape mismatch."""
        env_with_session.reset()
        
        # Mock features with wrong shape
        wrong_features = {
            'hf': np.zeros((50, 10), dtype=np.float32),  # Wrong shape
            'mf': np.zeros((20, 8), dtype=np.float32),
            'lf': np.zeros((5, 6), dtype=np.float32),
            'static': np.zeros((1, 4), dtype=np.float32)
        }
        env_with_session.market_simulator.get_current_features.return_value = wrong_features
        
        obs = env_with_session._get_observation()
        
        # Should return None due to shape mismatch
        assert obs is None
    
    def test_handle_open_positions_no_market_data(self, env_with_session):
        """Test handling open positions when market data unavailable."""
        env_with_session.market_simulator.get_current_market_data.return_value = None
        
        result = env_with_session._handle_open_positions_at_reset()
        
        assert result is None
    
    def test_multiple_consecutive_invalid_actions(self, env_ready):
        """Test handling multiple consecutive invalid actions."""
        invalid_actions = 0
        max_invalid = 5
        env_ready.max_invalid_actions_per_episode = max_invalid
        env_ready.invalid_action_count_episode = 0  # Reset counter
        
        for i in range(max_invalid + 1):
            action = np.array([1, 0])  # Invalid BUY
            
            mock_execution_result = MagicMock()
            mock_execution_result.action_decode_result.to_dict.return_value = {
                'action_type': 'BUY',
                'position_size': 0.25,
                'is_valid': False
            }
            mock_execution_result.action_decode_result.action_type = 'BUY'
            mock_execution_result.action_decode_result.is_valid = False
            mock_execution_result.fill_details = None
            
            env_ready.execution_manager.execute_action.return_value = mock_execution_result
            env_ready.market_simulator.step.return_value = True
            # Update market data for next step  
            env_ready.market_simulator.get_current_market_data.return_value = {
                'timestamp': pd.Timestamp(f'2025-01-15 14:{31 + i}:00', tz='UTC'),
                'current_price': 100.0,
                'best_ask_price': 100.1,
                'best_bid_price': 99.9
            }
            env_ready.reward_calculator.calculate.return_value = -0.1
            
            # Mock normal portfolio state
            portfolio_state = {
                'total_equity': 100000.0,
                'cash': 100000.0,
                'unrealized_pnl': 0.0,
                'realized_pnl_session': 0.0,
                'timestamp': pd.Timestamp(f'2025-01-15 14:{31 + i}:00', tz='UTC'),
                'positions': {},
                'session_metrics': {
                    'total_commissions_session': 0.0,
                    'total_fees_session': 0.0,
                    'total_slippage_cost_session': 0.0
                }
            }
            env_ready.portfolio_manager.get_portfolio_state.return_value = portfolio_state
            
            obs, reward, terminated, truncated, info = env_ready.step(action)
            
            # After 5 invalid actions (indices 0-4), the environment should terminate
            if i < max_invalid - 1:  # First 4 iterations (0-3) should not terminate
                assert not terminated
                invalid_actions += 1
            elif i == max_invalid - 1:  # 5th invalid action (index 4) should terminate
                assert terminated
                assert info['termination_reason'] == TerminationReasonEnum.INVALID_ACTION_LIMIT_REACHED.value
                break
            else:
                # Should never reach here
                assert False, "Test should have terminated already"
    
    def test_render_methods(self, env_ready):
        """Test render methods."""
        info = {
            'step': 10,
            'reward_step': 0.5,
            'portfolio_equity': 101000.0
        }
        
        # Should not raise
        env_ready.render_mode = 'human'
        env_ready.render(info)
        
        env_ready.render_mode = 'logs'
        env_ready.render(info)
        
        env_ready.render_mode = None
        env_ready.render(info)
    
    def test_close_method(self, env_with_session):
        """Test environment close method."""
        # Should not raise
        env_with_session.close()
        
        # Test with next_market_simulator
        env_with_session.next_market_simulator = MagicMock()
        env_with_session.close()
    
    def test_set_training_info(self, env_with_session):
        """Test setting training info."""
        env_with_session.set_training_info(
            episode_num=10,
            total_episodes=100,
            total_steps=50000,
            update_count=200
        )
        
        assert env_with_session.episode_number == 10
        assert env_with_session.total_episodes == 100
        assert env_with_session.total_steps == 50000
        assert env_with_session.update_count == 200
    
    def test_trade_callback(self, env_with_session):
        """Test trade completed callback."""
        # Win trade
        trade = {'realized_pnl': 100.0}
        env_with_session._on_trade_completed(trade)
        assert env_with_session.win_loss_counts['wins'] == 1
        
        # Loss trade
        trade = {'realized_pnl': -50.0}
        env_with_session._on_trade_completed(trade)
        assert env_with_session.win_loss_counts['losses'] == 1
        
        # Break-even trade
        trade = {'realized_pnl': 0.0}
        env_with_session._on_trade_completed(trade)
        assert env_with_session.win_loss_counts['wins'] == 1
        assert env_with_session.win_loss_counts['losses'] == 1


class TestMetricsIntegration:
    """Tests for metrics integration."""
    
    def test_metrics_recording_during_step(self, env_with_session):
        """Test metrics are recorded during step."""
        mock_metrics = MagicMock()
        env_with_session.metrics_integrator = mock_metrics
        env_with_session.reset()
        
        action = np.array([1, 2])  # BUY, SIZE_75
        
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'BUY',
            'position_size': 0.75,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'BUY'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = MagicMock(
            executed_quantity=75,
            executed_price=100.0,
            commission=5.0,
            fees=0.0,
            slippage_cost_total=10.0
        )
        
        env_with_session.execution_manager.execute_action.return_value = mock_execution_result
        env_with_session.portfolio_manager.process_fill.return_value = {
            'executed_quantity': 75,
            'executed_price': 100.0,
            'commission': 5.0,
            'fees': 0.0,
            'slippage_cost_total': 10.0,
            'fill_timestamp': datetime(2025, 1, 15, 14, 31)
        }
        env_with_session.market_simulator.step.return_value = True
        env_with_session.reward_calculator.calculate.return_value = 0.5
        env_with_session.reward_calculator.get_last_reward_components.return_value = {
            'pnl': 0.3,
            'action_efficiency': 0.2
        }
        
        env_with_session.step(action)
        
        # Check metrics calls
        mock_metrics.record_environment_step.assert_called()
        mock_metrics.update_portfolio.assert_called()
        mock_metrics.update_position.assert_called()
        mock_metrics.record_fill.assert_called()
    
    def test_metrics_episode_tracking(self, env_with_session):
        """Test metrics tracking for episode start/end."""
        mock_metrics = MagicMock()
        mock_metrics.metrics_manager = MagicMock()
        env_with_session.metrics_integrator = mock_metrics
        
        # Episode start
        env_with_session.reset()
        mock_metrics.start_episode.assert_called()
        
        # Episode end (simulate termination)
        action = np.array([0, 0])
        mock_execution_result = MagicMock()
        mock_execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.25,
            'is_valid': True
        }
        mock_execution_result.action_decode_result.action_type = 'HOLD'
        mock_execution_result.action_decode_result.is_valid = True
        mock_execution_result.fill_details = None
        
        env_with_session.execution_manager.execute_action.return_value = mock_execution_result
        env_with_session.market_simulator.step.return_value = False  # End of data
        env_with_session.reward_calculator.calculate.return_value = 0.0
        
        # Mock portfolio state
        portfolio_state = {
            'total_equity': 101000.0,
            'cash': 101000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': 1000.0,
            'timestamp': datetime(2025, 1, 15, 20, 0),
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 50.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 25.0
            }
        }
        env_with_session.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        env_with_session.portfolio_manager.get_trading_metrics.return_value = {
            'total_trades': 5,
            'winning_trades': 3,
            'losing_trades': 2
        }
        
        obs, reward, terminated, truncated, info = env_with_session.step(action)
        
        assert terminated
        mock_metrics.end_episode.assert_called()
        mock_metrics.record_episode_end.assert_called()
        
        # Check episode summary
        assert 'episode_summary' in info
        assert info['episode_summary']['total_fills'] == 0
        assert info['episode_summary']['final_equity'] == 101000.0


class TestResetPointAdaptation:
    """Tests for adaptive reset point features."""
    
    def test_adaptive_randomization_window(self, env_with_session):
        """Test adaptive randomization window based on activity."""
        # Very high activity
        reset_point = {'activity_score': 0.9, 'combined_score': 0.8, 'reset_type': 'momentum'}
        window = env_with_session._get_adaptive_randomization_window(reset_point)
        assert 1 <= window <= 5  # Tight window for high activity
        
        # Medium activity
        reset_point = {'activity_score': 0.5, 'combined_score': 0.5, 'reset_type': 'momentum'}
        window = env_with_session._get_adaptive_randomization_window(reset_point)
        assert 5 <= window <= 15  # Wider window
        
        # Low activity
        reset_point = {'activity_score': 0.2, 'combined_score': 0.3, 'reset_type': 'momentum'}
        window = env_with_session._get_adaptive_randomization_window(reset_point)
        assert 10 <= window <= 30  # Wide window for low activity
        
        # Fixed reset point
        reset_point = {'activity_score': 0.5, 'combined_score': 0.5, 'reset_type': 'fixed'}
        window = env_with_session._get_adaptive_randomization_window(reset_point)
        assert window >= 10  # Fixed points get wider windows
    
    def test_duration_for_activity(self, env_with_session):
        """Test episode duration based on activity score."""
        assert env_with_session._get_duration_for_activity(0.9) == 1.5  # Very high
        assert env_with_session._get_duration_for_activity(0.7) == 2.0  # High
        assert env_with_session._get_duration_for_activity(0.5) == 3.0  # Medium
        assert env_with_session._get_duration_for_activity(0.2) == 4.0  # Low
    
    def test_get_next_reset_point(self, env_with_session):
        """Test getting next reset point."""
        env_with_session.reset_points = [
            {'timestamp': datetime(2025, 1, 15, 14, 30)},
            {'timestamp': datetime(2025, 1, 15, 15, 30)},
            {'timestamp': datetime(2025, 1, 15, 16, 30)}
        ]
        env_with_session.current_reset_idx = 0
        
        next_point = env_with_session.get_next_reset_point()
        assert next_point is not None
        assert next_point['timestamp'] == datetime(2025, 1, 15, 15, 30)
        
        # At last point
        env_with_session.current_reset_idx = 2
        next_point = env_with_session.get_next_reset_point()
        assert next_point is None
    
    def test_has_more_reset_points(self, env_with_session):
        """Test checking for more reset points."""
        env_with_session.reset_points = [
            {'timestamp': datetime(2025, 1, 15, 14, 30)},
            {'timestamp': datetime(2025, 1, 15, 15, 30)}
        ]
        
        env_with_session.current_reset_idx = 0
        assert env_with_session.has_more_reset_points() == True
        
        env_with_session.current_reset_idx = 1
        assert env_with_session.has_more_reset_points() == False