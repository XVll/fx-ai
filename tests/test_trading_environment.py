"""
Comprehensive tests for TradingEnvironment class.

Tests focus on input/output behavior rather than implementation details,
covering all major functionality and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock, patch
import logging

from envs.trading_environment import (
    TradingEnvironment, 
    ActionTypeEnum, 
    PositionSizeTypeEnum, 
    TerminationReasonEnum
)
from config.schemas import Config, ModelConfig, EnvironmentConfig, SimulationConfig, RewardConfig
from data.data_manager import DataManager
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum
from simulators.market_simulator import MarketSimulator
from simulators.execution_simulator import ExecutionSimulator
from rewards.calculator import RewardSystem


class TestTradingEnvironment:
    """Test suite for TradingEnvironment."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=Config)
        
        # Model config
        config.model = Mock(spec=ModelConfig)
        config.model.hf_seq_len = 60
        config.model.hf_feat_dim = 7
        config.model.mf_seq_len = 30
        config.model.mf_feat_dim = 43
        config.model.lf_seq_len = 30
        config.model.lf_feat_dim = 19
        config.model.portfolio_seq_len = 5
        config.model.portfolio_feat_dim = 5
        
        # Environment config
        config.env = Mock(spec=EnvironmentConfig)
        config.env.max_episode_steps = 1000
        config.env.early_stop_loss_threshold = 0.95
        config.env.reward = Mock(spec=RewardConfig)
        
        # Simulation config
        config.simulation = Mock(spec=SimulationConfig)
        config.simulation.initial_capital = 100000.0
        config.simulation.min_commission_per_order = 1.0
        config.simulation.commission_per_share = 0.005
        
        return config
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        data_manager = Mock(spec=DataManager)
        
        # Mock momentum days
        momentum_days = pd.DataFrame({
            'symbol': ['MLGO', 'MLGO'],
            'date': [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            'activity_score': [0.8, 0.9],
            'max_intraday_move': [0.15, 0.20],
            'volume_multiplier': [2.5, 3.0]
        })
        data_manager.get_momentum_days.return_value = momentum_days
        
        # Mock reset points
        reset_points = pd.DataFrame({
            'timestamp': [
                datetime(2025, 1, 1, 9, 30),
                datetime(2025, 1, 1, 14, 0)
            ],
            'activity_score': [0.8, 0.6],
            'combined_score': [0.9, 0.7]
        })
        data_manager.get_reset_points.return_value = reset_points
        
        return data_manager
    
    @pytest.fixture
    def mock_market_simulator(self):
        """Create a mock market simulator."""
        market_sim = Mock(spec=MarketSimulator)
        market_sim.initialize_day.return_value = True
        market_sim.get_stats.return_value = {
            'total_seconds': 57600,  # 16 hours
            'warmup_info': {'has_warmup': True}
        }
        market_sim.reset.return_value = True
        market_sim.set_time.return_value = True
        market_sim.step.return_value = True
        
        # Mock market state
        mock_market_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'current_price': 10.50,
            'best_bid_price': 10.48,
            'best_ask_price': 10.52,
            'volume': 1000,
            'session_type': 'REGULAR'
        }
        market_sim.get_current_market_data.return_value = mock_market_state
        market_sim.get_market_state.return_value = Mock(timestamp=datetime(2025, 1, 1, 9, 30))
        
        # Mock features
        mock_features = {
            'hf': np.random.random((60, 7)),
            'mf': np.random.random((30, 43)),
            'lf': np.random.random((30, 19))
        }
        market_sim.get_current_features.return_value = mock_features
        
        return market_sim
    
    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        portfolio = Mock(spec=PortfolioSimulator)
        portfolio.initial_capital = 100000.0
        
        # Mock portfolio state
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 100000.0,
            'cash': 90000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': 0.0,
            'positions': {
                'MLGO': {
                    'quantity': 0.0,
                    'current_side': PositionSideEnum.FLAT,
                    'avg_entry_price': 0.0,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0
                }
            },
            'session_metrics': {
                'total_commissions_session': 0.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 0.0
            }
        }
        portfolio.get_portfolio_state.return_value = portfolio_state
        
        # Mock portfolio observation
        portfolio_obs = {
            'features': np.random.random((5, 5))
        }
        portfolio.get_portfolio_observation.return_value = portfolio_obs
        
        # Mock trading metrics
        portfolio.get_trading_metrics.return_value = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        return portfolio
    
    @pytest.fixture
    def mock_execution_manager(self):
        """Create a mock execution manager."""
        execution = Mock(spec=ExecutionSimulator)
        
        # Mock execution result
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'position_size': 0.0,
            'is_valid': True
        }
        execution.execute_action.return_value = execution_result
        
        return execution
    
    @pytest.fixture
    def mock_reward_calculator(self):
        """Create a mock reward calculator."""
        reward_calc = Mock(spec=RewardSystem)
        reward_calc.calculate.return_value = 0.1
        reward_calc.get_last_reward_components.return_value = {}
        return reward_calc
    
    @pytest.fixture
    def trading_env(self, mock_config, mock_data_manager):
        """Create a trading environment for testing."""
        with patch('envs.trading_environment.MarketSimulator'), \
             patch('envs.trading_environment.PortfolioSimulator'), \
             patch('envs.trading_environment.ExecutionSimulator'), \
             patch('envs.trading_environment.RewardSystem'):
            
            env = TradingEnvironment(
                config=mock_config,
                data_manager=mock_data_manager,
                logger=logging.getLogger('test')
            )
            return env
    
    def test_initialization(self, trading_env, mock_config):
        """Test environment initialization."""
        assert trading_env.config == mock_config
        assert trading_env.primary_asset is None
        assert trading_env.current_step == 0
        assert trading_env.episode_number == 0
        assert trading_env.invalid_action_count_episode == 0
        assert trading_env.episode_total_reward == 0.0
        assert trading_env.is_terminated == False
        assert trading_env.is_truncated == False
        
        # Test action space
        assert trading_env.action_space.nvec.tolist() == [3, 4]
        
        # Test observation space
        assert 'hf' in trading_env.observation_space.spaces
        assert 'mf' in trading_env.observation_space.spaces
        assert 'lf' in trading_env.observation_space.spaces
        assert 'portfolio' in trading_env.observation_space.spaces
    
    def test_setup_session_success(self, trading_env, mock_market_simulator):
        """Test successful session setup."""
        with patch.object(trading_env, '_initialize_simulators'), \
             patch.object(trading_env, '_generate_reset_points') as mock_reset_points:
            
            mock_reset_points.return_value = [{'timestamp': datetime(2025, 1, 1, 9, 30)}]
            
            # Mock MarketSimulator creation
            with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_simulator):
                trading_env.setup_session('MLGO', '2025-01-01')
            
            assert trading_env.primary_asset == 'MLGO'
            assert trading_env.current_session_date.date() == datetime(2025, 1, 1).date()
            assert len(trading_env.reset_points) == 1
            assert trading_env.current_reset_idx == 0
    
    def test_setup_session_invalid_symbol(self, trading_env):
        """Test setup session with invalid symbol."""
        with pytest.raises(ValueError, match="A valid symbol.*must be provided"):
            trading_env.setup_session('', '2025-01-01')
        
        with pytest.raises(ValueError, match="A valid symbol.*must be provided"):
            trading_env.setup_session(None, '2025-01-01')
    
    def test_setup_session_failed_initialization(self, trading_env):
        """Test setup session with failed market simulator initialization."""
        mock_market_sim = Mock()
        mock_market_sim.initialize_day.return_value = False
        
        with patch('envs.trading_environment.MarketSimulator', return_value=mock_market_sim):
            with pytest.raises(ValueError, match="Failed to initialize.*"):
                trading_env.setup_session('MLGO', '2025-01-01')
    
    def test_enum_values(self):
        """Test enum value consistency."""
        # ActionTypeEnum
        assert ActionTypeEnum.HOLD.value == 0
        assert ActionTypeEnum.BUY.value == 1
        assert ActionTypeEnum.SELL.value == 2
        
        # PositionSizeTypeEnum
        assert PositionSizeTypeEnum.SIZE_25.value == 0
        assert PositionSizeTypeEnum.SIZE_50.value == 1
        assert PositionSizeTypeEnum.SIZE_75.value == 2
        assert PositionSizeTypeEnum.SIZE_100.value == 3
        
        # Test value_float property
        assert PositionSizeTypeEnum.SIZE_25.value_float == 0.25
        assert PositionSizeTypeEnum.SIZE_50.value_float == 0.50
        assert PositionSizeTypeEnum.SIZE_75.value_float == 0.75
        assert PositionSizeTypeEnum.SIZE_100.value_float == 1.0
    
    def test_generate_reset_points_momentum(self, trading_env, mock_data_manager):
        """Test reset point generation with momentum data."""
        # Setup session first
        trading_env.primary_asset = 'MLGO'
        trading_env.current_session_date = datetime(2025, 1, 1)
        
        reset_points = trading_env._generate_reset_points()
        
        assert len(reset_points) == 2  # From mock data
        assert reset_points[0]['reset_type'] == 'momentum'
        assert 'activity_score' in reset_points[0]
        assert 'combined_score' in reset_points[0]
    
    def test_generate_reset_points_fixed_fallback(self, trading_env, mock_data_manager):
        """Test reset point generation fallback to fixed points."""
        # Mock empty momentum data
        mock_data_manager.get_reset_points.return_value = pd.DataFrame()
        
        trading_env.primary_asset = 'MLGO'
        trading_env.current_session_date = datetime(2025, 1, 1)
        
        reset_points = trading_env._generate_reset_points()
        
        assert len(reset_points) == 4  # Fixed times: 9:30, 10:30, 14:00, 15:30
        assert all(rp['reset_type'] == 'fixed' for rp in reset_points)
    
    def test_adaptive_randomization_window(self, trading_env):
        """Test adaptive randomization window calculation."""
        # High activity score - tight window
        reset_point = {'activity_score': 0.9, 'combined_score': 0.8, 'reset_type': 'momentum'}
        window = trading_env._get_adaptive_randomization_window(reset_point)
        assert window <= 5  # Should be tight for high activity
        
        # Low activity score - wide window
        reset_point = {'activity_score': 0.2, 'combined_score': 0.3, 'reset_type': 'momentum'}
        window = trading_env._get_adaptive_randomization_window(reset_point)
        assert window >= 10  # Should be wider for low activity
        
        # Fixed reset type - wider multiplier
        reset_point = {'activity_score': 0.5, 'combined_score': 0.5, 'reset_type': 'fixed'}
        window = trading_env._get_adaptive_randomization_window(reset_point)
        assert window >= 5  # Fixed points get wider window
    
    def test_duration_for_activity(self, trading_env):
        """Test episode duration calculation based on activity."""
        # High activity - shorter duration
        duration = trading_env._get_duration_for_activity(0.9)
        assert duration == 1.5
        
        # Medium activity
        duration = trading_env._get_duration_for_activity(0.5)
        assert duration == 3.0
        
        # Low activity - longer duration
        duration = trading_env._get_duration_for_activity(0.1)
        assert duration == 4.0
    
    def test_reset_at_point_success(self, trading_env):
        """Test successful reset at a specific point."""
        # Setup mocks
        with patch.object(trading_env, '_handle_open_positions_at_reset', return_value=None), \
             patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_dashboard_quality_metrics'):
            
            # Setup environment
            trading_env.primary_asset = 'MLGO'
            trading_env.current_session_date = datetime(2025, 1, 1)
            trading_env.reset_points = [{
                'timestamp': pd.Timestamp('2025-01-01 09:30:00', tz='UTC').to_pydatetime(),
                'activity_score': 0.8,
                'combined_score': 0.9,
                'max_duration_hours': 2,
                'reset_type': 'momentum'
            }]
            
            # Mock components
            trading_env.market_simulator = Mock()
            trading_env.market_simulator.reset.return_value = True
            trading_env.market_simulator.set_time.return_value = True
            trading_env.market_simulator.get_market_state.return_value = Mock(
                timestamp=pd.Timestamp('2025-01-01 09:30:00', tz='UTC').to_pydatetime()
            )
            
            trading_env.execution_manager = Mock()
            trading_env.portfolio_manager = Mock()
            trading_env.portfolio_manager.initial_capital = 100000.0
            trading_env.portfolio_manager.get_portfolio_state.return_value = {
                'timestamp': pd.Timestamp('2025-01-01 09:30:00', tz='UTC').to_pydatetime(),
                'total_equity': 100000.0,
                'cash': 90000.0,
                'unrealized_pnl': 0.0,
                'realized_pnl_session': 0.0,
                'positions': {'MLGO': {'quantity': 0.0}},
                'session_metrics': {
                    'total_commissions_session': 0.0,
                    'total_fees_session': 0.0,
                    'total_slippage_cost_session': 0.0
                }
            }
            trading_env.reward_calculator = Mock()
            trading_env.metrics_integrator = Mock()
            
            # Mock observation
            mock_obs.return_value = {
                'hf': np.zeros((60, 7)),
                'mf': np.zeros((30, 43)),
                'lf': np.zeros((30, 19)),
                'portfolio': np.zeros((5, 5))
            }
            
            # Mock np_random
            trading_env.np_random = Mock()
            trading_env.np_random.integers.return_value = 0
            
            observation, info = trading_env.reset_at_point(0)
            
            assert observation is not None
            assert 'hf' in observation
            assert trading_env.current_step == 0
            assert trading_env.episode_number == 1
            assert trading_env.invalid_action_count_episode == 0
    
    def test_reset_at_point_invalid_index(self, trading_env):
        """Test reset at invalid point index."""
        trading_env.reset_points = []
        
        observation, info = trading_env.reset_at_point(0)
        
        assert 'error' in info
        assert info['error'] == 'Invalid reset point'
    
    def test_reset_at_point_market_simulator_failure(self, trading_env):
        """Test reset when market simulator fails."""
        # Setup
        trading_env.current_session_date = datetime(2025, 1, 1)
        trading_env.reset_points = [{
            'timestamp': pd.Timestamp('2025-01-01 09:30:00', tz='UTC').to_pydatetime(),
            'activity_score': 0.8,
            'max_duration_hours': 2
        }]
        
        # Mock failed market simulator
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.reset.return_value = False
        
        # Mock np_random
        trading_env.np_random = Mock()
        trading_env.np_random.integers.return_value = 0
        
        with patch.object(trading_env, '_handle_open_positions_at_reset', return_value=None):
            observation, info = trading_env.reset_at_point(0)
        
        assert 'error' in info
        assert 'Market simulator reset failed' in info['error']
    
    def test_step_success(self, trading_env):
        """Test successful environment step."""
        # Setup environment state
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {
            'hf': np.zeros((60, 7)),
            'mf': np.zeros((30, 43)),
            'lf': np.zeros((30, 19)),
            'portfolio': np.zeros((5, 5))
        }
        trading_env.episode_end_time_utc = datetime.now() + timedelta(hours=1)
        
        # Mock components
        market_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'current_price': 10.50,
            'best_bid_price': 10.48,
            'best_ask_price': 10.52
        }
        
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        trading_env.market_simulator.step.return_value = True
        
        trading_env.portfolio_manager = Mock()
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 100000.0,
            'cash': 90000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': 0.0,
            'positions': {'MLGO': {'quantity': 0.0}},
            'session_metrics': {
                'total_commissions_session': 0.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 0.0
            }
        }
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        trading_env.execution_manager = Mock()
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {
            'action_type': 'HOLD',
            'is_valid': True
        }
        trading_env.execution_manager.execute_action.return_value = execution_result
        
        trading_env.reward_calculator = Mock()
        trading_env.reward_calculator.calculate.return_value = 0.1
        trading_env.reward_calculator.get_last_reward_components.return_value = {}
        
        trading_env.initial_capital_for_session = 100000.0
        
        # Mock observation update
        with patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_metrics'):
            
            mock_obs.return_value = {
                'hf': np.zeros((60, 7)),
                'mf': np.zeros((30, 43)),
                'lf': np.zeros((30, 19)),
                'portfolio': np.zeros((5, 5))
            }
            
            action = np.array([0, 0])  # HOLD, SIZE_25
            observation, reward, terminated, truncated, info = trading_env.step(action)
        
        assert observation is not None
        assert reward == 0.1
        assert not terminated
        assert not truncated
        assert trading_env.current_step == 1
        assert info['step'] == 1
    
    def test_step_invalid_state(self, trading_env):
        """Test step with invalid environment state."""
        # No last observation
        trading_env._last_observation = None
        
        action = np.array([0, 0])
        observation, reward, terminated, truncated, info = trading_env.step(action)
        
        assert reward == 0.0
        assert terminated
        assert 'error' in info
    
    def test_step_market_state_failure(self, trading_env):
        """Test step when market simulator fails."""
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        
        # Mock failed market simulator
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = None
        
        action = np.array([0, 0])
        observation, reward, terminated, truncated, info = trading_env.step(action)
        
        assert reward == 0.0
        assert terminated
        assert 'error' in info
    
    def test_termination_conditions(self, trading_env):
        """Test various termination conditions."""
        # Setup base state
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        trading_env.initial_capital_for_session = 100000.0
        trading_env.bankruptcy_threshold_factor = 0.1
        trading_env.max_session_loss_percentage = 0.05
        trading_env.episode_end_time_utc = datetime.now() + timedelta(hours=1)
        trading_env.episode_start_time_utc = datetime.now() - timedelta(minutes=30)
        trading_env.episode_start_time = datetime.now().timestamp() - 1800  # 30 minutes ago
        
        # Mock basic components
        market_state = {'timestamp': datetime(2025, 1, 1, 9, 30), 'current_price': 10.0}
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        trading_env.market_simulator.step.return_value = True
        
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {'action_type': 'HOLD', 'is_valid': True}
        trading_env.execution_manager = Mock()
        trading_env.execution_manager.execute_action.return_value = execution_result
        
        trading_env.reward_calculator = Mock()
        trading_env.reward_calculator.calculate.return_value = 0.0
        trading_env.reward_calculator.get_last_reward_components.return_value = {}
        
        with patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_metrics'):
            
            mock_obs.return_value = {'hf': np.zeros((60, 7))}
            
            # Test bankruptcy termination
            portfolio_state = {
                'timestamp': datetime(2025, 1, 1, 9, 30),
                'total_equity': 5000.0,  # Below bankruptcy threshold (10% of 100k)
                'cash': 5000.0,
                'unrealized_pnl': 0.0,
                'realized_pnl_session': -95000.0,
                'positions': {},
                'session_metrics': {
                    'total_commissions_session': 0.0,
                    'total_fees_session': 0.0,
                    'total_slippage_cost_session': 0.0
                }
            }
            trading_env.portfolio_manager = Mock()
            trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
            trading_env.portfolio_manager.get_trading_metrics.return_value = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
            
            action = np.array([0, 0])
            _, _, terminated, _, info = trading_env.step(action)
            
            assert terminated
            assert info.get('termination_reason') == TerminationReasonEnum.BANKRUPTCY.value
    
    def test_termination_max_loss(self, trading_env):
        """Test max loss termination."""
        # Setup
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        trading_env.initial_capital_for_session = 100000.0
        trading_env.max_session_loss_percentage = 0.05  # 5% max loss
        trading_env.episode_end_time_utc = datetime.now() + timedelta(hours=1)
        trading_env.episode_start_time_utc = datetime.now() - timedelta(minutes=30)
        trading_env.episode_start_time = datetime.now().timestamp() - 1800
        
        # Mock components
        market_state = {'timestamp': datetime(2025, 1, 1, 9, 30), 'current_price': 10.0}
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        trading_env.market_simulator.step.return_value = True
        
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {'action_type': 'HOLD', 'is_valid': True}
        trading_env.execution_manager = Mock()
        trading_env.execution_manager.execute_action.return_value = execution_result
        
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 94000.0,  # Below 95% of initial capital (max loss threshold)
            'cash': 94000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_session': -6000.0,
            'positions': {},
            'session_metrics': {
                'total_commissions_session': 0.0,
                'total_fees_session': 0.0,
                'total_slippage_cost_session': 0.0
            }
        }
        trading_env.portfolio_manager = Mock()
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        trading_env.reward_calculator = Mock()
        trading_env.reward_calculator.calculate.return_value = 0.0
        trading_env.reward_calculator.get_last_reward_components.return_value = {}
        
        with patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_metrics'):
            
            mock_obs.return_value = {'hf': np.zeros((60, 7))}
            
            action = np.array([0, 0])
            _, _, terminated, _, info = trading_env.step(action)
            
            assert terminated
            assert info.get('termination_reason') == TerminationReasonEnum.MAX_LOSS_REACHED.value
    
    # Invalid action limit tests removed - action masking prevents invalid actions
    
    def test_termination_max_steps(self, trading_env):
        """Test max episode steps termination (natural end)."""
        # Setup
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        trading_env.config.env.max_episode_steps = 10
        trading_env.current_step = 9  # Next step will reach limit
        trading_env.episode_end_time_utc = datetime.now() + timedelta(hours=1)
        trading_env.initial_capital_for_session = 100000.0
        
        # Mock components
        market_state = {'timestamp': datetime(2025, 1, 1, 9, 30), 'current_price': 10.0}
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        trading_env.market_simulator.step.return_value = True
        
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {'action_type': 'HOLD', 'is_valid': True}
        trading_env.execution_manager = Mock()
        trading_env.execution_manager.execute_action.return_value = execution_result
        
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 100000.0,
            'positions': {}
        }
        trading_env.portfolio_manager = Mock()
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        trading_env.reward_calculator = Mock()
        trading_env.reward_calculator.calculate.return_value = 0.0
        trading_env.reward_calculator.get_last_reward_components.return_value = {}
        
        with patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_metrics'):
            
            mock_obs.return_value = {'hf': np.zeros((60, 7))}
            
            action = np.array([0, 0])
            _, _, terminated, _, info = trading_env.step(action)
            
            assert terminated
            assert info.get('termination_reason') == TerminationReasonEnum.MAX_STEPS_REACHED.value
            assert trading_env.current_step == 10
    
    def test_termination_end_of_data(self, trading_env):
        """Test end of data termination."""
        # Setup
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        trading_env.episode_end_time_utc = datetime.now() + timedelta(hours=1)
        trading_env.initial_capital_for_session = 100000.0
        
        # Mock components - market simulator fails to advance
        market_state = {'timestamp': datetime(2025, 1, 1, 9, 30), 'current_price': 10.0}
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        trading_env.market_simulator.step.return_value = False  # Failed to advance
        
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {'action_type': 'HOLD', 'is_valid': True}
        trading_env.execution_manager = Mock()
        trading_env.execution_manager.execute_action.return_value = execution_result
        
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 100000.0,
            'positions': {}
        }
        trading_env.portfolio_manager = Mock()
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        trading_env.reward_calculator = Mock()
        trading_env.reward_calculator.calculate.return_value = 0.0
        trading_env.reward_calculator.get_last_reward_components.return_value = {}
        
        with patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_metrics'):
            
            mock_obs.return_value = {'hf': np.zeros((60, 7))}
            
            action = np.array([0, 0])
            _, _, terminated, _, info = trading_env.step(action)
            
            assert terminated
            assert info.get('termination_reason') == TerminationReasonEnum.END_OF_SESSION_DATA.value
    
    def test_get_observation_success(self, trading_env):
        """Test successful observation generation."""
        # Setup
        trading_env.market_simulator = Mock()
        market_data = {
            'timestamp': datetime(2025, 1, 1, 9, 30)
        }
        trading_env.market_simulator.get_current_market_data.return_value = market_data
        
        features = {
            'hf': np.random.random((60, 7)),
            'mf': np.random.random((30, 43)),
            'lf': np.random.random((30, 19))
        }
        trading_env.market_simulator.get_current_features.return_value = features
        
        trading_env.portfolio_manager = Mock()
        portfolio_state = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        portfolio_obs = {'features': np.random.random((5, 5))}
        trading_env.portfolio_manager.get_portfolio_observation.return_value = portfolio_obs
        
        observation = trading_env._get_observation()
        
        assert observation is not None
        assert 'hf' in observation
        assert 'mf' in observation
        assert 'lf' in observation
        assert 'portfolio' in observation
        assert observation['hf'].shape == (60, 7)
        assert observation['mf'].shape == (30, 43)
        assert observation['lf'].shape == (30, 19)
        assert observation['portfolio'].shape == (5, 5)
    
    def test_get_observation_nan_handling(self, trading_env):
        """Test observation NaN handling."""
        # Setup with NaN values
        trading_env.market_simulator = Mock()
        market_data = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.market_simulator.get_current_market_data.return_value = market_data
        
        # Create features with NaN values
        hf_features = np.random.random((60, 7))
        hf_features[0, 0] = np.nan
        features = {
            'hf': hf_features,
            'mf': np.random.random((30, 43)),
            'lf': np.random.random((30, 19))
        }
        trading_env.market_simulator.get_current_features.return_value = features
        
        trading_env.portfolio_manager = Mock()
        portfolio_state = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        portfolio_obs = {'features': np.random.random((5, 5))}
        trading_env.portfolio_manager.get_portfolio_observation.return_value = portfolio_obs
        
        observation = trading_env._get_observation()
        
        assert observation is not None
        assert not np.isnan(observation['hf']).any()  # NaN should be replaced with 0
        assert observation['hf'][0, 0] == 0.0  # Specific NaN should be 0
    
    def test_get_observation_missing_features(self, trading_env):
        """Test observation with missing features."""
        # Setup with None features
        trading_env.market_simulator = Mock()
        market_data = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.market_simulator.get_current_market_data.return_value = market_data
        
        features = {
            'hf': None,  # Missing feature
            'mf': np.random.random((30, 43)),
            'lf': np.random.random((30, 19))
        }
        trading_env.market_simulator.get_current_features.return_value = features
        
        trading_env.portfolio_manager = Mock()
        portfolio_state = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        portfolio_obs = {'features': np.random.random((5, 5))}
        trading_env.portfolio_manager.get_portfolio_observation.return_value = portfolio_obs
        
        observation = trading_env._get_observation()
        
        assert observation is not None
        assert observation['hf'].shape == (60, 7)  # Should use zeros
        assert np.all(observation['hf'] == 0.0)  # Should be all zeros
    
    def test_get_observation_shape_mismatch(self, trading_env):
        """Test observation with wrong feature shapes."""
        # Setup with wrong shapes
        trading_env.market_simulator = Mock()
        market_data = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.market_simulator.get_current_market_data.return_value = market_data
        
        features = {
            'hf': np.random.random((50, 7)),  # Wrong sequence length
            'mf': np.random.random((30, 43)),
            'lf': np.random.random((30, 19))
        }
        trading_env.market_simulator.get_current_features.return_value = features
        
        trading_env.portfolio_manager = Mock()
        portfolio_state = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        portfolio_obs = {'features': np.random.random((5, 5))}
        trading_env.portfolio_manager.get_portfolio_observation.return_value = portfolio_obs
        
        observation = trading_env._get_observation()
        
        assert observation is None  # Should fail due to shape mismatch
    
    def test_get_observation_failure(self, trading_env):
        """Test observation generation failure."""
        # Setup with None market data
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = None
        
        observation = trading_env._get_observation()
        
        assert observation is None
    
    def test_get_dummy_observation(self, trading_env):
        """Test dummy observation generation."""
        dummy_obs = trading_env._get_dummy_observation()
        
        assert 'hf' in dummy_obs
        assert 'mf' in dummy_obs
        assert 'lf' in dummy_obs
        assert 'portfolio' in dummy_obs
        assert dummy_obs['hf'].shape == (60, 7)
        assert dummy_obs['mf'].shape == (30, 43)
        assert dummy_obs['lf'].shape == (30, 19)
        assert dummy_obs['portfolio'].shape == (5, 5)
        assert np.all(dummy_obs['hf'] == 0.0)
    
    def test_momentum_day_selection(self, trading_env, mock_data_manager):
        """Test momentum day selection functionality."""
        # Test getting momentum days
        momentum_days = trading_env.get_momentum_days(min_activity=0.0)
        assert len(momentum_days) == 2  # From mock data
        
        # Test selecting next momentum day
        selected_day = trading_env.select_next_momentum_day()
        assert selected_day is not None
        assert 'symbol' in selected_day
        assert 'date' in selected_day
        assert 'quality_score' in selected_day
        
        # Test with exclusions
        exclude_dates = [datetime(2025, 1, 1)]
        selected_day = trading_env.select_next_momentum_day(exclude_dates)
        assert selected_day['date'] != datetime(2025, 1, 1).date()
    
    def test_momentum_day_selection_empty(self, trading_env, mock_data_manager):
        """Test momentum day selection with no available days."""
        # Mock empty momentum days
        mock_data_manager.get_momentum_days.return_value = pd.DataFrame()
        
        selected_day = trading_env.select_next_momentum_day()
        assert selected_day is None
    
    def test_reset_point_navigation(self, trading_env):
        """Test reset point navigation methods."""
        # Setup
        trading_env.reset_points = [
            {'timestamp': datetime(2025, 1, 1, 9, 30)},
            {'timestamp': datetime(2025, 1, 1, 14, 0)},
            {'timestamp': datetime(2025, 1, 1, 15, 30)}
        ]
        trading_env.current_reset_idx = 0
        
        # Test has_more_reset_points
        assert trading_env.has_more_reset_points()
        
        # Test get_next_reset_point
        next_point = trading_env.get_next_reset_point()
        assert next_point is not None
        assert next_point['timestamp'] == datetime(2025, 1, 1, 14, 0)
        
        # Move to last reset point
        trading_env.current_reset_idx = 2
        assert not trading_env.has_more_reset_points()
        assert trading_env.get_next_reset_point() is None
    
    def test_handle_open_positions_at_reset(self, trading_env):
        """Test handling of open positions during reset."""
        # Setup
        trading_env.portfolio_manager = Mock()
        trading_env.market_simulator = Mock()
        trading_env.reward_calculator = Mock()
        
        # Mock market data
        market_data = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'current_price': 10.50,
            'best_ask': 10.52,
            'best_bid': 10.48
        }
        trading_env.market_simulator.get_current_market_data.return_value = market_data
        
        # Mock portfolio state with open position
        portfolio_state = {
            'positions': {
                'MLGO': {
                    'quantity': 100.0,  # Open position
                    'avg_entry_price': 10.00,
                    'unrealized_pnl': 50.0
                }
            }
        }
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        # Mock config values
        trading_env.config.simulation.min_commission_per_order = 1.0
        trading_env.config.simulation.commission_per_share = 0.005
        
        close_pnl = trading_env._handle_open_positions_at_reset()
        
        assert close_pnl is not None
        assert close_pnl < 50.0  # Should be less than unrealized PnL due to costs
    
    def test_handle_open_positions_no_positions(self, trading_env):
        """Test handling reset with no open positions."""
        # Setup
        trading_env.portfolio_manager = Mock()
        trading_env.market_simulator = Mock()
        
        market_data = {'timestamp': datetime(2025, 1, 1, 9, 30)}
        trading_env.market_simulator.get_current_market_data.return_value = market_data
        
        # Mock portfolio state with no positions
        portfolio_state = {
            'positions': {
                'MLGO': {
                    'quantity': 0.0,  # No position
                    'avg_entry_price': 0.0,
                    'unrealized_pnl': 0.0
                }
            }
        }
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        close_pnl = trading_env._handle_open_positions_at_reset()
        
        assert close_pnl is None
    
    def test_prepare_next_session(self, trading_env):
        """Test preparation of next session."""
        with patch('envs.trading_environment.MarketSimulator') as mock_market_sim_class:
            mock_market_sim = Mock()
            mock_market_sim.initialize_day.return_value = True
            mock_market_sim_class.return_value = mock_market_sim
            
            trading_env.prepare_next_session('MLGO', '2025-01-02')
            
            assert trading_env.next_market_simulator is not None
            mock_market_sim.initialize_day.assert_called_once()
    
    def test_switch_to_next_session(self, trading_env):
        """Test switching to prepared next session."""
        # Setup prepared session
        trading_env.next_market_simulator = Mock()
        trading_env.execution_manager = Mock()
        
        with patch.object(trading_env, '_generate_reset_points') as mock_reset_points:
            mock_reset_points.return_value = [{'timestamp': datetime(2025, 1, 2, 9, 30)}]
            
            trading_env.switch_to_next_session()
            
            assert trading_env.market_simulator is not None
            assert trading_env.next_market_simulator is None
            assert trading_env.current_reset_idx == 0
    
    def test_switch_to_next_session_not_prepared(self, trading_env):
        """Test switching when no session is prepared."""
        trading_env.next_market_simulator = None
        
        with pytest.raises(ValueError, match="No next session prepared"):
            trading_env.switch_to_next_session()
    
    def test_training_info_updates(self, trading_env):
        """Test setting training information."""
        trading_env.set_training_info(
            episode_num=100,
            total_episodes=1000,
            total_steps=50000,
            update_count=25
        )
        
        assert trading_env.episode_number == 100
        assert trading_env.total_episodes == 1000
        assert trading_env.total_steps == 50000
        assert trading_env.update_count == 25
    
    def test_render_functionality(self, trading_env, capsys):
        """Test render method output."""
        trading_env.render_mode = 'human'
        
        info_dict = {
            'step': 100,
            'reward_step': 0.5,
            'portfolio_equity': 105000.0
        }
        
        trading_env.render(info_dict)
        
        captured = capsys.readouterr()
        assert "Step 100" in captured.out
        assert "Reward 0.5000" in captured.out
        assert "Equity $105000.00" in captured.out
    
    def test_close_functionality(self, trading_env):
        """Test environment cleanup on close."""
        # Setup simulators with close methods
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.close = Mock()
        trading_env.next_market_simulator = Mock()
        trading_env.next_market_simulator.close = Mock()
        
        trading_env.close()
        
        trading_env.market_simulator.close.assert_called_once()
        trading_env.next_market_simulator.close.assert_called_once()
    
    def test_action_space_validation(self, trading_env):
        """Test action space properties."""
        # Action space should be MultiDiscrete([3, 4])
        assert hasattr(trading_env.action_space, 'nvec')
        assert trading_env.action_space.nvec.tolist() == [3, 4]
        
        # Sample valid actions
        for _ in range(10):
            action = trading_env.action_space.sample()
            assert len(action) == 2
            assert 0 <= action[0] < 3  # Action type
            assert 0 <= action[1] < 4  # Position size
    
    def test_observation_space_validation(self, trading_env):
        """Test observation space properties."""
        obs_space = trading_env.observation_space
        
        # Check required keys
        required_keys = ['hf', 'mf', 'lf', 'portfolio']
        for key in required_keys:
            assert key in obs_space.spaces
        
        # Check shapes
        assert obs_space['hf'].shape == (60, 7)
        assert obs_space['mf'].shape == (30, 43)
        assert obs_space['lf'].shape == (30, 19)
        assert obs_space['portfolio'].shape == (5, 5)
        
        # All should be Box spaces with float32 dtype
        for key in required_keys:
            assert obs_space[key].dtype == np.float32
    
    # Invalid action limit tests removed - action masking prevents invalid actions
    
    def test_episode_end_handling(self, trading_env):
        """Test episode end handling and summary creation."""
        # Setup
        trading_env.episode_total_reward = 10.5
        trading_env.current_step = 100
        trading_env.episode_start_time = datetime.now().timestamp() - 60  # 1 minute ago
        trading_env.initial_capital_for_session = 100000.0
        
        # Mock portfolio manager
        trading_env.portfolio_manager = Mock()
        portfolio_state = {
            'total_equity': 105000.0,
            'realized_pnl_session': 5000.0,
            'session_metrics': {
                'total_commissions_session': 50.0,
                'total_fees_session': 25.0,
                'total_slippage_cost_session': 75.0
            }
        }
        trading_env.portfolio_manager.get_trading_metrics.return_value = {
            'total_trades': 5,
            'winning_trades': 3,
            'losing_trades': 2
        }
        
        # Mock metrics integrator
        trading_env.metrics_integrator = Mock()
        
        info = {}
        trading_env._handle_episode_end(portfolio_state, info)
        
        # Check episode summary
        assert 'episode_summary' in info
        summary = info['episode_summary']
        assert summary['total_reward'] == 10.5
        assert summary['steps'] == 100
        assert summary['final_equity'] == 105000.0
        assert summary['session_realized_pnl_net'] == 5000.0
    
    def test_get_current_info(self, trading_env):
        """Test info dictionary generation."""
        # Setup
        trading_env.current_step = 50
        trading_env.episode_total_reward = 5.5
        trading_env.episode_number = 10
        trading_env.current_reset_idx = 2
        trading_env.reset_points = [None, None, None, None]  # 4 reset points
        trading_env.primary_asset = 'MLGO'
        trading_env.invalid_action_count_episode = 3
        trading_env._last_decoded_action = {
            'action_type': 'BUY',
            'is_valid': True,
            'position_size': 0.5
        }
        
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 105000.0,
            'cash': 95000.0,
            'unrealized_pnl': 2000.0,
            'realized_pnl_session': 3000.0,
            'positions': {
                'MLGO': {
                    'quantity': 100.0,
                    'current_side': PositionSideEnum.LONG,
                    'avg_entry_price': 10.50
                }
            }
        }
        
        fill_details = [Mock()]
        
        info = trading_env._get_current_info(
            reward=0.8,
            current_portfolio_state_for_info=portfolio_state,
            fill_details_list=fill_details,
            termination_reason_enum=TerminationReasonEnum.MAX_STEPS_REACHED,
            is_terminated=True,
            is_truncated=False
        )
        
        # Check basic info
        assert info['step'] == 50
        assert info['reward_step'] == 0.8
        assert info['episode_cumulative_reward'] == 5.5
        assert info['episode_number'] == 10
        assert info['reset_point_idx'] == 2
        assert info['reset_points_total'] == 4
        assert info['portfolio_equity'] == 105000.0
        assert info['invalid_actions_total_episode'] == 3
        assert info['termination_reason'] == TerminationReasonEnum.MAX_STEPS_REACHED.value
        
        # Check position details
        assert info['position_MLGO_qty'] == 100.0
        assert info['position_MLGO_side'] == PositionSideEnum.LONG.value
        assert info['position_MLGO_avg_entry'] == 10.50
        
        # Check fills
        assert len(info['fills_step']) == 1
    
    def test_edge_case_episode_end_time_boundary(self, trading_env):
        """Test episode termination at exact end time boundary."""
        # Setup
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        
        # Set episode end time to exactly now
        now = datetime.now()
        trading_env.episode_end_time_utc = now
        
        # Mock market state returning exactly the end time
        market_state = {
            'timestamp': now,
            'current_price': 10.0
        }
        
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        
        action = np.array([0, 0])
        _, _, terminated, _, info = trading_env.step(action)
        
        assert terminated
        assert info.get('termination_reason') == TerminationReasonEnum.MAX_DURATION.value
    
    def test_edge_case_zero_initial_capital(self, trading_env):
        """Test handling of zero initial capital edge case."""
        trading_env.initial_capital_for_session = 0.0
        trading_env.primary_asset = 'MLGO'
        trading_env._last_observation = {'hf': np.zeros((60, 7))}
        trading_env.episode_end_time_utc = datetime.now() + timedelta(hours=1)
        
        # Mock components
        market_state = {'timestamp': datetime(2025, 1, 1, 9, 30), 'current_price': 10.0}
        trading_env.market_simulator = Mock()
        trading_env.market_simulator.get_current_market_data.return_value = market_state
        trading_env.market_simulator.step.return_value = True
        
        execution_result = Mock()
        execution_result.fill_details = None
        execution_result.action_decode_result = Mock()
        execution_result.action_decode_result.action_type = "HOLD"
        execution_result.action_decode_result.is_valid = True
        execution_result.action_decode_result.to_dict.return_value = {'action_type': 'HOLD', 'is_valid': True}
        trading_env.execution_manager = Mock()
        trading_env.execution_manager.execute_action.return_value = execution_result
        
        portfolio_state = {
            'timestamp': datetime(2025, 1, 1, 9, 30),
            'total_equity': 0.0,
            'positions': {}
        }
        trading_env.portfolio_manager = Mock()
        trading_env.portfolio_manager.get_portfolio_state.return_value = portfolio_state
        
        trading_env.reward_calculator = Mock()
        trading_env.reward_calculator.calculate.return_value = 0.0
        trading_env.reward_calculator.get_last_reward_components.return_value = {}
        
        with patch.object(trading_env, '_get_observation') as mock_obs, \
             patch.object(trading_env, '_update_metrics'):
            
            mock_obs.return_value = {'hf': np.zeros((60, 7))}
            
            action = np.array([0, 0])
            _, _, terminated, _, _ = trading_env.step(action)
            
            # Should not crash with division by zero in P&L calculation
            assert not terminated  # Should not terminate due to bankruptcy calculations with zero capital