"""
Shared test fixtures and configuration for pytest.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import commonly used modules for tests
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

# Import project modules for fixtures
from envs.trading_environment import TradingEnvironment
from simulators.portfolio_simulator import PositionSideEnum


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock()
    
    # Environment config
    config.env = MagicMock()
    config.env.initial_capital = 100000.0
    config.env.invalid_action_limit = 100
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
    
    # Simulation config
    config.simulation = MagicMock()
    config.simulation.default_position_value = 10000.0
    config.simulation.min_commission_per_order = 1.0
    config.simulation.commission_per_share = 0.005
    
    return config


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager."""
    data_manager = MagicMock()
    # Setup default return values
    data_manager.get_reset_points.return_value = pd.DataFrame()
    data_manager.get_momentum_days.return_value = pd.DataFrame()
    return data_manager


@pytest.fixture
def env_with_session(mock_config, mock_data_manager):
    """Create an environment with a mock session setup."""
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
    
    # Setup fixed reset points with timezone-aware timestamps
    env.reset_points = [
        {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00', tz='UTC'),
            'activity_score': 0.5,
            'combined_score': 0.5,
            'max_duration_hours': 4,
            'reset_type': 'fixed'
        }
    ]
    
    # Create mock simulators
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


@pytest.fixture
def env_ready(env_with_session):
    """Environment that has been reset and is ready for steps."""
    env_with_session.reset()
    return env_with_session