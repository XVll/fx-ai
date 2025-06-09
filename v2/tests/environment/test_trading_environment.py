"""
Test v2 TradingEnvironment integration

Verifies that the new clean TradingEnvironment works properly with:
- ActionMask functionality
- Episode management
- Observation generation
- Step mechanics
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from v2.envs import TradingEnvironment, EpisodeConfig, ActionMask
from v2.core.types import ActionType, PositionSizeType


class TestTradingEnvironment:
    """Test the v2 TradingEnvironment."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        
        # Model config
        config.model.hf_seq_len = 60
        config.model.hf_feat_dim = 10
        config.model.mf_seq_len = 12
        config.model.mf_feat_dim = 8
        config.model.lf_seq_len = 1
        config.model.lf_feat_dim = 6
        config.model.portfolio_seq_len = 1
        config.model.portfolio_feat_dim = 4
        
        # Environment config
        config.env.reward = Mock()
        
        # Simulation config
        config.simulation.initial_capital = 25000.0
        config.simulation.max_position_value_ratio = 0.95
        config.simulation.min_trade_value = 100.0
        
        return config
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create mock data manager."""
        return Mock()
    
    @pytest.fixture
    def env(self, mock_config, mock_data_manager):
        """Create test environment."""
        return TradingEnvironment(
            config=mock_config,
            data_manager=mock_data_manager,
            callback_manager=None,
        )
    
    def test_initialization(self, env):
        """Test environment initialization."""
        assert env is not None
        assert env.action_space.shape == (2,)  # [action_type, position_size]
        assert env.observation_space['hf'].shape == (60, 10)
        assert env.observation_space['mf'].shape == (12, 8)
        assert env.observation_space['lf'].shape == (1, 6)
        assert env.observation_space['portfolio'].shape == (1, 4)
    
    def test_episode_setup(self, env):
        """Test episode configuration setup."""
        episode_config = EpisodeConfig(
            symbol="AAPL",
            date=datetime(2024, 1, 15),
            start_time=datetime(2024, 1, 15, 14, 30),  # 9:30 AM ET in UTC
            end_time=datetime(2024, 1, 15, 16, 30),    # 11:30 AM ET in UTC
            max_steps=1000,
        )
        
        # Mock market simulator
        env.market_simulator = Mock()
        env.market_simulator.initialize_day.return_value = True
        env.market_simulator.get_stats.return_value = {
            'total_seconds': 7200,
            'warmup_info': {'has_warmup': True}
        }
        
        # Test setup
        result = env.setup_episode(episode_config)
        assert result is True
        assert env.episode_config == episode_config
    
    def test_compatibility_methods(self, env):
        """Test v2 compatibility adapter methods."""
        # Test setup_session
        env.setup_session("AAPL", "2024-01-15")
        assert hasattr(env, '_session_symbol')
        assert env._session_symbol == "AAPL"
        assert env._session_date.strftime('%Y-%m-%d') == "2024-01-15"
        
        # Mock required components for reset_at_point
        env.market_simulator = Mock()
        env.market_simulator.initialize_day.return_value = True
        env.market_simulator.reset.return_value = True
        env.market_simulator.set_time.return_value = True
        env.market_simulator.get_market_state.return_value = Mock(timestamp=datetime.now())
        env.market_simulator.get_stats.return_value = {
            'total_seconds': 7200,
            'warmup_info': {'has_warmup': True}
        }
        env.market_simulator.get_current_features.return_value = {
            'hf': np.zeros((60, 10), dtype=np.float32),
            'mf': np.zeros((12, 8), dtype=np.float32),
            'lf': np.zeros((1, 6), dtype=np.float32),
        }
        
        # Mock portfolio simulator
        env.portfolio_simulator = Mock()
        env.portfolio_simulator.initial_capital = 25000.0
        env.portfolio_simulator.reset = Mock()
        env.portfolio_simulator.get_portfolio_observation.return_value = {
            'features': np.zeros((1, 4), dtype=np.float32)
        }
        
        # Mock other components
        env.execution_simulator = Mock()
        env.execution_simulator.reset = Mock()
        env.reward_system = Mock()
        env.action_mask = Mock()
        
        # Test reset_at_point
        obs, info = env.reset_at_point(0)
        assert obs is not None
        assert 'hf' in obs
        assert 'mf' in obs
        assert 'lf' in obs
        assert 'portfolio' in obs
        assert info is not None


class TestActionMask:
    """Test the v2 ActionMask functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock simulation config."""
        config = Mock()
        config.max_position_value_ratio = 0.95
        config.min_trade_value = 100.0
        return config
    
    @pytest.fixture
    def action_mask(self, mock_config):
        """Create test action mask."""
        return ActionMask(config=mock_config)
    
    def test_action_mask_all_valid(self, action_mask):
        """Test action mask when all actions should be valid."""
        portfolio_state = {
            'cash': 10000.0,
            'total_equity': 10000.0,
            'position_value': 0.0,
            'positions': {}
        }
        
        market_state = {
            'current_price': 100.0,
            'best_bid_price': 99.95,
            'best_ask_price': 100.05
        }
        
        mask = action_mask.get_action_mask(portfolio_state, market_state)
        
        # HOLD actions (0-3) should always be valid
        assert np.all(mask[0:4] == True)
        
        # BUY actions (4-7) should be valid with sufficient cash
        assert np.all(mask[4:8] == True)
        
        # SELL actions (8-11) should be invalid with no position
        assert np.all(mask[8:12] == False)
    
    def test_action_mask_with_position(self, action_mask):
        """Test action mask when holding a position."""
        portfolio_state = {
            'cash': 5000.0,
            'total_equity': 10000.0,
            'position_value': 5000.0,
            'positions': {
                'AAPL': {'quantity': 50, 'avg_price': 100.0}
            }
        }
        
        market_state = {
            'current_price': 100.0,
            'best_bid_price': 99.95,
            'best_ask_price': 100.05
        }
        
        mask = action_mask.get_action_mask(portfolio_state, market_state)
        
        # HOLD actions should be valid
        assert np.all(mask[0:4] == True)
        
        # Some BUY actions might be limited by position limits
        assert mask[4] == True   # 25% of 5000 = 1250, should be OK
        assert mask[5] == True   # 50% of 5000 = 2500, should be OK
        
        # SELL actions should be valid with position
        assert np.all(mask[8:12] == True)
    
    def test_action_encoding_decoding(self, action_mask):
        """Test action encoding and decoding."""
        # Test all combinations
        for action_type in ActionType:
            for position_size in PositionSizeType:
                # Encode
                linear_idx = action_mask.encode_action(action_type, position_size)
                
                # Decode
                decoded_type, decoded_size = action_mask.decode_action(linear_idx)
                
                # Verify
                assert decoded_type == action_type
                assert decoded_size == position_size
                
                # Test description
                desc = action_mask.get_action_description(linear_idx)
                assert action_type.name in desc
                assert str(int((position_size.value + 1) * 25)) in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])