import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any
import gymnasium as gym

from envs.trading_env import TradingEnvironment
from envs.momentum_extensions import (
    MomentumTradingEnvironment,
    MomentumRewardSystem,
    SniperTradingRewards,
    MomentumContext
)


class TestMomentumTradingEnvironment:
    """Test momentum-specific modifications to TradingEnvironment."""
    
    @pytest.fixture
    def momentum_config(self):
        """Configuration for momentum trading environment."""
        return {
            'env': {
                'max_episode_duration': 14400,  # 4 hours
                'stop_loss': 0.05,              # 5%
                'single_day_only': True,
                'force_close_at_market_close': True
            },
            'momentum': {
                'time_multipliers': {
                    '0_30_seconds': 2.0,
                    '30_120_seconds': 1.5,
                    '120_300_seconds': 1.0,
                    'over_300_seconds': 0.5
                },
                'quick_profit_target': 0.003,    # 0.3%
                'momentum_alignment_bonus': 0.2,
                'max_position_duration': 300,    # 5 minutes
                'pattern_bonuses': {
                    'breakout': 0.15,
                    'flush': 0.10,
                    'bounce': 0.12
                }
            },
            'reward': {
                'components': [
                    'pnl',
                    'time_efficiency',
                    'momentum_alignment',
                    'quick_profit',
                    'position_sizing',
                    'risk_management'
                ]
            }
        }
    
    @pytest.fixture
    def mock_market_simulator(self):
        """Mock market simulator."""
        simulator = Mock()
        simulator.current_time = datetime(2025, 1, 15, 9, 30)
        simulator.get_current_market_state.return_value = {
            'bid': 10.0,
            'ask': 10.01,
            'last': 10.005,
            'volume': 100000,
            'bid_size': 5000,
            'ask_size': 5000
        }
        return simulator
    
    @pytest.fixture
    def mock_portfolio_manager(self):
        """Mock portfolio manager."""
        manager = Mock()
        manager.get_portfolio_state.return_value = {
            'cash': 100000,
            'positions': {},
            'total_value': 100000
        }
        return manager
    
    @pytest.fixture
    def mock_feature_extractor(self):
        """Mock feature extractor."""
        extractor = Mock()
        extractor.extract_features.return_value = np.array([0.5] * 100)
        return extractor
    
    @pytest.fixture
    def momentum_env(self, momentum_config, mock_market_simulator, 
                     mock_portfolio_manager, mock_feature_extractor):
        """Create momentum trading environment."""
        env = MomentumTradingEnvironment(
            config=momentum_config,
            market_simulator=mock_market_simulator,
            portfolio_manager=mock_portfolio_manager,
            feature_extractor=mock_feature_extractor
        )
        return env
    
    def test_momentum_environment_initialization(self, momentum_env, momentum_config):
        """Test momentum environment initialization."""
        assert isinstance(momentum_env, MomentumTradingEnvironment)
        assert momentum_env.momentum_config == momentum_config['momentum']
        assert momentum_env.single_day_only is True
        assert momentum_env.max_position_duration == 300
    
    def test_reset_for_momentum_training(self, momentum_env):
        """Test momentum-specific reset functionality."""
        reset_point = Mock(
            timestamp=datetime(2025, 1, 15, 9, 35),
            momentum_phase='front_breakout',
            quality_score=0.9,
            pattern_type='breakout',
            metadata={'volume_surge': 3.5}
        )
        
        obs = momentum_env.reset_for_momentum_training(reset_point)
        
        assert momentum_env.current_reset_point == reset_point
        assert momentum_env.momentum_context is not None
        assert momentum_env.momentum_context.phase == 'front_breakout'
        assert momentum_env.momentum_context.pattern == 'breakout'
        assert isinstance(obs, np.ndarray)
    
    def test_momentum_context_application(self, momentum_env):
        """Test application of momentum context to environment."""
        context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.9,
            volume_ratio=3.5,
            time_of_day='market_open',
            metadata={'consolidation_duration': 300}
        )
        
        momentum_env._apply_momentum_context(context)
        
        assert momentum_env.momentum_context == context
        assert momentum_env.reward_system.momentum_context == context
        
        # Should affect reward calculation
        base_reward = momentum_env.reward_system.calculate_reward(
            pnl=100, action='buy', position_held_time=30
        )
        
        # With breakout context, should get pattern bonus
        assert base_reward > 100  # PnL + bonuses
    
    def test_sniper_reward_calculation(self, momentum_env):
        """Test sniper-style reward calculation."""
        momentum_env.momentum_context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.9
        )
        
        # Quick profitable trade (best case)
        reward_quick = momentum_env._calculate_momentum_reward(
            pnl=300,  # 0.3% on 100k
            position_held_time=25,  # 25 seconds
            action_type='buy',
            momentum_aligned=True
        )
        
        # Slow profitable trade
        reward_slow = momentum_env._calculate_momentum_reward(
            pnl=300,
            position_held_time=400,  # Over 5 minutes
            action_type='buy',
            momentum_aligned=True
        )
        
        # Quick trade should have much higher reward
        assert reward_quick > reward_slow * 2
        
        # Test time multipliers
        assert momentum_env._get_time_multiplier(25) == 2.0      # 0-30s
        assert momentum_env._get_time_multiplier(60) == 1.5      # 30-120s
        assert momentum_env._get_time_multiplier(200) == 1.0     # 120-300s
        assert momentum_env._get_time_multiplier(400) == 0.5     # >300s
    
    def test_position_duration_enforcement(self, momentum_env, mock_portfolio_manager):
        """Test max position duration enforcement."""
        # Set up position that's too old
        mock_portfolio_manager.get_portfolio_state.return_value = {
            'cash': 95000,
            'positions': {
                'MLGO': {
                    'quantity': 500,
                    'entry_time': datetime(2025, 1, 15, 9, 30),
                    'unrealized_pnl': 50
                }
            },
            'total_value': 100050
        }
        
        # Current time is 6 minutes later
        momentum_env.market_simulator.current_time = datetime(2025, 1, 15, 9, 36)
        
        # Step should force position closure
        obs, reward, done, truncated, info = momentum_env.step(0)  # Hold action
        
        assert info.get('forced_position_closure', False) is True
        assert info.get('closure_reason') == 'max_duration_exceeded'
    
    def test_momentum_alignment_detection(self, momentum_env):
        """Test detection of momentum-aligned actions."""
        momentum_env.momentum_context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.9
        )
        
        # Buy during breakout - aligned
        aligned = momentum_env._is_momentum_aligned('buy', 'front_breakout')
        assert aligned is True
        
        # Sell during breakout - not aligned
        not_aligned = momentum_env._is_momentum_aligned('sell', 'front_breakout')
        assert not_aligned is False
        
        # Sell during flush - aligned
        momentum_env.momentum_context.phase = 'back_flush'
        aligned = momentum_env._is_momentum_aligned('sell', 'back_flush')
        assert aligned is True
    
    def test_episode_termination_info(self, momentum_env, mock_portfolio_manager):
        """Test episode termination info without forcing closure."""
        # Set up with open position
        mock_portfolio_manager.get_portfolio_state.return_value = {
            'cash': 95000,
            'positions': {
                'MLGO': {
                    'quantity': 500,
                    'side': 'long',
                    'avg_price': 10.0,
                    'current_price': 10.1,
                    'unrealized_pnl': 50,
                    'entry_time': datetime(2025, 1, 15, 9, 35)
                }
            },
            'total_value': 100050
        }
        
        info = momentum_env.get_episode_termination_info()
        
        assert 'portfolio_state' in info
        assert 'position_info' in info
        assert info['position_info']['has_position'] is True
        assert info['position_info']['quantity'] == 500
        assert info['position_info']['unrealized_pnl'] == 50
        assert 'termination_reason' in info
    
    def test_quick_profit_bonus(self, momentum_env):
        """Test quick profit target bonus."""
        momentum_env.momentum_context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.9
        )
        
        # Hit quick profit target
        reward_with_bonus = momentum_env._calculate_momentum_reward(
            pnl=300,  # 0.3% target hit
            position_held_time=20,
            action_type='buy',
            momentum_aligned=True,
            hit_profit_target=True
        )
        
        # Same profit but didn't hit target quickly
        reward_no_bonus = momentum_env._calculate_momentum_reward(
            pnl=300,
            position_held_time=200,  # Too slow
            action_type='buy',
            momentum_aligned=True,
            hit_profit_target=False
        )
        
        assert reward_with_bonus > reward_no_bonus
    
    def test_pattern_specific_rewards(self, momentum_env):
        """Test pattern-specific reward adjustments."""
        patterns = ['breakout', 'flush', 'bounce']
        base_pnl = 200
        
        rewards = {}
        for pattern in patterns:
            momentum_env.momentum_context = MomentumContext(
                phase='front_breakout' if pattern == 'breakout' else 'various',
                pattern=pattern,
                quality_score=0.8
            )
            
            reward = momentum_env._calculate_momentum_reward(
                pnl=base_pnl,
                position_held_time=60,
                action_type='buy',
                momentum_aligned=True
            )
            rewards[pattern] = reward
        
        # Different patterns should yield different rewards
        assert len(set(rewards.values())) > 1
        
        # Breakout should have highest bonus
        assert rewards['breakout'] >= max(rewards.values())
    
    def test_dead_zone_behavior(self, momentum_env):
        """Test environment behavior during dead zones."""
        momentum_env.momentum_context = MomentumContext(
            phase='dead',
            pattern=None,
            quality_score=0.2
        )
        
        # Taking position during dead zone
        reward = momentum_env._calculate_momentum_reward(
            pnl=-50,  # Small loss
            position_held_time=180,
            action_type='buy',
            momentum_aligned=False
        )
        
        # Should penalize trading in dead zones
        assert reward < -50  # Worse than just the loss
    
    def test_observation_space_extensions(self, momentum_env):
        """Test momentum-specific observation space extensions."""
        obs = momentum_env.reset_for_momentum_training(Mock(
            timestamp=datetime(2025, 1, 15, 9, 30),
            momentum_phase='front_breakout',
            quality_score=0.9,
            pattern_type='breakout'
        ))
        
        # Should include momentum context features
        assert obs.shape[0] >= 100  # Base features + momentum features
        
        # Check momentum feature indices
        momentum_features = momentum_env._extract_momentum_features()
        assert 'phase_encoding' in momentum_features
        assert 'pattern_encoding' in momentum_features
        assert 'quality_score' in momentum_features
    
    def test_action_masking_by_phase(self, momentum_env):
        """Test action masking based on momentum phase."""
        # During parabolic phase, might want to prevent new buys
        momentum_env.momentum_context = MomentumContext(
            phase='parabolic',
            pattern='breakout',
            quality_score=0.95
        )
        
        valid_actions = momentum_env.get_valid_actions()
        
        # Should mask certain actions during extreme momentum
        assert len(valid_actions) <= momentum_env.action_space.n
        
        # Specifically, large buys might be masked
        large_buy_actions = [8, 9, 10, 11]  # Assuming these are large buy actions
        for action in large_buy_actions:
            assert action not in valid_actions or momentum_env.should_mask_action(action)
    
    def test_info_dict_momentum_data(self, momentum_env):
        """Test momentum data in step info dictionary."""
        momentum_env.momentum_context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.9,
            metadata={'volume_surge': 3.5}
        )
        
        obs, reward, done, truncated, info = momentum_env.step(1)  # Buy action
        
        assert 'momentum_phase' in info
        assert 'pattern_type' in info
        assert 'reset_quality' in info
        assert 'time_multiplier' in info
        assert 'momentum_aligned' in info
        
        assert info['momentum_phase'] == 'front_breakout'
        assert info['pattern_type'] == 'breakout'
        assert info['reset_quality'] == 0.9
    
    def test_compatibility_with_base_environment(self, momentum_env):
        """Test that momentum environment is compatible with base."""
        # Should work with standard gym interface
        assert hasattr(momentum_env, 'reset')
        assert hasattr(momentum_env, 'step')
        assert hasattr(momentum_env, 'render')
        assert hasattr(momentum_env, 'close')
        
        # Standard reset should still work
        obs = momentum_env.reset()
        assert isinstance(obs, np.ndarray)
        
        # Standard step should work
        obs, reward, done, truncated, info = momentum_env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestMomentumRewardSystem:
    """Test the momentum-specific reward system."""
    
    @pytest.fixture
    def reward_config(self):
        """Reward system configuration."""
        return {
            'base_multiplier': 1.0,
            'time_multipliers': {
                '0_30_seconds': 2.0,
                '30_120_seconds': 1.5,
                '120_300_seconds': 1.0,
                'over_300_seconds': 0.5
            },
            'momentum_bonus': 0.2,
            'pattern_bonuses': {
                'breakout': 0.15,
                'flush': 0.10,
                'bounce': 0.12
            },
            'quick_profit_bonus': 0.25,
            'quick_profit_threshold': 0.003,
            'dead_zone_penalty': -0.3
        }
    
    @pytest.fixture
    def momentum_reward_system(self, reward_config):
        """Create momentum reward system."""
        return MomentumRewardSystem(reward_config)
    
    def test_time_based_rewards(self, momentum_reward_system):
        """Test time-based reward multipliers."""
        base_pnl = 100
        
        # Test different holding times
        rewards = {}
        for time in [10, 60, 180, 360]:
            reward = momentum_reward_system.calculate_reward(
                pnl=base_pnl,
                hold_time=time,
                momentum_context=MomentumContext(phase='front_breakout', pattern='breakout')
            )
            rewards[time] = reward
        
        # Shorter holds should have higher rewards
        assert rewards[10] > rewards[60] > rewards[180] > rewards[360]
        
        # Check specific multipliers
        assert rewards[10] == pytest.approx(base_pnl * 2.0 * 1.15, rel=0.1)  # 2x time + pattern
    
    def test_momentum_alignment_bonus(self, momentum_reward_system):
        """Test momentum alignment bonus calculation."""
        context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.9
        )
        
        # Aligned action
        reward_aligned = momentum_reward_system.calculate_reward(
            pnl=100,
            hold_time=60,
            action='buy',
            momentum_context=context,
            momentum_aligned=True
        )
        
        # Non-aligned action
        reward_not_aligned = momentum_reward_system.calculate_reward(
            pnl=100,
            hold_time=60,
            action='sell',
            momentum_context=context,
            momentum_aligned=False
        )
        
        assert reward_aligned > reward_not_aligned
        assert reward_aligned == pytest.approx(
            reward_not_aligned * (1 + momentum_reward_system.config['momentum_bonus']),
            rel=0.1
        )
    
    def test_pattern_specific_bonuses(self, momentum_reward_system):
        """Test pattern-specific reward bonuses."""
        patterns = ['breakout', 'flush', 'bounce', None]
        
        rewards = {}
        for pattern in patterns:
            context = MomentumContext(
                phase='various',
                pattern=pattern,
                quality_score=0.8
            )
            
            reward = momentum_reward_system.calculate_reward(
                pnl=100,
                hold_time=60,
                momentum_context=context
            )
            rewards[pattern] = reward
        
        # Pattern bonuses should apply
        assert rewards['breakout'] > rewards[None]
        assert rewards['flush'] > rewards[None]
        assert rewards['bounce'] > rewards[None]
        
        # Breakout should have highest bonus
        assert rewards['breakout'] == max(
            rewards['breakout'], rewards['flush'], rewards['bounce']
        )
    
    def test_quick_profit_achievement(self, momentum_reward_system):
        """Test quick profit target achievement bonus."""
        context = MomentumContext(phase='front_breakout', pattern='breakout')
        
        # Quick profit hit
        reward_quick = momentum_reward_system.calculate_reward(
            pnl=350,  # 0.35% > 0.3% threshold
            hold_time=20,
            momentum_context=context,
            portfolio_value=100000
        )
        
        # Same profit but slower
        reward_slow = momentum_reward_system.calculate_reward(
            pnl=350,
            hold_time=200,
            momentum_context=context,
            portfolio_value=100000
        )
        
        # Quick should have bonus
        assert reward_quick > reward_slow * 1.2
    
    def test_dead_zone_penalties(self, momentum_reward_system):
        """Test penalties for trading in dead zones."""
        dead_context = MomentumContext(
            phase='dead',
            pattern=None,
            quality_score=0.1
        )
        
        # Even profitable trades in dead zones are penalized
        reward = momentum_reward_system.calculate_reward(
            pnl=50,
            hold_time=120,
            momentum_context=dead_context
        )
        
        # Should be less than raw PnL due to penalty
        assert reward < 50
        
        # Losses in dead zones are extra penalized
        reward_loss = momentum_reward_system.calculate_reward(
            pnl=-50,
            hold_time=120,
            momentum_context=dead_context
        )
        
        assert reward_loss < -50 * 1.3  # Extra penalty
    
    def test_reward_clipping_and_normalization(self, momentum_reward_system):
        """Test reward clipping and normalization."""
        # Extreme profit
        reward_high = momentum_reward_system.calculate_reward(
            pnl=10000,
            hold_time=30,
            momentum_context=MomentumContext(phase='parabolic', pattern='breakout')
        )
        
        # Should be clipped
        assert reward_high <= momentum_reward_system.max_reward
        
        # Extreme loss
        reward_low = momentum_reward_system.calculate_reward(
            pnl=-10000,
            hold_time=300,
            momentum_context=MomentumContext(phase='back_flush', pattern='flush')
        )
        
        assert reward_low >= momentum_reward_system.min_reward
    
    def test_composite_reward_calculation(self, momentum_reward_system):
        """Test composite reward with all components."""
        context = MomentumContext(
            phase='front_breakout',
            pattern='breakout',
            quality_score=0.95,
            volume_ratio=4.0
        )
        
        # Best case scenario
        reward = momentum_reward_system.calculate_reward(
            pnl=400,              # Good profit
            hold_time=15,         # Very quick (2x multiplier)
            action='buy',
            momentum_aligned=True,  # +20% bonus
            portfolio_value=100000,
            hit_profit_target=True,  # +25% bonus
            momentum_context=context
        )
        
        # Should have all bonuses
        # Base: 400 * 2.0 (time) = 800
        # Pattern: 800 * 1.15 = 920
        # Momentum: 920 * 1.2 = 1104
        # Quick profit: 1104 * 1.25 = 1380
        assert reward > 1300
        assert reward < momentum_reward_system.max_reward