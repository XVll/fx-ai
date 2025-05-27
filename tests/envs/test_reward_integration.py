"""Test suite for environment integration with RewardSystemV2.

This covers:
- Reward calculation with momentum context
- Component-based reward tracking
- Time-based efficiency rewards
- Risk management penalties
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

from envs.environment_simulator import EnvironmentSimulator
from rewards.calculator import RewardSystemV2
from rewards.components import (
    PnLComponent,
    MomentumAlignmentComponent,
    TimeEfficiencyComponent,
    RiskManagementComponent,
    ActionCostComponent
)
from envs.momentum_episode_manager import MomentumContext, PatternType, PhaseType


class TestRewardIntegration:
    """Test integration between environment and reward system."""
    
    @pytest.fixture
    def reward_config(self):
        """Reward system configuration."""
        return {
            'reward': {
                'system_version': 'v2',
                'components': {
                    'pnl': {
                        'weight': 0.4,
                        'realized_weight': 0.7,
                        'unrealized_weight': 0.3,
                        'scaling_factor': 0.001  # Scale to reasonable range
                    },
                    'momentum_alignment': {
                        'weight': 0.2,
                        'alignment_bonus': 0.5,
                        'misalignment_penalty': -0.3,
                        'pattern_bonuses': {
                            'breakout': 0.2,
                            'flush': 0.15,
                            'bounce': 0.1,
                            'consolidation': -0.1
                        }
                    },
                    'time_efficiency': {
                        'weight': 0.2,
                        'optimal_hold_time': 120,  # 2 minutes
                        'time_decay_factor': 0.1,
                        'quick_profit_bonus': 0.3,
                        'quick_profit_threshold': 30  # 30 seconds
                    },
                    'risk_management': {
                        'weight': 0.15,
                        'max_drawdown_penalty': -1.0,
                        'drawdown_threshold': 0.02,  # 2%
                        'position_sizing_bonus': 0.2,
                        'overleveraging_penalty': -0.5
                    },
                    'action_cost': {
                        'weight': 0.05,
                        'trade_cost': -0.1,
                        'excessive_trading_penalty': -0.5,
                        'excessive_trading_threshold': 10  # trades per episode
                    }
                },
                'clipping': {
                    'min_reward': -2.0,
                    'max_reward': 2.0
                },
                'normalization': {
                    'method': 'standardization',
                    'running_mean_alpha': 0.99
                }
            }
        }
    
    @pytest.fixture
    def momentum_context(self):
        """Create sample momentum context."""
        return MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9,
            day_quality=0.85,
            intraday_move=0.15,
            volume_multiplier=3.5,
            time_of_day='market_open',
            metadata={
                'pre_consolidation_minutes': 10,
                'volume_surge': 4.0
            }
        )
    
    @pytest.fixture
    def reward_system(self, reward_config):
        """Create RewardSystemV2 instance."""
        return RewardSystemV2(reward_config['reward'])
    
    def test_basic_reward_calculation(self, reward_system):
        """Test basic reward calculation with all components."""
        state = {
            'realized_pnl': 500,
            'unrealized_pnl': 200,
            'position_held_time': 90,
            'action': 'buy',
            'momentum_aligned': True,
            'current_drawdown': 0.01,
            'trades_this_episode': 3,
            'portfolio_value': 100000
        }
        
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9
        )
        
        reward_info = reward_system.calculate(state, context)
        
        # Check structure
        assert 'total_reward' in reward_info
        assert 'components' in reward_info
        assert 'metadata' in reward_info
        
        # Check components
        components = reward_info['components']
        assert 'pnl' in components
        assert 'momentum_alignment' in components
        assert 'time_efficiency' in components
        assert 'risk_management' in components
        assert 'action_cost' in components
        
        # Total should be weighted sum
        total = sum(
            components[name] * reward_system.component_weights[name]
            for name in components
        )
        assert abs(reward_info['total_reward'] - total) < 0.001
    
    def test_pnl_component(self, reward_system):
        """Test P&L reward component calculation."""
        pnl_component = reward_system.components['pnl']
        
        # Profitable trade
        state = {
            'realized_pnl': 1000,
            'unrealized_pnl': 500,
            'portfolio_value': 100000
        }
        
        reward = pnl_component.calculate(state, None)
        
        # Should be positive and scaled
        assert reward > 0
        expected = (1000 * 0.7 + 500 * 0.3) * 0.001  # Weighted and scaled
        assert abs(reward - expected) < 0.001
        
        # Losing trade
        state['realized_pnl'] = -500
        state['unrealized_pnl'] = -200
        
        reward = pnl_component.calculate(state, None)
        assert reward < 0
    
    def test_momentum_alignment_component(self, reward_system, momentum_context):
        """Test momentum alignment reward component."""
        alignment_component = reward_system.components['momentum_alignment']
        
        # Aligned action during breakout
        state = {
            'action': 'buy',
            'momentum_aligned': True
        }
        
        reward = alignment_component.calculate(state, momentum_context)
        
        # Should get alignment bonus + pattern bonus
        assert reward > 0
        assert reward >= 0.5 + 0.2  # alignment + breakout bonus
        
        # Misaligned action
        state['momentum_aligned'] = False
        reward = alignment_component.calculate(state, momentum_context)
        
        # Should get penalty
        assert reward < 0
        assert reward <= -0.3  # misalignment penalty
        
        # Hold during consolidation (good)
        momentum_context.pattern = PatternType.CONSOLIDATION
        state = {
            'action': 'hold',
            'momentum_aligned': True
        }
        
        reward = alignment_component.calculate(state, momentum_context)
        assert reward > 0  # Holding during consolidation is good
    
    def test_time_efficiency_component(self, reward_system):
        """Test time efficiency reward component."""
        time_component = reward_system.components['time_efficiency']
        
        # Quick profitable trade
        state = {
            'position_held_time': 25,  # 25 seconds
            'realized_pnl': 300,
            'action': 'sell'  # Closing position
        }
        
        reward = time_component.calculate(state, None)
        
        # Should get quick profit bonus
        assert reward > 0.3  # Base quick profit bonus
        
        # Optimal hold time
        state['position_held_time'] = 120  # 2 minutes (optimal)
        state['realized_pnl'] = 300
        
        reward = time_component.calculate(state, None)
        assert reward > 0  # Still positive but no quick bonus
        
        # Too long hold
        state['position_held_time'] = 600  # 10 minutes
        reward = time_component.calculate(state, None)
        
        # Should decay with time
        assert reward < time_component.calculate({'position_held_time': 120}, None)
    
    def test_risk_management_component(self, reward_system):
        """Test risk management reward component."""
        risk_component = reward_system.components['risk_management']
        
        # Good risk management
        state = {
            'current_drawdown': 0.005,  # 0.5% drawdown
            'position_size_fraction': 0.25,  # Conservative sizing
            'max_position_fraction': 0.5,
            'portfolio_value': 100000
        }
        
        reward = risk_component.calculate(state, None)
        
        # Should get position sizing bonus
        assert reward > 0
        
        # Excessive drawdown
        state['current_drawdown'] = 0.03  # 3% drawdown
        reward = risk_component.calculate(state, None)
        
        # Should get penalty
        assert reward < 0
        
        # Overleveraging
        state['current_drawdown'] = 0.01
        state['position_size_fraction'] = 1.0  # Full size
        state['margin_used_fraction'] = 0.9  # 90% margin used
        
        reward = risk_component.calculate(state, None)
        assert reward < 0  # Overleveraging penalty
    
    def test_action_cost_component(self, reward_system):
        """Test action cost penalty component."""
        action_component = reward_system.components['action_cost']
        
        # Normal trading
        state = {
            'action': 'buy',
            'trades_this_episode': 2,
            'episode_duration': 3600  # 1 hour
        }
        
        reward = action_component.calculate(state, None)
        
        # Small penalty for trade
        assert reward < 0
        assert reward > -0.2  # Not excessive
        
        # Excessive trading
        state['trades_this_episode'] = 15
        state['episode_duration'] = 1800  # 30 minutes
        
        reward = action_component.calculate(state, None)
        
        # Should get excessive trading penalty
        assert reward < -0.5
        
        # Hold action (no cost)
        state['action'] = 'hold'
        state['trades_this_episode'] = 2
        
        reward = action_component.calculate(state, None)
        assert reward == 0  # No cost for holding
    
    def test_reward_clipping(self, reward_system):
        """Test reward clipping to prevent extreme values."""
        # Extreme profit
        state = {
            'realized_pnl': 50000,  # Huge profit
            'unrealized_pnl': 0,
            'momentum_aligned': True,
            'position_held_time': 60
        }
        
        context = MomentumContext(pattern=PatternType.BREAKOUT)
        reward_info = reward_system.calculate(state, context)
        
        # Should be clipped to max
        assert reward_info['total_reward'] <= reward_system.max_reward
        assert reward_info['metadata']['was_clipped'] is True
        
        # Extreme loss
        state['realized_pnl'] = -50000
        reward_info = reward_system.calculate(state, context)
        
        # Should be clipped to min
        assert reward_info['total_reward'] >= reward_system.min_reward
        assert reward_info['metadata']['was_clipped'] is True
    
    def test_reward_normalization(self, reward_system):
        """Test reward normalization."""
        # Generate multiple rewards to update running stats
        states = [
            {'realized_pnl': 100, 'unrealized_pnl': 50},
            {'realized_pnl': 200, 'unrealized_pnl': 0},
            {'realized_pnl': -50, 'unrealized_pnl': -25},
            {'realized_pnl': 300, 'unrealized_pnl': 100},
            {'realized_pnl': -100, 'unrealized_pnl': 0},
        ]
        
        rewards = []
        for state in states:
            # Add required fields
            state.update({
                'momentum_aligned': True,
                'position_held_time': 120,
                'current_drawdown': 0.01,
                'trades_this_episode': 2
            })
            
            reward_info = reward_system.calculate(state, None)
            rewards.append(reward_info['total_reward'])
        
        # Check normalization stats updated
        assert hasattr(reward_system, 'running_mean')
        assert hasattr(reward_system, 'running_std')
        
        # Rewards should be roughly normalized
        rewards_array = np.array(rewards)
        # Not exactly 0 mean and 1 std due to running average
        assert abs(np.mean(rewards_array)) < 2.0
        assert 0.1 < np.std(rewards_array) < 3.0
    
    def test_pattern_specific_rewards(self, reward_system):
        """Test pattern-specific reward adjustments."""
        patterns_to_test = [
            (PatternType.BREAKOUT, PhaseType.FRONT_SIDE, 'buy', True),  # Good
            (PatternType.FLUSH, PhaseType.BACK_SIDE, 'sell', True),     # Good
            (PatternType.BOUNCE, PhaseType.RECOVERY, 'buy', True),       # Good
            (PatternType.CONSOLIDATION, PhaseType.NEUTRAL, 'hold', True), # Good
            (PatternType.BREAKOUT, PhaseType.FRONT_SIDE, 'sell', False), # Bad
        ]
        
        base_state = {
            'realized_pnl': 100,
            'unrealized_pnl': 50,
            'position_held_time': 120,
            'current_drawdown': 0.01,
            'trades_this_episode': 2
        }
        
        rewards = {}
        for pattern, phase, action, aligned in patterns_to_test:
            context = MomentumContext(
                pattern=pattern,
                phase=phase,
                quality_score=0.8
            )
            
            state = base_state.copy()
            state['action'] = action
            state['momentum_aligned'] = aligned
            
            reward_info = reward_system.calculate(state, context)
            rewards[(pattern, action, aligned)] = reward_info['total_reward']
        
        # Aligned actions should have higher rewards
        assert rewards[(PatternType.BREAKOUT, 'buy', True)] > rewards[(PatternType.BREAKOUT, 'sell', False)]
        
        # Different patterns should yield different rewards even with same P&L
        assert rewards[(PatternType.BREAKOUT, 'buy', True)] != rewards[(PatternType.BOUNCE, 'buy', True)]
    
    def test_episode_context_rewards(self, reward_system):
        """Test rewards considering full episode context."""
        # Early episode - exploration bonus
        early_state = {
            'episode_step': 10,
            'episode_duration': 60,  # 1 minute in
            'realized_pnl': 50,
            'unrealized_pnl': 0,
            'action': 'buy',
            'momentum_aligned': True,
            'exploration_bonus': 0.1
        }
        
        # Late episode - closure bonus
        late_state = {
            'episode_step': 500,
            'episode_duration': 3000,  # 50 minutes in
            'realized_pnl': 50,
            'unrealized_pnl': 200,  # Open position
            'action': 'sell',  # Closing
            'momentum_aligned': True,
            'closure_bonus': 0.2
        }
        
        early_reward = reward_system.calculate(early_state, None)
        late_reward = reward_system.calculate(late_state, None)
        
        # Different rewards for same P&L based on context
        assert early_reward['total_reward'] != late_reward['total_reward']
    
    def test_component_weight_adjustment(self, reward_system):
        """Test dynamic component weight adjustment."""
        # Adjust weights based on training phase
        original_weights = reward_system.component_weights.copy()
        
        # Early training - emphasize exploration
        reward_system.adjust_weights_for_phase('exploration')
        assert reward_system.component_weights['action_cost'] < original_weights['action_cost']
        
        # Late training - emphasize profit
        reward_system.adjust_weights_for_phase('exploitation')
        assert reward_system.component_weights['pnl'] > original_weights['pnl']
        
        # Reset
        reward_system.component_weights = original_weights
    
    def test_reward_shaping_functions(self, reward_system):
        """Test various reward shaping functions."""
        # Test non-linear P&L scaling
        small_pnl = reward_system._scale_pnl(100)
        medium_pnl = reward_system._scale_pnl(1000)
        large_pnl = reward_system._scale_pnl(10000)
        
        # Should have diminishing returns
        assert (medium_pnl / small_pnl) > (large_pnl / medium_pnl)
        
        # Test time decay function
        short_time = reward_system._time_decay(30, optimal=120)
        optimal_time = reward_system._time_decay(120, optimal=120)
        long_time = reward_system._time_decay(600, optimal=120)
        
        assert optimal_time > short_time
        assert optimal_time > long_time
        
        # Test risk penalty function
        low_risk = reward_system._risk_penalty(0.01)  # 1% drawdown
        medium_risk = reward_system._risk_penalty(0.03)  # 3% drawdown
        high_risk = reward_system._risk_penalty(0.05)  # 5% drawdown
        
        assert low_risk > medium_risk > high_risk
        assert high_risk < 0  # Penalty
    
    def test_reward_history_tracking(self, reward_system):
        """Test tracking of reward history for analysis."""
        # Generate several rewards
        for i in range(10):
            state = {
                'realized_pnl': np.random.randn() * 200,
                'unrealized_pnl': np.random.randn() * 100,
                'momentum_aligned': np.random.random() > 0.5,
                'position_held_time': np.random.randint(30, 300),
                'current_drawdown': abs(np.random.randn() * 0.01),
                'trades_this_episode': np.random.randint(1, 5)
            }
            
            reward_system.calculate(state, None)
        
        # Get history
        history = reward_system.get_reward_history()
        
        assert len(history) == 10
        assert 'timestamp' in history[0]
        assert 'total_reward' in history[0]
        assert 'components' in history[0]
        
        # Get statistics
        stats = reward_system.get_reward_statistics()
        
        assert 'mean_total_reward' in stats
        assert 'std_total_reward' in stats
        assert 'component_means' in stats
        assert 'component_contributions' in stats