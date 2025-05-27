"""Comprehensive test suite for RewardSystemV2.

This test suite focuses on:
- Component-based reward calculation
- Momentum-aware rewards
- Anti-exploitation mechanisms
- Reward shaping and normalization
- Component weight management
- Historical tracking and analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
from enum import Enum

from rewards.calculator import RewardSystemV2, RewardCalculator
from rewards.components import (
    RewardComponent,
    PnLComponent,
    MomentumAlignmentComponent,
    TimeEfficiencyComponent,
    RiskManagementComponent,
    ActionCostComponent,
    PatternSpecificComponent,
    DrawdownPenaltyComponent,
    PositionSizingComponent,
    LearningCurveComponent
)
from rewards.core import RewardContext, RewardResult
from envs.momentum_episode_manager import MomentumContext, PatternType, PhaseType


class TestRewardSystemV2Core:
    """Test core RewardSystemV2 functionality."""
    
    @pytest.fixture
    def reward_config(self):
        """Comprehensive reward configuration."""
        return {
            'version': 'v2',
            'components': {
                'pnl': {
                    'enabled': True,
                    'weight': 0.35,
                    'config': {
                        'realized_weight': 0.7,
                        'unrealized_weight': 0.3,
                        'scaling_factor': 0.001,
                        'use_log_scaling': True,
                        'clip_threshold': 10000
                    }
                },
                'momentum_alignment': {
                    'enabled': True,
                    'weight': 0.20,
                    'config': {
                        'alignment_bonus': 0.5,
                        'misalignment_penalty': -0.3,
                        'pattern_bonuses': {
                            'breakout': 0.25,
                            'flush': 0.20,
                            'bounce': 0.15,
                            'consolidation': -0.05
                        },
                        'phase_multipliers': {
                            'front_side': 1.2,
                            'back_side': 0.8,
                            'recovery': 1.0,
                            'neutral': 0.5
                        }
                    }
                },
                'time_efficiency': {
                    'enabled': True,
                    'weight': 0.15,
                    'config': {
                        'optimal_hold_time': 180,  # 3 minutes
                        'time_decay_rate': 0.001,
                        'quick_profit_bonus': 0.4,
                        'quick_profit_time': 30,  # 30 seconds
                        'extended_hold_penalty': -0.2,
                        'extended_hold_time': 600  # 10 minutes
                    }
                },
                'risk_management': {
                    'enabled': True,
                    'weight': 0.15,
                    'config': {
                        'drawdown_penalty_rate': -10.0,
                        'drawdown_threshold': 0.02,
                        'position_sizing_bonus': 0.3,
                        'overleveraging_penalty': -0.5,
                        'max_leverage': 2.0,
                        'var_penalty_threshold': 0.05  # 5% VaR
                    }
                },
                'action_cost': {
                    'enabled': True,
                    'weight': 0.10,
                    'config': {
                        'trade_cost': -0.05,
                        'excessive_trading_penalty': -0.5,
                        'trade_frequency_threshold': 0.5,  # trades per minute
                        'hold_bonus': 0.02,
                        'pattern_change_cost': -0.1
                    }
                },
                'learning_curve': {
                    'enabled': True,
                    'weight': 0.05,
                    'config': {
                        'exploration_bonus': 0.2,
                        'consistency_bonus': 0.3,
                        'improvement_bonus': 0.4,
                        'stagnation_penalty': -0.2
                    }
                }
            },
            'shaping': {
                'clipping': {
                    'enabled': True,
                    'min_reward': -2.0,
                    'max_reward': 2.0
                },
                'normalization': {
                    'enabled': True,
                    'method': 'running_zscore',
                    'window_size': 1000,
                    'alpha': 0.99
                },
                'smoothing': {
                    'enabled': True,
                    'method': 'exponential',
                    'alpha': 0.95
                }
            },
            'anti_exploitation': {
                'enabled': True,
                'repetition_penalty': -0.5,
                'repetition_window': 10,
                'diversity_bonus': 0.1,
                'min_action_entropy': 0.5
            }
        }
    
    @pytest.fixture
    def reward_system(self, reward_config):
        """Create RewardSystemV2 instance."""
        return RewardSystemV2(config=reward_config, logger=Mock())
    
    @pytest.fixture
    def base_state(self):
        """Base state for testing."""
        return {
            'timestamp': pd.Timestamp.now(),
            'symbol': 'MLGO',
            'action': 'buy',
            'position': None,
            'portfolio_value': 100000,
            'cash': 100000,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'daily_pnl': 0,
            'position_held_time': 0,
            'trades_this_episode': 0,
            'episode_duration': 0,
            'current_drawdown': 0,
            'max_drawdown': 0,
            'market_state': {
                'price': 10.0,
                'volume': 100000,
                'volatility': 0.02,
                'spread': 0.02
            }
        }
    
    @pytest.fixture
    def momentum_context(self):
        """Sample momentum context."""
        return MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.85,
            day_quality=0.90,
            intraday_move=0.12,
            volume_multiplier=4.0,
            time_of_day='market_open',
            strength_score=0.8,
            metadata={
                'breakout_level': 10.50,
                'support_level': 9.80,
                'volume_surge_time': pd.Timestamp.now()
            }
        )
    
    def test_basic_reward_calculation(self, reward_system, base_state, momentum_context):
        """Test basic reward calculation flow."""
        # Profitable trade aligned with momentum
        state = base_state.copy()
        state.update({
            'realized_pnl': 500,
            'unrealized_pnl': 200,
            'action': 'buy',
            'position_held_time': 120,
            'trades_this_episode': 2
        })
        
        result = reward_system.calculate_reward(state, momentum_context)
        
        assert isinstance(result, RewardResult)
        assert result.total_reward is not None
        assert len(result.component_rewards) > 0
        assert result.metadata is not None
        
        # Verify components calculated
        assert 'pnl' in result.component_rewards
        assert 'momentum_alignment' in result.component_rewards
        assert 'time_efficiency' in result.component_rewards
        assert 'risk_management' in result.component_rewards
        assert 'action_cost' in result.component_rewards
    
    def test_component_weight_application(self, reward_system, base_state, momentum_context):
        """Test proper application of component weights."""
        state = base_state.copy()
        state['realized_pnl'] = 1000
        
        result = reward_system.calculate_reward(state, momentum_context)
        
        # Calculate expected weighted sum
        expected_total = 0
        for name, value in result.component_rewards.items():
            weight = reward_system.config['components'][name]['weight']
            expected_total += value * weight
        
        # Account for any post-processing
        if reward_system.config['shaping']['clipping']['enabled']:
            expected_total = np.clip(
                expected_total,
                reward_system.config['shaping']['clipping']['min_reward'],
                reward_system.config['shaping']['clipping']['max_reward']
            )
        
        # Should match (within numerical precision)
        assert abs(result.total_reward - expected_total) < 0.01
    
    def test_reward_clipping(self, reward_system, base_state, momentum_context):
        """Test reward clipping to prevent extreme values."""
        # Extreme profit
        state = base_state.copy()
        state['realized_pnl'] = 50000  # Huge profit
        
        result = reward_system.calculate_reward(state, momentum_context)
        
        assert result.total_reward <= reward_system.config['shaping']['clipping']['max_reward']
        assert result.metadata['was_clipped'] is True
        assert result.metadata['unclipped_reward'] > result.total_reward
        
        # Extreme loss
        state['realized_pnl'] = -50000
        state['unrealized_pnl'] = -10000
        
        result = reward_system.calculate_reward(state, momentum_context)
        
        assert result.total_reward >= reward_system.config['shaping']['clipping']['min_reward']
        assert result.metadata['was_clipped'] is True
    
    def test_reward_normalization(self, reward_system, base_state, momentum_context):
        """Test running normalization of rewards."""
        # Generate sequence of rewards
        rewards = []
        for i in range(100):
            state = base_state.copy()
            state['realized_pnl'] = np.random.normal(100, 50)
            state['unrealized_pnl'] = np.random.normal(0, 25)
            state['trades_this_episode'] = np.random.randint(1, 5)
            
            result = reward_system.calculate_reward(state, momentum_context)
            rewards.append(result.total_reward)
        
        # Later rewards should be roughly normalized
        recent_rewards = rewards[-20:]
        mean_recent = np.mean(recent_rewards)
        std_recent = np.std(recent_rewards)
        
        # Should be roughly normalized (not exact due to running stats)
        assert abs(mean_recent) < 0.5  # Close to 0
        assert 0.5 < std_recent < 1.5   # Close to 1
    
    def test_anti_exploitation_mechanisms(self, reward_system, base_state, momentum_context):
        """Test anti-exploitation features."""
        # Repetitive action pattern
        state = base_state.copy()
        action_history = ['buy', 'sell', 'buy', 'sell', 'buy', 'sell']
        
        for i, action in enumerate(action_history):
            state['action'] = action
            state['action_history'] = action_history[:i+1]
            state['realized_pnl'] = 50  # Small consistent profit
            
            result = reward_system.calculate_reward(state, momentum_context)
            
            # Should get repetition penalty after pattern detected
            if i >= 4:  # Pattern established
                assert result.metadata.get('repetition_penalty_applied', False)
                assert result.component_rewards.get('anti_exploitation', 0) < 0
    
    def test_component_enabling_disabling(self, reward_system, base_state, momentum_context):
        """Test enabling/disabling individual components."""
        # Disable time efficiency
        reward_system.config['components']['time_efficiency']['enabled'] = False
        
        state = base_state.copy()
        state['position_held_time'] = 1000  # Should trigger penalty if enabled
        
        result = reward_system.calculate_reward(state, momentum_context)
        
        assert 'time_efficiency' not in result.component_rewards
        
        # Re-enable
        reward_system.config['components']['time_efficiency']['enabled'] = True
        result2 = reward_system.calculate_reward(state, momentum_context)
        
        assert 'time_efficiency' in result2.component_rewards


class TestPnLComponent:
    """Test P&L reward component in detail."""
    
    @pytest.fixture
    def pnl_component(self):
        """Create PnL component."""
        config = {
            'realized_weight': 0.7,
            'unrealized_weight': 0.3,
            'scaling_factor': 0.001,
            'use_log_scaling': True,
            'clip_threshold': 10000,
            'normalize_by_portfolio': True
        }
        return PnLComponent(config=config)
    
    def test_basic_pnl_calculation(self, pnl_component):
        """Test basic P&L reward calculation."""
        state = {
            'realized_pnl': 500,
            'unrealized_pnl': 200,
            'portfolio_value': 100000
        }
        
        reward = pnl_component.calculate(state, None)
        
        # Weighted combination
        weighted_pnl = 500 * 0.7 + 200 * 0.3  # 350 + 60 = 410
        scaled_pnl = weighted_pnl * 0.001  # 0.41
        
        # Log scaling applied
        if weighted_pnl > 0:
            expected = np.sign(weighted_pnl) * np.log1p(abs(scaled_pnl))
        else:
            expected = -np.log1p(abs(scaled_pnl))
        
        assert abs(reward - expected) < 0.01
    
    def test_pnl_normalization(self, pnl_component):
        """Test P&L normalization by portfolio value."""
        # Same absolute P&L, different portfolio sizes
        state1 = {
            'realized_pnl': 1000,
            'unrealized_pnl': 0,
            'portfolio_value': 100000
        }
        
        state2 = {
            'realized_pnl': 1000,
            'unrealized_pnl': 0,
            'portfolio_value': 50000
        }
        
        reward1 = pnl_component.calculate(state1, None)
        reward2 = pnl_component.calculate(state2, None)
        
        # Smaller portfolio should get higher reward for same P&L
        assert reward2 > reward1
    
    def test_pnl_clipping(self, pnl_component):
        """Test P&L clipping for extreme values."""
        state = {
            'realized_pnl': 50000,  # Extreme
            'unrealized_pnl': 0,
            'portfolio_value': 100000
        }
        
        reward = pnl_component.calculate(state, None)
        
        # Should be clipped
        state_normal = state.copy()
        state_normal['realized_pnl'] = 5000
        reward_normal = pnl_component.calculate(state_normal, None)
        
        # Clipped reward should not be 10x despite 10x P&L
        assert reward < reward_normal * 10
    
    def test_loss_handling(self, pnl_component):
        """Test handling of losses."""
        state = {
            'realized_pnl': -500,
            'unrealized_pnl': -200,
            'portfolio_value': 100000
        }
        
        reward = pnl_component.calculate(state, None)
        
        assert reward < 0
        
        # Test asymmetry - losses might be penalized more
        state_profit = state.copy()
        state_profit['realized_pnl'] = 500
        state_profit['unrealized_pnl'] = 200
        
        reward_profit = pnl_component.calculate(state_profit, None)
        
        # Check if loss penalty is stronger (optional feature)
        if hasattr(pnl_component, 'loss_aversion_factor'):
            assert abs(reward) > reward_profit


class TestMomentumAlignmentComponent:
    """Test momentum alignment reward component."""
    
    @pytest.fixture
    def alignment_component(self):
        """Create momentum alignment component."""
        config = {
            'alignment_bonus': 0.5,
            'misalignment_penalty': -0.3,
            'pattern_bonuses': {
                'breakout': 0.25,
                'flush': 0.20,
                'bounce': 0.15,
                'consolidation': -0.05
            },
            'phase_multipliers': {
                'front_side': 1.2,
                'back_side': 0.8,
                'recovery': 1.0,
                'neutral': 0.5
            },
            'quality_weight': 0.3
        }
        return MomentumAlignmentComponent(config=config)
    
    def test_alignment_calculation(self, alignment_component):
        """Test basic alignment reward calculation."""
        # Aligned buy during breakout
        state = {
            'action': 'buy',
            'momentum_direction': 'up',
            'position': None
        }
        
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9
        )
        
        reward = alignment_component.calculate(state, context)
        
        # Base alignment + pattern bonus + phase multiplier
        expected = 0.5 + 0.25  # alignment + breakout
        expected *= 1.2  # front_side multiplier
        expected *= (1 + 0.9 * 0.3)  # quality weight
        
        assert abs(reward - expected) < 0.01
    
    def test_misalignment_penalty(self, alignment_component):
        """Test misalignment penalties."""
        # Selling during breakout (misaligned)
        state = {
            'action': 'sell',
            'momentum_direction': 'up',
            'position': Mock(side='long', quantity=1000)
        }
        
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.9
        )
        
        reward = alignment_component.calculate(state, context)
        
        assert reward < 0  # Should be penalized
        assert reward <= -0.3  # At least base penalty
    
    def test_pattern_specific_rewards(self, alignment_component):
        """Test pattern-specific reward adjustments."""
        patterns = [
            (PatternType.BREAKOUT, 'buy', True),
            (PatternType.FLUSH, 'sell', True),
            (PatternType.BOUNCE, 'buy', True),
            (PatternType.CONSOLIDATION, 'hold', True),
        ]
        
        base_state = {'position': None}
        
        rewards = {}
        for pattern, action, aligned in patterns:
            state = base_state.copy()
            state['action'] = action
            state['momentum_aligned'] = aligned
            
            context = MomentumContext(
                pattern=pattern,
                phase=PhaseType.NEUTRAL,
                quality_score=0.8
            )
            
            reward = alignment_component.calculate(state, context)
            rewards[pattern] = reward
        
        # Breakout should reward buying most
        assert rewards[PatternType.BREAKOUT] > rewards[PatternType.BOUNCE]
        
        # Consolidation should have smallest reward
        assert rewards[PatternType.CONSOLIDATION] < rewards[PatternType.BREAKOUT]
    
    def test_phase_impact(self, alignment_component):
        """Test momentum phase impact on rewards."""
        phases = [
            PhaseType.FRONT_SIDE,
            PhaseType.BACK_SIDE,
            PhaseType.RECOVERY,
            PhaseType.NEUTRAL
        ]
        
        state = {
            'action': 'buy',
            'momentum_aligned': True,
            'position': None
        }
        
        rewards = {}
        for phase in phases:
            context = MomentumContext(
                pattern=PatternType.BREAKOUT,
                phase=phase,
                quality_score=0.8
            )
            
            reward = alignment_component.calculate(state, context)
            rewards[phase] = reward
        
        # Front side should give highest reward
        assert rewards[PhaseType.FRONT_SIDE] > rewards[PhaseType.BACK_SIDE]
        assert rewards[PhaseType.FRONT_SIDE] > rewards[PhaseType.NEUTRAL]
    
    def test_quality_score_weighting(self, alignment_component):
        """Test quality score impact on rewards."""
        state = {
            'action': 'buy',
            'momentum_aligned': True,
            'position': None
        }
        
        # High quality momentum
        context_high = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.95
        )
        
        # Low quality momentum
        context_low = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.60
        )
        
        reward_high = alignment_component.calculate(state, context_high)
        reward_low = alignment_component.calculate(state, context_low)
        
        # Higher quality should give higher reward
        assert reward_high > reward_low


class TestTimeEfficiencyComponent:
    """Test time efficiency reward component."""
    
    @pytest.fixture
    def time_component(self):
        """Create time efficiency component."""
        config = {
            'optimal_hold_time': 180,  # 3 minutes
            'time_decay_rate': 0.001,
            'quick_profit_bonus': 0.4,
            'quick_profit_time': 30,
            'extended_hold_penalty': -0.2,
            'extended_hold_time': 600,
            'momentum_time_adjustment': True
        }
        return TimeEfficiencyComponent(config=config)
    
    def test_quick_profit_bonus(self, time_component):
        """Test quick profit bonus."""
        state = {
            'action': 'sell',  # Closing position
            'position_held_time': 25,
            'realized_pnl': 300,
            'position': Mock(entry_time=pd.Timestamp.now() - pd.Timedelta(seconds=25))
        }
        
        reward = time_component.calculate(state, None)
        
        assert reward > 0.4  # Should get quick profit bonus
    
    def test_optimal_hold_reward(self, time_component):
        """Test reward at optimal holding time."""
        state = {
            'action': 'sell',
            'position_held_time': 180,  # Optimal
            'realized_pnl': 300,
            'position': Mock()
        }
        
        reward = time_component.calculate(state, None)
        
        assert reward > 0  # Positive but no quick bonus
        assert reward < 0.4  # Less than quick profit bonus
    
    def test_extended_hold_penalty(self, time_component):
        """Test penalty for holding too long."""
        state = {
            'action': 'hold',
            'position_held_time': 900,  # 15 minutes
            'unrealized_pnl': 50,
            'position': Mock()
        }
        
        reward = time_component.calculate(state, None)
        
        assert reward < 0  # Should be penalized
    
    def test_momentum_adjusted_timing(self, time_component):
        """Test momentum-based time adjustments."""
        state = {
            'action': 'sell',
            'position_held_time': 300,  # 5 minutes
            'realized_pnl': 500,
            'position': Mock()
        }
        
        # Strong momentum - longer holds acceptable
        context_strong = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            strength_score=0.9
        )
        
        # Weak momentum - should exit quicker
        context_weak = MomentumContext(
            pattern=PatternType.CONSOLIDATION,
            phase=PhaseType.NEUTRAL,
            strength_score=0.3
        )
        
        reward_strong = time_component.calculate(state, context_strong)
        reward_weak = time_component.calculate(state, context_weak)
        
        # Strong momentum should tolerate longer holds better
        assert reward_strong > reward_weak
    
    def test_no_position_handling(self, time_component):
        """Test handling when no position held."""
        state = {
            'action': 'hold',
            'position_held_time': 0,
            'position': None
        }
        
        reward = time_component.calculate(state, None)
        
        assert reward == 0  # No reward/penalty without position


class TestRiskManagementComponent:
    """Test risk management reward component."""
    
    @pytest.fixture
    def risk_component(self):
        """Create risk management component."""
        config = {
            'drawdown_penalty_rate': -10.0,
            'drawdown_threshold': 0.02,
            'position_sizing_bonus': 0.3,
            'overleveraging_penalty': -0.5,
            'max_leverage': 2.0,
            'var_penalty_threshold': 0.05,
            'stop_loss_bonus': 0.2,
            'risk_reward_ratio_bonus': 0.3
        }
        return RiskManagementComponent(config=config)
    
    def test_drawdown_penalty(self, risk_component):
        """Test drawdown penalties."""
        # Below threshold - no penalty
        state = {
            'current_drawdown': 0.015,  # 1.5%
            'max_drawdown': 0.015,
            'portfolio_value': 98500
        }
        
        reward = risk_component.calculate(state, None)
        assert reward >= 0  # No penalty
        
        # Above threshold - penalty
        state['current_drawdown'] = 0.035  # 3.5%
        reward = risk_component.calculate(state, None)
        
        assert reward < 0
        expected_penalty = -10.0 * (0.035 - 0.02)  # Rate * excess
        assert abs(reward - expected_penalty) < 0.1
    
    def test_position_sizing_rewards(self, risk_component):
        """Test position sizing rewards."""
        # Conservative sizing
        state = {
            'action': 'buy',
            'position_size_fraction': 0.25,  # 25% of capital
            'current_drawdown': 0.01,
            'portfolio_value': 100000,
            'leverage_used': 0.25
        }
        
        reward = risk_component.calculate(state, None)
        assert reward > 0  # Should get sizing bonus
        
        # Aggressive sizing
        state['position_size_fraction'] = 1.0
        state['leverage_used'] = 2.0
        
        reward = risk_component.calculate(state, None)
        assert reward < 0  # Should get overleveraging penalty
    
    def test_stop_loss_rewards(self, risk_component):
        """Test rewards for using stop losses."""
        state = {
            'action': 'sell',
            'exit_reason': 'stop_loss',
            'realized_pnl': -100,  # Small loss
            'max_adverse_excursion': -500,  # Could have been worse
            'current_drawdown': 0.01
        }
        
        reward = risk_component.calculate(state, None)
        
        # Should get bonus for cutting losses
        assert reward > 0
    
    def test_risk_reward_ratio(self, risk_component):
        """Test risk/reward ratio considerations."""
        state = {
            'action': 'buy',
            'entry_setup': {
                'potential_profit': 500,
                'potential_loss': 100,
                'risk_reward_ratio': 5.0
            },
            'current_drawdown': 0.01
        }
        
        reward = risk_component.calculate(state, None)
        
        # Good risk/reward should be rewarded
        assert reward > 0
        
        # Poor risk/reward
        state['entry_setup']['risk_reward_ratio'] = 0.5
        reward_poor = risk_component.calculate(state, None)
        
        assert reward_poor < reward
    
    def test_portfolio_heat_management(self, risk_component):
        """Test portfolio heat (total risk) management."""
        state = {
            'portfolio_heat': 0.15,  # 15% at risk
            'max_portfolio_heat': 0.20,
            'current_drawdown': 0.01,
            'action': 'hold'
        }
        
        reward = risk_component.calculate(state, None)
        
        # Approaching heat limit but not over
        assert reward >= 0
        
        # Exceeding heat limit
        state['portfolio_heat'] = 0.25
        reward = risk_component.calculate(state, None)
        
        assert reward < 0  # Should be penalized


class TestActionCostComponent:
    """Test action cost component."""
    
    @pytest.fixture
    def action_cost_component(self):
        """Create action cost component."""
        config = {
            'trade_cost': -0.05,
            'excessive_trading_penalty': -0.5,
            'trade_frequency_threshold': 0.5,
            'hold_bonus': 0.02,
            'pattern_change_cost': -0.1,
            'whipsaw_penalty': -0.3
        }
        return ActionCostComponent(config=config)
    
    def test_basic_trade_costs(self, action_cost_component):
        """Test basic trading costs."""
        # Buy action
        state = {
            'action': 'buy',
            'trades_this_episode': 5,
            'episode_duration': 600  # 10 minutes
        }
        
        reward = action_cost_component.calculate(state, None)
        assert reward < 0  # Should have cost
        assert reward >= -0.05  # Just base cost
        
        # Hold action
        state['action'] = 'hold'
        reward = action_cost_component.calculate(state, None)
        assert reward == 0.02  # Hold bonus
    
    def test_excessive_trading_penalty(self, action_cost_component):
        """Test penalty for overtrading."""
        state = {
            'action': 'buy',
            'trades_this_episode': 20,
            'episode_duration': 600,  # 20 trades in 10 min = 2/min
            'trade_frequency': 2.0
        }
        
        reward = action_cost_component.calculate(state, None)
        
        # Should get excessive trading penalty
        assert reward < -0.05  # More than just trade cost
        assert reward <= -0.5  # Full penalty
    
    def test_pattern_change_costs(self, action_cost_component):
        """Test costs for changing patterns."""
        state = {
            'action': 'sell',
            'position': Mock(side='long'),
            'recent_actions': ['buy', 'buy', 'hold', 'sell'],  # Pattern change
            'trades_this_episode': 3,
            'episode_duration': 300
        }
        
        reward = action_cost_component.calculate(state, None)
        
        # Should include pattern change cost
        assert reward < -0.05
    
    def test_whipsaw_detection(self, action_cost_component):
        """Test whipsaw pattern detection and penalty."""
        state = {
            'action': 'buy',
            'recent_actions': ['buy', 'sell', 'buy', 'sell', 'buy'],
            'recent_pnls': [-50, -50, -50, -50],  # Losing on each reversal
            'trades_this_episode': 5,
            'episode_duration': 120  # Rapid reversals
        }
        
        reward = action_cost_component.calculate(state, None)
        
        # Should get whipsaw penalty
        assert reward < -0.3


class TestAdvancedRewardFeatures:
    """Test advanced reward system features."""
    
    @pytest.fixture
    def advanced_reward_system(self):
        """Create reward system with advanced features."""
        config = {
            'version': 'v2',
            'components': {
                'pnl': {'enabled': True, 'weight': 0.4},
                'momentum_alignment': {'enabled': True, 'weight': 0.3},
                'pattern_specific': {'enabled': True, 'weight': 0.2},
                'learning_curve': {'enabled': True, 'weight': 0.1}
            },
            'meta_learning': {
                'enabled': True,
                'adapt_weights': True,
                'performance_window': 100,
                'adaptation_rate': 0.01
            },
            'contextual_adjustment': {
                'market_regime': True,
                'volatility_scaling': True,
                'time_of_day': True
            }
        }
        return RewardSystemV2(config=config)
    
    def test_meta_learning_weight_adaptation(self, advanced_reward_system):
        """Test meta-learning weight adaptation."""
        # Track performance with different components
        performance_data = {
            'pnl': {'accuracy': 0.7, 'correlation': 0.8},
            'momentum_alignment': {'accuracy': 0.6, 'correlation': 0.5},
            'pattern_specific': {'accuracy': 0.8, 'correlation': 0.9},
            'learning_curve': {'accuracy': 0.5, 'correlation': 0.3}
        }
        
        # Adapt weights based on performance
        advanced_reward_system.adapt_weights(performance_data)
        
        # Better performing components should get higher weights
        weights = advanced_reward_system.get_component_weights()
        assert weights['pattern_specific'] > weights['learning_curve']
    
    def test_market_regime_adjustment(self, advanced_reward_system):
        """Test market regime-based reward adjustments."""
        base_state = {
            'realized_pnl': 100,
            'action': 'buy',
            'position': None
        }
        
        # Bull market context
        bull_context = Mock(
            market_regime='bull',
            trend_strength=0.8,
            volatility=0.015
        )
        
        # Bear market context
        bear_context = Mock(
            market_regime='bear',
            trend_strength=-0.7,
            volatility=0.025
        )
        
        reward_bull = advanced_reward_system.calculate_reward(base_state, bull_context)
        reward_bear = advanced_reward_system.calculate_reward(base_state, bear_context)
        
        # Same action might be rewarded differently in different regimes
        assert reward_bull.total_reward != reward_bear.total_reward
    
    def test_volatility_scaling(self, advanced_reward_system):
        """Test volatility-based reward scaling."""
        state = {
            'realized_pnl': 200,
            'position_held_time': 60,
            'action': 'sell'
        }
        
        # Low volatility
        context_low_vol = Mock(
            volatility=0.01,
            volatility_percentile=0.2
        )
        
        # High volatility
        context_high_vol = Mock(
            volatility=0.04,
            volatility_percentile=0.9
        )
        
        reward_low = advanced_reward_system.calculate_reward(state, context_low_vol)
        reward_high = advanced_reward_system.calculate_reward(state, context_high_vol)
        
        # Same P&L in high volatility worth less
        assert reward_low.total_reward > reward_high.total_reward
    
    def test_time_of_day_adjustments(self, advanced_reward_system):
        """Test time-of-day reward adjustments."""
        state = {
            'realized_pnl': 150,
            'action': 'buy',
            'position': None
        }
        
        # Market open - higher volatility expected
        context_open = Mock(
            time_of_day='market_open',
            minutes_since_open=5
        )
        
        # Mid-day - typically calmer
        context_midday = Mock(
            time_of_day='midday',
            minutes_since_open=180
        )
        
        reward_open = advanced_reward_system.calculate_reward(state, context_open)
        reward_midday = advanced_reward_system.calculate_reward(state, context_midday)
        
        # Rewards might be scaled differently
        assert reward_open.metadata['time_adjustment'] != reward_midday.metadata['time_adjustment']
    
    def test_reward_history_analysis(self, advanced_reward_system):
        """Test reward history tracking and analysis."""
        # Generate reward history
        for i in range(50):
            state = {
                'realized_pnl': np.random.normal(100, 50),
                'action': np.random.choice(['buy', 'sell', 'hold']),
                'position_held_time': np.random.randint(30, 300)
            }
            context = Mock(pattern=PatternType.BREAKOUT)
            
            advanced_reward_system.calculate_reward(state, context)
        
        # Analyze history
        analysis = advanced_reward_system.analyze_reward_history()
        
        assert 'mean_reward' in analysis
        assert 'component_contributions' in analysis
        assert 'reward_volatility' in analysis
        assert 'component_correlations' in analysis
        
        # Component contribution should sum to ~100%
        total_contribution = sum(analysis['component_contributions'].values())
        assert 0.95 < total_contribution < 1.05
    
    def test_reward_explanation(self, advanced_reward_system):
        """Test reward explanation generation."""
        state = {
            'realized_pnl': 300,
            'unrealized_pnl': 100,
            'action': 'buy',
            'position_held_time': 45,
            'momentum_aligned': True
        }
        
        context = MomentumContext(
            pattern=PatternType.BREAKOUT,
            phase=PhaseType.FRONT_SIDE,
            quality_score=0.85
        )
        
        result = advanced_reward_system.calculate_reward(state, context)
        explanation = advanced_reward_system.explain_reward(result)
        
        assert 'total_reward' in explanation
        assert 'main_factors' in explanation
        assert 'component_breakdown' in explanation
        assert len(explanation['main_factors']) > 0
        
        # Should identify P&L as major factor given the profit
        assert any('pnl' in factor.lower() for factor in explanation['main_factors'])