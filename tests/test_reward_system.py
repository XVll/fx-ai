# tests/test_reward_system.py - Comprehensive tests for the reward system

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from rewards.calculator import RewardSystem, TradeTracker
from rewards.core import RewardComponent, RewardAggregator, RewardState, RewardType, RewardMetadata
from rewards.components import (
    RealizedPnLReward, MarkToMarketReward, DifferentialSharpeReward,
    HoldingTimePenalty, OvertradingPenalty, QuickProfitIncentive,
    DrawdownPenalty, MAEPenalty, MFEPenalty, TerminalPenalty
)
from rewards.metrics import RewardMetricsTracker, ComponentMetrics
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum
from config.schemas import RewardConfig


@dataclass
class MockRewardConfig:
    """Mock reward config for testing"""
    scale_factor: float = 1.0
    clip_range: List[float] = None
    
    def __post_init__(self):
        if self.clip_range is None:
            self.clip_range = [-10.0, 10.0]
    
    # Component configs
    pnl = Mock(enabled=True, coefficient=1.0)
    holding_penalty = Mock(enabled=True, coefficient=1.0)
    action_penalty = Mock(enabled=True, coefficient=1.0)
    drawdown_penalty = Mock(enabled=True, coefficient=1.0)
    bankruptcy_penalty = Mock(enabled=True, coefficient=1.0)
    profitable_exit = Mock(enabled=True, coefficient=1.0)
    spread_penalty = Mock(enabled=False, coefficient=0.0)
    quick_profit = Mock(enabled=False, coefficient=0.0)
    invalid_action_penalty = Mock(enabled=False, coefficient=0.0)


class TestTradeTracker:
    """Test TradeTracker functionality"""
    
    def test_initialization(self):
        """Test TradeTracker initialization"""
        tracker = TradeTracker(entry_price=100.0, entry_step=10)
        assert tracker.entry_price == 100.0
        assert tracker.entry_step == 10
        assert tracker.max_unrealized_pnl == 0.0
        assert tracker.min_unrealized_pnl == 0.0
    
    def test_update_positive_pnl(self):
        """Test updating with positive unrealized PnL"""
        tracker = TradeTracker(entry_price=100.0, entry_step=10)
        tracker.update(5.0)
        assert tracker.max_unrealized_pnl == 5.0
        assert tracker.min_unrealized_pnl == 0.0
        
        tracker.update(8.0)
        assert tracker.max_unrealized_pnl == 8.0
        assert tracker.min_unrealized_pnl == 0.0
    
    def test_update_negative_pnl(self):
        """Test updating with negative unrealized PnL"""
        tracker = TradeTracker(entry_price=100.0, entry_step=10)
        tracker.update(-3.0)
        assert tracker.max_unrealized_pnl == 0.0
        assert tracker.min_unrealized_pnl == -3.0
        
        tracker.update(-5.0)
        assert tracker.max_unrealized_pnl == 0.0
        assert tracker.min_unrealized_pnl == -5.0
    
    def test_update_mixed_pnl(self):
        """Test updating with mixed positive/negative PnL"""
        tracker = TradeTracker(entry_price=100.0, entry_step=10)
        tracker.update(5.0)
        tracker.update(-2.0)
        tracker.update(10.0)
        tracker.update(-7.0)
        
        assert tracker.max_unrealized_pnl == 10.0
        assert tracker.min_unrealized_pnl == -7.0


class TestRewardComponents:
    """Test individual reward components"""
    
    def setup_method(self):
        """Setup common test data"""
        self.config = {'enabled': True, 'weight': 1.0}
        self.logger = Mock()
        
        # Mock portfolio states
        self.portfolio_before = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
        
        self.portfolio_after = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
        
        self.portfolio_next = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
        
        self.fill_details = []
        self.decoded_action = {'type': Mock(name='HOLD')}
        
        self.state = RewardState(
            portfolio_before=self.portfolio_before,
            portfolio_after_fills=self.portfolio_after,
            portfolio_next=self.portfolio_next,
            market_state_current={},
            market_state_next={},
            decoded_action=self.decoded_action,
            fill_details=self.fill_details,
            terminated=False,
            truncated=False,
            step_count=0
        )
    
    def test_realized_pnl_reward_positive(self):
        """Test RealizedPnLReward with positive PnL"""
        component = RealizedPnLReward(self.config, self.logger)
        
        # Setup profitable trade
        self.portfolio_after['realized_pnl_session'] = 100.0
        fill = Mock()
        fill.commission = 1.0
        fill.slippage_cost_total = 0.5
        fill.closes_position = True
        self.state.fill_details = [fill]
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 98.5  # 100 - 1.0 - 0.5
        assert diagnostics['gross_pnl'] == 100.0
        assert diagnostics['costs'] == 1.5
        assert diagnostics['trades_closed'] == 1
    
    def test_realized_pnl_reward_negative(self):
        """Test RealizedPnLReward with losing trade"""
        component = RealizedPnLReward(self.config, self.logger)
        
        # Setup losing trade
        self.portfolio_before['realized_pnl_session'] = 0.0
        self.portfolio_after['realized_pnl_session'] = -50.0
        fill = Mock()
        fill.commission = 1.0
        fill.slippage_cost_total = 0.5
        fill.closes_position = True
        self.state.fill_details = [fill]
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == -51.5  # -50 - 1.0 - 0.5
        assert diagnostics['net_pnl'] == -51.5
    
    def test_realized_pnl_reward_no_trade(self):
        """Test RealizedPnLReward with no trades"""
        component = RealizedPnLReward(self.config, self.logger)
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['trades_closed'] == 0
    
    def test_mark_to_market_reward(self):
        """Test MarkToMarketReward"""
        component = MarkToMarketReward(self.config, self.logger)
        
        # Setup unrealized PnL change
        self.portfolio_before['unrealized_pnl'] = 10.0
        self.portfolio_next['unrealized_pnl'] = 25.0
        self.portfolio_next['position_value'] = 50000.0
        self.portfolio_next['total_equity'] = 100000.0
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 15.0  # 25 - 10
        assert diagnostics['mtm_change'] == 15.0
        assert diagnostics['leverage'] == 0.5  # 50000 / 100000
    
    def test_mark_to_market_leverage_penalty(self):
        """Test MarkToMarketReward with over-leverage penalty"""
        config = {**self.config, 'max_leverage': 1.0}
        component = MarkToMarketReward(config, self.logger)
        
        # Setup over-leveraged position
        self.portfolio_before['unrealized_pnl'] = 0.0
        self.portfolio_next['unrealized_pnl'] = 20.0
        self.portfolio_next['position_value'] = 200000.0  # 2x leverage
        self.portfolio_next['total_equity'] = 100000.0
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 10.0  # 20 * (1.0 / 2.0) = 10
        assert diagnostics['leverage'] == 2.0
        assert diagnostics['leverage_penalty'] == 0.5
    
    def test_holding_time_penalty_no_position(self):
        """Test HoldingTimePenalty with no position"""
        component = HoldingTimePenalty(self.config, self.logger)
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['has_position'] == False
    
    def test_holding_time_penalty_within_limit(self):
        """Test HoldingTimePenalty within time limit"""
        component = HoldingTimePenalty(self.config, self.logger)
        
        # Setup position within limit
        self.portfolio_next['position_side'] = PositionSideEnum.LONG
        self.state.current_trade_duration = 30  # Below default max of 60
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['holding_time'] == 30
        assert diagnostics['excess_time'] == 0
    
    def test_holding_time_penalty_exceeded(self):
        """Test HoldingTimePenalty when limit exceeded"""
        config = {**self.config, 'max_holding_time': 60, 'penalty_per_step': 0.001, 'progressive_penalty': False}
        component = HoldingTimePenalty(config, self.logger)
        
        # Setup position exceeding limit
        self.portfolio_next['position_side'] = PositionSideEnum.LONG
        self.state.current_trade_duration = 90  # 30 steps over limit
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == -0.03  # -0.001 * 30
        assert diagnostics['excess_time'] == 30
    
    def test_holding_time_penalty_progressive(self):
        """Test HoldingTimePenalty with progressive penalty"""
        config = {**self.config, 'max_holding_time': 60, 'penalty_per_step': 0.001, 'progressive_penalty': True}
        component = HoldingTimePenalty(config, self.logger)
        
        # Setup position way over limit
        self.portfolio_next['position_side'] = PositionSideEnum.LONG
        self.state.current_trade_duration = 120  # 60 steps over limit
        
        reward, diagnostics = component.calculate(self.state)
        
        # Base penalty: -0.001 * 60 = -0.06
        # Progressive multiplier: (1 + 60/60) = 2.0
        # Final: -0.06 * 2.0 = -0.12
        assert reward == -0.12
    
    def test_overtrading_penalty_no_trade(self):
        """Test OvertradingPenalty with no trade"""
        component = OvertradingPenalty(self.config, self.logger)
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['trade_executed'] == False
    
    def test_overtrading_penalty_within_limit(self):
        """Test OvertradingPenalty within limit"""
        config = {**self.config, 'frequency_window': 100, 'max_trades_per_window': 5}
        component = OvertradingPenalty(config, self.logger)
        
        # Execute some trades within limit
        self.state.fill_details = [Mock()]
        self.state.step_count = 50
        
        # First trade - should be fine
        reward, diagnostics = component.calculate(self.state)
        assert reward == 0.0
        
        # Execute more trades
        for i in range(4):  # Total 5 trades
            self.state.step_count += 1
            reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0  # Still within limit
    
    def test_overtrading_penalty_exceeded(self):
        """Test OvertradingPenalty when limit exceeded"""
        config = {
            **self.config, 
            'frequency_window': 100, 
            'max_trades_per_window': 3,
            'penalty_per_excess_trade': 0.01,
            'exponential_penalty': False
        }
        component = OvertradingPenalty(config, self.logger)
        
        # Execute trades exceeding limit
        self.state.fill_details = [Mock()]
        for i in range(5):  # 2 trades over limit of 3
            self.state.step_count = i
            reward, diagnostics = component.calculate(self.state)
        
        # Last call should have penalty for 2 excess trades
        assert reward == -0.02  # -0.01 * 2
    
    def test_quick_profit_incentive_no_close(self):
        """Test QuickProfitIncentive with no position close"""
        component = QuickProfitIncentive(self.config, self.logger)
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['trades_closed'] == 0
    
    def test_quick_profit_incentive_profitable_quick(self):
        """Test QuickProfitIncentive with quick profitable close"""
        config = {**self.config, 'quick_profit_time': 30, 'bonus_rate': 0.5}
        component = QuickProfitIncentive(config, self.logger)
        
        # Setup quick profitable trade
        fill = Mock()
        fill.closes_position = True
        fill.realized_pnl = 100.0
        self.state.fill_details = [fill]
        self.state.current_trade_duration = 15  # Quick trade
        
        reward, diagnostics = component.calculate(self.state)
        
        # Time factor: 1.0 - (15/30) = 0.5
        # Incentive: 100 * 0.5 * 0.5 = 25.0
        assert reward == 25.0
        assert diagnostics['profitable_closes'] == 1
    
    def test_quick_profit_incentive_profitable_slow(self):
        """Test QuickProfitIncentive with slow profitable close"""
        config = {**self.config, 'quick_profit_time': 30, 'bonus_rate': 0.5}
        component = QuickProfitIncentive(config, self.logger)
        
        # Setup slow profitable trade
        fill = Mock()
        fill.closes_position = True
        fill.realized_pnl = 100.0
        self.state.fill_details = [fill]
        self.state.current_trade_duration = 40  # Too slow
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0  # No bonus for slow trades
    
    def test_quick_profit_incentive_losing_trade(self):
        """Test QuickProfitIncentive with losing trade"""
        component = QuickProfitIncentive(self.config, self.logger)
        
        # Setup losing trade
        fill = Mock()
        fill.closes_position = True
        fill.realized_pnl = -50.0
        self.state.fill_details = [fill]
        self.state.current_trade_duration = 10
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0  # No bonus for losing trades
        assert diagnostics['profitable_closes'] == 0
    
    def test_drawdown_penalty_no_drawdown(self):
        """Test DrawdownPenalty with no drawdown"""
        component = DrawdownPenalty(self.config, self.logger)
        
        self.portfolio_next['unrealized_pnl'] = 50.0  # Positive
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['drawdown_pct'] == 0.0
    
    def test_drawdown_penalty_small_drawdown(self):
        """Test DrawdownPenalty with small drawdown"""
        config = {**self.config, 'warning_threshold': 0.02, 'base_penalty': 0.01}
        component = DrawdownPenalty(config, self.logger)
        
        # Setup small drawdown (1%)
        self.portfolio_next['unrealized_pnl'] = -1000.0
        self.portfolio_next['total_equity'] = 100000.0
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0  # Below warning threshold
        assert diagnostics['drawdown_pct'] == 0.01
    
    def test_drawdown_penalty_warning_level(self):
        """Test DrawdownPenalty at warning level"""
        config = {**self.config, 'warning_threshold': 0.02, 'base_penalty': 0.01}
        component = DrawdownPenalty(config, self.logger)
        
        # Setup warning level drawdown (3%)
        self.portfolio_next['unrealized_pnl'] = -3000.0
        self.portfolio_next['total_equity'] = 100000.0
        
        reward, diagnostics = component.calculate(self.state)
        
        # Penalty: -0.01 * (0.03 / 0.02) = -0.015
        assert reward == -0.015
        assert diagnostics['drawdown_pct'] == 0.03
    
    def test_drawdown_penalty_severe_level(self):
        """Test DrawdownPenalty at severe level"""
        config = {
            **self.config, 
            'warning_threshold': 0.02, 
            'severe_threshold': 0.05,
            'base_penalty': 0.01
        }
        component = DrawdownPenalty(config, self.logger)
        
        # Setup severe drawdown (8%)
        self.portfolio_next['unrealized_pnl'] = -8000.0
        self.portfolio_next['total_equity'] = 100000.0
        
        reward, diagnostics = component.calculate(self.state)
        
        # Severe penalty: -0.01 * (0.08 / 0.02)^2 = -0.16
        assert reward == -0.16
        assert diagnostics['drawdown_pct'] == 0.08
    
    def test_terminal_penalty_bankruptcy(self):
        """Test TerminalPenalty for bankruptcy"""
        config = {**self.config, 'bankruptcy_penalty': 100.0}
        component = TerminalPenalty(config, self.logger)
        
        self.state.terminated = True
        self.state.termination_reason = 'BANKRUPTCY'
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == -100.0
        assert diagnostics['reason'] == 'BANKRUPTCY'
    
    def test_terminal_penalty_max_loss(self):
        """Test TerminalPenalty for max loss"""
        config = {**self.config, 'max_loss_penalty': 50.0}
        component = TerminalPenalty(config, self.logger)
        
        self.state.terminated = True
        self.state.termination_reason = 'MAX_LOSS'
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == -50.0
        assert diagnostics['reason'] == 'MAX_LOSS'
    
    def test_terminal_penalty_other_reason(self):
        """Test TerminalPenalty for other termination"""
        config = {**self.config, 'default_penalty': 10.0}
        component = TerminalPenalty(config, self.logger)
        
        self.state.terminated = True
        self.state.termination_reason = 'DATA_END'
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == -10.0
        assert diagnostics['reason'] == 'DATA_END'
    
    def test_terminal_penalty_not_terminated(self):
        """Test TerminalPenalty when not terminated"""
        component = TerminalPenalty(self.config, self.logger)
        
        self.state.terminated = False
        
        reward, diagnostics = component.calculate(self.state)
        
        assert reward == 0.0
        assert diagnostics['terminated'] == False


class TestRewardAggregator:
    """Test RewardAggregator functionality"""
    
    def setup_method(self):
        """Setup test components"""
        self.mock_comp1 = Mock(spec=RewardComponent)
        self.mock_comp1.metadata = Mock()
        self.mock_comp1.metadata.name = 'component1'
        self.mock_comp1.return_value = (5.0, {'test': True})
        
        self.mock_comp2 = Mock(spec=RewardComponent)
        self.mock_comp2.metadata = Mock()
        self.mock_comp2.metadata.name = 'component2'
        self.mock_comp2.return_value = (-2.0, {'test': True})
        
        self.components = [self.mock_comp1, self.mock_comp2]
        self.config = {'global_scale': 1.0}
        self.aggregator = RewardAggregator(self.components, self.config)
        
        self.state = Mock(spec=RewardState)
    
    def test_basic_aggregation(self):
        """Test basic reward aggregation"""
        total_reward, diagnostics = self.aggregator.calculate_total_reward(self.state)
        
        assert total_reward == 3.0  # 5.0 + (-2.0)
        assert 'component1' in diagnostics
        assert 'component2' in diagnostics
        assert diagnostics['summary']['total_reward'] == 3.0
        assert diagnostics['summary']['component_rewards']['component1'] == 5.0
        assert diagnostics['summary']['component_rewards']['component2'] == -2.0
    
    def test_global_scaling(self):
        """Test global scaling"""
        config = {'global_scale': 2.0}
        aggregator = RewardAggregator(self.components, config)
        
        total_reward, diagnostics = aggregator.calculate_total_reward(self.state)
        
        assert total_reward == 6.0  # (5.0 + (-2.0)) * 2.0
        assert diagnostics['summary']['global_scale'] == 2.0
    
    def test_smoothing(self):
        """Test reward smoothing"""
        config = {'global_scale': 1.0, 'use_smoothing': True, 'smoothing_window': 3}
        aggregator = RewardAggregator(self.components, config)
        
        # First calculation
        total_reward1, diagnostics1 = aggregator.calculate_total_reward(self.state)
        assert total_reward1 == 3.0  # No smoothing with single value
        
        # Second calculation
        total_reward2, diagnostics2 = aggregator.calculate_total_reward(self.state)
        assert total_reward2 == 3.0  # Average of [3.0, 3.0]
        
        # Third calculation with different component values
        self.mock_comp1.return_value = (10.0, {'test': True})
        total_reward3, diagnostics3 = aggregator.calculate_total_reward(self.state)
        expected = np.mean([3.0, 3.0, 8.0])  # 10 + (-2) = 8
        assert abs(total_reward3 - expected) < 1e-6
    
    def test_component_statistics(self):
        """Test component statistics tracking"""
        # Run multiple calculations
        for _ in range(3):
            self.aggregator.calculate_total_reward(self.state)
        
        stats = self.aggregator.get_component_statistics()
        
        assert 'component1' in stats
        assert 'component2' in stats
        assert stats['component1']['mean'] == 5.0
        assert stats['component1']['count'] == 3
        assert stats['component2']['mean'] == -2.0
        assert stats['component2']['count'] == 3
    
    def test_reset_statistics(self):
        """Test statistics reset"""
        # Run calculations
        self.aggregator.calculate_total_reward(self.state)
        
        # Verify stats exist
        stats_before = self.aggregator.get_component_statistics()
        assert len(stats_before) > 0
        
        # Reset and verify
        self.aggregator.reset_statistics()
        stats_after = self.aggregator.get_component_statistics()
        assert len(stats_after) == 0


class TestRewardSystem:
    """Test complete RewardSystem integration"""
    
    def setup_method(self):
        """Setup test RewardSystem"""
        self.config = MockRewardConfig()
        self.metrics_integrator = Mock()
        self.logger = Mock()
        
        self.reward_system = RewardSystem(
            config=self.config,
            metrics_integrator=self.metrics_integrator,
            logger=self.logger
        )
        
        # Mock portfolio states
        self.portfolio_before = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
        
        self.portfolio_after = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
        
        self.portfolio_next = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
    
    def test_initialization(self):
        """Test RewardSystem initialization"""
        assert len(self.reward_system.components) > 0
        assert self.reward_system.aggregator is not None
        assert self.reward_system.metrics_tracker is not None
        assert self.reward_system.step_count == 0
        assert self.reward_system.current_trade is None
    
    def test_reset(self):
        """Test RewardSystem reset"""
        # Set some state
        self.reward_system.step_count = 100
        self.reward_system.current_trade = TradeTracker(100.0, 50)
        
        # Reset
        self.reward_system.reset()
        
        assert self.reward_system.step_count == 0
        assert self.reward_system.current_trade is None
        assert self.reward_system.episode_started == True
    
    def test_calculate_basic(self):
        """Test basic reward calculation"""
        decoded_action = {'type': Mock(name='HOLD')}
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_before,
            portfolio_state_after_action_fills=self.portfolio_after,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
        assert self.reward_system.step_count == 1
    
    def test_calculate_with_profitable_trade(self):
        """Test reward calculation with profitable trade"""
        # Setup profitable trade
        self.portfolio_after['realized_pnl_session'] = 100.0
        
        fill = Mock()
        fill.commission = 1.0
        fill.slippage_cost_total = 0.5
        fill.closes_position = True
        fill.realized_pnl = 98.5
        
        decoded_action = {'type': Mock(name='SELL')}
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_before,
            portfolio_state_after_action_fills=self.portfolio_after,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[fill],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert reward > 0  # Should be positive due to PnL component
        
        # Check that metrics integrator was called
        self.metrics_integrator.record_environment_step.assert_called()
    
    def test_calculate_with_bankruptcy(self):
        """Test reward calculation with bankruptcy"""
        decoded_action = {'type': Mock(name='HOLD')}
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_before,
            portfolio_state_after_action_fills=self.portfolio_after,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=True,
            truncated=False,
            termination_reason='BANKRUPTCY'
        )
        
        assert reward < 0  # Should be negative due to terminal penalty
    
    def test_trade_tracking(self):
        """Test trade tracking functionality"""
        # Open position
        self.portfolio_next['position_side'] = PositionSideEnum.LONG
        self.portfolio_next['avg_entry_price'] = 100.0
        
        decoded_action = {'type': Mock(name='BUY')}
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_before,
            portfolio_state_after_action_fills=self.portfolio_after,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert self.reward_system.current_trade is not None
        assert self.reward_system.current_trade.entry_price == 100.0
        assert self.reward_system.current_trade.entry_step == 0
        
        # Update with unrealized PnL
        self.portfolio_next['unrealized_pnl'] = 50.0
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_next,
            portfolio_state_after_action_fills=self.portfolio_next,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action={'type': Mock(name='HOLD')},
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert self.reward_system.current_trade.max_unrealized_pnl == 50.0
        
        # Close position
        self.portfolio_next['position_side'] = PositionSideEnum.FLAT
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_next,
            portfolio_state_after_action_fills=self.portfolio_next,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action={'type': Mock(name='SELL')},
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert self.reward_system.current_trade is None
    
    def test_get_last_reward_components(self):
        """Test getting last reward components"""
        decoded_action = {'type': Mock(name='HOLD')}
        
        self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_before,
            portfolio_state_after_action_fills=self.portfolio_after,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        components = self.reward_system.get_last_reward_components()
        assert isinstance(components, dict)
        # Should have component names as keys
        assert any(name for name in components.keys())
    
    def test_get_episode_summary(self):
        """Test episode summary generation"""
        # Run some calculations
        for _ in range(5):
            decoded_action = {'type': Mock(name='HOLD')}
            self.reward_system.calculate(
                portfolio_state_before_action=self.portfolio_before,
                portfolio_state_after_action_fills=self.portfolio_after,
                portfolio_state_next_t=self.portfolio_next,
                market_state_at_decision={},
                market_state_next_t={},
                decoded_action=decoded_action,
                fill_details_list=[],
                terminated=False,
                truncated=False,
                termination_reason=None
            )
        
        final_portfolio = {'total_equity': 100000.0}
        summary = self.reward_system.get_episode_summary(final_portfolio)
        
        assert 'episode' in summary
        assert 'steps' in summary
        assert 'total_reward' in summary
        assert 'aggregator_statistics' in summary
    
    def test_apply_position_close_penalty(self):
        """Test position close penalty application"""
        close_pnl = -500.0
        
        self.reward_system.apply_position_close_penalty(close_pnl)
        
        assert hasattr(self.reward_system, '_pending_reset_penalty')
        assert self.reward_system._pending_reset_penalty == -5.0  # -500 * 0.01
    
    def test_get_metrics_for_dashboard(self):
        """Test dashboard metrics"""
        # Run some calculations first
        decoded_action = {'type': Mock(name='BUY')}
        self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_before,
            portfolio_state_after_action_fills=self.portfolio_after,
            portfolio_state_next_t=self.portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        metrics = self.reward_system.get_metrics_for_dashboard()
        
        assert 'current_step' in metrics
        assert 'episode_reward' in metrics
        assert 'component_stats' in metrics
        assert 'episode_trades' in metrics
        assert 'win_rate' in metrics


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup for edge case testing"""
        self.config = MockRewardConfig()
        self.reward_system = RewardSystem(config=self.config)
        
        # Base portfolio state
        self.portfolio_state = {
            'total_equity': 100000.0,
            'realized_pnl_session': 0.0,
            'unrealized_pnl': 0.0,
            'position_side': PositionSideEnum.FLAT,
            'position_value': 0.0
        }
    
    def test_zero_equity(self):
        """Test behavior with zero equity"""
        self.portfolio_state['total_equity'] = 0.0
        
        decoded_action = {'type': Mock(name='HOLD')}
        
        # Should not crash
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_state,
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
    
    def test_negative_equity(self):
        """Test behavior with negative equity"""
        self.portfolio_state['total_equity'] = -10000.0
        
        decoded_action = {'type': Mock(name='HOLD')}
        
        # Should not crash
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_state,
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
    
    def test_extreme_pnl_values(self):
        """Test with extreme PnL values"""
        self.portfolio_state['realized_pnl_session'] = 1e6  # Very large profit
        
        decoded_action = {'type': Mock(name='HOLD')}
        
        reward = self.reward_system.calculate(
            portfolio_state_before_action={'realized_pnl_session': 0.0},
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        # Reward should be clipped due to clip_range
        assert -10.0 <= reward <= 10.0
    
    def test_missing_portfolio_fields(self):
        """Test with missing portfolio fields"""
        incomplete_portfolio = {'total_equity': 100000.0}
        
        decoded_action = {'type': Mock(name='HOLD')}
        
        # Should handle missing fields gracefully
        reward = self.reward_system.calculate(
            portfolio_state_before_action=incomplete_portfolio,
            portfolio_state_after_action_fills=incomplete_portfolio,
            portfolio_state_next_t=incomplete_portfolio,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
    
    def test_none_values(self):
        """Test handling of None values"""
        fill = Mock()
        fill.commission = None
        fill.slippage_cost_total = None
        fill.closes_position = True
        fill.realized_pnl = None
        
        decoded_action = {'type': Mock(name='SELL')}
        
        # Should handle None values gracefully
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_state,
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[fill],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
    
    def test_invalid_action_structure(self):
        """Test with invalid action structure"""
        decoded_action = {}  # Missing 'type' field
        
        # Should not crash
        reward = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_state,
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action=decoded_action,
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
    
    def test_empty_fill_details(self):
        """Test with various fill detail configurations"""
        # Empty list
        reward1 = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_state,
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action={'type': Mock(name='HOLD')},
            fill_details_list=[],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        # Multiple fills
        fills = [Mock(), Mock(), Mock()]
        for fill in fills:
            fill.commission = 1.0
            fill.slippage_cost_total = 0.5
            fill.closes_position = False
            fill.realized_pnl = 0.0
            
        reward2 = self.reward_system.calculate(
            portfolio_state_before_action=self.portfolio_state,
            portfolio_state_after_action_fills=self.portfolio_state,
            portfolio_state_next_t=self.portfolio_state,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action={'type': Mock(name='BUY')},
            fill_details_list=fills,
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        assert isinstance(reward1, float)
        assert isinstance(reward2, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])