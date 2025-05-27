"""Tests for individual reward components."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

from rewards.core import RewardState
from rewards.components import (
    RealizedPnLReward,
    MarkToMarketReward,
    DifferentialSharpeReward,
    HoldingTimePenalty,
    OvertradingPenalty,
    QuickProfitIncentive,
    DrawdownPenalty,
    MAEPenalty,
    MFEPenalty,
    TerminalPenalty
)


class TestRealizedPnLReward:
    """Test RealizedPnLReward component."""
    
    def test_no_realized_pnl(self):
        """Test when no P&L is realized."""
        component = RealizedPnLReward()
        state = RewardState(
            portfolio_state_before={"realized_pnl": 1000.0},
            portfolio_state_after_fills={"realized_pnl": 1000.0},
            portfolio_state_next={"realized_pnl": 1000.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["realized_pnl_change"] == 0.0
    
    def test_positive_realized_pnl(self):
        """Test positive realized P&L."""
        component = RealizedPnLReward()
        state = RewardState(
            portfolio_state_before={"realized_pnl": 1000.0},
            portfolio_state_after_fills={"realized_pnl": 1000.0},
            portfolio_state_next={"realized_pnl": 1150.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[{"action": "SELL", "quantity": 100, "price": 11.5}],
            terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 150.0
        assert details["realized_pnl_change"] == 150.0
        assert details["scaled_reward"] == 150.0 / 1000.0  # Default scale factor
    
    def test_negative_realized_pnl(self):
        """Test negative realized P&L."""
        component = RealizedPnLReward(scale_factor=500.0)
        state = RewardState(
            portfolio_state_before={"realized_pnl": 1000.0},
            portfolio_state_after_fills={"realized_pnl": 1000.0},
            portfolio_state_next={"realized_pnl": 800.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[{"action": "SELL", "quantity": 100, "price": 8.0}],
            terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == -200.0
        assert details["realized_pnl_change"] == -200.0
        assert details["scaled_reward"] == -200.0 / 500.0


class TestMarkToMarketReward:
    """Test MarkToMarketReward component."""
    
    def test_no_position_no_reward(self):
        """Test no reward when no position."""
        component = MarkToMarketReward()
        state = RewardState(
            portfolio_state_before={"position": 0, "unrealized_pnl": 0.0},
            portfolio_state_after_fills={"position": 0, "unrealized_pnl": 0.0},
            portfolio_state_next={"position": 0, "unrealized_pnl": 0.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["unrealized_pnl_change"] == 0.0
        assert details["has_position"] == False
    
    def test_unrealized_gain(self):
        """Test unrealized gain reward."""
        component = MarkToMarketReward(scale_factor=100.0)
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": 50.0},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": 50.0},
            portfolio_state_next={"position": 100, "unrealized_pnl": 150.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == pytest.approx(100.0 / 100.0)  # 100 change / 100 scale
        assert details["unrealized_pnl_change"] == 100.0
        assert details["has_position"] == True
        assert details["leverage_penalty"] == 0.0
    
    def test_unrealized_loss_with_leverage_penalty(self):
        """Test unrealized loss with leverage penalty."""
        component = MarkToMarketReward(
            scale_factor=100.0,
            leverage_penalty_enabled=True,
            leverage_penalty_factor=0.1
        )
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": 0.0},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": 0.0},
            portfolio_state_next={"position": 100, "unrealized_pnl": -50.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        base_reward = -50.0 / 100.0
        leverage_penalty = -0.1 * abs(100)  # position size
        expected_reward = base_reward + leverage_penalty
        
        assert reward == pytest.approx(expected_reward)
        assert details["unrealized_pnl_change"] == -50.0
        assert details["leverage_penalty"] == leverage_penalty


class TestDifferentialSharpeReward:
    """Test DifferentialSharpeReward component."""
    
    def test_insufficient_returns_history(self):
        """Test when not enough returns history."""
        component = DifferentialSharpeReward(min_periods=20)
        component.returns_history = [0.01, 0.02, -0.01]  # Only 3 returns
        
        state = RewardState(
            portfolio_state_before={"total_value": 10000.0},
            portfolio_state_after_fills={"total_value": 10000.0},
            portfolio_state_next={"total_value": 10100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["current_return"] == pytest.approx(0.01)
        assert details["periods_collected"] == 4
        assert details["sharpe_improvement"] == 0.0
    
    def test_sharpe_improvement(self):
        """Test Sharpe ratio improvement calculation."""
        component = DifferentialSharpeReward(
            min_periods=5,
            lookback_window=5,
            scale_factor=1.0
        )
        # Pre-populate with some returns
        component.returns_history = [0.01, -0.005, 0.02, -0.01, 0.015]
        
        state = RewardState(
            portfolio_state_before={"total_value": 10000.0},
            portfolio_state_after_fills={"total_value": 10000.0},
            portfolio_state_next={"total_value": 10200.0},  # 2% gain
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "BUY"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Should calculate Sharpe before and after adding new return
        assert details["current_return"] == pytest.approx(0.02)
        assert "sharpe_before" in details
        assert "sharpe_after" in details
        assert "sharpe_improvement" in details
        # Reward should be positive if Sharpe improved
        assert isinstance(reward, float)
    
    def test_zero_variance_handling(self):
        """Test handling when all returns are identical."""
        component = DifferentialSharpeReward(min_periods=3)
        component.returns_history = [0.01, 0.01, 0.01]
        
        state = RewardState(
            portfolio_state_before={"total_value": 10000.0},
            portfolio_state_after_fills={"total_value": 10000.0},
            portfolio_state_next={"total_value": 10100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Should handle zero variance gracefully
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)


class TestHoldingTimePenalty:
    """Test HoldingTimePenalty component."""
    
    def test_no_position_no_penalty(self):
        """Test no penalty when no position."""
        component = HoldingTimePenalty()
        state = RewardState(
            portfolio_state_before={"position": 0},
            portfolio_state_after_fills={"position": 0},
            portfolio_state_next={"position": 0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["holding_time"] == 0
        assert details["has_position"] == False
    
    def test_penalty_increases_with_time(self):
        """Test penalty increases with holding time."""
        component = HoldingTimePenalty(
            penalty_per_timestep=0.001,
            max_penalty=0.1
        )
        
        # Simulate holding for multiple timesteps
        rewards = []
        for i in range(10):
            state = RewardState(
                portfolio_state_before={"position": 100},
                portfolio_state_after_fills={"position": 100},
                portfolio_state_next={"position": 100},
                market_state_current={}, market_state_next={},
                decoded_action={"action_type": "HOLD"},
                fill_details=[], terminated=False, termination_reason=None,
                trade_entry=None, trade_exit=None,
                current_mae=0.0, current_mfe=0.0
            )
            
            reward, details = component.calculate(state)
            rewards.append(reward)
            assert details["holding_time"] == i + 1
        
        # Verify penalties increase
        for i in range(1, len(rewards)):
            assert rewards[i] < rewards[i-1]  # More negative
        
        # Verify max penalty is respected
        assert rewards[-1] >= -0.1
    
    def test_penalty_reset_on_exit(self):
        """Test penalty resets when position exits."""
        component = HoldingTimePenalty(penalty_per_timestep=0.01)
        
        # Hold for 5 timesteps
        for _ in range(5):
            state = RewardState(
                portfolio_state_before={"position": 100},
                portfolio_state_after_fills={"position": 100},
                portfolio_state_next={"position": 100},
                market_state_current={}, market_state_next={},
                decoded_action={"action_type": "HOLD"},
                fill_details=[], terminated=False, termination_reason=None,
                trade_entry=None, trade_exit=None,
                current_mae=0.0, current_mfe=0.0
            )
            component.calculate(state)
        
        # Exit position
        state = RewardState(
            portfolio_state_before={"position": 100},
            portfolio_state_after_fills={"position": 0},
            portfolio_state_next={"position": 0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        assert details["holding_time"] == 0  # Reset


class TestOvertradingPenalty:
    """Test OvertradingPenalty component."""
    
    def test_no_trades_no_penalty(self):
        """Test no penalty when no trades."""
        component = OvertradingPenalty(lookback_window=10)
        
        state = RewardState(
            portfolio_state_before={"position": 0},
            portfolio_state_after_fills={"position": 0},
            portfolio_state_next={"position": 0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["trade_count"] == 0
        assert details["is_new_trade"] == False
    
    def test_penalty_for_frequent_trading(self):
        """Test penalty increases with trade frequency."""
        component = OvertradingPenalty(
            lookback_window=5,
            penalty_per_trade=0.01,
            max_trades_threshold=3
        )
        
        # Simulate multiple trades
        rewards = []
        for i in range(6):
            # Alternate between buy and sell
            action = "BUY" if i % 2 == 0 else "SELL"
            position_before = 0 if i % 2 == 0 else 100
            position_after = 100 if i % 2 == 0 else 0
            
            state = RewardState(
                portfolio_state_before={"position": position_before},
                portfolio_state_after_fills={"position": position_after},
                portfolio_state_next={"position": position_after},
                market_state_current={}, market_state_next={},
                decoded_action={"action_type": action},
                fill_details=[{"action": action}] if action != "HOLD" else [],
                terminated=False, termination_reason=None,
                trade_entry=None, trade_exit=None,
                current_mae=0.0, current_mfe=0.0
            )
            
            reward, details = component.calculate(state)
            rewards.append((reward, details["trade_count"]))
        
        # Verify penalties increase after threshold
        assert rewards[0][0] == 0.0  # First trade, no penalty
        assert rewards[1][0] == 0.0  # Second trade, no penalty
        assert rewards[2][0] == 0.0  # Third trade, no penalty
        assert rewards[3][0] < 0.0   # Fourth trade, penalty
        assert rewards[4][0] < 0.0   # Fifth trade, penalty
        # Window slides, oldest trade drops off
        assert rewards[5][1] == 5    # Still 5 trades in window


class TestQuickProfitIncentive:
    """Test QuickProfitIncentive component."""
    
    def test_no_exit_no_reward(self):
        """Test no reward when not exiting."""
        component = QuickProfitIncentive()
        
        state = RewardState(
            portfolio_state_before={"position": 100},
            portfolio_state_after_fills={"position": 100},
            portfolio_state_next={"position": 100},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry={"timestamp": datetime.now()},
            trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["is_exit"] == False
    
    def test_quick_profit_exit(self):
        """Test reward for quick profitable exit."""
        component = QuickProfitIncentive(
            time_decay_factor=0.01,
            min_profit_threshold=50.0,
            max_reward=1.0
        )
        
        # Create timestamps 10 seconds apart
        entry_time = datetime.now()
        exit_time = datetime.now()  # Same time for simplicity
        
        state = RewardState(
            portfolio_state_before={"position": 100, "realized_pnl": 1000.0},
            portfolio_state_after_fills={"position": 0, "realized_pnl": 1000.0},
            portfolio_state_next={"position": 0, "realized_pnl": 1100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[{"action": "SELL"}],
            terminated=False, termination_reason=None,
            trade_entry={"timestamp": entry_time},
            trade_exit={"timestamp": exit_time},
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward > 0.0
        assert details["is_exit"] == True
        assert details["pnl_change"] == 100.0
        assert details["hold_duration_seconds"] >= 0
        assert reward <= 1.0  # Max reward respected
    
    def test_no_reward_for_loss(self):
        """Test no reward for losing trade."""
        component = QuickProfitIncentive(min_profit_threshold=50.0)
        
        state = RewardState(
            portfolio_state_before={"position": 100, "realized_pnl": 1000.0},
            portfolio_state_after_fills={"position": 0, "realized_pnl": 1000.0},
            portfolio_state_next={"position": 0, "realized_pnl": 950.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[{"action": "SELL"}],
            terminated=False, termination_reason=None,
            trade_entry={"timestamp": datetime.now()},
            trade_exit={"timestamp": datetime.now()},
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["pnl_change"] == -50.0


class TestDrawdownPenalty:
    """Test DrawdownPenalty component."""
    
    def test_no_position_no_penalty(self):
        """Test no penalty when no position."""
        component = DrawdownPenalty()
        
        state = RewardState(
            portfolio_state_before={"position": 0, "unrealized_pnl": 0.0},
            portfolio_state_after_fills={"position": 0, "unrealized_pnl": 0.0},
            portfolio_state_next={"position": 0, "unrealized_pnl": 0.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["unrealized_pnl"] == 0.0
        assert details["has_position"] == False
    
    def test_penalty_for_negative_unrealized(self):
        """Test penalty for negative unrealized P&L."""
        component = DrawdownPenalty(
            penalty_factor=0.5,
            threshold=-50.0
        )
        
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": -100.0},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": -100.0},
            portfolio_state_next={"position": 100, "unrealized_pnl": -100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Penalty = factor * (unrealized - threshold) when below threshold
        expected_penalty = -0.5 * (-100.0 - (-50.0))
        assert reward == pytest.approx(expected_penalty)
        assert details["unrealized_pnl"] == -100.0
        assert details["penalty_amount"] == expected_penalty
    
    def test_no_penalty_above_threshold(self):
        """Test no penalty when above threshold."""
        component = DrawdownPenalty(threshold=-50.0)
        
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": -30.0},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": -30.0},
            portfolio_state_next={"position": 100, "unrealized_pnl": -30.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["unrealized_pnl"] == -30.0


class TestMAEPenalty:
    """Test MAEPenalty component."""
    
    def test_no_position_no_penalty(self):
        """Test no penalty when no position."""
        component = MAEPenalty()
        
        state = RewardState(
            portfolio_state_before={"position": 0},
            portfolio_state_after_fills={"position": 0},
            portfolio_state_next={"position": 0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["current_mae"] == 0.0
        assert details["has_position"] == False
    
    def test_mae_penalty_calculation(self):
        """Test MAE penalty calculation."""
        component = MAEPenalty(
            penalty_factor=0.1,
            threshold=-100.0
        )
        
        state = RewardState(
            portfolio_state_before={"position": 100},
            portfolio_state_after_fills={"position": 100},
            portfolio_state_next={"position": 100},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=-150.0,  # Worst drawdown so far
            current_mfe=50.0
        )
        
        reward, details = component.calculate(state)
        
        # Penalty when MAE worse than threshold
        expected_penalty = -0.1 * abs(-150.0 - (-100.0))
        assert reward == pytest.approx(expected_penalty)
        assert details["current_mae"] == -150.0
        assert details["penalty_amount"] == expected_penalty


class TestMFEPenalty:
    """Test MFEPenalty component."""
    
    def test_no_position_no_penalty(self):
        """Test no penalty when no position."""
        component = MFEPenalty()
        
        state = RewardState(
            portfolio_state_before={"position": 0, "unrealized_pnl": 0.0},
            portfolio_state_after_fills={"position": 0, "unrealized_pnl": 0.0},
            portfolio_state_next={"position": 0, "unrealized_pnl": 0.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["has_position"] == False
    
    def test_penalty_for_profit_giveback(self):
        """Test penalty when giving back profits from peak."""
        component = MFEPenalty(
            penalty_factor=0.2,
            giveback_threshold=0.3  # 30% giveback threshold
        )
        
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": 100.0},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": 100.0},
            portfolio_state_next={"position": 100, "unrealized_pnl": 60.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=-20.0,
            current_mfe=150.0  # Peak profit was 150
        )
        
        reward, details = component.calculate(state)
        
        # Giveback = (150 - 60) / 150 = 0.6 (60%)
        # Exceeds 30% threshold, so penalty applies
        giveback_amount = 150.0 - 60.0
        expected_penalty = -0.2 * giveback_amount
        
        assert reward == pytest.approx(expected_penalty)
        assert details["current_mfe"] == 150.0
        assert details["current_unrealized"] == 60.0
        assert details["giveback_ratio"] == pytest.approx(0.6)


class TestTerminalPenalty:
    """Test TerminalPenalty component."""
    
    def test_no_penalty_when_not_terminated(self):
        """Test no penalty when episode not terminated."""
        component = TerminalPenalty()
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["terminated"] == False
        assert details["termination_reason"] is None
    
    def test_bankruptcy_penalty(self):
        """Test penalty for bankruptcy termination."""
        component = TerminalPenalty(
            bankruptcy_penalty=-10.0,
            max_loss_penalty=-5.0,
            invalid_action_penalty=-1.0
        )
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={},
            fill_details=[], terminated=True, termination_reason="bankruptcy",
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == -10.0
        assert details["terminated"] == True
        assert details["termination_reason"] == "bankruptcy"
        assert details["penalty_amount"] == -10.0
    
    def test_max_loss_penalty(self):
        """Test penalty for max loss termination."""
        component = TerminalPenalty(
            bankruptcy_penalty=-10.0,
            max_loss_penalty=-5.0,
            invalid_action_penalty=-1.0
        )
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={},
            fill_details=[], terminated=True, termination_reason="max_loss",
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == -5.0
        assert details["termination_reason"] == "max_loss"
        assert details["penalty_amount"] == -5.0
    
    def test_invalid_action_penalty(self):
        """Test penalty for invalid action termination."""
        component = TerminalPenalty(invalid_action_penalty=-2.0)
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={},
            fill_details=[], terminated=True, termination_reason="invalid_action",
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == -2.0
        assert details["termination_reason"] == "invalid_action"
    
    def test_normal_termination_no_penalty(self):
        """Test no penalty for normal termination."""
        component = TerminalPenalty()
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={},
            fill_details=[], terminated=True, termination_reason="end_of_data",
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        assert reward == 0.0
        assert details["termination_reason"] == "end_of_data"
        assert details["penalty_amount"] == 0.0