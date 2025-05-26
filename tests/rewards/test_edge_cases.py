"""Edge case and error handling tests for reward system."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch
import math

from rewards.core import RewardState, RewardComponent, RewardMetadata, RewardType
from rewards.components import (
    RealizedPnLReward,
    MarkToMarketReward,
    DifferentialSharpeReward,
    HoldingTimePenalty,
    DrawdownPenalty
)
from rewards.calculator import RewardSystemV2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_nan_values_in_state(self):
        """Test handling of NaN values in state."""
        component = RealizedPnLReward()
        
        state = RewardState(
            portfolio_state_before={"realized_pnl": np.nan},
            portfolio_state_after_fills={"realized_pnl": 1000.0},
            portfolio_state_next={"realized_pnl": 1100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Should handle NaN gracefully
        assert not np.isnan(reward)
        assert isinstance(reward, float)
    
    def test_inf_values_in_state(self):
        """Test handling of infinite values."""
        component = MarkToMarketReward(scale_factor=0.0)  # Will cause division issues
        
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": float('inf')},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": 100.0},
            portfolio_state_next={"position": 100, "unrealized_pnl": 150.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Should handle infinity gracefully
        assert not np.isinf(reward)
        assert isinstance(reward, float)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields in state."""
        component = RealizedPnLReward()
        
        # State missing realized_pnl field
        state = RewardState(
            portfolio_state_before={},  # Missing realized_pnl
            portfolio_state_after_fills={},
            portfolio_state_next={"realized_pnl": 1000.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        # Should handle missing fields without crashing
        reward, details = component.calculate(state)
        assert isinstance(reward, float)
    
    def test_zero_division_scenarios(self):
        """Test scenarios that could cause zero division."""
        # Test 1: Sharpe ratio with zero variance
        sharpe_component = DifferentialSharpeReward(min_periods=2)
        sharpe_component.returns_history = [0.0, 0.0]  # Zero variance
        
        state = RewardState(
            portfolio_state_before={"total_value": 10000.0},
            portfolio_state_after_fills={"total_value": 10000.0},
            portfolio_state_next={"total_value": 10000.0},  # No change
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = sharpe_component.calculate(state)
        assert not np.isnan(reward)
        assert not np.isinf(reward)
        
        # Test 2: Scale factor of zero
        pnl_component = RealizedPnLReward(scale_factor=0.0)
        state2 = RewardState(
            portfolio_state_before={"realized_pnl": 1000.0},
            portfolio_state_after_fills={"realized_pnl": 1000.0},
            portfolio_state_next={"realized_pnl": 1100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward2, details2 = pnl_component.calculate(state2)
        assert not np.isnan(reward2)
        assert not np.isinf(reward2)
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        component = DrawdownPenalty(penalty_factor=1e10)  # Extreme penalty factor
        
        state = RewardState(
            portfolio_state_before={"position": 100, "unrealized_pnl": -1e6},
            portfolio_state_after_fills={"position": 100, "unrealized_pnl": -1e6},
            portfolio_state_next={"position": 100, "unrealized_pnl": -1e6},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Should be capped by anti-hacking measures if implemented
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_malformed_fill_details(self):
        """Test handling of malformed fill details."""
        component = RealizedPnLReward()
        
        # Various malformed fill details
        malformed_fills = [
            {},  # Empty dict
            {"action": "BUY"},  # Missing fields
            {"quantity": "not_a_number"},  # Wrong type
            None,  # None instead of dict
        ]
        
        for fills in [malformed_fills]:
            state = RewardState(
                portfolio_state_before={"realized_pnl": 1000.0},
                portfolio_state_after_fills={"realized_pnl": 1000.0},
                portfolio_state_next={"realized_pnl": 1100.0},
                market_state_current={}, market_state_next={},
                decoded_action={"action_type": "SELL"},
                fill_details=fills,
                terminated=False, termination_reason=None,
                trade_entry=None, trade_exit=None,
                current_mae=0.0, current_mfe=0.0
            )
            
            # Should not crash
            reward, details = component.calculate(state)
            assert isinstance(reward, float)
    
    def test_rapid_position_changes(self):
        """Test rapid position changes (e.g., buy then immediate sell)."""
        component = HoldingTimePenalty()
        
        # Buy position
        state1 = RewardState(
            portfolio_state_before={"position": 0},
            portfolio_state_after_fills={"position": 100},
            portfolio_state_next={"position": 100},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "BUY"},
            fill_details=[{"action": "BUY"}],
            terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward1, _ = component.calculate(state1)
        
        # Immediate sell
        state2 = RewardState(
            portfolio_state_before={"position": 100},
            portfolio_state_after_fills={"position": 0},
            portfolio_state_next={"position": 0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[{"action": "SELL"}],
            terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward2, details2 = component.calculate(state2)
        
        # Holding time should reset
        assert details2["holding_time"] == 0
    
    def test_concurrent_modifications(self):
        """Test thread safety with concurrent modifications."""
        from rewards.core import RewardAggregator
        
        components = [MockRewardComponent("test1"), MockRewardComponent("test2")]
        aggregator = RewardAggregator(components, smoothing_enabled=True)
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        # Simulate concurrent calls (in practice would use threading)
        rewards = []
        for _ in range(10):
            reward, _ = aggregator.aggregate(state)
            rewards.append(reward)
        
        # Should maintain consistency
        assert len(rewards) == 10
        assert all(isinstance(r, float) for r in rewards)
    
    def test_timestamp_edge_cases(self):
        """Test edge cases with timestamps."""
        from rewards.components import QuickProfitIncentive
        
        component = QuickProfitIncentive()
        
        # Test 1: Exit timestamp before entry (data error)
        now = datetime.now()
        earlier = datetime(2020, 1, 1)
        
        state = RewardState(
            portfolio_state_before={"position": 100, "realized_pnl": 1000.0},
            portfolio_state_after_fills={"position": 0, "realized_pnl": 1000.0},
            portfolio_state_next={"position": 0, "realized_pnl": 1100.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "SELL"},
            fill_details=[{"action": "SELL"}],
            terminated=False, termination_reason=None,
            trade_entry={"timestamp": now},
            trade_exit={"timestamp": earlier},  # Before entry!
            current_mae=0.0, current_mfe=0.0
        )
        
        # Should handle gracefully
        reward, details = component.calculate(state)
        assert isinstance(reward, float)
        assert not np.isnan(reward)
    
    def test_state_inconsistencies(self):
        """Test handling of inconsistent state data."""
        component = MarkToMarketReward()
        
        # Inconsistent: position is 0 but unrealized P&L is non-zero
        state = RewardState(
            portfolio_state_before={"position": 0, "unrealized_pnl": 100.0},
            portfolio_state_after_fills={"position": 0, "unrealized_pnl": 150.0},
            portfolio_state_next={"position": 0, "unrealized_pnl": 200.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "HOLD"},
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, details = component.calculate(state)
        
        # Should handle based on position check
        assert reward == 0.0  # No position, no reward
        assert details["has_position"] == False
    
    def test_action_type_validation(self):
        """Test handling of invalid action types."""
        component = RealizedPnLReward()
        
        # Invalid action type
        state = RewardState(
            portfolio_state_before={"realized_pnl": 1000.0},
            portfolio_state_after_fills={"realized_pnl": 1000.0},
            portfolio_state_next={"realized_pnl": 1000.0},
            market_state_current={}, market_state_next={},
            decoded_action={"action_type": "INVALID_ACTION"},  # Invalid
            fill_details=[], terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        # Should not crash
        reward, details = component.calculate(state)
        assert isinstance(reward, float)
    
    def test_memory_limits(self):
        """Test behavior at memory limits."""
        from rewards.metrics import RewardMetricsTracker
        
        tracker = RewardMetricsTracker(Mock())
        tracker.register_component("test", "FOUNDATIONAL")
        
        # Add many rewards to test memory limits
        for i in range(2000):  # Beyond default history limit
            tracker.update(
                total_reward=float(i),
                component_rewards={"test": {"reward": float(i), "details": {}}},
                action="HOLD"
            )
        
        # Should not exceed memory limits
        assert len(tracker.episode_rewards) <= 1000
        assert len(tracker.action_rewards["HOLD"]) <= 1000
        
        # Should still function correctly
        summary = tracker.get_episode_summary()
        assert isinstance(summary, dict)
        assert summary["total_steps"] == 2000


class MockRewardComponent(RewardComponent):
    """Mock component for testing."""
    
    def __init__(self, name: str):
        super().__init__(
            metadata=RewardMetadata(
                name=name,
                type=RewardType.SHAPING,
                description=f"Mock {name}",
                min_value=-1.0,
                max_value=1.0,
                is_penalty=False
            ),
            weight=1.0,
            enabled=True
        )
    
    def calculate(self, state: RewardState) -> tuple[float, Dict[str, Any]]:
        return 0.0, {}


class TestRewardSystemV2EdgeCases:
    """Test edge cases for the main reward system."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.env.reward_v2.pnl.enabled = True
        config.env.reward_v2.pnl.coefficient = 1.0
        config.env.reward_v2.holding_penalty.enabled = True
        config.env.reward_v2.holding_penalty.coefficient = 0.001
        config.env.reward_v2.drawdown_penalty.enabled = False
        config.env.reward_v2.drawdown_penalty.coefficient = 0.1
        return config
    
    def test_first_step_no_history(self, mock_config):
        """Test first step when no history exists."""
        system = RewardSystemV2(mock_config, None, Mock())
        system.reset()
        
        portfolio = {"cash": 10000, "position": 0, "total_value": 10000,
                    "unrealized_pnl": 0, "realized_pnl": 0}
        market = {"mid": 10.0}
        action = {"action_type": "HOLD"}
        
        # First step should work without history
        reward = system.calculate(
            portfolio_state_before=portfolio,
            portfolio_state_after_fills=portfolio,
            portfolio_state_next=portfolio,
            market_state_at_decision=market,
            market_state_next=market,
            decoded_action=action,
            fill_details=[],
            terminated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
        assert system.step_count == 1
    
    def test_all_components_disabled(self, mock_config):
        """Test when all components are disabled."""
        # Disable all components
        mock_config.env.reward_v2.pnl.enabled = False
        mock_config.env.reward_v2.holding_penalty.enabled = False
        mock_config.env.reward_v2.drawdown_penalty.enabled = False
        
        system = RewardSystemV2(mock_config, None, Mock())
        system.reset()
        
        portfolio = {"cash": 10000, "position": 0, "total_value": 10000,
                    "unrealized_pnl": 0, "realized_pnl": 0}
        market = {"mid": 10.0}
        action = {"action_type": "BUY"}
        
        reward = system.calculate(
            portfolio_state_before=portfolio,
            portfolio_state_after_fills=portfolio,
            portfolio_state_next=portfolio,
            market_state_at_decision=market,
            market_state_next=market,
            decoded_action=action,
            fill_details=[],
            terminated=False,
            termination_reason=None
        )
        
        # Should return 0 when no components
        assert reward == 0.0
    
    def test_none_values_in_input(self, mock_config):
        """Test handling of None values in inputs."""
        system = RewardSystemV2(mock_config, None, Mock())
        system.reset()
        
        # Various None scenarios
        test_cases = [
            # None in portfolio values
            ({"cash": None, "position": 0, "total_value": 10000,
              "unrealized_pnl": 0, "realized_pnl": 0}, {"mid": 10.0}),
            # None in market values
            ({"cash": 10000, "position": 0, "total_value": 10000,
              "unrealized_pnl": 0, "realized_pnl": 0}, {"mid": None}),
            # None action
            ({"cash": 10000, "position": 0, "total_value": 10000,
              "unrealized_pnl": 0, "realized_pnl": 0}, {"mid": 10.0}),
        ]
        
        for portfolio, market in test_cases:
            reward = system.calculate(
                portfolio_state_before=portfolio,
                portfolio_state_after_fills=portfolio,
                portfolio_state_next=portfolio,
                market_state_at_decision=market,
                market_state_next=market,
                decoded_action={"action_type": None} if portfolio["cash"] is not None and market["mid"] is not None else {"action_type": "HOLD"},
                fill_details=None,  # None instead of list
                terminated=False,
                termination_reason=None
            )
            
            # Should handle gracefully
            assert isinstance(reward, float)
            assert not np.isnan(reward)
            assert not np.isinf(reward)