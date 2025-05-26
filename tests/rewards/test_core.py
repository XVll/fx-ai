"""Tests for reward system core components."""

import pytest
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from rewards.core import (
    RewardType,
    RewardMetadata,
    RewardState,
    RewardComponent,
    RewardAggregator
)


class TestRewardState:
    """Test RewardState dataclass behavior."""
    
    def test_reward_state_creation_minimal(self):
        """Test creating RewardState with minimal required fields."""
        state = RewardState(
            portfolio_state_before={
                "cash": 10000.0,
                "position": 0,
                "total_value": 10000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            },
            portfolio_state_after_fills={
                "cash": 9000.0,
                "position": 100,
                "total_value": 10000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            },
            portfolio_state_next={
                "cash": 9000.0,
                "position": 100,
                "total_value": 10050.0,
                "unrealized_pnl": 50.0,
                "realized_pnl": 0.0
            },
            market_state_current={
                "bid": 9.99,
                "ask": 10.01,
                "mid": 10.00,
                "spread": 0.02,
                "volume": 50000
            },
            market_state_next={
                "bid": 10.04,
                "ask": 10.06,
                "mid": 10.05,
                "spread": 0.02,
                "volume": 55000
            },
            decoded_action={
                "action_type": "BUY",
                "position_size": 0.5,
                "raw_action": 1
            },
            fill_details=[],
            terminated=False,
            termination_reason=None,
            trade_entry=None,
            trade_exit=None,
            current_mae=0.0,
            current_mfe=0.0
        )
        
        assert state.portfolio_state_before["cash"] == 10000.0
        assert state.portfolio_state_after_fills["position"] == 100
        assert state.portfolio_state_next["unrealized_pnl"] == 50.0
        assert state.market_state_current["mid"] == 10.00
        assert state.decoded_action["action_type"] == "BUY"
        assert state.terminated == False
        assert state.current_mae == 0.0
        assert state.current_mfe == 0.0
    
    def test_reward_state_with_fills(self):
        """Test RewardState with fill details."""
        fill_details = [
            {
                "timestamp": datetime.now(),
                "action": "BUY",
                "quantity": 100,
                "price": 10.01,
                "commission": 1.0,
                "slippage": 0.01
            }
        ]
        
        state = RewardState(
            portfolio_state_before={"cash": 10000.0, "position": 0},
            portfolio_state_after_fills={"cash": 8999.0, "position": 100},
            portfolio_state_next={"cash": 8999.0, "position": 100},
            market_state_current={"mid": 10.00},
            market_state_next={"mid": 10.05},
            decoded_action={"action_type": "BUY"},
            fill_details=fill_details,
            terminated=False,
            termination_reason=None,
            trade_entry=None,
            trade_exit=None,
            current_mae=0.0,
            current_mfe=0.0
        )
        
        assert len(state.fill_details) == 1
        assert state.fill_details[0]["quantity"] == 100
        assert state.fill_details[0]["price"] == 10.01
        assert state.fill_details[0]["commission"] == 1.0
    
    def test_reward_state_with_trade_tracking(self):
        """Test RewardState with trade entry/exit tracking."""
        state = RewardState(
            portfolio_state_before={"cash": 10000.0, "position": 100},
            portfolio_state_after_fills={"cash": 11000.0, "position": 0},
            portfolio_state_next={"cash": 11000.0, "position": 0},
            market_state_current={"mid": 10.00},
            market_state_next={"mid": 10.05},
            decoded_action={"action_type": "SELL"},
            fill_details=[],
            terminated=False,
            termination_reason=None,
            trade_entry={"price": 9.50, "timestamp": datetime.now()},
            trade_exit={"price": 10.00, "timestamp": datetime.now()},
            current_mae=-0.20,
            current_mfe=0.60
        )
        
        assert state.trade_entry["price"] == 9.50
        assert state.trade_exit["price"] == 10.00
        assert state.current_mae == -0.20
        assert state.current_mfe == 0.60
    
    def test_reward_state_terminal(self):
        """Test RewardState in terminal state."""
        state = RewardState(
            portfolio_state_before={"cash": 100.0, "position": 0},
            portfolio_state_after_fills={"cash": 100.0, "position": 0},
            portfolio_state_next={"cash": 100.0, "position": 0},
            market_state_current={"mid": 10.00},
            market_state_next={"mid": 10.00},
            decoded_action={"action_type": "HOLD"},
            fill_details=[],
            terminated=True,
            termination_reason="bankruptcy",
            trade_entry=None,
            trade_exit=None,
            current_mae=0.0,
            current_mfe=0.0
        )
        
        assert state.terminated == True
        assert state.termination_reason == "bankruptcy"


class MockRewardComponent(RewardComponent):
    """Mock component for testing aggregation."""
    
    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True,
                 fixed_reward: float = 0.0):
        super().__init__(
            metadata=RewardMetadata(
                name=name,
                type=RewardType.SHAPING,
                description=f"Mock component {name}",
                min_value=-1.0,
                max_value=1.0,
                is_penalty=fixed_reward < 0
            ),
            weight=weight,
            enabled=enabled
        )
        self.fixed_reward = fixed_reward
        self.calculate_count = 0
    
    def calculate(self, state: RewardState) -> tuple[float, Dict[str, Any]]:
        """Return fixed reward for testing."""
        self.calculate_count += 1
        details = {
            "raw_reward": self.fixed_reward,
            "calculate_count": self.calculate_count
        }
        return self.fixed_reward, details


class TestRewardAggregator:
    """Test RewardAggregator functionality."""
    
    def test_aggregator_single_component(self):
        """Test aggregator with single component."""
        component = MockRewardComponent("test", weight=1.0, fixed_reward=0.5)
        aggregator = RewardAggregator(
            components=[component],
            smoothing_enabled=False
        )
        
        state = self._create_dummy_state()
        total_reward, details = aggregator.aggregate(state)
        
        assert total_reward == 0.5
        assert "test" in details
        assert details["test"]["raw_reward"] == 0.5
        assert details["test"]["weighted_reward"] == 0.5
    
    def test_aggregator_multiple_components_weighted(self):
        """Test aggregator with multiple weighted components."""
        components = [
            MockRewardComponent("comp1", weight=2.0, fixed_reward=0.5),
            MockRewardComponent("comp2", weight=1.0, fixed_reward=0.3),
            MockRewardComponent("comp3", weight=0.5, fixed_reward=-0.2)
        ]
        aggregator = RewardAggregator(
            components=components,
            smoothing_enabled=False
        )
        
        state = self._create_dummy_state()
        total_reward, details = aggregator.aggregate(state)
        
        # Expected: (2.0 * 0.5 + 1.0 * 0.3 + 0.5 * -0.2) = 1.2
        assert total_reward == pytest.approx(1.2)
        assert details["comp1"]["weighted_reward"] == 1.0
        assert details["comp2"]["weighted_reward"] == 0.3
        assert details["comp3"]["weighted_reward"] == -0.1
    
    def test_aggregator_disabled_component(self):
        """Test that disabled components are not calculated."""
        components = [
            MockRewardComponent("enabled", weight=1.0, enabled=True, fixed_reward=0.5),
            MockRewardComponent("disabled", weight=1.0, enabled=False, fixed_reward=0.3)
        ]
        aggregator = RewardAggregator(
            components=components,
            smoothing_enabled=False
        )
        
        state = self._create_dummy_state()
        total_reward, details = aggregator.aggregate(state)
        
        assert total_reward == 0.5
        assert "enabled" in details
        assert "disabled" not in details
        assert components[0].calculate_count == 1
        assert components[1].calculate_count == 0
    
    def test_aggregator_smoothing(self):
        """Test reward smoothing functionality."""
        component = MockRewardComponent("test", weight=1.0, fixed_reward=1.0)
        aggregator = RewardAggregator(
            components=[component],
            smoothing_enabled=True,
            smoothing_window=3
        )
        
        state = self._create_dummy_state()
        
        # First reward
        reward1, _ = aggregator.aggregate(state)
        assert reward1 == 1.0  # No smoothing on first reward
        
        # Update component to return different value
        component.fixed_reward = 0.0
        reward2, _ = aggregator.aggregate(state)
        assert reward2 == pytest.approx(0.5)  # Average of [1.0, 0.0]
        
        # Third reward
        component.fixed_reward = -1.0
        reward3, _ = aggregator.aggregate(state)
        assert reward3 == pytest.approx(0.0)  # Average of [1.0, 0.0, -1.0]
        
        # Fourth reward (window full)
        component.fixed_reward = 2.0
        reward4, _ = aggregator.aggregate(state)
        assert reward4 == pytest.approx(1/3)  # Average of [0.0, -1.0, 2.0]
    
    def test_aggregator_statistics(self):
        """Test aggregator statistics collection."""
        component = MockRewardComponent("test", weight=1.0, fixed_reward=0.5)
        aggregator = RewardAggregator(
            components=[component],
            smoothing_enabled=False
        )
        
        state = self._create_dummy_state()
        
        # Generate multiple rewards
        rewards = []
        for i in range(5):
            component.fixed_reward = i * 0.2
            reward, _ = aggregator.aggregate(state)
            rewards.append(reward)
        
        stats = aggregator.get_statistics()
        
        assert stats["total_calculations"] == 5
        assert stats["mean_reward"] == pytest.approx(np.mean(rewards))
        assert stats["std_reward"] == pytest.approx(np.std(rewards))
        assert stats["min_reward"] == pytest.approx(min(rewards))
        assert stats["max_reward"] == pytest.approx(max(rewards))
        assert len(stats["recent_rewards"]) == 5
    
    def test_aggregator_reset(self):
        """Test aggregator reset functionality."""
        component = MockRewardComponent("test", weight=1.0, fixed_reward=0.5)
        aggregator = RewardAggregator(
            components=[component],
            smoothing_enabled=True,
            smoothing_window=3
        )
        
        state = self._create_dummy_state()
        
        # Generate some rewards
        for _ in range(3):
            aggregator.aggregate(state)
        
        # Verify statistics exist
        stats = aggregator.get_statistics()
        assert stats["total_calculations"] == 3
        assert len(stats["recent_rewards"]) == 3
        
        # Reset
        aggregator.reset()
        
        # Verify statistics cleared
        stats = aggregator.get_statistics()
        assert stats["total_calculations"] == 0
        assert len(stats["recent_rewards"]) == 0
        assert stats["mean_reward"] == 0.0
    
    def test_aggregator_empty_components(self):
        """Test aggregator with no components."""
        aggregator = RewardAggregator(
            components=[],
            smoothing_enabled=False
        )
        
        state = self._create_dummy_state()
        total_reward, details = aggregator.aggregate(state)
        
        assert total_reward == 0.0
        assert details == {}
    
    def _create_dummy_state(self) -> RewardState:
        """Create a minimal RewardState for testing."""
        return RewardState(
            portfolio_state_before={"cash": 10000.0, "position": 0},
            portfolio_state_after_fills={"cash": 10000.0, "position": 0},
            portfolio_state_next={"cash": 10000.0, "position": 0},
            market_state_current={"mid": 10.00},
            market_state_next={"mid": 10.00},
            decoded_action={"action_type": "HOLD"},
            fill_details=[],
            terminated=False,
            termination_reason=None,
            trade_entry=None,
            trade_exit=None,
            current_mae=0.0,
            current_mfe=0.0
        )


class TestRewardComponent:
    """Test base RewardComponent functionality."""
    
    def test_component_anti_hacking_clipping(self):
        """Test reward clipping anti-hacking measure."""
        component = MockRewardComponent("test", weight=1.0, fixed_reward=2.0)
        component.clip_min = -1.0
        component.clip_max = 1.0
        
        state = RewardState(
            portfolio_state_before={}, portfolio_state_after_fills={},
            portfolio_state_next={}, market_state_current={},
            market_state_next={}, decoded_action={}, fill_details=[],
            terminated=False, termination_reason=None,
            trade_entry=None, trade_exit=None,
            current_mae=0.0, current_mfe=0.0
        )
        
        reward, _ = component.calculate(state)
        clipped_reward = component.apply_anti_hacking_measures(reward)
        
        assert clipped_reward == 1.0  # Clipped to max
        
        # Test negative clipping
        component.fixed_reward = -2.0
        reward, _ = component.calculate(state)
        clipped_reward = component.apply_anti_hacking_measures(reward)
        
        assert clipped_reward == -1.0  # Clipped to min
    
    def test_component_exponential_decay(self):
        """Test exponential decay anti-hacking measure."""
        component = MockRewardComponent("test", weight=1.0, fixed_reward=1.0)
        component.exponential_decay_enabled = True
        component.decay_rate = 0.5
        
        reward = 1.0
        
        # Test decay for positive rewards
        decayed = component.apply_anti_hacking_measures(reward)
        assert decayed < reward
        assert decayed == pytest.approx(1.0 - 0.5 * (1.0 - np.exp(-1.0)))
        
        # Test decay for negative rewards (should amplify)
        reward = -1.0
        decayed = component.apply_anti_hacking_measures(reward)
        assert decayed < reward  # More negative
        assert decayed == pytest.approx(-1.0 - 0.5 * (1.0 - np.exp(-1.0)))
    
    def test_component_weight_adjustment(self):
        """Test component weight adjustment."""
        component = MockRewardComponent("test", weight=2.0, fixed_reward=0.5)
        
        # Initial weight
        assert component.weight == 2.0
        
        # Update weight
        component.weight = 3.0
        assert component.weight == 3.0
        
        # Negative weight should be allowed (for penalties)
        component.weight = -1.0
        assert component.weight == -1.0
    
    def test_component_enable_disable(self):
        """Test component enable/disable functionality."""
        component = MockRewardComponent("test", enabled=True)
        
        assert component.enabled == True
        
        component.enabled = False
        assert component.enabled == False
        
        component.enabled = True
        assert component.enabled == True