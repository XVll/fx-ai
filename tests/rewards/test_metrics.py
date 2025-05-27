"""Tests for reward metrics tracking."""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock

from rewards.metrics import ComponentMetrics, RewardMetricsTracker


class TestComponentMetrics:
    """Test ComponentMetrics class."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = ComponentMetrics("test_component")
        
        assert metrics.name == "test_component"
        assert metrics.call_count == 0
        assert metrics.total_reward == 0.0
        assert metrics.mean_reward == 0.0
        assert metrics.std_reward == 0.0
        assert metrics.min_reward == float('inf')
        assert metrics.max_reward == float('-inf')
        assert len(metrics.recent_rewards) == 0
    
    def test_update_single_reward(self):
        """Test updating with single reward."""
        metrics = ComponentMetrics("test")
        
        metrics.update(0.5, {"detail": "test"})
        
        assert metrics.call_count == 1
        assert metrics.total_reward == 0.5
        assert metrics.mean_reward == 0.5
        assert metrics.std_reward == 0.0
        assert metrics.min_reward == 0.5
        assert metrics.max_reward == 0.5
        assert len(metrics.recent_rewards) == 1
        assert metrics.recent_rewards[0] == 0.5
        assert metrics.last_details == {"detail": "test"}
    
    def test_update_multiple_rewards(self):
        """Test updating with multiple rewards."""
        metrics = ComponentMetrics("test", history_size=5)
        
        rewards = [0.1, -0.2, 0.3, -0.1, 0.5]
        for r in rewards:
            metrics.update(r, {})
        
        assert metrics.call_count == 5
        assert metrics.total_reward == pytest.approx(sum(rewards))
        assert metrics.mean_reward == pytest.approx(np.mean(rewards))
        assert metrics.std_reward == pytest.approx(np.std(rewards))
        assert metrics.min_reward == min(rewards)
        assert metrics.max_reward == max(rewards)
        assert list(metrics.recent_rewards) == rewards
    
    def test_history_size_limit(self):
        """Test that history size is respected."""
        metrics = ComponentMetrics("test", history_size=3)
        
        # Add 5 rewards
        for i in range(5):
            metrics.update(float(i), {})
        
        # Should only keep last 3
        assert len(metrics.recent_rewards) == 3
        assert list(metrics.recent_rewards) == [2.0, 3.0, 4.0]
        
        # But statistics should be for all 5
        assert metrics.call_count == 5
        assert metrics.mean_reward == pytest.approx(2.0)  # mean of 0,1,2,3,4
    
    def test_get_summary(self):
        """Test summary generation."""
        metrics = ComponentMetrics("test")
        
        # Add some rewards
        metrics.update(0.5, {"action": "BUY"})
        metrics.update(-0.2, {"action": "SELL"})
        
        summary = metrics.get_summary()
        
        assert summary["name"] == "test"
        assert summary["call_count"] == 2
        assert summary["total_reward"] == pytest.approx(0.3)
        assert summary["mean_reward"] == pytest.approx(0.15)
        assert summary["std_reward"] == pytest.approx(0.35)
        assert summary["min_reward"] == -0.2
        assert summary["max_reward"] == 0.5
        assert "recent_rewards" in summary
        assert summary["last_details"] == {"action": "SELL"}
    
    def test_reset(self):
        """Test reset functionality."""
        metrics = ComponentMetrics("test")
        
        # Add rewards
        metrics.update(0.5, {})
        metrics.update(0.3, {})
        
        # Verify data exists
        assert metrics.call_count == 2
        assert metrics.total_reward == 0.8
        
        # Reset
        metrics.reset()
        
        # Verify reset
        assert metrics.call_count == 0
        assert metrics.total_reward == 0.0
        assert metrics.mean_reward == 0.0
        assert metrics.std_reward == 0.0
        assert metrics.min_reward == float('inf')
        assert metrics.max_reward == float('-inf')
        assert len(metrics.recent_rewards) == 0
        assert metrics.last_details is None


class TestRewardMetricsTracker:
    """Test RewardMetricsTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        logger = Mock()
        return RewardMetricsTracker(logger)
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.episode_count == 0
        assert len(tracker.component_metrics) == 0
        assert len(tracker.episode_rewards) == 0
        assert len(tracker.action_rewards) == 0
    
    def test_register_component(self, tracker):
        """Test component registration."""
        tracker.register_component("pnl", "FOUNDATIONAL")
        tracker.register_component("holding_penalty", "SHAPING")
        
        assert len(tracker.component_metrics) == 2
        assert "pnl" in tracker.component_metrics
        assert "holding_penalty" in tracker.component_metrics
        assert tracker.component_metrics["pnl"].name == "pnl"
    
    def test_update_metrics(self, tracker):
        """Test metrics update."""
        # Register components
        tracker.register_component("comp1", "FOUNDATIONAL")
        tracker.register_component("comp2", "SHAPING")
        
        # Update with reward details
        component_rewards = {
            "comp1": {"reward": 0.5, "details": {"pnl": 50}},
            "comp2": {"reward": -0.1, "details": {"penalty": 0.1}}
        }
        total_reward = 0.4
        action = "BUY"
        
        tracker.update(total_reward, component_rewards, action)
        
        # Verify episode rewards
        assert len(tracker.episode_rewards) == 1
        assert tracker.episode_rewards[-1] == 0.4
        
        # Verify component metrics
        assert tracker.component_metrics["comp1"].call_count == 1
        assert tracker.component_metrics["comp1"].total_reward == 0.5
        assert tracker.component_metrics["comp2"].call_count == 1
        assert tracker.component_metrics["comp2"].total_reward == -0.1
        
        # Verify action rewards
        assert action in tracker.action_rewards
        assert len(tracker.action_rewards[action]) == 1
        assert tracker.action_rewards[action][-1] == 0.4
    
    def test_multiple_updates(self, tracker):
        """Test multiple metric updates."""
        tracker.register_component("comp1", "FOUNDATIONAL")
        
        # Multiple updates
        for i in range(5):
            component_rewards = {
                "comp1": {"reward": i * 0.1, "details": {}}
            }
            action = "BUY" if i % 2 == 0 else "SELL"
            tracker.update(i * 0.1, component_rewards, action)
        
        # Verify counts
        assert len(tracker.episode_rewards) == 5
        assert tracker.component_metrics["comp1"].call_count == 5
        assert len(tracker.action_rewards["BUY"]) == 3
        assert len(tracker.action_rewards["SELL"]) == 2
    
    def test_get_episode_summary(self, tracker):
        """Test episode summary generation."""
        # Setup
        tracker.register_component("pnl", "FOUNDATIONAL")
        tracker.register_component("penalty", "SHAPING")
        
        # Add some data
        for i in range(10):
            rewards = {
                "pnl": {"reward": i * 0.1, "details": {}},
                "penalty": {"reward": -i * 0.01, "details": {}}
            }
            action = ["BUY", "SELL", "HOLD"][i % 3]
            total = i * 0.1 - i * 0.01
            tracker.update(total, rewards, action)
        
        # Get summary
        summary = tracker.get_episode_summary()
        
        assert summary["episode_number"] == 0
        assert summary["total_steps"] == 10
        assert "total_reward" in summary
        assert "mean_reward" in summary
        assert "component_summaries"] in summary
        assert len(summary["component_summaries"]) == 2
        assert "action_statistics" in summary
        assert len(summary["action_statistics"]) == 3
    
    def test_reset_episode(self, tracker):
        """Test episode reset."""
        # Setup with data
        tracker.register_component("test", "FOUNDATIONAL")
        tracker.update(0.5, {"test": {"reward": 0.5, "details": {}}}, "BUY")
        
        # Verify data exists
        assert len(tracker.episode_rewards) == 1
        assert tracker.episode_count == 0
        
        # Reset
        tracker.reset_episode()
        
        # Verify reset
        assert len(tracker.episode_rewards) == 0
        assert tracker.episode_count == 1
        assert len(tracker.action_rewards) == 0
        # Component metrics should NOT be reset
        assert tracker.component_metrics["test"].call_count == 1
    
    def test_get_component_correlations(self, tracker):
        """Test component correlation calculation."""
        # Register components
        tracker.register_component("comp1", "FOUNDATIONAL")
        tracker.register_component("comp2", "SHAPING")
        
        # Add correlated data
        for i in range(20):
            rewards = {
                "comp1": {"reward": i * 0.1, "details": {}},
                "comp2": {"reward": i * 0.05, "details": {}}  # Perfectly correlated
            }
            tracker.update(i * 0.15, rewards, "HOLD")
        
        correlations = tracker.get_component_correlations()
        
        assert isinstance(correlations, dict)
        # Should have correlation between comp1 and comp2
        assert ("comp1", "comp2") in correlations or ("comp2", "comp1") in correlations
        # Perfect correlation should be close to 1.0
        corr_value = correlations.get(("comp1", "comp2"), correlations.get(("comp2", "comp1")))
        assert corr_value == pytest.approx(1.0, abs=0.01)
    
    def test_get_action_profitability(self, tracker):
        """Test action profitability analysis."""
        # Add rewards for different actions
        tracker.update(1.0, {}, "BUY")
        tracker.update(0.5, {}, "BUY")
        tracker.update(-0.5, {}, "SELL")
        tracker.update(-0.2, {}, "SELL")
        tracker.update(0.0, {}, "HOLD")
        tracker.update(0.1, {}, "HOLD")
        
        profitability = tracker.get_action_profitability()
        
        assert len(profitability) == 3
        assert profitability["BUY"]["count"] == 2
        assert profitability["BUY"]["mean_reward"] == pytest.approx(0.75)
        assert profitability["BUY"]["total_reward"] == 1.5
        
        assert profitability["SELL"]["count"] == 2
        assert profitability["SELL"]["mean_reward"] == pytest.approx(-0.35)
        
        assert profitability["HOLD"]["count"] == 2
        assert profitability["HOLD"]["mean_reward"] == pytest.approx(0.05)
    
    def test_empty_tracker_methods(self, tracker):
        """Test methods on empty tracker."""
        # Should not crash on empty data
        summary = tracker.get_episode_summary()
        assert summary["total_steps"] == 0
        assert summary["total_reward"] == 0.0
        
        correlations = tracker.get_component_correlations()
        assert correlations == {}
        
        profitability = tracker.get_action_profitability()
        assert profitability == {}
    
    def test_history_limits(self, tracker):
        """Test that history limits are respected."""
        tracker.register_component("test", "FOUNDATIONAL")
        
        # Add more than history limit
        for i in range(1500):  # More than default 1000
            tracker.update(0.1, {"test": {"reward": 0.1, "details": {}}}, "HOLD")
        
        # Episode rewards should be limited
        assert len(tracker.episode_rewards) <= 1000
        
        # Action rewards should also be limited
        assert len(tracker.action_rewards["HOLD"]) <= 1000