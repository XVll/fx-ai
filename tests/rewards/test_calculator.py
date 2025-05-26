"""Tests for RewardSystemV2 calculator."""

import pytest
import logging
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

from rewards.calculator import RewardSystemV2


class TestRewardSystemV2:
    """Test RewardSystemV2 main calculator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.env.reward_v2.pnl.enabled = True
        config.env.reward_v2.pnl.coefficient = 1.0
        config.env.reward_v2.holding_penalty.enabled = True
        config.env.reward_v2.holding_penalty.coefficient = 0.001
        config.env.reward_v2.drawdown_penalty.enabled = True
        config.env.reward_v2.drawdown_penalty.coefficient = 0.1
        return config
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def reward_system(self, mock_config, mock_logger):
        """Create RewardSystemV2 instance."""
        return RewardSystemV2(
            config=mock_config,
            metrics_integrator=None,
            logger=mock_logger
        )
    
    def test_initialization(self, reward_system):
        """Test system initialization."""
        assert reward_system.step_count == 0
        assert reward_system.current_trade is None
        assert reward_system.episode_started is False
        assert len(reward_system.components) > 0
        assert reward_system.aggregator is not None
        assert reward_system.metrics_tracker is not None
    
    def test_reset(self, reward_system):
        """Test reset functionality."""
        # Modify state
        reward_system.step_count = 10
        reward_system.current_trade = {"entry": 100}
        reward_system.episode_started = True
        
        # Reset
        reward_system.reset()
        
        # Verify reset
        assert reward_system.step_count == 0
        assert reward_system.current_trade is None
        assert reward_system.episode_started is True
    
    def test_calculate_first_step(self, reward_system):
        """Test reward calculation on first step."""
        reward_system.reset()
        
        # Create test states
        portfolio_before = {
            "cash": 10000.0,
            "position": 0,
            "total_value": 10000.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0
        }
        portfolio_after = {
            "cash": 9000.0,
            "position": 100,
            "total_value": 10000.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0
        }
        portfolio_next = {
            "cash": 9000.0,
            "position": 100,
            "total_value": 10050.0,
            "unrealized_pnl": 50.0,
            "realized_pnl": 0.0
        }
        market_current = {"bid": 9.99, "ask": 10.01, "mid": 10.00}
        market_next = {"bid": 10.04, "ask": 10.06, "mid": 10.05}
        decoded_action = {"action_type": "BUY", "position_size": 0.5}
        fill_details = [{
            "timestamp": datetime.now(),
            "action": "BUY",
            "quantity": 100,
            "price": 10.01,
            "commission": 1.0
        }]
        
        reward = reward_system.calculate(
            portfolio_state_before=portfolio_before,
            portfolio_state_after_fills=portfolio_after,
            portfolio_state_next=portfolio_next,
            market_state_at_decision=market_current,
            market_state_next=market_next,
            decoded_action=decoded_action,
            fill_details=fill_details,
            terminated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
        assert reward_system.step_count == 1
    
    def test_calculate_with_position_exit(self, reward_system):
        """Test reward calculation when exiting position."""
        reward_system.reset()
        reward_system.current_trade = {
            "entry_price": 10.00,
            "entry_time": datetime.now(),
            "mae": -0.50,
            "mfe": 1.00
        }
        
        # Exit trade with profit
        portfolio_before = {
            "cash": 9000.0,
            "position": 100,
            "total_value": 10100.0,
            "unrealized_pnl": 100.0,
            "realized_pnl": 0.0
        }
        portfolio_after = {
            "cash": 10099.0,  # Sold at 10.10, minus commission
            "position": 0,
            "total_value": 10099.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0
        }
        portfolio_next = {
            "cash": 10099.0,
            "position": 0,
            "total_value": 10099.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 99.0
        }
        market_current = {"bid": 10.09, "ask": 10.11, "mid": 10.10}
        market_next = {"bid": 10.09, "ask": 10.11, "mid": 10.10}
        decoded_action = {"action_type": "SELL", "position_size": 1.0}
        fill_details = [{
            "timestamp": datetime.now(),
            "action": "SELL",
            "quantity": 100,
            "price": 10.10,
            "commission": 1.0
        }]
        
        reward = reward_system.calculate(
            portfolio_state_before=portfolio_before,
            portfolio_state_after_fills=portfolio_after,
            portfolio_state_next=portfolio_next,
            market_state_at_decision=market_current,
            market_state_next=market_next,
            decoded_action=decoded_action,
            fill_details=fill_details,
            terminated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
        assert reward_system.current_trade is None  # Trade closed
    
    def test_calculate_with_termination(self, reward_system):
        """Test reward calculation on termination."""
        reward_system.reset()
        
        portfolio_state = {
            "cash": 100.0,
            "position": 0,
            "total_value": 100.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": -9900.0
        }
        market_state = {"bid": 10.00, "ask": 10.02, "mid": 10.01}
        decoded_action = {"action_type": "BUY", "position_size": 1.0}
        
        reward = reward_system.calculate(
            portfolio_state_before=portfolio_state,
            portfolio_state_after_fills=portfolio_state,
            portfolio_state_next=portfolio_state,
            market_state_at_decision=market_state,
            market_state_next=market_state,
            decoded_action=decoded_action,
            fill_details=[],
            terminated=True,
            termination_reason="bankruptcy"
        )
        
        assert isinstance(reward, float)
        # Should include terminal penalty
        assert reward < 0
    
    def test_get_episode_summary(self, reward_system):
        """Test episode summary generation."""
        reward_system.reset()
        
        # Simulate a few steps
        for i in range(3):
            portfolio = {"cash": 10000 - i*100, "position": i*50,
                        "total_value": 10000, "unrealized_pnl": 0, "realized_pnl": 0}
            market = {"mid": 10.0 + i*0.1}
            action = {"action_type": "BUY" if i % 2 == 0 else "HOLD"}
            
            reward_system.calculate(
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
        
        summary = reward_system.get_episode_summary()
        
        assert isinstance(summary, dict)
        assert "total_steps" in summary
        assert summary["total_steps"] == 3
        assert "component_stats" in summary
        assert "action_distribution" in summary
    
    def test_get_metrics_for_dashboard(self, reward_system):
        """Test dashboard metrics generation."""
        reward_system.reset()
        
        # Generate a reward
        portfolio = {"cash": 10000, "position": 0, "total_value": 10000,
                    "unrealized_pnl": 0, "realized_pnl": 0}
        market = {"mid": 10.0}
        action = {"action_type": "HOLD"}
        
        reward_system.calculate(
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
        
        metrics = reward_system.get_metrics_for_dashboard()
        
        assert isinstance(metrics, dict)
        assert "total_reward" in metrics
        assert "component_rewards" in metrics
        assert "aggregator_stats" in metrics
        assert "tracker_stats" in metrics
    
    def test_get_wandb_metrics(self, reward_system):
        """Test W&B metrics generation."""
        reward_system.reset()
        
        # Generate some rewards
        for _ in range(5):
            portfolio = {"cash": 10000, "position": 0, "total_value": 10000,
                        "unrealized_pnl": 0, "realized_pnl": 0}
            market = {"mid": 10.0}
            action = {"action_type": "HOLD"}
            
            reward_system.calculate(
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
        
        metrics = reward_system.get_wandb_metrics()
        
        assert isinstance(metrics, dict)
        assert all(key.startswith("reward/") for key in metrics.keys())
    
    def test_mae_mfe_tracking(self, reward_system):
        """Test MAE/MFE tracking during trade."""
        reward_system.reset()
        
        # Enter position
        portfolio_before = {"cash": 10000, "position": 0, "total_value": 10000,
                           "unrealized_pnl": 0, "realized_pnl": 0}
        portfolio_after = {"cash": 9000, "position": 100, "total_value": 10000,
                          "unrealized_pnl": 0, "realized_pnl": 0}
        market = {"mid": 10.0}
        action = {"action_type": "BUY"}
        fill_details = [{"action": "BUY", "quantity": 100, "price": 10.0}]
        
        reward_system.calculate(
            portfolio_state_before=portfolio_before,
            portfolio_state_after_fills=portfolio_after,
            portfolio_state_next=portfolio_after,
            market_state_at_decision=market,
            market_state_next=market,
            decoded_action=action,
            fill_details=fill_details,
            terminated=False,
            termination_reason=None
        )
        
        assert reward_system.current_trade is not None
        assert reward_system.current_trade["entry_price"] == 10.0
        
        # Update with unrealized loss (MAE)
        portfolio_loss = {"cash": 9000, "position": 100, "total_value": 9950,
                         "unrealized_pnl": -50, "realized_pnl": 0}
        
        reward_system.calculate(
            portfolio_state_before=portfolio_after,
            portfolio_state_after_fills=portfolio_loss,
            portfolio_state_next=portfolio_loss,
            market_state_at_decision=market,
            market_state_next={"mid": 9.5},
            decoded_action={"action_type": "HOLD"},
            fill_details=[],
            terminated=False,
            termination_reason=None
        )
        
        assert reward_system.current_trade["mae"] == -50
        
        # Update with unrealized gain (MFE)
        portfolio_gain = {"cash": 9000, "position": 100, "total_value": 10200,
                         "unrealized_pnl": 200, "realized_pnl": 0}
        
        reward_system.calculate(
            portfolio_state_before=portfolio_loss,
            portfolio_state_after_fills=portfolio_gain,
            portfolio_state_next=portfolio_gain,
            market_state_at_decision={"mid": 9.5},
            market_state_next={"mid": 12.0},
            decoded_action={"action_type": "HOLD"},
            fill_details=[],
            terminated=False,
            termination_reason=None
        )
        
        assert reward_system.current_trade["mfe"] == 200
        assert reward_system.current_trade["mae"] == -50  # MAE doesn't change
    
    def test_invalid_states_handling(self, reward_system):
        """Test handling of invalid or missing states."""
        reward_system.reset()
        
        # Test with None values in states
        portfolio = {"cash": None, "position": 0, "total_value": 10000,
                    "unrealized_pnl": 0, "realized_pnl": 0}
        market = {"mid": None}
        action = {"action_type": "HOLD"}
        
        # Should handle gracefully without crashing
        try:
            reward = reward_system.calculate(
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
            # Should return some reward (likely 0 or penalty)
            assert isinstance(reward, float)
        except Exception as e:
            # If it raises, should be a specific handled exception
            assert False, f"Unexpected exception: {e}"
    
    def test_empty_fill_details(self, reward_system):
        """Test calculation with empty fill details."""
        reward_system.reset()
        
        portfolio = {"cash": 10000, "position": 0, "total_value": 10000,
                    "unrealized_pnl": 0, "realized_pnl": 0}
        market = {"mid": 10.0}
        action = {"action_type": "BUY"}  # Buy action but no fills
        
        reward = reward_system.calculate(
            portfolio_state_before=portfolio,
            portfolio_state_after_fills=portfolio,  # No change
            portfolio_state_next=portfolio,
            market_state_at_decision=market,
            market_state_next=market,
            decoded_action=action,
            fill_details=[],  # Empty
            terminated=False,
            termination_reason=None
        )
        
        assert isinstance(reward, float)
        # No position change, so current_trade should remain None
        assert reward_system.current_trade is None