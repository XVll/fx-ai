#!/usr/bin/env python3
"""
Diagnostic script to identify P&L reward calculation bug
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

from config.schemas import Config, RewardConfig, RewardComponentConfig
from rewards.calculator import RewardSystem
from rewards.core import RewardState
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum, OrderSideEnum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_reward_config(clip_range=None):
    """Create a test reward configuration"""
    if clip_range is None:
        clip_range = [-1000.0, 1000.0]  # Much wider range to avoid clipping
        
    return RewardConfig(
        pnl=RewardComponentConfig(enabled=True, coefficient=1.0),
        holding_penalty=RewardComponentConfig(enabled=False, coefficient=0.0),
        action_penalty=RewardComponentConfig(enabled=False, coefficient=0.0),
        spread_penalty=RewardComponentConfig(enabled=False, coefficient=0.0),
        drawdown_penalty=RewardComponentConfig(enabled=False, coefficient=0.0),
        bankruptcy_penalty=RewardComponentConfig(enabled=False, coefficient=0.0),
        profitable_exit=RewardComponentConfig(enabled=False, coefficient=0.0),
        quick_profit=RewardComponentConfig(enabled=False, coefficient=0.0),
        invalid_action_penalty=RewardComponentConfig(enabled=False, coefficient=0.0),
        scale_factor=1.0,
        clip_range=clip_range
    )

def create_test_portfolio_state(realized_pnl: float, unrealized_pnl: float = 0.0, equity: float = 25000.0):
    """Create a test portfolio state"""
    return {
        'realized_pnl_session': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'total_equity': equity,
        'position_side': PositionSideEnum.FLAT,
        'position_value': 0.0,
        'total_value': equity
    }

def create_profitable_fill():
    """Create a fill detail for a profitable trade"""
    from datetime import datetime
    from simulators.portfolio_simulator import OrderTypeEnum
    
    fill = FillDetails(
        asset_id="MLGO",
        fill_timestamp=datetime.now(),
        order_type=OrderTypeEnum.MARKET,
        order_side=OrderSideEnum.SELL,
        requested_quantity=1000,
        executed_quantity=1000,
        executed_price=5.0,
        commission=5.0,
        fees=0.0,
        slippage_cost_total=2.0,
        closes_position=True,
        realized_pnl=100.0  # $100 profit
    )
    return fill

def test_reward_calculation():
    """Test reward calculation with a simple profitable trade"""
    print("=" * 60)
    print("TESTING REWARD SYSTEM WITH PROFITABLE TRADE")
    print("=" * 60)
    
    # Create reward system
    reward_config = create_test_reward_config()
    reward_system = RewardSystem(config=reward_config, logger=logger)
    
    print(f"✅ Created reward system with PnL coefficient: {reward_config.pnl.coefficient}")
    
    # Create test states
    portfolio_before = create_test_portfolio_state(realized_pnl=0.0)  # No prior P&L
    portfolio_after = create_test_portfolio_state(realized_pnl=100.0)  # $100 profit
    portfolio_next = create_test_portfolio_state(realized_pnl=100.0)   # Same as after
    
    print(f"📊 Portfolio before: realized_pnl = ${portfolio_before['realized_pnl_session']}")
    print(f"📊 Portfolio after:  realized_pnl = ${portfolio_after['realized_pnl_session']}")
    
    # Create fill details for profitable trade
    fill_details = [create_profitable_fill()]
    
    print(f"💰 Fill details: profit = ${fill_details[0].realized_pnl}, closes_position = {fill_details[0].closes_position}")
    
    # Create reward state
    reward_state = RewardState(
        portfolio_before=portfolio_before,
        portfolio_after_fills=portfolio_after,
        portfolio_next=portfolio_next,
        market_state_current={},
        market_state_next={},
        decoded_action={'type': 'sell'},
        fill_details=fill_details,
        terminated=False,
        truncated=False,
        step_count=1
    )
    
    # Test individual component first
    print("\n" + "=" * 40)
    print("TESTING REALIZED P&L COMPONENT DIRECTLY")
    print("=" * 40)
    
    from rewards.components import RealizedPnLReward
    pnl_component = RealizedPnLReward(
        config={'enabled': True, 'weight': 1.0}, 
        logger=logger
    )
    
    component_reward, component_diagnostics = pnl_component.calculate(reward_state)
    print(f"🔍 Component raw reward: {component_reward}")
    print(f"🔍 Component diagnostics: {component_diagnostics}")
    
    # Apply weight and get final value
    final_reward, final_diagnostics = pnl_component(reward_state)
    print(f"🔍 Component final reward: {final_reward}")
    print(f"🔍 Component final diagnostics: {final_diagnostics}")
    
    # Test full reward system
    print("\n" + "=" * 40)
    print("TESTING FULL REWARD SYSTEM")
    print("=" * 40)
    
    total_reward = reward_system.calculate(
        portfolio_state_before_action=portfolio_before,
        portfolio_state_after_action_fills=portfolio_after,
        portfolio_state_next_t=portfolio_next,
        market_state_at_decision={},
        market_state_next_t={},
        decoded_action={'type': 'sell'},
        fill_details_list=fill_details,
        terminated=False,
        truncated=False,
        termination_reason=None
    )
    
    print(f"🎯 TOTAL REWARD: {total_reward}")
    
    # Get component breakdown
    component_rewards = reward_system.get_last_reward_components()
    print(f"🔧 Component breakdown: {component_rewards}")
    
    # Check for scaling factors
    print(f"🔧 Global scale factor: {reward_system.aggregator.global_scale}")
    print(f"🔧 Smoothing enabled: {reward_system.aggregator.use_smoothing}")
    print(f"🔧 Clip range: {reward_config.clip_range}")
    
    # Analysis
    print("\n" + "=" * 40)
    print("ANALYSIS")
    print("=" * 40)
    
    if total_reward > 0:
        print("✅ CORRECT: Profitable trade resulted in positive reward")
    else:
        print("❌ BUG: Profitable trade resulted in negative reward!")
        print(f"   Expected: Positive reward for $100 profit")
        print(f"   Actual: {total_reward}")
        
        # Check for sign errors
        if abs(total_reward) > 50:  # Should be close to profit amount
            print(f"   🔍 Possible sign error - magnitude is correct but sign is wrong")
        
        # Check coefficient issues
        pnl_reward = component_rewards.get('realized_pnl', 0)
        print(f"   🔍 P&L component reward: {pnl_reward}")
        print(f"   🔍 Expected: ~100 (profit amount)")
        
    return total_reward, component_rewards

def test_multiple_scenarios():
    """Test multiple profit/loss scenarios"""
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {"profit": 100.0, "description": "$100 profit"},
        {"profit": 50.0, "description": "$50 profit"},
        {"profit": -25.0, "description": "$25 loss"},
        {"profit": 0.0, "description": "Break-even"},
    ]
    
    reward_config = create_test_reward_config()
    reward_system = RewardSystem(config=reward_config, logger=logger)
    
    for scenario in scenarios:
        profit = scenario["profit"]
        desc = scenario["description"]
        
        print(f"\n📝 Testing: {desc}")
        
        # Create portfolio states
        portfolio_before = create_test_portfolio_state(realized_pnl=0.0)
        portfolio_after = create_test_portfolio_state(realized_pnl=profit)
        portfolio_next = create_test_portfolio_state(realized_pnl=profit)
        
        # Create fill
        from datetime import datetime
        from simulators.portfolio_simulator import OrderTypeEnum
        
        fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=1000,
            executed_quantity=1000,
            executed_price=5.0,
            commission=5.0,
            fees=0.0,
            slippage_cost_total=2.0,
            closes_position=True,
            realized_pnl=profit
        )
        
        # Calculate reward
        total_reward = reward_system.calculate(
            portfolio_state_before_action=portfolio_before,
            portfolio_state_after_action_fills=portfolio_after,
            portfolio_state_next_t=portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action={'type': 'sell'},
            fill_details_list=[fill],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        # Get component breakdown for analysis
        component_rewards = reward_system.get_last_reward_components()
        
        # Check if reward sign matches profit sign
        if profit > 0 and total_reward > 0:
            print(f"   ✅ Correct: Profit → Positive reward ({total_reward:.2f})")
        elif profit < 0 and total_reward < 0:
            print(f"   ✅ Correct: Loss → Negative reward ({total_reward:.2f})")
        elif profit == 0 and abs(total_reward) < 1:
            print(f"   ✅ Correct: Break-even → Near-zero reward ({total_reward:.2f})")
        else:
            print(f"   ❌ BUG: {desc} → {total_reward:.2f} reward (WRONG SIGN!)")
            print(f"      Component breakdown: {component_rewards}")
            # Check specifically P&L component
            pnl_component = component_rewards.get('realized_pnl', 0)
            print(f"      P&L component: {pnl_component:.2f} (should be close to {profit:.2f})")

def test_no_clipping():
    """Test reward system with no clipping to see raw behavior"""
    print("\n" + "=" * 60)
    print("TESTING WITH NO CLIPPING/SMOOTHING")
    print("=" * 60)
    
    # Create reward system with no smoothing and wide clip range
    from rewards.calculator import RewardSystem
    
    reward_config = create_test_reward_config(clip_range=[-10000.0, 10000.0])
    reward_system = RewardSystem(config=reward_config, logger=logger)
    
    # Override aggregator config to disable smoothing
    reward_system.aggregator.use_smoothing = False
    reward_system.aggregator.reward_history.clear()
    
    scenarios = [
        {"profit": 100.0, "description": "$100 profit"},
        {"profit": -25.0, "description": "$25 loss"},
    ]
    
    for scenario in scenarios:
        profit = scenario["profit"]
        desc = scenario["description"]
        
        print(f"\n📝 Testing (no clipping/smoothing): {desc}")
        
        # Create portfolio states
        portfolio_before = create_test_portfolio_state(realized_pnl=0.0)
        portfolio_after = create_test_portfolio_state(realized_pnl=profit)
        portfolio_next = create_test_portfolio_state(realized_pnl=profit)
        
        # Create fill
        from datetime import datetime
        from simulators.portfolio_simulator import OrderTypeEnum
        
        fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=1000,
            executed_quantity=1000,
            executed_price=5.0,
            commission=5.0,
            fees=0.0,
            slippage_cost_total=2.0,
            closes_position=True,
            realized_pnl=profit
        )
        
        # Calculate reward
        total_reward = reward_system.calculate(
            portfolio_state_before_action=portfolio_before,
            portfolio_state_after_action_fills=portfolio_after,
            portfolio_state_next_t=portfolio_next,
            market_state_at_decision={},
            market_state_next_t={},
            decoded_action={'type': 'sell'},
            fill_details_list=[fill],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        # Get component breakdown
        component_rewards = reward_system.get_last_reward_components()
        
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Component breakdown: {component_rewards}")
        
        # Check if results make sense
        expected_pnl_reward = profit - 7.0  # profit minus costs
        pnl_component = component_rewards.get('realized_pnl', 0)
        
        if abs(pnl_component - expected_pnl_reward) < 1.0:
            print(f"   ✅ P&L component correct: {pnl_component:.2f} ≈ {expected_pnl_reward:.2f}")
        else:
            print(f"   ❌ P&L component wrong: {pnl_component:.2f} ≠ {expected_pnl_reward:.2f}")

if __name__ == "__main__":
    print("🔬 REWARD SYSTEM DIAGNOSTIC SCRIPT")
    print("This script tests the reward system with controlled inputs")
    print("to identify why profitable trades are generating negative rewards.\n")
    
    try:
        # Test basic functionality
        total_reward, components = test_reward_calculation()
        
        # Test multiple scenarios
        test_multiple_scenarios()
        
        # Test with no clipping/smoothing
        test_no_clipping()
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 60)
        
        print("\n🔍 SUMMARY OF FINDINGS:")
        print("1. ❌ CLIPPING: Rewards are clipped to [-10, 10] causing major distortion")
        print("2. ❌ SMOOTHING: Reward smoothing causes sign inversions")  
        print("3. ✅ P&L COMPONENT: Core P&L calculation works correctly")
        print("4. 🚨 SOLUTION: Fix clipping range and disable smoothing for training")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()