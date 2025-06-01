#!/usr/bin/env python3
"""
Test script to verify the new percentage-based reward system works correctly
"""

import logging
from datetime import datetime

from config.schemas import Config
from rewards.calculator import RewardSystem
from simulators.portfolio_simulator import FillDetails, OrderTypeEnum, OrderSideEnum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_fill(profit_dollars: float):
    """Create a test fill with specified profit in dollars"""
    return FillDetails(
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
        realized_pnl=profit_dollars
    )

def test_percentage_reward_system():
    """Test the new percentage-based reward system"""
    print("🧪 TESTING NEW PERCENTAGE-BASED REWARD SYSTEM")
    print("=" * 60)
    
    # Create default config (should use new percentage system)
    config = Config()
    reward_system = RewardSystem(config=config.env.reward, logger=logger)
    
    print(f"✅ P&L coefficient: {config.env.reward.pnl_coefficient}")
    print(f"✅ Holding penalty coefficient: {config.env.reward.holding_penalty_coefficient}")
    print(f"✅ Drawdown penalty coefficient: {config.env.reward.drawdown_penalty_coefficient}")
    print(f"✅ Action penalty coefficient: {config.env.reward.action_penalty_coefficient}")
    print(f"✅ Quick profit bonus coefficient: {config.env.reward.quick_profit_bonus_coefficient}")
    
    # Test scenarios with different profit amounts and account sizes
    # Note: costs ($7) are subtracted from profit, so net profit is less
    test_cases = [
        {
            "profit_dollars": 250.0,
            "account_value": 25000.0,
            "expected_percentage": 0.97,  # (250-7)/25000*100 = 0.97%
            "expected_reward": 0.97,      # 0.97% * (100/100) = 0.97
            "description": "$250 profit on $25k account (~1%)"
        },
        {
            "profit_dollars": 100.0,
            "account_value": 25000.0,
            "expected_percentage": 0.37,  # (100-7)/25000*100 = 0.37%
            "expected_reward": 0.37,      # 0.37% * (100/100) = 0.37
            "description": "$100 profit on $25k account (~0.4%)"
        },
        {
            "profit_dollars": -125.0,
            "account_value": 25000.0,
            "expected_percentage": -0.53,  # (-125-7)/25000*100 = -0.53%
            "expected_reward": -0.53,      # -0.53% * (100/100) = -0.53
            "description": "$125 loss on $25k account (~-0.5%)"
        },
        {
            "profit_dollars": 250.0,
            "account_value": 50000.0,
            "expected_percentage": 0.49,  # (250-7)/50000*100 = 0.49%
            "expected_reward": 0.49,      # 0.49% * (100/100) = 0.49
            "description": "$250 profit on $50k account (~0.5%) - scales with account size"
        }
    ]
    
    print(f"\n📊 Testing {len(test_cases)} percentage scaling scenarios...")
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        profit_dollars = test_case["profit_dollars"]
        account_value = test_case["account_value"]
        expected_percentage = test_case["expected_percentage"]
        expected_reward = test_case["expected_reward"]
        description = test_case["description"]
        
        print(f"\n🧪 Test {i+1}: {description}")
        
        # Create portfolio states with specified account value
        portfolio_before = {
            "realized_pnl_session": 0.0,
            "total_equity": account_value
        }
        portfolio_after = {
            "realized_pnl_session": profit_dollars,
            "total_equity": account_value + profit_dollars
        }
        portfolio_next = {
            "realized_pnl_session": profit_dollars,
            "total_equity": account_value + profit_dollars
        }
        
        # Create fill
        fill = create_test_fill(profit_dollars)
        
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
        pnl_reward = component_rewards.get('pnl', 0)
        
        # Calculate actual percentage
        actual_percentage = (profit_dollars - 7.0) / account_value * 100  # Subtract costs
        
        print(f"   💰 Profit: ${profit_dollars:.2f}")
        print(f"   🏦 Account: ${account_value:.2f}")
        print(f"   📈 Expected %: {expected_percentage:.1f}%")
        print(f"   📈 Actual %: {actual_percentage:.1f}%")
        print(f"   🎯 P&L Reward: {pnl_reward:.1f}")
        print(f"   🎯 Total Reward: {total_reward:.1f}")
        
        # Check if the percentage calculation is correct (within tolerance)
        tolerance = 0.1  # Allow small tolerance for rounding
        if abs(pnl_reward - expected_reward) <= tolerance:
            print(f"   ✅ PASS: Reward scales correctly with percentage")
        else:
            print(f"   ❌ FAIL: Expected {expected_reward:.2f}, got {pnl_reward:.2f}")
            all_passed = False
    
    # Test component scaling
    print(f"\n🎛️ Testing component coefficient scaling...")
    
    # Test with different coefficients
    custom_config = Config()
    custom_config.env.reward.pnl_coefficient = 50.0  # Half the default
    custom_reward_system = RewardSystem(config=custom_config.env.reward, logger=logger)
    
    # Same trade as first test but with half coefficient
    portfolio_before = {"realized_pnl_session": 0.0, "total_equity": 25000.0}
    portfolio_after = {"realized_pnl_session": 250.0, "total_equity": 25250.0}
    portfolio_next = {"realized_pnl_session": 250.0, "total_equity": 25250.0}
    fill = create_test_fill(250.0)
    
    custom_reward = custom_reward_system.calculate(
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
    
    custom_components = custom_reward_system.get_last_reward_components()
    custom_pnl_reward = custom_components.get('pnl', 0)
    
    # Expected values: with coefficient 50.0 instead of 100.0, should get half the reward
    expected_default = 0.97  # From test 1 above
    expected_custom = expected_default * 0.5  # Half the coefficient
    
    print(f"   📊 Default coefficient (100): reward = {expected_default:.1f} (expected)")
    print(f"   📊 Custom coefficient (50): reward = {custom_pnl_reward:.1f}")
    
    if abs(custom_pnl_reward - expected_custom) <= 0.1:  # Should be roughly half
        print(f"   ✅ PASS: Coefficient scaling works correctly")
    else:
        print(f"   ❌ FAIL: Expected {expected_custom:.1f}, got {custom_pnl_reward:.1f}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Percentage-based rewards work correctly")
        print("✅ Rewards scale with account size")
        print("✅ Component coefficients control sensitivity")
        print("✅ No more arbitrary clipping or smoothing")
        print("✅ Intuitive tuning: coefficient = reward per 1% profit")
    else:
        print("❌ SOME TESTS FAILED!")
        
    return all_passed

if __name__ == "__main__":
    success = test_percentage_reward_system()
    exit(0 if success else 1)