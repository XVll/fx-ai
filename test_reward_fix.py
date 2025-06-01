#!/usr/bin/env python3
"""
Test script to verify the reward system fixes work correctly
"""

import logging
from datetime import datetime

from config.schemas import Config
from rewards.calculator import RewardSystem
from simulators.portfolio_simulator import FillDetails, OrderTypeEnum, OrderSideEnum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_fill(profit: float):
    """Create a test fill with specified profit"""
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
        realized_pnl=profit
    )

def test_reward_fixes():
    """Test that the reward system fixes work correctly"""
    print("🧪 TESTING REWARD SYSTEM FIXES")
    print("=" * 50)
    
    # Create default config (should now have fixed settings)
    config = Config()
    reward_system = RewardSystem(config=config.env.reward, logger=logger)
    
    print(f"✅ Clipping range: {config.env.reward.clip_range}")
    print(f"✅ Smoothing enabled: {reward_system.aggregator.use_smoothing}")
    
    # Test scenarios
    test_cases = [
        {"profit": 100.0, "expected_sign": 1, "description": "$100 profit"},
        {"profit": 50.0, "expected_sign": 1, "description": "$50 profit"}, 
        {"profit": -25.0, "expected_sign": -1, "description": "$25 loss"},
        {"profit": -50.0, "expected_sign": -1, "description": "$50 loss"},
        {"profit": 0.0, "expected_sign": -1, "description": "Break-even (costs only)"},
    ]
    
    print(f"\n📊 Testing {len(test_cases)} scenarios...")
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        profit = test_case["profit"]
        expected_sign = test_case["expected_sign"]
        description = test_case["description"]
        
        # Create portfolio states
        portfolio_before = {"realized_pnl_session": 0.0}
        portfolio_after = {"realized_pnl_session": profit}
        portfolio_next = {"realized_pnl_session": profit}
        
        # Create fill
        fill = create_test_fill(profit)
        
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
        
        # Check result
        actual_sign = 1 if total_reward > 0 else -1 if total_reward < 0 else 0
        
        if expected_sign == actual_sign:
            print(f"   ✅ {description}: {total_reward:.2f} (correct sign)")
        else:
            print(f"   ❌ {description}: {total_reward:.2f} (WRONG SIGN! Expected {expected_sign}, got {actual_sign})")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Reward system fixes are working correctly.")
        print("✅ Profitable trades now generate positive rewards")
        print("✅ Losing trades now generate negative rewards")
        print("✅ No more sign inversions from smoothing")
        print("✅ No more reward distortion from tight clipping")
    else:
        print("❌ SOME TESTS FAILED! Reward system still has issues.")
        
    return all_passed

if __name__ == "__main__":
    success = test_reward_fixes()
    exit(0 if success else 1)