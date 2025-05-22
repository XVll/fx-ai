#!/usr/bin/env python3
"""
Test script to demonstrate the PROPERLY FIXED dashboard functionality.
This follows the Rich Live documentation pattern exactly.
"""

import time
import logging
from datetime import datetime
from envs.env_dashboard import TradingDashboard


def test_dashboard():
    """Test the properly fixed dashboard following Rich Live docs pattern"""

    print("Starting dashboard test following Rich Live documentation pattern...")
    print("Expected behavior:")
    print("1. Dashboard appears at bottom")
    print("2. All logs appear ABOVE the dashboard")
    print("3. Dashboard updates with new data")
    print("4. Clean separation between logs and dashboard")
    print()

    # Create dashboard
    dashboard = TradingDashboard()
    dashboard.set_symbol("MLGO")
    dashboard.set_initial_capital(25000.0)

    try:
        # Start the dashboard - this sets up Rich Live properly
        dashboard.start()

        # Get logger after dashboard started (it's now configured to use live.console)
        logger = logging.getLogger("test")

        # These should appear ABOVE the dashboard
        logger.info("🚀 Starting test sequence...")
        logger.info("📊 Dashboard should now be visible at the bottom")
        logger.info("📝 These log messages should appear above the dashboard")

        # Test the dashboard's log_message method too
        dashboard.log_message("✅ Dashboard log method test - this appears above", "success")
        dashboard.log_message("⚠️ Warning message test", "warning")
        dashboard.log_message("❌ Error message test", "error")
        dashboard.log_message("ℹ️ Info message test", "info")

        # Simulate trading activity with logs appearing above
        for step in range(30):
            # Create mock data
            mock_info = {
                'step': step,
                'timestamp_iso': datetime.now().isoformat(),
                'episode_cumulative_reward': step * 0.1 - 2.5,
                'reward_step': 0.1 if step % 3 == 0 else -0.05,
                'portfolio_equity': 25000 + (step * 10) - 100,
                'portfolio_cash': 25000 - (step * 50),
                'portfolio_unrealized_pnl': step * 2.5,
                'portfolio_realized_pnl_session_net': step * 1.2,
                'action_decoded': {
                    'type': type('ActionType', (), {'name': ['HOLD', 'BUY', 'SELL'][step % 3]})(),
                    'size_enum': type('SizeEnum', (), {'name': ['SIZE_25', 'SIZE_50', 'SIZE_75'][step % 3]})(),
                    'invalid_reason': 'Test invalid reason' if step % 10 == 0 else None
                },
                'position_MLGO_qty': (step % 100) * 10,
                'position_MLGO_side': ['FLAT', 'LONG', 'SHORT'][step % 3],
                'position_MLGO_avg_entry': 5.25 + (step * 0.01),
                'invalid_actions_total_episode': step // 10,
                'fills_step': [
                    {
                        'order_side': type('OrderSide', (), {'value': ['BUY', 'SELL'][step % 2]})(),
                        'executed_quantity': 100 + step,
                        'executed_price': 5.20 + (step * 0.01)
                    }
                ] if step % 5 == 0 else []
            }

            mock_market_state = {
                'current_price': 5.25 + (step * 0.005),
                'best_bid_price': 5.24 + (step * 0.005),
                'best_ask_price': 5.26 + (step * 0.005),
                'best_bid_size': 1000 + (step * 10),
                'best_ask_size': 1500 + (step * 15),
                'market_session': ['PREMARKET', 'REGULAR', 'POSTMARKET'][step % 3]
            }

            # Update dashboard (this should update the bottom display)
            dashboard.update_state(mock_info, mock_market_state)

            # These logs should appear ABOVE the dashboard
            if step % 10 == 0:
                logger.info(f"📊 Step {step}: Portfolio equity: ${mock_info['portfolio_equity']:.2f}")
            elif step % 7 == 0:
                logger.warning(f"⚠️  Step {step}: High volatility detected")
            elif step % 5 == 0:
                logger.info(
                    f"💰 Step {step}: Trade executed - {mock_info['fills_step'][0]['order_side'].value} {mock_info['fills_step'][0]['executed_quantity']} @ ${mock_info['fills_step'][0]['executed_price']:.2f}")
                # Use dashboard's log method too
                dashboard.log_message(f"Trade logged via dashboard method", "success")
            elif step % 3 == 0:
                logger.debug(f"🔍 Step {step}: Action taken - {mock_info['action_decoded']['type'].name}")
            else:
                logger.info(f"Step {step}: Regular trading step, reward: {mock_info['reward_step']:.3f}")

            # Test episode summary
            if step == 25:
                logger.info("📈 Episode summary incoming...")
                mock_info['episode_summary'] = {
                    'total_reward': mock_info['episode_cumulative_reward'],
                    'steps': step,
                    'session_total_commissions': step * 0.5,
                    'session_total_fees': step * 0.3,
                    'session_total_slippage_cost': step * 0.2,
                    'termination_reason': 'END_OF_SESSION_DATA'
                }
                dashboard.update_state(mock_info, mock_market_state)
                logger.info("✅ Episode completed successfully!")
                dashboard.log_message("🎉 Episode finished!", "success")

            time.sleep(0.3)  # Slower for better visibility

        logger.info("🎉 Dashboard test completed successfully!")
        logger.info("✅ If logs appeared above dashboard, the fix is working!")
        dashboard.log_message("✅ Test completed - logs should be above this dashboard", "success")

        # Keep running to show final state
        logger.info("Press Ctrl+C to exit...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("👋 Dashboard test interrupted by user")
        dashboard.log_message("👋 Test interrupted", "info")
    except Exception as e:
        logger.error(f"❌ Error during dashboard test: {e}")
        dashboard.log_message(f"❌ Error: {e}", "error")
        raise
    finally:
        # Clean shutdown
        logger.info("🛑 Stopping dashboard...")
        dashboard.stop()
        print("\nDashboard test completed.")


if __name__ == "__main__":
    test_dashboard()