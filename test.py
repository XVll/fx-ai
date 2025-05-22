#!/usr/bin/env python3
"""
Test script for the trading dashboard.
This demonstrates how to use the dashboard independently or with the trading environment.
"""

import time
import random
from datetime import datetime

from envs.env_dashboard import TradingDashboard
from simulators.portfolio_simulator import OrderSideEnum


def demo_dashboard():
    """Demonstrate the dashboard with simulated trading data"""

    # Create dashboard
    dashboard = TradingDashboard(update_frequency=0.1)
    dashboard.set_symbol("MLGO")
    dashboard.set_initial_capital(25000.0)

    try:
        # Start the dashboard
        dashboard.start()
        print("Dashboard started! Watch the live display...")

        # Simulate some trading activity
        for step in range(1, 101):
            # Simulate market data
            base_price = 5.00
            price_change = random.uniform(-0.05, 0.05)
            current_price = base_price + price_change

            # Create mock info dict (like what trading env would provide)
            info_dict = {
                'step': step,
                'timestamp_iso': datetime.now().isoformat(),
                'reward_step': random.uniform(-0.1, 0.1),
                'episode_cumulative_reward': random.uniform(-5, 5),
                'portfolio_equity': 25000 + random.uniform(-1000, 1000),
                'portfolio_cash': 20000 + random.uniform(-5000, 5000),
                'portfolio_unrealized_pnl': random.uniform(-500, 500),
                'portfolio_realized_pnl_session_net': random.uniform(-200, 200),
                'position_MLGO_qty': random.uniform(0, 1000),
                'position_MLGO_side': random.choice(['FLAT', 'LONG', 'SHORT']),
                'position_MLGO_avg_entry': current_price + random.uniform(-0.1, 0.1),
                'invalid_actions_total_episode': random.randint(0, 5)
            }

            # Add action info occasionally
            if step % 5 == 0:
                info_dict['action_decoded'] = {
                    'type': type('ActionType', (), {'name': random.choice(['HOLD', 'BUY', 'SELL'])})(),
                    'size_enum': type('SizeEnum', (),
                                      {'name': random.choice(['SIZE_25', 'SIZE_50', 'SIZE_75', 'SIZE_100'])})(),
                    'invalid_reason': None if random.random() > 0.2 else "Test invalid reason"
                }

            # Add fills occasionally
            if step % 7 == 0:
                info_dict['fills_step'] = [{
                    'order_side': random.choice([OrderSideEnum.BUY, OrderSideEnum.SELL]),
                    'executed_quantity': random.uniform(10, 100),
                    'executed_price': current_price,
                    'commission': random.uniform(0.1, 2.0),
                    'fees': random.uniform(0.5, 5.0),
                    'slippage_cost_total': random.uniform(0.1, 1.0)
                }]

            # Create mock market state
            spread = 0.01
            market_state = {
                'current_price': current_price,
                'best_bid_price': current_price - spread / 2,
                'best_ask_price': current_price + spread / 2,
                'best_bid_size': random.uniform(100, 1000),
                'best_ask_size': random.uniform(100, 1000),
                'market_session': random.choice(['PREMARKET', 'REGULAR', 'POSTMARKET'])
            }

            # Update dashboard
            dashboard.update_state(info_dict, market_state)

            # Sleep to simulate real trading pace
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        # Stop dashboard
        dashboard.stop()
        print("Dashboard stopped")


if __name__ == "__main__":
    print("Starting trading dashboard demo...")
    print("Press Ctrl+C to stop")
    demo_dashboard()