#!/usr/bin/env python3
# demo_centralized_logging.py - Demo script showing the centralized logging system
"""
Demo script to showcase the new centralized logging system with enhanced dashboard.

This script demonstrates:
1. How to initialize the centralized logger
2. How to use logging from different modules
3. How the enhanced dashboard works with real-time logs
4. Integration between logging and Rich console display

Run this script to see the 2-column dashboard in action:
- Left column: Live logs from all modules
- Right column: Trading dashboard (simulated)
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import (
    initialize_logger, get_logger, log_info, log_warning,
    log_error, log_debug, log_critical
)
from envs.env_dashboard import EnhancedTradingDashboard


class MockTrainingSession:
    """Mock training session to demonstrate logging integration"""

    def __init__(self):
        self.logger_manager = get_logger()
        self.dashboard = EnhancedTradingDashboard(logger_manager=self.logger_manager)
        self.is_running = False
        self.step = 0
        self.episode = 0
        self.equity = 25000.0
        self.price = 5.0

    def start_demo(self):
        """Start the demo session"""
        log_info("🚀 Starting FX-AI Trading Demo", "demo")

        # Setup dashboard
        self.dashboard.set_symbol("MLGO")
        self.dashboard.set_initial_capital(25000.0)
        self.dashboard.start()

        log_info("📊 Enhanced dashboard started successfully", "demo")

        # Start background logging simulation
        self.is_running = True

        # Start different logging threads to simulate different modules
        threading.Thread(target=self._simulate_market_data, daemon=True).start()
        threading.Thread(target=self._simulate_portfolio_updates, daemon=True).start()
        threading.Thread(target=self._simulate_model_training, daemon=True).start()
        threading.Thread(target=self._simulate_execution_logs, daemon=True).start()

        # Main simulation loop
        self._run_simulation()

    def _simulate_market_data(self):
        """Simulate market data logging"""
        while self.is_running:
            # Simulate market tick
            self.price += random.uniform(-0.05, 0.05)
            self.price = max(0.1, self.price)  # Keep price positive

            if random.random() < 0.3:  # 30% chance to log
                if random.random() < 0.1:  # 10% chance for warnings
                    log_warning(f"📊 High volatility detected: price moved to ${self.price:.4f}", "market")
                else:
                    log_debug(f"📈 Market tick: ${self.price:.4f}", "market")

            time.sleep(random.uniform(0.5, 2.0))

    def _simulate_portfolio_updates(self):
        """Simulate portfolio manager logging"""
        while self.is_running:
            # Simulate equity changes
            equity_change = random.uniform(-50, 100)
            self.equity += equity_change

            if random.random() < 0.2:  # 20% chance to log
                if equity_change > 50:
                    log_info(f"💰 Portfolio gain: ${equity_change:.2f}, Total: ${self.equity:.2f}", "portfolio")
                elif equity_change < -30:
                    log_warning(f"📉 Portfolio loss: ${equity_change:.2f}, Total: ${self.equity:.2f}", "portfolio")
                else:
                    log_debug(f"💼 Portfolio update: ${self.equity:.2f}", "portfolio")

            time.sleep(random.uniform(1.0, 3.0))

    def _simulate_model_training(self):
        """Simulate model training logging"""
        while self.is_running:
            if random.random() < 0.15:  # 15% chance to log
                loss = random.uniform(0.01, 0.5)
                entropy = random.uniform(0.1, 1.0)

                if random.random() < 0.05:  # 5% chance for errors
                    log_error(f"🚨 Training error: gradient explosion detected", "model")
                elif loss < 0.1:
                    log_info(f"🧠 Model converging: loss={loss:.4f}, entropy={entropy:.3f}", "model")
                else:
                    log_debug(f"📊 Training step: loss={loss:.4f}", "model")

            time.sleep(random.uniform(0.8, 2.5))

    def _simulate_execution_logs(self):
        """Simulate execution and order logging"""
        while self.is_running:
            if random.random() < 0.1:  # 10% chance to log
                actions = ["BUY", "SELL", "HOLD"]
                action = random.choice(actions)

                if action in ["BUY", "SELL"]:
                    quantity = random.uniform(10, 1000)
                    if random.random() < 0.05:  # 5% chance for execution errors
                        log_error(f"❌ Order rejected: {action} {quantity:.2f} @ ${self.price:.4f}", "execution")
                    else:
                        log_info(f"✅ Order filled: {action} {quantity:.2f} @ ${self.price:.4f}", "execution")
                else:
                    log_debug(f"⏸️ Action: {action}", "execution")

            time.sleep(random.uniform(2.0, 5.0))

    def _run_simulation(self):
        """Main simulation loop"""
        try:
            while self.is_running:
                self.step += 1

                # Simulate step info for dashboard
                step_reward = random.uniform(-1.0, 2.0)
                episode_reward = random.uniform(-100, 500)

                # Create mock info dict for dashboard
                info_dict = {
                    'step': self.step,
                    'timestamp_iso': datetime.now().isoformat(),
                    'reward_step': step_reward,
                    'episode_cumulative_reward': episode_reward,
                    'portfolio_equity': self.equity,
                    'portfolio_cash': max(0, 25000 - abs(episode_reward)),
                    'portfolio_unrealized_pnl': random.uniform(-200, 200),
                    'portfolio_realized_pnl_session_net': random.uniform(-100, 150),
                    'position_MLGO_qty': random.uniform(-500, 500),
                    'position_MLGO_side': random.choice(['LONG', 'SHORT', 'FLAT']),
                    'position_MLGO_avg_entry': self.price * random.uniform(0.95, 1.05),
                    'invalid_actions_total_episode': random.randint(0, 5),
                    'action_decoded': {
                        'type': type('ActionType', (), {'name': random.choice(['BUY', 'SELL', 'HOLD'])})(),
                        'size_enum': type('SizeEnum', (),
                                          {'name': random.choice(['SIZE_25', 'SIZE_50', 'SIZE_75', 'SIZE_100'])})(),
                        'invalid_reason': None if random.random() > 0.1 else "Insufficient funds"
                    },
                    'fills_step': []
                }

                # Simulate market state
                market_state = {
                    'current_price': self.price,
                    'best_bid_price': self.price * 0.999,
                    'best_ask_price': self.price * 1.001,
                    'best_bid_size': random.uniform(100, 1000),
                    'best_ask_size': random.uniform(100, 1000),
                    'market_session': random.choice(['PREMARKET', 'REGULAR', 'POSTMARKET'])
                }

                # Update dashboard
                self.dashboard.update_state(info_dict, market_state)

                # Occasional episode end
                if self.step % 100 == 0:
                    self.episode += 1
                    log_info(f"🏁 Episode {self.episode} completed with reward {episode_reward:.2f}", "demo")
                    self.step = 0

                # Log progress
                if self.step % 20 == 0:
                    log_info(f"📈 Step {self.step}: Price=${self.price:.4f}, Equity=${self.equity:.2f}", "demo")

                time.sleep(0.2)  # Dashboard update frequency

        except KeyboardInterrupt:
            log_info("🛑 Demo interrupted by user", "demo")
        finally:
            self.stop_demo()

    def stop_demo(self):
        """Stop the demo session"""
        self.is_running = False
        if self.dashboard:
            self.dashboard.stop()
        log_info("👋 Demo session ended", "demo")


def main():
    """Main demo function"""
    print("🎯 FX-AI Centralized Logging System Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("• Centralized logging from multiple modules")
    print("• Enhanced 2-column dashboard (logs + trading info)")
    print("• Real-time log display with color coding")
    print("• Integration between logging and Rich console")
    print("=" * 50)
    print("Press Ctrl+C to stop the demo\n")

    # Initialize centralized logging
    log_file = f"demo_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger_manager = initialize_logger(
        app_name="fx-ai-demo",
        log_file=log_file,
        max_dashboard_logs=200
    )

    log_info("🔧 Centralized logging system initialized", "demo")
    log_info(f"📝 Logs will be saved to: {log_file}", "demo")

    # Show different log levels
    log_debug("🐛 This is a debug message", "demo")
    log_info("ℹ️ This is an info message", "demo")
    log_warning("⚠️ This is a warning message", "demo")
    log_error("❌ This is an error message (simulated)", "demo")
    log_critical("🚨 This is a critical message (simulated)", "demo")

    # Wait a moment for the messages to be processed
    time.sleep(2)

    # Start the demo session
    demo_session = MockTrainingSession()
    demo_session.start_demo()


if __name__ == "__main__":
    main()