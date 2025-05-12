# SIMPLIFIED main.py
import logging
import pandas as pd
from datetime import datetime

# Import components
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from feature.feature_extractor import FeatureExtractor
from simulation.simulator import Simulator
from envs.trading_env import TradingEnv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # 1. Create basic data pipeline
    provider = DabentoFileProvider("./dnb/mlgo")
    data_manager = DataManager(provider, logger=logger)

    # 2. Simple configuration for simulators
    sim_config = {
        'market_config': {'slippage_factor': 0.001},
        'execution_config': {'commission_per_share': 0.0},
        'portfolio_config': {'initial_cash': 100000.0}
    }

    # 3. Create simulator with minimal setup
    simulator = Simulator(data_manager, sim_config, logger=logger)

    # 4. Load sample data for a symbol
    symbol = "MLGO"
    date = "2025-03-27"

    logger.info(f"Initializing simulator for {symbol}")
    simulator.initialize_for_symbol(
        symbol, mode='backtesting',
        start_time=date, end_time=date,
        timeframes=["1m", "5m"]  # Reduced timeframes
    )

    # 5. Create environment with simple configuration
    env_config = {
        'random_reset': False,
        'state_dim': 20,  # Reduced state dimension
        'max_steps': 100
    }

    # 6. Create and test environment with minimal setup
    env = TradingEnv(simulator, env_config, logger=logger)

    # 7. Simple test
    for episode in range(2):  # Just 2 episodes
        state, info = env.reset()

        done = False
        step = 0
        total_reward = 0

        while not done and step < 10:  # Just 10 steps per episode
            # Take simple random action
            action = 0.5  # Simple fixed action for testing

            next_state, reward, terminated, truncated, info = env.step([action])
            done = terminated or truncated

            total_reward += reward
            step += 1

            logger.info(f"Step {step}: Reward={reward:.4f}, Total={total_reward:.4f}")

        logger.info(f"Episode {episode + 1} finished: Steps={step}, Total Reward={total_reward:.4f}")

    logger.info("Test completed successfully")


if __name__ == "__main__":
    main()