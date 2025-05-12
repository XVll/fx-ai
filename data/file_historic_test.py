# data/file_historic_test.py
from data.feature.data_processor import DataProcessor
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from data.data_manager import DataManager
import pandas as pd
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def state_update_callback(state):
    """Example callback for state updates."""
    position = state.get('current_position', 0)
    pnl = state.get('unrealized_pnl', 0)
    logger.info(f"State update: Position={position:.2f}, PnL={pnl:.2f}%")


def trade_callback(trade):
    """Example callback for completed trades."""
    logger.info(f"Trade completed: {trade['realized_pnl']:.2f}% PnL, Duration: {trade['duration']:.1f}s")


def main():
    # Make sure the data directory exists
    data_dir = "../dnb/Mlgo"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist. Please check the path.")
        return

    # Initialize the provider with more verbose output for troubleshooting
    provider = DabentoFileProvider(data_dir)

    # List available symbols
    symbols = provider.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")

    # Create data manager
    data_manager = DataManager(provider, logger=logger)

    # Create data processor with both callbacks
    data_processor = DataProcessor(data_manager, logger=logger)
    data_processor.add_state_update_callback(state_update_callback)
    data_processor.add_trade_callback(trade_callback)

    # Test symbol and date
    symbol = "MLGO"
    date_str = "2025-03-27"

    # Initialize for backtesting with explicit timeframes
    logger.info(f"Initializing for {symbol} on {date_str}")
    success = data_processor.initialize_for_symbol(
        symbol, mode='backtesting',
        start_time=date_str, end_time=date_str,
        timeframes=["1s", "1m", "5m", "1d"]
    )

    if not success:
        logger.error("Failed to initialize")
        return

    # Get features
    features_df = data_processor.get_features(symbol)

    if features_df.empty:
        logger.error("No features extracted. Exiting.")
        return

    logger.info(f"Extracted {len(features_df)} feature rows with {len(features_df.columns)} features")

    # Step to a specific timestamp
    if not features_df.empty:
        # Get a timestamp from the middle of the data
        timestamp = features_df.index[len(features_df) // 2]
        logger.info(f"Stepping to {timestamp}...")

        # Step to the timestamp
        state = data_processor.step_to_time(timestamp)

        # Get the state array
        state_array = data_processor.get_current_state_array()
        logger.info(f"State vector shape: {state_array.shape}")

        # Execute actions
        logger.info("Executing buy action...")
        data_processor.execute_action(0.5, 10.0)  # 50% position at $10.00

        # Step forward in time
        next_timestamp = timestamp + pd.Timedelta(seconds=60)
        logger.info(f"Stepping to {next_timestamp}...")
        data_processor.step_to_time(next_timestamp)

        # Increase position
        logger.info("Increasing position...")
        data_processor.execute_action(1.0, 11.0)  # Full position at $11.0

        # Step forward again
        exit_timestamp = next_timestamp + pd.Timedelta(seconds=120)
        logger.info(f"Stepping to {exit_timestamp}...")
        data_processor.step_to_time(exit_timestamp)

        # Close position
        logger.info("Closing position...")
        data_processor.execute_action(0.0, 12.0)  # Close at $12.0

        # Get trade statistics
        stats = data_processor.get_trade_statistics()
        logger.info(f"Trade statistics: {stats}")

    # Clean up
    data_manager.clear_cache()
    data_processor.reset()
    logger.info("Test completed")


if __name__ == "__main__":
    main()