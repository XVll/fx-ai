# test_feature_extraction.py - cleaner version
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from data.feature.feature_extractor import FeatureExtractor
from data.feature.state_manager import StateManager
import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the provider with verbose=False to reduce logging
    provider = DabentoFileProvider("../dnb/Mlgo", verbose=False)

    # Test symbol and date with known data
    symbol = "MLGO"
    date_str = "2025-03-27"  # Date with plenty of data

    # Load all data types
    data_dict = {}

    bars_1s = provider.get_bars(symbol, "1s", date_str, date_str)
    if not bars_1s.empty:
        data_dict['bars_1s'] = bars_1s

    bars_1m = provider.get_bars(symbol, "1m", date_str, date_str)
    if not bars_1m.empty:
        data_dict['bars_1m'] = bars_1m

    bars_5m = provider.get_bars(symbol, "5m", date_str, date_str)
    if not bars_5m.empty:
        data_dict['bars_5m'] = bars_5m

    bars_1d = provider.get_bars(symbol, "1d", date_str, date_str)
    if not bars_1d.empty:
        data_dict['bars_1d'] = bars_1d

    quotes = provider.get_quotes(symbol, date_str, date_str)
    if not quotes.empty:
        data_dict['quotes'] = quotes

    trades = provider.get_trades(symbol, date_str, date_str)
    if not trades.empty:
        data_dict['trades'] = trades

    status = provider.get_status(symbol, date_str, date_str)
    if not status.empty:
        data_dict['status'] = status

    # Initialize feature extractor
    logger.info("\nExtracting features...")
    feature_extractor = FeatureExtractor()

    try:
        features_df = feature_extractor.extract_features(data_dict)

        if features_df.empty:
            logger.warning("No features extracted.")
            return

        logger.info(f"Extracted {len(features_df)} feature rows with {len(features_df.columns)} features")

        # Initialize state manager
        logger.info("\nInitializing state manager...")
        state_manager = StateManager(feature_extractor)

        # Update state at a specific timestamp
        if not features_df.empty:
            # Get a timestamp from the middle of the data
            timestamp = features_df.index[len(features_df) // 2]
            logger.info(f"Updating state at {timestamp}...")
            state_manager.update_from_features(features_df, timestamp)

            # Get the state vector
            state_array = state_manager.get_state_array()
            logger.info(f"State vector shape: {state_array.shape}")

            # Simulate a position update
            logger.info("Simulating position update...")
            state_manager.update_position(0.5, 10.0, timestamp)  # 50% position at $10.00
            updated_state = state_manager.get_state_dict()
            logger.info(f"Position: {updated_state['current_position']}, Entry price: ${state_manager.entry_price:.2f}")

            # Try another update
            logger.info("Simulating position increase...")
            state_manager.update_position(1.0, 11.0, timestamp + pd.Timedelta(seconds=60))  # Full position at $11.0
            updated_state = state_manager.get_state_dict()
            logger.info(f"Position: {updated_state['current_position']}, Unrealized PnL: {updated_state['unrealized_pnl']:.2f}%")

            logger.info("\nFeature extraction and state management test completed successfully.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        for key, df in data_dict.items():
            logger.debug(f"{key} shape: {df.shape}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()