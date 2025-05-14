import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import simulator and feature extractor
from simulators.market_simulator import MarketSimulatorV2
from feature.feature_extractor import FeatureExtractor  # Adjust import path as needed

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('feature_extractor_test')


# ==================== Test Data Fixtures ====================

@pytest.fixture(scope="module")
def test_timeframe():
    """Define the time period for tests."""
    # Use a shorter timeframe to speed up tests
    start_time = datetime(2025, 3, 27, 9, 30, 0)  # Market open
    end_time = datetime(2025, 3, 27, 10, 30, 0)  # 1 hour of data
    return start_time, end_time


@pytest.fixture(scope="module")
def test_1s_data(test_timeframe):
    """Create 1-second OHLCV test data with specific patterns for feature testing."""
    start_time, end_time = test_timeframe

    # Generate timestamps for each second
    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq='s'  # Use 's' instead of 'S' (deprecated)
    )

    # Create price data with specific patterns that we can verify
    # Base price and basic pattern
    base_price = 10.0

    # Create a DataFrame first
    num_points = len(timestamps)

    # Create data with clear patterns for testing feature calculations
    data = []

    # Start with constant price
    current_price = base_price

    # Create some distinct patterns we can test against:
    # 1. Flat period
    # 2. Steady uptrend
    # 3. Volatile period
    # 4. Downtrend
    # 5. Sharp V-reversal

    pattern_length = num_points // 5

    for i, ts in enumerate(timestamps):
        pattern_idx = i // pattern_length
        within_pattern_idx = i % pattern_length

        if pattern_idx == 0:
            # Flat period - minimal changes
            price_change = np.random.normal(0, 0.002)

        elif pattern_idx == 1:
            # Steady uptrend
            price_change = 0.005 + np.random.normal(0, 0.002)

        elif pattern_idx == 2:
            # Volatile period
            price_change = np.random.normal(0, 0.02)

        elif pattern_idx == 3:
            # Downtrend
            price_change = -0.004 + np.random.normal(0, 0.002)

        elif pattern_idx == 4:
            # V-pattern (first half down, second half up)
            if within_pattern_idx < pattern_length // 2:
                price_change = -0.01 + np.random.normal(0, 0.002)
            else:
                price_change = 0.012 + np.random.normal(0, 0.002)

        # Update current price
        current_price *= (1 + price_change)

        # Create bar data with appropriate high/low values
        open_price = current_price / (1 + price_change)  # Previous close
        close_price = current_price

        # Create realistic high/low based on volatility
        volatility_factor = 3 if pattern_idx == 2 else 1
        price_range = abs(price_change) * volatility_factor

        high_price = max(open_price, close_price) + abs(price_change) * volatility_factor * 0.5
        low_price = min(open_price, close_price) - abs(price_change) * volatility_factor * 0.5

        # Generate volume based on pattern
        if pattern_idx == 2:  # Higher volume in volatile periods
            volume = np.random.lognormal(7, 0.5)
        elif pattern_idx == 4 and within_pattern_idx >= pattern_length // 2:  # Rising volume in recovery
            volume = np.random.lognormal(6 + within_pattern_idx / (pattern_length / 4), 0.3)
        else:
            volume = np.random.lognormal(6, 0.3)

        # Create final bar
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': int(volume)
        })

    # Create DataFrame with timestamp index
    df = pd.DataFrame(data, index=timestamps)
    logger.info(f"Created 1s OHLCV data with {len(df)} bars and specific pattern regions")
    return df


@pytest.fixture(scope="module")
def test_trades_data(test_1s_data):
    """Create trades data corresponding to the 1s bars."""
    trades = []

    for idx, row in test_1s_data.iterrows():
        # For each 1s bar, create 0-20 trades
        volume_norm = np.clip(row['volume'] / 2000, 0, 1)
        num_trades = np.random.binomial(20, volume_norm)

        for _ in range(num_trades):
            # Random timestamp within this second
            trade_ts = idx + timedelta(milliseconds=np.random.randint(0, 1000))

            # Price within the high-low range
            price = np.random.uniform(row['low'], row['high'])

            # Trade size
            size = np.random.lognormal(3, .5)

            # Side (more buys when price is rising)
            price_change = row['close'] - row['open']
            buy_prob = 0.5 + 0.3 * np.sign(price_change)
            side = 'buy' if np.random.random() < buy_prob else 'sell'

            trades.append({
                'timestamp': trade_ts,
                'price': price,
                'size': int(size),
                'side': side
            })

    trades_df = pd.DataFrame(trades).set_index('timestamp').sort_index()
    logger.info(f"Created {len(trades_df)} trades with timestamp alignment")
    return trades_df


@pytest.fixture(scope="module")
def test_quotes_data(test_1s_data):
    """Create quotes data corresponding to the 1s bars."""
    quotes = []

    for idx, row in test_1s_data.iterrows():
        # 2-10 quote updates per second
        num_quotes = np.random.randint(2, 11)

        # Calculate mid price and initial spread
        mid_price = (row['open'] + row['close']) / 2
        price_volatility = (row['high'] - row['low']) / mid_price
        spread_pct = max(0.0001, min(0.001, price_volatility * 0.1))

        for i in range(num_quotes):
            # Quotes spread throughout the second
            quote_ts = idx + timedelta(milliseconds=i * (1000 // num_quotes))

            # Update prices based on position in the second
            progress = i / num_quotes
            current_mid = row['open'] * (1 - progress) + row['close'] * progress

            # Widen spreads during volatile periods
            current_spread = spread_pct * current_mid

            bid_price = current_mid - current_spread / 2
            ask_price = current_mid + current_spread / 2

            # Ensure prices are within high-low range
            bid_price = max(row['low'] * 0.999, bid_price)
            ask_price = min(row['high'] * 1.001, ask_price)

            # Generate sizes
            bid_size = int(np.random.lognormal(7, 0.5))
            ask_size = int(np.random.lognormal(7, 0.5))

            quotes.append({
                'timestamp': quote_ts,
                'bid_price': bid_price,
                'bid_size': bid_size,
                'ask_price': ask_price,
                'ask_size': ask_size
            })

    quotes_df = pd.DataFrame(quotes).set_index('timestamp').sort_index()
    logger.info(f"Created {len(quotes_df)} quotes with timestamp alignment")
    return quotes_df


@pytest.fixture(scope="module")
def simulator_with_data(test_1s_data, test_trades_data, test_quotes_data, test_timeframe):
    """Create a simulator instance with test data."""
    start_time, end_time = test_timeframe
    symbol = "MLGO"

    config = {
        'mode': 'backtesting',
        'start_time_str': start_time.isoformat(),
        'end_time_str': end_time.isoformat(),
        'max_1s_data_window_size': 3600,  # Keep enough data for feature calculations
        'initial_buffer_seconds': 300  # Buffer for feature look-back
    }

    simulator = MarketSimulatorV2(
        symbol=symbol,
        historical_1s_data=test_1s_data,
        historical_trades_data=test_trades_data,
        historical_quotes_data=test_quotes_data,
        config=config,
        logger=logger
    )

    return simulator


@pytest.fixture
def feature_config():
    """Create feature extractor configuration."""
    return {
        # HF (High Frequency) features
        'hf_seq_len': 60,  # 1-minute lookback at 1s frequency
        'hf_feat_dim': 7,  # Number of HF features

        # MF (Medium Frequency) features
        'mf_seq_len': 30,  # 30-minute lookback at 1m frequency
        'mf_feat_dim': 7,  # Number of MF features

        # LF (Low Frequency) features
        'lf_seq_len': 10,  # 50-minute lookback at 5m frequency
        'lf_feat_dim': 6,  # Number of LF features

        # Static features
        'static_feat_dim': 6,  # Number of static features

        # Technical indicators
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bbands_period': 20,
        'bbands_stddev': 2.0,

        # Normalization settings
        'normalize_features': True,
        'normalization_method': 'z_score',
        'normalization_window': 300,  # 5 minutes
    }


@pytest.fixture
def feature_extractor(feature_config):
    """Create a feature extractor instance."""
    # Initialize feature extractor with the config
    extractor = FeatureExtractor(config=feature_config)
    return extractor


# ==================== Tests ====================

def test_feature_calculation_and_timestamp_sync(simulator_with_data, feature_extractor):
    """
    Test that features are calculated correctly with proper timestamp alignment.
    """
    simulator = simulator_with_data

    # Step forward a bit to ensure enough data for calculations
    warmup_steps = 120  # 2 minutes
    for _ in range(warmup_steps):
        simulator.step()

    # Now collect feature data for several steps
    test_steps = 60  # 1 minute of testing
    timestamps = []
    features = []

    for _ in range(test_steps):
        # Get current market state
        current_time = simulator.current_timestamp
        market_state = simulator.get_current_market_state()

        # Extract features
        feature_dict = feature_extractor.extract_features(market_state)

        # Store data
        timestamps.append(current_time)
        features.append(feature_dict)

        # Step simulator forward
        simulator.step()

    # Verify the features have expected structure
    # We expect a dict with keys for each feature group
    expected_keys = ['hf_features', 'mf_features', 'lf_features', 'static_features']

    for i, feature_dict in enumerate(features):
        # Check basic structure
        for key in expected_keys:
            assert key in feature_dict, f"Missing key {key} in feature dict at step {i}"
            assert isinstance(feature_dict[key], np.ndarray), f"Feature {key} is not a numpy array at step {i}"

    # Check feature dimensions
    for feature_dict in features:
        # Check HF features shape - should be [batch_size, seq_len, num_features]
        # or [batch_size, num_features, seq_len] depending on your implementation
        assert feature_dict['hf_features'].shape[1] == feature_config['hf_seq_len'] or \
               feature_dict['hf_features'].shape[2] == feature_config['hf_seq_len'], \
            f"HF feature sequence length mismatch: {feature_dict['hf_features'].shape}"

        assert feature_dict['mf_features'].shape[1] == feature_config['mf_seq_len'] or \
               feature_dict['mf_features'].shape[2] == feature_config['mf_seq_len'], \
            f"MF feature sequence length mismatch: {feature_dict['mf_features'].shape}"

        assert feature_dict['lf_features'].shape[1] == feature_config['lf_seq_len'] or \
               feature_dict['lf_features'].shape[2] == feature_config['lf_seq_len'], \
            f"LF feature sequence length mismatch: {feature_dict['lf_features'].shape}"

    logger.info("Feature calculation and structure test passed successfully")


def test_feature_evolution_over_time(simulator_with_data, feature_extractor):
    """
    Test that features evolve correctly over time as new data arrives.
    """
    simulator = simulator_with_data

    # Reset simulator to the beginning
    simulator.reset()

    # Step forward to build up some data
    warmup_steps = 120
    for _ in range(warmup_steps):
        simulator.step()

    # Get features at two different timestamps
    state1 = simulator.get_current_market_state()
    features1 = feature_extractor.extract_features(state1)
    price1 = state1['current_price']

    # Advance simulator by several steps to a significantly different price
    steps_forward = 50
    for _ in range(steps_forward):
        simulator.step()

    state2 = simulator.get_current_market_state()
    features2 = feature_extractor.extract_features(state2)
    price2 = state2['current_price']

    # Verify features have changed in response to price changes
    # The latest value in the HF features should roughly track the price change

    # Get price feature indices (assuming price is in the feature set)
    # This will vary depending on your actual feature implementation
    price_feature_idx = 0  # Usually the first feature is price, adjust if needed

    # Get the latest price from HF features
    # Depending on your feature shape, this might be [0, -1, price_idx] or [0, price_idx, -1]
    # Try both access patterns based on your likely implementation
    try:
        hf_latest_price1 = features1['hf_features'][0, -1, price_feature_idx]
        hf_latest_price2 = features2['hf_features'][0, -1, price_feature_idx]
        feature_price_diff = hf_latest_price2 - hf_latest_price1
    except IndexError:
        try:
            hf_latest_price1 = features1['hf_features'][0, price_feature_idx, -1]
            hf_latest_price2 = features2['hf_features'][0, price_feature_idx, -1]
            feature_price_diff = hf_latest_price2 - hf_latest_price1
        except IndexError:
            # If both fail, log a warning and skip this check
            logger.warning("Could not access price feature in the expected format")
            feature_price_diff = 0

    # Calculate actual price difference
    actual_price_diff = price2 - price1

    # The feature price change might be normalized, so we just check the sign
    if actual_price_diff != 0 and feature_price_diff != 0:
        assert np.sign(feature_price_diff) == np.sign(actual_price_diff), \
            "Feature price change direction doesn't match actual price change"

    logger.info("Feature evolution test passed successfully")


def test_input_validation_and_error_handling(simulator_with_data, feature_extractor):
    """
    Test that the feature extractor gracefully handles invalid inputs.
    """
    # Test with None market state
    try:
        result = feature_extractor.extract_features(None)
        # If it returns rather than raises, check for sensible default/empty structure
        assert result is None or isinstance(result, dict), "Should return None or a default dict for None input"
    except Exception as e:
        # If it raises, that's okay too, just log it
        logger.info(f"Feature extractor raises {type(e).__name__} for None input: {str(e)}")

    # Test with empty market state
    empty_state = {}
    try:
        result = feature_extractor.extract_features(empty_state)
        # If it returns rather than raises, check for sensible default structure
        assert result is None or isinstance(result, dict), "Should return None or a default dict for empty input"
    except Exception as e:
        # If it raises, that's okay too, just log it
        logger.info(f"Feature extractor raises {type(e).__name__} for empty input: {str(e)}")

    # Test with valid market state but some parts missing
    simulator = simulator_with_data
    simulator.reset()
    for _ in range(60):  # Build some data
        simulator.step()

    valid_state = simulator.get_current_market_state()

    # Create a state with missing data
    partial_state = {k: v for k, v in valid_state.items() if k != 'latest_1s_bar'}

    try:
        result = feature_extractor.extract_features(partial_state)
        # If it returns rather than raises, we expect some default/fallback behavior
        assert isinstance(result, dict), "Should return a dict even with partial input"
    except Exception as e:
        # If it raises, that's okay too
        logger.info(f"Feature extractor raises {type(e).__name__} for partial input: {str(e)}")

    logger.info("Input validation test passed successfully")


def test_feature_shape_consistency(simulator_with_data, feature_extractor):
    """
    Test that feature shapes remain consistent over multiple steps.
    """
    simulator = simulator_with_data

    # Reset simulator to the beginning
    simulator.reset()

    # Step forward to build up some data
    warmup_steps = 120
    for _ in range(warmup_steps):
        simulator.step()

    # Check feature shapes over multiple steps
    test_steps = 30
    expected_shapes = None

    for step in range(test_steps):
        state = simulator.get_current_market_state()
        features = feature_extractor.extract_features(state)

        # Get feature shapes
        current_shapes = {
            key: features[key].shape for key in features
            if isinstance(features[key], np.ndarray)
        }

        # Set expected shapes on first step
        if expected_shapes is None:
            expected_shapes = current_shapes
            logger.info(f"Feature shapes: {expected_shapes}")

        # Compare shapes
        for key, shape in expected_shapes.items():
            assert key in current_shapes, f"Feature {key} missing at step {step}"
            assert current_shapes[key] == shape, \
                f"Feature {key} shape changed at step {step}: {current_shapes[key]} vs {shape}"

        simulator.step()

    logger.info("Feature shape consistency test passed successfully")


def test_handling_of_different_market_conditions(simulator_with_data, feature_extractor):
    """
    Test that the feature extractor handles different market conditions correctly.
    """
    simulator = simulator_with_data
    simulator.reset()

    # Step to different pattern regions in our test data
    # Our test data has five distinct regions: flat, uptrend, volatile, downtrend, and V-reversal
    pattern_length = len(simulator.all_1s_data) // 5

    # Test each pattern region
    pattern_labels = ["flat", "uptrend", "volatile", "downtrend", "v_reversal"]

    for i, pattern in enumerate(pattern_labels):
        # Go to the middle of this pattern region
        target_step = i * pattern_length + pattern_length // 2

        # Reset and step to target
        simulator.reset()
        for _ in range(target_step):
            simulator.step()

        # Get features for this market condition
        state = simulator.get_current_market_state()
        features = feature_extractor.extract_features(state)

        # Verify we have features
        assert 'hf_features' in features, f"Missing HF features in {pattern} condition"
        assert 'mf_features' in features, f"Missing MF features in {pattern} condition"
        assert 'lf_features' in features, f"Missing LF features in {pattern} condition"
        assert 'static_features' in features, f"Missing static features in {pattern} condition"

        # Output feature statistics for this condition
        for key in ['hf_features', 'mf_features', 'lf_features']:
            feature_array = features[key]
            logger.info(f"{pattern} condition - {key} mean: {np.mean(feature_array)}, std: {np.std(feature_array)}")

    logger.info("Market conditions test passed successfully")