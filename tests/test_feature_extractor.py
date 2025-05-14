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
        freq='1S'
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
            size = np.random.lognormal(3, 1)

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
        'hf_features': [
            'price', 'volume', 'price_change', 'price_change_pct',
            'returns', 'log_returns', 'volatility'
        ],

        # MF (Medium Frequency) features
        'mf_seq_len': 30,  # 30-minute lookback at 1m frequency
        'mf_features': [
            'price', 'volume', 'vwap', 'price_change_pct',
            'volatility', 'rsi', 'return'
        ],

        # LF (Low Frequency) features
        'lf_seq_len': 10,  # 50-minute lookback at 5m frequency
        'lf_features': [
            'price', 'volume', 'vwap', 'volatility',
            'atr', 'rsi'
        ],

        # Static features
        'static_features': [
            'current_price', 'day_high', 'day_low', 'day_vwap',
            'nearest_support', 'nearest_resistance'
        ],

        # Technical indicators
        'calculate_vwap': True,
        'calculate_rsi': True,
        'rsi_period': 14,
        'calculate_macd': True,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'calculate_bbands': True,
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
    return FeatureExtractor(feature_config, logger=logger)


# ==================== Tests ====================

def test_feature_calculation_and_timestamp_alignment(simulator_with_data, feature_extractor):
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
    market_states = []

    for _ in range(test_steps):
        # Get current market state
        market_state = simulator.get_current_market_state()
        market_states.append(market_state)

        # Extract features
        feature_dict = feature_extractor.extract_features(market_state)

        # Store data
        timestamps.append(simulator.current_timestamp)
        features.append(feature_dict)

        # Step simulator forward
        simulator.step()

    # Now verify the alignment and feature calculations

    # 1. Test timestamp alignment
    # Features should be calculated at the simulator's current timestamp
    for i, (ts, feature_dict) in enumerate(zip(timestamps, features)):
        assert feature_dict['timestamp'] == ts, f"Feature timestamp misaligned at step {i}"

    # 2. Test feature data structure
    # Verify all expected feature groups are present
    for feature_dict in features:
        assert 'hf_features' in feature_dict, "High frequency features missing"
        assert 'mf_features' in feature_dict, "Medium frequency features missing"
        assert 'lf_features' in feature_dict, "Low frequency features missing"
        assert 'static_features' in feature_dict, "Static features missing"

        # Check feature dimensions
        assert feature_dict['hf_features'].shape[1] == feature_config['hf_seq_len'], \
            f"HF feature length mismatch: {feature_dict['hf_features'].shape[1]} vs {feature_config['hf_seq_len']}"

        assert feature_dict['mf_features'].shape[1] == feature_config['mf_seq_len'], \
            f"MF feature length mismatch: {feature_dict['mf_features'].shape[1]} vs {feature_config['mf_seq_len']}"

        assert feature_dict['lf_features'].shape[1] == feature_config['lf_seq_len'], \
            f"LF feature length mismatch: {feature_dict['lf_features'].shape[1]} vs {feature_config['lf_seq_len']}"

    # 3. Test feature continuity
    # HF features of consecutive timestamps should only differ by one step
    for i in range(1, len(features)):
        # Take the last N-1 steps of previous HF features
        prev_hf_features = features[i - 1]['hf_features'][:, 1:]
        # Take the first N-1 steps of current HF features
        current_hf_features = features[i]['hf_features'][:, :-1]

        # These should be identical if feature sliding window is working correctly
        np.testing.assert_allclose(
            prev_hf_features, current_hf_features,
            rtol=1e-5, atol=1e-8,
            err_msg=f"HF feature continuity broken at step {i}"
        )

    logger.info("Feature calculation and timestamp alignment test passed successfully")


def test_feature_calculation_with_gaps(simulator_with_data, feature_extractor):
    """
    Test feature calculation when there are gaps in market data.
    """
    simulator = simulator_with_data

    # Reset simulator to the beginning
    simulator.reset()

    # Step forward to build up some data
    warmup_steps = 120
    for _ in range(warmup_steps):
        simulator.step()

    # Save the current state
    base_state = simulator.get_current_market_state()
    base_features = feature_extractor.extract_features(base_state)

    # Now skip several steps (simulating a data gap)
    gap_size = 10
    for _ in range(gap_size):
        simulator.step()

    # Get new state and features
    gap_state = simulator.get_current_market_state()
    gap_features = feature_extractor.extract_features(gap_state)

    # Verify the feature extractor handled the gap properly

    # 1. Timestamps should reflect the gap
    time_diff = gap_features['timestamp'] - base_features['timestamp']
    assert time_diff.total_seconds() == gap_size, \
        f"Feature timestamp gap incorrect: {time_diff.total_seconds()} vs expected {gap_size}"

    # 2. HF features should still have the right shape
    assert gap_features['hf_features'].shape[1] == feature_config['hf_seq_len'], \
        "HF feature length incorrect after gap"

    # 3. Check for appropriate handling of missing data (varies by implementation)
    # This could be NaN values, forward fill, interpolation, etc.
    # The key is that feature extraction shouldn't crash with gaps

    logger.info("Feature calculation with gaps test passed successfully")


def test_technical_indicators_accuracy(simulator_with_data, feature_extractor):
    """
    Test that technical indicators are calculated correctly.
    """
    simulator = simulator_with_data

    # Reset simulator to the beginning
    simulator.reset()

    # Step forward to build up enough data for indicators
    warmup_steps = 200  # Need enough data for slow moving averages
    for _ in range(warmup_steps):
        simulator.step()

    # Collect market data and calculated features
    test_steps = 30
    market_data = []
    extracted_features = []

    for _ in range(test_steps):
        state = simulator.get_current_market_state()
        features = feature_extractor.extract_features(state)

        market_data.append(state)
        extracted_features.append(features)

        simulator.step()

    # Now calculate indicators manually to verify

    # Extract price data from market states
    prices = np.array([
        state['latest_1s_bar']['close']
        for state in market_data if state['latest_1s_bar'] is not None
    ])

    # Extract volumes
    volumes = np.array([
        state['latest_1s_bar']['volume']
        for state in market_data if state['latest_1s_bar'] is not None
    ])

    # 1. Test RSI calculation
    # Manually calculate RSI
    def calculate_rsi(prices, period=14):
        # Calculate price changes
        deltas = np.diff(prices)
        # Get gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        # Initialize with SMA
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Use EMA for remaining
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        # Calculate RS and RSI
        rs = avg_gain[period:] / (avg_loss[period:] + 1e-9)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Pad beginning with NaNs
        full_rsi = np.full_like(prices, np.nan)
        full_rsi[period:] = rsi
        return full_rsi

    # Calculate RSI with the same period as feature extractor
    rsi_period = feature_config['rsi_period']
    manual_rsi = calculate_rsi(prices, rsi_period)

    # Get RSI from features (exact location depends on your feature extractor implementation)
    # This example assumes RSI is in MF features, as the second feature
    feature_rsi = extracted_features[-1]['mf_features'][1, -1]  # Latest RSI value

    # Check if values are close (allowing for minor float precision differences)
    # We only check the latest value as the earlier ones might be affected by different buffer sizes
    # and initial calculations
    if not np.isnan(manual_rsi[-1]):
        assert abs(feature_rsi - manual_rsi[-1]) < 1.0, \
            f"RSI calculation mismatch: {feature_rsi} vs manual {manual_rsi[-1]}"

    # 2. Test VWAP calculation
    # Manual VWAP calculation
    def calculate_vwap(prices, volumes):
        return np.sum(prices * volumes) / np.sum(volumes)

    manual_vwap = calculate_vwap(prices, volumes)

    # Get VWAP from features
    feature_vwap = extracted_features[-1]['static_features'][3]  # day_vwap in static features

    # Check VWAP values are close
    assert abs(feature_vwap - manual_vwap) / manual_vwap < 0.01, \
        f"VWAP calculation mismatch: {feature_vwap} vs manual {manual_vwap}"

    logger.info("Technical indicators accuracy test passed successfully")


def test_rolling_window_updates(simulator_with_data, feature_extractor):
    """
    Test that rolling window calculations update correctly.
    """
    simulator = simulator_with_data

    # Reset simulator to the beginning
    simulator.reset()

    # Step forward to build up some data
    warmup_steps = 120
    for _ in range(warmup_steps):
        simulator.step()

    # Track rolling windows over several steps
    test_steps = 10
    hf_windows = []

    for _ in range(test_steps):
        state = simulator.get_current_market_state()
        features = feature_extractor.extract_features(state)

        # Store the rolling window of HF features
        hf_windows.append(features['hf_features'].copy())

        simulator.step()

    # Verify that windows roll forward correctly
    for i in range(1, len(hf_windows)):
        prev_window = hf_windows[i - 1]
        curr_window = hf_windows[i]

        # The current window should match the previous window shifted by one,
        # with the newest data point at the end
        np.testing.assert_allclose(
            prev_window[:, 1:], curr_window[:, :-1],
            rtol=1e-5, atol=1e-8,
            err_msg=f"Rolling window update incorrect at step {i}"
        )

    logger.info("Rolling window updates test passed successfully")


def test_normalization(simulator_with_data, feature_extractor):
    """
    Test that feature normalization works correctly.
    """
    simulator = simulator_with_data

    # Reset simulator to the beginning
    simulator.reset()

    # Step forward to build up some data
    warmup_steps = 180  # 3 minutes
    for _ in range(warmup_steps):
        simulator.step()

    # Get normalized features
    state = simulator.get_current_market_state()
    features = feature_extractor.extract_features(state)

    # Verify normalization

    # 1. If using z-score normalization, means should be near 0 and std near 1
    if feature_config['normalization_method'] == 'z_score':
        # Check mean and std of each feature dimension
        for feature_group in ['hf_features', 'mf_features', 'lf_features']:
            if feature_group in features:
                feature_array = features[feature_group]

                # Calculate mean and std across the time dimension
                means = np.mean(feature_array, axis=1)
                stds = np.std(feature_array, axis=1)

                # Means should be close to 0
                assert np.allclose(means, 0, atol=0.5), \
                    f"{feature_group} means not close to 0: {means}"

                # Stds should be close to 1
                # Some features might be constant, so we only check non-zero stds
                non_zero_stds = stds[stds > 1e-6]
                if len(non_zero_stds) > 0:
                    assert np.allclose(non_zero_stds, 1, rtol=0.5), \
                        f"{feature_group} stds not close to 1: {non_zero_stds}"

    # 2. If using min-max normalization, values should be between 0 and 1
    elif feature_config['normalization_method'] == 'min_max':
        for feature_group in ['hf_features', 'mf_features', 'lf_features']:
            if feature_group in features:
                feature_array = features[feature_group]

                # Check min and max values
                min_vals = np.min(feature_array)
                max_vals = np.max(feature_array)

                assert min_vals >= 0 - 1e-6, f"{feature_group} min value below 0: {min_vals}"
                assert max_vals <= 1 + 1e-6, f"{feature_group} max value above 1: {max_vals}"

    logger.info("Normalization test passed successfully")


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

        # Compare shapes
        for key, shape in expected_shapes.items():
            assert key in current_shapes, f"Feature {key} missing at step {step}"
            assert current_shapes[key] == shape, \
                f"Feature {key} shape changed at step {step}: {current_shapes[key]} vs {shape}"

        simulator.step()

    logger.info("Feature shape consistency test passed successfully")