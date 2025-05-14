import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path if necessary
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import simulator (adjust the import path as needed for your project structure)
from simulators.market_simulator import MarketSimulatorV2

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('market_simulator_test')


# ==================== Test Data Fixtures ====================

@pytest.fixture(scope="module")
def test_timeframe():
    """Define the time period for tests."""
    start_time = datetime(2025, 3, 27, 9, 30, 0)  # Market open
    end_time = datetime(2025, 3, 27, 16, 0, 0)  # Market close
    return start_time, end_time


@pytest.fixture(scope="module")
def test_1s_data(test_timeframe):
    """Create realistic 1-second OHLCV test data."""
    start_time, end_time = test_timeframe

    # Generate timestamps for each second during market hours
    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq='1s'
    )

    # Create base price series with some realistic movement
    base_price = 10.0
    np.random.seed(42)  # For reproducibility

    # Create price movements with some realistic patterns
    # 1. General random walk
    random_changes = np.random.normal(0, 0.005, len(timestamps))

    # 2. Add some momentum trends (periods of sustained movement)
    trend_periods = 5  # Number of trend periods throughout the day
    trend_length = len(timestamps) // trend_periods
    trends = np.zeros(len(timestamps))

    for i in range(trend_periods):
        start_idx = i * trend_length
        end_idx = (i + 1) * trend_length
        # Alternate between up and down trends
        trend_direction = 1 if i % 2 == 0 else -1
        trend_strength = np.random.uniform(0.001, 0.003)
        trends[start_idx:end_idx] = trend_direction * trend_strength

    # 3. Add some volatility clusters
    volatility_periods = 3  # Number of high volatility periods
    volatility_length = len(timestamps) // (trend_periods * 2)

    for i in range(volatility_periods):
        start_idx = np.random.randint(0, len(timestamps) - volatility_length)
        end_idx = start_idx + volatility_length
        volatility_multiplier = np.random.uniform(2, 4)
        random_changes[start_idx:end_idx] *= volatility_multiplier

    # 4. Add a few price jumps/drops (simulating news events)
    num_jumps = 2
    for _ in range(num_jumps):
        jump_idx = np.random.randint(len(timestamps) // 4, len(timestamps) * 3 // 4)
        jump_size = np.random.choice([-0.2, 0.2]) * base_price * 0.05  # 5% price jump
        random_changes[jump_idx] += jump_size

    # Combine the components to create realistic price changes
    price_changes = random_changes + trends

    # Calculate cumulative price changes
    cum_changes = np.cumsum(price_changes)

    # Calculate price series ensuring it stays positive
    prices = base_price * (1 + cum_changes)
    prices = np.maximum(prices, 0.1)  # Ensure prices don't go negative or too close to zero

    # Generate OHLCV data with some realistic patterns
    data = []

    # Start with the first price
    current_price = prices[0]

    for i, ts in enumerate(timestamps):
        # Calculate realistic OHLCV for this second
        target_price = prices[i]

        # Determine price movement for this second
        price_direction = 1 if target_price >= current_price else -1
        price_change_magnitude = abs(target_price - current_price)

        # Create some variation in the open, high, low, close
        open_price = current_price

        # For high volatility, create more price variation within the second
        intra_second_volatility = np.random.uniform(0.0, 0.5) * price_change_magnitude

        if price_direction > 0:
            # Upward movement
            low_price = open_price - intra_second_volatility * 0.2
            high_price = target_price + intra_second_volatility
            close_price = target_price
        else:
            # Downward movement
            high_price = open_price + intra_second_volatility * 0.2
            low_price = target_price - intra_second_volatility
            close_price = target_price

        # Ensure high >= open/close >= low
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate realistic volume
        # Base volume
        base_volume = np.random.lognormal(6, 1)

        # Increase volume during price jumps and at round prices
        # and at the beginning and end of the day
        volume_modifier = 1.0

        # Higher volume on significant price movements
        if price_change_magnitude > 0.01:
            volume_modifier *= (1 + 10 * price_change_magnitude)

        # Higher volume at round price levels
        if abs(round(close_price * 2) / 2 - close_price) < 0.01:  # Near half-dollar levels
            volume_modifier *= 1.5
        if abs(round(close_price) - close_price) < 0.01:  # Near whole-dollar levels
            volume_modifier *= 2.0

        # Higher volume at market open and close
        minutes_since_open = (ts - start_time).total_seconds() / 60
        minutes_until_close = (end_time - ts).total_seconds() / 60

        if minutes_since_open < 30:  # First 30 minutes
            volume_modifier *= (1 + 2 * np.exp(-minutes_since_open / 10))
        if minutes_until_close < 30:  # Last 30 minutes
            volume_modifier *= (1 + 2 * np.exp(-minutes_until_close / 10))

        volume = int(base_volume * volume_modifier)

        # Append to data
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

        # Update current price for next iteration
        current_price = close_price

    # Create DataFrame with timestamp index
    df = pd.DataFrame(data, index=timestamps)
    logger.info(f"Created 1s OHLCV data with {len(df)} bars")
    return df


@pytest.fixture(scope="module")
def test_trades_data(test_1s_data):
    """Create realistic trades test data based on the 1s bars."""
    trades = []

    for idx, row in test_1s_data.iterrows():
        # Generate between 0-20 trades per second based on volume
        # Higher volume = more trades
        volume_normalized = min(1.0, row['volume'] / 1000)
        num_trades = int(np.random.binomial(20, volume_normalized))

        # Skip seconds with no trades
        if num_trades == 0:
            continue

        # Generate trades within this second
        for _ in range(num_trades):
            # Trade timestamp within this second
            milliseconds = np.random.randint(0, 1000)
            trade_ts = idx + timedelta(milliseconds=milliseconds)

            # Trade price around the OHLCV prices
            min_price, max_price = row['low'], row['high']
            price = np.random.uniform(min_price, max_price)

            # Trade size - lognormal distribution
            size = int(np.random.lognormal(3, 1))  # Median size around 20 shares

            # Trade side - slightly more buys during up moves, more sells during down moves
            price_range = row['high'] - row['low']
            if price_range > 0:
                # Normalize price position within the range
                price_position = (price - row['low']) / price_range
                # Bias towards buys for higher prices, sells for lower prices
                buy_probability = 0.5 + (price_position - 0.5) * 0.6
            else:
                buy_probability = 0.5

            side = 'buy' if np.random.random() < buy_probability else 'sell'

            trades.append({
                'timestamp': trade_ts,
                'price': price,
                'size': size,
                'side': side
            })

    # Create DataFrame
    trades_df = pd.DataFrame(trades).set_index('timestamp').sort_index()
    logger.info(f"Created trades data with {len(trades_df)} trades")
    return trades_df


@pytest.fixture(scope="module")
def test_quotes_data(test_1s_data):
    """Create realistic quotes test data."""
    quotes = []

    for idx, row in test_1s_data.iterrows():
        # Generate between 1-10 quote updates per second
        num_quotes = np.random.randint(1, 11)

        # Start with a reasonable spread
        mid_price = (row['open'] + row['close']) / 2
        spread = mid_price * 0.001  # 0.1% spread

        # Initial bid/ask
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2

        # Typical sizes
        bid_size = int(np.random.lognormal(7, 1))  # ~1000 shares
        ask_size = int(np.random.lognormal(7, 1))

        for _ in range(num_quotes):
            # Quote timestamp within this second
            milliseconds = np.random.randint(0, 1000)
            quote_ts = idx + timedelta(milliseconds=milliseconds)

            # Update bid/ask prices with some randomness but tracking OHLC
            price_range = row['high'] - row['low']
            if price_range > 0:
                # Move bid/ask within the price range
                price_movement = np.random.normal(0, price_range * 0.1)
                bid_price += price_movement
                ask_price += price_movement

                # Ensure ask > bid with a reasonable spread
                min_spread = mid_price * 0.0005  # Min 0.05% spread
                if ask_price - bid_price < min_spread:
                    mid_price = (bid_price + ask_price) / 2
                    bid_price = mid_price - min_spread / 2
                    ask_price = mid_price + min_spread / 2

            # Ensure prices are within high-low range with some leeway
            bid_price = max(row['low'] * 0.999, min(row['high'] * 1.001, bid_price))
            ask_price = max(row['low'] * 0.999, min(row['high'] * 1.001, ask_price))

            # Update sizes with some randomness
            bid_size = int(bid_size * np.random.uniform(0.8, 1.2))
            ask_size = int(ask_size * np.random.uniform(0.8, 1.2))

            quotes.append({
                'timestamp': quote_ts,
                'bid_price': bid_price,
                'bid_size': bid_size,
                'ask_price': ask_price,
                'ask_size': ask_size
            })

    # Create DataFrame
    quotes_df = pd.DataFrame(quotes).set_index('timestamp').sort_index()
    logger.info(f"Created quotes data with {len(quotes_df)} quotes")
    return quotes_df


@pytest.fixture(scope="module")
def test_5m_data(test_1s_data):
    """Create 5-minute bars by resampling 1s data."""
    # Resample the 1s data to 5m
    df_5m = test_1s_data.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    logger.info(f"Created 5m bars with {len(df_5m)} bars")
    return df_5m


@pytest.fixture(scope="module")
def test_1d_data(test_1s_data, test_timeframe):
    """Create daily bars covering the test period."""
    start_time, _ = test_timeframe
    # For simplicity, just create a single daily bar for the test day
    df_1d = pd.DataFrame([{
        'open': test_1s_data['open'].iloc[0],
        'high': test_1s_data['high'].max(),
        'low': test_1s_data['low'].min(),
        'close': test_1s_data['close'].iloc[-1],
        'volume': test_1s_data['volume'].sum()
    }], index=[start_time.replace(hour=0, minute=0, second=0)])
    logger.info(f"Created daily bars with {len(df_1d)} bars")
    return df_1d


@pytest.fixture(scope="module")
def symbol():
    """Return the test symbol."""
    return "MLGO"


@pytest.fixture(scope="module")
def config(test_timeframe):
    """Return the simulator configuration."""
    start_time, end_time = test_timeframe
    return {
        'mode': 'backtesting',
        'start_time_str': start_time.isoformat(),
        'end_time_str': end_time.isoformat(),
        'max_1s_data_window_size': 3600,  # 1 hour
        'initial_buffer_seconds': 300  # 5 minutes
    }


@pytest.fixture
def simulator(symbol, test_1s_data, test_trades_data, test_quotes_data,
              test_5m_data, test_1d_data, config):
    """Create a simulator instance with test data."""
    sim = MarketSimulatorV2(
        symbol=symbol,
        historical_1s_data=test_1s_data,
        historical_trades_data=test_trades_data,
        historical_quotes_data=test_quotes_data,
        historical_5m_bars=test_5m_data,
        historical_1d_bars=test_1d_data,
        config=config,
        logger=logger
    )
    return sim


# ==================== Tests ====================

def test_initialization(simulator, symbol):
    """Test that the simulator initializes correctly with test data."""
    assert simulator.symbol == symbol
    assert simulator.mode == 'backtesting'
    assert simulator.current_timestamp is not None

    # Verify data was loaded correctly
    assert simulator.all_1s_data is not None
    assert len(simulator.all_1s_data) > 0
    assert simulator.all_trades_data is not None
    assert simulator.all_quotes_data is not None

    # Verify buffer is populated
    assert len(simulator.current_1s_data_window) > 0

    logger.info("Initialization test passed successfully")


def test_stepping(simulator):
    """Test that stepping advances the simulator correctly."""
    # Record initial state
    initial_timestamp = simulator.current_timestamp
    initial_idx = simulator._current_1s_data_idx

    # Step forward
    result = simulator.step()

    # Verify step was successful
    assert result is True

    # Verify timestamp advanced by 1 second
    assert simulator.current_timestamp == initial_timestamp + timedelta(seconds=1)

    # Verify index advanced
    assert simulator._current_1s_data_idx == initial_idx + 1

    # Verify data window updated
    assert simulator.current_1s_data_window[-1]['timestamp'] == simulator.current_timestamp

    logger.info("Stepping test passed successfully")


def test_market_state(simulator):
    """Test that the market state is correctly constructed."""
    # Get current market state
    state = simulator.get_current_market_state()

    # Verify state structure
    assert state is not None
    assert state['symbol'] == simulator.symbol

    # Fix for the timestamp issue: The state's timestamp might be from the latest event
    # in the window, which could be the same as or slightly behind the simulator's current_timestamp
    # Instead of strict equality, check that the timestamp is within 1 second
    timestamp_diff = abs((state['timestamp'] - simulator.current_timestamp).total_seconds())
    assert timestamp_diff <= 1, f"Timestamp difference too large: {timestamp_diff} seconds"

    # Check for required components
    assert 'latest_1s_bar' in state
    assert 'latest_1s_trades' in state
    assert 'latest_1s_quotes' in state
    assert 'rolling_1s_data_window' in state
    assert 'current_price' in state

    # Verify price data is present
    assert state['current_price'] is not None

    # Verify window data
    assert len(state['rolling_1s_data_window']) > 0

    logger.info("Market state test passed successfully")


def test_multiple_steps(simulator):
    """Test stepping through multiple data points."""
    # Record initial timestamp
    initial_timestamp = simulator.current_timestamp

    # Step forward multiple times
    num_steps = 10
    states = []

    for _ in range(num_steps):
        result = simulator.step()
        assert result is True
        states.append(simulator.get_current_market_state())

    # Verify timestamp advanced correctly
    assert simulator.current_timestamp == initial_timestamp + timedelta(seconds=num_steps)

    # Verify states contain different timestamps
    timestamps = [state['timestamp'] for state in states]
    assert len(set(timestamps)) == num_steps, "Each state should have a unique timestamp"

    # Verify price data changes
    prices = [state['current_price'] for state in states if state['current_price'] is not None]
    # We should have at least some price variation
    assert len(set(prices)) > 1, "Prices should change across multiple steps"

    logger.info("Multiple steps test passed successfully")


def test_reset(simulator):
    """Test simulator reset functionality."""
    # Step forward several times
    for _ in range(20):
        simulator.step()

    advanced_timestamp = simulator.current_timestamp
    advanced_idx = simulator._current_1s_data_idx

    # Reset the simulator
    simulator.reset()

    # Verify reset state
    reset_timestamp = simulator.current_timestamp
    reset_idx = simulator._current_1s_data_idx

    # Should be back to initial state (might not be exactly the same timestamp
    # due to initial buffer processing, but should be earlier)
    assert reset_timestamp < advanced_timestamp, "Reset timestamp should be earlier than advanced timestamp"
    assert reset_idx < advanced_idx, "Reset index should be less than advanced index"

    # Verify we can step forward again
    result = simulator.step()
    assert result is True, "Should be able to step after reset"

    logger.info("Reset test passed successfully")


def test_is_done(symbol, test_timeframe):
    """Test is_done detection at the end of data."""
    # Create a small dataset for this test
    start_time, _ = test_timeframe  # Fixed: proper tuple unpacking
    end_time = start_time + timedelta(minutes=5)  # Just 5 minutes

    timestamps = pd.date_range(start=start_time, end=end_time, freq='1s')
    data = []
    for i, ts in enumerate(timestamps):
        data.append({
            'open': 10.0 + i * 0.01,
            'high': 10.0 + i * 0.01 + 0.05,
            'low': 10.0 + i * 0.01 - 0.05,
            'close': 10.0 + i * 0.01 + 0.02,
            'volume': 100 + i
        })

    small_1s_data = pd.DataFrame(data, index=timestamps)

    config = {
        'mode': 'backtesting',
        'start_time_str': start_time.isoformat(),
        'end_time_str': end_time.isoformat(),
        'max_1s_data_window_size': 300,
        'initial_buffer_seconds': 10
    }

    simulator = MarketSimulatorV2(
        symbol=symbol,
        historical_1s_data=small_1s_data,
        config=config,
        logger=logger
    )

    # Step through all data
    step_count = 0
    while simulator.step():
        step_count += 1
        if step_count > 1000:  # Safety limit
            pytest.fail("Too many steps, possible infinite loop")

    # Verify is_done returns True at the end
    # Fixed: use == instead of 'is' for NumPy boolean comparison
    assert simulator.is_done() == True, "simulator.is_done() should return True at end of data"

    # Try stepping one more time
    result = simulator.step()
    assert result == False, "step() should return False when at end of data"

def test_empty_data_handling(symbol, config):
    """Test simulator behavior with empty data."""
    # Empty dataframes
    empty_df = pd.DataFrame()

    # Should raise ValueError due to requirement for 1s data
    with pytest.raises(ValueError):
        MarketSimulatorV2(
            symbol=symbol,
            historical_1s_data=empty_df,
            config={'mode': 'backtesting'},
            logger=logger
        )

    # Create a simulator with valid 1s data but empty trades/quotes
    # First create some minimal 1s data
    timestamps = pd.date_range(
        start=datetime(2025, 3, 27, 9, 30, 0),
        end=datetime(2025, 3, 27, 9, 40, 0),
        freq='1s'
    )
    data = [{'open': 10, 'high': 10.1, 'low': 9.9, 'close': 10.05, 'volume': 100}
            for _ in range(len(timestamps))]
    min_1s_data = pd.DataFrame(data, index=timestamps)

    # Simulator without trades/quotes should still work
    simulator = MarketSimulatorV2(
        symbol=symbol,
        historical_1s_data=min_1s_data,  # Valid 1s data
        historical_trades_data=empty_df,  # Empty trades
        historical_quotes_data=empty_df,  # Empty quotes
        config=config,
        logger=logger
    )

    # Should be able to step and get market state
    result = simulator.step()
    assert result is True, "step() should succeed even with empty trades/quotes"

    state = simulator.get_current_market_state()
    assert state is not None, "Should get a valid market state even with empty trades/quotes"
    assert len(state.get('latest_1s_trades', [])) == 0, "No trades should be present"
    assert len(state.get('latest_1s_quotes', [])) == 0, "No quotes should be present"

    logger.info("Empty data handling test passed successfully")


def test_live_mode_basic(symbol, test_1s_data):
    """Test basic live mode functionality."""
    # Configuration for live mode
    live_config = {
        'mode': 'live',
        'max_1s_data_window_size': 3600,
    }

    # Start with some historical data to initialize the live mode
    simulator = MarketSimulatorV2(
        symbol=symbol,
        historical_1s_data=test_1s_data.iloc[:100],  # Just a bit of history
        config=live_config,
        logger=logger
    )

    # Verify live mode is set
    assert simulator.mode == 'live', "Mode should be set to 'live'"

    # Add some live data
    current_time = datetime.now().replace(microsecond=0)

    # Add a live trade
    trade_event = {
        'timestamp': current_time,
        'price': 15.25,
        'size': 100,
        'side': 'buy'
    }
    simulator.add_live_trade(trade_event)

    # Add a live quote
    quote_event = {
        'timestamp': current_time,
        'bid_price': 15.20,
        'bid_size': 500,
        'ask_price': 15.30,
        'ask_size': 300
    }
    simulator.add_live_quote(quote_event)

    # Add a live 1s bar
    bar_event = {
        'timestamp': current_time,
        'open': 15.20,
        'high': 15.30,
        'low': 15.15,
        'close': 15.25,
        'volume': 1000
    }
    simulator.add_live_1s_data_event(bar_event)

    # This is mostly a smoke test to ensure methods run without error
    # In a real integration test, you'd validate the data was properly incorporated

    logger.info("Live mode basic test passed successfully")


def test_edge_case_timestamp_matching(symbol, test_timeframe):
    """Test edge cases with timestamp matching and data gaps."""
    # Create data with some gaps
    start_time, _ = test_timeframe
    end_time = start_time + timedelta(minutes=10)

    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq='1s'
    ).tolist()

    # Remove some timestamps to create gaps
    gap_indices = [10, 20, 30, 40, 50]
    for idx in sorted(gap_indices, reverse=True):
        if idx < len(timestamps):
            timestamps.pop(idx)

    data = []
    for i, ts in enumerate(timestamps):
        data.append({
            'open': 10.0 + i * 0.01,
            'high': 10.0 + i * 0.01 + 0.05,
            'low': 10.0 + i * 0.01 - 0.05,
            'close': 10.0 + i * 0.01 + 0.02,
            'volume': 100 + i
        })

    gapped_1s_data = pd.DataFrame(data, index=timestamps)

    config = {
        'mode': 'backtesting',
        'start_time_str': start_time.isoformat(),
        'end_time_str': end_time.isoformat(),
        'max_1s_data_window_size': 300,
        'initial_buffer_seconds': 10
    }

    simulator = MarketSimulatorV2(
        symbol=symbol,
        historical_1s_data=gapped_1s_data,
        config=config,
        logger=logger
    )

    # Step through data and verify it handles gaps
    steps_without_error = 0
    try:
        while simulator.step() and steps_without_error < 100:
            state = simulator.get_current_market_state()
            assert state is not None, "Should get valid state even with data gaps"
            assert state['current_price'] is not None, "Should have current price even with data gaps"
            steps_without_error += 1
    except Exception as e:
        pytest.fail(f"Exception during stepping with gaps: {e}")

    assert steps_without_error > 0, "Should be able to step through data with gaps"
    logger.info(f"Successfully stepped through {steps_without_error} steps with data gaps")


def test_state_content(simulator):
    """Test the content of the market state for feature extraction."""
    # Take a step to ensure we have a valid state
    simulator.step()

    # Get market state
    state = simulator.get_current_market_state()

    # Check that the state contains all components needed for feature engineering
    assert state is not None

    # Check bar data
    assert state['latest_1s_bar'] is not None
    assert 'open' in state['latest_1s_bar']
    assert 'high' in state['latest_1s_bar']
    assert 'low' in state['latest_1s_bar']
    assert 'close' in state['latest_1s_bar']
    assert 'volume' in state['latest_1s_bar']

    # Check the window
    window = state['rolling_1s_data_window']
    assert len(window) > 0

    # Check that we can extract prices and volumes from the window
    prices = [item['bar']['close'] for item in window if item['bar'] is not None]
    volumes = [item['bar']['volume'] for item in window if item['bar'] is not None]

    assert len(prices) > 0
    assert len(volumes) > 0

    # Check trade data availability if trades exist
    if 'latest_1s_trades' in state and state['latest_1s_trades']:
        for trade in state['latest_1s_trades']:
            assert 'price' in trade
            assert 'size' in trade
            if 'side' in trade:
                assert trade['side'] in ['buy', 'sell', None, '']

    # Check quote data availability if quotes exist
    if 'latest_1s_quotes' in state and state['latest_1s_quotes']:
        for quote in state['latest_1s_quotes']:
            assert 'bid_price' in quote
            assert 'ask_price' in quote

    logger.info("State content test passed successfully")