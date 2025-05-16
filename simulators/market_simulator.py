# market_simulator.py
import logging
from collections import deque
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple, Set
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import bisect

DEFAULT_MARKET_HOURS = {
    "PREMARKET_START": "04:00:00",
    "PREMARKET_END": "09:29:59",
    "REGULAR_START": "09:30:00",
    "REGULAR_END": "15:59:59",
    "POSTMARKET_START": "16:00:00",
    "POSTMARKET_END": "20:00:00",
    "TIMEZONE": "America/New_York"
}


class MarketSimulator:
    """
    Market simulator with unified state architecture.

    Features:
    - Pre-calculation of all market states during initialization
    - O(1) access to any timestamp's complete state
    - Consistent state calculation for both agent and execution
    - Prevention of future data leakage to the agent
    """

    def __init__(self,
                 symbol: str,
                 data_manager: Any,
                 mode: str = 'backtesting',
                 start_time: Optional[str | datetime] = None,
                 end_time: Optional[str | datetime] = None,
                 config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the market simulator.

        Args:
            symbol: Trading symbol
            data_manager: DataManager instance for retrieving data
            mode: 'backtesting' or 'live'
            start_time: Start time for historical data (backtesting)
            end_time: End time for historical data (backtesting)
            config: Configuration dictionary
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.data_manager = data_manager
        self.config = config or {}
        self.mode = mode

        # Parse start and end times
        self.start_time_utc = self._parse_datetime(start_time)
        self.end_time_utc = self._parse_datetime(end_time)

        # Exchange timezone and market hours setup
        self._setup_market_hours()

        # Main unified state store - the single source of truth
        self._precomputed_states = {}  # {timestamp: complete_state}
        self._all_timestamps = []  # Ordered list of all timestamps

        # Rolling window sizes for data context
        self.rolling_1s_window_size = self.config.get('rolling_1s_data_window_size', 60 * 60)
        self.rolling_1m_window_size = self.config.get('rolling_1m_data_window_size', 120)
        self.rolling_5m_window_size = self.config.get('rolling_5m_data_window_size', 48)

        # Agent timeline state
        self.current_timestamp_utc = None
        self._current_time_idx = 0

        # Load data and initialize the simulator
        self._initialize()

    def _setup_market_hours(self):
        """Configure timezone and market hours."""
        _market_hours_cfg_raw = self.config.get('market_hours', DEFAULT_MARKET_HOURS)
        self.exchange_timezone_str = _market_hours_cfg_raw.get('TIMEZONE', DEFAULT_MARKET_HOURS['TIMEZONE'])
        self.exchange_timezone = ZoneInfo(self.exchange_timezone_str)

        self.market_hours = {}
        for key, default_val_str in DEFAULT_MARKET_HOURS.items():
            if key.endswith(("_START", "_END")):
                time_str = _market_hours_cfg_raw.get(key, default_val_str)
                self.market_hours[key] = datetime.strptime(time_str, "%H:%M:%S").time()

    def _parse_datetime(self, dt_input: Optional[str | datetime]) -> Optional[datetime]:
        """Convert input to UTC datetime with robust parsing."""
        if dt_input is None:
            return None

        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None:
                self.logger.debug(f"Received naive datetime: {dt_input}. Assuming UTC.")
                return dt_input.replace(tzinfo=ZoneInfo("UTC"))
            return dt_input.astimezone(ZoneInfo("UTC"))

        if isinstance(dt_input, str):
            try:
                # Try ISO format first
                dt_obj = datetime.fromisoformat(dt_input.replace(" Z", "+00:00"))
                if dt_obj.tzinfo is None:
                    return dt_obj.replace(tzinfo=ZoneInfo("UTC"))
                return dt_obj.astimezone(ZoneInfo("UTC"))
            except ValueError:
                pass

            # Try pandas parsing as fallback
            try:
                dt_obj = pd.to_datetime(dt_input)
                if dt_obj.tzinfo is None:
                    return dt_obj.replace(tzinfo=ZoneInfo("UTC"))
                return dt_obj.astimezone(ZoneInfo("UTC"))
            except Exception as e:
                self.logger.error(f"Failed to parse datetime: {dt_input}, error: {e}")
                return None

        self.logger.error(f"Invalid datetime input type: {type(dt_input)}")
        return None

    def _initialize(self):
        """Initialize the simulator based on mode."""
        self.logger.info(f"Initializing MarketSimulator for {self.symbol} in {self.mode} mode")

        # Load initial data from data manager
        self._load_initial_data()

        # Initialize according to mode
        if self.mode == 'backtesting':
            self._initialize_backtesting()
        elif self.mode == 'live':
            self._initialize_live()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _load_initial_data(self):
        """Load required data from data manager."""
        if not self.data_manager:
            self.logger.error("No data_manager provided")
            return

        try:
            # Load all required data types for the specified time range
            self.logger.info(f"Loading data for {self.symbol} from {self.start_time_utc} to {self.end_time_utc}")

            data_result = self.data_manager.load_data(
                symbols=[self.symbol],
                start_time=self.start_time_utc,
                end_time=self.end_time_utc,
                data_types=["trades", "quotes", "bars_1s", "bars_1m", "bars_5m", "bars_1d", "status"]
            )

            # Extract data frames from the result
            if self.symbol not in data_result:
                self.logger.error(f"No data loaded for symbol {self.symbol}")
                return

            symbol_data = data_result[self.symbol]

            # Store raw data
            self.raw_trades = symbol_data.get("trades", pd.DataFrame())
            self.raw_quotes = symbol_data.get("quotes", pd.DataFrame())
            self.raw_1s_bars = symbol_data.get("bars_1s", pd.DataFrame())
            self.raw_1m_bars = symbol_data.get("bars_1m", pd.DataFrame())
            self.raw_5m_bars = symbol_data.get("bars_5m", pd.DataFrame())
            self.raw_1d_bars = symbol_data.get("bars_1d", pd.DataFrame())
            self.raw_status = symbol_data.get("status", pd.DataFrame())

            # Log data quantities
            self.logger.info(f"Loaded data: {len(self.raw_1s_bars)} 1s bars, "
                             f"{len(self.raw_trades)} trades, {len(self.raw_quotes)} quotes")

            # Ensure 1s bars exist - these are essential for the simulator
            if self.raw_1s_bars.empty and not self.raw_trades.empty:
                self.logger.info("No 1s bars found. Generating from trades...")
                self._generate_1s_bars_from_trades()

            # Get daily bars for a longer period if needed for support/resistance
            self._load_extended_daily_data()

        except Exception as e:
            self.logger.error(f"Error loading data: {e}", exc_info=True)

    def _generate_1s_bars_from_trades(self):
        """Generate 1-second OHLCV bars from trade data."""
        if self.raw_trades.empty:
            self.logger.warning("Cannot generate 1s bars: No trade data available")
            self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
            return

        self.logger.info("Generating 1-second bars from trade data...")

        try:
            # Check if required columns exist
            required_cols = ['price', 'size']
            if not all(col in self.raw_trades.columns for col in required_cols):
                missing = [col for col in required_cols if col not in self.raw_trades.columns]
                self.logger.error(f"Trade data missing required columns: {missing}")
                self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
                return

            # Resample to 1-second bars
            resampled = (
                self.raw_trades
                .assign(value=self.raw_trades['price'] * self.raw_trades['size'])
                .resample('1s')
                .agg(
                    open=('price', 'first'),
                    high=('price', 'max'),
                    low=('price', 'min'),
                    close=('price', 'last'),
                    volume=('size', 'sum'),
                    value_sum=('value', 'sum')
                )
            )

            # Drop rows with no trades
            resampled.dropna(subset=['open'], inplace=True)

            if not resampled.empty:
                # Calculate VWAP
                resampled['vwap'] = resampled['value_sum'] / resampled['volume']
                resampled.drop(columns=['value_sum'], inplace=True)

                # Handle NaN values
                resampled.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Forward fill VWAP where needed
                resampled['vwap'].ffill(inplace=True)
                resampled['vwap'].bfill(inplace=True)

            self.raw_1s_bars = resampled
            self.logger.info(f"Generated {len(self.raw_1s_bars)} 1-second bars")

        except Exception as e:
            self.logger.error(f"Error generating 1s bars: {e}", exc_info=True)
            self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])

    def _load_extended_daily_data(self):
        """Load extended daily bar data for support/resistance."""
        if self.start_time_utc:
            # Get 1 year of historical data if available
            extended_start = self.start_time_utc - timedelta(days=365)
            try:
                extended_daily = self.data_manager.get_bars(
                    symbol=self.symbol,
                    timeframe="1d",
                    start_time=extended_start,
                    end_time=self.end_time_utc or datetime.now(ZoneInfo("UTC"))
                )

                if not extended_daily.empty:
                    # Only replace if we got more data
                    if self.raw_1d_bars is None or len(extended_daily) > len(self.raw_1d_bars):
                        self.raw_1d_bars = extended_daily
                        self.logger.info(f"Loaded {len(self.raw_1d_bars)} extended daily bars")

            except Exception as e:
                self.logger.warning(f"Failed to load extended daily data: {e}")

    def _initialize_backtesting(self):
        """Initialize for backtesting mode by pre-computing all states."""
        self.logger.info("Initializing backtesting mode")

        if self.raw_1s_bars is None or self.raw_1s_bars.empty:
            self.logger.error("No 1s bar data available for backtesting")
            return

        # Get all timestamps from 1s bars - this forms our timeline
        all_timestamps = self.raw_1s_bars.index.tolist()

        if not all_timestamps:
            self.logger.error("No timestamps found in 1s bar data")
            return

        # Sort timestamps (should already be sorted, but just to be safe)
        all_timestamps.sort()

        # Set the timeline
        self._all_timestamps = all_timestamps

        # Pre-compute all market states
        self._precompute_all_states()

        # Configure initial buffer period
        initial_buffer_seconds = self.config.get('initial_buffer_seconds', 300)

        # Determine agent's starting position (after buffer)
        if len(self._all_timestamps) > initial_buffer_seconds:
            self._current_time_idx = initial_buffer_seconds
        else:
            self._current_time_idx = 0

        # Set current timestamp
        if self._current_time_idx < len(self._all_timestamps):
            self.current_timestamp_utc = self._all_timestamps[self._current_time_idx]
            self.logger.info(f"Agent will start at: {self.current_timestamp_utc}")
        else:
            self.logger.warning("Not enough data for the requested buffer period")

    def _initialize_live(self):
        """Initialize for live trading mode."""
        self.logger.info("Initializing live trading mode")

        # In live mode, we still pre-compute states for historical data
        # but will update states in real-time as new data arrives

        # Set current timestamp to now
        self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)

        if not self.raw_1s_bars.empty:
            # Get historical timestamps
            all_timestamps = self.raw_1s_bars.index.tolist()
            all_timestamps.sort()

            # Add current time if not in the list
            if self.current_timestamp_utc not in all_timestamps:
                all_timestamps.append(self.current_timestamp_utc)
                all_timestamps.sort()

            self._all_timestamps = all_timestamps

            # Pre-compute historical states
            self._precompute_all_states()

            # Set current index to match current timestamp
            self._current_time_idx = self._all_timestamps.index(self.current_timestamp_utc)
        else:
            # No historical data, just start with current time
            self._all_timestamps = [self.current_timestamp_utc]
            self._current_time_idx = 0

            # Create an initial state
            self._precomputed_states[self.current_timestamp_utc] = self._calculate_default_state(
                self.current_timestamp_utc)

    def _precompute_all_states(self):
        """
        Pre-compute all market states for all timestamps.
        This is the core of the unified architecture.
        """
        self.logger.info(f"Pre-computing {len(self._all_timestamps)} market states...")

        # Initialize for day tracking
        prev_day = None
        day_high = None
        day_low = None

        # Track 1m and 5m aggregation
        current_1m_bar = None
        current_5m_bar = None
        completed_1m_bars = deque(maxlen=self.rolling_1m_window_size)
        completed_5m_bars = deque(maxlen=self.rolling_5m_window_size)

        # Track rolling windows of raw data
        rolling_1s_data = deque(maxlen=self.rolling_1s_window_size)

        # Process all timestamps chronologically
        for i, timestamp in enumerate(self._all_timestamps):
            # Get the local date for day tracking
            local_date = timestamp.astimezone(self.exchange_timezone).date()

            # Check for new day
            if prev_day != local_date:
                self.logger.debug(f"Processing new day: {local_date}")
                prev_day = local_date
                day_high = None
                day_low = None

                # Reset aggregation bars
                current_1m_bar = None
                current_5m_bar = None

            # Get the 1s bar for this timestamp
            current_1s_bar = self._get_raw_bar_at(timestamp)

            if current_1s_bar is not None:
                # Update day high/low
                if day_high is None or current_1s_bar['high'] > day_high:
                    day_high = current_1s_bar['high']
                if day_low is None or current_1s_bar['low'] < day_low:
                    day_low = current_1s_bar['low']

                # Create the comprehensive event data for this timestamp
                event_data = {
                    'timestamp': timestamp,
                    'bar': current_1s_bar,
                    'trades': self._get_trades_in_second(timestamp),
                    'quotes': self._get_quotes_in_second(timestamp)
                }

                # Add to rolling window
                rolling_1s_data.append(event_data)

                # Update aggregation bars
                self._update_aggregation_bars(current_1s_bar, timestamp,
                                              current_1m_bar, current_5m_bar,
                                              completed_1m_bars, completed_5m_bars)

            # Determine market session
            market_session = self._determine_market_session(timestamp)

            # Calculate the complete state
            state = self._calculate_complete_state(
                timestamp=timestamp,
                current_1s_bar=current_1s_bar,
                rolling_1s_data=list(rolling_1s_data),
                day_high=day_high,
                day_low=day_low,
                market_session=market_session,
                current_1m_bar=current_1m_bar,
                current_5m_bar=current_5m_bar,
                completed_1m_bars=list(completed_1m_bars),
                completed_5m_bars=list(completed_5m_bars)
            )

            # Store the complete state
            self._precomputed_states[timestamp] = state

            # Log progress occasionally
            if i % 10000 == 0 and i > 0:
                self.logger.info(f"Pre-computed {i}/{len(self._all_timestamps)} states")

        self.logger.info(f"Finished pre-computing {len(self._precomputed_states)} market states")

    def _get_raw_bar_at(self, timestamp):
        """Get the raw 1s bar at the specified timestamp."""
        if self.raw_1s_bars is None or self.raw_1s_bars.empty:
            return None

        try:
            bar = self.raw_1s_bars.loc[timestamp].to_dict()
            bar['timestamp'] = timestamp
            return bar
        except (KeyError, TypeError):
            return None

    def _get_trades_in_second(self, timestamp):
        """Get all trades within the 1-second interval ending at timestamp."""
        if self.raw_trades is None or self.raw_trades.empty:
            return []

        end_time = timestamp
        start_time = end_time - timedelta(seconds=1)

        try:
            trades_slice = self.raw_trades[
                (self.raw_trades.index > start_time) &
                (self.raw_trades.index <= end_time)
                ]

            trade_records = []
            for idx, row in trades_slice.iterrows():
                trade_dict = row.to_dict()
                trade_dict['timestamp'] = idx
                trade_records.append(trade_dict)

            return trade_records
        except Exception:
            return []

    def _get_quotes_in_second(self, timestamp):
        """Get all quotes within the 1-second interval ending at timestamp."""
        if self.raw_quotes is None or self.raw_quotes.empty:
            return []

        end_time = timestamp
        start_time = end_time - timedelta(seconds=1)

        try:
            quotes_slice = self.raw_quotes[
                (self.raw_quotes.index > start_time) &
                (self.raw_quotes.index <= end_time)
                ]

            quote_records = []
            for idx, row in quotes_slice.iterrows():
                quote_dict = row.to_dict()
                quote_dict['timestamp'] = idx
                quote_records.append(quote_dict)

            return quote_records
        except Exception:
            return []

    def _update_aggregation_bars(self, current_1s_bar, timestamp,
                                 current_1m_bar, current_5m_bar,
                                 completed_1m_bars, completed_5m_bars):
        """Update 1m and 5m aggregation bars based on the current 1s bar."""
        if current_1s_bar is None:
            return current_1m_bar, current_5m_bar

        # Extract timing info
        bar_minute = timestamp.minute
        bar_hour = timestamp.hour

        # 1-minute bar logic
        minute_start = timestamp.replace(second=0, microsecond=0)

        if current_1m_bar is None or current_1m_bar['timestamp_start'] != minute_start:
            # Complete previous 1m bar if exists
            if current_1m_bar is not None:
                completed_1m_bars.append(current_1m_bar.copy())

            # Start new 1m bar
            current_1m_bar = {
                'timestamp_start': minute_start,
                'open': current_1s_bar['open'],
                'high': current_1s_bar['high'],
                'low': current_1s_bar['low'],
                'close': current_1s_bar['close'],
                'volume': current_1s_bar.get('volume', 0)
            }
        else:
            # Update existing 1m bar
            current_1m_bar['high'] = max(current_1m_bar['high'], current_1s_bar['high'])
            current_1m_bar['low'] = min(current_1m_bar['low'], current_1s_bar['low'])
            current_1m_bar['close'] = current_1s_bar['close']
            current_1m_bar['volume'] = current_1m_bar.get('volume', 0) + current_1s_bar.get('volume', 0)

        # 5-minute bar logic
        five_min_interval = bar_minute // 5
        five_min_start = timestamp.replace(minute=five_min_interval * 5, second=0, microsecond=0)

        if current_5m_bar is None or current_5m_bar['timestamp_start'] != five_min_start:
            # Complete previous 5m bar if exists
            if current_5m_bar is not None:
                completed_5m_bars.append(current_5m_bar.copy())

            # Start new 5m bar
            current_5m_bar = {
                'timestamp_start': five_min_start,
                'open': current_1s_bar['open'],
                'high': current_1s_bar['high'],
                'low': current_1s_bar['low'],
                'close': current_1s_bar['close'],
                'volume': current_1s_bar.get('volume', 0)
            }
        else:
            # Update existing 5m bar
            current_5m_bar['high'] = max(current_5m_bar['high'], current_1s_bar['high'])
            current_5m_bar['low'] = min(current_5m_bar['low'], current_1s_bar['low'])
            current_5m_bar['close'] = current_1s_bar['close']
            current_5m_bar['volume'] = current_5m_bar.get('volume', 0) + current_1s_bar.get('volume', 0)

        return current_1m_bar, current_5m_bar

    def _determine_market_session(self, timestamp):
        """Determine the market session based on timestamp."""
        local_time = timestamp.astimezone(self.exchange_timezone).time()

        if self.market_hours["PREMARKET_START"] <= local_time <= self.market_hours["PREMARKET_END"]:
            return "PREMARKET"
        elif self.market_hours["REGULAR_START"] <= local_time <= self.market_hours["REGULAR_END"]:
            return "REGULAR"
        elif self.market_hours["POSTMARKET_START"] <= local_time <= self.market_hours["POSTMARKET_END"]:
            return "POSTMARKET"
        return "CLOSED"

    def _calculate_complete_state(self, timestamp, current_1s_bar, rolling_1s_data,
                                  day_high, day_low, market_session,
                                  current_1m_bar, current_5m_bar,
                                  completed_1m_bars, completed_5m_bars):
        """
        Calculate the complete market state at the given timestamp.
        This is the unified calculation used for both model and execution.
        """
        # Get current price
        current_price = None
        if current_1s_bar:
            current_price = current_1s_bar.get('close')

        # Create the complete state dictionary
        state = {
            'timestamp_utc': timestamp,
            'current_market_session': market_session,
            'current_time': timestamp,  # Redundant but kept for backward compatibility
            'current_price': current_price,

            'intraday_high': day_high,
            'intraday_low': day_low,

            'current_1s_bar': current_1s_bar,
            'current_1m_bar_forming': current_1m_bar,
            'current_5m_bar_forming': current_5m_bar,

            'rolling_1s_data_window': rolling_1s_data,
            'completed_1m_bars_window': completed_1m_bars,
            'completed_5m_bars_window': completed_5m_bars,

            # Additional state that might be useful
            'historical_1d_bars': self.raw_1d_bars,
        }

        return state

    def _calculate_default_state(self, timestamp):
        """Create a default state when no data is available."""
        return {
            'timestamp_utc': timestamp,
            'current_market_session': self._determine_market_session(timestamp),
            'current_time': timestamp,
            'current_price': None,

            'intraday_high': None,
            'intraday_low': None,

            'current_1s_bar': None,
            'current_1m_bar_forming': None,
            'current_5m_bar_forming': None,

            'rolling_1s_data_window': [],
            'completed_1m_bars_window': [],
            'completed_5m_bars_window': [],

            'historical_1d_bars': None,
        }

    def get_state_at_time(self, timestamp):
        """
        Get the complete market state at any timestamp.
        This is the core method for unified state access.

        Args:
            timestamp: The timestamp to get state for

        Returns:
            The complete market state at that timestamp
        """
        # Direct lookup if timestamp exists
        if timestamp in self._precomputed_states:
            return self._precomputed_states[timestamp]

        # Find the closest timestamps (we may need to interpolate)
        if not self._all_timestamps:
            return self._calculate_default_state(timestamp)

        # Binary search to find position
        pos = bisect.bisect_left(self._all_timestamps, timestamp)

        # Handle edge cases
        if pos == 0:
            # Before the first timestamp, no state available
            self.logger.warning(f"Timestamp {timestamp} is before first available data point")
            if len(self._all_timestamps) > 0:
                return self._precomputed_states[self._all_timestamps[0]]
            return self._calculate_default_state(timestamp)

        if pos == len(self._all_timestamps):
            # After the last timestamp, use last state
            self.logger.warning(f"Timestamp {timestamp} is after last available data point")
            return self._precomputed_states[self._all_timestamps[-1]]

        # Interpolation case: between two known timestamps
        prev_ts = self._all_timestamps[pos - 1]
        next_ts = self._all_timestamps[pos]

        # Get surrounding states
        prev_state = self._precomputed_states[prev_ts]
        next_state = self._precomputed_states[next_ts]

        # Perform simple state interpolation
        # For most fields, we'll use the previous state's values
        # but we could implement more sophisticated interpolation if needed
        interpolated_state = prev_state.copy()

        # Update timestamp to the requested one
        interpolated_state['timestamp_utc'] = timestamp
        interpolated_state['current_time'] = timestamp

        return interpolated_state

    def get_current_market_state(self):
        """
        Get the market state at the agent's current timeline position.
        This prevents future data leakage.
        """
        if not self._all_timestamps or self._current_time_idx >= len(self._all_timestamps):
            return None

        current_ts = self._all_timestamps[self._current_time_idx]
        return self.get_state_at_time(current_ts)

    def step(self):
        """
        Advance the agent's timeline position by one step.

        Returns:
            bool: True if successful, False if at the end
        """
        self._current_time_idx += 1

        if self._current_time_idx < len(self._all_timestamps):
            self.current_timestamp_utc = self._all_timestamps[self._current_time_idx]
            return True

        self.logger.info("Reached the end of available data")
        return False

    def is_done(self):
        """Check if the simulation is done."""
        # Check if we're past the last timestamp
        if not self._all_timestamps or self._current_time_idx >= len(self._all_timestamps):
            return True

        # Check if we've reached the configured end time
        if self.end_time_utc and self.current_timestamp_utc >= self.end_time_utc:
            return True

        return False

    def reset(self, options=None):
        """
        Reset the simulator to its initial state or a random point.

        Args:
            options: Reset options, including random_start flag

        Returns:
            The initial market state after reset
        """
        options = options or {}
        self.logger.info("Resetting MarketSimulator")

        # Reset state tracking
        self.current_timestamp_utc = None

        # Determine reset position
        if options.get('random_start', False) and len(self._all_timestamps) > 1:
            # Get buffer size
            initial_buffer_seconds = self.config.get('initial_buffer_seconds', 300)

            # Calculate viable start indices
            min_idx = initial_buffer_seconds if len(self._all_timestamps) > initial_buffer_seconds else 0
            max_idx = len(self._all_timestamps) - 1

            if max_idx > min_idx:
                # Random valid index
                self._current_time_idx = np.random.randint(min_idx, max_idx)
                self.logger.info(f"Random reset to index {self._current_time_idx}")
            else:
                # Not enough data for random start, use beginning
                self._current_time_idx = min_idx
                self.logger.info(f"Not enough data for random start, using index {min_idx}")
        else:
            # Regular reset to the beginning
            initial_buffer_seconds = self.config.get('initial_buffer_seconds', 300)
            self._current_time_idx = initial_buffer_seconds if len(self._all_timestamps) > initial_buffer_seconds else 0
            self.logger.info(f"Reset to index {self._current_time_idx}")

        # Set current timestamp
        if self._current_time_idx < len(self._all_timestamps):
            self.current_timestamp_utc = self._all_timestamps[self._current_time_idx]

        # Return the current market state
        return self.get_current_market_state()

    def close(self):
        """Close the simulator and release resources."""
        self.logger.info("Closing MarketSimulator")
        # Clear any large data structures
        self._precomputed_states.clear()
        self._all_timestamps.clear()