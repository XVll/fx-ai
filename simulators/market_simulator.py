# market_simulator.py
import logging
from collections import deque
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import bisect
from typing import Any, Dict, List, Optional, Tuple, Union

from config.config import MarketConfig, FeatureConfig
from data.data_manager import DataManager

# Default market hours in Eastern Time
DEFAULT_MARKET_HOURS_ET = {
    "PREMARKET_START_ET": time(4, 0, 0),
    "REGULAR_START_ET": time(9, 30, 0),
    "REGULAR_END_ET": time(15, 59, 59),
    "POSTMARKET_END_ET": time(19, 59, 59),
    "SESSION_START_ET": time(4, 0, 0),
    "SESSION_END_ET": time(19, 59, 59),
    "TIMEZONE_ET": "America/New_York"
}
PREVIOUS_DAY_CLOSING_TIME_ET = time(19, 59, 59)


class MarketSimulator:
    """
    Optimized market simulator that efficiently handles different data timeframes.

    - High-frequency data (trades, quotes) loaded only for rolling window
    - Medium and low-frequency data (1m, 5m bars) loaded directly from provider
    - Proper handling of session boundaries and lookbacks
    """
    EPSILON = 1e-9

    def __init__(self,
                 symbol: str,
                 data_manager: DataManager,
                 market_config: MarketConfig,
                 feature_config: FeatureConfig,
                 mode: str = "backtesting",
                 np_random: Optional[np.random.Generator] = None,
                 start_time: Optional[Union[str, datetime]] = None,
                 end_time: Optional[Union[str, datetime]] = None,
                 logger: Optional[logging.Logger] = None):

        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.data_manager = data_manager
        self.market_config = market_config
        self.feature_config = feature_config
        self.mode = mode
        self.np_random = np_random or np.random.default_rng()

        # Setup market hours and timezone
        self._setup_market_hours_and_timezone()

        # Actual trading session times
        self.session_start_utc = self._parse_datetime_to_utc(start_time)
        self.session_end_utc = self._parse_datetime_to_utc(end_time)

        # Data storage
        self._precomputed_states: Dict[datetime, Dict[str, Any]] = {}
        self._agent_timeline_utc: List[datetime] = []

        # Window sizes for different frequencies
        self.hf_window_size = max(1, self.feature_config.hf_seq_len * 2)  # Double for safety
        self.mf_window_size = max(1, self.feature_config.mf_seq_len)
        self.lf_window_size = max(1, self.feature_config.lf_seq_len)

        # Current state
        self.current_timestamp_utc: Optional[datetime] = None
        self._current_agent_time_idx: int = -1

        # Raw data storage
        self.raw_trades_df: pd.DataFrame = pd.DataFrame()
        self.raw_quotes_df: pd.DataFrame = pd.DataFrame()
        self.raw_1m_bars_df: pd.DataFrame = pd.DataFrame()
        self.raw_5m_bars_df: pd.DataFrame = pd.DataFrame()
        self.raw_1s_bars_provider_df: pd.DataFrame = pd.DataFrame()
        self.historical_1d_bars_df: pd.DataFrame = pd.DataFrame()

        # Initialize simulator
        self._initialize_simulator()

    def _setup_market_hours_and_timezone(self):
        """Set up market hours and timezone from config."""
        cfg_hours = self.market_config.get('market_hours', {})
        self.exchange_timezone_str = cfg_hours.get('TIMEZONE', DEFAULT_MARKET_HOURS_ET['TIMEZONE_ET'])
        self.exchange_tz = ZoneInfo(self.exchange_timezone_str)

        self.session_start_time_local = cfg_hours.get('SESSION_START_ET', DEFAULT_MARKET_HOURS_ET['SESSION_START_ET'])
        self.session_end_time_local = cfg_hours.get('SESSION_END_ET', DEFAULT_MARKET_HOURS_ET['SESSION_END_ET'])
        self.prev_day_close_time_local = cfg_hours.get('PREVIOUS_DAY_CLOSING_TIME_ET', PREVIOUS_DAY_CLOSING_TIME_ET)

        self.regular_market_start_local = cfg_hours.get('REGULAR_START_ET', DEFAULT_MARKET_HOURS_ET['REGULAR_START_ET'])
        self.regular_market_end_local = cfg_hours.get('REGULAR_END_ET', DEFAULT_MARKET_HOURS_ET['REGULAR_END_ET'])

    def _parse_datetime_to_utc(self, dt_input: Optional[Union[str, datetime]]) -> Optional[datetime]:
        """Convert string or datetime to UTC timezone."""
        if dt_input is None:
            return None

        if isinstance(dt_input, datetime):
            # For datetime objects
            if dt_input.tzinfo is None:  # Naive, assume it's in exchange timezone
                try:
                    dt_with_tz = datetime.combine(dt_input.date(), dt_input.time(), tzinfo=self.exchange_tz)
                    return dt_with_tz.astimezone(ZoneInfo("UTC"))
                except Exception as e:
                    self.logger.error(f"Error converting naive datetime to UTC: {e}")
                    return dt_input.replace(tzinfo=ZoneInfo("UTC"))
            return dt_input.astimezone(ZoneInfo("UTC"))

        try:  # String
            parsed_dt = pd.Timestamp(dt_input)
            if parsed_dt.tzinfo is None:
                dt_with_tz = datetime.combine(parsed_dt.date(), parsed_dt.time(), tzinfo=self.exchange_tz)
                return dt_with_tz.astimezone(ZoneInfo("UTC"))
            return parsed_dt.to_pydatetime().astimezone(ZoneInfo("UTC"))
        except Exception as e:
            self.logger.error(f"Failed to parse datetime '{dt_input}': {e}")
            return None

    def _initialize_simulator(self):
        """Initialize the market simulator."""
        self.logger.info(f"Initializing MarketSimulator for {self.symbol} in {self.mode} mode.")

        # Load data efficiently based on feature requirements
        self._load_data_for_simulation()

        if self.mode == 'backtesting':
            if not self.session_start_utc or not self.session_end_utc:
                self.logger.error("Backtesting mode requires session_start_utc and session_end_utc.")
                return
            self._precompute_timeline_states()
            self.reset()  # Set initial agent time
        elif self.mode == 'live':
            # Handle live mode initialization
            self.logger.info("Live mode initialized, will stream data as it arrives")
            self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _calculate_data_loading_ranges(self) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Calculate optimal data loading ranges for different timeframes.

        Returns:
            Dictionary with data types as keys and (start_time, end_time) as values
        """
        if not self.session_start_utc:
            self.logger.error("Cannot calculate loading ranges: session_start_utc is not defined.")
            return {}

        # Determine session date for day boundary calculations
        session_date_local = self.session_start_utc.astimezone(self.exchange_tz).date()

        # Agent session timeline
        agent_range_start = self.session_start_utc
        agent_range_end = self.session_end_utc or (
                    self.session_start_utc + timedelta(hours=16))  # Default to 16 hours if not specified

        # 1. High frequency data (trades, quotes) - only needed for the rolling window
        hf_lookback_seconds = self.hf_window_size + 60  # Add a small buffer
        hf_start = agent_range_start - timedelta(seconds=hf_lookback_seconds)
        hf_end = agent_range_end

        # 2. Medium frequency data (1m bars) - load full session
        # Create a UTC datetime for the start of pre-market on session date
        # First create a local timezone datetime for session date at session_start_time_local
        session_premarket_start_local = datetime.combine(
            session_date_local,
            self.session_start_time_local,
            tzinfo=self.exchange_tz
        )
        # Convert to UTC
        session_premarket_start_utc = session_premarket_start_local.astimezone(ZoneInfo("UTC"))

        # For bars, we need additional lookback for feature calculation
        mf_lookback_minutes = self.mf_window_size + 10  # Add buffer
        mf_start = session_premarket_start_utc - timedelta(minutes=mf_lookback_minutes)
        mf_end = agent_range_end

        # 3. Low frequency data (5m bars) - load full session plus lookback
        lf_lookback_minutes = self.lf_window_size * 5 + 30  # lf_window_size * 5-minute intervals + buffer
        lf_start = session_premarket_start_utc - timedelta(minutes=lf_lookback_minutes)
        lf_end = agent_range_end

        # 4. Daily bars - load historical data
        daily_end = session_premarket_start_utc
        daily_start = daily_end - timedelta(days=self.market_config.get('historical_daily_bars_lookback_days', 730))

        # 5. General data buffer for all timeframes
        data_load_day_buffer = self.market_config.get('data_load_day_buffer', 2)

        # Apply buffer to HF data
        hf_start -= timedelta(days=data_load_day_buffer)

        return {
            "hf": (hf_start, hf_end),
            "mf": (mf_start, mf_end),
            "lf": (lf_start, lf_end),
            "daily": (daily_start, daily_end)
        }

    def _load_data_for_simulation(self):
        """Load only the required data for simulation efficiently."""
        loading_ranges = self._calculate_data_loading_ranges()
        if not loading_ranges:
            self.logger.error("Failed to calculate data loading ranges.")
            return

        self.logger.info(f"Data loading ranges for {self.symbol}:")
        for timeframe, (start, end) in loading_ranges.items():
            self.logger.info(f"  {timeframe}: {start} to {end}")

        # 1. Load high-frequency data (trades, quotes) only for the window period
        hf_start, hf_end = loading_ranges["hf"]
        hf_data = self.data_manager.load_data(
            symbols=[self.symbol],
            start_time=hf_start,
            end_time=hf_end,
            data_types=["trades", "quotes"]
        )
        symbol_hf_data = hf_data.get(self.symbol, {})
        self.raw_trades_df = self._prepare_dataframe(symbol_hf_data.get("trades"), ['price', 'size'])
        self.raw_quotes_df = self._prepare_dataframe(symbol_hf_data.get("quotes"),
                                                     ['bid_price', 'ask_price', 'bid_size', 'ask_size'])

        # 2. Load 1-minute bars directly
        mf_start, mf_end = loading_ranges["mf"]
        try:
            self.raw_1m_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1m",
                start_time=mf_start,
                end_time=mf_end
            )
            self.raw_1m_bars_df = self._prepare_dataframe(self.raw_1m_bars_df,
                                                          ['open', 'high', 'low', 'close', 'volume'])
            self.logger.info(f"Loaded {len(self.raw_1m_bars_df)} 1-minute bars")
        except Exception as e:
            self.logger.error(f"Error loading 1-minute bars: {e}")
            self.raw_1m_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            self.raw_1m_bars_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')

        # 3. Load 5-minute bars directly
        lf_start, lf_end = loading_ranges["lf"]
        try:
            self.raw_5m_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="5m",
                start_time=lf_start,
                end_time=lf_end
            )
            self.raw_5m_bars_df = self._prepare_dataframe(self.raw_5m_bars_df,
                                                          ['open', 'high', 'low', 'close', 'volume'])
            self.logger.info(f"Loaded {len(self.raw_5m_bars_df)} 5-minute bars")
        except Exception as e:
            self.logger.error(f"Error loading 5-minute bars: {e}")
            self.raw_5m_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            self.raw_5m_bars_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')

        # 4. Load daily bars for historical context
        daily_start, daily_end = loading_ranges["daily"]
        try:
            self.historical_1d_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1d",
                start_time=daily_start,
                end_time=daily_end
            )
            self.historical_1d_bars_df = self._prepare_dataframe(
                self.historical_1d_bars_df,
                ['open', 'high', 'low', 'close', 'volume', 'close_8pm']
            )

            # Add close_8pm if missing
            if 'close_8pm' not in self.historical_1d_bars_df.columns and 'close' in self.historical_1d_bars_df.columns:
                self.historical_1d_bars_df['close_8pm'] = self.historical_1d_bars_df['close']

            self.logger.info(f"Loaded {len(self.historical_1d_bars_df)} historical daily bars")
        except Exception as e:
            self.logger.error(f"Error loading historical daily bars: {e}")
            self.historical_1d_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm'])
            self.historical_1d_bars_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')

    def _prepare_dataframe(self, df: Optional[pd.DataFrame], required_cols: List[str]) -> pd.DataFrame:
        """Prepare DataFrame for use in simulation by ensuring proper timezone and required columns."""
        # Return empty DataFrame with proper structure if input is None or empty
        if df is None or df.empty:
            empty_df = pd.DataFrame(columns=required_cols)
            empty_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
            return empty_df

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("Input DataFrame does not have a DatetimeIndex. Returning empty DataFrame.")
            empty_df = pd.DataFrame(columns=required_cols)
            empty_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
            return empty_df

        # Convert to UTC if needed
        try:
            if df.index.tz is None:
                df = df.tz_localize('UTC')
            elif str(df.index.tz) != 'UTC':
                df = df.tz_convert('UTC')
        except Exception as e:
            self.logger.error(f"Error converting timezone: {e}. Returning empty DataFrame.")
            empty_df = pd.DataFrame(columns=required_cols)
            empty_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
            return empty_df

        # Ensure all required columns are present
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df.sort_index()

    def _get_previous_trading_day_close(self, current_session_date_utc: datetime.date) -> Optional[float]:
        """Get the closing price of the previous trading day."""
        if self.historical_1d_bars_df.empty:
            return None

        try:
            # Convert dates properly to ensure correct comparison
            date_series = pd.Series(self.historical_1d_bars_df.index.date)
            relevant_bars = self.historical_1d_bars_df[date_series < current_session_date_utc]

            if relevant_bars.empty:
                return None

            close_col = 'close_8pm' if 'close_8pm' in relevant_bars.columns else 'close'
            prev_close = relevant_bars.iloc[-1][close_col]
            return float(prev_close) if pd.notna(prev_close) else None
        except Exception as e:
            self.logger.error(f"Error getting previous day close: {e}")
            return None

    def _precompute_timeline_states(self):
        """
        Precompute market states for every second in the agent's timeline.
        Efficiently builds states using directly loaded data instead of deriving everything.
        """
        self.logger.info(f"Starting precomputation of timeline states for {self.symbol}.")
        if not self.session_start_utc or not self.session_end_utc:
            self.logger.error("Cannot precompute: session_start_utc or session_end_utc not set.")
            return

        # Build the complete 1-second timeline for the agent's session
        current_ts = self.session_start_utc
        agent_session_timeline = []
        while current_ts <= self.session_end_utc:
            agent_session_timeline.append(current_ts)
            current_ts += timedelta(seconds=1)
        self._agent_timeline_utc = agent_session_timeline

        if not self._agent_timeline_utc:
            self.logger.error("Agent session timeline is empty. Check session start/end times.")
            return

        self.logger.info(f"Agent visible timeline: {self._agent_timeline_utc[0]} to {self._agent_timeline_utc[-1]}")

        # Initialize tracking variables for LOCF (Last Observation Carried Forward)
        locf_current_price: Optional[float] = None
        locf_best_bid_price: Optional[float] = None
        locf_best_ask_price: Optional[float] = None
        locf_best_bid_size: int = 0
        locf_best_ask_size: int = 0

        # Initialize tracking variables for day handling
        current_processing_day_local: Optional[datetime.date] = None
        intraday_high: Optional[float] = None
        intraday_low: Optional[float] = None
        previous_day_close_price: Optional[float] = None

        # Initialize deques for the rolling windows
        rolling_1s_event_data = deque(maxlen=self.hf_window_size)

        # Directly use the preloaded bars for medium and low frequency data
        # Extract the 1-minute and 5-minute bars once, then access by timestamp
        completed_1m_bars_dict = {}
        completed_5m_bars_dict = {}

        # Convert bar DataFrames to dictionaries for faster lookup
        if not self.raw_1m_bars_df.empty:
            for idx, row in self.raw_1m_bars_df.iterrows():
                completed_1m_bars_dict[idx] = {
                    'timestamp_start': idx,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume', 0.0)
                }

        if not self.raw_5m_bars_df.empty:
            for idx, row in self.raw_5m_bars_df.iterrows():
                completed_5m_bars_dict[idx] = {
                    'timestamp_start': idx,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume', 0.0)
                }

        # Try to initialize LOCF values from initial quotes and trades
        try:
            if not self.raw_quotes_df.empty:
                initial_quotes = self.raw_quotes_df.head(10)  # Take a few initial quotes
                if not initial_quotes.empty:
                    last_quote = initial_quotes.iloc[-1]
                    if pd.notna(last_quote.get('bid_price')):
                        locf_best_bid_price = float(last_quote['bid_price'])
                    if pd.notna(last_quote.get('ask_price')):
                        locf_best_ask_price = float(last_quote['ask_price'])
                    locf_best_bid_size = int(last_quote['bid_size']) if pd.notna(last_quote.get('bid_size')) else 0
                    locf_best_ask_size = int(last_quote['ask_size']) if pd.notna(last_quote.get('ask_size')) else 0

            if not self.raw_trades_df.empty:
                initial_trades = self.raw_trades_df.head(10)  # Take a few initial trades
                if not initial_trades.empty:
                    last_trade = initial_trades.iloc[-1]
                    if pd.notna(last_trade.get('price')):
                        locf_current_price = float(last_trade['price'])
        except Exception as e:
            self.logger.warning(f"Error initializing LOCF values: {e}")

        # Process each second in the agent's timeline
        total_seconds = len(self._agent_timeline_utc)
        processed_count = 0

        for current_iter_ts in self._agent_timeline_utc:
            # Convert to local timezone for day boundary detection
            ts_local_datetime = current_iter_ts.astimezone(self.exchange_tz)
            ts_local_date = ts_local_datetime.date()

            # Handle day change
            if current_processing_day_local != ts_local_date:
                self.logger.debug(f"Processing new day {ts_local_date} at UTC {current_iter_ts}")
                current_processing_day_local = ts_local_date
                intraday_high = None
                intraday_low = None
                previous_day_close_price = self._get_previous_trading_day_close(current_processing_day_local)

            # Initialize variables for this second
            actual_1s_bar_data: Optional[Dict[str, Any]] = None
            trades_this_second: List[Dict] = []
            quotes_this_second: List[Dict] = []

            # Get trades and quotes for this second from raw data
            if not self.raw_trades_df.empty:
                trades_slice = self.raw_trades_df[
                    (self.raw_trades_df.index > current_iter_ts - timedelta(seconds=1)) &
                    (self.raw_trades_df.index <= current_iter_ts)
                    ]
                trades_this_second = [row.to_dict() for _, row in trades_slice.iterrows()]

            if not self.raw_quotes_df.empty:
                quotes_slice = self.raw_quotes_df[
                    (self.raw_quotes_df.index > current_iter_ts - timedelta(seconds=1)) &
                    (self.raw_quotes_df.index <= current_iter_ts)
                    ]
                quotes_this_second = [row.to_dict() for _, row in quotes_slice.iterrows()]

            # Build 1-second bar from trades if available
            if trades_this_second:
                # Use trades to create a bar
                actual_1s_bar_data = self._aggregate_trades_to_bar(trades_this_second, current_iter_ts)

            # Update LOCF values with new data
            if trades_this_second:
                locf_current_price = float(trades_this_second[-1]['price'])
            elif actual_1s_bar_data and pd.notna(actual_1s_bar_data.get('close')):
                locf_current_price = float(actual_1s_bar_data['close'])

            # Update BBO from quotes
            if quotes_this_second:
                last_quote = quotes_this_second[-1]
                if pd.notna(last_quote.get('bid_price')):
                    locf_best_bid_price = float(last_quote['bid_price'])
                if pd.notna(last_quote.get('ask_price')):
                    locf_best_ask_price = float(last_quote['ask_price'])
                if pd.notna(last_quote.get('bid_size')):
                    locf_best_bid_size = int(last_quote['bid_size'])
                if pd.notna(last_quote.get('ask_size')):
                    locf_best_ask_size = int(last_quote['ask_size'])

            # Update intraday high/low
            if actual_1s_bar_data:
                bar_high = actual_1s_bar_data.get('high')
                bar_low = actual_1s_bar_data.get('low')
                if pd.notna(bar_high):
                    intraday_high = bar_high if intraday_high is None else max(intraday_high, bar_high)
                if pd.notna(bar_low):
                    intraday_low = bar_low if intraday_low is None else min(intraday_low, bar_low)

            # Add event data to the rolling window
            event_for_deque = {
                'timestamp': current_iter_ts,
                'bar': actual_1s_bar_data,
                'trades': trades_this_second,
                'quotes': quotes_this_second
            }
            rolling_1s_event_data.append(event_for_deque)

            # Get 1-minute bars for this timestamp (window)
            minute_start_ts = current_iter_ts.replace(second=0, microsecond=0)
            completed_1m_bars_window = []

            # Find relevant 1-minute bars for this timestamp's window
            for i in range(self.mf_window_size):
                bar_ts = minute_start_ts - timedelta(minutes=i)
                if bar_ts in completed_1m_bars_dict:
                    completed_1m_bars_window.insert(0, completed_1m_bars_dict[bar_ts])

            # Get 5-minute bars for this timestamp (window)
            current_minute_val = current_iter_ts.minute
            five_min_slot_start_minute = (current_minute_val // 5) * 5
            five_min_start_ts = current_iter_ts.replace(minute=five_min_slot_start_minute, second=0, microsecond=0)
            completed_5m_bars_window = []

            # Find relevant 5-minute bars for this timestamp's window
            for i in range(self.lf_window_size):
                bar_ts = five_min_start_ts - timedelta(minutes=5 * i)
                if bar_ts in completed_5m_bars_dict:
                    completed_5m_bars_window.insert(0, completed_5m_bars_dict[bar_ts])

            # Store state for this timestamp
            self._precomputed_states[current_iter_ts] = {
                'timestamp_utc': current_iter_ts,
                'current_market_session': self._determine_market_session(current_iter_ts),
                'current_price': locf_current_price,
                'best_bid_price': locf_best_bid_price,
                'best_ask_price': locf_best_ask_price,
                'best_bid_size': locf_best_bid_size,
                'best_ask_size': locf_best_ask_size,
                'intraday_high': intraday_high,
                'intraday_low': intraday_low,
                'current_1s_bar': actual_1s_bar_data,
                'rolling_1s_data_window': list(rolling_1s_event_data),
                'completed_1m_bars_window': completed_1m_bars_window,
                'completed_5m_bars_window': completed_5m_bars_window,
                'historical_1d_bars': self.historical_1d_bars_df.copy() if not self.historical_1d_bars_df.empty else pd.DataFrame(),
                'previous_day_close_price': previous_day_close_price
            }

            # Progress logging
            processed_count += 1
            if processed_count % 10000 == 0 or processed_count == total_seconds:
                progress_pct = (processed_count / total_seconds) * 100
                self.logger.info(
                    f"State precomputation progress: {processed_count}/{total_seconds} seconds processed ({progress_pct:.1f}%). Current: {current_iter_ts}")

        self.logger.info(
            f"Finished precomputing states. {len(self._precomputed_states)} states stored for the agent's timeline.")

    def _aggregate_trades_to_bar(self, trades_in_interval: List[Dict], bar_timestamp: datetime) -> Optional[
        Dict[str, Any]]:
        """Aggregate trades into OHLCV bar data."""
        if not trades_in_interval:
            return None

        try:
            # Extract valid prices and sizes
            prices = [float(t['price']) for t in trades_in_interval if pd.notna(t.get('price'))]
            sizes = [float(t['size']) for t in trades_in_interval if pd.notna(t.get('size'))]

            if not prices or not sizes:
                return None

            total_volume = sum(sizes)
            if total_volume < self.EPSILON:
                # Handle cases with negligible volume
                price_val = prices[0]
                return {'timestamp': bar_timestamp, 'open': price_val, 'high': price_val, 'low': price_val,
                        'close': price_val, 'volume': 0.0, 'vwap': price_val}

            # Calculate VWAP
            value_sum = sum(p * s for p, s in zip(prices, sizes))

            return {
                'timestamp': bar_timestamp,
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'volume': total_volume,
                'vwap': value_sum / total_volume if total_volume > self.EPSILON else prices[-1]
            }
        except Exception as e:
            self.logger.warning(f"Error aggregating trades to bar: {e}")
            return None

    def _determine_market_session(self, timestamp_utc: datetime) -> str:
        """Determine the market session (PREMARKET, REGULAR, POSTMARKET, CLOSED) for a timestamp."""
        try:
            local_time = timestamp_utc.astimezone(self.exchange_tz).time()

            if self.session_start_time_local <= local_time <= self.session_end_time_local:
                if self.regular_market_start_local <= local_time <= self.regular_market_end_local:
                    return "REGULAR"
                elif local_time < self.regular_market_start_local:
                    return "PREMARKET"
                else:
                    return "POSTMARKET"
            return "CLOSED"
        except Exception as e:
            self.logger.warning(f"Error determining market session: {e}")
            return "UNKNOWN"

    def _create_empty_state(self, timestamp_utc: datetime) -> Dict[str, Any]:
        """Create an empty state for a timestamp when no precomputed state is available."""
        empty_state = {
            'timestamp_utc': timestamp_utc,
            'current_market_session': self._determine_market_session(timestamp_utc),
            'current_price': None, 'best_bid_price': None, 'best_ask_price': None,
            'best_bid_size': 0, 'best_ask_size': 0, 'intraday_high': None, 'intraday_low': None,
            'current_1s_bar': None,
            'rolling_1s_data_window': [],
            'completed_1m_bars_window': [],
            'completed_5m_bars_window': [],
            'historical_1d_bars': self.historical_1d_bars_df.copy() if not self.historical_1d_bars_df.empty else pd.DataFrame(),
            'previous_day_close_price': None
        }

        # Try to find a reasonable price for the state
        try:
            # First try: Get from first agent state if available
            if self._agent_timeline_utc and timestamp_utc < self._agent_timeline_utc[0]:
                first_agent_state_ts = self._agent_timeline_utc[0]
                if first_agent_state_ts in self._precomputed_states:
                    first_state = self._precomputed_states[first_agent_state_ts]
                    empty_state['previous_day_close_price'] = first_state.get('previous_day_close_price')
                    empty_state['current_price'] = first_state.get('current_price')
                    return empty_state

            # Second try: Check day's price history
            current_processing_day_local = timestamp_utc.astimezone(self.exchange_tz).date()
            prev_close = self._get_previous_trading_day_close(current_processing_day_local)
            if prev_close is not None:
                empty_state['previous_day_close_price'] = prev_close
                empty_state['current_price'] = prev_close  # Fallback to prev close
                return empty_state

            # Third try: Find closest earlier precomputed state
            if self._precomputed_states:
                sorted_keys = sorted(self._precomputed_states.keys())
                idx = bisect.bisect_left(sorted_keys, timestamp_utc)
                if idx > 0:
                    last_known_ts = sorted_keys[idx - 1]
                    last_state = self._precomputed_states[last_known_ts]
                    empty_state['previous_day_close_price'] = last_state.get('previous_day_close_price')
                    empty_state['current_price'] = last_state.get('current_price')
                elif sorted_keys:  # Use first state if timestamp is before all known states
                    first_state = self._precomputed_states[sorted_keys[0]]
                    empty_state['previous_day_close_price'] = first_state.get('previous_day_close_price')
                    empty_state['current_price'] = first_state.get('current_price')
        except Exception as e:
            self.logger.warning(f"Error setting values in empty state: {e}")

        return empty_state

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        """Get the market state at the current simulation time."""
        if self._current_agent_time_idx < 0 or self._current_agent_time_idx >= len(self._agent_timeline_utc):
            self.logger.warning(
                f"Current agent time index {self._current_agent_time_idx} is invalid for timeline of length {len(self._agent_timeline_utc)}.")
            if self.current_timestamp_utc:
                return self.get_state_at_time(self.current_timestamp_utc)
            # Fallback
            now_utc = datetime.now(ZoneInfo("UTC")).replace(
                microsecond=0) if self.session_start_utc is None else self.session_start_utc
            return self._create_empty_state(now_utc)

        current_ts_on_timeline = self._agent_timeline_utc[self._current_agent_time_idx]
        if self.current_timestamp_utc != current_ts_on_timeline:
            self.logger.warning(
                f"Internal timestamp mismatch: self.current_timestamp_utc {self.current_timestamp_utc} vs timeline {current_ts_on_timeline}. Syncing.")
            self.current_timestamp_utc = current_ts_on_timeline

        state = self._precomputed_states.get(self.current_timestamp_utc)
        if state is None:
            self.logger.warning(
                f"No precomputed state found for current timestamp {self.current_timestamp_utc}. Creating empty state.")
            return self._create_empty_state(self.current_timestamp_utc)

        return state

    def get_state_at_time(self, timestamp: datetime, tolerance_seconds: int = 1) -> Optional[Dict[str, Any]]:
        """Get the market state at a specific timestamp, with optional tolerance."""
        # Ensure timestamp is in UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
        elif timestamp.tzinfo != ZoneInfo("UTC"):
            timestamp = timestamp.astimezone(ZoneInfo("UTC"))

        # Try to get exact match first
        exact_match = self._precomputed_states.get(timestamp)
        if exact_match:
            return exact_match

        # If no exact match, handle various cases
        if not self._agent_timeline_utc:
            self.logger.warning(f"State requested for {timestamp}, but agent timeline is empty.")
            return self._create_empty_state(timestamp)

        # Try to find closest state within tolerance
        try:
            idx = bisect.bisect_left(self._agent_timeline_utc, timestamp)

            # Check if there's a state right before the requested timestamp
            if idx > 0:
                candidate_ts = self._agent_timeline_utc[idx - 1]
                if (timestamp - candidate_ts) <= timedelta(seconds=tolerance_seconds):
                    return self._precomputed_states.get(candidate_ts)
        except Exception as e:
            self.logger.warning(f"Error finding closest state: {e}")

        # Create empty state as fallback
        self.logger.warning(f"State for {timestamp} not found (tolerance {tolerance_seconds}s). Returning empty state.")
        return self._create_empty_state(timestamp)

    def step(self) -> bool:
        """
        Advance the simulation by one step.

        Returns:
            bool: True if successfully stepped to next state, False if done or error
        """
        if self.is_done():
            self.logger.info("Step called, but simulation is already done.")
            return False

        try:
            self._current_agent_time_idx += 1
            if self._current_agent_time_idx < len(self._agent_timeline_utc):
                self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
                return True
            else:
                # We've reached the end
                self._current_agent_time_idx = len(self._agent_timeline_utc) - 1
                if self._agent_timeline_utc:
                    self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
                else:
                    self.current_timestamp_utc = None
                self.logger.info("Reached end of timeline data.")
                return False
        except Exception as e:
            self.logger.error(f"Error during step: {e}")
            return False

    def is_done(self) -> bool:
        """Check if the simulation has reached the end."""
        if not self._agent_timeline_utc:
            return True
        return self._current_agent_time_idx >= len(self._agent_timeline_utc) - 1

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Reset the simulator to an initial state.

        Args:
            options: Optional dict with settings like 'random_start', 'start_time_offset_seconds'

        Returns:
            The initial market state after reset
        """
        options = options or {}
        self.logger.info(f"Resetting MarketSimulator with options: {options}")

        if not self._agent_timeline_utc:
            self.logger.error("Cannot reset: Agent timeline is not available.")
            default_ts = self.session_start_utc or datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
            return self._create_empty_state(default_ts)

        # Determine earliest valid start point after any initial buffer
        buffer_seconds = self.market_config.get('initial_buffer_seconds', 0)
        min_start_delay_from_session_start = timedelta(seconds=buffer_seconds)
        earliest_possible_agent_start_time = self._agent_timeline_utc[0] + min_start_delay_from_session_start

        start_idx_after_buffer = bisect.bisect_left(self._agent_timeline_utc, earliest_possible_agent_start_time)
        start_idx_after_buffer = min(start_idx_after_buffer, len(self._agent_timeline_utc) - 1)

        # Handle random start if requested
        if options.get('random_start', False):
            if start_idx_after_buffer < len(self._agent_timeline_utc) - 1:
                # Use np_random from the instance (expected to be set by trading_env)
                self._current_agent_time_idx = self.np_random.integers(start_idx_after_buffer,
                                                                       len(self._agent_timeline_utc))
            else:
                self._current_agent_time_idx = start_idx_after_buffer
            self.logger.info(f"Random reset to index {self._current_agent_time_idx} within agent timeline.")
        else:
            self._current_agent_time_idx = start_idx_after_buffer
            self.logger.info(f"Reset to index {self._current_agent_time_idx} (after initial buffer).")

        # Set current timestamp and return state
        if self._current_agent_time_idx < len(self._agent_timeline_utc):
            self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
            self.logger.info(f"Simulator reset. Current agent time: {self.current_timestamp_utc}")
            return self.get_current_market_state()
        else:
            self.logger.error("Reset resulted in invalid agent time index.")
            self._current_agent_time_idx = 0
            self.current_timestamp_utc = self._agent_timeline_utc[0]
            return self.get_current_market_state()

    def get_symbol_info(self) -> Dict[str, Any]:
        """Get information about the current symbol."""
        return {
            "symbol": self.symbol,
            "total_shares_outstanding": self.market_config.get('total_shares_outstanding', 100_000_000),
        }

    def close(self) -> None:
        """Clean up resources used by the simulator."""
        self.logger.info("Closing MarketSimulator")
        self._precomputed_states.clear()
        self._agent_timeline_utc.clear()
        if hasattr(self, 'raw_trades_df'): del self.raw_trades_df
        if hasattr(self, 'raw_quotes_df'): del self.raw_quotes_df
        if hasattr(self, 'raw_1m_bars_df'): del self.raw_1m_bars_df
        if hasattr(self, 'raw_5m_bars_df'): del self.raw_5m_bars_df
        if hasattr(self, 'raw_1s_bars_provider_df'): del self.raw_1s_bars_provider_df
        if hasattr(self, 'historical_1d_bars_df'): del self.historical_1d_bars_df
        self.logger.info("MarketSimulator closed and resources cleared.")