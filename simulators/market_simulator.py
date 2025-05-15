# market_simulator_v2.py
import logging
from collections import deque
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Deque
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # Import ZoneInfoNotFoundError
import numpy as np
import pandas as pd

DEFAULT_MARKET_HOURS = {
    "PREMARKET_START": "04:00:00",
    "PREMARKET_END": "09:29:59",
    "REGULAR_START": "09:30:00",
    "REGULAR_END": "15:59:59",
    "POSTMARKET_START": "16:00:00",
    "POSTMARKET_END": "20:00:00",
    "TIMEZONE": "America/New_York"
}


class MarketSimulatorV2:
    def __init__(self,
                 symbol: str,
                 data_manager: Any,  # Should be a DataManager-like object
                 mode: str = 'backtesting',  # 'backtesting' or 'live'
                 start_time: Optional[str | datetime] = None,
                 end_time: Optional[str | datetime] = None,
                 config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the market simulator.

        Args:
            symbol: Trading symbol.
            data_manager: DataManager instance for retrieving data.
            mode: 'backtesting' or 'live'.
            start_time: Start time for historical data (backtesting).
            end_time: End time for historical data (backtesting).
            config: Configuration dictionary.
            logger: Optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.data_manager = data_manager
        self.config = config or {}
        self.mode = mode

        # --- Timezone and Market Hours Setup ---
        _market_hours_cfg_raw = self.config.get('market_hours', DEFAULT_MARKET_HOURS)
        self.exchange_timezone_str = _market_hours_cfg_raw.get('TIMEZONE', DEFAULT_MARKET_HOURS['TIMEZONE'])
        self.exchange_timezone = ZoneInfo(self.exchange_timezone_str)

        self.market_hours: Dict[str, time] = {}
        for key, default_val_str in DEFAULT_MARKET_HOURS.items():
            if key.endswith(("_START", "_END")):
                time_str = _market_hours_cfg_raw.get(key, default_val_str)
                self.market_hours[key] = datetime.strptime(time_str, "%H:%M:%S").time()

        # --- Time Range Setup ---
        self.start_time_utc = self._parse_datetime(start_time)
        self.end_time_utc = self._parse_datetime(end_time)

        # --- Data Storage ---
        self.all_trades_data: Optional[pd.DataFrame] = None
        self.all_quotes_data: Optional[pd.DataFrame] = None
        self.all_1d_bars_data: Optional[pd.DataFrame] = None
        self.all_1s_bars_data: Optional[pd.DataFrame] = None

        # --- Current Time and Index ---
        self.current_timestamp_utc: Optional[datetime] = None
        self._current_bar_idx: int = 0


        # --- Intraday State Tracking ---
        self.current_market_session: str = "CLOSED"
        self._current_day_for_intraday_tracking: Optional[datetime.date] = None
        self.intraday_high: Optional[float] = None
        self.intraday_low: Optional[float] = None

        # --- Rolling Data Windows ---
        self.rolling_1s_data_window_size = self.config.get('rolling_1s_data_window_size', 60 * 60)
        self.rolling_1m_data_window_size = self.config.get('rolling_1m_data_window_size', 120)
        self.rolling_5m_data_window_size = self.config.get('rolling_5m_data_window_size', 48)
        self.rolling_1s_data_window: Deque[Dict[str, Any]] = deque(maxlen=self.rolling_1s_data_window_size)
        self.rolling_1m_data_window: Deque[Dict[str, Any]] = deque(maxlen=self.rolling_1m_data_window_size)
        self.rolling_5m_data_window: Deque[Dict[str, Any]] = deque(maxlen=self.rolling_5m_data_window_size)

        # --- Bar Aggregation State ---
        self._current_1m_bar_agg: Optional[Dict[str, Any]] = None
        self._current_5m_bar_agg: Optional[Dict[str, Any]] = None

        # --- Live Data Buffers ---
        self._live_trades_buffer: List[Dict] = []
        self._live_quotes_buffer: List[Dict] = []

        if not self.data_manager:
            self.logger.warning("No data_manager provided. MarketSimulator will have limited functionality.")
            return

        self._load_initial_data()

        if self.mode == 'backtesting':
            if self.all_1s_bars_data is None or self.all_1s_bars_data.empty:
                self.logger.warning("No 1s bars generated from trade data. Backtesting may not function correctly.")
            self._initialize_for_backtesting()
        elif self.mode == 'live':
            self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
            self.logger.info(
                f"MarketSimulator initialized for LIVE trading. Current UTC time: {self.current_timestamp_utc}")
            self._initialize_for_live_trading()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'backtesting' or 'live'.")

        self.logger.info(f"MarketSimulator initialized for {self.symbol}. Mode: {self.mode}.")

    def _parse_datetime(self, dt_input: Optional[str | datetime]) -> Optional[datetime]:
        """Convert input to UTC datetime, with enhanced string parsing."""
        if dt_input is None:
            return None

        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None:
                self.logger.debug(f"Received naive datetime object: {dt_input}. Assuming it's UTC.")
                return dt_input.replace(tzinfo=ZoneInfo("UTC"))
            return dt_input.astimezone(ZoneInfo("UTC"))

        if isinstance(dt_input, str):
            # Attempt 1: Standard ISO format (e.g., YYYY-MM-DDTHH:MM:SSZ or ...+00:00)
            try:
                dt_obj = datetime.fromisoformat(dt_input.replace(" Z", "+00:00"))  # Handle space before Z if present
                if dt_obj.tzinfo is None:  # Should be aware if fromisoformat parsed TZ
                    self.logger.debug(f"Parsed '{dt_input}' via fromisoformat as naive. Assuming UTC.")
                    return dt_obj.replace(tzinfo=ZoneInfo("UTC"))
                return dt_obj.astimezone(ZoneInfo("UTC"))
            except ValueError:
                self.logger.debug(f"'{dt_input}' is not in standard ISO format recognizable by fromisoformat.")

            # Attempt 2: Parse "YYYY-MM-DD HH:MM:SS TZNAME" (e.g., "2023-01-01 10:00:00 America/New_York")
            parts = dt_input.rsplit(' ', 1)
            if len(parts) == 2:
                dt_str_part, tz_str_part = parts
                try:
                    naive_dt = datetime.strptime(dt_str_part, '%Y-%m-%d %H:%M:%S')
                    tz = ZoneInfo(tz_str_part)  # This can raise ZoneInfoNotFoundError
                    localized_dt = naive_dt.replace(tzinfo=tz)
                    self.logger.debug(f"Parsed '{dt_input}' as datetime '{naive_dt}' in timezone '{tz_str_part}'.")
                    return localized_dt.astimezone(ZoneInfo("UTC"))
                except (ValueError, ZoneInfoNotFoundError) as e_custom:  # Catch parsing or invalid TZ
                    self.logger.debug(f"Could not parse '{dt_input}' as 'YYYY-MM-DD HH:MM:SS TZNAME': {e_custom}")
                    # Fall through to pandas if this specific format fails

            # Attempt 3: Pandas to_datetime as a general fallback
            try:
                # Pandas is generally good with a wide variety of string formats
                dt_obj = pd.to_datetime(dt_input)
                if dt_obj.tzinfo is None:
                    # If pandas parses it as naive, we assume it's UTC as a convention for this simulator's inputs.
                    # Or, the user should provide timezone-aware strings.
                    self.logger.warning(
                        f"Parsed '{dt_input}' with pandas as a naive datetime. Assuming this time is in UTC."
                    )
                    return dt_obj.replace(tzinfo=ZoneInfo("UTC"))
                return dt_obj.astimezone(ZoneInfo("UTC"))
            except Exception as e_pd:
                self.logger.error(
                    f"Failed to parse datetime string '{dt_input}' with all attempted methods. Pandas error: {e_pd}",
                    exc_info=True)
                return None

        self.logger.error(f"Invalid datetime input type provided: {type(dt_input)}")
        return None

    def _load_initial_data(self):
        """Load initial data from data_manager."""
        if not self.data_manager:
            self.logger.error("Cannot load data: data_manager is not available.")
            return

        self.logger.info(f"Loading initial data for {self.symbol}...")
        try:
            # Load trades for the requested time range (essential for 1s bars)
            if self.start_time_utc and self.end_time_utc:
                self.all_trades_data = self.data_manager.get_trades(
                    symbol=self.symbol,
                    start_time=self.start_time_utc,
                    end_time=self.end_time_utc
                )
                if self.all_trades_data is not None:
                    self.logger.info(f"Loaded {len(self.all_trades_data)} trades for {self.symbol}")
                    if not self.all_trades_data.index.is_monotonic_increasing:
                        self.all_trades_data.sort_index(inplace=True)

                # Load quotes (optional, but good for context)
                self.all_quotes_data = self.data_manager.get_quotes(
                    symbol=self.symbol,
                    start_time=self.start_time_utc,
                    end_time=self.end_time_utc
                )
                if self.all_quotes_data is not None:
                    self.logger.info(f"Loaded {len(self.all_quotes_data)} quotes for {self.symbol}")
                    if not self.all_quotes_data.index.is_monotonic_increasing:
                        self.all_quotes_data.sort_index(inplace=True)

            # Determine time range for daily bars (approx. 1 year back from end date)
            daily_end = self.end_time_utc or datetime.now(ZoneInfo("UTC"))
            daily_start_default = daily_end - timedelta(days=365)

            # If user provided start_time is earlier for daily history, use that
            daily_start = daily_start_default
            if self.start_time_utc and (self.start_time_utc - timedelta(days=365)) < daily_start_default:
                daily_start = self.start_time_utc - timedelta(days=365)

            self.all_1d_bars_data = self.data_manager.get_bars(
                symbol=self.symbol, timeframe="1d",
                start_time=daily_start, end_time=daily_end
            )
            if self.all_1d_bars_data is not None:
                self.logger.info(f"Loaded {len(self.all_1d_bars_data)} daily bars for {self.symbol}")

            # Generate 1s bars from trades (critical for simulation steps)
            if self.all_trades_data is not None and not self.all_trades_data.empty:
                self._generate_1s_bars_from_trades()
            else:
                self.logger.warning("No trade data loaded, cannot generate 1s bars.")

        except Exception as e:
            self.logger.error(f"Error loading initial data: {e}", exc_info=True)

    def _generate_1s_bars_from_trades(self):
        """Generate 1-second OHLCV bars from trade data."""
        if self.all_trades_data is None or self.all_trades_data.empty:
            self.logger.warning("Cannot generate 1s bars: No trade data available.")
            self.all_1s_bars_data = pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])  # Ensure it's an empty DF with columns
            return

        self.logger.info("Generating 1-second bars from trade data...")
        required_cols = ['price', 'size']
        if not all(col in self.all_trades_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.all_trades_data.columns]
            self.logger.error(f"Trade data missing required columns: {missing} for 1s bar generation.")
            self.all_1s_bars_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
            return

        try:
            resampled = (
                self.all_trades_data
                .assign(value=self.all_trades_data['price'] * self.all_trades_data['size'])
                .resample('1s')
                .agg(
                    open=('price', 'first'),
                    high=('price', 'max'),
                    low=('price', 'min'),
                    close=('price', 'last'),
                    volume=('size', 'sum'),
                    value_sum=('value', 'sum')  # temp column for vwap
                )
            )
            # Drop rows where 'open' is NaN (meaning no trades in that 1s interval)
            resampled.dropna(subset=['open'], inplace=True)
            if not resampled.empty:
                resampled['vwap'] = resampled['value_sum'] / resampled['volume']
                resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Forward fill VWAP for bars that had trades but result in NaN VWAP (e.g. volume was zero after some filtering)
                # This is less likely if we dropna on 'open' from 'price' and 'volume' comes from 'size'
                resampled['vwap'] = resampled['vwap'].ffill()
                # If first VWAP is NaN, backfill it. If all are NaN, it will remain NaN.
                resampled['vwap'] = resampled['vwap'].bfill()
            else:  # if resampled becomes empty after dropna
                resampled['vwap'] = pd.Series(dtype=float)

            self.all_1s_bars_data = resampled[['open', 'high', 'low', 'close', 'volume', 'vwap']].copy()
            self.logger.info(f"Generated {len(self.all_1s_bars_data)} 1-second bars.")
        except Exception as e:
            self.logger.error(f"Error generating 1s bars: {e}", exc_info=True)
            self.all_1s_bars_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])

    def _initialize_for_backtesting(self):
        """Initialize the simulator for backtesting mode."""
        if self.all_1s_bars_data is None or self.all_1s_bars_data.empty:
            self.logger.error("Cannot initialize backtesting: No 1s bars available.")
            return

        initial_buffer_seconds = self.config.get('initial_buffer_seconds', 300)

        # Determine the actual start of data to be used for simulation (respecting user's start_time)
        if self.start_time_utc:
            # Find the first data point at or after the user-defined start_time_utc
            try:
                data_start_idx = self.all_1s_bars_data.index.searchsorted(self.start_time_utc, side='left')
                if data_start_idx >= len(self.all_1s_bars_data.index):  # User start_time is past all data
                    self.logger.warning(
                        f"User start_time_utc {self.start_time_utc} is after the last available 1s bar data. Simulation may end immediately.")
                    self._current_bar_idx = len(self.all_1s_bars_data.index)
                    self.current_timestamp_utc = self.all_1s_bars_data.index[
                        -1] if not self.all_1s_bars_data.empty else None
                    return
                effective_data_start_ts = self.all_1s_bars_data.index[data_start_idx]
            except Exception as e:
                self.logger.error(
                    f"Error finding effective data start for {self.start_time_utc}: {e}. Defaulting to first bar.")
                effective_data_start_ts = self.all_1s_bars_data.index[0]
                data_start_idx = 0
        else:  # No start_time_utc provided by user, use the very first data point
            effective_data_start_ts = self.all_1s_bars_data.index[0]
            data_start_idx = 0
            self.logger.info(
                f"No specific start_time provided for backtesting, using earliest data point: {effective_data_start_ts}")

        # Agent's simulation should start after the buffer period from this effective_data_start_ts.
        agent_start_timestamp_utc = effective_data_start_ts + timedelta(seconds=initial_buffer_seconds)

        try:
            self._current_bar_idx = self.all_1s_bars_data.index.searchsorted(agent_start_timestamp_utc, side='left')
        except Exception as e:
            self.logger.warning(f"searchsorted failed for agent_start_timestamp_utc: {e}. Iterating.")
            temp_idx = data_start_idx  # Start searching from where our data effectively begins
            for i, ts in enumerate(self.all_1s_bars_data.index[data_start_idx:], start=data_start_idx):
                if ts >= agent_start_timestamp_utc:
                    temp_idx = i
                    break
            else:
                temp_idx = len(self.all_1s_bars_data)
            self._current_bar_idx = temp_idx

        if self._current_bar_idx >= len(self.all_1s_bars_data.index):
            self.logger.warning(f"Not enough data for configured start time/buffer. "
                                f"Agent start index {self._current_bar_idx} is at/beyond data length {len(self.all_1s_bars_data.index)}. "
                                f"Simulation may end immediately.")
            self.current_timestamp_utc = self.all_1s_bars_data.index[-1] if not self.all_1s_bars_data.empty else None
            # _current_bar_idx already set to len, so is_done() will be true
            return

        # Pre-fill window is from effective_data_start_ts up to (but not including) agent_start_timestamp_utc
        prefill_start_idx = data_start_idx
        prefill_end_idx = self._current_bar_idx  # Agent starts at _current_bar_idx, so prefill ends before it

        self.logger.info(
            f"Prefilling data from index {prefill_start_idx} (ts: {self.all_1s_bars_data.index[prefill_start_idx]}) "
            f"up to (exclusive) index {prefill_end_idx} (ts before: {self.all_1s_bars_data.index[prefill_end_idx - 1] if prefill_end_idx > prefill_start_idx else 'N/A'}).")

        for i in range(prefill_start_idx, prefill_end_idx):
            ts_utc = self.all_1s_bars_data.index[i]
            # self.current_timestamp_utc = ts_utc # Set for context during prefill internal calls
            self._update_intraday_state_and_session(ts_utc)

            one_sec_event_data = self._get_data_for_timestamp(ts_utc)
            if one_sec_event_data:
                self.rolling_1s_data_window.append(one_sec_event_data)
                latest_1s_bar = one_sec_event_data.get('bar')
                if latest_1s_bar:
                    if self.current_market_session != "CLOSED":  # Only update H/L if market is not closed during prefill
                        self._update_intraday_high_low(latest_1s_bar)
                    self._update_aggregate_bar(latest_1s_bar, 60, '_current_1m_bar_agg', self.rolling_1m_data_window)
                    self._update_aggregate_bar(latest_1s_bar, 300, '_current_5m_bar_agg', self.rolling_5m_data_window)

        # Set the definitive current_timestamp_utc for the agent's first step (which will process this bar)
        if self._current_bar_idx < len(self.all_1s_bars_data.index):
            self.current_timestamp_utc = self.all_1s_bars_data.index[self._current_bar_idx]
        else:
            self.current_timestamp_utc = self.all_1s_bars_data.index[-1] if not self.all_1s_bars_data.empty else None
            self.logger.warning("Agent start index is at or beyond data after prefill.")

        # Final state update for the day/session right before the agent's first bar is processed.
        # The intraday_high/low should reflect history *before* this bar.
        if self.current_timestamp_utc:
            self._update_intraday_state_and_session(self.current_timestamp_utc)

        self.logger.info(
            f"Backtesting initialized. Buffer filled. "
            f"Agent will start processing bar at index {self._current_bar_idx}, "
            f"UTC: {self.current_timestamp_utc}, "
            f"Exchange Time: {self.current_timestamp_utc.astimezone(self.exchange_timezone) if self.current_timestamp_utc else 'N/A'}"
        )

    def _initialize_for_live_trading(self):
        """Initialize the simulator for live trading mode."""
        self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
        self._update_intraday_state_and_session(self.current_timestamp_utc)  # Set initial day/session

        if hasattr(self.data_manager, 'initialize_live_data'):
            try:
                # Suggest what simulator might use or is capable of processing
                self.data_manager.initialize_live_data(
                    symbol=self.symbol,
                    timeframes=["1s", "1m", "5m", "1d"]
                )
                if hasattr(self.data_manager, 'add_trade_callback'):
                    self.data_manager.add_trade_callback(self.on_live_trade)
                if hasattr(self.data_manager, 'add_quote_callback'):
                    self.data_manager.add_quote_callback(self.on_live_quote)
                if hasattr(self.data_manager, 'add_bar_callback'):
                    self.data_manager.add_bar_callback(self.on_live_bar)
                self.logger.info(f"Registered live data callbacks for {self.symbol}")
            except Exception as e:
                self.logger.error(f"Error initializing live data subscriptions: {e}", exc_info=True)

        initial_lookback_minutes = self.config.get('live_initial_lookback_minutes', 60)  # Default to 1 hour
        initial_lookback = timedelta(minutes=initial_lookback_minutes)
        lookback_start = self.current_timestamp_utc - initial_lookback
        lookback_end = self.current_timestamp_utc  # Up to current moment

        try:
            self.logger.info(f"Live mode: Attempting to pre-fill with trades from {lookback_start} to {lookback_end}")
            recent_trades = self.data_manager.get_trades(
                symbol=self.symbol, start_time=lookback_start, end_time=lookback_end
            )
            if recent_trades is not None and not recent_trades.empty:
                if not recent_trades.index.is_monotonic_increasing:
                    recent_trades.sort_index(inplace=True)

                # For live mode, all_trades_data starts with this lookback and grows.
                self.all_trades_data = recent_trades
                self._generate_1s_bars_from_trades()

                if self.all_1s_bars_data is not None and not self.all_1s_bars_data.empty:
                    # Iterate through historical 1s bars for prefill, up to current time
                    for ts_utc in self.all_1s_bars_data.index:
                        if ts_utc >= self.current_timestamp_utc:
                            break  # Don't process "future" bars from historical load

                        self._update_intraday_state_and_session(ts_utc)
                        one_sec_event = self._get_data_for_timestamp(ts_utc)
                        if one_sec_event:
                            self.rolling_1s_data_window.append(one_sec_event)
                            latest_1s_bar = one_sec_event.get('bar')
                            if latest_1s_bar:
                                if self.current_market_session != "CLOSED":
                                    self._update_intraday_high_low(latest_1s_bar)
                                self._update_aggregate_bar(latest_1s_bar, 60, '_current_1m_bar_agg',
                                                           self.rolling_1m_data_window)
                                self._update_aggregate_bar(latest_1s_bar, 300, '_current_5m_bar_agg',
                                                           self.rolling_5m_data_window)
                    self.logger.info(f"Live trading pre-fill complete. Window size: {len(self.rolling_1s_data_window)}")
            else:
                self.logger.info("No recent trades found for pre-filling in live mode.")

        except Exception as e:
            self.logger.warning(f"Could not pre-fill live data window with historical trades: {e}", exc_info=True)

        # Ensure intraday state is for the current real time after prefill, before first live step
        self._update_intraday_state_and_session(datetime.now(ZoneInfo("UTC")).replace(microsecond=0))

    def _get_data_for_timestamp(self, timestamp_utc: datetime) -> Optional[Dict[str, Any]]:
        """Get market data (1s bar, trades, quotes) for a specific 1-second interval ending at timestamp_utc."""
        if self.all_1s_bars_data is None or self.all_1s_bars_data.empty:
            return None

        current_1s_bar_dict: Optional[Dict] = None
        try:
            # Ensure timestamp_utc is timezone-aware for loc
            if timestamp_utc.tzinfo is None:
                timestamp_utc = timestamp_utc.replace(tzinfo=ZoneInfo("UTC"))

            bar_series = self.all_1s_bars_data.loc[timestamp_utc]
            current_1s_bar_dict = bar_series.to_dict()
            current_1s_bar_dict['timestamp'] = timestamp_utc
        except KeyError:
            if self.config.get('ffill_1s_bar_gaps', True) and self.rolling_1s_data_window:
                last_event = self.rolling_1s_data_window[-1]
                if last_event and last_event.get('bar') and not last_event['bar'].get(
                        'ffilled_gap'):  # Don't ffill from an ffilled bar
                    prev_bar = last_event['bar']
                    current_1s_bar_dict = {
                        'open': prev_bar['close'],
                        'high': prev_bar['close'],
                        'low': prev_bar['close'],
                        'close': prev_bar['close'],
                        'volume': 0,
                        'vwap': prev_bar['vwap'],
                        'timestamp': timestamp_utc,
                        'ffilled_gap': True
                    }
                    self.logger.debug(f"Forward-filled 1s bar for gap at {timestamp_utc}")
            if not current_1s_bar_dict:
                self.logger.debug(f"No 1s bar data for {timestamp_utc}, and no ffill.")
                return None

        window_start_exclusive_utc = timestamp_utc - timedelta(seconds=1)

        trades_in_second: List[Dict] = []
        if self.all_trades_data is not None and not self.all_trades_data.empty:
            try:
                trades_df_slice = self.all_trades_data.loc[
                    (self.all_trades_data.index > window_start_exclusive_utc) &
                    (self.all_trades_data.index <= timestamp_utc)
                    ]
                if not trades_df_slice.empty:
                    # Add original timestamp from index to each record
                    records = []
                    for idx, row in trades_df_slice.iterrows():
                        rec = row.to_dict()
                        rec['timestamp'] = idx
                        records.append(rec)
                    trades_in_second = records
            except Exception as e:
                self.logger.debug(f"Error getting trades for ts {timestamp_utc}: {e}")

        quotes_in_second: List[Dict] = []
        if self.all_quotes_data is not None and not self.all_quotes_data.empty:
            try:
                quotes_df_slice = self.all_quotes_data.loc[
                    (self.all_quotes_data.index > window_start_exclusive_utc) &
                    (self.all_quotes_data.index <= timestamp_utc)
                    ]
                if not quotes_df_slice.empty:
                    records = []
                    for idx, row in quotes_df_slice.iterrows():
                        rec = row.to_dict()
                        rec['timestamp'] = idx
                        records.append(rec)
                    quotes_in_second = records
            except Exception as e:
                self.logger.debug(f"Error getting quotes for ts {timestamp_utc}: {e}")

        return {
            'timestamp': timestamp_utc,  # This is the timestamp of the 1s bar (end of interval)
            'bar': current_1s_bar_dict,
            'trades': trades_in_second,  # These have their original sub-second timestamps
            'quotes': quotes_in_second  # These also have their original sub-second timestamps
        }

    def _determine_market_session(self, timestamp_utc: datetime) -> str:
        """Determine the market session based on the timestamp."""
        local_time_dt = timestamp_utc.astimezone(self.exchange_timezone)
        current_time_obj = local_time_dt.time()

        if self.market_hours["PREMARKET_START"] <= current_time_obj <= self.market_hours["PREMARKET_END"]:
            return "PREMARKET"
        elif self.market_hours["REGULAR_START"] <= current_time_obj <= self.market_hours["REGULAR_END"]:
            return "REGULAR"
        elif self.market_hours["POSTMARKET_START"] <= current_time_obj <= self.market_hours["POSTMARKET_END"]:
            return "POSTMARKET"
        return "CLOSED"

    def _update_intraday_state_and_session(self, current_ts_utc: datetime):
        """
        Checks for new day, resets intraday levels if needed, and updates market session.
        This is called BEFORE processing the data for current_ts_utc.
        """
        current_date_local = current_ts_utc.astimezone(self.exchange_timezone).date()

        if self._current_day_for_intraday_tracking != current_date_local:
            self.logger.info(
                f"New trading day detected in exchange time: {current_date_local}. "
                f"Previous day was: {self._current_day_for_intraday_tracking}. Resetting intraday state."
            )
            self.intraday_high = None
            self.intraday_low = None
            self._current_1m_bar_agg = None
            self._current_5m_bar_agg = None
            self._current_day_for_intraday_tracking = current_date_local

        new_session = self._determine_market_session(current_ts_utc)
        if new_session != self.current_market_session:
            self.logger.debug(f"Market session changed from {self.current_market_session} to {new_session} at "
                              f"{current_ts_utc.astimezone(self.exchange_timezone)}")
            self.current_market_session = new_session

    def _update_intraday_high_low(self, current_1s_bar: Dict):
        """Updates intraday high/low based on the current 1s bar's H/L."""
        if not current_1s_bar or current_1s_bar.get('ffilled_gap'):
            return

        bar_high = current_1s_bar.get('high')
        bar_low = current_1s_bar.get('low')

        # Ensure we don't try to compare None with float
        if pd.notna(bar_high):
            if self.intraday_high is None or bar_high > self.intraday_high:
                self.intraday_high = bar_high
        if pd.notna(bar_low):
            if self.intraday_low is None or bar_low < self.intraday_low:
                self.intraday_low = bar_low

    def step(self) -> bool:
        if self.is_done():
            self.logger.info("MarketSimulator: End of data or configured end time reached.")
            return False

        if self.mode == 'backtesting':
            if self._current_bar_idx >= len(self.all_1s_bars_data.index):
                return False  # Should be caught by is_done
            self.current_timestamp_utc = self.all_1s_bars_data.index[self._current_bar_idx]
        elif self.mode == 'live':
            previous_ts_utc = self.current_timestamp_utc

            # Process any buffered data first. This might update self.generated_1s_bars.
            if self._live_trades_buffer or self._live_quotes_buffer:
                self._process_live_data_buffer()

            # For live, the effective current_timestamp_utc is the latest 1s bar generated,
            # or real current time if no new bars from buffer processing.
            if self.all_1s_bars_data is not None and not self.all_1s_bars_data.empty:
                # Prefer the timestamp of the latest available 1s bar if newer
                latest_bar_ts = self.all_1s_bars_data.index[-1]
                system_now_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)

                # If the latest bar is more recent than previous step's current_timestamp_utc
                if previous_ts_utc is None or latest_bar_ts > previous_ts_utc:
                    self.current_timestamp_utc = latest_bar_ts
                else:  # No new bar data, advance to current system time if it's a new second
                    if system_now_utc > (previous_ts_utc or system_now_utc - timedelta(seconds=1)):
                        self.current_timestamp_utc = system_now_utc
                    else:  # No new second, no new data to form a bar.
                        return True  # Still active, effectively waiting.
            else:  # No 1s bars at all, just use current time
                self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)

            if previous_ts_utc and self.current_timestamp_utc <= previous_ts_utc:
                return True

        if not self.current_timestamp_utc:
            self.logger.error("Cannot step: current_timestamp_utc is None.")
            return False

        self._update_intraday_state_and_session(self.current_timestamp_utc)
        one_second_event_data = self._get_data_for_timestamp(self.current_timestamp_utc)

        if not one_second_event_data:
            self.logger.debug(f"No 1s data event for {self.current_timestamp_utc}. Advancing time if backtesting.")
            if self.mode == 'backtesting':
                self._current_bar_idx += 1
            return True

        self.rolling_1s_data_window.append(one_second_event_data)
        latest_1s_bar = one_second_event_data.get('bar')

        if latest_1s_bar:
            if self.current_market_session != "CLOSED":
                self._update_intraday_high_low(latest_1s_bar)

            self._update_aggregate_bar(latest_1s_bar, 60, '_current_1m_bar_agg', self.rolling_1m_data_window)
            self._update_aggregate_bar(latest_1s_bar, 300, '_current_5m_bar_agg', self.rolling_5m_data_window)

        if self.mode == 'backtesting':
            self._current_bar_idx += 1
        return True

    def _update_aggregate_bar(self,
                              latest_1s_bar_with_ts: Dict,
                              bar_duration_seconds: int,
                              current_agg_bar_attr: str,
                              completed_bars_deque: Deque):
        """Update aggregate bars (1m, 5m) based on latest 1s bar."""
        if not latest_1s_bar_with_ts or 'timestamp' not in latest_1s_bar_with_ts \
                or latest_1s_bar_with_ts.get('ffilled_gap'):
            self.logger.debug(
                f"Skipping aggregation for invalid/ffilled 1s bar at {latest_1s_bar_with_ts.get('timestamp')}")
            return

        # Ensure all necessary fields are present in the 1s bar
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        if not all(pd.notna(latest_1s_bar_with_ts.get(field)) for field in required_fields):
            self.logger.debug(
                f"Skipping aggregation due to NaN in required fields for 1s bar at {latest_1s_bar_with_ts.get('timestamp')}: {latest_1s_bar_with_ts}")
            return

        bar_timestamp_utc: datetime = latest_1s_bar_with_ts['timestamp']
        current_agg_bar: Optional[Dict] = getattr(self, current_agg_bar_attr)

        bar_minute = bar_timestamp_utc.minute
        if bar_duration_seconds == 60:
            interval_start_minute = bar_minute
        elif bar_duration_seconds == 300:
            interval_start_minute = (bar_minute // 5) * 5
        else:
            self.logger.error(f"Unsupported bar_duration_seconds: {bar_duration_seconds}")
            return
        current_interval_start_ts_utc = bar_timestamp_utc.replace(minute=interval_start_minute, second=0, microsecond=0)

        if current_agg_bar is None or current_agg_bar['timestamp_start_utc'] != current_interval_start_ts_utc:
            if current_agg_bar is not None:
                completed_bars_deque.append(current_agg_bar.copy())
                self.logger.debug(
                    f"Completed {bar_duration_seconds // 60}m bar (UTC start {current_agg_bar['timestamp_start_utc']}) "
                    f"O:{current_agg_bar['open']} H:{current_agg_bar['high']} L:{current_agg_bar['low']} C:{current_agg_bar['close']} V:{current_agg_bar['volume']}")

            new_agg_bar = {
                'timestamp_start_utc': current_interval_start_ts_utc,
                'open': latest_1s_bar_with_ts['open'],
                'high': latest_1s_bar_with_ts['high'],
                'low': latest_1s_bar_with_ts['low'],
                'close': latest_1s_bar_with_ts['close'],
                'volume': latest_1s_bar_with_ts.get('volume', 0)
            }
            setattr(self, current_agg_bar_attr, new_agg_bar)
            self.logger.debug(
                f"Starting new {bar_duration_seconds // 60}m bar (UTC start {new_agg_bar['timestamp_start_utc']}) with 1s bar data.")
        else:
            current_agg_bar['high'] = max(current_agg_bar['high'], latest_1s_bar_with_ts['high'])
            current_agg_bar['low'] = min(current_agg_bar['low'], latest_1s_bar_with_ts['low'])
            current_agg_bar['close'] = latest_1s_bar_with_ts['close']
            current_agg_bar['volume'] += latest_1s_bar_with_ts.get('volume', 0)

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        """Get the current market state for the feature extractor."""
        if not self.current_timestamp_utc or not self.rolling_1s_data_window:
            # If current_timestamp_utc exists but window is empty (e.g. at the very start before first step processes data)
            # provide a minimal state if possible.
            if self.current_timestamp_utc and self.is_done() and self.mode == 'backtesting' and self._current_bar_idx == 0:
                # Special case: simulator initialized but no steps taken, and it's already "done" (e.g. no data for buffer)
                return {
                    'timestamp_utc': self.current_timestamp_utc,
                    'latest_1s_bar': None,
                    'rolling_1s_data_window': [],
                    'current_market_session': self.current_market_session,
                    'intraday_high': self.intraday_high,
                    'intraday_low': self.intraday_low,
                    'current_1m_bar_forming': None,
                    'current_5m_bar_forming': None,
                    'completed_1m_bars_window': [],
                    'completed_5m_bars_window': [],
                    'historical_1d_bars_full': self.all_1d_bars_data.copy() if self.all_1d_bars_data is not None else None,
                }
            self.logger.debug("Cannot get market state: current_timestamp_utc or current_1s_data_window is empty.")
            return None

        latest_event_in_window = self.rolling_1s_data_window[-1]
        state_timestamp_utc = latest_event_in_window['timestamp']

        if state_timestamp_utc != self.current_timestamp_utc and self.mode == 'backtesting' and not self.is_done():
            self.logger.warning(
                f"Timestamp mismatch in get_current_market_state: sim_utc_ts={self.current_timestamp_utc}, "
                f"window_utc_ts={state_timestamp_utc}. State reflects window_utc_ts."
            )

        market_state = {
            'timestamp_utc': state_timestamp_utc,
            'current_market_session': self.current_market_session,

            'intraday_high': self.intraday_high,
            'intraday_low': self.intraday_low,
            'next_daily_support': None,  # Placeholder for future implementation
            'next_daily_resistance': None,  # Placeholder for future implementation

            'current_1s_bar': latest_event_in_window.get('bar'),
            'current_1m_bar_forming': self._current_1m_bar_agg.copy() if self._current_1m_bar_agg else None,
            'current_5m_bar_forming': self._current_5m_bar_agg.copy() if self._current_5m_bar_agg else None,

            'rolling_1s_data_window': list(self.rolling_1s_data_window),
            'rolling_1m_data_window': list(self.rolling_1m_data_window),
            'rolling_5m_data_window': list(self.rolling_5m_data_window),

            # 'all_1s_bars_data': self.all_1s_bars_data.copy() if self.all_1s_bars_data is not None else None,
            # 'all_1m_bars_data': self.all_1m_bars_data.copy() if self.all_1m_bars_data is not None else None,
            # 'all_5m_bars_data': self.all_5m_bars_data.copy() if self.all_5m_bars_data is not None else None,
            'all_1d_bars_data': self.all_1d_bars_data.copy() if self.all_1d_bars_data is not None else None,
        }
        return market_state

    def is_done(self) -> bool:
        """Check if the simulation is done."""
        if self.mode == 'backtesting':
            if self.all_1s_bars_data is None or self.all_1s_bars_data.empty:
                return True

            is_past_last_index = self._current_bar_idx >= len(self.all_1s_bars_data.index)

            # Check against configured end_time_utc.
            # self.current_timestamp_utc is the timestamp of the bar *about to be processed or just processed*.
            is_past_config_end_time = False
            if self.end_time_utc:
                if self.current_timestamp_utc and self.current_timestamp_utc >= self.end_time_utc:
                    is_past_config_end_time = True
                # Also check if the *next* bar to be processed would be past end_time
                elif not is_past_last_index and self._current_bar_idx < len(self.all_1s_bars_data.index):
                    next_bar_ts = self.all_1s_bars_data.index[self._current_bar_idx]
                    if next_bar_ts > self.end_time_utc:  # Strictly greater for next bar
                        is_past_config_end_time = True

            if is_past_last_index: self.logger.debug("is_done: True (backtesting past last index)")
            if is_past_config_end_time: self.logger.debug(
                f"is_done: True (backtesting past configured end time {self.end_time_utc})")
            return is_past_last_index or is_past_config_end_time

        elif self.mode == 'live':
            if self.end_time_utc and self.current_timestamp_utc and self.current_timestamp_utc >= self.end_time_utc:
                self.logger.debug(f"is_done: True (live mode past configured end time {self.end_time_utc})")
                return True
            return False

        self.logger.warning(f"is_done: Mode '{self.mode}' not recognized for done check, returning True.")
        return True

    def reset(self, options: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Reset the simulator for a new episode."""
        self.logger.info(f"Resetting MarketSimulator for {self.symbol}.")
        options = options or {}
        user_config_start_time = self.config.get('start_time')  # Original start_time from __init__ config

        # Clear dynamic data structures
        self.rolling_1s_data_window.clear()
        self._live_trades_buffer.clear()
        self._live_quotes_buffer.clear()
        self.rolling_1m_data_window.clear()
        self.rolling_5m_data_window.clear()

        # Reset state tracking
        self._current_bar_idx = 0
        self.current_timestamp_utc = None  # Will be set by initialization
        self.current_market_session = "CLOSED"
        self._current_day_for_intraday_tracking = None
        self.intraday_high = None
        self.intraday_low = None
        self._current_1m_bar_agg = None
        self._current_5m_bar_agg = None

        # Restore self.start_time_utc to its original configured value before potentially changing for random_start
        # This ensures that if random_start isn't used or fails, we use the original setting.
        self.start_time_utc = self._parse_datetime(user_config_start_time)

        if self.mode == 'backtesting' and options.get('random_start', self.config.get('random_reset', False)):
            if self.all_1s_bars_data is not None and not self.all_1s_bars_data.empty and len(
                    self.all_1s_bars_data) > 1:
                initial_buffer_seconds = self.config.get('initial_buffer_seconds', 300)

                # Find the earliest possible timestamp from which we can select a random start,
                # ensuring there's enough data for the buffer before it.
                # Example: If data starts at T0, buffer is B, earliest random point for agent is T0+B.
                # The data fed to prefill must start at T0.

                earliest_possible_agent_start_ts = self.all_1s_bars_data.index[0] + timedelta(
                    seconds=initial_buffer_seconds)
                min_agent_start_idx = self.all_1s_bars_data.index.searchsorted(earliest_possible_agent_start_ts,
                                                                               side='left')

                # Latest agent start index: must be able to process at least one bar.
                max_agent_start_idx = len(self.all_1s_bars_data.index) - 1

                if max_agent_start_idx > min_agent_start_idx:
                    # Pick a random index for the agent's first bar
                    random_agent_start_idx = np.random.randint(min_agent_start_idx, max_agent_start_idx + 1)

                    # The self.start_time_utc for _initialize_for_backtesting needs to be set
                    # such that after the buffer, the agent lands on random_agent_start_idx.
                    # So, data loading for prefill must start 'initial_buffer_seconds' before that.
                    # But it cannot start before the actual data begins.
                    potential_prefill_start_ts = self.all_1s_bars_data.index[random_agent_start_idx] - timedelta(
                        seconds=initial_buffer_seconds)
                    self.start_time_utc = max(self.all_1s_bars_data.index[0], potential_prefill_start_ts)

                    self.logger.info(f"Random start: Agent will aim to start at index {random_agent_start_idx} "
                                     f"(ts: {self.all_1s_bars_data.index[random_agent_start_idx]}). "
                                     f"Effective data start_time_utc for prefill set to: {self.start_time_utc}")
                else:
                    self.logger.warning(
                        "Random start: Not enough data range for random start after buffer. Using default start_time.")
                    # self.start_time_utc is already set to original config, so no change needed here.
            else:
                self.logger.warning(
                    "Random start: No generated 1s bars or not enough data points. Using default start_time.")

        if self.mode == 'backtesting':
            self._initialize_for_backtesting()
        elif self.mode == 'live':
            self._initialize_for_live_trading()

        return self.get_current_market_state()

    # --- Live Data Ingestion Methods ---
    def on_live_trade(self, trade_data: Dict):
        """Callback for live trade data. Expected dict with 'timestamp', 'price', 'size'."""
        if not isinstance(trade_data, dict) or not all(k in trade_data for k in ['timestamp', 'price', 'size']):
            self.logger.warning(f"Invalid live trade data format: {trade_data}")
            return

        parsed_ts = self._parse_datetime(trade_data['timestamp'])
        if not parsed_ts:
            self.logger.warning(f"Could not parse timestamp for live trade: {trade_data.get('timestamp')}")
            return

        # Create a new dict to ensure original is not modified if passed by reference
        processed_trade_data = trade_data.copy()
        processed_trade_data['timestamp'] = parsed_ts

        self._live_trades_buffer.append(processed_trade_data)

    def on_live_quote(self, quote_data: Dict):
        """Callback for live quote data. Expected dict with 'timestamp', 'bid_price', 'ask_price', etc."""
        if not isinstance(quote_data, dict) or 'timestamp' not in quote_data:
            self.logger.warning(f"Invalid live quote data format: {quote_data}")
            return

        parsed_ts = self._parse_datetime(quote_data['timestamp'])
        if not parsed_ts:
            self.logger.warning(f"Could not parse timestamp for live quote: {quote_data.get('timestamp')}")
            return

        processed_quote_data = quote_data.copy()
        processed_quote_data['timestamp'] = parsed_ts
        self._live_quotes_buffer.append(processed_quote_data)

    def on_live_bar(self, bar_data: Dict):
        """
        Callback for externally provided live bar data (e.g., 1m, 5m from data provider).
        Expected: {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe'}
        """
        if not isinstance(bar_data, dict) or not all(
                k in bar_data for k in ['timestamp', 'timeframe', 'open', 'high', 'low', 'close']):
            self.logger.warning(f"Invalid live bar data format or missing OHLC: {bar_data}")
            return

        parsed_ts = self._parse_datetime(bar_data['timestamp'])
        if not parsed_ts:
            self.logger.warning(f"Could not parse timestamp for live bar: {bar_data.get('timestamp')}")
            return

        bar_dict_processed = {
            'timestamp_start_utc': parsed_ts,
            'open': bar_data.get('open'),
            'high': bar_data.get('high'),
            'low': bar_data.get('low'),
            'close': bar_data.get('close'),
            'volume': bar_data.get('volume', 0)
        }

        timeframe = bar_data.get('timeframe')
        if timeframe == '1m':
            self.rolling_1m_data_window.append(bar_dict_processed)
            self.logger.debug(f"Received external live 1m bar: {bar_dict_processed['timestamp_start_utc']}")
        elif timeframe == '5m':
            self.rolling_5m_data_window.append(bar_dict_processed)
            self.logger.debug(f"Received external live 5m bar: {bar_dict_processed['timestamp_start_utc']}")
        elif timeframe == '1d':
            if self.all_1d_bars_data is not None:
                # Create a DataFrame for the new bar, ensuring correct column names and index name
                new_bar_df = pd.DataFrame([{
                    'open': bar_dict_processed['open'],
                    'high': bar_dict_processed['high'],
                    'low': bar_dict_processed['low'],
                    'close': bar_dict_processed['close'],
                    'volume': bar_dict_processed['volume']
                }], index=pd.DatetimeIndex([bar_dict_processed['timestamp_start_utc']], name='timestamp_start_utc'))

                # Ensure new_bar_df has same columns as historical_1d_bars_full for clean concat
                # (assuming historical_1d_bars_full also has OHLCV)
                # If historical_1d_bars_full is empty, its columns might not be set.
                if not self.all_1d_bars_data.empty:
                    new_bar_df = new_bar_df.reindex(columns=self.all_1d_bars_data.columns)

                self.all_1d_bars_data = pd.concat([self.all_1d_bars_data, new_bar_df])
                if not self.all_1d_bars_data.index.is_monotonic_increasing:
                    self.all_1d_bars_data.sort_index(inplace=True)
                self.all_1d_bars_data = self.all_1d_bars_data[
                    ~self.all_1d_bars_data.index.duplicated(keep='last')]

                self.logger.debug(
                    f"Appended/Updated external live 1D bar for: {bar_dict_processed['timestamp_start_utc']}")
            else:  # historical_1d_bars_full was None
                self.all_1d_bars_data = pd.DataFrame([{
                    'open': bar_dict_processed['open'],
                    'high': bar_dict_processed['high'],
                    'low': bar_dict_processed['low'],
                    'close': bar_dict_processed['close'],
                    'volume': bar_dict_processed['volume']
                }], index=pd.DatetimeIndex([bar_dict_processed['timestamp_start_utc']], name='timestamp_start_utc'))


        else:
            self.logger.warning(f"Received live bar with unhandled timeframe '{timeframe}': {bar_data}")

    def _process_live_data_buffer(self):
        """Process accumulated live trade and quote data to update internal DataFrames and 1s bars."""
        if not self._live_trades_buffer and not self._live_quotes_buffer:
            return

        self.logger.debug(
            f"Processing live data buffer. Trades: {len(self._live_trades_buffer)}, Quotes: {len(self._live_quotes_buffer)}")
        trades_processed_for_1s_regen = False

        if self._live_trades_buffer:
            # Buffer contains dicts with 'timestamp' already as datetime objects
            new_trades_df = pd.DataFrame(self._live_trades_buffer).set_index('timestamp')
            self._live_trades_buffer.clear()

            if not new_trades_df.empty:
                if not new_trades_df.index.is_monotonic_increasing:  # Should be if live data comes in order
                    new_trades_df.sort_index(inplace=True)

                if self.all_trades_data is None or self.all_trades_data.empty:
                    self.all_trades_data = new_trades_df
                else:
                    # Concatenate and handle duplicates, keeping the last entry for a given timestamp
                    self.all_trades_data = pd.concat([self.all_trades_data, new_trades_df])
                    self.all_trades_data = self.all_trades_data[~self.all_trades_data.index.duplicated(keep='last')]
                    if not self.all_trades_data.index.is_monotonic_increasing:
                        self.all_trades_data.sort_index(inplace=True)  # Sort again after concat

                # Regenerate 1s bars based on the complete, updated all_trades_data
                self._generate_1s_bars_from_trades()
                trades_processed_for_1s_regen = True
                self.logger.debug(f"Updated all_trades_data and regenerated 1s bars from live trades.")

        if self._live_quotes_buffer:
            new_quotes_df = pd.DataFrame(self._live_quotes_buffer).set_index('timestamp')
            self._live_quotes_buffer.clear()

            if not new_quotes_df.empty:
                if not new_quotes_df.index.is_monotonic_increasing:
                    new_quotes_df.sort_index(inplace=True)

                if self.all_quotes_data is None or self.all_quotes_data.empty:
                    self.all_quotes_data = new_quotes_df
                else:
                    self.all_quotes_data = pd.concat([self.all_quotes_data, new_quotes_df])
                    self.all_quotes_data = self.all_quotes_data[~self.all_quotes_data.index.duplicated(keep='last')]
                    if not self.all_quotes_data.index.is_monotonic_increasing:
                        self.all_quotes_data.sort_index(inplace=True)
                self.logger.debug("Updated all_quotes_data from live quotes.")

        # If new trades were processed and 1s bars regenerated, the current_timestamp_utc for live mode
        # might need to jump to the latest available 1s bar if it's newer. This is handled in step().

    def close(self):
        """Close the simulator and clean up resources."""
        self.logger.info(f"MarketSimulatorV2 for {self.symbol} closing.")
        if self.data_manager and hasattr(self.data_manager, 'close'):
            try:
                self.data_manager.close()
            except Exception as e:
                self.logger.error(f"Error closing data_manager: {e}", exc_info=True)