import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
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


class MarketSimulator:
    # Dataframes has cleaned and sorted data with UTC index
    def __init__(self, symbol: str,
                 historical_1s_data: Optional[pd.DataFrame] = None,
                 historical_trades_data: Optional[pd.DataFrame] = None,  # For future use with FeatureExtractor
                 historical_quotes_data: Optional[pd.DataFrame] = None,  # For future use with FeatureExtractor
                 historical_1d_bars: Optional[pd.DataFrame] = None,
                 config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuration dictionary. Relevant keys:
                - 'mode': 'backtesting' or 'live'.
                - 'start_time_str': ISO format UTC string for backtesting start.
                - 'end_time_str': ISO format UTC string for backtesting end.
                - 'max_1s_data_window_size': Max 1s data events (default 3600 for 1 hour).
                - 'initial_buffer_seconds': Seconds of data to pre-fill (default 600 for 10 mins).
                - 'market_hours': Dict for session timings (see example below).
                - 'completed_1m_bars_maxlen': Max length for completed 1m bars deque (default 60).
                - 'completed_5m_bars_maxlen': Max length for completed 5m bars deque (default 24).
            logger: Optional logger.

        """
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.config = config or {}

        self.mode = self.config.get('mode', 'backtesting')

        # --- Timezone and Market Hours Setup ---
        # Timestamps are handled as UTC internally for data processing, converted for session logic
        self.start_time_utc = pd.to_datetime(self.config.get('start_time_str'), utc=True) if self.config.get('start_time_str') else None
        self.end_time_utc = pd.to_datetime(self.config.get('end_time_str'), utc=True) if self.config.get('end_time_str') else None
        self.exchange_timezone = None
        self.market_hours = {
            key: datetime.strptime(value, "%H:%M:%S").time()
            for key, value in DEFAULT_MARKET_HOURS.items() if key.endswith(("_START", "_END"))
        }

        # --- Data Storage (Timestamps are UTC) ---
        self.all_1s_data: Optional[pd.DataFrame] = self._prepare_historical_data(historical_1s_data)
        self.all_trades_data: Optional[pd.DataFrame] = self._prepare_historical_data(historical_trades_data)
        self.all_quotes_data: Optional[pd.DataFrame] = self._prepare_historical_data(historical_quotes_data)
        self.historical_1d_bars_full: Optional[pd.DataFrame] = self._prepare_historical_data(historical_1d_bars)

        # --- Current Time and Index (UTC) ---
        self.current_timestamp_utc: Optional[datetime] = None
        self._current_1s_data_idx: int = 0

        # --- Rolling Data Windows for Features ---
        self.max_1s_data_window_size = self.config.get('max_1s_data_window_size', 60 * 60)
        self.current_1s_data_window: deque = deque(maxlen=self.max_1s_data_window_size)

        # --- Session and Level Tracking State ---
        self.current_market_session: Optional[str] = "CLOSED"
        self.premarket_high: Optional[float] = None
        self.premarket_low: Optional[float] = None
        self.session_open_price: Optional[float] = None
        self.session_high: Optional[float] = None
        self.session_low: Optional[float] = None
        self._last_processed_session_type: Optional[str] = None  # Tracks PREMARKET, REGULAR etc.
        self._current_day_for_session_tracking: Optional[datetime.date] = None

        # --- Bar Aggregation State ---
        self.completed_1m_bars_maxlen = self.config.get('completed_1m_bars_maxlen', 60)
        self.completed_5m_bars_maxlen = self.config.get('completed_5m_bars_maxlen', 24)

        self._current_1m_bar_agg: Optional[Dict] = None
        self._current_5m_bar_agg: Optional[Dict] = None
        self.completed_1m_bars: deque = deque(maxlen=self.completed_1m_bars_maxlen)
        self.completed_5m_bars: deque = deque(maxlen=self.completed_5m_bars_maxlen)

        # --- Live Data Buffers ---
        self._live_trades_buffer: List[Dict] = []
        self._live_quotes_buffer: List[Dict] = []
        self._live_1s_bar_events: List[Dict] = []

        # --- Initialization ---
        if self.mode == 'backtesting':
            if self.all_1s_data is None or self.all_1s_data.empty:
                raise ValueError("Historical 1s data is required for backtesting mode.")
            self._initialize_for_backtesting()
        elif self.mode == 'live':
            self.current_timestamp_utc = datetime.now().astimezone(ZoneInfo("UTC")).replace(microsecond=0)
            self.logger.info(
                f"MarketSimulatorV3 initialized for LIVE trading. Current UTC time: {self.current_timestamp_utc}")

        self.logger.info(f"MarketSimulatorV3 initialized for {self.symbol}. Mode: {self.mode}.")
        if self.exchange_timezone:
            self.logger.info(f"Exchange timezone set to: {self.exchange_timezone}")

    def _prepare_historical_data(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is not None and not df.empty:
            if self.start_time_utc:
                df = df[df.index >= self.start_time_utc]
            if self.end_time_utc:
                df = df[df.index <= self.end_time_utc]
            return df
        return None

    def _initialize_for_backtesting(self):
        initial_buffer_seconds = self.config.get('initial_buffer_seconds', 600)

        if self.all_1s_data is None or self.all_1s_data.empty: return  # Should have been caught earlier

        first_valid_timestamp_utc = self.all_1s_data.index[0]
        # The actual start for the agent will be after the buffer period
        effective_start_timestamp_utc = first_valid_timestamp_utc + timedelta(seconds=initial_buffer_seconds)

        try:
            # Find the index for the data point that starts the simulation for the agent
            self._current_1s_data_idx = self.all_1s_data.index.searchsorted(effective_start_timestamp_utc)
        except Exception as e:  # Handle cases where searchsorted might fail with non-unique index or other issues
            self.logger.warning(f"Could not find exact start index via searchsorted: {e}. Will iterate.")
            idx = 0
            for ts_idx, ts_val in enumerate(self.all_1s_data.index):
                if ts_val >= effective_start_timestamp_utc:
                    idx = ts_idx
                    break
            self._current_1s_data_idx = idx

        if self._current_1s_data_idx >= len(self.all_1s_data):
            self.logger.warning("Not enough data for the configured start time and buffer. Starting at the end.")
            self._current_1s_data_idx = len(self.all_1s_data) - 1  # Start with the last point

        if self._current_1s_data_idx == 0 and initial_buffer_seconds > 0:
            self.logger.warning(
                f"Not enough data for full initial_buffer_seconds ({initial_buffer_seconds}). Starting with available data.")

        # Pre-fill buffers (1s data window, completed 1m/5m bars)
        start_fill_idx = max(0, self._current_1s_data_idx - self.max_1s_data_window_size)
        self.logger.info(f"Pre-filling data window from index {start_fill_idx} to {self._current_1s_data_idx - 1}")

        # Temporarily set current_timestamp_utc to iterate for pre-filling
        _temp_current_ts_utc = self.current_timestamp_utc
        for i in range(start_fill_idx, self._current_1s_data_idx):
            ts_utc = self.all_1s_data.index[i]
            self.current_timestamp_utc = ts_utc  # Simulate being at this time for prefill logic

            # Basic session determination for prefill, might not be perfect if prefill crosses day boundaries without full logic
            current_date_local = ts_utc.astimezone(self.exchange_timezone).date()
            if self._current_day_for_session_tracking != current_date_local:
                self._reset_daily_and_session_levels(current_date_local, ts_utc)

            one_sec_event = self._get_data_for_timestamp(ts_utc)
            if one_sec_event:
                self.current_1s_data_window.append(one_sec_event)
                if one_sec_event.get('bar'):
                    # Update aggregators during prefill
                    self._update_aggregate_bar(one_sec_event['bar'], 60, '_current_1m_bar_agg', self.completed_1m_bars)
                    self._update_aggregate_bar(one_sec_event['bar'], 300, '_current_5m_bar_agg', self.completed_5m_bars)

        # Restore or set the actual starting current_timestamp_utc for the agent
        if self._current_1s_data_idx < len(self.all_1s_data):
            self.current_timestamp_utc = self.all_1s_data.index[self._current_1s_data_idx]
        else:  # Reached end of data during initialization
            self.current_timestamp_utc = self.all_1s_data.index[-1] if not self.all_1s_data.empty else None
            self.logger.warning("Reached end of data during initialization.")

        # Final session update for the actual start time
        if self.current_timestamp_utc:
            current_date_local = self.current_timestamp_utc.astimezone(self.exchange_timezone).date()
            self._reset_daily_and_session_levels(current_date_local,
                                                 self.current_timestamp_utc)  # Ensures session state is correct at start

        self.logger.info(
            f"Backtesting initialized. Buffer filled. Starting at index {self._current_1s_data_idx}, "
            f"UTC timestamp: {self.current_timestamp_utc}, "
            f"Exchange time: {self.current_timestamp_utc.astimezone(self.exchange_timezone) if self.current_timestamp_utc and self.exchange_timezone else 'N/A'}"
        )
        self.current_timestamp_utc = _temp_current_ts_utc  # Restore if it was changed

    def _get_data_for_timestamp(self, timestamp_utc: datetime) -> Optional[Dict[str, Any]]:
        if self.all_1s_data is None: return None

        current_1s_bar = None
        try:
            bar_data_series = self.all_1s_data.loc[timestamp_utc]
            # bar_data_series will have OHLCV. Add the timestamp to the bar dict itself for _update_aggregate_bar
            current_1s_bar = bar_data_series.to_dict()
            current_1s_bar['timestamp'] = timestamp_utc  # Crucial for aggregator
        except KeyError:
            # Handle gaps - use previous bar if available, flag it
            if self.current_1s_data_window:
                last_event = self.current_1s_data_window[-1]
                if last_event and last_event.get('bar'):
                    current_1s_bar = last_event['bar'].copy()  # Use a copy
                    current_1s_bar['ffill_original_timestamp_utc'] = timestamp_utc  # Mark as ffilled
                    current_1s_bar['timestamp'] = timestamp_utc  # Update its timestamp to current tick
                    current_1s_bar['volume'] = 0  # Typically, ffilled bars have no new volume
                    self.logger.debug(f"No 1s bar for {timestamp_utc}, forward-filling from previous.")
            if not current_1s_bar:
                self.logger.debug(f"No 1s bar data for {timestamp_utc} and no previous bar to ffill.")
                return None  # Cannot proceed without any bar data

        # Trades and quotes for this 1-second interval (ending at timestamp_utc)
        window_start_utc = timestamp_utc - timedelta(seconds=1)  # Data *within* the second leading up to timestamp_utc

        trades_in_second = []
        if self.all_trades_data is not None:
            trades_df = self.all_trades_data[(self.all_trades_data.index > window_start_utc) &
                                             (self.all_trades_data.index <= timestamp_utc)]
            if not trades_df.empty: trades_in_second = [row.to_dict() for _, row in trades_df.iterrows()]

        quotes_in_second = []
        if self.all_quotes_data is not None:
            quotes_df = self.all_quotes_data[(self.all_quotes_data.index > window_start_utc) &
                                             (self.all_quotes_data.index <= timestamp_utc)]
            if not quotes_df.empty: quotes_in_second = [row.to_dict() for _, row in quotes_df.iterrows()]

        return {
            'timestamp': timestamp_utc,  # This is the timestamp of the 1s event itself
            'bar': current_1s_bar,  # OHLCV for the second ending at 'timestamp_utc'
            'trades': trades_in_second,
            'quotes': quotes_in_second
        }

    def _determine_market_session(self, timestamp_utc: datetime) -> str:
        if not self.exchange_timezone: return "UNKNOWN"  # Should not happen if init is correct

        # Convert UTC timestamp to exchange local time
        local_time_dt = timestamp_utc.astimezone(self.exchange_timezone)
        current_time_obj = local_time_dt.time()

        if self.market_hours["PREMARKET_START"] <= current_time_obj <= self.market_hours["PREMARKET_END"]:
            return "PREMARKET"
        elif self.market_hours["REGULAR_START"] <= current_time_obj <= self.market_hours["REGULAR_END"]:
            return "REGULAR"
        elif self.market_hours["POSTMARKET_START"] <= current_time_obj <= self.market_hours[
            "POSTMARKET_END"]:
            return "POSTMARKET"
        return "CLOSED"

    def _reset_daily_and_session_levels(self, current_date_local: datetime.date,
                                        timestamp_utc_for_session_calc: datetime):
        """Resets levels if day changes, and determines initial session state."""
        if self._current_day_for_session_tracking != current_date_local:
            self.logger.info(
                f"New trading day detected in exchange time: {current_date_local}. Resetting daily levels.")
            self.premarket_high = None
            self.premarket_low = None
            self.session_open_price = None
            self.session_high = None
            self.session_low = None
            # Do NOT clear completed_1m/5m_bars here, as they are rolling windows.
            # Clear forming bars as they are day-specific if not carried over.
            self._current_1m_bar_agg = None
            self._current_5m_bar_agg = None
            self._current_day_for_session_tracking = current_date_local
            self._last_processed_session_type = None  # Force session re-evaluation

        # Determine current session based on the new day/time
        self.current_market_session = self._determine_market_session(timestamp_utc_for_session_calc)
        self.logger.debug(
            f"Initial session for {current_date_local} at {timestamp_utc_for_session_calc} (UTC) is {self.current_market_session}")

        # Reset specific session levels if entering that session type for the first time today
        if self.current_market_session == "PREMARKET" and self._last_processed_session_type != "PREMARKET":
            self.premarket_high = None
            self.premarket_low = None
        elif self.current_market_session == "REGULAR" and self._last_processed_session_type != "REGULAR":
            self.session_open_price = None  # Will be set by first bar of regular session
            self.session_high = None
            self.session_low = None

        self._last_processed_session_type = self.current_market_session

    def step(self) -> bool:
        if self.is_done():
            self.logger.info("MarketSimulatorV3: End of data reached.")
            return False

        if self.mode == 'backtesting':
            self._current_1s_data_idx += 1
            if self._current_1s_data_idx >= len(self.all_1s_data.index):
                return False
            self.current_timestamp_utc = self.all_1s_data.index[self._current_1s_data_idx]
        elif self.mode == 'live':
            # In live mode, current_timestamp_utc would be updated by an external clock or event trigger
            self.current_timestamp_utc = datetime.now().astimezone(ZoneInfo("UTC")).replace(microsecond=0)
            # Process live buffers here (self._live_trades_buffer, etc.) and update main data stores if needed.
            # For this example, we assume data is already in all_1s_data for live mode too, or handled by add_live_X methods.

        # --- Session Management ---
        if self.current_timestamp_utc is None: return False  # Should not happen after init

        current_date_local = self.current_timestamp_utc.astimezone(self.exchange_timezone).date()
        if self._current_day_for_session_tracking != current_date_local:
            self._reset_daily_and_session_levels(current_date_local, self.current_timestamp_utc)

        new_session_type = self._determine_market_session(self.current_timestamp_utc)
        if new_session_type != self.current_market_session:  # True session type change
            self.logger.info(
                f"Market session changed from {self.current_market_session} to {new_session_type} at {self.current_timestamp_utc.astimezone(self.exchange_timezone)}")
            self.current_market_session = new_session_type
            if new_session_type == "PREMARKET":  # Entering Premarket
                self.premarket_high = None
                self.premarket_low = None
            elif new_session_type == "REGULAR":  # Entering Regular
                self.session_open_price = None  # Will be set by first bar
                self.session_high = None
                self.session_low = None
            # Update _last_processed_session_type to reflect the new state we are in
            self._last_processed_session_type = new_session_type

        # --- Get current 1s data event ---
        one_second_event_data = self._get_data_for_timestamp(self.current_timestamp_utc)
        if not one_second_event_data:
            self.logger.warning(f"No 1s data event for {self.current_timestamp_utc}. Skipping step.")
            return True  # Allow time to advance, but no data to process this tick. Or False if critical.

        self.current_1s_data_window.append(one_second_event_data)
        latest_1s_bar = one_second_event_data.get('bar')

        # --- Update Premarket/Session High/Low ---
        if latest_1s_bar:
            # Ensure bar timestamp is UTC for consistency if used later (already should be)
            # latest_1s_bar['timestamp'] = self.current_timestamp_utc

            bar_high = latest_1s_bar.get('high')
            bar_low = latest_1s_bar.get('low')

            if self.current_market_session == "PREMARKET":
                if bar_high is not None: self.premarket_high = max(self.premarket_high,
                                                                   bar_high) if self.premarket_high is not None else bar_high
                if bar_low is not None: self.premarket_low = min(self.premarket_low,
                                                                 bar_low) if self.premarket_low is not None else bar_low
            elif self.current_market_session == "REGULAR":
                if self.session_open_price is None:  # First bar of regular session
                    self.session_open_price = latest_1s_bar.get('open')  # Or close, based on definition
                    self.logger.info(
                        f"Regular session open price set: {self.session_open_price} at {self.current_timestamp_utc.astimezone(self.exchange_timezone)}")
                if bar_high is not None: self.session_high = max(self.session_high,
                                                                 bar_high) if self.session_high is not None else bar_high
                if bar_low is not None: self.session_low = min(self.session_low,
                                                               bar_low) if self.session_low is not None else bar_low

            # --- Bar Aggregation ---
            self._update_aggregate_bar(latest_1s_bar, 60, '_current_1m_bar_agg', self.completed_1m_bars)
            self._update_aggregate_bar(latest_1s_bar, 300, '_current_5m_bar_agg', self.completed_5m_bars)

        return True

    def _update_aggregate_bar(self, latest_1s_bar_with_ts: Dict, bar_duration_seconds: int,
                              current_agg_bar_attr: str, completed_bars_deque: deque):
        # latest_1s_bar_with_ts is assumed to have a 'timestamp' key (UTC datetime of the 1s bar event)
        # and OHLCV keys.
        if not latest_1s_bar_with_ts or 'timestamp' not in latest_1s_bar_with_ts:
            self.logger.error("Cannot update aggregate bar: latest_1s_bar_with_ts is invalid or missing timestamp.")
            return

        bar_timestamp_utc = latest_1s_bar_with_ts['timestamp']
        current_agg_bar = getattr(self, current_agg_bar_attr)

        # Determine the start timestamp of the interval this 1s bar belongs to (in UTC)
        if bar_duration_seconds == 60:  # 1-minute bar
            current_interval_start_ts_utc = bar_timestamp_utc.replace(second=0, microsecond=0)
        elif bar_duration_seconds == 300:  # 5-minute bar
            current_interval_start_ts_utc = bar_timestamp_utc.replace(second=0, microsecond=0)
            current_interval_start_ts_utc = current_interval_start_ts_utc.replace(
                minute=(bar_timestamp_utc.minute // 5) * 5)
        else:
            self.logger.error(f"Unsupported bar_duration_seconds: {bar_duration_seconds}")
            return

        if current_agg_bar is None or current_agg_bar['timestamp_start_utc'] != current_interval_start_ts_utc:
            if current_agg_bar is not None:
                completed_bars_deque.append(current_agg_bar.copy())
                self.logger.debug(
                    f"Completed {bar_duration_seconds // 60}m bar (UTC start {current_agg_bar['timestamp_start_utc']}): {current_agg_bar}")

            new_agg_bar = {
                'timestamp_start_utc': current_interval_start_ts_utc,  # Start of bar interval, UTC
                'open': latest_1s_bar_with_ts['open'],
                'high': latest_1s_bar_with_ts['high'],
                'low': latest_1s_bar_with_ts['low'],
                'close': latest_1s_bar_with_ts['close'],
                'volume': latest_1s_bar_with_ts.get('volume', 0)
                # 'timestamp_end_utc' could be added: current_interval_start_ts_utc + timedelta(seconds=bar_duration_seconds -1)
            }
            setattr(self, current_agg_bar_attr, new_agg_bar)
            self.logger.debug(
                f"Starting new {bar_duration_seconds // 60}m bar (UTC start {new_agg_bar['timestamp_start_utc']}): {new_agg_bar}")
        else:
            current_agg_bar['high'] = max(current_agg_bar['high'], latest_1s_bar_with_ts['high'])
            current_agg_bar['low'] = min(current_agg_bar['low'], latest_1s_bar_with_ts['low'])
            current_agg_bar['close'] = latest_1s_bar_with_ts['close']
            current_agg_bar['volume'] += latest_1s_bar_with_ts.get('volume', 0)

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        if not self.current_timestamp_utc or not self.current_1s_data_window:
            self.logger.debug("Cannot get market state: current_timestamp_utc or current_1s_data_window is empty.")
            return None

        latest_event_in_window = self.current_1s_data_window[-1]
        state_timestamp_utc = latest_event_in_window['timestamp']

        # Ensure current_timestamp_utc matches the latest event for consistency
        if state_timestamp_utc != self.current_timestamp_utc:
            if not self.is_done() and self.mode == 'backtesting':  # Only log if not actually done
                self.logger.debug(
                    f"Timestamp mismatch in get_current_market_state: sim_utc_ts={self.current_timestamp_utc}, "
                    f"window_utc_ts={state_timestamp_utc}. Using window_utc_ts for state."
                )

        # For completed bars, we need to ensure their timestamps are also what FeatureExtractor expects.
        # The FeatureExtractor expects dicts of OHLCV. Add 'timestamp' if it's not 'timestamp_start_utc'.
        # Let's assume FE can handle 'timestamp_start_utc' or we adapt FeatureExtractor slightly if needed.
        # For now, we pass the dicts as they are stored.

        market_state = {
            'timestamp': state_timestamp_utc,  # UTC timestamp of the current tick
            'latest_1s_bar': latest_event_in_window.get('bar'),
            'rolling_1s_data_window': list(self.current_1s_data_window),

            'current_market_session': self.current_market_session,
            'premarket_high': self.premarket_high,
            'premarket_low': self.premarket_low,
            'session_open_price': self.session_open_price,
            'session_high': self.session_high,
            'session_low': self.session_low,

            'current_1m_bar': self._current_1m_bar_agg.copy() if self._current_1m_bar_agg else None,
            'current_5m_bar': self._current_5m_bar_agg.copy() if self._current_5m_bar_agg else None,
            'completed_1m_bars_window': list(self.completed_1m_bars),
            'completed_5m_bars_window': list(self.completed_5m_bars),

            # Pass the full historical daily DataFrame. FeatureExtractor will handle it.
            'historical_1d_bars_full': self.historical_1d_bars_full.copy() if self.historical_1d_bars_full is not None else None,
        }
        return market_state

    def is_done(self) -> bool:
        if self.mode == 'backtesting':
            if self.all_1s_data is None or self.all_1s_data.empty: return True
            # Done if current_1s_data_idx is at or beyond the last available index
            is_past_last_index = self._current_1s_data_idx >= len(self.all_1s_data.index)
            # Also consider if current_timestamp_utc has passed the configured end_time_utc
            is_past_end_time = self.end_time_utc and self.current_timestamp_utc and self.current_timestamp_utc > self.end_time_utc
            return is_past_last_index or is_past_end_time
        elif self.mode == 'live':
            if self.end_time_utc and self.current_timestamp_utc and self.current_timestamp_utc >= self.end_time_utc:
                return True  # Live simulation with a defined end time
            return False  # Live mode typically doesn't "end" unless explicitly stopped
        return True

    def reset(self):
        """Resets the simulator to its initial state for a new episode."""
        self.logger.info(f"Resetting MarketSimulatorV3 for {self.symbol}.")
        self.current_1s_data_window.clear()
        self._live_trades_buffer.clear()
        self._live_quotes_buffer.clear()
        self._live_1s_bar_events.clear()

        self._current_1s_data_idx = 0
        self.current_timestamp_utc = None  # Will be set by _initialize_for_backtesting or live start

        # Reset session and level tracking state
        self.current_market_session = "CLOSED"
        self.premarket_high = None
        self.premarket_low = None
        self.session_open_price = None
        self.session_high = None
        self.session_low = None
        self._last_processed_session_type = None
        self._current_day_for_session_tracking = None

        # Reset bar aggregation state
        self._current_1m_bar_agg = None
        self._current_5m_bar_agg = None
        self.completed_1m_bars.clear()
        self.completed_5m_bars.clear()

        if self.mode == 'backtesting':
            self._initialize_for_backtesting()
        elif self.mode == 'live':
            self.current_timestamp_utc = datetime.now().astimezone(ZoneInfo("UTC")).replace(microsecond=0)
            # Potentially re-fill initial buffers for live mode if necessary from recent history
            self.logger.info(
                f"MarketSimulatorV3 reset for LIVE trading. Current UTC time: {self.current_timestamp_utc}")

        return self.get_current_market_state()

    # --- Live Data Ingestion Methods (simplified, from V2) ---
    def add_live_trade(self, trade_event: Dict[str, Any]):
        # Ensure timestamp is UTC datetime
        if isinstance(trade_event.get('timestamp'), str):
            trade_event['timestamp'] = pd.to_datetime(trade_event['timestamp'], utc=True)
        elif isinstance(trade_event.get('timestamp'), datetime) and trade_event['timestamp'].tzinfo is None:
            trade_event['timestamp'] = trade_event['timestamp'].replace(tzinfo=ZoneInfo("UTC"))
        elif isinstance(trade_event.get('timestamp'), datetime):
            trade_event['timestamp'] = trade_event['timestamp'].astimezone(ZoneInfo("UTC"))

        if self.mode == 'live' and self.all_trades_data is not None:
            new_trade_df = pd.DataFrame([trade_event]).set_index(pd.DatetimeIndex([trade_event['timestamp']]))
            self.all_trades_data = pd.concat([self.all_trades_data, new_trade_df]).sort_index()
        self.logger.debug(f"Live trade received: {trade_event}")

    # Similar UTC timestamp handling for add_live_quote and add_live_1s_data_event needed

    def close(self):
        self.logger.info(f"MarketSimulatorV3 for {self.symbol} closing.")
