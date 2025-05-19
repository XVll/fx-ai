# market_simulator.py
import logging
from collections import deque
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo # Use zoneinfo for timezone handling
import numpy as np
import pandas as pd
import bisect
from typing import Any, Dict, List, Optional, Tuple,  Union

# Assuming these are external and exist
from config.config import MarketConfig, FeatureConfig # User specified this path
from data.data_manager import DataManager # User specified this path

DEFAULT_MARKET_HOURS = { # Using a more descriptive name for clarity
    "PREMARKET_START_ET": time(4, 0, 0),
    "REGULAR_START_ET": time(9, 30, 0),
    "REGULAR_END_ET": time(15, 59, 59), # Market typically closes at 16:00:00
    "POSTMARKET_END_ET": time(19, 59, 59), # Session ends at 20:00:00
    "SESSION_START_ET": time(4, 0, 0), # Full extended session start
    "SESSION_END_ET": time(19, 59, 59),   # Full extended session end
    "TIMEZONE_ET": "America/New_York"
}
# Define the close time for "previous day close" (e.g., 8 PM ET)
PREVIOUS_DAY_CLOSING_TIME_ET = time(19, 59, 59) # Using 7:59:59 PM ET as official end of post-market

class MarketSimulator:
    """
    Market simulator with a complete, high-fidelity 1-second timeline.

    Handles sparse data by filling gaps, ensures correct BBO and current price
    at each second, and manages lookback data robustly, even across session
    boundaries for early pre-market starts.
    """
    EPSILON = 1e-9

    def __init__(self,
                 symbol: str,
                 data_manager: DataManager,
                 market_config: MarketConfig,
                 feature_config: FeatureConfig,
                 mode: str = "backtesting",
                 start_time: Optional[Union[str, datetime]] = None, # For backtesting: actual session start for agent
                 end_time: Optional[Union[str, datetime]] = None,   # For backtesting: actual session end for agent
                 logger: Optional[logging.Logger] = None):

        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.data_manager = data_manager
        self.market_config = market_config
        self.feature_config = feature_config
        self.mode = mode

        self._setup_market_hours_and_timezone()

        # These are the actual start/end times for the *agent's trading session*
        self.session_start_utc = self._parse_datetime_to_utc(start_time)
        self.session_end_utc = self._parse_datetime_to_utc(end_time)

        self._precomputed_states: Dict[datetime, Dict[str, Any]] = {}
        self._agent_timeline_utc: List[datetime] = [] # Timestamps agent will step through

        # Rolling window sizes (number of items in deques)
        self.rolling_1s_event_window_size = max(1, self.feature_config.hf_seq_len)
        self.completed_1m_bars_window_size = max(1, self.feature_config.mf_seq_len)
        self.completed_5m_bars_window_size = max(1, self.feature_config.lf_seq_len)

        # Determine data loading range for priming lookbacks
        self._priming_lookback_seconds = self._calculate_priming_lookback_seconds()
        self.logger.info(f"Priming lookback: {self._priming_lookback_seconds} seconds.")

        self.current_timestamp_utc: Optional[datetime] = None
        self._current_agent_time_idx: int = -1

        # Raw data storage (will be populated by _load_data_for_simulation)
        self.raw_trades_df: pd.DataFrame = pd.DataFrame()
        self.raw_quotes_df: pd.DataFrame = pd.DataFrame()
        self.raw_1s_bars_provider_df: pd.DataFrame = pd.DataFrame() # Bars from provider
        self.historical_1d_bars_df: pd.DataFrame = pd.DataFrame() # Long-term daily bars

        self._initialize_simulator()

    def _setup_market_hours_and_timezone(self):
        cfg_hours = self.market_config.get('market_hours', {})
        self.exchange_timezone_str = cfg_hours.get('TIMEZONE', DEFAULT_MARKET_HOURS['TIMEZONE_ET'])
        self.exchange_tz = ZoneInfo(self.exchange_timezone_str)

        self.session_start_time_local = cfg_hours.get('SESSION_START_ET', DEFAULT_MARKET_HOURS['SESSION_START_ET'])
        self.session_end_time_local = cfg_hours.get('SESSION_END_ET', DEFAULT_MARKET_HOURS['SESSION_END_ET'])
        self.prev_day_close_time_local = cfg_hours.get('PREVIOUS_DAY_CLOSING_TIME_ET', PREVIOUS_DAY_CLOSING_TIME_ET)

        self.regular_market_start_local = cfg_hours.get('REGULAR_START_ET', DEFAULT_MARKET_HOURS['REGULAR_START_ET'])
        self.regular_market_end_local = cfg_hours.get('REGULAR_END_ET', DEFAULT_MARKET_HOURS['REGULAR_END_ET'])


    def _parse_datetime_to_utc(self, dt_input: Optional[Union[str, datetime]]) -> Optional[datetime]:
        if dt_input is None:
            return None
        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None: # Naive, assume it's in exchange timezone
                return self.exchange_tz.localize(dt_input).astimezone(ZoneInfo("UTC"))
            return dt_input.astimezone(ZoneInfo("UTC"))
        try: # String
            parsed_dt = pd.Timestamp(dt_input)
            if parsed_dt.tzinfo is None:
                return self.exchange_tz.localize(parsed_dt.to_pydatetime()).astimezone(ZoneInfo("UTC"))
            return parsed_dt.tz_convert('UTC').to_pydatetime()
        except Exception as e:
            self.logger.error(f"Failed to parse datetime '{dt_input}': {e}")
            return None

    def _calculate_priming_lookback_seconds(self) -> int:
        # Max seconds needed for the data stored in deques that FeatureExtractor uses
        s1_data_duration = self.rolling_1s_event_window_size # N seconds
        m1_bars_duration = self.completed_1m_bars_window_size * 60 # N_1m_bars * 60s
        m5_bars_duration = self.completed_5m_bars_window_size * 5 * 60 # N_5m_bars * 300s
        return max(s1_data_duration, m1_bars_duration, m5_bars_duration, 3600) # Ensure at least 1 hour

    def _initialize_simulator(self):
        self.logger.info(f"Initializing MarketSimulator for {self.symbol} in {self.mode} mode.")
        self._load_data_for_simulation()

        if self.mode == 'backtesting':
            if not self.session_start_utc or not self.session_end_utc:
                self.logger.error("Backtesting mode requires session_start_utc and session_end_utc.")
                return
            self._precompute_timeline_states()
            self.reset() # Set initial agent time
        elif self.mode == 'live':
            # In live mode, precompute for historical part if session_start_utc is in past
            # And then prepare for real-time updates (not fully implemented here, focuses on backtest rewrite)
            self.logger.warning("Live mode initialization is simplified in this rewrite, focusing on backtesting path.")
            if self.session_start_utc and self.session_start_utc < datetime.now(ZoneInfo("UTC")):
                 self._precompute_timeline_states() # Precompute history
            # For true live, would need to connect to data stream
            self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
            if not self._agent_timeline_utc or self.current_timestamp_utc > self._agent_timeline_utc[-1]:
                 self._agent_timeline_utc.append(self.current_timestamp_utc) # Add current time if not covered
                 self._agent_timeline_utc.sort()
            # Ensure a state for current live time, may need to calculate it on the fly if not precomputed
            if self.current_timestamp_utc not in self._precomputed_states:
                # This part is complex for live if not precomputed, placeholder for default state:
                self._precomputed_states[self.current_timestamp_utc] = self._create_empty_state(self.current_timestamp_utc)

            idx = bisect.bisect_left(self._agent_timeline_utc, self.current_timestamp_utc)
            self._current_agent_time_idx = min(idx, len(self._agent_timeline_utc) -1) if self._agent_timeline_utc else -1

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _load_data_for_simulation(self):
        if not self.session_start_utc:
            self.logger.error("Cannot load data: session_start_utc is not defined.")
            return

        # 1. Determine data loading range (includes priming period)
        # Priming data starts well before the agent's session_start_utc
        load_range_start_utc = self.session_start_utc - timedelta(seconds=self._priming_lookback_seconds)
        # Add a buffer (e.g., 1-2 days) to ensure we get previous trading day's data for lookbacks at 4 AM
        load_range_start_utc -= timedelta(days=self.market_config.get('data_load_day_buffer', 2))

        # Data loading ends at the agent's session_end_utc (or later if live mode needs continuous feed)
        load_range_end_utc = self.session_end_utc
        if self.mode == 'live' and (not load_range_end_utc or load_range_end_utc < datetime.now(ZoneInfo("UTC"))):
            load_range_end_utc = datetime.now(ZoneInfo("UTC")) + timedelta(minutes=5) # Load a bit into future for live

        self.logger.info(f"Data loading range for symbol {self.symbol}: {load_range_start_utc} to {load_range_end_utc}")

        data_types = ["trades", "quotes"]
        if self.market_config.get("use_provider_1s_bars", False):
            data_types.append("bars_1s")

        # 2. Load trades, quotes, (optional) 1s bars for the determined range
        loaded_data = self.data_manager.load_data(
            symbols=[self.symbol],
            start_time=load_range_start_utc,
            end_time=load_range_end_utc,
            data_types=data_types
        )
        symbol_data = loaded_data.get(self.symbol, {})
        self.raw_trades_df = self._prepare_dataframe(symbol_data.get("trades"), ['price', 'size'])
        self.raw_quotes_df = self._prepare_dataframe(symbol_data.get("quotes"), ['price', 'size', 'side']) #
        if "bars_1s" in data_types:
            self.raw_1s_bars_provider_df = self._prepare_dataframe(symbol_data.get("bars_1s"), ['open', 'high', 'low', 'close', 'volume'])

        # 3. Load long-term historical daily bars for features (e.g., 2 years before load_range_start_utc)
        daily_bars_load_end_utc = load_range_start_utc # Load daily bars up to the start of our granular data
        daily_bars_load_start_utc = daily_bars_load_end_utc - timedelta(days=self.market_config.get('historical_daily_bars_lookback_days', 730)) # Approx 2 years
        try:
            self.historical_1d_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1d",
                start_time=daily_bars_load_start_utc,
                end_time=daily_bars_load_end_utc # Get data up to the day *before* our fine-grained data starts for priming.
            )
            self.historical_1d_bars_df = self._prepare_dataframe(self.historical_1d_bars_df, ['open', 'high', 'low', 'close', 'volume', 'close_8pm'])
            if 'close_8pm' not in self.historical_1d_bars_df.columns and 'close' in self.historical_1d_bars_df.columns:
                self.historical_1d_bars_df['close_8pm'] = self.historical_1d_bars_df['close'] # Placeholder if specific 8pm close not available
            self.logger.info(f"Loaded {len(self.historical_1d_bars_df)} historical 1D bars up to {daily_bars_load_end_utc}.")
        except Exception as e:
            self.logger.error(f"Failed to load historical 1D bars: {e}")
            self.historical_1d_bars_df = self._prepare_dataframe(None, ['open', 'high', 'low', 'close', 'volume', 'close_8pm'])

    def _prepare_dataframe(self, df: Optional[pd.DataFrame], required_cols: List[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=required_cols).set_index(pd.DatetimeIndex([], tz='UTC'))
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("Input DataFrame does not have a DatetimeIndex. Attempting conversion or returning empty.")
            # Potentially try df = df.set_index('timestamp_column_name') if applicable
            return pd.DataFrame(columns=required_cols).set_index(pd.DatetimeIndex([], tz='UTC'))

        if df.index.tzinfo is None:
            df = df.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        elif df.index.tzinfo != ZoneInfo("UTC"):
            df = df.tz_convert('UTC')

        # Ensure required columns exist
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan # Add missing columns with NaN
        return df.sort_index()

    def _get_previous_trading_day_close(self, current_session_date_utc: datetime.date) -> Optional[float]:
        if self.historical_1d_bars_df.empty:
            return None
        # Find the trading day strictly before current_session_date_utc
        # The dates in historical_1d_bars_df.index are typically the start of the day in UTC.
        # We need to find the bar for the trading session that concluded before current_session_date_utc.
        relevant_bars = self.historical_1d_bars_df[self.historical_1d_bars_df.index.date < current_session_date_utc]
        if relevant_bars.empty:
            return None
        # The last row of relevant_bars is the previous trading day.
        # Use 'close_8pm' if available (represents post-market close), else 'close'.
        close_col = 'close_8pm' if 'close_8pm' in relevant_bars.columns else 'close'
        prev_close = relevant_bars.iloc[-1][close_col]
        return float(prev_close) if pd.notna(prev_close) else None


    def _precompute_timeline_states(self):
        self.logger.info(f"Starting precomputation of timeline states for {self.symbol}.")
        if not self.session_start_utc or not self.session_end_utc:
             self.logger.error("Cannot precompute: session_start_utc or session_end_utc not set.")
             return

        # Define the full timeline for which states will be generated and stored
        # This is from the agent's session_start_utc to session_end_utc at 1-second resolution.
        current_ts = self.session_start_utc
        agent_session_timeline = []
        while current_ts <= self.session_end_utc:
            agent_session_timeline.append(current_ts)
            current_ts += timedelta(seconds=1)
        self._agent_timeline_utc = agent_session_timeline
        if not self._agent_timeline_utc:
            self.logger.error("Agent session timeline is empty. Check session start/end times.")
            return

        # Define the iteration range for building up history (includes priming period)
        iteration_start_utc = self._agent_timeline_utc[0] - timedelta(seconds=self._priming_lookback_seconds)
        iteration_end_utc = self._agent_timeline_utc[-1]

        self.logger.info(f"Iteration range for state building: {iteration_start_utc} to {iteration_end_utc}")
        self.logger.info(f"Agent visible timeline: {self._agent_timeline_utc[0]} to {self._agent_timeline_utc[-1]}")


        # Initialize LOCF (Last Observation Carried Forward) variables
        locf_current_price: Optional[float] = None
        locf_best_bid_price: Optional[float] = None
        locf_best_ask_price: Optional[float] = None
        locf_best_bid_size: int = 0
        locf_best_ask_size: int = 0

        # Initialize deques for rolling windows
        rolling_1s_event_data = deque(maxlen=self.rolling_1s_event_window_size)
        completed_1m_bars = deque(maxlen=self.completed_1m_bars_window_size)
        completed_5m_bars = deque(maxlen=self.completed_5m_bars_window_size)

        # Variables for aggregating 1m/5m bars
        current_1m_bar_forming: Optional[Dict[str, Any]] = None
        current_5m_bar_forming: Optional[Dict[str, Any]] = None

        current_processing_day_local: Optional[datetime.date] = None
        intraday_high: Optional[float] = None
        intraday_low: Optional[float] = None
        previous_day_close_price: Optional[float] = None


        # Pre-fetch initial LOCF values if possible from data just before iteration_start_utc
        initial_locf_time_end = iteration_start_utc - timedelta(seconds=1)
        initial_locf_time_start = initial_locf_time_end - timedelta(minutes=5) # Look back 5 mins for initial locf

        if not self.raw_trades_df.empty:
            initial_trades = self.raw_trades_df.loc[initial_locf_time_start:initial_locf_time_end]
            if not initial_trades.empty:
                locf_current_price = float(initial_trades.iloc[-1]['price'])

        if not self.raw_quotes_df.empty:
            initial_quotes_slice = self.raw_quotes_df.loc[initial_locf_time_start:initial_locf_time_end]
            if not initial_quotes_slice.empty:
                # Simplified BBO from last few quotes
                last_bids = initial_quotes_slice[initial_quotes_slice['side'].str.lower() == 'b'] #
                last_asks = initial_quotes_slice[initial_quotes_slice['side'].str.lower() == 'a'] #
                if not last_bids.empty:
                    locf_best_bid_price = float(last_bids.iloc[-1]['price'])
                    locf_best_bid_size = int(last_bids.iloc[-1]['size'])
                if not last_asks.empty:
                    locf_best_ask_price = float(last_asks.iloc[-1]['price'])
                    locf_best_ask_size = int(last_asks.iloc[-1]['size'])

        self.logger.info(f"Initial LOCF before main loop: Price={locf_current_price}, Bid={locf_best_bid_price}, Ask={locf_best_ask_price}")

        # Iterate second by second over the full range (priming + session)
        current_iter_ts = iteration_start_utc
        processed_count = 0
        while current_iter_ts <= iteration_end_utc:
            # --- Daily Resets (High/Low, Previous Day Close) ---
            ts_local_datetime = current_iter_ts.astimezone(self.exchange_tz)
            ts_local_date = ts_local_datetime.date()

            if current_processing_day_local != ts_local_date:
                self.logger.debug(f"Processing new day {ts_local_date} at UTC {current_iter_ts}")
                current_processing_day_local = ts_local_date
                intraday_high = None
                intraday_low = None
                # Fetch previous day's close based on the *new* current_processing_day_local
                previous_day_close_price = self._get_previous_trading_day_close(current_processing_day_local)
                self.logger.info(f"Previous trading day close for {current_processing_day_local}: {previous_day_close_price}")
                current_1m_bar_forming = None # Reset forming bars on new day
                current_5m_bar_forming = None


            # --- Initialize data for the current second ---
            actual_1s_bar_data: Optional[Dict[str, Any]] = None
            trades_this_second: List[Dict] = []
            quotes_this_second: List[Dict] = []

            # 1. Get/Generate 1s Bar for current_iter_ts
            # Try provider bars first
            if not self.raw_1s_bars_provider_df.empty:
                try:
                    bar_series = self.raw_1s_bars_provider_df.loc[current_iter_ts]
                    actual_1s_bar_data = bar_series.to_dict()
                    actual_1s_bar_data['timestamp'] = current_iter_ts
                except KeyError:
                    pass # No provider bar for this second

            # If no provider bar, try to generate from trades
            if actual_1s_bar_data is None and not self.raw_trades_df.empty:
                trades_for_bar_gen = self._get_raw_data_for_interval(self.raw_trades_df, current_iter_ts, timedelta(seconds=1))
                if trades_for_bar_gen:
                    actual_1s_bar_data = self._aggregate_trades_to_bar(trades_for_bar_gen, current_iter_ts)

            # If still no bar (no trades, no provider bar), create a synthetic bar using LOCF
            if actual_1s_bar_data is None:
                if locf_current_price is not None:
                    actual_1s_bar_data = {
                        'timestamp': current_iter_ts, 'open': locf_current_price, 'high': locf_current_price,
                        'low': locf_current_price, 'close': locf_current_price, 'volume': 0.0, 'vwap': locf_current_price
                    }
                # else: remains None if no locf_current_price yet (should be rare after initial LOCF)

            # 2. Get Trades for current_iter_ts
            if not self.raw_trades_df.empty:
                trades_this_second = self._get_raw_data_for_interval(self.raw_trades_df, current_iter_ts, timedelta(seconds=1))

            # 3. Get Quotes for current_iter_ts
            if not self.raw_quotes_df.empty:
                quotes_this_second = self._get_raw_data_for_interval(self.raw_quotes_df, current_iter_ts, timedelta(seconds=1))

            # --- Update LOCF values ---
            if trades_this_second:
                locf_current_price = float(trades_this_second[-1]['price'])
            elif actual_1s_bar_data and pd.notna(actual_1s_bar_data.get('close')):
                locf_current_price = float(actual_1s_bar_data['close'])
            # else: locf_current_price carries forward

            current_bbo = self._get_bbo_from_quotes(quotes_this_second)
            if current_bbo['bid_price'] is not None: locf_best_bid_price = current_bbo['bid_price']
            if current_bbo['ask_price'] is not None: locf_best_ask_price = current_bbo['ask_price']
            if current_bbo['bid_size'] > 0 : locf_best_bid_size = current_bbo['bid_size']
            if current_bbo['ask_size'] > 0 : locf_best_ask_size = current_bbo['ask_size']

            # Fallback for BBO if still None, using locf_current_price (e.g. very thin market or start)
            if locf_best_bid_price is None and locf_current_price is not None: locf_best_bid_price = locf_current_price
            if locf_best_ask_price is None and locf_current_price is not None: locf_best_ask_price = locf_current_price


            # --- Update Intraday High/Low ---
            if actual_1s_bar_data:
                bar_high = actual_1s_bar_data.get('high')
                bar_low = actual_1s_bar_data.get('low')
                if pd.notna(bar_high):
                    intraday_high = bar_high if intraday_high is None else max(intraday_high, bar_high)
                if pd.notna(bar_low):
                    intraday_low = bar_low if intraday_low is None else min(intraday_low, bar_low)


            # --- Update Aggregated Bars (1m, 5m) ---
            if actual_1s_bar_data: # Only update if there's some bar data for the second
                current_1m_bar_forming, current_5m_bar_forming = self._update_longer_timeframe_bars(
                    actual_1s_bar_data, current_iter_ts,
                    current_1m_bar_forming, current_5m_bar_forming,
                    completed_1m_bars, completed_5m_bars
                )

            # --- Store event data for rolling 1s window ---
            # This structure is what FeatureExtractor expects in `rolling_1s_data_window`
            event_for_deque = {
                'timestamp': current_iter_ts,
                'bar': actual_1s_bar_data, # This is the full 1s bar for this exact second
                'trades': trades_this_second,
                'quotes': quotes_this_second
            }
            rolling_1s_event_data.append(event_for_deque)

            # --- If current_iter_ts is part of the agent's actual session, precompute and store its state ---
            if self._agent_timeline_utc and self._agent_timeline_utc[0] <= current_iter_ts <= self._agent_timeline_utc[-1]:
                # Check if current_iter_ts is one of the exact seconds in agent_timeline_utc
                # This can be optimized if _agent_timeline_utc is guaranteed to be contiguous 1s steps
                # For now, a direct check is fine.
                # Since _agent_timeline_utc is already pre-generated with 1s steps, this check is sufficient.
                if current_iter_ts in self._agent_timeline_utc: # Optimization: could use set for faster lookups if timeline is huge
                    state = {
                        'timestamp_utc': current_iter_ts,
                        'current_market_session': self._determine_market_session(current_iter_ts),
                        'current_price': locf_current_price,
                        'best_bid_price': locf_best_bid_price,
                        'best_ask_price': locf_best_ask_price,
                        'best_bid_size': locf_best_bid_size,
                        'best_ask_size': locf_best_ask_size,
                        'intraday_high': intraday_high,
                        'intraday_low': intraday_low,
                        'current_1s_bar': actual_1s_bar_data, # The bar for this specific second
                        'current_1m_bar_forming': current_1m_bar_forming,
                        'current_5m_bar_forming': current_5m_bar_forming,
                        'rolling_1s_data_window': list(rolling_1s_event_data), # For FeatureExtractor
                        'completed_1m_bars_window': list(completed_1m_bars),  # For FeatureExtractor
                        'completed_5m_bars_window': list(completed_5m_bars),  # For FeatureExtractor
                        'historical_1d_bars': self.historical_1d_bars_df.copy() if not self.historical_1d_bars_df.empty else pd.DataFrame(), # For FeatureExtractor
                        'previous_day_close_price': previous_day_close_price
                    }
                    self._precomputed_states[current_iter_ts] = state

            processed_count += 1
            if processed_count % 10000 == 0:
                 self.logger.info(f"State precomputation progress: {processed_count} seconds processed. Last ts: {current_iter_ts}")

            current_iter_ts += timedelta(seconds=1)

        self.logger.info(f"Finished precomputing states. {len(self._precomputed_states)} states stored for the agent's timeline.")


    def _get_raw_data_for_interval(self, df: pd.DataFrame, interval_end_ts: datetime, interval_duration: timedelta) -> List[Dict]:
        """Fetches rows from df that fall strictly within (interval_end_ts - interval_duration, interval_end_ts]."""
        if df.empty:
            return []
        interval_start_ts = interval_end_ts - interval_duration
        # Slice data: index > start_ts AND index <= end_ts
        # Ensure interval_end_ts is exclusive for start if interval_duration is 1s (i.e. current second data)
        # For a 1s interval ending at T, we want data from (T-1s) to T.
        # If data timestamp is exactly T, it's included. If exactly T-1s, it's not.
        subset = df[(df.index > interval_start_ts) & (df.index <= interval_end_ts)]
        return [row.to_dict() for _, row in subset.iterrows()]


    def _aggregate_trades_to_bar(self, trades_in_interval: List[Dict], bar_timestamp: datetime) -> Optional[Dict[str, Any]]:
        if not trades_in_interval:
            return None
        prices = [float(t['price']) for t in trades_in_interval if pd.notna(t['price'])]
        sizes = [float(t['size']) for t in trades_in_interval if pd.notna(t['size'])]
        if not prices or not sizes: return None

        total_volume = sum(sizes)
        if total_volume < self.EPSILON:
            # If volume is zero, use first trade price for OHLVC
            # This case should be rare if trades_in_interval is not empty and sizes are valid
            price_val = prices[0]
            return {'timestamp': bar_timestamp, 'open': price_val, 'high': price_val, 'low': price_val,
                    'close': price_val, 'volume': 0.0, 'vwap': price_val}

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

    def _get_bbo_from_quotes(self, quotes_in_interval: List[Dict]) -> Dict[str, Any]:
        best_bid_price: Optional[float] = None
        best_ask_price: Optional[float] = None
        bid_size_at_best: int = 0
        ask_size_at_best: int = 0

        bids = {} # price -> total_size
        asks = {} # price -> total_size

        for q in quotes_in_interval:
            price = q.get('price')
            size = q.get('size')
            side = str(q.get('side','')).lower() #
            if pd.notna(price) and pd.notna(size) and side in ['bid', 'ask', 'b', 'a']: #
                try:
                    price_f = float(price)
                    size_i = int(size)
                    if side in ['bid', 'b']: #
                        bids[price_f] = bids.get(price_f, 0) + size_i
                    else: # ask or 'a'
                        asks[price_f] = asks.get(price_f, 0) + size_i
                except ValueError:
                    self.logger.debug(f"Could not parse quote price/size: {q}")


        if bids:
            best_bid_price = max(bids.keys())
            bid_size_at_best = bids[best_bid_price]
        if asks:
            best_ask_price = min(asks.keys())
            ask_size_at_best = asks[best_ask_price]

        return {'bid_price': best_bid_price, 'ask_price': best_ask_price,
                'bid_size': bid_size_at_best, 'ask_size': ask_size_at_best}


    def _update_longer_timeframe_bars(self,
                                     current_1s_bar_data: Dict[str, Any],
                                     timestamp: datetime,
                                     current_1m_bar_forming: Optional[Dict[str, Any]],
                                     current_5m_bar_forming: Optional[Dict[str, Any]],
                                     completed_1m_bars_deque: deque,
                                     completed_5m_bars_deque: deque
                                     ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        s_open = current_1s_bar_data.get('open')
        s_high = current_1s_bar_data.get('high')
        s_low = current_1s_bar_data.get('low')
        s_close = current_1s_bar_data.get('close')
        s_volume = current_1s_bar_data.get('volume', 0.0)

        if not all(pd.notna(v) for v in [s_open, s_high, s_low, s_close]):
            # Not enough data in 1s bar to update longer TFs
            return current_1m_bar_forming, current_5m_bar_forming

        # --- 1 Minute Bar Aggregation ---
        minute_start_ts = timestamp.replace(second=0, microsecond=0)
        if current_1m_bar_forming is None or current_1m_bar_forming['timestamp_start'] != minute_start_ts:
            if current_1m_bar_forming is not None: # A 1m bar just completed
                completed_1m_bars_deque.append(current_1m_bar_forming.copy())
            # Start new 1m bar
            current_1m_bar_forming = {
                'timestamp_start': minute_start_ts, 'open': s_open, 'high': s_high,
                'low': s_low, 'close': s_close, 'volume': s_volume
            }
        else: # Continue current 1m bar
            current_1m_bar_forming['high'] = max(current_1m_bar_forming['high'], s_high)
            current_1m_bar_forming['low'] = min(current_1m_bar_forming['low'], s_low)
            current_1m_bar_forming['close'] = s_close
            current_1m_bar_forming['volume'] += s_volume

        # --- 5 Minute Bar Aggregation ---
        current_minute_val = timestamp.minute
        five_min_slot_start_minute = (current_minute_val // 5) * 5
        five_min_start_ts = timestamp.replace(minute=five_min_slot_start_minute, second=0, microsecond=0)

        if current_5m_bar_forming is None or current_5m_bar_forming['timestamp_start'] != five_min_start_ts:
            if current_5m_bar_forming is not None: # A 5m bar just completed
                completed_5m_bars_deque.append(current_5m_bar_forming.copy())
            # Start new 5m bar
            current_5m_bar_forming = {
                'timestamp_start': five_min_start_ts, 'open': s_open, 'high': s_high,
                'low': s_low, 'close': s_close, 'volume': s_volume
            }
        else: # Continue current 5m bar
            current_5m_bar_forming['high'] = max(current_5m_bar_forming['high'], s_high)
            current_5m_bar_forming['low'] = min(current_5m_bar_forming['low'], s_low)
            current_5m_bar_forming['close'] = s_close
            current_5m_bar_forming['volume'] += s_volume

        return current_1m_bar_forming, current_5m_bar_forming

    def _determine_market_session(self, timestamp_utc: datetime) -> str:
        local_time = timestamp_utc.astimezone(self.exchange_tz).time()
        # Assuming PREMARKET_END < REGULAR_START < REGULAR_END < POSTMARKET_START < POSTMARKET_END
        if self.session_start_time_local <= local_time <= self.session_end_time_local:
            if self.regular_market_start_local <= local_time <= self.regular_market_end_local:
                return "REGULAR"
            elif local_time < self.regular_market_start_local:
                return "PREMARKET"
            else: # local_time > self.regular_market_end_local
                return "POSTMARKET"
        return "CLOSED"

    def _create_empty_state(self, timestamp_utc: datetime) -> Dict[str, Any]:
        """Creates a default/empty state for a given timestamp."""
        return {
            'timestamp_utc': timestamp_utc,
            'current_market_session': self._determine_market_session(timestamp_utc),
            'current_price': None, 'best_bid_price': None, 'best_ask_price': None,
            'best_bid_size': 0, 'best_ask_size': 0, 'intraday_high': None, 'intraday_low': None,
            'current_1s_bar': None, 'current_1m_bar_forming': None, 'current_5m_bar_forming': None,
            'rolling_1s_data_window': [], 'completed_1m_bars_window': [], 'completed_5m_bars_window': [],
            'historical_1d_bars': self.historical_1d_bars_df.copy() if not self.historical_1d_bars_df.empty else pd.DataFrame(),
            'previous_day_close_price': None # This would ideally be filled by _get_previous_trading_day_close logic
        }

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        if self._current_agent_time_idx < 0 or self._current_agent_time_idx >= len(self._agent_timeline_utc):
            self.logger.warning(f"Current agent time index {self._current_agent_time_idx} is invalid for timeline of length {len(self._agent_timeline_utc)}.")
            # Fallback: try to use current_timestamp_utc if set, else last known good state, or None
            if self.current_timestamp_utc:
                return self.get_state_at_time(self.current_timestamp_utc)
            return self._create_empty_state(datetime.now(ZoneInfo("UTC"))) if not self._precomputed_states else list(self._precomputed_states.values())[-1]


        current_ts_on_timeline = self._agent_timeline_utc[self._current_agent_time_idx]
        if self.current_timestamp_utc != current_ts_on_timeline:
             self.logger.warning(f"Internal timestamp mismatch: self.current_timestamp_utc {self.current_timestamp_utc} vs timeline {current_ts_on_timeline}. Syncing.")
             self.current_timestamp_utc = current_ts_on_timeline

        return self._precomputed_states.get(self.current_timestamp_utc)


    def get_state_at_time(self, timestamp: datetime, tolerance_seconds: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get precomputed state. If exact match not found, look for closest PAST state within tolerance.
        """
        # Ensure timestamp is UTC
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
        elif timestamp.tzinfo != ZoneInfo("UTC"):
            timestamp = timestamp.astimezone(ZoneInfo("UTC"))

        exact_match = self._precomputed_states.get(timestamp)
        if exact_match:
            return exact_match

        # If no exact match, find the closest past state within tolerance
        # bisect_left returns index `i` where all `e` in `a[:i]` have `e < x`, and all `e` in `a[i:]` have `e >= x`.
        # So, `self._agent_timeline_utc[i-1]` is the largest timestamp `< x`.
        if not self._agent_timeline_utc:
            self.logger.warning(f"State requested for {timestamp}, but agent timeline is empty.")
            return self._create_empty_state(timestamp)

        idx = bisect.bisect_left(self._agent_timeline_utc, timestamp)

        if idx > 0:
            candidate_ts = self._agent_timeline_utc[idx - 1]
            if (timestamp - candidate_ts) <= timedelta(seconds=tolerance_seconds):
                self.logger.debug(f"Exact state for {timestamp} not found. Using closest past state from {candidate_ts}.")
                return self._precomputed_states.get(candidate_ts)

        self.logger.warning(f"State for {timestamp} not found (tolerance {tolerance_seconds}s). Returning empty state.")
        # Try to create an empty state with the best available previous_day_close
        # This might occur if execution simulator requests a time slightly outside precomputed range
        # due to latency, and it's not found within tolerance.
        empty_state_with_potential_prev_close = self._create_empty_state(timestamp)
        if not self._precomputed_states: # No precomputed states at all
             current_processing_day_local = timestamp.astimezone(self.exchange_tz).date()
             empty_state_with_potential_prev_close['previous_day_close_price'] = self._get_previous_trading_day_close(current_processing_day_local)
        else: # Try to get from last known state
            last_known_state = list(self._precomputed_states.values())[-1]
            empty_state_with_potential_prev_close['previous_day_close_price'] = last_known_state.get('previous_day_close_price')

        return empty_state_with_potential_prev_close


    def step(self) -> bool:
        if self.is_done():
            self.logger.info("Step called, but simulation is already done.")
            return False

        self._current_agent_time_idx += 1
        if self._current_agent_time_idx < len(self._agent_timeline_utc):
            self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
            return True
        else: # Should be caught by is_done(), but as safeguard
            self._current_agent_time_idx = len(self._agent_timeline_utc) -1 # Stay at last step
            self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
            self.logger.info("Reached end of timeline data.")
            return False


    def is_done(self) -> bool:
        if not self._agent_timeline_utc: return True
        return self._current_agent_time_idx >= len(self._agent_timeline_utc) - 1

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        options = options or {}
        self.logger.info(f"Resetting MarketSimulator with options: {options}")

        if not self._agent_timeline_utc:
            self.logger.error("Cannot reset: Agent timeline is not available.")
            return self._create_empty_state(self.session_start_utc or datetime.now(ZoneInfo("UTC")))

        # The priming lookback is already handled by _precompute_timeline_states
        # The agent always starts on or after self._agent_timeline_utc[0]
        # initial_buffer_seconds from market_config is about how far into the *agent's visible session* to start
        buffer_seconds = self.market_config.get('initial_buffer_seconds', 0)
        min_start_delay_from_session_start = timedelta(seconds=buffer_seconds)
        earliest_possible_agent_start_time = self._agent_timeline_utc[0] + min_start_delay_from_session_start

        # Find index for earliest_possible_agent_start_time
        start_idx_after_buffer = bisect.bisect_left(self._agent_timeline_utc, earliest_possible_agent_start_time)
        start_idx_after_buffer = min(start_idx_after_buffer, len(self._agent_timeline_utc) - 1) # Ensure within bounds

        if options.get('random_start', False):
            # Random start point is between start_idx_after_buffer and end of timeline
            if start_idx_after_buffer < len(self._agent_timeline_utc) -1 :
                 # Ensure numpy random generator is available, e.g., from gym.Env
                 if hasattr(self, 'np_random') and self.np_random is not None:
                     self._current_agent_time_idx = self.np_random.integers(start_idx_after_buffer, len(self._agent_timeline_utc))
                 else: # Fallback if np_random not set (should be by gym.Env)
                     self._current_agent_time_idx = np.random.randint(start_idx_after_buffer, len(self._agent_timeline_utc))
            else: # Not enough space for random selection after buffer
                 self._current_agent_time_idx = start_idx_after_buffer
            self.logger.info(f"Random reset to index {self._current_agent_time_idx} within agent timeline.")
        else:
            self._current_agent_time_idx = start_idx_after_buffer
            self.logger.info(f"Reset to index {self._current_agent_time_idx} (after initial buffer).")

        if self._current_agent_time_idx < len(self._agent_timeline_utc):
            self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
            self.logger.info(f"Simulator reset. Current agent time: {self.current_timestamp_utc}")
            return self.get_current_market_state()
        else:
            self.logger.error("Reset resulted in invalid agent time index.")
            # Fallback if something went very wrong
            self._current_agent_time_idx = 0
            self.current_timestamp_utc = self._agent_timeline_utc[0]
            return self.get_current_market_state()

    def get_symbol_info(self):
        # This should ideally come from DataManager or config if available
        # Placeholder:
        return {
            "symbol": self.symbol,
            "total_shares_outstanding": self.market_config.get('total_shares_outstanding', 100_000_000),
            # Add other static symbol info if needed
        }

    def close(self):
        self.logger.info("Closing MarketSimulator")
        self._precomputed_states.clear()
        self._agent_timeline_utc.clear()
        # Release pandas DataFrames if they are large
        del self.raw_trades_df
        del self.raw_quotes_df
        del self.raw_1s_bars_provider_df
        del self.historical_1d_bars_df
        self.logger.info("MarketSimulator closed and resources cleared.")