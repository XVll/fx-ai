# market_simulator.py
import logging
from collections import deque
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo  # Use zoneinfo for timezone handling
import numpy as np
import pandas as pd
import bisect
from typing import Any, Dict, List, Optional, Tuple, Set, Union

# Assuming these are external and exist
from config.config import MarketConfig, FeatureConfig  # User specified this path
from data.data_manager import DataManager  # User specified this path

DEFAULT_MARKET_HOURS_州的东部时间 = {  # Using a more descriptive name for clarity
    "PREMARKET_START_ET": time(4, 0, 0),
    "REGULAR_START_ET": time(9, 30, 0),
    "REGULAR_END_ET": time(15, 59, 59),  # Market typically closes at 16:00:00
    "POSTMARKET_END_ET": time(19, 59, 59),  # Session ends at 20:00:00
    "SESSION_START_ET": time(4, 0, 0),  # Full extended session start
    "SESSION_END_ET": time(19, 59, 59),  # Full extended session end
    "TIMEZONE_ET": "America/New_York"
}
# Define the close time for "previous day close" (e.g., 8 PM ET)
PREVIOUS_DAY_CLOSING_TIME_ET = time(19, 59, 59)  # Using 7:59:59 PM ET as official end of post-market


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
                 start_time: Optional[Union[str, datetime]] = None,  # For backtesting: actual session start for agent
                 end_time: Optional[Union[str, datetime]] = None,  # For backtesting: actual session end for agent
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
        self._agent_timeline_utc: List[datetime] = []  # Timestamps agent will step through

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
        self.raw_1s_bars_provider_df: pd.DataFrame = pd.DataFrame()  # Bars from provider
        self.historical_1d_bars_df: pd.DataFrame = pd.DataFrame()  # Long-term daily bars

        self._initialize_simulator()

    def _setup_market_hours_and_timezone(self):
        cfg_hours = self.market_config.get('market_hours', {})
        self.exchange_timezone_str = cfg_hours.get('TIMEZONE', DEFAULT_MARKET_HOURS_州的东部时间['TIMEZONE_ET'])
        self.exchange_tz = ZoneInfo(self.exchange_timezone_str)

        self.session_start_time_local = cfg_hours.get('SESSION_START_ET', DEFAULT_MARKET_HOURS_州的东部时间['SESSION_START_ET'])
        self.session_end_time_local = cfg_hours.get('SESSION_END_ET', DEFAULT_MARKET_HOURS_州的东部时间['SESSION_END_ET'])
        self.prev_day_close_time_local = cfg_hours.get('PREVIOUS_DAY_CLOSING_TIME_ET', PREVIOUS_DAY_CLOSING_TIME_ET)

        self.regular_market_start_local = cfg_hours.get('REGULAR_START_ET', DEFAULT_MARKET_HOURS_州的东部时间['REGULAR_START_ET'])
        self.regular_market_end_local = cfg_hours.get('REGULAR_END_ET', DEFAULT_MARKET_HOURS_州的东部时间['REGULAR_END_ET'])

    def _parse_datetime_to_utc(self, dt_input: Optional[Union[str, datetime]]) -> Optional[datetime]:
        if dt_input is None:
            return None
        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None:  # Naive, assume it's in exchange timezone
                return self.exchange_tz.localize(dt_input).astimezone(ZoneInfo("UTC"))
            return dt_input.astimezone(ZoneInfo("UTC"))
        try:  # String
            parsed_dt = pd.Timestamp(dt_input)
            if parsed_dt.tzinfo is None:
                return self.exchange_tz.localize(parsed_dt.to_pydatetime()).astimezone(ZoneInfo("UTC"))
            return parsed_dt.tz_convert('UTC').to_pydatetime()
        except Exception as e:
            self.logger.error(f"Failed to parse datetime '{dt_input}': {e}")
            return None

    def _calculate_priming_lookback_seconds(self) -> int:
        # Max seconds needed for the data stored in deques that FeatureExtractor uses
        s1_data_duration = self.rolling_1s_event_window_size  # N seconds
        m1_bars_duration = self.completed_1m_bars_window_size * 60  # N_1m_bars * 60s
        m5_bars_duration = self.completed_5m_bars_window_size * 5 * 60  # N_5m_bars * 300s
        return max(s1_data_duration, m1_bars_duration, m5_bars_duration, 3600)  # Ensure at least 1 hour

    def _initialize_simulator(self):
        self.logger.info(f"Initializing MarketSimulator for {self.symbol} in {self.mode} mode.")
        self._load_data_for_simulation()

        if self.mode == 'backtesting':
            if not self.session_start_utc or not self.session_end_utc:
                self.logger.error("Backtesting mode requires session_start_utc and session_end_utc.")
                return
            self._precompute_timeline_states()
            self.reset()  # Set initial agent time
        elif self.mode == 'live':
            # In live mode, precompute for historical part if session_start_utc is in past
            # And then prepare for real-time updates (not fully implemented here, focuses on backtest rewrite)
            self.logger.warning("Live mode initialization is simplified in this rewrite, focusing on backtesting path.")
            if self.session_start_utc and self.session_start_utc < datetime.now(ZoneInfo("UTC")):
                self._precompute_timeline_states()  # Precompute history
            # For true live, would need to connect to data stream
            self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
            if not self._agent_timeline_utc or self.current_timestamp_utc > self._agent_timeline_utc[-1]:
                self._agent_timeline_utc.append(self.current_timestamp_utc)  # Add current time if not covered
                self._agent_timeline_utc.sort()
            # Ensure a state for current live time, may need to calculate it on the fly if not precomputed
            if self.current_timestamp_utc not in self._precomputed_states:
                # This part is complex for live if not precomputed, placeholder for default state:
                self._precomputed_states[self.current_timestamp_utc] = self._create_empty_state(self.current_timestamp_utc)

            idx = bisect.bisect_left(self._agent_timeline_utc, self.current_timestamp_utc)
            self._current_agent_time_idx = min(idx, len(self._agent_timeline_utc) - 1) if self._agent_timeline_utc else -1

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _load_data_for_simulation(self):
        if not self.session_start_utc:
            self.logger.error("Cannot load data: session_start_utc is not defined.")
            return

        # 1. Determine data loading range (includes priming period)
        load_range_start_utc = self.session_start_utc - timedelta(seconds=self._priming_lookback_seconds)
        load_range_start_utc -= timedelta(days=self.market_config.get('data_load_day_buffer', 2))

        load_range_end_utc = self.session_end_utc
        if self.mode == 'live' and (not load_range_end_utc or load_range_end_utc < datetime.now(ZoneInfo("UTC"))):
            load_range_end_utc = datetime.now(ZoneInfo("UTC")) + timedelta(minutes=5)

        self.logger.info(f"Data loading range for symbol {self.symbol}: {load_range_start_utc} to {load_range_end_utc}")

        data_types = ["trades", "quotes"]
        if self.market_config.get("use_provider_1s_bars", False):
            data_types.append("bars_1s")

        loaded_data = self.data_manager.load_data(
            symbols=[self.symbol],
            start_time=load_range_start_utc,
            end_time=load_range_end_utc,
            data_types=data_types
        )
        symbol_data = loaded_data.get(self.symbol, {})
        self.raw_trades_df = self._prepare_dataframe(symbol_data.get("trades"), ['price', 'size'])

        # For quotes, use the columns provided by DabentoFileProvider's get_quotes standardization
        # These are: bid_price, ask_price, bid_size, ask_size, (optional: bid_count, ask_count, side, exchange)
        # We need at least bid_price, ask_price, bid_size, ask_size for BBO.
        # 'side' from Databento mbp-1 is aggressor side, not strictly needed for BBO from snapshot.
        quote_cols = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
        self.raw_quotes_df = self._prepare_dataframe(symbol_data.get("quotes"), quote_cols)

        if "bars_1s" in data_types:
            self.raw_1s_bars_provider_df = self._prepare_dataframe(symbol_data.get("bars_1s"), ['open', 'high', 'low', 'close', 'volume'])

        daily_bars_load_end_utc = load_range_start_utc
        daily_bars_load_start_utc = daily_bars_load_end_utc - timedelta(days=self.market_config.get('historical_daily_bars_lookback_days', 730))
        try:
            self.historical_1d_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1d",
                start_time=daily_bars_load_start_utc,
                end_time=daily_bars_load_end_utc
            )
            self.historical_1d_bars_df = self._prepare_dataframe(self.historical_1d_bars_df, ['open', 'high', 'low', 'close', 'volume', 'close_8pm'])
            if 'close_8pm' not in self.historical_1d_bars_df.columns and 'close' in self.historical_1d_bars_df.columns:
                self.historical_1d_bars_df['close_8pm'] = self.historical_1d_bars_df['close']
            self.logger.info(f"Loaded {len(self.historical_1d_bars_df)} historical 1D bars up to {daily_bars_load_end_utc}.")
        except Exception as e:
            self.logger.error(f"Failed to load historical 1D bars: {e}")
            self.historical_1d_bars_df = self._prepare_dataframe(None, ['open', 'high', 'low', 'close', 'volume', 'close_8pm'])

    def _prepare_dataframe(self, df: Optional[pd.DataFrame], required_cols: List[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=required_cols).set_index(pd.DatetimeIndex([], tz='UTC'))
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("Input DataFrame does not have a DatetimeIndex. Attempting conversion or returning empty.")
            return pd.DataFrame(columns=required_cols).set_index(pd.DatetimeIndex([], tz='UTC'))

        if df.index.tzinfo is None:
            df = df.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        elif df.index.tzinfo != ZoneInfo("UTC"):
            df = df.tz_convert('UTC')

        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df.sort_index()

    def _get_previous_trading_day_close(self, current_session_date_utc: datetime.date) -> Optional[float]:
        if self.historical_1d_bars_df.empty:
            return None
        relevant_bars = self.historical_1d_bars_df[self.historical_1d_bars_df.index.date < current_session_date_utc]
        if relevant_bars.empty:
            return None
        close_col = 'close_8pm' if 'close_8pm' in relevant_bars.columns else 'close'
        prev_close = relevant_bars.iloc[-1][close_col]
        return float(prev_close) if pd.notna(prev_close) else None

    def _precompute_timeline_states(self):
        self.logger.info(f"Starting precomputation of timeline states for {self.symbol}.")
        if not self.session_start_utc or not self.session_end_utc:
            self.logger.error("Cannot precompute: session_start_utc or session_end_utc not set.")
            return

        current_ts = self.session_start_utc
        agent_session_timeline = []
        while current_ts <= self.session_end_utc:
            agent_session_timeline.append(current_ts)
            current_ts += timedelta(seconds=1)
        self._agent_timeline_utc = agent_session_timeline
        if not self._agent_timeline_utc:
            self.logger.error("Agent session timeline is empty. Check session start/end times.")
            return

        iteration_start_utc = self._agent_timeline_utc[0] - timedelta(seconds=self._priming_lookback_seconds)
        iteration_end_utc = self._agent_timeline_utc[-1]

        self.logger.info(f"Iteration range for state building: {iteration_start_utc} to {iteration_end_utc}")
        self.logger.info(f"Agent visible timeline: {self._agent_timeline_utc[0]} to {self._agent_timeline_utc[-1]}")

        locf_current_price: Optional[float] = None
        locf_best_bid_price: Optional[float] = None
        locf_best_ask_price: Optional[float] = None
        locf_best_bid_size: int = 0
        locf_best_ask_size: int = 0

        rolling_1s_event_data = deque(maxlen=self.rolling_1s_event_window_size)
        completed_1m_bars = deque(maxlen=self.completed_1m_bars_window_size)
        completed_5m_bars = deque(maxlen=self.completed_5m_bars_window_size)

        current_1m_bar_forming: Optional[Dict[str, Any]] = None
        current_5m_bar_forming: Optional[Dict[str, Any]] = None
        current_processing_day_local: Optional[datetime.date] = None
        intraday_high: Optional[float] = None
        intraday_low: Optional[float] = None
        previous_day_close_price: Optional[float] = None

        initial_locf_time_end = iteration_start_utc - timedelta(seconds=1)
        initial_locf_time_start = initial_locf_time_end - timedelta(minutes=5)

        if not self.raw_trades_df.empty:
            initial_trades = self.raw_trades_df.loc[initial_locf_time_start:initial_locf_time_end]
            if not initial_trades.empty:
                locf_current_price = float(initial_trades.iloc[-1]['price'])

        if not self.raw_quotes_df.empty:
            initial_quotes_slice = self.raw_quotes_df.loc[initial_locf_time_start:initial_locf_time_end]
            if not initial_quotes_slice.empty:
                last_quote_snapshot = initial_quotes_slice.iloc[-1]
                if pd.notna(last_quote_snapshot.get('bid_price')):
                    locf_best_bid_price = float(last_quote_snapshot['bid_price'])
                # Handle potential NaN for size by defaulting to 0
                locf_best_bid_size = int(last_quote_snapshot['bid_size']) if pd.notna(last_quote_snapshot.get('bid_size')) else 0

                if pd.notna(last_quote_snapshot.get('ask_price')):
                    locf_best_ask_price = float(last_quote_snapshot['ask_price'])
                # Handle potential NaN for size by defaulting to 0
                locf_best_ask_size = int(last_quote_snapshot['ask_size']) if pd.notna(last_quote_snapshot.get('ask_size')) else 0

        self.logger.info(
            f"Initial LOCF before main loop: Price={locf_current_price}, Bid={locf_best_bid_price}, Ask={locf_best_ask_price}, BidSize={locf_best_bid_size}, AskSize={locf_best_ask_size}")

        current_iter_ts = iteration_start_utc
        processed_count = 0
        while current_iter_ts <= iteration_end_utc:
            ts_local_datetime = current_iter_ts.astimezone(self.exchange_tz)
            ts_local_date = ts_local_datetime.date()

            if current_processing_day_local != ts_local_date:
                self.logger.debug(f"Processing new day {ts_local_date} at UTC {current_iter_ts}")
                current_processing_day_local = ts_local_date
                intraday_high = None
                intraday_low = None
                previous_day_close_price = self._get_previous_trading_day_close(current_processing_day_local)
                self.logger.info(f"Previous trading day close for {current_processing_day_local}: {previous_day_close_price}")
                current_1m_bar_forming = None
                current_5m_bar_forming = None

            actual_1s_bar_data: Optional[Dict[str, Any]] = None
            trades_this_second: List[Dict] = []
            quotes_this_second: List[Dict] = []

            if not self.raw_1s_bars_provider_df.empty:
                try:
                    bar_series = self.raw_1s_bars_provider_df.loc[current_iter_ts]
                    actual_1s_bar_data = bar_series.to_dict()
                    actual_1s_bar_data['timestamp'] = current_iter_ts
                except KeyError:
                    pass

            if actual_1s_bar_data is None and not self.raw_trades_df.empty:
                trades_for_bar_gen = self._get_raw_data_for_interval(self.raw_trades_df, current_iter_ts, timedelta(seconds=1))
                if trades_for_bar_gen:
                    actual_1s_bar_data = self._aggregate_trades_to_bar(trades_for_bar_gen, current_iter_ts)

            if actual_1s_bar_data is None:
                if locf_current_price is not None:
                    actual_1s_bar_data = {
                        'timestamp': current_iter_ts, 'open': locf_current_price, 'high': locf_current_price,
                        'low': locf_current_price, 'close': locf_current_price, 'volume': 0.0, 'vwap': locf_current_price
                    }

            if not self.raw_trades_df.empty:
                trades_this_second = self._get_raw_data_for_interval(self.raw_trades_df, current_iter_ts, timedelta(seconds=1))

            if not self.raw_quotes_df.empty:
                quotes_this_second = self._get_raw_data_for_interval(self.raw_quotes_df, current_iter_ts, timedelta(seconds=1))

            if trades_this_second:
                locf_current_price = float(trades_this_second[-1]['price'])
            elif actual_1s_bar_data and pd.notna(actual_1s_bar_data.get('close')):
                locf_current_price = float(actual_1s_bar_data['close'])

            current_bbo = self._get_bbo_from_quotes(quotes_this_second)  # Expects list of BBO snapshots for the second
            if current_bbo['bid_price'] is not None: locf_best_bid_price = current_bbo['bid_price']
            if current_bbo['ask_price'] is not None: locf_best_ask_price = current_bbo['ask_price']
            if current_bbo['bid_size'] > 0: locf_best_bid_size = current_bbo['bid_size']  # Already int from _get_bbo_from_quotes
            if current_bbo['ask_size'] > 0: locf_best_ask_size = current_bbo['ask_size']  # Already int

            if locf_best_bid_price is None and locf_current_price is not None: locf_best_bid_price = locf_current_price
            if locf_best_ask_price is None and locf_current_price is not None: locf_best_ask_price = locf_current_price

            if actual_1s_bar_data:
                bar_high = actual_1s_bar_data.get('high')
                bar_low = actual_1s_bar_data.get('low')
                if pd.notna(bar_high):
                    intraday_high = bar_high if intraday_high is None else max(intraday_high, bar_high)
                if pd.notna(bar_low):
                    intraday_low = bar_low if intraday_low is None else min(intraday_low, bar_low)

            if actual_1s_bar_data:
                current_1m_bar_forming, current_5m_bar_forming = self._update_longer_timeframe_bars(
                    actual_1s_bar_data, current_iter_ts,
                    current_1m_bar_forming, current_5m_bar_forming,
                    completed_1m_bars, completed_5m_bars
                )

            event_for_deque = {
                'timestamp': current_iter_ts,
                'bar': actual_1s_bar_data,
                'trades': trades_this_second,
                'quotes': quotes_this_second  # List of BBO snapshots for this second
            }
            rolling_1s_event_data.append(event_for_deque)

            if self._agent_timeline_utc and self._agent_timeline_utc[0] <= current_iter_ts <= self._agent_timeline_utc[-1]:
                if current_iter_ts in self._agent_timeline_utc:
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
                        'current_1s_bar': actual_1s_bar_data,
                        'current_1m_bar_forming': current_1m_bar_forming,
                        'current_5m_bar_forming': current_5m_bar_forming,
                        'rolling_1s_data_window': list(rolling_1s_event_data),
                        'completed_1m_bars_window': list(completed_1m_bars),
                        'completed_5m_bars_window': list(completed_5m_bars),
                        'historical_1d_bars': self.historical_1d_bars_df.copy() if not self.historical_1d_bars_df.empty else pd.DataFrame(),
                        'previous_day_close_price': previous_day_close_price
                    }
                    self._precomputed_states[current_iter_ts] = state

            processed_count += 1
            if processed_count % 10000 == 0:
                self.logger.info(f"State precomputation progress: {processed_count} seconds processed. Last ts: {current_iter_ts}")

            current_iter_ts += timedelta(seconds=1)

        self.logger.info(f"Finished precomputing states. {len(self._precomputed_states)} states stored for the agent's timeline.")

    def _get_raw_data_for_interval(self, df: pd.DataFrame, interval_end_ts: datetime, interval_duration: timedelta) -> List[Dict]:
        if df.empty:
            return []
        interval_start_ts = interval_end_ts - interval_duration
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
        """
        Extracts BBO from the last quote snapshot in the interval.
        Assumes quotes_in_interval contains BBO snapshots (e.g., from Databento MBP-1).
        """
        if not quotes_in_interval:  # No quotes in this specific second
            return {'bid_price': None, 'ask_price': None, 'bid_size': 0, 'ask_size': 0}

        # The last entry in quotes_in_interval is the latest BBO state for this second
        last_bbo_snapshot = quotes_in_interval[-1]

        best_bid_price_val = last_bbo_snapshot.get('bid_price')
        best_ask_price_val = last_bbo_snapshot.get('ask_price')
        bid_size_val = last_bbo_snapshot.get('bid_size')
        ask_size_val = last_bbo_snapshot.get('ask_size')

        return {
            'bid_price': float(best_bid_price_val) if pd.notna(best_bid_price_val) else None,
            'ask_price': float(best_ask_price_val) if pd.notna(best_ask_price_val) else None,
            'bid_size': int(bid_size_val) if pd.notna(bid_size_val) else 0,
            'ask_size': int(ask_size_val) if pd.notna(ask_size_val) else 0
        }

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
            return current_1m_bar_forming, current_5m_bar_forming

        minute_start_ts = timestamp.replace(second=0, microsecond=0)
        if current_1m_bar_forming is None or current_1m_bar_forming['timestamp_start'] != minute_start_ts:
            if current_1m_bar_forming is not None:
                completed_1m_bars_deque.append(current_1m_bar_forming.copy())
            current_1m_bar_forming = {
                'timestamp_start': minute_start_ts, 'open': s_open, 'high': s_high,
                'low': s_low, 'close': s_close, 'volume': s_volume
            }
        else:
            current_1m_bar_forming['high'] = max(current_1m_bar_forming['high'], s_high)
            current_1m_bar_forming['low'] = min(current_1m_bar_forming['low'], s_low)
            current_1m_bar_forming['close'] = s_close
            current_1m_bar_forming['volume'] += s_volume

        current_minute_val = timestamp.minute
        five_min_slot_start_minute = (current_minute_val // 5) * 5
        five_min_start_ts = timestamp.replace(minute=five_min_slot_start_minute, second=0, microsecond=0)

        if current_5m_bar_forming is None or current_5m_bar_forming['timestamp_start'] != five_min_start_ts:
            if current_5m_bar_forming is not None:
                completed_5m_bars_deque.append(current_5m_bar_forming.copy())
            current_5m_bar_forming = {
                'timestamp_start': five_min_start_ts, 'open': s_open, 'high': s_high,
                'low': s_low, 'close': s_close, 'volume': s_volume
            }
        else:
            current_5m_bar_forming['high'] = max(current_5m_bar_forming['high'], s_high)
            current_5m_bar_forming['low'] = min(current_5m_bar_forming['low'], s_low)
            current_5m_bar_forming['close'] = s_close
            current_5m_bar_forming['volume'] += s_volume

        return current_1m_bar_forming, current_5m_bar_forming

    def _determine_market_session(self, timestamp_utc: datetime) -> str:
        local_time = timestamp_utc.astimezone(self.exchange_tz).time()
        if self.session_start_time_local <= local_time <= self.session_end_time_local:
            if self.regular_market_start_local <= local_time <= self.regular_market_end_local:
                return "REGULAR"
            elif local_time < self.regular_market_start_local:
                return "PREMARKET"
            else:
                return "POSTMARKET"
        return "CLOSED"

    def _create_empty_state(self, timestamp_utc: datetime) -> Dict[str, Any]:
        empty_state = {
            'timestamp_utc': timestamp_utc,
            'current_market_session': self._determine_market_session(timestamp_utc),
            'current_price': None, 'best_bid_price': None, 'best_ask_price': None,
            'best_bid_size': 0, 'best_ask_size': 0, 'intraday_high': None, 'intraday_low': None,
            'current_1s_bar': None, 'current_1m_bar_forming': None, 'current_5m_bar_forming': None,
            'rolling_1s_data_window': [], 'completed_1m_bars_window': [], 'completed_5m_bars_window': [],
            'historical_1d_bars': self.historical_1d_bars_df.copy() if not self.historical_1d_bars_df.empty else pd.DataFrame(),
            'previous_day_close_price': None
        }
        # Try to populate previous_day_close_price if possible
        if not self._precomputed_states:  # If no states computed yet (e.g. very early error)
            current_processing_day_local = timestamp_utc.astimezone(self.exchange_tz).date()
            empty_state['previous_day_close_price'] = self._get_previous_trading_day_close(current_processing_day_local)
        elif self._agent_timeline_utc and timestamp_utc < self._agent_timeline_utc[0]:  # before agent timeline, try to get from first agent state
            first_agent_state_ts = self._agent_timeline_utc[0]
            if first_agent_state_ts in self._precomputed_states:
                empty_state['previous_day_close_price'] = self._precomputed_states[first_agent_state_ts].get('previous_day_close_price')
        elif self._precomputed_states:  # Try to get from last known state
            # Find the closest earlier precomputed state to derive prev_close
            # This is a simplified approach for empty states; ideally, prev_close is consistently calculated
            sorted_keys = sorted(self._precomputed_states.keys())
            idx = bisect.bisect_left(sorted_keys, timestamp_utc)
            if idx > 0:
                last_known_ts = sorted_keys[idx - 1]
                empty_state['previous_day_close_price'] = self._precomputed_states[last_known_ts].get('previous_day_close_price')
            elif sorted_keys:  # If timestamp_utc is before all known keys
                empty_state['previous_day_close_price'] = self._precomputed_states[sorted_keys[0]].get('previous_day_close_price')

        return empty_state

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        if self._current_agent_time_idx < 0 or self._current_agent_time_idx >= len(self._agent_timeline_utc):
            self.logger.warning(f"Current agent time index {self._current_agent_time_idx} is invalid for timeline of length {len(self._agent_timeline_utc)}.")
            if self.current_timestamp_utc:
                return self.get_state_at_time(self.current_timestamp_utc)
            # Fallback for when current_timestamp_utc might also be unset (e.g. before first reset)
            now_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0) if self.session_start_utc is None else self.session_start_utc
            return self._create_empty_state(now_utc)

        current_ts_on_timeline = self._agent_timeline_utc[self._current_agent_time_idx]
        if self.current_timestamp_utc != current_ts_on_timeline:
            self.logger.warning(
                f"Internal timestamp mismatch: self.current_timestamp_utc {self.current_timestamp_utc} vs timeline {current_ts_on_timeline}. Syncing.")
            self.current_timestamp_utc = current_ts_on_timeline

        return self._precomputed_states.get(self.current_timestamp_utc)

    def get_state_at_time(self, timestamp: datetime, tolerance_seconds: int = 1) -> Optional[Dict[str, Any]]:
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
        elif timestamp.tzinfo != ZoneInfo("UTC"):
            timestamp = timestamp.astimezone(ZoneInfo("UTC"))

        exact_match = self._precomputed_states.get(timestamp)
        if exact_match:
            return exact_match

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
        return self._create_empty_state(timestamp)

    def step(self) -> bool:
        if self.is_done():
            self.logger.info("Step called, but simulation is already done.")
            return False

        self._current_agent_time_idx += 1
        if self._current_agent_time_idx < len(self._agent_timeline_utc):
            self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
            return True
        else:
            self._current_agent_time_idx = len(self._agent_timeline_utc) - 1
            if self._agent_timeline_utc:  # Check if timeline is not empty
                self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
            else:  # Should not happen if initialized correctly
                self.current_timestamp_utc = None
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
            # Create a default timestamp if session_start_utc is None
            default_ts = self.session_start_utc or datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
            return self._create_empty_state(default_ts)

        buffer_seconds = self.market_config.get('initial_buffer_seconds', 0)
        min_start_delay_from_session_start = timedelta(seconds=buffer_seconds)
        earliest_possible_agent_start_time = self._agent_timeline_utc[0] + min_start_delay_from_session_start

        start_idx_after_buffer = bisect.bisect_left(self._agent_timeline_utc, earliest_possible_agent_start_time)
        start_idx_after_buffer = min(start_idx_after_buffer, len(self._agent_timeline_utc) - 1)

        if options.get('random_start', False):
            if start_idx_after_buffer < len(self._agent_timeline_utc) - 1:
                if hasattr(self.env, 'np_random') and self.env.np_random is not None:  # gym.Env provides np_random
                    self._current_agent_time_idx = self.env.np_random.integers(start_idx_after_buffer, len(self._agent_timeline_utc))
                elif hasattr(self, 'np_random') and self.np_random is not None:  # if np_random is directly on MarketSimulator
                    self._current_agent_time_idx = self.np_random.integers(start_idx_after_buffer, len(self._agent_timeline_utc))
                else:
                    self._current_agent_time_idx = np.random.randint(start_idx_after_buffer, len(self._agent_timeline_utc))
            else:
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
            self._current_agent_time_idx = 0
            self.current_timestamp_utc = self._agent_timeline_utc[0]
            return self.get_current_market_state()

    def get_symbol_info(self):
        return {
            "symbol": self.symbol,
            "total_shares_outstanding": self.market_config.get('total_shares_outstanding', 100_000_000),
        }

    def close(self):
        self.logger.info("Closing MarketSimulator")
        self._precomputed_states.clear()
        self._agent_timeline_utc.clear()
        if hasattr(self, 'raw_trades_df'): del self.raw_trades_df
        if hasattr(self, 'raw_quotes_df'): del self.raw_quotes_df
        if hasattr(self, 'raw_1s_bars_provider_df'): del self.raw_1s_bars_provider_df
        if hasattr(self, 'historical_1d_bars_df'): del self.historical_1d_bars_df
        self.logger.info("MarketSimulator closed and resources cleared.")