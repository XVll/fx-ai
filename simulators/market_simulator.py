# market_simulator.py
import logging
from collections import deque
from dataclasses import dataclass  # Added
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import bisect

# Assuming these are external and exist
from config.config import MarketConfig, FeatureConfig  # User specified this path
from data.data_manager import DataManager  # User specified this path

# Moved to top as requested
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
    - Robust lookback data population, even on random resets and across session boundaries,
      considering feature engineering sequence length requirements.
    """

    def __init__(self,
                 symbol: str,
                 data_manager: DataManager,
                 market_config: MarketConfig,  # Main market/simulator config
                 feature_config: FeatureConfig,  # Configuration for feature sequence lengths
                 mode: str = "backtesting",
                 start_time: Optional[str | datetime] = None,
                 end_time: Optional[str | datetime] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the market simulator.

        Args:
            symbol: Trading symbol
            data_manager: DataManager instance for retrieving data
            market_config: MarketConfig instance or dictionary for simulator settings
            feature_config: Optional FeatureConfig instance defining sequence lengths for features.
                            If provided, simulator ensures lookback windows accommodate these lengths.
            mode: 'backtesting' or 'live'
            start_time: Start time for historical data (backtesting)
            end_time: End time for historical data (backtesting)
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.data_manager = data_manager

        # Process market_config to a dictionary
        if isinstance(market_config, dict):
            self.market_config_dict = market_config
        elif hasattr(market_config, 'to_dict') and callable(market_config.to_dict):
            self.market_config_dict = market_config.to_dict()
        else:
            self.logger.warning("market_config is not a dict and has no to_dict() method. Using empty config for market settings.")
            self.market_config_dict = {}

        self.feature_config = feature_config  # Store the provided feature_config
        self.mode = mode

        self.start_time_utc = self._parse_datetime(start_time)
        self.end_time_utc = self._parse_datetime(end_time)

        self._setup_market_hours()  # Uses self.market_config_dict

        self._precomputed_states: Dict[datetime, Dict[str, Any]] = {}
        self._all_timestamps: List[datetime] = []  # Agent-accessible timeline

        # Rolling window sizes for data context, determined by market_config and feature_config
        # Initialize from market_config first (these are max lengths for deques in state)

        # Ensure rolling windows are large enough for feature sequence lengths

        self.rolling_1s_window_size = feature_config.hf_seq_len
        self.rolling_1m_window_size = feature_config.mf_seq_len
        self.rolling_5m_window_size = feature_config.lf_seq_len
        self.logger.info(f"Rolling window sizes determined considering FeatureConfig: "
                         f"1s raw data points (deque maxlen): {self.rolling_1s_window_size}, "
                         f"1m completed bars (deque maxlen): {self.rolling_1m_window_size}, "
                         f"5m completed bars (deque maxlen): {self.rolling_5m_window_size}")

        self._effective_max_lookback_seconds = self._calculate_effective_max_lookback_seconds()
        self.logger.info(f"Effective maximum data lookback seconds needed for priming data loading: {self._effective_max_lookback_seconds}")

        self.current_timestamp_utc: Optional[datetime] = None
        self._current_time_idx: int = 0

        self._initialize()

    def _calculate_effective_max_lookback_seconds(self) -> int:
        """
        Calculates the maximum lookback period (in seconds) required for data windows
        based on the determined rolling window sizes. This is used for priming data loading.
        """
        # self.rolling_Xs_window_size are already set considering both market and feature configs.
        s1_data_lookback_duration = self.rolling_1s_window_size  # This is in seconds (number of 1s data points)
        m1_data_lookback_duration = self.rolling_1m_window_size * 60  # num_1m_bars * 60_sec/min
        m5_data_lookback_duration = self.rolling_5m_window_size * 5 * 60  # num_5m_bars * 5_min/bar * 60_sec/min

        self.logger.debug(f"Data lookback components for priming (seconds) - "
                          f"based on 1s window size: {s1_data_lookback_duration}, "
                          f"based on 1m window size: {m1_data_lookback_duration}, "
                          f"based on 5m window size: {m5_data_lookback_duration}")

        return max(s1_data_lookback_duration, m1_data_lookback_duration, m5_data_lookback_duration, 1)  # Min 1 sec

    def get_symbol_info(self):  # No changes
        return {
            "symbol": self.symbol,
            "total_shares_outstanding": 100_000_000,
        }

    def _setup_market_hours(self):  # Changed to use self.market_config_dict
        """Configure timezone and market hours."""
        _market_hours_cfg_raw = self.market_config_dict.get('market_hours', DEFAULT_MARKET_HOURS)
        self.exchange_timezone_str = _market_hours_cfg_raw.get('TIMEZONE', DEFAULT_MARKET_HOURS['TIMEZONE'])
        self.exchange_timezone = ZoneInfo(self.exchange_timezone_str)

        self.market_hours = {}
        for key, default_val_str in DEFAULT_MARKET_HOURS.items():
            if key.endswith(("_START", "_END")):
                time_str = _market_hours_cfg_raw.get(key, default_val_str)
                self.market_hours[key] = datetime.strptime(time_str, "%H:%M:%S").time()

    def _parse_datetime(self, dt_input: Optional[str | datetime]) -> Optional[datetime]:  # No changes
        """Convert input to UTC datetime with robust parsing."""
        if dt_input is None:
            return None

        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None or dt_input.tzinfo.utcoffset(dt_input) is None:  # Naive
                self.logger.debug(f"Received naive datetime: {dt_input}. Assuming UTC.")
                return dt_input.replace(tzinfo=ZoneInfo("UTC"))
            return dt_input.astimezone(ZoneInfo("UTC"))

        if isinstance(dt_input, str):
            try:
                dt_obj = datetime.fromisoformat(dt_input.replace(" Z", "+00:00"))
                if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:  # Naive
                    return dt_obj.replace(tzinfo=ZoneInfo("UTC"))
                return dt_obj.astimezone(ZoneInfo("UTC"))
            except ValueError:
                self.logger.debug(f"Could not parse '{dt_input}' as ISO format, trying pandas.")
                pass
            try:
                dt_obj = pd.to_datetime(dt_input)
                if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
                    # pd.to_datetime might return naive if no tz info in string
                    # default to UTC if still naive after pandas parsing
                    dt_obj = dt_obj.replace(tzinfo=ZoneInfo("UTC"))
                    self.logger.debug(f"Pandas parsed {dt_input} as naive, localized to UTC: {dt_obj}")
                    return dt_obj
                return dt_obj.astimezone(ZoneInfo("UTC"))
            except Exception as e:
                self.logger.error(f"Failed to parse datetime string: {dt_input}, error: {e}")
                return None

        self.logger.error(f"Invalid datetime input type: {type(dt_input)}")
        return None

    def _initialize(self):  # No changes
        """Initialize the simulator based on mode."""
        self.logger.info(f"Initializing MarketSimulator for {self.symbol} in {self.mode} mode")
        self._load_initial_data()

        if self.mode == 'backtesting':
            self._initialize_backtesting()
        elif self.mode == 'live':
            self._initialize_live()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _load_initial_data(self):  # No functional changes, uses _effective_max_lookback_seconds
        """Load required data from data manager, including padding for lookbacks."""
        if not self.data_manager:
            self.logger.error("No data_manager provided")
            return

        effective_load_start_time = self.start_time_utc
        if self.start_time_utc:
            padding_duration = timedelta(seconds=self._effective_max_lookback_seconds)
            padding_duration += timedelta(days=1)  # Extra day buffer for cross-session robustness
            effective_load_start_time = self.start_time_utc - padding_duration
            self.logger.info(
                f"Original start_time_utc: {self.start_time_utc}. Adjusted data loading start time for lookback priming: {effective_load_start_time}")
        else:
            self.logger.info("No specific start_time_utc provided for initial load; DataManager will use its defaults or full range.")

        try:
            self.logger.info(f"Loading data for {self.symbol} from {effective_load_start_time} to {self.end_time_utc}")
            data_result = self.data_manager.load_data(
                symbols=[self.symbol],
                start_time=effective_load_start_time,
                end_time=self.end_time_utc,
                data_types=["trades", "quotes", "bars_1d", "status"]
            )

            if self.symbol not in data_result:
                self.logger.error(f"No data loaded for symbol {self.symbol}")
                self.raw_trades = pd.DataFrame(columns=['price', 'size']).set_index(pd.DatetimeIndex([]))
                self.raw_quotes = pd.DataFrame(columns=['price', 'size', 'side']).set_index(pd.DatetimeIndex([]))
                self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([]))
                self.raw_1m_bars = pd.DataFrame().set_index(pd.DatetimeIndex([]))
                self.raw_5m_bars = pd.DataFrame().set_index(pd.DatetimeIndex([]))
                self.raw_1d_bars = pd.DataFrame().set_index(pd.DatetimeIndex([]))
                self.raw_status = pd.DataFrame().set_index(pd.DatetimeIndex([]))
                return

            symbol_data = data_result[self.symbol]
            self.raw_trades = symbol_data.get("trades", pd.DataFrame(columns=['price', 'size']).set_index(pd.DatetimeIndex([])))
            self.raw_quotes = symbol_data.get("quotes", pd.DataFrame(columns=['price', 'size', 'side']).set_index(pd.DatetimeIndex([])))
            self.raw_1s_bars = symbol_data.get("bars_1s",
                                               pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([])))
            self.raw_1m_bars = symbol_data.get("bars_1m", pd.DataFrame().set_index(pd.DatetimeIndex([])))
            self.raw_5m_bars = symbol_data.get("bars_5m", pd.DataFrame().set_index(pd.DatetimeIndex([])))
            self.raw_1d_bars = symbol_data.get("bars_1d", pd.DataFrame().set_index(pd.DatetimeIndex([])))
            self.raw_status = symbol_data.get("status", pd.DataFrame().set_index(pd.DatetimeIndex([])))

            self.logger.info(f"Loaded data: {len(self.raw_1s_bars)} 1s bars (potentially including padding), "
                             f"{len(self.raw_trades)} trades, {len(self.raw_quotes)} quotes.")

            if self.raw_1s_bars.empty and not self.raw_trades.empty:
                self.logger.info("No 1s bars found. Generating from trades...")
                self._generate_1s_bars_from_trades()

            for df_name in ['raw_trades', 'raw_quotes', 'raw_1s_bars', 'raw_1m_bars', 'raw_5m_bars', 'raw_1d_bars', 'raw_status']:
                df = getattr(self, df_name)
                if isinstance(df.index, pd.DatetimeIndex) and (df.index.tzinfo is None or df.index.tzinfo.utcoffset(None) is None):  # Check if naive
                    try:
                        # Ensure it's not an empty index before trying to localize
                        if not df.index.empty:
                            setattr(self, df_name, df.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward'))
                            self.logger.debug(f"Localized naive index of {df_name} to UTC.")
                        # If empty, create an empty UTC DatetimeIndex
                        elif isinstance(getattr(self, df_name), pd.DataFrame):
                            setattr(self, df_name, df.set_index(pd.DatetimeIndex([], tz='UTC')))

                    except Exception as e:
                        self.logger.warning(f"Could not localize index of {df_name} to UTC: {e}. Index type: {type(df.index)}")

            self._load_extended_daily_data()

        except Exception as e:
            self.logger.error(f"Error loading data: {e}", exc_info=True)
            self.raw_trades = pd.DataFrame(columns=['price', 'size']).set_index(pd.DatetimeIndex([], tz='UTC'))
            self.raw_quotes = pd.DataFrame(columns=['price', 'size', 'side']).set_index(pd.DatetimeIndex([], tz='UTC'))
            self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([], tz='UTC'))
            self.raw_1m_bars = pd.DataFrame().set_index(pd.DatetimeIndex([], tz='UTC'))
            self.raw_5m_bars = pd.DataFrame().set_index(pd.DatetimeIndex([], tz='UTC'))
            self.raw_1d_bars = pd.DataFrame().set_index(pd.DatetimeIndex([], tz='UTC'))
            self.raw_status = pd.DataFrame().set_index(pd.DatetimeIndex([], tz='UTC'))

    def _generate_1s_bars_from_trades(self):  # No functional changes
        """Generate 1-second OHLCV bars from trade data."""
        if self.raw_trades.empty:
            self.logger.warning("Cannot generate 1s bars: No trade data available")
            self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([], tz='UTC'))
            return

        self.logger.info("Generating 1-second bars from trade data...")
        try:
            required_cols = ['price', 'size']
            if not all(col in self.raw_trades.columns for col in required_cols):
                missing = [col for col in required_cols if col not in self.raw_trades.columns]
                self.logger.error(f"Trade data missing required columns for 1s bar generation: {missing}")
                self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([], tz='UTC'))
                return

            if not isinstance(self.raw_trades.index, pd.DatetimeIndex):
                self.logger.error("Trade data index is not a DatetimeIndex. Cannot resample.")
                self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([], tz='UTC'))
                return

            # Ensure trade data index is UTC before resampling
            if self.raw_trades.index.tzinfo is None:
                self.raw_trades.index = self.raw_trades.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')

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
            resampled.dropna(subset=['open'], inplace=True)
            if not resampled.empty:
                resampled['vwap'] = resampled['value_sum'] / resampled['volume']
                resampled.drop(columns=['value_sum'], inplace=True)
                resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
                resampled.ffill({'vwap'},inplace=True)
                resampled.bfill({'vwap'},inplace=True)

            self.raw_1s_bars = resampled
            if self.raw_1s_bars.index.tzinfo is None and not self.raw_1s_bars.empty:  # Resample might lose tz
                self.raw_1s_bars.index = self.raw_1s_bars.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')

            self.logger.info(f"Generated {len(self.raw_1s_bars)} 1-second bars")

        except Exception as e:
            self.logger.error(f"Error generating 1s bars: {e}", exc_info=True)
            self.raw_1s_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).set_index(pd.DatetimeIndex([], tz='UTC'))

    def _load_extended_daily_data(self):  # No functional changes
        """Load extended daily bar data for support/resistance, if not already sufficiently covered."""
        if self.start_time_utc and (self.raw_1d_bars.empty or len(self.raw_1d_bars) < 200):
            extended_start = self.start_time_utc - timedelta(days=500)
            self.logger.info(f"Attempting to load more extensive daily data up to {self.start_time_utc}.")
            try:
                extended_daily_df = self.data_manager.get_bars(
                    symbol=self.symbol,
                    timeframe="1d",
                    start_time=extended_start,
                    end_time=self.start_time_utc
                )
                if not extended_daily_df.empty:
                    if extended_daily_df.index.tzinfo is None:  # Ensure UTC
                        extended_daily_df.index = extended_daily_df.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')

                    current_1d_bars = self.raw_1d_bars
                    if current_1d_bars.index.tzinfo is None and not current_1d_bars.empty:
                        current_1d_bars.index = current_1d_bars.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')

                    self.raw_1d_bars = pd.concat([extended_daily_df, current_1d_bars]).drop_duplicates(keep='last').sort_index()
                    self.logger.info(f"Loaded/updated extended daily bars. Total: {len(self.raw_1d_bars)}")
            except AttributeError:
                self.logger.warning(f"data_manager does not have get_bars method. Cannot load extended daily data.")
            except Exception as e:
                self.logger.warning(f"Failed to load extended daily data: {e}")

    def _initialize_backtesting(self):  # No functional changes
        """Initialize for backtesting mode by pre-computing all states with proper lookback priming."""
        self.logger.info("Initializing backtesting mode")

        if self.raw_1s_bars is None or self.raw_1s_bars.empty:
            self.logger.error("No 1s bar data available for backtesting initialization.")
            return

        all_data_timestamps = self.raw_1s_bars.index.unique().tolist()
        all_data_timestamps.sort()

        if not all_data_timestamps:
            self.logger.error("No timestamps found in 1s bar data after loading.")
            return

        agent_timestamps = all_data_timestamps
        if self.start_time_utc:
            agent_timestamps = [ts for ts in agent_timestamps if ts >= self.start_time_utc]
        if self.end_time_utc:
            agent_timestamps = [ts for ts in agent_timestamps if ts <= self.end_time_utc]

        if not agent_timestamps:
            self.logger.error(f"No agent-accessible timestamps found between {self.start_time_utc} and {self.end_time_utc} from the loaded data.")
            return

        self._all_timestamps = agent_timestamps

        self.logger.info(
            f"Starting precomputation. Iterating over {len(all_data_timestamps)} total timestamps. Storing states for {len(self._all_timestamps)} agent timestamps.")
        self._precompute_all_states(all_data_timestamps, set(self._all_timestamps))

        self.reset()

        if self.current_timestamp_utc:
            self.logger.info(f"Backtesting initialized. Agent will start at: {self.current_timestamp_utc} (index {self._current_time_idx}).")
        else:
            self.logger.warning("Backtesting initialized, but agent start timestamp could not be determined (likely no valid agent timestamps).")

    def _initialize_live(self):  # No functional changes
        """Initialize for live trading mode."""
        self.logger.info("Initializing live trading mode")

        historical_agent_timestamps = []
        if self.raw_1s_bars is not None and not self.raw_1s_bars.empty:
            all_loaded_data_ts = self.raw_1s_bars.index.unique().tolist()
            all_loaded_data_ts.sort()

            historical_agent_timestamps = all_loaded_data_ts
            if self.start_time_utc:
                historical_agent_timestamps = [ts for ts in historical_agent_timestamps if ts >= self.start_time_utc]
            if self.end_time_utc:
                historical_agent_timestamps = [ts for ts in historical_agent_timestamps if ts <= self.end_time_utc]

            if historical_agent_timestamps:
                self.logger.info(
                    f"Live mode: Pre-computing for {len(historical_agent_timestamps)} historical agent timestamps, using {len(all_loaded_data_ts)} total loaded timestamps for priming.")
                self._precompute_all_states(all_loaded_data_ts, set(historical_agent_timestamps))

        self._all_timestamps = historical_agent_timestamps

        self.current_timestamp_utc = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)

        if self.current_timestamp_utc in self._precomputed_states:
            try:
                self._current_time_idx = self._all_timestamps.index(self.current_timestamp_utc)
            except ValueError:  # Should not happen if it's in _precomputed_states and _all_timestamps is consistent
                self.logger.error(f"Timestamp {self.current_timestamp_utc} in precomputed_states but not in _all_timestamps. Rebuilding index.")
                # Fallback: position at the end or try to find again
                if self._all_timestamps:
                    self._current_time_idx = len(self._all_timestamps) - 1
                else:
                    self._all_timestamps = [self.current_timestamp_utc]  # If _all_timestamps was empty
                    self._current_time_idx = 0


        else:
            if self.current_timestamp_utc not in self._all_timestamps:
                self._all_timestamps.append(self.current_timestamp_utc)
                self._all_timestamps.sort()

            if self.current_timestamp_utc not in self._precomputed_states:
                self.logger.warning(f"Live mode: Current time {self.current_timestamp_utc} not in precomputed states. Creating default state.")
                self._precomputed_states[self.current_timestamp_utc] = self._calculate_default_state(self.current_timestamp_utc)

            try:
                self._current_time_idx = self._all_timestamps.index(self.current_timestamp_utc)
            except ValueError:
                self.logger.error(f"Live mode: Could not find current time {self.current_timestamp_utc} in _all_timestamps even after attempting to add it.")
                self._current_time_idx = len(self._all_timestamps) - 1 if self._all_timestamps else 0

        if self._all_timestamps and self._current_time_idx < len(self._all_timestamps):
            self.logger.info(f"Live mode initialized. Current agent time: {self._all_timestamps[self._current_time_idx]}")
        elif not self._all_timestamps:
            self.logger.warning("Live mode initialized, but no timestamps available.")
        else:
            self.logger.warning(
                f"Live mode initialized, but current time index {self._current_time_idx} seems problematic for timeline length {len(self._all_timestamps)}.")

    def _precompute_all_states(self, timestamps_to_iterate: List[datetime], store_state_for_ts: Set[datetime]):  # No functional changes
        """
        Pre-compute market states. Iterates `timestamps_to_iterate` (includes padding)
        to build rolling windows correctly. Stores states in `_precomputed_states`
        only for timestamps present in `store_state_for_ts` (agent's actual timeline).
        """
        self.logger.info(f"Pre-computing states. Iterating {len(timestamps_to_iterate)} timestamps, "
                         f"will store states for {len(store_state_for_ts)} target timestamps.")

        prev_day_local = None
        day_high = None
        day_low = None

        rolling_1s_data = deque(maxlen=self.rolling_1s_window_size if self.rolling_1s_window_size > 0 else None)
        completed_1m_bars = deque(maxlen=self.rolling_1m_window_size if self.rolling_1m_window_size > 0 else None)
        completed_5m_bars = deque(maxlen=self.rolling_5m_window_size if self.rolling_5m_window_size > 0 else None)

        current_1m_bar_forming: Optional[Dict[str, Any]] = None
        current_5m_bar_forming: Optional[Dict[str, Any]] = None

        best_bid_price: Optional[float] = None
        best_ask_price: Optional[float] = None
        best_bid_size: int = 0
        best_ask_size: int = 0
        current_price: Optional[float] = None

        for i, timestamp in enumerate(timestamps_to_iterate):
            local_datetime = timestamp.astimezone(self.exchange_timezone)
            local_date = local_datetime.date()

            if prev_day_local != local_date:
                self.logger.debug(f"Processing new day: {local_date} (UTC: {timestamp.date()})")
                prev_day_local = local_date
                day_high = None
                day_low = None
                current_1m_bar_forming = None
                current_5m_bar_forming = None

            current_1s_bar_data = self._get_raw_bar_at(timestamp)

            if current_1s_bar_data and current_1s_bar_data.get('close') is not None:
                try:
                    current_price = float(current_1s_bar_data['close'])
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert current_1s_bar_data['close'] to float: {current_1s_bar_data['close']} at {timestamp}")

            event_data_for_rolling_window = {
                'timestamp': timestamp,
                'bar': current_1s_bar_data,
                'trades': self._get_trades_in_second(timestamp),
                'quotes': self._get_quotes_in_second(timestamp)
            }
            rolling_1s_data.append(event_data_for_rolling_window)

            if current_1s_bar_data is not None:
                if day_high is None or current_1s_bar_data['high'] > day_high:
                    day_high = current_1s_bar_data['high']
                if day_low is None or current_1s_bar_data['low'] < day_low:
                    day_low = current_1s_bar_data['low']

                current_1m_bar_forming, current_5m_bar_forming = self._update_aggregation_bars(
                    current_1s_bar_data, timestamp,
                    current_1m_bar_forming, current_5m_bar_forming,
                    completed_1m_bars, completed_5m_bars
                )
            current_bbo_bid_px_this_sec: Optional[float] = None
            current_bbo_ask_px_this_sec: Optional[float] = None
            current_bbo_bid_sz_this_sec: int = 0
            current_bbo_ask_sz_this_sec: int = 0

            quotes_this_second = event_data_for_rolling_window.get('quotes', [])
            if quotes_this_second:  # Only if there are quotes for this exact second
                bbo_tuple = self._get_bbo_and_sizes_from_quotes(quotes_this_second)
                current_bbo_bid_px_this_sec = bbo_tuple[0]
                current_bbo_ask_px_this_sec = bbo_tuple[1]
                current_bbo_bid_sz_this_sec = bbo_tuple[2]
                current_bbo_ask_sz_this_sec = bbo_tuple[3]

            if current_bbo_bid_px_this_sec is not None and current_bbo_ask_px_this_sec is not None:
                # Valid BBO found for this second, update LOCF holders
                best_bid_price = current_bbo_bid_px_this_sec
                best_ask_price = current_bbo_ask_px_this_sec
                best_bid_size = current_bbo_bid_sz_this_sec
                best_ask_size = current_bbo_ask_sz_this_sec
            else:
                # No valid BBO from current second's quotes, apply LOCF
                current_bbo_bid_px_this_sec = best_bid_price
                current_bbo_ask_px_this_sec = best_ask_price
                current_bbo_bid_sz_this_sec = best_bid_size
                current_bbo_ask_sz_this_sec = best_ask_size

            if current_bbo_bid_px_this_sec is None and current_bbo_ask_px_this_sec is None and current_price is not None:  # Check if we have a trade-based locf_current_price
                current_bbo_bid_px_this_sec = current_price
                current_bbo_ask_px_this_sec = current_price
                current_bbo_bid_sz_this_sec = 0  # No actual quote depth
                current_bbo_ask_sz_this_sec = 0

            if timestamp in store_state_for_ts:
                market_session = self._determine_market_session(timestamp)
                state = self._calculate_complete_state(
                    timestamp=timestamp,
                    current_1s_bar=current_1s_bar_data,
                    determined_best_bid_price=current_bbo_bid_px_this_sec,
                    determined_best_ask_price=current_bbo_ask_px_this_sec,
                    determined_best_bid_size=current_bbo_bid_sz_this_sec,
                    determined_best_ask_size=current_bbo_ask_sz_this_sec,
                    current_price=current_price,
                    rolling_1s_data=list(rolling_1s_data),
                    day_high=day_high,
                    day_low=day_low,
                    market_session=market_session,
                    current_1m_bar_forming=current_1m_bar_forming,
                    current_5m_bar_forming=current_5m_bar_forming,
                    completed_1m_bars=list(completed_1m_bars),
                    completed_5m_bars=list(completed_5m_bars)
                )
                self._precomputed_states[timestamp] = state

            if (i + 1) % 10000 == 0:
                self.logger.info(f"Pre-computation progress: {i + 1}/{len(timestamps_to_iterate)} timestamps processed. "
                                 f"States stored so far: {len(self._precomputed_states)}")

        self.logger.info(f"Finished pre-computing. Stored {len(self._precomputed_states)} market states "
                         f"for the agent's timeline.")

    def _get_raw_bar_at(self, timestamp: datetime) -> Optional[Dict[str, Any]]:  # No functional changes
        """Get the raw 1s bar at the specified timestamp."""
        if self.raw_1s_bars is None or self.raw_1s_bars.empty:
            return None
        try:
            ts_utc = timestamp  # Assume timestamp is already UTC as per internal logic
            if ts_utc.tzinfo is None:  # Double check
                ts_utc = ts_utc.replace(tzinfo=ZoneInfo("UTC"))

            bar_series = self.raw_1s_bars.loc[ts_utc]
            bar = bar_series.to_dict()
            bar['timestamp'] = ts_utc
            return bar
        except KeyError:
            return None
        except Exception as e:
            self.logger.debug(f"Error fetching raw bar at {timestamp}: {e}")
            return None

    def _get_trades_in_second(self, timestamp: datetime) -> List[Dict[str, Any]]:  # No functional changes
        """Get all trades within the 1-second interval ending at timestamp."""
        if self.raw_trades is None or self.raw_trades.empty:
            return []

        end_time = timestamp
        start_time = end_time - timedelta(seconds=1)

        try:
            trades_slice = self.raw_trades.loc[start_time:end_time]
            trades_slice = trades_slice[trades_slice.index > start_time]

            trade_records = []
            for idx, row in trades_slice.iterrows():
                trade_dict = row.to_dict()
                trade_dict['timestamp'] = idx
                trade_records.append(trade_dict)
            return trade_records
        except Exception as e:
            self.logger.debug(f"Error slicing trades for timestamp {timestamp}: {e}")
            return []

    def _get_quotes_in_second(self, timestamp: datetime) -> List[Dict[str, Any]]:  # No functional changes
        """Get all quotes within the 1-second interval ending at timestamp."""
        if self.raw_quotes is None or self.raw_quotes.empty:
            return []

        end_time = timestamp
        start_time = end_time - timedelta(seconds=1)
        try:
            quotes_slice = self.raw_quotes.loc[start_time:end_time]
            quotes_slice = quotes_slice[quotes_slice.index > start_time]

            quote_records = []
            for idx, row in quotes_slice.iterrows():
                quote_dict = row.to_dict()
                quote_dict['timestamp'] = idx
                quote_records.append(quote_dict)
            return quote_records
        except Exception as e:
            self.logger.debug(f"Error slicing quotes for timestamp {timestamp}: {e}")
            return []

    def _update_aggregation_bars(self,  # No functional changes
                                 current_1s_bar_data: Dict[str, Any],
                                 timestamp: datetime,
                                 current_1m_bar_forming: Optional[Dict[str, Any]],
                                 current_5m_bar_forming: Optional[Dict[str, Any]],
                                 completed_1m_bars: deque,
                                 completed_5m_bars: deque) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Update 1m and 5m aggregation bars based on the current 1s bar."""
        minute_start_ts = timestamp.replace(second=0, microsecond=0)

        if current_1m_bar_forming is None or current_1m_bar_forming['timestamp_start'] != minute_start_ts:
            if current_1m_bar_forming is not None:
                completed_1m_bars.append(current_1m_bar_forming.copy())

            current_1m_bar_forming = {
                'timestamp_start': minute_start_ts,
                'open': current_1s_bar_data['open'],
                'high': current_1s_bar_data['high'],
                'low': current_1s_bar_data['low'],
                'close': current_1s_bar_data['close'],
                'volume': current_1s_bar_data.get('volume', 0.0)
            }
        else:
            current_1m_bar_forming['high'] = max(current_1m_bar_forming['high'], current_1s_bar_data['high'])
            current_1m_bar_forming['low'] = min(current_1m_bar_forming['low'], current_1s_bar_data['low'])
            current_1m_bar_forming['close'] = current_1s_bar_data['close']
            current_1m_bar_forming['volume'] = current_1m_bar_forming.get('volume', 0.0) + current_1s_bar_data.get('volume', 0.0)

        current_minute_val = timestamp.minute
        five_min_slot = current_minute_val // 5
        five_min_start_ts = timestamp.replace(minute=five_min_slot * 5, second=0, microsecond=0)

        if current_5m_bar_forming is None or current_5m_bar_forming['timestamp_start'] != five_min_start_ts:
            if current_5m_bar_forming is not None:
                completed_5m_bars.append(current_5m_bar_forming.copy())

            current_5m_bar_forming = {
                'timestamp_start': five_min_start_ts,
                'open': current_1s_bar_data['open'],
                'high': current_1s_bar_data['high'],
                'low': current_1s_bar_data['low'],
                'close': current_1s_bar_data['close'],
                'volume': current_1s_bar_data.get('volume', 0.0)
            }
        else:
            current_5m_bar_forming['high'] = max(current_5m_bar_forming['high'], current_1s_bar_data['high'])
            current_5m_bar_forming['low'] = min(current_5m_bar_forming['low'], current_1s_bar_data['low'])
            current_5m_bar_forming['close'] = current_1s_bar_data['close']
            current_5m_bar_forming['volume'] = current_5m_bar_forming.get('volume', 0.0) + current_1s_bar_data.get('volume', 0.0)

        return current_1m_bar_forming, current_5m_bar_forming

    def _determine_market_session(self, timestamp: datetime) -> str:  # No changes
        """Determine the market session based on timestamp."""
        local_time = timestamp.astimezone(self.exchange_timezone).time()

        if self.market_hours["PREMARKET_START"] <= local_time <= self.market_hours["PREMARKET_END"]:
            return "PREMARKET"
        elif self.market_hours["REGULAR_START"] <= local_time <= self.market_hours["REGULAR_END"]:
            return "REGULAR"
        elif self.market_hours["POSTMARKET_START"] <= local_time <= self.market_hours["POSTMARKET_END"]:
            return "POSTMARKET"
        return "CLOSED"

    def _calculate_complete_state(self, timestamp: datetime, current_1s_bar: Optional[Dict[str, Any]],  # No functional changes
                                  determined_best_bid_price: Optional[float],
                                  determined_best_ask_price: Optional[float],
                                  determined_best_bid_size: int,
                                  determined_best_ask_size: int,
                                  current_price: Optional[float],
                                  rolling_1s_data: List[Dict[str, Any]],
                                  day_high: Optional[float], day_low: Optional[float], market_session: str,
                                  current_1m_bar_forming: Optional[Dict[str, Any]],
                                  current_5m_bar_forming: Optional[Dict[str, Any]],
                                  completed_1m_bars: List[Dict[str, Any]],
                                  completed_5m_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the complete market state at the given timestamp.
        """

        state = {
            'timestamp_utc': timestamp,
            'current_market_session': market_session,
            'current_time': timestamp,
            'current_price': current_price,
            'best_bid_price': determined_best_bid_price,
            'best_ask_price': determined_best_ask_price,
            'best_bid_size': determined_best_bid_size,
            'best_ask_size': determined_best_ask_size,

            'intraday_high': day_high,
            'intraday_low': day_low,

            'current_1s_bar': current_1s_bar,
            'current_1m_bar_forming': current_1m_bar_forming,
            'current_5m_bar_forming': current_5m_bar_forming,

            'rolling_1s_data_window': rolling_1s_data,
            'completed_1m_bars_window': completed_1m_bars,
            'completed_5m_bars_window': completed_5m_bars,

            'historical_1d_bars': self.raw_1d_bars,
        }
        return state

    def _calculate_default_state(self, timestamp: datetime) -> Dict[str, Any]:  # No changes
        """Create a default state when no data is available or for placeholder."""
        return {
            'timestamp_utc': timestamp,
            'current_market_session': self._determine_market_session(timestamp),
            'current_time': timestamp,
            'current_price': None,
            'best_bid_price': None,
            'best_ask_price': None,
            'best_bid_size': 0,
            'best_ask_size': 0,
            'intraday_high': None,
            'intraday_low': None,
            'current_1s_bar': None,
            'current_1m_bar_forming': None,
            'current_5m_bar_forming': None,
            'rolling_1s_data_window': [],
            'completed_1m_bars_window': [],
            'completed_5m_bars_window': [],
            'historical_1d_bars': self.raw_1d_bars if hasattr(self, 'raw_1d_bars') else None,
        }

    def get_state_at_time(self, timestamp: datetime, tolerance_seconds: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get the complete market state at a given timestamp.
        If exact timestamp is not found, attempts to find the closest *past or current* state within tolerance.
        """
        state = self._precomputed_states.get(timestamp)
        if state is not None:
            return state  # Exact match found

        if not self._all_timestamps:
            self.logger.warning(f"State for timestamp {timestamp} not found, and _all_timestamps is empty.")
            return self._calculate_default_state(timestamp)

        # Find the insertion point for the requested timestamp.
        # bisect_left returns an index idx such that all e in a[:idx] have e < x,
        # and all e in a[idx:] have e >= x.
        idx = bisect.bisect_left(self._all_timestamps, timestamp)

        # Candidate 1: The timestamp at self._all_timestamps[idx-1]
        # This is the largest timestamp in _all_timestamps that is strictly less than 'timestamp'
        # (or equal if timestamp itself was found and idx was shifted, but we already checked for exact match).
        candidate_ts = None
        if idx > 0:  # Means there's at least one element before the insertion point
            ts_before = self._all_timestamps[idx - 1]
            # Ensure ts_before is indeed <= timestamp (it should be by bisect_left logic if no exact match)
            # and within tolerance.
            if ts_before <= timestamp and (timestamp - ts_before) <= timedelta(seconds=tolerance_seconds):
                candidate_ts = ts_before
                self.logger.debug(
                    f"Exact state for {timestamp} not found. Using closest past/present state from {candidate_ts} (diff: {timestamp - candidate_ts}).")
                return self._precomputed_states.get(candidate_ts)

        # If no suitable past/present timestamp was found within tolerance
        self.logger.warning(f"State for timestamp {timestamp} not found in precomputed states. "
                            f"No suitable past/present timestamp found within {tolerance_seconds}s tolerance. "
                            f"The closest preceding timestamp considered was {self._all_timestamps[idx - 1] if idx > 0 else 'N/A'}.")

        # Log details if timestamp is outside the agent's timeline (for context)
        if self._all_timestamps:  # Check if not empty
            if timestamp < self._all_timestamps[0] or timestamp > self._all_timestamps[-1]:
                self.logger.warning(f"Requested timestamp {timestamp} is outside the agent's timeline "
                                    f"[{self._all_timestamps[0]}, {self._all_timestamps[-1]}].")
        return self._calculate_default_state(timestamp)  # Fallback if no suitable data

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:  # No functional changes
        """Get the market state at the agent's current timeline position."""
        if not self._all_timestamps:
            self.logger.warning("No agent timeline (_all_timestamps) available to get current market state.")
            return self._calculate_default_state(datetime.now(ZoneInfo("UTC")))

        if self._current_time_idx >= len(self._all_timestamps) or self._current_time_idx < 0:
            self.logger.warning(f"Current time index {self._current_time_idx} is out of bounds for "
                                f"agent timeline (len: {len(self._all_timestamps)}).")
            if self._all_timestamps:
                actual_idx = max(0, min(self._current_time_idx, len(self._all_timestamps) - 1))
                last_valid_ts = self._all_timestamps[actual_idx]
                self.logger.warning(f"Falling back to state at index {actual_idx} ({last_valid_ts}).")
                return self.get_state_at_time(last_valid_ts)
            return self._calculate_default_state(datetime.now(ZoneInfo("UTC")))

        current_ts = self._all_timestamps[self._current_time_idx]
        if self.current_timestamp_utc != current_ts:
            self.logger.warning(f"Mismatch: self.current_timestamp_utc ({self.current_timestamp_utc}) "
                                f"!= _all_timestamps[idx] ({current_ts}). Forcing consistency.")
            self.current_timestamp_utc = current_ts

        return self.get_state_at_time(current_ts)

    def step(self) -> bool:  # No functional changes
        """Advance the agent's timeline position by one step."""
        if self.is_done():
            self.logger.info("Step called but simulation is already done.")
            return False

        self._current_time_idx += 1

        if self._current_time_idx < len(self._all_timestamps):
            self.current_timestamp_utc = self._all_timestamps[self._current_time_idx]
            return True
        else:
            self.logger.info("Reached the end of available agent timeline data.")
            if self._all_timestamps:
                self.current_timestamp_utc = self._all_timestamps[-1]
            return False

    def is_done(self) -> bool:  # No functional changes
        """Check if the simulation is done."""
        if not self._all_timestamps:
            return True

        if self._current_time_idx >= len(self._all_timestamps) - 1:
            return True
        return False

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:  # Changed to use self.market_config_dict
        """
        Reset the simulator. Agent starts at a point respecting `initial_buffer_seconds`.
        If `random_start` is true, chooses a random point after this buffer.
        Lookback data is populated by the precomputation strategy.
        """
        options = options or {}
        self.logger.info(f"Resetting MarketSimulator. Options: {options}")
        self.current_timestamp_utc = None

        if not self._all_timestamps:
            self.logger.error("Cannot reset: No simulation timestamps available (_all_timestamps is empty).")
            return self._calculate_default_state(self.start_time_utc or datetime.now(ZoneInfo("UTC")))

        buffer_seconds = self.market_config_dict.get('initial_buffer_seconds', 0)
        min_operational_idx = 0

        if buffer_seconds > 0:
            target_start_timestamp_for_agent = self._all_timestamps[0] + timedelta(seconds=buffer_seconds)
            min_operational_idx = bisect.bisect_left(self._all_timestamps, target_start_timestamp_for_agent)
            min_operational_idx = min(min_operational_idx, len(self._all_timestamps) - 1)
            min_operational_idx = max(0, min_operational_idx)

        if options.get('random_start', False) and len(self._all_timestamps) > 0:
            start_range_for_random = min_operational_idx
            end_range_for_random = len(self._all_timestamps) - 1

            if start_range_for_random <= end_range_for_random:
                self._current_time_idx = np.random.randint(start_range_for_random, end_range_for_random + 1)
                self.logger.info(f"Random reset to index {self._current_time_idx} (respecting buffer).")
            else:
                self._current_time_idx = min_operational_idx
                self.logger.warning(f"Random reset: valid range [{start_range_for_random}, {end_range_for_random}] "
                                    f"is too small or invalid after buffer. Using index {self._current_time_idx}.")
        else:
            self._current_time_idx = min_operational_idx
            self.logger.info(f"Regular reset to index {self._current_time_idx} (respecting buffer).")

        if self._current_time_idx < len(self._all_timestamps):
            self.current_timestamp_utc = self._all_timestamps[self._current_time_idx]
            self.logger.info(f"Simulator reset. Current agent time: {self.current_timestamp_utc}")
        else:
            self.logger.error(f"Reset resulted in invalid _current_time_idx ({self._current_time_idx}) "
                              f"for timeline length {len(self._all_timestamps)}.")
            if self._all_timestamps:
                self._current_time_idx = len(self._all_timestamps) - 1
                self.current_timestamp_utc = self._all_timestamps[self._current_time_idx]
                self.logger.warning(f"Corrected _current_time_idx to last valid: {self._current_time_idx}")
            else:
                self.current_timestamp_utc = None

        return self.get_current_market_state()

    def _get_bbo_and_sizes_from_quotes(self, current_second_quotes: List[Dict[str, Any]]) -> Tuple[
        Optional[float], Optional[float], int, int]:  # No functional changes
        """
        Helper function to calculate Best Bid & Offer (BBO) and their sizes from a list of quotes.
        """
        best_bid_price: Optional[float] = None
        best_ask_price: Optional[float] = None
        best_bid_size: int = 0
        best_ask_size: int = 0

        if not current_second_quotes:
            return best_bid_price, best_ask_price, best_bid_size, best_ask_size

        bids_this_instant = {}
        asks_this_instant = {}

        for quote in current_second_quotes:
            price = quote.get('price')
            size = quote.get('size')
            side = quote.get('side')

            if price is None or size is None or side is None:
                self.logger.debug(f"Skipping malformed quote in BBO calculation: {quote}")
                continue

            try:
                numeric_size = float(size)
            except ValueError:
                self.logger.warning(f"Invalid size '{size}' in quote, skipping: {quote}")
                continue

            if side.lower() == 'bid':
                bids_this_instant[float(price)] = bids_this_instant.get(float(price), 0) + numeric_size
            elif side.lower() == 'ask':
                asks_this_instant[float(price)] = asks_this_instant.get(float(price), 0) + numeric_size

        if bids_this_instant:
            best_bid_price = max(bids_this_instant.keys())
            best_bid_size = int(bids_this_instant[best_bid_price])

        if asks_this_instant:
            best_ask_price = min(asks_this_instant.keys())
            best_ask_size = int(asks_this_instant[best_ask_price])

        return best_bid_price, best_ask_price, best_bid_size, best_ask_size

    def close(self):  # No functional changes
        """Close the simulator and release resources."""
        self.logger.info("Closing MarketSimulator")
        self._precomputed_states.clear()
        self._all_timestamps.clear()
        attrs_to_clear = ['raw_trades', 'raw_quotes', 'raw_1s_bars', 'raw_1m_bars', 'raw_5m_bars', 'raw_1d_bars', 'raw_status']
        for attr_name in attrs_to_clear:
            if hasattr(self, attr_name):
                try:
                    delattr(self, attr_name)
                except AttributeError:  # Should not happen with hasattr, but for safety
                    pass
        self.logger.info("MarketSimulator closed and resources cleared.")
