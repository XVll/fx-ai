# market_simulator.py
import logging
from collections import deque
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import bisect
from typing import Any, Dict, List, Optional, Tuple, Union

from config.config import MarketConfig, FeatureConfig
from data.data_manager import DataManager

# MARKET HOURS CONFIGURATION (Eastern Time)
# These define the valid trading times for the simulator
MARKET_HOURS = {
    "PREMARKET_START": time(4, 0, 0),  # 4:00 AM ET
    "REGULAR_START": time(9, 30, 0),  # 9:30 AM ET
    "REGULAR_END": time(16, 0, 0),  # 4:00 PM ET
    "POSTMARKET_END": time(20, 0, 0),  # 8:00 PM ET
    "TIMEZONE": "America/New_York"  # Standard US Eastern timezone
}


class MarketSimulator:
    """
    Optimized market simulator for intraday trading with proper pre/post market handling.

    Features:
    - Efficient data loading limited to actual trading hours (4:00 AM - 8:00 PM ET)
    - Smart priming for pre-market sessions using previous day's data when needed
    - Consistent window sizes for all data frequencies
    - Simplified timezone handling with clear Eastern Time reference
    - Optimized memory usage for long simulations
    """

    def __init__(
            self,
            symbol: str,
            data_manager: DataManager,
            market_config: MarketConfig,
            feature_config: FeatureConfig,
            mode: str = "backtesting",
            np_random: Optional[np.random.Generator] = None,
            start_time: Optional[Union[str, datetime]] = None,
            end_time: Optional[Union[str, datetime]] = None,
            logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.data_manager = data_manager
        self.market_config = market_config
        self.feature_config = feature_config
        self.mode = mode
        self.np_random = np_random or np.random.default_rng()

        # Use a consistent timezone representation (US Eastern Time for market hours)
        self.market_tz = ZoneInfo(MARKET_HOURS["TIMEZONE"])
        self.utc_tz = ZoneInfo("UTC")

        # Window sizes for data buffering (all consistent with seq_len * 2 for safety)
        self.hf_window_size = max(1, self.feature_config.hf_seq_len * 2)
        self.mf_window_size = max(1, self.feature_config.mf_seq_len * 2)
        self.lf_window_size = max(1, self.feature_config.lf_seq_len * 2)

        # Convert and validate time boundaries
        self.session_start_utc = self._ensure_utc_datetime(start_time)
        self.session_end_utc = self._ensure_utc_datetime(end_time)

        if not self.session_start_utc or not self.session_end_utc:
            self.logger.error("Both start_time and end_time must be provided for backtesting mode")
            if self.mode == "backtesting":
                raise ValueError("Start and end times are required for backtesting mode")

        # Data storage
        self._precomputed_states = {}
        self._agent_timeline_utc = []

        # Current state tracking
        self.current_timestamp_utc = None
        self._current_agent_time_idx = -1

        # Raw data storage for different frequencies
        self.raw_trades_df = pd.DataFrame()
        self.raw_quotes_df = pd.DataFrame()
        self.raw_1m_bars_df = pd.DataFrame()
        self.raw_5m_bars_df = pd.DataFrame()
        self.raw_1d_bars_df = pd.DataFrame()  # Historical daily data

        # Initialize simulator
        self._initialize_simulator()

    def _ensure_utc_datetime(self, dt_input: Optional[Union[str, datetime]]) -> Optional[datetime]:
        """
        Convert input to a UTC datetime object with consistent handling.

        Args:
            dt_input: Input datetime (string, datetime with/without timezone)

        Returns:
            Datetime object in UTC timezone, or None if input is None
        """
        if dt_input is None:
            return None

        # Handle datetime objects
        if isinstance(dt_input, datetime):
            # Ensure proper timezone
            if dt_input.tzinfo is None:
                # Assume market timezone if none provided
                return datetime.combine(dt_input.date(), dt_input.time(), tzinfo=self.market_tz).astimezone(self.utc_tz)
            elif dt_input.tzinfo != self.utc_tz:
                # Convert to UTC if in a different timezone
                return dt_input.astimezone(self.utc_tz)
            return dt_input  # Already in UTC

        # Handle string inputs
        try:
            parsed_dt = pd.Timestamp(dt_input)
            if parsed_dt.tzinfo is None:
                # Assume market timezone for naive datetimes
                localized_dt = datetime.combine(
                    parsed_dt.date(),
                    parsed_dt.time(),
                    tzinfo=self.market_tz
                )
                return localized_dt.astimezone(self.utc_tz)
            return parsed_dt.to_pydatetime().astimezone(self.utc_tz)
        except Exception as e:
            self.logger.error(f"Failed to parse datetime '{dt_input}': {e}")
            return None

    def _initialize_simulator(self):
        """Initialize the market simulator by loading necessary data and preparing timeline."""
        self.logger.info(f"Initializing MarketSimulator for {self.symbol} in {self.mode} mode")

        # Load data efficiently based on feature requirements
        self._load_data_for_simulation()

        if self.mode == "backtesting":
            # Create agent timeline and precompute states for backtesting
            self._build_efficient_timeline()
            self._precompute_timeline_states()
            self.reset()  # Initialize agent time
        elif self.mode == "live":
            # Handle live mode initialization
            self.logger.info("Live mode initialized, will stream data as it arrives")
            self.current_timestamp_utc = datetime.now(self.utc_tz).replace(microsecond=0)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _build_efficient_timeline(self):
        """
        Build an efficient agent timeline considering only market hours.
        This avoids creating datapoints for the entire 24-hour day.
        """
        if not self.session_start_utc or not self.session_end_utc:
            self.logger.error("Session start/end times not defined for timeline generation")
            return

        self.logger.info("Building efficient agent timeline for market hours only")

        # Initialize timeline list
        timeline = []

        # Start with the session start time
        current_time = self.session_start_utc

        # Process each date between start and end
        while current_time <= self.session_end_utc:
            # Convert to market timezone to check if in trading hours
            market_time = current_time.astimezone(self.market_tz)
            market_date = market_time.date()
            market_time_of_day = market_time.time()

            # Check if this time is within trading hours
            if (MARKET_HOURS["PREMARKET_START"] <= market_time_of_day <= MARKET_HOURS["POSTMARKET_END"]):
                # Add to timeline if within trading hours
                timeline.append(current_time)

            # Move to next second
            current_time += timedelta(seconds=1)

            # If we've passed the postmarket end, jump to next day's premarket
            next_market_time = current_time.astimezone(self.market_tz)
            if (next_market_time.date() == market_date and
                    next_market_time.time() > MARKET_HOURS["POSTMARKET_END"]):
                # Calculate next day's premarket start
                next_date = market_date + timedelta(days=1)
                next_premarket_start = datetime.combine(
                    next_date,
                    MARKET_HOURS["PREMARKET_START"],
                    tzinfo=self.market_tz
                )

                # Convert back to UTC and set as current time
                current_time = next_premarket_start.astimezone(self.utc_tz)

                self.logger.info(f"Jumping from {market_date} postmarket to {next_date} premarket")

        self._agent_timeline_utc = timeline

        self.logger.info(
            f"Agent timeline created: {len(timeline)} seconds from "
            f"{timeline[0] if timeline else 'N/A'} to "
            f"{timeline[-1] if timeline else 'N/A'}"
        )

    def _calculate_data_loading_ranges(self) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Calculate optimal data loading ranges for different timeframes with proper lookbacks.

        Returns:
            Dictionary with data types as keys and (start_time, end_time) as values
        """
        if not self.session_start_utc:
            self.logger.error("Cannot calculate loading ranges: session_start_utc is not defined")
            return {}

        # Convert session dates to market timezone for easier calculations
        session_start_market = self.session_start_utc.astimezone(self.market_tz)
        session_end_market = self.session_end_utc.astimezone(self.market_tz)
        session_date = session_start_market.date()

        # Get previous trading day for historical context
        prev_trading_day = session_date - timedelta(
            days=1)  # Simplification - actual implementation should handle weekends

        # Calculate lookback periods with buffers

        # 1. High-frequency data (trades, quotes)
        hf_lookback_seconds = self.hf_window_size * 2  # Double for safety

        # If session starts near premarket open, we need previous day's data
        if session_start_market.time() < MARKET_HOURS["PREMARKET_START"] or (
                session_start_market.time() < time(4, 30, 0) and  # Within 30 min of premarket
                session_start_market.time() >= MARKET_HOURS["PREMARKET_START"]):

            # Need to include previous day's post-market for proper priming
            hf_start_dt = datetime.combine(
                prev_trading_day,
                MARKET_HOURS["REGULAR_END"],  # Start from regular close
                tzinfo=self.market_tz
            )
        else:
            # Normal lookback within the same day
            hf_start_dt = session_start_market - timedelta(seconds=hf_lookback_seconds)

            # Ensure we don't go before premarket start
            premarket_start_dt = datetime.combine(
                session_date,
                MARKET_HOURS["PREMARKET_START"],
                tzinfo=self.market_tz
            )

            if hf_start_dt < premarket_start_dt:
                # If lookback goes before premarket, use previous day's post-market
                hf_start_dt = datetime.combine(
                    prev_trading_day,
                    MARKET_HOURS["REGULAR_END"],
                    tzinfo=self.market_tz
                )

        # 2. Medium-frequency data (1m bars)
        mf_lookback_minutes = self.mf_window_size  # Now using 2x window size

        # Calculate start time similar to HF data
        if session_start_market.time() < MARKET_HOURS["PREMARKET_START"] or (
                session_start_market.time() < time(4, 30, 0) and
                session_start_market.time() >= MARKET_HOURS["PREMARKET_START"]):

            mf_start_dt = datetime.combine(
                prev_trading_day,
                MARKET_HOURS["REGULAR_END"],
                tzinfo=self.market_tz
            )
        else:
            mf_start_dt = session_start_market - timedelta(minutes=mf_lookback_minutes)

            premarket_start_dt = datetime.combine(
                session_date,
                MARKET_HOURS["PREMARKET_START"],
                tzinfo=self.market_tz
            )

            if mf_start_dt < premarket_start_dt:
                mf_start_dt = datetime.combine(
                    prev_trading_day,
                    time(16, 0, 0),  # Use 4:00 PM from previous day
                    tzinfo=self.market_tz
                )

        # 3. Low-frequency data (5m bars)
        lf_lookback_minutes = self.lf_window_size * 5  # Now using 2x window size (× 5 for 5m intervals)

        # Similar logic to MF data
        if session_start_market.time() < MARKET_HOURS["PREMARKET_START"] or (
                session_start_market.time() < time(4, 30, 0)):

            lf_start_dt = datetime.combine(
                prev_trading_day,
                MARKET_HOURS["REGULAR_END"],
                tzinfo=self.market_tz
            )
        else:
            lf_start_dt = session_start_market - timedelta(minutes=lf_lookback_minutes)

            premarket_start_dt = datetime.combine(
                session_date,
                MARKET_HOURS["PREMARKET_START"],
                tzinfo=self.market_tz
            )

            if lf_start_dt < premarket_start_dt:
                lf_start_dt = datetime.combine(
                    prev_trading_day,
                    time(16, 0, 0),
                    tzinfo=self.market_tz
                )

        # 4. Daily bars for historical context
        # Get data for several preceding days for adequate context
        daily_lookback_days = 60  # About 3 months of trading days
        daily_start_dt = datetime.combine(
            session_date - timedelta(days=daily_lookback_days),
            time(0, 0, 0),
            tzinfo=self.market_tz
        )
        daily_end_dt = datetime.combine(
            session_date,
            time(23, 59, 59),
            tzinfo=self.market_tz
        )

        # Convert all datetimes back to UTC for consistent data loading
        return {
            "hf": (hf_start_dt.astimezone(self.utc_tz), session_end_market.astimezone(self.utc_tz)),
            "mf": (mf_start_dt.astimezone(self.utc_tz), session_end_market.astimezone(self.utc_tz)),
            "lf": (lf_start_dt.astimezone(self.utc_tz), session_end_market.astimezone(self.utc_tz)),
            "daily": (daily_start_dt.astimezone(self.utc_tz), daily_end_dt.astimezone(self.utc_tz))
        }

    def _load_data_for_simulation(self):
        """Load only the required data for simulation efficiently."""
        loading_ranges = self._calculate_data_loading_ranges()
        if not loading_ranges:
            self.logger.error("Failed to calculate data loading ranges")
            return

        self.logger.info(f"Data loading ranges for {self.symbol}:")
        for timeframe, (start, end) in loading_ranges.items():
            self.logger.info(f"  {timeframe}: {start} to {end}")

        # 1. Load high-frequency data (trades, quotes)
        hf_start, hf_end = loading_ranges["hf"]
        try:
            hf_data = self.data_manager.load_data(
                symbols=[self.symbol],
                start_time=hf_start,
                end_time=hf_end,
                data_types=["trades", "quotes"]
            )

            symbol_hf_data = hf_data.get(self.symbol, {})

            self.raw_trades_df = self._prepare_dataframe(symbol_hf_data.get("trades"), ['price', 'size'])
            self.raw_quotes_df = self._prepare_dataframe(
                symbol_hf_data.get("quotes"),
                ['bid_price', 'ask_price', 'bid_size', 'ask_size']
            )

            self.logger.info(f"Loaded {len(self.raw_trades_df)} trades and {len(self.raw_quotes_df)} quotes")
        except Exception as e:
            self.logger.error(f"Error loading high-frequency data: {e}")
            self.raw_trades_df = pd.DataFrame()
            self.raw_quotes_df = pd.DataFrame()

        # 2. Load 1-minute bars
        mf_start, mf_end = loading_ranges["mf"]
        try:
            self.raw_1m_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1m",
                start_time=mf_start,
                end_time=mf_end
            )
            self.raw_1m_bars_df = self._prepare_dataframe(
                self.raw_1m_bars_df,
                ['open', 'high', 'low', 'close', 'volume']
            )
            self.logger.info(f"Loaded {len(self.raw_1m_bars_df)} 1-minute bars")
        except Exception as e:
            self.logger.error(f"Error loading 1-minute bars: {e}")
            self.raw_1m_bars_df = pd.DataFrame()

        # 3. Load 5-minute bars
        lf_start, lf_end = loading_ranges["lf"]
        try:
            self.raw_5m_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="5m",
                start_time=lf_start,
                end_time=lf_end
            )
            self.raw_5m_bars_df = self._prepare_dataframe(
                self.raw_5m_bars_df,
                ['open', 'high', 'low', 'close', 'volume']
            )
            self.logger.info(f"Loaded {len(self.raw_5m_bars_df)} 5-minute bars")
        except Exception as e:
            self.logger.error(f"Error loading 5-minute bars: {e}")
            self.raw_5m_bars_df = pd.DataFrame()

        # 4. Load daily bars for historical context
        daily_start, daily_end = loading_ranges["daily"]
        try:
            self.raw_1d_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1d",
                start_time=daily_start,
                end_time=daily_end
            )
            self.raw_1d_bars_df = self._prepare_dataframe(
                self.raw_1d_bars_df,
                ['open', 'high', 'low', 'close', 'volume']
            )
            self.logger.info(f"Loaded {len(self.raw_1d_bars_df)} daily bars")
        except Exception as e:
            self.logger.error(f"Error loading daily bars: {e}")
            self.raw_1d_bars_df = pd.DataFrame()

    def _prepare_dataframe(self, df: Optional[pd.DataFrame], required_cols: List[str]) -> pd.DataFrame:
        """
        Prepare DataFrame for use in simulation with consistent formatting and timezone.

        Args:
            df: Input DataFrame (can be None)
            required_cols: List of columns that should be present

        Returns:
            Properly formatted DataFrame with UTC timezone and required columns
        """
        # Return empty DataFrame with proper structure if input is None or empty
        if df is None or df.empty:
            empty_df = pd.DataFrame(columns=required_cols)
            empty_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
            return empty_df

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("Input DataFrame does not have a DatetimeIndex. Converting.")
            try:
                df.index = pd.to_datetime(df.index)
            except:
                self.logger.error("Failed to convert index to DatetimeIndex. Returning empty DataFrame.")
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

        # Sort by index for consistent access
        return df.sort_index()

    def _get_previous_trading_day_data(self, current_date: date) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
        """
        Get data from the previous trading day for priming purposes.

        Args:
            current_date: The current session date

        Returns:
            Tuple containing (prev_day_close_price, prev_day_postmarket_data)
        """
        # First check if we have daily bars to get previous close
        prev_close = None
        if not self.raw_1d_bars_df.empty:
            try:
                # Find the most recent bar before current date
                prev_bars = self.raw_1d_bars_df[self.raw_1d_bars_df.index.date < current_date]
                if not prev_bars.empty:
                    prev_close = float(prev_bars.iloc[-1]['close'])
                    self.logger.debug(f"Found previous day close: {prev_close}")
            except Exception as e:
                self.logger.error(f"Error getting previous day close from daily bars: {e}")

        # Now try to get the previous day's post-market data
        prev_postmarket_data = None
        try:
            # Calculate the previous day's regular market close time
            prev_day = current_date - timedelta(days=1)
            prev_market_close_dt = datetime.combine(
                prev_day,
                MARKET_HOURS["REGULAR_END"],
                tzinfo=self.market_tz
            ).astimezone(self.utc_tz)

            prev_postmarket_end_dt = datetime.combine(
                prev_day,
                MARKET_HOURS["POSTMARKET_END"],
                tzinfo=self.market_tz
            ).astimezone(self.utc_tz)

            # Try to get trades from previous day's post-market
            if not self.raw_trades_df.empty:
                prev_postmarket_trades = self.raw_trades_df[
                    (self.raw_trades_df.index >= prev_market_close_dt) &
                    (self.raw_trades_df.index <= prev_postmarket_end_dt)
                    ]
                if not prev_postmarket_trades.empty:
                    prev_postmarket_data = prev_postmarket_trades
                    self.logger.debug(f"Found {len(prev_postmarket_data)} previous day post-market trades")
        except Exception as e:
            self.logger.error(f"Error getting previous day post-market data: {e}")

        return prev_close, prev_postmarket_data

    def _precompute_timeline_states(self):
        """
        Precompute market states for every second in the agent's timeline.

        This builds states for all times in the agent's timeline using the most efficient
        data structures and search patterns.
        """
        self.logger.info(f"Precomputing timeline states for {self.symbol}")

        if not self._agent_timeline_utc:
            self.logger.error("Agent timeline is empty. Cannot precompute states.")
            return

        # Initialize LOCF (Last Observation Carried Forward) tracking
        last_price = None
        last_bid = None
        last_ask = None
        last_bid_size = 0
        last_ask_size = 0

        # Initialize day tracking
        current_day_market = None
        intraday_high = None
        intraday_low = None
        prev_day_close = None

        # Initialize rolling window data structures
        rolling_hf_window = deque(maxlen=self.hf_window_size)

        # Convert bars to dictionaries for O(1) lookup by time
        one_min_bars_dict = {}
        five_min_bars_dict = {}

        # Process 1-minute bars
        if not self.raw_1m_bars_df.empty:
            for idx, row in self.raw_1m_bars_df.iterrows():
                one_min_bars_dict[idx] = {
                    'timestamp': idx,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume', 0)
                }

        # Process 5-minute bars
        if not self.raw_5m_bars_df.empty:
            for idx, row in self.raw_5m_bars_df.iterrows():
                five_min_bars_dict[idx] = {
                    'timestamp': idx,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume', 0)
                }

        # Try to initialize basic LOCF values from first data
        self._initialize_locf_values(last_price, last_bid, last_ask, last_bid_size, last_ask_size)

        # Process each second in the agent's timeline
        total_timeline_len = len(self._agent_timeline_utc)
        progress_step = max(1, total_timeline_len // 20)  # Report progress ~20 times

        for idx, current_ts in enumerate(self._agent_timeline_utc):
            # Convert to market time for day boundary detection
            current_market_dt = current_ts.astimezone(self.market_tz)
            current_market_day = current_market_dt.date()

            # Handle day change
            if current_day_market != current_market_day:
                self.logger.debug(f"Processing new day: {current_market_day}")
                current_day_market = current_market_day
                intraday_high = None
                intraday_low = None

                # Get previous day data for priming
                prev_day_close, prev_postmarket_data = self._get_previous_trading_day_data(current_market_day)

                # If we're at the start of pre-market and have no existing data,
                # use previous day's close for initial price estimate
                if (current_market_dt.time() <= time(4, 30, 0) and
                        (last_price is None or last_bid is None or last_ask is None)):

                    if prev_day_close is not None:
                        last_price = prev_day_close
                        # Estimate bid/ask spread as 0.1% of price if not available
                        last_bid = prev_day_close * 0.999
                        last_ask = prev_day_close * 1.001
                        self.logger.debug(f"Using previous day close for initial price: {last_price}")

            # Get data for this second
            current_trades = []
            current_quotes = []

            # Find trades for this second
            if not self.raw_trades_df.empty:
                second_start = current_ts - timedelta(seconds=1)
                trades_slice = self.raw_trades_df[
                    (self.raw_trades_df.index > second_start) &
                    (self.raw_trades_df.index <= current_ts)
                    ]
                current_trades = [row.to_dict() for _, row in trades_slice.iterrows()]

            # Find quotes for this second
            if not self.raw_quotes_df.empty:
                second_start = current_ts - timedelta(seconds=1)
                quotes_slice = self.raw_quotes_df[
                    (self.raw_quotes_df.index > second_start) &
                    (self.raw_quotes_df.index <= current_ts)
                    ]
                current_quotes = [row.to_dict() for _, row in quotes_slice.iterrows()]

            # Build 1-second bar from trades if available
            current_1s_bar = None
            if current_trades:
                current_1s_bar = self._aggregate_trades_to_bar(current_trades, current_ts)

                # Update LOCF price
                if current_1s_bar and 'close' in current_1s_bar and pd.notna(current_1s_bar['close']):
                    last_price = current_1s_bar['close']

                # Update intraday range
                if current_1s_bar:
                    bar_high = current_1s_bar.get('high')
                    bar_low = current_1s_bar.get('low')

                    if pd.notna(bar_high):
                        intraday_high = bar_high if intraday_high is None else max(intraday_high, bar_high)

                    if pd.notna(bar_low):
                        intraday_low = bar_low if intraday_low is None else min(intraday_low, bar_low)

            # Update LOCF values from quotes
            if current_quotes:
                latest_quote = current_quotes[-1]

                if 'bid_price' in latest_quote and pd.notna(latest_quote['bid_price']):
                    last_bid = float(latest_quote['bid_price'])

                if 'ask_price' in latest_quote and pd.notna(latest_quote['ask_price']):
                    last_ask = float(latest_quote['ask_price'])

                if 'bid_size' in latest_quote and pd.notna(latest_quote['bid_size']):
                    last_bid_size = int(latest_quote['bid_size'])

                if 'ask_size' in latest_quote and pd.notna(latest_quote['ask_size']):
                    last_ask_size = int(latest_quote['ask_size'])

            # Update rolling window
            window_entry = {
                'timestamp': current_ts,
                'trades': current_trades,
                'quotes': current_quotes,
                '1s_bar': current_1s_bar
            }
            rolling_hf_window.append(window_entry)

            # Get 1-minute bars window for this timestamp
            minute_bars_window = self._get_bars_window(
                current_ts,
                one_min_bars_dict,
                window_size=self.mf_window_size,
                interval_minutes=1
            )

            # Get 5-minute bars window for this timestamp
            five_min_bars_window = self._get_bars_window(
                current_ts,
                five_min_bars_dict,
                window_size=self.lf_window_size,
                interval_minutes=5
            )

            # Determine market session
            market_session = self._determine_market_session(current_ts)

            # Calculate mid price if needed and we have both bid and ask
            mid_price = None
            if last_bid is not None and last_ask is not None:
                mid_price = (last_bid + last_ask) / 2

            # If no prices available yet, fall back to previous day close
            if last_price is None and mid_price is None and prev_day_close is not None:
                last_price = prev_day_close
                self.logger.debug(f"Using previous day close as fallback price: {last_price}")

            # Store the state for this timestamp
            self._precomputed_states[current_ts] = {
                'timestamp_utc': current_ts,
                'market_session': market_session,
                'current_price': last_price,
                'best_bid_price': last_bid,
                'best_ask_price': last_ask,
                'mid_price': mid_price,
                'best_bid_size': last_bid_size,
                'best_ask_size': last_ask_size,
                'intraday_high': intraday_high,
                'intraday_low': intraday_low,
                'previous_day_close': prev_day_close,
                'current_1s_bar': current_1s_bar,
                'hf_data_window': list(rolling_hf_window),
                '1m_bars_window': minute_bars_window,
                '5m_bars_window': five_min_bars_window
            }

            # Log progress
            if idx % progress_step == 0 or idx == total_timeline_len - 1:
                progress_pct = ((idx + 1) / total_timeline_len) * 100
                self.logger.info(
                    f"Precomputation progress: {idx + 1}/{total_timeline_len} "
                    f"({progress_pct:.1f}%). Current time: {current_ts}"
                )

        self.logger.info(f"Finished precomputing {len(self._precomputed_states)} market states")

    def _initialize_locf_values(self, last_price, last_bid, last_ask, last_bid_size, last_ask_size):
        """Initialize Last Observation Carried Forward values from initial data."""
        try:
            # Try quotes first for best bid/ask
            if not self.raw_quotes_df.empty:
                initial_quotes = self.raw_quotes_df.head(10)
                if not initial_quotes.empty:
                    last_quote = initial_quotes.iloc[-1]

                    if 'bid_price' in last_quote and pd.notna(last_quote['bid_price']):
                        last_bid = float(last_quote['bid_price'])

                    if 'ask_price' in last_quote and pd.notna(last_quote['ask_price']):
                        last_ask = float(last_quote['ask_price'])

                    if 'bid_size' in last_quote and pd.notna(last_quote['bid_size']):
                        last_bid_size = int(last_quote['bid_size'])

                    if 'ask_size' in last_quote and pd.notna(last_quote['ask_size']):
                        last_ask_size = int(last_quote['ask_size'])

            # Get last price from trades
            if not self.raw_trades_df.empty:
                initial_trades = self.raw_trades_df.head(10)
                if not initial_trades.empty:
                    last_trade = initial_trades.iloc[-1]

                    if 'price' in last_trade and pd.notna(last_trade['price']):
                        last_price = float(last_trade['price'])
        except Exception as e:
            self.logger.warning(f"Error initializing LOCF values: {e}")

        return last_price, last_bid, last_ask, last_bid_size, last_ask_size

    def _get_bars_window(self, current_ts: datetime, bars_dict: Dict,
                         window_size: int, interval_minutes: int) -> List[Dict]:
        """
        Get a window of bars ending at or before the current timestamp.

        Args:
            current_ts: Current timestamp
            bars_dict: Dictionary of bars indexed by timestamp
            window_size: Number of bars to include
            interval_minutes: Bar interval in minutes (1 or 5)

        Returns:
            List of bar dictionaries in chronological order
        """
        window = []

        # Calculate the bar start time for current timestamp
        # For 1-minute: floor to the minute
        # For 5-minute: floor to the nearest 5-minute mark
        current_market_time = current_ts.astimezone(self.market_tz)

        if interval_minutes == 1:
            # Simple minute flooring
            bar_start = current_market_time.replace(second=0, microsecond=0)
        else:  # 5-minute
            # Floor to the nearest 5-minute mark
            minute = current_market_time.minute
            floored_minute = (minute // interval_minutes) * interval_minutes
            bar_start = current_market_time.replace(minute=floored_minute, second=0, microsecond=0)

        # Convert back to UTC for dictionary lookup
        bar_start_utc = bar_start.astimezone(self.utc_tz)

        # Collect bars for the window
        collected_bars = []

        # Start with the current or most recent bar
        current_bar_ts = bar_start_utc

        # Look back through the required number of bars
        for _ in range(window_size):
            # Check if this bar exists in the dictionary
            if current_bar_ts in bars_dict:
                collected_bars.append(bars_dict[current_bar_ts])

            # Move to previous bar
            current_bar_ts -= timedelta(minutes=interval_minutes)

        # Return bars in chronological order (oldest first)
        return list(reversed(collected_bars))

    def _aggregate_trades_to_bar(self, trades: List[Dict], bar_end: datetime) -> Optional[Dict]:
        """
        Aggregate trades into a 1-second OHLCV bar.

        Args:
            trades: List of trade dictionaries
            bar_end: End timestamp for the bar

        Returns:
            Dictionary with OHLCV data or None if not enough data
        """
        if not trades:
            return None

        try:
            # Extract prices and sizes, handling potential missing data
            prices = []
            sizes = []

            for trade in trades:
                price = trade.get('price')
                size = trade.get('size')

                if pd.notna(price) and pd.notna(size):
                    prices.append(float(price))
                    sizes.append(float(size))

            if not prices:
                return None

            # Calculate OHLCV
            total_volume = sum(sizes)

            # Calculate VWAP if we have volume
            vwap = None
            if total_volume > 0:
                vwap = sum(p * s for p, s in zip(prices, sizes)) / total_volume

            return {
                'timestamp': bar_end,
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'volume': total_volume,
                'vwap': vwap
            }
        except Exception as e:
            self.logger.warning(f"Error aggregating trades to bar: {e}")
            return None

    def _determine_market_session(self, timestamp: datetime) -> str:
        """
        Determine the market session for a timestamp.

        Args:
            timestamp: UTC timestamp

        Returns:
            String indicating market session: "PREMARKET", "REGULAR", "POSTMARKET", or "CLOSED"
        """
        try:
            # Convert to market timezone for comparison
            local_time = timestamp.astimezone(self.market_tz).time()

            if MARKET_HOURS["PREMARKET_START"] <= local_time < MARKET_HOURS["REGULAR_START"]:
                return "PREMARKET"
            elif MARKET_HOURS["REGULAR_START"] <= local_time < MARKET_HOURS["REGULAR_END"]:
                return "REGULAR"
            elif MARKET_HOURS["REGULAR_END"] <= local_time <= MARKET_HOURS["POSTMARKET_END"]:
                return "POSTMARKET"
            return "CLOSED"
        except Exception as e:
            self.logger.warning(f"Error determining market session: {e}")
            return "UNKNOWN"

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the market state at the current simulation time.

        Returns:
            Dictionary with the current market state or None if not available
        """
        if self._current_agent_time_idx < 0 or self._current_agent_time_idx >= len(self._agent_timeline_utc):
            self.logger.warning(f"Invalid agent time index: {self._current_agent_time_idx}")
            return None

        current_ts = self._agent_timeline_utc[self._current_agent_time_idx]
        return self.get_state_at_time(current_ts)

    def get_state_at_time(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Get the market state at a specific timestamp.

        Args:
            timestamp: UTC timestamp to get state for

        Returns:
            Dictionary with market state or None if not available
        """
        # Ensure timestamp is in UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.utc_tz)
        elif timestamp.tzinfo != self.utc_tz:
            timestamp = timestamp.astimezone(self.utc_tz)

        # Check if this exact timestamp is in our precomputed states
        if timestamp in self._precomputed_states:
            return self._precomputed_states[timestamp]

        # Not found - look for closest state
        if self._agent_timeline_utc:
            # Binary search for closest time
            idx = bisect.bisect_left(self._agent_timeline_utc, timestamp)

            # Check if we found a valid index
            if 0 <= idx < len(self._agent_timeline_utc):
                closest_ts = self._agent_timeline_utc[idx]
                return self._precomputed_states.get(closest_ts)

            # If timestamp is before all our data, use the first state
            if idx == 0 and self._agent_timeline_utc:
                first_ts = self._agent_timeline_utc[0]
                return self._precomputed_states.get(first_ts)

            # If timestamp is after all our data, use the last state
            if idx == len(self._agent_timeline_utc) and self._agent_timeline_utc:
                last_ts = self._agent_timeline_utc[-1]
                return self._precomputed_states.get(last_ts)

        self.logger.warning(f"No state found for timestamp {timestamp}")
        return None

    def step(self) -> bool:
        """
        Advance the simulation by one step.

        Returns:
            True if successfully stepped, False if at the end of data
        """
        if self.is_done():
            self.logger.info("Step called but simulation is already done")
            return False

        # Increment agent time index
        self._current_agent_time_idx += 1

        # Check if we've reached the end
        if self._current_agent_time_idx >= len(self._agent_timeline_utc):
            self._current_agent_time_idx = len(self._agent_timeline_utc) - 1
            self.logger.info("Reached end of simulation data")
            return False

        # Update current timestamp
        self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]
        return True

    def is_done(self) -> bool:
        """Check if the simulation has reached the end."""
        return (self._agent_timeline_utc and
                self._current_agent_time_idx >= len(self._agent_timeline_utc) - 1)

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Reset the simulator to the start of the timeline or a random point.

        Args:
            options: Dictionary with reset options:
                    - random_start: If True, start at a random point in the timeline
                    - start_time_offset_seconds: Offset from the beginning in seconds

        Returns:
            Initial market state
        """
        options = options or {}
        options['random_start'] = False # Todo : Remove
        self.logger.info(f"Resetting simulation with options: {options}")

        if not self._agent_timeline_utc:
            self.logger.error("Cannot reset: timeline is empty")
            return None

        # Handle random start
        if options.get('random_start', False):
            if len(self._agent_timeline_utc) > 1:
                self._current_agent_time_idx = self.np_random.integers(
                    0,  # Start from the very beginning - no buffer needed
                    len(self._agent_timeline_utc) - 1
                )
                self.logger.info(f"Random reset to index {self._current_agent_time_idx}")
            else:
                self._current_agent_time_idx = 0
        else:
            # Apply any specified offset, starting from the very beginning
            offset = options.get('start_time_offset_seconds', 0)
            self._current_agent_time_idx = min(offset, len(self._agent_timeline_utc) - 1)
            self.logger.info(f"Reset to index {self._current_agent_time_idx}")

        # Set current timestamp
        self.current_timestamp_utc = self._agent_timeline_utc[self._current_agent_time_idx]

        # Return the current state
        return self.get_current_market_state()

    def close(self):
        """Release resources used by the simulator."""
        self.logger.info("Closing market simulator and releasing resources")

        # Clear large data structures
        self._precomputed_states.clear()
        self._agent_timeline_utc.clear()

        # Delete DataFrames to free memory
        del self.raw_trades_df
        del self.raw_quotes_df
        del self.raw_1m_bars_df
        del self.raw_5m_bars_df
        del self.raw_1d_bars_df

        # Reset references
        self.raw_trades_df = pd.DataFrame()
        self.raw_quotes_df = pd.DataFrame()
        self.raw_1m_bars_df = pd.DataFrame()
        self.raw_5m_bars_df = pd.DataFrame()
        self.raw_1d_bars_df = pd.DataFrame()