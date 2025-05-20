# market_simulator.py
import logging
from collections import deque
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import bisect
from typing import Any, Dict, List, Optional, Tuple, Union

from config.config import MarketConfig, ModelConfig
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
    Market simulator for intraday trading with proper pre/post market handling.

    Features:
    - Efficient data loading for full trading hours (4:00 AM - 8:00 PM ET)
    - Proper handling of previous day's data for early session feature extraction
    - Uniform timeline construction with 1-second intervals
    - Clear market session identification
    """

    def __init__(
            self,
            symbol: str,
            data_manager: DataManager,
            market_config: MarketConfig,
            model_config: ModelConfig,
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
        self.model_config = model_config
        self.mode = mode
        self.np_random = np_random or np.random.default_rng()

        # Use a consistent timezone representation
        self.market_tz = ZoneInfo(MARKET_HOURS["TIMEZONE"])
        self.utc_tz = ZoneInfo("UTC")

        # Window sizes for data buffering (all consistent with seq_len * 2 for safety)
        self.hf_window_size = max(1, self.model_config.hf_seq_len * 2)
        self.mf_window_size = max(1, self.model_config.mf_seq_len * 2)
        self.lf_window_size = max(1, self.model_config.lf_seq_len * 2)

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

        # Bar timelines - complete expected bar times, even if no data
        self._one_min_bar_timeline = []
        self._five_min_bar_timeline = []

        # Current state tracking
        self.current_timestamp_utc = None
        self._current_agent_time_idx = -1

        # Raw data storage for different frequencies
        self.raw_trades_df = pd.DataFrame()
        self.raw_quotes_df = pd.DataFrame()
        self.raw_1s_bars_df = pd.DataFrame()
        self.raw_1m_bars_df = pd.DataFrame()
        self.raw_5m_bars_df = pd.DataFrame()
        self.raw_1d_bars_df = pd.DataFrame()  # Historical daily data

        # Previous day data storage
        self.prev_day_date = None
        self.prev_day_open = None
        self.prev_day_high = None
        self.prev_day_low = None
        self.prev_day_close = None
        self.prev_day_vwap = None

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

        # Load daily data first to find previous trading day
        session_market_dt = self.session_start_utc.astimezone(self.market_tz)
        session_date = session_market_dt.date()

        # Load a year of daily data for context
        try:
            daily_start_dt = datetime.combine(session_date - timedelta(days=365), time(0, 0, 0), tzinfo=self.market_tz)
            daily_end_dt = datetime.combine(session_date, time(23, 59, 59), tzinfo=self.market_tz)

            self.raw_1d_bars_df = self.data_manager.get_bars(
                symbol=self.symbol,
                timeframe="1d",
                start_time=daily_start_dt,
                end_time=daily_end_dt
            )
            self.raw_1d_bars_df = self._prepare_dataframe(
                self.raw_1d_bars_df,
                ['open', 'high', 'low', 'close', 'volume']
            )
            self.logger.info(f"Loaded {len(self.raw_1d_bars_df)} daily bars")
        except Exception as e:
            self.logger.error(f"Error loading daily bars: {e}")

        # Now load data for simulation
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

    def _find_previous_valid_trading_day(self, reference_date: date) -> Optional[date]:
        """
        Find the most recent valid trading day before the reference date.

        Args:
            reference_date: The date to start searching from

        Returns:
            The most recent valid trading day or None if not found
        """
        # Check if we have daily bars data
        if self.raw_1d_bars_df.empty:
            self.logger.warning("No daily bars data available to determine previous trading day")
            # Fallback: try the day before, but no guarantee it's a trading day
            return reference_date - timedelta(days=1)

        # Get all dates in our daily data
        trading_dates = sorted(set(self.raw_1d_bars_df.index.date))

        # Find the most recent date before reference_date
        valid_dates = [d for d in trading_dates if d < reference_date]

        if not valid_dates:
            self.logger.warning(f"No valid trading day found before {reference_date}")
            return None

        prev_trading_day = max(valid_dates)
        self.logger.info(f"Previous valid trading day for {reference_date} is {prev_trading_day}")
        return prev_trading_day

    def _calculate_data_loading_ranges(self) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Calculate data loading ranges for both current day and previous day.

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

        # Find the previous valid trading day
        prev_day_date = self._find_previous_valid_trading_day(session_date)

        if not prev_day_date:
            self.logger.warning("Could not determine previous trading day, using single day loading")
            # Just load current day data if we can't determine previous day
            return self._calculate_single_day_loading_ranges(session_date)

        self.prev_day_date = prev_day_date

        # Calculate ranges for both current day and previous day

        # 1. Current day full range
        current_day_start = datetime.combine(
            session_date,
            MARKET_HOURS["PREMARKET_START"],
            tzinfo=self.market_tz
        )

        current_day_end = datetime.combine(
            session_date,
            MARKET_HOURS["POSTMARKET_END"],
            tzinfo=self.market_tz
        )

        # 2. Previous day full range
        prev_day_start = datetime.combine(
            prev_day_date,
            MARKET_HOURS["PREMARKET_START"],
            tzinfo=self.market_tz
        )

        prev_day_end = datetime.combine(
            prev_day_date,
            MARKET_HOURS["POSTMARKET_END"],
            tzinfo=self.market_tz
        )

        # Combine ranges for comprehensive data loading
        combined_hf_start = prev_day_start.astimezone(self.utc_tz)
        combined_hf_end = current_day_end.astimezone(self.utc_tz)

        combined_mf_start = prev_day_start.astimezone(self.utc_tz)
        combined_mf_end = current_day_end.astimezone(self.utc_tz)

        combined_lf_start = prev_day_start.astimezone(self.utc_tz)
        combined_lf_end = current_day_end.astimezone(self.utc_tz)

        # Daily data range (already loaded earlier)
        daily_start = datetime.combine(
            session_date - timedelta(days=365),
            time(0, 0, 0),
            tzinfo=self.market_tz
        ).astimezone(self.utc_tz)

        daily_end = datetime.combine(
            session_date,
            time(23, 59, 59),
            tzinfo=self.market_tz
        ).astimezone(self.utc_tz)

        return {
            "hf": (combined_hf_start, combined_hf_end),
            "mf": (combined_mf_start, combined_mf_end),
            "lf": (combined_lf_start, combined_lf_end),
            "daily": (daily_start, daily_end)
        }

    def _calculate_single_day_loading_ranges(self, session_date: date) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Calculate data loading ranges for just the current day.

        Args:
            session_date: The current session date

        Returns:
            Dictionary with data types as keys and (start_time, end_time) as values
        """
        # Current day full range
        current_day_start = datetime.combine(
            session_date,
            MARKET_HOURS["PREMARKET_START"],
            tzinfo=self.market_tz
        )

        current_day_end = datetime.combine(
            session_date,
            MARKET_HOURS["POSTMARKET_END"],
            tzinfo=self.market_tz
        )

        # Convert to UTC for data loading
        hf_start = current_day_start.astimezone(self.utc_tz)
        hf_end = current_day_end.astimezone(self.utc_tz)

        # Daily data range - convert date objects to timezone-aware datetime objects
        daily_start = datetime.combine(
            session_date - timedelta(days=365),  # date object
            time(0, 0, 0),  # time at midnight
            tzinfo=self.market_tz  # timezone
        ).astimezone(self.utc_tz)

        daily_end = datetime.combine(
            session_date,  # date object
            time(23, 59, 59),  # end of day
            tzinfo=self.market_tz  # timezone
        ).astimezone(self.utc_tz)

        return {
            "hf": (hf_start, hf_end),
            "mf": (hf_start, hf_end),
            "lf": (hf_start, hf_end),
            "daily": (daily_start, daily_end)
        }

    def _load_data_for_simulation(self):
        """Load required data for simulation."""
        loading_ranges = self._calculate_data_loading_ranges()
        if not loading_ranges:
            self.logger.error("Failed to calculate data loading ranges")
            return

        self.logger.info(f"Data loading ranges for {self.symbol}:")
        for timeframe, (start, end) in loading_ranges.items():
            self.logger.info(f"  {timeframe}: {start} to {end}")

        # Load high-frequency data (trades, quotes)
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

        # Load 1-minute bars
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

        # Load 5-minute bars
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

        # Extract previous day data
        if self.prev_day_date:
            self.logger.info(f"Extracting previous day data for {self.prev_day_date}")
            self.get_previous_day_data()  # This sets the class variables

    def _build_efficient_timeline(self):
        """
        Build uniform timelines for 1-second, 1-minute, and 5-minute bars
        for the entire session (4:00 AM to 8:00 PM ET).
        """
        if not self.session_start_utc or not self.session_end_utc:
            self.logger.error("Session start/end times not defined for timeline generation")
            return

        # Get the session date in market timezone
        session_market_dt = self.session_start_utc.astimezone(self.market_tz)
        session_date = session_market_dt.date()

        self.logger.info(f"Building timeline for session date {session_date} from 4:00 AM to 8:00 PM ET")

        # Define the start and end of the timeline (4:00 AM to 8:00 PM on session date)
        timeline_start = datetime.combine(
            session_date,
            MARKET_HOURS["PREMARKET_START"],
            tzinfo=self.market_tz
        ).astimezone(self.utc_tz)

        timeline_end = datetime.combine(
            session_date,
            MARKET_HOURS["POSTMARKET_END"],
            tzinfo=self.market_tz
        ).astimezone(self.utc_tz)

        # Create 1-second timeline
        self._agent_timeline_utc = []
        current_time = timeline_start

        while current_time <= timeline_end:
            self._agent_timeline_utc.append(current_time)
            current_time += timedelta(seconds=1)

        # Create 1-minute bar timeline (every minute on the minute)
        self._one_min_bar_timeline = []
        current_time = timeline_start.replace(second=0, microsecond=0)

        while current_time <= timeline_end:
            self._one_min_bar_timeline.append(current_time)
            current_time += timedelta(minutes=1)

        # Create 5-minute bar timeline (every 5 minutes)
        self._five_min_bar_timeline = []
        # Floor to the nearest 5-minute mark
        minute = timeline_start.minute
        floored_minute = (minute // 5) * 5
        current_time = timeline_start.replace(minute=floored_minute, second=0, microsecond=0)

        while current_time <= timeline_end:
            self._five_min_bar_timeline.append(current_time)
            current_time += timedelta(minutes=5)

        self.logger.info(
            f"Agent timeline created: {len(self._agent_timeline_utc)} seconds from "
            f"{self._agent_timeline_utc[0] if self._agent_timeline_utc else 'N/A'} to "
            f"{self._agent_timeline_utc[-1] if self._agent_timeline_utc else 'N/A'}"
        )
        self.logger.info(
            f"Created {len(self._one_min_bar_timeline)} 1-minute bars and "
            f"{len(self._five_min_bar_timeline)} 5-minute bars timelines"
        )

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

    def get_previous_day_data(self) -> Dict[str, Any]:
        """
        Get key data points from the previous trading day.

        Returns:
            Dictionary with previous day's OHLC, VWAP, and other relevant data
        """
        if not self.prev_day_date:
            self.logger.warning("Previous day date not set, cannot provide previous day data")
            return {}

        # Return previously stored values if available
        if self.prev_day_open is not None:
            return {
                "date": self.prev_day_date,
                "open": self.prev_day_open,
                "high": self.prev_day_high,
                "low": self.prev_day_low,
                "close": self.prev_day_close,
                "vwap": self.prev_day_vwap
            }

        # Try to extract from 1d bars
        if not self.raw_1d_bars_df.empty:
            # Find the bar for the previous day
            prev_day_bars = self.raw_1d_bars_df[self.raw_1d_bars_df.index.date == self.prev_day_date]

            if not prev_day_bars.empty:
                # Get the last bar for the day (should only be one)
                prev_day_bar = prev_day_bars.iloc[-1]

                self.prev_day_open = float(prev_day_bar.get('open', 0.0))
                self.prev_day_high = float(prev_day_bar.get('high', 0.0))
                self.prev_day_low = float(prev_day_bar.get('low', 0.0))
                self.prev_day_close = float(prev_day_bar.get('close', 0.0))

                # Try to calculate VWAP if we have intraday data
                self.prev_day_vwap = self._calculate_prev_day_vwap()

                return {
                    "date": self.prev_day_date,
                    "open": self.prev_day_open,
                    "high": self.prev_day_high,
                    "low": self.prev_day_low,
                    "close": self.prev_day_close,
                    "vwap": self.prev_day_vwap
                }

        self.logger.warning("Could not find previous day data in daily bars")
        return {}

    def _calculate_prev_day_vwap(self) -> Optional[float]:
        """
        Calculate VWAP for the previous day using minute bars if available.

        Returns:
            VWAP value or None if can't be calculated
        """
        if self.prev_day_date is None:
            return None

        # Try to use 1-minute bars for more accurate VWAP
        if not self.raw_1m_bars_df.empty:
            # Filter to previous day's regular market hours
            prev_day_regular_start = datetime.combine(
                self.prev_day_date,
                MARKET_HOURS["REGULAR_START"],
                tzinfo=self.market_tz
            ).astimezone(self.utc_tz)

            prev_day_regular_end = datetime.combine(
                self.prev_day_date,
                MARKET_HOURS["REGULAR_END"],
                tzinfo=self.market_tz
            ).astimezone(self.utc_tz)

            prev_day_bars = self.raw_1m_bars_df[
                (self.raw_1m_bars_df.index >= prev_day_regular_start) &
                (self.raw_1m_bars_df.index <= prev_day_regular_end)
                ]

            if not prev_day_bars.empty:
                # Calculate VWAP: sum(price * volume) / sum(volume)
                # Using typical price (H+L+C)/3 as the price
                typical_prices = (
                                         prev_day_bars['high'] +
                                         prev_day_bars['low'] +
                                         prev_day_bars['close']
                                 ) / 3

                volume = prev_day_bars['volume']

                # Guard against zero volume
                if volume.sum() > 0:
                    vwap = (typical_prices * volume).sum() / volume.sum()
                    return float(vwap)

        # Fallback: use simple average of OHLC4
        if self.prev_day_open is not None:
            return (self.prev_day_open + self.prev_day_high + self.prev_day_low + self.prev_day_close) / 4

        return None

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

    def _create_synthetic_bar(self,
                              timestamp: datetime,
                              prev_bar: Optional[Dict[str, Any]] = None,
                              prev_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a synthetic bar when no actual data is available.

        Args:
            timestamp: Bar timestamp
            prev_bar: Previous bar values if available
            prev_price: Previous price if no bar is available

        Returns:
            Dictionary with synthetic bar data
        """
        # Default values
        bar = {
            'timestamp': timestamp,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0.0  # No volume for synthetic bars
        }

        # Use previous bar if available
        if prev_bar is not None:
            bar['open'] = prev_bar.get('close')
            bar['high'] = prev_bar.get('close')
            bar['low'] = prev_bar.get('close')
            bar['close'] = prev_bar.get('close')
            return bar

        # Use previous price if available
        if prev_price is not None:
            bar['open'] = prev_price
            bar['high'] = prev_price
            bar['low'] = prev_price
            bar['close'] = prev_price
            return bar

        # If all else fails, use prev day close if available
        if self.prev_day_close is not None:
            bar['open'] = self.prev_day_close
            bar['high'] = self.prev_day_close
            bar['low'] = self.prev_day_close
            bar['close'] = self.prev_day_close

        return bar

    def _create_complete_bar_dictionaries(self):
        """
        Create dictionaries of bars for all expected bar timestamps,
        filling in gaps with synthetic bars.
        """
        one_min_bars_dict = {}
        five_min_bars_dict = {}

        # Create full dictionaries with actual data and synthetic fills

        # Process 1-minute bars
        prev_1m_bar = None

        # Initialize with actual data from raw bars
        if not self.raw_1m_bars_df.empty:
            # Create dict from actual data
            for idx, row in self.raw_1m_bars_df.iterrows():
                # Store using the exact same timestamp objects as in the timeline
                # to avoid floating-point precision issues
                closest_timeline_ts = self._find_closest_timeline_ts(idx, self._one_min_bar_timeline)

                if closest_timeline_ts:
                    one_min_bars_dict[closest_timeline_ts] = {
                        'timestamp': closest_timeline_ts,
                        'open': row.get('open'),
                        'high': row.get('high'),
                        'low': row.get('low'),
                        'close': row.get('close'),
                        'volume': row.get('volume', 0),
                        'is_synthetic': False
                    }
                    prev_1m_bar = one_min_bars_dict[closest_timeline_ts]
                else:
                    # If no close match in timeline, use original timestamp
                    one_min_bars_dict[idx] = {
                        'timestamp': idx,
                        'open': row.get('open'),
                        'high': row.get('high'),
                        'low': row.get('low'),
                        'close': row.get('close'),
                        'volume': row.get('volume', 0),
                        'is_synthetic': False
                    }
                    prev_1m_bar = one_min_bars_dict[idx]

        # Fill in missing 1-minute bars using timeline
        for bar_time in self._one_min_bar_timeline:
            if bar_time not in one_min_bars_dict:
                synthetic_bar = self._create_synthetic_bar(
                    timestamp=bar_time,
                    prev_bar=prev_1m_bar,
                    prev_price=self.prev_day_close
                )
                synthetic_bar['is_synthetic'] = True
                one_min_bars_dict[bar_time] = synthetic_bar
                prev_1m_bar = synthetic_bar

        # Process 5-minute bars
        prev_5m_bar = None

        # Initialize with actual data from raw bars
        if not self.raw_5m_bars_df.empty:
            # Create dict from actual data
            for idx, row in self.raw_5m_bars_df.iterrows():
                # Store using the exact same timestamp objects as in the timeline
                closest_timeline_ts = self._find_closest_timeline_ts(idx, self._five_min_bar_timeline)

                if closest_timeline_ts:
                    five_min_bars_dict[closest_timeline_ts] = {
                        'timestamp': closest_timeline_ts,
                        'open': row.get('open'),
                        'high': row.get('high'),
                        'low': row.get('low'),
                        'close': row.get('close'),
                        'volume': row.get('volume', 0),
                        'is_synthetic': False
                    }
                    prev_5m_bar = five_min_bars_dict[closest_timeline_ts]
                else:
                    # If no close match in timeline, use original timestamp
                    five_min_bars_dict[idx] = {
                        'timestamp': idx,
                        'open': row.get('open'),
                        'high': row.get('high'),
                        'low': row.get('low'),
                        'close': row.get('close'),
                        'volume': row.get('volume', 0),
                        'is_synthetic': False
                    }
                    prev_5m_bar = five_min_bars_dict[idx]

        # Fill in missing 5-minute bars using timeline
        for bar_time in self._five_min_bar_timeline:
            if bar_time not in five_min_bars_dict:
                synthetic_bar = self._create_synthetic_bar(
                    timestamp=bar_time,
                    prev_bar=prev_5m_bar,
                    prev_price=self.prev_day_close
                )
                synthetic_bar['is_synthetic'] = True
                five_min_bars_dict[bar_time] = synthetic_bar
                prev_5m_bar = synthetic_bar

        return one_min_bars_dict, five_min_bars_dict

    def _find_closest_timeline_ts(self, timestamp, timeline, max_diff_seconds=5):
        """
        Find the closest timestamp in a timeline.

        Args:
            timestamp: The timestamp to find a match for
            timeline: List of timeline timestamps
            max_diff_seconds: Maximum difference allowed in seconds

        Returns:
            The closest matching timestamp or None if no close match
        """
        if not timeline:
            return None

        # Ensure timestamp is in UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.utc_tz)
        elif timestamp.tzinfo != self.utc_tz:
            timestamp = timestamp.astimezone(self.utc_tz)

        # Find closest match
        closest_ts = None
        min_diff = timedelta(seconds=max_diff_seconds)  # Maximum allowed difference

        for ts in timeline:
            diff = abs(ts - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_ts = ts

        return closest_ts

    def _precompute_timeline_states(self):
        """
        Precompute market states for every second in the agent's timeline.

        This builds states for all times in the agent's timeline using data
        from both the current day and previous day as needed.
        """
        self.logger.info(f"Precomputing timeline states for {self.symbol}")

        if not self._agent_timeline_utc:
            self.logger.error("Agent timeline is empty. Cannot precompute states.")
            return

        # Create complete bar dictionaries (with synthetic fills for missing bars)
        one_min_bars_dict, five_min_bars_dict = self._create_complete_bar_dictionaries()

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

        # Get previous day data
        prev_day_data = self.get_previous_day_data()
        prev_day_close = prev_day_data.get('close')

        # Initialize rolling window data structures for high-frequency data
        empty_hf_entry = {
            'timestamp': None,
            'trades': [],
            'quotes': [],
            '1s_bar': None
        }
        rolling_hf_window = deque([empty_hf_entry] * self.hf_window_size, maxlen=self.hf_window_size)

        # Try to initialize basic LOCF values from first data
        if prev_day_close is not None:
            last_price = prev_day_close
            # Estimate bid/ask spread as 0.1% of price
            last_bid = prev_day_close * 0.999
            last_ask = prev_day_close * 1.001
            self.logger.debug(f"Using previous day close for initial price: {last_price}")
        else:
            # Try to initialize from first available data
            last_price, last_bid, last_ask, last_bid_size, last_ask_size = self._initialize_locf_values(
                last_price, last_bid, last_ask, last_bid_size, last_ask_size
            )

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

                # If we're at the start of pre-market and have previous day data,
                # use it for initial values
                if current_market_dt.time() <= MARKET_HOURS["PREMARKET_START"]:
                    if prev_day_close is not None:
                        last_price = prev_day_close
                        last_bid = prev_day_close * 0.999
                        last_ask = prev_day_close * 1.001

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

            # Create 1-second bar if none exists (carryforward from last known price)
            if current_1s_bar is None and last_price is not None:
                current_1s_bar = {
                    'timestamp': current_ts,
                    'open': last_price,
                    'high': last_price,
                    'low': last_price,
                    'close': last_price,
                    'volume': 0.0,
                    'is_synthetic': True
                }

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
                'previous_day_data': prev_day_data,
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
        # Collect bars for the window
        collected_bars = []

        # Get the appropriate bar timeline
        if interval_minutes == 1:
            bar_timeline = self._one_min_bar_timeline
        else:  # 5-minute
            bar_timeline = self._five_min_bar_timeline

        if not bar_timeline:
            self.logger.error(f"No {interval_minutes}-minute bar timeline available")
            return []

        # Calculate the bar start time for current timestamp
        # For 1-minute: floor to the minute
        # For 5-minute: floor to the nearest 5-minute mark
        current_market_time = current_ts.astimezone(self.market_tz)

        if interval_minutes == 1:
            # Simple minute flooring
            bar_time = current_market_time.replace(second=0, microsecond=0)
        else:  # 5-minute
            # Floor to the nearest 5-minute mark
            minute = current_market_time.minute
            floored_minute = (minute // interval_minutes) * interval_minutes
            bar_time = current_market_time.replace(minute=floored_minute, second=0, microsecond=0)

        # Convert back to UTC for timeline comparison
        bar_time_utc = bar_time.astimezone(self.utc_tz)

        # Find the index of the current bar in the timeline
        try:
            current_bar_idx = bar_timeline.index(bar_time_utc)
        except ValueError:
            # If not found (shouldn't happen with complete timelines), find closest
            for i, t in enumerate(bar_timeline):
                if t >= bar_time_utc:
                    current_bar_idx = i
                    break
            else:
                current_bar_idx = len(bar_timeline) - 1

        # Get bars from timeline
        start_idx = max(0, current_bar_idx - window_size + 1)
        end_idx = current_bar_idx + 1  # exclusive end

        for timeline_ts in bar_timeline[start_idx:end_idx]:
            # Look for exact timestamp match first
            if timeline_ts in bars_dict:
                collected_bars.append(bars_dict[timeline_ts])
                continue

            # If exact match not found, find best match
            # (this should never happen with our implementation, but just to be safe)
            best_match = None
            min_diff = timedelta(minutes=1)  # Maximum allowed difference

            for bar_ts in bars_dict:
                diff = abs(bar_ts - timeline_ts)
                if diff < min_diff:
                    min_diff = diff
                    best_match = bar_ts

            if best_match:
                collected_bars.append(bars_dict[best_match])
            else:
                # Create synthetic bar if needed
                if collected_bars:
                    # Use the last bar's close
                    prev_bar = collected_bars[-1]
                    synthetic_bar = self._create_synthetic_bar(
                        timestamp=timeline_ts,
                        prev_bar=prev_bar
                    )
                else:
                    # Use previous day close if available
                    synthetic_bar = self._create_synthetic_bar(
                        timestamp=timeline_ts,
                        prev_price=self.prev_day_close
                    )

                synthetic_bar['is_synthetic'] = True
                collected_bars.append(synthetic_bar)

        # Return in chronological order (should already be in order)
        return collected_bars

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
                'vwap': vwap,
                'is_synthetic': False
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
        self.logger.info(f"Resetting simulation with options: {options}")

        if not self._agent_timeline_utc:
            self.logger.error("Cannot reset: timeline is empty")
            return None

        # Handle random start
        if options.get('random_start', False):
            # Set a minimum number of seconds into the day to ensure enough lookback data
            # For example, at least 1 hour (3600 seconds) after premarket start
            min_offset = max(self.hf_window_size, self.mf_window_size, self.lf_window_size)

            if len(self._agent_timeline_utc) > (min_offset + 1):
                self._current_agent_time_idx = self.np_random.integers(
                    min_offset,
                    len(self._agent_timeline_utc) - 1
                )
                self.logger.info(f"Random reset to index {self._current_agent_time_idx}")
            else:
                self._current_agent_time_idx = min(min_offset, len(self._agent_timeline_utc) - 1)
        else:
            # Apply any specified offset, with minimum to ensure enough lookback data
            min_offset = max(self.hf_window_size, self.mf_window_size, self.lf_window_size)
            offset = max(min_offset, options.get('start_time_offset_seconds', min_offset))

            self._current_agent_time_idx = min(offset, len(self._agent_timeline_utc) - 1)
            self.logger.info(f"Reset to index {self._current_agent_time_idx} with offset {offset}")

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
        self._one_min_bar_timeline.clear()
        self._five_min_bar_timeline.clear()

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