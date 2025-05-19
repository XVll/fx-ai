# feature_extractor.py
import logging
from collections import deque
from datetime import datetime, timezone, timedelta, time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from config.config import FeatureConfig
from simulators.market_simulator import MarketSimulator


class FeatureExtractor:
    """
    Feature extractor for trading environment.

    Extracts features from market data at multiple timeframes:
    - HF: High-frequency features from 1-second data
    - MF: Medium-frequency features from 1-minute and 5-minute data
    - LF: Low-frequency features from daily data
    - Static: Constant features like time of day, market cap, etc.

    Handles missing data gracefully and provides robust feature calculation.
    """
    # Constants
    EPSILON = 1e-9  # Small value to avoid division by zero
    ET_TZ = ZoneInfo("America/New_York")  # Exchange timezone

    # Time-of-day constants
    SESSION_START_HOUR_ET = 4  # 4:00 AM ET
    SESSION_END_HOUR_ET = 20  # 8:00 PM ET
    REGULAR_SESSION_START_HOUR_ET = 9.5  # 9:30 AM ET
    REGULAR_SESSION_END_HOUR_ET = 16  # 4:00 PM ET
    TOTAL_SECONDS_IN_FULL_TRADING_DAY = (SESSION_END_HOUR_ET - SESSION_START_HOUR_ET) * 3600

    # Feature configuration constants
    # HF (High-Frequency) settings
    HF_ROLLING_WINDOW_SECONDS = 60
    HF_LARGE_TRADE_THRESHOLD_MULTIPLIER = 5.0

    # MF (Medium-Frequency) settings
    MF_EMA_SHORT_PERIOD = 9
    MF_EMA_LONG_PERIOD = 20
    MF_MACD_FAST_PERIOD = 12
    MF_MACD_SLOW_PERIOD = 26
    MF_MACD_SIGNAL_PERIOD = 9
    MF_ATR_PERIOD = 14
    MF_VOLUME_AVG_RECENT_BARS_WINDOW = 20
    MF_SWING_LOOKBACK_BARS = 10
    MF_SWING_STRENGTH_BARS = 2

    # LF (Low-Frequency) settings
    LF_AVG_DAILY_VOLUME_PERIOD_DAYS = 10

    def __init__(self, symbol: str, market_simulator: MarketSimulator, config: FeatureConfig,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the feature extractor.

        Args:
            symbol: Symbol being traded
            market_simulator: Market simulator providing data
            config: Feature configuration
            logger: Optional logger
        """
        self.symbol = symbol
        self.market_simulator = market_simulator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Get symbol info from market simulator
        try:
            symbol_info = self.market_simulator.get_symbol_info()
            self.total_shares_outstanding = symbol_info.get('total_shares_outstanding')
            if self.total_shares_outstanding is None:
                self.logger.warning("Total shares outstanding not found in symbol info. Market cap feature will be unavailable.")
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            self.total_shares_outstanding = None

        # Define feature names (maintain same dimensions as original)
        self.static_feature_names = [
            "S_Time_Of_Day_Seconds_Encoded_Sin",
            "S_Time_Of_Day_Seconds_Encoded_Cos",
            "S_Market_Cap_Million"
        ]

        self.hf_feature_names = [
            "HF_1s_PriceChange_Pct", "HF_1s_Volume_Ratio_To_Own_Avg", "HF_1s_Volume_Delta_Pct",
            "HF_1s_HighLow_Spread_Rel", "HF_Tape_1s_Trades_Count_Ratio_To_Own_Avg",
            "HF_Tape_1s_Trades_Count_Delta_Pct", "HF_Tape_1s_Normalized_Volume_Imbalance",
            "HF_Tape_1s_Normalized_Volume_Imbalance_Delta", "HF_Tape_1s_Avg_Trade_Size_Ratio_To_Own_Avg",
            "HF_Tape_1s_Avg_Trade_Size_Delta_Pct", "HF_Tape_1s_Large_Trade_Count",
            "HF_Tape_1s_Large_Trade_Net_Volume_Ratio_To_Total_Vol", "HF_Tape_1s_Trades_VWAP",
            "HF_Quote_1s_Spread_Rel", "HF_Quote_1s_Spread_Rel_Delta", "HF_Quote_1s_Quote_Imbalance_Value_Ratio",
            "HF_Quote_1s_Quote_Imbalance_Value_Ratio_Delta", "HF_Quote_1s_Bid_Value_USD_Ratio_To_Own_Avg",
            "HF_Quote_1s_Ask_Value_USD_Ratio_To_Own_Avg"
        ]

        self.mf_feature_names = [
            "MF_1m_PriceChange_Pct", "MF_5m_PriceChange_Pct", "MF_1m_PriceChange_Pct_Delta",
            "MF_5m_PriceChange_Pct_Delta", "MF_1m_Position_In_CurrentCandle_Range",
            "MF_5m_Position_In_CurrentCandle_Range", "MF_1m_Position_In_PreviousCandle_Range",
            "MF_5m_Position_In_PreviousCandle_Range", "MF_1m_Dist_To_EMA9_Pct", "MF_1m_Dist_To_EMA20_Pct",
            "MF_5m_Dist_To_EMA9_Pct", "MF_5m_Dist_To_EMA20_Pct", "MF_Dist_To_Rolling_HF_High_Pct",
            "MF_Dist_To_Rolling_HF_Low_Pct", "MF_1m_MACD_Line", "MF_1m_MACD_Signal", "MF_1m_MACD_Hist",
            "MF_1m_ATR_Pct", "MF_5m_ATR_Pct", "MF_1m_BodySize_Rel", "MF_1m_UpperWick_Rel",
            "MF_1m_LowerWick_Rel", "MF_5m_BodySize_Rel", "MF_5m_UpperWick_Rel", "MF_5m_LowerWick_Rel",
            "MF_1m_BarVol_Ratio_To_TodaySoFarVol", "MF_5m_BarVol_Ratio_To_TodaySoFarVol",
            "MF_1m_Volume_Rel_To_Avg_Recent_Bars", "MF_5m_Volume_Rel_To_Avg_Recent_Bars",
            "MF_1m_Dist_To_Recent_SwingHigh_Pct", "MF_1m_Dist_To_Recent_SwingLow_Pct",
            "MF_5m_Dist_To_Recent_SwingHigh_Pct", "MF_5m_Dist_To_Recent_SwingLow_Pct"
        ]

        self.lf_feature_names = [
            "LF_Position_In_Daily_Range", "LF_Position_In_PrevDay_Range", "LF_Pct_Change_From_Prev_Close",
            "LF_RVol_Pct_From_Avg_10d_Timed", "LF_Dist_To_Session_VWAP_Pct", "LF_Daily_Dist_To_EMA9_Pct",
            "LF_Daily_Dist_To_EMA20_Pct", "LF_Daily_Dist_To_EMA200_Pct", "LF_Dist_To_Closest_LT_Support_Pct",
            "LF_Dist_To_Closest_LT_Resistance_Pct"
        ]

        # Make sure feature dimensions match config
        self.config.static_feat_dim = len(self.static_feature_names)
        self.config.hf_feat_dim = len(self.hf_feature_names)
        self.config.mf_feat_dim = len(self.mf_feature_names)
        self.config.lf_feat_dim = len(self.lf_feature_names)

        # Initialize data buffers
        self.hf_seq_len = self.config.hf_seq_len
        self.mf_seq_len = self.config.mf_seq_len
        self.lf_seq_len = self.config.lf_seq_len

        # History tracking
        self.hf_features_history = deque(maxlen=self.hf_seq_len)
        self.mf_features_history = deque(maxlen=self.mf_seq_len)
        self.lf_features_history = deque(maxlen=self.lf_seq_len)

        # Track values from the previous step for delta calculations
        self.prev_values = {
            "price": None,
            "volume": None,
            "trades_count": None,
            "trade_avg_size": None,
            "volume_imbalance": None,
            "spread_rel": None,
            "quote_imbalance_ratio": None,
            "1m_price_change_pct": None,
            "5m_price_change_pct": None
        }

        # Session tracking
        self.daily_cumulative_volume = 0.0
        self.daily_cumulative_price_volume_product = 0.0
        self.last_processed_day = None

        # Initialize buffers with NaN values
        self._initialize_feature_buffers()

        self.logger.info(f"FeatureExtractor initialized for {symbol} with {len(self.static_feature_names)} static features, "
                         f"{len(self.hf_feature_names)} HF features, {len(self.mf_feature_names)} MF features, "
                         f"and {len(self.lf_feature_names)} LF features.")

    def _initialize_feature_buffers(self):
        """Initialize feature history buffers with NaN values."""
        # Initialize with NaN arrays of the correct shape
        hf_nan_array = np.full(self.config.hf_feat_dim, np.nan, dtype=np.float32)
        mf_nan_array = np.full(self.config.mf_feat_dim, np.nan, dtype=np.float32)
        lf_nan_array = np.full(self.config.lf_feat_dim, np.nan, dtype=np.float32)

        # Fill the deques with NaN arrays
        for _ in range(self.hf_seq_len):
            self.hf_features_history.append(hf_nan_array.copy())

        for _ in range(self.mf_seq_len):
            self.mf_features_history.append(mf_nan_array.copy())

        for _ in range(self.lf_seq_len):
            self.lf_features_history.append(lf_nan_array.copy())

        self.logger.debug("Feature buffers initialized with NaN values.")

    def extract_features(self) -> Dict[str, np.ndarray]:
        """
        Extract all features from current market state.

        Returns:
            Dictionary with feature arrays for each timeframe
        """
        # Get current market state
        market_state = self.market_simulator.get_current_market_state()
        if market_state is None:
            self.logger.error("Cannot extract features: market state is None.")
            return self._get_empty_features()

        # Check for new day and reset daily metrics if needed
        self._check_and_handle_new_day(market_state)

        # Update price-volume for session VWAP
        self._update_cumulative_price_volume(market_state)

        # Calculate features for each timeframe
        static_features = self._calculate_static_features(market_state)
        hf_features = self._calculate_hf_features(market_state)
        mf_features = self._calculate_mf_features(market_state)
        lf_features = self._calculate_lf_features(market_state)

        # Update feature history
        self.hf_features_history.append(hf_features)
        self.mf_features_history.append(mf_features)
        self.lf_features_history.append(lf_features)

        # Debug: Check for NaN values
        self._debug_check_for_nans(static_features, hf_features, mf_features, lf_features)

        # Return the feature dictionary
        return {
            'static': static_features.astype(np.float32),
            'hf': np.array(list(self.hf_features_history), dtype=np.float32),
            'mf': np.array(list(self.mf_features_history), dtype=np.float32),
            'lf': np.array(list(self.lf_features_history), dtype=np.float32)
        }

    def _get_empty_features(self) -> Dict[str, np.ndarray]:
        """Return empty feature arrays when no data is available."""
        return {
            'static': np.zeros((1, self.config.static_feat_dim), dtype=np.float32),
            'hf': np.zeros((self.hf_seq_len, self.config.hf_feat_dim), dtype=np.float32),
            'mf': np.zeros((self.mf_seq_len, self.config.mf_feat_dim), dtype=np.float32),
            'lf': np.zeros((self.lf_seq_len, self.config.lf_feat_dim), dtype=np.float32)
        }

    def _debug_check_for_nans(self, static_features, hf_features, mf_features, lf_features):
        """Check for and log NaN values in feature vectors for debugging."""
        for name, features, feature_names in [
            ("static", static_features, self.static_feature_names),
            ("hf", hf_features, self.hf_feature_names),
            ("mf", mf_features, self.mf_feature_names),
            ("lf", lf_features, self.lf_feature_names)
        ]:
            nan_indices = np.where(np.isnan(features))[0]
            if len(nan_indices) > 0:
                nan_features = [feature_names[i] for i in nan_indices]
                self.logger.warning(f"NaN values detected in {name} features: {nan_features}")

    def _check_and_handle_new_day(self, market_state: Dict[str, Any]):
        """Check if it's a new trading day and reset daily metrics if needed."""
        current_timestamp = market_state.get('timestamp_utc')
        if current_timestamp is None:
            return

        current_date = current_timestamp.date()
        if self.last_processed_day is None or current_date != self.last_processed_day:
            self.logger.info(f"New trading day detected: {current_date}")
            self.daily_cumulative_volume = 0.0
            self.daily_cumulative_price_volume_product = 0.0
            self.last_processed_day = current_date

    def _update_cumulative_price_volume(self, market_state: Dict[str, Any]):
        """Update cumulative price-volume product for session VWAP calculation."""
        current_1s_bar = market_state.get('current_1s_bar')
        if current_1s_bar is None:
            return

        volume = self._safe_get(current_1s_bar, 'volume', 0.0)
        if volume <= self.EPSILON:
            return

        # Use VWAP from the bar if available, otherwise calculate from close price
        price = self._safe_get(current_1s_bar, 'vwap')
        if price is None:
            price = self._safe_get(current_1s_bar, 'close')
            if price is None:
                return

        self.daily_cumulative_volume += volume
        self.daily_cumulative_price_volume_product += price * volume

    def _calculate_static_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate static features.

        Features:
        - Time of day (sine and cosine encoding)
        - Market cap in millions

        Args:
            market_state: Current market state

        Returns:
            Array of static features
        """
        features = np.full(self.config.static_feat_dim, np.nan, dtype=np.float32)
        current_timestamp = market_state.get('timestamp_utc')

        # Time of Day Encoding (sine/cosine for cyclical representation)
        if current_timestamp:
            try:
                # Convert to exchange timezone
                current_time_et = current_timestamp.astimezone(self.ET_TZ)

                # Calculate seconds from midnight
                seconds_from_midnight = (current_time_et.hour * 3600 +
                                         current_time_et.minute * 60 +
                                         current_time_et.second)

                # Calculate seconds from session start
                session_start_seconds = self.SESSION_START_HOUR_ET * 3600
                seconds_from_session_start = max(0, seconds_from_midnight - session_start_seconds)

                # Calculate angle (0 to 2π) based on position in trading day
                angle = 2 * np.pi * seconds_from_session_start / self.TOTAL_SECONDS_IN_FULL_TRADING_DAY

                # Sine and cosine encoding
                features[0] = np.sin(angle)
                features[1] = np.cos(angle)
            except Exception as e:
                self.logger.warning(f"Error calculating time of day features: {e}")

        # Market Cap (in millions)
        # Need current price and total shares outstanding
        current_price = market_state.get('current_price')
        if self.total_shares_outstanding is not None and current_price is not None and not np.isnan(current_price):
            features[2] = (self.total_shares_outstanding * current_price) / 1_000_000.0

        return features

    def _calculate_hf_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate high-frequency (1-second) features.

        Features include:
        - Price changes
        - Volume metrics
        - Trade metrics
        - Quote metrics

        Args:
            market_state: Current market state

        Returns:
            Array of HF features
        """
        features = np.full(self.config.hf_feat_dim, np.nan, dtype=np.float32)

        # Get required data
        current_1s_bar = market_state.get('current_1s_bar', {})
        if current_1s_bar is None:
            current_1s_bar = {}

        rolling_1s_window = market_state.get('rolling_1s_data_window', [])

        # 1. Price Change Percentage (close-to-close)
        current_close = self._safe_get(current_1s_bar, 'close')
        if current_close is not None and self.prev_values['price'] is not None:
            features[0] = self._calculate_pct_change(current_close, self.prev_values['price'])
        self.prev_values['price'] = current_close

        # 2. Volume Ratio to Average
        current_volume = self._safe_get(current_1s_bar, 'volume', 0.0)
        avg_volume = self._calculate_rolling_avg(
            [self._safe_get(event.get('bar', {}), 'volume', 0.0) for event in rolling_1s_window[-self.HF_ROLLING_WINDOW_SECONDS:]],
            exclude_current=True
        )
        features[1] = 0.0 if avg_volume is None or avg_volume < self.EPSILON else current_volume / avg_volume

        # 3. Volume Delta Percentage
        if self.prev_values['volume'] is not None:
            features[2] = self._calculate_pct_change(current_volume, self.prev_values['volume'])
        self.prev_values['volume'] = current_volume

        # 4. High-Low Spread Relative to Mid
        high = self._safe_get(current_1s_bar, 'high')
        low = self._safe_get(current_1s_bar, 'low')
        if high is not None and low is not None and high > low:
            spread_abs = high - low
            mid_price = (high + low) / 2
            features[3] = 0.0 if mid_price < self.EPSILON else (spread_abs / mid_price) * 100.0

        # Get trades from last second
        last_event = rolling_1s_window[-1] if rolling_1s_window else {}
        trades = last_event.get('trades', [])

        # 5-6. Trades Count Ratio and Delta
        trades_count = len(trades)
        avg_trades_count = self._calculate_rolling_avg(
            [len(event.get('trades', [])) for event in rolling_1s_window[-self.HF_ROLLING_WINDOW_SECONDS:]],
            exclude_current=True
        )
        features[4] = 0.0 if avg_trades_count is None or avg_trades_count < self.EPSILON else trades_count / avg_trades_count

        if self.prev_values['trades_count'] is not None:
            features[5] = self._calculate_pct_change(trades_count, self.prev_values['trades_count'])
        self.prev_values['trades_count'] = trades_count

        # 7-8. Volume Imbalance
        buy_volume = sum(self._safe_get(t, 'size', 0.0) for t in trades if self._safe_get(t, 'side') == 'B')
        sell_volume = sum(self._safe_get(t, 'size', 0.0) for t in trades if self._safe_get(t, 'side') == 'A')
        total_trade_volume = buy_volume + sell_volume

        current_imbalance = 0.0
        if total_trade_volume > self.EPSILON:
            current_imbalance = (buy_volume - sell_volume) / total_trade_volume

        features[6] = current_imbalance

        if self.prev_values['volume_imbalance'] is not None:
            features[7] = current_imbalance - self.prev_values['volume_imbalance']
        self.prev_values['volume_imbalance'] = current_imbalance

        # 9-10. Average Trade Size
        current_avg_trade_size = 0.0
        if trades_count > 0:
            current_avg_trade_size = total_trade_volume / trades_count

        # Get historical average trade size
        historical_trade_sizes = []
        for event in rolling_1s_window[-self.HF_ROLLING_WINDOW_SECONDS:]:
            event_trades = event.get('trades', [])
            if event_trades:
                event_volume = sum(self._safe_get(t, 'size', 0.0) for t in event_trades)
                if event_volume > 0 and len(event_trades) > 0:
                    historical_trade_sizes.append(event_volume / len(event_trades))

        avg_trade_size = self._calculate_avg(historical_trade_sizes, exclude_current=True)
        features[8] = 0.0 if avg_trade_size is None or avg_trade_size < self.EPSILON else current_avg_trade_size / avg_trade_size

        if self.prev_values['trade_avg_size'] is not None:
            features[9] = self._calculate_pct_change(current_avg_trade_size, self.prev_values['trade_avg_size'])
        self.prev_values['trade_avg_size'] = current_avg_trade_size

        # 11-12. Large Trades
        large_trade_threshold = 0.0
        if avg_trade_size is not None and avg_trade_size > 0:
            large_trade_threshold = avg_trade_size * self.HF_LARGE_TRADE_THRESHOLD_MULTIPLIER

        large_trades = [t for t in trades if self._safe_get(t, 'size', 0.0) > large_trade_threshold] if large_trade_threshold > 0 else []
        features[10] = len(large_trades)

        large_buy_vol = sum(self._safe_get(t, 'size', 0.0) for t in large_trades if self._safe_get(t, 'side') == 'B')
        large_sell_vol = sum(self._safe_get(t, 'size', 0.0) for t in large_trades if self._safe_get(t, 'side') == 'A')

        if current_volume > self.EPSILON:
            features[11] = (large_buy_vol - large_sell_vol) / current_volume

        # 13. VWAP from Trades
        features[12] = self._safe_get(current_1s_bar, 'vwap')

        # 14-15. Quote Spread Relative & Delta
        bid_price = market_state.get('best_bid_price')
        ask_price = market_state.get('best_ask_price')

        current_spread_rel = 0.0
        if bid_price is not None and ask_price is not None and bid_price > self.EPSILON:
            mid_price = (bid_price + ask_price) / 2
            if mid_price > self.EPSILON:
                current_spread_rel = ((ask_price - bid_price) / mid_price) * 100.0

        features[13] = current_spread_rel

        if self.prev_values['spread_rel'] is not None:
            features[14] = current_spread_rel - self.prev_values['spread_rel']
        self.prev_values['spread_rel'] = current_spread_rel

        # 16-17. Quote Imbalance Value Ratio & Delta
        bid_size = market_state.get('best_bid_size', 0)
        ask_size = market_state.get('best_ask_size', 0)

        bid_value = bid_price * bid_size if bid_price is not None else 0.0
        ask_value = ask_price * ask_size if ask_price is not None else 0.0
        total_value = bid_value + ask_value

        current_quote_imbalance = 0.0
        if total_value > self.EPSILON:
            current_quote_imbalance = (bid_value - ask_value) / total_value

        features[15] = current_quote_imbalance

        if self.prev_values['quote_imbalance_ratio'] is not None:
            features[16] = current_quote_imbalance - self.prev_values['quote_imbalance_ratio']
        self.prev_values['quote_imbalance_ratio'] = current_quote_imbalance

        # 18-19. Bid/Ask Value Ratio to Average
        historical_bid_values = []
        historical_ask_values = []

        for event in rolling_1s_window[-self.HF_ROLLING_WINDOW_SECONDS:]:
            quotes = event.get('quotes', [])
            if quotes:
                last_quote = quotes[-1]
                event_bid_price = self._safe_get(last_quote, 'bid_price')
                event_ask_price = self._safe_get(last_quote, 'ask_price')
                event_bid_size = self._safe_get(last_quote, 'bid_size', 0)
                event_ask_size = self._safe_get(last_quote, 'ask_size', 0)

                if event_bid_price is not None:
                    historical_bid_values.append(event_bid_price * event_bid_size)
                if event_ask_price is not None:
                    historical_ask_values.append(event_ask_price * event_ask_size)

        avg_bid_value = self._calculate_avg(historical_bid_values, exclude_current=True)
        avg_ask_value = self._calculate_avg(historical_ask_values, exclude_current=True)

        features[17] = 0.0 if avg_bid_value is None or avg_bid_value < self.EPSILON else bid_value / avg_bid_value
        features[18] = 0.0 if avg_ask_value is None or avg_ask_value < self.EPSILON else ask_value / avg_ask_value

        return features

    def _calculate_mf_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate medium-frequency (1-minute and 5-minute) features.

        Features include:
        - Price changes
        - Candle position metrics
        - Technical indicators
        - Volume metrics

        Args:
            market_state: Current market state

        Returns:
            Array of MF features
        """
        features = np.full(self.config.mf_feat_dim, np.nan, dtype=np.float32)

        # Get required data
        completed_1m_bars = market_state.get('completed_1m_bars_window', [])
        completed_5m_bars = market_state.get('completed_5m_bars_window', [])
        current_1m_bar_forming = market_state.get('current_1m_bar_forming')
        current_5m_bar_forming = market_state.get('current_5m_bar_forming')
        rolling_1s_window = market_state.get('rolling_1s_data_window', [])

        # Current price for relative calculations
        current_price = market_state.get('current_price')
        if current_price is None:
            current_price = self._safe_get(market_state.get('current_1s_bar', {}), 'close')

        # 1-2. Price Change Percentage for 1m and 5m
        if completed_1m_bars and len(completed_1m_bars) >= 2:
            last_bar = completed_1m_bars[-1]
            prev_bar = completed_1m_bars[-2]
            last_close = self._safe_get(last_bar, 'close')
            prev_close = self._safe_get(prev_bar, 'close')

            if last_close is not None and prev_close is not None and prev_close > self.EPSILON:
                features[0] = ((last_close - prev_close) / prev_close) * 100.0

        if completed_5m_bars and len(completed_5m_bars) >= 2:
            last_bar = completed_5m_bars[-1]
            prev_bar = completed_5m_bars[-2]
            last_close = self._safe_get(last_bar, 'close')
            prev_close = self._safe_get(prev_bar, 'close')

            if last_close is not None and prev_close is not None and prev_close > self.EPSILON:
                features[1] = ((last_close - prev_close) / prev_close) * 100.0

        # 3-4. Price Change Delta
        if features[0] != np.nan and self.prev_values['1m_price_change_pct'] is not None:
            features[2] = features[0] - self.prev_values['1m_price_change_pct']
        self.prev_values['1m_price_change_pct'] = features[0]

        if features[1] != np.nan and self.prev_values['5m_price_change_pct'] is not None:
            features[3] = features[1] - self.prev_values['5m_price_change_pct']
        self.prev_values['5m_price_change_pct'] = features[1]

        # 5-8. Position in Candle Range
        def position_in_range(price, bar):
            if price is None:
                return None

            high = self._safe_get(bar, 'high')
            low = self._safe_get(bar, 'low')

            if high is not None and low is not None and high > low:
                return (price - low) / (high - low)
            return None

        # Current candles
        if current_price is not None:
            if current_1m_bar_forming:
                features[4] = position_in_range(current_price, current_1m_bar_forming)

            if current_5m_bar_forming:
                features[5] = position_in_range(current_price, current_5m_bar_forming)

        # Previous completed candles
        if current_price is not None:
            if completed_1m_bars:
                features[6] = position_in_range(current_price, completed_1m_bars[-1])

            if completed_5m_bars:
                features[7] = position_in_range(current_price, completed_5m_bars[-1])

        # 9-12. Distance to EMAs
        # 1m EMAs
        if current_price is not None and completed_1m_bars and len(completed_1m_bars) >= self.MF_EMA_LONG_PERIOD:
            # Convert to DataFrame for TA calculations
            df_1m = pd.DataFrame([
                {
                    'close': self._safe_get(bar, 'close'),
                    'timestamp': self._safe_get(bar, 'timestamp_start')
                }
                for bar in completed_1m_bars
            ]).dropna(subset=['close'])

            if not df_1m.empty and len(df_1m) >= self.MF_EMA_SHORT_PERIOD:
                try:
                    # Calculate EMAs
                    ema9 = df_1m['close'].ewm(span=self.MF_EMA_SHORT_PERIOD, adjust=False).mean().iloc[-1]
                    if not np.isnan(ema9) and ema9 > self.EPSILON:
                        features[8] = ((current_price - ema9) / ema9) * 100.0

                    if len(df_1m) >= self.MF_EMA_LONG_PERIOD:
                        ema20 = df_1m['close'].ewm(span=self.MF_EMA_LONG_PERIOD, adjust=False).mean().iloc[-1]
                        if not np.isnan(ema20) and ema20 > self.EPSILON:
                            features[9] = ((current_price - ema20) / ema20) * 100.0
                except Exception as e:
                    self.logger.warning(f"Error calculating 1m EMAs: {e}")

        # 5m EMAs
        if current_price is not None and completed_5m_bars and len(completed_5m_bars) >= self.MF_EMA_LONG_PERIOD:
            # Convert to DataFrame for TA calculations
            df_5m = pd.DataFrame([
                {
                    'close': self._safe_get(bar, 'close'),
                    'timestamp': self._safe_get(bar, 'timestamp_start')
                }
                for bar in completed_5m_bars
            ]).dropna(subset=['close'])

            if not df_5m.empty and len(df_5m) >= self.MF_EMA_SHORT_PERIOD:
                try:
                    # Calculate EMAs
                    ema9 = df_5m['close'].ewm(span=self.MF_EMA_SHORT_PERIOD, adjust=False).mean().iloc[-1]
                    if not np.isnan(ema9) and ema9 > self.EPSILON:
                        features[10] = ((current_price - ema9) / ema9) * 100.0

                    if len(df_5m) >= self.MF_EMA_LONG_PERIOD:
                        ema20 = df_5m['close'].ewm(span=self.MF_EMA_LONG_PERIOD, adjust=False).mean().iloc[-1]
                        if not np.isnan(ema20) and ema20 > self.EPSILON:
                            features[11] = ((current_price - ema20) / ema20) * 100.0
                except Exception as e:
                    self.logger.warning(f"Error calculating 5m EMAs: {e}")

        # 13-14. Distance to Rolling HF High/Low
        if current_price is not None and rolling_1s_window:
            # Extract high and low values from the rolling window
            recent_highs = []
            recent_lows = []

            for event in rolling_1s_window[-60:]:  # Last minute
                bar = event.get('bar')
                if bar:
                    high = self._safe_get(bar, 'high')
                    low = self._safe_get(bar, 'low')
                    if high is not None:
                        recent_highs.append(high)
                    if low is not None:
                        recent_lows.append(low)

            if recent_highs:
                rolling_high = max(recent_highs)
                if rolling_high > current_price:
                    features[12] = ((rolling_high - current_price) / current_price) * 100.0

            if recent_lows:
                rolling_low = min(recent_lows)
                if rolling_low < current_price:
                    features[13] = ((current_price - rolling_low) / current_price) * 100.0

        # 15-17. MACD
        if completed_1m_bars and len(completed_1m_bars) >= self.MF_MACD_SLOW_PERIOD:
            # Convert to DataFrame for MACD calculation
            df_1m = pd.DataFrame([
                {
                    'close': self._safe_get(bar, 'close'),
                    'timestamp': self._safe_get(bar, 'timestamp_start')
                }
                for bar in completed_1m_bars
            ]).dropna(subset=['close'])

            if not df_1m.empty and len(df_1m) >= self.MF_MACD_SLOW_PERIOD:
                try:
                    # Calculate MACD components
                    ema_fast = df_1m['close'].ewm(span=self.MF_MACD_FAST_PERIOD, adjust=False).mean()
                    ema_slow = df_1m['close'].ewm(span=self.MF_MACD_SLOW_PERIOD, adjust=False).mean()

                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(span=self.MF_MACD_SIGNAL_PERIOD, adjust=False).mean()
                    histogram = macd_line - signal_line

                    features[14] = macd_line.iloc[-1]
                    features[15] = signal_line.iloc[-1]
                    features[16] = histogram.iloc[-1]
                except Exception as e:
                    self.logger.warning(f"Error calculating MACD: {e}")

        # 18-19. ATR Percentage
        # 1m ATR
        if completed_1m_bars and len(completed_1m_bars) >= self.MF_ATR_PERIOD:
            # Convert to DataFrame for ATR calculation
            df_1m = pd.DataFrame([
                {
                    'high': self._safe_get(bar, 'high'),
                    'low': self._safe_get(bar, 'low'),
                    'close': self._safe_get(bar, 'close'),
                    'timestamp': self._safe_get(bar, 'timestamp_start')
                }
                for bar in completed_1m_bars
            ]).dropna(subset=['high', 'low', 'close'])

            if not df_1m.empty and len(df_1m) >= self.MF_ATR_PERIOD:
                try:
                    # Calculate True Range
                    df_1m['tr0'] = abs(df_1m['high'] - df_1m['low'])
                    df_1m['tr1'] = abs(df_1m['high'] - df_1m['close'].shift())
                    df_1m['tr2'] = abs(df_1m['low'] - df_1m['close'].shift())
                    df_1m['tr'] = df_1m[['tr0', 'tr1', 'tr2']].max(axis=1)

                    # Calculate ATR
                    atr = df_1m['tr'].rolling(self.MF_ATR_PERIOD).mean().iloc[-1]

                    if current_price is not None and current_price > self.EPSILON:
                        features[17] = (atr / current_price) * 100.0
                except Exception as e:
                    self.logger.warning(f"Error calculating 1m ATR: {e}")

        # 5m ATR
        if completed_5m_bars and len(completed_5m_bars) >= self.MF_ATR_PERIOD:
            # Convert to DataFrame for ATR calculation
            df_5m = pd.DataFrame([
                {
                    'high': self._safe_get(bar, 'high'),
                    'low': self._safe_get(bar, 'low'),
                    'close': self._safe_get(bar, 'close'),
                    'timestamp': self._safe_get(bar, 'timestamp_start')
                }
                for bar in completed_5m_bars
            ]).dropna(subset=['high', 'low', 'close'])

            if not df_5m.empty and len(df_5m) >= self.MF_ATR_PERIOD:
                try:
                    # Calculate True Range
                    df_5m['tr0'] = abs(df_5m['high'] - df_5m['low'])
                    df_5m['tr1'] = abs(df_5m['high'] - df_5m['close'].shift())
                    df_5m['tr2'] = abs(df_5m['low'] - df_5m['close'].shift())
                    df_5m['tr'] = df_5m[['tr0', 'tr1', 'tr2']].max(axis=1)

                    # Calculate ATR
                    atr = df_5m['tr'].rolling(self.MF_ATR_PERIOD).mean().iloc[-1]

                    if current_price is not None and current_price > self.EPSILON:
                        features[18] = (atr / current_price) * 100.0
                except Exception as e:
                    self.logger.warning(f"Error calculating 5m ATR: {e}")

        # 20-25. Candle Shape Metrics
        def calculate_candle_shape(bar):
            open_price = self._safe_get(bar, 'open')
            high = self._safe_get(bar, 'high')
            low = self._safe_get(bar, 'low')
            close = self._safe_get(bar, 'close')

            if None in (open_price, high, low, close) or high <= low:
                return None, None, None

            range_size = high - low
            body_size = abs(close - open_price)
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low

            # Normalize by total range
            body_rel = body_size / range_size
            upper_rel = upper_wick / range_size
            lower_rel = lower_wick / range_size

            return body_rel, upper_rel, lower_rel

        # 1m candle shape
        if completed_1m_bars:
            last_1m_bar = completed_1m_bars[-1]
            body_rel, upper_rel, lower_rel = calculate_candle_shape(last_1m_bar) or (None, None, None)
            features[19], features[20], features[21] = body_rel, upper_rel, lower_rel

        # 5m candle shape
        if completed_5m_bars:
            last_5m_bar = completed_5m_bars[-1]
            body_rel, upper_rel, lower_rel = calculate_candle_shape(last_5m_bar) or (None, None, None)
            features[22], features[23], features[24] = body_rel, upper_rel, lower_rel

        # 26-27. Bar Volume Ratio to Today's Volume
        if self.daily_cumulative_volume > self.EPSILON:
            # 1m volume ratio
            if completed_1m_bars:
                last_1m_vol = self._safe_get(completed_1m_bars[-1], 'volume', 0.0)
                features[25] = last_1m_vol / self.daily_cumulative_volume

            # 5m volume ratio
            if completed_5m_bars:
                last_5m_vol = self._safe_get(completed_5m_bars[-1], 'volume', 0.0)
                features[26] = last_5m_vol / self.daily_cumulative_volume

        # 28-29. Volume Relative to Average Recent Bars
        # 1m volume
        if completed_1m_bars and len(completed_1m_bars) > 1:
            recent_1m_vols = [
                self._safe_get(bar, 'volume', 0.0)
                for bar in completed_1m_bars[-(self.MF_VOLUME_AVG_RECENT_BARS_WINDOW + 1):-1]
            ]
            avg_recent_1m_vol = self._calculate_avg(recent_1m_vols)

            if avg_recent_1m_vol is not None and avg_recent_1m_vol > self.EPSILON:
                last_1m_vol = self._safe_get(completed_1m_bars[-1], 'volume', 0.0)
                features[27] = last_1m_vol / avg_recent_1m_vol

        # 5m volume
        if completed_5m_bars and len(completed_5m_bars) > 1:
            recent_5m_vols = [
                self._safe_get(bar, 'volume', 0.0)
                for bar in completed_5m_bars[-(self.MF_VOLUME_AVG_RECENT_BARS_WINDOW + 1):-1]
            ]
            avg_recent_5m_vol = self._calculate_avg(recent_5m_vols)

            if avg_recent_5m_vol is not None and avg_recent_5m_vol > self.EPSILON:
                last_5m_vol = self._safe_get(completed_5m_bars[-1], 'volume', 0.0)
                features[28] = last_5m_vol / avg_recent_5m_vol

        # 30-33. Distance to Recent Swing High/Low Points
        def find_swing_points(bars, lookback, strength):
            if len(bars) < lookback:
                return [], []

            recent_bars = bars[-lookback:]
            highs = [self._safe_get(bar, 'high') for bar in recent_bars]
            lows = [self._safe_get(bar, 'low') for bar in recent_bars]

            # Remove None values
            highs = [h for h in highs if h is not None]
            lows = [l for l in lows if l is not None]

            if len(highs) < (2 * strength + 1) or len(lows) < (2 * strength + 1):
                return [], []

            swing_highs = []
            swing_lows = []

            # Find swing points
            for i in range(strength, len(highs) - strength):
                window = highs[i - strength: i + strength + 1]
                if highs[i] == max(window):
                    swing_highs.append(highs[i])

            for i in range(strength, len(lows) - strength):
                window = lows[i - strength: i + strength + 1]
                if lows[i] == min(window):
                    swing_lows.append(lows[i])

            return swing_highs, swing_lows

        if current_price is not None:
            # 1m swing points
            swing_highs_1m, swing_lows_1m = find_swing_points(
                completed_1m_bars,
                self.MF_SWING_LOOKBACK_BARS,
                self.MF_SWING_STRENGTH_BARS
            )

            if swing_highs_1m:
                closest_high_1m = min(swing_highs_1m, key=lambda h: abs(h - current_price))
                if closest_high_1m > current_price:
                    features[29] = ((closest_high_1m - current_price) / current_price) * 100.0

            if swing_lows_1m:
                closest_low_1m = min(swing_lows_1m, key=lambda l: abs(l - current_price))
                if closest_low_1m < current_price:
                    features[30] = ((current_price - closest_low_1m) / current_price) * 100.0

            # 5m swing points
            swing_highs_5m, swing_lows_5m = find_swing_points(
                completed_5m_bars,
                self.MF_SWING_LOOKBACK_BARS,
                self.MF_SWING_STRENGTH_BARS
            )

            if swing_highs_5m:
                closest_high_5m = min(swing_highs_5m, key=lambda h: abs(h - current_price))
                if closest_high_5m > current_price:
                    features[31] = ((closest_high_5m - current_price) / current_price) * 100.0

            if swing_lows_5m:
                closest_low_5m = min(swing_lows_5m, key=lambda l: abs(l - current_price))
                if closest_low_5m < current_price:
                    features[32] = ((current_price - closest_low_5m) / current_price) * 100.0

        return features

    def _calculate_lf_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate low-frequency (daily) features.

        Features include:
        - Position in daily range
        - Gap metrics
        - VWAP metrics
        - Daily EMAs
        - Support/resistance metrics

        Args:
            market_state: Current market state

        Returns:
            Array of LF features
        """
        features = np.full(self.config.lf_feat_dim, np.nan, dtype=np.float32)

        # Get required data
        current_price = market_state.get('current_price')
        intraday_high = market_state.get('intraday_high')
        intraday_low = market_state.get('intraday_low')
        previous_day_close = market_state.get('previous_day_close_price')
        historical_1d_bars = market_state.get('historical_1d_bars')

        # 1. Position in Daily Range
        if (current_price is not None and intraday_high is not None and
                intraday_low is not None and intraday_high > intraday_low):
            features[0] = (current_price - intraday_low) / (intraday_high - intraday_low)

        # 2. Position in Previous Day Range
        if current_price is not None and isinstance(historical_1d_bars, pd.DataFrame) and not historical_1d_bars.empty:
            try:
                prev_day_data = historical_1d_bars.iloc[-1]
                prev_day_high = self._safe_get(prev_day_data, 'high')
                prev_day_low = self._safe_get(prev_day_data, 'low')

                if prev_day_high is not None and prev_day_low is not None and prev_day_high > prev_day_low:
                    features[1] = (current_price - prev_day_low) / (prev_day_high - prev_day_low)
            except Exception as e:
                self.logger.warning(f"Error calculating position in previous day range: {e}")

        # 3. Percentage Change from Previous Close
        if current_price is not None and previous_day_close is not None and previous_day_close > self.EPSILON:
            features[2] = ((current_price - previous_day_close) / previous_day_close) * 100.0

        # 4. Relative Volume (Percentage from Average 10-Day Timed)
        current_timestamp = market_state.get('timestamp_utc')
        if (current_timestamp is not None and self.daily_cumulative_volume > self.EPSILON and
                isinstance(historical_1d_bars, pd.DataFrame) and not historical_1d_bars.empty):
            try:
                # Calculate seconds elapsed in session
                current_time_et = current_timestamp.astimezone(self.ET_TZ)
                seconds_from_midnight = (current_time_et.hour * 3600 +
                                         current_time_et.minute * 60 +
                                         current_time_et.second)
                session_start_seconds = self.SESSION_START_HOUR_ET * 3600
                seconds_elapsed = max(0, seconds_from_midnight - session_start_seconds)
                fraction_day_elapsed = seconds_elapsed / self.TOTAL_SECONDS_IN_FULL_TRADING_DAY

                # Get average daily volume from historical data
                volumes = historical_1d_bars['volume'].tail(self.LF_AVG_DAILY_VOLUME_PERIOD_DAYS)
                avg_daily_volume = volumes.mean() if not volumes.empty else 0

                if avg_daily_volume > self.EPSILON and fraction_day_elapsed > self.EPSILON:
                    expected_volume_now = avg_daily_volume * fraction_day_elapsed
                    relative_volume = (self.daily_cumulative_volume / expected_volume_now) - 1.0
                    features[3] = relative_volume * 100.0  # Convert to percentage
            except Exception as e:
                self.logger.warning(f"Error calculating relative volume: {e}")

        # 5. Distance to Session VWAP
        if (current_price is not None and self.daily_cumulative_volume > self.EPSILON and
                self.daily_cumulative_price_volume_product > self.EPSILON):
            session_vwap = self.daily_cumulative_price_volume_product / self.daily_cumulative_volume
            if session_vwap > self.EPSILON:
                features[4] = ((current_price - session_vwap) / session_vwap) * 100.0

        # 6-8. Distance to Daily EMAs
        if current_price is not None and isinstance(historical_1d_bars, pd.DataFrame) and not historical_1d_bars.empty:
            try:
                close_series = historical_1d_bars['close']

                # EMA 9
                if len(close_series) >= 9:
                    ema9 = close_series.ewm(span=9, adjust=False).mean().iloc[-1]
                    if ema9 > self.EPSILON:
                        features[5] = ((current_price - ema9) / ema9) * 100.0

                # EMA 20
                if len(close_series) >= 20:
                    ema20 = close_series.ewm(span=20, adjust=False).mean().iloc[-1]
                    if ema20 > self.EPSILON:
                        features[6] = ((current_price - ema20) / ema20) * 100.0

                # EMA 200
                if len(close_series) >= 200:
                    ema200 = close_series.ewm(span=200, adjust=False).mean().iloc[-1]
                    if ema200 > self.EPSILON:
                        features[7] = ((current_price - ema200) / ema200) * 100.0
            except Exception as e:
                self.logger.warning(f"Error calculating daily EMAs: {e}")

        # 9-10. Distance to Closest Support/Resistance
        if current_price is not None and isinstance(historical_1d_bars, pd.DataFrame) and not historical_1d_bars.empty:
            try:
                # Simplified support/resistance identification using highs and lows
                highs = historical_1d_bars['high'].values
                lows = historical_1d_bars['low'].values

                # Find levels above current price (resistance)
                levels_above = [h for h in highs if h > current_price]

                # Find levels below current price (support)
                levels_below = [l for l in lows if l < current_price]

                if levels_above:
                    closest_resistance = min(levels_above)
                    features[9] = ((closest_resistance - current_price) / current_price) * 100.0

                if levels_below:
                    closest_support = max(levels_below)
                    features[8] = ((current_price - closest_support) / current_price) * 100.0
            except Exception as e:
                self.logger.warning(f"Error calculating support/resistance distances: {e}")

        return features

    def _safe_get(self, obj: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
        """Safely get a value from a dictionary, handling None dictionaries and missing keys."""
        if obj is None:
            return default
        return obj.get(key, default)

    def _calculate_pct_change(self, current: Optional[float], previous: Optional[float]) -> Optional[float]:
        """Calculate percentage change between two values."""
        if current is None or previous is None or abs(previous) < self.EPSILON:
            return None
        return ((current - previous) / previous) * 100.0

    def _calculate_rolling_avg(self, values: List[Any], exclude_current: bool = False) -> Optional[float]:
        """Calculate average of values in a rolling window."""
        if not values:
            return None

        if exclude_current and len(values) > 1:
            values = values[:-1]

        # Filter out None values and convert to float
        valid_values = [float(v) for v in values if v is not None and not np.isnan(v)]

        if not valid_values:
            return None

        return sum(valid_values) / len(valid_values)

    def _calculate_avg(self, values: List[Any], exclude_current: bool = False) -> Optional[float]:
        """Calculate simple average of values."""
        return self._calculate_rolling_avg(values, exclude_current)