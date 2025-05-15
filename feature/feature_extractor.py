# feature_extractor.py
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureExtractor:
    """
    Calculates, manages historical lookbacks of, and provides structured input
    for a multi-branch financial market prediction model.
    It is designed for fast-paced trading of low-float, high-volume,
    catalyst-driven stocks, operating in both premarket and regular market hours.

    Feature Design Philosophy for Lookbacks:
    Features calculated per tick should be dynamic or represent evolving relationships
    to make the lookback sequences informative. For instance, instead of just raw price,
    features like price change, distance to moving averages, or relative volume are
    more suitable for lookback sequences. Static or slowly changing features are
    handled separately. The goal is for each row in a lookback matrix (e.g., hf_features)
    to represent the state of dynamic indicators at that specific tick, allowing the
    model to learn from sequences of these states.
    """

    def __init__(self, symbol: str, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the FeatureExtractor.

        Args:
            symbol (str): The stock symbol.
            config (Optional[Dict]): Configuration dictionary. Expected keys:
                'hf_lookback_length': int (default 60)
                'mf_lookback_length': int (default 30)
                'lf_lookback_length': int (default 30)
                'sr_num_levels': int (default 5) - for S/R detection
                'long_term_sr_days': int (default 252) - for long-term S/R levels
            logger (Optional[logging.Logger]): Logger instance.
        """
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Configuration for lookback lengths
        self.hf_lookback_length = self.config.get('hf_lookback_length', 60)
        self.mf_lookback_length = self.config.get('mf_lookback_length', 30)
        self.lf_lookback_length = self.config.get('lf_lookback_length', 30)
        self.sr_num_levels = self.config.get('sr_num_levels', 5)
        self.long_term_sr_days = self.config.get('long_term_sr_days', 252)  # Approx 1 trading year

        # Internal state for daily data and S/R levels
        self.prev_day_data: Optional[Dict[str, Any]] = None  # Stores OHLC, ATR of previous day
        self.long_term_daily_sr: Optional[Dict[str, List[float]]] = {'support': [], 'resistance': []}
        self.historical_1year_daily_data: Optional[pd.DataFrame] = None  # Store for dynamic S/R if needed

        # Session VWAP accumulators (reset per session)
        self.session_vwap_sum_price_volume: float = 0.0
        self.session_vwap_sum_volume: float = 0.0
        self.current_session_vwap: Optional[float] = None
        self._current_market_session_for_vwap: Optional[str] = None

        # --- Ordered lists of feature names ---
        # These lists define the exact order for the output NumPy arrays.
        self.static_feature_names: List[str] = [
            "S_Day_Of_Week_Encoded_0", "S_Day_Of_Week_Encoded_1", "S_Day_Of_Week_Encoded_2",
            "S_Day_Of_Week_Encoded_3", "S_Day_Of_Week_Encoded_4",  # Assuming Mon-Fri, one-hot
            "S_Time_Of_Day_Seconds_Encoded_Sin", "S_Time_Of_Day_Seconds_Encoded_Cos",
            "S_Market_Session_Encoded_PREMARKET", "S_Market_Session_Encoded_REGULAR",
            "S_Market_Session_Encoded_POSTMARKET", "S_Market_Session_Encoded_OTHER",  # One-hot
            "S_Stock_Float_Category",  # Placeholder for actual categories
            "S_Has_Catalyst",
            "S_Days_Since_Catalyst",
            "S_Initial_Scan_RelVol",
            "S_Initial_Scan_Price_ROC"
        ]

        self.hf_feature_names: List[str] = [
            # HF_1s_Bar features
            "HF_1s_ClosePrice", "HF_1s_PriceChange_Abs", "HF_1s_PriceChange_Pct",
            "HF_1s_Return_Log", "HF_1s_Volume", "HF_1s_HighLow_Spread_Abs",
            "HF_1s_HighLow_Spread_Rel", "HF_1s_Close_Location_In_Bar",
            "HF_1s_Is_Gap_Up_Bar", "HF_1s_Is_Gap_Down_Bar",
            # HF_Tape (1s window)
            "HF_Tape_1s_Trades_Count", "HF_Tape_1s_Trades_Volume_Total",
            "HF_Tape_1s_Trades_Volume_Buy", "HF_Tape_1s_Trades_Volume_Sell",
            "HF_Tape_1s_Trades_Imbalance_Volume", "HF_Tape_1s_Trades_Imbalance_Count",
            "HF_Tape_1s_Trades_VWAP", "HF_Tape_1s_Avg_Trade_Size",
            "HF_Tape_1s_Large_Trade_Count", "HF_Tape_1s_Large_Trade_Imbalance",
            "HF_Tape_1s_Tape_Speed", "HF_Tape_1s_Trade_Price_StdDev",
            # HF_Tape (3s window) - Features would be similar, add suffix _3s
            "HF_Tape_3s_Trades_Count", "HF_Tape_3s_Trades_Volume_Total",  # ... and so on for all tape features
            "HF_Tape_3s_Trades_VWAP", "HF_Tape_3s_Tape_Speed",
            # HF_Tape (5s window) - Features would be similar, add suffix _5s
            "HF_Tape_5s_Trades_Count", "HF_Tape_5s_Trades_Volume_Total",  # ... and so on for all tape features
            "HF_Tape_5s_Trades_VWAP", "HF_Tape_5s_Tape_Speed",
            # HF_Quote (L1, 1s window)
            "HF_Quote_1s_MidPrice", "HF_Quote_1s_Spread_Abs", "HF_Quote_1s_Spread_Rel",
            "HF_Quote_1s_Bid_Size", "HF_Quote_1s_Ask_Size", "HF_Quote_1s_Quote_Imbalance_Size",
            "HF_Quote_1s_Is_Large_Bid", "HF_Quote_1s_Is_Large_Ask",
            "HF_Quote_1s_MidPrice_Change", "HF_Quote_1s_Spread_Change",
            # HF_Quote (L1, 3s window) - Suffix _3s
            "HF_Quote_3s_MidPrice", "HF_Quote_3s_Spread_Abs",  # ... and so on
            # HF_Quote (L1, 5s window) - Suffix _5s
            "HF_Quote_5s_MidPrice", "HF_Quote_5s_Spread_Abs",  # ... and so on
        ]  # This list needs to be very comprehensive based on Appendix

        self.mf_feature_names: List[str] = [
            # MF_1s_Data_Roll (window: 30s) - Suffix _30s
            "MF_SMA_30s", "MF_EMA_30s", "MF_VWAP_30s", "MF_StdDev_Price_30s",
            "MF_BB_Upper_30s", "MF_BB_Lower_30s", "MF_Price_ROC_30s",
            "MF_Dist_To_EMA_30s_Pct", "MF_Dist_To_VWAP_30s_Pct",
            "MF_Highest_Price_In_Window_30s", "MF_Lowest_Price_In_Window_30s",
            "MF_Dist_To_Highest_30s_Pct", "MF_Dist_To_Lowest_30s_Pct",
            "MF_RSI_30s", "MF_MACD_30s", "MF_MACD_Signal_30s", "MF_MACD_Hist_30s", "MF_ATR_1s_Avg_30s",
            # MF_1s_Data_Roll (window: 1m) - Suffix _1m (distinct from 1m bar features)
            "MF_SMA_1m", "MF_EMA_1m", "MF_VWAP_1m",  # ... and so on for all 1s_Data_Roll features
            "MF_RSI_1m",
            # ... other windows (5m, 10m, 20m)
            "MF_SMA_5m", "MF_EMA_5m",  # ...
            "MF_SMA_10m", "MF_EMA_10m",  # ...
            "MF_SMA_20m", "MF_EMA_20m",  # ...
            # MF_1s_Vol_Roll (windows: 30s, 1m, 5m, 10m, 20m) - Suffix _<window>
            "MF_Volume_Avg_30s", "MF_Volume_StdDev_30s", "MF_Volume_Sum_30s",
            "MF_Volume_Rel_To_DailyAvgFactor_30s", "MF_Volume_Price_Correlation_30s",
            "MF_Volume_Avg_1m",  # ... and so on for other windows
            # MF_1m_Bar_Agg (current forming _0 & last N_1m)
            "MF_1m_Open_0", "MF_1m_High_0", "MF_1m_Low_0", "MF_1m_Close_0", "MF_1m_Volume_0",
            "MF_1m_BodySize_0_Rel", "MF_1m_UpperWick_0_Rel", "MF_1m_LowerWick_0_Rel",
            "MF_1m_Is_ToppingTail_Last1", "MF_1m_Is_BottomingTail_Last1", "MF_1m_Is_Engulfing_Last1",
            # Example for last 1 completed
            "MF_1m_Volume_Rel_To_Avg_1m_Bars_Last1",  # Example for last 1 completed
            # ... potentially for Last2, Last3 etc. or an aggregation
            # MF_1m_Bar_Seq (last N_1m)
            "MF_1m_Trend_Slope_Closes_LastN", "MF_1m_Trend_Slope_Highs_LastN", "MF_1m_Trend_Slope_Lows_LastN",
            "MF_1m_High_LastN", "MF_1m_Low_LastN", "MF_1m_Consecutive_Up_Bars", "MF_1m_Consecutive_Down_Bars",
            "MF_1m_Avg_BodySize_LastN", "MF_1m_Pullback_LowVol_Flag_LastN",
            "MF_1m_EMA_Closes_LastN", "MF_1m_VWAP_Closes_LastN",
            "MF_1m_Dist_To_EMA_Closes_LastN_Pct", "MF_1m_Dist_To_VWAP_Closes_LastN_Pct",
            # MF_5m_Bar_Agg & MF_5m_Bar_Seq (similar to 1m)
            "MF_5m_Open_0", "MF_5m_High_0", "MF_5m_Low_0", "MF_5m_Close_0", "MF_5m_Volume_0",  # ... etc.
            "MF_5m_Trend_Slope_Closes_LastN",  # ... etc.
            # MF_SR_ShortTerm
            "MF_Dist_To_Recent_SwingHigh_1m_1_Pct", "MF_Dist_To_Recent_SwingLow_1m_1_Pct",  # k=1
            "MF_Dist_To_Recent_SwingHigh_5m_1_Pct", "MF_Dist_To_Recent_SwingLow_5m_1_Pct",  # k=1
            # ... for k=2, k=3 etc.
            "MF_Is_Near_1m_Swing_Level", "MF_Is_Near_5m_Swing_Level",
            "MF_Time_Since_Last_1m_Swing", "MF_Time_Since_Last_5m_Swing",
            "MF_Volume_At_Last_1m_Swing", "MF_Volume_At_Last_5m_Swing",
            "MF_Is_Forming_Base_1m", "MF_Is_Forming_Base_5m"
        ]  # This list needs to be very comprehensive

        self.lf_feature_names: List[str] = [
            "LF_Dist_To_PrevDay_High_Pct", "LF_Dist_To_PrevDay_Low_Pct", "LF_Dist_To_PrevDay_Close_Pct",
            "LF_Dist_To_Premarket_High_Pct", "LF_Dist_To_Premarket_Low_Pct",
            "LF_Dist_To_Session_Open_Pct",  # Added as session_open_price is available
            "LF_Dist_To_Session_High_Pct", "LF_Dist_To_Session_Low_Pct",
            "LF_Is_Breaking_PrevDay_High", "LF_Is_Breaking_Premarket_High",
            "LF_Session_Range_Expansion_Ratio_vs_PrevDayATR",
            "LF_Session_VWAP", "LF_Dist_To_Session_VWAP_Pct",
            "LF_Cumulative_Session_Volume_Rel_To_Historical_Avg",
            # Dist_To_LT_Daily_Resistance/Support_k_Pct
            # Example for 3 support/resistance levels
        ]
        for i in range(1, self.sr_num_levels + 1):
            self.lf_feature_names.append(f"LF_Dist_To_LT_Daily_Resistance_{i}_Pct")
            self.lf_feature_names.append(f"LF_Dist_To_LT_Daily_Support_{i}_Pct")

        # History deques for lookback features
        self.hf_history: deque = deque(maxlen=self.hf_lookback_length)
        self.mf_history: deque = deque(maxlen=self.mf_lookback_length)
        self.lf_history: deque = deque(maxlen=self.lf_lookback_length)

        # Latest calculated features
        self.latest_static_features: Optional[np.array] = None
        self.latest_timestamp: Optional[datetime] = None

        self.logger.info(f"FeatureExtractor initialized for {self.symbol} with "
                         f"HF lookback: {self.hf_lookback_length}, "
                         f"MF lookback: {self.mf_lookback_length}, "
                         f"LF lookback: {self.lf_lookback_length}.")
        self.logger.info(f"Num static features: {len(self.static_feature_names)}")
        self.logger.info(f"Num HF features per tick: {len(self.hf_feature_names)}")
        self.logger.info(f"Num MF features per tick: {len(self.mf_feature_names)}")
        self.logger.info(f"Num LF features per tick: {len(self.lf_feature_names)}")

    def _detect_significant_sr_levels(self, daily_df: pd.DataFrame, current_price: Optional[float], num_levels: int) -> \
    Dict[str, List[float]]:
        """
        Placeholder for detecting significant S/R levels from daily data.
        This should be a more sophisticated algorithm (e.g., clustering, pivot points, volume profiles).
        For now, it might just return recent highs/lows or pre-defined levels if available.

        Args:
            daily_df (pd.DataFrame): DataFrame of historical daily bars (OHLC).
            current_price (Optional[float]): Current market price to find relevant S/R around.
            num_levels (int): Number of support and resistance levels to return.

        Returns:
            Dict[str, List[float]]: {'support': [s1, s2,...], 'resistance': [r1, r2,...]}
        """
        # STUB: Implement actual S/R detection logic
        # This is a very naive placeholder.
        # Consider using pivot points, recent swing highs/lows over the period, volume profile levels, etc.
        if daily_df is None or daily_df.empty:
            return {'support': [np.nan] * num_levels, 'resistance': [np.nan] * num_levels}

        # Simplified: Take N highest highs and N lowest lows from the last `long_term_sr_days`
        recent_data = daily_df.tail(self.long_term_sr_days)
        if recent_data.empty:
            return {'support': [np.nan] * num_levels, 'resistance': [np.nan] * num_levels}

        supports = sorted(list(set(recent_data['low'].nsmallest(num_levels * 2).tolist())))  # *2 for more candidates
        resistances = sorted(list(set(recent_data['high'].nlargest(num_levels * 2).tolist())))

        # Crude selection based on current price if available
        if current_price is not None:
            final_supports = sorted([s for s in supports if s < current_price], reverse=True)[:num_levels]
            final_resistances = sorted([r for r in resistances if r > current_price])[:num_levels]
        else:
            final_supports = supports[:num_levels]
            final_resistances = resistances[:num_levels]

        # Pad with NaN if not enough levels found
        final_supports.extend([np.nan] * (num_levels - len(final_supports)))
        final_resistances.extend([np.nan] * (num_levels - len(final_resistances)))

        return {'support': final_supports, 'resistance': final_resistances}

    def update_daily_levels(self, full_historical_daily_df: Optional[pd.DataFrame],
                            current_processing_utc_dt: datetime):
        """
        Processes full_historical_daily_df to set self.prev_day_data (OHLC, ATR)
        and self.long_term_daily_sr (long-term support/resistance).
        This should be called once at the start of a trading day or when new daily data is available.

        Args:
            full_historical_daily_df (Optional[pd.DataFrame]): DataFrame of all available daily bars.
                                                                Expected columns: ['open', 'high', 'low', 'close', 'volume'].
                                                                Index should be DatetimeIndex.
            current_processing_utc_dt (datetime): The current processing timestamp (UTC).
                                                  Used to identify the previous trading day.
        """
        self.prev_day_data = None
        self.long_term_daily_sr = {'support': [np.nan] * self.sr_num_levels,
                                   'resistance': [np.nan] * self.sr_num_levels}

        if full_historical_daily_df is None or full_historical_daily_df.empty:
            self.logger.warning("Cannot update daily levels: full_historical_daily_df is None or empty.")
            return

        try:
            # Ensure DataFrame index is datetime
            if not isinstance(full_historical_daily_df.index, pd.DatetimeIndex):
                full_historical_daily_df.index = pd.to_datetime(full_historical_daily_df.index)

            # Sort by date just in case
            df_sorted = full_historical_daily_df.sort_index()

            # Find previous trading day's data
            # Filter out any data from the current_processing_utc_dt's date onwards
            # to ensure we are only looking at *completed* previous days.
            prev_day_candidates = df_sorted[df_sorted.index.normalize() < current_processing_utc_dt.normalize()]

            if not prev_day_candidates.empty:
                last_day_data = prev_day_candidates.iloc[-1]
                self.prev_day_data = {
                    'open': last_day_data['open'],
                    'high': last_day_data['high'],
                    'low': last_day_data['low'],
                    'close': last_day_data['close'],
                    'volume': last_day_data['volume'],
                    'timestamp': last_day_data.name  # Store the date of this prev_day_data
                }
                # Calculate ATR (e.g., 14-day ATR for the previous day)
                if len(prev_day_candidates) >= 15:  # Need 14 prior days + current for ATR(14)
                    tr = pd.DataFrame(index=prev_day_candidates.index)
                    tr['h-l'] = prev_day_candidates['high'] - prev_day_candidates['low']
                    tr['h-pc'] = abs(prev_day_candidates['high'] - prev_day_candidates['close'].shift(1))
                    tr['l-pc'] = abs(prev_day_candidates['low'] - prev_day_candidates['close'].shift(1))
                    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
                    self.prev_day_data['atr'] = tr['tr'].rolling(window=14).mean().iloc[-1]
                else:
                    self.prev_day_data['atr'] = np.nan
                self.logger.info(f"Previous day data set for {self.prev_day_data.get('timestamp')}: "
                                 f"H={self.prev_day_data['high']:.2f}, L={self.prev_day_data['low']:.2f}, "
                                 f"C={self.prev_day_data['close']:.2f}, ATR={self.prev_day_data.get('atr', 'N/A'):.2f}")

            else:
                self.logger.warning(f"No previous day data found before {current_processing_utc_dt.date()}.")

            # Set long-term S/R levels using data up to (but not including) current_processing_utc_dt's date
            self.historical_1year_daily_data = prev_day_candidates.copy()  # Store for reference if needed
            # Current price is not known at this stage of daily update, pass None or a recent close.
            last_close_for_sr = self.prev_day_data['close'] if self.prev_day_data else None
            self.long_term_daily_sr = self._detect_significant_sr_levels(
                self.historical_1year_daily_data,
                current_price=last_close_for_sr,
                num_levels=self.sr_num_levels
            )
            self.logger.info(f"Long-term daily S/R levels updated. "
                             f"Supports: {self.long_term_daily_sr['support']}, "
                             f"Resistances: {self.long_term_daily_sr['resistance']}")

        except Exception as e:
            self.logger.error(f"Error in update_daily_levels: {e}", exc_info=True)

    def calculate_features_and_get_model_input(self, market_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main method called every second to calculate features and assemble model input.

        Args:
            market_state (Dict[str, Any]): Input market state from MarketSimulator. Expected keys:
                'timestamp': datetime (current tick timestamp)
                'latest_1s_bar': Dict (OHLCV for current 1s bar)
                'rolling_1s_data_window': List[Dict] (recent 1s data events: 'bar', 'trades', 'quotes')
                'current_market_session': str ("PREMARKET", "REGULAR", etc.)
                'premarket_high': Optional[float]
                'premarket_low': Optional[float]
                'session_open_price': Optional[float] (for the active session)
                'session_high': Optional[float] (for the active session)
                'session_low': Optional[float] (for the active session)
                'current_1m_bar': Optional[Dict] (OHLCV of forming 1m bar)
                'current_5m_bar': Optional[Dict] (OHLCV of forming 5m bar)
                'completed_1m_bars_window': List[Dict] (rolling window of completed 1m bars)
                'completed_5m_bars_window': List[Dict] (rolling window of completed 5m bars)
                'historical_1d_bars_full': pd.DataFrame (long-term daily data for prev. day stats & long-term S/R).
                                                      (Note: This is usually passed to update_daily_levels,
                                                       but might be included for dynamic daily updates if system designed so.
                                                       For per-second, primary daily info comes from self.prev_day_data)

        Returns:
            Optional[Dict[str, Any]]: Model input dictionary if all history deques are full, else None.
                Format:
                'timestamp': datetime
                'static_features': np.array shape (N_static,)
                'hf_features': np.array shape (hf_lookback_length, N_hf)
                'mf_features': np.array shape (mf_lookback_length, N_mf)
                'lf_features': np.array shape (lf_lookback_length, N_lf)
        """
        try:
            # Step 0: Unpack market_state and basic checks
            current_ts: datetime = market_state.get('timestamp')
            if not current_ts:
                self.logger.warning("Timestamp missing in market_state.")
                return None
            self.latest_timestamp = current_ts

            latest_1s_bar: Optional[Dict] = market_state.get('latest_1s_bar')
            if not latest_1s_bar or 'close' not in latest_1s_bar:  # Require at least a close price for current tick
                self.logger.debug(
                    f"latest_1s_bar is missing or incomplete at {current_ts}. Skipping feature calculation.")
                return None
            current_price: float = latest_1s_bar['close']  # Primary current price for many calculations

            rolling_1s_data_window: List[Dict] = market_state.get('rolling_1s_data_window', [])
            current_market_session: str = market_state.get('current_market_session', 'UNKNOWN')

            # Update Session VWAP
            # Check if session changed to reset VWAP accumulators
            if self._current_market_session_for_vwap != current_market_session:
                self.logger.info(
                    f"Market session changed from {self._current_market_session_for_vwap} to {current_market_session}. Resetting session VWAP.")
                self.session_vwap_sum_price_volume = 0.0
                self.session_vwap_sum_volume = 0.0
                self.current_session_vwap = None
                self._current_market_session_for_vwap = current_market_session

            if latest_1s_bar and 'volume' in latest_1s_bar and latest_1s_bar['volume'] > 0:
                # Use typical price for VWAP calculation (e.g., (H+L+C)/3 or just C)
                # Assuming latest_1s_bar has 'high', 'low', 'close', 'volume'
                typical_price = (latest_1s_bar.get('high', current_price) + latest_1s_bar.get('low',
                                                                                              current_price) + current_price) / 3
                self.session_vwap_sum_price_volume += typical_price * latest_1s_bar['volume']
                self.session_vwap_sum_volume += latest_1s_bar['volume']
                if self.session_vwap_sum_volume > 0:
                    self.current_session_vwap = self.session_vwap_sum_price_volume / self.session_vwap_sum_volume
                else:
                    self.current_session_vwap = current_price  # Fallback if volume is zero

            # Step 1: Calculate Single-Tick Feature Vectors
            # Static features are calculated once or less frequently, but we ensure they are available.
            # They might be updated based on current_ts or current_market_session if they change (e.g. time-based static features)
            current_static_vector = self._calculate_static_features_vector(
                current_ts=current_ts,
                current_market_session=current_market_session,
                # Potentially pass other relevant static info like catalyst data from a manager class
            )
            self.latest_static_features = current_static_vector  # Update latest static features

            current_hf_vector = self._calculate_hf_features_vector(
                latest_1s_bar=latest_1s_bar,
                rolling_1s_data_window=rolling_1s_data_window,
                current_price=current_price
            )
            current_mf_vector = self._calculate_mf_features_vector(
                current_ts=current_ts,
                latest_1s_bar=latest_1s_bar,
                rolling_1s_data_window=rolling_1s_data_window,
                current_1m_bar=market_state.get('current_1m_bar'),
                completed_1m_bars=market_state.get('completed_1m_bars_window', []),
                current_5m_bar=market_state.get('current_5m_bar'),
                completed_5m_bars=market_state.get('completed_5m_bars_window', []),
                current_price=current_price
            )
            current_lf_vector = self._calculate_lf_features_vector(
                current_ts=current_ts,
                current_price=current_price,
                prev_day_data=self.prev_day_data,
                premarket_high=market_state.get('premarket_high'),
                premarket_low=market_state.get('premarket_low'),
                session_open_price=market_state.get('session_open_price'),
                session_high=market_state.get('session_high'),
                session_low=market_state.get('session_low'),
                current_session_vwap=self.current_session_vwap,
                long_term_daily_sr=self.long_term_daily_sr
            )

            # Step 2: Update History Deques
            if current_hf_vector is not None: self.hf_history.append(current_hf_vector)
            if current_mf_vector is not None: self.mf_history.append(current_mf_vector)
            if current_lf_vector is not None: self.lf_history.append(current_lf_vector)

            # Step 3: Check Readiness & Assemble Model Input
            if (len(self.hf_history) == self.hf_lookback_length and
                    len(self.mf_history) == self.mf_lookback_length and
                    len(self.lf_history) == self.lf_lookback_length and
                    self.latest_static_features is not None):

                model_input = {
                    'timestamp': current_ts,
                    'static_features': self.latest_static_features,
                    'hf_features': np.array(list(self.hf_history), dtype=np.float32),
                    'mf_features': np.array(list(self.mf_history), dtype=np.float32),
                    'lf_features': np.array(list(self.lf_history), dtype=np.float32),
                }
                return model_input
            else:
                # Not enough data in history deques yet
                return None

        except Exception as e:
            self.logger.error(
                f"Error in calculate_features_and_get_model_input at {market_state.get('timestamp')}: {e}",
                exc_info=True)
            # In case of error, append NaNs to keep shapes consistent if partial vectors were calculated?
            # Or simply return None. For now, return None.
            # To prevent breaking downstream, ensure deques might get placeholder NaNs if needed, but error handling is better.
            return None

    # --- Private Helper Methods for Single-Tick Feature Vector Calculation (Stubs) ---

    def _calculate_static_features_vector(self, current_ts: datetime, current_market_session: str) -> np.array:
        """
        Calculates the static feature vector for the current state.
        These features are typically constant or change very infrequently (e.g., daily).
        """
        feature_values_for_tick: Dict[str, Any] = {}

        # --- Static (S_) Features ---
        # S_Day_Of_Week_Encoded: (uses current_ts) e.g., one-hot encode current_ts.weekday()
        # Example: Monday=0, Tuesday=1, ... Friday=4
        # feature_values_for_tick["S_Day_Of_Week_Encoded_0"] = 1 if current_ts.weekday() == 0 else 0
        # ... for all 5 days

        # S_Time_Of_Day_Seconds_Encoded (cyclical): (uses current_ts)
        # seconds_in_day = current_ts.hour * 3600 + current_ts.minute * 60 + current_ts.second
        # feature_values_for_tick["S_Time_Of_Day_Seconds_Encoded_Sin"] = np.sin(2 * np.pi * seconds_in_day / (24 * 3600))
        # feature_values_for_tick["S_Time_Of_Day_Seconds_Encoded_Cos"] = np.cos(2 * np.pi * seconds_in_day / (24 * 3600))

        # S_Market_Session_Encoded: (uses current_market_session) e.g., one-hot encode
        # feature_values_for_tick["S_Market_Session_Encoded_PREMARKET"] = 1 if current_market_session == "PREMARKET" else 0
        # ... for REGULAR, POSTMARKET, OTHER

        # S_Stock_Float_Category: (external data, assumed fixed for the stock or updated daily)
        # This would come from an external source, e.g., loaded at init or daily.
        # feature_values_for_tick["S_Stock_Float_Category"] = self.stock_float_category_value (e.g. 0 for low, 1 for mid, 2 for high)

        # S_Has_Catalyst: (external data, potentially updated)
        # feature_values_for_tick["S_Has_Catalyst"] = self.catalyst_info.get('has_catalyst', 0) (binary)

        # S_Days_Since_Catalyst: (external data, current_ts)
        # if self.catalyst_info.get('date'):
        #   days_since = (current_ts.date() - self.catalyst_info['date']).days
        #   feature_values_for_tick["S_Days_Since_Catalyst"] = days_since
        # else:
        #   feature_values_for_tick["S_Days_Since_Catalyst"] = -1 # or a large number for no catalyst

        # S_Initial_Scan_RelVol: (external data, from the initial scan that identified the stock)
        # feature_values_for_tick["S_Initial_Scan_RelVol"] = self.scan_data.get('relative_volume_at_scan_time')

        # S_Initial_Scan_Price_ROC: (external data, from initial scan)
        # feature_values_for_tick["S_Initial_Scan_Price_ROC"] = self.scan_data.get('price_roc_at_scan_time')

        # --- End of Static Features ---

        # Convert to ordered NumPy array
        # Ensure all features in self.static_feature_names are present, using np.nan for missing ones.
        # For static features, they should ideally always be present once initialized.
        return np.array([feature_values_for_tick.get(name, np.nan) for name in self.static_feature_names],
                        dtype=np.float32)

    def _calculate_hf_features_vector(self, latest_1s_bar: Dict, rolling_1s_data_window: List[Dict],
                                      current_price: float) -> np.array:
        """
        Calculates the High-Frequency (HF) feature vector for the current 1-second tick.
        """
        feature_values_for_tick: Dict[str, Any] = {}
        # rolling_1s_data_window is a list of dicts, each dict: {'timestamp': ..., 'bar': ..., 'trades': [...], 'quotes': [...]}
        # latest_1s_bar is the current bar, also typically the last element in rolling_1s_data_window[-1]['bar']

        # --- HF_1s_Bar Features (uses latest_1s_bar) ---
        # feature_values_for_tick["HF_1s_ClosePrice"] = latest_1s_bar.get('close')
        # feature_values_for_tick["HF_1s_PriceChange_Abs"] = latest_1s_bar.get('close') - latest_1s_bar.get('open') # Or change from prev close
        # feature_values_for_tick["HF_1s_PriceChange_Pct"] = (price_change_abs / latest_1s_bar.get('open')) * 100 if latest_1s_bar.get('open') else 0
        # feature_values_for_tick["HF_1s_Return_Log"] = np.log(latest_1s_bar.get('close') / latest_1s_bar.get('open')) if latest_1s_bar.get('open') and latest_1s_bar.get('close') > 0 else 0
        # feature_values_for_tick["HF_1s_Volume"] = latest_1s_bar.get('volume')
        # spread_abs = latest_1s_bar.get('high') - latest_1s_bar.get('low')
        # feature_values_for_tick["HF_1s_HighLow_Spread_Abs"] = spread_abs
        # feature_values_for_tick["HF_1s_HighLow_Spread_Rel"] = (spread_abs / latest_1s_bar.get('low')) * 100 if latest_1s_bar.get('low') else 0
        # feature_values_for_tick["HF_1s_Close_Location_In_Bar"] = ((latest_1s_bar.get('close') - latest_1s_bar.get('low')) / spread_abs) if spread_abs > 0 else 0.5
        # prev_bar_close = rolling_1s_data_window[-2]['bar']['close'] if len(rolling_1s_data_window) > 1 and rolling_1s_data_window[-2].get('bar') else None
        # if prev_bar_close and latest_1s_bar.get('open'):
        #   feature_values_for_tick["HF_1s_Is_Gap_Up_Bar"] = 1 if latest_1s_bar['open'] > prev_bar_close else 0
        #   feature_values_for_tick["HF_1s_Is_Gap_Down_Bar"] = 1 if latest_1s_bar['open'] < prev_bar_close else 0

        # --- HF_Tape (windows: 1s, 3s, 5s) ---
        # (uses 'trades' from rolling_1s_data_window for the respective time windows)
        # Example for 1s window (typically trades from latest_1s_bar's interval, i.e., rolling_1s_data_window[-1]['trades'])
        # trades_1s = rolling_1s_data_window[-1].get('trades', []) if rolling_1s_data_window else []
        # feature_values_for_tick["HF_Tape_1s_Trades_Count"] = len(trades_1s)
        # feature_values_for_tick["HF_Tape_1s_Trades_Volume_Total"] = sum(t['size'] for t in trades_1s)
        # buy_volume = sum(t['size'] for t in trades_1s if t.get('side') == 'BUY') # Assuming 'side' exists
        # sell_volume = sum(t['size'] for t in trades_1s if t.get('side') == 'SELL')
        # feature_values_for_tick["HF_Tape_1s_Trades_Volume_Buy"] = buy_volume
        # feature_values_for_tick["HF_Tape_1s_Trades_Volume_Sell"] = sell_volume
        # feature_values_for_tick["HF_Tape_1s_Trades_Imbalance_Volume"] = buy_volume - sell_volume
        # feature_values_for_tick["HF_Tape_1s_Trades_Imbalance_Count"] = sum(1 for t in trades_1s if t.get('side') == 'BUY') - sum(1 for t in trades_1s if t.get('side') == 'SELL')
        # vwap_sum_price_vol = sum(t['price'] * t['size'] for t in trades_1s)
        # total_vol = feature_values_for_tick["HF_Tape_1s_Trades_Volume_Total"]
        # feature_values_for_tick["HF_Tape_1s_Trades_VWAP"] = vwap_sum_price_vol / total_vol if total_vol > 0 else current_price
        # feature_values_for_tick["HF_Tape_1s_Avg_Trade_Size"] = total_vol / len(trades_1s) if len(trades_1s) > 0 else 0
        # large_trade_threshold = ... (define based on stock price/vol)
        # feature_values_for_tick["HF_Tape_1s_Large_Trade_Count"] = sum(1 for t in trades_1s if t['size'] > large_trade_threshold)
        # feature_values_for_tick["HF_Tape_1s_Large_Trade_Imbalance"] = ...
        # feature_values_for_tick["HF_Tape_1s_Tape_Speed"] = len(trades_1s) # Trades per second (already count for 1s window)
        # trade_prices = [t['price'] for t in trades_1s]
        # feature_values_for_tick["HF_Tape_1s_Trade_Price_StdDev"] = np.std(trade_prices) if len(trade_prices) > 1 else 0

        # For 3s and 5s windows, aggregate trades from the last 3 or 5 elements of rolling_1s_data_window
        # Example for 3s: trades_3s = []
        # for event in rolling_1s_data_window[-3:]: trades_3s.extend(event.get('trades',[]))
        # Then calculate all tape features similar to above, e.g., HF_Tape_3s_Trades_Count = len(trades_3s)

        # --- HF_Quote (L1, windows: 1s, 3s, 5s) ---
        # (uses 'quotes' from rolling_1s_data_window for the respective time windows, often the last quote in each second)
        # Example for 1s window (latest quote in the last second)
        # latest_quotes_1s = rolling_1s_data_window[-1].get('quotes', []) if rolling_1s_data_window else []
        # if latest_quotes_1s:
        #   last_quote = latest_quotes_1s[-1] # Assuming last quote in the second is most relevant
        #   bid_price = last_quote.get('bid_price')
        #   ask_price = last_quote.get('ask_price')
        #   feature_values_for_tick["HF_Quote_1s_MidPrice"] = (bid_price + ask_price) / 2 if bid_price and ask_price else current_price
        #   spread_abs = ask_price - bid_price if bid_price and ask_price else 0
        #   feature_values_for_tick["HF_Quote_1s_Spread_Abs"] = spread_abs
        #   feature_values_for_tick["HF_Quote_1s_Spread_Rel"] = (spread_abs / mid_price) * 100 if mid_price else 0
        #   feature_values_for_tick["HF_Quote_1s_Bid_Size"] = last_quote.get('bid_size')
        #   feature_values_for_tick["HF_Quote_1s_Ask_Size"] = last_quote.get('ask_size')
        #   feature_values_for_tick["HF_Quote_1s_Quote_Imbalance_Size"] = last_quote.get('bid_size') - last_quote.get('ask_size')
        #   large_size_threshold = ...
        #   feature_values_for_tick["HF_Quote_1s_Is_Large_Bid"] = 1 if last_quote.get('bid_size',0) > large_size_threshold else 0
        #   feature_values_for_tick["HF_Quote_1s_Is_Large_Ask"] = 1 if last_quote.get('ask_size',0) > large_size_threshold else 0
        #   # MidPrice_Change and Spread_Change need previous tick's quote data
        #   prev_quote_event = rolling_1s_data_window[-2].get('quotes', []) if len(rolling_1s_data_window) > 1 else None
        #   if prev_quote_event and prev_quote_event[-1]:
        #       prev_mid_price = (prev_quote_event[-1]['bid_price'] + prev_quote_event[-1]['ask_price']) / 2
        #       feature_values_for_tick["HF_Quote_1s_MidPrice_Change"] = feature_values_for_tick["HF_Quote_1s_MidPrice"] - prev_mid_price
        #       prev_spread = prev_quote_event[-1]['ask_price'] - prev_quote_event[-1]['bid_price']
        #       feature_values_for_tick["HF_Quote_1s_Spread_Change"] = feature_values_for_tick["HF_Quote_1s_Spread_Abs"] - prev_spread

        # For 3s and 5s windows, might use stats of quotes over window (e.g. avg midprice, avg spread) or last quote of window.
        # Example for 3s: quotes_3s_events = [ev.get('quotes',[]) for ev in rolling_1s_data_window[-3:]]
        # last_quote_in_3s_window = quotes_3s_events[-1][-1] if quotes_3s_events[-1] else None
        # Then calculate quote features as above, e.g. feature_values_for_tick["HF_Quote_3s_MidPrice"] = ...

        # --- End of HF Features ---

        return np.array([feature_values_for_tick.get(name, np.nan) for name in self.hf_feature_names], dtype=np.float32)

    def _calculate_mf_features_vector(self, current_ts: datetime, latest_1s_bar: Dict,
                                      rolling_1s_data_window: List[Dict],
                                      current_1m_bar: Optional[Dict], completed_1m_bars: List[Dict],
                                      current_5m_bar: Optional[Dict], completed_5m_bars: List[Dict],
                                      current_price: float) -> np.array:
        """
        Calculates the Medium-Frequency (MF) feature vector for the current tick.
        """
        feature_values_for_tick: Dict[str, Any] = {}
        # rolling_1s_data_window contains 1-second level bar/trade/quote events.
        # completed_1m_bars and completed_5m_bars are lists of completed OHLCV bars.

        # --- MF_1s_Data_Roll (windows: 30s, 1m, 5m, 10m, 20m) ---
        # (uses 'bar' data, typically 'close' prices, from rolling_1s_data_window)
        # Example for 30s window:
        # prices_30s = [event['bar']['close'] for event in rolling_1s_data_window[-30:] if event.get('bar')]
        # volumes_30s = [event['bar']['volume'] for event in rolling_1s_data_window[-30:] if event.get('bar')]
        # if len(prices_30s) >= N_min_for_calc: # N_min_for_calc e.g. 5 or 10
        #   feature_values_for_tick["MF_SMA_30s"] = np.mean(prices_30s)
        #   # feature_values_for_tick["MF_EMA_30s"] = calculate_ema(prices_30s, period=...) # needs state or full series
        #   # feature_values_for_tick["MF_VWAP_30s"] = np.sum(np.array(prices_30s) * np.array(volumes_30s)) / np.sum(volumes_30s) if np.sum(volumes_30s) > 0 else current_price
        #   std_dev = np.std(prices_30s)
        #   feature_values_for_tick["MF_StdDev_Price_30s"] = std_dev
        #   feature_values_for_tick["MF_BB_Upper_30s"] = feature_values_for_tick["MF_SMA_30s"] + 2 * std_dev
        #   feature_values_for_tick["MF_BB_Lower_30s"] = feature_values_for_tick["MF_SMA_30s"] - 2 * std_dev
        #   feature_values_for_tick["MF_Price_ROC_30s"] = (prices_30s[-1] - prices_30s[0]) / prices_30s[0] * 100 if prices_30s[0] else 0
        #   # feature_values_for_tick["MF_Dist_To_EMA_30s_Pct"] = (current_price - feature_values_for_tick["MF_EMA_30s"]) / feature_values_for_tick["MF_EMA_30s"] * 100 if feature_values_for_tick["MF_EMA_30s"] else 0
        #   feature_values_for_tick["MF_Highest_Price_In_Window_30s"] = np.max(prices_30s)
        #   feature_values_for_tick["MF_Lowest_Price_In_Window_30s"] = np.min(prices_30s)
        #   # feature_values_for_tick["MF_RSI_30s"] = calculate_rsi(prices_30s, period=...) # needs state or full series
        #   # feature_values_for_tick["MF_MACD_30s"], MF_MACD_Signal_30s, MF_MACD_Hist_30s = calculate_macd(prices_30s, ...)
        #   # atr_1s_values = [ev['bar']['high'] - ev['bar']['low'] for ev in rolling_1s_data_window[-30:] if ev.get('bar')] # Simplified ATR from H-L
        #   # feature_values_for_tick["MF_ATR_1s_Avg_30s"] = np.mean(atr_1s_values) if atr_1s_values else 0
        # Repeat for 1m, 5m, 10m, 20m windows similarly.

        # --- MF_1s_Vol_Roll (windows: as above) ---
        # (uses 'volume' from 'bar' data in rolling_1s_data_window)
        # Example for 30s window (volumes_30s already extracted above):
        # if len(volumes_30s) >= N_min_for_calc:
        #   feature_values_for_tick["MF_Volume_Avg_30s"] = np.mean(volumes_30s)
        #   feature_values_for_tick["MF_Volume_StdDev_30s"] = np.std(volumes_30s)
        #   feature_values_for_tick["MF_Volume_Sum_30s"] = np.sum(volumes_30s)
        #   # feature_values_for_tick["MF_Volume_Rel_To_DailyAvgFactor_30s"] = (np.mean(volumes_30s) * (24*3600/30)) / self.avg_daily_volume_1s_tick if self.avg_daily_volume_1s_tick else 0
        #   # feature_values_for_tick["MF_Volume_Price_Correlation_30s"] = np.corrcoef(prices_30s, volumes_30s)[0,1] if len(prices_30s) > 1 and len(volumes_30s) > 1 else 0
        # Repeat for other windows.

        # --- MF_1m_Bar_Agg (current forming bar _0 & last N_1m from completed_1m_bars_window) ---
        # (uses current_1m_bar for _0, and completed_1m_bars for historical)
        # if current_1m_bar:
        #   feature_values_for_tick["MF_1m_Open_0"] = current_1m_bar.get('open')
        #   feature_values_for_tick["MF_1m_High_0"] = current_1m_bar.get('high')
        #   # ... Close_0, Volume_0
        #   body_size = abs(current_1m_bar.get('close',0) - current_1m_bar.get('open',0))
        #   total_range = current_1m_bar.get('high',0) - current_1m_bar.get('low',0)
        #   # feature_values_for_tick["MF_1m_BodySize_0_Rel"] = body_size / total_range if total_range > 0 else 0
        #   # feature_values_for_tick["MF_1m_UpperWick_0_Rel"] = (current_1m_bar.get('high',0) - max(current_1m_bar.get('open',0), current_1m_bar.get('close',0))) / total_range if total_range > 0 else 0
        # if completed_1m_bars: # Example for last 1 completed bar
        #   last_1m_bar = completed_1m_bars[-1]
        #   # feature_values_for_tick["MF_1m_Is_ToppingTail_Last1"] = check_topping_tail(last_1m_bar)
        #   # feature_values_for_tick["MF_1m_Is_Engulfing_Last1"] = check_engulfing(last_1m_bar, completed_1m_bars[-2] if len(completed_1m_bars)>1 else None)
        #   # avg_vol_1m_bars = np.mean([b['volume'] for b in completed_1m_bars[-10:]]) # Avg vol of last 10 1m bars
        #   # feature_values_for_tick["MF_1m_Volume_Rel_To_Avg_1m_Bars_Last1"] = last_1m_bar['volume'] / avg_vol_1m_bars if avg_vol_1m_bars > 0 else 0

        # --- MF_1m_Bar_Seq (last N_1m from completed_1m_bars_window) ---
        # (uses completed_1m_bars)
        # N_1m_bars = completed_1m_bars[-10:] # Example: use last 10 completed 1m bars
        # if len(N_1m_bars) >= 2:
        #   closes_1m = [b['close'] for b in N_1m_bars]
        #   # feature_values_for_tick["MF_1m_Trend_Slope_Closes_LastN"] = calculate_slope(closes_1m)
        #   # feature_values_for_tick["MF_1m_High_LastN"] = max(b['high'] for b in N_1m_bars)
        #   # feature_values_for_tick["MF_1m_Consecutive_Up_Bars"] = count_consecutive_up_bars(N_1m_bars)
        #   # ... and other sequence features

        # --- MF_5m_Bar_Agg & MF_5m_Bar_Seq ---
        # Similar to 1m features, but using current_5m_bar and completed_5m_bars_window.

        # --- MF_SR_ShortTerm ---
        # (uses completed_1m_bars, completed_5m_bars to find recent swing highs/lows)
        # swing_highs_1m = find_swing_highs(completed_1m_bars, k=...) # k is number of swings
        # if swing_highs_1m:
        #   # feature_values_for_tick["MF_Dist_To_Recent_SwingHigh_1m_1_Pct"] = (current_price - swing_highs_1m[0]['price']) / swing_highs_1m[0]['price'] * 100 if swing_highs_1m[0]['price'] else 0
        # # feature_values_for_tick["MF_Is_Near_1m_Swing_Level"] = 1 if abs_dist_to_swing < threshold else 0
        # # feature_values_for_tick["MF_Time_Since_Last_1m_Swing"] = (current_ts - swing_highs_1m[0]['timestamp']).total_seconds()
        # # feature_values_for_tick["MF_Is_Forming_Base_1m"] = check_base_formation(completed_1m_bars, current_price)

        # --- End of MF Features ---

        return np.array([feature_values_for_tick.get(name, np.nan) for name in self.mf_feature_names], dtype=np.float32)

    def _calculate_lf_features_vector(self, current_ts: datetime, current_price: float,
                                      prev_day_data: Optional[Dict],
                                      premarket_high: Optional[float], premarket_low: Optional[float],
                                      session_open_price: Optional[float], session_high: Optional[float],
                                      session_low: Optional[float],
                                      current_session_vwap: Optional[float],
                                      long_term_daily_sr: Optional[Dict]) -> np.array:
        """
        Calculates the Low-Frequency (LF) feature vector for the current tick.
        """
        feature_values_for_tick: Dict[str, Any] = {}

        # --- LF_Dist_To_PrevDay_High/Low/Close_Pct ---
        # (uses self.prev_day_data which is set by update_daily_levels)
        # if prev_day_data and prev_day_data.get('close') is not None:
        #   feature_values_for_tick["LF_Dist_To_PrevDay_High_Pct"] = (current_price - prev_day_data['high']) / prev_day_data['high'] * 100 if prev_day_data.get('high') else np.nan
        #   feature_values_for_tick["LF_Dist_To_PrevDay_Low_Pct"] = (current_price - prev_day_data['low']) / prev_day_data['low'] * 100 if prev_day_data.get('low') else np.nan
        #   feature_values_for_tick["LF_Dist_To_PrevDay_Close_Pct"] = (current_price - prev_day_data['close']) / prev_day_data['close'] * 100 if prev_day_data.get('close') else np.nan

        # --- LF_Dist_To_Premarket_High/Low_Pct ---
        # (uses premarket_high, premarket_low from market_state)
        # if premarket_high is not None:
        #   feature_values_for_tick["LF_Dist_To_Premarket_High_Pct"] = (current_price - premarket_high) / premarket_high * 100
        # if premarket_low is not None:
        #   feature_values_for_tick["LF_Dist_To_Premarket_Low_Pct"] = (current_price - premarket_low) / premarket_low * 100

        # --- LF_Dist_To_Session_Open/High/Low_Pct ---
        # (uses session_open_price, session_high, session_low from market_state)
        # if session_open_price is not None:
        #   feature_values_for_tick["LF_Dist_To_Session_Open_Pct"] = (current_price - session_open_price) / session_open_price * 100
        # if session_high is not None:
        #   feature_values_for_tick["LF_Dist_To_Session_High_Pct"] = (current_price - session_high) / session_high * 100
        # if session_low is not None:
        #   feature_values_for_tick["LF_Dist_To_Session_Low_Pct"] = (current_price - session_low) / session_low * 100

        # --- LF_Is_Breaking_PrevDay/Premarket_High ---
        # if prev_day_data and prev_day_data.get('high') is not None:
        #   feature_values_for_tick["LF_Is_Breaking_PrevDay_High"] = 1 if current_price > prev_day_data['high'] else 0
        # if premarket_high is not None:
        #   feature_values_for_tick["LF_Is_Breaking_Premarket_High"] = 1 if current_price > premarket_high else 0

        # --- LF_Session_Range_Expansion_Ratio (vs. PrevDay ATR) ---
        # if session_high is not None and session_low is not None and prev_day_data and prev_day_data.get('atr') and prev_day_data['atr'] > 0:
        #   current_session_range = session_high - session_low
        #   feature_values_for_tick["LF_Session_Range_Expansion_Ratio_vs_PrevDayATR"] = current_session_range / prev_day_data['atr']

        # --- LF_Session_VWAP, LF_Dist_To_Session_VWAP_Pct ---
        # (Session VWAP is calculated in calculate_features_and_get_model_input and passed as current_session_vwap)
        # feature_values_for_tick["LF_Session_VWAP"] = current_session_vwap
        # if current_session_vwap is not None and current_session_vwap > 0 :
        #   feature_values_for_tick["LF_Dist_To_Session_VWAP_Pct"] = (current_price - current_session_vwap) / current_session_vwap * 100

        # --- LF_Cumulative_Session_Volume_Rel_To_Historical_Avg ---
        # (needs historical hourly volume profile for current time of day, and cumulative session volume from market_state)
        # cumulative_session_volume = market_state.get('cumulative_session_volume') # Assuming this is provided
        # historical_avg_volume_for_this_time = get_historical_avg_volume(current_ts.time()) # Helper needed
        # if cumulative_session_volume is not None and historical_avg_volume_for_this_time is not None and historical_avg_volume_for_this_time > 0:
        #   feature_values_for_tick["LF_Cumulative_Session_Volume_Rel_To_Historical_Avg"] = cumulative_session_volume / historical_avg_volume_for_this_time

        # --- LF_Dist_To_LT_Daily_Resistance/Support_k_Pct ---
        # (uses self.long_term_daily_sr which is set by update_daily_levels)
        # if long_term_daily_sr:
        #   for i, r_level in enumerate(long_term_daily_sr.get('resistance', [])):
        #     if i < self.sr_num_levels and r_level is not np.nan and r_level > 0:
        #       feature_values_for_tick[f"LF_Dist_To_LT_Daily_Resistance_{i+1}_Pct"] = (current_price - r_level) / r_level * 100
        #     else: feature_values_for_tick[f"LF_Dist_To_LT_Daily_Resistance_{i+1}_Pct"] = np.nan # Or a large default value
        #   for i, s_level in enumerate(long_term_daily_sr.get('support', [])):
        #     if i < self.sr_num_levels and s_level is not np.nan and s_level > 0:
        #       feature_values_for_tick[f"LF_Dist_To_LT_Daily_Support_{i+1}_Pct"] = (current_price - s_level) / s_level * 100
        #     else: feature_values_for_tick[f"LF_Dist_To_LT_Daily_Support_{i+1}_Pct"] = np.nan # Or a large default value

        # --- End of LF Features ---

        return np.array([feature_values_for_tick.get(name, np.nan) for name in self.lf_feature_names], dtype=np.float32)

    def get_feature_names_by_category(self) -> Dict[str, List[str]]:
        """
        Returns the dictionary of the ordered feature name lists.
        """
        return {
            "static": self.static_feature_names,
            "hf": self.hf_feature_names,
            "mf": self.mf_feature_names,
            "lf": self.lf_feature_names,
        }
