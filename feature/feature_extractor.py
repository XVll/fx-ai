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
        self.static_feature_names: List[str] = [
            "S_Day_Of_Week_Encoded_0", "S_Day_Of_Week_Encoded_1", "S_Day_Of_Week_Encoded_2",
            "S_Day_Of_Week_Encoded_3", "S_Day_Of_Week_Encoded_4",
            "S_Time_Of_Day_Seconds_Encoded_Sin", "S_Time_Of_Day_Seconds_Encoded_Cos",
            "S_Market_Session_Encoded_PREMARKET", "S_Market_Session_Encoded_REGULAR",
            "S_Market_Session_Encoded_POSTMARKET", "S_Market_Session_Encoded_OTHER",
            "S_Stock_Float_Category",
            "S_Has_Catalyst",
            "S_Days_Since_Catalyst",
            "S_Initial_Scan_RelVol",
            "S_Initial_Scan_Price_ROC"
        ]

        self.hf_feature_names: List[str] = [
            "HF_1s_ClosePrice", "HF_1s_PriceChange_Abs", "HF_1s_PriceChange_Pct",
            "HF_1s_Return_Log", "HF_1s_Volume", "HF_1s_HighLow_Spread_Abs",
            "HF_1s_HighLow_Spread_Rel", "HF_1s_Close_Location_In_Bar",
            "HF_1s_Is_Gap_Up_Bar", "HF_1s_Is_Gap_Down_Bar",
            "HF_Tape_1s_Trades_Count", "HF_Tape_1s_Trades_Volume_Total",
            "HF_Tape_1s_Trades_Volume_Buy", "HF_Tape_1s_Trades_Volume_Sell",
            "HF_Tape_1s_Trades_Imbalance_Volume", "HF_Tape_1s_Trades_Imbalance_Count",
            "HF_Tape_1s_Trades_VWAP", "HF_Tape_1s_Avg_Trade_Size",
            "HF_Tape_1s_Large_Trade_Count", "HF_Tape_1s_Large_Trade_Imbalance",
            "HF_Tape_1s_Tape_Speed", "HF_Tape_1s_Trade_Price_StdDev",
            "HF_Tape_3s_Trades_Count", "HF_Tape_3s_Trades_Volume_Total",
            "HF_Tape_3s_Trades_VWAP", "HF_Tape_3s_Tape_Speed",
            "HF_Tape_5s_Trades_Count", "HF_Tape_5s_Trades_Volume_Total",
            "HF_Tape_5s_Trades_VWAP", "HF_Tape_5s_Tape_Speed",
            "HF_Quote_1s_MidPrice", "HF_Quote_1s_Spread_Abs", "HF_Quote_1s_Spread_Rel",
            "HF_Quote_1s_Bid_Size", "HF_Quote_1s_Ask_Size", "HF_Quote_1s_Quote_Imbalance_Size",
            "HF_Quote_1s_Is_Large_Bid", "HF_Quote_1s_Is_Large_Ask",
            "HF_Quote_1s_MidPrice_Change", "HF_Quote_1s_Spread_Change",
            "HF_Quote_3s_MidPrice", "HF_Quote_3s_Spread_Abs",
            "HF_Quote_5s_MidPrice", "HF_Quote_5s_Spread_Abs",
        ]

        self.mf_feature_names: List[str] = [
            "MF_SMA_30s", "MF_EMA_30s", "MF_VWAP_30s", "MF_StdDev_Price_30s",
            "MF_BB_Upper_30s", "MF_BB_Lower_30s", "MF_Price_ROC_30s",
            "MF_Dist_To_EMA_30s_Pct", "MF_Dist_To_VWAP_30s_Pct",
            "MF_Highest_Price_In_Window_30s", "MF_Lowest_Price_In_Window_30s",
            "MF_Dist_To_Highest_30s_Pct", "MF_Dist_To_Lowest_30s_Pct",
            "MF_RSI_30s", "MF_MACD_30s", "MF_MACD_Signal_30s", "MF_MACD_Hist_30s", "MF_ATR_1s_Avg_30s",
            "MF_SMA_1m", "MF_EMA_1m", "MF_VWAP_1m", "MF_RSI_1m",
            "MF_SMA_5m", "MF_EMA_5m",
            "MF_SMA_10m", "MF_EMA_10m",
            "MF_SMA_20m", "MF_EMA_20m",
            "MF_Volume_Avg_30s", "MF_Volume_StdDev_30s", "MF_Volume_Sum_30s",
            "MF_Volume_Rel_To_DailyAvgFactor_30s", "MF_Volume_Price_Correlation_30s",
            "MF_Volume_Avg_1m",
            "MF_1m_Open_0", "MF_1m_High_0", "MF_1m_Low_0", "MF_1m_Close_0", "MF_1m_Volume_0",
            "MF_1m_BodySize_0_Rel", "MF_1m_UpperWick_0_Rel", "MF_1m_LowerWick_0_Rel",
            "MF_1m_Is_ToppingTail_Last1", "MF_1m_Is_BottomingTail_Last1", "MF_1m_Is_Engulfing_Last1",
            "MF_1m_Volume_Rel_To_Avg_1m_Bars_Last1",
            "MF_1m_Trend_Slope_Closes_LastN", "MF_1m_Trend_Slope_Highs_LastN", "MF_1m_Trend_Slope_Lows_LastN",
            "MF_1m_High_LastN", "MF_1m_Low_LastN", "MF_1m_Consecutive_Up_Bars", "MF_1m_Consecutive_Down_Bars",
            "MF_1m_Avg_BodySize_LastN", "MF_1m_Pullback_LowVol_Flag_LastN",
            "MF_1m_EMA_Closes_LastN", "MF_1m_VWAP_Closes_LastN",
            "MF_1m_Dist_To_EMA_Closes_LastN_Pct", "MF_1m_Dist_To_VWAP_Closes_LastN_Pct",
            "MF_5m_Open_0", "MF_5m_High_0", "MF_5m_Low_0", "MF_5m_Close_0", "MF_5m_Volume_0",
            "MF_5m_Trend_Slope_Closes_LastN",
            "MF_Dist_To_Recent_SwingHigh_1m_1_Pct", "MF_Dist_To_Recent_SwingLow_1m_1_Pct",
            "MF_Dist_To_Recent_SwingHigh_5m_1_Pct", "MF_Dist_To_Recent_SwingLow_5m_1_Pct",
            "MF_Is_Near_1m_Swing_Level", "MF_Is_Near_5m_Swing_Level",
            "MF_Time_Since_Last_1m_Swing", "MF_Time_Since_Last_5m_Swing",
            "MF_Volume_At_Last_1m_Swing", "MF_Volume_At_Last_5m_Swing",
            "MF_Is_Forming_Base_1m", "MF_Is_Forming_Base_5m"
        ]

        self.lf_feature_names: List[str] = [
            "LF_Dist_To_PrevDay_High_Pct", "LF_Dist_To_PrevDay_Low_Pct", "LF_Dist_To_PrevDay_Close_Pct",
            "LF_Dist_To_Intraday_High_Pct",  # Changed
            "LF_Dist_To_Intraday_Low_Pct",  # Changed
            "LF_Is_Breaking_PrevDay_High",
            "LF_Session_Range_Expansion_Ratio_vs_PrevDayATR",  # Uses intraday H/L now
            "LF_Session_VWAP", "LF_Dist_To_Session_VWAP_Pct",
            "LF_Cumulative_Session_Volume_Rel_To_Historical_Avg",
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
        if daily_df is None or daily_df.empty:
            return {'support': [np.nan] * num_levels, 'resistance': [np.nan] * num_levels}
        recent_data = daily_df.tail(self.long_term_sr_days)
        if recent_data.empty:
            return {'support': [np.nan] * num_levels, 'resistance': [np.nan] * num_levels}

        supports = sorted(list(set(recent_data['low'].nsmallest(num_levels * 2).tolist())))
        resistances = sorted(list(set(recent_data['high'].nlargest(num_levels * 2).tolist())))

        if current_price is not None:
            final_supports = sorted([s for s in supports if s < current_price], reverse=True)[:num_levels]
            final_resistances = sorted([r for r in resistances if r > current_price])[:num_levels]
        else:
            final_supports = supports[:num_levels]
            final_resistances = resistances[:num_levels]

        final_supports.extend([np.nan] * (num_levels - len(final_supports)))
        final_resistances.extend([np.nan] * (num_levels - len(final_resistances)))
        return {'support': final_supports, 'resistance': final_resistances}

    def update_daily_levels(self, full_historical_daily_df: Optional[pd.DataFrame],
                            current_processing_utc_dt: datetime):
        self.prev_day_data = None
        self.long_term_daily_sr = {'support': [np.nan] * self.sr_num_levels,
                                   'resistance': [np.nan] * self.sr_num_levels}
        if full_historical_daily_df is None or full_historical_daily_df.empty:
            self.logger.warning("Cannot update daily levels: full_historical_daily_df is None or empty.")
            return
        try:
            if not isinstance(full_historical_daily_df.index, pd.DatetimeIndex):
                full_historical_daily_df.index = pd.to_datetime(full_historical_daily_df.index)
            df_sorted = full_historical_daily_df.sort_index()
            prev_day_candidates = df_sorted[df_sorted.index.normalize() < current_processing_utc_dt.normalize()]

            if not prev_day_candidates.empty:
                last_day_data = prev_day_candidates.iloc[-1]
                self.prev_day_data = {
                    'open': last_day_data['open'], 'high': last_day_data['high'],
                    'low': last_day_data['low'], 'close': last_day_data['close'],
                    'volume': last_day_data['volume'], 'timestamp': last_day_data.name
                }
                if len(prev_day_candidates) >= 15:
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

            self.historical_1year_daily_data = prev_day_candidates.copy()
            last_close_for_sr = self.prev_day_data['close'] if self.prev_day_data else None
            self.long_term_daily_sr = self._detect_significant_sr_levels(
                self.historical_1year_daily_data, current_price=last_close_for_sr, num_levels=self.sr_num_levels
            )
            self.logger.info(f"Long-term daily S/R levels updated. "
                             f"Supports: {self.long_term_daily_sr['support']}, "
                             f"Resistances: {self.long_term_daily_sr['resistance']}")
        except Exception as e:
            self.logger.error(f"Error in update_daily_levels: {e}", exc_info=True)

    def extract_features(self, market_state: Dict[str, Any], portfolio_state:Dict[str,Any]) -> Optional[Dict[str, Any]]:
        try:
            # Extract timestamp from market_state - key name changed to 'timestamp_utc'
            current_ts: Optional[datetime] = market_state.get('timestamp_utc')
            if not current_ts:
                self.logger.warning("Timestamp (timestamp_utc) missing in market_state.")
                return None
            self.latest_timestamp = current_ts

            # Get the current 1s bar - key name changed from 'latest_1s_bar' to 'current_1s_bar'
            latest_1s_bar: Optional[Dict] = market_state.get('current_1s_bar')
            if not latest_1s_bar or 'close' not in latest_1s_bar:
                self.logger.debug(
                    f"current_1s_bar is missing or incomplete at {current_ts}. Skipping feature calculation.")
                return None
            current_price: float = latest_1s_bar['close']

            # Other data fields - ensure we use the correct keys
            rolling_1s_data_window: List[Dict] = market_state.get('rolling_1s_data_window', [])
            current_market_session: str = market_state.get('current_market_session', 'UNKNOWN')

            # VWAP session tracking
            if self._current_market_session_for_vwap != current_market_session:
                self.logger.info(
                    f"Market session changed from {self._current_market_session_for_vwap} to {current_market_session}. Resetting session VWAP.")
                self.session_vwap_sum_price_volume = 0.0
                self.session_vwap_sum_volume = 0.0
                self.current_session_vwap = None
                self._current_market_session_for_vwap = current_market_session

            # Update VWAP calculation
            if latest_1s_bar and 'volume' in latest_1s_bar and latest_1s_bar['volume'] > 0:
                typical_price = (latest_1s_bar.get('high', current_price) + latest_1s_bar.get('low',
                                                                                              current_price) + current_price) / 3
                self.session_vwap_sum_price_volume += typical_price * latest_1s_bar['volume']
                self.session_vwap_sum_volume += latest_1s_bar['volume']
                if self.session_vwap_sum_volume > 0:
                    self.current_session_vwap = self.session_vwap_sum_price_volume / self.session_vwap_sum_volume
                else:
                    self.current_session_vwap = current_price

            # Calculate feature vectors
            current_static_vector = self._calculate_static_features_vector(
                current_ts=current_ts, current_market_session=current_market_session
            )
            self.latest_static_features = current_static_vector

            current_hf_vector = self._calculate_hf_features_vector(
                latest_1s_bar=latest_1s_bar, rolling_1s_data_window=rolling_1s_data_window, current_price=current_price
            )

            # Note MarketSimulatorV2 uses 'current_1m_bar_forming' and 'current_5m_bar_forming'
            current_mf_vector = self._calculate_mf_features_vector(
                current_ts=current_ts, latest_1s_bar=latest_1s_bar,
                rolling_1s_data_window=rolling_1s_data_window,
                current_1m_bar=market_state.get('current_1m_bar_forming'),
                completed_1m_bars=market_state.get('completed_1m_bars_window', []),
                current_5m_bar=market_state.get('current_5m_bar_forming'),
                completed_5m_bars=market_state.get('completed_5m_bars_window', []),
                current_price=current_price
            )

            # Using the renamed fields for high/low values
            current_lf_vector = self._calculate_lf_features_vector(
                current_ts=current_ts, current_price=current_price,
                prev_day_data=self.prev_day_data,
                intraday_high=market_state.get('intraday_high'),
                intraday_low=market_state.get('intraday_low'),
                current_session_vwap=self.current_session_vwap,
                long_term_daily_sr=self.long_term_daily_sr
            )

            # Update feature history
            if current_hf_vector is not None: self.hf_history.append(current_hf_vector)
            if current_mf_vector is not None: self.mf_history.append(current_mf_vector)
            if current_lf_vector is not None: self.lf_history.append(current_lf_vector)

            # Check if we have enough data for model input
            if (len(self.hf_history) == self.hf_lookback_length and
                    len(self.mf_history) == self.mf_lookback_length and
                    len(self.lf_history) == self.lf_lookback_length and
                    self.latest_static_features is not None):
                model_input = {
                    'timestamp': current_ts,  # Keep 'timestamp' in output dict for model interface consistency
                    'static_features': self.latest_static_features,
                    'hf_features': np.array(list(self.hf_history), dtype=np.float32),
                    'mf_features': np.array(list(self.mf_history), dtype=np.float32),
                    'lf_features': np.array(list(self.lf_history), dtype=np.float32),
                }
                return model_input
            else:
                return None
        except Exception as e:
            self.logger.error(
                f"Error in calculate_features_and_get_model_input at {market_state.get('timestamp_utc')}: {e}",
                exc_info=True)
            return None

    def normalize_features(self, features):
        """
        Normalize features for the model using rolling window approach.

        Args:
            features: Dict containing feature arrays

        Returns:
            Dict containing normalized feature arrays
        """
        if not features:
            return {}

        normalized = {}

        # Pass through non-feature items
        if 'timestamp' in features:
            normalized['timestamp'] = features['timestamp']

        # Normalize static features - these might just be one value per sample
        if 'static_features' in features and features['static_features'] is not None:
            static_feat = features['static_features']
            # For one-hot encoded features, don't normalize
            is_binary = np.all(np.logical_or(np.isclose(static_feat, 0), np.isclose(static_feat, 1)))
            if is_binary:
                normalized['static_features'] = static_feat
            else:
                # For other features, standard normalization
                mean = np.nanmean(static_feat)
                std = np.nanstd(static_feat) + 1e-8
                normalized['static_features'] = (static_feat - mean) / std

        # Handle sequence features (hf, mf, lf)
        for key in ['hf_features', 'mf_features', 'lf_features']:
            if key in features and features[key] is not None:
                seq_feat = features[key]

                # Apply normalization that preserves temporal structure
                # For each feature dimension, compute stats across just the sequence
                feat_normalized = np.zeros_like(seq_feat)

                # Normalize each feature independently along the sequence
                # This respects the temporal structure
                for i in range(seq_feat.shape[1]):  # For each feature dimension
                    feature_slice = seq_feat[:, i]
                    # Skip one-hot or constant features
                    unique_vals = np.unique(feature_slice[~np.isnan(feature_slice)])
                    if len(unique_vals) <= 2:
                        feat_normalized[:, i] = feature_slice
                    else:
                        # For robust normalization, use percentiles rather than min/max
                        # And handle NaN values properly
                        feat_mean = np.nanmean(feature_slice)
                        feat_std = np.nanstd(feature_slice) + 1e-8
                        feat_normalized[:, i] = (feature_slice - feat_mean) / feat_std

                normalized[key] = feat_normalized

        return normalized

    def _calculate_static_features_vector(self, current_ts: datetime, current_market_session: str) -> np.array:
        feature_values_for_tick: Dict[str, Any] = {}
        # S_Day_Of_Week_Encoded
        day_of_week = current_ts.weekday()
        for i in range(5): feature_values_for_tick[f"S_Day_Of_Week_Encoded_{i}"] = 1 if day_of_week == i else 0
        # S_Time_Of_Day_Seconds_Encoded
        seconds_in_day = current_ts.hour * 3600 + current_ts.minute * 60 + current_ts.second
        feature_values_for_tick["S_Time_Of_Day_Seconds_Encoded_Sin"] = np.sin(2 * np.pi * seconds_in_day / (24 * 3600))
        feature_values_for_tick["S_Time_Of_Day_Seconds_Encoded_Cos"] = np.cos(2 * np.pi * seconds_in_day / (24 * 3600))
        # S_Market_Session_Encoded
        sessions = ["PREMARKET", "REGULAR", "POSTMARKET", "OTHER"]
        for s in sessions: feature_values_for_tick[
            f"S_Market_Session_Encoded_{s}"] = 1 if current_market_session == s else 0
        if current_market_session not in sessions: feature_values_for_tick["S_Market_Session_Encoded_OTHER"] = 1
        # Placeholder for other static features - these would typically be loaded or set elsewhere
        feature_values_for_tick["S_Stock_Float_Category"] = np.nan  # Example
        feature_values_for_tick["S_Has_Catalyst"] = 0  # Example
        feature_values_for_tick["S_Days_Since_Catalyst"] = -1  # Example
        feature_values_for_tick["S_Initial_Scan_RelVol"] = np.nan  # Example
        feature_values_for_tick["S_Initial_Scan_Price_ROC"] = np.nan  # Example
        raw = np.array([feature_values_for_tick.get(name, np.nan) for name in self.static_feature_names], dtype = np.float32)
        # replace NaN/Inf with zeros so the network sees valid inputs
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def _calculate_hf_features_vector(self, latest_1s_bar: Dict, rolling_1s_data_window: List[Dict],
                                      current_price: float) -> np.array:
        feature_values_for_tick: Dict[str, Any] = {}
        # STUB: Implement actual HF feature calculations based on Appendix/Design
        # This is a placeholder and needs to be filled with actual logic for each defined HF feature.
        # Example for one feature:
        feature_values_for_tick["HF_1s_ClosePrice"] = latest_1s_bar.get('close', np.nan)
        # ... many more HF features
        raw = np.array([feature_values_for_tick.get(name, np.nan) for name in self.static_feature_names],
                       dtype=np.float32)
        # replace NaN/Inf with zeros so the network sees valid inputs
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    def _calculate_mf_features_vector(self, current_ts: datetime, latest_1s_bar: Dict,
                                      rolling_1s_data_window: List[Dict],
                                      current_1m_bar: Optional[Dict], completed_1m_bars: List[Dict],
                                      current_5m_bar: Optional[Dict], completed_5m_bars: List[Dict],
                                      current_price: float) -> np.array:
        feature_values_for_tick: Dict[str, Any] = {}
        # STUB: Implement actual MF feature calculations
        # This is a placeholder.
        # Example:
        if current_1m_bar:
            feature_values_for_tick["MF_1m_Close_0"] = current_1m_bar.get('close', np.nan)
        # ... many more MF features
        raw = np.array([feature_values_for_tick.get(name, np.nan) for name in self.static_feature_names],
                       dtype=np.float32)
        # replace NaN/Inf with zeros so the network sees valid inputs
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    def _calculate_lf_features_vector(self, current_ts: datetime, current_price: float,
                                      prev_day_data: Optional[Dict],
                                      intraday_high: Optional[float],  # MODIFIED ARG
                                      intraday_low: Optional[float],  # MODIFIED ARG
                                      current_session_vwap: Optional[float],
                                      long_term_daily_sr: Optional[Dict]) -> np.array:
        feature_values_for_tick: Dict[str, Any] = {}

        if prev_day_data:
            if prev_day_data.get('high') is not None and prev_day_data['high'] > 0:
                feature_values_for_tick["LF_Dist_To_PrevDay_High_Pct"] = (current_price - prev_day_data['high']) / \
                                                                         prev_day_data['high'] * 100
            else:
                feature_values_for_tick["LF_Dist_To_PrevDay_High_Pct"] = np.nan
            if prev_day_data.get('low') is not None and prev_day_data['low'] > 0:
                feature_values_for_tick["LF_Dist_To_PrevDay_Low_Pct"] = (current_price - prev_day_data['low']) / \
                                                                        prev_day_data['low'] * 100
            else:
                feature_values_for_tick["LF_Dist_To_PrevDay_Low_Pct"] = np.nan
            if prev_day_data.get('close') is not None and prev_day_data['close'] > 0:
                feature_values_for_tick["LF_Dist_To_PrevDay_Close_Pct"] = (current_price - prev_day_data['close']) / \
                                                                          prev_day_data['close'] * 100
            else:
                feature_values_for_tick["LF_Dist_To_PrevDay_Close_Pct"] = np.nan
            if prev_day_data.get('high') is not None:
                feature_values_for_tick["LF_Is_Breaking_PrevDay_High"] = 1 if current_price > prev_day_data[
                    'high'] else 0
            else:
                feature_values_for_tick["LF_Is_Breaking_PrevDay_High"] = np.nan

        # Use intraday high/low
        if intraday_high is not None and intraday_high > 0:
            feature_values_for_tick["LF_Dist_To_Intraday_High_Pct"] = (
                                                                                  current_price - intraday_high) / intraday_high * 100
        else:
            feature_values_for_tick["LF_Dist_To_Intraday_High_Pct"] = np.nan

        if intraday_low is not None and intraday_low > 0:
            feature_values_for_tick["LF_Dist_To_Intraday_Low_Pct"] = (current_price - intraday_low) / intraday_low * 100
        else:
            feature_values_for_tick["LF_Dist_To_Intraday_Low_Pct"] = np.nan

        # Session Range Expansion Ratio using intraday high/low
        if intraday_high is not None and intraday_low is not None and \
                prev_day_data and prev_day_data.get('atr') and prev_day_data['atr'] > 0:
            current_intraday_range = intraday_high - intraday_low
            feature_values_for_tick["LF_Session_Range_Expansion_Ratio_vs_PrevDayATR"] = current_intraday_range / \
                                                                                        prev_day_data['atr']
        else:
            feature_values_for_tick["LF_Session_Range_Expansion_Ratio_vs_PrevDayATR"] = np.nan

        feature_values_for_tick["LF_Session_VWAP"] = current_session_vwap
        if current_session_vwap is not None and current_session_vwap > 0:
            feature_values_for_tick["LF_Dist_To_Session_VWAP_Pct"] = (
                                                                                 current_price - current_session_vwap) / current_session_vwap * 100
        else:
            feature_values_for_tick["LF_Dist_To_Session_VWAP_Pct"] = np.nan

        # Placeholder for LF_Cumulative_Session_Volume_Rel_To_Historical_Avg
        feature_values_for_tick["LF_Cumulative_Session_Volume_Rel_To_Historical_Avg"] = np.nan

        if long_term_daily_sr:
            for i, r_level in enumerate(long_term_daily_sr.get('resistance', [])):
                if i < self.sr_num_levels:
                    if r_level is not np.nan and r_level > 0:
                        feature_values_for_tick[f"LF_Dist_To_LT_Daily_Resistance_{i + 1}_Pct"] = (
                                                                                                             current_price - r_level) / r_level * 100
                    else:
                        feature_values_for_tick[f"LF_Dist_To_LT_Daily_Resistance_{i + 1}_Pct"] = np.nan
            for i, s_level in enumerate(long_term_daily_sr.get('support', [])):
                if i < self.sr_num_levels:
                    if s_level is not np.nan and s_level > 0:
                        feature_values_for_tick[f"LF_Dist_To_LT_Daily_Support_{i + 1}_Pct"] = (
                                                                                                          current_price - s_level) / s_level * 100
                    else:
                        feature_values_for_tick[f"LF_Dist_To_LT_Daily_Support_{i + 1}_Pct"] = np.nan

        raw = np.array([feature_values_for_tick.get(name, np.nan) for name in self.static_feature_names],
                       dtype=np.float32)
        # replace NaN/Inf with zeros so the network sees valid inputs
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    def get_feature_names_by_category(self) -> Dict[str, List[str]]:
        return {
            "static": self.static_feature_names,
            "hf": self.hf_feature_names,
            "mf": self.mf_feature_names,
            "lf": self.lf_feature_names,
        }

    def reset(self):
        """Reset the feature extractor state between episodes."""
        # Reset the history deques
        self.hf_history.clear()
        self.mf_history.clear()
        self.lf_history.clear()

        # Reset latest features
        self.latest_static_features = None
        self.latest_timestamp = None

        # Reset session VWAP accumulators if needed
        self.session_vwap_sum_price_volume = 0.0
        self.session_vwap_sum_volume = 0.0
        self.current_session_vwap = None
        self._current_market_session_for_vwap = None

        self.logger.info(f"FeatureExtractor for {self.symbol} reset")