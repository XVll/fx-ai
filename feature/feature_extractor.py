# feature_extractor.py (Updated with newest MF Volume interpretation)
import logging
from collections import deque
from datetime import datetime, timezone, timedelta, time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ta
from ta import trend, volatility

from config.config import FeatureConfig
from simulators.market_simulator import MarketSimulator

ET_TZ = timezone(timedelta(hours=-4))


class FeatureExtractor:
    EPSILON = 1e-9
    SESSION_START_HOUR_ET = 4
    SESSION_END_HOUR_ET = 20
    REGULAR_SESSION_START_HOUR_ET = 9.5
    REGULAR_SESSION_END_HOUR_ET = 16
    TOTAL_SECONDS_IN_FULL_TRADING_DAY = (SESSION_END_HOUR_ET - SESSION_START_HOUR_ET) * 3600
    TOTAL_MINUTES_IN_FULL_TRADING_DAY = TOTAL_SECONDS_IN_FULL_TRADING_DAY / 60

    HF_ROLLING_WINDOW_SECONDS = 60
    HF_PRICE_CHANGE_BASIS = "Close_1s_bar"
    HF_BAR_HLC_SOURCE = "Trades_and_Quotes"
    HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL = "Bar_MidPoint"
    HF_AGGRESSOR_ID_METHOD = "LeeReady"
    HF_NORMALIZED_IMBALANCE_DENOMINATOR_TYPE = "SumOfAggressorVolumes"
    HF_LARGE_TRADE_THRESHOLD_TYPE = "FactorOfRollingAvg"
    HF_LARGE_TRADE_FACTOR = 5.0
    HF_LARGE_TRADE_ROLLING_AVG_WINDOW_SECONDS = 60

    MF_PRICE_CHANGE_BASIS = "Close_Xm_bar_vs_PrevClose_Xm_bar"
    MF_CURRENT_PRICE_SOURCE_FOR_DISTS = "LiveTickPrice"
    MF_DIST_PCT_DENOMINATOR = "Level"
    MF_EMA_SHORT_PERIOD = 9
    MF_EMA_LONG_PERIOD = 20
    MF_ROLLING_HF_DATA_WINDOW_SECONDS = 60
    MF_MACD_FAST_PERIOD = 12
    MF_MACD_SLOW_PERIOD = 26
    MF_MACD_SIGNAL_PERIOD = 9
    MF_ATR_PERIOD = 14
    MF_VOLUME_DAILY_AVG_PROFILE_LOOKBACK_DAYS = 20  # Not used by the re-defined MF bar vol features, but kept for other potential uses or simulator
    MF_VOLUME_AVG_RECENT_BARS_WINDOW = 20
    MF_SWING_LOOKBACK_BARS = 10
    MF_SWING_STRENGTH_BARS = 2

    LF_AVG_DAILY_VOLUME_PERIOD_DAYS = 10  # Used by LF_RVol feature
    LF_PREV_DAY_HL_SOURCE = "RegularTradingHours"
    LF_PREV_DAY_CLOSE_SOURCE = "PostMarketClose_8PM"
    LF_VWAP_SESSION_SCOPE = "FullDay_4AM_Start_NoReset"
    LF_DAILY_EMA_PRICE_SOURCE = "PostMarketClose_8PM"
    LF_DAILY_EMA_SHORT_PERIOD = 9
    LF_DAILY_EMA_MEDIUM_PERIOD = 20
    LF_DAILY_EMA_LONG_PERIOD = 200
    LF_LT_SR_LOOKBACK_YEARS = 1

    def __init__(self, symbol: str, market_simulator: MarketSimulator, config: FeatureConfig, logger: Optional[logging.Logger] = None):
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        self.market_simulator = market_simulator

        try:
            symbol_info = self.market_simulator.get_symbol_info()
            self.total_shares_outstanding: Optional[float] = symbol_info['total_shares_outstanding']
        except Exception as e:
            self.logger.error(f"Could not get total_shares_outstanding: {e}. Market cap feature will be NaN.")
            self.total_shares_outstanding = None

        self.static_feature_names: List[str] = [
            "S_Time_Of_Day_Seconds_Encoded_Sin",
            "S_Time_Of_Day_Seconds_Encoded_Cos",
            "S_Market_Cap_Million",
        ]
        # "S_Initial_PreMarket_Gap_Pct",  # Todo: Removed for now, requires to fetch previous day's data.
        # "S_Regular_Open_Gap_Pct",
        self.hf_feature_names: List[str] = [
            "HF_1s_PriceChange_Pct", "HF_1s_Volume_Ratio_To_Own_Avg", "HF_1s_Volume_Delta_Pct", "HF_1s_HighLow_Spread_Rel",
            "HF_Tape_1s_Trades_Count_Ratio_To_Own_Avg", "HF_Tape_1s_Trades_Count_Delta_Pct", "HF_Tape_1s_Normalized_Volume_Imbalance",
            "HF_Tape_1s_Normalized_Volume_Imbalance_Delta", "HF_Tape_1s_Avg_Trade_Size_Ratio_To_Own_Avg", "HF_Tape_1s_Avg_Trade_Size_Delta_Pct",
            "HF_Tape_1s_Large_Trade_Count", "HF_Tape_1s_Large_Trade_Net_Volume_Ratio_To_Total_Vol", "HF_Tape_1s_Trades_VWAP",
            "HF_Quote_1s_Spread_Rel", "HF_Quote_1s_Spread_Rel_Delta", "HF_Quote_1s_Quote_Imbalance_Value_Ratio",
            "HF_Quote_1s_Quote_Imbalance_Value_Ratio_Delta", "HF_Quote_1s_Bid_Value_USD_Ratio_To_Own_Avg", "HF_Quote_1s_Ask_Value_USD_Ratio_To_Own_Avg",
        ]
        self.mf_feature_names: List[str] = [
            "MF_1m_PriceChange_Pct", "MF_5m_PriceChange_Pct", "MF_1m_PriceChange_Pct_Delta", "MF_5m_PriceChange_Pct_Delta",
            "MF_1m_Position_In_CurrentCandle_Range", "MF_5m_Position_In_CurrentCandle_Range", "MF_1m_Position_In_PreviousCandle_Range",
            "MF_5m_Position_In_PreviousCandle_Range",
            "MF_1m_Dist_To_EMA9_Pct", "MF_1m_Dist_To_EMA20_Pct", "MF_5m_Dist_To_EMA9_Pct", "MF_5m_Dist_To_EMA20_Pct",
            "MF_Dist_To_Rolling_HF_High_Pct", "MF_Dist_To_Rolling_HF_Low_Pct",
            "MF_1m_MACD_Line", "MF_1m_MACD_Signal", "MF_1m_MACD_Hist", "MF_1m_ATR_Pct", "MF_5m_ATR_Pct",
            "MF_1m_BodySize_Rel", "MF_1m_UpperWick_Rel", "MF_1m_LowerWick_Rel", "MF_5m_BodySize_Rel", "MF_5m_UpperWick_Rel", "MF_5m_LowerWick_Rel",
            "MF_1m_BarVol_Ratio_To_TodaySoFarVol",  # RENAMED and logic changed
            "MF_5m_BarVol_Ratio_To_TodaySoFarVol",  # RENAMED and logic changed
            "MF_1m_Volume_Rel_To_Avg_Recent_Bars", "MF_5m_Volume_Rel_To_Avg_Recent_Bars",
            "MF_1m_Dist_To_Recent_SwingHigh_Pct", "MF_1m_Dist_To_Recent_SwingLow_Pct", "MF_5m_Dist_To_Recent_SwingHigh_Pct",
            "MF_5m_Dist_To_Recent_SwingLow_Pct",
        ]
        self.lf_feature_names: List[str] = [
            "LF_Position_In_Daily_Range", "LF_Position_In_PrevDay_Range", "LF_Pct_Change_From_Prev_Close",
            "LF_RVol_Pct_From_Avg_10d_Timed", "LF_Dist_To_Session_VWAP_Pct",
            "LF_Daily_Dist_To_EMA9_Pct", "LF_Daily_Dist_To_EMA20_Pct", "LF_Daily_Dist_To_EMA200_Pct",
            "LF_Dist_To_Closest_LT_Support_Pct", "LF_Dist_To_Closest_LT_Resistance_Pct",
        ]

        self.config.static_feat_dim = len(self.static_feature_names)
        self.config.hf_feat_dim = len(self.hf_feature_names)
        self.config.mf_feat_dim = len(self.mf_feature_names)
        self.config.lf_feat_dim = len(self.lf_feature_names)

        self.hf_seq_len = self.config.hf_seq_len
        self.mf_seq_len = self.config.mf_seq_len
        self.lf_seq_len = self.config.lf_seq_len

        self.market_data: Dict[str, Any] = {}

        self.completed_1m_bars_df = pd.DataFrame()
        self.completed_5m_bars_df = pd.DataFrame()
        self.historical_1d_bars_df = pd.DataFrame()

        self.hf_features_history = deque(maxlen=self.hf_seq_len)
        self.mf_features_history = deque(maxlen=self.mf_seq_len)
        self.lf_features_history = deque(maxlen=self.lf_seq_len)

        self.max_hf_lookback_seconds = max(
            self.HF_ROLLING_WINDOW_SECONDS,
            self.HF_LARGE_TRADE_ROLLING_AVG_WINDOW_SECONDS, 1
        )
        self.recent_1s_volumes = deque(maxlen=self.max_hf_lookback_seconds)
        self.recent_1s_trade_counts = deque(maxlen=self.max_hf_lookback_seconds)
        self.recent_1s_total_trade_shares = deque(maxlen=self.max_hf_lookback_seconds)
        self.recent_1s_bid_values_usd = deque(maxlen=self.HF_ROLLING_WINDOW_SECONDS)
        self.recent_1s_ask_values_usd = deque(maxlen=self.HF_ROLLING_WINDOW_SECONDS)

        self.prev_tick_values: Dict[str, Any] = {
            "HF_1s_PriceChange_basis_value": np.nan, "HF_1s_Volume": np.nan,
            "HF_Tape_1s_Trades_Count": np.nan, "HF_Tape_1s_Normalized_Volume_Imbalance": np.nan,
            "HF_Tape_1s_Avg_Trade_Size": np.nan, "HF_Quote_1s_Spread_Rel": np.nan,
            "HF_Quote_1s_Quote_Imbalance_Value_Ratio": np.nan,
            "MF_1m_PriceChange_Pct": np.nan, "MF_5m_PriceChange_Pct": np.nan,
        }

        self.s_initial_premarket_gap_pct: Optional[float] = None
        self.s_regular_open_gap_pct: Optional[float] = None
        self.prev_day_close_price_for_gaps: Optional[float] = None
        self.first_tick_price_today: Optional[float] = None
        self.rth_open_price_today: Optional[float] = None

        self.daily_cumulative_volume = 0.0  # Ensure float
        self.daily_cumulative_price_volume_product = 0.0  # Ensure float
        self.last_processed_day_for_daily_reset: Optional[datetime.date] = None

        self._initialize_buffers_with_nan()

        # Initial fetch of prev_day_close for gap calculations at startup
        # This ensures that if extract_features is called before a daily reset occurs on the first day,
        # prev_day_close_price_for_gaps is already populated.
        # _update_market_data will also call this if a new day is detected.
        # We need to ensure historical_1d_bars_df is populated before calling _update_prev_day_close_for_gaps.
        initial_market_state_for_setup = self.market_simulator.get_current_market_state()  # Use a fresh call
        hist_1d_init = initial_market_state_for_setup.get('historical_1d_bars')
        if isinstance(hist_1d_init, pd.DataFrame) and not hist_1d_init.empty:
            self.historical_1d_bars_df = hist_1d_init.copy()
            if 'timestamp' in self.historical_1d_bars_df.columns and self.historical_1d_bars_df.index.name != 'timestamp':
                self.historical_1d_bars_df['timestamp'] = pd.to_datetime(self.historical_1d_bars_df['timestamp'])
                self.historical_1d_bars_df.set_index('timestamp', inplace=True)
            elif self.historical_1d_bars_df.index.name == 'timestamp' or isinstance(self.historical_1d_bars_df.index, pd.DatetimeIndex):
                self.historical_1d_bars_df.index = pd.to_datetime(self.historical_1d_bars_df.index)
            self._update_prev_day_close_for_gaps()
        else:
            self.logger.warning("Initial historical_1d_bars not available at __init__. Previous day close might be initially NaN.")
            self.historical_1d_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm'])

    def _initialize_buffers_with_nan(self):
        self.config.mf_feat_dim = len(self.mf_feature_names)  # Ensure this is set before use

        nan_hf = np.full(self.config.hf_feat_dim, np.nan)
        for _ in range(self.hf_seq_len): self.hf_features_history.append(nan_hf.copy())
        nan_mf = np.full(self.config.mf_feat_dim, np.nan)
        for _ in range(self.mf_seq_len): self.mf_features_history.append(nan_mf.copy())
        nan_lf = np.full(self.config.lf_feat_dim, np.nan)
        for _ in range(self.lf_seq_len): self.lf_features_history.append(nan_lf.copy())

    def _safe_get_from_bar(self, bar_data: Optional[Dict], key: str, default: Any = np.nan) -> Any:
        if bar_data is None: return default
        return bar_data.get(key, default)

    def _update_market_data(self):
        self.market_data = self.market_simulator.get_current_market_state()
        current_ts_utc = self.market_data.get('timestamp_utc')

        if current_ts_utc:
            current_date = current_ts_utc.date()
            if self.last_processed_day_for_daily_reset is None or self.last_processed_day_for_daily_reset != current_date:
                self.logger.info(f"New trading day {current_date}. Resetting daily state.")
                self.daily_cumulative_volume = 0.0
                self.daily_cumulative_price_volume_product = 0.0
                self.s_initial_premarket_gap_pct = None  # Re-evaluate on new day
                self.s_regular_open_gap_pct = None  # Re-evaluate on new day
                self.first_tick_price_today = None  # Re-evaluate
                self.rth_open_price_today = None  # Re-evaluate
                self.last_processed_day_for_daily_reset = current_date
                # Update prev_day_close_price_for_gaps for the new day
                # Ensure historical_1d_bars_df is updated first if it changes daily
                hist_1d_update = self.market_data.get('historical_1d_bars')
                if isinstance(hist_1d_update, pd.DataFrame) and not hist_1d_update.empty:
                    self.historical_1d_bars_df = hist_1d_update.copy()
                    if 'timestamp' in self.historical_1d_bars_df.columns and self.historical_1d_bars_df.index.name != 'timestamp':
                        self.historical_1d_bars_df['timestamp'] = pd.to_datetime(self.historical_1d_bars_df['timestamp'])
                        self.historical_1d_bars_df.set_index('timestamp', inplace=True)
                    elif self.historical_1d_bars_df.index.name == 'timestamp' or isinstance(self.historical_1d_bars_df.index, pd.DatetimeIndex):
                        self.historical_1d_bars_df.index = pd.to_datetime(self.historical_1d_bars_df.index)
                self._update_prev_day_close_for_gaps()

        def _df_from_window(window_data, cols):
            if window_data:
                df = pd.DataFrame(window_data)
                if 'timestamp_start' in df.columns:
                    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'])
                    df.set_index('timestamp_start', inplace=True, drop=False)  # Keep timestamp_start as column too for ta lib if needed
                return df
            return pd.DataFrame(columns=cols + ['timestamp_start']).set_index('timestamp_start', drop=False)

        self.completed_1m_bars_df = _df_from_window(
            self.market_data.get('completed_1m_bars_window', []),
            ['open', 'high', 'low', 'close', 'volume']
        )
        self.completed_5m_bars_df = _df_from_window(
            self.market_data.get('completed_5m_bars_window', []),
            ['open', 'high', 'low', 'close', 'volume']
        )

        # historical_1d_bars_df is already updated at init and potential daily reset
        # but ensure it's the latest if simulator provides updates frequently
        hist_1d = self.market_data.get('historical_1d_bars')
        if isinstance(hist_1d, pd.DataFrame) and not hist_1d.empty:
            # Only copy if it's different from what we have, to avoid overhead
            # This check might be too simplistic for real scenarios.
            if not self.historical_1d_bars_df.equals(hist_1d):
                self.historical_1d_bars_df = hist_1d.copy()
                if 'timestamp' in self.historical_1d_bars_df.columns and self.historical_1d_bars_df.index.name != 'timestamp':
                    self.historical_1d_bars_df['timestamp'] = pd.to_datetime(self.historical_1d_bars_df['timestamp'])
                    self.historical_1d_bars_df.set_index('timestamp', inplace=True)
                elif self.historical_1d_bars_df.index.name == 'timestamp' or isinstance(self.historical_1d_bars_df.index, pd.DatetimeIndex):
                    self.historical_1d_bars_df.index = pd.to_datetime(self.historical_1d_bars_df.index)
        elif self.historical_1d_bars_df.empty:  # If it was empty and now we got data
            self.historical_1d_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm'])

        current_1s = self.market_data.get('current_1s_bar')
        rolling_1s_window = self.market_data.get('rolling_1s_data_window', [])

        default_val_deque = 0.0
        if current_1s:
            self.recent_1s_volumes.append(float(current_1s.get('volume', default_val_deque)))
            last_1s_event_data = rolling_1s_window[-1] if rolling_1s_window else None
            if last_1s_event_data:
                trades_this_sec = last_1s_event_data.get('trades', [])
                self.recent_1s_trade_counts.append(float(len(trades_this_sec)))
                self.recent_1s_total_trade_shares.append(float(sum(t.get('size', 0) for t in trades_this_sec)))
            else:
                self.recent_1s_trade_counts.append(default_val_deque)
                self.recent_1s_total_trade_shares.append(default_val_deque)

            bid_val = float(self.market_data.get('best_bid_price', 0) * self.market_data.get('best_bid_size', 0))
            ask_val = float(self.market_data.get('best_ask_price', 0) * self.market_data.get('best_ask_size', 0))
            self.recent_1s_bid_values_usd.append(bid_val)
            self.recent_1s_ask_values_usd.append(ask_val)
        else:  # If no current_1s_bar, append defaults to deques to maintain size
            self.recent_1s_volumes.append(default_val_deque)
            self.recent_1s_trade_counts.append(default_val_deque)
            self.recent_1s_total_trade_shares.append(default_val_deque)
            self.recent_1s_bid_values_usd.append(default_val_deque)
            self.recent_1s_ask_values_usd.append(default_val_deque)

        if self.LF_VWAP_SESSION_SCOPE == "FullDay_4AM_Start_NoReset" and current_1s:
            vol_1s = float(current_1s.get('volume', 0))
            vwap_1s = current_1s.get('vwap')
            if vol_1s > self.EPSILON:  # only add if there's volume
                price_vol_prod = 0.0
                if vwap_1s is not None:
                    price_vol_prod = float(vwap_1s) * vol_1s
                else:
                    price_1s = (float(current_1s.get('high', 0)) + float(current_1s.get('low', 0))) / 2.0
                    if not np.isnan(price_1s):
                        price_vol_prod = price_1s * vol_1s

                self.daily_cumulative_price_volume_product += price_vol_prod
                self.daily_cumulative_volume += vol_1s

    def _update_prev_day_close_for_gaps(self):
        if self.historical_1d_bars_df.empty:
            self.prev_day_close_price_for_gaps = np.nan
            return
        try:
            # Ensure DataFrame index is DatetimeIndex and sorted if relying on iloc[-1] for "previous day"
            if not isinstance(self.historical_1d_bars_df.index, pd.DatetimeIndex):
                self.logger.warning("historical_1d_bars_df index is not DatetimeIndex. Previous day data might be incorrect.")
                self.prev_day_close_price_for_gaps = np.nan
                return

            # Assuming the last row is indeed the previous trading day relative to current processing time
            prev_day_data = self.historical_1d_bars_df.iloc[-1]
            col_map = {"PostMarketClose_8PM": "close_8pm", "RegularSessionClose_4PM": "close"}
            close_col = col_map.get(self.LF_PREV_DAY_CLOSE_SOURCE, 'close')
            self.prev_day_close_price_for_gaps = float(prev_day_data.get(close_col, prev_day_data.get('close', np.nan)))
        except IndexError:
            self.prev_day_close_price_for_gaps = np.nan
            self.logger.warning("IndexError fetching previous day close for gaps.")
        except Exception as e:
            self.prev_day_close_price_for_gaps = np.nan
            self.logger.error(f"Exception fetching previous day close: {e}")

    def _get_price_for_hf_change_basis(self, bar_data: Optional[Dict]) -> float:
        if bar_data is None: return np.nan
        key = 'close' if self.HF_PRICE_CHANGE_BASIS == "Close_1s_bar" else \
            'vwap' if self.HF_PRICE_CHANGE_BASIS == "VWAP_1s_bar" else None
        return float(bar_data.get(key, np.nan)) if key else np.nan

    def _get_current_price_for_mf_dists(self) -> float:
        if self.MF_CURRENT_PRICE_SOURCE_FOR_DISTS == "LiveTickPrice":
            return float(self.market_data.get('current_price', np.nan))
        elif self.MF_CURRENT_PRICE_SOURCE_FOR_DISTS == "BarClosePrice":
            return self._safe_get_from_bar(self.market_data.get('current_1s_bar'), 'close')
        return np.nan

    def _calculate_pct_dist(self, price: float, level: float) -> float:
        if np.isnan(price) or np.isnan(level): return np.nan
        denom = level if self.MF_DIST_PCT_DENOMINATOR == "Level" else \
            price if self.MF_DIST_PCT_DENOMINATOR == "CurrentPrice" else np.nan
        if np.isnan(denom) or abs(denom) < self.EPSILON: return np.nan
        return ((price - level) / denom) * 100.0

    def _safe_divide(self, num, den, default_val=np.nan):
        if den is None or num is None or np.isnan(den) or np.isnan(num) or abs(float(den)) < self.EPSILON:
            return default_val
        return float(num) / float(den)

    def _pct_change(self, cur, prev, default_val=np.nan):
        if prev is None or cur is None or np.isnan(prev) or np.isnan(cur) or abs(float(prev)) < self.EPSILON:
            return default_val
        return ((float(cur) - float(prev)) / float(prev)) * 100.0

    def _rolling_mean_deque(self, dq: deque, window: Optional[int] = None) -> float:
        if not dq: return np.nan
        elements = list(dq)
        if window is not None:
            actual_elements = elements[-window:]
        else:
            actual_elements = elements
        return np.mean([float(x) for x in actual_elements if not np.isnan(x)]) if actual_elements else np.nan

    def _get_daily_bar_value(self, date_offset: int, column_name: str, source_pref: Optional[str] = None) -> float:
        if self.historical_1d_bars_df.empty: return np.nan
        try:
            idx = date_offset
            if not (0 <= abs(idx) < len(self.historical_1d_bars_df)) and idx < 0:  # Ensure index is valid for negative offset
                # Check if requested index is within bounds
                if len(self.historical_1d_bars_df) < abs(idx):
                    self.logger.debug(f"Not enough historical daily bars for offset {idx}. Have {len(self.historical_1d_bars_df)}")
                    return np.nan
            elif idx >= 0 and idx >= len(self.historical_1d_bars_df):  # For positive offset (not typical for prev day)
                return np.nan

            day_data = self.historical_1d_bars_df.iloc[idx]

            col_to_get = column_name
            if column_name == 'high':
                if source_pref == "FullTradingDay_4AM_8PM" and 'full_day_high' in day_data: col_to_get = 'full_day_high'
            elif column_name == 'low':
                if source_pref == "FullTradingDay_4AM_8PM" and 'full_day_low' in day_data: col_to_get = 'full_day_low'
            elif column_name == 'close':
                col_map = {"PostMarketClose_8PM": "close_8pm", "RegularSessionClose_4PM": "close"}
                col_to_get = col_map.get(source_pref, 'close')

            val = day_data.get(col_to_get, day_data.get(column_name, np.nan))
            return float(val) if not pd.isna(val) else np.nan
        except (IndexError, KeyError) as e:
            self.logger.warning(f"Error in _get_daily_bar_value for offset {date_offset}, col {column_name}: {e}")
            return np.nan

    def _find_swing_points(self, prices: pd.Series, strength: int) -> (List[float], List[float]):
        if prices.empty or len(prices) < (2 * strength + 1): return [], []
        highs, lows = [], []
        # Ensure series is float
        prices = prices.astype(float)
        for i in range(strength, len(prices) - strength):
            window = prices.iloc[i - strength: i + strength + 1]
            current_price_val = prices.iloc[i]
            if not np.isnan(current_price_val):
                if current_price_val == window.max(): highs.append(current_price_val)
                if current_price_val == window.min(): lows.append(current_price_val)
        return highs, lows

    def extract_features(self) -> Dict[str, np.ndarray]:
        self._update_market_data()

        static_vec = self._calculate_static_features()
        hf_vec = self._calculate_hf_features()
        mf_vec = self._calculate_mf_features()
        lf_vec = self._calculate_lf_features()

        self.hf_features_history.append(hf_vec)
        self.mf_features_history.append(mf_vec)
        self.lf_features_history.append(lf_vec)

        for key, vec in {'hf': hf_vec, 'mf': mf_vec, 'lf': lf_vec, 'static': static_vec}.items():
            if np.isnan(vec).any():
                nan_count = np.isnan(vec).sum()
                self.logger.warning(f"NaN values detected in {key} feature vector: {nan_count}/{len(vec)} features")

        return {
            'hf': np.array(self.hf_features_history, dtype=np.float32),
            'mf': np.array(self.mf_features_history, dtype=np.float32),
            'lf': np.array(self.lf_features_history, dtype=np.float32),
            'static': static_vec.reshape(1, -1).astype(np.float32),
        }

    def _calculate_static_features(self) -> np.ndarray:
        features = np.full(self.config.static_feat_dim, np.nan)
        current_ts_utc = self.market_data.get('timestamp_utc')

        # Time of Day Encoding
        if current_ts_utc:
            try:
                current_time_et = current_ts_utc.astimezone(ET_TZ)
                secs_from_midnight_et = current_time_et.hour * 3600 + current_time_et.minute * 60 + current_time_et.second
                session_start_secs_et = self.SESSION_START_HOUR_ET * 3600
                secs_from_session_start = secs_from_midnight_et - session_start_secs_et

                angle = 2 * np.pi * secs_from_session_start / self.TOTAL_SECONDS_IN_FULL_TRADING_DAY
                features[0] = np.sin(angle)
                features[1] = np.cos(angle)
            except Exception as e:
                self.logger.warning(f"Time of day calc error: {e}")

        # Market Cap
        if self.total_shares_outstanding and self.prev_day_close_price_for_gaps and not np.isnan(self.prev_day_close_price_for_gaps):
            features[2] = (self.total_shares_outstanding * self.prev_day_close_price_for_gaps) / 1_000_000.0
        else:
            if self.total_shares_outstanding is None: self.logger.debug("Market Cap: TotalSharesOutstanding missing from simulator info.")
            if self.prev_day_close_price_for_gaps is None or np.isnan(self.prev_day_close_price_for_gaps): self.logger.debug(
                "Market Cap: Prev day close for gaps unavailable.")

        # Gap Calculations using get_state_at_time
        # if current_ts_utc:
        # today_et_date = current_ts_utc.astimezone(ET_TZ).date()

        # Initial PreMarket Gap
        # if self.s_initial_premarket_gap_pct is None:  # Calculate once per day
        #     if self.first_tick_price_today is None:
        #         try:
        #             # Construct datetime for pre-market open in ET, then convert to UTC
        #             pm_open_time_et = datetime.combine(today_et_date, time(self.SESSION_START_HOUR_ET, 0), tzinfo=ET_TZ)
        #             pm_open_time_utc = pm_open_time_et.astimezone(timezone.utc)
        #
        #             state_at_pm_open = self.market_simulator.get_state_at_time(pm_open_time_utc)
        #             # Assuming get_state_at_time returns a dict with 'current_1s_bar' or 'current_price'
        #             bar_at_pm_open = state_at_pm_open.get('current_1s_bar')
        #             if bar_at_pm_open and 'open' in bar_at_pm_open:
        #                 self.first_tick_price_today = float(bar_at_pm_open['open'])
        #             elif 'current_price' in state_at_pm_open:  # Fallback
        #                 self.first_tick_price_today = float(state_at_pm_open['current_price'])
        #             else:
        #                 self.logger.warning("Could not determine pre-market open price from get_state_at_time.")
        #         except Exception as e:
        #             self.logger.error(f"Error getting pre-market open state: {e}")
        #
        #     if self.first_tick_price_today is not None and self.prev_day_close_price_for_gaps is not None and not np.isnan(
        #             self.prev_day_close_price_for_gaps):
        #         self.s_initial_premarket_gap_pct = self._pct_change(self.first_tick_price_today, self.prev_day_close_price_for_gaps)
        # features[3] = self.s_initial_premarket_gap_pct

        # # Regular Open Gap
        # if self.s_regular_open_gap_pct is None:  # Calculate once per day
        #     if self.rth_open_price_today is None:
        #         try:
        #             rth_hour = int(self.REGULAR_SESSION_START_HOUR_ET)
        #             rth_minute = int((self.REGULAR_SESSION_START_HOUR_ET % 1) * 60)
        #             rth_open_time_et = datetime.combine(today_et_date, time(rth_hour, rth_minute), tzinfo=ET_TZ)
        #             rth_open_time_utc = rth_open_time_et.astimezone(timezone.utc)
        #
        #             state_at_rth_open = self.market_simulator.get_state_at_time(rth_open_time_utc)
        #             bar_at_rth_open = state_at_rth_open.get('current_1s_bar')
        #             if bar_at_rth_open and 'open' in bar_at_rth_open:
        #                 self.rth_open_price_today = float(bar_at_rth_open['open'])
        #             elif 'current_price' in state_at_rth_open:  # Fallback
        #                 self.rth_open_price_today = float(state_at_rth_open['current_price'])
        #             else:
        #                 self.logger.warning("Could not determine RTH open price from get_state_at_time.")
        #         except Exception as e:
        #             self.logger.error(f"Error getting RTH open state: {e}")
        #
        #     if self.rth_open_price_today is not None and self.prev_day_close_price_for_gaps is not None and not np.isnan(
        #             self.prev_day_close_price_for_gaps):
        #         self.s_regular_open_gap_pct = self._pct_change(self.rth_open_price_today, self.prev_day_close_price_for_gaps)
        # features[4] = self.s_regular_open_gap_pct

            # Debug NaN values
            nan_indices = np.where(np.isnan(features))[0]
            if len(nan_indices) > 0:
                nan_features = [self.static_feature_names[i] for i in nan_indices]
                self.logger.warning(f"NaN values detected in HF features: {nan_features}")
                # Print the actual calculations that led to these NaNs
                for i in nan_indices:
                    feature_name = self.static_feature_names[i]
                    self.logger.warning(f"NaN in {feature_name} calculation")

        return features.astype(np.float32)

    def _calculate_hf_features(self) -> np.ndarray:
        features = np.full(self.config.hf_feat_dim, np.nan)
        current_1s = self.market_data.get('current_1s_bar')
        rolling_1s = self.market_data.get('rolling_1s_data_window', [])
        if not current_1s:
            self.logger.warning("HF Features: current_1s_bar missing.")
            return features.astype(np.float32)

        cur_px_basis = self._get_price_for_hf_change_basis(current_1s)
        prev_px_basis = self.prev_tick_values["HF_1s_PriceChange_basis_value"]
        features[0] = self._pct_change(cur_px_basis, prev_px_basis)
        self.prev_tick_values["HF_1s_PriceChange_basis_value"] = cur_px_basis

        cur_vol = self._safe_get_from_bar(current_1s, 'volume', 0.0)
        avg_vol_roll = self._rolling_mean_deque(self.recent_1s_volumes, self.HF_ROLLING_WINDOW_SECONDS)
        features[1] = self._safe_divide(cur_vol, avg_vol_roll)

        prev_vol = self.prev_tick_values["HF_1s_Volume"]
        features[2] = self._pct_change(cur_vol, prev_vol)
        self.prev_tick_values["HF_1s_Volume"] = cur_vol

        h, l = self._safe_get_from_bar(current_1s, 'high'), self._safe_get_from_bar(current_1s, 'low')
        spread_abs = float(h) - float(l) if not (np.isnan(h) or np.isnan(l)) else np.nan
        den_spread_rel = np.nan
        if self.HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL == "Bar_MidPoint":
            if not (np.isnan(h) or np.isnan(l)): den_spread_rel = (float(h) + float(l)) / 2.0
        elif self.HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL == "Bar_Open":
            den_spread_rel = self._safe_get_from_bar(current_1s, 'open')
        elif self.HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL == "Bar_Close":
            den_spread_rel = self._safe_get_from_bar(current_1s, 'close')
        features[3] = self._safe_divide(spread_abs, den_spread_rel)

        last_1s_event = rolling_1s[-1] if rolling_1s else {}
        trades = last_1s_event.get('trades', [])
        cur_trades_cnt = float(len(trades))

        avg_trades_cnt_roll = self._rolling_mean_deque(self.recent_1s_trade_counts, self.HF_ROLLING_WINDOW_SECONDS)
        features[4] = self._safe_divide(cur_trades_cnt, avg_trades_cnt_roll)

        prev_trades_cnt = self.prev_tick_values["HF_Tape_1s_Trades_Count"]
        features[5] = self._pct_change(cur_trades_cnt, prev_trades_cnt)
        self.prev_tick_values["HF_Tape_1s_Trades_Count"] = cur_trades_cnt

        buy_vol = sum(float(t.get('size', 0)) for t in trades if t.get('side') == 'B')
        sell_vol = sum(float(t.get('size', 0)) for t in trades if t.get('side') == 'A')

        imbal_num = buy_vol - sell_vol
        imbal_den = np.nan
        if self.HF_NORMALIZED_IMBALANCE_DENOMINATOR_TYPE == "SumOfAggressorVolumes":
            imbal_den = buy_vol + sell_vol
        elif self.HF_NORMALIZED_IMBALANCE_DENOMINATOR_TYPE == "TotalBarVolume":
            imbal_den = cur_vol
        cur_norm_vol_imbal = self._safe_divide(imbal_num, imbal_den)
        features[6] = cur_norm_vol_imbal

        prev_norm_vol_imbal = self.prev_tick_values["HF_Tape_1s_Normalized_Volume_Imbalance"]
        features[7] = cur_norm_vol_imbal - prev_norm_vol_imbal if not (np.isnan(cur_norm_vol_imbal) or np.isnan(prev_norm_vol_imbal)) else np.nan
        self.prev_tick_values["HF_Tape_1s_Normalized_Volume_Imbalance"] = cur_norm_vol_imbal

        total_shares_1s = sum(float(t.get('size', 0)) for t in trades)
        cur_avg_trade_size = self._safe_divide(total_shares_1s, cur_trades_cnt, 0.0)

        roll_total_shares_list = [float(x) for x in list(self.recent_1s_total_trade_shares)[-self.HF_ROLLING_WINDOW_SECONDS:] if not np.isnan(x)]
        roll_total_counts_list = [float(x) for x in list(self.recent_1s_trade_counts)[-self.HF_ROLLING_WINDOW_SECONDS:] if not np.isnan(x)]
        avg_trade_size_roll = self._safe_divide(np.sum(roll_total_shares_list), np.sum(roll_total_counts_list))
        features[8] = self._safe_divide(cur_avg_trade_size, avg_trade_size_roll)

        prev_avg_trade_size = self.prev_tick_values["HF_Tape_1s_Avg_Trade_Size"]
        features[9] = self._pct_change(cur_avg_trade_size, prev_avg_trade_size)
        self.prev_tick_values["HF_Tape_1s_Avg_Trade_Size"] = cur_avg_trade_size

        large_trade_thresh = np.nan
        if self.HF_LARGE_TRADE_THRESHOLD_TYPE == "FactorOfRollingAvg":
            roll_total_shares_large_list = [float(x) for x in list(self.recent_1s_total_trade_shares)[-self.HF_LARGE_TRADE_ROLLING_AVG_WINDOW_SECONDS:] if
                                            not np.isnan(x)]
            roll_total_counts_large_list = [float(x) for x in list(self.recent_1s_trade_counts)[-self.HF_LARGE_TRADE_ROLLING_AVG_WINDOW_SECONDS:] if
                                            not np.isnan(x)]
            avg_trade_size_large_roll = self._safe_divide(np.sum(roll_total_shares_large_list), np.sum(roll_total_counts_large_list))

            if not np.isnan(avg_trade_size_large_roll):
                large_trade_thresh = self.HF_LARGE_TRADE_FACTOR * avg_trade_size_large_roll

        large_trades_cnt = 0.0
        large_buy_vol = 0.0
        large_sell_vol = 0.0
        if not np.isnan(large_trade_thresh):
            for t in trades:
                trade_size = float(t.get('size', 0))
                if trade_size > large_trade_thresh:
                    large_trades_cnt += 1
                    if t.get('side') == 'B':
                        large_buy_vol += trade_size
                    elif t.get('side') == 'A':
                        large_sell_vol += trade_size
        features[10] = large_trades_cnt
        features[11] = self._safe_divide(large_buy_vol - large_sell_vol, cur_vol)

        features[12] = self._safe_get_from_bar(current_1s, 'vwap')

        bid_px, ask_px = float(self.market_data.get('best_bid_price', np.nan)), float(self.market_data.get('best_ask_price', np.nan))
        bid_sz, ask_sz = float(self.market_data.get('best_bid_size', np.nan)), float(self.market_data.get('best_ask_size', np.nan))

        spread_abs_q = ask_px - bid_px if not (np.isnan(ask_px) or np.isnan(bid_px)) else np.nan
        mid_px_q = (ask_px + bid_px) / 2.0 if not (np.isnan(ask_px) or np.isnan(bid_px)) else np.nan
        cur_spread_rel_q = self._safe_divide(spread_abs_q, mid_px_q)
        features[13] = cur_spread_rel_q

        prev_spread_rel_q = self.prev_tick_values["HF_Quote_1s_Spread_Rel"]
        features[14] = cur_spread_rel_q - prev_spread_rel_q if not (np.isnan(cur_spread_rel_q) or np.isnan(prev_spread_rel_q)) else np.nan
        self.prev_tick_values["HF_Quote_1s_Spread_Rel"] = cur_spread_rel_q

        bid_val_usd = bid_px * bid_sz if not (np.isnan(bid_px) or np.isnan(bid_sz)) else 0.0
        ask_val_usd = ask_px * ask_sz if not (np.isnan(ask_px) or np.isnan(ask_sz)) else 0.0

        q_imbal_num = bid_val_usd - ask_val_usd
        q_imbal_den = bid_val_usd + ask_val_usd
        cur_q_imbal_ratio = self._safe_divide(q_imbal_num, q_imbal_den)
        features[15] = cur_q_imbal_ratio

        prev_q_imbal_ratio = self.prev_tick_values["HF_Quote_1s_Quote_Imbalance_Value_Ratio"]
        features[16] = cur_q_imbal_ratio - prev_q_imbal_ratio if not (np.isnan(cur_q_imbal_ratio) or np.isnan(prev_q_imbal_ratio)) else np.nan
        self.prev_tick_values["HF_Quote_1s_Quote_Imbalance_Value_Ratio"] = cur_q_imbal_ratio

        avg_bid_val_roll = self._rolling_mean_deque(self.recent_1s_bid_values_usd, self.HF_ROLLING_WINDOW_SECONDS)
        features[17] = self._safe_divide(bid_val_usd, avg_bid_val_roll)

        avg_ask_val_roll = self._rolling_mean_deque(self.recent_1s_ask_values_usd, self.HF_ROLLING_WINDOW_SECONDS)
        features[18] = self._safe_divide(ask_val_usd, avg_ask_val_roll)

        # Debug NaN values
        nan_indices = np.where(np.isnan(features))[0]
        if len(nan_indices) > 0:
            nan_features = [self.hf_feature_names[i] for i in nan_indices]
            self.logger.warning(f"NaN values detected in HF features: {nan_features}")
            # Print the actual calculations that led to these NaNs
            for i in nan_indices:
                feature_name = self.hf_feature_names[i]
                self.logger.warning(f"NaN in {feature_name} calculation")

        return features.astype(np.float32)

    def _calculate_mf_features(self) -> np.ndarray:
        features = np.full(self.config.mf_feat_dim, np.nan)
        cur_px_mf = self._get_current_price_for_mf_dists()

        df_1m, df_5m = self.completed_1m_bars_df, self.completed_5m_bars_df

        # Price Change Pct for 1m and 5m (Indices 0-3)
        def get_price_change_pct(df, basis):
            if df.empty or len(df) < 1: return np.nan
            last_bar = df.iloc[-1]
            close_t = float(last_bar.get('close', np.nan))
            if basis == "Close_Xm_bar_vs_PrevClose_Xm_bar":
                if len(df) < 2: return np.nan
                prev_close_t_minus_1 = float(df.iloc[-2].get('close', np.nan))
                return self._pct_change(close_t, prev_close_t_minus_1)
            elif basis == "Close_Xm_bar_vs_Open_Xm_bar":
                open_t = float(last_bar.get('open', np.nan))
                return self._pct_change(close_t, open_t)
            return np.nan

        mf_1m_px_chg_pct = get_price_change_pct(df_1m, self.MF_PRICE_CHANGE_BASIS)
        features[0] = mf_1m_px_chg_pct
        mf_5m_px_chg_pct = get_price_change_pct(df_5m, self.MF_PRICE_CHANGE_BASIS)
        features[1] = mf_5m_px_chg_pct

        prev_mf_1m_px_chg_pct = self.prev_tick_values["MF_1m_PriceChange_Pct"]
        features[2] = mf_1m_px_chg_pct - prev_mf_1m_px_chg_pct if not (np.isnan(mf_1m_px_chg_pct) or np.isnan(prev_mf_1m_px_chg_pct)) else np.nan
        self.prev_tick_values["MF_1m_PriceChange_Pct"] = mf_1m_px_chg_pct

        prev_mf_5m_px_chg_pct = self.prev_tick_values["MF_5m_PriceChange_Pct"]
        features[3] = mf_5m_px_chg_pct - prev_mf_5m_px_chg_pct if not (np.isnan(mf_5m_px_chg_pct) or np.isnan(prev_mf_5m_px_chg_pct)) else np.nan
        self.prev_tick_values["MF_5m_PriceChange_Pct"] = mf_5m_px_chg_pct

        # Position in Candle Range (Indices 4-7)
        def get_pos_in_range(price, bar_h, bar_l):
            bar_h, bar_l = float(bar_h), float(bar_l)
            candle_range = bar_h - bar_l if not (np.isnan(bar_h) or np.isnan(bar_l)) else np.nan
            return self._safe_divide(price - bar_l, candle_range)

        bar_1m_form = self.market_data.get('current_1m_bar_forming')
        if bar_1m_form and not np.isnan(cur_px_mf):
            features[4] = get_pos_in_range(cur_px_mf, bar_1m_form.get('high'), bar_1m_form.get('low'))
        bar_5m_form = self.market_data.get('current_5m_bar_forming')
        if bar_5m_form and not np.isnan(cur_px_mf):
            features[5] = get_pos_in_range(cur_px_mf, bar_5m_form.get('high'), bar_5m_form.get('low'))

        if not df_1m.empty and not np.isnan(cur_px_mf):
            prev_1m = df_1m.iloc[-1]
            features[6] = get_pos_in_range(cur_px_mf, prev_1m.get('high'), prev_1m.get('low'))
        if not df_5m.empty and not np.isnan(cur_px_mf):
            prev_5m = df_5m.iloc[-1]
            features[7] = get_pos_in_range(cur_px_mf, prev_5m.get('high'), prev_5m.get('low'))

        # EMAs (Indices 8-11)
        if len(df_1m) >= self.MF_EMA_LONG_PERIOD:
            c1 = df_1m['close'].astype(float).dropna()
            if len(c1) >= self.MF_EMA_SHORT_PERIOD: features[8] = self._calculate_pct_dist(cur_px_mf, ta.trend.EMAIndicator(c1,
                                                                                                                            window=self.MF_EMA_SHORT_PERIOD).ema_indicator().iloc[
                -1])
            if len(c1) >= self.MF_EMA_LONG_PERIOD: features[9] = self._calculate_pct_dist(cur_px_mf, ta.trend.EMAIndicator(c1,
                                                                                                                           window=self.MF_EMA_LONG_PERIOD).ema_indicator().iloc[
                -1])
        if len(df_5m) >= self.MF_EMA_LONG_PERIOD:
            c5 = df_5m['close'].astype(float).dropna()
            if len(c5) >= self.MF_EMA_SHORT_PERIOD: features[10] = self._calculate_pct_dist(cur_px_mf, ta.trend.EMAIndicator(c5,
                                                                                                                             window=self.MF_EMA_SHORT_PERIOD).ema_indicator().iloc[
                -1])
            if len(c5) >= self.MF_EMA_LONG_PERIOD: features[11] = self._calculate_pct_dist(cur_px_mf, ta.trend.EMAIndicator(c5,
                                                                                                                            window=self.MF_EMA_LONG_PERIOD).ema_indicator().iloc[
                -1])

        # Dist to Rolling HF High/Low (Indices 12-13)
        hf_bars_data = [d.get('bar') for d in self.market_data.get('rolling_1s_data_window', []) if d.get('bar')]
        if len(hf_bars_data) >= self.MF_ROLLING_HF_DATA_WINDOW_SECONDS:
            hf_df = pd.DataFrame(hf_bars_data[-self.MF_ROLLING_HF_DATA_WINDOW_SECONDS:])
            if not hf_df.empty:
                features[12] = self._calculate_pct_dist(cur_px_mf, hf_df['high'].astype(float).max())
                features[13] = self._calculate_pct_dist(cur_px_mf, hf_df['low'].astype(float).min())

        # MACD (1m) (Indices 14-16)
        if len(df_1m) >= self.MF_MACD_SLOW_PERIOD:
            c1 = df_1m['close'].astype(float).dropna()
            if len(c1) >= self.MF_MACD_SLOW_PERIOD:  # recheck after dropna
                macd = ta.trend.MACD(c1, window_slow=self.MF_MACD_SLOW_PERIOD, window_fast=self.MF_MACD_FAST_PERIOD, window_sign=self.MF_MACD_SIGNAL_PERIOD)
                features[14], features[15], features[16] = macd.macd().iloc[-1], macd.macd_signal().iloc[-1], macd.macd_diff().iloc[-1]

        # ATR (Indices 17-18)
        def get_atr_pct(df, period, price):
            if len(df) >= period and all(c in df for c in ['high', 'low', 'close']):
                hlc = df[['high', 'low', 'close']].astype(float).dropna()
                if len(hlc) >= period:  # recheck after dropna
                    atr_val = ta.volatility.AverageTrueRange(high=hlc['high'], low=hlc['low'], close=hlc['close'], window=period).average_true_range().iloc[-1]
                    return self._safe_divide(atr_val * 100.0, price)
            return np.nan

        features[17] = get_atr_pct(df_1m, self.MF_ATR_PERIOD, cur_px_mf)
        features[18] = get_atr_pct(df_5m, self.MF_ATR_PERIOD, cur_px_mf)

        # Candle Shape (Indices 19-24)
        def get_candle_shape(bar_o, bar_h, bar_l, bar_c):
            bar_o, bar_h, bar_l, bar_c = float(bar_o), float(bar_h), float(bar_l), float(bar_c)
            if any(pd.isna([bar_o, bar_h, bar_l, bar_c])): return np.nan, np.nan, np.nan
            bar_range = bar_h - bar_l
            body = self._safe_divide(abs(bar_c - bar_o), bar_range)
            upper = self._safe_divide(bar_h - max(bar_o, bar_c), bar_range)
            lower = self._safe_divide(min(bar_o, bar_c) - bar_l, bar_range)
            return body, upper, lower

        if not df_1m.empty:
            b = df_1m.iloc[-1]
            features[19], features[20], features[21] = get_candle_shape(b.get('open'), b.get('high'), b.get('low'), b.get('close'))
        if not df_5m.empty:
            b = df_5m.iloc[-1]
            features[22], features[23], features[24] = get_candle_shape(b.get('open'), b.get('high'), b.get('low'), b.get('close'))

        # MF_Xm_BarVol_Ratio_To_TodaySoFarVol (Indices 25-26) - Corrected Logic
        total_daily_vol_so_far = self.daily_cumulative_volume
        if not df_1m.empty:
            actual_1m_bar_volume = float(df_1m.iloc[-1].get('volume', np.nan))
            features[25] = self._safe_divide(actual_1m_bar_volume, total_daily_vol_so_far)

        if not df_5m.empty:
            actual_5m_bar_volume = float(df_5m.iloc[-1].get('volume', np.nan))
            features[26] = self._safe_divide(actual_5m_bar_volume, total_daily_vol_so_far)

        # MF_Xm_Volume_Rel_To_Avg_Recent_Bars (Indices 27-28)
        if len(df_1m) >= self.MF_VOLUME_AVG_RECENT_BARS_WINDOW and self.MF_VOLUME_AVG_RECENT_BARS_WINDOW > 0:
            vol1 = df_1m['volume'].astype(float)
            if len(vol1) > 1:  # Need at least one bar for current, one for mean
                # Mean of N bars *excluding* the current completed one
                avg_vol_1m_recent = vol1.iloc[-(self.MF_VOLUME_AVG_RECENT_BARS_WINDOW + 1): -1].mean()
                features[27] = self._safe_divide(vol1.iloc[-1], avg_vol_1m_recent)

        if len(df_5m) >= self.MF_VOLUME_AVG_RECENT_BARS_WINDOW and self.MF_VOLUME_AVG_RECENT_BARS_WINDOW > 0:
            vol5 = df_5m['volume'].astype(float)
            if len(vol5) > 1:
                avg_vol_5m_recent = vol5.iloc[-(self.MF_VOLUME_AVG_RECENT_BARS_WINDOW + 1): -1].mean()
                features[28] = self._safe_divide(vol5.iloc[-1], avg_vol_5m_recent)

        # Swing Levels (Indices 29-32)
        def get_dist_to_swing(df, lookback, strength, price, is_high):
            # Ensure enough data for lookback window itself, then for swing point detection within that window
            if len(df) >= lookback:
                series_for_swings = df['high' if is_high else 'low'].astype(float).iloc[-lookback:].dropna()
                # _find_swing_points needs at least 2*strength + 1 bars
                if len(series_for_swings) >= (2 * strength + 1):
                    swings_h, swings_l = self._find_swing_points(series_for_swings, strength)
                    points = swings_h if is_high else swings_l
                    if points:
                        target_level = max(points) if is_high else min(points)
                        return self._calculate_pct_dist(price, target_level)
            return np.nan

        features[29] = get_dist_to_swing(df_1m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, True)
        features[30] = get_dist_to_swing(df_1m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, False)
        features[31] = get_dist_to_swing(df_5m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, True)
        features[32] = get_dist_to_swing(df_5m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, False)

        # Debug NaN values
        nan_indices = np.where(np.isnan(features))[0]
        if len(nan_indices) > 0:
            nan_features = [self.mf_feature_names[i] for i in nan_indices]
            self.logger.warning(f"NaN values detected in HF features: {nan_features}")
            # Print the actual calculations that led to these NaNs
            for i in nan_indices:
                feature_name = self.mf_feature_names[i]
                self.logger.warning(f"NaN in {feature_name} calculation")

        return features.astype(np.float32)

    def _calculate_lf_features(self) -> np.ndarray:
        features = np.full(self.config.lf_feat_dim, np.nan)
        cur_px_lf = float(self.market_data.get('current_price', np.nan))
        df_hist_1d = self.historical_1d_bars_df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}, errors='ignore')
        if 'close_8pm' in df_hist_1d.columns: df_hist_1d['close_8pm'] = df_hist_1d['close_8pm'].astype(float)

        d_low, d_high = float(self.market_data.get('intraday_low', np.nan)), float(self.market_data.get('intraday_high', np.nan))
        d_range = d_high - d_low if not (np.isnan(d_high) or np.isnan(d_low)) else np.nan
        features[0] = self._safe_divide(cur_px_lf - d_low, d_range)

        prev_d_low = self._get_daily_bar_value(-1, 'low', self.LF_PREV_DAY_HL_SOURCE)
        prev_d_high = self._get_daily_bar_value(-1, 'high', self.LF_PREV_DAY_HL_SOURCE)
        prev_d_range = prev_d_high - prev_d_low if not (np.isnan(prev_d_high) or np.isnan(prev_d_low)) else np.nan
        features[1] = self._safe_divide(cur_px_lf - prev_d_low, prev_d_range)

        prev_d_close = self._get_daily_bar_value(-1, 'close', self.LF_PREV_DAY_CLOSE_SOURCE)
        features[2] = self._pct_change(cur_px_lf, prev_d_close)

        vol_today = self.daily_cumulative_volume
        avg_d_vol = np.nan
        if not df_hist_1d.empty and 'volume' in df_hist_1d.columns:
            lookback_days = self.LF_AVG_DAILY_VOLUME_PERIOD_DAYS
            if len(df_hist_1d) >= lookback_days:
                avg_d_vol = df_hist_1d['volume'].iloc[-lookback_days:].mean()

        frac_day_passed = np.nan
        current_ts_utc = self.market_data.get('timestamp_utc')
        if current_ts_utc:
            current_time_et = current_ts_utc.astimezone(ET_TZ)
            # Use datetime.time for comparison to avoid date part if current_ts_utc is from a different day but within hours
            session_start_time_obj = time(self.SESSION_START_HOUR_ET, 0, 0)
            current_time_obj_et = current_time_et.time()

            # Construct full datetime objects for today to calculate difference
            today_et = current_time_et.date()
            session_start_dt_et = datetime.combine(today_et, session_start_time_obj, tzinfo=ET_TZ)

            # Ensure current_time_et is also timezone-aware for proper subtraction
            current_dt_et_aware = datetime.combine(today_et, current_time_obj_et, tzinfo=ET_TZ)

            if current_dt_et_aware >= session_start_dt_et:  # Ensure we are within or past session start
                secs_in_session = (current_dt_et_aware - session_start_dt_et).total_seconds()
                if secs_in_session >= 0 and self.TOTAL_SECONDS_IN_FULL_TRADING_DAY > 0:  # secs_in_session could be slightly negative due to DST if not careful
                    frac_day_passed = min(secs_in_session / self.TOTAL_SECONDS_IN_FULL_TRADING_DAY, 1.0)
            else:  # Before session start
                frac_day_passed = 0.0

        if not any(np.isnan([vol_today, avg_d_vol, frac_day_passed])) and frac_day_passed > self.EPSILON and avg_d_vol > self.EPSILON:
            expected_vol = avg_d_vol * frac_day_passed
            features[3] = (self._safe_divide(vol_today, expected_vol, default_val=1.0) - 1.0) * 100.0  # Default to 0% RVol if expected_vol is 0
        else:
            features[3] = 0.0  # Default to 0% if cannot calculate (e.g. start of day)

        session_vwap = np.nan
        if self.LF_VWAP_SESSION_SCOPE == "FullDay_4AM_Start_NoReset":
            if self.daily_cumulative_volume > self.EPSILON:
                session_vwap = self.daily_cumulative_price_volume_product / self.daily_cumulative_volume
        features[4] = self._calculate_pct_dist(cur_px_lf, session_vwap)

        close_col_daily = 'close'  # Default
        if self.LF_DAILY_EMA_PRICE_SOURCE == "PostMarketClose_8PM" and 'close_8pm' in df_hist_1d.columns:
            close_col_daily = 'close_8pm'
        elif 'close' not in df_hist_1d.columns:
            close_col_daily = None  # Cannot proceed if no close columns

        if close_col_daily and not df_hist_1d.empty and close_col_daily in df_hist_1d.columns:
            d_closes = df_hist_1d[close_col_daily].astype(float).dropna()
            if len(d_closes) >= self.LF_DAILY_EMA_SHORT_PERIOD: features[5] = self._calculate_pct_dist(cur_px_lf, ta.trend.EMAIndicator(d_closes,
                                                                                                                                        window=self.LF_DAILY_EMA_SHORT_PERIOD).ema_indicator().iloc[
                -1])
            if len(d_closes) >= self.LF_DAILY_EMA_MEDIUM_PERIOD: features[6] = self._calculate_pct_dist(cur_px_lf, ta.trend.EMAIndicator(d_closes,
                                                                                                                                         window=self.LF_DAILY_EMA_MEDIUM_PERIOD).ema_indicator().iloc[
                -1])
            if len(d_closes) >= self.LF_DAILY_EMA_LONG_PERIOD: features[7] = self._calculate_pct_dist(cur_px_lf, ta.trend.EMAIndicator(d_closes,
                                                                                                                                       window=self.LF_DAILY_EMA_LONG_PERIOD).ema_indicator().iloc[
                -1])

        # Simplified S/R
        closest_resistance, closest_support = np.nan, np.nan
        if not df_hist_1d.empty and not np.isnan(cur_px_lf):
            # Determine number of days to look back based on LF_LT_SR_LOOKBACK_YEARS
            num_days_lookback = self.LF_LT_SR_LOOKBACK_YEARS * 252  # Approx trading days
            relevant_daily_bars = df_hist_1d.iloc[-num_days_lookback:] if len(df_hist_1d) > num_days_lookback else df_hist_1d

            # Iterate backwards from most recent historical day
            for idx in range(len(relevant_daily_bars) - 1, -1, -1):
                day_data = relevant_daily_bars.iloc[idx]
                day_high = float(day_data.get('high', np.nan))
                day_low = float(day_data.get('low', np.nan))

                if np.isnan(closest_resistance) and not np.isnan(day_high) and day_high > cur_px_lf:
                    closest_resistance = day_high
                if np.isnan(closest_support) and not np.isnan(day_low) and day_low < cur_px_lf:
                    closest_support = day_low

                if not np.isnan(closest_resistance) and not np.isnan(closest_support):
                    break  # Found both

        features[8] = self._calculate_pct_dist(cur_px_lf, closest_support)
        features[9] = self._calculate_pct_dist(cur_px_lf, closest_resistance)

        # Debug NaN values
        nan_indices = np.where(np.isnan(features))[0]
        if len(nan_indices) > 0:
            nan_features = [self.lf_feature_names[i] for i in nan_indices]
            self.logger.warning(f"NaN values detected in HF features: {nan_features}")
            # Print the actual calculations that led to these NaNs
            for i in nan_indices:
                feature_name = self.lf_feature_names[i]
                self.logger.warning(f"NaN in {feature_name} calculation")

        return features.astype(np.float32)
