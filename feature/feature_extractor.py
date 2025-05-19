# feature_extractor.py
import logging
from collections import deque
from datetime import datetime, timezone, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta  # type: ignore
from ta import trend, volatility  # type: ignore

from config.config import FeatureConfig  # Assuming your Config class is here
from simulators.market_simulator import MarketSimulator  # Assuming your MarketSimulator class is here

ET_TZ = timezone(timedelta(hours=-4))  # Example: Eastern Time, adjust if necessary


class FeatureExtractor:
    EPSILON = 1e-9
    SESSION_START_HOUR_ET = 4
    SESSION_END_HOUR_ET = 20
    REGULAR_SESSION_START_HOUR_ET = 9.5
    REGULAR_SESSION_END_HOUR_ET = 16
    TOTAL_SECONDS_IN_FULL_TRADING_DAY = (SESSION_END_HOUR_ET - SESSION_START_HOUR_ET) * 3600
    TOTAL_MINUTES_IN_FULL_TRADING_DAY = TOTAL_SECONDS_IN_FULL_TRADING_DAY / 60

    # Configuration constants (can be moved to FeatureConfig if preferred)
    HF_ROLLING_WINDOW_SECONDS = 60
    HF_PRICE_CHANGE_BASIS = "Close_1s_bar"  # Options: "Close_1s_bar", "VWAP_1s_bar"
    HF_BAR_HLC_SOURCE = "Trades_and_Quotes"  # Currently informational, not directly changing bar source
    HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL = "Bar_MidPoint"  # Options: "Bar_MidPoint", "Bar_Open", "Bar_Close"
    HF_AGGRESSOR_ID_METHOD = "LeeReady"  # Currently informational
    HF_NORMALIZED_IMBALANCE_DENOMINATOR_TYPE = "SumOfAggressorVolumes"  # Options: "SumOfAggressorVolumes", "TotalBarVolume"
    HF_LARGE_TRADE_THRESHOLD_TYPE = "FactorOfRollingAvg"
    HF_LARGE_TRADE_FACTOR = 5.0
    HF_LARGE_TRADE_ROLLING_AVG_WINDOW_SECONDS = 60

    MF_PRICE_CHANGE_BASIS = "Close_Xm_bar_vs_PrevClose_Xm_bar"  # Options: "Close_Xm_bar_vs_PrevClose_Xm_bar", "Close_Xm_bar_vs_Open_Xm_bar"
    MF_CURRENT_PRICE_SOURCE_FOR_DISTS = "LiveTickPrice"  # Options: "LiveTickPrice", "BarClosePrice"
    MF_DIST_PCT_DENOMINATOR = "Level"  # Options: "Level", "CurrentPrice"
    MF_EMA_SHORT_PERIOD = 9
    MF_EMA_LONG_PERIOD = 20
    MF_ROLLING_HF_DATA_WINDOW_SECONDS = 60
    MF_MACD_FAST_PERIOD = 12
    MF_MACD_SLOW_PERIOD = 26
    MF_MACD_SIGNAL_PERIOD = 9
    MF_ATR_PERIOD = 14
    MF_VOLUME_DAILY_AVG_PROFILE_LOOKBACK_DAYS = 20
    MF_VOLUME_AVG_RECENT_BARS_WINDOW = 20
    MF_SWING_LOOKBACK_BARS = 10
    MF_SWING_STRENGTH_BARS = 2

    LF_AVG_DAILY_VOLUME_PERIOD_DAYS = 10
    LF_PREV_DAY_HL_SOURCE = "RegularTradingHours"  # Options: "RegularTradingHours", "FullTradingDay_4AM_8PM"
    LF_PREV_DAY_CLOSE_SOURCE = "PostMarketClose_8PM"  # Options: "RegularSessionClose_4PM", "PostMarketClose_8PM"
    LF_VWAP_SESSION_SCOPE = "FullDay_4AM_Start_NoReset"
    LF_DAILY_EMA_PRICE_SOURCE = "PostMarketClose_8PM"  # Options: "RegularSessionClose_4PM", "PostMarketClose_8PM"
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
            # Assuming get_symbol_info might not be available or could fail
            symbol_info = getattr(self.market_simulator, 'get_symbol_info', lambda: None)()
            self.total_shares_outstanding: Optional[float] = symbol_info.get('total_shares_outstanding', None) if symbol_info else None
        except Exception as e:
            self.logger.warning(f"Could not get total_shares_outstanding: {e}. Market cap feature may be NaN.")
            self.total_shares_outstanding = None

        self.static_feature_names: List[str] = [
            "S_Time_Of_Day_Seconds_Encoded_Sin",
            "S_Time_Of_Day_Seconds_Encoded_Cos",
            "S_Market_Cap_Million",
            # "S_Initial_PreMarket_Gap_Pct", # Requires robust way to get first premarket tick / prev close handling
            # "S_Regular_Open_Gap_Pct",      # Requires robust way to get RTH open tick
        ]
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
            "MF_1m_BarVol_Ratio_To_TodaySoFarVol",
            "MF_5m_BarVol_Ratio_To_TodaySoFarVol",
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

        self.hf_features_history: deque = deque(maxlen=max(1, self.hf_seq_len))
        self.mf_features_history: deque = deque(maxlen=max(1, self.mf_seq_len))
        self.lf_features_history: deque = deque(maxlen=max(1, self.lf_seq_len))

        self.max_hf_lookback_seconds = max(
            self.HF_ROLLING_WINDOW_SECONDS,
            self.HF_LARGE_TRADE_ROLLING_AVG_WINDOW_SECONDS, 1
        )
        self.recent_1s_volumes: deque = deque(maxlen=self.max_hf_lookback_seconds)
        self.recent_1s_trade_counts: deque = deque(maxlen=self.max_hf_lookback_seconds)
        self.recent_1s_total_trade_shares: deque = deque(maxlen=self.max_hf_lookback_seconds)
        self.recent_1s_bid_values_usd: deque = deque(maxlen=self.HF_ROLLING_WINDOW_SECONDS)
        self.recent_1s_ask_values_usd: deque = deque(maxlen=self.HF_ROLLING_WINDOW_SECONDS)

        self.prev_tick_values: Dict[str, Any] = {
            "HF_1s_PriceChange_basis_value": np.nan, "HF_1s_Volume": np.nan,
            "HF_Tape_1s_Trades_Count": np.nan, "HF_Tape_1s_Normalized_Volume_Imbalance": np.nan,
            "HF_Tape_1s_Avg_Trade_Size": np.nan, "HF_Quote_1s_Spread_Rel": np.nan,
            "HF_Quote_1s_Quote_Imbalance_Value_Ratio": np.nan,
            "MF_1m_PriceChange_Pct": np.nan, "MF_5m_PriceChange_Pct": np.nan,
        }

        self.prev_day_close_price_for_gaps: Optional[float] = np.nan  # Used by Static Market Cap
        # For daily resets and VWAP
        self.daily_cumulative_volume: float = 0.0
        self.daily_cumulative_price_volume_product: float = 0.0
        self.last_processed_day_for_daily_reset: Optional[datetime.date] = None

        self._initialize_buffers_with_nan()
        self._initial_prev_day_close_load()

    def _initial_prev_day_close_load(self):
        """Tries to load historical_1d_bars at init to get prev_day_close for Market Cap."""
        try:
            # Attempt to get an initial market state to populate historical_1d_bars_df
            initial_market_state = self.market_simulator.get_current_market_state()
            if initial_market_state:
                hist_1d_init = initial_market_state.get('historical_1d_bars')
                if isinstance(hist_1d_init, pd.DataFrame) and not hist_1d_init.empty:
                    self.historical_1d_bars_df = self._ensure_datetime_index(hist_1d_init.copy())
                    self._update_prev_day_close_for_gaps()
                else:
                    self.logger.warning("Initial historical_1d_bars is empty or not a DataFrame. Prev day close might be initially NaN.")
                    self.historical_1d_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm'])  # Ensure schema if empty
            else:
                self.logger.warning("Initial market state is None. Prev day close might be initially NaN.")
                self.historical_1d_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm'])
        except Exception as e:
            self.logger.error(f"Error during _initial_prev_day_close_load: {e}")
            self.historical_1d_bars_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm'])

    def _ensure_datetime_index(self, df: pd.DataFrame, ts_col_name: str = 'timestamp') -> pd.DataFrame:
        if df.empty:
            if ts_col_name not in df.columns and df.index.name != ts_col_name:  # if completely empty and no ts column or index
                df = pd.DataFrame(columns=[ts_col_name] + [c for c in df.columns if c != ts_col_name])  # add ts_col
                df[ts_col_name] = pd.to_datetime(df[ts_col_name])
                df = df.set_index(ts_col_name)
            return df

        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tzinfo != timezone.utc:
                df.index = df.index.tz_convert('UTC')
        elif ts_col_name in df.columns:
            df[ts_col_name] = pd.to_datetime(df[ts_col_name])
            if df[ts_col_name].dt.tz is None:
                df[ts_col_name] = df[ts_col_name].dt.tz_localize('UTC')
            elif df[ts_col_name].dt.tz != timezone.utc:
                df[ts_col_name] = df[ts_col_name].dt.tz_convert('UTC')
            df = df.set_index(ts_col_name)
        else:  # If no ts_col_name and index is not datetime, it's problematic
            self.logger.warning(f"DataFrame missing '{ts_col_name}' column and index is not DatetimeIndex. Indexing may fail.")
        return df

    def _initialize_buffers_with_nan(self):
        nan_hf = np.full(self.config.hf_feat_dim, np.nan)
        for _ in range(self.hf_features_history.maxlen or 0): self.hf_features_history.append(nan_hf.copy())
        nan_mf = np.full(self.config.mf_feat_dim, np.nan)
        for _ in range(self.mf_features_history.maxlen or 0): self.mf_features_history.append(nan_mf.copy())
        nan_lf = np.full(self.config.lf_feat_dim, np.nan)
        for _ in range(self.lf_features_history.maxlen or 0): self.lf_features_history.append(nan_lf.copy())

    def _safe_get_from_bar(self, bar_data: Optional[Dict], key: str, default: Any = np.nan) -> Any:
        if bar_data is None: return default
        val = bar_data.get(key)
        if val is None: return default  # Handles if key exists but value is None
        try:
            return float(val)  # Attempt to convert to float, common case
        except (ValueError, TypeError):
            # If it's not float convertible but not None (e.g. a string timestamp), return as is or handle
            if isinstance(val, (datetime, pd.Timestamp)): return val  # Return datetimes as is
            return default  # Fallback for non-convertible non-None types if float is expected

    def _update_market_data(self):
        self.market_data = self.market_simulator.get_current_market_state()
        if not self.market_data:
            self.logger.warning("Market data is None. Cannot update feature extractor state.")
            # Potentially fill all internal states with defaults or NaNs if this happens
            return

        current_ts_utc = self.market_data.get('timestamp_utc')

        if current_ts_utc:
            current_date = current_ts_utc.date()
            if self.last_processed_day_for_daily_reset is None or self.last_processed_day_for_daily_reset != current_date:
                self.logger.info(f"New trading day {current_date}. Resetting daily state for VWAP etc.")
                self.daily_cumulative_volume = 0.0
                self.daily_cumulative_price_volume_product = 0.0
                # self.s_initial_premarket_gap_pct = None # Re-evaluate on new day, if feature enabled
                # self.s_regular_open_gap_pct = None    # Re-evaluate on new day, if feature enabled
                self.last_processed_day_for_daily_reset = current_date

                hist_1d_update = self.market_data.get('historical_1d_bars')
                if isinstance(hist_1d_update, pd.DataFrame):  # No need for not empty check, _ensure_datetime_index handles empty
                    self.historical_1d_bars_df = self._ensure_datetime_index(hist_1d_update.copy())
                elif self.historical_1d_bars_df.empty:  # If no update and it was already empty
                    self.historical_1d_bars_df = self._ensure_datetime_index(pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm']))

                self._update_prev_day_close_for_gaps()  # Update prev day close for new day

        # Update completed bars DataFrames
        def _df_from_window(window_data_list: Optional[List[Dict]], cols: List[str], ts_col_name: str = 'timestamp_start') -> pd.DataFrame:
            if window_data_list and isinstance(window_data_list, list):
                df = pd.DataFrame(window_data_list)
                if not df.empty:
                    return self._ensure_datetime_index(df, ts_col_name)
            # Return empty DataFrame with correct columns and index if window_data is None or empty
            empty_df_cols = cols + ([ts_col_name] if ts_col_name not in cols else [])
            empty_df = pd.DataFrame(columns=empty_df_cols)
            return self._ensure_datetime_index(empty_df, ts_col_name)

        self.completed_1m_bars_df = _df_from_window(
            self.market_data.get('completed_1m_bars_window'),
            ['open', 'high', 'low', 'close', 'volume'], 'timestamp_start'
        )
        self.completed_5m_bars_df = _df_from_window(
            self.market_data.get('completed_5m_bars_window'),
            ['open', 'high', 'low', 'close', 'volume'], 'timestamp_start'
        )

        # Ensure historical_1d_bars_df is the latest version
        hist_1d_live = self.market_data.get('historical_1d_bars')
        if isinstance(hist_1d_live, pd.DataFrame):
            self.historical_1d_bars_df = self._ensure_datetime_index(hist_1d_live.copy())
        elif self.historical_1d_bars_df.empty:  # If no live update and it was already empty
            self.historical_1d_bars_df = self._ensure_datetime_index(pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'close_8pm']))

        # Update rolling 1s deques
        current_1s_bar = self.market_data.get('current_1s_bar')  # This can be None
        rolling_1s_data_window = self.market_data.get('rolling_1s_data_window', [])

        # Default for appending to deques if no current_1s_bar activity
        default_val_deque = 0.0

        if current_1s_bar:  # if there is a bar for the current second
            self.recent_1s_volumes.append(self._safe_get_from_bar(current_1s_bar, 'volume', default_val_deque))

            # Trades and quotes are within the 'rolling_1s_data_window' items
            # The last item in rolling_1s_data_window corresponds to the current second's detailed data
            last_1s_detailed_event = None
            if rolling_1s_data_window and isinstance(rolling_1s_data_window, list):
                # Check if timestamps match, or just assume last entry is for current_1s_bar
                # For simplicity, let's assume if current_1s_bar exists, the relevant trades/quotes
                # are in the last element of rolling_1s_data_window if its timestamp matches.
                # Or, more simply, the problem implies 'trades' and 'quotes' lists within each element of rolling_1s_data_window.
                # The features are usually calculated based on the *most recent* 1s data.
                # If current_1s_bar is the primary source, we need trades/quotes for *this specific second*.
                # The data structure implies rolling_1s_data_window[-1] contains the trades/quotes for current_1s_bar.
                if rolling_1s_data_window:
                    last_1s_detailed_event = rolling_1s_data_window[-1]

            trades_this_sec = last_1s_detailed_event.get('trades', []) if last_1s_detailed_event else []
            self.recent_1s_trade_counts.append(float(len(trades_this_sec)))
            self.recent_1s_total_trade_shares.append(sum(self._safe_get_from_bar(t, 'size', 0.0) for t in trades_this_sec))
        else:  # No current_1s_bar, so no volume, trades for this exact second
            self.recent_1s_volumes.append(default_val_deque)
            self.recent_1s_trade_counts.append(default_val_deque)
            self.recent_1s_total_trade_shares.append(default_val_deque)

        # Update BBO related deques based on top-level BBO info (most recent for the timestamp)
        # These are L1 quotes, not from the 'quotes' list in rolling_1s_data_window unless specified
        best_bid_p = self.market_data.get('best_bid_price')
        best_bid_s = self.market_data.get('best_bid_size')
        best_ask_p = self.market_data.get('best_ask_price')
        best_ask_s = self.market_data.get('best_ask_size')

        bid_val = 0.0
        if best_bid_p is not None and best_bid_s is not None:
            try:
                bid_val = float(best_bid_p) * float(best_bid_s)
            except (ValueError, TypeError):
                pass
        self.recent_1s_bid_values_usd.append(bid_val)

        ask_val = 0.0
        if best_ask_p is not None and best_ask_s is not None:
            try:
                ask_val = float(best_ask_p) * float(best_ask_s)
            except (ValueError, TypeError):
                pass
        self.recent_1s_ask_values_usd.append(ask_val)

        # Daily VWAP accumulation
        if self.LF_VWAP_SESSION_SCOPE == "FullDay_4AM_Start_NoReset" and current_1s_bar:
            vol_1s = self._safe_get_from_bar(current_1s_bar, 'volume', 0.0)
            if vol_1s > self.EPSILON:
                price_vol_prod_1s = 0.0
                vwap_1s = self._safe_get_from_bar(current_1s_bar, 'vwap', np.nan)

                if not np.isnan(vwap_1s):
                    price_vol_prod_1s = vwap_1s * vol_1s
                else:  # Fallback if vwap is not in bar
                    h_val = self._safe_get_from_bar(current_1s_bar, 'high', np.nan)
                    l_val = self._safe_get_from_bar(current_1s_bar, 'low', np.nan)
                    c_val = self._safe_get_from_bar(current_1s_bar, 'close', np.nan)  # Final fallback price

                    price_to_use = np.nan
                    if not np.isnan(h_val) and not np.isnan(l_val):
                        price_to_use = (h_val + l_val) / 2.0
                    elif not np.isnan(c_val):
                        price_to_use = c_val

                    if not np.isnan(price_to_use):
                        price_vol_prod_1s = price_to_use * vol_1s

                self.daily_cumulative_price_volume_product += price_vol_prod_1s
                self.daily_cumulative_volume += vol_1s

    def _update_prev_day_close_for_gaps(self):
        """Updates self.prev_day_close_price_for_gaps based on historical_1d_bars_df."""
        if self.historical_1d_bars_df.empty:
            self.prev_day_close_price_for_gaps = np.nan
            return
        try:
            # Ensure DataFrame index is DatetimeIndex and sorted for iloc[-1] to be meaningful
            # This is now handled by _ensure_datetime_index
            if not isinstance(self.historical_1d_bars_df.index, pd.DatetimeIndex):
                self.logger.warning("historical_1d_bars_df index is not DatetimeIndex after _ensure_datetime_index. Prev day data might be incorrect.")
                self.prev_day_close_price_for_gaps = np.nan
                return

            # We need the close of the day *before* the current processing day
            # If last_processed_day_for_daily_reset is today, we need data from "yesterday"
            current_processing_date = self.last_processed_day_for_daily_reset
            if current_processing_date is None and self.market_data.get('timestamp_utc'):  # If first run
                current_processing_date = self.market_data.get('timestamp_utc').date()

            if current_processing_date:
                # Find the bar for the day immediately preceding current_processing_date
                # This requires historical_1d_bars_df to be sorted by date
                # And index to be easily searchable.
                target_prev_date = current_processing_date - timedelta(days=1)
                # Search backwards for the first available date <= target_prev_date
                prev_day_data_series = None
                for dt in self.historical_1d_bars_df.index[::-1]:  # Iterate reverse
                    if dt.date() <= target_prev_date:
                        prev_day_data_series = self.historical_1d_bars_df.loc[dt]
                        break

                if prev_day_data_series is not None:
                    col_map = {"PostMarketClose_8PM": "close_8pm", "RegularSessionClose_4PM": "close"}
                    close_col_key = self.LF_PREV_DAY_CLOSE_SOURCE
                    actual_col_name = col_map.get(close_col_key, 'close')  # Default to 'close'

                    if actual_col_name in prev_day_data_series:
                        self.prev_day_close_price_for_gaps = self._safe_get_from_bar(prev_day_data_series, actual_col_name, np.nan)
                    elif 'close' in prev_day_data_series:  # Fallback to 'close' if specific one not found
                        self.prev_day_close_price_for_gaps = self._safe_get_from_bar(prev_day_data_series, 'close', np.nan)
                    else:
                        self.prev_day_close_price_for_gaps = np.nan
                        self.logger.warning(f"Previous day close column ('{actual_col_name}' or 'close') not found for {target_prev_date}.")
                else:
                    self.prev_day_close_price_for_gaps = np.nan
                    self.logger.warning(f"No historical data found for or before {target_prev_date} to determine previous day's close.")
            else:
                self.prev_day_close_price_for_gaps = np.nan  # Should not happen if _update_market_data runs
                self.logger.warning("current_processing_date is None, cannot determine previous day's close.")

        except Exception as e:
            self.prev_day_close_price_for_gaps = np.nan
            self.logger.error(f"Exception fetching previous day close for gaps: {e}", exc_info=True)

    def _get_price_for_hf_change_basis(self, current_1s_bar: Optional[Dict]) -> float:
        if current_1s_bar is None: return np.nan
        key_to_use = 'close'
        if self.HF_PRICE_CHANGE_BASIS == "VWAP_1s_bar":
            key_to_use = 'vwap'
        return self._safe_get_from_bar(current_1s_bar, key_to_use, np.nan)

    def _get_current_price_for_mf_dists(self) -> float:
        # current_price is the latest trade price or mid-quote, usually L1
        if self.MF_CURRENT_PRICE_SOURCE_FOR_DISTS == "LiveTickPrice":
            live_price = self.market_data.get('current_price')
            return float(live_price) if live_price is not None else np.nan
        elif self.MF_CURRENT_PRICE_SOURCE_FOR_DISTS == "BarClosePrice":
            # Use close of the most recent 1s bar available
            current_1s_bar = self.market_data.get('current_1s_bar')
            return self._safe_get_from_bar(current_1s_bar, 'close', np.nan)
        return np.nan

    def _calculate_pct_dist(self, price: Optional[float], level: Optional[float]) -> float:
        if price is None or level is None or np.isnan(price) or np.isnan(level):
            return np.nan

        denom = np.nan
        if self.MF_DIST_PCT_DENOMINATOR == "Level":
            denom = level
        elif self.MF_DIST_PCT_DENOMINATOR == "CurrentPrice":
            denom = price

        if np.isnan(denom) or abs(denom) < self.EPSILON:
            return np.nan
        return ((price - level) / denom) * 100.0

    def _safe_divide(self, numerator: Optional[float], denominator: Optional[float], default_val: float = np.nan) -> float:
        if numerator is None or denominator is None or \
                np.isnan(numerator) or np.isnan(denominator) or \
                abs(float(denominator)) < self.EPSILON:
            return default_val
        return float(numerator) / float(denominator)

    def _pct_change(self, current_val: Optional[float], previous_val: Optional[float], default_val: float = np.nan) -> float:
        if current_val is None or previous_val is None or \
                np.isnan(current_val) or np.isnan(previous_val) or \
                abs(float(previous_val)) < self.EPSILON:
            return default_val
        return ((float(current_val) - float(previous_val)) / float(previous_val)) * 100.0

    def _rolling_mean_deque(self, dq: deque, window: Optional[int] = None) -> float:
        if not dq: return np.nan

        elements_to_consider = list(dq)
        if window is not None and window > 0:  # Ensure window is positive
            actual_elements = elements_to_consider[-window:]
        else:  # If window is None or 0, use all elements
            actual_elements = elements_to_consider

        # Filter out None and NaN before attempting mean
        valid_floats = [el for el in actual_elements if el is not None and not np.isnan(el)]

        return np.mean(valid_floats) if valid_floats else np.nan

    def _get_daily_bar_value(self, date_offset: int, column_name: str, source_pref: Optional[str] = None) -> float:
        """
        Get a value from historical_1d_bars_df.
        date_offset: -1 for previous day, -2 for day before, etc. 0 for current (if available, though usually not complete).
        """
        if self.historical_1d_bars_df.empty: return np.nan

        try:
            # Ensure index is sorted for iloc to be meaningful relative to 'current day'
            # (assuming historical_1d_bars_df is sorted ascending by date)
            target_iloc_idx = len(self.historical_1d_bars_df) - 1 + date_offset  # -1 for last row, so -1 + (-1) = -2 for prev day

            if not (0 <= target_iloc_idx < len(self.historical_1d_bars_df)):
                self.logger.debug(
                    f"Not enough historical daily bars for date_offset {date_offset}. Have {len(self.historical_1d_bars_df)}, need index {target_iloc_idx}.")
                return np.nan

            day_data_series = self.historical_1d_bars_df.iloc[target_iloc_idx]

            col_to_get = column_name
            # Preferences for specific column names based on source_pref
            if column_name == 'high':
                if source_pref == "FullTradingDay_4AM_8PM" and 'full_day_high' in day_data_series: col_to_get = 'full_day_high'
            elif column_name == 'low':
                if source_pref == "FullTradingDay_4AM_8PM" and 'full_day_low' in day_data_series: col_to_get = 'full_day_low'
            elif column_name == 'close':
                col_map = {"PostMarketClose_8PM": "close_8pm", "RegularSessionClose_4PM": "close"}
                if source_pref in col_map and col_map[source_pref] in day_data_series:
                    col_to_get = col_map[source_pref]
                elif 'close' not in day_data_series and 'close_8pm' in day_data_series:  # fallback if preferred not there
                    col_to_get = 'close_8pm'

            val = day_data_series.get(col_to_get, day_data_series.get(column_name, np.nan))  # Fallback to original column_name
            return float(val) if not pd.isna(val) else np.nan

        except (IndexError, KeyError) as e:
            self.logger.warning(f"Error in _get_daily_bar_value for offset {date_offset}, col '{column_name}': {e}")
            return np.nan
        except Exception as e_gen:
            self.logger.error(f"Generic error in _get_daily_bar_value: {e_gen}", exc_info=True)
            return np.nan

    def _find_swing_points(self, prices: pd.Series, strength: int) -> tuple[List[float], List[float]]:
        if prices.empty or len(prices) < (2 * strength + 1): return [], []

        # Ensure prices is float and handle NaNs by dropping them for swing detection
        # Swings should be based on actual price points.
        clean_prices = prices.astype(float).dropna()
        if len(clean_prices) < (2 * strength + 1): return [], []  # Not enough valid data points

        highs, lows = [], []
        for i in range(strength, len(clean_prices) - strength):
            window = clean_prices.iloc[i - strength: i + strength + 1]
            current_price_val = clean_prices.iloc[i]

            # Max/min of a window containing NaNs will skip NaNs by default in pandas Series.
            if current_price_val == window.max(): highs.append(current_price_val)
            if current_price_val == window.min(): lows.append(current_price_val)
        return highs, lows

    def extract_features(self) -> Dict[str, np.ndarray]:
        self._update_market_data()

        static_vec = self._calculate_static_features()
        hf_vec = self._calculate_hf_features()
        mf_vec = self._calculate_mf_features()
        lf_vec = self._calculate_lf_features()

        # Replace NaNs with zeros for model stability
        hf_vec = np.nan_to_num(hf_vec, nan=0.0)
        mf_vec = np.nan_to_num(mf_vec, nan=0.0)
        lf_vec = np.nan_to_num(lf_vec, nan=0.0)
        static_vec = np.nan_to_num(static_vec, nan=0.0)

        # Update feature history with sanitized vectors
        self.hf_features_history.append(hf_vec)
        self.mf_features_history.append(mf_vec)
        self.lf_features_history.append(lf_vec)

        # Ensure all historical data is also NaN-free
        hf_array = np.array(self.hf_features_history, dtype=np.float32)
        mf_array = np.array(self.mf_features_history, dtype=np.float32)
        lf_array = np.array(self.lf_features_history, dtype=np.float32)

        return {
            'hf': np.nan_to_num(hf_array, nan=0.0),
            'mf': np.nan_to_num(mf_array, nan=0.0),
            'lf': np.nan_to_num(lf_array, nan=0.0),
            'static': np.nan_to_num(static_vec.reshape(1, -1), nan=0.0).astype(np.float32),
        }

    def _calculate_static_features(self) -> np.ndarray:
        features = np.full(self.config.static_feat_dim, np.nan)
        current_ts_utc = self.market_data.get('timestamp_utc')

        # Time of Day Encoding
        if current_ts_utc:
            try:
                current_time_et = current_ts_utc.astimezone(ET_TZ)
                # Seconds from midnight ET
                secs_from_midnight_et = current_time_et.hour * 3600 + current_time_et.minute * 60 + current_time_et.second
                # Seconds from trading session start (e.g., 4 AM ET)
                session_start_secs_et = self.SESSION_START_HOUR_ET * 3600
                secs_from_session_start = secs_from_midnight_et - session_start_secs_et

                # Normalize to [0, 2*pi] range over the full trading day duration
                angle = (2 * np.pi * secs_from_session_start) / self.TOTAL_SECONDS_IN_FULL_TRADING_DAY
                features[0] = np.sin(angle)  # S_Time_Of_Day_Seconds_Encoded_Sin
                features[1] = np.cos(angle)  # S_Time_Of_Day_Seconds_Encoded_Cos
            except Exception as e:
                self.logger.warning(f"Static feature: Time of day calculation error: {e}")

        # Market Cap
        # self.prev_day_close_price_for_gaps is updated by _update_prev_day_close_for_gaps
        if self.total_shares_outstanding is not None and \
                self.prev_day_close_price_for_gaps is not None and \
                not np.isnan(self.prev_day_close_price_for_gaps):
            market_cap = self.total_shares_outstanding * self.prev_day_close_price_for_gaps
            features[2] = market_cap / 1_000_000.0  # S_Market_Cap_Million
        else:
            if self.total_shares_outstanding is None: self.logger.debug("Market Cap: TotalSharesOutstanding missing.")
            if self.prev_day_close_price_for_gaps is None or np.isnan(self.prev_day_close_price_for_gaps): self.logger.debug(
                "Market Cap: Prev day close unavailable.")
            features[2] = np.nan

        # Note: Gap features (S_Initial_PreMarket_Gap_Pct, S_Regular_Open_Gap_Pct) are commented out
        # as they require robust fetching of specific historical prices (first premarket tick, RTH open)
        # which can be complex and depends on the MarketSimulator's capabilities.
        # If implemented, ensure _update_prev_day_close_for_gaps provides the correct previous day's close.

        return features.astype(np.float32)

    def _calculate_hf_features(self) -> np.ndarray:
        features = np.full(self.config.hf_feat_dim, np.nan)
        current_1s_bar = self.market_data.get('current_1s_bar')  # Can be None
        rolling_1s_data = self.market_data.get('rolling_1s_data_window', [])  # List of dicts

        # Price Change Pct (Feature 0)
        cur_px_basis = self._get_price_for_hf_change_basis(current_1s_bar)  # Handles None bar
        prev_px_basis = self.prev_tick_values["HF_1s_PriceChange_basis_value"]
        features[0] = self._pct_change(cur_px_basis, prev_px_basis)
        self.prev_tick_values["HF_1s_PriceChange_basis_value"] = cur_px_basis if not np.isnan(cur_px_basis) else prev_px_basis

        # Volume Features (Features 1, 2)
        cur_vol = self._safe_get_from_bar(current_1s_bar, 'volume', 0.0)  # Default to 0 volume if bar/key missing
        avg_vol_roll = self._rolling_mean_deque(self.recent_1s_volumes, self.HF_ROLLING_WINDOW_SECONDS)
        features[1] = self._safe_divide(cur_vol, avg_vol_roll)  # HF_1s_Volume_Ratio_To_Own_Avg

        prev_vol = self.prev_tick_values["HF_1s_Volume"]
        features[2] = self._pct_change(cur_vol, prev_vol)  # HF_1s_Volume_Delta_Pct
        self.prev_tick_values["HF_1s_Volume"] = cur_vol  # Already float or 0.0

        # HighLow Spread (Feature 3)
        h = self._safe_get_from_bar(current_1s_bar, 'high', np.nan)
        l = self._safe_get_from_bar(current_1s_bar, 'low', np.nan)
        spread_abs = h - l if not (np.isnan(h) or np.isnan(l)) else np.nan

        den_spread_rel = np.nan
        if not np.isnan(h) and not np.isnan(l):  # Ensure h,l are valid for mid-point
            if self.HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL == "Bar_MidPoint":
                den_spread_rel = (h + l) / 2.0
            elif self.HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL == "Bar_Open":
                den_spread_rel = self._safe_get_from_bar(current_1s_bar, 'open', np.nan)
            elif self.HF_BAR_MIDPRICE_DENOMINATOR_FOR_SPREAD_REL == "Bar_Close":
                den_spread_rel = self._safe_get_from_bar(current_1s_bar, 'close', np.nan)
        features[3] = self._safe_divide(spread_abs, den_spread_rel)  # HF_1s_HighLow_Spread_Rel

        # Tape Features from rolling_1s_data_window's last element
        # Assuming the last element of rolling_1s_data_window contains trades/quotes for the current_1s_bar period
        trades_this_second: List[Dict] = []
        if rolling_1s_data and isinstance(rolling_1s_data, list):
            last_1s_event_data = rolling_1s_data[-1]  # Get the data for the most recent second in the window
            if isinstance(last_1s_event_data, dict):
                trades_this_second = last_1s_event_data.get('trades', [])  # Default to empty list

        cur_trades_cnt = float(len(trades_this_second))

        avg_trades_cnt_roll = self._rolling_mean_deque(self.recent_1s_trade_counts, self.HF_ROLLING_WINDOW_SECONDS)
        features[4] = self._safe_divide(cur_trades_cnt, avg_trades_cnt_roll)  # HF_Tape_1s_Trades_Count_Ratio_To_Own_Avg

        prev_trades_cnt = self.prev_tick_values["HF_Tape_1s_Trades_Count"]
        features[5] = self._pct_change(cur_trades_cnt, prev_trades_cnt)  # HF_Tape_1s_Trades_Count_Delta_Pct
        self.prev_tick_values["HF_Tape_1s_Trades_Count"] = cur_trades_cnt

        # Volume Imbalance (Feature 6, 7)
        buy_vol = sum(self._safe_get_from_bar(t, 'size', 0.0) for t in trades_this_second if t.get('side') == 'B')
        sell_vol = sum(self._safe_get_from_bar(t, 'size', 0.0) for t in trades_this_second if t.get('side') == 'A')

        imbal_num = buy_vol - sell_vol
        imbal_den = np.nan
        if self.HF_NORMALIZED_IMBALANCE_DENOMINATOR_TYPE == "SumOfAggressorVolumes":
            imbal_den = buy_vol + sell_vol
        elif self.HF_NORMALIZED_IMBALANCE_DENOMINATOR_TYPE == "TotalBarVolume":  # cur_vol already calculated
            imbal_den = cur_vol

        # If no trades, imbal_num=0, imbal_den=0. Imbalance is 0.
        cur_norm_vol_imbal = self._safe_divide(imbal_num, imbal_den, default_val=0.0)
        features[6] = cur_norm_vol_imbal  # HF_Tape_1s_Normalized_Volume_Imbalance

        prev_norm_vol_imbal = self.prev_tick_values["HF_Tape_1s_Normalized_Volume_Imbalance"]
        features[7] = cur_norm_vol_imbal - prev_norm_vol_imbal if not (np.isnan(cur_norm_vol_imbal) or np.isnan(prev_norm_vol_imbal)) else np.nan
        self.prev_tick_values["HF_Tape_1s_Normalized_Volume_Imbalance"] = cur_norm_vol_imbal if not np.isnan(cur_norm_vol_imbal) else prev_norm_vol_imbal

        # Average Trade Size (Feature 8, 9)
        total_shares_1s = sum(self._safe_get_from_bar(t, 'size', 0.0) for t in trades_this_second)
        cur_avg_trade_size = self._safe_divide(total_shares_1s, cur_trades_cnt, default_val=0.0)  # Avg size is 0 if no trades

        roll_total_shares = self._rolling_mean_deque(self.recent_1s_total_trade_shares,
                                                     self.HF_ROLLING_WINDOW_SECONDS) * self.HF_ROLLING_WINDOW_SECONDS  # sum approx
        roll_total_counts = self._rolling_mean_deque(self.recent_1s_trade_counts, self.HF_ROLLING_WINDOW_SECONDS) * self.HF_ROLLING_WINDOW_SECONDS  # sum approx
        avg_trade_size_roll = self._safe_divide(roll_total_shares, roll_total_counts, default_val=0.0)

        features[8] = self._safe_divide(cur_avg_trade_size, avg_trade_size_roll)  # HF_Tape_1s_Avg_Trade_Size_Ratio_To_Own_Avg

        prev_avg_trade_size = self.prev_tick_values["HF_Tape_1s_Avg_Trade_Size"]
        features[9] = self._pct_change(cur_avg_trade_size, prev_avg_trade_size)  # HF_Tape_1s_Avg_Trade_Size_Delta_Pct
        self.prev_tick_values["HF_Tape_1s_Avg_Trade_Size"] = cur_avg_trade_size  # Already float or 0.0

        # Large Trades (Feature 10, 11)
        large_trade_thresh = np.nan
        if self.HF_LARGE_TRADE_THRESHOLD_TYPE == "FactorOfRollingAvg":
            # Use avg_trade_size_roll already computed, more robust than recomputing sums from deque here
            if not np.isnan(avg_trade_size_roll) and avg_trade_size_roll > self.EPSILON:  # Ensure avg is positive
                large_trade_thresh = self.HF_LARGE_TRADE_FACTOR * avg_trade_size_roll

        large_trades_cnt_val = 0.0
        large_buy_vol_val = 0.0
        large_sell_vol_val = 0.0
        if not np.isnan(large_trade_thresh):
            for t in trades_this_second:
                trade_size = self._safe_get_from_bar(t, 'size', 0.0)
                if trade_size > large_trade_thresh:
                    large_trades_cnt_val += 1
                    if t.get('side') == 'B':
                        large_buy_vol_val += trade_size
                    elif t.get('side') == 'A':
                        large_sell_vol_val += trade_size
        features[10] = large_trades_cnt_val  # HF_Tape_1s_Large_Trade_Count
        features[11] = self._safe_divide(large_buy_vol_val - large_sell_vol_val, cur_vol)  # HF_Tape_1s_Large_Trade_Net_Volume_Ratio_To_Total_Vol

        # Trades VWAP (Feature 12)
        features[12] = self._safe_get_from_bar(current_1s_bar, 'vwap', np.nan)  # HF_Tape_1s_Trades_VWAP

        # Quote Features (using top-level BBO from market_data)
        bid_px = self.market_data.get('best_bid_price')  # Can be None
        ask_px = self.market_data.get('best_ask_price')  # Can be None
        bid_sz = self.market_data.get('best_bid_size')  # Can be None
        ask_sz = self.market_data.get('best_ask_size')  # Can be None

        # Convert to float safely, defaulting to np.nan if None
        f_bid_px = float(bid_px) if bid_px is not None else np.nan
        f_ask_px = float(ask_px) if ask_px is not None else np.nan
        f_bid_sz = float(bid_sz) if bid_sz is not None else 0.0  # Size 0 if None
        f_ask_sz = float(ask_sz) if ask_sz is not None else 0.0  # Size 0 if None

        spread_abs_q = f_ask_px - f_bid_px if not (np.isnan(f_ask_px) or np.isnan(f_bid_px)) else np.nan
        mid_px_q = (f_ask_px + f_bid_px) / 2.0 if not (np.isnan(f_ask_px) or np.isnan(f_bid_px)) else np.nan
        cur_spread_rel_q = self._safe_divide(spread_abs_q, mid_px_q)
        features[13] = cur_spread_rel_q  # HF_Quote_1s_Spread_Rel

        prev_spread_rel_q = self.prev_tick_values["HF_Quote_1s_Spread_Rel"]
        features[14] = cur_spread_rel_q - prev_spread_rel_q if not (np.isnan(cur_spread_rel_q) or np.isnan(prev_spread_rel_q)) else np.nan
        self.prev_tick_values["HF_Quote_1s_Spread_Rel"] = cur_spread_rel_q if not np.isnan(cur_spread_rel_q) else prev_spread_rel_q

        # Quote Imbalance (Feature 15, 16)
        bid_val_usd = f_bid_px * f_bid_sz if not np.isnan(f_bid_px) else 0.0  # Value is 0 if price is NaN
        ask_val_usd = f_ask_px * f_ask_sz if not np.isnan(f_ask_px) else 0.0  # Value is 0 if price is NaN

        q_imbal_num = bid_val_usd - ask_val_usd
        q_imbal_den = bid_val_usd + ask_val_usd
        # If no quotes (bid_val_usd=0, ask_val_usd=0), then imbalance is 0.
        cur_q_imbal_ratio = self._safe_divide(q_imbal_num, q_imbal_den, default_val=0.0)
        features[15] = cur_q_imbal_ratio  # HF_Quote_1s_Quote_Imbalance_Value_Ratio

        prev_q_imbal_ratio = self.prev_tick_values["HF_Quote_1s_Quote_Imbalance_Value_Ratio"]
        features[16] = cur_q_imbal_ratio - prev_q_imbal_ratio if not (np.isnan(cur_q_imbal_ratio) or np.isnan(prev_q_imbal_ratio)) else np.nan
        self.prev_tick_values["HF_Quote_1s_Quote_Imbalance_Value_Ratio"] = cur_q_imbal_ratio if not np.isnan(cur_q_imbal_ratio) else prev_q_imbal_ratio

        # Bid/Ask Value Ratios (Feature 17, 18)
        avg_bid_val_roll = self._rolling_mean_deque(self.recent_1s_bid_values_usd, self.HF_ROLLING_WINDOW_SECONDS)
        features[17] = self._safe_divide(bid_val_usd, avg_bid_val_roll)  # HF_Quote_1s_Bid_Value_USD_Ratio_To_Own_Avg

        avg_ask_val_roll = self._rolling_mean_deque(self.recent_1s_ask_values_usd, self.HF_ROLLING_WINDOW_SECONDS)
        features[18] = self._safe_divide(ask_val_usd, avg_ask_val_roll)  # HF_Quote_1s_Ask_Value_USD_Ratio_To_Own_Avg

        return features.astype(np.float32)

    def _calculate_mf_features(self) -> np.ndarray:
        features = np.full(self.config.mf_feat_dim, np.nan)
        cur_px_mf = self._get_current_price_for_mf_dists()  # Can be NaN

        df_1m = self.completed_1m_bars_df  # Already DataFrame from _update_market_data
        df_5m = self.completed_5m_bars_df  # Already DataFrame from _update_market_data

        # Price Change Pct for 1m and 5m (Indices 0-3)
        def get_price_change_pct(df: pd.DataFrame, basis: str) -> float:
            if df.empty or not isinstance(df.index, pd.DatetimeIndex) or len(df) < 1: return np.nan

            last_bar = df.iloc[-1]  # Assuming df is sorted by time
            close_t = self._safe_get_from_bar(last_bar, 'close', np.nan)

            if basis == "Close_Xm_bar_vs_PrevClose_Xm_bar":
                if len(df) < 2: return np.nan
                prev_bar = df.iloc[-2]
                prev_close_t_minus_1 = self._safe_get_from_bar(prev_bar, 'close', np.nan)
                return self._pct_change(close_t, prev_close_t_minus_1)
            elif basis == "Close_Xm_bar_vs_Open_Xm_bar":
                open_t = self._safe_get_from_bar(last_bar, 'open', np.nan)
                return self._pct_change(close_t, open_t)
            return np.nan

        mf_1m_px_chg_pct = get_price_change_pct(df_1m, self.MF_PRICE_CHANGE_BASIS)
        features[0] = mf_1m_px_chg_pct
        mf_5m_px_chg_pct = get_price_change_pct(df_5m, self.MF_PRICE_CHANGE_BASIS)
        features[1] = mf_5m_px_chg_pct

        prev_mf_1m_px_chg_pct = self.prev_tick_values["MF_1m_PriceChange_Pct"]
        features[2] = mf_1m_px_chg_pct - prev_mf_1m_px_chg_pct if not (np.isnan(mf_1m_px_chg_pct) or np.isnan(prev_mf_1m_px_chg_pct)) else np.nan
        self.prev_tick_values["MF_1m_PriceChange_Pct"] = mf_1m_px_chg_pct if not np.isnan(mf_1m_px_chg_pct) else prev_mf_1m_px_chg_pct

        prev_mf_5m_px_chg_pct = self.prev_tick_values["MF_5m_PriceChange_Pct"]
        features[3] = mf_5m_px_chg_pct - prev_mf_5m_px_chg_pct if not (np.isnan(mf_5m_px_chg_pct) or np.isnan(prev_mf_5m_px_chg_pct)) else np.nan
        self.prev_tick_values["MF_5m_PriceChange_Pct"] = mf_5m_px_chg_pct if not np.isnan(mf_5m_px_chg_pct) else prev_mf_5m_px_chg_pct

        # Position in Candle Range (Indices 4-7)
        def get_pos_in_range(price: Optional[float], bar_data: Optional[Dict]) -> float:
            if price is None or np.isnan(price) or bar_data is None: return np.nan
            bar_h = self._safe_get_from_bar(bar_data, 'high', np.nan)
            bar_l = self._safe_get_from_bar(bar_data, 'low', np.nan)
            if np.isnan(bar_h) or np.isnan(bar_l): return np.nan
            candle_range = bar_h - bar_l
            return self._safe_divide(price - bar_l, candle_range)  # Range can be 0, _safe_divide handles

        bar_1m_forming = self.market_data.get('current_1m_bar_forming')  # Can be None
        features[4] = get_pos_in_range(cur_px_mf, bar_1m_forming)
        bar_5m_forming = self.market_data.get('current_5m_bar_forming')  # Can be None
        features[5] = get_pos_in_range(cur_px_mf, bar_5m_forming)

        if not df_1m.empty: features[6] = get_pos_in_range(cur_px_mf, df_1m.iloc[-1].to_dict())
        if not df_5m.empty: features[7] = get_pos_in_range(cur_px_mf, df_5m.iloc[-1].to_dict())

        # EMAs (Indices 8-11)
        for i, (df, period_short, period_long, feat_idx_s, feat_idx_l) in enumerate([
            (df_1m, self.MF_EMA_SHORT_PERIOD, self.MF_EMA_LONG_PERIOD, 8, 9),
            (df_5m, self.MF_EMA_SHORT_PERIOD, self.MF_EMA_LONG_PERIOD, 10, 11)
        ]):
            if not df.empty and 'close' in df.columns:
                closes = df['close'].astype(float).dropna()  # Ensure float and remove NaNs for TA Lib
                if len(closes) >= period_short:
                    try:
                        ema_short = ta.trend.EMAIndicator(closes, window=period_short).ema_indicator()
                        if not ema_short.empty: features[feat_idx_s] = self._calculate_pct_dist(cur_px_mf, ema_short.iloc[-1])
                    except Exception as e:
                        self.logger.debug(f"Error calculating EMA short for df {i}: {e}")
                if len(closes) >= period_long:
                    try:
                        ema_long = ta.trend.EMAIndicator(closes, window=period_long).ema_indicator()
                        if not ema_long.empty: features[feat_idx_l] = self._calculate_pct_dist(cur_px_mf, ema_long.iloc[-1])
                    except Exception as e:
                        self.logger.debug(f"Error calculating EMA long for df {i}: {e}")

        # Dist to Rolling HF High/Low (Indices 12-13)
        hf_bars_list = [item.get('bar') for item in self.market_data.get('rolling_1s_data_window', []) if item.get('bar') is not None]

        if len(hf_bars_list) >= self.MF_ROLLING_HF_DATA_WINDOW_SECONDS:  # Check against actual required length
            # Take the most recent ones
            recent_hf_bars_list = hf_bars_list[-self.MF_ROLLING_HF_DATA_WINDOW_SECONDS:]
            if recent_hf_bars_list:
                hf_highs = [self._safe_get_from_bar(b, 'high', np.nan) for b in recent_hf_bars_list]
                hf_lows = [self._safe_get_from_bar(b, 'low', np.nan) for b in recent_hf_bars_list]

                valid_hf_highs = [h for h in hf_highs if not np.isnan(h)]
                valid_hf_lows = [l for l in hf_lows if not np.isnan(l)]

                if valid_hf_highs: features[12] = self._calculate_pct_dist(cur_px_mf, max(valid_hf_highs))
                if valid_hf_lows: features[13] = self._calculate_pct_dist(cur_px_mf, min(valid_hf_lows))

        # MACD (1m) (Indices 14-16)
        if not df_1m.empty and 'close' in df_1m.columns:
            closes_1m = df_1m['close'].astype(float).dropna()
            if len(closes_1m) >= self.MF_MACD_SLOW_PERIOD:
                try:
                    macd_indicator = ta.trend.MACD(closes_1m, window_slow=self.MF_MACD_SLOW_PERIOD, window_fast=self.MF_MACD_FAST_PERIOD,
                                                   window_sign=self.MF_MACD_SIGNAL_PERIOD)
                    if not macd_indicator.macd().empty: features[14] = macd_indicator.macd().iloc[-1]
                    if not macd_indicator.macd_signal().empty: features[15] = macd_indicator.macd_signal().iloc[-1]
                    if not macd_indicator.macd_diff().empty: features[16] = macd_indicator.macd_diff().iloc[-1]
                except Exception as e:
                    self.logger.debug(f"Error calculating MACD(1m): {e}")

        # ATR (Indices 17-18)
        def get_atr_pct(df: pd.DataFrame, period: int, current_price: Optional[float]) -> float:
            if current_price is None or np.isnan(current_price): return np.nan
            if not df.empty and all(c in df.columns for c in ['high', 'low', 'close']):
                hlc_df = df[['high', 'low', 'close']].astype(float).dropna()  # Drop rows with any NaN in H,L,C for ATR
                if len(hlc_df) >= period:
                    try:
                        atr_values = ta.volatility.AverageTrueRange(high=hlc_df['high'], low=hlc_df['low'], close=hlc_df['close'],
                                                                    window=period).average_true_range()
                        if not atr_values.empty:
                            atr_val = atr_values.iloc[-1]
                            return self._safe_divide(atr_val * 100.0, current_price)  # ATR as % of current price
                    except Exception as e:
                        self.logger.debug(f"Error calculating ATR for period {period}: {e}")
            return np.nan

        features[17] = get_atr_pct(df_1m, self.MF_ATR_PERIOD, cur_px_mf)
        features[18] = get_atr_pct(df_5m, self.MF_ATR_PERIOD, cur_px_mf)

        # Candle Shape (Indices 19-24)
        def get_candle_shape_features(bar_data: Optional[Dict]) -> Tuple[float, float, float]:
            if bar_data is None: return np.nan, np.nan, np.nan
            bar_o = self._safe_get_from_bar(bar_data, 'open', np.nan)
            bar_h = self._safe_get_from_bar(bar_data, 'high', np.nan)
            bar_l = self._safe_get_from_bar(bar_data, 'low', np.nan)
            bar_c = self._safe_get_from_bar(bar_data, 'close', np.nan)
            if any(pd.isna([bar_o, bar_h, bar_l, bar_c])): return np.nan, np.nan, np.nan

            bar_range = bar_h - bar_l
            body_size_rel = self._safe_divide(abs(bar_c - bar_o), bar_range, default_val=0.0)  # Body is 0 if range is 0
            upper_wick_rel = self._safe_divide(bar_h - max(bar_o, bar_c), bar_range, default_val=0.0)
            lower_wick_rel = self._safe_divide(min(bar_o, bar_c) - bar_l, bar_range, default_val=0.0)
            return body_size_rel, upper_wick_rel, lower_wick_rel

        if not df_1m.empty: features[19], features[20], features[21] = get_candle_shape_features(df_1m.iloc[-1].to_dict())
        if not df_5m.empty: features[22], features[23], features[24] = get_candle_shape_features(df_5m.iloc[-1].to_dict())

        # MF_Xm_BarVol_Ratio_To_TodaySoFarVol (Indices 25-26)
        total_daily_vol_so_far = self.daily_cumulative_volume  # Updated in _update_market_data
        if not df_1m.empty:
            vol_1m_bar = self._safe_get_from_bar(df_1m.iloc[-1], 'volume', 0.0)
            features[25] = self._safe_divide(vol_1m_bar, total_daily_vol_so_far, default_val=0.0)  # If total daily vol is 0, ratio is 0
        if not df_5m.empty:
            vol_5m_bar = self._safe_get_from_bar(df_5m.iloc[-1], 'volume', 0.0)
            features[26] = self._safe_divide(vol_5m_bar, total_daily_vol_so_far, default_val=0.0)

        # MF_Xm_Volume_Rel_To_Avg_Recent_Bars (Indices 27-28)
        for i, (df, window, feat_idx) in enumerate([
            (df_1m, self.MF_VOLUME_AVG_RECENT_BARS_WINDOW, 27),
            (df_5m, self.MF_VOLUME_AVG_RECENT_BARS_WINDOW, 28)
        ]):
            if not df.empty and 'volume' in df.columns and len(df) > window and window > 0:  # Need more than 'window' to have 'window' previous bars
                volumes = df['volume'].astype(float)  # NaNs will be kept by astype
                current_bar_volume = volumes.iloc[-1]
                # Average of N bars *excluding* the current completed one
                # Ensure enough previous bars for mean after potential NaNs are dropped by .mean()
                avg_vol_recent_bars = volumes.iloc[-(window + 1): -1].mean()  # mean() handles NaNs by default
                features[feat_idx] = self._safe_divide(current_bar_volume, avg_vol_recent_bars)

        # Swing Levels (Indices 29-32)
        def get_dist_to_swing(df: pd.DataFrame, lookback_bars: int, strength_bars: int, current_price: Optional[float], is_high: bool) -> float:
            if current_price is None or np.isnan(current_price) or df.empty: return np.nan

            price_col = 'high' if is_high else 'low'
            if price_col not in df.columns or len(df) < lookback_bars: return np.nan  # Need enough bars for the lookback period

            # Prices from the lookback window for swing detection
            series_for_swings = df[price_col].astype(float).iloc[-lookback_bars:]  # NaNs are fine here, _find_swing_points handles

            # _find_swing_points needs at least 2*strength_bars + 1 VALID points
            # It internally drops NaNs from the series passed to it.
            swing_high_points, swing_low_points = self._find_swing_points(series_for_swings, strength_bars)

            target_points = swing_high_points if is_high else swing_low_points
            if not target_points: return np.nan  # No swings found

            # Find the swing point closest to current_price or most relevant (e.g. highest high, lowest low)
            # For simplicity, using max of swing highs and min of swing lows found in the window.
            relevant_swing_level = max(target_points) if is_high else min(target_points)
            return self._calculate_pct_dist(current_price, relevant_swing_level)

        features[29] = get_dist_to_swing(df_1m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, True)  # 1m Swing High
        features[30] = get_dist_to_swing(df_1m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, False)  # 1m Swing Low
        features[31] = get_dist_to_swing(df_5m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, True)  # 5m Swing High
        features[32] = get_dist_to_swing(df_5m, self.MF_SWING_LOOKBACK_BARS, self.MF_SWING_STRENGTH_BARS, cur_px_mf, False)  # 5m Swing Low

        return features.astype(np.float32)

    def _calculate_lf_features(self) -> np.ndarray:
        features = np.full(self.config.lf_feat_dim, np.nan)
        cur_px_lf = self.market_data.get('current_price')  # This is top-level current_price
        if cur_px_lf is None:
            cur_px_lf = np.nan
        else:
            cur_px_lf = float(cur_px_lf)

        df_hist_1d = self.historical_1d_bars_df  # Already DataFrame, index set up

        # Position In Daily Range (Feature 0)
        d_low = self.market_data.get('intraday_low')  # Can be None
        d_high = self.market_data.get('intraday_high')  # Can be None
        if d_low is not None and d_high is not None and not np.isnan(cur_px_lf):
            f_d_low, f_d_high = float(d_low), float(d_high)
            daily_range = f_d_high - f_d_low
            features[0] = self._safe_divide(cur_px_lf - f_d_low, daily_range)

        # Position In PrevDay Range (Feature 1)
        # _get_daily_bar_value uses iloc, assumes sorted df_hist_1d
        prev_d_low_val = self._get_daily_bar_value(-1, 'low', self.LF_PREV_DAY_HL_SOURCE)
        prev_d_high_val = self._get_daily_bar_value(-1, 'high', self.LF_PREV_DAY_HL_SOURCE)
        if not np.isnan(prev_d_low_val) and not np.isnan(prev_d_high_val) and not np.isnan(cur_px_lf):
            prev_daily_range = prev_d_high_val - prev_d_low_val
            features[1] = self._safe_divide(cur_px_lf - prev_d_low_val, prev_daily_range)

        # Pct Change From Prev Close (Feature 2)
        prev_d_close_val = self._get_daily_bar_value(-1, 'close', self.LF_PREV_DAY_CLOSE_SOURCE)
        features[2] = self._pct_change(cur_px_lf, prev_d_close_val)

        # RVol (Feature 3)
        vol_today_cumulative = self.daily_cumulative_volume  # From _update_market_data
        avg_daily_vol_10d = np.nan
        if not df_hist_1d.empty and 'volume' in df_hist_1d.columns:
            volumes_1d = df_hist_1d['volume'].astype(float)
            if len(volumes_1d) >= self.LF_AVG_DAILY_VOLUME_PERIOD_DAYS:
                # Ensure we take the N most recent days for the average, excluding "today" if it's partially there
                avg_daily_vol_10d = volumes_1d.iloc[-self.LF_AVG_DAILY_VOLUME_PERIOD_DAYS:].mean()

        frac_day_passed_val = np.nan
        current_ts_utc_lf = self.market_data.get('timestamp_utc')
        if current_ts_utc_lf:
            current_time_et_lf = current_ts_utc_lf.astimezone(ET_TZ)
            session_start_dt_et_lf = datetime.combine(current_time_et_lf.date(), time(self.SESSION_START_HOUR_ET, 0), tzinfo=ET_TZ)

            if current_time_et_lf >= session_start_dt_et_lf:
                secs_in_session_lf = (current_time_et_lf - session_start_dt_et_lf).total_seconds()
                if self.TOTAL_SECONDS_IN_FULL_TRADING_DAY > 0:
                    frac_day_passed_val = min(max(secs_in_session_lf / self.TOTAL_SECONDS_IN_FULL_TRADING_DAY, 0.0), 1.0)
            else:  # Before session start
                frac_day_passed_val = 0.0

        if not any(np.isnan([vol_today_cumulative, avg_daily_vol_10d, frac_day_passed_val])) and \
                frac_day_passed_val > self.EPSILON and avg_daily_vol_10d > self.EPSILON:
            expected_vol_at_time = avg_daily_vol_10d * frac_day_passed_val
            # RVol as (current_vol / expected_vol_at_time) - 1.0. Or % deviation.
            features[3] = (self._safe_divide(vol_today_cumulative, expected_vol_at_time, default_val=1.0) - 1.0) * 100.0
        elif vol_today_cumulative == 0.0 and avg_daily_vol_10d == 0.0:  # Handle 0/0 for Rvol
            features[3] = 0.0  # If both current and avg are zero, no relative volume change
        elif avg_daily_vol_10d > self.EPSILON and frac_day_passed_val <= self.EPSILON:  # start of day, no time passed
            features[3] = (self._safe_divide(vol_today_cumulative, 1.0, default_val=0.0) - avg_daily_vol_10d)  # Treat as deviation from expected (which is 0)
        else:  # Default to 0% RVol if cannot calculate (e.g., avg daily vol is zero, or very start of day)
            features[3] = 0.0

        # Dist To Session VWAP (Feature 4)
        session_vwap_val = np.nan
        if self.LF_VWAP_SESSION_SCOPE == "FullDay_4AM_Start_NoReset":
            if self.daily_cumulative_volume > self.EPSILON:
                session_vwap_val = self.daily_cumulative_price_volume_product / self.daily_cumulative_volume
        features[4] = self._calculate_pct_dist(cur_px_lf, session_vwap_val)

        # Daily EMAs (Features 5-7)
        # Determine which close column to use for daily EMAs
        daily_close_col_name = 'close'  # Default
        if self.LF_DAILY_EMA_PRICE_SOURCE == "PostMarketClose_8PM" and 'close_8pm' in df_hist_1d.columns:
            daily_close_col_name = 'close_8pm'
        elif 'close' not in df_hist_1d.columns and 'close_8pm' in df_hist_1d.columns:  # Fallback if 'close' not there
            daily_close_col_name = 'close_8pm'

        if not df_hist_1d.empty and daily_close_col_name in df_hist_1d.columns:
            daily_closes = df_hist_1d[daily_close_col_name].astype(float).dropna()
            for period, feat_idx in [
                (self.LF_DAILY_EMA_SHORT_PERIOD, 5),
                (self.LF_DAILY_EMA_MEDIUM_PERIOD, 6),
                (self.LF_DAILY_EMA_LONG_PERIOD, 7)
            ]:
                if len(daily_closes) >= period:
                    try:
                        ema_daily = ta.trend.EMAIndicator(daily_closes, window=period).ema_indicator()
                        if not ema_daily.empty: features[feat_idx] = self._calculate_pct_dist(cur_px_lf, ema_daily.iloc[-1])
                    except Exception as e:
                        self.logger.debug(f"Error calculating Daily EMA {period}: {e}")

        # Dist To Closest LT Support/Resistance (Features 8-9)
        closest_resistance_val, closest_support_val = np.nan, np.nan
        if not df_hist_1d.empty and not np.isnan(cur_px_lf):
            num_days_for_sr = self.LF_LT_SR_LOOKBACK_YEARS * 252  # Approx trading days
            relevant_sr_bars = df_hist_1d.iloc[-num_days_for_sr:] if len(df_hist_1d) > num_days_for_sr else df_hist_1d

            if not relevant_sr_bars.empty:
                # Find highest high above current price (resistance)
                highs_above_current = relevant_sr_bars['high'][relevant_sr_bars['high'] > cur_px_lf].astype(float)
                if not highs_above_current.empty: closest_resistance_val = highs_above_current.min()  # Closest high above

                # Find lowest low below current price (support)
                lows_below_current = relevant_sr_bars['low'][relevant_sr_bars['low'] < cur_px_lf].astype(float)
                if not lows_below_current.empty: closest_support_val = lows_below_current.max()  # Closest low below

        features[8] = self._calculate_pct_dist(cur_px_lf, closest_support_val)
        features[9] = self._calculate_pct_dist(cur_px_lf, closest_resistance_val)

        return features.astype(np.float32)

    def reset(self):
        """Resets history and stateful elements of the feature extractor."""
        self.logger.info("Resetting FeatureExtractor state.")
        self._initialize_buffers_with_nan()  # Clears hf/mf/lf_features_history

        # Reset deques for rolling calculations
        for dq in [self.recent_1s_volumes, self.recent_1s_trade_counts,
                   self.recent_1s_total_trade_shares, self.recent_1s_bid_values_usd,
                   self.recent_1s_ask_values_usd]:
            dq.clear()
            # Optionally pre-fill with NaNs if strict length is needed from step 0,
            # but typically they fill up naturally. For now, just clearing.

        # Reset previous tick values
        for key in self.prev_tick_values:
            self.prev_tick_values[key] = np.nan

        # Reset daily accumulators, but keep last_processed_day to allow for correct first-day processing
        # self.daily_cumulative_volume = 0.0 # These are reset by _update_market_data on new day
        # self.daily_cumulative_price_volume_product = 0.0
        # self.last_processed_day_for_daily_reset = None # This will force daily reset on next _update_market_data

        # Re-load initial state for prev_day_close, similar to __init__
        self._initial_prev_day_close_load()

        # Market data itself will be fetched fresh in the next call to extract_features -> _update_market_data
        self.market_data = {}
        self.completed_1m_bars_df = self._ensure_datetime_index(pd.DataFrame(), 'timestamp_start')
        self.completed_5m_bars_df = self._ensure_datetime_index(pd.DataFrame(), 'timestamp_start')
        # historical_1d_bars_df is loaded by _initial_prev_day_close_load or _update_market_data