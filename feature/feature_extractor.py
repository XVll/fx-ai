# feature_extractor_v2.py
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging


class FeatureExtractor:
    """
    Extracts features with efficient rolling calculations for HF, MF, and LF bands.
    It processes a rolling window of 1-second data events provided by the market simulator.
    """

    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # HF (High Frequency - e.g., 1-second granularity)
        self.hf_steps = self.config['hf_steps']
        self.hf_feature_size = self.config['hf_feature_size']  # Size of feature vector per 1s step

        # MF (Medium Frequency - e.g., 1-minute granularity)
        self.mf_periods = self.config['mf_periods']
        self.mf_period_seconds = self.config['mf_period_seconds']
        self.mf_feature_size = self.config['mf_feature_size']  # Size of feature vector per MF period
        self.completed_mf_period_features: deque = deque(maxlen=self.mf_periods - 1)  # -1 for current
        self.last_mf_boundary_ts: Optional[datetime] = None
        self.mf_ema_short_span = self.config.get('mf_ema_short_span', 5)  # e.g., 5-period MF EMA
        self.mf_ema_long_span = self.config.get('mf_ema_long_span', 12)  # e.g., 12-period MF EMA
        self.rolling_mf_closes_for_ema: deque = deque(
            maxlen=max(self.mf_ema_short_span, self.mf_ema_long_span) + 5)  # Buffer for EMA calc

        # LF (Low Frequency - e.g., 5-minute granularity)
        self.lf_periods = self.config['lf_periods']
        self.lf_period_seconds = self.config['lf_period_seconds']
        self.lf_feature_size = self.config['lf_feature_size']  # Size of feature vector per LF period
        self.completed_lf_period_features: deque = deque(maxlen=self.lf_periods - 1)  # -1 for current
        self.last_lf_boundary_ts: Optional[datetime] = None
        self.lf_ema_short_span = self.config.get('lf_ema_short_span', 5)  # e.g., 5-period LF EMA
        self.lf_ema_long_span = self.config.get('lf_ema_long_span', 12)  # e.g., 12-period LF EMA
        self.rolling_lf_closes_for_ema: deque = deque(maxlen=max(self.lf_ema_long_span, self.lf_ema_long_span) + 5)

        # Static features
        self.static_feature_size = self.config['static_feature_size']

        self.is_initialized = False
        self.logger.info(f"FeatureExtractorV2 initialized with config: {self.config}")

    def _get_default_config(self) -> Dict:
        return {
            'hf_steps': 60, 'hf_feature_size': 8,  # Open,High,Low,Close,Volume, TradeCount, AvgTradeSize, QuoteCount
            'mf_periods': 30, 'mf_period_seconds': 60, 'mf_feature_size': 10,
            # OHLCV, BarSize, EMA5, EMA12, DistToEMA5, VolumeDelta
            'lf_periods': 30, 'lf_period_seconds': 300, 'lf_feature_size': 10,
            'static_feature_size': 5,  # Placeholder
            'mf_ema_short_span': 5, 'mf_ema_long_span': 12,
            'lf_ema_short_span': 5, 'lf_ema_long_span': 12,
        }

    def reset_state(self):
        """Resets the internal state of the feature extractor (e.g., EMAs, completed periods)."""
        self.completed_mf_period_features.clear()
        self.last_mf_boundary_ts = None
        self.rolling_mf_closes_for_ema.clear()

        self.completed_lf_period_features.clear()
        self.last_lf_boundary_ts = None
        self.rolling_lf_closes_for_ema.clear()

        self.is_initialized = False
        self.logger.info("FeatureExtractorV2 state has been reset.")

    def _calculate_ema(self, data: pd.Series, span: int) -> pd.Series:
        if len(data) < span:  # Not enough data for full EMA
            return pd.Series([np.nan] * len(data), index=data.index)
        return data.ewm(span=span, adjust=False).mean()

    def _aggregate_1s_to_period_features(
            self,
            one_second_events_for_period: List[Dict[str, Any]],
            period_type: str = 'mf'  # 'mf' or 'lf'
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregates a list of 1-second data events into features for a single period.
        """
        if not one_second_events_for_period:
            return None

        bars = [ev['bar'] for ev in one_second_events_for_period if ev.get('bar') is not None]
        if not bars:
            return None  # No bar data to aggregate

        period_open = bars[0]['open']
        period_high = max(b['high'] for b in bars)
        period_low = min(b['low'] for b in bars)
        period_close = bars[-1]['close']
        period_volume = sum(b['volume'] for b in bars)

        # Derived features for the period
        bar_size = period_high - period_low
        # Volume delta could be vs previous period, requires more state or access to it here

        # Placeholder for other features like VWAP, order flow imbalance from trades/quotes
        trades_in_period = sum([len(ev.get('trades', [])) for ev in one_second_events_for_period])
        avg_trade_size = sum(t['size'] for ev in one_second_events_for_period for t in
                             ev.get('trades', [])) / trades_in_period if trades_in_period > 0 else 0

        agg_features = {
            'timestamp': one_second_events_for_period[-1]['timestamp'],  # End timestamp of the period
            'open': period_open, 'high': period_high, 'low': period_low,
            'close': period_close, 'volume': period_volume,
            'bar_size': bar_size,
            'trade_count': trades_in_period,
            'avg_trade_size': avg_trade_size,
            # EMA related features will be added after this dict is created and EMA is calculated
        }

        # Update rolling closes for EMA calculation for this period_type
        rolling_closes_deque = self.rolling_mf_closes_for_ema if period_type == 'mf' else self.rolling_lf_closes_for_ema
        ema_short_span = self.mf_ema_short_span if period_type == 'mf' else self.lf_ema_short_span
        ema_long_span = self.mf_ema_long_span if period_type == 'mf' else self.lf_ema_long_span

        # Note: EMA is calculated based on the history of *completed* period closes.
        # The `period_close` of the current aggregation is a new data point for this history.

        # For the EMA that characterizes THIS period, we use the historical closes UP TO the one BEFORE this.
        # Or, more commonly, the EMA is calculated and the period's close is compared to the EMA value *entering* this period.
        # Let's calculate EMA based on all closes available *including* the current one for simplicity of state.

        temp_closes_for_ema_calc = list(rolling_closes_deque)
        # We add the current period's close to calculate the EMA *as of the end of this period*
        # This EMA value could then be used by the *next* period as its "entry EMA"

        # If we want "distance to EMA" for *this* period, the EMA should be based on data *prior* to this period.
        # This is a common source of lookahead bias if not handled carefully.
        # Let's assume EMA features are based on EMAs calculated from prior completed periods.

        if len(rolling_closes_deque) >= 1:  # Need at least one previous close
            closes_series = pd.Series(list(rolling_closes_deque))

            ema_short_values = self._calculate_ema(closes_series, ema_short_span)
            ema_long_values = self._calculate_ema(closes_series, ema_long_span)

            if not ema_short_values.empty and not np.isnan(ema_short_values.iloc[-1]):
                agg_features[f'{period_type}_ema_short'] = ema_short_values.iloc[-1]
                agg_features[f'{period_type}_dist_to_ema_short'] = period_close - ema_short_values.iloc[-1]
            else:
                agg_features[f'{period_type}_ema_short'] = np.nan
                agg_features[f'{period_type}_dist_to_ema_short'] = np.nan

            if not ema_long_values.empty and not np.isnan(ema_long_values.iloc[-1]):
                agg_features[f'{period_type}_ema_long'] = ema_long_values.iloc[-1]
            else:
                agg_features[f'{period_type}_ema_long'] = np.nan
        else:  # Not enough data for any EMA
            agg_features[f'{period_type}_ema_short'] = np.nan
            agg_features[f'{period_type}_dist_to_ema_short'] = np.nan
            agg_features[f'{period_type}_ema_long'] = np.nan

        return agg_features

    def _vectorize_period_features(self, period_agg: Optional[Dict[str, Any]], feature_size: int,
                                   period_type: str = 'mf') -> np.ndarray:
        """Converts the aggregated feature dict for a period into a numpy vector."""
        if period_agg is None:
            return np.zeros(feature_size)  # Or np.full(feature_size, np.nan)

        # Define the order of features in the vector
        # This must match mf_feature_size and lf_feature_size
        # Example for MF/LF:
        feature_vector = np.array([
            period_agg.get('open', 0),
            period_agg.get('high', 0),
            period_agg.get('low', 0),
            period_agg.get('close', 0),
            period_agg.get('volume', 0),
            period_agg.get('bar_size', 0),
            period_agg.get(f'{period_type}_ema_short', 0),  # Use 0 or NaN if missing
            period_agg.get(f'{period_type}_ema_long', 0),
            period_agg.get(f'{period_type}_dist_to_ema_short', 0),
            period_agg.get('trade_count', 0)  # Example, ensure size matches
            # Add more features up to feature_size
        ])
        # Ensure feature_vector is of the correct size, padding if necessary
        if len(feature_vector) < feature_size:
            padding = np.zeros(feature_size - len(feature_vector))  # Or np.nan
            feature_vector = np.concatenate((feature_vector, padding))
        elif len(feature_vector) > feature_size:
            feature_vector = feature_vector[:feature_size]  # Truncate

        return np.nan_to_num(feature_vector)  # Replace NaNs with 0, or handle as per model req.

    def _vectorize_hf_features(self, hf_event: Dict[str, Any], feature_size: int) -> np.ndarray:
        if hf_event is None or 'bar' not in hf_event or hf_event['bar'] is None:
            return np.zeros(feature_size)

        bar = hf_event['bar']
        trades = hf_event.get('trades', [])
        quotes = hf_event.get('quotes', [])  # For more advanced features like BBO, spread

        feature_vector = np.array([
            bar.get('open', 0),
            bar.get('high', 0),
            bar.get('low', 0),
            bar.get('close', 0),
            bar.get('volume', 0),
            len(trades),
            sum(t.get('size', 0) for t in trades) / len(trades) if trades else 0,  # Avg trade size
            len(quotes)  # Number of quote updates
        ])
        if len(feature_vector) < feature_size:
            padding = np.zeros(feature_size - len(feature_vector))
            feature_vector = np.concatenate((feature_vector, padding))
        elif len(feature_vector) > feature_size:
            feature_vector = feature_vector[:feature_size]
        return np.nan_to_num(feature_vector)

    def _initialize_completed_periods(self, all_1s_events: List[Dict[str, Any]], current_ts: datetime):
        """Populate completed MF and LF period deques on first call or reset."""
        self.logger.debug(f"Initializing completed periods. Total 1s events: {len(all_1s_events)}")

        # Initialize MF
        current_mf_slot_start_ts = current_ts.replace(
            minute=(current_ts.minute // self.mf_period_seconds) * (self.mf_period_seconds // 60), second=0,
            microsecond=0)
        self.last_mf_boundary_ts = current_mf_slot_start_ts

        num_mf_to_fill = self.completed_mf_period_features.maxlen
        for i in range(num_mf_to_fill):
            end_of_period_to_fill = current_mf_slot_start_ts - timedelta(seconds=i * self.mf_period_seconds)
            start_of_period_to_fill = end_of_period_to_fill - timedelta(seconds=self.mf_period_seconds)

            period_events = [ev for ev in all_1s_events if
                             start_of_period_to_fill <= ev['timestamp'] < end_of_period_to_fill]
            if period_events:
                agg_feats = self._aggregate_1s_to_period_features(period_events, 'mf')
                if agg_feats:
                    self.completed_mf_period_features.appendleft(agg_feats)  # Oldest first
                    self.rolling_mf_closes_for_ema.appendleft(agg_feats['close'])  # Maintain correct order for EMA

        # Initialize LF
        current_lf_slot_start_ts = current_ts.replace(
            minute=(current_ts.minute // (self.lf_period_seconds // 60)) * (self.lf_period_seconds // 60), second=0,
            microsecond=0)
        self.last_lf_boundary_ts = current_lf_slot_start_ts

        num_lf_to_fill = self.completed_lf_period_features.maxlen
        for i in range(num_lf_to_fill):
            end_of_period_to_fill = current_lf_slot_start_ts - timedelta(seconds=i * self.lf_period_seconds)
            start_of_period_to_fill = end_of_period_to_fill - timedelta(seconds=self.lf_period_seconds)

            period_events = [ev for ev in all_1s_events if
                             start_of_period_to_fill <= ev['timestamp'] < end_of_period_to_fill]
            if period_events:
                agg_feats = self._aggregate_1s_to_period_features(period_events, 'lf')
                if agg_feats:
                    self.completed_lf_period_features.appendleft(agg_feats)
                    self.rolling_lf_closes_for_ema.appendleft(agg_feats['close'])

        self.is_initialized = True
        self.logger.info(
            f"Initialization complete. MF deque size: {len(self.completed_mf_period_features)}, LF deque size: {len(self.completed_lf_period_features)}")

    def _update_and_get_frequency_features(
            self,
            all_1s_events: List[Dict[str, Any]],
            current_ts: datetime,
            num_periods: int,
            period_seconds: int,
            feature_size: int,
            completed_period_features_deque: deque,
            rolling_closes_deque: deque,
            last_boundary_ts_attr_name: str,
            period_type_str: str  # 'mf' or 'lf'
    ) -> np.ndarray:

        output_features_matrix = np.zeros((num_periods, feature_size))

        # 1. Current, in-progress period
        # Ensure current_period_slot_start_ts calculation is correct
        current_period_slot_start_ts = current_ts.replace(
            minute=(current_ts.minute // (period_seconds // 60)) * (period_seconds // 60),
            second=0, microsecond=0
        )

        in_progress_events = [ev for ev in all_1s_events if ev['timestamp'] >= current_period_slot_start_ts]
        current_period_agg = self._aggregate_1s_to_period_features(in_progress_events, period_type_str)
        output_features_matrix[0, :] = self._vectorize_period_features(current_period_agg, feature_size,
                                                                       period_type_str)

        # 2. Update and retrieve completed periods
        last_boundary_ts = getattr(self, last_boundary_ts_attr_name)

        if current_period_slot_start_ts > last_boundary_ts:
            # New period(s) completed
            num_new_periods = int((current_period_slot_start_ts - last_boundary_ts).total_seconds() / period_seconds)

            for i in range(num_new_periods):
                # Process from oldest new period to newest new period
                end_of_newly_completed_period = last_boundary_ts + timedelta(seconds=(i + 1) * period_seconds)
                start_of_newly_completed_period = end_of_newly_completed_period - timedelta(seconds=period_seconds)

                newly_completed_events = [
                    ev for ev in all_1s_events
                    if start_of_newly_completed_period <= ev['timestamp'] < end_of_newly_completed_period
                ]
                if newly_completed_events:
                    newly_completed_agg = self._aggregate_1s_to_period_features(newly_completed_events, period_type_str)
                    if newly_completed_agg:
                        completed_period_features_deque.append(newly_completed_agg)
                        rolling_closes_deque.append(newly_completed_agg['close'])  # For EMA

            setattr(self, last_boundary_ts_attr_name, current_period_slot_start_ts)

        # Fill the rest of the matrix from the deque (most recent completed first)
        # Index 0 is current/in-progress, so completed go from index 1 onwards
        for i, completed_agg in enumerate(reversed(list(completed_period_features_deque))):
            if (i + 1) < num_periods:  # Max num_periods-1 from deque
                output_features_matrix[i + 1, :] = self._vectorize_period_features(completed_agg, feature_size,
                                                                                   period_type_str)
            else:
                break

        return output_features_matrix

    def extract_features(self, market_state: Dict[str, Any], portfolio_state: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """
        Extracts features based on the provided market state.
        market_state is expected from MarketSimulatorV2.
        portfolio_state can be used for static features like current position.
        """
        # Get timestamp and 1-second event data
        all_1s_events = market_state.get('rolling_1s_data_window', [])
        current_ts = market_state.get('timestamp')

        # Handle empty data
        if not all_1s_events or not current_ts:
            self.logger.warning("Insufficient data in market state for feature extraction")
            # Return placeholder features with the right shapes
            return {
                'timestamp': current_ts or datetime.now(),
                'hf_features': np.zeros((self.hf_steps, self.hf_feature_size)),
                'mf_features': np.zeros((self.mf_periods, self.mf_feature_size)),
                'lf_features': np.zeros((self.lf_periods, self.lf_feature_size)),
                'static_features': np.zeros(self.static_feature_size)
            }

        # Initialize state if needed
        if not self.is_initialized:
            self._initialize_completed_periods(all_1s_events, current_ts)

        # --- Extract HF features ---
        hf_feature_matrix = np.zeros((self.hf_steps, self.hf_feature_size))
        for i in range(min(self.hf_steps, len(all_1s_events))):
            # Safe indexing with bounds checking
            idx = len(all_1s_events) - i - 1
            if idx >= 0:
                hf_event = all_1s_events[idx]
                hf_feature_matrix[i] = self._vectorize_hf_features(hf_event, self.hf_feature_size)

        # --- Process MF features ---
        mf_feature_matrix = self._update_and_get_frequency_features(
            all_1s_events=all_1s_events,
            current_ts=current_ts,
            num_periods=self.mf_periods,
            period_seconds=self.mf_period_seconds,
            feature_size=self.mf_feature_size,
            completed_period_features_deque=self.completed_mf_period_features,
            rolling_closes_deque=self.rolling_mf_closes_for_ema,
            last_boundary_ts_attr_name='last_mf_boundary_ts',
            period_type_str='mf'
        )

        # --- Process LF features ---
        lf_feature_matrix = self._update_and_get_frequency_features(
            all_1s_events=all_1s_events,
            current_ts=current_ts,
            num_periods=self.lf_periods,
            period_seconds=self.lf_period_seconds,
            feature_size=self.lf_feature_size,
            completed_period_features_deque=self.completed_lf_period_features,
            rolling_closes_deque=self.rolling_lf_closes_for_ema,
            last_boundary_ts_attr_name='last_lf_boundary_ts',
            period_type_str='lf'
        )

        # --- Process static features ---
        static_feature_vector = np.zeros(self.static_feature_size)

        # Position data from portfolio state
        if portfolio_state:
            if 0 < self.static_feature_size:
                static_feature_vector[0] = portfolio_state.get('position', 0.0)
            if 1 < self.static_feature_size:
                static_feature_vector[1] = portfolio_state.get('total_pnl', 0.0)
            if 2 < self.static_feature_size:
                static_feature_vector[2] = portfolio_state.get('unrealized_pnl', 0.0)

        # Current price from market state (for convenience)
        if 3 < self.static_feature_size:
            static_feature_vector[3] = market_state.get('current_price', 0.0)

        # Distance to nearest whole/half dollar level if available
        if 4 < self.static_feature_size and market_state.get('current_price'):
            current_price = market_state.get('current_price')
            nearest_half_dollar = round(current_price * 2) / 2
            static_feature_vector[4] = abs(current_price - nearest_half_dollar)

        # Return all features with timestamp for easier testing
        return {
            'timestamp': current_ts,
            'hf_features': hf_feature_matrix,
            'mf_features': mf_feature_matrix,
            'lf_features': lf_feature_matrix,
            'static_features': static_feature_vector
        }

    def _calculate_support_resistance(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate support and resistance levels from historical data."""
        # Get current price
        current_price = market_state.get('current_price', 0)
        if current_price <= 0:
            return {}

        # Get historical 5m bars for S/R
        historical_5m = market_state.get('historical_5m_for_sr')
        if historical_5m is None or historical_5m.empty:
            return {}

        # Find recent pivots (simplified approach)
        highs = historical_5m['high'].values
        lows = historical_5m['low'].values

        # Find potential support levels (recent lows)
        supports = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                supports.append(lows[i])

        # Find potential resistance levels (recent highs)
        resistances = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                resistances.append(highs[i])

        # Filter levels below current price for support
        supports = [s for s in supports if s < current_price]

        # Filter levels above current price for resistance
        resistances = [r for r in resistances if r > current_price]

        # Get nearest levels
        nearest_support = max(supports) if supports else 0
        nearest_resistance = min(resistances) if resistances else 0

        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'distance_to_support': current_price - nearest_support if nearest_support > 0 else 0,
            'distance_to_resistance': nearest_resistance - current_price if nearest_resistance > 0 else 0
        }