# feature/feature_extractor.py - CLEAN: Minimal logging, focus on feature extraction

import logging
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime

from config.config import ModelConfig
from simulators.market_simulator import MarketSimulator


class FeatureExtractor:
    """Clean feature extractor with minimal logging"""

    def __init__(self, symbol: str, market_simulator: MarketSimulator,
                 config: ModelConfig, logger: Optional[logging.Logger] = None):
        self.symbol = symbol
        self.market_simulator = market_simulator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Feature dimensions
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.static_feat_dim = config.static_feat_dim

        # Cache for efficiency
        self._last_features = None
        self._last_timestamp = None

    def reset(self):
        """Reset feature extractor state"""
        self._last_features = None
        self._last_timestamp = None

    def extract_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract features for current market state"""
        try:
            current_market_state = self.market_simulator.get_current_market_state()
            if not current_market_state:
                return None

            current_timestamp = current_market_state.get('timestamp_utc')

            # Use cache if same timestamp
            if (self._last_features is not None and
                    current_timestamp == self._last_timestamp):
                return self._last_features

            # Extract different frequency features
            hf_features = self._extract_hf_features(current_market_state)
            mf_features = self._extract_mf_features(current_market_state)
            lf_features = self._extract_lf_features(current_market_state)
            static_features = self._extract_static_features(current_market_state)

            if any(feat is None for feat in [hf_features, mf_features, lf_features, static_features]):
                return None

            features = {
                'hf': hf_features,
                'mf': mf_features,
                'lf': lf_features,
                'static': static_features
            }

            # Cache results
            self._last_features = features
            self._last_timestamp = current_timestamp

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _extract_hf_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract high-frequency features from 1-second data"""
        try:
            hf_window = market_state.get('hf_data_window', [])
            if len(hf_window) < self.hf_seq_len:
                # Pad with the last available data or zeros
                if hf_window:
                    padding = [hf_window[-1]] * (self.hf_seq_len - len(hf_window))
                    hf_window = padding + hf_window
                else:
                    return np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)

            # Take last N entries
            hf_window = hf_window[-self.hf_seq_len:]

            features_list = []
            for window_entry in hf_window:
                features = self._process_hf_window_entry(window_entry)
                features_list.append(features)

            return np.array(features_list, dtype=np.float32)

        except Exception as e:
            self.logger.debug(f"HF feature extraction error: {e}")
            return np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)

    def _process_hf_window_entry(self, window_entry: Dict[str, Any]) -> np.ndarray:
        """Process a single HF window entry into features"""
        features = np.zeros(self.hf_feat_dim, dtype=np.float32)

        try:
            # Get 1s bar data
            bar_1s = window_entry.get('1s_bar')
            if bar_1s:
                features[0] = float(bar_1s.get('open', 0.0))
                features[1] = float(bar_1s.get('high', 0.0))
                features[2] = float(bar_1s.get('low', 0.0))
                features[3] = float(bar_1s.get('close', 0.0))
                features[4] = float(bar_1s.get('volume', 0.0))

                # VWAP if available
                if 'vwap' in bar_1s and bar_1s['vwap'] is not None:
                    features[5] = float(bar_1s['vwap'])
                else:
                    # Use close as fallback
                    features[5] = features[3]

            # Trade count and quote updates
            trades = window_entry.get('trades', [])
            quotes = window_entry.get('quotes', [])
            features[6] = float(len(trades))
            features[7] = float(len(quotes))

            # Trade size statistics
            if trades:
                trade_sizes = [float(t.get('size', 0)) for t in trades if 'size' in t]
                if trade_sizes:
                    features[8] = np.mean(trade_sizes)  # avg trade size
                    features[9] = np.max(trade_sizes)  # max trade size

            # Quote spread
            if quotes:
                last_quote = quotes[-1]
                bid = float(last_quote.get('bid_price', 0))
                ask = float(last_quote.get('ask_price', 0))
                if bid > 0 and ask > bid:
                    features[10] = ask - bid  # spread
                    features[11] = (ask - bid) / ((ask + bid) / 2) * 10000  # spread bps

            # Fill remaining features with derived values or zeros
            if len(features) > 12:
                # Price momentum (close vs open)
                if features[0] > 0 and features[3] > 0:
                    features[12] = (features[3] - features[0]) / features[0]

                # Volume momentum (placeholder)
                if len(features) > 13:
                    features[13] = features[4] / max(1.0, np.mean([features[4], 1.0]))

        except Exception as e:
            self.logger.debug(f"HF entry processing error: {e}")

        return features

    def _extract_mf_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract medium-frequency features from 1-minute bars"""
        try:
            bars_1m = market_state.get('1m_bars_window', [])
            if len(bars_1m) < self.mf_seq_len:
                # Pad if needed
                if bars_1m:
                    padding = [bars_1m[-1]] * (self.mf_seq_len - len(bars_1m))
                    bars_1m = padding + bars_1m
                else:
                    return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)

            # Take last N bars
            bars_1m = bars_1m[-self.mf_seq_len:]

            features_list = []
            for bar in bars_1m:
                features = self._process_mf_bar(bar)
                features_list.append(features)

            return np.array(features_list, dtype=np.float32)

        except Exception as e:
            self.logger.debug(f"MF feature extraction error: {e}")
            return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)

    def _process_mf_bar(self, bar: Dict[str, Any]) -> np.ndarray:
        """Process a 1-minute bar into features"""
        features = np.zeros(self.mf_feat_dim, dtype=np.float32)

        try:
            features[0] = float(bar.get('open', 0.0))
            features[1] = float(bar.get('high', 0.0))
            features[2] = float(bar.get('low', 0.0))
            features[3] = float(bar.get('close', 0.0))
            features[4] = float(bar.get('volume', 0.0))

            # Calculate additional features
            o, h, l, c, v = features[0], features[1], features[2], features[3], features[4]

            if o > 0:
                features[5] = (c - o) / o  # return
                features[6] = (h - l) / o  # range/open ratio

            if h > l and h > 0:
                features[7] = (c - l) / (h - l)  # close position in range

            if v > 0:
                features[8] = (c * v) if c > 0 else 0  # volume * price

            # Additional technical indicators (simple versions)
            if len(features) > 9:
                features[9] = h - l  # absolute range
                if len(features) > 10:
                    features[10] = (h + l + c) / 3  # typical price

        except Exception as e:
            self.logger.debug(f"MF bar processing error: {e}")

        return features

    def _extract_lf_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract low-frequency features from 5-minute bars"""
        try:
            bars_5m = market_state.get('5m_bars_window', [])
            if len(bars_5m) < self.lf_seq_len:
                # Pad if needed
                if bars_5m:
                    padding = [bars_5m[-1]] * (self.lf_seq_len - len(bars_5m))
                    bars_5m = padding + bars_5m
                else:
                    return np.zeros((self.lf_seq_len, self.lf_feat_dim), dtype=np.float32)

            # Take last N bars
            bars_5m = bars_5m[-self.lf_seq_len:]

            features_list = []
            for bar in bars_5m:
                features = self._process_lf_bar(bar)
                features_list.append(features)

            return np.array(features_list, dtype=np.float32)

        except Exception as e:
            self.logger.debug(f"LF feature extraction error: {e}")
            return np.zeros((self.lf_seq_len, self.lf_feat_dim), dtype=np.float32)

    def _process_lf_bar(self, bar: Dict[str, Any]) -> np.ndarray:
        """Process a 5-minute bar into features"""
        features = np.zeros(self.lf_feat_dim, dtype=np.float32)

        try:
            features[0] = float(bar.get('open', 0.0))
            features[1] = float(bar.get('high', 0.0))
            features[2] = float(bar.get('low', 0.0))
            features[3] = float(bar.get('close', 0.0))
            features[4] = float(bar.get('volume', 0.0))

            # Calculate additional features similar to MF but for longer timeframe
            o, h, l, c, v = features[0], features[1], features[2], features[3], features[4]

            if o > 0:
                features[5] = (c - o) / o  # return
                features[6] = (h - l) / o  # range/open ratio

            if h > l and h > 0:
                features[7] = (c - l) / (h - l)  # close position in range

            if v > 0:
                features[8] = (c * v) if c > 0 else 0  # volume * price

            # Additional features for 5m timeframe
            if len(features) > 9:
                features[9] = h - l  # absolute range
                if len(features) > 10:
                    features[10] = (h + l + c) / 3  # typical price
                    if len(features) > 11:
                        # Volume-weighted metrics
                        features[11] = v / max(1.0, np.mean([v, 1000.0]))  # normalized volume

        except Exception as e:
            self.logger.debug(f"LF bar processing error: {e}")

        return features

    def _extract_static_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract static/contextual features"""
        try:
            features = np.zeros((1, self.static_feat_dim), dtype=np.float32)

            # Current market state info
            current_price = market_state.get('current_price', 0.0)
            prev_day_close = market_state.get('previous_day_close', 0.0)

            if prev_day_close and prev_day_close > 0 and current_price > 0:
                features[0, 0] = (current_price - prev_day_close) / prev_day_close  # day change

            # Market session indicator
            session = market_state.get('market_session', 'UNKNOWN')
            if session == 'PREMARKET':
                features[0, 1] = 0.25
            elif session == 'REGULAR':
                features[0, 1] = 1.0
            elif session == 'POSTMARKET':
                features[0, 1] = 0.75
            else:
                features[0, 1] = 0.0

            # Intraday high/low relative position
            intraday_high = market_state.get('intraday_high')
            intraday_low = market_state.get('intraday_low')

            if (intraday_high and intraday_low and current_price > 0 and
                    intraday_high > intraday_low):
                range_position = (current_price - intraday_low) / (intraday_high - intraday_low)
                features[0, 2] = range_position

            # Price levels relative to previous day
            prev_day_data = market_state.get('previous_day_data', {})
            if prev_day_data and current_price > 0:
                prev_high = prev_day_data.get('high', 0)
                prev_low = prev_day_data.get('low', 0)

                if prev_high > 0:
                    features[0, 3] = current_price / prev_high
                if prev_low > 0 and len(features[0]) > 4:
                    features[0, 4] = current_price / prev_low

            return features

        except Exception as e:
            self.logger.debug(f"Static feature extraction error: {e}")
            return np.zeros((1, self.static_feat_dim), dtype=np.float32)