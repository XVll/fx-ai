# feature/feature_extractor.py - CLEAN: Minimal logging, focused on feature extraction

import logging
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

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

        # Feature dimensions from config
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.static_feat_dim = config.static_feat_dim

        # Feature history buffers
        self.hf_buffer = deque(maxlen=self.hf_seq_len)
        self.mf_buffer = deque(maxlen=self.mf_seq_len)
        self.lf_buffer = deque(maxlen=self.lf_seq_len)

        # Previous values for rate calculations
        self.prev_price = None
        self.prev_volume = None

        # Only log initialization
        self.logger.info(f"FeatureExtractor initialized for {symbol}")

    def reset(self):
        """Reset buffers for new episode"""
        self.hf_buffer.clear()
        self.mf_buffer.clear()
        self.lf_buffer.clear()
        self.prev_price = None
        self.prev_volume = None

    def extract_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract all features with error handling"""
        try:
            market_state = self.market_simulator.get_current_market_state()
            if not market_state:
                return None

            # Extract each feature type
            hf_features = self._extract_hf_features(market_state)
            mf_features = self._extract_mf_features(market_state)
            lf_features = self._extract_lf_features(market_state)
            static_features = self._extract_static_features(market_state)

            if hf_features is None or mf_features is None or lf_features is None or static_features is None:
                return None

            return {
                'hf': hf_features,
                'mf': mf_features,
                'lf': lf_features,
                'static': static_features
            }

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _extract_hf_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract high-frequency features (1-second data)"""
        try:
            # Get current 1s bar and recent trades
            current_1s_bar = market_state.get('current_1s_bar')
            hf_data_window = market_state.get('hf_data_window', [])

            # Calculate features from current state
            current_price = market_state.get('current_price', 0.0)
            bid_price = market_state.get('best_bid_price', 0.0)
            ask_price = market_state.get('best_ask_price', 0.0)
            bid_size = market_state.get('best_bid_size', 0.0)
            ask_size = market_state.get('best_ask_size', 0.0)

            # Basic price and spread features
            spread = ask_price - bid_price if ask_price > bid_price else 0.0
            mid_price = (bid_price + ask_price) / 2 if (bid_price > 0 and ask_price > 0) else current_price

            # Price change from previous
            price_change = 0.0
            if self.prev_price and current_price > 0:
                price_change = (current_price - self.prev_price) / self.prev_price
            self.prev_price = current_price

            # Volume and trade features
            volume = 0.0
            num_trades = 0.0
            if current_1s_bar:
                volume = current_1s_bar.get('volume', 0.0)
                if 'trades' in current_1s_bar:
                    num_trades = float(len(current_1s_bar['trades']))

            # Volume change
            volume_change = 0.0
            if self.prev_volume and volume > 0:
                volume_change = (volume - self.prev_volume) / max(self.prev_volume, 1.0)
            self.prev_volume = volume

            # OHLC features from 1s bar
            bar_open = current_1s_bar.get('open', current_price) if current_1s_bar else current_price
            bar_high = current_1s_bar.get('high', current_price) if current_1s_bar else current_price
            bar_low = current_1s_bar.get('low', current_price) if current_1s_bar else current_price
            bar_close = current_1s_bar.get('close', current_price) if current_1s_bar else current_price

            # OHLC ratios
            bar_range = (bar_high - bar_low) / max(bar_close, 0.01) if bar_close > 0 else 0.0
            body_ratio = abs(bar_close - bar_open) / max(bar_high - bar_low, 0.0001)

            # Order book imbalance
            book_imbalance = 0.0
            if bid_size + ask_size > 0:
                book_imbalance = (bid_size - ask_size) / (bid_size + ask_size)

            # Create feature vector (20 features to match config)
            features = np.array([
                current_price / 100.0,  # Normalized price
                price_change,  # Price change rate
                spread / max(current_price, 0.01),  # Relative spread
                mid_price / 100.0,  # Normalized mid price
                volume / 1000.0,  # Normalized volume
                volume_change,  # Volume change rate
                num_trades / 10.0,  # Normalized trade count
                bid_size / 1000.0,  # Normalized bid size
                ask_size / 1000.0,  # Normalized ask size
                book_imbalance,  # Order book imbalance
                bar_range,  # Bar range ratio
                body_ratio,  # Candle body ratio
                (bar_close - bar_open) / max(bar_open, 0.01),  # Bar return
                (bar_high - bar_open) / max(bar_open, 0.01),  # Upper wick
                (bar_open - bar_low) / max(bar_open, 0.01),  # Lower wick
                1.0 if current_1s_bar and not current_1s_bar.get('is_synthetic', False) else 0.0,  # Data quality
                0.0,  # Reserved for momentum
                0.0,  # Reserved for volatility
                0.0,  # Reserved for trend
                0.0  # Reserved for market microstructure
            ], dtype=np.float32)

            # Handle NaN/inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            features = np.clip(features, -10.0, 10.0)

            # Add to buffer
            self.hf_buffer.append(features)

            # Create sequence of required length
            if len(self.hf_buffer) < self.hf_seq_len:
                # Pad with first available feature
                padding = [features] * (self.hf_seq_len - len(self.hf_buffer))
                sequence = np.array(padding + list(self.hf_buffer), dtype=np.float32)
            else:
                sequence = np.array(list(self.hf_buffer), dtype=np.float32)

            return sequence

        except Exception as e:
            self.logger.error(f"HF feature extraction failed: {e}")
            return np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)

    def _extract_mf_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract medium-frequency features (1-minute data)"""
        try:
            bars_1m = market_state.get('1m_bars_window', [])

            if not bars_1m or len(bars_1m) < self.mf_seq_len:
                return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)

            # Take the most recent bars
            recent_bars = bars_1m[-self.mf_seq_len:]
            features_list = []

            for i, bar in enumerate(recent_bars):
                if bar is None:
                    features_list.append(np.zeros(self.mf_feat_dim, dtype=np.float32))
                    continue

                # Extract OHLCV
                open_price = bar.get('open', 0.0)
                high_price = bar.get('high', 0.0)
                low_price = bar.get('low', 0.0)
                close_price = bar.get('close', 0.0)
                volume = bar.get('volume', 0.0)

                # Calculate technical indicators
                returns = 0.0
                volatility = 0.0

                if i > 0 and recent_bars[i - 1]:
                    prev_close = recent_bars[i - 1].get('close', 0.0)
                    if prev_close > 0:
                        returns = (close_price - prev_close) / prev_close

                # Range and body ratios
                price_range = (high_price - low_price) / max(close_price, 0.01) if close_price > 0 else 0.0
                body_size = abs(close_price - open_price) / max(close_price, 0.01) if close_price > 0 else 0.0

                # Volume analysis
                vol_norm = volume / 1000.0

                # VWAP if available
                vwap = bar.get('vwap', close_price)
                vwap_deviation = (close_price - vwap) / max(vwap, 0.01) if vwap > 0 else 0.0

                # Create feature vector (20 features)
                features = np.array([
                    close_price / 100.0,  # Normalized close
                    returns,  # Returns
                    volatility,  # Volatility (placeholder)
                    price_range,  # Price range
                    body_size,  # Body size
                    vol_norm,  # Normalized volume
                    vwap_deviation,  # VWAP deviation
                    (high_price - close_price) / max(close_price, 0.01),  # Upper shadow
                    (close_price - low_price) / max(close_price, 0.01),  # Lower shadow
                    1.0 if not bar.get('is_synthetic', False) else 0.0,  # Data quality
                    0.0, 0.0, 0.0, 0.0, 0.0,  # Reserved for additional indicators
                    0.0, 0.0, 0.0, 0.0, 0.0  # Reserved for more features
                ], dtype=np.float32)

                # Handle NaN/inf values
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                features = np.clip(features, -10.0, 10.0)

                features_list.append(features)

            return np.array(features_list, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"MF feature extraction failed: {e}")
            return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)

    def _extract_lf_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract low-frequency features (5-minute data)"""
        try:
            bars_5m = market_state.get('5m_bars_window', [])

            if not bars_5m or len(bars_5m) < self.lf_seq_len:
                return np.zeros((self.lf_seq_len, self.lf_feat_dim), dtype=np.float32)

            # Take the most recent bars
            recent_bars = bars_5m[-self.lf_seq_len:]
            features_list = []

            for i, bar in enumerate(recent_bars):
                if bar is None:
                    features_list.append(np.zeros(self.lf_feat_dim, dtype=np.float32))
                    continue

                # Extract OHLCV
                open_price = bar.get('open', 0.0)
                high_price = bar.get('high', 0.0)
                low_price = bar.get('low', 0.0)
                close_price = bar.get('close', 0.0)
                volume = bar.get('volume', 0.0)

                # Calculate longer-term indicators
                returns = 0.0
                momentum = 0.0

                # Multi-period analysis
                if i >= 1 and recent_bars[i - 1]:
                    prev_close = recent_bars[i - 1].get('close', 0.0)
                    if prev_close > 0:
                        returns = (close_price - prev_close) / prev_close

                if i >= 4 and recent_bars[i - 4]:  # 5-period momentum
                    old_close = recent_bars[i - 4].get('close', 0.0)
                    if old_close > 0:
                        momentum = (close_price - old_close) / old_close

                # Trend analysis
                trend_strength = 0.0
                if i >= 2:
                    prices = [recent_bars[j].get('close', 0.0) for j in range(max(0, i - 2), i + 1)
                              if recent_bars[j]]
                    if len(prices) >= 3:
                        trend_strength = (prices[-1] - prices[0]) / max(prices[0], 0.01)

                # Volume analysis
                vol_norm = volume / 5000.0  # 5-min volume normalization

                # Support/Resistance levels (simplified)
                support_level = min([recent_bars[j].get('low', float('inf'))
                                     for j in range(max(0, i - 9), i + 1)
                                     if recent_bars[j]])
                resistance_level = max([recent_bars[j].get('high', 0.0)
                                        for j in range(max(0, i - 9), i + 1)
                                        if recent_bars[j]])

                support_dist = (close_price - support_level) / max(close_price, 0.01) if support_level != float('inf') else 0.0
                resistance_dist = (resistance_level - close_price) / max(close_price, 0.01) if resistance_level > 0 else 0.0

                # Create feature vector (20 features)
                features = np.array([
                    close_price / 100.0,  # Normalized close
                    returns,  # Period returns
                    momentum,  # Multi-period momentum
                    trend_strength,  # Trend strength
                    vol_norm,  # Normalized volume
                    support_dist,  # Distance to support
                    resistance_dist,  # Distance to resistance
                    (high_price - low_price) / max(close_price, 0.01),  # Range
                    (close_price - open_price) / max(open_price, 0.01),  # Body
                    1.0 if not bar.get('is_synthetic', False) else 0.0,  # Data quality
                    0.0, 0.0, 0.0, 0.0, 0.0,  # Reserved for RSI, MACD, etc.
                    0.0, 0.0, 0.0, 0.0, 0.0  # Reserved for more indicators
                ], dtype=np.float32)

                # Handle NaN/inf values
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                features = np.clip(features, -10.0, 10.0)

                features_list.append(features)

            return np.array(features_list, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"LF feature extraction failed: {e}")
            return np.zeros((self.lf_seq_len, self.lf_feat_dim), dtype=np.float32)

    def _extract_static_features(self, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract static/contextual features"""
        try:
            # Market session information
            session = market_state.get('market_session', 'UNKNOWN')
            session_encoding = {
                'PREMARKET': 0.0,
                'REGULAR': 1.0,
                'POSTMARKET': 0.5,
                'CLOSED': -1.0,
                'UNKNOWN': 0.0
            }.get(session, 0.0)

            # Previous day context
            prev_day_data = market_state.get('previous_day_data', {})
            prev_close = prev_day_data.get('close', 0.0)
            current_price = market_state.get('current_price', 0.0)

            # Distance from previous close
            prev_close_distance = 0.0
            if prev_close > 0 and current_price > 0:
                prev_close_distance = (current_price - prev_close) / prev_close

            # Intraday range
            intraday_high = market_state.get('intraday_high', current_price)
            intraday_low = market_state.get('intraday_low', current_price)

            intraday_position = 0.0
            if intraday_high > intraday_low and current_price > 0:
                intraday_position = (current_price - intraday_low) / (intraday_high - intraday_low)

            # Market volatility proxy
            volatility_proxy = 0.0
            if intraday_high > 0 and intraday_low > 0:
                volatility_proxy = (intraday_high - intraday_low) / max(current_price, 0.01)

            # Create static feature vector (5 features)
            features = np.array([
                session_encoding,  # Market session
                prev_close_distance,  # Distance from prev close
                intraday_position,  # Position in intraday range
                volatility_proxy,  # Volatility proxy
                1.0  # Market state indicator
            ], dtype=np.float32)

            # Handle NaN/inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            features = np.clip(features, -5.0, 5.0)

            # Ensure correct shape for model
            return features.reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Static feature extraction failed: {e}")
            return np.zeros((1, self.static_feat_dim), dtype=np.float32)