"""Professional-grade features using pandas and ta library."""

import pandas as pd
import numpy as np
from ta import trend, momentum, volume, volatility
from typing import Dict, Any
from ..feature_base import BaseFeature


class ProfessionalEMASystemFeature(BaseFeature):
    """EMA analysis using ta library - industry standard implementation."""

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate EMA system strength using ta library."""
        try:
            bars_1m = market_data.get("1m_bars_window", [])
            if len(bars_1m) < 25:
                return 0.0

            # Convert to pandas DataFrame - professional standard
            df = pd.DataFrame(bars_1m)
            if df.empty or "close" not in df.columns:
                return 0.0

            # Use ta library for EMAs - optimized and tested
            df["ema9"] = trend.EMAIndicator(df["close"], window=9).ema_indicator()
            df["ema20"] = trend.EMAIndicator(df["close"], window=20).ema_indicator()

            # Remove NaN values
            df = df.dropna()
            if len(df) < 10:
                return 0.0

            # Professional EMA analysis
            recent_data = df.tail(10)

            # 1. EMA alignment (bullish when EMA9 > EMA20)
            ema_alignment = (recent_data["ema9"] > recent_data["ema20"]).mean()
            alignment_direction = (
                1
                if recent_data["ema9"].iloc[-1] > recent_data["ema20"].iloc[-1]
                else -1
            )

            # 2. EMA slope strength (trending vs flat)
            ema9_slope = (
                recent_data["ema9"].iloc[-1] - recent_data["ema9"].iloc[0]
            ) / recent_data["ema9"].iloc[0]
            ema20_slope = (
                recent_data["ema20"].iloc[-1] - recent_data["ema20"].iloc[0]
            ) / recent_data["ema20"].iloc[0]

            # 3. Price position relative to EMAs
            price_above_ema9 = (recent_data["close"] > recent_data["ema9"]).mean()
            price_above_ema20 = (recent_data["close"] > recent_data["ema20"]).mean()

            # Combined EMA system strength
            ema_strength = (
                ema_alignment * 0.3  # Alignment consistency
                + abs(ema9_slope) * 10 * 0.3  # EMA9 slope strength
                + abs(ema20_slope) * 20 * 0.2  # EMA20 slope strength
                + max(price_above_ema9, 1 - price_above_ema9) * 0.2  # Price consistency
            )

            return float(np.clip(ema_strength * alignment_direction, -1.0, 1.0))

        except Exception:
            return 0.0

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {"min": -1.0, "max": 1.0, "range_type": "symmetric"}

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "mf_data", "lookback": 25, "fields": ["close"]}


class ProfessionalVWAPAnalysisFeature(BaseFeature):
    """VWAP analysis using pandas - vectorized operations."""

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate VWAP relationship using pandas for efficiency."""
        try:
            bars_1m = market_data.get("1m_bars_window", [])
            if len(bars_1m) < 15:
                return 0.0

            # Use pandas for efficient time series operations
            df = pd.DataFrame(bars_1m)
            required_cols = ["close", "volume", "high", "low"]
            if not all(col in df.columns for col in required_cols):
                return 0.0

            # Calculate VWAP if not available - professional standard
            if "vwap" not in df.columns:
                typical_price = (df["high"] + df["low"] + df["close"]) / 3
                df["vwap"] = (typical_price * df["volume"]).cumsum() / df[
                    "volume"
                ].cumsum()

            # Drop NaN and ensure we have enough data
            df = df.dropna()
            if len(df) < 10:
                return 0.0

            recent_data = df.tail(15)

            # Professional VWAP analysis using pandas vectorized operations
            # 1. Price-VWAP relationship
            price_vwap_ratio = recent_data["close"] / recent_data["vwap"]
            relative_position = (price_vwap_ratio - 1.0).rolling(window=3).mean()

            # 2. VWAP slope using pandas
            vwap_returns = recent_data["vwap"].pct_change().dropna()
            vwap_trend = (
                vwap_returns.rolling(window=5).mean().iloc[-1]
                if len(vwap_returns) >= 5
                else 0
            )

            # 3. Support/resistance behavior
            # Count touches within 0.2% of VWAP
            touch_threshold = 0.002
            touches = (abs(price_vwap_ratio - 1.0) < touch_threshold).sum()

            # 4. Volume confirmation around VWAP
            above_vwap = recent_data["close"] > recent_data["vwap"]
            volume_above = recent_data.loc[above_vwap, "volume"].mean()
            volume_below = recent_data.loc[~above_vwap, "volume"].mean()

            volume_bias = 0.0
            if volume_above > 0 and volume_below > 0:
                volume_bias = (volume_above - volume_below) / (
                    volume_above + volume_below
                )

            # Combine VWAP metrics
            current_position = (
                relative_position.iloc[-1] if not relative_position.empty else 0
            )
            touch_score = min(touches / 5.0, 1.0)

            vwap_score = (
                current_position * 2.0 * 0.4  # Current position (scaled)
                + vwap_trend * 50 * 0.3  # VWAP trend
                + touch_score * 0.2  # Support/resistance
                + volume_bias * 0.1  # Volume confirmation
            )

            return float(np.clip(vwap_score, -1.0, 1.0))

        except Exception:
            return 0.0

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {"min": -1.0, "max": 1.0, "range_type": "symmetric"}

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "mf_data",
            "lookback": 15,
            "fields": ["close", "volume", "high", "low"],
        }


class ProfessionalMomentumQualityFeature(BaseFeature):
    """Momentum quality using ta library indicators."""

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate momentum quality using professional indicators."""
        try:
            bars_1m = market_data.get("1m_bars_window", [])
            if len(bars_1m) < 20:
                return 0.0

            df = pd.DataFrame(bars_1m)
            required_cols = ["close", "volume", "high", "low"]
            if not all(col in df.columns for col in required_cols):
                return 0.0

            # Use ta library for professional momentum indicators
            # RSI for momentum strength
            df["rsi"] = momentum.RSIIndicator(df["close"], window=14).rsi()

            # MACD for trend momentum
            macd = trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_histogram"] = macd.macd_diff()

            # Volume indicators
            df["volume_sma"] = volume.VolumeSMAIndicator(
                df["close"], df["volume"]
            ).volume_sma()

            # Clean data
            df = df.dropna()
            if len(df) < 10:
                return 0.0

            recent = df.tail(10)

            # Professional momentum analysis
            # 1. RSI momentum (50-70 = bullish momentum, 30-50 = bearish momentum)
            current_rsi = recent["rsi"].iloc[-1]
            if current_rsi > 50:
                rsi_momentum = min((current_rsi - 50) / 20, 1.0)  # Bullish momentum
            else:
                rsi_momentum = max((current_rsi - 50) / 20, -1.0)  # Bearish momentum

            # 2. MACD momentum quality
            macd_value = recent["macd"].iloc[-1]
            macd_signal = recent["macd_signal"].iloc[-1]
            macd_histogram = recent["macd_histogram"].iloc[-1]

            # MACD above signal and histogram positive = bullish momentum
            macd_momentum = (
                1.0 if macd_value > macd_signal and macd_histogram > 0 else -1.0
            )
            macd_strength = abs(macd_histogram) / (abs(macd_value) + 1e-8)
            macd_strength = min(macd_strength, 1.0)

            # 3. Volume confirmation
            recent_volume = recent["volume"].mean()
            avg_volume = (
                recent["volume_sma"].iloc[-1]
                if not pd.isna(recent["volume_sma"].iloc[-1])
                else recent_volume
            )
            volume_confirmation = (
                min(recent_volume / (avg_volume + 1e-8), 2.0) / 2.0
            )  # Normalize to [0,1]

            # Combined momentum quality
            momentum_quality = (
                rsi_momentum * 0.4  # RSI momentum direction
                + macd_momentum * macd_strength * 0.4  # MACD momentum quality
                + volume_confirmation * 0.2  # Volume support
            )

            return float(np.clip(momentum_quality, -1.0, 1.0))

        except Exception:
            return 0.0

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {"min": -1.0, "max": 1.0, "range_type": "symmetric"}

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "mf_data",
            "lookback": 20,
            "fields": ["close", "volume", "high", "low"],
        }


class ProfessionalVolatilityRegimeFeature(BaseFeature):
    """Volatility regime detection using ta library."""

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Detect volatility regime using professional indicators."""
        try:
            bars_1m = market_data.get("1m_bars_window", [])
            if len(bars_1m) < 20:
                return 0.5  # Neutral

            df = pd.DataFrame(bars_1m)
            required_cols = ["close", "high", "low"]
            if not all(col in df.columns for col in required_cols):
                return 0.5

            # Use ta library for volatility indicators
            # Bollinger Bands for volatility measurement
            bb = volatility.BollingerBands(df["close"], window=14)
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

            # Average True Range for volatility
            df["atr"] = volatility.AverageTrueRange(
                df["high"], df["low"], df["close"]
            ).average_true_range()

            # Clean data
            df = df.dropna()
            if len(df) < 10:
                return 0.5

            recent = df.tail(10)

            # Volatility regime analysis
            # 1. Current vs historical BB width
            current_bb_width = recent["bb_width"].iloc[-1]
            historical_bb_width = df["bb_width"].median()
            bb_ratio = current_bb_width / (historical_bb_width + 1e-8)

            # 2. ATR regime
            current_atr = recent["atr"].iloc[-1]
            historical_atr = df["atr"].median()
            atr_ratio = current_atr / (historical_atr + 1e-8)

            # 3. Recent volatility trend
            bb_trend = (
                recent["bb_width"].iloc[-1] - recent["bb_width"].iloc[0]
            ) / recent["bb_width"].iloc[0]
            atr_trend = (recent["atr"].iloc[-1] - recent["atr"].iloc[0]) / recent[
                "atr"
            ].iloc[0]

            # Combined volatility regime
            # High values = high volatility regime, Low values = low volatility regime
            volatility_score = (
                min(bb_ratio, 3.0) / 3.0 * 0.4  # BB width relative to history
                + min(atr_ratio, 3.0) / 3.0 * 0.4  # ATR relative to history
                + np.clip(bb_trend + atr_trend, -1, 1) * 0.2  # Volatility trend
            )

            return float(np.clip(volatility_score, 0.0, 1.0))

        except Exception:
            return 0.5

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {"min": 0.0, "max": 1.0}

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "mf_data",
            "lookback": 20,
            "fields": ["close", "high", "low"],
        }
