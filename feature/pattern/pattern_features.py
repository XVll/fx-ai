"""Pattern detection features for identifying trading setups like squeezes, triangles, and swing points."""

import numpy as np
from typing import Dict, List, Tuple, Any


class PatternFeatures:
    """Extract pattern-based features for squeeze and momentum trading."""

    def __init__(self):
        self.feature_names = [
            # Swing point features
            "swing_high_distance",  # Distance to last swing high
            "swing_low_distance",  # Distance to last swing low
            "swing_high_price_pct",  # % from current price to swing high
            "swing_low_price_pct",  # % from current price to swing low
            "bars_since_swing_high",  # Time since last swing high
            "bars_since_swing_low",  # Time since last swing low
            # Pattern detection
            "higher_highs_count",  # Number of higher highs in window
            "higher_lows_count",  # Number of higher lows in window
            "lower_highs_count",  # Number of lower highs in window
            "lower_lows_count",  # Number of lower lows in window
            # Range and consolidation
            "range_compression",  # Current range vs average range
            "consolidation_score",  # How tight is the consolidation
            "triangle_apex_distance",  # Distance to triangle apex if detected
            # Momentum patterns
            "momentum_alignment",  # Are multiple timeframes aligned
            "breakout_potential",  # Likelihood of breakout from range
            "squeeze_intensity",  # Combined squeeze indicators
        ]

    def calculate(
        self, market_data: Dict[str, Any], context: str = "mf"
    ) -> Dict[str, float]:
        """Calculate pattern features from market data."""
        features = {}

        # Get the appropriate data window based on context
        if context == "mf":
            bars = market_data.get("mf_bars_1m", [])
            lookback = min(20, len(bars))  # 20 minute lookback
        elif context == "lf":
            bars = market_data.get("lf_bars_5m", [])
            lookback = min(12, len(bars))  # 1 hour lookback
        else:
            return self._get_default_features()

        if len(bars) < 3:
            return self._get_default_features()

        # Extract price data
        highs = [b["high"] for b in bars[-lookback:]]
        lows = [b["low"] for b in bars[-lookback:]]
        closes = [b["close"] for b in bars[-lookback:]]
        volumes = [b["volume"] for b in bars[-lookback:]]

        current_price = closes[-1] if closes else 0

        # Find swing points
        swing_high_idx, swing_high_price = self._find_swing_high(highs)
        swing_low_idx, swing_low_price = self._find_swing_low(lows)

        # Swing point features
        features["swing_high_distance"] = swing_high_price - current_price
        features["swing_low_distance"] = current_price - swing_low_price
        features["swing_high_price_pct"] = (
            (swing_high_price - current_price) / current_price
            if current_price > 0
            else 0
        )
        features["swing_low_price_pct"] = (
            (current_price - swing_low_price) / current_price
            if current_price > 0
            else 0
        )
        features["bars_since_swing_high"] = (
            len(highs) - 1 - swing_high_idx if swing_high_idx >= 0 else len(highs)
        )
        features["bars_since_swing_low"] = (
            len(lows) - 1 - swing_low_idx if swing_low_idx >= 0 else len(lows)
        )

        # Pattern detection
        features["higher_highs_count"] = self._count_higher_highs(highs)
        features["higher_lows_count"] = self._count_higher_lows(lows)
        features["lower_highs_count"] = self._count_lower_highs(highs)
        features["lower_lows_count"] = self._count_lower_lows(lows)

        # Range and consolidation
        recent_range = max(highs[-5:]) - min(lows[-5:]) if len(highs) >= 5 else 0
        avg_range = np.mean([highs[i] - lows[i] for i in range(len(highs))])
        features["range_compression"] = recent_range / avg_range if avg_range > 0 else 1

        # Consolidation score (lower is tighter)
        features["consolidation_score"] = (
            np.std(closes[-10:]) / np.mean(closes[-10:])
            if len(closes) >= 10 and np.mean(closes[-10:]) > 0
            else 0
        )

        # Triangle detection (simplified)
        features["triangle_apex_distance"] = self._detect_triangle_apex(highs, lows)

        # Momentum alignment (simplified - would need multiple timeframes)
        short_ma = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
        long_ma = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        features["momentum_alignment"] = 1.0 if short_ma > long_ma else -1.0

        # Breakout potential (volume + range compression)
        recent_vol = np.mean(volumes[-3:]) if len(volumes) >= 3 else 0
        avg_vol = np.mean(volumes) if volumes else 1
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 1
        features["breakout_potential"] = vol_surge * (
            1 / (features["range_compression"] + 0.1)
        )

        # Squeeze intensity (combination of factors)
        features["squeeze_intensity"] = (
            (1 / (features["consolidation_score"] + 0.01))
            * features["range_compression"]
            * vol_surge
        )

        return features

    def _find_swing_high(self, highs: List[float]) -> Tuple[int, float]:
        """Find the most recent swing high in the data."""
        if len(highs) < 3:
            return -1, highs[-1] if highs else 0

        for i in range(len(highs) - 2, 0, -1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                return i, highs[i]

        # If no swing found, return the highest point
        max_idx = np.argmax(highs)
        return max_idx, highs[max_idx]

    def _find_swing_low(self, lows: List[float]) -> Tuple[int, float]:
        """Find the most recent swing low in the data."""
        if len(lows) < 3:
            return -1, lows[-1] if lows else 0

        for i in range(len(lows) - 2, 0, -1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                return i, lows[i]

        # If no swing found, return the lowest point
        min_idx = np.argmin(lows)
        return min_idx, lows[min_idx]

    def _count_higher_highs(self, highs: List[float]) -> int:
        """Count the number of higher highs in the sequence."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] > highs[i - 1]:
                count += 1
        return count

    def _count_higher_lows(self, lows: List[float]) -> int:
        """Count the number of higher lows in the sequence."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] > lows[i - 1]:
                count += 1
        return count

    def _count_lower_highs(self, highs: List[float]) -> int:
        """Count the number of lower highs in the sequence."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] < highs[i - 1]:
                count += 1
        return count

    def _count_lower_lows(self, lows: List[float]) -> int:
        """Count the number of lower lows in the sequence."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] < lows[i - 1]:
                count += 1
        return count

    def _detect_triangle_apex(self, highs: List[float], lows: List[float]) -> float:
        """Detect if price is forming a triangle pattern and estimate distance to apex."""
        if len(highs) < 5:
            return 0.0

        # Simple triangle detection: converging highs and lows
        high_slope = (highs[-1] - highs[0]) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)

        # If slopes are converging (high slope negative, low slope positive)
        if high_slope < 0 and low_slope > 0:
            # Estimate where lines would meet
            convergence_bars = (highs[-1] - lows[-1]) / (low_slope - high_slope)
            return max(0, convergence_bars)

        return 0.0

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values."""
        return {name: 0.0 for name in self.feature_names}
