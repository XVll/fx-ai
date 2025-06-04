"""Medium-frequency swing high/low features"""

from typing import Dict, Any, Optional, List
from feature.feature_base import BaseFeature, FeatureConfig


class SwingHighDistance1mFeature(BaseFeature):
    """Distance to recent swing high on 1-minute timeframe"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="1m_swing_high_distance",
                normalize=False,  # Already normalized to [0, 1]
            )
        )
        self.lookback_periods = 20  # Look for swings in last 20 bars

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to swing high"""
        current_price = market_data.get("current_price", 0.0)
        bars_1m = market_data.get("1m_bars_window", [])

        if not bars_1m or current_price <= 0:
            return 0.5  # Default middle distance

        # Find swing high
        swing_high = self._find_swing_high(bars_1m)

        if swing_high is None or swing_high <= current_price:
            return 0.0  # At or above swing high

        # Calculate normalized distance
        # 0 = at swing high, 1 = far from swing high
        # Normalize by percentage distance
        distance_pct = (swing_high - current_price) / current_price

        # Cap at 10% for normalization
        normalized_distance = min(distance_pct / 0.1, 1.0)

        return normalized_distance

    def _find_swing_high(self, bars: List[Dict]) -> Optional[float]:
        """Find the most recent swing high"""
        if len(bars) < 3:
            return None

        highs = [float(bar.get("high", 0)) for bar in bars[-self.lookback_periods :]]

        # Find local maxima (swing highs)
        swing_highs = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append(highs[i])

        if not swing_highs:
            # No swing high found, use highest point
            return max(highs) if highs else None

        # Return most recent swing high
        return swing_highs[-1]

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 20,
            "fields": ["current_price", "1m_bars_window"],
        }


class SwingLowDistance1mFeature(BaseFeature):
    """Distance to recent swing low on 1-minute timeframe"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="1m_swing_low_distance",
                normalize=False,  # Already normalized to [0, 1]
            )
        )
        self.lookback_periods = 20

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to swing low"""
        current_price = market_data.get("current_price", 0.0)
        bars_1m = market_data.get("1m_bars_window", [])

        if not bars_1m or current_price <= 0:
            return 0.5  # Default middle distance

        # Find swing low
        swing_low = self._find_swing_low(bars_1m)

        if swing_low is None or swing_low >= current_price:
            return 0.0  # At or below swing low

        # Calculate normalized distance
        # 0 = at swing low, 1 = far from swing low
        distance_pct = (current_price - swing_low) / current_price

        # Cap at 10% for normalization
        normalized_distance = min(distance_pct / 0.1, 1.0)

        return normalized_distance

    def _find_swing_low(self, bars: List[Dict]) -> Optional[float]:
        """Find the most recent swing low"""
        if len(bars) < 3:
            return None

        lows = [
            float(bar.get("low", float("inf")))
            for bar in bars[-self.lookback_periods :]
        ]

        # Find local minima (swing lows)
        swing_lows = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(lows[i])

        if not swing_lows:
            # No swing low found, use lowest point
            return min(lows) if lows and min(lows) != float("inf") else None

        # Return most recent swing low
        return swing_lows[-1]

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 20,
            "fields": ["current_price", "1m_bars_window"],
        }


class SwingHighDistance5mFeature(BaseFeature):
    """Distance to recent swing high on 5-minute timeframe"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="5m_swing_high_distance",
                normalize=False,  # Already normalized to [0, 1]
            )
        )
        self.lookback_periods = 20

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to swing high"""
        current_price = market_data.get("current_price", 0.0)
        bars_5m = market_data.get("5m_bars_window", [])

        if not bars_5m or current_price <= 0:
            return 0.5  # Default middle distance

        # Find swing high
        swing_high = self._find_swing_high(bars_5m)

        if swing_high is None or swing_high <= current_price:
            return 0.0  # At or above swing high

        # Calculate normalized distance
        distance_pct = (swing_high - current_price) / current_price

        # Cap at 10% for normalization
        normalized_distance = min(distance_pct / 0.1, 1.0)

        return normalized_distance

    def _find_swing_high(self, bars: List[Dict]) -> Optional[float]:
        """Find the most recent swing high"""
        if len(bars) < 3:
            return None

        highs = [float(bar.get("high", 0)) for bar in bars[-self.lookback_periods :]]

        # Find local maxima
        swing_highs = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append(highs[i])

        if not swing_highs:
            return max(highs) if highs else None

        return swing_highs[-1]

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 20,
            "fields": ["current_price", "5m_bars_window"],
        }


class SwingLowDistance5mFeature(BaseFeature):
    """Distance to recent swing low on 5-minute timeframe"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="5m_swing_low_distance",
                normalize=False,  # Already normalized to [0, 1]
            )
        )
        self.lookback_periods = 20

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to swing low"""
        current_price = market_data.get("current_price", 0.0)
        bars_5m = market_data.get("5m_bars_window", [])

        if not bars_5m or current_price <= 0:
            return 0.5  # Default middle distance

        # Find swing low
        swing_low = self._find_swing_low(bars_5m)

        if swing_low is None or swing_low >= current_price:
            return 0.0  # At or below swing low

        # Calculate normalized distance
        distance_pct = (current_price - swing_low) / current_price

        # Cap at 10% for normalization
        normalized_distance = min(distance_pct / 0.1, 1.0)

        return normalized_distance

    def _find_swing_low(self, bars: List[Dict]) -> Optional[float]:
        """Find the most recent swing low"""
        if len(bars) < 3:
            return None

        lows = [
            float(bar.get("low", float("inf")))
            for bar in bars[-self.lookback_periods :]
        ]

        # Find local minima
        swing_lows = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(lows[i])

        if not swing_lows:
            return min(lows) if lows and min(lows) != float("inf") else None

        return swing_lows[-1]

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 20,
            "fields": ["current_price", "5m_bars_window"],
        }
