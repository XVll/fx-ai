"""Support/resistance and price level features"""

import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig


def detect_support_levels(daily_bars: List[Dict], lookback: int = 20) -> List[float]:
    """Detect support levels from daily bars"""
    if not daily_bars:
        return []

    # Use last N bars
    bars = daily_bars[-lookback:] if len(daily_bars) > lookback else daily_bars

    # Collect all lows
    lows = [bar.get("low", 0) for bar in bars if bar.get("low")]

    if not lows:
        return []

    # Simple approach: find local minima
    support_levels = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            support_levels.append(lows[i])

    # Also add the absolute minimum
    if lows:
        support_levels.append(min(lows))

    # Remove duplicates and sort
    support_levels = sorted(list(set(support_levels)))

    return support_levels


def detect_resistance_levels(daily_bars: List[Dict], lookback: int = 20) -> List[float]:
    """Detect resistance levels from daily bars"""
    if not daily_bars:
        return []

    # Use last N bars
    bars = daily_bars[-lookback:] if len(daily_bars) > lookback else daily_bars

    # Collect all highs
    highs = [bar.get("high", 0) for bar in bars if bar.get("high")]

    if not highs:
        return []

    # Simple approach: find local maxima
    resistance_levels = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resistance_levels.append(highs[i])

    # Also add the absolute maximum
    if highs:
        resistance_levels.append(max(highs))

    # Remove duplicates and sort
    resistance_levels = sorted(list(set(resistance_levels)))

    return resistance_levels


class DistanceToClosestSupportFeature(BaseFeature):
    """Distance to nearest support level"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="support_distance", normalize=True)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to closest support"""
        current_price = market_data.get("current_price", 0)

        if current_price == 0:
            return 0.0

        # Check for explicit support levels
        support_levels = market_data.get("support_levels", [])

        # If not provided, detect from daily bars
        if not support_levels:
            daily_bars = market_data.get("daily_bars_window", [])
            support_levels = detect_support_levels(daily_bars)

        if not support_levels:
            return 0.05  # Default distance

        # Find closest support below current price
        supports_below = [s for s in support_levels if s < current_price]

        if not supports_below:
            # No support below, use the highest support
            closest_support = max(support_levels)
        else:
            closest_support = max(supports_below)

        # Calculate distance as percentage
        distance = abs(current_price - closest_support) / current_price

        return distance

    def get_default_value(self) -> float:
        return 0.05

    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0=at support, 1=far (20%+ away)"""
        return {"min": 0.0, "max": 0.20}

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "daily_bars",
            "lookback": 20,
            "fields": ["current_price", "daily_bars_window"],
        }


class DistanceToClosestResistanceFeature(BaseFeature):
    """Distance to nearest resistance level"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="resistance_distance", normalize=False)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to closest resistance"""
        current_price = market_data.get("current_price", 0)

        if current_price == 0:
            return 0.0

        # Check for explicit resistance levels
        resistance_levels = market_data.get("resistance_levels", [])

        # If not provided, detect from daily bars
        if not resistance_levels:
            daily_bars = market_data.get("daily_bars_window", [])
            resistance_levels = detect_resistance_levels(daily_bars)

        if not resistance_levels:
            return 0.05  # Default distance

        # Find closest resistance above current price
        resistances_above = [r for r in resistance_levels if r > current_price]

        if not resistances_above:
            # No resistance above, use the lowest resistance
            closest_resistance = min(resistance_levels)
        else:
            closest_resistance = min(resistances_above)

        # Calculate distance as percentage
        distance = abs(closest_resistance - current_price) / current_price

        return distance

    def get_default_value(self) -> float:
        return 0.05

    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0=at resistance, 1=far (20%+ away)"""
        return {"min": 0.0, "max": 0.20}

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "daily_bars",
            "lookback": 20,
            "fields": ["current_price", "daily_bars_window"],
        }


class WholeDollarProximityFeature(BaseFeature):
    """Proximity to whole dollar levels"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="whole_dollar_proximity", normalize=True)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate proximity to nearest whole dollar"""
        current_price = market_data.get("current_price")

        if current_price is None or current_price == 0:
            return 0.5

        # Find distance to nearest whole dollar
        lower_whole = np.floor(current_price)
        upper_whole = np.ceil(current_price)

        # Distance to nearest
        dist_to_lower = current_price - lower_whole
        dist_to_upper = upper_whole - current_price

        min_distance = min(dist_to_lower, dist_to_upper)

        return min_distance

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0=at level, 1=max distance (50 cents)"""
        return {"min": 0.0, "max": 0.50}

    def get_requirements(self) -> Dict[str, Any]:
        return {"fields": ["current_price"], "data_type": "current"}


class HalfDollarProximityFeature(BaseFeature):
    """Proximity to half dollar levels"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="half_dollar_proximity", normalize=True)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate proximity to nearest half dollar"""
        current_price = market_data.get("current_price")

        if current_price is None or current_price == 0:
            return 0.25

        # Find distance to nearest half dollar (0.00, 0.50, 1.00, 1.50, etc.)
        scaled_price = current_price * 2
        lower_half = np.floor(scaled_price) / 2
        upper_half = np.ceil(scaled_price) / 2

        # Distance to nearest
        dist_to_lower = current_price - lower_half
        dist_to_upper = upper_half - current_price

        min_distance = min(dist_to_lower, dist_to_upper)

        return min_distance

    def get_default_value(self) -> float:
        return 0.25

    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0=at level, 1=max distance (25 cents)"""
        return {"min": 0.0, "max": 0.25}

    def get_requirements(self) -> Dict[str, Any]:
        return {"fields": ["current_price"], "data_type": "current"}
