"""Medium-frequency candle analysis features"""

from typing import Dict, Any, Optional
from feature.feature_base import BaseFeature, FeatureConfig


class PositionInPreviousCandle1mFeature(BaseFeature):
    """Position of current price in previous 1-minute candle range"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="1m_position_in_previous_candle",
                normalize=False,  # Already normalized to [0, 1]
            )
        )

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position in previous candle"""
        current_price = market_data.get("current_price", 0.0)
        bars_1m = market_data.get("1m_bars_window", [])

        if not bars_1m or current_price <= 0:
            return 0.5  # Default to middle

        # Get previous candle (second to last if current is incomplete)
        if len(bars_1m) >= 2:
            prev_bar = bars_1m[-2]
        else:
            prev_bar = bars_1m[-1]

        high = float(prev_bar.get("high", current_price))
        low = float(prev_bar.get("low", current_price))

        # Handle no range
        if high == low:
            return 0.5

        # Calculate position and clamp to [0, 1]
        position = (current_price - low) / (high - low)
        return max(0.0, min(1.0, position))

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 2,
            "fields": ["current_price", "1m_bars_window"],
        }


class PositionInPreviousCandle5mFeature(BaseFeature):
    """Position of current price in previous 5-minute candle range"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="5m_position_in_previous_candle",
                normalize=False,  # Already normalized to [0, 1]
            )
        )

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position in previous candle"""
        current_price = market_data.get("current_price", 0.0)
        bars_5m = market_data.get("5m_bars_window", [])

        if not bars_5m or current_price <= 0:
            return 0.5  # Default to middle

        # Get previous candle
        if len(bars_5m) >= 2:
            prev_bar = bars_5m[-2]
        else:
            prev_bar = bars_5m[-1]

        high = float(prev_bar.get("high", current_price))
        low = float(prev_bar.get("low", current_price))

        # Handle no range
        if high == low:
            return 0.5

        # Calculate position and clamp to [0, 1]
        position = (current_price - low) / (high - low)
        return max(0.0, min(1.0, position))

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 2,
            "fields": ["current_price", "5m_bars_window"],
        }


class UpperWickRelative1mFeature(BaseFeature):
    """Upper wick size relative to candle range for 1-minute bars"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="1m_upper_wick_relative",
                normalize=False,  # Already normalized to [0, 1]
            )
        )

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative upper wick size"""
        bars_1m = market_data.get("1m_bars_window", [])

        if not bars_1m:
            return 0.0

        # Get last complete bar
        bar = bars_1m[-1]

        open_price = float(bar.get("open", 0.0))
        high = float(bar.get("high", 0.0))
        low = float(bar.get("low", 0.0))
        close = float(bar.get("close", 0.0))

        # Handle invalid data
        if high == 0 or low == 0 or high < low:
            return 0.0

        # Calculate range
        candle_range = high - low
        if candle_range == 0:
            return 0.0  # No range, no wick

        # Upper wick = high - max(open, close)
        body_top = max(open_price, close)
        upper_wick = high - body_top

        # Relative wick size
        return upper_wick / candle_range

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "1m_bars", "lookback": 1, "fields": ["1m_bars_window"]}


class LowerWickRelative1mFeature(BaseFeature):
    """Lower wick size relative to candle range for 1-minute bars"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="1m_lower_wick_relative",
                normalize=False,  # Already normalized to [0, 1]
            )
        )

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative lower wick size"""
        bars_1m = market_data.get("1m_bars_window", [])

        if not bars_1m:
            return 0.0

        # Get last complete bar
        bar = bars_1m[-1]

        open_price = float(bar.get("open", 0.0))
        high = float(bar.get("high", 0.0))
        low = float(bar.get("low", 0.0))
        close = float(bar.get("close", 0.0))

        # Handle invalid data
        if high == 0 or low == 0 or high < low:
            return 0.0

        # Calculate range
        candle_range = high - low
        if candle_range == 0:
            return 0.0  # No range, no wick

        # Lower wick = min(open, close) - low
        body_bottom = min(open_price, close)
        lower_wick = body_bottom - low

        # Relative wick size
        return lower_wick / candle_range

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "1m_bars", "lookback": 1, "fields": ["1m_bars_window"]}


class UpperWickRelative5mFeature(BaseFeature):
    """Upper wick size relative to candle range for 5-minute bars"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="5m_upper_wick_relative",
                normalize=False,  # Already normalized to [0, 1]
            )
        )

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative upper wick size"""
        bars_5m = market_data.get("5m_bars_window", [])

        if not bars_5m:
            return 0.0

        # Get last complete bar
        bar = bars_5m[-1]

        open_price = float(bar.get("open", 0.0))
        high = float(bar.get("high", 0.0))
        low = float(bar.get("low", 0.0))
        close = float(bar.get("close", 0.0))

        # Handle invalid data
        if high == 0 or low == 0 or high < low:
            return 0.0

        # Calculate range
        candle_range = high - low
        if candle_range == 0:
            return 0.0  # No range, no wick

        # Upper wick = high - max(open, close)
        body_top = max(open_price, close)
        upper_wick = high - body_top

        # Relative wick size
        return upper_wick / candle_range

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "5m_bars", "lookback": 1, "fields": ["5m_bars_window"]}


class LowerWickRelative5mFeature(BaseFeature):
    """Lower wick size relative to candle range for 5-minute bars"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(
            config
            or FeatureConfig(
                name="5m_lower_wick_relative",
                normalize=False,  # Already normalized to [0, 1]
            )
        )

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative lower wick size"""
        bars_5m = market_data.get("5m_bars_window", [])

        if not bars_5m:
            return 0.0

        # Get last complete bar
        bar = bars_5m[-1]

        open_price = float(bar.get("open", 0.0))
        high = float(bar.get("high", 0.0))
        low = float(bar.get("low", 0.0))
        close = float(bar.get("close", 0.0))

        # Handle invalid data
        if high == 0 or low == 0 or high < low:
            return 0.0

        # Calculate range
        candle_range = high - low
        if candle_range == 0:
            return 0.0  # No range, no wick

        # Lower wick = min(open, close) - low
        body_bottom = min(open_price, close)
        lower_wick = body_bottom - low

        # Relative wick size
        return lower_wick / candle_range

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "5m_bars", "lookback": 1, "fields": ["5m_bars_window"]}
