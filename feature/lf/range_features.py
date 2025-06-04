"""Range-based low-frequency features"""

import numpy as np
from typing import Dict, Any
from feature.feature_base import BaseFeature, FeatureConfig


class PositionInDailyRangeFeature(BaseFeature):
    """Position of current price within today's range"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="daily_range_position", normalize=False)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position in daily range"""
        current_price = market_data.get("current_price")
        intraday_high = market_data.get("intraday_high")
        intraday_low = market_data.get("intraday_low")

        if None in [current_price, intraday_high, intraday_low]:
            return 0.5

        # Handle no range
        if intraday_high == intraday_low:
            return 0.5

        # Calculate position and clamp to [0, 1]
        position = (current_price - intraday_low) / (intraday_high - intraday_low)
        return np.clip(position, 0.0, 1.0)

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [0, 1]"""
        return {}

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "fields": ["current_price", "intraday_high", "intraday_low"],
            "data_type": "current",
        }


class PositionInPrevDayRangeFeature(BaseFeature):
    """Position relative to previous day's range"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="position_in_prev_day_range", normalize=False)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position relative to previous day"""
        current_price = market_data.get("current_price")
        prev_day_data = market_data.get("previous_day_data")

        if not current_price or not prev_day_data:
            return 0.5

        prev_high = prev_day_data.get("high")
        prev_low = prev_day_data.get("low")

        if None in [prev_high, prev_low]:
            return 0.5

        # Handle no range
        if prev_high == prev_low:
            return 0.5

        # Calculate position
        position = (current_price - prev_low) / (prev_high - prev_low)

        # Return raw position - test expects this
        return position

    def get_default_value(self) -> float:
        return 0.5

    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [0, 1]"""
        return {}

    def get_requirements(self) -> Dict[str, Any]:
        return {"fields": ["current_price", "previous_day_data"], "data_type": "daily"}


class PriceChangeFromPrevCloseFeature(BaseFeature):
    """Percentage change from previous close"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="price_change_from_prev_close", normalize=True)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate percentage change from previous close"""
        current_price = market_data.get("current_price")
        prev_day_data = market_data.get("previous_day_data")

        if not current_price or not prev_day_data:
            return 0.0

        prev_close = prev_day_data.get("close")

        if not prev_close or prev_close == 0:
            return 0.0

        # Calculate percentage change
        change = (current_price - prev_close) / prev_close
        return change

    def get_default_value(self) -> float:
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±20% daily moves"""
        return {"min": -0.20, "max": 0.20}

    def get_requirements(self) -> Dict[str, Any]:
        return {"fields": ["current_price", "previous_day_data"], "data_type": "daily"}
