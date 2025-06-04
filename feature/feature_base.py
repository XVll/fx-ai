"""Base feature interface and configuration"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class FeatureConfig:
    """Configuration for a feature"""

    name: str
    enabled: bool = True
    normalize: bool = True
    cache_enabled: bool = False
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BaseFeature(ABC):
    """Base class for all features"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self._cache = {}
        self._last_timestamp = None
        self._last_value = None

    def calculate(self, market_data: Dict[str, Any]) -> float:
        """Calculate normalized feature value"""
        # Check cache if enabled
        if self.config.cache_enabled:
            timestamp = market_data.get("timestamp")
            if timestamp and timestamp == self._last_timestamp:
                return self._last_value

        # Calculate raw value
        try:
            raw_value = self.calculate_raw(market_data)
        except Exception:
            raw_value = self.get_default_value()

        # Handle edge cases
        if raw_value is None or np.isnan(raw_value) or np.isinf(raw_value):
            raw_value = self.get_default_value()

        # Normalize if enabled
        if self.config.normalize:
            value = self.normalize(raw_value)
        else:
            value = raw_value

        # Cache if enabled
        if self.config.cache_enabled and "timestamp" in market_data:
            self._last_timestamp = market_data["timestamp"]
            self._last_value = value

        return value

    def get_value(self, market_data: Dict[str, Any]) -> float:
        """Alias for calculate for compatibility"""
        return self.calculate(market_data)

    def normalize(self, raw_value: float) -> float:
        """Normalize the raw value"""
        params = self.get_normalization_params()

        if "min" in params and "max" in params:
            min_val = params["min"]
            max_val = params["max"]

            if max_val == min_val:
                return 0.0

            # Check if this is a symmetric range that should map to [-1, 1]
            range_type = params.get("range_type", "default")

            if range_type == "symmetric" or (
                min_val < 0 and max_val > 0 and abs(min_val) == abs(max_val)
            ):
                # Symmetric normalization: maps to [-1, 1] with 0 -> 0
                clipped = np.clip(raw_value, min_val, max_val)
                return clipped / max_val
            else:
                # Standard min-max normalization: maps to [0, 1]
                clipped = np.clip(raw_value, min_val, max_val)
                return (clipped - min_val) / (max_val - min_val)

        return raw_value

    def validate_data(self, market_data: Dict[str, Any], required_fields: list) -> None:
        """Validate that required fields are present"""
        for field in required_fields:
            if field not in market_data:
                raise ValueError(f"Missing required field: {field}")

    @abstractmethod
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate the raw feature value"""
        pass

    @abstractmethod
    def get_default_value(self) -> float:
        """Get default value when calculation fails"""
        pass

    @abstractmethod
    def get_normalization_params(self) -> Dict[str, Any]:
        """Get parameters for normalization"""
        pass

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Get data requirements for this feature"""
        pass
