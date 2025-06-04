"""Base class for pattern features with common implementations."""

from typing import Dict, Any
from feature.feature_base import BaseFeature, FeatureConfig
from .pattern_features import PatternFeatures


# Shared pattern feature extractor
_pattern_extractor = PatternFeatures()


class PatternFeatureBase(BaseFeature):
    """Base class for pattern-based features."""

    def __init__(self, feature_name: str, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name=feature_name, normalize=True)
        super().__init__(config)
        self.feature_name = feature_name

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate the raw feature value using pattern extractor."""
        features = _pattern_extractor.calculate(market_data, context="mf")
        return features.get(self.feature_name, 0.0)

    def get_default_value(self) -> float:
        """Default value for pattern features."""
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        """Get normalization parameters based on feature type."""
        # Distance and percentage features
        if "distance" in self.feature_name or "pct" in self.feature_name:
            return {"min": -0.1, "max": 0.1, "range_type": "symmetric"}

        # Count features
        elif "count" in self.feature_name or "bars_since" in self.feature_name:
            return {"min": 0, "max": 20}

        # Score features (0-1 range)
        elif (
            "score" in self.feature_name
            or "potential" in self.feature_name
            or "intensity" in self.feature_name
        ):
            return {"min": 0, "max": 5}

        # Compression and ratio features
        elif "compression" in self.feature_name:
            return {"min": 0, "max": 2}

        # Alignment features (-1 to 1)
        elif "alignment" in self.feature_name:
            return {"min": -1, "max": 1, "range_type": "symmetric"}

        # Default
        else:
            return {"min": -1, "max": 1, "range_type": "symmetric"}

    def get_requirements(self) -> Dict[str, Any]:
        """Data requirements for pattern features."""
        return {
            "mf_bars_1m": {"lookback": 20, "fields": ["high", "low", "close", "volume"]}
        }
