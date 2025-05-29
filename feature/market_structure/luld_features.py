"""Limit Up/Limit Down (LULD) band features for volatility management."""

import numpy as np
from typing import Dict, Any, Tuple
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


class LULDCalculator:
    """Helper class to calculate LULD bands based on SEC rules."""
    
    @staticmethod
    def calculate_luld_bands(reference_price: float, tier: int = 1) -> Tuple[float, float]:
        """Calculate LULD bands based on reference price and tier.
        
        Tier 1: S&P 500 and Russell 1000 stocks
        Tier 2: Other NMS stocks
        
        Simplified bands (actual rules are more complex):
        - Price >= $3.00: 5% bands (Tier 1) or 10% bands (Tier 2)
        - Price $0.75-$3.00: 20% bands
        - Price < $0.75: Lesser of $0.15 or 75%
        """
        if reference_price <= 0:
            return 0.0, 0.0
            
        if reference_price >= 3.00:
            # Normal trading range
            percentage = 0.05 if tier == 1 else 0.10
        elif reference_price >= 0.75:
            # Wider bands for lower priced stocks
            percentage = 0.20
        else:
            # Penny stocks
            percentage = min(0.75, 0.15 / reference_price)
            
        lower_band = reference_price * (1 - percentage)
        upper_band = reference_price * (1 + percentage)
        
        return lower_band, upper_band


@feature_registry.register("distance_to_luld_up", category="lf")
class DistanceToLULDUpFeature(BaseFeature):
    """Distance to upper LULD band as percentage of current price.
    
    Normalized to [0, 1] where 0 means at the band and 1 means far from band.
    Values close to 0 indicate potential limit up halt risk.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="distance_to_luld_up", normalize=True)
        super().__init__(config)
        self.luld_calc = LULDCalculator()
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to upper LULD band."""
        current_price = market_data.get('current_price', 0)
        
        # Use previous close as reference (simplified)
        reference_price = market_data.get('prev_day_close', current_price)
        
        if current_price <= 0 or reference_price <= 0:
            return 0.10  # Default 10% distance
            
        # Assume Tier 2 for MLGO (low float stock)
        lower_band, upper_band = self.luld_calc.calculate_luld_bands(reference_price, tier=2)
        
        # Calculate distance as percentage
        distance = (upper_band - current_price) / current_price
        
        # Clamp to reasonable range
        return max(0.0, min(0.20, distance))  # Max 20% distance
    
    def get_default_value(self) -> float:
        """Default to 10% distance."""
        return 0.10
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0.2 (20%) maps to 1."""
        return {'min': 0.0, 'max': 0.20}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires price data."""
        return {
            'current': {
                'fields': ['current_price', 'prev_day_close']
            }
        }


@feature_registry.register("distance_to_luld_down", category="lf")
class DistanceToLULDDownFeature(BaseFeature):
    """Distance to lower LULD band as percentage of current price.
    
    Normalized to [0, 1] where 0 means at the band and 1 means far from band.
    Values close to 0 indicate potential limit down halt risk.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="distance_to_luld_down", normalize=True)
        super().__init__(config)
        self.luld_calc = LULDCalculator()
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to lower LULD band."""
        current_price = market_data.get('current_price', 0)
        
        # Use previous close as reference (simplified)
        reference_price = market_data.get('prev_day_close', current_price)
        
        if current_price <= 0 or reference_price <= 0:
            return 0.10  # Default 10% distance
            
        # Assume Tier 2 for MLGO (low float stock)
        lower_band, upper_band = self.luld_calc.calculate_luld_bands(reference_price, tier=2)
        
        # Calculate distance as percentage
        distance = (current_price - lower_band) / current_price
        
        # Clamp to reasonable range
        return max(0.0, min(0.20, distance))  # Max 20% distance
    
    def get_default_value(self) -> float:
        """Default to 10% distance."""
        return 0.10
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0.2 (20%) maps to 1."""
        return {'min': 0.0, 'max': 0.20}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires price data."""
        return {
            'current': {
                'fields': ['current_price', 'prev_day_close']
            }
        }


@feature_registry.register("luld_band_width", category="lf")
class LULDBandWidthFeature(BaseFeature):
    """Width of LULD bands as percentage of price.
    
    Wider bands indicate higher allowed volatility.
    Normalized to [0, 1] where higher values mean wider bands.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="luld_band_width", normalize=True)
        super().__init__(config)
        self.luld_calc = LULDCalculator()
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate LULD band width."""
        current_price = market_data.get('current_price', 0)
        reference_price = market_data.get('prev_day_close', current_price)
        
        if reference_price <= 0:
            return 0.20  # Default 20% width
            
        # Assume Tier 2 for MLGO
        lower_band, upper_band = self.luld_calc.calculate_luld_bands(reference_price, tier=2)
        
        # Calculate width as percentage
        width = (upper_band - lower_band) / reference_price
        
        return min(0.40, width)  # Cap at 40%
    
    def get_default_value(self) -> float:
        """Default to 20% width."""
        return 0.20
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 0.4 (40%) maps to 1."""
        return {'min': 0.0, 'max': 0.40}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires price data."""
        return {
            'current': {
                'fields': ['current_price', 'prev_day_close']
            }
        }