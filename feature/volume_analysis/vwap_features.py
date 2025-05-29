"""Volume Weighted Average Price (VWAP) features for support/resistance."""

import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


@feature_registry.register("distance_to_vwap", category="mf")
class DistanceToVWAPFeature(BaseFeature):
    """Distance from current price to VWAP as percentage.
    
    VWAP often acts as support/resistance in momentum trading.
    Positive values mean price is above VWAP, negative below.
    Normalized to [-1, 1] range.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="distance_to_vwap", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to VWAP."""
        current_price = market_data.get('current_price', 0)
        vwap = market_data.get('session_vwap', current_price)
        
        if current_price <= 0 or vwap <= 0:
            return 0.0
            
        # Calculate percentage distance
        distance = (current_price - vwap) / vwap
        
        # Clamp to reasonable range
        return max(-0.10, min(0.10, distance))  # ±10% max
    
    def get_default_value(self) -> float:
        """Default to at VWAP."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] symmetric range."""
        return {
            'min': -0.10,
            'max': 0.10,
            'range_type': 'symmetric'
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires session VWAP."""
        return {
            'current': {
                'fields': ['current_price', 'session_vwap']
            }
        }


@feature_registry.register("vwap_slope", category="mf")
class VWAPSlopeFeature(BaseFeature):
    """Rate of change of VWAP over recent period.
    
    Positive slope indicates buying pressure, negative indicates selling.
    Useful for identifying trend strength in momentum moves.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="vwap_slope", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate VWAP slope."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 5:
            return 0.0
            
        # Get VWAP values from recent bars
        vwap_values = []
        for bar in bars[-10:]:  # Last 10 minutes
            vwap = bar.get('vwap')
            if vwap and vwap > 0:
                vwap_values.append(vwap)
                
        if len(vwap_values) < 2:
            return 0.0
            
        # Calculate linear regression slope
        x = np.arange(len(vwap_values))
        slope = np.polyfit(x, vwap_values, 1)[0]
        
        # Normalize by average VWAP
        avg_vwap = np.mean(vwap_values)
        if avg_vwap > 0:
            normalized_slope = slope / avg_vwap
            # Clamp to reasonable range
            return max(-0.01, min(0.01, normalized_slope))  # ±1% per minute max
        
        return 0.0
    
    def get_default_value(self) -> float:
        """Default to flat slope."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] symmetric range."""
        return {
            'min': -0.01,
            'max': 0.01,
            'range_type': 'symmetric'
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires 1m bars with VWAP."""
        return {
            'mf_bars_1m': {
                'lookback': 10,
                'fields': ['vwap']
            }
        }


@feature_registry.register("price_vwap_divergence", category="mf")
class PriceVWAPDivergenceFeature(BaseFeature):
    """Measures divergence between price trend and VWAP trend.
    
    Large divergence can indicate unsustainable moves or reversal potential.
    Positive means price rising faster than VWAP (bullish momentum).
    Negative means price falling faster than VWAP (bearish momentum).
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="price_vwap_divergence", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate price-VWAP divergence."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 5:
            return 0.0
            
        # Get price and VWAP series
        prices = []
        vwaps = []
        
        for bar in bars[-10:]:  # Last 10 minutes
            close = bar.get('close', 0)
            vwap = bar.get('vwap', close)
            
            if close > 0 and vwap > 0:
                prices.append(close)
                vwaps.append(vwap)
                
        if len(prices) < 2:
            return 0.0
            
        # Calculate percentage changes
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        vwap_change = (vwaps[-1] - vwaps[0]) / vwaps[0] if vwaps[0] > 0 else 0
        
        # Divergence is the difference
        divergence = price_change - vwap_change
        
        # Clamp to reasonable range
        return max(-0.05, min(0.05, divergence))  # ±5% max divergence
    
    def get_default_value(self) -> float:
        """Default to no divergence."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] symmetric range."""
        return {
            'min': -0.05,
            'max': 0.05,
            'range_type': 'symmetric'
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires 1m bars with close and VWAP."""
        return {
            'mf_bars_1m': {
                'lookback': 10,
                'fields': ['close', 'vwap']
            }
        }