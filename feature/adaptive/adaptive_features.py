"""Adaptive features that adjust to changing market conditions."""

import numpy as np
from typing import Dict, Any
from feature.feature_base import BaseFeature, FeatureConfig


class VolatilityAdjustedMomentumFeature(BaseFeature):
    """Price momentum normalized by current volatility regime.
    
    In high volatility: requires bigger moves to signal momentum
    In low volatility: smaller moves are more significant
    Truly adaptive feature that changes with market conditions.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="volatility_adjusted_momentum", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-adjusted momentum."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 20:
            return 0.0
            
        # Get price series
        prices = [bar.get('close', 0) for bar in bars[-20:]]
        
        if len(prices) < 10:
            return 0.0
            
        # Calculate recent momentum
        recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        
        # Calculate current volatility (rolling standard deviation)
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
                
        if len(price_changes) < 5:
            return 0.0
            
        current_volatility = np.std(price_changes[-10:]) if len(price_changes) >= 10 else np.std(price_changes)
        
        # Adjust momentum by volatility
        if current_volatility > 0:
            adjusted_momentum = recent_momentum / current_volatility
            return max(-3.0, min(3.0, adjusted_momentum))  # ±3 sigma max
            
        return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -3.0, 'max': 3.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'mf_bars_1m': {'lookback': 20, 'fields': ['close']}}


class RegimeRelativeVolumeFeature(BaseFeature):
    """Volume relative to current volatility regime.
    
    During quiet periods: normal volume is more significant
    During volatile periods: requires higher volume to signal
    Adapts to market regime automatically.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="regime_relative_volume", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate regime-adjusted relative volume."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 30:
            return 1.0  # Default to normal volume
            
        # Current volume
        current_volume = bars[-1].get('volume', 0)
        
        # Calculate volatility regime
        prices = [bar.get('close', 0) for bar in bars[-30:]]
        price_changes = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change = abs((prices[i] - prices[i-1]) / prices[i-1])
                price_changes.append(change)
                
        if len(price_changes) < 10:
            return 1.0
            
        # Classify regime
        recent_volatility = np.mean(price_changes[-10:])
        historical_volatility = np.mean(price_changes)
        
        if historical_volatility <= 0:
            return 1.0
            
        volatility_ratio = recent_volatility / historical_volatility
        
        # Get volumes
        volumes = [bar.get('volume', 0) for bar in bars[-30:]]
        
        # Adjust expected volume based on regime
        if volatility_ratio > 1.5:  # High volatility regime
            # Expect higher volume, so current volume is less significant
            expected_volume = np.mean(volumes) * 1.5
        elif volatility_ratio < 0.7:  # Low volatility regime
            # Expect lower volume, so current volume is more significant
            expected_volume = np.mean(volumes) * 0.7
        else:
            # Normal regime
            expected_volume = np.mean(volumes)
            
        if expected_volume > 0:
            relative_volume = current_volume / expected_volume
            return min(5.0, relative_volume)  # Cap at 5x
            
        return 1.0
    
    def get_default_value(self) -> float:
        return 1.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 5.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'mf_bars_1m': {'lookback': 30, 'fields': ['close', 'volume']}}


class AdaptiveSupportResistanceFeature(BaseFeature):
    """Support/resistance levels that adapt to recent price action.
    
    Instead of static historical levels, this calculates dynamic S/R
    based on recent volatility and price clustering.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="adaptive_support_resistance", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate adaptive support/resistance distance."""
        bars = market_data.get('lf_5m_window', [])
        current_price = market_data.get('current_price', 0)
        
        if len(bars) < 20 or current_price <= 0:
            return 0.0
            
        # Get recent highs and lows
        highs = [bar.get('high', 0) for bar in bars[-20:]]
        lows = [bar.get('low', 0) for bar in bars[-20:]]
        
        # Calculate current volatility
        price_changes = []
        for i in range(1, len(bars)):
            prev_close = bars[i-1].get('close', 0)
            curr_close = bars[i].get('close', 0)
            if prev_close > 0:
                change = abs((curr_close - prev_close) / prev_close)
                price_changes.append(change)
                
        if not price_changes:
            return 0.0
            
        volatility = np.mean(price_changes[-10:]) if len(price_changes) >= 10 else np.mean(price_changes)
        
        # Find adaptive support/resistance
        # Use volatility to determine significance threshold
        significance_threshold = current_price * volatility * 2  # 2x current volatility
        
        # Find resistance (high above current price)
        resistance_levels = [h for h in highs if h > current_price + significance_threshold]
        nearest_resistance = min(resistance_levels) if resistance_levels else max(highs)
        
        # Find support (low below current price)
        support_levels = [l for l in lows if l < current_price - significance_threshold]
        nearest_support = max(support_levels) if support_levels else min(lows)
        
        # Calculate relative position between adaptive S/R
        if nearest_resistance > nearest_support:
            position = (current_price - nearest_support) / (nearest_resistance - nearest_support)
            return max(0.0, min(1.0, position))
            
        return 0.5  # Neutral if no clear levels
    
    def get_default_value(self) -> float:
        return 0.5
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'lf_5m_window': {'lookback': 20, 'fields': ['high', 'low', 'close']}}