"""Relative volume features for detecting unusual activity."""

import numpy as np
from typing import Dict, Any, List
from datetime import time
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


@feature_registry.register("relative_volume", category="mf")
class RelativeVolumeFeature(BaseFeature):
    """Current volume compared to average volume over recent period.
    
    Values > 1 indicate above average volume (potential breakout).
    Values < 1 indicate below average volume (potential consolidation).
    Critical for squeeze breakout detection.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="relative_volume", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative volume."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 5:
            return 1.0  # Default to average
            
        # Get recent and historical volumes
        recent_volumes = [bar.get('volume', 0) for bar in bars[-5:]]  # Last 5 minutes
        historical_volumes = [bar.get('volume', 0) for bar in bars[-30:]]  # Last 30 minutes
        
        recent_avg = np.mean(recent_volumes) if recent_volumes else 0
        historical_avg = np.mean(historical_volumes) if historical_volumes else 1
        
        if historical_avg <= 0:
            return 1.0
            
        # Calculate ratio
        rel_volume = recent_avg / historical_avg
        
        # Clamp to reasonable range
        return min(5.0, rel_volume)  # Cap at 5x average
    
    def get_default_value(self) -> float:
        """Default to average volume."""
        return 1.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 5x maps to 1."""
        return {'min': 0.0, 'max': 5.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires 1m volume data."""
        return {
            'mf_bars_1m': {
                'lookback': 30,
                'fields': ['volume']
            }
        }


@feature_registry.register("volume_surge", category="mf")
class VolumeSurgeFeature(BaseFeature):
    """Detects sudden volume spikes compared to recent average.
    
    Specifically looks for volume surges that often precede breakouts.
    More sensitive than relative_volume for short-term spikes.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="volume_surge", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume surge."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 2:
            return 0.0
            
        # Current volume vs recent average
        current_volume = bars[-1].get('volume', 0)
        recent_volumes = [bar.get('volume', 0) for bar in bars[-10:-1]]  # Previous 9 bars
        
        if not recent_volumes:
            return 0.0
            
        avg_volume = np.mean(recent_volumes)
        
        if avg_volume <= 0:
            return 0.0
            
        # Calculate surge ratio
        surge = (current_volume / avg_volume) - 1.0  # 0 = average, 1 = 2x average
        
        # Clamp to reasonable range
        return max(0.0, min(4.0, surge))  # 0 to 5x surge
    
    def get_default_value(self) -> float:
        """Default to no surge."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [0, 1] where 4 (5x volume) maps to 1."""
        return {'min': 0.0, 'max': 4.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires 1m volume data."""
        return {
            'mf_bars_1m': {
                'lookback': 10,
                'fields': ['volume']
            }
        }


@feature_registry.register("cumulative_volume_delta", category="mf")
class CumulativeVolumeDeltaFeature(BaseFeature):
    """Cumulative difference between buy and sell volume.
    
    Positive values indicate net buying pressure.
    Negative values indicate net selling pressure.
    Key indicator for momentum direction.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="cumulative_volume_delta", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate cumulative volume delta."""
        trades = market_data.get('hf_data_window', [])
        
        if not trades:
            return 0.0
            
        buy_volume = 0
        sell_volume = 0
        
        # Aggregate buy/sell volume from trades
        for window_data in trades[-30:]:  # Last 30 seconds
            window_trades = window_data.get('trades', [])
            
            for trade in window_trades:
                size = trade.get('size', 0)
                
                # Simple classification: trades at ask are buys, at bid are sells
                # This is simplified - real implementation would use trade conditions
                if self._is_buy_trade(trade, window_data):
                    buy_volume += size
                else:
                    sell_volume += size
                    
        # Calculate delta
        total_volume = buy_volume + sell_volume
        if total_volume <= 0:
            return 0.0
            
        delta = (buy_volume - sell_volume) / total_volume  # Normalized to [-1, 1]
        
        return delta
    
    def _is_buy_trade(self, trade: Dict[str, Any], window_data: Dict[str, Any]) -> bool:
        """Classify trade as buy or sell (simplified)."""
        # In real implementation, would use trade conditions
        # For now, use price vs quotes
        price = trade.get('price', 0)
        quotes = window_data.get('quotes', [])
        
        if quotes:
            last_quote = quotes[-1]
            mid = (last_quote.get('bid_price', price) + last_quote.get('ask_price', price)) / 2
            return price >= mid
            
        return True  # Default to buy
    
    def get_default_value(self) -> float:
        """Default to neutral."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [-1, 1]."""
        return {
            'min': -1.0,
            'max': 1.0,
            'range_type': 'symmetric'
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires HF trade data."""
        return {
            'hf_data_window': {
                'lookback': 30,
                'fields': ['trades', 'quotes']
            }
        }


@feature_registry.register("volume_momentum", category="mf")
class VolumeMomentumFeature(BaseFeature):
    """Rate of change in volume over time.
    
    Positive values indicate accelerating volume.
    Useful for detecting the start of breakouts.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="volume_momentum", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume momentum."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 3:
            return 0.0
            
        # Get volume series
        volumes = [bar.get('volume', 0) for bar in bars[-5:]]
        
        if len(volumes) < 2:
            return 0.0
            
        # Calculate rate of change
        recent_change = (volumes[-1] - volumes[-2]) / max(1, volumes[-2]) if len(volumes) >= 2 else 0
        older_change = (volumes[-2] - volumes[-3]) / max(1, volumes[-3]) if len(volumes) >= 3 else 0
        
        # Momentum is acceleration of volume
        momentum = recent_change - older_change
        
        # Clamp to reasonable range
        return max(-1.0, min(1.0, momentum))
    
    def get_default_value(self) -> float:
        """Default to no momentum."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] symmetric range."""
        return {
            'min': -1.0,
            'max': 1.0,
            'range_type': 'symmetric'
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires 1m volume data."""
        return {
            'mf_bars_1m': {
                'lookback': 5,
                'fields': ['volume']
            }
        }