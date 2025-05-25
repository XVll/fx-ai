"""High-frequency price-based features"""
import numpy as np
from typing import Dict, Any, List
from ..base.feature_base import BaseFeature, FeatureConfig
from ..base.feature_registry import feature_registry


@feature_registry.register("price_velocity", category="hf")
class PriceVelocityFeature(BaseFeature):
    """1-second price velocity"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="price_velocity", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate price velocity over 1 second"""
        hf_window = market_data.get('hf_data_window', [])
        
        if len(hf_window) < 2:
            return 0.0
        
        # Get current and previous prices
        current_bar = hf_window[-1].get('1s_bar')
        prev_bar = hf_window[-2].get('1s_bar')
        
        if not current_bar or not prev_bar:
            return 0.0
        
        current_price = current_bar.get('close')
        prev_price = prev_bar.get('close')
        
        if current_price is None or prev_price is None:
            return 0.0
        
        # Handle zero price
        if prev_price == 0:
            return 0.0
        
        # Calculate velocity as percentage change
        velocity = (current_price - prev_price) / prev_price
        
        return velocity
    
    def get_default_value(self) -> float:
        """Default to no movement"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±10% per second max"""
        return {
            "min": -0.1,  # -10% per second
            "max": 0.1    # +10% per second
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1s_bars",
            "lookback": 2,
            "fields": ["hf_data_window"]
        }


@feature_registry.register("price_acceleration", category="hf")
class PriceAccelerationFeature(BaseFeature):
    """1-second price acceleration (change in velocity)"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="price_acceleration", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate price acceleration over 1 second"""
        hf_window = market_data.get('hf_data_window', [])
        
        if len(hf_window) < 3:
            return 0.0
        
        # Get three consecutive prices
        prices = []
        for i in [-3, -2, -1]:
            bar = hf_window[i].get('1s_bar')
            if bar and 'close' in bar:
                prices.append(bar['close'])
            else:
                return 0.0
        
        # Avoid division by zero
        if prices[0] == 0 or prices[1] == 0:
            return 0.0
        
        # Calculate velocities
        velocity1 = (prices[1] - prices[0]) / prices[0]
        velocity2 = (prices[2] - prices[1]) / prices[1]
        
        # Acceleration is change in velocity
        acceleration = velocity2 - velocity1
        
        return acceleration
    
    def get_default_value(self) -> float:
        """Default to no acceleration"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for reasonable acceleration"""
        return {
            "min": -0.05,  # -5% change in velocity
            "max": 0.05    # +5% change in velocity
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1s_bars",
            "lookback": 3,
            "fields": ["hf_data_window"]
        }