"""Medium-frequency acceleration features"""
from typing import Dict, Any, Optional
import numpy as np
from feature.feature_base import BaseFeature, FeatureConfig
from feature.normalizers import MinMaxNormalizer
from feature.feature_registry import feature_registry


@feature_registry.register("1m_price_acceleration", category="mf")
class PriceAcceleration1mFeature(BaseFeature):
    """1-minute price acceleration - change in price velocity"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="1m_price_acceleration",
            normalize=True
        ))
        # Normalize to [-1, 1] for reasonable acceleration ranges
        self.normalizer = MinMaxNormalizer(min_val=-0.02, max_val=0.02)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate 1-minute price acceleration"""
        bars_1m = market_data.get('1m_bars_window', [])
        
        if len(bars_1m) < 3:
            return 0.0
        
        # Get last 3 closes
        closes = []
        for i in range(-3, 0):
            bar = bars_1m[i]
            close = bar.get('close', 0.0)
            closes.append(float(close))
        
        # Avoid division by zero
        if closes[0] == 0 or closes[1] == 0:
            return 0.0
        
        # Calculate velocities
        velocity1 = (closes[1] - closes[0]) / closes[0]
        velocity2 = (closes[2] - closes[1]) / closes[1]
        
        # Acceleration is change in velocity
        acceleration = velocity2 - velocity1
        return acceleration
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {
            "min": -0.02,  # -2% acceleration per minute
            "max": 0.02    # +2% acceleration per minute
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 3,
            "fields": ["1m_bars_window", "close"]
        }


@feature_registry.register("5m_price_acceleration", category="mf")
class PriceAcceleration5mFeature(BaseFeature):
    """5-minute price acceleration - change in price velocity"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="5m_price_acceleration",
            normalize=True
        ))
        self.normalizer = MinMaxNormalizer(min_val=-0.05, max_val=0.05)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate 5-minute price acceleration"""
        bars_5m = market_data.get('5m_bars_window', [])
        
        if len(bars_5m) < 3:
            return 0.0
        
        # Get last 3 closes
        closes = []
        for i in range(-3, 0):
            bar = bars_5m[i]
            close = bar.get('close', 0.0)
            closes.append(float(close))
        
        # Avoid division by zero
        if closes[0] == 0 or closes[1] == 0:
            return 0.0
        
        # Calculate velocities
        velocity1 = (closes[1] - closes[0]) / closes[0]
        velocity2 = (closes[2] - closes[1]) / closes[1]
        
        # Acceleration is change in velocity
        acceleration = velocity2 - velocity1
        return acceleration
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {
            "min": -0.05,  # -5% acceleration per 5 minutes
            "max": 0.05    # +5% acceleration per 5 minutes
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 3,
            "fields": ["5m_bars_window", "close"]
        }


@feature_registry.register("1m_volume_acceleration", category="mf")
class VolumeAcceleration1mFeature(BaseFeature):
    """1-minute volume acceleration - change in volume velocity"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="1m_volume_acceleration",
            normalize=True
        ))
        self.normalizer = MinMaxNormalizer(min_val=-5.0, max_val=5.0)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate 1-minute volume acceleration"""
        bars_1m = market_data.get('1m_bars_window', [])
        
        if len(bars_1m) < 3:
            return 0.0
        
        # Get last 3 volumes
        volumes = []
        for i in range(-3, 0):
            bar = bars_1m[i]
            volume = float(bar.get('volume', 0.0))
            volumes.append(max(0.0, volume))  # Ensure non-negative
        
        # Calculate velocities
        velocity1 = 0.0
        velocity2 = 0.0
        
        if volumes[0] > 0:
            velocity1 = (volumes[1] - volumes[0]) / volumes[0]
        elif volumes[1] > 0:
            velocity1 = 1.0
            
        if volumes[1] > 0:
            velocity2 = (volumes[2] - volumes[1]) / volumes[1]
        elif volumes[2] > 0:
            velocity2 = 1.0
        
        # Acceleration is change in velocity
        acceleration = velocity2 - velocity1
        return acceleration
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {
            "min": -5.0,  # -500% acceleration
            "max": 5.0    # +500% acceleration
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 3,
            "fields": ["1m_bars_window", "volume"]
        }


@feature_registry.register("5m_volume_acceleration", category="mf")
class VolumeAcceleration5mFeature(BaseFeature):
    """5-minute volume acceleration - change in volume velocity"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="5m_volume_acceleration",
            normalize=True
        ))
        self.normalizer = MinMaxNormalizer(min_val=-5.0, max_val=5.0)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate 5-minute volume acceleration"""
        bars_5m = market_data.get('5m_bars_window', [])
        
        if len(bars_5m) < 3:
            return 0.0
        
        # Get last 3 volumes
        volumes = []
        for i in range(-3, 0):
            bar = bars_5m[i]
            volume = float(bar.get('volume', 0.0))
            volumes.append(max(0.0, volume))  # Ensure non-negative
        
        # Calculate velocities
        velocity1 = 0.0
        velocity2 = 0.0
        
        if volumes[0] > 0:
            velocity1 = (volumes[1] - volumes[0]) / volumes[0]
        elif volumes[1] > 0:
            velocity1 = 1.0
            
        if volumes[1] > 0:
            velocity2 = (volumes[2] - volumes[1]) / volumes[1]
        elif volumes[2] > 0:
            velocity2 = 1.0
        
        # Acceleration is change in velocity
        acceleration = velocity2 - velocity1
        return acceleration
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {
            "min": -5.0,  # -500% acceleration
            "max": 5.0    # +500% acceleration
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 3,
            "fields": ["5m_bars_window", "volume"]
        }