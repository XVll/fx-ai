"""High-frequency volume features"""
from typing import Dict, Any, Optional
import numpy as np
from feature.feature_base import BaseFeature, FeatureConfig
from feature.normalizers import MinMaxNormalizer
from feature.feature_registry import feature_registry


@feature_registry.register("hf_volume_velocity", category="hf")
class VolumeVelocityFeature(BaseFeature):
    """1-second volume velocity - rate of volume change"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="hf_volume_velocity",
            normalize=True
        ))
        # Normalize to [-1, 1] for ±1000% volume change per second
        self.normalizer = MinMaxNormalizer(min_val=-10.0, max_val=10.0)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate raw 1-second volume velocity"""
        hf_window = market_data.get('hf_data_window', [])
        
        if len(hf_window) < 2:
            return 0.0
        
        # Get current and previous volumes
        current_bar = hf_window[-1].get('1s_bar', {})
        previous_bar = hf_window[-2].get('1s_bar', {})
        
        current_volume = float(current_bar.get('volume', 0.0))
        previous_volume = float(previous_bar.get('volume', 0.0))
        
        # Handle edge cases
        if current_volume < 0:
            current_volume = 0.0
        if previous_volume < 0:
            previous_volume = 0.0
        
        # Calculate velocity
        if previous_volume == 0:
            if current_volume > 0:
                return 1.0  # Max increase from zero
            return 0.0
        
        velocity = (current_volume - previous_volume) / previous_volume
        return velocity
    
    def get_default_value(self) -> float:
        """Default to no change"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalization parameters"""
        return {
            "min": -10.0,  # -1000% per second
            "max": 10.0    # +1000% per second
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1s_bars",
            "lookback": 2,
            "fields": ["hf_data_window", "volume"]
        }


@feature_registry.register("hf_volume_acceleration", category="hf")
class VolumeAccelerationFeature(BaseFeature):
    """1-second volume acceleration - change in volume velocity"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="hf_volume_acceleration",
            normalize=True
        ))
        # Normalize acceleration 
        self.normalizer = MinMaxNormalizer(min_val=-5.0, max_val=5.0)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate raw 1-second volume acceleration"""
        hf_window = market_data.get('hf_data_window', [])
        
        if len(hf_window) < 3:
            return 0.0
        
        # Get volumes for velocity calculation
        volumes = []
        for i in range(-3, 0):
            bar = hf_window[i].get('1s_bar', {})
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
        """Default to no acceleration"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalization parameters"""
        return {
            "min": -5.0,  # -500% acceleration
            "max": 5.0    # +500% acceleration
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1s_bars",
            "lookback": 3,
            "fields": ["hf_data_window", "volume"]
        }