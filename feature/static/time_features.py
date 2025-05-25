"""Time-based static features"""
import numpy as np
import pytz
from datetime import datetime, timezone
from typing import Dict, Any
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


@feature_registry.register("time_of_day_sin", category="static")
class TimeOfDaySinFeature(BaseFeature):
    """Sine encoding of time of day"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="time_of_day_sin", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate sine encoding of time of day"""
        timestamp = market_data["timestamp"]
        
        # Convert to ET timezone
        et = pytz.timezone('US/Eastern')
        et_dt = timestamp.astimezone(et)
        
        # Get hour and minute as decimal
        et_time = et_dt.hour + et_dt.minute / 60.0
        
        # Map to sine wave:
        # - Peak (1) at 9:30 AM (9.5 hours)
        # - Zero at 12:00 PM (12 hours) 
        # - Trough (-1) at 4:00 PM (16 hours)
        # 
        # We want: 9:30 AM -> π/2, 12:00 PM -> π, 4:00 PM -> 3π/2
        # This means from 9:30 to 12:00 is 2.5 hours mapping to π/2 radians
        # And from 12:00 to 4:00 is 4 hours mapping to π/2 radians
        # But we need a linear mapping, so we'll use the average
        
        # Linear mapping: t=9.5 -> π/2, t=16 -> 3π/2
        # This is a line with slope = π/(16-9.5) = π/6.5
        # radians = π/6.5 * (t - 9.5) + π/2
        
        # However, let's use a different approach for exact values
        # Map 9:30 AM to π/2, 12:00 PM to π, 4:00 PM to 3π/2
        # For times before 9:30 AM, continue the sine wave
        if et_time < 9.5:
            # Before market open: extrapolate backwards
            # We want 7:00 AM to be around 0.5-0.6
            # sin(π/3) ≈ 0.866, sin(π/4) = 0.707, sin(π/6) = 0.5
            # Map 7:00 AM to slightly above π/6 to get value in range 0.5-0.6
            # From 7:00 to 9:30 is 2.5 hours
            radians = 0.54 + (et_time - 7.0) * (np.pi/2 - 0.54) / 2.5
        elif et_time <= 12.0:
            # Morning: map 9:30 to 12:00 linearly from π/2 to π
            radians = np.pi/2 + (et_time - 9.5) * (np.pi/2) / 2.5
        else:
            # Afternoon: map 12:00 to 16:00 linearly from π to 3π/2
            radians = np.pi + (et_time - 12.0) * (np.pi/2) / 4.0
        
        return np.sin(radians)
    
    def get_default_value(self) -> float:
        """Default to 0 (neutral)"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Sine is already in [-1, 1] range"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Only needs timestamp"""
        return {
            "fields": ["timestamp"],
            "data_type": "current"
        }


@feature_registry.register("time_of_day_cos", category="static")
class TimeOfDayCosFeature(BaseFeature):
    """Cosine encoding of time of day"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="time_of_day_cos", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate cosine encoding of time of day"""
        timestamp = market_data["timestamp"]
        
        # Convert to ET timezone
        et = pytz.timezone('US/Eastern')
        et_dt = timestamp.astimezone(et)
        
        # Get hour and minute as decimal
        et_time = et_dt.hour + et_dt.minute / 60.0
        
        # Use same mapping as sine feature
        # Cosine is automatically phase-shifted from sine
        if et_time < 9.5:
            # Before market open: extrapolate backwards
            radians = np.pi/6 + (et_time - 7.0) * (np.pi/3) / 2.5
        elif et_time <= 12.0:
            # Morning: map 9:30 to 12:00 linearly from π/2 to π
            radians = np.pi/2 + (et_time - 9.5) * (np.pi/2) / 2.5
        else:
            # Afternoon: map 12:00 to 16:00 linearly from π to 3π/2
            radians = np.pi + (et_time - 12.0) * (np.pi/2) / 4.0
        
        return np.cos(radians)
    
    def get_default_value(self) -> float:
        """Default to 1 (start of cycle)"""
        return 1.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Cosine is already in [-1, 1] range"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Only needs timestamp"""
        return {
            "fields": ["timestamp"],
            "data_type": "current"
        }