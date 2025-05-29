"""Time-based features for low-frequency analysis."""

import numpy as np
from typing import Dict, Any
from datetime import time
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


@feature_registry.register("time_of_day_sin", category="lf")
class TimeOfDaySinFeature(BaseFeature):
    """Sine encoding of time of day for cyclical representation.
    
    Maps trading hours to [-1, 1] using sine function.
    Useful for capturing intraday patterns.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="time_of_day_sin", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate sine encoding of current time."""
        timestamp = market_data.get('timestamp')
        if timestamp is None:
            return 0.0
            
        # Extract time of day
        current_time = timestamp.time()
        
        # Convert to minutes since market open (9:30 AM)
        market_open = time(9, 30)
        minutes_since_open = (current_time.hour * 60 + current_time.minute) - (9 * 60 + 30)
        
        # Trading day is 6.5 hours = 390 minutes
        trading_minutes = 390
        
        # Map to [0, 2π] for full cycle
        angle = (minutes_since_open / trading_minutes) * 2 * np.pi
        
        return float(np.sin(angle))
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        # Already in [-1, 1] range
        return {'min': -1.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'current': {'fields': ['timestamp']}}


@feature_registry.register("time_of_day_cos", category="lf")
class TimeOfDayCosFeature(BaseFeature):
    """Cosine encoding of time of day for cyclical representation.
    
    Maps trading hours to [-1, 1] using cosine function.
    Combined with sine encoding for complete cyclical representation.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="time_of_day_cos", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate cosine encoding of current time."""
        timestamp = market_data.get('timestamp')
        if timestamp is None:
            return 1.0  # Cosine of 0
            
        # Extract time of day
        current_time = timestamp.time()
        
        # Convert to minutes since market open (9:30 AM)
        market_open = time(9, 30)
        minutes_since_open = (current_time.hour * 60 + current_time.minute) - (9 * 60 + 30)
        
        # Trading day is 6.5 hours = 390 minutes
        trading_minutes = 390
        
        # Map to [0, 2π] for full cycle
        angle = (minutes_since_open / trading_minutes) * 2 * np.pi
        
        return float(np.cos(angle))
    
    def get_default_value(self) -> float:
        return 1.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        # Already in [-1, 1] range
        return {'min': -1.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'current': {'fields': ['timestamp']}}


@feature_registry.register("market_session_type", category="lf")
class MarketSessionTypeFeature(BaseFeature):
    """Market session type indicator.
    
    Categorizes current time into trading sessions:
    - Pre-market: -1.0
    - Morning (9:30-11:30): 0.5 (prime momentum time)
    - Midday (11:30-2:00): 0.0 (lunch chop)
    - Afternoon (2:00-3:30): 0.3 (afternoon momentum)
    - Power hour (3:30-4:00): 0.8 (closing volatility)
    - After-hours: -0.5
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="market_session_type", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Determine market session type."""
        timestamp = market_data.get('timestamp')
        if timestamp is None:
            return 0.0
            
        current_time = timestamp.time()
        hour = current_time.hour
        minute = current_time.minute
        
        # Convert to decimal hours
        decimal_hour = hour + minute / 60.0
        
        # Pre-market (4:00 AM - 9:30 AM)
        if decimal_hour < 9.5:
            return -1.0
        
        # Morning session (9:30 AM - 11:30 AM) - Prime momentum time
        elif decimal_hour < 11.5:
            return 0.5
        
        # Midday session (11:30 AM - 2:00 PM) - Lunch chop
        elif decimal_hour < 14.0:
            return 0.0
        
        # Afternoon session (2:00 PM - 3:30 PM)
        elif decimal_hour < 15.5:
            return 0.3
        
        # Power hour (3:30 PM - 4:00 PM)
        elif decimal_hour < 16.0:
            return 0.8
        
        # After-hours (4:00 PM - 8:00 PM)
        else:
            return -0.5
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'current': {'fields': ['timestamp']}}


