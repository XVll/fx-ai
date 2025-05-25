"""EMA-based medium-frequency features"""
import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


def calculate_ema(values: List[float], period: int) -> float:
    """Calculate EMA for a list of values"""
    if not values or period <= 0:
        return 0.0
    
    # If we have fewer values than period, use what we have
    if len(values) < period:
        period = len(values)
    
    # Use last 'period' values
    values = values[-period:]
    
    if not values:
        return 0.0
    
    # Simple EMA calculation
    multiplier = 2 / (period + 1)
    ema = values[0]
    
    for value in values[1:]:
        ema = (value * multiplier) + (ema * (1 - multiplier))
    
    return ema


@feature_registry.register("1m_ema9_distance", category="mf")
class DistanceToEMA9_1mFeature(BaseFeature):
    """Distance to 9-period EMA on 1-minute bars"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="distance_to_ema9_1m", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to EMA9"""
        current_price = market_data.get('current_price', 0)
        bars = market_data.get('1m_bars_window', [])
        
        if not bars or current_price == 0:
            return 0.0
        
        # Extract close prices
        closes = [bar.get('close', 0) for bar in bars if bar.get('close')]
        
        if len(closes) < 1:
            return 0.0
        
        # Calculate EMA
        ema9 = calculate_ema(closes, 9)
        
        if ema9 == 0:
            return 0.0
        
        # Calculate distance as percentage
        distance = (current_price - ema9) / ema9
        return distance
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±50% distance"""
        return {
            "min": -0.5,
            "max": 0.5
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 9,
            "fields": ["1m_bars_window", "current_price"]
        }


@feature_registry.register("1m_ema20_distance", category="mf")
class DistanceToEMA20_1mFeature(BaseFeature):
    """Distance to 20-period EMA on 1-minute bars"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="1m_ema20_distance", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to EMA20"""
        current_price = market_data.get('current_price', 0)
        bars = market_data.get('1m_bars_window', [])
        
        if not bars or current_price == 0:
            return 0.0
        
        # Extract close prices
        closes = [bar.get('close', 0) for bar in bars if bar.get('close')]
        
        if len(closes) < 1:
            return 0.0
        
        # Calculate EMA
        ema20 = calculate_ema(closes, 20)
        
        if ema20 == 0:
            return 0.0
        
        # Calculate distance as percentage
        distance = (current_price - ema20) / ema20
        return distance
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±50% distance"""
        return {
            "min": -0.5,
            "max": 0.5
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 20,
            "fields": ["1m_bars_window", "current_price"]
        }


@feature_registry.register("5m_ema9_distance", category="mf")
class DistanceToEMA9_5mFeature(BaseFeature):
    """Distance to 9-period EMA on 5-minute bars"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="5m_ema9_distance", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to EMA9"""
        current_price = market_data.get('current_price', 0)
        bars = market_data.get('5m_bars_window', [])
        
        if not bars or current_price == 0:
            return 0.0
        
        # Extract close prices
        closes = [bar.get('close', 0) for bar in bars if bar.get('close')]
        
        if len(closes) < 1:
            return 0.0
        
        # Calculate EMA
        ema9 = calculate_ema(closes, 9)
        
        if ema9 == 0:
            return 0.0
        
        # Calculate distance as percentage
        distance = (current_price - ema9) / ema9
        return distance
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±50% distance"""
        return {
            "min": -0.5,
            "max": 0.5
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 9,
            "fields": ["5m_bars_window", "current_price"]
        }


@feature_registry.register("5m_ema20_distance", category="mf")
class DistanceToEMA20_5mFeature(BaseFeature):
    """Distance to 20-period EMA on 5-minute bars"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="5m_ema20_distance", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate distance to EMA20"""
        current_price = market_data.get('current_price', 0)
        bars = market_data.get('5m_bars_window', [])
        
        if not bars or current_price == 0:
            return 0.0
        
        # Extract close prices
        closes = [bar.get('close', 0) for bar in bars if bar.get('close')]
        
        if len(closes) < 1:
            return 0.0
        
        # Calculate EMA
        ema20 = calculate_ema(closes, 20)
        
        if ema20 == 0:
            return 0.0
        
        # Calculate distance as percentage
        distance = (current_price - ema20) / ema20
        return distance
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±50% distance"""
        return {
            "min": -0.5,
            "max": 0.5
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 20,
            "fields": ["5m_bars_window", "current_price"]
        }