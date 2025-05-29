"""Candle pattern features"""
import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig


class PositionInCurrentCandle1mFeature(BaseFeature):
    """Position of current price within current 1m candle"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="position_in_current_candle_1m", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position in current candle"""
        current_price = market_data.get('current_price')
        bars = market_data.get('1m_bars_window', [])
        
        if not bars or current_price is None:
            return 0.5
        
        # Get the current (last) bar
        current_bar = None
        for bar in reversed(bars):
            if bar.get('is_current'):
                current_bar = bar
                break
        
        if not current_bar:
            current_bar = bars[-1]
        
        high = current_bar.get('high')
        low = current_bar.get('low')
        
        if high is None or low is None:
            return 0.5
        
        # Handle no range
        if high == low:
            return 0.5
        
        # Calculate position and clamp to [0, 1]
        position = (current_price - low) / (high - low)
        return np.clip(position, 0.0, 1.0)
    
    def get_default_value(self) -> float:
        return 0.5
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [0, 1]"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 1,
            "fields": ["1m_bars_window", "current_price"]
        }


class PositionInCurrentCandle5mFeature(BaseFeature):
    """Position of current price within current 5m candle"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="5m_position_in_current_candle", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position in current candle"""
        current_price = market_data.get('current_price')
        bars = market_data.get('5m_bars_window', [])
        
        if not bars or current_price is None:
            return 0.5
        
        # Get the current (last) bar
        current_bar = None
        for bar in reversed(bars):
            if bar.get('is_current'):
                current_bar = bar
                break
        
        if not current_bar:
            current_bar = bars[-1]
        
        high = current_bar.get('high')
        low = current_bar.get('low')
        
        if high is None or low is None:
            return 0.5
        
        # Handle no range
        if high == low:
            return 0.5
        
        # Calculate position and clamp to [0, 1]
        position = (current_price - low) / (high - low)
        return np.clip(position, 0.0, 1.0)
    
    def get_default_value(self) -> float:
        return 0.5
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [0, 1]"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 1,
            "fields": ["5m_bars_window", "current_price"]
        }


class BodySizeRelative1mFeature(BaseFeature):
    """Relative body size of 1m candles"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="1m_body_size_relative", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative body size"""
        bars = market_data.get('1m_bars_window', [])
        
        if not bars:
            return 0.0
        
        last_bar = bars[-1]
        open_price = last_bar.get('open')
        close_price = last_bar.get('close')
        high = last_bar.get('high')
        low = last_bar.get('low')
        
        if None in [open_price, close_price, high, low]:
            return 0.0
        
        # Handle no range
        if high == low:
            return 0.0
        
        # Calculate body size relative to range
        body_size = abs(close_price - open_price)
        candle_range = high - low
        
        relative_size = body_size / candle_range
        return np.clip(relative_size, 0.0, 1.0)
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [0, 1]"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 1,
            "fields": ["1m_bars_window"]
        }


class BodySizeRelative5mFeature(BaseFeature):
    """Relative body size of 5m candles"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="5m_body_size_relative", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate relative body size"""
        bars = market_data.get('5m_bars_window', [])
        
        if not bars:
            return 0.0
        
        last_bar = bars[-1]
        open_price = last_bar.get('open')
        close_price = last_bar.get('close')
        high = last_bar.get('high')
        low = last_bar.get('low')
        
        if None in [open_price, close_price, high, low]:
            return 0.0
        
        # Handle no range
        if high == low:
            return 0.0
        
        # Calculate body size relative to range
        body_size = abs(close_price - open_price)
        candle_range = high - low
        
        relative_size = body_size / candle_range
        return np.clip(relative_size, 0.0, 1.0)
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Already normalized to [0, 1]"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 1,
            "fields": ["5m_bars_window"]
        }