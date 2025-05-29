"""Medium-frequency velocity features"""
import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig


class PriceVelocity1mFeature(BaseFeature):
    """1-minute price velocity"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="price_velocity_1m", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate price velocity over 1 minute"""
        bars = market_data.get('1m_bars_window', [])
        
        if len(bars) < 2:
            return 0.0
        
        current_close = bars[-1].get('close')
        prev_close = bars[-2].get('close')
        
        if current_close is None or prev_close is None or prev_close == 0:
            return 0.0
        
        velocity = (current_close - prev_close) / prev_close
        return velocity
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±5% per minute"""
        return {
            "min": -0.05,
            "max": 0.05
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 2,
            "fields": ["1m_bars_window"]
        }


class PriceVelocity5mFeature(BaseFeature):
    """5-minute price velocity"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="5m_price_velocity", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate price velocity over 5 minutes"""
        bars = market_data.get('5m_bars_window', [])
        
        if len(bars) < 2:
            return 0.0
        
        current_close = bars[-1].get('close')
        prev_close = bars[-2].get('close')
        
        if current_close is None or prev_close is None or prev_close == 0:
            return 0.0
        
        velocity = (current_close - prev_close) / prev_close
        return velocity
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±10% per 5 minutes"""
        return {
            "min": -0.10,
            "max": 0.10
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 2,
            "fields": ["5m_bars_window"]
        }


class VolumeVelocity1mFeature(BaseFeature):
    """1-minute volume velocity"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="1m_volume_velocity", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume velocity over 1 minute"""
        bars = market_data.get('1m_bars_window', [])
        
        if len(bars) < 2:
            return 0.0
        
        current_vol = bars[-1].get('volume', 0)
        prev_vol = bars[-2].get('volume', 0)
        
        if prev_vol == 0:
            return 0.0
        
        velocity = (current_vol - prev_vol) / prev_vol
        return velocity
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±200% volume change"""
        return {
            "min": -2.0,
            "max": 2.0
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "1m_bars",
            "lookback": 2,
            "fields": ["1m_bars_window"]
        }


class VolumeVelocity5mFeature(BaseFeature):
    """5-minute volume velocity"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="5m_volume_velocity", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume velocity over 5 minutes"""
        bars = market_data.get('5m_bars_window', [])
        
        if len(bars) < 2:
            return 0.0
        
        current_vol = bars[-1].get('volume', 0)
        prev_vol = bars[-2].get('volume', 0)
        
        if prev_vol == 0:
            return 0.0
        
        velocity = (current_vol - prev_vol) / prev_vol
        return velocity
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±200% volume change"""
        return {
            "min": -2.0,
            "max": 2.0
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "5m_bars",
            "lookback": 2,
            "fields": ["5m_bars_window"]
        }