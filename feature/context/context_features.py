"""Context features that provide market environment information."""

import numpy as np
from typing import Dict, Any
from datetime import datetime
from feature.feature_base import BaseFeature, FeatureConfig


class SessionProgressFeature(BaseFeature):
    """How far through the trading session we are.
    
    0 = start of session, 1 = end of session
    More dynamic than time encoding because it adapts to session type.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="session_progress", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate session progress."""
        timestamp = market_data.get('timestamp')
        session = market_data.get('session', 'REGULAR')
        
        if not timestamp:
            return 0.0
            
        time_obj = timestamp.time()
        
        # Define session bounds (ET)
        if session == 'PREMARKET':
            start_hour, start_min = 4, 0
            end_hour, end_min = 9, 30
        elif session == 'REGULAR':
            start_hour, start_min = 9, 30
            end_hour, end_min = 16, 0
        elif session == 'POSTMARKET':
            start_hour, start_min = 16, 0
            end_hour, end_min = 20, 0
        else:
            return 0.0
            
        # Convert to minutes
        current_minutes = time_obj.hour * 60 + time_obj.minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        
        if current_minutes < start_minutes:
            return 0.0
        elif current_minutes > end_minutes:
            return 1.0
        else:
            progress = (current_minutes - start_minutes) / (end_minutes - start_minutes)
            return max(0.0, min(1.0, progress))
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'current': {'fields': ['timestamp', 'session']}}


class MarketStressFeature(BaseFeature):
    """Current market stress/volatility level.
    
    Combines multiple indicators to assess market stress:
    - VIX-like calculation from price movements
    - Halt frequency
    - Volume spikes across timeframes
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="market_stress_level", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate market stress level."""
        # Price volatility component
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 10:
            return 0.0
            
        # Calculate recent volatility
        price_changes = []
        for i in range(1, min(len(bars), 30)):
            prev_close = bars[i-1].get('close', 0)
            curr_close = bars[i].get('close', 0)
            if prev_close > 0:
                change = abs((curr_close - prev_close) / prev_close)
                price_changes.append(change)
                
        if not price_changes:
            return 0.0
            
        recent_volatility = np.mean(price_changes[-10:]) if len(price_changes) >= 10 else np.mean(price_changes)
        
        # Volume stress component
        volumes = [bar.get('volume', 0) for bar in bars[-10:]]
        avg_volume = np.mean(volumes) if volumes else 0
        historical_volumes = [bar.get('volume', 0) for bar in bars[:-10]]
        historical_avg = np.mean(historical_volumes) if historical_volumes else avg_volume
        
        volume_stress = 0.0
        if historical_avg > 0:
            volume_stress = min(2.0, avg_volume / historical_avg) / 2.0  # Normalize to [0,1]
        
        # Halt stress component
        is_halted = 1.0 if market_data.get('is_halted', False) else 0.0
        time_since_halt = market_data.get('time_since_halt', 3600)
        halt_stress = max(0.0, 1.0 - (time_since_halt / 3600))  # Decay over 1 hour
        
        # Combine components
        stress_level = (recent_volatility * 100 * 0.5) + (volume_stress * 0.3) + (halt_stress * 0.2)
        return min(1.0, stress_level)
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            'mf_bars_1m': {'lookback': 30, 'fields': ['close', 'volume']},
            'current': {'fields': ['is_halted', 'time_since_halt']}
        }


class SessionVolumeProfileFeature(BaseFeature):
    """Where we are relative to typical session volume pattern.
    
    Different sessions have different volume profiles.
    This captures how current volume compares to typical patterns.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="session_volume_profile", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate session volume profile position."""
        session_volume = market_data.get('session_volume', 0)
        session = market_data.get('session', 'REGULAR')
        timestamp = market_data.get('timestamp')
        
        if not timestamp or session_volume <= 0:
            return 0.5  # Neutral
            
        # Get session progress
        progress = SessionProgressFeature().calculate_raw(market_data)
        
        # Typical volume patterns (simplified)
        if session == 'PREMARKET':
            # Low volume, gradually increasing
            expected_ratio = 0.1 + (progress * 0.3)  # 10% to 40% of regular session
        elif session == 'REGULAR':
            # High volume at open/close, lower in middle
            if progress < 0.5:
                expected_ratio = 1.0 - (progress * 0.5)  # 100% to 50%
            else:
                expected_ratio = 0.5 + ((progress - 0.5) * 1.0)  # 50% to 100%
        elif session == 'POSTMARKET':
            # Decreasing volume
            expected_ratio = 0.3 * (1.0 - progress)  # 30% to 0%
        else:
            expected_ratio = 0.1
            
        # Compare to expected
        # Note: This is simplified - real implementation would use historical data
        baseline_volume = 100000  # Would come from historical averages
        expected_volume = baseline_volume * expected_ratio
        
        if expected_volume > 0:
            relative_volume = session_volume / expected_volume
            return min(1.0, relative_volume / 3.0)  # Normalize, cap at 3x expected
            
        return 0.5
    
    def get_default_value(self) -> float:
        return 0.5
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'current': {'fields': ['session_volume', 'session', 'timestamp']}}