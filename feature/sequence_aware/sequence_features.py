"""Sequence-aware features that utilize full temporal windows dynamically."""

import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig


class TrendAccelerationFeature(BaseFeature):
    """Measures how trend strength is accelerating over the sequence.
    
    Instead of just current velocity, this tracks if momentum is building
    or fading across the entire window. Dynamic and sequence-aware.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="trend_acceleration", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate trend acceleration over sequence."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 10:
            return 0.0
            
        # Get price series
        prices = [bar.get('close', 0) for bar in bars[-20:]]
        
        if len(prices) < 10:
            return 0.0
            
        # Calculate velocity for each segment
        early_segment = prices[:10]
        late_segment = prices[-10:]
        
        early_velocity = (early_segment[-1] - early_segment[0]) / len(early_segment) if early_segment[0] > 0 else 0
        late_velocity = (late_segment[-1] - late_segment[0]) / len(late_segment) if late_segment[0] > 0 else 0
        
        # Acceleration = change in velocity
        if early_segment[0] > 0:
            early_velocity_norm = early_velocity / early_segment[0]
            late_velocity_norm = late_velocity / early_segment[0] if len(prices) >= 20 else 0
            acceleration = late_velocity_norm - early_velocity_norm
            return max(-0.01, min(0.01, acceleration))  # ±1% per bar max
            
        return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -0.01, 'max': 0.01, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'mf_bars_1m': {'lookback': 20, 'fields': ['close']}}


class VolumePatternEvolutionFeature(BaseFeature):
    """Tracks how volume patterns evolve over the sequence.
    
    Captures whether volume is building consistently (squeeze setup)
    or spiking randomly (noise). True sequence utilization.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="volume_pattern_evolution", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume pattern evolution."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 15:
            return 0.0
            
        # Get volume series
        volumes = [bar.get('volume', 0) for bar in bars[-15:]]
        
        if len(volumes) < 15:
            return 0.0
            
        # Calculate volume trend over segments
        segment_size = 5
        trends = []
        
        for i in range(0, len(volumes) - segment_size + 1, segment_size):
            segment = volumes[i:i + segment_size]
            if len(segment) >= 3:
                # Linear regression slope
                x = np.arange(len(segment))
                slope = np.polyfit(x, segment, 1)[0]
                avg_vol = np.mean(segment)
                if avg_vol > 0:
                    trends.append(slope / avg_vol)  # Normalized slope
                    
        if len(trends) < 2:
            return 0.0
            
        # Evolution = are volume trends getting stronger?
        evolution = trends[-1] - trends[0] if len(trends) >= 2 else 0
        return max(-1.0, min(1.0, evolution * 100))  # Scale and clamp
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'mf_bars_1m': {'lookback': 15, 'fields': ['volume']}}


class MomentumQualityFeature(BaseFeature):
    """Measures quality of momentum - smooth vs choppy.
    
    High quality = consistent directional moves
    Low quality = choppy, noisy price action
    Uses entire sequence to assess momentum smoothness.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="momentum_quality", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate momentum quality over sequence."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 10:
            return 0.0
            
        # Get price series
        prices = [bar.get('close', 0) for bar in bars[-10:]]
        
        if len(prices) < 5:
            return 0.0
            
        # Calculate directional consistency
        moves = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                move = (prices[i] - prices[i-1]) / prices[i-1]
                moves.append(move)
                
        if len(moves) < 3:
            return 0.0
            
        # Quality = how consistent are the moves?
        move_directions = [1 if m > 0 else -1 for m in moves]
        
        # Count direction changes
        direction_changes = sum(1 for i in range(1, len(move_directions)) 
                              if move_directions[i] != move_directions[i-1])
        
        # Fewer changes = higher quality
        max_changes = len(move_directions) - 1
        if max_changes > 0:
            quality = 1.0 - (direction_changes / max_changes)
            return quality
            
        return 0.5  # Neutral
    
    def get_default_value(self) -> float:
        return 0.5
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'mf_bars_1m': {'lookback': 10, 'fields': ['close']}}


class PatternMaturationFeature(BaseFeature):
    """Tracks how close a pattern is to completion/breakout.
    
    Uses sequence to detect pattern development stage:
    0 = just starting, 1 = fully mature/ready for breakout
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="pattern_maturation", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate pattern maturation level."""
        bars = market_data.get('mf_bars_1m', [])
        
        if len(bars) < 20:
            return 0.0
            
        # Get highs and lows
        highs = [bar.get('high', 0) for bar in bars[-20:]]
        lows = [bar.get('low', 0) for bar in bars[-20:]]
        volumes = [bar.get('volume', 0) for bar in bars[-20:]]
        
        if not all([highs, lows, volumes]):
            return 0.0
            
        # Calculate range compression over time
        early_range = max(highs[:10]) - min(lows[:10]) if len(highs) >= 10 else 0
        late_range = max(highs[-10:]) - min(lows[-10:]) if len(highs) >= 10 else 0
        
        compression = 0.0
        if early_range > 0:
            compression = 1.0 - (late_range / early_range)
            compression = max(0.0, min(1.0, compression))
            
        # Volume building factor
        early_vol = np.mean(volumes[:10]) if len(volumes) >= 10 else 0
        late_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
        
        volume_factor = 0.0
        if early_vol > 0:
            volume_factor = min(2.0, late_vol / early_vol) / 2.0  # Normalize to [0,1]
            
        # Combine factors
        maturation = (compression * 0.7) + (volume_factor * 0.3)
        return min(1.0, maturation)
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'mf_bars_1m': {'lookback': 20, 'fields': ['high', 'low', 'volume']}}