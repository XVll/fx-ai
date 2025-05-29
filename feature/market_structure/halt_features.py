"""Trading halt and market status features."""

from typing import Dict, Any
from feature.feature_base import BaseFeature, FeatureConfig


class HaltStateFeature(BaseFeature):
    """Trading halt status indicator.
    
    Returns 1.0 if trading is halted, 0.0 otherwise.
    This is critical for avoiding trades during market halts.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="is_halted", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Check if trading is currently halted."""
        return 1.0 if market_data.get('is_halted', False) else 0.0
    
    def get_default_value(self) -> float:
        """Default to not halted."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Binary feature, no normalization needed."""
        return {'min': 0.0, 'max': 1.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires current market status."""
        return {
            'status': {
                'lookback': 1,
                'fields': ['is_halted']
            }
        }


class TimeSinceHaltFeature(BaseFeature):
    """Time since last trading halt in seconds.
    
    Normalized to [0, 1] where 1 represents >1 hour since halt.
    Recent halts may indicate volatility.
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="time_since_halt", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate seconds since last halt."""
        current_time = market_data.get('timestamp')
        last_halt_time = market_data.get('last_halt_time')
        
        if last_halt_time is None or current_time is None:
            # No halt in recent history
            return 3600.0  # 1 hour
        
        time_diff = (current_time - last_halt_time).total_seconds()
        return max(0.0, time_diff)
    
    def get_default_value(self) -> float:
        """Default to max time (no recent halt)."""
        return 3600.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to 1 hour max."""
        return {'min': 0.0, 'max': 3600.0}
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requires halt history."""
        return {
            'status': {
                'lookback': 3600,  # Look back 1 hour
                'fields': ['is_halted', 'timestamp']
            }
        }