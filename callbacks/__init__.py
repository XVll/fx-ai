"""
Callback System

Simple but powerful callback architecture for training:
- 15 essential event hooks covering all training aspects
- Simple state management with JSON persistence  
- Error isolation and performance tracking
- Easy to understand and extend
"""

from .core.base import BaseCallback
from .core import CallbackManager
from .core.examples import (
    MetricsCallback,
    CheckpointCallback, 
    PerformanceCallback,
    AttributionCallback,
    TradingAnalysisCallback,
    create_callbacks
)

# Factory function (kept for compatibility)
from .core.factory import create_callbacks_from_config

__all__ = [
    # Core system
    "BaseCallback",
    "CallbackManager",
    
    # Example callbacks
    "MetricsCallback",
    "CheckpointCallback", 
    "PerformanceCallback",
    "AttributionCallback",
    "TradingAnalysisCallback",
    "create_callbacks",
    
    # Factory
    "create_callbacks_from_config",
]