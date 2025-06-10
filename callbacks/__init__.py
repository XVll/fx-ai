"""
V2 Callback System

Simplified, unified callback architecture for the v2 trading system.
Provides a clean interface with rich context events and organized callback types.
"""

from .core.base import BaseCallback
from .core.manager import CallbackManager
from .core.factory import create_callbacks_from_config

# Core callbacks
from .core.metrics_callback import MetricsCallback
from .core.checkpoint_callback import CheckpointCallback
from .core.wandb_callback import WandBCallback

# Analysis callbacks  
from .analysis.attribution_callback import AttributionCallback
from .analysis.performance_callback import PerformanceCallback

# Optimization callbacks
from .optimization.optuna_callback import OptunaCallback
from .optimization.early_stopping_callback import EarlyStoppingCallback

__all__ = [
    # Core system
    "BaseCallback",
    "CallbackManager", 
    "create_callbacks_from_config",
    
    # Core callbacks
    "MetricsCallback",
    "CheckpointCallback", 
    "WandBCallback",
    
    # Analysis callbacks
    "AttributionCallback",
    "PerformanceCallback",
    
    # Optimization callbacks
    "OptunaCallback",
    "EarlyStoppingCallback",
]