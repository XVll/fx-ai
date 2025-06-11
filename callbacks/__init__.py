"""
V2 Callback System

Enhanced callback architecture with comprehensive event support.
Provides 30+ event hooks, strongly-typed contexts, and advanced features.
"""

# New enhanced system (primary)
from .core.base_v2 import BaseCallbackV2
from .core.manager_v2 import CallbackManagerV2
from .core.events import EventType, EventPriority, EventFilter
from .core.context_v2 import (
    BaseContext, StepContext, EpisodeContext, UpdateContext,
    BatchContext, ModelContext, EvaluationContext, DataContext
)

# Legacy system (for backward compatibility)
from .core.base import BaseCallback
from .core.manager import CallbackManager
from .core.factory import create_callbacks_from_config

# Alias new system as default
BaseCallback = BaseCallbackV2  # Default to new system
CallbackManager = CallbackManagerV2  # Default to new system

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

# Evaluation callbacks
from .evaluation.evaluation_callback import EvaluationCallback

__all__ = [
    # New enhanced system
    "BaseCallbackV2",
    "CallbackManagerV2",
    "EventType",
    "EventPriority", 
    "EventFilter",
    "BaseContext",
    "StepContext",
    "EpisodeContext",
    "UpdateContext",
    "BatchContext",
    "ModelContext",
    "EvaluationContext",
    "DataContext",
    
    # Default aliases (point to new system)
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
    
    # Evaluation callbacks
    "EvaluationCallback",
]