"""
Enhanced callback system core components.

This module provides a comprehensive callback system with:
- 30+ event hooks covering all aspects of training
- Strongly-typed contexts for each event
- Priority-based execution ordering
- Async event support
- State management and persistence
- Performance profiling
- Event filtering and routing
"""

# Base classes
from .base import BaseCallback
from .base_v2 import BaseCallbackV2

# Context system
from .context import (
    TrainingStartContext,
    EpisodeEndContext,
    UpdateEndContext,
    TrainingEndContext,
    CustomEventContext,
    CallbackContext
)

from .context_v2 import (
    BaseContext,
    StepContext,
    EpisodeContext,
    RolloutContext,
    UpdateContext,
    BatchContext,
    ModelContext,
    EvaluationContext,
    DataContext,
    ErrorContext,
    CustomContext,
    get_context_class
)

# Event system
from .events import (
    EventType,
    EventPriority,
    EventMetadata,
    EventFilter,
    EventRegistry
)

# Managers
from .manager import CallbackManager
from .manager_v2 import CallbackManagerV2

# Utilities
from .utils import (
    # Decorators
    event_handler,
    requires_components,
    profile_performance,
    throttle,
    batch_events,
    
    # State management
    StateManager,
    
    # Metric tracking
    MetricTracker,
    
    # Event routing
    EventRouter,
    
    # Performance profiling
    CallbackProfiler,
    
    # Aggregation utilities
    aggregate_episode_metrics,
    compute_rolling_statistics
)

# Hook registry
from .hooks import (
    HookInfo,
    HookRegistry,
    PerformanceImpact
)

# Factory
from .factory import create_callbacks_from_config

# Core callbacks
from .metrics_callback import MetricsCallback
from .checkpoint_callback import CheckpointCallback
from .wandb_callback import WandBCallback

# Version exports
__all__ = [
    # Base classes
    'BaseCallback',
    'BaseCallbackV2',
    
    # Contexts - v1
    'TrainingStartContext',
    'EpisodeEndContext', 
    'UpdateEndContext',
    'TrainingEndContext',
    'CustomEventContext',
    'CallbackContext',
    
    # Contexts - v2
    'BaseContext',
    'StepContext',
    'EpisodeContext',
    'RolloutContext',
    'UpdateContext',
    'BatchContext',
    'ModelContext',
    'EvaluationContext',
    'DataContext',
    'ErrorContext',
    'CustomContext',
    'get_context_class',
    
    # Events
    'EventType',
    'EventPriority',
    'EventMetadata',
    'EventFilter',
    'EventRegistry',
    
    # Managers
    'CallbackManager',
    'CallbackManagerV2',
    
    # Utilities
    'event_handler',
    'requires_components',
    'profile_performance',
    'throttle',
    'batch_events',
    'StateManager',
    'MetricTracker',
    'EventRouter',
    'CallbackProfiler',
    'aggregate_episode_metrics',
    'compute_rolling_statistics',
    
    # Hook registry
    'HookInfo',
    'HookRegistry',
    'PerformanceImpact',
    
    # Factory
    'create_callbacks_from_config',
    
    # Core callbacks
    'MetricsCallback',
    'CheckpointCallback',
    'WandBCallback'
]