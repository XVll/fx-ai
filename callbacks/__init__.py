"""
Callback system for FxAI training infrastructure.

Provides event-driven callbacks for monitoring, metrics collection,
model management, and training lifecycle events.
"""

from .core import CallbackManager, BaseCallback
from .core.factory import create_callbacks_from_config, create_wandb_callbacks_from_config

__all__ = [
    "CallbackManager",
    "BaseCallback", 
    "create_callbacks_from_config",
    "create_wandb_callbacks_from_config"
]