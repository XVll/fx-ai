"""
Simple callback factory for compatibility.
"""

from typing import Dict, Any, List

from config import CallbackConfig
from . import CallbackManager
from .base import BaseCallback
from ..continuous_training_callback import ContinuousTrainingCallback


def create_callbacks_from_config(config: CallbackConfig) -> CallbackManager:
    """
    Create callback manager from simple config.
    
    Args:
        config: Simple config dict with callback settings
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        CallbackManager with configured callbacks
    """
    callbacks = []
    
    # Add basic callbacks based on config
    if config.continuous.enabled:
        metrics_config = config.continuous
        callbacks.append(ContinuousTrainingCallback(,config.continuous))
    

    return CallbackManager(callbacks)