"""
Simple callback factory for compatibility.
"""

from typing import Dict, Any, List
from .base import BaseCallback
from .. import CallbackManager
from .examples import MetricsCallback, CheckpointCallback, PerformanceCallback


def create_callbacks_from_config(config: Dict[str, Any], **kwargs) -> CallbackManager:
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
    if config.get('metrics', {}).get('enabled', True):
        metrics_config = config.get('metrics', {})
        callbacks.append(MetricsCallback(
            log_frequency=metrics_config.get('log_freq', 100)
        ))
    
    if config.get('checkpoint', {}).get('enabled', True):
        checkpoint_config = config.get('checkpoint', {})
        callbacks.append(CheckpointCallback(
            save_frequency=checkpoint_config.get('save_freq', 200)
        ))
    
    if config.get('performance', {}).get('enabled', False):
        callbacks.append(PerformanceCallback())
    
    return CallbackManager(callbacks)