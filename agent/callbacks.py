"""Enhanced callback system for training, monitoring, and analysis.

This module provides a clean callback-based architecture to replace the complex metrics system.
Each callback is responsible for its own calculations and can be easily enabled/disabled.
"""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque, defaultdict


class BaseCallback(ABC):
    """Base callback class with comprehensive hooks for all training events.
    
    This replaces the complex metrics system with a clean, extensible callback interface.
    When a callback is disabled, NO calculations are performed.
    """
    
    def __init__(self, enabled: bool = True):
        """Initialize callback.
        
        Args:
            enabled: Whether this callback is active. If False, all methods become no-ops.
        """
        self.enabled = enabled
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __call__(self, method_name: str, *args, **kwargs):
        """Dynamic method caller that respects enabled state."""
        if not self.enabled:
            return
        
        method = getattr(self, method_name, None)
        if method and callable(method):
            return method(*args, **kwargs)
    
    # ========== Training Lifecycle ==========
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        pass
    
    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Called when training ends."""
        pass
    
    # ========== Episode Lifecycle ==========
    
    def on_episode_start(self, episode_num: int, reset_info: Dict[str, Any]) -> None:
        """Called at the start of each episode.
        
        Args:
            episode_num: Current episode number
            reset_info: Information from environment reset (symbol, date, time, etc.)
        """
        pass
    
    def on_episode_end(self, episode_num: int, episode_data: Dict[str, Any]) -> None:
        """Called at the end of each episode.
        
        Args:
            episode_num: Current episode number
            episode_data: Episode summary including reward, length, final portfolio state
        """
        pass
    
    def on_episode_step(self, step_data: Dict[str, Any]) -> None:
        """Called after each environment step.
        
        Args:
            step_data: Contains state, action, reward, next_state, info, step_num
        """
        pass
    
    # ========== PPO Training ==========
    
    def on_rollout_start(self) -> None:
        """Called before collecting rollouts."""
        pass
    
    def on_rollout_end(self, rollout_data: Dict[str, Any]) -> None:
        """Called after collecting rollouts.
        
        Args:
            rollout_data: Statistics about collected rollouts
        """
        pass
    
    def on_update_start(self, update_num: int) -> None:
        """Called before PPO update."""
        pass
    
    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Called after PPO update.
        
        Args:
            update_num: Current update number
            update_metrics: Losses, gradients, learning rate, etc.
        """
        pass
    
    def on_evaluation_start(self) -> None:
        """Called before evaluation episodes."""
        pass
    
    def on_evaluation_end(self, eval_results: Dict[str, Any]) -> None:
        """Called after evaluation episodes."""
        pass
    
    # ========== Trading Events ==========
    
    def on_order_placed(self, order_data: Dict[str, Any]) -> None:
        """Called when an order is placed."""
        pass
    
    def on_order_filled(self, fill_data: Dict[str, Any]) -> None:
        """Called when an order is filled."""
        pass
    
    def on_position_opened(self, position_data: Dict[str, Any]) -> None:
        """Called when a new position is opened."""
        pass
    
    def on_position_closed(self, trade_result: Dict[str, Any]) -> None:
        """Called when a position is closed (trade completed)."""
        pass
    
    def on_portfolio_update(self, portfolio_state: Dict[str, Any]) -> None:
        """Called when portfolio state changes."""
        pass
    
    # ========== Model Events ==========
    
    def on_model_forward(self, forward_data: Dict[str, Any]) -> None:
        """Called after model forward pass.
        
        Args:
            forward_data: Includes features, outputs, attention weights, internals
        """
        pass
    
    def on_gradient_update(self, gradient_data: Dict[str, Any]) -> None:
        """Called after gradient computation."""
        pass
    
    # ========== Feature Attribution ==========
    
    def on_attribution_computed(self, attribution_data: Dict[str, Any]) -> None:
        """Called when feature attribution is computed."""
        pass
    
    # ========== Momentum Training Events ==========
    
    def on_momentum_day_change(self, day_info: Dict[str, Any]) -> None:
        """Called when switching to a new momentum day."""
        pass
    
    def on_reset_point_selected(self, reset_info: Dict[str, Any]) -> None:
        """Called when a reset point is selected."""
        pass
    
    def on_curriculum_stage_change(self, stage_info: Dict[str, Any]) -> None:
        """Called when curriculum stage changes."""
        pass
    
    # ========== Custom Events ==========
    
    def on_custom_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Called for custom events."""
        pass


class CallbackManager:
    """Manages multiple callbacks and dispatches events."""
    
    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """Initialize callback manager.
        
        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []
        self.logger = logging.getLogger(__name__)
    
    def add_callback(self, callback: BaseCallback) -> None:
        """Add a callback to the manager."""
        if callback.enabled:
            self.callbacks.append(callback)
            self.logger.debug(f"Added {callback.__class__.__name__} to callback manager")
    
    def remove_callback(self, callback: BaseCallback) -> None:
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def trigger(self, event_name: str, *args, **kwargs) -> None:
        """Trigger an event on all enabled callbacks.
        
        Args:
            event_name: Name of the callback method to call
            *args, **kwargs: Arguments to pass to the callback method
        """
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    method = getattr(callback, event_name, None)
                    if method and callable(method):
                        method(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error in {callback.__class__.__name__}.{event_name}: {e}")
    
    def get_callback(self, callback_type: type) -> Optional[BaseCallback]:
        """Get a specific callback by type."""
        for callback in self.callbacks:
            if isinstance(callback, callback_type):
                return callback
        return None
    
    def disable_all(self) -> None:
        """Disable all callbacks."""
        for callback in self.callbacks:
            callback.enabled = False
    
    def enable_all(self) -> None:
        """Enable all callbacks."""
        for callback in self.callbacks:
            callback.enabled = True


# Convenience functions for creating callback managers
def create_callback_manager(config: Dict[str, Any]) -> CallbackManager:
    """Create a callback manager based on configuration.
    
    Args:
        config: Configuration dictionary with callback settings
        
    Returns:
        Configured CallbackManager instance
    """
    from agent.optuna_callback import OptunaCallback
    from agent.wandb_callback import WandBCallback
    from agent.dashboard_callback import DashboardCallback
    
    callbacks = []
    
    # Add Optuna callback if in Optuna trial
    if config.get("optuna_trial"):
        callbacks.append(OptunaCallback(
            trial=config["optuna_trial"],
            enabled=True
        ))
    
    # Add WandB callback if enabled
    if config.get("wandb", {}).get("enabled", False):
        callbacks.append(WandBCallback(
            config=config.get("wandb", {}),
            enabled=True
        ))
    
    # Add Dashboard callback if enabled
    if config.get("dashboard", {}).get("enabled", False):
        # Try to get dashboard state
        dashboard_state = None
        try:
            from dashboard.shared_state import dashboard_state as global_state
            dashboard_state = global_state
        except ImportError:
            pass
        
        if dashboard_state:
            callbacks.append(DashboardCallback(
                config=config.get("dashboard", {}),
                dashboard_state=dashboard_state,
                enabled=True
            ))
    
    # Add any custom callbacks from config
    for callback_config in config.get("callbacks", []):
        callback_class = callback_config.get("class")
        if callback_class:
            # Dynamic import and instantiation
            module_name, class_name = callback_class.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            callback_cls = getattr(module, class_name)
            callback = callback_cls(**callback_config.get("params", {}))
            callbacks.append(callback)
    
    return CallbackManager(callbacks)