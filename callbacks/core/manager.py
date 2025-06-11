"""
Callback manager for v2 system.

Simplified manager that provides a single consistent interface
for all callback events with strongly typed contexts and error isolation.
"""

from typing import List, Optional, Union
import logging

from training.training_manager import ComponentState
from .base import BaseCallback
from .context import (
    TrainingStartContext,
    EpisodeEndContext,
    UpdateEndContext,
    TrainingEndContext,
    CustomEventContext,
    CallbackContext
)


class CallbackManager:
    """
    Simplified callback manager for v2.
    
    Provides a single consistent interface for triggering callback events
    with comprehensive error handling and logging.
    
    Key improvements over v1:
    - Single consistent trigger() interface
    - No dual usage patterns (manager only)
    - Comprehensive error isolation
    - Rich logging and debugging
    - Focus on event coordination only (shutdown handled directly)
    """

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """
        Initialize callback manager.
        
        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []
        self.logger = logging.getLogger("callback_manager")
        self._total_events = 0
        self._error_count = 0

        self.logger.info(f"CallbackManager initialized with {len(self.callbacks)} callbacks")
        for callback in self.callbacks:
            self.logger.info(f"  - {callback.name}: {'enabled' if callback.enabled else 'disabled'}")

    def add_callback(self, callback: BaseCallback) -> None:
        """
        Add a callback to the manager.
        
        Args:
            callback: Callback instance to add
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            self.logger.info(f"Added callback: {callback.name}")
        else:
            self.logger.warning(f"Callback {callback.name} already exists")

    def remove_callback(self, callback: BaseCallback) -> None:
        """
        Remove a callback from the manager.
        
        Args:
            callback: Callback instance to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.info(f"Removed callback: {callback.name}")
        else:
            self.logger.warning(f"Callback {callback.name} not found")

    def trigger(self, event_name: str, context: ComponentState) -> None:
        """
        Trigger event on all enabled callbacks.
        
        Single consistent interface for all events with strongly typed contexts.
        Handles errors gracefully to prevent callback failures from crashing training.
        
        Args:
            event_name: Name of event to trigger (e.g., "on_episode_end")
            context: Strongly typed event context
        """
        self._total_events += 1

        enabled_callbacks = [cb for cb in self.callbacks if cb.enabled]
        if not enabled_callbacks:
            return

        self.logger.debug(f"Triggering {event_name} on {len(enabled_callbacks)} callbacks")

        for callback in enabled_callbacks:
            try:
                # Check if callback has the event method
                if hasattr(callback, event_name):
                    method = getattr(callback, event_name)
                    if callable(method):
                        method(context)
                    else:
                        self.logger.warning(f"Event {event_name} on {callback.name} is not callable")
                else:
                    self.logger.debug(f"Callback {callback.name} does not handle {event_name}")

            except Exception as e:
                self._error_count += 1
                self.logger.error(
                    f"Callback {callback.name} failed on {event_name}: {e}",
                    exc_info=True
                )
                # Continue with other callbacks despite error

    def trigger_training_start(self, context: TrainingStartContext) -> None:
        """Convenience method for training start."""
        self.trigger("on_training_start", context)

    def trigger_episode_end(self, context: EpisodeEndContext) -> None:
        """Convenience method for episode end."""
        self.trigger("on_episode_end", context)

    def trigger_update_end(self, context: UpdateEndContext) -> None:
        """Convenience method for update end."""
        self.trigger("on_update_end", context)

    def trigger_training_end(self, context: TrainingEndContext) -> None:
        """Convenience method for training end."""
        self.trigger("on_training_end", context)

    def trigger_custom_event(self, context: CustomEventContext) -> None:
        """Convenience method for custom events."""
        self.trigger("on_custom_event", context)

    def enable_callback(self, name: str) -> bool:
        """
        Enable callback by name.
        
        Args:
            name: Callback name to enable
            
        Returns:
            True if callback found and enabled
        """
        for callback in self.callbacks:
            if callback.name == name:
                callback.enable()
                self.logger.info(f"Enabled callback: {name}")
                return True

        self.logger.warning(f"Callback not found: {name}")
        return False

    def disable_callback(self, name: str) -> bool:
        """
        Disable callback by name.
        
        Args:
            name: Callback name to disable
            
        Returns:
            True if callback found and disabled
        """
        for callback in self.callbacks:
            if callback.name == name:
                callback.disable()
                self.logger.info(f"Disabled callback: {name}")
                return True

        self.logger.warning(f"Callback not found: {name}")
        return False

    def get_callback(self, name: str) -> Optional[BaseCallback]:
        """
        Get callback by name.
        
        Args:
            name: Callback name to find
            
        Returns:
            Callback instance or None if not found
        """
        for callback in self.callbacks:
            if callback.name == name:
                return callback
        return None

    def get_enabled_callbacks(self) -> List[BaseCallback]:
        """Get list of enabled callbacks."""
        return [cb for cb in self.callbacks if cb.enabled]

    def register_trainer(self, trainer) -> None:
        """Register a trainer with callbacks that need it."""
        for callback in self.callbacks:
            callback.trainer = trainer

    def register_environment(self, environment) -> None:
        """Register an environment with callbacks that need it."""
        for callback in self.callbacks:
            callback.environment = environment

    def get_callbacks(self) -> List[BaseCallback]:
        """
        Get all managed callbacks.
        
        Returns:
            List of callback instances
        """
        return self.callbacks.copy()

    @property
    def callback_count(self) -> int:
        """Number of callbacks managed."""
        return len(self.callbacks)

    @property
    def enabled_count(self) -> int:
        """Number of enabled callbacks."""
        return len(self.get_enabled_callbacks())

    @property
    def total_events(self) -> int:
        """Total events processed."""
        return self._total_events

    @property
    def error_count(self) -> int:
        """Total errors encountered."""
        return self._error_count

    def __repr__(self) -> str:
        """String representation."""
        return f"CallbackManager({self.enabled_count}/{self.callback_count} enabled, {self._total_events} events, {self._error_count} errors)"
