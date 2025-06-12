import logging
from typing import Optional, List, Dict, Any

from callbacks import BaseCallback


class CallbackManager:
    """
    Simple callback manager with essential features.

    Provides:
    - Event triggering to all-callbacks
    - Error isolation
    - Component registration
    - Basic performance tracking
    """

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        self.callbacks = callbacks or []
        self.logger = logging.getLogger("callback_manager")

        # Performance tracking
        self.total_events = 0
        self.error_count = 0

        # Component references
        self.trainer = None
        self.environment = None
        self.data_manager = None

        self.logger.info(f"CallbackManager initialized with {len(self.callbacks)} callbacks")

    def add_callback(self, callback: BaseCallback) -> None:
        """Add a callback."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)

            # Set component references
            callback.trainer = self.trainer
            callback.environment = self.environment
            callback.data_manager = self.data_manager

            self.logger.info(f"Added callback: {callback.name}")

    def remove_callback(self, callback: BaseCallback) -> None:
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.info(f"Removed callback: {callback.name}")

    def trigger_event(self, event_name: str, context: Dict[str, Any]) -> None:
        """Trigger an event on all enabled callbacks."""
        self.total_events += 1

        enabled_callbacks = [cb for cb in self.callbacks if cb.enabled]
        if not enabled_callbacks:
            return

        self.logger.debug(f"Triggering {event_name} on {len(enabled_callbacks)} callbacks")

        for callback in enabled_callbacks:
            try:
                # Get the handler method
                handler = getattr(callback, f"on_{event_name}", None)
                if handler and callable(handler):
                    if event_name == "custom_event":
                        # Custom events have different signature
                        custom_event_name = context.get('event_name', 'unknown')
                        handler(custom_event_name, context)
                    else:
                        handler(context)

            except Exception as e:
                self.error_count += 1
                self.logger.error(
                    f"Callback {callback.name} failed on {event_name}: {e}",
                    exc_info=True
                )

    # Convenience methods for common events

    def trigger_training_start(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("training_start", context or {})

    def trigger_training_end(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("training_end", context or {})

    def trigger_episode_start(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("episode_start", context or {})

    def trigger_episode_end(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("episode_end", context or {})

    def trigger_step_end(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("step_end", context or {})

    def trigger_update_end(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("update_end", context or {})

    def trigger_evaluation_complete(self, context: Dict[str, Any] = None) -> None:
        self.trigger_event("evaluation_complete", context or {})

    def trigger_custom_event(self, event_name: str, context: Dict[str, Any] = None) -> None:
        ctx = context or {}
        ctx['event_name'] = event_name
        self.trigger_event("custom_event", ctx)

    # Component registration

    def register_trainer(self, trainer) -> None:
        """Register trainer with all callbacks."""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.trainer = trainer

    def register_environment(self, environment) -> None:
        """Register environment with all callbacks."""
        self.environment = environment
        for callback in self.callbacks:
            callback.environment = environment

    def register_data_manager(self, data_manager) -> None:
        """Register data manager with all callbacks."""
        self.data_manager = data_manager
        for callback in self.callbacks:
            callback.data_manager = data_manager

    # Utilities

    def get_callback(self, name: str) -> Optional[BaseCallback]:
        """Get callback by name."""
        for callback in self.callbacks:
            if callback.name == name:
                return callback
        return None

    def get_enabled_callbacks(self) -> List[BaseCallback]:
        """Get all enabled callbacks."""
        return [cb for cb in self.callbacks if cb.enabled]

    def enable_callback(self, name: str) -> bool:
        """Enable callback by name."""
        callback = self.get_callback(name)
        if callback:
            callback.enable()
            return True
        return False

    def disable_callback(self, name: str) -> bool:
        """Disable callback by name."""
        callback = self.get_callback(name)
        if callback:
            callback.disable()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'total_callbacks': len(self.callbacks),
            'enabled_callbacks': len(self.get_enabled_callbacks()),
            'total_events': self.total_events,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.total_events)
        }

    def shutdown(self) -> None:
        """Shutdown all callbacks."""
        for callback in self.callbacks:
            try:
                callback.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {callback.name}: {e}")

        stats = self.get_stats()
        self.logger.info(f"Shutdown complete: {stats}")

    def __repr__(self) -> str:
        enabled = len(self.get_enabled_callbacks())
        total = len(self.callbacks)
        return f"CallbackManager({enabled}/{total} enabled, {self.total_events} events)"
