"""
Base callback class for v2 system.

Simplified unified callback architecture with strongly typed context events.
"""

from typing import Optional
from abc import ABC
import logging

from v2.core.shutdown import IShutdownHandler
from .context import (
    TrainingStartContext,
    EpisodeEndContext,
    UpdateEndContext,
    TrainingEndContext,
    CustomEventContext
)


class BaseCallback(ABC,IShutdownHandler):
    """
    Unified base callback for v2 system.
    
    Provides a clean, simple interface with only 4 core events
    and rich context dictionaries instead of multiple parameters.
    
    Key improvements over v1:
    - Single inheritance hierarchy (no BaseCallback vs TrainingCallback confusion)
    - Rich context events instead of multiple parameters
    - Only essential lifecycle events
    - Consistent enable/disable mechanism
    - Error isolation
    """
    
    def __init__(self, config=None, enabled: bool = True, name: Optional[str] = None):
        """
        Initialize base callback.
        
        Args:
            config: Callback configuration (with shutdown_timeout)
            enabled: Whether callback is active (overrides config if provided)
            name: Optional custom name (defaults to class name)
        """
        self.config = config
        self.enabled = enabled if config is None else config.enabled
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"callback.{self.name}")
        
        # Callback state
        self._training_started = False
        self._episode_count = 0
        self._update_count = 0
    
    def on_training_start(self, context: TrainingStartContext) -> None:
        """
        Called once at training start.
        
        Args:
            context: Typed training start context with config, trainer, environment, etc.
        """
        self._training_started = True
        self.logger.debug(f"Training started - {self.name}")
    
    def on_episode_end(self, context: EpisodeEndContext) -> None:
        """
        Called after each episode completion.
        
        Args:
            context: Typed episode context with episode info, metrics, portfolio state, etc.
        """
        self._episode_count += 1
        
    def on_update_end(self, context: UpdateEndContext) -> None:
        """
        Called after each PPO update completion.
        
        Args:
            context: Typed update context with losses, metrics, model state, etc.
        """
        self._update_count += 1
        
    def on_training_end(self, context: TrainingEndContext) -> None:
        """
        Called once at training completion.
        
        Args:
            context: Typed training end context with final metrics, duration, etc.
        """
        self._training_started = False
        self.logger.debug(f"Training ended - {self.name}")
    
    def on_custom_event(self, context: CustomEventContext) -> None:
        """
        Handle custom events for specialized callbacks.
        
        Args:
            context: Typed custom event context with event name and data
            
        This allows callbacks to handle specialized events like:
        - model_checkpoint, attribution_complete, optuna_trial_complete, etc.
        """
        pass
    
    def shutdown(self) -> None:
        """
        Perform callback shutdown and cleanup.
        
        Called during graceful shutdown to allow callbacks
        to save state, close connections, etc.
        """
        self.logger.debug(f"Shutdown - {self.name}")
    
    def get_shutdown_timeout(self) -> float:
        """
        Get shutdown timeout from configuration.
        
        Returns:
            Timeout in seconds from config or default
        """
        if self.config and hasattr(self.config, 'shutdown_timeout'):
            return self.config.shutdown_timeout
        return 10.0  # Default fallback
    
    # Helper methods for subclasses
    
    def is_enabled(self) -> bool:
        """Check if callback is enabled."""
        return self.enabled
    
    def enable(self) -> None:
        """Enable callback."""
        self.enabled = True
        self.logger.debug(f"Enabled - {self.name}")
    
    def disable(self) -> None:
        """Disable callback."""
        self.enabled = False
        self.logger.debug(f"Disabled - {self.name}")
    
    @property
    def episode_count(self) -> int:
        """Number of episodes processed."""
        return self._episode_count
    
    @property 
    def update_count(self) -> int:
        """Number of updates processed."""
        return self._update_count
    
    @property
    def is_training_active(self) -> bool:
        """Whether training is currently active."""
        return self._training_started
    
    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name}({status}, episodes={self._episode_count}, updates={self._update_count})"