"""
Enhanced base callback class with comprehensive event support.

Provides a rich interface for callbacks with 30+ event hooks, state management,
priority support, and async capabilities.
"""

from typing import Optional, Dict, Any, Set, List, Callable
from abc import ABC
import logging
import asyncio
from pathlib import Path
import pickle
from datetime import datetime

from core.shutdown import IShutdownHandler, get_global_shutdown_manager
from .events import EventType, EventPriority, EventFilter
from .context_v2 import (
    BaseContext, StepContext, EpisodeContext, RolloutContext,
    UpdateContext, BatchContext, ModelContext, EvaluationContext,
    DataContext, ErrorContext, CustomContext
)


class BaseCallbackV2(ABC, IShutdownHandler):
    """
    Enhanced base callback with comprehensive event support.
    
    Key features:
    - 30+ event hooks covering all aspects of training
    - Built-in state management with persistence
    - Priority-based execution ordering
    - Async event support for long-running operations
    - Event filtering and selective listening
    - Resource management and cleanup
    - Performance profiling built-in
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        priority: EventPriority = EventPriority.NORMAL,
        event_filter: Optional[EventFilter] = None,
        state_dir: Optional[Path] = None,
        async_events: Optional[Set[EventType]] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize enhanced callback.
        
        Args:
            name: Callback name (defaults to class name)
            enabled: Whether callback is active
            priority: Execution priority for ordering
            event_filter: Filter to select which events to receive
            state_dir: Directory for state persistence
            async_events: Set of events to handle asynchronously
            config: Additional configuration
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.priority = priority
        self.event_filter = event_filter
        self.state_dir = Path(state_dir) if state_dir else None
        self.async_events = async_events or set()
        self.config = config
        
        # Logging
        self.logger = logging.getLogger(f"callback.{self.name}")
        
        # State management
        self._state: Dict[str, Any] = {}
        self._state_dirty = False
        self._last_save_time = datetime.now()
        
        # Performance tracking
        self._event_counts: Dict[EventType, int] = {}
        self._event_times: Dict[EventType, List[float]] = {}
        self._total_time = 0.0
        
        # Event history (optional)
        self._event_history: List[tuple[EventType, datetime]] = []
        self._max_history_size = 1000
        
        # Load persisted state if available
        if self.state_dir:
            self._load_state()
    
    # Core event hooks - Training lifecycle
    
    def on_training_start(self, context: BaseContext) -> None:
        """Called once at training start."""
        pass
    
    def on_training_end(self, context: BaseContext) -> None:
        """Called once at training end."""
        pass
    
    def on_training_error(self, context: ErrorContext) -> None:
        """Called when training error occurs."""
        pass
    
    # Episode events
    
    def on_episode_start(self, context: EpisodeContext) -> None:
        """Called at episode start."""
        pass
    
    def on_episode_end(self, context: EpisodeContext) -> None:
        """Called at episode end."""
        pass
    
    def on_episode_reset(self, context: EpisodeContext) -> None:
        """Called when environment resets."""
        pass
    
    def on_episode_terminated(self, context: EpisodeContext) -> None:
        """Called when episode terminates naturally."""
        pass
    
    def on_episode_truncated(self, context: EpisodeContext) -> None:
        """Called when episode is truncated."""
        pass
    
    # Step events
    
    def on_step_start(self, context: StepContext) -> None:
        """Called before environment step."""
        pass
    
    def on_step_end(self, context: StepContext) -> None:
        """Called after environment step."""
        pass
    
    def on_action_selected(self, context: StepContext) -> None:
        """Called after action selection."""
        pass
    
    def on_reward_computed(self, context: StepContext) -> None:
        """Called after reward computation."""
        pass
    
    # Rollout events
    
    def on_rollout_start(self, context: RolloutContext) -> None:
        """Called before rollout collection."""
        pass
    
    def on_rollout_end(self, context: RolloutContext) -> None:
        """Called after rollout collection."""
        pass
    
    def on_buffer_add(self, context: BaseContext) -> None:
        """Called when data added to buffer."""
        pass
    
    def on_buffer_ready(self, context: BaseContext) -> None:
        """Called when buffer ready for training."""
        pass
    
    # Update events
    
    def on_update_start(self, context: UpdateContext) -> None:
        """Called before policy update."""
        pass
    
    def on_update_end(self, context: UpdateContext) -> None:
        """Called after policy update."""
        pass
    
    def on_gradient_computed(self, context: UpdateContext) -> None:
        """Called after gradient computation."""
        pass
    
    def on_optimizer_step(self, context: UpdateContext) -> None:
        """Called after optimizer step."""
        pass
    
    def on_batch_start(self, context: BatchContext) -> None:
        """Called before batch processing."""
        pass
    
    def on_batch_end(self, context: BatchContext) -> None:
        """Called after batch processing."""
        pass
    
    def on_epoch_start(self, context: UpdateContext) -> None:
        """Called at epoch start."""
        pass
    
    def on_epoch_end(self, context: UpdateContext) -> None:
        """Called at epoch end."""
        pass
    
    # Model events
    
    def on_model_saved(self, context: ModelContext) -> None:
        """Called when model is saved."""
        pass
    
    def on_model_loaded(self, context: ModelContext) -> None:
        """Called when model is loaded."""
        pass
    
    def on_model_improved(self, context: ModelContext) -> None:
        """Called when model performance improves."""
        pass
    
    def on_learning_rate_updated(self, context: BaseContext) -> None:
        """Called when learning rate changes."""
        pass
    
    # Evaluation events
    
    def on_evaluation_start(self, context: EvaluationContext) -> None:
        """Called when evaluation starts."""
        pass
    
    def on_evaluation_end(self, context: EvaluationContext) -> None:
        """Called when evaluation ends."""
        pass
    
    def on_evaluation_episode(self, context: EpisodeContext) -> None:
        """Called for each evaluation episode."""
        pass
    
    # Data events
    
    def on_data_loaded(self, context: DataContext) -> None:
        """Called when new data is loaded."""
        pass
    
    def on_day_switched(self, context: DataContext) -> None:
        """Called when switching trading day."""
        pass
    
    def on_symbol_switched(self, context: DataContext) -> None:
        """Called when switching symbol."""
        pass
    
    # Performance events
    
    def on_memory_warning(self, context: BaseContext) -> None:
        """Called on high memory usage."""
        pass
    
    def on_performance_log(self, context: BaseContext) -> None:
        """Called with performance metrics."""
        pass
    
    # Custom events
    
    def on_custom_event(self, context: CustomContext) -> None:
        """Called for custom events."""
        pass
    
    # Async event handlers (optional overrides)
    
    async def on_step_end_async(self, context: StepContext) -> None:
        """Async version of on_step_end."""
        pass
    
    async def on_episode_end_async(self, context: EpisodeContext) -> None:
        """Async version of on_episode_end."""
        pass
    
    async def on_update_end_async(self, context: UpdateContext) -> None:
        """Async version of on_update_end."""
        pass
    
    # State management
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set state value."""
        self._state[key] = value
        self._state_dirty = True
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values."""
        self._state.update(updates)
        self._state_dirty = True
    
    def clear_state(self) -> None:
        """Clear all state."""
        self._state.clear()
        self._state_dirty = True
    
    def save_state(self, force: bool = False) -> None:
        """Save state to disk."""
        if not self.state_dir or (not force and not self._state_dirty):
            return
        
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.state_dir / f"{self.name}_state.pkl"
        
        try:
            with open(state_file, 'wb') as f:
                pickle.dump(self._state, f)
            self._state_dirty = False
            self._last_save_time = datetime.now()
            self.logger.debug(f"State saved to {state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.state_dir:
            return
        
        state_file = self.state_dir / f"{self.name}_state.pkl"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'rb') as f:
                self._state = pickle.load(f)
            self.logger.debug(f"State loaded from {state_file}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            self._state = {}
    
    # Event filtering
    
    def should_handle_event(self, event_type: EventType) -> bool:
        """Check if callback should handle this event type."""
        if not self.enabled:
            return False
        
        if not self.event_filter:
            return True
        
        # Check event type filters
        if self.event_filter.event_types and event_type not in self.event_filter.event_types:
            return False
        
        if self.event_filter.exclude_types and event_type in self.event_filter.exclude_types:
            return False
        
        return True
    
    # Performance tracking
    
    def record_event_time(self, event_type: EventType, duration: float) -> None:
        """Record event processing time."""
        if event_type not in self._event_times:
            self._event_times[event_type] = []
        self._event_times[event_type].append(duration)
        self._total_time += duration
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get callback performance statistics."""
        stats = {
            'total_events': sum(self._event_counts.values()),
            'total_time': self._total_time,
            'events_by_type': dict(self._event_counts),
            'avg_time_by_type': {}
        }
        
        for event_type, times in self._event_times.items():
            if times:
                stats['avg_time_by_type'][event_type.name] = sum(times) / len(times)
        
        return stats
    
    # Lifecycle management
    
    def register_shutdown(self) -> None:
        """Register with shutdown manager."""
        shutdown_manager = get_global_shutdown_manager()
        shutdown_manager.register_component(
            component=self,
            timeout=self.get_shutdown_timeout(),
            name=f"callback_{self.name}"
        )
        self.logger.debug(f"Registered with shutdown manager")
    
    def shutdown(self) -> None:
        """Perform cleanup on shutdown."""
        self.logger.debug(f"Shutting down {self.name}")
        
        # Save state if needed
        if self._state_dirty:
            self.save_state(force=True)
        
        # Log performance stats
        stats = self.get_performance_stats()
        if stats['total_events'] > 0:
            self.logger.info(
                f"Performance: {stats['total_events']} events, "
                f"{stats['total_time']:.2f}s total time"
            )
    
    def get_shutdown_timeout(self) -> float:
        """Get shutdown timeout."""
        if self.config and hasattr(self.config, 'shutdown_timeout'):
            return self.config.shutdown_timeout
        return 10.0
    
    # Utilities
    
    def enable(self) -> None:
        """Enable callback."""
        self.enabled = True
        self.logger.debug(f"Enabled")
    
    def disable(self) -> None:
        """Disable callback."""
        self.enabled = False
        self.logger.debug(f"Disabled")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.name}("
            f"enabled={self.enabled}, "
            f"priority={self.priority.name}, "
            f"events={sum(self._event_counts.values())}"
            f")"
        )