"""
Enhanced callback manager with comprehensive event routing and management.

Provides advanced features like event prioritization, async handling,
batching, and performance monitoring.
"""

from typing import List, Optional, Dict, Any, Set, Callable, Union
import logging
import asyncio
from collections import defaultdict
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import queue

from .base_v2 import BaseCallbackV2
from .events import EventType, EventPriority, EventMetadata, EventFilter
from .context_v2 import BaseContext, get_context_class


class CallbackManagerV2:
    """
    Enhanced callback manager with advanced event routing.
    
    Features:
    - Priority-based callback execution
    - Async event handling with thread pool
    - Event batching for performance
    - Comprehensive error isolation
    - Performance monitoring
    - Dynamic callback registration
    - Event history tracking
    - Event filtering and routing
    """
    
    def __init__(
        self,
        callbacks: Optional[List[BaseCallbackV2]] = None,
        async_executor: Optional[ThreadPoolExecutor] = None,
        max_async_workers: int = 4,
        enable_history: bool = False,
        max_history_size: int = 10000,
        batch_events: Optional[Set[EventType]] = None,
        batch_timeout: float = 0.1
    ):
        """
        Initialize enhanced callback manager.
        
        Args:
            callbacks: Initial list of callbacks
            async_executor: Thread pool for async events (creates one if None)
            max_async_workers: Max workers for async execution
            enable_history: Whether to track event history
            max_history_size: Maximum events to keep in history
            batch_events: Event types to batch for performance
            batch_timeout: Timeout for batched events
        """
        self.logger = logging.getLogger("callback_manager_v2")
        
        # Callback storage by priority
        self._callbacks_by_priority: Dict[EventPriority, List[BaseCallbackV2]] = defaultdict(list)
        self._callbacks_by_name: Dict[str, BaseCallbackV2] = {}
        self._callbacks_by_event: Dict[EventType, List[BaseCallbackV2]] = defaultdict(list)
        
        # Async handling
        self.async_executor = async_executor or ThreadPoolExecutor(max_workers=max_async_workers)
        self._async_tasks: List[asyncio.Task] = []
        self._async_queue: queue.Queue = queue.Queue()
        
        # Event batching
        self.batch_events = batch_events or set()
        self.batch_timeout = batch_timeout
        self._event_batches: Dict[EventType, List[tuple[BaseContext, datetime]]] = defaultdict(list)
        self._last_batch_time: Dict[EventType, datetime] = {}
        
        # Event history
        self.enable_history = enable_history
        self.max_history_size = max_history_size
        self._event_history: List[tuple[EventMetadata, datetime]] = []
        
        # Performance tracking
        self._total_events = 0
        self._error_count = 0
        self._event_counts: Dict[EventType, int] = defaultdict(int)
        self._event_times: Dict[EventType, List[float]] = defaultdict(list)
        self._callback_times: Dict[str, Dict[EventType, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Component references
        self._components: Dict[str, Any] = {}
        
        # Add initial callbacks
        if callbacks:
            for callback in callbacks:
                self.add_callback(callback)
        
        self.logger.info(
            f"CallbackManagerV2 initialized with {len(self._callbacks_by_name)} callbacks, "
            f"async_workers={max_async_workers}"
        )
    
    def add_callback(self, callback: BaseCallbackV2) -> None:
        """Add a callback with proper indexing."""
        if callback.name in self._callbacks_by_name:
            self.logger.warning(f"Callback {callback.name} already exists, replacing")
            self.remove_callback(callback.name)
        
        # Index by priority
        self._callbacks_by_priority[callback.priority].append(callback)
        
        # Index by name
        self._callbacks_by_name[callback.name] = callback
        
        # Index by event types if filtered
        if callback.event_filter and callback.event_filter.event_types:
            for event_type in callback.event_filter.event_types:
                self._callbacks_by_event[event_type].append(callback)
        else:
            # Callback handles all events
            for event_type in EventType:
                self._callbacks_by_event[event_type].append(callback)
        
        # Register components with callback
        for comp_name, component in self._components.items():
            if hasattr(callback, comp_name):
                setattr(callback, comp_name, component)
        
        self.logger.info(f"Added callback: {callback.name} (priority={callback.priority.name})")
    
    def remove_callback(self, name: str) -> bool:
        """Remove a callback by name."""
        callback = self._callbacks_by_name.get(name)
        if not callback:
            self.logger.warning(f"Callback {name} not found")
            return False
        
        # Remove from all indices
        self._callbacks_by_priority[callback.priority].remove(callback)
        del self._callbacks_by_name[name]
        
        for event_callbacks in self._callbacks_by_event.values():
            if callback in event_callbacks:
                event_callbacks.remove(callback)
        
        self.logger.info(f"Removed callback: {name}")
        return True
    
    def trigger_event(
        self,
        event_type: EventType,
        context: BaseContext,
        event_metadata: Optional[EventMetadata] = None
    ) -> None:
        """
        Trigger an event on all applicable callbacks.
        
        Args:
            event_type: Type of event to trigger
            context: Event context with data
            event_metadata: Optional event metadata
        """
        self._total_events += 1
        self._event_counts[event_type] += 1
        
        # Create metadata if not provided
        if not event_metadata:
            event_metadata = EventMetadata(
                event_type=event_type,
                source=context.__class__.__name__
            )
        
        # Set event metadata in context
        context.event_metadata = event_metadata
        
        # Record history if enabled
        if self.enable_history:
            self._record_event_history(event_metadata)
        
        # Check if this is a batched event
        if event_type in self.batch_events:
            self._add_to_batch(event_type, context)
            return
        
        # Execute event on callbacks
        self._execute_event(event_type, context)
    
    def _execute_event(self, event_type: EventType, context: BaseContext) -> None:
        """Execute event on all applicable callbacks in priority order."""
        start_time = time.time()
        
        # Get callbacks for this event type
        callbacks = self._get_callbacks_for_event(event_type, context)
        
        if not callbacks:
            return
        
        self.logger.debug(f"Executing {event_type.name} on {len(callbacks)} callbacks")
        
        # Group by priority and execute in order
        for priority in EventPriority:
            priority_callbacks = [cb for cb in callbacks if cb.priority == priority]
            
            for callback in priority_callbacks:
                self._execute_callback_event(callback, event_type, context)
        
        # Record timing
        duration = time.time() - start_time
        self._event_times[event_type].append(duration)
    
    def _execute_callback_event(
        self,
        callback: BaseCallbackV2,
        event_type: EventType,
        context: BaseContext
    ) -> None:
        """Execute a single event on a callback with error handling."""
        if not callback.enabled or not callback.should_handle_event(event_type):
            return
        
        # Get the handler method
        handler_name = f"on_{event_type.name.lower()}"
        handler = getattr(callback, handler_name, None)
        
        if not handler:
            return
        
        start_time = time.time()
        
        try:
            # Check if this should be async
            if event_type in callback.async_events:
                # Execute async handler
                async_handler_name = f"{handler_name}_async"
                async_handler = getattr(callback, async_handler_name, None)
                
                if async_handler:
                    # Submit to executor
                    future = self.async_executor.submit(
                        self._run_async_handler,
                        async_handler,
                        context,
                        callback.name,
                        event_type
                    )
                    # Don't wait for completion
                else:
                    # Run sync handler in thread pool
                    self.async_executor.submit(handler, context)
            else:
                # Execute synchronously
                handler(context)
            
            # Update event count for callback
            callback._event_counts[event_type] = callback._event_counts.get(event_type, 0) + 1
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(
                f"Callback {callback.name} failed on {event_type.name}: {e}",
                exc_info=True
            )
        
        finally:
            # Record timing
            duration = time.time() - start_time
            callback.record_event_time(event_type, duration)
            self._callback_times[callback.name][event_type].append(duration)
    
    def _run_async_handler(
        self,
        handler: Callable,
        context: BaseContext,
        callback_name: str,
        event_type: EventType
    ) -> None:
        """Run an async handler in a new event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handler(context))
        except Exception as e:
            self.logger.error(
                f"Async handler {callback_name}.{handler.__name__} failed: {e}",
                exc_info=True
            )
        finally:
            loop.close()
    
    def _get_callbacks_for_event(
        self,
        event_type: EventType,
        context: BaseContext
    ) -> List[BaseCallbackV2]:
        """Get all callbacks that should handle this event."""
        callbacks = []
        
        # Get callbacks registered for this event type
        event_callbacks = self._callbacks_by_event.get(event_type, [])
        
        for callback in event_callbacks:
            if not callback.enabled:
                continue
            
            # Apply event filter
            if callback.event_filter:
                # Check source filter
                if (callback.event_filter.sources and 
                    context.event_metadata.source not in callback.event_filter.sources):
                    continue
                
                # Check tag filter
                if (callback.event_filter.tags and
                    not callback.event_filter.tags.intersection(context.event_metadata.tags)):
                    continue
                
                # Check priority filter
                if (callback.event_filter.min_priority and
                    context.event_metadata.priority.value > callback.event_filter.min_priority.value):
                    continue
            
            callbacks.append(callback)
        
        return callbacks
    
    # Event batching
    
    def _add_to_batch(self, event_type: EventType, context: BaseContext) -> None:
        """Add event to batch for later processing."""
        self._event_batches[event_type].append((context, datetime.now()))
        
        # Check if batch should be flushed
        last_time = self._last_batch_time.get(event_type)
        if not last_time or (datetime.now() - last_time).total_seconds() > self.batch_timeout:
            self.flush_batch(event_type)
    
    def flush_batch(self, event_type: EventType) -> None:
        """Flush a batch of events."""
        batch = self._event_batches.get(event_type, [])
        if not batch:
            return
        
        self.logger.debug(f"Flushing batch of {len(batch)} {event_type.name} events")
        
        # Process all events in batch
        for context, timestamp in batch:
            self._execute_event(event_type, context)
        
        # Clear batch
        self._event_batches[event_type].clear()
        self._last_batch_time[event_type] = datetime.now()
    
    def flush_all_batches(self) -> None:
        """Flush all pending batches."""
        for event_type in list(self._event_batches.keys()):
            self.flush_batch(event_type)
    
    # Component registration
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component for callback access."""
        self._components[name] = component
        
        # Update all callbacks
        for callback in self._callbacks_by_name.values():
            if hasattr(callback, name):
                setattr(callback, name, component)
        
        self.logger.debug(f"Registered component: {name}")
    
    def register_trainer(self, trainer: Any) -> None:
        """Register trainer component."""
        self.register_component('trainer', trainer)
    
    def register_environment(self, environment: Any) -> None:
        """Register environment component."""
        self.register_component('environment', environment)
    
    def register_data_manager(self, data_manager: Any) -> None:
        """Register data manager component."""
        self.register_component('data_manager', data_manager)
    
    def register_episode_manager(self, episode_manager: Any) -> None:
        """Register episode manager component."""
        self.register_component('episode_manager', episode_manager)
    
    def register_model_manager(self, model_manager: Any) -> None:
        """Register model manager component."""
        self.register_component('model_manager', model_manager)
    
    # Event history
    
    def _record_event_history(self, event_metadata: EventMetadata) -> None:
        """Record event in history."""
        self._event_history.append((event_metadata, datetime.now()))
        
        # Trim history if needed
        if len(self._event_history) > self.max_history_size:
            self._event_history = self._event_history[-self.max_history_size:]
    
    def get_event_history(
        self,
        event_types: Optional[Set[EventType]] = None,
        limit: int = 100
    ) -> List[tuple[EventMetadata, datetime]]:
        """Get recent event history."""
        history = self._event_history
        
        if event_types:
            history = [(meta, ts) for meta, ts in history if meta.event_type in event_types]
        
        return history[-limit:]
    
    # Performance monitoring
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'total_events': self._total_events,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(1, self._total_events),
            'events_by_type': dict(self._event_counts),
            'avg_time_by_event': {},
            'callback_performance': {}
        }
        
        # Average time by event type
        for event_type, times in self._event_times.items():
            if times:
                stats['avg_time_by_event'][event_type.name] = {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times) * 1000,
                    'total_ms': sum(times) * 1000
                }
        
        # Performance by callback
        for callback_name, event_times in self._callback_times.items():
            callback_stats = {}
            total_time = 0
            total_events = 0
            
            for event_type, times in event_times.items():
                if times:
                    callback_stats[event_type.name] = {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times) * 1000
                    }
                    total_time += sum(times)
                    total_events += len(times)
            
            stats['callback_performance'][callback_name] = {
                'events': callback_stats,
                'total_events': total_events,
                'total_time_ms': total_time * 1000
            }
        
        return stats
    
    # Utilities
    
    def get_callback(self, name: str) -> Optional[BaseCallbackV2]:
        """Get callback by name."""
        return self._callbacks_by_name.get(name)
    
    def get_callbacks(self) -> List[BaseCallbackV2]:
        """Get all callbacks."""
        return list(self._callbacks_by_name.values())
    
    def get_enabled_callbacks(self) -> List[BaseCallbackV2]:
        """Get all enabled callbacks."""
        return [cb for cb in self._callbacks_by_name.values() if cb.enabled]
    
    def enable_callback(self, name: str) -> bool:
        """Enable a callback by name."""
        callback = self.get_callback(name)
        if callback:
            callback.enable()
            return True
        return False
    
    def disable_callback(self, name: str) -> bool:
        """Disable a callback by name."""
        callback = self.get_callback(name)
        if callback:
            callback.disable()
            return True
        return False
    
    def shutdown(self) -> None:
        """Shutdown manager and cleanup resources."""
        self.logger.info("Shutting down CallbackManagerV2")
        
        # Flush any pending batches
        self.flush_all_batches()
        
        # Shutdown executor
        self.async_executor.shutdown(wait=True)
        
        # Log performance stats
        stats = self.get_performance_stats()
        self.logger.info(
            f"Final stats: {stats['total_events']} events, "
            f"{stats['error_count']} errors ({stats['error_rate']:.1%} error rate)"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        enabled = len(self.get_enabled_callbacks())
        total = len(self._callbacks_by_name)
        return (
            f"CallbackManagerV2("
            f"{enabled}/{total} callbacks, "
            f"{self._total_events} events, "
            f"{self._error_count} errors)"
        )