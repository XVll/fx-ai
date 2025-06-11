"""
Utilities for callback development.

Provides helper functions, decorators, and utilities for building
advanced callbacks.
"""

from typing import Any, Dict, List, Optional, Callable, Set, Union
from functools import wraps
import time
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque, defaultdict

from .events import EventType, EventPriority
from .context_v2 import BaseContext


# Decorators

def event_handler(*event_types: EventType):
    """
    Decorator to mark methods as handlers for specific event types.
    
    Usage:
        @event_handler(EventType.EPISODE_END, EventType.UPDATE_END)
        def handle_completion_events(self, context):
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._handled_events = set(event_types)
        return func
    return decorator


def requires_components(*components: str):
    """
    Decorator to ensure required components are available.
    
    Usage:
        @requires_components('trainer', 'environment')
        def on_update_end(self, context):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, context: BaseContext, *args, **kwargs):
            for component in components:
                if not hasattr(context, component) or getattr(context, component) is None:
                    self.logger.warning(
                        f"Skipping {func.__name__}: required component '{component}' not available"
                    )
                    return
            return func(self, context, *args, **kwargs)
        return wrapper
    return decorator


def profile_performance(func: Callable) -> Callable:
    """
    Decorator to profile callback method performance.
    
    Usage:
        @profile_performance
        def on_episode_end(self, context):
            ...
    """
    @wraps(func)
    def wrapper(self, context: BaseContext, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, context, *args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            if hasattr(self, '_performance_stats'):
                if func.__name__ not in self._performance_stats:
                    self._performance_stats[func.__name__] = []
                self._performance_stats[func.__name__].append(duration)
    return wrapper


def throttle(min_interval: float):
    """
    Decorator to throttle event handler execution.
    
    Args:
        min_interval: Minimum seconds between executions
        
    Usage:
        @throttle(1.0)  # Max once per second
        def on_step_end(self, context):
            ...
    """
    def decorator(func: Callable) -> Callable:
        last_execution = {}
        
        @wraps(func)
        def wrapper(self, context: BaseContext, *args, **kwargs):
            key = f"{self.name}.{func.__name__}"
            now = time.time()
            
            if key in last_execution:
                if now - last_execution[key] < min_interval:
                    return
            
            last_execution[key] = now
            return func(self, context, *args, **kwargs)
        
        return wrapper
    return decorator


def batch_events(batch_size: int = 10, timeout: float = 1.0):
    """
    Decorator to batch events before processing.
    
    Args:
        batch_size: Number of events to batch
        timeout: Max time to wait before processing
        
    Usage:
        @batch_events(batch_size=100, timeout=5.0)
        def on_step_end(self, context):
            ...
    """
    def decorator(func: Callable) -> Callable:
        batch = []
        last_flush = time.time()
        
        @wraps(func)
        def wrapper(self, context: BaseContext, *args, **kwargs):
            nonlocal batch, last_flush
            
            batch.append((context, args, kwargs))
            now = time.time()
            
            if len(batch) >= batch_size or now - last_flush >= timeout:
                # Process batch
                contexts = [item[0] for item in batch]
                func(self, contexts)  # Pass list of contexts
                
                batch.clear()
                last_flush = now
        
        return wrapper
    return decorator


# State management utilities

class StateManager:
    """
    Enhanced state management for callbacks.
    
    Provides persistent storage, time-series tracking, and aggregation.
    """
    
    def __init__(self, callback_name: str, state_dir: Optional[Path] = None):
        self.callback_name = callback_name
        self.state_dir = Path(state_dir) if state_dir else None
        self.logger = logging.getLogger(f"state_manager.{callback_name}")
        
        # In-memory state
        self._state: Dict[str, Any] = {}
        self._time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Load persisted state
        if self.state_dir:
            self._load_state()
    
    def set(self, key: str, value: Any) -> None:
        """Set a state value."""
        self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self._state.get(key, default)
    
    def increment(self, key: str, amount: float = 1) -> float:
        """Increment a numeric value."""
        current = self.get(key, 0)
        new_value = current + amount
        self.set(key, new_value)
        return new_value
    
    def append_time_series(self, key: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Append to time series data."""
        if timestamp is None:
            timestamp = datetime.now()
        self._time_series[key].append((timestamp, value))
    
    def get_time_series(self, key: str, last_n: Optional[int] = None) -> List[tuple[datetime, float]]:
        """Get time series data."""
        series = list(self._time_series.get(key, []))
        if last_n:
            return series[-last_n:]
        return series
    
    def update_aggregate(self, key: str, value: float) -> None:
        """Update aggregate statistics."""
        agg = self._aggregates[key]
        
        if 'count' not in agg:
            agg['count'] = 0
            agg['sum'] = 0
            agg['min'] = float('inf')
            agg['max'] = float('-inf')
            agg['values'] = deque(maxlen=1000)  # Keep last 1000 for percentiles
        
        agg['count'] += 1
        agg['sum'] += value
        agg['mean'] = agg['sum'] / agg['count']
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['values'].append(value)
        
        # Calculate std and percentiles if enough data
        if len(agg['values']) > 1:
            values = np.array(agg['values'])
            agg['std'] = float(np.std(values))
            agg['p50'] = float(np.percentile(values, 50))
            agg['p95'] = float(np.percentile(values, 95))
            agg['p99'] = float(np.percentile(values, 99))
    
    def get_aggregate(self, key: str) -> Dict[str, float]:
        """Get aggregate statistics."""
        return dict(self._aggregates.get(key, {}))
    
    def save(self) -> None:
        """Save state to disk."""
        if not self.state_dir:
            return
        
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main state
        state_file = self.state_dir / f"{self.callback_name}_state.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(self._state, f)
        
        # Save aggregates as JSON for readability
        agg_file = self.state_dir / f"{self.callback_name}_aggregates.json"
        # Convert deques to lists for JSON serialization
        json_aggregates = {}
        for key, agg in self._aggregates.items():
            json_agg = {k: v for k, v in agg.items() if k != 'values'}
            json_aggregates[key] = json_agg
        
        with open(agg_file, 'w') as f:
            json.dump(json_aggregates, f, indent=2)
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / f"{self.callback_name}_state.pkl"
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    self._state = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")


# Metric tracking utilities

class MetricTracker:
    """
    Advanced metric tracking with windowing and statistics.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._episode_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        self._metrics[metric_name].append(value)
    
    def record_episode(self, metric_name: str, value: float) -> None:
        """Record an episode-level metric."""
        self._episode_metrics[metric_name].append(value)
    
    def get_current(self, metric_name: str) -> Optional[float]:
        """Get most recent value."""
        values = self._metrics.get(metric_name)
        return values[-1] if values else None
    
    def get_mean(self, metric_name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get mean of recent values."""
        values = list(self._metrics.get(metric_name, []))
        if not values:
            return None
        
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get comprehensive statistics."""
        values = list(self._metrics.get(metric_name, []))
        if not values:
            return {}
        
        arr = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'recent_mean': float(np.mean(arr[-100:])) if len(arr) > 100 else float(np.mean(arr))
        }
    
    def get_episode_stats(self, metric_name: str) -> Dict[str, float]:
        """Get episode-level statistics."""
        values = self._episode_metrics.get(metric_name, [])
        if not values:
            return {}
        
        arr = np.array(values)
        return {
            'episodes': len(values),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'improving': self._is_improving(values)
        }
    
    def _is_improving(self, values: List[float], window: int = 100) -> bool:
        """Check if metric is improving over time."""
        if len(values) < window * 2:
            return False
        
        early_mean = np.mean(values[:window])
        recent_mean = np.mean(values[-window:])
        
        # Assume higher is better (can be customized)
        return recent_mean > early_mean * 1.05  # 5% improvement
    
    def plot_metric(self, metric_name: str, save_path: Optional[Path] = None) -> None:
        """Plot metric history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            values = list(self._metrics.get(metric_name, []))
            if not values:
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f"{metric_name} Over Time")
            plt.xlabel("Steps")
            plt.ylabel(metric_name)
            
            # Add rolling mean
            if len(values) > 50:
                rolling_mean = pd.Series(values).rolling(50).mean()
                plt.plot(rolling_mean, label='50-step MA', alpha=0.7)
                plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logging.warning("Matplotlib not available for plotting")


# Event filtering utilities

class EventRouter:
    """
    Advanced event routing and filtering.
    """
    
    def __init__(self):
        self._routes: Dict[str, List[Callable]] = defaultdict(list)
        self._filters: Dict[str, Callable] = {}
    
    def route(self, pattern: str, handler: Callable) -> None:
        """Add a route for event patterns."""
        self._routes[pattern].append(handler)
    
    def add_filter(self, name: str, filter_func: Callable) -> None:
        """Add a named filter."""
        self._filters[name] = filter_func
    
    def should_handle(self, event_type: EventType, context: BaseContext) -> bool:
        """Check if event should be handled based on filters."""
        for filter_func in self._filters.values():
            if not filter_func(event_type, context):
                return False
        return True
    
    def get_handlers(self, event_type: EventType) -> List[Callable]:
        """Get handlers matching event type."""
        handlers = []
        event_name = event_type.name
        
        for pattern, pattern_handlers in self._routes.items():
            if self._matches_pattern(event_name, pattern):
                handlers.extend(pattern_handlers)
        
        return handlers
    
    def _matches_pattern(self, event_name: str, pattern: str) -> bool:
        """Check if event name matches pattern (supports wildcards)."""
        if pattern == '*':
            return True
        
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return event_name.startswith(prefix)
        
        return event_name == pattern


# Performance profiling

class CallbackProfiler:
    """
    Detailed performance profiling for callbacks.
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._memory_usage: Dict[str, List[float]] = defaultdict(list)
    
    def profile_method(self, method_name: str, duration: float, memory_delta: Optional[float] = None) -> None:
        """Record method profiling data."""
        self._timings[method_name].append(duration)
        self._call_counts[method_name] += 1
        
        if memory_delta is not None:
            self._memory_usage[method_name].append(memory_delta)
    
    def get_profile_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        report = {
            'methods': {},
            'total_time': 0,
            'total_calls': 0
        }
        
        for method, timings in self._timings.items():
            total_time = sum(timings)
            report['methods'][method] = {
                'calls': self._call_counts[method],
                'total_time': total_time,
                'avg_time': total_time / len(timings),
                'min_time': min(timings),
                'max_time': max(timings),
                'time_std': float(np.std(timings)) if len(timings) > 1 else 0
            }
            
            if method in self._memory_usage:
                mem_deltas = self._memory_usage[method]
                report['methods'][method]['avg_memory_delta'] = np.mean(mem_deltas)
            
            report['total_time'] += total_time
            report['total_calls'] += self._call_counts[method]
        
        # Sort by total time
        report['methods'] = dict(
            sorted(report['methods'].items(), 
                   key=lambda x: x[1]['total_time'], 
                   reverse=True)
        )
        
        return report


# Common aggregation utilities

def aggregate_episode_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across episodes."""
    if not episodes:
        return {}
    
    # Extract numeric metrics
    numeric_metrics = defaultdict(list)
    
    for episode in episodes:
        for key, value in episode.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key].append(value)
    
    # Compute aggregates
    aggregates = {}
    for key, values in numeric_metrics.items():
        arr = np.array(values)
        aggregates[f"{key}_mean"] = float(np.mean(arr))
        aggregates[f"{key}_std"] = float(np.std(arr))
        aggregates[f"{key}_min"] = float(np.min(arr))
        aggregates[f"{key}_max"] = float(np.max(arr))
    
    return aggregates


def compute_rolling_statistics(
    values: List[float],
    window_sizes: List[int] = [10, 50, 100]
) -> Dict[str, List[float]]:
    """Compute rolling statistics for different window sizes."""
    if not values:
        return {}
    
    series = pd.Series(values)
    stats = {}
    
    for window in window_sizes:
        if len(values) >= window:
            stats[f"rolling_mean_{window}"] = series.rolling(window).mean().tolist()
            stats[f"rolling_std_{window}"] = series.rolling(window).std().tolist()
    
    return stats