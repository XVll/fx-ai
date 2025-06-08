"""
Monitoring and metrics interfaces for system observability.

These interfaces enable comprehensive monitoring, logging,
and metrics collection across all system components.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable, Callable
from datetime import datetime
from pathlib import Path
import pandas as pd
from enum import Enum

from ..types.common import (
    RunMode, EpisodeMetrics, ModelVersion,
    Configurable
)


class MetricType(Enum):
    """Types of metrics for categorization."""
    COUNTER = "COUNTER"        # Monotonically increasing
    GAUGE = "GAUGE"           # Point-in-time value
    HISTOGRAM = "HISTOGRAM"    # Distribution of values
    SUMMARY = "SUMMARY"       # Statistical summary


class LogLevel(Enum):
    """Log levels for filtering."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@runtime_checkable
class IMetric(Protocol):
    """Base interface for individual metrics.
    
    Design principles:
    - Self-describing metrics
    - Efficient collection
    - Support aggregation
    """
    
    @property
    def name(self) -> str:
        """Metric name.
        
        Returns:
            Unique identifier
        """
        ...
    
    @property
    def metric_type(self) -> MetricType:
        """Type of metric.
        
        Returns:
            Metric type
        """
        ...
    
    @property
    def labels(self) -> dict[str, str]:
        """Metric labels/tags.
        
        Returns:
            Label dictionary
            
        Design notes:
        - Used for filtering/grouping
        - Examples: mode, symbol, version
        """
        ...
    
    def record(
        self,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record metric value.
        
        Args:
            value: Metric value
            timestamp: Optional timestamp
            
        Design notes:
        - Thread-safe implementation
        - Efficient storage
        """
        ...
    
    def get_value(self) -> Any:
        """Get current metric value.
        
        Returns:
            Current value (type depends on metric)
        """
        ...


class IMetricsCollector(Configurable):
    """Interface for metrics collection.
    
    Design principles:
    - Centralized metrics management
    - Multiple backend support
    - Minimal performance impact
    """
    
    @abstractmethod
    def create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Optional[dict[str, str]] = None
    ) -> IMetric:
        """Create a metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Human-readable description
            labels: Default labels
            
        Returns:
            Metric instance
            
        Design notes:
        - Validate name uniqueness
        - Set up backend storage
        """
        ...
    
    @abstractmethod
    def record_batch(
        self,
        metrics: dict[str, float],
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """Record multiple metrics at once.
        
        Args:
            metrics: Dict of metric name to value
            labels: Common labels
            
        Design notes:
        - More efficient than individual records
        - Atomic operation
        """
        ...
    
    @abstractmethod
    def get_metrics(
        self,
        pattern: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Query metrics.
        
        Args:
            pattern: Name pattern (glob)
            labels: Label filters
            start_time: Time range start
            end_time: Time range end
            
        Returns:
            DataFrame of metrics
            
        Design notes:
        - Support flexible queries
        - Return time series data
        """
        ...
    
    @abstractmethod
    def export_metrics(
        self,
        format: str = "prometheus"
    ) -> str:
        """Export metrics in standard format.
        
        Args:
            format: Export format
            
        Returns:
            Formatted metrics
            
        Design notes:
        - Support Prometheus, StatsD, etc.
        - Include metadata
        """
        ...


class ILogger(Protocol):
    """Interface for structured logging.
    
    Design principles:
    - Structured log entries
    - Multiple output targets
    - Contextual information
    """
    
    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[dict[str, Any]] = None
    ) -> None:
        """Log a message.
        
        Args:
            level: Log level
            message: Log message
            context: Additional context
            
        Design notes:
        - Include timestamp automatically
        - Add system context
        """
        ...
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        ...
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ...
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        ...
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ...
    
    def set_context(
        self,
        context: dict[str, Any]
    ) -> None:
        """Set persistent context.
        
        Args:
            context: Context dict
            
        Design notes:
        - Added to all subsequent logs
        - Examples: mode, episode, version
        """
        ...


class IEventLogger(Protocol):
    """Interface for event logging.
    
    Design principles:
    - Track significant events
    - Enable event sourcing
    - Support replay/analysis
    """
    
    def log_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log an event.
        
        Args:
            event_type: Type of event
            event_data: Event details
            timestamp: Event time
            
        Design notes:
        - Schema per event type
        - Immutable storage
        """
        ...
    
    def query_events(
        self,
        event_types: Optional[list[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Query events.
        
        Args:
            event_types: Filter by type
            start_time: Time range start
            end_time: Time range end
            filters: Additional filters
            
        Returns:
            List of events
        """
        ...


class IMonitor(Configurable):
    """High-level monitoring interface.
    
    Design principles:
    - Coordinate metrics, logs, events
    - Provide unified interface
    - Enable dashboards/alerts
    """
    
    @abstractmethod
    def start_episode(
        self,
        episode_id: str,
        context: dict[str, Any]
    ) -> None:
        """Start episode monitoring.
        
        Args:
            episode_id: Episode identifier
            context: Episode context
            
        Design notes:
        - Set up episode-specific tracking
        - Initialize collectors
        """
        ...
    
    @abstractmethod
    def end_episode(
        self,
        episode_id: str,
        metrics: EpisodeMetrics
    ) -> None:
        """End episode monitoring.
        
        Args:
            episode_id: Episode identifier
            metrics: Final metrics
            
        Design notes:
        - Finalize tracking
        - Trigger aggregations
        """
        ...
    
    @abstractmethod
    def record_step(
        self,
        step: int,
        metrics: dict[str, float],
        mode: RunMode
    ) -> None:
        """Record training step.
        
        Args:
            step: Step number
            metrics: Step metrics
            mode: Current mode
        """
        ...
    
    @abstractmethod
    def add_alert(
        self,
        name: str,
        condition: Callable[[dict[str, float]], bool],
        message: str,
        severity: str = "warning"
    ) -> None:
        """Add monitoring alert.
        
        Args:
            name: Alert name
            condition: Alert condition
            message: Alert message
            severity: Alert severity
        """
        ...
    
    @abstractmethod
    def get_dashboard_url(self) -> Optional[str]:
        """Get monitoring dashboard URL.
        
        Returns:
            Dashboard URL if available
        """
        ...


class IProfiler(Protocol):
    """Interface for performance profiling.
    
    Design principles:
    - Profile critical paths
    - Minimal overhead
    - Actionable insights
    """
    
    def start_profiling(
        self,
        session_name: str,
        components: Optional[list[str]] = None
    ) -> None:
        """Start profiling session.
        
        Args:
            session_name: Session identifier
            components: Components to profile
        """
        ...
    
    def stop_profiling(self) -> dict[str, Any]:
        """Stop profiling and get results.
        
        Returns:
            Profiling results
        """
        ...
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> tuple[Any, dict[str, float]]:
        """Profile single function call.
        
        Args:
            func: Function to profile
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, profile_data)
        """
        ...


class IHealthChecker(Protocol):
    """Interface for system health checking.
    
    Design principles:
    - Monitor component health
    - Detect degradation
    - Enable auto-recovery
    """
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], tuple[bool, str]],
        interval_seconds: int = 60
    ) -> None:
        """Register health check.
        
        Args:
            name: Check name
            check_func: Function returning (healthy, message)
            interval_seconds: Check interval
        """
        ...
    
    def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get system health status.
        
        Returns:
            Health status by component
        """
        ...
    
    def set_recovery_action(
        self,
        check_name: str,
        action: Callable[[], None]
    ) -> None:
        """Set recovery action for failed check.
        
        Args:
            check_name: Check name
            action: Recovery function
        """
        ...
