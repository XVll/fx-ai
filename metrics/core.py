# metrics/core.py - Core metrics framework with clear interfaces

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
import time
from collections import defaultdict, deque
import threading


class MetricType(Enum):
    """Types of metrics for categorization"""
    COUNTER = "counter"  # Incremental counters
    GAUGE = "gauge"  # Current values
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Rate of change
    PERCENTAGE = "percentage"  # Percentage values
    CURRENCY = "currency"  # Monetary values
    TIME = "time"  # Time-based metrics
    BOOLEAN = "boolean"  # True/False metrics


class MetricCategory(Enum):
    """Categories for organizing metrics"""
    MODEL = "model"  # Model architecture, parameters, gradients
    TRAINING = "training"  # Training process, episodes, performance
    TRADING = "trading"  # Trading performance, P&L, positions
    EXECUTION = "execution"  # Order execution, fills, costs
    ENVIRONMENT = "environment"  # Environment state, rewards, actions
    SYSTEM = "system"  # System performance, timing, memory


@dataclass
class MetricMetadata:
    """Metadata for metrics with clear categorization"""
    category: MetricCategory
    metric_type: MetricType
    description: str
    unit: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    aggregation: str = "last"  # last, mean, sum, max, min
    frequency: str = "step"  # step, episode, update, manual
    enabled: bool = True


@dataclass
class MetricValue:
    """Container for metric values with timestamp"""
    value: Union[float, int, bool, str]
    timestamp: float = field(default_factory=time.time)
    step: Optional[int] = None
    episode: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricCollector(ABC):
    """Abstract base class for metric collectors"""

    def __init__(self, name: str, category: MetricCategory):
        self.name = name
        self.category = category
        self.enabled = True
        self._metrics: Dict[str, MetricMetadata] = {}

    @abstractmethod
    def collect(self) -> Dict[str, MetricValue]:
        """Collect and return current metrics"""
        pass

    def register_metric(self, name: str, metadata: MetricMetadata):
        """Register a new metric with metadata"""
        full_name = f"{self.category.value}.{self.name}.{name}"
        self._metrics[full_name] = metadata
        return full_name

    def get_metric_metadata(self, name: str) -> Optional[MetricMetadata]:
        """Get metadata for a metric"""
        return self._metrics.get(name)

    def enable(self):
        """Enable this collector"""
        self.enabled = True

    def disable(self):
        """Disable this collector"""
        self.enabled = False


class MetricTransmitter(ABC):
    """Abstract base class for metric transmitters"""

    @abstractmethod
    def transmit(self, metrics: Dict[str, MetricValue], step: Optional[int] = None):
        """Transmit metrics to the destination"""
        pass

    @abstractmethod
    def close(self):
        """Close the transmitter and cleanup resources"""
        pass


class MetricAggregator:
    """Handles metric aggregation and buffering"""

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self._lock = threading.RLock()

    def add_value(self, metric_name: str, value: MetricValue):
        """Add a value to the buffer"""
        with self._lock:
            self._buffers[metric_name].append(value)

    def get_aggregated(self, metric_name: str, aggregation: str = "last") -> Optional[float]:
        """Get aggregated value for a metric"""
        with self._lock:
            buffer = self._buffers.get(metric_name)
            if not buffer:
                return None

            values = [v.value for v in buffer if isinstance(v.value, (int, float))]
            if not values:
                return None

            if aggregation == "last":
                return values[-1]
            elif aggregation == "mean":
                return sum(values) / len(values)
            elif aggregation == "sum":
                return sum(values)
            elif aggregation == "max":
                return max(values)
            elif aggregation == "min":
                return min(values)
            else:
                return values[-1]

    def get_recent_values(self, metric_name: str, count: int = 10) -> List[MetricValue]:
        """Get recent values for a metric"""
        with self._lock:
            buffer = self._buffers.get(metric_name, deque())
            return list(buffer)[-count:]

    def clear_buffer(self, metric_name: str):
        """Clear buffer for a specific metric"""
        with self._lock:
            if metric_name in self._buffers:
                self._buffers[metric_name].clear()

    def clear_all_buffers(self):
        """Clear all buffers"""
        with self._lock:
            self._buffers.clear()


class MetricFilter:
    """Filter metrics based on various criteria"""

    def __init__(self):
        self.category_filters: List[MetricCategory] = []
        self.name_patterns: List[str] = []
        self.enabled_only: bool = True
        self.frequency_filters: List[str] = []

    def filter_categories(self, *categories: MetricCategory):
        """Filter by categories"""
        self.category_filters = list(categories)
        return self

    def filter_patterns(self, *patterns: str):
        """Filter by name patterns"""
        self.name_patterns = list(patterns)
        return self

    def filter_frequency(self, *frequencies: str):
        """Filter by frequency"""
        self.frequency_filters = list(frequencies)
        return self

    def include_disabled(self):
        """Include disabled metrics"""
        self.enabled_only = False
        return self

    def should_include(self, name: str, metadata: MetricMetadata) -> bool:
        """Check if metric should be included based on filters"""
        if self.enabled_only and not metadata.enabled:
            return False

        if self.category_filters and metadata.category not in self.category_filters:
            return False

        if self.frequency_filters and metadata.frequency not in self.frequency_filters:
            return False

        if self.name_patterns:
            for pattern in self.name_patterns:
                if pattern in name:
                    return True
            return False

        return True