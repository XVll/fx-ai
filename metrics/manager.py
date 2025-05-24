# metrics/manager.py - Central metrics management system

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from .core import (
    MetricCollector, MetricTransmitter, MetricAggregator, MetricFilter,
    MetricValue, MetricMetadata, MetricCategory, MetricType
)


class MetricsManager:
    """Central metrics management system"""

    def __init__(self,
                 transmit_interval: float = 1.0,
                 auto_transmit: bool = True,
                 buffer_size: int = 1000):
        """
        Initialize the metrics manager

        Args:
            transmit_interval: Interval in seconds between automatic transmissions
            auto_transmit: Whether to automatically transmit metrics
            buffer_size: Size of metric buffers
        """
        self.logger = logging.getLogger(__name__)

        # Core components
        self.collectors: Dict[str, MetricCollector] = {}
        self.transmitters: List[MetricTransmitter] = []
        self.aggregator = MetricAggregator(buffer_size)

        # Configuration
        self.transmit_interval = transmit_interval
        self.auto_transmit = auto_transmit

        # State tracking
        self.current_step = 0
        self.current_episode = 0
        self.current_update = 0
        self.is_training = False
        self.is_evaluating = False

        # Threading for auto transmission
        self._stop_auto_transmit = threading.Event()
        self._transmit_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Metrics metadata registry
        self._metric_registry: Dict[str, MetricMetadata] = {}

        # Frequency-based collection tracking
        self._last_collections: Dict[str, float] = defaultdict(float)
        
        # Visualization collector reference
        self.visualization_collector = None
        
        # Dashboard integration
        self.dashboard_collector = None
        self._dashboard_enabled = False

        self.logger.info("MetricsManager initialized")

    def register_collector(self, collector: MetricCollector) -> str:
        """Register a metric collector"""
        with self._lock:
            collector_id = f"{collector.category.value}_{collector.name}"
            self.collectors[collector_id] = collector

            # Register collector's metrics
            for metric_name, metadata in collector._metrics.items():
                self._metric_registry[metric_name] = metadata
                
            # Special handling for visualization collector
            if hasattr(collector, 'start_episode') and hasattr(collector, 'end_episode'):
                self.visualization_collector = collector

            self.logger.info(f"Registered collector: {collector_id} with {len(collector._metrics)} metrics")
            return collector_id

    def unregister_collector(self, collector_id: str):
        """Unregister a metric collector"""
        with self._lock:
            if collector_id in self.collectors:
                collector = self.collectors.pop(collector_id)
                # Remove collector's metrics from registry
                to_remove = [name for name in self._metric_registry
                             if name.startswith(f"{collector.category.value}.{collector.name}.")]
                for name in to_remove:
                    del self._metric_registry[name]

                self.logger.info(f"Unregistered collector: {collector_id}")

    def add_transmitter(self, transmitter: MetricTransmitter):
        """Add a metric transmitter"""
        with self._lock:
            self.transmitters.append(transmitter)
            self.logger.info(f"Added transmitter: {type(transmitter).__name__}")

    def remove_transmitter(self, transmitter: MetricTransmitter):
        """Remove a metric transmitter"""
        with self._lock:
            if transmitter in self.transmitters:
                self.transmitters.remove(transmitter)
                self.logger.info(f"Removed transmitter: {type(transmitter).__name__}")

    def update_state(self, step: Optional[int] = None,
                     episode: Optional[int] = None,
                     update: Optional[int] = None,
                     is_training: Optional[bool] = None,
                     is_evaluating: Optional[bool] = None):
        """Update the current state for metric collection"""
        with self._lock:
            if step is not None:
                old_step = self.current_step
                self.current_step = step
                # Debug logging for step changes
                if old_step != step and step % 50 == 0:
                    self.logger.debug(f"Step updated: {old_step} -> {step}")
            if episode is not None:
                self.current_episode = episode
            if update is not None:
                self.current_update = update
            if is_training is not None:
                self.is_training = is_training
            if is_evaluating is not None:
                self.is_evaluating = is_evaluating

    def collect_metrics(self,
                        categories: Optional[List[MetricCategory]] = None,
                        frequency: Optional[str] = None,
                        force: bool = False) -> Dict[str, MetricValue]:
        """
        Collect metrics from all registered collectors

        Args:
            categories: Specific categories to collect from
            frequency: Specific frequency to collect (step, episode, update, manual)
            force: Force collection regardless of frequency settings
        """
        collected_metrics = {}
        current_time = time.time()

        with self._lock:
            # Capture state values at the start of collection to ensure consistency
            collection_step = self.current_step
            collection_episode = self.current_episode
            
            for collector_id, collector in self.collectors.items():
                if not collector.enabled:
                    continue

                # Check if we should collect from this collector
                if categories and collector.category not in categories:
                    continue

                # Check frequency-based collection
                if not force and frequency:
                    last_collection = self._last_collections.get(collector_id, 0)
                    if frequency == "step" and current_time - last_collection < 1.0:
                        continue
                    elif frequency == "episode" and current_time - last_collection < 5.0:
                        continue
                    elif frequency == "update" and current_time - last_collection < 10.0:
                        continue

                try:
                    collector_metrics = collector.collect()
                    for name, value in collector_metrics.items():
                        # Add current state to metric value using captured values
                        value.step = collection_step
                        value.episode = collection_episode
                        value.timestamp = current_time

                        collected_metrics[name] = value

                        # Add to aggregator
                        self.aggregator.add_value(name, value)

                    self._last_collections[collector_id] = current_time

                except Exception as e:
                    self.logger.error(f"Error collecting from {collector_id}: {e}")

        return collected_metrics

    def record_metric(self, name: str, value: Any,
                      category: MetricCategory = MetricCategory.SYSTEM,
                      metric_type: MetricType = MetricType.GAUGE,
                      description: str = "",
                      unit: Optional[str] = None,
                      tags: Optional[List[str]] = None):
        """Record a single metric manually"""

        # Register metric if not exists
        full_name = f"{category.value}.manual.{name}"
        if full_name not in self._metric_registry:
            metadata = MetricMetadata(
                category=category,
                metric_type=metric_type,
                description=description,
                unit=unit,
                tags=tags or [],
                frequency="manual"
            )
            self._metric_registry[full_name] = metadata

        # Create metric value
        metric_value = MetricValue(
            value=value,
            step=self.current_step,
            episode=self.current_episode
        )

        # Add to aggregator
        self.aggregator.add_value(full_name, metric_value)

        return full_name

    def transmit_metrics(self,
                         metrics: Optional[Dict[str, MetricValue]] = None,
                         filter_obj: Optional[MetricFilter] = None):
        """Transmit metrics to all registered transmitters"""

        # Capture current step atomically before collection/transmission
        with self._lock:
            transmit_step = self.current_step

        if metrics is None:
            # Collect metrics from all collectors
            metrics = self.collect_metrics()

        if not metrics:
            return

        # Apply filters if provided
        if filter_obj:
            filtered_metrics = {}
            for name, value in metrics.items():
                metadata = self._metric_registry.get(name)
                if metadata and filter_obj.should_include(name, metadata):
                    filtered_metrics[name] = value
            metrics = filtered_metrics

        # Transmit to all transmitters using the captured step
        for transmitter in self.transmitters:
            try:
                transmitter.transmit(metrics, transmit_step)
            except Exception as e:
                self.logger.error(f"Error transmitting with {type(transmitter).__name__}: {e}")

    def get_metric_value(self, name: str, aggregation: str = "last") -> Optional[float]:
        """Get current value of a metric"""
        return self.aggregator.get_aggregated(name, aggregation)

    def get_metric_history(self, name: str, count: int = 10) -> List[MetricValue]:
        """Get recent history of a metric"""
        return self.aggregator.get_recent_values(name, count)

    def get_metric_metadata(self, name: str) -> Optional[MetricMetadata]:
        """Get metadata for a metric"""
        return self._metric_registry.get(name)

    def list_metrics(self,
                     categories: Optional[List[MetricCategory]] = None,
                     enabled_only: bool = True) -> Dict[str, MetricMetadata]:
        """List all registered metrics with optional filtering"""
        result = {}
        for name, metadata in self._metric_registry.items():
            if enabled_only and not metadata.enabled:
                continue
            if categories and metadata.category not in categories:
                continue
            result[name] = metadata
        return result

    def enable_metric(self, name: str):
        """Enable a specific metric"""
        if name in self._metric_registry:
            self._metric_registry[name].enabled = True

    def disable_metric(self, name: str):
        """Disable a specific metric"""
        if name in self._metric_registry:
            self._metric_registry[name].enabled = False

    def start_auto_transmit(self):
        """Start automatic metric transmission"""
        if self.auto_transmit and self._transmit_thread is None:
            self._stop_auto_transmit.clear()
            self._transmit_thread = threading.Thread(target=self._auto_transmit_loop, daemon=True)
            self._transmit_thread.start()
            self.logger.info("Started automatic metric transmission")

    def stop_auto_transmit(self):
        """Stop automatic metric transmission"""
        if self._transmit_thread:
            self._stop_auto_transmit.set()
            self._transmit_thread.join(timeout=5.0)
            self._transmit_thread = None
            self.logger.info("Stopped automatic metric transmission")

    def _auto_transmit_loop(self):
        """Background loop for automatic metric transmission"""
        while not self._stop_auto_transmit.wait(self.transmit_interval):
            try:
                # Only transmit if we have actual data (training has started)
                if self.current_step > 0 or self.is_training:
                    self.transmit_metrics()
            except Exception as e:
                self.logger.error(f"Error in auto transmit loop: {e}")

    def close(self):
        """Close the metrics manager and cleanup resources"""
        self.logger.info("Closing MetricsManager")

        # Stop auto transmission
        self.stop_auto_transmit()

        # Close all transmitters
        for transmitter in self.transmitters:
            try:
                transmitter.close()
            except Exception as e:
                self.logger.error(f"Error closing transmitter: {e}")

        # Clear all data
        self.collectors.clear()
        self.transmitters.clear()
        self.aggregator.clear_all_buffers()
        self._metric_registry.clear()

        self.logger.info("MetricsManager closed")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metrics system state"""
        return {
            "collectors": len(self.collectors),
            "transmitters": len(self.transmitters),
            "registered_metrics": len(self._metric_registry),
            "current_step": self.current_step,
            "current_episode": self.current_episode,
            "current_update": self.current_update,
            "is_training": self.is_training,
            "is_evaluating": self.is_evaluating,
            "auto_transmit_active": self._transmit_thread is not None
        }
        
    # Episode visualization methods
    def start_episode_visualization(self, episode_num: int, symbol: str, date: str):
        """Start collecting visualization data for a new episode"""
        if self.visualization_collector:
            self.visualization_collector.start_episode(episode_num, symbol, date)
            
    def collect_step_visualization(self, step_data: Dict[str, Any]):
        """Collect visualization data for a single step"""
        if self.visualization_collector:
            self.visualization_collector.collect_step(step_data)
            
    def collect_trade_visualization(self, trade_data: Dict[str, Any]):
        """Record a trade for visualization"""
        if self.visualization_collector:
            self.visualization_collector.collect_trade(trade_data)
            
    def end_episode_visualization(self):
        """Generate and transmit episode visualizations"""
        if self.visualization_collector:
            viz_metrics = self.visualization_collector.end_episode()
            if viz_metrics:
                # Transmit visualization metrics through all transmitters
                for transmitter in self.transmitters:
                    try:
                        transmitter.transmit(viz_metrics, self.current_step)
                    except Exception as e:
                        self.logger.error(f"Error transmitting visualizations: {e}")
                        
    # Dashboard methods
    def enable_dashboard(self, port: int = 8050, open_browser: bool = True):
        """Enable and start the live dashboard"""
        if not self._dashboard_enabled:
            try:
                from dashboard.dashboard_integration import DashboardMetricsCollector
                self.dashboard_collector = DashboardMetricsCollector()
                self.dashboard_collector.start(open_browser=open_browser)
                self._dashboard_enabled = True
                self.logger.info(f"Dashboard enabled on port {port}")
            except Exception as e:
                self.logger.error(f"Failed to start dashboard: {e}")
                
    def disable_dashboard(self):
        """Disable and stop the dashboard"""
        if self._dashboard_enabled and self.dashboard_collector:
            self.dashboard_collector.stop()
            self._dashboard_enabled = False
            self.logger.info("Dashboard disabled")
            
    def update_dashboard_step(self, step_data: Dict[str, Any]):
        """Update dashboard with step data"""
        if self._dashboard_enabled and self.dashboard_collector:
            self.dashboard_collector.on_step(step_data)
            
    def update_dashboard_trade(self, trade_data: Dict[str, Any]):
        """Update dashboard with trade data"""
        if self._dashboard_enabled and self.dashboard_collector:
            self.dashboard_collector.on_trade(trade_data)
            
    def update_dashboard_episode(self, episode_data: Dict[str, Any]):
        """Update dashboard with episode end data"""
        if self._dashboard_enabled and self.dashboard_collector:
            self.dashboard_collector.on_episode_end(episode_data)
    
    def update_dashboard_training(self, training_data: Dict[str, Any]):
        """Update dashboard with training progress"""
        if self._dashboard_enabled and self.dashboard_collector:
            self.dashboard_collector.on_training_update(training_data)
    
    def update_dashboard_ppo(self, ppo_data: Dict[str, Any]):
        """Update dashboard with PPO metrics"""
        if self._dashboard_enabled and self.dashboard_collector:
            self.dashboard_collector.on_ppo_metrics(ppo_data)