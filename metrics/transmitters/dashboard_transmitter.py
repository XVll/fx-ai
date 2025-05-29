# metrics/transmitters/dashboard_transmitter.py - Dashboard metric transmitter

import logging
import threading
from typing import Dict, Optional, Any, List
from queue import Queue, Empty
import time

from ..core import MetricTransmitter, MetricValue, MetricFilter, MetricCategory


class DashboardTransmitter(MetricTransmitter):
    """Transmitter that sends metrics to the live dashboard"""
    
    def __init__(self, port: int = 8050, update_interval: float = 0.1):
        """
        Initialize the dashboard transmitter
        
        Args:
            port: Port to run the dashboard on
            update_interval: Interval for batching updates
        """
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.update_interval = update_interval
        
        # Dashboard instance (lazy loaded)
        self._dashboard = None
        self._is_started = False
        
        # Metric filtering
        self.filter = MetricFilter()
        self.filter.filter_categories(
            MetricCategory.TRAINING,
            MetricCategory.TRADING,
            MetricCategory.EXECUTION,
            MetricCategory.ENVIRONMENT
        )
        
        # Event queue for dashboard updates
        self._event_queue = Queue(maxsize=1000)
        self._update_thread = None
        self._stop_event = threading.Event()
        
        # Metric buffering for efficient updates
        self._metric_buffer = {}
        self._buffer_lock = threading.Lock()
        
        # Track custom events
        self._supported_events = {
            'episode_end', 'momentum_day_change', 'curriculum_progress',
            'reward_components', 'reset_point_performance', 'trade_execution',
            'training_update', 'ppo_metrics'
        }
        
    def start(self, open_browser: bool = True):
        """Start the dashboard and update thread"""
        if not self._is_started:
            try:
                # Lazy import to avoid circular dependencies
                from dashboard.dashboard import MomentumDashboard
                self._dashboard = MomentumDashboard(port=self.port)
                self._dashboard.start(open_browser=open_browser)
                
                # Start update thread
                self._stop_event.clear()
                self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
                self._update_thread.start()
                
                self._is_started = True
                self.logger.info(f"Dashboard transmitter started on port {self.port}")
            except Exception as e:
                self.logger.error(f"Failed to start dashboard transmitter: {e}")
    
    def transmit(self, metrics: Dict[str, MetricValue], step: Optional[int] = None):
        """Transmit metrics to the dashboard"""
        if not self._is_started:
            return
            
        # Filter metrics
        filtered_metrics = {}
        for name, value in metrics.items():
            parts = name.split('.')
            if len(parts) >= 1:
                category_str = parts[0]
                try:
                    category = MetricCategory(category_str)
                    if category in [MetricCategory.TRAINING, MetricCategory.TRADING, 
                                  MetricCategory.EXECUTION, MetricCategory.ENVIRONMENT]:
                        filtered_metrics[name] = value
                except ValueError:
                    pass
        
        if filtered_metrics:
            # Buffer metrics for batched updates
            with self._buffer_lock:
                self._metric_buffer.update({
                    name: {
                        'value': value.value,
                        'timestamp': value.timestamp,
                        'step': value.step or step,
                        'episode': value.episode
                    }
                    for name, value in filtered_metrics.items()
                })
    
    def on_event(self, event_name: str, event_data: Dict[str, Any]):
        """Handle custom events for dashboard-specific updates"""
        if not self._is_started or event_name not in self._supported_events:
            return
            
        try:
            self._event_queue.put_nowait({
                'type': 'event',
                'name': event_name,
                'data': event_data,
                'timestamp': time.time()
            })
        except:
            # Queue full, drop event
            pass
    
    def _update_loop(self):
        """Background thread that processes updates"""
        while not self._stop_event.is_set():
            try:
                # Process buffered metrics
                with self._buffer_lock:
                    if self._metric_buffer and self._dashboard:
                        # Convert metrics to dashboard format
                        dashboard_update = self._convert_metrics_to_dashboard_format(self._metric_buffer)
                        if dashboard_update:
                            self._dashboard.update_training_state(dashboard_update)
                        self._metric_buffer.clear()
                
                # Process events (with timeout to allow checking stop event)
                deadline = time.time() + self.update_interval
                while time.time() < deadline:
                    timeout = deadline - time.time()
                    if timeout <= 0:
                        break
                        
                    try:
                        event = self._event_queue.get(timeout=timeout)
                        if event['type'] == 'event':
                            self._process_event(event['name'], event['data'])
                    except Empty:
                        break
                        
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on errors
    
    def _convert_metrics_to_dashboard_format(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to dashboard-specific format"""
        dashboard_data = {}
        
        for metric_name, metric_data in metrics.items():
            value = metric_data['value']
            
            # Map metrics to dashboard fields
            if 'training.ppo' in metric_name:
                if 'policy_loss' in metric_name:
                    dashboard_data['policy_loss'] = value
                elif 'value_loss' in metric_name:
                    dashboard_data['value_loss'] = value
                elif 'entropy' in metric_name:
                    dashboard_data['entropy'] = value
                elif 'learning_rate' in metric_name:
                    dashboard_data['learning_rate'] = value
                elif 'clip_fraction' in metric_name:
                    dashboard_data['clip_fraction'] = value
                    
            elif 'training.episode' in metric_name:
                if 'reward' in metric_name and 'mean' in metric_name:
                    dashboard_data['mean_episode_reward'] = value
                elif 'length' in metric_name and 'mean' in metric_name:
                    dashboard_data['mean_episode_length'] = value
                    
            elif 'trading.performance' in metric_name:
                if 'total_pnl' in metric_name:
                    dashboard_data['total_pnl'] = value
                elif 'win_rate' in metric_name:
                    dashboard_data['win_rate'] = value
                elif 'sharpe_ratio' in metric_name:
                    dashboard_data['sharpe_ratio'] = value
                    
            elif 'environment.reward' in metric_name:
                if 'total' in metric_name:
                    dashboard_data['current_reward'] = value
                    
        # Add metadata if available
        if metrics:
            first_metric = next(iter(metrics.values()))
            dashboard_data['step'] = first_metric.get('step', 0)
            dashboard_data['episode'] = first_metric.get('episode', 0)
            
        return dashboard_data
    
    def _process_event(self, event_name: str, event_data: Dict[str, Any]):
        """Process custom events for dashboard"""
        if not self._dashboard:
            return
            
        try:
            if event_name == 'training_update':
                # Direct training state update
                self._dashboard.update_training_state(event_data)
                
            elif event_name == 'ppo_metrics':
                # PPO metrics update (also sent as training update)
                self._dashboard.update_training_state(event_data)
                
            elif event_name == 'episode_end':
                self._dashboard.state.update_episode_data(event_data)
                
            elif event_name == 'momentum_day_change':
                # Convert to MomentumDay object
                from dashboard.dashboard_data import MomentumDay
                momentum_day = MomentumDay(
                    date=event_data.get('date'),
                    symbol=event_data.get('symbol', 'UNKNOWN'),
                    activity_score=event_data.get('activity_score', 0.0),
                    max_intraday_move=event_data.get('max_intraday_move', 0.0),
                    volume_multiplier=event_data.get('volume_multiplier', 0.0),
                    reset_points=event_data.get('reset_points', []),
                    is_front_side=event_data.get('is_front_side', False),
                    is_back_side=event_data.get('is_back_side', False),
                    halt_count=event_data.get('halt_count', 0)
                )
                self._dashboard.update_momentum_day(momentum_day)
                
            elif event_name == 'curriculum_progress':
                progress = event_data.get('progress', 0.0)
                strategy = event_data.get('strategy')
                self._dashboard.update_curriculum_progress(progress, strategy)
                
            elif event_name == 'reward_components':
                components = event_data.get('components', {})
                self._dashboard.state.update_reward_components(components)
                
            elif event_name == 'reset_point_performance':
                reset_idx = event_data.get('reset_point_idx', 0)
                performance = event_data.get('performance_data', {})
                self._dashboard.state.update_reset_point_performance(reset_idx, performance)
                
            elif event_name == 'trade_execution':
                self._dashboard.state.update_trade_data(event_data)
                
        except Exception as e:
            self.logger.error(f"Error processing event {event_name}: {e}")
    
    def close(self):
        """Close the dashboard transmitter"""
        self.logger.info("Closing dashboard transmitter")
        
        # Stop update thread
        if self._update_thread:
            self._stop_event.set()
            self._update_thread.join(timeout=2.0)
            self._update_thread = None
        
        # Stop dashboard
        if self._is_started and self._dashboard:
            self._dashboard.stop()
            self._dashboard = None
            self._is_started = False
            
        self.logger.info("Dashboard transmitter closed")
    
    def get_url(self) -> str:
        """Get the dashboard URL"""
        return f"http://127.0.0.1:{self.port}"
    
    def is_running(self) -> bool:
        """Check if dashboard is running"""
        return self._is_started and self._dashboard is not None