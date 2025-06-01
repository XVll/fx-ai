# metrics/transmitters/dashboard_transmitter.py - Dashboard metric transmitter

import logging
import threading
from typing import Dict, Optional, Any, List
from queue import Queue, Empty
import time

from ..core import MetricTransmitter, MetricValue, MetricFilter, MetricCategory
from dashboard.shared_state import dashboard_state


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
        
        # We now use shared state instead of dashboard instance
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
        self._dashboard_thread = None
        self._stop_event = threading.Event()
        
        # Metric buffering for efficient updates
        self._metric_buffer = {}
        self._buffer_lock = threading.Lock()
        
        # Track custom events
        self._supported_events = {
            'episode_end', 'momentum_day_change', 'curriculum_progress',
            'reward_components', 'reset_point_performance', 'trade_execution',
            'training_update', 'ppo_metrics', 'episode_actions',
            'reset_point_selection', 'cycle_completion', 'curriculum_detail'
        }
        
    def start(self, open_browser: bool = True):
        """Start the dashboard transmitter and dashboard server"""
        if not self._is_started:
            try:
                # Start update thread
                self._stop_event.clear()
                self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
                self._update_thread.start()
                
                # Start the actual dashboard server
                from dashboard import start_dashboard
                self._dashboard_thread = threading.Thread(
                    target=lambda: start_dashboard(port=self.port, open_browser=open_browser),
                    daemon=True
                )
                self._dashboard_thread.start()
                
                self._is_started = True
                self.logger.info(f"Dashboard transmitter started")
                self.logger.info(f"Dashboard server starting on port {self.port}")
                
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
                    if self._metric_buffer:
                        # Convert metrics to dashboard format and update shared state
                        dashboard_update = self._convert_metrics_to_dashboard_format(self._metric_buffer)
                        if dashboard_update:
                            # Debug logging for actions and rewards
                            action_metrics = [k for k in dashboard_update.keys() if 'action' in k]
                            reward_metrics = [k for k in dashboard_update.keys() if 'reward' in k]
                            if action_metrics:
                                self.logger.debug(f"Transmitting action metrics: {action_metrics}")
                            if reward_metrics:
                                self.logger.debug(f"Transmitting reward metrics: {reward_metrics}")
                                
                            dashboard_state.update_metrics(dashboard_update)
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
                elif 'kl_divergence' in metric_name:
                    dashboard_data['kl_divergence'] = value
                    
            elif 'training.process' in metric_name:
                # Training process metrics
                if 'episode_count' in metric_name:
                    dashboard_data['total_episodes'] = value  # Global episode count across all sessions
                elif 'global_step' in metric_name:
                    dashboard_data['global_steps'] = value
                elif 'update_count' in metric_name:
                    dashboard_data['updates'] = value
                elif 'episodes_per_hour' in metric_name:
                    dashboard_data['episodes_per_hour'] = value
                elif 'steps_per_second' in metric_name:
                    dashboard_data['steps_per_second'] = value
                elif 'updates_per_hour' in metric_name:
                    dashboard_data['updates_per_second'] = value / 3600  # Convert to per second
                    
            elif 'training.episode' in metric_name:
                if 'reward' in metric_name and 'mean' in metric_name:
                    dashboard_data['mean_episode_reward'] = value
                elif 'length' in metric_name and 'mean' in metric_name:
                    dashboard_data['mean_episode_length'] = value
                    
            elif 'trading.portfolio' in metric_name:
                # Portfolio metrics
                if 'total_equity' in metric_name:
                    dashboard_data['total_equity'] = value
                elif 'cash_balance' in metric_name:
                    dashboard_data['cash_balance'] = value
                elif 'realized_pnl_session' in metric_name:
                    dashboard_data['session_pnl'] = value
                    dashboard_data['realized_pnl'] = value
                elif 'unrealized_pnl' in metric_name:
                    dashboard_data['unrealized_pnl'] = value
                elif 'max_drawdown_pct' in metric_name:
                    dashboard_data['max_drawdown'] = value / 100.0  # Convert to decimal
                elif 'sharpe_ratio' in metric_name:
                    dashboard_data['sharpe_ratio'] = value
                    
            elif 'trading.trades' in metric_name:
                if 'win_rate' in metric_name:
                    dashboard_data['win_rate'] = value / 100.0  # Convert to decimal
                    
            elif 'trading.performance' in metric_name:
                if 'total_pnl' in metric_name:
                    dashboard_data['total_pnl'] = value
                elif 'win_rate' in metric_name:
                    dashboard_data['win_rate'] = value
                elif 'sharpe_ratio' in metric_name:
                    dashboard_data['sharpe_ratio'] = value
                    
            elif 'execution.environment' in metric_name or 'environment.environment' in metric_name:
                # Action counts
                if 'action_hold_count' in metric_name:
                    dashboard_data['execution.environment.action_hold_count'] = value
                elif 'action_buy_count' in metric_name:
                    dashboard_data['execution.environment.action_buy_count'] = value
                elif 'action_sell_count' in metric_name:
                    dashboard_data['execution.environment.action_sell_count'] = value
                # Environment episode metrics
                elif 'current_step' in metric_name:
                    dashboard_data['current_step'] = value
                elif 'max_steps' in metric_name:
                    dashboard_data['max_steps'] = value
                elif 'cumulative_reward' in metric_name:
                    dashboard_data['cumulative_reward'] = value
                elif 'step_reward' in metric_name:
                    dashboard_data['last_step_reward'] = value
                elif 'episode_number' in metric_name:
                    dashboard_data['episode_number'] = value
                # Reward components
                elif 'reward_' in metric_name:
                    # Pass through reward component metrics
                    dashboard_data[metric_name] = value
                    
            elif 'environment.reward' in metric_name:
                if 'total' in metric_name:
                    dashboard_data['current_reward'] = value
                # Pass through all environment reward metrics for components
                elif 'reward_' in metric_name:
                    dashboard_data[metric_name] = value
        
        # Add metadata if available
        if metrics:
            first_metric = next(iter(metrics.values()))
            dashboard_data['step'] = first_metric.get('step', 0)
            dashboard_data['episode'] = first_metric.get('episode', 0)
            
        return dashboard_data
    
    def _process_event(self, event_name: str, event_data: Dict[str, Any]):
        """Process custom events for dashboard"""
        try:
            if event_name in ['training_update', 'ppo_metrics']:
                # Update metrics in shared state
                dashboard_state.update_metrics(event_data)
                
            elif event_name == 'episode_end':
                # Extract episode data
                episode_metrics = {
                    'episode_number': event_data.get('episode_number', 0),
                    'cumulative_reward': event_data.get('episode_reward', 0),
                    'episode_length': event_data.get('episode_length', 0)
                }
                dashboard_state.update_metrics(episode_metrics)
                # Reset episode-level data for next episode
                dashboard_state.reset_episode()
                
            elif event_name == 'reward_components':
                components = event_data.get('components', {})
                dashboard_state.update_metrics({'reward_components': components})
                
                
            elif event_name == 'episode_actions':
                # Handle episode action counts
                dashboard_state.update_metrics(event_data)
                
            elif event_name == 'curriculum_progress':
                # Handle curriculum learning progression
                curriculum_data = {
                    'curriculum_stage': event_data.get('stage', 'stage_1_beginner'),
                    'curriculum_progress': event_data.get('stage_progress', event_data.get('progress', 0.0)),
                    'curriculum_min_quality': event_data.get('min_quality', 0.8),
                    'total_episodes_for_curriculum': event_data.get('total_episodes', 0),
                    'min_roc_score': event_data.get('min_roc_score', 0.0),
                    'min_activity_score': event_data.get('min_activity_score', 0.0),
                    'min_direction_score': event_data.get('min_direction_score', 0.0)
                }
                dashboard_state.update_metrics(curriculum_data)
                
            elif event_name == 'momentum_day_change':
                # Handle momentum day changes - extract from nested day_info structure
                day_info = event_data.get('day_info', {})
                reset_points = event_data.get('reset_points', [])
                print(f"DEBUG MOMENTUM EVENT: day_info = {day_info}")
                
                # Update momentum day tracking metrics
                momentum_data = {
                    'current_momentum_day_date': day_info.get('day_date', ''),
                    'current_momentum_day_quality': day_info.get('day_quality', 0.0),
                    'episodes_on_current_day': day_info.get('episodes_on_day', 0),
                    'reset_point_cycles_completed': day_info.get('cycles_completed', 0),
                    'total_momentum_days_used': day_info.get('total_days_used', 0)
                }
                print(f"DEBUG MOMENTUM SENDING: {momentum_data}")
                dashboard_state.update_metrics(momentum_data)
                
                # Update reset points data for chart markers
                if reset_points:
                    dashboard_state.update_reset_points_data(reset_points)
                    print(f"DEBUG MOMENTUM: Updated {len(reset_points)} reset points")
                    
            elif event_name == 'reset_point_selection':
                # Handle reset point selection tracking
                print(f"DEBUG DASHBOARD TRANSMITTER: Received reset_point_selection event: {event_data}")
                reset_point_data = {
                    'selected_reset_point_index': event_data.get('selected_index', 0),
                    'selected_reset_point_timestamp': event_data.get('selected_timestamp', ''),
                    'total_available_points': event_data.get('total_available_points', 0),
                    'points_used_in_cycle': event_data.get('points_used_in_cycle', 0),
                    'points_remaining_in_cycle': event_data.get('points_remaining_in_cycle', 0)
                }
                print(f"DEBUG DASHBOARD TRANSMITTER: Updating with reset point data: {reset_point_data}")
                dashboard_state.update_metrics(reset_point_data)
                
            elif event_name == 'cycle_completion':
                # Handle cycle completion tracking
                print(f"DEBUG DASHBOARD TRANSMITTER: Received cycle_completion event: {event_data}")
                cycle_data = {
                    'cycles_completed': event_data.get('cycles_completed', 0),
                    'target_cycles_per_day': event_data.get('target_cycles_per_day', 10),
                    'cycles_remaining_for_day_switch': event_data.get('cycles_remaining_for_day_switch', 10),
                    'day_switch_progress_pct': event_data.get('day_switch_progress_pct', 0.0),
                    'episodes_on_current_day': event_data.get('episodes_on_current_day', 0)
                }
                print(f"DEBUG DASHBOARD TRANSMITTER: Updating with cycle data: {cycle_data}")
                dashboard_state.update_metrics(cycle_data)
                
            elif event_name == 'curriculum_detail':
                # Handle enhanced curriculum tracking
                print(f"DEBUG DASHBOARD TRANSMITTER: Received curriculum_detail event: {event_data}")
                curriculum_detail_data = {
                    'episodes_to_next_stage': event_data.get('episodes_to_next_stage', 0),
                    'next_stage_name': event_data.get('next_stage_name', ''),
                    'episodes_per_day_config': event_data.get('episodes_per_day_config', 10),
                    'curriculum_strategy': event_data.get('curriculum_strategy', 'quality_based')
                }
                print(f"DEBUG DASHBOARD TRANSMITTER: Updating with curriculum detail data: {curriculum_detail_data}")
                dashboard_state.update_metrics(curriculum_detail_data)
                
            # Other events can be handled as needed
                
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
        
        self._is_started = False
            
        self.logger.info("Dashboard transmitter closed")
    
    def get_url(self) -> str:
        """Get the dashboard URL"""
        return f"http://127.0.0.1:{self.port}"
    
    def is_running(self) -> bool:
        """Check if dashboard is running"""
        return self._is_started and self._dashboard_thread is not None