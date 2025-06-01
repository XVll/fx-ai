# dashboard/shared_state.py - Shared state for dashboard with thread-safe access

import threading
from typing import Dict, Any, List, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
from .event_stream import TradingEvent, EventType, event_stream


@dataclass
class SharedDashboardState:
    """Thread-safe shared state for dashboard data"""
    
    # Session info
    session_start_time: datetime = field(default_factory=datetime.now)
    model_name: str = "MLGO_v1"
    symbol: str = "MLGO"
    
    # Market data (from event stream)
    current_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    volume: int = 0
    ny_time: str = ""
    trading_hours: str = "CLOSED"
    current_timestamp: Optional[datetime] = None
    
    # Position data (from event stream)
    position_side: str = "FLAT"
    position_qty: int = 0
    avg_entry_price: float = 0.0
    position_pnl_dollar: float = 0.0
    position_pnl_percent: float = 0.0
    
    # Portfolio data (from metrics)
    total_equity: float = 100000.0
    cash_balance: float = 100000.0
    session_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    
    # Episode info (from metrics)
    current_step: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    last_step_reward: float = 0.0
    episode_number: int = 0
    
    # Training state (from metrics)
    mode: str = "Idle"
    stage: str = "Not Started"
    overall_progress: float = 0.0
    updates: int = 0
    global_steps: int = 0
    total_episodes: int = 0
    
    # Progress tracking for UI
    rollout_steps: int = 0
    rollout_total: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    stage_status: str = ""
    max_updates: int = 0  # For training completion progress
    
    # Performance metrics (from metrics)
    steps_per_second: float = 0.0
    episodes_per_hour: float = 0.0
    updates_per_second: float = 0.0
    
    # PPO metrics (from metrics)
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    learning_rate: float = 0.0
    
    # Reward components (from metrics)
    reward_components: Dict[str, float] = field(default_factory=dict)
    episode_reward_components: Dict[str, float] = field(default_factory=dict)
    session_reward_components: Dict[str, float] = field(default_factory=dict)
    
    # Reward component counts (how many times each triggered)
    episode_reward_component_counts: Dict[str, int] = field(default_factory=dict)
    session_reward_component_counts: Dict[str, int] = field(default_factory=dict)
    
    # Time series data for charts
    price_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=3600))  # 1 hour at 1s
    ppo_metrics_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    reward_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    candle_data_1m: List[Dict[str, Any]] = field(default_factory=list)  # 1-minute OHLCV bars
    
    # Recent events for display
    recent_trades: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    recent_executions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))  # All executions
    recent_actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    action_distribution: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    
    # Trade counters
    episode_total_trades: int = 0
    episode_winning_trades: int = 0
    episode_losing_trades: int = 0
    session_total_trades: int = 0
    session_winning_trades: int = 0
    session_losing_trades: int = 0
    
    # Action tracking from metrics
    episode_action_distribution: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    session_action_distribution: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    
    # Environment info
    momentum_score: float = 0.0
    volatility: float = 0.0
    data_quality: float = 1.0
    
    # Quality metrics (from environment reset points)
    day_activity_score: float = 0.0
    volume_ratio: float = 1.0
    is_front_side: bool = False
    is_back_side: bool = False
    reset_point_quality: float = 0.0
    halt_count: int = 0
    max_intraday_move: float = 0.0
    avg_spread: float = 0.0
    
    # 2-component scores from current reset point
    current_roc_score: float = 0.0
    current_activity_score: float = 0.0
    
    # Curriculum learning metrics
    curriculum_stage: str = "stage_1_beginner"
    curriculum_progress: float = 0.0
    curriculum_min_quality: float = 0.8
    total_episodes_for_curriculum: int = 0
    roc_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    activity_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    
    # Momentum day tracking
    current_momentum_day_date: str = ""
    current_momentum_day_quality: float = 0.0
    episodes_on_current_day: int = 0
    reset_point_cycles_completed: int = 0
    total_momentum_days_used: int = 0
    
    # Reset points data for chart markers
    reset_points_data: List[Dict[str, Any]] = field(default_factory=list)
    
    # Enhanced reset point tracking
    selected_reset_point_index: int = 0
    selected_reset_point_timestamp: str = ""
    total_available_points: int = 0
    points_used_in_cycle: int = 0
    points_remaining_in_cycle: int = 0
    
    # Cycle and day switching tracking
    cycles_completed: int = 0
    target_cycles_per_day: int = 10
    cycles_remaining_for_day_switch: int = 10
    day_switch_progress_pct: float = 0.0
    
    # Enhanced curriculum tracking
    episodes_to_next_stage: int = 0
    next_stage_name: str = ""
    episodes_per_day_config: int = 10
    curriculum_strategy: str = "quality_based"


class DashboardStateManager:
    """Singleton manager for dashboard state with thread-safe operations"""
    
    _instance = None
    _lock = threading.Lock()
    _state: SharedDashboardState = None
    _update_callbacks: List[callable] = []
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._state = SharedDashboardState()
                # Subscribe to event stream
                event_stream.subscribe(cls._instance._handle_event)
            return cls._instance
    
    def _handle_event(self, event: TradingEvent):
        """Handle events from the event stream"""
        with self._lock:
            if event.event_type == EventType.MARKET_UPDATE:
                data = event.data
                self._state.current_price = data.get('price', 0)
                self._state.bid_price = data.get('bid', 0)
                self._state.ask_price = data.get('ask', 0)
                self._state.spread = data.get('spread', 0)
                self._state.spread_pct = data.get('spread_pct', 0)
                self._state.volume = data.get('volume', 0)
                
                # Extract market timestamp and format as NY time
                market_timestamp = data.get('timestamp')
                if market_timestamp:
                    try:
                        # Convert to pandas timestamp
                        ts = pd.Timestamp(market_timestamp)
                        # If timezone-aware, convert to NY time
                        if ts.tz is not None:
                            ts_ny = ts.tz_convert('America/New_York')
                        else:
                            # Assume UTC and convert to NY
                            ts_ny = ts.tz_localize('UTC').tz_convert('America/New_York')
                        self._state.ny_time = ts_ny.strftime('%H:%M:%S')
                        # Store current timestamp for chart vertical line (timezone-naive)
                        self._state.current_timestamp = ts_ny.tz_localize(None)
                    except:
                        self._state.ny_time = datetime.now().strftime('%H:%M:%S')
                        self._state.current_timestamp = None
                
                # Get trading session
                self._state.trading_hours = data.get('market_session', 'MARKET')
                
                # Add to price history
                self._state.price_history.append({
                    'time': event.timestamp,
                    'price': data.get('price', 0),
                    'bid': data.get('bid', 0),
                    'ask': data.get('ask', 0),
                    'volume': data.get('volume', 0)
                })
                
            elif event.event_type == EventType.TRADE_EXECUTION:
                data = event.data
                # Format timestamp as NY time for display
                trade_timestamp = event.timestamp
                try:
                    # Convert to pandas timestamp
                    ts = pd.Timestamp(trade_timestamp)
                    # If timezone-aware, convert to NY time
                    if ts.tz is not None:
                        ts_ny = ts.tz_convert('America/New_York')
                    else:
                        # Assume UTC and convert to NY
                        ts_ny = ts.tz_localize('UTC').tz_convert('America/New_York')
                    trade_time_str = ts_ny.strftime('%H:%M:%S')
                    # Store timezone-naive version for chart
                    trade_timestamp_chart = ts_ny.tz_localize(None)
                except:
                    trade_time_str = str(trade_timestamp)
                    trade_timestamp_chart = trade_timestamp
                
                # Check if this is a completed trade or just an execution
                is_completed_trade = data.get('is_completed_trade', False)
                
                if is_completed_trade:
                    # This is a completed trade (full round-trip from entry to exit)
                    entry_ts = data.get('entry_timestamp')
                    exit_ts = data.get('exit_timestamp')
                    entry_time_str = self._format_timestamp_ny(entry_ts)
                    exit_time_str = self._format_timestamp_ny(exit_ts)
                    hold_time_seconds = data.get('holding_time_seconds', 0)
                    hold_time_minutes = hold_time_seconds / 60
                    
                    pnl = data.get('pnl', 0)
                    self._state.recent_trades.append({
                        'entry_time': entry_time_str,
                        'exit_time': exit_time_str,
                        'side': data.get('side'),
                        'quantity': data.get('quantity'),
                        'entry_price': data.get('price'),
                        'exit_price': data.get('fill_price'),
                        'pnl': pnl,
                        'hold_time': f"{hold_time_minutes:.1f}m" if hold_time_minutes else "N/A",
                        'status': 'CLOSED'
                    })
                    
                    # Update trade counters
                    self._state.episode_total_trades += 1
                    self._state.session_total_trades += 1
                    
                    if pnl > 0:
                        self._state.episode_winning_trades += 1
                        self._state.session_winning_trades += 1
                    elif pnl < 0:
                        self._state.episode_losing_trades += 1
                        self._state.session_losing_trades += 1
                else:
                    # This is just an execution (individual fill), store it separately
                    self._state.recent_executions.append({
                        'timestamp': trade_time_str,
                        'timestamp_raw': trade_timestamp_chart,
                        'side': data.get('side'),
                        'quantity': data.get('quantity'),
                        'price': data.get('price'),
                        'fill_price': data.get('fill_price'),
                        'pnl': data.get('pnl', 0),
                        'closes_position': data.get('closes_position', False)
                    })
                
            elif event.event_type == EventType.POSITION_UPDATE:
                data = event.data
                self._state.position_side = data.get('side', 'FLAT')
                self._state.position_qty = data.get('quantity', 0)
                self._state.avg_entry_price = data.get('avg_price', 0)
                self._state.unrealized_pnl = data.get('unrealized_pnl', 0)
                self._state.realized_pnl = data.get('realized_pnl', 0)
                
                # USE PORTFOLIO SIMULATOR'S CALCULATED VALUES - DO NOT RECALCULATE
                # Get market_value from position update (which includes proper P&L calculation)
                market_value = data.get('market_value', 0)
                self._state.position_pnl_dollar = self._state.unrealized_pnl  # Use portfolio simulator's unrealized P&L
                
                # Calculate percentage only if we have valid entry value
                if self._state.position_qty > 0 and self._state.avg_entry_price > 0:
                    entry_value = self._state.avg_entry_price * self._state.position_qty
                    self._state.position_pnl_percent = (self._state.position_pnl_dollar / entry_value) * 100 if entry_value > 0 else 0
                else:
                    self._state.position_pnl_percent = 0
                    
            elif event.event_type == EventType.ACTION_DECISION:
                data = event.data
                action = data.get('action', 'HOLD')
                self._state.recent_actions.append({
                    'timestamp': event.timestamp,
                    'action': action,
                    'confidence': data.get('confidence', 0),
                    'reasoning': data.get('reasoning', {})
                })
                
                # Update both episode and general action distributions
                if action in self._state.action_distribution:
                    self._state.action_distribution[action] += 1
                if action in self._state.episode_action_distribution:
                    self._state.episode_action_distribution[action] += 1
                    # Debug log action updates
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Action {action} recorded. Episode counts: {self._state.episode_action_distribution}, Session total: {self._state.action_distribution}")
        
        # Notify callbacks
        self._notify_callbacks()
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update state from metrics system"""
        with self._lock:
            # Training metrics
            if 'mode' in metrics:
                self._state.mode = metrics['mode']
            if 'stage' in metrics:
                self._state.stage = metrics['stage']
            if 'overall_progress' in metrics:
                self._state.overall_progress = metrics['overall_progress']
                
            # Performance metrics
            for key in ['steps_per_second', 'episodes_per_hour', 'updates_per_second',
                       'global_steps', 'total_episodes', 'updates']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
                    
            # Progress tracking metrics
            for key in ['rollout_steps', 'rollout_total', 'current_epoch', 'total_epochs',
                       'current_batch', 'total_batches', 'stage_status', 'max_updates']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
            
            # PPO metrics
            ppo_updated = False
            for key in ['policy_loss', 'value_loss', 'entropy', 'clip_fraction', 
                       'approx_kl', 'learning_rate']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
                    ppo_updated = True
                    
            if ppo_updated:
                self._state.ppo_metrics_history.append({
                    'timestamp': datetime.now(),
                    'policy_loss': self._state.policy_loss,
                    'value_loss': self._state.value_loss,
                    'entropy': self._state.entropy
                })
            
            # Trading metrics
            for key in ['total_equity', 'cash_balance', 'session_pnl', 'realized_pnl',
                       'unrealized_pnl', 'max_drawdown', 'sharpe_ratio', 'win_rate']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
            
            # Episode metrics
            for key in ['current_step', 'max_steps', 'cumulative_reward', 
                       'last_step_reward', 'episode_number']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
                    
            # Momentum day tracking metrics
            for key in ['current_momentum_day_date', 'current_momentum_day_quality',
                       'episodes_on_current_day', 'reset_point_cycles_completed',
                       'total_momentum_days_used']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
            
            # Enhanced reset point tracking
            for key in ['selected_reset_point_index', 'selected_reset_point_timestamp',
                       'total_available_points', 'points_used_in_cycle', 'points_remaining_in_cycle']:
                if key in metrics:
                    print(f"DEBUG SHARED STATE: Updating {key} = {metrics[key]}")
                    setattr(self._state, key, metrics[key])
            
            # Cycle and day switching tracking  
            for key in ['cycles_completed', 'target_cycles_per_day', 'cycles_remaining_for_day_switch',
                       'day_switch_progress_pct']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
            
            # Enhanced curriculum tracking
            for key in ['episodes_to_next_stage', 'next_stage_name', 'episodes_per_day_config',
                       'curriculum_strategy']:
                if key in metrics:
                    print(f"DEBUG SHARED STATE: Updating curriculum {key} = {metrics[key]}")
                    setattr(self._state, key, metrics[key])
                    
            # Reward components (current step)
            if 'reward_components' in metrics:
                self._state.reward_components = metrics['reward_components']
                # Also update episode components (accumulate)
                for component, value in metrics['reward_components'].items():
                    if component not in self._state.episode_reward_components:
                        self._state.episode_reward_components[component] = 0.0
                    self._state.episode_reward_components[component] += value
                    
                    # Update session components (accumulate)
                    if component not in self._state.session_reward_components:
                        self._state.session_reward_components[component] = 0.0
                    self._state.session_reward_components[component] += value
                    
                    # Track component counts (only if value is non-zero)
                    if abs(value) > 1e-8:  # Small threshold to avoid counting tiny floating point errors
                        if component not in self._state.episode_reward_component_counts:
                            self._state.episode_reward_component_counts[component] = 0
                        self._state.episode_reward_component_counts[component] += 1
                        
                        if component not in self._state.session_reward_component_counts:
                            self._state.session_reward_component_counts[component] = 0
                        self._state.session_reward_component_counts[component] += 1
                
            # Action tracking from metrics (session-level from execution collector)
            action_updated = False
            if 'execution.environment.action_hold_count' in metrics:
                self._state.session_action_distribution['HOLD'] = int(metrics['execution.environment.action_hold_count'])
                action_updated = True
            if 'execution.environment.action_buy_count' in metrics:
                self._state.session_action_distribution['BUY'] = int(metrics['execution.environment.action_buy_count'])
                action_updated = True
            if 'execution.environment.action_sell_count' in metrics:
                self._state.session_action_distribution['SELL'] = int(metrics['execution.environment.action_sell_count'])
                action_updated = True
                
            # Episode action tracking (from episode events)
            if 'episode_action_hold_count' in metrics:
                self._state.episode_action_distribution['HOLD'] = int(metrics['episode_action_hold_count'])
                action_updated = True
            if 'episode_action_buy_count' in metrics:
                self._state.episode_action_distribution['BUY'] = int(metrics['episode_action_buy_count'])
                action_updated = True
            if 'episode_action_sell_count' in metrics:
                self._state.episode_action_distribution['SELL'] = int(metrics['episode_action_sell_count'])
                action_updated = True
                
            # Debug logging for action updates
            if action_updated:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Action counts updated - Episode: {self._state.episode_action_distribution}, Session: {self._state.session_action_distribution}")
                
            # Update reward history
            if 'mean_episode_reward' in metrics:
                self._state.reward_history.append({
                    'timestamp': datetime.now(),
                    'value': metrics['mean_episode_reward']
                })
        
        # Notify callbacks
        self._notify_callbacks()
    
    def update_quality_metrics(self, metrics: Dict[str, Any]):
        """Update quality metrics from reset point data"""
        with self._lock:
            # Update quality metrics if provided
            for key in ['day_activity_score', 'volume_ratio', 'halt_count', 
                       'is_front_side', 'is_back_side', 'reset_point_quality',
                       'max_intraday_move', 'avg_spread',
                       'current_roc_score', 'current_activity_score']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
            
            # Update curriculum metrics if provided
            for key in ['curriculum_stage', 'curriculum_progress', 
                       'curriculum_min_quality', 'total_episodes_for_curriculum',
                       'roc_range', 'activity_range']:
                if key in metrics:
                    setattr(self._state, key, metrics[key])
        
        # Notify callbacks
        self._notify_callbacks()
    
    def reset_episode_counters(self):
        """Reset episode-level counters at episode start"""
        with self._lock:
            self._state.episode_total_trades = 0
            self._state.episode_winning_trades = 0
            self._state.episode_losing_trades = 0
            self._state.episode_action_distribution = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    
    def get_state(self) -> SharedDashboardState:
        """Get current state (thread-safe copy)"""
        with self._lock:
            # For now, return reference (could deep copy if needed)
            return self._state
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a dictionary snapshot of current state"""
        with self._lock:
            return {
                # Basic info
                'model_name': self._state.model_name,
                'symbol': self._state.symbol,
                'mode': self._state.mode,
                'stage': self._state.stage,
                
                # Market data
                'current_price': self._state.current_price,
                'bid_price': self._state.bid_price,
                'ask_price': self._state.ask_price,
                'spread': self._state.spread,
                'spread_pct': self._state.spread_pct,
                'volume': self._state.volume,
                
                # Position
                'position_side': self._state.position_side,
                'position_qty': self._state.position_qty,
                'position_pnl_dollar': self._state.position_pnl_dollar,
                'position_pnl_percent': self._state.position_pnl_percent,
                
                # Portfolio
                'total_equity': self._state.total_equity,
                'session_pnl': self._state.session_pnl,
                'win_rate': self._state.win_rate,
                'sharpe_ratio': self._state.sharpe_ratio,
                
                # Training
                'global_steps': self._state.global_steps,
                'total_episodes': self._state.total_episodes,
                'policy_loss': self._state.policy_loss,
                'value_loss': self._state.value_loss,
                'entropy': self._state.entropy,
                
                # Momentum day tracking
                'current_momentum_day_date': self._state.current_momentum_day_date,
                'current_momentum_day_quality': self._state.current_momentum_day_quality,
                'episodes_on_current_day': self._state.episodes_on_current_day,
                'reset_point_cycles_completed': self._state.reset_point_cycles_completed,
                'total_momentum_days_used': self._state.total_momentum_days_used,
                
                # Time series lengths
                'price_history_length': len(self._state.price_history),
                'trades_count': len(self._state.recent_trades),
                'actions_count': len(self._state.recent_actions)
            }
    
    def register_callback(self, callback: callable):
        """Register a callback for state updates"""
        with self._lock:
            self._update_callbacks.append(callback)
            
    def unregister_callback(self, callback: callable):
        """Unregister a callback"""
        with self._lock:
            if callback in self._update_callbacks:
                self._update_callbacks.remove(callback)
                
    def _format_timestamp_ny(self, timestamp) -> str:
        """Format timestamp as NY time string"""
        if timestamp is None:
            return ""
        try:
            ts = pd.Timestamp(timestamp)
            if ts.tz is not None:
                ts_ny = ts.tz_convert('America/New_York')
            else:
                ts_ny = ts.tz_localize('UTC').tz_convert('America/New_York')
            return ts_ny.strftime('%H:%M:%S')
        except:
            return str(timestamp)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks of state change"""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception as e:
                # Log but don't crash
                pass
    
    def reset(self):
        """Reset state to defaults"""
        with self._lock:
            self._state = SharedDashboardState()
            
    def reset_episode(self):
        """Reset episode-level data while preserving session data"""
        with self._lock:
            # Log current episode action counts before reset
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Resetting episode - Previous episode actions: {self._state.episode_action_distribution}")
            
            self._state.episode_reward_components = {}
            self._state.episode_reward_component_counts = {}
            self._state.episode_action_distribution = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
            # Clear trade markers but keep candle data
            self._state.recent_trades.clear()
            # Clear executions for current episode only
            self._state.recent_executions.clear()
            # Don't reset action_distribution here - it's managed by event stream
            
            logger.debug(f"Episode reset complete - New episode actions: {self._state.episode_action_distribution}")
            
    def update_candle_data(self, candle_data_1m: List[Dict[str, Any]]):
        """Update the 1-minute candle data for the chart"""
        with self._lock:
            self._state.candle_data_1m = candle_data_1m
            
    def update_reset_points_data(self, reset_points_data: List[Dict[str, Any]]):
        """Update the reset points data for chart markers"""
        with self._lock:
            self._state.reset_points_data = reset_points_data


# Global instance
dashboard_state = DashboardStateManager()