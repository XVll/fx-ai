# dashboard/shared_state.py - Shared state for dashboard with thread-safe access

import threading
from typing import Dict, Any, List, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
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
    
    # Time series data for charts
    price_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=3600))  # 1 hour at 1s
    ppo_metrics_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    reward_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    candle_data_1m: List[Dict[str, Any]] = field(default_factory=list)  # 1-minute OHLCV bars
    
    # Recent events for display
    recent_trades: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    recent_actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    action_distribution: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    
    # Action tracking from metrics
    episode_action_distribution: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    session_action_distribution: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    
    # Environment info
    momentum_score: float = 0.0
    volatility: float = 0.0
    data_quality: float = 1.0


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
                self._state.recent_trades.append({
                    'timestamp': event.timestamp,
                    'side': data.get('side'),
                    'quantity': data.get('quantity'),
                    'price': data.get('price'),
                    'fill_price': data.get('fill_price'),
                    'pnl': data.get('pnl', 0)
                })
                
            elif event.event_type == EventType.POSITION_UPDATE:
                data = event.data
                self._state.position_side = data.get('side', 'FLAT')
                self._state.position_qty = data.get('quantity', 0)
                self._state.avg_entry_price = data.get('avg_price', 0)
                self._state.unrealized_pnl = data.get('unrealized_pnl', 0)
                self._state.realized_pnl = data.get('realized_pnl', 0)
                
                # Calculate position P&L
                if self._state.position_qty > 0 and self._state.avg_entry_price > 0:
                    if self._state.position_side == 'LONG':
                        self._state.position_pnl_dollar = (self._state.current_price - self._state.avg_entry_price) * self._state.position_qty
                    elif self._state.position_side == 'SHORT':
                        self._state.position_pnl_dollar = (self._state.avg_entry_price - self._state.current_price) * self._state.position_qty
                    self._state.position_pnl_percent = (self._state.position_pnl_dollar / (self._state.avg_entry_price * self._state.position_qty)) * 100
                else:
                    self._state.position_pnl_dollar = 0
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
                
                # Update action distribution
                if action in self._state.action_distribution:
                    self._state.action_distribution[action] += 1
        
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
                
            # Action tracking from metrics (session-level from execution collector)
            if 'execution.environment.action_hold_count' in metrics:
                self._state.session_action_distribution['HOLD'] = int(metrics['execution.environment.action_hold_count'])
            if 'execution.environment.action_buy_count' in metrics:
                self._state.session_action_distribution['BUY'] = int(metrics['execution.environment.action_buy_count'])
            if 'execution.environment.action_sell_count' in metrics:
                self._state.session_action_distribution['SELL'] = int(metrics['execution.environment.action_sell_count'])
                
            # Episode action tracking (from episode events)
            if 'episode_action_hold_count' in metrics:
                self._state.episode_action_distribution['HOLD'] = int(metrics['episode_action_hold_count'])
            if 'episode_action_buy_count' in metrics:
                self._state.episode_action_distribution['BUY'] = int(metrics['episode_action_buy_count'])
            if 'episode_action_sell_count' in metrics:
                self._state.episode_action_distribution['SELL'] = int(metrics['episode_action_sell_count'])
                
            # Update reward history
            if 'mean_episode_reward' in metrics:
                self._state.reward_history.append({
                    'timestamp': datetime.now(),
                    'value': metrics['mean_episode_reward']
                })
        
        # Notify callbacks
        self._notify_callbacks()
    
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
            self._state.episode_reward_components = {}
            self._state.episode_action_distribution = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
            # Clear trade markers but keep candle data
            self._state.recent_trades.clear()
            
    def update_candle_data(self, candle_data_1m: List[Dict[str, Any]]):
        """Update the 1-minute candle data for the chart"""
        with self._lock:
            self._state.candle_data_1m = candle_data_1m


# Global instance
dashboard_state = DashboardStateManager()