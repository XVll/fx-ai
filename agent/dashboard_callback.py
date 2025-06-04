"""Dashboard callback for real-time training visualization."""

import logging
from typing import Dict, Any, Optional, List, Deque
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd

from agent.callbacks import BaseCallback


class DashboardCallback(BaseCallback):
    """Callback for real-time dashboard visualization.
    
    This callback:
    - Updates dashboard state in real-time
    - Tracks episode visualization data
    - Manages training progress display
    - Handles model performance metrics
    - Only performs calculations when enabled
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        dashboard_state: Optional['SharedDashboardState'] = None,
        enabled: bool = True
    ):
        """Initialize dashboard callback.
        
        Args:
            config: Dashboard configuration
            dashboard_state: Shared dashboard state object
            enabled: Whether this callback is active
        """
        super().__init__(enabled)
        
        self.config = config
        self.dashboard_state = dashboard_state
        
        # If no dashboard state provided, try to get from global
        if self.dashboard_state is None and enabled:
            try:
                from dashboard.shared_state import dashboard_state as global_state
                self.dashboard_state = global_state
            except ImportError:
                self.logger.warning("Dashboard state not available. DashboardCallback disabled.")
                self.enabled = False
                return
        
        # Episode tracking
        self.current_episode_data = {}
        self.episode_history = deque(maxlen=100)
        
        # Training progress
        self.training_start_time = None
        self.total_episodes = 0
        self.total_updates = 0
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=50)
        self.recent_lengths = deque(maxlen=50)
        
        # Update frequency control
        self.update_frequency = config.get('update_frequency', 1)
        self.chart_update_frequency = config.get('chart_update_frequency', 10)
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Initialize dashboard for training session."""
        if not self.enabled or not self.dashboard_state:
            return
        
        self.training_start_time = datetime.now()
        
        # Update session info
        self.dashboard_state.session_start_time = self.training_start_time
        self.dashboard_state.model_name = config.get('experiment_name', 'training')
        # Initialize attributes if they don't exist
        if hasattr(self.dashboard_state, 'total_episodes'):
            self.dashboard_state.total_episodes = 0
        if hasattr(self.dashboard_state, 'total_updates'):
            self.dashboard_state.total_updates = 0
        
        # Reset tracking (initialize if needed)
        if not hasattr(self.dashboard_state, 'episode_history'):
            self.dashboard_state.episode_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'training_events'):
            self.dashboard_state.training_events = deque(maxlen=50)
        
        self.dashboard_state.episode_history.clear()
        self.dashboard_state.training_events.clear()
        
        # Add training start event
        self._add_training_event("Training Started", "success")
        
        self.logger.info("Dashboard initialized for training session")
    
    def on_episode_start(self, episode_num: int, reset_info: Dict[str, Any]) -> None:
        """Initialize episode tracking."""
        if not self.enabled or not self.dashboard_state:
            return
        
        self.total_episodes = episode_num
        if hasattr(self.dashboard_state, 'total_episodes'):
            self.dashboard_state.total_episodes = episode_num
        
        # Initialize episode data
        self.current_episode_data = {
            'episode_num': episode_num,
            'start_time': datetime.now(),
            'symbol': reset_info.get('symbol', 'UNKNOWN'),
            'date': reset_info.get('date', datetime.now()),
            'reset_time': reset_info.get('reset_time', ''),
            'initial_price': reset_info.get('initial_price', 0.0),
            'prices': [],
            'actions': [],
            'positions': [],
            'rewards': [],
            'timestamps': [],
        }
        
        # Update dashboard state
        self.dashboard_state.symbol = self.current_episode_data['symbol']
        # Note: current_episode might not exist in SharedDashboardState
        if hasattr(self.dashboard_state, 'current_episode'):
            self.dashboard_state.current_episode = episode_num
        
        # Clear position for new episode
        self.dashboard_state.position_side = "FLAT"
        self.dashboard_state.position_qty = 0
        self.dashboard_state.position_pnl_dollar = 0.0
        self.dashboard_state.position_pnl_percent = 0.0
    
    def on_episode_step(self, step_data: Dict[str, Any]) -> None:
        """Track episode step data for visualization."""
        if not self.enabled or not self.dashboard_state:
            return
        
        info = step_data.get('info', {})
        
        # Update current market data
        self.dashboard_state.current_price = info.get('current_price', 0.0)
        self.dashboard_state.bid_price = info.get('bid_price', self.dashboard_state.current_price)
        self.dashboard_state.ask_price = info.get('ask_price', self.dashboard_state.current_price)
        self.dashboard_state.spread = self.dashboard_state.ask_price - self.dashboard_state.bid_price
        self.dashboard_state.spread_pct = (self.dashboard_state.spread / 
                                         max(0.01, self.dashboard_state.current_price)) * 100
        
        # Update position data
        position = info.get('position', 0)
        if position > 0:
            self.dashboard_state.position_side = "LONG"
            self.dashboard_state.position_qty = position
        elif position < 0:
            self.dashboard_state.position_side = "SHORT"
            self.dashboard_state.position_qty = abs(position)
        else:
            self.dashboard_state.position_side = "FLAT"
            self.dashboard_state.position_qty = 0
        
        # Update portfolio
        self.dashboard_state.total_equity = info.get('total_equity', 100000.0)
        self.dashboard_state.cash_balance = info.get('cash', 100000.0)
        self.dashboard_state.unrealized_pnl = info.get('unrealized_pnl', 0.0)
        
        # Track episode data
        self.current_episode_data['prices'].append(self.dashboard_state.current_price)
        self.current_episode_data['actions'].append(step_data.get('action', 0))
        self.current_episode_data['positions'].append(position)
        self.current_episode_data['rewards'].append(step_data.get('reward', 0.0))
        self.current_episode_data['timestamps'].append(info.get('timestamp', datetime.now()))
        
        # Update time
        if 'timestamp' in info:
            self.dashboard_state.current_timestamp = info['timestamp']
            self.dashboard_state.ny_time = info['timestamp'].strftime('%H:%M:%S')
    
    def on_episode_end(self, episode_num: int, episode_data: Dict[str, Any]) -> None:
        """Update dashboard with episode results."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Extract episode metrics
        episode_reward = episode_data.get('episode_reward', 0.0)
        episode_length = episode_data.get('episode_length', 0)
        
        # Update tracking
        self.recent_rewards.append(episode_reward)
        self.recent_lengths.append(episode_length)
        
        # Calculate running averages
        mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        mean_length = np.mean(self.recent_lengths) if self.recent_lengths else 0.0
        
        # Update dashboard state
        self.dashboard_state.episode_reward = episode_reward
        self.dashboard_state.episode_length = episode_length
        self.dashboard_state.mean_episode_reward = mean_reward
        self.dashboard_state.mean_episode_length = mean_length
        
        # Add episode to history
        episode_summary = {
            'episode': episode_num,
            'reward': episode_reward,
            'length': episode_length,
            'final_equity': episode_data.get('final_equity', 100000.0),
            'num_trades': episode_data.get('num_trades', 0),
            'symbol': self.current_episode_data.get('symbol', 'UNKNOWN'),
            'date': self.current_episode_data.get('date', datetime.now()),
        }
        
        self.dashboard_state.episode_history.append(episode_summary)
        
        # Update episode visualization data if needed
        if episode_num % self.chart_update_frequency == 0:
            self._update_episode_chart_data()
        
        # Add event for significant episodes
        if episode_reward > mean_reward * 1.5:
            self._add_training_event(
                f"High reward episode: {episode_reward:.2f}",
                "success"
            )
        elif episode_data.get('termination_reason') == 'bankruptcy':
            self._add_training_event(
                f"Episode {episode_num} ended in bankruptcy",
                "danger"
            )
    
    def on_update_start(self, update_num: int) -> None:
        """Track PPO update start."""
        if not self.enabled or not self.dashboard_state:
            return
        
        self.total_updates = update_num
        if hasattr(self.dashboard_state, 'total_updates'):
            self.dashboard_state.total_updates = update_num
        if hasattr(self.dashboard_state, 'is_updating'):
            self.dashboard_state.is_updating = True
    
    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Update dashboard with PPO metrics."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Update loss metrics
        self.dashboard_state.policy_loss = update_metrics.get('policy_loss', 0.0)
        self.dashboard_state.value_loss = update_metrics.get('value_loss', 0.0)
        self.dashboard_state.entropy = update_metrics.get('entropy', 0.0)
        self.dashboard_state.total_loss = update_metrics.get('total_loss', 0.0)
        self.dashboard_state.learning_rate = update_metrics.get('learning_rate', 0.0)
        
        # Update training metrics
        self.dashboard_state.clip_fraction = update_metrics.get('clip_fraction', 0.0)
        self.dashboard_state.kl_divergence = update_metrics.get('kl_divergence', 0.0)
        self.dashboard_state.explained_variance = update_metrics.get('explained_variance', 0.0)
        
        # Calculate steps per second
        if self.training_start_time:
            elapsed = (datetime.now() - self.training_start_time).total_seconds()
            total_steps = update_metrics.get('total_steps', 0)
            if hasattr(self.dashboard_state, 'steps_per_second'):
                self.dashboard_state.steps_per_second = total_steps / max(1.0, elapsed)
        
        if hasattr(self.dashboard_state, 'is_updating'):
            self.dashboard_state.is_updating = False
        
        # Add update event periodically
        if update_num % 10 == 0:
            self._add_training_event(
                f"Update {update_num} completed",
                "info"
            )
    
    def on_evaluation_end(self, eval_results: Dict[str, Any]) -> None:
        """Update dashboard with evaluation results."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Update evaluation metrics
        self.dashboard_state.eval_mean_reward = eval_results.get('mean_reward', 0.0)
        self.dashboard_state.eval_win_rate = eval_results.get('win_rate', 0.0)
        self.dashboard_state.eval_sharpe_ratio = eval_results.get('sharpe_ratio', 0.0)
        
        # Add evaluation event
        self._add_training_event(
            f"Evaluation: reward={eval_results.get('mean_reward', 0):.2f}",
            "primary"
        )
    
    def on_order_filled(self, fill_data: Dict[str, Any]) -> None:
        """Update execution metrics."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Track execution quality
        slippage = fill_data.get('slippage', 0.0)
        if hasattr(self.dashboard_state, 'recent_slippage'):
            self.dashboard_state.recent_slippage.append(slippage)
            self.dashboard_state.avg_slippage = np.mean(self.dashboard_state.recent_slippage)
    
    def on_position_closed(self, trade_result: Dict[str, Any]) -> None:
        """Update trade statistics."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Update trade stats
        self.dashboard_state.session_total_trades += 1
        
        pnl = trade_result.get('pnl', 0.0)
        self.dashboard_state.realized_pnl += pnl
        
        if pnl > 0:
            self.dashboard_state.session_winning_trades += 1
        
        # Update win rate
        self.dashboard_state.win_rate = (self.dashboard_state.session_winning_trades / 
                                       max(1, self.dashboard_state.session_total_trades))
        
        # Add trade to recent trades (initialize if needed)
        if not hasattr(self.dashboard_state, 'recent_trades'):
            self.dashboard_state.recent_trades = deque(maxlen=20)
            
        self.dashboard_state.recent_trades.append({
            'timestamp': datetime.now(),
            'symbol': trade_result.get('symbol', 'UNKNOWN'),
            'side': trade_result.get('side', 'unknown'),
            'pnl': pnl,
            'return_pct': trade_result.get('return_pct', 0.0),
        })
    
    def on_model_forward(self, forward_data: Dict[str, Any]) -> None:
        """Update model internals display."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Update action probabilities (if attribute exists)
        if 'action_probs' in forward_data and hasattr(self.dashboard_state, 'action_probabilities'):
            self.dashboard_state.action_probabilities = forward_data['action_probs'].tolist()
        
        # Update feature stats periodically (if attribute exists)
        if (self.total_episodes % 100 == 0 and 'feature_stats' in forward_data and 
            hasattr(self.dashboard_state, 'feature_stats')):
            self.dashboard_state.feature_stats = forward_data['feature_stats']
    
    def on_momentum_day_change(self, day_info: Dict[str, Any]) -> None:
        """Update momentum training info."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Update momentum info
        date_obj = day_info.get('date')
        if hasattr(date_obj, 'strftime'):
            momentum_day = date_obj.strftime('%Y-%m-%d')
        else:
            momentum_day = str(date_obj) if date_obj else ''
        
        self.dashboard_state.current_momentum_day_date = momentum_day
        self.dashboard_state.current_momentum_day_quality = day_info.get('quality_score', 0.0)
        self.dashboard_state.curriculum_stage = str(day_info.get('curriculum_stage', 1))
        
        # Add event
        date_str = momentum_day if momentum_day else 'Unknown'
        self._add_training_event(
            f"Switched to {date_str}: quality={day_info.get('quality_score', 0):.3f}",
            "info"
        )
    
    def on_curriculum_stage_change(self, stage_info: Dict[str, Any]) -> None:
        """Update curriculum progress."""
        if not self.enabled or not self.dashboard_state:
            return
        
        self.dashboard_state.curriculum_stage = stage_info.get('stage', 1)
        self.dashboard_state.curriculum_progress = stage_info.get('progress', 0.0)
        
        # Add milestone event
        self._add_training_event(
            f"Advanced to curriculum stage {stage_info.get('stage', 1)}",
            "success"
        )
    
    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Finalize dashboard for training session."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Calculate session duration
        if self.training_start_time:
            duration = (datetime.now() - self.training_start_time).total_seconds()
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            
            self._add_training_event(
                f"Training completed: {hours}h {minutes}m, "
                f"{self.total_episodes} episodes, "
                f"{self.total_updates} updates",
                "success"
            )
        
        # Update final stats
        self.dashboard_state.training_complete = True
        self.dashboard_state.final_mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        self.dashboard_state.final_stats = final_stats
    
    def _update_episode_chart_data(self) -> None:
        """Update episode visualization data."""
        if not self.current_episode_data.get('prices'):
            return
        
        # Create episode chart data
        chart_data = {
            'timestamps': self.current_episode_data['timestamps'][-1000:],  # Last 1000 points
            'prices': self.current_episode_data['prices'][-1000:],
            'positions': self.current_episode_data['positions'][-1000:],
            'rewards': self.current_episode_data['rewards'][-1000:],
        }
        
        # Update dashboard
        self.dashboard_state.episode_chart_data = chart_data
    
    def _add_training_event(self, message: str, event_type: str = "info") -> None:
        """Add event to training event log."""
        # Initialize training_events if it doesn't exist
        if not hasattr(self.dashboard_state, 'training_events'):
            self.dashboard_state.training_events = deque(maxlen=50)
            
        event = {
            'timestamp': datetime.now(),
            'message': message,
            'type': event_type,  # success, info, warning, danger
        }
        self.dashboard_state.training_events.append(event)