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
        
        # Handle dashboard state - it could be None, a manager, or a state object
        if dashboard_state is None and enabled:
            # Try to get from global
            try:
                from dashboard.shared_state import dashboard_state as global_state
                self.dashboard_manager = global_state  # Store the manager
                self.dashboard_state = global_state.get_state()  # Get state reference for direct access
            except ImportError:
                self.logger.warning("Dashboard state not available. DashboardCallback disabled.")
                self.enabled = False
                return
        elif dashboard_state is not None:
            # Check if it's a manager or state object
            if hasattr(dashboard_state, 'get_state'):
                # It's a manager
                self.dashboard_manager = dashboard_state
                self.dashboard_state = dashboard_state.get_state()
            else:
                # It's already a state object
                self.dashboard_state = dashboard_state
                self.dashboard_manager = None
        else:
            self.dashboard_state = None
            self.dashboard_manager = None
        
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
        
        # Initialize critical dashboard state attributes if they don't exist
        if self.enabled and self.dashboard_state is not None:
            self._initialize_dashboard_attributes()
            self._start_dashboard_server()
        self.chart_update_frequency = config.get('chart_update_frequency', 10)
        
        # Market data defaults
        if self.enabled and self.dashboard_state is not None:
            if not hasattr(self.dashboard_state, 'volume'):
                self.dashboard_state.volume = 0
    
    def _initialize_dashboard_attributes(self) -> None:
        """Initialize critical dashboard state attributes."""
        # Episode attributes
        if not hasattr(self.dashboard_state, 'episode_number'):
            self.dashboard_state.episode_number = 0
        if not hasattr(self.dashboard_state, 'current_step'):
            self.dashboard_state.current_step = 0
        if not hasattr(self.dashboard_state, 'max_steps'):
            self.dashboard_state.max_steps = 0
        if not hasattr(self.dashboard_state, 'cumulative_reward'):
            self.dashboard_state.cumulative_reward = 0.0
        if not hasattr(self.dashboard_state, 'last_step_reward'):
            self.dashboard_state.last_step_reward = 0.0
        if not hasattr(self.dashboard_state, 'episode_reward'):
            self.dashboard_state.episode_reward = 0.0
        if not hasattr(self.dashboard_state, 'episode_length'):
            self.dashboard_state.episode_length = 0
        if not hasattr(self.dashboard_state, 'mean_episode_reward'):
            self.dashboard_state.mean_episode_reward = 0.0
        if not hasattr(self.dashboard_state, 'mean_episode_length'):
            self.dashboard_state.mean_episode_length = 0.0
            
        # Training attributes
        if not hasattr(self.dashboard_state, 'total_episodes'):
            self.dashboard_state.total_episodes = 0
        if not hasattr(self.dashboard_state, 'total_updates'):
            self.dashboard_state.total_updates = 0
        if not hasattr(self.dashboard_state, 'updates'):
            self.dashboard_state.updates = 0
        if not hasattr(self.dashboard_state, 'global_steps'):
            self.dashboard_state.global_steps = 0
            
        # Trade attributes
        if not hasattr(self.dashboard_state, 'session_total_trades'):
            self.dashboard_state.session_total_trades = 0
        if not hasattr(self.dashboard_state, 'session_winning_trades'):
            self.dashboard_state.session_winning_trades = 0
        if not hasattr(self.dashboard_state, 'realized_pnl'):
            self.dashboard_state.realized_pnl = 0.0
        if not hasattr(self.dashboard_state, 'win_rate'):
            self.dashboard_state.win_rate = 0.0
            
        # Curriculum attributes
        if not hasattr(self.dashboard_state, 'curriculum_stage'):
            self.dashboard_state.curriculum_stage = 'stage_1'
        if not hasattr(self.dashboard_state, 'curriculum_progress'):
            self.dashboard_state.curriculum_progress = 0.0
        if not hasattr(self.dashboard_state, 'roc_range'):
            self.dashboard_state.roc_range = [0.0, 1.0]
        if not hasattr(self.dashboard_state, 'activity_range'):
            self.dashboard_state.activity_range = [0.0, 1.0]
        
        # Chart data attributes
        if not hasattr(self.dashboard_state, 'reset_points_data'):
            self.dashboard_state.reset_points_data = []
            
        # Performance metrics attributes
        if not hasattr(self.dashboard_state, 'steps_per_second'):
            self.dashboard_state.steps_per_second = 0.0
        if not hasattr(self.dashboard_state, 'episodes_per_hour'):
            self.dashboard_state.episodes_per_hour = 0.0
        if not hasattr(self.dashboard_state, 'updates_per_hour'):
            self.dashboard_state.updates_per_hour = 0.0
        if not hasattr(self.dashboard_state, 'updates_per_second'):
            self.dashboard_state.updates_per_second = 0.0
            
        # PPO metrics attributes
        if not hasattr(self.dashboard_state, 'policy_loss'):
            self.dashboard_state.policy_loss = 0.0
        if not hasattr(self.dashboard_state, 'value_loss'):
            self.dashboard_state.value_loss = 0.0
        if not hasattr(self.dashboard_state, 'entropy'):
            self.dashboard_state.entropy = 0.0
        if not hasattr(self.dashboard_state, 'clip_fraction'):
            self.dashboard_state.clip_fraction = 0.0
        if not hasattr(self.dashboard_state, 'kl_divergence'):
            self.dashboard_state.kl_divergence = 0.0
        if not hasattr(self.dashboard_state, 'explained_variance'):
            self.dashboard_state.explained_variance = 0.0
        if not hasattr(self.dashboard_state, 'learning_rate'):
            self.dashboard_state.learning_rate = 0.0
        if not hasattr(self.dashboard_state, 'total_loss'):
            self.dashboard_state.total_loss = 0.0
            
        # Reward component attributes
        if not hasattr(self.dashboard_state, 'reward_components'):
            self.dashboard_state.reward_components = {}
        if not hasattr(self.dashboard_state, 'episode_reward_components'):
            self.dashboard_state.episode_reward_components = {}
        if not hasattr(self.dashboard_state, 'session_reward_components'):
            self.dashboard_state.session_reward_components = {}
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Initialize dashboard for training session."""
        if not self.enabled or not self.dashboard_state:
            return
        
        self.training_start_time = datetime.now()
        
        # Update session info
        self.dashboard_state.session_start_time = self.training_start_time
        self.dashboard_state.model_name = config.get('experiment_name', 'training')
        # Initialize attributes if they don't exist
        if not hasattr(self.dashboard_state, 'total_episodes'):
            self.dashboard_state.total_episodes = 0
        if not hasattr(self.dashboard_state, 'total_updates'):
            self.dashboard_state.total_updates = 0
        if not hasattr(self.dashboard_state, 'episode_number'):
            self.dashboard_state.episode_number = 0
        if not hasattr(self.dashboard_state, 'max_steps'):
            self.dashboard_state.max_steps = 0
        if not hasattr(self.dashboard_state, 'cumulative_reward'):
            self.dashboard_state.cumulative_reward = 0.0
        
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
        self.dashboard_state.total_episodes = episode_num
        self.dashboard_state.episode_number = episode_num
        
        # Initialize episode tracking attributes
        if not hasattr(self.dashboard_state, 'current_step'):
            self.dashboard_state.current_step = 0
        if not hasattr(self.dashboard_state, 'cumulative_reward'):
            self.dashboard_state.cumulative_reward = 0.0
        
        # Reset episode-level counters
        self.dashboard_state.current_step = 0
        self.dashboard_state.cumulative_reward = 0.0
        
        # Reset episode reward components
        if hasattr(self.dashboard_state, 'episode_reward_components'):
            self.dashboard_state.episode_reward_components = {}
        if hasattr(self.dashboard_state, 'reward_components'):
            self.dashboard_state.reward_components = {}
        
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
        
        # Set max_steps from reset_info or use a reasonable default
        if hasattr(self.dashboard_state, 'max_steps'):
            self.dashboard_state.max_steps = reset_info.get('max_steps', 1000)
        
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
        
        # Update current market data with proper fallbacks
        current_price = info.get('current_price', 0.0)
        if current_price > 0:
            self.dashboard_state.current_price = current_price
            self.dashboard_state.bid_price = info.get('bid_price', current_price)
            self.dashboard_state.ask_price = info.get('ask_price', current_price)
            self.dashboard_state.spread = self.dashboard_state.ask_price - self.dashboard_state.bid_price
            self.dashboard_state.spread_pct = (self.dashboard_state.spread / 
                                             max(0.01, self.dashboard_state.current_price)) * 100
            
            # Update volume if available
            volume = info.get('volume', 0)
            if volume > 0:
                self.dashboard_state.volume = volume
        
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
        
        # Update portfolio with proper P&L tracking
        self.dashboard_state.total_equity = info.get('total_equity', 100000.0)
        self.dashboard_state.cash_balance = info.get('cash', 100000.0)
        self.dashboard_state.unrealized_pnl = info.get('unrealized_pnl', 0.0)
        
        # Update session P&L from multiple possible sources
        session_pnl = (
            info.get('realized_pnl', 0.0) or 
            info.get('portfolio_realized_pnl_session_net', 0.0) or
            info.get('total_pnl', 0.0) or
            0.0
        )
        self.dashboard_state.session_pnl = session_pnl
        self.dashboard_state.realized_pnl = session_pnl
        
        # Update step tracking
        self.dashboard_state.current_step = info.get('step', 0)
        self.dashboard_state.last_step_reward = step_data.get('reward', 0.0)
        self.dashboard_state.cumulative_reward = getattr(self.dashboard_state, 'cumulative_reward', 0.0) + self.dashboard_state.last_step_reward
        
        # Update max_steps from step_data if available
        if 'max_steps' in step_data:
            self.dashboard_state.max_steps = step_data['max_steps']
        elif 'info' in step_data and 'max_steps' in step_data['info']:
            self.dashboard_state.max_steps = step_data['info']['max_steps']
        
        # Update reward components if available
        if 'reward_components' in info:
            if not hasattr(self.dashboard_state, 'reward_components'):
                self.dashboard_state.reward_components = {}
            if not hasattr(self.dashboard_state, 'episode_reward_components'):
                self.dashboard_state.episode_reward_components = {}
            if not hasattr(self.dashboard_state, 'session_reward_components'):
                self.dashboard_state.session_reward_components = {}
                
            # Update current reward components
            reward_components = info['reward_components']
            self.dashboard_state.reward_components = reward_components
            
            # Accumulate episode reward components
            for component, value in reward_components.items():
                if component not in self.dashboard_state.episode_reward_components:
                    self.dashboard_state.episode_reward_components[component] = 0.0
                self.dashboard_state.episode_reward_components[component] += value
                
                if component not in self.dashboard_state.session_reward_components:
                    self.dashboard_state.session_reward_components[component] = 0.0
                self.dashboard_state.session_reward_components[component] += value
        
        # Track episode data
        self.current_episode_data['prices'].append(self.dashboard_state.current_price)
        self.current_episode_data['actions'].append(step_data.get('action', 0))
        self.current_episode_data['positions'].append(position)
        self.current_episode_data['rewards'].append(step_data.get('reward', 0.0))
        self.current_episode_data['timestamps'].append(info.get('timestamp', datetime.now()))
        
        # Update time
        if 'timestamp' in info:
            timestamp = info['timestamp']
            try:
                # Parse ISO string to datetime
                import pandas as pd
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                # Convert to NY timezone if needed and remove timezone info for chart
                if hasattr(timestamp, 'tz_localize'):
                    if timestamp.tz is None:
                        timestamp_ny = timestamp.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)
                    else:
                        timestamp_ny = timestamp.tz_convert('America/New_York').tz_localize(None)
                else:
                    timestamp_ny = timestamp
                
                self.dashboard_state.current_timestamp = timestamp_ny
                self.dashboard_state.ny_time = timestamp_ny.strftime('%H:%M:%S')
            except Exception as e:
                self.logger.warning(f"Error processing timestamp {timestamp}: {e}")
                # Fallback to current time
                self.dashboard_state.current_timestamp = datetime.now()
                self.dashboard_state.ny_time = datetime.now().strftime('%H:%M:%S')
    
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
        if not hasattr(self.dashboard_state, 'episode_reward'):
            self.dashboard_state.episode_reward = 0.0
        if not hasattr(self.dashboard_state, 'episode_length'):
            self.dashboard_state.episode_length = 0
        if not hasattr(self.dashboard_state, 'mean_episode_reward'):
            self.dashboard_state.mean_episode_reward = 0.0
        if not hasattr(self.dashboard_state, 'mean_episode_length'):
            self.dashboard_state.mean_episode_length = 0.0
            
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
        self.dashboard_state.total_updates = update_num
        self.dashboard_state.updates = update_num
        if hasattr(self.dashboard_state, 'is_updating'):
            self.dashboard_state.is_updating = True
    
    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Update dashboard with PPO metrics."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Update loss metrics (now using consistent naming)
        self.dashboard_state.policy_loss = update_metrics.get('policy_loss', 0.0)
        self.dashboard_state.value_loss = update_metrics.get('value_loss', 0.0)
        self.dashboard_state.entropy = update_metrics.get('entropy', 0.0)
        self.dashboard_state.total_loss = update_metrics.get('total_loss', 0.0)
        self.dashboard_state.learning_rate = update_metrics.get('learning_rate', 0.0)
        
        # Update training metrics (now using consistent naming)
        self.dashboard_state.clip_fraction = update_metrics.get('clip_fraction', 0.0)
        self.dashboard_state.kl_divergence = update_metrics.get('kl_divergence', 0.0)
        self.dashboard_state.explained_variance = update_metrics.get('explained_variance', 0.0)
        self.dashboard_state.mean_episode_reward = update_metrics.get('mean_episode_reward', 0.0)
        
        # Initialize and update PPO metric histories
        if not hasattr(self.dashboard_state, 'policy_loss_history'):
            self.dashboard_state.policy_loss_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'value_loss_history'):
            self.dashboard_state.value_loss_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'entropy_history'):
            self.dashboard_state.entropy_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'kl_divergence_history'):
            self.dashboard_state.kl_divergence_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'clip_fraction_history'):
            self.dashboard_state.clip_fraction_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'learning_rate_history'):
            self.dashboard_state.learning_rate_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'explained_variance_history'):
            self.dashboard_state.explained_variance_history = deque(maxlen=100)
        if not hasattr(self.dashboard_state, 'mean_episode_reward_history'):
            self.dashboard_state.mean_episode_reward_history = deque(maxlen=100)
            
        # Add current values to histories
        self.dashboard_state.policy_loss_history.append(self.dashboard_state.policy_loss)
        self.dashboard_state.value_loss_history.append(self.dashboard_state.value_loss)
        self.dashboard_state.entropy_history.append(self.dashboard_state.entropy)
        self.dashboard_state.kl_divergence_history.append(self.dashboard_state.kl_divergence)
        self.dashboard_state.clip_fraction_history.append(self.dashboard_state.clip_fraction)
        self.dashboard_state.learning_rate_history.append(self.dashboard_state.learning_rate)
        self.dashboard_state.explained_variance_history.append(self.dashboard_state.explained_variance)
        self.dashboard_state.mean_episode_reward_history.append(self.dashboard_state.mean_episode_reward)
        
        # Calculate performance metrics
        if self.training_start_time:
            elapsed = (datetime.now() - self.training_start_time).total_seconds()
            total_steps = update_metrics.get('total_steps', 0)
            
            # Update performance metrics
            if elapsed > 0:
                self.dashboard_state.steps_per_second = total_steps / elapsed
                self.dashboard_state.episodes_per_hour = (self.total_episodes / elapsed) * 3600 if self.total_episodes > 0 else 0.0
                self.dashboard_state.updates_per_hour = (update_num / elapsed) * 3600 if update_num > 0 else 0.0
        
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
        
        # Initialize trade counter attributes if they don't exist
        if not hasattr(self.dashboard_state, 'session_total_trades'):
            self.dashboard_state.session_total_trades = 0
        if not hasattr(self.dashboard_state, 'session_winning_trades'):
            self.dashboard_state.session_winning_trades = 0
        if not hasattr(self.dashboard_state, 'realized_pnl'):
            self.dashboard_state.realized_pnl = 0.0
        if not hasattr(self.dashboard_state, 'win_rate'):
            self.dashboard_state.win_rate = 0.0
        
        # Update trade stats
        self.dashboard_state.session_total_trades += 1
        
        pnl = trade_result.get('pnl', 0.0)
        self.dashboard_state.realized_pnl += pnl
        
        if pnl > 0:
            self.dashboard_state.session_winning_trades += 1
        
        # Update win rate
        self.dashboard_state.win_rate = (self.dashboard_state.session_winning_trades / 
                                       max(1, self.dashboard_state.session_total_trades))
        
        # Calculate profit factor
        if not hasattr(self, 'total_winning_pnl'):
            self.total_winning_pnl = 0.0
        if not hasattr(self, 'total_losing_pnl'):
            self.total_losing_pnl = 0.0
            
        if pnl > 0:
            self.total_winning_pnl += pnl
        elif pnl < 0:
            self.total_losing_pnl += abs(pnl)
        
        # Calculate and update profit factor
        if self.total_losing_pnl > 0:
            profit_factor = self.total_winning_pnl / self.total_losing_pnl
        else:
            profit_factor = self.total_winning_pnl if self.total_winning_pnl > 0 else 0.0
            
        if not hasattr(self.dashboard_state, 'profit_factor'):
            self.dashboard_state.profit_factor = 0.0
        self.dashboard_state.profit_factor = profit_factor
        
        # Note: Don't add trades to recent_trades here as they're already added
        # by the event stream system to prevent duplicates. The callback system
        # is only responsible for trade statistics and profit factor calculation.
    
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
    
    def on_momentum_day_change(self, event_data: Dict[str, Any]) -> None:
        """Update momentum training info."""
        if not self.enabled or not self.dashboard_state:
            return
        
        # Extract day_info from the event data structure
        day_info = event_data.get('day_info', event_data)
        
        # Update momentum info
        date_obj = day_info.get('date')
        if hasattr(date_obj, 'strftime'):
            momentum_day = date_obj.strftime('%Y-%m-%d')
        else:
            momentum_day = str(date_obj) if date_obj else 'N/A'
        
        self.dashboard_state.current_momentum_day_date = momentum_day
        self.dashboard_state.current_momentum_day_quality = day_info.get('quality_score', 0.0)
        
        # Extract quality score for display
        quality_score = day_info.get('quality_score', 0.0)
        if quality_score == 0.0:
            quality_score = day_info.get('day_quality', 0.0)
        
        # Ensure attributes exist before setting
        if not hasattr(self.dashboard_state, 'current_momentum_day_date'):
            self.dashboard_state.current_momentum_day_date = momentum_day
        if not hasattr(self.dashboard_state, 'current_momentum_day_quality'):
            self.dashboard_state.current_momentum_day_quality = quality_score
        
        self.dashboard_state.current_momentum_day_date = momentum_day
        self.dashboard_state.current_momentum_day_quality = quality_score
        
        # Process reset points data for chart display
        reset_points_data = event_data.get('reset_points', [])
        if reset_points_data and hasattr(self.dashboard_state, 'reset_points_data'):
            # Update dashboard state with reset points for chart markers
            self.dashboard_state.reset_points_data = reset_points_data
            self.logger.info(f"📍 Updated dashboard with {len(reset_points_data)} reset points")
        elif not hasattr(self.dashboard_state, 'reset_points_data'):
            # Initialize if attribute doesn't exist
            self.dashboard_state.reset_points_data = reset_points_data
        
        # Add event
        date_str = momentum_day if momentum_day != 'N/A' else 'Unknown'
        self._add_training_event(
            f"Switched to {date_str}: quality={quality_score:.3f}",
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
    
    def on_custom_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Handle custom events for dashboard updates."""
        if not self.enabled or not self.dashboard_state:
            return
            
        if event_name == 'training_update':
            # Update training progress information
            self.dashboard_state.mode = event_data.get('mode', 'Training')
            self.dashboard_state.stage = event_data.get('stage', 'Active')
            self.dashboard_state.overall_progress = event_data.get('overall_progress', 0.0)
            self.dashboard_state.updates = event_data.get('updates', 0)
            self.dashboard_state.global_steps = event_data.get('global_steps', 0)
            self.dashboard_state.total_episodes = event_data.get('total_episodes', 0)
            self.dashboard_state.stage_status = event_data.get('stage_status', '')
            
            # Update performance metrics with actual values (convert updates_per_second to updates_per_hour)
            self.dashboard_state.steps_per_second = event_data.get('steps_per_second', 0.0)
            self.dashboard_state.episodes_per_hour = event_data.get('episodes_per_hour', 0.0)
            updates_per_second = event_data.get('updates_per_second', 0.0)
            self.dashboard_state.updates_per_hour = updates_per_second * 3600 if updates_per_second > 0 else 0.0
            
            # Progress tracking
            self.dashboard_state.rollout_steps = event_data.get('rollout_steps', 0)
            self.dashboard_state.rollout_total = event_data.get('rollout_total', 0)
            self.dashboard_state.current_epoch = event_data.get('current_epoch', 0)
            self.dashboard_state.total_epochs = event_data.get('total_epochs', 0)
            self.dashboard_state.current_batch = event_data.get('current_batch', 0)
            self.dashboard_state.total_batches = event_data.get('total_batches', 0)
            
        elif event_name == 'ppo_metrics':
            # Update PPO training metrics and maintain history
            self.dashboard_state.policy_loss = event_data.get('policy_loss', 0.0)
            self.dashboard_state.value_loss = event_data.get('value_loss', 0.0)
            self.dashboard_state.entropy = event_data.get('entropy', 0.0)
            self.dashboard_state.clip_fraction = event_data.get('clip_fraction', 0.0)
            self.dashboard_state.learning_rate = event_data.get('learning_rate', 0.0)
            self.dashboard_state.kl_divergence = event_data.get('kl_divergence', 0.0)
            self.dashboard_state.explained_variance = event_data.get('explained_variance', 0.0)
            self.dashboard_state.mean_episode_reward = event_data.get('mean_episode_reward', 0.0)
            
            # Update PPO metric histories for charts
            if not hasattr(self.dashboard_state, 'policy_loss_history'):
                self.dashboard_state.policy_loss_history = deque(maxlen=100)
            if not hasattr(self.dashboard_state, 'value_loss_history'):
                self.dashboard_state.value_loss_history = deque(maxlen=100)
            if not hasattr(self.dashboard_state, 'entropy_history'):
                self.dashboard_state.entropy_history = deque(maxlen=100)
                
            self.dashboard_state.policy_loss_history.append(self.dashboard_state.policy_loss)
            self.dashboard_state.value_loss_history.append(self.dashboard_state.value_loss)
            self.dashboard_state.entropy_history.append(self.dashboard_state.entropy)
            
        elif event_name == 'curriculum_progress':
            # Update curriculum tracking
            self.dashboard_state.curriculum_progress = event_data.get('progress', 0.0)
            self.dashboard_state.curriculum_stage = event_data.get('stage', 'stage_1')
            
        elif event_name == 'curriculum_detail':
            # Update detailed curriculum information
            self.dashboard_state.curriculum_stage = event_data.get('current_stage', 'stage_1')
            self.dashboard_state.episodes_to_next_stage = event_data.get('episodes_to_next_stage', 0)
            self.dashboard_state.next_stage_name = event_data.get('next_stage_name', '')
            self.dashboard_state.episodes_per_day_config = event_data.get('episodes_per_day_config', 10)
            self.dashboard_state.curriculum_strategy = event_data.get('curriculum_method', 'quality_based')
            
        elif event_name == 'cycle_completion':
            # Update cycle and day switching tracking
            self.dashboard_state.cycles_completed = event_data.get('cycles_completed', 0)
            self.dashboard_state.target_cycles_per_day = event_data.get('target_cycles_per_day', 10)
            self.dashboard_state.cycles_remaining_for_day_switch = event_data.get('cycles_remaining_for_day_switch', 10)
            self.dashboard_state.episodes_on_current_day = event_data.get('episodes_on_current_day', 0)
            self.dashboard_state.day_switch_progress_pct = event_data.get('day_switch_progress_pct', 0.0)
            
        elif event_name == 'reward_calculation':
            # Update reward components from reward calculator
            if not hasattr(self.dashboard_state, 'reward_components'):
                self.dashboard_state.reward_components = {}
            if not hasattr(self.dashboard_state, 'episode_reward_components'):
                self.dashboard_state.episode_reward_components = {}
            if not hasattr(self.dashboard_state, 'session_reward_components'):
                self.dashboard_state.session_reward_components = {}
            
            # Extract reward components from event data
            components = event_data.get('components', {})
            total_reward = event_data.get('total_reward', 0.0)
            
            # Update current step reward components
            self.dashboard_state.reward_components = components
            
            # Accumulate episode reward components
            for component, value in components.items():
                if component not in self.dashboard_state.episode_reward_components:
                    self.dashboard_state.episode_reward_components[component] = 0.0
                self.dashboard_state.episode_reward_components[component] += value
                
                # Accumulate session reward components
                if component not in self.dashboard_state.session_reward_components:
                    self.dashboard_state.session_reward_components[component] = 0.0
                self.dashboard_state.session_reward_components[component] += value
            
    def on_reset_point_selected(self, tracking_data: Dict[str, Any]) -> None:
        """Handle reset point selection tracking."""
        if not self.enabled or not self.dashboard_state:
            return
            
        # Update reset point tracking information
        self.dashboard_state.selected_reset_point_index = tracking_data.get('selected_index', 0)
        self.dashboard_state.selected_reset_point_timestamp = tracking_data.get('selected_timestamp', '')
        self.dashboard_state.total_available_points = tracking_data.get('total_available_points', 0)
        self.dashboard_state.points_used_in_cycle = tracking_data.get('points_used_in_cycle', 0)
        self.dashboard_state.points_remaining_in_cycle = tracking_data.get('points_remaining_in_cycle', 0)
        self.dashboard_state.current_roc_score = tracking_data.get('roc_score', 0.0)
        self.dashboard_state.current_activity_score = tracking_data.get('activity_score', 0.0)
        
        # Update curriculum ranges
        roc_range = tracking_data.get('roc_range', [0.0, 1.0])
        activity_range = tracking_data.get('activity_range', [0.0, 1.0])
        if hasattr(self.dashboard_state, 'roc_range'):
            self.dashboard_state.roc_range = roc_range
        if hasattr(self.dashboard_state, 'activity_range'):
            self.dashboard_state.activity_range = activity_range

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
        if hasattr(self.dashboard_state, 'training_complete'):
            self.dashboard_state.training_complete = True
        if hasattr(self.dashboard_state, 'final_mean_reward'):
            self.dashboard_state.final_mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        if hasattr(self.dashboard_state, 'final_stats'):
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
    
    def _start_dashboard_server(self) -> None:
        """Start the dashboard server if not already running."""
        try:
            from dashboard import start_dashboard
            import threading
            
            # Get port from config
            port = self.config.get('port', 8051)
            
            # Start dashboard in background thread
            def start_server():
                try:
                    self.logger.info(f"🌐 Starting dashboard server on port {port}")
                    start_dashboard(port=port, open_browser=False)
                except Exception as e:
                    self.logger.warning(f"Failed to start dashboard server: {e}")
            
            dashboard_thread = threading.Thread(target=start_server, daemon=True)
            dashboard_thread.start()
            
            # Give it a moment to start, then log URL
            def log_url():
                import time
                time.sleep(2)
                self.logger.info(f"📊 Dashboard available at: http://localhost:{port}")
            
            url_thread = threading.Thread(target=log_url, daemon=True)
            url_thread.start()
            
        except ImportError as e:
            self.logger.warning(f"Dashboard server not available: {e}")
        except Exception as e:
            self.logger.error(f"Error starting dashboard server: {e}")