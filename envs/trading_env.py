import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
import heapq
import os
import torch

from omegaconf import DictConfig


class MarketEvent:
    """Represents a market event with a specific timestamp."""

    def __init__(self, timestamp: pd.Timestamp, event_type: str, data: Dict = None, priority: int = 0):
        self.timestamp = timestamp
        self.event_type = event_type  # 'price', 'trade', 'quote'
        self.data = data or {}
        self.priority = priority  # Lower numbers = higher priority

    def __lt__(self, other):
        # For priority queue ordering
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class SimpleReward:
    """Minimal trading reward function."""

    def __init__(self,
                 reward_scaling=1.0,
                 trade_penalty=0.01,
                 hold_penalty=0.0):
        """Initialize reward parameters."""
        self.reward_scaling = reward_scaling
        self.trade_penalty = trade_penalty
        self.hold_penalty = hold_penalty

    def calculate(self,
                  portfolio_change,
                  trade_executed=False,
                  momentum_strength=0.0,
                  relative_volume=1.0):
        """Calculate reward based on portfolio change and trading decisions."""
        # Base reward is portfolio change
        reward = portfolio_change * self.reward_scaling

        # Apply trading penalties
        if trade_executed:
            reward -= self.trade_penalty
        else:
            reward -= self.hold_penalty

        # Simple momentum bonus
        if trade_executed and momentum_strength > 0.5 and relative_volume > 3.0:
            reward *= 1.5  # 50% boost for trading with momentum

        return reward


class StateManager:
    """
    State manager that maintains state for the RL agent.
    Integrated directly within the TradingEnv.
    """

    def __init__(self, feature_config: Dict = None,
                 lookback_periods: Dict[str, int] = None,
                 logger: logging.Logger = None):
        """
        Initialize the state manager.

        Args:
            feature_config: Configuration for feature handling
            lookback_periods: Dict mapping timeframes to number of periods to include in state
            logger: Optional logger
        """
        self.feature_config = feature_config or {}
        self.lookback_periods = lookback_periods or {'1m': 10, '5m': 5}
        self.logger = logger or logging.getLogger(__name__)

        # Minimal state storage
        self.current_time = None
        self.current_features = None
        self.current_state = None
        self.state_history = {}  # Timestamped state history

        # Position tracking
        self.current_position = 0.0  # 0.0 = flat, 1.0 = fully long
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0

        # Trade tracking
        self.trade_history = []
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

    def get_state_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the current state as a dictionary of tensors suitable for the transformer model.

        Returns:
            Dictionary with state tensors for each branch
        """
        if not hasattr(self, 'tensor_state'):
            # Create dummy tensors if not available
            batch_size = 1
            hf_seq_len = 60
            hf_feat_dim = 20
            mf_seq_len = 30
            mf_feat_dim = 15
            lf_seq_len = 30
            lf_feat_dim = 10
            static_feat_dim = 15

            self.tensor_state = {
                'hf_features': torch.zeros((batch_size, hf_seq_len, hf_feat_dim)),
                'mf_features': torch.zeros((batch_size, mf_seq_len, mf_feat_dim)),
                'lf_features': torch.zeros((batch_size, lf_seq_len, lf_feat_dim)),
                'static_features': torch.zeros((batch_size, static_feat_dim))
            }

            # Add position-related features to static features
            if self.current_state:
                static_features = self.tensor_state['static_features']
                static_features[0, 0] = self.current_position
                static_features[0, 1] = self.unrealized_pnl
                static_features[0, 2] = self.entry_price if self.entry_price > 0 else 0
                static_features[0, 3] = self.last_price if self.last_price > 0 else 0
                static_features[0, 4] = self.win_count / max(1, self.trade_count)

        return self.tensor_state

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def update_from_features(self, features_df: pd.DataFrame, timestamp: datetime = None):
        """
        Update the state from pre-computed features.

        Args:
            features_df: DataFrame with features
            timestamp: Optional timestamp to update to (if None, use latest in features_df)
        """
        if features_df.empty:
            return

        # If timestamp not provided, use the latest in the features DataFrame
        if timestamp is None:
            timestamp = features_df.index[-1]

        # Find features at or before the timestamp
        available_features = features_df[features_df.index <= timestamp]
        if available_features.empty:
            return

        # Get the exact timestamp or closest prior timestamp
        closest_timestamp = available_features.index[-1]

        # Update current time and features
        self.current_time = closest_timestamp
        self.current_features = available_features.loc[closest_timestamp]

        # Update the current state
        self._update_state_vector()

        # Store in history
        self.state_history[closest_timestamp] = self.current_state.copy() if self.current_state else {}

    def update_position(self, new_position: float, price: float, timestamp: datetime = None):
        """
        Update the current position information.

        Args:
            new_position: New position size (0.0 = flat, 1.0 = fully long)
            price: Current price for P&L calculation
            timestamp: Optional timestamp (defaults to current_time)
        """
        # Validate inputs
        if new_position < 0.0 or new_position > 1.0:
            new_position = max(0.0, min(1.0, new_position))

        if price <= 0.0:
            return

        # Update timestamp if provided
        if timestamp is not None:
            self.current_time = timestamp

        # Detect trade completion (position flattened)
        if self.current_position > 0.0 and new_position == 0.0:
            # Calculate realized P&L
            realized_pnl = price - self.entry_price

            # Update trade statistics
            self.trade_count += 1
            if realized_pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            self.total_pnl += realized_pnl

            # Record trade
            trade = {
                'entry_price': self.entry_price,
                'exit_price': price,
                'realized_pnl': realized_pnl
            }

            self.trade_history.append(trade)

            # Reset position tracking
            self.entry_price = 0.0

        # Detect new position
        elif self.current_position == 0.0 and new_position > 0.0:
            # Starting a new position
            self.entry_price = price

        # Update current values
        self.current_position = new_position
        self.last_price = price

        # Calculate unrealized P&L
        if self.current_position > 0.0 and self.entry_price > 0.0:
            self.unrealized_pnl = self.last_price - self.entry_price
        else:
            self.unrealized_pnl = 0.0

        # Update the state
        self._update_state_vector()

        # Store in history
        if self.current_time:
            self.state_history[self.current_time] = self.current_state.copy() if self.current_state else {}

    def _update_state_vector(self):
        """Update the current state vector with minimal features."""
        if self.current_features is None:
            self.current_state = {}
            return

        # Create a minimal state dictionary
        self.current_state = {
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'entry_price': self.entry_price,
            'last_price': self.last_price,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(1, self.trade_count),
            'total_pnl': self.total_pnl
        }

        # Add some basic features if available
        if self.current_features is not None:
            for col in self.current_features.index:
                # Add just a few features to keep it simple
                if 'price_change' in col or 'volume' in col or 'sma' in col:
                    self.current_state[col] = self.current_features[col]

    def get_state_array(self) -> np.ndarray:
        """
        Get the current state as a flat array for RL models.

        Returns:
            NumPy array with the flattened state representation
        """
        if self.current_state is None:
            return np.array([])

        # Extract values from state dictionary
        state_values = list(self.current_state.values())

        # Convert to numpy array
        return np.array(state_values, dtype=np.float32)

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the current state as a dictionary.

        Returns:
            Dictionary with state components
        """
        return self.current_state if self.current_state is not None else {}

    def get_state_at_time(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get the state at a specific timestamp.

        Args:
            timestamp: Timestamp to get state for

        Returns:
            State dictionary at the timestamp
        """
        if timestamp in self.state_history:
            return self.state_history[timestamp]

        # Find closest timestamp before the requested one
        closest_time = None
        for state_time in self.state_history.keys():
            if state_time <= timestamp and (closest_time is None or state_time > closest_time):
                closest_time = state_time

        if closest_time:
            return self.state_history[closest_time]

        return {}

    def get_state_history(self) -> Dict[datetime, Dict[str, Any]]:
        """
        Get the full state history.

        Returns:
            Dictionary mapping timestamps to state dictionaries
        """
        return self.state_history

    def reset(self):
        """Reset the state manager to initial conditions."""
        self.current_time = None
        self.current_features = None
        self.current_state = None
        self.state_history = {}

        self.current_position = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0

        self.trade_history = []
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

    def get_trade_history(self) -> List[Dict]:
        """
        Get the history of completed trades.

        Returns:
            List of trade dictionaries
        """
        return self.trade_history

    def get_trade_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics of trading performance.

        Returns:
            Dictionary with trade statistics
        """
        stats = {
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_count / max(1, self.trade_count),
            'total_pnl': self.total_pnl,
        }

        if self.trade_history:
            # Calculate averages
            win_trades = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] > 0]
            loss_trades = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] <= 0]

            stats['avg_win'] = sum(win_trades) / max(1, len(win_trades))
            stats['avg_loss'] = sum(loss_trades) / max(1, len(loss_trades))

        return stats


class TradingEnv(gym.Env):
    """Trading environment with integrated state management for momentum trading."""

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 data_manager=None,  # We'll accept but not use this for compatibility
                 cfg=None,  # Config object
                 logger=None):
        """Initialize the trading environment to be compatible with main.py."""
        self.logger = logger or logging.getLogger(__name__)
        self.cfg = cfg

        # Extract configuration parameters with sane defaults
        if hasattr(cfg, '_to_dict'):
            config_dict = cfg._to_dict()
        else:
            config_dict = cfg if cfg else {}

        # Get reward config
        reward_config = config_dict.get('reward', {})
        self.reward_scaling = reward_config.get('scaling', 2.0)
        self.trade_penalty = reward_config.get('trade_penalty', 0.1)
        self.hold_penalty = reward_config.get('hold_penalty', 0.0)

        # Get environment config
        self.max_steps = config_dict.get('max_steps', 500)
        self.state_dim = config_dict.get('state_dim', 20)

        # Initialize the state manager
        feature_config = config_dict.get('feature_config', {})
        self.state_manager = StateManager(
            feature_config=feature_config,
            logger=self.logger
        )

        # Time management settings
        self.min_execution_latency_ms = 50
        self.max_execution_latency_ms = 250

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Initialize reward function
        self.reward_fn = SimpleReward(
            reward_scaling=self.reward_scaling,
            trade_penalty=self.trade_penalty,
            hold_penalty=self.hold_penalty
        )

        # These will be set when initialize_for_symbol is called
        self.symbol = None
        self.start_date = None
        self.end_date = None
        self.market_data = None
        self.data_dir = None

        # Event queue for timestamp management
        self.event_queue = []

        # State variables
        self.current_timestamp = None
        self.current_price = 0.0
        self.position = 0.0  # -1.0 to 1.0 (short to long)
        self.cash = 100000.0
        self.portfolio_value = self.cash
        self.trades = []

        # Market indicators
        self.price_history = []
        self.volume_history = []
        self.momentum_strength = 0.0
        self.relative_volume = 1.0

        # Tracking variables
        self.steps = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.pending_executions = []  # Orders waiting to execute

        # Make state manager aware of simulator
        self.simulator = self  # For compatibility with existing code

        self._log("Trading environment initialized with integrated state manager")

    def _log(self, message: str, level: int = logging.INFO):
        """Log message if logger is available."""
        if self.logger:
            self.logger.log(level, message)

    def initialize_for_symbol(self, symbol: str,
                              mode: str = 'backtesting',
                              start_time: Union[datetime, str] = None,
                              end_time: Union[datetime, str] = None,
                              timeframes: List[str] = None) -> bool:
        """Initialize the environment for a specific symbol - for main.py compatibility."""
        self.symbol = symbol

        # Set date range
        if isinstance(start_time, str):
            self.start_date = pd.Timestamp(start_time)
        else:
            self.start_date = start_time

        if isinstance(end_time, str):
            # Make sure end_time includes a time component
            if 'T' not in end_time and ' ' not in end_time:
                # If just a date is provided, set to end of day
                self.end_date = pd.Timestamp(f"{end_time} 23:59:59")
            else:
                self.end_date = pd.Timestamp(end_time)
        else:
            self.end_date = end_time

        # Load market data
        try:
            self.market_data = self._load_market_data(symbol, timeframes)
            if not self.market_data:
                self._log(f"Failed to load data for {symbol}", logging.ERROR)
                return False

            # Initialize event queue
            self._initialize_event_queue()
            return True
        except Exception as e:
            self._log(f"Error initializing data for {symbol}: {str(e)}", logging.ERROR)
            return False

    def _load_market_data(self, symbol, timeframes=None) -> Dict[str, pd.DataFrame]:
        """
        Load market data for the symbol.

        In a real implementation, this would load data from actual files.
        For this simplified version, we generate synthetic data.
        """
        self._log(f"Loading market data for {symbol} from {self.start_date} to {self.end_date}")

        # Create price data with 1-second intervals
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='1s')

        # Generate synthetic price data (random walk with momentum)
        np.random.seed(42)  # For reproducibility
        price_changes = np.random.normal(0, 0.01, size=len(date_range))

        # Add some momentum patterns - occasional surges
        for i in range(20):  # Add 20 momentum surges
            surge_start = np.random.randint(0, len(date_range) - 100)
            surge_length = np.random.randint(10, 100)
            surge_direction = np.random.choice([-1, 1])
            surge_strength = np.random.uniform(0.05, 0.2)

            # Create a momentum pattern
            pattern = surge_direction * np.linspace(0, surge_strength, surge_length)
            price_changes[surge_start:surge_start + surge_length] += pattern

        prices = 10.0 + np.cumsum(price_changes)  # Start at $10

        # Generate volume data with occasional volume spikes
        base_volume = np.random.exponential(1000, size=len(date_range))
        volume_spikes = np.zeros(len(date_range))

        for i in range(15):  # Add 15 volume spikes
            spike_start = np.random.randint(0, len(date_range) - 50)
            spike_length = np.random.randint(5, 50)
            spike_factor = np.random.uniform(3, 10)

            # Create a volume spike pattern
            volume_spikes[spike_start:spike_start + spike_length] = spike_factor

        # Combine base volume with spikes
        volume = base_volume * (1 + volume_spikes)

        # Create DataFrame for price data
        price_df = pd.DataFrame({
            'price': prices,
            'volume': volume
        }, index=date_range)

        # Create synthetic bid/ask data
        quote_df = pd.DataFrame({
            'bid': prices - np.random.uniform(0.01, 0.05, size=len(date_range)),
            'ask': prices + np.random.uniform(0.01, 0.05, size=len(date_range)),
            'bid_size': np.random.exponential(500, size=len(date_range)),
            'ask_size': np.random.exponential(500, size=len(date_range))
        }, index=date_range)

        # Create synthetic trade data
        trade_indices = np.random.choice(
            range(len(date_range)),
            size=int(len(date_range) * 0.2),
            replace=False
        )
        trade_timestamps = date_range[trade_indices]

        trade_df = pd.DataFrame({
            'price': prices[trade_indices],
            'size': np.random.exponential(200, size=len(trade_indices)),
            'side': np.random.choice(['buy', 'sell'], size=len(trade_indices))
        }, index=trade_timestamps)

        # Create OHLC bars for different timeframes
        bars_dict = {}
        for tf in ['1s', '1m', '5m']:
            if not timeframes or tf in timeframes:
                # Convert to OHLC
                freq = tf.replace('s', 'S').replace('m', 'min')
                ohlc = price_df['price'].resample(freq).ohlc()
                volume = price_df['volume'].resample(freq).sum()

                bars_df = pd.concat([ohlc, volume], axis=1)
                bars_dict[f'bars_{tf}'] = bars_df

        # Combine all data types
        data_dict = {
            'trades': trade_df,
            'quotes': quote_df
        }
        data_dict.update(bars_dict)

        return data_dict

    def _initialize_event_queue(self):
        """Initialize the event queue from raw market data."""
        self.event_queue = []

        # Process price data from 1-second bars
        if 'bars_1s' in self.market_data:
            for idx, row in self.market_data['bars_1s'].iterrows():
                event = MarketEvent(idx, 'price', {
                    'price': row['close'],
                    'volume': row['volume']
                }, priority=3)
                heapq.heappush(self.event_queue, event)

        # Process quote data
        if 'quotes' in self.market_data:
            for idx, row in self.market_data['quotes'].iterrows():
                event = MarketEvent(idx, 'quote', {
                    'bid': row['bid'],
                    'ask': row['ask'],
                    'bid_size': row['bid_size'],
                    'ask_size': row['ask_size']
                }, priority=2)
                heapq.heappush(self.event_queue, event)

        # Process trade data
        if 'trades' in self.market_data:
            for idx, row in self.market_data['trades'].iterrows():
                event = MarketEvent(idx, 'trade', {
                    'price': row['price'],
                    'size': row['size'],
                    'side': row['side']
                }, priority=1)  # Trades have highest priority
                heapq.heappush(self.event_queue, event)

        self._log(f"Initialized event queue with {len(self.event_queue)} events")

    def _update_market_indicators(self, price, volume):
        """Update market indicators for reward calculation."""
        # Store price and volume history
        self.price_history.append(price)
        self.volume_history.append(volume)

        # Limit history size
        window_size = 60  # 1-minute window for 1-second data
        if len(self.price_history) > window_size:
            self.price_history = self.price_history[-window_size:]
            self.volume_history = self.volume_history[-window_size:]

        # Calculate momentum strength (based on recent price changes)
        if len(self.price_history) > 10:
            # Simple momentum: price change over last 10 seconds
            recent_change = (self.price_history[-1] / self.price_history[-10]) - 1

            # Normalize to 0-1 range (where >0.5 indicates positive momentum)
            self.momentum_strength = min(1.0, max(0.0, recent_change * 10 + 0.5))

        # Calculate relative volume
        if len(self.volume_history) > 30:
            avg_volume = sum(self.volume_history[:-10]) / max(1, len(self.volume_history) - 10)
            recent_volume = sum(self.volume_history[-10:]) / 10
            self.relative_volume = recent_volume / max(1, avg_volume)

    def _process_events_until(self, target_timestamp: pd.Timestamp) -> List[Dict]:
        """Process all events up to the target timestamp."""
        processed_events = []

        # Process events until we reach the target timestamp
        while self.event_queue and self.event_queue[0].timestamp <= target_timestamp:
            event = heapq.heappop(self.event_queue)

            # Update current state based on event type
            if event.event_type == 'price':
                self.current_price = event.data['price']
                # Update market indicators
                self._update_market_indicators(
                    event.data['price'],
                    event.data.get('volume', 0)
                )

                # Update state manager with new price
                self.state_manager.update_position(
                    self.position,  # Current position (unchanged)
                    self.current_price,  # New price
                    event.timestamp  # Event timestamp
                )
            elif event.event_type == 'quote':
                # Update current price as midpoint of bid/ask
                self.current_price = (event.data['bid'] + event.data['ask']) / 2
            elif event.event_type == 'trade':
                self.current_price = event.data['price']

            # Record processed event
            processed_events.append({
                'timestamp': event.timestamp,
                'type': event.event_type,
                'data': event.data
            })

            # Update current timestamp
            self.current_timestamp = event.timestamp

        return processed_events

    def _process_pending_executions(self, target_timestamp: pd.Timestamp) -> List[Dict]:
        """Process pending execution orders up to the target timestamp."""
        executed_orders = []
        remaining_orders = []

        for order in self.pending_executions:
            execution_time = order['scheduled_time']

            if execution_time <= target_timestamp:
                # Execute the order
                action_value = order['action_value']

                # Simplified trading logic
                old_position = self.position
                self.position = action_value  # Directly set position based on action

                # Calculate portfolio impact
                position_change = self.position - old_position
                trade_value = position_change * self.current_price

                # Update state manager with new position
                self.state_manager.update_position(
                    self.position,
                    self.current_price,
                    execution_time
                )

                # Record execution
                execution = {
                    'timestamp': execution_time,
                    'action_value': action_value,
                    'position_change': position_change,
                    'execution_price': self.current_price,
                    'trade_value': trade_value
                }

                executed_orders.append(execution)

                # Record trade if position changed
                if abs(position_change) > 0.01:
                    self.trades.append({
                        'timestamp': execution_time,
                        'position_change': position_change,
                        'price': self.current_price,
                        'value': trade_value
                    })
            else:
                # Keep for future processing
                remaining_orders.append(order)

        # Update pending executions
        self.pending_executions = remaining_orders

        return executed_orders

    def _calculate_state(self) -> np.ndarray:
        """Calculate the current state vector for the agent."""
        # Get state from state manager for consistency
        state_array = self.state_manager.get_state_array()

        # If state_array is too small, pad it to match state_dim
        if len(state_array) < self.state_dim:
            padding = np.zeros(self.state_dim - len(state_array), dtype=np.float32)
            state_array = np.concatenate([state_array, padding])
        # If state_array is too large, truncate it
        elif len(state_array) > self.state_dim:
            state_array = state_array[:self.state_dim]

        return state_array

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset state variables
        self.steps = 0
        self.total_reward = 0.0
        self.position = 0.0
        self.cash = 100000.0
        self.portfolio_value = self.cash
        self.trades = []
        self.pending_executions = []

        # Reset market indicators
        self.price_history = []
        self.volume_history = []
        self.momentum_strength = 0.0
        self.relative_volume = 1.0

        # Reset state manager
        self.state_manager.reset()

        # Reinitialize event queue
        self._initialize_event_queue()

        # Set timestamp to start of data
        if self.event_queue:
            self.current_timestamp = self.event_queue[0].timestamp

            # Process events to initialize state
            self._process_events_until(self.current_timestamp)
        else:
            self.current_timestamp = self.start_date

        # Calculate initial state
        state = self._calculate_state()

        # Increment episode counter
        self.episode_count += 1

        self._log(f"Environment reset to {self.current_timestamp}")

        return state, {}

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take an action in the environment."""
        # 1. Convert action to float if it's an array
        action_value = float(action[0]) if hasattr(action, "__len__") else float(action)

        # 2. Get current state for later reward calculation
        prev_portfolio_value = self.portfolio_value
        decision_time = self.current_timestamp

        # 3. Calculate execution latency
        latency_ms = np.random.uniform(
            self.min_execution_latency_ms,
            self.max_execution_latency_ms
        )
        execution_time = decision_time + pd.Timedelta(milliseconds=latency_ms)

        # 4. Schedule the execution
        self.pending_executions.append({
            'action_value': action_value,
            'decision_time': decision_time,
            'scheduled_time': execution_time,
            'latency_ms': latency_ms
        })

        # 5. Determine next decision point (next event in queue)
        if self.event_queue:
            next_timestamp = self.event_queue[0].timestamp
        else:
            # No more events, episode is done
            next_timestamp = None

        # 6. Process events until next timestamp
        if next_timestamp:
            processed_events = self._process_events_until(next_timestamp)

        # 7. Process any pending executions that should have happened
        executed_orders = self._process_pending_executions(self.current_timestamp)

        # 8. Update portfolio value based on current position and price
        self.portfolio_value = self.cash + (self.position * self.current_price)

        # 9. Calculate reward using our reward function
        portfolio_change = self.portfolio_value - prev_portfolio_value
        trade_executed = len(executed_orders) > 0

        reward = self.reward_fn.calculate(
            portfolio_change=portfolio_change,
            trade_executed=trade_executed,
            momentum_strength=self.momentum_strength,
            relative_volume=self.relative_volume
        )

        self.total_reward += reward

        # 10. Check if episode is done
        done = False
        if next_timestamp is None or next_timestamp > self.end_date or self.steps >= self.max_steps:
            done = True

        # 11. Get the new state
        next_state = self._calculate_state()

        # 12. Prepare info dictionary
        info = {
            'timestamp': self.current_timestamp,
            'decision_time': decision_time,
            'execution_time': execution_time,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'price': self.current_price,
            'executed_orders': executed_orders,
            'momentum_strength': self.momentum_strength,
            'relative_volume': self.relative_volume,
            'reward_components': {
                'portfolio_change': portfolio_change,
                'trade_executed': trade_executed
            }
        }

        # 13. Update step count
        self.steps += 1

        # 14. Add episode summary if done
        if done:
            # Get trade statistics from state manager
            trade_stats = self.state_manager.get_trade_statistics()

            info['episode'] = {
                'steps': self.steps,
                'total_reward': self.total_reward,
                'final_portfolio': self.portfolio_value,
                'return_pct': (self.portfolio_value / self.cash - 1) * 100,
                'trade_count': trade_stats.get('trade_count', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'avg_win': trade_stats.get('avg_win', 0),
                'avg_loss': trade_stats.get('avg_loss', 0)
            }

            self._log(f"Episode finished: steps={self.steps}, "
                      f"reward={self.total_reward:.2f}, "
                      f"return={info['episode']['return_pct']:.2f}%")

        return next_state, reward, done, False, info

    def get_current_state_tensor_dict(self):
        """Get current state as tensor dict for the transformer model."""
        return self.state_manager.get_state_tensor_dict()

    def get_portfolio_state(self):
        """For compatibility with the TradingSimulator interface."""
        return {
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'current_price': self.current_price,
            'timestamp': self.current_timestamp
        }

    def get_market_state(self):
        """For compatibility with the TradingSimulator interface."""
        return {
            'price': self.current_price,
            'timestamp': self.current_timestamp,
            'momentum_strength': self.momentum_strength,
            'relative_volume': self.relative_volume
        }

    def get_trade_statistics(self):
        """Get trade statistics from the state manager."""
        return self.state_manager.get_trade_statistics()