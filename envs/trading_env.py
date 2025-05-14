# simplified_trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
import heapq
import os

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


class TradingEnv(gym.Env):
    """Simplified trading environment with integrated timestamp and reward handling."""

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

        self._log("Simplified trading environment initialized")

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
        # Simple state with price, position, and market indicator information
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Add current price and position
        state[0] = self.current_price
        state[1] = self.position

        # Add recent price changes (last 5 steps)
        if len(self.price_history) >= 6:
            prices = self.price_history[-6:]
            # Calculate price changes
            price_changes = np.diff(prices) / prices[:-1]

            # Add to state
            for i in range(min(5, len(price_changes))):
                state[2 + i] = price_changes[-(i + 1)]

        # Add position and portfolio information
        state[7] = self.portfolio_value / self.cash  # Relative portfolio value

        # Add market indicators
        state[8] = self.momentum_strength  # 0-1 value indicating momentum strength
        state[9] = self.relative_volume  # Relative volume (>1 means above average)

        # Remaining elements can be used for additional features

        return state

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
            info['episode'] = {
                'steps': self.steps,
                'total_reward': self.total_reward,
                'final_portfolio': self.portfolio_value,
                'return_pct': (self.portfolio_value / self.cash - 1) * 100,
                'trade_count': len(self.trades)
            }

            self._log(f"Episode finished: steps={self.steps}, "
                      f"reward={self.total_reward:.2f}, "
                      f"return={info['episode']['return_pct']:.2f}%")

        return next_state, reward, done, False, info

    def get_current_state_tensor_dict(self):
        """For compatibility with the transformer model interface."""
        # Create a minimal tensor dict compatible with the transformer model
        batch_size = 1

        # Feature dimensions would normally come from your model config
        hf_seq_len, hf_feat_dim = 60, 20
        mf_seq_len, mf_feat_dim = 30, 15
        lf_seq_len, lf_feat_dim = 30, 10
        static_feat_dim = 15

        # Create dummy tensors - in a real implementation,
        # these would contain meaningful features
        hf_features = np.zeros((batch_size, hf_seq_len, hf_feat_dim))
        mf_features = np.zeros((batch_size, mf_seq_len, mf_feat_dim))
        lf_features = np.zeros((batch_size, lf_seq_len, lf_feat_dim))

        # Add our actual features to the static tensor
        static_features = np.zeros((batch_size, static_feat_dim))
        static_features[0, 0] = self.position
        static_features[0, 1] = self.portfolio_value / self.cash - 1  # Normalized P&L
        static_features[0, 2] = self.momentum_strength
        static_features[0, 3] = self.relative_volume

        # Convert to PyTorch tensors
        import torch
        return {
            'hf_features': torch.FloatTensor(hf_features),
            'mf_features': torch.FloatTensor(mf_features),
            'lf_features': torch.FloatTensor(lf_features),
            'static_features': torch.FloatTensor(static_features)
        }

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
        """For compatibility with the TradingSimulator interface."""
        win_trades = [t for t in self.trades if t['value'] > 0]
        loss_trades = [t for t in self.trades if t['value'] <= 0]

        return {
            'total_trades': len(self.trades),
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'win_rate': len(win_trades) / max(1, len(self.trades)),
            'avg_win': sum([t['value'] for t in win_trades]) / max(1, len(win_trades)),
            'avg_loss': sum([t['value'] for t in loss_trades]) / max(1, len(loss_trades)),
            'total_pnl': self.portfolio_value - self.cash
        }