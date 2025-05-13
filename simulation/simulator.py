# simulation/simulator.py (updated for Hydra)

# Import section remains the same
from typing import Dict, List, Union, Tuple, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import torch
from omegaconf import DictConfig

from feature.feature_extractor import FeatureExtractor
from feature.state_manager import StateManager
from simulation.market_simulator import MarketSimulator
from simulation.execution_simulator import ExecutionSimulator
from simulation.portfolio_simulator import PortfolioSimulator


class Simulator:
    """Minimal central simulator that coordinates all components"""

    def __init__(self, data_manager, config: Union[Dict, DictConfig] = None, logger: logging.Logger = None):
        self.data_manager = data_manager
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Convert OmegaConf to dict if needed for backward compatibility
        # Convert OmegaConf to dict if needed for backward compatibility
        if hasattr(self.config, "_to_dict"):
            config_dict = self.config._to_dict()
        else:
            config_dict = self.config

        # Extract feature config - try multiple ways to find it
        feature_config = config_dict.get('feature_config', config_dict.get('feature', {}))
        if not feature_config and 'feature' in config_dict:
            # Try to get from top-level config
            feature_config = config_dict['feature']
            if hasattr(feature_config, "_to_dict"):
                feature_config = feature_config._to_dict()

        # Component configs - extract from Hydra structure if available
        self.feature_config = feature_config
        self.market_config = config_dict.get('market_config', {})
        self.execution_config = config_dict.get('execution_config', {})
        self.portfolio_config = config_dict.get('portfolio_config', {})
        self.strategy_config = config_dict.get('strategy', {})

        # Initialize components
        self.feature_extractor = FeatureExtractor(self.feature_config, logger=logger)
        self.state_manager = StateManager({}, {}, logger=logger)
        self.market_simulator = MarketSimulator(self.market_config, logger=logger)
        self.execution_simulator = ExecutionSimulator(
            self.market_simulator,
            self.execution_config,
            logger=logger
        )
        self.portfolio_simulator = PortfolioSimulator(
            self.market_simulator,
            self.execution_simulator,
            self.portfolio_config,
            logger=logger
        )

        # State variables
        self.current_mode = 'idle'  # 'idle', 'training', 'backtesting', 'live'
        self.current_symbol = None
        self.current_timestamp = None
        self.start_timestamp = None
        self.end_timestamp = None
        self.features_cache = {}
        self.raw_data = {}

        # Tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0

        # Callbacks
        self.state_update_callbacks = []
        self.trade_callbacks = []
        self.portfolio_update_callbacks = []

    # Rest of the class remains the same
    # ...

    def _log(self, message: str, level: int = logging.INFO):
        if self.logger:
            self.logger.log(level, message)

    def get_current_state_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the current state as a dictionary of tensors for the transformer model.

        Returns:
            Dictionary with tensors for different branches
        """
        # Try to get structured tensor state from state manager
        if hasattr(self.state_manager, 'get_state_tensor_dict'):
            return self.state_manager.get_state_tensor_dict()

        # Fallback: create dummy tensors with right shapes
        batch_size = 1
        hf_seq_len = 60
        hf_feat_dim = 20
        mf_seq_len = 30
        mf_feat_dim = 15
        lf_seq_len = 30
        lf_feat_dim = 10
        static_feat_dim = 15

        state_dict = {
            'hf_features': torch.zeros((batch_size, hf_seq_len, hf_feat_dim)),
            'mf_features': torch.zeros((batch_size, mf_seq_len, mf_feat_dim)),
            'lf_features': torch.zeros((batch_size, lf_seq_len, lf_feat_dim)),
            'static_features': torch.zeros((batch_size, static_feat_dim))
        }

        # Add position features to static tensor
        if self.state_manager:
            pos_info = self.state_manager.get_state_dict()
            if pos_info:
                state_dict['static_features'][0, 0] = pos_info.get('current_position', 0)
                state_dict['static_features'][0, 1] = pos_info.get('unrealized_pnl', 0)
                state_dict['static_features'][0, 2] = pos_info.get('entry_price', 0)
                state_dict['static_features'][0, 3] = pos_info.get('last_price', 0)

        return state_dict
    def initialize_for_symbol(self, symbol: str,
                              mode: str = 'backtesting',
                              start_time: Union[datetime, str] = None,
                              end_time: Union[datetime, str] = None,
                              timeframes: List[str] = None,
                              load_features: bool = True) -> bool:
        """Initialize the simulator for a specific symbol - minimal implementation"""
        self.current_symbol = symbol
        self.current_mode = mode

        # Convert timestamps to pandas Timestamp objects
        if isinstance(start_time, str):
            self.start_timestamp = pd.Timestamp(start_time)
        else:
            self.start_timestamp = start_time

        if isinstance(end_time, str):
            # Make sure end_time includes a time component
            if 'T' not in end_time and ' ' not in end_time:
                # If just a date is provided, set to end of day
                self.end_timestamp = pd.Timestamp(f"{end_time} 23:59:59")
            else:
                self.end_timestamp = pd.Timestamp(end_time)
        else:
            self.end_timestamp = end_time

        # Ensure consistent timezone information
        if self.start_timestamp and self.end_timestamp:
            if self.start_timestamp.tzinfo is not None and self.end_timestamp.tzinfo is None:
                # Make end_timestamp timezone-aware
                self.end_timestamp = self.end_timestamp.tz_localize(self.start_timestamp.tzinfo)
            elif self.start_timestamp.tzinfo is None and self.end_timestamp.tzinfo is not None:
                # Make start_timestamp timezone-aware
                self.start_timestamp = self.start_timestamp.tz_localize(self.end_timestamp.tzinfo)

        self._log(f"Initializing market from {self.start_timestamp} to {self.end_timestamp}")

        # Load data
        timeframes = timeframes or ["1m", "5m"]
        data_types = [f"bars_{tf}" for tf in timeframes]
        data_types.extend(["trades", "quotes", "status"])

        data_dict = self.data_manager.load_data(symbol, start_time, end_time, data_types)

        if not data_dict:
            self._log(f"Failed to load data for {symbol}", logging.ERROR)
            return False

        # Store raw data
        self.raw_data = data_dict

        # Print data stats
        for key, df in data_dict.items():
            if not df.empty:
                self._log(f"Loaded {len(df)} rows of {key} data")
            else:
                self._log(f"No data loaded for {key}")

        # Initialize components
        # First the market simulator
        quotes_df = data_dict.get('quotes', pd.DataFrame())
        trades_df = data_dict.get('trades', pd.DataFrame())

        # Get the appropriate bar data
        bars_df = None
        for tf in ['bars_1m', 'bars_5m', 'bars_1d']:
            if tf in data_dict and not data_dict[tf].empty:
                bars_df = data_dict[tf]
                break

        status_df = data_dict.get('status', pd.DataFrame())

        # Initialize simulators
        self.market_simulator.initialize_from_data(quotes_df, trades_df, bars_df, status_df)
        self.execution_simulator.set_market_simulator(self.market_simulator)
        self.portfolio_simulator.set_market_simulator(self.market_simulator)
        self.portfolio_simulator.set_execution_simulator(self.execution_simulator)

        # Extract features if requested
        if load_features:
            self._log(f"Extracting features for {symbol}")
            self.features_cache[symbol] = self.feature_extractor.extract_features(data_dict)

            if symbol in self.features_cache and not self.features_cache[symbol].empty:
                # Initialize state manager
                first_timestamp = self.features_cache[symbol].index[0]
                self.state_manager.update_from_features(self.features_cache[symbol], first_timestamp)
                self.current_timestamp = first_timestamp

                # Update simulators
                self._update_simulators_to_timestamp(first_timestamp)
            else:
                self._log(f"No features extracted for {symbol}", logging.WARNING)

        return True

    def _update_simulators_to_timestamp(self, timestamp: datetime) -> None:
        """Update all simulators to a specific timestamp"""
        self.market_simulator.update_to_timestamp(timestamp)
        self.execution_simulator.update_to_timestamp(timestamp)
        self.portfolio_simulator.update_to_timestamp(timestamp)

    # Fixed reset() method for simulator.py
    def reset(self, random_day: bool = False) -> Dict[str, Any]:
        """Reset the simulator for a new episode"""
        # Reset counters
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        # IMPORTANT: Create fresh simulator instances
        self.market_simulator = MarketSimulator(self.market_config, logger=self.logger)
        self.execution_simulator = ExecutionSimulator(self.market_simulator, self.execution_config, logger=self.logger)
        self.portfolio_simulator = PortfolioSimulator(
            self.market_simulator,
            self.execution_simulator,
            self.portfolio_config,
            logger=self.logger
        )

        # Re-initialize market simulator with original data
        if self.raw_data:
            quotes_df = self.raw_data.get('quotes', pd.DataFrame())
            trades_df = self.raw_data.get('trades', pd.DataFrame())

            # Get appropriate bar data
            bars_df = None
            for tf in ['bars_1m', 'bars_5m', 'bars_1d']:
                if tf in self.raw_data and not self.raw_data[tf].empty:
                    bars_df = self.raw_data[tf]
                    break

            status_df = self.raw_data.get('status', pd.DataFrame())

            # Re-initialize with original data
            self.market_simulator.initialize_from_data(quotes_df, trades_df, bars_df, status_df)

        # Reset state manager
        self.state_manager.reset()

        # Set initial timestamp
        if self.features_cache and self.current_symbol in self.features_cache:
            # Sort features by timestamp to ensure we start at the earliest
            features_df = self.features_cache[self.current_symbol]
            if not features_df.empty:
                # Get earliest timestamp
                reset_timestamp = features_df.index[0]
                self.current_timestamp = reset_timestamp

                # Update state manager
                self.state_manager.update_from_features(self.features_cache[self.current_symbol], reset_timestamp)

        # Update simulators to the reset timestamp
        if self.current_timestamp:
            self._update_simulators_to_timestamp(self.current_timestamp)

        # Get initial state
        initial_state = self.state_manager.get_state_dict()

        self._log(f"Resetting to timestamp: {self.current_timestamp}")

        return initial_state

    def step(self, action: float) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment by executing an action"""
        # Record state before action
        prev_portfolio_value = self.portfolio_simulator.get_portfolio_value()

        # Execute action
        result = self.portfolio_simulator.execute_action(
            self.current_symbol,
            action,
            timestamp=self.current_timestamp
        )

        # Enhance info dictionary with data for reward function
        info = {
            'timestamp': self.current_timestamp,
            'action_result': result,
            'portfolio_value': self.portfolio_simulator.get_portfolio_value(),
            'portfolio_change': self.portfolio_simulator.get_portfolio_value() - prev_portfolio_value,
            'step_count': self.step_count,
            'prev_portfolio_value': prev_portfolio_value
        }

        # Add market state information for reward calculation
        market_state = self.market_simulator.get_current_market_state()
        info['price'] = market_state.get('price', 0)

        # Add feature-based signals for reward calculation
        if self.current_symbol in self.features_cache:
            features = self.features_cache[self.current_symbol]
            current_features = features[features.index <= self.current_timestamp]

            if not current_features.empty:
                latest_features = current_features.iloc[-1]

                # Extract momentum and volume signals for reward calculation
                for col in latest_features.index:
                    if 'momentum' in col:
                        info['momentum_strength'] = latest_features[col]
                    if 'rel_volume' in col:
                        info['relative_volume'] = latest_features[col]
                    if 'tape_speed' in col:
                        info['tape_speed'] = latest_features[col]
                    if 'tape_imbalance' in col:
                        info['tape_imbalance'] = latest_features[col]


        # Record trade if executed
        if result['success'] and result.get('action') != 'hold' and self.trade_callbacks:
            for callback in self.trade_callbacks:
                callback(result)

        # Get next timestamp
        next_timestamp = self._get_next_timestamp()

        # Check if we've reached the end
        if next_timestamp is None:
            # End of data
            self._log("Reached end of data")
            done = True
        elif self.end_timestamp is not None:
            # Ensure timestamps have the same timezone info before comparison
            end_ts = self.end_timestamp
            next_ts = next_timestamp

            # Make timezone info consistent
            if end_ts.tzinfo is not None and next_ts.tzinfo is None:
                next_ts = next_ts.tz_localize(end_ts.tzinfo)
            elif end_ts.tzinfo is None and next_ts.tzinfo is not None:
                end_ts = end_ts.tz_localize(next_ts.tzinfo)

            done = next_ts > end_ts
            if done:
                self._log(f"Reached end timestamp: {self.end_timestamp}")
        else:
            # Not done, update to next timestamp
            done = False

        if not done:
            # Update timestamp and simulators
            self.current_timestamp = next_timestamp
            self._update_simulators_to_timestamp(next_timestamp)

            # Update state
            self.state_manager.update_from_features(self.features_cache[self.current_symbol], next_timestamp)

        # Calculate reward (simple portfolio change)
        current_portfolio_value = self.portfolio_simulator.get_portfolio_value()
        portfolio_change = current_portfolio_value - prev_portfolio_value
        reward = portfolio_change
        self.total_reward += reward

        # Get next state
        next_state = self.state_manager.get_state_dict()

        # Update step count
        self.step_count += 1

        # Create info dict
        info = {
            'timestamp': self.current_timestamp,
            'action_result': result,
            'portfolio_value': current_portfolio_value,
            'portfolio_change': portfolio_change,
            'step_count': self.step_count
        }

        # Add episode info if done
        if done:
            total_pnl = current_portfolio_value - self.portfolio_simulator.initial_cash
            total_pnl_pct = total_pnl / self.portfolio_simulator.initial_cash
            trade_stats = self.portfolio_simulator.get_statistics()

            info['episode'] = {
                'steps': self.step_count,
                'total_reward': self.total_reward,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'trade_count': trade_stats.get('total_trades', 0),
                'win_rate': trade_stats.get('win_rate', 0.0)
            }

            self._log(
                f"Episode {self.episode_count} finished: {self.step_count} steps, reward: {self.total_reward:.2f}, PnL: ${total_pnl:.2f}")
            self._log(
                f"Episode finished - PnL: ${total_pnl:.2f} ({total_pnl_pct:.2%}), Win Rate: {trade_stats.get('win_rate', 0.0):.1%}, Trades: {trade_stats.get('total_trades', 0)}")

        return next_state, reward, done, info

    def _get_next_timestamp(self) -> datetime:
        """Get the next timestamp in the sequence - minimal implementation"""
        if self.current_symbol not in self.features_cache or self.features_cache[self.current_symbol].empty:
            return None

        # Find next timestamp in feature data
        features_df = self.features_cache[self.current_symbol]

        # Get timestamps AFTER the current timestamp
        future_timestamps = features_df[features_df.index > self.current_timestamp].index

        if len(future_timestamps) == 0:
            return None  # No more timestamps

        # Return the next timestamp
        return future_timestamps[0]
    def add_state_update_callback(self, callback_fn: Callable):
        """Add a callback for state updates"""
        self.state_update_callbacks.append(callback_fn)

    def add_trade_callback(self, callback_fn: Callable):
        """Add a callback for trades"""
        self.trade_callbacks.append(callback_fn)

    def add_portfolio_update_callback(self, callback_fn: Callable):
        """Add a callback for portfolio updates"""
        self.portfolio_update_callbacks.append(callback_fn)

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state"""
        return self.state_manager.get_state_dict()

    def get_current_state_array(self) -> np.ndarray:
        """Get the current state as an array"""
        return self.state_manager.get_state_array()

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get the current portfolio state"""
        return self.portfolio_simulator.get_portfolio_state()

    def get_market_state(self) -> Dict[str, Any]:
        """Get the current market state"""
        return self.market_simulator.get_current_market_state()

    def get_trade_statistics(self) -> Dict[str, float]:
        """Get trading statistics"""
        return self.portfolio_simulator.get_statistics()

    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.portfolio_simulator.get_trade_history()