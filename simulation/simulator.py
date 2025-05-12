# simulation/simulator.py
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json

from data.data_manager import DataManager
from feature.feature_extractor import FeatureExtractor
from feature.state_manager import StateManager
from simulation.market_simulator import MarketSimulator
from simulation.execution_simulator import ExecutionSimulator
from simulation.portfolio_simulator import PortfolioSimulator


class Simulator:
    """
    Main simulator that coordinates all components:
    - Data Management
    - Feature Extraction
    - Market Simulation
    - Order Execution
    - Portfolio Management
    - Event Coordination

    Acts as the central component for training and backtesting,
    providing a gym-like environment for RL agents.
    """

    def __init__(self,
                 data_manager: DataManager,
                 config: Dict = None,
                 logger: logging.Logger = None):
        """
        Initialize the simulator.

        Args:
            data_manager: DataManager instance for data access
            config: Configuration dictionary with simulator parameters
            logger: Optional logger
        """
        self.data_manager = data_manager
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Configuration for each component
        self.feature_config = self.config.get('feature_config', {})
        self.state_config = self.config.get('state_config', {})
        self.market_config = self.config.get('market_config', {})
        self.execution_config = self.config.get('execution_config', {})
        self.portfolio_config = self.config.get('portfolio_config', {})

        # Initialize components
        self.feature_extractor = FeatureExtractor(self.feature_config, logger=logger)
        self.state_manager = StateManager(self.feature_config, self.state_config, logger=logger)

        self.market_simulator = MarketSimulator(self.market_config, logger=logger)
        self.execution_simulator = ExecutionSimulator(self.market_simulator, self.execution_config, logger=logger)
        self.portfolio_simulator = PortfolioSimulator(
            self.market_simulator,
            self.execution_simulator,
            self.portfolio_config,
            logger=logger
        )

        # Features cache for different symbols
        self.features_cache = {}

        # Callbacks for events
        self.state_update_callbacks = []
        self.feature_update_callbacks = []
        self.trade_callbacks = []
        self.portfolio_update_callbacks = []
        self.order_callbacks = []

        # Mode tracking
        self.current_mode = 'idle'  # 'idle', 'training', 'backtesting', 'live'
        self.current_symbol = None
        self.current_timestamp = None
        self.start_timestamp = None
        self.end_timestamp = None

        # Episode tracking for RL
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_trades = []
        self.episode_steps = []
        self.episode_pnls = []

        # Data buffers
        self.raw_data = {}

        # Save/Load path
        self.save_path = self.config.get('save_path', 'sim_data')
        os.makedirs(self.save_path, exist_ok=True)

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def initialize_for_symbol(self, symbol: str,
                              mode: str = 'backtesting',
                              start_time: Union[datetime, str] = None,
                              end_time: Union[datetime, str] = None,
                              timeframes: List[str] = None,
                              load_features: bool = True) -> bool:
        """
        Initialize the simulator for a specific symbol.

        Args:
            symbol: Symbol to initialize
            mode: Operation mode ('training', 'backtesting', 'live')
            start_time: Start time for historical data
            end_time: End time for historical data
            timeframes: List of timeframes to load
            load_features: Whether to load and compute features immediately

        Returns:
            Boolean indicating success
        """
        # Reset state
        self.current_symbol = symbol
        self.current_mode = mode

        # Ensure timestamps are timezone-aware for consistency
        if start_time:
            self.start_timestamp = pd.Timestamp(start_time)
            if self.start_timestamp.tzinfo is None:
                self.start_timestamp = self.start_timestamp.tz_localize('UTC')
        if end_time:
            self.end_timestamp = pd.Timestamp(end_time)
            if self.end_timestamp.tzinfo is None:
                self.end_timestamp = self.end_timestamp.tz_localize('UTC')

        # Clear existing features for this symbol
        if symbol in self.features_cache:
            del self.features_cache[symbol]

        # Reset simulators
        self.market_simulator = MarketSimulator(self.market_config, logger=self.logger)
        self.execution_simulator = ExecutionSimulator(self.market_simulator, self.execution_config, logger=self.logger)
        self.portfolio_simulator = PortfolioSimulator(
            self.market_simulator,
            self.execution_simulator,
            self.portfolio_config,
            logger=self.logger
        )

        # Reset state manager
        self.state_manager.reset()

        # Reset episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_trades = []
        self.episode_steps = []
        self.episode_pnls = []

        # Initialize based on mode
        if mode in ('training', 'backtesting'):
            if not start_time or not end_time:
                self._log(f"start_time and end_time required for {mode} mode", logging.ERROR)
                return False

            # Load the data
            data_types = []
            if timeframes:
                data_types = [f"bars_{tf}" for tf in timeframes]
                data_types.extend(["trades", "quotes", "status"])

            self._log(f"Loading data for {symbol} from {start_time} to {end_time}")
            data_dict = self.data_manager.load_data(symbol, start_time, end_time, data_types)

            if not data_dict:
                self._log(f"Failed to load data for {symbol}", logging.ERROR)
                return False

            # Store the raw data
            self.raw_data = data_dict

            # Now sync all data to start from the same time
            # Find common start time to ensure all data sources align
            common_start_time = self._find_common_start_time(data_dict)
            if common_start_time is None:
                self._log(f"Failed to find common start time across all data sources", logging.ERROR)
                return False

            self._log(f"Identified common start time: {common_start_time}")
            self.current_timestamp = common_start_time

            # Initialize market simulator with data
            quotes_df = data_dict.get('quotes', pd.DataFrame())
            trades_df = data_dict.get('trades', pd.DataFrame())

            # Get appropriate bar data - prefer 1m if available
            bars_df = None
            for tf in ['bars_1m', 'bars_1s', 'bars_5m', 'bars_1d']:
                if tf in data_dict and not data_dict[tf].empty:
                    bars_df = data_dict[tf]
                    break

            status_df = data_dict.get('status', pd.DataFrame())

            # Filter all data to start from the common start time
            if not quotes_df.empty:
                quotes_df = quotes_df[quotes_df.index >= common_start_time]
            if not trades_df.empty:
                trades_df = trades_df[trades_df.index >= common_start_time]
            if bars_df is not None and not bars_df.empty:
                bars_df = bars_df[bars_df.index >= common_start_time]
            if not status_df.empty:
                status_df = status_df[status_df.index >= common_start_time]

            self.market_simulator.initialize_from_data(quotes_df, trades_df, bars_df, status_df)

            if load_features:
                # Extract features
                self._log(f"Extracting features for {symbol}")
                self.features_cache[symbol] = self.feature_extractor.extract_features(data_dict)

                if symbol in self.features_cache and self.features_cache[symbol].empty:
                    self._log(f"No features extracted for {symbol}", logging.WARNING)
                    return False

                # Filter features to start from common start time
                if not self.features_cache[symbol].empty:
                    self.features_cache[symbol] = self.features_cache[symbol][
                        self.features_cache[symbol].index >= common_start_time]
                    if self.features_cache[symbol].empty:
                        self._log(f"No features left after filtering to common start time", logging.WARNING)
                        return False

                # Notify feature update callbacks
                for callback in self.feature_update_callbacks:
                    callback(symbol, self.features_cache[symbol])

                # Initialize state manager with the first timestamp
                if not self.features_cache[symbol].empty:
                    first_timestamp = self.features_cache[symbol].index[0]
                    self.state_manager.update_from_features(self.features_cache[symbol], first_timestamp)
                    self.current_timestamp = first_timestamp

                    # Update simulators to the first timestamp
                    self.market_simulator.update_to_timestamp(first_timestamp)
                    self.execution_simulator.update_to_timestamp(first_timestamp)
                    self.portfolio_simulator.update_to_timestamp(first_timestamp)

                    # Notify state update callbacks
                    for callback in self.state_update_callbacks:
                        callback(self.state_manager.get_state_dict())

            return True

        elif mode == 'live':
            # Initialize live data
            if not timeframes:
                timeframes = ["1s", "1m", "5m", "1d"]

            self.data_manager.initialize_live_data(symbol, timeframes)

            # Features and state will be updated as data arrives
            return True

        else:
            self._log(f"Invalid mode: {mode}", logging.ERROR)
            return False

    def _find_common_start_time(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[datetime]:
        """
        Find the latest start time across all data sources to ensure synchronization.

        Args:
            data_dict: Dictionary of DataFrames with market data

        Returns:
            Common start timestamp or None if no valid time can be found
        """
        start_times = []

        # Collect all start times
        for key, df in data_dict.items():
            if not df.empty and hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
                start_times.append(df.index.min())

        if not start_times:
            return None

        # Find the latest start time (all data sources will have data from this point)
        latest_start = max(start_times)

        # Check if we have data from all sources after this time
        valid_start = True
        for key, df in data_dict.items():
            if not df.empty and hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
                # If this data source doesn't have entries after the latest start time
                if df.index.min() > latest_start:
                    valid_start = False
                    break

        return latest_start if valid_start else None

    def reset(self, random_day: bool = False) -> Dict[str, Any]:
        """
        Reset the simulator to start a new episode.

        Args:
            random_day: If True, randomly select a trading day (for training)

        Returns:
            Initial state dictionary
        """
        # Reset episode counters
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        # Reset simulators
        self.market_simulator = MarketSimulator(self.market_config, logger=self.logger)
        self.execution_simulator = ExecutionSimulator(self.market_simulator, self.execution_config, logger=self.logger)
        self.portfolio_simulator = PortfolioSimulator(
            self.market_simulator,
            self.execution_simulator,
            self.portfolio_config,
            logger=self.logger
        )

        # Reset state manager
        self.state_manager.reset()

        if self.current_mode not in ('training', 'backtesting'):
            self._log("Reset only applicable in training or backtesting mode", logging.WARNING)
            return {}

        # Get data for initialization
        if self.raw_data:
            # Find common start time
            common_start_time = self._find_common_start_time(self.raw_data)
            if common_start_time is None:
                self._log("Failed to find common start time during reset", logging.ERROR)
                return {}

            # Filter data to start from common start time
            quotes_df = self.raw_data.get('quotes', pd.DataFrame())
            if not quotes_df.empty:
                quotes_df = quotes_df[quotes_df.index >= common_start_time]

            trades_df = self.raw_data.get('trades', pd.DataFrame())
            if not trades_df.empty:
                trades_df = trades_df[trades_df.index >= common_start_time]

            # Get appropriate bar data - prefer 1m if available
            bars_df = None
            for tf in ['bars_1m', 'bars_1s', 'bars_5m', 'bars_1d']:
                if tf in self.raw_data and not self.raw_data[tf].empty:
                    bars_df = self.raw_data[tf]
                    if bars_df is not None and not bars_df.empty:
                        bars_df = bars_df[bars_df.index >= common_start_time]
                    break

            status_df = self.raw_data.get('status', pd.DataFrame())
            if not status_df.empty:
                status_df = status_df[status_df.index >= common_start_time]

            # Initialize market simulator with data
            self.market_simulator.initialize_from_data(quotes_df, trades_df, bars_df, status_df)

        # Determine start time
        if random_day and bars_df is not None and not bars_df.empty:
            # Get all available days
            days = bars_df.index.normalize().unique()
            if len(days) > 1:
                # Randomly select a day
                selected_day = np.random.choice(days)

                # Get data for that day
                day_start = selected_day
                day_end = selected_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                # Find actual data range within that day
                day_data = bars_df[(bars_df.index >= day_start) & (bars_df.index <= day_end)]
                if not day_data.empty:
                    first_timestamp = day_data.index[0]
                    self.current_timestamp = first_timestamp
                else:
                    # Fallback to first timestamp
                    first_timestamp = bars_df.index[0]
                    self.current_timestamp = first_timestamp
            else:
                # Only one day available
                first_timestamp = bars_df.index[0]
                self.current_timestamp = first_timestamp
        else:
            # Use the first synchronized timestamp
            first_timestamp = common_start_time
            if self.features_cache and self.current_symbol in self.features_cache:
                # Filter features to ensure they start at or after the common start time
                features_df = self.features_cache[self.current_symbol]
                if not features_df.empty:
                    features_df = features_df[features_df.index >= common_start_time]
                    if not features_df.empty:
                        first_timestamp = features_df.index[0]

            self.current_timestamp = first_timestamp

        # Update simulators to the first timestamp
        if self.current_timestamp:
            self._log(f"Resetting to timestamp: {self.current_timestamp}")
            self.market_simulator.update_to_timestamp(self.current_timestamp)
            self.execution_simulator.update_to_timestamp(self.current_timestamp)
            self.portfolio_simulator.update_to_timestamp(self.current_timestamp)

        # Update state
        if self.features_cache and self.current_symbol in self.features_cache:
            self.state_manager.update_from_features(self.features_cache[self.current_symbol], self.current_timestamp)

        # Get the initial state
        initial_state = self.state_manager.get_state_dict()

        # Notify callbacks
        for callback in self.state_update_callbacks:
            callback(initial_state)

        # Return the initial state
        return initial_state

    def step(self, action: float) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by executing an action.

        Args:
            action: Action to take (-1.0 to 1.0)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_mode not in ('training', 'backtesting'):
            self._log("Step function only applicable in training or backtesting mode", logging.WARNING)
            return {}, 0.0, True, {}

        if self.current_timestamp is None:
            self._log("Cannot step: no current timestamp set", logging.ERROR)
            return {}, 0.0, True, {'error': 'no_timestamp'}

        # Record the state before action
        prev_portfolio_value = self.portfolio_simulator.get_portfolio_value()
        prev_position = self.portfolio_simulator.get_position(self.current_symbol)
        prev_position_size = prev_position['quantity'] if prev_position else 0

        # Execute the action
        result = self.portfolio_simulator.execute_action(
            self.current_symbol,
            action,
            timestamp=self.current_timestamp
        )

        # Record trade if a trade was executed
        if result['success'] and result.get('action') != 'hold':
            for callback in self.trade_callbacks:
                callback(result)

        # Move to the next timestep
        next_timestamp = self._get_next_timestamp()

        # Update step counter now, so it's always accurate even if we return early
        self.step_count += 1

        # Handle episode termination
        done = False

        # Check if we've reached the end of data
        if next_timestamp is None:
            self._log(f"End of data reached after {self.step_count} steps", logging.INFO)
            done = True
        elif self.end_timestamp is not None:
            # Ensure consistent timezone awareness before comparison
            end_ts = pd.Timestamp(self.end_timestamp)
            next_ts = pd.Timestamp(next_timestamp)

            # Convert both to UTC if needed
            if end_ts.tzinfo is not None and next_ts.tzinfo is None:
                next_ts = next_ts.tz_localize('UTC')
            elif end_ts.tzinfo is None and next_ts.tzinfo is not None:
                end_ts = end_ts.tz_localize('UTC')

            done = next_ts > end_ts
            if done:
                self._log(f"Reached end timestamp: {end_ts}", logging.INFO)

        # Check for maximum steps
        if self.step_count >= self.config.get('max_steps', 1000):
            done = True
            self._log(f"Reached maximum steps: {self.step_count}", logging.INFO)

        if done:
            # Force close any open positions
            if prev_position_size != 0:
                self._log(f"Closing position of {prev_position_size} at episode end", logging.INFO)
                self.portfolio_simulator.execute_action(
                    self.current_symbol,
                    0.0,  # Flat position
                    timestamp=self.current_timestamp
                )

            # We don't need to update to next timestamp if done
            next_state = self.state_manager.get_state_dict()
        else:
            # Advance time
            self.current_timestamp = next_timestamp
            self._log(f"Advancing to timestamp: {self.current_timestamp}", logging.DEBUG)

            # Update simulators
            self.market_simulator.update_to_timestamp(self.current_timestamp)
            self.execution_simulator.update_to_timestamp(self.current_timestamp)
            self.portfolio_simulator.update_to_timestamp(self.current_timestamp)

            # Update state
            if self.features_cache and self.current_symbol in self.features_cache:
                self.state_manager.update_from_features(self.features_cache[self.current_symbol],
                                                        self.current_timestamp)

            # Get the next state
            next_state = self.state_manager.get_state_dict()

        # Calculate reward
        reward = self._calculate_reward(result, prev_portfolio_value)

        # Notify callbacks
        for callback in self.state_update_callbacks:
            callback(next_state)

        for callback in self.portfolio_update_callbacks:
            callback(self.portfolio_simulator.get_portfolio_state())

        # Update episode tracking
        self.total_reward += reward

        # Create info dict
        info = {
            'timestamp': self.current_timestamp,
            'action_result': result,
            'portfolio_value': self.portfolio_simulator.get_portfolio_value(),
            'portfolio_change': self.portfolio_simulator.get_portfolio_value() - prev_portfolio_value,
            'position': self.portfolio_simulator.get_position(self.current_symbol),
            'trade_stats': self.portfolio_simulator.get_statistics(),
            'market_state': self.market_simulator.get_current_market_state(),
            'step_count': self.step_count
        }

        # If episode is done, record episode stats
        if done:
            episode_summary = {
                'episode': self.episode_count,
                'steps': self.step_count,
                'total_reward': self.total_reward,
                'final_portfolio_value': self.portfolio_simulator.get_portfolio_value(),
                'pnl': self.portfolio_simulator.get_portfolio_value() - self.portfolio_config.get('initial_cash',
                                                                                                  100000.0),
                'trade_count': len(self.portfolio_simulator.get_trade_history()),
                'win_rate': self.portfolio_simulator.get_statistics().get('win_rate', 0.0),
                'sharpe_ratio': self.portfolio_simulator.get_statistics().get('sharpe_ratio', 0.0),
                'max_drawdown': self.portfolio_simulator.get_statistics().get('max_drawdown', 0.0)
            }

            self.episode_rewards.append(self.total_reward)
            self.episode_steps.append(self.step_count)
            self.episode_pnls.append(episode_summary['pnl'])
            self.episode_trades.append(episode_summary['trade_count'])

            info['episode'] = episode_summary

            self._log(
                f"Episode {self.episode_count} finished: {self.step_count} steps, reward: {self.total_reward:.2f}, PnL: ${episode_summary['pnl']:.2f}")

        return next_state, reward, done, info

    def _get_next_timestamp(self) -> Optional[datetime]:
        """
        Get the next timestamp in the sequence.

        Returns:
            Next timestamp or None if no more data
        """
        if self.current_symbol not in self.features_cache or self.features_cache[self.current_symbol].empty:
            self._log("No features data available to get next timestamp", logging.WARNING)
            return None

        features_df = self.features_cache[self.current_symbol]

        # Debug info
        self._log(f"Looking for next timestamp after {self.current_timestamp}", logging.DEBUG)

        # Make sure current timestamp has consistent timezone info
        current_ts = pd.Timestamp(self.current_timestamp)
        if current_ts.tzinfo is None and features_df.index.tzinfo is not None:
            current_ts = current_ts.tz_localize(features_df.index.tzinfo)
        elif current_ts.tzinfo is not None and features_df.index.tzinfo is None:
            features_df_localized = features_df.copy()
            features_df_localized.index = features_df_localized.index.tz_localize(current_ts.tzinfo)
            features_df = features_df_localized

        try:
            # Simple approach - find timestamps greater than current
            future_timestamps = features_df.index[features_df.index > current_ts]

            if len(future_timestamps) == 0:
                self._log("No more timestamps available in features data", logging.INFO)
                return None

            # Get the immediate next timestamp
            next_timestamp = future_timestamps[0]
            self._log(f"Next timestamp found: {next_timestamp}", logging.DEBUG)
            return next_timestamp

        except Exception as e:
            # Log the error for debugging
            self._log(f"Error finding next timestamp: {e}", logging.ERROR)
            return None

    def _calculate_reward(self, action_result: Dict[str, Any], prev_portfolio_value: float) -> float:
        """
        Calculate reward based on the action result and portfolio change.

        Args:
            action_result: Result of the executed action
            prev_portfolio_value: Portfolio value before the action

        Returns:
            Calculated reward
        """
        # Get current portfolio value
        current_portfolio_value = self.portfolio_simulator.get_portfolio_value()

        # Calculate portfolio change
        portfolio_change = current_portfolio_value - prev_portfolio_value

        # Check for realized PnL (from a closed trade)
        realized_pnl = action_result.get('realized_pnl', 0.0) if action_result.get('success') else 0.0

        # Base reward is the portfolio change
        reward = portfolio_change

        # Add bonus for realized profits
        if realized_pnl > 0:
            reward += realized_pnl * 0.1  # Bonus for profits

        # Penalize trading costs
        if action_result.get('success') and action_result.get('action') != 'hold':
            # Apply small penalty for trading (to discourage excessive trading)
            reward -= 1.0  # Fixed penalty per trade

        # Penalize trying to average down
        if action_result.get('success') is False and action_result.get('reason') == 'insufficient_cash':
            reward -= 5.0  # Penalty for bad cash management

        # Check for maximum drawdown violation
        max_drawdown_pct = self.portfolio_simulator.get_statistics().get('max_drawdown_pct', 0.0)
        if max_drawdown_pct > self.portfolio_config.get('max_drawdown_pct', 0.05):
            reward -= 10.0  # Severe penalty for exceeding max drawdown

        return reward

    def execute_action(self, action: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Execute a trading action directly.

        Args:
            action: Normalized action (-1.0 to 1.0)
            timestamp: Timestamp for the action (defaults to current timestamp)

        Returns:
            Action result dictionary
        """
        action_time = timestamp or self.current_timestamp

        if action_time is None:
            self._log("Cannot execute action: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'reason': 'no_timestamp'}

        # If we're moving forward in time, update simulators
        if action_time > self.current_timestamp:
            self.market_simulator.update_to_timestamp(action_time)
            self.execution_simulator.update_to_timestamp(action_time)
            self.portfolio_simulator.update_to_timestamp(action_time)
            self.current_timestamp = action_time

        # Execute the action
        result = self.portfolio_simulator.execute_action(
            self.current_symbol,
            action,
            timestamp=action_time
        )

        # Update state manager position
        if result['success']:
            # Get the updated position
            position = self.portfolio_simulator.get_position(self.current_symbol)
            position_size = position['quantity'] if position else 0
            normalized_position = position_size / self.portfolio_config.get('position_size_limits', {}).get(
                self.current_symbol, 100)

            # Get the price
            price = result.get('fill_price') if result.get('success') else self.market_simulator.current_price

            # Update state manager
            self.state_manager.update_position(normalized_position, price, action_time)

        # Notify callbacks
        for callback in self.trade_callbacks:
            callback(result)

        for callback in self.portfolio_update_callbacks:
            callback(self.portfolio_simulator.get_portfolio_state())

        return result

    def update_live_data(self) -> Dict[str, Any]:
        """
        Update data and state based on latest live data.

        Returns:
            Current state dictionary
        """
        if self.current_mode != 'live':
            self._log("update_live_data should only be called in live mode", logging.WARNING)
            return {}

        # Get latest data from data manager
        latest_data = self.data_manager.get_data(self.current_symbol)

        if not latest_data:
            self._log(f"No data available for {self.current_symbol}", logging.WARNING)
            return {}

        # Extract features
        if self.current_symbol in self.features_cache and not self.features_cache[self.current_symbol].empty:
            # Update existing features
            self.features_cache[self.current_symbol] = self.feature_extractor.update_features(
                latest_data, self.features_cache[self.current_symbol])
        else:
            # Extract features from scratch
            self.features_cache[self.current_symbol] = self.feature_extractor.extract_features(latest_data)

        if self.current_symbol not in self.features_cache or self.features_cache[self.current_symbol].empty:
            self._log(f"No features extracted for {self.current_symbol}", logging.WARNING)
            return {}

        # Notify feature update callbacks
        for callback in self.feature_update_callbacks:
            callback(self.current_symbol, self.features_cache[self.current_symbol])

        # Get the latest timestamp
        latest_timestamp = self.features_cache[self.current_symbol].index[-1]

        # Update simulators
        self.market_simulator.update_to_timestamp(latest_timestamp)
        self.execution_simulator.update_to_timestamp(latest_timestamp)
        self.portfolio_simulator.update_to_timestamp(latest_timestamp)

        # Update state manager
        self.state_manager.update_from_features(self.features_cache[self.current_symbol], latest_timestamp)

        # Get the current state
        state_dict = self.state_manager.get_state_dict()

        # Notify state update callbacks
        for callback in self.state_update_callbacks:
            callback(state_dict)

        # Update current timestamp
        self.current_timestamp = latest_timestamp

        return state_dict

    def get_features(self, symbol: str = None) -> pd.DataFrame:
        """
        Get cached features for a symbol.

        Args:
            symbol: Symbol to get features for (defaults to current_symbol)

        Returns:
            DataFrame with features
        """
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified", logging.ERROR)
            return pd.DataFrame()

        if symbol not in self.features_cache:
            self._log(f"No features cached for {symbol}", logging.WARNING)
            return pd.DataFrame()

        return self.features_cache[symbol]

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        return self.state_manager.get_state_dict()

    def get_current_state_array(self) -> np.ndarray:
        """Get the current state as a NumPy array."""
        return self.state_manager.get_state_array()

    def get_state_at_time(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get the state at a specific timestamp.

        Args:
            timestamp: Timestamp to get state for

        Returns:
            State dictionary
        """
        return self.state_manager.get_state_at_time(timestamp)

    def add_state_update_callback(self, callback_fn: Callable):
        """
        Add a callback for state updates.

        Args:
            callback_fn: Function that takes a state dictionary
        """
        self.state_update_callbacks.append(callback_fn)

    def add_feature_update_callback(self, callback_fn: Callable):
        """
        Add a callback for feature updates.

        Args:
            callback_fn: Function that takes a symbol and features DataFrame
        """
        self.feature_update_callbacks.append(callback_fn)

    def add_trade_callback(self, callback_fn: Callable):
        """
        Add a callback for completed trades.

        Args:
            callback_fn: Function that takes a trade dictionary
        """
        self.trade_callbacks.append(callback_fn)

    def add_portfolio_update_callback(self, callback_fn: Callable):
        """
        Add a callback for portfolio updates.

        Args:
            callback_fn: Function that takes a portfolio state dictionary
        """
        self.portfolio_update_callbacks.append(callback_fn)

    def add_order_callback(self, callback_fn: Callable):
        """
        Add a callback for order updates.

        Args:
            callback_fn: Function that takes an order dictionary
        """
        self.order_callbacks.append(callback_fn)

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.

        Returns:
            Portfolio state dictionary
        """
        return self.portfolio_simulator.get_portfolio_state()

    def get_market_state(self) -> Dict[str, Any]:
        """
        Get the current market state.

        Returns:
            Market state dictionary
        """
        return self.market_simulator.get_current_market_state()

    def get_trade_statistics(self) -> Dict[str, float]:
        """
        Get statistics about completed trades.

        Returns:
            Dictionary with trade statistics
        """
        return self.portfolio_simulator.get_statistics()

    def get_trade_history(self) -> List[Dict]:
        """
        Get history of completed trades.

        Returns:
            List of trade dictionaries
        """
        return self.portfolio_simulator.get_trade_history()

    def get_portfolio_history(self) -> pd.Series:
        """
        Get portfolio value history.

        Returns:
            Series with portfolio values
        """
        return self.portfolio_simulator.get_portfolio_history()

    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about episodes.

        Returns:
            Dictionary with episode statistics
        """
        return {
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_pnl': np.mean(self.episode_pnls) if self.episode_pnls else 0.0,
            'mean_steps': np.mean(self.episode_steps) if self.episode_steps else 0.0,
            'mean_trades': np.mean(self.episode_trades) if self.episode_trades else 0.0
        }

    def save_state(self, filename: str = None) -> str:
        """
        Save the simulator state to a file.

        Args:
            filename: Filename to save to (defaults to timestamp)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sim_state_{timestamp_str}.json"

        filepath = os.path.join(self.save_path, filename)

        # Create state dictionary
        state = {
            'current_mode': self.current_mode,
            'current_symbol': self.current_symbol,
            'current_timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'start_timestamp': self.start_timestamp.isoformat() if self.start_timestamp else None,
            'end_timestamp': self.end_timestamp.isoformat() if self.end_timestamp else None,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'portfolio_state': self.portfolio_simulator.get_portfolio_state(),
            'trade_history': self.portfolio_simulator.get_trade_history(),
            'trade_statistics': self.portfolio_simulator.get_statistics()
        }

        # Convert timestamps and non-serializable objects
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                return {key: convert_timestamps(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        state = convert_timestamps(state)

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        self._log(f"Simulator state saved to {filepath}")

        return filepath

    def load_state(self, filepath: str) -> bool:
        """
        Load the simulator state from a file.

        Args:
            filepath: Path to state file

        Returns:
            Boolean indicating success
        """
        if not os.path.exists(filepath):
            self._log(f"State file {filepath} does not exist", logging.ERROR)
            return False

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore simulator state
            self.current_mode = state['current_mode']
            self.current_symbol = state['current_symbol']

            # Convert ISO timestamps back to pd.Timestamp
            if state['current_timestamp']:
                self.current_timestamp = pd.Timestamp(state['current_timestamp'])
            if state['start_timestamp']:
                self.start_timestamp = pd.Timestamp(state['start_timestamp'])
            if state['end_timestamp']:
                self.end_timestamp = pd.Timestamp(state['end_timestamp'])

            # Restore episode tracking
            self.episode_count = state['episode_count']
            self.step_count = state['step_count']
            self.total_reward = state['total_reward']
            self.episode_rewards = state['episode_rewards']
            self.episode_steps = state['episode_steps']
            self.episode_pnls = state['episode_pnls']
            self.episode_trades = state['episode_trades']

            # Note: We don't restore the raw data, features cache, or simulator states
            # Those need to be reinitialized separately

            self._log(f"Simulator state loaded from {filepath}")
            return True

        except Exception as e:
            self._log(f"Error loading state file {filepath}: {str(e)}", logging.ERROR)
            return False