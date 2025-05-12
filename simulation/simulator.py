# data/simulator.py
from typing import Dict, List, Union, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from data.data_manager import DataManager
from feature.feature_extractor import FeatureExtractor
from feature.state_manager import StateManager


class Simulator:
    """
    Coordinates data, feature extraction, and state management.
    Acts as the central component for training and live trading.
    Designed with:
    - Clear coordination role
    - Minimized data handling (delegated to DataManager)
    - Support for different execution modes (training, backtesting, live)
    - Event-driven architecture
    """

    def __init__(self, data_manager: DataManager,
                 feature_config: Dict = None,
                 state_config: Dict = None,
                 logger: logging.Logger = None):
        """
        Initialize the data processor.

        Args:
            data_manager: DataManager instance for data access
            feature_config: Configuration for FeatureExtractor
            state_config: Configuration for StateManager
            logger: Optional logger
        """
        self.data_manager = data_manager
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.feature_extractor = FeatureExtractor(feature_config, logger=logger)
        self.state_manager = StateManager(feature_config, state_config, logger=logger)

        # Features cache for different symbols
        self.features_cache = {}

        # Callbacks for events
        self.state_update_callbacks = []
        self.feature_update_callbacks = []
        self.trade_callbacks = []

        # Mode tracking
        self.current_mode = 'idle'  # 'idle', 'training', 'backtesting', 'live'
        self.current_symbol = None

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def initialize_for_symbol(self, symbol: str, mode: str = 'backtesting',
                              start_time: Union[datetime, str] = None,
                              end_time: Union[datetime, str] = None,
                              timeframes: List[str] = None) -> bool:
        """
        Initialize the processor for a specific symbol.

        Args:
            symbol: Symbol to initialize
            mode: Operation mode ('training', 'backtesting', 'live')
            start_time: Start time for historical data (if applicable)
            end_time: End time for historical data (if applicable)
            timeframes: List of timeframes to load (if applicable)

        Returns:
            Boolean indicating success
        """
        # Reset state
        self.current_symbol = symbol
        self.current_mode = mode

        # Clear existing features for this symbol
        if symbol in self.features_cache:
            del self.features_cache[symbol]

        # Reset state manager
        self.state_manager.reset()

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

            data_dict = self.data_manager.load_data(symbol, start_time, end_time, data_types)

            if not data_dict:
                self._log(f"Failed to load data for {symbol}", logging.ERROR)
                return False

            # Extract features
            self._log(f"Extracting features for {symbol}")
            self.features_cache[symbol] = self.feature_extractor.extract_features(data_dict)

            if symbol in self.features_cache and self.features_cache[symbol].empty:
                self._log(f"No features extracted for {symbol}", logging.WARNING)
                return False

            # Notify feature update callbacks
            for callback in self.feature_update_callbacks:
                callback(symbol, self.features_cache[symbol])

            # Initialize state manager with the first timestamp
            if not self.features_cache[symbol].empty:
                first_timestamp = self.features_cache[symbol].index[0]
                self.state_manager.update_from_features(self.features_cache[symbol], first_timestamp)

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

    def process_batch(self, symbol: str = None, batch_size: int = 1) -> pd.DataFrame:
        """
        Process a batch of data for training.

        Args:
            symbol: Symbol to process (defaults to current_symbol)
            batch_size: Number of timesteps to process

        Returns:
            DataFrame with processed features
        """
        if self.current_mode != 'training':
            self._log("process_batch should only be called in training mode", logging.WARNING)
            return pd.DataFrame()

        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified", logging.ERROR)
            return pd.DataFrame()

        if symbol not in self.features_cache or self.features_cache[symbol].empty:
            self._log(f"No features cached for {symbol}", logging.ERROR)
            return pd.DataFrame()

        # Get the current timestamp from the state manager
        current_time = self.state_manager.current_time

        if current_time is None:
            # Initialize with the first timestamp
            current_time = self.features_cache[symbol].index[0]

        # Get the index position of the current time
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.Timestamp(current_time)

        # Find the index of the current timestamp
        try:
            current_idx = self.features_cache[symbol].index.get_indexer([current_time], method='ffill')[0]
        except:
            self._log(f"Could not find index for {current_time}", logging.ERROR)
            return pd.DataFrame()

        # Calculate the end index for the batch
        end_idx = min(current_idx + batch_size, len(self.features_cache[symbol]) - 1)

        # Extract the batch
        batch_features = self.features_cache[symbol].iloc[current_idx:end_idx + 1]

        if batch_features.empty:
            self._log("No features in batch", logging.WARNING)
            return pd.DataFrame()

        # Update state to the last timestamp in the batch
        last_timestamp = batch_features.index[-1]
        self.state_manager.update_from_features(batch_features, last_timestamp)

        # Notify state update callbacks
        for callback in self.state_update_callbacks:
            callback(self.state_manager.get_state_dict())

        return batch_features

    def step_to_time(self, timestamp: datetime, symbol: str = None) -> Dict[str, Any]:
        """
        Move the state to a specific timestamp.

        Args:
            timestamp: Target timestamp
            symbol: Symbol to process (defaults to current_symbol)

        Returns:
            Current state dictionary
        """
        if self.current_mode not in ('training', 'backtesting'):
            self._log("step_to_time should only be called in training or backtesting mode", logging.WARNING)
            return {}

        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified", logging.ERROR)
            return {}

        if symbol not in self.features_cache or self.features_cache[symbol].empty:
            self._log(f"No features cached for {symbol}", logging.ERROR)
            return {}

        # Update state to the specified timestamp
        self.state_manager.update_from_features(self.features_cache[symbol], timestamp)

        # Get the current state
        state_dict = self.state_manager.get_state_dict()

        # Notify callbacks
        for callback in self.state_update_callbacks:
            callback(state_dict)

        return state_dict

    def update_live_data(self, symbol: str = None) -> Dict[str, Any]:
        """
        Update data and state based on latest live data.

        Args:
            symbol: Symbol to update (defaults to current_symbol)

        Returns:
            Current state dictionary
        """
        if self.current_mode != 'live':
            self._log("update_live_data should only be called in live mode", logging.WARNING)
            return {}

        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified", logging.ERROR)
            return {}

        # Get latest data from data manager
        latest_data = self.data_manager.get_data(symbol)

        if not latest_data:
            self._log(f"No data available for {symbol}", logging.WARNING)
            return {}

        # Extract features
        if symbol in self.features_cache and not self.features_cache[symbol].empty:
            # Update existing features
            self.features_cache[symbol] = self.feature_extractor.update_features(
                latest_data, self.features_cache[symbol])
        else:
            # Extract features from scratch
            self.features_cache[symbol] = self.feature_extractor.extract_features(latest_data)

        if symbol not in self.features_cache or self.features_cache[symbol].empty:
            self._log(f"No features extracted for {symbol}", logging.WARNING)
            return {}

        # Notify feature update callbacks
        for callback in self.feature_update_callbacks:
            callback(symbol, self.features_cache[symbol])

        # Get the latest timestamp
        latest_timestamp = self.features_cache[symbol].index[-1]

        # Update state manager
        self.state_manager.update_from_features(self.features_cache[symbol], latest_timestamp)

        # Get the current state
        state_dict = self.state_manager.get_state_dict()

        # Notify state update callbacks
        for callback in self.state_update_callbacks:
            callback(state_dict)

        return state_dict

    def execute_action(self, action: float, price: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Execute a trading action.

        Args:
            action: Desired position (0.0 to 1.0)
            price: Execution price
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Updated state dictionary
        """
        # Update position in state manager
        self.state_manager.update_position(action, price, timestamp)

        # Get updated state
        state_dict = self.state_manager.get_state_dict()

        # Notify state update callbacks
        for callback in self.state_update_callbacks:
            callback(state_dict)

        # Check if a trade was completed
        trade_history = self.state_manager.get_trade_history()
        if trade_history and (not hasattr(self, '_last_trade_count') or len(trade_history) > self._last_trade_count):
            # New trade completed
            self._last_trade_count = len(trade_history)
            latest_trade = trade_history[-1]

            # Notify trade callbacks
            for callback in self.trade_callbacks:
                callback(latest_trade)

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

    def get_trade_statistics(self) -> Dict[str, float]:
        """
        Get statistics about completed trades.

        Returns:
            Dictionary with trade statistics
        """
        return self.state_manager.get_trade_statistics()

    def get_trade_history(self) -> List[Dict]:
        """
        Get history of completed trades.

        Returns:
            List of trade dictionaries
        """
        return self.state_manager.get_trade_history()

    def reset(self):
        """Reset the processor state."""
        self.state_manager.reset()
        self.current_mode = 'idle'

        # Don't clear features cache to allow reuse

        # Reset trade tracking
        self._last_trade_count = 0