# data/data_processor.py
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data.feature.feature_extractor import FeatureExtractor
from data.feature.state_manager import StateManager
from data.provider.data_provider import DataProvider, LiveDataProvider, HistoricalDataProvider


class DataProcessor:
    """
    Main entry point for data processing in the AI trading system.
    Handles data acquisition, feature extraction, and state management.
    Works with both historical and live data providers.
    """

    def __init__(self, provider: DataProvider,
                 feature_config: Dict = None,
                 lookback_periods: Dict[str, int] = None,
                 logger: logging.Logger = None):
        """
        Initialize the data processor.

        Args:
            provider: DataProvider instance (historical or live)
            feature_config: Configuration for FeatureExtractor
            lookback_periods: Dict mapping timeframes to lookback periods
            logger: Optional logger
        """
        self.provider = provider
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.feature_extractor = FeatureExtractor(feature_config)
        self.state_manager = StateManager(self.feature_extractor, lookback_periods)

        # Data caches
        self.data_cache = {}
        self.features_cache = {}

        # Live data tracking
        self.is_live = isinstance(provider, LiveDataProvider)
        self.current_symbol = None
        self.subscribed = False

        # Callbacks
        self.state_update_callbacks = []

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def load_historical_data(self, symbol: str, start_time: Union[datetime, str],
                             end_time: Union[datetime, str], timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for a symbol and precompute features.

        Args:
            symbol: Symbol to load data for
            start_time: Start time for data
            end_time: End time for data
            timeframes: List of timeframes to load (e.g., ["1s", "1m", "5m", "1d"])
                        If None, loads all available timeframes

        Returns:
            Dictionary mapping data types to DataFrames
        """
        if not isinstance(self.provider, HistoricalDataProvider):
            raise TypeError("Provider must be a HistoricalDataProvider for load_historical_data")

        self.current_symbol = symbol

        # Default timeframes if not specified
        if timeframes is None:
            timeframes = ["1s", "1m", "5m", "1d"]

        # Clear cache for this symbol
        self.data_cache = {}
        self.features_cache = {}

        # Load data for each type
        try:
            # Load trades
            self._log(f"Loading trades data for {symbol} from {start_time} to {end_time}")
            self.data_cache['trades'] = self.provider.get_trades(symbol, start_time, end_time)

            # Load quotes
            self._log(f"Loading quotes data for {symbol} from {start_time} to {end_time}")
            self.data_cache['quotes'] = self.provider.get_quotes(symbol, start_time, end_time)

            # Load bars for each timeframe
            for tf in timeframes:
                key = f'bars_{tf}'
                self._log(f"Loading {tf} bars for {symbol} from {start_time} to {end_time}")
                self.data_cache[key] = self.provider.get_bars(symbol, tf, start_time, end_time)

            # Load status updates
            self._log(f"Loading status data for {symbol} from {start_time} to {end_time}")
            self.data_cache['status'] = self.provider.get_status(symbol, start_time, end_time)

            # Extract features
            self._log(f"Extracting features for {symbol}")
            self.features_cache = self.feature_extractor.extract_features(self.data_cache)

            return self.data_cache

        except Exception as e:
            self._log(f"Error loading historical data: {e}", logging.ERROR)
            raise

    def get_features(self) -> pd.DataFrame:
        """Get the cached features."""
        return self.features_cache

    def initialize_live_data(self, symbol: str, timeframes: List[str] = None):
        """
        Initialize live data streaming for a symbol.

        Args:
            symbol: Symbol to stream data for
            timeframes: List of timeframes to subscribe to
        """
        if not isinstance(self.provider, LiveDataProvider):
            raise TypeError("Provider must be a LiveDataProvider for initialize_live_data")

        self.current_symbol = symbol

        # Default timeframes if not specified
        if timeframes is None:
            timeframes = ["1s", "1m", "5m", "1d"]

        # Map timeframes to data types for subscription
        data_types = ["trades", "quotes", "status"]
        for tf in timeframes:
            data_types.append(f"bars_{tf}")

        # Register callbacks
        self.provider.add_trade_callback(self._on_trade_update)
        self.provider.add_quote_callback(self._on_quote_update)
        self.provider.add_bar_callback(self._on_bar_update)
        self.provider.add_status_callback(self._on_status_update)

        # Subscribe to the data
        self.provider.subscribe([symbol], data_types)
        self.subscribed = True

        self._log(f"Initialized live data for {symbol} with timeframes {timeframes}")

    def _on_trade_update(self, trade_data: Dict):
        """Callback for trade updates."""
        if not self.subscribed:
            return

        symbol = trade_data['symbol']
        if symbol != self.current_symbol:
            return

        # Update state
        self._update_live_state('trade', trade_data)

    def _on_quote_update(self, quote_data: Dict):
        """Callback for quote updates."""
        if not self.subscribed:
            return

        symbol = quote_data['symbol']
        if symbol != self.current_symbol:
            return

        # Update state
        self._update_live_state('quote', quote_data)

    def _on_bar_update(self, bar_data: Dict):
        """Callback for bar updates."""
        if not self.subscribed:
            return

        symbol = bar_data['symbol']
        if symbol != self.current_symbol:
            return

        # Update state
        self._update_live_state('bar', bar_data)

    def _on_status_update(self, status_data: Dict):
        """Callback for status updates."""
        if not self.subscribed:
            return

        symbol = status_data['symbol']
        if symbol != self.current_symbol:
            return

        # Update state
        self._update_live_state('status', status_data)

    def _update_live_state(self, update_type: str, data: Dict):
        """Update the live state based on new data."""
        # This is a simplified version - in reality, you would accumulate the data
        # and periodically recompute features and state

        timestamp = data.get('timestamp', datetime.now())

        # For trade updates, we can update last price and unrealized P&L
        if update_type == 'trade':
            price = data.get('price', 0.0)
            self.state_manager.update_position(
                self.state_manager.current_position,  # Keep same position
                price,
                timestamp
            )

        # Every few updates (or on each tick), recompute features
        # This is simplified - you would track state more efficiently in practice

        # Notify callbacks
        for callback in self.state_update_callbacks:
            callback(self.state_manager.get_state_dict())

    def get_historical_state_at_time(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get the state at a specific timestamp from historical data.

        Args:
            timestamp: The timestamp to get state for

        Returns:
            State dictionary
        """
        if not self.features_cache.empty:
            # Update state manager to this timestamp
            self.state_manager.update_from_features(self.features_cache, timestamp)
            return self.state_manager.get_state_dict()
        else:
            self._log("No features cached. Call load_historical_data first.", logging.WARNING)
            return {}

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state."""
        return self.state_manager.get_state_dict()

    def get_current_state_array(self) -> np.ndarray:
        """Get the current state as a NumPy array."""
        return self.state_manager.get_state_array()

    def update_position(self, position: float, price: float):
        """
        Update the current position information.

        Args:
            position: New position size (0.0 to 1.0)
            price: Current price
        """
        self.state_manager.update_position(position, price)

    def add_state_update_callback(self, callback_fn: Callable):
        """
        Add a callback for state updates.

        Args:
            callback_fn: Function to call with new state
        """
        self.state_update_callbacks.append(callback_fn)

    def close(self):
        """Clean up resources."""
        if self.is_live and self.subscribed:
            if isinstance(self.provider, LiveDataProvider):
                self.provider.unsubscribe([self.current_symbol], [])
                self.provider.close()
            self.subscribed = False