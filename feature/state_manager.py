# feature/state_manager.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging


class StateManager:
    """
    Simplified state manager that maintains minimal state for RL agent
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

        # Position tracking - simplified
        self.current_position = 0.0  # 0.0 = flat, 1.0 = fully long
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0

        # Trade tracking - simplified
        self.trade_history = []
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

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