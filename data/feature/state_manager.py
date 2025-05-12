# data/feature/state_manager.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging


class StateManager:
    """
    Manages the state for RL agent, combining features from multiple time frames
    and maintaining a coherent representation of the current trading state.
    Designed with:
    - Decoupled feature handling
    - Efficient state representation
    - Support for historical or live updates
    - Clean training iteration
    """

    def __init__(self, feature_config: Dict = None,
                 lookback_periods: Dict[str, int] = None,
                 logger: logging.Logger = None):
        """
        Initialize the state manager.

        Args:
            feature_config: Configuration for feature handling
            lookback_periods: Dict mapping timeframes to number of periods to include in state
                Example: {'1s': 30, '1m': 10, '5m': 5, '1d': 1}
            logger: Optional logger
        """
        self.feature_config = feature_config or {}
        self.lookback_periods = lookback_periods or {'1s': 30, '1m': 10, '5m': 5, '1d': 1}
        self.logger = logger or logging.getLogger(__name__)

        # State storage
        self.current_time = None
        self.current_features = None
        self.feature_history = {}
        self.current_state = None
        self.state_history = {}  # Timestamped state history for backtesting/training

        # Position tracking for RL state
        self.current_position = 0.0  # 0.0 = flat, 1.0 = fully long
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0
        self.max_unrealized_pnl = 0.0
        self.position_start_time = None
        self.position_duration = 0.0

        # Trade tracking
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
            self._log("Cannot update state: features DataFrame is empty", logging.WARNING)
            return

        # If timestamp not provided, use the latest in the features DataFrame
        if timestamp is None:
            timestamp = features_df.index[-1]

        # Filter features to up to the specified timestamp
        available_features = features_df[features_df.index <= timestamp]

        if available_features.empty:
            self._log(f"No features available at or before {timestamp}", logging.WARNING)
            return

        # Get the exact timestamp or closest prior timestamp
        closest_timestamp = available_features.index[-1]

        # Update current time and features
        self.current_time = closest_timestamp
        self.current_features = available_features.loc[closest_timestamp]

        # Update feature history
        self._update_feature_history(features_df, closest_timestamp)

        # Update the current state vector
        self._update_state_vector()

        # Store this state in the state history
        self.state_history[closest_timestamp] = self.current_state.copy()

    def _update_feature_history(self, features_df: pd.DataFrame, timestamp: datetime):
        """
        Update the feature history for a given timestamp.

        Args:
            features_df: DataFrame with features
            timestamp: Current timestamp to extract history for
        """
        # Filter features up to and including the timestamp
        available_features = features_df[features_df.index <= timestamp]

        if available_features.empty:
            return

        # Update feature history for each timeframe
        for timeframe, periods in self.lookback_periods.items():
            # Get features for this timeframe
            tf_features = {col: available_features[col]
                           for col in available_features.columns
                           if col.startswith(timeframe)}

            if not tf_features:
                continue

            # For each feature in this timeframe, get the history
            for feature_name, series in tf_features.items():
                # Get the history for this feature, up to the specified number of periods
                history = series.iloc[-periods:].values

                # Store the history
                self.feature_history[feature_name] = history

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
            self._log(f"Invalid position size: {new_position}. Must be between 0.0 and 1.0", logging.WARNING)
            new_position = max(0.0, min(1.0, new_position))

        if price <= 0.0:
            self._log(f"Invalid price: {price}. Must be positive", logging.WARNING)
            return

        # Update timestamp if provided
        if timestamp is not None:
            self.current_time = timestamp

        # Detect trade completion (position flattened)
        if self.current_position > 0.0 and new_position == 0.0:
            # Calculate realized P&L
            realized_pnl = (price - self.entry_price) / self.entry_price * 100.0

            # Update trade statistics
            self.trade_count += 1
            if realized_pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            self.total_pnl += realized_pnl

            # Record trade
            trade = {
                'entry_time': self.position_start_time,
                'exit_time': self.current_time,
                'duration': (
                            self.current_time - self.position_start_time).total_seconds() if self.position_start_time else 0,
                'entry_price': self.entry_price,
                'exit_price': price,
                'position_size': self.current_position,
                'realized_pnl': realized_pnl,
                'max_unrealized_pnl': self.max_unrealized_pnl
            }

            self.trade_history.append(trade)

            # Reset position tracking
            self.entry_price = 0.0
            self.position_start_time = None
            self.max_unrealized_pnl = 0.0

        # Detect new position or position increase
        elif self.current_position == 0.0 and new_position > 0.0:
            # Starting a new position
            self.entry_price = price
            self.position_start_time = self.current_time
            self.max_unrealized_pnl = 0.0

        elif self.current_position > 0.0 and new_position > self.current_position:
            # Increasing position - recalculate entry price as weighted average
            new_shares = new_position - self.current_position
            total_shares = new_position

            # Weighted average of entry prices
            self.entry_price = (self.entry_price * self.current_position + price * new_shares) / total_shares

        # Update position and price
        self.current_position = new_position
        self.last_price = price

        # Calculate unrealized P&L if in a position
        if self.current_position > 0.0 and self.entry_price > 0.0:
            self.unrealized_pnl = (self.last_price - self.entry_price) / self.entry_price * 100.0

            # Track max unrealized P&L for the current position
            self.max_unrealized_pnl = max(self.max_unrealized_pnl, self.unrealized_pnl)

            # Update position duration
            if self.position_start_time:
                self.position_duration = (self.current_time - self.position_start_time).total_seconds()
        else:
            self.unrealized_pnl = 0.0
            self.position_duration = 0.0

        # Update the state vector to reflect new position
        self._update_state_vector()

        # Store the updated state
        if self.current_time:
            self.state_history[self.current_time] = self.current_state.copy()

    def _update_state_vector(self):
        """Update the current state vector for the RL agent."""
        if self.current_features is None:
            return

        # Extract lookback window for each feature
        state_dict = {}

        # First add single scalar features (like current position)
        state_dict['current_position'] = self.current_position
        state_dict['unrealized_pnl'] = self.unrealized_pnl
        state_dict['max_unrealized_pnl'] = self.max_unrealized_pnl
        state_dict['position_duration'] = self.position_duration

        # Add trade statistics
        state_dict['trade_count'] = self.trade_count
        state_dict['win_rate'] = self.win_count / max(1, self.trade_count)
        state_dict['total_pnl'] = self.total_pnl

        # Then add time series features with their history
        for feature, history in self.feature_history.items():
            # Add the feature history as a list
            state_dict[feature] = history.tolist() if isinstance(history, np.ndarray) else history

        self.current_state = state_dict

    def get_state_array(self) -> np.ndarray:
        """
        Get the current state as a flat array for RL models.

        Returns:
            NumPy array with the flattened state representation
        """
        if self.current_state is None:
            return np.array([])

        # Flatten the state dictionary into an array
        state_array = []

        # Add scalar features
        state_array.append(self.current_position)
        state_array.append(self.unrealized_pnl)
        state_array.append(self.max_unrealized_pnl)
        state_array.append(self.position_duration)
        state_array.append(self.trade_count)
        state_array.append(self.win_count / max(1, self.trade_count))  # Win rate
        state_array.append(self.total_pnl)

        # Add time series features from feature history
        for tf, periods in self.lookback_periods.items():
            for feature, history in self.feature_history.items():
                if feature.startswith(tf):
                    # Convert history to numpy array if it's not already
                    if not isinstance(history, np.ndarray):
                        history = np.array(history)

                    # Pad history if needed to ensure right length
                    if len(history) < periods:
                        padding = np.zeros(periods - len(history))
                        padded_history = np.concatenate([padding, history])
                    else:
                        padded_history = history[-periods:]

                    # Add to state array
                    state_array.extend(padded_history)

        return np.array(state_array)

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
        self.feature_history = {}
        self.current_state = None
        self.state_history = {}

        self.current_position = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0
        self.max_unrealized_pnl = 0.0
        self.position_start_time = None
        self.position_duration = 0.0

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
            'avg_pnl': self.total_pnl / max(1, self.trade_count),
        }

        # Calculate additional statistics
        if self.trade_history:
            # Average win and loss
            win_trades = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] > 0]
            loss_trades = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] <= 0]

            stats['avg_win'] = sum(win_trades) / max(1, len(win_trades))
            stats['avg_loss'] = sum(loss_trades) / max(1, len(loss_trades))

            # Profit factor
            stats['profit_factor'] = abs(sum(win_trades) / max(0.001, abs(sum(loss_trades))))

            # Average trade duration
            durations = [t['duration'] for t in self.trade_history]
            stats['avg_duration'] = sum(durations) / max(1, len(durations))

        return stats