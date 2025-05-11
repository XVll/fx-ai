# data/feature/state_manager.py
from typing import Dict,  Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .feature_extractor import FeatureExtractor

class StateManager:
    """
    Manages the state for RL agent, combining features from multiple time frames
    and maintaining a coherent representation of the current trading state.
    """

    def __init__(self, feature_extractor: FeatureExtractor, lookback_periods: Dict[str, int] = None):
        """
        Initialize the state manager.

        Args:
            feature_extractor: FeatureExtractor instance to use
            lookback_periods: Dict mapping timeframes to number of periods to include in state
                Example: {'1s': 30, '1m': 10, '5m': 5, '1d': 1}
        """
        self.feature_extractor = feature_extractor
        self.lookback_periods = lookback_periods or {'1s': 30, '1m': 10, '5m': 5, '1d': 1}

        # State storage
        self.current_time = None
        self.current_features = None
        self.feature_history = {}
        self.current_state = None

        # Position tracking for RL state
        self.current_position = 0.0  # 0.0 = flat, 1.0 = fully long
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0
        self.max_unrealized_pnl = 0.0
        self.position_start_time = None

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

        # Filter features to up to the specified timestamp
        features_df = features_df[features_df.index <= timestamp]

        if features_df.empty:
            return

        # Update current time and features
        self.current_time = timestamp
        self.current_features = features_df.loc[timestamp]

        # Update feature history
        for col in features_df.columns:
            if col not in self.feature_history:
                self.feature_history[col] = []

            # Add the latest value
            self.feature_history[col].append(features_df.loc[timestamp, col])

            # Keep only the needed lookback periods
            # This assumes columns are named with timeframe prefixes
            for tf, periods in self.lookback_periods.items():
                if col.startswith(tf):
                    self.feature_history[col] = self.feature_history[col][-periods:]
                    break

        # Update the current state vector
        self._update_state_vector()

    def update_from_data(self, data_dict: Dict[str, pd.DataFrame], timestamp: datetime = None):
        """
        Update the state from raw data by extracting features first.

        Args:
            data_dict: Dictionary mapping data types to DataFrames
            timestamp: Optional timestamp to update to (if None, use latest in data)
        """
        # Extract features from the data
        features_df = self.feature_extractor.extract_features(data_dict)

        # Update state from the features
        self.update_from_features(features_df, timestamp)

    def update_position(self, new_position: float, price: float, timestamp: datetime = None):
        """
        Update the current position information.

        Args:
            new_position: New position size (0.0 = flat, 1.0 = fully long)
            price: Current price for P&L calculation
            timestamp: Optional timestamp (defaults to current_time)
        """
        # Update timestamp if provided
        if timestamp is not None:
            self.current_time = timestamp

        # If position was flat and now entering a position
        if self.current_position == 0.0 and new_position != 0.0:
            self.entry_price = price
            self.position_start_time = self.current_time
            self.max_unrealized_pnl = 0.0

        # If closing out a position
        elif self.current_position != 0.0 and new_position == 0.0:
            self.entry_price = 0.0
            self.position_start_time = None
            self.max_unrealized_pnl = 0.0

        # Update position and price
        self.current_position = new_position
        self.last_price = price

        # Calculate unrealized P&L if in a position
        if self.current_position != 0.0 and self.entry_price != 0.0:
            self.unrealized_pnl = (self.last_price - self.entry_price) / self.entry_price * 100.0
            # Track max unrealized P&L for the current position
            self.max_unrealized_pnl = max(self.max_unrealized_pnl, self.unrealized_pnl)
        else:
            self.unrealized_pnl = 0.0

        # Update the state vector to reflect new position
        self._update_state_vector()

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

        if self.position_start_time is not None:
            state_dict['position_duration'] = (self.current_time - self.position_start_time).total_seconds()
        else:
            state_dict['position_duration'] = 0.0

        # Then add time series features with their history
        for feature, history in self.feature_history.items():
            # Add the feature history as a list
            state_dict[feature] = history

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

        if self.position_start_time is not None:
            state_array.append((self.current_time - self.position_start_time).total_seconds())
        else:
            state_array.append(0.0)

        # Add time series features
        for tf, periods in self.lookback_periods.items():
            for feature, history in self.feature_history.items():
                if feature.startswith(tf):
                    # Pad history if needed to ensure right length
                    padded_history = history[-periods:]
                    if len(padded_history) < periods:
                        padded_history = [0.0] * (periods - len(padded_history)) + padded_history

                    state_array.extend(padded_history)

        return np.array(state_array)

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the current state as a dictionary.

        Returns:
            Dictionary with state components
        """
        return self.current_state if self.current_state is not None else {}

    def reset(self):
        """Reset the state manager to initial conditions."""
        self.current_time = None
        self.current_features = None
        self.feature_history = {}
        self.current_state = None

        self.current_position = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.unrealized_pnl = 0.0
        self.max_unrealized_pnl = 0.0
        self.position_start_time = None