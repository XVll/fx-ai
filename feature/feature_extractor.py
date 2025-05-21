# feature/feature_extractor.py
import logging
import numpy as np
from typing import Dict, List

from config.config import ModelConfig


class FeatureExtractor:
    def __init__(self, symbol: str, market_simulator, config: ModelConfig, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        self.symbol = symbol
        self.market_simulator = market_simulator
        self.config = config

        # Get dimensions from config
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.static_feat_dim = config.static_feat_dim

        self.hf_feature_names: List[str] = [
            "HF_1s_Price_Velocity",
            "HF_1s_Price_Acceleration",
            "HF_3s_Volume_Velocity",
            "HF_3s_Volume_Acceleration",
            "HF_10s_Volume_Velocity",
            "HF_10s_Volume_Acceleration",
        ]

    def reset(self):
        self.logger.info("FeatureExtractor reset")

    # In feature/feature_extractor.py

    def extract_features(self) -> Dict[str, np.ndarray]:
        current_state = self.market_simulator.get_current_market_state()
        if current_state is None:
            self.logger.warning("No current market state available for feature extraction")

        # Extract features - keeping non-batched dimensions as per design
        hf_features = self._extract_high_frequency_features(current_state)
        mf_features = self._extract_medium_frequency_features(current_state)
        lf_features = self._extract_low_frequency_features(current_state)
        static_features = self._extract_static_features(current_state)


        # Validate the shapes to ensure they match expected dimensions
        assert hf_features.shape == (self.hf_seq_len, self.hf_feat_dim), \
            f"HF features have incorrect shape: {hf_features.shape}, expected ({self.hf_seq_len}, {self.hf_feat_dim})"
        assert mf_features.shape == (self.mf_seq_len, self.mf_feat_dim), \
            f"MF features have incorrect shape: {mf_features.shape}, expected ({self.mf_seq_len}, {self.mf_feat_dim})"
        assert lf_features.shape == (self.lf_seq_len, self.lf_feat_dim), \
            f"LF features have incorrect shape: {lf_features.shape}, expected ({self.lf_seq_len}, {self.lf_feat_dim})"
        assert static_features.shape == (self.static_feat_dim,), \
            f"Static features have incorrect shape: {static_features.shape}, expected ({self.static_feat_dim},)"

        return {
            'hf': hf_features,
            'mf': mf_features,
            'lf': lf_features,
            'static': static_features,
        }

    def _extract_high_frequency_features(self, current_state) -> np.ndarray:
        """
        Extract high-frequency features from market data with a focus on price and volume dynamics.
        Features include:
        1. HF_1s_Price_Velocity - Rate of change of price over 1-second intervals
        2. HF_1s_Price_Acceleration - Change in price velocity over 1-second intervals
        3. HF_3s_Volume_Velocity - Rate of change of volume over 3-second intervals
        4. HF_3s_Volume_Acceleration - Change in volume velocity over 3-second intervals
        5. HF_10s_Volume_Velocity - Rate of change of volume over 10-second intervals
        6. HF_10s_Volume_Acceleration - Change in volume velocity over 10-second intervals
        """
        # Initialize features array with the right shape
        features = np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)

        if current_state is None or 'hf_data_window' not in current_state:
            self.logger.warning("No high-frequency data available in market state")
            return features

        # Extract the 1-second bar data from the current state
        hf_data = current_state['hf_data_window']

        # Get prices and volumes from the hf_data window
        # Note: actual structure will depend on your market_simulator's data format
        prices = []
        volumes = []

        for entry in hf_data:
            # Extract price and volume from 1s bars if available
            bar = entry.get('1s_bar', None)
            if bar:
                prices.append(bar.get('close', None))
                volumes.append(bar.get('volume', 0.0))
            else:
                # Use current_price as fallback
                prices.append(entry.get('current_price', None))
                volumes.append(0.0)  # No volume data for this timepoint

        # Filter out None values and replace with previous valid values
        for i in range(1, len(prices)):
            if prices[i] is None:
                prices[i] = prices[i-1]

        # Convert to numpy arrays for calculations
        prices = np.array(prices, dtype=np.float32)
        volumes = np.array(volumes, dtype=np.float32)

        # Calculate features for each position in the sequence
        for i in range(self.hf_seq_len):
            feat_idx = 0  # Feature dimension index

            # Get the appropriate data slice for this position
            # For the latest position, we want the most recent data
            end_idx = len(prices) - (self.hf_seq_len - i)

            # Ensure we have enough data
            if end_idx <= 0:
                # Not enough data, use zeros
                features[i, :] = 0.0
                continue

            # --- Price Velocity (1s) ---
            if end_idx > 1:
                price_diffs_1s = np.diff(prices[max(0, end_idx-2):end_idx])
                price_velocity_1s = np.mean(price_diffs_1s) if len(price_diffs_1s) > 0 else 0.0
                features[i, feat_idx] = price_velocity_1s
            feat_idx += 1

            # --- Price Acceleration (1s) ---
            if end_idx > 2:
                # Calculate velocities for two adjacent periods
                price_diffs_1s_period1 = prices[end_idx-1] - prices[end_idx-2] if end_idx-2 >= 0 else 0.0
                price_diffs_1s_period2 = prices[end_idx] - prices[end_idx-1] if end_idx-1 >= 0 else 0.0
                # Acceleration is change in velocity
                price_acceleration_1s = price_diffs_1s_period2 - price_diffs_1s_period1
                features[i, feat_idx] = price_acceleration_1s
            feat_idx += 1

            # --- Volume Velocity (3s) ---
            if end_idx > 3:
                volume_3s = np.sum(volumes[max(0, end_idx-3):end_idx])
                volume_prev_3s = np.sum(volumes[max(0, end_idx-6):max(0, end_idx-3)])
                volume_velocity_3s = (volume_3s - volume_prev_3s) / 3.0 if volume_prev_3s > 0 else 0.0
                features[i, feat_idx] = volume_velocity_3s
            feat_idx += 1

            # --- Volume Acceleration (3s) ---
            if end_idx > 6:
                # Current velocity
                volume_3s_current = np.sum(volumes[max(0, end_idx-3):end_idx])
                volume_3s_prev1 = np.sum(volumes[max(0, end_idx-6):max(0, end_idx-3)])
                volume_velocity_current = (volume_3s_current - volume_3s_prev1) / 3.0

                # Previous velocity
                volume_3s_prev2 = np.sum(volumes[max(0, end_idx-9):max(0, end_idx-6)])
                volume_velocity_prev = (volume_3s_prev1 - volume_3s_prev2) / 3.0

                # Acceleration
                volume_acceleration_3s = (volume_velocity_current - volume_velocity_prev) / 3.0
                features[i, feat_idx] = volume_acceleration_3s
            feat_idx += 1

            # --- Volume Velocity (10s) ---
            if end_idx > 10:
                volume_10s = np.sum(volumes[max(0, end_idx-10):end_idx])
                volume_prev_10s = np.sum(volumes[max(0, end_idx-20):max(0, end_idx-10)])
                volume_velocity_10s = (volume_10s - volume_prev_10s) / 10.0 if volume_prev_10s > 0 else 0.0
                features[i, feat_idx] = volume_velocity_10s
            feat_idx += 1

            # --- Volume Acceleration (10s) ---
            if end_idx > 20:
                # Current velocity
                volume_10s_current = np.sum(volumes[max(0, end_idx-10):end_idx])
                volume_10s_prev1 = np.sum(volumes[max(0, end_idx-20):max(0, end_idx-10)])
                volume_velocity_current = (volume_10s_current - volume_10s_prev1) / 10.0

                # Previous velocity
                volume_10s_prev2 = np.sum(volumes[max(0, end_idx-30):max(0, end_idx-20)])
                volume_velocity_prev = (volume_10s_prev1 - volume_10s_prev2) / 10.0

                # Acceleration
                volume_acceleration_10s = (volume_velocity_current - volume_velocity_prev) / 10.0
                features[i, feat_idx] = volume_acceleration_10s
            feat_idx += 1

        # Normalize features to improve training stability
        # This is a simple min-max normalization - you might want to use other methods
        for j in range(self.hf_feat_dim):
            feature_col = features[:, j]
            if np.any(feature_col != 0):  # Only normalize non-zero columns
                # Get column min and max, avoiding division by zero
                col_min = np.min(feature_col)
                col_max = np.max(feature_col)
                if col_max > col_min:
                    features[:, j] = (feature_col - col_min) / (col_max - col_min)

        return features

    def _extract_medium_frequency_features(self, current_state) -> np.ndarray:
        """Generate reliable medium-frequency features with non-zero values."""
        features = np.ones((self.mf_seq_len, self.mf_feat_dim)) * 0.2
        for i in range(self.mf_seq_len):
            for j in range(self.mf_feat_dim):
                features[i, j] += (i * 0.01 + j * 0.001)
        return features

    def _extract_low_frequency_features(self, current_state) -> np.ndarray:
        """Generate reliable low-frequency features with non-zero values."""
        features = np.ones((self.lf_seq_len, self.lf_feat_dim)) * 0.3
        for i in range(self.lf_seq_len):
            for j in range(self.lf_feat_dim):
                features[i, j] += (i * 0.01 + j * 0.001)
        return features

    def _extract_static_features(self, current_state) -> np.ndarray:
        """Generate reliable static features with non-zero values."""
        # Important: shape must match observation space (static_feat_dim,)
        features = np.ones(self.static_feat_dim) * 0.4
        for i in range(self.static_feat_dim):
            features[i] += (i * 0.01)
        return features
