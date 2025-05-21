# feature/feature_extractor.py
import logging
import numpy as np
from typing import Dict

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

    def reset(self):
        self.logger.info("FeatureExtractor reset")

    # In feature/feature_extractor.py

    def extract_features(self) -> Dict[str, np.ndarray]:
        """
        Extract features from the current market state.

        Returns:
            Dictionary with properly shaped feature arrays:
                - hf: High frequency features shape (hf_seq_len, hf_feat_dim)
                - mf: Medium frequency features shape (mf_seq_len, mf_feat_dim)
                - lf: Low frequency features shape (lf_seq_len, lf_feat_dim)
                - portfolio: Portfolio features shape (portfolio_seq_len, portfolio_feat_dim)
                - static: Static features shape (static_feat_dim,)
        """
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
            'hf': self._ensure_shape(hf_features,(self.hf_seq_len, self.hf_feat_dim), 'hf'),
            'mf': self._ensure_shape(mf_features,(self.mf_seq_len, self.mf_feat_dim), 'mf'),
            'lf': self._ensure_shape(lf_features,(self.lf_seq_len, self.lf_feat_dim), 'lf'),
            'static': self._ensure_shape(static_features,(self.static_feat_dim,), 'static'),
        }

    def _extract_high_frequency_features(self, current_state) -> np.ndarray:
        features = np.zeros((self.hf_seq_len, self.hf_feat_dim))
        return features

    def _extract_medium_frequency_features(self, current_state) -> np.ndarray:
        features = np.zeros((self.mf_seq_len, self.mf_feat_dim))
        return features

    def _extract_low_frequency_features(self, current_state) -> np.ndarray:
        features = np.zeros((self.lf_seq_len, self.lf_feat_dim))
        return features

    def _extract_static_features(self, current_state) -> np.ndarray:
        # Extract static features from the current state
        features = np.zeros(self.static_feat_dim,)
        return features

    def _ensure_shape(self, arr: np.ndarray, expected_shape: tuple, name: str) -> np.ndarray:
        """Ensure array has the expected shape, fix if needed."""
        if arr.shape != expected_shape:
            self.logger.warning(f"{name} features have shape {arr.shape}, expected {expected_shape}. Reshaping.")
            if len(expected_shape) == 1:  # 1D array
                # Resize, pad, or truncate to match expected size
                result = np.zeros(expected_shape)
                copy_size = min(arr.size, expected_shape[0])
                result[:copy_size] = arr[:copy_size]
                return result
            else:  # 2D array
                result = np.zeros(expected_shape)
                # Copy as much as possible
                copy_rows = min(arr.shape[0], expected_shape[0])
                copy_cols = min(arr.shape[1] if arr.ndim > 1 else 1, expected_shape[1])
                if arr.ndim == 1:
                    for i in range(min(arr.size, copy_rows)):
                        result[i, 0] = arr[i]
                else:
                    result[:copy_rows, :copy_cols] = arr[:copy_rows, :copy_cols]
                return result
        return arr
