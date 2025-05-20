# feature/feature_extractor.py
import logging
import numpy as np
from typing import Dict

from config.config import FeatureConfig


class FeatureExtractor:
    def __init__(self, symbol: str, market_simulator, config: FeatureConfig, logger=None):
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

    def extract_features(self) -> Dict[str, np.ndarray]:
        current_state = self.market_simulator.get_current_market_state()
        if current_state is None:
            self.logger.warning("No current market state available for feature extraction")

        hf_features = self._extract_high_frequency_features(current_state)
        mf_features = self._extract_medium_frequency_features(current_state)
        lf_features = self._extract_low_frequency_features(current_state)
        static_features = self._extract_static_features(current_state)

        return {
            'hf': hf_features,
            'mf': mf_features,
            'lf': lf_features,
            'static': static_features
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
        features = np.zeros((1, self.static_feat_dim))
        return features
