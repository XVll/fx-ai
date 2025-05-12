# SIMPLIFIED feature_extractor.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging


class FeatureExtractor:
    """
    Simplified feature extractor with minimal features.
    """

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Simplified parameters
        self.price_windows = [5, 10]  # Just 2 windows instead of many
        self.feature_groups = {
            'price': self._extract_price_features,
            'volume': self._extract_volume_features,
        }
        self.feature_cache = {}

    def _extract_price_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # Just 3 simple features
        # 1. Price change
        features[f"{timeframe}_price_change"] = bars_df['close'].pct_change()

        # 2. Simple moving average
        features[f"{timeframe}_sma_10"] = bars_df['close'].rolling(10).mean()

        # 3. Price relative to SMA
        features[f"{timeframe}_rel_sma"] = \
            bars_df['close'] / features[f"{timeframe}_sma_10"].replace(0, np.nan)

        return features

    def _extract_volume_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # Just 2 volume features
        # 1. Volume change
        features[f"{timeframe}_volume_change"] = bars_df['volume'].pct_change()

        # 2. Relative volume (to recent average)
        vol_avg = bars_df['volume'].rolling(10).mean()
        features[f"{timeframe}_rel_volume"] = bars_df['volume'] / vol_avg.replace(0, np.nan)

        return features

    def extract_features(self, data_dict: Dict[str, pd.DataFrame],
                         feature_groups: List[str] = None,
                         cache_key: str = None) -> pd.DataFrame:
        """Simplified feature extraction with minimal processing."""
        # Rest of method remains intact but with simplified processing
        all_features = {}
        feature_groups = feature_groups or list(self.feature_groups.keys())

        # Process bars for each timeframe - but with fewer timeframes
        for tf in ['1m', '5m']:  # Just 2 timeframes instead of 4
            key = f'bars_{tf}'
            if key in data_dict and not data_dict[key].empty:
                bars_df = data_dict[key]

                for group in feature_groups:
                    if group in self.feature_groups:
                        try:
                            group_features = self.feature_groups[group](bars_df, tf)
                            if not group_features.empty:
                                all_features[f'{tf}_{group}'] = group_features
                        except Exception as e:
                            self.logger.error(f"Error extracting {group} features: {str(e)}")

        # Basic feature combination
        if not all_features:
            return pd.DataFrame()

        # Find highest frequency index
        main_index = None
        for df in all_features.values():
            if df is not None and not df.empty:
                if main_index is None or len(df.index) > len(main_index):
                    main_index = df.index

        # Combine features
        combined = pd.DataFrame(index=main_index)
        for name, df in all_features.items():
            if df is not None and not df.empty:
                # Simple forward fill for missing values
                reindexed = df.reindex(main_index, method='ffill')
                combined = combined.join(reindexed)

        # Handle NaN values simply
        combined = combined.fillna(0)

        # Cache if requested
        if cache_key:
            self.feature_cache[cache_key] = combined

        return combined