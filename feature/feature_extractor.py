# feature/feature_extractor.py (updated for Hydra)
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
import torch
from omegaconf import DictConfig


class FeatureExtractor:
    """
    Feature extractor that provides multi-timeframe features
    compatible with the Multi-Branch Transformer model.
    """

    def __init__(self, config: Union[Dict, DictConfig] = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Support for Hydra config - convert to dict if needed
        cfg = self.config
        if hasattr(cfg, "_to_dict"):
            cfg = cfg._to_dict()

        # Feature dimension configurations - use config values or defaults
        self.hf_seq_len = cfg.get('hf_seq_len', 60)
        self.hf_feat_dim = cfg.get('hf_feat_dim', 20)
        self.mf_seq_len = cfg.get('mf_seq_len', 30)
        self.mf_feat_dim = cfg.get('mf_feat_dim', 15)
        self.lf_seq_len = cfg.get('lf_seq_len', 30)
        self.lf_feat_dim = cfg.get('lf_feat_dim', 10)
        self.static_feat_dim = cfg.get('static_feat_dim', 15)

        # Feature extraction configuration
        self.use_volume_profile = cfg.get('use_volume_profile', True)
        self.use_tape_features = cfg.get('use_tape_features', True)
        self.use_level2_features = cfg.get('use_level2_features', True)
        self.feature_normalization = cfg.get('feature_normalization', 'standardize')
        self.moving_avg_window = cfg.get('moving_avg_window', 20)

        # Simplified parameters
        self.feature_groups = {
            'price': self._extract_price_features,
            'volume': self._extract_volume_features,
        }

        # Add optional feature groups based on config
        if self.use_tape_features:
            self.feature_groups['tape'] = self._extract_tape_features
        if self.use_level2_features:
            self.feature_groups['level2'] = self._extract_level2_features

        self.feature_cache = {}

    def _extract_price_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # Simple price features
        if 'close' in bars_df.columns:
            features[f"{timeframe}_price_change"] = bars_df['close'].pct_change()
            features[f"{timeframe}_sma_10"] = bars_df['close'].rolling(10).mean()
            features[f"{timeframe}_rel_sma"] = bars_df['close'] / features[f"{timeframe}_sma_10"].replace(0, np.nan)

            # Add some trend features
            features[f"{timeframe}_roc_5"] = bars_df['close'].pct_change(5)
            features[f"{timeframe}_roc_10"] = bars_df['close'].pct_change(10)

            # Simple volatility feature
            features[f"{timeframe}_volatility"] = bars_df['close'].pct_change().rolling(10).std()

        return features

    # Add a new method for tape features based on config
    def _extract_tape_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # Simple tape speed simulation (would use actual tape data in production)
        if 'volume' in bars_df.columns:
            # Calculate tape speed (trades per second proxy)
            features[f"{timeframe}_tape_speed"] = bars_df['volume'] / bars_df['volume'].rolling(5).mean()

            # Tape imbalance proxy
            if 'close' in bars_df.columns and 'open' in bars_df.columns:
                price_change = bars_df['close'] - bars_df['open']
                features[f"{timeframe}_tape_imbalance"] = np.sign(price_change) * bars_df['volume']

        return features

    # Add a new method for L2 features based on config
    def _extract_level2_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # This would use actual L2 data in production
        # Here we're just creating placeholder features
        if all(col in bars_df.columns for col in ['high', 'low', 'close']):
            # Approximate bid-ask spread
            features[f"{timeframe}_spread_proxy"] = (bars_df['high'] - bars_df['low']) / bars_df['close']

        return features

    # Fix for the volume features extraction method in feature_extractor.py

    def _extract_volume_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # Volume features
        if 'volume' in bars_df.columns:
            features[f"{timeframe}_volume_change"] = bars_df['volume'].pct_change()
            vol_avg = bars_df['volume'].rolling(10).mean()
            features[f"{timeframe}_rel_volume"] = bars_df['volume'] / vol_avg.replace(0, np.nan)

            # Volume momentum
            features[f"{timeframe}_vol_momentum"] = bars_df['volume'].diff(3)

            # Volume and price interaction - FIX CORRELATION ERROR
            if 'close' in bars_df.columns:
                # Create separate series for correlation to avoid the DataFrame/Series error
                close_pct = bars_df['close'].pct_change()
                vol_pct = bars_df['volume'].pct_change()

                # Only attempt correlation if we have enough non-NaN values
                if len(close_pct.dropna()) > 10 and len(vol_pct.dropna()) > 10:
                    # Compute rolling correlation correctly
                    rolling_corr = pd.Series(index=bars_df.index)
                    window_size = 10

                    for i in range(window_size, len(bars_df)):
                        window_close = close_pct.iloc[i - window_size:i]
                        window_vol = vol_pct.iloc[i - window_size:i]
                        if window_close.notna().all() and window_vol.notna().all():
                            rolling_corr.iloc[i] = window_close.corr(window_vol)

                    features[f"{timeframe}_volume_price_corr"] = rolling_corr
                else:
                    # Not enough data for correlation
                    features[f"{timeframe}_volume_price_corr"] = np.nan

        return features

    def extract_features(self, data_dict: Dict[str, pd.DataFrame],
                         feature_groups: List[str] = None,
                         cache_key: str = None) -> pd.DataFrame:
        """Extract features from data dictionary."""
        all_features = {}
        feature_groups = feature_groups or list(self.feature_groups.keys())

        # Process bars for different timeframes
        for tf in ['1s', '1m', '5m']:
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

        # Combine features
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

        # Handle NaN values
        combined = combined.fillna(0)

        # Cache if requested
        if cache_key:
            self.feature_cache[cache_key] = combined

        return combined

    def extract_multi_timeframe_features(self, data_dict: Dict[str, pd.DataFrame],
                                         current_timestamp: pd.Timestamp = None) -> Dict[str, torch.Tensor]:
        """
        Extract multi-timeframe features for the transformer model.
        Returns features structured for the multi-branch transformer model.

        Args:
            data_dict: Dictionary of data frames
            current_timestamp: Current timestamp to extract features at

        Returns:
            Dictionary with hf_features, mf_features, lf_features, static_features
        """
        # First, get regular features
        combined_features = self.extract_features(data_dict)

        if combined_features.empty:
            self.logger.warning("No features extracted, returning empty tensors")
            return self._create_empty_features()

        # If timestamp provided, use data up to that point
        if current_timestamp is not None:
            features_before = combined_features[combined_features.index <= current_timestamp]
            if features_before.empty:
                self.logger.warning(f"No features before {current_timestamp}, returning empty tensors")
                return self._create_empty_features()
            combined_features = features_before

        # Create simplified multi-timeframe features
        features_dict = self._create_multi_timeframe_tensors(combined_features)

        return features_dict

    def _create_multi_timeframe_tensors(self, features_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Create tensors for different timeframes from combined features.
        This is a simplified version that just segments the features by prefix.

        Args:
            features_df: Combined features dataframe

        Returns:
            Dictionary with tensors for each branch
        """
        # Default batch size of 1 for inference
        batch_size = 1

        # Filter columns by timeframe prefix and create tensors
        hf_cols = [col for col in features_df.columns if col.startswith('1s_')]
        mf_cols = [col for col in features_df.columns if col.startswith('1m_')]
        lf_cols = [col for col in features_df.columns if col.startswith('5m_')]

        # Get available features
        hf_features = features_df[hf_cols].iloc[-self.hf_seq_len:] if hf_cols else pd.DataFrame()
        mf_features = features_df[mf_cols].iloc[-self.mf_seq_len:] if mf_cols else pd.DataFrame()
        lf_features = features_df[lf_cols].iloc[-self.lf_seq_len:] if lf_cols else pd.DataFrame()

        # Create static features (simplified)
        # In practice, this would include position info, support/resistance, etc.
        static_features = np.zeros((batch_size, self.static_feat_dim))

        # Convert to tensors with padding if necessary
        hf_tensor = self._dataframe_to_tensor(hf_features, self.hf_seq_len, self.hf_feat_dim)
        mf_tensor = self._dataframe_to_tensor(mf_features, self.mf_seq_len, self.mf_feat_dim)
        lf_tensor = self._dataframe_to_tensor(lf_features, self.lf_seq_len, self.lf_feat_dim)
        static_tensor = torch.FloatTensor(static_features)

        return {
            'hf_features': hf_tensor,
            'mf_features': mf_tensor,
            'lf_features': lf_tensor,
            'static_features': static_tensor
        }

    def _dataframe_to_tensor(self, df: pd.DataFrame, seq_len: int, feat_dim: int) -> torch.Tensor:
        """
        Convert a dataframe to a tensor with the right shape, padding if necessary.

        Args:
            df: DataFrame to convert
            seq_len: Required sequence length
            feat_dim: Required feature dimension

        Returns:
            Tensor of shape [1, seq_len, feat_dim]
        """
        if df.empty:
            # Return empty tensor of right shape
            return torch.zeros((1, seq_len, feat_dim))

        # Get values as numpy array
        values = df.values

        # Handle shape mismatches
        if values.shape[1] > feat_dim:
            # Too many features, truncate
            values = values[:, :feat_dim]
        elif values.shape[1] < feat_dim:
            # Too few features, pad with zeros
            padding = np.zeros((values.shape[0], feat_dim - values.shape[1]))
            values = np.concatenate([values, padding], axis=1)

        if values.shape[0] > seq_len:
            # Too long, truncate
            values = values[-seq_len:, :]
        elif values.shape[0] < seq_len:
            # Too short, pad with zeros
            padding = np.zeros((seq_len - values.shape[0], values.shape[1]))
            values = np.concatenate([padding, values], axis=0)

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(values).unsqueeze(0)
        return tensor

    def _create_empty_features(self) -> Dict[str, torch.Tensor]:
        """Create empty feature tensors with the right shapes."""
        batch_size = 1
        return {
            'hf_features': torch.zeros((batch_size, self.hf_seq_len, self.hf_feat_dim)),
            'mf_features': torch.zeros((batch_size, self.mf_seq_len, self.mf_feat_dim)),
            'lf_features': torch.zeros((batch_size, self.lf_seq_len, self.lf_feat_dim)),
            'static_features': torch.zeros((batch_size, self.static_feat_dim))
        }