# feature/feature_adapter.py
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


class FeatureAdapter:
    """
    Adapter that transforms flat features from the current feature extractor
    into the multi-branch structure required by the transformer model.
    """

    def __init__(
            self,
            hf_seq_len: int = 60,
            hf_feat_dim: int = 20,
            mf_seq_len: int = 30,
            mf_feat_dim: int = 15,
            lf_seq_len: int = 30,
            lf_feat_dim: int = 10,
            static_feat_dim: int = 15,
            device: torch.device = None
    ):
        self.hf_seq_len = hf_seq_len
        self.hf_feat_dim = hf_feat_dim
        self.mf_seq_len = mf_seq_len
        self.mf_feat_dim = mf_feat_dim
        self.lf_seq_len = lf_seq_len
        self.lf_feat_dim = lf_feat_dim
        self.static_feat_dim = static_feat_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature mapping - which columns go to which branch
        # This will be very simple for now
        self.hf_feature_list = []  # Will be populated dynamically
        self.mf_feature_list = []
        self.lf_feature_list = []
        self.static_feature_list = []

    def adapt_features(self, features_df: pd.DataFrame, current_state: Dict[str, Any] = None) -> Dict[
        str, torch.Tensor]:
        """
        Transform feature dataframe into the format required by the multi-branch transformer.

        Args:
            features_df: DataFrame containing all features
            current_state: Current state dictionary with position info

        Returns:
            Dictionary with tensors for hf_features, mf_features, lf_features, static_features
        """
        if features_df.empty:
            # Return empty tensors if no features available
            return self._create_empty_state_dict()

        # For a minimal implementation, we'll assume:
        # - 1s features are HF features
        # - 1m features are MF features
        # - 5m features are LF features
        # - Position info and other non-timeseries data are static features

        # 1. Identify features by prefix
        if not self.hf_feature_list:
            self._categorize_features(features_df.columns)

        # 2. Create high-frequency features from recent data
        hf_features = self._create_hf_features(features_df)

        # 3. Create medium-frequency features from recent data
        mf_features = self._create_mf_features(features_df)

        # 4. Create low-frequency features from recent data
        lf_features = self._create_lf_features(features_df)

        # 5. Create static features (including position information)
        static_features = self._create_static_features(features_df, current_state)

        # 6. Assemble state dictionary
        state_dict = {
            'hf_features': hf_features,
            'mf_features': mf_features,
            'lf_features': lf_features,
            'static_features': static_features
        }

        return state_dict

    def _categorize_features(self, columns: pd.Index) -> None:
        """
        Categorize features into HF, MF, LF, and static based on column names.
        This is a very simplistic approach for initial testing.
        """
        for col in columns:
            if '1s_' in col or '_price_change' in col:
                self.hf_feature_list.append(col)
            elif '1m_' in col or '_sma_10' in col:
                self.mf_feature_list.append(col)
            elif '5m_' in col:
                self.lf_feature_list.append(col)
            else:
                # Everything else goes to static for now
                self.static_feature_list.append(col)

        # Ensure we don't exceed our feature dimensions
        self.hf_feature_list = self.hf_feature_list[:self.hf_feat_dim]
        self.mf_feature_list = self.mf_feature_list[:self.mf_feat_dim]
        self.lf_feature_list = self.lf_feature_list[:self.lf_feat_dim]
        self.static_feature_list = self.static_feature_list[:self.static_feat_dim]

    def _create_hf_features(self, features_df: pd.DataFrame) -> torch.Tensor:
        """Create high-frequency features tensor."""
        # For HF features, we want recent history
        # Take the last hf_seq_len rows if available
        if len(features_df) >= self.hf_seq_len:
            recent_data = features_df.iloc[-self.hf_seq_len:]
        else:
            # Pad with zeros if not enough history
            padding_rows = self.hf_seq_len - len(features_df)
            padding = pd.DataFrame(0, index=range(padding_rows), columns=features_df.columns)
            recent_data = pd.concat([padding, features_df])

        # Select HF features
        hf_cols = [col for col in self.hf_feature_list if col in recent_data.columns]
        if not hf_cols:
            # If no HF features available, use the first few columns (or zeros)
            if len(recent_data.columns) >= self.hf_feat_dim:
                hf_data = recent_data.iloc[:, :self.hf_feat_dim].values
            else:
                hf_data = np.zeros((self.hf_seq_len, self.hf_feat_dim))
        else:
            hf_data = recent_data[hf_cols].values

            # If we have fewer features than hf_feat_dim, pad with zeros
            if hf_data.shape[1] < self.hf_feat_dim:
                padding = np.zeros((hf_data.shape[0], self.hf_feat_dim - hf_data.shape[1]))
                hf_data = np.concatenate([hf_data, padding], axis=1)

        # Create tensor with batch dimension
        hf_tensor = torch.tensor(hf_data, dtype=torch.float32, device=self.device)
        hf_tensor = hf_tensor.unsqueeze(0)  # Add batch dimension

        return hf_tensor

    def _create_mf_features(self, features_df: pd.DataFrame) -> torch.Tensor:
        """Create medium-frequency features tensor."""
        # Similar to HF but with downsampling for medium frequency features
        # Here we'll take every 2nd row to simulate medium frequency
        if len(features_df) >= self.mf_seq_len * 2:
            # Take every 2nd row from recent history
            indices = range(len(features_df) - self.mf_seq_len * 2, len(features_df), 2)
            recent_data = features_df.iloc[indices]
        else:
            # Just take what we have and pad
            recent_data = features_df.iloc[::2]  # Take every 2nd row
            padding_rows = self.mf_seq_len - len(recent_data)
            if padding_rows > 0:
                padding = pd.DataFrame(0, index=range(padding_rows), columns=features_df.columns)
                recent_data = pd.concat([padding, recent_data])
            recent_data = recent_data.iloc[-self.mf_seq_len:]

        # Select MF features
        mf_cols = [col for col in self.mf_feature_list if col in recent_data.columns]
        if not mf_cols:
            # If no MF features available, use different columns or zeros
            if len(recent_data.columns) >= self.mf_feat_dim:
                mf_data = recent_data.iloc[:, self.hf_feat_dim:self.hf_feat_dim + self.mf_feat_dim].values
            else:
                mf_data = np.zeros((self.mf_seq_len, self.mf_feat_dim))
        else:
            mf_data = recent_data[mf_cols].values

            # Pad if needed
            if mf_data.shape[1] < self.mf_feat_dim:
                padding = np.zeros((mf_data.shape[0], self.mf_feat_dim - mf_data.shape[1]))
                mf_data = np.concatenate([mf_data, padding], axis=1)

        # Create tensor with batch dimension
        mf_tensor = torch.tensor(mf_data, dtype=torch.float32, device=self.device)
        mf_tensor = mf_tensor.unsqueeze(0)  # Add batch dimension

        return mf_tensor

    def _create_lf_features(self, features_df: pd.DataFrame) -> torch.Tensor:
        """Create low-frequency features tensor."""
        # Similar to MF but with more downsampling
        # Here we'll take every 4th row to simulate low frequency
        if len(features_df) >= self.lf_seq_len * 4:
            indices = range(len(features_df) - self.lf_seq_len * 4, len(features_df), 4)
            recent_data = features_df.iloc[indices]
        else:
            recent_data = features_df.iloc[::4]  # Take every 4th row
            padding_rows = self.lf_seq_len - len(recent_data)
            if padding_rows > 0:
                padding = pd.DataFrame(0, index=range(padding_rows), columns=features_df.columns)
                recent_data = pd.concat([padding, recent_data])
            recent_data = recent_data.iloc[-self.lf_seq_len:]

        # Select LF features
        lf_cols = [col for col in self.lf_feature_list if col in recent_data.columns]
        if not lf_cols:
            # If no LF features available, use different columns or zeros
            if len(recent_data.columns) >= self.lf_feat_dim:
                lf_data = recent_data.iloc[:, self.hf_feat_dim + self.mf_feat_dim:
                                              self.hf_feat_dim + self.mf_feat_dim + self.lf_feat_dim].values
            else:
                lf_data = np.zeros((self.lf_seq_len, self.lf_feat_dim))
        else:
            lf_data = recent_data[lf_cols].values

            # Pad if needed
            if lf_data.shape[1] < self.lf_feat_dim:
                padding = np.zeros((lf_data.shape[0], self.lf_feat_dim - lf_data.shape[1]))
                lf_data = np.concatenate([lf_data, padding], axis=1)

        # Create tensor with batch dimension
        lf_tensor = torch.tensor(lf_data, dtype=torch.float32, device=self.device)
        lf_tensor = lf_tensor.unsqueeze(0)  # Add batch dimension

        return lf_tensor

    def _create_static_features(self, features_df: pd.DataFrame, current_state: Dict[str, Any] = None) -> torch.Tensor:
        """Create static features tensor."""
        # For static features, we'll use the last row of data for consistent features
        last_row = features_df.iloc[-1]

        # Start with any explicitly defined static features
        static_cols = [col for col in self.static_feature_list if col in last_row.index]
        if static_cols:
            static_data = last_row[static_cols].values
        else:
            static_data = np.array([])

        # Add position information if available
        if current_state:
            position_info = np.array([
                current_state.get('current_position', 0.0),
                current_state.get('unrealized_pnl', 0.0),
                current_state.get('entry_price', 0.0),
                current_state.get('last_price', 0.0),
                current_state.get('win_rate', 0.5)
            ])

            # Combine feature data with position info
            if len(static_data) > 0:
                static_data = np.concatenate([static_data, position_info])
            else:
                static_data = position_info

        # Ensure correct dimension
        if len(static_data) < self.static_feat_dim:
            padding = np.zeros(self.static_feat_dim - len(static_data))
            static_data = np.concatenate([static_data, padding])
        elif len(static_data) > self.static_feat_dim:
            static_data = static_data[:self.static_feat_dim]

        # Create tensor with batch dimension
        static_tensor = torch.tensor(static_data, dtype=torch.float32, device=self.device)
        static_tensor = static_tensor.unsqueeze(0)  # Add batch dimension

        return static_tensor

    def _create_empty_state_dict(self) -> Dict[str, torch.Tensor]:
        """Create empty state dictionary with zeros."""
        return {
            'hf_features': torch.zeros((1, self.hf_seq_len, self.hf_feat_dim), device=self.device),
            'mf_features': torch.zeros((1, self.mf_seq_len, self.mf_feat_dim), device=self.device),
            'lf_features': torch.zeros((1, self.lf_seq_len, self.lf_feat_dim), device=self.device),
            'static_features': torch.zeros((1, self.static_feat_dim), device=self.device)
        }
