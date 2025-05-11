# data/feature/feature_extractor.py
from typing import Dict, List, Union, Tuple, Optional
import pandas as pd
import numpy as np
from ..utils.indicators import calculate_ema, calculate_macd, calculate_vwap

class FeatureExtractor:
    """
    Extracts features from raw market data for use by the AI model.
    Generates features across multiple time frames.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary with parameters like window sizes, indicators, etc.
        """
        self.config = config or {}

        # Default parameters if not in config
        self.price_windows = self.config.get('price_windows', [5, 10, 20, 50])
        self.volume_windows = self.config.get('volume_windows', [5, 10, 20, 50])
        self.ema_periods = self.config.get('ema_periods', [9, 20, 50, 200])
        self.vwap_enabled = self.config.get('vwap_enabled', True)
        self.macd_params = self.config.get('macd_params', {'fast': 12, 'slow': 26, 'signal': 9})

        # Feature storage
        self.feature_cache = {}

    def extract_price_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract price-based features from OHLCV data.

        Args:
            bars_df: DataFrame with OHLCV bars data
            timeframe: Timeframe string (e.g., "1s", "1m")

        Returns:
            DataFrame with extracted features
        """
        if bars_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=bars_df.index)

        # Price returns for different windows
        for window in self.price_windows:
            col_name = f"{timeframe}_return_{window}"
            features[col_name] = bars_df['close'].pct_change(window)

        # High-Low range relative to close
        features[f"{timeframe}_hlc_ratio"] = (bars_df['high'] - bars_df['low']) / bars_df['close']

        # Distance from high/low of different windows
        for window in self.price_windows:
            # Percent off high
            high_window = bars_df['high'].rolling(window).max()
            features[f"{timeframe}_pct_off_high_{window}"] = (bars_df['close'] - high_window) / high_window

            # Percent off low
            low_window = bars_df['low'].rolling(window).min()
            features[f"{timeframe}_pct_off_low_{window}"] = (bars_df['close'] - low_window) / low_window

        # Whole and half dollar proximity
        features[f"{timeframe}_dist_to_whole_dollar"] = bars_df['close'].apply(
            lambda x: abs(x - round(x))
        )
        features[f"{timeframe}_dist_to_half_dollar"] = bars_df['close'].apply(
            lambda x: abs(x - round(x * 2) / 2)
        )

        return features

    def extract_volume_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract volume-based features from OHLCV data.

        Args:
            bars_df: DataFrame with OHLCV bars data
            timeframe: Timeframe string (e.g., "1s", "1m")

        Returns:
            DataFrame with extracted features
        """
        if bars_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=bars_df.index)

        # Volume relative to average for different windows
        for window in self.volume_windows:
            vol_avg = bars_df['volume'].rolling(window).mean()
            features[f"{timeframe}_vol_ratio_{window}"] = bars_df['volume'] / vol_avg

        # Volume increasing/decreasing flags
        for window in self.volume_windows:
            prev_vol = bars_df['volume'].shift(window)
            features[f"{timeframe}_vol_incr_{window}"] = (bars_df['volume'] > prev_vol).astype(int)

        # Volume spikes (> 5x average)
        for window in self.volume_windows:
            vol_avg = bars_df['volume'].rolling(window).mean()
            features[f"{timeframe}_vol_spike_{window}"] = (bars_df['volume'] > vol_avg * 5).astype(int)

        # Price-volume correlation
        for window in self.volume_windows:
            price_returns = bars_df['close'].pct_change(1).rolling(window)
            volume_changes = bars_df['volume'].pct_change(1).rolling(window)

            # Correlation between price changes and volume changes
            features[f"{timeframe}_price_vol_corr_{window}"] = price_returns.corr(volume_changes)

        return features

    def extract_indicator_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract technical indicator features from OHLCV data.

        Args:
            bars_df: DataFrame with OHLCV bars data
            timeframe: Timeframe string (e.g., "1s", "1m")

        Returns:
            DataFrame with extracted features
        """
        if bars_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=bars_df.index)

        # EMAs
        for period in self.ema_periods:
            ema_name = f"{timeframe}_ema_{period}"
            ema_values = calculate_ema(bars_df['close'], period)
            features[ema_name] = ema_values

            # Distance from EMA (percentage)
            features[f"{timeframe}_pct_from_ema_{period}"] = (bars_df['close'] - ema_values) / ema_values

            # Above/below EMA (binary indicator)
            features[f"{timeframe}_above_ema_{period}"] = (bars_df['close'] > ema_values).astype(int)

        # VWAP (if enabled)
        if self.vwap_enabled:
            vwap_values = calculate_vwap(bars_df)
            features[f"{timeframe}_vwap"] = vwap_values
            features[f"{timeframe}_pct_from_vwap"] = (bars_df['close'] - vwap_values) / vwap_values
            features[f"{timeframe}_above_vwap"] = (bars_df['close'] > vwap_values).astype(int)

        # MACD
        macd, signal, hist = calculate_macd(
            bars_df['close'],
            self.macd_params['fast'],
            self.macd_params['slow'],
            self.macd_params['signal']
        )
        features[f"{timeframe}_macd"] = macd
        features[f"{timeframe}_macd_signal"] = signal
        features[f"{timeframe}_macd_hist"] = hist
        features[f"{timeframe}_macd_crossover"] = (hist > 0) & (hist.shift(1) <= 0)
        features[f"{timeframe}_macd_crossunder"] = (hist < 0) & (hist.shift(1) >= 0)

        return features

    def extract_tape_features(self, trades_df: pd.DataFrame, timeframe: str = "1s") -> pd.DataFrame:
        """
        Extract features from time & sales data (trades).

        Args:
            trades_df: DataFrame with trades data
            timeframe: Timeframe to resample trades (default "1s")

        Returns:
            DataFrame with extracted features
        """
        if trades_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame()

        # Resample trades to the specified timeframe
        resampled = trades_df.resample(timeframe).agg({
            'price': ['first', 'last', 'mean', 'count'],
            'size': ['sum', 'mean', 'max']
        })

        resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]

        # Trade frequency (trades per second)
        features['trade_frequency'] = resampled['price_count']

        # Average trade size
        features['avg_trade_size'] = resampled['size_mean']

        # Total volume
        features['total_volume'] = resampled['size_sum']

        # Largest trade
        features['max_trade_size'] = resampled['size_max']

        # Classify trades by size
        # Assuming trades_df has a 'side' column with 'A' for sell and 'B' for buy
        if 'side' in trades_df.columns:
            # Buy volume
            buy_vol = trades_df[trades_df['side'] == 'B'].resample(timeframe)['size'].sum()
            features['buy_volume'] = buy_vol

            # Sell volume
            sell_vol = trades_df[trades_df['side'] == 'A'].resample(timeframe)['size'].sum()
            features['sell_volume'] = sell_vol

            # Tape imbalance (Buy - Sell) / (Buy + Sell)
            total_vol = buy_vol + sell_vol
            features['tape_imbalance'] = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0)

        return features

    def extract_quote_features(self, quotes_df: pd.DataFrame, timeframe: str = "1s") -> pd.DataFrame:
        """
        Extract features from quote data (bid/ask).

        Args:
            quotes_df: DataFrame with quote data
            timeframe: Timeframe to resample quotes (default "1s")

        Returns:
            DataFrame with extracted features
        """
        if quotes_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame()

        # Resample quotes to the specified timeframe
        resampled = quotes_df.resample(timeframe).last()

        # Spread (absolute and relative)
        features['bid_ask_spread'] = resampled['ask_px_00'] - resampled['bid_px_00']
        features['spread_pct'] = features['bid_ask_spread'] / resampled['bid_px_00']

        # Quote imbalance (bid size vs ask size)
        total_size = resampled['bid_sz_00'] + resampled['ask_sz_00']
        features['quote_imbalance'] = np.where(total_size > 0,
                                               (resampled['bid_sz_00'] - resampled['ask_sz_00']) / total_size,
                                               0)

        # Bid/Ask size
        features['bid_size'] = resampled['bid_sz_00']
        features['ask_size'] = resampled['ask_sz_00']

        # Quote count
        features['bid_count'] = resampled['bid_ct_00']
        features['ask_count'] = resampled['ask_ct_00']

        return features

    def extract_status_features(self, status_df: pd.DataFrame, timeframe: str = "1s") -> pd.DataFrame:
        """
        Extract features from status updates (halts, etc.).

        Args:
            status_df: DataFrame with status data
            timeframe: Timeframe to resample status (default "1s")

        Returns:
            DataFrame with extracted features
        """
        if status_df.empty:
            return pd.DataFrame()

        # Create a DataFrame covering the entire period
        start_time = status_df.index.min()
        end_time = status_df.index.max()
        if pd.isna(start_time) or pd.isna(end_time):
            return pd.DataFrame()

        idx = pd.date_range(start=start_time, end=end_time, freq=timeframe)
        features = pd.DataFrame(index=idx)

        # Fill values for the whole period
        features['is_trading'] = 1  # Default: trading
        features['is_quoting'] = 1  # Default: quoting
        features['is_halted'] = 0   # Default: not halted
        features['is_short_sell_restricted'] = 0  # Default: not restricted

        # Update based on status messages
        for _, row in status_df.iterrows():
            timestamp = row.name

            # Find the relevant index in our features DataFrame
            idx_pos = features.index.get_indexer([timestamp], method='ffill')[0]
            if idx_pos < 0:
                continue

            # Update trading status
            if 'is_trading' in row and row['is_trading'] == 'N':
                features.loc[features.index[idx_pos:], 'is_trading'] = 0
            elif 'is_trading' in row and row['is_trading'] == 'Y':
                features.loc[features.index[idx_pos:], 'is_trading'] = 1

            # Update quoting status
            if 'is_quoting' in row and row['is_quoting'] == 'N':
                features.loc[features.index[idx_pos:], 'is_quoting'] = 0
            elif 'is_quoting' in row and row['is_quoting'] == 'Y':
                features.loc[features.index[idx_pos:], 'is_quoting'] = 1

            # Update halt status (action 8 is a halt)
            if 'action' in row and row['action'] == 8:
                features.loc[features.index[idx_pos:], 'is_halted'] = 1
            # Resume trading (action 7)
            elif 'action' in row and row['action'] == 7:
                features.loc[features.index[idx_pos:], 'is_halted'] = 0

            # Update short sell restriction
            if 'is_short_sell_restricted' in row and row['is_short_sell_restricted'] == 'Y':
                features.loc[features.index[idx_pos:], 'is_short_sell_restricted'] = 1
            elif 'is_short_sell_restricted' in row and row['is_short_sell_restricted'] == 'N':
                features.loc[features.index[idx_pos:], 'is_short_sell_restricted'] = 0

        return features

    def extract_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract all features from provided data.

        Args:
            data_dict: Dictionary mapping data types to DataFrames
                - 'bars_1s': 1-second OHLCV bars
                - 'bars_1m': 1-minute OHLCV bars
                - 'bars_5m': 5-minute OHLCV bars
                - 'bars_1d': Daily OHLCV bars
                - 'trades': Trade data
                - 'quotes': Quote data
                - 'status': Status updates

        Returns:
            DataFrame with all extracted features
        """
        all_features = {}

        # Process OHLCV bars for each timeframe
        for tf in ['1s', '1m', '5m', '1d']:
            key = f'bars_{tf}'
            if key in data_dict and not data_dict[key].empty:
                # Extract price features
                price_features = self.extract_price_features(data_dict[key], tf)
                if not price_features.empty:
                    all_features[f'{tf}_price'] = price_features

                # Extract volume features
                volume_features = self.extract_volume_features(data_dict[key], tf)
                if not volume_features.empty:
                    all_features[f'{tf}_volume'] = volume_features

                # Extract indicator features
                indicator_features = self.extract_indicator_features(data_dict[key], tf)
                if not indicator_features.empty:
                    all_features[f'{tf}_indicators'] = indicator_features

        # Process trade data
        if 'trades' in data_dict and not data_dict['trades'].empty:
            tape_features = self.extract_tape_features(data_dict['trades'])
            if not tape_features.empty:
                all_features['tape'] = tape_features

        # Process quote data
        if 'quotes' in data_dict and not data_dict['quotes'].empty:
            quote_features = self.extract_quote_features(data_dict['quotes'])
            if not quote_features.empty:
                all_features['quotes'] = quote_features

        # Process status data
        if 'status' in data_dict and not data_dict['status'].empty:
            status_features = self.extract_status_features(data_dict['status'])
            if not status_features.empty:
                all_features['status'] = status_features

        # Combine all features
        if not all_features:
            return pd.DataFrame()

        # Get the collective timeframe (highest frequency)
        main_index = None
        for features_df in all_features.values():
            if main_index is None or len(features_df.index) > len(main_index):
                main_index = features_df.index

        # Create a combined features DataFrame
        combined_features = pd.DataFrame(index=main_index)

        # Add each feature set, forward-filling if necessary
        for feature_name, features_df in all_features.items():
            # Reindex to main index
            reindexed = features_df.reindex(main_index, method='ffill')

            # Add prefix to column names
            reindexed.columns = [f"{feature_name}_{col}" for col in reindexed.columns]

            # Join to combined features
            combined_features = combined_features.join(reindexed)

        # Handle NaN values
        combined_features.fillna(0, inplace=True)

        return combined_features