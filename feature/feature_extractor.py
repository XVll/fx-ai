# data/feature/feature_extractor.py
from typing import Dict, List, Union, Tuple, Optional, Any, Set, Callable
import pandas as pd
import numpy as np
from functools import lru_cache
import logging
from data.utils.indicators import calculate_ema, calculate_macd, calculate_vwap


class FeatureExtractor:
    """
    Extracts features from raw market data for use by the AI model.
    Designed with:
    - Modular feature groups
    - Efficient caching
    - Support for incremental updates
    - Extensible architecture
    """

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary with parameters like window sizes, indicators, etc.
            logger: Optional logger
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Default parameters if not in config
        self.price_windows = self.config.get('price_windows', [5, 10, 20, 50])
        self.volume_windows = self.config.get('volume_windows', [5, 10, 20, 50])
        self.ema_periods = self.config.get('ema_periods', [9, 20, 50, 200])
        self.vwap_enabled = self.config.get('vwap_enabled', True)
        self.macd_params = self.config.get('macd_params', {'fast': 12, 'slow': 26, 'signal': 9})

        # Feature group registry - mapping feature group names to feature generation functions
        self.feature_groups = {
            'price': self._extract_price_features,
            'volume': self._extract_volume_features,
            'indicators': self._extract_indicator_features,
            'tape': self._extract_tape_features,
            'quote': self._extract_quote_features,
            'status': self._extract_status_features,
        }

        # Custom feature extractors can be registered
        self.custom_extractors = {}

        # Cached feature DataFrames
        self.feature_cache = {}

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def _extract_price_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract price-based features from OHLCV data.

        Args:
            bars_df: DataFrame with OHLCV bars data
            timeframe: Timeframe string (e.g., "1s", "1m")

        Returns:
            DataFrame with extracted features
        """
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

        features = pd.DataFrame(index=bars_df.index)

        # Price returns for different windows
        for window in self.price_windows:
            col_name = f"return_{window}"
            features[col_name] = bars_df['close'].pct_change(window, fill_method=None)

        # High-Low range relative to close
        features[f"hlc_ratio"] = (bars_df['high'] - bars_df['low']) / bars_df['close']

        # Distance from high/low of different windows
        for window in self.price_windows:
            # Percent off high
            high_window = bars_df['high'].rolling(window).max()
            features[f"pct_off_high_{window}"] = (bars_df['close'] - high_window) / high_window

            # Percent off low
            low_window = bars_df['low'].rolling(window).min()
            features[f"pct_off_low_{window}"] = (bars_df['close'] - low_window) / low_window

        # Whole and half dollar proximity
        features[f"dist_to_whole_dollar"] = bars_df['close'].apply(
            lambda x: abs(x - round(x)) if not pd.isna(x) else np.nan
        )
        features[f"dist_to_half_dollar"] = bars_df['close'].apply(
            lambda x: abs(x - round(x * 2) / 2) if not pd.isna(x) else np.nan
        )

        return features

    def _extract_volume_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract volume-based features from OHLCV data.

        Args:
            bars_df: DataFrame with OHLCV bars data
            timeframe: Timeframe string (e.g., "1s", "1m")

        Returns:
            DataFrame with extracted features
        """
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

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
            if window > 1 and len(bars_df) >= window:
                # Calculate returns and volume changes
                price_changes = bars_df['close'].pct_change(1, fill_method=None)
                volume_changes = bars_df['volume'].pct_change(1, fill_method=None)

                # Create a temporary DataFrame with both series
                temp_df = pd.DataFrame({
                    'price': price_changes,
                    'volume': volume_changes
                })

                # Apply rolling correlation
                rolling_corr = temp_df['price'].rolling(window).corr(temp_df['volume'])
                features[f"{timeframe}_price_vol_corr_{window}"] = rolling_corr
            else:
                features[f"{timeframe}_price_vol_corr_{window}"] = float('nan')

        return features

    def _extract_indicator_features(self, bars_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract technical indicator features from OHLCV data.

        Args:
            bars_df: DataFrame with OHLCV bars data
            timeframe: Timeframe string (e.g., "1s", "1m")

        Returns:
            DataFrame with extracted features
        """
        if bars_df.empty:
            return pd.DataFrame(index=bars_df.index)

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

    def _extract_tape_features(self, trades_df: pd.DataFrame, timeframe: str = "1s") -> pd.DataFrame:
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

        # Convert index to datetime if not already
        if not isinstance(trades_df.index, pd.DatetimeIndex):
            self._log("Trades DataFrame index is not DatetimeIndex. Skipping tape features.", logging.WARNING)
            return pd.DataFrame()

        features = pd.DataFrame()

        # Resample trades to the specified timeframe
        try:
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

            # Tape color/imbalance (if side information is available)
            if 'side' in trades_df.columns:
                # Create buy/sell masks
                buy_mask = trades_df['side'] == 'B'
                sell_mask = trades_df['side'] == 'A'

                # Compute buy volume
                buy_vol = trades_df[buy_mask].resample(timeframe)['size'].sum()
                features['buy_volume'] = buy_vol

                # Compute sell volume
                sell_vol = trades_df[sell_mask].resample(timeframe)['size'].sum()
                features['sell_volume'] = sell_vol

                # Tape imbalance (Buy - Sell) / (Buy + Sell)
                total_vol = buy_vol + sell_vol
                features['tape_imbalance'] = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0)

                # Large trades (identify prints larger than X% of average size)
                avg_size = trades_df['size'].mean()
                large_threshold = avg_size * 5  # 5x average

                large_buys = trades_df[buy_mask & (trades_df['size'] > large_threshold)]
                large_sells = trades_df[sell_mask & (trades_df['size'] > large_threshold)]

                large_buy_vol = large_buys.resample(timeframe)['size'].sum()
                large_sell_vol = large_sells.resample(timeframe)['size'].sum()

                features['large_buy_volume'] = large_buy_vol
                features['large_sell_volume'] = large_sell_vol

                # Count of large trades
                features['large_buy_count'] = large_buys.resample(timeframe).size()
                features['large_sell_count'] = large_sells.resample(timeframe).size()

                # Flag for very large prints (10x average)
                very_large_threshold = avg_size * 10  # 10x average
                features['very_large_buy'] = (trades_df[buy_mask & (trades_df['size'] > very_large_threshold)]
                                              .resample(timeframe).size() > 0).astype(int)
                features['very_large_sell'] = (trades_df[sell_mask & (trades_df['size'] > very_large_threshold)]
                                               .resample(timeframe).size() > 0).astype(int)

            return features

        except Exception as e:
            self._log(f"Error extracting tape features: {str(e)}", logging.ERROR)
            return pd.DataFrame()

    def _extract_quote_features(self, quotes_df: pd.DataFrame, timeframe: str = "1s") -> pd.DataFrame:
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

        # Convert index to datetime if not already
        if not isinstance(quotes_df.index, pd.DatetimeIndex):
            self._log("Quotes DataFrame index is not DatetimeIndex. Skipping quote features.", logging.WARNING)
            return pd.DataFrame()

        features = pd.DataFrame()

        try:
            # Resample quotes to the specified timeframe
            resampled = quotes_df.resample(timeframe).last()

            # Spread (absolute and relative)
            if 'ask_px_00' in resampled.columns and 'bid_px_00' in resampled.columns:
                features['bid_ask_spread'] = resampled['ask_px_00'] - resampled['bid_px_00']
                features['spread_pct'] = features['bid_ask_spread'] / resampled['bid_px_00']

            # Quote imbalance (bid size vs ask size)
            if 'bid_sz_00' in resampled.columns and 'ask_sz_00' in resampled.columns:
                total_size = resampled['bid_sz_00'] + resampled['ask_sz_00']
                features['quote_imbalance'] = np.where(total_size > 0,
                                                       (resampled['bid_sz_00'] - resampled['ask_sz_00']) / total_size,
                                                       0)

                # Bid/Ask size
                features['bid_size'] = resampled['bid_sz_00']
                features['ask_size'] = resampled['ask_sz_00']

            # Quote count
            if 'bid_ct_00' in resampled.columns and 'ask_ct_00' in resampled.columns:
                features['bid_count'] = resampled['bid_ct_00']
                features['ask_count'] = resampled['ask_ct_00']

                # Compute size/count ratios - average order size
                features['bid_avg_size'] = np.where(resampled['bid_ct_00'] > 0,
                                                    resampled['bid_sz_00'] / resampled['bid_ct_00'], 0)
                features['ask_avg_size'] = np.where(resampled['ask_ct_00'] > 0,
                                                    resampled['ask_sz_00'] / resampled['ask_ct_00'], 0)

            # Quote dynamics
            # Calculate changes in bid/ask
            if len(resampled) > 1:
                # Bid/ask price movement
                if 'bid_px_00' in resampled.columns:
                    features['bid_price_change'] = resampled['bid_px_00'].diff()
                if 'ask_px_00' in resampled.columns:
                    features['ask_price_change'] = resampled['ask_px_00'].diff()

                # Size changes
                if 'bid_sz_00' in resampled.columns:
                    features['bid_size_change'] = resampled['bid_sz_00'].diff()
                if 'ask_sz_00' in resampled.columns:
                    features['ask_size_change'] = resampled['ask_sz_00'].diff()

                # Spread changes
                if 'bid_ask_spread' in features.columns:
                    features['spread_change'] = features['bid_ask_spread'].diff()

            return features

        except Exception as e:
            self._log(f"Error extracting quote features: {str(e)}", logging.ERROR)
            return pd.DataFrame()

    def _extract_status_features(self, status_df: pd.DataFrame, timeframe: str = "1s") -> pd.DataFrame:
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

        try:
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
            features['is_halted'] = 0  # Default: not halted
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

        except Exception as e:
            self._log(f"Error extracting status features: {str(e)}", logging.ERROR)
            return pd.DataFrame()

    def register_custom_extractor(self, name: str, extractor_fn: Callable):
        """
        Register a custom feature extractor function.

        Args:
            name: Name of the feature extractor
            extractor_fn: Function that takes data and returns features
        """
        self.custom_extractors[name] = extractor_fn

    def extract_features(self, data_dict: Dict[str, pd.DataFrame],
                         feature_groups: List[str] = None,
                         cache_key: str = None) -> pd.DataFrame:
        """
        Extract features from provided data.

        Args:
            data_dict: Dictionary mapping data types to DataFrames
            feature_groups: List of feature groups to extract. If None, extracts all
            cache_key: Optional key for caching features (e.g., 'symbol_date')

        Returns:
            DataFrame with all extracted features
        """
        all_features = {}

        # Determine which feature groups to include
        if feature_groups is None:
            # Use all registered feature groups
            feature_groups = list(self.feature_groups.keys()) + list(self.custom_extractors.keys())

        # Process OHLCV bars for each timeframe
        for tf in ['1s', '1m', '5m', '1d']:
            key = f'bars_{tf}'
            if key in data_dict and not data_dict[key].empty:
                bars_df = data_dict[key]

                # Extract price features if requested
                if 'price' in feature_groups:
                    try:
                        price_features = self._extract_price_features(bars_df, tf)
                        if not price_features.empty:
                            all_features[f'{tf}_price'] = price_features
                    except Exception as e:
                        self._log(f"Error extracting price features for {tf}: {str(e)}", logging.ERROR)

                # Extract volume features if requested
                if 'volume' in feature_groups:
                    try:
                        volume_features = self._extract_volume_features(bars_df, tf)
                        if not volume_features.empty:
                            all_features[f'{tf}_volume'] = volume_features
                    except Exception as e:
                        self._log(f"Error extracting volume features for {tf}: {str(e)}", logging.ERROR)

                # Extract indicator features if requested
                if 'indicators' in feature_groups:
                    try:
                        indicator_features = self._extract_indicator_features(bars_df, tf)
                        if not indicator_features.empty:
                            all_features[f'{tf}_indicators'] = indicator_features
                    except Exception as e:
                        self._log(f"Error extracting indicator features for {tf}: {str(e)}", logging.ERROR)

        # Process trade data
        if 'tape' in feature_groups and 'trades' in data_dict and not data_dict['trades'].empty:
            try:
                tape_features = self._extract_tape_features(data_dict['trades'])
                if not tape_features.empty:
                    all_features['tape'] = tape_features
            except Exception as e:
                self._log(f"Error extracting tape features: {str(e)}", logging.ERROR)

        # Process quote data
        if 'quote' in feature_groups and 'quotes' in data_dict and not data_dict['quotes'].empty:
            try:
                quote_features = self._extract_quote_features(data_dict['quotes'])
                if not quote_features.empty:
                    all_features['quotes'] = quote_features
            except Exception as e:
                self._log(f"Error extracting quote features: {str(e)}", logging.ERROR)

        # Process status data
        if 'status' in feature_groups and 'status' in data_dict and not data_dict['status'].empty:
            try:
                status_features = self._extract_status_features(data_dict['status'])
                if not status_features.empty:
                    all_features['status'] = status_features
            except Exception as e:
                self._log(f"Error extracting status features: {str(e)}", logging.ERROR)

        # Process custom extractors
        for name, extractor_fn in self.custom_extractors.items():
            if name in feature_groups:
                try:
                    features = extractor_fn(data_dict)
                    if features is not None and not features.empty:
                        all_features[name] = features
                except Exception as e:
                    self._log(f"Error in custom extractor '{name}': {str(e)}", logging.ERROR)

        # Combine all features
        if not all_features:
            self._log("No features extracted. Check input data and feature groups.", logging.WARNING)
            return pd.DataFrame()

        # Get the highest frequency index (most points)
        main_index = None
        for features_df in all_features.values():
            if features_df is not None and not features_df.empty:
                if main_index is None or len(features_df.index) > len(main_index):
                    main_index = features_df.index

        if main_index is None:
            return pd.DataFrame()

        # Create a combined features DataFrame
        combined_features = pd.DataFrame(index=main_index)

        # Add each feature set, forward-filling if necessary
        for feature_name, features_df in all_features.items():
            if features_df is None or features_df.empty:
                continue

            # Reindex to main index
            reindexed = features_df.reindex(main_index, method='ffill')

            # Add prefix to column names to avoid collisions
            # Make sure each column has a unique prefix that includes the feature set name
            reindexed.columns = [f"{feature_name}_{col}" if not col.startswith(feature_name) else col
                                 for col in reindexed.columns]

            # Join to combined features
            combined_features = combined_features.join(reindexed)

        # Handle NaN values
        combined_features = combined_features.ffill().fillna(0)

        # Cache if a cache key is provided
        if cache_key:
            self.feature_cache[cache_key] = combined_features

        return combined_features

    def update_features(self, data_dict: Dict[str, pd.DataFrame],
                        existing_features: pd.DataFrame,
                        feature_groups: List[str] = None) -> pd.DataFrame:
        """
        Update existing features with new data.

        Args:
            data_dict: Dictionary mapping data types to DataFrames with new data
            existing_features: DataFrame with existing features
            feature_groups: List of feature groups to extract. If None, extracts all

        Returns:
            DataFrame with updated features
        """
        if existing_features.empty:
            return self.extract_features(data_dict, feature_groups)

        # Extract features from new data
        new_features = self.extract_features(data_dict, feature_groups)

        if new_features.empty:
            return existing_features

        # Find the latest timestamp in existing features
        latest_timestamp = existing_features.index[-1]

        # Get only new features (after the latest timestamp)
        newer_features = new_features[new_features.index > latest_timestamp]

        if newer_features.empty:
            return existing_features

        # Combine existing and new features
        updated_features = pd.concat([existing_features, newer_features])

        return updated_features

    def get_cached_features(self, cache_key: str) -> pd.DataFrame:
        """
        Get cached features by key.

        Args:
            cache_key: Key for cached features

        Returns:
            DataFrame with cached features or empty DataFrame if not found
        """
        return self.feature_cache.get(cache_key, pd.DataFrame())

    def clear_cache(self, cache_key: str = None):
        """
        Clear feature cache.

        Args:
            cache_key: Specific key to clear. If None, clears all cache
        """
        if cache_key:
            if cache_key in self.feature_cache:
                del self.feature_cache[cache_key]
        else:
            self.feature_cache = {}