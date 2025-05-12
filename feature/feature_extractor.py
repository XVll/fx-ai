# feature/feature_extractor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import logging
from datetime import datetime, timedelta


class FeatureExtractor:
    """
    Extracts trading features from raw market data.
    Designed to capture multi-timeframe signals for momentum trading.

    Features extracted include:
    - Price returns across multiple timeframes
    - Volume metrics
    - Technical indicators (EMA, VWAP, MACD)
    - Tape and order book features
    - Volatility metrics
    """

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary with feature parameters
            logger: Optional logger
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Feature configuration
        self.include_price_features = self.config.get('include_price_features', True)
        self.include_volume_features = self.config.get('include_volume_features', True)
        self.include_technical_indicators = self.config.get('include_technical_indicators', True)
        self.include_tape_features = self.config.get('include_tape_features', True)
        self.include_order_book_features = self.config.get('include_order_book_features', True)
        self.include_volatility_features = self.config.get('include_volatility_features', True)

        # Price return windows
        self.price_return_windows = self.config.get('price_return_windows', [5, 10, 20, 50, 100])

        # Moving average windows
        self.ema_windows = self.config.get('ema_windows', [9, 20, 50, 200])

        # VWAP period
        self.vwap_period = self.config.get('vwap_period', 'day')

        # MACD parameters
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)

        # Volume windows
        self.volume_windows = self.config.get('volume_windows', [5, 10, 20, 50])

        # Volatility windows
        self.volatility_windows = self.config.get('volatility_windows', [10, 20, 50])

        # Support/resistance levels
        self.sup_res_lookback = self.config.get('sup_res_lookback', 100)
        self.sup_res_threshold = self.config.get('sup_res_threshold', 0.005)

        # Half/whole dollars
        self.half_dollar_threshold = self.config.get('half_dollar_threshold', 0.05)  # 5% of price

        # Resampling frequencies
        self.resample_frequencies = self.config.get('resample_frequencies',
                                                    {'1s' , '1m', '5m'})

        # Processing options
        self.use_parallel = self.config.get('use_parallel', False)
        self.fillna_method = self.config.get('fillna_method', 'ffill')

        # Cache for computed features
        self._feature_cache = {}

    def extract_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract features from raw market data.

        Args:
            data_dict: Dictionary with market data
                {
                    'bars_1s': DataFrame with 1-second bars,
                    'bars_1m': DataFrame with 1-minute bars,
                    'bars_5m': DataFrame with 5-minute bars,
                    'bars_1d': DataFrame with daily bars,
                    'trades': DataFrame with trades,
                    'quotes': DataFrame with quotes,
                    'status': DataFrame with status updates
                }

        Returns:
            DataFrame with extracted features
        """
        # Clear cache
        self._feature_cache = {}

        # Check for required data
        if not data_dict:
            self.logger.error("No data provided for feature extraction")
            return pd.DataFrame()

        # Create a base timeline for all features
        timeline_df = self._create_timeline(data_dict)
        if timeline_df.empty:
            self.logger.error("Could not create timeline for features")
            return pd.DataFrame()

        self.logger.info(f"Created timeline with {len(timeline_df)} timestamps")

        # Extract features from each data source
        feature_dfs = []

        # Price features from bars
        if self.include_price_features:
            # 1-second features
            if 'bars_1s' in data_dict and not data_dict['bars_1s'].empty:
                self.logger.info("Extracting 1-second price features")
                feature_dfs.append(self._extract_price_features(data_dict['bars_1s'], '1s'))

            # 1-minute features
            if 'bars_1m' in data_dict and not data_dict['bars_1m'].empty:
                self.logger.info("Extracting 1-minute price features")
                feature_dfs.append(self._extract_price_features(data_dict['bars_1m'], '1m'))

            # 5-minute features
            if 'bars_5m' in data_dict and not data_dict['bars_5m'].empty:
                self.logger.info("Extracting 5-minute price features")
                feature_dfs.append(self._extract_price_features(data_dict['bars_5m'], '5m'))

            # Daily features
            if 'bars_1d' in data_dict and not data_dict['bars_1d'].empty:
                self.logger.info("Extracting daily price features")
                feature_dfs.append(self._extract_price_features(data_dict['bars_1d'], '1d'))

        # Volume features
        if self.include_volume_features:
            # From bars
            for timeframe in ['bars_1s', 'bars_1m', 'bars_5m']:
                if timeframe in data_dict and not data_dict[timeframe].empty:
                    self.logger.info(f"Extracting volume features from {timeframe}")
                    tf_short = timeframe.split('_')[1]
                    feature_dfs.append(self._extract_volume_features(data_dict[timeframe], tf_short))

            # From trades
            if 'trades' in data_dict and not data_dict['trades'].empty:
                self.logger.info("Extracting volume features from trades")
                feature_dfs.append(self._extract_tape_features(data_dict['trades']))

        # Technical indicators
        if self.include_technical_indicators:
            # From 1-minute bars
            if 'bars_1m' in data_dict and not data_dict['bars_1m'].empty:
                self.logger.info("Extracting technical indicators from 1-minute bars")
                feature_dfs.append(self._extract_technical_indicators(data_dict['bars_1m'], '1m'))

            # From 5-minute bars
            if 'bars_5m' in data_dict and not data_dict['bars_5m'].empty:
                self.logger.info("Extracting technical indicators from 5-minute bars")
                feature_dfs.append(self._extract_technical_indicators(data_dict['bars_5m'], '5m'))

        # Order book features
        if self.include_order_book_features and 'quotes' in data_dict and not data_dict['quotes'].empty:
            self.logger.info("Extracting order book features")
            feature_dfs.append(self._extract_order_book_features(data_dict['quotes']))

        # Volatility features
        if self.include_volatility_features:
            # From 1-second bars
            if 'bars_1s' in data_dict and not data_dict['bars_1s'].empty:
                self.logger.info("Extracting volatility features from 1-second bars")
                feature_dfs.append(self._extract_volatility_features(data_dict['bars_1s'], '1s'))

            # From 1-minute bars
            if 'bars_1m' in data_dict and not data_dict['bars_1m'].empty:
                self.logger.info("Extracting volatility features from 1-minute bars")
                feature_dfs.append(self._extract_volatility_features(data_dict['bars_1m'], '1m'))

        # Status features (halts, etc.)
        if 'status' in data_dict and not data_dict['status'].empty:
            self.logger.info("Extracting status features")
            feature_dfs.append(self._extract_status_features(data_dict['status']))

        # Combine all features
        if not feature_dfs:
            self.logger.error("No features were extracted")
            return pd.DataFrame()

        self.logger.info(f"Combining {len(feature_dfs)} feature sets")
        combined_features = timeline_df.copy()

        for df in feature_dfs:
            if not df.empty:
                # Align to timeline and join
                combined_features = combined_features.join(df, how='left')

        # Fill missing values
        if self.fillna_method == 'ffill':
            combined_features = combined_features.ffill()
        elif self.fillna_method == 'bfill':
            combined_features = combined_features.bfill()
        elif self.fillna_method == 'zero':
            combined_features = combined_features.fillna(0)

        # Drop rows with too many missing values
        # Calculate what percentage of features should be present
        min_feature_pct = 0.5  # At least 50% of features should be present
        min_features = int(len(combined_features.columns) * min_feature_pct)

        # Count non-null values in each row
        non_null_counts = combined_features.count(axis=1)

        # Keep only rows with enough features
        valid_rows = non_null_counts >= min_features
        combined_features = combined_features[valid_rows]

        # Final cleanup
        self.logger.info(
            f"Final feature set has {len(combined_features)} rows and {len(combined_features.columns)} columns")
        return combined_features

    def update_features(self, data_dict: Dict[str, pd.DataFrame], existing_features: pd.DataFrame) -> pd.DataFrame:
        """
        Update existing features with new data.

        Args:
            data_dict: New market data
            existing_features: Existing features DataFrame

        Returns:
            Updated features DataFrame
        """
        # Extract features from new data
        new_features = self.extract_features(data_dict)

        if new_features.empty:
            return existing_features

        if existing_features.empty:
            return new_features

        # Get last timestamp from existing features
        last_timestamp = existing_features.index[-1]

        # Get new data after last timestamp
        new_data = new_features[new_features.index > last_timestamp]

        # Combine with existing features
        updated_features = pd.concat([existing_features, new_data])

        return updated_features

    def _create_timeline(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a base timeline for all features.

        Args:
            data_dict: Dictionary with market data

        Returns:
            DataFrame with timeline
        """
        # Get all timestamps from all data sources
        all_timestamps = []

        for key, df in data_dict.items():
            if not df.empty and hasattr(df, 'index'):
                all_timestamps.extend(df.index.tolist())

        if not all_timestamps:
            return pd.DataFrame()

        # Create unique, sorted timeline
        timeline = pd.Series(1, index=pd.DatetimeIndex(sorted(set(all_timestamps))))

        # Create a DataFrame with the timeline index and a placeholder column
        timeline_df = pd.DataFrame({'timeline': 1}, index=timeline.index)

        return timeline_df

    def _extract_price_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract price-based features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string (e.g. '1s', '1m', '5m', '1d')

        Returns:
            DataFrame with price features
        """
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            # Try to find alternative column names
            if 'price' in df.columns:
                # Create OHLCV from trade price
                close_col = 'price'
                resampled = True
            else:
                self.logger.warning(f"Missing required columns for price features in {timeframe} data")
                return pd.DataFrame()
        else:
            close_col = 'close'
            resampled = False

        # Create features DataFrame with same index as input
        features = pd.DataFrame(index=df.index)

        # If we need to resample to the target timeframe
        if resampled:
            # Get the resample frequency
            resample_freq = self.resample_frequencies.get(timeframe, timeframe)

            # Resample the data
            resampled_df = df.resample(resample_freq).agg({
                'price': ['first', 'max', 'min', 'last', 'mean'],
                'size': 'sum'
            })

            # Flatten the MultiIndex columns
            resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]

            # Rename columns to match OHLCV
            resampled_df = resampled_df.rename(columns={
                'price_first': 'open',
                'price_max': 'high',
                'price_min': 'low',
                'price_last': 'close',
                'price_mean': 'vwap',
                'size_sum': 'volume'
            })

            # Use the resampled DataFrame
            df_to_use = resampled_df
            close_col = 'close'
        else:
            # Use the original DataFrame
            df_to_use = df

        # Price changes and returns
        for window in self.price_return_windows:
            # Absolute price changes
            features[f'{timeframe}_price_change_{window}'] = df_to_use[close_col].diff(window)

            # Percentage returns
            features[f'{timeframe}_price_return_{window}'] = df_to_use[close_col].pct_change(window) * 100

        if not resampled and all(col in df.columns for col in required_cols):
            # HLC ratio (close relative to day's range)
            features[f'{timeframe}_price_hlc_ratio'] = (df_to_use['close'] - df_to_use['low']) / (
                        df_to_use['high'] - df_to_use['low'])

            # Percentage off high/low
            for window in [5, 10, 20, 50]:
                if len(df_to_use) >= window:
                    # Rolling high/low
                    rolling_high = df_to_use['high'].rolling(window).max()
                    rolling_low = df_to_use['low'].rolling(window).min()

                    # Percentage off high/low
                    features[f'{timeframe}_price_pct_off_high_{window}'] = (rolling_high - df_to_use[
                        'close']) / rolling_high * 100
                    features[f'{timeframe}_price_pct_off_low_{window}'] = (df_to_use[
                                                                               'close'] - rolling_low) / rolling_low * 100

        # Half/whole dollar proximity
        if timeframe in ['1s', '1m']:  # Only for short timeframes
            features[f'{timeframe}_dist_to_half_dollar'] = self._distance_to_half_dollar(df_to_use[close_col])
            features[f'{timeframe}_dist_to_whole_dollar'] = self._distance_to_whole_dollar(df_to_use[close_col])

            # Normalize to percentage of price
            avg_price = df_to_use[close_col].mean()
            if avg_price > 0:
                features[f'{timeframe}_dist_to_half_dollar_pct'] = features[
                                                                       f'{timeframe}_dist_to_half_dollar'] / avg_price * 100
                features[f'{timeframe}_dist_to_whole_dollar_pct'] = features[
                                                                        f'{timeframe}_dist_to_whole_dollar'] / avg_price * 100

        # Price gaps between timeframes (only for higher timeframes)
        if timeframe in ['1m', '5m', '1d']:
            features[f'{timeframe}_price_gap'] = df_to_use['open'] - df_to_use['close'].shift(1)
            features[f'{timeframe}_price_gap_pct'] = features[f'{timeframe}_price_gap'] / df_to_use['close'].shift(
                1) * 100

        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')

        # Add timeframe prefix to avoid column name collisions
        features.columns = [f"{timeframe}_{col}" if not col.startswith(f"{timeframe}_") else col for col in
                            features.columns]

        return features

    def _extract_volume_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract volume-based features.

        Args:
            df: DataFrame with volume data
            timeframe: Timeframe string

        Returns:
            DataFrame with volume features
        """
        # Check for volume column
        volume_col = 'volume' if 'volume' in df.columns else 'size' if 'size' in df.columns else None
        if volume_col is None:
            self.logger.warning(f"No volume column found in {timeframe} data")
            return pd.DataFrame()

        # Create features DataFrame with same index as input
        features = pd.DataFrame(index=df.index)

        # Volume features
        if len(df) > max(self.volume_windows):
            # Volume changes
            for window in self.volume_windows:
                # Absolute volume change
                features[f'volume_change_{window}'] = df[volume_col].diff(window)

                # Percentage volume change
                features[f'volume_change_pct_{window}'] = df[volume_col].pct_change(window) * 100

            # Rolling volume metrics
            for window in self.volume_windows:
                # Rolling average volume
                features[f'volume_avg_{window}'] = df[volume_col].rolling(window).mean()

                # Volume relative to moving average
                features[f'volume_rel_avg_{window}'] = df[volume_col] / features[f'volume_avg_{window}']

                # Cumulative volume in window
                features[f'volume_cum_{window}'] = df[volume_col].rolling(window).sum()

        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')

        # Add timeframe prefix to avoid column name collisions
        features.columns = [f"{timeframe}_{col}" if not col.startswith(f"{timeframe}_") else col for col in
                            features.columns]

        return features

    def _extract_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract technical indicators like EMAs, VWAP, MACD.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string

        Returns:
            DataFrame with technical indicators
        """
        # Check required columns
        if 'close' not in df.columns:
            self.logger.warning(f"No close column found in {timeframe} data")
            return pd.DataFrame()

        # Create features DataFrame with same index as input
        features = pd.DataFrame(index=df.index)

        # Exponential Moving Averages (EMAs)
        for window in self.ema_windows:
            if len(df) >= window:
                # EMA calculation
                features[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

                # Price relative to EMA
                features[f'close_to_ema_{window}'] = df['close'] / features[f'ema_{window}'] - 1

                # EMA slope (1-period change)
                features[f'ema_{window}_slope'] = features[f'ema_{window}'].diff() / features[f'ema_{window}'].shift(
                    1) * 100

        # VWAP calculation
        if 'volume' in df.columns or 'size' in df.columns:
            volume_col = 'volume' if 'volume' in df.columns else 'size'

            if self.vwap_period == 'day':
                # Daily VWAP - reset at the start of each day
                # Group by date
                df['date'] = df.index.date
                grouped = df.groupby('date')

                # Calculate VWAP for each day
                vwap_list = []
                for date, group in grouped:
                    # Cumulative sum of price * volume and volume
                    cumulative_pv = (group['close'] * group[volume_col]).cumsum()
                    cumulative_volume = group[volume_col].cumsum()

                    # VWAP = cumulative_pv / cumulative_volume
                    vwap = cumulative_pv / cumulative_volume
                    vwap_list.append(vwap)

                # Combine all VWAPs
                if vwap_list:
                    vwap_series = pd.concat(vwap_list)
                    features['vwap'] = vwap_series

                    # Price relative to VWAP
                    features['close_to_vwap'] = df['close'] / features['vwap'] - 1
            else:
                # Rolling VWAP for a specific window
                window = int(self.vwap_period) if self.vwap_period.isdigit() else 100

                # Cumulative sum of price * volume and volume for the rolling window
                rolling_pv = (df['close'] * df[volume_col]).rolling(window).sum()
                rolling_volume = df[volume_col].rolling(window).sum()

                # VWAP = rolling_pv / rolling_volume
                features['vwap'] = rolling_pv / rolling_volume

                # Price relative to VWAP
                features['close_to_vwap'] = df['close'] / features['vwap'] - 1

        # MACD calculation
        if len(df) >= max(self.macd_slow, self.macd_fast, self.macd_signal):
            # Fast and slow EMAs
            ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()

            # MACD line
            features['macd'] = ema_fast - ema_slow

            # Signal line
            features['macd_signal'] = features['macd'].ewm(span=self.macd_signal, adjust=False).mean()

            # Histogram
            features['macd_hist'] = features['macd'] - features['macd_signal']

            # MACD momentum (change in histogram)
            features['macd_momentum'] = features['macd_hist'].diff()

        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')

        # Add timeframe prefix to avoid column name collisions
        features.columns = [f"{timeframe}_{col}" if not col.startswith(f"{timeframe}_") else col for col in
                            features.columns]

        return features

    def _extract_tape_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the trade tape (Time & Sales).

        Args:
            trades_df: DataFrame with trades

        Returns:
            DataFrame with tape features
        """
        if trades_df.empty:
            return pd.DataFrame()

        # Get the timeline - we'll need to resample trades to a uniform timeline
        timeline_index = pd.date_range(
            start=trades_df.index.min(),
            end=trades_df.index.max(),
            freq='1s'  # 1-second frequency
        )

        # Create features DataFrame with timeline index
        features = pd.DataFrame(index=timeline_index)

        # Check required columns
        required_cols = ['price', 'side']
        size_col = 'size' if 'size' in trades_df.columns else 'volume' if 'volume' in trades_df.columns else None

        if not all(col in trades_df.columns for col in required_cols) or size_col is None:
            self.logger.warning("Missing required columns for tape features")
            return pd.DataFrame()

        # Resample trades to 1-second bins
        resampled = trades_df.resample('1s').agg({
            'price': ['count', 'mean', 'min', 'max', 'last'],
            size_col: 'sum'
        })

        # Flatten the MultiIndex columns
        resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]

        # Rename columns
        resampled = resampled.rename(columns={
            f'price_count': 'trade_count',
            'price_mean': 'price_mean',
            'price_min': 'price_min',
            'price_max': 'price_max',
            'price_last': 'price_last',
            f'{size_col}_sum': 'volume'
        })

        # Add calculated features
        features['trade_count'] = resampled['trade_count'].fillna(0)
        features['price'] = resampled['price_last'].ffill()
        features['volume'] = resampled['volume'].fillna(0)

        # Calculate trade speed (trades per second) for different windows
        for window in [5, 10, 30, 60]:
            features[f'trade_speed_{window}s'] = features['trade_count'].rolling(window).sum() / window

        # Separate trades by side (buy/sell)
        buy_trades = trades_df[trades_df['side'] == 'B']
        sell_trades = trades_df[trades_df['side'] == 'A']

        # Resample buy/sell trades to 1-second bins
        buy_resampled = buy_trades.resample('1s').agg({size_col: 'sum'})
        sell_resampled = sell_trades.resample('1s').agg({size_col: 'sum'})

        # Calculate tape imbalance (buy volume - sell volume) / total volume
        features['buy_volume'] = buy_resampled[size_col].fillna(0)
        features['sell_volume'] = sell_resampled[size_col].fillna(0)

        # Calculate tape imbalance for different windows
        for window in [5, 10, 30, 60]:
            # Sum volumes over window
            buy_vol_window = features['buy_volume'].rolling(window).sum()
            sell_vol_window = features['sell_volume'].rolling(window).sum()
            total_vol_window = buy_vol_window + sell_vol_window

            # Calculate imbalance
            with np.errstate(divide='ignore', invalid='ignore'):
                features[f'tape_imbalance_{window}s'] = np.where(
                    total_vol_window > 0,
                    (buy_vol_window - sell_vol_window) / total_vol_window,
                    0
                )

        # Detect large trades (sudden volume spikes)
        median_volume = features['volume'].rolling(60).median()
        features['volume_ratio'] = features['volume'] / median_volume

        # Flag large trades (e.g., 5x normal volume)
        features['large_trade_flag'] = (features['volume_ratio'] > 5).astype(int)

        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')

        # Add 'tape' prefix to avoid column name collisions
        features.columns = [f"tape_{col}" if not col.startswith("tape_") else col for col in features.columns]

        return features

    def _extract_order_book_features(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the order book (quotes).

        Args:
            quotes_df: DataFrame with quotes

        Returns:
            DataFrame with order book features
        """
        if quotes_df.empty:
            return pd.DataFrame()

        # Get the timeline - we'll need to resample quotes to a uniform timeline
        timeline_index = pd.date_range(
            start=quotes_df.index.min(),
            end=quotes_df.index.max(),
            freq='1s'  # 1-second frequency
        )

        # Create features DataFrame with timeline index
        features = pd.DataFrame(index=timeline_index)

        # Check required columns
        bid_px_col = 'bid_px_00' if 'bid_px_00' in quotes_df.columns else 'bid' if 'bid' in quotes_df.columns else None
        ask_px_col = 'ask_px_00' if 'ask_px_00' in quotes_df.columns else 'ask' if 'ask' in quotes_df.columns else None
        bid_sz_col = 'bid_sz_00' if 'bid_sz_00' in quotes_df.columns else 'bid_size' if 'bid_size' in quotes_df.columns else None
        ask_sz_col = 'ask_sz_00' if 'ask_sz_00' in quotes_df.columns else 'ask_size' if 'ask_size' in quotes_df.columns else None

        if not all(col is not None for col in [bid_px_col, ask_px_col, bid_sz_col, ask_sz_col]):
            self.logger.warning("Missing required columns for order book features")
            return pd.DataFrame()

        # Resample quotes to 1-second bins
        # For price, we want the last value in the bin
        # For size, we could use mean, last, or max depending on the use case
        resampled = quotes_df.resample('1s').agg({
            bid_px_col: 'last',
            ask_px_col: 'last',
            bid_sz_col: 'last',
            ask_sz_col: 'last'
        })

        # Add price features
        features['bid'] = resampled[bid_px_col].ffill()
        features['ask'] = resampled[ask_px_col].ffill()
        features['mid'] = (features['bid'] + features['ask']) / 2
        features['spread'] = features['ask'] - features['bid']
        features['spread_pct'] = features['spread'] / features['mid'] * 100

        # Add size features
        features['bid_size'] = resampled[bid_sz_col].ffill()
        features['ask_size'] = resampled[ask_sz_col].ffill()
        features['total_size'] = features['bid_size'] + features['ask_size']

        # Calculate order book imbalance
        with np.errstate(divide='ignore', invalid='ignore'):
            features['ob_imbalance'] = np.where(
                features['total_size'] > 0,
                (features['bid_size'] - features['ask_size']) / features['total_size'],
                0
            )

        # Calculate rolling statistics for order book features
        for window in [5, 10, 30, 60]:
            # Spread moving average
            features[f'spread_ma_{window}s'] = features['spread'].rolling(window).mean()
            features[f'spread_pct_ma_{window}s'] = features['spread_pct'].rolling(window).mean()

            # Imbalance moving average
            features[f'ob_imbalance_ma_{window}s'] = features['ob_imbalance'].rolling(window).mean()

            # Bid/ask size moving average
            features[f'bid_size_ma_{window}s'] = features['bid_size'].rolling(window).mean()
            features[f'ask_size_ma_{window}s'] = features['ask_size'].rolling(window).mean()

        # Detect significant order book changes
        features['spread_change'] = features['spread'].diff()
        features['spread_pct_change'] = features['spread_pct'].diff()
        features['ob_imbalance_change'] = features['ob_imbalance'].diff()

        # Flag significant changes (e.g., sudden spread widening or narrowing)
        features['spread_widening'] = (features['spread_change'] > features['spread'].rolling(30).std()).astype(int)
        features['spread_narrowing'] = (features['spread_change'] < -features['spread'].rolling(30).std()).astype(int)

        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')

        # Add 'ob' prefix to avoid column name collisions
        features.columns = [f"ob_{col}" if not col.startswith("ob_") else col for col in features.columns]

        return features

    def _extract_volatility_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Extract volatility features.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string

        Returns:
            DataFrame with volatility features
        """
        # Check required columns
        if 'close' not in df.columns:
            self.logger.warning(f"No close column found in {timeframe} data")
            return pd.DataFrame()

        # Create features DataFrame with same index as input
        features = pd.DataFrame(index=df.index)

        # Calculate returns
        returns = df['close'].pct_change()

        # Calculate volatility for different windows
        for window in self.volatility_windows:
            if len(df) >= window:
                # Standard deviation of returns (volatility)
                features[f'volatility_{window}'] = returns.rolling(window).std() * 100  # Convert to percentage

                # Annualized volatility (depends on timeframe)
                annualization_factor = {
                    '1s': np.sqrt(252 * 6.5 * 60 * 60),  # 252 days * 6.5 hours * 60 minutes * 60 seconds
                    '1m': np.sqrt(252 * 6.5 * 60),  # 252 days * 6.5 hours * 60 minutes
                    '5m': np.sqrt(252 * 6.5 * 12),  # 252 days * 6.5 hours * 12 (5-minute bars per hour)
                    '1d': np.sqrt(252)  # 252 trading days
                }.get(timeframe, np.sqrt(252))

                features[f'volatility_annual_{window}'] = features[f'volatility_{window}'] * annualization_factor

        # Calculate high-low range
        if all(col in df.columns for col in ['high', 'low']):
            # High-low range
            features['hl_range'] = (df['high'] - df['low']) / df['close'] * 100  # Convert to percentage

            # Rolling high-low range
            for window in self.volatility_windows:
                features[f'hl_range_{window}'] = features['hl_range'].rolling(window).mean()

        # Calculate LULD bands (limit up/limit down)
        if timeframe in ['1m', '5m']:  # Only for minute-level data
            # Use 5-minute moving average for reference price
            reference_price = df['close'].rolling(5).mean()

            # Default tier 2 stock percentages
            tier2_up_pct = 0.10  # 10%
            tier2_down_pct = 0.10  # 10%

            # Calculate LULD bands
            features['luld_up'] = reference_price * (1 + tier2_up_pct)
            features['luld_down'] = reference_price * (1 - tier2_down_pct)

            # Distance to LULD bands (as percentage of current price)
            features['dist_to_luld_up_pct'] = (features['luld_up'] - df['close']) / df['close'] * 100
            features['dist_to_luld_down_pct'] = (df['close'] - features['luld_down']) / df['close'] * 100

        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')

        # Add timeframe prefix to avoid column name collisions
        features.columns = [f"{timeframe}_{col}" if not col.startswith(f"{timeframe}_") else col for col in
                            features.columns]

        return features

    def _extract_status_features(self, status_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from status updates (halts, etc.).

        Args:
            status_df: DataFrame with status updates

        Returns:
            DataFrame with status features
        """
        if status_df.empty:
            return pd.DataFrame()

        # Get the timeline - we'll need to forward-fill status to a uniform timeline
        timeline_index = pd.date_range(
            start=status_df.index.min(),
            end=status_df.index.max(),
            freq='1s'  # 1-second frequency
        )

        # Create features DataFrame with timeline index
        features = pd.DataFrame(index=timeline_index)

        # Check required columns
        required_cols = ['is_trading', 'is_quoting']
        if not all(col in status_df.columns for col in required_cols):
            self.logger.warning("Missing required columns for status features")
            return pd.DataFrame()

        # Convert status indicators to binary
        is_trading = status_df['is_trading'].apply(lambda x: 1 if x == 'Y' else 0 if x == 'N' else np.nan)
        is_quoting = status_df['is_quoting'].apply(lambda x: 1 if x == 'Y' else 0 if x == 'N' else np.nan)

        # Resample to 1-second frequency with forward-fill
        features['is_trading'] = is_trading.reindex(timeline_index, method='ffill')
        features['is_quoting'] = is_quoting.reindex(timeline_index, method='ffill')

        # Add 'status' prefix to avoid column name collisions
        features.columns = [f"status_{col}" for col in features.columns]

        return features

    def _distance_to_half_dollar(self, prices: pd.Series) -> pd.Series:
        """
        Calculate distance to the nearest half-dollar level.

        Args:
            prices: Series of prices

        Returns:
            Series with distances
        """
        # Calculate the distance to the nearest half-dollar
        floor_half = np.floor(prices * 2) / 2
        ceil_half = np.ceil(prices * 2) / 2

        # Return the minimum distance
        return np.minimum(prices - floor_half, ceil_half - prices)

    def _distance_to_whole_dollar(self, prices: pd.Series) -> pd.Series:
        """
        Calculate distance to the nearest whole-dollar level.

        Args:
            prices: Series of prices

        Returns:
            Series with distances
        """
        # Calculate the distance to the nearest whole-dollar
        floor_dollar = np.floor(prices)
        ceil_dollar = np.ceil(prices)

        # Return the minimum distance
        return np.minimum(prices - floor_dollar, ceil_dollar - prices)