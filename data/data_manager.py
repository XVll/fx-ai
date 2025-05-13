# data/data_manager.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import logging

from data.provider.data_provider import DataProvider, HistoricalDataProvider, LiveDataProvider
from data.utils.helpers import ensure_timezone_aware


class DataManager:
    """
    Centralized data management class responsible for:
    - Loading and caching data in memory
    - Providing access to different data types
    - Managing data lifecycle (loading, unloading)
    """

    def __init__(self, provider: DataProvider, logger=None):
        """
        Initialize the data manager.

        Args:
            provider: DataProvider instance (historical or live)
            logger: Optional logger
        """
        self.provider = provider
        self.logger = logger or logging.getLogger(__name__)

        # Data cache structure:
        # {symbol: {data_type: {cache_key: DataFrame}}}
        self.data_cache = {}

        # Track loaded data ranges:
        # {symbol: {data_type: [(start_date, end_date)]}}
        self.loaded_ranges = {}

        # Current state tracking
        self.current_symbol = None
        self.is_live = isinstance(provider, LiveDataProvider)

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def _create_cache_key(self, start_time: datetime, end_time: datetime) -> str:
        """Create a unique key for caching based on time range."""
        return f"{start_time.isoformat()}_{end_time.isoformat()}"

    def _check_memory_cache(self, symbol: str, data_type: str,
                            start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Check if data is in memory cache and return it if found."""
        if symbol not in self.data_cache or data_type not in self.data_cache[symbol]:
            return None

        # First check if any existing cache key exactly matches our request
        cache_key = self._create_cache_key(start_time, end_time)
        if cache_key in self.data_cache[symbol][data_type]:
            self._log(f"Cache hit for {symbol} {data_type} in memory cache")
            return self.data_cache[symbol][data_type][cache_key]

        # If no exact match, check if we have a superset of the data
        for key, df in self.data_cache[symbol][data_type].items():
            if not df.empty:
                try:
                    df_start = df.index.min()
                    df_end = df.index.max()

                    # If the cached data fully contains our request
                    if df_start <= start_time and df_end >= end_time:
                        self._log(f"Partial cache hit for {symbol} {data_type} in memory cache")
                        # Extract the slice we need and cache it for next time
                        result = df[start_time:end_time]
                        if not result.empty:
                            self._save_to_memory_cache(symbol, data_type, start_time, end_time, result)
                            return result
                except Exception as e:
                    self._log(f"Error checking cache: {e}", logging.DEBUG)

        return None

    def _save_to_memory_cache(self, symbol: str, data_type: str,
                              start_time: datetime, end_time: datetime,
                              df: pd.DataFrame) -> None:
        """Save data to memory cache."""
        if df.empty:
            return

        if symbol not in self.data_cache:
            self.data_cache[symbol] = {}

        if data_type not in self.data_cache[symbol]:
            self.data_cache[symbol][data_type] = {}

        cache_key = self._create_cache_key(start_time, end_time)
        self.data_cache[symbol][data_type][cache_key] = df

    def _update_loaded_ranges(self, symbol: str, data_type: str,
                              start_time: datetime, end_time: datetime) -> None:
        """Update the tracking of loaded date ranges."""
        if symbol not in self.loaded_ranges:
            self.loaded_ranges[symbol] = {}

        if data_type not in self.loaded_ranges[symbol]:
            self.loaded_ranges[symbol][data_type] = []

        # Add the range
        self.loaded_ranges[symbol][data_type].append((start_time, end_time))

        # Merge overlapping ranges
        ranges = self.loaded_ranges[symbol][data_type]
        if len(ranges) > 1:
            ranges.sort(key=lambda r: r[0])  # Sort by start time
            merged = [ranges[0]]

            for current in ranges[1:]:
                prev = merged[-1]
                if current[0] <= prev[1]:  # Ranges overlap
                    merged[-1] = (prev[0], max(prev[1], current[1]))
                else:
                    merged.append(current)

            self.loaded_ranges[symbol][data_type] = merged

    def load_data(self, symbols: Union[str, List[str]],
                  start_time: Union[datetime, str],
                  end_time: Union[datetime, str],
                  data_types: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load data for symbols in a date range.

        Args:
            symbols: Symbol or list of symbols to load data for
            start_time: Start time for data
            end_time: End time for data
            data_types: List of data types to load. If None, loads all available types
                        Supported types: 'bars_1s', 'bars_1m', 'bars_5m', 'bars_1d',
                                        'trades', 'quotes', 'status'

        Returns:
            Dictionary mapping symbols to dictionaries of data types
        """
        if self.is_live:
            self._log(f"Cannot load historical data with live provider", logging.WARNING)
            return {}

        if not isinstance(self.provider, HistoricalDataProvider):
            self._log(f"Provider must be a HistoricalDataProvider for load_data", logging.ERROR)
            return {}

        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Default data types if not specified
        if data_types is None:
            data_types = ["bars_1s", "bars_1m", "bars_5m", "bars_1d", "trades", "quotes", "status"]

        # Convert string times to datetime
        start_dt = ensure_timezone_aware(start_time, is_end_time=False)
        end_dt = ensure_timezone_aware(end_time, is_end_time=True)

        # Load data for each symbol
        results = {}
        for symbol in symbols:
            self.current_symbol = symbol
            self._log(f"Loading data for {symbol} from {start_dt} to {end_dt}")

            symbol_data = {}

            # Process each data type
            for data_type in data_types:
                # Skip unsupported timeframes
                if data_type.startswith('bars_'):
                    timeframe = data_type.split('_')[1]
                    if timeframe not in ["1s", "1m", "5m", "1d"]:
                        self._log(f"Skipping unsupported timeframe: {timeframe}")
                        continue

                try:
                    # Check memory cache
                    cached_data = self._check_memory_cache(symbol, data_type, start_dt, end_dt)
                    if cached_data is not None:
                        symbol_data[data_type] = cached_data
                        continue

                    # Load from provider
                    self._log(f"Loading {data_type} from provider for {symbol}")
                    df = self._load_from_provider(symbol, data_type, start_dt, end_dt)

                    if df is not None and not df.empty:
                        # Save to memory cache
                        self._save_to_memory_cache(symbol, data_type, start_dt, end_dt, df)

                        # Update loaded ranges
                        self._update_loaded_ranges(symbol, data_type, start_dt, end_dt)

                        symbol_data[data_type] = df
                        self._log(f"Loaded {len(df)} rows of {data_type} data for {symbol}")
                    else:
                        self._log(f"No {data_type} data found for {symbol}", logging.WARNING)

                except Exception as e:
                    self._log(f"Error loading {data_type} for {symbol}: {e}", logging.ERROR)
                    import traceback
                    self._log(f"Traceback: {traceback.format_exc()}", logging.DEBUG)

            results[symbol] = symbol_data

        return results

    def _load_from_provider(self, symbol: str, data_type: str,
                            start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Load data from the provider based on data type."""
        provider = self.provider

        if data_type == 'trades':
            return provider.get_trades(symbol, start_dt, end_dt)
        elif data_type == 'quotes':
            return provider.get_quotes(symbol, start_dt, end_dt)
        elif data_type == 'status':
            return provider.get_status(symbol, start_dt, end_dt)
        elif data_type.startswith('bars_'):
            timeframe = data_type.split('_')[1]
            return provider.get_bars(symbol, timeframe, start_dt, end_dt)
        else:
            self._log(f"Unknown data type: {data_type}", logging.WARNING)
            return pd.DataFrame()

    def get_bars(self, symbol: str = None, timeframe: str = "1m",
                 start_time: Union[datetime, str] = None,
                 end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """
        Get bars data for a specific timeframe.

        Args:
            symbol: Symbol to get data for. If None, uses current_symbol
            timeframe: Timeframe for bars (e.g., "1s", "1m", "5m", "1d")
            start_time: Start time filter. If None, returns all cached data
            end_time: End time filter. If None, returns all cached data

        Returns:
            DataFrame with bars data
        """
        # Use current symbol if not specified
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return pd.DataFrame()

        # Check if data is cached
        data_type = f"bars_{timeframe}"

        # Get range limits to check cache
        if start_time and end_time:
            start_dt = ensure_timezone_aware(start_time, is_end_time=False)
            end_dt = ensure_timezone_aware(end_time, is_end_time=True)

            # Check memory cache
            cached_data = self._check_memory_cache(symbol, data_type, start_dt, end_dt)
            if cached_data is not None:
                return cached_data

        # If we get here, need to load the data
        if self.is_live:
            self._log(f"No {timeframe} data in cache and cannot load historical data with live provider",
                      logging.WARNING)
            return pd.DataFrame()

        # Try to load if we have a historical provider
        if isinstance(self.provider, HistoricalDataProvider):
            # For backwards compatibility, wrap in a call to load_data
            if start_time is None or end_time is None:
                self._log(f"Must specify start_time and end_time for uncached data", logging.WARNING)
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, [data_type])
            return result.get(symbol, {}).get(data_type, pd.DataFrame())
        else:
            self._log(f"No {timeframe} data in cache for {symbol}", logging.WARNING)
            return pd.DataFrame()

    def get_trades(self, symbol: str = None,
                   start_time: Union[datetime, str] = None,
                   end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """
        Get trades data.

        Args:
            symbol: Symbol to get data for. If None, uses current_symbol
            start_time: Start time filter. If None, returns all cached data
            end_time: End time filter. If None, returns all cached data

        Returns:
            DataFrame with trades data
        """
        # Use current symbol if not specified
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return pd.DataFrame()

        # Similar structure to get_bars - check cache, then load if needed
        if start_time and end_time:
            start_dt = ensure_timezone_aware(start_time, is_end_time=False)
            end_dt = ensure_timezone_aware(end_time, is_end_time=True)

            # Check memory cache
            cached_data = self._check_memory_cache(symbol, 'trades', start_dt, end_dt)
            if cached_data is not None:
                return cached_data

        # If we get here, need to load the data
        if self.is_live:
            self._log(f"No trades data in cache and cannot load historical data with live provider",
                      logging.WARNING)
            return pd.DataFrame()

        # Try to load if we have a historical provider
        if isinstance(self.provider, HistoricalDataProvider):
            if start_time is None or end_time is None:
                self._log(f"Must specify start_time and end_time for uncached data", logging.WARNING)
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, ['trades'])
            return result.get(symbol, {}).get('trades', pd.DataFrame())
        else:
            self._log(f"No trades data in cache for {symbol}", logging.WARNING)
            return pd.DataFrame()

    def get_quotes(self, symbol: str = None,
                   start_time: Union[datetime, str] = None,
                   end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """
        Get quotes data.

        Args:
            symbol: Symbol to get data for. If None, uses current_symbol
            start_time: Start time filter. If None, returns all cached data
            end_time: End time filter. If None, returns all cached data

        Returns:
            DataFrame with quotes data
        """
        # Similar to get_trades but for quotes
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return pd.DataFrame()

        if start_time and end_time:
            start_dt = ensure_timezone_aware(start_time, is_end_time=False)
            end_dt = ensure_timezone_aware(end_time, is_end_time=True)

            cached_data = self._check_memory_cache(symbol, 'quotes', start_dt, end_dt)
            if cached_data is not None:
                return cached_data

        if self.is_live:
            self._log(f"No quotes data in cache and cannot load historical data with live provider",
                      logging.WARNING)
            return pd.DataFrame()

        if isinstance(self.provider, HistoricalDataProvider):
            if start_time is None or end_time is None:
                self._log(f"Must specify start_time and end_time for uncached data", logging.WARNING)
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, ['quotes'])
            return result.get(symbol, {}).get('quotes', pd.DataFrame())
        else:
            self._log(f"No quotes data in cache for {symbol}", logging.WARNING)
            return pd.DataFrame()

    def get_status(self, symbol: str = None,
                   start_time: Union[datetime, str] = None,
                   end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """
        Get status data.

        Args:
            symbol: Symbol to get data for. If None, uses current_symbol
            start_time: Start time filter. If None, returns all cached data
            end_time: End time filter. If None, returns all cached data

        Returns:
            DataFrame with status data
        """
        # Similar to get_trades but for status
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return pd.DataFrame()

        if start_time and end_time:
            start_dt = ensure_timezone_aware(start_time, is_end_time=False)
            end_dt = ensure_timezone_aware(end_time, is_end_time=True)

            cached_data = self._check_memory_cache(symbol, 'status', start_dt, end_dt)
            if cached_data is not None:
                return cached_data

        if self.is_live:
            self._log(f"No status data in cache and cannot load historical data with live provider",
                      logging.WARNING)
            return pd.DataFrame()

        if isinstance(self.provider, HistoricalDataProvider):
            if start_time is None or end_time is None:
                self._log(f"Must specify start_time and end_time for uncached data", logging.WARNING)
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, ['status'])
            return result.get(symbol, {}).get('status', pd.DataFrame())
        else:
            self._log(f"No status data in cache for {symbol}", logging.WARNING)
            return pd.DataFrame()

    def clear_cache(self, symbol: str = None):
        """
        Clear data from cache for a symbol or all symbols.

        Args:
            symbol: Symbol to clear data for. If None, clears all cache
        """
        if symbol:
            if symbol in self.data_cache:
                del self.data_cache[symbol]
                if symbol in self.loaded_ranges:
                    del self.loaded_ranges[symbol]
                self._log(f"Cleared cache for {symbol}")

                # Reset current symbol if it was cleared
                if self.current_symbol == symbol:
                    self.current_symbol = None
        else:
            self.data_cache = {}
            self.loaded_ranges = {}
            self.current_symbol = None
            self._log("Cleared all data cache")

    def initialize_live_data(self, symbol: str, timeframes: List[str] = None):
        """
        Initialize live data streaming for a symbol.

        Args:
            symbol: Symbol to stream data for
            timeframes: List of timeframes to subscribe to
        """
        if not self.is_live:
            self._log("Cannot initialize live data with historical provider", logging.WARNING)
            return

        if not isinstance(self.provider, LiveDataProvider):
            self._log("Provider must be a LiveDataProvider for initialize_live_data", logging.ERROR)
            return

        self.current_symbol = symbol

        # Clear any existing cache for this symbol
        if symbol in self.data_cache:
            del self.data_cache[symbol]

        # Default timeframes if not specified
        if timeframes is None:
            timeframes = ["1s", "1m", "5m", "1d"]

        # Filter to supported timeframes
        timeframes = [tf for tf in timeframes if tf in ["1s", "1m", "5m", "1d"]]

        # Map timeframes to data types for subscription
        data_types = ["trades", "quotes", "status"]
        for tf in timeframes:
            data_types.append(f"bars_{tf}")

        # Subscribe to the data
        self.provider.subscribe([symbol], data_types)
        self._log(f"Initialized live data for {symbol} with timeframes {timeframes}")

    def close(self):
        """Close the data manager and release resources."""
        if self.is_live and isinstance(self.provider, LiveDataProvider):
            self.provider.close()

        self.clear_cache()
        self._log("Data manager closed")