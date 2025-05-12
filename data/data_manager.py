# data/data_manager.py
from typing import Dict, List, Union, Tuple, Optional
import pandas as pd
from datetime import datetime
import logging
from .provider.data_provider import DataProvider, HistoricalDataProvider, LiveDataProvider
from .utils.helpers import ensure_timezone_aware


class DataManager:
    """
    Centralized data management class responsible for:
    - Loading and caching data
    - Providing access to different data types
    - Managing data lifecycle (loading, unloading)

    This class serves as the single point of access for all data needs
    in the AI trading system.
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
        # {symbol: {data_type: DataFrame}}
        self.data_cache = {}

        # Track loaded data ranges
        # {symbol: (start_date, end_date)}
        self.loaded_ranges = {}

        # Current state tracking
        self.current_symbol = None
        self.is_live = isinstance(provider, LiveDataProvider)

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def load_data(self, symbol: str,
                  start_time: Union[datetime, str],
                  end_time: Union[datetime, str],
                  data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for a symbol in a date range.

        Args:
            symbol: Symbol to load data for
            start_time: Start time for data
            end_time: End time for data
            data_types: List of data types to load. If None, loads all available types
                        Supported types: 'bars_1s', 'bars_1m', 'bars_5m', 'bars_1d',
                                        'trades', 'quotes', 'status'

        Returns:
            Dictionary mapping data types to DataFrames
        """
        if self.is_live:
            self._log(f"Cannot load historical data with live provider for {symbol}", logging.WARNING)
            return {}

        if not isinstance(self.provider, HistoricalDataProvider):
            self._log(f"Provider must be a HistoricalDataProvider for load_data", logging.ERROR)
            return {}

        self.current_symbol = symbol

        # Default data types if not specified
        if data_types is None:
            data_types = ["bars_1s", "bars_1m", "bars_5m", "bars_1d", "trades", "quotes", "status"]

        # Initialize cache for this symbol if not exists
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {}

        # Update loaded range for this symbol
        if start_time and end_time:
            start_utc = ensure_timezone_aware(start_time, is_end_time=False)
            end_utc = ensure_timezone_aware(end_time, is_end_time=True)

            self.loaded_ranges[symbol] = (start_utc, end_utc)

        # Load data for each requested type
        loaded_data = {}

        try:
            for data_type in data_types:
                self._log(f"Loading {data_type} for {symbol} from {start_time} to {end_time}")

                if data_type == 'trades':
                    df = self.provider.get_trades(symbol, start_time, end_time)
                elif data_type == 'quotes':
                    df = self.provider.get_quotes(symbol, start_time, end_time)
                elif data_type == 'status':
                    df = self.provider.get_status(symbol, start_time, end_time)
                elif data_type.startswith('bars_'):
                    timeframe = data_type.split('_')[1]
                    df = self.provider.get_bars(symbol, timeframe, start_time, end_time)
                else:
                    self._log(f"Unknown data type: {data_type}", logging.WARNING)
                    continue

                if not df.empty:
                    # Cache the data
                    self.data_cache[symbol][data_type] = df
                    loaded_data[data_type] = df
                else:
                    self._log(f"No data found for {symbol} {data_type}", logging.WARNING)

            self._log(f"Loaded data for {symbol}: {', '.join(loaded_data.keys())}")
            return loaded_data

        except Exception as e:
            self._log(f"Error loading data for {symbol}: {e}", logging.ERROR)
            return {}

    def get_data(self, symbol: str = None, data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get cached data for a symbol.

        Args:
            symbol: Symbol to get data for. If None, uses current_symbol
            data_types: List of data types to get. If None, gets all available

        Returns:
            Dictionary mapping data types to DataFrames
        """
        # Use current symbol if not specified
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return {}

        if symbol not in self.data_cache:
            self._log(f"No data cached for {symbol}", logging.WARNING)
            return {}

        # Get all cached data types if not specified
        if data_types is None:
            return self.data_cache[symbol]

        # Get only requested data types
        return {dt: self.data_cache[symbol][dt]
                for dt in data_types
                if dt in self.data_cache[symbol]}

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
        if symbol not in self.data_cache or data_type not in self.data_cache[symbol]:
            # Try to load if we have a historical provider
            if not self.is_live and isinstance(self.provider, HistoricalDataProvider):
                if symbol in self.loaded_ranges:
                    start_range, end_range = self.loaded_ranges[symbol]
                    df = self.provider.get_bars(symbol, timeframe, start_range, end_range)
                    if not df.empty:
                        # Cache the data
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = {}
                        self.data_cache[symbol][data_type] = df
                    else:
                        self._log(f"No {timeframe} bars data available for {symbol}", logging.WARNING)
                        return pd.DataFrame()
                else:
                    self._log(f"No date range info for {symbol}", logging.WARNING)
                    return pd.DataFrame()
            else:
                self._log(f"No {timeframe} bars data available for {symbol}", logging.WARNING)
                return pd.DataFrame()

        df = self.data_cache[symbol][data_type]

        # Apply time filters if specified
        if start_time:
            start_utc = ensure_timezone_aware(start_time, is_end_time=False)
            df = df[df.index >= start_utc]

        if end_time:
            end_utc = ensure_timezone_aware(end_time, is_end_time=True)
            df = df[df.index <= end_utc]

        return df

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

        # Check if data is cached
        if symbol not in self.data_cache or 'trades' not in self.data_cache[symbol]:
            # Try to load if we have a historical provider
            if not self.is_live and isinstance(self.provider, HistoricalDataProvider):
                if symbol in self.loaded_ranges:
                    start_range, end_range = self.loaded_ranges[symbol]
                    df = self.provider.get_trades(symbol, start_range, end_range)
                    if not df.empty:
                        # Cache the data
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = {}
                        self.data_cache[symbol]['trades'] = df
                    else:
                        self._log(f"No trades data available for {symbol}", logging.WARNING)
                        return pd.DataFrame()
                else:
                    self._log(f"No date range info for {symbol}", logging.WARNING)
                    return pd.DataFrame()
            else:
                self._log(f"No trades data available for {symbol}", logging.WARNING)
                return pd.DataFrame()

        df = self.data_cache[symbol]['trades']

        # Apply time filters if specified
        if start_time:
            start_utc = ensure_timezone_aware(start_time, is_end_time=False)
            df = df[df.index >= start_utc]

        if end_time:
            end_utc = ensure_timezone_aware(end_time, is_end_time=True)
            df = df[df.index <= end_utc]

        return df

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
        # Use current symbol if not specified
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return pd.DataFrame()

        # Check if data is cached
        if symbol not in self.data_cache or 'quotes' not in self.data_cache[symbol]:
            # Try to load if we have a historical provider
            if not self.is_live and isinstance(self.provider, HistoricalDataProvider):
                if symbol in self.loaded_ranges:
                    start_range, end_range = self.loaded_ranges[symbol]
                    df = self.provider.get_quotes(symbol, start_range, end_range)
                    if not df.empty:
                        # Cache the data
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = {}
                        self.data_cache[symbol]['quotes'] = df
                    else:
                        self._log(f"No quotes data available for {symbol}", logging.WARNING)
                        return pd.DataFrame()
                else:
                    self._log(f"No date range info for {symbol}", logging.WARNING)
                    return pd.DataFrame()
            else:
                self._log(f"No quotes data available for {symbol}", logging.WARNING)
                return pd.DataFrame()

        df = self.data_cache[symbol]['quotes']

        # Apply time filters if specified
        if start_time:
            start_utc = ensure_timezone_aware(start_time, is_end_time=False)
            df = df[df.index >= start_utc]

        if end_time:
            end_utc = ensure_timezone_aware(end_time, is_end_time=True)
            df = df[df.index <= end_utc]

        return df

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
        # Use current symbol if not specified
        symbol = symbol or self.current_symbol

        if not symbol:
            self._log("No symbol specified and no current symbol set", logging.WARNING)
            return pd.DataFrame()

        # Check if data is cached
        if symbol not in self.data_cache or 'status' not in self.data_cache[symbol]:
            # Try to load if we have a historical provider
            if not self.is_live and isinstance(self.provider, HistoricalDataProvider):
                if symbol in self.loaded_ranges:
                    start_range, end_range = self.loaded_ranges[symbol]
                    df = self.provider.get_status(symbol, start_range, end_range)
                    if not df.empty:
                        # Cache the data
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = {}
                        self.data_cache[symbol]['status'] = df
                    else:
                        self._log(f"No status data available for {symbol}", logging.WARNING)
                        return pd.DataFrame()
                else:
                    self._log(f"No date range info for {symbol}", logging.WARNING)
                    return pd.DataFrame()
            else:
                self._log(f"No status data available for {symbol}", logging.WARNING)
                return pd.DataFrame()

        df = self.data_cache[symbol]['status']

        # Apply time filters if specified
        if start_time:
            start_utc = ensure_timezone_aware(start_time, is_end_time=False)
            df = df[df.index >= start_utc]

        if end_time:
            end_utc = ensure_timezone_aware(end_time, is_end_time=True)
            df = df[df.index <= end_utc]

        return df

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