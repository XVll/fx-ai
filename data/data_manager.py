# data/data_manager.py - FIXED: Clean, efficient logging with data loading stats
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import logging

from data.provider.data_provider import DataProvider, HistoricalDataProvider, LiveDataProvider
from data.utils.helpers import ensure_timezone_aware


class DataManager:
    """
    Centralized data management with clean, informative logging.
    """

    def __init__(self, provider: DataProvider, logger=None):
        """
        Initialize the data manager.

        Args:
            provider: DataProvider instance (historical or live)
            logger: Optional logger
        """
        self.provider:HistoricalDataProvider = provider
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

        # Session stats for better logging
        self.session_stats = {
            'total_rows_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'data_types_loaded': set(),
            'symbols_loaded': set()
        }

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
            self.session_stats['cache_misses'] += 1
            return None

        # First check if any existing cache key exactly matches our request
        cache_key = self._create_cache_key(start_time, end_time)
        if cache_key in self.data_cache[symbol][data_type]:
            self.session_stats['cache_hits'] += 1
            return self.data_cache[symbol][data_type][cache_key]

        # If no exact match, check if we have a superset of the data
        for key, df in self.data_cache[symbol][data_type].items():
            if not df.empty:
                try:
                    df_start = df.index.min()
                    df_end = df.index.max()

                    # If the cached data fully contains our request
                    if df_start <= start_time and df_end >= end_time:
                        # Extract the slice we need and cache it for next time
                        result = df[start_time:end_time]
                        if not result.empty:
                            self._save_to_memory_cache(symbol, data_type, start_time, end_time, result)
                            self.session_stats['cache_hits'] += 1
                            return result
                except Exception as e:
                    # Silent fail for cache checking
                    pass

        self.session_stats['cache_misses'] += 1
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
        Load data for symbols in a date range with comprehensive logging.

        Args:
            symbols: Symbol or list of symbols to load data for
            start_time: Start time for data
            end_time: End time for data
            data_types: List of data types to load. If None, loads all available types

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

        # Initialize session tracking
        load_start_time = datetime.now()
        total_rows_this_load = 0
        successful_loads = {}
        failed_loads = {}

        # Load data for each symbol
        results = {}
        for symbol in symbols:
            self.current_symbol = symbol
            self.session_stats['symbols_loaded'].add(symbol)

            symbol_data = {}
            symbol_rows = 0

            # Process each data type
            for data_type in data_types:
                # Skip unsupported timeframes
                if data_type.startswith('bars_'):
                    timeframe = data_type.split('_')[1]
                    if timeframe not in ["1s", "1m", "5m", "1d"]:
                        continue

                try:
                    # Check memory cache
                    cached_data = self._check_memory_cache(symbol, data_type, start_dt, end_dt)
                    if cached_data is not None:
                        symbol_data[data_type] = cached_data
                        continue

                    # Load from provider
                    df = self._load_from_provider(symbol, data_type, start_dt, end_dt)

                    if df is not None and not df.empty:
                        # Save to memory cache
                        self._save_to_memory_cache(symbol, data_type, start_dt, end_dt, df)

                        # Update loaded ranges
                        self._update_loaded_ranges(symbol, data_type, start_dt, end_dt)

                        symbol_data[data_type] = df
                        rows_loaded = len(df)
                        symbol_rows += rows_loaded
                        total_rows_this_load += rows_loaded
                        self.session_stats['total_rows_loaded'] += rows_loaded
                        self.session_stats['data_types_loaded'].add(data_type)

                        successful_loads[f"{symbol}_{data_type}"] = rows_loaded

                except Exception as e:
                    failed_loads[f"{symbol}_{data_type}"] = str(e)

            results[symbol] = symbol_data

        # Comprehensive load summary
        load_duration = (datetime.now() - load_start_time).total_seconds()

        if successful_loads or failed_loads:
            self._log(f"📈 Data Load Summary:")
            self._log(f"   ⏱️  Duration: {load_duration:.2f}s")
            self._log(f"   📊 Total rows: {total_rows_this_load:,}")
            self._log(f"   ✅ Successful: {len(successful_loads)} data types")

            if successful_loads:
                top_loads = sorted(successful_loads.items(), key=lambda x: x[1], reverse=True)[:3]
                for data_type, rows in top_loads:
                    self._log(f"      • {data_type}: {rows:,} rows")

            if failed_loads:
                self._log(f"   ❌ Failed: {len(failed_loads)} data types", logging.WARNING)
                for data_type, error in list(failed_loads.items())[:2]:  # Show only first 2 errors
                    self._log(f"      • {data_type}: {error}", logging.WARNING)

        return results

    def _load_from_provider(self, symbol: str, data_type: str,
                            start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Load data from the provider based on data type."""
        provider:HistoricalDataProvider = self.provider

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
            return pd.DataFrame()

        # Try to load if we have a historical provider
        if isinstance(self.provider, HistoricalDataProvider):
            # For backwards compatibility, wrap in a call to load_data
            if start_time is None or end_time is None:
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, [data_type])
            return result.get(symbol, {}).get(data_type, pd.DataFrame())
        else:
            return pd.DataFrame()

    def get_trades(self, symbol: str = None,
                   start_time: Union[datetime, str] = None,
                   end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """Get trades data."""
        symbol = symbol or self.current_symbol

        if not symbol:
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
            return pd.DataFrame()

        # Try to load if we have a historical provider
        if isinstance(self.provider, HistoricalDataProvider):
            if start_time is None or end_time is None:
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, ['trades'])
            return result.get(symbol, {}).get('trades', pd.DataFrame())
        else:
            return pd.DataFrame()

    def get_quotes(self, symbol: str = None,
                   start_time: Union[datetime, str] = None,
                   end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """Get quotes data."""
        symbol = symbol or self.current_symbol

        if not symbol:
            return pd.DataFrame()

        if start_time and end_time:
            start_dt = ensure_timezone_aware(start_time, is_end_time=False)
            end_dt = ensure_timezone_aware(end_time, is_end_time=True)

            cached_data = self._check_memory_cache(symbol, 'quotes', start_dt, end_dt)
            if cached_data is not None:
                return cached_data

        if self.is_live:
            return pd.DataFrame()

        if isinstance(self.provider, HistoricalDataProvider):
            if start_time is None or end_time is None:
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, ['quotes'])
            return result.get(symbol, {}).get('quotes', pd.DataFrame())
        else:
            return pd.DataFrame()

    def get_status(self, symbol: str = None,
                   start_time: Union[datetime, str] = None,
                   end_time: Union[datetime, str] = None) -> pd.DataFrame:
        """Get status data."""
        symbol = symbol or self.current_symbol

        if not symbol:
            return pd.DataFrame()

        if start_time and end_time:
            start_dt = ensure_timezone_aware(start_time, is_end_time=False)
            end_dt = ensure_timezone_aware(end_time, is_end_time=True)

            cached_data = self._check_memory_cache(symbol, 'status', start_dt, end_dt)
            if cached_data is not None:
                return cached_data

        if self.is_live:
            return pd.DataFrame()

        if isinstance(self.provider, HistoricalDataProvider):
            if start_time is None or end_time is None:
                return pd.DataFrame()

            result = self.load_data([symbol], start_time, end_time, ['status'])
            return result.get(symbol, {}).get('status', pd.DataFrame())
        else:
            return pd.DataFrame()

    def clear_cache(self, symbol: str = None):
        """Clear data from cache for a symbol or all symbols."""
        if symbol:
            if symbol in self.data_cache:
                del self.data_cache[symbol]
                if symbol in self.loaded_ranges:
                    del self.loaded_ranges[symbol]

                # Reset current symbol if it was cleared
                if self.current_symbol == symbol:
                    self.current_symbol = None
        else:
            self.data_cache = {}
            self.loaded_ranges = {}
            self.current_symbol = None

    def initialize_live_data(self, symbol: str, timeframes: List[str] = None):
        """Initialize live data streaming for a symbol."""
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
        self._log(f"🔴 Live data initialized for {symbol} with timeframes {timeframes}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            **self.session_stats,
            'cache_hit_rate': self.session_stats['cache_hits'] / max(1, self.session_stats['cache_hits'] + self.session_stats['cache_misses']) * 100,
            'cached_symbols': len(self.data_cache),
            'data_types_loaded': list(self.session_stats['data_types_loaded']),
            'symbols_loaded': list(self.session_stats['symbols_loaded'])
        }

    def close(self):
        """Close the data manager and release resources."""
        stats = self.get_session_stats()

        self._log(f"📊 Data Manager Session Summary:")
        self._log(f"   📈 Total rows loaded: {stats['total_rows_loaded']:,}")
        self._log(f"   🎯 Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        self._log(f"   📂 Symbols: {len(stats['symbols_loaded'])}")
        self._log(f"   📋 Data types: {len(stats['data_types_loaded'])}")

        self.clear_cache()