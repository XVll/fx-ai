# data/data_manager.py - Enhanced with 2-tier caching and momentum index support
from typing import Dict, List, Union, Tuple, Optional, Any, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from data.provider.data_provider import DataProvider, HistoricalDataProvider, LiveDataProvider
from data.utils.helpers import ensure_timezone_aware
from data.scanner.momentum_scanner import MomentumScanner


class DataManager:
    """
    Enhanced data management with 2-tier caching and momentum index support.
    
    Features:
    - L1 Cache: Active day data in memory
    - L2 Cache: Pre-loaded next 2-3 days
    - Momentum index integration for smart day selection
    - Efficient single-day episode support
    """

    def __init__(self, provider: DataProvider, 
                 momentum_scanner: Optional[MomentumScanner] = None,
                 preload_days: int = 2,
                 logger=None):
        """
        Initialize the enhanced data manager.

        Args:
            provider: DataProvider instance (historical or live)
            momentum_scanner: Optional momentum scanner for index-based loading
            preload_days: Number of days to keep in L2 cache
            logger: Optional logger
        """
        self.provider: HistoricalDataProvider = provider
        self.momentum_scanner = momentum_scanner
        self.preload_days = preload_days
        self.logger = logger or logging.getLogger(__name__)

        # Two-tier cache structure
        # L1 - Active day cache (single day episode data)
        self.l1_cache = {
            'symbol': None,
            'date': None,
            'data': {},  # {data_type: DataFrame}
            'prev_day_data': {}  # Previous day for lookback
        }
        
        # L2 - Pre-load buffer (next 2-3 days)
        self.l2_cache = {}  # {(symbol, date): {'data': {}, 'prev_day_data': {}}}
        self.l2_queue = []  # Order of days in L2
        
        # Background pre-loading
        self.preload_executor = ThreadPoolExecutor(max_workers=1)
        self.preload_future = None
        self.preload_lock = threading.Lock()
        
        # Momentum index cache
        self.momentum_days_cache = None
        self.reset_points_cache = None
        self._load_momentum_indices()

        # Cache for legacy compatibility
        self.data_cache = {}
        self.loaded_ranges = {}
        
        # Current state tracking
        self.current_symbol = None
        self.current_date = None
        self.is_live = isinstance(provider, LiveDataProvider)

        # Session stats for better logging
        self.session_stats = {
            'total_rows_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'data_types_loaded': set(),
            'symbols_loaded': set(),
            'days_loaded': 0,
            'preload_count': 0
        }

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)
            
    def _load_momentum_indices(self):
        """Load momentum indices if scanner is available."""
        if self.momentum_scanner:
            try:
                self.momentum_days_cache, self.reset_points_cache = self.momentum_scanner.load_index()
                if not self.momentum_days_cache.empty:
                    self._log(f"Loaded momentum index with {len(self.momentum_days_cache)} days")
            except Exception as e:
                self._log(f"Error loading momentum indices: {e}", logging.WARNING)

    def load_day(self, symbol: str, date: datetime, 
                 data_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load a full day's data (4 AM - 8 PM ET) with previous day for lookback.
        
        This is the primary method for episode-based training where each episode
        operates within a single day with multiple reset points.
        
        Args:
            symbol: Symbol to load
            date: Date to load (will load full trading day)
            data_types: Data types to load (default: all)
            
        Returns:
            Dictionary of data types to DataFrames
        """
        # Check L1 cache first
        if (self.l1_cache['symbol'] == symbol and 
            self.l1_cache['date'] == pd.Timestamp(date).date()):
            self.session_stats['l1_hits'] += 1
            self._log(f"L1 cache hit for {symbol} on {date}")
            return self.l1_cache['data']
            
        # Check L2 cache
        cache_key = (symbol, pd.Timestamp(date).date())
        with self.preload_lock:
            if cache_key in self.l2_cache:
                self.session_stats['l2_hits'] += 1
                self._log(f"L2 cache hit for {symbol} on {date}")
                
                # Promote to L1
                cached = self.l2_cache.pop(cache_key)
                self.l2_queue.remove(cache_key)
                
                self.l1_cache['symbol'] = symbol
                self.l1_cache['date'] = cache_key[1]
                self.l1_cache['data'] = cached['data']
                self.l1_cache['prev_day_data'] = cached['prev_day_data']
                
                # Trigger background preload of next day
                self._trigger_preload(symbol, date)
                
                return self.l1_cache['data']
        
        # Cache miss - load from disk
        self.session_stats['cache_misses'] += 1
        self._log(f"Cache miss for {symbol} on {date}, loading from disk...")
        
        # Load the requested day and previous day
        day_data = self._load_full_day(symbol, date, data_types)
        prev_day_data = self._load_previous_day(symbol, date, data_types)
        
        # Update L1 cache
        self.l1_cache['symbol'] = symbol
        self.l1_cache['date'] = pd.Timestamp(date).date()
        self.l1_cache['data'] = day_data
        self.l1_cache['prev_day_data'] = prev_day_data
        
        self.session_stats['days_loaded'] += 1
        
        # Trigger background preload of next days
        self._trigger_preload(symbol, date)
        
        return day_data
        
    def _load_full_day(self, symbol: str, date: datetime,
                      data_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load full trading day data (4 AM - 8 PM ET)."""
        # Convert date to timezone-aware timestamps for full trading day
        date_obj = pd.Timestamp(date).date()
        
        # Create ET timezone timestamps for 4 AM - 8 PM
        start_et = pd.Timestamp(date_obj).tz_localize('America/New_York').replace(hour=4, minute=0, second=0)
        end_et = pd.Timestamp(date_obj).tz_localize('America/New_York').replace(hour=20, minute=0, second=0)
        
        # Convert to UTC for data loading
        start_utc = start_et.tz_convert('UTC')
        end_utc = end_et.tz_convert('UTC')
        
        # Use the existing load_data method
        result = self.load_data([symbol], start_utc, end_utc, data_types)
        return result.get(symbol, {})
        
    def _load_previous_day(self, symbol: str, date: datetime,
                          data_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load previous trading day for lookback calculations."""
        # Find previous trading day
        prev_date = self._get_previous_trading_day(date)
        if not prev_date:
            return {}
            
        return self._load_full_day(symbol, prev_date, data_types)
        
    def _get_previous_trading_day(self, date: datetime) -> Optional[datetime]:
        """Get the previous trading day, handling weekends and holidays."""
        date_obj = pd.Timestamp(date).date()
        
        # Simple logic - go back 1 day, skip weekends
        # In production, would check against market calendar
        prev = date_obj - timedelta(days=1)
        
        # Skip weekends
        while prev.weekday() >= 5:  # Saturday = 5, Sunday = 6
            prev -= timedelta(days=1)
            
        return prev
        
    def _trigger_preload(self, symbol: str, current_date: datetime):
        """Trigger background preloading of next days."""
        if self.preload_future and not self.preload_future.done():
            return  # Preload already in progress
            
        self.preload_future = self.preload_executor.submit(
            self._preload_next_days, symbol, current_date
        )
        
    def _preload_next_days(self, symbol: str, current_date: datetime):
        """Background task to preload next momentum days."""
        try:
            # Get next momentum days from index
            next_days = self._get_next_momentum_days(symbol, current_date, self.preload_days)
            
            for next_date in next_days:
                cache_key = (symbol, next_date.date())
                
                # Skip if already in cache
                with self.preload_lock:
                    if cache_key in self.l2_cache:
                        continue
                        
                # Load the day
                self._log(f"Pre-loading {symbol} for {next_date.date()}")
                day_data = self._load_full_day(symbol, next_date)
                prev_day_data = self._load_previous_day(symbol, next_date)
                
                # Add to L2 cache
                with self.preload_lock:
                    # Manage cache size
                    while len(self.l2_queue) >= self.preload_days:
                        oldest = self.l2_queue.pop(0)
                        self.l2_cache.pop(oldest, None)
                        
                    self.l2_cache[cache_key] = {
                        'data': day_data,
                        'prev_day_data': prev_day_data
                    }
                    self.l2_queue.append(cache_key)
                    
                self.session_stats['preload_count'] += 1
                
        except Exception as e:
            self._log(f"Error in preload task: {e}", logging.ERROR)
            
    def _get_next_momentum_days(self, symbol: str, current_date: datetime, 
                               count: int) -> List[pd.Timestamp]:
        """Get next momentum days from index or fallback to sequential days."""
        if self.momentum_days_cache is not None and not self.momentum_days_cache.empty:
            # Filter momentum days for symbol after current date
            current_date_obj = pd.Timestamp(current_date).date()
            symbol_days = self.momentum_days_cache[
                (self.momentum_days_cache['symbol'] == symbol.upper()) &
                (self.momentum_days_cache['date'].dt.date > current_date_obj)
            ].sort_values('date')
            
            if not symbol_days.empty:
                # Return up to 'count' days
                return symbol_days['date'].head(count).tolist()
                
        # Fallback to next sequential trading days
        next_days = []
        current = pd.Timestamp(current_date).date()
        
        while len(next_days) < count:
            current += timedelta(days=1)
            # Skip weekends
            if current.weekday() < 5:
                next_days.append(pd.Timestamp(current))
                
        return next_days
        
    def get_active_day_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Get data from the active day (L1 cache) for a specific data type."""
        if self.l1_cache['data'] and data_type in self.l1_cache['data']:
            return self.l1_cache['data'][data_type]
        return None
        
    def get_previous_day_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Get previous day data from L1 cache for lookback calculations."""
        if self.l1_cache['prev_day_data'] and data_type in self.l1_cache['prev_day_data']:
            return self.l1_cache['prev_day_data'][data_type]
        return None
        
    def get_momentum_days(self, symbol: str, min_activity: float = 0.0) -> pd.DataFrame:
        """Get available momentum days for a symbol from index."""
        if self.momentum_days_cache is None or self.momentum_days_cache.empty:
            return pd.DataFrame()
            
        if symbol is None:
            return pd.DataFrame()
            
        # Filter by symbol and activity score
        mask = (
            (self.momentum_days_cache['symbol'] == symbol.upper()) &
            (self.momentum_days_cache['activity_score'] >= min_activity)
        )
            
        return self.momentum_days_cache[mask].sort_values('activity_score', ascending=False)
    
    def get_all_momentum_days(self) -> List[Dict[str, Any]]:
        """Get all momentum days as a list of dictionaries.
        
        Returns:
            List of momentum day dictionaries with keys: symbol, date, quality_score, etc.
        """
        if self.momentum_days_cache is None or self.momentum_days_cache.empty:
            return []
            
        # Convert DataFrame to list of dicts
        momentum_days = []
        for _, row in self.momentum_days_cache.iterrows():
            momentum_days.append({
                'symbol': row['symbol'],
                'date': row['date'],
                'quality_score': row.get('activity_score', 0.0),  # Map activity_score to quality_score
                'max_intraday_move': row.get('max_intraday_move', 0.0),
                'volume_multiplier': row.get('volume_multiplier', 1.0),
                'metadata': row.to_dict()
            })
            
        return momentum_days
        
    def get_reset_points(self, symbol: str, date: datetime, 
                        min_roc: float = 0.0, min_activity: float = 0.0) -> pd.DataFrame:
        """Get reset points for a symbol on a specific date with 2-component filtering.
        
        Args:
            symbol: Symbol to filter for
            date: Date to filter for
            min_roc: Minimum absolute ROC score (directional momentum magnitude)
            min_activity: Minimum activity score [0.0, 1.0]
        """
        if self.reset_points_cache is None or self.reset_points_cache.empty:
            return pd.DataFrame()
            
        if symbol is None:
            return pd.DataFrame()
            
        date_obj = pd.Timestamp(date).date()
        mask = (
            (self.reset_points_cache['symbol'] == symbol.upper()) &
            (self.reset_points_cache['date'] == date_obj)
        )
        
        # Apply 3-component filters if available (use absolute ROC for directional momentum)
        if 'roc_score' in self.reset_points_cache.columns:
            mask &= self.reset_points_cache['roc_score'].abs() >= min_roc
        if 'activity_score' in self.reset_points_cache.columns:
            mask &= self.reset_points_cache['activity_score'] >= min_activity
            
        result = self.reset_points_cache[mask]
        
        # Sort by combined score if available, otherwise by absolute roc score
        if 'combined_score' in result.columns:
            return result.sort_values('combined_score', ascending=False)
        elif 'roc_score' in result.columns:
            return result.reindex(result['roc_score'].abs().sort_values(ascending=False).index)
        else:
            return result

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

    def _create_cache_key(self, start_time: datetime, end_time: datetime) -> str:
        """Create a cache key from start and end times."""
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if start_time else 'None'
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if end_time else 'None'
        return f"{start_str}_{end_str}"

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
            # Clear from L1 if it matches
            if self.l1_cache['symbol'] == symbol:
                self.l1_cache = {
                    'symbol': None,
                    'date': None,
                    'data': {},
                    'prev_day_data': {}
                }
                
            # Clear from L2
            with self.preload_lock:
                keys_to_remove = [k for k in self.l2_cache.keys() if k[0] == symbol]
                for key in keys_to_remove:
                    self.l2_cache.pop(key, None)
                    if key in self.l2_queue:
                        self.l2_queue.remove(key)
                        
            # Clear from old cache structure
            if symbol in self.data_cache:
                del self.data_cache[symbol]
            if symbol in self.loaded_ranges:
                del self.loaded_ranges[symbol]

            # Reset current symbol if it was cleared
            if self.current_symbol == symbol:
                self.current_symbol = None
        else:
            # Clear all caches
            self.l1_cache = {
                'symbol': None,
                'date': None,
                'data': {},
                'prev_day_data': {}
            }
            
            with self.preload_lock:
                self.l2_cache = {}
                self.l2_queue = []
                
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
        total_hits = self.session_stats['cache_hits'] + self.session_stats['l1_hits'] + self.session_stats['l2_hits']
        total_requests = total_hits + self.session_stats['cache_misses']
        
        return {
            **self.session_stats,
            'cache_hit_rate': total_hits / max(1, total_requests) * 100,
            'l1_hit_rate': self.session_stats['l1_hits'] / max(1, total_requests) * 100,
            'l2_hit_rate': self.session_stats['l2_hits'] / max(1, total_requests) * 100,
            'cached_symbols': len(set(k[0] for k in self.l2_cache.keys())) + (1 if self.l1_cache['symbol'] else 0),
            'l2_cache_size': len(self.l2_cache),
            'data_types_loaded': list(self.session_stats['data_types_loaded']),
            'symbols_loaded': list(self.session_stats['symbols_loaded'])
        }

    def get_day_data(self, symbol: str, date: datetime) -> Dict[str, pd.DataFrame]:
        """Get all data for a specific day.
        
        Returns:
            Dict with keys: 'ohlcv_1s', 'quotes', 'trades', 'status', 'mbp'
        """
        # Check if this is the active day
        if (self.l1_cache['symbol'] == symbol and 
            self.l1_cache['date'] and 
            self.l1_cache['date'].date() == date.date()):
            return self.l1_cache['data'].copy()
        
        # Check L2 cache
        for day_data in self.l2_cache:
            if (day_data['symbol'] == symbol and 
                day_data['date'] and 
                day_data['date'].date() == date.date()):
                return day_data['data'].copy()
        
        # Not in cache - need to load
        day_data = self.load_day(symbol, date)
        if day_data:
            return self.l1_cache['data'].copy()
        
        return {}
    
    def get_provider(self, symbol: str) -> DataProvider:
        """Get the data provider for a symbol."""
        return self.provider
    
    def preload_day_async(self, symbol: str, date: datetime, with_lookback: bool = False):
        """Asynchronously preload a day's data."""
        from concurrent.futures import Future
        future = Future()
        
        def _load():
            try:
                result = self.load_day(symbol, date, with_lookback)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        if self.preload_executor:
            self.preload_executor.submit(_load)
        else:
            # Synchronous fallback
            _load()
        
        return future

    def close(self):
        """Close the data manager and release resources."""
        # Shutdown preload executor
        if self.preload_executor:
            self.preload_executor.shutdown(wait=False)
            
        stats = self.get_session_stats()

        self._log(f"📊 Data Manager Session Summary:")
        self._log(f"   📈 Total rows loaded: {stats['total_rows_loaded']:,}")
        self._log(f"   🎯 Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        self._log(f"     • L1 hits: {stats['l1_hit_rate']:.1f}%")
        self._log(f"     • L2 hits: {stats['l2_hit_rate']:.1f}%")
        self._log(f"   📅 Days loaded: {stats['days_loaded']}")
        self._log(f"   🔄 Pre-loaded: {stats['preload_count']} days")
        self._log(f"   📂 Symbols: {len(stats['symbols_loaded'])}")
        self._log(f"   📋 Data types: {len(stats['data_types_loaded'])}")

        self.clear_cache()