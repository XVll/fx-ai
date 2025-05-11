# data/provider/data_bento/databento_file_provider.py
import databento as db
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import logging

from data.provider.data_provider import HistoricalDataProvider


class DabentoFileProvider(HistoricalDataProvider):
    """Implementation of Historical Provider using Databento file storage."""

    def __init__(self, data_dir: str, symbol_info_file: str = None, verbose: bool = False):
        """
        Initialize the Databento file provider.

        Args:
            data_dir: Directory containing Databento data files
            symbol_info_file: Optional path to a file with symbol metadata
            verbose: Enable verbose logging
        """
        self.data_dir = data_dir
        self._symbol_info = {}
        self.file_paths = []
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Load symbol info if provided
        if symbol_info_file and os.path.exists(symbol_info_file):
            self._symbol_info = pd.read_csv(symbol_info_file).set_index('symbol').to_dict('index')

        # Scan for files
        self._scan_files()

    def _log(self, msg: str, level: int = logging.INFO):
        """Log messages based on verbose setting."""
        if self.verbose or level >= logging.WARNING:
            self.logger.log(level, msg)

    def _scan_files(self):
        """Scan for all .dbn.zst files in directory and subdirectories"""
        self._log(f"Scanning for Databento files in {self.data_dir}")

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.dbn.zst') or file.endswith('.dbn'):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)

        self.logger.info(f"Found {len(self.file_paths)} DBN files")

        # Print sample files only if verbose
        if self.verbose and self.file_paths:
            samples = [os.path.basename(f) for f in self.file_paths[:5]]
            self._log(f"Sample files: {', '.join(samples)}" +
                      (f" and {len(self.file_paths) - 5} more" if len(self.file_paths) > 5 else ""))

    def _find_dataset_files_for_date(self, schema: str, date_str: str) -> List[str]:
        """Find files for a given schema and date."""
        # Find files for this schema and date
        matching_files = [f for f in self.file_paths if
                          schema.lower() in os.path.basename(f).lower() and
                          date_str in os.path.basename(f)]

        if not matching_files:
            # Try to find files that might contain this date range
            matching_files = [f for f in self.file_paths if
                              schema.lower() in os.path.basename(f).lower()]

            # For OHLCV-1d specifically, look for ranges that might include our date
            if schema == 'ohlcv-1d':
                range_files = []
                for f in matching_files:
                    basename = os.path.basename(f)
                    parts = basename.split('.')
                    if len(parts) > 1:
                        name_parts = parts[0].split('-')
                        if len(name_parts) >= 3:  # Might be a date range
                            try:
                                # Check if we can parse start and end dates
                                for part in name_parts:
                                    if len(part) == 8 and part.isdigit():
                                        range_files.append(f)
                                        break
                            except:
                                pass

                if range_files:
                    return range_files

        return matching_files

    def _ensure_timezone_aware(self, dt: Union[datetime, str], is_end_time: bool = False) -> pd.Timestamp:
        """
        Ensure datetime is timezone-aware, converting to UTC if needed.

        Args:
            dt: datetime or string to convert
            is_end_time: If True and input is a date without time, will set time to 23:59:59

        Returns:
            Timezone-aware pandas Timestamp
        """
        # Handle date strings by expanding to full days
        if isinstance(dt, str):
            if len(dt) <= 10 and '-' in dt:  # YYYY-MM-DD format
                if is_end_time:
                    dt = f"{dt} 23:59:59"  # End of day for end timestamps
                else:
                    dt = f"{dt} 00:00:00"  # Start of day for start timestamps
            dt = pd.Timestamp(dt)

        # If datetime is naive, make it timezone aware (UTC)
        if dt.tzinfo is None:
            dt = pd.Timestamp(dt).tz_localize('UTC')
        elif dt.tzinfo.tzname(dt) != 'UTC':
            dt = dt.astimezone('UTC')

        return dt

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        if symbol in self._symbol_info:
            return self._symbol_info[symbol]
        else:
            # Return minimal info
            return {"symbol": symbol, "description": f"Unknown symbol {symbol}"}

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        # This finds all unique symbols in the dataset
        symbols = set()
        checked_files = 0

        # Only check the first few files to avoid scanning everything
        for file_path in self.file_paths[:10]:  # Limit to first 10 files
            try:
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()
                if not df.empty and 'symbol' in df.columns:
                    unique_symbols = df['symbol'].unique()
                    for symbol in unique_symbols:
                        symbols.add(symbol)
                checked_files += 1
            except:
                pass

        if not symbols:
            # Fallback to dataset names from filenames
            for file in self.file_paths:
                basename = os.path.basename(file)
                parts = basename.split('.')[0].split('_')
                if parts:
                    symbols.add(parts[0])

        return list(symbols)

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical trades for a symbol in a time range."""
        self._log(f"Loading trades for {symbol} between {start_time} and {end_time}")

        # Make sure dates are timezone-aware datetime objects
        start_time = self._ensure_timezone_aware(start_time, is_end_time=False)
        end_time = self._ensure_timezone_aware(end_time, is_end_time=True)

        # Format date for search
        date_str = start_time.strftime('%Y%m%d')

        # Find trade files for this date
        trade_files = self._find_dataset_files_for_date('trades', date_str)

        if not trade_files:
            self._log(f"No trade files found for {symbol} on {date_str}", logging.WARNING)
            return pd.DataFrame()

        # Extract trades from each file
        all_trades = []

        for file_path in trade_files:
            basename = os.path.basename(file_path)
            self._log(f"Processing {basename}", logging.DEBUG)

            try:
                # Read the file
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    continue

                # Try to filter by symbol if symbol column exists
                if 'symbol' in df.columns:
                    symbol_df = df[df['symbol'] == symbol]
                    if symbol_df.empty:
                        continue
                    df = symbol_df

                # Ensure index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                # Filter for the time range
                mask = (df.index >= start_time) & (df.index <= end_time)
                filtered_df = df[mask]

                if not filtered_df.empty:
                    all_trades.append(filtered_df)

            except Exception as e:
                self._log(f"Error processing {basename}: {str(e)}", logging.WARNING)

        # Combine all trades
        if not all_trades:
            self._log(f"No trades found for {symbol} in the specified date range", logging.WARNING)
            return pd.DataFrame()

        trades_df = pd.concat(all_trades)
        trades_df = trades_df.sort_index()

        self.logger.info(f"Loaded {len(trades_df)} trades for {symbol}")
        return trades_df

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical quotes for a symbol in a time range."""
        self._log(f"Loading quotes for {symbol} between {start_time} and {end_time}")

        # Make sure dates are timezone-aware datetime objects
        start_time = self._ensure_timezone_aware(start_time, is_end_time=False)
        end_time = self._ensure_timezone_aware(end_time, is_end_time=True)

        # Format date for search
        date_str = start_time.strftime('%Y%m%d')

        # Find quote files for this date (mbp-1)
        quote_files = self._find_dataset_files_for_date('mbp-1', date_str)

        if not quote_files:
            self._log(f"No quote files found for {symbol} on {date_str}", logging.WARNING)
            return pd.DataFrame()

        # Extract quotes from each file
        all_quotes = []

        for file_path in quote_files:
            basename = os.path.basename(file_path)
            self._log(f"Processing {basename}", logging.DEBUG)

            try:
                # Read the file
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    continue

                # Try to filter by symbol if symbol column exists
                if 'symbol' in df.columns:
                    symbol_df = df[df['symbol'] == symbol]
                    if symbol_df.empty:
                        continue
                    df = symbol_df

                # Ensure index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                # Filter for the time range
                mask = (df.index >= start_time) & (df.index <= end_time)
                filtered_df = df[mask]

                if not filtered_df.empty:
                    all_quotes.append(filtered_df)

            except Exception as e:
                self._log(f"Error processing {basename}: {str(e)}", logging.WARNING)

        # Combine all quotes
        if not all_quotes:
            self._log(f"No quotes found for {symbol} in the specified date range", logging.WARNING)
            return pd.DataFrame()

        quotes_df = pd.concat(all_quotes)
        quotes_df = quotes_df.sort_index()

        self.logger.info(f"Loaded {len(quotes_df)} quotes for {symbol}")
        return quotes_df

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get OHLCV bars for a symbol, timeframe in a time range."""
        self._log(f"Loading {timeframe} bars for {symbol} between {start_time} and {end_time}")

        # Make sure dates are timezone-aware datetime objects
        start_time = self._ensure_timezone_aware(start_time, is_end_time=False)
        end_time = self._ensure_timezone_aware(end_time, is_end_time=True)

        # Format date for search
        date_str = start_time.strftime('%Y%m%d')

        # Map the timeframe string to Databento's schema format
        timeframe_map = {
            "1s": "ohlcv-1s",
            "1m": "ohlcv-1m",
            "5m": "ohlcv-5m",
            "1d": "ohlcv-1d"
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_map.keys())}")

        schema = timeframe_map[timeframe]

        # Find bar files for this date
        bar_files = self._find_dataset_files_for_date(schema, date_str)

        # Special case for 1d timeframe - check for files that contain date ranges
        if timeframe == "1d" and not bar_files:
            # Try to find daily ohlc files that might contain multiple dates
            bar_files = [f for f in self.file_paths if
                         'ohlcv-1d' in os.path.basename(f).lower()]

        if not bar_files:
            # Special case for 5m timeframe which might not exist directly
            if timeframe == "5m":
                return self._create_5m_bars_from_1m(symbol, start_time, end_time)

            self._log(f"No {timeframe} bar files found for {symbol} on {date_str}", logging.WARNING)
            return pd.DataFrame()

        # Extract bars from each file
        all_bars = []

        for file_path in bar_files:
            basename = os.path.basename(file_path)
            self._log(f"Processing {basename}", logging.DEBUG)

            try:
                # Read the file
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    continue

                # Try to filter by symbol if symbol column exists
                if 'symbol' in df.columns:
                    symbol_df = df[df['symbol'] == symbol]
                    if symbol_df.empty:
                        continue
                    df = symbol_df

                # Ensure index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                # Filter for the time range
                mask = (df.index >= start_time) & (df.index <= end_time)
                filtered_df = df[mask]

                if not filtered_df.empty:
                    all_bars.append(filtered_df)

            except Exception as e:
                self._log(f"Error processing {basename}: {str(e)}", logging.WARNING)

        # Combine all bars
        if not all_bars:
            # If timeframe is 5m and we couldn't find direct data, try to create from 1m
            if timeframe == "5m":
                self._log("Trying to create 5m bars from 1m data", logging.INFO)
                return self._create_5m_bars_from_1m(symbol, start_time, end_time)

            self._log(f"No {timeframe} bars found for {symbol} in the specified date range", logging.WARNING)
            return pd.DataFrame()

        bars_df = pd.concat(all_bars)
        bars_df = bars_df.sort_index()

        self.logger.info(f"Loaded {len(bars_df)} {timeframe} bars for {symbol}")
        return bars_df

    def _create_5m_bars_from_1m(self, symbol: str, start_time: Union[datetime, str],
                                end_time: Union[datetime, str]) -> pd.DataFrame:
        """Create 5-minute bars by resampling 1-minute data."""
        # Get 1-minute bars
        bars_1m = self.get_bars(symbol, "1m", start_time, end_time)

        if bars_1m.empty:
            self._log("No 1m data found to create 5m bars", logging.WARNING)
            return pd.DataFrame()

        # Resample to 5-minute bars
        resampled = bars_1m.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        self._log(f"Created {len(resampled)} 5m bars from 1m data")
        return resampled

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get status updates (halts, etc.) for a symbol in a time range."""
        self._log(f"Loading status updates for {symbol} between {start_time} and {end_time}")

        # Make sure dates are timezone-aware datetime objects
        start_time = self._ensure_timezone_aware(start_time, is_end_time=False)
        end_time = self._ensure_timezone_aware(end_time, is_end_time=True)

        # Format date for search
        date_str = start_time.strftime('%Y%m%d')

        # Find status files for this date
        status_files = self._find_dataset_files_for_date('status', date_str)

        if not status_files:
            self._log(f"No status files found for {symbol} on {date_str}", logging.INFO)
            return pd.DataFrame()

        # Extract status updates from each file
        all_status = []

        for file_path in status_files:
            basename = os.path.basename(file_path)
            self._log(f"Processing {basename}", logging.DEBUG)

            try:
                # Read the file
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    continue

                # Try to filter by symbol if symbol column exists
                if 'symbol' in df.columns:
                    symbol_df = df[df['symbol'] == symbol]
                    if symbol_df.empty:
                        continue
                    df = symbol_df

                # Ensure index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                # Filter for the time range
                mask = (df.index >= start_time) & (df.index <= end_time)
                filtered_df = df[mask]

                if not filtered_df.empty:
                    all_status.append(filtered_df)

            except Exception as e:
                self._log(f"Error processing {basename}: {str(e)}", logging.WARNING)

        # Combine all status updates
        if not all_status:
            self._log(f"No status updates found for {symbol} in the specified date range", logging.INFO)
            return pd.DataFrame()

        status_df = pd.concat(all_status)
        status_df = status_df.sort_index()

        self.logger.info(f"Loaded {len(status_df)} status updates for {symbol}")
        return status_df