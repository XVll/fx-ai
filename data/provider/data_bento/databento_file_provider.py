import databento as db
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import os
import logging
import json
from functools import lru_cache

from data.provider.data_provider import HistoricalDataProvider
from data.utils.helpers import ensure_timezone_aware
from data.utils.cleaning import clean_ohlc_data, clean_trades_data, clean_quotes_data


class DatabentoFileProvider(HistoricalDataProvider):
    """Implementation of HistoricalDataProvider using local Databento file storage."""

    _TIMEFRAME_TO_SCHEMA_MAP = {
        "1s": "ohlcv-1s",
        "1m": "ohlcv-1m",
        "5m": "ohlcv-5m",
        "1d": "ohlcv-1d",
        "trades": "trades",
        "quotes": "mbp-1",
        "status": "status"
    }

    def __init__(self, data_dir: str, symbol_info_file: Optional[str] = None,
                 verbose: bool = False, dbn_cache_size: int = 32):
        """
        Initialize the Databento file provider.

        Args:
            data_dir: Directory containing Databento job folders
            symbol_info_file: Optional path to a CSV file with symbol metadata
            verbose: Enable verbose logging for debugging
            dbn_cache_size: Max DBN file contents to keep in LRU cache
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Set log level based on verbosity
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Load symbol info if provided
        self._symbol_info_map = {}
        if symbol_info_file and os.path.exists(symbol_info_file):
            try:
                df_info = pd.read_csv(symbol_info_file)
                if 'symbol' in df_info.columns:
                    df_info['symbol'] = df_info['symbol'].astype(str).str.upper()
                    self._symbol_info_map = df_info.set_index('symbol').to_dict('index')
                    self.logger.info(f"Loaded symbol info for {len(self._symbol_info_map)} symbols")
                else:
                    self.logger.warning(f"'symbol' column not found in {symbol_info_file}")
            except Exception as e:
                self.logger.error(f"Error loading symbol info: {e}")

        # Scan directories for metadata and build an index
        self.metadata_index = []
        self._scan_and_build_metadata_index()

        # Create LRU cache for file reading
        @lru_cache(maxsize=dbn_cache_size)
        def _read_dbn_file_to_df_cached(file_path: str) -> pd.DataFrame:
            self.logger.debug(f"Reading DBN file: {file_path}")
            try:
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                return df
            except Exception as e:
                self.logger.error(f"Error reading DBN file: {e}")
                return pd.DataFrame()

        self._cached_read_dbn_file = _read_dbn_file_to_df_cached

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def _scan_and_build_metadata_index(self):
        """Scan the data directory and build an index of available files."""
        self.logger.info(f"Scanning for metadata in '{self.data_dir}'...")
        found_dirs = 0

        for root, dirs, files in os.walk(self.data_dir, topdown=True):
            if "metadata.json" in files and "manifest.json" in files:
                found_dirs += 1
                job_dir = root

                try:
                    # Read metadata.json
                    with open(os.path.join(job_dir, "metadata.json"), 'r') as f:
                        metadata = json.load(f)

                    # Read manifest.json
                    with open(os.path.join(job_dir, "manifest.json"), 'r') as f:
                        manifest = json.load(f)

                    # Extract query info
                    query_info = metadata.get("query", {})
                    schema = query_info.get("schema")
                    symbols = [str(s).upper() for s in query_info.get("symbols", [])]
                    start_ns = query_info.get("start")
                    end_ns = query_info.get("end")

                    if not all([schema, symbols, start_ns is not None, end_ns is not None]):
                        self.logger.warning(f"Missing essential fields in {job_dir}")
                        continue

                    # Convert timestamps
                    start_utc = pd.Timestamp(start_ns, unit='ns', tz='UTC')
                    end_utc = pd.Timestamp(end_ns, unit='ns', tz='UTC')

                    # Process each file in the manifest
                    for file_entry in manifest.get("files", []):
                        filename = file_entry.get("filename", "")
                        if filename.endswith((".dbn", ".dbn.zst")):
                            # Check if file matches schema
                            if schema.lower() in filename.lower():
                                file_path = os.path.join(job_dir, filename)
                                if os.path.exists(file_path):
                                    self.metadata_index.append({
                                        'file_path': file_path,
                                        'schema': schema.lower(),
                                        'symbols': symbols,
                                        'start_time': start_utc,
                                        'end_time': end_utc,
                                        'job_id': metadata.get("job_id", os.path.basename(job_dir))
                                    })
                                    self._log(
                                        f"Indexed DBN: '{filename}' for symbols {symbols}, schema '{schema}'",
                                        logging.DEBUG)
                                else:
                                    self._log(f"DBN file '{file_path}' listed in manifest but not found on disk.",
                                              logging.WARNING)

                except Exception as e:
                    self.logger.error(f"Error processing {job_dir}: {e}")

        self.logger.info(f"Found {found_dirs} directories, indexed {len(self.metadata_index)} files")

    def _find_dbn_files(self, schema: str, symbol: str,
                        start_time: pd.Timestamp, end_time: pd.Timestamp) -> List[str]:
        """Find DBN files matching the query parameters."""
        symbol_upper = symbol.upper()
        matching_files = []

        for entry in self.metadata_index:
            # Check schema match
            if entry['schema'] != schema.lower():
                continue

            # Check symbol match
            if symbol_upper not in entry['symbols']:
                continue

            # Check time overlap
            file_start = entry['start_time']
            file_end = entry['end_time']

            if max(file_start, start_time) <= min(file_end, end_time):
                matching_files.append(entry['file_path'])

        return sorted(set(matching_files))  # Remove duplicates and sort

    def _load_and_clean_dbn_files(self, files: List[str], symbol: str,
                                  start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """Load and clean data from DBN files."""
        if not files:
            return pd.DataFrame()

        all_dfs = []
        symbol_upper = symbol.upper()

        for file_path in files:
            # Load the file
            df = self._cached_read_dbn_file(file_path)

            if df.empty:
                continue

            # Filter to the specific symbol if needed
            if 'symbol' in df.columns:
                df = df[df['symbol'].astype(str).str.upper() == symbol_upper]

            # Filter to the requested time range
            if not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    self.logger.warning("Input DataFrame index is timezone-naive. Localizing to UTC.")
                    df.index = df.index.tz_localize('UTC')
                elif str(df.index.tz).upper() != 'UTC':  # Check if it's aware but not UTC
                    self.logger.info(f"Input DataFrame index is timezone-aware ({df.index.tz}). Converting to UTC.")
                    df.index = df.index.tz_convert('UTC')

                # If it's already a UTC DatetimeIndex, nothing happens here for timezone.
                if not df.index.is_monotonic_increasing:
                    # First sort if needed
                    df = df.sort_index()

                # Then filter
                df = df[start_time:end_time]

            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(all_dfs)

        # Remove duplicates if needed
        if not combined_df.index.is_unique:
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Sort
        return combined_df.sort_index()

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        symbol_upper = symbol.upper()

        # Check if we have info in the symbol map
        if symbol_upper in self._symbol_info_map:
            return self._symbol_info_map[symbol_upper]

        # Build basic info from metadata index
        jobs = set()
        for entry in self.metadata_index:
            if symbol_upper in entry['symbols']:
                jobs.add(entry['job_id'])

        # Create a basic info dict
        info = {
            'symbol': symbol_upper,
            'description': f"Data for {symbol_upper} found in {len(jobs)} jobs",
            'jobs': list(jobs)
        }

        return info

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        all_symbols = set()
        for entry in self.metadata_index:
            all_symbols.update(entry['symbols'])

        return sorted(list(all_symbols))

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized trade data for a symbol.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized trade columns
        """
        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Find matching files
        schema = self._TIMEFRAME_TO_SCHEMA_MAP['trades']
        files = self._find_dbn_files(schema, symbol, start_utc, end_utc)

        if not files:
            self._log(f"No trade files found for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame()

        # Load and combine data
        raw_df = self._load_and_clean_dbn_files(files, symbol, start_utc, end_utc)

        if raw_df.empty:
            return pd.DataFrame()

        # Clean using utility function
        raw_df = clean_trades_data(raw_df)

        # Map to standard format
        std_df = pd.DataFrame(index=raw_df.index)

        # Map columns from Databento format to standard format
        if 'price' in raw_df.columns:
            std_df['price'] = raw_df['price']
        else:
            std_df['price'] = 0.0  # Default

        # Size
        if 'size' in raw_df.columns:
            std_df['size'] = raw_df['size']
        else:
            std_df['size'] = 0.0  # Default

        # Side (keep Databento format: A=ask/sell, B=bid/buy, N=none)
        if 'side' in raw_df.columns:
            std_df['side'] = raw_df['side']
        else:
            std_df['side'] = 'N'  # Default

        # Exchange from publisher_id
        if 'publisher_id' in raw_df.columns:
            std_df['exchange'] = raw_df['publisher_id'].astype(str)
        else:
            std_df['exchange'] = 'UNKNOWN'  # Default

        # Conditions from flags
        if 'flags' in raw_df.columns:
            flag_map = {
                128: 'last',
                64: 'top_of_book',
                32: 'snapshot',
                16: 'mbp',
                8: 'bad_ts_recv'
            }

            # Apply flag mapping to each row
            def get_flags(flag_value):
                if pd.isna(flag_value):
                    return []
                return [flag_map[bit] for bit in flag_map.keys() if flag_value & bit]

            std_df['conditions'] = raw_df['flags'].apply(get_flags)
        else:
            # Default empty list for each row
            std_df['conditions'] = [[] for _ in range(len(std_df))]

        # Trade ID from sequence
        if 'sequence' in raw_df.columns:
            std_df['trade_id'] = raw_df['sequence'].astype(str)
        else:
            # Generate a unique ID based on timestamp
            std_df['trade_id'] = [f"gen_{idx.strftime('%Y%m%d%H%M%S%f')}" for idx in std_df.index]

        self._log(f"Loaded {len(std_df)} trades for {symbol}")
        return std_df

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized quote data for a symbol.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized quote columns
        """
        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Find matching files
        schema = self._TIMEFRAME_TO_SCHEMA_MAP['quotes']
        files = self._find_dbn_files(schema, symbol, start_utc, end_utc)

        if not files:
            self._log(f"No quote files found for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame()

        # Load and combine data
        raw_df = self._load_and_clean_dbn_files(files, symbol, start_utc, end_utc)

        if raw_df.empty:
            return pd.DataFrame()

        # Clean using utility function
        raw_df = clean_quotes_data(raw_df)

        # Map to standard format with all required fields
        std_df = pd.DataFrame(index=raw_df.index)

        if 'bid_px_00' in raw_df.columns:
            std_df['bid_price'] = raw_df['bid_px_00']
        else:
            std_df['bid_price'] = 0.0  # Default

        if 'ask_px_00' in raw_df.columns:
            std_df['ask_price'] = raw_df['ask_px_00']
        else:
            std_df['ask_price'] = 0.0  # Default

        # Bid size
        if 'bid_sz_00' in raw_df.columns:
            std_df['bid_size'] = raw_df['bid_sz_00']
        else:
            std_df['bid_size'] = 0.0  # Default

        # Ask size
        if 'ask_sz_00' in raw_df.columns:
            std_df['ask_size'] = raw_df['ask_sz_00']
        else:
            std_df['ask_size'] = 0.0  # Default

        # Bid count
        if 'bid_ct_00' in raw_df.columns:
            std_df['bid_count'] = raw_df['bid_ct_00']
        else:
            std_df['bid_count'] = 0  # Default

        # Ask count
        if 'ask_ct_00' in raw_df.columns:
            std_df['ask_count'] = raw_df['ask_ct_00']
        else:
            std_df['ask_count'] = 0  # Default

        # Side Can be Ask for sell aggresor, Bid for buy aggresor, or None for no aggressor
        if 'side' in raw_df.columns:
            std_df['side'] = raw_df['side']
        else:
            std_df['side'] = 'N'

        # Exchange from publisher_id
        if 'publisher_id' in raw_df.columns:
            std_df['exchange'] = raw_df['publisher_id'].astype(str)
        else:
            std_df['exchange'] = 'UNKNOWN'  # Default

        self._log(f"Loaded {len(std_df)} quotes for {symbol}")
        return std_df

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized OHLCV bar data for a symbol.

        Args:
            symbol: Symbol to get data for
            timeframe: Bar timeframe ("1s", "1m", "5m", "1d")
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized bar columns
        """
        # Validate timeframe
        if timeframe not in ["1s", "1m", "5m", "1d"]:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: 1s, 1m, 5m, 1d")

        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # For 5m timeframe, we might need to resample from 1m
        if timeframe == "5m":
            # Try to get directly if available
            direct_5m = self._get_bars_direct(symbol, timeframe, start_utc, end_utc)
            if not direct_5m.empty:
                return direct_5m

            # If not available, try to build from 1m
            self._log(f"No direct 5m bars found for {symbol}, building from 1m bars")
            return self._build_5m_bars_from_1m(symbol, start_utc, end_utc)

        # For other timeframes, get directly
        return self._get_bars_direct(symbol, timeframe, start_utc, end_utc)

    def _get_bars_direct(self, symbol: str, timeframe: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        """Get bar data directly from files."""
        schema = self._TIMEFRAME_TO_SCHEMA_MAP[timeframe]
        files = self._find_dbn_files(schema, symbol, start_utc, end_utc)

        if not files:
            self._log(f"No {timeframe} bar files found for {symbol}")
            return pd.DataFrame()

        # Load and combine data
        raw_df = self._load_and_clean_dbn_files(files, symbol, start_utc, end_utc)

        if raw_df.empty:
            return pd.DataFrame()

        # Clean using utility function
        raw_df = clean_ohlc_data(raw_df)

        # Map to standard format
        std_df = pd.DataFrame(index=raw_df.index)

        # Map columns from Databento format to standard format
        # Open price
        if 'open' in raw_df.columns:
            std_df['open'] = raw_df['open']
        else:
            std_df['open'] = 0.0  # Default

        # High price
        if 'high' in raw_df.columns:
            std_df['high'] = raw_df['high']
        else:
            std_df['high'] = 0.0  # Default

        # Low price
        if 'low' in raw_df.columns:
            std_df['low'] = raw_df['low']
        else:
            std_df['low'] = 0.0  # Default

        # Close price
        if 'close' in raw_df.columns:
            std_df['close'] = raw_df['close']
        else:
            std_df['close'] = 0.0  # Default

        # Volume
        if 'volume' in raw_df.columns:
            std_df['volume'] = raw_df['volume']
        else:
            std_df['volume'] = 0.0  # Default

        # Add timeframe column
        std_df['timeframe'] = timeframe

        self._log(f"Loaded {len(std_df)} {timeframe} bars for {symbol}")
        return std_df

    def _build_5m_bars_from_1m(self, symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        """Build 5-minute bars from 1-minute bars."""
        # Get 1-minute bars
        bars_1m = self.get_bars(symbol, "1m", start_utc, end_utc)

        if bars_1m.empty:
            self._log(f"No 1m bars found to build 5m bars for {symbol}")
            return pd.DataFrame()

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in bars_1m.columns for col in required_cols):
            self._log(f"Missing required columns in 1m bars: {set(required_cols) - set(bars_1m.columns)}")
            return pd.DataFrame()

        # Resample to 5-minute bars
        # Group by 5-minute intervals
        bars_5m = bars_1m.resample('5min', closed='left', label='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Drop rows with NaN values (empty intervals)
        bars_5m = bars_5m.dropna(subset=['open'])

        # Add timeframe column
        bars_5m['timeframe'] = "5m"

        self._log(f"Built {len(bars_5m)} 5m bars from 1m bars for {symbol}")
        return bars_5m

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized status data for a symbol.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized status columns
        """
        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Find matching files
        schema = self._TIMEFRAME_TO_SCHEMA_MAP['status']
        files = self._find_dbn_files(schema, symbol, start_utc, end_utc)

        if not files:
            self._log(f"No status files found for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame()

        # Load and combine data
        raw_df = self._load_and_clean_dbn_files(files, symbol, start_utc, end_utc)

        if raw_df.empty:
            return pd.DataFrame()

        # Map to standard format
        std_df = pd.DataFrame(index=raw_df.index)

        # Status - always include
        if 'action' in raw_df.columns:
            # Define status code mapping based on Databento's documentation
            status_map = {
                0: 'NONE',
                1: 'PRE_OPEN',
                2: 'PRE_CROSS',
                3: 'QUOTING',
                4: 'CROSS',
                5: 'ROTATION',
                6: 'PRICE_INDICATION',
                7: 'TRADING',
                8: 'HALTED',
                9: 'PAUSED',
                10: 'SUSPENDED',
                11: 'PRE_CLOSE',
                12: 'CLOSED',
                13: 'POST_CLOSE',
                14: 'SHORT_SELL_RESTRICTION',
                15: 'NOT_AVAILABLE'
            }
            std_df['status'] = raw_df['action'].map(lambda x: status_map.get(x, f"UNKNOWN_{x}"))
        else:
            std_df['status'] = 'UNKNOWN'  # Default

        # Reason - always include
        if 'reason' in raw_df.columns:
            # Define reason code mapping
            reason_map = {
                0: 'NONE',
                1: 'SCHEDULED',
                2: 'SURVEILLANCE',
                3: 'MARKET_EVENT',
                4: 'INSTRUMENT_ACTIVATION',
                5: 'INSTRUMENT_EXPIRATION',
                6: 'RECOVERY_IN_PROCESS',
                10: 'REGULATORY',
                # Additional values...
                100: 'CORPORATE_ACTION'
            }
            std_df['reason'] = raw_df['reason'].map(lambda x: reason_map.get(x, f"UNKNOWN_{x}"))
        else:
            std_df['reason'] = 'UNKNOWN'  # Default

        # Trading status - always include
        if 'is_trading' in raw_df.columns:
            std_df['is_trading'] = raw_df['is_trading'] == 'Y'
        else:
            # Default based on status if available
            if 'status' in std_df.columns:
                trading_statuses = ['TRADING', 'QUOTING', 'ROTATION', 'PRICE_INDICATION']
                std_df['is_trading'] = std_df['status'].isin(trading_statuses)
            else:
                std_df['is_trading'] = True  # Default assumption

        # Halted status - always include
        if 'is_halted' in raw_df.columns:
            std_df['is_halted'] = raw_df['is_halted'] == 'Y'
        else:
            # Derive from status if possible
            if 'status' in std_df.columns:
                halted_statuses = ['HALTED', 'PAUSED', 'SUSPENDED']
                std_df['is_halted'] = std_df['status'].isin(halted_statuses)
            else:
                std_df['is_halted'] = False  # Default

        # Short sell restriction - always include
        if 'is_short_sell_restricted' in raw_df.columns:
            std_df['is_short_sell_restricted'] = raw_df['is_short_sell_restricted'] == 'Y'
        else:
            # Check for SHORT_SELL_RESTRICTION status
            if 'status' in std_df.columns:
                std_df['is_short_sell_restricted'] = std_df['status'] == 'SHORT_SELL_RESTRICTION'
            else:
                std_df['is_short_sell_restricted'] = False  # Default

        self._log(f"Loaded {len(std_df)} status records for {symbol}")
        return std_df
