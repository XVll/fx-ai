# data/provider/data_bento/databento_file_provider.py
import databento as db
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import pandas as pd
# import numpy as np # Not strictly needed in this version
import os
import logging
import json
from functools import lru_cache

# Assuming HistoricalDataProvider is in a reachable path like this:
from data.provider.data_provider import HistoricalDataProvider
from data.utils.helpers import ensure_timezone_aware


class DabentoFileProvider(HistoricalDataProvider):
    """
    Implementation of HistoricalDataProvider using local Databento file storage,
    leveraging Databento's metadata.json and manifest.json for efficient file discovery.
    """

    _TIMEFRAME_TO_SCHEMA_MAP = {
        "trades": "trades",
        "quotes": "mbp-1",  # Default, adjust if using e.g., mbp-10
        "status": "status",
        "1s": "ohlcv-1s",
        "1m": "ohlcv-1m",
        "5m": "ohlcv-5m",
        "1h": "ohlcv-1h",
        "1d": "ohlcv-1d",
    }
    _BAR_SCHEMAS = {"ohlcv-1s", "ohlcv-1m", "ohlcv-5m", "ohlcv-1h", "ohlcv-1d"}

    def __init__(self, data_dir: str,
                 symbol_info_file: Optional[str] = None,
                 verbose: bool = False,
                 dbn_cache_size: int = 32):
        """
        Initialize the Databento file provider.

        Args:
            data_dir: Directory containing Databento job folders.
            symbol_info_file: Optional path to a CSV file with symbol metadata.
            verbose: Enable verbose logging for debugging.
            dbn_cache_size: Max DBN file contents (DataFrames) to keep in LRU cache.
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)  # Default to INFO

        self._symbol_info_map: Dict[str, Dict[str, Any]] = {}
        if symbol_info_file and os.path.exists(symbol_info_file):
            try:
                df_info = pd.read_csv(symbol_info_file)
                if 'symbol' in df_info.columns:
                    # Normalize symbol to uppercase for consistent lookup
                    df_info['symbol'] = df_info['symbol'].astype(str).str.upper()
                    self._symbol_info_map = df_info.set_index('symbol').to_dict('index')
                    self.logger.info(
                        f"Loaded symbol info from {symbol_info_file} for {len(self._symbol_info_map)} symbols.")
                else:
                    self.logger.warning(f"'symbol' column not found in {symbol_info_file}. Symbol info not loaded.")
            except Exception as e:
                self.logger.error(f"Error loading symbol info from {symbol_info_file}: {e}")

        self.metadata_index: List[Dict[str, Any]] = []
        self._scan_and_build_metadata_index()

        @lru_cache(maxsize=dbn_cache_size)
        def _read_dbn_file_to_df_cached(file_path: str) -> pd.DataFrame:
            self._log(f"CACHE MISS: Reading DBN file: {file_path}", logging.DEBUG)
            try:
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'ts_event' in df.columns:  # Common timestamp column from Databento
                        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', errors='coerce')
                        df = df.set_index('ts_event')
                    elif pd.api.types.is_datetime64_any_dtype(df.index):  # Already a DatetimeIndex but not instance?
                        pass  # Index is already datetime-like
                    elif len(df.columns) > 0 and pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                        # Try to set the first column as index if it's datetime-like
                        df = df.set_index(df.columns[0])
                    else:
                        self.logger.warning(
                            f"Cannot determine DatetimeIndex for DBN file {file_path}. Index: {df.index}, Columns: {df.columns}")
                        return pd.DataFrame()

                if df.empty or not isinstance(df.index, pd.DatetimeIndex):  # Re-check after potential index setting
                    self.logger.debug(f"DataFrame is empty or has no DatetimeIndex after processing {file_path}.")
                    return pd.DataFrame()

                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                elif str(df.index.tz).upper() != 'UTC':
                    df.index = df.index.tz_convert('UTC')

                # Normalize symbol column to uppercase if exists, for consistent filtering
                if 'symbol' in df.columns:
                    df['symbol'] = df['symbol'].astype(str).str.upper()
                return df
            except Exception as e:
                self.logger.error(f"Error reading or processing DBN file {file_path}: {e}")
                return pd.DataFrame()

        self._cached_read_dbn_file = _read_dbn_file_to_df_cached

    def _log(self, msg: str, level: int = logging.INFO):
        if level >= self.logger.level:  # Respect the logger's configured level
            self.logger.log(level, msg)

    def _scan_and_build_metadata_index(self):
        self.metadata_index = []
        self.logger.debug(f"Scanning for metadata in '{self.data_dir}'...")
        found_scheme_dir = 0
        for root, dirs, files in os.walk(self.data_dir, topdown=True):
            if "metadata.json" in files and "manifest.json" in files:
                found_scheme_dir += 1
                job_id_dir = root
                job_id_name = os.path.basename(job_id_dir)
                self._log(f"Processing Databento job directory: {job_id_dir}", logging.DEBUG)
                try:
                    with open(os.path.join(job_id_dir, "metadata.json"), 'r') as f_meta:
                        meta_content = json.load(f_meta)
                    with open(os.path.join(job_id_dir, "manifest.json"), 'r') as f_manifest:
                        manifest_content = json.load(f_manifest)

                    query_info = meta_content.get("query", {})
                    schema = query_info.get("schema")
                    # Normalize symbols from metadata to uppercase
                    symbols_list = [str(s).upper() for s in query_info.get("symbols", [])]
                    start_ns = query_info.get("start")
                    end_ns = query_info.get("end")

                    if not all([schema, symbols_list, start_ns is not None, end_ns is not None]):
                        self._log(f"Skipping '{job_id_dir}': missing essential fields in metadata.json's query.", logging.WARNING)
                        continue

                    start_utc = pd.Timestamp(start_ns, unit='ns', tz='UTC')
                    end_utc = pd.Timestamp(end_ns, unit='ns', tz='UTC')

                    for file_entry in manifest_content.get("files", []):
                        dbn_filename = file_entry.get("filename", "")
                        if dbn_filename.endswith((".dbn", ".dbn.zst")):
                            if schema.lower() in dbn_filename.lower():
                                dbn_file_path = os.path.join(job_id_dir, dbn_filename)
                                if os.path.exists(dbn_file_path):
                                    self.metadata_index.append({
                                        'dbn_file_path': dbn_file_path,
                                        'schema': schema.lower(),
                                        'symbols': symbols_list,
                                        'start_time_utc': start_utc,
                                        'end_time_utc': end_utc,
                                        'job_id': meta_content.get("job_id", job_id_name)
                                    })
                                    self._log(
                                        f"Indexed DBN: '{dbn_filename}' for symbols {symbols_list}, schema '{schema}'",
                                        logging.DEBUG)
                                else:
                                    self._log(f"DBN file '{dbn_file_path}' listed in manifest but not found on disk.",
                                              logging.WARNING)
                except Exception as e:  # Catch broader exceptions during file processing
                    self._log(f"Error processing directory '{job_id_dir}': {e}", logging.ERROR)
        self.logger.info(
            f"Scanning finished in {self.data_dir}. Found {found_scheme_dir} scheme directories, indexed {len(self.metadata_index)} DBN file entries.")
        if not self.metadata_index:
            self.logger.warning(
                f"No DBN files were indexed. Check 'data_dir' ('{self.data_dir}') and Databento download structure.")

    def _find_dbn_files_for_query(self, schema_filter: str, symbol_filter: str,
                                  query_start_utc: pd.Timestamp, query_end_utc: pd.Timestamp) -> List[str]:
        matching_file_paths = set()
        schema_filter_lower = schema_filter.lower()
        symbol_filter_upper = str(symbol_filter).upper()  # Normalize query symbol

        for entry in self.metadata_index:
            if entry['schema'] != schema_filter_lower:
                continue
            # entry['symbols'] is already list of uppercase strings from _scan_and_build_metadata_index
            if symbol_filter_upper not in entry['symbols']:
                continue
            if max(entry['start_time_utc'], query_start_utc) <= min(entry['end_time_utc'], query_end_utc):
                matching_file_paths.add(entry['dbn_file_path'])

        if not matching_file_paths:
            self._log(f"No DBN files found in index for: {symbol_filter_upper}@{schema_filter_lower} "
                      f"between {query_start_utc} and {query_end_utc}", logging.DEBUG)
        return sorted(list(matching_file_paths))

    def _load_data_for_request(self, schema_key: str, symbol: str,
                               start_time_utc: pd.Timestamp, end_time_utc: pd.Timestamp) -> pd.DataFrame:
        databento_schema = self._TIMEFRAME_TO_SCHEMA_MAP.get(schema_key.lower(), schema_key.lower())
        symbol_upper = str(symbol).upper()  # Normalize query symbol

        dbn_file_paths = self._find_dbn_files_for_query(databento_schema, symbol_upper, start_time_utc, end_time_utc)
        if not dbn_file_paths:
            return pd.DataFrame()

        all_symbol_data_in_range = []
        for file_path in dbn_file_paths:
            df_full_file = self._cached_read_dbn_file(file_path)  # Symbols already uppercased here
            if df_full_file.empty:
                continue

            df_symbol_specific = df_full_file
            if 'symbol' in df_full_file.columns:
                # Filter by the specific symbol (already uppercased in df_full_file)
                df_symbol_specific = df_full_file[df_full_file['symbol'] == symbol_upper]

            if df_symbol_specific.empty:
                self._log(f"Symbol '{symbol_upper}' not found in data from file '{file_path}' after filtering.",
                          logging.DEBUG)
                continue

            if not df_symbol_specific.index.is_monotonic_increasing:
                df_symbol_specific = df_symbol_specific.sort_index()

            # Pandas loc slicing is inclusive of both start and end if they exist in the index
            df_in_time_range = df_symbol_specific.loc[start_time_utc:end_time_utc]

            if not df_in_time_range.empty:
                all_symbol_data_in_range.append(df_in_time_range)

        if not all_symbol_data_in_range:
            self._log(
                f"No data for {symbol_upper}@{databento_schema} found within DBN files in range {start_time_utc} to {end_time_utc}",
                logging.DEBUG)
            return pd.DataFrame()

        final_df = pd.concat(all_symbol_data_in_range)
        if not final_df.index.is_unique:
            final_df = final_df[~final_df.index.duplicated(keep='first')]
        return final_df.sort_index()

    def get_symbol_info(self, symbol: str) -> Dict:
        symbol_upper = str(symbol).upper()
        if symbol_upper in self._symbol_info_map:
            return self._symbol_info_map[symbol_upper]

        related_jobs = [entry['job_id'] for entry in self.metadata_index if symbol_upper in entry['symbols']]
        description = f"Data potentially available for {symbol_upper}."
        if related_jobs:
            description += f" Associated with Databento job(s): {', '.join(list(set(related_jobs))[:3])}"
            if len(related_jobs) > 3: description += "..."

        return {"symbol": symbol_upper, "description": description}

    def get_available_symbols(self) -> List[str]:
        if not self.metadata_index:
            self.logger.warning("Metadata index is empty. Cannot determine available symbols.")
            return []
        all_symbols = set()
        for entry in self.metadata_index:
            all_symbols.update(entry['symbols'])  # entry['symbols'] is already a list of uppercase strings
        return sorted(list(all_symbols))

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get historical trades for a symbol in a time range.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with trades data
        """
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        df_trades = self._load_data_for_request("trades", symbol, start_utc, end_utc)

        if not df_trades.empty:
            self.logger.info(
                f"Loaded {len(df_trades)} trades data for {symbol} from {start_utc.date()} to {end_utc.date()}")
        else:
            self.logger.warning(f"No trades data loaded for {symbol} in range {start_utc} to {end_utc}")

        return df_trades

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get historical quotes for a symbol in a time range.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with quotes data
        """
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        df_quotes = self._load_data_for_request(self._TIMEFRAME_TO_SCHEMA_MAP["quotes"], symbol, start_utc, end_utc)

        if not df_quotes.empty:
            self.logger.info(
                f"Loaded {len(df_quotes)} quotes data for {symbol} from {start_utc.date()} to {end_utc.date()}")
        else:
            self.logger.warning(f"No quotes data loaded for {symbol} in range {start_utc} to {end_utc}")

        return df_quotes

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get OHLCV bars for a symbol, timeframe in a time range.

        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe for bars (e.g., "1s", "1m", "5m", "1d")
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with OHLCV bars
        """
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        databento_schema = self._TIMEFRAME_TO_SCHEMA_MAP.get(timeframe.lower())
        if not databento_schema:
            msg = f"Unsupported timeframe: {timeframe}. Supported: {list(self._TIMEFRAME_TO_SCHEMA_MAP.keys())}"
            self.logger.error(msg)
            raise ValueError(msg)

        df_bars = self._load_data_for_request(databento_schema, symbol, start_utc, end_utc)

        # If 5m bars not found directly, try to create from 1m
        if timeframe.lower() == "5m" and df_bars.empty:
            self._log(f"No direct 5m bars found for {symbol}. Trying to create from 1m bars.", logging.DEBUG)
            df_bars = self._create_5m_bars_from_1m(symbol, start_utc, end_utc)

        if not df_bars.empty:
            self.logger.info(
                f"Loaded {len(df_bars)} {timeframe} bars for {symbol} from {start_utc.date()} to {end_utc.date()}")
        else:
            self.logger.warning(f"No {timeframe} bars loaded for {symbol} in range {start_utc} to {end_utc}")

        return df_bars

    def _create_5m_bars_from_1m(self, symbol: str,
                                start_time_utc: pd.Timestamp,
                                end_time_utc: pd.Timestamp) -> pd.DataFrame:
        self.logger.debug(
            f"Attempting to fetch 1m bars for {symbol} ({start_time_utc} to {end_time_utc}) to resample into 5m bars.")
        bars_1m_df = self._load_data_for_request(self._TIMEFRAME_TO_SCHEMA_MAP["1m"], symbol, start_time_utc,
                                                 end_time_utc)

        if bars_1m_df.empty:
            self.logger.info(
                f"No 1m data found for {symbol} to create 5m bars in range {start_time_utc}-{end_time_utc}.")
            return pd.DataFrame()
        if not isinstance(bars_1m_df.index, pd.DatetimeIndex):
            self.logger.warning("1m bars data does not have a DatetimeIndex. Cannot resample.")
            return pd.DataFrame()

        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(bars_1m_df.columns):
            missing = required_cols - set(bars_1m_df.columns)
            self.logger.warning(f"1m bars data is missing required columns for OHLCV resampling: {missing}")
            return pd.DataFrame()
        try:
            resampled_df = bars_1m_df.resample('5min', closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            })
            resampled_df.dropna(subset=['open'], inplace=True)
            self._log(f"Successfully created {len(resampled_df)} 5m bars from 1m data for {symbol}.", logging.DEBUG)
            return resampled_df
        except Exception as e:
            self.logger.error(f"Error resampling 1m to 5m bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get status updates (halts, etc.) for a symbol in a time range.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with status data
        """
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        df_status = self._load_data_for_request(self._TIMEFRAME_TO_SCHEMA_MAP["status"], symbol, start_utc, end_utc)

        if not df_status.empty:
            self.logger.info(
                f"Loaded {len(df_status)} status data for {symbol} from {start_utc.date()} to {end_utc.date()}")
        else:
            self.logger.info(f"No status data found for {symbol} in range {start_utc} to {end_utc}")

        return df_status
