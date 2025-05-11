# data/providers/databento/databento_file_provider.py
import databento as db
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import os

from data.provider.data_provider import HistoricalDataProvider


class DabentoFileProvider(HistoricalDataProvider):
    """Implementation of Historical Provider using Databento file storage."""

    def __init__(self, data_dir: str, symbol_info_file: str = None):
        """
        Initialize the Databento file provider.

        Args:
            data_dir: Directory containing Databento data files
            symbol_info_file: Optional path to a file with symbol metadata
        """
        self.data_dir = data_dir
        self._symbol_info = {}

        # Load symbol info if provided
        if symbol_info_file and os.path.exists(symbol_info_file):
            self._symbol_info = pd.read_csv(symbol_info_file).set_index('symbol').to_dict('index')

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        if symbol in self._symbol_info:
            return self._symbol_info[symbol]
        else:
            # Return minimal info
            return {"symbol": symbol, "description": f"Unknown symbol {symbol}"}

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        # Infer from directory structure or filenames
        # This is a simplified implementation
        symbols = set()
        for file in os.listdir(self.data_dir):
            if file.endswith('.dbn'):
                # Assume filename format: symbol_schema_date.dbn
                symbol = file.split('_')[0]
                symbols.add(symbol)
        return list(symbols)

    def _get_file_path(self, symbol: str, schema: str, date: Union[datetime, str]) -> str:
        """Helper to get the file path for a symbol, schema, and date."""
        if isinstance(date, datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            # Assume ISO format string
            date_str = datetime.fromisoformat(date.replace('Z', '+00:00')).strftime('%Y%m%d')

        file_name = f"{symbol}_{schema}_{date_str}.dbn"
        return os.path.join(self.data_dir, file_name)

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical trades for a symbol in a time range."""
        # Simplification: assume one file per day, use start_time to determine file
        file_path = self._get_file_path(symbol, "trades", start_time)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trades file not found: {file_path}")

        # Use Databento's DBN reader to load the file
        store = db.DBNStore.from_file(file_path)
        df = store.to_df()

        # Filter for the requested time range
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        mask = (df.index >= start_time) & (df.index < end_time)
        return df[mask]

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical quotes for a symbol in a time range."""
        file_path = self._get_file_path(symbol, "mbp-1", start_time)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Quotes file not found: {file_path}")

        store = db.DBNStore.from_file(file_path)
        df = store.to_df()

        # Filter for the requested time range
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        mask = (df.index >= start_time) & (df.index < end_time)
        return df[mask]

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get OHLCV bars for a symbol, timeframe in a time range."""
        # Map the timeframe string to Databento's schema format
        timeframe_map = {
            "1s": "ohlcv-1s",
            "1m": "ohlcv-1m",
            "1h": "ohlcv-1h",
            "1d": "ohlcv-1d"
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_map.keys())}")

        file_path = self._get_file_path(symbol, timeframe_map[timeframe], start_time)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bars file not found: {file_path}")

        store = db.DBNStore.from_file(file_path)
        df = store.to_df()

        # Filter for the requested time range
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        mask = (df.index >= start_time) & (df.index < end_time)
        return df[mask]

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get status updates (halts, etc.) for a symbol in a time range."""
        file_path = self._get_file_path(symbol, "status", start_time)

        if not os.path.exists(file_path):
            # Status data might not exist for all days, return empty DataFrame
            columns = ['publisher_id', 'instrument_id', 'ts_event', 'ts_recv',
                       'action', 'reason', 'trading_event', 'is_trading',
                       'is_quoting', 'is_short_sell_restricted']
            return pd.DataFrame(columns=columns)

        store = db.DBNStore.from_file(file_path)
        df = store.to_df()

        # Filter for the requested time range
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        mask = (df.index >= start_time) & (df.index < end_time)
        return df[mask]