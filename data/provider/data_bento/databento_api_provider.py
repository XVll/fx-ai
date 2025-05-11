# data/providers/databento/databento_api_provider.py
import databento as db
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from data.provider.data_provider import HistoricalDataProvider


class DabentoAPIProvider(HistoricalDataProvider):
    """Implementation of Historical Provider using Databento API."""

    def __init__(self, api_key: str, dataset: str = None):
        """
        Initialize the Databento API provider.

        Args:
            api_key: Databento API key
            dataset: Default dataset to use (e.g., "XNAS.ITCH")
        """
        self.client = db.Historical(api_key)
        self.dataset = dataset

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        # Use Databento's metadata API to get symbol information
        symbol_metadata = self.client.metadata.lookup_symbols(
            dataset=self.dataset,
            symbols=[symbol],
            stype_in="raw_symbol"
        )

        if not symbol_metadata:
            raise ValueError(f"Symbol {symbol} not found in dataset {self.dataset}")

        return symbol_metadata[0]

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        # This would typically use Databento's metadata API
        # For simplicity, we're returning a limited implementation
        return self.client.metadata.list_symbols(dataset=self.dataset)

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical trades for a symbol in a time range."""
        trades_data = self.client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[symbol],
            start=start_time,
            end=end_time,
            stype_in="raw_symbol",
            schema="trades"
        )

        return trades_data.to_df()

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical quotes for a symbol in a time range."""
        quotes_data = self.client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[symbol],
            start=start_time,
            end=end_time,
            stype_in="raw_symbol",
            schema="mbp-1"  # Market By Price - Level 1 (best bid/ask)
        )

        return quotes_data.to_df()

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

        bars_data = self.client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[symbol],
            start=start_time,
            end=end_time,
            stype_in="raw_symbol",
            schema=timeframe_map[timeframe]
        )

        return bars_data.to_df()

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get status updates (halts, etc.) for a symbol in a time range."""
        status_data = self.client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[symbol],
            start=start_time,
            end=end_time,
            stype_in="raw_symbol",
            schema="status"
        )

        return status_data.to_df()