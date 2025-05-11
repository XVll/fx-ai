# data/data_provider.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

class DataProvider(ABC):
    """Base abstract class for all data providers."""

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        pass

class HistoricalDataProvider(DataProvider):
    """Abstract class for historical data providers (file or API)."""

    @abstractmethod
    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical trades for a symbol in a time range."""
        pass

    @abstractmethod
    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get historical quotes for a symbol in a time range."""
        pass

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get OHLCV bars for a symbol, timeframe in a time range."""
        pass

    @abstractmethod
    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get status updates (halts, etc.) for a symbol in a time range."""
        pass

class LiveDataProvider(DataProvider):
    """Abstract class for live data providers."""

    @abstractmethod
    def subscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Subscribe to live data for symbols."""
        pass

    @abstractmethod
    def unsubscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Unsubscribe from live data for symbols."""
        pass

    @abstractmethod
    def get_latest_trade(self, symbol: str) -> Dict:
        """Get the latest trade for a symbol."""
        pass

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Dict:
        """Get the latest quote for a symbol."""
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str, timeframe: str) -> Dict:
        """Get the latest OHLCV bar for a symbol and timeframe."""
        pass

    @abstractmethod
    def add_trade_callback(self, callback_fn) -> None:
        """Add callback for trade updates."""
        pass

    @abstractmethod
    def add_quote_callback(self, callback_fn) -> None:
        """Add callback for quote updates."""
        pass

    @abstractmethod
    def add_bar_callback(self, callback_fn) -> None:
        """Add callback for bar updates."""
        pass