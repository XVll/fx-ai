# data/provider/data_provider.py
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
    """Abstract class for historical data providers."""

    @abstractmethod
    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get historical trades for a symbol in a time range.

        Returns:
            DataFrame with standardized columns:
                - timestamp (index): UTC timestamp from ts_event
                - price: Trade price
                - size: Trade size
                - side: Trade side ('B', 'S', 'N')
                - exchange: Exchange identifier
                - conditions: Trade conditions
                - trade_id: Unique trade identifier
        """
        pass

    @abstractmethod
    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get historical quotes for a symbol in a time range.

        Returns:
            DataFrame with standardized columns:
                - timestamp (index): UTC timestamp from ts_event
                - bid_price: Best bid price
                - ask_price: Best ask price
                - bid_size: Best bid size
                - ask_size: Best ask size
                - bid_count: Number of orders at bid
                - ask_count: Number of orders at ask
                - exchange: Exchange identifier
        """
        pass

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get OHLCV bars for a symbol, timeframe in a time range.
        Only supports: "1s", "1m", "5m", "1d"

        Returns:
            DataFrame with standardized columns:
                - timestamp (index): UTC timestamp from ts_event
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Volume traded
                - timeframe: Bar timeframe
        """
        pass

    @abstractmethod
    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get status updates (halts, etc.) for a symbol in a time range.

        Returns:
            DataFrame with standardized columns:
                - timestamp (index): UTC timestamp from ts_event
                - status: Trading status code
                - reason: Reason for status change
                - is_trading: Whether trading is allowed
                - is_halted: Whether trading is halted
                - is_short_sell_restricted: Whether short selling is restricted
        """
        pass


class LiveDataProvider(DataProvider):
    """Abstract class for live data providers."""

    @abstractmethod
    def subscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """
        Subscribe to live data for symbols.

        Args:
            symbols: List of symbols to subscribe to
            data_types: List of data types to subscribe to
                        Supported: "trades", "quotes", "status", "bars_1s", "bars_1m", "bars_5m", "bars_1d"
        """
        pass

    @abstractmethod
    def unsubscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """
        Unsubscribe from live data for symbols.

        Args:
            symbols: List of symbols to unsubscribe from
            data_types: List of data types to unsubscribe from
                        Supported: "trades", "quotes", "status", "bars_1s", "bars_1m", "bars_5m", "bars_1d"
        """
        pass

    @abstractmethod
    def get_latest_trade(self, symbol: str) -> Dict:
        """
        Get the latest trade for a symbol.

        Returns:
            Dictionary with trade information in standardized format
        """
        pass

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Dict:
        """
        Get the latest quote for a symbol.

        Returns:
            Dictionary with quote information in standardized format
        """
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str, timeframe: str) -> Dict:
        """
        Get the latest OHLCV bar for a symbol and timeframe.

        Args:
            symbol: Symbol to get data for
            timeframe: Bar timeframe ("1s", "1m", "5m", "1d")

        Returns:
            Dictionary with bar information in standardized format
        """
        pass

    @abstractmethod
    def add_trade_callback(self, callback_fn) -> None:
        """
        Add callback for trade updates.

        Args:
            callback_fn: Function to call when a new trade is received
                        Function signature: callback_fn(trade_data: Dict)
        """
        pass

    @abstractmethod
    def add_quote_callback(self, callback_fn) -> None:
        """
        Add callback for quote updates.

        Args:
            callback_fn: Function to call when a new quote is received
                        Function signature: callback_fn(quote_data: Dict)
        """
        pass

    @abstractmethod
    def add_bar_callback(self, callback_fn) -> None:
        """
        Add callback for bar updates.

        Args:
            callback_fn: Function to call when a new bar is completed
                        Function signature: callback_fn(bar_data: Dict)
        """
        pass

    @abstractmethod
    def add_status_callback(self, callback_fn) -> None:
        """
        Add callback for status updates.

        Args:
            callback_fn: Function to call when a status update is received
                        Function signature: callback_fn(status_data: Dict)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the provider and clean up resources.
        Should be called when the provider is no longer needed.
        """
        pass