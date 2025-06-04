# data/provider/data_provider.py - Enhanced with unified interface
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from enum import Enum


class DataMode(Enum):
    """Enumeration for data provider modes."""

    HISTORICAL = "historical"
    LIVE = "live"
    HYBRID = "hybrid"  # Historical with live transition


class DataProvider(ABC):
    """Base abstract class for all data providers."""

    def __init__(self):
        self._mode = DataMode.HISTORICAL
        self._current_timestamp = None

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        pass

    @property
    def mode(self) -> DataMode:
        """Get current data mode."""
        return self._mode

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Get current timestamp for data queries."""
        return self._current_timestamp


class HistoricalDataProvider(DataProvider):
    """Abstract class for historical data providers."""

    @abstractmethod
    def get_trades(
        self,
        symbol: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
    ) -> pd.DataFrame:
        """
        Get historical trades for a symbol in a time range.

        Returns:
            DataFrame with standardized columns:
                - price: Trade price
                - size: Trade size
                - side: Trade side ('B', 'S', 'N')
                - exchange: Exchange identifier
                - conditions: Trade conditions
                - trade_id: Unique trade identifier
        """
        pass

    @abstractmethod
    def get_quotes(
        self,
        symbol: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
    ) -> pd.DataFrame:
        """
        Get historical quotes for a symbol in a time range.

        Returns:
            DataFrame with standardized columns:
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
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
    ) -> pd.DataFrame:
        """
        Get OHLCV bars for a symbol, timeframe in a time range.
        Only supports: "1s", "1m", "5m", "1d"

        Returns:
            DataFrame with standardized columns:
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Volume traded
                - timeframe: Bar timeframe
        """
        pass

    @abstractmethod
    def get_status(
        self,
        symbol: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
    ) -> pd.DataFrame:
        """
        Get status updates (halts, etc.) for a symbol in a time range.

        Returns:
            DataFrame with standardized columns:
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


class UnifiedDataProvider(DataProvider):
    """Unified interface for seamless historical/live data access.

    This provider abstracts the difference between historical and live data,
    providing a consistent interface for both training and production.
    """

    def __init__(
        self,
        historical_provider: Optional[HistoricalDataProvider] = None,
        live_provider: Optional[LiveDataProvider] = None,
    ):
        """Initialize unified provider.

        Args:
            historical_provider: Provider for historical data
            live_provider: Provider for live data
        """
        super().__init__()
        self.historical_provider = historical_provider
        self.live_provider = live_provider

        # Determine mode based on available providers
        if historical_provider and live_provider:
            self._mode = DataMode.HYBRID
        elif live_provider:
            self._mode = DataMode.LIVE
        else:
            self._mode = DataMode.HISTORICAL

        # State for hybrid mode
        self._transition_timestamp = None
        self._historical_buffer = {}  # Cache for lookback data

    def get_data_at_timestamp(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_seconds: int = 0,
        lookahead_seconds: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """Get all data types at a specific timestamp with optional lookback/lookahead.

        This is the primary interface for market simulator integration.

        Args:
            symbol: Symbol to get data for
            timestamp: Point-in-time to query
            lookback_seconds: Seconds of historical data to include
            lookahead_seconds: Seconds of future data (for execution simulation)

        Returns:
            Dict mapping data types to DataFrames with requested time range
        """
        self._current_timestamp = timestamp

        if self._mode == DataMode.HISTORICAL:
            return self._get_historical_data(
                symbol, timestamp, lookback_seconds, lookahead_seconds
            )
        elif self._mode == DataMode.LIVE:
            return self._get_live_data(symbol, timestamp, lookback_seconds)
        else:  # HYBRID
            return self._get_hybrid_data(
                symbol, timestamp, lookback_seconds, lookahead_seconds
            )

    def _get_historical_data(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_seconds: int,
        lookahead_seconds: int,
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data around a timestamp."""
        if not self.historical_provider:
            return {}

        start_time = timestamp - pd.Timedelta(seconds=lookback_seconds)
        end_time = timestamp + pd.Timedelta(seconds=lookahead_seconds)

        result = {}

        # Get all data types
        for timeframe in ["1s", "1m", "5m"]:
            bars = self.historical_provider.get_bars(
                symbol, timeframe, start_time, end_time
            )
            if not bars.empty:
                result[f"bars_{timeframe}"] = bars

        trades = self.historical_provider.get_trades(symbol, start_time, end_time)
        if not trades.empty:
            result["trades"] = trades

        quotes = self.historical_provider.get_quotes(symbol, start_time, end_time)
        if not quotes.empty:
            result["quotes"] = quotes

        status = self.historical_provider.get_status(symbol, start_time, end_time)
        if not status.empty:
            result["status"] = status

        return result

    def _get_live_data(
        self, symbol: str, timestamp: datetime, lookback_seconds: int
    ) -> Dict[str, pd.DataFrame]:
        """Get live data with historical lookback."""
        if not self.live_provider:
            return {}

        # For live mode, we need to maintain a rolling buffer of recent data
        # This would be implemented based on the specific live provider
        # For now, return empty as this requires live data infrastructure
        return {}

    def _get_hybrid_data(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_seconds: int,
        lookahead_seconds: int,
    ) -> Dict[str, pd.DataFrame]:
        """Get data in hybrid mode, transitioning from historical to live."""
        # Determine if we're in historical or live territory
        if self._transition_timestamp and timestamp >= self._transition_timestamp:
            # We're in live mode now
            return self._get_live_data(symbol, timestamp, lookback_seconds)
        else:
            # Still in historical mode
            return self._get_historical_data(
                symbol, timestamp, lookback_seconds, lookahead_seconds
            )

    def set_transition_point(self, timestamp: datetime):
        """Set the timestamp where historical transitions to live data."""
        self._transition_timestamp = timestamp

        # Pre-load historical data for smooth transition
        if self.historical_provider and self.live_provider:
            # Load last N days of historical data for context
            # This would be used by live provider for indicator calculations
            pass

    def get_latest_data(self, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """Get the most recent data available."""
        if self._mode == DataMode.LIVE and self.live_provider:
            if data_type == "trades":
                return pd.DataFrame([self.live_provider.get_latest_trade(symbol)])
            elif data_type == "quotes":
                return pd.DataFrame([self.live_provider.get_latest_quote(symbol)])
            elif data_type.startswith("bars_"):
                timeframe = data_type.split("_")[1]
                return pd.DataFrame(
                    [self.live_provider.get_latest_bar(symbol, timeframe)]
                )
        return None

    def subscribe(self, symbols: List[str], data_types: List[str]):
        """Subscribe to live data updates."""
        if self.live_provider:
            self.live_provider.subscribe(symbols, data_types)

    def unsubscribe(self, symbols: List[str], data_types: List[str]):
        """Unsubscribe from live data updates."""
        if self.live_provider:
            self.live_provider.unsubscribe(symbols, data_types)

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol metadata from appropriate provider."""
        if self.historical_provider:
            return self.historical_provider.get_symbol_info(symbol)
        elif self.live_provider:
            return self.live_provider.get_symbol_info(symbol)
        return {}

    def get_available_symbols(self) -> List[str]:
        """Get available symbols from appropriate provider."""
        symbols = set()

        if self.historical_provider:
            symbols.update(self.historical_provider.get_available_symbols())
        if self.live_provider:
            symbols.update(self.live_provider.get_available_symbols())

        return sorted(list(symbols))

    def close(self):
        """Close all providers."""
        if self.historical_provider:
            # HistoricalDataProvider doesn't have close method, but we could add it
            pass
        if self.live_provider:
            self.live_provider.close()
