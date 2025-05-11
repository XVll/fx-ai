# data/providers/databento/databento_live_provider.py
import databento as db
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import queue
import threading

from data.provider.data_provider import LiveDataProvider


class DabentoLiveProvider(LiveDataProvider):
    """Implementation of Live Provider using Databento's real-time API."""

    def __init__(self, api_key: str, dataset: str = None):
        """
        Initialize the Databento live provider.

        Args:
            api_key: Databento API key
            dataset: Default dataset to use (e.g., "XNAS.ITCH")
        """
        self.client = db.Live(api_key)
        self.dataset = dataset

        # Store subscription state
        self.subscribed_symbols = set()
        self.subscribed_data_types = set()

        # Callbacks
        self.trade_callbacks = []
        self.quote_callbacks = []
        self.bar_callbacks = []
        self.status_callbacks = []

        # Latest data cache
        self._latest_trades = {}
        self._latest_quotes = {}
        self._latest_bars = {}
        self._latest_status = {}

        # Message processing thread and queue
        self._msg_queue = queue.Queue()
        self._running = False
        self._processing_thread = None

    def _start_processing_thread(self):
        """Start the message processing thread."""
        if self._processing_thread is not None and self._processing_thread.is_alive():
            return  # Already running

        self._running = True
        self._processing_thread = threading.Thread(target=self._process_messages)
        self._processing_thread.daemon = True
        self._processing_thread.start()

    def _process_messages(self):
        """Process messages from the queue and dispatch to callbacks."""
        while self._running:
            try:
                message = self._msg_queue.get(timeout=0.1)

                # Process message based on type
                if message['type'] == 'trade':
                    symbol = message['symbol']
                    self._latest_trades[symbol] = message
                    for callback in self.trade_callbacks:
                        callback(message)

                elif message['type'] == 'quote':
                    symbol = message['symbol']
                    self._latest_quotes[symbol] = message
                    for callback in self.quote_callbacks:
                        callback(message)

                elif message['type'] == 'bar':
                    symbol = message['symbol']
                    timeframe = message['timeframe']
                    key = f"{symbol}_{timeframe}"
                    self._latest_bars[key] = message
                    for callback in self.bar_callbacks:
                        callback(message)

                elif message['type'] == 'status':
                    symbol = message['symbol']
                    self._latest_status[symbol] = message
                    for callback in self.status_callbacks:
                        callback(message)

                self._msg_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")

    def _on_message(self, msg):
        """Callback for Databento real-time messages."""
        # Parse the message and put it in the queue
        parsed_msg = self._parse_message(msg)
        if parsed_msg:
            self._msg_queue.put(parsed_msg)

    def _parse_message(self, msg):
        """Parse a Databento message into a standardized format."""
        # This would need to be adapted based on Databento's actual message format
        # For now, this is a placeholder

        record_type = msg.record_type

        if record_type == db.RecordType.MBP_0:  # Trades
            return {
                'type': 'trade',
                'symbol': msg.symbol,
                'price': msg.price,
                'size': msg.size,
                'timestamp': msg.ts_event,
                'exchange_timestamp': msg.ts_recv,
                'side': msg.side,
                'raw_message': msg
            }

        elif record_type == db.RecordType.MBP_1:  # Quotes (Level 1)
            return {
                'type': 'quote',
                'symbol': msg.symbol,
                'bid_price': msg.bid_px_00,
                'ask_price': msg.ask_px_00,
                'bid_size': msg.bid_sz_00,
                'ask_size': msg.ask_sz_00,
                'timestamp': msg.ts_event,
                'exchange_timestamp': msg.ts_recv,
                'raw_message': msg
            }

        elif record_type in [db.RecordType.OHLCV_1S, db.RecordType.OHLCV_1M,
                             db.RecordType.OHLCV_1H, db.RecordType.OHLCV_1D]:
            # Map record type to timeframe
            timeframe_map = {
                db.RecordType.OHLCV_1S: "1s",
                db.RecordType.OHLCV_1M: "1m",
                db.RecordType.OHLCV_1H: "1h",
                db.RecordType.OHLCV_1D: "1d"
            }

            return {
                'type': 'bar',
                'symbol': msg.symbol,
                'timeframe': timeframe_map[record_type],
                'open': msg.open,
                'high': msg.high,
                'low': msg.low,
                'close': msg.close,
                'volume': msg.volume,
                'timestamp': msg.ts_event,
                'raw_message': msg
            }

        elif record_type == db.RecordType.STATUS:
            return {
                'type': 'status',
                'symbol': msg.symbol,
                'action': msg.action,
                'reason': msg.reason,
                'is_trading': msg.is_trading,
                'is_quoting': msg.is_quoting,
                'is_short_sell_restricted': msg.is_short_sell_restricted,
                'timestamp': msg.ts_event,
                'raw_message': msg
            }

        return None

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        # Use Historical client's metadata API for symbol info
        historical_client = db.Historical(self.client.api_key)
        symbol_metadata = historical_client.metadata.lookup_symbols(
            dataset=self.dataset,
            symbols=[symbol],
            stype_in="raw_symbol"
        )

        if not symbol_metadata:
            raise ValueError(f"Symbol {symbol} not found in dataset {self.dataset}")

        return symbol_metadata[0]

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        # Use Historical client for this
        historical_client = db.Historical(self.client.api_key)
        return historical_client.metadata.list_symbols(dataset=self.dataset)

    def subscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Subscribe to live data for symbols."""
        # Convert data types to Databento schema
        schema_map = {
            "trades": "trades",
            "quotes": "mbp-1",
            "bars_1s": "ohlcv-1s",
            "bars_1m": "ohlcv-1m",
            "bars_1h": "ohlcv-1h",
            "bars_1d": "ohlcv-1d",
            "status": "status"
        }

        # Filter out already subscribed symbols
        new_symbols = [s for s in symbols if s not in self.subscribed_symbols]

        if not new_symbols:
            return  # Nothing new to subscribe to

        # Convert data_types to schemas
        schemas = []
        for dt in data_types:
            if dt in schema_map:
                schemas.append(schema_map[dt])
                self.subscribed_data_types.add(dt)

        if not schemas:
            raise ValueError(f"No valid data types in {data_types}. Supported: {list(schema_map.keys())}")

        # Subscribe to the data
        for symbol in new_symbols:
            self.client.subscribe(
                dataset=self.dataset,
                symbols=[symbol],
                stype_in="raw_symbol",
                schemas=schemas,
                callback=self._on_message
            )
            self.subscribed_symbols.add(symbol)

        # Start the processing thread if not running
        self._start_processing_thread()

    def unsubscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Unsubscribe from live data for symbols."""
        # Convert data types to Databento schema
        schema_map = {
            "trades": "trades",
            "quotes": "mbp-1",
            "bars_1s": "ohlcv-1s",
            "bars_1m": "ohlcv-1m",
            "bars_1h": "ohlcv-1h",
            "bars_1d": "ohlcv-1d",
            "status": "status"
        }

        # Filter to only subscribed symbols
        symbols_to_unsub = [s for s in symbols if s in self.subscribed_symbols]

        if not symbols_to_unsub:
            return  # Nothing to unsubscribe from

        # Convert data_types to schemas
        schemas = []
        for dt in data_types:
            if dt in schema_map:
                schemas.append(schema_map[dt])

        if not schemas:
            raise ValueError(f"No valid data types in {data_types}. Supported: {list(schema_map.keys())}")

        # Unsubscribe from the data
        for symbol in symbols_to_unsub:
            self.client.unsubscribe(
                dataset=self.dataset,
                symbols=[symbol],
                stype_in="raw_symbol",
                schemas=schemas
            )
            if all_schemas_unsubscribed:  # This would require tracking subscriptions per symbol
                self.subscribed_symbols.remove(symbol)

    def get_latest_trade(self, symbol: str) -> Dict:
        """Get the latest trade for a symbol."""
        if symbol not in self._latest_trades:
            raise ValueError(f"No trade data available for {symbol}. Subscribe first.")

        return self._latest_trades[symbol]

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get the latest quote for a symbol."""
        if symbol not in self._latest_quotes:
            raise ValueError(f"No quote data available for {symbol}. Subscribe first.")

        return self._latest_quotes[symbol]

    def get_latest_bar(self, symbol: str, timeframe: str) -> Dict:
        """Get the latest OHLCV bar for a symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        if key not in self._latest_bars:
            raise ValueError(f"No {timeframe} bar data available for {symbol}. Subscribe first.")

        return self._latest_bars[key]

    def add_trade_callback(self, callback_fn: Callable) -> None:
        """Add callback for trade updates."""
        self.trade_callbacks.append(callback_fn)

    def add_quote_callback(self, callback_fn: Callable) -> None:
        """Add callback for quote updates."""
        self.quote_callbacks.append(callback_fn)

    def add_bar_callback(self, callback_fn: Callable) -> None:
        """Add callback for bar updates."""
        self.bar_callbacks.append(callback_fn)

    def add_status_callback(self, callback_fn: Callable) -> None:
        """Add callback for status updates."""
        self.status_callbacks.append(callback_fn)

    def close(self):
        """Close the live data connection and cleanup resources."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

        # Unsubscribe from all
        if self.subscribed_symbols:
            self.client.unsubscribe_all()
            self.subscribed_symbols.clear()
            self.subscribed_data_types.clear()

        # Close the client connection if needed
        if hasattr(self.client, 'close'):
            self.client.close()