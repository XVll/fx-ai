# data/providers/databento/databento_live_sim_provider.py
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import threading
import time

from data.provider.data_provider import LiveDataProvider, HistoricalDataProvider


class DabentoLiveSimProvider(LiveDataProvider):
    """
    Simulates a live data provider using historical data from a Databento provider.
    Useful for testing and development.
    """

    def __init__(self, historical_provider: HistoricalDataProvider,
                 replay_speed: float = 1.0, start_time: Union[str, datetime] = None):
        """
        Initialize the simulated live provider.

        Args:
            historical_provider: HistoricalDataProvider to source data from
            replay_speed: Speed multiplier for replay (1.0 = real-time, 2.0 = 2x speed)
            start_time: Optional start time for the simulation
        """
        self.historical_provider = historical_provider
        self.replay_speed = replay_speed

        # Set default.yaml start time if not provided
        if start_time is None:
            self.start_time = datetime.now() - timedelta(hours=1)
        else:
            self.start_time = start_time if isinstance(start_time, datetime) else pd.Timestamp(start_time).to_pydatetime()

        # Current simulation time
        self.current_time = self.start_time

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

        # Simulation control
        self._running = False
        self._sim_thread = None

        # Data cache
        self._data_cache = {}

    def _load_data(self, symbol: str, data_types: List[str],
                   start_time: datetime, end_time: datetime) -> Dict:
        """Load data for a symbol into the cache."""
        cache_key = f"{symbol}_{start_time.isoformat()}_{end_time.isoformat()}"

        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        data = {}

        for dt in data_types:
            if dt == "trades":
                data["trades"] = self.historical_provider.get_trades(
                    symbol, start_time, end_time
                )
            elif dt == "quotes":
                data["quotes"] = self.historical_provider.get_quotes(
                    symbol, start_time, end_time
                )
            elif dt.startswith("bars_"):
                timeframe = dt.split("_")[1]
                data[dt] = self.historical_provider.get_bars(
                    symbol, timeframe, start_time, end_time
                )
            elif dt == "status":
                data["status"] = self.historical_provider.get_status(
                    symbol, start_time, end_time
                )

        self._data_cache[cache_key] = data
        return data

    def _simulate_data_feed(self):
        """Main simulation loop that replays historical data."""
        # Determine end time (e.g., current time + 1 hour)
        end_time = self.current_time + timedelta(hours=1)

        # Load data for all subscribed symbols and data types
        all_data = {}
        for symbol in self.subscribed_symbols:
            all_data[symbol] = self._load_data(
                symbol,
                list(self.subscribed_data_types),
                self.current_time,
                end_time
            )

        # Create a consolidated timeline of all events
        events = []

        for symbol, data_dict in all_data.items():
            for data_type, df in data_dict.items():
                if df.empty:
                    continue

                # Reset index to get timestamp as a column
                df_events = df.reset_index()
                df_events['symbol'] = symbol
                df_events['data_type'] = data_type

                for _, row in df_events.iterrows():
                    event_time = row['ts_event'] if 'ts_event' in row else row.index
                    events.append({
                        'time': event_time,
                        'symbol': symbol,
                        'data_type': data_type,
                        'data': row.to_dict()
                    })

        # Sort events by time
        events.sort(key=lambda x: x['time'])

        # Start the simulation
        real_start_time = datetime.now()
        sim_start_time = self.current_time

        last_event_real_time = real_start_time
        last_event_sim_time = sim_start_time

        for event in events:
            if not self._running:
                break

            # Calculate the appropriate delay
            event_sim_time = event['time']
            sim_time_delta = (event_sim_time - last_event_sim_time).total_seconds()
            real_time_delta = sim_time_delta / self.replay_speed

            # Sleep until it's time for this event
            time_to_sleep = max(0, real_time_delta - (datetime.now() - last_event_real_time).total_seconds())
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            # Process the event
            self._process_event(event)

            # Update the last event times
            last_event_real_time = datetime.now()
            last_event_sim_time = event_sim_time

            # Update current simulation time
            self.current_time = event_sim_time

    def _process_event(self, event):
        """Process a simulated data event."""
        symbol = event['symbol']
        data_type = event['data_type']
        data = event['data']

        if data_type == "trades":
            trade_data = {
                'type': 'trade',
                'symbol': symbol,
                'price': data.get('price', 0),
                'size': data.get('size', 0),
                'timestamp': data.get('ts_event', event['time']),
                'side': data.get('side', 'N'),
                'raw_message': data
            }

            self._latest_trades[symbol] = trade_data
            for callback in self.trade_callbacks:
                callback(trade_data)

        elif data_type == "quotes":
            quote_data = {
                'type': 'quote',
                'symbol': symbol,
                'bid_price': data.get('bid_px_00', 0),
                'ask_price': data.get('ask_px_00', 0),
                'bid_size': data.get('bid_sz_00', 0),
                'ask_size': data.get('ask_sz_00', 0),
                'timestamp': data.get('ts_event', event['time']),
                'raw_message': data
            }

            self._latest_quotes[symbol] = quote_data
            for callback in self.quote_callbacks:
                callback(quote_data)

        elif data_type.startswith("bars_"):
            timeframe = data_type.split("_")[1]
            bar_data = {
                'type': 'bar',
                'symbol': symbol,
                'timeframe': timeframe,
                'open': data.get('open', 0),
                'high': data.get('high', 0),
                'low': data.get('low', 0),
                'close': data.get('close', 0),
                'volume': data.get('volume', 0),
                'timestamp': data.get('ts_event', event['time']),
                'raw_message': data
            }

            bar_key = f"{symbol}_{timeframe}"
            self._latest_bars[bar_key] = bar_data
            for callback in self.bar_callbacks:
                callback(bar_data)

        elif data_type == "status":
            status_data = {
                'type': 'status',
                'symbol': symbol,
                'action': data.get('action', 0),
                'reason': data.get('reason', 0),
                'is_trading': data.get('is_trading', 'Y'),
                'is_quoting': data.get('is_quoting', 'Y'),
                'is_short_sell_restricted': data.get('is_short_sell_restricted', 'N'),
                'timestamp': data.get('ts_event', event['time']),
                'raw_message': data
            }

            self._latest_status[symbol] = status_data
            for callback in self.status_callbacks:
                callback(status_data)

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        return self.historical_provider.get_symbol_info(symbol)

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        return self.historical_provider.get_available_symbols()

    def subscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Subscribe to live data for symbols."""
        # Update subscription state
        self.subscribed_symbols.update(symbols)
        self.subscribed_data_types.update(data_types)

        # Start the simulation if not already running
        if not self._running:
            self._running = True
            self._sim_thread = threading.Thread(target=self._simulate_data_feed)
            self._sim_thread.daemon = True
            self._sim_thread.start()

    def unsubscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Unsubscribe from live data for symbols."""
        # Remove from subscription state
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)

        for data_type in data_types:
            self.subscribed_data_types.discard(data_type)

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
        """Stop the simulation and cleanup resources."""
        self._running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)

        self.subscribed_symbols.clear()
        self.subscribed_data_types.clear()
        self._data_cache.clear()