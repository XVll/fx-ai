# market_simulator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
from collections import deque


class MarketSimulator:
    """
    Market simulator that manages all market data types and provides synchronized
    access as we step through time on a 1-second granularity.
    """

    def __init__(self, symbol: str, mode: str = 'backtesting',
                 start_time: Optional[str] = None, end_time: Optional[str] = None,
                 logger=None):
        """
        Initialize the market simulator.

        Args:
            symbol: Trading symbol
            mode: 'backtesting' or 'live'
            start_time: Start time for historical data
            end_time: End time for historical data
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.mode = mode

        # Convert string times to datetime if provided
        self.start_time = pd.to_datetime(start_time) if start_time else None
        self.end_time = pd.to_datetime(end_time) if end_time else None

        # Data storage
        # Bar data at different timeframes
        self.data_1s = None  # Primary 1-second bars
        self.data_5m = None  # 5-minute bars
        self.data_1d = None  # Daily bars

        # Market microstructure data
        self.data_quotes = None  # Quotes data (bid/ask)
        self.data_trades = None  # Trades data (tape)

        # Current position in the data
        self.current_index = 0
        self.current_timestamp = None
        self.prev_timestamp = None

        # Rolling window buffers for bar data
        self.buffer_1s = deque(maxlen=600)  # 600 seconds (10 minutes)
        self.buffer_5m = deque(maxlen=78)  # Full trading day (~6.5 hours)
        self.buffer_1d = deque(maxlen=90)  # 90 days

        # Consolidated data for the current second
        self.current_second_data = {
            'bar': None,  # The current 1-second bar
            'quotes': [],  # All quotes in this second
            'trades': [],  # All trades in this second
        }

        # Load data if in backtesting mode
        if mode == 'backtesting':
            self._load_data()
            self._initialize_buffers()

    def _load_data(self):
        """
        Load historical data for all required data types and timeframes.
        """
        try:
            from data.data_manager import DataManager
            from data.provider.data_bento.databento_file_provider import DabentoFileProvider

            # Initialize data provider and manager
            provider = DabentoFileProvider(data_dir="./data")
            data_manager = DataManager(provider, logger=self.logger)

            # Load bar data at different timeframes
            self.data_1s = data_manager.get_bars(
                symbol=self.symbol,
                timeframe='1s',
                start_time=self.start_time,
                end_time=self.end_time
            )

            # Try to load 5m data
            try:
                self.data_5m = data_manager.get_bars(
                    symbol=self.symbol,
                    timeframe='5m',
                    start_time=self.start_time,
                    end_time=self.end_time
                )
            except Exception as e:
                self.logger.warning(f"Could not load 5-minute data: {e}")

            # Load daily data (go back further for S/R levels)
            sr_start_time = self.start_time - timedelta(days=90) if self.start_time else None
            self.data_1d = data_manager.get_bars(
                symbol=self.symbol,
                timeframe='1d',
                start_time=sr_start_time,
                end_time=self.end_time
            )

            # Load quote data
            try:
                self.data_quotes = data_manager.get_quotes(
                    symbol=self.symbol,
                    start_time=self.start_time,
                    end_time=self.end_time
                )
            except Exception as e:
                self.logger.warning(f"Could not load quotes data: {e}")

            # Load trades data
            try:
                self.data_trades = data_manager.get_trades(
                    symbol=self.symbol,
                    start_time=self.start_time,
                    end_time=self.end_time
                )
            except Exception as e:
                self.logger.warning(f"Could not load trades data: {e}")

            # Verify we have the essential 1-second data
            if self.data_1s is None or self.data_1s.empty:
                self.logger.error(f"No 1-second data available for {self.symbol}")
                raise ValueError(f"No 1-second data available for {self.symbol}")

            # Initialize timestamp from the 1-second data
            self.current_timestamp = self.data_1s.index[0]

            # Log data sizes
            data_sizes = {
                "1s bars": len(self.data_1s) if self.data_1s is not None else 0,
                "5m bars": len(self.data_5m) if self.data_5m is not None else 0,
                "1d bars": len(self.data_1d) if self.data_1d is not None else 0,
                "quotes": len(self.data_quotes) if self.data_quotes is not None else 0,
                "trades": len(self.data_trades) if self.data_trades is not None else 0
            }
            self.logger.info(f"Loaded data for {self.symbol}: {data_sizes}")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _initialize_buffers(self):
        """
        Initialize rolling window buffers with historical data.
        """
        # Clear all buffers
        self.buffer_1s.clear()
        self.buffer_5m.clear()
        self.buffer_1d.clear()

        # Set starting position
        # We start a bit into the data to have history for feature calculations
        if self.data_1s is not None and not self.data_1s.empty:
            # Start at least 600 bars in (10 minutes) if possible
            buffer_start_idx = min(600, len(self.data_1s) // 2)
            self.current_index = buffer_start_idx
            self.current_timestamp = self.data_1s.index[self.current_index]

            # Fill 1-second buffer with historical data
            for i in range(self.current_index - min(self.current_index, self.buffer_1s.maxlen),
                           self.current_index + 1):
                if i >= 0 and i < len(self.data_1s):
                    # Get the 1-second bar
                    bar_data = self.data_1s.iloc[i].to_dict()
                    bar_timestamp = self.data_1s.index[i]
                    bar_data['timestamp'] = bar_timestamp

                    # Create a state object for this second
                    second_data = {
                        'bar': bar_data,
                        'timestamp': bar_timestamp,
                        'quotes': [],  # Will be filled below
                        'trades': []  # Will be filled below
                    }

                    # Get quotes for this 1-second window
                    if self.data_quotes is not None and not self.data_quotes.empty:
                        # Define the 1-second window
                        window_start = bar_timestamp
                        window_end = window_start + pd.Timedelta(seconds=1)

                        # Get quotes in this window
                        window_quotes = self.data_quotes[
                            (self.data_quotes.index >= window_start) &
                            (self.data_quotes.index < window_end)
                            ]

                        if not window_quotes.empty:
                            for q_idx, q_row in window_quotes.iterrows():
                                quote = q_row.to_dict()
                                quote['timestamp'] = q_idx
                                second_data['quotes'].append(quote)

                    # Get trades for this 1-second window
                    if self.data_trades is not None and not self.data_trades.empty:
                        # Get trades in this window
                        window_start = bar_timestamp
                        window_end = window_start + pd.Timedelta(seconds=1)

                        window_trades = self.data_trades[
                            (self.data_trades.index >= window_start) &
                            (self.data_trades.index < window_end)
                            ]

                        if not window_trades.empty:
                            for t_idx, t_row in window_trades.iterrows():
                                trade = t_row.to_dict()
                                trade['timestamp'] = t_idx
                                second_data['trades'].append(trade)

                    # Add to buffer
                    self.buffer_1s.append(second_data)

            # Fill 5-minute buffer with historical data if available
            if self.data_5m is not None and not self.data_5m.empty:
                # Find all 5m bars up to current timestamp
                relevant_5m = self.data_5m[self.data_5m.index <= self.current_timestamp]
                recent_5m = relevant_5m.tail(self.buffer_5m.maxlen)

                for idx, row in recent_5m.iterrows():
                    bar_data = row.to_dict()
                    bar_data['timestamp'] = idx
                    self.buffer_5m.append(bar_data)

            # Fill daily buffer with historical data if available
            if self.data_1d is not None and not self.data_1d.empty:
                # Find all daily bars up to current timestamp
                relevant_1d = self.data_1d[self.data_1d.index <= self.current_timestamp]
                recent_1d = relevant_1d.tail(self.buffer_1d.maxlen)

                for idx, row in recent_1d.iterrows():
                    bar_data = row.to_dict()
                    bar_data['timestamp'] = idx
                    self.buffer_1d.append(bar_data)

            # Set current second data
            if self.buffer_1s:
                self.current_second_data = self.buffer_1s[-1]

            self.logger.info(f"Initialized buffers with historical data: "
                             f"1s={len(self.buffer_1s)}, 5m={len(self.buffer_5m)}, "
                             f"1d={len(self.buffer_1d)}")

    def step(self):
        """
        Advance the simulator to the next 1-second data point.

        Returns:
            bool: True if successful, False if at the end of data
        """
        if self.is_done():
            return False

        # Save current timestamp as previous
        self.prev_timestamp = self.current_timestamp

        # Advance to next timestep
        self.current_index += 1

        # Update current timestamp
        if self.current_index < len(self.data_1s):
            self.current_timestamp = self.data_1s.index[self.current_index]

            # Get the 1-second bar
            bar_data = self.data_1s.iloc[self.current_index].to_dict()
            bar_data['timestamp'] = self.current_timestamp

            # Create a state object for this second
            second_data = {
                'bar': bar_data,
                'timestamp': self.current_timestamp,
                'quotes': [],  # Will be filled below
                'trades': []  # Will be filled below
            }

            # Get quotes for this 1-second window
            if self.data_quotes is not None and not self.data_quotes.empty:
                # Define the 1-second window
                window_start = self.current_timestamp
                window_end = window_start + pd.Timedelta(seconds=1)

                # Get quotes in this window
                window_quotes = self.data_quotes[
                    (self.data_quotes.index >= window_start) &
                    (self.data_quotes.index < window_end)
                    ]

                if not window_quotes.empty:
                    for idx, row in window_quotes.iterrows():
                        quote = row.to_dict()
                        quote['timestamp'] = idx
                        second_data['quotes'].append(quote)

            # Get trades for this 1-second window
            if self.data_trades is not None and not self.data_trades.empty:
                # Get trades in this window
                window_start = self.current_timestamp
                window_end = window_start + pd.Timedelta(seconds=1)

                window_trades = self.data_trades[
                    (self.data_trades.index >= window_start) &
                    (self.data_trades.index < window_end)
                    ]

                if not window_trades.empty:
                    for idx, row in window_trades.iterrows():
                        trade = row.to_dict()
                        trade['timestamp'] = idx
                        second_data['trades'].append(trade)

            # Update current second data
            self.current_second_data = second_data

            # Add to 1-second buffer
            self.buffer_1s.append(second_data)

            # Update 5-minute buffer if needed
            if self.data_5m is not None and not self.data_5m.empty:
                # Find 5-minute bars at or before current timestamp that aren't in the buffer
                latest_5m_in_buffer = self.buffer_5m[-1]['timestamp'] if self.buffer_5m else pd.Timestamp.min

                # Get new 5-min bars
                new_5m_bars = self.data_5m[
                    (self.data_5m.index > latest_5m_in_buffer) &
                    (self.data_5m.index <= self.current_timestamp)
                    ]

                # Add any new bars
                for idx, row in new_5m_bars.iterrows():
                    bar_data = row.to_dict()
                    bar_data['timestamp'] = idx
                    self.buffer_5m.append(bar_data)

            # Update daily buffer if needed
            if self.data_1d is not None and not self.data_1d.empty:
                # Find new daily bars
                latest_1d_in_buffer = self.buffer_1d[-1]['timestamp'] if self.buffer_1d else pd.Timestamp.min

                new_1d_bars = self.data_1d[
                    (self.data_1d.index > latest_1d_in_buffer) &
                    (self.data_1d.index <= self.current_timestamp)
                    ]

                # Add any new bars
                for idx, row in new_1d_bars.iterrows():
                    bar_data = row.to_dict()
                    bar_data['timestamp'] = idx
                    self.buffer_1d.append(bar_data)

            return True
        else:
            return False

    def reset(self, options=None):
        """
        Reset the simulator to the initial state or a random state.

        Args:
            options: Dict with reset options, such as:
                - random_start: True to start at a random point (default: False)
                - max_steps: Max steps for the episode
        """
        options = options or {}

        # Check if we should start at a random point
        random_start = options.get('random_start', False)

        if random_start and self.data_1s is not None and len(self.data_1s) > 0:
            # Start at a random point, leaving enough room for a full episode
            max_start_idx = max(0, len(self.data_1s) - options.get('max_steps', 500) - 1)
            if max_start_idx > 0:
                # Start at least buffer_size in to have enough history for features
                min_start_idx = min(600, max_start_idx // 2)
                self.current_index = np.random.randint(min_start_idx, max_start_idx)
                self.current_timestamp = self.data_1s.index[self.current_index]
            else:
                # Not enough data for random start with buffer
                self.current_index = min(600, len(self.data_1s) - 1)
                self.current_timestamp = self.data_1s.index[self.current_index]
        else:
            # Start at the beginning + buffer size
            self.current_index = min(600, len(self.data_1s) - 1)
            if self.data_1s is not None and len(self.data_1s) > 0:
                self.current_timestamp = self.data_1s.index[self.current_index]

        # Re-initialize buffers
        self._initialize_buffers()

        return True

    def is_done(self):
        """
        Check if we've reached the end of data.

        Returns:
            bool: True if done, False otherwise
        """
        return self.data_1s is None or self.current_index >= len(self.data_1s) - 1

    def get_current_market_state(self):
        """
        Get the current market state with all data necessary for feature extraction.

        Returns:
            dict: Current market state with all the raw data
        """
        if not self.current_second_data:
            return None

        # Start with the current 1-second data
        state = {
            'timestamp': self.current_timestamp,
            'symbol': self.symbol,

            # Current 1-second bar data
            'current_bar': self.current_second_data['bar'],

            # All trades and quotes for the current second
            'current_second_trades': self.current_second_data['trades'],
            'current_second_quotes': self.current_second_data['quotes'],

            # Buffers for feature calculation
            'buffer_1s': list(self.buffer_1s),  # Last 600 seconds of data (for HF features)
            'buffer_5m': list(self.buffer_5m),  # Last 78 5-minute bars (for LF features)
            'buffer_1d': list(self.buffer_1d),  # Last 90 daily bars (for S/R levels)
        }

        # Add computed buffers for MF/LF features
        # These can be derived from the 1s buffer
        state['rolling_1m'] = self._compute_rolling_1m_from_1s()
        state['rolling_5m'] = self._compute_rolling_5m_from_1s()

        return state

    def _compute_rolling_1m_from_1s(self):
        """
        Compute rolling 1-minute bars from 1-second data in the buffer.

        Returns:
            list: List of 1-minute bars
        """
        if len(self.buffer_1s) < 60:
            return []

        # Get the 1-second bars from the buffer
        bar_list = [item['bar'] for item in self.buffer_1s if item.get('bar')]

        # Group by minute
        minute_groups = {}
        for bar in bar_list:
            timestamp = bar['timestamp']
            minute_key = timestamp.floor('Min')

            if minute_key not in minute_groups:
                minute_groups[minute_key] = []

            minute_groups[minute_key].append(bar)

        # Compute 1-minute bars from 1-second bars
        one_minute_bars = []
        for minute_key, bars in sorted(minute_groups.items()):
            if len(bars) > 0:
                # Aggregate OHLCV data
                first_bar = bars[0]
                last_bar = bars[-1]
                high = max(bar.get('high', bar.get('close', 0)) for bar in bars)
                low = min(bar.get('low', bar.get('close', 0)) for bar in bars)
                volume = sum(bar.get('volume', 0) for bar in bars)

                one_minute_bar = {
                    'timestamp': minute_key,
                    'open': first_bar.get('open', first_bar.get('close', 0)),
                    'high': high,
                    'low': low,
                    'close': last_bar.get('close', 0),
                    'volume': volume
                }
                one_minute_bars.append(one_minute_bar)

        # Return the last 30 1-minute bars (if available)
        return one_minute_bars[-30:] if len(one_minute_bars) >= 30 else one_minute_bars

    def _compute_rolling_5m_from_1s(self):
        """
        Compute rolling 5-minute bars from the computed 1-minute bars.

        Returns:
            list: List of 5-minute bars
        """
        # First get 1-minute bars
        one_minute_bars = self._compute_rolling_1m_from_1s()

        if len(one_minute_bars) < 5:
            # If 1-minute data is insufficient, use loaded 5-minute data if available
            return list(self.buffer_5m) if self.buffer_5m else []

        # Group by 5-minute interval
        five_min_groups = {}
        for bar in one_minute_bars:
            timestamp = bar['timestamp']
            five_min_key = timestamp.floor('5Min')

            if five_min_key not in five_min_groups:
                five_min_groups[five_min_key] = []

            five_min_groups[five_min_key].append(bar)

        # Compute 5-minute bars from 1-minute bars
        five_minute_bars = []
        for five_min_key, bars in sorted(five_min_groups.items()):
            if len(bars) > 0:
                # Aggregate OHLCV data
                first_bar = bars[0]
                last_bar = bars[-1]
                high = max(bar['high'] for bar in bars)
                low = min(bar['low'] for bar in bars)
                volume = sum(bar['volume'] for bar in bars)

                five_minute_bar = {
                    'timestamp': five_min_key,
                    'open': first_bar['open'],
                    'high': high,
                    'low': low,
                    'close': last_bar['close'],
                    'volume': volume
                }
                five_minute_bars.append(five_minute_bar)

        # Return the last 30 5-minute bars (if available)
        return five_minute_bars[-30:] if len(five_minute_bars) >= 30 else five_minute_bars

    def close(self):
        """Clean up resources."""
        # Close data providers or any open resources
        pass