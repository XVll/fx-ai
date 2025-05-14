# market_simulator_v2.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from collections import deque


class MarketSimulatorV2:
    """
    Enhanced Market Simulator for 1-second granularity, handling historical and live data,
    and providing robust state for rolling feature calculations.
    """

    def __init__(self, symbol: str,
                 historical_1s_data: Optional[pd.DataFrame] = None,
                 historical_trades_data: Optional[pd.DataFrame] = None,
                 historical_quotes_data: Optional[pd.DataFrame] = None,
                 historical_5m_bars: Optional[pd.DataFrame] = None,  # For S/R levels
                 historical_1d_bars: Optional[pd.DataFrame] = None,  # For S/R levels
                 config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the market simulator.

        Args:
            symbol: Trading symbol.
            historical_1s_data: DataFrame with 1-second OHLCV bars. Expected index: DateTimeIndex.
                                Columns: ['open', 'high', 'low', 'close', 'volume'].
            historical_trades_data: DataFrame with individual trades. Expected index: DateTimeIndex.
                                   Columns: ['price', 'size', 'side', ...].
            historical_quotes_data: DataFrame with quotes. Expected index: DateTimeIndex.
                                   Columns: ['bid_price', 'bid_size', 'ask_price', 'ask_size', ...].
            historical_5m_bars: DataFrame with 5-minute bars (for S/R). Index: DateTimeIndex.
            historical_1d_bars: DataFrame with daily bars (for S/R). Index: DateTimeIndex.
            config: Configuration dictionary. Relevant keys:
                - 'mode': 'backtesting' or 'live'.
                - 'start_time_str': ISO format string for backtesting start.
                - 'end_time_str': ISO format string for backtesting end.
                - 'max_1s_data_window_size': Max number of 1-second data events to keep for features (e.g., 3600 for 1 hour).
                - 'initial_buffer_seconds': Seconds of data to pre-fill in buffers (e.g., 600 for 10 mins).
            logger: Optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.symbol = symbol
        self.config = config or {}

        self.mode = self.config.get('mode', 'backtesting')
        self.start_time = pd.to_datetime(self.config.get('start_time_str')) if self.config.get(
            'start_time_str') else None
        self.end_time = pd.to_datetime(self.config.get('end_time_str')) if self.config.get('end_time_str') else None

        # --- Data Storage ---
        self.all_1s_data: Optional[pd.DataFrame] = self._prepare_historical_data(historical_1s_data)
        self.all_trades_data: Optional[pd.DataFrame] = self._prepare_historical_data(historical_trades_data)
        self.all_quotes_data: Optional[pd.DataFrame] = self._prepare_historical_data(historical_quotes_data)

        # Data for S/R levels (typically less frequent updates needed for buffers)
        self.historical_5m_bars_for_sr: Optional[pd.DataFrame] = self._prepare_historical_data(historical_5m_bars)
        self.historical_1d_bars_for_sr: Optional[pd.DataFrame] = self._prepare_historical_data(historical_1d_bars)

        # --- Current Time and Index ---
        self.current_timestamp: Optional[datetime] = None
        self._current_1s_data_idx: int = 0  # Index for self.all_1s_data

        # --- Rolling Data Windows for Features ---
        # This window holds combined 1s data events (bar, trades in that sec, quotes in that sec)
        # Each element could be a dict: {'timestamp': ..., 'bar': ..., 'trades': [...], 'quotes': [...]}
        self.max_1s_data_window_size = self.config.get('max_1s_data_window_size', 60 * 60)  # Default: 1 hour of 1s data
        self.current_1s_data_window: deque = deque(maxlen=self.max_1s_data_window_size)

        # Buffers for S/R calculation (can be simpler, e.g., just the DFs sliced up to current_timestamp)
        # Or deques if rolling S/R calculation is needed over many days/weeks.
        # For simplicity, we'll assume S/R calculations can use slices of the DFs.

        # --- Live Data Ingestion Buffers (processed each second) ---
        self._live_trades_buffer: List[Dict] = []
        self._live_quotes_buffer: List[Dict] = []
        self._live_1s_bar_events: List[Dict] = []  # If 1s bars also come via WS

        # --- Initialization ---
        if self.mode == 'backtesting':
            if self.all_1s_data is None or self.all_1s_data.empty:
                raise ValueError("Historical 1s data is required for backtesting mode.")
            self._initialize_for_backtesting()
        elif self.mode == 'live':
            self.current_timestamp = datetime.now().replace(microsecond=0)  # Or from an external time source
            # Live mode might require pre-loading some recent historical data to fill buffers
            self.logger.info(f"MarketSimulatorV2 initialized for LIVE trading. Current time: {self.current_timestamp}")

        self.logger.info(f"MarketSimulatorV2 initialized for {self.symbol}. Mode: {self.mode}")

    def _prepare_historical_data(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be a DatetimeIndex.")
            df = df.sort_index()  # Ensure data is chronologically sorted
            if self.start_time:
                df = df[df.index >= self.start_time]
            if self.end_time:
                df = df[df.index <= self.end_time]
            return df
        return None

    def _initialize_for_backtesting(self):
        initial_buffer_seconds = self.config.get('initial_buffer_seconds', 600)  # 10 minutes

        # Set start index to allow for initial buffer population
        # Find the timestamp in all_1s_data that is `initial_buffer_seconds` after the first available timestamp
        first_valid_timestamp = self.all_1s_data.index[0]
        self.current_timestamp = first_valid_timestamp

        # Find the index corresponding to the initial buffer end time
        # This ensures we have `initial_buffer_seconds` of data *before* the first `current_timestamp` used by the agent
        # The agent will effectively start trading at `current_timestamp` after the loop.

        temp_start_time_for_buffer = self.all_1s_data.index[0]
        initial_target_timestamp = temp_start_time_for_buffer + timedelta(seconds=initial_buffer_seconds)

        # Find the closest index to this target timestamp for starting the simulation
        # self._current_1s_data_idx will be the index of the *first* data point the agent sees.
        # The buffer will be filled with data *before* this index.

        potential_start_indices = self.all_1s_data.index.searchsorted(initial_target_timestamp)
        self._current_1s_data_idx = min(potential_start_indices, len(self.all_1s_data) - 1)

        if self._current_1s_data_idx == 0 and initial_buffer_seconds > 0:
            self.logger.warning(
                f"Not enough data for full initial_buffer_seconds ({initial_buffer_seconds}). Starting with available data.")

        # Pre-fill the current_1s_data_window
        start_fill_idx = max(0, self._current_1s_data_idx - self.max_1s_data_window_size)
        for i in range(start_fill_idx, self._current_1s_data_idx):
            ts = self.all_1s_data.index[i]
            one_sec_event = self._get_data_for_timestamp(ts, is_initial_fill=True)
            if one_sec_event:
                self.current_1s_data_window.append(one_sec_event)

        # Set the current_timestamp to the first data point the agent will process
        if self._current_1s_data_idx < len(self.all_1s_data):
            self.current_timestamp = self.all_1s_data.index[self._current_1s_data_idx]
        else:  # Reached end of data during initialization
            self.current_timestamp = self.all_1s_data.index[-1] if not self.all_1s_data.empty else None
            self.logger.warning("Reached end of data during initialization.")

        self.logger.info(
            f"Backtesting initialized. Buffer filled with {len(self.current_1s_data_window)} events. Starting at index {self._current_1s_data_idx}, timestamp: {self.current_timestamp}")

    def _get_data_for_timestamp(self, timestamp: datetime, is_initial_fill: bool = False) -> Optional[Dict[str, Any]]:
        """
        Consolidates all market data for a given 1-second interval.
        For backtesting, this fetches from historical data.
        For live, this would incorporate processed live events for this second.
        """
        if self.all_1s_data is None:
            return None

        # Get 1-second bar
        try:
            # For initial fill, we iterate using index `i`. For step, `_current_1s_data_idx` points to current bar.
            # Here we use timestamp to find the bar.
            bar_data_series = self.all_1s_data.loc[timestamp] if timestamp in self.all_1s_data.index else None
            current_1s_bar = bar_data_series.to_dict() if bar_data_series is not None else None
        except KeyError:  # Timestamp might not exist if there are gaps
            current_1s_bar = None

        if not current_1s_bar and not is_initial_fill and self.mode == 'backtesting':
            # In backtesting, if a 1s bar is missing at the current step, it's a gap.
            # We might want to forward-fill or handle it, but for now, log and skip.
            # self.logger.debug(f"No 1s bar data for {timestamp} at index {self._current_1s_data_idx}")
            # For robustness, let's try to get the *previous* bar if current is missing
            # and it's not the very first data point. This helps with feature continuity.
            if self._current_1s_data_idx > 0:
                prev_ts = self.all_1s_data.index[self._current_1s_data_idx - 1]
                bar_data_series = self.all_1s_data.loc[prev_ts] if prev_ts in self.all_1s_data.index else None
                current_1s_bar = bar_data_series.to_dict() if bar_data_series is not None else None
                if current_1s_bar: current_1s_bar['timestamp_original_missing'] = True  # Flag it
            else:
                return None  # Truly no bar data possible

        # Define the 1-second window for trades and quotes
        window_start = timestamp
        window_end = timestamp + timedelta(seconds=1)  # Exclusive end

        # Get trades in this 1-second window
        trades_in_second = []
        if self.all_trades_data is not None and not self.all_trades_data.empty:
            trades_df = self.all_trades_data[(self.all_trades_data.index >= window_start) &
                                             (self.all_trades_data.index < window_end)]
            if not trades_df.empty:
                trades_in_second = [row.to_dict() for _, row in trades_df.iterrows()]

        # Get quotes in this 1-second window
        quotes_in_second = []
        if self.all_quotes_data is not None and not self.all_quotes_data.empty:
            quotes_df = self.all_quotes_data[(self.all_quotes_data.index >= window_start) &
                                             (self.all_quotes_data.index < window_end)]
            if not quotes_df.empty:
                quotes_in_second = [row.to_dict() for _, row in quotes_df.iterrows()]

        # For live mode, you would also integrate from _live_trades_buffer etc. here
        if self.mode == 'live':
            # Process and clear relevant items from _live_trades_buffer, _live_quotes_buffer
            # For simplicity, assuming live data is pushed into all_trades_data equivalent for now
            pass

        if not current_1s_bar and not trades_in_second and not quotes_in_second and not is_initial_fill:
            # If there's absolutely no data for this second (no bar, no trades, no quotes)
            # and it's not part of the initial buffer fill where gaps might be more acceptable
            # depending on data source.
            # self.logger.debug(f"No market activity found for timestamp {timestamp}")
            # To ensure the window keeps advancing, we might need to create a synthetic event
            # or use the last known bar.
            if self.current_1s_data_window:  # use last known bar
                last_event = self.current_1s_data_window[-1]
                current_1s_bar = last_event['bar']  # Forward fill the bar
                current_1s_bar['ffill_timestamp'] = timestamp  # Mark as forward filled
            else:  # No prior data, cannot proceed
                return None

        return {
            'timestamp': timestamp,
            'bar': current_1s_bar,  # This is the OHLCV for the second ending at 'timestamp'
            'trades': trades_in_second,  # Trades that occurred during this 1s interval
            'quotes': quotes_in_second  # Quotes that occurred during this 1s interval
        }

    def step(self) -> bool:
        """
        Advance the simulator to the next 1-second data point.

        Returns:
            bool: True if successful and data is available, False if at the end of data or error.
        """
        if self.is_done():
            self.logger.info("MarketSimulatorV2: End of data reached.")
            return False

        if self.mode == 'backtesting':
            # Advance to the next 1s bar in historical data
            self._current_1s_data_idx += 1
            if self._current_1s_data_idx >= len(self.all_1s_data.index):
                self.current_timestamp = self.all_1s_data.index[-1] + timedelta(seconds=1)  # Theoretical next second
                return False  # End of data
            self.current_timestamp = self.all_1s_data.index[self._current_1s_data_idx]

        elif self.mode == 'live':
            # For live mode, time advances naturally or via an external clock
            # We assume self.current_timestamp is updated externally or ticks per second
            # For this simulation, let's advance it manually for demonstration
            self.current_timestamp += timedelta(seconds=1)

            # Process any incoming live data that has accumulated
            # This is where data from add_live_trade etc. would be sorted and merged
            # For example, new trades in _live_trades_buffer get appended to self.all_trades_data
            # or a live-specific deque.
            # Then, clear the _live_trades_buffer, _live_quotes_buffer.
            if self._live_1s_bar_events:
                # Example: process a 1s bar event if it arrived via WS
                # This would typically involve updating something like self.all_1s_data
                # or a live-specific structure. For now, just logging.
                self.logger.debug(
                    f"Processing {len(self._live_1s_bar_events)} live 1s bar events for {self.current_timestamp}")
                self._live_1s_bar_events.clear()
            # Similar processing for trades and quotes

        # Get consolidated data for the new current_timestamp
        one_second_event_data = self._get_data_for_timestamp(self.current_timestamp)

        if one_second_event_data:
            self.current_1s_data_window.append(one_second_event_data)
            return True
        else:
            # If no data could be constructed (e.g., end of backtest data, or persistent live gap)
            # We might still want to advance time and append a minimal event to keep features ticking
            # For now, if _get_data_for_timestamp returns None, it indicates a critical issue or end.
            self.logger.warning(f"No data event constructed for {self.current_timestamp}. May be end of relevant data.")
            # If we want to ensure the window always has *something* even for empty seconds:
            # empty_event = {'timestamp': self.current_timestamp, 'bar': None, 'trades': [], 'quotes': []}
            # self.current_1s_data_window.append(empty_event)
            return False  # Or True if we allow empty ticks

    def get_current_market_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the current market state for feature extraction and decision making.
        This state represents the market *as of* self.current_timestamp.
        """
        if not self.current_timestamp or not self.current_1s_data_window:
            # self.logger.warning("Attempted to get market state but simulator not ready or no data in window.")
            return None

        latest_event_in_window = self.current_1s_data_window[-1]

        # Ensure the latest event corresponds to the current_timestamp
        # It might not if step() returned False due to end of data or gap
        if latest_event_in_window['timestamp'] != self.current_timestamp:
            # This can happen if we just hit the end of data; the window has the last valid data,
            # but current_timestamp might be one step beyond.
            # Or, if _get_data_for_timestamp returned None and step() is designed to still advance time.
            # For robustness, we should provide the state as of the latest *valid* data in the window.
            # self.logger.debug(f"Current timestamp {self.current_timestamp} differs from latest window event {latest_event_in_window['timestamp']}. Providing state for latest window event.")
            if not self.is_done() and self.mode == 'backtesting':  # Only log if not actually done
                self.logger.debug(
                    f"Timestamp mismatch: current_ts={self.current_timestamp}, window_ts={latest_event_in_window['timestamp']}")

        # Data for S/R levels: Slice historical bars up to the current time
        current_historical_5m_bars = None
        if self.historical_5m_bars_for_sr is not None:
            current_historical_5m_bars = self.historical_5m_bars_for_sr[
                self.historical_5m_bars_for_sr.index <= self.current_timestamp]

        current_historical_1d_bars = None
        if self.historical_1d_bars_for_sr is not None:
            current_historical_1d_bars = self.historical_1d_bars_for_sr[
                self.historical_1d_bars_for_sr.index <= self.current_timestamp]

        return {
            'symbol': self.symbol,
            'timestamp': latest_event_in_window['timestamp'],  # Timestamp of the data being provided

            # Data for the most recent 1-second interval
            'latest_1s_bar': latest_event_in_window.get('bar'),
            'latest_1s_trades': latest_event_in_window.get('trades'),
            'latest_1s_quotes': latest_event_in_window.get('quotes'),

            # Rolling window of 1-second data events (for HF, MF, LF feature calculations)
            # Each element is a dict: {'timestamp': ..., 'bar': ..., 'trades': [...], 'quotes': [...]}
            'rolling_1s_data_window': list(self.current_1s_data_window),  # Send a copy

            # Historical bars for S/R levels (typically longer term)
            'historical_5m_for_sr': current_historical_5m_bars,
            'historical_1d_for_sr': current_historical_1d_bars,

            # Direct access to the most recent price for convenience (e.g., for execution simulator)
            'current_price': latest_event_in_window.get('bar', {}).get('close') if latest_event_in_window.get(
                'bar') else None,
            'current_bid': latest_event_in_window.get('quotes', [{}])[-1].get(
                'bid_price') if latest_event_in_window.get('quotes') else None,
            'current_ask': latest_event_in_window.get('quotes', [{}])[-1].get(
                'ask_price') if latest_event_in_window.get('quotes') else None,
        }

    def is_done(self) -> bool:
        """Check if we've reached the end of data in backtesting mode."""
        if self.mode == 'backtesting':
            if self.all_1s_data is None or self.all_1s_data.empty:
                return True
            # Done if current_1s_data_idx is at or beyond the last available index
            return self._current_1s_data_idx >= len(self.all_1s_data.index) - 1  # -1 because idx is for *next* step
        elif self.mode == 'live':
            # Live mode typically doesn't "end" unless explicitly stopped
            # Or if a predefined simulation end time is reached
            if self.end_time and self.current_timestamp and self.current_timestamp >= self.end_time:
                return True
            return False
        return True  # Default to done if mode is unknown

    def reset(self, start_time_str: Optional[str] = None, end_time_str: Optional[str] = None):
        """
        Reset the simulator to an initial state.
        For backtesting, this means re-initializing based on available data and new times if provided.
        """
        self.current_1s_data_window.clear()
        self._live_trades_buffer.clear()
        self._live_quotes_buffer.clear()
        self._live_1s_bar_events.clear()

        self._current_1s_data_idx = 0
        self.current_timestamp = None

        if start_time_str: self.start_time = pd.to_datetime(start_time_str)
        if end_time_str: self.end_time = pd.to_datetime(end_time_str)

        # Re-filter data if start/end times changed (conceptually, real data wouldn't be reloaded like this)
        # For actual episode management (next day, etc.), you'd likely reinstantiate the simulator
        # or have a more sophisticated data manager that feeds data for specific periods.
        # For this example, we assume `all_1s_data` might be a larger dataset, and reset adjusts view.

        # Re-initialize based on the (potentially new) start/end times
        if self.mode == 'backtesting':
            # The original dataframes are kept, _prepare_historical_data logic needs to be invoked again if data source is external.
            # For simplicity, assume they are already loaded, and we are resetting pointers and buffers.
            if self.all_1s_data is None or self.all_1s_data.empty:
                raise ValueError("Historical 1s data is required for backtesting mode to reset.")
            self._initialize_for_backtesting()
        elif self.mode == 'live':
            self.current_timestamp = datetime.now().replace(microsecond=0)
            # Potentially re-fill buffers with recent history for live mode if needed
            self.logger.info(f"MarketSimulatorV2 reset for LIVE trading. Current time: {self.current_timestamp}")

        self.logger.info(
            f"MarketSimulatorV2 reset. Mode: {self.mode}, New Start: {self.start_time}, New End: {self.end_time}")
        return self.get_current_market_state()  # Return initial state

    # --- Methods for Live Data Ingestion ---
    def add_live_trade(self, trade_event: Dict[str, Any]):
        """
        Add a live trade event. Expected to have 'timestamp', 'price', 'size', 'side'.
        Timestamp should be a datetime object or parsable string.
        """
        # In a real system, this would buffer events. They are processed in _get_data_for_timestamp or step().
        # For robust handling, include precise event timestamp.
        # trade_event['timestamp'] = pd.to_datetime(trade_event['timestamp'])
        # self._live_trades_buffer.append(trade_event)
        # For now, if live, directly add to all_trades_data if it's a DataFrame
        if self.mode == 'live' and self.all_trades_data is not None:
            # This is a simplified way; real live systems use more complex appends or time-series DBs
            new_trade = pd.DataFrame([trade_event]).set_index(pd.DatetimeIndex([trade_event['timestamp']]))
            self.all_trades_data = pd.concat([self.all_trades_data, new_trade])
            self.all_trades_data.sort_index(inplace=True)  # Keep sorted
        self.logger.debug(f"Live trade received: {trade_event}")

    def add_live_quote(self, quote_event: Dict[str, Any]):
        """
        Add a live quote event. Expected: 'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size'.
        """
        if self.mode == 'live' and self.all_quotes_data is not None:
            new_quote = pd.DataFrame([quote_event]).set_index(pd.DatetimeIndex([quote_event['timestamp']]))
            self.all_quotes_data = pd.concat([self.all_quotes_data, new_quote])
            self.all_quotes_data.sort_index(inplace=True)
        self.logger.debug(f"Live quote received: {quote_event}")

    def add_live_1s_data_event(self, bar_event: Dict[str, Any]):
        """
        Add a live 1-second bar event if it comes through a separate feed.
        Expected: 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        """
        # This would typically update the primary 1s data source for live mode.
        # self._live_1s_bar_events.append(bar_event)
        if self.mode == 'live' and self.all_1s_data is not None:
            new_bar = pd.DataFrame([bar_event]).set_index(pd.DatetimeIndex([bar_event['timestamp']]))
            self.all_1s_data = pd.concat([self.all_1s_data, new_bar])
            self.all_1s_data.sort_index(inplace=True)
        self.logger.debug(f"Live 1s bar event received: {bar_event}")

    def close(self):
        """Clean up resources if any."""
        self.logger.info(f"MarketSimulatorV2 for {self.symbol} closing.")
        # No specific resources to close in this version unless data sources were actual files/connections.


# Example Usage (Illustrative - data loading would be separate)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Create dummy data for testing
    start_dt = datetime(2023, 1, 1, 9, 30, 0)
    num_seconds = 700  # A bit more than 10 minutes
    time_index = [start_dt + timedelta(seconds=i) for i in range(num_seconds)]

    dummy_1s_data = pd.DataFrame({
        'open': np.random.rand(num_seconds) * 100 + 50,
        'high': np.random.rand(num_seconds) * 2 + 100,  # open + something
        'low': np.random.rand(num_seconds) * -2 + 100,  # open - something
        'close': np.random.rand(num_seconds) * 100 + 50,
        'volume': np.random.randint(100, 1000, num_seconds)
    }, index=pd.DatetimeIndex(time_index))
    dummy_1s_data['high'] = dummy_1s_data[['open', 'close']].max(axis=1) + np.random.rand(num_seconds)
    dummy_1s_data['low'] = dummy_1s_data[['open', 'close']].min(axis=1) - np.random.rand(num_seconds)

    # Dummy trades (sparse)
    trade_times = np.random.choice(time_index, size=num_seconds // 5, replace=False)
    trade_times.sort()
    dummy_trades = pd.DataFrame({
        'price': np.random.rand(len(trade_times)) * 100 + 50,
        'size': np.random.randint(1, 10, len(trade_times)),
        'side': np.random.choice(['buy', 'sell'], len(trade_times))
    }, index=pd.DatetimeIndex(trade_times))

    sim_config = {
        'mode': 'backtesting',
        'initial_buffer_seconds': 300,  # 5 minutes
        'max_1s_data_window_size': 600  # Keep 10 minutes of 1s events in window
    }

    simulator = MarketSimulatorV2(
        symbol="TEST_STOCK",
        historical_1s_data=dummy_1s_data,
        historical_trades_data=dummy_trades,
        # historical_quotes_data=... (can add dummy quotes too)
        config=sim_config,
        logger=logger
    )

    max_steps = 20
    for step_num in range(max_steps):
        if simulator.is_done():
            logger.info(f"Simulation ended early at step {step_num} by is_done().")
            break

        logger.info(f"--- Step {step_num + 1}/{max_steps} ---")

        if not simulator.step():
            logger.info(f"Simulator step failed or end of data at step {step_num + 1}.")
            break

        current_state = simulator.get_current_market_state()
        if current_state:
            logger.info(f"Timestamp: {current_state['timestamp']}")
            if current_state['latest_1s_bar']:
                logger.info(f"  Bar Close: {current_state['latest_1s_bar']['close']:.2f}")
            logger.info(f"  Trades in last sec: {len(current_state['latest_1s_trades'])}")
            logger.info(f"  Window size: {len(current_state['rolling_1s_data_window'])}")
            if current_state['rolling_1s_data_window']:
                logger.info(f"  Oldest in window: {current_state['rolling_1s_data_window'][0]['timestamp']}")
                logger.info(f"  Newest in window: {current_state['rolling_1s_data_window'][-1]['timestamp']}")

        else:
            logger.warning("No current state available.")

        # Illustrate adding a live trade (though in backtesting, it just appends to the historical set)
        if step_num == 5 and simulator.mode == 'live':  # This condition won't be met with current config
            live_trade_event = {
                'timestamp': simulator.current_timestamp + timedelta(milliseconds=500),
                # trade within the current second
                'price': 105.50, 'size': 5, 'side': 'buy'
            }
            simulator.add_live_trade(live_trade_event)

    logger.info("Simulation finished.")
    simulator.close()