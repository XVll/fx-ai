# simulation/market_simulator.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta


class MarketSimulator:
    """
    Simulates realistic market behavior including:
    - Price impact and slippage
    - Order book dynamics
    - Execution latency
    - Limit Up/Limit Down (LULD) halts
    - Trading suspensions
    """

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        Initialize the market simulator.

        Args:
            config: Configuration dictionary with simulation parameters
            logger: Optional logger
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Default configuration
        self.min_spread_pct = self.config.get('min_spread_pct', 0.0005)  # 5bps min spread
        self.typical_spread_pct = self.config.get('typical_spread_pct', 0.001)  # 10bps typical spread
        self.max_spread_pct = self.config.get('max_spread_pct', 0.005)  # 50bps max spread in normal conditions
        self.slippage_factor = self.config.get('slippage_factor', 0.5)  # Slippage as fraction of spread
        self.volatility_slippage_factor = self.config.get('volatility_slippage_factor', 0.2)  # Higher vol = higher slip
        self.latency_ms = self.config.get('latency_ms', 250)  # Simulated latency in milliseconds
        self.latency_variance_ms = self.config.get('latency_variance_ms', 100)  # Variance in latency
        self.random_fill_failure_pct = self.config.get('random_fill_failure_pct', 0.01)  # 1% chance of fill failure
        self.luld_enabled = self.config.get('luld_enabled', True)  # Enable LULD circuit breakers
        self.luld_band_pct = self.config.get('luld_band_pct', 0.05)  # 5% LULD band (tier 2 stocks)
        self.min_price_increment = self.config.get('min_price_increment', 0.0001)  # Min price movement (e.g., $0.0001)
        self.enforce_time_ordering = self.config.get('enforce_time_ordering',
                                                     True)  # Strictly enforce timestamp ordering
        self.allow_backwards_time = self.config.get('allow_backwards_time',
                                                    False)  # Allow time to go backwards (for debug)

        # Current market state
        self.current_price = None
        self.current_bid = None
        self.current_ask = None
        self.current_spread = None
        self.current_timestamp = None
        self.last_trades = []  # Recent trades for volume imbalance calculation
        self.halted = False
        self.halt_start_time = None
        self.halt_end_time = None
        self.cached_volatility = 0.01  # Default volatility estimate
        self.tape_imbalance = 0.0  # Range from -1 (all sells) to 1 (all buys)

        # Reference to the actual market data (set by the main simulator)
        self.quotes_df = None
        self.trades_df = None
        self.bars_df = None
        self.status_df = None

        # Track initialization state
        self.initialized = False

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def initialize_from_data(self, quotes_df: pd.DataFrame, trades_df: pd.DataFrame,
                             bars_df: pd.DataFrame, status_df: pd.DataFrame = None) -> None:
        """
        Initialize the market simulator with actual market data.

        Args:
            quotes_df: DataFrame with quote data (bid/ask)
            trades_df: DataFrame with trade data
            bars_df: DataFrame with OHLCV bars
            status_df: DataFrame with status data (halts, etc.)
        """
        self.quotes_df = quotes_df
        self.trades_df = trades_df
        self.bars_df = bars_df
        self.status_df = status_df

        # Reset market state
        self.current_price = None
        self.current_bid = None
        self.current_ask = None
        self.current_spread = None
        self.current_timestamp = None
        self.halted = False
        self.halt_start_time = None
        self.halt_end_time = None

        # Find the earliest timestamp across all data sources
        earliest_time = None

        for df in [quotes_df, trades_df, bars_df, status_df]:
            if df is not None and not df.empty:
                if earliest_time is None or df.index.min() < earliest_time:
                    earliest_time = df.index.min()

        if earliest_time is None:
            self._log("No data available for market initialization", logging.WARNING)
            return

        self._log(f"Initializing market from common start time: {earliest_time}")

        # Initialize from the earliest common timestamp
        self.current_timestamp = earliest_time

        # Try to initialize from various data sources, in order of preference
        if not quotes_df.empty and 'bid_px_00' in quotes_df.columns and 'ask_px_00' in quotes_df.columns:
            # Get initial bid/ask from first available quote
            quotes_at_time = quotes_df[quotes_df.index >= earliest_time]
            if not quotes_at_time.empty:
                first_row = quotes_at_time.iloc[0]
                if pd.notna(first_row['bid_px_00']) and pd.notna(first_row['ask_px_00']):
                    self.current_bid = first_row['bid_px_00']
                    self.current_ask = first_row['ask_px_00']
                    self.current_spread = self.current_ask - self.current_bid
                    self.current_price = (self.current_bid + self.current_ask) / 2
                    self.current_timestamp = quotes_at_time.index[0]
                    self._log(
                        f"Initialized market from quotes: Price=${self.current_price:.4f}, Bid=${self.current_bid:.4f}, Ask=${self.current_ask:.4f}")

        # If quotes didn't provide valid data, try trades
        if self.current_price is None and not trades_df.empty and 'price' in trades_df.columns:
            trades_at_time = trades_df[trades_df.index >= earliest_time]
            if not trades_at_time.empty:
                first_price = trades_at_time['price'].iloc[0]
                if pd.notna(first_price) and first_price > 0:
                    self.current_price = first_price
                    # Synthesize bid/ask with typical spread
                    spread = self.current_price * self.typical_spread_pct
                    self.current_bid = self.current_price - spread / 2
                    self.current_ask = self.current_price + spread / 2
                    self.current_spread = spread
                    self.current_timestamp = trades_at_time.index[0]
                    self._log(f"Initialized market from trades: Price=${self.current_price:.4f}")

        # If neither quotes nor trades worked, try bars
        if self.current_price is None and not bars_df.empty:
            bars_at_time = bars_df[bars_df.index >= earliest_time]
            if not bars_at_time.empty and 'open' in bars_at_time.columns:
                first_price = bars_at_time['open'].iloc[0]
                if pd.notna(first_price) and first_price > 0:
                    # Use open price from first bar
                    self.current_price = first_price
                    # Synthesize bid/ask
                    spread = self.current_price * self.typical_spread_pct
                    self.current_bid = self.current_price - spread / 2
                    self.current_ask = self.current_price + spread / 2
                    self.current_spread = spread
                    self.current_timestamp = bars_at_time.index[0]
                    self._log(f"Initialized market from bars: Price=${self.current_price:.4f}")

        # If still not initialized, set some default values for testing
        if self.current_price is None:
            self._log("WARNING: Could not initialize market state from data, using default values", logging.WARNING)
            # Set default values for testing
            self.current_price = 10.0  # Default test price
            self.current_bid = 9.95
            self.current_ask = 10.05
            self.current_spread = 0.10

        # Check if market is halted at the start
        if status_df is not None and not status_df.empty:
            self._update_halt_status(self.current_timestamp)

        # Calculate initial volatility
        if not bars_df.empty:
            bars_at_time = bars_df[bars_df.index >= earliest_time]
            if len(bars_at_time) > 10:
                # Use 10-bar rolling volatility
                returns = bars_at_time['close'].pct_change().iloc[1:11]
                self.cached_volatility = np.std(returns) if len(returns) > 0 else 0.01

        self._log(
            f"Market initialized: Price=${self.current_price:.4f}, Bid=${self.current_bid:.4f}, Ask=${self.current_ask:.4f}, Spread=${self.current_spread:.4f}")
        self.initialized = True

    def update_to_timestamp(self, timestamp: datetime) -> None:
        """
        Update market state to a specific timestamp.

        Args:
            timestamp: Target timestamp to update to
        """
        if not self.initialized:
            self._log("Market simulator not initialized yet", logging.WARNING)
            return

        # Handle timezone comparison safely
        if self.current_timestamp is not None:
            # Ensure both timestamps have consistent timezone information
            current_ts = pd.Timestamp(self.current_timestamp)
            target_ts = pd.Timestamp(timestamp)

            # If one has timezone and the other doesn't, make them consistent
            if current_ts.tzinfo is not None and target_ts.tzinfo is None:
                target_ts = target_ts.tz_localize(current_ts.tzinfo)
            elif current_ts.tzinfo is None and target_ts.tzinfo is not None:
                current_ts = current_ts.tz_localize(target_ts.tzinfo)

            # Now we can safely compare
            if target_ts == current_ts:
                # Already at this timestamp, no update needed
                return

            if target_ts < current_ts and self.enforce_time_ordering and not self.allow_backwards_time:
                # If strict time ordering is enforced, log a warning but allow the update
                # This makes simulation more robust in the face of slightly out-of-order data
                self._log(f"Cannot move market backwards from {self.current_timestamp} to {timestamp}", logging.WARNING)
                return

        self.current_timestamp = timestamp

        # Update market state based on available data up to this timestamp
        self._update_market_state(timestamp)
        self._update_volatility(timestamp)
        self._update_tape_imbalance(timestamp)
        self._update_halt_status(timestamp)

    def _update_market_state(self, timestamp: datetime) -> None:
        """
        Update bid, ask, and price based on market data up to a timestamp.

        Args:
            timestamp: Timestamp to update to
        """
        # Update from quotes
        if self.quotes_df is not None and not self.quotes_df.empty:
            # Get the latest quote up to timestamp
            quotes_up_to = self.quotes_df[self.quotes_df.index <= timestamp]
            if not quotes_up_to.empty:
                latest_quote = quotes_up_to.iloc[-1]
                if 'bid_px_00' in latest_quote and 'ask_px_00' in latest_quote:
                    if pd.notna(latest_quote['bid_px_00']) and pd.notna(latest_quote['ask_px_00']):
                        self.current_bid = latest_quote['bid_px_00']
                        self.current_ask = latest_quote['ask_px_00']
                        self.current_spread = self.current_ask - self.current_bid
                        self.current_price = (self.current_bid + self.current_ask) / 2
                        return

        # If no quotes or incomplete quote data, use trades
        if self.trades_df is not None and not self.trades_df.empty:
            trades_up_to = self.trades_df[self.trades_df.index <= timestamp]
            if not trades_up_to.empty and 'price' in trades_up_to.columns:
                latest_price = trades_up_to.iloc[-1]['price']
                if pd.notna(latest_price) and latest_price > 0:
                    # Update price from trade
                    self.current_price = latest_price

                    # Only update bid/ask if we don't have current values
                    if self.current_bid is None or self.current_ask is None:
                        spread = self.current_price * self.typical_spread_pct
                        self.current_bid = self.current_price - spread / 2
                        self.current_ask = self.current_price + spread / 2
                        self.current_spread = spread
                    return

        # If we still need prices, try bars
        if self.bars_df is not None and not self.bars_df.empty:
            bars_up_to = self.bars_df[self.bars_df.index <= timestamp]
            if not bars_up_to.empty:
                latest_close = bars_up_to.iloc[-1]['close']
                if pd.notna(latest_close) and latest_close > 0:
                    self.current_price = latest_close
                    # Synthesize bid/ask if needed
                    if self.current_bid is None or self.current_ask is None:
                        spread = self.current_price * self.typical_spread_pct
                        self.current_bid = self.current_price - spread / 2
                        self.current_ask = self.current_price + spread / 2
                        self.current_spread = spread

    def _update_volatility(self, timestamp: datetime) -> None:
        """
        Update volatility estimate based on recent price movements.

        Args:
            timestamp: Current timestamp
        """
        # Use 1-minute bars for volatility calculation
        if self.bars_df is not None and not self.bars_df.empty:
            # Get bars up to 10 minutes before timestamp
            lookback_time = timestamp - timedelta(minutes=10)
            recent_bars = self.bars_df[(self.bars_df.index >= lookback_time) & (self.bars_df.index <= timestamp)]

            if len(recent_bars) > 1:
                # Calculate returns
                returns = recent_bars['close'].pct_change().dropna()
                if len(returns) > 0:
                    # Annualize volatility: std * sqrt(bars per year)
                    # For 1-minute bars: sqrt(252 * 6.5 * 60) = ~1600
                    self.cached_volatility = np.std(returns) * np.sqrt(len(returns))

                    # Cap volatility
                    self.cached_volatility = min(0.5, max(0.005, self.cached_volatility))

    def _update_tape_imbalance(self, timestamp: datetime) -> None:
        """
        Update tape imbalance based on recent trades.

        Args:
            timestamp: Current timestamp
        """
        if self.trades_df is not None and not self.trades_df.empty:
            # Get trades in last 5 seconds
            lookback_time = timestamp - timedelta(seconds=5)
            recent_trades = self.trades_df[
                (self.trades_df.index >= lookback_time) & (self.trades_df.index <= timestamp)]

            self.last_trades = recent_trades

            if len(recent_trades) > 0 and 'side' in recent_trades.columns and 'size' in recent_trades.columns:
                # Calculate volume imbalance
                buy_mask = recent_trades['side'] == 'B'
                sell_mask = recent_trades['side'] == 'A'

                buy_vol = recent_trades[buy_mask]['size'].sum() if any(buy_mask) else 0
                sell_vol = recent_trades[sell_mask]['size'].sum() if any(sell_mask) else 0

                total_vol = buy_vol + sell_vol
                if total_vol > 0:
                    self.tape_imbalance = (buy_vol - sell_vol) / total_vol
                else:
                    self.tape_imbalance = 0.0

    def _update_halt_status(self, timestamp: datetime) -> None:
        """
        Update trading halt status based on status data.

        Args:
            timestamp: Current timestamp
        """
        if self.status_df is not None and not self.status_df.empty:
            # Get status updates up to timestamp
            status_up_to = self.status_df[self.status_df.index <= timestamp]

            if not status_up_to.empty:
                # Get the latest status update
                latest_status = status_up_to.iloc[-1]

                # Check for halt status
                if 'is_trading' in latest_status:
                    self.halted = latest_status['is_trading'] == 'N'

                # Check for halt action
                if 'action' in latest_status:
                    # Action 8 is a halt, Action 7 is a resume
                    if latest_status['action'] == 8 and not self.halted:
                        self.halted = True
                        self.halt_start_time = timestamp
                        self.halt_end_time = None
                    elif latest_status['action'] == 7 and self.halted:
                        self.halted = False
                        self.halt_end_time = timestamp

                # Additional LULD simulation
                if self.luld_enabled and self.current_price is not None:
                    self._check_luld_bands(timestamp)

    def _check_luld_bands(self, timestamp: datetime) -> None:
        """
        Check if current price has breached LULD bands and simulate a halt if needed.

        Args:
            timestamp: Current timestamp
        """
        if not self.luld_enabled or self.halted or self.current_price is None:
            return

        # Get reference price (typically 5-minute moving average)
        reference_price = self.current_price
        if self.bars_df is not None and not self.bars_df.empty:
            # Get 5-minute lookback
            lookback_time = timestamp - timedelta(minutes=5)
            recent_bars = self.bars_df[(self.bars_df.index >= lookback_time) & (self.bars_df.index <= timestamp)]

            if not recent_bars.empty:
                reference_price = recent_bars['close'].mean()

        # Calculate LULD bands
        upper_band = reference_price * (1 + self.luld_band_pct)
        lower_band = reference_price * (1 - self.luld_band_pct)

        # Check if price has breached bands
        if self.current_price > upper_band or self.current_price < lower_band:
            # Simulate a halt
            self.halted = True
            self.halt_start_time = timestamp
            # Halt lasts for 5 minutes
            self.halt_end_time = timestamp + timedelta(minutes=5)

            band_type = "upper" if self.current_price > upper_band else "lower"
            self._log(
                f"LULD halt triggered at {timestamp}: Price ${self.current_price:.4f} breached {band_type} band ${upper_band if band_type == 'upper' else lower_band:.4f}",
                logging.WARNING)

    def simulate_market_order_execution(self, side: str, size: float, time: datetime = None) -> Dict[str, Any]:
        """
        Simulate a market order execution with realistic fill dynamics.

        Args:
            side: Order side ('buy' or 'sell')
            size: Order size (quantity)
            time: Execution time (defaults to current_timestamp)

        Returns:
            Dictionary with execution details
        """
        execution_time = time or self.current_timestamp

        # Check if market is halted
        if self.halted:
            halt_msg = f"Market halted at {execution_time}"
            if self.halt_end_time:
                halt_msg += f", expected to resume at {self.halt_end_time}"
            self._log(halt_msg, logging.WARNING)
            return {
                'success': False,
                'filled_size': 0.0,
                'fill_price': None,
                'slippage': 0.0,
                'time': execution_time,
                'latency_ms': 0,
                'reason': 'market_halted'
            }

        # Add latency
        latency = np.random.normal(self.latency_ms, self.latency_variance_ms)
        latency = max(5, min(1000, latency))  # Ensure between 5-1000ms
        actual_exec_time = execution_time + timedelta(milliseconds=latency)

        # Update market to execution time (including latency)
        self.update_to_timestamp(actual_exec_time)

        # Random chance of fill failure
        if np.random.random() < self.random_fill_failure_pct:
            self._log(f"{side.upper()} order failed to fill due to random market conditions", logging.WARNING)
            return {
                'success': False,
                'filled_size': 0.0,
                'fill_price': None,
                'slippage': 0.0,
                'time': actual_exec_time,
                'latency_ms': latency,
                'reason': 'random_failure'
            }

        # Calculate realistic slippage
        spread_slippage = self.current_spread * self.slippage_factor
        vol_slippage = self.current_price * self.cached_volatility * self.volatility_slippage_factor

        # Adjust slippage based on order size (larger orders = more slippage)
        # This is a simplified model - in reality, it would depend on order book depth
        size_factor = 1.0
        if self.quotes_df is not None and not self.quotes_df.empty:
            # Estimate typical size at best level
            typical_size = 100  # Default
            quotes_at_time = self.quotes_df[self.quotes_df.index <= actual_exec_time]
            if not quotes_at_time.empty:
                quote = quotes_at_time.iloc[-1]
                if side.lower() == 'buy' and 'ask_sz_00' in quote:
                    typical_size = quote['ask_sz_00']
                elif side.lower() == 'sell' and 'bid_sz_00' in quote:
                    typical_size = quote['bid_sz_00']

            # Calculate size factor
            size_factor = min(3.0, max(1.0, size / typical_size))

        # Total slippage
        total_slippage = (spread_slippage + vol_slippage) * size_factor

        # Apply tape imbalance effect
        if side.lower() == 'buy':
            # High sell imbalance increases buy slippage
            imbalance_adjustment = (1 - self.tape_imbalance) * 0.5
            total_slippage *= (1 + imbalance_adjustment)
        else:
            # High buy imbalance increases sell slippage
            imbalance_adjustment = (1 + self.tape_imbalance) * 0.5
            total_slippage *= (1 + imbalance_adjustment)

        # Apply slippage to get fill price
        if side.lower() == 'buy':
            fill_price = self.current_ask + total_slippage
        else:  # sell
            fill_price = self.current_bid - total_slippage

        # Round to price increment
        fill_price = round(fill_price / self.min_price_increment) * self.min_price_increment

        # Ensure price is valid
        fill_price = max(0.0001, fill_price)

        # Calculate slippage percentage
        slippage_pct = ((fill_price - self.current_price) / self.current_price) * 100.0

        self._log(
            f"{side.upper()} order filled: {size} shares @ ${fill_price:.4f} with {slippage_pct:.3f}% slippage after {latency:.1f}ms latency")

        return {
            'success': True,
            'filled_size': size,
            'fill_price': fill_price,
            'slippage': slippage_pct,
            'time': actual_exec_time,
            'latency_ms': latency,
            'reason': 'filled'
        }

    def simulate_limit_order_execution(self, side: str, size: float, price: float,
                                       time: datetime = None,
                                       expiration_time: datetime = None) -> Dict[str, Any]:
        """
        Simulate a limit order execution with realistic fill dynamics.

        Args:
            side: Order side ('buy' or 'sell')
            size: Order size (quantity)
            price: Limit price
            time: Execution time (defaults to current_timestamp)
            expiration_time: Time limit for the order (defaults to current + 5s)

        Returns:
            Dictionary with execution details
        """
        execution_time = time or self.current_timestamp

        if expiration_time is None:
            expiration_time = execution_time + timedelta(seconds=5)

        # Check if market is halted
        if self.halted:
            self._log(f"Market halted at {execution_time}, limit order not placed", logging.WARNING)
            return {
                'success': False,
                'filled_size': 0.0,
                'fill_price': None,
                'time': execution_time,
                'latency_ms': 0,
                'reason': 'market_halted'
            }

        # Add latency for order placement
        latency = np.random.normal(self.latency_ms, self.latency_variance_ms)
        latency = max(5, min(1000, latency))  # Ensure between 5-1000ms
        order_time = execution_time + timedelta(milliseconds=latency)

        # Update market to order time (after latency)
        self.update_to_timestamp(order_time)

        # Check if order price is valid for immediate execution
        would_execute = False
        if side.lower() == 'buy' and price >= self.current_ask:
            would_execute = True
            fill_price = min(price, self.current_ask)  # Fill at ask or better
        elif side.lower() == 'sell' and price <= self.current_bid:
            would_execute = True
            fill_price = max(price, self.current_bid)  # Fill at bid or better

        # If order would execute immediately, fill it
        if would_execute:
            # Round fill price to price increment
            fill_price = round(fill_price / self.min_price_increment) * self.min_price_increment

            self._log(f"{side.upper()} limit order filled immediately: {size} shares @ ${fill_price:.4f}")

            return {
                'success': True,
                'filled_size': size,
                'fill_price': fill_price,
                'time': order_time,
                'latency_ms': latency,
                'reason': 'filled_immediate'
            }

        # If we reach here, the order wasn't immediately filled
        # For simulation, check if it would fill by expiration

        # Find all price updates between order_time and expiration_time
        fill_time = None
        fill_price = None

        # Check if quotes cross the limit price during the valid period
        if self.quotes_df is not None and not self.quotes_df.empty:
            quotes_during_valid = self.quotes_df[(self.quotes_df.index > order_time) &
                                                 (self.quotes_df.index <= expiration_time)]

            if not quotes_during_valid.empty:
                for idx, quote in quotes_during_valid.iterrows():
                    if side.lower() == 'buy' and 'ask_px_00' in quote and quote['ask_px_00'] <= price:
                        fill_time = idx
                        fill_price = quote['ask_px_00']
                        break
                    elif side.lower() == 'sell' and 'bid_px_00' in quote and quote['bid_px_00'] >= price:
                        fill_time = idx
                        fill_price = quote['bid_px_00']
                        break

        # If no quotes crossed, check if trades would have filled the order
        if fill_time is None and self.trades_df is not None and not self.trades_df.empty:
            trades_during_valid = self.trades_df[(self.trades_df.index > order_time) &
                                                 (self.trades_df.index <= expiration_time)]

            if not trades_during_valid.empty:
                for idx, trade in trades_during_valid.iterrows():
                    if side.lower() == 'buy' and 'price' in trade and trade['price'] <= price:
                        fill_time = idx
                        fill_price = trade['price']
                        break
                    elif side.lower() == 'sell' and 'price' in trade and trade['price'] >= price:
                        fill_time = idx
                        fill_price = trade['price']
                        break

        # If order would be filled
        if fill_time is not None and fill_price is not None:
            # Update market to fill time
            self.update_to_timestamp(fill_time)

            # Round fill price to price increment
            fill_price = round(fill_price / self.min_price_increment) * self.min_price_increment

            # Calculate latency as time from order to fill
            fill_latency = (fill_time - execution_time).total_seconds() * 1000

            self._log(
                f"{side.upper()} limit order filled after waiting: {size} shares @ ${fill_price:.4f} at {fill_time}")

            return {
                'success': True,
                'filled_size': size,
                'fill_price': fill_price,
                'time': fill_time,
                'latency_ms': fill_latency,
                'reason': 'filled_delayed'
            }

        # Order did not fill within the expiration time
        self._log(f"{side.upper()} limit order expired unfilled: {size} shares @ ${price:.4f}")

        return {
            'success': False,
            'filled_size': 0.0,
            'fill_price': None,
            'time': expiration_time,
            'latency_ms': (expiration_time - execution_time).total_seconds() * 1000,
            'reason': 'expired_unfilled'
        }

    def get_current_market_state(self) -> Dict[str, Any]:
        """
        Get the current market state.

        Returns:
            Dictionary with market state information
        """
        return {
            'timestamp': self.current_timestamp,
            'price': self.current_price,
            'bid': self.current_bid,
            'ask': self.current_ask,
            'spread': self.current_spread,
            'spread_pct': (self.current_spread / self.current_price) if self.current_price else 0,
            'volatility': self.cached_volatility,
            'tape_imbalance': self.tape_imbalance,
            'halted': self.halted,
            'halt_start_time': self.halt_start_time,
            'halt_end_time': self.halt_end_time
        }

    def get_mid_price(self) -> float:
        """Get the current mid-price (average of bid and ask)."""
        if self.current_bid and self.current_ask:
            return (self.current_bid + self.current_ask) / 2
        return self.current_price