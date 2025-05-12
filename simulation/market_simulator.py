# simulation/market_simulator.py
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta


class MarketSimulator:
    """Bare-bones market simulator that provides minimal functionality"""

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Minimal state
        self.current_price = 10.0
        self.current_bid = 9.95
        self.current_ask = 10.05
        self.current_spread = 0.10
        self.current_timestamp = None
        self.halted = False

        # Data references
        self.quotes_df = None
        self.trades_df = None
        self.bars_df = None
        self.status_df = None

    def _log(self, message: str, level: int = logging.INFO):
        if self.logger:
            self.logger.log(level, message)

    def initialize_from_data(self, quotes_df: pd.DataFrame, trades_df: pd.DataFrame,
                             bars_df: pd.DataFrame, status_df: pd.DataFrame = None) -> None:
        """Initialize the market simulator with actual market data."""
        self.quotes_df = quotes_df
        self.trades_df = trades_df
        self.bars_df = bars_df
        self.status_df = status_df

        # Find first valid data across all sources
        valid_price_found = False

        # Try to initialize from quotes first
        if not quotes_df.empty and 'bid_px_00' in quotes_df.columns and 'ask_px_00' in quotes_df.columns:
            # Make sure we find a row with valid bid/ask
            for i, row in quotes_df.iterrows():
                if pd.notna(row['bid_px_00']) and pd.notna(row['ask_px_00']):
                    self.current_bid = row['bid_px_00']
                    self.current_ask = row['ask_px_00']
                    self.current_spread = self.current_ask - self.current_bid
                    self.current_price = (self.current_bid + self.current_ask) / 2
                    self.current_timestamp = i  # Use row index as timestamp
                    valid_price_found = True
                    self._log(
                        f"Initialized market from quotes: Price=${self.current_price:.4f}, Bid=${self.current_bid:.4f}, Ask=${self.current_ask:.4f}")
                    break

        # If quotes didn't provide valid data, try trades
        if not valid_price_found and not trades_df.empty and 'price' in trades_df.columns:
            for i, row in trades_df.iterrows():
                if pd.notna(row['price']) and row['price'] > 0:
                    self.current_price = row['price']
                    # Create a simple spread
                    self.current_bid = self.current_price * 0.995  # 0.5% below price
                    self.current_ask = self.current_price * 1.005  # 0.5% above price
                    self.current_spread = self.current_ask - self.current_bid
                    self.current_timestamp = i
                    valid_price_found = True
                    self._log(f"Initialized market from trades: Price=${self.current_price:.4f}")
                    break

        # If still not initialized, try bars
        if not valid_price_found and not bars_df.empty:
            for i, row in bars_df.iterrows():
                if 'open' in row and pd.notna(row['open']) and row['open'] > 0:
                    self.current_price = row['open']
                    # Create a simple spread
                    self.current_bid = self.current_price * 0.995  # 0.5% below price
                    self.current_ask = self.current_price * 1.005  # 0.5% above price
                    self.current_spread = self.current_ask - self.current_bid
                    self.current_timestamp = i
                    valid_price_found = True
                    self._log(f"Initialized market from bars: Price=${self.current_price:.4f}")
                    break

        # If still not initialized, set default values
        if not valid_price_found:
            self._log("WARNING: Could not find valid price data, using default values", logging.WARNING)
            self.current_price = 10.0  # Default test price
            self.current_bid = 9.95
            self.current_ask = 10.05
            self.current_spread = 0.10
            self.current_timestamp = datetime.now() if self.quotes_df.empty else self.quotes_df.index[0]

        self._log(
            f"Market initialized: Price=${self.current_price:.4f}, Bid=${self.current_bid:.4f}, Ask=${self.current_ask:.4f}, Spread=${self.current_spread:.4f}")

    def update_to_timestamp(self, timestamp: datetime) -> None:
        """Simply update the current timestamp"""
        # Basic timestamp validation
        if self.current_timestamp and timestamp < self.current_timestamp:
            self._log(f"Cannot move backwards from {self.current_timestamp} to {timestamp}", logging.WARNING)
            return

        self.current_timestamp = timestamp

        # Extremely simple price update from data
        if self.quotes_df is not None and not self.quotes_df.empty:
            quotes_before = self.quotes_df[self.quotes_df.index <= timestamp]
            if not quotes_before.empty:
                row = quotes_before.iloc[-1]
                if 'bid_px_00' in row and 'ask_px_00' in row:
                    self.current_bid = row['bid_px_00']
                    self.current_ask = row['ask_px_00']
                    self.current_price = (self.current_bid + self.current_ask) / 2
                    self.current_spread = self.current_ask - self.current_bid

    def simulate_market_order_execution(self, side: str, size: float, time: datetime = None) -> Dict[str, Any]:
        """Simulate order execution with minimal logic"""
        execution_time = time or self.current_timestamp

        # Always succeed with a simple price
        fill_price = self.current_ask if side.lower() == 'buy' else self.current_bid

        self._log(f"{side.upper()} order filled: {size} shares @ ${fill_price:.4f}")

        return {
            'success': True,
            'filled_size': size,
            'fill_price': fill_price,
            'time': execution_time,
            'latency_ms': 100,  # Fixed latency
            'reason': 'filled'
        }

    def simulate_limit_order_execution(self, side: str, size: float, price: float,
                                       time: datetime = None,
                                       expiration_time: datetime = None) -> Dict[str, Any]:
        """Minimal limit order implementation - always fills"""
        execution_time = time or self.current_timestamp

        # For bare minimum, just execute like a market order
        return self.simulate_market_order_execution(side, size, execution_time)

    def get_current_market_state(self) -> Dict[str, Any]:
        """Return minimal market state"""
        return {
            'timestamp': self.current_timestamp,
            'price': self.current_price,
            'bid': self.current_bid,
            'ask': self.current_ask,
            'spread': self.current_spread,
            'halted': False
        }

    def get_mid_price(self) -> float:
        """Return mid price"""
        return self.current_price