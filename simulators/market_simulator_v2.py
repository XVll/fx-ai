"""Enhanced Market Simulator with O(1) lookups and future buffer support.

This simulator provides efficient market state queries and realistic execution
simulation for the enhanced data layer architecture.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from dataclasses import dataclass

from data.data_manager import DataManager
from data.provider.data_provider import UnifiedDataProvider


@dataclass
class MarketState:
    """Point-in-time market state."""
    timestamp: pd.Timestamp
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float
    volume: float
    is_halted: bool
    spread: float


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""
    order_id: str
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    requested_price: float
    executed_price: float
    requested_size: float
    executed_size: float
    slippage: float
    commission: float
    latency_ms: float
    rejection_reason: Optional[str] = None


class MarketSimulatorV2:
    """Enhanced market simulator with efficient lookups and realistic execution.
    
    Features:
    - O(1) timestamp lookups using hash indices
    - Future buffer for execution simulation
    - Intelligent state interpolation
    - Halt and status awareness
    - Reset point context integration
    """
    
    def __init__(self, data_manager: DataManager,
                 future_buffer_minutes: int = 5,
                 default_latency_ms: float = 100,
                 commission_per_share: float = 0.005,
                 logger: Optional[logging.Logger] = None):
        """Initialize the market simulator.
        
        Args:
            data_manager: Data manager instance with loaded day data
            future_buffer_minutes: Minutes of future data to maintain
            default_latency_ms: Default execution latency in milliseconds
            commission_per_share: Commission per share traded
            logger: Optional logger
        """
        self.data_manager = data_manager
        self.future_buffer_minutes = future_buffer_minutes
        self.default_latency_ms = default_latency_ms
        self.commission_per_share = commission_per_share
        self.logger = logger or logging.getLogger(__name__)
        
        # Time-indexed lookups for O(1) access
        self.ohlcv_index = {}  # {timestamp: row_index}
        self.quote_index = {}  # {timestamp: row_index}
        self.trade_index = {}  # {timestamp: row_index}
        self.status_index = {}  # {timestamp: status}
        
        # Cached data references
        self.ohlcv_1s = None
        self.quotes = None
        self.trades = None
        self.status = None
        
        # Current state
        self.current_symbol = None
        self.current_date = None
        self.current_timestamp = None
        
        # Execution tracking
        self.execution_history = []
        
    def initialize_day(self, symbol: str, date: datetime):
        """Initialize simulator for a specific trading day.
        
        Args:
            symbol: Symbol to simulate
            date: Trading date
        """
        self.current_symbol = symbol
        self.current_date = pd.Timestamp(date).date()
        
        # Get data from L1 cache
        self.ohlcv_1s = self.data_manager.get_active_day_data('bars_1s')
        self.quotes = self.data_manager.get_active_day_data('quotes')
        self.trades = self.data_manager.get_active_day_data('trades')
        self.status = self.data_manager.get_active_day_data('status')
        
        # Build time indices
        self._build_indices()
        
        self.logger.info(f"Initialized market simulator for {symbol} on {date}")
        
    def _build_indices(self):
        """Build hash indices for O(1) timestamp lookups."""
        # Clear existing indices
        self.ohlcv_index.clear()
        self.quote_index.clear()
        self.trade_index.clear()
        self.status_index.clear()
        
        # Build OHLCV index
        if self.ohlcv_1s is not None and not self.ohlcv_1s.empty:
            for idx, timestamp in enumerate(self.ohlcv_1s.index):
                self.ohlcv_index[timestamp] = idx
                
        # Build quote index
        if self.quotes is not None and not self.quotes.empty:
            for idx, timestamp in enumerate(self.quotes.index):
                self.quote_index[timestamp] = idx
                
        # Build trade index
        if self.trades is not None and not self.trades.empty:
            for idx, timestamp in enumerate(self.trades.index):
                self.trade_index[timestamp] = idx
                
        # Build status index (store actual status, not index)
        if self.status is not None and not self.status.empty:
            for _, row in self.status.iterrows():
                self.status_index[row.name] = {
                    'is_halted': row.get('is_halted', False),
                    'is_trading': row.get('is_trading', True),
                    'status': row.get('status', 'TRADING')
                }
                
    def get_market_state(self, timestamp: pd.Timestamp) -> MarketState:
        """Get market state at exact timestamp with O(1) lookup.
        
        Args:
            timestamp: Timestamp to query
            
        Returns:
            MarketState object
        """
        self.current_timestamp = timestamp
        
        # Get OHLCV data
        ohlcv_data = self._get_ohlcv_at_timestamp(timestamp)
        
        # Get quote data
        quote_data = self._get_quote_at_timestamp(timestamp)
        
        # Get trade data for last price
        trade_data = self._get_last_trade_before_timestamp(timestamp)
        
        # Get status
        is_halted, is_trading = self._get_trading_status_at_timestamp(timestamp)
        
        # Construct market state
        bid_price = quote_data.get('bid_price', ohlcv_data.get('close', 0))
        ask_price = quote_data.get('ask_price', ohlcv_data.get('close', 0))
        
        # Ensure valid spread
        if bid_price >= ask_price or bid_price <= 0 or ask_price <= 0:
            # Use OHLCV to construct synthetic spread
            last_price = ohlcv_data.get('close', 0)
            if last_price > 0:
                spread = max(0.01, last_price * 0.001)  # 0.1% minimum spread
                bid_price = last_price - spread / 2
                ask_price = last_price + spread / 2
            else:
                bid_price = 0
                ask_price = 0
                
        return MarketState(
            timestamp=timestamp,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=quote_data.get('bid_size', 100),
            ask_size=quote_data.get('ask_size', 100),
            last_price=trade_data.get('price', ohlcv_data.get('close', 0)),
            last_size=trade_data.get('size', 0),
            volume=ohlcv_data.get('volume', 0),
            is_halted=is_halted,
            spread=ask_price - bid_price if ask_price > bid_price else 0
        )
        
    def _get_ohlcv_at_timestamp(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get OHLCV data at or before timestamp."""
        if self.ohlcv_1s is None or self.ohlcv_1s.empty:
            return {}
            
        # Try exact match first (O(1))
        if timestamp in self.ohlcv_index:
            row = self.ohlcv_1s.iloc[self.ohlcv_index[timestamp]]
            return row.to_dict()
            
        # Find closest previous timestamp
        # Get all timestamps less than or equal to target
        valid_times = [ts for ts in self.ohlcv_index.keys() if ts <= timestamp]
        
        if not valid_times:
            return {}
            
        closest_time = max(valid_times)
        row = self.ohlcv_1s.iloc[self.ohlcv_index[closest_time]]
        return row.to_dict()
        
    def _get_quote_at_timestamp(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get quote data at or before timestamp."""
        if self.quotes is None or self.quotes.empty:
            return {}
            
        # Try exact match first (O(1))
        if timestamp in self.quote_index:
            row = self.quotes.iloc[self.quote_index[timestamp]]
            return row.to_dict()
            
        # Find closest previous timestamp
        valid_times = [ts for ts in self.quote_index.keys() if ts <= timestamp]
        
        if not valid_times:
            return {}
            
        closest_time = max(valid_times)
        row = self.quotes.iloc[self.quote_index[closest_time]]
        return row.to_dict()
        
    def _get_last_trade_before_timestamp(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Get last trade before or at timestamp."""
        if self.trades is None or self.trades.empty:
            return {}
            
        valid_times = [ts for ts in self.trade_index.keys() if ts <= timestamp]
        
        if not valid_times:
            return {}
            
        last_trade_time = max(valid_times)
        row = self.trades.iloc[self.trade_index[last_trade_time]]
        return row.to_dict()
        
    def _get_trading_status_at_timestamp(self, timestamp: pd.Timestamp) -> Tuple[bool, bool]:
        """Get trading status at timestamp."""
        # Check status index
        if timestamp in self.status_index:
            status = self.status_index[timestamp]
            return status['is_halted'], status['is_trading']
            
        # Find most recent status before timestamp
        valid_times = [ts for ts in self.status_index.keys() if ts <= timestamp]
        
        if valid_times:
            recent_time = max(valid_times)
            status = self.status_index[recent_time]
            return status['is_halted'], status['is_trading']
            
        # Default to trading allowed
        return False, True
        
    def get_future_prices(self, timestamp: pd.Timestamp, 
                         seconds_ahead: int = None) -> pd.DataFrame:
        """Get future price data for execution simulation.
        
        Args:
            timestamp: Current timestamp
            seconds_ahead: Seconds to look ahead (default: future_buffer_minutes * 60)
            
        Returns:
            DataFrame with future OHLCV data
        """
        if seconds_ahead is None:
            seconds_ahead = self.future_buffer_minutes * 60
            
        end_timestamp = timestamp + pd.Timedelta(seconds=seconds_ahead)
        
        if self.ohlcv_1s is None or self.ohlcv_1s.empty:
            return pd.DataFrame()
            
        # Get slice of future data
        mask = (self.ohlcv_1s.index > timestamp) & (self.ohlcv_1s.index <= end_timestamp)
        return self.ohlcv_1s[mask].copy()
        
    def simulate_order_execution(self, timestamp: pd.Timestamp, side: str,
                               size: float, order_type: str = 'market',
                               limit_price: Optional[float] = None) -> ExecutionResult:
        """Simulate realistic order execution with latency and slippage.
        
        Args:
            timestamp: Order submission time
            side: 'buy' or 'sell'
            size: Number of shares
            order_type: 'market' or 'limit'
            limit_price: Limit price for limit orders
            
        Returns:
            ExecutionResult with execution details
        """
        order_id = f"{timestamp.strftime('%Y%m%d%H%M%S%f')}_{side}_{size}"
        
        # Get current market state
        current_state = self.get_market_state(timestamp)
        
        # Check if trading is allowed
        if current_state.is_halted:
            return ExecutionResult(
                order_id=order_id,
                timestamp=timestamp,
                symbol=self.current_symbol,
                side=side,
                requested_price=limit_price or current_state.ask_price if side == 'buy' else current_state.bid_price,
                executed_price=0,
                requested_size=size,
                executed_size=0,
                slippage=0,
                commission=0,
                latency_ms=0,
                rejection_reason="Trading halted"
            )
            
        # Calculate execution latency
        latency_ms = self._calculate_latency(timestamp, size)
        execution_timestamp = timestamp + pd.Timedelta(milliseconds=latency_ms)
        
        # Get market state at execution time
        execution_state = self.get_market_state(execution_timestamp)
        
        # Calculate execution price with slippage
        if order_type == 'market':
            executed_price, slippage = self._calculate_market_execution_price(
                execution_state, side, size, execution_timestamp
            )
            executed_size = size
        else:  # limit order
            executed_price, executed_size, slippage = self._calculate_limit_execution(
                execution_state, side, size, limit_price, execution_timestamp
            )
            
        # Calculate commission
        commission = executed_size * self.commission_per_share
        
        # Create execution result
        result = ExecutionResult(
            order_id=order_id,
            timestamp=execution_timestamp,
            symbol=self.current_symbol,
            side=side,
            requested_price=limit_price or (current_state.ask_price if side == 'buy' else current_state.bid_price),
            executed_price=executed_price,
            requested_size=size,
            executed_size=executed_size,
            slippage=slippage,
            commission=commission,
            latency_ms=latency_ms
        )
        
        # Store execution history
        self.execution_history.append(result)
        
        return result
        
    def _calculate_latency(self, timestamp: pd.Timestamp, size: float) -> float:
        """Calculate realistic execution latency."""
        # Base latency
        latency = self.default_latency_ms
        
        # Add size-based latency (larger orders take longer)
        if size > 10000:
            latency += 50
        elif size > 5000:
            latency += 25
            
        # Add time-of-day factor
        hour = timestamp.hour
        if 9 <= hour <= 10:  # Market open
            latency *= 1.5
        elif 15 <= hour <= 16:  # Market close
            latency *= 1.3
            
        # Add random jitter
        jitter = np.random.normal(0, latency * 0.1)
        
        return max(10, latency + jitter)  # Minimum 10ms
        
    def _calculate_market_execution_price(self, state: MarketState, side: str,
                                        size: float, exec_timestamp: pd.Timestamp) -> Tuple[float, float]:
        """Calculate market order execution price with slippage."""
        # Base price
        if side == 'buy':
            base_price = state.ask_price
        else:
            base_price = state.bid_price
            
        if base_price <= 0:
            return 0, 0
            
        # Calculate market impact based on size
        avg_trade_size = 1000  # Assumed average
        size_ratio = size / avg_trade_size
        
        # Base slippage from spread
        spread_slippage = state.spread * 0.5
        
        # Size-based slippage
        size_slippage = base_price * 0.0001 * size_ratio  # 1bp per avg size
        
        # Volatility-based slippage
        future_prices = self.get_future_prices(exec_timestamp, seconds_ahead=60)
        if not future_prices.empty:
            volatility = future_prices['close'].pct_change().std()
            vol_slippage = base_price * volatility * 2
        else:
            vol_slippage = base_price * 0.001
            
        # Total slippage
        total_slippage = spread_slippage + size_slippage + vol_slippage
        
        # Apply slippage
        if side == 'buy':
            executed_price = base_price + total_slippage
        else:
            executed_price = base_price - total_slippage
            
        return max(0.01, executed_price), total_slippage
        
    def _calculate_limit_execution(self, state: MarketState, side: str,
                                 size: float, limit_price: float,
                                 exec_timestamp: pd.Timestamp) -> Tuple[float, float, float]:
        """Calculate limit order execution."""
        # Check if limit price is marketable
        if side == 'buy' and limit_price >= state.ask_price:
            # Marketable buy limit - execute as market
            executed_price, slippage = self._calculate_market_execution_price(
                state, side, size, exec_timestamp
            )
            return executed_price, size, slippage
        elif side == 'sell' and limit_price <= state.bid_price:
            # Marketable sell limit - execute as market
            executed_price, slippage = self._calculate_market_execution_price(
                state, side, size, exec_timestamp
            )
            return executed_price, size, slippage
        else:
            # Non-marketable limit order - would need to check if it gets filled
            # For now, assume no fill
            return 0, 0, 0
            
    def interpolate_state(self, timestamp: pd.Timestamp) -> MarketState:
        """Interpolate market state between available data points.
        
        This provides smooth state transitions even with sparse data.
        """
        # Get surrounding data points
        before_state = None
        after_state = None
        
        # Find closest timestamps before and after
        ohlcv_times = sorted(self.ohlcv_index.keys())
        
        before_times = [t for t in ohlcv_times if t <= timestamp]
        after_times = [t for t in ohlcv_times if t > timestamp]
        
        if before_times:
            before_state = self.get_market_state(before_times[-1])
        if after_times:
            after_state = self.get_market_state(after_times[0])
            
        # If we have both, interpolate
        if before_state and after_state:
            # Calculate interpolation weight
            total_seconds = (after_state.timestamp - before_state.timestamp).total_seconds()
            elapsed_seconds = (timestamp - before_state.timestamp).total_seconds()
            weight = elapsed_seconds / total_seconds if total_seconds > 0 else 0
            
            # Interpolate prices
            bid_price = before_state.bid_price + (after_state.bid_price - before_state.bid_price) * weight
            ask_price = before_state.ask_price + (after_state.ask_price - before_state.ask_price) * weight
            
            # Use most recent volume and status
            return MarketState(
                timestamp=timestamp,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=before_state.bid_size,
                ask_size=before_state.ask_size,
                last_price=before_state.last_price,
                last_size=before_state.last_size,
                volume=before_state.volume,
                is_halted=before_state.is_halted,
                spread=ask_price - bid_price
            )
        elif before_state:
            # Only have before state
            return MarketState(
                timestamp=timestamp,
                bid_price=before_state.bid_price,
                ask_price=before_state.ask_price,
                bid_size=before_state.bid_size,
                ask_size=before_state.ask_size,
                last_price=before_state.last_price,
                last_size=before_state.last_size,
                volume=before_state.volume,
                is_halted=before_state.is_halted,
                spread=before_state.spread
            )
        else:
            # No data available
            return MarketState(
                timestamp=timestamp,
                bid_price=0,
                ask_price=0,
                bid_size=0,
                ask_size=0,
                last_price=0,
                last_size=0,
                volume=0,
                is_halted=True,
                spread=0
            )
            
    def get_historical_bars(self, until_timestamp: pd.Timestamp, 
                           lookback_minutes: int = None) -> pd.DataFrame:
        """Get historical OHLCV bars up to specified timestamp.
        
        Args:
            until_timestamp: Get bars up to this timestamp (inclusive)
            lookback_minutes: How many minutes to look back (None = all available)
            
        Returns:
            DataFrame with OHLCV data up to timestamp
        """
        if self.ohlcv_1s is None or self.ohlcv_1s.empty:
            return pd.DataFrame()
            
        # Filter data up to timestamp (no future data leakage)
        filtered_data = self.ohlcv_1s[self.ohlcv_1s.index <= until_timestamp]
        
        if lookback_minutes is not None and not filtered_data.empty:
            start_time = until_timestamp - pd.Timedelta(minutes=lookback_minutes)
            filtered_data = filtered_data[filtered_data.index >= start_time]
            
        return filtered_data.copy()
    
    def get_historical_quotes(self, until_timestamp: pd.Timestamp,
                             lookback_minutes: int = None) -> pd.DataFrame:
        """Get historical quote data up to specified timestamp.
        
        Args:
            until_timestamp: Get quotes up to this timestamp (inclusive)
            lookback_minutes: How many minutes to look back (None = all available)
            
        Returns:
            DataFrame with quote data up to timestamp
        """
        if self.quotes is None or self.quotes.empty:
            return pd.DataFrame()
            
        filtered_data = self.quotes[self.quotes.index <= until_timestamp]
        
        if lookback_minutes is not None and not filtered_data.empty:
            start_time = until_timestamp - pd.Timedelta(minutes=lookback_minutes)
            filtered_data = filtered_data[filtered_data.index >= start_time]
            
        return filtered_data.copy()
    
    def get_historical_trades(self, until_timestamp: pd.Timestamp,
                             lookback_minutes: int = None) -> pd.DataFrame:
        """Get historical trade data up to specified timestamp.
        
        Args:
            until_timestamp: Get trades up to this timestamp (inclusive) 
            lookback_minutes: How many minutes to look back (None = all available)
            
        Returns:
            DataFrame with trade data up to timestamp
        """
        if self.trades is None or self.trades.empty:
            return pd.DataFrame()
            
        filtered_data = self.trades[self.trades.index <= until_timestamp]
        
        if lookback_minutes is not None and not filtered_data.empty:
            start_time = until_timestamp - pd.Timedelta(minutes=lookback_minutes)
            filtered_data = filtered_data[filtered_data.index >= start_time]
            
        return filtered_data.copy()
        
    def get_current_bars(self, lookback_minutes: int = 60) -> pd.DataFrame:
        """Get historical bars up to current timestamp.
        
        Args:
            lookback_minutes: How many minutes to look back
            
        Returns:
            DataFrame with recent OHLCV data
        """
        if self.current_timestamp is None:
            return pd.DataFrame()
            
        return self.get_historical_bars(self.current_timestamp, lookback_minutes)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution simulation statistics."""
        if not self.execution_history:
            return {}
            
        df = pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'side': e.side,
                'executed_size': e.executed_size,
                'slippage': e.slippage,
                'commission': e.commission,
                'latency_ms': e.latency_ms
            }
            for e in self.execution_history
        ])
        
        return {
            'total_executions': len(self.execution_history),
            'total_shares': df['executed_size'].sum(),
            'avg_slippage': df['slippage'].mean(),
            'total_slippage': df['slippage'].sum(),
            'total_commission': df['commission'].sum(),
            'avg_latency_ms': df['latency_ms'].mean(),
            'buy_count': len(df[df['side'] == 'buy']),
            'sell_count': len(df[df['side'] == 'sell'])
        }
        
    def reset(self):
        """Reset simulator state."""
        self.ohlcv_index.clear()
        self.quote_index.clear()
        self.trade_index.clear()
        self.status_index.clear()
        
        self.ohlcv_1s = None
        self.quotes = None
        self.trades = None
        self.status = None
        
        self.current_symbol = None
        self.current_date = None
        self.current_timestamp = None
        
        self.execution_history.clear()