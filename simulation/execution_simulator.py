# simulation/execution_simulator.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from simulation.market_simulator import MarketSimulator


class ExecutionSimulator:
    """
    Simulates order execution with:
    - Order placement and tracking
    - Fill simulation (market and limit orders)
    - Commission and fee calculation
    - Partial fills and order modifications
    - Order cancellations
    """

    def __init__(self, market_simulator: MarketSimulator = None, config: Dict = None, logger: logging.Logger = None):
        """
        Initialize the execution simulator.

        Args:
            market_simulator: MarketSimulator instance for market dynamics
            config: Configuration dictionary with execution parameters
            logger: Optional logger
        """
        self.market_simulator = market_simulator
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Default configuration
        self.commission_per_share = self.config.get('commission_per_share', 0.0)  # Commission per share
        self.min_commission = self.config.get('min_commission', 0.0)  # Minimum commission per order
        self.max_commission = self.config.get('max_commission', float('inf'))  # Maximum commission per order
        self.sec_fee_rate = self.config.get('sec_fee_rate', 0.0000229)  # SEC fee for sell orders (currently 0.00229%)
        self.finra_taf_fee = self.config.get('finra_taf_fee', 0.000119)  # FINRA TAF fee per share (currently $0.000119)
        self.enable_partial_fills = self.config.get('enable_partial_fills', True)  # Enable partial fills
        self.partial_fill_threshold = self.config.get('partial_fill_threshold', 0.8)  # Min fill % for partial fills
        self.limit_order_expiry_seconds = self.config.get('limit_order_expiry_seconds', 5)  # Default limit order expiry

        # Order tracking
        self.orders = {}  # Dictionary of active orders
        self.order_history = []  # History of all orders
        self.next_order_id = 1000  # Starting order ID

        # Current timestamp
        self.current_timestamp = None

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def set_market_simulator(self, market_simulator: MarketSimulator) -> None:
        """
        Set the market simulator instance.

        Args:
            market_simulator: MarketSimulator instance
        """
        self.market_simulator = market_simulator

    def update_to_timestamp(self, timestamp: datetime) -> None:
        """
        Update execution state to a specific timestamp.

        Args:
            timestamp: Target timestamp to update to
        """
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
            if target_ts < current_ts:
                self._log(f"Cannot move execution backwards from {self.current_timestamp} to {timestamp}",
                          logging.WARNING)
                return

        self.current_timestamp = timestamp

        # If we have a market simulator, update it as well
        if self.market_simulator:
            self.market_simulator.update_to_timestamp(timestamp)

        # Check for expired orders
        self._process_expired_orders(timestamp)

    def _process_expired_orders(self, timestamp: datetime) -> None:
        """
        Process any orders that have expired.

        Args:
            timestamp: Current timestamp
        """
        expired_order_ids = []
        
        for order_id, order in self.orders.items():
            if order['status'] in ['new', 'partially_filled'] and order['expiry_time'] <= timestamp:
                # Order has expired
                expired_order = self.orders[order_id].copy()
                expired_order['status'] = 'expired'
                expired_order['close_timestamp'] = timestamp
                
                # Add to history
                self.order_history.append(expired_order)
                
                # Mark for removal
                expired_order_ids.append(order_id)
                
                self._log(f"Order {order_id} ({order['order_type']}, {order['side']}) expired at {timestamp}")
        
        # Remove expired orders from active orders
        for order_id in expired_order_ids:
            del self.orders[order_id]

    def _calculate_fees(self, side: str, price: float, quantity: float) -> Dict[str, float]:
        """
        Calculate fees and commissions for an order.

        Args:
            side: Order side ('buy' or 'sell')
            price: Execution price
            quantity: Shares quantity

        Returns:
            Dictionary with fee amounts
        """
        # Basic commission
        commission = self.commission_per_share * quantity
        commission = max(self.min_commission, min(self.max_commission, commission))
        
        # SEC fee (applies to sells only)
        sec_fee = 0.0
        if side.lower() == 'sell':
            sec_fee = price * quantity * self.sec_fee_rate
        
        # FINRA TAF fee
        finra_fee = self.finra_taf_fee * quantity
        
        # Total fee
        total_fee = commission + sec_fee + finra_fee
        
        return {
            'commission': commission,
            'sec_fee': sec_fee,
            'finra_fee': finra_fee,
            'total_fee': total_fee
        }

    def place_market_order(self, side: str, quantity: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            side: Order side ('buy' or 'sell')
            quantity: Shares quantity
            timestamp: Order time (defaults to current timestamp)

        Returns:
            Order information dictionary
        """
        if self.market_simulator is None:
            self._log("Cannot execute order: no market simulator set", logging.ERROR)
            return {'success': False, 'order_id': None, 'reason': 'no_market_simulator'}
        
        order_time = timestamp or self.current_timestamp
        
        if order_time is None:
            self._log("Cannot execute order: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'order_id': None, 'reason': 'no_timestamp'}
        
        # Generate order ID
        order_id = self.next_order_id
        self.next_order_id += 1
        
        # Create order record
        order = {
            'order_id': order_id,
            'side': side.lower(),
            'order_type': 'market',
            'quantity': quantity,
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'price': None,  # Market order has no price
            'status': 'new',
            'create_timestamp': order_time,
            'update_timestamp': order_time,
            'close_timestamp': None,
            'expiry_time': order_time + timedelta(seconds=self.limit_order_expiry_seconds),  # Still need an expiry
            'executions': [],
            'fees': {'commission': 0.0, 'sec_fee': 0.0, 'finra_fee': 0.0, 'total_fee': 0.0},
            'total_cost': 0.0,
            'avg_fill_price': 0.0
        }
        
        # Add to orders dictionary
        self.orders[order_id] = order
        
        # Simulate market order execution
        execution_result = self.market_simulator.simulate_market_order_execution(
            side=side,
            size=quantity,
            time=order_time
        )
        
        if execution_result['success']:
            # Order filled
            fill_price = execution_result['fill_price']
            filled_quantity = execution_result['filled_size']
            
            # Calculate fees
            fees = self._calculate_fees(side, fill_price, filled_quantity)
            
            # Calculate total cost (depends on side)
            total_cost = 0.0
            if side.lower() == 'buy':
                total_cost = (fill_price * filled_quantity) + fees['total_fee']
            else:  # sell
                total_cost = -((fill_price * filled_quantity) - fees['total_fee'])
            
            # Record execution
            execution = {
                'timestamp': execution_result['time'],
                'price': fill_price,
                'quantity': filled_quantity,
                'fees': fees,
                'latency_ms': execution_result['latency_ms']
            }
            
            # Update order record
            order['status'] = 'filled'
            order['filled_quantity'] = filled_quantity
            order['remaining_quantity'] = 0.0
            order['update_timestamp'] = execution_result['time']
            order['close_timestamp'] = execution_result['time']
            order['executions'].append(execution)
            order['fees'] = fees
            order['total_cost'] = total_cost
            order['avg_fill_price'] = fill_price
            
            # Add to history
            self.order_history.append(order.copy())
            
            # Remove from active orders
            del self.orders[order_id]
            
            self._log(f"Market order {order_id} ({side}) filled: {filled_quantity} shares @ ${fill_price:.4f}")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': 'filled',
                'fill_price': fill_price,
                'filled_quantity': filled_quantity,
                'fees': fees,
                'total_cost': total_cost,
                'timestamp': execution_result['time']
            }
        else:
            # Order failed
            order['status'] = 'rejected'
            order['close_timestamp'] = execution_result['time'] if 'time' in execution_result else order_time
            
            # Add to history
            self.order_history.append(order.copy())
            
            # Remove from active orders
            del self.orders[order_id]
            
            reason = execution_result.get('reason', 'unknown')
            self._log(f"Market order {order_id} ({side}) rejected: {reason}", logging.WARNING)
            
            return {
                'success': False,
                'order_id': order_id,
                'status': 'rejected',
                'reason': reason,
                'timestamp': execution_result.get('time', order_time)
            }

    def place_limit_order(self, side: str, quantity: float, price: float, 
                         timestamp: datetime = None,
                         expiry_seconds: int = None) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            side: Order side ('buy' or 'sell')
            quantity: Shares quantity
            price: Limit price
            timestamp: Order time (defaults to current timestamp)
            expiry_seconds: Order expiration in seconds (defaults to limit_order_expiry_seconds)

        Returns:
            Order information dictionary
        """
        if self.market_simulator is None:
            self._log("Cannot execute order: no market simulator set", logging.ERROR)
            return {'success': False, 'order_id': None, 'reason': 'no_market_simulator'}
        
        order_time = timestamp or self.current_timestamp
        
        if order_time is None:
            self._log("Cannot execute order: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'order_id': None, 'reason': 'no_timestamp'}
        
        # Use default expiry if not specified
        if expiry_seconds is None:
            expiry_seconds = self.limit_order_expiry_seconds
        
        # Generate order ID
        order_id = self.next_order_id
        self.next_order_id += 1
        
        # Create order record
        order = {
            'order_id': order_id,
            'side': side.lower(),
            'order_type': 'limit',
            'quantity': quantity,
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'price': price,
            'status': 'new',
            'create_timestamp': order_time,
            'update_timestamp': order_time,
            'close_timestamp': None,
            'expiry_time': order_time + timedelta(seconds=expiry_seconds),
            'executions': [],
            'fees': {'commission': 0.0, 'sec_fee': 0.0, 'finra_fee': 0.0, 'total_fee': 0.0},
            'total_cost': 0.0,
            'avg_fill_price': 0.0
        }
        
        # Add to orders dictionary
        self.orders[order_id] = order
        
        # Simulate limit order execution
        execution_result = self.market_simulator.simulate_limit_order_execution(
            side=side,
            size=quantity,
            price=price,
            time=order_time,
            expiration_time=order['expiry_time']
        )
        
        if execution_result['success']:
            # Order filled
            fill_price = execution_result['fill_price']
            filled_quantity = execution_result['filled_size']
            
            # Calculate fees
            fees = self._calculate_fees(side, fill_price, filled_quantity)
            
            # Calculate total cost (depends on side)
            total_cost = 0.0
            if side.lower() == 'buy':
                total_cost = (fill_price * filled_quantity) + fees['total_fee']
            else:  # sell
                total_cost = -((fill_price * filled_quantity) - fees['total_fee'])
            
            # Record execution
            execution = {
                'timestamp': execution_result['time'],
                'price': fill_price,
                'quantity': filled_quantity,
                'fees': fees,
                'latency_ms': execution_result['latency_ms']
            }
            
            # Update order record
            order['status'] = 'filled'
            order['filled_quantity'] = filled_quantity
            order['remaining_quantity'] = 0.0
            order['update_timestamp'] = execution_result['time']
            order['close_timestamp'] = execution_result['time']
            order['executions'].append(execution)
            order['fees'] = fees
            order['total_cost'] = total_cost
            order['avg_fill_price'] = fill_price
            
            # Add to history
            self.order_history.append(order.copy())
            
            # Remove from active orders
            del self.orders[order_id]
            
            self._log(f"Limit order {order_id} ({side} @ ${price:.4f}) filled: {filled_quantity} shares @ ${fill_price:.4f}")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': 'filled',
                'fill_price': fill_price,
                'filled_quantity': filled_quantity,
                'fees': fees,
                'total_cost': total_cost,
                'timestamp': execution_result['time']
            }
        else:
            # Order failed or expired
            if execution_result.get('reason') == 'expired_unfilled':
                order['status'] = 'expired'
                self._log(f"Limit order {order_id} ({side} @ ${price:.4f}) expired unfilled")
            else:
                order['status'] = 'rejected'
                self._log(f"Limit order {order_id} ({side} @ ${price:.4f}) rejected: {execution_result.get('reason', 'unknown')}", logging.WARNING)
            
            order['close_timestamp'] = execution_result['time'] if 'time' in execution_result else order_time
            
            # Add to history
            self.order_history.append(order.copy())
            
            # Remove from active orders
            del self.orders[order_id]
            
            return {
                'success': False,
                'order_id': order_id,
                'status': order['status'],
                'reason': execution_result.get('reason', 'unknown'),
                'timestamp': execution_result.get('time', order_time)
            }

    def cancel_order(self, order_id: int, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel
            timestamp: Cancellation time (defaults to current timestamp)

        Returns:
            Order information dictionary
        """
        if order_id not in self.orders:
            return {
                'success': False,
                'order_id': order_id,
                'reason': 'order_not_found'
            }
        
        cancel_time = timestamp or self.current_timestamp
        
        if cancel_time is None:
            self._log("Cannot cancel order: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'order_id': order_id, 'reason': 'no_timestamp'}
        
        order = self.orders[order_id]
        
        # Cannot cancel already-filled orders
        if order['status'] not in ['new', 'partially_filled']:
            return {
                'success': False,
                'order_id': order_id,
                'reason': f"order_status_{order['status']}"
            }
        
        # Update order
        order['status'] = 'cancelled'
        order['update_timestamp'] = cancel_time
        order['close_timestamp'] = cancel_time
        
        # Add to history
        self.order_history.append(order.copy())
        
        # Remove from active orders
        del self.orders[order_id]
        
        self._log(f"Order {order_id} cancelled: {order['filled_quantity']} of {order['quantity']} shares filled")
        
        return {
            'success': True,
            'order_id': order_id,
            'status': 'cancelled',
            'filled_quantity': order['filled_quantity'],
            'remaining_quantity': order['remaining_quantity'],
            'timestamp': cancel_time
        }

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of all open orders.

        Returns:
            List of open order dictionaries
        """
        return list(self.orders.values())

    def get_order_by_id(self, order_id: int) -> Dict[str, Any]:
        """
        Get details of an order by ID.

        Args:
            order_id: Order ID to look up

        Returns:
            Order dictionary, or None if not found
        """
        # Check active orders first
        if order_id in self.orders:
            return self.orders[order_id]
        
        # Check order history
        for order in self.order_history:
            if order['order_id'] == order_id:
                return order
        
        return None

    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Get list of all historical orders.

        Returns:
            List of order dictionaries
        """
        return self.order_history