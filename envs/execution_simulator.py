# simulation/execution_simulator.py
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta


class ExecutionSimulator:
    """Minimal execution simulator that provides basic order tracking"""

    def __init__(self, market_simulator=None, config: Dict = None, logger: logging.Logger = None):
        self.market_simulator = market_simulator
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Minimal state
        self.orders = {}  # Orders dictionary
        self.order_history = []  # Order history
        self.next_order_id = 1000  # Next order ID
        self.current_timestamp = None  # Current timestamp

    def _log(self, message: str, level: int = logging.INFO):
        if self.logger:
            self.logger.log(level, message)

    def set_market_simulator(self, market_simulator) -> None:
        """Set the market simulator"""
        self.market_simulator = market_simulator

    def update_to_timestamp(self, timestamp: datetime) -> None:
        """Update execution state to a specific timestamp"""
        self.current_timestamp = timestamp

        # Update market simulator if available
        if self.market_simulator:
            self.market_simulator.update_to_timestamp(timestamp)

    def place_market_order(self, side: str, quantity: float, timestamp: datetime = None) -> Dict[str, Any]:
        """Place a market order with minimal logic"""
        order_time = timestamp or self.current_timestamp

        # Generate order ID
        order_id = self.next_order_id
        self.next_order_id += 1

        # Create minimal order record
        order = {
            'order_id': order_id,
            'side': side,
            'quantity': quantity,
            'order_type': 'market',
            'status': 'new',
            'create_timestamp': order_time
        }

        # Execute the order through market simulator
        if self.market_simulator:
            execution = self.market_simulator.simulate_market_order_execution(side, quantity, order_time)

            if execution['success']:
                order['status'] = 'filled'
                order['fill_price'] = execution['fill_price']
                order['fill_time'] = execution['time']

                self._log(
                    f"Market order {order_id} ({side}) filled: {quantity} shares @ ${execution['fill_price']:.4f}")

                # Store in history and remove from active orders
                self.order_history.append(order)

                return {
                    'success': True,
                    'order_id': order_id,
                    'status': 'filled',
                    'side': side,
                    'fill_price': execution['fill_price'],
                    'filled_quantity': quantity,
                    'timestamp': execution['time']
                }
            else:
                order['status'] = 'rejected'
                self.order_history.append(order)
                return {
                    'success': False,
                    'order_id': order_id,
                    'status': 'rejected',
                    'reason': execution.get('reason', 'unknown')
                }
        else:
            # No market simulator, just assume success
            order['status'] = 'filled'
            order['fill_price'] = 10.0  # Default price
            order['fill_time'] = order_time

            self.order_history.append(order)

            return {
                'success': True,
                'order_id': order_id,
                'status': 'filled',
                'side': side,
                'fill_price': 10.0,
                'filled_quantity': quantity,
                'timestamp': order_time
            }

    def place_limit_order(self, side: str, quantity: float, price: float,
                          timestamp: datetime = None,
                          expiry_seconds: int = None) -> Dict[str, Any]:
        """Place a limit order - simplified to behave like a market order"""
        # For ultra-minimal implementation, just use market order logic
        return self.place_market_order(side, quantity, timestamp)

    def cancel_order(self, order_id: int, timestamp: datetime = None) -> Dict[str, Any]:
        """Cancel an active order - simplified to just return success"""
        cancel_time = timestamp or self.current_timestamp

        return {
            'success': True,
            'order_id': order_id,
            'status': 'cancelled',
            'timestamp': cancel_time
        }

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get list of all open orders"""
        return list(self.orders.values())

    def get_order_by_id(self, order_id: int) -> Dict[str, Any]:
        """Get details of an order by ID"""
        # Check active orders
        if order_id in self.orders:
            return self.orders[order_id]

        # Check order history
        for order in self.order_history:
            if order.get('order_id') == order_id:
                return order

        return None

    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get list of all historical orders"""
        return self.order_history