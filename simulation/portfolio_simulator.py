# simulation/portfolio_simulator.py
from typing import Dict, List, Any
import logging
from datetime import datetime


class PortfolioSimulator:
    """Minimal portfolio simulator that tracks positions and P&L"""

    def __init__(self, market_simulator=None, execution_simulator=None,
                 config: Dict = None, logger: logging.Logger = None):
        self.market_simulator = market_simulator
        self.execution_simulator = execution_simulator
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Configuration
        self.initial_cash = self.config.get('initial_cash', 100000.0)

        # Portfolio state
        self.cash = self.initial_cash
        self.positions = {}  # Symbol -> position details
        self.portfolio_value_history = {}  # Timestamp -> total value
        self.trade_history = []  # Completed trades
        self.current_timestamp = None

    def _log(self, message: str, level: int = logging.INFO):
        if self.logger:
            self.logger.log(level, message)

    def set_market_simulator(self, market_simulator) -> None:
        """Set the market simulator"""
        self.market_simulator = market_simulator

    def set_execution_simulator(self, execution_simulator) -> None:
        """Set the execution simulator"""
        self.execution_simulator = execution_simulator

    def reset(self) -> None:
        """Reset the portfolio simulator"""
        self.cash = self.initial_cash
        self.positions = {}
        self.portfolio_value_history = {}
        self.trade_history = []
        self.current_timestamp = None

    def update_to_timestamp(self, timestamp: datetime) -> None:
        """Update portfolio state to a specific timestamp"""
        self.current_timestamp = timestamp

        # Update position values using current prices
        if self.market_simulator:
            self.market_simulator.update_to_timestamp(timestamp)

            for symbol, position in self.positions.items():
                # Update position value with current price
                current_price = self.market_simulator.current_price
                position['current_price'] = current_price
                position['current_value'] = position['quantity'] * current_price
                position['unrealized_pnl'] = (current_price - position['avg_price']) * position['quantity']

        # Record portfolio value
        total_value = self.cash
        for symbol, position in self.positions.items():
            total_value += position.get('current_value', 0)

        self.portfolio_value_history[timestamp] = total_value

    def execute_market_order(self, symbol: str, quantity: float, timestamp: datetime = None) -> Dict[str, Any]:
        """Execute a market order - simplified logic"""
        if self.execution_simulator is None:
            self._log("No execution simulator set", logging.ERROR)
            return {'success': False, 'reason': 'no_executor'}

        order_time = timestamp or self.current_timestamp

        # Determine order side
        side = 'buy' if quantity > 0 else 'sell'
        abs_quantity = abs(quantity)

        # Execute through execution simulator
        result = self.execution_simulator.place_market_order(
            side=side,
            quantity=abs_quantity,
            timestamp=order_time
        )

        if not result['success']:
            return result

        # Update portfolio
        fill_price = result['fill_price']
        filled_quantity = result['filled_quantity']

        # Update cash
        if side == 'buy':
            self.cash -= fill_price * filled_quantity
        else:
            self.cash += fill_price * filled_quantity

        # Update position
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = {
                'quantity': filled_quantity if side == 'buy' else -filled_quantity,
                'avg_price': fill_price,
                'current_price': fill_price,
                'current_value': fill_price * filled_quantity,
                'unrealized_pnl': 0.0,
                'open_time': order_time
            }
        else:
            # Update existing position
            current_position = self.positions[symbol]
            old_quantity = current_position['quantity']
            old_avg_price = current_position['avg_price']

            # Calculate new position
            new_quantity = old_quantity + (filled_quantity if side == 'buy' else -filled_quantity)

            # Calculate new average price (simplified)
            if new_quantity != 0:
                new_avg_price = ((old_quantity * old_avg_price) +
                                 (
                                     filled_quantity * fill_price if side == 'buy' else -filled_quantity * fill_price)) / new_quantity
            else:
                new_avg_price = 0

            # Handle position close or reversal
            if (old_quantity > 0 and new_quantity <= 0) or (old_quantity < 0 and new_quantity >= 0):
                # Position closed or reversed
                realized_pnl = (fill_price - old_avg_price) * min(abs(old_quantity), abs_quantity)

                # Record completed trade
                if old_quantity != 0:
                    trade = {
                        'symbol': symbol,
                        'open_time': current_position['open_time'],
                        'close_time': order_time,
                        'entry_price': old_avg_price,
                        'exit_price': fill_price,
                        'quantity': old_quantity,
                        'realized_pnl': realized_pnl
                    }
                    self.trade_history.append(trade)

            # Update position
            self.positions[symbol] = {
                'quantity': new_quantity,
                'avg_price': new_avg_price,
                'current_price': fill_price,
                'current_value': new_quantity * fill_price,
                'unrealized_pnl': (fill_price - new_avg_price) * new_quantity,
                'open_time': current_position['open_time'] if new_quantity != 0 else None
            }

            # Remove position if quantity is zero
            if new_quantity == 0:
                del self.positions[symbol]

        return result

    def execute_limit_order(self, symbol: str, quantity: float, price: float,
                            timestamp: datetime = None,
                            expiry_seconds: int = None) -> Dict[str, Any]:
        """Execute a limit order - simplified to use market order logic"""
        # For bare minimum, just forward to market order function
        return self.execute_market_order(symbol, quantity, timestamp)

    def execute_action(self, symbol: str, action: float, current_price: float = None, timestamp: datetime = None) -> \
    Dict[str, Any]:
        """Execute a normalized action from an RL agent - minimal implementation"""
        action_time = timestamp or self.current_timestamp

        # Convert action (-1 to 1) to quantity
        max_position = 100  # Default max position

        # Get current position
        current_position = 0
        if symbol in self.positions:
            current_position = self.positions[symbol]['quantity']

        # Calculate target position
        target_position = int(action * max_position)

        # Calculate quantity change
        quantity_change = target_position - current_position

        if quantity_change == 0:
            return {
                'success': True,
                'action': 'hold',
                'symbol': symbol,
                'current_position': current_position,
                'target_position': target_position,
                'position_change': 0,
                'timestamp': action_time
            }

        # Execute market order for the change
        result = self.execute_market_order(symbol, quantity_change, action_time)

        # Enhance result with additional info
        if result['success']:
            result.update({
                'action': 'buy' if quantity_change > 0 else 'sell',
                'symbol': symbol,
                'current_position': current_position,
                'target_position': target_position,
                'position_change': quantity_change
            })

        return result

    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        total_value = self.cash
        for symbol, position in self.positions.items():
            total_value += position.get('current_value', 0)
        return total_value

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return {
            'cash': self.cash,
            'positions': self.positions,
            'total_value': self.get_portfolio_value(),
            'timestamp': self.current_timestamp
        }

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get information about a specific position"""
        return self.positions.get(symbol, None)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get all completed trades"""
        return self.trade_history

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        # Minimal statistics
        win_count = 0
        loss_count = 0
        total_pnl = 0.0

        for trade in self.trade_history:
            pnl = trade.get('realized_pnl', 0)
            total_pnl += pnl

            if pnl > 0:
                win_count += 1
            elif pnl < 0:
                loss_count += 1

        total_trades = len(self.trade_history)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
        }

    def get_portfolio_history(self) -> Dict[datetime, float]:
        """Get portfolio value history"""
        return self.portfolio_value_history