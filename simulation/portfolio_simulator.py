# simulation/portfolio_simulator.py
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from simulation.market_simulator import MarketSimulator
from simulation.execution_simulator import ExecutionSimulator


class PortfolioSimulator:
    """
    Simulates portfolio state with:
    - Position tracking
    - Cash management
    - P&L calculation
    - Risk metrics (drawdown, Sharpe, etc.)
    - Trade statistics
    """

    def __init__(self, market_simulator: MarketSimulator = None,
                 execution_simulator: ExecutionSimulator = None,
                 config: Dict = None, logger: logging.Logger = None):
        """
        Initialize the portfolio simulator.

        Args:
            market_simulator: MarketSimulator instance
            execution_simulator: ExecutionSimulator instance
            config: Configuration dictionary with portfolio parameters
            logger: Optional logger
        """
        self.market_simulator = market_simulator
        self.execution_simulator = execution_simulator
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Default configuration
        self.initial_cash = self.config.get('initial_cash', 100000.0)  # Starting cash
        self.max_position_pct = self.config.get('max_position_pct', 1.0)  # Max position size as % of portfolio
        self.max_drawdown_pct = self.config.get('max_drawdown_pct', 0.05)  # Max drawdown before stopping (5%)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.0)  # Risk-free rate for Sharpe calculation
        self.enforce_pattern_day_trader = self.config.get('enforce_pattern_day_trader', False)  # PDT rule enforcement
        self.position_size_limits = self.config.get('position_size_limits', {})  # Symbol-specific position limits
        self.enable_shorting = self.config.get('enable_shorting', False)  # Allow short positions
        self.track_commissions = self.config.get('track_commissions', True)  # Track commission costs

        # Portfolio state
        self.cash = self.initial_cash
        self.positions = {}  # Current positions {symbol: {quantity, avg_price, ...}}
        self.portfolio_value_history = {}  # Historical portfolio values
        self.position_history = {}  # Historical positions
        self.max_portfolio_value = self.initial_cash  # For drawdown calculation
        self.trade_history = []  # Completed trades

        # Current timestamp
        self.current_timestamp = None

        # Trade statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'total_pnl': 0.0,
            'total_commission': 0.0,
            'max_profit_trade': 0.0,
            'max_loss_trade': 0.0,
            'avg_profit_per_winning_trade': 0.0,
            'avg_loss_per_losing_trade': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0
        }

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

    def set_execution_simulator(self, execution_simulator: ExecutionSimulator) -> None:
        """
        Set the execution simulator instance.

        Args:
            execution_simulator: ExecutionSimulator instance
        """
        self.execution_simulator = execution_simulator

    def reset(self) -> None:
        """Reset the portfolio simulator to initial state."""
        self.cash = self.initial_cash
        self.positions = {}
        self.portfolio_value_history = {}
        self.position_history = {}
        self.max_portfolio_value = self.initial_cash
        self.trade_history = []
        self.current_timestamp = None

        # Reset statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'total_pnl': 0.0,
            'total_commission': 0.0,
            'max_profit_trade': 0.0,
            'max_loss_trade': 0.0,
            'avg_profit_per_winning_trade': 0.0,
            'avg_loss_per_losing_trade': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0
        }

    def update_to_timestamp(self, timestamp: datetime) -> None:
        """
        Update portfolio state to a specific timestamp.

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
                self._log(f"Cannot move portfolio backwards from {self.current_timestamp} to {timestamp}",
                          logging.WARNING)
                return

        # Update market and execution simulators if available
        if self.market_simulator:
            self.market_simulator.update_to_timestamp(timestamp)

        if self.execution_simulator:
            self.execution_simulator.update_to_timestamp(timestamp)

        # First update to current price
        self._update_position_values(timestamp)

        # Then record the portfolio state
        self._record_portfolio_state(timestamp)

        # Update the current timestamp
        self.current_timestamp = timestamp

    def _update_position_values(self, timestamp: datetime) -> None:
        """
        Update the values of all positions based on current market prices.

        Args:
            timestamp: Current timestamp
        """
        if not self.market_simulator:
            return  # Can't update without price information

        for symbol, position in self.positions.items():
            # Update market simulator to get current price
            self.market_simulator.update_to_timestamp(timestamp)

            # Get current price
            current_price = self.market_simulator.current_price

            if current_price is None:
                continue  # Skip if no price available

            # Update position value and unrealized P&L
            quantity = position['quantity']
            avg_price = position['avg_price']

            position['current_price'] = current_price
            position['current_value'] = quantity * current_price
            position['unrealized_pnl'] = (current_price - avg_price) * quantity
            position['unrealized_pnl_pct'] = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
            position['last_update_time'] = timestamp

    def _record_portfolio_state(self, timestamp: datetime) -> None:
        """
        Record the current portfolio state for historical tracking.

        Args:
            timestamp: Current timestamp
        """
        # Calculate total portfolio value
        total_value = self.cash

        for symbol, position in self.positions.items():
            total_value += position.get('current_value', 0)

        # Record to history
        self.portfolio_value_history[timestamp] = total_value

        # Record position snapshot
        position_snapshot = {}
        for symbol, position in self.positions.items():
            position_snapshot[symbol] = position.copy()

        self.position_history[timestamp] = position_snapshot

        # Update max portfolio value for drawdown calculation
        if total_value > self.max_portfolio_value:
            self.max_portfolio_value = total_value

        # Calculate current drawdown
        current_drawdown = self.max_portfolio_value - total_value
        current_drawdown_pct = current_drawdown / self.max_portfolio_value if self.max_portfolio_value > 0 else 0

        # Update max drawdown if needed
        if current_drawdown_pct > self.stats['max_drawdown_pct']:
            self.stats['max_drawdown'] = current_drawdown
            self.stats['max_drawdown_pct'] = current_drawdown_pct

    def execute_market_order(self, symbol: str, quantity: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Execute a market order and update portfolio.

        Args:
            symbol: Instrument symbol
            quantity: Order quantity (positive for buy, negative for sell)
            timestamp: Order time (defaults to current timestamp)

        Returns:
            Dictionary with order result
        """
        if self.execution_simulator is None:
            self._log("Cannot execute order: no execution simulator set", logging.ERROR)
            return {'success': False, 'reason': 'no_execution_simulator'}

        if self.market_simulator is None:
            self._log("Cannot execute order: no market simulator set", logging.ERROR)
            return {'success': False, 'reason': 'no_market_simulator'}

        order_time = timestamp or self.current_timestamp

        if order_time is None:
            self._log("Cannot execute order: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'reason': 'no_timestamp'}

        # Validate quantity
        if quantity == 0:
            return {'success': False, 'reason': 'zero_quantity'}

        # Determine order side
        side = 'buy' if quantity > 0 else 'sell'
        abs_quantity = abs(quantity)

        # For sells, validate we have enough shares
        if side == 'sell':
            if symbol not in self.positions or self.positions[symbol]['quantity'] < abs_quantity:
                self._log(f"Cannot sell {abs_quantity} shares of {symbol}: insufficient position", logging.WARNING)
                return {'success': False, 'reason': 'insufficient_position'}

        # For buys, check if we would exceed position limits
        if side == 'buy':
            # Check position size limit for this symbol
            max_symbol_position = self.position_size_limits.get(symbol, float('inf'))
            if symbol in self.positions:
                new_quantity = self.positions[symbol]['quantity'] + abs_quantity
                if new_quantity > max_symbol_position:
                    self._log(f"Buy would exceed position limit for {symbol}", logging.WARNING)
                    return {'success': False, 'reason': 'position_limit_exceeded'}

            # Check if we would exceed max position as % of portfolio
            # First, estimate the cost
            estimated_price = self.market_simulator.current_price
            if estimated_price is None:
                self._log(f"Cannot estimate order cost: no price available for {symbol}", logging.WARNING)
                return {'success': False, 'reason': 'no_price_available'}

            estimated_cost = estimated_price * abs_quantity
            current_portfolio_value = self.cash
            for sym, pos in self.positions.items():
                current_portfolio_value += pos.get('current_value', 0)

            max_position_value = current_portfolio_value * self.max_position_pct

            if symbol in self.positions:
                new_position_value = self.positions[symbol]['current_value'] + estimated_cost
                if new_position_value > max_position_value:
                    self._log(f"Buy would exceed max position % for {symbol}", logging.WARNING)
                    return {'success': False, 'reason': 'max_position_pct_exceeded'}
            elif estimated_cost > max_position_value:
                self._log(f"Buy would exceed max position % for {symbol}", logging.WARNING)
                return {'success': False, 'reason': 'max_position_pct_exceeded'}

            # Check if we have enough cash
            if estimated_cost > self.cash:
                self._log(f"Insufficient cash for order: ${estimated_cost:.2f} needed, ${self.cash:.2f} available",
                          logging.WARNING)
                return {'success': False, 'reason': 'insufficient_cash'}

        # Execute the order through execution simulator
        order_result = self.execution_simulator.place_market_order(
            side=side,
            quantity=abs_quantity,
            timestamp=order_time
        )

        if order_result['success']:
            # Order was filled, update portfolio
            return self._process_filled_order(symbol, order_result, order_time)
        else:
            # Order was rejected
            self._log(f"Order rejected: {order_result.get('reason', 'unknown')}", logging.WARNING)
            return order_result

    def execute_limit_order(self, symbol: str, quantity: float, price: float,
                            timestamp: datetime = None,
                            expiry_seconds: int = None) -> Dict[str, Any]:
        """
        Execute a limit order and update portfolio.

        Args:
            symbol: Instrument symbol
            quantity: Order quantity (positive for buy, negative for sell)
            price: Limit price
            timestamp: Order time (defaults to current timestamp)
            expiry_seconds: Order expiration in seconds

        Returns:
            Dictionary with order result
        """
        if self.execution_simulator is None:
            self._log("Cannot execute order: no execution simulator set", logging.ERROR)
            return {'success': False, 'reason': 'no_execution_simulator'}

        order_time = timestamp or self.current_timestamp

        if order_time is None:
            self._log("Cannot execute order: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'reason': 'no_timestamp'}

        # Validate quantity
        if quantity == 0:
            return {'success': False, 'reason': 'zero_quantity'}

        # Determine order side
        side = 'buy' if quantity > 0 else 'sell'
        abs_quantity = abs(quantity)

        # For sells, validate we have enough shares
        if side == 'sell':
            if symbol not in self.positions or self.positions[symbol]['quantity'] < abs_quantity:
                self._log(f"Cannot sell {abs_quantity} shares of {symbol}: insufficient position", logging.WARNING)
                return {'success': False, 'reason': 'insufficient_position'}

        # For buys, check if we would exceed position limits and if we have enough cash
        if side == 'buy':
            # Check position size limit for this symbol
            max_symbol_position = self.position_size_limits.get(symbol, float('inf'))
            if symbol in self.positions:
                new_quantity = self.positions[symbol]['quantity'] + abs_quantity
                if new_quantity > max_symbol_position:
                    self._log(f"Buy would exceed position limit for {symbol}", logging.WARNING)
                    return {'success': False, 'reason': 'position_limit_exceeded'}

            # Check if we would exceed max position as % of portfolio
            estimated_cost = price * abs_quantity
            current_portfolio_value = self.cash
            for sym, pos in self.positions.items():
                current_portfolio_value += pos.get('current_value', 0)

            max_position_value = current_portfolio_value * self.max_position_pct

            if symbol in self.positions:
                current_position_value = self.positions[symbol]['current_value']
                new_position_value = current_position_value + estimated_cost
                if new_position_value > max_position_value:
                    self._log(f"Buy would exceed max position % for {symbol}", logging.WARNING)
                    return {'success': False, 'reason': 'max_position_pct_exceeded'}
            elif estimated_cost > max_position_value:
                self._log(f"Buy would exceed max position % for {symbol}", logging.WARNING)
                return {'success': False, 'reason': 'max_position_pct_exceeded'}

            # Check if we have enough cash
            if estimated_cost > self.cash:
                self._log(f"Insufficient cash for order: ${estimated_cost:.2f} needed, ${self.cash:.2f} available",
                          logging.WARNING)
                return {'success': False, 'reason': 'insufficient_cash'}

        # Execute the order through execution simulator
        order_result = self.execution_simulator.place_limit_order(
            side=side,
            quantity=abs_quantity,
            price=price,
            timestamp=order_time,
            expiry_seconds=expiry_seconds
        )

        if order_result['success']:
            # Order was filled, update portfolio
            return self._process_filled_order(symbol, order_result, order_time)
        else:
            # Order was rejected or expired
            reason = order_result.get('reason', 'unknown')
            self._log(f"Limit order not filled: {reason}", logging.WARNING)
            return order_result

    def _process_filled_order(self, symbol: str, order_result: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """
        Process a filled order and update portfolio state.

        Args:
            symbol: Instrument symbol
            order_result: Order execution result from execution simulator
            timestamp: Order execution time

        Returns:
            Updated order result dictionary
        """
        # Extract order details
        filled_quantity = order_result['filled_quantity']
        fill_price = order_result['fill_price']
        is_buy = order_result.get('status') == 'filled' and 'side' in order_result and order_result['side'] == 'buy'

        # Adjust quantity sign based on side
        signed_quantity = filled_quantity if is_buy else -filled_quantity

        # Calculate order cost
        order_cost = filled_quantity * fill_price

        # Extract fees if available
        fees = order_result.get('fees', {})
        total_fee = fees.get('total_fee', 0.0)

        # Update cash balance
        if is_buy:
            # Buying: reduce cash by order cost plus fees
            self.cash -= (order_cost + total_fee)
        else:
            # Selling: increase cash by order cost minus fees
            self.cash += (order_cost - total_fee)

        # Update position
        realized_pnl = 0.0

        if symbol not in self.positions:
            # New position
            if is_buy:
                self.positions[symbol] = {
                    'quantity': filled_quantity,
                    'avg_price': fill_price,
                    'cost_basis': order_cost + total_fee,
                    'open_time': timestamp,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'current_price': fill_price,
                    'current_value': order_cost,
                    'realized_pnl': 0.0,
                    'last_update_time': timestamp
                }
            else:
                # Should not happen (selling without position), but handle anyway
                self._log(f"Selling {filled_quantity} shares of {symbol} without position", logging.WARNING)
                self.positions[symbol] = {
                    'quantity': -filled_quantity,
                    'avg_price': fill_price,
                    'cost_basis': -order_cost - total_fee,
                    'open_time': timestamp,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'current_price': fill_price,
                    'current_value': -order_cost,
                    'realized_pnl': 0.0,
                    'last_update_time': timestamp
                }
        else:
            # Existing position
            current_position = self.positions[symbol]
            current_quantity = current_position['quantity']
            current_avg_price = current_position['avg_price']
            current_cost_basis = current_position['cost_basis']

            new_quantity = current_quantity + signed_quantity

            if (current_quantity > 0 and signed_quantity < 0) or (current_quantity < 0 and signed_quantity > 0):
                # Reducing position - calculate realized P&L
                # Only count P&L for the reduced portion
                reduction_quantity = min(abs(current_quantity), abs(signed_quantity))
                realized_pnl = ((fill_price - current_avg_price) * reduction_quantity) if current_quantity > 0 else \
                    ((current_avg_price - fill_price) * reduction_quantity)

                # If completely closing the position or flipping sides
                if abs(signed_quantity) >= abs(current_quantity) or (new_quantity > 0) != (current_quantity > 0):
                    # Close out current position or flip
                    if new_quantity == 0:
                        # Fully closed position
                        closed_position = current_position.copy()
                        closed_position['close_time'] = timestamp
                        closed_position['close_price'] = fill_price
                        closed_position['realized_pnl'] += realized_pnl
                        closed_position['pnl_pct'] = (realized_pnl / abs(
                            current_cost_basis)) * 100 if current_cost_basis != 0 else 0

                        # Record trade
                        self._record_completed_trade(symbol, closed_position)

                        # Remove position
                        del self.positions[symbol]
                    else:
                        # Position flipped from long to short or vice versa
                        # First, close out the original position
                        closed_position = current_position.copy()
                        closed_position['close_time'] = timestamp
                        closed_position['close_price'] = fill_price
                        closed_position['realized_pnl'] += realized_pnl
                        closed_position['pnl_pct'] = (realized_pnl / abs(
                            current_cost_basis)) * 100 if current_cost_basis != 0 else 0

                        # Record trade
                        self._record_completed_trade(symbol, closed_position)

                        # Then create a new position for the remaining quantity
                        remaining_quantity = abs(new_quantity)
                        self.positions[symbol] = {
                            'quantity': new_quantity,
                            'avg_price': fill_price,
                            'cost_basis': remaining_quantity * fill_price + total_fee,
                            'open_time': timestamp,
                            'unrealized_pnl': 0.0,
                            'unrealized_pnl_pct': 0.0,
                            'current_price': fill_price,
                            'current_value': remaining_quantity * fill_price,
                            'realized_pnl': 0.0,
                            'last_update_time': timestamp
                        }
                else:
                    # Just reducing the position, not closing it
                    # Update the cost basis and average price, tracking realized P&L
                    new_cost_basis = current_cost_basis - (abs(signed_quantity) * current_avg_price)
                    self.positions[symbol]['quantity'] = new_quantity
                    self.positions[symbol]['avg_price'] = current_avg_price  # Avg price doesn't change when reducing
                    self.positions[symbol]['cost_basis'] = new_cost_basis
                    self.positions[symbol]['realized_pnl'] += realized_pnl
                    self.positions[symbol]['current_price'] = fill_price
                    self.positions[symbol]['current_value'] = new_quantity * fill_price
                    self.positions[symbol]['last_update_time'] = timestamp
            else:
                # Increasing position - update avg price and cost basis
                if new_quantity != 0:  # Safeguard against division by zero
                    # For increasing long position or increasing short position
                    new_cost_basis = current_cost_basis + (abs(signed_quantity) * fill_price) + total_fee
                    new_avg_price = new_cost_basis / abs(new_quantity)

                    self.positions[symbol]['quantity'] = new_quantity
                    self.positions[symbol]['avg_price'] = new_avg_price
                    self.positions[symbol]['cost_basis'] = new_cost_basis
                    self.positions[symbol]['current_price'] = fill_price
                    self.positions[symbol]['current_value'] = new_quantity * fill_price
                    self.positions[symbol]['last_update_time'] = timestamp

        # Force update of all positions' current values
        self._update_position_values(timestamp)

        # Record the updated portfolio state
        self._record_portfolio_state(timestamp)

        # Enhance the order result with portfolio information
        enhanced_result = order_result.copy()
        enhanced_result['realized_pnl'] = realized_pnl
        enhanced_result['cash_balance'] = self.cash
        enhanced_result['position'] = self.positions.get(symbol, {}).copy()

        return enhanced_result

    def _record_completed_trade(self, symbol: str, position: Dict[str, Any]) -> None:
        """
        Record a completed trade in trade history and update statistics.

        Args:
            symbol: Instrument symbol
            position: Position dictionary with trade details
        """
        # Create trade record
        trade = {
            'symbol': symbol,
            'quantity': position['quantity'],
            'side': 'long' if position['quantity'] > 0 else 'short',
            'open_time': position['open_time'],
            'close_time': position['close_time'],
            'duration': (position['close_time'] - position['open_time']).total_seconds(),
            'entry_price': position['avg_price'],
            'exit_price': position['close_price'],
            'realized_pnl': position['realized_pnl'],
            'pnl_pct': position['pnl_pct'],
            'cost_basis': position['cost_basis']
        }

        # Add to trade history
        self.trade_history.append(trade)

        # Update statistics
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += trade['realized_pnl']

        # Categorize trade
        pnl = trade['realized_pnl']
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['max_profit_trade'] = max(self.stats['max_profit_trade'], pnl)
        elif pnl < 0:
            self.stats['losing_trades'] += 1
            self.stats['max_loss_trade'] = min(self.stats['max_loss_trade'], pnl)
        else:
            self.stats['breakeven_trades'] += 1

        # Recalculate derived statistics
        self._update_trade_statistics()

        self._log(
            f"Completed trade for {symbol}: {'PROFIT' if pnl > 0 else 'LOSS'} ${pnl:.2f} ({trade['pnl_pct']:.2f}%)")

    def _update_trade_statistics(self) -> None:
        """Update derived trade statistics."""
        winning_trades = self.stats['winning_trades']
        losing_trades = self.stats['losing_trades']
        total_trades = self.stats['total_trades']

        # Win rate
        self.stats['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0

        # Average profit/loss per trade
        profitable_trades = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] > 0]
        losing_trades = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] < 0]

        self.stats['avg_profit_per_winning_trade'] = np.mean(profitable_trades) if profitable_trades else 0
        self.stats['avg_loss_per_losing_trade'] = np.mean(losing_trades) if losing_trades else 0

        # Profit factor
        total_profits = sum(profitable_trades) if profitable_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0

        self.stats['profit_factor'] = total_profits / total_losses if total_losses > 0 else float('inf')

        # Sharpe ratio (if we have enough data)
        if len(self.portfolio_value_history) > 1:
            # Convert portfolio history to returns
            values = pd.Series(self.portfolio_value_history)
            returns = values.pct_change().dropna()

            if len(returns) > 0:
                # Annualize based on avg time between samples
                avg_time_delta = pd.Series(values.index).diff().mean().total_seconds()
                annualization_factor = (252 * 6.5 * 60 * 60) / avg_time_delta  # Assuming 252 trading days, 6.5 hours

                mean_return = returns.mean()
                std_return = returns.std()

                if std_return > 0:
                    self.stats['sharpe_ratio'] = ((mean_return - self.risk_free_rate) / std_return) * np.sqrt(
                        annualization_factor)

    def reset_simulator(self) -> None:
        """Reset the simulator to initial state."""
        self.reset()

    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.

        Returns:
            Total portfolio value (cash + positions)
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            total_value += position.get('current_value', 0)

        return total_value

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.

        Returns:
            Dictionary with portfolio state information
        """
        return {
            'cash': self.cash,
            'positions': self.positions,
            'total_value': self.get_portfolio_value(),
            'max_drawdown': self.stats['max_drawdown'],
            'max_drawdown_pct': self.stats['max_drawdown_pct'],
            'timestamp': self.current_timestamp
        }

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a specific position.

        Args:
            symbol: Instrument symbol

        Returns:
            Position dictionary, or None if not found
        """
        return self.positions.get(symbol, None)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get all completed trades.

        Returns:
            List of trade dictionaries
        """
        return self.trade_history

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        return self.stats

    def get_portfolio_history(self) -> pd.Series:
        """
        Get portfolio value history as a pandas Series.

        Returns:
            Time series of portfolio values
        """
        return pd.Series(self.portfolio_value_history)

    def execute_action(self, symbol: str, action: float, current_price: float = None, timestamp: datetime = None) -> \
    Dict[str, Any]:
        """
        Execute a normalized action from an RL agent.

        Args:
            symbol: Instrument symbol
            action: Normalized action (-1.0 to 1.0 where -1 = full sell, 0 = hold, 1 = full buy)
            current_price: Current price (for simulation without market data)
            timestamp: Action time (defaults to current timestamp)

        Returns:
            Dictionary with action result
        """
        if action < -1.0 or action > 1.0:
            self._log(f"Invalid action value: {action}. Must be between -1.0 and 1.0", logging.WARNING)
            return {'success': False, 'reason': 'invalid_action_value'}

        action_time = timestamp or self.current_timestamp

        if action_time is None:
            self._log("Cannot execute action: no timestamp provided and no current timestamp set", logging.ERROR)
            return {'success': False, 'reason': 'no_timestamp'}

        # Get current price if not provided
        if current_price is None:
            if self.market_simulator:
                self.market_simulator.update_to_timestamp(action_time)
                current_price = self.market_simulator.current_price

            if current_price is None:
                self._log("Cannot execute action: no price available", logging.ERROR)
                return {'success': False, 'reason': 'no_price_available'}

        # Get current position
        current_position = 0.0
        max_position = self.position_size_limits.get(symbol, 100)  # Default max position

        if symbol in self.positions:
            current_position = self.positions[symbol]['quantity']

        # Convert action to target position
        target_position = round(action * max_position)

        # Calculate change in position
        position_change = target_position - current_position

        if position_change == 0:
            # No change in position
            return {
                'success': True,
                'action': 'hold',
                'symbol': symbol,
                'current_position': current_position,
                'target_position': target_position,
                'position_change': 0,
                'timestamp': action_time
            }

        # Execute market order for the position change
        if position_change > 0:
            # Buy
            result = self.execute_market_order(symbol, position_change, action_time)
            if result['success']:
                return {
                    'success': True,
                    'action': 'buy',
                    'symbol': symbol,
                    'current_position': current_position,
                    'target_position': target_position,
                    'position_change': position_change,
                    'fill_price': result['fill_price'],
                    'filled_quantity': result['filled_quantity'],
                    'timestamp': result.get('timestamp', action_time),
                    'realized_pnl': result.get('realized_pnl', 0.0),
                    'order_details': result
                }
            else:
                return {
                    'success': False,
                    'action': 'buy',
                    'symbol': symbol,
                    'current_position': current_position,
                    'target_position': target_position,
                    'position_change': position_change,
                    'reason': result.get('reason', 'unknown'),
                    'timestamp': action_time,
                    'order_details': result
                }
        else:  # position_change < 0
            # Sell
            result = self.execute_market_order(symbol, position_change, action_time)
            if result['success']:
                return {
                    'success': True,
                    'action': 'sell',
                    'symbol': symbol,
                    'current_position': current_position,
                    'target_position': target_position,
                    'position_change': position_change,
                    'fill_price': result['fill_price'],
                    'filled_quantity': abs(result['filled_quantity']),
                    'timestamp': result.get('timestamp', action_time),
                    'realized_pnl': result.get('realized_pnl', 0.0),
                    'order_details': result
                }
            else:
                return {
                    'success': False,
                    'action': 'sell',
                    'symbol': symbol,
                    'current_position': current_position,
                    'target_position': target_position,
                    'position_change': position_change,
                    'reason': result.get('reason', 'unknown'),
                    'timestamp': action_time,
                    'order_details': result
                }