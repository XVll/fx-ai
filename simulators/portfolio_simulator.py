# portfolio_simulator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime


class PortfolioSimulator:
    """
    Simulator for portfolio management, tracking positions and P&L.
    """

    def __init__(self, config=None, logger=None):
        """
        Initialize the portfolio simulator.

        Args:
            config: Configuration dictionary
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Portfolio state
        self.initial_cash = self.config.get('initial_cash', 100000.0)
        self.cash = self.initial_cash
        self.position = 0.0
        self.position_value = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees = 0.0

        # Position limits
        self.max_position = self.config.get('max_position', 1.0)

        # Trade history
        self.trade_history = []
        self.current_trade = None

    def update_portfolio(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update portfolio based on an execution result.

        Args:
            execution_result: Execution result from ExecutionSimulator

        Returns:
            Updated portfolio state
        """
        # Skip if the execution was not successful
        if execution_result['status'] != 'executed':
            return self.get_portfolio_state()

        # Get execution details
        executed_price = execution_result['executed_price']
        executed_size = execution_result['executed_size']  # Signed size
        commission = execution_result['commission']
        timestamp = execution_result['timestamp']
        execution_timestamp = execution_result['execution_timestamp']

        # Update position and cash
        old_position = self.position
        old_position_value = self.position_value

        # Update position
        self.position += executed_size

        # Update cash (subtract for buys, add for sells)
        transaction_value = executed_price * abs(executed_size)
        self.cash -= (executed_size * executed_price + commission)

        # Update average entry price for buys/sells
        if self.position != 0:
            if old_position == 0:
                # New position
                self.avg_entry_price = executed_price
            elif old_position * self.position > 0:
                # Adding to position
                self.avg_entry_price = ((old_position * self.avg_entry_price) +
                                        (executed_size * executed_price)) / self.position
            elif abs(self.position) < abs(old_position):
                # Reducing position, keep same avg entry
                pass
            else:
                # Flipped position
                self.avg_entry_price = executed_price

        # Update position value and unrealized P&L
        current_price = executed_price  # Use execution price as the current price
        self.position_value = self.position * current_price
        self.unrealized_pnl = self.position_value - (self.position * self.avg_entry_price)

        # Calculate realized P&L for this trade
        if executed_size * old_position < 0:  # If reducing/closing position
            # How much of the position was closed
            closed_size = min(abs(old_position), abs(executed_size))
            if old_position > 0:  # Was long, now selling
                realized_pnl = closed_size * (executed_price - self.avg_entry_price)
            else:  # Was short, now buying
                realized_pnl = closed_size * (self.avg_entry_price - executed_price)

            self.realized_pnl += realized_pnl - commission
        else:
            # Just opening or adding to position, no realized PnL yet
            self.realized_pnl -= commission

        # Add fees to total
        self.total_fees += commission

        # Update trade history
        self._update_trade_history(
            executed_price=executed_price,
            executed_size=executed_size,
            timestamp=timestamp,
            execution_timestamp=execution_timestamp,
            commission=commission
        )

        return self.get_portfolio_state()

    def _update_trade_history(self, executed_price, executed_size, timestamp,
                              execution_timestamp, commission):
        """
        Update trade history with a new execution.
        """
        # Same implementation as before...

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.
        """
        return {
            'cash': self.cash,
            'position': self.position,
            'position_value': self.position_value,
            'avg_entry_price': self.avg_entry_price,
            'total_value': self.cash + self.position_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'total_fees': self.total_fees,
            'max_position': self.max_position
        }

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get the trade history."""
        return list(self.trade_history)

    def reset(self):
        """Reset the portfolio simulator."""
        self.cash = self.initial_cash
        self.position = 0.0
        self.position_value = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees = 0.0
        self.trade_history.clear()
        self.current_trade = None