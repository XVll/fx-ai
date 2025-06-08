"""
Portfolio Simulator Implementation Schema

This module provides the concrete implementation of the PortfolioSimulator interface
for managing portfolio state, positions, P&L, and risk metrics.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from v2.core.interfaces import (
    PortfolioSimulator, Position, Trade, PortfolioState,
    RiskMetrics, PortfolioConfig
)


class PortfolioSimulatorImpl(PortfolioSimulator):
    """
    Concrete implementation of portfolio simulator.
    
    Manages complete portfolio state including:
    - Position tracking and P&L
    - Cash management
    - Risk calculations
    - Trade history
    - Performance metrics
    - Margin and buying power
    
    Features:
    - Multi-asset position management
    - Real-time P&L calculation
    - Risk limit enforcement
    - Transaction cost tracking
    - Performance attribution
    """
    
    def __init__(
        self,
        config: PortfolioConfig,
        initial_capital: float = 100000.0,
        leverage_limit: float = 1.0
    ):
        """
        Initialize the portfolio simulator.
        
        Args:
            config: Portfolio configuration
            initial_capital: Starting capital
            leverage_limit: Maximum leverage allowed
        """
        self.config = config
        self.initial_capital = initial_capital
        self.leverage_limit = leverage_limit
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        self.current_leverage = 0.0
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.trade_returns: List[float] = []
        
        # Transaction costs
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        
        # TODO: Initialize risk models and limits
        
    def update_portfolio(
        self,
        timestamp: datetime,
        market_prices: Dict[str, float],
        fills: Optional[List[Any]] = None
    ) -> PortfolioState:
        """
        Update portfolio with new market prices and fills.
        
        Implementation:
        1. Process any new fills
        2. Update position marks
        3. Calculate P&L
        4. Update risk metrics
        5. Check risk limits
        6. Return current state
        
        Args:
            timestamp: Current timestamp
            market_prices: Current market prices
            fills: Optional list of new fills
            
        Returns:
            Updated portfolio state
        """
        # Process fills if any
        if fills:
            for fill in fills:
                self._process_fill(fill, timestamp)
        
        # Update position values
        self._mark_positions(market_prices, timestamp)
        
        # Calculate P&L
        self._calculate_pnl()
        
        # Update risk metrics
        self._update_risk_metrics(timestamp)
        
        # Record equity curve point
        total_value = self.get_total_value()
        self.equity_curve.append((timestamp, total_value))
        
        # TODO: Check risk limits and generate alerts
        
        # Create portfolio state
        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=dict(self.positions),  # Copy positions
            total_value=total_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            leverage=self.current_leverage,
            risk_metrics=self.get_risk_metrics()
        )
        
        return state
    
    def execute_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        timestamp: datetime,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Trade:
        """
        Execute a trade and update portfolio.
        
        Implementation:
        1. Validate trade parameters
        2. Check buying power
        3. Update or create position
        4. Deduct cash and costs
        5. Record trade
        6. Update metrics
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            side: Buy or sell
            timestamp: Trade timestamp
            commission: Commission cost
            slippage: Slippage cost
            
        Returns:
            Executed trade object
        """
        # Validate trade
        if quantity <= 0:
            raise ValueError("Invalid quantity")
        
        # Calculate trade value
        trade_value = quantity * price
        total_cost = trade_value + commission + slippage
        
        # Check buying power for buys
        if side == "buy":
            if total_cost > self.get_buying_power():
                raise ValueError("Insufficient buying power")
        
        # TODO: Implement trade execution
        # 1. Update position
        if symbol in self.positions:
            position = self.positions[symbol]
            self._update_position(position, quantity, price, side, timestamp)
        else:
            # Create new position
            position = self._create_position(symbol, quantity, price, side, timestamp)
            self.positions[symbol] = position
        
        # 2. Update cash
        if side == "buy":
            self.cash -= total_cost
        else:
            self.cash += trade_value - commission - slippage
        
        # 3. Track costs
        self.total_commissions += commission
        self.total_slippage += slippage
        
        # 4. Create trade record
        trade = Trade(
            trade_id=f"{symbol}_{timestamp.timestamp()}",
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage,
            value=trade_value
        )
        
        self.trades.append(trade)
        
        return trade
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Implementation:
        1. Look up position
        2. Return copy to prevent modification
        
        Args:
            symbol: Symbol to look up
            
        Returns:
            Position object or None
        """
        if symbol in self.positions:
            # Return a copy to prevent external modification
            pos = self.positions[symbol]
            return Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                average_price=pos.average_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                realized_pnl=pos.realized_pnl,
                opened_at=pos.opened_at,
                last_updated=pos.last_updated
            )
        return None
    
    def get_total_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Implementation:
        1. Sum cash
        2. Add position values
        3. Account for unrealized P&L
        
        Returns:
            Total portfolio value
        """
        total = self.cash
        
        for position in self.positions.values():
            # Position value = quantity * current_price
            position_value = abs(position.quantity) * position.current_price
            
            # Add unrealized P&L
            total += position_value + position.unrealized_pnl
        
        return total
    
    def get_buying_power(self) -> float:
        """
        Calculate available buying power.
        
        Implementation:
        1. Start with cash
        2. Apply leverage if allowed
        3. Subtract margin requirements
        4. Apply buffer
        
        Returns:
            Available buying power
        """
        # Basic buying power is cash
        buying_power = self.cash
        
        # Apply leverage if configured
        if self.leverage_limit > 1.0:
            # Calculate current leverage usage
            position_value = sum(
                abs(p.quantity) * p.current_price 
                for p in self.positions.values()
            )
            total_value = self.get_total_value()
            
            # Available leverage
            max_position_value = total_value * self.leverage_limit
            available_leverage = max_position_value - position_value
            
            buying_power += available_leverage
        
        # Apply safety buffer (5%)
        buying_power *= 0.95
        
        return max(0, buying_power)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current risk metrics.
        
        Implementation:
        1. Calculate position concentration
        2. Compute portfolio volatility
        3. Calculate max drawdown
        4. Measure leverage
        5. Compute Sharpe ratio
        
        Returns:
            Current risk metrics
        """
        total_value = self.get_total_value()
        
        # Position concentration
        position_values = {}
        for symbol, pos in self.positions.items():
            position_values[symbol] = abs(pos.quantity) * pos.current_price
        
        max_position_pct = 0.0
        if position_values and total_value > 0:
            max_position_pct = max(position_values.values()) / total_value
        
        # Calculate returns for risk metrics
        returns = self._calculate_returns()
        
        # Portfolio volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)
        else:
            volatility = 0.0
        
        # Sharpe ratio
        if volatility > 0 and len(returns) > 0:
            sharpe = (np.mean(returns) * 252) / volatility
        else:
            sharpe = 0.0
        
        # Current leverage
        position_value = sum(position_values.values())
        leverage = position_value / total_value if total_value > 0 else 0.0
        
        metrics = RiskMetrics(
            total_value=total_value,
            cash=self.cash,
            leverage=leverage,
            max_drawdown=self.max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe,
            position_concentration=max_position_pct,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl
        )
        
        return metrics
    
    def _process_fill(self, fill: Any, timestamp: datetime) -> None:
        """
        Process a fill from execution simulator.
        
        Implementation:
        1. Extract fill details
        2. Execute corresponding trade
        3. Update position
        4. Track execution quality
        
        Args:
            fill: Fill object from executor
            timestamp: Current timestamp
        """
        # TODO: Implement fill processing
        # Extract details from fill object
        # Call execute_trade with fill parameters
        pass
    
    def _mark_positions(
        self,
        market_prices: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """
        Mark positions to market prices.
        
        Implementation:
        1. Update each position's current price
        2. Calculate unrealized P&L
        3. Update position timestamps
        
        Args:
            market_prices: Current prices
            timestamp: Current time
        """
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                old_price = position.current_price
                new_price = market_prices[symbol]
                
                # Update price
                position.current_price = new_price
                position.last_updated = timestamp
                
                # Calculate unrealized P&L
                if position.quantity > 0:
                    # Long position
                    position.unrealized_pnl = (
                        position.quantity * (new_price - position.average_price)
                    )
                else:
                    # Short position
                    position.unrealized_pnl = (
                        -position.quantity * (position.average_price - new_price)
                    )
    
    def _calculate_pnl(self) -> None:
        """
        Calculate total P&L across portfolio.
        
        Implementation:
        1. Sum unrealized P&L from positions
        2. Add to realized P&L
        3. Update total P&L
        """
        # Sum unrealized P&L
        self.unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        
        # Total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
    
    def _update_position(
        self,
        position: Position,
        quantity: int,
        price: float,
        side: str,
        timestamp: datetime
    ) -> None:
        """
        Update existing position with new trade.
        
        Implementation:
        1. Handle position increases
        2. Handle position decreases
        3. Handle position flips
        4. Update average price
        5. Track realized P&L
        
        Args:
            position: Position to update
            quantity: Trade quantity
            price: Trade price
            side: Trade side
            timestamp: Trade time
        """
        old_quantity = position.quantity
        
        if side == "buy":
            new_quantity = old_quantity + quantity
        else:
            new_quantity = old_quantity - quantity
        
        # Position flip (long to short or vice versa)
        if old_quantity > 0 and new_quantity < 0:
            # Was long, now short
            # Realize P&L on closed long portion
            closed_qty = old_quantity
            realized = closed_qty * (price - position.average_price)
            position.realized_pnl += realized
            self.realized_pnl += realized
            
            # New short position
            position.quantity = new_quantity
            position.average_price = price
            
        elif old_quantity < 0 and new_quantity > 0:
            # Was short, now long
            # Realize P&L on closed short portion
            closed_qty = -old_quantity
            realized = closed_qty * (position.average_price - price)
            position.realized_pnl += realized
            self.realized_pnl += realized
            
            # New long position
            position.quantity = new_quantity
            position.average_price = price
            
        elif abs(new_quantity) < abs(old_quantity):
            # Reducing position
            closed_qty = abs(old_quantity) - abs(new_quantity)
            
            if old_quantity > 0:
                # Reducing long
                realized = closed_qty * (price - position.average_price)
            else:
                # Reducing short
                realized = closed_qty * (position.average_price - price)
            
            position.realized_pnl += realized
            self.realized_pnl += realized
            position.quantity = new_quantity
            
        else:
            # Increasing position
            # Update average price
            if old_quantity == 0:
                position.average_price = price
            else:
                total_value = (
                    abs(old_quantity) * position.average_price +
                    quantity * price
                )
                position.average_price = total_value / abs(new_quantity)
            
            position.quantity = new_quantity
        
        position.last_updated = timestamp
    
    def _create_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        timestamp: datetime
    ) -> Position:
        """
        Create new position.
        
        Implementation:
        1. Initialize position object
        2. Set initial values
        3. Adjust quantity for side
        
        Args:
            symbol: Position symbol
            quantity: Initial quantity
            price: Entry price
            side: Buy or sell
            timestamp: Creation time
            
        Returns:
            New position object
        """
        # Adjust quantity sign based on side
        signed_quantity = quantity if side == "buy" else -quantity
        
        position = Position(
            symbol=symbol,
            quantity=signed_quantity,
            average_price=price,
            current_price=price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            opened_at=timestamp,
            last_updated=timestamp
        )
        
        return position
    
    def _update_risk_metrics(self, timestamp: datetime) -> None:
        """
        Update portfolio risk metrics.
        
        Implementation:
        1. Calculate current drawdown
        2. Update peak value
        3. Calculate leverage
        4. Update return series
        
        Args:
            timestamp: Current timestamp
        """
        current_value = self.get_total_value()
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate leverage
        position_value = sum(
            abs(p.quantity) * p.current_price 
            for p in self.positions.values()
        )
        self.current_leverage = position_value / current_value if current_value > 0 else 0
        
        # Update return series if we have history
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
    
    def _calculate_returns(self) -> List[float]:
        """
        Calculate return series for risk metrics.
        
        Implementation:
        1. Use daily returns if available
        2. Fall back to equity curve
        3. Handle edge cases
        
        Returns:
            List of returns
        """
        if self.daily_returns:
            return self.daily_returns
        
        # Calculate from equity curve
        if len(self.equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1][1]
            curr_value = self.equity_curve[i][1]
            
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
        
        return returns
    
    def reset(self) -> None:
        """
        Reset portfolio to initial state.
        
        Implementation:
        1. Clear all positions
        2. Reset cash to initial
        3. Clear P&L
        4. Reset metrics
        5. Clear history
        """
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        self.current_leverage = 0.0
        
        self.equity_curve.clear()
        self.daily_returns.clear()
        self.trade_returns.clear()
        
        self.total_commissions = 0.0
        self.total_slippage = 0.0
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get comprehensive performance summary.
        
        Implementation:
        1. Calculate return metrics
        2. Calculate risk metrics  
        3. Calculate efficiency metrics
        4. Format summary
        
        Returns:
            Performance summary dictionary
        """
        total_value = self.get_total_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate trade statistics
        n_trades = len(self.trades)
        if n_trades > 0:
            winning_trades = sum(1 for t in self.trade_returns if t > 0)
            win_rate = winning_trades / n_trades
            
            avg_win = np.mean([t for t in self.trade_returns if t > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t for t in self.trade_returns if t <= 0]) if winning_trades < n_trades else 0
            
            profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * (n_trades - winning_trades)) if avg_loss != 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        summary = {
            'total_return': total_return,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_commissions': self.total_commissions,
            'total_slippage': self.total_slippage,
            'current_leverage': self.current_leverage
        }
        
        # Add risk metrics if available
        risk_metrics = self.get_risk_metrics()
        summary['sharpe_ratio'] = risk_metrics.sharpe_ratio
        summary['volatility'] = risk_metrics.volatility
        
        return summary