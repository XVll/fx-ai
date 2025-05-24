# metrics/collectors/trading_metrics.py - Trading and portfolio metrics collector

import logging
from typing import Dict, Optional, Any, List
from collections import deque
import numpy as np

from ..core import MetricCollector, MetricValue, MetricCategory, MetricType, MetricMetadata


class PortfolioMetricsCollector(MetricCollector):
    """Collector for portfolio metrics"""

    def __init__(self, initial_capital: float = 25000.0, buffer_size: int = 1000):
        super().__init__("portfolio", MetricCategory.TRADING)
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.buffer_size = buffer_size

        # Current state
        self.current_equity = initial_capital
        self.current_cash = initial_capital
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

        # History for calculations
        self.equity_history = deque(maxlen=buffer_size)
        self.drawdown_history = deque(maxlen=buffer_size)

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register portfolio metrics"""

        self.register_metric("total_equity", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Total portfolio equity",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("cash_balance", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Available cash balance",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("unrealized_pnl", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Unrealized profit and loss",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("realized_pnl_session", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Realized P&L for current session",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("total_return_pct", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.PERCENTAGE,
            description="Total return percentage",
            unit="%",
            frequency="step"
        ))

        self.register_metric("max_drawdown_pct", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.PERCENTAGE,
            description="Maximum drawdown percentage",
            unit="%",
            frequency="step"
        ))

        self.register_metric("current_drawdown_pct", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.PERCENTAGE,
            description="Current drawdown percentage",
            unit="%",
            frequency="step"
        ))

        self.register_metric("sharpe_ratio", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.GAUGE,
            description="Sharpe ratio (rolling)",
            unit="ratio",
            frequency="step"
        ))

        self.register_metric("volatility_pct", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.PERCENTAGE,
            description="Portfolio volatility",
            unit="%",
            frequency="step"
        ))

    def collect(self) -> Dict[str, MetricValue]:
        """Collect portfolio metrics"""
        metrics = {}

        try:
            # Basic portfolio values
            metrics[f"{self.category.value}.{self.name}.total_equity"] = MetricValue(self.current_equity)
            metrics[f"{self.category.value}.{self.name}.cash_balance"] = MetricValue(self.current_cash)
            metrics[f"{self.category.value}.{self.name}.unrealized_pnl"] = MetricValue(self.unrealized_pnl)
            metrics[f"{self.category.value}.{self.name}.realized_pnl_session"] = MetricValue(self.realized_pnl)

            # Return percentage
            total_return_pct = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
            metrics[f"{self.category.value}.{self.name}.total_return_pct"] = MetricValue(total_return_pct)

            # Drawdown calculations
            if self.equity_history:
                max_equity = max(self.equity_history)
                current_dd_pct = ((max_equity - self.current_equity) / max_equity) * 100 if max_equity > 0 else 0
                metrics[f"{self.category.value}.{self.name}.current_drawdown_pct"] = MetricValue(current_dd_pct)

                # Calculate maximum drawdown
                max_dd = self._calculate_max_drawdown()
                metrics[f"{self.category.value}.{self.name}.max_drawdown_pct"] = MetricValue(max_dd)

                # Calculate Sharpe ratio and volatility
                if len(self.equity_history) > 10:
                    sharpe, volatility = self._calculate_risk_metrics()
                    metrics[f"{self.category.value}.{self.name}.sharpe_ratio"] = MetricValue(sharpe)
                    metrics[f"{self.category.value}.{self.name}.volatility_pct"] = MetricValue(volatility)

        except Exception as e:
            self.logger.debug(f"Error collecting portfolio metrics: {e}")

        return metrics

    def update_portfolio_state(self, equity: float, cash: float,
                               unrealized_pnl: float, realized_pnl: float):
        """Update the current portfolio state"""
        self.current_equity = equity
        self.current_cash = cash
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl

        # Add to history
        self.equity_history.append(equity)

        # Calculate current drawdown
        if self.equity_history:
            max_equity = max(self.equity_history)
            current_dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
            self.drawdown_history.append(current_dd)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if len(self.equity_history) < 2:
            return 0.0

        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        max_dd = np.max(drawdowns) * 100

        return max_dd

    def _calculate_risk_metrics(self) -> tuple[float, float]:
        """Calculate Sharpe ratio and volatility"""
        if len(self.equity_history) < 10:
            return 0.0, 0.0

        # Calculate returns
        equity_array = np.array(self.equity_history)
        returns = np.diff(equity_array) / equity_array[:-1]

        if len(returns) < 2:
            return 0.0, 0.0

        # Calculate volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) * 100  # Assuming daily steps

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        mean_return = np.mean(returns)
        if volatility > 0:
            sharpe = (mean_return * 252) / (volatility / 100)  # Annualized
        else:
            sharpe = 0.0

        return sharpe, volatility

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)


class PositionMetricsCollector(MetricCollector):
    """Collector for position metrics"""

    def __init__(self, symbol: str):
        super().__init__("position", MetricCategory.TRADING)
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol

        # Current position state
        self.current_quantity = 0.0
        self.current_side = "FLAT"
        self.avg_entry_price = 0.0
        self.market_value = 0.0
        self.position_pnl = 0.0
        self.current_price = 0.0

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register position metrics"""

        self.register_metric("quantity", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.GAUGE,
            description=f"Position quantity for {self.symbol}",
            unit="shares",
            frequency="step"
        ))

        self.register_metric("side", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.GAUGE,
            description=f"Position side for {self.symbol}",
            unit="side",
            frequency="step"
        ))

        self.register_metric("avg_entry_price", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description=f"Average entry price for {self.symbol}",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("market_value", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description=f"Market value of position for {self.symbol}",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("unrealized_pnl", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description=f"Unrealized P&L for {self.symbol}",
            unit="USD",
            frequency="step"
        ))

        self.register_metric("unrealized_pnl_pct", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.PERCENTAGE,
            description=f"Unrealized P&L percentage for {self.symbol}",
            unit="%",
            frequency="step"
        ))

        self.register_metric("current_price", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description=f"Current market price for {self.symbol}",
            unit="USD",
            frequency="step"
        ))

    def collect(self) -> Dict[str, MetricValue]:
        """Collect position metrics"""
        metrics = {}

        try:
            metrics[f"{self.category.value}.{self.name}.quantity"] = MetricValue(self.current_quantity)

            # Convert side to numeric for W&B
            side_numeric = {"FLAT": 0, "LONG": 1, "SHORT": -1}.get(self.current_side, 0)
            metrics[f"{self.category.value}.{self.name}.side"] = MetricValue(side_numeric)

            metrics[f"{self.category.value}.{self.name}.avg_entry_price"] = MetricValue(self.avg_entry_price)
            metrics[f"{self.category.value}.{self.name}.market_value"] = MetricValue(self.market_value)
            metrics[f"{self.category.value}.{self.name}.unrealized_pnl"] = MetricValue(self.position_pnl)
            metrics[f"{self.category.value}.{self.name}.current_price"] = MetricValue(self.current_price)

            # Calculate unrealized P&L percentage
            if self.avg_entry_price > 0 and self.current_quantity != 0:
                pnl_pct = ((self.current_price - self.avg_entry_price) / self.avg_entry_price) * 100
                if self.current_side == "SHORT":
                    pnl_pct = -pnl_pct
                metrics[f"{self.category.value}.{self.name}.unrealized_pnl_pct"] = MetricValue(pnl_pct)

        except Exception as e:
            self.logger.debug(f"Error collecting position metrics: {e}")

        return metrics

    def update_position(self, quantity: float, side: str, avg_entry_price: float,
                        market_value: float, unrealized_pnl: float, current_price: float):
        """Update position state"""
        self.current_quantity = quantity
        self.current_side = side
        self.avg_entry_price = avg_entry_price
        self.market_value = market_value
        self.position_pnl = unrealized_pnl
        self.current_price = current_price

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)


class TradeMetricsCollector(MetricCollector):
    """Collector for trade performance metrics"""

    def __init__(self, buffer_size: int = 100):
        super().__init__("trades", MetricCategory.TRADING)
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size

        # Trade tracking
        self.completed_trades = deque(maxlen=buffer_size)
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Register metrics
        self._register_metrics()

    def _register_metrics(self):
        """Register trade metrics"""

        self.register_metric("total_trades", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.COUNTER,
            description="Total number of completed trades",
            unit="trades",
            frequency="episode"
        ))

        self.register_metric("win_rate", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of winning trades",
            unit="%",
            frequency="episode"
        ))

        self.register_metric("avg_trade_pnl", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Average trade P&L",
            unit="USD",
            frequency="episode"
        ))

        self.register_metric("avg_winning_trade", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Average winning trade P&L",
            unit="USD",
            frequency="episode"
        ))

        self.register_metric("avg_losing_trade", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Average losing trade P&L",
            unit="USD",
            frequency="episode"
        ))

        self.register_metric("profit_factor", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.GAUGE,
            description="Ratio of gross profit to gross loss",
            unit="ratio",
            frequency="episode"
        ))

        self.register_metric("largest_win", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Largest winning trade",
            unit="USD",
            frequency="episode"
        ))

        self.register_metric("largest_loss", MetricMetadata(
            category=MetricCategory.TRADING,
            metric_type=MetricType.CURRENCY,
            description="Largest losing trade",
            unit="USD",
            frequency="episode"
        ))

    def collect(self) -> Dict[str, MetricValue]:
        """Collect trade metrics"""
        metrics = {}

        if not self.completed_trades:
            return metrics

        try:
            # Basic counts
            metrics[f"{self.category.value}.{self.name}.total_trades"] = MetricValue(self.trade_count)

            # Calculate win rate
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            metrics[f"{self.category.value}.{self.name}.win_rate"] = MetricValue(win_rate)

            # Calculate P&L metrics
            trade_pnls = [trade['realized_pnl'] for trade in self.completed_trades if 'realized_pnl' in trade]

            if trade_pnls:
                avg_pnl = np.mean(trade_pnls)
                metrics[f"{self.category.value}.{self.name}.avg_trade_pnl"] = MetricValue(avg_pnl)

                winning_pnls = [pnl for pnl in trade_pnls if pnl > 0]
                losing_pnls = [pnl for pnl in trade_pnls if pnl <= 0]

                if winning_pnls:
                    avg_win = np.mean(winning_pnls)
                    largest_win = max(winning_pnls)
                    metrics[f"{self.category.value}.{self.name}.avg_winning_trade"] = MetricValue(avg_win)
                    metrics[f"{self.category.value}.{self.name}.largest_win"] = MetricValue(largest_win)

                if losing_pnls:
                    avg_loss = np.mean(losing_pnls)
                    largest_loss = min(losing_pnls)
                    metrics[f"{self.category.value}.{self.name}.avg_losing_trade"] = MetricValue(avg_loss)
                    metrics[f"{self.category.value}.{self.name}.largest_loss"] = MetricValue(largest_loss)

                # Profit factor
                gross_profit = sum(winning_pnls) if winning_pnls else 0
                gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
                metrics[f"{self.category.value}.{self.name}.profit_factor"] = MetricValue(profit_factor)

        except Exception as e:
            self.logger.debug(f"Error collecting trade metrics: {e}")

        return metrics

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade"""
        self.completed_trades.append(trade_data)
        self.trade_count += 1

        realized_pnl = trade_data.get('realized_pnl', 0)
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)