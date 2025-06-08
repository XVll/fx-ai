"""
Simulation interfaces for market, execution, and portfolio management.

These interfaces enable realistic trading simulation with proper
separation of concerns between market data, order execution,
and portfolio tracking.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable
from datetime import datetime
import pandas as pd
import numpy as np

from v2.core.types import Symbol, MarketDataPoint, Configurable, Resettable, OrderSide, Quantity, OrderType, Price, ExecutionInfo, Cash


@runtime_checkable
class IMarketSimulator(Protocol):
    """Interface for market data simulation.
    
    Design principles:
    - Provide realistic market data replay
    - Support multiple data types (trades, quotes, bars)
    - Handle market hours and holidays
    - Enable time control for backtesting
    """
    
    @property
    def current_time(self) -> Optional[datetime]:
        """Current simulation time.
        
        Returns:
            Current timestamp or None if not initialized
        """
        ...
    
    @property
    def is_market_open(self) -> bool:
        """Whether market is currently open.
        
        Returns:
            True if within trading hours
            
        Design notes:
        - Consider pre/post market
        - Handle holidays
        """
        ...
    
    def initialize(
        self,
        symbol: Symbol,
        date: datetime,
        data: dict[str, pd.DataFrame]
    ) -> bool:
        """Initialize for specific symbol and date.
        
        Args:
            symbol: Trading symbol
            date: Simulation date
            data: Pre-loaded market data
            
        Returns:
            True if successful
            
        Design notes:
        - Validate data completeness
        - Set up time boundaries
        - Prepare data structures
        """
        ...
    
    def reset(
        self,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Reset to specific time or market open.
        
        Args:
            timestamp: Reset time (market open if None)
            
        Returns:
            True if successful
            
        Design notes:
        - Clear internal state
        - Position at valid market time
        """
        ...
    
    def step(
        self,
        seconds: int = 1
    ) -> bool:
        """Advance simulation time.
        
        Args:
            seconds: Seconds to advance
            
        Returns:
            True if successful, False if end of data
            
        Design notes:
        - Handle gaps in data
        - Update internal state
        - Check market hours
        """
        ...
    
    def get_market_data(
        self,
        lookback_seconds: int = 0
    ) -> MarketDataPoint:
        """Get current market data.
        
        Args:
            lookback_seconds: Seconds to look back
            
        Returns:
            Market data at current time
            
        Design notes:
        - Aggregate multiple data sources
        - Handle missing data
        - Compute derived fields (spread, VWAP)
        """
        ...
    
    def get_historical_data(
        self,
        start: datetime,
        end: datetime,
        data_type: str = "trades"
    ) -> pd.DataFrame:
        """Get historical data range.
        
        Args:
            start: Start time
            end: End time
            data_type: Type of data
            
        Returns:
            DataFrame of market data
            
        Design notes:
        - Respect current simulation time
        - No future data leakage
        """
        ...


class IExecutionSimulator(Configurable, Resettable):
    """Interface for order execution simulation.
    
    Design principles:
    - Realistic order execution with slippage
    - Support multiple order types
    - Model market impact and latency
    - Handle partial fills and rejections
    """
    
    @abstractmethod
    def execute_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        order_type: OrderType,
        price: Optional[Price] = None,
        time_in_force: str = "DAY"
    ) -> ExecutionInfo:
        """Execute a trading order.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order size
            order_type: Market, limit, etc.
            price: Limit price if applicable
            time_in_force: Order duration
            
        Returns:
            Execution details
            
        Design notes:
        - Simulate realistic fills
        - Model market impact
        - Handle order validation
        - Consider liquidity
        """
        ...
    
    @abstractmethod
    def estimate_execution(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        order_type: OrderType = OrderType.MARKET
    ) -> dict[str, float]:
        """Estimate execution without placing order.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell  
            quantity: Order size
            order_type: Order type
            
        Returns:
            Dict with estimates:
            - expected_price
            - expected_slippage
            - expected_impact
            - success_probability
            
        Design notes:
        - Used for planning
        - No side effects
        """
        ...
    
    @abstractmethod
    def set_latency_model(
        self,
        mean_ms: float,
        std_ms: float
    ) -> None:
        """Configure execution latency.
        
        Args:
            mean_ms: Mean latency in milliseconds
            std_ms: Standard deviation
            
        Design notes:
        - Model network delays
        - Add realism
        """
        ...
    
    @abstractmethod
    def set_slippage_model(
        self,
        model_type: str,
        parameters: dict[str, float]
    ) -> None:
        """Configure slippage model.
        
        Args:
            model_type: "linear", "sqrt", "custom"
            parameters: Model parameters
            
        Design notes:
        - Different models for different markets
        - Consider order size and volatility
        """
        ...


class IPortfolioSimulator(Configurable, Resettable):
    """Interface for portfolio management simulation.
    
    Design principles:
    - Track positions and cash accurately
    - Calculate P&L in real-time
    - Handle multiple assets
    - Enforce risk limits
    """
    
    @property
    def total_equity(self) -> Cash:
        """Total portfolio value.
        
        Returns:
            Cash + market value of positions
        """
        ...
    
    @property
    def cash_balance(self) -> Cash:
        """Available cash.
        
        Returns:
            Current cash balance
        """
        ...
    
    @property
    def positions(self) -> dict[Symbol, dict[str, Any]]:
        """Current positions.
        
        Returns:
            Dict mapping symbol to position info:
            - quantity: Position size
            - side: Long/short/flat
            - avg_entry_price: Average entry
            - market_value: Current value
            - unrealized_pnl: Unrealized P&L
        """
        ...
    
    @abstractmethod
    def initialize(
        self,
        initial_cash: Cash,
        initial_positions: Optional[dict[Symbol, Quantity]] = None
    ) -> None:
        """Initialize portfolio.
        
        Args:
            initial_cash: Starting cash
            initial_positions: Starting positions
            
        Design notes:
        - Set up tracking structures
        - Validate initial state
        """
        ...
    
    @abstractmethod
    def process_execution(
        self,
        execution: ExecutionInfo,
        current_time: datetime
    ) -> dict[str, Any]:
        """Process trade execution.
        
        Args:
            execution: Execution details
            current_time: Current time
            
        Returns:
            Portfolio update info
            
        Design notes:
        - Update positions
        - Deduct cash/commissions
        - Track trade history
        """
        ...
    
    @abstractmethod
    def update_market_values(
        self,
        market_prices: dict[Symbol, Price],
        current_time: datetime
    ) -> None:
        """Update position values with market prices.
        
        Args:
            market_prices: Current prices
            current_time: Current time
            
        Design notes:
        - Calculate unrealized P&L
        - Update equity curve
        - Check risk limits
        """
        ...
    
    @abstractmethod
    def get_position(
        self,
        symbol: Symbol
    ) -> dict[str, Any]:
        """Get position details.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position information
        """
        ...
    
    @abstractmethod
    def get_performance_metrics(
        self,
        lookback_periods: Optional[int] = None
    ) -> dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            lookback_periods: Periods to analyze
            
        Returns:
            Dict with metrics:
            - total_return
            - sharpe_ratio
            - max_drawdown
            - win_rate
            - avg_win_loss_ratio
            
        Design notes:
        - Standard metrics
        - Handle edge cases
        """
        ...
    
    @abstractmethod
    def check_risk_limits(
        self
    ) -> dict[str, bool]:
        """Check if risk limits are breached.
        
        Returns:
            Dict of limit checks:
            - max_position_size
            - max_leverage
            - max_drawdown
            - concentration_limit
        """
        ...


class ISimulationOrchestrator(Protocol):
    """Interface for coordinating simulation components.
    
    Design principles:
    - Coordinate market, execution, portfolio
    - Ensure consistency across components
    - Handle simulation lifecycle
    """
    
    def initialize_simulation(
        self,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime,
        initial_cash: Cash
    ) -> None:
        """Initialize full simulation.
        
        Args:
            symbol: Trading symbol
            start_date: Simulation start
            end_date: Simulation end
            initial_cash: Starting capital
            
        Design notes:
        - Load required data
        - Initialize all components
        - Validate setup
        """
        ...
    
    def process_action(
        self,
        action: dict[str, Any]
    ) -> dict[str, Any]:
        """Process trading action.
        
        Args:
            action: Action specification
            
        Returns:
            Result including:
            - execution_info
            - portfolio_update
            - market_impact
            
        Design notes:
        - Coordinate execution flow
        - Update all components
        - Return comprehensive result
        """
        ...
    
    def advance_time(
        self,
        seconds: int = 1
    ) -> bool:
        """Advance simulation time.
        
        Args:
            seconds: Time to advance
            
        Returns:
            True if successful
            
        Design notes:
        - Sync all components
        - Update market values
        - Check termination
        """
        ...
    
    def get_state(
        self
    ) -> dict[str, Any]:
        """Get complete simulation state.
        
        Returns:
            State dictionary
        """
        ...


class IBacktestEngine(Protocol):
    """High-level backtesting interface.
    
    Design principles:
    - Simple API for strategy testing
    - Comprehensive result analysis
    - Support multiple strategies
    """
    
    def run_backtest(
        self,
        strategy: Any,
        symbols: list[Symbol],
        start_date: datetime,
        end_date: datetime,
        initial_cash: Cash
    ) -> pd.DataFrame:
        """Run backtest on strategy.
        
        Args:
            strategy: Trading strategy
            symbols: Symbols to trade
            start_date: Backtest start
            end_date: Backtest end
            initial_cash: Starting capital
            
        Returns:
            Results DataFrame
            
        Design notes:
        - Handle multiple symbols
        - Aggregate results
        - Include transaction costs
        """
        ...
    
    def analyze_results(
        self,
        results: pd.DataFrame
    ) -> dict[str, Any]:
        """Analyze backtest results.
        
        Args:
            results: Backtest results
            
        Returns:
            Analysis dictionary
        """
        ...
