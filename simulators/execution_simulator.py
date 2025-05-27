# simulators/execution_simulator.py - CLEAN: Streamlined execution with metrics integration

import logging
from datetime import datetime, timedelta
from typing import Optional, NamedTuple
import numpy as np
import pandas as pd

from config.schemas import SimulationConfig
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import OrderTypeEnum, OrderSideEnum, FillDetails
from enum import Enum
from dataclasses import dataclass


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    """Request to execute an order."""
    side: str
    quantity: float
    order_type: OrderType
    symbol: Optional[str] = None
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __init__(self, side: str, quantity: float, order_type: OrderType, 
                 symbol: Optional[str] = None, price: Optional[float] = None, 
                 timestamp: Optional[datetime] = None):
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp


class ExecutionResult(NamedTuple):
    """Result of order execution."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    requested_price: float
    executed_price: float
    requested_size: float
    executed_size: float
    slippage: float
    commission: float
    latency_ms: float
    rejection_reason: Optional[str] = None
    
    @property
    def executed(self) -> bool:
        """Check if order was executed."""
        return self.executed_size > 0 and self.rejection_reason is None


class ExecutionSimulator:
    """Clean execution simulator with metrics integration"""

    def __init__(self,
                 logger: logging.Logger,
                 simulation_config: SimulationConfig,
                 np_random: np.random.Generator,
                 market_simulator: MarketSimulator,
                 metrics_integrator=None):
        self.logger = logger
        self.simulation_config:SimulationConfig = simulation_config
        self.np_random = np_random
        self.market_simulator = market_simulator
        self.metrics_integrator = metrics_integrator

        # Execution parameters
        self.mean_latency_ms = self.simulation_config.mean_latency_ms
        self.latency_std_dev_ms = self.simulation_config.latency_std_dev_ms
        self.base_slippage_bps = self.simulation_config.base_slippage_bps
        self.size_impact_slippage_bps_per_unit = self.simulation_config.size_impact_slippage_bps_per_unit
        self.max_total_slippage_bps = self.simulation_config.max_total_slippage_bps

        # Cost parameters
        self.commission_per_share = self.simulation_config.commission_per_share
        self.fee_per_share = self.simulation_config.fee_per_share
        self.min_commission_per_order = self.simulation_config.min_commission_per_order
        self.max_commission_pct_of_value = self.simulation_config.max_commission_pct_of_value

        # Session tracking
        self.session_fills = 0
        self.session_volume = 0.0
        self.session_turnover = 0.0

    def _simulate_latency(self) -> timedelta:
        """Simulate execution latency"""
        if self.latency_std_dev_ms <= 1e-9:
            latency_ms = self.mean_latency_ms
        else:
            latency_ms = self.np_random.normal(self.mean_latency_ms, self.latency_std_dev_ms)
        return timedelta(milliseconds=max(0, round(latency_ms)))

    def execute_order(self,
                      asset_id: str,
                      order_type: OrderTypeEnum,
                      order_side: OrderSideEnum,
                      requested_quantity: float,
                      ideal_decision_price_ask: float,
                      ideal_decision_price_bid: float,
                      decision_timestamp: datetime
                      ) -> Optional[FillDetails]:
        """Execute order with metrics tracking"""

        if order_type != OrderTypeEnum.MARKET:
            self.logger.warning(f"Only MARKET orders supported. Received: {order_type}")
            return None

        if requested_quantity <= 1e-9:
            return None

        # Calculate fees and commission
        fees = requested_quantity * self.fee_per_share
        commission = requested_quantity * self.commission_per_share

        # Apply minimum commission
        if self.min_commission_per_order is not None:
            commission = max(commission, self.min_commission_per_order)

        # Cap commission as percentage of trade value if specified
        total_transaction_value = requested_quantity * (
            ideal_decision_price_ask if order_side == OrderSideEnum.BUY else ideal_decision_price_bid)

        if self.max_commission_pct_of_value is not None:
            max_comm_by_value = total_transaction_value * (self.max_commission_pct_of_value / 100.0)
            commission = min(commission, max_comm_by_value)

        # Ensure non-negative values
        commission = max(0, commission)
        fees = max(0, fees)

        # Calculate execution price with slippage
        slippage_factor = self.base_slippage_bps / 10000.0
        if order_side == OrderSideEnum.BUY:
            executed_price = ideal_decision_price_ask * (1 + slippage_factor)
        else:  # SELL
            executed_price = ideal_decision_price_bid * (1 - slippage_factor)

        # Calculate slippage cost
        ideal_price = ideal_decision_price_ask if order_side == OrderSideEnum.BUY else ideal_decision_price_bid
        slippage_cost_total = abs(executed_price - ideal_price) * requested_quantity

        # Update session tracking
        self.session_fills += 1
        self.session_volume += requested_quantity
        self.session_turnover += requested_quantity * executed_price

        # Create fill details
        fill_details = FillDetails(
            asset_id=asset_id,
            fill_timestamp=decision_timestamp,
            order_type=order_type,
            order_side=order_side,
            requested_quantity=requested_quantity,
            executed_quantity=requested_quantity,
            executed_price=executed_price,
            commission=commission,
            fees=fees,
            slippage_cost_total=slippage_cost_total
        )

        # Record fill in metrics if available
        if self.metrics_integrator:
            self.metrics_integrator.record_fill({
                'executed_quantity': requested_quantity,
                'executed_price': executed_price,
                'commission': commission,
                'fees': fees,
                'slippage_cost_total': slippage_cost_total
            })

        return fill_details

    def get_session_stats(self) -> dict:
        """Get execution statistics for current session"""
        return {
            'total_fills': self.session_fills,
            'total_volume': self.session_volume,
            'total_turnover': self.session_turnover,
            'avg_fill_size': self.session_volume / self.session_fills if self.session_fills > 0 else 0.0
        }

    def reset(self, np_random_seed_source: Optional[np.random.Generator] = None):
        """Reset for new episode/session"""
        if np_random_seed_source:
            self.np_random = np_random_seed_source

        # Reset session tracking
        self.session_fills = 0
        self.session_volume = 0.0
        self.session_turnover = 0.0
    
    def simulate_execution(self, order_type: str, order_side: str, 
                          requested_quantity: float, symbol: str,
                          **kwargs) -> ExecutionResult:
        """Simulate order execution and return ExecutionResult."""
        # Get current timestamp
        timestamp = kwargs.get('decision_timestamp', pd.Timestamp.now())
        
        # Simulate latency
        latency = self._simulate_latency()
        latency_ms = latency.total_seconds() * 1000
        
        # Get prices
        ask_price = kwargs.get('ideal_decision_price_ask', 10.0)
        bid_price = kwargs.get('ideal_decision_price_bid', 10.0)
        
        # Determine execution price based on side
        if order_side.lower() == 'buy':
            requested_price = ask_price
            executed_price = ask_price * (1 + self.base_slippage_bps / 10000.0)
        else:
            requested_price = bid_price
            executed_price = bid_price * (1 - self.base_slippage_bps / 10000.0)
        
        # Calculate commission
        commission = requested_quantity * self.commission_per_share
        commission = max(commission, self.min_commission_per_order or 0)
        
        # Calculate slippage
        slippage = abs(executed_price - requested_price) / requested_price
        
        # Create execution result
        return ExecutionResult(
            order_id=f"ORD_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp + latency,
            symbol=symbol,
            side=order_side.lower(),
            requested_price=requested_price,
            executed_price=executed_price,
            requested_size=requested_quantity,
            executed_size=requested_quantity,  # Assume full fill
            slippage=slippage,
            commission=commission,
            latency_ms=latency_ms,
            rejection_reason=None
        )