"""
Enhanced Execution Simulator - Handles action decoding and realistic execution

This simulator is responsible for:
1. Decoding agent actions into order parameters
2. Realistic execution simulation with improved slippage, latency, fees
3. Order validation and rejection handling
4. Market impact modeling
5. Session tracking and statistics

The environment's job is simplified to just handle portfolio updates.
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, NamedTuple, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import (
    OrderTypeEnum,
    OrderSideEnum,
    PositionSideEnum,
    FillDetails,
)


class RejectionReason(Enum):
    """Order rejection reasons."""

    INVALID_PRICES = "INVALID_PRICES"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    NO_POSITION_TO_SELL = "NO_POSITION_TO_SELL"
    QUANTITY_TOO_SMALL = "QUANTITY_TOO_SMALL"
    MARKET_CLOSED = "MARKET_CLOSED"
    POSITION_LIMIT = "POSITION_LIMIT"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class ActionDecodeResult:
    """Result of action decoding."""

    action_type: str  # "BUY", "SELL", "HOLD"
    size_float: float  # 0.25, 0.50, 0.75, 1.0
    raw_action: List[int]
    is_valid: bool = True
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "action_type": self.action_type,
            "size_float": self.size_float,
            "raw_action": self.raw_action,
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason.value
            if self.rejection_reason
            else None,
            "rejection_details": self.rejection_details,
        }


@dataclass
class OrderRequest:
    """Validated order request."""

    asset_id: str
    order_type: OrderTypeEnum
    order_side: OrderSideEnum
    quantity: float
    ideal_ask_price: float
    ideal_bid_price: float
    decision_timestamp: datetime

    @property
    def ideal_price(self) -> float:
        """Get ideal price based on order side."""
        return (
            self.ideal_ask_price
            if self.order_side == OrderSideEnum.BUY
            else self.ideal_bid_price
        )


@dataclass
class ExecutionContext:
    """Context for order execution."""

    market_state: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    session_volume: float
    session_turnover: float
    time_of_day: float  # Fraction of trading day (0.0 = open, 1.0 = close)


class ExecutionResult(NamedTuple):
    """Complete execution result."""

    fill_details: Optional[FillDetails]
    action_decode_result: ActionDecodeResult
    order_request: Optional[OrderRequest]
    execution_stats: Dict[str, Any]


class ExecutionSimulator:
    """Enhanced execution simulator with action decoding and realistic execution."""

    def __init__(
        self,
        logger: logging.Logger,
        simulation_config: Any,  # Config.simulation
        np_random: np.random.Generator,
        market_simulator: MarketSimulator,
    ):
        self.logger = logger
        self.simulation_config = simulation_config
        self.np_random = np_random
        self.market_simulator = market_simulator

        # Action space configuration - single index mapping
        # Import here to avoid circular imports
        from core.types import single_index_to_type_size
        self.action_mapper = single_index_to_type_size

        # Execution parameters from schema
        self.base_latency_ms = simulation_config.mean_latency_ms
        self.latency_std_ms = simulation_config.latency_std_dev_ms

        # Slippage model parameters from schema
        self.base_slippage_bps = simulation_config.base_slippage_bps
        self.max_slippage_bps = simulation_config.max_total_slippage_bps
        self.volume_impact_factor = (
            simulation_config.size_impact_slippage_bps_per_unit / 10000.0
        )  # Convert bps to factor
        self.market_impact_coefficient = simulation_config.market_impact_coefficient

        # Commission and fees from schema
        self.commission_per_share = simulation_config.commission_per_share
        self.fee_per_share = simulation_config.fee_per_share
        self.min_commission = simulation_config.min_commission_per_order
        self.max_commission_pct = simulation_config.max_commission_pct_of_value

        # Trading limits from schema
        self.allow_shorting = simulation_config.allow_shorting

        # Session tracking
        self.session_fills = 0
        self.session_volume = 0.0
        self.session_turnover = 0.0
        self.session_commission = 0.0
        self.session_slippage = 0.0
        self.total_orders_attempted = 0
        self.total_orders_filled = 0
        self.total_orders_rejected = 0

        # Performance tracking
        self.rejection_counts = {reason: 0 for reason in RejectionReason}
        self.fill_latencies = []
        self.slippage_history = []

    def decode_action(
        self, raw_action, current_time: Optional[datetime] = None
    ) -> ActionDecodeResult:
        """
        Decode agent's raw action into structured format.

        Args:
            raw_action: Agent's action (single integer index 0-6)
            current_time: Current simulation time for market hours check

        Returns:
            ActionDecodeResult with decoded action and validation
        """
        try:
            # Convert to single integer index
            if isinstance(raw_action, (tuple, list)):
                action_index = int(raw_action[0])  # Take first element if tuple/list
                raw_action_list = [action_index]
            elif hasattr(raw_action, "item"):  # NumPy array or PyTorch tensor
                action_index = int(raw_action.item())
                raw_action_list = [action_index]
            else:
                action_index = int(raw_action)
                raw_action_list = [action_index]

            # Validate action index bounds
            if not 0 <= action_index <= 6:
                self.logger.warning(f"Invalid action index: {action_index}, defaulting to HOLD")
                action_index = 0
                raw_action_list = [0]

            # Map single index to (action_type, size)
            action_type_enum, size_float = self.action_mapper(action_index)
            action_type = action_type_enum.name  # Convert enum to string

            # Basic validation
            if current_time and self._is_market_closed(current_time):
                return ActionDecodeResult(
                    action_type=action_type,
                    size_float=size_float,
                    raw_action=raw_action_list,
                    is_valid=False,
                    rejection_reason=RejectionReason.MARKET_CLOSED,
                    rejection_details="Market is closed",
                )

            return ActionDecodeResult(
                action_type=action_type,
                size_float=size_float,
                raw_action=raw_action_list,
                is_valid=True,
            )

        except Exception as e:
            self.logger.error(f"Error decoding action {raw_action}: {e}")
            return ActionDecodeResult(
                action_type="HOLD",
                size_float=0.0,
                raw_action=[0],
                is_valid=False,
                rejection_reason=RejectionReason.SYSTEM_ERROR,
                rejection_details=str(e),
            )

    def validate_and_create_order(
        self,
        action_result: ActionDecodeResult,
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        primary_asset: str,
        portfolio_manager,
    ) -> Optional[OrderRequest]:
        """
        Validate decoded action and create order request.

        This handles the order logic that was previously in the trading environment.
        """
        if not action_result.is_valid or action_result.action_type == "HOLD":
            return None

        # Get market prices
        ideal_ask = market_state.get("best_ask_price")
        ideal_bid = market_state.get("best_bid_price")
        current_price = market_state.get("current_price")

        # Handle missing prices
        if ideal_ask is None or ideal_bid is None:
            if current_price and current_price > 0:
                spread_factor = 0.0002  # 2 bps default spread
                ideal_ask = current_price * (1 + spread_factor)
                ideal_bid = current_price * (1 - spread_factor)
            else:
                action_result.is_valid = False
                action_result.rejection_reason = RejectionReason.INVALID_PRICES
                action_result.rejection_details = "Missing market prices"
                return None

        # Validate prices
        if ideal_ask <= 0 or ideal_bid <= 0 or ideal_ask <= ideal_bid:
            action_result.is_valid = False
            action_result.rejection_reason = RejectionReason.INVALID_PRICES
            action_result.rejection_details = (
                f"Invalid BBO: Ask ${ideal_ask:.4f}, Bid ${ideal_bid:.4f}"
            )
            return None

        # Get position data
        positions = portfolio_state.get("positions", {})
        pos_data = positions.get(primary_asset)
        if not pos_data:
            action_result.is_valid = False
            action_result.rejection_reason = RejectionReason.INVALID_SYMBOL
            action_result.rejection_details = f"No position data for {primary_asset}"
            return None

        current_qty = pos_data.get("quantity", 0.0)
        current_side = pos_data.get("current_side", PositionSideEnum.FLAT)
        cash = portfolio_state.get("cash", 0.0)
        total_equity = portfolio_state.get("total_equity", 0.0)

        # Get portfolio configuration from simulation config
        max_position_ratio = self.simulation_config.max_position_value_ratio
        max_pos_value = total_equity * max_position_ratio

        # Determine order parameters
        quantity_to_trade = 0.0
        order_side = None
        # Get timestamp from market state (it's already in the simulation time)
        decision_timestamp = market_state.get("timestamp")
        if decision_timestamp is None:
            decision_timestamp = datetime.now(timezone.utc)
        elif isinstance(decision_timestamp, str):
            decision_timestamp = pd.Timestamp(decision_timestamp).to_pydatetime()

        if action_result.action_type == "BUY":
            # Calculate target buy value based on available cash
            target_value = cash * action_result.size_float
            # Limit by max position value
            target_value = min(target_value, max_pos_value)

            if target_value < 10.0:  # Minimum $10 order
                action_result.is_valid = False
                action_result.rejection_reason = RejectionReason.INSUFFICIENT_CASH
                action_result.rejection_details = (
                    f"Insufficient buying power: ${target_value:.2f}"
                )
                return None

            quantity_to_trade = target_value / ideal_ask
            order_side = OrderSideEnum.BUY

            # Handle covering short positions
            if current_side == PositionSideEnum.SHORT:
                quantity_to_trade += abs(current_qty)

        elif action_result.action_type == "SELL":
            if current_side == PositionSideEnum.LONG and current_qty > 0:
                quantity_to_trade = action_result.size_float * current_qty
                order_side = OrderSideEnum.SELL
            else:
                action_result.is_valid = False
                action_result.rejection_reason = RejectionReason.NO_POSITION_TO_SELL
                action_result.rejection_details = "Cannot sell without long position"
                return None

        # Final validation
        if quantity_to_trade < 1.0:  # Minimum 1 share
            action_result.is_valid = False
            action_result.rejection_reason = RejectionReason.QUANTITY_TOO_SMALL
            action_result.rejection_details = (
                f"Quantity too small: {quantity_to_trade:.6f}"
            )
            return None

        return OrderRequest(
            asset_id=primary_asset,
            order_type=OrderTypeEnum.MARKET,
            order_side=order_side,
            quantity=abs(quantity_to_trade),
            ideal_ask_price=ideal_ask,
            ideal_bid_price=ideal_bid,
            decision_timestamp=decision_timestamp,
        )

    def execute_order(
        self, order_request: OrderRequest, execution_context: ExecutionContext
    ) -> Optional[FillDetails]:
        """
        Execute validated order with realistic simulation.

        Args:
            order_request: Validated order to execute
            execution_context: Market and session context

        Returns:
            FillDetails if order filled, None if rejected
        """
        try:
            self.total_orders_attempted += 1

            # Simulate execution latency
            latency_ms = self._simulate_latency()

            # Calculate execution price with slippage
            executed_price, slippage_bps = self._calculate_execution_price(
                order_request, execution_context
            )

            # Calculate costs
            commission = self._calculate_commission(
                order_request.quantity, executed_price
            )
            fees = self._calculate_fees(order_request.quantity)
            slippage_cost = (
                abs(executed_price - order_request.ideal_price) * order_request.quantity
            )

            # Create fill details
            decision_ts = order_request.decision_timestamp
            # Convert pandas Timestamp to datetime if needed
            if hasattr(decision_ts, 'to_pydatetime'):
                decision_ts = decision_ts.to_pydatetime()
            
            fill_timestamp = decision_ts + timedelta(
                milliseconds=latency_ms
            )

            fill_details = FillDetails(
                asset_id=order_request.asset_id,
                fill_timestamp=fill_timestamp,
                order_type=order_request.order_type,
                order_side=order_request.order_side,
                requested_quantity=order_request.quantity,
                executed_quantity=order_request.quantity,  # Assume full fill for market orders
                executed_price=executed_price,
                commission=commission,
                fees=fees,
                slippage_cost_total=slippage_cost,
            )

            # Update session tracking
            self._update_session_stats(fill_details, latency_ms, slippage_bps)

            # NOTE: Execution metrics will be recorded by callbacks in TradingEnvironment
            # after portfolio processing with correct P&L calculation

            self.total_orders_filled += 1
            return fill_details

        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            self.total_orders_rejected += 1
            return None

    def execute_action(
        self,
        raw_action,
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        primary_asset: str,
        portfolio_manager,
    ) -> ExecutionResult:
        """
        Complete action processing pipeline: decode -> validate -> execute.

        This is the main entry point for the trading environment.
        """
        current_time = market_state.get("timestamp_utc")

        # Step 1: Decode action
        action_result = self.decode_action(raw_action, current_time)

        # Step 2: Validate and create order
        order_request = None
        if action_result.is_valid and action_result.action_type != "HOLD":
            order_request = self.validate_and_create_order(
                action_result,
                market_state,
                portfolio_state,
                primary_asset,
                portfolio_manager,
            )

        # Step 3: Execute order
        fill_details = None
        if order_request:
            execution_context = ExecutionContext(
                market_state=market_state,
                portfolio_state=portfolio_state,
                session_volume=self.session_volume,
                session_turnover=self.session_turnover,
                time_of_day=self._get_time_of_day(current_time),
            )
            fill_details = self.execute_order(order_request, execution_context)

            if fill_details is None:
                self.total_orders_rejected += 1
                if action_result.is_valid:  # Order was created but execution failed
                    action_result.is_valid = False
                    action_result.rejection_reason = RejectionReason.SYSTEM_ERROR
                    action_result.rejection_details = "Execution failed"
        elif not action_result.is_valid:
            self.total_orders_rejected += 1
            if action_result.rejection_reason:
                self.rejection_counts[action_result.rejection_reason] += 1

        # Compile execution stats
        execution_stats = {
            "session_fills": self.session_fills,
            "session_volume": self.session_volume,
            "session_turnover": self.session_turnover,
            "total_attempted": self.total_orders_attempted,
            "total_filled": self.total_orders_filled,
            "total_rejected": self.total_orders_rejected,
            "fill_rate": self.total_orders_filled / max(1, self.total_orders_attempted),
        }

        return ExecutionResult(
            fill_details=fill_details,
            action_decode_result=action_result,
            order_request=order_request,
            execution_stats=execution_stats,
        )

    def _simulate_latency(self) -> float:
        """Simulate execution latency in milliseconds."""
        if self.latency_std_ms <= 0:
            return self.base_latency_ms

        latency = self.np_random.normal(self.base_latency_ms, self.latency_std_ms)
        return max(1.0, latency)  # Minimum 1ms latency

    def _calculate_execution_price(
        self, order_request: OrderRequest, context: ExecutionContext
    ) -> Tuple[float, float]:
        """
        Calculate execution price with improved slippage model.

        Returns:
            (executed_price, slippage_bps)
        """
        base_price = order_request.ideal_price

        # Base slippage
        base_slippage_factor = self.base_slippage_bps / 10000.0

        # Volume impact based on relative order size
        volume_impact = 0.0
        if context.session_turnover > 0:
            order_value = order_request.quantity * base_price
            relative_order_size = order_value / context.session_turnover
            volume_impact = relative_order_size * self.volume_impact_factor

        # Market impact based on order size
        market_impact = 0.0
        order_value = order_request.quantity * base_price
        if self.simulation_config.market_impact_model == "linear":
            market_impact = order_value * self.market_impact_coefficient
        elif self.simulation_config.market_impact_model == "square_root":
            market_impact = (order_value**0.5) * self.market_impact_coefficient

        # Time-of-day impact (market open/close = higher slippage)
        time_impact = 0.0
        if 0 <= context.time_of_day <= 0.1 or 0.9 <= context.time_of_day <= 1.0:
            time_impact = 5.0  # 5 bps additional at open/close

        # Total slippage in bps
        total_slippage_bps = (
            self.base_slippage_bps
            + (volume_impact * 10000)
            + market_impact
            + time_impact
        )
        total_slippage_bps = min(total_slippage_bps, self.max_slippage_bps)
        total_slippage_factor = total_slippage_bps / 10000.0

        # Apply slippage based on order side
        if order_request.order_side == OrderSideEnum.BUY:
            executed_price = base_price * (1 + total_slippage_factor)
        else:  # SELL
            executed_price = base_price * (1 - total_slippage_factor)

        return executed_price, total_slippage_bps

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission with min/max limits."""
        trade_value = quantity * price

        # Base commission
        commission = quantity * self.commission_per_share

        # Apply minimum
        commission = max(commission, self.min_commission)

        # Apply maximum as percentage of trade value
        max_commission = trade_value * (self.max_commission_pct / 100.0)
        commission = min(commission, max_commission)

        return commission

    def _calculate_fees(self, quantity: float) -> float:
        """Calculate regulatory and exchange fees."""
        return quantity * self.fee_per_share

    def _update_session_stats(
        self, fill: FillDetails, latency_ms: float, slippage_bps: float
    ):
        """Update session tracking statistics."""
        self.session_fills += 1
        self.session_volume += fill.executed_quantity
        self.session_turnover += fill.executed_quantity * fill.executed_price
        self.session_commission += fill.commission
        self.session_slippage += fill.slippage_cost_total

        # Performance tracking
        self.fill_latencies.append(latency_ms)
        self.slippage_history.append(slippage_bps)

    # Removed _record_execution_metrics - handled by callbacks in TradingEnvironment

    def _is_market_closed(self, current_time: datetime) -> bool:
        """Check if market is closed."""
        # Simple market hours check (4 AM - 8 PM ET)
        # This should be enhanced with actual market calendar
        et_time = current_time.astimezone(timezone.utc)  # Simplified
        hour = et_time.hour
        return hour < 4 or hour > 20

    def _get_time_of_day(self, current_time: Optional[datetime]) -> float:
        """Get fraction of trading day (0.0 = open, 1.0 = close)."""
        if not current_time:
            return 0.5

        # Simplified: 4 AM = 0.0, 8 PM = 1.0
        et_time = current_time.astimezone(timezone.utc)
        hour = et_time.hour + et_time.minute / 60.0

        if hour < 4:
            return 0.0
        elif hour > 20:
            return 1.0
        else:
            return (hour - 4) / 16.0  # 16 hour trading day

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        avg_latency = np.mean(self.fill_latencies) if self.fill_latencies else 0.0
        avg_slippage = np.mean(self.slippage_history) if self.slippage_history else 0.0

        return {
            "session_fills": self.session_fills,
            "session_volume": self.session_volume,
            "session_turnover": self.session_turnover,
            "session_commission": self.session_commission,
            "session_slippage": self.session_slippage,
            "total_orders_attempted": self.total_orders_attempted,
            "total_orders_filled": self.total_orders_filled,
            "total_orders_rejected": self.total_orders_rejected,
            "fill_rate": self.total_orders_filled / max(1, self.total_orders_attempted),
            "avg_latency_ms": avg_latency,
            "avg_slippage_bps": avg_slippage,
            "rejection_counts": dict(self.rejection_counts),
            "avg_trade_size": self.session_volume / max(1, self.session_fills),
        }

    def reset(self, np_random_seed_source: Optional[np.random.Generator] = None):
        """Reset for new episode/session."""
        if np_random_seed_source:
            self.np_random = np_random_seed_source

        # Reset session tracking
        self.session_fills = 0
        self.session_volume = 0.0
        self.session_turnover = 0.0
        self.session_commission = 0.0
        self.session_slippage = 0.0
        self.total_orders_attempted = 0
        self.total_orders_filled = 0
        self.total_orders_rejected = 0

        # Reset performance tracking
        self.rejection_counts = {reason: 0 for reason in RejectionReason}
        self.fill_latencies = []
        self.slippage_history = []
