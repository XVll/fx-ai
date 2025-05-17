import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import numpy as np

from config.config import ExecutionConfig
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import OrderTypeEnum, OrderSideEnum, FillDetails


class ExecutionSimulator:
    def __init__(self,
                 logger: logging.Logger,
                 config_exec: ExecutionConfig,
                 np_random: np.random.Generator,
                 market_simulator: MarketSimulator):
        self.logger = logger
        self.config = config_exec
        self.np_random = np_random
        self.market_simulator = market_simulator

        # Latency parameters
        self.mean_latency_ms = self.config.get('mean_latency_ms', 50)
        self.latency_std_dev_ms = self.config.get('latency_std_dev_ms', 10)

        # Slippage parameters
        self.base_slippage_bps = self.config.get('base_slippage_bps', 1.0)  # For crossing spread, market friction
        self.size_impact_slippage_bps_per_unit = self.config.get('size_impact_slippage_bps_per_unit', 0.05)
        self.max_total_slippage_bps = self.config.get('max_total_slippage_bps', 100.0)

        # NEW: Per-share commission and fee parameters
        self.commission_per_share = self.config.get('commission_per_share', 0.005)  # e.g., $0.005 per share
        self.fee_per_share = self.config.get('fee_per_share', 0.0005)  # e.g., $0.0005 per share
        self.min_commission_per_order = self.config.get('min_commission_per_order', None)  # e.g., 1.00 for $1 minimum
        # Optional: Cap commission as a percentage of trade value
        self.max_commission_pct_of_value = self.config.get('max_commission_pct_of_value', None)  # e.g., 1.0 for 1%

    def _simulate_latency(self) -> timedelta:
        if self.latency_std_dev_ms <= 1e-9:
            latency_ms = self.mean_latency_ms
        else:
            latency_ms = self.np_random.normal(self.mean_latency_ms, self.latency_std_dev_ms)
        return timedelta(milliseconds=max(0, latency_ms))

    def execute_order(self,
                      asset_id: str,
                      order_type: OrderTypeEnum,
                      order_side: OrderSideEnum,
                      requested_quantity: float,
                      ideal_decision_price_ask: float,
                      ideal_decision_price_bid: float,
                      decision_timestamp: datetime
                      ) -> Optional[FillDetails]:

        if order_type != OrderTypeEnum.MARKET:
            self.logger.warning(f"Only MARKET orders currently supported. Received: {order_type}")
            return None

        if requested_quantity <= 1e-9:
            self.logger.debug("Requested quantity too small, no execution.")
            return None

        latency_duration = self._simulate_latency()
        execution_attempt_timestamp = decision_timestamp + latency_duration
        market_state_at_exec = self.market_simulator.get_state_at_time(execution_attempt_timestamp)

        if not market_state_at_exec or \
                market_state_at_exec.get('best_bid_price') is None or \
                market_state_at_exec.get('best_ask_price') is None:
            self.logger.warning(f"No valid BBO at execution time {execution_attempt_timestamp} for {asset_id}. Order fails.")
            return None

        current_exec_bid_price = market_state_at_exec['best_bid_price']
        current_exec_ask_price = market_state_at_exec['best_ask_price']

        # --- Slippage Calculation (same as before) ---
        total_slippage_bps = self.base_slippage_bps
        total_slippage_bps += self.size_impact_slippage_bps_per_unit * requested_quantity
        total_slippage_bps = min(total_slippage_bps, self.max_total_slippage_bps)
        slippage_factor = total_slippage_bps / 10000.0
        executed_price = 0.0

        if order_side == OrderSideEnum.BUY:
            executed_price = current_exec_ask_price * (1 + slippage_factor)
        elif order_side == OrderSideEnum.SELL:
            executed_price = current_exec_bid_price * (1 - slippage_factor)

        executed_quantity = requested_quantity

        slippage_cost_total = 0.0
        if order_side == OrderSideEnum.BUY:
            slippage_cost_total = (executed_price - ideal_decision_price_ask) * executed_quantity
        elif order_side == OrderSideEnum.SELL:
            slippage_cost_total = (ideal_decision_price_bid - executed_price) * executed_quantity
        slippage_cost_total = max(0, slippage_cost_total)
        # --- End Slippage Calculation ---

        commission = executed_quantity * self.commission_per_share

        if self.min_commission_per_order is not None:
            commission = max(commission, self.min_commission_per_order)

        if self.max_commission_pct_of_value is not None:
            trade_value = executed_quantity * executed_price
            max_comm_by_value = trade_value * (self.max_commission_pct_of_value / 100.0)
            # Ensure the commission doesn't exceed this cap, but also respect min_commission if both are set
            # If min_commission_per_order is higher than max_comm_by_value, min_commission would typically take precedence,
            # or the broker might have specific rules. For simplicity here, we'll take the capped value if it's lower
            # than the per-share calc but ensure min_commission is still met.
            if commission > max_comm_by_value:
                commission = max_comm_by_value
                if self.min_commission_per_order is not None:  # Re-check min if capped by %
                    commission = max(commission, self.min_commission_per_order)

        fees = executed_quantity * self.fee_per_share
        # --- End Commission and Fee Calculation ---

        fill_details = FillDetails(
            asset_id=asset_id,
            fill_timestamp=execution_attempt_timestamp,
            order_type=order_type,
            order_side=order_side,
            requested_quantity=requested_quantity,
            executed_quantity=executed_quantity,
            executed_price=executed_price,
            commission=commission,
            fees=fees,
            slippage_cost_total=slippage_cost_total
        )
        self.logger.debug(f"Fill generated: {fill_details}")
        return fill_details

    def reset(self, np_random_seed_source: Optional[np.random.Generator] = None):
        if np_random_seed_source:
            self.np_random = np_random_seed_source
        self.logger.info("RealisticExecutionSimulator reset.")