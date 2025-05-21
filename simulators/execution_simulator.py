import logging
from datetime import datetime, timedelta
from typing import Optional
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
        self.mean_latency_ms = self.config.mean_latency_ms
        self.latency_std_dev_ms = self.config.latency_std_dev_ms

        # Slippage parameters
        self.base_slippage_bps = self.config.base_slippage_bps
        self.size_impact_slippage_bps_per_unit = self.config.size_impact_slippage_bps_per_unit
        self.max_total_slippage_bps = self.config.max_total_slippage_bps

        # NEW: Per-share commission and fee parameters
        self.commission_per_share = self.config.commission_per_share
        self.fee_per_share = self.config.fee_per_share
        self.min_commission_per_order = self.config.min_commission_per_order
        # Optional: Cap commission as a percentage of trade value
        self.max_commission_pct_of_value = self.config.max_commission_pct_of_value

    def _simulate_latency(self) -> timedelta:
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

        if order_type != OrderTypeEnum.MARKET:
            self.logger.warning(f"Only MARKET orders currently supported. Received: {order_type}")
            return None

        # Validate inputs
        if requested_quantity <= 1e-9:
            self.logger.debug("Requested quantity too small, no execution.")
            return None

        # Precise fee and commission calculation
        # Fees: $0.003 per share
        fees = requested_quantity * self.fee_per_share

        # Commission: $0.0007 per share
        commission = requested_quantity * self.commission_per_share

        # Total transaction value
        total_transaction_value = requested_quantity * (ideal_decision_price_ask if order_side == OrderSideEnum.BUY else ideal_decision_price_bid)

        # Optional: Apply minimum commission if set
        if self.min_commission_per_order is not None:
            commission = max(commission, self.min_commission_per_order)

        # Optional: Cap commission as a percentage of trade value
        if self.max_commission_pct_of_value is not None:
            max_comm_by_value = total_transaction_value * (self.max_commission_pct_of_value / 100.0)
            commission = min(commission, max_comm_by_value)

        # Ensure non-negative values
        commission = max(0, commission)
        fees = max(0, fees)

        # Simulate execution price with slippage
        slippage_factor = self.base_slippage_bps / 10000.0
        if order_side == OrderSideEnum.BUY:
            executed_price = ideal_decision_price_ask * (1 + slippage_factor)
        else:  # SELL
            executed_price = ideal_decision_price_bid * (1 - slippage_factor)

        # Calculate slippage cost
        slippage_cost_total = abs(
            executed_price - (ideal_decision_price_ask if order_side == OrderSideEnum.BUY else ideal_decision_price_bid)) * requested_quantity

        # Detailed logging for verification
        # self.logger.info(
        #     f"Order Execution Details:\n"
        #     f"  Side       : {order_side}\n"
        #     f"  Quantity   : {requested_quantity:.2f}\n"
        #     f"  Ideal Price: ${ideal_decision_price_ask if order_side == OrderSideEnum.BUY else ideal_decision_price_bid:.2f}\n"
        #     f"  Exec Price : ${executed_price:.2f}\n"
        #     f"  Fees       : ${fees:.2f} (${self.fee_per_share:.4f} per share)\n"
        #     f"  Commission : ${commission:.2f} (${self.commission_per_share:.4f} per share)\n"
        #     f"  Slippage   : ${slippage_cost_total:.2f}"
        # )
        #
        return FillDetails(
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

    def reset(self, np_random_seed_source: Optional[np.random.Generator] = None):
        if np_random_seed_source:
            self.np_random = np_random_seed_source
        self.logger.info("RealisticExecutionSimulator reset.")