# execution_simulator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import random


class ExecutionSimulator:
    """
    Simulator for realistic order execution including latency, slippage, and partial fills.
    """

    def __init__(self, market_simulator, config=None, logger=None):
        """
        Initialize the execution simulator.

        Args:
            market_simulator: MarketSimulator instance
            config: Configuration dictionary for execution parameters
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.market_simulator = market_simulator
        self.config = config or {}

        # Execution parameters
        self.base_latency_ms = self.config.get('base_latency_ms', 100)
        self.latency_variation_ms = self.config.get('latency_variation_ms', 50)
        self.slippage_factor = self.config.get('slippage_factor', 0.001)
        self.volatility_slippage_multiplier = self.config.get('volatility_slippage_multiplier', 5.0)
        self.volume_slippage_factor = self.config.get('volume_slippage_factor', 0.1)
        self.min_execution_rate = self.config.get('min_execution_rate', 0.5)
        self.commission_per_share = self.config.get('commission_per_share', 0.0)

        # Execution tracking
        self.execution_history = []

    def execute_order(self, order_type: str, size: float, timestamp: datetime) -> Dict[str, Any]:
        """
        Execute an order with realistic constraints.

        Args:
            order_type: 'buy' or 'sell'
            size: Order size (positive number)
            timestamp: Timestamp when order was placed

        Returns:
            Dict with execution details
        """
        # Validate inputs
        if size == 0:
            return {
                'status': 'rejected',
                'reason': 'zero_size',
                'timestamp': timestamp,
                'executed_price': None,
                'executed_size': 0,
                'commission': 0
            }

        # Determine direction
        direction = 1 if order_type.lower() == 'buy' else -1
        order_size = abs(size)  # Ensure size is positive

        # 1. Apply latency - determine when the order will execute
        latency_ms = self._calculate_latency()
        execution_timestamp = timestamp + timedelta(milliseconds=latency_ms)

        # 2. Get market state at execution time (considering latency)
        execution_market_state = self._get_market_state_at(execution_timestamp)

        # If we couldn't get a price (e.g., market closed), reject the order
        if not execution_market_state or 'price' not in execution_market_state:
            return {
                'status': 'rejected',
                'reason': 'no_market_data',
                'timestamp': timestamp,
                'execution_timestamp': execution_timestamp,
                'latency_ms': latency_ms,
                'executed_price': None,
                'executed_size': 0,
                'commission': 0
            }

        # 3. Calculate execution price with slippage
        base_price = execution_market_state['price']
        slippage = self._calculate_slippage(
            base_price=base_price,
            order_size=order_size,
            order_type=order_type,
            market_state=execution_market_state
        )

        # Apply slippage to price (buys pay more, sells receive less)
        executed_price = base_price + (direction * slippage)

        # 4. Determine fill amount (might be partial)
        available_volume = execution_market_state.get('volume', float('inf'))
        if available_volume < order_size:
            # Partial fill based on available volume and minimum execution rate
            executed_size = max(
                self.min_execution_rate * order_size,
                min(order_size, available_volume)
            )
        else:
            executed_size = order_size

        # 5. Calculate commission
        commission = self.commission_per_share * executed_size

        # 6. Create execution result
        execution_result = {
            'status': 'executed',
            'order_type': order_type,
            'timestamp': timestamp,
            'execution_timestamp': execution_timestamp,
            'latency_ms': latency_ms,
            'base_price': base_price,
            'slippage': slippage,
            'executed_price': executed_price,
            'requested_size': order_size,
            'executed_size': executed_size * direction,  # Apply direction to size
            'available_volume': available_volume,
            'commission': commission
        }

        # 7. Log the execution
        self.execution_history.append(execution_result)
        self.logger.debug(f"Order executed: {execution_result}")

        return execution_result

    def _calculate_latency(self) -> float:
        """
        Calculate realistic latency for order execution.
        """
        # Same implementation as before

    def _get_market_state_at(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Get market state at a specific timestamp (considering latency).
        """
        # Same implementation as before

    def _calculate_slippage(self, base_price: float, order_size: float,
                            order_type: str, market_state: Dict[str, Any]) -> float:
        """
        Calculate realistic slippage based on market conditions and order size.
        """
        # Same implementation as before

    def reset(self):
        """Reset the execution simulator."""
        self.execution_history.clear()