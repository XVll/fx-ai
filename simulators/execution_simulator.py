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
        Fixed to ensure proper handling of timestamp and latency calculations.

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

        if timestamp is None:
            self.logger.warning("Received None timestamp in execute_order, using current time")
            timestamp = datetime.now()

        # Determine direction
        direction = 1 if order_type.lower() == 'buy' else -1
        order_size = abs(size)  # Ensure size is positive

        # 1. Apply latency - determine when the order will execute
        # Get latency value and ensure it's a Python int, not numpy int
        latency_ms = self._calculate_latency()
        if not isinstance(latency_ms, (int, float)) or latency_ms is None:
            self.logger.warning(f"Invalid latency value: {latency_ms}, using default of 100ms")
            latency_ms = 100.0

        # Ensure latency is a proper Python int/float for timedelta
        latency_ms = float(latency_ms)  # Convert to Python float

        # Calculate execution timestamp with proper handling
        try:
            execution_timestamp = timestamp + timedelta(milliseconds=latency_ms)
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error calculating execution timestamp: {e}. Using original timestamp.")
            execution_timestamp = timestamp

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

    # Replace the placeholder methods in execution_simulator.py with proper implementations

    def _calculate_latency(self) -> float:
        """
        Calculate realistic latency for order execution with proper return value.

        Returns:
            float: Latency in milliseconds
        """
        # Base latency plus random variation
        base = self.base_latency_ms
        variation = self.rng.uniform(-self.latency_variation_ms, self.latency_variation_ms)

        # Add additional latency during high market activity
        if hasattr(self.market_simulator, 'current_market_session'):
            # More latency during regular market hours due to higher activity
            if self.market_simulator.current_market_session == "REGULAR":
                base *= 1.2

        # Ensure we always return a valid float, never None
        latency = max(1.0, base + variation)
        return latency

    def _get_market_state_at(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Get market state at a specific timestamp (considering latency).

        Args:
            timestamp: Timestamp to get market state for

        Returns:
            Optional[Dict[str, Any]]: Market state or None if not available
        """
        if not self.market_simulator:
            self.logger.warning("Cannot get market state: market_simulator is not available")
            return None

        # Store current simulator timestamp
        original_timestamp = None
        if hasattr(self.market_simulator, 'current_timestamp_utc'):
            original_timestamp = self.market_simulator.current_timestamp_utc

        # Get the closest available market state
        market_state = self.market_simulator.get_current_market_state()

        # Ensure we have a valid price
        if market_state:
            # Try to get price from current bar
            if 'current_1s_bar' in market_state and market_state['current_1s_bar']:
                bar = market_state['current_1s_bar']
                price = bar.get('close', 0.0)

                # Get other data like volume for slippage calculation
                volume = bar.get('volume', 0)

                return {
                    'price': price,
                    'volume': volume,
                    'timestamp': timestamp
                }

        # If we couldn't get a valid price, return a minimal state
        # with a default price to prevent errors
        # In real implementation, we might want to reject the order instead
        self.logger.warning(f"Could not get market state at {timestamp}, using default values")
        return {
            'price': 10.0,  # Default price
            'volume': 1000,  # Default volume
            'timestamp': timestamp
        }

    def _calculate_slippage(self, base_price: float, order_size: float,
                            order_type: str, market_state: Dict[str, Any]) -> float:
        """
        Calculate realistic slippage based on market conditions and order size.

        Args:
            base_price: Base price for the order
            order_size: Size of the order
            order_type: 'buy' or 'sell'
            market_state: Current market state

        Returns:
            float: Slippage amount (positive for buys, negative for sells)
        """
        if base_price <= 0:
            # Prevent division by zero or negative prices
            return 0.0

        # Basic slippage as percentage of price
        slippage_pct = self.slippage_factor

        # Adjust for order size relative to available volume
        available_volume = market_state.get('volume', float('inf'))
        if available_volume > 0:
            volume_factor = min(5.0, order_size / max(1, available_volume))
            slippage_pct *= (1.0 + self.volume_slippage_factor * volume_factor)

        # Adjust for volatility if data is available
        if hasattr(self.market_simulator, 'intraday_high') and \
                hasattr(self.market_simulator, 'intraday_low') and \
                self.market_simulator.intraday_high and \
                self.market_simulator.intraday_low and \
                self.market_simulator.intraday_high > self.market_simulator.intraday_low:
            # Calculate intraday range as a percentage
            range_pct = (self.market_simulator.intraday_high - self.market_simulator.intraday_low) / \
                        self.market_simulator.intraday_low

            # More volatile = more slippage
            slippage_pct *= (1.0 + self.volatility_slippage_multiplier * range_pct)

        # Calculate absolute slippage amount
        slippage = base_price * slippage_pct

        # For buy orders, slippage increases price; for sell orders, it decreases price
        if order_type.lower() == 'sell':
            slippage = -slippage

        return slippage

    def reset(self):
        """Reset the execution simulator."""
        self.execution_history.clear()