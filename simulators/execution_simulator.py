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

          # ——— Initialize RNG for latency/slippage randomness ———
          # if you want reproducible runs you can pass a `random_seed` in execution_config
        seed = self.config.get('random_seed', None)
          # numpy RandomState will seed from OS entropy if seed is None
        self.rng = np.random.RandomState(seed)

        # Execution tracking
        self.execution_history = []

    # In simulators/execution_simulator.py
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

        if timestamp is None:
            self.logger.warning("Received None timestamp in execute_order, using current time")
            timestamp = datetime.now()

        # Determine direction
        direction = 1 if order_type.lower() == 'buy' else -1
        order_size = abs(size)  # Ensure size is positive

        # 1. Apply latency - determine when the order will execute
        latency_ms = self._calculate_latency()
        execution_timestamp = timestamp + timedelta(milliseconds=latency_ms)

        # 2. Get market state at execution time (considering latency)
        # This now uses the unified state access method - no need for fallbacks!
        execution_market_state = self.market_simulator.get_state_at_time(execution_timestamp)

        # Extract price from state
        if not execution_market_state:
            self.logger.error(f"No market state available at {execution_timestamp}")
            return {
                'status': 'rejected',
                'reason': 'no_market_data',
                'timestamp': timestamp,
                'execution_timestamp': execution_timestamp,
                'executed_price': None,
                'executed_size': 0,
                'commission': 0
            }

        # Get price from state - try multiple sources in state but all from the same unified calculation
        base_price = None
        if 'current_price' in execution_market_state and execution_market_state['current_price']:
            base_price = execution_market_state['current_price']
        elif 'current_1s_bar' in execution_market_state and execution_market_state['current_1s_bar']:
            base_price = execution_market_state['current_1s_bar'].get('close')

        if not base_price or base_price <= 0:
            self.logger.error(f"No valid price in market state at {execution_timestamp}")
            return {
                'status': 'rejected',
                'reason': 'invalid_price',
                'timestamp': timestamp,
                'execution_timestamp': execution_timestamp,
                'executed_price': None,
                'executed_size': 0,
                'commission': 0
            }

        # 3. Calculate execution price with slippage
        slippage = self._calculate_slippage(
            base_price=base_price,
            order_size=order_size,
            order_type=order_type,
            market_state=execution_market_state
        )

        # Apply slippage to price (buys pay more, sells receive less)
        executed_price = base_price + (direction * slippage)

        # 4. Determine fill amount (might be partial)
        available_volume = 0
        if 'current_1s_bar' in execution_market_state and execution_market_state['current_1s_bar']:
            available_volume = execution_market_state['current_1s_bar'].get('volume', 0)

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
        self.logger.info(f"Executed {order_type.upper()}: {executed_size:.4f} @ {executed_price:.4f}")

        return execution_result

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

    # In simulators/execution_simulator.py
    def _get_market_state_at(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Get market state at a specific timestamp (considering latency).
        Fixed to ensure valid prices are returned.
        """
        if not self.market_simulator:
            self.logger.warning("Cannot get market state: market_simulator is not available")
            return None

        # Get the closest available market state
        market_state = self.market_simulator.get_current_market_state()

        # Ensure we have a valid price
        if market_state:
            price = None
            volume = 1000  # Default volume

            # Try multiple sources to get a valid price
            if 'current_1s_bar' in market_state and market_state['current_1s_bar']:
                bar = market_state['current_1s_bar']
                price = bar.get('close')
                volume = bar.get('volume', volume)

            if (price is None or price <= 0) and 'current_price' in market_state:
                price = market_state.get('current_price')

            # Additional fallback for price
            if price is None or price <= 0:
                if 'rolling_1s_data_window' in market_state and market_state['rolling_1s_data_window']:
                    for event in reversed(market_state['rolling_1s_data_window']):
                        if event.get('bar') and event['bar'].get('close', 0) > 0:
                            price = event['bar']['close']
                            break

            # If we still don't have a valid price, use a default value for testing
            if price is None or price <= 0:
                self.logger.warning(f"Could not get valid price at {timestamp}, using default test price of 10.0")
                price = 10.0  # Default test price

            return {
                'price': price,
                'volume': volume,
                'timestamp': timestamp
            }

        # If market state is not available
        self.logger.warning(f"Could not get market state at {timestamp}")
        return None

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