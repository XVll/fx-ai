# reward.py
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta


class RewardCalculator:
    """
    Reward calculator for trading environments, specializing in momentum/squeeze trading.

    This class calculates rewards based on:
    1. PnL changes (realized and unrealized)
    2. Trade entry/exit timing relative to momentum indicators
    3. Avoiding drawdowns and unnecessary trading
    4. Successfully predicting and avoiding market flushes
    """

    def __init__(self, config=None, logger=None):
        """
        Initialize the reward calculator.

        Args:
            config: Configuration dictionary with reward parameters
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Extract reward parameters
        self.reward_type = self.config.get('type', 'momentum')
        self.reward_scaling = self.config.get('scaling', 2.0)
        self.trade_penalty = self.config.get('trade_penalty', 0.1)
        self.hold_penalty = self.config.get('hold_penalty', 0.0)
        self.early_exit_bonus = self.config.get('early_exit_bonus', 0.5)
        self.flush_prediction_bonus = self.config.get('flush_prediction_bonus', 2.0)

        # Strategy-specific thresholds
        self.momentum_threshold = self.config.get('momentum_threshold', 0.5)
        self.volume_surge_threshold = self.config.get('volume_surge_threshold', 5.0)
        self.tape_speed_threshold = self.config.get('tape_speed_threshold', 3.0)
        self.tape_imbalance_threshold = self.config.get('tape_imbalance_threshold', 0.7)

        # State tracking
        self.previous_pnl = 0.0
        self.previous_position = 0.0
        self.previous_price = None
        self.position_entry_price = None
        self.position_entry_time = None
        self.price_history = []  # Recent price history for trend calculations

        self.logger.info(f"Reward calculator initialized with type={self.reward_type}, scaling={self.reward_scaling}")

    def reset(self):
        """Reset the reward calculator state."""
        self.previous_pnl = 0.0
        self.previous_position = 0.0
        self.previous_price = None
        self.position_entry_price = None
        self.position_entry_time = None
        self.price_history = []

    def calculate(self, market_state, portfolio_state, info=None):
        """
        Calculate the reward based on the current market and portfolio state.

        Args:
            market_state: Current market state from the simulator
            portfolio_state: Current portfolio state
            info: Additional information dictionary

        Returns:
            float: Calculated reward value
        """
        if not market_state or not portfolio_state:
            return 0.0

        info = info or {}
        action_result = info.get('action_result', {})

        # Get current values
        current_price = market_state.get('current_price', 0.0)
        current_position = portfolio_state.get('position', 0.0)
        total_pnl = portfolio_state.get('total_pnl', 0.0)
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        realized_pnl = portfolio_state.get('realized_pnl', 0.0)

        # Store current price in history for trend calculations
        if current_price > 0:
            self.price_history.append(current_price)
            # Keep only recent history (last 60 seconds)
            if len(self.price_history) > 60:
                self.price_history = self.price_history[-60:]

        # Initialize reward
        reward = 0.0

        # Calculate base reward - PnL change since last step
        pnl_change = total_pnl - self.previous_pnl
        base_reward = pnl_change * self.reward_scaling

        # Add the base reward
        reward += base_reward

        # Check if position changed
        position_changed = abs(current_position - self.previous_position) > 0.001

        # Track position entry
        if abs(self.previous_position) < 0.001 and abs(current_position) > 0.001:
            # New position opened
            self.position_entry_price = current_price
            self.position_entry_time = market_state.get('timestamp')

            # Check if entry aligns with momentum indicators
            if self._is_momentum_entry(market_state, current_position):
                # Bonus for good momentum entry
                entry_quality = self._calculate_entry_quality(market_state, current_position)
                reward += entry_quality * self.reward_scaling
            else:
                # Small penalty for entry without momentum
                reward -= self.trade_penalty

        # Check for position exit
        elif abs(self.previous_position) > 0.001 and abs(current_position) < 0.001:
            # Position closed
            if self.position_entry_price is not None:
                # Check if exit aligns with momentum indicators
                if self._is_momentum_slowdown(market_state, self.previous_position):
                    # Good exit on momentum slowdown
                    reward += self.early_exit_bonus

                # Check if exit predicted a flush
                if self._predicted_flush(market_state, self.previous_position):
                    reward += self.flush_prediction_bonus

            # Reset entry tracking
            self.position_entry_price = None
            self.position_entry_time = None

        # Apply holding penalty for positions against momentum
        if abs(current_position) > 0.001 and not self._is_momentum_aligned(market_state, current_position):
            reward -= self.hold_penalty

        # Apply trading penalty for unnecessary position changes
        if position_changed and action_result.get('status') == 'executed':
            reward -= self.trade_penalty

        # Update previous state for next calculation
        self.previous_pnl = total_pnl
        self.previous_position = current_position
        self.previous_price = current_price
        # Add action-specific rewards
        if info and 'action_result' in info:
            action_result = info['action_result']
            action_name = action_result.get('action_name', '')

            # Bonus/penalty based on action
            if 'ENTER_LONG' in action_name and self._is_momentum_entry(market_state, current_position):
                # Bonus for good entry timing
                entry_quality = self._calculate_entry_quality(market_state, current_position)
                reward += entry_quality * self.reward_scaling

            elif 'SCALE_IN' in action_name and self._is_momentum_aligned(market_state, current_position):
                # Bonus for good scale-in timing
                reward += 0.5 * self.reward_scaling

            elif 'SCALE_OUT' in action_name and not self._is_momentum_aligned(market_state, current_position):
                # Bonus for scaling out when momentum is fading
                reward += 0.5 * self.reward_scaling

            elif 'EXIT' in action_name and self._predicted_flush(market_state, current_position):
                # Bonus for exiting before a flush
                reward += self.flush_prediction_bonus

        return reward

    def _is_momentum_entry(self, market_state, position):
        """
        Check if the entry aligns with momentum indicators.

        Args:
            market_state: Current market state
            position: Current position (positive for long, negative for short)

        Returns:
            bool: True if entry aligns with momentum
        """
        # Extract momentum indicators from market state
        buffer_1s = market_state.get('buffer_1s', [])

        if not buffer_1s or len(buffer_1s) < 10:
            return False

        # Calculate price momentum (simple implementation)
        recent_prices = [item['bar']['close'] for item in buffer_1s[-10:] if 'bar' in item]
        if len(recent_prices) < 10:
            return False

        price_momentum = (recent_prices[-1] / recent_prices[0]) - 1.0

        # Calculate volume increase
        recent_volumes = [item['bar']['volume'] for item in buffer_1s[-10:] if 'bar' in item]
        if len(recent_volumes) < 10:
            return False

        avg_recent_volume = sum(recent_volumes[-3:]) / 3
        avg_prior_volume = sum(recent_volumes[:-3]) / 7

        if avg_prior_volume > 0:
            volume_increase = avg_recent_volume / avg_prior_volume
        else:
            volume_increase = 1.0

        # Check tape speed and imbalance if available
        tape_data = market_state.get('current_second_trades', [])
        tape_speed = len(tape_data)

        # Calculate tape imbalance (buy vs sell volume)
        buy_volume = sum(trade.get('size', 0) for trade in tape_data if trade.get('side', '').lower() == 'buy')
        sell_volume = sum(trade.get('size', 0) for trade in tape_data if trade.get('side', '').lower() == 'sell')

        tape_imbalance = 0.5  # Neutral default
        if buy_volume + sell_volume > 0:
            tape_imbalance = buy_volume / (buy_volume + sell_volume)

        # Check alignment with position direction
        if position > 0:  # Long position
            # For long entries, we want positive momentum, volume surge, high tape speed, buy imbalance
            return (price_momentum > self.momentum_threshold and
                    volume_increase > self.volume_surge_threshold and
                    tape_speed > self.tape_speed_threshold and
                    tape_imbalance > self.tape_imbalance_threshold)
        else:  # Short position
            # For short entries, we want negative momentum, volume surge, high tape speed, sell imbalance
            return (price_momentum < -self.momentum_threshold and
                    volume_increase > self.volume_surge_threshold and
                    tape_speed > self.tape_speed_threshold and
                    tape_imbalance < (1 - self.tape_imbalance_threshold))

    def _calculate_entry_quality(self, market_state, position):
        """
        Calculate the quality of an entry based on momentum indicators.

        Args:
            market_state: Current market state
            position: Current position

        Returns:
            float: Entry quality score (0 to 1)
        """
        # Similar to _is_momentum_entry, but returns a quality score instead of boolean
        buffer_1s = market_state.get('buffer_1s', [])

        if not buffer_1s or len(buffer_1s) < 10:
            return 0.0

        # Calculate price momentum
        recent_prices = [item['bar']['close'] for item in buffer_1s[-10:] if 'bar' in item]
        if len(recent_prices) < 10:
            return 0.0

        price_momentum = (recent_prices[-1] / recent_prices[0]) - 1.0

        # Calculate volume increase
        recent_volumes = [item['bar']['volume'] for item in buffer_1s[-10:] if 'bar' in item]
        if len(recent_volumes) < 10:
            return 0.0

        avg_recent_volume = sum(recent_volumes[-3:]) / 3
        avg_prior_volume = sum(recent_volumes[:-3]) / 7

        if avg_prior_volume > 0:
            volume_increase = min(10.0, avg_recent_volume / avg_prior_volume)  # Cap at 10x
        else:
            volume_increase = 1.0

        # Get tape data
        tape_data = market_state.get('current_second_trades', [])
        tape_speed = min(10.0, len(tape_data))  # Cap tape speed score

        # Calculate tape imbalance
        buy_volume = sum(trade.get('size', 0) for trade in tape_data if trade.get('side', '').lower() == 'buy')
        sell_volume = sum(trade.get('size', 0) for trade in tape_data if trade.get('side', '').lower() == 'sell')

        tape_imbalance = 0.5  # Neutral default
        if buy_volume + sell_volume > 0:
            tape_imbalance = buy_volume / (buy_volume + sell_volume)

        # Calculate quality score based on indicators
        if position > 0:  # Long position
            momentum_score = max(0.0, min(1.0, price_momentum / self.momentum_threshold))
            volume_score = max(0.0, min(1.0, (volume_increase - 1.0) / (self.volume_surge_threshold - 1.0)))
            tape_speed_score = max(0.0, min(1.0, tape_speed / self.tape_speed_threshold))
            imbalance_score = max(0.0, min(1.0, (tape_imbalance - 0.5) / (self.tape_imbalance_threshold - 0.5)))
        else:  # Short position
            momentum_score = max(0.0, min(1.0, -price_momentum / self.momentum_threshold))
            volume_score = max(0.0, min(1.0, (volume_increase - 1.0) / (self.volume_surge_threshold - 1.0)))
            tape_speed_score = max(0.0, min(1.0, tape_speed / self.tape_speed_threshold))
            imbalance_score = max(0.0, min(1.0, (0.5 - tape_imbalance) / (0.5 - (1 - self.tape_imbalance_threshold))))

        # Combine scores (weighted average)
        quality_score = (momentum_score * 0.3 +
                         volume_score * 0.3 +
                         tape_speed_score * 0.2 +
                         imbalance_score * 0.2)

        return quality_score

    def _is_momentum_slowdown(self, market_state, position):
        """
        Check if momentum is slowing down, indicating a good time to exit.

        Args:
            market_state: Current market state
            position: Current position

        Returns:
            bool: True if momentum is slowing down
        """
        buffer_1s = market_state.get('buffer_1s', [])

        if not buffer_1s or len(buffer_1s) < 10:
            return False

        # Calculate recent price changes
        recent_prices = [item['bar']['close'] for item in buffer_1s[-10:] if 'bar' in item]
        if len(recent_prices) < 10:
            return False

        # Calculate momentum change (comparing first and second half of window)
        first_half_change = (recent_prices[4] / recent_prices[0]) - 1.0
        second_half_change = (recent_prices[-1] / recent_prices[5]) - 1.0

        # Calculate volume change
        recent_volumes = [item['bar']['volume'] for item in buffer_1s[-10:] if 'bar' in item]
        first_half_volume = sum(recent_volumes[:5])
        second_half_volume = sum(recent_volumes[5:])

        volume_decreasing = second_half_volume < first_half_volume

        # Check for momentum slowdown based on position direction
        if position > 0:  # Long position
            # For long positions, momentum is slowing if price increase is slowing
            return second_half_change < first_half_change or volume_decreasing
        else:  # Short position
            # For short positions, momentum is slowing if price decrease is slowing
            return second_half_change > first_half_change or volume_decreasing

    def _predicted_flush(self, market_state, position):
        """
        Check if the exit predicted a market flush (sharp reversal).

        Args:
            market_state: Current market state
            position: Position that was just closed

        Returns:
            bool: True if exit appears to have predicted a flush
        """
        buffer_1s = market_state.get('buffer_1s', [])

        if not buffer_1s or len(buffer_1s) < 5:
            return False

        # Get recent prices
        recent_prices = [item['bar']['close'] for item in buffer_1s[-5:] if 'bar' in item]
        if len(recent_prices) < 5:
            return False

        # Get last price
        current_price = recent_prices[-1]

        # Get recent quotes to check for price pressure
        quotes = market_state.get('current_second_quotes', [])

        # Calculate bid/ask imbalance to detect potential flush
        if quotes:
            total_bid_size = sum(quote.get('bid_size', 0) for quote in quotes)
            total_ask_size = sum(quote.get('ask_size', 0) for quote in quotes)

            if total_bid_size + total_ask_size > 0:
                bid_ask_ratio = total_bid_size / (total_bid_size + total_ask_size)
            else:
                bid_ask_ratio = 0.5
        else:
            bid_ask_ratio = 0.5

        # Check for potential flush based on position direction
        if position > 0:  # Was long, now flat
            # Potential downward flush: price starting to drop, heavy selling pressure
            price_dropping = recent_prices[-1] < recent_prices[-2]
            selling_pressure = bid_ask_ratio < 0.3  # Significantly more asking than bidding
            return price_dropping and selling_pressure
        else:  # Was short, now flat
            # Potential upward flush: price starting to rise, heavy buying pressure
            price_rising = recent_prices[-1] > recent_prices[-2]
            buying_pressure = bid_ask_ratio > 0.7  # Significantly more bidding than asking
            return price_rising and buying_pressure

    def _is_momentum_aligned(self, market_state, position):
        """
        Check if the current position is aligned with market momentum.

        Args:
            market_state: Current market state
            position: Current position

        Returns:
            bool: True if position is aligned with momentum
        """
        buffer_1s = market_state.get('buffer_1s', [])

        if not buffer_1s or len(buffer_1s) < 10:
            return True  # Default to aligned if not enough data

        # Calculate price momentum
        recent_prices = [item['bar']['close'] for item in buffer_1s[-10:] if 'bar' in item]
        if len(recent_prices) < 10:
            return True

        price_momentum = (recent_prices[-1] / recent_prices[0]) - 1.0

        # Check alignment
        if position > 0:  # Long position
            return price_momentum >= 0  # Aligned if momentum is positive
        else:  # Short position
            return price_momentum <= 0  # Aligned if momentum is negative