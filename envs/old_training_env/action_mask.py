"""
Action Masking System for Trading Environment

This module implements dynamic action masking to prevent invalid actions
and eliminate model exploitation via invalid action spam.

Key Features:
- BUY actions: Based on available cash and position limits
- SELL actions: Based on current position size
- HOLD actions: Always valid
- Prevents over-leveraging and impossible trades
"""

import logging
from typing import Dict, Any, Optional
import numpy as np


class ActionMask:
    """
    Dynamic action masking for trading environment.

    Determines valid actions based on current portfolio state and market conditions.
    Prevents model exploitation by eliminating invalid actions as an escape mechanism.
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Configuration parameters
        self.max_position_ratio = (
            config.simulation.max_position_value_ratio
        )  # Max position as % of equity
        self.min_order_value = 25.0  # Minimum $25 order (more flexible)
        self.min_shares_to_sell = (
            0.1  # Minimum 0.1 shares (allow fractional for percentages)
        )

        # Action space configuration (matches environment)
        self.action_types = ["HOLD", "BUY", "SELL"]  # indices 0, 1, 2
        self.position_sizes = [0.25, 0.50, 0.75, 1.0]  # indices 0, 1, 2, 3

        self.logger.info(
            f"Action masking initialized - max_position_ratio: {self.max_position_ratio}"
        )

    def get_action_mask(
        self, portfolio_state: Dict[str, Any], market_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate boolean mask for all 12 actions (3 types × 4 sizes).

        Args:
            portfolio_state: Current portfolio state (cash, position, etc.)
            market_state: Current market state (price, etc.)

        Returns:
            Boolean array of shape (12,) where True = valid action
        """
        mask = np.zeros(12, dtype=bool)

        # Extract portfolio info
        cash = portfolio_state.get("cash", 0.0)
        total_equity = portfolio_state.get("total_equity", cash)
        current_position_value = abs(portfolio_state.get("position_value", 0.0))

        # Get shares from positions (primary asset)
        current_shares = 0
        positions = portfolio_state.get("positions", {})
        for asset_id, position_info in positions.items():
            if position_info and position_info.get("quantity", 0) > 0:
                current_shares = position_info.get("quantity", 0)
                break  # Assume single asset trading

        # Extract market info
        current_price = market_state.get("current_price", 0.0)
        ask_price = market_state.get("best_ask_price", current_price)
        bid_price = market_state.get("best_bid_price", current_price)

        # Use ask/bid or fallback to current price
        buy_price = ask_price if ask_price and ask_price > 0 else current_price
        sell_price = bid_price if bid_price and bid_price > 0 else current_price

        if buy_price <= 0 or sell_price <= 0:
            # Invalid prices - only allow HOLD
            mask[0:4] = True  # HOLD actions always valid
            return mask

        # Calculate maximum allowed position value
        max_position_value = total_equity * self.max_position_ratio

        # HOLD actions (indices 0-3) - always valid
        mask[0:4] = True

        # BUY actions (indices 4-7) - check cash and position limits
        for i, size_pct in enumerate(self.position_sizes):
            buy_value = cash * size_pct
            new_total_position = current_position_value + buy_value

            # Check constraints
            can_buy = (
                buy_value >= self.min_order_value  # Minimum order size
                and new_total_position <= max_position_value  # Position limit
                and cash >= buy_value  # Sufficient cash
            )
            mask[4 + i] = can_buy

        # SELL actions (indices 8-11) - check if position exists
        if current_shares > 0 and sell_price > 0:
            for i, size_pct in enumerate(self.position_sizes):
                shares_to_sell = current_shares * size_pct
                sell_value = shares_to_sell * sell_price

                # Check constraints
                can_sell = (
                    shares_to_sell >= self.min_shares_to_sell  # At least 1 share
                    and sell_value >= self.min_order_value  # Minimum order value
                )
                mask[8 + i] = can_sell
        else:
            # No position or invalid price - can't sell
            mask[8:12] = False

        # Log mask for debugging (only if enabled)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log_mask_debug(mask, portfolio_state, market_state)

        return mask

    def _log_mask_debug(
        self,
        mask: np.ndarray,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ):
        """Log detailed mask information for debugging."""
        cash = portfolio_state.get("cash", 0.0)
        position_value = portfolio_state.get("position_value", 0.0)
        price = market_state.get("current_price", 0.0)

        # Get shares from positions
        shares = 0
        positions = portfolio_state.get("positions", {})
        for asset_id, position_info in positions.items():
            if position_info and position_info.get("quantity", 0) > 0:
                shares = position_info.get("quantity", 0)
                break

        valid_actions = []
        for i, is_valid in enumerate(mask):
            if is_valid:
                action_type_idx = i // 4
                size_idx = i % 4
                action_type = self.action_types[action_type_idx]
                size_pct = self.position_sizes[size_idx]
                valid_actions.append(f"{action_type}_{int(size_pct * 100)}%")

        self.logger.debug(
            f"Action mask - Cash: ${cash:.0f}, Shares: {shares}, "
            f"Position: ${position_value:.0f}, Price: ${price:.2f}, "
            f"Valid: {valid_actions}"
        )

    def mask_action_probabilities(
        self,
        action_probs: np.ndarray,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> np.ndarray:
        """
        Apply action mask to action probabilities by setting invalid actions to 0.

        Args:
            action_probs: Raw action probabilities from policy network
            portfolio_state: Current portfolio state
            market_state: Current market state

        Returns:
            Masked and renormalized action probabilities
        """
        mask = self.get_action_mask(portfolio_state, market_state)

        # Apply mask
        masked_probs = action_probs * mask

        # Renormalize to ensure probabilities sum to 1
        prob_sum = masked_probs.sum()
        if prob_sum > 0:
            masked_probs = masked_probs / prob_sum
        else:
            # Fallback to uniform over valid actions
            valid_count = mask.sum()
            if valid_count > 0:
                masked_probs = mask.astype(float) / valid_count
            else:
                # Emergency fallback - allow all HOLD actions
                masked_probs = np.zeros_like(action_probs)
                masked_probs[0:4] = 0.25  # Equal probability for all HOLD sizes

        return masked_probs

    def is_action_valid(
        self,
        action_idx: int,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> bool:
        """
        Check if a specific action index is valid.

        Args:
            action_idx: Linear action index (0-11)
            portfolio_state: Current portfolio state
            market_state: Current market state

        Returns:
            True if action is valid, False otherwise
        """
        mask = self.get_action_mask(portfolio_state, market_state)
        return bool(mask[action_idx]) if 0 <= action_idx < len(mask) else False

    def get_valid_actions(
        self, portfolio_state: Dict[str, Any], market_state: Dict[str, Any]
    ) -> list:
        """
        Get list of valid action descriptions for debugging/logging.

        Returns:
            List of valid action strings like ["HOLD_25%", "BUY_50%", "SELL_100%"]
        """
        mask = self.get_action_mask(portfolio_state, market_state)
        valid_actions = []

        for i, is_valid in enumerate(mask):
            if is_valid:
                action_type_idx = i // 4
                size_idx = i % 4
                action_type = self.action_types[action_type_idx]
                size_pct = int(self.position_sizes[size_idx] * 100)
                valid_actions.append(f"{action_type}_{size_pct}%")

        return valid_actions

    def get_action_description(self, action_idx: int) -> str:
        """Get human-readable description of action index."""
        if not (0 <= action_idx < 12):
            return f"INVALID_ACTION_{action_idx}"

        action_type_idx = action_idx // 4
        size_idx = action_idx % 4
        action_type = self.action_types[action_type_idx]
        size_pct = int(self.position_sizes[size_idx] * 100)

        return f"{action_type}_{size_pct}%"