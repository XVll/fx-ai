"""
Action Masking System for V2 Trading Environment

Clean implementation of action masking with clear separation of concerns.
Prevents invalid actions and simplifies agent training by eliminating 
impossible actions from the action space.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from core.types import ActionType, PositionSizeType, MaskArray, ProbabilityArray
from config.simulation import SimulationConfig


class ActionMask:
    """
    Dynamic action masking for v2 trading environment.
    
    Design principles:
    - Clean, stateless operations
    - Clear validation logic
    - Type-safe interfaces
    - Efficient computation
    """
    
    def __init__(self, config: SimulationConfig, logger: Optional[logging.Logger] = None):
        """Initialize action mask with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract key parameters
        self.max_position_ratio = config.max_position_value_ratio  # Max position as % of equity
        self.min_order_value = config.min_trade_value  # Minimum order value in $
        self.min_shares_fraction = 0.01  # Minimum 1% of position for partial sells
        
        # Define action space structure (7 single actions)
        self.n_actions = 7  # HOLD, BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100
        
        self.logger.debug(
            f"ActionMask initialized: max_position_ratio={self.max_position_ratio:.2f}, "
            f"min_order_value=${self.min_order_value:.2f}"
        )
    
    def get_action_mask(
        self, 
        portfolio_state: Dict[str, Any], 
        market_state: Dict[str, Any]
    ) -> MaskArray:
        """
        Generate boolean mask for all possible actions.
        
        Args:
            portfolio_state: Current portfolio state containing:
                - cash: Available cash
                - total_equity: Total portfolio value
                - position_value: Current position value
                - positions: Dict of current positions
            market_state: Current market state containing:
                - current_price: Current asset price
                - best_bid_price: Current bid
                - best_ask_price: Current ask
                
        Returns:
            Boolean array of shape (12,) where True = valid action
        """
        # Initialize mask (all actions invalid by default)
        mask = np.zeros(self.n_actions, dtype=bool)
        
        # Extract portfolio information
        cash = portfolio_state.get("cash", 0.0)
        total_equity = portfolio_state.get("total_equity", cash)
        current_position_value = abs(portfolio_state.get("position_value", 0.0))
        
        # Extract position information
        current_shares = self._get_current_shares(portfolio_state)
        
        # Extract market prices
        current_price = market_state.get("current_price", 0.0)
        ask_price = market_state.get("best_ask_price", current_price)
        bid_price = market_state.get("best_bid_price", current_price)
        
        # Validate prices
        buy_price = ask_price if ask_price > 0 else current_price
        sell_price = bid_price if bid_price > 0 else current_price
        
        if buy_price <= 0 or sell_price <= 0:
            # Invalid market data - only allow HOLD action
            mask[0] = True  # HOLD is index 0
            return mask
        
        # Calculate position limits
        max_position_value = total_equity * self.max_position_ratio
        
        # Action 0: HOLD - always valid
        mask[0] = True
        
        # Actions 1-3: BUY actions (25%, 50%, 100%)
        buy_sizes = [0.25, 0.50, 1.00]
        for i, size_fraction in enumerate(buy_sizes):
            buy_value = cash * size_fraction
            new_position_value = current_position_value + buy_value
            
            # Validate buy action
            can_buy = (
                buy_value >= self.min_order_value and  # Meets minimum order size
                new_position_value <= max_position_value and  # Within position limits
                cash >= buy_value  # Have sufficient cash
            )
            
            mask[1 + i] = can_buy  # Actions 1, 2, 3
        
        # Actions 4-6: SELL actions (25%, 50%, 100%)
        if current_shares > 0 and sell_price > 0:
            sell_sizes = [0.25, 0.50, 1.00]
            for i, size_fraction in enumerate(sell_sizes):
                shares_to_sell = current_shares * size_fraction
                sell_value = shares_to_sell * sell_price
                
                # Validate sell action
                can_sell = (
                    shares_to_sell >= current_shares * self.min_shares_fraction and  # At least min fraction
                    sell_value >= self.min_order_value  # Meets minimum order value
                )
                
                mask[4 + i] = can_sell  # Actions 4, 5, 6
        
        return mask
    
    def mask_action_probabilities(
        self,
        action_probs: ProbabilityArray,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> ProbabilityArray:
        """
        Apply action mask to raw action probabilities.
        
        Args:
            action_probs: Raw action probabilities from policy network
            portfolio_state: Current portfolio state
            market_state: Current market state
            
        Returns:
            Masked and renormalized action probabilities
        """
        # Get valid action mask
        mask = self.get_action_mask(portfolio_state, market_state)
        
        # Apply mask to probabilities
        masked_probs = action_probs * mask
        
        # Renormalize probabilities
        prob_sum = masked_probs.sum()
        if prob_sum > 0:
            masked_probs = masked_probs / prob_sum
        else:
            # Emergency fallback - set HOLD action to 1.0
            masked_probs = np.zeros_like(action_probs)
            masked_probs[0] = 1.0  # HOLD action
            self.logger.warning("No valid actions found, defaulting to HOLD")
        
        return masked_probs
    
    def decode_action(self, action_idx: int) -> Tuple[ActionType, Optional[float]]:
        """
        Decode single action index into action type and position size.
        
        Args:
            action_idx: Single action index (0-6)
            
        Returns:
            Tuple of (ActionType, size_float)
        """
        if not 0 <= action_idx < self.n_actions:
            raise ValueError(f"Invalid action index: {action_idx}")
        
        # Use the same mapping as in core.types
        from core.types import single_index_to_type_size
        return single_index_to_type_size(action_idx)
    
    def encode_action(self, action_type: ActionType, size_float: Optional[float] = None) -> int:
        """
        Encode action type and position size into single action index.
        
        Args:
            action_type: Type of action
            size_float: Size as float (0.25, 0.50, 0.75) or None for HOLD
            
        Returns:
            Single action index (0-6)
        """
        # Use the same mapping as in core.types
        from core.types import type_size_to_single_index
        return type_size_to_single_index(action_type, size_float)
    
    def get_action_description(self, action_idx: int) -> str:
        """
        Get human-readable description of action.
        
        Args:
            action_idx: Single action index (0-6)
            
        Returns:
            Action description string
        """
        try:
            action_type, size_float = self.decode_action(action_idx)
            if size_float is None:
                return action_type.name  # HOLD
            else:
                size_pct = int(size_float * 100)
                return f"{action_type.name}_{size_pct}%"
        except ValueError:
            return f"INVALID_ACTION_{action_idx}"
    
    def get_valid_actions(
        self, 
        portfolio_state: Dict[str, Any], 
        market_state: Dict[str, Any]
    ) -> list[str]:
        """
        Get list of valid action descriptions for debugging.
        
        Returns:
            List of valid action strings like ["HOLD_25%", "BUY_50%", "SELL_100%"]
        """
        mask = self.get_action_mask(portfolio_state, market_state)
        valid_actions = []
        
        for idx in range(self.n_actions):
            if mask[idx]:
                valid_actions.append(self.get_action_description(idx))
        
        return valid_actions
    
    
    def _get_current_shares(self, portfolio_state: Dict[str, Any]) -> float:
        """Extract current share count from portfolio state."""
        positions = portfolio_state.get("positions", {})
        
        # Assume single asset trading - get first position
        for asset_id, position_info in positions.items():
            if position_info and position_info.get("quantity", 0) > 0:
                return position_info.get("quantity", 0)
        
        return 0.0