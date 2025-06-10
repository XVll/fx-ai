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
        
        # Define action space structure (3 types × 4 sizes = 12 actions)
        self.n_action_types = len(ActionType)
        self.n_position_sizes = len(PositionSizeType)
        self.n_actions = self.n_action_types * self.n_position_sizes
        
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
            # Invalid market data - only allow HOLD actions
            mask[self._get_action_indices(ActionType.HOLD)] = True
            return mask
        
        # Calculate position limits
        max_position_value = total_equity * self.max_position_ratio
        
        # HOLD actions - always valid
        hold_indices = self._get_action_indices(ActionType.HOLD)
        mask[hold_indices] = True
        
        # BUY actions - check cash and position limits
        buy_indices = self._get_action_indices(ActionType.BUY)
        for i, size_type in enumerate(PositionSizeType):
            size_fraction = self._get_size_fraction(size_type)
            buy_value = cash * size_fraction
            new_position_value = current_position_value + buy_value
            
            # Validate buy action
            can_buy = (
                buy_value >= self.min_order_value and  # Meets minimum order size
                new_position_value <= max_position_value and  # Within position limits
                cash >= buy_value  # Have sufficient cash
            )
            
            mask[buy_indices[i]] = can_buy
        
        # SELL actions - check if we have position to sell
        if current_shares > 0 and sell_price > 0:
            sell_indices = self._get_action_indices(ActionType.SELL)
            for i, size_type in enumerate(PositionSizeType):
                size_fraction = self._get_size_fraction(size_type)
                shares_to_sell = current_shares * size_fraction
                sell_value = shares_to_sell * sell_price
                
                # Validate sell action
                can_sell = (
                    shares_to_sell >= current_shares * self.min_shares_fraction and  # At least min fraction
                    sell_value >= self.min_order_value  # Meets minimum order value
                )
                
                mask[sell_indices[i]] = can_sell
        
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
            # Emergency fallback - uniform distribution over HOLD actions
            hold_indices = self._get_action_indices(ActionType.HOLD)
            masked_probs = np.zeros_like(action_probs)
            masked_probs[hold_indices] = 1.0 / len(hold_indices)
            self.logger.warning("No valid actions found, defaulting to HOLD")
        
        return masked_probs
    
    def decode_action(self, action_idx: int) -> Tuple[ActionType, PositionSizeType]:
        """
        Decode linear action index into action type and position size.
        
        Args:
            action_idx: Linear action index (0-11)
            
        Returns:
            Tuple of (ActionType, PositionSizeType)
        """
        if not 0 <= action_idx < self.n_actions:
            raise ValueError(f"Invalid action index: {action_idx}")
        
        action_type_idx = action_idx // self.n_position_sizes
        size_idx = action_idx % self.n_position_sizes
        
        action_type = list(ActionType)[action_type_idx]
        position_size = list(PositionSizeType)[size_idx]
        
        return action_type, position_size
    
    def encode_action(self, action_type: ActionType, position_size: PositionSizeType) -> int:
        """
        Encode action type and position size into linear index.
        
        Args:
            action_type: Type of action
            position_size: Size of position
            
        Returns:
            Linear action index (0-11)
        """
        action_type_idx = list(ActionType).index(action_type)
        size_idx = list(PositionSizeType).index(position_size)
        
        return action_type_idx * self.n_position_sizes + size_idx
    
    def get_action_description(self, action_idx: int) -> str:
        """
        Get human-readable description of action.
        
        Args:
            action_idx: Linear action index
            
        Returns:
            Action description string
        """
        try:
            action_type, position_size = self.decode_action(action_idx)
            size_pct = int(self._get_size_fraction(position_size) * 100)
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
    
    def _get_action_indices(self, action_type: ActionType) -> NDArray[np.int32]:
        """Get array of action indices for a specific action type."""
        action_type_idx = list(ActionType).index(action_type)
        start_idx = action_type_idx * self.n_position_sizes
        return np.arange(start_idx, start_idx + self.n_position_sizes, dtype=np.int32)
    
    def _get_size_fraction(self, position_size: PositionSizeType) -> float:
        """Convert position size enum to fraction."""
        # Map SIZE_25 -> 0.25, SIZE_50 -> 0.50, etc.
        return (position_size.value + 1) * 0.25
    
    def _get_current_shares(self, portfolio_state: Dict[str, Any]) -> float:
        """Extract current share count from portfolio state."""
        positions = portfolio_state.get("positions", {})
        
        # Assume single asset trading - get first position
        for asset_id, position_info in positions.items():
            if position_info and position_info.get("quantity", 0) > 0:
                return position_info.get("quantity", 0)
        
        return 0.0