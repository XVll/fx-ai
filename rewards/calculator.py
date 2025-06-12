# rewards/calculator.py - Clean percentage-based reward system

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from config import Config
from callbacks import CallbackManager
from rewards.core import RewardState
from rewards.components import (
    PnLReward,
    HoldingTimePenalty,
    DrawdownPenalty,
    ProfitGivebackPenalty,
    MaxDrawdownPenalty,
    ProfitClosingBonus,
    CleanTradeBonus,
    BankruptcyPenalty,
    TradingActivityBonus,
    InactivityPenalty,
)
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum


@dataclass
class TradeTracker:
    """Tracks information about the current trade"""

    entry_price: float
    entry_step: int
    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0

    def update(self, unrealized_pnl: float):
        """Update MAE/MFE tracking"""
        self.max_unrealized_pnl = max(self.max_unrealized_pnl, unrealized_pnl)
        self.min_unrealized_pnl = min(self.min_unrealized_pnl, unrealized_pnl)


class RewardSystem:
    """
    Clean percentage-based reward system without complex metrics
    """

    def __init__(
        self,
        config: Config,
        callback_manager: Optional[CallbackManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config.env.reward
        self.callback_manager = callback_manager or CallbackManager()
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components based on new config
        self.components = self._initialize_components()

        # Simple tracking
        self.step_count = 0
        self.current_trade: Optional[TradeTracker] = None
        self.episode_total_reward = 0.0
        self.component_totals = {}

        # Component tracking for display
        self._last_component_rewards = {}

        self.logger.info(
            f"Initialized percentage-based reward system with {len(self.components)} components"
        )

    def _initialize_components(self) -> List:
        """Initialize reward components based on configuration"""
        components = []

        # Core P&L reward (always enabled if flag is True)
        if self.config.enable_pnl_reward:
            pnl_config = {"pnl_coefficient": self.config.pnl_coefficient}
            components.append(PnLReward(pnl_config, self.logger))
            self.logger.debug(
                f"Enabled P&L reward (coefficient: {self.config.pnl_coefficient})"
            )

        # Holding time penalty
        if self.config.enable_holding_penalty:
            holding_config = {
                "holding_penalty_coefficient": self.config.holding_penalty_coefficient,
                "max_holding_time_steps": self.config.max_holding_time_steps,
            }
            components.append(HoldingTimePenalty(holding_config, self.logger))
            self.logger.debug(
                f"Enabled holding penalty (coefficient: {self.config.holding_penalty_coefficient})"
            )

        # Drawdown penalty
        if self.config.enable_drawdown_penalty:
            drawdown_config = {
                "drawdown_penalty_coefficient": self.config.drawdown_penalty_coefficient
            }
            components.append(DrawdownPenalty(drawdown_config, self.logger))
            self.logger.debug(
                f"Enabled drawdown penalty (coefficient: {self.config.drawdown_penalty_coefficient})"
            )

        # Profit giveback penalty (MFE protection)
        if self.config.enable_profit_giveback_penalty:
            profit_giveback_config = {
                "profit_giveback_penalty_coefficient": self.config.profit_giveback_penalty_coefficient,
                "profit_giveback_threshold": self.config.profit_giveback_threshold,
            }
            components.append(
                ProfitGivebackPenalty(profit_giveback_config, self.logger)
            )
            self.logger.debug(
                f"Enabled profit giveback penalty (coefficient: {self.config.profit_giveback_penalty_coefficient})"
            )

        # Max drawdown penalty (MAE protection)
        if self.config.enable_max_drawdown_penalty:
            max_drawdown_config = {
                "max_drawdown_penalty_coefficient": self.config.max_drawdown_penalty_coefficient,
                "max_drawdown_threshold_percent": self.config.max_drawdown_threshold_percent,
            }
            components.append(MaxDrawdownPenalty(max_drawdown_config, self.logger))
            self.logger.debug(
                f"Enabled max drawdown penalty (coefficient: {self.config.max_drawdown_penalty_coefficient})"
            )

        # Profit closing bonus
        if self.config.enable_profit_closing_bonus:
            profit_closing_config = {
                "profit_closing_bonus_coefficient": self.config.profit_closing_bonus_coefficient
            }
            components.append(ProfitClosingBonus(profit_closing_config, self.logger))
            self.logger.debug(
                f"Enabled profit closing bonus (coefficient: {self.config.profit_closing_bonus_coefficient})"
            )

        # Clean trade bonus
        if self.config.enable_clean_trade_bonus:
            clean_trade_config = {
                "base_multiplier": self.config.base_multiplier,
                "max_mae_threshold": self.config.max_mae_threshold,
                "min_gain_threshold": self.config.min_gain_threshold,
            }
            components.append(CleanTradeBonus(clean_trade_config, self.logger))
            self.logger.debug(
                f"Enabled clean trade bonus (base_multiplier: {self.config.base_multiplier})"
            )

        # Trading activity bonus
        if self.config.enable_trading_activity_bonus:
            activity_config = {
                "activity_bonus_per_trade": self.config.activity_bonus_per_trade
            }
            components.append(TradingActivityBonus(activity_config, self.logger))
            self.logger.debug(
                f"Enabled trading activity bonus (bonus per trade: {self.config.activity_bonus_per_trade})"
            )

        # Inactivity penalty
        if self.config.enable_inactivity_penalty:
            inactivity_config = {
                "hold_penalty_per_step": self.config.hold_penalty_per_step
            }
            components.append(InactivityPenalty(inactivity_config, self.logger))
            self.logger.debug(
                f"Enabled inactivity penalty (penalty per HOLD step: {self.config.hold_penalty_per_step})"
            )

        # Bankruptcy penalty (always enabled - safety mechanism)
        bankruptcy_config = {
            "bankruptcy_penalty_coefficient": self.config.bankruptcy_penalty_coefficient
        }
        components.append(BankruptcyPenalty(bankruptcy_config, self.logger))
        self.logger.debug(
            f"Enabled bankruptcy penalty (coefficient: {self.config.bankruptcy_penalty_coefficient})"
        )

        return components

    def reset(self):
        """Reset for new episode"""
        self.step_count = 0
        self.current_trade = None
        self.episode_total_reward = 0.0
        self.component_totals = {}
        self._last_component_rewards = {}

        # Reset components that have state
        for component in self.components:
            if hasattr(component, "reset"):
                component.reset()

    def _update_trade_tracking(self, state: RewardState):
        """Update trade tracking for MAE/MFE"""
        position_side = state.portfolio_next.get("position_side")

        # Check if we opened a new position
        if position_side and position_side != PositionSideEnum.FLAT:
            if self.current_trade is None:
                # New trade opened
                avg_entry_price = state.portfolio_next.get("avg_entry_price", 0.0)
                self.current_trade = TradeTracker(
                    entry_price=avg_entry_price, entry_step=self.step_count
                )
        else:
            # Position closed
            if self.current_trade is not None:
                self.current_trade = None

        # Update MAE/MFE if in trade
        if self.current_trade is not None:
            unrealized_pnl = state.portfolio_next.get("unrealized_pnl", 0.0)
            self.current_trade.update(unrealized_pnl)

    def calculate(
        self,
        portfolio_state_before_action: PortfolioState,
        portfolio_state_after_action_fills: PortfolioState,
        portfolio_state_next_t: PortfolioState,
        market_state_at_decision: Dict[str, Any],
        market_state_next_t: Optional[Dict[str, Any]],
        decoded_action: Dict[str, Any],
        fill_details_list: List[FillDetails],
        terminated: bool,
        truncated: bool,
        termination_reason: Optional[Any],
    ) -> float:
        """
        Calculate total reward without complex metrics
        """

        # Create reward state
        trade_duration = 0
        if self.current_trade:
            trade_duration = self.step_count - self.current_trade.entry_step

        state = RewardState(
            portfolio_before=portfolio_state_before_action,
            portfolio_after_fills=portfolio_state_after_action_fills,
            portfolio_next=portfolio_state_next_t,
            market_state_current=market_state_at_decision,
            market_state_next=market_state_next_t,
            decoded_action=decoded_action,
            fill_details=fill_details_list,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            step_count=self.step_count,
            episode_trades=len(fill_details_list),
            current_trade_entry_price=self.current_trade.entry_price
            if self.current_trade
            else None,
            current_trade_max_unrealized_pnl=self.current_trade.max_unrealized_pnl
            if self.current_trade
            else None,
            current_trade_min_unrealized_pnl=self.current_trade.min_unrealized_pnl
            if self.current_trade
            else None,
            current_trade_duration=trade_duration,
        )

        # Calculate component rewards
        total_reward = 0.0
        component_rewards = {}

        for component in self.components:
            reward_value, diagnostics = component(state)
            total_reward += reward_value
            component_rewards[component.metadata.name] = reward_value

            # Track component totals
            if component.metadata.name not in self.component_totals:
                self.component_totals[component.metadata.name] = 0.0
            self.component_totals[component.metadata.name] += reward_value

        # Store last component rewards for access
        self._last_component_rewards = component_rewards.copy()

        # Update trade tracking
        self._update_trade_tracking(state)

        # Update episode total
        self.episode_total_reward += total_reward

        # Simplified callback - no metrics collection
        if self.callback_manager:
            self.callback_manager.trigger(
                "on_custom_event",
                "reward_calculation",
                {
                    "reward": total_reward,
                    "reward_components": component_rewards,
                    "action": str(decoded_action.get("type", "unknown")),
                    "is_invalid": bool(decoded_action.get("invalid_reason")),
                    "step_count": self.step_count,
                    "episode_total_reward": self.episode_total_reward,
                },
            )

        self.step_count += 1
        return total_reward

    def get_last_reward_components(self) -> Dict[str, float]:
        """Get the last reward components for compatibility"""
        return self._last_component_rewards.copy()

    def get_episode_summary(
        self, final_portfolio_state: PortfolioState
    ) -> Dict[str, Any]:
        """Get episode summary"""
        account_value = final_portfolio_state.get("total_equity", 25000.0)
        realized_pnl = final_portfolio_state.get("realized_pnl_session", 0.0)

        return {
            "total_reward": self.episode_total_reward,
            "component_totals": self.component_totals.copy(),
            "final_account_value": account_value,
            "realized_pnl_dollars": realized_pnl,
            "realized_pnl_percentage": (realized_pnl / 25000.0)
            * 100,  # Assuming 25k initial
            "total_steps": self.step_count,
        }

    def get_metrics_for_display(self) -> Dict[str, Any]:
        """Get current metrics formatted for display"""
        return {
            "episode_reward": self.episode_total_reward,
            "component_rewards": self._last_component_rewards.copy(),
            "component_totals": self.component_totals.copy(),
            "current_step": self.step_count,
        }