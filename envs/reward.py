# reward.py

import logging
from typing import Any, Dict, List, Optional

# --- Use canonical imports for Config and PortfolioState ---
from config.config import Config  # Assuming this is your main Config definition
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum

# --- Remove or comment out local placeholder TypedDicts for Config and PortfolioState ---
# class RewardConfig(TypedDict, total=False): ... # This can stay if it's specific to reward_config
# class Config(TypedDict, total=False): ... # REMOVE THIS PLACEHOLDER
# class PortfolioState(TypedDict, total=False): ... # REMOVE THIS PLACEHOLDER
# class FillDetails(TypedDict, total=False): ... # REMOVE THIS PLACEHOLDER (if it was also a placeholder)


# Attempt to import Enums from their likely locations.
# These try-except blocks are generally fine for enums if they might be missing in some contexts.
try:
    # If ActionTypeEnumForEnv and TerminationReasonEnumForEnv are defined in trading_env.py
    # and reward.py is in a submodule like envs/reward.py, you might need:
    # from .trading_env import ActionTypeEnumForEnv, TerminationReasonEnumForEnv
    # Or adjust based on your project structure. For now, assume direct or placeholder.
    from envs.trading_env import ActionTypeEnum  # Assuming ActionTypeEnum is the one used by decoded_action
    from envs.trading_env import TerminationReasonEnum as TerminationReasonEnumForEnv  # Alias if names differ
except ImportError:
    from enum import Enum


    class ActionTypeEnum(Enum):  # Placeholder if direct import fails
        HOLD = 0
        # ... other actions from your trading_env.ActionTypeEnum ...


    class TerminationReasonEnumForEnv(Enum):  # Placeholder
        BANKRUPTCY = "BANKRUPTCY"
        MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
        # ... other reasons ...

logger_reward = logging.getLogger(__name__)  # Use a specific logger for this module


class RewardCalculator:
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):  # Expects the main Config
        self.config_main = config  # Store the main config
        self.reward_config = config.env.reward  # Access reward_config from the main Config

        self.logger = logger or logger_reward  # Use passed logger or module logger

        self.weight_equity_change = self.reward_config.weight_equity_change
        self.weight_realized_pnl = self.reward_config.weight_realized_pnl
        self.penalty_transaction_fill = self.reward_config.penalty_transaction_fill
        self.penalty_holding_inaction = self.reward_config.penalty_holding_inaction
        self.penalty_drawdown_step = self.reward_config.penalty_drawdown_step
        self.penalty_invalid_action = self.reward_config.penalty_invalid_action
        self.terminal_penalty_bankruptcy = self.reward_config.terminal_penalty_bankruptcy
        self.terminal_penalty_max_loss = self.reward_config.terminal_penalty_max_loss
        self.reward_scaling_factor = self.reward_config.reward_scaling_factor
        self.log_reward_components = self.reward_config.log_reward_components

        # Fallback for initial capital if not directly in portfolio_state for some reason
        self.initial_capital_fallback = self.config_main.simulation.portfolio_config.initial_cash

        self.logger.info(f"RewardCalculator initialized with reward_specific_config: {self.reward_config}")

    def reset(self):
        self.logger.debug("RewardCalculator reset.")
        pass

    def calculate(self,
                  portfolio_state_before_action: PortfolioState,  # Uses imported PortfolioState
                  portfolio_state_after_action_fills: PortfolioState,  # Uses imported PortfolioState
                  portfolio_state_next_t: PortfolioState,  # Uses imported PortfolioState
                  market_state_at_decision: Dict[str, Any],
                  market_state_next_t: Optional[Dict[str, Any]],
                  decoded_action: Dict[str, Any],  # Should contain ActionTypeEnum from trading_env
                  fill_details_list: List[FillDetails],  # Uses imported FillDetails
                  terminated: bool,
                  truncated: bool,
                  termination_reason: Optional[TerminationReasonEnumForEnv]  # Uses aliased/imported enum
                  ) -> float:

        reward_components = {}
        action_type_from_env = decoded_action.get('type')  # This should be envs.trading_env.ActionTypeEnum

        # 1. Primary Reward: Change in Total Equity
        equity_before = portfolio_state_before_action['total_equity']
        equity_after_market_move = portfolio_state_next_t['total_equity']
        equity_change = equity_after_market_move - equity_before
        reward_components['equity_change_reward'] = self.weight_equity_change * equity_change

        # 2. Optional: Explicit reward for Realized PnL
        if self.weight_realized_pnl > 0:
            realized_pnl_before = portfolio_state_before_action['realized_pnl_session']
            realized_pnl_after_fills = portfolio_state_after_action_fills['realized_pnl_session']
            step_realized_pnl = realized_pnl_after_fills - realized_pnl_before
            reward_components['realized_pnl_bonus'] = self.weight_realized_pnl * step_realized_pnl

        # 3. Penalty for Transaction Fills
        if self.penalty_transaction_fill > 0 and fill_details_list:
            num_fills = len(fill_details_list)
            reward_components['transaction_fill_penalty'] = -self.penalty_transaction_fill * num_fills

        # 4. Penalty for Holding/Inaction
        # Ensure ActionTypeEnum.HOLD is the correct enum instance from trading_env
        if self.penalty_holding_inaction > 0 and action_type_from_env == ActionTypeEnum.HOLD:
            has_open_position = False
            # Ensure portfolio_state_before_action['positions'] exists and is a dict
            positions_before = portfolio_state_before_action.get('positions', {})
            if positions_before:  # Check if dict is not empty
                for asset_id, pos_data in positions_before.items():
                    # Ensure pos_data is a dict and has 'current_side'
                    if isinstance(pos_data, dict) and pos_data.get('current_side') != PositionSideEnum.FLAT:
                        has_open_position = True
                        break
            if has_open_position:
                reward_components['holding_inaction_penalty'] = -self.penalty_holding_inaction

        # 5. Step-wise Drawdown Penalty
        if self.penalty_drawdown_step > 0 > equity_change:  # Check equity_change < 0
            reward_components['drawdown_step_penalty'] = self.penalty_drawdown_step * equity_change

        # 6. Penalty for Invalid Actions
        if self.penalty_invalid_action > 0 and decoded_action.get('invalid_reason'):
            reward_components['invalid_action_penalty'] = -self.penalty_invalid_action

        # 7. Terminal Rewards/Penalties
        if terminated:
            if termination_reason == TerminationReasonEnumForEnv.BANKRUPTCY:
                reward_components['terminal_bankruptcy_penalty'] = -self.terminal_penalty_bankruptcy
            elif termination_reason == TerminationReasonEnumForEnv.MAX_LOSS_REACHED:
                reward_components['terminal_max_loss_penalty'] = -self.terminal_penalty_max_loss

        total_reward = sum(reward_components.values())
        total_reward *= self.reward_scaling_factor

        if self.log_reward_components:
            # Ensure all components are serializable for logging if using structured logging
            loggable_components = {k: (v.value if isinstance(v, Enum) else v) for k, v in reward_components.items()}
            self.logger.debug(
                f"Step Reward Calculation: Components: {loggable_components}, Total Raw: {sum(reward_components.values()):.4f}, Scaled Total: {total_reward:.4f}")

        return total_reward