import logging
from typing import Any, Dict, List, Optional

# Assuming Config is defined in a way that it can be imported, e.g., from a root config module
# from config.config import Config  # Adjust if your Config structure is different
# For demonstration, we'll use a TypedDict placeholder if direct import isn't set up for this snippet
from typing import TypedDict


class RewardConfig(TypedDict, total=False):
    weight_equity_change: float
    weight_realized_pnl: float
    penalty_transaction_fill: float  # Penalty per fill
    penalty_holding_inaction: float
    penalty_drawdown_step: float  # Penalty factor for negative equity change in a step
    penalty_invalid_action: float
    terminal_penalty_bankruptcy: float
    terminal_penalty_max_loss: float
    reward_scaling_factor: float
    log_reward_components: bool


class Config(TypedDict, total=False):  # Placeholder for the main Config structure
    env: Dict[str, Any]
    simulation: Dict[str, Any]
    model: Dict[str, Any]
    reward_config: RewardConfig


# Attempt to import Enums from their likely locations.
# If these are defined elsewhere (e.g., a common types.py), adjust the import path.
try:
    from envs.trading_env import ActionTypeEnumForEnv, TerminationReasonEnumForEnv
except ImportError:
    # Placeholder Enums if direct import fails (e.g., during standalone generation)
    # In a real project, ensure these are correctly imported from their definition location.
    from enum import Enum


    class ActionTypeEnumForEnv(Enum):
        HOLD = 0
        # ... other actions ...


    class TerminationReasonEnumForEnv(Enum):
        BANKRUPTCY = "BANKRUPTCY"
        MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
        # ... other reasons ...

try:
    from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum
except ImportError:
    # Placeholder TypedDicts if direct import fails
    class PortfolioState(TypedDict, total=False):
        total_equity: float
        realized_pnl_session: float
        initial_capital: float  # Assuming this might be part of state or config accessible
        positions: Dict[str, Dict[str, Any]]
        # ... other fields used by reward calculator or environment ...


    class FillDetails(TypedDict, total=False):
        commission: float
        fees: float
        slippage_cost_total: float
        # ... other fields ...


    class PositionSideEnum(Enum):  # From portfolio_simulator.py
        FLAT = "FLAT"
        LONG = "LONG"
        SHORT = "SHORT"

logger = logging.getLogger(__name__)


class RewardCalculator:
    def __init__(self, config: Config):
        self.config = config
        self.reward_config: RewardConfig = config.get('reward_config', {})

        # Default values can be set here or fetched from a more detailed default config
        self.weight_equity_change = self.reward_config.get('weight_equity_change', 1.0)
        self.weight_realized_pnl = self.reward_config.get('weight_realized_pnl', 0.0)
        self.penalty_transaction_fill = self.reward_config.get('penalty_transaction_fill', 0.01)  # e.g., small penalty per fill
        self.penalty_holding_inaction = self.reward_config.get('penalty_holding_inaction', 0.0001)
        self.penalty_drawdown_step = self.reward_config.get('penalty_drawdown_step', 0.5)  # Applied to negative equity change
        self.penalty_invalid_action = self.reward_config.get('penalty_invalid_action', 0.1)
        self.terminal_penalty_bankruptcy = self.reward_config.get('terminal_penalty_bankruptcy', 100.0)
        self.terminal_penalty_max_loss = self.reward_config.get('terminal_penalty_max_loss', 50.0)
        self.reward_scaling_factor = self.reward_config.get('reward_scaling_factor', 1.0)  # Normalize rewards if needed
        self.log_reward_components = self.reward_config.get('log_reward_components', False)

        self.initial_capital_fallback = config.get('simulation', {}).get('portfolio_config', {}).get('initial_cash', 100000.0)

        logger.info(f"RewardCalculator initialized with config: {self.reward_config}")

    def reset(self):
        # Reset any stateful parts of the reward calculator if necessary (e.g., tracking episode HWM)
        # For now, most rewards are stateless per step or use initial capital.
        logger.debug("RewardCalculator reset.")
        pass

    def calculate(self,
                  portfolio_state_before_action: PortfolioState,
                  portfolio_state_after_action_fills: PortfolioState,
                  portfolio_state_next_t: PortfolioState,
                  market_state_at_decision: Dict[str, Any],
                  market_state_next_t: Optional[Dict[str, Any]],
                  decoded_action: Dict[str, Any],
                  fill_details_list: List[FillDetails],
                  terminated: bool,
                  truncated: bool,
                  termination_reason: Optional[TerminationReasonEnumForEnv]
                  ) -> float:

        reward_components = {}

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

        # 3. Penalty for Transaction Fills (discourage over-trading if not offset by profit)
        if self.penalty_transaction_fill > 0 and fill_details_list:
            num_fills = len(fill_details_list)
            reward_components['transaction_fill_penalty'] = -self.penalty_transaction_fill * num_fills
            # Alternative: Penalize based on actual costs
            # total_costs = sum(f['commission'] + f['fees'] + f.get('slippage_cost_total', 0.0) for f in fill_details_list)
            # reward_components['transaction_cost_penalty'] = -self.penalty_transaction_cost_value * total_costs

        # 4. Penalty for Holding/Inaction (if configured and holding a position)
        # Assuming ActionTypeEnumForEnv.HOLD is available
        if self.penalty_holding_inaction > 0 and \
                decoded_action.get('type') == ActionTypeEnumForEnv.HOLD:
            has_open_position = False
            if portfolio_state_before_action.get('positions'):
                for asset_id, pos_data in portfolio_state_before_action['positions'].items():
                    if pos_data.get('current_side') != PositionSideEnum.FLAT:
                        has_open_position = True
                        break
            if has_open_position:
                reward_components['holding_inaction_penalty'] = -self.penalty_holding_inaction

        # 5. Step-wise Drawdown Penalty (penalty for negative equity change in this step)
        if self.penalty_drawdown_step > 0 > equity_change:
            # Penalize the negative change. equity_change is already negative here.
            reward_components['drawdown_step_penalty'] = self.penalty_drawdown_step * equity_change  # equity_change is negative

        # 6. Penalty for Invalid Actions
        if self.penalty_invalid_action > 0 and decoded_action.get('invalid_reason'):
            reward_components['invalid_action_penalty'] = -self.penalty_invalid_action

        # 7. Terminal Rewards/Penalties
        if terminated:
            if termination_reason == TerminationReasonEnumForEnv.BANKRUPTCY:
                reward_components['terminal_bankruptcy_penalty'] = -self.terminal_penalty_bankruptcy
            elif termination_reason == TerminationReasonEnumForEnv.MAX_LOSS_REACHED:
                reward_components['terminal_max_loss_penalty'] = -self.terminal_penalty_max_loss
            # Can add bonus for finishing episode profitably, e.g.
            # final_pnl = portfolio_state_next_t['total_equity'] - portfolio_state_next_t.get('initial_capital', self.initial_capital_fallback)
            # if final_pnl > 0: reward_components['terminal_profit_bonus'] = some_value * final_pnl

        total_reward = sum(reward_components.values())

        # Apply global scaling factor
        total_reward *= self.reward_scaling_factor

        if self.log_reward_components:
            logger.debug(
                f"Step Reward Calculation: Components: {reward_components}, Total Raw: {sum(reward_components.values())}, Scaled Total: {total_reward}")

        return total_reward
