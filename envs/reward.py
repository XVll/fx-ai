# envs/reward.py - CLEAN: Streamlined reward calculator with metrics integration

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
import time

from config.schemas import Config
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum

try:
    from envs.trading_environment import ActionTypeEnum
    from envs.trading_environment import TerminationReasonEnum as TerminationReasonEnumForEnv
except ImportError:
    from enum import Enum


    class ActionTypeEnum(Enum):
        HOLD = 0
        BUY = 1
        SELL = 2


    class TerminationReasonEnumForEnv(Enum):
        BANKRUPTCY = "BANKRUPTCY"
        MAX_LOSS_REACHED = "MAX_LOSS_REACHED"


class RewardCalculator:
    """Clean reward calculator with metrics integration"""

    def __init__(self, config: Config, metrics_integrator=None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.reward_config = config.env.reward
        self.metrics_integrator = metrics_integrator
        self.logger = logger or logging.getLogger(__name__)

        # Core reward weights
        self.weight_equity_change = self.reward_config.weight_equity_change
        self.weight_realized_pnl = self.reward_config.weight_realized_pnl

        # Penalties
        self.penalty_transaction_fill = self.reward_config.penalty_transaction_fill
        self.penalty_holding_inaction = self.reward_config.penalty_holding_inaction
        self.penalty_drawdown_step = self.reward_config.penalty_drawdown_step
        self.penalty_invalid_action = self.reward_config.penalty_invalid_action

        # Terminal penalties
        self.terminal_penalty_bankruptcy = self.reward_config.terminal_penalty_bankruptcy
        self.terminal_penalty_max_loss = self.reward_config.terminal_penalty_max_loss

        # Scaling
        self.reward_scaling_factor = self.reward_config.reward_scaling_factor

        # Episode tracking
        self.step_count = 0
        self.last_reward_components = {}
        self.episode_reward_summary = {
            "total_equity_change_reward": 0.0,
            "total_realized_pnl_bonus": 0.0,
            "total_transaction_penalties": 0.0,
            "total_invalid_action_penalties": 0.0,
            "total_terminal_penalties": 0.0,
        }

    def reset(self):
        """Reset for new episode"""
        self.step_count = 0
        self.last_reward_components = {}
        self.episode_reward_summary = {
            "total_equity_change_reward": 0.0,
            "total_realized_pnl_bonus": 0.0,
            "total_transaction_penalties": 0.0,
            "total_invalid_action_penalties": 0.0,
            "total_terminal_penalties": 0.0,
        }

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
        action_type = decoded_action.get('type')

        # 1. Primary Reward: Change in Total Equity
        equity_before = portfolio_state_before_action['total_equity']
        equity_after = portfolio_state_next_t['total_equity']
        equity_change = equity_after - equity_before
        reward_components['equity_change'] = self.weight_equity_change * equity_change
        self.episode_reward_summary["total_equity_change_reward"] += reward_components['equity_change']

        # 2. Realized PnL bonus
        if self.weight_realized_pnl > 0:
            realized_pnl_before = portfolio_state_before_action['realized_pnl_session']
            realized_pnl_after = portfolio_state_after_action_fills['realized_pnl_session']
            step_realized_pnl = realized_pnl_after - realized_pnl_before
            reward_components['realized_pnl'] = self.weight_realized_pnl * step_realized_pnl
            self.episode_reward_summary["total_realized_pnl_bonus"] += reward_components['realized_pnl']

        # 3. Transaction penalties
        if self.penalty_transaction_fill > 0 and fill_details_list:
            reward_components['transaction_cost'] = -self.penalty_transaction_fill * len(fill_details_list)
            self.episode_reward_summary["total_transaction_penalties"] += reward_components['transaction_cost']

        # 4. Invalid action penalties
        if self.penalty_invalid_action > 0 and decoded_action.get('invalid_reason'):
            reward_components['invalid_action'] = -self.penalty_invalid_action
            self.episode_reward_summary["total_invalid_action_penalties"] += reward_components['invalid_action']

        # 5. Terminal penalties
        if terminated:
            if termination_reason == TerminationReasonEnumForEnv.BANKRUPTCY:
                reward_components['bankruptcy_penalty'] = -self.terminal_penalty_bankruptcy
                self.episode_reward_summary["total_terminal_penalties"] += reward_components['bankruptcy_penalty']
            elif termination_reason == TerminationReasonEnumForEnv.MAX_LOSS_REACHED:
                reward_components['max_loss_penalty'] = -self.terminal_penalty_max_loss
                self.episode_reward_summary["total_terminal_penalties"] += reward_components['max_loss_penalty']

        # Calculate total reward
        total_reward = sum(reward_components.values()) * self.reward_scaling_factor

        # Store components for metrics
        self.last_reward_components = reward_components.copy()
        self.step_count += 1

        # Send to metrics if available
        if self.metrics_integrator:
            action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)
            self.metrics_integrator.record_environment_step(
                reward=total_reward,
                action=action_name,
                is_invalid=bool(decoded_action.get('invalid_reason')),
                reward_components=reward_components
            )

        # Log significant events only
        if abs(total_reward) > 0.01 or fill_details_list or terminated:
            action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)

            if terminated and termination_reason:
                self.logger.warning(f"Episode terminated: {termination_reason.value}, Final reward: {total_reward:.4f}")
            elif fill_details_list:
                self.logger.debug(f"{action_name} executed, Reward: {total_reward:.4f}, Equity change: ${equity_change:.2f}")

        return total_reward

    def get_last_reward_components(self) -> Dict[str, float]:
        """Get the last reward components for metrics"""
        return self.last_reward_components.copy()

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get episode reward summary"""
        return {
            **self.episode_reward_summary,
            'total_steps': self.step_count,
            'last_components': self.last_reward_components.copy()
        }