# envs/reward.py - FIXED: Balanced reward function

import logging
from typing import Any, Dict, List, Optional

from config.config import Config
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum

try:
    from envs.trading_env import ActionTypeEnum
    from envs.trading_env import TerminationReasonEnum as TerminationReasonEnumForEnv
except ImportError:
    from enum import Enum


    class ActionTypeEnum(Enum):
        HOLD = 0
        BUY = 1
        SELL = 2


    class TerminationReasonEnumForEnv(Enum):
        BANKRUPTCY = "BANKRUPTCY"
        MAX_LOSS_REACHED = "MAX_LOSS_REACHED"

logger_reward = logging.getLogger(__name__)


class RewardCalculator:
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config_main = config
        self.reward_config = config.env.reward
        self.logger = logger or logger_reward

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

        # Scaling and logging
        self.reward_scaling_factor = self.reward_config.reward_scaling_factor
        self.log_reward_components = self.reward_config.log_reward_components

        # Fallback for initial capital
        self.initial_capital_fallback = self.config_main.simulation.portfolio_config.initial_cash

        # BIAS TRACKING: Monitor reward distribution by action type
        self.action_reward_tracking = {"HOLD": [], "BUY": [], "SELL": []}
        self.step_count = 0

        self.logger.info(f"RewardCalculator initialized with balanced reward config: {self.reward_config}")

    def reset(self):
        # Reset bias tracking for new episode
        self.action_reward_tracking = {"HOLD": [], "BUY": [], "SELL": []}
        self.step_count = 0
        self.logger.debug("RewardCalculator reset.")

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
        action_type_from_env = decoded_action.get('type')
        action_name = action_type_from_env.name if hasattr(action_type_from_env, 'name') else str(action_type_from_env)

        # 1. Primary Reward: Change in Total Equity (NEUTRAL - works for any action)
        equity_before = portfolio_state_before_action['total_equity']
        equity_after_market_move = portfolio_state_next_t['total_equity']
        equity_change = equity_after_market_move - equity_before
        reward_components['equity_change_reward'] = self.weight_equity_change * equity_change

        # 2. Optional: Explicit reward for Realized PnL (NEUTRAL - rewards profitable exits)
        if self.weight_realized_pnl > 0:
            realized_pnl_before = portfolio_state_before_action['realized_pnl_session']
            realized_pnl_after_fills = portfolio_state_after_action_fills['realized_pnl_session']
            step_realized_pnl = realized_pnl_after_fills - realized_pnl_before
            reward_components['realized_pnl_bonus'] = self.weight_realized_pnl * step_realized_pnl

        # 3. Penalty for Transaction Fills (NEUTRAL - discourages overtrading)
        if self.penalty_transaction_fill > 0 and fill_details_list:
            num_fills = len(fill_details_list)
            reward_components['transaction_fill_penalty'] = -self.penalty_transaction_fill * num_fills

        # 4. BALANCED Penalty for Holding/Inaction
        # FIXED: Only penalize HOLD when there's an obvious opportunity missed
        if self.penalty_holding_inaction > 0 and action_type_from_env == ActionTypeEnum.HOLD:
            has_open_position = False
            positions_before = portfolio_state_before_action.get('positions', {})
            if positions_before:
                for asset_id, pos_data in positions_before.items():
                    if isinstance(pos_data, dict) and pos_data.get('current_side') != PositionSideEnum.FLAT:
                        has_open_position = True
                        break

            # IMPROVEMENT: Only penalize HOLD if we have a position AND market moved against us
            if has_open_position and equity_change < 0:
                reward_components['holding_inaction_penalty'] = -self.penalty_holding_inaction
            # ALTERNATIVELY: Small base penalty for excessive HOLDing
            elif not has_open_position:
                # Very small penalty for holding when flat (encourages some action)
                reward_components['holding_inaction_penalty'] = -self.penalty_holding_inaction * 0.1

        # 5. Step-wise Drawdown Penalty (NEUTRAL - penalizes losses regardless of action)
        if self.penalty_drawdown_step > 0 and equity_change < 0:
            reward_components['drawdown_step_penalty'] = self.penalty_drawdown_step * equity_change

        # 6. Penalty for Invalid Actions (NEUTRAL - discourages invalid actions)
        if self.penalty_invalid_action > 0 and decoded_action.get('invalid_reason'):
            reward_components['invalid_action_penalty'] = -self.penalty_invalid_action

        # 7. Terminal Rewards/Penalties (NEUTRAL)
        if terminated:
            if termination_reason == TerminationReasonEnumForEnv.BANKRUPTCY:
                reward_components['terminal_bankruptcy_penalty'] = -self.terminal_penalty_bankruptcy
            elif termination_reason == TerminationReasonEnumForEnv.MAX_LOSS_REACHED:
                reward_components['terminal_max_loss_penalty'] = -self.terminal_penalty_max_loss

        # 8. BALANCE BOOST: Small bonus for profitable actions (ANY profitable action)
        if equity_change > 0:
            reward_components['profit_bonus'] = 0.01 * equity_change  # Small bonus for any profitable move

        # 9. RISK MANAGEMENT: Penalize excessive position size relative to equity
        try:
            positions_after = portfolio_state_next_t.get('positions', {})
            total_equity = portfolio_state_next_t['total_equity']

            for asset_id, pos_data in positions_after.items():
                if isinstance(pos_data, dict):
                    position_value = abs(pos_data.get('market_value', 0.0))
                    if total_equity > 0 and position_value > 0:
                        position_ratio = position_value / total_equity
                        # Penalize positions over 150% of equity
                        if position_ratio > 1.5:
                            reward_components['excessive_leverage_penalty'] = -0.01 * (position_ratio - 1.5)
        except Exception as e:
            self.logger.debug(f"Error calculating position ratio penalty: {e}")

        # Calculate total reward
        total_reward = sum(reward_components.values())
        total_reward *= self.reward_scaling_factor

        # BIAS TRACKING: Track rewards by action type
        self.action_reward_tracking[action_name].append(total_reward)
        self.step_count += 1

        # Log bias analysis every 50 steps
        if self.step_count % 50 == 0:
            self._log_bias_analysis()

        if self.log_reward_components:
            loggable_components = {
                k: (v.value if hasattr(v, 'value') else v)
                for k, v in reward_components.items()
            }
            self.logger.debug(
                f"Step Reward: {action_name} | Components: {loggable_components} | "
                f"Total Raw: {sum(reward_components.values()):.4f} | Scaled: {total_reward:.4f} | "
                f"Equity Δ: {equity_change:.4f}"
            )

        return total_reward

    def _log_bias_analysis(self):
        """Log analysis of reward distribution by action type to detect bias"""
        try:
            analysis = {}
            for action_type, rewards in self.action_reward_tracking.items():
                if rewards:
                    analysis[action_type] = {
                        'count': len(rewards),
                        'mean_reward': sum(rewards) / len(rewards),
                        'total_reward': sum(rewards),
                        'positive_count': sum(1 for r in rewards if r > 0),
                        'negative_count': sum(1 for r in rewards if r < 0)
                    }
                else:
                    analysis[action_type] = {
                        'count': 0, 'mean_reward': 0.0, 'total_reward': 0.0,
                        'positive_count': 0, 'negative_count': 0
                    }

            total_actions = sum(a['count'] for a in analysis.values())

            if total_actions > 0:
                self.logger.info("=== REWARD BIAS ANALYSIS ===")
                for action_type, stats in analysis.items():
                    if stats['count'] > 0:
                        action_pct = (stats['count'] / total_actions) * 100
                        pos_rate = (stats['positive_count'] / stats['count']) * 100
                        self.logger.info(
                            f"{action_type}: {action_pct:.1f}% of actions | "
                            f"Mean reward: {stats['mean_reward']:.4f} | "
                            f"Positive rate: {pos_rate:.1f}% | "
                            f"Total: {stats['total_reward']:.4f}"
                        )

                # Check for obvious bias
                if analysis['BUY']['count'] > 0 and analysis['SELL']['count'] > 0:
                    buy_mean = analysis['BUY']['mean_reward']
                    sell_mean = analysis['SELL']['mean_reward']
                    if abs(buy_mean - sell_mean) > 0.01:  # Significant difference
                        bias_direction = "BUY" if buy_mean > sell_mean else "SELL"
                        self.logger.warning(f"POTENTIAL REWARD BIAS detected favoring {bias_direction}")

        except Exception as e:
            self.logger.debug(f"Error in bias analysis: {e}")

    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of reward bias for end-of-episode analysis"""
        summary = {}
        total_steps = sum(len(rewards) for rewards in self.action_reward_tracking.values())

        for action_type, rewards in self.action_reward_tracking.items():
            if rewards:
                summary[action_type] = {
                    'count': len(rewards),
                    'percentage': (len(rewards) / total_steps) * 100 if total_steps > 0 else 0,
                    'mean_reward': sum(rewards) / len(rewards),
                    'total_reward': sum(rewards),
                    'positive_reward_rate': (sum(1 for r in rewards if r > 0) / len(rewards)) * 100
                }
            else:
                summary[action_type] = {
                    'count': 0, 'percentage': 0, 'mean_reward': 0,
                    'total_reward': 0, 'positive_reward_rate': 0
                }

        return summary