# envs/reward.py - FIXED: Improved logging for reward system analysis

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

        # FIXED: Enhanced tracking for reward system analysis
        self.action_reward_tracking = {"HOLD": [], "BUY": [], "SELL": []}
        self.step_count = 0
        self.episode_reward_summary = {
            "total_equity_change_reward": 0.0,
            "total_realized_pnl_bonus": 0.0,
            "total_transaction_penalties": 0.0,
            "total_inaction_penalties": 0.0,
            "total_drawdown_penalties": 0.0,
            "total_invalid_action_penalties": 0.0,
            "total_profit_bonuses": 0.0,
            "profitable_actions": 0,
            "unprofitable_actions": 0,
            "neutral_actions": 0
        }

        # FIXED: Reduce logging frequency to essential insights only
        self.significant_reward_threshold = 0.01  # Only log significant rewards
        self.analysis_frequency = 100  # Analyze every 100 steps instead of 50

        self.logger.info(f"RewardCalculator initialized with balanced reward config")

    def reset(self):
        """Reset tracking for new episode"""
        self.action_reward_tracking = {"HOLD": [], "BUY": [], "SELL": []}
        self.step_count = 0
        self.episode_reward_summary = {
            "total_equity_change_reward": 0.0,
            "total_realized_pnl_bonus": 0.0,
            "total_transaction_penalties": 0.0,
            "total_inaction_penalties": 0.0,
            "total_drawdown_penalties": 0.0,
            "total_invalid_action_penalties": 0.0,
            "total_profit_bonuses": 0.0,
            "profitable_actions": 0,
            "unprofitable_actions": 0,
            "neutral_actions": 0
        }
        self.logger.debug("RewardCalculator reset for new episode")

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

        # 1. Primary Reward: Change in Total Equity
        equity_before = portfolio_state_before_action['total_equity']
        equity_after_market_move = portfolio_state_next_t['total_equity']
        equity_change = equity_after_market_move - equity_before
        reward_components['equity_change_reward'] = self.weight_equity_change * equity_change
        self.episode_reward_summary["total_equity_change_reward"] += reward_components['equity_change_reward']

        # 2. Realized PnL bonus
        if self.weight_realized_pnl > 0:
            realized_pnl_before = portfolio_state_before_action['realized_pnl_session']
            realized_pnl_after_fills = portfolio_state_after_action_fills['realized_pnl_session']
            step_realized_pnl = realized_pnl_after_fills - realized_pnl_before
            reward_components['realized_pnl_bonus'] = self.weight_realized_pnl * step_realized_pnl
            self.episode_reward_summary["total_realized_pnl_bonus"] += reward_components['realized_pnl_bonus']

        # 3. Transaction penalties
        if self.penalty_transaction_fill > 0 and fill_details_list:
            num_fills = len(fill_details_list)
            reward_components['transaction_fill_penalty'] = -self.penalty_transaction_fill * num_fills
            self.episode_reward_summary["total_transaction_penalties"] += reward_components['transaction_fill_penalty']

        # 4. Inaction penalties (improved logic)
        if self.penalty_holding_inaction > 0 and action_type_from_env == ActionTypeEnum.HOLD:
            has_open_position = False
            positions_before = portfolio_state_before_action.get('positions', {})
            if positions_before:
                for asset_id, pos_data in positions_before.items():
                    if isinstance(pos_data, dict) and pos_data.get('current_side') != PositionSideEnum.FLAT:
                        has_open_position = True
                        break

            if has_open_position and equity_change < 0:
                reward_components['holding_inaction_penalty'] = -self.penalty_holding_inaction
                self.episode_reward_summary["total_inaction_penalties"] += reward_components['holding_inaction_penalty']
            elif not has_open_position:
                reward_components['holding_inaction_penalty'] = -self.penalty_holding_inaction * 0.1
                self.episode_reward_summary["total_inaction_penalties"] += reward_components['holding_inaction_penalty']

        # 5. Drawdown penalties
        if self.penalty_drawdown_step > 0 and equity_change < 0:
            reward_components['drawdown_step_penalty'] = self.penalty_drawdown_step * equity_change
            self.episode_reward_summary["total_drawdown_penalties"] += reward_components['drawdown_step_penalty']

        # 6. Invalid action penalties
        if self.penalty_invalid_action > 0 and decoded_action.get('invalid_reason'):
            reward_components['invalid_action_penalty'] = -self.penalty_invalid_action
            self.episode_reward_summary["total_invalid_action_penalties"] += reward_components['invalid_action_penalty']

        # 7. Terminal penalties
        if terminated:
            if termination_reason == TerminationReasonEnumForEnv.BANKRUPTCY:
                reward_components['terminal_bankruptcy_penalty'] = -self.terminal_penalty_bankruptcy
            elif termination_reason == TerminationReasonEnumForEnv.MAX_LOSS_REACHED:
                reward_components['terminal_max_loss_penalty'] = -self.terminal_penalty_max_loss

        # 8. Profit bonuses
        if equity_change > 0:
            reward_components['profit_bonus'] = 0.01 * equity_change
            self.episode_reward_summary["total_profit_bonuses"] += reward_components['profit_bonus']
            self.episode_reward_summary["profitable_actions"] += 1
        elif equity_change < 0:
            self.episode_reward_summary["unprofitable_actions"] += 1
        else:
            self.episode_reward_summary["neutral_actions"] += 1

        # 9. Risk management penalties
        try:
            positions_after = portfolio_state_next_t.get('positions', {})
            total_equity = portfolio_state_next_t['total_equity']

            for asset_id, pos_data in positions_after.items():
                if isinstance(pos_data, dict):
                    position_value = abs(pos_data.get('market_value', 0.0))
                    if total_equity > 0 and position_value > 0:
                        position_ratio = position_value / total_equity
                        if position_ratio > 1.5:
                            reward_components['excessive_leverage_penalty'] = -0.01 * (position_ratio - 1.5)
        except Exception as e:
            self.logger.debug(f"Error calculating position ratio penalty: {e}")

        # Calculate total reward
        total_reward = sum(reward_components.values())
        total_reward *= self.reward_scaling_factor

        # Track rewards by action type
        self.action_reward_tracking[action_name].append(total_reward)
        self.step_count += 1

        # FIXED: Smart logging - only log significant events and periodic analysis
        should_log_detail = (
                abs(total_reward) > self.significant_reward_threshold or  # Significant reward
                fill_details_list or  # Any fills occurred
                decoded_action.get('invalid_reason') or  # Invalid action
                terminated or truncated  # Episode end
        )

        if should_log_detail and self.log_reward_components:
            loggable_components = {
                k: (v.value if hasattr(v, 'value') else v)
                for k, v in reward_components.items()
                if abs(v) > 0.001  # Only log non-trivial components
            }

            if loggable_components:  # Only log if there are significant components
                self.logger.info(
                    f"Reward {action_name}: {total_reward:.4f} | "
                    f"Equity Δ: ${equity_change:.4f} | "
                    f"Components: {loggable_components}"
                )

        # Periodic bias analysis (less frequent)
        if self.step_count % self.analysis_frequency == 0:
            self._log_reward_analysis()

        # Episode end summary
        if terminated or truncated:
            self._log_episode_summary()

        return total_reward

    def _log_reward_analysis(self):
        """FIXED: Enhanced reward system analysis with actionable insights"""
        try:
            analysis = {}
            total_actions = 0

            for action_type, rewards in self.action_reward_tracking.items():
                if rewards:
                    analysis[action_type] = {
                        'count': len(rewards),
                        'mean_reward': sum(rewards) / len(rewards),
                        'total_reward': sum(rewards),
                        'positive_count': sum(1 for r in rewards if r > 0.001),
                        'negative_count': sum(1 for r in rewards if r < -0.001),
                        'max_reward': max(rewards),
                        'min_reward': min(rewards)
                    }
                    total_actions += len(rewards)

            if total_actions > 0:
                self.logger.info("=== REWARD SYSTEM ANALYSIS ===")

                # Action distribution and performance
                for action_type, stats in analysis.items():
                    if stats['count'] > 0:
                        action_pct = (stats['count'] / total_actions) * 100
                        success_rate = (stats['positive_count'] / stats['count']) * 100

                        performance_indicator = "🟢" if stats['mean_reward'] > 0.01 else "🔴" if stats['mean_reward'] < -0.01 else "🟡"

                        self.logger.info(
                            f"{performance_indicator} {action_type}: {action_pct:.1f}% of actions | "
                            f"Mean: {stats['mean_reward']:.4f} | Success: {success_rate:.1f}% | "
                            f"Range: [{stats['min_reward']:.4f}, {stats['max_reward']:.4f}]"
                        )

                # System health indicators
                profitable_actions = self.episode_reward_summary["profitable_actions"]
                total_episode_actions = (profitable_actions +
                                         self.episode_reward_summary["unprofitable_actions"] +
                                         self.episode_reward_summary["neutral_actions"])

                if total_episode_actions > 0:
                    profitability_rate = (profitable_actions / total_episode_actions) * 100
                    self.logger.info(f"📊 System Profitability: {profitability_rate:.1f}% profitable actions")

                # Detect potential issues
                if analysis.get('HOLD', {}).get('mean_reward', 0) > analysis.get('BUY', {}).get('mean_reward', 0):
                    self.logger.warning("⚠️ HOLD actions more rewarded than BUY - check action bias")

                if analysis.get('BUY', {}).get('count', 0) == 0 and analysis.get('SELL', {}).get('count', 0) == 0:
                    self.logger.warning("⚠️ No trading actions taken - agent may be too conservative")

        except Exception as e:
            self.logger.debug(f"Error in reward analysis: {e}")

    def _log_episode_summary(self):
        """FIXED: Log comprehensive episode reward summary"""
        try:
            total_reward = sum(self.episode_reward_summary.values())

            # Calculate reward composition
            significant_components = {
                k: v for k, v in self.episode_reward_summary.items()
                if abs(v) > 0.01  # Only significant components
            }

            if significant_components:
                self.logger.info("=== EPISODE REWARD SUMMARY ===")
                self.logger.info(f"Total Episode Reward: {total_reward:.4f}")

                for component, value in significant_components.items():
                    percentage = (value / total_reward * 100) if total_reward != 0 else 0
                    component_name = component.replace('total_', '').replace('_', ' ').title()
                    self.logger.info(f"  {component_name}: {value:.4f} ({percentage:.1f}%)")

                # Action outcome summary
                total_actions = (self.episode_reward_summary["profitable_actions"] +
                                 self.episode_reward_summary["unprofitable_actions"] +
                                 self.episode_reward_summary["neutral_actions"])

                if total_actions > 0:
                    profit_rate = (self.episode_reward_summary["profitable_actions"] / total_actions) * 100
                    status_icon = "🟢" if profit_rate > 60 else "🟡" if profit_rate > 40 else "🔴"
                    self.logger.info(
                        f"{status_icon} Action Success Rate: {profit_rate:.1f}% ({self.episode_reward_summary['profitable_actions']}/{total_actions})")

        except Exception as e:
            self.logger.debug(f"Error in episode summary: {e}")

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

        # Add episode summary
        summary['episode_summary'] = self.episode_reward_summary.copy()
        return summary