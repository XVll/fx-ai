# envs/reward.py - FIXED: Enhanced reward calculator with better component tracking for dashboard

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

        # FIXED: Enhanced tracking for reward analysis
        self.step_count = 0
        self.episode_reward_summary = {
            "total_equity_change_reward": 0.0,
            "total_realized_pnl_bonus": 0.0,
            "total_transaction_penalties": 0.0,
            "total_inaction_penalties": 0.0,
            "total_drawdown_penalties": 0.0,
            "total_invalid_action_penalties": 0.0,
            "total_terminal_penalties": 0.0,
            "total_profit_bonuses": 0.0,
            "profitable_steps": 0,
            "unprofitable_steps": 0,
            "neutral_steps": 0
        }

        # FIXED: Component tracking for dashboard integration
        self.last_reward_components = {}
        self.significant_reward_threshold = 0.001  # Reduced threshold for better tracking

        self.logger.info(f"RewardCalculator initialized with enhanced component tracking")

    def reset(self):
        """Reset tracking for new episode"""
        self.step_count = 0
        self.episode_reward_summary = {
            "total_equity_change_reward": 0.0,
            "total_realized_pnl_bonus": 0.0,
            "total_transaction_penalties": 0.0,
            "total_inaction_penalties": 0.0,
            "total_drawdown_penalties": 0.0,
            "total_invalid_action_penalties": 0.0,
            "total_terminal_penalties": 0.0,
            "total_profit_bonuses": 0.0,
            "profitable_steps": 0,
            "unprofitable_steps": 0,
            "neutral_steps": 0
        }
        self.last_reward_components = {}
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
        reward_components['equity_change'] = self.weight_equity_change * equity_change
        self.episode_reward_summary["total_equity_change_reward"] += reward_components['equity_change']

        # 2. Realized PnL bonus
        if self.weight_realized_pnl > 0:
            realized_pnl_before = portfolio_state_before_action['realized_pnl_session']
            realized_pnl_after_fills = portfolio_state_after_action_fills['realized_pnl_session']
            step_realized_pnl = realized_pnl_after_fills - realized_pnl_before
            reward_components['realized_pnl'] = self.weight_realized_pnl * step_realized_pnl
            self.episode_reward_summary["total_realized_pnl_bonus"] += reward_components['realized_pnl']

        # 3. Transaction penalties
        if self.penalty_transaction_fill > 0 and fill_details_list:
            num_fills = len(fill_details_list)
            reward_components['transaction_cost'] = -self.penalty_transaction_fill * num_fills
            self.episode_reward_summary["total_transaction_penalties"] += reward_components['transaction_cost']

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
                reward_components['inaction_penalty'] = -self.penalty_holding_inaction
                self.episode_reward_summary["total_inaction_penalties"] += reward_components['inaction_penalty']
            elif not has_open_position:
                reward_components['inaction_penalty'] = -self.penalty_holding_inaction * 0.1
                self.episode_reward_summary["total_inaction_penalties"] += reward_components['inaction_penalty']

        # 5. Drawdown penalties
        if self.penalty_drawdown_step > 0 and equity_change < 0:
            reward_components['drawdown_penalty'] = self.penalty_drawdown_step * equity_change
            self.episode_reward_summary["total_drawdown_penalties"] += reward_components['drawdown_penalty']

        # 6. Invalid action penalties
        if self.penalty_invalid_action > 0 and decoded_action.get('invalid_reason'):
            reward_components['invalid_action'] = -self.penalty_invalid_action
            self.episode_reward_summary["total_invalid_action_penalties"] += reward_components['invalid_action']

        # 7. Terminal penalties
        if terminated:
            if termination_reason == TerminationReasonEnumForEnv.BANKRUPTCY:
                reward_components['bankruptcy_penalty'] = -self.terminal_penalty_bankruptcy
                self.episode_reward_summary["total_terminal_penalties"] += reward_components['bankruptcy_penalty']
            elif termination_reason == TerminationReasonEnumForEnv.MAX_LOSS_REACHED:
                reward_components['max_loss_penalty'] = -self.terminal_penalty_max_loss
                self.episode_reward_summary["total_terminal_penalties"] += reward_components['max_loss_penalty']

        # 8. Profit bonuses for encouraging positive outcomes
        if equity_change > 0:
            reward_components['profit_bonus'] = 0.01 * equity_change
            self.episode_reward_summary["total_profit_bonuses"] += reward_components['profit_bonus']
            self.episode_reward_summary["profitable_steps"] += 1
        elif equity_change < 0:
            self.episode_reward_summary["unprofitable_steps"] += 1
        else:
            self.episode_reward_summary["neutral_steps"] += 1

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
                            reward_components['leverage_penalty'] = -0.01 * (position_ratio - 1.5)
        except Exception as e:
            self.logger.debug(f"Error calculating leverage penalty: {e}")

        # Calculate total reward
        total_reward = sum(reward_components.values())
        total_reward *= self.reward_scaling_factor

        # FIXED: Store components for dashboard (always store, regardless of significance)
        self.last_reward_components = reward_components.copy()
        self.step_count += 1

        # FIXED: Smart logging - only log significant events and key decisions
        should_log_detail = (
                abs(total_reward) > self.significant_reward_threshold or  # Significant reward
                fill_details_list or  # Any fills occurred
                decoded_action.get('invalid_reason') or  # Invalid action
                terminated or truncated or  # Episode end
                equity_change != 0  # Any equity change
        )

        if should_log_detail and self.log_reward_components:
            # Filter out zero components for cleaner logging
            non_zero_components = {
                k: v for k, v in reward_components.items()
                if abs(v) > 0.0001
            }

            if non_zero_components:
                self.logger.info(f"💰 {action_name} Reward: {total_reward:.4f} | "
                                 f"Equity Δ: ${equity_change:.4f} | "
                                 f"Components: {non_zero_components}")

        # Periodic comprehensive analysis (every 100 steps)
        if self.step_count % 100 == 0:
            self._log_reward_analysis()

        # Episode end summary
        if terminated or truncated:
            self._log_episode_summary()

        return total_reward

    def get_last_reward_components(self) -> Dict[str, float]:
        """FIXED: Get the last reward components for dashboard integration"""
        return self.last_reward_components.copy()

    def _log_reward_analysis(self):
        """FIXED: Enhanced periodic reward analysis"""
        try:
            if self.step_count == 0:
                return

            total_reward_so_far = sum(self.episode_reward_summary.values())
            profitable_rate = (self.episode_reward_summary["profitable_steps"] / self.step_count) * 100

            # Identify dominant reward components
            dominant_components = []
            for component, total_value in self.episode_reward_summary.items():
                if abs(total_value) > abs(total_reward_so_far) * 0.1:  # More than 10% of total
                    dominant_components.append(f"{component}: {total_value:.3f}")

            self.logger.info(f"📊 Reward Analysis (Step {self.step_count}): "
                             f"Total: {total_reward_so_far:.4f}, "
                             f"Profitable: {profitable_rate:.1f}%, "
                             f"Dominant: {', '.join(dominant_components) if dominant_components else 'Balanced'}")

            # Alert for potential issues
            if profitable_rate < 30:
                self.logger.warning(f"⚠️ Low profitability rate: {profitable_rate:.1f}% - check reward balance")

        except Exception as e:
            self.logger.debug(f"Error in reward analysis: {e}")

    def _log_episode_summary(self):
        """FIXED: Log comprehensive episode reward summary"""
        try:
            total_episode_reward = sum(self.episode_reward_summary.values())

            # Calculate component percentages
            component_analysis = {}
            for component, value in self.episode_reward_summary.items():
                if abs(value) > 0.001:  # Only significant components
                    percentage = (abs(value) / abs(total_episode_reward) * 100) if total_episode_reward != 0 else 0
                    component_analysis[component] = {
                        'value': value,
                        'percentage': percentage
                    }

            # Sort by absolute impact
            sorted_components = sorted(component_analysis.items(),
                                       key=lambda x: abs(x[1]['value']), reverse=True)

            self.logger.info("=== EPISODE REWARD SUMMARY ===")
            self.logger.info(f"Total Episode Reward: {total_episode_reward:.4f}")

            # Show top reward drivers
            for component, analysis in sorted_components[:5]:  # Top 5 components
                component_name = component.replace('total_', '').replace('_', ' ').title()
                value = analysis['value']
                percentage = analysis['percentage']
                impact_icon = "🟢" if value > 0 else "🔴" if value < 0 else "⚪"
                self.logger.info(f"  {impact_icon} {component_name}: {value:.4f} ({percentage:.1f}%)")

            # Step outcome summary
            total_steps = (self.episode_reward_summary["profitable_steps"] +
                           self.episode_reward_summary["unprofitable_steps"] +
                           self.episode_reward_summary["neutral_steps"])

            if total_steps > 0:
                profit_rate = (self.episode_reward_summary["profitable_steps"] / total_steps) * 100
                status_icon = "🟢" if profit_rate > 60 else "🟡" if profit_rate > 40 else "🔴"
                self.logger.info(f"{status_icon} Step Success Rate: {profit_rate:.1f}% "
                                 f"({self.episode_reward_summary['profitable_steps']}/{total_steps})")

        except Exception as e:
            self.logger.debug(f"Error in episode summary: {e}")

    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of reward components for analysis"""
        summary = {
            'episode_summary': self.episode_reward_summary.copy(),
            'total_steps': self.step_count,
            'last_components': self.last_reward_components.copy()
        }

        # Calculate component impact percentages
        total_reward = sum(self.episode_reward_summary.values())
        if abs(total_reward) > 0.001:
            for component, value in self.episode_reward_summary.items():
                impact_key = f"{component}_impact_pct"
                summary[impact_key] = (abs(value) / abs(total_reward)) * 100

        return summary