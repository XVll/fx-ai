# envs/reward.py - UPDATED: Enhanced reward calculator with comprehensive dashboard integration

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
import time

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

        # UPDATED: Enhanced tracking for comprehensive dashboard integration
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

        # UPDATED: Enhanced component tracking for comprehensive dashboard
        self.last_reward_components = {}
        self.reward_component_history = deque(maxlen=100)  # Track recent reward components
        self.component_impact_analysis = defaultdict(lambda: {"total": 0.0, "count": 0, "avg": 0.0})
        self.significant_reward_threshold = 0.001

        # UPDATED: Action-reward correlation tracking for dashboard analytics
        self.action_reward_correlation = defaultdict(lambda: {"total_reward": 0.0, "count": 0, "positive_count": 0})

        # UPDATED: Step-by-step reward analysis for dashboard
        self.recent_step_rewards = deque(maxlen=50)  # Track recent step rewards for trend analysis
        self.reward_trend_analysis = {"improving": 0, "declining": 0, "stable": 0}

        self.logger.info(f"RewardCalculator initialized with comprehensive dashboard integration")

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
        # Note: Don't reset component_impact_analysis and action_reward_correlation
        # as these are useful across episodes
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

        # UPDATED: Store components for comprehensive dashboard
        self.last_reward_components = reward_components.copy()
        self.step_count += 1

        # UPDATED: Track reward component history and analysis for dashboard
        self.reward_component_history.append({
            'step': self.step_count,
            'timestamp': time.time(),
            'components': reward_components.copy(),
            'total_reward': total_reward,
            'action_type': action_name
        })

        # UPDATED: Update component impact analysis for dashboard analytics
        for component, value in reward_components.items():
            if abs(value) > 0.0001:  # Only track significant components
                self.component_impact_analysis[component]["total"] += value
                self.component_impact_analysis[component]["count"] += 1
                self.component_impact_analysis[component]["avg"] = (
                        self.component_impact_analysis[component]["total"] /
                        self.component_impact_analysis[component]["count"]
                )

        # UPDATED: Track action-reward correlation for dashboard
        self.action_reward_correlation[action_name]["total_reward"] += total_reward
        self.action_reward_correlation[action_name]["count"] += 1
        if total_reward > 0:
            self.action_reward_correlation[action_name]["positive_count"] += 1

        # UPDATED: Track step rewards for trend analysis
        self.recent_step_rewards.append(total_reward)
        if len(self.recent_step_rewards) >= 10:
            recent_avg = sum(list(self.recent_step_rewards)[-10:]) / 10
            older_avg = sum(list(self.recent_step_rewards)[-20:-10]) / 10 if len(self.recent_step_rewards) >= 20 else recent_avg

            if recent_avg > older_avg * 1.05:
                self.reward_trend_analysis["improving"] += 1
            elif recent_avg < older_avg * 0.95:
                self.reward_trend_analysis["declining"] += 1
            else:
                self.reward_trend_analysis["stable"] += 1

        # UPDATED: Smart logging - enhanced for dashboard integration
        should_log_detail = (
                abs(total_reward) > self.significant_reward_threshold or
                fill_details_list or
                decoded_action.get('invalid_reason') or
                terminated or truncated or
                equity_change != 0
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

        # UPDATED: Enhanced periodic analysis for dashboard (every 100 steps)
        if self.step_count % 100 == 0:
            self._log_comprehensive_reward_analysis()

        # Episode end comprehensive summary
        if terminated or truncated:
            self._log_comprehensive_episode_summary()

        return total_reward

    def get_last_reward_components(self) -> Dict[str, float]:
        """Get the last reward components for dashboard integration"""
        return self.last_reward_components.copy()

    def get_comprehensive_reward_analysis(self) -> Dict[str, Any]:
        """UPDATED: Get comprehensive reward analysis for dashboard integration"""
        analysis = {
            'episode_summary': self.episode_reward_summary.copy(),
            'total_steps': self.step_count,
            'last_components': self.last_reward_components.copy(),
            'component_impact_analysis': dict(self.component_impact_analysis),
            'action_reward_correlation': dict(self.action_reward_correlation),
            'reward_trend_analysis': self.reward_trend_analysis.copy()
        }

        # Calculate component impact percentages
        total_episode_reward = sum(self.episode_reward_summary.values())
        if abs(total_episode_reward) > 0.001:
            for component, value in self.episode_reward_summary.items():
                impact_key = f"{component}_impact_pct"
                analysis[impact_key] = (abs(value) / abs(total_episode_reward)) * 100

        # Add action reward statistics
        for action_type, stats in self.action_reward_correlation.items():
            if stats["count"] > 0:
                stats["avg_reward"] = stats["total_reward"] / stats["count"]
                stats["positive_rate"] = (stats["positive_count"] / stats["count"]) * 100

        return analysis

    def _log_comprehensive_reward_analysis(self):
        """UPDATED: Enhanced periodic reward analysis for dashboard"""
        try:
            if self.step_count == 0:
                return

            total_reward_so_far = sum(self.episode_reward_summary.values())
            profitable_rate = (self.episode_reward_summary["profitable_steps"] / self.step_count) * 100

            # UPDATED: Identify dominant reward components with more detail
            dominant_components = []
            for component, total_value in self.episode_reward_summary.items():
                if abs(total_value) > abs(total_reward_so_far) * 0.1:  # More than 10% of total
                    percentage = (abs(total_value) / abs(total_reward_so_far)) * 100 if total_reward_so_far != 0 else 0
                    dominant_components.append(f"{component}: {total_value:.3f} ({percentage:.1f}%)")

            # UPDATED: Action effectiveness analysis
            best_action = max(self.action_reward_correlation.items(),
                              key=lambda x: x[1]["total_reward"] / x[1]["count"] if x[1]["count"] > 0 else 0,
                              default=("NONE", {"total_reward": 0, "count": 1}))

            best_action_avg = best_action[1]["total_reward"] / best_action[1]["count"] if best_action[1]["count"] > 0 else 0

            self.logger.info(f"📊 Comprehensive Reward Analysis (Step {self.step_count}): "
                             f"Total: {total_reward_so_far:.4f}, "
                             f"Profitable: {profitable_rate:.1f}%, "
                             f"Best Action: {best_action[0]} (avg: {best_action_avg:.4f})")

            if dominant_components:
                self.logger.info(f"   🔍 Dominant Components: {', '.join(dominant_components[:3])}")

            # Alert for potential issues
            if profitable_rate < 30:
                self.logger.warning(f"⚠️ Low profitability rate: {profitable_rate:.1f}% - check reward balance")

            # UPDATED: Trend analysis
            if len(self.recent_step_rewards) >= 20:
                trend_summary = (f"Trends - Improving: {self.reward_trend_analysis['improving']}, "
                                 f"Declining: {self.reward_trend_analysis['declining']}, "
                                 f"Stable: {self.reward_trend_analysis['stable']}")
                self.logger.info(f"   📈 {trend_summary}")

        except Exception as e:
            self.logger.debug(f"Error in comprehensive reward analysis: {e}")

    def _log_comprehensive_episode_summary(self):
        """UPDATED: Log comprehensive episode reward summary for dashboard"""
        try:
            total_episode_reward = sum(self.episode_reward_summary.values())

            # UPDATED: Calculate comprehensive component analysis
            component_analysis = {}
            for component, value in self.episode_reward_summary.items():
                if abs(value) > 0.001:
                    percentage = (abs(value) / abs(total_episode_reward) * 100) if total_episode_reward != 0 else 0
                    component_analysis[component] = {
                        'value': value,
                        'percentage': percentage,
                        'impact_level': 'HIGH' if percentage > 20 else 'MEDIUM' if percentage > 5 else 'LOW'
                    }

            # Sort by absolute impact
            sorted_components = sorted(component_analysis.items(),
                                       key=lambda x: abs(x[1]['value']), reverse=True)

            self.logger.info("=== COMPREHENSIVE EPISODE REWARD SUMMARY ===")
            self.logger.info(f"Total Episode Reward: {total_episode_reward:.4f}")

            # UPDATED: Show top reward drivers with impact levels
            for component, analysis in sorted_components[:5]:  # Top 5 components
                component_name = component.replace('total_', '').replace('_', ' ').title()
                value = analysis['value']
                percentage = analysis['percentage']
                impact_level = analysis['impact_level']
                impact_icon = "🟢" if value > 0 else "🔴" if value < 0 else "⚪"
                level_icon = "🔥" if impact_level == 'HIGH' else "⚡" if impact_level == 'MEDIUM' else "💡"

                self.logger.info(f"  {impact_icon}{level_icon} {component_name}: {value:.4f} ({percentage:.1f}%)")

            # UPDATED: Step outcome summary with trend analysis
            total_steps = (self.episode_reward_summary["profitable_steps"] +
                           self.episode_reward_summary["unprofitable_steps"] +
                           self.episode_reward_summary["neutral_steps"])

            if total_steps > 0:
                profit_rate = (self.episode_reward_summary["profitable_steps"] / total_steps) * 100
                status_icon = "🟢" if profit_rate > 60 else "🟡" if profit_rate > 40 else "🔴"
                self.logger.info(f"{status_icon} Step Success Rate: {profit_rate:.1f}% "
                                 f"({self.episode_reward_summary['profitable_steps']}/{total_steps})")

            # UPDATED: Action effectiveness summary
            self.logger.info("📊 Action Effectiveness Summary:")
            for action_type, stats in self.action_reward_correlation.items():
                if stats["count"] > 0:
                    avg_reward = stats["total_reward"] / stats["count"]
                    positive_rate = (stats["positive_count"] / stats["count"]) * 100
                    effectiveness_icon = "🟢" if avg_reward > 0 else "🔴" if avg_reward < 0 else "⚪"
                    self.logger.info(f"  {effectiveness_icon} {action_type}: Avg={avg_reward:.4f}, "
                                     f"Success={positive_rate:.1f}%, Count={stats['count']}")

        except Exception as e:
            self.logger.debug(f"Error in comprehensive episode summary: {e}")

    def get_reward_bias_summary(self) -> Dict[str, Any]:
        """UPDATED: Get comprehensive reward bias summary for dashboard analytics"""
        summary = self.get_comprehensive_reward_analysis()

        # Add bias detection
        bias_indicators = {}

        # Check for dominant components (potential over-weighting)
        total_reward = sum(self.episode_reward_summary.values())
        if abs(total_reward) > 0.001:
            for component, value in self.episode_reward_summary.items():
                impact_pct = (abs(value) / abs(total_reward)) * 100
                if impact_pct > 50:
                    bias_indicators[f"{component}_dominance"] = impact_pct

        # Check for action bias
        if len(self.action_reward_correlation) > 1:
            action_rewards = {action: stats["total_reward"] / stats["count"]
                              for action, stats in self.action_reward_correlation.items()
                              if stats["count"] > 0}

            if action_rewards:
                max_reward = max(action_rewards.values())
                min_reward = min(action_rewards.values())
                if max_reward > 0 and min_reward < 0 and abs(max_reward / min_reward) > 5:
                    bias_indicators["action_reward_imbalance"] = max_reward / min_reward

        summary['bias_indicators'] = bias_indicators
        return summary