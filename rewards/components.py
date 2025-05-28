# rewards/components.py - Specific reward component implementations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from rewards.core import RewardComponent, RewardMetadata, RewardState, RewardType
from simulators.portfolio_simulator import PositionSideEnum


class RealizedPnLReward(RewardComponent):
    """Reward based on realized P&L from closed trades"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="realized_pnl",
            type=RewardType.FOUNDATIONAL,
            description="Reward based on realized profit/loss from closed trades"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        # Calculate realized PnL change
        realized_pnl_before = state.portfolio_before.get('realized_pnl_session', 0.0)
        realized_pnl_after = state.portfolio_after_fills.get('realized_pnl_session', 0.0)
        pnl_change = realized_pnl_after - realized_pnl_before
        
        # Include commission and slippage
        total_costs = 0.0
        for fill in state.fill_details:
            total_costs += fill.get('commission', 0.0) + fill.get('slippage_cost', 0.0)
        
        net_pnl = pnl_change - total_costs
        
        diagnostics = {
            'gross_pnl': pnl_change,
            'costs': total_costs,
            'net_pnl': net_pnl,
            'trades_closed': len([f for f in state.fill_details if f.get('closes_position', False)])
        }
        
        return net_pnl, diagnostics


class MarkToMarketReward(RewardComponent):
    """Immediate feedback on unrealized P&L changes"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="mark_to_market",
            type=RewardType.FOUNDATIONAL,
            description="Reward based on changes in unrealized P&L"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        # Calculate unrealized PnL change
        unrealized_before = state.portfolio_before.get('unrealized_pnl', 0.0)
        unrealized_after = state.portfolio_next.get('unrealized_pnl', 0.0)
        mtm_change = unrealized_after - unrealized_before
        
        # Scale based on position size to prevent over-leveraging
        position_value = abs(state.portfolio_next.get('position_value', 0.0))
        equity = state.portfolio_next.get('total_equity', 1.0)
        leverage = position_value / equity if equity > 0 else 0.0
        
        # Reduce reward if over-leveraged
        leverage_penalty = 1.0
        max_leverage = self.config.get('max_leverage', 2.0)
        if leverage > max_leverage:
            leverage_penalty = max_leverage / leverage
        
        adjusted_mtm = mtm_change * leverage_penalty
        
        diagnostics = {
            'mtm_change': mtm_change,
            'leverage': leverage,
            'leverage_penalty': leverage_penalty,
            'adjusted_mtm': adjusted_mtm
        }
        
        return adjusted_mtm, diagnostics


class DifferentialSharpeReward(RewardComponent):
    """Reward based on changes in risk-adjusted returns"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.window_size = config.get('window_size', 20)
        self.returns_buffer = deque(maxlen=self.window_size)
        self.min_periods = config.get('min_periods', 5)
        
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="differential_sharpe",
            type=RewardType.FOUNDATIONAL,
            description="Reward based on improvements in Sharpe ratio"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        # Calculate return for this step
        equity_before = state.portfolio_before.get('total_equity', 1.0)
        equity_after = state.portfolio_next.get('total_equity', 1.0)
        
        if equity_before > 0:
            step_return = (equity_after - equity_before) / equity_before
        else:
            step_return = 0.0
            
        self.returns_buffer.append(step_return)
        
        # Not enough data yet
        if len(self.returns_buffer) < self.min_periods:
            return 0.0, {'insufficient_data': True, 'buffer_size': len(self.returns_buffer)}
        
        # Calculate Sharpe ratio
        returns_array = np.array(self.returns_buffer)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return > 1e-8:
            current_sharpe = mean_return / std_return * np.sqrt(252 * 390)  # Annualized for minute data
        else:
            current_sharpe = np.sign(mean_return) * 10.0  # Cap at reasonable value
            
        # Calculate differential (improvement)
        if hasattr(self, 'previous_sharpe'):
            sharpe_diff = current_sharpe - self.previous_sharpe
        else:
            sharpe_diff = 0.0
            
        self.previous_sharpe = current_sharpe
        
        # Scale the reward
        reward = sharpe_diff * self.config.get('sharpe_scale', 0.1)
        
        diagnostics = {
            'step_return': step_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'current_sharpe': current_sharpe,
            'sharpe_diff': sharpe_diff,
            'buffer_size': len(self.returns_buffer)
        }
        
        return reward, diagnostics


class HoldingTimePenalty(RewardComponent):
    """Penalty for holding positions too long"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="holding_time_penalty",
            type=RewardType.SHAPING,
            description="Penalty for excessive position holding time",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check if we have a position
        position_side = state.portfolio_next.get('position_side')
        if position_side is None or position_side == PositionSideEnum.FLAT:
            return 0.0, {'has_position': False}
            
        # Get holding time threshold
        max_holding_time = self.config.get('max_holding_time', 60)  # steps
        penalty_per_step = self.config.get('penalty_per_step', 0.001)
        
        # Track holding time
        holding_time = state.current_trade_duration
        
        if holding_time > max_holding_time:
            excess_time = holding_time - max_holding_time
            penalty = -penalty_per_step * excess_time
            
            # Progressive penalty - gets worse over time
            if self.config.get('progressive_penalty', True):
                penalty *= (1 + excess_time / max_holding_time)
                
        diagnostics = {
            'holding_time': holding_time,
            'max_holding_time': max_holding_time,
            'excess_time': max(0, holding_time - max_holding_time),
            'penalty': penalty
        }
        
        return penalty, diagnostics


class OvertradingPenalty(RewardComponent):
    """Penalty for excessive trading frequency"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.trade_times = deque(maxlen=config.get('lookback_trades', 20))
        
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="overtrading_penalty",
            type=RewardType.SHAPING,
            description="Penalty for trading too frequently",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check if a trade was executed
        if state.fill_details:
            self.trade_times.append(state.step_count)
            
            # Calculate trading frequency
            window_size = self.config.get('frequency_window', 100)  # steps
            max_trades = self.config.get('max_trades_per_window', 5)
            
            # Count trades in window
            recent_trades = sum(1 for t in self.trade_times if state.step_count - t <= window_size)
            
            if recent_trades > max_trades:
                excess_trades = recent_trades - max_trades
                penalty_per_trade = self.config.get('penalty_per_excess_trade', 0.01)
                penalty = -penalty_per_trade * excess_trades
                
                # Exponential penalty for severe overtrading
                if self.config.get('exponential_penalty', True):
                    penalty *= (1 + excess_trades / max_trades)
                    
        diagnostics = {
            'recent_trades': len([t for t in self.trade_times if state.step_count - t <= self.config.get('frequency_window', 100)]),
            'trade_executed': bool(state.fill_details),
            'penalty': penalty
        }
        
        return penalty, diagnostics


class QuickProfitIncentive(RewardComponent):
    """Incentive for taking profits quickly"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="quick_profit_incentive",
            type=RewardType.SHAPING,
            description="Bonus for capturing profits quickly"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        incentive = 0.0
        
        # Check if we closed a profitable position
        for fill in state.fill_details:
            if fill.get('closes_position', False):
                realized_pnl = fill.get('realized_pnl', 0.0)
                # Ensure realized_pnl is not None
                if realized_pnl is None:
                    realized_pnl = 0.0
                
                if realized_pnl > 0:
                    # Get quick profit threshold
                    quick_time = self.config.get('quick_profit_time', 30)  # steps
                    bonus_rate = self.config.get('bonus_rate', 0.5)
                    
                    if state.current_trade_duration <= quick_time:
                        # Scale bonus by how quickly profit was taken
                        time_factor = 1.0 - (state.current_trade_duration / quick_time)
                        incentive += realized_pnl * bonus_rate * time_factor
                        
        diagnostics = {
            'trades_closed': len([f for f in state.fill_details if f.get('closes_position', False)]),
            'profitable_closes': len([f for f in state.fill_details if f.get('closes_position', False) and (f.get('realized_pnl') or 0) > 0]),
            'incentive': incentive
        }
        
        return incentive, diagnostics


class DrawdownPenalty(RewardComponent):
    """Penalty for large unrealized losses"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="drawdown_penalty",
            type=RewardType.SHAPING,
            description="Penalty for holding large losing positions",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check unrealized P&L
        unrealized_pnl = state.portfolio_next.get('unrealized_pnl', 0.0)
        equity = state.portfolio_next.get('total_equity', 1.0)
        
        if unrealized_pnl < 0 and equity > 0:
            # Calculate drawdown percentage
            drawdown_pct = abs(unrealized_pnl) / equity
            
            # Get thresholds
            warning_threshold = self.config.get('warning_threshold', 0.02)  # 2%
            severe_threshold = self.config.get('severe_threshold', 0.05)   # 5%
            
            if drawdown_pct > warning_threshold:
                base_penalty = self.config.get('base_penalty', 0.01)
                
                # Progressive penalty
                if drawdown_pct > severe_threshold:
                    penalty = -base_penalty * (drawdown_pct / warning_threshold) ** 2
                else:
                    penalty = -base_penalty * (drawdown_pct / warning_threshold)
                    
        diagnostics = {
            'unrealized_pnl': unrealized_pnl,
            'drawdown_pct': abs(unrealized_pnl) / equity if equity > 0 else 0,
            'penalty': penalty
        }
        
        return penalty, diagnostics


class MAEPenalty(RewardComponent):
    """Maximum Adverse Excursion penalty - penalizes high risk during trades"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="mae_penalty",
            type=RewardType.TRADE_SPECIFIC,
            description="Penalty based on maximum adverse excursion during trade",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Only calculate when closing a trade
        for fill in state.fill_details:
            if fill.get('closes_position', False):
                # Get MAE data
                mae = abs(state.current_trade_min_unrealized_pnl or 0.0)
                entry_price = state.current_trade_entry_price or fill.get('price', 0.0)
                exit_price = fill.get('price', 0.0)
                
                if entry_price > 0:
                    # Calculate MAE as percentage of entry
                    quantity = fill.get('quantity') or 0
                    mae_pct = mae / (entry_price * quantity) if quantity > 0 else 0
                    
                    # Penalty based on excessive risk
                    risk_threshold = self.config.get('mae_threshold', 0.02)  # 2%
                    
                    if mae_pct > risk_threshold:
                        base_penalty = self.config.get('base_penalty', 0.1)
                        penalty = -base_penalty * (mae_pct / risk_threshold) ** 2
                        
                        # Extra penalty if trade was ultimately losing
                        realized_pnl = fill.get('realized_pnl') or 0
                        if realized_pnl < 0:
                            penalty *= self.config.get('loss_multiplier', 1.5)
                            
        diagnostics = {
            'mae': abs(state.current_trade_min_unrealized_pnl or 0.0),
            'trades_closed': len([f for f in state.fill_details if f.get('closes_position', False)]),
            'penalty': penalty
        }
        
        return penalty, diagnostics


class MFEPenalty(RewardComponent):
    """Maximum Favorable Excursion penalty - penalizes giving back profits"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="mfe_penalty",
            type=RewardType.TRADE_SPECIFIC,
            description="Penalty for giving back profits from peak",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Only calculate when closing a trade
        for fill in state.fill_details:
            if fill.get('closes_position', False):
                # Get MFE data
                mfe = state.current_trade_max_unrealized_pnl or 0.0
                realized_pnl = fill.get('realized_pnl', 0.0)
                
                if mfe > 0:
                    # Calculate profit give-back
                    profit_captured = realized_pnl / mfe if mfe > 0 else 0
                    give_back = 1.0 - profit_captured
                    
                    # Penalty for excessive give-back
                    give_back_threshold = self.config.get('give_back_threshold', 0.5)  # 50%
                    
                    if give_back > give_back_threshold:
                        base_penalty = self.config.get('base_penalty', 0.05)
                        penalty = -base_penalty * give_back
                        
                        # Extra penalty if turned a winner into a loser
                        if realized_pnl < 0:
                            penalty *= self.config.get('reversal_multiplier', 2.0)
                            
        diagnostics = {
            'mfe': state.current_trade_max_unrealized_pnl or 0.0,
            'trades_closed': len([f for f in state.fill_details if f.get('closes_position', False)]),
            'penalty': penalty
        }
        
        return penalty, diagnostics


class TerminalPenalty(RewardComponent):
    """Penalties for terminal states (bankruptcy, max loss)"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="terminal_penalty",
            type=RewardType.TERMINAL,
            description="Large penalties for catastrophic outcomes",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        if state.terminated and state.termination_reason:
            reason = str(state.termination_reason)
            
            if 'BANKRUPTCY' in reason:
                penalty = -self.config.get('bankruptcy_penalty', 100.0)
            elif 'MAX_LOSS' in reason:
                penalty = -self.config.get('max_loss_penalty', 50.0)
            else:
                # Generic terminal penalty
                penalty = -self.config.get('default_penalty', 10.0)
                
        diagnostics = {
            'terminated': state.terminated,
            'reason': str(state.termination_reason) if state.termination_reason else None,
            'penalty': penalty
        }
        
        return penalty, diagnostics

# Aliases for test compatibility
PnLComponent = RealizedPnLReward


class MomentumAlignmentComponent(RewardComponent):
    """Reward for trading aligned with momentum patterns."""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="momentum_alignment",
            type=RewardType.SHAPING,
            description="Reward for actions aligned with momentum patterns"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """Calculate momentum alignment reward."""
        reward = 0.0
        
        # Check if we have momentum context
        momentum_context = state.metadata.get('momentum_context', {})
        if not momentum_context:
            return 0.0, {'no_context': True}
        
        # Get current phase and pattern
        current_phase = momentum_context.get('phase', 'neutral')
        current_pattern = momentum_context.get('pattern', 'unknown')
        
        # Get action taken
        action_type = state.action.get('type', 'hold')
        
        # Reward alignment
        if current_phase == 'front_side' and action_type == 'buy':
            reward = 1.0
        elif current_phase == 'back_side' and action_type == 'sell':
            reward = 1.0
        elif current_phase == 'exhaustion' and action_type == 'hold':
            reward = 0.5
        elif current_phase == 'front_side' and action_type == 'sell':
            reward = -1.0  # Penalty for counter-trend
        
        diagnostics = {
            'phase': current_phase,
            'pattern': current_pattern,
            'action': action_type,
            'alignment_reward': reward
        }
        
        return reward, diagnostics


class TimeEfficiencyComponent(RewardComponent):
    """Reward for efficient use of time in trades."""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="time_efficiency",
            type=RewardType.SHAPING,
            description="Reward for time-efficient trading"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """Calculate time efficiency reward."""
        reward = 0.0
        
        # Check holding time for closed positions
        for fill in state.fill_details:
            if fill.get('closes_position', False):
                holding_time = fill.get('holding_time_minutes', 0)
                
                # Reward quick profitable trades
                realized_pnl = fill.get('realized_pnl') or 0
                if realized_pnl > 0:
                    if holding_time < 30:  # Less than 30 minutes
                        reward += 0.5
                    elif holding_time < 60:  # Less than 1 hour
                        reward += 0.25
                else:
                    # Penalty for holding losing positions too long
                    if holding_time > 60:
                        reward -= 0.5
        
        diagnostics = {
            'trades_closed': len([f for f in state.fill_details if f.get('closes_position', False)]),
            'efficiency_reward': reward
        }
        
        return reward, diagnostics


class RiskManagementComponent(RewardComponent):
    """Reward for good risk management."""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="risk_management",
            type=RewardType.SHAPING,
            description="Reward for proper risk management"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """Calculate risk management reward."""
        reward = 0.0
        
        # Get position metrics
        position_value = state.portfolio_after_fills.get('position_value', 0)
        account_value = state.portfolio_after_fills.get('total_value', 100000)
        
        # Check position sizing
        position_pct = abs(position_value) / account_value if account_value > 0 else 0
        
        # Reward appropriate position sizing
        if 0.1 <= position_pct <= 0.3:  # 10-30% of account
            reward += 0.5
        elif position_pct > 0.5:  # Over 50% is too risky
            reward -= 1.0
        
        # Check drawdown
        current_drawdown = state.portfolio_after_fills.get('current_drawdown_pct', 0)
        if current_drawdown > 0.1:  # More than 10% drawdown
            reward -= 0.5
        
        diagnostics = {
            'position_pct': position_pct,
            'drawdown_pct': current_drawdown,
            'risk_reward': reward
        }
        
        return reward, diagnostics


class ActionCostComponent(RewardComponent):
    """Penalty for excessive actions."""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="action_cost",
            type=RewardType.PENALTY,
            description="Penalty for taking actions"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """Calculate action cost penalty."""
        penalty = 0.0
        
        # Fixed cost per action (except hold)
        action_type = state.action.get('type', 'hold')
        if action_type != 'hold':
            penalty = -self.config.get('action_cost', 0.01)
        
        # Additional cost for reversals
        portfolio_before = state.portfolio_before
        current_position = portfolio_before.get('position_side', 'FLAT')
        
        if current_position == 'LONG' and action_type == 'sell':
            penalty -= self.config.get('reversal_cost', 0.02)
        elif current_position == 'SHORT' and action_type == 'buy':
            penalty -= self.config.get('reversal_cost', 0.02)
        
        diagnostics = {
            'action': action_type,
            'action_penalty': penalty
        }
        
        return penalty, diagnostics
