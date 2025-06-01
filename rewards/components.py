# rewards/components.py - Percentage-based reward component implementations

from typing import Any, Dict, Tuple

from rewards.core import RewardComponent, RewardMetadata, RewardState, RewardType
from simulators.portfolio_simulator import PositionSideEnum


class PnLReward(RewardComponent):
    """Percentage-based P&L reward - the core reward component"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="pnl",
            type=RewardType.FOUNDATIONAL,
            description="Reward based on realized P&L as percentage of account value"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        # Calculate realized PnL change
        realized_pnl_before = state.portfolio_before.get('realized_pnl_session', 0.0)
        realized_pnl_after = state.portfolio_after_fills.get('realized_pnl_session', 0.0)
        pnl_change = realized_pnl_after - realized_pnl_before
        
        # Include transaction costs
        total_costs = 0.0
        for fill in state.fill_details:
            commission = fill.commission if fill.commission is not None else 0.0
            slippage = fill.slippage_cost_total if fill.slippage_cost_total is not None else 0.0
            total_costs += commission + slippage
        
        net_pnl_dollars = pnl_change - total_costs
        
        # Convert to percentage of account value
        account_value = state.portfolio_before.get('total_equity', 25000.0)
        pnl_percentage = (net_pnl_dollars / account_value * 100.0) if account_value > 0 else 0.0
        
        # Scale by coefficient (coefficient = reward per 1% profit)
        # Example: 1% profit with coefficient=100.0 gives 1.0 reward
        pnl_coefficient = self.config.get('pnl_coefficient', 100.0)
        reward = pnl_percentage * (pnl_coefficient / 100.0)
        
        diagnostics = {
            'net_pnl_dollars': net_pnl_dollars,
            'pnl_percentage': pnl_percentage,  # Already in percentage form
            'account_value': account_value,
            'total_costs': total_costs,
            'trades_closed': len([f for f in state.fill_details if f.closes_position])
        }
        
        return reward, diagnostics


class HoldingTimePenalty(RewardComponent):
    """Penalty for holding positions too long"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="holding_penalty",
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
        
        # Get configuration
        max_holding_time = self.config.get('max_holding_time_steps', 180)
        holding_coefficient = self.config.get('holding_penalty_coefficient', 2.0)
        
        # Calculate penalty based on excessive holding time
        holding_time = state.current_trade_duration
        penalty_ratio = 0.0
        
        if holding_time > max_holding_time:
            excess_time = holding_time - max_holding_time
            # Linear penalty: max excess time gets full coefficient penalty
            max_excess = max_holding_time  # After 2x max time, full penalty
            penalty_ratio = min(excess_time / max_excess, 1.0)
            penalty = -holding_coefficient * penalty_ratio
        
        diagnostics = {
            'holding_time': holding_time,
            'max_holding_time': max_holding_time,
            'excess_time': max(0, holding_time - max_holding_time),
            'penalty_ratio': penalty_ratio
        }
        
        return penalty, diagnostics


class DrawdownPenalty(RewardComponent):
    """Penalty for unrealized losses as percentage of account"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="drawdown_penalty",
            type=RewardType.SHAPING,
            description="Penalty for unrealized losses as percentage of account",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Get unrealized P&L
        unrealized_pnl = state.portfolio_next.get('unrealized_pnl', 0.0)
        account_value = state.portfolio_next.get('total_equity', 25000.0)
        
        if unrealized_pnl < 0 and account_value > 0:
            # Calculate drawdown as percentage of account
            drawdown_percentage = abs(unrealized_pnl) / account_value
            
            # Scale by coefficient
            drawdown_coefficient = self.config.get('drawdown_penalty_coefficient', 5.0)
            penalty = -drawdown_percentage * drawdown_coefficient
        
        diagnostics = {
            'unrealized_pnl': unrealized_pnl,
            'account_value': account_value,
            'drawdown_percentage': (abs(unrealized_pnl) / account_value * 100) if account_value > 0 and unrealized_pnl < 0 else 0
        }
        
        return penalty, diagnostics




class ProfitGivebackPenalty(RewardComponent):
    """Penalty for giving back MFE (Maximum Favorable Excursion) profits"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="profit_giveback_penalty",
            type=RewardType.SHAPING,
            description="Penalty for giving back maximum favorable excursion profits",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check if we have an open position with MFE data
        max_unrealized_pnl = state.current_trade_max_unrealized_pnl
        current_unrealized_pnl = state.portfolio_next.get('unrealized_pnl', 0.0)
        
        if max_unrealized_pnl is not None and max_unrealized_pnl > 0:
            # Calculate how much profit has been given back
            profit_giveback = max_unrealized_pnl - current_unrealized_pnl
            
            if profit_giveback > 0:
                # Calculate giveback ratio
                giveback_ratio = profit_giveback / max_unrealized_pnl
                
                # Apply penalty if significant giveback (>30%)
                giveback_threshold = self.config.get('profit_giveback_threshold', 0.3)
                if giveback_ratio > giveback_threshold:
                    excess_giveback = giveback_ratio - giveback_threshold
                    
                    # Scale penalty by account value for percentage-based approach
                    account_value = state.portfolio_next.get('total_equity', 25000.0)
                    profit_percentage = profit_giveback / account_value if account_value > 0 else 0.0
                    
                    coefficient = self.config.get('profit_giveback_penalty_coefficient', 10.0)
                    penalty = -profit_percentage * coefficient * excess_giveback
        
        diagnostics = {
            'max_unrealized_pnl': max_unrealized_pnl or 0.0,
            'current_unrealized_pnl': current_unrealized_pnl,
            'profit_giveback': max_unrealized_pnl - current_unrealized_pnl if max_unrealized_pnl else 0.0,
            'giveback_ratio': (max_unrealized_pnl - current_unrealized_pnl) / max_unrealized_pnl if max_unrealized_pnl and max_unrealized_pnl > 0 else 0.0
        }
        
        return penalty, diagnostics


class MaxDrawdownPenalty(RewardComponent):
    """Penalty for exceeding MAE (Maximum Adverse Excursion) thresholds"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="max_drawdown_penalty",
            type=RewardType.SHAPING,
            description="Penalty for exceeding maximum adverse excursion thresholds",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check if we have MAE data
        min_unrealized_pnl = state.current_trade_min_unrealized_pnl
        
        if min_unrealized_pnl is not None and min_unrealized_pnl < 0:
            # Calculate loss as percentage of account
            account_value = state.portfolio_next.get('total_equity', 25000.0)
            loss_percentage = abs(min_unrealized_pnl) / account_value if account_value > 0 else 0.0
            
            # Apply escalating penalty for larger MAE
            mae_threshold = self.config.get('max_drawdown_threshold_percent', 0.01)  # 1% of account
            
            if loss_percentage > mae_threshold:
                excess_loss = loss_percentage - mae_threshold
                coefficient = self.config.get('max_drawdown_penalty_coefficient', 15.0)
                
                # Exponential penalty for larger losses
                penalty = -coefficient * (excess_loss ** 1.5)
        
        diagnostics = {
            'min_unrealized_pnl': min_unrealized_pnl or 0.0,
            'loss_percentage': abs(min_unrealized_pnl) / state.portfolio_next.get('total_equity', 25000.0) * 100 if min_unrealized_pnl and min_unrealized_pnl < 0 else 0.0,
            'mae_exceeded': min_unrealized_pnl is not None and abs(min_unrealized_pnl) / state.portfolio_next.get('total_equity', 25000.0) > self.config.get('max_drawdown_threshold_percent', 0.01) if min_unrealized_pnl else False
        }
        
        return penalty, diagnostics


class ProfitClosingBonus(RewardComponent):
    """Bonus for closing profitable trades, scales with profit size"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="profit_closing_bonus",
            type=RewardType.SHAPING,
            description="Bonus for closing profitable trades, scales with profit size"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        bonus = 0.0
        profitable_closes = 0
        
        # Check if we closed profitable positions
        for fill in state.fill_details:
            if fill.closes_position:
                realized_pnl = fill.realized_pnl or 0.0
                
                if realized_pnl > 0:  # Profitable trade
                    profitable_closes += 1
                    
                    # Calculate bonus as percentage of account
                    account_value = state.portfolio_next.get('total_equity', 25000.0)
                    profit_percentage = realized_pnl / account_value if account_value > 0 else 0.0
                    
                    # Scale bonus with profit size (quadratic for larger profits)
                    coefficient = self.config.get('profit_closing_bonus_coefficient', 5.0)
                    bonus += coefficient * (profit_percentage ** 1.2)  # Slightly superlinear
        
        diagnostics = {
            'trades_closed': len([f for f in state.fill_details if f.closes_position]),
            'profitable_closes': profitable_closes,
            'total_bonus': bonus
        }
        
        return bonus, diagnostics



class BankruptcyPenalty(RewardComponent):
    """Large penalty for bankruptcy or extreme losses"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="bankruptcy_penalty",
            type=RewardType.TERMINAL,
            description="Large penalty for bankruptcy or extreme losses",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check for bankruptcy condition
        if state.terminated and state.termination_reason:
            reason = str(state.termination_reason)
            
            if 'BANKRUPTCY' in reason:
                bankruptcy_coefficient = self.config.get('bankruptcy_penalty_coefficient', 50.0)
                penalty = -bankruptcy_coefficient
        
        diagnostics = {
            'terminated': state.terminated,
            'reason': str(state.termination_reason) if state.termination_reason else None,
            'is_bankruptcy': 'BANKRUPTCY' in str(state.termination_reason) if state.termination_reason else False
        }
        
        return penalty, diagnostics