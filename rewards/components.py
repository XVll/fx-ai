# rewards/components.py - Percentage-based reward component implementations

from typing import Any, Dict, Tuple
from collections import deque

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
        max_holding_time = self.config.get('max_holding_time_steps', 60)
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


class ActionPenalty(RewardComponent):
    """Penalty for excessive trading actions"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.action_count = 0
        self.action_history = deque(maxlen=100)  # Track last 100 actions
        
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="action_penalty",
            type=RewardType.SHAPING,
            description="Penalty for excessive trading frequency",
            is_penalty=True
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        
        # Check if a trading action was taken (not HOLD)
        action_type = state.decoded_action.get('type', 'hold')
        is_trading_action = action_type not in ['hold', 'HOLD']
        
        if is_trading_action:
            self.action_count += 1
            self.action_history.append(state.step_count)
            
            # Calculate recent action frequency
            recent_window = 50  # Look at last 50 steps
            recent_actions = sum(1 for step in self.action_history 
                               if state.step_count - step <= recent_window)
            
            # Penalty if too many actions in recent window
            max_actions_in_window = 5  # Max 5 actions per 50 steps
            if recent_actions > max_actions_in_window:
                excess_actions = recent_actions - max_actions_in_window
                action_coefficient = self.config.get('action_penalty_coefficient', 1.0)
                penalty = -action_coefficient * (excess_actions / max_actions_in_window)
        
        diagnostics = {
            'is_trading_action': is_trading_action,
            'total_actions': self.action_count,
            'recent_actions': len([s for s in self.action_history if state.step_count - s <= 50]),
            'action_type': str(action_type)
        }
        
        return penalty, diagnostics


class QuickProfitBonus(RewardComponent):
    """Bonus for taking profits quickly"""
    
    def _get_metadata(self) -> RewardMetadata:
        return RewardMetadata(
            name="quick_profit_bonus",
            type=RewardType.SHAPING,
            description="Bonus for capturing profits quickly"
        )
    
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        bonus = 0.0
        
        # Check if we closed a profitable position
        for fill in state.fill_details:
            if fill.closes_position:
                realized_pnl = fill.realized_pnl or 0.0
                
                if realized_pnl > 0:  # Profitable trade
                    # Check if it was quick
                    quick_time_threshold = self.config.get('quick_profit_time_threshold', 30)
                    
                    if state.current_trade_duration <= quick_time_threshold:
                        # Calculate bonus as percentage of account
                        account_value = state.portfolio_next.get('total_equity', 25000.0)
                        profit_percentage = realized_pnl / account_value if account_value > 0 else 0.0
                        
                        # Scale by coefficient and time factor
                        quick_coefficient = self.config.get('quick_profit_bonus_coefficient', 10.0)
                        time_factor = 1.0 - (state.current_trade_duration / quick_time_threshold)
                        bonus += profit_percentage * quick_coefficient * time_factor
        
        diagnostics = {
            'trades_closed': len([f for f in state.fill_details if f.closes_position]),
            'profitable_closes': len([f for f in state.fill_details 
                                    if f.closes_position and (f.realized_pnl or 0) > 0]),
            'trade_duration': state.current_trade_duration,
            'bonus_awarded': bonus > 0
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