# rewards/calculator.py - Clean percentage-based reward system

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.schemas import RewardConfig
from rewards.core import RewardState
from rewards.components import (
    PnLReward,
    HoldingTimePenalty,
    DrawdownPenalty,
    ActionPenalty,
    QuickProfitBonus,
    BankruptcyPenalty
)
from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum


@dataclass
class TradeTracker:
    """Tracks information about the current trade"""
    entry_price: float
    entry_step: int
    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0
    
    def update(self, unrealized_pnl: float):
        """Update MAE/MFE tracking"""
        self.max_unrealized_pnl = max(self.max_unrealized_pnl, unrealized_pnl)
        self.min_unrealized_pnl = min(self.min_unrealized_pnl, unrealized_pnl)


class RewardSystem:
    """
    Clean percentage-based reward system without clipping or smoothing
    """
    
    def __init__(self, config: RewardConfig, metrics_integrator=None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.metrics_integrator = metrics_integrator
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components based on new config
        self.components = self._initialize_components()
        
        # Simple tracking - no complex aggregator
        self.step_count = 0
        self.current_trade: Optional[TradeTracker] = None
        self.episode_total_reward = 0.0
        self.component_totals = {}
        
        # Component tracking for dashboard
        self._last_component_rewards = {}
        
        self.logger.info(f"Initialized percentage-based reward system with {len(self.components)} components")

    def _initialize_components(self) -> List:
        """Initialize reward components based on configuration"""
        components = []
        
        # Core P&L reward (always enabled if flag is True)
        if self.config.enable_pnl_reward:
            pnl_config = {
                'pnl_coefficient': self.config.pnl_coefficient
            }
            components.append(PnLReward(pnl_config, self.logger))
            self.logger.info(f"Enabled P&L reward (coefficient: {self.config.pnl_coefficient})")
        
        # Holding time penalty
        if self.config.enable_holding_penalty:
            holding_config = {
                'holding_penalty_coefficient': self.config.holding_penalty_coefficient,
                'max_holding_time_steps': self.config.max_holding_time_steps
            }
            components.append(HoldingTimePenalty(holding_config, self.logger))
            self.logger.info(f"Enabled holding penalty (coefficient: {self.config.holding_penalty_coefficient})")
        
        # Drawdown penalty
        if self.config.enable_drawdown_penalty:
            drawdown_config = {
                'drawdown_penalty_coefficient': self.config.drawdown_penalty_coefficient
            }
            components.append(DrawdownPenalty(drawdown_config, self.logger))
            self.logger.info(f"Enabled drawdown penalty (coefficient: {self.config.drawdown_penalty_coefficient})")
        
        # Action frequency penalty
        if self.config.enable_action_penalty:
            action_config = {
                'action_penalty_coefficient': self.config.action_penalty_coefficient
            }
            components.append(ActionPenalty(action_config, self.logger))
            self.logger.info(f"Enabled action penalty (coefficient: {self.config.action_penalty_coefficient})")
        
        # Quick profit bonus
        if self.config.enable_quick_profit_bonus:
            quick_config = {
                'quick_profit_bonus_coefficient': self.config.quick_profit_bonus_coefficient,
                'quick_profit_time_threshold': self.config.quick_profit_time_threshold
            }
            components.append(QuickProfitBonus(quick_config, self.logger))
            self.logger.info(f"Enabled quick profit bonus (coefficient: {self.config.quick_profit_bonus_coefficient})")
        
        # Bankruptcy penalty (always enabled - safety mechanism)
        bankruptcy_config = {
            'bankruptcy_penalty_coefficient': self.config.bankruptcy_penalty_coefficient
        }
        components.append(BankruptcyPenalty(bankruptcy_config, self.logger))
        self.logger.info(f"Enabled bankruptcy penalty (coefficient: {self.config.bankruptcy_penalty_coefficient})")
        
        return components
    
    def reset(self):
        """Reset for new episode"""
        self.step_count = 0
        self.current_trade = None
        self.episode_total_reward = 0.0
        self.component_totals = {}
        
    def _update_trade_tracking(self, state: RewardState):
        """Update trade tracking for MAE/MFE"""
        position_side = state.portfolio_next.get('position_side')
        
        # Check if we opened a new position
        if position_side and position_side != PositionSideEnum.FLAT:
            if self.current_trade is None:
                # New trade opened
                avg_entry_price = state.portfolio_next.get('avg_entry_price', 0.0)
                self.current_trade = TradeTracker(
                    entry_price=avg_entry_price,
                    entry_step=self.step_count
                )
        else:
            # Position closed
            if self.current_trade is not None:
                self.current_trade = None
                
        # Update MAE/MFE if in trade
        if self.current_trade is not None:
            unrealized_pnl = state.portfolio_next.get('unrealized_pnl', 0.0)
            self.current_trade.update(unrealized_pnl)
    
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
                  termination_reason: Optional[Any]) -> float:
        """
        Calculate total reward without clipping or smoothing
        """
        
        # Create reward state
        trade_duration = 0
        if self.current_trade:
            trade_duration = self.step_count - self.current_trade.entry_step
            
        state = RewardState(
            portfolio_before=portfolio_state_before_action,
            portfolio_after_fills=portfolio_state_after_action_fills,
            portfolio_next=portfolio_state_next_t,
            market_state_current=market_state_at_decision,
            market_state_next=market_state_next_t,
            decoded_action=decoded_action,
            fill_details=fill_details_list,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            step_count=self.step_count,
            episode_trades=len(fill_details_list),
            current_trade_entry_price=self.current_trade.entry_price if self.current_trade else None,
            current_trade_max_unrealized_pnl=self.current_trade.max_unrealized_pnl if self.current_trade else None,
            current_trade_min_unrealized_pnl=self.current_trade.min_unrealized_pnl if self.current_trade else None,
            current_trade_duration=trade_duration
        )
        
        # Calculate component rewards
        total_reward = 0.0
        component_rewards = {}
        
        for component in self.components:
            reward_value, diagnostics = component(state)
            total_reward += reward_value
            component_rewards[component.metadata.name] = reward_value
            
            # Track component totals
            if component.metadata.name not in self.component_totals:
                self.component_totals[component.metadata.name] = 0.0
            self.component_totals[component.metadata.name] += reward_value
        
        # Store last component rewards for access
        self._last_component_rewards = component_rewards.copy()
        
        # Update trade tracking
        self._update_trade_tracking(state)
        
        # Update episode total
        self.episode_total_reward += total_reward
        
        # Send to external metrics integrator if available
        if self.metrics_integrator:
            self.metrics_integrator.record_environment_step(
                reward=total_reward,
                action=str(decoded_action.get('type', 'unknown')),
                is_invalid=bool(decoded_action.get('invalid_reason')),
                reward_components=component_rewards
            )
        
        self.step_count += 1
        return total_reward
    
    def get_last_reward_components(self) -> Dict[str, float]:
        """Get the last reward components for compatibility"""
        return self._last_component_rewards.copy()
    
    def get_episode_summary(self, final_portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Get episode summary"""
        account_value = final_portfolio_state.get('total_equity', 25000.0)
        realized_pnl = final_portfolio_state.get('realized_pnl_session', 0.0)
        
        return {
            'total_reward': self.episode_total_reward,
            'component_totals': self.component_totals.copy(),
            'final_account_value': account_value,
            'realized_pnl_dollars': realized_pnl,
            'realized_pnl_percentage': (realized_pnl / 25000.0) * 100,  # Assuming 25k initial
            'total_steps': self.step_count
        }
    
    def get_metrics_for_dashboard(self) -> Dict[str, Any]:
        """Get current metrics formatted for dashboard display"""
        return {
            'episode_reward': self.episode_total_reward,
            'component_rewards': self._last_component_rewards.copy(),
            'component_totals': self.component_totals.copy(),
            'current_step': self.step_count
        }
    
    def get_wandb_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for W&B logging"""
        metrics = {}
        
        # Component totals
        for comp_name, total_value in self.component_totals.items():
            metrics[f'reward/{comp_name}/total'] = total_value
            
        # Episode statistics
        metrics['episode/total_reward'] = self.episode_total_reward
        metrics['episode/steps'] = self.step_count
        
        return metrics