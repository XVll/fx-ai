# rewards/calculator.py - Main reward system calculator with full integration

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
import numpy as np

from config.schemas import RewardConfig
from rewards.core import RewardAggregator, RewardState
from rewards.components import (
    RealizedPnLReward,
    MarkToMarketReward, 
    DifferentialSharpeReward,
    HoldingTimePenalty,
    OvertradingPenalty,
    QuickProfitIncentive,
    DrawdownPenalty,
    MAEPenalty,
    MFEPenalty,
    TerminalPenalty
)
from rewards.metrics import RewardMetricsTracker
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
    Advanced reward system with comprehensive metrics and anti-hacking measures
    """
    

    def __init__(self, config: RewardConfig, metrics_integrator=None, logger: Optional[logging.Logger] = None):
        self.reward_config = config

        self.metrics_integrator = metrics_integrator
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.components = self._initialize_components()
        

        # Initialize aggregator with default config
        aggregator_config = {
            'global_scale': self.reward_config.scale_factor,
            'clip_min': self.reward_config.clip_range[0],
            'clip_max': self.reward_config.clip_range[1],
            'use_smoothing': False,  # DISABLED: Smoothing causes sign inversions for losses
            'smoothing_window': 10
        }

        self.aggregator = RewardAggregator(self.components, aggregator_config, self.logger)
        
        # Initialize metrics tracker
        self.metrics_tracker = RewardMetricsTracker(self.logger)
        for component in self.components:
            self.metrics_tracker.register_component(
                component.metadata.name,
                component.metadata.type.value
            )
        
        # Register special components
        self.metrics_tracker.register_component(
            'position_reset_penalty',
            'special'
        )
        
        # Trade tracking
        self.current_trade: Optional[TradeTracker] = None
        self.step_count = 0
        
        # Episode tracking
        self.episode_started = False

    def _initialize_components(self) -> List:

        """Initialize reward components based on Pydantic config"""
        components = []
        
        # Initialize PnL component
        if self.reward_config.pnl.enabled:
            pnl_config = {
                'enabled': True,
                'weight': self.reward_config.pnl.coefficient,
                'clip_min': self.reward_config.clip_range[0],
                'clip_max': self.reward_config.clip_range[1]
            }
            components.append(RealizedPnLReward(pnl_config, self.logger))
            self.logger.info("Initialized PnL reward component")
            
        # Initialize holding penalty
        if self.reward_config.holding_penalty.enabled:
            holding_config = {
                'enabled': True,
                'weight': self.reward_config.holding_penalty.coefficient,
                'penalty_per_step': 0.001,
                'max_holding_steps': 300
            }
            components.append(HoldingTimePenalty(holding_config, self.logger))
            self.logger.info("Initialized holding penalty component")
            
        # Initialize action penalty (overtrading)
        if self.reward_config.action_penalty.enabled:
            overtrading_config = {
                'enabled': True,
                'weight': self.reward_config.action_penalty.coefficient,
                'penalty_per_action': 0.0001,
                'window_size': 20
            }
            components.append(OvertradingPenalty(overtrading_config, self.logger))
            self.logger.info("Initialized overtrading penalty component")
            
        # Initialize drawdown penalty
        if self.reward_config.drawdown_penalty.enabled:
            drawdown_config = {
                'enabled': True,
                'weight': self.reward_config.drawdown_penalty.coefficient,
                'drawdown_threshold': 0.05,
                'penalty_scale': 1.0
            }
            components.append(DrawdownPenalty(drawdown_config, self.logger))
            self.logger.info("Initialized drawdown penalty component")
            
        # Initialize bankruptcy penalty (terminal penalty)
        if self.reward_config.bankruptcy_penalty.enabled:
            terminal_config = {
                'enabled': True,
                'weight': self.reward_config.bankruptcy_penalty.coefficient,
                'bankruptcy_penalty': 100.0,
                'max_loss_penalty': 50.0,
                'default_penalty': 10.0
            }
            components.append(TerminalPenalty(terminal_config, self.logger))
            self.logger.info("Initialized terminal penalty component")
            
        # Initialize profitable exit bonus
        if self.reward_config.profitable_exit.enabled:
            profit_config = {
                'enabled': True,
                'weight': self.reward_config.profitable_exit.coefficient,
                'min_profit_threshold': 0.001,
                'max_incentive': 1.0
            }
            components.append(QuickProfitIncentive(profit_config, self.logger))
            self.logger.info("Initialized quick profit incentive component")
            
        # Initialize spread penalty 
        if self.reward_config.spread_penalty.enabled:
            spread_config = {
                'enabled': True,
                'weight': self.reward_config.spread_penalty.coefficient,
                'penalty_per_trade': 0.001
            }
            # Note: SpreadPenalty component needs to be implemented
            # components.append(SpreadPenalty(spread_config, self.logger))
            self.logger.info("Spread penalty component not yet implemented")
            
        # Initialize quick profit bonus
        if self.reward_config.quick_profit.enabled:
            quick_profit_config = {
                'enabled': True,
                'weight': self.reward_config.quick_profit.coefficient,
                'quick_profit_time': 30,
                'bonus_rate': 0.5
            }
            # QuickProfitIncentive already added above as profitable_exit
            # This is a duplicate - keeping only one
            # components.append(QuickProfitIncentive(quick_profit_config, self.logger))
            self.logger.info("Quick profit incentive already added as profitable_exit")
            
        # Initialize invalid action penalty
        if self.reward_config.invalid_action_penalty.enabled:
            invalid_config = {
                'enabled': True,
                'weight': self.reward_config.invalid_action_penalty.coefficient,
                'penalty_per_invalid': 0.01
            }
            # Note: InvalidActionPenalty component needs to be implemented
            # components.append(InvalidActionPenalty(invalid_config, self.logger))
            self.logger.info("Invalid action penalty component not yet implemented")
            
        # Initialize MAE penalty
        mae_config = {
            'enabled': True,
            'weight': 1.0,
            'mae_threshold': 0.02,
            'base_penalty': 0.1,
            'loss_multiplier': 1.5
        }
        components.append(MAEPenalty(mae_config, self.logger))
        self.logger.info("Initialized MAE penalty component")
        
        # Initialize MFE penalty
        mfe_config = {
            'enabled': True,
            'weight': 1.0,
            'give_back_threshold': 0.5,
            'base_penalty': 0.05,
            'reversal_multiplier': 2.0
        }
        components.append(MFEPenalty(mfe_config, self.logger))
        self.logger.info("Initialized MFE penalty component")
        
        # Initialize Mark-to-Market reward
        mtm_config = {
            'enabled': True,
            'weight': 0.5,
            'max_leverage': 2.0
        }
        components.append(MarkToMarketReward(mtm_config, self.logger))
        self.logger.info("Initialized Mark-to-Market reward component")

        return components
    
    def reset(self):
        """Reset for new episode"""
        self.step_count = 0
        self.current_trade = None
        self.episode_started = True
        
        # Reset aggregator statistics
        self.aggregator.reset_statistics()
        
        # Don't reset metrics tracker - we want to track across episodes
        
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
                self.logger.debug(f"New trade opened at price {avg_entry_price}")
        else:
            # Position closed
            if self.current_trade is not None:
                self.current_trade = None
                self.logger.debug("Trade closed")
                
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
        Calculate total reward and update metrics
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
            episode_trades=self.metrics_tracker.episode_trades,
            current_trade_entry_price=self.current_trade.entry_price if self.current_trade else None,
            current_trade_max_unrealized_pnl=self.current_trade.max_unrealized_pnl if self.current_trade else None,
            current_trade_min_unrealized_pnl=self.current_trade.min_unrealized_pnl if self.current_trade else None,
            current_trade_duration=trade_duration
        )
        
        # Calculate total reward
        total_reward, all_diagnostics = self.aggregator.calculate_total_reward(state)
        
        # Update trade tracking
        self._update_trade_tracking(state)
        
        # Update metrics
        action_type = decoded_action.get('type')
        action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)
        
        # Check if this was a profitable trade
        is_profitable_trade = None
        if fill_details_list:
            for fill in fill_details_list:
                if fill.closes_position:
                    realized_pnl = fill.realized_pnl or 0
                    is_profitable_trade = realized_pnl > 0
                    
        # Update component metrics
        for comp_name, comp_diagnostics in all_diagnostics.items():
            if comp_name != 'summary' and 'final_value' in comp_diagnostics:
                self.metrics_tracker.update_component(
                    comp_name,
                    comp_diagnostics['final_value'],
                    comp_diagnostics,
                    self.step_count,
                    action_name,
                    is_profitable_trade or False
                )
                
        # Update step metrics
        self.metrics_tracker.update_step(
            total_reward,
            action_name,
            bool(fill_details_list),
            is_profitable_trade
        )
        
        # Send to external metrics integrator if available
        if self.metrics_integrator:
            self.metrics_integrator.record_environment_step(
                reward=total_reward,
                action=action_name,
                is_invalid=bool(decoded_action.get('invalid_reason')),
                reward_components=all_diagnostics.get('summary', {}).get('component_rewards', {})
            )
            
            # Send detailed diagnostics
            for comp_name, comp_diagnostics in all_diagnostics.items():
                if comp_name != 'summary':
                    self.metrics_integrator.record_custom_metrics({
                        f'reward_component/{comp_name}/value': comp_diagnostics.get('final_value', 0),
                        f'reward_component/{comp_name}/triggered': comp_diagnostics.get('enabled', True) and comp_diagnostics.get('final_value', 0) != 0
                    })
                    
        # Log significant events (disabled to reduce spam)
        # if abs(total_reward) > 0.01 or fill_details_list or terminated:
        #     components_str = ', '.join([f"{k}:{v:.3f}" for k, v in all_diagnostics.get('summary', {}).get('component_rewards', {}).items() if v != 0])
        #     self.logger.info(f"Step {self.step_count}: {action_name}, Reward: {total_reward:.4f} [{components_str}]")
            
        self.step_count += 1
        
        return total_reward
    
    def get_last_reward_components(self) -> Dict[str, float]:
        """Get the last reward components for compatibility with existing code"""
        if hasattr(self.aggregator, '_last_component_rewards'):
            return self.aggregator._last_component_rewards.copy()
        return {}
    
    def get_episode_summary(self, final_portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Get comprehensive episode summary"""
        # Get episode metrics
        episode_summary = self.metrics_tracker.end_episode(final_portfolio_state)
        
        # Add aggregator statistics
        component_stats = self.aggregator.get_component_statistics()
        episode_summary['aggregator_statistics'] = component_stats
        
        # Add analysis
        if self.metrics_tracker.total_episodes >= 10:
            episode_summary['component_analysis'] = self.metrics_tracker.get_component_analysis()
            episode_summary['correlation_analysis'] = self.metrics_tracker.get_correlation_analysis()
            
        return episode_summary
    
    def get_metrics_for_dashboard(self) -> Dict[str, Any]:
        """Get current metrics formatted for dashboard display"""
        current_stats = {}
        
        # Get latest component values
        for comp_name, comp_metrics in self.metrics_tracker.component_metrics.items():
            if comp_metrics.values:
                current_stats[comp_name] = {
                    'last_value': comp_metrics.values[-1],
                    'mean_value': np.mean(comp_metrics.values),
                    'trigger_rate': comp_metrics.count_triggered / max(1, self.step_count)
                }
                
        return {
            'current_step': self.step_count,
            'episode_reward': self.metrics_tracker.episode_total_reward,
            'component_stats': current_stats,
            'episode_trades': self.metrics_tracker.episode_trades,
            'win_rate': self.metrics_tracker.episode_profitable_trades / max(1, self.metrics_tracker.episode_trades)
        }
    
    def apply_position_close_penalty(self, close_pnl: float):
        """Apply a penalty/reward for positions that are open at episode reset.
        
        This ensures the model experiences the consequences of holding positions
        when episodes end, preventing it from gaming the system by holding losers
        knowing they'll disappear.
        
        Args:
            close_pnl: The P&L that would be realized if the position was closed
        """
        # Scale the P&L to match reward scaling
        # Use a scaling factor that's significant but not overwhelming
        scaled_penalty = close_pnl * 0.01  # 1% of P&L as reward impact
        
        # Track this as a special component
        if hasattr(self, 'metrics_tracker'):
            self.metrics_tracker.update_component(
                'position_reset_penalty',
                scaled_penalty,
                {'close_pnl': close_pnl, 'scaled_value': scaled_penalty},
                self.step_count,
                'RESET',
                close_pnl > 0
            )
            
        self.logger.info(f"Applied position reset penalty: P&L ${close_pnl:+.2f} → Reward impact {scaled_penalty:+.4f}")
        
        # Store this for the next calculate() call to include
        self._pending_reset_penalty = scaled_penalty
    
    def get_wandb_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for W&B logging"""
        metrics = {}
        
        # Component statistics
        for comp_name, comp_metrics in self.metrics_tracker.component_metrics.items():
            if comp_metrics.values:
                metrics[f'reward/{comp_name}/mean'] = np.mean(comp_metrics.values)
                metrics[f'reward/{comp_name}/std'] = np.std(comp_metrics.values)
                metrics[f'reward/{comp_name}/trigger_rate'] = comp_metrics.count_triggered / max(1, self.step_count)
                
        # Episode statistics
        if self.metrics_tracker.episode_metrics:
            latest_episode = self.metrics_tracker.episode_metrics[-1]
            metrics['episode/total_reward'] = latest_episode['total_reward']
            metrics['episode/mean_reward'] = latest_episode['mean_reward_per_step']
            metrics['episode/trades'] = latest_episode['total_trades']
            metrics['episode/win_rate'] = latest_episode['win_rate']
            
            # Component dominance
            if latest_episode.get('dominant_positive_component'):
                metrics['episode/dominant_positive'] = latest_episode['dominant_positive_component']['name']
                metrics['episode/dominant_positive_value'] = latest_episode['dominant_positive_component']['total']
                
            if latest_episode.get('dominant_negative_component'):
                metrics['episode/dominant_negative'] = latest_episode['dominant_negative_component']['name']
                metrics['episode/dominant_negative_value'] = latest_episode['dominant_negative_component']['total']
                
        return metrics
    
    def save_config(self, path: str):
        """Save current reward configuration"""
        # Convert Pydantic config to dict for saving
        config_data = {
            'reward_system': 'v2',
            'components': {
                'pnl': {
                    'enabled': self.reward_config.pnl.enabled,
                    'coefficient': self.reward_config.pnl.coefficient
                },
                'holding_penalty': {
                    'enabled': self.reward_config.holding_penalty.enabled,
                    'coefficient': self.reward_config.holding_penalty.coefficient
                },
                'action_penalty': {
                    'enabled': self.reward_config.action_penalty.enabled,
                    'coefficient': self.reward_config.action_penalty.coefficient
                },
                'spread_penalty': {
                    'enabled': self.reward_config.spread_penalty.enabled,
                    'coefficient': self.reward_config.spread_penalty.coefficient
                },
                'drawdown_penalty': {
                    'enabled': self.reward_config.drawdown_penalty.enabled,
                    'coefficient': self.reward_config.drawdown_penalty.coefficient
                },
                'bankruptcy_penalty': {
                    'enabled': self.reward_config.bankruptcy_penalty.enabled,
                    'coefficient': self.reward_config.bankruptcy_penalty.coefficient
                },
                'profitable_exit': {
                    'enabled': self.reward_config.profitable_exit.enabled,
                    'coefficient': self.reward_config.profitable_exit.coefficient
                },
                'quick_profit': {
                    'enabled': self.reward_config.quick_profit.enabled,
                    'coefficient': self.reward_config.quick_profit.coefficient
                },
                'invalid_action_penalty': {
                    'enabled': self.reward_config.invalid_action_penalty.enabled,
                    'coefficient': self.reward_config.invalid_action_penalty.coefficient
                }
            },
            'global_settings': {
                'scale_factor': self.reward_config.scale_factor,
                'clip_range': self.reward_config.clip_range
            },
            'metrics': {
                'total_episodes': self.metrics_tracker.total_episodes,
                'total_steps': self.metrics_tracker.total_steps
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
        self.logger.info(f"Saved reward configuration to {path}")