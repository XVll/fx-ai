# rewards/core.py - Core reward system architecture

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from simulators.portfolio_simulator import PortfolioState, FillDetails, PositionSideEnum


class RewardType(Enum):
    """Types of reward components"""
    FOUNDATIONAL = "foundational"
    SHAPING = "shaping"
    TERMINAL = "terminal"
    TRADE_SPECIFIC = "trade_specific"


@dataclass
class RewardMetadata:
    """Metadata for a reward component"""
    name: str
    type: RewardType
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_penalty: bool = False


@dataclass
class RewardState:
    """State information passed to reward components"""
    portfolio_before: PortfolioState
    portfolio_after_fills: PortfolioState
    portfolio_next: PortfolioState
    market_state_current: Dict[str, Any]
    market_state_next: Optional[Dict[str, Any]]
    decoded_action: Dict[str, Any]
    fill_details: List[FillDetails]
    terminated: bool
    truncated: bool
    termination_reason: Optional[Any] = None
    step_count: int = 0
    episode_trades: int = 0
    
    # Trade-specific tracking (for MAE/MFE)
    current_trade_entry_price: Optional[float] = None
    current_trade_max_unrealized_pnl: Optional[float] = None
    current_trade_min_unrealized_pnl: Optional[float] = None
    current_trade_duration: int = 0


class RewardComponent(ABC):
    """Base class for all reward components"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.metadata = self._get_metadata()
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 1.0)
        
        # Anti-hacking measures
        self.clip_min = config.get('clip_min', None)
        self.clip_max = config.get('clip_max', None)
        self.exponential_decay = config.get('exponential_decay', None)
        
    @abstractmethod
    def _get_metadata(self) -> RewardMetadata:
        """Return metadata for this reward component"""
        pass
    
    @abstractmethod
    def calculate(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward value and return diagnostics
        Returns: (reward_value, diagnostics_dict)
        """
        pass
    
    def apply_anti_hacking_measures(self, reward: float, state: RewardState) -> float:
        """Apply measures to prevent reward hacking"""
        # Clipping
        if self.clip_min is not None and reward < self.clip_min:
            reward = self.clip_min
        if self.clip_max is not None and reward > self.clip_max:
            reward = self.clip_max
            
        # Exponential decay for repeated behaviors
        if self.exponential_decay is not None and state.step_count > 0:
            decay_factor = np.exp(-self.exponential_decay * state.step_count / 1000)
            reward *= decay_factor
            
        return reward
    
    def __call__(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """Main entry point for calculating reward"""
        if not self.enabled:
            return 0.0, {'enabled': False}
            
        reward, diagnostics = self.calculate(state)
        reward = self.apply_anti_hacking_measures(reward, state)
        reward *= self.weight
        
        diagnostics['weight'] = self.weight
        diagnostics['final_value'] = reward
        
        return reward, diagnostics


class RewardAggregator:
    """Aggregates multiple reward components with anti-hacking measures"""
    
    def __init__(self, components: List[RewardComponent], config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        self.components = components
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Global scaling
        self.global_scale = config.get('global_scale', 1.0)
        
        # Anti-hacking: reward smoothing
        self.use_smoothing = config.get('use_smoothing', False)
        self.smoothing_window = config.get('smoothing_window', 10)
        self.reward_history = []
        
        # Component tracking
        self.component_stats = {comp.metadata.name: {
            'total': 0.0,
            'count': 0,
            'min': float('inf'),
            'max': float('-inf')
        } for comp in components}
        
        # Store last component rewards for access
        self._last_component_rewards = {}
        
    def calculate_total_reward(self, state: RewardState) -> Tuple[float, Dict[str, Any]]:
        """Calculate total reward from all components"""
        total_reward = 0.0
        all_diagnostics = {}
        component_rewards = {}
        
        # Calculate each component
        for component in self.components:
            reward, diagnostics = component(state)
            total_reward += reward
            
            # Track statistics
            comp_name = component.metadata.name
            self.component_stats[comp_name]['total'] += reward
            self.component_stats[comp_name]['count'] += 1
            self.component_stats[comp_name]['min'] = min(self.component_stats[comp_name]['min'], reward)
            self.component_stats[comp_name]['max'] = max(self.component_stats[comp_name]['max'], reward)
            
            component_rewards[comp_name] = reward
            all_diagnostics[comp_name] = diagnostics
            
        # Store last component rewards
        self._last_component_rewards = component_rewards.copy()
        
        # Apply global scaling
        total_reward *= self.global_scale
        
        # Apply smoothing if enabled
        if self.use_smoothing:
            self.reward_history.append(total_reward)
            if len(self.reward_history) > self.smoothing_window:
                self.reward_history.pop(0)
            smoothed_reward = np.mean(self.reward_history)
            all_diagnostics['smoothing'] = {
                'original': total_reward,
                'smoothed': smoothed_reward,
                'window_size': len(self.reward_history)
            }
            total_reward = smoothed_reward
            
        # Add summary diagnostics
        all_diagnostics['summary'] = {
            'total_reward': total_reward,
            'component_rewards': component_rewards,
            'global_scale': self.global_scale
        }
        
        return total_reward, all_diagnostics
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all components"""
        stats = {}
        for comp_name, comp_stats in self.component_stats.items():
            if comp_stats['count'] > 0:
                stats[comp_name] = {
                    'mean': comp_stats['total'] / comp_stats['count'],
                    'total': comp_stats['total'],
                    'count': comp_stats['count'],
                    'min': comp_stats['min'],
                    'max': comp_stats['max']
                }
        return stats
    
    def reset_statistics(self):
        """Reset component statistics for new episode"""
        for comp_name in self.component_stats:
            self.component_stats[comp_name] = {
                'total': 0.0,
                'count': 0,
                'min': float('inf'),
                'max': float('-inf')
            }
        self.reward_history.clear()