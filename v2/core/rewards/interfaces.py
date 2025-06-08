"""
Reward system interfaces for reinforcement learning.

These interfaces enable modular reward design with clear
components that can be combined and tuned independently.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable
from datetime import datetime
import numpy as np

from ..types.common import (
    Reward, PnL, ActionType, PositionSide,
    Configurable, Resettable
)


@runtime_checkable
class IRewardComponent(Protocol):
    """Base interface for individual reward components.
    
    Design principles:
    - Each component calculates one aspect of reward
    - Components are composable
    - Clear, interpretable contribution
    - Bounded output range
    """
    
    @property
    def name(self) -> str:
        """Component name.
        
        Returns:
            Unique identifier
        """
        ...
    
    @property
    def weight(self) -> float:
        """Component weight in total reward.
        
        Returns:
            Weight value (can be negative)
            
        Design notes:
        - Weights are relative
        - Can be tuned via hyperparameter search
        """
        ...
    
    @property
    def bounds(self) -> tuple[float, float]:
        """Expected output bounds.
        
        Returns:
            Tuple of (min, max) values
            
        Design notes:
        - Used for normalization
        - Helps with debugging
        """
        ...
    
    def calculate(
        self,
        state_before: dict[str, Any],
        action: dict[str, Any],
        state_after: dict[str, Any],
        done: bool
    ) -> float:
        """Calculate component reward.
        
        Args:
            state_before: State before action
            action: Action taken
            state_after: State after action
            done: Episode terminated
            
        Returns:
            Component reward value
            
        Design notes:
        - Access only needed state elements
        - Handle edge cases gracefully
        - Consider computational efficiency
        """
        ...
    
    def get_info(self) -> dict[str, Any]:
        """Get component information.
        
        Returns:
            Dict with:
            - description: What it rewards/penalizes
            - rationale: Why it's important
            - tuning_hints: How to adjust
        """
        ...


class IPnLComponent(IRewardComponent):
    """Interface for P&L-based reward component.
    
    Design principles:
    - Core component for trading
    - Handle realized and unrealized P&L
    - Scale appropriately
    """
    
    @abstractmethod
    def set_pnl_scaling(
        self,
        scaling_method: str,
        parameters: dict[str, float]
    ) -> None:
        """Configure P&L scaling.
        
        Args:
            scaling_method: "linear", "tanh", "log", etc.
            parameters: Method-specific parameters
            
        Design notes:
        - Prevent extreme values
        - Maintain sensitivity
        """
        ...


class IActionPenaltyComponent(IRewardComponent):
    """Interface for action penalty component.
    
    Design principles:
    - Discourage excessive trading
    - Penalize invalid actions
    - Encourage efficient execution
    """
    
    @abstractmethod
    def set_penalty_rates(
        self,
        trade_penalty: float,
        invalid_penalty: float,
        pattern_penalty: float
    ) -> None:
        """Configure penalty rates.
        
        Args:
            trade_penalty: Per-trade penalty
            invalid_penalty: Invalid action penalty
            pattern_penalty: Bad pattern penalty
        """
        ...


class IRiskComponent(IRewardComponent):
    """Interface for risk-based reward component.
    
    Design principles:
    - Encourage risk management
    - Penalize excessive exposure
    - Reward prudent position sizing
    """
    
    @abstractmethod
    def set_risk_limits(
        self,
        max_position_size: float,
        max_drawdown: float,
        max_leverage: float
    ) -> None:
        """Configure risk limits.
        
        Args:
            max_position_size: Maximum position
            max_drawdown: Maximum drawdown
            max_leverage: Maximum leverage
        """
        ...


class IRewardCalculator(Configurable, Resettable):
    """Main reward calculation interface.
    
    Design principles:
    - Combine multiple components
    - Track component contributions
    - Support reward shaping
    - Enable analysis and debugging
    """
    
    @abstractmethod
    def add_component(
        self,
        component: IRewardComponent,
        enabled: bool = True
    ) -> None:
        """Add reward component.
        
        Args:
            component: Component instance
            enabled: Whether to use it
            
        Design notes:
        - Support dynamic enable/disable
        - Validate component compatibility
        """
        ...
    
    @abstractmethod
    def calculate_reward(
        self,
        state_before: dict[str, Any],
        action: dict[str, Any], 
        state_after: dict[str, Any],
        done: bool
    ) -> tuple[float, dict[str, float]]:
        """Calculate total reward.
        
        Args:
            state_before: State before action
            action: Action taken
            state_after: State after action  
            done: Episode terminated
            
        Returns:
            Tuple of (total_reward, component_rewards)
            
        Design notes:
        - Return breakdown for analysis
        - Handle component failures
        - Apply clipping if configured
        """
        ...
    
    @abstractmethod
    def set_reward_shaping(
        self,
        shaping_function: Optional[Any] = None,
        clip_range: Optional[tuple[float, float]] = None
    ) -> None:
        """Configure reward shaping.
        
        Args:
            shaping_function: Optional shaping
            clip_range: Min/max clipping
            
        Design notes:
        - Post-process total reward
        - Improve learning stability
        """
        ...
    
    @abstractmethod
    def get_component_stats(
        self,
        lookback: int = 100
    ) -> dict[str, dict[str, float]]:
        """Get component statistics.
        
        Args:
            lookback: Episodes to analyze
            
        Returns:
            Stats per component:
            - mean
            - std
            - min/max
            - contribution_pct
            
        Design notes:
        - Used for tuning
        - Identify dominant components
        """
        ...
    
    @abstractmethod
    def tune_weights(
        self,
        target_ratios: dict[str, float]
    ) -> dict[str, float]:
        """Auto-tune component weights.
        
        Args:
            target_ratios: Desired contribution ratios
            
        Returns:
            New weights
            
        Design notes:
        - Based on recent statistics
        - Gradual adjustment
        """
        ...


class IRewardAnalyzer(Protocol):
    """Interface for reward analysis and debugging.
    
    Design principles:
    - Understand reward dynamics
    - Identify issues
    - Support experimentation
    """
    
    def analyze_episode(
        self,
        episode_rewards: list[dict[str, float]]
    ) -> dict[str, Any]:
        """Analyze episode rewards.
        
        Args:
            episode_rewards: List of reward breakdowns
            
        Returns:
            Analysis with:
            - component_correlations
            - temporal_patterns  
            - anomalies
            - recommendations
        """
        ...
    
    def compare_reward_systems(
        self,
        systems: dict[str, IRewardCalculator],
        test_episodes: list[Any]
    ) -> pd.DataFrame:
        """Compare different reward systems.
        
        Args:
            systems: Named reward calculators
            test_episodes: Episodes to test on
            
        Returns:
            Comparison DataFrame
        """
        ...
    
    def suggest_improvements(
        self,
        current_stats: dict[str, dict[str, float]],
        target_behavior: dict[str, Any]
    ) -> dict[str, Any]:
        """Suggest reward improvements.
        
        Args:
            current_stats: Current component stats
            target_behavior: Desired behavior
            
        Returns:
            Suggestions for:
            - weight_adjustments
            - new_components
            - parameter_changes
        """
        ...


class IRewardComponentFactory(Protocol):
    """Factory for creating reward components.
    
    Design principles:
    - Centralized component creation
    - Consistent configuration
    - Easy experimentation
    """
    
    def create_component(
        self,
        component_type: str,
        config: dict[str, Any]
    ) -> IRewardComponent:
        """Create reward component.
        
        Args:
            component_type: Type identifier
            config: Component configuration
            
        Returns:
            Component instance
        """
        ...
    
    def get_available_components(self) -> list[str]:
        """Get available component types.
        
        Returns:
            List of type identifiers
        """
        ...
    
    def get_default_config(
        self,
        component_type: str
    ) -> dict[str, Any]:
        """Get default configuration.
        
        Args:
            component_type: Type identifier
            
        Returns:
            Default config dict
        """
        ...
