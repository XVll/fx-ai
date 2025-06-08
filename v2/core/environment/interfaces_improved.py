"""
Improved Environment interfaces with clear responsibility separation.

Key improvements:
- Remove curriculum/training logic from environment
- Focus on session and episode management only
- Clear separation of concerns
- Better state management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, runtime_checkable
from enum import Enum
import numpy as np
import numpy.typing as npt


# Keep existing type definitions
class ActionType(Enum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionSizeType(Enum):
    """Position size types."""
    SIZE_25 = 0  # 25%
    SIZE_50 = 1  # 50%
    SIZE_75 = 2  # 75%
    SIZE_100 = 3  # 100%

    @property
    def value_float(self) -> float:
        """Returns the float multiplier for the size (0.25, 0.50, 0.75, 1.0)."""
        return (self.value + 1) * 0.25


class TerminationReason(Enum):
    """Episode termination reasons."""
    END_OF_SESSION_DATA = "END_OF_SESSION_DATA"
    MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
    BANKRUPTCY = "BANKRUPTCY"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    MARKET_CLOSE = "MARKET_CLOSE"
    MAX_DURATION = "MAX_DURATION"


# Type aliases
ObservationDict = Dict[str, npt.NDArray[np.float32]]
InfoDict = Dict[str, Any]
ActionArray = npt.NDArray[np.int32]
ActionMask = npt.NDArray[np.bool_]
ActionProbabilities = npt.NDArray[np.float32]

# Environment-specific types (NOT curriculum types)
ResetPoint = Dict[str, Union[datetime, float, int, str]]
SessionOptions = Dict[str, Any]
EpisodeInfo = Dict[str, Union[int, float, str, datetime, ResetPoint]]


@runtime_checkable
class ITradingEnvironment(Protocol):
    """
    CLEAN Trading Environment Interface - PURE ENVIRONMENT RESPONSIBILITY
    
    Responsibilities:
    - Session management (setup, data loading)
    - Episode execution (reset, step, termination)
    - Action masking and validation
    - Market simulation and state tracking
    
    NOT responsible for:
    - Training orchestration
    - Curriculum management  
    - Day selection logic
    - Training metrics collection
    """

    # === SESSION MANAGEMENT ===
    def setup_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """Setup trading session for specific symbol and date.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            date: Trading date
            
        Raises:
            ValueError: If symbol or date is invalid
            
        Responsibilities:
        - Load market data for the date
        - Initialize market simulator
        - Load available reset points for the session
        - Prepare environment state
        """
        ...

    @property
    def current_symbol(self) -> Optional[str]:
        """Currently loaded symbol."""
        ...
    
    @property 
    def current_date(self) -> Optional[datetime]:
        """Currently loaded date."""
        ...
    
    @property
    def session_active(self) -> bool:
        """Whether a session is currently active."""
        ...

    # === EPISODE MANAGEMENT ===
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[SessionOptions] = None
    ) -> Tuple[ObservationDict, InfoDict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options:
                - reset_point_idx: Specific reset point to use
                - randomize_start: Add random offset to start time
                
        Returns:
            Tuple of (observation, info)
            
        Responsibilities:
        - Reset all simulators (market, portfolio, execution)
        - Select reset point (from options or default)
        - Initialize episode state
        - Return initial observation
        """
        ...

    def reset_at_point(self, reset_point_idx: int) -> Tuple[ObservationDict, InfoDict]:
        """Reset to specific reset point within session.
        
        Args:
            reset_point_idx: Index of reset point to use
            
        Returns:
            Tuple of (observation, info)
            
        Responsibilities:
        - Validate reset point index
        - Reset simulators to specific time
        - Handle position carryover if needed
        """
        ...

    def step(self, action: ActionArray) -> Tuple[ObservationDict, float, bool, bool, InfoDict]:
        """Execute one environment step.
        
        Args:
            action: Action to execute [action_type, position_size]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            
        Responsibilities:
        - Validate action via action mask
        - Execute trade through execution simulator
        - Update portfolio state
        - Advance market simulation
        - Calculate reward
        - Check termination conditions
        - Return next observation
        """
        ...

    def close(self) -> None:
        """Close environment and cleanup resources."""
        ...

    # === ACTION MANAGEMENT ===
    def get_action_mask(self) -> Optional[ActionMask]:
        """Get current action mask for valid actions.
        
        Returns:
            Boolean array where True = valid action, None if disabled
            
        Responsibilities:
        - Check portfolio constraints (buying power, position limits)
        - Check market conditions (halts, liquidity)
        - Return valid action combinations
        """
        ...

    def mask_action_probabilities(self, action_probs: ActionProbabilities) -> ActionProbabilities:
        """Apply action masking to probabilities.
        
        Args:
            action_probs: Raw action probabilities from policy
            
        Returns:
            Masked and renormalized probabilities
        """
        ...

    # === RESET POINT MANAGEMENT (Environment State Only) ===
    def get_available_reset_points(self) -> List[ResetPoint]:
        """Get all available reset points for current session.
        
        Returns:
            List of reset point dictionaries
            
        Note: This is environment state, NOT curriculum logic
        """
        ...

    def get_next_reset_point(self) -> Optional[ResetPoint]:
        """Get next available reset point in sequence."""
        ...

    def has_more_reset_points(self) -> bool:
        """Check if more reset points are available."""
        ...

    # === OBSERVATION AND STATE ===
    def get_current_observation(self) -> ObservationDict:
        """Get current environment observation."""
        ...
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        ...
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get current market state."""
        ...

    # === RENDERING AND VISUALIZATION ===
    def render(self, mode: str = "human") -> Optional[Any]:
        """Render environment state.
        
        Args:
            mode: Rendering mode ("human", "rgb_array", etc.)
        """
        ...


@runtime_checkable
class ISessionManager(Protocol):
    """Manages session preparation and switching for efficient training.
    
    Responsibility: Handle session lifecycle and background preparation
    """
    
    def prepare_session(
        self, 
        symbol: str, 
        date: Union[str, datetime],
        priority: int = 0
    ) -> str:
        """Prepare session in background.
        
        Args:
            symbol: Trading symbol
            date: Trading date  
            priority: Preparation priority
            
        Returns:
            Session ID for tracking
        """
        ...
    
    def is_session_ready(self, session_id: str) -> bool:
        """Check if prepared session is ready."""
        ...
    
    def switch_to_session(self, session_id: str) -> None:
        """Switch to prepared session."""
        ...
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up prepared session resources."""
        ...


@runtime_checkable  
class IActionMaskableEnvironment(Protocol):
    """Environment with advanced action masking capabilities."""
    
    def get_action_mask_detailed(self) -> Dict[str, ActionMask]:
        """Get detailed action masks by category.
        
        Returns:
            Dict with keys like "portfolio_constraints", "market_conditions"
        """
        ...
    
    def explain_invalid_actions(self, action: ActionArray) -> List[str]:
        """Explain why specific actions are invalid.
        
        Args:
            action: Action to validate
            
        Returns:
            List of constraint violation reasons
        """
        ...


@runtime_checkable
class IMultiAssetEnvironment(Protocol):
    """Environment supporting multiple assets simultaneously."""
    
    def setup_multi_asset_session(
        self,
        assets: List[Tuple[str, datetime]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> None:
        """Setup session with multiple assets."""
        ...
    
    def get_cross_asset_features(self) -> Dict[str, ObservationDict]:
        """Get features for all assets."""
        ...


@runtime_checkable
class IEnvironmentFactory(Protocol):
    """Factory for creating different environment types."""
    
    def create_environment(
        self,
        env_type: str,
        config: Dict[str, Any]
    ) -> ITradingEnvironment:
        """Create environment instance."""
        ...
    
    def create_session_manager(
        self,
        config: Dict[str, Any]
    ) -> ISessionManager:
        """Create session manager."""
        ...


# Wrapper interfaces for environment extensions
@runtime_checkable
class IEnvironmentWrapper(Protocol):
    """Base wrapper interface."""
    
    @property
    def unwrapped(self) -> ITradingEnvironment:
        """Get underlying environment."""
        ...


@runtime_checkable
class IRewardWrapper(IEnvironmentWrapper):
    """Wrapper for reward function modifications."""
    
    def modify_reward(
        self,
        reward: float,
        observation: ObservationDict,
        action: ActionArray,
        info: InfoDict
    ) -> float:
        """Modify reward signal."""
        ...


@runtime_checkable
class IObservationWrapper(IEnvironmentWrapper):
    """Wrapper for observation modifications."""
    
    def modify_observation(
        self,
        observation: ObservationDict
    ) -> ObservationDict:
        """Modify observation format."""
        ...