from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, runtime_checkable
from enum import Enum
import numpy as np
import numpy.typing as npt


# Type definitions for strong typing
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


class PositionSide(Enum):
    """Position sides."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"

class OrderSide(Enum):
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"


class TerminationReason(Enum):
    """Episode termination reasons."""
    END_OF_SESSION_DATA = "END_OF_SESSION_DATA"
    MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
    BANKRUPTCY = "BANKRUPTCY"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    MARKET_CLOSE = "MARKET_CLOSE"
    MAX_DURATION = "MAX_DURATION"


# Type aliases for complex types
ObservationDict = Dict[str, npt.NDArray[np.float32]]
InfoDict = Dict[str, Any]
ActionArray = npt.NDArray[np.int32]
ActionMask = npt.NDArray[np.bool_]
ActionProbabilities = npt.NDArray[np.float32]

# Market data types
MarketData = Dict[str, Union[float, int, datetime, str]]
PriceDict = Dict[str, float]

# Portfolio state types
PositionData = Dict[str, Union[float, int, PositionSide, datetime]]
PortfolioPositions = Dict[str, PositionData]
PortfolioState = Dict[str, Union[float, int, datetime, PortfolioPositions, Dict[str, Any]]]

# Trading types
FillDetails = Dict[str, Union[str, float, int, datetime, OrderSide]]
ExecutionResult = Dict[str, Any]

# Reset point and momentum day types
ResetPoint = Dict[str, Union[datetime, float, int, str]]
MomentumDay = Dict[str, Union[str, datetime, float, int]]

# Session and episode types
SessionOptions = Dict[str, Any]
EpisodeInfo = Dict[str, Union[int, float, str, datetime, ResetPoint]]


@runtime_checkable
class ITradingEnvironment(Protocol):
    """
    Interface for trading environments.
    
    Comprehensive interface that includes all trading environment capabilities:
    - Basic gym environment operations (reset, step, close, render)
    - Session management (setup, prepare, switch)
    - Episode management (reset points, momentum days)
    - Action masking and validation
    - Training information tracking
    """

    # Basic environment operations
    def setup_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """
        Setup a new trading session for a specific symbol and date.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            date: Trading date
            
        Raises:
            ValueError: If symbol or date is invalid
        """
        ...

    def reset(self, seed: Optional[int] = None, options: Optional[SessionOptions] = None) -> Tuple[ObservationDict, InfoDict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        ...

    def step(self, action: ActionArray) -> Tuple[ObservationDict, float, bool, bool, InfoDict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        ...

    def close(self) -> None:
        """
        Close environment and cleanup resources.
        """
        ...

    def render(self, info_dict: Optional[InfoDict] = None) -> None:
        """
        Render environment state.
        
        Args:
            info_dict: Optional info dictionary for rendering
        """
        ...

    # Action masking
    def get_action_mask(self) -> Optional[ActionMask]:
        """
        Get current action mask for valid actions.
        
        Returns:
            Boolean array where True = valid action, None if masking disabled
        """
        ...

    def mask_action_probabilities(self, action_probs: ActionProbabilities) -> ActionProbabilities:
        """
        Apply action masking to action probabilities.
        
        Args:
            action_probs: Raw action probabilities from policy
            
        Returns:
            Masked and renormalized action probabilities
        """
        ...

    # Episode management
    def reset_at_point(self, reset_point_idx: Optional[int] = None) -> Tuple[ObservationDict, InfoDict]:
        """
        Reset to a specific reset point within the loaded session.
        
        Args:
            reset_point_idx: Index of reset point to use
            
        Returns:
            Tuple of (observation, info)
        """
        ...

    def get_next_reset_point(self) -> Optional[ResetPoint]:
        """
        Get the next available reset point.
        
        Returns:
            Reset point dictionary or None if no more points
        """
        ...

    def has_more_reset_points(self) -> bool:
        """
        Check if there are more reset points available.
        
        Returns:
            True if more reset points exist
        """
        ...

    def select_next_momentum_day(self, exclude_dates: Optional[List[datetime]] = None) -> Optional[MomentumDay]:
        """
        Select next momentum day for training.
        
        Args:
            exclude_dates: Dates to exclude from selection
            
        Returns:
            Momentum day info or None if no suitable days
        """
        ...

    # Session management
    def prepare_next_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """
        Prepare next session in background for fast switching.
        
        Args:
            symbol: Trading symbol
            date: Trading date
        """
        ...

    def switch_to_next_session(self) -> None:
        """
        Switch to the prepared next session.
        
        Raises:
            ValueError: If no session was prepared
        """
        ...

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                         total_steps: int = 0, update_count: int = 0) -> None:
        """
        Set training information for metrics tracking.
        
        Args:
            episode_num: Current episode number
            total_episodes: Total episodes planned
            total_steps: Total steps taken
            update_count: Number of model updates
        """
        ...