"""
Common type definitions used across the trading system.

These types provide the foundation for type safety and consistency
across all components of the system.
"""

from typing import TypedDict, NewType, Union, Any, Protocol, TypeVar, Generic, List, Optional
from datetime import datetime
from enum import Enum, IntEnum
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Basic type aliases
Symbol = NewType('Symbol', str)
Timestamp = Union[datetime, pd.Timestamp]
Price = float
Volume = float
Quantity = float
Cash = float
PnL = float
Reward = float
EpisodeId = NewType('EpisodeId', str)
SessionId = NewType('SessionId', str)
ModelVersion = NewType('ModelVersion', str)

# Array types
FeatureArray = NDArray[np.float32]
ActionArray = NDArray[np.int32]
ObservationArray = NDArray[np.float32]
MaskArray = NDArray[np.bool_]
ProbabilityArray = NDArray[np.float32]

# Generic type vars
T = TypeVar('T')
ConfigT = TypeVar('ConfigT')
StateT = TypeVar('StateT')


class ActionType(IntEnum):
    """Trading action types.
    
    Design: Use IntEnum for gymnasium compatibility while maintaining readability.
    """
    HOLD = 0  # Do nothing
    BUY = 1   # Enter/increase long position
    SELL = 2  # Enter/increase short position


class PositionSizeType(IntEnum):
    """Position size as percentage of available capital.
    
    Design: Discrete sizes simplify execution and risk management.
    """
    SIZE_25 = 0   # 25% of available capital
    SIZE_50 = 1   # 50% of available capital
    SIZE_75 = 2   # 75% of available capital
    SIZE_100 = 3  # 100% of available capital


class ActionIndex(IntEnum):
    """Single-index action space for clean 7 actions.
    
    Eliminates redundant HOLD variants and provides direct action mapping.
    """
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2  
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6


class OrderType(Enum):
    """Order types for execution."""
    MARKET = "MARKET"      # Immediate execution at current price
    LIMIT = "LIMIT"        # Execute at specified price or better
    STOP = "STOP"          # Trigger when price crosses threshold
    STOP_LIMIT = "STOP_LIMIT"  # Stop that becomes limit order


class OrderSide(Enum):
    """Order side for execution."""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    """Current position side."""
    LONG = "LONG"    # Owns the asset
    SHORT = "SHORT"  # Borrowed and sold
    FLAT = "FLAT"    # No position


class TerminationReason(Enum):
    """Reasons for episode termination."""
    BANKRUPTCY = "BANKRUPTCY"              # Lost all capital
    MAX_LOSS_EXCEEDED = "MAX_LOSS_EXCEEDED"  # Hit stop loss
    END_OF_DATA = "END_OF_DATA"           # No more market data
    MAX_STEPS = "MAX_STEPS"               # Episode length limit
    INVALID_ACTION = "INVALID_ACTION"      # Too many invalid actions
    MANUAL = "MANUAL"                      # User intervention
    ERROR = "ERROR"                        # System error
    EXTERNAL_ERROR = "EXTERNAL_ERROR"      # External system error
    
    # Training termination reasons
    MAX_EPISODES_REACHED = "MAX_EPISODES_REACHED"
    MAX_UPDATES_REACHED = "MAX_UPDATES_REACHED"
    MAX_CYCLES_REACHED = "MAX_CYCLES_REACHED"
    PERFORMANCE_PLATEAU = "PERFORMANCE_PLATEAU"


class TerminationReasonEnum(Enum):
    """Episode termination reasons for environment."""
    BANKRUPTCY = "bankruptcy"
    MAX_LOSS_REACHED = "max_loss_reached"
    END_OF_SESSION_DATA = "end_of_session_data"
    MAX_STEPS_REACHED = "max_steps_reached"
    MAX_DURATION = "max_duration"
    OBSERVATION_FAILURE = "observation_failure"
    ERROR = "error"




class ActionTypeEnum(Enum):
    """Defines the type of action the agent can take."""

    HOLD = 0
    BUY = 1
    SELL = 2


class PositionSizeTypeEnum(Enum):
    """Defines the relative size of the position for an action."""

    SIZE_25 = 0  # 25%
    SIZE_50 = 1  # 50%
    SIZE_75 = 2  # 75%
    SIZE_100 = 3  # 100%

    @property
    def value_float(self) -> float:
        """Returns the float multiplier for the size (0.25, 0.50, 0.75, 1.0)."""
        return (self.value + 1) * 0.25


class RunMode(Enum):
    """System execution modes.
    
    Design: Modes separate concerns and enable specialized workflows.
    Each mode has its own lifecycle and optimization goals.
    """
    CONTINUOUS_TRAINING = "CONTINUOUS_TRAINING"  # Primary training mode with adaptive configs
    BENCHMARK_EVALUATION = "BENCHMARK_EVALUATION"  # Performance evaluation
    OPTUNA_OPTIMIZATION = "OPTUNA_OPTIMIZATION"   # Hyperparameter optimization
    LIVE = "LIVE"                             # Live trading (future)



class FeatureFrequency(Enum):
    """Feature update frequencies.
    
    Design: Multi-scale features capture different market dynamics.
    """
    HIGH = "HIGH"      # 1-second updates
    MEDIUM = "MEDIUM"  # 1-minute aggregates  
    LOW = "LOW"        # 5-minute or daily
    STATIC = "STATIC"  # Unchanging features


# Structured data types
class MarketDataPoint(TypedDict):
    """Single market data observation.
    
    Design: TypedDict provides structure while remaining JSON-serializable.
    """
    timestamp: datetime
    symbol: str
    price: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    volume: float
    trade_count: int
    vwap: float


class ExecutionInfo(TypedDict):
    """Information about trade execution.
    
    Design: Captures all details needed for performance analysis.
    """
    timestamp: datetime
    symbol: str
    side: str  # OrderSide value
    quantity: float
    executed_price: float
    requested_price: float
    commission: float
    slippage: float
    market_impact: float
    latency_ms: float


class EpisodeMetrics(TypedDict):
    """Metrics collected per episode.
    
    Design: Comprehensive metrics enable debugging and optimization.
    """
    episode_id: str
    total_reward: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    invalid_actions: int
    episode_length: int
    termination_reason: str


class ModelCheckpoint(TypedDict):
    """Model checkpoint metadata.
    
    Design: Rich metadata enables model selection and rollback.
    """
    version: str
    timestamp: datetime
    episode: int
    total_steps: int
    avg_reward: float
    best_reward: float
    config_hash: str
    metrics: dict[str, float]


class Resettable(Protocol):
    """Protocol for components that maintain state.
    
    Design: Ensures clean state management across episodes.
    """
    def reset(self) -> None:
        """Reset internal state to initial conditions."""
        ...


# Training system data structures
@dataclass
class RolloutResult:
    """Result from trainer rollout collection.
    
    Design: Encapsulates all information from rollout phase for
    training manager decision making.
    """
    steps_collected: int
    episodes_completed: int  
    buffer_ready: bool
    interrupted: bool = False

@dataclass 
class UpdateResult:
    """Result from trainer policy update.
    
    Design: Contains all metrics and status information from
    a single policy update for callbacks and analysis.
    """
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    gradient_norm: float = 0.0
    explained_variance: float = 0.0
    advantage_mean: float = 0.0
    interrupted: bool = False


# Action mapping utilities for single-index action space
ACTION_TO_TYPE_SIZE = {
    ActionIndex.HOLD: (ActionType.HOLD, None),
    ActionIndex.BUY_25: (ActionType.BUY, 0.25),
    ActionIndex.BUY_50: (ActionType.BUY, 0.50),
    ActionIndex.BUY_100: (ActionType.BUY, 1.00),
    ActionIndex.SELL_25: (ActionType.SELL, 0.25),
    ActionIndex.SELL_50: (ActionType.SELL, 0.50),
    ActionIndex.SELL_100: (ActionType.SELL, 1.00),
}

TYPE_SIZE_TO_ACTION = {v: k for k, v in ACTION_TO_TYPE_SIZE.items()}


def single_index_to_type_size(action_index: int) -> tuple[ActionType, Optional[float]]:
    """Convert single action index to (ActionType, size) tuple."""
    if action_index not in ACTION_TO_TYPE_SIZE:
        raise ValueError(f"Invalid action index: {action_index}. Must be 0-6.")
    return ACTION_TO_TYPE_SIZE[action_index]


def type_size_to_single_index(action_type: ActionType, size: Optional[float] = None) -> int:
    """Convert (ActionType, size) tuple to single action index."""
    key = (action_type, size)
    if key not in TYPE_SIZE_TO_ACTION:
        raise ValueError(f"Invalid action combination: {action_type}, {size}")
    return TYPE_SIZE_TO_ACTION[key]
