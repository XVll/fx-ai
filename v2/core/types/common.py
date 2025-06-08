"""
Common type definitions used across the trading system.

These types provide the foundation for type safety and consistency
across all components of the system.
"""

from typing import TypedDict, NewType, Union, Any, Protocol, TypeVar, Generic
from datetime import datetime
from enum import Enum, IntEnum
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


class RunMode(Enum):
    """System execution modes.
    
    Design: Modes separate concerns and enable specialized workflows.
    Each mode has its own lifecycle and optimization goals.
    """
    TRAINING = "TRAINING"          # Standard RL training
    CONTINUOUS = "CONTINUOUS"      # Continuous improvement mode
    BENCHMARK = "BENCHMARK"        # Performance evaluation
    OPTUNA = "OPTUNA"             # Hyperparameter optimization
    BACKTEST = "BACKTEST"         # Historical evaluation
    LIVE = "LIVE"                 # Live trading (future)
    REPLAY = "REPLAY"             # Replay specific episodes


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


# Protocol definitions for duck typing
class Configurable(Protocol):
    """Protocol for configurable components.
    
    Design: Enables dynamic configuration without tight coupling.
    """
    def get_config(self) -> dict[str, Any]:
        """Return current configuration."""
        ...
    
    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration dynamically."""
        ...


class Resettable(Protocol):
    """Protocol for components that maintain state.
    
    Design: Ensures clean state management across episodes.
    """
    def reset(self) -> None:
        """Reset internal state to initial conditions."""
        ...


class Serializable(Protocol):
    """Protocol for components that can be serialized.
    
    Design: Enables checkpointing and distributed execution.
    """
    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        ...
    
    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from dictionary."""
        ...
