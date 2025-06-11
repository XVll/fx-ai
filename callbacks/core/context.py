"""
Typed context models for callback events.

Replaces dictionary-based context with strongly typed dataclasses
for better validation, IDE support, and documentation.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import torch


@dataclass
class EpisodeInfo:
    """Episode information."""
    num: int                                            # Episode number
    reward: float                                       # Episode reward
    length: int                                         # Episode length in steps
    terminated: bool                                    # Whether episode terminated naturally
    truncated: bool                                     # Whether episode was truncated
    start_time: Optional[datetime] = None               # Episode start time
    end_time: Optional[datetime] = None                 # Episode end time
    symbol: Optional[str] = None                        # Trading symbol
    reset_point_idx: Optional[int] = None               # Reset point index used


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    portfolio_value: float                              # Current portfolio value
    total_profit: float                                 # Total profit/loss
    trades_count: int                                   # Number of trades in episode
    win_rate: float                                     # Win rate (0.0 to 1.0)
    avg_trade_profit: float                             # Average profit per trade
    max_drawdown: float                                 # Maximum drawdown
    sharpe_ratio: Optional[float] = None                # Sharpe ratio
    total_fees: float = 0.0                             # Total trading fees
    total_slippage: float = 0.0                         # Total slippage costs


@dataclass
class PortfolioState:
    """Portfolio state information."""
    cash: float                                         # Available cash
    position_value: float                               # Value of current positions
    total_value: float                                  # Total portfolio value
    position_count: int                                 # Number of open positions
    buying_power: float                                 # Available buying power
    unrealized_pnl: float                               # Unrealized profit/loss
    realized_pnl: float                                 # Realized profit/loss


@dataclass
class Trade:
    """Individual trade information."""
    timestamp: datetime                                 # Trade timestamp
    symbol: str                                         # Trading symbol
    side: str                                           # Trade side (buy/sell)
    quantity: int                                       # Trade quantity
    price: float                                        # Trade price
    fees: float                                         # Trading fees
    slippage: float                                     # Slippage cost
    pnl: Optional[float] = None                         # Trade P&L (for closes)


@dataclass
class ModelInfo:
    """Model state information."""
    version: Optional[int] = None                       # Model version
    training_step: int = 0                              # Current training step
    learning_rate: float = 0.0                          # Current learning rate
    clip_epsilon: float = 0.0                           # Current PPO clip epsilon
    total_parameters: Optional[int] = None              # Total model parameters
    device: str = "cpu"                                 # Model device


@dataclass
class UpdateInfo:
    """Training update information."""
    num: int                                            # Update number
    learning_rate: float                                # Learning rate used
    clip_epsilon: float                                 # PPO clip epsilon used
    batch_size: int                                     # Batch size used
    n_epochs: int                                       # Number of epochs


@dataclass
class TrainingLosses:
    """Training loss information."""
    policy_loss: float                                  # Policy loss
    value_loss: float                                   # Value function loss
    entropy_loss: float                                 # Entropy loss
    total_loss: float                                   # Total loss


@dataclass
class TrainingMetrics:
    """Training metrics."""
    kl_divergence: float                                # KL divergence
    clip_fraction: float                                # Fraction of clipped actions
    gradient_norm: float                                # Gradient norm
    explained_variance: float                           # Explained variance
    advantage_mean: float                               # Mean advantage
    advantage_std: float                                # Advantage standard deviation


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""
    episode_time: Optional[float] = None                # Episode duration in seconds
    update_time: Optional[float] = None                 # Update duration in seconds
    data_loading_time: Optional[float] = None           # Data loading time
    inference_time: Optional[float] = None              # Model inference time
    fps: Optional[float] = None                         # Frames per second


@dataclass
class TrainingStartContext:
    """Context for training start event."""
    timestamp: datetime = field(default_factory=datetime.now)  # Training start time


@dataclass
class EpisodeEndContext:
    """Context for episode end event."""
    episode: EpisodeInfo                                # Episode information
    metrics: TradingMetrics                             # Trading metrics
    portfolio: PortfolioState                           # Portfolio state
    trades: List[Trade]                                 # Trades executed in episode
    model: ModelInfo                                    # Model information
    environment: Any                                    # Environment instance
    trainer: Any                                        # Trainer instance
    performance: Optional[PerformanceMetrics] = None    # Performance metrics


@dataclass
class UpdateEndContext:
    """Context for update end event."""
    update_num: int                                     # Update number
    policy_loss: float                                  # Policy loss
    value_loss: float                                   # Value loss
    entropy_loss: float                                 # Entropy loss
    total_loss: float                                   # Total loss
    kl_divergence: float = 0.0                          # KL divergence
    clip_fraction: float = 0.0                          # Clip fraction
    gradient_norm: float = 0.0                          # Gradient norm
    explained_variance: float = 0.0                     # Explained variance
    advantage_mean: float = 0.0                         # Mean advantage
    trainer: Any = None                                 # Trainer instance
    performance: Optional[PerformanceMetrics] = None    # Performance metrics


@dataclass
class TrainingEndContext:
    """Context for training end event."""
    global_episodes: int                                # Total episodes completed
    global_updates: int                                 # Total updates completed
    global_steps: int                                   # Total steps completed
    global_cycles: int                                  # Total cycles completed
    reason: str                                         # Training termination reason


@dataclass
class CustomEventContext:
    """Context for custom events."""
    event_name: str                                     # Name of custom event
    data: Dict[str, Any]                                # Event-specific data
    timestamp: datetime = field(default_factory=datetime.now)  # Event timestamp
    source: Optional[str] = None                        # Event source component


# Union type for all context types
CallbackContext = Union[
    TrainingStartContext,
    EpisodeEndContext,
    UpdateEndContext,
    TrainingEndContext,
    CustomEventContext
]