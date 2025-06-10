"""
Typed context models for callback events.

Replaces dictionary-based context with strongly typed Pydantic models
for better validation, IDE support, and documentation.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
import torch


class EpisodeInfo(BaseModel):
    """Episode information."""
    num: int = Field(description="Episode number")
    reward: float = Field(description="Episode reward")
    length: int = Field(description="Episode length in steps")
    terminated: bool = Field(description="Whether episode terminated naturally")
    truncated: bool = Field(description="Whether episode was truncated")
    start_time: Optional[datetime] = Field(default=None, description="Episode start time")
    end_time: Optional[datetime] = Field(default=None, description="Episode end time")
    symbol: Optional[str] = Field(default=None, description="Trading symbol")
    reset_point_idx: Optional[int] = Field(default=None, description="Reset point index used")


class TradingMetrics(BaseModel):
    """Trading performance metrics."""
    portfolio_value: float = Field(description="Current portfolio value")
    total_profit: float = Field(description="Total profit/loss")
    trades_count: int = Field(description="Number of trades in episode")
    win_rate: float = Field(description="Win rate (0.0 to 1.0)")
    avg_trade_profit: float = Field(description="Average profit per trade")
    max_drawdown: float = Field(description="Maximum drawdown")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    total_fees: float = Field(default=0.0, description="Total trading fees")
    total_slippage: float = Field(default=0.0, description="Total slippage costs")


class PortfolioState(BaseModel):
    """Portfolio state information."""
    cash: float = Field(description="Available cash")
    position_value: float = Field(description="Value of current positions")
    total_value: float = Field(description="Total portfolio value")
    position_count: int = Field(description="Number of open positions")
    buying_power: float = Field(description="Available buying power")
    unrealized_pnl: float = Field(description="Unrealized profit/loss")
    realized_pnl: float = Field(description="Realized profit/loss")


class Trade(BaseModel):
    """Individual trade information."""
    timestamp: datetime = Field(description="Trade timestamp")
    symbol: str = Field(description="Trading symbol")
    side: str = Field(description="Trade side (buy/sell)")
    quantity: int = Field(description="Trade quantity")
    price: float = Field(description="Trade price")
    fees: float = Field(description="Trading fees")
    slippage: float = Field(description="Slippage cost")
    pnl: Optional[float] = Field(default=None, description="Trade P&L (for closes)")


class ModelInfo(BaseModel):
    """Model state information."""
    version: Optional[int] = Field(default=None, description="Model version")
    training_step: int = Field(description="Current training step")
    learning_rate: float = Field(description="Current learning rate")
    clip_epsilon: float = Field(description="Current PPO clip epsilon")
    total_parameters: Optional[int] = Field(default=None, description="Total model parameters")
    device: str = Field(description="Model device")


class UpdateInfo(BaseModel):
    """Training update information."""
    num: int = Field(description="Update number")
    learning_rate: float = Field(description="Learning rate used")
    clip_epsilon: float = Field(description="PPO clip epsilon used")
    batch_size: int = Field(description="Batch size used")
    n_epochs: int = Field(description="Number of epochs")


class TrainingLosses(BaseModel):
    """Training loss information."""
    policy_loss: float = Field(description="Policy loss")
    value_loss: float = Field(description="Value function loss")
    entropy_loss: float = Field(description="Entropy loss")
    total_loss: float = Field(description="Total loss")


class TrainingMetrics(BaseModel):
    """Training metrics."""
    kl_divergence: float = Field(description="KL divergence")
    clip_fraction: float = Field(description="Fraction of clipped actions")
    gradient_norm: float = Field(description="Gradient norm")
    explained_variance: float = Field(description="Explained variance")
    advantage_mean: float = Field(description="Mean advantage")
    advantage_std: float = Field(description="Advantage standard deviation")


class PerformanceMetrics(BaseModel):
    """Performance timing metrics."""
    episode_time: Optional[float] = Field(default=None, description="Episode duration in seconds")
    update_time: Optional[float] = Field(default=None, description="Update duration in seconds")
    data_loading_time: Optional[float] = Field(default=None, description="Data loading time")
    inference_time: Optional[float] = Field(default=None, description="Model inference time")
    fps: Optional[float] = Field(default=None, description="Frames per second")


class TrainingStartContext(BaseModel):
    """Context for training start event."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Training start time")

    class Config:
        arbitrary_types_allowed = True


class EpisodeEndContext(BaseModel):
    """Context for episode end event."""
    episode: EpisodeInfo = Field(description="Episode information")
    metrics: TradingMetrics = Field(description="Trading metrics")
    portfolio: PortfolioState = Field(description="Portfolio state")
    trades: List[Trade] = Field(description="Trades executed in episode")
    model: ModelInfo = Field(description="Model information")
    environment: Any = Field(description="Environment instance")
    trainer: Any = Field(description="Trainer instance")
    performance: Optional[PerformanceMetrics] = Field(default=None, description="Performance metrics")
    
    class Config:
        arbitrary_types_allowed = True


class UpdateEndContext(BaseModel):
    """Context for update end event."""
    update: UpdateInfo = Field(description="Update information")
    losses: TrainingLosses = Field(description="Training losses")
    metrics: TrainingMetrics = Field(description="Training metrics")
    model: ModelInfo = Field(description="Model information")
    trainer: Any = Field(description="Trainer instance")
    performance: Optional[PerformanceMetrics] = Field(default=None, description="Performance metrics")
    
    class Config:
        arbitrary_types_allowed = True


class TrainingEndContext(BaseModel):
    """Context for training end event."""
    global_episodes: int = Field(description="Total episodes completed")
    global_updates: int = Field(description="Total updates completed")
    global_steps: int = Field(description="Total steps completed")
    global_cycles: int = Field(description="Total cycles completed")
    reason: str = Field(description="Training termination reason")


class CustomEventContext(BaseModel):
    """Context for custom events."""
    event_name: str = Field(description="Name of custom event")
    data: Dict[str, Any] = Field(description="Event-specific data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    source: Optional[str] = Field(default=None, description="Event source component")


# Union type for all context types
CallbackContext = Union[
    TrainingStartContext,
    EpisodeEndContext,
    UpdateEndContext,
    TrainingEndContext,
    CustomEventContext
]