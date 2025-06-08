"""
Pydantic configuration schemas for type safety and validation.

These schemas define the structure and validation rules for all
configuration objects in the system.
"""

from typing import Optional, Any, Union, Literal
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

from ...core import RunMode, ActionType, PositionSizeType


# Base configuration classes
class BaseConfig(BaseModel):
    """Base configuration with common settings."""
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent typos by forbidding extra fields
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Use enum values instead of objects
    
    name: str = Field(..., description="Configuration name")
    version: str = Field("1.0", description="Configuration version")
    description: Optional[str] = Field(None, description="Configuration description")


# Agent configurations
class NetworkConfig(BaseModel):
    """Neural network architecture configuration."""
    
    shared_layers: list[int] = Field(
        [512, 512],
        description="Shared layer sizes for feature extraction"
    )
    policy_layers: list[int] = Field(
        [256],
        description="Policy head layer sizes"
    )
    value_layers: list[int] = Field(
        [256],
        description="Value head layer sizes"
    )
    activation: Literal["relu", "tanh", "gelu", "selu"] = Field(
        "relu",
        description="Activation function"
    )
    dropout: float = Field(
        0.0,
        ge=0.0,
        le=0.5,
        description="Dropout rate for regularization"
    )
    layer_norm: bool = Field(
        True,
        description="Use layer normalization"
    )
    initialization: Literal["xavier", "kaiming", "orthogonal"] = Field(
        "orthogonal",
        description="Weight initialization method"
    )


class PPOConfig(BaseConfig):
    """PPO agent configuration."""
    
    # Learning parameters
    learning_rate: float = Field(
        3e-4,
        gt=0,
        le=1.0,
        description="Learning rate for optimizer"
    )
    lr_schedule: Literal["constant", "linear", "cosine", "exponential"] = Field(
        "constant",
        description="Learning rate schedule"
    )
    
    # PPO-specific parameters
    gamma: float = Field(
        0.99,
        ge=0,
        le=1,
        description="Discount factor for future rewards"
    )
    gae_lambda: float = Field(
        0.95,
        ge=0,
        le=1,
        description="GAE lambda for advantage estimation"
    )
    clip_epsilon: float = Field(
        0.2,
        gt=0,
        le=0.5,
        description="PPO clipping parameter"
    )
    value_clip: Optional[float] = Field(
        0.2,
        ge=0,
        description="Value function clipping (None to disable)"
    )
    
    # Loss coefficients
    value_coef: float = Field(
        0.5,
        ge=0,
        description="Value loss coefficient"
    )
    entropy_coef: float = Field(
        0.01,
        ge=0,
        description="Entropy regularization coefficient"
    )
    
    # Training parameters
    n_epochs: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of epochs per training batch"
    )
    batch_size: int = Field(
        64,
        ge=1,
        description="Minibatch size for training"
    )
    max_grad_norm: Optional[float] = Field(
        0.5,
        ge=0,
        description="Maximum gradient norm for clipping"
    )
    
    # Network architecture
    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Neural network configuration"
    )
    
    # Advanced options
    normalize_advantage: bool = Field(
        True,
        description="Normalize advantages"
    )
    target_kl: Optional[float] = Field(
        None,
        ge=0,
        description="Target KL divergence for early stopping"
    )
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v, values):
        """Validate learning rate based on schedule."""
        if values.get('lr_schedule') != 'constant' and v > 1e-3:
            raise ValueError("High learning rate with non-constant schedule may be unstable")
        return v


# Environment configurations
class ObservationConfig(BaseModel):
    """Observation space configuration."""
    
    hf_seq_len: int = Field(60, ge=1, description="High-frequency sequence length")
    hf_features: list[str] = Field(
        ["price_momentum", "volume_imbalance", "spread", "trade_intensity"],
        description="High-frequency feature names"
    )
    
    mf_seq_len: int = Field(20, ge=1, description="Medium-frequency sequence length")
    mf_features: list[str] = Field(
        ["rsi", "bollinger_bands", "vwap_deviation", "volume_profile"],
        description="Medium-frequency feature names"
    )
    
    lf_seq_len: int = Field(10, ge=1, description="Low-frequency sequence length")
    lf_features: list[str] = Field(
        ["market_regime", "support_resistance", "trend_strength"],
        description="Low-frequency feature names"
    )
    
    portfolio_features: list[str] = Field(
        ["position_size", "unrealized_pnl", "cash_ratio", "drawdown"],
        description="Portfolio state features"
    )
    
    normalization: Literal["standard", "minmax", "robust", "none"] = Field(
        "standard",
        description="Feature normalization method"
    )


class ActionConfig(BaseModel):
    """Action space configuration."""
    
    action_types: list[ActionType] = Field(
        [ActionType.HOLD, ActionType.BUY, ActionType.SELL],
        description="Available action types"
    )
    position_sizes: list[PositionSizeType] = Field(
        [PositionSizeType.SIZE_25, PositionSizeType.SIZE_50, 
         PositionSizeType.SIZE_75, PositionSizeType.SIZE_100],
        description="Available position sizes"
    )
    
    # Action masking
    enable_action_masking: bool = Field(
        True,
        description="Enable invalid action masking"
    )
    min_cash_threshold: float = Field(
        100.0,
        ge=0,
        description="Minimum cash required for trading"
    )
    
    @property
    def action_space_size(self) -> int:
        """Calculate total action space size."""
        return len(self.action_types) * len(self.position_sizes)


class TerminationConfig(BaseModel):
    """Episode termination configuration."""
    
    max_episode_steps: int = Field(
        1000,
        ge=1,
        description="Maximum steps per episode"
    )
    min_equity_ratio: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Minimum equity ratio before bankruptcy"
    )
    max_drawdown: float = Field(
        0.2,
        ge=0,
        le=1,
        description="Maximum drawdown before termination"
    )
    max_invalid_actions: int = Field(
        10,
        ge=1,
        description="Maximum invalid actions before termination"
    )


class TradingEnvironmentConfig(BaseConfig):
    """Complete trading environment configuration."""
    
    # Market configuration
    symbol: Optional[str] = Field(
        None,
        description="Default trading symbol (can be overridden in reset)"
    )
    initial_cash: float = Field(
        25000.0,
        gt=0,
        description="Starting capital"
    )
    
    # Sub-configurations
    observation: ObservationConfig = Field(
        default_factory=ObservationConfig,
        description="Observation space configuration"
    )
    action: ActionConfig = Field(
        default_factory=ActionConfig,
        description="Action space configuration"
    )
    termination: TerminationConfig = Field(
        default_factory=TerminationConfig,
        description="Termination conditions"
    )
    
    # Execution configuration
    commission_rate: float = Field(
        0.001,
        ge=0,
        le=0.01,
        description="Commission rate per trade"
    )
    slippage_model: Literal["none", "linear", "sqrt", "market_impact"] = Field(
        "linear",
        description="Slippage model type"
    )
    
    # Rendering
    render_mode: Optional[Literal["human", "rgb_array"]] = Field(
        None,
        description="Rendering mode"
    )


# Reward configurations
class RewardComponentConfig(BaseModel):
    """Base reward component configuration."""
    
    weight: float = Field(
        1.0,
        description="Component weight in total reward"
    )
    enabled: bool = Field(
        True,
        description="Whether component is active"
    )
    clip_range: Optional[tuple[float, float]] = Field(
        None,
        description="Clip component output to range"
    )


class PnLComponentConfig(RewardComponentConfig):
    """P&L reward component configuration."""
    
    scaling_method: Literal["linear", "tanh", "log", "sqrt"] = Field(
        "tanh",
        description="P&L scaling method"
    )
    scale_factor: float = Field(
        0.001,
        gt=0,
        description="Scaling factor for P&L"
    )
    include_unrealized: bool = Field(
        True,
        description="Include unrealized P&L"
    )
    
    @validator('scale_factor')
    def validate_scale_factor(cls, v, values):
        """Validate scale factor based on method."""
        method = values.get('scaling_method')
        if method == 'tanh' and v > 0.01:
            raise ValueError("Large scale factor with tanh may saturate")
        return v


class RewardSystemConfig(BaseConfig):
    """Complete reward system configuration."""
    
    components: list[dict[str, Any]] = Field(
        [
            {"type": "pnl", "weight": 1.0, "config": {}},
            {"type": "action_penalty", "weight": 0.1, "config": {}},
            {"type": "risk", "weight": 0.2, "config": {}},
        ],
        description="List of reward components"
    )
    
    # Global settings
    clip_range: Optional[tuple[float, float]] = Field(
        (-10.0, 10.0),
        description="Global reward clipping range"
    )
    smoothing: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Exponential smoothing factor"
    )
    
    @validator('components')
    def validate_components(cls, v):
        """Validate component configurations."""
        for comp in v:
            if 'type' not in comp:
                raise ValueError("Each component must have a 'type'")
            if 'weight' not in comp:
                comp['weight'] = 1.0
        return v


# Training mode configurations
class CurriculumStage(BaseModel):
    """Single curriculum stage configuration."""
    
    name: str = Field(..., description="Stage name")
    symbols: list[str] = Field(..., min_items=1, description="Symbols to trade")
    min_performance: float = Field(
        0.0,
        description="Minimum performance to advance"
    )
    reset_point_quality: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Minimum reset point quality"
    )
    difficulty_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional difficulty parameters"
    )
    episodes_required: int = Field(
        100,
        ge=1,
        description="Minimum episodes before advancement"
    )


class ContinuousTrainingConfig(BaseConfig):
    """Continuous training mode configuration."""
    
    # Model management
    save_directory: Path = Field(
        Path("./models/continuous"),
        description="Directory for model checkpoints"
    )
    keep_top_k: int = Field(
        5,
        ge=1,
        description="Number of best models to keep"
    )
    
    # Evaluation settings
    evaluation_episodes: int = Field(
        10,
        ge=1,
        description="Episodes for model evaluation"
    )
    evaluation_frequency: int = Field(
        100,
        ge=1,
        description="Evaluate every N episodes"
    )
    
    # Improvement criteria
    improvement_metric: str = Field(
        "sharpe_ratio",
        description="Metric to optimize"
    )
    improvement_threshold: float = Field(
        0.01,
        gt=0,
        description="Minimum relative improvement"
    )
    patience: int = Field(
        20,
        ge=1,
        description="Episodes without improvement before action"
    )
    
    # Curriculum learning
    curriculum: list[CurriculumStage] = Field(
        [],
        description="Curriculum stages (empty for no curriculum)"
    )
    
    # Model versioning
    version_prefix: str = Field(
        "model_v",
        description="Prefix for model versions"
    )
    metadata_fields: list[str] = Field(
        ["total_episodes", "sharpe_ratio", "total_return", "win_rate"],
        description="Metrics to save in metadata"
    )
    
    @validator('save_directory')
    def validate_save_directory(cls, v):
        """Ensure save directory exists."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @root_validator
    def validate_curriculum(cls, values):
        """Validate curriculum progression."""
        curriculum = values.get('curriculum', [])
        if len(curriculum) > 1:
            # Ensure stages have increasing difficulty
            for i in range(1, len(curriculum)):
                if curriculum[i].min_performance <= curriculum[i-1].min_performance:
                    raise ValueError("Curriculum stages must have increasing performance requirements")
        return values


# Complete system configuration
class SystemConfig(BaseModel):
    """Complete system configuration."""
    
    mode: RunMode = Field(
        RunMode.TRAINING,
        description="Execution mode"
    )
    
    # Component configurations
    agent: Union[PPOConfig, dict[str, Any]] = Field(
        ...,
        description="Agent configuration"
    )
    environment: TradingEnvironmentConfig = Field(
        default_factory=TradingEnvironmentConfig,
        description="Environment configuration"
    )
    reward: RewardSystemConfig = Field(
        default_factory=RewardSystemConfig,
        description="Reward system configuration"
    )
    
    # Mode-specific configuration
    mode_config: Union[ContinuousTrainingConfig, dict[str, Any]] = Field(
        ...,
        description="Mode-specific configuration"
    )
    
    # Data configuration
    data: dict[str, Any] = Field(
        {"provider": "databento", "cache_dir": "./cache/data"},
        description="Data provider configuration"
    )
    
    # Monitoring configuration
    monitoring: dict[str, Any] = Field(
        {"wandb": {"project": "trading-rl"}, "tensorboard": {"log_dir": "./logs"}},
        description="Monitoring configuration"
    )
    
    @validator('agent')
    def validate_agent_config(cls, v, values):
        """Validate agent configuration matches mode."""
        if isinstance(v, dict) and 'type' in v:
            # Convert dict to appropriate config class
            agent_type = v['type']
            if agent_type == 'ppo':
                return PPOConfig(**v)
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True
        use_enum_values = False  # Keep enums as objects for system config
