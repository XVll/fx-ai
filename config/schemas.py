"""
Pydantic configuration schemas - Single source of truth for all configs
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class ActionType(str, Enum):
    """Trading action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SessionType(str, Enum):
    """Market session types"""
    PREMARKET = "PREMARKET"
    REGULAR = "REGULAR"
    POSTMARKET = "POSTMARKET"
    CLOSED = "CLOSED"


class ModelConfig(BaseModel):
    """Transformer model configuration"""
    model_config = ConfigDict(protected_namespaces=())
    
    # Architecture
    d_model: int = 64
    d_fused: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Branch-specific heads and layers
    hf_layers: int = 2
    mf_layers: int = 2
    lf_layers: int = 2
    portfolio_layers: int = 2
    static_layers: int = 2

    hf_heads: int = 4
    lf_heads: int = 4
    mf_heads: int = 4
    portfolio_heads: int = 4
    static_heads: int = 4

    # Feature dimensions
    hf_seq_len: int = 60
    hf_feat_dim: int = 20
    mf_seq_len: int = 30
    mf_feat_dim: int = 20  # Medium-frequency features (1m/5m timeframe)
    lf_seq_len: int = 30   # Low-frequency sequence length (daily/session timeframe)
    lf_feat_dim: int = 20  # Low-frequency features (daily timeframe)
    static_feat_dim: int = 5  # Static features (market cap, time encodings)
    portfolio_seq_len: int = 5  # Portfolio history length
    portfolio_feat_dim: int = 5  # Portfolio features (position, P&L, risk metrics)
    
    # Action space - Single source of truth
    action_dim: List[int] = Field(default=[3, 4], description="[action_types, position_sizes]")
    continuous_action: bool = False
    
    @field_validator('action_dim')
    @classmethod
    def validate_action_dim(cls, v):
        if len(v) != 2:
            raise ValueError("action_dim must have exactly 2 elements [action_types, position_sizes]")
        if v[0] != 3:  # BUY, SELL, HOLD
            raise ValueError("action_types must be 3")
        return v


class RewardComponentConfig(BaseModel):
    """Individual reward component configuration"""
    enabled: bool = True
    coefficient: float = 1.0
    
    
class RewardConfig(BaseModel):
    """Reward system configuration"""
    # Core components
    pnl: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=1.0),
        description="Profit and loss reward"
    )
    holding_penalty: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.001),
        description="Penalty for holding positions"
    )
    
    # Action efficiency
    action_penalty: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.0005),
        description="Penalty for excessive trading"
    )
    spread_penalty: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.1),
        description="Penalty for bid-ask spread costs"
    )
    
    # Risk management
    drawdown_penalty: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.5),
        description="Penalty for drawdowns"
    )
    bankruptcy_penalty: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=10.0),
        description="Large penalty for bankruptcy"
    )
    
    # Pattern rewards
    profitable_exit: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.2),
        description="Bonus for profitable exits"
    )
    quick_profit: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.1),
        description="Bonus for quick profits"
    )
    
    # Invalid action handling
    invalid_action_penalty: RewardComponentConfig = Field(
        default=RewardComponentConfig(coefficient=0.01),
        description="Penalty for invalid actions"
    )
    
    # Global settings
    scale_factor: float = Field(default=10.0, description="Global reward scaling")
    clip_range: List[float] = Field(default=[-10.0, 10.0], description="Reward clipping range")


class EnvConfig(BaseModel):
    """Trading environment configuration"""
    # Symbol settings
    symbol: str = Field(default="MLGO", description="Trading symbol")
    
    # Capital and risk
    initial_capital: float = Field(default=25000.0, description="Starting capital")
    max_position_size: float = Field(default=1.0, description="Max position as fraction of capital")
    leverage: float = Field(default=1.0, description="Trading leverage")
    
    # Trading costs
    commission_rate: float = Field(default=0.001, description="Trading commission rate")
    slippage_rate: float = Field(default=0.0005, description="Slippage rate")
    min_transaction_amount: float = Field(default=100.0, description="Minimum trade size")
    
    # Risk limits
    max_drawdown: float = Field(default=0.5, description="Maximum allowed drawdown")
    stop_loss_pct: float = Field(default=0.1, description="Stop loss percentage")
    daily_loss_limit: float = Field(default=0.2, description="Daily loss limit as fraction of capital")
    
    # Invalid action handling
    invalid_action_limit: Optional[int] = Field(default=None, description="Max invalid actions before termination")
    max_invalid_actions_per_episode: Optional[int] = Field(default=None, description="Alias for invalid_action_limit")
    
    # Reward configuration
    reward: RewardConfig = Field(default_factory=RewardConfig)
    
    # Features (for feature manager)
    feature_update_interval: int = Field(default=1, description="Steps between feature updates")
    
    # Episode settings
    max_episode_steps: Optional[int] = Field(default=None, description="Max steps per episode")
    max_steps: Optional[int] = Field(default=None, description="Alias for max_episode_steps")
    early_stop_loss_threshold: float = Field(default=0.9, description="Stop if equity < threshold * initial")
    random_reset: bool = Field(default=True, description="Random episode start within session")
    max_episode_loss_percent: float = Field(default=0.2, description="Max loss percentage before termination")
    bankruptcy_threshold_factor: float = Field(default=0.01, description="Bankruptcy threshold as fraction of initial capital")
    
    # Environment settings
    render_mode: Literal["human", "logs", "none"] = Field(default="none", description="Rendering mode")
    training_mode: bool = Field(default=True, description="Whether in training mode")
    
    @field_validator('max_invalid_actions_per_episode', mode='before')
    @classmethod
    def sync_invalid_action_limit(cls, v, info):
        """Keep max_invalid_actions_per_episode in sync with invalid_action_limit"""
        if v is None and info.data.get('invalid_action_limit') is not None:
            return info.data['invalid_action_limit']
        return v
    
    @field_validator('max_steps', mode='before')
    @classmethod
    def sync_max_steps(cls, v, info):
        """Keep max_steps in sync with max_episode_steps"""
        if v is None and info.data.get('max_episode_steps') is not None:
            return info.data['max_episode_steps']
        return v


class DataConfig(BaseModel):
    """Data source configuration"""
    provider: Literal["databento"] = "databento"
    
    # Databento specific
    data_dir: str = Field(default="dnb", description="Data directory")
    symbols: List[str] = Field(default=["MLGO"], description="Symbols to load")
    
    # Data types to load
    load_trades: bool = True
    load_quotes: bool = True
    load_order_book: bool = True
    load_ohlcv: bool = True
    
    # Date range
    start_date: Optional[str] = Field(default="2025-03-27", description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default="2025-03-27", description="End date YYYY-MM-DD")
    
    # Activity filtering for day selection
    min_activity_score: float = Field(default=0.0, description="Minimum activity score for training days")
    max_activity_score: float = Field(default=1.0, description="Maximum activity score for training days")
    
    # Training order
    training_order: Literal["random", "sequential", "activity_desc", "activity_asc"] = Field(
        default="activity_desc", 
        description="Order to process training days"
    )
    
    # Performance
    cache_enabled: bool = True
    cache_dir: str = "cache"
    preload_days: int = Field(default=2, description="Days to preload for session features")
    
    # Momentum indices
    index_dir: str = Field(default="indices", description="Directory for momentum indices")
    auto_build_index: bool = Field(default=True, description="Auto-build index if missing")


class TrainingConfig(BaseModel):
    """Training configuration"""
    # Basic settings
    device: str = Field(default="cuda", description="Training device")
    seed: int = Field(default=42, description="Random seed")
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Rollout settings
    rollout_steps: int = Field(default=4096, description="Steps per rollout")
    
    # Learning rate schedule
    use_lr_annealing: bool = True
    lr_annealing_factor: float = 0.5
    lr_annealing_patience: int = 50
    min_learning_rate: float = 1e-6
    
    # Continuous training
    continue_training: bool = False
    total_updates: int = Field(default=500, description="Total training updates")
    checkpoint_interval: int = Field(default=10, description="Updates between checkpoints")
    keep_best_n_models: int = Field(default=5, description="Number of best models to keep")
    
    # Early stopping
    early_stop_patience: int = Field(default=100, description="Updates without improvement before stopping")
    early_stop_min_delta: float = Field(default=0.01, description="Minimum improvement to reset patience")
    
    # Evaluation
    eval_frequency: int = Field(default=10, description="Updates between evaluations")
    eval_episodes: int = Field(default=5, description="Episodes for evaluation")
    
    # Model selection metric
    best_model_metric: str = Field(default="reward", description="Metric for model selection")


class SimulationConfig(BaseModel):
    """Market simulation configuration"""
    # Execution simulation
    execution_delay_ms: int = Field(default=50, description="Order execution delay")
    partial_fill_probability: float = Field(default=0.0, description="Probability of partial fills")
    allow_shorting: bool = Field(default=False, description="Allow short selling (default: long-only)")
    
    # Latency simulation
    mean_latency_ms: float = Field(default=50.0, description="Mean execution latency")
    latency_std_dev_ms: float = Field(default=10.0, description="Latency standard deviation")
    
    # Slippage parameters
    base_slippage_bps: float = Field(default=5.0, description="Base slippage in basis points")
    size_impact_slippage_bps_per_unit: float = Field(default=0.1, description="Size impact slippage")
    max_total_slippage_bps: float = Field(default=50.0, description="Max total slippage")
    
    # Cost parameters
    commission_per_share: float = Field(default=0.005, description="Commission per share")
    fee_per_share: float = Field(default=0.001, description="Fee per share")
    min_commission_per_order: float = Field(default=1.0, description="Minimum commission per order")
    max_commission_pct_of_value: float = Field(default=0.5, description="Max commission as % of trade value")
    
    # Market impact
    market_impact_model: Literal["linear", "square_root", "none"] = "linear"
    market_impact_coefficient: float = 0.0001
    
    # Spread modeling
    spread_model: Literal["fixed", "dynamic", "historical"] = "historical"
    fixed_spread_bps: float = Field(default=10.0, description="Fixed spread in basis points")
    
    # Random start for training
    random_start_prob: float = Field(default=0.8, description="Probability of random episode start")
    warmup_steps: int = Field(default=60, description="Steps to warmup features before trading")
    
    # Portfolio configuration
    initial_cash: float = Field(default=25000.0, description="Initial portfolio cash")
    max_position_value_ratio: float = Field(default=1.0, description="Max position value as ratio of portfolio")
    max_position_holding_seconds: Optional[int] = Field(default=None, description="Max seconds to hold a position")
    default_position_value: float = Field(default=10000.0, description="Default position value for sizing")


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Console output
    console_enabled: bool = True
    console_format: str = "simple"  # "simple" or "detailed"
    
    # File output
    file_enabled: bool = True
    log_dir: str = "logs"
    
    # Metrics logging
    log_interval: int = Field(default=10, description="Steps between metric logs")
    
    # Component-specific logging
    log_rewards: bool = True
    log_actions: bool = True
    log_features: bool = False  # Can be verbose
    log_portfolio: bool = True


class WandbConfig(BaseModel):
    """Weights & Biases configuration"""
    enabled: bool = True
    project: str = "fx-ai-v2"
    entity: Optional[str] = None
    
    # Run settings
    name: Optional[str] = Field(default=None, description="Run name (auto-generated if None)")
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    
    # Logging settings
    log_frequency: Dict[str, int] = Field(
        default_factory=lambda: {
            "training": 1,
            "episode": 1,
            "rollout": 1,
            "evaluation": 1
        }
    )
    
    # What to save
    save_code: bool = True
    save_model: bool = True


class ActivityScoringConfig(BaseModel):
    """Activity-based scoring configuration"""
    # Score calculation
    volume_ratio_cap: float = Field(default=5.0, description="Max volume ratio for scoring")
    price_change_cap: float = Field(default=0.10, description="Max price change for scoring (10%)")
    
    # Reset point generation
    reset_points_per_day: int = Field(default=25, description="Number of reset points per day")
    min_activity_threshold: float = Field(default=0.1, description="Minimum activity score to include")
    
    # Direction detection
    front_side_threshold: float = Field(default=0.05, description="Min positive move for front side")
    back_side_threshold: float = Field(default=-0.05, description="Min negative move for back side")
    
    # Randomization windows (minutes) based on activity level
    high_activity_window: List[int] = Field(default=[3, 5], description="Window for activity >= 0.8")
    medium_activity_window: List[int] = Field(default=[10, 15], description="Window for 0.3-0.8")
    low_activity_window: List[int] = Field(default=[20, 30], description="Window for < 0.3")


class CurriculumStageConfig(BaseModel):
    """Configuration for a curriculum training stage"""
    episode_range: List[Optional[int]] = Field(description="[start, end] episode range")
    min_activity_score: float = Field(default=0.0, description="Minimum activity score for days")
    direction_filter: Literal["both", "front_side", "back_side"] = Field(
        default="both", 
        description="Direction filter for training"
    )


class CurriculumConfig(BaseModel):
    """Curriculum learning configuration"""
    stage_1_beginner: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[0, 1000],
            min_activity_score=0.7,
            direction_filter="both"
        )
    )
    stage_2_intermediate: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[1000, 3000],
            min_activity_score=0.3,
            direction_filter="both"
        )
    )
    stage_3_advanced: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[3000, 5000],
            min_activity_score=0.0,
            direction_filter="both"
        )
    )
    stage_4_specialization: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[5000, None],
            min_activity_score=0.0,
            direction_filter="both"
        )
    )


class DashboardConfig(BaseModel):
    """Live dashboard configuration"""
    enabled: bool = True
    port: int = 8050
    update_interval: float = Field(default=1.0, description="Seconds between updates")
    
    # Display settings
    max_episodes_shown: int = 20
    max_trades_shown: int = 100
    chart_height: int = 400
    
    # Features to show in heatmap
    heatmap_features: List[str] = Field(
        default_factory=lambda: [
            "price_velocity", "tape_imbalance", "spread_compression",
            "1m_ema9_distance", "1m_swing_high_distance"
        ]
    )


class Config(BaseModel):
    """Root configuration object"""
    model_config = ConfigDict(extra="forbid")  # Fail on unknown fields
    
    # All sub-configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    # Activity-based configurations
    activity_scoring: ActivityScoringConfig = Field(default_factory=ActivityScoringConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    
    # Experiment settings
    experiment_name: str = Field(default="default", description="Experiment identifier")
    mode: Literal["train", "eval", "backtest"] = "train"
    
    @classmethod
    def load(cls, overrides_path: Optional[str] = None) -> "Config":
        """Load config with optional YAML overrides"""
        # Start with default config
        config = cls()
        
        # Apply overrides if provided
        if overrides_path:
            import yaml
            with open(overrides_path) as f:
                overrides = yaml.safe_load(f)
            
            # Update config with overrides
            config = config.model_copy(update=overrides, deep=True)
        
        return config
    
    def save_used_config(self, path: str):
        """Save the actual config used for reproducibility"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)