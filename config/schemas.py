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
    
    # Architecture - optimized for momentum trading
    d_model: int = 128
    d_fused: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Branch-specific heads and layers - balanced for multi-timeframe analysis
    hf_layers: int = 3
    mf_layers: int = 3
    lf_layers: int = 2
    portfolio_layers: int = 2

    hf_heads: int = 8
    lf_heads: int = 4
    mf_heads: int = 8
    portfolio_heads: int = 4

    # Feature dimensions
    hf_seq_len: int = 60
    hf_feat_dim: int = 7  # High-frequency features (1s timeframe) - removed 5 velocity features, added 3 aggregated
    mf_seq_len: int = 30
    mf_feat_dim: int = 43  # MF features - using professional pandas/ta library instead of manual calculations (net -2)
    lf_seq_len: int = 30   # Low-frequency sequence length (daily/session timeframe)
    lf_feat_dim: int = 19  # LF features - original + LULD + adaptive + session/time context (moved from misnamed "static")
    portfolio_seq_len: int = 5  # Portfolio history length
    portfolio_feat_dim: int = 10  # Portfolio features (position, P&L, risk metrics, MFE/MAE)
    
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


class RewardConfig(BaseModel):
    """Percentage-based reward system configuration - all rewards as % of account value"""
    
    # Core P&L reward (most important)
    pnl_coefficient: float = Field(default=100.0, description="P&L scaling: 1% profit = coefficient reward (default: 1% = 1.0 reward)")
    
    # Risk management penalties
    holding_penalty_coefficient: float = Field(default=2.0, description="Holding time penalty: max holding = -coefficient penalty")
    drawdown_penalty_coefficient: float = Field(default=5.0, description="Drawdown penalty: 1% drawdown = -coefficient penalty") 
    bankruptcy_penalty_coefficient: float = Field(default=50.0, description="Bankruptcy penalty: fixed large penalty")
    
    # MFE/MAE based penalties
    profit_giveback_penalty_coefficient: float = Field(default=2.0, description="Penalty for giving back MFE profits (reduced from 10.0)")
    profit_giveback_threshold: float = Field(default=0.3, description="Threshold for profit giveback penalty (30%)")
    max_drawdown_penalty_coefficient: float = Field(default=15.0, description="Penalty for exceeding MAE thresholds")
    max_drawdown_threshold_percent: float = Field(default=0.01, description="MAE threshold as % of account (1%)")
    
    # Trading behavior bonuses
    profit_closing_bonus_coefficient: float = Field(default=100.0, description="Bonus for closing profitable trades, scales with profit")
    
    # Clean trade bonus (exponential scaling configuration)
    clean_trade_coefficient: float = Field(default=20.0, description="DEPRECATED: Legacy parameter, use base_multiplier instead")
    max_clean_drawdown_percent: float = Field(default=0.01, description="DEPRECATED: Legacy parameter, use max_mae_threshold instead")
    base_multiplier: float = Field(default=5000, description="Base scaling multiplier for clean trade bonus")
    max_mae_threshold: float = Field(default=0.02, description="Maximum allowed MAE drawdown (2%)")
    min_gain_threshold: float = Field(default=0.01, description="Minimum gain required for clean trade bonus (1%)")
    
    # Trading activity incentives
    activity_bonus_per_trade: float = Field(default=0.025, description="Bonus for each trading action to encourage activity")
    hold_penalty_per_step: float = Field(default=0.01, description="Small penalty per HOLD action to create opportunity cost")
    
    # Thresholds and limits
    max_holding_time_steps: int = Field(default=180, description="Maximum holding time before penalties (3 minutes)")
    
    # Component enable/disable flags
    enable_pnl_reward: bool = Field(default=True, description="Enable P&L reward component")
    enable_holding_penalty: bool = Field(default=True, description="Enable holding time penalty")
    enable_drawdown_penalty: bool = Field(default=True, description="Enable drawdown penalty")
    enable_profit_giveback_penalty: bool = Field(default=True, description="Enable profit giveback penalty (MFE protection)")
    enable_max_drawdown_penalty: bool = Field(default=True, description="Enable max drawdown penalty (MAE protection)")
    enable_profit_closing_bonus: bool = Field(default=True, description="Enable profit closing bonus")
    enable_clean_trade_bonus: bool = Field(default=True, description="Enable clean trade bonus")
    enable_trading_activity_bonus: bool = Field(default=True, description="Enable trading activity bonus")
    enable_inactivity_penalty: bool = Field(default=True, description="Enable inactivity penalty")


class EnvConfig(BaseModel):
    """Trading environment configuration"""
    # Symbol settings
    symbol: str = Field(default="MLGO", description="Trading symbol")
    
    # Capital and risk - optimized for momentum trading
    initial_capital: float = Field(default=25000.0, description="Starting capital")
    max_position_size: float = Field(default=1.0, description="Max position as fraction of capital")
    leverage: float = Field(default=1.0, description="Trading leverage")
    
    # Trading costs - realistic for retail trading
    commission_rate: float = Field(default=0.001, description="Trading commission rate")
    slippage_rate: float = Field(default=0.0005, description="Slippage rate")
    min_transaction_amount: float = Field(default=100.0, description="Minimum trade size")
    
    # Risk limits - appropriate for momentum strategies
    max_drawdown: float = Field(default=0.3, description="Maximum allowed drawdown")
    stop_loss_pct: float = Field(default=0.15, description="Stop loss percentage")
    daily_loss_limit: float = Field(default=0.25, description="Daily loss limit as fraction of capital")
    
    # Invalid action handling
    invalid_action_limit: Optional[int] = Field(default=None, description="Max invalid actions before termination")
    max_invalid_actions_per_episode: Optional[int] = Field(default=None, description="Alias for invalid_action_limit")
    
    # Reward configuration
    reward: RewardConfig = Field(default_factory=RewardConfig)
    
    # Features (for feature manager)
    feature_update_interval: int = Field(default=1, description="Steps between feature updates")
    
    # Episode settings - optimized for momentum patterns
    max_episode_steps: int = Field(default=256, description="Natural episode length - no penalty when reached")
    max_training_steps: Optional[int] = Field(default=None, description="Training step limit with penalty if reached")
    max_steps: int = Field(default=1000, description="Legacy alias - maps to max_episode_steps")
    early_stop_loss_threshold: float = Field(default=0.85, description="Stop if equity < threshold * initial")
    random_reset: bool = Field(default=True, description="Random episode start within session")
    max_episode_loss_percent: float = Field(default=0.15, description="Max loss percentage before termination")
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
        """Keep max_steps in sync with max_episode_steps (legacy compatibility)"""
        # max_steps is legacy alias for max_episode_steps
        if v is None:
            return info.data.get('max_episode_steps', 2048)
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
    
    # Date range - full momentum dataset
    start_date: Optional[str] = Field(default="2025-02-03", description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default="2025-04-29", description="End date YYYY-MM-DD")
    
    # Performance
    cache_enabled: bool = True
    cache_dir: str = "cache"
    preload_days: int = Field(default=2, description="Days to preload for session features")
    
    # Index configuration
    index_dir: str = Field(default="cache/indices", description="Directory for momentum indices")
    auto_build_index: bool = Field(default=True, description="Auto-build index if missing")


class TrainingConfig(BaseModel):
    """Training configuration"""
    # Basic settings
    device: str = Field(default="cuda", description="Training device")
    seed: int = Field(default=42, description="Random seed")
    
    # PPO hyperparameters - optimized for momentum trading
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Rollout settings - larger for stability
    rollout_steps: int = Field(default=2048, description="Steps per rollout")
    
    # Learning rate schedule
    use_lr_annealing: bool = True
    lr_annealing_factor: float = 0.5
    lr_annealing_patience: int = 100
    min_learning_rate: float = 1e-6
    
    # Continuous training - production ready
    continue_training: bool = False
    total_updates: int = Field(default=3000, description="Total training updates")
    checkpoint_interval: int = Field(default=50, description="Updates between checkpoints")
    keep_best_n_models: int = Field(default=5, description="Number of best models to keep")
    
    # Early stopping - more patient for momentum learning
    early_stop_patience: int = Field(default=300, description="Updates without improvement before stopping")
    early_stop_min_delta: float = Field(default=0.01, description="Minimum improvement to reset patience")
    
    # Evaluation
    eval_frequency: int = Field(default=25, description="Updates between evaluations")
    eval_episodes: int = Field(default=10, description="Episodes for evaluation")
    
    # Model selection metric
    best_model_metric: str = Field(default="mean_reward", description="Metric for model selection")


class SimulationConfig(BaseModel):
    """Market simulation configuration"""
    # Execution simulation - realistic for retail momentum trading
    execution_delay_ms: int = Field(default=100, description="Order execution delay")
    partial_fill_probability: float = Field(default=0.0, description="Probability of partial fills")
    allow_shorting: bool = Field(default=False, description="Allow short selling (default: long-only)")
    
    # Latency simulation - realistic retail latency
    mean_latency_ms: float = Field(default=100.0, description="Mean execution latency")
    latency_std_dev_ms: float = Field(default=20.0, description="Latency standard deviation")
    
    # Slippage parameters - appropriate for low-float stocks
    base_slippage_bps: float = Field(default=10.0, description="Base slippage in basis points")
    size_impact_slippage_bps_per_unit: float = Field(default=0.2, description="Size impact slippage")
    max_total_slippage_bps: float = Field(default=100.0, description="Max total slippage")
    
    # Cost parameters - realistic retail trading costs
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
    
    # Random start for training - high randomization for momentum patterns
    random_start_prob: float = Field(default=0.95, description="Probability of random episode start")
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
    project: str = "fx-ai-momentum"
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


class MomentumScanningConfig(BaseModel):
    """Momentum day scanning configuration - simplified without direction"""
    # Day selection criteria
    min_daily_move: float = Field(default=0.10, description="Minimum 10% intraday movement")
    min_volume_multiplier: float = Field(default=2.0, description="Minimum 2x average volume")
    max_daily_move: Optional[float] = Field(default=None, description="Remove cap - capture all volatility")
    max_volume_multiplier: Optional[float] = Field(default=None, description="Remove cap - capture all volume spikes")




class SessionVolumeConfig(BaseModel):
    """Session-aware volume calculations"""
    # Market sessions for volume profiling
    premarket_start: str = Field(default="04:00", description="Pre-market start time")
    premarket_end: str = Field(default="09:30", description="Pre-market end time")
    regular_start: str = Field(default="09:30", description="Regular market start time")
    regular_end: str = Field(default="16:00", description="Regular market end time")
    postmarket_start: str = Field(default="16:00", description="Post-market start time")
    postmarket_end: str = Field(default="20:00", description="Post-market end time")
    
    # Volume baseline calculation
    volume_window_days: int = Field(default=10, description="Days for volume baseline")
    use_session_specific_baselines: bool = Field(default=True, description="Separate volume baselines per session")


class ProgressiveEpisodeConfig(BaseModel):
    """Progressive episode length configuration for sniper training"""
    # Stage-based episode lengths (steps)
    stage_1_length: int = Field(default=256, description="Stage 1: Basic entry/exit (4.3 min)")
    stage_2_length: int = Field(default=512, description="Stage 2: Momentum riding (8.5 min)")
    stage_3_length: int = Field(default=768, description="Stage 3: Full cycles (12.8 min)")
    stage_4_length: int = Field(default=1024, description="Stage 4: Complex strategies (17 min)")
    
    # Batch sizes for each stage (reduced for more responsive training)
    stage_1_batch_size: int = Field(default=2048, description="Stage 1 batch size (~34 min)")
    stage_2_batch_size: int = Field(default=4096, description="Stage 2 batch size (~68 min)")
    stage_3_batch_size: int = Field(default=6144, description="Stage 3 batch size (~102 min)")
    stage_4_batch_size: int = Field(default=8192, description="Stage 4 batch size (~136 min)")
    
    # Offset strategies
    stage_1_offset_ratio: float = Field(default=1.0, description="Stage 1: No overlap")
    stage_2_offset_ratio: float = Field(default=1.0, description="Stage 2: No overlap")
    stage_3_offset_ratio: float = Field(default=0.75, description="Stage 3: 25% overlap")
    stage_4_offset_ratio: float = Field(default=0.75, description="Stage 4: 25% overlap")




class CurriculumStageConfig(BaseModel):
    """Configuration for a curriculum training stage with direct range-based approach"""
    episode_range: List[Optional[int]] = Field(description="[start, end] episode range")
    episode_length: int = Field(description="Episode length in steps")
    batch_size: int = Field(description="Rollout batch size")
    offset_ratio: float = Field(description="Episode offset ratio (1.0 = no overlap)")
    
    # Direct range configuration
    roc_range: List[float] = Field(default=[0.0, 1.0], description="ROC score range [min, max]")
    activity_range: List[float] = Field(default=[0.0, 1.0], description="Activity score range [min, max]")
    
    @field_validator('roc_range', 'activity_range')
    @classmethod
    def validate_ranges(cls, v):
        if len(v) != 2 or v[0] >= v[1] or not (0 <= v[0] <= 1) or not (0 <= v[1] <= 1):
            raise ValueError("Range must be [min, max] with 0 <= min < max <= 1")
        return v




class MomentumScoringConfig(BaseModel):
    """Configuration for momentum scoring system - directional ROC and activity scores"""
    
    # ROC scoring  
    roc_lookback_minutes: int = Field(default=5, description="Minutes for ROC calculation")
    
    # Activity scoring
    activity_lookback_minutes: int = Field(default=10, description="Minutes for volume rolling average")
    
    # Reset point generation
    min_reset_points: int = Field(default=60, description="Minimum reset points per day")
    
    # Weights for combined scoring
    roc_weight: float = Field(default=0.6, description="ROC score weight")
    activity_weight: float = Field(default=0.4, description="Activity score weight")


class CurriculumConfig(BaseModel):
    """Range-based curriculum learning configuration"""
    
    # Momentum scoring configuration
    scoring: MomentumScoringConfig = Field(default_factory=MomentumScoringConfig)
    
    # Training stages with direct range assignments
    stage_1_beginner: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[0, 2000],
            episode_length=256,
            batch_size=2048,
            offset_ratio=1.0,
            roc_range=[0.05, 1.0],
            activity_range=[0.0, 1.0]
        )
    )
    stage_2_intermediate: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[2000, 5000],
            episode_length=512,
            batch_size=4096,
            offset_ratio=1.0,
            roc_range=[0.6, 1.0],
            activity_range=[0.3, 1.0]
        )
    )
    stage_3_advanced: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[5000, 8000],
            episode_length=768,
            batch_size=6144,
            offset_ratio=0.75,
            roc_range=[0.4, 1.0],
            activity_range=[0.2, 1.0]
        )
    )
    stage_4_specialization: CurriculumStageConfig = Field(
        default=CurriculumStageConfig(
            episode_range=[8000, None],
            episode_length=1024,
            batch_size=8192,
            offset_ratio=0.75,
            roc_range=[0.0, 1.0],
            activity_range=[0.0, 1.0]
        )
    )


class DashboardConfig(BaseModel):
    """Live dashboard configuration"""
    enabled: bool = True
    port: int = 8051
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
    
    # Momentum scanning configurations
    momentum_scanning: MomentumScanningConfig = Field(default_factory=MomentumScanningConfig)
    momentum_scoring: MomentumScoringConfig = Field(default_factory=MomentumScoringConfig)
    session_volume: SessionVolumeConfig = Field(default_factory=SessionVolumeConfig)
    progressive_episodes: ProgressiveEpisodeConfig = Field(default_factory=ProgressiveEpisodeConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    
    # Experiment settings
    experiment_name: str = Field(default="momentum_training", description="Experiment identifier")
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