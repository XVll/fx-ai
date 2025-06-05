"""
Cleaned and organized configuration schemas for fx-ai trading system.
Single source of truth with removed duplicates and unused parameters.
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


# =============================================================================
# CORE ENUMS
# =============================================================================


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


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================



class ModelConfig(BaseModel):
    """Multi-branch transformer model configuration"""

    model_config = ConfigDict(protected_namespaces=())

    # Core architecture
    d_model: int = Field(128, description="Model dimension")
    d_fused: int = Field(512, description="Fused representation dimension")
    n_heads: int = Field(8, description="Attention heads")
    n_layers: int = Field(6, description="Transformer layers")
    d_ff: int = Field(2048, description="Feed-forward dimension")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")

    # Branch-specific configuration
    hf_layers: int = Field(3, description="High-frequency branch layers")
    mf_layers: int = Field(3, description="Medium-frequency branch layers")
    lf_layers: int = Field(2, description="Low-frequency branch layers")
    portfolio_layers: int = Field(2, description="Portfolio branch layers")

    hf_heads: int = Field(8, description="High-frequency attention heads")
    mf_heads: int = Field(8, description="Medium-frequency attention heads")
    lf_heads: int = Field(4, description="Low-frequency attention heads")
    portfolio_heads: int = Field(4, description="Portfolio attention heads")

    # Feature dimensions (aligned with actual implementation)
    hf_seq_len: int = Field(60, description="High-frequency sequence length")
    hf_feat_dim: int = Field(9, description="High-frequency features (corrected)")
    mf_seq_len: int = Field(30, description="Medium-frequency sequence length")
    mf_feat_dim: int = Field(43, description="Medium-frequency features")
    lf_seq_len: int = Field(30, description="Low-frequency sequence length")
    lf_feat_dim: int = Field(19, description="Low-frequency features")
    portfolio_seq_len: int = Field(5, description="Portfolio sequence length")
    portfolio_feat_dim: int = Field(10, description="Portfolio features")

    # Action space
    action_dim: List[int] = Field([3, 4], description="[action_types, position_sizes]")

    @field_validator("action_dim", mode="before")  # noinspection PyNestedDecorators
    @classmethod
    def validate_action_dim(cls, v: Any) -> Any:
        if len(v) != 2:
            raise ValueError("action_dim must have exactly 2 elements")
        if v[0] != 3:
            raise ValueError("action_types must be 3 (BUY, SELL, HOLD)")
        return v


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


class TrainingConfig(BaseModel):
    """PPO training configuration"""

    # Core settings
    device: str = Field("mps", description="Training device")
    seed: int = Field(42, description="Random seed")

    # PPO hyperparameters
    learning_rate: float = Field(1.5e-4, gt=0.0, description="Learning rate")
    batch_size: int = Field(64, gt=0, description="Batch size")
    n_epochs: int = Field(8, gt=0, description="Training epochs per update")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor")
    gae_lambda: float = Field(0.95, ge=0.0, le=1.0, description="GAE lambda")
    clip_epsilon: float = Field(0.15, gt=0.0, description="PPO clip range")
    value_coef: float = Field(0.5, ge=0.0, description="Value function coefficient")
    entropy_coef: float = Field(0.01, ge=0.0, description="Entropy coefficient")
    max_grad_norm: float = Field(0.3, gt=0.0, description="Gradient clipping")

    # Rollout settings
    rollout_steps: int = Field(2048, gt=0, description="Steps per rollout")


    # Model management
    continue_training: bool = Field(False, description="Continue from best model")
    checkpoint_interval: int = Field(50, description="Updates between checkpoints")
    keep_best_n_models: int = Field(5, description="Number of best models to keep")

    # Early stopping
    early_stop_patience: int = Field(300, description="Updates without improvement")
    early_stop_min_delta: float = Field(0.01, description="Minimum improvement")

    # Evaluation
    eval_frequency: int = Field(5, description="Updates between evaluations")
    eval_episodes: int = Field(10, description="Episodes for evaluation")
    best_model_metric: str = Field("mean_reward", description="Model selection metric")


# =============================================================================
# REWARD SYSTEM CONFIGURATION
# =============================================================================


class RewardConfig(BaseModel):
    """Modular reward system configuration"""

    # Core PnL rewards
    pnl_coefficient: float = Field(100.0, description="P&L scaling coefficient")

    # Risk management
    holding_penalty_coefficient: float = Field(2.0, description="Holding time penalty")
    drawdown_penalty_coefficient: float = Field(5.0, description="Drawdown penalty")
    bankruptcy_penalty_coefficient: float = Field(
        50.0, description="Bankruptcy penalty"
    )

    # MFE/MAE penalties
    profit_giveback_penalty_coefficient: float = Field(
        2.0, description="Profit giveback penalty"
    )
    profit_giveback_threshold: float = Field(0.3, description="Giveback threshold")
    max_drawdown_penalty_coefficient: float = Field(
        15.0, description="Max drawdown penalty"
    )
    max_drawdown_threshold_percent: float = Field(0.01, description="MAE threshold")

    # Trading bonuses
    profit_closing_bonus_coefficient: float = Field(
        100.0, description="Profit closing bonus"
    )
    base_multiplier: float = Field(5000, description="Clean trade base multiplier")
    max_mae_threshold: float = Field(0.02, description="Max allowed MAE")
    min_gain_threshold: float = Field(0.01, description="Min gain for clean trade")

    # Activity incentives
    activity_bonus_per_trade: float = Field(0.025, description="Trading activity bonus")
    hold_penalty_per_step: float = Field(0.01, description="Hold action penalty")
    action_penalty_coefficient: float = Field(
        0.1, description="Action penalty coefficient"
    )
    quick_profit_bonus_coefficient: float = Field(
        1.0, description="Quick profit bonus coefficient"
    )

    # Limits
    max_holding_time_steps: int = Field(180, description="Max holding time")

    # Component toggles
    enable_pnl_reward: bool = Field(True, description="Enable P&L reward")
    enable_holding_penalty: bool = Field(True, description="Enable holding penalty")
    enable_drawdown_penalty: bool = Field(True, description="Enable drawdown penalty")
    enable_profit_giveback_penalty: bool = Field(
        True, description="Enable giveback penalty"
    )
    enable_max_drawdown_penalty: bool = Field(
        True, description="Enable max drawdown penalty"
    )
    enable_profit_closing_bonus: bool = Field(True, description="Enable profit bonus")
    enable_clean_trade_bonus: bool = Field(True, description="Enable clean trade bonus")
    enable_trading_activity_bonus: bool = Field(
        True, description="Enable activity bonus"
    )
    enable_inactivity_penalty: bool = Field(
        True, description="Enable inactivity penalty"
    )


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================


class EnvironmentConfig(BaseModel):
    """Trading environment settings"""

    # Episode control
    max_steps: int = Field(256, gt=0, description="Maximum steps per episode")
    max_training_steps: Optional[int] = Field(
        None, description="Maximum training steps"
    )

    # Training mode
    use_momentum_training: bool = Field(True, description="Use momentum-based training")

    # Termination conditions
    early_stop_loss_threshold: float = Field(0.85, description="Early stop threshold")
    max_episode_loss_percent: float = Field(0.15, description="Max episode loss")
    bankruptcy_threshold_factor: float = Field(0.01, description="Bankruptcy threshold")

    # Environment settings
    random_reset: bool = Field(True, description="Random episode start")
    render_mode: Literal["human", "logs", "none"] = Field(
        "none", description="Render mode"
    )
    feature_update_interval: int = Field(1, description="Feature update interval")

    # Reward system
    reward: RewardConfig = Field(default_factory=lambda: RewardConfig())

    # Training management  
    training_manager: "TrainingManagerConfig" = Field(default_factory=lambda: TrainingManagerConfig())


# =============================================================================
# DATA CONFIGURATION
# =============================================================================


class DataConfig(BaseModel):
    """Data source and processing configuration"""

    # Provider settings
    provider: Literal["databento"] = Field("databento", description="Data provider")
    data_dir: str = Field("dnb", description="Data directory")
    symbols: List[str] = Field(["MLGO"], description="Trading symbols")

    # Data types
    load_trades: bool = Field(True, description="Load trade data")
    load_quotes: bool = Field(True, description="Load quote data")
    load_order_book: bool = Field(True, description="Load order book data")
    load_ohlcv: bool = Field(True, description="Load OHLCV data")

    # Caching
    cache_enabled: bool = Field(True, description="Enable data caching")
    cache_dir: str = Field("cache", description="Cache directory")
    preload_days: int = Field(2, description="Days to preload")

    # Index configuration
    index_dir: str = Field("cache/indices", description="Index directory")
    auto_build_index: bool = Field(True, description="Auto-build index")


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================


class SimulationConfig(BaseModel):
    """Market simulation and trading parameters"""

    # Capital settings
    initial_capital: float = Field(25000.0, gt=0.0, description="Starting capital")
    max_position_value_ratio: float = Field(
        1.0, ge=0.0, le=1.0, description="Max position as fraction of equity"
    )
    leverage: float = Field(1.0, gt=0.0, description="Trading leverage")

    # Trading costs
    commission_rate: float = Field(0.001, ge=0.0, description="Commission rate")
    slippage_rate: float = Field(0.0005, ge=0.0, description="Slippage rate")
    min_transaction_amount: float = Field(100.0, ge=0.0, description="Min trade size")

    # Risk limits
    max_drawdown: float = Field(0.3, ge=0.0, le=1.0, description="Max allowed drawdown")
    stop_loss_pct: float = Field(
        0.15, ge=0.0, le=1.0, description="Stop loss percentage"
    )
    daily_loss_limit: float = Field(
        0.25, ge=0.0, le=1.0, description="Daily loss limit"
    )

    # Execution settings
    execution_delay_ms: int = Field(100, description="Order execution delay")
    partial_fill_probability: float = Field(0.0, description="Partial fill probability")
    allow_shorting: bool = Field(False, description="Allow short selling")

    # Latency simulation
    mean_latency_ms: float = Field(100.0, description="Mean execution latency")
    latency_std_dev_ms: float = Field(20.0, description="Latency std dev")

    # Slippage parameters
    base_slippage_bps: float = Field(10.0, description="Base slippage (bps)")
    size_impact_slippage_bps_per_unit: float = Field(
        0.2, description="Size impact slippage"
    )
    max_total_slippage_bps: float = Field(100.0, description="Max total slippage")

    # Cost parameters
    commission_per_share: float = Field(0.005, description="Commission per share")
    fee_per_share: float = Field(0.001, description="Fee per share")
    min_commission_per_order: float = Field(1.0, description="Min commission")
    max_commission_pct_of_value: float = Field(0.5, description="Max commission %")

    # Market impact
    market_impact_model: Literal["linear", "square_root", "none"] = Field("linear")
    market_impact_coefficient: float = Field(0.0001, description="Market impact coeff")

    # Spread modeling
    spread_model: Literal["fixed", "dynamic", "historical"] = Field("historical")
    fixed_spread_bps: float = Field(10.0, description="Fixed spread (bps)")

    # Episode randomization
    random_start_prob: float = Field(0.95, description="Random start probability")
    warmup_steps: int = Field(60, description="Warmup steps")

    # Portfolio settings
    max_position_holding_seconds: Optional[int] = Field(
        None, description="Max holding time"
    )


# =============================================================================
# TRAINING MANAGER CONFIGURATION
# =============================================================================


class TerminationConfig(BaseModel):
    """Training termination configuration"""
    
    # Hard limits (always enforced) - for ending entire training
    training_max_episodes: Optional[int] = Field(None, description="Maximum total episodes before training termination")
    training_max_updates: Optional[int] = Field(None, description="Maximum total updates before training termination")
    training_max_cycles: Optional[int] = Field(None, description="Maximum total data cycles before training termination")
    
    # Intelligent termination (production mode only)
    intelligent_termination: bool = Field(True, description="Enable intelligent termination")
    plateau_patience: int = Field(50, description="Updates without improvement before plateau termination")
    degradation_threshold: float = Field(0.05, description="Performance degradation threshold (5%)")


# EpisodeConfig removed - moved to data_lifecycle


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    frequency: int = Field(50, description="Updates between evaluations")
    episodes: int = Field(10, description="Episodes per evaluation")


class ContinuousTrainingConfig(BaseModel):
    """Continuous training advisor and model management configuration"""
    
    # Performance analysis
    performance_window: int = Field(50, description="Performance history window size")
    recommendation_frequency: int = Field(10, description="Episodes between recommendations")
    
    # Model management
    checkpoint_frequency: int = Field(25, description="Updates between checkpoints")
    
    # Evaluation settings (centralized)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Data difficulty adaptation
    adaptation_enabled: bool = Field(True, description="Enable adaptive data difficulty")


class AdaptiveDataConfig(BaseModel):
    """Adaptive data selection configuration"""
    
    # Static constraints (never change)
    symbols: List[str] = Field(default_factory=lambda: ["MLGO"], description="Trading symbols")
    date_range: List[Optional[str]] = Field(default_factory=lambda: [None, None], description="Date range [start, end]")
    
    # Adaptive ranges (ContinuousTraining can modify these)
    day_score_range: List[float] = Field(default_factory=lambda: [0.7, 1.0], description="Day quality score range")
    roc_range: List[float] = Field(default_factory=lambda: [0.05, 1.0], description="ROC score range")
    activity_range: List[float] = Field(default_factory=lambda: [0.0, 1.0], description="Activity score range")
    
    # SIMPLIFIED SELECTION MODES
    # DAY SELECTION: How to pick which trading day to use next
    day_selection_mode: Literal["sequential", "quality_weighted", "random"] = Field(
        "sequential", 
        description="Day selection: 'sequential' (chronological), 'quality_weighted' (by score), 'random'"
    )
    
    # RESET POINT SELECTION: How to cycle through reset points within each day  
    reset_point_selection_mode: Literal["sequential", "random"] = Field(
        "sequential", 
        description="Reset point selection: 'sequential' (by quality), 'random'"
    )
    
    # BACKWARD COMPATIBILITY: Keep old fields for migration
    selection_mode: Optional[Literal["sequential", "random", "quality_weighted"]] = Field(
        None, description="DEPRECATED: Use day_selection_mode instead"
    )
    randomize_order: Optional[bool] = Field(None, description="DEPRECATED: Removed for simplicity")


class DataCycleConfig(BaseModel):
    """Data cycle management - when to switch days/reset points"""
    
    # Day switching conditions (when ANY is met, switch to next day)
    day_max_episodes: Optional[int] = Field(None, description="Max episodes per day before switching to next day")
    day_max_updates: Optional[int] = Field(None, description="Max updates per day before switching to next day")  
    day_max_cycles: Optional[int] = Field(3, description="How many times to cycle through ALL reset points before switching to next day")


class ResetPointConfig(BaseModel):
    """Reset point management configuration"""
    selection_mode: Literal["sequential", "random"] = Field("sequential", description="How to order reset points: sequential (by quality) or random")


class DaySelectionConfig(BaseModel):
    """Day selection configuration"""
    selection_mode: Literal["sequential", "random", "quality_weighted", "curriculum_ordered"] = Field("quality_weighted", description="Selection mode")
    randomize_order: bool = Field(False, description="Randomize selection order")


class PreloadingConfig(BaseModel):
    """Data preloading configuration"""
    preload_enabled: bool = Field(True, description="Enable data preloading")


class DataLifecycleConfig(BaseModel):
    """Data lifecycle management configuration"""
    
    # Enable/disable data lifecycle management
    enabled: bool = Field(True, description="Enable data lifecycle management")
    
    # Data cycle management (when to switch days/reset points)
    cycles: DataCycleConfig = Field(default_factory=DataCycleConfig)
    
    # Adaptive data selection (replaces stages, includes all selection modes)
    adaptive_data: AdaptiveDataConfig = Field(default_factory=AdaptiveDataConfig)
    
    # Preloading
    preloading: PreloadingConfig = Field(default_factory=PreloadingConfig)
    
    # BACKWARD COMPATIBILITY: Keep old configs for migration
    reset_points: Optional[ResetPointConfig] = Field(None, description="DEPRECATED: Use adaptive_data instead")
    day_selection: Optional[DaySelectionConfig] = Field(None, description="DEPRECATED: Use adaptive_data instead")


class TrainingManagerConfig(BaseModel):
    """Training Manager - Central authority for training lifecycle"""
    
    # Core mode selection
    mode: Literal["sweep", "production"] = Field("production", description="Training mode")
    
    # Core configuration sections
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    continuous: ContinuousTrainingConfig = Field(default_factory=ContinuousTrainingConfig)
    data_lifecycle: DataLifecycleConfig = Field(default_factory=DataLifecycleConfig)


# =============================================================================
# SCANNER CONFIGURATION
# =============================================================================


class ScannerConfig(BaseModel):
    """Consolidated scanner configuration"""

    # Momentum scanning
    min_daily_move: float = Field(0.10, description="Min 10% intraday movement")
    min_volume_multiplier: float = Field(2.0, description="Min 2x average volume")
    max_daily_move: Optional[float] = Field(
        None, description="Max daily move (uncapped)"
    )
    max_volume_multiplier: Optional[float] = Field(
        None, description="Max volume (uncapped)"
    )

    # Momentum scoring
    roc_lookback_minutes: int = Field(5, description="ROC calculation window")
    activity_lookback_minutes: int = Field(
        10, description="Activity calculation window"
    )
    min_reset_points: int = Field(60, description="Min reset points per day")
    roc_weight: float = Field(0.6, description="ROC score weight")
    activity_weight: float = Field(0.4, description="Activity score weight")

    # Session volume calculations
    premarket_start: str = Field("04:00", description="Pre-market start")
    premarket_end: str = Field("09:30", description="Pre-market end")
    regular_start: str = Field("09:30", description="Regular market start")
    regular_end: str = Field("16:00", description="Regular market end")
    postmarket_start: str = Field("16:00", description="Post-market start")
    postmarket_end: str = Field("20:00", description="Post-market end")
    volume_window_days: int = Field(10, description="Volume baseline window")
    use_session_specific_baselines: bool = Field(
        True, description="Session-specific baselines"
    )


# =============================================================================
# LOGGING AND MONITORING
# =============================================================================


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    console_enabled: bool = Field(True, description="Console logging")
    console_format: str = Field("simple", description="Console format")
    file_enabled: bool = Field(True, description="File logging")
    log_dir: str = Field("logs", description="Log directory")
    log_interval: int = Field(10, description="Metric log interval")

    # Component logging
    log_rewards: bool = Field(True, description="Log rewards")
    log_actions: bool = Field(True, description="Log actions")
    log_features: bool = Field(False, description="Log features")
    log_portfolio: bool = Field(True, description="Log portfolio")


class WandbConfig(BaseModel):
    """Weights & Biases configuration"""

    enabled: bool = Field(True, description="Enable W&B")
    project: str = Field("fx-ai-momentum", description="Project name")
    entity: Optional[str] = Field(None, description="W&B entity")
    name: Optional[str] = Field(None, description="Run name")
    tags: List[str] = Field(default_factory=list, description="Tags")
    notes: Optional[str] = Field(None, description="Run notes")

    log_frequency: Dict[str, int] = Field(
        default_factory=lambda: {
            "training": 1,
            "episode": 1,
            "rollout": 1,
            "evaluation": 1,
        }
    )

    save_code: bool = Field(True, description="Save code")
    save_model: bool = Field(True, description="Save model")


class DashboardConfig(BaseModel):
    """Live dashboard configuration"""

    enabled: bool = Field(True, description="Enable dashboard")
    port: int = Field(8051, description="Dashboard port")
    update_interval: float = Field(1.0, description="Update interval")
    max_episodes_shown: int = Field(20, description="Max episodes displayed")
    max_trades_shown: int = Field(100, description="Max trades displayed")
    chart_height: int = Field(400, description="Chart height")

    heatmap_features: List[str] = Field(
        default_factory=lambda: [
            "price_velocity",
            "tape_imbalance",
            "spread_compression",
            "1m_ema9_distance",
            "1m_swing_high_distance",
        ]
    )


# =============================================================================
# DAY SELECTION CONFIGURATION
# =============================================================================


class DaySelectionConfig(BaseModel):
    """Day selection configuration for training"""

    episodes_per_day: int = Field(5, description="Episodes per trading day")
    reset_point_quality_range: List[float] = Field(
        [0.5, 1.0], description="Reset point quality range"
    )
    day_switching_strategy: str = Field(
        "quality_based", description="Day switching strategy"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================


class Config(BaseModel):
    """Main configuration container"""

    model_config = ConfigDict(extra="forbid")

    # Core components
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # Scanner configuration
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    # DEPRECATED: Day selection (now in env.training_manager.data_lifecycle.adaptive_data)
    day_selection: Optional[DaySelectionConfig] = Field(None, description="DEPRECATED: Use env.training_manager.data_lifecycle.adaptive_data instead")

    # Monitoring
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    # Runtime settings
    experiment_name: str = Field("momentum_training", description="Experiment name")
    mode: Literal["train", "eval", "backtest"] = Field(
        "train", description="Execution mode"
    )

    @classmethod
    def load(cls, overrides_path: Optional[str] = None) -> "Config":
        """Load config with optional YAML overrides"""
        config = cls()

        if overrides_path:
            import yaml

            with open(overrides_path) as f:
                overrides = yaml.safe_load(f)
            config = config.model_copy(update=overrides, deep=True)

        return config

    def save_used_config(self, path: str):
        """Save the actual config used for reproducibility"""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
