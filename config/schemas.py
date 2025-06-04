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

    # Learning rate scheduling
    use_lr_annealing: bool = Field(True, description="Enable LR annealing")
    lr_annealing_factor: float = Field(0.7, description="LR decay factor")
    lr_annealing_patience: int = Field(50, description="Patience for LR decay")
    min_learning_rate: float = Field(1e-6, description="Minimum learning rate")

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
    max_episode_steps: int = Field(256, gt=0, description="Maximum steps per episode")
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

    # Curriculum learning
    curriculum: "CurriculumConfig" = Field(default_factory=lambda: CurriculumConfig())


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
# CURRICULUM CONFIGURATION
# =============================================================================


class CurriculumStageConfig(BaseModel):
    """Single curriculum stage configuration"""

    enabled: bool = Field(True, description="Stage enabled")
    symbols: List[str] = Field(default_factory=list, description="Training symbols")
    date_range: List[Optional[str]] = Field([None, None], description="Date range")
    day_score_range: List[float] = Field([0.0, 1.0], description="Day quality range")
    roc_range: List[float] = Field([0.0, 1.0], description="ROC score range")
    activity_range: List[float] = Field([0.0, 1.0], description="Activity score range")

    # Stage transition conditions
    max_updates: Optional[int] = Field(
        None, description="Max updates before transition"
    )
    max_episodes: Optional[int] = Field(
        None, description="Max episodes before transition"
    )
    max_cycles: Optional[int] = Field(None, description="Max cycles before transition")

    @field_validator("roc_range", mode="before")  # noinspection PyNestedDecorators
    @classmethod
    def validate_roc_range(cls, v: Any) -> Any:
        if (
            len(v) != 2
            or v[0] >= v[1]
            or not (-1 <= v[0] <= 1)
            or not (-1 <= v[1] <= 1)
        ):
            raise ValueError("ROC range must be [min, max] with -1 <= min < max <= 1")
        return v

    @field_validator(
        "activity_range", "day_score_range", mode="before"
    )  # noinspection PyNestedDecorators
    @classmethod
    def validate_other_ranges(cls, v: Any) -> Any:
        if len(v) != 2 or v[0] >= v[1] or not (0 <= v[0] <= 1) or not (0 <= v[1] <= 1):
            raise ValueError("Range must be [min, max] with 0 <= min < max <= 1")
        return v


class CurriculumConfig(BaseModel):
    """Curriculum learning configuration"""

    stage_1: CurriculumStageConfig = Field(
        default_factory=lambda: CurriculumStageConfig(
            day_score_range=[0.7, 1.0], roc_range=[0.05, 1.0], max_updates=500
        )
    )

    stage_2: CurriculumStageConfig = Field(
        default_factory=lambda: CurriculumStageConfig(
            day_score_range=[0.5, 0.9],
            roc_range=[0.6, 1.0],
            activity_range=[0.3, 1.0],
            max_updates=800,
        )
    )

    stage_3: CurriculumStageConfig = Field(
        default_factory=lambda: CurriculumStageConfig(
            day_score_range=[0.3, 0.7], roc_range=[0.4, 1.0], activity_range=[0.2, 1.0]
        )
    )


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

    # Day selection
    day_selection: DaySelectionConfig = Field(default_factory=DaySelectionConfig)

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
