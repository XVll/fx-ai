# config/config.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Dict
from hydra.core.config_store import ConfigStore




@dataclass
class RewardConfig:
    weight_equity_change: float = 1.0  # Primary weight for PnL
    weight_realized_pnl: float = 0.1  # Small bonus for realized PnL
    penalty_transaction_fill: float = 0.005  # Small penalty per fill to discourage excessive noise trades
    penalty_holding_inaction: float = 0.0001  # Tiny penalty for doing nothing while holding a position
    penalty_drawdown_step: float = 0.2  # If equity drops in a step, penalize 20% of that drop amount
    penalty_invalid_action: float = 0.05  # Penalty if the environment flags the action as invalid
    terminal_penalty_bankruptcy: float = 20.0  # Large penalty normalized to potential reward scale
    terminal_penalty_max_loss: float = 10.0  # Penalty normalized to potential reward scale
    reward_scaling_factor: float = 0.01  # Example: if PnL is in dollars, scale down for typical RL reward ranges
    log_reward_components: bool = True  # For debugging


@dataclass
class EnvConfig:
    training_mode: str= "backtesting"  # Training mode (training, validation, testing)
    render_interval: int = 10  # Interval for rendering (in steps)
    max_episode_loss_percent: float = 0.5  # Maximum loss percentage per episode
    bankruptcy_threshold_factor: float = 0.5  # Bankruptcy threshold as a factor of initial capital
    max_invalid_actions_per_episode: int = 20  # Maximum invalid actions allowed per episode

    state_dimension: int = 1000  # State dimensions, size of feature vectors
    max_steps: int = 500  # Maximum number of steps in an episode
    normalize_state: bool = True  # Should feature vectors be normalized?
    random_reset: bool = True  # Should the environment be reset randomly?
    max_position: float = 1.0  # Maximum position size
    render_mode: str = "human"  # Rendering mode (none, human, rgb_array)
    position_multipliers: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    reward: RewardConfig = field(default_factory=RewardConfig)

@dataclass
class ModelConfig:
    # Feature config
    hf_seq_len: int = 60
    hf_feat_dim: int = 20
    mf_seq_len: int = 30
    mf_feat_dim: int = 20
    lf_seq_len: int = 30
    lf_feat_dim: int = 20
    static_feat_dim: int = 5
    portfolio_feat_dim: int = 5
    portfolio_seq_len: int = 5

    # Model dimensions
    d_model: int = 64
    d_fused: int = 256

    # Transformer parameters
    hf_layers: int = 2
    mf_layers: int = 2
    lf_layers: int = 2
    portfolio_layers: int = 2
    hf_heads: int = 4
    mf_heads: int = 4
    lf_heads: int = 4
    portfolio_heads: int = 4

    # Output parameters
    action_dim = [5,4]
    continuous_action: bool = False

    # Other parameters
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    critic_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    n_episodes_per_update: int = 8
    total_updates: int = 100
    device: str = "auto"
    best_model_path: str = "best_model"
    best_model_metrics: str = "reward"

    enabled: bool = False
    load_best_model: bool = False
    best_models_dir: str = "./best_models"
    checkpoint_sync_frequency: int = 2
    startup_evaluation: bool = False
    max_best_models: int = 5


@dataclass
class LogFrequencyConfig:
    steps: int = 1
    gradients: int = 100
    model_checkpoints: int = 5
    custom_charts: int = 5


@dataclass
class WandbConfig:
    enabled: bool = True
    project_name: str = "fx-ai"
    entity: Optional[str] = None
    log_code: bool = True
    log_model: bool = True
    log_frequency: LogFrequencyConfig = field(default_factory=LogFrequencyConfig)
    watch_model: bool = True
    log_batch_metrics: bool = False
    visualizations: Dict[str, bool] = field(default_factory=lambda: {
        "position_charts": True,
        "trade_analysis": True,
        "market_indicators": True,
        "action_distributions": True
    })


@dataclass
class DataConfig:
    provider_type: str = "file"
    data_dir: str = "./dnb/mlgo"
    symbol_info_file: str = "symbols.json"
    symbol: str = "MLGO"
    start_date: str = "2025-03-27"
    end_date: str = "2025-03-27"
    timeframes: List[str] = field(default_factory=lambda: ["1s", "1m", "5m", "1d"])
    data_types: List[str] = field(
        default_factory=lambda: ["trades", "quotes", "bars_1s", "bars_1m", "bars_5m", "bars_1d", "status"])
    use_cache: bool = True
    verbose: bool = False
    dbn_cache_size: int = 32

    # API settings
    api: Dict[str, Any] = field(default_factory=dict)
    # Live settings
    live: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class CallbacksConfig:
    save_freq: int = 5
    log_freq: int = 10
    save_best_only: bool = False  # If true, only save ai that improve mean reward
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class MarketConfig:
    slippage_factor: float = 0.001


@dataclass
class ExecutionConfig:
    allow_shorting: bool = False  # Allow shorting positions
    mean_latency_ms: float = 250.0  # Average latency in milliseconds
    latency_std_dev_ms: float = 10.0  # Standard deviation of latency in milliseconds ????????

    base_slippage_bps: float = 1.5  # Base slippage in basis points
    size_impact_slippage_bps_per_unit: float = 0.03  # Slippage per unit of size in basis points
    max_total_slippage_bps: float = 50.0  # Maximum total slippage in basis points

    commission_per_share: float = 0.0007  # Commission per share in dollars
    fee_per_share: float = 0.003  # Fee per share in dollars
    min_commission_per_order: Optional[float] = 0.1  # Minimum commission per order in dollars
    max_commission_pct_of_value: Optional[float] = None  # Maximum commission as a percentage of trade value


@dataclass
class PortfolioConfig:
    initial_cash: float = 100000.0
    max_position_value_ratio: float = 2.0  # Fixed: removed the comma - Example: Up to 2x leverage on equity
    max_position_holding_seconds: int = 300


@dataclass
class SimulationConfig:
    market_config: MarketConfig = field(default_factory=MarketConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)


@dataclass
class Config:
    # Main config sections
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # General settings
    quick_test: bool = False
    project_name: str = "Fx AI"


# Register all schemas with Hydra's ConfigStore - Updated

cs = ConfigStore.instance()

# Register all schemas with their correct paths
cs.store(name="model/transformer", node=ModelConfig)
cs.store(name="env/trading", node=EnvConfig)
cs.store(name="training/ppo", node=TrainingConfig)
cs.store(name="data/databento", node=DataConfig)
cs.store(name="wandb/default", node=WandbConfig)
cs.store(name="simulation/default", node=SimulationConfig)
cs.store(name="callbacks/default", node=CallbacksConfig)  # Added callbacks registration

# Export the Config class and all config components for direct import
__all__ = ['Config', 'EnvConfig', 'ModelConfig', 'TrainingConfig', 'DataConfig',
           'WandbConfig', 'SimulationConfig', 'CallbacksConfig', 'RewardConfig', 'ExecutionConfig', 'MarketConfig', 'PortfolioConfig']