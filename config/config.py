# config/config.py
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
from hydra.core.config_store import ConfigStore


@dataclass
class RewardConfig:
    type: str = "momentum"
    scaling: float = 2.0
    trade_penalty: float = 0.1
    hold_penalty: float = 0.0
    early_exit_bonus: float = 0.5
    flush_prediction_bonus: float = 2.0

    momentum_threshold: float = 0.5
    volume_surge_threshold: float = 5.0
    tape_speed_threshold: float = 3.0
    tape_imbalance_threshold: float = 0.7


@dataclass
class EnvConfig:
    state_dimension: int = 1000  # State dimensions, size of feature vectors
    max_steps: int = 500  # Maximum number of steps in an episode
    normalize_state: bool = True  # Should feature vectors be normalized?
    random_reset: bool = True  # Should the environment be reset randomly?
    max_position: float = 1.0  # Maximum position size
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class ModelConfig:
    # Feature dimensions
    hf_seq_len: int = 60
    hf_feat_dim: int = 20
    mf_seq_len: int = 30
    mf_feat_dim: int = 15
    lf_seq_len: int = 30
    lf_feat_dim: int = 10
    static_feat_dim: int = 15

    # Model dimensions
    d_model: int = 64
    d_fused: int = 256

    # Transformer parameters
    hf_layers: int = 2
    mf_layers: int = 2
    lf_layers: int = 2
    hf_heads: int = 4
    mf_heads: int = 4
    lf_heads: int = 4

    # Output parameters
    action_dim: int = 1
    continuous_action: bool = True

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
    data_dir: str = "./data"
    symbol_info_file: Optional[str] = None
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
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class FeatureConfig:
    hf_seq_len: int = 60
    hf_feat_dim: int = 20
    mf_seq_len: int = 30
    mf_feat_dim: int = 15
    lf_seq_len: int = 30
    lf_feat_dim: int = 10
    static_feat_dim: int = 15
    use_volume_profile: bool = True
    use_tape_features: bool = True
    use_level2_features: bool = True


@dataclass
class MarketConfig:
    slippage_factor: float = 0.001


@dataclass
class ExecutionConfig:
    commission_per_share: float = 0.0


@dataclass
class PortfolioConfig:
    initial_cash: float = 100000.0


@dataclass
class SimulationConfig:
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
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


# Register schemas with Hydra's ConfigStore
cs = ConfigStore.instance()

# Register all schemas
cs.store(name="model/transformer", node=ModelConfig)
cs.store(name="env/trading", node=EnvConfig)
cs.store(name="training/ppo", node=TrainingConfig)
cs.store(name="data/databento", node=DataConfig)
cs.store(name="wandb/default", node=WandbConfig)

# Export the Config class and all config components for direct import
__all__ = ['Config', 'EnvConfig', 'ModelConfig', 'TrainingConfig', 'DataConfig',
           'WandbConfig', 'SimulationConfig', 'RewardConfig']
