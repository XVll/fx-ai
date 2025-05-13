from dataclasses import dataclass, field, MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class RewardConfig:
    type: str = MISSING
    scaling: float = MISSING
    trade_penalty: float = MISSING
    hold_penalty: float = MISSING
    early_exit_bonus: float = MISSING
    flush_prediction_bonus: float = MISSING

    momentum_threshold: float = MISSING
    volume_surge_threshold: float = MISSING
    tape_speed_threshold: float = MISSING
    tape_imbalance_threshold: float = MISSING


@dataclass
class EnvConfig:
    state_dim: int = MISSING
    max_steps: int = MISSING
    normalize_state: bool = MISSING
    random_reset: bool = MISSING
    max_position: float = MISSING
    reward: RewardConfig = MISSING


@dataclass
class Config:  # This is your main configuration schema
    env: EnvConfig
    project_name: str = "Fx AI"


cs = ConfigStore.instance()

cs.store(name="base_config", node=Config)
cs.store(group="env", name="base_trading", node=EnvConfig)  # Schema name
