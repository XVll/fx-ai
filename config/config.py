"""
Hydra structured configuration schema for FxAI trading system.

This replaces the Pydantic-based config system with Hydra structured configs,
providing both type safety and powerful configuration composition.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from hydra.core.config_store import ConfigStore

# Import all the individual config schemas
from .model.model_config import ModelConfig
from .model.model_storage_config import ModelStorageConfig
from .training.training_config import TrainingConfig, TrainingManagerConfig
from .environment.environment_config import EnvironmentConfig
from .data.data_config import DataConfig
from .simulation.simulation_config import SimulationConfig
from .scanner.scanner_config import ScannerConfig
from .logging.logging_config import LoggingConfig
from .callbacks.callback_config import CallbackConfig
from .attribution.attribution_config import AttributionConfig
from .captum.captum_config import CaptumConfig
from .optuna.optuna_config import StudyConfig
from .rewards.reward_config import RewardConfig
from .evaluation.evaluation_config import EvaluationConfig


@dataclass
class Config:
    """Main configuration container for FxAI trading system.
    
    Hydra automatically handles output directory management.
    """
    
    # Core components
    model: ModelConfig = field(default_factory=ModelConfig)
    model_storage: ModelStorageConfig = field(default_factory=ModelStorageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Scanner configuration
    scanner: ScannerConfig = field(default_factory=ScannerConfig)

    # Monitoring
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    
    # Reward system
    rewards: RewardConfig = field(default_factory=RewardConfig)
    
    # Evaluation system
    evaluation: Optional[EvaluationConfig] = None    # Model evaluation config

    # Feature attribution analysis
    attribution: Optional[AttributionConfig] = None  # Feature attribution analysis config
    captum: Optional[CaptumConfig] = None            # Legacy Captum config (deprecated)

    # Hyperparameter optimization
    optuna: Optional[StudyConfig] = None             # Optuna hyperparameter optimization config

    # Hydra composition defaults
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        "model: transformer",
        "model_storage: default",
        "training: default", 
        "env: trading",
        "data: databento",
        "simulation: default",
        "scanner: momentum",
        "logging: default",
        "callbacks: default",
        "rewards: default"
    ])


def register_configs():
    """Register all configurations with Hydra's ConfigStore.
    
    This enables type safety, autocompletion, and validation.
    """
    cs = ConfigStore.instance()
    
    # Register main config
    cs.store(name="config", node=Config)
    
    # Register component configs for composition
    cs.store(group="model", name="transformer", node=ModelConfig)
    cs.store(group="model_storage", name="default", node=ModelStorageConfig)
    cs.store(group="training", name="default", node=TrainingConfig)
    cs.store(group="env", name="trading", node=EnvironmentConfig)
    cs.store(group="data", name="databento", node=DataConfig)
    cs.store(group="simulation", name="default", node=SimulationConfig)
    cs.store(group="scanner", name="momentum", node=ScannerConfig)
    cs.store(group="logging", name="default", node=LoggingConfig)
    cs.store(group="callbacks", name="default", node=CallbackConfig)
    cs.store(group="attribution", name="default", node=AttributionConfig)
    cs.store(group="captum", name="default", node=CaptumConfig)
    cs.store(group="optuna", name="default", node=StudyConfig)
    cs.store(group="rewards", name="default", node=RewardConfig)
    cs.store(group="evaluation", name="default", node=EvaluationConfig)