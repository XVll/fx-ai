from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

from v2.config.callbacks import CallbackConfig
from v2.config.core import ActionType, SessionType
from v2.config.model import ModelConfig
from v2.config.training import TrainingConfig, TrainingManagerConfig
from v2.config.data import DataConfig, DataLifecycleConfig
from v2.config.rewards import RewardConfig
from v2.config.simulation import SimulationConfig
from v2.config.environment import EnvironmentConfig
from v2.config.scanner import ScannerConfig
from v2.config.logging import LoggingConfig, WandbConfig
from v2.config.captum import CaptumConfig
from v2.config.optuna.optuna_config import StudyConfig, OptunaStudySpec

class Config(BaseModel):
    """Main configuration container"""
    output_dir: str = Field("outputs", description="Base output directory for results")

    # Core components
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # Scanner configuration
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    # Monitoring
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    callbacks: CallbackConfig = Field(default_factory=CallbackConfig)

    # Feature attribution analysis
    captum: Optional[CaptumConfig] = Field(None, description="Captum feature attribution config")
    
    # Hyperparameter optimization
    optuna: Optional[StudyConfig] = Field(None, description="Optuna hyperparameter optimization config")

    @classmethod
    def load(cls, override_file: Optional[str] = None, spec_file: Optional[str] = None) -> "Config":
        """Load config with optional YAML overrides
        
        Args:
            override_file: Path to override YAML file or name in config/overrides/
            spec_file: Optuna spec file name in config/optuna/overrides/
            
        Returns:
            Validated Config object
        """
        import yaml
        import logging
        from pathlib import Path

        logger = logging.getLogger(__name__)
        
        # Start with defaults
        config = cls()

        # Apply overrides if specified
        if override_file:
            override_path = Path(override_file)
            
            # Check in multiple locations
            if not override_path.is_file():
                # Try in v2/config/overrides directory
                override_path = Path("v2/config/overrides") / override_file
                if not override_path.is_file():
                    # Try with .yaml extension
                    override_path = Path("v2/config/overrides") / f"{override_file}.yaml"
            
            if override_path.is_file():
                logger.info(f"Loading config overrides from: {override_path}")
                with open(override_path) as f:
                    overrides = yaml.safe_load(f)
                
                # Validate and apply overrides using Pydantic's model_copy
                try:
                    config = config.model_copy(update=overrides, deep=True)
                    logger.info("Config overrides applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply config overrides: {e}")
                    raise ValueError(f"Invalid config overrides: {e}")
            else:
                raise FileNotFoundError(f"Config override file not found: {override_file}")
        
        # Apply optuna spec if specified
        if spec_file:

            spec_path = Path(f"v2/config/optuna/overrides/{spec_file}")
            if not spec_path.is_file():
                spec_path = Path(f"v2/config/optuna/overrides/{spec_file}.yaml")
            
            if spec_path.is_file():
                logger.info(f"Loading optuna spec from: {spec_path}")
                with open(spec_path) as f:
                    spec_data = yaml.safe_load(f)
                
                # Create OptunaStudySpec and get first study
                try:
                    optuna_spec = OptunaStudySpec(**spec_data)
                    study_config = optuna_spec.studies[0]  # Use first study
                    
                    # Convert to dict and apply to config.optuna
                    config.optuna = study_config
                    logger.info("Optuna spec applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply optuna spec: {e}")
                    raise ValueError(f"Invalid optuna spec: {e}")
            else:
                raise FileNotFoundError(f"Optuna spec file not found: {spec_file}")
        
        # Save the complete config for reproducibility
        config._save_used_config()
        
        return config
    
    def _save_used_config(self):
        """Save the complete config used for this run"""
        import yaml
        from pathlib import Path
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Create output directory
        output_dir = Path("outputs/configs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_{self.experiment_name}_{timestamp}.yaml"
        filepath = output_dir / filename
        
        # Save config using Pydantic's model_dump
        with open(filepath, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved used config to: {filepath}")

    def save_used_config(self, path: str):
        """Save the actual config used for reproducibility"""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)