import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from data.data_manager import DataManager
from data.scanner.momentum_scanner import MomentumScanner
from data import DatabentoFileProvider
from envs.trading_environment import TradingEnvironment
from agent.ppo_agent import PPOTrainer
from model.transformer import MultiBranchTransformer
from callbacks import create_callbacks_from_config, CallbackManager
from core.logger import setup_rich_logging
from core.model_manager import ModelManager
from core.shutdown import (
    ShutdownReason,
    graceful_shutdown_context, get_global_shutdown_manager
)
from config.config import Config, register_configs
from training.training_manager import TrainingManager, TrainingMode

logger = logging.getLogger(__name__)


class ApplicationBootstrap:
    """Bootstrap the FxAI application with proper dependency injection and graceful shutdown."""

    def __init__(self, cfg: DictConfig):
        # Hydra automatically manages output directory
        self.output_path: Path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.config: Config = hydra.utils.instantiate(cfg, _recursive_=False)
        self.logger = logging.getLogger(f"{__name__}.Application")
        self.shutdown_manager = get_global_shutdown_manager()

        # Component instances
        self.training_manager: Optional[TrainingManager] = None
        self.trainer: Optional[PPOTrainer] = None
        self.environment: Optional[TradingEnvironment] = None
        self.data_manager: Optional[DataManager] = None
        self.device: Optional[torch.device] = None
        self.callback_manager: Optional[CallbackManager] = None

    def initialize(self) -> None:
        """Initialize the application with Hydra configuration."""

        self._setup_logging()
        self._initialize_components()

        logger.info("=" * 80)
        logger.info(f"[bold green] Hydra output directory:[/bold green] {self.output_path}")
        logger.info(f"[bold green] Configuration loaded successfully[/bold green]")
        logger.info("🚀 FxAI Application bootstrapped successfully")
        logger.info("=" * 80)

    # Hydra handles directory setup automatically

    def _setup_logging(self) -> None:
        """Setup application logging with rich formatting."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        setup_rich_logging(
            level=log_level,
            show_time=self.config.logging.show_time,
            show_path=self.config.logging.show_path,
            compact_errors=True
        )

    def _create_training_manager(self) -> TrainingManager:
        """Create and configure a training manager with config."""
        
        # Create a model manager
        model_manager = ModelManager()

        training_manager = TrainingManager(
            config=self.config.training.training_manager,
            model_manager=model_manager,
            device=self.device,
            output_path=self.output_path,
            run_id=self.output_path.name  # Use Hydra's directory name as run_id
        )
        
        # Set data manager reference for integrated data lifecycle
        training_manager.data_manager = self.data_manager
        training_manager.callback_manager = self.callback_manager

        return training_manager

    def _initialize_components(self) -> None:
        """Initialize application components"""
        
        self.logger.info("🔧 Initializing application components")

        self.device = self._create_device()

        self.callback_manager = create_callbacks_from_config(
            config=self.config.callbacks,
            trainer=None,  # Will be set after creation
            environment=None,  # Will be set after creation
            output_path=self.output_path,
            shutdown_manager=self.shutdown_manager
        )

        self.data_manager = self._create_data_manager()
        self.environment = self._create_environment()
        self.trainer = self._create_trainer()
        self.training_manager = self._create_training_manager()

        self.callback_manager.register_trainer(trainer=self.trainer)
        self.callback_manager.register_environment(environment=self.environment)

        # Register components for shutdown
        self.training_manager.register_shutdown(self.shutdown_manager)

        self.logger.info("✅ All components initialized successfully")

    def _create_device(self) -> torch.device:
        """Create and configure a PyTorch device."""
        np.random.seed(self.config.training.seed)
        torch.manual_seed(self.config.training.seed)

        if self.config.training.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training.seed)
            device = torch.device("cuda")
        elif self.config.training.device == "mps" and torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config.training.seed)
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.logger.info(f"🔧 Using device: {device}")
        return device

    def _create_data_manager(self) -> DataManager:
        """Create a data manager from configuration."""
        self.logger.info("📊 Creating data manager")

        # Create data provider
        if self.config.data.provider == "databento":
            data_provider = DatabentoFileProvider(self.config.data, self.config.training.training_manager.symbols)
        else:
            raise ValueError(f"Unsupported data provider: {self.config.data.provider}")

        momentum_scanner = MomentumScanner(config=self.config.scanner)
        data_manager = DataManager(provider=data_provider, momentum_scanner=momentum_scanner, config=self.config.data)

        self.logger.info("✅ Data manager created")
        return data_manager

    def _create_environment(self) -> TradingEnvironment:
        """Create a trading environment from configuration."""
        self.logger.info("🏢 Creating trading environment")

        # Create a trading environment
        environment = TradingEnvironment(
            config=self.config,
            data_manager=self.data_manager,
            callback_manager=self.callback_manager,
        )

        self.logger.info("✅ Trading environment created")
        return environment

    def _create_trainer(self) -> PPOTrainer:
        """Create a trainer / agent from configuration."""
        self.logger.info("🤖 Creating trainer")

        # Create trainer with callback manager
        trainer = PPOTrainer(
            config=self.config.training,
            model=self._create_model(),
            device=self.device,
        )

        self.logger.info("✅ Trainer created")
        return trainer

    def _create_model(self) -> MultiBranchTransformer:
        """Create a model instance."""
        model = MultiBranchTransformer(model_config=self.config.model, device=self.device).to(self.device)
        
        return model


# Hydra handles argument parsing automatically


def execute_training(training_manager: TrainingManager, config: Config, app: 'ApplicationBootstrap') -> None:
    """Execute training based on configuration."""
    mode_type = TrainingMode(config.training.training_manager.mode)
    logger.info(f"🎯 Starting {mode_type.value} based on configuration")

    training_manager.start(
        trainer=app.trainer,
        environment=app.environment,
        data_manager=app.data_manager,
        callback_manager=app.callback_manager,
    )

    logger.info(f"✅ Training completed successfully")



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for the FxAI trading system using Hydra.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        int: Exit code (0 for success, 130 for graceful shutdown, 1 for error)
    """
    with graceful_shutdown_context() as shutdown_manager:
        try:
            # Create application with Hydra config
            app = ApplicationBootstrap(cfg)

            # Initialize application
            app.initialize()

            logger.info("🚀 Starting training with graceful shutdown support")

            # Execute training with bootstrap components
            execute_training(
                app.training_manager,
                app.config,
                app
            )

            # Check if shutdown was requested during training
            if shutdown_manager.is_shutdown_requested():
                logger.info("🛑 Training completed due to shutdown request")
                return 130  # Standard exit code for SIGINT

            return 0

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            shutdown_manager.initiate_shutdown(ShutdownReason.ERROR_CONDITION)
            return 1


if __name__ == "__main__":
    # Register structured configs with Hydra
    register_configs()
    
    # Hydra will handle the rest
    exit_code = main()
    sys.exit(exit_code)
