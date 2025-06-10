"""
V2 FxAI Main Entry Point - TDD Implementation (Config-Driven)

This is the new entry point for the FxAI trading system using TDD approach.
Features purely config-driven training with minimal CLI arguments.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import numpy as np
import torch

from data.data_manager import DataManager
from data import DatabentoFileProvider
from agent.ppo_agent import PPOTrainer
from callbacks import create_callbacks_from_config, CallbackManager
from core.logger import setup_rich_logging
from core.shutdown import (
    ShutdownReason,
    graceful_shutdown_context, get_global_shutdown_manager
)
from config.config import Config
from envs import TradingEnvironment
from training.training_manager import TrainingManager, TrainingMode

logger = logging.getLogger(__name__)


class ApplicationBootstrap():
    """Bootstrap the FxAI application with proper dependency injection and graceful shutdown."""

    def __init__(self):
        self.output_path: Optional[Path] = None
        self.config: Optional[Config] = None
        self.logger = logging.getLogger(f"{__name__}.Application")
        self.shutdown_manager = get_global_shutdown_manager()

        # Component instances
        self.training_manager: Optional[TrainingManager] = None
        self.trainer: Optional[PPOTrainer] = None
        self.environment: Optional[TradingEnvironment] = None
        self.data_manager: Optional[DataManager] = None
        self.device: Optional[torch.device] = None
        self.callback_manager: Optional[CallbackManager] = None

    def initialize(self, args: Optional[argparse.Namespace]) -> None:
        """Initialize application with configuration."""

        self._setup_config(args.config, args.spec)
        self._setup_logging()
        self._initialize_components()
        self.training_manager = self._create_training_manager()

        self._setup_directories()
        self.config.save_used_config(str(self.output_path / "config.yaml"))

        logger.info("=" * 80)
        logger.info(f"[bold green] Loaded configuration:[/bold green] {self.config or 'defaults'}")
        logger.info(f"[bold green] Output directory created:[/bold green] {self.output_path}")
        logger.info("🚀 FxAI Application bootstrapped successfully")
        logger.info("=" * 80)

    def _setup_directories(self) -> None:
        """Setup necessary directories for outputs and logs."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir

    def _setup_config(self, config: Optional[str], spec: Optional[str]) -> None:
        """Setup application configuration from v2 config."""
        self.config = Config.load(config, spec)

    def _setup_logging(self) -> None:
        """Setup application logging using v2 rich logging system."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        setup_rich_logging(
            level=log_level,
            show_time=self.config.logging.show_time,
            show_path=self.config.logging.show_path,
            compact_errors=True
        )

    def _create_training_manager(self) -> TrainingManager:
        """Create and configure training manager with config."""
        
        # Create model manager
        from core.model_manager import ModelManager
        model_manager = ModelManager()

        training_manager = TrainingManager(
            config=self.config.training.training_manager,
            model_manager=model_manager
        )
        
        # Set data manager reference for integrated data lifecycle
        training_manager.data_manager = self.data_manager
        training_manager.callback_manager = self.callback_manager

        return training_manager

    def _initialize_components(self) -> None:
        """Initialize application components.
        
        This method will create data managers, environments, etc.
        """
        self.logger.info("🔧 Initializing application components")

        # Create device first
        self.device = self._create_device()

        # Create callback manager first (before other components)
        self.callback_manager = create_callbacks_from_config(
            config=self.config.callbacks,
            trainer=None,  # Will be set after creation
            environment=None,  # Will be set after creation
            output_path=self.output_path,
        )

        # Create core components
        self.data_manager = self._create_data_manager()
        self.environment = self._create_environment()
        self.trainer = self._create_trainer()

        # Register components with callback manager after creation
        self.callback_manager.register_trainer(self.trainer)
        self.callback_manager.register_environment(self.environment)

        # Register callbacks individually with shutdown system for parallel shutdown
        for callback in self.callback_manager.get_callbacks():
            self.shutdown_manager.register_component(
                component=callback,
                timeout=callback.get_shutdown_timeout(),
                name=f"callback_{callback.name}"
            )

        self.shutdown_manager.register_component(
            component=self.data_manager,
            timeout=self.config.data.shutdown_timeout
        )
        self.shutdown_manager.register_component(
            component=self.environment,
            timeout=self.config.env.shutdown_timeout
        )
        self.shutdown_manager.register_component(
            component=self.trainer,
            timeout=self.config.training.shutdown_timeout
        )
        self.shutdown_manager.register_component(
            component=self.training_manager,
            timeout=self.config.training.shutdown_timeout
        )

        self.logger.info("✅ All components initialized successfully")

    def _create_device(self) -> torch.device:
        """Create and configure PyTorch device."""
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

    def _create_data_manager(self) -> Any:
        """Create data manager from configuration."""
        self.logger.info("📊 Creating data manager")

        # Import v1 data manager implementation (temporary)
        from data.data_manager import DataManager
        from data.scanner.momentum_scanner import MomentumScanner

        # Create data provider
        if self.config.data.provider == "databento":
            data_provider = DatabentoFileProvider(self.config.data, self.config.training.training_manager.symbols)
        else:
            raise ValueError(f"Unsupported data provider: {self.config.data.provider}")

        momentum_scanner = MomentumScanner(config=self.config.scanner)
        data_manager = DataManager(provider=data_provider, momentum_scanner=momentum_scanner, config=self.config.data, logger=self.logger)

        self.logger.info("✅ Data manager created")
        return data_manager

    def _create_environment(self) -> Any:
        """Create trading environment from configuration."""
        self.logger.info("🏢 Creating trading environment")

        # Import v2 environment implementation
        from envs.trading_environment import TradingEnvironment

        # Create trading environment
        environment = TradingEnvironment(
            config=self.config,
            data_manager=self.data_manager,
            callback_manager=self.callback_manager,
        )

        self.logger.info("✅ Trading environment created")
        return environment

    def _create_trainer(self) -> Any:
        """Create trainer/agent from configuration."""
        self.logger.info("🤖 Creating trainer")

        # Import v2 trainer implementation
        from agent.ppo_agent import PPOTrainer

        # Create trainer with callback manager
        trainer = PPOTrainer(
            config=self.config.training,
            model=self._create_model(),
            device=self.device,
        )

        self.logger.info("✅ Trainer created")
        return trainer

    def _create_model(self):
        """Create model instance."""
        from model.transformer import MultiBranchTransformer
        
        model = MultiBranchTransformer(
            config=self.config.model,
            action_space_size=self.config.model.action_space_size,
        ).to(self.device)
        
        return model


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments - simplified config-driven approach."""
    parser = argparse.ArgumentParser(description="Fx-AI Trading System", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="training-default", help="Configuration override file (determines training mode and all settings)")
    parser.add_argument("--spec", type=str, default=None, help="Optuna study specification file (only for optuna configs)")
    return parser.parse_args()


def execute_training(training_manager: TrainingManager, config: Config, app: 'ApplicationBootstrap') -> None:
    """Execute training based on configuration."""
    mode_type = TrainingMode(config.training.training_manager.mode)
    logger.info(f"🎯 Starting {mode_type.value} based on configuration")

    training_manager.start(
        trainer=app.trainer,
        environment= app.environment,
        data_manager=app.data_manager,
        callback_manager=app.callback_manager,
    )

    logger.info(f"✅ Training completed successfully")



def main() -> int:
    """Main entry point for V2 FxAI system with graceful shutdown support.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Use graceful shutdown context manager
    with graceful_shutdown_context() as shutdown_manager:
        try:
            # Parse simplified arguments
            args = parse_arguments()

            # Create application with shutdown manager
            app = ApplicationBootstrap()

            # Initialize application (now includes components)
            app.initialize(args)

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
    exit_code = main()
    sys.exit(exit_code)
