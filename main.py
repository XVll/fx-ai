import sys
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import hydra
import wandb
from hydra.core.hydra_config import HydraConfig

from core.evaluation import Evaluator
from core.model_manager import ModelManager
from core.path_manager import initialize_paths, get_path_manager
from data.data_manager import DataManager
from data.scanner.momentum_scanner import MomentumScanner
from data import DatabentoFileProvider
from envs.trading_environment import TradingEnvironment
from agent.ppo_agent import PPOTrainer
from model.transformer import MultiBranchTransformer
from training.episode_manager import EpisodeManager
from training.training_manager import TrainingManager
from callbacks import create_callbacks_from_config, CallbackManager
from core.logger import setup_rich_logging
from core.shutdown import (
    ShutdownReason,
    graceful_shutdown_context, get_global_shutdown_manager
)
from config.config import Config, register_configs
from training.training_manager import TrainingManager, TrainingMode
from core.evaluation.benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)


class ApplicationBootstrap:
    """Bootstrap the FxAI application with proper dependency injection and graceful shutdown."""

    def __init__(self, config: Config):
        # Initialize PathManager with Hydra integration
        self.output_path: Path = Path(HydraConfig.get().runtime.output_dir)
        self.path_manager = initialize_paths(
            use_hydra_output=True,
            experiment_dir=self.output_path
        )
        
        self.config: Config = config
        self.logger = logging.getLogger(f"{__name__}.Application")
        self.shutdown_manager = get_global_shutdown_manager()
        
        # Log path configuration
        self.logger.info(f"PathManager initialized: {self.path_manager}")

        # Component instances
        self.training_manager: Optional[TrainingManager] = None
        self.trainer: Optional[PPOTrainer] = None
        self.environment: Optional[TradingEnvironment] = None
        self.data_manager: Optional[DataManager] = None
        self.device: Optional[torch.device] = None
        self.callback_manager: Optional[CallbackManager] = None
        self.episode_manager: Optional[EpisodeManager] = None
        self.training_manager: Optional[TrainingManager] = None
        self.model_manager: Optional[ModelManager] = None

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
            compact_errors=self.config.logging.compact_errors,
            # Todo : add /logs output directory support (self.output_path / "logs")
        )

    def _create_training_manager(self) -> TrainingManager:
        """Create and configure a training manager with config."""
        self.logger.info("🎯 Creating training manager")

        training_manager = TrainingManager(
            config=self.config.training.training_manager,
            model_manager=self.model_manager,
            episode_manager=self.episode_manager
        )

        self.logger.info("✅ Training manager created")
        return training_manager

    def _initialize_components(self) -> None:
        """Initialize application components"""

        self.logger.info("🔧 Initializing application components")

        self.device = self._create_device()
        self.model_manager = ModelManager(self.config.model_storage)

        self.logger.info("🔧 Creating callbacks from configuration")
        self.callback_manager = create_callbacks_from_config(
            config=self.config.callbacks,
            attribution_config=self.config.attribution
        )
        self.logger.info(f"📊 Created {len(self.callback_manager.callbacks)} callbacks")

        self.data_manager = self._create_data_manager()
        self.environment = self._create_environment()
        self.trainer = self._create_trainer()
        self.episode_manager = EpisodeManager(self.config.training.training_manager, self.data_manager, self.callback_manager)
        self.training_manager = self._create_training_manager()

        self.callback_manager.register_trainer(trainer=self.trainer)
        self.callback_manager.register_environment(environment=self.environment)
        self.callback_manager.register_data_manager(data_manager=self.data_manager)
        self.callback_manager.register_episode_manager(episode_manager=self.training_manager.episode_manager)

        # Components automatically register for shutdown via IShutdownHandler metaclass

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

        momentum_scanner = MomentumScanner(scanner_config=self.config.scanner)
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
            callback_manager=self.callback_manager,
        )

        self.logger.info("✅ Trainer created")
        return trainer

    def _create_model(self) -> MultiBranchTransformer:
        """Create a model instance."""
        model = MultiBranchTransformer(model_config=self.config.model, device=self.device).to(self.device)

        return model



# Hydra handles argument parsing automatically


def execute_benchmark(config: Config, app: 'ApplicationBootstrap') -> None:
    """Execute benchmark based on configuration."""
    logger.info("🎯 Starting benchmark mode")

    # Create and run benchmark
    benchmark_runner = BenchmarkRunner(
        config.evaluation, 
        model_manager=app.training_manager.model_manager
    )
    result = benchmark_runner.run_benchmark(
        trainer=app.trainer,
        environment=app.environment,
        data_manager=app.data_manager,
        model_manager=app.training_manager.model_manager
    )

    if not result:
        raise RuntimeError("Benchmark execution failed")


def execute_training(training_manager: TrainingManager, app: ApplicationBootstrap) -> None:
    logger.info("🔄 Starting training loop")
    
    # Initialize WandB for experiment tracking
    try:
        import wandb
        import os
        if not wandb.run:
            # Set local WandB server configuration
            # os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'
            
            # Create simplified config for WandB (only serializable values)
            wandb_config = {
                "model_dim": getattr(app.config.model, 'dim', None),
                "learning_rate": getattr(app.config.training, 'learning_rate', None),
                "batch_size": getattr(app.config.training, 'batch_size', None),
                "n_epochs": getattr(app.config.training, 'n_epochs', None),
                "gamma": getattr(app.config.training, 'gamma', None),
                "device": getattr(app.config.training, 'device', None),
                "environment": getattr(app.config.env, 'name', 'TradingEnvironment'),
                "symbol": getattr(app.config.data, 'symbol', None),
                "rollout_steps": getattr(app.config.training.training_manager, 'rollout_steps', None)
            }
            # Filter out None values
            wandb_config = {k: v for k, v in wandb_config.items() if v is not None}
            
            wandb.init(
                project="fx-ai",
                entity="onur03-fx",
                config=wandb_config,
                tags=["momentum-trading", "ppo", "transformer"]
            )
            logger.info("🎯 WandB tracking initialized with local server")
    except ImportError:
        logger.warning("WandB not available - training will continue without experiment tracking")
    except Exception as e:
        logger.warning(f"WandB initialization failed: {e} - training will continue without experiment tracking")
    
    training_manager.start(
        trainer=app.trainer,
        environment=app.environment,
        data_manager=app.data_manager,
        callback_manager=app.callback_manager,
    )


def execute(training_manager: TrainingManager, config: Config, app: 'ApplicationBootstrap') -> None:
    """Execute training based on configuration."""
    mode_type = TrainingMode(config.training.training_manager.mode)
    logger.info(f"🎯 Starting {mode_type.value} based on configuration")

    # Check if this is a benchmark mode
    if mode_type == TrainingMode.BENCHMARK:
        execute_benchmark(config, app)
        return
    elif mode_type == TrainingMode.TRAINING:
        execute_training(training_manager, app)
        return

    logger.info(f"✅ Training completed successfully")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Config) -> int:
    """
    The Main entry point for the FxAI trading system.
    
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
            execute(
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
    wandb.finish()
    sys.exit(exit_code)
