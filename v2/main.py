"""
V2 FxAI Main Entry Point - TDD Implementation (Config-Driven)

This is the new entry point for the FxAI trading system using TDD approach.
Features purely config-driven training with minimal CLI arguments.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch

# V2 imports - using new interfaces
from v2.training.interfaces import ITrainingManager, RunMode
from v2.core.types import TerminationReason
from v2.core.shutdown import (
    IShutdownHandler, ShutdownManager, ShutdownReason, 
    graceful_shutdown_context, get_global_shutdown_manager
)
from v2.config.config import Config
from v2.utils.logger import setup_rich_logging, get_logger, console

logger = logging.getLogger(__name__)


class ApplicationBootstrap(IShutdownHandler):
    """Bootstrap the FxAI application with proper dependency injection and graceful shutdown."""

    def __init__(self, shutdown_manager: Optional[ShutdownManager] = None):
        self.config: Optional[Config] = None
        self.training_manager: Optional[ITrainingManager] = None
        self.shutdown_manager = shutdown_manager or get_global_shutdown_manager()
        self.logger = logging.getLogger(f"{__name__}.ApplicationBootstrap")
        
        # Register self for shutdown
        self.shutdown_manager.register_component(
            component=self,
            timeout=60.0  # 1 minute for app bootstrap cleanup
        )

    def initialize(self, args:Optional[argparse.Namespace]) -> None:
        """Initialize application with configuration."""

        self._setup_config(args.config, args.spec)
        self._setup_logging()
        self.training_manager = self._create_training_manager()
        self._initialize_components()

        output_path = self._setup_directories()
        self.config.save_used_config(str(output_path/"config.yaml"))

        logger.info("=" * 80)
        logger.info(f"[bold green] Loaded configuration:[/bold green] {self.config or 'defaults'}")
        logger.info(f"[bold green] Output directory created:[/bold green] {output_path}")
        logger.info("🚀 FxAI Application bootstrapped successfully")
        logger.info("=" * 80)

    def _setup_directories(self) -> Path:
        """Setup necessary directories for outputs and logs."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _setup_config(self,config:Optional[str], spec:Optional[str] ) -> Config:
        """Setup application configuration from v2 config."""
        self.config =  Config.load(config, spec)

    def _setup_logging(self) -> None:
        """Setup application logging using v2 rich logging system."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        setup_rich_logging(
            level=log_level,
            show_time=self.config.logging.show_time,
            show_path=self.config.logging.show_path,
            compact_errors=True
        )

    def _create_training_manager(self) -> ITrainingManager:
        """Create and configure training manager with v2 config."""
        from v2.training.training_manager import create_training_manager

        # Convert pydantic config to dict for training manager
        config_dict = self.config.model_dump()
        training_manager = create_training_manager(config_dict)

        return training_manager

    def _initialize_components(self) -> None:
        """Initialize application components.
        
        This method will create data managers, environments, etc.
        """
        # TODO: Implement component initialization
        pass

    def create_device(self) -> torch.device:
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

        logger.info(f"Using device: {device}")
        return device
    
    # IShutdownHandler implementation
    
    def shutdown(self) -> None:
        """Perform graceful shutdown - save state and cleanup resources."""
        self.logger.info("🛑 Shutting down ApplicationBootstrap")
        
        try:
            # Only clean up ApplicationBootstrap's own resources
            # Note: Other components (training_manager, etc.) are handled by ShutdownManager
            
            # Reset own state
            self.training_manager = None
            self.config = None
            
            self.logger.info("✅ ApplicationBootstrap shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during ApplicationBootstrap shutdown: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments - simplified config-driven approach."""
    parser = argparse.ArgumentParser(description="Fx-AI Trading System", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="training-default", help="Configuration override file (determines training mode and all settings)")
    parser.add_argument("--spec", type=str, default=None, help="Optuna study specification file (only for optuna configs)")
    return parser.parse_args()


def determine_training_mode(config: Config) -> RunMode:
    """Determine training mode from configuration. """
    # Mode is determined by config structure, not CLI arguments
    if hasattr(config, 'optuna') and config.optuna is not None:
        return RunMode.OPTUNA_OPTIMIZATION
    elif hasattr(config, 'benchmark') and getattr(config, 'benchmark', None) is not None:
        return RunMode.BENCHMARK_EVALUATION
    else:
        return RunMode.CONTINUOUS_TRAINING  # Default primary mode


def execute_training(training_manager: ITrainingManager, config: Config) -> Dict[str, Any]:
    """Execute training based on configuration. """
    # Determine mode from config
    mode_type = determine_training_mode(config)
    logger.info(f"🎯 Starting {mode_type.value} based on configuration")

    # TODO: Create trainer and environment components from config
    # trainer = create_trainer_from_config(config)
    # environment = create_environment_from_config(config)

    # For now, placeholder components for interface design
    trainer = None  # Will be actual trainer implementation
    environment = None  # Will be actual environment implementation

    # Execute the training mode
    try:
        # Convert pydantic config to dict for mode
        mode_config = config.model_dump()

        results = training_manager.start_mode(
            mode_type=mode_type,
            config=mode_config,
            trainer=trainer,
            environment=environment,
            background=False
        )

        logger.info(f"✅ Training completed successfully")
        return results

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        training_manager.request_termination(
            reason=TerminationReason.EXTERNAL_ERROR,
            mode_type=mode_type
        )
        raise


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
            app = ApplicationBootstrap(shutdown_manager)
            

            # Initialize application
            app.initialize(args)
            device = app.create_device()
            
            logger.info("🚀 Starting training with graceful shutdown support")
            
            # Execute training based on configuration
            results = execute_training(
                app.training_manager,
                app.config
            )
            
            # Check if shutdown was requested during training
            if shutdown_manager.is_shutdown_requested():
                logger.info("🛑 Training completed due to shutdown request")
                return 130  # Standard exit code for SIGINT
            
            # Log results
            logger.info(f"✅ Training completed successfully: {results}")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            shutdown_manager.initiate_shutdown(ShutdownReason.ERROR_CONDITION)
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
