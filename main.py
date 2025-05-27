import os
import sys
import signal
import threading
import logging
import time
from datetime import datetime
import argparse
from pathlib import Path

import torch
import numpy as np
import wandb

from config.loader import load_config, check_unused_configs
from config.schemas import (
    Config, ModelConfig, EnvConfig, TrainingConfig, 
    DataConfig, SimulationConfig, WandbConfig, DashboardConfig
)
from simulators.portfolio_simulator import PortfolioSimulator
from utils.logger import console, setup_rich_logging

# Import utilities
from utils.model_manager import ModelManager

# Import components
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DatabentoFileProvider
from simulators.market_simulator import MarketSimulator
from simulators.execution_simulator import ExecutionSimulator
from feature.feature_extractor import FeatureExtractor
from envs.trading_env import TradingEnvironment
from agent.ppo_agent import PPOTrainer
from ai.transformer import MultiBranchTransformer
from metrics.factory import create_trading_metrics_system
from agent.base_callbacks import ModelCheckpointCallback, EarlyStoppingCallback, TrainingCallback
from agent.continous_training_callbacks import ContinuousTrainingCallback

# Global variables for signal handling
training_interrupted = False
cleanup_called = False
current_components = {}  # Store all components for cleanup

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global training_interrupted, cleanup_called
    
    if cleanup_called:
        console.print("\n[bold red]Force exit...[/bold red]")
        os._exit(1)
    
    training_interrupted = True
    console.print("\n" + "=" * 50)
    console.print("[bold red]INTERRUPT SIGNAL RECEIVED[/bold red]")
    console.print("Attempting graceful shutdown...")
    console.print("Press Ctrl+C again to force exit")
    console.print("=" * 50)
    
    # Stop dashboard immediately if running
    if 'metrics_manager' in current_components:
        metrics_manager = current_components['metrics_manager']
        if hasattr(metrics_manager, 'dashboard_collector') and metrics_manager.dashboard_collector:
            try:
                metrics_manager.dashboard_collector.stop()
            except:
                pass
    
    def force_exit():
        time.sleep(5)
        if training_interrupted and not cleanup_called:
            console.print("\n[bold red]Graceful shutdown timeout. Force exiting...[/bold red]")
            os._exit(1)
    
    timer = threading.Timer(10, force_exit)
    timer.daemon = True
    timer.start()


def cleanup_resources():
    """Clean up resources gracefully"""
    global cleanup_called, current_components
    
    if cleanup_called:
        return
    cleanup_called = True
    
    try:
        logging.info("Starting resource cleanup...")
        
        # Clean up in reverse order of creation
        cleanup_order = ['metrics_manager', 'trainer', 'env', 'data_manager']
        
        for component_name in cleanup_order:
            if component_name in current_components:
                component = current_components[component_name]
                try:
                    if hasattr(component, 'close'):
                        component.close()
                    logging.info(f"{component_name} closed")
                except Exception as e:
                    logging.error(f"Error closing {component_name}: {e}")
        
        # Save interrupted model if trainer exists
        if 'trainer' in current_components and 'model_manager' in current_components:
            trainer = current_components['trainer']
            model_manager = current_components['model_manager']
            if hasattr(trainer, 'model'):
                try:
                    interrupted_path = Path(model_manager.base_dir) / "interrupted_model.pt"
                    model_manager.save_checkpoint(
                        trainer.model,
                        trainer.optimizer,
                        trainer.global_step_counter,
                        trainer.global_episode_counter,
                        trainer.global_update_counter,
                        {"interrupted": True}
                    )
                    logging.info(f"Saved interrupted model to: {interrupted_path}")
                except Exception as e:
                    logging.error(f"Error saving interrupted model: {e}")
        
        logging.info("Resource cleanup completed")
        
    except Exception as e:
        console.print(f"[bold red]Error during cleanup: {e}[/bold red]")


def setup_environment(training_config: TrainingConfig) -> torch.device:
    """Set up training environment with config"""
    # Set random seeds
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.seed)
    
    # Select device
    device_str = training_config.device
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def create_data_provider(data_config: DataConfig):
    """Create data provider from config"""
    logging.info(f"Initializing data provider: {data_config.provider}")
    
    if data_config.provider == "databento":
        # Construct paths for databento
        # Use the actual directory name which is "Mlgo" with capital M
        data_path = Path(data_config.data_dir) / "Mlgo"
        
        return DatabentoFileProvider(
            data_dir=str(data_path),
            symbol_info_file=None,  # Optional CSV file, not needed
            verbose=True,
            dbn_cache_size=10  # Default cache size
        )
    else:
        raise ValueError(f"Unknown provider type: {data_config.provider}")


def create_data_components(config: Config, logger: logging.Logger):
    """Create data-related components with proper config passing"""
    # Data Manager with DataConfig
    data_provider = create_data_provider(config.data)
    data_manager = DataManager(
        provider=data_provider,
        logger=logger
    )
    
    # Market Simulator with SimulationConfig, ModelConfig and symbol
    market_simulator = MarketSimulator(
        symbol=config.env.symbol,
        data_manager=data_manager,
        simulation_config=config.simulation,
        model_config=config.model,
        start_time=config.data.start_date,
        end_time=config.data.end_date,
        logger=logger
    )
    
    return data_manager, market_simulator


def create_simulation_components(env_config: EnvConfig, 
                               simulation_config: SimulationConfig,
                               model_config: ModelConfig,
                               logger: logging.Logger):
    """Create simulation components with proper config passing"""
    # NOTE: These simulators are not actually used by TradingEnvironment
    # The environment creates its own simulators in setup_session()
    # This function is kept for backward compatibility but could be removed
    return None, None


def create_env_components(config: Config, market_simulator, logger: logging.Logger):
    """Create environment and related components with proper config passing"""
    # NOTE: Feature extractor is created inside TradingEnvironment.setup_session()
    # so we don't need to create it here
    
    # Trading Environment with full config (for now, will be refactored later)
    env = TradingEnvironment(
        config=config,  # TODO: Refactor to take specific configs
        data_manager=market_simulator.data_manager,
        logger=logger,
        metrics_integrator=None  # Will be set later
    )
    
    return env


def create_metrics_components(config: Config, logger: logging.Logger):
    """Create metrics and dashboard components with proper config passing"""
    if not config.wandb.enabled:
        logging.warning("W&B disabled - no metrics will be tracked")
        return None, None
    
    logging.info("📊 Initializing metrics system with W&B integration")
    
    # Create run name
    run_name = config.wandb.name
    if not run_name and config.training.continue_training:
        run_name = f"continuous_{config.env.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create metrics system with WandbConfig
    metrics_manager, metrics_integrator = create_trading_metrics_system(
        project_name=config.wandb.project,
        symbol=config.env.symbol,
        initial_capital=config.env.initial_capital,
        entity=config.wandb.entity,
        run_name=run_name,
        config=config.model_dump()  # Pass entire config for tracking
    )
    
    # Start auto-transmission
    metrics_manager.start_auto_transmit()
    
    # Start system tracking
    metrics_integrator.start_system_tracking()
    
    logging.info("✅ Metrics system initialized, auto-transmission started")
    
    # Enable dashboard if configured
    if config.dashboard.enabled:
        from dashboard import LiveTradingDashboard
        from dashboard import DashboardMetricsCollector
        
        dashboard = LiveTradingDashboard(
            port=config.dashboard.port,
            update_interval=config.dashboard.update_interval
        )
        
        dashboard_collector = DashboardMetricsCollector(dashboard)
        metrics_manager.dashboard_collector = dashboard_collector
        metrics_manager._dashboard_enabled = True
        
        dashboard_collector.start(open_browser=True)
        logging.info(f"🚀 Live dashboard enabled at http://localhost:{config.dashboard.port}")
        
        # Set initial model info
        model_name = f"PPO_Transformer_{config.env.symbol}"
        dashboard_collector.set_model_info(model_name)
    
    return metrics_manager, metrics_integrator


def create_model_components(config: Config, env, device: torch.device, 
                           output_dir: str, logger: logging.Logger):
    """Create model and training components with proper config passing"""
    # Model Manager with TrainingConfig
    model_manager = ModelManager(
        base_dir=str(Path(output_dir) / "models"),
        best_models_dir="best_models",
        model_prefix=config.env.symbol,
        max_best_models=config.training.keep_best_n_models,
        symbol=config.env.symbol
    )
    
    # Get observation to initialize model
    obs, info = env.reset()
    
    # Create model with ModelConfig
    model = MultiBranchTransformer(
        model_config=config.model,
        device=device,
        logger=logger
    )
    logging.info("✅ Model created successfully")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate
    )
    
    return model, optimizer, model_manager


def create_training_callbacks(config: Config, model_manager, output_dir: str,
                            loaded_metadata: dict = None) -> list:
    """Create training callbacks with proper config passing"""
    callbacks: list[TrainingCallback] = []
    
    # Model checkpoint callback
    callbacks.append(
        ModelCheckpointCallback(
            save_dir=str(Path(output_dir) / "models"),
            save_freq=config.training.checkpoint_interval,
            prefix=config.env.symbol,
            save_best_only=True
        )
    )
    
    # Early stopping if configured
    if config.training.early_stop_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                patience=config.training.early_stop_patience,
                min_delta=config.training.early_stop_min_delta
            )
        )
        logging.info(f"⏹️ Early stopping enabled (patience: {config.training.early_stop_patience})")
    
    # Continuous training callback
    if config.training.continue_training:
        try:
            continuous_callback = ContinuousTrainingCallback(
                model_manager=model_manager,
                reward_metric=config.training.best_model_metric,
                checkpoint_sync_frequency=config.training.checkpoint_interval,
                lr_annealing={
                    'enabled': config.training.use_lr_annealing,
                    'factor': config.training.lr_annealing_factor,
                    'patience': config.training.lr_annealing_patience,
                    'min_lr': config.training.min_learning_rate
                },
                load_metadata=loaded_metadata or {}
            )
            callbacks.append(continuous_callback)
            logging.info("🔄 Continuous training callback added")
        except Exception as e:
            logging.error(f"Failed to initialize continuous training callback: {e}")
            logging.warning("Continuing without continuous training callback...")
    
    return callbacks


def train(config: Config):
    """Main training function with proper config passing"""
    global current_components
    
    # Setup logging
    logger = setup_rich_logging(
        level=config.logging.level,
    )
    
    logger.info("="*80)
    logger.info(f"🚀 Starting FX-AI Training System")
    logger.info(f"📊 Experiment: {config.experiment_name}")
    logger.info(f"📈 Symbol: {config.env.symbol}")
    logger.info(f"🧠 Model: d_model={config.model.d_model}, layers={config.model.n_layers}")
    logger.info(f"🎯 Action space: {config.model.action_dim[0]}×{config.model.action_dim[1]}")
    logger.info("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"{config.experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save used config
    config.save_used_config(str(output_dir / "config_used.yaml"))
    
    # Setup environment
    device = setup_environment(config.training)
    
    # Create components with proper config passing
    try:
        # Data components
        data_manager, market_simulator = create_data_components(config, logger)
        current_components['data_manager'] = data_manager
        
        # Simulation components (not used, but kept for compatibility)
        _, _ = create_simulation_components(
            config.env, config.simulation, config.model, logger
        )
        
        # Environment components
        env = create_env_components(
            config, market_simulator, logger
        )
        current_components['env'] = env
        
        # Setup environment session
        env.setup_session(
            symbol=config.env.symbol,
            start_time=config.data.start_date,
            end_time=config.data.end_date
        )
        
        # Metrics components
        metrics_manager, metrics_integrator = create_metrics_components(config, logger)
        current_components['metrics_manager'] = metrics_manager
        
        # Update environment with metrics
        if metrics_integrator:
            env.metrics_integrator = metrics_integrator
        
        # Model components
        model, optimizer, model_manager = create_model_components(
            config, env, device, str(output_dir), logger
        )
        current_components['model_manager'] = model_manager
        
        # Register model metrics
        if metrics_manager and metrics_integrator:
            from metrics.collectors.model_metrics import ModelMetricsCollector, OptimizerMetricsCollector
            model_collector = ModelMetricsCollector(model)
            optimizer_collector = OptimizerMetricsCollector(optimizer)
            metrics_manager.register_collector(model_collector)
            metrics_manager.register_collector(optimizer_collector)
            logger.info("📊 Model and optimizer collectors registered")
        
        # Load best model if continuing
        loaded_metadata = {}
        if config.training.continue_training:
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                logger.info(f"📂 Loading best model: {best_model_info['path']}")
                model, training_state = model_manager.load_model(
                    model, optimizer, best_model_info['path']
                )
                loaded_metadata = training_state.get('metadata', {})
                logger.info(f"✅ Model loaded: step={training_state.get('global_step', 0)}")
            else:
                logger.info("🆕 No previous model found. Starting fresh.")
        
        # Create callbacks
        callbacks = create_training_callbacks(
            config, model_manager, str(output_dir), loaded_metadata
        )
        
        # Extract training parameters for PPO
        training_params = {
            "lr": config.training.learning_rate,
            "gamma": config.training.gamma,
            "gae_lambda": config.training.gae_lambda,
            "clip_eps": config.training.clip_epsilon,
            "critic_coef": config.training.value_coef,
            "entropy_coef": config.training.entropy_coef,
            "max_grad_norm": config.training.max_grad_norm,
            "ppo_epochs": config.training.n_epochs,
            "batch_size": config.training.batch_size,
            "rollout_steps": config.training.rollout_steps
        }
        
        # Create trainer
        trainer = PPOTrainer(
            env=env,
            model=model,
            metrics_integrator=metrics_integrator,
            model_config=config.model,
            device=device,
            output_dir=str(output_dir),
            callbacks=callbacks,
            **training_params
        )
        current_components['trainer'] = trainer
        
        # Check for interruption
        if training_interrupted:
            logger.warning("Training interrupted before starting")
            return {"interrupted": True}
        
        # Start training metrics
        if metrics_integrator:
            metrics_integrator.start_training()
        
        # Main training loop
        logger.info(f"🚀 Starting training for {config.training.total_updates} updates")
        logger.info(f"   ({config.training.total_updates * config.training.rollout_steps:,} steps)")
        
        try:
            training_stats = trainer.train(
                total_training_steps=config.training.total_updates * config.training.rollout_steps,
                eval_freq_steps=config.training.eval_frequency * config.training.rollout_steps
            )
            
            if training_interrupted:
                logger.warning("⚠️ Training was interrupted by user")
                training_stats["interrupted"] = True
            else:
                logger.info("🎉 Training completed successfully!")
            
            return training_stats
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user")
            return {"interrupted": True}
        
    except Exception as e:
        logger.error(f"Critical error during training: {e}", exc_info=True)
        raise
    
    finally:
        # Check for unused configs
        check_unused_configs()
        
        # Cleanup
        cleanup_resources()
        
        if config.wandb.enabled:
            wandb.finish()


def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="FX-AI Trading System")
    parser.add_argument('--config', type=str, default=None,
                        help='Config override file (e.g., quick_test, production)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Trading symbol (overrides config)')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue from latest checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], default=None,
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply command line overrides
        if args.experiment:
            config.experiment_name = args.experiment
        if args.symbol:
            config.env.symbol = args.symbol
            config.data.symbols = [args.symbol]
        if args.continue_training:
            config.training.continue_training = True
        if args.device:
            config.training.device = args.device
        
        # Log configuration
        console.print(f"[bold green]Loaded configuration:[/bold green] {args.config or 'defaults'}")
        if args.symbol:
            console.print(f"[bold blue]Symbol override:[/bold blue] {args.symbol}")
        if args.continue_training:
            console.print(f"[bold yellow]Continuing from checkpoint[/bold yellow]")
        
        # Run training
        train(config)
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Config file not found:[/bold red] {e}")
        console.print("Available configs in config/overrides/:")
        overrides_dir = Path("config/overrides")
        if overrides_dir.exists():
            for f in overrides_dir.glob("*.yaml"):
                console.print(f"  - {f.stem}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()