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
    Config, TrainingConfig, DataConfig, WandbConfig
)
from simulators.portfolio_simulator import PortfolioSimulator
from utils.logger import console, setup_rich_logging, get_logger

# Import utilities
from utils.model_manager import ModelManager

# Import components
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DatabentoFileProvider
from envs.trading_environment import TradingEnvironment
from agent.ppo_agent import PPOTrainer
from ai.transformer import MultiBranchTransformer
from metrics.factory import MetricsConfig
from agent.base_callbacks import ModelCheckpointCallback, EarlyStoppingCallback, TrainingCallback, MomentumTrackingCallback
from agent.continuous_training_callbacks import ContinuousTrainingCallback

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
        try:
            # Close all transmitters which includes dashboard
            for transmitter in metrics_manager.transmitters:
                if hasattr(transmitter, 'close'):
                    transmitter.close()
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
        # Use the actual directory name which is "mlgo" with capital M
        data_path = Path(data_config.data_dir) / "mlgo"
        
        return DatabentoFileProvider(
            data_dir=str(data_path),
            symbol_info_file=None,  # Optional CSV file, not needed
            verbose=False,
            dbn_cache_size=10  # Default cache size
        )
    else:
        raise ValueError(f"Unknown provider type: {data_config.provider}")


def create_data_components(config: Config, log: logging.Logger):
    """Create data-related components with proper config passing"""
    # Data Manager with DataConfig
    data_provider = create_data_provider(config.data)
    data_manager = DataManager(
        provider=data_provider,
        logger=log
    )
    
    return data_manager


# Simulation components are now created inside TradingEnvironment.setup_session()


def create_env_components(config: Config, data_manager: DataManager, log: logging.Logger):
    """Create environment and related components with proper config passing"""
    # Trading Environment with full config
    env = TradingEnvironment(
        config=config,
        data_manager=data_manager,
        logger=log,
        metrics_integrator=None  # Will be set later
    )
    
    return env


def create_metrics_components(config: Config, log: logging.Logger):
    """Create metrics and dashboard components with proper config passing"""
    # Dashboard can work without W&B - only return None if both are disabled
    if not config.wandb.enabled and not config.dashboard.enabled:
        logging.warning("Both W&B and dashboard disabled - no metrics will be tracked")
        return None, None
    
    # Only log about W&B if it's enabled
    if config.wandb.enabled:
        logging.info("📊 Initializing metrics system with W&B integration")
    else:
        logging.info("📊 Initializing metrics system (W&B disabled)")
    
    # Create run name
    run_name = config.wandb.name
    if not run_name and config.training.continue_training:
        run_name = f"continuous_{config.env.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create metrics config
    metrics_config = MetricsConfig(
        wandb_project=config.wandb.project if config.wandb.enabled else "local",
        wandb_entity=config.wandb.entity if config.wandb.enabled else None,
        wandb_run_name=run_name,
        wandb_tags=["trading", "ppo", config.env.symbol],
        symbol=config.env.symbol,
        initial_capital=config.env.initial_capital,
        enable_dashboard=config.dashboard.enabled,
        dashboard_port=config.dashboard.port,
        transmit_interval=0.5  # More frequent for trading
    )
    
    # Create metrics system
    metrics_manager, metrics_integrator = metrics_config.create_metrics_system()
    
    # Start auto-transmission
    metrics_manager.start_auto_transmit()
    
    # Start system tracking
    metrics_integrator.start_system_tracking()
    
    logging.info("✅ Metrics system initialized, auto-transmission started")
    
    # Start dashboard if enabled
    if config.dashboard.enabled:
        metrics_manager.start_dashboard(open_browser=True)
        logging.info(f"🚀 Live dashboard enabled at http://localhost:{config.dashboard.port}")
    
    return metrics_manager, metrics_integrator


def create_model_components(config: Config, device: torch.device, 
                           output_dir: str, log: logging.Logger):
    """Create model and training components with proper config passing"""
    # Model Manager with TrainingConfig
    model_manager = ModelManager(
        base_dir=str(Path(output_dir) / "models"),
        best_models_dir="best_models",
        model_prefix=config.env.symbol,
        max_best_models=config.training.keep_best_n_models,
        symbol=config.env.symbol
    )
    
    # Model dimensions are known from config, no need to reset env here
    
    # Create model with ModelConfig
    model = MultiBranchTransformer(
        model_config=config.model,
        device=device,
        logger=log
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
    
    # Momentum tracking callback for momentum-based training
    if hasattr(config.env, 'use_momentum_training') and config.env.use_momentum_training:
        momentum_callback = MomentumTrackingCallback(log_frequency=10)
        callbacks.append(momentum_callback)
        logging.info("🎯 Momentum tracking callback added")
    
    return callbacks


def train(config: Config):
    """Main training function with proper config passing"""
    global current_components
    
    # Setup logging
    setup_rich_logging(
        level=config.logging.level,
    )
    logger = get_logger("fx-ai")
    
    logger.info("="*80)
    logger.info(f"🚀 Starting FX-AI Training System")
    logger.info(f"📊 Experiment: {config.experiment_name}")
    logger.info(f"📈 Symbol: {config.env.symbol}")
    
    # Handle both dict and object access patterns
    model_cfg = config.model
    d_model = model_cfg.d_model if hasattr(model_cfg, 'd_model') else model_cfg.get('d_model', 64)
    n_layers = model_cfg.n_layers if hasattr(model_cfg, 'n_layers') else model_cfg.get('n_layers', 4)
    action_dim = model_cfg.action_dim if hasattr(model_cfg, 'action_dim') else model_cfg.get('action_dim', [3, 4])
    
    logger.info(f"🧠 Model: d_model={d_model}, layers={n_layers}")
    logger.info(f"🎯 Action space: {action_dim[0]}×{action_dim[1]}")
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
        data_manager = create_data_components(config, logger)
        current_components['data_manager'] = data_manager
        
        # Environment components  
        env = create_env_components(
            config, data_manager, logger
        )
        current_components['env'] = env
        
        # Setup environment session
        # Get available momentum days from data manager
        try:
            momentum_days = data_manager.get_momentum_days(
                symbol=config.env.symbol,
                min_quality=0.5  # Use configurable threshold in future
            )
            if not momentum_days.empty:
                # Use the first available momentum day
                first_day = momentum_days.iloc[0]
                session_date = first_day['date']
                logger.info(f"🎯 Selected momentum day: {session_date} (quality: {first_day['activity_score']:.3f})")
            else:
                # Fallback to latest available date in the data
                logger.warning("No momentum days found, using latest available date")
                session_date = datetime(2025, 4, 29)  # Last available data date
        except Exception as e:
            logger.warning(f"Failed to get momentum days: {e}, using fallback date")
            session_date = datetime(2025, 4, 29)  # Last available data date
            
        env.setup_session(
            symbol=config.env.symbol,
            date=session_date
        )
        
        # Metrics components
        metrics_manager, metrics_integrator = create_metrics_components(config, logger)
        current_components['metrics_manager'] = metrics_manager
        
        # Update environment with metrics
        if metrics_integrator:
            env.metrics_integrator = metrics_integrator
        
        # Model components
        model, optimizer, model_manager = create_model_components(
            config, device, str(output_dir), logger
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
        
        # Add momentum-based training parameters
        if hasattr(config.env, 'use_momentum_training') and config.env.use_momentum_training:
            training_params.update({
                "curriculum_strategy": getattr(config.env, 'curriculum_strategy', 'quality_based'),
                "min_quality_threshold": getattr(config.env, 'min_quality_threshold', 0.3),
                "episode_selection_mode": "momentum_days"
            })
        else:
            training_params.update({
                "episode_selection_mode": "standard"
            })
        
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