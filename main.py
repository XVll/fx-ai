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
from data.scanner.momentum_scanner import MomentumScanner
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
    
    # Stop metrics auto-transmission immediately if running
    if 'metrics_manager' in current_components:
        metrics_manager = current_components['metrics_manager']
        try:
            # Stop auto-transmission first to prevent further W&B logging
            if hasattr(metrics_manager, 'stop_auto_transmit'):
                metrics_manager.stop_auto_transmit()
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
        
        # Stop metrics auto-transmission first to prevent further W&B logging
        if 'metrics_manager' in current_components:
            metrics_manager = current_components['metrics_manager']
            try:
                if hasattr(metrics_manager, 'stop_auto_transmit'):
                    metrics_manager.stop_auto_transmit()
                    logging.info("Stopped metrics auto-transmission")
                # Give threads time to stop
                import time
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error stopping auto-transmission: {e}")
        
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
                    saved_path = model_manager.save_checkpoint(
                        trainer.model,
                        trainer.optimizer,
                        trainer.global_step_counter,
                        trainer.global_episode_counter,
                        trainer.global_update_counter,
                        {"interrupted": True},
                        str(interrupted_path)
                    )
                    if saved_path:
                        logging.info(f"Saved interrupted model to: {saved_path}")
                    else:
                        logging.warning("Failed to save interrupted model")
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
    
    # Create momentum scanner with rank-based scoring
    momentum_scanner = MomentumScanner(
        data_dir=str(Path(config.data.data_dir) / "mlgo"),
        output_dir=f"{config.data.index_dir}/momentum_index",
        momentum_config=config.momentum_scanning,
        scoring_config=config.momentum_scoring,
        session_config=config.session_volume,
        logger=log
    )
    
    data_manager = DataManager(
        provider=data_provider,
        momentum_scanner=momentum_scanner,
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


def get_feature_names_from_config():
    """Get feature names for each branch based on the feature system"""
    return {
        'hf': [
            'price_velocity', 'price_acceleration', 'tape_imbalance', 
            'tape_aggression_ratio', 'spread_compression', 'quote_velocity',
            'quote_imbalance', 'volume_velocity', 'volume_acceleration'
        ],
        'mf': [
            '1m_position_in_current_candle', '5m_position_in_current_candle',
            '1m_body_size_relative', '5m_body_size_relative',
            'distance_to_ema9_1m', 'distance_to_ema20_1m', 'distance_to_ema9_5m', 'distance_to_ema20_5m',
            'ema_interaction_pattern', 'ema_crossover_dynamics', 'ema_trend_alignment',
            'swing_high_distance_1m', 'swing_low_distance_1m', 'swing_high_distance_5m', 'swing_low_distance_5m',
            'price_velocity_1m', 'price_velocity_5m', 'volume_velocity_1m', 'volume_velocity_5m',
            'price_acceleration_1m', 'price_acceleration_5m', 'volume_acceleration_1m', 'volume_acceleration_5m',
            'distance_to_vwap', 'vwap_slope', 'price_vwap_divergence',
            'vwap_interaction_dynamics', 'vwap_breakout_quality', 'vwap_mean_reversion_tendency',
            'relative_volume', 'volume_surge', 'cumulative_volume_delta', 'volume_momentum',
            'professional_ema_system', 'professional_vwap_analysis', 'professional_momentum_quality', 'professional_volatility_regime',
            'trend_acceleration', 'volume_pattern_evolution', 'momentum_quality', 'pattern_maturation',
            'mf_trend_consistency', 'mf_volume_price_divergence', 'mf_momentum_persistence',
            'volatility_adjusted_momentum', 'regime_relative_volume',
            '1m_position_in_previous_candle', '5m_position_in_previous_candle',
            '1m_upper_wick_relative', '1m_lower_wick_relative', '5m_upper_wick_relative', '5m_lower_wick_relative'
        ],
        'lf': [
            'daily_range_position', 'prev_day_range_position', 'price_change_from_prev_close',
            'support_distance', 'resistance_distance', 'whole_dollar_proximity', 'half_dollar_proximity',
            'market_session_type', 'time_of_day_sin', 'time_of_day_cos',
            'halt_state', 'time_since_halt', 'distance_to_luld_up', 'distance_to_luld_down', 'luld_band_width',
            'session_progress', 'market_stress', 'session_volume_profile',
            'adaptive_support_resistance', 'hf_momentum_summary', 'hf_volume_dynamics', 'hf_microstructure_quality'
        ],
        'portfolio': [
            'position_side', 'position_size', 'unrealized_pnl', 'realized_pnl', 'total_pnl'
        ]
    }


def create_metrics_components(config: Config, log: logging.Logger, model: torch.nn.Module = None):
    """Create metrics and dashboard components with feature attribution support"""
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
    
    # Get feature names for attribution analysis
    feature_names = get_feature_names_from_config()
    logging.info(f"🔍 Feature attribution enabled with {sum(len(names) for names in feature_names.values())} total features")
    
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
    
    # Create metrics system with feature attribution
    additional_config = {
        'feature_names': feature_names,
        'enable_feature_attribution': True,
        'model_config': config.model
    }
    metrics_manager, metrics_integrator = metrics_config.create_metrics_system(
        model=model,
        additional_config=additional_config
    )
    
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
    # Model Manager with cache directory structure
    model_manager = ModelManager(
        base_dir="cache/model/checkpoint",
        best_models_dir="cache/model/best",
        model_prefix="model",
        max_best_models=config.training.keep_best_n_models
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
        
        # For momentum-based training, don't pre-setup environment session
        # The PPO agent will handle day selection and environment setup
        logger.info("🎯 Momentum-based training enabled - PPO agent will manage day selection")
        
        # Model components (create first to pass to metrics)
        model, optimizer, model_manager = create_model_components(
            config, device, str(output_dir), logger
        )
        current_components['model_manager'] = model_manager
        
        # Metrics components with feature attribution
        metrics_manager, metrics_integrator = create_metrics_components(config, logger, model)
        current_components['metrics_manager'] = metrics_manager
        
        # Update environment with metrics
        if metrics_integrator:
            env.metrics_integrator = metrics_integrator
        
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
        else:
            # Starting fresh training - save initial model
            logger.info("🆕 Starting fresh training - saving initial model")
            initial_metrics = {
                "mean_reward": 0.0,
                "episode_count": 0,
                "update_iter": 0,
                "timestamp": time.time(),
                "initial_model": True,
                "symbol": config.env.symbol
            }
            
            # Create a temporary checkpoint to save as initial model
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Save model state to temporary file
            import torch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step_counter': 0,
                'global_episode_counter': 0,
                'global_update_counter': 0,
                'model_config': config.model.model_dump() if hasattr(config.model, 'model_dump') else {}
            }, temp_path)
            
            # Save to best_models directory
            model_manager.save_best_model(temp_path, initial_metrics, 0.0)
            
            # Clean up temporary file
            import os
            os.unlink(temp_path)
            
            logger.info("✅ Initial model saved to best_models directory")
        
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
        
        # Enable momentum-based training with curriculum learning
        training_params.update({
            "curriculum_method": "quality_based",
            "min_quality_threshold": 0.3,
            "episode_selection_mode": "momentum_days"
        })
        
        # Add day selection configuration if available
        if hasattr(config, 'day_selection'):
            training_params.update({
                "episodes_per_day": config.day_selection.episodes_per_day,
                "reset_point_quality_range": config.day_selection.reset_point_quality_range,
                "day_switching_strategy": config.day_selection.day_switching_strategy
            })
        
        # Create trainer
        trainer = PPOTrainer(
            env=env,
            model=model,
            metrics_integrator=metrics_integrator,
            model_config=config.model,
            config=config,  # Pass full config for curriculum access
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
        
        # Initialize momentum-based training by selecting first momentum day
        logger.info("🎯 Initializing momentum-based training...")
        
        # Set the primary asset first so momentum methods work
        env.primary_asset = config.env.symbol
        
        if not trainer._select_next_momentum_day():
            logger.error("Failed to select initial momentum day, falling back to config end date")
            # Fallback to setting up environment with config end date
            fallback_date = datetime.strptime(config.data.end_date, "%Y-%m-%d") if config.data.end_date else datetime.now()
            env.setup_session(symbol=config.env.symbol, date=fallback_date)
        else:
            # Set up environment with the selected momentum day
            current_day = trainer.current_momentum_day
            env.setup_session(
                symbol=current_day['symbol'], 
                date=current_day['date']
            )
            logger.info(f"✅ Initial momentum day set: {current_day['date'].strftime('%Y-%m-%d')} "
                       f"(quality: {current_day.get('quality_score', 0):.3f})")
        
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
    parser.add_argument('--no-dashboard', dest='no_dashboard', action='store_true',
                        help='Disable dashboard for automated runs')
    
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
        if args.no_dashboard:
            config.dashboard.enabled = False
        
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