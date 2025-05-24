# main_continuous.py - UPDATED: Metrics-based training without dashboard

import os
import sys
import signal
import threading
import logging
import time
from datetime import datetime

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Import Rich logging setup
from utils.logger import setup_rich_logging, console

from agent.continous_training_callback import ContinuousTrainingCallback
from ai.transformer import MultiBranchTransformer
from config.config import Config
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from data.provider.data_bento.databento_api_provider import DabentoAPIProvider
from data.provider.data_bento.databento_live_provider import DabentoLiveProvider
from data.provider.data_bento.databento_live_sim_provider import DabentoLiveSimProvider
from data.provider.dummy_data_provider import DummyDataProvider

from envs.trading_env import TradingEnvironment
from envs.wrappers.observation_wrapper import NormalizeDictObservation

from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, EarlyStoppingCallback
from utils.model_manager import ModelManager

# Import new metrics system
from metrics.factory import create_trading_metrics_system

# Global variables for signal handling
training_interrupted = False
cleanup_called = False
current_trainer = None
current_env = None
current_data_manager = None
metrics_manager = None

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global training_interrupted, cleanup_called

    if cleanup_called:
        console.print("\n[bold red]Force exit...[/bold red]")
        sys.exit(1)

    training_interrupted = True
    console.print("\n" + "=" * 50)
    console.print("[bold red]INTERRUPT SIGNAL RECEIVED[/bold red]")
    console.print("Attempting graceful shutdown...")
    console.print("Press Ctrl+C again to force exit")
    console.print("=" * 50)

    def force_exit():
        time.sleep(10)
        if training_interrupted and not cleanup_called:
            console.print("\n[bold red]Graceful shutdown timeout. Force exiting...[/bold red]")
            sys.exit(1)

    timer = threading.Timer(10, force_exit)
    timer.daemon = True
    timer.start()


def cleanup_resources():
    """Clean up resources gracefully"""
    global cleanup_called, current_trainer, current_env, current_data_manager, metrics_manager

    if cleanup_called:
        return
    cleanup_called = True

    try:
        logging.info("Starting resource cleanup...")

        # Close metrics manager
        if metrics_manager:
            try:
                metrics_manager.close()
                logging.info("Metrics manager closed")
            except Exception as e:
                logging.error(f"Error closing metrics manager: {e}")

        # Stop trainer
        if current_trainer and hasattr(current_trainer, 'model'):
            try:
                interrupted_model_path = os.path.join(
                    getattr(current_trainer, 'model_dir', './models'),
                    "interrupted_model.pt"
                )
                current_trainer.save_model(interrupted_model_path)
                logging.info(f"Saved interrupted model to: {interrupted_model_path}")
            except Exception as e:
                logging.error(f"Error saving interrupted model: {e}")

        # Close environment
        if current_env and hasattr(current_env, 'close'):
            try:
                current_env.close()
                logging.info("Environment closed")
            except Exception as e:
                logging.error(f"Error closing environment: {e}")

        # Close data manager
        if current_data_manager and hasattr(current_data_manager, 'close'):
            try:
                current_data_manager.close()
                logging.info("Data manager closed")
            except Exception as e:
                logging.error(f"Error closing data manager: {e}")

        logging.info("Resource cleanup completed")

    except Exception as e:
        console.print(f"[bold red]Error during cleanup: {e}[/bold red]")


def select_device(device_str):
    """Select device based on config or auto-detect."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
    else:
        device = torch.device(device_str)
        logging.info(f"Using specified device: {device}")

    return device


def create_data_provider(config: Config):
    """Create appropriate data provider based on config."""
    data_cfg = config.data
    logging.info(f"Initializing data provider: {data_cfg.provider_type}")

    if data_cfg.provider_type == "file":
        return DabentoFileProvider(
            data_dir=data_cfg.data_dir,
            symbol_info_file=data_cfg.symbol_info_file,
            verbose=data_cfg.verbose,
            dbn_cache_size=data_cfg.dbn_cache_size
        )
    elif data_cfg.provider_type == "api":
        return DabentoAPIProvider(
            api_key=data_cfg.api.get("api_key", ""),
            dataset=data_cfg.api.get("dataset", "XNAS.ITCH")
        )
    elif data_cfg.provider_type == "live":
        return DabentoLiveProvider(
            api_key=data_cfg.live.get("api_key", ""),
            dataset=data_cfg.live.get("dataset", "XNAS.ITCH")
        )
    elif data_cfg.provider_type == "live_sim":
        hist_provider = DabentoFileProvider(
            data_dir=data_cfg.data_dir,
            symbol_info_file=data_cfg.symbol_info_file,
            verbose=data_cfg.verbose,
            dbn_cache_size=data_cfg.dbn_cache_size
        )
        return DabentoLiveSimProvider(
            historical_provider=hist_provider,
            replay_speed=data_cfg.live.get("replay_speed", 1.0),
            start_time=data_cfg.start_date
        )
    elif data_cfg.provider_type == "dummy":
        return DummyDataProvider(config={
            'debug_window_mins': 120,
            'data_sparsity': 1,
            'num_squeezes': 2,
            'base_price': 5.00,
            'volatility': 0.03,
            'squeeze_magnitude': 0.30,
            'symbols': ['MLGO']
        })
    else:
        raise ValueError(f"Unknown provider type: {data_cfg.provider_type}")


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def run_training(cfg: Config):
    """Main training function with metrics integration"""
    global current_trainer, current_env, current_data_manager, metrics_manager

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Setup logging
        setup_rich_logging(level=logging.INFO, show_time=True, show_path=False)

        # Get output directory
        if HydraConfig.initialized():
            output_dir = HydraConfig.get().runtime.output_dir
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join("outputs", timestamp)
            os.makedirs(output_dir, exist_ok=True)

        model_dir = os.path.join(output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # Log startup info
        logging.info("🚀 Starting FX-AI Training System with Metrics Integration")
        logging.info(f"📁 Output directory: {output_dir}")

        # Save config for reference
        config_path = os.path.join(output_dir, "config_used.yaml")
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

        # Check continuous training mode
        continuous_mode = getattr(cfg.training, 'enabled', False)
        load_best_model = getattr(cfg.training, 'load_best_model', False) if continuous_mode else False
        best_models_dir = getattr(cfg.training, 'best_models_dir', "./best_models")
        os.makedirs(best_models_dir, exist_ok=True)

        if continuous_mode:
            logging.info(f"🔄 Continuous training mode enabled. Best models dir: {best_models_dir}")

        # Get symbol and date range
        symbol = cfg.data.symbol
        start_date = cfg.data.start_date
        end_date = cfg.data.end_date

        logging.info(f"📈 Trading symbol: {symbol}")
        logging.info(f"📅 Date range: {start_date} to {end_date}")

        # Initialize metrics system
        if cfg.wandb.enabled:
            logging.info("📊 Initializing metrics system with W&B integration")
            metrics_manager, metrics_integrator = create_trading_metrics_system(
                project_name=cfg.wandb.project_name,
                symbol=symbol,
                initial_capital=cfg.simulation.portfolio_config.initial_cash,
                entity=cfg.wandb.entity,
                run_name=f"continuous_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if continuous_mode else None
            )

            # Start system tracking
            metrics_integrator.start_system_tracking()
            logging.info("✅ Metrics system initialized and tracking started")
        else:
            logging.warning("W&B disabled - no metrics will be tracked")
            metrics_manager = None
            metrics_integrator = None

        # Create data provider and manager
        logging.info("📂 Initializing data provider and manager")
        data_provider = create_data_provider(cfg)
        current_data_manager = DataManager(provider=data_provider, logger=logging.getLogger("DataManager"))

        # Initialize model manager
        model_manager = ModelManager(
            base_dir=model_dir,
            best_models_dir=best_models_dir,
            model_prefix=symbol,
            max_best_models=getattr(cfg.training, 'max_best_models', 5),
            symbol=symbol
        )

        # Create environment
        logging.info(f"🏗️ Creating trading environment for {symbol}")
        current_env = TradingEnvironment(
            config=cfg,
            data_manager=current_data_manager,
            logger=logging.getLogger("TradingEnv"),
            metrics_integrator=metrics_integrator  # Pass metrics instead of dashboard
        )

        # Setup environment session
        current_env.setup_session(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date
        )

        # Wrap environment with observation normalization if required
        if cfg.env.normalize_state:
            logging.info("🔄 Wrapping environment with observation normalization")
            current_env = NormalizeDictObservation(current_env)

        # Check for interruption
        if training_interrupted:
            logging.warning("Training interrupted during setup")
            return {}

        # Select device
        device = select_device(cfg.training.device)

        # Create model
        logging.info("🧠 Creating multi-branch transformer model")
        model_config = OmegaConf.to_container(cfg.model, resolve=True)

        obs, info = current_env.reset()

        try:
            model = MultiBranchTransformer(
                **model_config,
                device=device,
                logger=logging.getLogger("Transformer")
            )
            logging.info("✅ Model created successfully")
        except Exception as e:
            logging.error(f"Error creating model: {e}")
            raise

        # Update metrics system with model
        if metrics_integrator:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
            # Add model and optimizer to metrics system
            from metrics.collectors.model_metrics import ModelMetricsCollector, OptimizerMetricsCollector
            model_collector = ModelMetricsCollector(model)
            optimizer_collector = OptimizerMetricsCollector(optimizer)
            metrics_manager.register_collector(model_collector)
            metrics_manager.register_collector(optimizer_collector)

        # Load best model for continuous training
        loaded_metadata = {}
        if continuous_mode and load_best_model:
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                logging.info(f"📂 Loading best model: {best_model_info['path']}")

                if not metrics_integrator:
                    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

                model, training_state = model_manager.load_model(model, optimizer, best_model_info['path'])
                loaded_metadata = training_state.get('metadata', {})

                logging.info("✅ Model loaded successfully")
                logging.info(f"📊 Training state: step={training_state.get('global_step', 0)}, "
                             f"episode={training_state.get('global_episode', 0)}, "
                             f"update={training_state.get('global_update', 0)}")
            else:
                logging.info("🆕 No previous model found. Starting fresh.")

        # Set up training callbacks
        logging.info("⚙️ Setting up training callbacks")
        callbacks = [
            ModelCheckpointCallback(
                save_dir=model_dir,
                save_freq=cfg.callbacks.save_freq,
                prefix=symbol,
                save_best_only=cfg.callbacks.save_best_only,
            ),
        ]

        # Add early stopping if enabled
        if cfg.callbacks.early_stopping.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    patience=cfg.callbacks.early_stopping.patience,
                    min_delta=cfg.callbacks.early_stopping.min_delta
                )
            )
            logging.info(f"⏹️ Early stopping enabled (patience: {cfg.callbacks.early_stopping.patience})")

        # Add continuous training callback if in continuous mode
        if continuous_mode:
            try:
                checkpoint_sync_frequency = getattr(cfg.training, 'checkpoint_sync_frequency', 5)
                lr_annealing = getattr(cfg.training, 'lr_annealing', {})

                continuous_callback = ContinuousTrainingCallback(
                    model_manager=model_manager,
                    reward_metric="mean_reward",
                    checkpoint_sync_frequency=checkpoint_sync_frequency,
                    lr_annealing=lr_annealing,
                    load_metadata=loaded_metadata
                )
                callbacks.append(continuous_callback)
                logging.info("🔄 Continuous training callback added")
            except Exception as e:
                logging.error(f"Failed to initialize continuous training callback: {e}")
                logging.warning("Continuing without continuous training callback...")

        # Extract training parameters
        training_params = {
            "lr": cfg.training.lr,
            "gamma": cfg.training.gamma,
            "gae_lambda": cfg.training.gae_lambda,
            "clip_eps": cfg.training.clip_eps,
            "critic_coef": cfg.training.critic_coef,
            "entropy_coef": cfg.training.entropy_coef,
            "max_grad_norm": cfg.training.max_grad_norm,
            "ppo_epochs": cfg.training.n_epochs,
            "batch_size": cfg.training.batch_size,
            "rollout_steps": cfg.training.buffer_size
        }

        # Create trainer
        current_trainer = PPOTrainer(
            env=current_env,
            model=model,
            metrics_integrator=metrics_integrator,  # Pass metrics integrator
            model_config=model_config,
            device=device,
            output_dir=output_dir,
            callbacks=callbacks,
            **training_params
        )

        # Check for interruption before training
        if training_interrupted:
            logging.warning("Training interrupted before starting")
            return {}

        # Train the model
        total_updates = cfg.training.total_updates
        total_steps = total_updates * cfg.training.buffer_size

        logging.info(f"🚀 Starting training for approximately {total_steps} steps ({total_updates} updates)")

        # Start training metrics
        if metrics_integrator:
            metrics_integrator.start_training()

        # Evaluate before training if in continuous mode
        if continuous_mode and getattr(cfg.training, 'startup_evaluation', False) and load_best_model:
            if training_interrupted:
                logging.warning("Training interrupted before startup evaluation")
                return {}

            logging.info("🔍 Performing initial evaluation...")
            try:
                eval_stats = current_trainer.evaluate(n_episodes=5)
                logging.info(f"📊 Initial evaluation results: Mean reward: {eval_stats.get('mean_reward', 'N/A')}")
            except Exception as e:
                if training_interrupted:
                    logging.warning("Evaluation interrupted by user")
                else:
                    logging.error(f"Error during initial evaluation: {e}")

        if training_interrupted:
            logging.warning("Training interrupted")
            return {}

        try:
            # Main training loop
            training_stats = current_trainer.train(
                total_training_steps=total_steps,
                eval_freq_steps=total_steps // 10  # Evaluate 10 times during training
            )

            # Handle training completion
            if training_interrupted:
                logging.warning("⚠️ Training was interrupted by user")
                training_stats = {
                    "interrupted": True,
                    "completed_updates": getattr(current_trainer, 'global_update_counter', 0),
                    "total_planned_updates": total_updates
                }
            else:
                logging.info("🎉 Training completed successfully!")
                training_stats = {
                    "total_episodes": getattr(current_trainer, 'global_episode_counter', 0),
                    "total_steps": getattr(current_trainer, 'global_step_counter', 0),
                    "completed_updates": getattr(current_trainer, 'global_update_counter', 0),
                    "interrupted": False
                }

            return training_stats

        except KeyboardInterrupt:
            logging.warning("⚠️ Training interrupted by user")
            return {"interrupted": True}

    except Exception as e:
        logging.error(f"Critical error: {e}")
        return {"error": str(e)}

    finally:
        cleanup_resources()


def main():
    """Main entry point with error handling"""
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_STRICT_CFG"] = "0"

    # Set UTF-8 encoding for Windows
    if os.name == 'nt':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    try:
        run_training()
    except KeyboardInterrupt:
        console.print("\n[bold red]Training interrupted by user[/bold red]")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error running training: {e}[/bold red]")
        console.print("Trying to continue with default configuration...")
        try:
            from config.config import Config
            config = Config()
            config.data.data_dir = "./dnb/mlgo"
            config.data.symbol_info_file = "./dnb/mlgo/symbols.json"
            run_training(config)
        except Exception as e2:
            console.print(f"[bold red]Failed to run with default config: {e2}[/bold red]")
            sys.exit(1)
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()