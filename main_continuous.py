#!/usr/bin/env python
# main_continuous.py
import os
import sys

from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import logging
import torch
import wandb
from datetime import datetime
import argparse

from agent.continous_training_callback import ContinuousTrainingCallback
from ai.transformer import MultiBranchTransformer
# Import custom modules
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
from agent.wandb_callback import WandbCallback
from utils.medel_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"trading_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
log = logging.getLogger("main")


def select_device(device_str):
    """Select device based on config or auto-detect."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            log.info("Using CPU device")
    else:
        device = torch.device(device_str)
        log.info(f"Using specified device: {device}")

    return device


def create_data_provider(config: Config):
    """Create appropriate data provider based on config."""
    data_cfg = config.data
    log.info(f"Initializing data provider: {data_cfg.provider_type}")

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
        # First create historical provider
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
        # For quick testing
        return DummyDataProvider(config={
            'debug_window_mins': 300,  # 5 hours of data
            'data_sparsity': 5,  # Generate data every 5 seconds
            'num_squeezes': 3,  # 3 momentum squeezes
            'base_price': 5.00,  # Start around $5
            'volatility': 0.05,  # Base volatility
            'squeeze_magnitude': 0.30,  # 30% average squeeze magnitude
            'symbols': ['MLGO']  # Symbol to use
        })
    else:
        raise ValueError(f"Unknown provider type: {data_cfg.provider_type}")


def parse_args():
    """Parse command line arguments for easier IDE integration."""
    parser = argparse.ArgumentParser(description="Trading AI Continuous Training")

    # Main operation mode
    parser.add_argument("--mode", type=str, choices=["train", "test", "backtest", "sweep"],
                        default="train", help="Operation mode")

    # Continuous training options
    parser.add_argument("--continue-training", action="store_true",
                        help="Continue training from previous best model")
    parser.add_argument("--best-models-dir", type=str, default="./best_models",
                        help="Directory for storing best models")

    # Symbol and data range
    parser.add_argument("--symbol", type=str, help="Trading symbol")
    parser.add_argument("--start-date", type=str, help="Start date for data")
    parser.add_argument("--end-date", type=str, help="End date for data")

    # Config overrides
    parser.add_argument("--config", type=str, help="Path to custom config YAML")
    parser.add_argument("--quick-test", action="store_true", help="Use quick test settings")

    # Convert to Hydra-compatible args (to be added to sys.argv)
    args, unknown = parser.parse_known_args()

    # Build hydra args from parsed args
    hydra_args = []

    if args.mode == "train" and args.continue_training:
        hydra_args.append("training=continuous")

    if args.quick_test:
        hydra_args.append("quick_test=true")

    if args.symbol:
        hydra_args.append(f"data.symbol={args.symbol}")

    if args.start_date:
        hydra_args.append(f"data.start_date={args.start_date}")

    if args.end_date:
        hydra_args.append(f"data.end_date={args.end_date}")

    if args.best_models_dir:
        hydra_args.append(f"training.best_models_dir={args.best_models_dir}")

    if args.config:
        hydra_args.append(f"--config-path={os.path.dirname(args.config)}")
        hydra_args.append(f"--config-name={os.path.basename(args.config).replace('.yaml', '')}")

    # Return parsed args and hydra-compatible args
    return args, hydra_args


@main(version_base="1.2", config_path="config", config_name="config")
def run_training(cfg: Config):
    """
    Main training function using Hydra configuration.
    Args:
        cfg: Configuration object loaded by Hydra
    Returns:
        dict: Training statistics
    """
    # Convert to our dedicated Config class
    config = cfg

    # Get output directory from Hydra
    output_dir = HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save the config for reference
    config_path = os.path.join(output_dir, "config_used.yaml")
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    log.info(f"Training run initialized in directory: {output_dir}")
    log.info(f"Configuration saved to: {config_path}")

    # Check if we're in continuous training mode
    continuous_mode = getattr(config.training, 'enabled', False)
    load_best_model = getattr(config.training, 'load_best_model', False) if continuous_mode else False
    best_models_dir = getattr(config.training, 'best_models_dir', "./best_models")

    if continuous_mode:
        log.info(f"Continuous training mode enabled. Will use/save best models in: {best_models_dir}")

    # Initialize W&B if enabled
    if config.wandb.enabled:
        try:
            wandb.init(
                project=config.wandb.project_name,
                entity=config.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True),
                save_code=config.wandb.log_code,
                dir=output_dir
            )
            log.info(f"W&B initialized: {wandb.run.name} ({wandb.run.id})")
        except Exception as e:
            log.error(f"Failed to initialize W&B: {e}")
            log.info("Continuing without W&B...")

    try:
        # Create data provider and manager
        log.info("Initializing data provider and manager")
        data_provider = create_data_provider(config)
        data_manager = DataManager(provider=data_provider, logger=log.getChild("DataManager"))

        # Get symbol and date range
        symbol = config.data.symbol

        # Set trading dates (use specific dates for reproducibility)
        start_date = config.data.start_date
        end_date = config.data.end_date

        # Initialize model manager for continuous training support
        model_manager = ModelManager(
            base_dir=model_dir,
            best_models_dir=best_models_dir,
            model_prefix=symbol,
            max_best_models=getattr(config.training, 'max_best_models', 5),
            symbol=symbol
        )

        # Create environment
        log.info(f"Creating trading environment for {symbol} from {start_date} to {end_date}")
        env = TradingEnvironment(
            config=config,
            data_manager=data_manager,
            logger=log.getChild("TradingEnv")
        )

        # Setup environment session
        env.setup_session(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date
        )

        # Wrap environment with observation normalization if required
        if config.env.normalize_state:
            log.info("Wrapping environment with observation normalization")
            env = NormalizeDictObservation(env)

        # Select device for training
        device = select_device(config.training.device)

        # Create model
        log.info("Creating multi-branch transformer model")
        model_config = OmegaConf.to_container(cfg.model, resolve=True)

        # Make an initial reset to get observation shape for sanity check
        obs, info = env.reset()

        try:
            model = MultiBranchTransformer(**model_config, device=device, logger=log.getChild("Transformer"))
            log.info(f"Model created successfully with config: {model_config}")
        except Exception as e:
            log.error(f"Error creating model: {e}")
            raise

        # For continuous training, load the best model if available
        loaded_metadata = {}
        if continuous_mode and load_best_model:
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                log.info(f"Loading best model for continuous training: {best_model_info['path']}")

                # Initialize optimizer before loading
                # Extract training parameters from config for optimizer
                training_params = {
                    "lr": config.training.lr,
                    "max_grad_norm": config.training.max_grad_norm,
                }

                # Create a temporary optimizer for loading
                optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])

                # Load the model weights and state
                model, training_state = model_manager.load_model(model, optimizer, best_model_info['path'])
                loaded_metadata = training_state.get('metadata', {})

                log.info(f"Loaded model from {best_model_info['path']}")
                log.info(f"Training state loaded: "
                         f"step={training_state.get('global_step', 0)}, "
                         f"episode={training_state.get('global_episode', 0)}, "
                         f"update={training_state.get('global_update', 0)}")
            else:
                log.info("No previous model found for continuous training. Starting fresh.")

        # Set up training callbacks
        log.info("Setting up training callbacks")
        callbacks = [
            ModelCheckpointCallback(
                save_dir=model_dir,
                save_freq=config.callbacks.save_freq,
                prefix=symbol,
                save_best_only=config.callbacks.save_best_only,
            ),
        ]

        # Add early stopping if enabled
        if config.callbacks.early_stopping.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    patience=config.callbacks.early_stopping.patience,
                    min_delta=config.callbacks.early_stopping.min_delta
                )
            )

        # Add W&B callback if enabled
        if config.wandb.enabled:
            try:
                wandb_callback = WandbCallback(
                    project_name=config.wandb.project_name,
                    entity=config.wandb.entity,
                    log_freq=config.wandb.log_frequency.steps,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    log_model=config.wandb.log_model,
                    log_code=config.wandb.log_code
                )
                callbacks.append(wandb_callback)
                log.info("W&B callback added successfully")
            except Exception as e:
                log.error(f"Failed to initialize W&B callback: {e}")
                log.info("Continuing without W&B callback...")

        # Add continuous training callback if in continuous mode
        if continuous_mode:
            try:
                # Get continuous training settings
                checkpoint_sync_frequency = getattr(config.training, 'checkpoint_sync_frequency', 5)
                lr_annealing = getattr(config.training, 'lr_annealing', {})

                # Create the callback
                continuous_callback = ContinuousTrainingCallback(
                    model_manager=model_manager,
                    reward_metric="mean_reward",
                    checkpoint_sync_frequency=checkpoint_sync_frequency,
                    lr_annealing=lr_annealing,
                    load_metadata=loaded_metadata
                )
                callbacks.append(continuous_callback)
                log.info("Continuous training callback added")
            except Exception as e:
                log.error(f"Failed to initialize continuous training callback: {e}")
                log.info("Continuing without continuous training callback...")

        # Extract training parameters from config
        training_params = {
            "lr": config.training.lr,
            "gamma": config.training.gamma,
            "gae_lambda": config.training.gae_lambda,
            "clip_eps": config.training.clip_eps,
            "critic_coef": config.training.critic_coef,
            "entropy_coef": config.training.entropy_coef,
            "max_grad_norm": config.training.max_grad_norm,
            "ppo_epochs": config.training.n_epochs,
            "batch_size": config.training.batch_size,
            "rollout_steps": config.training.buffer_size
        }

        log.info(f"Training parameters: {training_params}")

        # Create trainer
        trainer = PPOTrainer(
            env=env,
            model=model,
            model_config=model_config,
            device=device,
            output_dir=output_dir,
            use_wandb=config.wandb.enabled,
            callbacks=callbacks,
            **training_params
        )

        # Train the model
        total_updates = config.training.total_updates
        total_steps = total_updates * config.training.buffer_size
        log.info(f"Starting training for approximately {total_steps} steps "
                 f"({total_updates} updates)")

        # Evaluate the model before training if in continuous mode
        if continuous_mode and getattr(config.training, 'startup_evaluation', False) and load_best_model:
            log.info("Performing initial evaluation...")
            eval_stats = trainer.evaluate(n_episodes=5)
            log.info(f"Initial evaluation results: Mean reward: {eval_stats.get('mean_reward', 'N/A')}")

            # Log to W&B if enabled
            if config.wandb.enabled and wandb.run:
                wandb.log({"initial_eval": eval_stats})

        try:
            training_stats = trainer.train(total_updates)

            # Log final statistics
            log.info("Training completed")
            log.info(f"Total episodes: {training_stats.get('total_episodes', 0)}")
            log.info(f"Total steps: {training_stats.get('total_steps', 0)}")
            log.info(f"Best mean reward: {training_stats.get('best_mean_reward', 0)}")

            if 'best_model_path' in training_stats:
                log.info(f"Best model saved to: {training_stats['best_model_path']}")

            # Evaluate the best model
            log.info("Evaluating best model")
            # Load the best model if available
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                model, _ = model_manager.load_model(model, None, best_model_info['path'])
                log.info(f"Loaded best model for evaluation: {best_model_info['path']}")

            eval_stats = trainer.evaluate(n_episodes=10)
            log.info(f"Evaluation results: Mean reward: {eval_stats.get('mean_reward', 'N/A')}")

            # Log to W&B if enabled
            if config.wandb.enabled and wandb.run:
                wandb.log({"final_eval": eval_stats})

            return training_stats

        except KeyboardInterrupt:
            log.info("Training interrupted by user")
            # Save current model
            interrupted_model_path = os.path.join(model_dir, "interrupted_model.pt")
            trainer.save_model(interrupted_model_path)
            log.info(f"Saved interrupted model to: {interrupted_model_path}")

            # Try to save to best_models as well for continuity
            if continuous_mode:
                try:
                    model_manager.save_best_model(
                        interrupted_model_path,
                        {"interrupted": True, "timestamp": datetime.now().timestamp()},
                        -9999.0  # Low priority for interrupted models
                    )
                except Exception as e:
                    log.warning(f"Failed to save interrupted model to best_models: {e}")

        except Exception as e:
            log.exception(f"Error during training: {e}")
            # Try to save current model
            try:
                error_model_path = os.path.join(model_dir, "error_model.pt")
                trainer.save_model(error_model_path)
                log.info(f"Saved model at error to: {error_model_path}")
            except:
                log.error("Failed to save model after error")

    except Exception as e:
        log.exception(f"Critical error: {e}")

    finally:
        # Clean up
        if 'data_manager' in locals():
            log.info("Closing data manager")
            data_manager.close()

        # Finalize W&B logging if used
        if config.wandb.enabled and wandb.run:
            log.info("Finalizing W&B run")
            wandb.finish()


if __name__ == "__main__":
    # Parse command line arguments
    args, hydra_args = parse_args()

    # Add hydra args to sys.argv
    for arg in hydra_args:
        if arg not in sys.argv:
            sys.argv.append(arg)

    # Set environment variable for detailed error messages
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Run appropriate mode
    if args.mode == "sweep":
        try:
            # Import here to avoid circular imports
            from run_sweep import main as sweep_main

            sweep_main()
        except ImportError:
            log.error("run_sweep.py not found. Cannot run hyperparameter sweep.")
        except Exception as e:
            log.exception(f"Error during sweep: {e}")
    else:
        # Run normal training
        run_training()