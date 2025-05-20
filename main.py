#!/usr/bin/env python
# main.py
import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import  OmegaConf
import logging
import torch
import wandb
from datetime import datetime

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

from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, EarlyStoppingCallback
from agent.wandb_callback import WandbCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
            'data_sparsity': 5,  # Every 5th second
            'num_squeezes': 3,  # 3 squeeze events
            'symbols': [data_cfg.symbol]
        })
    else:
        raise ValueError(f"Unknown provider type: {data_cfg.provider_type}")


@hydra.main(version_base="1.2", config_path="config", config_name="config")
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
        start_date = config.data.start_date
        end_date = config.data.end_date

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

        # Set up training callbacks
        log.info("Setting up training callbacks")
        callbacks = [
            ModelCheckpointCallback(
                save_dir=model_dir,
                save_freq=config.callbacks.save_freq,
                prefix=config.data.symbol,
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

        # Create trainer
        log.info("Setting up PPO trainer")

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

        global_best_model_dir = os.path.join(os.getcwd(), config.training.best_model_path)
        global_best_model_path = os.path.join(global_best_model_dir, f"{config.data.symbol}_best_model.pt")

        # Train the model
        total_steps = config.training.total_updates * config.training.buffer_size
        log.info(f"Starting training for approximately {total_steps} steps "
                 f"({config.training.total_updates} updates)")

        try:
            training_stats = trainer.train(config.training.total_updates)

            # Log final statistics
            log.info("Training completed")
            log.info(f"Total episodes: {training_stats.get('total_episodes', 0)}")
            log.info(f"Total steps: {training_stats.get('total_steps', 0)}")
            log.info(f"Best mean reward: {training_stats.get('best_mean_reward', 0)}")

            if 'best_model_path' in training_stats:
                log.info(f"Best model saved to: {training_stats['best_model_path']}")

            # Evaluate the best model if requested
            if config.get('evaluate_after_training', True):
                log.info("Evaluating best model")
                # Load the best model
                best_model_path = training_stats.get('best_model_path')
                if best_model_path and os.path.exists(best_model_path):
                    trainer.load_model(best_model_path)
                    eval_stats = trainer.evaluate(n_episodes=10)
                    log.info(f"Evaluation results: {eval_stats}")

                    # Log to W&B if enabled
                    if config.wandb.enabled and wandb.run:
                        wandb.log({"final_eval": eval_stats})
                else:
                    log.warning("Best model not found, skipping evaluation")

            return training_stats

        except KeyboardInterrupt:
            log.info("Training interrupted by user")
            # Save current model
            interrupted_model_path = os.path.join(model_dir, "interrupted_model.pt")
            trainer.save_model(interrupted_model_path)
            log.info(f"Saved interrupted model to: {interrupted_model_path}")

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


def run_sweep():
    """
    Run a hyperparameter sweep using W&B.
    This is a separate entry point that can be called to run a sweep
    instead of a single training run.
    """
    try:
        # Import here to avoid circular imports
        from run_sweep import main as sweep_main

        # Run the sweep
        sweep_main()
    except ImportError:
        log.error("run_sweep.py not found. Cannot run hyperparameter sweep.")
    except Exception as e:
        log.exception(f"Error during sweep: {e}")


if __name__ == "__main__":
    # Set environment variable for detailed error messages
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Check for special command line flags
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        # Remove the --sweep flag and run sweep
        sys.argv.pop(1)
        run_sweep()
    else:
        # For quick testing, add default.yaml arg if not present
        if len(sys.argv) == 1 and "quick_test" not in "".join(sys.argv):
            sys.argv.extend(["quick_test=true"])

        # Run normal training
        run_training()
