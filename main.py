import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import typed config directly
from config.config import Config, EnvConfig, DataConfig, SimulationConfig

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from envs.trading_env import TradingEnv
from envs.trading_simulator import TradingSimulator
from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, EarlyStoppingCallback
from agent.wandb_callback import WandbCallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def run_training(cfg: Config):
    # Get output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Initialize data provider and manager
    log.info("Initializing data provider")
    data_dir = cfg.data.data_dir
    log.info(f"Data directory: {data_dir}")

    provider = DabentoFileProvider(data_dir)
    data_manager = DataManager(provider, logger=log)

    # 2. Set up simulator
    log.info("Setting up simulator")
    simulator = TradingSimulator(data_manager, cfg.simulation, logger=log)

    # 3. Initialize simulator with data
    symbol = cfg.data.symbol
    start_date = cfg.data.start_date
    end_date = cfg.data.end_date
    timeframes = cfg.data.timeframes

    log.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    log.info(f"Using timeframes: {timeframes}")

    success = simulator.initialize_for_symbol(
        symbol,
        mode='backtesting',
        start_time=start_date,
        end_time=end_date,
        timeframes=timeframes
    )

    if not success:
        log.error(f"Failed to initialize simulator for {symbol}")
        return {'error': 'Failed to initialize simulator'}

    # 4. Create environment
    log.info("Creating trading environment")
    # Convert DictConfig to our strongly-typed EnvConfig
    env_config = OmegaConf.to_object(cfg.env)
    env = TradingEnv(simulator, env_config, logger=log)

    # 5. Select device
    if cfg.training.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            log.info("Using CPU device")
    else:
        device = torch.device(cfg.training.device)
        log.info(f"Using specified device: {device}")

    # 6. Create model
    log.info("Creating multi-branch transformer model")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    log.info(f"Model config: {model_config}")

    try:
        model = MultiBranchTransformer(**model_config, device=device)
        log.info(f"Model created successfully")
    except Exception as e:
        log.error(f"Error creating model: {e}")
        raise

    # 7. Set up training callbacks
    callbacks = [
        ModelCheckpointCallback(
            model_dir,
            save_freq=cfg.callbacks.save_freq,
            prefix=cfg.data.symbol
        ),
    ]

    # Add early stopping if enabled
    if hasattr(cfg.callbacks, "early_stopping") and cfg.callbacks.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                patience=cfg.callbacks.early_stopping.patience,
                min_delta=cfg.callbacks.early_stopping.min_delta
            )
        )

    # Add W&B callback if enabled
    if cfg.wandb.enabled:
        try:
            # Prepare combined config for W&B
            flat_config = OmegaConf.to_container(cfg, resolve=True)

            wandb_callback = WandbCallback(
                project_name=cfg.wandb.project_name,
                entity=cfg.wandb.entity,
                log_freq=cfg.wandb.log_frequency.steps,
                config=flat_config,
                log_model=cfg.wandb.log_model,
                log_code=cfg.wandb.log_code
            )
            callbacks.append(wandb_callback)
            log.info("W&B callback added successfully")
        except Exception as e:
            log.error(f"Failed to initialize W&B: {e}")
            log.info("Continuing without W&B...")

    # 8. Create trainer
    log.info("Setting up PPO trainer")

    # Extract training parameters
    training_params = {
        "lr": cfg.training.lr,
        "gamma": cfg.training.gamma,
        "gae_lambda": cfg.training.gae_lambda,
        "clip_eps": cfg.training.clip_eps,
        "critic_coef": cfg.training.critic_coef,
        "entropy_coef": cfg.training.entropy_coef,
        "max_grad_norm": cfg.training.max_grad_norm,
        "n_epochs": cfg.training.n_epochs,
        "batch_size": cfg.training.batch_size,
        "buffer_size": cfg.training.buffer_size,
        "n_episodes_per_update": cfg.training.n_episodes_per_update,
    }

    log.info(f"Training parameters: {training_params}")

    trainer = PPOTrainer(
        env=env,
        model=model,
        model_config=model_config,
        device=device,
        output_dir=output_dir,
        logger=log,
        callbacks=callbacks,
        **training_params
    )

    # 9. Train the model
    log.info(f"Starting training for {cfg.training.total_updates} updates")
    training_stats = trainer.train(cfg.training.total_updates)

    # 10. Log final statistics
    log.info("Training completed")
    log.info(f"Total episodes: {training_stats['total_episodes']}")
    log.info(f"Total steps: {training_stats['total_steps']}")
    log.info(f"Best mean reward: {training_stats['best_mean_reward']}")
    log.info(f"Best model saved to: {training_stats['best_model_path']}")

    return training_stats


if __name__ == "__main__":
    # Set environment variable for detailed error messages
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # For quick testing, add default arg
    if len(sys.argv) == 1 and "quick_test" not in sys.argv:
        sys.argv.extend(["quick_test=true"])
    run_training()
