# main.py
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wandb

# Import typed config directly
from config.config import Config
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from envs.trading_env import TradingEnv

from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, EarlyStoppingCallback
from agent.wandb_callback import WandbCallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def run_training(cfg: Config):
    """
    Main training function using Hydra configuration.

    Args:
        cfg: Configuration object loaded by Hydra

    Returns:
        dict: Training statistics
    """
    # Get output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Log configuration
    log.info(f"Loaded configuration: {OmegaConf.to_yaml(cfg)}")

    # Create environment
    log.info("Creating trading environment")
    db_provider = DabentoFileProvider(cfg.data.data_dir, "MLGO")
    dm = DataManager(provider=db_provider, logger=log)

    env = TradingEnv(dm, cfg=cfg.env, logger=log)

    # Initialize environment for symbol
    symbol = cfg.data.symbol
    start_date = cfg.data.start_date
    end_date = cfg.data.end_date

    log.info(f"Initializing environment for {symbol} from {start_date} to {end_date}")

    # Determine timeframes to load
    timeframes = cfg.data.timeframes

    # Initialize the environment (this creates all required components)
    success = env.initialize_for_symbol(
        symbol,
        mode='backtesting',
        start_time=start_date,
        end_time=end_date,
        timeframes=timeframes
    )

    if not success:
        log.error(f"Failed to initialize environment for {symbol}")
        return {'error': 'Failed to initialize environment'}

    # Select device based on config or auto-detect
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

    # Create model
    log.info("Creating multi-branch transformer model")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)

    # Ensure model config is compatible with our state representation
    # Get a sample observation to determine state dimensions
    obs, info = env.reset()
    state_dict = env.get_model_state_dict()

    # If observation structure doesn't match model config, adjust model config
    if state_dict is not None and 'hf_features' in state_dict:
        hf_features = state_dict['hf_features']
        mf_features = state_dict['mf_features']
        lf_features = state_dict['lf_features']
        static_features = state_dict['static_features']

        # Update model config with actual dimensions if needed
        if hf_features is not None and 'hf_feat_dim' in model_config:
            if hf_features.shape[-1] != model_config['hf_feat_dim']:
                log.warning(f"Adjusting model hf_feat_dim from {model_config['hf_feat_dim']} "
                            f"to {hf_features.shape[-1]}")
                model_config['hf_feat_dim'] = hf_features.shape[-1]

        # Similarly, update for other features
        # (add more checks as needed based on your model requirements)

    try:
        model = MultiBranchTransformer(**model_config, device=device)
        log.info(f"Model created successfully with config: {model_config}")
    except Exception as e:
        log.error(f"Error creating model: {e}")
        raise

    # Set up training callbacks
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

    # Create trainer
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

    # Train the model
    log.info(f"Starting training for {cfg.training.total_updates} updates")
    training_stats = trainer.train(cfg.training.total_updates)

    # Log final statistics
    log.info("Training completed")
    log.info(f"Total episodes: {training_stats['total_episodes']}")
    log.info(f"Total steps: {training_stats['total_steps']}")
    log.info(f"Best mean reward: {training_stats['best_mean_reward']}")
    log.info(f"Best model saved to: {training_stats['best_model_path']}")

    # Evaluate the best model if requested
    if cfg.get('evaluate_after_training', True):
        log.info("Evaluating best model")
        # Load the best model
        best_model_path = training_stats.get('best_model_path')
        if best_model_path and os.path.exists(best_model_path):
            trainer.load_model(best_model_path)
            eval_stats = trainer.evaluate(n_episodes=10)
            log.info(f"Evaluation results: {eval_stats}")

            # Log to W&B if enabled
            if cfg.wandb.enabled and wandb.run:
                wandb.log({"final_eval": eval_stats})
        else:
            log.warning("Best model not found, skipping evaluation")

    # Finalize W&B logging if used
    if cfg.wandb.enabled and wandb.run:
        wandb.finish()

    return training_stats


def run_sweep():
    """
    Run a hyperparameter sweep using W&B.

    This is a separate entry point that can be called to run a sweep
    instead of a single training run.
    """
    # Import here to avoid circular imports
    from run_sweep import main as sweep_main

    # Run the sweep
    sweep_main()


if __name__ == "__main__":
    # Set environment variable for detailed error messages
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Check for special command line flags
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        # Remove the --sweep flag and run sweep
        sys.argv.pop(1)
        run_sweep()
    else:
        # For quick testing, add default arg
        if len(sys.argv) == 1 and "quick_test" not in sys.argv:
            sys.argv.extend(["quick_test=true"])

        # Run normal training
        run_training()
