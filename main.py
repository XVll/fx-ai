import sys
import os
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import logging
import torch

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from envs.trading_env import TradingEnv
from envs.trading_simulator import TradingSimulator
from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, TensorboardCallback, EarlyStoppingCallback
from agent.wandb_callback import WandbCallback

# Setup logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_training(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object containing all parameters
    """

    print(OmegaConf.to_yaml(cfg))
    # Get output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Initialize data provider and manager
    log.info("Initializing data provider")
    provider = DabentoFileProvider(cfg.data.data_dir)
    data_manager = DataManager(provider, logger=log)

    # 2. Set up simulator
    log.info("Setting up simulator")
    simulator = TradingSimulator(data_manager, cfg.simulation, logger=log)

    # 3. Initialize simulator with data
    log.info(f"Loading data for {cfg.data.symbol} from {cfg.data.start_date} to {cfg.data.end_date}")
    simulator.initialize_for_symbol(
        cfg.data.symbol,
        mode='backtesting',
        start_time=cfg.data.start_date,
        end_time=cfg.data.end_date,
        timeframes=cfg.data.timeframes
    )

    # 4. Create environment
    log.info("Creating trading environment")
    env = TradingEnv(simulator, cfg.env, logger=log)

    # 5. Select device
    if cfg.training.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.training.device)

    log.info(f"Using device: {device}")

    # 6. Create model
    log.info("Creating multi-branch transformer model")
    model = MultiBranchTransformer(**cfg.model, device=device)

    # 7. Set up training callbacks
    callbacks = [
        ModelCheckpointCallback(
            model_dir,
            save_freq=cfg.callbacks.save_freq,
            prefix=cfg.data.symbol
        ),
        TensorboardCallback(log_freq=cfg.callbacks.log_freq)
    ]

    # Add early stopping if enabled
    if cfg.callbacks.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                patience=cfg.callbacks.early_stopping.patience,
                min_delta=cfg.callbacks.early_stopping.min_delta
            )
        )

        # Add W&B callback if enabled - Enhanced integration
    if cfg.wandb.enabled:
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

    # 8. Create trainer
    log.info("Setting up PPO trainer")
    trainer = PPOTrainer(
        env=env,
        model=model,
        model_config=dict(cfg.model),
        # Training params from config
        lr=cfg.training.lr,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        clip_eps=cfg.training.clip_eps,
        n_epochs=cfg.training.n_epochs,
        batch_size=cfg.training.batch_size,
        buffer_size=cfg.training.buffer_size,
        n_episodes_per_update=cfg.training.n_episodes_per_update,
        # Other params
        device=device,
        output_dir=output_dir,
        logger=log,
        callbacks=callbacks
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
    # For quick testing, add default arg
    if len(sys.argv) == 1 and "quick_test" not in sys.argv:
        sys.argv.extend(["quick_test=true"])
    run_training()
