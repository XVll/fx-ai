#!/usr/bin/env python
# main_continuous.py - Main entry point for training AI models with continuous training support
import os
import sys
import logging

import hydra
import torch
import wandb
from datetime import datetime
from typing import Dict

from hydra.core.hydra_config import HydraConfig
from hydra import initialize, compose, initialize_config_module
from omegaconf import OmegaConf, DictConfig

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
from agent.wandb_callback import WandbCallback
from utils.model_manager import ModelManager

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
            'debug_window_mins': 120,  # 5 hours of data
            'data_sparsity': 1,  # Generate data every 5 seconds
            'num_squeezes': 2,  # 3 momentum squeezes
            'base_price': 5.00,  # Start around $5
            'volatility': 0.03,  # Base volatility
            'squeeze_magnitude': 0.30,  # 30% average squeeze magnitude
            'symbols': ['MLGO']  # Symbol to use
        })
    else:
        raise ValueError(f"Unknown provider type: {data_cfg.provider_type}")

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def run_training(cfg: Config):
    """
    Main training function.

    Args:
        cfg: Optional Config object. If None, config will be loaded using Hydra.

    Returns:
        Dict[str, Any]: Training statistics
    """
    # Get output directory - either from Hydra or create a new one
    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("outputs", timestamp)
        os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save the config for reference
    config_path = os.path.join(output_dir, "config_used.yaml")
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    log.info(f"Training run initialized in directory: {output_dir}")
    log.info(f"Configuration saved to: {config_path}")

    # Check if we're in continuous training mode
    continuous_mode = getattr(cfg.training, 'enabled', False)
    load_best_model = getattr(cfg.training, 'load_best_model', False) if continuous_mode else False
    best_models_dir = getattr(cfg.training, 'best_models_dir', "./best_models")

    # Create best_models_dir if it doesn't exist
    os.makedirs(best_models_dir, exist_ok=True)

    if continuous_mode:
        log.info(f"Continuous training mode enabled. Will use/save best models in: {best_models_dir}")

    # Initialize W&B if enabled
    if cfg.wandb.enabled:
        try:
            wandb.init(
                project=cfg.wandb.project_name,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True),
                save_code=cfg.wandb.log_code,
                dir=output_dir
            )
            log.info(f"W&B initialized: {wandb.run.name} ({wandb.run.id})")
        except Exception as e:
            log.error(f"Failed to initialize W&B: {e}")
            log.info("Continuing without W&B...")

    try:
        # Create data provider and manager
        log.info("Initializing data provider and manager")
        data_provider = create_data_provider(cfg)
        data_manager = DataManager(provider=data_provider, logger=log.getChild("DataManager"))

        # Get symbol and date range
        symbol = cfg.data.symbol

        # Set trading dates (use specific dates for reproducibility)
        start_date = cfg.data.start_date
        end_date = cfg.data.end_date

        # Initialize model manager for continuous training support
        model_manager = ModelManager(
            base_dir=model_dir,
            best_models_dir=best_models_dir,
            model_prefix=symbol,
            max_best_models=getattr(cfg.training, 'max_best_models', 5),
            symbol=symbol
        )

        # Create environment
        log.info(f"Creating trading environment for {symbol} from {start_date} to {end_date}")
        env = TradingEnvironment(
            config=cfg,
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
        if cfg.env.normalize_state:
            log.info("Wrapping environment with observation normalization")
            env = NormalizeDictObservation(env)

        # Select device for training
        device = select_device(cfg.training.device)

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
                    "lr": cfg.training.lr,
                    "max_grad_norm": cfg.training.max_grad_norm,
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

        # Add W&B callback if enabled
        if cfg.wandb.enabled:
            try:
                wandb_callback = WandbCallback(
                    project_name=cfg.wandb.project_name,
                    entity=cfg.wandb.entity,
                    log_freq=cfg.wandb.log_frequency.steps,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    log_model=cfg.wandb.log_model,
                    log_code=cfg.wandb.log_code
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
                checkpoint_sync_frequency = getattr(cfg.training, 'checkpoint_sync_frequency', 5)
                lr_annealing = getattr(cfg.training, 'lr_annealing', {})

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

        log.info(f"Training parameters: {training_params}")

        # Create trainer
        trainer = PPOTrainer(
            env=env,
            model=model,
            model_config=model_config,
            device=device,
            output_dir=output_dir,
            use_wandb=cfg.wandb.enabled,
            callbacks=callbacks,
            **training_params
        )

        # Train the model
        total_updates = cfg.training.total_updates
        total_steps = total_updates * cfg.training.buffer_size
        log.info(f"Starting training for approximately {total_steps} steps "
                 f"({total_updates} updates)")

        # Evaluate the model before training if in continuous mode
        if continuous_mode and getattr(cfg.training, 'startup_evaluation', False) and load_best_model:
            log.info("Performing initial evaluation...")
            eval_stats = trainer.evaluate(n_episodes=5)
            log.info(f"Initial evaluation results: Mean reward: {eval_stats.get('mean_reward', 'N/A')}")

            # Log to W&B if enabled
            if cfg.wandb.enabled and wandb.run:
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
            if cfg.wandb.enabled and wandb.run:
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
        if cfg.wandb.enabled and wandb.run:
            log.info("Finalizing W&B run")
            wandb.finish()

def main():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_STRICT_CFG"] = "0"  # Allow fields not in struct

    try:
        run_training()
    except Exception as e:
        log.exception(f"Error initializing Hydra: {e}")
        # Provide a fallback option with default paths
        log.info("Trying to continue with default configuration...")
        from config.config import Config
        config = Config()
        # Override with hard-coded values instead of Hydra interpolation
        config.data.data_dir = "./dnb/mlgo"
        config.data.symbol_info_file = "./dnb/mlgo/symbols.json"
        run_training(config)


if __name__ == "__main__":
    main()