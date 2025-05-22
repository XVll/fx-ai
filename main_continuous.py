#!/usr/bin/env python
# main_continuous.py - Updated with signal handling for Ctrl+C support
import os
import sys
import signal
import threading
from datetime import datetime

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Import centralized logging
from utils.logger import initialize_logger, get_logger, log_info, log_error, log_warning

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

# Global variables for signal handling
training_interrupted = False
cleanup_called = False
current_trainer = None
current_env = None
current_data_manager = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global training_interrupted, cleanup_called

    if cleanup_called:
        print("\nForce exit...")
        sys.exit(1)

    training_interrupted = True
    print("\n" + "=" * 50)
    print("INTERRUPT SIGNAL RECEIVED")
    print("Attempting graceful shutdown...")
    print("Press Ctrl+C again to force exit")
    print("=" * 50)

    # Set a timer for forced exit if graceful shutdown takes too long
    def force_exit():
        import time
        time.sleep(10)  # Wait 10 seconds for graceful shutdown
        if training_interrupted and not cleanup_called:
            print("\nGraceful shutdown timeout. Force exiting...")
            sys.exit(1)

    timer = threading.Timer(10, force_exit)
    timer.daemon = True
    timer.start()


def cleanup_resources():
    """Clean up resources gracefully"""
    global cleanup_called, current_trainer, current_env, current_data_manager

    if cleanup_called:
        return
    cleanup_called = True

    try:
        log_info("[CLEANUP] Starting resource cleanup...", "cleanup")

        # Stop trainer
        if current_trainer and hasattr(current_trainer, 'model'):
            try:
                # Save current model
                interrupted_model_path = os.path.join(
                    getattr(current_trainer, 'model_dir', './models'),
                    "interrupted_model.pt"
                )
                current_trainer.save_model(interrupted_model_path)
                log_info(f"[CLEANUP] Saved interrupted model to: {interrupted_model_path}", "cleanup")
            except Exception as e:
                log_error(f"[CLEANUP] Error saving interrupted model: {e}", "cleanup")

        # Close environment dashboard
        if current_env and hasattr(current_env, 'dashboard') and current_env.dashboard:
            try:
                current_env.dashboard.stop()
                log_info("[CLEANUP] Dashboard stopped", "cleanup")
            except Exception as e:
                log_error(f"[CLEANUP] Error stopping dashboard: {e}", "cleanup")

        # Close environment
        if current_env and hasattr(current_env, 'close'):
            try:
                current_env.close()
                log_info("[CLEANUP] Environment closed", "cleanup")
            except Exception as e:
                log_error(f"[CLEANUP] Error closing environment: {e}", "cleanup")

        # Close data manager
        if current_data_manager and hasattr(current_data_manager, 'close'):
            try:
                current_data_manager.close()
                log_info("[CLEANUP] Data manager closed", "cleanup")
            except Exception as e:
                log_error(f"[CLEANUP] Error closing data manager: {e}", "cleanup")

        # Finalize W&B if active
        if wandb.run:
            try:
                wandb.finish()
                log_info("[CLEANUP] W&B run finished", "cleanup")
            except Exception as e:
                log_error(f"[CLEANUP] Error finishing W&B: {e}", "cleanup")

        log_info("[CLEANUP] Resource cleanup completed", "cleanup")

    except Exception as e:
        print(f"Error during cleanup: {e}")


def select_device(device_str):
    """Select device based on config or auto-detect."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log_info(f"Using CUDA device: {torch.cuda.get_device_name(0)}", "device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log_info("Using MPS (Apple Silicon) device", "device")
        else:
            device = torch.device("cpu")
            log_info("Using CPU device", "device")
    else:
        device = torch.device(device_str)
        log_info(f"Using specified device: {device}", "device")

    return device


def create_data_provider(config: Config):
    """Create appropriate data provider based on config."""
    data_cfg = config.data
    log_info(f"Initializing data provider: {data_cfg.provider_type}", "data")

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
    Main training function with centralized logging and signal handling.

    Args:
        cfg: Config object from Hydra

    Returns:
        Dict[str, Any]: Training statistics
    """
    global current_trainer, current_env, current_data_manager

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Get output directory - either from Hydra or create a new one
        if HydraConfig.initialized():
            output_dir = HydraConfig.get().runtime.output_dir
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join("outputs", timestamp)
            os.makedirs(output_dir, exist_ok=True)

        model_dir = os.path.join(output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # Initialize centralized logging FIRST
        log_file = os.path.join(output_dir, "training.log")
        logger_manager = initialize_logger(
            app_name="fx-ai",
            log_file=log_file,
            max_dashboard_logs=1000
        )

        # Log startup
        log_info("[START] Starting FX-AI Training System", "main")
        log_info(f"[DIR] Output directory: {output_dir}", "main")
        log_info(f"[FILE] Log file: {log_file}", "main")

        # Save the config for reference
        config_path = os.path.join(output_dir, "config_used.yaml")
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        log_info(f"[SAVE] Configuration saved to: {config_path}", "main")

        # Check if we're in continuous training mode
        continuous_mode = getattr(cfg.training, 'enabled', False)
        load_best_model = getattr(cfg.training, 'load_best_model', False) if continuous_mode else False
        best_models_dir = getattr(cfg.training, 'best_models_dir', "./best_models")

        # Create best_models_dir if it doesn't exist
        os.makedirs(best_models_dir, exist_ok=True)

        if continuous_mode:
            log_info(f"[REFRESH] Continuous training mode enabled. Best models dir: {best_models_dir}", "training")

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
                log_info(f"[CHART] W&B initialized: {wandb.run.name} ({wandb.run.id})", "wandb")
            except Exception as e:
                log_error(f"Failed to initialize W&B: {e}", "wandb")
                log_warning("Continuing without W&B...", "wandb")

        # Create data provider and manager
        log_info("[LOAD] Initializing data provider and manager", "data")
        data_provider = create_data_provider(cfg)
        current_data_manager = DataManager(provider=data_provider, logger=get_logger().get_logger("DataManager"))

        # Get symbol and date range
        symbol = cfg.data.symbol
        start_date = cfg.data.start_date
        end_date = cfg.data.end_date

        log_info(f"[UP] Trading symbol: {symbol}", "config")
        log_info(f"[DATE] Date range: {start_date} to {end_date}", "config")

        # Initialize model manager for continuous training support
        model_manager = ModelManager(
            base_dir=model_dir,
            best_models_dir=best_models_dir,
            model_prefix=symbol,
            max_best_models=getattr(cfg.training, 'max_best_models', 5),
            symbol=symbol
        )

        # Create environment with centralized logging
        log_info(f"[BUILD] Creating trading environment for {symbol}", "env")
        current_env = TradingEnvironment(
            config=cfg,
            data_manager=current_data_manager,
            logger=get_logger().get_logger("TradingEnv")
        )

        # Setup environment session
        current_env.setup_session(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date
        )

        # Wrap environment with observation normalization if required
        if cfg.env.normalize_state:
            log_info("[REFRESH] Wrapping environment with observation normalization", "env")
            current_env = NormalizeDictObservation(current_env)

        # Check for interruption
        if training_interrupted:
            log_warning("Training interrupted during setup", "main")
            return {}

        # Select device for training
        device = select_device(cfg.training.device)

        # Create model
        log_info("[BRAIN] Creating multi-branch transformer model", "model")
        model_config = OmegaConf.to_container(cfg.model, resolve=True)

        # Make an initial reset to get observation shape for sanity check
        obs, info = current_env.reset()

        try:
            model = MultiBranchTransformer(
                **model_config,
                device=device,
                logger=get_logger().get_logger("Transformer")
            )
            log_info("[OK] Model created successfully", "model")
        except Exception as e:
            log_error(f"Error creating model: {e}", "model")
            raise

        # For continuous training, load the best model if available
        loaded_metadata = {}
        if continuous_mode and load_best_model:
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                log_info(f"[LOAD] Loading best model for continuous training: {best_model_info['path']}", "training")

                # Initialize optimizer before loading
                training_params = {
                    "lr": cfg.training.lr,
                    "max_grad_norm": cfg.training.max_grad_norm,
                }

                # Create a temporary optimizer for loading
                optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])

                # Load the model weights and state
                model, training_state = model_manager.load_model(model, optimizer, best_model_info['path'])
                loaded_metadata = training_state.get('metadata', {})

                log_info("[OK] Model loaded successfully", "training")
                log_info(f"[CHART] Training state: step={training_state.get('global_step', 0)}, "
                         f"episode={training_state.get('global_episode', 0)}, "
                         f"update={training_state.get('global_update', 0)}", "training")
            else:
                log_info("[START] No previous model found for continuous training. Starting fresh.", "training")

        # Set up training callbacks
        log_info("[CONFIG] Setting up training callbacks", "training")
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
            log_info(f"[PAUSE] Early stopping enabled (patience: {cfg.callbacks.early_stopping.patience})", "training")

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
                log_info("[UP] W&B callback added successfully", "wandb")
            except Exception as e:
                log_error(f"Failed to initialize W&B callback: {e}", "wandb")
                log_warning("Continuing without W&B callback...", "wandb")

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
                log_info("[REFRESH] Continuous training callback added", "training")
            except Exception as e:
                log_error(f"Failed to initialize continuous training callback: {e}", "training")
                log_warning("Continuing without continuous training callback...", "training")

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

        log_info("[CONFIG] Training parameters configured", "training")

        # Create trainer with centralized logging
        current_trainer = PPOTrainer(
            env=current_env,
            model=model,
            model_config=model_config,
            device=device,
            output_dir=output_dir,
            use_wandb=cfg.wandb.enabled,
            callbacks=callbacks,
            **training_params
        )

        # Check for interruption before training
        if training_interrupted:
            log_warning("Training interrupted before starting", "main")
            return {}

        # Train the model
        total_updates = cfg.training.total_updates
        total_steps = total_updates * cfg.training.buffer_size
        log_info(f"[RUN] Starting training for approximately {total_steps} steps "
                 f"({total_updates} updates)", "training")

        # Evaluate the model before training if in continuous mode
        if continuous_mode and getattr(cfg.training, 'startup_evaluation', False) and load_best_model:
            if training_interrupted:
                log_warning("Training interrupted before startup evaluation", "main")
                return {}

            log_info("[SEARCH] Performing initial evaluation...", "eval")
            try:
                eval_stats = current_trainer.evaluate(n_episodes=5)
                log_info(f"[CHART] Initial evaluation results: Mean reward: {eval_stats.get('mean_reward', 'N/A')}",
                         "eval")

                # Log to W&B if enabled
                if cfg.wandb.enabled and wandb.run:
                    wandb.log({"initial_eval": eval_stats})
            except Exception as e:
                if training_interrupted:
                    log_warning("Evaluation interrupted by user", "eval")
                else:
                    log_error(f"Error during initial evaluation: {e}", "eval")

        if training_interrupted:
            log_warning("Training interrupted", "main")
            return {}

        try:
            # Modified training loop with interruption checks
            training_stats = {}
            update_counter = 0

            while update_counter < total_updates and not training_interrupted:
                try:
                    # Collect rollout data
                    if training_interrupted:
                        break

                    rollout_stats = current_trainer.collect_rollout_data()

                    if training_interrupted:
                        break

                    # Update policy
                    update_metrics = current_trainer.update_policy()

                    # Call callbacks
                    for callback in current_trainer.callbacks:
                        if hasattr(callback, 'on_update_iteration_end'):
                            callback.on_update_iteration_end(current_trainer, update_counter, update_metrics,
                                                             rollout_stats)

                    update_counter += 1

                    # Log progress
                    if update_counter % 5 == 0:
                        log_info(f"[UP] Completed {update_counter}/{total_updates} updates. "
                                 f"Mean reward: {rollout_stats.get('mean_reward', 0):.2f}", "training")

                except KeyboardInterrupt:
                    log_warning("Training interrupted by KeyboardInterrupt", "training")
                    break
                except Exception as e:
                    if training_interrupted:
                        log_warning("Training interrupted during update", "training")
                        break
                    else:
                        log_error(f"Error during training update {update_counter}: {e}", "training")
                        # Continue with next update rather than stopping entirely
                        continue

            # Handle training completion vs interruption
            if training_interrupted:
                log_warning("[WARN] Training was interrupted by user", "training")
                training_stats = {
                    "interrupted": True,
                    "completed_updates": update_counter,
                    "total_planned_updates": total_updates
                }
            else:
                log_info("[PARTY] Training completed successfully!", "training")
                training_stats = {
                    "total_episodes": getattr(current_trainer, 'global_episode_counter', 0),
                    "total_steps": getattr(current_trainer, 'global_step_counter', 0),
                    "completed_updates": update_counter,
                    "interrupted": False
                }

            # Final model evaluation (only if not interrupted)
            if not training_interrupted:
                log_info("[SEARCH] Evaluating best model", "eval")
                # Load the best model if available
                best_model_info = model_manager.find_best_model()
                if best_model_info:
                    model, _ = model_manager.load_model(model, None, best_model_info['path'])
                    log_info(f"[LOAD] Loaded best model for evaluation: {best_model_info['path']}", "eval")

                try:
                    eval_stats = current_trainer.evaluate(n_episodes=10)
                    log_info(f"[UP] Final evaluation results: Mean reward: {eval_stats.get('mean_reward', 'N/A')}",
                             "eval")

                    # Log to W&B if enabled
                    if cfg.wandb.enabled and wandb.run:
                        wandb.log({"final_eval": eval_stats})
                except Exception as e:
                    log_error(f"Error during final evaluation: {e}", "eval")

            return training_stats

        except KeyboardInterrupt:
            log_warning("[WARN] Training interrupted by user", "training")
            return {"interrupted": True}

    except Exception as e:
        log_error(f"Critical error: {e}", "main")
        return {"error": str(e)}

    finally:
        # Always clean up resources
        cleanup_resources()


def main():
    """Main entry point with error handling"""
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_STRICT_CFG"] = "0"  # Allow fields not in struct

    # Set UTF-8 encoding for Windows
    if os.name == 'nt':  # Windows
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    try:
        run_training()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        # Fallback logging if centralized logger fails
        print(f"Error running training: {e}")
        print("Trying to continue with default configuration...")
        try:
            from config.config import Config
            config = Config()
            # Override with hard-coded values instead of Hydra interpolation
            config.data.data_dir = "./dnb/mlgo"
            config.data.symbol_info_file = "./dnb/mlgo/symbols.json"
            run_training(config)
        except Exception as e2:
            print(f"Failed to run with default config: {e2}")
            sys.exit(1)
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()