import os
import sys
import threading
import logging
import time
from datetime import datetime
import argparse
from pathlib import Path

import torch
import numpy as np
import wandb

from config.config import Config, TrainingConfig, DataConfig
from utils.logger import console, setup_rich_logging, get_logger
from utils.graceful_shutdown import get_shutdown_manager

# Import utilities
from utils.model_manager import ModelManager

# Import components
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DatabentoFileProvider
from data.scanner.momentum_scanner import MomentumScanner
from envs.trading_environment import TradingEnvironment
from agent.ppo_agent import PPOTrainer
from ai.transformer import MultiBranchTransformer
from agent.callbacks import (
    create_callback_manager as create_callback_manager_base,
    CallbackManager,
)
from agent.base_callbacks import (
    ModelCheckpointCallback,
    EarlyStoppingCallback,
    TrainingCallback,
    MomentumTrackingCallback,
)
from agent.continuous_training_callbacks import ContinuousTrainingCallback

logger = logging.getLogger(__name__)

# Global variables for cleanup management
cleanup_called = False
current_components = {}
_shutdown_manager = None
training_interrupted = False  # Global flag for backward compatibility


def cleanup_resources():
    try:
        if "trainer" in current_components and "model_manager" in current_components:
            trainer = current_components["trainer"]
            model_manager = current_components["model_manager"]
            if hasattr(trainer, "model"):
                try:
                    interrupted_path = (
                        Path(model_manager.base_dir) / "interrupted_model.pt"
                    )
                    saved_path = model_manager.save_checkpoint(
                        trainer.model,
                        trainer.optimizer,
                        trainer.global_step_counter,
                        trainer.global_episode_counter,
                        trainer.global_update_counter,
                        {"interrupted": True},
                        str(interrupted_path),
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
            dbn_cache_size=10,  # Default cache size
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
        scanner_config=config.scanner,
        logger=log,
    )

    # Extract date range from config if available
    date_range = None
    if hasattr(config, 'env') and hasattr(config.env, 'training_manager'):
        training_manager_config = config.env.training_manager
        if hasattr(training_manager_config, 'data_lifecycle'):
            data_lifecycle = training_manager_config.data_lifecycle
            if hasattr(data_lifecycle, 'adaptive_data'):
                date_range = data_lifecycle.adaptive_data.date_range
                if date_range:
                    log.info(f"📅 DataManager configured with date range: {date_range}")

    data_manager = DataManager(
        provider=data_provider, 
        momentum_scanner=momentum_scanner, 
        logger=log,
        date_range=date_range,
        include_weekends=False  # Exclude weekends - trading days only
    )

    return data_manager


# Simulation components are now created inside TradingEnvironment.setup_session()


def create_env_components(
    config: Config, data_manager: DataManager, log: logging.Logger
):
    """Create environment and related components with proper config passing"""
    # Trading Environment with full config
    env = TradingEnvironment(
        config=config,
        data_manager=data_manager,
        logger=log,
        callback_manager=None,  # Will be set later
    )

    return env


def get_feature_names_from_config():
    """Get feature names for each branch based on the feature system"""
    return {
        "hf": [
            "price_velocity",
            "price_acceleration",
            "tape_imbalance",
            "tape_aggression_ratio",
            "spread_compression",
            "quote_velocity",
            "quote_imbalance",
            "volume_velocity",
            "volume_acceleration",
        ],
        "mf": [
            "1m_position_in_current_candle",
            "5m_position_in_current_candle",
            "1m_body_size_relative",
            "5m_body_size_relative",
            "distance_to_ema9_1m",
            "distance_to_ema20_1m",
            "distance_to_ema9_5m",
            "distance_to_ema20_5m",
            "ema_interaction_pattern",
            "ema_crossover_dynamics",
            "ema_trend_alignment",
            "swing_high_distance_1m",
            "swing_low_distance_1m",
            "swing_high_distance_5m",
            "swing_low_distance_5m",
            "price_velocity_1m",
            "price_velocity_5m",
            "volume_velocity_1m",
            "volume_velocity_5m",
            "price_acceleration_1m",
            "price_acceleration_5m",
            "volume_acceleration_1m",
            "volume_acceleration_5m",
            "distance_to_vwap",
            "vwap_slope",
            "price_vwap_divergence",
            "vwap_interaction_dynamics",
            "vwap_breakout_quality",
            "vwap_mean_reversion_tendency",
            "relative_volume",
            "volume_surge",
            "cumulative_volume_delta",
            "volume_momentum",
            "professional_ema_system",
            "professional_vwap_analysis",
            "professional_momentum_quality",
            "professional_volatility_regime",
            "trend_acceleration",
            "volume_pattern_evolution",
            "momentum_quality",
            "pattern_maturation",
            "mf_trend_consistency",
            "mf_volume_price_divergence",
            "mf_momentum_persistence",
            "volatility_adjusted_momentum",
            "regime_relative_volume",
            "1m_position_in_previous_candle",
            "5m_position_in_previous_candle",
            "1m_upper_wick_relative",
            "1m_lower_wick_relative",
            "5m_upper_wick_relative",
            "5m_lower_wick_relative",
        ],
        "lf": [
            "daily_range_position",
            "prev_day_range_position",
            "price_change_from_prev_close",
            "support_distance",
            "resistance_distance",
            "whole_dollar_proximity",
            "half_dollar_proximity",
            "market_session_type",
            "time_of_day_sin",
            "time_of_day_cos",
            "halt_state",
            "time_since_halt",
            "distance_to_luld_up",
            "distance_to_luld_down",
            "luld_band_width",
            "session_progress",
            "market_stress",
            "session_volume_profile",
            "adaptive_support_resistance",
            "hf_momentum_summary",
            "hf_volume_dynamics",
            "hf_microstructure_quality",
        ],
        "portfolio": [
            "position_side",
            "position_size",
            "unrealized_pnl",
            "realized_pnl",
            "total_pnl",
        ],
    }


def get_adaptive_symbols(config):
    """Extract symbols from adaptive data configuration"""
    if hasattr(config, 'env') and hasattr(config.env, 'training_manager'):
        training_manager_config = config.env.training_manager
        if hasattr(training_manager_config, 'data_lifecycle'):
            data_lifecycle = training_manager_config.data_lifecycle
            if hasattr(data_lifecycle, 'adaptive_data'):
                return data_lifecycle.adaptive_data.symbols
    
    # Fallback to default
    return ["MLGO"]


def create_callback_manager(
    config: Config, log: logging.Logger, model: torch.nn.Module = None
) -> CallbackManager:
    """Create callback manager with all configured callbacks"""
    # Convert Config object to dict for callback creation
    config_dict = {
        "wandb": config.wandb.__dict__
        if hasattr(config.wandb, "__dict__")
        else config.wandb,
        "optuna_trial": getattr(config, "optuna_trial", None),
        "callbacks": getattr(config, "callbacks", []),
        "model": model,
        "training": config.training.__dict__
        if hasattr(config.training, "__dict__")
        else config.training,
        "simulation": config.simulation.__dict__
        if hasattr(config.simulation, "__dict__")
        else config.simulation,
        "captum": config.captum.model_dump()
        if hasattr(config, "captum") and config.captum and hasattr(config.captum, "model_dump")
        else getattr(config, "captum", None),
    }

    # Debug captum config
    log.info(f"🔍 DEBUG: Original config.captum: {config.captum}")
    log.info(f"🔍 DEBUG: Original config.captum type: {type(config.captum)}")
    if config.captum:
        log.info(f"🔍 DEBUG: Original config.captum.enabled: {config.captum.enabled}")
    log.info(f"🔍 DEBUG: config_dict['captum']: {config_dict.get('captum')}")
    log.info(f"🔍 DEBUG: config_dict['captum'] type: {type(config_dict.get('captum'))}")

    # Check for Optuna trial info (subprocess mode)
    optuna_trial_info = getattr(config, "optuna_trial_info", None)
    if optuna_trial_info and optuna_trial_info.get("is_optuna_trial", False):
        # This is an Optuna trial running in subprocess mode
        # Set a None trial to trigger subprocess mode in OptunaCallback
        config_dict["optuna_trial"] = None
        config_dict["optuna_trial_info"] = optuna_trial_info
        log.info(
            f"🔍 Detected Optuna trial {optuna_trial_info.get('trial_number', 'unknown')} - running in subprocess mode"
        )


    # Add adaptive symbols for tracking
    adaptive_symbols = get_adaptive_symbols(config)
    config_dict["adaptive_symbols"] = adaptive_symbols
    config_dict["primary_symbol"] = (
        adaptive_symbols[0] if adaptive_symbols else "adaptive"
    )

    # Create callback manager
    callback_manager = create_callback_manager_base(config_dict)

    # Log enabled callbacks
    enabled_callbacks = [
        cb.__class__.__name__ for cb in callback_manager.callbacks if cb.enabled
    ]
    if enabled_callbacks:
        logging.info(
            f"✅ Callback system initialized with: {', '.join(enabled_callbacks)}"
        )
    else:
        logging.info("📊 No callbacks enabled")

    # Note: on_training_start will be triggered by PPO trainer when training actually begins
    return callback_manager


def create_model_components(
    config: Config, device: torch.device, output_dir: str, log: logging.Logger
):
    """Create model and training components with proper config passing"""
    # Model Manager with cache directory structure
    model_manager = ModelManager(
        base_dir="cache/model/checkpoint",
        best_models_dir="cache/model/best",
        model_prefix="model",
        max_best_models=config.training.keep_best_n_models,
    )

    # Model dimensions are known from config, no need to reset env here

    # Create model with ModelConfig
    model = MultiBranchTransformer(model_config=config.model, device=device, logger=log)
    logging.info("✅ Model created successfully")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    return model, optimizer, model_manager


def create_training_callbacks(
    config: Config, model_manager, output_dir: str, loaded_metadata: dict = None
) -> list:
    """Create training callbacks with proper config passing"""
    callbacks: list[TrainingCallback] = []

    # Model checkpoint callback
    callbacks.append(
        ModelCheckpointCallback(
            save_dir=str(Path(output_dir) / "models"),
            save_freq=config.training.checkpoint_interval,
            prefix="model",
            save_best_only=True,
        )
    )

    # Early stopping if configured
    if config.training.early_stop_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                patience=config.training.early_stop_patience,
                min_delta=config.training.early_stop_min_delta,
            )
        )
        logging.info(
            f"⏹️ Early stopping enabled (patience: {config.training.early_stop_patience})"
        )

    # Continuous training callback
    if config.training.continue_training:
        try:
            continuous_callback = ContinuousTrainingCallback(
                model_manager=model_manager,
                reward_metric=config.training.best_model_metric,
                checkpoint_sync_frequency=config.training.checkpoint_interval,
                load_metadata=loaded_metadata or {},
            )
            callbacks.append(continuous_callback)
            logging.info("🔄 Continuous training callback added")
        except Exception as e:
            logging.error(f"Failed to initialize continuous training callback: {e}")
            logging.warning("Continuing without continuous training callback...")

    # Momentum tracking callback for momentum-based training
    if (
        hasattr(config.env, "use_momentum_training")
        and config.env.use_momentum_training
    ):
        momentum_callback = MomentumTrackingCallback(log_frequency=10)
        callbacks.append(momentum_callback)
        logging.info("🎯 Momentum tracking callback added")


    return callbacks


def train(config: Config):
    """Main training function with proper config passing"""
    global current_components

    logger.info("🚀 Starting FX-AI Training System")
    # Get adaptive symbols for logging
    adaptive_symbols = get_adaptive_symbols(config)
    logger.info(f"📈 Symbols: {adaptive_symbols}")



    # Create components with proper config passing
    try:
        # Data components
        data_manager = create_data_components(config, logger)
        env = create_env_components(config, data_manager, logger)

        # For momentum-based training, don't pre-setup environment session
        # The PPO agent will handle day selection and environment setup
        logger.info("🎯 Momentum-based training enabled - PPO agent will manage day selection")

        # Model components (create first to pass to callbacks)
        model, optimizer, model_manager = create_model_components(
            config, device, str(output_dir), logger
        )
        # Callback components
        callback_manager = create_callback_manager(config, logger, model)

        if callback_manager:
            env.callback_manager = callback_manager

        # Load best model if continuing
        loaded_metadata = {}
        if config.training.continue_training:
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                logger.info(f"📂 Loading best model: {best_model_info['path']}")
                model, training_state = model_manager.load_model(
                    model, optimizer, best_model_info["path"]
                )
                loaded_metadata = training_state.get("metadata", {})
                logger.info(
                    f"✅ Model loaded: step={training_state.get('global_step', 0)}"
                )
            else:
                logger.info("🆕 No previous model found. Starting fresh.")
        else:
            # Starting fresh training - don't save initial model with 0 reward
            # Let the continuous training system save models after actual training progress
            logger.info("🆕 Starting fresh training - initial model will be saved after training progress")

        # Create callbacks
        callbacks = create_training_callbacks(
            config, model_manager, str(output_dir), loaded_metadata
        )
        
        # Callbacks created

        # Create trainer with simplified config-based constructor
        trainer = PPOTrainer(
            env=env,
            model=model,
            callback_manager=callback_manager,
            config=config,  # Pass full config - trainer will extract needed parameters
            device=device,
            output_dir=str(output_dir),
            callbacks=callbacks,
        )
        current_components["trainer"] = trainer
        
        # Register trainer for immediate shutdown
        _shutdown_manager.register_component(
            "PPOTrainer",
            lambda: setattr(trainer, 'stop_training', True),
            timeout=5.0,
            critical=True
        )

        # Check for shutdown request
        if _shutdown_manager.is_shutdown_requested():
            logger.warning("Shutdown requested before training start")
            return {"shutdown_requested": True}

        # TrainingManager will handle all data lifecycle initialization
        logger.info("🎯 TrainingManager will handle data lifecycle initialization...")

        # Set primary asset - this will be determined by adaptive data
        adaptive_symbols = get_adaptive_symbols(config)
        primary_symbol = adaptive_symbols[0] if adaptive_symbols else "adaptive"
        env.primary_asset = primary_symbol

        # Note: on_training_start will be triggered by PPO trainer.train() method

        # Main training loop - TrainingManager controlled
        logger.info("🚀 Starting TrainingManager-controlled training")
        logger.info("   Training will complete based on TrainingManager termination criteria")

        try:
            # Use new TrainingManager system
            training_stats = trainer.train_with_manager()

            if _shutdown_manager.is_shutdown_requested():
                logger.warning("⚠️ Training was interrupted by shutdown request")
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

        # Cleanup
        cleanup_resources()

        if config.wandb.enabled:
            wandb.finish()


def main():
    try:
        train(config)
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()
