"""Optuna training integration module.

This module provides the training function that integrates with the main
application bootstrap and callback system for Optuna hyperparameter optimization.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from callbacks import create_callbacks_from_config
from callbacks.optuna_callback import OptunaCallback
from core.evaluation import Evaluator
from core.model_manager import ModelManager
from data.data_manager import DataManager
from data import DatabentoFileProvider
from envs.trading_environment import TradingEnvironment
from agent.ppo_agent import PPOTrainer
from model.transformer import MultiBranchTransformer
from training.training_manager import TrainingManager, TrainingMode
from core.logger import setup_rich_logging

logger = logging.getLogger(__name__)


def run_optuna_trial(
    config: DictConfig,
    trial: Optional[optuna.Trial] = None,
    metric_name: str = "mean_reward",
) -> float:
    """Run a single training trial for Optuna optimization.
    
    This function initializes all components and runs training with the
    Optuna callback integrated into the callback system.
    
    Args:
        config: Hydra configuration with parameters applied
        trial: Optuna trial object (optional, for standalone runs)
        metric_name: Name of the metric to optimize
        
    Returns:
        The metric value for optimization
        
    Raises:
        optuna.TrialPruned: If the trial should be pruned
    """
    # Set up logging
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    setup_rich_logging(
        level=log_level,
        show_time=config.logging.show_time,
        show_path=config.logging.show_path,
        compact_errors=config.logging.compact_errors,
    )
    
    logger.info(f"🚀 Starting Optuna trial {trial.number if trial else 'standalone'}")
    
    try:
        # Create device
        device = torch.device(
            "cuda" if torch.cuda.is_available() and config.system.device == "cuda" else "cpu"
        )
        
        # Initialize components
        model_manager = ModelManager(config.model_storage)
        
        # Create evaluator
        evaluator = Evaluator(config.evaluation)
        
        # Create callbacks with Optuna callback
        callback_manager = create_callbacks_from_config(
            config=config.callbacks,
            model_manager=model_manager,
            evaluator=evaluator,
            attribution_config=config.attribution,
            optuna_trial=trial,  # Pass the trial for OptunaCallback
        )
        
        # Create data provider and manager
        file_provider = DatabentoFileProvider(config.data.databento_file_provider)
        data_manager = DataManager(
            scanner_config=config.data.scanner,
            provider=file_provider,
            cache_config=config.data.cache,
        )
        
        # Create model
        model = MultiBranchTransformer(
            model_config=config.model,
            action_space_size=config.env.action_space_size,
        ).to(device)
        
        # Create environment
        environment = TradingEnvironment(
            config=config.env,
            reset_callback=data_manager.reset_callback,
            reward_config=config.env.reward,
        )
        
        # Create PPO trainer
        trainer = PPOTrainer(
            model=model,
            config=config.training,
            feature_config=config.features,
            device=device,
        )
        
        # Create training manager
        training_manager = TrainingManager(
            config=config.training.training_manager,
            model_manager=model_manager,
        )
        
        # Set references
        training_manager.data_manager = data_manager
        training_manager.callback_manager = callback_manager
        
        # Override mode to training (not benchmark)
        training_manager.mode = TrainingMode.TRAINING
        
        # Run training
        logger.info("🔄 Starting training loop for Optuna trial")
        training_manager.start(
            trainer=trainer,
            environment=environment,
            data_manager=data_manager,
            callback_manager=callback_manager,
        )
        
        # Extract metric value from the final state
        # The OptunaCallback should have reported values during training
        # Here we get the final value from the training stats
        
        # Try to get from evaluation results first
        if hasattr(training_manager.state, 'evaluation_results') and training_manager.state.evaluation_results:
            last_eval = training_manager.state.evaluation_results[-1]
            if hasattr(last_eval, metric_name):
                metric_value = float(getattr(last_eval, metric_name))
                logger.info(f"✅ Trial completed with {metric_name}={metric_value:.4f}")
                return metric_value
        
        # Fallback to training metrics
        if hasattr(training_manager.state, 'best_mean_reward'):
            metric_value = float(training_manager.state.best_mean_reward)
            logger.warning(f"Using fallback metric best_mean_reward={metric_value:.4f}")
            return metric_value
            
        # If no metric found, return worst value
        logger.error(f"Could not extract metric {metric_name} from training results")
        return float('-inf')
        
    except optuna.TrialPruned:
        logger.info("✂️ Trial pruned by Optuna")
        raise
    except Exception as e:
        logger.error(f"❌ Trial failed with error: {e}", exc_info=True)
        raise
    finally:
        # Clean up Hydra global state for next trial
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()