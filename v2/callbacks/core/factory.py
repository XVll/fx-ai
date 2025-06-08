"""
Callback factory for creating callbacks from configuration.

Provides a clean interface for creating and configuring callbacks
based on configuration dictionaries.
"""

from typing import Dict, Any, List, Optional
import logging
from .base import BaseCallback
from .manager import CallbackManager


def create_callbacks_from_config(
    config: Dict[str, Any],
    trainer: Optional[Any] = None,
    environment: Optional[Any] = None,
    output_path: Optional[str] = None
) -> CallbackManager:
    """
    Create callback manager with callbacks from configuration.
    
    Args:
        config: Configuration dictionary with callback settings
        trainer: PPO trainer instance (for callbacks that need model access)
        environment: Trading environment (for callbacks that need env access)
        output_path: Output directory path
        
    Returns:
        Configured CallbackManager instance
        
    Example config structure:
    {
        "callbacks": {
            "wandb": {
                "enabled": true,
                "project": "fxai-v2",
                "tags": ["momentum", "ppo"]
            },
            "checkpoint": {
                "enabled": true,
                "save_freq": 100,
                "keep_best": 5
            },
            "metrics": {
                "enabled": true,
                "log_freq": 10
            }
        }
    }
    """
    logger = logging.getLogger("callback_factory")
    callbacks = []
    
    callback_config = config.get("callbacks", {})
    if not callback_config:
        logger.warning("No callback configuration found")
        return CallbackManager(callbacks)
    
    # Create core callbacks
    if "metrics" in callback_config:
        metrics_callback = _create_metrics_callback(
            callback_config["metrics"], trainer, environment
        )
        if metrics_callback:
            callbacks.append(metrics_callback)
    
    if "checkpoint" in callback_config:
        checkpoint_callback = _create_checkpoint_callback(
            callback_config["checkpoint"], trainer, output_path
        )
        if checkpoint_callback:
            callbacks.append(checkpoint_callback)
    
    if "wandb" in callback_config:
        wandb_callback = _create_wandb_callback(
            callback_config["wandb"], trainer, environment
        )
        if wandb_callback:
            callbacks.append(wandb_callback)
    
    # Create analysis callbacks
    if "attribution" in callback_config:
        attribution_callback = _create_attribution_callback(
            callback_config["attribution"], trainer, environment
        )
        if attribution_callback:
            callbacks.append(attribution_callback)
    
    if "performance" in callback_config:
        performance_callback = _create_performance_callback(
            callback_config["performance"], trainer, environment
        )
        if performance_callback:
            callbacks.append(performance_callback)
    
    # Create optimization callbacks
    if "optuna" in callback_config:
        optuna_callback = _create_optuna_callback(
            callback_config["optuna"], trainer
        )
        if optuna_callback:
            callbacks.append(optuna_callback)
    
    if "early_stopping" in callback_config:
        early_stopping_callback = _create_early_stopping_callback(
            callback_config["early_stopping"], trainer
        )
        if early_stopping_callback:
            callbacks.append(early_stopping_callback)
    
    logger.info(f"Created {len(callbacks)} callbacks from configuration")
    return CallbackManager(callbacks)


def _create_metrics_callback(
    config: Dict[str, Any], 
    trainer: Optional[Any], 
    environment: Optional[Any]
) -> Optional[BaseCallback]:
    """Create metrics callback from config."""
    if not config.get("enabled", True):
        return None
    
    try:
        from ..core.metrics_callback import MetricsCallback
        return MetricsCallback(
            enabled=config.get("enabled", True),
            log_freq=config.get("log_freq", 10),
            console_output=config.get("console_output", True)
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create MetricsCallback: {e}")
        return None


def _create_checkpoint_callback(
    config: Dict[str, Any],
    trainer: Optional[Any],
    output_path: Optional[str]
) -> Optional[BaseCallback]:
    """Create checkpoint callback from config."""
    if not config.get("enabled", True):
        return None
    
    try:
        from ..core.checkpoint_callback import CheckpointCallback
        return CheckpointCallback(
            enabled=config.get("enabled", True),
            save_freq=config.get("save_freq", 100),
            keep_best=config.get("keep_best", 5),
            output_path=output_path,
            trainer=trainer
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create CheckpointCallback: {e}")
        return None


def _create_wandb_callback(
    config: Dict[str, Any],
    trainer: Optional[Any],
    environment: Optional[Any]
) -> Optional[BaseCallback]:
    """Create wandb callback from config."""
    if not config.get("enabled", True):
        return None
    
    try:
        from ..core.wandb_callback import WandBCallback
        return WandBCallback(
            enabled=config.get("enabled", True),
            project=config.get("project", "fxai-v2"),
            entity=config.get("entity"),
            tags=config.get("tags", []),
            log_freq=config.get("log_freq", 10),
            log_gradients=config.get("log_gradients", False),
            log_parameters=config.get("log_parameters", True)
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create WandBCallback: {e}")
        return None


def _create_attribution_callback(
    config: Dict[str, Any],
    trainer: Optional[Any],
    environment: Optional[Any]
) -> Optional[BaseCallback]:
    """Create attribution callback from config.""" 
    if not config.get("enabled", True):
        return None
    
    try:
        from ..analysis.attribution_callback import AttributionCallback
        return AttributionCallback(
            enabled=config.get("enabled", True),
            analysis_freq=config.get("analysis_freq", 1000),
            methods=config.get("methods", ["integrated_gradients"]),
            trainer=trainer,
            environment=environment
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create AttributionCallback: {e}")
        return None


def _create_performance_callback(
    config: Dict[str, Any],
    trainer: Optional[Any],
    environment: Optional[Any]
) -> Optional[BaseCallback]:
    """Create performance callback from config."""
    if not config.get("enabled", True):
        return None
    
    try:
        from ..analysis.performance_callback import PerformanceCallback
        return PerformanceCallback(
            enabled=config.get("enabled", True),
            analysis_freq=config.get("analysis_freq", 100),
            metrics=config.get("metrics", ["sharpe_ratio", "max_drawdown"])
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create PerformanceCallback: {e}")
        return None


def _create_optuna_callback(
    config: Dict[str, Any],
    trainer: Optional[Any]
) -> Optional[BaseCallback]:
    """Create optuna callback from config."""
    if not config.get("enabled", True):
        return None
    
    try:
        from ..optimization.optuna_callback import OptunaCallback
        return OptunaCallback(
            enabled=config.get("enabled", True),
            study_name=config.get("study_name"),
            storage_url=config.get("storage_url"),
            trainer=trainer
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create OptunaCallback: {e}")
        return None


def _create_early_stopping_callback(
    config: Dict[str, Any],
    trainer: Optional[Any]
) -> Optional[BaseCallback]:
    """Create early stopping callback from config."""
    if not config.get("enabled", True):
        return None
    
    try:
        from ..optimization.early_stopping_callback import EarlyStoppingCallback
        return EarlyStoppingCallback(
            enabled=config.get("enabled", True),
            patience=config.get("patience", 100),
            min_delta=config.get("min_delta", 0.001),
            metric=config.get("metric", "episode_reward"),
            trainer=trainer
        )
    except ImportError as e:
        logging.getLogger("callback_factory").warning(f"Could not create EarlyStoppingCallback: {e}")
        return None