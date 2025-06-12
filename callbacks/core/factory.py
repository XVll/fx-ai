"""
Callback factory for creating configured callbacks.

Supports all callback implementations except old_callback_system.
"""

from typing import Any, List, Optional

from config import CallbackConfig
from core.evaluation import Evaluator
from core.model_manager import ModelManager
from . import CallbackManager
from ..continuous_training_callback import ContinuousTrainingCallback
from ..evaluation_callback import EvaluationCallback
from ..metrics import (
    PPOMetricsCallback,
    ExecutionMetricsCallback,
    PortfolioMetricsCallback,
    ModelMetricsCallback,
    SessionMetricsCallback,
)

# Import Captum callback with graceful fallback
try:
    from ..metrics.captum_attribution_callback import CaptumAttributionCallback
    CAPTUM_CALLBACK_AVAILABLE = True
except ImportError:
    CAPTUM_CALLBACK_AVAILABLE = False
    CaptumAttributionCallback = None


def create_callbacks_from_config(
    config: CallbackConfig,
    model_manager: Optional[ModelManager] = None,
    evaluator: Optional[Evaluator] = None,
    captum_config: Optional[Any] = None,
) -> CallbackManager:
    """
    Create callback manager from configuration.
    
    Args:
        config: Callback configuration
        model_manager: ModelManager instance for continuous training
        evaluator: Evaluator instance for evaluation callback
        captum_config: Captum configuration for attribution analysis
        
    Returns:
        CallbackManager with configured callbacks
    """
    callbacks = []
    
    # Continuous training callback
    if config.continuous.enabled:
        if model_manager is None:
            raise ValueError("ModelManager required for continuous training callback")
        callbacks.append(ContinuousTrainingCallback(
            model_manager=model_manager,
            config=config.continuous
        ))
    
    # Evaluation callback
    if config.evaluation.enabled:
        if evaluator is None:
            raise ValueError("Evaluator required for evaluation callback")
        callbacks.append(EvaluationCallback(
            evaluator=evaluator,
            update_frequency=config.evaluation.update_frequency,
            episode_frequency=config.evaluation.episode_frequency,
            time_frequency_minutes=config.evaluation.time_frequency_minutes,
            enabled=config.evaluation.enabled,
        ))
    
    # PPO metrics callback
    if config.ppo_metrics.enabled:
        callbacks.append(PPOMetricsCallback(
            buffer_size=config.ppo_metrics.buffer_size,
            enabled=config.ppo_metrics.enabled
        ))
    
    # Execution metrics callback
    if config.execution_metrics.enabled:
        callbacks.append(ExecutionMetricsCallback(
            buffer_size=config.execution_metrics.buffer_size,
            enabled=config.execution_metrics.enabled
        ))
    
    # Portfolio metrics callback
    if config.portfolio_metrics.enabled:
        callbacks.append(PortfolioMetricsCallback(
            buffer_size=config.portfolio_metrics.buffer_size,
            log_frequency=config.portfolio_metrics.log_frequency,
            enabled=config.portfolio_metrics.enabled
        ))
    
    # Model metrics callback
    if config.model_metrics.enabled:
        callbacks.append(ModelMetricsCallback(
            buffer_size=config.model_metrics.buffer_size,
            enabled=config.model_metrics.enabled
        ))
    
    # Session metrics callback
    if config.session_metrics.enabled:
        callbacks.append(SessionMetricsCallback(
            log_frequency=config.session_metrics.log_frequency,
            track_system_resources=config.session_metrics.track_system_resources,
            enabled=config.session_metrics.enabled
        ))
    
    # Captum attribution callback
    if (config.captum_attribution.enabled and 
        captum_config is not None and 
        CAPTUM_CALLBACK_AVAILABLE and 
        CaptumAttributionCallback is not None):
        callbacks.append(CaptumAttributionCallback(
            config=captum_config,
            enabled=config.captum_attribution.enabled
        ))
    elif config.captum_attribution.enabled and not CAPTUM_CALLBACK_AVAILABLE:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Captum attribution callback requested but not available (Captum not installed)")
    elif config.captum_attribution.enabled and captum_config is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Captum attribution callback enabled but no captum config provided")
    
    return CallbackManager(callbacks)