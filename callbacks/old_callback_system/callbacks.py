"""Enhanced callback system for training, monitoring, and analysis.

This module provides a clean callback-based architecture to replace the complex metrics system.
Each callback is responsible for its own calculations and can be easily enabled/disabled.
"""

import logging
from abc import ABC
from typing import Dict, Any, List, Optional


class V1BaseCallback(ABC):
    """Base callback class with comprehensive hooks for all training events.

    This replaces the complex metrics system with a clean, extensible callback interface.
    When a callback is disabled, NO calculations are performed.
    """

    def __init__(self, enabled: bool = True):
        """Initialize callback.

        Args:
            enabled: Whether this callback is active. If False, all methods become no-ops.
        """
        self.enabled = enabled
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, method_name: str, *args, **kwargs):
        """Dynamic method caller that respects enabled state."""
        if not self.enabled:
            return

        method = getattr(self, method_name, None)
        if method and callable(method):
            return method(*args, **kwargs)

    # ========== Training Lifecycle ==========

    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        pass

    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Called when training ends."""
        pass

    # ========== Episode Lifecycle ==========

    def on_episode_start(self, episode_num: int, reset_info: Dict[str, Any]) -> None:
        """Called at the start of each episode.

        Args:
            episode_num: Current episode number
            reset_info: Information from environment reset (symbol, date, time, etc.)
        """
        pass

    def on_episode_end(self, episode_num: int, episode_data: Dict[str, Any]) -> None:
        """Called at the end of each episode.

        Args:
            episode_num: Current episode number
            episode_data: Episode data including reward, length, termination info, etc.
        """
        pass

    def on_episode_step(self, step_data: Dict[str, Any]) -> None:
        """Called after each environment step.

        Args:
            step_data: Contains state, action, reward, next_state, info, step_num
        """
        pass

    def on_step(self, trainer, state, action, reward, next_state, info) -> None:
        """Called after each training step.

        Args:
            trainer: The PPO trainer instance
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            info: Additional step information
        """
        pass

    # ========== PPO Training ==========

    def on_rollout_start(self, trainer) -> None:
        """Called before collecting rollouts."""
        pass

    def on_rollout_end(self, rollout_data: Dict[str, Any]) -> None:
        """Called after collecting rollouts.

        Args:
            rollout_data: Statistics about collected rollouts
        """
        pass

    def on_update_start(self, update_num: int) -> None:
        """Called before PPO update.
        
        Args:
            update_num: Current update number
        """
        pass

    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Called after PPO update.

        Args:
            update_num: Current update number
            update_metrics: Losses, gradients, learning rate, etc.
        """
        pass

    def on_update_iteration_end(self, trainer, update_num: int, update_metrics: Dict[str, Any], rollout_info: Dict[str, Any]) -> None:
        """Called after each update iteration.

        Args:
            trainer: The PPO trainer instance
            update_num: Current update number
            update_metrics: Update performance metrics
            rollout_info: Information about the rollout
        """
        pass

    def on_evaluation_start(self) -> None:
        """Called before evaluation episodes."""
        pass

    def on_evaluation_end(self, eval_results: Dict[str, Any]) -> None:
        """Called after evaluation episodes."""
        pass

    # ========== Trading Events ==========

    def on_order_placed(self, order_data: Dict[str, Any]) -> None:
        """Called when an order is placed."""
        pass

    def on_order_filled(self, fill_data: Dict[str, Any]) -> None:
        """Called when an order is filled."""
        pass

    def on_position_opened(self, position_data: Dict[str, Any]) -> None:
        """Called when a new position is opened."""
        pass

    def on_position_closed(self, trade_result: Dict[str, Any]) -> None:
        """Called when a position is closed (trade completed)."""
        pass

    def on_portfolio_update(self, portfolio_state: Dict[str, Any]) -> None:
        """Called when portfolio state changes."""
        pass

    # ========== Model Events ==========

    def on_model_forward(self, forward_data: Dict[str, Any]) -> None:
        """Called after model forward pass.

        Args:
            forward_data: Includes features, outputs, attention weights, internals
        """
        pass

    def on_gradient_update(self, gradient_data: Dict[str, Any]) -> None:
        """Called after gradient computation."""
        pass

    # ========== Feature Attribution ==========

    def on_attribution_computed(self, attribution_data: Dict[str, Any]) -> None:
        """Called when feature attribution is computed."""
        pass

    # ========== Momentum Training Events ==========

    def on_momentum_day_change(self, day_info: Dict[str, Any]) -> None:
        """Called when switching to a new momentum day."""
        pass

    def on_reset_point_selected(self, reset_info: Dict[str, Any]) -> None:
        """Called when a reset point is selected."""
        pass

    def on_curriculum_stage_change(self, stage_info: Dict[str, Any]) -> None:
        """Called when curriculum stage changes."""
        pass

    # ========== Custom Events ==========

    def on_custom_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Called for custom events."""
        pass


class CallbackManager:
    """Manages multiple callbacks and dispatches events."""

    def __init__(self, callbacks: Optional[List[V1BaseCallback]] = None):
        """Initialize callback manager.

        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []
        self.logger = logging.getLogger(__name__)

    def add_callback(self, callback: V1BaseCallback) -> None:
        """Add a callback to the manager."""
        if callback.enabled:
            self.callbacks.append(callback)
            self.logger.debug(
                f"Added {callback.__class__.__name__} to callback manager"
            )

    def remove_callback(self, callback: V1BaseCallback) -> None:
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def trigger(self, event_name: str, *args, **kwargs) -> None:
        """Trigger an event on all enabled callbacks.

        Args:
            event_name: Name of the callback method to call
            *args, **kwargs: Arguments to pass to the callback method
        """
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    method = getattr(callback, event_name, None)
                    if method and callable(method):
                        method(*args, **kwargs)
                except Exception as e:
                    self.logger.error(
                        f"Error in {callback.__class__.__name__}.{event_name}: {e}"
                    )

    def get_callback(self, callback_type: type) -> Optional[V1BaseCallback]:
        """Get a specific callback by type."""
        for callback in self.callbacks:
            if isinstance(callback, callback_type):
                return callback
        return None

    def disable_all(self) -> None:
        """Disable all callbacks."""
        for callback in self.callbacks:
            callback.enabled = False

    def enable_all(self) -> None:
        """Enable all callbacks."""
        for callback in self.callbacks:
            callback.enabled = True


# Convenience functions for creating callback managers
def create_callback_manager(config: Dict[str, Any]) -> CallbackManager:
    """Create a callback manager based on configuration.

    Args:
        config: Configuration dictionary with callback settings

    Returns:
        Configured CallbackManager instance
    """
    from callbacks.old_callback_system.optuna_callback import OptunaCallbackV1
    from callbacks.old_callback_system.wandb_callback import WandBCallbackV1
    from callbacks.old_callback_system.captum_callback import CaptumCallbackV1
    from feature.attribution.captum_attribution import AttributionConfig

    callbacks = []

    # Add Optuna callback if in Optuna trial (including subprocess mode)
    if config.get("optuna_trial") is not None or config.get("optuna_trial_info"):
        trial_obj = config.get("optuna_trial")  # Could be None for subprocess mode
        optuna_info = config.get("optuna_trial_info", {})
        metric_name = optuna_info.get("metric_name", "mean_reward")

        callbacks.append(
            OptunaCallbackV1(
                trial=trial_obj,  # None for subprocess mode
                metric_name=metric_name,
                enabled=True,
            )
        )

    # Add WandB callback if enabled
    if config.get("wandb", {}).get("enabled", False):
        callbacks.append(WandBCallbackV1(config=config.get("wandb", {}), enabled=True))


    # Add Captum callback if configured
    captum_config = config.get("captum")
    logging.getLogger(__name__).info(f"🔍 DEBUG: Captum config found: {captum_config is not None}")
    logging.getLogger(__name__).info(f"🔍 DEBUG: Captum config type: {type(captum_config)}")
    logging.getLogger(__name__).info(f"🔍 DEBUG: Captum config content: {captum_config}")
    if captum_config:
        # Check if Captum is enabled
        captum_enabled = captum_config.get("enabled", True) if isinstance(captum_config, dict) else getattr(captum_config, "enabled", True)
        logging.getLogger(__name__).info(f"🔍 DEBUG: Captum enabled value: {captum_enabled}")
        logging.getLogger(__name__).info(f"🔍 DEBUG: Captum config is dict: {isinstance(captum_config, dict)}")
        if not captum_enabled:
            logging.getLogger(__name__).info("❌ Captum is disabled via config, skipping initialization")
        else:
            logging.getLogger(__name__).info("✅ Initializing Captum callback...")
            try:
                # Handle both dict and Pydantic model
                if hasattr(captum_config, "model_dump"):
                    # It's a Pydantic model
                    captum_dict = captum_config.model_dump()
                else:
                    # It's already a dict
                    captum_dict = captum_config.copy() if isinstance(captum_config, dict) else {}
                
                # Extract callback settings
                callback_config = captum_dict.pop("callback", {})
                
                # Remove fields that are not part of AttributionConfig
                captum_dict.pop("enabled", None)
                captum_dict.pop("use_feature_manager_names", None)  # Always true
                captum_dict.pop("aggregate_features", None)  # Always true
                captum_dict.pop("feature_groups", None)  # Always auto-generated
                
                # Handle Pydantic callback config
                if hasattr(callback_config, "model_dump"):
                    callback_dict = callback_config.model_dump()
                elif isinstance(callback_config, dict):
                    callback_dict = callback_config
                else:
                    callback_dict = {}
                
                # Create AttributionConfig
                attribution_config = AttributionConfig(**captum_dict)
                
                # Create Captum callback
                captum_callback = CaptumCallbackV1(
                    config=attribution_config,
                    analyze_every_n_episodes=callback_dict.get("analyze_every_n_episodes", 10),
                    analyze_every_n_updates=callback_dict.get("analyze_every_n_updates", 5),
                    save_to_wandb=callback_dict.get("save_to_wandb", True),
                    save_to_dashboard=callback_dict.get("save_to_dashboard", False),
                    output_dir=callback_dict.get("output_dir", "outputs/captum"),
                )
                callbacks.append(captum_callback)
                logging.getLogger(__name__).info("🔍 Captum feature attribution callback added")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to initialize Captum callback: {e}")

    # Add any custom callbacks from config
    for callback_config in config.get("callbacks", []):
        callback_class = callback_config.get("class")
        if callback_class:
            # Dynamic import and instantiation
            module_name, class_name = callback_class.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            callback_cls = getattr(module, class_name)
            callback = callback_cls(**callback_config.get("params", {}))
            callbacks.append(callback)

    return CallbackManager(callbacks)
