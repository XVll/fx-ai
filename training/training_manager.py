import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from data.data_manager import DataManager
from .episode_manager import EpisodeManager, EpisodeManagerException
from ..agent.ppo_agent import PPOTrainer
from ..callbacks import CallbackManager
from ..core.types import RolloutResult, UpdateResult
from ..core.model_manager import ModelManager
from ..core.shutdown import IShutdownHandler, get_global_shutdown_manager
from ..config.training.training_config import TrainingManagerConfig
from ..envs import TradingEnvironment


class TrainingMode(Enum):
    """Training mode enumeration"""
    TRAINING = "training"
    OPTUNA = "optuna"
    BENCHMARK = "benchmark"


@dataclass
class TrainingState:
    """Training state - a single source of truth for all training progress"""
    initial_model_metadata: Dict[str, Any] = None
    # Global counters (authoritative)
    global_steps: int = 0
    global_episodes: int = 0
    global_updates: int = 0
    global_cycles: int = 0

    start_timestamp: datetime = datetime.now()


@dataclass
class ComponentState:
    """Simple state container for a callback system - no calculations, just component references"""
    trainer: Optional[PPOTrainer] = None
    environment: Optional[TradingEnvironment] = None
    episode_manager: Optional[EpisodeManager] = None
    training_state: Optional[TrainingState] = None
    model_manager: Optional[ModelManager] = None
    data_manager: Optional[DataManager] = None
    event_data: Optional[Dict[str, Any]] = None  # For event-specific data
    timestamp: datetime = datetime.now()


class TrainingManager(IShutdownHandler):
    """
    Responsibilities:
    - Training termination decisions (single source of truth)
    - Coordination between components 
    - Training state management
    - Callback coordination
    """

    def __init__(self, config: TrainingManagerConfig, model_manager: ModelManager):
        """Initialize TrainingManager as a single source of truth for training state."""
        self.config = config
        self.model_manager = model_manager
        self.mode = TrainingMode(config.mode)
        self.logger = logging.getLogger(f"{__name__}.TrainingManager")

        # Core state - single source of truth
        self.state = TrainingState()
        self.termination_reason: Optional[str] = None

        # Component references (set during start)
        self.trainer: Optional[PPOTrainer] = None
        self.environment: Optional[TradingEnvironment] = None
        self.data_manager: Optional[DataManager] = None
        self.callback_manager: Optional[CallbackManager] = None
        self.episode_manager: Optional[EpisodeManager] = None

        self.logger.info(f"🎯 TrainingManager initialized in {self.mode.value} mode as single source of truth")

    def start(self, trainer: PPOTrainer, environment: TradingEnvironment, data_manager: DataManager, callback_manager: CallbackManager):
        """Start the main training loop with distributed loop design and a single source of truth."""

        self.logger.info(f"🎯 TRAINING LIFECYCLE STARTUP")
        self.logger.info(f"├── 🚀 TrainingManager started in {self.mode.value} mode")

        # Store component references
        self.trainer = trainer
        self.environment = environment
        self.data_manager = data_manager
        self.callback_manager = callback_manager

        # Handle model loading if continuing training
        self.state.initial_model_metadata = self._load_model()

        self.episode_manager = EpisodeManager(self.config, self.data_manager)
        self.episode_manager.register_shutdown()

        # Initialize episode manager (it manages day/reset point loops internally)
        try:
            self.episode_manager.initialize()
        except EpisodeManagerException as e:
            self.logger.error(f"Episode manager initialization failed: {e.reason.value}")
            self._finalize_training(f"episode_manager_failed_{e.reason.value}")
            return

        # Initialize callbacks
        self._trigger_callback("training_start")

        try:
            while not self._should_terminate_training():
                # 1. Request the next episode from the episode manager
                episode_context = self.episode_manager.get_next_episode()
                self._trigger_callback("episode_start", {"episode_context": episode_context})

                # 2. Setup environment with episode context
                if not self._setup_environment_with_context(episode_context):
                    self.logger.error("Failed to setup environment")
                    self.termination_reason = "environment_setup_failed"
                    break

                # 3. Collect rollout (trainer executes, no state tracking)
                self._trigger_callback("rollout_start")
                rollout_result: RolloutResult = self.trainer.collect_rollout(
                    self.environment,
                    num_steps=self.config.rollout_steps
                )
                self._trigger_callback("rollout_end", {"rollout_result": rollout_result})

                # 4. Update state (TrainingManager is a source of truth)
                self.state.global_steps += rollout_result.steps_collected
                self.state.global_episodes += rollout_result.episodes_completed

                # 5. Update policy if the buffer is ready
                if rollout_result.buffer_ready:
                    self._trigger_callback("update_start", {"rollout_result": rollout_result})
                    update_info: UpdateResult = self.trainer.update_policy()  # Pure execution

                    # Update state (single source of truth)
                    self.state.global_updates += 1

                    # Notify episode manager about update
                    self.episode_manager.on_update_completed(update_info)

                    # Trigger update callbacks (for intelligent management)
                    self._trigger_callback("update_end", {"update_info": update_info})

                # 6. Notify episode manager about episode completions
                if rollout_result.episodes_completed > 0:
                    self.episode_manager.on_episodes_completed(count=rollout_result.episodes_completed)

                    # Trigger episode callbacks (for intelligent management)
                    self._trigger_callback("episode_end", {"rollout_result": rollout_result})

                # 7. Update global cycle count from episode manager
                self.state.global_cycles = self.episode_manager.get_completed_cycles()

        except EpisodeManagerException as e:
            self.logger.info(f"Episode manager terminated: {e.reason.value}")
            self.termination_reason = f"episode_manager_{e.reason.value}"
        except Exception as e:
            self.logger.error(f"🚨 Training error: {e}", exc_info=True)
            self.termination_reason = "error"

        self._finalize_training(self.termination_reason)

    def _setup_environment_with_context(self, episode_context) -> bool:
        """Setup environment with episode context from episode manager."""
        try:
            # Extract session info from episode context
            symbol = episode_context.symbol
            # Convert date to datetime if it's a string
            if isinstance(episode_context.date, str):
                import datetime as dt
                date = dt.datetime.strptime(episode_context.date, '%Y-%m-%d')
            else:
                date = episode_context.date
            reset_point = episode_context.reset_point

            self.logger.info(f"🎯 Setting up episode: {symbol} {date} at reset point {reset_point.timestamp}")

            # Setup trading session in environment
            self.environment.setup_session(symbol, date)

            # Reset environment to specific reset point
            reset_point_info = {
                'timestamp': reset_point.timestamp,
                'quality_score': reset_point.quality_score,
                'roc_score': reset_point.roc_score,
                'activity_score': reset_point.activity_score,
                'price': reset_point.price
            }
            _, _ = self.environment.reset_at_point(reset_point.index, reset_point_info)

            self.logger.debug(f"✅ Episode setup complete: {symbol} {date}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup episode: {e}")
            return False



    def _should_terminate_training(self) -> bool:
        """Check if training should terminate based on global limits."""
        # Note: Callbacks will handle intelligent termination

        # Check for the shutdown request first
        shutdown_manager = get_global_shutdown_manager()
        if shutdown_manager.is_shutdown_requested():
            self.termination_reason = "shutdown_requested"
            return True

        if self.config.termination_max_episodes and self.state.global_episodes >= self.config.termination_max_episodes:
            self.termination_reason = f"max_episodes_reached_{self.config.termination_max_episodes}"
            return True

        if self.config.termination_max_updates and self.state.global_updates >= self.config.termination_max_updates:
            self.termination_reason = f"max_updates_reached_{self.config.termination_max_updates}"
            return True

        if self.config.termination_max_cycles and self.state.global_cycles >= self.config.termination_max_cycles:
            self.termination_reason = f"max_cycles_reached_{self.config.termination_max_cycles}"
            return True

        return False

    def _get_component_state(self, event_data: Optional[Dict[str, Any]] = None) -> ComponentState:
        """Get the current component state for callbacks - no calculations, just references."""
        return ComponentState(
            trainer=self.trainer,
            environment=self.environment,
            episode_manager=self.episode_manager,
            training_state=self.state,
            model_manager=self.model_manager,
            data_manager=self.data_manager,
            event_data=event_data or {},
            timestamp=datetime.now()
        )

    def _trigger_callback(self, event_name: str, event_data: Optional[Dict[str, Any]] = None) -> None:
        """Simple callback trigger - just pass the component state."""
        if self.callback_manager:
            try:
                state:ComponentState = self._get_component_state(event_data)
                # For now, use a generic trigger method (will be replaced with a proper event system)
                if hasattr(self.callback_manager, 'trigger'):
                    self.callback_manager.trigger(event_name, state)
                self.logger.debug(f"Triggered callback: {event_name}")
            except Exception as e:
                self.logger.warning(f"Failed to trigger callback {event_name}: {e}")


    def register_shutdown(self) -> None:
        """Register this component with the global shutdown manager."""
        shutdown_manager = get_global_shutdown_manager()
        shutdown_manager.register_component(
            component=self,
            timeout=10,
            name="TrainingManager"
        )
        self.logger.info("📝 TrainingManager registered with shutdown manager")

    def shutdown(self) -> None:
        """Perform graceful shutdown - stop training and cleanup resources."""
        self.episode_manager = None
        self.trainer = None
        self.environment = None
        self.callback_manager = None
        self.data_manager = None

        self.logger.info("✅ TrainingManager shutdown completed")

    def _finalize_training(self, termination_reason: Optional[str]):
        """Finalize training with proper cleanup and callbacks."""
        reason_str = termination_reason or "UNKNOWN"
        self.logger.info(f"🏁 Training finalized. Reason: {reason_str}")

        # Minimal callback triggering for now
        self._trigger_callback("training_end", {"reason": reason_str})

        # Log final stats from an authoritative source
        self.logger.info(
            f"📊 Final stats: {self.state.global_cycles} cycles, {self.state.global_updates} updates, {self.state.global_episodes} episodes, {self.state.global_steps} steps")

        return None

    def _load_model(self) -> Dict[str, Any]:
        """Handle model loading if continuing training."""
        loaded_metadata = {}

        if self.config.continue_with_best_model:
            try:
                best_model_info = self.model_manager.find_best_model()
                if best_model_info:
                    self.logger.info(f"📂 Loading best model: {best_model_info['path']}")

                    # Load model and training state
                    model, model_state = self.model_manager.load_model(
                        self.trainer.model,
                        self.trainer.optimizer,
                        best_model_info["path"]
                    )
                    # Todo : We should use typed model state and metadata, both here and model_manager
                    # Restore training state (TrainingManager is source of truth)
                    self.state.global_steps = model_state.get("global_step", 0)
                    self.state.global_episodes = model_state.get("global_episode", 0)
                    self.state.global_updates = model_state.get("global_update", 0)
                    self.state.global_cycles = model_state.get("global_cycle", 0)

                    loaded_metadata = model_state.get("metadata", {})

                    self.logger.info(f"✅ Model loaded: step={model_state.get('global_step', 0)}")
                else:
                    self.logger.info("🆕 No previous model found. Starting fresh.")
            except Exception as e:
                self.logger.error(f"❌ Failed to load model: {e}")
                self.logger.info("🆕 Starting fresh training due to model loading failure")
        else:
            self.logger.info("🆕 Starting fresh training")

        return loaded_metadata
