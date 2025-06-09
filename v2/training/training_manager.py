import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .episode_manager import EpisodeManager, EpisodeTerminationReason
from ..agent.ppo_agent import PPOTrainer
from ..callbacks import CallbackManager
from ..callbacks.core.context import TrainingStartContext, TrainingEndContext, EpisodeEndContext
from ..core.types import TerminationReason
from ..core.shutdown import IShutdownHandler, get_global_shutdown_manager
from ..config.training.training_config import TrainingManagerConfig
from ..data.data_manager_impl import DataManager


class TrainingMode(Enum):
    """Training mode enumeration"""
    TRAINING = "training"
    OPTUNA = "optuna"
    BENCHMARK = "benchmark"


@dataclass
class TrainingState:
    """Current training state"""
    episodes: int = 0
    updates: int = 0
    cycles: int = 0
    global_steps: int = 0


class TrainingManager(IShutdownHandler):
    """
    V2 TrainingManager - Clean separation of concerns
    
    Responsibilities:
    - Training termination decisions (single source of truth)
    - Coordination between components 
    - Training state management
    - Callback coordination
    
    NOT responsible for:
    - Data management (handled by DataManager)
    - Episode management (handled by Environment) 
    - Model training (handled by Agent)
    """

    def __init__(self, config: TrainingManagerConfig):
        """Initialize TrainingManager with clean separation of concerns."""
        self.config = config
        self.mode = TrainingMode(config.mode)
        self.logger = logging.getLogger(f"{__name__}.TrainingManager")

        # Core state
        self.state = TrainingState()

        # Component references (set during start)
        self.trainer = Optional[PPOTrainer]
        self.environment = Optional[Any]
        self.data_manager = Optional[DataManager]
        self.callback_manager: Optional[CallbackManager]
        self.episode_manager = Optional[EpisodeManager]

        get_global_shutdown_manager().register_component(self, timeout=30)

        self.logger.info(f"🎯 TrainingManager initialized in {self.mode.value} mode")

    def start(self, trainer: PPOTrainer, environment: Any, data_manager: DataManager, callback_manager: CallbackManager) -> None:
        """Start the main training loop - core of the system."""

        self.logger.info(f"🎯 TRAINING LIFECYCLE STARTUP")
        self.logger.info(f"├── 🚀 TrainingManager started in {self.mode.value} mode")

        # Store component references
        self.trainer = trainer
        self.environment = environment
        self.data_manager = data_manager
        self.callback_manager = callback_manager

        # Initialize training state
        self.state = TrainingState()

        # Internal methods

        self.episode_manager = EpisodeManager(self.config, self.data_manager)

        if not self.episode_manager.initialize():
            return self._finalize_training(TerminationReason.DATA_EXHAUSTED)

        # Initialize callbacks
        context = self._create_training_start_context()
        self.callback_manager.trigger_training_start(context)

        termination_reason: TerminationReason | EpisodeTerminationReason = None;
        try:
            # Main training loop
            while not self.should_terminate():
                # Update training state from trainer
                self._update_training_state()

                # Check if episode manager should terminate
                episode_termination = self._should_terminate_episode_manager()
                if episode_termination:
                    termination_reason = episode_termination
                    break

                # Let trainer run one training step
                should_continue = self.trainer.run_training_step()

                if not should_continue:
                    termination_reason = TerminationReason.TRAINER_STOPPED
                    break

                # Let callbacks handle their features
                context = self._create_episode_end_context()
                self.callback_manager.trigger_episode_end(context)

                # Check episode advancement (episode manager)
                self._advance_episode_on_completion()

        except Exception as e:
            self.logger.error(f"🚨 Training error: {e}")
            termination_reason = TerminationReason.ERROR

        self.callback_manager.trigger_training_end(self._create_training_end_context(termination_reason))
        return None

    def should_terminate(self) -> bool:
        """Check if training should terminate."""

        if self.config.termination_max_episodes and self.state.episodes >= self.config.termination_max_episodes:
            return True
        if self.config.termination_max_updates and self.state.updates >= self.config.termination_max_updates:
            return True
        if self.config.termination_max_cycles and self.state.cycles >= self.config.termination_max_cycles:
            return True
        return False

    def _should_terminate_episode_manager(self) -> Optional[EpisodeTerminationReason]:
        """Check if episode manager should terminate."""
        if self.episode_manager:
            return self.episode_manager.should_terminate()
        return None

    def _update_training_state(self):
        """Update training state from trainer."""
        self.state.episodes = getattr(self.trainer, 'global_episode_counter', 0)
        self.state.updates = getattr(self.trainer, 'global_update_counter', 0)
        self.state.global_steps = getattr(self.trainer, 'global_step_counter', 0)

        # Update episode manager progress
        if self.episode_manager:
            self.episode_manager.update_progress(self.state.episodes, self.state.updates)

    def _advance_episode_on_completion(self):
        """Advance to next episode after completion."""
        if self.episode_manager:
            result = self.episode_manager.advance_episode()
            if result:
                self.logger.debug("✅ Advanced to next episode")
            else:
                self.logger.warning("❌ Failed to advance to next episode")
            return result
        return False

    def _create_training_start_context(self) -> TrainingStartContext:
        """Create a context dictionary for callbacks."""
        return TrainingStartContext(
            config=self.config,
            trainer=self.trainer,
            environment=self.environment,
            model=self.trainer.model,
            device=None,  # Todo pass from main
            output_path=None,  # Todo pass from main
            run_id=None,  # Todo wth is this.
            timestamp=None  # TodoTimestamp of what ?
        )

    def shutdown(self) -> None:
        """Perform graceful shutdown - stop training and cleanup resources."""
        self.episode_manager = None
        self.trainer = None
        self.environment = None
        self.callback_manager = None
        self.data_manager = None

        self.logger.info("✅ TrainingManager shutdown completed")

    def _create_training_end_context(self, reason:TerminationReason | EpisodeTerminationReason) -> TrainingEndContext:
        # Todo
        pass

    def _create_episode_end_context(self) -> EpisodeEndContext:
        # Todo
        pass
