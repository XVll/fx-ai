import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from data.data_manager import DataManager
from .episode_manager import EpisodeManager, EpisodeTerminationReason
from ..agent.ppo_agent import PPOTrainer
from ..callbacks import CallbackManager
from ..callbacks.core.context import TrainingStartContext, TrainingEndContext, EpisodeEndContext
from ..core.types import TerminationReason
from ..core.shutdown import IShutdownHandler, get_global_shutdown_manager
from ..config.training.training_config import TrainingManagerConfig
from ..envs import TradingEnvironment
from utils.model_manager import ModelManager


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

    def __init__(self, config: TrainingManagerConfig, model_manager: ModelManager):
        """Initialize TrainingManager with clean separation of concerns."""
        self.config = config
        self.model_manager = model_manager
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

    def start(self, trainer: PPOTrainer, environment: TradingEnvironment, data_manager: DataManager, callback_manager: CallbackManager) -> None:
        """Start the main training loop - core of the system."""

        self.logger.info(f"🎯 TRAINING LIFECYCLE STARTUP")
        self.logger.info(f"├── 🚀 TrainingManager started in {self.mode.value} mode")

        # Store component references
        self.trainer = trainer
        self.environment = environment
        self.data_manager = data_manager
        self.callback_manager = callback_manager

        # Handle model loading if continuing training
        loaded_metadata = self._load_model()

        # Initialize training state
        self.state = TrainingState()

        self.episode_manager = EpisodeManager(self.config, self.data_manager)

        if not self.episode_manager.initialize():
            return self._finalize_training(TerminationReason.DATA_EXHAUSTED)

        # Initialize callbacks
        context = self._create_training_start_context()
        self.callback_manager.trigger_training_start(context)

        # Initialize environment with first episode
        if not self._setup_next_episode():
            return self._finalize_training(TerminationReason.ERROR)

        termination_reason: TerminationReason | EpisodeTerminationReason = None;
        try:
            # Main training loop - V2 design with clean separation
            while not self.should_terminate():
                # 1. Collect rollout data from environment (TrainingManager controls environment)
                rollout_result = self.trainer.collect_rollout(self.environment)
                
                if rollout_result.interrupted:
                    termination_reason = TerminationReason.TRAINER_STOPPED
                    break
                
                # 2. Update policy if buffer is ready
                if rollout_result.buffer_ready:
                    update_result = self.trainer.update_policy()
                    
                    if update_result.interrupted:
                        termination_reason = TerminationReason.TRAINER_STOPPED
                        break
                    
                    # Update training state from trainer metrics
                    self._update_training_state()
                    
                    # Trigger callbacks with training metrics
                    metrics = self.trainer.get_training_metrics()
                    context = self._create_episode_end_context(metrics, rollout_result, update_result)
                    self.callback_manager.trigger_episode_end(context)

                # Check if episode manager should terminate
                episode_termination = self._should_terminate_episode_manager()
                if episode_termination:
                    termination_reason = episode_termination
                    break

                # Advance to next episode if needed
                if self._should_advance_episode(rollout_result):
                    # Advance episode manager to next reset point
                    if not self._advance_episode_on_completion():
                        termination_reason = TerminationReason.DATA_EXHAUSTED
                        break
                    
                    # Setup environment for the new episode
                    if not self._setup_next_episode():
                        self.logger.error("Failed to setup next episode")
                        termination_reason = TerminationReason.ERROR
                        break

        except Exception as e:
            self.logger.error(f"🚨 Training error: {e}")
            termination_reason = TerminationReason.ERROR

        return self._finalize_training(termination_reason)

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
        """Update training state from trainer metrics."""
        metrics = self.trainer.get_training_metrics()
        self.state.episodes = metrics.global_episodes
        self.state.updates = metrics.global_updates
        self.state.global_steps = metrics.global_steps

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

    def _setup_next_episode(self):
        """Setup environment for next episode using EpisodeManager configuration."""
        if not self.episode_manager or not self.environment:
            self.logger.error("Missing episode_manager or environment for episode setup")
            return False
        
        # Get current episode configuration from EpisodeManager
        episode_config = self.episode_manager.get_current_episode_config()
        if not episode_config:
            self.logger.error("Failed to get episode configuration from EpisodeManager")
            return False
        
        try:
            # Extract session info from episode config
            day_info = episode_config['day_info']
            symbol = day_info['symbol']
            date = day_info['date']
            reset_point_index = episode_config['reset_point_index']
            
            self.logger.info(f"🎯 Setting up episode: {symbol} {date} at reset point {reset_point_index}")
            
            # Setup trading session in environment
            self.environment.setup_session(symbol, date)
            
            # Reset environment to specific reset point
            initial_state, info = self.environment.reset_at_point(reset_point_index)
            
            self.logger.debug(f"✅ Episode setup complete: {symbol} {date} at reset point {reset_point_index}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup episode: {e}")
            return False

    def _should_advance_episode(self, rollout_result) -> bool:
        """Determine if we should advance to next episode."""
        # For now, advance if any episodes completed in rollout
        return rollout_result.total_episodes > 0

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


    def _create_training_end_context(self, reason: TerminationReason | EpisodeTerminationReason) -> Optional[TrainingEndContext]:
        """Create training end context for callbacks."""
        try:
            return TrainingEndContext(
                termination_reason=reason,
                final_metrics=self.trainer.get_training_metrics() if self.trainer else None,
                total_episodes=self.state.episodes,
                total_updates=self.state.updates,
                total_cycles=self.state.cycles
            )
        except Exception as e:
            self.logger.warning(f"Failed to create training end context: {e}")
            return None


    def _create_episode_end_context(self, metrics=None, rollout_result=None, update_result=None) -> EpisodeEndContext:
        """Create episode end context for callbacks."""
        # Todo: Implement proper context creation
        return EpisodeEndContext(
            episode_num=metrics.global_episodes if metrics else 0,
            episode_reward=metrics.last_episode_reward if metrics else 0.0,
            episode_length=metrics.last_episode_length if metrics else 0,
            # Add more context as needed
        )
    def shutdown(self) -> None:
        """Perform graceful shutdown - stop training and cleanup resources."""
        self.episode_manager = None
        self.trainer = None
        self.environment = None
        self.callback_manager = None
        self.data_manager = None

        self.logger.info("✅ TrainingManager shutdown completed")
    
    def _finalize_training(self, termination_reason: TerminationReason | EpisodeTerminationReason):
        """Finalize training with proper cleanup and callbacks."""
        self.logger.info(f"🏁 Training finalized. Reason: {termination_reason}")
        
        # Trigger training end callback
        context = self._create_training_end_context(termination_reason)
        if self.callback_manager and context:
            self.callback_manager.trigger_training_end(context)
        
        # Log final stats
        if self.trainer:
            metrics = self.trainer.get_training_metrics()
            self.logger.info(f"📊 Final stats: {metrics.global_episodes} episodes, {metrics.global_updates} updates")
        
        return None

    def _load_model(self) -> Dict[str, Any]:
        """Handle model loading if continuing training."""
        loaded_metadata = {}
        
        if self.config.continue_training:
            best_model_info = self.model_manager.find_best_model()
            if best_model_info:
                self.logger.info(f"📂 Loading best model: {best_model_info['path']}")
                
                # Load model and training state
                model, model_state = self.model_manager.load_model(
                    self.trainer.model, 
                    self.trainer.optimizer, 
                    best_model_info["path"]
                )

                # Todo Handle these counters in Training Manager
                # Restore trainer counters
                self.trainer.global_step_counter = model_state.get("global_step", 0)
                self.trainer.global_episode_counter = model_state.get("global_episode", 0)
                self.trainer.global_update_counter = model_state.get("global_update", 0)
                
                loaded_metadata = model_state.get("metadata", {})
                self.logger.info(f"✅ Model loaded: step={model_state.get('global_step', 0)}")
            else:
                self.logger.info("🆕 No previous model found. Starting fresh.")
        else:
            self.logger.info("🆕 Starting fresh training")
        
        return loaded_metadata
