import logging
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from callbacks.core import CallbackManager
from data.data_manager import DataManager
from .episode_manager import EpisodeManager, EpisodeManagerException
from agent.ppo_agent import PPOTrainer
from core.types import RolloutResult, UpdateResult
from core.model_manager import ModelManager
from config.training.training_config import TrainingManagerConfig
from envs import TradingEnvironment


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


class TrainingManager:
    """
    Responsibilities:
    - Training termination decisions (single source of truth)
    - Coordination between components 
    - Training state management
    - Callback coordination
    """

    def __init__(self, config: TrainingManagerConfig, model_manager: ModelManager, episode_manager: EpisodeManager):
        """Initialize TrainingManager as a single source of truth for training state."""
        self.config = config
        self.mode = TrainingMode(config.mode)
        self.logger = logging.getLogger(f"{__name__}.TrainingManager")

        # Core state - single source of truth
        self.state = TrainingState()
        self.termination_reason: Optional[str] = None

        # Component references (set during start)
        self.model_manager = model_manager
        self.episode_manager: Optional[EpisodeManager] =  episode_manager
        self.trainer: Optional[PPOTrainer] = None
        self.environment: Optional[TradingEnvironment] = None
        self.data_manager: Optional[DataManager] = None
        self.callback_manager: Optional[CallbackManager] = None

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

        # Initialize episode manager (it manages day/reset point loops internally)
        try:
            self.episode_manager.initialize()
        except EpisodeManagerException as e:
            self.logger.error(f"Episode manager initialization failed: {e.reason.value}")
            self._finalize_training(f"episode_manager_failed_{e.reason.value}")
            return

        # Initialize callbacks
        self.callback_manager.trigger_training_start({'training_state': self.state})

        try:
            while not self._should_terminate_training():
                # 1. Request the next episode from the episode manager
                episode_context = self.episode_manager.get_next_episode()
                self.callback_manager.trigger_episode_start({
                    'episode_context': episode_context,
                    'training_state': self.state
                })

                # 2. Setup environment with episode context
                setup_success, initial_obs = self._setup_episode(episode_context)
                if not setup_success:
                    self.logger.error("Failed to setup environment")
                    self.termination_reason = "environment_setup_failed"
                    break

                # 3. Collect rollout (trainer executes, no state tracking)
                self.callback_manager.trigger_event("rollout_start", {'training_state': self.state})
                rollout_result: RolloutResult = self.trainer.collect_rollout(
                    self.environment,
                    num_steps=self.config.rollout_steps,
                    initial_obs=initial_obs
                )
                self.callback_manager.trigger_event("rollout_end", {
                    'rollout_result': rollout_result,
                    'training_state': self.state
                })

                # 4. Update state (TrainingManager is a source of truth)
                self.state.global_steps += rollout_result.steps_collected
                self.state.global_episodes += rollout_result.episodes_completed

                # 5. Update policy if the buffer is ready
                if rollout_result.buffer_ready:
                    self.callback_manager.trigger_event("update_start", {
                        'rollout_result': rollout_result,
                        'training_state': self.state
                    })
                    update_info: UpdateResult = self.trainer.update_policy()  # Pure execution

                    # Update state (single source of truth)
                    self.state.global_updates += 1

                    # Notify episode manager about update
                    self.episode_manager.on_update_completed(update_info)

                    # Trigger update callbacks (for intelligent management)
                    self.callback_manager.trigger_update_end({
                        'update_info': update_info,
                        'training_state': self.state
                    })

                # 6. Notify episode manager about episode completions
                if rollout_result.episodes_completed > 0:
                    self.episode_manager.on_episodes_completed(count=rollout_result.episodes_completed)

                    # Trigger episode callbacks (for intelligent management)
                    self.callback_manager.trigger_episode_end({
                        'rollout_result': rollout_result,
                        'training_state': self.state
                    })

                # 7. Update global cycle count from episode manager
                self.state.global_cycles = self.episode_manager.get_completed_cycles()

        except EpisodeManagerException as e:
            self.logger.info(f"Episode manager terminated: {e.reason.value}")
            self.termination_reason = f"episode_manager_{e.reason.value}"
        except Exception as e:
            self.logger.error(f"🚨 Training error: {e}", exc_info=True)
            self.termination_reason = "error"

        self._finalize_training(self.termination_reason)

    def _setup_episode(self, episode_context) -> tuple[bool, Optional[Dict]]:
        """Setup episode using environment's consolidated setup logic."""
        try:
            # Extract session info from episode context
            symbol = episode_context.symbol
            date = episode_context.date
            reset_point = episode_context.reset_point

            self.logger.info(f"🎯 Setting up episode: {symbol} {date} at reset point {reset_point.timestamp}")

            # Let environment handle both initialization and reset
            obs, info = self.environment.reset(
                symbol=symbol,
                date=date,
                reset_point=reset_point
            )

            self.logger.debug(f"✅ Episode setup complete at {reset_point.timestamp}")
            return True, obs

        except Exception as e:
            self.logger.error(f"Failed to setup episode: {e}")
            return False, None

    def _should_terminate_training(self) -> bool:
        """Check if training should terminate based on global limits."""
        # Note: Callbacks will handle intelligent termination

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

    def _finalize_training(self, termination_reason: Optional[str]):
        """Finalize training with proper cleanup and callbacks."""
        reason_str = termination_reason or "UNKNOWN"
        self.logger.info(f"🏁 Training finalized. Reason: {reason_str}")

        # Trigger training end callback
        self.callback_manager.trigger_training_end({
            'reason': reason_str,
            'training_state': self.state
        })

        # Log final stats from an authoritative source
        self.logger.info(
            f"📊 Final stats: {self.state.global_cycles} cycles, {self.state.global_updates} updates, {self.state.global_episodes} episodes, {self.state.global_steps} steps")

        return None

    def _load_model(self) -> Dict[str, Any]:
        """Handle model loading if continuing training."""
        loaded_metadata = {}

        if self.config.continue_with_best_model:
            try:
                best_model = self.model_manager.find_best_model()
                if best_model:
                    self.logger.info(f"📂 Loading best model: {best_model['path']}")

                    # Load model and training state
                    model, training_state = self.model_manager.load_model(
                        self.trainer.model,
                        self.trainer.optimizer,
                        best_model["path"]
                    )
                    
                    # Restore training state (TrainingManager is source of truth)
                    self.state.global_steps = training_state.get("global_step", 0)
                    self.state.global_episodes = training_state.get("global_episode", 0)
                    self.state.global_updates = training_state.get("global_update", 0)
                    self.state.global_cycles = training_state.get("global_cycle", 0)

                    loaded_metadata = training_state.get("metadata", {})

                    self.logger.info(f"✅ Model loaded: step={training_state.get('global_step', 0)}")
                else:
                    self.logger.info("🆕 No previous model found. Starting fresh.")
            except Exception as e:
                self.logger.error(f"❌ Failed to load model: {e}")
                self.logger.info("🆕 Starting fresh training due to model loading failure")
        else:
            self.logger.info("🆕 Starting fresh training")

        return loaded_metadata

