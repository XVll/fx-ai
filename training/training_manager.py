import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path
import torch

from data.data_manager import DataManager
from .episode_manager import EpisodeManager, EpisodeTerminationReason
from ..agent.ppo_agent import PPOTrainer
from ..callbacks import CallbackManager
from ..callbacks.core.context import TrainingStartContext, TrainingEndContext, EpisodeEndContext
from ..core.types import TerminationReason, RolloutResult, UpdateResult
from ..core.model_manager import ModelManager
from ..core.shutdown import IShutdownHandler, IShutdownManager
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
    # Global counters (authoritative)
    total_episodes: int = 0
    total_updates: int = 0
    total_cycles: int = 0
    global_steps: int = 0
    
    # Training session info
    session_start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    last_episode_time: Optional[datetime] = None


class TrainingManager(IShutdownHandler):
    """
    Responsibilities:
    - Training termination decisions (single source of truth)
    - Coordination between components 
    - Training state management
    - Callback coordination
    """

    def __init__(self, config: TrainingManagerConfig, model_manager: ModelManager, device: torch.device, output_path: Path, run_id: str):
        """Initialize TrainingManager as a single source of truth for training state."""
        self.config = config
        self.model_manager = model_manager
        self.mode = TrainingMode(config.mode)
        self.logger = logging.getLogger(f"{__name__}.TrainingManager")
        
        # Context parameters
        self.device = device
        self.output_path = output_path
        self.run_id = run_id
        self.start_timestamp = datetime.now()

        # Core state - single source of truth
        self.state = TrainingState(session_start_time=self.start_timestamp)
        self.termination_reason: Optional[str] = None

        # Component references (set during start)
        self.trainer: Optional[PPOTrainer] = None
        self.environment: Optional[TradingEnvironment] = None
        self.data_manager: Optional[DataManager] = None
        self.callback_manager: Optional[CallbackManager] = None
        self.episode_manager: Optional[EpisodeManager] = None

        self.logger.info(f"🎯 TrainingManager initialized in {self.mode.value} mode as single source of truth")

    def register_shutdown(self, shutdown_manager: IShutdownManager) -> None:
        """Register this component with the shutdown manager."""
        shutdown_manager.register_component(
            component=self,
            timeout=self.config.shutdown_timeout,
            name="TrainingManager"
        )
        self.logger.info("📝 TrainingManager registered with shutdown manager")

    def start(self, trainer: PPOTrainer, environment: TradingEnvironment, data_manager: DataManager, callback_manager: CallbackManager) -> None:
        """Start the main training loop with distributed loop design and single source of truth."""

        self.logger.info(f"🎯 TRAINING LIFECYCLE STARTUP")
        self.logger.info(f"├── 🚀 TrainingManager started in {self.mode.value} mode")

        # Store component references
        self.trainer = trainer
        self.environment = environment
        self.data_manager = data_manager
        self.callback_manager = callback_manager

        # Handle model loading if continuing training
        loaded_metadata = self._load_model()

        # Create episode manager with its own loop
        self.episode_manager = EpisodeManager(self.config, self.data_manager)
        
        # Start episode manager (it manages day/reset point loops internally)
        if not self.episode_manager.start(
            symbols=self.config.symbols,
            date_range=self.config.date_range,
            daily_limits={
                'max_episodes': self.config.daily_max_episodes,
                'max_updates': self.config.daily_max_updates,
                'max_cycles': self.config.daily_max_cycles
            }
        ):
            return self._finalize_training("episode_manager_failed")

        # Initialize callbacks
        context = self._create_training_start_context()
        self.callback_manager.trigger_training_start(context)

        # MAIN TRAINING LOOP - Single, clean loop
        try:
            while not self.should_terminate_training():
                # 1. Request next episode from episode manager
                episode_context = self.episode_manager.get_next_episode()
                
                if episode_context is None:
                    self.logger.info("No more episodes available")
                    self.termination_reason = "no_more_data"
                    break
                
                # 2. Setup environment with episode context
                if not self._setup_environment_with_context(episode_context):
                    self.logger.error("Failed to setup environment")
                    self.termination_reason = "environment_setup_failed"
                    break
                
                # 3. Collect rollout (trainer executes, no state tracking)
                rollout_result: RolloutResult = self.trainer.collect_rollout(
                    self.environment,
                    target_steps=self.config.rollout_steps
                )
                
                # 4. Update state (TrainingManager is source of truth)
                self.state.global_steps += rollout_result.steps_collected
                self.state.total_episodes += rollout_result.episodes_completed
                if rollout_result.episodes_completed > 0:
                    self.state.last_episode_time = datetime.now()
                
                # 5. Update policy if buffer ready
                if rollout_result.buffer_ready:
                    update_info: UpdateResult = self.trainer.update_policy()  # Pure execution
                    
                    # Update state (single source of truth)
                    self.state.total_updates += 1
                    self.state.last_update_time = datetime.now()
                    
                    # Notify episode manager about update
                    self.episode_manager.on_update_completed(update_info)
                    
                    # Trigger update callbacks (for intelligent management)
                    self._trigger_update_callbacks(update_info)
                
                # 6. Notify episode manager about episode completions
                if rollout_result.episodes_completed > 0:
                    self.episode_manager.on_episodes_completed(
                        count=rollout_result.episodes_completed,
                        metrics=getattr(rollout_result, 'episode_metrics', [])
                    )
                    
                    # Trigger episode callbacks (for intelligent management)
                    self._trigger_episode_callbacks(rollout_result)

        except Exception as e:
            self.logger.error(f"🚨 Training error: {e}", exc_info=True)
            self.termination_reason = "error"

        return self._finalize_training(self.termination_reason)

    def _setup_environment_with_context(self, episode_context) -> bool:
        """Setup environment with episode context from episode manager."""
        try:
            # Extract session info from episode context
            symbol = episode_context.symbol
            date = episode_context.date
            reset_point = episode_context.reset_point
            
            self.logger.info(f"🎯 Setting up episode: {symbol} {date} at reset point {reset_point.timestamp}")
            
            # Setup trading session in environment
            self.environment.setup_session(symbol, date)
            
            # Reset environment to specific reset point
            initial_state, info = self.environment.reset_at_point(reset_point.index if hasattr(reset_point, 'index') else 0)
            
            self.logger.debug(f"✅ Episode setup complete: {symbol} {date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup episode: {e}")
            return False

    def _trigger_update_callbacks(self, update_info):
        """Trigger callbacks for policy updates - minimal for now."""
        if self.callback_manager:
            self.logger.debug(f"Update completed: {self.state.total_updates}")

    def _trigger_episode_callbacks(self, rollout_result):
        """Trigger callbacks for episode completions - minimal for now."""
        if self.callback_manager and rollout_result.episodes_completed > 0:
            self.logger.debug(f"Episodes completed: {rollout_result.episodes_completed}")

    def should_terminate_training(self) -> bool:
        """Check if training should terminate based on global limits."""
        # Note: Intelligent termination will be handled by callbacks
        
        if self.config.termination_max_episodes and self.state.total_episodes >= self.config.termination_max_episodes:
            self.termination_reason = f"max_episodes_reached_{self.config.termination_max_episodes}"
            return True
            
        if self.config.termination_max_updates and self.state.total_updates >= self.config.termination_max_updates:
            self.termination_reason = f"max_updates_reached_{self.config.termination_max_updates}"
            return True
            
        if self.config.termination_max_cycles and self.state.total_cycles >= self.config.termination_max_cycles:
            self.termination_reason = f"max_cycles_reached_{self.config.termination_max_cycles}"
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
        """Create training start context for callbacks."""
        return TrainingStartContext(
            config=self.config,
            trainer=self.trainer,
            environment=self.environment,
            model=self.trainer.model,
            device=self.device,
            output_path=self.output_path,
            run_id=self.run_id,
            timestamp=self.start_timestamp
        )


    def _create_training_end_context(self, reason: str) -> Optional[TrainingEndContext]:
        """Create training end context for callbacks."""
        try:
            from datetime import timedelta
            duration = datetime.now() - self.state.session_start_time if self.state.session_start_time else timedelta(0)
            
            return TrainingEndContext(
                final_metrics=None,  # Will be populated by callbacks if needed
                total_episodes=self.state.total_episodes,
                total_updates=self.state.total_updates,
                duration=duration,
                reason=reason,
                model=None,  # Will be populated by callbacks if needed
                best_reward=0.0,  # Will be populated by callbacks if needed
                average_reward=0.0,  # Will be populated by callbacks if needed
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.warning(f"Failed to create training end context: {e}")
            return None


    def _create_episode_end_context(self, metrics=None, rollout_result=None, update_result=None) -> Optional[EpisodeEndContext]:
        """Create minimal episode end context for callbacks."""
        # Minimal implementation - will be expanded later
        return None
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
        reason_str = termination_reason or "unknown"
        self.logger.info(f"🏁 Training finalized. Reason: {reason_str}")
        
        # Minimal callback triggering for now
        if self.callback_manager:
            context = self._create_training_end_context(reason_str)
            if context:
                self.callback_manager.trigger_training_end(context)
        
        # Log final stats from authoritative source
        self.logger.info(f"📊 Final stats: {self.state.total_episodes} episodes, {self.state.total_updates} updates, {self.state.global_steps} steps")
        
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

                # Restore training state (TrainingManager is source of truth)
                self.state.global_steps = model_state.get("global_step", 0)
                self.state.total_episodes = model_state.get("global_episode", 0)
                self.state.total_updates = model_state.get("global_update", 0)
                
                # Trainer doesn't track state - no counters to restore
                
                loaded_metadata = model_state.get("metadata", {})
                self.logger.info(f"✅ Model loaded: step={model_state.get('global_step', 0)}")
            else:
                self.logger.info("🆕 No previous model found. Starting fresh.")
        else:
            self.logger.info("🆕 Starting fresh training")
        
        return loaded_metadata
