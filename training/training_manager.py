"""
Training Manager - Central Authority for Training Lifecycle
Single source of truth for all training decisions and termination.
Coordinates between training lifecycle and data lifecycle management.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from training.continuous_training import ContinuousTraining
from training.data_lifecycle_manager import DataLifecycleManager, DataTerminationReason
from utils.graceful_shutdown import get_shutdown_manager


class TrainingMode(Enum):
    """Training mode enumeration"""
    SWEEP = "sweep"
    PRODUCTION = "production"
    OPTUNA = "optuna"  # Pure hyperparameter optimization mode


class TerminationReason(Enum):
    """Reasons for training termination"""
    # Training-specific termination
    MAX_EPISODES_REACHED = "max_episodes_reached"
    MAX_UPDATES_REACHED = "max_updates_reached"
    MAX_CYCLES_REACHED = "max_cycles_reached"
    PERFORMANCE_PLATEAU = "performance_plateau"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_INTERRUPT = "user_interrupt"
    EXTERNAL_SIGNAL = "external_signal"
    SWEEP_TRIAL_COMPLETE = "sweep_trial_complete"
    
    # Data lifecycle termination
    DATA_EXHAUSTED = "data_exhausted"
    ALL_STAGES_COMPLETED = "all_stages_completed"
    NO_SUITABLE_DATA = "no_suitable_data"
    PRELOAD_FAILED = "preload_failed"


@dataclass
class TrainingState:
    """Current training state"""
    episodes: int = 0
    updates: int = 0
    global_steps: int = 0
    training_hours: float = 0.0
    start_time: Optional[datetime] = None
    current_performance: float = 0.0
    best_performance: float = float('-inf')
    termination_votes: List[TerminationReason] = None
    
    # Data lifecycle state
    current_stage: Optional[str] = None
    cycle_count: int = 0
    current_day: Optional[str] = None
    current_symbol: Optional[str] = None
    stage_progress: float = 0.0
    data_preload_ready: bool = False
    
    def __post_init__(self):
        if self.termination_votes is None:
            self.termination_votes = []
        if self.start_time is None:
            self.start_time = datetime.now()


@dataclass
class TrainingRecommendations:
    """Recommendations from continuous training advisor"""
    data_difficulty_change: Optional[Dict[str, Any]] = None
    training_parameter_changes: Optional[Dict[str, Any]] = None
    termination_suggestion: Optional[TerminationReason] = None
    checkpoint_request: bool = False
    evaluation_request: bool = False
    
    @classmethod
    def no_changes(cls) -> 'TrainingRecommendations':
        """Create empty recommendations"""
        return cls()


class TerminationController:
    """Controls training termination decisions"""
    
    def __init__(self, config: Dict[str, Any], mode: TrainingMode):
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Termination criteria - handle None values from config
        self.training_max_episodes = config.get('training_max_episodes', float('inf'))
        if self.training_max_episodes is None:
            self.training_max_episodes = float('inf')
            
        self.training_max_updates = config.get('training_max_updates', float('inf'))
        if self.training_max_updates is None:
            self.training_max_updates = float('inf')
            
        self.training_max_cycles = config.get('training_max_cycles', float('inf'))
        if self.training_max_cycles is None:
            self.training_max_cycles = float('inf')
        
        # Intelligent termination (only for production mode)
        self.enable_intelligent_termination = (
            mode == TrainingMode.PRODUCTION and 
            config.get('intelligent_termination', True)
        )
        self.plateau_patience = config.get('plateau_patience', 50)
        self.degradation_threshold = config.get('degradation_threshold', 0.05)
        
        # Performance tracking for intelligent termination
        self.performance_history: List[float] = []
        self.updates_since_improvement = 0
        self.best_performance = float('-inf')
        
    def should_terminate(self, state: TrainingState) -> Optional[TerminationReason]:
        """Single source of truth for termination decisions"""
        
        # Check hard limits (always enforced)
        if state.episodes >= self.training_max_episodes:
            return TerminationReason.MAX_EPISODES_REACHED
            
        if state.updates >= self.training_max_updates:
            return TerminationReason.MAX_UPDATES_REACHED
            
        if state.cycle_count >= self.training_max_cycles:
            return TerminationReason.MAX_CYCLES_REACHED
        
        # Check external termination votes
        if state.termination_votes:
            return state.termination_votes[0]  # Take first vote
        
        # Intelligent termination (only in production mode, never in optuna mode)
        if self.enable_intelligent_termination and self.mode == TrainingMode.PRODUCTION:
            intelligent_reason = self._check_intelligent_termination(state)
            if intelligent_reason:
                return intelligent_reason
        
        return None  # Continue training
    
    def _check_intelligent_termination(self, state: TrainingState) -> Optional[TerminationReason]:
        """Check intelligent termination criteria"""
        if len(self.performance_history) < 20:
            return None  # Need more data
        
        # Check for performance plateau
        if self.updates_since_improvement >= self.plateau_patience:
            self.logger.info(
                f"🛑 Performance plateau detected: {self.updates_since_improvement} updates without improvement"
            )
            return TerminationReason.PERFORMANCE_PLATEAU
        
        # Check for performance degradation
        if len(self.performance_history) >= 50:
            recent_performance = sum(self.performance_history[-10:]) / 10
            older_performance = sum(self.performance_history[-50:-40]) / 10
            
            if older_performance > 0 and (recent_performance - older_performance) / older_performance < -self.degradation_threshold:
                self.logger.info(
                    f"🛑 Performance degradation detected: {recent_performance:.4f} vs {older_performance:.4f}"
                )
                return TerminationReason.PERFORMANCE_DEGRADATION
        
        return None
    
    def update_performance(self, performance: float, update_count: int):
        """Update performance tracking for intelligent termination"""
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Track improvements
        if performance > self.best_performance:
            self.best_performance = performance
            self.updates_since_improvement = 0
        else:
            self.updates_since_improvement = update_count - getattr(self, '_last_update_count', 0)
        
        self._last_update_count = update_count


# EpisodeController removed - episode management moved to data lifecycle


class TrainingManager:
    """
    Central Authority for Training Lifecycle Management
    
    Responsibilities:
    - Training termination (single source of truth)
    - Coordination between training and data lifecycles
    - Mode switching (sweep vs production)
    - Configuration coordination
    - Integration with continuous training advisor
    """
    
    def __init__(self, config: Dict[str, Any], mode: str = "production", available_days: List = None):
        self.config = config
        self.mode = TrainingMode(mode)
        self.logger = logging.getLogger(__name__)
        
        # Core state
        self.state = TrainingState()
        self.should_stop = False
        self.termination_reason: Optional[TerminationReason] = None
        
        # Controllers - ensure termination_config is always a dict
        termination_config = self._convert_pydantic_to_dict(config, 'termination', {
            'training_max_episodes': float('inf'),
            'training_max_updates': float('inf'),
            'training_max_cycles': float('inf'),
            'intelligent_termination': False,
            'plateau_patience': 10,
            'degradation_threshold': 0.1,
        })
        
        self.termination_controller = TerminationController(
            termination_config, self.mode
        )
        
        # Data lifecycle manager
        self.data_lifecycle_manager = None
        
        # Handle data lifecycle config access
        data_lifecycle_config_dict = self._convert_pydantic_to_dict(config, 'data_lifecycle', {'enabled': True})
        data_lifecycle_enabled = data_lifecycle_config_dict.get('enabled', True)
            
        if available_days and data_lifecycle_enabled:
            # Get data lifecycle config - handle both object access and dict access
            data_lifecycle_config = None
            if hasattr(config, 'data_lifecycle'):
                data_lifecycle_config = config.data_lifecycle
            elif isinstance(config, dict) and 'data_lifecycle' in config:
                lifecycle_data = config['data_lifecycle']
                # Check if it's already a config object or needs to be created
                if hasattr(lifecycle_data, '__dict__'):  # It's already a config object
                    data_lifecycle_config = lifecycle_data
                else:  # It's a dict, create config object
                    from config.config import DataLifecycleConfig
                    data_lifecycle_config = DataLifecycleConfig(**lifecycle_data)
            else:
                # Use default config
                from config.config import DataLifecycleConfig
                data_lifecycle_config = DataLifecycleConfig()
                
            self.data_lifecycle_manager = DataLifecycleManager(
                data_lifecycle_config,
                available_days
            )
        
        # Continuous training advisor and model manager
        continuous_config = self._convert_pydantic_to_dict(config, 'continuous', {})
        
        # In optuna mode, disable continuous training entirely
        if self.mode == TrainingMode.OPTUNA:
            continuous_config['enabled'] = False
            continuous_config['bypass_all_features'] = True
            
        self.continuous_training = ContinuousTraining(
            config=continuous_config,
            mode=self.mode.value,
            enabled=self.mode != TrainingMode.OPTUNA  # Disable for optuna
        )
        
        # Initialize shutdown manager
        self._init_shutdown_manager()
        
        self.logger.info(f"🎯 TrainingManager initialized in {self.mode.value} mode")
    
    def _convert_pydantic_to_dict(self, config, key: str, default_values: dict) -> dict:
        """Convert Pydantic object to dict with fallback to manual extraction"""
        result = {}
        
        if hasattr(config, key):
            # Config is object with attribute
            obj = getattr(config, key)
            if hasattr(obj, 'model_dump'):
                result = obj.model_dump()
            elif hasattr(obj, '__dict__'):
                result = obj.__dict__
            else:
                # Manual extraction
                result = {k: getattr(obj, k, v) for k, v in default_values.items()}
        elif isinstance(config, dict):
            # Config is dict
            raw_obj = config.get(key, {})
            if hasattr(raw_obj, 'model_dump'):
                result = raw_obj.model_dump()
            elif hasattr(raw_obj, '__dict__'):
                result = raw_obj.__dict__
            elif isinstance(raw_obj, dict):
                result = raw_obj
            else:
                # Manual extraction
                result = {k: getattr(raw_obj, k, v) for k, v in default_values.items()}
        else:
            result = default_values
            
        return result
        
    # Register for graceful shutdown
    def _init_shutdown_manager(self):
        """Initialize shutdown manager after all other setup is complete"""
        from utils.graceful_shutdown import get_shutdown_manager
        self.shutdown_manager = get_shutdown_manager()
        self.shutdown_manager.register_component(
            "TrainingManager", 
            self._graceful_shutdown,
            timeout=60.0,
            critical=True
        )
        
    def start_training(self, trainer) -> Dict[str, Any]:
        """
        Start training with the configured trainer
        Returns final training statistics
        """
        self.logger.info("🎯 TRAINING LIFECYCLE STARTUP")
        self.logger.info("├── 🚀 TrainingManager started in %s mode", self.mode.value)
        self.state.start_time = datetime.now()
        
        # Set TrainingManager reference on trainer for data lifecycle integration
        if hasattr(trainer, 'set_training_manager'):
            trainer.set_training_manager(self)
        
        # Initialize data lifecycle
        if self.data_lifecycle_manager:
            self.logger.info("├── 🔄 Initializing adaptive data lifecycle...")
            if not self.data_lifecycle_manager.initialize():
                self.logger.error("│   └── ❌ Failed to initialize data lifecycle")
                return self._finalize_training(trainer)
            else:
                self.logger.info("│   └── ✅ Data lifecycle ready")
        else:
            self.logger.warning("├── ⚠️ No DataLifecycleManager - using fallback")
        
        # Initialize continuous training
        self.continuous_training.initialize(trainer)
        
        # Trigger trainer's on_training_start callback to initialize WandB and other callbacks
        if hasattr(trainer, 'callback_manager'):
            training_config = {
                "mode": self.mode.value,
                "full_config": self.config,
                "experiment_name": getattr(self.config, "experiment_name", "training_manager"),
                "model": getattr(trainer, 'model', None),
                "model_config": getattr(trainer, 'model_config', {}),
            }
            trainer.callback_manager.trigger("on_training_start", training_config)
            
            # Also call trainer callbacks directly for compatibility
            for callback in getattr(trainer, 'callbacks', []):
                if hasattr(callback, 'on_training_start'):
                    callback.on_training_start(trainer)
        
        try:
            while not self.should_stop and not self.shutdown_manager.is_shutdown_requested():
                # Update training state
                self._update_training_state(trainer)
                
                # Check data lifecycle termination FIRST
                if self.data_lifecycle_manager:
                    data_termination = self.data_lifecycle_manager.should_terminate_data_lifecycle()
                    if data_termination:
                        self._terminate_training(self._map_data_termination(data_termination))
                        break
                
                # Check training termination conditions
                training_termination = self.termination_controller.should_terminate(self.state)
                if training_termination:
                    self._terminate_training(training_termination)
                    break
                
                # Get recommendations from continuous training advisor (skip in optuna mode)
                if self.mode != TrainingMode.OPTUNA:
                    recommendations = self.continuous_training.get_recommendations(
                        self.state, self._get_performance_metrics(trainer)
                    )
                    
                    # Process recommendations
                    self._process_recommendations(recommendations, trainer)
                
                
                # Let trainer run one step/episode
                should_continue = self._run_training_step(trainer)
                if not should_continue:
                    self._terminate_training(TerminationReason.EXTERNAL_SIGNAL)
                    break
            
            # Check if we exited due to shutdown request
            if self.shutdown_manager.is_shutdown_requested():
                self._terminate_training(TerminationReason.USER_INTERRUPT)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Training interrupted by user")
            self._terminate_training(TerminationReason.USER_INTERRUPT)
        except Exception as e:
            self.logger.error(f"🚨 Training error: {e}")
            self._terminate_training(TerminationReason.EXTERNAL_SIGNAL)
            raise
        
        return self._finalize_training(trainer)
    
    def _map_data_termination(self, data_reason: DataTerminationReason) -> TerminationReason:
        """Map data termination reason to training termination reason"""
        mapping = {
            DataTerminationReason.CYCLE_LIMIT_REACHED: TerminationReason.ALL_STAGES_COMPLETED,
            DataTerminationReason.EPISODE_LIMIT_REACHED: TerminationReason.ALL_STAGES_COMPLETED,
            DataTerminationReason.UPDATE_LIMIT_REACHED: TerminationReason.ALL_STAGES_COMPLETED,
            DataTerminationReason.NO_MORE_RESET_POINTS: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.NO_MORE_DAYS: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.DATE_RANGE_EXHAUSTED: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.QUALITY_CRITERIA_NOT_MET: TerminationReason.NO_SUITABLE_DATA,
            DataTerminationReason.ALL_STAGES_COMPLETED: TerminationReason.ALL_STAGES_COMPLETED,
            DataTerminationReason.PRELOAD_FAILED: TerminationReason.PRELOAD_FAILED,
        }
        return mapping.get(data_reason, TerminationReason.DATA_EXHAUSTED)
    
    def _update_training_state(self, trainer):
        """Update current training state"""
        self.state.episodes = getattr(trainer, 'global_episode_counter', 0)
        self.state.updates = getattr(trainer, 'global_update_counter', 0)
        self.state.global_steps = getattr(trainer, 'global_step_counter', 0)
        
        # Calculate training time
        if self.state.start_time:
            elapsed = datetime.now() - self.state.start_time
            self.state.training_hours = elapsed.total_seconds() / 3600
        
        # Update data lifecycle manager
        if self.data_lifecycle_manager:
            self.data_lifecycle_manager.update_progress(self.state.episodes, self.state.updates)
            
            # Get data lifecycle status
            data_status = self.data_lifecycle_manager.get_data_lifecycle_status()
            self.state.current_stage = data_status.get('stage_name', 'adaptive')
            self.state.cycle_count = data_status.get('cycle_count', 0)
            self.state.current_day = data_status.get('current_day')
            self.state.current_symbol = data_status.get('current_symbol')
            self.state.stage_progress = data_status.get('stage_progress', 0.0)
            self.state.data_preload_ready = data_status.get('preload_ready', False)
        else:
            # Set defaults when no data lifecycle manager is available
            if self.state.current_stage is None:
                self.state.current_stage = 'adaptive'
        
        # Update performance tracking
        performance_metrics = self._get_performance_metrics(trainer)
        if performance_metrics and 'mean_reward' in performance_metrics:
            self.state.current_performance = performance_metrics['mean_reward']
            self.state.best_performance = max(
                self.state.best_performance, self.state.current_performance
            )
            
            # Update termination controller
            self.termination_controller.update_performance(
                self.state.current_performance, self.state.updates
            )
    
    def _get_performance_metrics(self, trainer) -> Dict[str, Any]:
        """Get current performance metrics from trainer"""
        return {
            'mean_reward': getattr(trainer, 'mean_episode_reward', 0.0),
            'episodes': self.state.episodes,
            'updates': self.state.updates,
            'global_steps': self.state.global_steps
        }
    
    def _process_recommendations(self, recommendations: TrainingRecommendations, trainer):
        """Process recommendations from continuous training advisor"""
        
        # Handle termination suggestion (only in production mode)
        if (recommendations.termination_suggestion and 
            self.mode == TrainingMode.PRODUCTION):
            self.state.termination_votes.append(recommendations.termination_suggestion)
            self.logger.info(f"📋 Continuous training suggests termination: {recommendations.termination_suggestion.value}")
        
        # Handle data difficulty changes
        if recommendations.data_difficulty_change:
            self._apply_data_difficulty_change(recommendations.data_difficulty_change, trainer)
        
        # Handle training parameter changes (only in production mode)
        if (recommendations.training_parameter_changes and 
            self.mode == TrainingMode.PRODUCTION):
            self._apply_training_parameter_changes(recommendations.training_parameter_changes, trainer)
        
        # Handle checkpoint requests
        if recommendations.checkpoint_request:
            self.continuous_training.handle_checkpoint_request(trainer)
        
        # Handle evaluation requests
        if recommendations.evaluation_request:
            self._handle_evaluation_request(trainer)
    
    def _apply_data_difficulty_change(self, change: Dict[str, Any], trainer):
        """Apply data difficulty changes"""
        if hasattr(trainer, 'apply_data_difficulty_change'):
            trainer.apply_data_difficulty_change(change)
            self.logger.info(f"📊 Applied data difficulty change: {change}")
    
    def _apply_training_parameter_changes(self, changes: Dict[str, Any], trainer):
        """Apply training parameter changes (production mode only)"""
        self.logger.info(f"⚙️ Applied training parameter changes: {changes}")
        # Implementation depends on trainer interface
    
    def _handle_evaluation_request(self, trainer):
        """Handle evaluation request from continuous training"""
        if hasattr(trainer, 'evaluate'):
            # Get number of episodes from continuous training config
            n_episodes = getattr(self.continuous_training, 'evaluation_episodes', 10)
            eval_results = trainer.evaluate(n_episodes=n_episodes)
            self.continuous_training.process_evaluation_results(eval_results)
    
    
    def _run_training_step(self, trainer) -> bool:
        """Run one training step/episode"""
        # Use the trainer's run_training_step method if available
        if hasattr(trainer, 'run_training_step'):
            return trainer.run_training_step()
        else:
            # Fallback: check if trainer wants to stop
            return not getattr(trainer, 'stop_training', False)
    
    def _terminate_training(self, reason: TerminationReason):
        """Terminate training with given reason"""
        self.should_stop = True
        self.termination_reason = reason
        self.logger.info(f"🏁 Training terminated: {reason.value}")
        
        # Notify continuous training
        self.continuous_training.on_training_termination(reason)
    
    def _finalize_training(self, trainer) -> Dict[str, Any]:
        """Finalize training and return statistics"""
        # In optuna mode, return simplified stats
        if self.mode == TrainingMode.OPTUNA:
            final_stats = {
                'mean_reward': self.state.current_performance,
                'total_episodes': self.state.episodes,
                'total_updates': self.state.updates,
            }
        else:
            final_stats = {
                'termination_reason': self.termination_reason.value if self.termination_reason else 'unknown',
                'total_episodes': self.state.episodes,
                'total_updates': self.state.updates,
                'total_steps': self.state.global_steps,
                'training_hours': self.state.training_hours,
                'final_performance': self.state.current_performance,
                'best_performance': self.state.best_performance,
                'mean_reward': self.state.current_performance,  # Add mean_reward for optuna compatibility
            }
        
        # Finalize continuous training (skip in optuna mode)
        if self.mode != TrainingMode.OPTUNA:
            continuous_stats = self.continuous_training.finalize_training(final_stats)
            final_stats.update(continuous_stats)
        
        self.logger.info(f"🎯 Training completed: {final_stats}")
        return final_stats
    
    def request_termination(self, reason: TerminationReason):
        """External request for training termination"""
        self.state.termination_votes.append(reason)
        self.logger.info(f"📥 External termination request: {reason.value}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            'mode': self.mode.value,
            'should_stop': self.should_stop,
            'episodes': self.state.episodes,
            'updates': self.state.updates,
            'global_steps': self.state.global_steps,
            'training_hours': self.state.training_hours,
            'current_performance': self.state.current_performance,
            'best_performance': self.state.best_performance,
            'termination_reason': self.termination_reason.value if self.termination_reason else None,
            
            # Data lifecycle status
            'current_stage': self.state.current_stage,
            'cycle_count': self.state.cycle_count,
            'current_day': self.state.current_day,
            'current_symbol': self.state.current_symbol,
            'stage_progress': self.state.stage_progress,
            'data_preload_ready': self.state.data_preload_ready
        }
        
        # Add detailed data lifecycle status if available
        if self.data_lifecycle_manager:
            data_status = self.data_lifecycle_manager.get_data_lifecycle_status()
            status['data_lifecycle'] = data_status
        
        return status
    
    def get_episode_config(self) -> Dict[str, Any]:
        """Get current episode configuration from data lifecycle (for cycle management only)"""
        if self.data_lifecycle_manager:
            cycle_config = self.data_lifecycle_manager.config.cycles
            return {
                'day_max_episodes': cycle_config.day_max_episodes,
                'day_max_updates': cycle_config.day_max_updates,
                'day_max_cycles': cycle_config.day_max_cycles
            }
        else:
            # Use data lifecycle config from training manager config if available
            data_lifecycle_config_dict = self._convert_pydantic_to_dict(self.config, 'data_lifecycle', {'enabled': True})
            if 'cycles' in data_lifecycle_config_dict:
                cycles_config = data_lifecycle_config_dict['cycles']
                if isinstance(cycles_config, dict):
                    return {
                        'day_max_episodes': cycles_config.get('day_max_episodes'),
                        'day_max_updates': cycles_config.get('day_max_updates'),
                        'day_max_cycles': cycles_config.get('day_max_cycles', 3)
                    }
            
            return {'day_max_cycles': 3}  # Final fallback
    
    def get_current_training_data(self) -> Optional[Dict[str, Any]]:
        """Get current training data configuration"""
        if self.data_lifecycle_manager:
            return self.data_lifecycle_manager.get_current_training_data()
        return None
    
    def _graceful_shutdown(self):
        """Graceful shutdown of training manager"""
        self.logger.info("🔄 TrainingManager graceful shutdown initiated")
        
        # Stop training loop
        self.should_stop = True
        
        # Finalize continuous training
        if hasattr(self, 'continuous_training'):
            try:
                self.continuous_training.finalize_training({})
                self.logger.info("✅ ContinuousTraining finalized")
            except Exception as e:
                self.logger.error(f"❌ Error finalizing ContinuousTraining: {e}")
        
        # Clean up data lifecycle manager
        if hasattr(self, 'data_lifecycle_manager') and self.data_lifecycle_manager:
            try:
                self.data_lifecycle_manager.force_termination(DataTerminationReason.ALL_STAGES_COMPLETED)
                self.logger.info("✅ DataLifecycleManager terminated")
            except Exception as e:
                self.logger.error(f"❌ Error terminating DataLifecycleManager: {e}")
        
        self.logger.info("✅ TrainingManager graceful shutdown completed")