"""
V2 TrainingManager - Complete Training Lifecycle Manager
Combines V1 TrainingManager with integrated DataLifecycleManager functionality.
Adapted for V2 config system and callback-driven architecture.
"""

import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

from .interfaces import ITrainingManager
from ..core.types import TerminationReason, TrainingPhase
from ..core.shutdown import IShutdownHandler, ShutdownReason
from ..config.training.training_config import TrainingManagerConfig
from ..data.interfaces import IDataManager


class TrainingMode(Enum):
    """Training mode enumeration"""
    TRAINING = "training"
    OPTUNA = "optuna"
    BENCHMARK = "benchmark"


class DataTerminationReason(Enum):
    """Reasons for data lifecycle termination"""
    CYCLE_LIMIT_REACHED = "cycle_limit_reached"
    EPISODE_LIMIT_REACHED = "episode_limit_reached"
    UPDATE_LIMIT_REACHED = "update_limit_reached"
    NO_MORE_RESET_POINTS = "no_more_reset_points"
    NO_MORE_DAYS = "no_more_days" 
    DATE_RANGE_EXHAUSTED = "date_range_exhausted"
    QUALITY_CRITERIA_NOT_MET = "quality_criteria_not_met"
    ALL_STAGES_COMPLETED = "all_stages_completed"
    PRELOAD_FAILED = "preload_failed"


class SelectionMode(Enum):
    """Data selection modes"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    QUALITY_WEIGHTED = "quality_weighted"
    CURRICULUM_ORDERED = "curriculum_ordered"


@dataclass
class ResetPointInfo:
    """Information about a reset point"""
    timestamp: str
    quality_score: float
    roc_score: float
    activity_score: float
    price: float = 0.0
    used_count: int = 0
    last_used: Optional[datetime] = None
    
    def meets_criteria(self, quality_range: List[float], roc_range: List[float], 
                      activity_range: List[float]) -> bool:
        """Check if reset point meets selection criteria"""
        return (quality_range[0] <= self.quality_score <= quality_range[1] and
                roc_range[0] <= self.roc_score <= roc_range[1] and 
                activity_range[0] <= self.activity_score <= activity_range[1])


@dataclass
class DayInfo:
    """Information about a trading day"""
    date: str
    symbol: str
    day_score: float
    reset_points: List[ResetPointInfo] = field(default_factory=list)
    used_count: int = 0
    last_used: Optional[datetime] = None
    
    def get_available_reset_points(self, quality_range: List[float], 
                                  roc_range: List[float], activity_range: List[float],
                                  max_reuse: int = 3) -> List[ResetPointInfo]:
        """Get reset points that meet criteria and haven't been overused"""
        return [rp for rp in self.reset_points 
                if rp.meets_criteria(quality_range, roc_range, activity_range) 
                and rp.used_count < max_reuse]


@dataclass
class CycleState:
    """Current cycle state for data lifecycle"""
    stage_index: int = 0
    cycle_count: int = 0
    episode_count: int = 0
    update_count: int = 0
    total_cycles_completed: int = 0
    
    # Current selections
    current_day: Optional[DayInfo] = None
    current_reset_points: List[ResetPointInfo] = field(default_factory=list)
    
    # Tracking
    used_days: Set[str] = field(default_factory=set)
    used_reset_points: Set[str] = field(default_factory=set)
    
    # Progress tracking
    episodes_in_current_cycle: int = 0
    episodes_in_current_day: int = 0
    
    def reset_for_new_cycle(self):
        """Reset counters for new cycle"""
        self.cycle_count += 1
        self.total_cycles_completed += 1
        self.episodes_in_current_cycle = 0
        self.current_reset_points = []
    
    def reset_for_new_day(self):
        """Reset counters for new day"""
        self.episodes_in_current_day = 0
        self.current_day = None


@dataclass
class TrainingState:
    """Current training state - enhanced from V1"""
    episodes: int = 0
    updates: int = 0
    global_steps: int = 0
    training_hours: float = 0.0
    start_time: Optional[datetime] = None
    current_performance: float = 0.0
    best_performance: float = float('-inf')
    termination_votes: List[TerminationReason] = field(default_factory=list)
    current_phase: TrainingPhase = TrainingPhase.INITIALIZATION
    
    # Data lifecycle state (derived from cycle_state)
    current_stage: Optional[str] = None
    cycle_count: int = 0
    current_day: Optional[str] = None
    current_symbol: Optional[str] = None
    stage_progress: float = 0.0
    data_preload_ready: bool = False

    def __post_init__(self):
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
    """Controls training termination decisions - from V1"""
    
    def __init__(self, config: TrainingManagerConfig, mode: TrainingMode):
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Termination criteria
        self.termination_max_episodes = config.termination_max_episodes
        self.termination_max_updates = config.termination_max_updates
        self.termination_max_cycles = config.termination_max_cycles
        
        # Intelligent termination (only for training mode)
        self.enable_intelligent_termination = (
            mode == TrainingMode.TRAINING and
            config.intelligent_termination
        )
        self.plateau_patience = config.plateau_patience or 50
        self.degradation_threshold = config.degradation_threshold or 0.05
        
        # Performance tracking for intelligent termination
        self.performance_history: List[float] = []
        self.updates_since_improvement = 0
        self.best_performance = float('-inf')
        
    def should_terminate(self, state: TrainingState) -> Optional[TerminationReason]:
        """Single source of truth for termination decisions"""
        
        # Check hard limits (always enforced)
        if self.termination_max_episodes and state.episodes >= self.termination_max_episodes:
            return TerminationReason.MAX_EPISODES_REACHED
            
        if self.termination_max_updates and state.updates >= self.termination_max_updates:
            return TerminationReason.MAX_UPDATES_REACHED
            
        if self.termination_max_cycles and state.cycle_count >= self.termination_max_cycles:
            return TerminationReason.MAX_CYCLES_REACHED
        
        # Check external termination votes
        if state.termination_votes:
            return state.termination_votes[0]  # Take first vote
        
        # Intelligent termination (only in training mode)
        if self.enable_intelligent_termination and self.mode == TrainingMode.TRAINING:
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
                f"=� Performance plateau detected: {self.updates_since_improvement} updates without improvement"
            )
            return TerminationReason.PERFORMANCE_PLATEAU
        
        # Check for performance degradation
        if len(self.performance_history) >= 50:
            recent_performance = sum(self.performance_history[-10:]) / 10
            older_performance = sum(self.performance_history[-50:-40]) / 10
            
            if older_performance > 0 and (recent_performance - older_performance) / older_performance < -self.degradation_threshold:
                self.logger.info(
                    f"=� Performance degradation detected: {recent_performance:.4f} vs {older_performance:.4f}"
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


class TrainingManager(ITrainingManager, IShutdownHandler):
    """
    V2 TrainingManager - Complete Training Lifecycle Manager
    
    Combines V1 TrainingManager functionality with integrated DataLifecycleManager.
    Central authority for training lifecycle based on proven V1 patterns.
    
    Responsibilities:
    - Execute main training loop (episodes, updates, termination)
    - Manage integrated data lifecycle (day selection, reset point cycling)
    - Control termination conditions and training state
    - Coordinate with callbacks for feature-specific behavior
    - Handle graceful termination and cleanup
    """
    
    def __init__(self, config: TrainingManagerConfig):
        """Initialize TrainingManager with V2 configuration."""
        self.config = config
        self.mode = TrainingMode(config.mode)
        self.logger = logging.getLogger(f"{__name__}.TrainingManager")
        
        # Core state
        self.state = TrainingState()
        self.should_stop = False
        self.termination_reason: Optional[TerminationReason] = None
        
        # Termination controller
        self.termination_controller = TerminationController(config, self.mode)
        
        # Integrated data lifecycle (replaces external DataLifecycleManager)
        self.cycle_state = CycleState()
        self.available_days: List[DayInfo] = []
        self.current_reset_point = None
        self.current_day_episodes = 0
        self.current_day_updates = 0
        
        # Reset point cycler state
        self.ordered_reset_points: List[ResetPointInfo] = []
        self.current_reset_index = 0
        self.current_reset_cycle = 0
        
        # Component references (set during start_training)
        self.data_manager = None
        self.trainer = None
        self.environment = None
        self.callback_manager = None
        
        # Register with shutdown manager
        from ..core.shutdown import get_global_shutdown_manager
        shutdown_manager = get_global_shutdown_manager()
        shutdown_manager.register_component(
            component=self,
            timeout=120.0
        )
        
        self.logger.info(f"=' TrainingManager initialized in {self.mode.value} mode")

    # ITrainingManager implementation
    
    def start(self, trainer: Any, environment: Any, data_manager: Any, callback_manager: Any) -> Dict[str, Any]:
        """Start the main training loop - core of the system."""
        
        self.logger.info(f"TrainingManager started in {self.mode.value} mode")
        
        # Store component references
        self.trainer = trainer
        self.environment = environment
        self.data_manager:IDataManager = data_manager
        self.callback_manager = callback_manager
        
        # Initialize training state
        self.state = TrainingState()

        start_date: Optional[str] = None
        end_date: Optional[str] = None
        if(self.config.date_range) and len(self.config.date_range) > 2:
            start_date = self.config.date_range[0]
            end_date = self.config.date_range[1]

        self.logger.info(f"🔍 Loading momentum days with filters:")
        self.logger.info(f"   📊 Symbols: {self.config.symbols}")
        self.logger.info(f"   📅 Date range: {start_date or 'None'} to {end_date or 'None'}")

        momentum_days_dicts =self.data_manager.get_momentum_days(
            symbols = self.config.symbols,
            start_date=start_date,
            end_date=end_date,
            min_activity = 0.0, # Todo ? Do we filter based on logic here or from source ?
        )
        self.logger.info(f"   📊 Found {len(momentum_days_dicts)} momentum days.")

        for day_dict in momentum_days_dicts:
            reset_points = []
            reset_points_df = self.data_manager.get_reset_points(day_dict["symbol"], day_dict["date"])
            for _, rp_row in reset_points_df.iterrows():
                reset_point = ResetPointInfo(
                    timestamp=rp_row['timestamp'],
                    quality_score=rp_row['quality_score'],
                    roc_score=rp_row['roc_score'],
                    activity_score=rp_row['activity_score'],
                    price=rp_row['price']
                )
                reset_points.append(reset_point)

            day_info = DayInfo(
                date=day_dict["date"],
                symbol=day_dict["symbol"],
                day_score=day_dict["quality_score"],
                reset_points=reset_points
            )
            self.available_days.append(day_info) #

        try:
            if not self._initialize_data_lifecycle():
                return self._finalize_training()

            # Initialize callbacks
            context = self._create_callback_context()
            self.callback_manager.on_training_start(context)
            
            while not self.should_terminate():
                # Update training state from trainer
                self._update_training_state()
                
                # Check data lifecycle termination first (highest priority)
                data_termination = self._should_terminate_data_lifecycle()
                if data_termination:
                    self._request_termination_internal(self._map_data_termination(data_termination))
                    break
                
                # Check training termination conditions
                training_termination = self.termination_controller.should_terminate(self.state)
                if training_termination:
                    self._request_termination_internal(training_termination)
                    break
                
                # Get current episode configuration
                episode_config = self.get_current_episode_config()
                if not episode_config:
                    self.logger.warning("No episode configuration available")
                    self._request_termination_internal(TerminationReason.DATA_EXHAUSTED)
                    break
                
                should_continue = self.trainer.run_training_step()
                if not should_continue:
                    self._request_termination_internal(TerminationReason.TRAINER_STOPPED)
                    break
                
                # Let callbacks handle their features
                context = self._create_callback_context()
                self.callback_manager.on_episode_end(context)
                
                # Check episode advancement (integrated data lifecycle)
                self._advance_cycle_on_episode_completion()

        return self._finalize_training()

    def should_terminate(self) -> bool:
        """Check if training should terminate."""
        # External termination request
        if self.should_stop:
            return True
        
        # Termination controller checks hard limits and intelligent termination
        termination_reason = self.termination_controller.should_terminate(self.state)
        if termination_reason:
            self._request_termination_internal(termination_reason)
            return True
        
        return False

    def get_current_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring and callbacks."""
        return {
            'episodes': self.state.episodes,
            'updates': self.state.updates,
            'global_steps': self.state.global_steps,
            'training_hours': self.state.training_hours,
            'current_performance': self.state.current_performance,
            'best_performance': self.state.best_performance,
            'current_phase': self.state.current_phase.value,
            'current_day': self.state.current_day,
            'current_symbol': self.state.current_symbol,
            'cycle_count': self.state.cycle_count,
            'stage_progress': self.state.stage_progress,
            'mode': self.mode.value,
            'termination_reason': self.termination_reason.value if self.termination_reason else None
        }

    def get_current_episode_config(self) -> Optional[Dict[str, Any]]:
        """Get current episode configuration from integrated data lifecycle."""
        return self._get_current_training_data()

    def request_termination(self, reason: TerminationReason) -> None:
        """Request graceful training termination (public interface)."""
        self.state.termination_votes.append(reason)
        self.logger.info(f"=� Termination requested: {reason.value}")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'final_performance': self.state.current_performance,
            'best_performance': self.state.best_performance,
            'total_episodes': self.state.episodes,
            'total_updates': self.state.updates,
            'total_steps': self.state.global_steps,
            'training_duration_hours': self.state.training_hours,
            'termination_reason': self.termination_reason.value if self.termination_reason else None,
            'mode': self.mode.value,
            'current_day': self.state.current_day,
            'current_symbol': self.state.current_symbol,
            'cycle_count': self.state.cycle_count
        }
        
        # Add integrated data lifecycle statistics
        data_stats = self._get_data_lifecycle_statistics()
        stats.update(data_stats)
        
        return stats

    # Compatibility method for main.py integration
    def start_mode(
        self,
        mode_type: Any,
        config: Dict[str, Any],
        trainer: Any,
        environment: Any,
        background: bool = False
    ) -> Dict[str, Any]:
        """Start training in specified mode (compatibility interface)."""
        self.logger.info(f"<� Starting {mode_type} mode")
        
        # For now, all modes use the same training loop
        # Mode-specific behavior is controlled by configuration and callbacks
        return self.start_training(
            trainer=trainer,
            environment=environment,
            data_manager=self.data_manager,
            callback_manager=self.callback_manager
        )

    # Internal methods
    
    def _request_termination_internal(self, reason: TerminationReason) -> None:
        """Internal termination request."""
        self.should_stop = True
        self.termination_reason = reason
        self.logger.info(f"<� Training terminated: {reason.value}")

    def _map_data_termination(self, data_reason: DataTerminationReason) -> TerminationReason:
        """Map data termination reason to training termination reason."""
        mapping = {
            DataTerminationReason.CYCLE_LIMIT_REACHED: TerminationReason.MAX_CYCLES_REACHED,
            DataTerminationReason.EPISODE_LIMIT_REACHED: TerminationReason.MAX_EPISODES_REACHED,
            DataTerminationReason.UPDATE_LIMIT_REACHED: TerminationReason.MAX_UPDATES_REACHED,
            DataTerminationReason.NO_MORE_RESET_POINTS: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.NO_MORE_DAYS: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.DATE_RANGE_EXHAUSTED: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.QUALITY_CRITERIA_NOT_MET: TerminationReason.DATA_EXHAUSTED,
            DataTerminationReason.ALL_STAGES_COMPLETED: TerminationReason.MAX_CYCLES_REACHED,
            DataTerminationReason.PRELOAD_FAILED: TerminationReason.DATA_EXHAUSTED,
        }
        return mapping.get(data_reason, TerminationReason.DATA_EXHAUSTED)

    def _update_training_state(self):
        # Update counters from trainer
        self.state.episodes = getattr(self.trainer, 'global_episode_counter', 0)
        self.state.updates = getattr(self.trainer, 'global_update_counter', 0)
        self.state.global_steps = getattr(self.trainer, 'global_step_counter', 0)
        
        # Calculate training time
        if self.state.start_time:
            elapsed = datetime.now() - self.state.start_time
            self.state.training_hours = elapsed.total_seconds() / 3600
        
        # Update integrated data lifecycle state
        self._update_data_lifecycle_progress(self.state.episodes, self.state.updates)
        
        # Get data lifecycle status from cycle state
        if self.cycle_state.current_day:
            self.state.current_day = self.cycle_state.current_day.date
            self.state.current_symbol = self.cycle_state.current_day.symbol
            self.state.cycle_count = self.cycle_state.cycle_count
        
        # Update performance metrics
        performance_metrics = self._get_performance_metrics()
        if performance_metrics and 'mean_reward' in performance_metrics:
            self.state.current_performance = performance_metrics['mean_reward']
            self.state.best_performance = max(
                self.state.best_performance, self.state.current_performance
            )
            
            # Update termination controller
            self.termination_controller.update_performance(
                self.state.current_performance, self.state.updates
            )

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from trainer."""
        if not self.trainer:
            return {}
        
        return {
            'mean_reward': getattr(self.trainer, 'mean_episode_reward', 0.0),
            'episodes': self.state.episodes,
            'updates': self.state.updates,
            'global_steps': self.state.global_steps
        }


    def _advance_cycle_on_episode_completion(self):
        """Advance to next reset point after episode completion."""
        self.logger.info("= Episode completed - advancing to next reset point")
        result = self._advance_to_next_cycle()
        if result:
            self.logger.info(f" Advanced to reset point at {self.current_reset_point.timestamp}")
        else:
            self.logger.warning("L Failed to advance to next reset point")
        return result

    def _create_callback_context(self) -> Dict[str, Any]:
        """Create context dictionary for callbacks."""
        return {
            'training_state': self.get_current_training_state(),
            'performance_metrics': self._get_performance_metrics(),
            'config': self.config,
            'trainer': self.trainer,
            'environment': self.environment,
            'data_manager': self.data_manager,
            'mode': self.mode.value,
            'episode_config': self.get_current_episode_config()
        }

    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and return statistics."""
        self.state.current_phase = TrainingPhase.TERMINATION
        
        # Finalize callbacks
        if self.callback_manager:
            context = self._create_callback_context()
            self.callback_manager.on_training_end(context)
        
        # Get final statistics
        final_stats = self.get_training_statistics()
        
        self.logger.info("<� TRAINING LIFECYCLE COMPLETE")
        self.logger.info(f"   =� Final Performance: {self.state.current_performance:.4f}")
        self.logger.info(f"   <� Best Performance: {self.state.best_performance:.4f}")
        self.logger.info(f"   =� Episodes: {self.state.episodes}")
        self.logger.info(f"   = Updates: {self.state.updates}")
        self.logger.info(f"   � Duration: {self.state.training_hours:.2f}h")
        if self.termination_reason:
            self.logger.info(f"   =� Reason: {self.termination_reason.value}")
        
        return final_stats

    # Integrated Data Lifecycle Methods (from V1 DataLifecycleManager)

    def _initialize_data_lifecycle(self) -> bool:
        """Initialize integrated data lifecycle - select first day and reset points."""
        try:
            # Load available days from data manager
            if hasattr(self.data_manager, 'get_available_days'):
                self.available_days = self.data_manager.get_available_days()
            elif hasattr(self.data_manager, 'available_days'):
                self.available_days = self.data_manager.available_days
            else:
                self.logger.warning("No available days found in data manager")
                return False

            if not self.available_days:
                self.logger.error("No available training days")
                return False

            # Initialize cycle state
            self.cycle_state = CycleState()

            # Select first day and reset points
            if not self._advance_to_next_day():
                self.logger.error("Failed to select initial day")
                return False
                
            # Log final summary
            day_date = self.cycle_state.current_day.date
            symbol = self.cycle_state.current_day.symbol
            reset_count = len(self.cycle_state.current_reset_points)
            quality = self.cycle_state.current_day.day_score
            
            self.logger.info("Selected: %s %s (quality: %.3f)", symbol, day_date, quality)
            self.logger.info("Reset points: %d available", reset_count)
            return True
        except Exception as e:
            self.logger.error("Failed to initialize data lifecycle: %s", e)
            return False

    def _should_terminate_data_lifecycle(self) -> Optional[DataTerminationReason]:
        """Check if data lifecycle should terminate."""
        # Check if should switch to next day based on cycle config
        if self._should_switch_day():
            if not self._advance_to_next_day():
                self.logger.info("= No more days available, but checking training termination first")
                self.logger.info("= No more days available, continuing with current day reset points")
                # Reset the index but keep the cycle count for training termination
                self.current_reset_index = 0
        
        return None

    def _get_current_training_data(self) -> Optional[Dict[str, Any]]:
        """Get current training data configuration."""
        if not self.cycle_state.current_day or not self.cycle_state.current_reset_points:
            return None
        
        # Format data for trainer compatibility
        day_info = {
            'date': self.cycle_state.current_day.date,
            'symbol': self.cycle_state.current_day.symbol,
            'quality_score': self.cycle_state.current_day.day_score,
        }
        
        # Get current reset point index
        current_reset_point_index = max(0, self.current_reset_index - 1)
        
        # If we have a current reset point, find its index in the list
        if self.current_reset_point:
            for i, rp in enumerate(self.ordered_reset_points):
                if rp.timestamp == self.current_reset_point.timestamp:
                    current_reset_point_index = i
                    break
        
        # Format ALL reset points for display
        reset_points_formatted = []
        for rp in self.ordered_reset_points:
            reset_point_dict = {
                'timestamp': rp.timestamp,
                'quality_score': rp.quality_score,
                'roc_score': rp.roc_score,
                'activity_score': rp.activity_score,
                'combined_score': rp.quality_score,
                'price': rp.price
            }
            reset_points_formatted.append(reset_point_dict)
        
        return {
            'day_info': day_info,
            'reset_points': reset_points_formatted,
            'reset_point_index': current_reset_point_index,
            'stage': 'integrated_data_lifecycle'
        }

    def _update_data_lifecycle_progress(self, episodes: int, updates: int):
        """Update progress counters for integrated data lifecycle."""
        # Update global counters
        self.cycle_state.episode_count = episodes
        self.cycle_state.update_count = updates
        
        # Update local counters
        self.cycle_state.episodes_in_current_cycle += 1
        self.cycle_state.episodes_in_current_day += 1
        
        # Update day-level counters for switching logic
        self.current_day_episodes += 1
        
        self.logger.debug(f"Progress updated: episodes={episodes}, updates={updates}, day_episodes={self.current_day_episodes}")

    def _should_switch_day(self) -> bool:
        """Check if should switch to next day based on config."""
        # Get cycle limits from config
        day_max_episodes = getattr(self.config, 'day_max_episodes', None)
        day_max_updates = getattr(self.config, 'day_max_updates', None) 
        day_max_cycles = getattr(self.config, 'day_max_cycles', 3)
        
        # Check if any day limit is reached
        if day_max_episodes and self.current_day_episodes >= day_max_episodes:
            self.logger.info(f"= Day episode limit reached: {self.current_day_episodes}/{day_max_episodes}")
            return True
        
        if day_max_updates and self.current_day_updates >= day_max_updates:
            self.logger.info(f"= Day update limit reached: {self.current_day_updates}/{day_max_updates}")
            return True
        
        if day_max_cycles and self.current_reset_cycle >= day_max_cycles:
            self.logger.info(f"= Day cycle limit reached: {self.current_reset_cycle}/{day_max_cycles}")
            return True
        
        return False

    def _advance_to_next_day(self) -> bool:
        """Advance to next day - select new day and reset points."""
        # Select next day using simple criteria
        next_day = self._select_next_day()
        if not next_day:
            self.logger.warning("No suitable days available")
            return False
        
        # Reset day-level counters
        self.current_day_episodes = 0
        self.current_day_updates = 0
        
        # Update state
        self.cycle_state.reset_for_new_day()
        self.cycle_state.current_day = next_day
        
        # Track usage
        self.cycle_state.used_days.add(next_day.date)
        
        # Initialize reset point cycler for this day
        if not self._initialize_day_reset_points(next_day):
            self.logger.warning(f"Failed to initialize reset points for day {next_day.date}")
            return False
        
        # Get first reset point
        if not self._advance_to_next_cycle():
            return False
        
        self.logger.info(f"<� Advanced to new day: {next_day.date} ({next_day.symbol})")
        return True

    def _select_next_day(self) -> Optional[DayInfo]:
        """Select next training day based on simple criteria."""
        if not self.available_days:
            return None
        
        # Filter unused days first
        unused_days = [day for day in self.available_days if day.date not in self.cycle_state.used_days]
        if unused_days:
            filtered_days = unused_days
        else:
            # Reset and use all days if we've used them all
            self.cycle_state.used_days.clear()
            filtered_days = self.available_days
        
        # Simple selection - take first available day
        selected = filtered_days[0]
        
        # Mark as used
        selected.used_count += 1
        selected.last_used = datetime.now()
        
        return selected

    def _initialize_day_reset_points(self, day: DayInfo) -> bool:
        """Initialize reset point order for a new day."""
        # Get all reset points for this day (simple - just use all)
        available = day.reset_points.copy()
        
        if not available:
            self.logger.warning(f"No reset points for day {day.date}")
            return False
        
        # Order by quality score (highest first)
        self.ordered_reset_points = sorted(available, key=lambda rp: rp.quality_score, reverse=True)
        self.current_reset_cycle = 0
        self.current_reset_index = 0
        
        self.logger.info(f"= Initialized {len(self.ordered_reset_points)} reset points for day {day.date}")
        return True

    def _advance_to_next_cycle(self) -> bool:
        """Advance to next reset point within current day."""
        if not self.cycle_state.current_day:
            return False
        
        # Track previous cycle count to detect when a full cycle is completed
        previous_cycle_count = self.current_reset_cycle
        
        # Get next reset point from cycler
        next_reset_point = self._get_next_reset_point()
        if not next_reset_point:
            self.logger.warning(f"No more reset points available for day {self.cycle_state.current_day.date}")
            return False
        
        # Check if we completed a full cycle (went through all reset points)
        cycle_completed = self.current_reset_cycle > previous_cycle_count
        
        # Only update cycle state when a full cycle through ALL reset points is completed
        if cycle_completed:
            self.cycle_state.reset_for_new_cycle()
            self.logger.info(f"= Completed full cycle {self.current_reset_cycle}, total cycles: {self.cycle_state.total_cycles_completed}")
        else:
            # Just increment episode counter for this cycle, but don't count as new cycle
            self.cycle_state.episodes_in_current_cycle += 1
        
        self.current_reset_point = next_reset_point
        self.cycle_state.current_reset_points = [next_reset_point]
        
        # Track usage
        self.cycle_state.used_reset_points.add(next_reset_point.timestamp)
        
        self.logger.info(
            f"= Advanced to reset point {self.current_reset_index}/{len(self.ordered_reset_points)} "
            f"(cycle {self.current_reset_cycle}) in day {self.cycle_state.current_day.date}"
        )
        return True

    def _get_next_reset_point(self) -> Optional[ResetPointInfo]:
        """Get next reset point in the cycle."""
        if not self.ordered_reset_points:
            return None
        
        # Get current reset point
        reset_point = self.ordered_reset_points[self.current_reset_index]
        
        # Advance to next
        self.current_reset_index += 1
        
        # Check if we completed a full cycle through all reset points
        if self.current_reset_index >= len(self.ordered_reset_points):
            self.current_reset_index = 0
            self.current_reset_cycle += 1
            self.logger.info(f"= Completed cycle {self.current_reset_cycle}, starting over")
        
        return reset_point

    def _get_data_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get integrated data lifecycle statistics."""
        return {
            'data_lifecycle': {
                'total_cycles': self.cycle_state.total_cycles_completed,
                'current_cycle': self.cycle_state.cycle_count,
                'current_day': self.cycle_state.current_day.date if self.cycle_state.current_day else None,
                'current_symbol': self.cycle_state.current_day.symbol if self.cycle_state.current_day else None,
                'used_days_count': len(self.cycle_state.used_days),
                'used_reset_points_count': len(self.cycle_state.used_reset_points)
            }
        }

    # IShutdownHandler implementation

    def shutdown(self) -> None:
        """Perform graceful shutdown - stop training and cleanup resources."""
        self.logger.info("=� Shutting down TrainingManager")
        
        try:
            # Request termination
            self.request_termination(TerminationReason.SHUTDOWN_REQUESTED)
            
            # Clean up components
            self.trainer = None
            self.environment = None
            self.callback_manager = None
            self.data_manager = None
            
        except Exception as e:
            self.logger.error(f"L Error during TrainingManager shutdown: {e}")