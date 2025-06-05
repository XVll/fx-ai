"""
Data Lifecycle Manager - Authority for Training Data Management
Handles reset point cycling, day selection, stage management, and data-related termination.
"""

import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum


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
class StageConfig:
    """Configuration for a training stage"""
    name: str
    symbols: List[str] = field(default_factory=list)
    date_range: List[Optional[str]] = field(default_factory=lambda: [None, None])
    day_score_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    roc_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    activity_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    
    # Stage transition conditions
    max_episodes: Optional[int] = None
    max_updates: Optional[int] = None
    max_cycles: Optional[int] = None
    
    # Data management settings
    episodes_per_day: int = 1
    reset_points_per_cycle: int = 1
    max_reset_point_reuse: int = 3
    selection_mode: SelectionMode = SelectionMode.SEQUENTIAL
    randomize_order: bool = False
    preload_threshold: float = 0.8  # Start preloading when 80% complete


@dataclass
class CycleState:
    """Current cycle state"""
    stage_index: int = 0
    cycle_count: int = 0
    episode_count: int = 0
    update_count: int = 0
    
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
        self.episodes_in_current_cycle = 0
        self.current_reset_points = []
    
    def reset_for_new_day(self):
        """Reset counters for new day"""
        self.episodes_in_current_day = 0
        self.current_day = None


@dataclass
class PreloadState:
    """State of data preloading"""
    next_stage_ready: bool = False
    preload_progress: float = 0.0
    preload_error: Optional[str] = None
    next_days: List[DayInfo] = field(default_factory=list)


class ResetPointCycler:
    """Manages reset point cycling within days - cycles through ALL reset points"""
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        # Handle both old and new config names for backward compatibility
        if hasattr(config, 'reset_point_selection_mode'):
            self.selection_mode = SelectionMode(config.reset_point_selection_mode)
        elif hasattr(config, 'selection_mode'):
            self.selection_mode = SelectionMode(config.selection_mode)
        else:
            self.selection_mode = SelectionMode.SEQUENTIAL
        
        # Track state for cycling through all reset points
        self.current_cycle = 0
        self.current_index = 0
        self.ordered_reset_points = []
        
    def initialize_day(self, day: DayInfo, quality_criteria: Dict[str, Any]) -> bool:
        """Initialize reset point order for a new day"""
        # Get all suitable reset points for this day
        available = []
        for rp in day.reset_points:
            if rp.meets_criteria(
                quality_criteria['day_score_range'],
                quality_criteria['roc_range'], 
                quality_criteria['activity_range']
            ):
                available.append(rp)
        
        if not available:
            self.logger.warning(f"No suitable reset points for day {day.date}")
            return False
        
        # Order them based on selection mode
        if self.selection_mode == SelectionMode.SEQUENTIAL:
            # Order by quality score (highest first)
            self.ordered_reset_points = sorted(available, key=lambda rp: rp.quality_score, reverse=True)
        else:  # RANDOM
            # Randomize order but still go through each once
            self.ordered_reset_points = available.copy()
            random.shuffle(self.ordered_reset_points)
        
        self.current_cycle = 0
        self.current_index = 0
        
        self.logger.info(f"🔄 Initialized {len(self.ordered_reset_points)} reset points for day {day.date} in {self.selection_mode.value} mode")
        return True
        
    def get_next_reset_point(self) -> Optional[ResetPointInfo]:
        """Get next reset point in the cycle"""
        if not self.ordered_reset_points:
            return None
        
        # Get current reset point
        reset_point = self.ordered_reset_points[self.current_index]
        
        # Advance to next
        self.current_index += 1
        
        # Check if we completed a full cycle through all reset points
        if self.current_index >= len(self.ordered_reset_points):
            self.current_index = 0
            self.current_cycle += 1
            self.logger.info(f"🔄 Completed cycle {self.current_cycle}, starting over")
            
            # If random mode, re-shuffle for next cycle
            if self.selection_mode == SelectionMode.RANDOM:
                random.shuffle(self.ordered_reset_points)
                self.logger.debug("🔀 Re-shuffled reset points for next cycle")
        
        return reset_point
    
    def get_cycle_status(self) -> Dict[str, Any]:
        """Get current cycle status"""
        return {
            'current_cycle': self.current_cycle,
            'current_index': self.current_index,
            'total_reset_points': len(self.ordered_reset_points),
            'progress_in_cycle': self.current_index / max(1, len(self.ordered_reset_points))
        }


class DaySelector:
    """Manages day selection and quality filtering"""
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        # Handle both old and new config names for backward compatibility
        if hasattr(config, 'day_selection_mode'):
            self.selection_mode = SelectionMode(config.day_selection_mode)
        elif hasattr(config, 'selection_mode'):
            self.selection_mode = SelectionMode(config.selection_mode)
        else:
            self.selection_mode = SelectionMode.SEQUENTIAL
        
        # Remove randomize_order - it was confusing and unnecessary
        self.randomize_order = False
        
    def select_day(self, available_days: List[DayInfo], stage_config: StageConfig,
                   used_days: Set[str]) -> Optional[DayInfo]:
        """Select next training day based on criteria"""
        
        # Filter by criteria
        filtered_days = []
        self.logger.info(f"📅 Filtering {len(available_days)} available days for date range {stage_config.date_range}")
        
        # Show all available days first
        all_day_dates = [day.date if isinstance(day.date, str) else (day.date.strftime('%Y-%m-%d') if hasattr(day.date, 'strftime') else str(day.date)) for day in available_days]
        self.logger.info(f"📅 Available days in momentum index: {sorted(all_day_dates)}")
        
        for day in available_days:
            day_date_str = day.date if isinstance(day.date, str) else (day.date.strftime('%Y-%m-%d') if hasattr(day.date, 'strftime') else str(day.date))
            
            # Check date range
            if self._day_in_date_range(day, stage_config.date_range):
                # Check symbol
                if not stage_config.symbols or day.symbol in stage_config.symbols:
                    # Check day quality
                    if (stage_config.day_score_range[0] <= day.day_score <= stage_config.day_score_range[1]):
                        # Check if has available reset points
                        available_rps = day.get_available_reset_points(
                            stage_config.day_score_range,
                            stage_config.roc_range,
                            stage_config.activity_range
                        )
                        if available_rps:
                            filtered_days.append(day)
                            self.logger.debug(f"📅 ✅ Day {day.symbol} {day_date_str} passes all filters (quality: {day.day_score:.3f}, reset_points: {len(available_rps)})")
                        else:
                            self.logger.debug(f"📅 ❌ Day {day.symbol} {day_date_str} has no available reset points")
                    else:
                        self.logger.debug(f"📅 ❌ Day {day.symbol} {day_date_str} quality {day.day_score:.3f} outside range {stage_config.day_score_range}")
                else:
                    self.logger.debug(f"📅 ❌ Day {day.symbol} {day_date_str} symbol not in {stage_config.symbols}")
            else:
                self.logger.debug(f"📅 ❌ Day {day.symbol} {day_date_str} outside date range {stage_config.date_range}")
        
        if not filtered_days:
            self.logger.warning("No days meet current stage criteria")
            return None
        
        # Remove recently used (if not randomized)
        if not self.randomize_order:
            unused_days = [day for day in filtered_days if day.date not in used_days]
            if unused_days:
                filtered_days = unused_days
        
        # Select based on mode
        if self.selection_mode == SelectionMode.RANDOM:
            selected = random.choice(filtered_days)
        elif self.selection_mode == SelectionMode.QUALITY_WEIGHTED:
            weights = [day.day_score for day in filtered_days]
            selected = random.choices(filtered_days, weights=weights)[0]
        else:  # SEQUENTIAL or CURRICULUM_ORDERED
            # For sequential mode, sort by date (earliest first) to respect date range order
            # Parse dates for proper sorting
            def parse_date(day):
                if isinstance(day.date, str):
                    return datetime.strptime(day.date, "%Y-%m-%d").date()
                elif hasattr(day.date, 'date'):
                    return day.date.date()
                else:
                    return day.date
            
            filtered_days.sort(key=parse_date)
            selected = filtered_days[0]
            
            # Log the selection process
            all_dates = [parse_date(day).strftime('%Y-%m-%d') if hasattr(parse_date(day), 'strftime') else str(parse_date(day)) for day in filtered_days]
            self.logger.info(f"📅 Sequential selection from {len(filtered_days)} filtered days: {all_dates}")
            selected_date = parse_date(selected)
            selected_date_str = selected_date.strftime('%Y-%m-%d') if hasattr(selected_date, 'strftime') else str(selected_date)
            self.logger.info(f"📅 Selected earliest date: {selected_date_str} (quality: {selected.day_score:.3f})")
        
        # Mark as used
        selected.used_count += 1
        selected.last_used = datetime.now()
        
        return selected
    
    def _day_in_date_range(self, day: DayInfo, date_range: List[Optional[str]]) -> bool:
        """Check if day falls within date range"""
        if not date_range[0] and not date_range[1]:
            return True
        
        # Handle both string and Timestamp inputs for day.date
        if hasattr(day.date, 'date'):
            # It's a pandas Timestamp or datetime object
            day_date = day.date.date()
        elif isinstance(day.date, str):
            # It's a string, parse it
            day_date = datetime.strptime(day.date, "%Y-%m-%d").date()
        else:
            # It's already a date object
            day_date = day.date
        
        if date_range[0]:
            start_date = datetime.strptime(date_range[0], "%Y-%m-%d").date()
            if day_date < start_date:
                return False
        
        if date_range[1]:
            end_date = datetime.strptime(date_range[1], "%Y-%m-%d").date()
            if day_date > end_date:
                return False
        
        return True


class StageManager:
    """Manages progression through training stages"""
    
    def __init__(self, stages: List[StageConfig]):
        self.stages = stages
        self.current_stage_index = 0
        self.logger = logging.getLogger(__name__)
        
    def get_current_stage(self) -> Optional[StageConfig]:
        """Get current training stage"""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None
    
    def should_advance_stage(self, cycle_state: CycleState) -> bool:
        """Check if should advance to next stage"""
        current_stage = self.get_current_stage()
        if not current_stage:
            return False
        
        # Check episode limit
        if (current_stage.max_episodes and 
            cycle_state.episode_count >= current_stage.max_episodes):
            return True
        
        # Check update limit  
        if (current_stage.max_updates and
            cycle_state.update_count >= current_stage.max_updates):
            return True
        
        # Check cycle limit
        if (current_stage.max_cycles and
            cycle_state.cycle_count >= current_stage.max_cycles):
            return True
        
        return False
    
    def advance_stage(self) -> bool:
        """Advance to next stage"""
        if self.current_stage_index + 1 < len(self.stages):
            self.current_stage_index += 1
            self.logger.info(f"🎯 Advanced to stage {self.current_stage_index + 1}: {self.stages[self.current_stage_index].name}")
            return True
        else:
            self.logger.info("🏁 All stages completed")
            return False
    
    def get_stage_progress(self, cycle_state: CycleState) -> float:
        """Get progress through current stage (0.0 to 1.0)"""
        current_stage = self.get_current_stage()
        if not current_stage:
            return 1.0
        
        progress_values = []
        
        if current_stage.max_episodes:
            progress_values.append(cycle_state.episode_count / current_stage.max_episodes)
        
        if current_stage.max_updates:
            progress_values.append(cycle_state.update_count / current_stage.max_updates)
        
        if current_stage.max_cycles:
            progress_values.append(cycle_state.cycle_count / current_stage.max_cycles)
        
        return max(progress_values) if progress_values else 0.0


class DataPreloader:
    """Handles preloading of next stage data"""
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.enabled = config.preload_enabled
        
    def should_preload(self, stage_progress: float, stage_config: StageConfig) -> bool:
        """Check if should start preloading next stage"""
        return (self.enabled and 
                stage_progress >= stage_config.preload_threshold)
    
    def preload_next_stage(self, next_stage: StageConfig, 
                          available_days: List[DayInfo]) -> PreloadState:
        """Preload data for next stage"""
        try:
            # Filter days for next stage
            suitable_days = []
            day_selector = DaySelector({})
            
            for day in available_days:
                if day_selector._day_in_date_range(day, next_stage.date_range):
                    if not next_stage.symbols or day.symbol in next_stage.symbols:
                        if (next_stage.day_score_range[0] <= day.day_score <= next_stage.day_score_range[1]):
                            suitable_days.append(day)
            
            if not suitable_days:
                return PreloadState(
                    next_stage_ready=False,
                    preload_error="No suitable days found for next stage"
                )
            
            self.logger.info(f"✅ Preloaded {len(suitable_days)} days for next stage")
            return PreloadState(
                next_stage_ready=True,
                preload_progress=1.0,
                next_days=suitable_days
            )
            
        except Exception as e:
            self.logger.error(f"❌ Preload failed: {e}")
            return PreloadState(
                next_stage_ready=False,
                preload_error=str(e)
            )


class DataLifecycleManager:
    """
    Authority for Training Data Management
    
    Responsibilities:
    - Reset point cycling and tracking
    - Day selection and quality management
    - Stage progression and management
    - Cycle counting (episodes, updates, cycles)
    - Data preloading for smooth transitions
    - Edge case handling (no data, exhausted ranges)
    - Data-specific termination conditions
    """
    
    def __init__(self, config, available_days: List[DayInfo]):
        self.config = config
        self.available_days = available_days
        self.logger = logging.getLogger(__name__)
        
        # Initialize components - handle both old and new config structure
        if hasattr(config, 'reset_points'):
            # Old config structure
            self.reset_point_cycler = ResetPointCycler(config.reset_points)
        else:
            # New config structure - use adaptive_data
            self.reset_point_cycler = ResetPointCycler(config.adaptive_data)
            
        if hasattr(config, 'day_selection'):
            # Old config structure
            self.day_selector = DaySelector(config.day_selection)
        else:
            # New config structure - use adaptive_data
            self.day_selector = DaySelector(config.adaptive_data)
        
        # Initialize preloader
        self.preloader = DataPreloader(config.preloading)
        
        # State management
        self.cycle_state = CycleState()
        self.preload_state = PreloadState()
        self.should_terminate = False
        self.termination_reason: Optional[DataTerminationReason] = None
        
        # Day switching counters based on DataCycleConfig
        self.current_day_episodes = 0
        self.current_day_updates = 0
        
        # Current reset point from cycler
        self.current_reset_point = None
        
        # Constructor initialization is silent - details logged during initialize()
    
    def initialize(self) -> bool:
        """Initialize data lifecycle - select first day and reset points"""
        try:
            # First select initial day (this already selects the first reset point)
            if not self._advance_to_next_day():
                self.logger.error("│   └── ❌ Failed to select initial day")
                return False
                
            # Log final summary
            day_date = self.cycle_state.current_day.date  # Already a string
            symbol = self.cycle_state.current_day.symbol
            reset_count = len(self.cycle_state.current_reset_points)
            quality = self.cycle_state.current_day.day_score
            
            self.logger.info("│   ├── 📅 Selected: %s %s (quality: %.3f)", symbol, day_date, quality)
            self.logger.info("│   └── 🔄 Reset points: %d available", reset_count)
            return True
        except Exception as e:
            self.logger.error("│   └── ❌ Failed to initialize data lifecycle: %s", e)
            return False
    
    def should_terminate_data_lifecycle(self) -> Optional[DataTerminationReason]:
        """Check if data lifecycle should terminate"""
        if self.should_terminate:
            return self.termination_reason
        
        # Check if should switch to next day based on cycle config
        if self._should_switch_day():
            if not self._advance_to_next_day():
                return DataTerminationReason.NO_MORE_DAYS
        
        # Note: Cycle advancement is now handled explicitly via advance_cycle_on_episode_completion()
        # Don't automatically advance here
        
        return None
    
    def get_current_training_data(self) -> Optional[Dict[str, Any]]:
        """Get current training data configuration"""
        if not self.cycle_state.current_day or not self.cycle_state.current_reset_points:
            return None
        
        # Format data for PPOTrainer compatibility
        day_info = {
            'date': self.cycle_state.current_day.date,
            'symbol': self.cycle_state.current_day.symbol,
            'quality_score': self.cycle_state.current_day.day_score,
        }
        
        # Get current reset point index (use 0 for first reset point)
        current_reset_point_index = 0  # TODO: Track current reset point in cycle state
        
        # Format reset points as dictionaries for dashboard compatibility
        reset_points_formatted = []
        for rp in self.cycle_state.current_reset_points:
            reset_point_dict = {
                'timestamp': rp.timestamp,
                'quality_score': rp.quality_score,
                'roc_score': rp.roc_score,
                'activity_score': rp.activity_score,
                'combined_score': rp.quality_score,  # Use quality_score as combined
                'price': rp.price  # Use actual price from reset point
            }
            reset_points_formatted.append(reset_point_dict)
        
        return {
            'day_info': day_info,
            'reset_points': reset_points_formatted,
            'reset_point_index': current_reset_point_index,
            'stage': 'adaptive_data_lifecycle'  # Fixed stage name for adaptive system
        }
    
    def update_progress(self, episodes: int, updates: int):
        """Update progress counters"""
        # Update global counters
        self.cycle_state.episode_count = episodes
        self.cycle_state.update_count = updates
        
        # Update local counters
        self.cycle_state.episodes_in_current_cycle += 1
        self.cycle_state.episodes_in_current_day += 1
        
        # Update day-level counters for switching logic
        self.current_day_episodes += 1
        # Update count is incremented separately when updates happen
        
        self.logger.debug(f"Progress updated: episodes={episodes}, updates={updates}, day_episodes={self.current_day_episodes}")
    
    def get_data_lifecycle_status(self) -> Dict[str, Any]:
        """Get current data lifecycle status"""
        return {
            'cycle_count': self.cycle_state.cycle_count,
            'episodes_in_cycle': self.cycle_state.episodes_in_current_cycle,
            'episodes_in_day': self.current_day_episodes,
            'updates_in_day': self.current_day_updates,
            'cycle_status': self.reset_point_cycler.get_cycle_status() if hasattr(self.reset_point_cycler, 'get_cycle_status') else {},
            'current_day': self.cycle_state.current_day.date if self.cycle_state.current_day else None,
            'current_symbol': self.cycle_state.current_day.symbol if self.cycle_state.current_day else None,
            'reset_points_count': len(self.cycle_state.current_reset_points),
            'preload_ready': self.preload_state.next_stage_ready,
            'should_terminate': self.should_terminate,
            'termination_reason': self.termination_reason.value if self.termination_reason else None
        }
    
    def _should_switch_day(self) -> bool:
        """Check if should switch to next day based on data cycle config"""
        cycle_config = self.config.cycles
        
        # Check if any day limit is reached
        if cycle_config.day_max_episodes and self.current_day_episodes >= cycle_config.day_max_episodes:
            self.logger.info(f"🔄 Day episode limit reached: {self.current_day_episodes}/{cycle_config.day_max_episodes}")
            return True
        
        if cycle_config.day_max_updates and self.current_day_updates >= cycle_config.day_max_updates:
            self.logger.info(f"🔄 Day update limit reached: {self.current_day_updates}/{cycle_config.day_max_updates}")
            return True
        
        if cycle_config.day_max_cycles and self.reset_point_cycler.current_cycle >= cycle_config.day_max_cycles:
            self.logger.info(f"🔄 Day cycle limit reached: {self.reset_point_cycler.current_cycle}/{cycle_config.day_max_cycles}")
            return True
        
        return False
    
    def _should_advance_cycle(self) -> bool:
        """Check if should advance to next reset point"""
        # Only advance when explicitly requested (after episode completion)
        # Don't advance automatically on every check
        return False
    
    def _advance_to_next_day(self) -> bool:
        """Advance to next day - select new day and reset points"""
        # Get adaptive data config for day selection
        adaptive_config = self.config.adaptive_data
        
        # Create a temporary stage config for compatibility
        # Handle both old and new config names
        day_selection_mode = getattr(adaptive_config, 'day_selection_mode', 
                                   getattr(adaptive_config, 'selection_mode', 'sequential'))
        
        temp_stage = StageConfig(
            name="adaptive",
            symbols=adaptive_config.symbols,
            date_range=adaptive_config.date_range,
            day_score_range=adaptive_config.day_score_range,
            roc_range=adaptive_config.roc_range,
            activity_range=adaptive_config.activity_range,
            selection_mode=SelectionMode(day_selection_mode),
            randomize_order=False  # Removed this confusing option
        )
        
        # Select next day
        next_day = self.day_selector.select_day(
            self.available_days, 
            temp_stage,
            self.cycle_state.used_days
        )
        
        if not next_day:
            self.logger.warning("No suitable days available for current adaptive criteria")
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
        quality_criteria = {
            'day_score_range': adaptive_config.day_score_range,
            'roc_range': adaptive_config.roc_range,
            'activity_range': adaptive_config.activity_range
        }
        
        if not self.reset_point_cycler.initialize_day(next_day, quality_criteria):
            self.logger.warning(f"Failed to initialize reset points for day {next_day.date}")
            return False
        
        # Get first reset point
        if not self._advance_to_next_cycle():
            return False
        
        self.logger.info(f"🏙️ Advanced to new day: {next_day.date} ({next_day.symbol})")
        return True
    
    def _advance_to_next_cycle(self) -> bool:
        """Advance to next reset point within current day"""
        if not self.cycle_state.current_day:
            return False
        
        # Get next reset point from cycler
        next_reset_point = self.reset_point_cycler.get_next_reset_point()
        if not next_reset_point:
            self.logger.warning(f"No more reset points available for day {self.cycle_state.current_day.date}")
            return False
        
        # Update state
        self.cycle_state.reset_for_new_cycle()
        self.current_reset_point = next_reset_point
        self.cycle_state.current_reset_points = [next_reset_point]  # Keep as list for compatibility
        
        # Track usage
        self.cycle_state.used_reset_points.add(next_reset_point.timestamp)
        
        cycle_status = self.reset_point_cycler.get_cycle_status()
        self.logger.info(
            f"🔄 Advanced to reset point {cycle_status['current_index']}/{cycle_status['total_reset_points']} "
            f"(cycle {cycle_status['current_cycle']}) in day {self.cycle_state.current_day.date}"
        )
        return True
    
    def update_day_update_count(self):
        """Update the day-level update counter"""
        self.current_day_updates += 1
        self.logger.debug(f"Day update count: {self.current_day_updates}")
    
    def advance_cycle_on_episode_completion(self) -> bool:
        """Explicitly advance to next reset point after episode completion"""
        return self._advance_to_next_cycle()
    
    def apply_dynamic_adaptation(self, adaptation: Dict[str, Any]) -> bool:
        """Apply dynamic adaptation from ContinuousTraining"""
        try:
            # Update adaptive data config if it exists
            if hasattr(self.config, 'adaptive_data') and 'adaptive_data' in adaptation:
                adaptive_changes = adaptation['adaptive_data']
                
                if 'day_score_range' in adaptive_changes:
                    self.config.adaptive_data.day_score_range = adaptive_changes['day_score_range']
                if 'roc_range' in adaptive_changes:
                    self.config.adaptive_data.roc_range = adaptive_changes['roc_range']
                if 'activity_range' in adaptive_changes:
                    self.config.adaptive_data.activity_range = adaptive_changes['activity_range']
                # Handle both old and new selection mode names
                if 'selection_mode' in adaptive_changes:
                    # Map old to new if using old name
                    if hasattr(self.config.adaptive_data, 'day_selection_mode'):
                        self.config.adaptive_data.day_selection_mode = adaptive_changes['selection_mode']
                    else:
                        self.config.adaptive_data.selection_mode = adaptive_changes['selection_mode']
                if 'day_selection_mode' in adaptive_changes:
                    self.config.adaptive_data.day_selection_mode = adaptive_changes['day_selection_mode']
                if 'reset_point_selection_mode' in adaptive_changes:
                    self.config.adaptive_data.reset_point_selection_mode = adaptive_changes['reset_point_selection_mode']
                
                self.logger.info(f"🔄 Applied dynamic adaptation to data lifecycle: {adaptive_changes}")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to apply dynamic adaptation: {e}")
            return False
    
    def force_termination(self, reason: DataTerminationReason):
        """Force termination of data lifecycle"""
        self.should_terminate = True
        self.termination_reason = reason
        self.logger.info(f"🛑 Data lifecycle terminated: {reason.value}")