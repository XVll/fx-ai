"""
Episode Manager - Authority for Training Episode Management
Handles reset point cycling, day selection, and episode-related termination.
Consolidated from DataLifecycleManager for v2 architecture.
"""

import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, cast
from dataclasses import dataclass, field
from enum import Enum

import pendulum
from pendulum import DateTime

from config.training.training_config import TrainingManagerConfig


class EpisodeTerminationReason(Enum):
    """Reasons for episode manager termination"""
    CYCLE_LIMIT_REACHED = "cycle_limit_reached"
    EPISODE_LIMIT_REACHED = "episode_limit_reached"
    UPDATE_LIMIT_REACHED = "update_limit_reached"
    NO_MORE_RESET_POINTS = "no_more_reset_points"
    NO_MORE_DAYS = "no_more_days" 
    DATE_RANGE_EXHAUSTED = "date_range_exhausted"
    QUALITY_CRITERIA_NOT_MET = "quality_criteria_not_met"
    PRELOAD_FAILED = "preload_failed"


class SelectionMode(Enum):
    """Data selection modes"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    QUALITY = "quality"


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
class EpisodeManagerState:
    """Consolidated state for episode manager"""
    # Cycle tracking
    cycle_count: int = 0
    episode_count: int = 0
    update_count: int = 0
    total_cycles_completed: int = 0
    
    # Current selections
    current_day: Optional[DayInfo] = None
    current_reset_points: List[ResetPointInfo] = field(default_factory=list)
    current_reset_point: Optional[ResetPointInfo] = None
    
    # Tracking sets
    used_days: Set[str] = field(default_factory=set)
    used_reset_points: Set[str] = field(default_factory=set)
    
    # Progress tracking
    episodes_in_current_cycle: int = 0
    episodes_in_current_day: int = 0
    current_day_episodes: int = 0
    current_day_updates: int = 0
    
    # Reset point cycling state
    current_reset_point_cycle: int = 0
    current_reset_point_index: int = 0
    ordered_reset_points: List[ResetPointInfo] = field(default_factory=list)
    
    # Termination state
    should_terminate: bool = False
    termination_reason: Optional[EpisodeTerminationReason] = None
    
    def reset_for_new_cycle(self):
        """Reset counters for new cycle"""
        self.cycle_count += 1
        self.total_cycles_completed += 1
        self.episodes_in_current_cycle = 0
        self.current_reset_points = []
    
    def reset_for_new_day(self):
        """Reset counters for new day"""
        self.episodes_in_current_day = 0
        self.current_day_episodes = 0
        self.current_day_updates = 0
        self.current_day = None


class EpisodeManager:
    """
    Episode Manager - Authority for Training Episode Management
    
    Single consolidated class handling all episode management responsibilities:
    - Reset point cycling and tracking
    - Day selection and quality management
    - Cycle counting (episodes, updates, cycles)
    - Edge case handling (no data, exhausted ranges)
    - Episode-specific termination conditions
    """
    
    def __init__(self, config: TrainingManagerConfig, data_manager=None):
        """Initialize episode manager with configuration and data manager"""
        self.config = config
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Single consolidated state
        self.state = EpisodeManagerState()
        
        # Extract configuration directly from typed config
        self.day_selection_mode = SelectionMode(config.day_selection_mode)
        self.reset_point_selection_mode = SelectionMode(config.reset_point_selection_mode)
        self.symbols = config.symbols
        self.date_range = config.date_range
        self.day_score_range = config.day_score_range
        self.roc_range = config.reset_roc_range
        self.activity_range = config.reset_activity_range
        
        # Daily limits from config
        self.daily_max_episodes = config.daily_max_episodes
        self.daily_max_updates = config.daily_max_updates
        self.daily_max_cycles = config.daily_max_cycles
        
        # Load available days from data manager
        self.available_days = self._load_available_days()
        
        self.logger.debug(f"EpisodeManager initialized with {len(self.available_days)} available days")
    
    def _load_available_days(self) -> List[DayInfo]:
        """Load available momentum days from data manager"""
        available_days = []
        
        if not self.data_manager or not hasattr(self.data_manager, 'get_all_momentum_days'):
            self.logger.warning("No data manager or get_all_momentum_days method available")
            return available_days
        
        # Parse date range from config
        start_date = None
        end_date = None
        start_date_str = self.date_range[0] if self.date_range[0] else None
        end_date_str = self.date_range[1] if self.date_range[1] else None
        
        if start_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        if end_date_str:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        self.logger.info(f"🔍 Loading momentum days with filters:")
        self.logger.info(f"   📊 Symbols: {self.symbols}")
        self.logger.info(f"   📅 Date range: {start_date_str or 'None'} to {end_date_str or 'None'}")
        
        try:
            # Get momentum days from data manager
            momentum_days_dicts = self.data_manager.get_all_momentum_days(
                symbols=self.symbols if self.symbols else None,
                start_date=start_date,
                end_date=end_date,
                min_activity=0.0  # No activity filtering here, let episode manager handle it
            )
            
            self.logger.info(f"   📊 Found {len(momentum_days_dicts)} momentum days after filtering")
            
            # Convert dictionary format to DayInfo objects
            for day_dict in momentum_days_dicts:
                # Get reset points for this day
                reset_points = []
                if hasattr(self.data_manager, 'get_reset_points'):
                    try:
                        reset_points_df = self.data_manager.get_reset_points(
                            day_dict['symbol'], day_dict['date']
                        )
                        # Convert reset points to ResetPointInfo objects
                        for _, rp_row in reset_points_df.iterrows():
                            reset_point = ResetPointInfo(
                                timestamp=str(rp_row['timestamp']),
                                quality_score=rp_row.get('combined_score', 0.5),
                                roc_score=rp_row.get('roc_score', 0.0),
                                activity_score=rp_row.get('activity_score', 0.5),
                                price=rp_row.get('price', 0.0)
                            )
                            reset_points.append(reset_point)
                    except Exception as e:
                        self.logger.warning(f"Failed to load reset points for {day_dict['symbol']} {day_dict['date']}: {e}")
                
                # Convert date to string format for DayInfo
                if hasattr(day_dict['date'], 'strftime'):
                    date_str = day_dict['date'].strftime('%Y-%m-%d')
                else:
                    date_str = str(day_dict['date'])
                
                day_info = DayInfo(
                    date=date_str,
                    symbol=day_dict['symbol'],
                    day_score=day_dict.get('quality_score', 0.5),
                    reset_points=reset_points
                )
                available_days.append(day_info)
            
        except Exception as e:
            self.logger.error(f"Failed to load momentum days: {e}")
        
        return available_days
    
    def initialize(self) -> bool:
        """Initialize episode manager - select first day and reset points"""
        try:
            if not self._advance_to_next_day():
                self.logger.error("Failed to select initial day")
                return False
                
            # Log final summary
            if self.state.current_day:
                day_date = self.state.current_day.date
                symbol = self.state.current_day.symbol
                reset_count = len(self.state.ordered_reset_points)
                quality = self.state.current_day.day_score
                
                self.logger.info(f"📅 Selected: {symbol} {day_date} (quality: {quality:.3f})")
                self.logger.info(f"🔄 Reset points: {reset_count} available")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize episode manager: {e}")
            return False
    
    def should_terminate(self) -> Optional[EpisodeTerminationReason]:
        """Check if episode manager should terminate"""
        if self.state.should_terminate:
            return self.state.termination_reason
        
        # Check if should switch to next day
        if self._should_switch_day():
            if not self._advance_to_next_day():
                self.logger.info("🔄 No more days available, continuing with current day reset points")
                # Reset to beginning of reset points for current day
                self.state.current_reset_point_index = 0
        
        return None
    
    def get_current_episode_config(self) -> Optional[Dict[str, Any]]:
        """Get current episode configuration for training"""
        if not self.state.current_day or not self.state.current_reset_point:
            return None
        
        # Format data for compatibility
        day_info = {
            'date': self.state.current_day.date,
            'symbol': self.state.current_day.symbol,
            'quality_score': self.state.current_day.day_score,
        }
        
        # Format ALL reset points for display
        reset_points_formatted = []
        for rp in self.state.ordered_reset_points:
            reset_point_dict = {
                'timestamp': rp.timestamp,
                'quality_score': rp.quality_score,
                'roc_score': rp.roc_score,
                'activity_score': rp.activity_score,
                'combined_score': rp.quality_score,
                'price': rp.price
            }
            reset_points_formatted.append(reset_point_dict)
        
        # Find current reset point index
        current_reset_point_index = max(0, self.state.current_reset_point_index - 1)
        
        return {
            'day_info': day_info,
            'reset_points': reset_points_formatted,
            'reset_point_index': current_reset_point_index,
            'stage': 'episode_manager'
        }
    
    def advance_episode(self) -> bool:
        """Advance to next episode (next reset point)"""
        self.logger.info("🔄 Episode completed - advancing to next reset point")
        result = self._advance_to_next_reset_point()
        if result and self.state.current_reset_point:
            self.logger.info(f"✅ Advanced to reset point at {self.state.current_reset_point.timestamp}")
        else:
            self.logger.warning("❌ Failed to advance to next reset point")
        return result
    
    def update_progress(self, episodes: int, updates: int):
        """Update progress counters"""
        # Update global counters
        self.state.episode_count = episodes
        self.state.update_count = updates
        
        # Update local counters
        self.state.episodes_in_current_cycle += 1
        self.state.episodes_in_current_day += 1
        self.state.current_day_episodes += 1
        
        self.logger.debug(f"Progress updated: episodes={episodes}, updates={updates}, day_episodes={self.state.current_day_episodes}")
    
    def update_day_update_count(self):
        """Update the day-level update counter"""
        self.state.current_day_updates += 1
        self.logger.debug(f"Day update count: {self.state.current_day_updates}")
    
    
    def force_termination(self, reason: EpisodeTerminationReason):
        """Force termination of episode manager"""
        self.state.should_terminate = True
        self.state.termination_reason = reason
        self.logger.info(f"🛑 Episode manager terminated: {reason.value}")
    
    # Private methods for internal logic
    
    def _should_switch_day(self) -> bool:
        """Check if should switch to next day based on configuration"""
        # Check daily limits
        if self.daily_max_episodes and self.state.current_day_episodes >= self.daily_max_episodes:
            self.logger.info(f"🔄 Daily episode limit reached: {self.state.current_day_episodes}/{self.daily_max_episodes}")
            return True
        
        if self.daily_max_updates and self.state.current_day_updates >= self.daily_max_updates:
            self.logger.info(f"🔄 Daily update limit reached: {self.state.current_day_updates}/{self.daily_max_updates}")
            return True
        
        if self.daily_max_cycles and self.state.current_reset_point_cycle >= self.daily_max_cycles:
            self.logger.info(f"🔄 Daily cycle limit reached: {self.state.current_reset_point_cycle}/{self.daily_max_cycles}")
            return True
        
        return False
    
    def _advance_to_next_day(self) -> bool:
        """Advance to next day - select new day and reset points"""
        next_day = self._select_day()
        
        if not next_day:
            self.logger.warning("No suitable days available for current criteria")
            return False
        
        # Reset day-level counters
        self.state.reset_for_new_day()
        self.state.current_day = next_day
        
        # Track usage
        self.state.used_days.add(next_day.date)
        
        # Initialize reset points for this day
        if not self._initialize_reset_points_for_day(next_day):
            self.logger.warning(f"Failed to initialize reset points for day {next_day.date}")
            return False
        
        # Get first reset point
        if not self._advance_to_next_reset_point():
            return False
        
        self.logger.info(f"🏙️ Advanced to new day: {next_day.date} ({next_day.symbol})")
        return True
    
    def _select_day(self) -> Optional[DayInfo]:
        """Select next training day based on criteria"""
        # Filter by criteria
        filtered_days = []
        for day in self.available_days:
            if self._day_in_date_range(day, self.date_range):
                if not self.symbols or day.symbol in self.symbols:
                    if self.day_score_range[0] <= day.day_score <= self.day_score_range[1]:
                        available_rps = day.get_available_reset_points(
                            self.day_score_range,
                            self.roc_range,
                            self.activity_range
                        )
                        if available_rps:
                            filtered_days.append(day)
        
        if not filtered_days:
            self.logger.warning("No days meet current criteria")
            return None
        
        # Remove recently used days (unless using random mode)
        if self.day_selection_mode != SelectionMode.RANDOM:
            unused_days = [day for day in filtered_days if day.date not in self.state.used_days]
            if unused_days:
                filtered_days = unused_days
        
        # Select based on mode
        if self.day_selection_mode == SelectionMode.RANDOM:
            selected = random.choice(filtered_days)
        elif self.day_selection_mode == SelectionMode.QUALITY:
            # Sort by quality score (highest first)
            filtered_days.sort(key=lambda day: day.day_score, reverse=True)
            selected = filtered_days[0]
        else:  # SEQUENTIAL
            # Sort by date (earliest first)
            filtered_days.sort(key=lambda day: pendulum.parse(day.date))
            selected = filtered_days[0]
        
        # Mark as used
        selected.used_count += 1
        selected.last_used = datetime.now()
        
        return selected
    
    def _day_in_date_range(self, day: DayInfo, date_range: List[Optional[str]]) -> bool:
        """Check if day falls within date range using pendulum"""
        if not date_range[0] and not date_range[1]:
            return True
        
        day_date = cast(DateTime, pendulum.parse(day.date))
        
        if date_range[0]:
            start_date = cast(DateTime, pendulum.parse(date_range[0]))
            if day_date < start_date:
                return False
        
        if date_range[1]:
            end_date = cast(DateTime, pendulum.parse(date_range[1]))
            if day_date > end_date:
                return False
        
        return True
    
    def _initialize_reset_points_for_day(self, day: DayInfo) -> bool:
        """Initialize reset point order for a new day"""
        # Get all suitable reset points
        available = []
        for rp in day.reset_points:
            if rp.meets_criteria(self.day_score_range, self.roc_range, self.activity_range):
                available.append(rp)
        
        if not available:
            self.logger.warning(f"No suitable reset points for day {day.date}")
            return False
        
        # Order based on selection mode
        if self.reset_point_selection_mode == SelectionMode.SEQUENTIAL:
            # Order by timestamp (sequential time order)
            self.state.ordered_reset_points = sorted(available, key=lambda rp: rp.timestamp)
        elif self.reset_point_selection_mode == SelectionMode.QUALITY:
            # Order by quality score (highest first)
            self.state.ordered_reset_points = sorted(available, key=lambda rp: rp.quality_score, reverse=True)
        else:  # RANDOM
            self.state.ordered_reset_points = available.copy()
            random.shuffle(self.state.ordered_reset_points)
        
        # Reset cycling state
        self.state.current_reset_point_cycle = 0
        self.state.current_reset_point_index = 0
        
        self.logger.info(f"🔄 Initialized {len(self.state.ordered_reset_points)} reset points for day {day.date} in {self.reset_point_selection_mode.value} mode")
        return True
    
    def _advance_to_next_reset_point(self) -> bool:
        """Advance to next reset point within current day"""
        if not self.state.ordered_reset_points:
            self.logger.warning("No reset points available")
            return False
        
        # Track previous cycle count to detect completion
        previous_cycle_count = self.state.current_reset_point_cycle
        
        # Get current reset point
        reset_point = self.state.ordered_reset_points[self.state.current_reset_point_index]
        
        self.logger.info(f"🎯 Reset point cycle: index {self.state.current_reset_point_index}/{len(self.state.ordered_reset_points)}, time: {reset_point.timestamp}")
        
        # Advance to next
        self.state.current_reset_point_index += 1
        
        # Check if completed full cycle
        if self.state.current_reset_point_index >= len(self.state.ordered_reset_points):
            self.state.current_reset_point_index = 0
            self.state.current_reset_point_cycle += 1
            self.logger.info(f"🔄 Completed cycle {self.state.current_reset_point_cycle}, starting over")
            
            # Re-shuffle for random mode
            if self.reset_point_selection_mode == SelectionMode.RANDOM:
                random.shuffle(self.state.ordered_reset_points)
                self.logger.debug("🔀 Re-shuffled reset points for next cycle")
        
        # Check if cycle was completed for state management
        cycle_completed = self.state.current_reset_point_cycle > previous_cycle_count
        if cycle_completed:
            self.state.reset_for_new_cycle()
        else:
            self.state.episodes_in_current_cycle += 1
        
        # Store current reset point
        self.state.current_reset_point = reset_point
        self.state.current_reset_points = [reset_point]  # Keep as list for compatibility
        
        # Track usage
        self.state.used_reset_points.add(reset_point.timestamp)
        
        return True
    
    def get_cycle_status(self) -> Dict[str, Any]:
        """Get current cycle status"""
        return {
            'current_cycle': self.state.current_reset_point_cycle,
            'current_index': self.state.current_reset_point_index,
            'total_reset_points': len(self.state.ordered_reset_points),
            'progress_in_cycle': self.state.current_reset_point_index / max(1, len(self.state.ordered_reset_points)),
            'episodes_in_cycle': self.state.episodes_in_current_cycle,
            'episodes_in_day': self.state.episodes_in_current_day,
            'total_cycles_completed': self.state.total_cycles_completed
        }