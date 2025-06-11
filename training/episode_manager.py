"""
Episode Manager - Authority for Training Episode Management
Handles reset point cycling, day selection, and episode-related termination.
Consolidated from DataLifecycleManager for v2 architecture.
"""

import random
import logging
from typing import  Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

from pendulum import Date

from core.utils import day_in_range
from core.utils.time_utils import (
    to_date, format_date, now_utc,
)

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
    index: int = 0  # Index within the day for environment reset

    def meets_criteria(self, quality_range: List[float], roc_range: List[float],
                       activity_range: List[float]) -> bool:
        """Check if reset point meets selection criteria"""
        return (quality_range[0] <= self.quality_score <= quality_range[1] and
                roc_range[0] <= self.roc_score <= roc_range[1] and
                activity_range[0] <= self.activity_score <= activity_range[1])


@dataclass
class DayInfo:
    """Information about a trading day"""
    date: Date  # Pendulum Date object
    symbol: str
    day_score: float
    reset_points: List[ResetPointInfo] = field(default_factory=list)
    used_count: int = 0

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
    total_cycles_completed: int = 0

    # Current selections
    current_day: Optional[DayInfo] = None
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

    def reset_for_new_day(self):
        """Reset counters for new day"""
        self.episodes_in_current_day = 0
        self.current_day_episodes = 0
        self.current_day_updates = 0
        self.current_day = None


@dataclass
class EpisodeContext:
    """Context for a training episode."""
    symbol: str
    date: Date  # Pendulum Date object
    reset_point: ResetPointInfo
    day_info: DayInfo


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

        if not self.data_manager: return available_days

        start_date = to_date(self.date_range[0])
        end_date = to_date(self.date_range[1])
        
        if not start_date or not end_date:
            self.logger.error(f"Invalid date range format: {self.date_range}")
            return available_days

        self.logger.info("🔍 Loading momentum days with filters:")
        self.logger.info(f"   📊 Symbols: {self.symbols}")
        self.logger.info(f"   📅 Date range: {format_date(start_date)} to {format_date(end_date)}")

        try:
            # Get momentum days from a data manager
            momentum_days_dicts = self.data_manager.get_all_momentum_days(
                symbols=self.symbols,
                start_date=start_date,
                end_date=end_date,
            )

            self.logger.info(f"   📊 Found {len(momentum_days_dicts)} momentum days after filtering")

            # Convert dictionary format to DayInfo objects
            for day_dict in momentum_days_dicts:
                # Get reset points for this day
                reset_points = []
                try:
                    reset_points_df = self.data_manager.get_reset_points(
                        day_dict['symbol'], day_dict['date']
                    )
                    # Convert reset points to ResetPointInfo objects
                    for idx, (_, rp_row) in enumerate(reset_points_df.iterrows()):
                        # Validate required fields - no fallback values for quality metrics
                        if 'combined_score' not in rp_row or rp_row['combined_score'] is None:
                            self.logger.warning(f"Skipping reset point {idx} - missing combined_score")
                            continue
                        if 'roc_score' not in rp_row or rp_row['roc_score'] is None:
                            self.logger.warning(f"Skipping reset point {idx} - missing roc_score")
                            continue
                        if 'activity_score' not in rp_row or rp_row['activity_score'] is None:
                            self.logger.warning(f"Skipping reset point {idx} - missing activity_score")
                            continue

                        reset_point = ResetPointInfo(
                            timestamp=str(rp_row['timestamp']),
                            quality_score=float(rp_row['combined_score']),
                            roc_score=float(rp_row['roc_score']),
                            activity_score=float(rp_row['activity_score']),
                            price=float(rp_row.get('price', 0.0)),  # Price can default to 0.0
                            index=idx
                        )
                        reset_points.append(reset_point)
                except Exception as e:
                    self.logger.warning(f"Failed to load reset points for {day_dict['symbol']} {day_dict['date']}: {e}")

                if 'quality_score' not in day_dict or day_dict['quality_score'] is None:
                    self.logger.warning(f"Skipping day {day_dict['symbol']} {day_dict['date']} - missing quality_score")
                    continue

                # Convert date string to Date object
                day = to_date(day_dict['date'])
                if not day:
                    self.logger.warning(f"Skipping day {day_dict['symbol']} - invalid date format: {day_dict['date']}")
                    continue

                if not reset_points:
                    self.logger.warning(f"Skipping day {day_dict['symbol']} {format_date(day)} - no valid reset points")
                    continue

                day_info = DayInfo(
                    date=day,
                    symbol=day_dict['symbol'],
                    day_score=float(day_dict['quality_score']),
                    reset_points=reset_points
                )
                available_days.append(day_info)

        except Exception as e:
            self.logger.error(f"Failed to load momentum days: {e}")

        self.logger.info(f"✅ Loaded {len(available_days)} valid days with reset points")
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

                self.logger.info(f"📅 Selected: {symbol} {format_date(day_date)} (quality: {quality:.3f})")
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

    def get_next_episode(self) -> Optional[EpisodeContext]:
        """Get next episode context, handling day/reset point transitions internally."""

        # Check if we need to switch days
        if self._should_switch_day():
            if not self._advance_to_next_day():
                self.logger.info("No more days available")
                return None

        # Get next reset point (handles cycling internally)
        if not self.state.current_reset_point:
            if not self._advance_to_next_reset_point():
                self.logger.warning("No reset points available")
                return None

        # Create episode context - we know these are not None due to checks above
        if not self.state.current_day or not self.state.current_reset_point:
            self.logger.error("Missing current day or reset point for episode context")
            return None

        return EpisodeContext(
            symbol=self.state.current_day.symbol,
            date=self.state.current_day.date,
            reset_point=self.state.current_reset_point,
            day_info=self.state.current_day
        )

    def on_episodes_completed(self, count: int):
        """Handle notification of completed episodes."""
        if count <= 0:
            self.logger.warning(f"Invalid episode count: {count}")
            return

        self.state.current_day_episodes += count
        self.state.episodes_in_current_day += count
        self.state.episodes_in_current_cycle += count

        self.logger.debug(f"Episodes completed: +{count}, day total: {self.state.current_day_episodes}")

        # Advance to the next reset point after episode completion
        if not self._advance_to_next_reset_point():
            self.logger.debug("Completed all reset points in current cycle")

    def on_update_completed(self, update_info: Any):
        """Handle notification of a completed policy update."""
        self.state.current_day_updates += 1

        self.logger.debug(f"Update completed, day total: {self.state.current_day_updates}")

        # Validate state consistency
        if self.state.current_day_updates < 0:
            self.logger.warning("Negative update count detected, resetting")
            self.state.current_day_updates = 0

    def force_termination(self, reason: EpisodeTerminationReason):
        """Force termination of episode manager"""
        self.state.should_terminate = True
        self.state.termination_reason = reason
        self.logger.info(f"🛑 Force terminating episode manager: {reason.value}")

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

        # Track usage - convert Date to string for set storage
        self.state.used_days.add(format_date(next_day.date))

        # Initialize reset points for this day
        if not self._initialize_reset_points_for_day(next_day):
            self.logger.warning(f"Failed to initialize reset points for day {format_date(next_day.date)}")
            return False

        # Get first reset point
        if not self._advance_to_next_reset_point():
            return False

        self.logger.info(f"🏙️ Advanced to new day: {format_date(next_day.date)} ({next_day.symbol})")
        return True

    def _select_day(self) -> Optional[DayInfo]:
        """Select next training day based on criteria"""
        if not self.available_days:
            self.logger.error("No available days loaded")
            return None

        # Filter by criteria
        filtered_days = []
        for day in self.available_days:
            try:
                if day_in_range(day.date, self.date_range):
                    if not self.symbols or day.symbol in self.symbols:
                        if self.day_score_range[0] <= day.day_score <= self.day_score_range[1]:
                            available_rps = day.get_available_reset_points(
                                self.day_score_range,
                                self.roc_range,
                                self.activity_range
                            )
                            if available_rps:
                                filtered_days.append(day)
            except Exception as e:
                self.logger.warning(f"Error filtering day {format_date(day.date)}: {e}")
                continue

        if not filtered_days:
            self.logger.warning("No days meet current criteria")
            return None

        # Remove recently used days (unless using random mode)
        if self.day_selection_mode != SelectionMode.RANDOM:
            unused_days = [day for day in filtered_days if format_date(day.date) not in self.state.used_days]
            if unused_days:
                filtered_days = unused_days
            else:
                # If all days have been used, clear the used set and start over
                self.logger.info("All days have been used, resetting usage tracking")
                self.state.used_days.clear()

        # Select based on mode
        if self.day_selection_mode == SelectionMode.RANDOM:
            selected = random.choice(filtered_days)
        elif self.day_selection_mode == SelectionMode.QUALITY:
            # Sort by quality score (highest first)
            filtered_days.sort(key=lambda d: day.day_score, reverse=True)
            selected = filtered_days[0]
        else:  # SEQUENTIAL
            # Sort by date (earliest first)
            filtered_days.sort(key=lambda d: day.date)
            selected = filtered_days[0]

        # Mark as used
        selected.used_count += 1

        return selected

    def _initialize_reset_points_for_day(self, day: DayInfo) -> bool:
        """Initialize reset point order for a new day"""
        if not day or not day.reset_points:
            self.logger.error(f"Invalid day info or no reset points: {format_date(day.date) if day else 'None'}")
            return False

        # Get all suitable reset points
        available = []
        for reset_point in day.reset_points:
            try:
                if reset_point.meets_criteria(self.day_score_range, self.roc_range, self.activity_range):
                    available.append(reset_point)
            except Exception as e:
                self.logger.warning(f"Error checking reset point criteria: {e}")
                continue

        if not available:
            self.logger.warning(f"No suitable reset points for day {format_date(day.date)}")
            # Fallback: use all reset points if none meet criteria
            if day.reset_points:
                self.logger.info("Using all available reset points as fallback")
                available = day.reset_points[:]
            else:
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

        self.logger.info(
            f"🔄 Initialized {len(self.state.ordered_reset_points)} reset points for day {format_date(day.date)} in {self.reset_point_selection_mode.value} mode")
        return True

    def _advance_to_next_reset_point(self) -> bool:
        """Advance to next reset point within current day"""
        if not self.state.ordered_reset_points:
            self.logger.warning("No reset points available")
            return False

        # Validate current state
        if self.state.current_reset_point_index >= len(self.state.ordered_reset_points):
            self.logger.warning(f"Invalid reset point index: {self.state.current_reset_point_index}/{len(self.state.ordered_reset_points)}")
            self.state.current_reset_point_index = 0

        # Track previous cycle count to detect completion
        previous_cycle_count = self.state.current_reset_point_cycle

        # Get current reset point
        reset_point = self.state.ordered_reset_points[self.state.current_reset_point_index]

        self.logger.info(
            f"🎯 Reset point cycle: index {self.state.current_reset_point_index}/{len(self.state.ordered_reset_points)}, time: {reset_point.timestamp}")

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

        self.state.current_reset_point = reset_point

        # Track usage
        self.state.used_reset_points.add(reset_point.timestamp)

        return True
