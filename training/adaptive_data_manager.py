"""
Adaptive Data Manager - Replaces Stage-based System
Dynamically adjusts data difficulty based on performance without fixed stages.
"""

import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from training.data_lifecycle_manager import (
    DataTerminationReason, SelectionMode, ResetPointInfo, DayInfo,
    ResetPointCycler, DaySelector, DataPreloader, CycleState, PreloadState
)


@dataclass 
class AdaptiveCriteria:
    """Current adaptive data selection criteria"""
    day_score_range: List[float] = field(default_factory=lambda: [0.7, 1.0])
    roc_range: List[float] = field(default_factory=lambda: [0.05, 1.0])
    activity_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    symbols: List[str] = field(default_factory=lambda: ["MLGO"])
    date_range: List[Optional[str]] = field(default_factory=lambda: [None, None])
    selection_mode: SelectionMode = SelectionMode.QUALITY_WEIGHTED
    randomize_order: bool = False
    
    def copy(self) -> 'AdaptiveCriteria':
        """Create a copy of current criteria"""
        return AdaptiveCriteria(
            day_score_range=self.day_score_range.copy(),
            roc_range=self.roc_range.copy(),
            activity_range=self.activity_range.copy(),
            symbols=self.symbols.copy(),
            date_range=self.date_range.copy(),
            selection_mode=self.selection_mode,
            randomize_order=self.randomize_order
        )
    
    def apply_difficulty_change(self, change: Dict[str, Any]):
        """Apply difficulty change to criteria"""
        if 'day_score_range' in change:
            self.day_score_range = change['day_score_range']
        if 'roc_range' in change:
            self.roc_range = change['roc_range']
        if 'activity_range' in change:
            self.activity_range = change['activity_range']
        if 'selection_mode' in change:
            self.selection_mode = SelectionMode(change['selection_mode'])


class AdaptiveDataManager:
    """
    Manages adaptive data selection without fixed stages
    
    Responsibilities:
    - Dynamic data difficulty adjustment based on performance
    - Reset point cycling with adaptive criteria
    - Day selection with quality adaptation
    - Seamless transitions without stage boundaries
    """
    
    def __init__(self, config, available_days: List[DayInfo]):
        self.config = config
        self.available_days = available_days
        self.logger = logging.getLogger(__name__)
        
        # Initialize adaptive criteria from config
        self.current_criteria = AdaptiveCriteria(
            day_score_range=config.adaptive_data.day_score_range.copy(),
            roc_range=config.adaptive_data.roc_range.copy(),
            activity_range=config.adaptive_data.activity_range.copy(),
            symbols=config.adaptive_data.symbols.copy(),
            date_range=config.adaptive_data.date_range.copy(),
            selection_mode=SelectionMode(config.adaptive_data.selection_mode),
            randomize_order=config.adaptive_data.randomize_order
        )
        
        # Initialize components
        self.reset_point_cycler = ResetPointCycler(config.reset_points)
        self.day_selector = DaySelector(config.day_selection)
        self.preloader = DataPreloader(config.preloading)
        
        # State management
        self.cycle_state = CycleState()
        self.preload_state = PreloadState()
        self.should_terminate = False
        self.termination_reason: Optional[DataTerminationReason] = None
        
        # Adaptation tracking
        self.adaptation_history: List[Dict[str, Any]] = []
        self.current_day: Optional[DayInfo] = None
        self.current_reset_points: List[ResetPointInfo] = []
        
        # Day switching counters based on DataCycleConfig
        self.current_day_episodes = 0
        self.current_day_updates = 0
        
        # Current reset point from cycler
        self.current_reset_point = None
        
        self.logger.info(f"🎯 AdaptiveDataManager initialized with criteria: {self.current_criteria}")
    

    def should_terminate_data_lifecycle(self) -> Optional[DataTerminationReason]:
        """Check if data lifecycle should terminate"""
        if self.should_terminate:
            return self.termination_reason
        
        # Check if should switch to next day based on cycle config
        if self._should_switch_day():
            if not self._advance_to_next_day():
                # Fallback: Reset reset point cycle and continue on current day
                self.reset_point_cycler.current_index = 0
                self.reset_point_cycler.current_cycle = 0
                self.logger.info("🔄 No more days available, restarting reset points on current day")
                # Continue training - don't terminate
        
        # Check if current reset points exhausted
        elif self._should_advance_cycle():
            if not self._advance_to_next_cycle():
                return DataTerminationReason.NO_MORE_RESET_POINTS
        
        return None
    
    def get_current_training_data(self) -> Optional[Dict[str, Any]]:
        """Get current training data configuration"""
        if not self.current_day or not self.current_reset_points:
            return None
        
        return {
            'date': self.current_day.date,
            'symbol': self.current_day.symbol,
            'reset_points': [rp.timestamp for rp in self.current_reset_points],
            'day_score': self.current_day.day_score,
            'criteria': {
                'day_score_range': self.current_criteria.day_score_range,
                'roc_range': self.current_criteria.roc_range,
                'activity_range': self.current_criteria.activity_range,
                'selection_mode': self.current_criteria.selection_mode.value
            }
        }
    
    def update_progress(self, episodes: int, updates: int):
        """Update progress counters"""
        # Update global counters
        self.cycle_state.episode_count = episodes
        self.cycle_state.update_count = updates
        
        # Update local counters
        self.cycle_state.episodes_in_current_cycle += 1
        
        # Update day-level counters for switching logic
        self.current_day_episodes += 1
        # Update count is incremented separately when updates happen
        
        self.logger.debug(f"Progress updated: episodes={episodes}, updates={updates}, day_episodes={self.current_day_episodes}")
    
    def apply_difficulty_adaptation(self, adaptation: Dict[str, Any]) -> bool:
        """Apply difficulty adaptation from ContinuousTraining"""
        try:
            previous_criteria = self.current_criteria.copy()
            self.current_criteria.apply_difficulty_change(adaptation)
            
            # Track adaptation
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'previous': previous_criteria.__dict__,
                'new': self.current_criteria.__dict__,
                'reason': adaptation.get('reason', 'Unknown')
            })
            
            self.logger.info(f"🔄 Applied difficulty adaptation: {adaptation}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply adaptation: {e}")
            return False
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        # Use ResetPointCycler's current_cycle for accurate cycle counting
        # This represents completed cycles through ALL reset points, not individual reset point advances
        actual_cycle_count = self.reset_point_cycler.current_cycle if self.reset_point_cycler else 0
        
        return {
            'current_criteria': self.current_criteria.__dict__,
            'cycle_count': actual_cycle_count,  # Use actual completed cycles for consistency
            'episodes_in_cycle': self.cycle_state.episodes_in_current_cycle,
            'episodes_in_day': self.current_day_episodes,
            'updates_in_day': self.current_day_updates,
            'cycle_status': self.reset_point_cycler.get_cycle_status() if hasattr(self.reset_point_cycler, 'get_cycle_status') else {},
            'current_day': self.current_day.date if self.current_day else None,
            'current_symbol': self.current_day.symbol if self.current_day else None,
            'reset_points_count': len(self.current_reset_points),
            'adaptations_applied': len(self.adaptation_history),
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
        """Check if should advance to next reset point cycle within the same day"""
        cycle_config = self.config.cycles
        
        # Check if we've used all reset points according to max_reset_point_reuse
        max_reuse = self.config.reset_points.max_reset_point_reuse
        if not self.current_reset_points:
            return True
        
        # Check if all current reset points have been overused
        all_overused = all(rp.used_count >= max_reuse for rp in self.current_reset_points)
        return all_overused
    
    def _advance_to_next_day(self) -> bool:
        """Advance to next day - select new day and reset points"""
        # Select next day using current criteria
        filtered_days = self._filter_days_by_criteria(self.available_days)
        if not filtered_days:
            self.logger.warning("No days meet current adaptive criteria")
            return False
        
        # Use day selector to pick from filtered days
        next_day = self._select_day_from_filtered(filtered_days)
        if not next_day:
            self.logger.warning("Day selector couldn't pick from filtered days")
            return False
        
        # Reset day-level counters
        self.current_day_episodes = 0
        self.current_day_updates = 0
        
        # Update state
        self.cycle_state.reset_for_new_day()
        self.current_day = next_day
        
        # Track usage
        self.cycle_state.used_days.add(next_day.date)
        
        # Initialize first cycle for this day
        if not self._advance_to_next_cycle():
            return False
        
        self.logger.info(f"🏙️ Advanced to new day: {next_day.date} ({next_day.symbol})")
        return True
    
    def _advance_to_next_cycle(self) -> bool:
        """Advance to next reset point cycle within current day"""
        if not self.current_day:
            return False
        
        # Select reset points using current criteria
        reset_points = self._select_reset_points_for_day(self.current_day)
        if not reset_points:
            self.logger.warning(f"No suitable reset points for day {self.current_day.date}")
            return False
        
        # Update state
        self.cycle_state.reset_for_new_cycle()
        self.current_reset_points = reset_points
        
        # Track usage
        for rp in reset_points:
            self.cycle_state.used_reset_points.add(rp.timestamp)
        
        self.logger.info(
            f"🔄 Advanced to cycle {self.cycle_state.cycle_count} in day {self.current_day.date} with {len(reset_points)} reset points"
        )
        return True
    
    def _filter_days_by_criteria(self, days: List[DayInfo]) -> List[DayInfo]:
        """Filter days by current adaptive criteria"""
        filtered = []
        
        for day in days:
            # Check symbol
            if day.symbol not in self.current_criteria.symbols:
                continue
            
            # Check date range
            if not self._day_in_date_range(day):
                continue
            
            # Check day quality
            if not (self.current_criteria.day_score_range[0] <= day.day_score <= self.current_criteria.day_score_range[1]):
                continue
            
            # Check if has suitable reset points
            suitable_rps = day.get_available_reset_points(
                self.current_criteria.day_score_range,
                self.current_criteria.roc_range,
                self.current_criteria.activity_range
            )
            if suitable_rps:
                filtered.append(day)
        
        return filtered
    
    def _day_in_date_range(self, day: DayInfo) -> bool:
        """Check if day falls within adaptive criteria date range"""
        if not self.current_criteria.date_range[0] and not self.current_criteria.date_range[1]:
            return True
        
        day_date = datetime.strptime(day.date, "%Y-%m-%d").date()
        
        if self.current_criteria.date_range[0]:
            start_date = datetime.strptime(self.current_criteria.date_range[0], "%Y-%m-%d").date()
            if day_date < start_date:
                return False
        
        if self.current_criteria.date_range[1]:
            end_date = datetime.strptime(self.current_criteria.date_range[1], "%Y-%m-%d").date()
            if day_date > end_date:
                return False
        
        return True
    
    def _select_day_from_filtered(self, filtered_days: List[DayInfo]) -> Optional[DayInfo]:
        """Select day from pre-filtered list"""
        if not filtered_days:
            return None
        
        # Remove recently used if not randomized
        if not self.current_criteria.randomize_order:
            unused_days = [day for day in filtered_days if day.date not in self.cycle_state.used_days]
            if unused_days:
                filtered_days = unused_days
        
        # Select based on current criteria mode
        if self.current_criteria.selection_mode == SelectionMode.RANDOM:
            selected = random.choice(filtered_days)
        elif self.current_criteria.selection_mode == SelectionMode.QUALITY_WEIGHTED:
            weights = [day.day_score for day in filtered_days]
            selected = random.choices(filtered_days, weights=weights)[0]
        else:  # SEQUENTIAL
            filtered_days.sort(key=lambda d: d.day_score, reverse=True)
            selected = filtered_days[0]
        
        # Mark as used
        selected.used_count += 1
        selected.last_used = datetime.now()
        
        return selected
    
    def _select_reset_points_for_day(self, day: DayInfo) -> List[ResetPointInfo]:
        """Select reset points for day using current criteria"""
        # Create a temporary StageConfig for compatibility with reset_point_cycler
        from training.data_lifecycle_manager import StageConfig
        temp_stage = StageConfig(
            name="adaptive",
            day_score_range=self.current_criteria.day_score_range,
            roc_range=self.current_criteria.roc_range,
            activity_range=self.current_criteria.activity_range
        )
        return self.reset_point_cycler.select_reset_points(
            day, temp_stage, self.cycle_state.used_reset_points
        )
    
    def update_day_update_count(self):
        """Update the day-level update counter"""
        self.current_day_updates += 1
        self.logger.debug(f"Day update count: {self.current_day_updates}")
    
    def force_termination(self, reason: DataTerminationReason):
        """Force termination of adaptive data lifecycle"""
        self.should_terminate = True
        self.termination_reason = reason
        self.logger.info(f"🛑 Adaptive data lifecycle terminated: {reason.value}")