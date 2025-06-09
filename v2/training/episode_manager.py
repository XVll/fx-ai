"""
Episode Manager - Authority for Training Episode Management
Handles reset point cycling, day selection, stage management, and episode-related termination.
Renamed from DataLifecycleManager for better clarity.
"""

import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum


class EpisodeTerminationReason(Enum):
    """Reasons for episode manager termination"""
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
    """Episode selection modes"""
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
class EpisodeState:
    """Current episode state for episode manager"""
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


class EpisodeManager:
    """
    Episode Manager - Authority for Training Episode Management
    
    Responsibilities:
    - Day selection and rotation
    - Reset point cycling within days
    - Episode progression tracking
    - Episode-related termination decisions
    """
    
    def __init__(self, config, data_manager):
        """Initialize episode manager with configuration and data manager."""
        # TODO: Implement initialization
        pass
    
    def initialize(self) -> bool:
        """Initialize episode manager - load days and reset points."""
        # TODO: Move from v1 DataLifecycleManager.initialize()
        return True
    
    def should_terminate(self) -> Optional[EpisodeTerminationReason]:
        """Check if episode manager should terminate."""
        # TODO: Move from v1 DataLifecycleManager.should_terminate_data_lifecycle()
        return None
    
    def get_current_episode_config(self) -> Optional[Dict[str, Any]]:
        """Get current episode configuration."""
        # TODO: Move from v1 DataLifecycleManager.get_current_training_data()
        return {}
    
    def advance_episode(self) -> bool:
        """Advance to next episode/reset point."""
        # TODO: Move from v1 DataLifecycleManager.advance_cycle_on_episode_completion()
        return True
    
    def update_progress(self, episodes: int, updates: int):
        """Update progress counters."""
        # TODO: Move from v1 DataLifecycleManager.update_progress()
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current episode manager status."""
        # TODO: Move from v1 DataLifecycleManager.get_data_lifecycle_status()
        return {}