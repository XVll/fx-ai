"""Momentum Episode Manager for curriculum-based episode selection.

This module manages:
- Loading and querying pre-computed momentum indices
- Curriculum-based episode selection
- Performance-adaptive difficulty adjustment
- Episode tracking and statistics
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Any, NamedTuple

import numpy as np

from data.utils.index_utils import IndexManager, MomentumDay
from envs.environment_simulator import ResetPoint


class PatternType(Enum):
    """Trading pattern types."""
    BREAKOUT = "breakout"
    FLUSH = "flush"
    BOUNCE = "bounce"
    CONSOLIDATION = "consolidation"
    PARABOLIC = "parabolic"
    MARKET_OPEN = "market_open"
    POWER_HOUR = "power_hour"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    DEAD = "dead"


class PhaseType(Enum):
    """Market phase types."""
    FRONT_SIDE = "front_side"
    BACK_SIDE = "back_side"
    NEUTRAL = "neutral"
    RECOVERY = "recovery"
    EXHAUSTION = "exhaustion"
    ACCUMULATION = "accumulation"


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    episodes: List[int]  # [start, end]
    min_quality: float
    patterns: Optional[List[str]] = None
    avoid_patterns: Optional[List[str]] = None


@dataclass
class MomentumContext:
    """Context information for momentum episodes."""
    pattern: PatternType
    phase: PhaseType
    quality_score: float
    day_quality: float = 0.0
    intraday_move: float = 0.0
    volume_multiplier: float = 1.0
    time_of_day: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def calculate_momentum_strength(self) -> float:
        """Calculate overall momentum strength."""
        # Pattern strength
        pattern_strengths = {
            PatternType.BREAKOUT: 1.0,
            PatternType.PARABOLIC: 0.9,
            PatternType.BOUNCE: 0.8,
            PatternType.POWER_HOUR: 0.7,
            PatternType.FLUSH: 0.6,
            PatternType.CONSOLIDATION: 0.3,
            PatternType.DEAD: 0.1
        }
        pattern_strength = pattern_strengths.get(self.pattern, 0.5)
        
        # Phase strength
        phase_strengths = {
            PhaseType.FRONT_SIDE: 1.0,
            PhaseType.RECOVERY: 0.8,
            PhaseType.NEUTRAL: 0.5,
            PhaseType.BACK_SIDE: 0.3,
            PhaseType.EXHAUSTION: 0.2
        }
        phase_strength = phase_strengths.get(self.phase, 0.5)
        
        # Calculate weighted strength
        strength = (
            pattern_strength * 0.4 +
            phase_strength * 0.3 +
            self.quality_score * 0.2 +
            min(self.volume_multiplier / 5.0, 1.0) * 0.1
        )
        
        return strength
    
    def get_action_hints(self) -> Dict[str, str]:
        """Get action hints based on context."""
        hints = {
            'preferred_side': 'neutral',
            'urgency': 'medium',
            'risk_level': 'medium'
        }
        
        # Pattern-based hints
        if self.pattern in [PatternType.BREAKOUT, PatternType.BOUNCE]:
            hints['preferred_side'] = 'buy'
            hints['urgency'] = 'high'
        elif self.pattern in [PatternType.FLUSH, PatternType.DISTRIBUTION]:
            hints['preferred_side'] = 'sell'
            hints['urgency'] = 'high'
            hints['risk_level'] = 'high'
        elif self.pattern == PatternType.PARABOLIC:
            hints['preferred_side'] = 'sell'  # Look for reversal
            hints['urgency'] = 'medium'
            hints['risk_level'] = 'very_high'
        
        return hints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern': self.pattern.value,
            'phase': self.phase.value,
            'quality_score': self.quality_score,
            'day_quality': self.day_quality,
            'intraday_move': self.intraday_move,
            'volume_multiplier': self.volume_multiplier,
            'time_of_day': self.time_of_day,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MomentumContext':
        """Create from dictionary."""
        return cls(
            pattern=PatternType(data['pattern']),
            phase=PhaseType(data['phase']),
            quality_score=data['quality_score'],
            day_quality=data.get('day_quality', 0.0),
            intraday_move=data.get('intraday_move', 0.0),
            volume_multiplier=data.get('volume_multiplier', 1.0),
            time_of_day=data.get('time_of_day'),
            metadata=data.get('metadata', {})
        )


class Episode(NamedTuple):
    """Episode definition."""
    day: MomentumDay
    reset_point: ResetPoint
    id: str
    last_used: Optional[datetime] = None


class EpisodeSelector:
    """Selects episodes based on various strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights_config = config.get('weights', {
            'quality': 0.4,
            'diversity': 0.2,
            'recency': 0.1,
            'performance': 0.3
        })
        self.diversity_bonus_per_use = config.get('diversity_bonus_per_use', 0.05)
        self.recency_penalty_days = config.get('recency_penalty_days', 3)
        self.performance_adjustment_range = config.get('performance_adjustment_range', 0.2)
    
    def calculate_weights(self, episodes: List[Episode], 
                         used_episodes: List[Episode],
                         performance_score: float) -> List[float]:
        """Calculate selection weights for episodes."""
        weights = []
        used_ids = {ep.id for ep in used_episodes}
        
        for episode in episodes:
            # Base quality weight
            quality_weight = episode.reset_point.quality_score * self.weights_config['quality']
            
            # Diversity bonus
            diversity_weight = 0.0
            if episode.id not in used_ids:
                diversity_weight = self.weights_config['diversity'] * (1 + self.diversity_bonus_per_use)
            
            # Recency penalty
            recency_weight = self.weights_config['recency']
            if episode.last_used:
                days_since_used = (datetime.now() - episode.last_used).days
                if days_since_used < self.recency_penalty_days:
                    recency_weight *= (days_since_used / self.recency_penalty_days)
            
            # Performance adjustment
            performance_weight = self.weights_config['performance']
            # High performance -> prefer harder episodes
            # Low performance -> prefer easier episodes
            difficulty_preference = performance_score - 0.5  # -0.5 to 0.5
            episode_difficulty = episode.reset_point.quality_score - 0.7  # Normalized
            alignment = 1.0 + (difficulty_preference * episode_difficulty * self.performance_adjustment_range)
            performance_weight *= alignment
            
            # Combine weights
            total_weight = quality_weight + diversity_weight + recency_weight + performance_weight
            weights.append(max(total_weight, 0.01))  # Minimum weight
        
        return weights
    
    def weighted_random_select(self, episodes: List[Episode], weights: List[float]) -> Episode:
        """Select episode using weighted random selection."""
        if not episodes:
            raise ValueError("No episodes to select from")
        
        if len(episodes) == 1:
            return episodes[0]
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            # Uniform selection
            return episodes[0]
        
        probs = [w / total for w in weights]
        
        # Random selection based on weights
        # TODO: Use proper random state
        cumsum = np.cumsum(probs)
        r = np.random.random()
        idx = np.searchsorted(cumsum, r)
        idx = min(idx, len(episodes) - 1)
        
        return episodes[idx]


class MomentumEpisodeManager:
    """Manages episode selection from momentum indices."""
    
    def __init__(self, config: Dict[str, Any], index_manager: IndexManager,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.index_manager = index_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration
        self.indices_config = config['indices']
        self.curriculum_config = config['curriculum']
        self.selection_config = config['selection']
        self.pattern_preferences = config.get('pattern_preferences', {})
        
        # Load index for symbol
        self.symbol = self.indices_config['symbols'][0]  # Primary symbol
        self.momentum_index = self.index_manager.load_index(self.symbol)
        
        # Curriculum setup
        self.curriculum_enabled = self.curriculum_config['enabled']
        self.curriculum_stages = self._create_curriculum_stages()
        
        # Selection strategy
        self.selection_strategy = self.selection_config['strategy']
        self.episode_selector = EpisodeSelector(self.selection_config)
        
        # Tracking
        self.episode_count = 0
        self.episodes_completed = 0
        self._completed_episodes: List[Episode] = []
        self._episode_cache: Dict[str, Episode] = {}
        
        # Performance tracking
        self.performance_tracker = {
            'episode_count': 0,
            'recent_returns': [],
            'win_rate': 0.5,
            'avg_episode_duration': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
    
    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create curriculum stages from config."""
        stages = []
        
        for stage_name, stage_config in self.curriculum_config['stages'].items():
            stage = CurriculumStage(
                name=stage_name,
                episodes=stage_config['episodes'],
                min_quality=stage_config['min_quality'],
                patterns=stage_config.get('patterns'),
                avoid_patterns=stage_config.get('avoid_patterns')
            )
            stages.append(stage)
        
        return stages
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        for stage in self.curriculum_stages:
            if stage.episodes[0] <= self.episode_count < (stage.episodes[1] or float('inf')):
                return stage
        return self.curriculum_stages[-1]
    
    def get_available_days(self) -> List[MomentumDay]:
        """Get available momentum days for current stage."""
        stage = self.get_current_stage()
        
        # Get days from index
        all_days = self.momentum_index.get(self.symbol, [])
        
        # Filter by quality
        days = [d for d in all_days if d.quality_score >= stage.min_quality]
        
        # Filter by patterns if specified
        if stage.patterns or stage.avoid_patterns:
            filtered_days = []
            
            for day in days:
                # Check if day has allowed patterns
                day_patterns = {rp.pattern for rp in day.reset_points}
                
                if stage.patterns:
                    if not any(p in day_patterns for p in stage.patterns):
                        continue
                
                if stage.avoid_patterns:
                    if any(p in day_patterns for p in stage.avoid_patterns):
                        continue
                
                filtered_days.append(day)
            
            days = filtered_days
        
        return days
    
    def _get_candidate_episodes(self) -> List[Episode]:
        """Get all candidate episodes."""
        episodes = []
        available_days = self.get_available_days()
        
        for day in available_days:
            for reset_point in day.reset_points:
                episode_id = f"{day.symbol}_{day.date.strftime('%Y%m%d')}_{reset_point.timestamp.strftime('%H%M')}"
                
                # Check cache
                if episode_id in self._episode_cache:
                    episode = self._episode_cache[episode_id]
                else:
                    episode = Episode(
                        day=day,
                        reset_point=reset_point,
                        id=episode_id
                    )
                    self._episode_cache[episode_id] = episode
                
                episodes.append(episode)
        
        return episodes
    
    def select_next_episode(self) -> Episode:
        """Select next episode based on strategy."""
        candidates = self._get_candidate_episodes()
        
        if not candidates:
            raise RuntimeError("No candidate episodes available")
        
        if self.selection_strategy == 'random':
            # Random selection
            idx = np.random.randint(0, len(candidates))
            return candidates[idx]
        
        elif self.selection_strategy == 'sequential':
            # Sequential by timestamp
            candidates.sort(key=lambda ep: ep.reset_point.timestamp)
            
            # Find first unused
            for episode in candidates:
                if episode not in self._completed_episodes:
                    return episode
            
            # All used, start over
            return candidates[0]
        
        elif self.selection_strategy == 'adaptive':
            # Adaptive selection based on performance
            performance_score = self._calculate_performance_score()
            
            # Calculate weights
            weights = self.episode_selector.calculate_weights(
                candidates,
                self._completed_episodes,
                performance_score
            )
            
            # Select episode
            return self.episode_selector.weighted_random_select(candidates, weights)
        
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def _calculate_performance_score(self) -> float:
        """Calculate current performance score (0-1)."""
        if not self.performance_tracker['recent_returns']:
            return 0.5  # Neutral
        
        # Factors to consider
        win_rate = self.performance_tracker['win_rate']
        avg_return = np.mean(self.performance_tracker['recent_returns'][-20:])
        
        # Normalize to 0-1
        # Win rate is already 0-1
        # Returns: assume -5% to +5% maps to 0-1
        return_score = np.clip((avg_return + 0.05) / 0.10, 0, 1)
        
        # Weighted average
        score = win_rate * 0.6 + return_score * 0.4
        
        return np.clip(score, 0, 1)
    
    def _calculate_target_difficulty(self) -> float:
        """Calculate target difficulty based on performance."""
        performance_score = self._calculate_performance_score()
        
        # Map performance to difficulty
        # High performance -> higher difficulty
        # Low performance -> lower difficulty
        return performance_score
    
    def create_momentum_context(self, episode: Episode) -> MomentumContext:
        """Create momentum context for episode."""
        return MomentumContext(
            pattern=PatternType(episode.reset_point.pattern),
            phase=PhaseType(episode.reset_point.phase),
            quality_score=episode.reset_point.quality_score,
            day_quality=episode.day.quality_score,
            intraday_move=episode.day.max_intraday_move,
            volume_multiplier=episode.day.volume_multiplier,
            time_of_day=self._get_time_of_day(episode.reset_point.timestamp),
            metadata=episode.reset_point.metadata
        )
    
    def _get_time_of_day(self, timestamp: datetime) -> str:
        """Categorize time of day."""
        hour = timestamp.hour
        
        if hour < 7:
            return 'pre_market'
        elif hour < 10:
            return 'market_open'
        elif hour < 12:
            return 'morning'
        elif hour < 14:
            return 'midday'
        elif hour < 16:
            return 'afternoon'
        else:
            return 'after_hours'
    
    def mark_episode_used(self, episode: Episode):
        """Mark episode as used."""
        # Update last used time
        updated = Episode(
            day=episode.day,
            reset_point=episode.reset_point,
            id=episode.id,
            last_used=datetime.now()
        )
        
        # Update cache
        self._episode_cache[episode.id] = updated
        
        # Add to completed if not already there
        if not any(ep.id == episode.id for ep in self._completed_episodes):
            self._completed_episodes.append(updated)
    
    def complete_episode(self, episode: Episode, metrics: Dict[str, Any]):
        """Record episode completion."""
        self.mark_episode_used(episode)
        self.episodes_completed += 1
        
        # Update performance tracker
        self.performance_tracker['episode_count'] += 1
        
        if 'total_return' in metrics:
            self.performance_tracker['recent_returns'].append(metrics['total_return'])
            # Keep only recent history
            if len(self.performance_tracker['recent_returns']) > 100:
                self.performance_tracker['recent_returns'].pop(0)
        
        if metrics.get('win', False):
            self.performance_tracker['consecutive_wins'] += 1
            self.performance_tracker['consecutive_losses'] = 0
        else:
            self.performance_tracker['consecutive_losses'] += 1
            self.performance_tracker['consecutive_wins'] = 0
        
        # Update win rate
        if self.episodes_completed > 0:
            wins = sum(1 for ep_metrics in self.performance_tracker.get('all_metrics', []) 
                      if ep_metrics.get('win', False))
            self.performance_tracker['win_rate'] = wins / self.episodes_completed
    
    def get_episodes_by_time_range(self, start_time: time, end_time: time) -> List[Episode]:
        """Get episodes within time range."""
        candidates = self._get_candidate_episodes()
        
        filtered = []
        for episode in candidates:
            ep_time = episode.reset_point.timestamp.time()
            if start_time <= ep_time <= end_time:
                filtered.append(episode)
        
        return filtered
    
    def _get_all_episodes_sorted(self) -> List[Episode]:
        """Get all episodes sorted by timestamp."""
        episodes = self._get_candidate_episodes()
        episodes.sort(key=lambda ep: ep.reset_point.timestamp)
        return episodes
    
    def refresh_indices(self):
        """Refresh momentum indices from disk."""
        self.momentum_index = self.index_manager.load_index(self.symbol)
        self._episode_cache.clear()
        self.logger.info(f"Refreshed momentum indices for {self.symbol}")
    
    def save_state(self) -> Dict[str, Any]:
        """Save manager state."""
        return {
            'episode_count': self.episode_count,
            'episodes_completed': self.episodes_completed,
            'completed_episodes': [ep.id for ep in self._completed_episodes],
            'performance_history': self.performance_tracker
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load manager state."""
        self.episode_count = state.get('episode_count', 0)
        self.episodes_completed = state.get('episodes_completed', 0)
        
        # Restore completed episodes
        completed_ids = state.get('completed_episodes', [])
        # TODO: Reconstruct Episode objects from IDs
        
        # Restore performance history
        if 'performance_history' in state:
            self.performance_tracker.update(state['performance_history'])