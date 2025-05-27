"""Day Manager for handling full trading day data and reset points.

This module manages:
- Loading full trading days with lookback
- Managing multiple reset points within a day
- Day caching and preloading
- Transition between trading days
"""

import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Iterator, Any, NamedTuple

import pandas as pd

from data.data_manager import DataManager
from data.utils.index_utils import IndexManager, MomentumDay
from envs.environment_simulator import ResetPoint, DayData


class TradingDay(NamedTuple):
    """Container for a full trading day with reset points."""
    date: datetime
    symbol: str
    quality_score: float
    reset_points: List[ResetPoint]
    data: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any] = {}


class ResetPointSelector:
    """Selects reset points based on various strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selection_strategy = config.get('selection_strategy', 'quality_weighted')
        self.quality_threshold = config.get('quality_threshold', 0.6)
        self.diversity_bonus = config.get('diversity_bonus', 0.1)
        self.recency_penalty = config.get('recency_penalty', 0.05)
        self.pattern_weights = config.get('pattern_weights', {})
        self.time_preferences = config.get('time_preferences', {})
    
    def select_next(self, reset_points: List[ResetPoint], 
                   used_indices: List[int],
                   prefer_high_quality: bool = False) -> ResetPoint:
        """Select next reset point."""
        available = [rp for i, rp in enumerate(reset_points) if i not in used_indices]
        
        if not available:
            raise ValueError("No available reset points")
        
        if len(available) == 1:
            return available[0]
        
        # Calculate weights
        weights = self._calculate_selection_weights(reset_points, used_indices)
        
        # Apply quality preference if requested
        if prefer_high_quality:
            for i, rp in enumerate(reset_points):
                if i not in used_indices:
                    weights[i] *= (1 + rp.quality_score)
        
        # Normalize weights
        available_weights = [weights[i] for i in range(len(reset_points)) if i not in used_indices]
        total = sum(available_weights)
        
        if total == 0:
            # Fallback to uniform selection
            return available[0]
        
        probs = [w / total for w in available_weights]
        
        # TODO: Implement weighted random selection
        # For now, select highest weight
        max_idx = available_weights.index(max(available_weights))
        return available[max_idx]
    
    def _calculate_selection_weights(self, reset_points: List[ResetPoint], 
                                   used_indices: List[int]) -> List[float]:
        """Calculate selection weights for reset points."""
        weights = []
        
        for i, rp in enumerate(reset_points):
            weight = rp.quality_score
            
            # Pattern weight
            if rp.pattern in self.pattern_weights:
                weight *= self.pattern_weights[rp.pattern]
            
            # Diversity bonus for unused points
            if i not in used_indices:
                weight *= (1 + self.diversity_bonus)
            
            # Time preference
            hour = rp.timestamp.hour
            if hour < 11:
                time_period = 'morning'
            elif hour < 14:
                time_period = 'midday'
            else:
                time_period = 'afternoon'
            
            if time_period in self.time_preferences:
                weight *= self.time_preferences[time_period]
            
            weights.append(weight)
        
        return weights
    
    def filter_by_quality(self, reset_points: List[ResetPoint]) -> List[ResetPoint]:
        """Filter reset points by minimum quality."""
        return [rp for rp in reset_points if rp.quality_score >= self.quality_threshold]


class DayTransitionHandler:
    """Handles transitions between trading days."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def check_position_carryover(self, portfolio_state: Any, 
                                allow_carryover: bool) -> Dict[str, Any]:
        """Check if positions need to be carried over or closed."""
        positions = portfolio_state.positions
        
        if not positions:
            return {
                'has_position': False,
                'requires_closure': False
            }
        
        # Check each position
        for symbol, position in positions.items():
            if position.get('quantity', 0) != 0:
                return {
                    'has_position': True,
                    'requires_closure': not allow_carryover,
                    'position_details': {
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'side': position.get('side', 'unknown')
                    }
                }
        
        return {
            'has_position': False,
            'requires_closure': False
        }
    
    def handle_day_transition(self, current_day: TradingDay,
                            next_day: TradingDay,
                            portfolio_state: Any,
                            force_position_close: bool) -> Dict[str, Any]:
        """Handle transition from one day to another."""
        result = {
            'from_date': current_day.date,
            'to_date': next_day.date,
            'position_closed': False,
            'day_return': 0.0
        }
        
        # Calculate day return
        if hasattr(portfolio_state, 'initial_value') and hasattr(portfolio_state, 'total_value'):
            result['day_return'] = (portfolio_state.total_value / portfolio_state.initial_value) - 1.0
        
        # Check positions
        position_info = self.check_position_carryover(portfolio_state, not force_position_close)
        
        if position_info['requires_closure']:
            result['position_closed'] = True
            self.logger.info(f"Closing position at day transition: {position_info['position_details']}")
        
        return result
    
    def generate_day_summary(self, date: datetime, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary for a day."""
        summary = {
            'date': date,
            'total_pnl': metrics.get('total_pnl', 0),
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': 0.0,
            'episodes_completed': metrics.get('episodes_completed', 0)
        }
        
        # Calculate win rate
        if metrics.get('total_trades', 0) > 0:
            summary['win_rate'] = metrics.get('winning_trades', 0) / metrics['total_trades']
        
        # Performance grade
        if summary['total_pnl'] > 1000:
            grade = 'A'
        elif summary['total_pnl'] > 500:
            grade = 'B'
        elif summary['total_pnl'] > 0:
            grade = 'C'
        elif summary['total_pnl'] > -500:
            grade = 'D'
        else:
            grade = 'F'
        
        summary['performance_grade'] = grade
        
        return summary
    
    def preserve_training_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve important training state across days."""
        preserved = {
            'episode_count': state.get('episode_count', 0),
            'curriculum_stage': state.get('curriculum_stage', 'stage_1'),
            'performance_history': state.get('performance_history', [])
        }
        return preserved
    
    def restore_training_state(self, preserved: Dict[str, Any]) -> Dict[str, Any]:
        """Restore training state."""
        return preserved.copy()


class DayManager:
    """Manages full trading day data loading and caching."""
    
    def __init__(self, config: Dict[str, Any], data_manager: DataManager,
                 index_manager: IndexManager, symbol: str, 
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.data_manager = data_manager
        self.index_manager = index_manager
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)
        
        # Trading hours configuration
        self.session_start = config['trading_hours']['session_start']
        self.session_end = config['trading_hours']['session_end']
        self.market_open = config['trading_hours']['market_open']
        self.market_close = config['trading_hours']['market_close']
        
        # Preloading configuration
        self.preloading_enabled = config['preloading']['enabled']
        self.background_thread = config['preloading']['background_thread']
        self.cache_size = config['preloading']['cache_size']
        
        # Reset point configuration
        self.fixed_reset_times = config['reset_points']['fixed_times']
        self.min_reset_spacing_minutes = config['reset_points']['min_spacing_minutes']
        self.max_reset_points = config['reset_points']['max_per_day']
        
        # State
        self.current_day: Optional[TradingDay] = None
        self._day_cache: OrderedDict[datetime, TradingDay] = OrderedDict()
        self._completed_days: set = set()
        self._preload_executor = ThreadPoolExecutor(max_workers=1) if self.background_thread else None
        
        # Components
        self.reset_selector = ResetPointSelector(config.get('reset_selection', {}))
        self.transition_handler = DayTransitionHandler(self.logger)
        
        # Curriculum constraints
        self._min_quality = 0.5
        self._max_difficulty = 1.0
    
    def load_day(self, momentum_day: MomentumDay) -> TradingDay:
        """Load a full trading day."""
        # Check cache first
        if momentum_day.date in self._day_cache:
            self.logger.info(f"Using cached day: {momentum_day.date}")
            trading_day = self._day_cache[momentum_day.date]
            self.current_day = trading_day
            return trading_day
        
        # Load day data
        success = self.data_manager.load_day(
            symbol=self.symbol,
            date=momentum_day.date,
            with_lookback=True
        )
        
        if not success:
            raise ValueError(f"Failed to load day data for {momentum_day.date}")
        
        # Get day data
        day_data_dict = self.data_manager.get_day_data(self.symbol, momentum_day.date)
        
        # Validate data
        if not self._validate_day_data(day_data_dict):
            raise ValueError("Invalid or incomplete day data")
        
        # Create reset points
        reset_points = self._create_reset_points(momentum_day)
        
        # Create trading day
        trading_day = TradingDay(
            date=momentum_day.date,
            symbol=self.symbol,
            quality_score=momentum_day.quality_score,
            reset_points=reset_points,
            data=day_data_dict,
            metadata={
                'max_intraday_move': momentum_day.max_intraday_move,
                'volume_multiplier': momentum_day.volume_multiplier
            }
        )
        
        # Cache day
        self._cache_day(trading_day)
        self.current_day = trading_day
        
        self.logger.info(f"Loaded trading day {momentum_day.date} with {len(reset_points)} reset points")
        
        return trading_day
    
    def _validate_day_data(self, day_data: Dict[str, Any]) -> bool:
        """Validate day data completeness."""
        required_keys = ['ohlcv_1s', 'quotes', 'trades']
        
        for key in required_keys:
            if key not in day_data:
                return False
            
            data = day_data[key]
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                return False
        
        return True
    
    def _create_reset_points(self, momentum_day: MomentumDay) -> List[ResetPoint]:
        """Create reset points combining momentum and fixed times."""
        reset_points = []
        
        # Add momentum reset points
        reset_points.extend(momentum_day.reset_points)
        
        # Add fixed reset times
        for fixed_time in self.fixed_reset_times:
            timestamp = momentum_day.date.replace(
                hour=fixed_time.hour,
                minute=fixed_time.minute,
                second=0,
                microsecond=0
            )
            
            # Check if time already exists
            exists = any(abs((rp.timestamp - timestamp).total_seconds()) < 60 
                        for rp in reset_points)
            
            if not exists:
                reset_points.append(ResetPoint(
                    timestamp=timestamp,
                    pattern=self._get_pattern_for_time(fixed_time),
                    phase='neutral',
                    quality_score=0.7,
                    metadata={'fixed_time': True}
                ))
        
        # Sort by timestamp
        reset_points.sort(key=lambda rp: rp.timestamp)
        
        # Apply spacing constraints
        reset_points = self._apply_spacing_constraints(reset_points)
        
        # Limit total number
        if len(reset_points) > self.max_reset_points:
            # Keep highest quality points
            reset_points.sort(key=lambda rp: rp.quality_score, reverse=True)
            reset_points = reset_points[:self.max_reset_points]
            reset_points.sort(key=lambda rp: rp.timestamp)
        
        return reset_points
    
    def _get_pattern_for_time(self, fixed_time: time) -> str:
        """Get pattern name for fixed time."""
        if fixed_time == time(9, 30):
            return 'market_open'
        elif fixed_time == time(10, 30):
            return 'post_open'
        elif fixed_time == time(14, 0):
            return 'afternoon'
        elif fixed_time == time(15, 30):
            return 'power_hour'
        else:
            return 'fixed_time'
    
    def _apply_spacing_constraints(self, reset_points: List[ResetPoint]) -> List[ResetPoint]:
        """Apply minimum spacing between reset points."""
        if not reset_points:
            return reset_points
        
        filtered = [reset_points[0]]
        
        for rp in reset_points[1:]:
            time_diff = (rp.timestamp - filtered[-1].timestamp).total_seconds() / 60
            
            if time_diff >= self.min_reset_spacing_minutes:
                filtered.append(rp)
            else:
                # Keep higher quality point
                if rp.quality_score > filtered[-1].quality_score:
                    filtered[-1] = rp
        
        return filtered
    
    def _cache_day(self, trading_day: TradingDay):
        """Cache trading day with LRU eviction."""
        self._day_cache[trading_day.date] = trading_day
        
        # Evict oldest if over limit
        while len(self._day_cache) > self.cache_size:
            oldest_date = next(iter(self._day_cache))
            del self._day_cache[oldest_date]
    
    def get_current_day(self) -> Optional[TradingDay]:
        """Get current trading day."""
        return self.current_day
    
    def mark_day_completed(self, trading_day: TradingDay):
        """Mark a day as completed."""
        self._completed_days.add(trading_day.date)
    
    def preload_next_day(self) -> Optional[Future]:
        """Preload next trading day in background."""
        if not self.preloading_enabled:
            return None
        
        # Get next momentum day
        next_momentum_day = self.index_manager.get_next_day(
            self.current_day.date if self.current_day else None,
            min_quality=self._min_quality
        )
        
        if not next_momentum_day:
            return None
        
        # Check if already cached
        if next_momentum_day.date in self._day_cache:
            return None
        
        # Preload asynchronously
        if self._preload_executor:
            future = self._preload_executor.submit(
                self._preload_day_async,
                next_momentum_day
            )
            return future
        else:
            # Synchronous preload
            self._preload_day_async(next_momentum_day)
            future = Future()
            future.set_result(True)
            return future
    
    def _preload_day_async(self, momentum_day: MomentumDay) -> bool:
        """Asynchronously preload a day."""
        try:
            return self.data_manager.preload_day_async(
                symbol=self.symbol,
                date=momentum_day.date,
                with_lookback=True
            ).result()
        except Exception as e:
            self.logger.error(f"Failed to preload day {momentum_day.date}: {e}")
            return False
    
    def get_next_trading_day(self) -> TradingDay:
        """Get next trading day, using preloaded if available."""
        # Get next momentum day
        next_momentum_day = self.index_manager.get_next_day(
            self.current_day.date if self.current_day else None,
            min_quality=self._min_quality
        )
        
        if not next_momentum_day:
            raise ValueError("No next trading day available")
        
        # Load the day
        return self.load_day(next_momentum_day)
    
    def create_reset_iterator(self, trading_day: TradingDay) -> Iterator[ResetPoint]:
        """Create iterator for reset points."""
        for reset_point in trading_day.reset_points:
            yield reset_point
    
    def set_curriculum_constraints(self, min_quality: float, max_difficulty: float):
        """Set constraints for curriculum-based selection."""
        self._min_quality = min_quality
        self._max_difficulty = max_difficulty
    
    def cleanup_old_days(self):
        """Clean up old completed days from memory."""
        # Remove completed days older than cache size
        if len(self._completed_days) > self.cache_size * 2:
            sorted_days = sorted(self._completed_days)
            to_remove = sorted_days[:len(sorted_days) - self.cache_size]
            
            for day in to_remove:
                self._completed_days.discard(day)
                if day in self._day_cache:
                    del self._day_cache[day]
    
    def close(self):
        """Clean up resources."""
        if self._preload_executor:
            self._preload_executor.shutdown(wait=False)