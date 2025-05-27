"""Utility functions for working with momentum indices."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging


class IndexManager:
    """Manages momentum index queries and curriculum-based selection."""
    
    def __init__(self, momentum_days_df: pd.DataFrame, 
                 reset_points_df: pd.DataFrame,
                 logger: Optional[logging.Logger] = None):
        """Initialize index manager.
        
        Args:
            momentum_days_df: DataFrame with momentum day index
            reset_points_df: DataFrame with reset point index
            logger: Optional logger
        """
        self.momentum_days = momentum_days_df
        self.reset_points = reset_points_df
        self.logger = logger or logging.getLogger(__name__)
        
        # Track usage for anti-overfitting
        self.usage_counts = {}  # {(symbol, date): count}
        
        # Curriculum stage tracking
        self.total_episodes = 0
        
    def select_training_day(self, symbol: str, 
                           curriculum_stage: Optional[str] = None,
                           exclude_dates: Optional[List[datetime]] = None) -> Optional[pd.Series]:
        """Select a training day based on curriculum and diversity requirements.
        
        Args:
            symbol: Symbol to select day for
            curriculum_stage: Optional stage override ('early', 'intermediate', 'advanced')
            exclude_dates: Dates to exclude from selection
            
        Returns:
            Series with day information or None if no days available
        """
        # Get symbol's momentum days
        symbol_days = self.momentum_days[
            self.momentum_days['symbol'] == symbol.upper()
        ].copy()
        
        if symbol_days.empty:
            self.logger.warning(f"No momentum days found for {symbol}")
            return None
            
        # Apply exclusions
        if exclude_dates:
            exclude_dates = [pd.Timestamp(d).date() for d in exclude_dates]
            symbol_days = symbol_days[~symbol_days['date'].isin(exclude_dates)]
            
        if symbol_days.empty:
            return None
            
        # Determine curriculum stage if not provided
        if curriculum_stage is None:
            curriculum_stage = self._get_curriculum_stage()
            
        # Apply curriculum-based filtering
        symbol_days = self._apply_curriculum_filter(symbol_days, curriculum_stage)
        
        if symbol_days.empty:
            return None
            
        # Apply diversity weighting
        symbol_days = self._apply_diversity_weights(symbol_days)
        
        # Select day based on weighted probability
        weights = symbol_days['selection_weight'].values
        weights = weights / weights.sum()
        
        selected_idx = np.random.choice(len(symbol_days), p=weights)
        selected_day = symbol_days.iloc[selected_idx]
        
        # Update usage count
        day_key = (symbol.upper(), selected_day['date'])
        self.usage_counts[day_key] = self.usage_counts.get(day_key, 0) + 1
        
        return selected_day
        
    def get_day_reset_points(self, symbol: str, date: datetime,
                           max_points: Optional[int] = None) -> pd.DataFrame:
        """Get reset points for a specific day.
        
        Args:
            symbol: Symbol
            date: Date to get reset points for
            max_points: Maximum number of points to return
            
        Returns:
            DataFrame with reset points sorted by quality
        """
        date_obj = pd.Timestamp(date).date()
        
        points = self.reset_points[
            (self.reset_points['symbol'] == symbol.upper()) &
            (self.reset_points['date'] == date_obj)
        ].copy()
        
        if points.empty:
            return pd.DataFrame()
            
        # Sort by combined quality score
        points = points.sort_values('combined_score', ascending=False)
        
        if max_points:
            points = points.head(max_points)
            
        return points
        
    def _get_curriculum_stage(self) -> str:
        """Determine curriculum stage based on episode count."""
        if self.total_episodes < 10000:
            return 'early'
        elif self.total_episodes < 50000:
            return 'intermediate'
        else:
            return 'advanced'
            
    def _apply_curriculum_filter(self, days_df: pd.DataFrame, 
                                stage: str) -> pd.DataFrame:
        """Apply curriculum-based filtering to momentum days."""
        if stage == 'early':
            # Focus on high-quality days
            quality_threshold = 0.8
            days_df = days_df[days_df['quality_score'] >= quality_threshold]
            
            # Categorize days
            days_df['category'] = pd.cut(
                days_df['quality_score'],
                bins=[0.8, 0.85, 0.9, 1.0],
                labels=['good', 'very_good', 'excellent']
            )
            
            # Set selection probabilities
            category_weights = {
                'good': 0.2,
                'very_good': 0.3,
                'excellent': 0.5
            }
            days_df['curriculum_weight'] = days_df['category'].map(category_weights)
            
        elif stage == 'intermediate':
            # Broader quality range
            quality_threshold = 0.6
            days_df = days_df[days_df['quality_score'] >= quality_threshold]
            
            # Categorize with more bins
            days_df['category'] = pd.cut(
                days_df['quality_score'],
                bins=[0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['moderate', 'good', 'very_good', 'excellent']
            )
            
            category_weights = {
                'moderate': 0.2,
                'good': 0.3,
                'very_good': 0.3,
                'excellent': 0.2
            }
            days_df['curriculum_weight'] = days_df['category'].map(category_weights)
            
        else:  # advanced
            # Include all days with equal weighting
            days_df['curriculum_weight'] = 1.0
            
        return days_df
        
    def _apply_diversity_weights(self, days_df: pd.DataFrame) -> pd.DataFrame:
        """Apply diversity weights based on usage counts."""
        # Calculate usage weight (inverse of usage count)
        def get_usage_weight(row):
            key = (row['symbol'], row['date'])
            usage = self.usage_counts.get(key, 0)
            # Exponential decay: weight = exp(-usage/10)
            return np.exp(-usage / 10.0)
            
        days_df['usage_weight'] = days_df.apply(get_usage_weight, axis=1)
        
        # Combine curriculum and usage weights
        if 'curriculum_weight' in days_df.columns:
            days_df['selection_weight'] = days_df['curriculum_weight'] * days_df['usage_weight']
        else:
            days_df['selection_weight'] = days_df['usage_weight']
            
        # Add small random factor for variety
        days_df['selection_weight'] *= (1 + np.random.uniform(-0.1, 0.1, len(days_df)))
        
        # Ensure positive weights
        days_df['selection_weight'] = days_df['selection_weight'].clip(lower=0.01)
        
        return days_df
        
    def get_pattern_distribution(self, symbol: Optional[str] = None) -> Dict[str, int]:
        """Get distribution of pattern types in reset points.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dictionary mapping pattern types to counts
        """
        points = self.reset_points
        
        if symbol:
            points = points[points['symbol'] == symbol.upper()]
            
        return points['pattern_type'].value_counts().to_dict()
        
    def get_quality_distribution(self, symbol: Optional[str] = None,
                               bins: int = 10) -> pd.DataFrame:
        """Get distribution of quality scores.
        
        Args:
            symbol: Optional symbol filter
            bins: Number of bins for histogram
            
        Returns:
            DataFrame with quality score bins and counts
        """
        days = self.momentum_days
        
        if symbol:
            days = days[days['symbol'] == symbol.upper()]
            
        # Create quality score bins
        days['quality_bin'] = pd.cut(days['quality_score'], bins=bins)
        
        return days.groupby('quality_bin').size().reset_index(name='count')
        
    def get_temporal_distribution(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get temporal distribution of momentum days.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            DataFrame with monthly counts
        """
        days = self.momentum_days.copy()
        
        if symbol:
            days = days[days['symbol'] == symbol.upper()]
            
        # Extract year-month
        days['year_month'] = days['date'].dt.to_period('M')
        
        return days.groupby('year_month').size().reset_index(name='count')
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        if not self.usage_counts:
            return {
                'total_unique_days': 0,
                'avg_usage_per_day': 0,
                'max_usage': 0,
                'most_used_days': []
            }
            
        usage_values = list(self.usage_counts.values())
        
        # Get most used days
        sorted_usage = sorted(self.usage_counts.items(), key=lambda x: x[1], reverse=True)
        most_used = [
            {'symbol': k[0], 'date': k[1], 'count': v}
            for k, v in sorted_usage[:10]
        ]
        
        return {
            'total_unique_days': len(self.usage_counts),
            'avg_usage_per_day': np.mean(usage_values),
            'max_usage': max(usage_values),
            'min_usage': min(usage_values),
            'std_usage': np.std(usage_values),
            'most_used_days': most_used
        }
        
    def increment_episode_count(self, count: int = 1):
        """Increment total episode count for curriculum tracking."""
        self.total_episodes += count
        
    def reset_usage_counts(self):
        """Reset usage counts for fresh diversity."""
        self.usage_counts.clear()
        self.logger.info("Reset usage counts for all days")