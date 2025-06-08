"""
Data management interfaces for market data and features.

These interfaces define contracts for data loading, caching,
and feature extraction in a modular way.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, Iterator, runtime_checkable
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import numpy as np

from ..types.common import (
    Symbol, Timestamp, MarketDataPoint, FeatureArray,
    Configurable, Resettable, FeatureFrequency
)


@runtime_checkable
class IDataProvider(Protocol):
    """Interface for market data providers.
    
    Design principles:
    - Support multiple data sources (files, APIs, databases)
    - Enable efficient data iteration
    - Handle different data formats
    - Provide metadata about available data
    """
    
    @property
    def available_symbols(self) -> list[Symbol]:
        """Get list of available symbols.
        
        Returns:
            List of symbols with data
            
        Design notes:
        - Used for validation and discovery
        - Should be cached for performance
        """
        ...
    
    @property
    def data_range(self) -> tuple[datetime, datetime]:
        """Get overall data time range.
        
        Returns:
            Tuple of (start_time, end_time)
            
        Design notes:
        - Used for backtesting bounds
        - Should reflect actual data availability
        """
        ...
    
    def get_data(
        self,
        symbol: Symbol,
        start: Timestamp,
        end: Timestamp,
        data_type: str = "trades"
    ) -> pd.DataFrame:
        """Get market data for symbol and time range.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)
            data_type: Type of data (trades, quotes, bars, etc.)
            
        Returns:
            DataFrame with market data
            
        Design notes:
        - Should handle missing data gracefully
        - Return empty DataFrame if no data
        - Standardize column names across providers
        - Consider memory efficiency for large requests
        """
        ...
    
    def stream_data(
        self,
        symbol: Symbol,
        start: Timestamp,
        data_type: str = "trades",
        chunk_size: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Stream data in chunks for memory efficiency.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp
            data_type: Type of data
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrames of market data
            
        Design notes:
        - Essential for large datasets
        - Should maintain chronological order
        - Allow early termination
        """
        ...
    
    def get_metadata(
        self,
        symbol: Symbol
    ) -> dict[str, Any]:
        """Get metadata for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Metadata dict with keys like:
            - market_cap
            - sector
            - float_shares
            - average_volume
            
        Design notes:
        - Used for feature engineering
        - Cache aggressively
        """
        ...


class IDataManager(Configurable, Resettable):
    """High-level data management interface.
    
    Design principles:
    - Coordinate multiple data providers
    - Handle caching and optimization
    - Provide unified data access
    - Support momentum/reset point detection
    """
    
    @abstractmethod
    def initialize(
        self,
        providers: list[IDataProvider]
    ) -> None:
        """Initialize with data providers.
        
        Args:
            providers: List of data providers to use
            
        Design notes:
        - Validate provider compatibility
        - Build unified symbol list
        - Set up caching strategy
        """
        ...
    
    @abstractmethod
    def load_symbol_data(
        self,
        symbol: Symbol,
        date: date,
        include_previous_day: bool = True
    ) -> dict[str, pd.DataFrame]:
        """Load all data for symbol on given date.
        
        Args:
            symbol: Trading symbol
            date: Trading date
            include_previous_day: Whether to load previous day for continuity
            
        Returns:
            Dict mapping data type to DataFrame
            
        Design notes:
        - Central method for environment setup
        - Handle timezone conversions
        - Cache for repeated access
        """
        ...
    
    @abstractmethod
    def get_momentum_days(
        self,
        symbol: Symbol,
        min_quality_score: float = 0.5
    ) -> list[dict[str, Any]]:
        """Get high-momentum trading days.
        
        Args:
            symbol: Trading symbol
            min_quality_score: Minimum quality threshold
            
        Returns:
            List of momentum day info dicts with:
            - date: Trading date
            - quality_score: Momentum quality (0-1)
            - metrics: Supporting metrics
            
        Design notes:
        - Used for curriculum learning
        - Quality based on volume, volatility, trends
        - Cache results as expensive to compute
        """
        ...
    
    @abstractmethod
    def get_reset_points(
        self,
        symbol: Symbol,
        date: date
    ) -> list[dict[str, Any]]:
        """Get optimal episode reset points.
        
        Args:
            symbol: Trading symbol  
            date: Trading date
            
        Returns:
            List of reset point dicts with:
            - timestamp: Reset timestamp
            - quality_score: Reset quality (0-1)
            - context: Market context at point
            
        Design notes:
        - Reset points are moments of opportunity
        - Based on technical/momentum indicators
        - Balance diversity and quality
        """
        ...


class IFeatureExtractor(Configurable, Resettable):
    """Interface for feature extraction from market data.
    
    Design principles:
    - Modular feature engineering
    - Support multiple timeframes
    - Enable feature versioning
    - Optimize for performance
    """
    
    @abstractmethod
    def extract_features(
        self,
        market_data: dict[str, pd.DataFrame],
        timestamp: datetime,
        lookback_periods: dict[FeatureFrequency, int]
    ) -> dict[FeatureFrequency, FeatureArray]:
        """Extract features at given timestamp.
        
        Args:
            market_data: Dict of DataFrames by data type
            timestamp: Current timestamp
            lookback_periods: Lookback window by frequency
            
        Returns:
            Dict mapping frequency to feature arrays
            
        Design notes:
        - Handle missing data gracefully
        - Maintain feature consistency
        - Consider computational efficiency
        - Support incremental updates
        """
        ...
    
    @abstractmethod
    def get_feature_names(
        self,
        frequency: FeatureFrequency
    ) -> list[str]:
        """Get feature names for frequency.
        
        Args:
            frequency: Feature frequency
            
        Returns:
            List of feature names
            
        Design notes:
        - Used for interpretation
        - Must match extraction order
        """
        ...
    
    @abstractmethod
    def get_feature_metadata(
        self
    ) -> dict[str, dict[str, Any]]:
        """Get metadata about features.
        
        Returns:
            Dict mapping feature name to metadata:
            - description: What it measures
            - frequency: Update frequency
            - dependencies: Required data types
            - normalization: Suggested normalization
            
        Design notes:
        - Used for documentation and validation
        - Helps with feature selection
        """
        ...


class IDataCache(Protocol):
    """Interface for data caching.
    
    Design principles:
    - Reduce redundant computation
    - Support different storage backends
    - Handle cache invalidation
    - Monitor cache performance
    """
    
    def get(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """Get cached value.
        
        Args:
            key: Cache key
            default: Default if not found
            
        Returns:
            Cached value or default
        """
        ...
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        ...
    
    def invalidate(
        self,
        pattern: Optional[str] = None
    ) -> int:
        """Invalidate cache entries.
        
        Args:
            pattern: Key pattern to match (glob-style)
            
        Returns:
            Number of entries invalidated
        """
        ...
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with stats like:
            - hit_rate
            - size_bytes
            - num_entries
        """
        ...


class IMomentumScanner(Configurable):
    """Interface for detecting high-momentum periods.
    
    Design principles:
    - Identify profitable trading opportunities
    - Support multiple momentum indicators
    - Enable curriculum learning
    """
    
    @abstractmethod
    def scan(
        self,
        symbol: Symbol,
        start_date: date,
        end_date: date,
        min_quality: float = 0.0
    ) -> pd.DataFrame:
        """Scan for momentum periods.
        
        Args:
            symbol: Trading symbol
            start_date: Start of scan period
            end_date: End of scan period
            min_quality: Minimum quality threshold
            
        Returns:
            DataFrame with columns:
            - date: Trading date
            - quality_score: Overall quality (0-1)
            - volume_score: Volume component
            - volatility_score: Volatility component
            - trend_score: Trend strength
            - num_reset_points: Count of good reset points
            
        Design notes:
        - Balance different quality metrics
        - Consider market regime
        - Avoid lookahead bias
        """
        ...
    
    @abstractmethod
    def find_reset_points(
        self,
        market_data: pd.DataFrame,
        date: date
    ) -> pd.DataFrame:
        """Find reset points within a day.
        
        Args:
            market_data: Intraday market data
            date: Trading date
            
        Returns:
            DataFrame with columns:
            - timestamp: Reset point time
            - quality_score: Reset quality
            - momentum_type: Type of momentum
            - context: Market context features
            
        Design notes:
        - Look for momentum shifts
        - Consider microstructure
        - Ensure diversity
        """
        ...
