"""
Feature engineering interfaces for market data transformation.

These interfaces enable modular feature creation with
clear contracts for different feature types and frequencies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

from ..types.common import (
    Symbol, FeatureArray, FeatureFrequency,
    Configurable, Resettable
)


class FeatureType(Enum):
    """Types of features for categorization.
    
    Design: Group features by their characteristics
    for better organization and optimization.
    """
    PRICE = "PRICE"              # Price-based (returns, ratios)
    VOLUME = "VOLUME"            # Volume-based (VWAP, accumulation)
    MICROSTRUCTURE = "MICROSTRUCTURE"  # Spread, depth, imbalance
    TECHNICAL = "TECHNICAL"      # Traditional indicators
    STATISTICAL = "STATISTICAL"  # Statistical measures
    PATTERN = "PATTERN"          # Pattern recognition
    SENTIMENT = "SENTIMENT"      # Market sentiment indicators
    CONTEXTUAL = "CONTEXTUAL"    # Time of day, market regime


@runtime_checkable
class IFeature(Protocol):
    """Base interface for individual features.
    
    Design principles:
    - Each feature is self-contained
    - Clear dependencies and requirements
    - Efficient computation
    - Proper handling of edge cases
    """
    
    @property
    def name(self) -> str:
        """Unique feature name.
        
        Returns:
            Feature identifier (e.g., "price_momentum_5m")
            
        Design notes:
        - Use descriptive, hierarchical names
        - Include timeframe in name
        """
        ...
    
    @property
    def feature_type(self) -> FeatureType:
        """Feature category.
        
        Returns:
            Type of feature
        """
        ...
    
    @property
    def frequency(self) -> FeatureFrequency:
        """Update frequency.
        
        Returns:
            How often feature updates
        """
        ...
    
    @property
    def lookback_required(self) -> int:
        """Minimum lookback period needed.
        
        Returns:
            Number of periods required
            
        Design notes:
        - Used for data buffering
        - Includes warmup period
        """
        ...
    
    @property
    def dependencies(self) -> list[str]:
        """Data dependencies.
        
        Returns:
            List of required data types (trades, quotes, etc.)
        """
        ...
    
    def compute(
        self,
        data: pd.DataFrame,
        timestamp: datetime
    ) -> float:
        """Compute feature value.
        
        Args:
            data: Market data DataFrame
            timestamp: Current timestamp
            
        Returns:
            Feature value
            
        Design notes:
        - Handle missing data gracefully (return NaN)
        - Avoid lookahead bias
        - Optimize for repeated calls
        """
        ...
    
    def compute_batch(
        self,
        data: pd.DataFrame,
        timestamps: list[datetime]
    ) -> np.ndarray:
        """Compute feature for multiple timestamps.
        
        Args:
            data: Market data DataFrame
            timestamps: List of timestamps
            
        Returns:
            Array of feature values
            
        Design notes:
        - More efficient than multiple compute() calls
        - Maintain chronological order
        """
        ...


class IFeatureGroup(Configurable, Resettable):
    """Interface for groups of related features.
    
    Design principles:
    - Group features that share computation
    - Enable efficient batch processing
    - Provide feature metadata
    """
    
    @abstractmethod
    def get_features(self) -> list[IFeature]:
        """Get all features in group.
        
        Returns:
            List of feature instances
            
        Design notes:
        - Features should be immutable
        - Order matters for array indexing
        """
        ...
    
    @abstractmethod
    def compute_all(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> dict[str, float]:
        """Compute all features in group.
        
        Args:
            data: Dict of DataFrames by data type
            timestamp: Current timestamp
            
        Returns:
            Dict mapping feature name to value
            
        Design notes:
        - Share intermediate computations
        - Handle failures gracefully
        """
        ...
    
    @abstractmethod
    def get_feature_info(self) -> dict[str, dict[str, Any]]:
        """Get information about features.
        
        Returns:
            Dict mapping feature name to info:
            - description: What it measures
            - range: Expected value range
            - normalization: Suggested method
            - importance: Relative importance
            
        Design notes:
        - Used for documentation
        - Helps with feature selection
        """
        ...


class IFeatureRegistry(Protocol):
    """Central registry for all features.
    
    Design principles:
    - Single source of truth for features
    - Enable feature discovery
    - Support versioning
    - Facilitate A/B testing
    """
    
    def register_feature(
        self,
        feature: IFeature,
        version: str = "1.0"
    ) -> None:
        """Register a feature.
        
        Args:
            feature: Feature instance
            version: Feature version
            
        Design notes:
        - Check for name conflicts
        - Validate feature implementation
        """
        ...
    
    def register_group(
        self,
        group: IFeatureGroup,
        version: str = "1.0"
    ) -> None:
        """Register a feature group.
        
        Args:
            group: Feature group instance
            version: Group version
        """
        ...
    
    def get_feature(
        self,
        name: str,
        version: Optional[str] = None
    ) -> IFeature:
        """Get feature by name.
        
        Args:
            name: Feature name
            version: Specific version (latest if None)
            
        Returns:
            Feature instance
        """
        ...
    
    def get_features_by_frequency(
        self,
        frequency: FeatureFrequency
    ) -> list[IFeature]:
        """Get all features for frequency.
        
        Args:
            frequency: Feature frequency
            
        Returns:
            List of features
        """
        ...
    
    def get_features_by_type(
        self,
        feature_type: FeatureType
    ) -> list[IFeature]:
        """Get all features of given type.
        
        Args:
            feature_type: Type of features
            
        Returns:
            List of features
        """
        ...


class IFeatureStore(Protocol):
    """Interface for feature storage and retrieval.
    
    Design principles:
    - Enable feature precomputation
    - Support time-travel queries
    - Handle versioning
    - Optimize for read performance
    """
    
    def store_features(
        self,
        symbol: Symbol,
        timestamp: datetime,
        features: dict[str, float],
        version: str
    ) -> None:
        """Store computed features.
        
        Args:
            symbol: Trading symbol
            timestamp: Feature timestamp
            features: Dict of feature values
            version: Feature set version
            
        Design notes:
        - Batch writes for efficiency
        - Handle duplicates
        """
        ...
    
    def load_features(
        self,
        symbol: Symbol,
        start: datetime,
        end: datetime,
        feature_names: Optional[list[str]] = None,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """Load precomputed features.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp
            end: End timestamp
            feature_names: Specific features (all if None)
            version: Feature version (latest if None)
            
        Returns:
            DataFrame with features as columns
            
        Design notes:
        - Support partial loads
        - Handle missing data
        """
        ...
    
    def get_coverage(
        self,
        symbol: Symbol,
        version: str
    ) -> tuple[datetime, datetime]:
        """Get time range of stored features.
        
        Args:
            symbol: Trading symbol
            version: Feature version
            
        Returns:
            Tuple of (start, end) timestamps
        """
        ...


class IFeaturePipeline(Configurable):
    """Interface for feature transformation pipeline.
    
    Design principles:
    - Chain feature transformations
    - Support online and batch modes
    - Enable feature engineering workflows
    """
    
    @abstractmethod
    def add_stage(
        self,
        name: str,
        transformer: Any
    ) -> None:
        """Add transformation stage.
        
        Args:
            name: Stage name
            transformer: Transformation function/object
            
        Design notes:
        - Stages execute in order
        - Each stage can add/remove features
        """
        ...
    
    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame
    ) -> None:
        """Fit pipeline on training data.
        
        Args:
            data: Training data
            
        Design notes:
        - Learn normalization parameters
        - Compute feature statistics
        """
        ...
    
    @abstractmethod
    def transform(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform data through pipeline.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
            
        Design notes:
        - Apply learned transformations
        - Handle new data appropriately
        """
        ...
    
    @abstractmethod
    def transform_online(
        self,
        features: dict[str, float]
    ) -> dict[str, float]:
        """Transform single observation.
        
        Args:
            features: Input features
            
        Returns:
            Transformed features
            
        Design notes:
        - For real-time inference
        - Maintain state if needed
        """
        ...
