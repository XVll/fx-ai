"""
Attribution interfaces for model interpretability.

Defines contracts for feature attribution and model explanation
systems including Captum integration.
"""

from typing import Protocol, Optional, Any, Union, runtime_checkable
from datetime import datetime
import numpy as np
import pandas as pd
from abc import abstractmethod
from enum import Enum

from ..types import ObservationArray, ActionArray, Metrics


class AttributionMethod(Enum):
    """Available attribution methods."""
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEP_LIFT = "deep_lift"
    DEEP_LIFT_SHAP = "deep_lift_shap"
    INPUT_X_GRADIENT = "input_x_gradient"
    SALIENCY = "saliency"
    FEATURE_ABLATION = "feature_ablation"
    OCCLUSION = "occlusion"
    SHAPLEY_VALUE = "shapley_value"
    LIME = "lime"


class AttributionTarget(Enum):
    """Attribution target types."""
    ACTION_PROBABILITIES = "action_probabilities"
    VALUE_FUNCTION = "value_function"
    SPECIFIC_ACTION = "specific_action"
    ADVANTAGE = "advantage"
    Q_VALUES = "q_values"


@runtime_checkable
class IAttributionResult(Protocol):
    """Result of attribution analysis.
    
    Implementation requirements:
    - Store attribution scores
    - Provide visualization methods
    - Support aggregation
    - Handle multi-dimensional features
    """
    
    @property
    @abstractmethod
    def attributions(self) -> dict[str, np.ndarray]:
        """Raw attribution scores by feature group."""
        ...
    
    @property
    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """Feature importance rankings."""
        ...
    
    @property
    @abstractmethod
    def temporal_importance(self) -> dict[str, np.ndarray]:
        """Importance over time steps."""
        ...
    
    @abstractmethod
    def get_top_features(
        self,
        n: int = 10,
        group: Optional[str] = None
    ) -> list[tuple[str, float]]:
        """Get top N important features.
        
        Args:
            n: Number of features
            group: Feature group filter
            
        Returns:
            List of (feature_name, importance) tuples
        """
        ...
    
    @abstractmethod
    def visualize(
        self,
        method: str = "heatmap",
        save_path: Optional[str] = None
    ) -> Any:
        """Visualize attributions.
        
        Args:
            method: Visualization method
            save_path: Optional save path
            
        Returns:
            Visualization object
        """
        ...
    
    @abstractmethod
    def aggregate_by_group(self) -> dict[str, float]:
        """Aggregate attributions by feature group.
        
        Returns:
            Group-level importance scores
        """
        ...


@runtime_checkable
class IBaselineProvider(Protocol):
    """Provides baseline inputs for attribution.
    
    Implementation requirements:
    - Generate appropriate baselines
    - Support multiple baseline strategies
    - Handle different feature types
    - Validate baseline validity
    """
    
    @abstractmethod
    def get_baseline(
        self,
        observation: ObservationArray,
        method: str = "zero"
    ) -> ObservationArray:
        """Get baseline observation.
        
        Args:
            observation: Original observation
            method: Baseline method (zero, mean, random, etc.)
            
        Returns:
            Baseline observation
        """
        ...
    
    @abstractmethod
    def get_multiple_baselines(
        self,
        observation: ObservationArray,
        n_baselines: int,
        method: str = "random"
    ) -> list[ObservationArray]:
        """Get multiple baselines for averaging.
        
        Args:
            observation: Original observation
            n_baselines: Number of baselines
            method: Baseline generation method
            
        Returns:
            List of baseline observations
        """
        ...
    
    @abstractmethod
    def validate_baseline(
        self,
        baseline: ObservationArray,
        original: ObservationArray
    ) -> bool:
        """Validate baseline is appropriate.
        
        Args:
            baseline: Baseline observation
            original: Original observation
            
        Returns:
            True if valid
        """
        ...


@runtime_checkable
class IFeatureAttributor(Protocol):
    """Computes feature attributions for model decisions.
    
    Implementation requirements:
    - Support multiple attribution methods
    - Handle sequential features
    - Provide action-specific attributions
    - Cache computations efficiently
    """
    
    @abstractmethod
    def attribute(
        self,
        observation: ObservationArray,
        action: Optional[ActionArray] = None,
        method: AttributionMethod = AttributionMethod.INTEGRATED_GRADIENTS,
        target: AttributionTarget = AttributionTarget.ACTION_PROBABILITIES,
        baseline: Optional[ObservationArray] = None,
        **kwargs
    ) -> IAttributionResult:
        """Compute feature attributions.
        
        Args:
            observation: Input observation
            action: Specific action to explain
            method: Attribution method
            target: What to explain
            baseline: Baseline for comparison
            **kwargs: Method-specific parameters
            
        Returns:
            Attribution result
        """
        ...
    
    @abstractmethod
    def batch_attribute(
        self,
        observations: list[ObservationArray],
        actions: Optional[list[ActionArray]] = None,
        method: AttributionMethod = AttributionMethod.INTEGRATED_GRADIENTS,
        target: AttributionTarget = AttributionTarget.ACTION_PROBABILITIES,
        **kwargs
    ) -> list[IAttributionResult]:
        """Compute attributions for batch.
        
        Args:
            observations: Batch of observations
            actions: Corresponding actions
            method: Attribution method
            target: What to explain
            **kwargs: Method-specific parameters
            
        Returns:
            List of attribution results
        """
        ...
    
    @abstractmethod
    def get_supported_methods(self) -> list[AttributionMethod]:
        """Get supported attribution methods.
        
        Returns:
            List of supported methods
        """
        ...
    
    @abstractmethod
    def validate_inputs(
        self,
        observation: ObservationArray,
        method: AttributionMethod
    ) -> bool:
        """Validate inputs for attribution.
        
        Args:
            observation: Input observation
            method: Attribution method
            
        Returns:
            True if valid
        """
        ...


@runtime_checkable
class IAttributionAnalyzer(Protocol):
    """High-level attribution analysis.
    
    Implementation requirements:
    - Analyze attribution patterns
    - Identify feature interactions
    - Track attribution changes over time
    - Provide actionable insights
    """
    
    @abstractmethod
    def analyze_episode(
        self,
        observations: list[ObservationArray],
        actions: list[ActionArray],
        rewards: list[float],
        attributor: IFeatureAttributor
    ) -> dict[str, Any]:
        """Analyze attributions for entire episode.
        
        Args:
            observations: Episode observations
            actions: Episode actions
            rewards: Episode rewards
            attributor: Feature attributor
            
        Returns:
            Analysis results
        """
        ...
    
    @abstractmethod
    def find_critical_features(
        self,
        attributions: list[IAttributionResult],
        threshold: float = 0.8
    ) -> list[str]:
        """Find consistently important features.
        
        Args:
            attributions: Attribution results
            threshold: Importance threshold
            
        Returns:
            Critical feature names
        """
        ...
    
    @abstractmethod
    def analyze_temporal_patterns(
        self,
        attributions: list[IAttributionResult],
        window_size: int = 10
    ) -> dict[str, np.ndarray]:
        """Analyze temporal attribution patterns.
        
        Args:
            attributions: Time-ordered attributions
            window_size: Analysis window
            
        Returns:
            Temporal patterns by feature
        """
        ...
    
    @abstractmethod
    def detect_feature_interactions(
        self,
        attributions: list[IAttributionResult],
        min_correlation: float = 0.5
    ) -> list[tuple[str, str, float]]:
        """Detect feature interactions.
        
        Args:
            attributions: Attribution results
            min_correlation: Minimum correlation
            
        Returns:
            List of (feature1, feature2, correlation)
        """
        ...
    
    @abstractmethod
    def generate_report(
        self,
        analysis_results: dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate analysis report.
        
        Args:
            analysis_results: Analysis outputs
            save_path: Optional save location
            
        Returns:
            Report content
        """
        ...


@runtime_checkable
class ICaptumIntegration(Protocol):
    """Integration with Captum library.
    
    Implementation requirements:
    - Wrap Captum attribution methods
    - Handle PyTorch model requirements
    - Support custom attribution targets
    - Provide unified interface
    """
    
    @abstractmethod
    def setup_model_wrapper(
        self,
        agent: Any,
        target_layer: Optional[str] = None
    ) -> Any:
        """Setup model wrapper for Captum.
        
        Args:
            agent: RL agent
            target_layer: Target layer for attribution
            
        Returns:
            Captum-compatible model
        """
        ...
    
    @abstractmethod
    def create_attribution_method(
        self,
        method: AttributionMethod,
        model: Any,
        **kwargs
    ) -> Any:
        """Create Captum attribution method.
        
        Args:
            method: Attribution method type
            model: Wrapped model
            **kwargs: Method-specific args
            
        Returns:
            Captum attribution object
        """
        ...
    
    @abstractmethod
    def convert_observation(
        self,
        observation: ObservationArray
    ) -> Any:
        """Convert observation to Captum format.
        
        Args:
            observation: RL observation
            
        Returns:
            Captum-compatible input
        """
        ...
    
    @abstractmethod
    def convert_attribution(
        self,
        captum_output: Any,
        original_shape: dict[str, tuple]
    ) -> dict[str, np.ndarray]:
        """Convert Captum output to standard format.
        
        Args:
            captum_output: Captum attribution output
            original_shape: Original feature shapes
            
        Returns:
            Standard attribution format
        """
        ...


@runtime_checkable
class IAttributionCache(Protocol):
    """Caches attribution computations.
    
    Implementation requirements:
    - Cache attribution results
    - Support different cache strategies
    - Handle memory constraints
    - Provide cache statistics
    """
    
    @abstractmethod
    def get(
        self,
        key: str
    ) -> Optional[IAttributionResult]:
        """Get cached attribution.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None
        """
        ...
    
    @abstractmethod
    def put(
        self,
        key: str,
        result: IAttributionResult
    ) -> None:
        """Cache attribution result.
        
        Args:
            key: Cache key
            result: Attribution result
        """
        ...
    
    @abstractmethod
    def clear(
        self,
        pattern: Optional[str] = None
    ) -> int:
        """Clear cache entries.
        
        Args:
            pattern: Key pattern to clear
            
        Returns:
            Number of entries cleared
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        ...


@runtime_checkable
class IAttributionVisualizer(Protocol):
    """Visualizes attribution results.
    
    Implementation requirements:
    - Create various visualization types
    - Support interactive visualizations
    - Handle high-dimensional data
    - Export to different formats
    """
    
    @abstractmethod
    def plot_feature_importance(
        self,
        result: IAttributionResult,
        top_k: int = 20,
        group_by: Optional[str] = None
    ) -> Any:
        """Plot feature importance.
        
        Args:
            result: Attribution result
            top_k: Number of features to show
            group_by: Grouping strategy
            
        Returns:
            Plot object
        """
        ...
    
    @abstractmethod
    def plot_temporal_heatmap(
        self,
        results: list[IAttributionResult],
        features: Optional[list[str]] = None
    ) -> Any:
        """Plot temporal attribution heatmap.
        
        Args:
            results: Time-ordered results
            features: Features to include
            
        Returns:
            Heatmap object
        """
        ...
    
    @abstractmethod
    def create_interactive_dashboard(
        self,
        results: list[IAttributionResult],
        metadata: dict[str, Any]
    ) -> Any:
        """Create interactive dashboard.
        
        Args:
            results: Attribution results
            metadata: Additional metadata
            
        Returns:
            Dashboard object
        """
        ...
    
    @abstractmethod
    def export_visualization(
        self,
        visualization: Any,
        format: str,
        path: str
    ) -> None:
        """Export visualization.
        
        Args:
            visualization: Visualization object
            format: Export format
            path: Export path
        """
        ...