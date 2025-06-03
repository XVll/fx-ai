"""Feature Attribution Analysis Package

Provides comprehensive gradient-based feature importance analysis using Captum.
"""

from .captum_feature_analyzer import CaptumFeatureAnalyzer, AttributionMethod, AttributionConfig

__all__ = ['CaptumFeatureAnalyzer', 'AttributionMethod', 'AttributionConfig']