"""Feature Attribution Analysis Package

Provides feature importance analysis using Captum.

The FeatureRegistry is used as the single source of truth for all feature
names and groupings, ensuring consistency between feature extraction and
attribution analysis.
"""

try:
    from .captum_attribution import (
        CaptumAttributionAnalyzer,
        AttributionConfig,
        MultiBranchTransformerWrapper,
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    CaptumAttributionAnalyzer = None
    AttributionConfig = None
    MultiBranchTransformerWrapper = None

__all__ = [
    "CaptumAttributionAnalyzer",
    "AttributionConfig", 
    "MultiBranchTransformerWrapper",
    "CAPTUM_AVAILABLE",
]
