"""Feature Attribution Analysis Package

Provides feature importance analysis using simple methods and Captum.
"""

from .simple_attribution import SimpleFeatureAttribution

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
    "SimpleFeatureAttribution",
    "CaptumAttributionAnalyzer",
    "AttributionConfig",
    "MultiBranchTransformerWrapper",
    "CAPTUM_AVAILABLE",
]
