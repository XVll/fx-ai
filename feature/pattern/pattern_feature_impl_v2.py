"""Individual pattern feature implementations using base class."""

from .pattern_feature_base import PatternFeatureBase


# Register all pattern features with minimal boilerplate
pattern_features = [
    "swing_high_distance",
    "swing_low_distance", 
    "swing_high_price_pct",
    "swing_low_price_pct",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "higher_highs_count",
    "higher_lows_count",
    "lower_highs_count",
    "lower_lows_count",
    "range_compression",
    "consolidation_score",
    "triangle_apex_distance",
    "momentum_alignment",
    "breakout_potential",
    "squeeze_intensity"
]


# Dynamically create and register feature classes
for feature_name in pattern_features:
    # Create a class name from feature name
    class_name = ''.join(word.capitalize() for word in feature_name.split('_')) + 'Feature'
    
    # Create the feature class
    feature_class = type(class_name, (PatternFeatureBase,), {
        '__init__': lambda self, config=None, fname=feature_name: 
            PatternFeatureBase.__init__(self, fname, config),
        '__doc__': f'{feature_name.replace("_", " ").title()} feature for pattern detection'
    })
    
    # Register it
    # Registry removed - feature_class is ready to use directly