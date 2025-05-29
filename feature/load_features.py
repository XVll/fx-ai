"""Load all feature implementations to register them"""

def load_all_features():
    """Import all feature modules to register them with the registry"""
    # Static features
    from feature.static import time_features, session_features
    
    # HF features
    from feature.hf import price_features, tape_features, quote_features, volume_features
    
    # MF features
    from feature.mf import velocity_features, ema_features, candle_features
    from feature.mf import acceleration_features, candle_analysis_features, swing_features
    
    # LF features
    from feature.lf import range_features, level_features
    
    # Portfolio features
    from feature.portfolio import portfolio_features
    
    # Pattern features
    from feature.pattern import pattern_features, pattern_feature_impl_v2
    
    # Market structure features
    from feature.market_structure import halt_features, luld_features
    
    # Volume analysis features
    from feature.volume_analysis import vwap_features, relative_volume_features
    
    # Sequence-aware features
    from feature.sequence_aware import sequence_features
    
    # Adaptive features
    from feature.adaptive import adaptive_features
    
    # Context features (better than "static")
    from feature.context import context_features
    
    # Aggregated features that efficiently use sequence windows
    from feature.aggregated import aggregated_features
    
    return True