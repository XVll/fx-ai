"""Load all feature implementations to register them"""

def load_all_features():
    """Import all feature modules to register them with the registry"""
    # Static features
    from feature.static import time_features, session_features
    
    # HF features
    from feature.hf import price_features, tape_features, quote_features
    
    # MF features
    from feature.mf import velocity_features, ema_features, candle_features
    
    # LF features
    from feature.lf import range_features, level_features
    
    return True