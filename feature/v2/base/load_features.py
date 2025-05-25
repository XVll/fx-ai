"""Load all feature implementations to register them"""

def load_all_features():
    """Import all feature modules to register them with the registry"""
    # Static features
    from ..static import time_features, session_features
    
    # HF features
    from ..hf import price_features, tape_features
    
    # MF features
    from ..mf import velocity_features, ema_features, candle_features
    
    # LF features
    from ..lf import range_features, level_features
    
    return True