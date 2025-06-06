"""Feature Registry - Single source of truth for feature names and ordering.

This module ensures consistency between feature extraction and attribution analysis.
When features are added/removed from the system, this registry is the only place
that needs to be updated.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FeatureInfo:
    """Information about a feature."""
    name: str
    description: str
    category: str  # hf, mf, lf, portfolio
    group: str  # price_action, volume, etc.
    index: int  # Position in feature vector
    active: bool = True  # Whether this feature is currently used


class FeatureRegistry:
    """Central registry for all features in the system.
    
    IMPORTANT: This is the SINGLE SOURCE OF TRUTH for features.
    - SimpleFeatureManager should use this to know which features to create
    - Config schemas should use this for dimensions
    - Captum attribution should use this for feature names
    
    Current dimensions (matching config/schemas.py ModelConfig):
    - HF: 9 features (all active)
    - MF: 43 features (out of 52 total, some inactive) 
    - LF: 19 features (out of 22 total, some inactive)
    - Portfolio: 10 features (all active)
    """
    
    # High-Frequency Features (1-second bars) - ALL 9 are active
    HF_FEATURES = [
        ("price_velocity", "1-second price velocity", "price_action", True),
        ("price_acceleration", "1-second price acceleration", "price_action", True),
        ("tape_imbalance", "Buy/sell tape imbalance", "tape_analysis", True),
        ("tape_aggression_ratio", "Aggressive vs passive orders", "tape_analysis", True),
        ("spread_compression", "Bid-ask spread compression", "microstructure", True),
        ("quote_velocity", "Quote update velocity", "microstructure", True),
        ("quote_imbalance", "Bid/ask size imbalance", "microstructure", True),
        ("volume_velocity", "1-second volume velocity", "volume_dynamics", True),
        ("volume_acceleration", "1-second volume acceleration", "volume_dynamics", True),
    ]
    
    # Medium-Frequency Features - 43 active out of 52 total
    MF_FEATURES = [
        # Candle features (10) - ALL ACTIVE
        ("1m_position_in_current_candle", "Position within current 1m candle", "candle_patterns", True),
        ("5m_position_in_current_candle", "Position within current 5m candle", "candle_patterns", True),
        ("1m_body_size_relative", "1m candle body size", "candle_patterns", True),
        ("5m_body_size_relative", "5m candle body size", "candle_patterns", True),
        ("1m_position_in_previous_candle", "Position in previous 1m candle", "candle_patterns", True),
        ("5m_position_in_previous_candle", "Position in previous 5m candle", "candle_patterns", True),
        ("1m_upper_wick_relative", "1m upper wick size", "candle_patterns", True),
        ("1m_lower_wick_relative", "1m lower wick size", "candle_patterns", True),
        ("5m_upper_wick_relative", "5m upper wick size", "candle_patterns", True),
        ("5m_lower_wick_relative", "5m lower wick size", "candle_patterns", True),
        # EMA features (7) - ALL ACTIVE
        ("distance_to_ema9_1m", "Distance to 9 EMA on 1m", "ema_system", True),
        ("distance_to_ema20_1m", "Distance to 20 EMA on 1m", "ema_system", True),
        ("distance_to_ema9_5m", "Distance to 9 EMA on 5m", "ema_system", True),
        ("distance_to_ema20_5m", "Distance to 20 EMA on 5m", "ema_system", True),
        ("ema_interaction_pattern", "EMA crossover patterns", "ema_system", True),
        ("ema_crossover_dynamics", "EMA crossover momentum", "ema_system", True),
        ("ema_trend_alignment", "Multi-timeframe EMA alignment", "ema_system", True),
        # Swing features (4) - ALL ACTIVE
        ("swing_high_distance_1m", "Distance to 1m swing high", "swing_analysis", True),
        ("swing_low_distance_1m", "Distance to 1m swing low", "swing_analysis", True),
        ("swing_high_distance_5m", "Distance to 5m swing high", "swing_analysis", True),
        ("swing_low_distance_5m", "Distance to 5m swing low", "swing_analysis", True),
        # Velocity/acceleration features (8) - ALL ACTIVE
        ("price_velocity_1m", "1-minute price velocity", "velocity_acceleration", True),
        ("price_velocity_5m", "5-minute price velocity", "velocity_acceleration", True),
        ("volume_velocity_1m", "1-minute volume velocity", "velocity_acceleration", True),
        ("volume_velocity_5m", "5-minute volume velocity", "velocity_acceleration", True),
        ("price_acceleration_1m", "1-minute price acceleration", "velocity_acceleration", True),
        ("price_acceleration_5m", "5-minute price acceleration", "velocity_acceleration", True),
        ("volume_acceleration_1m", "1-minute volume acceleration", "velocity_acceleration", True),
        ("volume_acceleration_5m", "5-minute volume acceleration", "velocity_acceleration", True),
        # VWAP features (6) - ALL ACTIVE
        ("distance_to_vwap", "Distance to VWAP", "vwap_analysis", True),
        ("vwap_slope", "VWAP slope/momentum", "vwap_analysis", True),
        ("price_vwap_divergence", "Price-VWAP divergence", "vwap_analysis", True),
        ("vwap_interaction_dynamics", "VWAP touch/break dynamics", "vwap_analysis", True),
        ("vwap_breakout_quality", "VWAP breakout strength", "vwap_analysis", True),
        ("vwap_mean_reversion_tendency", "Mean reversion to VWAP", "vwap_analysis", True),
        # Volume features (4) - ALL ACTIVE
        ("relative_volume", "Volume vs average", "volume_dynamics", True),
        ("volume_surge", "Volume spike detection", "volume_dynamics", True),
        ("cumulative_volume_delta", "Cumulative buy/sell volume", "volume_dynamics", True),
        ("volume_momentum", "Volume trend strength", "volume_dynamics", True),
        # Professional indicators (4) - ALL ACTIVE
        ("professional_ema_system", "Complex EMA analysis", "professional_indicators", True),
        ("professional_vwap_analysis", "Advanced VWAP metrics", "professional_indicators", True),
        ("professional_momentum_quality", "Momentum quality score", "professional_indicators", True),
        ("professional_volatility_regime", "Volatility classification", "professional_indicators", True),
        # --- TOTAL: 43 ACTIVE features above this line ---
        # Sequence patterns (4) - INACTIVE (would exceed 43 limit)
        ("trend_acceleration", "Multi-bar trend acceleration", "sequence_patterns", False),
        ("volume_pattern_evolution", "Volume pattern development", "sequence_patterns", False),
        ("momentum_quality", "Momentum consistency", "sequence_patterns", False),
        ("pattern_maturation", "Pattern completion score", "sequence_patterns", False),
        # Aggregated signals (3) - INACTIVE
        ("mf_trend_consistency", "Cross-timeframe trend alignment", "aggregated_signals", False),
        ("mf_volume_price_divergence", "Volume-price divergence", "aggregated_signals", False),
        ("mf_momentum_persistence", "Momentum sustainability", "aggregated_signals", False),
        # Adaptive features (2) - INACTIVE
        ("volatility_adjusted_momentum", "Volatility-normalized momentum", "adaptive_features", False),
        ("regime_relative_volume", "Volume relative to regime", "adaptive_features", False),
    ]
    
    # Low-Frequency Features - 19 active out of 22 total
    LF_FEATURES = [
        # Range features (3) - ALL ACTIVE
        ("daily_range_position", "Position in daily range", "range_position", True),
        ("prev_day_range_position", "Position vs yesterday's range", "range_position", True),
        ("price_change_from_prev_close", "% change from prev close", "range_position", True),
        # Level features (4) - ALL ACTIVE
        ("support_distance", "Distance to nearest support", "support_resistance", True),
        ("resistance_distance", "Distance to nearest resistance", "support_resistance", True),
        ("whole_dollar_proximity", "Distance to whole dollar", "support_resistance", True),
        ("half_dollar_proximity", "Distance to half dollar", "support_resistance", True),
        # Time features (3) - ALL ACTIVE
        ("market_session_type", "Pre/regular/post market", "time_context", True),
        ("time_of_day_sin", "Time encoding (sin)", "time_context", True),
        ("time_of_day_cos", "Time encoding (cos)", "time_context", True),
        # Market structure features (5) - ALL ACTIVE
        ("halt_state", "Trading halt status", "market_structure", True),
        ("time_since_halt", "Time since last halt", "market_structure", True),
        ("distance_to_luld_up", "Distance to upper limit", "market_structure", True),
        ("distance_to_luld_down", "Distance to lower limit", "market_structure", True),
        ("luld_band_width", "LULD band width", "market_structure", True),
        # Context features (3) - ALL ACTIVE
        ("session_progress", "Progress through session", "session_context", True),
        ("market_stress", "Market stress indicator", "session_context", True),
        ("session_volume_profile", "Intraday volume profile", "session_context", True),
        # Adaptive features (1) - ACTIVE
        ("adaptive_support_resistance", "Dynamic S/R levels", "adaptive_features", True),
        # --- TOTAL: 19 ACTIVE features above this line ---
        # HF summary features (3) - INACTIVE (would exceed 19 limit)
        ("hf_momentum_summary", "HF momentum aggregation", "hf_summaries", False),
        ("hf_volume_dynamics", "HF volume summary", "hf_summaries", False),
        ("hf_microstructure_quality", "HF market quality", "hf_summaries", False),
    ]
    
    # Portfolio Features - ALL 10 are active
    PORTFOLIO_FEATURES = [
        ("position_size_normalized", "Normalized position size (-1 to 1)", "position_tracking", True),
        ("unrealized_pnl_normalized", "Unrealized P&L (normalized)", "pnl_tracking", True),
        ("time_in_position", "Holding time (normalized)", "position_tracking", True),
        ("cash_ratio", "Available cash ratio", "position_tracking", True),
        ("session_pnl_percentage", "Session P&L percentage", "pnl_tracking", True),
        ("max_favorable_excursion", "Max favorable excursion (MFE)", "trade_quality", True),
        ("max_adverse_excursion", "Max adverse excursion (MAE)", "trade_quality", True),
        ("profit_giveback_ratio", "Profit giveback ratio", "trade_quality", True),
        ("recovery_ratio", "Recovery from drawdown", "trade_quality", True),
        ("trade_quality_score", "Current P&L vs MFE/MAE range", "trade_quality", True),
    ]
    
    @classmethod
    def get_feature_names(cls, category: str, active_only: bool = True) -> List[str]:
        """Get feature names for a category.
        
        Args:
            category: Feature category (hf, mf, lf, portfolio)
            active_only: If True, only return active features
        """
        if category == "hf":
            features = cls.HF_FEATURES
        elif category == "mf":
            features = cls.MF_FEATURES
        elif category == "lf":
            features = cls.LF_FEATURES
        elif category == "portfolio":
            features = cls.PORTFOLIO_FEATURES
        else:
            raise ValueError(f"Unknown category: {category}")
        
        if active_only:
            return [f[0] for f in features if f[3]]
        else:
            return [f[0] for f in features]
    
    @classmethod
    def get_feature_info(cls, category: str, active_only: bool = True) -> List[FeatureInfo]:
        """Get detailed feature information for a category."""
        features = []
        
        if category == "hf":
            feature_list = cls.HF_FEATURES
        elif category == "mf":
            feature_list = cls.MF_FEATURES
        elif category == "lf":
            feature_list = cls.LF_FEATURES
        elif category == "portfolio":
            feature_list = cls.PORTFOLIO_FEATURES
        else:
            raise ValueError(f"Unknown category: {category}")
        
        idx = 0
        for name, desc, group, active in feature_list:
            if not active_only or active:
                features.append(FeatureInfo(
                    name=name,
                    description=desc,
                    category=category,
                    group=group,
                    index=idx,
                    active=active
                ))
                idx += 1
        
        return features
    
    @classmethod
    def get_feature_groups(cls, category: str, active_only: bool = True) -> Dict[str, List[str]]:
        """Get features organized by groups for a category."""
        groups = {}
        
        if category == "hf":
            feature_list = cls.HF_FEATURES
        elif category == "mf":
            feature_list = cls.MF_FEATURES
        elif category == "lf":
            feature_list = cls.LF_FEATURES
        elif category == "portfolio":
            feature_list = cls.PORTFOLIO_FEATURES
        else:
            raise ValueError(f"Unknown category: {category}")
        
        for name, desc, group, active in feature_list:
            if not active_only or active:
                group_key = f"{category}_{group}"
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(name)
        
        return groups
    
    @classmethod
    def get_all_feature_groups(cls, active_only: bool = True) -> Dict[str, List[str]]:
        """Get all features organized by category and group."""
        all_groups = {}
        
        for category in ["hf", "mf", "lf", "portfolio"]:
            groups = cls.get_feature_groups(category, active_only)
            all_groups.update(groups)
        
        return all_groups
    
    @classmethod
    def get_feature_dimensions(cls, active_only: bool = True) -> Dict[str, int]:
        """Get expected feature dimensions for each category."""
        return {
            "hf": len([f for f in cls.HF_FEATURES if not active_only or f[3]]),
            "mf": len([f for f in cls.MF_FEATURES if not active_only or f[3]]), 
            "lf": len([f for f in cls.LF_FEATURES if not active_only or f[3]]),
            "portfolio": len([f for f in cls.PORTFOLIO_FEATURES if not active_only or f[3]]),
        }
    
    @classmethod
    def validate_feature_manager(cls, feature_manager) -> bool:
        """Validate that a feature manager has the expected features."""
        try:
            dimensions = cls.get_feature_dimensions(active_only=True)
            
            for category in ["hf", "mf", "lf"]:
                expected = cls.get_feature_names(category, active_only=True)
                actual = feature_manager.get_enabled_features(category)
                
                # Check count
                if len(actual) != dimensions[category]:
                    print(f"Warning: {category} has {len(actual)} features, expected {dimensions[category]}")
                
                # Check order matches for active features
                for i, expected_name in enumerate(expected):
                    if i < len(actual) and actual[i] != expected_name:
                        print(f"Warning: {category}[{i}] mismatch: expected '{expected_name}', got '{actual[i]}'")
                        return False
            
            return True
        except Exception as e:
            print(f"Error validating feature manager: {e}")
            return False
    
    @classmethod
    def verify_dimensions(cls) -> None:
        """Verify that registry dimensions match expected config dimensions."""
        dims = cls.get_feature_dimensions(active_only=True)
        print(f"Feature Registry Active Dimensions:")
        print(f"  HF: {dims['hf']} features (expected: 9)")
        print(f"  MF: {dims['mf']} features (expected: 43)")
        print(f"  LF: {dims['lf']} features (expected: 19)")
        print(f"  Portfolio: {dims['portfolio']} features (expected: 10)")
        
        assert dims['hf'] == 9, f"HF features mismatch: {dims['hf']} != 9"
        assert dims['mf'] == 43, f"MF features mismatch: {dims['mf']} != 43"
        assert dims['lf'] == 19, f"LF features mismatch: {dims['lf']} != 19"
        assert dims['portfolio'] == 10, f"Portfolio features mismatch: {dims['portfolio']} != 10"
        
        # Also show total features
        total_dims = cls.get_feature_dimensions(active_only=False)
        print(f"\nTotal Features (including inactive):")
        print(f"  HF: {total_dims['hf']} total")
        print(f"  MF: {total_dims['mf']} total ({total_dims['mf'] - dims['mf']} inactive)")
        print(f"  LF: {total_dims['lf']} total ({total_dims['lf'] - dims['lf']} inactive)")
        print(f"  Portfolio: {total_dims['portfolio']} total")
    
    @classmethod
    def get_class_mapping(cls, category: str) -> Dict[str, Tuple[str, str]]:
        """Get mapping of feature names to their class names and modules.
        
        Returns:
            Dict mapping feature_name -> (class_name, module_path)
        """
        if category == "hf":
            return {
                "price_velocity": ("PriceVelocityFeature", "hf.price_features"),
                "price_acceleration": ("PriceAccelerationFeature", "hf.price_features"),
                "tape_imbalance": ("TapeImbalanceFeature", "hf.tape_features"),
                "tape_aggression_ratio": ("TapeAggressionRatioFeature", "hf.tape_features"),
                "spread_compression": ("SpreadCompressionFeature", "hf.quote_features"),
                "quote_velocity": ("QuoteVelocityFeature", "hf.quote_features"),
                "quote_imbalance": ("QuoteImbalanceFeature", "hf.quote_features"),
                "volume_velocity": ("VolumeVelocityFeature", "hf.volume_features"),
                "volume_acceleration": ("VolumeAccelerationFeature", "hf.volume_features"),
            }
        elif category == "mf":
            return {
                # Candle features
                "1m_position_in_current_candle": ("PositionInCurrentCandle1mFeature", "mf.candle_features"),
                "5m_position_in_current_candle": ("PositionInCurrentCandle5mFeature", "mf.candle_features"),
                "1m_body_size_relative": ("BodySizeRelative1mFeature", "mf.candle_features"),
                "5m_body_size_relative": ("BodySizeRelative5mFeature", "mf.candle_features"),
                "1m_position_in_previous_candle": ("PositionInPreviousCandle1mFeature", "mf.candle_analysis_features"),
                "5m_position_in_previous_candle": ("PositionInPreviousCandle5mFeature", "mf.candle_analysis_features"),
                "1m_upper_wick_relative": ("UpperWickRelative1mFeature", "mf.candle_analysis_features"),
                "1m_lower_wick_relative": ("LowerWickRelative1mFeature", "mf.candle_analysis_features"),
                "5m_upper_wick_relative": ("UpperWickRelative5mFeature", "mf.candle_analysis_features"),
                "5m_lower_wick_relative": ("LowerWickRelative5mFeature", "mf.candle_analysis_features"),
                # EMA features
                "distance_to_ema9_1m": ("DistanceToEMA9_1mFeature", "mf.ema_features"),
                "distance_to_ema20_1m": ("DistanceToEMA20_1mFeature", "mf.ema_features"),
                "distance_to_ema9_5m": ("DistanceToEMA9_5mFeature", "mf.ema_features"),
                "distance_to_ema20_5m": ("DistanceToEMA20_5mFeature", "mf.ema_features"),
                "ema_interaction_pattern": ("EMAInteractionPatternFeature", "mf.ema_features_v2"),
                "ema_crossover_dynamics": ("EMACrossoverDynamicsFeature", "mf.ema_features_v2"),
                "ema_trend_alignment": ("EMATrendAlignmentFeature", "mf.ema_features_v2"),
                # Continue for all MF features...
            }
        # Add LF mappings...
        else:
            return {}