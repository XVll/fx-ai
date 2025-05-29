"""Feature manager for batch calculation and management"""
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from .feature_base import BaseFeature, FeatureConfig
from .feature_registry import feature_registry
from .load_features import load_all_features
from .contexts import MarketContext

# Load all features on import
load_all_features()


class FeatureManager:
    """Manages feature calculation and aggregation"""
    
    def __init__(self, symbol: str, config: Any, logger: Optional[logging.Logger] = None):
        self.symbol = symbol
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.registry = feature_registry
        self._features: Dict[str, Dict[str, BaseFeature]] = {}
        self._feature_order: Dict[str, List[str]] = {}
        
        # Feature dimensions from config
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.static_feat_dim = config.static_feat_dim
        self.portfolio_seq_len = config.portfolio_seq_len
        self.portfolio_feat_dim = config.portfolio_feat_dim
        
        # Load configured features
        self._load_features()
    
    def _load_features(self):
        """Load features based on configuration"""
        # Define all available features by category
        available_features = {
            'hf': [
                'spread_compression', 'tape_imbalance', 'tape_aggression_ratio',
                'quote_imbalance',
                # Aggregated features that efficiently use sequence windows
                'hf_momentum_summary', 'hf_volume_dynamics', 'hf_microstructure_quality'
            ],
            'mf': [
                '1m_position_in_current_candle', '5m_position_in_current_candle',
                '1m_position_in_previous_candle', '5m_position_in_previous_candle',
                '1m_body_size_relative', '5m_body_size_relative',
                '1m_upper_wick_relative', '5m_upper_wick_relative',
                '1m_lower_wick_relative', '5m_lower_wick_relative',
                '1m_ema9_distance', '1m_ema20_distance',
                '5m_ema9_distance', '5m_ema20_distance',
                # Pattern detection features (removed redundant swing features)
                'swing_high_distance', 'swing_low_distance',
                'swing_high_price_pct', 'swing_low_price_pct',
                'bars_since_swing_high', 'bars_since_swing_low',
                'higher_highs_count', 'higher_lows_count',
                'lower_highs_count', 'lower_lows_count',
                'range_compression', 'consolidation_score',
                'triangle_apex_distance', 'momentum_alignment',
                'breakout_potential', 'squeeze_intensity',
                # Volume analysis features
                'distance_to_vwap', 'vwap_slope', 'price_vwap_divergence',
                'relative_volume', 'volume_surge', 'cumulative_volume_delta',
                'volume_momentum',
                # Sequence-aware features (new)
                'trend_acceleration', 'volume_pattern_evolution',
                'momentum_quality', 'pattern_maturation',
                # Adaptive features (new)
                'volatility_adjusted_momentum', 'regime_relative_volume',
                # Aggregated features that efficiently use sequence windows
                'mf_trend_consistency', 'mf_volume_price_divergence', 'mf_momentum_persistence'
            ],
            'lf': [
                'support_distance', 'resistance_distance',
                'whole_dollar_proximity', 'half_dollar_proximity',
                'daily_range_position', 'position_in_prev_day_range',
                'price_change_from_prev_close',
                # Market structure features
                'distance_to_luld_up', 'distance_to_luld_down', 'luld_band_width',
                # Adaptive features (new)
                'adaptive_support_resistance'
            ],
            'static': [
                'market_session_type', 'time_of_day_sin', 'time_of_day_cos',
                # Market structure features
                'is_halted', 'time_since_halt',
                # Context features (new - more dynamic than traditional "static")
                'session_progress', 'market_stress_level', 'session_volume_profile'
            ],
            'portfolio': [
                'portfolio_position_size', 'portfolio_average_price', 'portfolio_unrealized_pnl',
                'portfolio_time_in_position', 'portfolio_max_adverse_excursion'
            ]
        }
        
        # Load all available features for each category
        for category, feature_names in available_features.items():
            self._features[category] = {}
            self._feature_order[category] = feature_names
            
            for feature_name in feature_names:
                if self.registry.has_feature(feature_name):
                    feature_class = self.registry.get_feature_class(feature_name)
                    config = FeatureConfig(
                        name=feature_name,
                        enabled=True,
                        normalize=True,  # Always normalize for now
                        params={}
                    )
                    self._features[category][feature_name] = feature_class(config)
                    self.logger.debug(f"Loaded feature: {feature_name} in category {category}")
                else:
                    self.logger.warning(f"Feature not found in registry: {feature_name}")
    
    def get_enabled_features(self, category: str) -> List[str]:
        """Get list of enabled features in a category"""
        if category not in self._features:
            return []
        
        return [name for name, feature in self._features[category].items() 
                if feature.enabled]
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        for category_features in self._features.values():
            if feature_name in category_features:
                return category_features[feature_name].enabled
        return False
    
    def enable_feature(self, feature_name: str):
        """Enable a feature"""
        for category_features in self._features.values():
            if feature_name in category_features:
                category_features[feature_name].enabled = True
                break
    
    def disable_feature(self, feature_name: str):
        """Disable a feature"""
        for category_features in self._features.values():
            if feature_name in category_features:
                category_features[feature_name].enabled = False
                break
    
    def calculate_features(self, market_data: Dict[str, Any], 
                         category: str) -> Dict[str, float]:
        """Calculate all features in a category"""
        if category not in self._features:
            self.logger.error(f"Category {category} not loaded")
            return {}
        
        results = {}
        
        # Calculate all enabled features in the category
        for name, feature in self._features[category].items():
            if not feature.enabled:
                continue
            
            try:
                value = feature.calculate(market_data)
                # Ensure value is not None, NaN, or infinite
                if value is None or np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Feature {name} returned invalid value ({value}), using 0.0")
                    results[name] = 0.0
                else:
                    results[name] = float(value)  # Ensure it's a float
            except Exception as e:
                self.logger.error(f"Error calculating feature {name}: {e}")
                # Use feature's default value if calculation fails
                results[name] = 0.0
        
        return results
    
    def vectorize_features(self, features: Dict[str, float], 
                          category: str) -> np.ndarray:
        """Convert features to numpy array in consistent order"""
        if category not in self._feature_order:
            self.logger.error(f"No feature order defined for category {category}")
            return np.array([], dtype=np.float32)
        
        # Get features in specified order
        ordered_features = []
        for feature_name in self._feature_order[category]:
            if feature_name in features:
                value = features[feature_name]
                # Double-check for invalid values during vectorization
                if value is None or np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Feature {feature_name} has invalid value ({value}) during vectorization, using 0.0")
                    ordered_features.append(0.0)
                else:
                    ordered_features.append(float(value))
            else:
                self.logger.warning(f"Feature {feature_name} not found in results for category {category}, using 0.0. Available features: {list(features.keys())}")
                ordered_features.append(0.0)
        
        # Ensure we have a valid numpy array with no NaN/inf values
        vector = np.array(ordered_features, dtype=np.float32)
        
        # Final safety check - replace any remaining NaN/inf with 0
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return vector
    
    def get_data_requirements(self) -> Dict[str, Any]:
        """Get aggregated data requirements from all features"""
        # Use get_aggregated_requirements which has the right format
        aggregated = self.get_aggregated_requirements()
        
        # Convert to expected format
        requirements = {}
        for data_type, info in aggregated.items():
            if data_type != 'unknown' and data_type != 'current':
                requirements[data_type] = {
                    'lookback': info['lookback'],
                    'fields': list(info['fields'])
                }
        
        return requirements
    
    def get_aggregated_requirements(self) -> Dict[str, Any]:
        """Get detailed requirements by data type"""
        detailed_reqs = {}
        
        for category_features in self._features.values():
            for feature in category_features.values():
                if not feature.enabled:
                    continue
                
                feat_req = feature.get_requirements()
                data_type = feat_req.get('data_type', 'unknown')
                
                if data_type not in detailed_reqs:
                    detailed_reqs[data_type] = {
                        'lookback': 0,
                        'fields': set()
                    }
                
                # Update lookback for this data type
                if 'lookback' in feat_req:
                    detailed_reqs[data_type]['lookback'] = max(
                        detailed_reqs[data_type]['lookback'],
                        feat_req['lookback']
                    )
                
                # Update fields
                if 'fields' in feat_req:
                    detailed_reqs[data_type]['fields'].update(feat_req['fields'])
        
        return detailed_reqs
    
    def reset(self):
        """Reset feature manager state"""
        # Reset any stateful features if needed
        pass
    
    def extract_features(self, context: MarketContext) -> Optional[Dict[str, np.ndarray]]:
        """Extract features from market context for all categories"""
        try:
            # Extract HF features
            hf_features = self._extract_hf_features(context)
            if hf_features is None:
                return None
                
            # Extract MF features  
            mf_features = self._extract_mf_features(context)
            if mf_features is None:
                return None
                
            # Extract LF features
            lf_features = self._extract_lf_features(context)
            if lf_features is None:
                return None
                
            # Extract static features
            static_features = self._extract_static_features(context)
            if static_features is None:
                return None
                
            # Extract portfolio features
            portfolio_features = self._extract_portfolio_features(context)
            if portfolio_features is None:
                return None
                
            return {
                'hf': hf_features,
                'mf': mf_features,
                'lf': lf_features,
                'static': static_features,
                'portfolio': portfolio_features
            }
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _extract_hf_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract high-frequency features"""
        try:
            hf_window = context.hf_window
            if len(hf_window) < self.hf_seq_len:
                # Pad with zeros if not enough data
                return np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)
            
            # Take last N entries
            hf_window = hf_window[-self.hf_seq_len:]
            
            # Extract features for each timestamp
            feature_matrix = []
            for i, entry in enumerate(hf_window):
                # Handle None entries
                if entry is None:
                    entry = {
                        'timestamp': None,
                        'trades': [],
                        'quotes': [],
                        '1s_bar': None
                    }
                
                # Prepare market data for this timestamp
                market_data = {
                    'timestamp': entry.get('timestamp'),
                    'current_price': entry.get('1s_bar', {}).get('close', 100.0) if entry.get('1s_bar') else 100.0,
                    'hf_data_window': hf_window[max(0, i-10):i+1],  # Include history for velocity calculations
                    'quotes': entry.get('quotes', []),
                    'trades': entry.get('trades', []),
                    '1s_bar': entry.get('1s_bar', {})
                }
                
                # Calculate all HF features
                features = self.calculate_features(market_data, 'hf')
                vector = self.vectorize_features(features, 'hf')
                
                # Ensure vector has correct dimensions and no invalid values
                if len(vector) < self.hf_feat_dim:
                    # Pad with zeros if too few features
                    padded = np.zeros(self.hf_feat_dim, dtype=np.float32)
                    padded[:len(vector)] = vector
                    vector = padded
                elif len(vector) > self.hf_feat_dim:
                    # Truncate if too many features
                    vector = vector[:self.hf_feat_dim]
                    self.logger.warning(f"HF vector too long ({len(vector)}), truncating to {self.hf_feat_dim}")
                
                # Final safety check for this timestamp
                vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
                feature_matrix.append(vector)
            
            return np.array(feature_matrix, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"HF feature extraction failed: {e}")
            return np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)
    
    def _extract_mf_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract medium-frequency features"""
        try:
            mf_window = context.mf_1m_window
            if len(mf_window) < self.mf_seq_len:
                # Pad with zeros if not enough data
                return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)
            
            # Take last N entries
            mf_window = mf_window[-self.mf_seq_len:]
            
            # Also get 5m window for features that need it
            bars_5m = context.lf_5m_window
            
            # Extract features for each timestamp
            feature_matrix = []
            for i, bar in enumerate(mf_window):
                # Prepare market data for this timestamp
                market_data = {
                    'timestamp': bar.get('timestamp'),
                    'current_price': bar.get('close', 100.0),
                    '1m_bars_window': mf_window[max(0, i-20):i+1],  # Include history for EMA calculations
                    '5m_bars_window': bars_5m,
                    '1m_bar': bar
                }
                
                # Calculate all MF features
                features = self.calculate_features(market_data, 'mf')
                vector = self.vectorize_features(features, 'mf')
                
                # Ensure vector has correct dimensions and no invalid values
                if len(vector) < self.mf_feat_dim:
                    # Pad with zeros if too few features
                    padded = np.zeros(self.mf_feat_dim, dtype=np.float32)
                    padded[:len(vector)] = vector
                    vector = padded
                elif len(vector) > self.mf_feat_dim:
                    # Truncate if too many features
                    original_len = len(vector)
                    vector = vector[:self.mf_feat_dim]
                    self.logger.warning(f"MF vector too long ({original_len}), truncating to {self.mf_feat_dim}")
                
                # Final safety check for this timestamp
                vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
                feature_matrix.append(vector)
            
            return np.array(feature_matrix, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"MF feature extraction failed: {e}")
            return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)
    
    def _extract_lf_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract low-frequency features"""
        try:
            # For LF features, we use daily data but repeat current values for the sequence
            # This is because LF features are about current position relative to historical levels
            
            # Extract features once with current market state
            market_data = {
                'timestamp': context.timestamp,
                'current_price': context.current_price,
                'intraday_high': context.session_high,
                'intraday_low': context.session_low,
                'previous_day_data': {
                    'close': context.prev_day_close,
                    'high': context.prev_day_high,
                    'low': context.prev_day_low
                },
                'daily_bars_window': []  # Could be populated if we had daily bar history
            }
            
            # Calculate all LF features
            features = self.calculate_features(market_data, 'lf')
            vector = self.vectorize_features(features, 'lf')
            
            # Ensure vector has correct dimensions and no invalid values
            if len(vector) < self.lf_feat_dim:
                # Pad with zeros if too few features
                padded = np.zeros(self.lf_feat_dim, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
            elif len(vector) > self.lf_feat_dim:
                # Truncate if too many features
                vector = vector[:self.lf_feat_dim]
                self.logger.warning(f"LF vector too long ({len(vector)}), truncating to {self.lf_feat_dim}")
            
            # Final safety check
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Repeat the same features for all timesteps in the sequence
            # This is appropriate for LF features which change slowly
            feature_matrix = np.tile(vector, (self.lf_seq_len, 1))
            
            return feature_matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"LF feature extraction failed: {e}")
            return np.zeros((self.lf_seq_len, self.lf_feat_dim), dtype=np.float32)
    
    def _extract_static_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract static features"""
        try:
            # Prepare market data for static features
            market_data = {
                'timestamp': context.timestamp,
                'current_price': context.current_price,
                'market_session': context.session,
                'market_cap': context.market_cap
            }
            
            # Calculate all static features
            features = self.calculate_features(market_data, 'static')
            vector = self.vectorize_features(features, 'static')
            
            # Ensure vector has correct dimensions and no invalid values
            if len(vector) < self.static_feat_dim:
                # Pad with zeros if too few features
                padded = np.zeros(self.static_feat_dim, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
            elif len(vector) > self.static_feat_dim:
                # Truncate if too many features
                vector = vector[:self.static_feat_dim]
                self.logger.warning(f"Static vector too long ({len(vector)}), truncating to {self.static_feat_dim}")
            
            # Final safety check
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Return as 2D array with shape (1, static_feat_dim)
            return vector.reshape(1, -1).astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Static feature extraction failed: {e}")
            return np.zeros((1, self.static_feat_dim), dtype=np.float32)
    
    def _extract_portfolio_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract portfolio features"""
        try:
            # If no portfolio state, return zeros
            if not context.portfolio_state:
                return np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)
            
            # Prepare market data for portfolio features
            market_data = {
                'timestamp': context.timestamp,
                'current_price': context.current_price,
                'portfolio_state': context.portfolio_state
            }
            
            # Calculate all portfolio features
            features = self.calculate_features(market_data, 'portfolio')
            vector = self.vectorize_features(features, 'portfolio')
            
            # Ensure vector has correct dimensions and no invalid values
            if len(vector) < self.portfolio_feat_dim:
                # Pad with zeros if too few features
                padded = np.zeros(self.portfolio_feat_dim, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
            elif len(vector) > self.portfolio_feat_dim:
                # Truncate if too many features
                vector = vector[:self.portfolio_feat_dim]
                self.logger.warning(f"Portfolio vector too long ({len(vector)}), truncating to {self.portfolio_feat_dim}")
            
            # Final safety check
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Repeat the same features for all timesteps in the sequence
            feature_matrix = np.tile(vector, (self.portfolio_seq_len, 1))
            
            return feature_matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Portfolio feature extraction failed: {e}")
            return np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)
    
