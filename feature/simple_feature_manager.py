"""Simplified feature manager without complex registration system"""
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from .feature_base import BaseFeature, FeatureConfig
from .contexts import MarketContext


class SimpleFeatureManager:
    """Simple feature manager that directly imports and uses feature classes"""
    
    def __init__(self, symbol: str, config: Any, logger: Optional[logging.Logger] = None):
        self.symbol = symbol
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Feature dimensions from config
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.portfolio_seq_len = config.portfolio_seq_len
        self.portfolio_feat_dim = config.portfolio_feat_dim
        
        # Initialize feature collections
        self._feature_collections = self._initialize_features()
    
    def _initialize_features(self) -> Dict[str, List[BaseFeature]]:
        """Initialize all feature collections directly"""
        return {
            'hf': self._create_hf_features(),
            'mf': self._create_mf_features(), 
            'lf': self._create_lf_features(),
            'portfolio': self._create_portfolio_features()
        }
    
    def _create_hf_features(self) -> List[BaseFeature]:
        """Create high-frequency features"""
        features = []
        
        try:
            # Import actual HF feature classes
            from .hf.price_features import PriceVelocityFeature, PriceAccelerationFeature
            from .hf.tape_features import TapeImbalanceFeature, TapeAggressionRatioFeature
            from .hf.quote_features import SpreadCompressionFeature, QuoteImbalanceFeature
            from .hf.volume_features import VolumeVelocityFeature
            
            hf_feature_classes = [
                ('price_velocity', PriceVelocityFeature),
                ('price_acceleration', PriceAccelerationFeature),
                ('tape_imbalance', TapeImbalanceFeature),
                ('tape_aggression_ratio', TapeAggressionRatioFeature),
                ('spread_compression', SpreadCompressionFeature),
                ('quote_imbalance', QuoteImbalanceFeature),
                ('volume_velocity', VolumeVelocityFeature)
            ]
            
            for name, feature_class in hf_feature_classes:
                try:
                    config = FeatureConfig(name=name, enabled=True, normalize=True)
                    features.append(feature_class(config))
                    self.logger.debug(f"Created HF feature: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to create HF feature {name}: {e}")
                    
        except ImportError as e:
            self.logger.warning(f"Failed to import HF features: {e}")
        
        return features
    
    def _create_mf_features(self) -> List[BaseFeature]:
        """Create medium-frequency features"""
        features = []
        
        try:
            # Import actual MF feature classes
            from .mf.candle_features import PositionInCurrentCandle1mFeature, BodySizeRelative1mFeature
            from .mf.ema_features import DistanceToEMA9_1mFeature, DistanceToEMA20_1mFeature
            from .mf.swing_features import SwingHighDistance1mFeature, SwingLowDistance1mFeature
            from .mf.velocity_features import PriceVelocity1mFeature, VolumeVelocity1mFeature
            from .volume_analysis.vwap_features import DistanceToVWAPFeature
            from .volume_analysis.relative_volume_features import RelativeVolumeFeature
            
            mf_feature_classes = [
                ('1m_position_in_current_candle', PositionInCurrentCandle1mFeature),
                ('1m_body_size_relative', BodySizeRelative1mFeature),
                ('distance_to_ema9_1m', DistanceToEMA9_1mFeature),
                ('distance_to_ema20_1m', DistanceToEMA20_1mFeature),
                ('swing_high_distance', SwingHighDistance1mFeature),
                ('swing_low_distance', SwingLowDistance1mFeature),
                ('price_velocity_1m', PriceVelocity1mFeature),
                ('volume_velocity_1m', VolumeVelocity1mFeature),
                ('distance_to_vwap', DistanceToVWAPFeature),
                ('relative_volume', RelativeVolumeFeature)
            ]
            
            for name, feature_class in mf_feature_classes:
                try:
                    config = FeatureConfig(name=name, enabled=True, normalize=True)
                    features.append(feature_class(config))
                    self.logger.debug(f"Created MF feature: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to create MF feature {name}: {e}")
                    
        except ImportError as e:
            self.logger.warning(f"Failed to import MF features: {e}")
        
        return features
    
    def _create_lf_features(self) -> List[BaseFeature]:
        """Create low-frequency features"""
        features = []
        
        try:
            # Import actual LF feature classes
            from .lf.range_features import PositionInDailyRangeFeature, PriceChangeFromPrevCloseFeature
            from .lf.level_features import DistanceToClosestSupportFeature, WholeDollarProximityFeature
            from .lf.time_features import MarketSessionTypeFeature, TimeOfDaySinFeature, TimeOfDayCosFeature
            
            lf_feature_classes = [
                ('daily_range_position', PositionInDailyRangeFeature),
                ('price_change_from_prev_close', PriceChangeFromPrevCloseFeature),
                ('support_distance', DistanceToClosestSupportFeature),
                ('whole_dollar_proximity', WholeDollarProximityFeature),
                ('market_session_type', MarketSessionTypeFeature),
                ('time_of_day_sin', TimeOfDaySinFeature),
                ('time_of_day_cos', TimeOfDayCosFeature)
            ]
            
            for name, feature_class in lf_feature_classes:
                try:
                    config = FeatureConfig(name=name, enabled=True, normalize=True)
                    features.append(feature_class(config))
                    self.logger.debug(f"Created LF feature: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to create LF feature {name}: {e}")
                    
        except ImportError as e:
            self.logger.warning(f"Failed to import LF features: {e}")
        
        return features
    
    def _create_portfolio_features(self) -> List[BaseFeature]:
        """Create portfolio features"""
        features = []
        
        try:
            # Import actual Portfolio feature classes
            from .portfolio.portfolio_features import PortfolioPositionSizeFeature, PortfolioUnrealizedPnLFeature
            
            portfolio_feature_classes = [
                ('portfolio_position_size', PortfolioPositionSizeFeature),
                ('portfolio_unrealized_pnl', PortfolioUnrealizedPnLFeature)
            ]
            
            for name, feature_class in portfolio_feature_classes:
                try:
                    config = FeatureConfig(name=name, enabled=True, normalize=True)
                    features.append(feature_class(config))
                    self.logger.debug(f"Created Portfolio feature: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to create Portfolio feature {name}: {e}")
                    
        except ImportError as e:
            self.logger.warning(f"Failed to import Portfolio features: {e}")
        
        return features
    
    def calculate_features(self, market_data: Dict[str, Any], category: str) -> Dict[str, float]:
        """Calculate all features in a category"""
        if category not in self._feature_collections:
            self.logger.error(f"Category {category} not found")
            return {}
        
        results = {}
        
        for feature in self._feature_collections[category]:
            if not feature.enabled:
                continue
            
            try:
                value = feature.calculate(market_data)
                # Ensure value is valid
                if value is None or np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Feature {feature.name} returned invalid value ({value}), using 0.0")
                    results[feature.name] = 0.0
                else:
                    results[feature.name] = float(value)
            except Exception as e:
                self.logger.error(f"Error calculating feature {feature.name}: {e}")
                results[feature.name] = 0.0
        
        return results
    
    def vectorize_features(self, features: Dict[str, float], category: str) -> np.ndarray:
        """Convert features to numpy array in order of feature creation"""
        if category not in self._feature_collections:
            self.logger.error(f"No features defined for category {category}")
            return np.array([], dtype=np.float32)
        
        # Get features in the order they were created
        ordered_features = []
        for feature in self._feature_collections[category]:
            if feature.name in features:
                value = features[feature.name]
                # Double-check for invalid values
                if value is None or np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Feature {feature.name} has invalid value ({value}) during vectorization, using 0.0")
                    ordered_features.append(0.0)
                else:
                    ordered_features.append(float(value))
            else:
                self.logger.warning(f"Feature {feature.name} not found in results for category {category}, using 0.0")
                ordered_features.append(0.0)
        
        # Create vector and ensure no NaN/inf values
        vector = np.array(ordered_features, dtype=np.float32)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return vector
    
    def get_enabled_features(self, category: str) -> List[str]:
        """Get list of enabled features in a category"""
        if category not in self._feature_collections:
            return []
        
        return [feature.name for feature in self._feature_collections[category] if feature.enabled]
    
    def enable_feature(self, feature_name: str):
        """Enable a feature by name"""
        for features in self._feature_collections.values():
            for feature in features:
                if feature.name == feature_name:
                    feature.enabled = True
                    return
        self.logger.warning(f"Feature {feature_name} not found")
    
    def disable_feature(self, feature_name: str):
        """Disable a feature by name"""
        for features in self._feature_collections.values():
            for feature in features:
                if feature.name == feature_name:
                    feature.enabled = False
                    return
        self.logger.warning(f"Feature {feature_name} not found")
    
    def extract_features(self, context: MarketContext) -> Optional[Dict[str, np.ndarray]]:
        """Extract features from market context for all categories"""
        try:
            # Extract features for each category
            results = {}
            
            for category in ['hf', 'mf', 'lf', 'portfolio']:
                if category == 'hf':
                    features = self._extract_hf_features(context)
                elif category == 'mf':
                    features = self._extract_mf_features(context)
                elif category == 'lf':
                    features = self._extract_lf_features(context)
                elif category == 'portfolio':
                    features = self._extract_portfolio_features(context)
                
                if features is None:
                    return None
                
                results[category] = features
            
            return results
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _extract_hf_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract high-frequency features"""
        try:
            hf_window = context.hf_window
            if len(hf_window) < self.hf_seq_len:
                return np.zeros((self.hf_seq_len, self.hf_feat_dim), dtype=np.float32)
            
            hf_window = hf_window[-self.hf_seq_len:]
            
            feature_matrix = []
            for i, entry in enumerate(hf_window):
                if entry is None:
                    entry = {'timestamp': None, 'trades': [], 'quotes': [], '1s_bar': None}
                
                market_data = {
                    'timestamp': entry.get('timestamp'),
                    'current_price': entry.get('1s_bar', {}).get('close', 100.0) if entry.get('1s_bar') else 100.0,
                    'hf_data_window': hf_window[max(0, i-10):i+1],
                    'quotes': entry.get('quotes', []),
                    'trades': entry.get('trades', []),
                    '1s_bar': entry.get('1s_bar', {})
                }
                
                features = self.calculate_features(market_data, 'hf')
                vector = self.vectorize_features(features, 'hf')
                
                # Ensure correct dimensions
                if len(vector) < self.hf_feat_dim:
                    padded = np.zeros(self.hf_feat_dim, dtype=np.float32)
                    padded[:len(vector)] = vector
                    vector = padded
                elif len(vector) > self.hf_feat_dim:
                    vector = vector[:self.hf_feat_dim]
                
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
                return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)
            
            mf_window = mf_window[-self.mf_seq_len:]
            bars_5m = context.lf_5m_window
            
            feature_matrix = []
            for i, bar in enumerate(mf_window):
                market_data = {
                    'timestamp': bar.get('timestamp'),
                    'current_price': bar.get('close', 100.0),
                    '1m_bars_window': mf_window[max(0, i-20):i+1],
                    '5m_bars_window': bars_5m,
                    '1m_bar': bar
                }
                
                features = self.calculate_features(market_data, 'mf')
                vector = self.vectorize_features(features, 'mf')
                
                # Ensure correct dimensions
                if len(vector) < self.mf_feat_dim:
                    padded = np.zeros(self.mf_feat_dim, dtype=np.float32)
                    padded[:len(vector)] = vector
                    vector = padded
                elif len(vector) > self.mf_feat_dim:
                    vector = vector[:self.mf_feat_dim]
                
                vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
                feature_matrix.append(vector)
            
            return np.array(feature_matrix, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"MF feature extraction failed: {e}")
            return np.zeros((self.mf_seq_len, self.mf_feat_dim), dtype=np.float32)
    
    def _extract_lf_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract low-frequency features"""
        try:
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
                'daily_bars_window': []
            }
            
            features = self.calculate_features(market_data, 'lf')
            vector = self.vectorize_features(features, 'lf')
            
            # Ensure correct dimensions
            if len(vector) < self.lf_feat_dim:
                padded = np.zeros(self.lf_feat_dim, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
            elif len(vector) > self.lf_feat_dim:
                vector = vector[:self.lf_feat_dim]
            
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Repeat for all timesteps in sequence
            feature_matrix = np.tile(vector, (self.lf_seq_len, 1))
            
            return feature_matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"LF feature extraction failed: {e}")
            return np.zeros((self.lf_seq_len, self.lf_feat_dim), dtype=np.float32)
    
    def _extract_portfolio_features(self, context: MarketContext) -> Optional[np.ndarray]:
        """Extract portfolio features"""
        try:
            if not context.portfolio_state:
                return np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)
            
            market_data = {
                'timestamp': context.timestamp,
                'current_price': context.current_price,
                'portfolio_state': context.portfolio_state
            }
            
            features = self.calculate_features(market_data, 'portfolio')
            vector = self.vectorize_features(features, 'portfolio')
            
            # Ensure correct dimensions
            if len(vector) < self.portfolio_feat_dim:
                padded = np.zeros(self.portfolio_feat_dim, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
            elif len(vector) > self.portfolio_feat_dim:
                vector = vector[:self.portfolio_feat_dim]
            
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Repeat for all timesteps in sequence
            feature_matrix = np.tile(vector, (self.portfolio_seq_len, 1))
            
            return feature_matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Portfolio feature extraction failed: {e}")
            return np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)
    
    def reset(self):
        """Reset feature manager state"""
        pass
    
    def get_data_requirements(self) -> Dict[str, Any]:
        """Get aggregated data requirements from all features"""
        requirements = {}
        
        for category_features in self._feature_collections.values():
            for feature in category_features:
                if not feature.enabled:
                    continue
                
                feat_req = feature.get_requirements()
                data_type = feat_req.get('data_type', 'unknown')
                
                if data_type != 'unknown' and data_type != 'current':
                    if data_type not in requirements:
                        requirements[data_type] = {
                            'lookback': 0,
                            'fields': set()
                        }
                    
                    if 'lookback' in feat_req:
                        requirements[data_type]['lookback'] = max(
                            requirements[data_type]['lookback'],
                            feat_req['lookback']
                        )
                    
                    if 'fields' in feat_req:
                        requirements[data_type]['fields'].update(feat_req['fields'])
        
        # Convert sets to lists
        for data_type in requirements:
            requirements[data_type]['fields'] = list(requirements[data_type]['fields'])
        
        return requirements