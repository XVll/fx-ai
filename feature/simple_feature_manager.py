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
        
        # Initialize feature collections
        self._feature_collections = self._initialize_features()
    
    def _initialize_features(self) -> Dict[str, List[BaseFeature]]:
        """Initialize all feature collections directly"""
        return {
            'hf': self._create_hf_features(),
            'mf': self._create_mf_features(), 
            'lf': self._create_lf_features()
        }
    
    def _create_hf_features(self) -> List[BaseFeature]:
        """Create ALL high-frequency features"""
        features = []
        
        try:
            # Import ALL HF feature classes
            from .hf.price_features import PriceVelocityFeature, PriceAccelerationFeature
            from .hf.tape_features import TapeImbalanceFeature, TapeAggressionRatioFeature
            from .hf.quote_features import SpreadCompressionFeature, QuoteVelocityFeature, QuoteImbalanceFeature
            from .hf.volume_features import VolumeVelocityFeature, VolumeAccelerationFeature
            
            hf_feature_classes = [
                ('price_velocity', PriceVelocityFeature),
                ('price_acceleration', PriceAccelerationFeature),
                ('tape_imbalance', TapeImbalanceFeature),
                ('tape_aggression_ratio', TapeAggressionRatioFeature),
                ('spread_compression', SpreadCompressionFeature),
                ('quote_velocity', QuoteVelocityFeature),
                ('quote_imbalance', QuoteImbalanceFeature),
                ('volume_velocity', VolumeVelocityFeature),
                ('volume_acceleration', VolumeAccelerationFeature)
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
        """Create ALL medium-frequency features"""
        features = []
        
        try:
            # Import ALL MF feature classes
            from .mf.candle_features import (
                PositionInCurrentCandle1mFeature, PositionInCurrentCandle5mFeature,
                BodySizeRelative1mFeature, BodySizeRelative5mFeature
            )
            from .mf.candle_analysis_features import (
                PositionInPreviousCandle1mFeature, PositionInPreviousCandle5mFeature,
                UpperWickRelative1mFeature, LowerWickRelative1mFeature,
                UpperWickRelative5mFeature, LowerWickRelative5mFeature
            )
            from .mf.ema_features import (
                DistanceToEMA9_1mFeature, DistanceToEMA20_1mFeature,
                DistanceToEMA9_5mFeature, DistanceToEMA20_5mFeature
            )
            from .mf.ema_features_v2 import (
                EMAInteractionPatternFeature, EMACrossoverDynamicsFeature, EMATrendAlignmentFeature
            )
            from .mf.swing_features import (
                SwingHighDistance1mFeature, SwingLowDistance1mFeature,
                SwingHighDistance5mFeature, SwingLowDistance5mFeature
            )
            from .mf.velocity_features import (
                PriceVelocity1mFeature, PriceVelocity5mFeature,
                VolumeVelocity1mFeature, VolumeVelocity5mFeature
            )
            from .mf.acceleration_features import (
                PriceAcceleration1mFeature, PriceAcceleration5mFeature,
                VolumeAcceleration1mFeature, VolumeAcceleration5mFeature
            )
            from .volume_analysis.vwap_features import (
                DistanceToVWAPFeature, VWAPSlopeFeature, PriceVWAPDivergenceFeature
            )
            from .volume_analysis.vwap_features_v2 import (
                VWAPInteractionDynamicsFeature, VWAPBreakoutQualityFeature, VWAPMeanReversionTendencyFeature
            )
            from .volume_analysis.relative_volume_features import (
                RelativeVolumeFeature, VolumeSurgeFeature, CumulativeVolumeDeltaFeature, VolumeMomentumFeature
            )
            from .professional.ta_features import (
                ProfessionalEMASystemFeature, ProfessionalVWAPAnalysisFeature,
                ProfessionalMomentumQualityFeature, ProfessionalVolatilityRegimeFeature
            )
            from .sequence_aware.sequence_features import (
                TrendAccelerationFeature, VolumePatternEvolutionFeature,
                MomentumQualityFeature, PatternMaturationFeature
            )
            from .aggregated.aggregated_features import (
                MFTrendConsistencyFeature, MFVolumePriceDivergenceFeature, MFMomentumPersistenceFeature
            )
            from .adaptive.adaptive_features import (
                VolatilityAdjustedMomentumFeature, RegimeRelativeVolumeFeature
            )
            
            mf_feature_classes = [
                # Candle features
                ('1m_position_in_current_candle', PositionInCurrentCandle1mFeature),
                ('5m_position_in_current_candle', PositionInCurrentCandle5mFeature),
                ('1m_body_size_relative', BodySizeRelative1mFeature),
                ('5m_body_size_relative', BodySizeRelative5mFeature),
                ('1m_position_in_previous_candle', PositionInPreviousCandle1mFeature),
                ('5m_position_in_previous_candle', PositionInPreviousCandle5mFeature),
                ('1m_upper_wick_relative', UpperWickRelative1mFeature),
                ('1m_lower_wick_relative', LowerWickRelative1mFeature),
                ('5m_upper_wick_relative', UpperWickRelative5mFeature),
                ('5m_lower_wick_relative', LowerWickRelative5mFeature),
                
                # EMA features
                ('distance_to_ema9_1m', DistanceToEMA9_1mFeature),
                ('distance_to_ema20_1m', DistanceToEMA20_1mFeature),
                ('distance_to_ema9_5m', DistanceToEMA9_5mFeature),
                ('distance_to_ema20_5m', DistanceToEMA20_5mFeature),
                ('ema_interaction_pattern', EMAInteractionPatternFeature),
                ('ema_crossover_dynamics', EMACrossoverDynamicsFeature),
                ('ema_trend_alignment', EMATrendAlignmentFeature),
                
                # Swing features
                ('swing_high_distance_1m', SwingHighDistance1mFeature),
                ('swing_low_distance_1m', SwingLowDistance1mFeature),
                ('swing_high_distance_5m', SwingHighDistance5mFeature),
                ('swing_low_distance_5m', SwingLowDistance5mFeature),
                
                # Velocity features
                ('price_velocity_1m', PriceVelocity1mFeature),
                ('price_velocity_5m', PriceVelocity5mFeature),
                ('volume_velocity_1m', VolumeVelocity1mFeature),
                ('volume_velocity_5m', VolumeVelocity5mFeature),
                
                # Acceleration features
                ('price_acceleration_1m', PriceAcceleration1mFeature),
                ('price_acceleration_5m', PriceAcceleration5mFeature),
                ('volume_acceleration_1m', VolumeAcceleration1mFeature),
                ('volume_acceleration_5m', VolumeAcceleration5mFeature),
                
                # VWAP features
                ('distance_to_vwap', DistanceToVWAPFeature),
                ('vwap_slope', VWAPSlopeFeature),
                ('price_vwap_divergence', PriceVWAPDivergenceFeature),
                ('vwap_interaction_dynamics', VWAPInteractionDynamicsFeature),
                ('vwap_breakout_quality', VWAPBreakoutQualityFeature),
                ('vwap_mean_reversion_tendency', VWAPMeanReversionTendencyFeature),
                
                # Volume features
                ('relative_volume', RelativeVolumeFeature),
                ('volume_surge', VolumeSurgeFeature),
                ('cumulative_volume_delta', CumulativeVolumeDeltaFeature),
                ('volume_momentum', VolumeMomentumFeature),
                
                # Professional features
                ('professional_ema_system', ProfessionalEMASystemFeature),
                ('professional_vwap_analysis', ProfessionalVWAPAnalysisFeature),
                ('professional_momentum_quality', ProfessionalMomentumQualityFeature),
                ('professional_volatility_regime', ProfessionalVolatilityRegimeFeature),
                
                # Sequence-aware features
                ('trend_acceleration', TrendAccelerationFeature),
                ('volume_pattern_evolution', VolumePatternEvolutionFeature),
                ('momentum_quality', MomentumQualityFeature),
                ('pattern_maturation', PatternMaturationFeature),
                
                # Aggregated features
                ('mf_trend_consistency', MFTrendConsistencyFeature),
                ('mf_volume_price_divergence', MFVolumePriceDivergenceFeature),
                ('mf_momentum_persistence', MFMomentumPersistenceFeature),
                
                # Adaptive features
                ('volatility_adjusted_momentum', VolatilityAdjustedMomentumFeature),
                ('regime_relative_volume', RegimeRelativeVolumeFeature)
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
        """Create ALL low-frequency features"""
        features = []
        
        try:
            # Import ALL LF feature classes
            from .lf.range_features import (
                PositionInDailyRangeFeature, PositionInPrevDayRangeFeature, PriceChangeFromPrevCloseFeature
            )
            from .lf.level_features import (
                DistanceToClosestSupportFeature, DistanceToClosestResistanceFeature,
                WholeDollarProximityFeature, HalfDollarProximityFeature
            )
            from .lf.time_features import (
                MarketSessionTypeFeature, TimeOfDaySinFeature, TimeOfDayCosFeature
            )
            from .market_structure.halt_features import (
                HaltStateFeature, TimeSinceHaltFeature
            )
            from .market_structure.luld_features import (
                DistanceToLULDUpFeature, DistanceToLULDDownFeature, LULDBandWidthFeature
            )
            from .context.context_features import (
                SessionProgressFeature, MarketStressFeature, SessionVolumeProfileFeature
            )
            from .adaptive.adaptive_features import (
                AdaptiveSupportResistanceFeature,
            )
            from .aggregated.aggregated_features import (
                HFMomentumSummaryFeature, HFVolumeDynamicsFeature, HFMicrostructureQualityFeature
            )
            
            lf_feature_classes = [
                # Range features
                ('daily_range_position', PositionInDailyRangeFeature),
                ('prev_day_range_position', PositionInPrevDayRangeFeature),
                ('price_change_from_prev_close', PriceChangeFromPrevCloseFeature),
                
                # Level features
                ('support_distance', DistanceToClosestSupportFeature),
                ('resistance_distance', DistanceToClosestResistanceFeature),
                ('whole_dollar_proximity', WholeDollarProximityFeature),
                ('half_dollar_proximity', HalfDollarProximityFeature),
                
                # Time features
                ('market_session_type', MarketSessionTypeFeature),
                ('time_of_day_sin', TimeOfDaySinFeature),
                ('time_of_day_cos', TimeOfDayCosFeature),
                
                # Market structure features
                ('halt_state', HaltStateFeature),
                ('time_since_halt', TimeSinceHaltFeature),
                ('distance_to_luld_up', DistanceToLULDUpFeature),
                ('distance_to_luld_down', DistanceToLULDDownFeature),
                ('luld_band_width', LULDBandWidthFeature),
                
                # Context features
                ('session_progress', SessionProgressFeature),
                ('market_stress', MarketStressFeature),
                ('session_volume_profile', SessionVolumeProfileFeature),
                
                # Adaptive features
                ('adaptive_support_resistance', AdaptiveSupportResistanceFeature),
                
                # Aggregated HF summary features (moved to LF for daily context)
                ('hf_momentum_summary', HFMomentumSummaryFeature),
                ('hf_volume_dynamics', HFVolumeDynamicsFeature),
                ('hf_microstructure_quality', HFMicrostructureQualityFeature)
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
    
    def calculate_features(self, market_data: Dict[str, Any], category: str) -> Dict[str, float]:
        """Calculate all features in a category"""
        if category not in self._feature_collections:
            self.logger.error(f"Category {category} not found")
            return {}
        
        results = {}
        
        total_features = len([f for f in self._feature_collections[category] if f.enabled])
        feature_idx = 0
        
        for feature in self._feature_collections[category]:
            if not feature.enabled:
                continue
            
            feature_idx += 1
            self.logger.debug(f"DEBUG: Calculating {category} feature {feature_idx}/{total_features}: {feature.name}")
            
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
            
            self.logger.debug(f"DEBUG: Feature {feature.name} calculated: {results.get(feature.name, 'ERROR')}")
        
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
            
            for category in ['hf', 'mf', 'lf']:
                self.logger.debug(f"DEBUG: Extracting {category} features")
                
                features = None
                if category == 'hf':
                    features = self._extract_hf_features(context)
                elif category == 'mf':
                    features = self._extract_mf_features(context)
                elif category == 'lf':
                    features = self._extract_lf_features(context)
                
                self.logger.debug(f"DEBUG: {category} features extracted, shape: {features.shape if features is not None else 'None'}")
                
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