"""Feature manager for batch calculation and management"""
from typing import Dict, List, Any, Optional
import numpy as np
from .feature_base import BaseFeature, FeatureConfig
from .feature_registry import feature_registry
from .load_features import load_all_features

# Load all features on import
load_all_features()


class FeatureManager:
    """Manages feature calculation and aggregation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = feature_registry
        self._features: Dict[str, Dict[str, BaseFeature]] = {}
        self._feature_order: Dict[str, List[str]] = {}
        
        # Load configured features
        self._load_features()
    
    def _load_features(self):
        """Load features based on configuration"""
        feature_config = self.config.get('features', {})
        
        for category, feature_names in feature_config.items():
            self._features[category] = {}
            self._feature_order[category] = feature_names
            
            for feature_name in feature_names:
                # Get feature class from registry
                if self.registry.has_feature(feature_name):
                    feature_class = self.registry.get_feature_class(feature_name)
                    
                    # Create feature instance
                    config = FeatureConfig(
                        name=feature_name,
                        enabled=True,
                        normalize=self.config.get('normalization', {}).get('enabled', True),
                        params={}
                    )
                    
                    self._features[category][feature_name] = feature_class(config)
                else:
                    # Feature not found in registry - will be handled by error handling
                    pass
    
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
        # Check if we need to reload features for missing ones
        if category in self.config.get('features', {}):
            for feature_name in self.config['features'][category]:
                if (category not in self._features or 
                    feature_name not in self._features[category]):
                    # Try to load missing feature
                    if self.registry.has_feature(feature_name):
                        if category not in self._features:
                            self._features[category] = {}
                        feature_class = self.registry.get_feature_class(feature_name)
                        config = FeatureConfig(
                            name=feature_name,
                            enabled=True,
                            normalize=self.config.get('normalization', {}).get('enabled', True),
                            params={}
                        )
                        self._features[category][feature_name] = feature_class(config)
        
        if category not in self._features:
            return {}
        
        results = {}
        error_handling = self.config.get('error_handling', {})
        
        for name in self.config.get('features', {}).get(category, []):
            if category in self._features and name in self._features[category]:
                feature = self._features[category][name]
                if not feature.enabled:
                    continue
                
                try:
                    value = feature.calculate(market_data)
                    results[name] = value
                except Exception as e:
                    if error_handling.get('on_error') == 'use_default':
                        results[name] = error_handling.get('default_value', 0.0)
                    else:
                        raise
            else:
                # Feature not found
                if error_handling.get('on_error') == 'use_default':
                    results[name] = error_handling.get('default_value', 0.0)
        
        return results
    
    def vectorize_features(self, features: Dict[str, float], 
                          category: str) -> np.ndarray:
        """Convert features to numpy array in consistent order"""
        if category not in self._feature_order:
            return np.array([], dtype=np.float32)
        
        # Get features in specified order
        ordered_features = []
        for feature_name in self._feature_order[category]:
            if feature_name in features:
                ordered_features.append(features[feature_name])
            else:
                # Use default if missing
                ordered_features.append(0.0)
        
        return np.array(ordered_features, dtype=np.float32)
    
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
    
    def extract_all_features(self, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract all features for all categories"""
        seq_lengths = self.config.get('sequence_lengths', {})
        results = {}
        
        for category in self._features:
            if category == 'static':
                # Static features are single timestep
                features = self.calculate_features(market_data, category)
                vector = self.vectorize_features(features, category)
                results[category] = vector.reshape(1, -1)
            
            elif category == 'hf':
                # High-frequency features from window
                seq_len = seq_lengths.get('hf', 60)
                hf_window = market_data.get('hf_data_window', [])
                
                # Extract features for each timestep
                feature_matrix = []
                for i in range(seq_len):
                    if i < len(hf_window):
                        step_data = {
                            'timestamp': hf_window[i].get('timestamp'),
                            'current_price': hf_window[i].get('1s_bar', {}).get('close', 100.0),
                            'hf_data_window': hf_window[max(0, i-10):i+1]
                        }
                    else:
                        step_data = {'current_price': 100.0, 'hf_data_window': []}
                    
                    features = self.calculate_features(step_data, category)
                    vector = self.vectorize_features(features, category)
                    feature_matrix.append(vector)
                
                results[category] = np.array(feature_matrix, dtype=np.float32)
            
            elif category == 'mf':
                # Medium-frequency features
                seq_len = seq_lengths.get('mf', 30)
                bars_1m = market_data.get('1m_bars_window', [])
                
                feature_matrix = []
                for i in range(seq_len):
                    if i < len(bars_1m):
                        step_data = {
                            'timestamp': bars_1m[i].get('timestamp'),
                            'current_price': bars_1m[i].get('close', 100.0),
                            '1m_bars_window': bars_1m[max(0, i-20):i+1],
                            '5m_bars_window': market_data.get('5m_bars_window', [])
                        }
                    else:
                        step_data = {'current_price': 100.0, '1m_bars_window': []}
                    
                    features = self.calculate_features(step_data, category)
                    vector = self.vectorize_features(features, category)
                    feature_matrix.append(vector)
                
                results[category] = np.array(feature_matrix, dtype=np.float32)
            
            elif category == 'lf':
                # Low-frequency features
                seq_len = seq_lengths.get('lf', 10)
                daily_bars = market_data.get('daily_bars_window', [])
                
                feature_matrix = []
                for i in range(seq_len):
                    if i < len(daily_bars):
                        step_data = {
                            'timestamp': daily_bars[i].get('timestamp'),
                            'current_price': daily_bars[i].get('close', 100.0),
                            'intraday_high': daily_bars[i].get('high', 100.0),
                            'intraday_low': daily_bars[i].get('low', 100.0),
                            'daily_bars_window': daily_bars[max(0, i-20):i],
                            'previous_day_data': daily_bars[i-1] if i > 0 else None
                        }
                    else:
                        step_data = {'current_price': 100.0}
                    
                    features = self.calculate_features(step_data, category)
                    vector = self.vectorize_features(features, category)
                    feature_matrix.append(vector)
                
                results[category] = np.array(feature_matrix, dtype=np.float32)
        
        return results