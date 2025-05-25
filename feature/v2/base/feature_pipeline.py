"""Feature pipeline for complete feature extraction"""
import numpy as np
from typing import Dict, Any, Optional
from .feature_manager import FeatureManager


class FeaturePipeline:
    """Pipeline for extracting all features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_manager = FeatureManager(config)
    
    def extract_all_features(self, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract all features based on configuration"""
        output_format = self.config.get('output_format', 'dict')
        
        # Extract features using feature manager
        features = self.feature_manager.extract_all_features(market_data)
        
        if output_format == 'concatenated':
            # Concatenate all features into single array
            all_features = []
            
            # Static features first (flattened)
            if 'static' in features:
                all_features.extend(features['static'].flatten())
            
            # HF features (flattened)
            if 'hf' in features:
                all_features.extend(features['hf'].flatten())
            
            # MF features (flattened)
            if 'mf' in features:
                all_features.extend(features['mf'].flatten())
            
            # LF features (flattened)
            if 'lf' in features:
                all_features.extend(features['lf'].flatten())
            
            return np.array(all_features, dtype=np.float32)
        else:
            # Return as dictionary
            return features