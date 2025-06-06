#!/usr/bin/env python3
"""Test script to verify feature consistency across the system."""

import sys
import os
import logging
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_feature_consistency():
    """Test that all components use the same feature definitions."""
    
    # Import components
    from feature.feature_registry import FeatureRegistry
    from feature.simple_feature_manager import SimpleFeatureManager
    from config.schemas import ModelConfig
    from feature.attribution.captum_attribution import AttributionConfig
    
    logger.info("=" * 80)
    logger.info("Testing Feature Consistency - Single Source of Truth")
    logger.info("=" * 80)
    
    # 1. Get dimensions from FeatureRegistry
    logger.info("\n1. FeatureRegistry Dimensions:")
    registry_dims = FeatureRegistry.get_feature_dimensions(active_only=True)
    for category, count in registry_dims.items():
        logger.info(f"   {category}: {count} active features")
    
    # 2. Create SimpleFeatureManager and check feature counts
    logger.info("\n2. SimpleFeatureManager Feature Counts:")
    
    # Mock config for testing
    class MockConfig:
        hf_seq_len = 60
        hf_feat_dim = registry_dims['hf']
        mf_seq_len = 30
        mf_feat_dim = registry_dims['mf']
        lf_seq_len = 30
        lf_feat_dim = registry_dims['lf']
    
    manager = SimpleFeatureManager("TEST", MockConfig(), logger)
    
    for category in ["hf", "mf", "lf"]:
        enabled_features = manager.get_enabled_features(category)
        logger.info(f"   {category}: {len(enabled_features)} features created")
        
        # Check if counts match
        if len(enabled_features) != registry_dims[category]:
            logger.error(f"   ❌ MISMATCH: Expected {registry_dims[category]}, got {len(enabled_features)}")
        else:
            logger.info(f"   ✅ Matches registry count")
    
    # 3. Check ModelConfig dimensions
    logger.info("\n3. ModelConfig Dimensions:")
    model_config = ModelConfig()
    config_dims = {
        'hf': model_config.hf_feat_dim,
        'mf': model_config.mf_feat_dim,
        'lf': model_config.lf_feat_dim,
        'portfolio': model_config.portfolio_feat_dim
    }
    
    for category, dim in config_dims.items():
        logger.info(f"   {category}_feat_dim: {dim}")
        if dim != registry_dims[category]:
            logger.warning(f"   ⚠️  Config has {dim} but registry has {registry_dims[category]}")
    
    # 4. Test ModelConfig.from_registry()
    logger.info("\n4. ModelConfig.from_registry():")
    registry_config = ModelConfig.from_registry()
    registry_config_dims = {
        'hf': registry_config.hf_feat_dim,
        'mf': registry_config.mf_feat_dim,
        'lf': registry_config.lf_feat_dim,
        'portfolio': registry_config.portfolio_feat_dim
    }
    
    for category, dim in registry_config_dims.items():
        logger.info(f"   {category}_feat_dim: {dim}")
        if dim == registry_dims[category]:
            logger.info(f"   ✅ Correctly set from registry")
    
    # 5. Check feature names match between registry and manager
    logger.info("\n5. Feature Name Consistency:")
    for category in ["hf", "mf", "lf"]:
        registry_names = FeatureRegistry.get_feature_names(category, active_only=True)
        manager_names = manager.get_enabled_features(category)
        
        logger.info(f"\n   {category.upper()} Features:")
        if len(registry_names) != len(manager_names):
            logger.error(f"   ❌ Count mismatch: Registry has {len(registry_names)}, Manager has {len(manager_names)}")
        else:
            # Check order
            for i, (reg_name, mgr_name) in enumerate(zip(registry_names, manager_names)):
                if reg_name != mgr_name:
                    logger.error(f"   ❌ Order mismatch at index {i}: Registry='{reg_name}', Manager='{mgr_name}'")
                    break
            else:
                logger.info(f"   ✅ All {len(registry_names)} features match in order")
    
    # 6. Check Captum integration
    logger.info("\n6. Captum Integration:")
    attr_config = AttributionConfig()
    feature_groups = FeatureRegistry.get_all_feature_groups(active_only=True)
    logger.info(f"   Total feature groups: {len(feature_groups)}")
    
    total_features = sum(len(features) for features in feature_groups.values())
    expected_total = sum(registry_dims.values())
    
    if total_features == expected_total:
        logger.info(f"   ✅ All {total_features} features are grouped correctly")
    else:
        logger.error(f"   ❌ Feature group total {total_features} != expected {expected_total}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Test Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_feature_consistency()