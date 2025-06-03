#!/usr/bin/env python3
"""
Debug script to check Captum attribution system status
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from feature.attribution import CaptumFeatureAnalyzer, AttributionConfig
from main import get_feature_names_from_config

def test_captum_attribution():
    """Test if Captum attribution system is working"""
    
    print("🔍 Testing Captum Attribution System")
    print("=" * 50)
    
    # Test 1: Check Captum import
    try:
        import captum
        print(f"✅ Captum installed: version {captum.__version__}")
    except ImportError as e:
        print(f"❌ Captum import failed: {e}")
        return False
    
    # Test 2: Check feature names
    try:
        feature_names = get_feature_names_from_config()
        total_features = sum(len(names) for names in feature_names.values())
        print(f"✅ Feature names loaded: {total_features} total features")
        for branch, names in feature_names.items():
            print(f"   - {branch}: {len(names)} features")
    except Exception as e:
        print(f"❌ Feature names failed: {e}")
        return False
    
    # Test 3: Check AttributionConfig
    try:
        config = AttributionConfig(
            n_steps=5,  # Very small for testing
            n_samples=3,
            use_noise_tunnel=False,  # Disable for speed
            track_gradients=True,
            track_activations=True
        )
        print(f"✅ AttributionConfig created: {config.primary_method}")
    except Exception as e:
        print(f"❌ AttributionConfig failed: {e}")
        return False
    
    # Test 4: Try to create CaptumFeatureAnalyzer (without model)
    try:
        # This will fail without a model, but we can see the error
        analyzer = CaptumFeatureAnalyzer(
            model=None,  # This will cause a controlled failure
            feature_names=feature_names,
            config=config,
            logger=logging.getLogger(__name__)
        )
    except Exception as e:
        if "model" in str(e).lower():
            print(f"✅ CaptumFeatureAnalyzer class loadable (expected model error: {e})")
        else:
            print(f"❌ CaptumFeatureAnalyzer unexpected error: {e}")
            return False
    
    return True

def check_metrics_system():
    """Check if metrics system is properly configured"""
    
    print("\n🔍 Testing Metrics System Integration")
    print("=" * 50)
    
    try:
        from metrics.collectors.model_internals_metrics import ModelInternalsCollector
        
        # Test basic collector creation
        collector = ModelInternalsCollector(
            model=None,
            feature_names=None,
            enable_attribution=True
        )
        
        print(f"✅ ModelInternalsCollector created")
        print(f"   - Attribution enabled: {collector.enable_attribution}")
        print(f"   - Attribution analyzer: {collector.attribution_analyzer}")
        print(f"   - Analysis interval: {collector.attribution_analysis_interval}s")
        print(f"   - State buffer size: {len(collector.state_buffer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics system error: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    success &= test_captum_attribution()
    success &= check_metrics_system()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed - Captum attribution system should work")
        print("\n🔍 Debug Info for Running Training:")
        print("- Attribution runs every 10 updates (so after update 10, 20, 30...)")
        print("- Attribution needs 30 second cooldown between runs")
        print("- Attribution needs 5+ states in buffer")
        print("- Look for logs: '🔍 Starting Captum attribution analysis'")
        print("- Expected W&B metrics: model.internals.top_feature_importance_*")
    else:
        print("❌ Some tests failed - attribution system has issues")
    
    print("\n🎯 To force attribution analysis, reduce the interval to 1 second:")
    print("   Edit metrics/collectors/model_internals_metrics.py line 50")
    print("   Change: attribution_analysis_interval = 1  # 1 second for testing")