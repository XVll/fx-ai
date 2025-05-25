"""Tests for base feature system - TDD approach"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Import statements will fail initially (TDD)
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import FeatureRegistry
from feature.feature_manager import FeatureManager
from feature.normalizers import MinMaxNormalizer, StandardNormalizer


class TestBaseFeature:
    """Test base feature interface"""
    
    def test_base_feature_interface(self):
        """Test that BaseFeature has required interface"""
        # Create a simple feature implementation
        class SimpleFeature(BaseFeature):
            def __init__(self, config: FeatureConfig):
                super().__init__(config)
            
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                return 1.0
            
            def get_default_value(self) -> float:
                return 0.5
            
            def get_normalization_params(self) -> Dict[str, Any]:
                return {"min": 0.0, "max": 2.0}
            
            def get_requirements(self) -> Dict[str, Any]:
                return {"data_type": "1s_bars", "lookback": 1}
        
        config = FeatureConfig(
            name="test_feature",
            enabled=True,
            normalize=True,
            params={}
        )
        
        feature = SimpleFeature(config)
        
        # Test interface methods
        assert feature.name == "test_feature"
        assert feature.enabled is True
        
        # Test calculation returns normalized value
        result = feature.calculate({})
        assert result == 0.5  # (1.0 - 0.0) / (2.0 - 0.0) = 0.5
        assert 0.0 <= result <= 1.0  # Normalized
        
        # Test raw calculation
        assert feature.calculate_raw({}) == 1.0
        assert feature.get_requirements() == {"data_type": "1s_bars", "lookback": 1}
    
    def test_feature_validation(self):
        """Test feature validates input data"""
        class ValidatingFeature(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                # Instead of raising, use defaults for missing data
                price = market_data.get("current_price", self.get_default_value())
                return price
            
            def get_default_value(self) -> float:
                return 100.0  # Default price
            
            def get_normalization_params(self) -> Dict[str, Any]:
                return {"min": 0.0, "max": 200.0}
            
            def get_requirements(self) -> Dict[str, Any]:
                return {"fields": ["current_price", "volume"]}
        
        config = FeatureConfig(name="validating_feature", enabled=True, normalize=True)
        feature = ValidatingFeature(config)
        
        # Should use default on missing data (never NaN)
        result = feature.calculate({})
        assert result == 0.5  # 100.0 normalized to [0,1]
        assert not np.isnan(result)
        
        # Should work with complete data
        result = feature.calculate({"current_price": 150.0, "volume": 1000})
        assert result == 0.75  # 150.0 normalized to [0,1]
        assert not np.isnan(result)
    
    def test_no_nan_values(self):
        """Test that features never return NaN values"""
        class RobustFeature(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                # Simulate division by zero scenario
                volume = market_data.get("volume", 0)
                if volume == 0:
                    return self.get_default_value()
                return 1000.0 / volume
            
            def get_default_value(self) -> float:
                return 0.0
            
            def get_normalization_params(self) -> Dict[str, Any]:
                return {"min": 0.0, "max": 10.0}
            
            def get_requirements(self) -> Dict[str, Any]:
                return {"fields": ["volume"]}
        
        config = FeatureConfig(name="robust_feature", enabled=True, normalize=True)
        feature = RobustFeature(config)
        
        # Test various edge cases that could produce NaN
        test_cases = [
            {},  # Empty data
            {"volume": 0},  # Division by zero
            {"volume": None},  # None value
            {"volume": np.nan},  # Explicit NaN
            {"volume": float('inf')},  # Infinity
        ]
        
        for data in test_cases:
            result = feature.calculate(data)
            assert not np.isnan(result), f"NaN produced with data: {data}"
            assert np.isfinite(result), f"Non-finite value produced with data: {data}"
            assert 0.0 <= result <= 1.0, f"Value out of normalized range with data: {data}"
    
    def test_feature_caching(self):
        """Test feature caching capability"""
        class CachedFeature(BaseFeature):
            def __init__(self, config: FeatureConfig):
                super().__init__(config)
                self._call_count = 0
            
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                self._call_count += 1
                return self._call_count
            
            def get_default_value(self) -> float:
                return 0.0
            
            def get_normalization_params(self) -> Dict[str, Any]:
                return {}
            
            def get_requirements(self) -> Dict[str, Any]:
                return {"cache_duration": 1}  # 1 second cache
        
        config = FeatureConfig(name="cached_feature", enabled=True, cache_enabled=True)
        feature = CachedFeature(config)
        
        timestamp = datetime.now(timezone.utc)
        data = {"timestamp": timestamp}
        
        # First call
        result1 = feature.get_value(data)
        assert result1 == 1
        
        # Second call with same timestamp - should use cache
        result2 = feature.get_value(data)
        assert result2 == 1  # Same result, not incremented
        
        # Call with different timestamp - should recalculate
        from datetime import timedelta
        data["timestamp"] = timestamp + timedelta(seconds=2)
        result3 = feature.get_value(data)
        assert result3 == 2  # Incremented


class TestFeatureRegistry:
    """Test feature registry system"""
    
    def test_registry_registration(self):
        """Test registering features"""
        registry = FeatureRegistry()
        
        # Register a feature class
        @registry.register("test_feature")
        class TestFeature(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                return 42.0
            
            def get_default_value(self) -> float:
                return 42.0
            
            def get_normalization_params(self) -> Dict[str, Any]:
                return {}
            
            def get_requirements(self) -> Dict[str, Any]:
                return {}
        
        # Check registration
        assert registry.has_feature("test_feature")
        assert not registry.has_feature("unknown_feature")
        
        # Get feature class
        feature_class = registry.get_feature_class("test_feature")
        assert feature_class == TestFeature
        
        # List all features
        features = registry.list_features()
        assert "test_feature" in features
    
    def test_registry_categories(self):
        """Test feature categorization"""
        registry = FeatureRegistry()
        
        # Register features in different categories
        @registry.register("hf_velocity", category="hf")
        class HFFeature(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                return 0.0
            def get_default_value(self) -> float:
                return 0.0
            def get_normalization_params(self) -> Dict[str, Any]:
                return {}
            def get_requirements(self) -> Dict[str, Any]:
                return {}
        
        @registry.register("mf_ema", category="mf")
        class MFFeature(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                return 0.0
            def get_default_value(self) -> float:
                return 0.0
            def get_normalization_params(self) -> Dict[str, Any]:
                return {}
            def get_requirements(self) -> Dict[str, Any]:
                return {}
        
        # Get features by category
        hf_features = registry.get_features_by_category("hf")
        mf_features = registry.get_features_by_category("mf")
        
        assert "hf_velocity" in hf_features
        assert "mf_ema" in mf_features
        assert "hf_velocity" not in mf_features
    
    def test_registry_duplicate_prevention(self):
        """Test that duplicate registrations are prevented"""
        registry = FeatureRegistry()
        
        @registry.register("duplicate_test")
        class Feature1(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                return 0.0
            def get_default_value(self) -> float:
                return 0.0
            def get_normalization_params(self) -> Dict[str, Any]:
                return {}
            def get_requirements(self) -> Dict[str, Any]:
                return {}
        
        # Should raise on duplicate registration
        with pytest.raises(ValueError, match="already registered"):
            @registry.register("duplicate_test")
            class Feature2(BaseFeature):
                def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                    return 0.0
                def get_default_value(self) -> float:
                    return 0.0
                def get_normalization_params(self) -> Dict[str, Any]:
                    return {}
                def get_requirements(self) -> Dict[str, Any]:
                    return {}


class TestFeatureNormalization:
    """Test feature normalization"""
    
    def test_minmax_normalizer(self):
        """Test min-max normalization"""
        normalizer = MinMaxNormalizer(min_val=0.0, max_val=100.0)
        
        # Test normal cases
        assert normalizer.normalize(50.0) == 0.5
        assert normalizer.normalize(0.0) == 0.0
        assert normalizer.normalize(100.0) == 1.0
        
        # Test out of range values (should be clipped)
        assert normalizer.normalize(-10.0) == 0.0
        assert normalizer.normalize(110.0) == 1.0
        
        # Test edge cases
        assert normalizer.normalize(np.inf) == 1.0
        assert normalizer.normalize(-np.inf) == 0.0
        assert normalizer.normalize(np.nan) == 0.5  # Default to middle
    
    def test_standard_normalizer(self):
        """Test standard (z-score) normalization"""
        normalizer = StandardNormalizer(mean=100.0, std=10.0, clip_range=3.0)
        
        # Test normal cases
        assert normalizer.normalize(100.0) == 0.0  # At mean
        assert normalizer.normalize(110.0) == 1.0  # +1 std
        assert normalizer.normalize(90.0) == -1.0  # -1 std
        
        # Test clipping at 3 std
        assert normalizer.normalize(130.0) == 3.0  # Clipped
        assert normalizer.normalize(70.0) == -3.0  # Clipped
        
        # Test edge cases
        assert normalizer.normalize(np.nan) == 0.0  # Default to mean
        assert abs(normalizer.normalize(np.inf)) == 3.0  # Clipped


class TestFeatureManager:
    """Test feature management system"""
    
    def test_feature_manager_initialization(self):
        """Test feature manager setup"""
        config = {
            "features": {
                "static": ["time_of_day_sin", "time_of_day_cos", "market_session_type"],
                "hf": ["price_velocity", "tape_imbalance"],
                "mf": ["1m_price_velocity", "1m_ema9_distance"],
                "lf": ["daily_range_position", "support_distance"]
            },
            "normalization": {
                "enabled": True,
                "method": "minmax"  # or "standard"
            }
        }
        
        manager = FeatureManager(config)
        
        # Check feature loading
        assert len(manager.get_enabled_features("static")) == 3
        assert len(manager.get_enabled_features("hf")) == 2
        assert len(manager.get_enabled_features("mf")) == 2
        assert len(manager.get_enabled_features("lf")) == 2
    
    def test_feature_calculation_batch(self):
        """Test batch feature calculation"""
        manager = FeatureManager({
            "features": {
                "static": ["time_of_day_sin", "market_session_type"]
            }
        })
        
        market_data = {
            "timestamp": datetime(2025, 1, 25, 10, 30, 0, tzinfo=timezone.utc),
            "current_price": 100.0,
            "market_session": "REGULAR"
        }
        
        # Calculate all features
        features = manager.calculate_features(market_data, category="static")
        
        assert "time_of_day_sin" in features
        assert "market_session_type" in features
        assert isinstance(features["time_of_day_sin"], float)
        assert -1.0 <= features["time_of_day_sin"] <= 1.0
    
    def test_feature_requirements_aggregation(self):
        """Test aggregating data requirements from all features"""
        manager = FeatureManager({
            "features": {
                "hf": ["price_velocity", "tape_imbalance"],
                "mf": ["1m_ema9_distance"]
            }
        })
        
        # Get aggregated requirements
        requirements = manager.get_data_requirements()
        
        # Should have requirements for different data types
        assert "1s_bars" in requirements
        assert "1m_bars" in requirements
        assert "trades" in requirements
        
        # Check lookback periods
        assert requirements["1s_bars"]["lookback"] >= 2  # For velocity
        assert requirements["1m_bars"]["lookback"] >= 9  # For EMA9
    
    def test_feature_enable_disable(self):
        """Test enabling/disabling features at runtime"""
        manager = FeatureManager({
            "features": {
                "hf": ["price_velocity", "tape_imbalance"]
            }
        })
        
        # Initially all enabled
        assert manager.is_feature_enabled("price_velocity")
        assert manager.is_feature_enabled("tape_imbalance")
        
        # Disable a feature
        manager.disable_feature("tape_imbalance")
        assert not manager.is_feature_enabled("tape_imbalance")
        
        # Calculate features - disabled one should not appear
        features = manager.calculate_features({}, category="hf")
        assert "price_velocity" in features
        assert "tape_imbalance" not in features
        
        # Re-enable
        manager.enable_feature("tape_imbalance")
        assert manager.is_feature_enabled("tape_imbalance")
    
    def test_feature_error_handling(self):
        """Test graceful error handling in feature calculation"""
        manager = FeatureManager({
            "features": {
                "hf": ["price_velocity", "failing_feature"]
            },
            "error_handling": {
                "on_error": "use_default",
                "default_value": 0.0,
                "log_errors": True
            }
        })
        
        # Register a failing feature
        @manager.registry.register("failing_feature", category="hf")
        class FailingFeature(BaseFeature):
            def calculate_raw(self, market_data: Dict[str, Any]) -> float:
                raise RuntimeError("Feature calculation failed")
            
            def get_default_value(self) -> float:
                return 0.0
            
            def get_normalization_params(self) -> Dict[str, Any]:
                return {}
            
            def get_requirements(self) -> Dict[str, Any]:
                return {}
        
        # Calculate features - should not crash
        features = manager.calculate_features({}, category="hf")
        
        # Failed feature should have default value
        assert features["failing_feature"] == 0.0
        # Other features should still work
        assert "price_velocity" in features
    
    def test_feature_vectorization(self):
        """Test converting features to numpy arrays for model input"""
        manager = FeatureManager({
            "features": {
                "static": ["time_of_day_sin", "time_of_day_cos", "market_session_type"],
                "hf": ["price_velocity", "tape_imbalance"]
            },
            "feature_order": {
                "static": ["time_of_day_sin", "time_of_day_cos", "market_session_type"],
                "hf": ["price_velocity", "tape_imbalance"]
            },
            "normalization": {"enabled": True}
        })
        
        market_data = {
            "timestamp": datetime.now(timezone.utc),
            "current_price": 100.0,
            "market_session": "REGULAR"
        }
        
        # Calculate and vectorize
        static_features = manager.calculate_features(market_data, category="static")
        static_vector = manager.vectorize_features(static_features, category="static")
        
        # Check shape and type
        assert isinstance(static_vector, np.ndarray)
        assert static_vector.shape == (3,)  # 3 static features
        assert static_vector.dtype == np.float32
        
        # Check no NaN values
        assert not np.any(np.isnan(static_vector))
        
        # Check all values are normalized (between -1 and 1 for trig, 0-1 for others)
        assert np.all(np.isfinite(static_vector))
        assert -1.0 <= static_vector[0] <= 1.0  # sin
        assert -1.0 <= static_vector[1] <= 1.0  # cos
        assert 0.0 <= static_vector[2] <= 1.0   # session type
    
    def test_final_output_format(self):
        """Test the final output format for model consumption"""
        manager = FeatureManager({
            "features": {
                "static": ["time_of_day_sin", "market_session_type"],
                "hf": ["price_velocity"],
                "mf": ["1m_price_velocity"],
                "lf": ["daily_range_position"]
            },
            "sequence_lengths": {
                "hf": 60,  # 60 seconds
                "mf": 30,  # 30 minutes
                "lf": 10   # 10 days
            }
        })
        
        # Simulate market data with time series
        market_data = {
            "timestamp": datetime.now(timezone.utc),
            "current_price": 100.0,
            "market_session": "REGULAR",
            "hf_data_window": [{"1s_bar": {"close": 100.0 + i*0.01}} for i in range(60)],
            "1m_bars_window": [{"close": 100.0 + i*0.1} for i in range(30)],
            "daily_bars_window": [{"high": 105.0, "low": 95.0} for _ in range(10)]
        }
        
        # Get complete feature set
        features = manager.extract_all_features(market_data)
        
        # Check output structure
        assert "static" in features
        assert "hf" in features
        assert "mf" in features
        assert "lf" in features
        
        # Check shapes
        assert features["static"].shape == (1, 2)  # 1 time step, 2 features
        assert features["hf"].shape == (60, 1)     # 60 time steps, 1 feature
        assert features["mf"].shape == (30, 1)     # 30 time steps, 1 feature
        assert features["lf"].shape == (10, 1)     # 10 time steps, 1 feature
        
        # Check no NaN values in any output
        for key, array in features.items():
            assert not np.any(np.isnan(array)), f"NaN found in {key} features"
            assert np.all(np.isfinite(array)), f"Non-finite values found in {key} features"
            
        # Portfolio features handled separately (as mentioned)
        assert "portfolio" not in features