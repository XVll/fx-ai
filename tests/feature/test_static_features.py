"""Tests for static features - TDD approach"""
import pytest
import numpy as np
from datetime import datetime, timezone, time
from typing import Dict, Any, Optional

# These imports will fail initially (TDD)
from feature.static.time_features import TimeOfDaySinFeature, TimeOfDayCosFeature
from feature.static.session_features import MarketSessionTypeFeature


class TestTimeFeatures:
    """Test time-based static features"""
    
    def test_time_of_day_sin_feature(self):
        """Test sine encoding of time of day"""
        feature = TimeOfDaySinFeature()
        
        # Test cases for different times
        test_cases = [
            # Market open (9:30 AM ET)
            {
                "timestamp": datetime(2025, 1, 25, 14, 30, 0, tzinfo=timezone.utc),  # 9:30 AM ET
                "expected_range": (0.99, 1.0)  # Near peak (morning)
            },
            # Noon (12:00 PM ET)
            {
                "timestamp": datetime(2025, 1, 25, 17, 0, 0, tzinfo=timezone.utc),  # 12:00 PM ET
                "expected_range": (0.0, 0.1)  # Near zero
            },
            # Market close (4:00 PM ET)
            {
                "timestamp": datetime(2025, 1, 25, 21, 0, 0, tzinfo=timezone.utc),  # 4:00 PM ET
                "expected_range": (-1.0, -0.99)  # Near trough (afternoon)
            },
            # Pre-market (7:00 AM ET)
            {
                "timestamp": datetime(2025, 1, 25, 12, 0, 0, tzinfo=timezone.utc),  # 7:00 AM ET
                "expected_range": (0.5, 0.6)  # Morning side
            }
        ]
        
        for test_case in test_cases:
            market_data = {"timestamp": test_case["timestamp"]}
            result = feature.calculate(market_data)
            
            assert isinstance(result, float)
            assert -1.0 <= result <= 1.0
            assert test_case["expected_range"][0] <= result <= test_case["expected_range"][1]
    
    def test_time_of_day_cos_feature(self):
        """Test cosine encoding of time of day"""
        feature = TimeOfDayCosFeature()
        
        # Test complementary nature with sine
        test_times = [
            datetime(2025, 1, 25, 14, 30, 0, tzinfo=timezone.utc),  # 9:30 AM ET
            datetime(2025, 1, 25, 17, 0, 0, tzinfo=timezone.utc),   # 12:00 PM ET
            datetime(2025, 1, 25, 21, 0, 0, tzinfo=timezone.utc),   # 4:00 PM ET
        ]
        
        sin_feature = TimeOfDaySinFeature()
        
        for timestamp in test_times:
            market_data = {"timestamp": timestamp}
            
            sin_val = sin_feature.calculate(market_data)
            cos_val = feature.calculate(market_data)
            
            # Check that sin^2 + cos^2 = 1 (within floating point tolerance)
            assert abs(sin_val**2 + cos_val**2 - 1.0) < 1e-6
    
    def test_time_encoding_continuity(self):
        """Test that time encoding is continuous"""
        sin_feature = TimeOfDaySinFeature()
        cos_feature = TimeOfDayCosFeature()
        
        # Test continuity across day boundary
        end_of_day = datetime(2025, 1, 25, 23, 59, 59, tzinfo=timezone.utc)
        start_of_day = datetime(2025, 1, 26, 0, 0, 0, tzinfo=timezone.utc)
        
        sin_end = sin_feature.calculate({"timestamp": end_of_day})
        sin_start = sin_feature.calculate({"timestamp": start_of_day})
        
        cos_end = cos_feature.calculate({"timestamp": end_of_day})
        cos_start = cos_feature.calculate({"timestamp": start_of_day})
        
        # Values should be very close (continuous)
        assert abs(sin_end - sin_start) < 0.01
        assert abs(cos_end - cos_start) < 0.01


class TestSessionFeatures:
    """Test market session features"""
    
    def test_market_session_type_feature(self):
        """Test market session type encoding"""
        feature = MarketSessionTypeFeature()
        
        test_cases = [
            # Pre-market (4:00 AM - 9:30 AM ET)
            {
                "timestamp": datetime(2025, 1, 25, 9, 0, 0, tzinfo=timezone.utc),  # 4:00 AM ET
                "market_session": "PREMARKET",
                "expected": 0.25
            },
            {
                "timestamp": datetime(2025, 1, 25, 13, 0, 0, tzinfo=timezone.utc),  # 8:00 AM ET
                "market_session": "PREMARKET",
                "expected": 0.25
            },
            # Regular hours (9:30 AM - 4:00 PM ET)
            {
                "timestamp": datetime(2025, 1, 25, 14, 30, 0, tzinfo=timezone.utc),  # 9:30 AM ET
                "market_session": "REGULAR",
                "expected": 1.0
            },
            {
                "timestamp": datetime(2025, 1, 25, 19, 0, 0, tzinfo=timezone.utc),  # 2:00 PM ET
                "market_session": "REGULAR",
                "expected": 1.0
            },
            # Post-market (4:00 PM - 8:00 PM ET)
            {
                "timestamp": datetime(2025, 1, 25, 21, 0, 0, tzinfo=timezone.utc),  # 4:00 PM ET
                "market_session": "POSTMARKET",
                "expected": 0.75
            },
            {
                "timestamp": datetime(2025, 1, 25, 23, 0, 0, tzinfo=timezone.utc),  # 6:00 PM ET
                "market_session": "POSTMARKET",
                "expected": 0.75
            },
            # Closed/Unknown
            {
                "timestamp": datetime(2025, 1, 25, 2, 0, 0, tzinfo=timezone.utc),  # 9:00 PM ET
                "market_session": "CLOSED",
                "expected": 0.0
            },
            {
                "timestamp": datetime(2025, 1, 25, 3, 0, 0, tzinfo=timezone.utc),
                "market_session": "UNKNOWN",
                "expected": 0.0
            }
        ]
        
        for test_case in test_cases:
            market_data = {
                "timestamp": test_case["timestamp"],
                "market_session": test_case["market_session"]
            }
            result = feature.calculate(market_data)
            
            assert result == test_case["expected"]
    
    def test_session_feature_infers_from_time(self):
        """Test that session feature can infer session from timestamp if not provided"""
        feature = MarketSessionTypeFeature()
        
        # Regular hours timestamp without explicit session
        market_data = {
            "timestamp": datetime(2025, 1, 24, 15, 0, 0, tzinfo=timezone.utc)  # 10:00 AM ET
        }
        
        result = feature.calculate(market_data)
        assert result == 1.0  # Should infer REGULAR session
        
        # Pre-market timestamp
        market_data = {
            "timestamp": datetime(2025, 1, 24, 10, 0, 0, tzinfo=timezone.utc)  # 5:00 AM ET
        }
        
        result = feature.calculate(market_data)
        assert result == 0.25  # Should infer PREMARKET session
    
    def test_session_feature_handles_weekends(self):
        """Test session feature handles weekends properly"""
        feature = MarketSessionTypeFeature()
        
        # Saturday
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc).replace(day=24)  # Saturday
        }
        
        result = feature.calculate(market_data)
        assert result == 0.0  # Market closed on weekends
        
        # Sunday
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc).replace(day=26)  # Sunday
        }
        
        result = feature.calculate(market_data)
        assert result == 0.0  # Market closed on weekends


class TestStaticFeatureIntegration:
    """Test integration of multiple static features"""
    
    def test_all_static_features_together(self):
        """Test calculating all static features together"""
        features = [
            TimeOfDaySinFeature(),
            TimeOfDayCosFeature(),
            MarketSessionTypeFeature()
        ]
        
        # Regular trading hours
        market_data = {
            "timestamp": datetime(2025, 1, 25, 16, 30, 0, tzinfo=timezone.utc),  # 11:30 AM ET
            "market_session": "REGULAR"
        }
        
        results = {}
        for feature in features:
            results[feature.name] = feature.calculate(market_data)
        
        # Check all features return valid values
        assert len(results) == 3
        assert all(isinstance(v, float) for v in results.values())
        
        # Verify expected relationships
        assert -1.0 <= results["time_of_day_sin"] <= 1.0
        assert -1.0 <= results["time_of_day_cos"] <= 1.0
        assert results["market_session_type"] == 1.0
        
        # Check sin^2 + cos^2 = 1
        assert abs(results["time_of_day_sin"]**2 + results["time_of_day_cos"]**2 - 1.0) < 1e-6
    
    def test_static_features_requirements(self):
        """Test that static features declare minimal requirements"""
        features = [
            TimeOfDaySinFeature(),
            TimeOfDayCosFeature(),
            MarketSessionTypeFeature()
        ]
        
        for feature in features:
            requirements = feature.get_requirements()
            
            # Static features should have minimal requirements
            assert "lookback" not in requirements or requirements["lookback"] == 0
            assert "data_type" not in requirements or requirements["data_type"] == "current"
            
            # Should only require timestamp and maybe session
            required_fields = requirements.get("fields", [])
            assert "timestamp" in required_fields
            assert len(required_fields) <= 2  # timestamp and maybe market_session