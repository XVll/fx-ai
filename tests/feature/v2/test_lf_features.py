"""Tests for low-frequency features - TDD approach"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# These imports will fail initially (TDD)
from feature.v2.lf.range_features import (
    PositionInDailyRangeFeature, PositionInPrevDayRangeFeature,
    PriceChangeFromPrevCloseFeature
)
from feature.v2.lf.level_features import (
    DistanceToClosestSupportFeature, DistanceToClosestResistanceFeature,
    WholeDollarProximityFeature, HalfDollarProximityFeature
)


class TestLFRangeFeatures:
    """Test low-frequency range-based features"""
    
    def test_position_in_daily_range_feature(self):
        """Test position of current price within today's range"""
        feature = PositionInDailyRangeFeature()
        
        market_data = {
            "current_price": 100.75,
            "intraday_high": 101.00,
            "intraday_low": 100.00,
            "timestamp": datetime(2025, 1, 25, 15, 30, 0, tzinfo=timezone.utc)
        }
        
        result = feature.calculate(market_data)
        
        # Position = (current - low) / (high - low)
        # = (100.75 - 100.00) / (101.00 - 100.00) = 0.75
        assert abs(result - 0.75) < 1e-6
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0  # Already normalized to [0, 1]
        
        # Test requirements
        requirements = feature.get_requirements()
        assert "intraday_high" in requirements["fields"]
        assert "intraday_low" in requirements["fields"]
    
    def test_position_in_daily_range_edge_cases(self):
        """Test daily range position edge cases"""
        feature = PositionInDailyRangeFeature()
        
        # Price at day's high
        market_data = {
            "current_price": 101.00,
            "intraday_high": 101.00,
            "intraday_low": 100.00
        }
        result = feature.calculate(market_data)
        assert result == 1.0
        assert not np.isnan(result)
        
        # Price at day's low
        market_data["current_price"] = 100.00
        result = feature.calculate(market_data)
        assert result == 0.0
        assert not np.isnan(result)
        
        # No range (high == low)
        market_data = {
            "current_price": 100.00,
            "intraday_high": 100.00,
            "intraday_low": 100.00
        }
        result = feature.calculate(market_data)
        assert result == 0.5  # Default to middle
        assert not np.isnan(result)
        
        # Price outside range (gap up) - should be clamped
        market_data = {
            "current_price": 102.00,
            "intraday_high": 101.00,
            "intraday_low": 100.00
        }
        result = feature.calculate(market_data)
        assert result == 1.0  # Clamped to max
        assert not np.isnan(result)
        
        # Price below range (gap down) - should be clamped
        market_data = {
            "current_price": 99.00,
            "intraday_high": 101.00,
            "intraday_low": 100.00
        }
        result = feature.calculate(market_data)
        assert result == 0.0  # Clamped to min
        assert not np.isnan(result)
        
        # Missing intraday data
        result = feature.calculate({"current_price": 100.00})
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
        
        # All missing data
        result = feature.calculate({})
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
    
    def test_position_in_prev_day_range_feature(self):
        """Test position relative to previous day's range"""
        feature = PositionInPrevDayRangeFeature()
        
        market_data = {
            "current_price": 101.50,
            "previous_day_data": {
                "high": 102.00,
                "low": 100.00,
                "close": 101.00
            }
        }
        
        result = feature.calculate(market_data)
        
        # Position = (current - prev_low) / (prev_high - prev_low)
        # = (101.50 - 100.00) / (102.00 - 100.00) = 0.75
        assert abs(result - 0.75) < 1e-6
    
    def test_price_change_from_prev_close_feature(self):
        """Test percentage change from previous close"""
        feature = PriceChangeFromPrevCloseFeature()
        
        market_data = {
            "current_price": 102.50,
            "previous_day_data": {
                "close": 100.00
            }
        }
        
        result = feature.calculate(market_data)
        
        # Raw change = (current - prev_close) / prev_close = 2.5%
        # Should be normalized to reasonable range (e.g., [-1, 1] for ±20% daily moves)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Normalized
        assert result > 0  # Positive change
    
    def test_prev_day_features_missing_data(self):
        """Test previous day features with missing data"""
        feature = PositionInPrevDayRangeFeature()
        
        # No previous day data
        market_data = {
            "current_price": 100.00,
            "previous_day_data": None
        }
        
        result = feature.calculate(market_data)
        assert result == 0.5  # Default to neutral
        assert not np.isnan(result)
        
        # Empty previous day data
        market_data["previous_day_data"] = {}
        result = feature.calculate(market_data)
        assert result == 0.5  # Default to neutral
        assert not np.isnan(result)
        
        # Previous close is zero
        change_feature = PriceChangeFromPrevCloseFeature()
        market_data = {
            "current_price": 100.00,
            "previous_day_data": {"close": 0.0}
        }
        result = change_feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Should handle division by zero


class TestLFLevelFeatures:
    """Test support/resistance and price level features"""
    
    def test_distance_to_closest_support_feature(self):
        """Test distance to nearest support level"""
        feature = DistanceToClosestSupportFeature()
        
        market_data = {
            "current_price": 100.50,
            "support_levels": [98.00, 99.50, 100.00, 101.00],  # Historical support levels
            "daily_bars_window": [
                {"low": 98.00},
                {"low": 99.50},
                {"low": 100.00}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Raw distance = (current - support) / current
        # Should be normalized to reasonable range
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0  # Normalized (0 = at support, 1 = far from support)
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["data_type"] == "daily_bars"
        assert requirements["lookback"] >= 20  # Need history for support levels
    
    def test_distance_to_closest_resistance_feature(self):
        """Test distance to nearest resistance level"""
        feature = DistanceToClosestResistanceFeature()
        
        market_data = {
            "current_price": 100.50,
            "resistance_levels": [99.00, 100.00, 101.00, 102.00],  # Historical resistance
            "daily_bars_window": [
                {"high": 102.00},
                {"high": 101.00},
                {"high": 100.00}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Closest resistance above current price is 101.00
        # Distance = (resistance - current) / current
        # = (101.00 - 100.50) / 100.50 = 0.00497...
        expected_distance = (101.00 - 100.50) / 100.50
        assert abs(result - expected_distance) < 1e-6
    
    def test_support_resistance_auto_detection(self):
        """Test automatic support/resistance detection from price history"""
        feature = DistanceToClosestSupportFeature()
        
        # No explicit levels, should detect from daily bars
        market_data = {
            "current_price": 100.50,
            "daily_bars_window": [
                {"high": 99.00, "low": 98.00, "close": 98.50},
                {"high": 99.50, "low": 98.50, "close": 99.00},
                {"high": 100.00, "low": 99.00, "close": 99.50},  # 99.00 tested twice as low
                {"high": 100.50, "low": 99.00, "close": 100.00}, # 99.00 confirmed support
                {"high": 101.00, "low": 99.50, "close": 100.50}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Should detect support and return normalized distance
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
        assert result > 0  # Not at support
        
        # Test with no daily bars - should use default
        result = feature.calculate({"current_price": 100.00})
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
    
    def test_whole_dollar_proximity_feature(self):
        """Test proximity to whole dollar levels"""
        feature = WholeDollarProximityFeature()
        
        test_cases = [
            {"current_price": 100.05, "expected_cents": 0.05},   # 5 cents from $100
            {"current_price": 99.95, "expected_cents": 0.05},    # 5 cents from $100
            {"current_price": 100.50, "expected_cents": 0.50},   # 50 cents from both $100 and $101
            {"current_price": 100.00, "expected_cents": 0.00},   # Exactly at whole dollar
            {"current_price": 150.10, "expected_cents": 0.10},   # Works at higher prices
        ]
        
        for case in test_cases:
            market_data = {"current_price": case["current_price"]}
            result = feature.calculate(market_data)
            
            # Should be normalized to [0, 1] where 0 = at level, 1 = max distance (50 cents)
            assert not np.isnan(result)
            assert 0.0 <= result <= 1.0
            
            # Verify relative ordering
            if case["expected_cents"] == 0.00:
                assert result == 0.0
            elif case["expected_cents"] == 0.50:
                assert result == 1.0  # Max distance
    
    def test_half_dollar_proximity_feature(self):
        """Test proximity to half dollar levels"""
        feature = HalfDollarProximityFeature()
        
        test_cases = [
            {"current_price": 100.55, "expected_cents": 0.05},   # 5 cents from $100.50
            {"current_price": 100.45, "expected_cents": 0.05},   # 5 cents from $100.50
            {"current_price": 100.25, "expected_cents": 0.25},   # 25 cents from both $100.00 and $100.50
            {"current_price": 100.50, "expected_cents": 0.00},   # Exactly at half dollar
            {"current_price": 100.00, "expected_cents": 0.00},   # Whole dollars are also half dollars
        ]
        
        for case in test_cases:
            market_data = {"current_price": case["current_price"]}
            result = feature.calculate(market_data)
            
            # Should be normalized to [0, 1] where 0 = at level, 1 = max distance (25 cents)
            assert not np.isnan(result)
            assert 0.0 <= result <= 1.0
            
            # Verify relative ordering
            if case["expected_cents"] == 0.00:
                assert result == 0.0
            elif case["expected_cents"] == 0.25:
                assert result == 1.0  # Max distance for half dollar
    
    def test_price_level_features_low_prices(self):
        """Test price level features work correctly with low-priced stocks"""
        whole_feature = WholeDollarProximityFeature()
        half_feature = HalfDollarProximityFeature()
        
        # Low price stock
        market_data = {"current_price": 2.47}
        
        whole_result = whole_feature.calculate(market_data)
        half_result = half_feature.calculate(market_data)
        
        # Both should return valid normalized values
        assert not np.isnan(whole_result)
        assert not np.isnan(half_result)
        assert 0.0 <= whole_result <= 1.0
        assert 0.0 <= half_result <= 1.0
        
        # Should measure distance to $2.00/$3.00 for whole and $2.50 for half
        assert whole_result > 0  # Not at whole dollar
        assert half_result > 0   # Not at half dollar
        
        # Half dollar should be closer (3 cents vs 47 cents)
        assert half_result < whole_result
        
        # Test with missing price
        result = whole_feature.calculate({})
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0


class TestLFFeatureIntegration:
    """Test integration of multiple LF features"""
    
    def test_all_lf_features_together(self):
        """Test calculating all LF features from same data"""
        features = [
            PositionInDailyRangeFeature(),
            PositionInPrevDayRangeFeature(),
            PriceChangeFromPrevCloseFeature(),
            DistanceToClosestSupportFeature(),
            DistanceToClosestResistanceFeature(),
            WholeDollarProximityFeature(),
            HalfDollarProximityFeature()
        ]
        
        # Rich market data
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 30, 0, tzinfo=timezone.utc),
            "current_price": 100.47,
            "intraday_high": 101.50,
            "intraday_low": 99.50,
            "previous_day_data": {
                "high": 100.00,
                "low": 98.00,
                "close": 99.00
            },
            "support_levels": [98.00, 99.00, 99.50],
            "resistance_levels": [100.50, 101.00, 102.00],
            "daily_bars_window": [
                {"high": 100.00, "low": 98.00} for _ in range(20)
            ]
        }
        
        results = {}
        for feature in features:
            results[feature.name] = feature.calculate(market_data)
        
        # Check all features return valid values
        assert len(results) == 7
        assert all(isinstance(v, float) for v in results.values())
        assert all(not np.isnan(v) for v in results.values())
        assert all(np.isfinite(v) for v in results.values())
        
        # Check normalized ranges
        assert 0.0 <= results["position_in_daily_range"] <= 1.0
        assert 0.0 <= results["position_in_prev_day_range"] <= 1.0
        assert -1.0 <= results["price_change_from_prev_close"] <= 1.0
        assert 0.0 <= results["distance_to_closest_support"] <= 1.0
        assert 0.0 <= results["distance_to_closest_resistance"] <= 1.0
        assert 0.0 <= results["whole_dollar_proximity"] <= 1.0
        assert 0.0 <= results["half_dollar_proximity"] <= 1.0
    
    def test_lf_features_final_output(self):
        """Test LF features produce correct output format"""
        # Create 10 days of daily bars
        daily_bars = []
        base_price = 100.0
        
        for i in range(10):
            # Add some realistic daily movement
            open_price = base_price + np.random.uniform(-1, 1)
            close_price = open_price + np.random.uniform(-2, 2)
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, 1))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, 1))
            
            daily_bars.append({
                "date": datetime(2025, 1, 15 + i, 16, 0, 0, tzinfo=timezone.utc),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": np.random.randint(1000000, 5000000)
            })
            
            base_price = close_price
        
        # Create feature extractors
        features = [
            PositionInDailyRangeFeature(),
            PriceChangeFromPrevCloseFeature(),
            DistanceToClosestSupportFeature()
        ]
        
        # Extract features for each day
        feature_matrix = np.zeros((10, len(features)), dtype=np.float32)
        
        for t in range(10):
            # Simulate intraday data
            current_bar = daily_bars[t]
            intraday_price = current_bar["close"]  # End of day
            
            market_data = {
                "timestamp": current_bar["date"],
                "current_price": intraday_price,
                "intraday_high": current_bar["high"],
                "intraday_low": current_bar["low"],
                "previous_day_data": daily_bars[t-1] if t > 0 else None,
                "daily_bars_window": daily_bars[max(0, t-20):t]  # Last 20 days
            }
            
            for f_idx, feature in enumerate(features):
                value = feature.calculate(market_data)
                feature_matrix[t, f_idx] = value
        
        # Verify output properties
        assert feature_matrix.shape == (10, 3)
        assert feature_matrix.dtype == np.float32
        assert not np.any(np.isnan(feature_matrix))
        assert np.all(np.isfinite(feature_matrix))
        
        # All values should be in normalized ranges
        assert np.all(feature_matrix[:, 0] >= 0.0)  # Position in range
        assert np.all(feature_matrix[:, 0] <= 1.0)
        assert np.all(feature_matrix[:, 1] >= -1.0)  # Price change
        assert np.all(feature_matrix[:, 1] <= 1.0)
        assert np.all(feature_matrix[:, 2] >= 0.0)  # Distance to support
        assert np.all(feature_matrix[:, 2] <= 1.0)
    
    def test_lf_features_data_requirements(self):
        """Test that LF features properly declare their data needs"""
        features = [
            PositionInDailyRangeFeature(),
            DistanceToClosestSupportFeature(),
            WholeDollarProximityFeature()
        ]
        
        # Collect all requirements
        all_fields = set()
        max_lookback = 0
        
        for feature in features:
            req = feature.get_requirements()
            if "fields" in req:
                all_fields.update(req["fields"])
            if "lookback" in req:
                max_lookback = max(max_lookback, req["lookback"])
        
        # Should require various data
        assert "current_price" in all_fields
        assert "intraday_high" in all_fields or "daily_bars" in req.get("data_type", "")
        assert max_lookback >= 20  # For support/resistance detection