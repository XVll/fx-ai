"""Tests for medium-frequency features - TDD approach"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# These imports will fail initially (TDD)
from feature.v2.mf.velocity_features import (
    PriceVelocity1mFeature, PriceVelocity5mFeature,
    VolumeVelocity1mFeature, VolumeVelocity5mFeature
)
from feature.v2.mf.ema_features import (
    DistanceToEMA9_1mFeature, DistanceToEMA20_1mFeature,
    DistanceToEMA9_5mFeature, DistanceToEMA20_5mFeature
)
from feature.v2.mf.candle_features import (
    PositionInCurrentCandle1mFeature, PositionInCurrentCandle5mFeature,
    BodySizeRelative1mFeature, BodySizeRelative5mFeature
)


class TestMFVelocityFeatures:
    """Test medium-frequency velocity features"""
    
    def test_price_velocity_1m_feature(self):
        """Test 1-minute price velocity calculation"""
        feature = PriceVelocity1mFeature()
        
        # Create 1-minute bars
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 2, 0, tzinfo=timezone.utc),
            "1m_bars_window": [
                {"timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc),
                 "open": 100.00, "high": 100.50, "low": 99.80, "close": 100.20, "volume": 1000},
                {"timestamp": datetime(2025, 1, 25, 15, 1, 0, tzinfo=timezone.utc),
                 "open": 100.20, "high": 100.60, "low": 100.10, "close": 100.50, "volume": 1500},
                {"timestamp": datetime(2025, 1, 25, 15, 2, 0, tzinfo=timezone.utc),
                 "open": 100.50, "high": 101.00, "low": 100.40, "close": 100.80, "volume": 2000}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Raw velocity = (current_close - previous_close) / previous_close
        # Should be normalized to reasonable range for 1-minute changes
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Normalized to [-1, 1]
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["data_type"] == "1m_bars"
        assert requirements["lookback"] >= 2
        assert "1m_bars_window" in requirements["fields"]
    
    def test_price_velocity_5m_feature(self):
        """Test 5-minute price velocity calculation"""
        feature = PriceVelocity5mFeature()
        
        # Create 5-minute bars
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 10, 0, tzinfo=timezone.utc),
            "5m_bars_window": [
                {"timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc),
                 "close": 100.00},
                {"timestamp": datetime(2025, 1, 25, 15, 5, 0, tzinfo=timezone.utc),
                 "close": 100.50},
                {"timestamp": datetime(2025, 1, 25, 15, 10, 0, tzinfo=timezone.utc),
                 "close": 101.00}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Velocity over 5m
        expected_velocity = (101.00 - 100.50) / 100.50
        assert abs(result - expected_velocity) < 1e-6
    
    def test_volume_velocity_1m_feature(self):
        """Test 1-minute volume velocity calculation"""
        feature = VolumeVelocity1mFeature()
        
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 2, 0, tzinfo=timezone.utc),
            "1m_bars_window": [
                {"volume": 1000},
                {"volume": 1500},
                {"volume": 3000}  # Volume doubled
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Volume velocity = (current_vol - previous_vol) / previous_vol
        expected_velocity = (3000 - 1500) / 1500  # 100% increase
        assert abs(result - expected_velocity) < 1e-6
    
    def test_velocity_edge_cases(self):
        """Test velocity features with edge cases"""
        feature = PriceVelocity1mFeature()
        
        # Test with insufficient data
        market_data = {"1m_bars_window": [{"close": 100.00}]}
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Should use default
        
        # Test with zero previous value
        market_data = {
            "1m_bars_window": [
                {"close": 0.0},
                {"close": 100.00}
            ]
        }
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Should handle division by zero
        
        # Test with missing bars window
        result = feature.calculate({})
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
        
        # Test with None/NaN values
        market_data = {
            "1m_bars_window": [
                {"close": None},
                {"close": np.nan},
                {"close": 100.00}
            ]
        }
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0


class TestMFEMAFeatures:
    """Test EMA distance features"""
    
    def test_distance_to_ema9_1m_feature(self):
        """Test distance to 9-period EMA on 1-minute bars"""
        feature = DistanceToEMA9_1mFeature()
        
        # Create bars with known pattern
        closes = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 
                  104.5, 105.0]  # 11 bars, steady uptrend
        
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 10, 0, tzinfo=timezone.utc),
            "current_price": 105.0,
            "1m_bars_window": [
                {"close": close, "timestamp": datetime(2025, 1, 25, 15, i, 0, tzinfo=timezone.utc)}
                for i, close in enumerate(closes)
            ]
        }
        
        result = feature.calculate(market_data)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        # Distance normalized to reasonable range (e.g., -0.5 to 0.5 for ±50% distance)
        assert -1.0 <= result <= 1.0
        assert result > 0  # Price above EMA in uptrend
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["lookback"] >= 9
    
    def test_distance_to_ema20_1m_feature(self):
        """Test distance to 20-period EMA on 1-minute bars"""
        feature = DistanceToEMA20_1mFeature()
        
        # Create 25 bars
        closes = [100.0 + i * 0.1 for i in range(25)]
        
        market_data = {
            "current_price": closes[-1],
            "1m_bars_window": [{"close": close} for close in closes]
        }
        
        result = feature.calculate(market_data)
        
        # In steady uptrend, price should be above EMA20
        assert result > 0
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["lookback"] >= 20
    
    def test_ema_with_insufficient_data(self):
        """Test EMA features with insufficient data"""
        feature = DistanceToEMA9_1mFeature()
        
        # Only 5 bars when we need 9
        market_data = {
            "current_price": 100.0,
            "1m_bars_window": [{"close": 100.0} for _ in range(5)]
        }
        
        result = feature.calculate(market_data)
        
        # Should use available data or return default
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Should be bounded
        
        # Test with no data
        result = feature.calculate({})
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
        
        # Test with empty bars window
        result = feature.calculate({"current_price": 100.0, "1m_bars_window": []})
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
    
    def test_ema_distance_negative(self):
        """Test EMA distance when price is below EMA"""
        feature = DistanceToEMA9_1mFeature()
        
        # Downtrend pattern
        closes = [100.0 - i * 0.5 for i in range(10)]
        
        market_data = {
            "current_price": closes[-1],
            "1m_bars_window": [{"close": close} for close in closes]
        }
        
        result = feature.calculate(market_data)
        
        # In downtrend, price should be below EMA
        assert result < 0
    
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Helper to calculate EMA"""
        if not values:
            return 0.0
        
        # Simple EMA calculation
        multiplier = 2 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return ema


class TestMFCandleFeatures:
    """Test candle pattern features"""
    
    def test_position_in_current_candle_1m_feature(self):
        """Test position of current price within current 1m candle"""
        feature = PositionInCurrentCandle1mFeature()
        
        # Current candle with known range
        market_data = {
            "current_price": 100.75,
            "1m_bars_window": [
                {"open": 100.00, "high": 101.00, "low": 100.00, "close": 100.50,
                 "timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc)},
                {"open": 100.50, "high": 101.00, "low": 100.50, "close": 100.75,
                 "timestamp": datetime(2025, 1, 25, 15, 1, 0, tzinfo=timezone.utc),
                 "is_current": True}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Position = (current - low) / (high - low)
        # = (100.75 - 100.50) / (101.00 - 100.50) = 0.25 / 0.50 = 0.5
        assert abs(result - 0.5) < 1e-6
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0  # Already normalized to [0, 1]
    
    def test_position_in_current_candle_edge_cases(self):
        """Test candle position edge cases"""
        feature = PositionInCurrentCandle1mFeature()
        
        # Price at high
        market_data = {
            "current_price": 101.00,
            "1m_bars_window": [
                {"high": 101.00, "low": 100.00, "is_current": True}
            ]
        }
        result = feature.calculate(market_data)
        assert result == 1.0
        assert not np.isnan(result)
        
        # Price at low
        market_data["current_price"] = 100.00
        result = feature.calculate(market_data)
        assert result == 0.0
        assert not np.isnan(result)
        
        # No range (high == low)
        market_data["1m_bars_window"][0] = {"high": 100.00, "low": 100.00, "is_current": True}
        market_data["current_price"] = 100.00
        result = feature.calculate(market_data)
        assert result == 0.5  # Default to middle
        assert not np.isnan(result)
        
        # Price outside range (gap)
        market_data = {
            "current_price": 102.00,
            "1m_bars_window": [{"high": 101.00, "low": 100.00, "is_current": True}]
        }
        result = feature.calculate(market_data)
        assert result == 1.0  # Should be clamped
        assert not np.isnan(result)
        
        # Missing current price
        result = feature.calculate({"1m_bars_window": [{"high": 101.00, "low": 100.00}]})
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
        
        # No bars
        result = feature.calculate({})
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
    
    def test_body_size_relative_1m_feature(self):
        """Test relative body size of candles"""
        feature = BodySizeRelative1mFeature()
        
        market_data = {
            "1m_bars_window": [
                {"open": 100.00, "high": 101.00, "low": 99.50, "close": 100.80},  # Body: 0.80, Range: 1.50
                {"open": 100.80, "high": 101.20, "low": 100.60, "close": 101.00}, # Body: 0.20, Range: 0.60
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Body size relative = abs(close - open) / (high - low)
        # For last candle: 0.20 / 0.60 = 0.333...
        expected = abs(101.00 - 100.80) / (101.20 - 100.60)
        assert abs(result - expected) < 1e-6
    
    def test_candle_features_with_doji(self):
        """Test candle features with doji (open == close)"""
        feature = BodySizeRelative1mFeature()
        
        market_data = {
            "1m_bars_window": [
                {"open": 100.00, "high": 100.50, "low": 99.50, "close": 100.00}  # Doji
            ]
        }
        
        result = feature.calculate(market_data)
        assert result == 0.0  # No body


class TestMFFeatureIntegration:
    """Test integration of multiple MF features"""
    
    def test_multiple_mf_features_together(self):
        """Test calculating multiple MF features from same data"""
        features = [
            PriceVelocity1mFeature(),
            DistanceToEMA9_1mFeature(),
            PositionInCurrentCandle1mFeature()
        ]
        
        # Create rich market data
        bars = []
        for i in range(15):
            bar = {
                "timestamp": datetime(2025, 1, 25, 15, i, 0, tzinfo=timezone.utc),
                "open": 100.0 + i * 0.1,
                "high": 100.0 + i * 0.1 + 0.3,
                "low": 100.0 + i * 0.1 - 0.1,
                "close": 100.0 + i * 0.1 + 0.2,
                "volume": 1000 + i * 100
            }
            bars.append(bar)
        
        # Mark last bar as current
        bars[-1]["is_current"] = True
        
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 14, 30, tzinfo=timezone.utc),
            "current_price": 101.5,
            "1m_bars_window": bars
        }
        
        results = {}
        for feature in features:
            results[feature.name] = feature.calculate(market_data)
        
        # Check all features return valid values
        assert len(results) == 3
        assert all(isinstance(v, float) for v in results.values())
        assert all(not np.isnan(v) for v in results.values())
        assert all(np.isfinite(v) for v in results.values())
        
        # Check normalized ranges
        assert -1.0 <= results["price_velocity_1m"] <= 1.0
        assert -1.0 <= results["distance_to_ema9_1m"] <= 1.0
        assert 0.0 <= results["position_in_current_candle_1m"] <= 1.0
    
    def test_mf_features_final_output(self):
        """Test MF features produce correct output format"""
        # Create 30 minutes of 1m bars
        bars_1m = []
        base_price = 100.0
        
        for i in range(30):
            # Add some realistic price movement
            noise = np.random.normal(0, 0.1)
            trend = i * 0.01  # Slight uptrend
            
            open_price = base_price + trend
            close_price = open_price + noise
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.05))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.05))
            
            bars_1m.append({
                "timestamp": datetime(2025, 1, 25, 15, i, 0, tzinfo=timezone.utc),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": np.random.randint(1000, 5000)
            })
            
            base_price = close_price
        
        # Create 5m bars from 1m bars
        bars_5m = []
        for i in range(0, 30, 5):
            chunk = bars_1m[i:i+5]
            if chunk:
                bars_5m.append({
                    "timestamp": chunk[0]["timestamp"],
                    "open": chunk[0]["open"],
                    "high": max(b["high"] for b in chunk),
                    "low": min(b["low"] for b in chunk),
                    "close": chunk[-1]["close"],
                    "volume": sum(b["volume"] for b in chunk)
                })
        
        # Test feature extraction
        features_1m = [PriceVelocity1mFeature(), DistanceToEMA9_1mFeature()]
        features_5m = [PriceVelocity5mFeature(), DistanceToEMA9_5mFeature()]
        
        # Extract features for sequences
        feature_matrix_1m = np.zeros((30, len(features_1m)), dtype=np.float32)
        feature_matrix_5m = np.zeros((6, len(features_5m)), dtype=np.float32)
        
        # 1-minute features
        for t in range(30):
            market_data = {
                "timestamp": bars_1m[t]["timestamp"],
                "current_price": bars_1m[t]["close"],
                "1m_bars_window": bars_1m[max(0, t-20):t+1]  # Use last 20 bars
            }
            
            for f_idx, feature in enumerate(features_1m):
                value = feature.calculate(market_data)
                feature_matrix_1m[t, f_idx] = value
        
        # 5-minute features
        for t in range(6):
            market_data = {
                "timestamp": bars_5m[t]["timestamp"],
                "current_price": bars_5m[t]["close"],
                "5m_bars_window": bars_5m[max(0, t-10):t+1]
            }
            
            for f_idx, feature in enumerate(features_5m):
                value = feature.calculate(market_data)
                feature_matrix_5m[t, f_idx] = value
        
        # Verify output properties
        assert feature_matrix_1m.shape == (30, 2)
        assert feature_matrix_5m.shape == (6, 2)
        
        for matrix in [feature_matrix_1m, feature_matrix_5m]:
            assert matrix.dtype == np.float32
            assert not np.any(np.isnan(matrix))
            assert np.all(np.isfinite(matrix))
            assert np.all(matrix >= -1.0)
            assert np.all(matrix <= 1.0)
    
    def test_mf_timeframe_consistency(self):
        """Test that 1m and 5m features are consistent"""
        # Create features for both timeframes
        features_1m = PriceVelocity1mFeature()
        features_5m = PriceVelocity5mFeature()
        
        # Create data where 5m velocity should be roughly average of 5x 1m velocities
        market_data = {
            "1m_bars_window": [
                {"close": 100.0 + i * 0.1} for i in range(6)
            ],
            "5m_bars_window": [
                {"close": 100.0},  # Average of first 5 1m bars
                {"close": 100.5}   # Current
            ]
        }
        
        result_1m = features_1m.calculate(market_data)
        result_5m = features_5m.calculate(market_data)
        
        # 5m velocity should be larger timeframe view
        assert isinstance(result_1m, float)
        assert isinstance(result_5m, float)
        assert not np.isnan(result_1m)
        assert not np.isnan(result_5m)
        assert -1.0 <= result_1m <= 1.0
        assert -1.0 <= result_5m <= 1.0