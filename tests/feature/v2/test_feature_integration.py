"""Integration tests for complete feature system - TDD approach"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# These imports will fail initially (TDD)
from feature.v2.base.feature_manager import FeatureManager
from feature.v2.base.feature_pipeline import FeaturePipeline


class TestFeatureSystemIntegration:
    """Test complete feature system integration"""
    
    def test_complete_feature_extraction_pipeline(self):
        """Test extracting all features for model input"""
        # Configure feature pipeline
        config = {
            "features": {
                "static": ["time_of_day_sin", "time_of_day_cos", "market_session_type"],
                "hf": ["price_velocity", "price_acceleration", "tape_imbalance", "tape_aggression_ratio"],
                "mf": ["1m_price_velocity", "1m_ema9_distance", "5m_price_velocity", "5m_ema9_distance"],
                "lf": ["daily_range_position", "price_change_from_prev_close", "support_distance", "whole_dollar_proximity"]
            },
            "sequence_lengths": {
                "hf": 60,   # 60 seconds
                "mf": 30,   # 30 bars (1m) or 6 bars (5m)
                "lf": 10    # 10 days
            },
            "normalization": {
                "enabled": True,
                "method": "minmax"
            }
        }
        
        pipeline = FeaturePipeline(config)
        
        # Create realistic market data
        market_data = self._create_complete_market_data()
        
        # Extract all features
        features = pipeline.extract_all_features(market_data)
        
        # Verify output structure
        assert "static" in features
        assert "hf" in features
        assert "mf" in features
        assert "lf" in features
        
        # Check shapes match configuration
        assert features["static"].shape == (1, 3)      # 1 timestep, 3 features
        assert features["hf"].shape == (60, 4)         # 60 timesteps, 4 features
        assert features["mf"].shape == (30, 4)         # 30 timesteps, 4 features
        assert features["lf"].shape == (10, 4)         # 10 timesteps, 4 features
        
        # Verify no NaN values
        for branch, array in features.items():
            assert not np.any(np.isnan(array)), f"NaN found in {branch} features"
            assert np.all(np.isfinite(array)), f"Non-finite values in {branch} features"
        
        # Verify normalization
        for branch, array in features.items():
            if branch in ["static"]:  # Sin/cos can be [-1, 1]
                assert np.all(array >= -1.0) and np.all(array <= 1.0)
            else:
                # Most features normalized to [0, 1] or [-1, 1]
                assert np.all(array >= -1.0) and np.all(array <= 1.0)
    
    def test_sparse_data_handling(self):
        """Test feature extraction with sparse/missing data"""
        pipeline = FeaturePipeline({
            "features": {
                "static": ["market_session_type"],
                "hf": ["tape_imbalance"],
                "mf": ["1m_price_velocity"],
                "lf": ["daily_range_position"]
            }
        })
        
        # Minimal data - many missing elements
        market_data = {
            "timestamp": datetime.now(timezone.utc),
            "current_price": 100.0,
            # No market session
            # No trades
            # No bars
            # No intraday data
        }
        
        features = pipeline.extract_all_features(market_data)
        
        # Should handle gracefully with defaults
        for branch, array in features.items():
            assert not np.any(np.isnan(array)), f"NaN in {branch} with sparse data"
            assert array.size > 0, f"Empty array for {branch}"
    
    def test_extreme_market_conditions(self):
        """Test features under extreme market conditions"""
        pipeline = FeaturePipeline({
            "features": {
                "hf": ["price_velocity", "tape_imbalance"],
                "mf": ["1m_price_velocity"],
                "lf": ["price_change_from_prev_close"]
            }
        })
        
        # Test cases for extreme conditions
        test_cases = [
            # Flash crash scenario
            {
                "name": "flash_crash",
                "current_price": 50.0,
                "previous_price": 100.0,  # 50% drop
                "expected_normalized": True
            },
            # Circuit breaker scenario
            {
                "name": "circuit_breaker",
                "current_price": 110.0,
                "previous_price": 100.0,  # 10% up
                "expected_normalized": True
            },
            # Penny stock scenario
            {
                "name": "penny_stock",
                "current_price": 0.01,
                "previous_price": 0.02,  # 50% drop but tiny absolute
                "expected_normalized": True
            },
            # Zero price (halted)
            {
                "name": "halted",
                "current_price": 0.0,
                "previous_price": 100.0,
                "expected_normalized": True
            }
        ]
        
        for case in test_cases:
            market_data = self._create_extreme_scenario_data(
                case["current_price"],
                case["previous_price"]
            )
            
            features = pipeline.extract_all_features(market_data)
            
            # All features should be normalized and valid
            for branch, array in features.items():
                assert not np.any(np.isnan(array)), f"NaN in {case['name']} scenario"
                assert np.all(np.isfinite(array)), f"Infinite values in {case['name']}"
                assert np.all(array >= -1.0) and np.all(array <= 1.0), \
                    f"Values out of range in {case['name']}"
    
    def test_feature_consistency_across_timeframes(self):
        """Test that features are consistent across different timeframes"""
        pipeline = FeaturePipeline({
            "features": {
                "hf": ["price_velocity"],
                "mf": ["1m_price_velocity", "5m_price_velocity"],
                "lf": ["price_change_from_prev_close"]
            }
        })
        
        # Create data with consistent trend
        market_data = self._create_trending_market_data(trend="up")
        
        features = pipeline.extract_all_features(market_data)
        
        # All velocity features should show same direction
        hf_velocities = features["hf"][:, 0]
        mf_1m_velocities = features["mf"][:, 0]
        mf_5m_velocities = features["mf"][:, 1]
        lf_changes = features["lf"][:, 0]
        
        # In uptrend, most velocities should be positive
        assert np.mean(hf_velocities > 0) > 0.6
        assert np.mean(mf_1m_velocities > 0) > 0.6
        assert np.mean(mf_5m_velocities > 0) > 0.6
        assert np.mean(lf_changes > 0) > 0.6
    
    def test_feature_caching_performance(self):
        """Test that feature caching improves performance"""
        pipeline = FeaturePipeline({
            "features": {
                "static": ["time_of_day_sin", "time_of_day_cos"],
                "hf": ["price_velocity", "tape_imbalance"]
            },
            "caching": {
                "enabled": True,
                "ttl_seconds": 1
            }
        })
        
        market_data = self._create_complete_market_data()
        
        # First extraction - should calculate
        import time
        start = time.time()
        features1 = pipeline.extract_all_features(market_data)
        time1 = time.time() - start
        
        # Second extraction with same data - should use cache
        start = time.time()
        features2 = pipeline.extract_all_features(market_data)
        time2 = time.time() - start
        
        # Cache should be faster
        assert time2 < time1 * 0.5  # At least 2x faster
        
        # Results should be identical
        for branch in features1:
            assert np.array_equal(features1[branch], features2[branch])
    
    def test_feature_data_requirements(self):
        """Test that feature system correctly reports data requirements"""
        manager = FeatureManager({
            "features": {
                "hf": ["price_velocity", "price_acceleration", "tape_imbalance"],
                "mf": ["1m_ema9_distance", "5m_ema20_distance"],
                "lf": ["support_distance"]
            }
        })
        
        requirements = manager.get_aggregated_requirements()
        
        # Should require various data types
        assert "1s_bars" in requirements
        assert "1m_bars" in requirements
        assert "5m_bars" in requirements
        assert "trades" in requirements
        assert "daily_bars" in requirements
        
        # Check lookback requirements
        assert requirements["1s_bars"]["lookback"] >= 3    # For acceleration
        assert requirements["1m_bars"]["lookback"] >= 9    # For EMA9
        assert requirements["5m_bars"]["lookback"] >= 20  # For EMA20
        assert requirements["daily_bars"]["lookback"] >= 20  # For support detection
    
    def test_final_model_input_format(self):
        """Test the final format ready for model consumption"""
        pipeline = FeaturePipeline({
            "features": {
                "static": ["time_of_day_sin", "time_of_day_cos", "market_session_type"],
                "hf": ["price_velocity", "tape_imbalance"],
                "mf": ["1m_price_velocity", "1m_ema9_distance"],
                "lf": ["daily_range_position", "support_distance"]
            },
            "sequence_lengths": {
                "hf": 60,
                "mf": 30,
                "lf": 10
            },
            "output_format": "dict"  # or "concatenated"
        })
        
        market_data = self._create_complete_market_data()
        features = pipeline.extract_all_features(market_data)
        
        # Test dictionary format
        assert isinstance(features, dict)
        assert set(features.keys()) == {"static", "hf", "mf", "lf"}
        
        # Test concatenated format
        pipeline.config["output_format"] = "concatenated"
        features_concat = pipeline.extract_all_features(market_data)
        
        # Should be a single array with all features
        expected_size = (
            1 * 3 +     # static: 1 timestep * 3 features
            60 * 2 +    # hf: 60 timesteps * 2 features
            30 * 2 +    # mf: 30 timesteps * 2 features
            10 * 2      # lf: 10 timesteps * 2 features
        )
        assert features_concat.shape == (expected_size,)
        assert not np.any(np.isnan(features_concat))
    
    def _create_complete_market_data(self) -> Dict[str, Any]:
        """Create realistic complete market data for testing"""
        current_time = datetime(2025, 1, 25, 15, 30, 0, tzinfo=timezone.utc)
        
        # Create HF data (1-second)
        hf_window = []
        for i in range(60):
            timestamp = current_time - timedelta(seconds=60-i)
            trades = []
            if np.random.random() > 0.3:  # 70% chance of trades
                for _ in range(np.random.randint(1, 5)):
                    trades.append({
                        "price": 100.0 + np.random.uniform(-0.1, 0.1),
                        "size": np.random.randint(100, 1000),
                        "conditions": ["BUY"] if np.random.random() > 0.5 else ["SELL"]
                    })
            
            hf_window.append({
                "timestamp": timestamp,
                "1s_bar": {
                    "close": 100.0 + np.random.uniform(-0.1, 0.1)
                } if trades else None,
                "trades": trades,
                "quotes": [{
                    "bid_price": 99.95 + np.random.uniform(0, 0.05),
                    "ask_price": 100.05 + np.random.uniform(0, 0.05)
                }]
            })
        
        # Create MF data (1-minute bars)
        bars_1m = []
        for i in range(30):
            timestamp = current_time - timedelta(minutes=30-i)
            base = 100.0 + i * 0.01
            bars_1m.append({
                "timestamp": timestamp,
                "open": base,
                "high": base + np.random.uniform(0, 0.2),
                "low": base - np.random.uniform(0, 0.1),
                "close": base + np.random.uniform(-0.05, 0.15),
                "volume": np.random.randint(1000, 5000)
            })
        
        # Create LF data (daily bars)
        daily_bars = []
        for i in range(20):
            date = current_time - timedelta(days=20-i)
            base = 95.0 + i * 0.25
            daily_bars.append({
                "timestamp": date,
                "open": base,
                "high": base + np.random.uniform(0, 2),
                "low": base - np.random.uniform(0, 1),
                "close": base + np.random.uniform(-0.5, 1.5),
                "volume": np.random.randint(1000000, 5000000)
            })
        
        return {
            "timestamp": current_time,
            "current_price": 100.5,
            "market_session": "REGULAR",
            "intraday_high": 101.5,
            "intraday_low": 99.5,
            "previous_day_data": {
                "open": 99.0,
                "high": 100.0,
                "low": 98.0,
                "close": 99.5
            },
            "hf_data_window": hf_window,
            "1m_bars_window": bars_1m,
            "5m_bars_window": bars_1m[::5],  # Every 5th bar
            "daily_bars_window": daily_bars
        }
    
    def _create_extreme_scenario_data(self, current_price: float, 
                                     previous_price: float) -> Dict[str, Any]:
        """Create market data for extreme scenarios"""
        data = self._create_complete_market_data()
        
        # Update prices
        data["current_price"] = current_price
        data["previous_day_data"]["close"] = previous_price
        
        # Update recent bars to reflect the extreme move
        if data["hf_data_window"]:
            for entry in data["hf_data_window"][-10:]:
                if entry["1s_bar"]:
                    entry["1s_bar"]["close"] = current_price
        
        return data
    
    def _create_trending_market_data(self, trend: str = "up") -> Dict[str, Any]:
        """Create market data with consistent trend"""
        data = self._create_complete_market_data()
        
        # Adjust all prices to show trend
        multiplier = 1.001 if trend == "up" else 0.999
        
        # Update HF data
        for i, entry in enumerate(data["hf_data_window"]):
            if entry["1s_bar"]:
                entry["1s_bar"]["close"] = 100.0 * (multiplier ** i)
        
        # Update MF data
        for i, bar in enumerate(data["1m_bars_window"]):
            factor = multiplier ** (i * 60)
            bar["close"] = 100.0 * factor
            bar["open"] = bar["close"] * 0.999 if trend == "up" else bar["close"] * 1.001
        
        return data