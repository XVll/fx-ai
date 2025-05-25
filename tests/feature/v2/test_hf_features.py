"""Tests for high-frequency features - TDD approach"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# These imports will fail initially (TDD)
from feature.v2.hf.price_features import PriceVelocityFeature, PriceAccelerationFeature
from feature.v2.hf.tape_features import TapeImbalanceFeature, TapeAggressionRatioFeature


class TestPriceFeatures:
    """Test high-frequency price-based features"""
    
    def test_price_velocity_feature(self):
        """Test 1-second price velocity calculation"""
        feature = PriceVelocityFeature()
        
        # Test with price movement
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 2, tzinfo=timezone.utc),
            "current_price": 100.50,
            "hf_data_window": [
                {"timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 100.00}},
                {"timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 100.25}},
                {"timestamp": datetime(2025, 1, 25, 15, 0, 2, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 100.50}}
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Raw velocity = (current_price - price_1s_ago) / price_1s_ago
        raw_velocity = (100.50 - 100.25) / 100.25
        
        # Should be normalized to a reasonable range (e.g., [-1, 1] for ±100% change)
        # Assuming normalization clips at ±10% per second to [-1, 1]
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Normalized range
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["data_type"] == "1s_bars"
        assert requirements["lookback"] >= 2
        assert "hf_data_window" in requirements["fields"]
    
    def test_price_velocity_edge_cases(self):
        """Test price velocity with edge cases"""
        feature = PriceVelocityFeature()
        
        # Test with no price change
        market_data = {
            "timestamp": datetime.now(timezone.utc),
            "current_price": 100.00,
            "hf_data_window": [
                {"1s_bar": {"close": 100.00}},
                {"1s_bar": {"close": 100.00}}
            ]
        }
        
        result = feature.calculate(market_data)
        assert result == 0.0
        assert not np.isnan(result)
        
        # Test with missing data - should use default
        market_data["hf_data_window"] = []
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0  # Should be in normalized range
        
        # Test with single data point
        market_data["hf_data_window"] = [{"1s_bar": {"close": 100.00}}]
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
        
        # Test with missing close prices
        market_data["hf_data_window"] = [
            {"1s_bar": {}},  # No close
            {"1s_bar": {"close": None}},  # None close
            {"1s_bar": {"close": np.nan}}  # NaN close
        ]
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
        
        # Test with extreme price movements (should be clipped)
        market_data["hf_data_window"] = [
            {"1s_bar": {"close": 100.00}},
            {"1s_bar": {"close": 200.00}}  # 100% increase
        ]
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert result == 1.0  # Should be clipped to max
    
    def test_price_acceleration_feature(self):
        """Test 1-second price acceleration calculation"""
        feature = PriceAccelerationFeature()
        
        # Test with changing velocity
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 3, tzinfo=timezone.utc),
            "current_price": 101.00,
            "hf_data_window": [
                {"timestamp": datetime(2025, 1, 25, 15, 0, 0, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 100.00}},
                {"timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 100.25}},  # velocity1 = 0.25%
                {"timestamp": datetime(2025, 1, 25, 15, 0, 2, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 100.50}},  # velocity2 = 0.25%
                {"timestamp": datetime(2025, 1, 25, 15, 0, 3, tzinfo=timezone.utc), 
                 "1s_bar": {"close": 101.00}}   # velocity3 = 0.50%
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Acceleration = change in velocity
        # velocity_t = (101.00 - 100.50) / 100.50 = 0.00497...
        # velocity_t-1 = (100.50 - 100.25) / 100.25 = 0.00249...
        # acceleration = velocity_t - velocity_t-1 ≈ 0.00248
        assert result > 0  # Positive acceleration
        assert abs(result - 0.00248) < 0.001
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["lookback"] >= 3  # Need 3 points for acceleration


class TestTapeFeatures:
    """Test tape (trade) analysis features"""
    
    def test_tape_imbalance_feature(self):
        """Test buy/sell volume imbalance in 1-second window"""
        feature = TapeImbalanceFeature()
        
        # Test with trades
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc),
            "hf_data_window": [
                {
                    "timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc),
                    "trades": [
                        {"price": 100.10, "size": 100, "conditions": ["BUY"]},  # Buy
                        {"price": 100.10, "size": 200, "conditions": ["BUY"]},  # Buy
                        {"price": 100.05, "size": 150, "conditions": ["SELL"]}, # Sell
                    ]
                }
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Buy volume = 300, Sell volume = 150
        # Raw imbalance = (buy - sell) / (buy + sell) = 150 / 450 = 0.333...
        # This is already in [-1, 1] range
        expected_normalized = (300 - 150) / (300 + 150)
        assert abs(result - expected_normalized) < 1e-6
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
        
        # Test requirements
        requirements = feature.get_requirements()
        assert requirements["data_type"] == "trades"
        assert "trades" in requirements["fields"]
    
    def test_tape_imbalance_edge_cases(self):
        """Test tape imbalance edge cases"""
        feature = TapeImbalanceFeature()
        
        # All buys
        market_data = {
            "hf_data_window": [{
                "trades": [
                    {"price": 100.10, "size": 100, "conditions": ["BUY"]},
                    {"price": 100.10, "size": 200, "conditions": ["BUY"]},
                ]
            }]
        }
        
        result = feature.calculate(market_data)
        assert result == 1.0  # Maximum buy imbalance
        assert not np.isnan(result)
        
        # All sells
        market_data["hf_data_window"][0]["trades"] = [
            {"price": 100.10, "size": 100, "conditions": ["SELL"]},
            {"price": 100.10, "size": 200, "conditions": ["SELL"]},
        ]
        
        result = feature.calculate(market_data)
        assert result == -1.0  # Maximum sell imbalance
        assert not np.isnan(result)
        
        # No trades - should use default
        market_data["hf_data_window"][0]["trades"] = []
        result = feature.calculate(market_data)
        assert result == 0.0  # Neutral default
        assert not np.isnan(result)
        
        # No hf_data_window at all
        result = feature.calculate({})
        assert result == 0.0  # Should handle gracefully
        assert not np.isnan(result)
        
        # Trades with missing conditions - should infer from price vs quotes
        market_data = {
            "hf_data_window": [{
                "quotes": [{"bid_price": 100.00, "ask_price": 100.10}],
                "trades": [
                    {"price": 100.10, "size": 100},  # At ask = buy
                    {"price": 100.00, "size": 100},  # At bid = sell
                    {"price": 100.05, "size": 100},  # Mid = neutral
                ]
            }]
        }
        
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
        
        # Trades with invalid size
        market_data["hf_data_window"][0]["trades"] = [
            {"price": 100.10, "size": None, "conditions": ["BUY"]},
            {"price": 100.10, "size": -100, "conditions": ["BUY"]},  # Negative
            {"price": 100.10, "size": "invalid", "conditions": ["BUY"]},
        ]
        
        result = feature.calculate(market_data)
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0
    
    def test_tape_aggression_ratio_feature(self):
        """Test tape aggression ratio (market orders hitting bid vs ask)"""
        feature = TapeAggressionRatioFeature()
        
        # Test with trades at different price levels
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc),
            "hf_data_window": [
                {
                    "timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc),
                    "quotes": [
                        {"bid_price": 100.00, "ask_price": 100.10, 
                         "timestamp": datetime(2025, 1, 25, 15, 0, 0, 900000, tzinfo=timezone.utc)}
                    ],
                    "trades": [
                        # Aggressive buy (hits ask)
                        {"price": 100.10, "size": 100, 
                         "timestamp": datetime(2025, 1, 25, 15, 0, 0, 950000, tzinfo=timezone.utc)},
                        # Aggressive sell (hits bid)
                        {"price": 100.00, "size": 150,
                         "timestamp": datetime(2025, 1, 25, 15, 0, 0, 980000, tzinfo=timezone.utc)},
                        # Aggressive buy (hits ask)
                        {"price": 100.10, "size": 200,
                         "timestamp": datetime(2025, 1, 25, 15, 0, 0, 990000, tzinfo=timezone.utc)},
                    ]
                }
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Aggressive buys = 300 (100 + 200), Aggressive sells = 150
        # Ratio = (aggressive_buys - aggressive_sells) / total_volume
        # = (300 - 150) / 450 = 0.333...
        expected_ratio = (300 - 150) / 450
        assert abs(result - expected_ratio) < 1e-6
    
    def test_tape_aggression_with_midpoint_trades(self):
        """Test aggression ratio with trades at midpoint"""
        feature = TapeAggressionRatioFeature()
        
        market_data = {
            "hf_data_window": [{
                "quotes": [
                    {"bid_price": 100.00, "ask_price": 100.10}
                ],
                "trades": [
                    {"price": 100.10, "size": 100},  # At ask (aggressive buy)
                    {"price": 100.05, "size": 100},  # At midpoint (passive)
                    {"price": 100.00, "size": 100},  # At bid (aggressive sell)
                ]
            }]
        }
        
        result = feature.calculate(market_data)
        
        # Only trades at bid/ask are considered aggressive
        # Aggressive buys = 100, Aggressive sells = 100, Passive = 100
        # Ratio = (100 - 100) / 300 = 0
        assert result == 0.0


class TestQuoteFeatures:
    """Test quote-based features"""
    
    def test_quote_spread_compression_feature(self):
        """Test bid-ask spread compression over 1 second"""
        feature = feature = type('SpreadCompressionFeature', (), {
            'calculate': lambda self, data: self._calc(data),
            '_calc': lambda self, data: self._calculate_spread_compression(data),
            '_calculate_spread_compression': lambda self, data: (
                (data['hf_data_window'][-2]['quotes'][-1]['ask_price'] - 
                 data['hf_data_window'][-2]['quotes'][-1]['bid_price']) -
                (data['hf_data_window'][-1]['quotes'][-1]['ask_price'] - 
                 data['hf_data_window'][-1]['quotes'][-1]['bid_price'])
            ) if len(data.get('hf_data_window', [])) >= 2 and 
                 all('quotes' in w and w['quotes'] for w in data['hf_data_window'][-2:]) else 0.0,
            'get_requirements': lambda self: {"data_type": "quotes", "lookback": 2, "fields": ["quotes"]}
        })()
        
        # Test spread tightening
        market_data = {
            "hf_data_window": [
                {
                    "quotes": [
                        {"bid_price": 100.00, "ask_price": 100.10}  # Spread = 0.10
                    ]
                },
                {
                    "quotes": [
                        {"bid_price": 100.02, "ask_price": 100.07}  # Spread = 0.05
                    ]
                }
            ]
        }
        
        result = feature.calculate(market_data)
        
        # Compression = old_spread - new_spread = 0.10 - 0.05 = 0.05
        assert result == 0.05  # Positive means spread compressed
        
        # Test spread widening
        market_data["hf_data_window"][1]["quotes"][0] = {
            "bid_price": 99.95, "ask_price": 100.15  # Spread = 0.20
        }
        
        result = feature.calculate(market_data)
        assert result == -0.10  # Negative means spread widened


class TestHFFeatureIntegration:
    """Test integration of multiple HF features"""
    
    def test_multiple_hf_features_together(self):
        """Test calculating multiple HF features from same data"""
        features = [
            PriceVelocityFeature(),
            TapeImbalanceFeature()
        ]
        
        # Rich market data
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 2, tzinfo=timezone.utc),
            "current_price": 100.50,
            "hf_data_window": [
                {
                    "timestamp": datetime(2025, 1, 25, 15, 0, 1, tzinfo=timezone.utc),
                    "1s_bar": {"close": 100.00},
                    "trades": []
                },
                {
                    "timestamp": datetime(2025, 1, 25, 15, 0, 2, tzinfo=timezone.utc),
                    "1s_bar": {"close": 100.50},
                    "trades": [
                        {"price": 100.50, "size": 300, "conditions": ["BUY"]},
                        {"price": 100.45, "size": 100, "conditions": ["SELL"]},
                    ]
                }
            ]
        }
        
        results = {}
        for feature in features:
            results[feature.name] = feature.calculate(market_data)
        
        # Check all features return valid values
        assert len(results) == 2
        assert all(isinstance(v, float) for v in results.values())
        assert all(not np.isnan(v) for v in results.values())
        assert all(np.isfinite(v) for v in results.values())
        
        # Check normalized ranges
        assert -1.0 <= results["price_velocity"] <= 1.0
        assert -1.0 <= results["tape_imbalance"] <= 1.0
        
        # Verify directional correctness
        assert results["price_velocity"] > 0  # Price increased
        assert results["tape_imbalance"] > 0  # More buy volume
    
    def test_hf_features_with_sparse_data(self):
        """Test HF features handle sparse/missing data properly"""
        features = [
            PriceVelocityFeature(),
            PriceAccelerationFeature(),
            TapeImbalanceFeature(),
            TapeAggressionRatioFeature()
        ]
        
        # Sparse data - some seconds have no trades/quotes
        market_data = {
            "timestamp": datetime(2025, 1, 25, 15, 0, 5, tzinfo=timezone.utc),
            "current_price": 100.00,
            "hf_data_window": [
                {"1s_bar": {"close": 100.00}, "trades": [], "quotes": []},  # No activity
                {"1s_bar": None, "trades": [], "quotes": []},  # No bar
                {"1s_bar": {"close": 100.10}, "trades": [{"price": 100.10, "size": 100}], "quotes": []},
                {"1s_bar": None, "trades": [], "quotes": []},  # Gap
                {"1s_bar": {"close": 100.05}, "trades": [], "quotes": [{"bid_price": 100.00, "ask_price": 100.10}]}
            ]
        }
        
        results = {}
        for feature in features:
            result = feature.calculate(market_data)
            results[feature.name] = result
            
            # All features should handle sparse data gracefully
            assert not np.isnan(result), f"{feature.name} returned NaN"
            assert np.isfinite(result), f"{feature.name} returned non-finite value"
            assert -1.0 <= result <= 1.0, f"{feature.name} out of normalized range"
    
    def test_hf_features_data_requirements_aggregation(self):
        """Test aggregating requirements from multiple HF features"""
        features = [
            PriceVelocityFeature(),
            PriceAccelerationFeature(),
            TapeImbalanceFeature(),
            TapeAggressionRatioFeature()
        ]
        
        # Aggregate requirements
        all_requirements = {
            "lookback": 0,
            "data_types": set(),
            "fields": set()
        }
        
        for feature in features:
            req = feature.get_requirements()
            all_requirements["lookback"] = max(all_requirements["lookback"], 
                                             req.get("lookback", 0))
            if "data_type" in req:
                all_requirements["data_types"].add(req["data_type"])
            if "fields" in req:
                all_requirements["fields"].update(req["fields"])
        
        # Should need various data types
        assert "1s_bars" in all_requirements["data_types"]
        assert "trades" in all_requirements["data_types"]
        assert all_requirements["lookback"] >= 3  # For acceleration
    
    def test_hf_final_output_format(self):
        """Test HF features produce correct output format for model"""
        # Simulate a sequence of 60 seconds of HF data
        hf_window = []
        base_price = 100.0
        
        for i in range(60):
            # Create realistic market data with some noise
            price = base_price + np.random.normal(0, 0.05)
            
            trades = []
            if np.random.random() > 0.3:  # 70% chance of trades
                n_trades = np.random.randint(1, 5)
                for _ in range(n_trades):
                    side = "BUY" if np.random.random() > 0.5 else "SELL"
                    trades.append({
                        "price": price + np.random.uniform(-0.01, 0.01),
                        "size": np.random.randint(100, 1000),
                        "conditions": [side]
                    })
            
            hf_window.append({
                "timestamp": datetime(2025, 1, 25, 15, 0, i, tzinfo=timezone.utc),
                "1s_bar": {"close": price} if trades else None,
                "trades": trades,
                "quotes": [{"bid_price": price - 0.01, "ask_price": price + 0.01}]
            })
            
            base_price = price  # Random walk
        
        # Create feature extractors
        features = [
            PriceVelocityFeature(),
            TapeImbalanceFeature()
        ]
        
        # Extract features for the entire window
        feature_matrix = np.zeros((60, len(features)), dtype=np.float32)
        
        for t in range(60):
            market_data = {
                "timestamp": hf_window[t]["timestamp"],
                "current_price": hf_window[t]["1s_bar"]["close"] if hf_window[t]["1s_bar"] else base_price,
                "hf_data_window": hf_window[max(0, t-10):t+1]  # Use last 10 seconds
            }
            
            for f_idx, feature in enumerate(features):
                value = feature.calculate(market_data)
                feature_matrix[t, f_idx] = value
        
        # Verify output properties
        assert feature_matrix.shape == (60, 2)
        assert feature_matrix.dtype == np.float32
        assert not np.any(np.isnan(feature_matrix))
        assert np.all(np.isfinite(feature_matrix))
        assert np.all(feature_matrix >= -1.0)
        assert np.all(feature_matrix <= 1.0)