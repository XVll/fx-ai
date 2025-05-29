"""Tests for point-in-time features to ensure they return appropriate single values."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock

from feature.feature_base import FeatureConfig
from feature.hf.quote_features import SpreadCompressionFeature
from feature.hf.tape_features import TapeImbalanceFeature, TapeAggressionRatioFeature
from feature.hf.quote_features import QuoteImbalanceFeature
from feature.market_structure.halt_features import HaltStateFeature, TimeSinceHaltFeature
from feature.volume_analysis.vwap_features import DistanceToVWAPFeature
from feature.volume_analysis.relative_volume_features import RelativeVolumeFeature


class TestPointTimeFeatures:
    """Test point-in-time features that should return single values based on current state."""

    def test_spread_compression_single_value(self):
        """Test spread compression returns single value from current market state."""
        feature = SpreadCompressionFeature(FeatureConfig(name="spread_compression"))
        
        # HF data with spread compression (wide to tight)
        compression_data = {
            'hf_data_window': [
                {  # Previous window - wide spread
                    'quotes': [{
                        'bid_price': 99.95,
                        'ask_price': 100.05  # 0.10 spread
                    }]
                },
                {  # Current window - tight spread
                    'quotes': [{
                        'bid_price': 99.99,
                        'ask_price': 100.01  # 0.02 spread
                    }]
                }
            ]
        }
        
        # HF data with spread expansion (tight to wide)
        expansion_data = {
            'hf_data_window': [
                {  # Previous window - tight spread
                    'quotes': [{
                        'bid_price': 99.99,
                        'ask_price': 100.01  # 0.02 spread
                    }]
                },
                {  # Current window - wide spread
                    'quotes': [{
                        'bid_price': 99.95,
                        'ask_price': 100.05  # 0.10 spread
                    }]
                }
            ]
        }
        
        compression_result = feature.calculate_raw(compression_data)
        expansion_result = feature.calculate_raw(expansion_data)
        
        # Validate single value outputs
        assert isinstance(compression_result, float), "Should return single float value"
        assert isinstance(expansion_result, float), "Should return single float value"
        
        # Compression should be positive (old_spread - new_spread > 0)
        assert compression_result > 0, "Spread compression should be positive"
        
        # Expansion should be negative (old_spread - new_spread < 0)
        assert expansion_result < 0, "Spread expansion should be negative"

    def test_tape_imbalance_point_in_time(self):
        """Test tape imbalance calculates from current tape data."""
        feature = TapeImbalanceFeature(FeatureConfig(name="tape_imbalance"))
        
        # Buy-heavy tape with explicit conditions
        buy_heavy_data = {
            'hf_data_window': [{
                'trades': [
                    {'conditions': ['BUY'], 'size': 1000},
                    {'conditions': ['BUY'], 'size': 500},
                    {'conditions': ['SELL'], 'size': 200},
                    {'conditions': ['BUY'], 'size': 300},
                ]
            }]
        }
        
        # Sell-heavy tape with explicit conditions
        sell_heavy_data = {
            'hf_data_window': [{
                'trades': [
                    {'conditions': ['SELL'], 'size': 1000},
                    {'conditions': ['SELL'], 'size': 800},
                    {'conditions': ['BUY'], 'size': 200},
                    {'conditions': ['SELL'], 'size': 500},
                ]
            }]
        }
        
        buy_imbalance = feature.calculate_raw(buy_heavy_data)
        sell_imbalance = feature.calculate_raw(sell_heavy_data)
        
        # Validate point-in-time calculation
        assert isinstance(buy_imbalance, float), "Should return single float"
        assert isinstance(sell_imbalance, float), "Should return single float"
        
        # Buy-heavy should be positive, sell-heavy should be negative
        assert buy_imbalance > 0, "Buy-heavy tape should have positive imbalance"
        assert sell_imbalance < 0, "Sell-heavy tape should have negative imbalance"
        
        # Test empty tape
        empty_result = feature.calculate_raw({'hf_data_window': [{'trades': []}]})
        assert empty_result == 0.0, "Empty tape should return 0.0"

    def test_quote_imbalance_current_state(self):
        """Test quote imbalance uses current bid/ask sizes."""
        feature = QuoteImbalanceFeature(FeatureConfig(name="quote_imbalance"))
        
        # Bid-heavy market (more size on bid)
        bid_heavy_data = {
            'current_quote': {
                'bid_px': 99.99,
                'ask_px': 100.01,
                'bid_sz': 5000,  # Large bid
                'ask_sz': 1000   # Small ask
            }
        }
        
        # Ask-heavy market (more size on ask)
        ask_heavy_data = {
            'current_quote': {
                'bid_px': 99.99,
                'ask_px': 100.01,
                'bid_sz': 800,   # Small bid
                'ask_sz': 4000   # Large ask
            }
        }
        
        # Balanced market
        balanced_data = {
            'current_quote': {
                'bid_px': 99.99,
                'ask_px': 100.01,
                'bid_sz': 2000,  # Equal sizes
                'ask_sz': 2000
            }
        }
        
        bid_heavy_result = feature.calculate_raw(bid_heavy_data)
        ask_heavy_result = feature.calculate_raw(ask_heavy_data)
        balanced_result = feature.calculate_raw(balanced_data)
        
        # Validate single values
        assert isinstance(bid_heavy_result, float), "Should return single float"
        assert isinstance(ask_heavy_result, float), "Should return single float"
        assert isinstance(balanced_result, float), "Should return single float"
        
        # Bid-heavy should be positive
        assert bid_heavy_result > 0, "Bid-heavy market should have positive imbalance"
        
        # Ask-heavy should be negative
        assert ask_heavy_result < 0, "Ask-heavy market should have negative imbalance"
        
        # Balanced should be near zero
        assert abs(balanced_result) < 0.1, "Balanced market should be near zero"

    def test_halt_state_binary_value(self):
        """Test halt state returns binary 0/1 value."""
        feature = HaltStateFeature(FeatureConfig(name="is_halted"))
        
        # Trading halted
        halted_data = {'is_halted': True}
        
        # Trading active
        active_data = {'is_halted': False}
        
        # Missing data
        missing_data = {}
        
        halted_result = feature.calculate_raw(halted_data)
        active_result = feature.calculate_raw(active_data)
        missing_result = feature.calculate_raw(missing_data)
        
        # Validate binary outputs
        assert halted_result == 1.0, "Halted should return 1.0"
        assert active_result == 0.0, "Active should return 0.0"
        assert missing_result == 0.0, "Missing data should default to 0.0"
        
        # All should be exact binary values
        assert halted_result in [0.0, 1.0], "Should be binary value"
        assert active_result in [0.0, 1.0], "Should be binary value"

    def test_time_since_halt_point_calculation(self):
        """Test time since halt calculates from current timestamp."""
        feature = TimeSinceHaltFeature(FeatureConfig(name="time_since_halt"))
        
        current_time = datetime.now()
        
        # Recent halt (5 minutes ago)
        recent_halt_data = {
            'timestamp': current_time,
            'last_halt_time': current_time - timedelta(minutes=5)
        }
        
        # Old halt (2 hours ago)
        old_halt_data = {
            'timestamp': current_time,
            'last_halt_time': current_time - timedelta(hours=2)
        }
        
        # No halt history
        no_halt_data = {
            'timestamp': current_time,
            'last_halt_time': None
        }
        
        recent_result = feature.calculate_raw(recent_halt_data)
        old_result = feature.calculate_raw(old_halt_data)
        no_halt_result = feature.calculate_raw(no_halt_data)
        
        # Validate time calculations
        assert isinstance(recent_result, float), "Should return single float"
        assert isinstance(old_result, float), "Should return single float"
        assert isinstance(no_halt_result, float), "Should return single float"
        
        # Recent halt should be ~300 seconds
        assert 250 <= recent_result <= 350, f"Recent halt should be ~300s, got {recent_result}"
        
        # Old halt should be capped at max (3600s)
        assert old_result == 3600.0, f"Old halt should be capped at 3600s, got {old_result}"
        
        # No halt should return max
        assert no_halt_result == 3600.0, "No halt should return max time"

    def test_distance_from_vwap_current_calculation(self):
        """Test VWAP distance uses current price vs session VWAP."""
        feature = DistanceToVWAPFeature(FeatureConfig(name="distance_to_vwap"))
        
        # Price above VWAP
        above_vwap_data = {
            'current_price': 102.50,
            'session_vwap': 100.00
        }
        
        # Price below VWAP
        below_vwap_data = {
            'current_price': 97.50,
            'session_vwap': 100.00
        }
        
        # Price at VWAP
        at_vwap_data = {
            'current_price': 100.00,
            'session_vwap': 100.00
        }
        
        above_result = feature.calculate_raw(above_vwap_data)
        below_result = feature.calculate_raw(below_vwap_data)
        at_result = feature.calculate_raw(at_vwap_data)
        
        # Validate single value calculations
        assert isinstance(above_result, float), "Should return single float"
        assert isinstance(below_result, float), "Should return single float"
        assert isinstance(at_result, float), "Should return single float"
        
        # Above VWAP should be positive
        assert above_result > 0, "Price above VWAP should be positive"
        
        # Below VWAP should be negative
        assert below_result < 0, "Price below VWAP should be negative"
        
        # At VWAP should be zero
        assert abs(at_result) < 1e-6, "Price at VWAP should be zero"
        
        # Validate percentage calculation
        expected_above = (102.50 - 100.00) / 100.00  # 2.5%
        expected_below = (97.50 - 100.00) / 100.00   # -2.5%
        
        assert abs(above_result - expected_above) < 0.01, "Should calculate correct percentage"
        assert abs(below_result - expected_below) < 0.01, "Should calculate correct percentage"

    def test_relative_volume_ratio_calculation(self):
        """Test relative volume compares current to historical average."""
        feature = RelativeVolumeFeature(FeatureConfig(name="relative_volume"))
        
        # High volume day
        high_volume_data = {
            'cumulative_volume_today': 2500000,
            'avg_volume_same_time': 1000000
        }
        
        # Low volume day
        low_volume_data = {
            'cumulative_volume_today': 500000,
            'avg_volume_same_time': 1000000
        }
        
        # Average volume day
        avg_volume_data = {
            'cumulative_volume_today': 1000000,
            'avg_volume_same_time': 1000000
        }
        
        high_result = feature.calculate_raw(high_volume_data)
        low_result = feature.calculate_raw(low_volume_data)
        avg_result = feature.calculate_raw(avg_volume_data)
        
        # Validate single value ratios
        assert isinstance(high_result, float), "Should return single float"
        assert isinstance(low_result, float), "Should return single float"
        assert isinstance(avg_result, float), "Should return single float"
        
        # High volume should be > 1
        assert high_result > 1.0, "High volume should have ratio > 1"
        
        # Low volume should be < 1
        assert low_result < 1.0, "Low volume should have ratio < 1"
        
        # Average volume should be ~1
        assert abs(avg_result - 1.0) < 0.01, "Average volume should be ~1"
        
        # Validate exact ratios
        assert abs(high_result - 2.5) < 0.01, "Should calculate correct ratio"
        assert abs(low_result - 0.5) < 0.01, "Should calculate correct ratio"

    def test_point_features_handle_missing_data(self):
        """Test point-in-time features handle missing/invalid data gracefully."""
        
        features_to_test = [
            SpreadCompressionFeature(FeatureConfig(name="spread_compression")),
            TapeImbalanceFeature(FeatureConfig(name="tape_imbalance")),
            QuoteImbalanceFeature(FeatureConfig(name="quote_imbalance")),
            HaltStateFeature(FeatureConfig(name="is_halted")),
            TimeSinceHaltFeature(FeatureConfig(name="time_since_halt")),
            DistanceToVWAPFeature(FeatureConfig(name="distance_to_vwap")),
            RelativeVolumeFeature(FeatureConfig(name="relative_volume"))
        ]
        
        # Test with empty data
        empty_data = {}
        
        for feature in features_to_test:
            result = feature.calculate_raw(empty_data)
            
            # Should return valid float (likely default value)
            assert isinstance(result, float), f"{feature.__class__.__name__} should return float for empty data"
            assert np.isfinite(result), f"{feature.__class__.__name__} should return finite value"
            
            # Should match default value
            expected_default = feature.get_default_value()
            assert result == expected_default, f"{feature.__class__.__name__} should return default value for empty data"

    def test_point_features_normalization_bounds(self):
        """Test that point-in-time features respect normalization bounds."""
        
        # Test extreme data that might push features out of bounds
        extreme_data_sets = [
            # Extreme spread compression
            {
                'current_quote': {'bid_px': 99.999, 'ask_px': 100.001, 'bid_sz': 10000, 'ask_sz': 10000},
                'avg_spread_5m': 1.0  # Very wide historical average
            },
            # Extreme tape imbalance
            {
                'recent_tape_1m': [{'side': 'buy', 'price': 100.0, 'size': 1000000}] * 100
            },
            # Extreme quote imbalance
            {
                'current_quote': {'bid_px': 99.0, 'ask_px': 101.0, 'bid_sz': 1000000, 'ask_sz': 1}
            },
            # Extreme VWAP distance
            {
                'current_price': 150.0,
                'session_vwap': 100.0
            },
            # Extreme relative volume
            {
                'cumulative_volume_today': 50000000,
                'avg_volume_same_time': 100000
            }
        ]
        
        features = [
            SpreadCompressionFeature(FeatureConfig(name="spread_compression")),
            TapeImbalanceFeature(FeatureConfig(name="tape_imbalance")),
            QuoteImbalanceFeature(FeatureConfig(name="quote_imbalance")),
            DistanceToVWAPFeature(FeatureConfig(name="distance_to_vwap")),
            RelativeVolumeFeature(FeatureConfig(name="relative_volume"))
        ]
        
        for i, feature in enumerate(features):
            if i < len(extreme_data_sets):
                result = feature.calculate_raw(extreme_data_sets[i])
                norm_params = feature.get_normalization_params()
                
                # Check if result respects normalization bounds (if specified)
                if 'min' in norm_params and 'max' in norm_params:
                    min_val = norm_params['min']
                    max_val = norm_params['max']
                    
                    assert min_val <= result <= max_val, \
                        f"{feature.__class__.__name__} result {result} outside bounds [{min_val}, {max_val}]"

    def test_point_features_deterministic_output(self):
        """Test that point-in-time features produce deterministic outputs."""
        
        # Fixed test data
        test_data = {
            'current_quote': {
                'bid_px': 99.99,
                'ask_px': 100.01,
                'bid_sz': 1500,
                'ask_sz': 1200
            },
            'recent_tape_1m': [
                {'side': 'buy', 'price': 100.01, 'size': 1000},
                {'side': 'sell', 'price': 99.99, 'size': 800},
                {'side': 'buy', 'price': 100.01, 'size': 600}
            ],
            'avg_spread_5m': 0.03,
            'current_price': 100.005,
            'session_vwap': 99.95,
            'cumulative_volume_today': 1500000,
            'avg_volume_same_time': 1200000,
            'is_halted': False
        }
        
        features = [
            SpreadCompressionFeature(FeatureConfig(name="spread_compression")),
            TapeImbalanceFeature(FeatureConfig(name="tape_imbalance")), 
            QuoteImbalanceFeature(FeatureConfig(name="quote_imbalance")),
            HaltStateFeature(FeatureConfig(name="is_halted")),
            DistanceToVWAPFeature(FeatureConfig(name="distance_to_vwap")),
            RelativeVolumeFeature(FeatureConfig(name="relative_volume"))
        ]
        
        # Calculate results multiple times
        for feature in features:
            results = []
            for _ in range(5):
                result = feature.calculate_raw(test_data)
                results.append(result)
            
            # All results should be identical (deterministic)
            assert all(r == results[0] for r in results), \
                f"{feature.__class__.__name__} should produce deterministic output"
            
            # Result should be finite
            assert np.isfinite(results[0]), \
                f"{feature.__class__.__name__} should produce finite result"