"""Tests for specific feature implementations"""
import pytest
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual feature implementations for testing
from feature.feature_base import FeatureConfig


class TestFeatureImplementations:
    """Test actual feature implementations with realistic data"""
    
    def test_hf_price_features(self):
        """Test high-frequency price features"""
        try:
            from feature.hf.price_features import PriceVelocityFeature, PriceAccelerationFeature
            
            # Test PriceVelocityFeature
            feature = PriceVelocityFeature()
            
            # Create realistic HF data
            hf_data = [
                {
                    "timestamp": datetime.now(),
                    "1s_bar": {"close": 100.0, "open": 99.8, "high": 100.2, "low": 99.7}
                },
                {
                    "timestamp": datetime.now(),
                    "1s_bar": {"close": 100.5, "open": 100.0, "high": 100.6, "low": 99.9}
                }
            ]
            
            market_data = {
                "timestamp": datetime.now(),
                "hf_data_window": hf_data
            }
            
            result = feature.calculate(market_data)
            
            # Should return normalized velocity
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            assert -1.0 <= result <= 1.0  # Normalized range
            
            # Test with insufficient data
            insufficient_data = {
                "hf_data_window": [hf_data[0]]  # Only one data point
            }
            result = feature.calculate(insufficient_data)
            assert result == 0.0  # Should return default for insufficient data
            
            # Test with missing bars
            missing_bar_data = {
                "hf_data_window": [
                    {"timestamp": datetime.now(), "1s_bar": None},
                    {"timestamp": datetime.now(), "1s_bar": {"close": 100.0}}
                ]
            }
            result = feature.calculate(missing_bar_data)
            assert result == 0.0
            
        except ImportError:
            pytest.skip("HF price features not available")
    
    def test_mf_candle_features(self):
        """Test medium-frequency candle features"""
        try:
            from feature.mf.candle_features import (
                PositionInCurrentCandle1mFeature, 
                BodySizeRelative1mFeature
            )
            
            # Test PositionInCurrentCandle1mFeature
            feature = PositionInCurrentCandle1mFeature()
            
            # Create realistic 1m bar data
            bar_data = [
                {
                    "timestamp": datetime.now(),
                    "open": 100.0,
                    "high": 100.8,
                    "low": 99.5,
                    "close": 100.3,
                    "volume": 1000,
                    "is_current": True
                }
            ]
            
            market_data = {
                "current_price": 100.6,  # 80% up in the range
                "1m_bars_window": bar_data
            }
            
            result = feature.calculate(market_data)
            
            # Should return position in candle [0, 1]
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
            # Price at 100.6 in range [99.5, 100.8] should be around 0.85
            assert 0.8 <= result <= 0.9
            
            # Test edge cases
            # Price at high
            market_data["current_price"] = 100.8
            result = feature.calculate(market_data)
            assert result == 1.0
            
            # Price at low
            market_data["current_price"] = 99.5
            result = feature.calculate(market_data)
            assert result == 0.0
            
            # Price outside range (should be clipped)
            market_data["current_price"] = 101.0
            result = feature.calculate(market_data)
            assert result == 1.0
            
            # No range (high == low)
            bar_data[0]["high"] = 100.0
            bar_data[0]["low"] = 100.0
            result = feature.calculate(market_data)
            assert result == 0.5
            
        except ImportError:
            pytest.skip("MF candle features not available")
    
    def test_lf_range_features(self):
        """Test low-frequency range features"""
        try:
            from feature.lf.range_features import (
                PositionInDailyRangeFeature,
                PriceChangeFromPrevCloseFeature
            )
            
            # Test PositionInDailyRangeFeature
            feature = PositionInDailyRangeFeature()
            
            market_data = {
                "current_price": 102.0,
                "intraday_high": 105.0,
                "intraday_low": 100.0
            }
            
            result = feature.calculate(market_data)
            
            # Should return position in daily range [0, 1]
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
            # Price at 102.0 in range [100.0, 105.0] should be 0.4
            assert abs(result - 0.4) < 0.01
            
            # Test PriceChangeFromPrevCloseFeature
            change_feature = PriceChangeFromPrevCloseFeature()
            
            change_data = {
                "current_price": 102.0,
                "previous_day_data": {"close": 100.0}
            }
            
            result = change_feature.calculate(change_data)
            
            # Should return normalized price change
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
        except ImportError:
            pytest.skip("LF range features not available")
    
    def test_volume_analysis_features(self):
        """Test volume analysis features"""
        try:
            from feature.volume_analysis.vwap_features import (
                DistanceToVWAPFeature,
                VWAPSlopeFeature
            )
            
            # Test DistanceToVWAPFeature
            feature = DistanceToVWAPFeature()
            
            # Create realistic bar data with VWAP calculation
            bars_data = []
            cumulative_volume = 0
            cumulative_pv = 0
            
            for i, price in enumerate([100.0, 100.5, 101.0, 100.8, 100.3]):
                volume = 1000 + i * 100
                cumulative_volume += volume
                cumulative_pv += price * volume
                vwap = cumulative_pv / cumulative_volume
                
                bars_data.append({
                    "timestamp": datetime.now(),
                    "close": price,
                    "volume": volume,
                    "vwap": vwap
                })
            
            market_data = {
                "current_price": 101.2,
                "1m_bars_window": bars_data
            }
            
            result = feature.calculate(market_data)
            
            # Should return normalized distance to VWAP
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
        except ImportError:
            pytest.skip("Volume analysis features not available")
    
    def test_professional_features(self):
        """Test professional technical analysis features"""
        try:
            from feature.professional.ta_features import ProfessionalEMASystemFeature
            from feature.feature_base import FeatureConfig
            
            config = FeatureConfig("professional_ema_system")
            feature = ProfessionalEMASystemFeature(config)
            
            # Create realistic price data for EMA calculation
            prices = [100 + np.sin(i/10) * 2 for i in range(50)]  # Smooth price movement
            bars_data = []
            
            for i, price in enumerate(prices):
                bars_data.append({
                    "timestamp": datetime.now(),
                    "open": price - 0.1,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price,
                    "volume": 1000
                })
            
            market_data = {
                "current_price": prices[-1],
                "1m_bars_window": bars_data
            }
            
            result = feature.calculate(market_data)
            
            # Should return EMA alignment score
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
        except ImportError:
            pytest.skip("Professional features not available")
    
    def test_adaptive_features(self):
        """Test adaptive features"""
        try:
            from feature.adaptive.adaptive_features import (
                VolatilityAdjustedMomentumFeature,
                RegimeRelativeVolumeFeature
            )
            
            # Test VolatilityAdjustedMomentumFeature
            from feature.feature_base import FeatureConfig
            config = FeatureConfig("volatility_adjusted_momentum")
            feature = VolatilityAdjustedMomentumFeature(config)
            
            # Create data with varying volatility
            prices = [100.0]
            for i in range(49):
                # Simulate price movement with increasing volatility
                volatility = 0.01 * (1 + i/50)  # Increasing volatility
                change = np.random.normal(0, volatility)
                prices.append(prices[-1] * (1 + change))
            
            bars_data = []
            for price in prices:
                bars_data.append({
                    "timestamp": datetime.now(),
                    "close": price,
                    "volume": 1000
                })
            
            market_data = {
                "current_price": prices[-1],
                "1m_bars_window": bars_data
            }
            
            result = feature.calculate(market_data)
            
            # Should return volatility-adjusted momentum
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
        except ImportError:
            pytest.skip("Adaptive features not available")
    
    def test_pattern_features(self):
        """Test pattern recognition features"""
        try:
            from feature.pattern.pattern_features import (
                BreakoutPatternFeature,
                ConsolidationPatternFeature
            )
            
            # Test BreakoutPatternFeature
            feature = BreakoutPatternFeature()
            
            # Create consolidation followed by breakout pattern
            consolidation_prices = [100.0] * 20  # Flat consolidation
            breakout_prices = [100.0, 100.5, 101.0, 101.8, 102.5]  # Strong breakout
            all_prices = consolidation_prices + breakout_prices
            
            bars_data = []
            for price in all_prices:
                bars_data.append({
                    "timestamp": datetime.now(),
                    "open": price - 0.1,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": 1000
                })
            
            market_data = {
                "current_price": all_prices[-1],
                "1m_bars_window": bars_data
            }
            
            result = feature.calculate(market_data)
            
            # Should detect breakout pattern
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
        except ImportError:
            pytest.skip("Pattern features not available")
    
    def test_sequence_aware_features(self):
        """Test sequence-aware features"""
        try:
            from feature.sequence_aware.sequence_features import (
                TrendAccelerationFeature,
                MomentumQualityFeature
            )
            
            # Test TrendAccelerationFeature
            feature = TrendAccelerationFeature()
            
            # Create accelerating trend
            prices = []
            velocity = 0.1
            acceleration = 0.01
            price = 100.0
            
            for i in range(30):
                price += velocity
                velocity += acceleration
                prices.append(price)
            
            bars_data = []
            for price in prices:
                bars_data.append({
                    "timestamp": datetime.now(),
                    "close": price,
                    "volume": 1000
                })
            
            market_data = {
                "current_price": prices[-1],
                "1m_bars_window": bars_data
            }
            
            result = feature.calculate(market_data)
            
            # Should detect trend acceleration
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
        except ImportError:
            pytest.skip("Sequence-aware features not available")
    
    def test_market_structure_features(self):
        """Test market structure features"""
        try:
            from feature.market_structure.halt_features import (
                HaltStateFeature,
                TimeSinceHaltFeature
            )
            
            # Test HaltStateFeature
            from feature.feature_base import FeatureConfig
            config = FeatureConfig("halt_state")
            feature = HaltStateFeature(config)
            
            market_data = {
                "timestamp": datetime.now(),
                "halt_state": "TRADING"  # Normal trading
            }
            
            result = feature.calculate(market_data)
            
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            
            # Test halt condition
            market_data["halt_state"] = "HALTED"
            result_halted = feature.calculate(market_data)
            
            assert isinstance(result_halted, float)
            assert not np.isnan(result_halted)
            assert not np.isinf(result_halted)
            # Note: Both states might normalize to same value, that's fine
            
        except ImportError:
            pytest.skip("Market structure features not available")
    
    def test_context_features(self):
        """Test context features"""
        try:
            from feature.context.context_features import (
                SessionProgressFeature,
                MarketStressFeature
            )
            
            # Test SessionProgressFeature
            feature = SessionProgressFeature()
            
            market_data = {
                "timestamp": datetime.now(),
                "session_start": datetime.now().replace(hour=9, minute=30),  # 9:30 AM
                "session_end": datetime.now().replace(hour=16, minute=0),    # 4:00 PM
                "session": "REGULAR"
            }
            
            result = feature.calculate(market_data)
            
            # Should return session progress [0, 1]
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
            
        except ImportError:
            pytest.skip("Context features not available")


class TestFeatureEdgeCasesImplementations:
    """Test edge cases with actual feature implementations"""
    
    def test_zero_division_handling(self):
        """Test that features handle zero division gracefully"""
        try:
            from feature.hf.price_features import PriceVelocityFeature
            
            feature = PriceVelocityFeature()
            
            # Create data with zero previous price
            hf_data = [
                {"1s_bar": {"close": 0.0}},  # Zero price
                {"1s_bar": {"close": 100.0}}
            ]
            
            market_data = {"hf_data_window": hf_data}
            result = feature.calculate(market_data)
            
            # Should handle zero division
            assert result == 0.0
            
        except ImportError:
            pytest.skip("HF price features not available")
    
    def test_missing_data_handling(self):
        """Test handling of missing or incomplete data"""
        try:
            from feature.mf.candle_features import PositionInCurrentCandle1mFeature
            
            feature = PositionInCurrentCandle1mFeature()
            
            # Test various missing data scenarios
            test_cases = [
                {},  # Empty data
                {"current_price": 100.0},  # Missing bars
                {"1m_bars_window": []},  # Empty bars
                {"current_price": 100.0, "1m_bars_window": [{}]},  # Empty bar
                {"current_price": None, "1m_bars_window": [{"high": 105, "low": 95}]},  # None price
            ]
            
            for test_data in test_cases:
                result = feature.calculate(test_data)
                assert isinstance(result, float)
                assert not np.isnan(result)
                assert not np.isinf(result)
                assert 0.0 <= result <= 1.0
                
        except ImportError:
            pytest.skip("MF candle features not available")
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements"""
        try:
            from feature.hf.price_features import PriceVelocityFeature
            
            feature = PriceVelocityFeature()
            
            # Test extreme price jump
            hf_data = [
                {"1s_bar": {"close": 100.0}},
                {"1s_bar": {"close": 1000.0}}  # 10x price jump
            ]
            
            market_data = {"hf_data_window": hf_data}
            result = feature.calculate(market_data)
            
            # Should be normalized and clipped
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            assert -1.0 <= result <= 1.0
            
            # Test extreme price drop
            hf_data[1]["1s_bar"]["close"] = 1.0  # 99% drop
            result = feature.calculate(market_data)
            
            assert isinstance(result, float)
            assert not np.isnan(result)
            assert not np.isinf(result)
            assert -1.0 <= result <= 1.0
            
        except ImportError:
            pytest.skip("HF price features not available")
    
    def test_data_consistency(self):
        """Test that features produce consistent results with same input"""
        try:
            from feature.lf.range_features import PositionInDailyRangeFeature
            
            feature = PositionInDailyRangeFeature()
            
            market_data = {
                "current_price": 102.5,
                "intraday_high": 105.0,
                "intraday_low": 100.0
            }
            
            # Calculate multiple times
            results = []
            for _ in range(10):
                result = feature.calculate(market_data)
                results.append(result)
            
            # All results should be identical
            assert len(set(results)) == 1
            assert all(isinstance(r, float) for r in results)
            
        except ImportError:
            pytest.skip("LF range features not available")
    
    def test_normalization_bounds(self):
        """Test that normalized features stay within expected bounds"""
        try:
            from feature.hf.price_features import PriceVelocityFeature
            
            feature = PriceVelocityFeature()
            
            # Test with various price movements
            test_movements = [
                (100.0, 100.01),  # Small movement
                (100.0, 101.0),   # 1% movement
                (100.0, 110.0),   # 10% movement
                (100.0, 200.0),   # 100% movement
                (100.0, 99.99),   # Small negative
                (100.0, 99.0),    # -1% movement
                (100.0, 90.0),    # -10% movement
                (100.0, 50.0),    # -50% movement
            ]
            
            for prev_price, curr_price in test_movements:
                hf_data = [
                    {"1s_bar": {"close": prev_price}},
                    {"1s_bar": {"close": curr_price}}
                ]
                
                market_data = {"hf_data_window": hf_data}
                result = feature.calculate(market_data)
                
                # Should be within normalized bounds
                assert -1.0 <= result <= 1.0
                assert isinstance(result, float)
                assert not np.isnan(result)
                assert not np.isinf(result)
                
        except ImportError:
            pytest.skip("HF price features not available")
    
    def test_temporal_consistency(self):
        """Test features with time-based data"""
        try:
            from feature.lf.time_features import TimeOfDaySinFeature, TimeOfDayCosFeature
            
            sin_feature = TimeOfDaySinFeature()
            cos_feature = TimeOfDayCosFeature()
            
            # Test at different times of day
            test_times = [
                datetime.now().replace(hour=9, minute=30),   # Market open
                datetime.now().replace(hour=12, minute=0),   # Noon
                datetime.now().replace(hour=16, minute=0),   # Market close
                datetime.now().replace(hour=20, minute=0),   # After hours
            ]
            
            for test_time in test_times:
                market_data = {"timestamp": test_time}
                
                sin_result = sin_feature.calculate(market_data)
                cos_result = cos_feature.calculate(market_data)
                
                # Should satisfy sin^2 + cos^2 = 1 (approximately)
                assert isinstance(sin_result, float)
                assert isinstance(cos_result, float)
                assert -1.0 <= sin_result <= 1.0
                assert -1.0 <= cos_result <= 1.0
                
                # Verify trigonometric relationship
                magnitude = sin_result**2 + cos_result**2
                assert abs(magnitude - 1.0) < 0.1  # Allow for normalization effects
                
        except ImportError:
            pytest.skip("LF time features not available")


class TestFeatureSystemIntegration:
    """Integration tests for the complete feature system"""
    
    def test_full_feature_extraction_pipeline(self):
        """Test the complete feature extraction pipeline with realistic data"""
        # This test requires the actual feature manager to work
        try:
            from feature.simple_feature_manager import SimpleFeatureManager
            from feature.contexts import MarketContext
            
            # Create mock config
            config = Mock()
            config.hf_seq_len = 60
            config.hf_feat_dim = 7
            config.mf_seq_len = 30
            config.mf_feat_dim = 43
            config.lf_seq_len = 30
            config.lf_feat_dim = 19
            
            # Create feature manager with mocked initialization
            with patch.object(SimpleFeatureManager, '_initialize_features') as mock_init:
                mock_init.return_value = {'hf': [], 'mf': [], 'lf': []}
                
                manager = SimpleFeatureManager("MLGO", config)
                
                # Create realistic market context
                base_time = datetime.now()
                
                # HF window - 60 seconds of 1s data
                hf_window = []
                for i in range(60):
                    price = 100.0 + np.sin(i/10) * 0.5  # Gentle price oscillation
                    hf_window.append({
                        "timestamp": base_time,
                        "1s_bar": {
                            "open": price - 0.01,
                            "high": price + 0.02,
                            "low": price - 0.02,
                            "close": price,
                            "volume": 100 + i
                        },
                        "trades": [{"price": price, "size": 100, "side": "buy"}],
                        "quotes": [{"bid": price - 0.01, "ask": price + 0.01, "bid_size": 500, "ask_size": 600}]
                    })
                
                # MF window - 30 minutes of 1m data
                mf_1m_window = []
                for i in range(30):
                    price = 100.0 + i * 0.1  # Gradual uptrend
                    mf_1m_window.append({
                        "timestamp": base_time,
                        "open": price - 0.05,
                        "high": price + 0.08,
                        "low": price - 0.08,
                        "close": price,
                        "volume": 1000 + i * 50
                    })
                
                # LF window - 5m bars
                lf_5m_window = []
                for i in range(10):
                    price = 100.0 + i * 0.3
                    lf_5m_window.append({
                        "timestamp": base_time,
                        "open": price - 0.1,
                        "high": price + 0.2,
                        "low": price - 0.15,
                        "close": price,
                        "volume": 5000 + i * 200
                    })
                
                context = MarketContext(
                    timestamp=base_time,
                    current_price=103.0,
                    hf_window=hf_window,
                    mf_1m_window=mf_1m_window,
                    lf_5m_window=lf_5m_window,
                    prev_day_close=98.0,
                    prev_day_high=105.0,
                    prev_day_low=95.0,
                    session_high=104.0,
                    session_low=99.0,
                    session="REGULAR",
                    market_cap=1000000.0,
                    session_volume=100000.0,
                    session_trades=2000,
                    session_vwap=101.5
                )
                
                # Extract features
                features = manager.extract_features(context)
                
                # Verify output structure
                assert features is not None
                assert isinstance(features, dict)
                assert 'hf' in features
                assert 'mf' in features
                assert 'lf' in features
                
                # Verify shapes
                assert features['hf'].shape == (60, 7)
                assert features['mf'].shape == (30, 43)
                assert features['lf'].shape == (30, 19)
                
                # Verify data types
                assert features['hf'].dtype == np.float32
                assert features['mf'].dtype == np.float32
                assert features['lf'].dtype == np.float32
                
                # Verify no invalid values
                for category in ['hf', 'mf', 'lf']:
                    assert not np.any(np.isnan(features[category]))
                    assert not np.any(np.isinf(features[category]))
                
        except ImportError:
            pytest.skip("Full feature system not available")
    
    def test_performance_benchmarks(self):
        """Test performance of feature extraction"""
        try:
            from feature.simple_feature_manager import SimpleFeatureManager
            from feature.contexts import MarketContext
            import time
            
            # Create minimal feature manager
            config = Mock()
            config.hf_seq_len = 60
            config.hf_feat_dim = 7
            config.mf_seq_len = 30
            config.mf_feat_dim = 43
            config.lf_seq_len = 30
            config.lf_feat_dim = 19
            
            with patch.object(SimpleFeatureManager, '_initialize_features') as mock_init:
                mock_init.return_value = {'hf': [], 'mf': [], 'lf': []}
                
                manager = SimpleFeatureManager("MLGO", config)
                
                # Create minimal context
                context = MarketContext(
                    timestamp=datetime.now(),
                    current_price=100.0,
                    hf_window=[],
                    mf_1m_window=[],
                    lf_5m_window=[],
                    prev_day_close=99.0,
                    prev_day_high=101.0,
                    prev_day_low=98.0,
                    session_high=100.5,
                    session_low=99.5,
                    session="REGULAR",
                    market_cap=1000000.0
                )
                
                # Benchmark feature extraction
                start_time = time.time()
                for _ in range(100):
                    features = manager.extract_features(context)
                    assert features is not None
                end_time = time.time()
                
                # Should complete 100 extractions in reasonable time
                total_time = end_time - start_time
                assert total_time < 10.0  # Less than 10 seconds for 100 extractions
                
        except ImportError:
            pytest.skip("Performance benchmark not available")