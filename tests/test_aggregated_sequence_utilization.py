"""Tests to verify aggregated features properly utilize full sequence windows."""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from unittest.mock import Mock

from feature.feature_base import FeatureConfig
from feature.aggregated.aggregated_features import (
    HFMomentumSummaryFeature,
    HFVolumeDynamicsFeature,
    HFMicrostructureQualityFeature,
    MFTrendConsistencyFeature,
    MFVolumePriceDivergenceFeature,
    MFMomentumPersistenceFeature
)


class TestAggregatedSequenceUtilization:
    """Test that aggregated features efficiently use entire sequence windows."""

    def create_test_hf_sequence(self, length: int, pattern: str = 'random') -> List[Dict[str, Any]]:
        """Create HF sequence with specific patterns for testing."""
        sequence = []
        
        for i in range(length):
            if pattern == 'momentum_buildup':
                # Gradual momentum increase
                price = 100.0 + (i ** 1.5) * 0.01
                volume = 1000 + (i * 50)
            elif pattern == 'momentum_fade':
                # Strong early momentum that fades
                momentum_factor = max(0.1, (length - i) / length)
                price = 100.0 + (momentum_factor * i * 0.05)
                volume = 1000 + (momentum_factor * i * 100)
            elif pattern == 'volatile':
                # High volatility with reversals
                price = 100.0 + ((-1) ** i) * 0.2 * (i % 3)
                volume = 1000 + np.random.normal(0, 300)
            elif pattern == 'quality_degradation':
                # Market quality degrades over time
                spread_factor = 1 + (i * 0.1)
                price = 100.0 + np.random.normal(0, 0.05)
                volume = max(100, 1000 - (i * 30))
                
                # Wider spreads over time
                bid = price - (0.01 * spread_factor)
                ask = price + (0.01 * spread_factor)
            else:  # random
                price = 100.0 + np.random.normal(0, 0.1)
                volume = max(100, 1000 + np.random.normal(0, 200))
            
            # Default spread if not set above
            if pattern != 'quality_degradation':
                bid = price - 0.005
                ask = price + 0.005
            
            entry = {
                '1s_bar': {
                    'open': price - 0.002,
                    'high': price + 0.003,
                    'low': price - 0.003,
                    'close': price,
                    'volume': volume
                },
                'quotes': [{
                    'bid_px': bid,
                    'ask_px': ask,
                    'bid_sz': max(100, 1000 - abs(np.random.normal(0, 200))),
                    'ask_sz': max(100, 1000 - abs(np.random.normal(0, 200)))
                }]
            }
            sequence.append(entry)
            
        return sequence

    def create_test_mf_sequence(self, length: int, pattern: str = 'random') -> List[Dict[str, Any]]:
        """Create MF sequence with specific patterns."""
        sequence = []
        base_price = 100.0
        
        for i in range(length):
            if pattern == 'consistent_trend':
                # Consistent upward trend
                price = base_price + (i * 0.3)
                volume = 5000 + (i * 100)
            elif pattern == 'trend_reversal':
                # Trend that reverses halfway
                if i < length // 2:
                    price = base_price + (i * 0.4)
                else:
                    price = base_price + ((length // 2) * 0.4) - ((i - length // 2) * 0.3)
                volume = 5000 + np.random.normal(0, 500)
            elif pattern == 'volume_divergence':
                # Price up but volume down (bearish divergence)
                price = base_price + (i * 0.2)
                volume = max(1000, 8000 - (i * 200))
            elif pattern == 'choppy':
                # Choppy, inconsistent moves
                direction = (-1) ** (i // 2)
                price = base_price + (direction * (i % 3) * 0.3)
                volume = 5000 + np.random.normal(0, 1000)
            else:  # random
                price = base_price + np.random.normal(0, 0.2)
                volume = max(1000, 5000 + np.random.normal(0, 1000))
            
            bar = {
                'open': price - 0.05,
                'high': price + 0.08,
                'low': price - 0.08,
                'close': price,
                'volume': volume
            }
            sequence.append(bar)
            
        return sequence

    def test_hf_momentum_full_window_utilization(self):
        """Test HF momentum summary uses entire window effectively."""
        feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        
        # Test different sequence patterns
        buildup_seq = self.create_test_hf_sequence(30, 'momentum_buildup')
        fade_seq = self.create_test_hf_sequence(30, 'momentum_fade')
        volatile_seq = self.create_test_hf_sequence(30, 'volatile')
        
        buildup_score = feature.calculate_raw({'hf_data_window': buildup_seq})
        fade_score = feature.calculate_raw({'hf_data_window': fade_seq})
        volatile_score = feature.calculate_raw({'hf_data_window': volatile_seq})
        
        # Momentum buildup should score highest (recent weighted momentum)
        assert buildup_score > fade_score, "Momentum buildup should score higher than fade"
        assert buildup_score > volatile_score, "Momentum buildup should score higher than volatile"
        
        # Test that feature truly uses full window by comparing different window sizes
        short_buildup = self.create_test_hf_sequence(10, 'momentum_buildup')
        medium_buildup = self.create_test_hf_sequence(20, 'momentum_buildup')
        long_buildup = self.create_test_hf_sequence(40, 'momentum_buildup')
        
        short_score = feature.calculate_raw({'hf_data_window': short_buildup})
        medium_score = feature.calculate_raw({'hf_data_window': medium_buildup})
        long_score = feature.calculate_raw({'hf_data_window': long_buildup})
        
        # Longer sequences should provide different (potentially stronger) signals
        assert not (short_score == medium_score == long_score), "Different window sizes should yield different results"

    def test_hf_volume_dynamics_sequence_analysis(self):
        """Test volume dynamics analyzes patterns across entire sequence."""
        feature = HFVolumeDynamicsFeature(FeatureConfig(name="hf_volume_dynamics"))
        
        # Create sequences with different volume patterns
        increasing_vol_seq = []
        decreasing_vol_seq = []
        stable_vol_seq = []
        
        for i in range(25):
            # Increasing volume
            inc_volume = 1000 + (i * 100)
            inc_entry = {'1s_bar': {'close': 100.0, 'volume': inc_volume}}
            increasing_vol_seq.append(inc_entry)
            
            # Decreasing volume
            dec_volume = max(200, 3500 - (i * 120))
            dec_entry = {'1s_bar': {'close': 100.0, 'volume': dec_volume}}
            decreasing_vol_seq.append(dec_entry)
            
            # Stable volume
            stable_volume = 2000 + np.random.normal(0, 50)
            stable_entry = {'1s_bar': {'close': 100.0, 'volume': max(500, stable_volume)}}
            stable_vol_seq.append(stable_entry)
        
        inc_dynamics = feature.calculate_raw({'hf_data_window': increasing_vol_seq})
        dec_dynamics = feature.calculate_raw({'hf_data_window': decreasing_vol_seq})
        stable_dynamics = feature.calculate_raw({'hf_data_window': stable_vol_seq})
        
        # Increasing volume should have positive dynamics
        assert inc_dynamics > 0, "Increasing volume should have positive dynamics"
        
        # Decreasing volume should have negative dynamics
        assert dec_dynamics < 0, "Decreasing volume should have negative dynamics"
        
        # Test correlation calculation uses full sequence
        # Volume trend should be detected across the full window
        assert inc_dynamics > stable_dynamics, "Increasing volume should score higher than stable"
        assert stable_dynamics > dec_dynamics, "Stable volume should score higher than decreasing"
        
        # Verify the feature actually calculates trends (not just averages)
        # This tests that np.corrcoef is being used properly across the sequence
        flat_vol_seq = [{'1s_bar': {'close': 100.0, 'volume': 2000}} for _ in range(25)]
        flat_dynamics = feature.calculate_raw({'hf_data_window': flat_vol_seq})
        
        # Flat volume should have minimal trend component
        assert abs(flat_dynamics) < abs(inc_dynamics), "Flat volume should have weaker dynamics than trending"

    def test_microstructure_quality_aggregation(self):
        """Test microstructure quality aggregates across full sequence."""
        feature = HFMicrostructureQualityFeature(FeatureConfig(name="hf_microstructure_quality"))
        
        # High quality sequence (consistent tight spreads, many quotes)
        high_qual_seq = self.create_test_hf_sequence(20, 'random')
        for entry in high_qual_seq:
            # Override with tight, consistent spreads
            entry['quotes'] = [
                {'bid_px': 99.995, 'ask_px': 100.005},  # 0.01 spread
                {'bid_px': 99.996, 'ask_px': 100.004},  # Multiple quotes
                {'bid_px': 99.994, 'ask_px': 100.006}
            ]
        
        # Degrading quality sequence
        degrad_qual_seq = self.create_test_hf_sequence(20, 'quality_degradation')
        
        # Poor quality sequence (wide, unstable spreads)
        poor_qual_seq = []
        for i in range(20):
            spread_width = 0.05 + (i * 0.01)  # Widening spreads
            entry = {
                'quotes': [{
                    'bid_px': 100.0 - spread_width,
                    'ask_px': 100.0 + spread_width
                }]
            }
            poor_qual_seq.append(entry)
        
        high_quality = feature.calculate_raw({'hf_data_window': high_qual_seq})
        degrad_quality = feature.calculate_raw({'hf_data_window': degrad_qual_seq})
        poor_quality = feature.calculate_raw({'hf_data_window': poor_qual_seq})
        
        # Quality ranking should reflect aggregated metrics
        assert high_quality > degrad_quality, "High quality should score better than degrading"
        assert degrad_quality > poor_quality, "Degrading quality should score better than poor"
        
        # Test that spread stability calculation uses full sequence
        # Feature should detect spread volatility across the entire window
        volatile_spread_seq = []
        for i in range(20):
            # Alternating tight/wide spreads
            if i % 2 == 0:
                spread = 0.01  # Tight
            else:
                spread = 0.08  # Wide
            
            entry = {
                'quotes': [{
                    'bid_px': 100.0 - spread,
                    'ask_px': 100.0 + spread
                }]
            }
            volatile_spread_seq.append(entry)
        
        volatile_quality = feature.calculate_raw({'hf_data_window': volatile_spread_seq})
        
        # Volatile spreads should score poorly due to stability component
        assert poor_quality > volatile_quality, "Consistent poor quality should beat volatile quality"

    def test_mf_trend_consistency_full_analysis(self):
        """Test MF trend consistency analyzes entire window for patterns."""
        feature = MFTrendConsistencyFeature(FeatureConfig(name="mf_trend_consistency"))
        
        # Test different trend patterns
        consistent_seq = self.create_test_mf_sequence(15, 'consistent_trend')
        reversal_seq = self.create_test_mf_sequence(15, 'trend_reversal')
        choppy_seq = self.create_test_mf_sequence(15, 'choppy')
        
        consistent_score = feature.calculate_raw({'1m_bars_window': consistent_seq})
        reversal_score = feature.calculate_raw({'1m_bars_window': reversal_seq})
        choppy_score = feature.calculate_raw({'1m_bars_window': choppy_seq})
        
        # Consistent trend should score highest
        assert consistent_score > reversal_score, "Consistent trend should score higher than reversal"
        assert consistent_score > choppy_score, "Consistent trend should score higher than choppy"
        
        # Test that full window is used for consistency calculation
        # Feature should detect that reversals break consistency
        assert reversal_score < consistent_score, "Trend reversals should reduce consistency score"
        
        # Verify the feature looks at return direction ratios across full sequence
        # This tests the pos_ratio/neg_ratio calculation uses all returns
        all_positive_seq = []
        for i in range(12):
            price = 100.0 + (i * 0.2)  # Monotonically increasing
            bar = {'open': price-0.1, 'high': price+0.1, 'low': price-0.1, 'close': price, 'volume': 1000}
            all_positive_seq.append(bar)
        
        all_positive_score = feature.calculate_raw({'1m_bars_window': all_positive_seq})
        
        # All positive moves should have maximum consistency
        assert all_positive_score > consistent_score, "All positive moves should have highest consistency"
        assert all_positive_score > 0.8, "All positive moves should score very high"

    def test_volume_price_divergence_full_correlation(self):
        """Test volume-price divergence uses full sequence for correlation."""
        feature = MFVolumePriceDivergenceFeature(FeatureConfig(name="mf_volume_price_divergence"))
        
        # Test different volume-price relationships
        confirming_seq = self.create_test_mf_sequence(12, 'consistent_trend')  # Price up, volume up
        diverging_seq = self.create_test_mf_sequence(12, 'volume_divergence')  # Price up, volume down
        
        confirming_score = feature.calculate_raw({'1m_bars_window': confirming_seq})
        diverging_score = feature.calculate_raw({'1m_bars_window': diverging_seq})
        
        # Confirming volume should score positively, diverging negatively
        assert confirming_score > 0, "Volume confirmation should score positively"
        assert diverging_score < 0, "Volume divergence should score negatively"
        
        # Test correlation calculation uses full sequence
        # Create perfect positive correlation
        perfect_pos_seq = []
        for i in range(10):
            price = 100.0 + (i * 0.3)
            volume = 5000 + (i * 300)  # Volume increases with price
            bar = {'close': price, 'volume': volume}
            perfect_pos_seq.append(bar)
        
        # Create perfect negative correlation  
        perfect_neg_seq = []
        for i in range(10):
            price = 100.0 + (i * 0.3)
            volume = 8000 - (i * 300)  # Volume decreases as price increases
            bar = {'close': price, 'volume': volume}
            perfect_neg_seq.append(bar)
        
        perfect_pos_score = feature.calculate_raw({'1m_bars_window': perfect_pos_seq})
        perfect_neg_score = feature.calculate_raw({'1m_bars_window': perfect_neg_seq})
        
        # Perfect correlations should be stronger than typical patterns
        assert abs(perfect_pos_score) > abs(confirming_score), "Perfect correlation should be stronger"
        assert abs(perfect_neg_score) > abs(diverging_score), "Perfect negative correlation should be stronger"
        
        # Test mid-point calculation uses full sequence
        # The feature splits the sequence to compare early vs late periods
        assert perfect_pos_score > 0, "Perfect positive correlation should be positive"
        assert perfect_neg_score < 0, "Perfect negative correlation should be negative"

    def test_momentum_persistence_rolling_analysis(self):
        """Test momentum persistence analyzes rolling windows across full sequence."""
        feature = MFMomentumPersistenceFeature(FeatureConfig(name="mf_momentum_persistence"))
        
        # Persistent momentum (same direction throughout)
        persistent_seq = []
        for i in range(15):
            price = 100.0 + (i * 0.25)  # Steady upward momentum
            bar = {'close': price, 'volume': 1000}
            persistent_seq.append(bar)
        
        # Reversing momentum (changes direction frequently)
        reversing_seq = []
        for i in range(15):
            # Change direction every 3 bars
            cycle_pos = i % 6
            if cycle_pos < 3:
                price = 100.0 + (cycle_pos * 0.3)
            else:
                price = 100.9 - ((cycle_pos - 3) * 0.3)
            bar = {'close': price, 'volume': 1000}
            reversing_seq.append(bar)
        
        persistent_score = feature.calculate_raw({'1m_bars_window': persistent_seq})
        reversing_score = feature.calculate_raw({'1m_bars_window': reversing_seq})
        
        # Persistent momentum should score higher
        assert persistent_score > reversing_score, "Persistent momentum should score higher than reversing"
        
        # Test rolling window calculation
        # Feature should detect direction changes across rolling 3-bar windows
        single_reversal_seq = []
        for i in range(12):
            if i < 6:
                price = 100.0 + (i * 0.2)  # Up for first half
            else:
                price = 101.0 - ((i - 6) * 0.15)  # Down for second half
            bar = {'close': price, 'volume': 1000}
            single_reversal_seq.append(bar)
        
        single_reversal_score = feature.calculate_raw({'1m_bars_window': single_reversal_seq})
        
        # Single reversal should be between persistent and frequent reversing
        assert persistent_score > single_reversal_score > reversing_score, \
            "Single reversal should score between persistent and frequent reversing"

    def test_sequence_utilization_efficiency(self):
        """Test that aggregated features efficiently use all available data points."""
        
        features = [
            HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary")),
            HFVolumeDynamicsFeature(FeatureConfig(name="hf_volume_dynamics")),
            HFMicrostructureQualityFeature(FeatureConfig(name="hf_microstructure_quality")),
            MFTrendConsistencyFeature(FeatureConfig(name="mf_trend_consistency")),
            MFVolumePriceDivergenceFeature(FeatureConfig(name="mf_volume_price_divergence")),
            MFMomentumPersistenceFeature(FeatureConfig(name="mf_momentum_persistence"))
        ]
        
        # Test that features produce different results with different sequence lengths
        for feature in features:
            if 'HF' in feature.__class__.__name__:
                # Test HF features
                short_seq = self.create_test_hf_sequence(5, 'momentum_buildup')
                medium_seq = self.create_test_hf_sequence(15, 'momentum_buildup')
                long_seq = self.create_test_hf_sequence(30, 'momentum_buildup')
                
                short_result = feature.calculate_raw({'hf_data_window': short_seq})
                medium_result = feature.calculate_raw({'hf_data_window': medium_seq})
                long_result = feature.calculate_raw({'hf_data_window': long_seq})
                
                data_key = 'hf_data_window'
            else:
                # Test MF features
                short_seq = self.create_test_mf_sequence(6, 'consistent_trend')
                medium_seq = self.create_test_mf_sequence(12, 'consistent_trend')
                long_seq = self.create_test_mf_sequence(20, 'consistent_trend')
                
                short_result = feature.calculate_raw({'1m_bars_window': short_seq})
                medium_result = feature.calculate_raw({'1m_bars_window': medium_seq})
                long_result = feature.calculate_raw({'1m_bars_window': long_seq})
                
                data_key = '1m_bars_window'
            
            # At least one should be different (proving sequence length matters)
            results = [short_result, medium_result, long_result]
            assert len(set(results)) > 1 or short_result == 0.0, \
                f"{feature.__class__.__name__} should utilize different sequence lengths differently"
            
            # All results should be finite and in bounds
            for result in results:
                assert np.isfinite(result), f"{feature.__class__.__name__} should return finite values"
                assert -1.0 <= result <= 1.0, f"{feature.__class__.__name__} should respect bounds"

    def test_aggregation_vs_point_in_time_comparison(self):
        """Verify aggregated features provide richer information than point-in-time equivalents."""
        
        # Create a sequence where the current state differs from the trend
        misleading_current_state_seq = []
        
        # Strong upward trend for most of sequence
        for i in range(18):
            price = 100.0 + (i * 0.4)
            volume = 1000 + (i * 200)
            entry = {
                '1s_bar': {
                    'close': price,
                    'volume': volume
                }
            }
            misleading_current_state_seq.append(entry)
        
        # Sudden drop at the end (misleading current state)
        final_entries = []
        for i in range(2):
            price = 107.0 - (i * 0.3)  # Sharp drop
            volume = 500  # Low volume
            entry = {
                '1s_bar': {
                    'close': price,
                    'volume': volume
                }
            }
            final_entries.append(entry)
        
        misleading_current_state_seq.extend(final_entries)
        
        # Test HF momentum summary (should detect overall upward trend despite recent drop)
        momentum_feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        momentum_score = momentum_feature.calculate_raw({'hf_data_window': misleading_current_state_seq})
        
        # Despite recent drop, overall trend should still be positive due to aggregation
        assert momentum_score > 0, "Aggregated momentum should detect overall upward trend"
        
        # Test volume dynamics (should detect overall increasing volume pattern)
        volume_feature = HFVolumeDynamicsFeature(FeatureConfig(name="hf_volume_dynamics"))
        volume_score = volume_feature.calculate_raw({'hf_data_window': misleading_current_state_seq})
        
        # Should detect the overall volume increase pattern
        assert volume_score > 0, "Aggregated volume dynamics should detect overall increasing pattern"
        
        # Point-in-time comparison: if we only looked at last 2 entries, we'd see decline
        recent_only_seq = final_entries
        recent_momentum = momentum_feature.calculate_raw({'hf_data_window': recent_only_seq})
        
        # Recent-only should be negative or zero (due to minimum length requirements)
        assert recent_momentum <= 0, "Recent-only view should show decline"
        
        # This proves aggregated features capture broader context that point-in-time misses
        assert momentum_score > recent_momentum, "Full sequence aggregation should differ from point-in-time"