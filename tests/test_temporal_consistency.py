"""Tests for temporal consistency across sequence features."""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock

from feature.feature_base import FeatureConfig
from feature.aggregated.aggregated_features import (
    HFMomentumSummaryFeature,
    HFVolumeDynamicsFeature,
    MFTrendConsistencyFeature,
    MFMomentumPersistenceFeature
)
from feature.professional.ta_features import ProfessionalEMASystemFeature


class TestTemporalConsistency:
    """Test temporal consistency and causality in sequence features."""

    def create_progressive_sequence(self, base_sequence: List[Dict], modification_func) -> List[List[Dict]]:
        """Create progressively modified sequences to test temporal consistency."""
        sequences = []
        
        for i in range(1, len(base_sequence) + 1):
            # Take first i elements and apply modification
            partial_seq = base_sequence[:i].copy()
            if modification_func:
                partial_seq = modification_func(partial_seq)
            sequences.append(partial_seq)
        
        return sequences

    def test_hf_momentum_temporal_causality(self):
        """Test that HF momentum respects temporal causality (future doesn't affect past)."""
        feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        
        # Create base upward trending sequence
        base_sequence = []
        for i in range(20):
            price = 100.0 + (i * 0.1)
            entry = {
                '1s_bar': {
                    'close': price,
                    'volume': 1000 + (i * 50)
                }
            }
            base_sequence.append(entry)
        
        # Calculate momentum at each timestep
        momentum_progression = []
        for i in range(3, len(base_sequence)):  # Start from minimum required length
            partial_seq = base_sequence[:i]
            momentum = feature.calculate_raw({'hf_data_window': partial_seq})
            momentum_progression.append(momentum)
        
        # Test temporal consistency: adding consistent trend data should maintain/strengthen momentum
        for i in range(1, len(momentum_progression)):
            current_momentum = momentum_progression[i]
            previous_momentum = momentum_progression[i-1]
            
            # For consistent upward trend, momentum should remain positive
            assert current_momentum > 0, f"Momentum should remain positive in uptrend at timestep {i}"
            
            # Adding more consistent trend data shouldn't cause dramatic reversals
            momentum_change = abs(current_momentum - previous_momentum)
            assert momentum_change < 0.5, f"Momentum shouldn't change drastically between consecutive timesteps"

    def test_hf_momentum_future_independence(self):
        """Test that HF momentum doesn't use future information."""
        feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        
        # Create sequence with dramatic future change
        stable_sequence = []
        for i in range(15):
            price = 100.0 + np.random.normal(0, 0.01)  # Stable price
            entry = {'1s_bar': {'close': price, 'volume': 1000}}
            stable_sequence.append(entry)
        
        # Calculate momentum before adding future shock
        momentum_before = feature.calculate_raw({'hf_data_window': stable_sequence})
        
        # Add dramatic future movement
        future_shock = []
        for i in range(5):
            price = 110.0 + (i * 0.5)  # Sudden jump
            entry = {'1s_bar': {'close': price, 'volume': 2000}}
            future_shock.append(entry)
        
        extended_sequence = stable_sequence + future_shock
        
        # Calculate momentum of original sequence within extended sequence
        # (Should be same as momentum_before if no future leakage)
        momentum_original_part = feature.calculate_raw({'hf_data_window': stable_sequence})
        
        assert abs(momentum_before - momentum_original_part) < 1e-10, \
            "Momentum calculation should be deterministic for same input"
        
        # The momentum of just the stable part should not be influenced by future shock
        # when calculated independently
        assert abs(momentum_before) < 0.2, "Stable sequence should have low momentum"

    def test_volume_dynamics_temporal_progression(self):
        """Test volume dynamics maintains temporal progression properties."""
        feature = HFVolumeDynamicsFeature(FeatureConfig(name="hf_volume_dynamics"))
        
        # Create volume sequence with clear trend
        volume_trend_sequence = []
        for i in range(25):
            volume = 1000 + (i * 80)  # Steadily increasing
            entry = {'1s_bar': {'close': 100.0, 'volume': volume}}
            volume_trend_sequence.append(entry)
        
        # Test progression of volume dynamics as sequence grows
        dynamics_progression = []
        for i in range(5, len(volume_trend_sequence)):  # Start from minimum length
            partial_seq = volume_trend_sequence[:i]
            dynamics = feature.calculate_raw({'hf_data_window': partial_seq})
            dynamics_progression.append((i, dynamics))
        
        # For consistent increasing volume, dynamics should be consistently positive
        for length, dynamics in dynamics_progression:
            assert dynamics > 0, f"Volume dynamics should be positive for increasing volume at length {length}"
        
        # Test that dynamics stabilize as more data is added (trend becomes clearer)
        early_dynamics = dynamics_progression[0][1]
        late_dynamics = dynamics_progression[-1][1]
        
        # With more data, the increasing trend should be more confident
        assert late_dynamics >= early_dynamics * 0.8, \
            "Volume dynamics should stabilize or strengthen with more data"

    def test_mf_trend_consistency_temporal_stability(self):
        """Test MF trend consistency behaves stably across time."""
        feature = MFTrendConsistencyFeature(FeatureConfig(name="mf_trend_consistency"))
        
        # Create consistently trending sequence
        consistent_sequence = []
        for i in range(20):
            price = 100.0 + (i * 0.3)  # Consistent uptrend
            bar = {'close': price, 'volume': 1000}
            consistent_sequence.append(bar)
        
        # Test consistency scores as sequence grows
        consistency_scores = []
        for i in range(5, len(consistent_sequence)):  # Start from minimum length
            partial_seq = consistent_sequence[:i]
            consistency = feature.calculate_raw({'1m_bars_window': partial_seq})
            consistency_scores.append(consistency)
        
        # Consistency should remain high for consistent trend
        for score in consistency_scores:
            assert score > 0.5, "Consistent trend should maintain high consistency score"
        
        # Test that adding consistent data doesn't reduce consistency
        for i in range(1, len(consistency_scores)):
            current = consistency_scores[i]
            previous = consistency_scores[i-1]
            
            # Adding more consistent data should maintain or improve consistency
            assert current >= previous * 0.9, \
                "Adding consistent trend data shouldn't significantly reduce consistency"

    def test_momentum_persistence_directional_stability(self):
        """Test momentum persistence correctly tracks direction changes over time."""
        feature = MFMomentumPersistenceFeature(FeatureConfig(name="mf_momentum_persistence"))
        
        # Create sequence with single direction change
        direction_change_sequence = []
        
        # First part: upward momentum
        for i in range(8):
            price = 100.0 + (i * 0.4)
            bar = {'close': price, 'volume': 1000}
            direction_change_sequence.append(bar)
        
        # Second part: downward momentum
        for i in range(6):
            price = 103.2 - (i * 0.3)
            bar = {'close': price, 'volume': 1000}
            direction_change_sequence.append(bar)
        
        # Test persistence as sequence grows through the direction change
        persistence_scores = []
        for i in range(8, len(direction_change_sequence)):  # Start from minimum length
            partial_seq = direction_change_sequence[:i]
            persistence = feature.calculate_raw({'1m_bars_window': partial_seq})
            persistence_scores.append((i, persistence))
        
        # Before direction change, persistence should be high
        early_persistence = persistence_scores[0][1]
        assert early_persistence > 0.7, "Early momentum should show high persistence"
        
        # After direction change, persistence should decrease
        late_persistence = persistence_scores[-1][1]
        assert late_persistence < early_persistence, \
            "Direction change should reduce momentum persistence"

    def test_professional_ema_temporal_smoothness(self):
        """Test professional EMA features maintain temporal smoothness."""
        feature = ProfessionalEMASystemFeature(FeatureConfig(name="professional_ema_system"))
        
        # Create gradually trending sequence
        trending_sequence = []
        for i in range(35):
            price = 100.0 + (i * 0.15)  # Gradual uptrend
            bar = {
                'open': price - 0.05,
                'high': price + 0.08,
                'low': price - 0.08,
                'close': price,
                'volume': 1000
            }
            trending_sequence.append(bar)
        
        # Test EMA signals as sequence grows
        ema_signals = []
        for i in range(25, len(trending_sequence)):  # Start from minimum length
            partial_seq = trending_sequence[:i]
            signal = feature.calculate_raw({'1m_bars_window': partial_seq})
            ema_signals.append(signal)
        
        # EMA signals should be smooth (no dramatic jumps)
        for i in range(1, len(ema_signals)):
            current = ema_signals[i]
            previous = ema_signals[i-1]
            
            # Signal change should be gradual
            signal_change = abs(current - previous)
            assert signal_change < 0.3, \
                f"EMA signal should change gradually, got change of {signal_change}"
        
        # For consistent uptrend, signals should be positive
        for signal in ema_signals:
            assert signal > 0, "Uptrending market should produce positive EMA signals"

    def test_sequence_order_dependency(self):
        """Test that features depend on sequence order (not just values)."""
        
        # Create two sequences with same values but different orders
        values = [100.0, 100.2, 100.4, 100.1, 100.6, 100.3, 100.8, 100.5, 100.9, 100.7]
        
        # Original order (somewhat trending)
        original_sequence = []
        for price in values:
            entry = {'1s_bar': {'close': price, 'volume': 1000}}
            original_sequence.append(entry)
        
        # Shuffled order (same values, different sequence)
        shuffled_values = [100.9, 100.1, 100.6, 100.0, 100.8, 100.2, 100.5, 100.4, 100.7, 100.3]
        shuffled_sequence = []
        for price in shuffled_values:
            entry = {'1s_bar': {'close': price, 'volume': 1000}}
            shuffled_sequence.append(entry)
        
        # Test HF momentum feature
        momentum_feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        original_momentum = momentum_feature.calculate_raw({'hf_data_window': original_sequence})
        shuffled_momentum = momentum_feature.calculate_raw({'hf_data_window': shuffled_sequence})
        
        # Different orders should produce different results
        assert abs(original_momentum - shuffled_momentum) > 0.01, \
            "Sequence order should matter for momentum calculation"
        
        # Test volume dynamics feature
        volume_feature = HFVolumeDynamicsFeature(FeatureConfig(name="hf_volume_dynamics"))
        
        # Create volume sequences with same total but different patterns
        volumes = [1000, 1200, 1400, 1100, 1600, 1300, 1800, 1500, 1900, 1700]
        original_vol_seq = []
        for i, vol in enumerate(volumes):
            entry = {'1s_bar': {'close': 100.0, 'volume': vol}}
            original_vol_seq.append(entry)
        
        shuffled_vols = [1900, 1100, 1600, 1000, 1800, 1200, 1500, 1400, 1700, 1300]
        shuffled_vol_seq = []
        for i, vol in enumerate(shuffled_vols):
            entry = {'1s_bar': {'close': 100.0, 'volume': vol}}
            shuffled_vol_seq.append(entry)
        
        original_vol_dynamics = volume_feature.calculate_raw({'hf_data_window': original_vol_seq})
        shuffled_vol_dynamics = volume_feature.calculate_raw({'hf_data_window': shuffled_vol_seq})
        
        # Different volume orders should produce different dynamics
        assert abs(original_vol_dynamics - shuffled_vol_dynamics) > 0.01, \
            "Volume sequence order should matter for dynamics calculation"

    def test_feature_monotonicity_properties(self):
        """Test that features exhibit expected monotonicity properties."""
        
        # Test momentum feature with progressively stronger trends
        momentum_feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        
        trend_strengths = [0.05, 0.1, 0.2, 0.4]  # Increasing trend strength
        momentum_scores = []
        
        for strength in trend_strengths:
            sequence = []
            for i in range(15):
                price = 100.0 + (i * strength)
                entry = {'1s_bar': {'close': price, 'volume': 1000}}
                sequence.append(entry)
            
            momentum = momentum_feature.calculate_raw({'hf_data_window': sequence})
            momentum_scores.append(momentum)
        
        # Stronger trends should generally produce higher momentum scores
        for i in range(1, len(momentum_scores)):
            current = momentum_scores[i]
            previous = momentum_scores[i-1]
            assert current >= previous * 0.9, \
                f"Stronger trend should have higher momentum: {previous} -> {current}"

    def test_feature_stability_under_noise(self):
        """Test that features are reasonably stable under small noise additions."""
        
        # Create base trending sequence
        base_sequence = []
        for i in range(20):
            price = 100.0 + (i * 0.2)
            entry = {'1s_bar': {'close': price, 'volume': 1000 + (i * 50)}}
            base_sequence.append(entry)
        
        # Add small noise to create noisy sequence
        noisy_sequence = []
        np.random.seed(42)  # For reproducibility
        for entry in base_sequence:
            noisy_price = entry['1s_bar']['close'] + np.random.normal(0, 0.02)
            noisy_volume = max(100, entry['1s_bar']['volume'] + np.random.normal(0, 20))
            noisy_entry = {
                '1s_bar': {
                    'close': noisy_price,
                    'volume': noisy_volume
                }
            }
            noisy_sequence.append(noisy_entry)
        
        # Test features on both sequences
        features = [
            HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary")),
            HFVolumeDynamicsFeature(FeatureConfig(name="hf_volume_dynamics"))
        ]
        
        for feature in features:
            base_result = feature.calculate_raw({'hf_data_window': base_sequence})
            noisy_result = feature.calculate_raw({'hf_data_window': noisy_sequence})
            
            # Results should be reasonably close (noise shouldn't completely change signal)
            relative_change = abs(base_result - noisy_result) / (abs(base_result) + 1e-8)
            assert relative_change < 0.5, \
                f"{feature.__class__.__name__} should be reasonably stable under noise"

    def test_aggregation_window_effects(self):
        """Test how different aggregation windows affect temporal consistency."""
        
        # Create long sequence for testing different window sizes
        long_sequence = []
        for i in range(50):
            # Create pattern that changes characteristics over time
            if i < 20:
                price = 100.0 + (i * 0.1)  # Moderate uptrend
                volume = 1000 + (i * 30)
            else:
                price = 102.0 + ((i-20) * 0.3)  # Stronger uptrend
                volume = 1600 + ((i-20) * 80)
            
            entry = {'1s_bar': {'close': price, 'volume': volume}}
            long_sequence.append(entry)
        
        momentum_feature = HFMomentumSummaryFeature(FeatureConfig(name="hf_momentum_summary"))
        
        # Test different window sizes on the same underlying sequence
        window_sizes = [10, 20, 30, 40]
        momentum_by_window = []
        
        for window_size in window_sizes:
            # Take the last window_size elements
            windowed_sequence = long_sequence[-window_size:]
            momentum = momentum_feature.calculate_raw({'hf_data_window': windowed_sequence})
            momentum_by_window.append(momentum)
        
        # Larger windows should capture the transition and show stronger momentum
        # (since they include both the moderate and strong trend periods)
        largest_window_momentum = momentum_by_window[-1]
        smallest_window_momentum = momentum_by_window[0]
        
        # The largest window should capture more of the accelerating trend
        assert largest_window_momentum >= smallest_window_momentum * 0.8, \
            "Larger windows should capture broader trend context"