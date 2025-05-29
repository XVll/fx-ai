"""Tests to validate sequence features produce correct temporal values and sequences."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from typing import Dict, Any, List

from feature.feature_manager import FeatureManager
from feature.feature_base import FeatureConfig
from feature.aggregated.aggregated_features import (
    HFMomentumSummaryFeature,
    HFVolumeDynamicsFeature, 
    HFMicrostructureQualityFeature,
    MFTrendConsistencyFeature,
    MFVolumePriceDivergenceFeature,
    MFMomentumPersistenceFeature
)
from feature.professional.ta_features import ProfessionalEMASystemFeature


class TestSequenceFeatureValidation:
    """Test that sequence features properly utilize temporal data."""

    def create_hf_sequence(self, length: int, trend: str = 'flat') -> List[Dict[str, Any]]:
        """Create HF data sequence with specific characteristics."""
        sequence = []
        base_price = 100.0
        base_volume = 1000
        
        for i in range(length):
            # Price trend - make more pronounced
            if trend == 'up':
                price = base_price + (i * 0.3)  # Stronger uptrend
            elif trend == 'down':
                price = base_price - (i * 0.3)  # Stronger downtrend
            elif trend == 'volatile':
                price = base_price + ((-1) ** i) * 0.5
            else:  # flat
                price = base_price + np.random.normal(0, 0.02)
            
            # Volume trend
            if trend == 'up':
                volume = base_volume + (i * 50)
            elif trend == 'down':
                volume = max(100, base_volume - (i * 30))
            else:
                volume = base_volume + np.random.normal(0, 50)
                
            # Quotes
            spread = 0.01
            bid = price - spread/2
            ask = price + spread/2
            
            entry = {
                '1s_bar': {
                    'open': price - 0.005,
                    'high': price + 0.005,
                    'low': price - 0.005,
                    'close': price,
                    'volume': volume
                },
                'quotes': [{
                    'bid_px': bid,
                    'ask_px': ask,
                    'bid_sz': 100,
                    'ask_sz': 100
                }]
            }
            sequence.append(entry)
            
        return sequence

    def create_mf_sequence(self, length: int, trend: str = 'flat') -> List[Dict[str, Any]]:
        """Create MF data sequence (1m bars)."""
        sequence = []
        base_price = 100.0
        base_volume = 5000
        
        for i in range(length):
            # Price progression
            if trend == 'up':
                price = base_price + (i * 0.5)
            elif trend == 'down':
                price = base_price - (i * 0.5)
            elif trend == 'choppy':
                price = base_price + ((-1) ** i) * 0.3
            else:  # flat
                price = base_price + np.random.normal(0, 0.1)
                
            # Volume progression
            if trend == 'up':
                volume = base_volume + (i * 200)
            else:
                volume = base_volume + np.random.normal(0, 500)
            
            bar = {
                'open': price - 0.02,
                'high': price + 0.02,
                'low': price - 0.02,
                'close': price,
                'volume': max(100, volume)
            }
            sequence.append(bar)
            
        return sequence

    def test_hf_momentum_sequence_sensitivity(self):
        """Test HF momentum feature responds to different sequence patterns."""
        config = FeatureConfig(name="hf_momentum_summary")
        feature = HFMomentumSummaryFeature(config)
        
        # Test with uptrend sequence
        up_sequence = self.create_hf_sequence(20, 'up')
        up_data = {'hf_data_window': up_sequence}
        up_momentum = feature.calculate_raw(up_data)
        
        # Test with downtrend sequence  
        down_sequence = self.create_hf_sequence(20, 'down')
        down_data = {'hf_data_window': down_sequence}
        down_momentum = feature.calculate_raw(down_data)
        
        # Test with flat sequence
        flat_sequence = self.create_hf_sequence(20, 'flat')
        flat_data = {'hf_data_window': flat_sequence}
        flat_momentum = feature.calculate_raw(flat_data)
        
        # Validate sequence sensitivity
        assert up_momentum > flat_momentum, "Uptrend should have higher momentum than flat"
        assert down_momentum < flat_momentum, "Downtrend should have lower momentum than flat"
        assert up_momentum > down_momentum, "Uptrend should have higher momentum than downtrend"
        
        # Test sequence length dependency
        short_up = self.create_hf_sequence(5, 'up')
        short_data = {'hf_data_window': short_up}
        short_momentum = feature.calculate_raw(short_data)
        
        # Longer sequences should provide different signals (proving they use more data)
        assert up_momentum != short_momentum, "Different sequence lengths should yield different results"

    def test_hf_volume_dynamics_temporal_analysis(self):
        """Test volume dynamics analyzes entire sequence."""
        config = FeatureConfig(name="hf_volume_dynamics")
        feature = HFVolumeDynamicsFeature(config)
        
        # Increasing volume sequence
        increasing_seq = []
        for i in range(15):
            volume = 1000 + (i * 100)  # Steadily increasing
            entry = {
                '1s_bar': {
                    'close': 100.0,
                    'volume': volume
                }
            }
            increasing_seq.append(entry)
        
        # Decreasing volume sequence
        decreasing_seq = []
        for i in range(15):
            volume = 2500 - (i * 100)  # Steadily decreasing
            entry = {
                '1s_bar': {
                    'close': 100.0,
                    'volume': volume
                }
            }
            decreasing_seq.append(entry)
        
        # Flat volume sequence
        flat_seq = self.create_hf_sequence(15, 'flat')
        
        inc_dynamics = feature.calculate_raw({'hf_data_window': increasing_seq})
        dec_dynamics = feature.calculate_raw({'hf_data_window': decreasing_seq})
        flat_dynamics = feature.calculate_raw({'hf_data_window': flat_seq})
        
        # Increasing volume should score higher than decreasing
        assert inc_dynamics > dec_dynamics, "Increasing volume should score higher than decreasing"
        
        # Should handle edge cases
        empty_data = feature.calculate_raw({'hf_data_window': []})
        assert empty_data == 0.0, "Empty sequence should return 0.0"

    def test_microstructure_quality_sequence_analysis(self):
        """Test microstructure quality uses full sequence."""
        config = FeatureConfig(name="hf_microstructure_quality")
        feature = HFMicrostructureQualityFeature(config)
        
        # High quality sequence (tight spreads, many quotes)
        high_quality_seq = []
        for i in range(10):
            entry = {
                'quotes': [
                    {'bid_px': 99.99, 'ask_px': 100.01},  # Tight spread
                    {'bid_px': 99.995, 'ask_px': 100.005},  # Multiple quotes
                    {'bid_px': 99.98, 'ask_px': 100.02}
                ]
            }
            high_quality_seq.append(entry)
        
        # Low quality sequence (wide spreads, few quotes)
        low_quality_seq = []
        for i in range(10):
            entry = {
                'quotes': [
                    {'bid_px': 99.90, 'ask_px': 100.10}  # Wide spread, single quote
                ]
            }
            low_quality_seq.append(entry)
        
        high_quality = feature.calculate_raw({'hf_data_window': high_quality_seq})
        low_quality = feature.calculate_raw({'hf_data_window': low_quality_seq})
        
        # High quality market should score better
        assert high_quality > low_quality, "High quality market should score better than low quality"
        
        # Verify bounds
        assert -1.0 <= high_quality <= 1.0, "Quality score should be in bounds"
        assert -1.0 <= low_quality <= 1.0, "Quality score should be in bounds"

    def test_mf_trend_consistency_sequence_behavior(self):
        """Test MF trend consistency analyzes entire window."""
        config = FeatureConfig(name="mf_trend_consistency")
        feature = MFTrendConsistencyFeature(config)
        
        # Consistent uptrend
        consistent_seq = self.create_mf_sequence(10, 'up')
        
        # Choppy/inconsistent sequence
        choppy_seq = self.create_mf_sequence(10, 'choppy')
        
        # Flat sequence
        flat_seq = self.create_mf_sequence(10, 'flat')
        
        consistent_score = feature.calculate_raw({'1m_bars_window': consistent_seq})
        choppy_score = feature.calculate_raw({'1m_bars_window': choppy_seq})
        flat_score = feature.calculate_raw({'1m_bars_window': flat_seq})
        
        # Consistent trend should score higher than choppy
        assert consistent_score > choppy_score, "Consistent trend should score higher than choppy"
        
        # Test with different sequence lengths
        short_seq = self.create_mf_sequence(3, 'up')
        short_score = feature.calculate_raw({'1m_bars_window': short_seq})
        assert short_score == 0.0, "Short sequences should return 0.0"

    def test_momentum_persistence_temporal_tracking(self):
        """Test momentum persistence tracks changes over time."""
        config = FeatureConfig(name="mf_momentum_persistence")
        feature = MFMomentumPersistenceFeature(config)
        
        # Persistent momentum (same direction)
        persistent_seq = []
        for i in range(12):
            price = 100.0 + (i * 0.3)  # Steady upward momentum
            bar = {
                'open': price - 0.1,
                'high': price + 0.1,
                'low': price - 0.1,
                'close': price,
                'volume': 1000
            }
            persistent_seq.append(bar)
        
        # Reversing momentum (changes direction)
        reversing_seq = []
        for i in range(12):
            # Alternate up/down every 3 bars
            direction = 1 if (i // 3) % 2 == 0 else -1
            price = 100.0 + (direction * (i % 3) * 0.3)
            bar = {
                'open': price - 0.1,
                'high': price + 0.1,
                'low': price - 0.1, 
                'close': price,
                'volume': 1000
            }
            reversing_seq.append(bar)
        
        persistent_score = feature.calculate_raw({'1m_bars_window': persistent_seq})
        reversing_score = feature.calculate_raw({'1m_bars_window': reversing_seq})
        
        # Persistent momentum should score higher
        assert persistent_score > reversing_score, "Persistent momentum should score higher than reversing"
        
        # Verify bounds
        assert -1.0 <= persistent_score <= 1.0, "Persistence score should be in bounds"
        assert -1.0 <= reversing_score <= 1.0, "Persistence score should be in bounds"

    def test_professional_ema_sequence_utilization(self):
        """Test professional EMA feature uses sequence data properly."""
        config = FeatureConfig(name="professional_ema_system")
        feature = ProfessionalEMASystemFeature(config)
        
        # Create trending price sequence
        trending_bars = []
        for i in range(30):
            price = 100.0 + (i * 0.2)  # Steady uptrend
            bar = {
                'open': price - 0.05,
                'high': price + 0.05,
                'low': price - 0.05,
                'close': price,
                'volume': 1000
            }
            trending_bars.append(bar)
        
        # Create sideways price sequence
        sideways_bars = []
        for i in range(30):
            price = 100.0 + np.random.normal(0, 0.05)  # Random walk around 100
            bar = {
                'open': price - 0.02,
                'high': price + 0.02,
                'low': price - 0.02,
                'close': price,
                'volume': 1000
            }
            sideways_bars.append(bar)
        
        trending_signal = feature.calculate_raw({'1m_bars_window': trending_bars})
        sideways_signal = feature.calculate_raw({'1m_bars_window': sideways_bars})
        
        # Trending market should have stronger EMA signal
        assert abs(trending_signal) > abs(sideways_signal), "Trending market should have stronger EMA signal"
        
        # Test insufficient data
        short_data = feature.calculate_raw({'1m_bars_window': trending_bars[:5]})
        assert short_data == 0.0, "Insufficient data should return 0.0"

    def test_sequence_length_requirements(self):
        """Test features handle various sequence lengths correctly."""
        
        # Test HF features with different lengths
        hf_config = FeatureConfig(name="hf_momentum_summary")
        hf_feature = HFMomentumSummaryFeature(hf_config)
        
        # Too short - should return default
        short_hf = self.create_hf_sequence(2, 'up')
        result = hf_feature.calculate_raw({'hf_data_window': short_hf})
        assert result == 0.0, "Too short HF sequence should return 0.0"
        
        # Adequate length - should return valid result
        adequate_hf = self.create_hf_sequence(10, 'up')
        result = hf_feature.calculate_raw({'hf_data_window': adequate_hf})
        assert isinstance(result, float), "Adequate HF sequence should return float"
        assert -1.0 <= result <= 1.0, "Result should be in bounds"
        
        # Test MF features with different lengths
        mf_config = FeatureConfig(name="mf_trend_consistency")
        feature = MFTrendConsistencyFeature(config)
        
        # Too short - should return default
        short_mf = self.create_mf_sequence(3, 'up')
        result = mf_feature.calculate_raw({'1m_bars_window': short_mf})
        assert result == 0.0, "Too short MF sequence should return 0.0"
        
        # Adequate length - should return valid result
        adequate_mf = self.create_mf_sequence(10, 'up')
        result = mf_feature.calculate_raw({'1m_bars_window': adequate_mf})
        assert isinstance(result, float), "Adequate MF sequence should return float"

    def test_temporal_weighting_in_aggregated_features(self):
        """Test that aggregated features properly weight recent vs older data."""
        config = FeatureConfig(name="hf_momentum_summary")
        feature = HFMomentumSummaryFeature(config)
        
        # Sequence with recent strong momentum
        recent_momentum_seq = []
        for i in range(20):
            if i < 15:
                price = 100.0  # Flat early
            else:
                price = 100.0 + ((i - 15) * 0.5)  # Strong momentum recently
            
            entry = {
                '1s_bar': {
                    'close': price,
                    'volume': 1000
                }
            }
            recent_momentum_seq.append(entry)
        
        # Sequence with early strong momentum that faded
        early_momentum_seq = []
        for i in range(20):
            if i < 5:
                price = 100.0 + (i * 0.5)  # Early momentum
            else:
                price = 102.0  # Then flat
            
            entry = {
                '1s_bar': {
                    'close': price,
                    'volume': 1000
                }
            }
            early_momentum_seq.append(entry)
        
        recent_score = feature.calculate_raw({'hf_data_window': recent_momentum_seq})
        early_score = feature.calculate_raw({'hf_data_window': early_momentum_seq})
        
        # Recent momentum should be weighted higher due to exponential weighting
        assert recent_score > early_score, "Recent momentum should be weighted higher than early momentum"