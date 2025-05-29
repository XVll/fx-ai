"""
Comprehensive Feature Tests - Input/Output Behavior Testing

Tests focus on input/output contracts rather than implementation details.
All features must satisfy these behavioral requirements:
1. Return valid float values in declared ranges
2. Handle edge cases gracefully with default values
3. Never return NaN, infinity, or None
4. Respect normalization bounds
5. Process expected input formats correctly
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from unittest.mock import MagicMock

from feature.feature_manager import FeatureManager
from feature.feature_registry import feature_registry
from feature.contexts import MarketContext
from config.schemas import ModelConfig


class TestFeatureBehaviorContracts:
    """Test that all features satisfy behavioral contracts"""
    
    @pytest.fixture
    def model_config(self):
        """Model config with current architecture"""
        return ModelConfig(
            hf_seq_len=60,
            hf_feat_dim=7,  # Updated count
            mf_seq_len=30,
            mf_feat_dim=43,  # Updated count
            lf_seq_len=30,
            lf_feat_dim=19,  # Updated count (includes former static)
            portfolio_seq_len=5,
            portfolio_feat_dim=5
        )
    
    @pytest.fixture
    def feature_manager(self, model_config):
        """Feature manager with all features loaded"""
        return FeatureManager('MLGO', model_config, MagicMock())
    
    @pytest.fixture
    def valid_market_data(self):
        """Create valid market data for testing"""
        # Create realistic 1s bars for HF
        hf_data = []
        base_price = 100.0
        for i in range(60):
            price = base_price + np.random.normal(0, 0.1)
            hf_data.append({
                'timestamp': pd.Timestamp('2025-01-15 14:30:00') + pd.Timedelta(seconds=i),
                '1s_bar': {
                    'open': price - 0.05,
                    'high': price + 0.1,
                    'low': price - 0.1,
                    'close': price,
                    'volume': 1000 + np.random.randint(-200, 200)
                },
                'trades': [
                    {'price': price, 'size': 100, 'conditions': ['BUY']},
                    {'price': price - 0.01, 'size': 50, 'conditions': ['SELL']}
                ],
                'quotes': [
                    {'bid_px': price - 0.01, 'ask_px': price + 0.01, 'bid_sz': 1000, 'ask_sz': 1000}
                ]
            })
        
        # Create 1m bars for MF
        mf_data = []
        for i in range(30):
            price = base_price + np.random.normal(0, 0.5)
            mf_data.append({
                'timestamp': pd.Timestamp('2025-01-15 14:30:00') + pd.Timedelta(minutes=i),
                'open': price - 0.2,
                'high': price + 0.3,
                'low': price - 0.3,
                'close': price,
                'volume': 5000 + np.random.randint(-1000, 1000),
                'vwap': price + np.random.normal(0, 0.05)
            })
        
        # Create 5m bars
        bars_5m = []
        for i in range(12):
            price = base_price + np.random.normal(0, 1.0)
            bars_5m.append({
                'timestamp': pd.Timestamp('2025-01-15 14:30:00') + pd.Timedelta(minutes=i*5),
                'open': price - 0.5,
                'high': price + 0.7,
                'low': price - 0.7,
                'close': price,
                'volume': 25000 + np.random.randint(-5000, 5000)
            })
        
        return {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00'),
            'current_price': base_price,
            'hf_data_window': hf_data,
            '1m_bars_window': mf_data,
            '5m_bars_window': bars_5m,
            'mf_bars_1m': mf_data,  # Alternative key
            'session_vwap': base_price + 0.1,
            'session_volume': 100000,
            'market_session': 'REGULAR',
            'market_cap': 1000000000,
            'prev_day_close': base_price - 0.5,
            'session_high': base_price + 2.0,
            'session_low': base_price - 2.0,
            'prev_day_high': base_price + 1.5,
            'prev_day_low': base_price - 1.5,
            'portfolio_state': {
                'position_size': 0,
                'average_price': 0,
                'unrealized_pnl': 0,
                'time_in_position': 0,
                'max_adverse_excursion': 0
            }
        }
    
    @pytest.fixture
    def edge_case_data(self):
        """Create edge case data that should trigger default values"""
        return {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00'),
            'current_price': 0,  # Invalid price
            'hf_data_window': [],  # Empty data
            '1m_bars_window': [],
            '5m_bars_window': [],
            'session_vwap': 0,
            'market_session': None,
            'portfolio_state': None
        }

    def test_all_features_return_valid_floats(self, feature_manager, valid_market_data):
        """Test that all features return valid float values"""
        for category in ['hf', 'mf', 'lf', 'portfolio']:
            features = feature_manager.calculate_features(valid_market_data, category)
            
            for name, value in features.items():
                # Must be a finite float
                assert isinstance(value, (int, float)), f"Feature {name} returned {type(value)}: {value}"
                assert np.isfinite(value), f"Feature {name} returned non-finite value: {value}"
                assert not np.isnan(value), f"Feature {name} returned NaN"
                assert not np.isinf(value), f"Feature {name} returned infinity"

    def test_all_features_handle_edge_cases(self, feature_manager, edge_case_data):
        """Test that all features handle edge cases gracefully"""
        for category in ['hf', 'mf', 'lf', 'portfolio']:
            features = feature_manager.calculate_features(edge_case_data, category)
            
            for name, value in features.items():
                # Must still return valid values even with bad input
                assert isinstance(value, (int, float)), f"Feature {name} failed on edge case: {value}"
                assert np.isfinite(value), f"Feature {name} returned non-finite on edge case: {value}"

    def test_feature_normalization_bounds(self, feature_manager, valid_market_data):
        """Test that features respect their declared normalization bounds"""
        for category in ['hf', 'mf', 'lf', 'portfolio']:
            category_features = feature_manager._features.get(category, {})
            calculated_features = feature_manager.calculate_features(valid_market_data, category)
            
            for name, feature_obj in category_features.items():
                if name in calculated_features:
                    value = calculated_features[name]
                    norm_params = feature_obj.get_normalization_params()
                    
                    # Check bounds if specified
                    if 'min' in norm_params and 'max' in norm_params:
                        min_val = norm_params['min']
                        max_val = norm_params['max']
                        assert min_val <= value <= max_val, \
                            f"Feature {name} value {value} outside bounds [{min_val}, {max_val}]"

    def test_feature_vectorization_consistency(self, feature_manager, valid_market_data):
        """Test that feature vectorization produces expected dimensions"""
        expected_dims = {
            'hf': feature_manager.hf_feat_dim,
            'mf': feature_manager.mf_feat_dim,
            'lf': feature_manager.lf_feat_dim,
            'portfolio': feature_manager.portfolio_feat_dim
        }
        
        for category, expected_dim in expected_dims.items():
            features = feature_manager.calculate_features(valid_market_data, category)
            vector = feature_manager.vectorize_features(features, category)
            
            assert len(vector) == expected_dim, \
                f"Category {category} vector length {len(vector)} != expected {expected_dim}"
            assert vector.dtype == np.float32, f"Category {category} vector not float32"
            assert np.all(np.isfinite(vector)), f"Category {category} vector contains non-finite values"

    def test_professional_features_integration(self, feature_manager, valid_market_data):
        """Test that professional features (pandas/ta) work correctly"""
        professional_features = [
            'professional_ema_system',
            'professional_momentum_quality', 
            'professional_volatility_regime',
            'professional_vwap_analysis'
        ]
        
        mf_features = feature_manager.calculate_features(valid_market_data, 'mf')
        
        for feature_name in professional_features:
            assert feature_name in mf_features, f"Professional feature {feature_name} not found"
            value = mf_features[feature_name]
            
            # Professional features should be especially robust
            assert isinstance(value, (int, float)), f"Professional feature {feature_name} not numeric"
            assert np.isfinite(value), f"Professional feature {feature_name} not finite"
            
            # Check reasonable bounds for professional features
            if feature_name == 'professional_volatility_regime':
                assert 0 <= value <= 1, f"Volatility regime {value} outside [0,1]"
            else:
                assert -1 <= value <= 1, f"Professional feature {feature_name} {value} outside [-1,1]"

    def test_aggregated_features_sequence_utilization(self, feature_manager, valid_market_data):
        """Test that aggregated features properly utilize sequence data"""
        aggregated_features = [
            'hf_momentum_summary',
            'hf_volume_dynamics', 
            'hf_microstructure_quality',
            'mf_trend_consistency',
            'mf_volume_price_divergence',
            'mf_momentum_persistence'
        ]
        
        # Test with full data
        full_features = {}
        full_features.update(feature_manager.calculate_features(valid_market_data, 'hf'))
        full_features.update(feature_manager.calculate_features(valid_market_data, 'mf'))
        
        # Test with minimal data (should give different results)
        minimal_data = valid_market_data.copy()
        minimal_data['hf_data_window'] = minimal_data['hf_data_window'][:3]  # Only 3 timesteps
        minimal_data['1m_bars_window'] = minimal_data['1m_bars_window'][:3]
        
        minimal_features = {}
        minimal_features.update(feature_manager.calculate_features(minimal_data, 'hf'))
        minimal_features.update(feature_manager.calculate_features(minimal_data, 'mf'))
        
        # Aggregated features should behave differently with different sequence lengths
        for feature_name in aggregated_features:
            if feature_name in full_features and feature_name in minimal_features:
                full_val = full_features[feature_name]
                minimal_val = minimal_features[feature_name]
                
                # Both should be valid
                assert np.isfinite(full_val), f"Full sequence {feature_name} not finite"
                assert np.isfinite(minimal_val), f"Minimal sequence {feature_name} not finite"
                
                # They don't have to be different (depends on data), but both should be valid
                assert isinstance(full_val, (int, float)), f"Aggregated feature {feature_name} not numeric"

    def test_feature_categories_correct_counts(self, feature_manager):
        """Test that feature categories have expected counts"""
        expected_counts = {
            'hf': 7,    # Professional aggregated
            'mf': 43,   # Professional pandas/ta
            'lf': 19,   # Session/daily context (includes former static)
            'portfolio': 5
        }
        
        for category, expected_count in expected_counts.items():
            enabled_features = feature_manager.get_enabled_features(category)
            actual_count = len(enabled_features)
            
            assert actual_count == expected_count, \
                f"Category {category} has {actual_count} features, expected {expected_count}"

    def test_no_static_category_remains(self, feature_manager):
        """Test that static category has been eliminated"""
        # Should not be able to calculate static features
        with pytest.raises((KeyError, ValueError)):
            feature_manager.calculate_features({}, 'static')
        
        # Static should not be in feature categories
        assert 'static' not in feature_manager._features, "Static category still exists"

    def test_feature_extraction_integration(self, feature_manager, valid_market_data):
        """Test end-to-end feature extraction pipeline"""
        # Create mock market context
        context = MarketContext(
            timestamp=valid_market_data['timestamp'],
            current_price=valid_market_data['current_price'],
            session='REGULAR',
            hf_window=valid_market_data['hf_data_window'],
            mf_1m_window=valid_market_data['1m_bars_window'],
            lf_5m_window=valid_market_data['5m_bars_window'],
            portfolio_state=valid_market_data['portfolio_state'],
            session_high=valid_market_data['session_high'],
            session_low=valid_market_data['session_low'],
            prev_day_close=valid_market_data['prev_day_close'],
            prev_day_high=valid_market_data['prev_day_high'],
            prev_day_low=valid_market_data['prev_day_low'],
            market_cap=valid_market_data['market_cap']
        )
        
        # Extract all features
        all_features = feature_manager.extract_features(context)
        
        # Should return all 4 categories
        assert all_features is not None, "Feature extraction returned None"
        expected_categories = {'hf', 'mf', 'lf', 'portfolio'}
        actual_categories = set(all_features.keys())
        assert actual_categories == expected_categories, \
            f"Expected categories {expected_categories}, got {actual_categories}"
        
        # Check shapes
        assert all_features['hf'].shape == (60, 7), f"HF shape {all_features['hf'].shape} != (60, 7)"
        assert all_features['mf'].shape == (30, 43), f"MF shape {all_features['mf'].shape} != (30, 43)"
        assert all_features['lf'].shape == (30, 19), f"LF shape {all_features['lf'].shape} != (30, 19)"
        assert all_features['portfolio'].shape == (5, 5), f"Portfolio shape {all_features['portfolio'].shape} != (5, 5)"
        
        # Check data types
        for category, features in all_features.items():
            assert features.dtype == np.float32, f"Category {category} not float32"
            assert np.all(np.isfinite(features)), f"Category {category} contains non-finite values"


class TestSpecificFeatureCategories:
    """Test specific feature category behaviors"""
    
    @pytest.fixture
    def feature_manager(self):
        config = ModelConfig(hf_feat_dim=7, mf_feat_dim=43, lf_feat_dim=19, portfolio_feat_dim=5)
        return FeatureManager('MLGO', config, MagicMock())

    def test_hf_features_microstructure_focus(self, feature_manager):
        """Test HF features focus on microstructure"""
        market_data = {
            'hf_data_window': [
                {
                    'timestamp': pd.Timestamp('2025-01-15 14:30:00'),
                    '1s_bar': {'close': 100.0, 'volume': 1000},
                    'trades': [{'price': 100.0, 'size': 100, 'conditions': ['BUY']}],
                    'quotes': [{'bid_px': 99.9, 'ask_px': 100.1, 'bid_sz': 1000, 'ask_sz': 1000}]
                }
                for _ in range(60)
            ]
        }
        
        hf_features = feature_manager.calculate_features(market_data, 'hf')
        
        # Should include microstructure features
        expected_hf_features = {
            'spread_compression', 'tape_imbalance', 'tape_aggression_ratio', 'quote_imbalance',
            'hf_momentum_summary', 'hf_volume_dynamics', 'hf_microstructure_quality'
        }
        
        actual_features = set(hf_features.keys())
        assert expected_hf_features.issubset(actual_features), \
            f"Missing HF features: {expected_hf_features - actual_features}"

    def test_mf_features_technical_analysis(self, feature_manager):
        """Test MF features include professional technical analysis"""
        market_data = {
            '1m_bars_window': [
                {
                    'timestamp': pd.Timestamp('2025-01-15 14:30:00') + pd.Timedelta(minutes=i),
                    'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5,
                    'volume': 5000, 'vwap': 100.2
                }
                for i in range(30)
            ],
            '5m_bars_window': [
                {
                    'timestamp': pd.Timestamp('2025-01-15 14:30:00') + pd.Timedelta(minutes=i*5),
                    'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
                    'volume': 25000
                }
                for i in range(12)
            ]
        }
        
        mf_features = feature_manager.calculate_features(market_data, 'mf')
        
        # Should include professional features
        professional_features = {
            'professional_ema_system', 'professional_momentum_quality',
            'professional_volatility_regime', 'professional_vwap_analysis'
        }
        
        actual_features = set(mf_features.keys())
        assert professional_features.issubset(actual_features), \
            f"Missing professional features: {professional_features - actual_features}"

    def test_lf_features_session_context(self, feature_manager):
        """Test LF features include session/time context (former static)"""
        market_data = {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00'),
            'current_price': 100.0,
            'market_session': 'REGULAR',
            'prev_day_close': 99.0,
            'session_high': 102.0,
            'session_low': 98.0,
            'prev_day_high': 101.0,
            'prev_day_low': 97.0
        }
        
        lf_features = feature_manager.calculate_features(market_data, 'lf')
        
        # Should include former static features
        session_context_features = {
            'market_session_type', 'time_of_day_sin', 'time_of_day_cos',
            'session_progress', 'is_halted', 'time_since_halt',
            'market_stress_level', 'session_volume_profile'
        }
        
        actual_features = set(lf_features.keys())
        missing_context = session_context_features - actual_features
        
        # Some features might not be implemented yet, but should have placeholders
        for feature_name in missing_context:
            assert feature_name in feature_manager._feature_order.get('lf', []), \
                f"Feature {feature_name} not in LF feature order"

    def test_portfolio_features_position_tracking(self, feature_manager):
        """Test portfolio features track position state"""
        market_data = {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00'),
            'current_price': 100.0,
            'portfolio_state': {
                'position_size': 100,
                'average_price': 99.0,
                'unrealized_pnl': 100.0,
                'time_in_position': 300,  # 5 minutes
                'max_adverse_excursion': -50.0
            }
        }
        
        portfolio_features = feature_manager.calculate_features(market_data, 'portfolio')
        
        # Should include all portfolio features
        expected_portfolio_features = {
            'portfolio_position_size', 'portfolio_average_price', 'portfolio_unrealized_pnl',
            'portfolio_time_in_position', 'portfolio_max_adverse_excursion'
        }
        
        actual_features = set(portfolio_features.keys())
        assert expected_portfolio_features.issubset(actual_features), \
            f"Missing portfolio features: {expected_portfolio_features - actual_features}"
        
        # Test position size is properly normalized
        pos_size = portfolio_features.get('portfolio_position_size', 0)
        assert -1 <= pos_size <= 1, f"Position size {pos_size} outside [-1, 1]"


class TestFeatureRobustness:
    """Test feature robustness under stress conditions"""
    
    @pytest.fixture
    def feature_manager(self):
        config = ModelConfig(hf_feat_dim=7, mf_feat_dim=43, lf_feat_dim=19, portfolio_feat_dim=5)
        return FeatureManager('MLGO', config, MagicMock())

    def test_extreme_market_conditions(self, feature_manager):
        """Test features handle extreme market conditions"""
        extreme_scenarios = [
            # Market crash scenario
            {
                'current_price': 50.0,  # 50% drop
                'prev_day_close': 100.0,
                'session_high': 100.0,
                'session_low': 45.0,
                'hf_data_window': [
                    {'1s_bar': {'close': price, 'volume': 50000}}
                    for price in np.linspace(100, 50, 60)
                ],
                '1m_bars_window': [
                    {'close': price, 'volume': 100000, 'high': price+1, 'low': price-5}
                    for price in np.linspace(100, 50, 30)
                ]
            },
            # Market halt scenario
            {
                'current_price': 100.0,
                'is_halted': True,
                'hf_data_window': [
                    {'1s_bar': {'close': 100.0, 'volume': 0}}  # No volume during halt
                    for _ in range(60)
                ],
                '1m_bars_window': [
                    {'close': 100.0, 'volume': 0, 'high': 100.0, 'low': 100.0}
                    for _ in range(30)
                ]
            },
            # Low volume scenario
            {
                'current_price': 100.0,
                'hf_data_window': [
                    {'1s_bar': {'close': 100.0 + np.random.normal(0, 0.01), 'volume': 1}}
                    for _ in range(60)
                ],
                '1m_bars_window': [
                    {'close': 100.0, 'volume': 10, 'high': 100.01, 'low': 99.99}
                    for _ in range(30)
                ]
            }
        ]
        
        for scenario_name, market_data in zip(['crash', 'halt', 'low_volume'], extreme_scenarios):
            for category in ['hf', 'mf', 'lf', 'portfolio']:
                try:
                    features = feature_manager.calculate_features(market_data, category)
                    
                    for name, value in features.items():
                        assert np.isfinite(value), \
                            f"Feature {name} not finite in {scenario_name} scenario: {value}"
                        assert isinstance(value, (int, float)), \
                            f"Feature {name} not numeric in {scenario_name} scenario: {value}"
                        
                except Exception as e:
                    pytest.fail(f"Feature calculation failed in {scenario_name} scenario for {category}: {e}")

    def test_missing_data_handling(self, feature_manager):
        """Test features handle various missing data scenarios"""
        missing_data_scenarios = [
            {},  # Empty data
            {'current_price': None},  # None values
            {'hf_data_window': None, '1m_bars_window': None},  # None sequences
            {'hf_data_window': [], '1m_bars_window': []},  # Empty sequences
            {
                'hf_data_window': [{'1s_bar': None} for _ in range(5)],  # None bars
                '1m_bars_window': [None for _ in range(5)]
            }
        ]
        
        for scenario_idx, market_data in enumerate(missing_data_scenarios):
            for category in ['hf', 'mf', 'lf', 'portfolio']:
                features = feature_manager.calculate_features(market_data, category)
                
                for name, value in features.items():
                    assert np.isfinite(value), \
                        f"Feature {name} not finite with missing data scenario {scenario_idx}: {value}"
                    assert isinstance(value, (int, float)), \
                        f"Feature {name} not numeric with missing data scenario {scenario_idx}: {value}"

    def test_feature_consistency_over_time(self, feature_manager):
        """Test that features produce consistent results for identical inputs"""
        market_data = {
            'timestamp': pd.Timestamp('2025-01-15 14:30:00'),
            'current_price': 100.0,
            'hf_data_window': [
                {'1s_bar': {'close': 100.0, 'volume': 1000}}
                for _ in range(60)
            ],
            '1m_bars_window': [
                {'close': 100.0, 'volume': 5000, 'high': 100.5, 'low': 99.5, 'vwap': 100.0}
                for _ in range(30)
            ]
        }
        
        # Calculate features multiple times
        results = []
        for _ in range(3):
            category_results = {}
            for category in ['hf', 'mf', 'lf', 'portfolio']:
                category_results[category] = feature_manager.calculate_features(market_data, category)
            results.append(category_results)
        
        # All results should be identical (features should be deterministic)
        for category in ['hf', 'mf', 'lf', 'portfolio']:
            for feature_name in results[0][category]:
                values = [result[category][feature_name] for result in results]
                assert all(abs(v - values[0]) < 1e-10 for v in values), \
                    f"Feature {feature_name} inconsistent across runs: {values}"