import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

from envs.momentum_episode_scanner import (
    MomentumEpisodeScanner,
    MomentumPhase,
    CategorizedResetPoint,
    PatternType,
    TimeOfDayQuality
)


class TestMomentumEpisodeScanner:
    """Test suite for MomentumEpisodeScanner component."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager with test data."""
        data_manager = Mock()
        return data_manager
    
    @pytest.fixture
    def scanner_config(self):
        """Test configuration for scanner."""
        return {
            'momentum_detection': {
                'volume_surge_threshold': 2.0,
                'price_velocity_threshold': 0.002,
                'atr_expansion_threshold': 1.5,
                'consolidation_minutes': 5
            },
            'quality_weights': {
                'phase_weight': 0.4,
                'volume_weight': 0.3,
                'setup_weight': 0.2,
                'spread_weight': 0.1
            },
            'time_multipliers': {
                'early_premarket': 0.3,    # 4-7 AM
                'active_premarket': 1.0,   # 7-9:30 AM
                'market_open': 1.0,        # 9:30-10:30 AM
                'midday': 0.4,             # 10:30 AM-2 PM
                'power_hour': 0.8,         # 2-4 PM
                'after_hours': 0.5         # 4-8 PM
            }
        }
    
    @pytest.fixture
    def scanner(self, mock_data_manager, scanner_config):
        """Create scanner instance."""
        return MomentumEpisodeScanner(mock_data_manager, scanner_config)
    
    @pytest.fixture
    def sample_day_data(self):
        """Generate sample market data for a full trading day."""
        # Create 1-second data from 4 AM to 8 PM
        date = datetime(2025, 1, 15)
        start = date.replace(hour=4, minute=0, second=0)
        end = date.replace(hour=20, minute=0, second=0)
        
        timestamps = pd.date_range(start=start, end=end, freq='1s')
        n_points = len(timestamps)
        
        # Generate realistic price movement with momentum events
        base_price = 10.0
        prices = np.zeros(n_points)
        volumes = np.zeros(n_points)
        
        # Normal market hours pattern
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Base volume pattern
            if 9 <= hour < 10:  # Market open
                base_volume = 50000
            elif 10 <= hour < 14:  # Midday
                base_volume = 10000
            elif 14 <= hour < 16:  # Power hour
                base_volume = 30000
            else:  # Pre/post market
                base_volume = 5000
                
            # Add momentum events
            if hour == 9 and 30 <= minute <= 35:  # Opening squeeze
                prices[i] = base_price * (1 + 0.05 * (minute - 30) / 5)
                volumes[i] = base_volume * 5
            elif hour == 10 and 0 <= minute <= 5:  # Back side flush
                prices[i] = base_price * 1.05 * (1 - 0.03 * minute / 5)
            elif hour == 14 and 30 <= minute <= 40:  # Power hour breakout
                prices[i] = base_price * (1 + 0.03 * (minute - 30) / 10)
                volumes[i] = base_volume * 3
            else:
                # Normal price movement
                prices[i] = base_price * (1 + 0.001 * np.sin(i / 1000))
                volumes[i] = base_volume * (1 + 0.5 * np.random.random())
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': volumes,
            'bid': prices * 0.999,
            'ask': prices * 1.001,
            'bid_size': volumes * 0.4,
            'ask_size': volumes * 0.6
        }, index=timestamps)
        
        return df
    
    def test_scanner_initialization(self, scanner, scanner_config):
        """Test scanner initialization with config."""
        assert scanner.volume_surge_threshold == 2.0
        assert scanner.price_velocity_threshold == 0.002
        assert scanner.atr_expansion_threshold == 1.5
        assert scanner.quality_weights['phase_weight'] == 0.4
    
    def test_scan_single_day_categorization(self, scanner, mock_data_manager, sample_day_data):
        """Test scanning a full day and categorizing reset points."""
        # Setup mock
        mock_data_manager.get_day_data.return_value = sample_day_data
        
        # Scan the day
        result = scanner.scan_single_day('MLGO', datetime(2025, 1, 15))
        
        # Verify structure
        assert isinstance(result, dict)
        assert 'prime_momentum' in result
        assert 'secondary_momentum' in result
        assert 'risk_scenarios' in result
        assert 'dead_zones' in result
        assert 'time_based' in result
        
        # Should find reset points in each category
        assert len(result['prime_momentum']) > 0
        assert len(result['time_based']) >= 4  # Fixed time points
        
        # Verify reset point structure
        prime_point = result['prime_momentum'][0]
        assert isinstance(prime_point, CategorizedResetPoint)
        assert prime_point.quality_score > 0.8
        assert prime_point.momentum_phase in MomentumPhase
    
    def test_momentum_phase_detection(self, scanner):
        """Test detection of different momentum phases."""
        # Create test data for each phase
        test_cases = [
            # (price_change_pct, volume_ratio, atr_ratio, expected_phase)
            (0.05, 5.0, 2.0, MomentumPhase.FRONT_SIDE_BREAKOUT),
            (0.08, 8.0, 3.0, MomentumPhase.PARABOLIC),
            (-0.04, 4.0, 2.5, MomentumPhase.BACK_SIDE_FLUSH),
            (0.01, 3.0, 1.5, MomentumPhase.BOUNCE),
            (0.001, 0.5, 0.8, MomentumPhase.CONSOLIDATION),
            (0.0, 0.2, 0.5, MomentumPhase.DEAD)
        ]
        
        for price_change, vol_ratio, atr_ratio, expected in test_cases:
            # Create minimal test data
            timestamps = pd.date_range('2025-01-15 09:30:00', periods=300, freq='1s')
            base_price = 10.0
            
            df = pd.DataFrame({
                'close': base_price * (1 + price_change * np.linspace(0, 1, 300)),
                'volume': 10000 * vol_ratio,
                'high': base_price * 1.01,
                'low': base_price * 0.99
            }, index=timestamps)
            
            phase = scanner._detect_momentum_phase(df, timestamps[150])
            assert phase == expected
    
    def test_quality_score_calculation(self, scanner):
        """Test quality score calculation with various inputs."""
        # Test time-based multipliers
        test_times = [
            (datetime(2025, 1, 15, 5, 0), 0.3),   # Early premarket
            (datetime(2025, 1, 15, 8, 0), 1.0),   # Active premarket
            (datetime(2025, 1, 15, 9, 45), 1.0),  # Market open
            (datetime(2025, 1, 15, 12, 0), 0.4),  # Midday
            (datetime(2025, 1, 15, 15, 0), 0.8),  # Power hour
            (datetime(2025, 1, 15, 17, 0), 0.5)   # After hours
        ]
        
        for test_time, expected_multiplier in test_times:
            metrics = scanner._calculate_quality_metrics(
                Mock(),  # data not used in this test
                test_time,
                phase_score=1.0,
                volume_score=1.0,
                setup_quality=1.0,
                spread_tightness=1.0
            )
            
            # Quality should be affected by time multiplier
            expected_quality = expected_multiplier
            assert abs(metrics['quality'] - expected_quality) < 0.01
    
    def test_pattern_detection(self, scanner):
        """Test detection of various trading patterns."""
        # Test breakout pattern
        breakout_data = self._create_breakout_pattern()
        pattern = scanner._detect_pattern_type(breakout_data, breakout_data.index[100])
        assert pattern == PatternType.BREAKOUT
        
        # Test flush pattern
        flush_data = self._create_flush_pattern()
        pattern = scanner._detect_pattern_type(flush_data, flush_data.index[100])
        assert pattern == PatternType.FLUSH
        
        # Test bounce pattern
        bounce_data = self._create_bounce_pattern()
        pattern = scanner._detect_pattern_type(bounce_data, bounce_data.index[100])
        assert pattern == PatternType.BOUNCE
    
    def test_volume_surge_detection(self, scanner):
        """Test volume surge detection logic."""
        # Create data with volume surge
        timestamps = pd.date_range('2025-01-15 09:30:00', periods=120, freq='1s')
        normal_volume = 10000
        
        volumes = np.ones(120) * normal_volume
        volumes[60:80] = normal_volume * 3  # 3x surge
        
        df = pd.DataFrame({
            'volume': volumes,
            'close': 10.0
        }, index=timestamps)
        
        # Should detect surge
        surge_detected = scanner._detect_volume_surge(df, timestamps[70])
        assert surge_detected is True
        
        # Should not detect surge in normal period
        no_surge = scanner._detect_volume_surge(df, timestamps[30])
        assert no_surge is False
    
    def test_categorization_logic(self, scanner):
        """Test reset point categorization based on quality scores."""
        reset_points = [
            CategorizedResetPoint(
                timestamp=datetime.now(),
                momentum_phase=MomentumPhase.FRONT_SIDE_BREAKOUT,
                quality_score=0.9,
                volume_ratio=3.0,
                pattern_type=PatternType.BREAKOUT
            ),
            CategorizedResetPoint(
                timestamp=datetime.now(),
                momentum_phase=MomentumPhase.CONSOLIDATION,
                quality_score=0.7,
                volume_ratio=1.5,
                pattern_type=PatternType.ACCUMULATION
            ),
            CategorizedResetPoint(
                timestamp=datetime.now(),
                momentum_phase=MomentumPhase.BACK_SIDE_FLUSH,
                quality_score=0.5,
                volume_ratio=2.0,
                pattern_type=PatternType.FLUSH
            ),
            CategorizedResetPoint(
                timestamp=datetime.now(),
                momentum_phase=MomentumPhase.DEAD,
                quality_score=0.2,
                volume_ratio=0.5,
                pattern_type=None
            )
        ]
        
        categorized = scanner._categorize_reset_points(reset_points)
        
        assert len(categorized['prime_momentum']) == 1
        assert len(categorized['secondary_momentum']) == 1
        assert len(categorized['risk_scenarios']) == 1
        assert len(categorized['dead_zones']) == 1
    
    def test_edge_case_insufficient_data(self, scanner, mock_data_manager):
        """Test handling of insufficient data."""
        # Return very short data
        short_data = pd.DataFrame({
            'close': [10.0, 10.1],
            'volume': [1000, 1100]
        }, index=pd.date_range('2025-01-15 09:30:00', periods=2, freq='1s'))
        
        mock_data_manager.get_day_data.return_value = short_data
        
        result = scanner.scan_single_day('MLGO', datetime(2025, 1, 15))
        
        # Should handle gracefully, maybe only time-based points
        assert len(result['time_based']) > 0
        assert len(result['prime_momentum']) == 0  # Not enough data for patterns
    
    def test_concurrent_scanning_support(self, scanner, mock_data_manager):
        """Test that scanner supports concurrent operation."""
        # This tests the design consideration for parallel scanning
        dates = [datetime(2025, 1, i) for i in range(15, 20)]
        
        # Scanner should be stateless and support multiple calls
        results = []
        for date in dates:
            mock_data_manager.get_day_data.return_value = self._create_simple_day_data(date)
            result = scanner.scan_single_day('MLGO', date)
            results.append(result)
        
        assert len(results) == 5
        # Each result should be independent
        for i, result in enumerate(results):
            assert result is not results[(i + 1) % 5]
    
    def test_metadata_generation(self, scanner):
        """Test that reset points contain proper metadata."""
        reset_point = CategorizedResetPoint(
            timestamp=datetime(2025, 1, 15, 9, 30),
            momentum_phase=MomentumPhase.FRONT_SIDE_BREAKOUT,
            quality_score=0.95,
            volume_ratio=4.5,
            pattern_type=PatternType.BREAKOUT,
            metadata={
                'pre_consolidation_minutes': 10,
                'breakout_magnitude': 0.025,
                'volume_acceleration': 2.5,
                'spread_tightness': 0.001
            }
        )
        
        assert reset_point.metadata['pre_consolidation_minutes'] == 10
        assert reset_point.metadata['breakout_magnitude'] == 0.025
        assert 'volume_acceleration' in reset_point.metadata
        assert 'spread_tightness' in reset_point.metadata
    
    # Helper methods for pattern creation
    def _create_breakout_pattern(self) -> pd.DataFrame:
        """Create data exhibiting a breakout pattern."""
        timestamps = pd.date_range('2025-01-15 09:30:00', periods=600, freq='1s')
        prices = np.ones(600) * 10.0
        
        # Consolidation phase
        prices[:300] = 10.0 + np.random.normal(0, 0.001, 300)
        
        # Breakout
        prices[300:] = 10.0 + np.linspace(0, 0.5, 300)
        
        volumes = np.ones(600) * 10000
        volumes[300:] = 50000  # Volume surge on breakout
        
        return pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': prices * 1.001,
            'low': prices * 0.999
        }, index=timestamps)
    
    def _create_flush_pattern(self) -> pd.DataFrame:
        """Create data exhibiting a flush pattern."""
        timestamps = pd.date_range('2025-01-15 10:00:00', periods=300, freq='1s')
        
        # Start near highs then flush
        prices = 10.5 - np.linspace(0, 0.4, 300)
        volumes = np.linspace(20000, 80000, 300)  # Increasing volume on flush
        
        return pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': prices * 1.001,
            'low': prices * 0.999
        }, index=timestamps)
    
    def _create_bounce_pattern(self) -> pd.DataFrame:
        """Create data exhibiting a bounce pattern."""
        timestamps = pd.date_range('2025-01-15 11:00:00', periods=300, freq='1s')
        
        # Sharp decline then bounce
        prices = np.concatenate([
            10.0 - np.linspace(0, 0.3, 150),  # Decline
            9.7 + np.linspace(0, 0.1, 150)    # Bounce
        ])
        
        return pd.DataFrame({
            'close': prices,
            'volume': 30000,
            'high': prices * 1.001,
            'low': prices * 0.999
        }, index=timestamps)
    
    def _create_simple_day_data(self, date: datetime) -> pd.DataFrame:
        """Create simple day data for testing."""
        start = date.replace(hour=4, minute=0)
        end = date.replace(hour=20, minute=0)
        timestamps = pd.date_range(start=start, end=end, freq='1min')
        
        return pd.DataFrame({
            'close': 10.0,
            'volume': 10000,
            'high': 10.01,
            'low': 9.99
        }, index=timestamps)


class TestCategorizedResetPoint:
    """Test the CategorizedResetPoint data class."""
    
    def test_reset_point_creation(self):
        """Test creating a reset point with all attributes."""
        timestamp = datetime(2025, 1, 15, 9, 30)
        
        reset_point = CategorizedResetPoint(
            timestamp=timestamp,
            momentum_phase=MomentumPhase.FRONT_SIDE_BREAKOUT,
            quality_score=0.92,
            volume_ratio=3.5,
            pattern_type=PatternType.BREAKOUT,
            metadata={'test': 'value'}
        )
        
        assert reset_point.timestamp == timestamp
        assert reset_point.momentum_phase == MomentumPhase.FRONT_SIDE_BREAKOUT
        assert reset_point.quality_score == 0.92
        assert reset_point.volume_ratio == 3.5
        assert reset_point.pattern_type == PatternType.BREAKOUT
        assert reset_point.metadata['test'] == 'value'
    
    def test_reset_point_comparison(self):
        """Test reset point comparison by quality score."""
        high_quality = CategorizedResetPoint(
            timestamp=datetime.now(),
            momentum_phase=MomentumPhase.FRONT_SIDE_BREAKOUT,
            quality_score=0.95,
            volume_ratio=4.0,
            pattern_type=PatternType.BREAKOUT
        )
        
        low_quality = CategorizedResetPoint(
            timestamp=datetime.now(),
            momentum_phase=MomentumPhase.DEAD,
            quality_score=0.2,
            volume_ratio=0.5,
            pattern_type=None
        )
        
        # Should be comparable by quality score
        assert high_quality > low_quality
        assert low_quality < high_quality
    
    def test_reset_point_serialization(self):
        """Test reset point can be serialized for storage."""
        reset_point = CategorizedResetPoint(
            timestamp=datetime(2025, 1, 15, 9, 30),
            momentum_phase=MomentumPhase.FRONT_SIDE_BREAKOUT,
            quality_score=0.92,
            volume_ratio=3.5,
            pattern_type=PatternType.BREAKOUT,
            metadata={'consolidation_minutes': 5}
        )
        
        # Should be serializable
        data = reset_point.to_dict()
        assert data['timestamp'] == '2025-01-15T09:30:00'
        assert data['momentum_phase'] == 'front_breakout'
        assert data['quality_score'] == 0.92
        
        # Should be deserializable
        loaded = CategorizedResetPoint.from_dict(data)
        assert loaded.timestamp == reset_point.timestamp
        assert loaded.momentum_phase == reset_point.momentum_phase