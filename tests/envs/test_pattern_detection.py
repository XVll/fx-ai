import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Optional, Tuple

from envs.pattern_detection import (
    PatternDetector,
    PatternType,
    Pattern,
    PatternQuality,
    BreakoutPattern,
    FlushPattern,
    BouncePattern,
    AccumulationPattern,
    DistributionPattern
)


class TestPatternDetector:
    """Test suite for pattern detection system."""
    
    @pytest.fixture
    def detector_config(self):
        """Configuration for pattern detector."""
        return {
            'patterns': {
                'breakout': {
                    'consolidation_minutes': 5,
                    'volume_required': 3.0,
                    'price_expansion_min': 0.003,
                    'atr_expansion_min': 1.5
                },
                'flush': {
                    'from_high_threshold': 0.02,
                    'volume_required': 2.0,
                    'velocity_threshold': -0.002,
                    'duration_max_seconds': 300
                },
                'bounce': {
                    'prior_decline_min': 0.03,
                    'recovery_min': 0.005,
                    'volume_spike_min': 2.5,
                    'time_window': 600
                },
                'accumulation': {
                    'volatility_threshold': 0.001,
                    'near_vwap_threshold': 0.002,
                    'bid_ask_ratio_min': 1.2,
                    'min_duration_minutes': 10
                },
                'distribution': {
                    'near_high_threshold': 0.005,
                    'momentum_decay_rate': 0.5,
                    'volume_decline_rate': 0.3,
                    'min_duration_minutes': 5
                }
            },
            'quality_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }
    
    @pytest.fixture
    def pattern_detector(self, detector_config):
        """Create pattern detector instance."""
        return PatternDetector(detector_config)
    
    @pytest.fixture
    def market_data_generator(self):
        """Utility to generate market data for testing."""
        def generate(start_time: datetime, duration_seconds: int, 
                    pattern_type: Optional[str] = None) -> pd.DataFrame:
            timestamps = pd.date_range(
                start=start_time, 
                periods=duration_seconds, 
                freq='1s'
            )
            
            n = len(timestamps)
            base_price = 10.0
            
            if pattern_type == 'breakout':
                # Consolidation then breakout
                prices = np.ones(n) * base_price
                prices[:n//2] += np.random.normal(0, 0.0001, n//2)  # Tight range
                prices[n//2:] = base_price + np.linspace(0, 0.05, n//2)  # Breakout
                
                volumes = np.ones(n) * 10000
                volumes[n//2:] *= 4  # Volume surge
                
            elif pattern_type == 'flush':
                # Sharp decline from highs
                prices = base_price * 1.03 - np.linspace(0, 0.05, n)
                volumes = np.linspace(20000, 80000, n)  # Increasing volume
                
            elif pattern_type == 'bounce':
                # Decline then recovery
                prices = np.concatenate([
                    base_price - np.linspace(0, 0.04, n//2),
                    base_price * 0.96 + np.linspace(0, 0.015, n//2)
                ])
                volumes = np.ones(n) * 15000
                volumes[n//2:n//2+60] *= 3  # Volume spike at bounce
                
            else:
                # Random walk
                prices = base_price + np.cumsum(np.random.normal(0, 0.001, n))
                volumes = np.random.uniform(8000, 12000, n)
            
            # Calculate derived values
            highs = prices * (1 + np.random.uniform(0, 0.001, n))
            lows = prices * (1 - np.random.uniform(0, 0.001, n))
            
            # VWAP calculation
            cumulative_pv = np.cumsum(prices * volumes)
            cumulative_v = np.cumsum(volumes)
            vwap = cumulative_pv / cumulative_v
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes,
                'bid': prices * 0.999,
                'ask': prices * 1.001,
                'bid_size': volumes * 0.45,
                'ask_size': volumes * 0.55,
                'vwap': vwap,
                'trades': np.random.randint(50, 200, n)
            }, index=timestamps)
        
        return generate
    
    def test_pattern_detector_initialization(self, pattern_detector, detector_config):
        """Test pattern detector initialization."""
        assert pattern_detector.config == detector_config
        assert hasattr(pattern_detector, 'detect_patterns')
        assert hasattr(pattern_detector, 'calculate_pattern_quality')
    
    def test_breakout_pattern_detection(self, pattern_detector, market_data_generator):
        """Test detection of breakout patterns."""
        # Generate breakout pattern data
        start_time = datetime(2025, 1, 15, 9, 30)
        data = market_data_generator(start_time, 600, 'breakout')
        
        # Detect patterns in the window
        patterns = pattern_detector.detect_patterns(
            data, 
            start_time + timedelta(seconds=300),
            lookback_seconds=300
        )
        
        # Should detect breakout pattern
        breakout_patterns = [p for p in patterns if p.pattern_type == PatternType.BREAKOUT]
        assert len(breakout_patterns) > 0
        
        breakout = breakout_patterns[0]
        assert isinstance(breakout, BreakoutPattern)
        assert breakout.consolidation_duration >= 150  # At least half the pre-breakout period
        assert breakout.volume_surge_ratio >= 3.0
        assert breakout.price_expansion >= 0.003
    
    def test_flush_pattern_detection(self, pattern_detector, market_data_generator):
        """Test detection of flush patterns."""
        start_time = datetime(2025, 1, 15, 10, 0)
        data = market_data_generator(start_time, 300, 'flush')
        
        patterns = pattern_detector.detect_patterns(
            data,
            start_time + timedelta(seconds=150),
            lookback_seconds=150
        )
        
        flush_patterns = [p for p in patterns if p.pattern_type == PatternType.FLUSH]
        assert len(flush_patterns) > 0
        
        flush = flush_patterns[0]
        assert isinstance(flush, FlushPattern)
        assert flush.decline_percent >= 0.02
        assert flush.velocity < -0.002
        assert flush.from_session_high is True
    
    def test_bounce_pattern_detection(self, pattern_detector, market_data_generator):
        """Test detection of bounce patterns."""
        start_time = datetime(2025, 1, 15, 11, 0)
        data = market_data_generator(start_time, 600, 'bounce')
        
        patterns = pattern_detector.detect_patterns(
            data,
            start_time + timedelta(seconds=400),
            lookback_seconds=400
        )
        
        bounce_patterns = [p for p in patterns if p.pattern_type == PatternType.BOUNCE]
        assert len(bounce_patterns) > 0
        
        bounce = bounce_patterns[0]
        assert isinstance(bounce, BouncePattern)
        assert bounce.prior_decline >= 0.03
        assert bounce.recovery_percent >= 0.005
        assert bounce.volume_at_turn > bounce.average_volume * 2
    
    def test_accumulation_pattern_detection(self, pattern_detector):
        """Test detection of accumulation patterns."""
        # Create accumulation pattern - low volatility near VWAP
        timestamps = pd.date_range('2025-01-15 11:00:00', periods=900, freq='1s')
        n = len(timestamps)
        
        vwap = 10.0
        prices = vwap + np.random.normal(0, 0.0005, n)  # Very tight range
        volumes = np.ones(n) * 12000
        
        # Simulate accumulation - more bids than asks
        bid_sizes = volumes * np.random.uniform(0.55, 0.65, n)
        ask_sizes = volumes * np.random.uniform(0.35, 0.45, n)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'bid': prices * 0.999,
            'ask': prices * 1.001,
            'bid_size': bid_sizes,
            'ask_size': ask_sizes,
            'vwap': vwap,
            'high': prices + 0.001,
            'low': prices - 0.001
        }, index=timestamps)
        
        patterns = pattern_detector.detect_patterns(
            data,
            timestamps[600],
            lookback_seconds=600
        )
        
        acc_patterns = [p for p in patterns if p.pattern_type == PatternType.ACCUMULATION]
        assert len(acc_patterns) > 0
        
        accumulation = acc_patterns[0]
        assert isinstance(accumulation, AccumulationPattern)
        assert accumulation.volatility < 0.001
        assert accumulation.bid_ask_ratio > 1.2
        assert accumulation.distance_from_vwap < 0.002
    
    def test_distribution_pattern_detection(self, pattern_detector):
        """Test detection of distribution patterns."""
        # Create distribution pattern - near highs with declining momentum
        timestamps = pd.date_range('2025-01-15 14:00:00', periods=600, freq='1s')
        n = len(timestamps)
        
        # Price near highs but momentum declining
        session_high = 10.5
        prices = session_high - np.linspace(0, 0.02, n)
        prices += np.random.normal(0, 0.001, n)
        
        # Volume declining
        volumes = np.linspace(30000, 15000, n)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': session_high,
            'low': prices - 0.01,
            'bid': prices * 0.999,
            'ask': prices * 1.001,
            'vwap': 10.2
        }, index=timestamps)
        
        patterns = pattern_detector.detect_patterns(
            data,
            timestamps[400],
            lookback_seconds=300
        )
        
        dist_patterns = [p for p in patterns if p.pattern_type == PatternType.DISTRIBUTION]
        assert len(dist_patterns) > 0
        
        distribution = dist_patterns[0]
        assert isinstance(distribution, DistributionPattern)
        assert distribution.distance_from_high < 0.005
        assert distribution.momentum_decay > 0.3
        assert distribution.volume_decline > 0.2
    
    def test_pattern_quality_scoring(self, pattern_detector):
        """Test pattern quality scoring system."""
        # Create a high-quality breakout pattern
        pattern = BreakoutPattern(
            timestamp=datetime.now(),
            pattern_type=PatternType.BREAKOUT,
            confidence=0.95,
            consolidation_duration=300,
            volume_surge_ratio=5.0,
            price_expansion=0.008,
            atr_expansion=2.0,
            pre_breakout_volatility=0.0003
        )
        
        quality = pattern_detector.calculate_pattern_quality(pattern)
        assert quality == PatternQuality.HIGH
        assert pattern.quality_score > 0.8
        
        # Create a low-quality pattern
        weak_pattern = BreakoutPattern(
            timestamp=datetime.now(),
            pattern_type=PatternType.BREAKOUT,
            confidence=0.6,
            consolidation_duration=60,  # Too short
            volume_surge_ratio=1.5,     # Weak volume
            price_expansion=0.002,      # Small move
            atr_expansion=1.1,          # Low expansion
            pre_breakout_volatility=0.002
        )
        
        quality = pattern_detector.calculate_pattern_quality(weak_pattern)
        assert quality == PatternQuality.LOW
        assert weak_pattern.quality_score < 0.6
    
    def test_multiple_pattern_detection(self, pattern_detector, market_data_generator):
        """Test detection of multiple patterns in same window."""
        # Create data with multiple patterns
        timestamps = pd.date_range('2025-01-15 09:30:00', periods=1800, freq='1s')
        n = len(timestamps)
        
        prices = np.ones(n) * 10.0
        volumes = np.ones(n) * 10000
        
        # Add consolidation (0-600s)
        prices[:600] += np.random.normal(0, 0.0001, 600)
        
        # Add breakout (600-900s)
        prices[600:900] = 10.0 + np.linspace(0, 0.04, 300)
        volumes[600:900] *= 4
        
        # Add flush (1200-1500s)
        prices[1200:1500] = 10.04 - np.linspace(0, 0.05, 300)
        volumes[1200:1500] = np.linspace(20000, 60000, 300)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'bid': prices * 0.999,
            'ask': prices * 1.001,
            'vwap': 10.0
        }, index=timestamps)
        
        # Detect patterns at different points
        patterns_at_breakout = pattern_detector.detect_patterns(
            data, timestamps[700], lookback_seconds=300
        )
        patterns_at_flush = pattern_detector.detect_patterns(
            data, timestamps[1300], lookback_seconds=300
        )
        
        # Should find appropriate patterns
        breakout_found = any(p.pattern_type == PatternType.BREAKOUT for p in patterns_at_breakout)
        flush_found = any(p.pattern_type == PatternType.FLUSH for p in patterns_at_flush)
        
        assert breakout_found
        assert flush_found
    
    def test_pattern_validation(self, pattern_detector):
        """Test pattern validation logic."""
        # Test invalid consolidation duration
        with pytest.raises(ValueError):
            BreakoutPattern(
                timestamp=datetime.now(),
                pattern_type=PatternType.BREAKOUT,
                confidence=0.9,
                consolidation_duration=-10,  # Invalid
                volume_surge_ratio=3.0,
                price_expansion=0.005,
                atr_expansion=1.5,
                pre_breakout_volatility=0.001
            )
        
        # Test invalid volume ratio
        with pytest.raises(ValueError):
            FlushPattern(
                timestamp=datetime.now(),
                pattern_type=PatternType.FLUSH,
                confidence=0.9,
                decline_percent=0.03,
                velocity=-0.003,
                volume_ratio=0.5,  # Too low
                from_session_high=True,
                duration_seconds=120
            )
    
    def test_pattern_edge_cases(self, pattern_detector):
        """Test edge cases in pattern detection."""
        # Test with minimal data
        timestamps = pd.date_range('2025-01-15 09:30:00', periods=10, freq='1s')
        minimal_data = pd.DataFrame({
            'close': 10.0,
            'volume': 10000,
            'high': 10.01,
            'low': 9.99,
            'bid': 9.99,
            'ask': 10.01,
            'vwap': 10.0
        }, index=timestamps)
        
        patterns = pattern_detector.detect_patterns(
            minimal_data, timestamps[5], lookback_seconds=5
        )
        
        # Should handle gracefully, likely no patterns
        assert isinstance(patterns, list)
        assert len(patterns) == 0
        
        # Test with missing data
        missing_data = minimal_data.copy()
        missing_data.loc[timestamps[3:6], 'volume'] = np.nan
        
        patterns = pattern_detector.detect_patterns(
            missing_data, timestamps[8], lookback_seconds=5
        )
        
        # Should handle NaN values
        assert isinstance(patterns, list)
    
    def test_pattern_persistence(self, pattern_detector):
        """Test pattern persistence and evolution."""
        # Create evolving breakout pattern
        timestamps = pd.date_range('2025-01-15 09:30:00', periods=600, freq='1s')
        n = len(timestamps)
        
        # Consolidation that gradually tightens
        prices = np.ones(n) * 10.0
        for i in range(0, 400, 50):
            volatility = 0.002 * (1 - i/400)  # Decreasing volatility
            prices[i:i+50] += np.random.normal(0, volatility, 50)
        
        # Then breakout
        prices[400:] = 10.0 + np.linspace(0, 0.04, 200)
        
        volumes = np.ones(n) * 10000
        volumes[400:] *= 4
        
        data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'vwap': 10.0
        }, index=timestamps)
        
        # Detect pattern evolution
        early_patterns = pattern_detector.detect_patterns(
            data, timestamps[200], lookback_seconds=200
        )
        late_patterns = pattern_detector.detect_patterns(
            data, timestamps[350], lookback_seconds=200
        )
        breakout_patterns = pattern_detector.detect_patterns(
            data, timestamps[450], lookback_seconds=200
        )
        
        # Should show pattern evolution
        assert len(early_patterns) <= len(late_patterns)  # More defined pattern later
        assert len(breakout_patterns) > 0
        assert any(p.pattern_type == PatternType.BREAKOUT for p in breakout_patterns)
    
    def test_pattern_metadata(self, pattern_detector):
        """Test pattern metadata generation."""
        pattern = BreakoutPattern(
            timestamp=datetime(2025, 1, 15, 9, 45),
            pattern_type=PatternType.BREAKOUT,
            confidence=0.92,
            consolidation_duration=300,
            volume_surge_ratio=4.5,
            price_expansion=0.006,
            atr_expansion=1.8,
            pre_breakout_volatility=0.0004,
            metadata={
                'volume_acceleration': 2.5,
                'breakout_angle': 35,
                'false_breakout_probability': 0.15,
                'similar_patterns_today': 2
            }
        )
        
        assert pattern.metadata['volume_acceleration'] == 2.5
        assert pattern.metadata['breakout_angle'] == 35
        assert 'false_breakout_probability' in pattern.metadata
        
        # Test serialization
        pattern_dict = pattern.to_dict()
        assert pattern_dict['timestamp'] == '2025-01-15T09:45:00'
        assert pattern_dict['pattern_type'] == 'BREAKOUT'
        assert pattern_dict['confidence'] == 0.92
        assert 'metadata' in pattern_dict


class TestPatternInteraction:
    """Test pattern interactions and complex scenarios."""
    
    @pytest.fixture
    def pattern_detector(self, detector_config):
        """Create pattern detector instance."""
        return PatternDetector(detector_config)
    
    def test_pattern_sequence_detection(self, pattern_detector):
        """Test detection of pattern sequences (e.g., consolidation → breakout → distribution)."""
        timestamps = pd.date_range('2025-01-15 09:30:00', periods=3600, freq='1s')
        n = len(timestamps)
        
        prices = np.ones(n) * 10.0
        volumes = np.ones(n) * 10000
        
        # Phase 1: Accumulation (0-1200s)
        prices[:1200] += np.random.normal(0, 0.0003, 1200)
        
        # Phase 2: Breakout (1200-1800s)
        prices[1200:1800] = 10.0 + np.linspace(0, 0.06, 600)
        volumes[1200:1800] *= 5
        
        # Phase 3: Distribution (1800-2400s)
        prices[1800:2400] = 10.06 + np.random.normal(0, 0.001, 600)
        volumes[1800:2400] = np.linspace(40000, 20000, 600)
        
        # Phase 4: Flush (2400-3000s)
        prices[2400:3000] = 10.06 - np.linspace(0, 0.07, 600)
        volumes[2400:3000] = np.linspace(30000, 80000, 600)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': np.maximum.accumulate(prices),
            'low': prices * 0.998,
            'bid': prices * 0.999,
            'ask': prices * 1.001,
            'bid_size': volumes * 0.5,
            'ask_size': volumes * 0.5,
            'vwap': 10.0
        }, index=timestamps)
        
        # Detect patterns at each phase
        phase_results = []
        for phase_time in [600, 1500, 2100, 2700]:
            patterns = pattern_detector.detect_patterns(
                data, timestamps[phase_time], lookback_seconds=300
            )
            phase_results.append(patterns)
        
        # Verify sequence
        assert any(p.pattern_type == PatternType.ACCUMULATION for p in phase_results[0])
        assert any(p.pattern_type == PatternType.BREAKOUT for p in phase_results[1])
        assert any(p.pattern_type == PatternType.DISTRIBUTION for p in phase_results[2])
        assert any(p.pattern_type == PatternType.FLUSH for p in phase_results[3])
    
    def test_failed_pattern_detection(self, pattern_detector):
        """Test detection of failed patterns (e.g., failed breakout)."""
        timestamps = pd.date_range('2025-01-15 10:00:00', periods=600, freq='1s')
        n = len(timestamps)
        
        prices = np.ones(n) * 10.0
        volumes = np.ones(n) * 10000
        
        # Consolidation
        prices[:300] += np.random.normal(0, 0.0003, 300)
        
        # Failed breakout - starts to break then reverses
        prices[300:350] = 10.0 + np.linspace(0, 0.02, 50)
        volumes[300:350] *= 3
        
        # Reversal
        prices[350:450] = 10.02 - np.linspace(0, 0.025, 100)
        volumes[350:450] *= 2
        
        data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'vwap': 10.0
        }, index=timestamps)
        
        # Detect at failed breakout point
        patterns = pattern_detector.detect_patterns(
            data, timestamps[400], lookback_seconds=200
        )
        
        # Should detect the failed breakout or subsequent flush
        assert len(patterns) > 0
        
        # Check for failed breakout metadata
        for pattern in patterns:
            if hasattr(pattern, 'metadata') and pattern.metadata:
                if 'failed_breakout' in pattern.metadata:
                    assert pattern.metadata['failed_breakout'] is True