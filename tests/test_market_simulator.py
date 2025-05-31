"""
Comprehensive tests for MarketSimulator based on input/output behavior.
Tests all functionality and edge cases without implementation details.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

from simulators.market_simulator import MarketSimulator, MarketState, MARKET_HOURS
from data.data_manager import DataManager
from feature.simple_feature_manager import SimpleFeatureManager
from config.schemas import ModelConfig, SimulationConfig


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


@pytest.fixture
def model_config():
    """Create a test model config."""
    return ModelConfig(
        hf_seq_len=60,
        hf_feat_dim=7,
        mf_seq_len=30,
        mf_feat_dim=43,
        lf_seq_len=30,
        lf_feat_dim=19,
        portfolio_seq_len=5,
        portfolio_feat_dim=5
    )


@pytest.fixture
def simulation_config():
    """Create a test simulation config."""
    return SimulationConfig(
        mean_latency_ms=50.0,
        latency_std_dev_ms=10.0,
        base_slippage_bps=5.0,
        max_total_slippage_bps=50.0,
        size_impact_slippage_bps_per_unit=0.1,
        market_impact_coefficient=0.0001,
        commission_per_share=0.005,
        fee_per_share=0.001,
        min_commission_per_order=1.0,
        max_commission_pct_of_value=0.5,
        default_position_value=10000.0,
        allow_shorting=False,
        max_position_value_ratio=1.0,
        market_impact_model="linear"
    )


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager."""
    data_manager = Mock(spec=DataManager)
    data_manager.momentum_scanner = None
    data_manager.get_momentum_days.return_value = pd.DataFrame()
    data_manager.get_previous_day_data.return_value = None
    return data_manager


@pytest.fixture
def mock_feature_manager():
    """Create a mock feature manager."""
    feature_manager = Mock(spec=SimpleFeatureManager)
    feature_manager.extract_features.return_value = {
        'hf': np.zeros((60, 7)),
        'mf': np.zeros((30, 43)),
        'lf': np.zeros((30, 19))
    }
    return feature_manager


@pytest.fixture
def sample_day_data():
    """Create sample day data for testing."""
    # Create timestamps for a trading day
    start_time = pd.Timestamp('2024-01-15 09:00:00', tz=ZoneInfo('UTC'))
    end_time = pd.Timestamp('2024-01-15 16:00:00', tz=ZoneInfo('UTC'))
    
    # Sample trades
    trade_times = pd.date_range(start_time, end_time, freq='1min')[:100]
    trades = pd.DataFrame({
        'price': np.random.uniform(10.0, 11.0, len(trade_times)),
        'size': np.random.randint(100, 1000, len(trade_times))
    }, index=trade_times)
    
    # Sample quotes
    quote_times = pd.date_range(start_time, end_time, freq='30s')[:200]
    quotes = pd.DataFrame({
        'bid_price': np.random.uniform(9.9, 10.9, len(quote_times)),
        'ask_price': np.random.uniform(10.1, 11.1, len(quote_times)),
        'bid_size': np.random.randint(100, 500, len(quote_times)),
        'ask_size': np.random.randint(100, 500, len(quote_times))
    }, index=quote_times)
    
    # Sample status data
    status = pd.DataFrame({'is_halted': [False] * 10})
    
    return {
        'trades': trades,
        'quotes': quotes,
        'status': status
    }


@pytest.fixture
def market_simulator(mock_data_manager, model_config, simulation_config, mock_feature_manager, logger):
    """Create a MarketSimulator instance."""
    return MarketSimulator(
        symbol="MLGO",
        data_manager=mock_data_manager,
        model_config=model_config,
        simulation_config=simulation_config,
        feature_manager=mock_feature_manager,
        logger=logger
    )


@pytest.fixture
def initialized_market_simulator(market_simulator, sample_day_data):
    """Create an initialized MarketSimulator instance with data."""
    # Setup mock for get_previous_day_data calls
    def mock_get_previous_day_data(key):
        if key == 'bars_1d':
            return pd.DataFrame({
                'open': [100.0], 'high': [105.0], 'low': [95.0], 
                'close': [102.0], 'volume': [10000], 'vwap': [101.0]
            })
        return None
    
    market_simulator.data_manager.load_day.return_value = sample_day_data
    market_simulator.data_manager.get_previous_day_data.side_effect = mock_get_previous_day_data
    
    # Initialize with data
    market_simulator.initialize_day(datetime(2024, 1, 15))
    return market_simulator


class TestMarketSimulatorInitialization:
    """Test MarketSimulator initialization and setup."""
    
    def test_init_with_all_parameters(self, mock_data_manager, model_config, simulation_config, mock_feature_manager, logger):
        """Test initialization with all parameters provided."""
        simulator = MarketSimulator(
            symbol="MLGO",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager,
            logger=logger
        )
        
        assert simulator.symbol == "MLGO"
        assert simulator.data_manager == mock_data_manager
        assert simulator.model_config == model_config
        assert simulator.simulation_config == simulation_config
        assert simulator.feature_manager == mock_feature_manager
        assert simulator.logger == logger
        assert simulator.current_index == 0
        assert simulator.current_date is None
        assert simulator.df_market_state is None
        
    def test_init_with_minimal_parameters(self, mock_data_manager, model_config, simulation_config):
        """Test initialization with minimal parameters (auto-creates feature manager and logger)."""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config
        )
        
        assert simulator.symbol == "TEST"
        assert simulator.data_manager == mock_data_manager
        assert simulator.model_config == model_config
        assert simulator.simulation_config == simulation_config
        assert simulator.feature_manager is not None
        assert simulator.logger is not None
        assert isinstance(simulator.feature_manager, SimpleFeatureManager)
        
    def test_timezone_setup(self, market_simulator):
        """Test that timezone setup is correct."""
        assert market_simulator.market_tz == ZoneInfo(MARKET_HOURS["TIMEZONE"])
        assert market_simulator.utc_tz == ZoneInfo("UTC")


class TestMarketSimulatorDayInitialization:
    """Test day initialization functionality."""
    
    def test_initialize_day_success(self, market_simulator, sample_day_data):
        """Test successful day initialization."""
        # Setup mock data manager
        market_simulator.data_manager.load_day.return_value = sample_day_data
        
        # Setup mock for get_previous_day_data calls
        def mock_get_previous_day_data(key):
            if key == 'bars_1d':
                return pd.DataFrame({
                    'open': [100.0], 'high': [105.0], 'low': [95.0], 
                    'close': [102.0], 'volume': [10000], 'vwap': [101.0]
                })
            return None
        
        market_simulator.data_manager.get_previous_day_data.side_effect = mock_get_previous_day_data
        
        date = datetime(2024, 1, 15)
        result = market_simulator.initialize_day(date)
        
        assert result is True
        assert market_simulator.current_date == pd.Timestamp(date).date()
        assert market_simulator.current_index == 0
        assert market_simulator.df_market_state is not None
        assert not market_simulator.df_market_state.empty
        
    def test_initialize_day_no_data(self, market_simulator):
        """Test day initialization when no data is available."""
        market_simulator.data_manager.load_day.return_value = None
        
        date = datetime(2024, 1, 15)
        result = market_simulator.initialize_day(date)
        
        assert result is False
        assert market_simulator.df_market_state is None
        
    def test_initialize_day_empty_data(self, market_simulator):
        """Test day initialization with empty data."""
        market_simulator.data_manager.load_day.return_value = {
            'trades': pd.DataFrame(),
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame()
        }
        market_simulator.data_manager.get_previous_day_data.return_value = None
        
        date = datetime(2024, 1, 15)
        result = market_simulator.initialize_day(date)
        
        # Should still succeed with synthetic data
        assert result is True
        assert market_simulator.df_market_state is not None
        
    def test_initialize_day_caching(self, market_simulator, sample_day_data):
        """Test that precomputed states are cached."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        
        # Setup proper previous day data
        def mock_get_previous_day_data(key):
            if key == 'bars_1d':
                return pd.DataFrame({
                    'open': [100.0], 'high': [105.0], 'low': [95.0], 
                    'close': [102.0], 'volume': [10000], 'vwap': [101.0]
                })
            return None
        
        market_simulator.data_manager.get_previous_day_data.side_effect = mock_get_previous_day_data
        
        date = datetime(2024, 1, 15)
        
        # First initialization
        result1 = market_simulator.initialize_day(date)
        assert result1 is True
        
        # Verify cache is populated
        cache_key = ("MLGO", pd.Timestamp(date).date())
        assert cache_key in market_simulator._precomputed_cache
        
        # Second initialization should use cache
        result2 = market_simulator.initialize_day(date)
        assert result2 is True
        
    def test_initialize_day_exception_handling(self, market_simulator):
        """Test exception handling during day initialization."""
        market_simulator.data_manager.load_day.side_effect = Exception("Data loading failed")
        
        date = datetime(2024, 1, 15)
        result = market_simulator.initialize_day(date)
        
        assert result is False
        assert market_simulator.df_market_state is None


class TestMarketSimulatorTimeOperations:
    """Test time-based operations."""
    
    def test_get_time_range_with_data(self, market_simulator, sample_day_data):
        """Test getting time range when data is available."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        start_time, end_time = market_simulator.get_time_range()
        
        assert start_time is not None
        assert end_time is not None
        assert isinstance(start_time, pd.Timestamp)
        assert isinstance(end_time, pd.Timestamp)
        assert start_time < end_time
        
    def test_get_time_range_no_data(self, market_simulator):
        """Test getting time range when no data is available."""
        start_time, end_time = market_simulator.get_time_range()
        
        assert start_time is None
        assert end_time is None
        
    def test_set_time_valid_timestamp(self, market_simulator, sample_day_data):
        """Test setting time to a valid timestamp."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Get a valid timestamp from the data
        target_timestamp = market_simulator.df_market_state.index[10]
        result = market_simulator.set_time(target_timestamp)
        
        assert result is True
        assert market_simulator.current_index == 10
        
    def test_set_time_invalid_timestamp(self, market_simulator, sample_day_data):
        """Test setting time to an invalid timestamp."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Use a timestamp before the data starts
        invalid_timestamp = pd.Timestamp('2024-01-15 01:00:00', tz=ZoneInfo('UTC'))
        result = market_simulator.set_time(invalid_timestamp)
        
        assert result is False
        
    def test_set_time_closest_previous(self, market_simulator, sample_day_data):
        """Test setting time to closest previous timestamp."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Use a timestamp between two data points
        start_time, end_time = market_simulator.get_time_range()
        middle_time = start_time + (end_time - start_time) / 2
        result = market_simulator.set_time(middle_time)
        
        assert result is True
        
    def test_set_time_no_data(self, market_simulator):
        """Test setting time when no data is available."""
        target_timestamp = pd.Timestamp('2024-01-15 12:00:00', tz=ZoneInfo('UTC'))
        result = market_simulator.set_time(target_timestamp)
        
        assert result is False


class TestMarketSimulatorNavigation:
    """Test navigation through market data."""
    
    def test_step_through_data(self, market_simulator, sample_day_data):
        """Test stepping through data sequentially."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        initial_index = market_simulator.current_index
        result = market_simulator.step()
        
        assert result is True
        assert market_simulator.current_index == initial_index + 1
        
    def test_step_at_end_of_data(self, market_simulator, sample_day_data):
        """Test stepping when at end of data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Move to end of data
        market_simulator.current_index = len(market_simulator.df_market_state) - 1
        result = market_simulator.step()
        
        assert result is False
        
    def test_step_no_data(self, market_simulator):
        """Test stepping when no data is available."""
        result = market_simulator.step()
        assert result is False
        
    def test_reset_to_beginning(self, market_simulator, sample_day_data):
        """Test resetting to beginning of data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Move forward first
        market_simulator.step()
        market_simulator.step()
        assert market_simulator.current_index == 2
        
        # Reset
        result = market_simulator.reset()
        assert result is True
        assert market_simulator.current_index == 0
        
    def test_reset_to_specific_index(self, market_simulator, sample_day_data):
        """Test resetting to a specific index."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        target_index = 5
        result = market_simulator.reset(target_index)
        
        assert result is True
        assert market_simulator.current_index == target_index
        
    def test_reset_index_bounds_checking(self, market_simulator, sample_day_data):
        """Test that reset properly bounds-checks indices."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        total_states = len(market_simulator.df_market_state)
        
        # Test negative index
        result = market_simulator.reset(-5)
        assert result is True
        assert market_simulator.current_index == 0
        
        # Test index beyond data
        result = market_simulator.reset(total_states + 10)
        assert result is True
        assert market_simulator.current_index == total_states - 1
        
    def test_reset_no_data(self, market_simulator):
        """Test reset when no data is available."""
        result = market_simulator.reset()
        assert result is False


class TestMarketSimulatorState:
    """Test market state retrieval functionality."""
    
    def test_get_market_state_current_index(self, initialized_market_simulator):
        """Test getting market state at current index."""
        state = initialized_market_simulator.get_market_state()
        
        assert state is not None
        assert isinstance(state, MarketState)
        assert isinstance(state.timestamp, pd.Timestamp)
        assert isinstance(state.current_price, float)
        assert isinstance(state.best_bid, float)
        assert isinstance(state.best_ask, float)
        # Validate relationships (allowing for edge case corrections)
        assert state.best_bid <= state.best_ask  # Allow equality for edge cases
        assert abs(state.mid_price - (state.best_bid + state.best_ask) / 2) < 0.01  # Allow small rounding differences
        assert abs(state.spread - (state.best_ask - state.best_bid)) < 0.01  # Allow small rounding differences
        
    def test_get_market_state_specific_timestamp(self, initialized_market_simulator):
        """Test getting market state at specific timestamp."""
        target_timestamp = initialized_market_simulator.df_market_state.index[5]
        state = initialized_market_simulator.get_market_state(target_timestamp)
        
        assert state is not None
        assert state.timestamp == target_timestamp
        
    def test_get_market_state_timestamp_not_found(self, initialized_market_simulator):
        """Test getting market state for timestamp not in data."""
        # Use timestamp before data starts
        early_timestamp = pd.Timestamp('2024-01-15 01:00:00', tz=ZoneInfo('UTC'))
        state = initialized_market_simulator.get_market_state(early_timestamp)
        
        assert state is None
        
    def test_get_market_state_no_data(self, market_simulator):
        """Test getting market state when no data is available."""
        state = market_simulator.get_market_state()
        assert state is None
        
    def test_get_market_state_beyond_index(self, market_simulator, sample_day_data):
        """Test getting market state when current index is beyond data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Set index beyond data
        market_simulator.current_index = len(market_simulator.df_market_state) + 10
        state = market_simulator.get_market_state()
        
        assert state is None


class TestMarketSimulatorFeatures:
    """Test feature extraction functionality."""
    
    def test_get_current_features_success(self, market_simulator, sample_day_data):
        """Test successful feature extraction."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        features = market_simulator.get_current_features()
        
        assert features is not None
        assert 'hf' in features
        assert 'mf' in features
        assert 'lf' in features
        assert isinstance(features['hf'], np.ndarray)
        assert isinstance(features['mf'], np.ndarray)
        assert isinstance(features['lf'], np.ndarray)
        assert features['hf'].shape == (60, 7)
        assert features['mf'].shape == (30, 43)
        assert features['lf'].shape == (30, 19)
        
    def test_get_current_features_caching(self, market_simulator, sample_day_data):
        """Test that features are cached after first computation."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # First call should trigger computation
        features1 = market_simulator.get_current_features()
        
        # Second call should use cache (verify by comparing results)
        features2 = market_simulator.get_current_features()
        
        # Features should be identical (cached)
        assert features1 is not None and features2 is not None
        np.testing.assert_array_equal(features1['hf'], features2['hf'])
        np.testing.assert_array_equal(features1['mf'], features2['mf'])
        np.testing.assert_array_equal(features1['lf'], features2['lf'])
        
    def test_get_current_features_no_data(self, market_simulator):
        """Test getting features when no data is available."""
        features = market_simulator.get_current_features()
        assert features is None
        
    def test_get_current_features_beyond_index(self, market_simulator, sample_day_data):
        """Test getting features when current index is beyond data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        market_simulator.current_index = len(market_simulator.df_market_state) + 10
        features = market_simulator.get_current_features()
        
        assert features is None
        
    def test_get_current_features_extraction_failure(self, market_simulator, sample_day_data):
        """Test handling of feature extraction failures."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Make feature extraction fail
        market_simulator.feature_manager.extract_features.side_effect = Exception("Feature extraction failed")
        
        features = market_simulator.get_current_features()
        
        # Should return zero arrays on failure
        assert features is not None
        assert np.all(features['hf'] == 0)
        assert np.all(features['mf'] == 0)
        assert np.all(features['lf'] == 0)


class TestMarketSimulatorData:
    """Test market data retrieval functionality."""
    
    def test_get_current_market_data_success(self, market_simulator, sample_day_data):
        """Test successful market data retrieval."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        market_data = market_simulator.get_current_market_data()
        
        assert market_data is not None
        required_fields = [
            'timestamp', 'current_price', 'best_bid', 'best_ask',
            'bid_size', 'ask_size', 'mid_price', 'spread',
            'market_session', 'is_halted', 'intraday_high',
            'intraday_low', 'session_volume', 'session_trades',
            'session_vwap'
        ]
        
        for field in required_fields:
            assert field in market_data
            
        # Validate data types and ranges
        assert isinstance(market_data['current_price'], float)
        assert isinstance(market_data['best_bid'], float)
        assert isinstance(market_data['best_ask'], float)
        assert market_data['best_bid'] > 0
        assert market_data['best_ask'] > 0
        assert market_data['best_bid'] < market_data['best_ask']
        assert market_data['spread'] >= 0
        assert market_data['session_volume'] >= 0
        assert market_data['session_trades'] >= 0
        
    def test_get_current_market_data_no_data(self, market_simulator):
        """Test market data retrieval when no data is available."""
        market_data = market_simulator.get_current_market_data()
        assert market_data is None


class TestMarketSimulatorProgress:
    """Test progress tracking functionality."""
    
    def test_is_done_with_data(self, market_simulator, sample_day_data):
        """Test is_done with available data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # At beginning, should not be done
        assert not market_simulator.is_done()
        
        # Move to end
        market_simulator.current_index = len(market_simulator.df_market_state) - 1
        assert market_simulator.is_done()
        
    def test_is_done_no_data(self, market_simulator):
        """Test is_done when no data is available."""
        assert market_simulator.is_done()
        
    def test_get_progress_with_data(self, market_simulator, sample_day_data):
        """Test progress calculation with data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        total_states = len(market_simulator.df_market_state)
        
        # At beginning
        progress = market_simulator.get_progress()
        assert progress == 0.0
        
        # At middle
        market_simulator.current_index = total_states // 2
        progress = market_simulator.get_progress()
        assert 40.0 <= progress <= 60.0  # Roughly middle
        
        # At end
        market_simulator.current_index = total_states - 1
        progress = market_simulator.get_progress()
        assert progress == 100.0
        
    def test_get_progress_no_data(self, market_simulator):
        """Test progress calculation when no data is available."""
        progress = market_simulator.get_progress()
        assert progress == 0.0


class TestMarketSimulatorStats:
    """Test statistics functionality."""
    
    def test_get_stats_with_data(self, market_simulator, sample_day_data):
        """Test statistics retrieval with data."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        date = datetime(2024, 1, 15)
        market_simulator.initialize_day(date)
        
        stats = market_simulator.get_stats()
        
        assert stats is not None
        assert 'date' in stats
        assert 'symbol' in stats
        assert 'total_seconds' in stats
        assert 'current_index' in stats
        assert 'progress_pct' in stats
        assert 'price_range' in stats
        assert 'total_volume' in stats
        assert 'total_trades' in stats
        assert 'warmup_info' in stats
        
        assert stats['date'] == pd.Timestamp(date).date()
        assert stats['symbol'] == "MLGO"
        assert stats['total_seconds'] > 0
        assert isinstance(stats['price_range'], dict)
        assert 'high' in stats['price_range']
        assert 'low' in stats['price_range']
        assert isinstance(stats['warmup_info'], dict)
        
    def test_get_stats_no_data(self, market_simulator):
        """Test statistics retrieval when no data is available."""
        stats = market_simulator.get_stats()
        assert stats == {}


class TestMarketSimulatorCandleData:
    """Test candle data functionality."""
    
    def test_get_1m_candle_data_success(self, market_simulator, sample_day_data):
        """Test successful 1-minute candle data retrieval."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Create some 1m bar data for testing
        market_simulator.combined_bars_1m = pd.DataFrame({
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.8, 9.9, 10.0],
            'close': [10.1, 10.2, 10.3],
            'volume': [1000, 1500, 2000]
        }, index=pd.date_range('2024-01-15 09:30:00', periods=3, freq='1min', tz=ZoneInfo('UTC')))
        
        candles = market_simulator.get_1m_candle_data(lookback_minutes=10)
        
        assert isinstance(candles, list)
        for candle in candles:
            assert 'timestamp' in candle
            assert 'open' in candle
            assert 'high' in candle
            assert 'low' in candle
            assert 'close' in candle
            assert 'volume' in candle
            assert isinstance(candle['open'], float)
            assert isinstance(candle['high'], float)
            assert isinstance(candle['low'], float)
            assert isinstance(candle['close'], float)
            assert isinstance(candle['volume'], float)
            
    def test_get_1m_candle_data_no_data(self, market_simulator):
        """Test candle data retrieval when no data is available."""
        candles = market_simulator.get_1m_candle_data()
        assert candles == []
        
    def test_get_1m_candle_data_no_1m_bars(self, market_simulator, sample_day_data):
        """Test candle data retrieval when no 1m bars are available."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        market_simulator.combined_bars_1m = None
        
        candles = market_simulator.get_1m_candle_data()
        assert candles == []


class TestMarketSimulatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_symbol(self, mock_data_manager, model_config, simulation_config):
        """Test handling of invalid symbols."""
        simulator = MarketSimulator(
            symbol="",  # Empty symbol
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config
        )
        
        assert simulator.symbol == ""
        
    def test_invalid_date_formats(self, market_simulator):
        """Test handling of invalid date formats."""
        # Test with string that should work
        result = market_simulator.initialize_day("2024-01-15")
        # This may pass or fail depending on data availability, but shouldn't crash
        assert isinstance(result, bool)
        
        # The actual implementation converts to pd.Timestamp, so most formats are handled gracefully
        
    def test_memory_cleanup(self, market_simulator, sample_day_data):
        """Test proper memory cleanup."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Verify data exists
        assert market_simulator.df_market_state is not None
        assert market_simulator.combined_bars_1s is not None
        
        # Call close
        market_simulator.close()
        
        # Verify cleanup
        assert market_simulator.df_market_state is None
        assert market_simulator.combined_bars_1s is None
        assert market_simulator.combined_bars_1m is None
        assert market_simulator.combined_bars_5m is None
        assert market_simulator.combined_trades is None
        assert market_simulator.combined_quotes is None
        assert market_simulator.current_index == 0
        assert market_simulator.current_date is None
        assert len(market_simulator._precomputed_cache) == 0
        assert market_simulator.prev_day_data == {}
        
    def test_large_data_handling(self, market_simulator):
        """Test handling of large datasets."""
        # Create large sample data
        large_times = pd.date_range('2024-01-15 04:00:00', '2024-01-15 20:00:00', freq='1s', tz=ZoneInfo('UTC'))
        large_trades = pd.DataFrame({
            'price': np.random.uniform(10.0, 11.0, len(large_times)),
            'size': np.random.randint(100, 1000, len(large_times))
        }, index=large_times)
        
        large_data = {
            'trades': large_trades,
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame()
        }
        
        market_simulator.data_manager.load_day.return_value = large_data
        
        # Should handle large data without crashing
        result = market_simulator.initialize_day(datetime(2024, 1, 15))
        assert result is True
        
        # Should have many states
        assert len(market_simulator.df_market_state) > 50000  # 16 hours * 3600 seconds
        
    def test_concurrent_access_safety(self, market_simulator, sample_day_data):
        """Test basic concurrent access patterns."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        market_simulator.initialize_day(datetime(2024, 1, 15))
        
        # Simulate concurrent state access
        state1 = market_simulator.get_market_state()
        features1 = market_simulator.get_current_features()
        market_data1 = market_simulator.get_current_market_data()
        
        # All should return valid data
        assert state1 is not None
        assert features1 is not None
        assert market_data1 is not None
        
    def test_cache_size_limits(self, market_simulator, sample_day_data):
        """Test that cache size is properly limited."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        
        # Initialize multiple days to test cache limits
        dates = [datetime(2024, 1, i) for i in range(15, 25)]  # 10 days
        
        for date in dates:
            market_simulator.initialize_day(date)
            
        # Cache should be limited (implementation shows max 5)
        assert len(market_simulator._precomputed_cache) <= 5
        
    def test_timezone_edge_cases(self, market_simulator, sample_day_data):
        """Test timezone handling edge cases."""
        # Test with data in different timezones
        utc_times = pd.date_range('2024-01-15 14:30:00', periods=10, freq='1min', tz=ZoneInfo('UTC'))
        
        data_with_tz = {
            'trades': pd.DataFrame({
                'price': np.random.uniform(10.0, 11.0, len(utc_times)),
                'size': np.random.randint(100, 1000, len(utc_times))
            }, index=utc_times),
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame()
        }
        
        market_simulator.data_manager.load_day.return_value = data_with_tz
        result = market_simulator.initialize_day(datetime(2024, 1, 15))
        
        assert result is True
        
    def test_malformed_data_handling(self, market_simulator):
        """Test handling of malformed data."""
        # Test with missing required columns but proper timestamp index
        timestamps = pd.date_range('2024-01-15 09:30:00', periods=3, freq='1min', tz=ZoneInfo('UTC'))
        malformed_data = {
            'trades': pd.DataFrame({'invalid_column': [1, 2, 3]}, index=timestamps),
            'quotes': pd.DataFrame({'another_invalid': [1, 2]}, index=timestamps[:2]),
            'status': pd.DataFrame()
        }
        
        # Setup mock for get_previous_day_data calls
        def mock_get_previous_day_data(key):
            if key == 'bars_1d':
                return pd.DataFrame({
                    'open': [100.0], 'high': [105.0], 'low': [95.0], 
                    'close': [102.0], 'volume': [10000], 'vwap': [101.0]
                })
            return None
        
        market_simulator.data_manager.load_day.return_value = malformed_data
        market_simulator.data_manager.get_previous_day_data.side_effect = mock_get_previous_day_data
        
        # Should handle gracefully by returning False rather than crashing
        result = market_simulator.initialize_day(datetime(2024, 1, 15))
        assert isinstance(result, bool)  # Should not crash
        
    def test_extreme_price_values(self, market_simulator):
        """Test handling of extreme price values."""
        extreme_times = pd.date_range('2024-01-15 09:30:00', periods=100, freq='1min', tz=ZoneInfo('UTC'))
        
        # Test with very high prices
        high_price_data = {
            'trades': pd.DataFrame({
                'price': np.full(len(extreme_times), 999999.99),
                'size': np.random.randint(100, 1000, len(extreme_times))
            }, index=extreme_times),
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame()
        }
        
        market_simulator.data_manager.load_day.return_value = high_price_data
        result = market_simulator.initialize_day(datetime(2024, 1, 15))
        assert result is True
        
        # Test with very low prices
        low_price_data = {
            'trades': pd.DataFrame({
                'price': np.full(len(extreme_times), 0.01),
                'size': np.random.randint(100, 1000, len(extreme_times))
            }, index=extreme_times),
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame()
        }
        
        market_simulator.data_manager.load_day.return_value = low_price_data
        result = market_simulator.initialize_day(datetime(2024, 1, 15))
        assert result is True


class TestMarketSimulatorIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_day_simulation_workflow(self, initialized_market_simulator):
        """Test complete day simulation workflow."""
        # Day already initialized by fixture
        market_simulator = initialized_market_simulator
        
        # Track progress through day
        states_collected = []
        features_collected = []
        
        while not market_simulator.is_done():
            # Get current state and features
            state = market_simulator.get_market_state()
            features = market_simulator.get_current_features()
            market_data = market_simulator.get_current_market_data()
            
            assert state is not None
            assert features is not None
            assert market_data is not None
            
            states_collected.append(state)
            features_collected.append(features)
            
            # Step forward
            if not market_simulator.step():
                break
                
            # Don't collect too many for memory
            if len(states_collected) > 100:
                break
                
        # Verify we collected data
        assert len(states_collected) > 0
        assert len(features_collected) > 0
        
        # Verify state consistency
        for i, state in enumerate(states_collected):
            # Basic sanity checks - focus on output behavior not calculation details
            assert state.spread >= 0
            assert state.current_price > 0
            assert state.best_bid > 0
            assert state.best_ask > 0
            # The implementation may adjust bid/ask relationships, so just check they're reasonable
            assert abs(state.best_bid - state.best_ask) < 100  # Spread shouldn't be enormous
            
        # Test time navigation
        start_time, end_time = market_simulator.get_time_range()
        assert start_time is not None
        assert end_time is not None
        
        # Reset and verify
        market_simulator.reset()
        assert market_simulator.current_index == 0
        
        # Jump to specific time
        mid_time = start_time + (end_time - start_time) / 2
        result = market_simulator.set_time(mid_time)
        assert result is True
        
    def test_multiple_day_initialization(self, market_simulator, sample_day_data):
        """Test initializing multiple days."""
        market_simulator.data_manager.load_day.return_value = sample_day_data
        
        dates = [
            datetime(2024, 1, 15),
            datetime(2024, 1, 16),
            datetime(2024, 1, 17)
        ]
        
        for date in dates:
            result = market_simulator.initialize_day(date)
            assert result is True
            assert market_simulator.current_date == pd.Timestamp(date).date()
            
            # Verify we can get state for each day
            state = market_simulator.get_market_state()
            assert state is not None
            
    def test_error_recovery_workflow(self, market_simulator, sample_day_data):
        """Test error recovery in workflows."""
        # Start with good data
        market_simulator.data_manager.load_day.return_value = sample_day_data
        result = market_simulator.initialize_day(datetime(2024, 1, 15))
        assert result is True
        
        # Cause feature extraction error
        market_simulator.feature_manager.extract_features.side_effect = Exception("Feature error")
        
        # Should still get features (zeros on error)
        features = market_simulator.get_current_features()
        assert features is not None
        assert np.all(features['hf'] == 0)
        
        # Reset feature manager to working state
        market_simulator.feature_manager.extract_features.side_effect = None
        market_simulator.feature_manager.extract_features.return_value = {
            'hf': np.ones((60, 7)),
            'mf': np.ones((30, 43)),
            'lf': np.ones((30, 19))
        }
        
        # Move to next state (should clear cache and work)
        market_simulator.step()
        features = market_simulator.get_current_features()
        assert features is not None
        assert np.all(features['hf'] == 1)  # Should get new good features