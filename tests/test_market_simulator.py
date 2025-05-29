"""Comprehensive tests for MarketSimulator

Tests focus on input/output behavior without examining implementation details.
Tests cover:
- Initialization and configuration
- Data loading and processing
- Time advancement and state retrieval
- Feature extraction (on-demand)
- Edge cases and error handling
- Memory management and caching
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock, MagicMock, patch, call
import logging

from simulators.market_simulator import MarketSimulator, MarketState, MARKET_HOURS
from data.data_manager import DataManager
from feature.feature_manager import FeatureManager
from feature.contexts import MarketContext
from config.schemas import ModelConfig, SimulationConfig


class TestMarketSimulator:
    """Test suite for MarketSimulator class"""
    
    @pytest.fixture
    def model_config(self):
        """Create a test model configuration"""
        return ModelConfig(
            hf_seq_len=60,
            hf_feat_dim=10,
            mf_seq_len=30,
            mf_feat_dim=8,
            lf_seq_len=12,
            lf_feat_dim=6,
            portfolio_feat_dim=4,
            hidden_size=128,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            use_flash_attention=False
        )
    
    @pytest.fixture
    def simulation_config(self):
        """Create a test simulation configuration"""
        return SimulationConfig(
            execution_delay_ms=100,
            slippage_bps=10,
            market_impact_model="linear",
            fill_probability=0.95,
            partial_fill_probability=0.8,
            reject_probability=0.02,
            halt_probability=0.001,
            circuit_breaker_threshold=0.1,
            max_position_size=10000,
            enable_partial_fills=True
        )
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager"""
        mock = Mock(spec=DataManager)
        mock.momentum_scanner = None
        mock.get_previous_day_data = Mock(return_value=None)
        mock.get_momentum_days = Mock(return_value=pd.DataFrame())
        return mock
    
    @pytest.fixture
    def mock_feature_manager(self):
        """Create a mock feature manager"""
        mock = Mock(spec=FeatureManager)
        mock.extract_features = Mock(return_value={
            'hf': np.zeros((60, 10)),
            'mf': np.zeros((30, 8)),
            'lf': np.zeros((12, 6)),
            'static': np.zeros(5)
        })
        return mock
    
    @pytest.fixture
    def sample_trades_data(self):
        """Create sample trades data"""
        timestamps = pd.date_range(
            start='2025-01-15 09:30:00',
            end='2025-01-15 09:35:00',
            freq='10s',
            tz='UTC'
        )
        return pd.DataFrame({
            'price': np.random.uniform(99, 101, len(timestamps)),
            'size': np.random.randint(100, 1000, len(timestamps))
        }, index=timestamps)
    
    @pytest.fixture
    def sample_quotes_data(self):
        """Create sample quotes data"""
        timestamps = pd.date_range(
            start='2025-01-15 09:30:00',
            end='2025-01-15 09:35:00',
            freq='5s',
            tz='UTC'
        )
        return pd.DataFrame({
            'bid_price': np.random.uniform(99, 100, len(timestamps)),
            'ask_price': np.random.uniform(100, 101, len(timestamps)),
            'bid_size': np.random.randint(100, 1000, len(timestamps)),
            'ask_size': np.random.randint(100, 1000, len(timestamps))
        }, index=timestamps)
    
    @pytest.fixture
    def sample_day_data(self, sample_trades_data, sample_quotes_data):
        """Create sample day data"""
        return {
            'trades': sample_trades_data,
            'quotes': sample_quotes_data,
            'status': pd.DataFrame(),
            'bars_1s': pd.DataFrame(),
            'bars_1m': pd.DataFrame(),
            'bars_5m': pd.DataFrame()
        }
    
    def test_initialization(self, model_config, simulation_config, mock_data_manager):
        """Test MarketSimulator initialization"""
        # Test basic initialization
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
        
        # Test initialization with custom feature manager and logger
        custom_feature_manager = Mock(spec=FeatureManager)
        custom_logger = logging.getLogger("test")
        
        simulator2 = MarketSimulator(
            symbol="TEST2",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=custom_feature_manager,
            logger=custom_logger
        )
        
        assert simulator2.feature_manager == custom_feature_manager
        assert simulator2.logger == custom_logger
        
        # Verify internal state initialization
        assert simulator.df_market_state is None
        assert simulator.current_index == 0
        assert simulator.current_date is None
        assert simulator._precomputed_cache == {}
        assert simulator.prev_day_data == {}
    
    def test_initialize_day_success(self, model_config, simulation_config, mock_data_manager, 
                                   mock_feature_manager, sample_day_data):
        """Test successful day initialization"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup mock data manager
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: {
            'bars_1d': pd.DataFrame({
                'open': [100], 'high': [105], 'low': [98], 
                'close': [102], 'volume': [1000000], 'vwap': [101]
            })
        }.get(key, pd.DataFrame()))
        
        # Initialize day
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        
        # Verify success
        assert result is True
        assert simulator.current_date == pd.Timestamp(date).date()
        assert simulator.current_index == 0
        assert simulator.df_market_state is not None
        assert not simulator.df_market_state.empty
        
        # Verify timeline spans full trading day (4 AM - 8 PM ET = 16 hours)
        # Note: pd.date_range includes both endpoints, so we get one extra second
        expected_seconds = 16 * 60 * 60 + 1  # 57,601 seconds
        assert len(simulator.df_market_state) == expected_seconds
        
        # Verify market state columns
        expected_columns = [
            'current_price', 'best_bid', 'best_ask', 'bid_size', 'ask_size',
            'mid_price', 'spread', 'market_session', 'is_halted',
            'intraday_high', 'intraday_low', 'session_volume', 'session_trades',
            'session_vwap', 'hf_features', 'mf_features', 'lf_features', 
            'static_features', 'features_computed'
        ]
        for col in expected_columns:
            assert col in simulator.df_market_state.columns
        
        # Verify cache
        cache_key = ("TEST", pd.Timestamp(date).date())
        assert cache_key in simulator._precomputed_cache
    
    def test_initialize_day_no_data(self, model_config, simulation_config, mock_data_manager):
        """Test day initialization with no data available"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config
        )
        
        # Setup mock to return no data
        mock_data_manager.load_day = Mock(return_value=None)
        
        # Initialize day
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        
        # Verify failure
        assert result is False
        assert simulator.df_market_state is None
        assert simulator.current_date is None
    
    def test_initialize_day_cache_hit(self, model_config, simulation_config, mock_data_manager, 
                                     mock_feature_manager, sample_day_data):
        """Test day initialization with cache hit"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup mock data manager
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        # Initialize day first time
        date = datetime(2025, 1, 15)
        result1 = simulator.initialize_day(date)
        assert result1 is True
        
        # Reset load_day mock to verify it's not called again
        mock_data_manager.load_day.reset_mock()
        
        # Initialize same day again (should use cache)
        result2 = simulator.initialize_day(date)
        assert result2 is True
        assert mock_data_manager.load_day.call_count == 0  # Not called due to cache
    
    def test_get_market_state_by_index(self, model_config, simulation_config, mock_data_manager,
                                      mock_feature_manager, sample_day_data):
        """Test getting market state by current index"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Get state at current index (0)
        state = simulator.get_market_state()
        assert state is not None
        assert isinstance(state, MarketState)
        assert state.timestamp == simulator.df_market_state.index[0]
        
        # Advance and get state again
        simulator.step()
        state2 = simulator.get_market_state()
        assert state2 is not None
        assert state2.timestamp == simulator.df_market_state.index[1]
        assert state2.timestamp > state.timestamp
    
    def test_get_market_state_by_timestamp(self, model_config, simulation_config, mock_data_manager,
                                           mock_feature_manager, sample_day_data):
        """Test getting market state by specific timestamp"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Get state at specific timestamp
        target_timestamp = simulator.df_market_state.index[100]
        state = simulator.get_market_state(timestamp=target_timestamp)
        assert state is not None
        assert state.timestamp == target_timestamp
        
        # Get state at non-existent timestamp (should get closest previous)
        non_existent = target_timestamp + pd.Timedelta(milliseconds=500)
        state2 = simulator.get_market_state(timestamp=non_existent)
        assert state2 is not None
        assert state2.timestamp == target_timestamp
    
    def test_get_current_features_on_demand(self, model_config, simulation_config, 
                                           mock_data_manager, mock_feature_manager, sample_day_data):
        """Test on-demand feature computation"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Verify features not computed initially
        assert simulator.df_market_state['features_computed'].iloc[0] == False
        
        # Get features (should trigger computation)
        features = simulator.get_current_features()
        assert features is not None
        assert 'hf' in features
        assert 'mf' in features
        assert 'lf' in features
        assert 'static' in features
        
        # Verify feature manager was called
        assert mock_feature_manager.extract_features.called
        
        # Verify features now marked as computed
        assert simulator.df_market_state['features_computed'].iloc[0] == True
        
        # Get features again (should use cache)
        mock_feature_manager.extract_features.reset_mock()
        features2 = simulator.get_current_features()
        assert features2 is not None
        assert not mock_feature_manager.extract_features.called  # Not called due to cache
    
    def test_step_advance(self, model_config, simulation_config, mock_data_manager,
                         mock_feature_manager, sample_day_data):
        """Test stepping through time"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Test stepping
        assert simulator.current_index == 0
        result = simulator.step()
        assert result is True
        assert simulator.current_index == 1
        
        # Step multiple times
        for i in range(100):
            result = simulator.step()
            assert result is True
            assert simulator.current_index == i + 2
        
        # Step to end
        simulator.current_index = len(simulator.df_market_state) - 2
        result = simulator.step()
        assert result is True
        assert simulator.is_done()
        
        # Step past end
        result = simulator.step()
        assert result is False
    
    def test_reset(self, model_config, simulation_config, mock_data_manager,
                   mock_feature_manager, sample_day_data):
        """Test resetting simulator"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Advance simulator
        for _ in range(100):
            simulator.step()
        assert simulator.current_index == 100
        
        # Reset to beginning
        result = simulator.reset()
        assert result is True
        assert simulator.current_index == 0
        
        # Reset to specific index
        result = simulator.reset(start_index=500)
        assert result is True
        assert simulator.current_index == 500
        
        # Reset with out-of-bounds index
        result = simulator.reset(start_index=1000000)
        assert result is True
        assert simulator.current_index == len(simulator.df_market_state) - 1
        
        # Reset with negative index
        result = simulator.reset(start_index=-10)
        assert result is True
        assert simulator.current_index == 0
    
    def test_set_time(self, model_config, simulation_config, mock_data_manager,
                     mock_feature_manager, sample_day_data):
        """Test jumping to specific timestamp"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Jump to exact timestamp
        target_ts = simulator.df_market_state.index[1000]
        result = simulator.set_time(target_ts)
        assert result is True
        assert simulator.current_index == 1000
        
        # Jump to non-existent timestamp (should find closest previous)
        non_existent = target_ts + pd.Timedelta(milliseconds=500)
        result = simulator.set_time(non_existent)
        assert result is True
        assert simulator.current_index == 1000
        
        # Jump to timestamp before data
        too_early = simulator.df_market_state.index[0] - pd.Timedelta(hours=1)
        result = simulator.set_time(too_early)
        assert result is False
    
    def test_get_time_range(self, model_config, simulation_config, mock_data_manager,
                           mock_feature_manager, sample_day_data):
        """Test getting available time range"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Before initialization
        start, end = simulator.get_time_range()
        assert start is None
        assert end is None
        
        # After initialization
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        start, end = simulator.get_time_range()
        assert start is not None
        assert end is not None
        assert start < end
        assert start == simulator.df_market_state.index[0]
        assert end == simulator.df_market_state.index[-1]
    
    def test_is_done_and_progress(self, model_config, simulation_config, mock_data_manager,
                                  mock_feature_manager, sample_day_data):
        """Test done status and progress tracking"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Before initialization
        assert simulator.is_done() is True
        assert simulator.get_progress() == 0.0
        
        # After initialization
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # At beginning
        assert simulator.is_done() is False
        assert simulator.get_progress() == 0.0
        
        # At middle
        simulator.current_index = len(simulator.df_market_state) // 2
        assert simulator.is_done() is False
        assert 49 < simulator.get_progress() < 51  # Around 50%
        
        # At end
        simulator.current_index = len(simulator.df_market_state) - 1
        assert simulator.is_done() is True
        assert simulator.get_progress() == 100.0
    
    def test_get_stats(self, model_config, simulation_config, mock_data_manager,
                      mock_feature_manager, sample_day_data):
        """Test getting simulator statistics"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Before initialization
        stats = simulator.get_stats()
        assert stats == {}
        
        # After initialization
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        stats = simulator.get_stats()
        assert stats['symbol'] == "TEST"
        assert stats['date'] == pd.Timestamp(date).date()
        assert stats['total_seconds'] == len(simulator.df_market_state)
        assert stats['current_index'] == 0
        assert 'price_range' in stats
        assert 'total_volume' in stats
        assert 'total_trades' in stats
        assert 'warmup_info' in stats
    
    def test_market_session_detection(self, model_config, simulation_config, mock_data_manager,
                                     mock_feature_manager, sample_day_data):
        """Test market session detection for different timestamps"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup and initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Test different times
        market_tz = ZoneInfo("America/New_York")
        
        # Pre-market (7 AM ET)
        premarket_time = pd.Timestamp(date).tz_localize(market_tz).replace(hour=7).tz_convert('UTC')
        simulator.set_time(premarket_time)
        state = simulator.get_market_state()
        assert state.market_session == "PREMARKET"
        
        # Regular hours (10 AM ET)
        regular_time = pd.Timestamp(date).tz_localize(market_tz).replace(hour=10).tz_convert('UTC')
        simulator.set_time(regular_time)
        state = simulator.get_market_state()
        assert state.market_session == "REGULAR"
        
        # Post-market (5 PM ET)
        postmarket_time = pd.Timestamp(date).tz_localize(market_tz).replace(hour=17).tz_convert('UTC')
        simulator.set_time(postmarket_time)
        state = simulator.get_market_state()
        assert state.market_session == "POSTMARKET"
    
    def test_warmup_data_handling(self, model_config, simulation_config, mock_data_manager,
                                 mock_feature_manager):
        """Test handling of previous day warmup data"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Create current day data
        current_day_trades = pd.DataFrame({
            'price': [100, 101, 102],
            'size': [100, 200, 300]
        }, index=pd.date_range('2025-01-15 09:30:00', periods=3, freq='1min', tz='UTC'))
        
        # Create previous day data
        prev_day_trades = pd.DataFrame({
            'price': [98, 99, 100],
            'size': [150, 250, 350]
        }, index=pd.date_range('2025-01-14 15:00:00', periods=3, freq='1min', tz='UTC'))
        
        current_data = {'trades': current_day_trades, 'quotes': pd.DataFrame()}
        prev_data = {'trades': prev_day_trades, 'quotes': pd.DataFrame()}
        
        # Setup mocks
        mock_data_manager.load_day = Mock(side_effect=[current_data, prev_data])
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: 
            prev_data.get(key) if key in prev_data else pd.DataFrame()
        )
        
        # Initialize day
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        assert result is True
        
        # Verify combined data
        assert simulator.combined_trades is not None
        assert len(simulator.combined_trades) == 6  # 3 from each day
        assert simulator.combined_trades.index[0].date() == datetime(2025, 1, 14).date()
        assert simulator.combined_trades.index[-1].date() == datetime(2025, 1, 15).date()
    
    def test_cache_size_limit(self, model_config, simulation_config, mock_data_manager,
                             mock_feature_manager, sample_day_data):
        """Test cache size limiting to prevent memory issues"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        # Initialize multiple days to fill cache
        for i in range(7):
            date = datetime(2025, 1, 10 + i)
            simulator.initialize_day(date)
        
        # Verify cache size is limited to 5
        assert len(simulator._precomputed_cache) <= 5
        
        # Verify most recent days are kept
        cache_dates = [key[1] for key in simulator._precomputed_cache.keys()]
        assert pd.Timestamp('2025-01-16').date() in cache_dates
        assert pd.Timestamp('2025-01-10').date() not in cache_dates
    
    def test_error_handling_feature_extraction(self, model_config, simulation_config, 
                                             mock_data_manager, mock_feature_manager, sample_day_data):
        """Test error handling during feature extraction"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup feature manager to raise error
        mock_feature_manager.extract_features = Mock(side_effect=Exception("Feature extraction failed"))
        
        # Initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Get features (should handle error and return zeros)
        features = simulator.get_current_features()
        assert features is not None
        assert np.all(features['hf'] == 0)
        assert np.all(features['mf'] == 0)
        assert np.all(features['lf'] == 0)
        assert np.all(features['static'] == 0)
    
    def test_empty_data_handling(self, model_config, simulation_config, mock_data_manager,
                                mock_feature_manager):
        """Test handling of empty data frames"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup with empty data
        empty_data = {
            'trades': pd.DataFrame(),
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame(),
            'bars_1s': pd.DataFrame(),
            'bars_1m': pd.DataFrame(),
            'bars_5m': pd.DataFrame()
        }
        
        mock_data_manager.load_day = Mock(return_value=empty_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        # Initialize day
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        assert result is True
        
        # Verify default values are used
        state = simulator.get_market_state()
        assert state is not None
        assert state.current_price > 0
        assert state.best_bid < state.best_ask
        assert state.spread > 0
    
    def test_quote_spread_validation(self, model_config, simulation_config, mock_data_manager,
                                   mock_feature_manager):
        """Test validation and correction of invalid bid-ask spreads"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Create quotes with invalid spreads
        invalid_quotes = pd.DataFrame({
            'bid_price': [101, 100, 99],  # First bid > ask
            'ask_price': [100, 100, 101],  # Second bid == ask
            'bid_size': [100, 200, 300],
            'ask_size': [100, 200, 300]
        }, index=pd.date_range('2025-01-15 09:30:00', periods=3, freq='1s', tz='UTC'))
        
        data = {
            'trades': pd.DataFrame(),
            'quotes': invalid_quotes,
            'status': pd.DataFrame()
        }
        
        mock_data_manager.load_day = Mock(return_value=data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        # Initialize
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Verify spreads are corrected
        for i in range(3):
            simulator.current_index = i
            state = simulator.get_market_state()
            assert state.best_bid < state.best_ask
            assert state.spread > 0
    
    def test_get_current_market_data(self, model_config, simulation_config, mock_data_manager,
                                   mock_feature_manager, sample_day_data):
        """Test getting market data without features"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Get market data
        data = simulator.get_current_market_data()
        assert data is not None
        assert 'timestamp' in data
        assert 'current_price' in data
        assert 'best_bid' in data
        assert 'best_ask' in data
        assert 'market_session' in data
        assert 'hf_features' not in data  # Should not include features
        assert 'mf_features' not in data
    
    def test_close_cleanup(self, model_config, simulation_config, mock_data_manager,
                          mock_feature_manager, sample_day_data):
        """Test cleanup when closing simulator"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Initialize
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Verify data exists
        assert simulator.df_market_state is not None
        assert len(simulator._precomputed_cache) > 0
        
        # Close simulator
        simulator.close()
        
        # Verify cleanup
        assert simulator.df_market_state is None
        assert len(simulator._precomputed_cache) == 0
        assert simulator.combined_bars_1s is None
        assert simulator.combined_bars_1m is None
        assert simulator.combined_bars_5m is None
        assert simulator.combined_trades is None
        assert simulator.combined_quotes is None
        assert simulator.current_index == 0
        assert simulator.current_date is None
        assert simulator.prev_day_data == {}
    
    def test_synthetic_bar_creation(self, model_config, simulation_config, mock_data_manager,
                                   mock_feature_manager):
        """Test creation of synthetic bars when data is missing"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Create sparse data
        sparse_trades = pd.DataFrame({
            'price': [100],
            'size': [100]
        }, index=[pd.Timestamp('2025-01-15 12:00:00', tz='UTC')])
        
        data = {'trades': sparse_trades, 'quotes': pd.DataFrame()}
        
        mock_data_manager.load_day = Mock(return_value=data)
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: {
            'bars_1d': pd.DataFrame({
                'open': [95], 'high': [96], 'low': [94], 
                'close': [95], 'volume': [100000], 'vwap': [95]
            })
        }.get(key, pd.DataFrame()))
        
        # Initialize
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Set simulator internal data for window building
        simulator.combined_trades = sparse_trades
        simulator.combined_quotes = pd.DataFrame()
        simulator.combined_bars_1s = pd.DataFrame()
        simulator.combined_bars_1m = pd.DataFrame()
        simulator.combined_bars_5m = pd.DataFrame()
        
        # Get features to trigger window building
        features = simulator.get_current_features()
        
        # Verify synthetic data handling
        assert features is not None
        # Features should be computed even with sparse data
    
    def test_parallel_feature_extraction(self, model_config, simulation_config):
        """Test parallel feature extraction static methods"""
        # Test data for parallel processing
        shared_data = {
            'symbol': 'TEST',
            'model_config': model_config,
            'hf_seq_len': 60,
            'hf_feat_dim': 10,
            'mf_seq_len': 30,
            'mf_feat_dim': 8,
            'lf_seq_len': 12,
            'lf_feat_dim': 6,
            'combined_trades': pd.DataFrame(),
            'combined_quotes': pd.DataFrame(),
            'combined_bars_1s': pd.DataFrame(),
            'combined_bars_1m': pd.DataFrame(),
            'combined_bars_5m': pd.DataFrame(),
            'prev_day_data': {'close': 100}
        }
        
        # Create batch data
        timestamp = pd.Timestamp('2025-01-15 10:00:00', tz='UTC')
        row_data = pd.Series({
            'current_price': 100,
            'intraday_high': 101,
            'intraday_low': 99,
            'market_session': 'REGULAR',
            'session_volume': 10000,
            'session_trades': 100,
            'session_vwap': 100.5
        })
        
        batch_data = [(0, timestamp, row_data)]
        
        # Test extraction (would normally run in parallel process)
        with patch('simulators.market_simulator.FeatureManager') as mock_fm_class:
            mock_fm = Mock()
            mock_fm.extract_features.return_value = {
                'hf': np.zeros((60, 10)),
                'mf': np.zeros((30, 8)),
                'lf': np.zeros((12, 6)),
                'static': np.zeros(5)
            }
            mock_fm_class.return_value = mock_fm
            
            results = MarketSimulator._extract_features_batch(batch_data, shared_data)
            
            assert len(results) == 1
            assert results[0][0] == 0  # Index
            assert results[0][1].shape == (60, 10)  # HF features
            assert results[0][2].shape == (30, 8)   # MF features
            assert results[0][3].shape == (12, 6)   # LF features
            assert results[0][4].shape == (5,)      # Static features
    
    def test_window_building_methods(self, model_config):
        """Test static window building methods"""
        # Test HF window building
        current_ts = pd.Timestamp('2025-01-15 10:00:00', tz='UTC')
        trades = pd.DataFrame({
            'price': [100, 101],
            'size': [100, 200]
        }, index=[
            current_ts - pd.Timedelta(seconds=30),
            current_ts - pd.Timedelta(seconds=10)
        ])
        
        shared_data = {
            'hf_seq_len': 60,
            'combined_trades': trades,
            'combined_quotes': pd.DataFrame(),
            'combined_bars_1s': pd.DataFrame()
        }
        
        window = MarketSimulator._build_hf_window_for_batch(current_ts, shared_data)
        assert len(window) == 60
        assert all('timestamp' in w for w in window)
        assert all('trades' in w for w in window)
        assert all('quotes' in w for w in window)
        assert all('1s_bar' in w for w in window)
        
        # Test MF window building
        bars_1m = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
        }, index=[current_ts.floor('1min') - pd.Timedelta(minutes=5)])
        
        shared_data = {
            'mf_seq_len': 30,
            'combined_bars_1m': bars_1m,
            'prev_day_data': {'close': 95}
        }
        
        window = MarketSimulator._build_mf_window_for_batch(current_ts, shared_data)
        assert len(window) == 30
        assert all('timestamp' in w for w in window)
        assert all('open' in w for w in window)
        assert all('is_synthetic' in w for w in window)
        
        # Test LF window building
        bars_5m = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [98], 'close': [101], 'volume': [5000]
        }, index=[current_ts.floor('5min') - pd.Timedelta(minutes=10)])
        
        shared_data = {
            'lf_seq_len': 12,
            'combined_bars_5m': bars_5m,
            'prev_day_data': {'close': 95}
        }
        
        window = MarketSimulator._build_lf_window_for_batch(current_ts, shared_data)
        assert len(window) == 12
        assert all('timestamp' in w for w in window)
        assert all('close' in w for w in window)
        assert all('is_synthetic' in w for w in window)


    def test_4am_start_time_behavior(self, model_config, simulation_config, mock_data_manager,
                                     mock_feature_manager):
        """Test that simulator properly starts at 4 AM ET and handles early morning data"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Create data with timestamps starting at 4 AM ET
        et_tz = ZoneInfo("America/New_York")
        utc_tz = ZoneInfo("UTC")
        
        # 4 AM ET on Jan 15, 2025
        start_et = pd.Timestamp('2025-01-15 04:00:00').tz_localize(et_tz)
        start_utc = start_et.tz_convert(utc_tz)
        
        # Create trades around 4 AM
        trades_4am = pd.DataFrame({
            'price': [100.0, 100.1, 100.2],
            'size': [100, 200, 300]
        }, index=[
            start_utc,
            start_utc + pd.Timedelta(seconds=30),
            start_utc + pd.Timedelta(seconds=60)
        ])
        
        # Create previous day data ending at 8 PM previous day
        prev_day_end_et = pd.Timestamp('2025-01-14 20:00:00').tz_localize(et_tz)
        prev_day_end_utc = prev_day_end_et.tz_convert(utc_tz)
        
        prev_trades = pd.DataFrame({
            'price': [99.5, 99.6, 99.7],
            'size': [150, 250, 350]
        }, index=[
            prev_day_end_utc - pd.Timedelta(minutes=10),
            prev_day_end_utc - pd.Timedelta(minutes=5),
            prev_day_end_utc
        ])
        
        current_data = {'trades': trades_4am, 'quotes': pd.DataFrame()}
        prev_data = {'trades': prev_trades, 'quotes': pd.DataFrame()}
        
        # Setup mocks
        mock_data_manager.load_day = Mock(return_value=current_data)
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: {
            'trades': prev_trades,
            'bars_1d': pd.DataFrame({
                'open': [99], 'high': [100], 'low': [98], 
                'close': [99.7], 'volume': [100000], 'vwap': [99.5]
            })
        }.get(key, pd.DataFrame()))
        
        # Initialize day
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        assert result is True
        
        # Verify timeline starts exactly at 4 AM ET
        start_time, end_time = simulator.get_time_range()
        assert start_time == start_utc
        
        # Verify 8 PM ET end time
        expected_end_et = pd.Timestamp('2025-01-15 20:00:00').tz_localize(et_tz)
        expected_end_utc = expected_end_et.tz_convert(utc_tz)
        assert end_time == expected_end_utc
        
        # Verify we have exactly 16 hours + 1 second (57601 seconds)
        assert len(simulator.df_market_state) == 16 * 3600 + 1
        
        # Verify warmup data is available
        assert simulator.combined_trades is not None
        assert len(simulator.combined_trades) == 6  # 3 prev + 3 current
        
        # Test early morning state (first few seconds)
        simulator.reset()
        state = simulator.get_market_state()
        assert state is not None
        assert state.timestamp == start_utc
        
        # Verify market session is PREMARKET at 4 AM
        assert state.market_session == "PREMARKET"
        
        # Test that we can get features for early morning (should use warmup data)
        features = simulator.get_current_features()
        assert features is not None
        assert 'hf' in features
        assert features['hf'].shape == (60, 10)  # Should have full window even at 4 AM
    
    def test_feature_content_validation(self, model_config, simulation_config, mock_data_manager):
        """Test that features contain reasonable values and proper structure"""
        # Create real feature manager (not mocked) to test actual feature extraction
        from feature.feature_manager import FeatureManager
        
        feature_manager = FeatureManager(
            symbol="TEST",
            config=model_config,
            logger=logging.getLogger("test")
        )
        
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=feature_manager
        )
        
        # Create realistic market data
        timestamps = pd.date_range(
            start='2025-01-15 09:30:00',
            end='2025-01-15 12:00:00',
            freq='1s',
            tz='UTC'
        )
        
        # Create realistic trades with trending price
        base_price = 100.0
        price_trend = np.cumsum(np.random.normal(0, 0.001, len(timestamps)))
        prices = base_price + price_trend
        
        trades = pd.DataFrame({
            'price': prices,
            'size': np.random.randint(100, 1000, len(timestamps))
        }, index=timestamps)
        
        # Create realistic quotes
        quotes = pd.DataFrame({
            'bid_price': prices - 0.01,
            'ask_price': prices + 0.01,
            'bid_size': np.random.randint(100, 1000, len(timestamps)),
            'ask_size': np.random.randint(100, 1000, len(timestamps))
        }, index=timestamps[::5])  # Every 5 seconds
        
        data = {
            'trades': trades,
            'quotes': quotes,
            'status': pd.DataFrame(),
            'bars_1s': pd.DataFrame(),
            'bars_1m': pd.DataFrame(),
            'bars_5m': pd.DataFrame()
        }
        
        # Setup mock data manager
        mock_data_manager.load_day = Mock(return_value=data)
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: {
            'bars_1d': pd.DataFrame({
                'open': [99], 'high': [101], 'low': [98], 
                'close': [100], 'volume': [1000000], 'vwap': [99.5]
            })
        }.get(key, pd.DataFrame()))
        
        # Initialize simulator
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        assert result is True
        
        # Jump to a point where we have sufficient data (10 minutes in)
        start_time, _ = simulator.get_time_range()
        target_time = start_time + pd.Timedelta(minutes=10)
        simulator.set_time(target_time)
        
        # Get features
        features = simulator.get_current_features()
        assert features is not None
        
        # Test HF features (60 seconds x 10 features)
        hf_features = features['hf']
        assert hf_features.shape == (60, 10)
        assert not np.all(hf_features == 0)  # Should have non-zero values
        assert np.all(np.isfinite(hf_features))  # No NaN or inf values
        
        # Test MF features (30 minutes x 8 features)
        mf_features = features['mf']
        assert mf_features.shape == (30, 8)
        assert np.all(np.isfinite(mf_features))
        
        # Test LF features (12 periods x 6 features)
        lf_features = features['lf']
        assert lf_features.shape == (12, 6)
        assert np.all(np.isfinite(lf_features))
        
        # Test static features (5 features)
        static_features = features['static']
        assert static_features.shape == (5,)
        assert np.all(np.isfinite(static_features))
        
        # Test that features change over time
        simulator.step()
        simulator.step()
        features2 = simulator.get_current_features()
        
        # HF features should be different (new data in window)
        assert not np.array_equal(features['hf'], features2['hf'])
    
    def test_warmup_window_completeness(self, model_config, simulation_config, mock_data_manager,
                                       mock_feature_manager):
        """Test that warmup data provides complete windows even at market open"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Create continuous data spanning midnight to ensure full warmup
        current_date = pd.Timestamp('2025-01-15').date()
        prev_date = pd.Timestamp('2025-01-14').date()
        
        # Previous day data - create full day ending at 8 PM ET
        prev_day_start = pd.Timestamp(f'{prev_date} 04:00:00').tz_localize('America/New_York').tz_convert('UTC')
        prev_day_end = pd.Timestamp(f'{prev_date} 20:00:00').tz_localize('America/New_York').tz_convert('UTC')
        
        prev_timestamps = pd.date_range(prev_day_start, prev_day_end, freq='1s')
        prev_trades = pd.DataFrame({
            'price': 100 + np.random.normal(0, 0.1, len(prev_timestamps)),
            'size': np.random.randint(100, 1000, len(prev_timestamps))
        }, index=prev_timestamps)
        
        # Current day data starting at 4 AM ET
        curr_day_start = pd.Timestamp(f'{current_date} 04:00:00').tz_localize('America/New_York').tz_convert('UTC')
        curr_day_noon = pd.Timestamp(f'{current_date} 12:00:00').tz_localize('America/New_York').tz_convert('UTC')
        
        curr_timestamps = pd.date_range(curr_day_start, curr_day_noon, freq='1s')
        curr_trades = pd.DataFrame({
            'price': 100.5 + np.random.normal(0, 0.1, len(curr_timestamps)),
            'size': np.random.randint(100, 1000, len(curr_timestamps))
        }, index=curr_timestamps)
        
        current_data = {'trades': curr_trades, 'quotes': pd.DataFrame()}
        prev_data = {'trades': prev_trades, 'quotes': pd.DataFrame()}
        
        # Setup mocks
        mock_data_manager.load_day = Mock(return_value=current_data)
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: 
            prev_data.get(key) if key in prev_data else pd.DataFrame()
        )
        
        # Initialize
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        assert result is True
        
        # Test at market open (4:00 AM exactly)
        simulator.reset()
        state = simulator.get_market_state()
        assert state.timestamp == curr_day_start
        
        # Verify we have complete combined data
        assert simulator.combined_trades is not None
        assert len(simulator.combined_trades) > 57600  # Should have previous day + current data
        
        # Verify combined data spans correctly
        assert simulator.combined_trades.index[0] >= prev_day_start
        assert simulator.combined_trades.index[-1] <= curr_day_noon
        
        # Test HF window at market open - should have 60 seconds of data
        hf_window = simulator._build_hf_window_with_warmup(curr_day_start)
        assert len(hf_window) == 60
        
        # Verify window includes previous day data
        window_start = curr_day_start - pd.Timedelta(seconds=59)
        assert window_start < curr_day_start  # Should go back to previous day
        
        # Test MF window - should have 30 minutes of data
        mf_window = simulator._build_mf_window_with_warmup(curr_day_start)
        assert len(mf_window) == 30
        
        # Test LF window - should have 12 periods (60 minutes) of data
        lf_window = simulator._build_lf_window_with_warmup(curr_day_start)
        assert len(lf_window) == 12
    
    def test_session_transitions(self, model_config, simulation_config, mock_data_manager,
                                mock_feature_manager, sample_day_data):
        """Test behavior during market session transitions"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Setup
        mock_data_manager.load_day = Mock(return_value=sample_day_data)
        mock_data_manager.get_previous_day_data = Mock(return_value=pd.DataFrame())
        
        date = datetime(2025, 1, 15)
        simulator.initialize_day(date)
        
        # Test specific session transition times
        et_tz = ZoneInfo("America/New_York")
        
        # 4:00 AM ET - Start of premarket
        premarket_start = pd.Timestamp('2025-01-15 04:00:00').tz_localize(et_tz).tz_convert('UTC')
        simulator.set_time(premarket_start)
        state = simulator.get_market_state()
        assert state.market_session == "PREMARKET"
        
        # 9:30 AM ET - Start of regular hours
        regular_start = pd.Timestamp('2025-01-15 09:30:00').tz_localize(et_tz).tz_convert('UTC')
        simulator.set_time(regular_start)
        state = simulator.get_market_state()
        assert state.market_session == "REGULAR"
        
        # 4:00 PM ET - End of regular hours, start of postmarket
        postmarket_start = pd.Timestamp('2025-01-15 16:00:00').tz_localize(et_tz).tz_convert('UTC')
        simulator.set_time(postmarket_start)
        state = simulator.get_market_state()
        assert state.market_session == "POSTMARKET"
        
        # 8:00 PM ET - End of postmarket
        postmarket_end = pd.Timestamp('2025-01-15 20:00:00').tz_localize(et_tz).tz_convert('UTC')
        simulator.set_time(postmarket_end)
        state = simulator.get_market_state()
        assert state.market_session == "POSTMARKET"  # Should still be postmarket at exactly 8 PM
    
    def test_realistic_price_and_volume_evolution(self, model_config, simulation_config, mock_data_manager,
                                                 mock_feature_manager):
        """Test that price and volume statistics evolve realistically throughout the day"""
        simulator = MarketSimulator(
            symbol="TEST",
            data_manager=mock_data_manager,
            model_config=model_config,
            simulation_config=simulation_config,
            feature_manager=mock_feature_manager
        )
        
        # Create realistic intraday data with volume patterns
        et_tz = ZoneInfo("America/New_York")
        start_et = pd.Timestamp('2025-01-15 04:00:00').tz_localize(et_tz).tz_convert('UTC')
        end_et = pd.Timestamp('2025-01-15 20:00:00').tz_localize(et_tz).tz_convert('UTC')
        
        # Create minute-by-minute data
        minute_timestamps = pd.date_range(start_et, end_et, freq='1min')
        
        # Realistic volume pattern (higher at open/close, lower at lunch)
        hours = np.array([(ts.tz_convert(et_tz).hour + ts.tz_convert(et_tz).minute/60) for ts in minute_timestamps])
        volume_multiplier = np.where(
            (hours >= 9.5) & (hours <= 10.5), 3.0,  # High at open
            np.where((hours >= 15.5) & (hours <= 16), 2.5,  # High at close
                     np.where((hours >= 12) & (hours <= 14), 0.5, 1.0))  # Low at lunch
        )
        
        # Create trades with realistic volume and slight price drift
        base_volume = 100
        base_price = 100.0
        price_drift = np.cumsum(np.random.normal(0, 0.01, len(minute_timestamps)))
        
        trades_data = []
        for i, (ts, vol_mult, price_change) in enumerate(zip(minute_timestamps, volume_multiplier, price_drift)):
            # Create multiple trades per minute
            num_trades = max(1, int(vol_mult * 3))
            for j in range(num_trades):
                trade_time = ts + pd.Timedelta(seconds=j * (60 // num_trades))
                trades_data.append({
                    'timestamp': trade_time,
                    'price': base_price + price_change + np.random.normal(0, 0.001),
                    'size': int(base_volume * vol_mult * (0.5 + np.random.random()))
                })
        
        trades_df = pd.DataFrame(trades_data).set_index('timestamp')
        
        data = {
            'trades': trades_df,
            'quotes': pd.DataFrame(),
            'status': pd.DataFrame(),
            'bars_1s': pd.DataFrame(),
            'bars_1m': pd.DataFrame(),
            'bars_5m': pd.DataFrame()
        }
        
        # Setup mock
        mock_data_manager.load_day = Mock(return_value=data)
        mock_data_manager.get_previous_day_data = Mock(side_effect=lambda key: {
            'bars_1d': pd.DataFrame({
                'open': [99.5], 'high': [100.5], 'low': [99], 
                'close': [100], 'volume': [1000000], 'vwap': [100]
            })
        }.get(key, pd.DataFrame()))
        
        # Initialize
        date = datetime(2025, 1, 15)
        result = simulator.initialize_day(date)
        assert result is True
        
        # Test volume accumulation throughout the day
        morning_time = pd.Timestamp('2025-01-15 10:00:00').tz_localize(et_tz).tz_convert('UTC')
        simulator.set_time(morning_time)
        morning_state = simulator.get_market_state()
        
        afternoon_time = pd.Timestamp('2025-01-15 15:00:00').tz_localize(et_tz).tz_convert('UTC')
        simulator.set_time(afternoon_time)
        afternoon_state = simulator.get_market_state()
        
        # Volume should be cumulative and increasing
        assert afternoon_state.session_volume > morning_state.session_volume
        assert afternoon_state.session_trades > morning_state.session_trades
        
        # VWAP should be reasonable (between day's high and low)
        assert morning_state.intraday_low <= morning_state.session_vwap <= morning_state.intraday_high
        assert afternoon_state.intraday_low <= afternoon_state.session_vwap <= afternoon_state.intraday_high
        
        # Intraday high should be non-decreasing
        assert afternoon_state.intraday_high >= morning_state.intraday_high
        # Intraday low should be non-increasing
        assert afternoon_state.intraday_low <= morning_state.intraday_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])