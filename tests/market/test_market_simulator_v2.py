"""Comprehensive tests for MarketSimulatorV2.

These tests focus on behavior and outputs rather than implementation details,
ensuring the simulator correctly handles:
- Pre-market warmup data (4 AM trading)
- Accurate timestamp tracking and state queries
- Sparse data interpolation
- Order execution simulation
- Halt detection and trading status
- No future data leakage (critical for RL training)
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional

from simulators.market_simulator_v2 import MarketSimulatorV2, MarketState, ExecutionResult
from data.data_manager import DataManager


class TestMarketSimulatorV2:
    """Test suite for MarketSimulatorV2."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager with test data."""
        data_manager = Mock(spec=DataManager)
        return data_manager
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample 1-second OHLCV data for testing."""
        # Create data from 4:00 AM to 8:00 PM ET (16 hours)
        start_time = pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern')
        end_time = pd.Timestamp('2024-03-15 20:00:00', tz='US/Eastern')
        
        # Create sparse data (not every second has data)
        timestamps = []
        current = start_time
        while current <= end_time:
            # Add data point
            timestamps.append(current)
            # Skip random 1-10 seconds to simulate sparse data
            skip_seconds = np.random.randint(1, 11)
            current += pd.Timedelta(seconds=skip_seconds)
        
        # Generate price data with realistic movement
        base_price = 10.0
        prices = []
        for i in range(len(timestamps)):
            # Add some random walk
            change = np.random.normal(0, 0.01)
            base_price *= (1 + change)
            prices.append(base_price)
        
        # Create OHLCV dataframe
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            data.append({
                'open': price * (1 + np.random.uniform(-0.001, 0.001)),
                'high': price * (1 + np.random.uniform(0, 0.002)),
                'low': price * (1 + np.random.uniform(-0.002, 0)),
                'close': price,
                'volume': np.random.randint(100, 10000)
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    @pytest.fixture
    def sample_quote_data(self):
        """Create sample quote data."""
        start_time = pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern')
        
        # Create quotes every 100ms with some gaps
        timestamps = []
        for i in range(1000):  # 100 seconds worth
            ts = start_time + pd.Timedelta(milliseconds=i * 100)
            if np.random.random() > 0.1:  # 90% chance of having a quote
                timestamps.append(ts)
        
        # Generate quote data
        data = []
        base_price = 10.0
        for ts in timestamps:
            spread = np.random.uniform(0.01, 0.05)
            mid_price = base_price + np.random.normal(0, 0.01)
            data.append({
                'bid_price': mid_price - spread/2,
                'ask_price': mid_price + spread/2,
                'bid_size': np.random.randint(100, 5000),
                'ask_size': np.random.randint(100, 5000)
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    @pytest.fixture
    def sample_trade_data(self):
        """Create sample trade data."""
        start_time = pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern')
        
        # Create trades at irregular intervals
        timestamps = []
        current = start_time
        for i in range(500):
            current += pd.Timedelta(seconds=np.random.exponential(2))  # Average 2 seconds between trades
            timestamps.append(current)
        
        # Generate trade data
        data = []
        base_price = 10.0
        for ts in timestamps:
            price = base_price + np.random.normal(0, 0.02)
            data.append({
                'price': price,
                'size': np.random.randint(100, 5000),
                'conditions': []
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    @pytest.fixture
    def sample_status_data(self):
        """Create sample status data including halts."""
        timestamps = [
            pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 10:15:00', tz='US/Eastern'),  # Halt start
            pd.Timestamp('2024-03-15 10:30:00', tz='US/Eastern'),  # Halt end
            pd.Timestamp('2024-03-15 16:00:00', tz='US/Eastern')
        ]
        
        data = [
            {'is_halted': False, 'is_trading': True, 'status': 'PRE_MARKET'},
            {'is_halted': False, 'is_trading': True, 'status': 'TRADING'},
            {'is_halted': True, 'is_trading': False, 'status': 'HALTED'},
            {'is_halted': False, 'is_trading': True, 'status': 'TRADING'},
            {'is_halted': False, 'is_trading': False, 'status': 'POST_MARKET'}
        ]
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    @pytest.fixture
    def simulator(self, mock_data_manager):
        """Create a MarketSimulatorV2 instance."""
        return MarketSimulatorV2(
            data_manager=mock_data_manager,
            future_buffer_minutes=5,
            default_latency_ms=100,
            commission_per_share=0.005
        )
    
    def test_initialization(self, simulator):
        """Test simulator initialization with correct parameters."""
        assert simulator.future_buffer_minutes == 5
        assert simulator.default_latency_ms == 100
        assert simulator.commission_per_share == 0.005
        assert simulator.current_symbol is None
        assert simulator.current_date is None
        assert len(simulator.execution_history) == 0
    
    def test_initialize_day(self, simulator, mock_data_manager, sample_ohlcv_data, 
                           sample_quote_data, sample_trade_data, sample_status_data):
        """Test initializing simulator for a trading day."""
        # Setup mock data
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': sample_quote_data,
            'trades': sample_trade_data,
            'status': sample_status_data
        }.get(data_type)
        
        # Initialize day
        test_date = datetime(2024, 3, 15)
        simulator.initialize_day('MLGO', test_date)
        
        # Verify state
        assert simulator.current_symbol == 'MLGO'
        assert simulator.current_date == pd.Timestamp(test_date).date()
        
        # Verify indices were built
        assert len(simulator.ohlcv_index) == len(sample_ohlcv_data)
        assert len(simulator.quote_index) == len(sample_quote_data)
        assert len(simulator.trade_index) == len(sample_trade_data)
        assert len(simulator.status_index) == len(sample_status_data)
        
        # Verify O(1) lookup works
        first_ohlcv_time = sample_ohlcv_data.index[0]
        assert first_ohlcv_time in simulator.ohlcv_index
        assert simulator.ohlcv_index[first_ohlcv_time] == 0
    
    def test_pre_market_4am_data_handling(self, simulator, mock_data_manager):
        """Test that pre-market 4 AM data is correctly handled."""
        # Create pre-market data starting at 4 AM
        pre_market_start = pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern')
        market_open = pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern')
        
        # Create OHLCV data with pre-market activity
        timestamps = pd.date_range(pre_market_start, market_open, freq='1min')
        ohlcv_data = pd.DataFrame({
            'open': np.random.uniform(9.5, 10.5, len(timestamps)),
            'high': np.random.uniform(10, 11, len(timestamps)),
            'low': np.random.uniform(9, 10, len(timestamps)),
            'close': np.random.uniform(9.5, 10.5, len(timestamps)),
            'volume': np.random.randint(100, 1000, len(timestamps))
        }, index=timestamps)
        
        # Setup mock
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        # Initialize day
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test querying pre-market timestamps
        pre_market_time = pd.Timestamp('2024-03-15 04:30:00', tz='US/Eastern')
        state = simulator.get_market_state(pre_market_time)
        
        assert state.timestamp == pre_market_time
        assert state.last_price > 0  # Should have price data
        assert state.volume >= 0
        assert not state.is_halted  # Should not be halted in pre-market
    
    def test_accurate_timestamp_tracking(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test that current timestamp is accurately maintained."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test multiple timestamp queries
        test_timestamps = [
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 12:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 15:30:00', tz='US/Eastern'),
        ]
        
        for ts in test_timestamps:
            state = simulator.get_market_state(ts)
            assert simulator.current_timestamp == ts
            assert state.timestamp == ts
    
    def test_market_state_with_exact_timestamp_match(self, simulator, mock_data_manager,
                                                    sample_ohlcv_data, sample_quote_data):
        """Test getting market state when timestamp exactly matches data."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': sample_quote_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Use exact timestamp from data
        exact_time = sample_ohlcv_data.index[10]
        state = simulator.get_market_state(exact_time)
        
        # Verify state matches data
        ohlcv_row = sample_ohlcv_data.loc[exact_time]
        assert state.timestamp == exact_time
        assert state.last_price == ohlcv_row['close']
        assert state.volume == ohlcv_row['volume']
    
    def test_market_state_with_interpolation(self, simulator, mock_data_manager):
        """Test market state queries between data points (sparse data handling)."""
        # Create sparse OHLCV data
        timestamps = [
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:10', tz='US/Eastern'),  # 10 second gap
        ]
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0, 10.1],
            'high': [10.05, 10.15],
            'low': [9.95, 10.05],
            'close': [10.02, 10.12],
            'volume': [1000, 1500]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query timestamp between data points
        query_time = pd.Timestamp('2024-03-15 09:30:05', tz='US/Eastern')
        state = simulator.get_market_state(query_time)
        
        # Should use the last available data point
        assert state.timestamp == query_time
        assert state.last_price == 10.02  # From first data point
        assert state.volume == 1000
    
    def test_interpolate_state_method(self, simulator, mock_data_manager):
        """Test the interpolate_state method for smooth transitions."""
        # Create two data points with a gap
        timestamps = [
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:10', tz='US/Eastern'),
        ]
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0, 10.2],
            'high': [10.1, 10.3],
            'low': [9.9, 10.1],
            'close': [10.0, 10.2],
            'volume': [1000, 2000]
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': [9.99, 10.19],
            'ask_price': [10.01, 10.21],
            'bid_size': [500, 600],
            'ask_size': [500, 600]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test interpolation at midpoint
        mid_time = pd.Timestamp('2024-03-15 09:30:05', tz='US/Eastern')
        interpolated_state = simulator.interpolate_state(mid_time)
        
        # Verify interpolation
        assert interpolated_state.timestamp == mid_time
        assert 9.99 < interpolated_state.bid_price < 10.19  # Should be between the two values
        assert 10.01 < interpolated_state.ask_price < 10.21
        assert interpolated_state.spread > 0
    
    def test_halt_detection(self, simulator, mock_data_manager, sample_ohlcv_data, sample_status_data):
        """Test detection of trading halts."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': sample_status_data
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test during normal trading
        normal_time = pd.Timestamp('2024-03-15 10:00:00', tz='US/Eastern')
        state = simulator.get_market_state(normal_time)
        assert not state.is_halted
        
        # Test during halt
        halt_time = pd.Timestamp('2024-03-15 10:20:00', tz='US/Eastern')
        state = simulator.get_market_state(halt_time)
        assert state.is_halted
        
        # Test after halt resumed
        resumed_time = pd.Timestamp('2024-03-15 10:35:00', tz='US/Eastern')
        state = simulator.get_market_state(resumed_time)
        assert not state.is_halted
    
    def test_market_order_execution(self, simulator, mock_data_manager, 
                                  sample_ohlcv_data, sample_quote_data):
        """Test market order execution simulation."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': sample_quote_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Execute a buy market order
        order_time = pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern')
        result = simulator.simulate_order_execution(
            timestamp=order_time,
            side='buy',
            size=1000,
            order_type='market'
        )
        
        # Verify execution result
        assert result.symbol == 'MLGO'
        assert result.side == 'buy'
        assert result.executed_size == 1000
        assert result.executed_price > 0
        assert result.slippage >= 0  # Should have some slippage
        assert result.commission == 1000 * 0.005  # 5 dollars commission
        assert result.latency_ms > 0
        assert result.rejection_reason is None
        
        # Verify execution happened in the future (due to latency)
        assert result.timestamp > order_time
    
    def test_order_execution_during_halt(self, simulator, mock_data_manager,
                                       sample_ohlcv_data, sample_status_data):
        """Test that orders are rejected during halts."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': sample_status_data
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Try to execute order during halt
        halt_time = pd.Timestamp('2024-03-15 10:20:00', tz='US/Eastern')
        result = simulator.simulate_order_execution(
            timestamp=halt_time,
            side='buy',
            size=1000,
            order_type='market'
        )
        
        # Verify rejection
        assert result.executed_size == 0
        assert result.executed_price == 0
        assert result.commission == 0
        assert result.rejection_reason == "Trading halted"
    
    def test_future_price_buffer(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test future price buffer for execution simulation."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Get future prices
        current_time = sample_ohlcv_data.index[50]  # Some point in the data
        future_data = simulator.get_future_prices(current_time, seconds_ahead=300)  # 5 minutes
        
        # Verify future data
        assert not future_data.empty
        assert (future_data.index > current_time).all()
        assert (future_data.index <= current_time + pd.Timedelta(seconds=300)).all()
    
    def test_latency_calculation(self, simulator):
        """Test realistic latency calculation."""
        # Test base latency
        base_latency = simulator._calculate_latency(
            pd.Timestamp('2024-03-15 12:00:00', tz='US/Eastern'),
            size=100
        )
        assert 10 <= base_latency <= 200  # Should be within reasonable range
        
        # Test large order latency (should be higher)
        large_order_latency = simulator._calculate_latency(
            pd.Timestamp('2024-03-15 12:00:00', tz='US/Eastern'),
            size=20000
        )
        assert large_order_latency > base_latency
        
        # Test market open latency (should be higher)
        open_latency = simulator._calculate_latency(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            size=100
        )
        assert open_latency > base_latency * 1.2  # At least 20% higher
    
    def test_slippage_calculation(self, simulator, mock_data_manager):
        """Test slippage calculation for different order sizes."""
        # Create data with some volatility
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:35:00', tz='US/Eastern'),
            freq='1s'
        )
        
        # Add volatility
        prices = 10 + np.cumsum(np.random.normal(0, 0.01, len(timestamps)))
        
        ohlcv_data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.02,
            'low': prices - 0.02,
            'close': prices,
            'volume': np.random.randint(1000, 5000, len(timestamps))
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': prices - 0.01,
            'ask_price': prices + 0.01,
            'bid_size': 1000,
            'ask_size': 1000
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test small order
        small_result = simulator.simulate_order_execution(
            timestamp=timestamps[0],
            side='buy',
            size=100,
            order_type='market'
        )
        
        # Test large order (should have more slippage)
        large_result = simulator.simulate_order_execution(
            timestamp=timestamps[0],
            side='buy',
            size=10000,
            order_type='market'
        )
        
        assert large_result.slippage > small_result.slippage
    
    def test_execution_statistics(self, simulator, mock_data_manager, 
                                sample_ohlcv_data, sample_quote_data):
        """Test execution statistics tracking."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': sample_quote_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Execute several orders
        timestamps = sample_ohlcv_data.index[:5]
        for i, ts in enumerate(timestamps):
            simulator.simulate_order_execution(
                timestamp=ts,
                side='buy' if i % 2 == 0 else 'sell',
                size=1000,
                order_type='market'
            )
        
        # Get statistics
        stats = simulator.get_execution_stats()
        
        assert stats['total_executions'] == 5
        assert stats['total_shares'] == 5000
        assert stats['buy_count'] == 3
        assert stats['sell_count'] == 2
        assert stats['avg_slippage'] > 0
        assert stats['total_commission'] == 5000 * 0.005
    
    def test_reset_functionality(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test simulator reset clears all state."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        # Initialize and add some state
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        simulator.simulate_order_execution(
            timestamp=sample_ohlcv_data.index[0],
            side='buy',
            size=1000,
            order_type='market'
        )
        
        # Verify state exists
        assert len(simulator.ohlcv_index) > 0
        assert len(simulator.execution_history) > 0
        assert simulator.current_symbol is not None
        
        # Reset
        simulator.reset()
        
        # Verify state cleared
        assert len(simulator.ohlcv_index) == 0
        assert len(simulator.execution_history) == 0
        assert simulator.current_symbol is None
        assert simulator.ohlcv_1s is None
    
    def test_edge_case_no_data(self, simulator, mock_data_manager):
        """Test handling when no data is available."""
        mock_data_manager.get_active_day_data.return_value = pd.DataFrame()
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Should handle gracefully
        state = simulator.get_market_state(pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'))
        
        assert state.bid_price == 0
        assert state.ask_price == 0
        assert state.volume == 0
        # The actual implementation returns False for is_halted when no status data
        # This is acceptable behavior - no data doesn't necessarily mean halted
        assert state.is_halted == False
    
    def test_edge_case_invalid_spread(self, simulator, mock_data_manager):
        """Test handling of invalid bid/ask spreads."""
        # Create data with invalid spread (bid > ask)
        timestamps = [pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern')]
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0],
            'high': [10.1],
            'low': [9.9],
            'close': [10.0],
            'volume': [1000]
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': [10.5],  # Invalid: bid > ask
            'ask_price': [10.0],
            'bid_size': [500],
            'ask_size': [500]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Should fix the spread
        state = simulator.get_market_state(timestamps[0])
        
        assert state.bid_price < state.ask_price
        assert state.spread > 0
        assert state.bid_price > 0
        assert state.ask_price > 0
    
    def test_market_impact_modeling(self, simulator, mock_data_manager):
        """Test that market impact is modeled based on order size."""
        # Create stable market data
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:31:00', tz='US/Eastern'),
            freq='1s'
        )
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0] * len(timestamps),
            'high': [10.01] * len(timestamps),
            'low': [9.99] * len(timestamps),
            'close': [10.0] * len(timestamps),
            'volume': [1000] * len(timestamps)
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': [9.99] * len(timestamps),
            'ask_price': [10.01] * len(timestamps),
            'bid_size': [1000] * len(timestamps),
            'ask_size': [1000] * len(timestamps)
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Execute orders of different sizes
        sizes = [100, 1000, 5000, 10000]
        slippages = []
        
        for size in sizes:
            result = simulator.simulate_order_execution(
                timestamp=timestamps[0],
                side='buy',
                size=size,
                order_type='market'
            )
            slippages.append(result.slippage)
        
        # Verify increasing slippage with size
        for i in range(1, len(slippages)):
            assert slippages[i] >= slippages[i-1]
    
    def test_limit_order_execution(self, simulator, mock_data_manager):
        """Test limit order execution logic."""
        timestamps = [pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern')]
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0],
            'high': [10.1],
            'low': [9.9],
            'close': [10.0],
            'volume': [1000]
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': [9.99],
            'ask_price': [10.01],
            'bid_size': [1000],
            'ask_size': [1000]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test marketable limit order (should execute)
        marketable_result = simulator.simulate_order_execution(
            timestamp=timestamps[0],
            side='buy',
            size=1000,
            order_type='limit',
            limit_price=10.05  # Above ask, marketable
        )
        
        assert marketable_result.executed_size == 1000
        assert marketable_result.executed_price > 0
        
        # Test non-marketable limit order (should not execute)
        non_marketable_result = simulator.simulate_order_execution(
            timestamp=timestamps[0],
            side='buy',
            size=1000,
            order_type='limit',
            limit_price=9.95  # Below bid, non-marketable
        )
        
        assert non_marketable_result.executed_size == 0
        assert non_marketable_result.executed_price == 0
    
    def test_continuous_time_progression(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test that simulator maintains consistent time progression."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query states at increasing timestamps
        base_time = pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern')
        states = []
        
        for i in range(10):
            ts = base_time + pd.Timedelta(seconds=i)
            state = simulator.get_market_state(ts)
            states.append(state)
        
        # Verify timestamps are correct and increasing
        for i, state in enumerate(states):
            assert state.timestamp == base_time + pd.Timedelta(seconds=i)
            if i > 0:
                assert state.timestamp > states[i-1].timestamp
    
    def test_no_future_data_leakage_in_market_state(self, simulator, mock_data_manager):
        """Test that market state queries never return data from the future."""
        # Create data with distinct values at each timestamp
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:10', tz='US/Eastern'),
            freq='1s'
        )
        
        # Create prices that increase linearly so we can detect future leakage
        prices = np.arange(10.0, 10.0 + len(timestamps) * 0.01, 0.01)
        
        ohlcv_data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.01,
            'low': prices - 0.01,
            'close': prices,
            'volume': range(1000, 1000 + len(timestamps))
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query at the 5th second
        query_time = timestamps[5]
        state = simulator.get_market_state(query_time)
        
        # The price should be exactly the 5th price, not any future price
        assert state.last_price == prices[5]
        assert state.volume == 1005  # Should be the 5th volume
        
        # Query between timestamps (at 5.5 seconds)
        between_time = timestamps[5] + pd.Timedelta(milliseconds=500)
        state_between = simulator.get_market_state(between_time)
        
        # Should still use data from timestamp 5, not 6
        assert state_between.last_price == prices[5]
        assert state_between.volume == 1005
    
    def test_no_future_quotes_in_market_state(self, simulator, mock_data_manager):
        """Test that quote data never includes future information."""
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:05', tz='US/Eastern'),
            freq='1s'
        )
        
        # Create quotes with increasing spreads to detect future leakage
        quotes_data = pd.DataFrame({
            'bid_price': [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            'ask_price': [10.01, 10.11, 10.21, 10.31, 10.41, 10.51],
            'bid_size': [100, 200, 300, 400, 500, 600],
            'ask_size': [100, 200, 300, 400, 500, 600]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': pd.DataFrame(),
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query at second 2
        query_time = timestamps[2]
        state = simulator.get_market_state(query_time)
        
        # Should get exact values from second 2
        assert state.bid_price == 10.2
        assert state.ask_price == 10.21
        assert state.bid_size == 300
        assert state.ask_size == 300
        
        # Query between seconds 2 and 3
        between_time = timestamps[2] + pd.Timedelta(milliseconds=500)
        state_between = simulator.get_market_state(between_time)
        
        # Should still get values from second 2, not 3
        assert state_between.bid_price == 10.2
        assert state_between.ask_price == 10.21
    
    def test_interpolation_uses_only_past_data(self, simulator, mock_data_manager):
        """Test that interpolation only uses past and current data, never future."""
        # Create sparse data with gaps
        timestamps = [
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:05', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:10', tz='US/Eastern'),
        ]
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'high': [10.1, 11.1, 12.1],
            'low': [9.9, 10.9, 11.9],
            'close': [10.0, 11.0, 12.0],
            'volume': [1000, 2000, 3000]
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': [9.99, 10.99, 11.99],
            'ask_price': [10.01, 11.01, 12.01],
            'bid_size': [100, 200, 300],
            'ask_size': [100, 200, 300]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query at second 3 (between first and second data points)
        query_time = pd.Timestamp('2024-03-15 09:30:03', tz='US/Eastern')
        state = simulator.get_market_state(query_time)
        
        # Should only use data from first timestamp (no future data)
        assert state.last_price == 10.0
        assert state.bid_price == 9.99
        assert state.ask_price == 10.01
        
        # Test interpolate_state method directly
        interpolated = simulator.interpolate_state(query_time)
        
        # When interpolating at second 3, it should use data from seconds 0 and 5
        # But the values should be between the two points
        assert 9.99 <= interpolated.bid_price <= 10.99
        assert 10.01 <= interpolated.ask_price <= 11.01
    
    def test_future_buffer_isolation(self, simulator, mock_data_manager):
        """Test that future buffer is only used for execution, not state queries."""
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:35:00', tz='US/Eastern'),
            freq='10s'
        )
        
        # Create data with clear pattern
        ohlcv_data = pd.DataFrame({
            'open': range(100, 100 + len(timestamps)),
            'high': range(101, 101 + len(timestamps)),
            'low': range(99, 99 + len(timestamps)),
            'close': range(100, 100 + len(timestamps)),
            'volume': range(1000, 1000 + len(timestamps))
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query state at first timestamp
        query_time = timestamps[0]
        state = simulator.get_market_state(query_time)
        
        # Get future prices (this should work)
        future_prices = simulator.get_future_prices(query_time, seconds_ahead=60)
        
        # Verify state doesn't include future data
        assert state.last_price == 100  # First price
        assert state.volume == 1000  # First volume
        
        # Verify future buffer contains actual future data
        assert not future_prices.empty
        assert (future_prices.index > query_time).all()
        assert future_prices['close'].iloc[0] > 100  # Future prices
    
    def test_execution_uses_future_data_correctly(self, simulator, mock_data_manager):
        """Test that execution simulation uses future data for realistic slippage."""
        # Create data with volatility spike in the future
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:31:00', tz='US/Eastern'),
            freq='1s'
        )
        
        # Stable prices initially, then spike
        prices = [10.0] * 30 + [10.5] * 31  # Spike after 30 seconds
        
        ohlcv_data = pd.DataFrame({
            'open': prices,
            'high': [p + 0.01 for p in prices],
            'low': [p - 0.01 for p in prices],
            'close': prices,
            'volume': [1000] * len(timestamps)
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': [p - 0.01 for p in prices],
            'ask_price': [p + 0.01 for p in prices],
            'bid_size': [1000] * len(timestamps),
            'ask_size': [1000] * len(timestamps)
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Execute order at second 25 (before spike)
        order_time = timestamps[25]
        
        # First verify market state shows no spike yet
        state = simulator.get_market_state(order_time)
        assert abs(state.last_price - 10.0) < 0.02
        
        # Execute order with high latency to hit the spike
        simulator.default_latency_ms = 5000  # 5 second latency
        result = simulator.simulate_order_execution(
            timestamp=order_time,
            side='buy',
            size=10000,  # Large order
            order_type='market'
        )
        
        # Execution should reflect future volatility in slippage
        assert result.executed_price > 10.02  # More than just spread
        assert result.slippage > 0.02  # Significant slippage due to future movement
        
        # But querying state at order time should still show pre-spike prices
        state_at_order = simulator.get_market_state(order_time)
        assert abs(state_at_order.last_price - 10.0) < 0.02
    
    def test_halt_status_temporal_consistency(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test that halt status respects temporal boundaries."""
        # Create status changes over time
        status_data = pd.DataFrame([
            {'is_halted': False, 'is_trading': True, 'status': 'TRADING'},
            {'is_halted': True, 'is_trading': False, 'status': 'HALTED'},
            {'is_halted': False, 'is_trading': True, 'status': 'TRADING'},
        ], index=[
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 10:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 10:30:00', tz='US/Eastern'),
        ])
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': status_data
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test before halt
        state_before = simulator.get_market_state(pd.Timestamp('2024-03-15 09:45:00', tz='US/Eastern'))
        assert not state_before.is_halted
        
        # Test during halt
        state_during = simulator.get_market_state(pd.Timestamp('2024-03-15 10:15:00', tz='US/Eastern'))
        assert state_during.is_halted
        
        # Test after halt (but query time before resume)
        state_query = simulator.get_market_state(pd.Timestamp('2024-03-15 10:25:00', tz='US/Eastern'))
        assert state_query.is_halted  # Should still be halted
        
        # Test after resume
        state_after = simulator.get_market_state(pd.Timestamp('2024-03-15 10:35:00', tz='US/Eastern'))
        assert not state_after.is_halted
    
    def test_data_consistency_across_queries(self, simulator, mock_data_manager):
        """Test that repeated queries at the same timestamp return consistent data."""
        # Create deterministic data
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:31:00', tz='US/Eastern'),
            freq='5s'
        )
        
        ohlcv_data = pd.DataFrame({
            'open': [10.0 + i * 0.01 for i in range(len(timestamps))],
            'high': [10.1 + i * 0.01 for i in range(len(timestamps))],
            'low': [9.9 + i * 0.01 for i in range(len(timestamps))],
            'close': [10.0 + i * 0.01 for i in range(len(timestamps))],
            'volume': [1000 + i * 10 for i in range(len(timestamps))]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Query same timestamp multiple times
        query_time = timestamps[5]
        states = []
        for _ in range(10):
            state = simulator.get_market_state(query_time)
            states.append(state)
        
        # All states should be identical
        first_state = states[0]
        for state in states[1:]:
            assert state.timestamp == first_state.timestamp
            assert state.last_price == first_state.last_price
            assert state.volume == first_state.volume
            assert state.bid_price == first_state.bid_price
            assert state.ask_price == first_state.ask_price
    
    def test_weekend_warmup_data_handling(self, simulator, mock_data_manager):
        """Test that simulator handles weekend gaps when loading warmup data for Monday 4AM trading."""
        # Monday March 18, 2024
        monday_date = datetime(2024, 3, 18)
        
        # Create Monday 4AM data
        monday_timestamps = pd.date_range(
            pd.Timestamp('2024-03-18 04:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-18 04:05:00', tz='US/Eastern'),
            freq='1min'
        )
        
        monday_ohlcv = pd.DataFrame({
            'open': [10.5] * len(monday_timestamps),
            'high': [10.6] * len(monday_timestamps),
            'low': [10.4] * len(monday_timestamps),
            'close': [10.55] * len(monday_timestamps),
            'volume': [500] * len(monday_timestamps)
        }, index=monday_timestamps)
        
        # Mock data manager should handle weekend gap
        # In real implementation, DataManager would load Friday's data for lookback
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': monday_ohlcv,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        # This tests that the simulator can initialize on Monday
        # The DataManager is responsible for loading previous Friday data
        simulator.initialize_day('MLGO', monday_date)
        
        # Query Monday 4AM state
        monday_4am = pd.Timestamp('2024-03-18 04:00:00', tz='US/Eastern')
        state = simulator.get_market_state(monday_4am)
        
        # Should have valid state from Monday pre-market
        assert state.timestamp == monday_4am
        assert state.last_price > 0
        assert not state.is_halted
        
        # The actual weekend handling logic would be in DataManager
        # which would load Friday data as lookback
        # MarketSimulatorV2 just works with what DataManager provides
    
    def test_holiday_warmup_data_handling(self, simulator, mock_data_manager):
        """Test handling of trading after market holidays."""
        # Tuesday after a Monday holiday
        tuesday_date = datetime(2024, 1, 2)  # Day after New Year's Day
        
        # Create Tuesday 4AM data
        tuesday_timestamps = pd.date_range(
            pd.Timestamp('2024-01-02 04:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-01-02 04:10:00', tz='US/Eastern'),
            freq='30s'
        )
        
        tuesday_ohlcv = pd.DataFrame({
            'open': np.linspace(10.0, 10.2, len(tuesday_timestamps)),
            'high': np.linspace(10.1, 10.3, len(tuesday_timestamps)),
            'low': np.linspace(9.9, 10.1, len(tuesday_timestamps)),
            'close': np.linspace(10.0, 10.2, len(tuesday_timestamps)),
            'volume': np.linspace(1000, 2000, len(tuesday_timestamps))
        }, index=tuesday_timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': tuesday_ohlcv,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        # Initialize for Tuesday (DataManager would handle loading last trading day)
        simulator.initialize_day('MLGO', tuesday_date)
        
        # Query early morning state
        early_morning = pd.Timestamp('2024-01-02 04:05:00', tz='US/Eastern')
        state = simulator.get_market_state(early_morning)
        
        assert state.timestamp == early_morning
        assert state.last_price > 0
        assert state.volume > 0
    
    def test_pre_market_data_with_previous_day_context(self, simulator, mock_data_manager):
        """Test that pre-market queries work correctly with previous day context.
        
        This tests the scenario where we need previous day's close for indicators
        but are trading in pre-market hours.
        """
        # Current day (Tuesday)
        current_date = datetime(2024, 3, 19)
        
        # Create sparse pre-market data (4AM - 9:30AM)
        pre_market_times = [
            pd.Timestamp('2024-03-19 04:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 04:15:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 04:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 05:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 06:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 07:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 08:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 09:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-19 09:25:00', tz='US/Eastern'),
        ]
        
        # Simulate gap up from previous close
        previous_close = 10.0
        gap_up_open = 10.5  # 5% gap up
        
        pre_market_ohlcv = pd.DataFrame({
            'open': [gap_up_open + i * 0.01 for i in range(len(pre_market_times))],
            'high': [gap_up_open + 0.05 + i * 0.01 for i in range(len(pre_market_times))],
            'low': [gap_up_open - 0.05 + i * 0.01 for i in range(len(pre_market_times))],
            'close': [gap_up_open + i * 0.01 for i in range(len(pre_market_times))],
            'volume': [100 + i * 50 for i in range(len(pre_market_times))]  # Low pre-market volume
        }, index=pre_market_times)
        
        # Add some quote data for pre-market
        pre_market_quotes = pd.DataFrame({
            'bid_price': [gap_up_open - 0.02 + i * 0.01 for i in range(len(pre_market_times))],
            'ask_price': [gap_up_open + 0.02 + i * 0.01 for i in range(len(pre_market_times))],
            'bid_size': [50] * len(pre_market_times),  # Small pre-market sizes
            'ask_size': [50] * len(pre_market_times)
        }, index=pre_market_times)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': pre_market_ohlcv,
            'quotes': pre_market_quotes,
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', current_date)
        
        # Test querying at various pre-market times
        test_times = [
            ('Early pre-market', pd.Timestamp('2024-03-19 04:00:00', tz='US/Eastern')),
            ('Mid pre-market', pd.Timestamp('2024-03-19 06:30:00', tz='US/Eastern')),
            ('Late pre-market', pd.Timestamp('2024-03-19 09:15:00', tz='US/Eastern')),
        ]
        
        for desc, query_time in test_times:
            state = simulator.get_market_state(query_time)
            
            # Verify we get valid pre-market state
            assert state.timestamp == query_time, f"Failed at {desc}"
            assert state.last_price > previous_close, f"Gap up not reflected at {desc}"
            assert state.bid_price > 0 and state.ask_price > 0, f"No quotes at {desc}"
            assert state.spread > 0, f"Invalid spread at {desc}"
            assert not state.is_halted, f"Should not be halted in pre-market at {desc}"
    
    def test_first_trading_day_of_symbol(self, simulator, mock_data_manager):
        """Test handling when this is the first day of data for a symbol (no previous day exists)."""
        # First trading day for a new IPO
        ipo_date = datetime(2024, 3, 15)
        
        # Create data starting from market open (no pre-market data for IPO)
        ipo_timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:35:00', tz='US/Eastern'),
            freq='30s'
        )
        
        ipo_ohlcv = pd.DataFrame({
            'open': [20.0] * len(ipo_timestamps),  # IPO price
            'high': [20.5] * len(ipo_timestamps),
            'low': [19.5] * len(ipo_timestamps),
            'close': [20.2] * len(ipo_timestamps),
            'volume': [100000] * len(ipo_timestamps)  # High IPO volume
        }, index=ipo_timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ipo_ohlcv,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        # Should handle initialization without previous day data
        simulator.initialize_day('NEWIPO', ipo_date)
        
        # Query at market open
        market_open = pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern')
        state = simulator.get_market_state(market_open)
        
        assert state.timestamp == market_open
        assert state.last_price == 20.2
        assert state.volume == 100000
        
        # Query before market open (no pre-market data)
        pre_open = pd.Timestamp('2024-03-15 09:00:00', tz='US/Eastern')
        early_state = simulator.get_market_state(pre_open)
        
        # Should return empty/halted state
        assert early_state.timestamp == pre_open
        assert early_state.last_price == 0 or early_state.is_halted
    
    def test_transition_from_pre_market_to_regular_hours(self, simulator, mock_data_manager):
        """Test smooth transition from pre-market to regular trading hours."""
        # Create data spanning pre-market to regular hours
        all_timestamps = []
        
        # Pre-market (4AM - 9:30AM)
        pre_market = pd.date_range(
            pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:29:00', tz='US/Eastern'),
            freq='1min'
        )
        all_timestamps.extend(pre_market)
        
        # Regular hours (9:30AM onwards)
        regular_hours = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 10:00:00', tz='US/Eastern'),
            freq='1s'
        )
        all_timestamps.extend(regular_hours)
        
        # Create continuous data
        all_timestamps = sorted(all_timestamps)
        base_price = 10.0
        
        # Pre-market has less volatility, regular hours more
        prices = []
        volumes = []
        for ts in all_timestamps:
            if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
                # Pre-market: low volatility, low volume
                price_change = np.random.normal(0, 0.001)
                volume = np.random.randint(10, 100)
            else:
                # Regular hours: higher volatility, higher volume
                price_change = np.random.normal(0, 0.005)
                volume = np.random.randint(1000, 5000)
            
            base_price *= (1 + price_change)
            prices.append(base_price)
            volumes.append(volume)
        
        combined_ohlcv = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=all_timestamps)
        
        # Status data showing transition
        status_data = pd.DataFrame([
            {'is_halted': False, 'is_trading': True, 'status': 'PRE_MARKET'},
            {'is_halted': False, 'is_trading': True, 'status': 'TRADING'},
        ], index=[
            pd.Timestamp('2024-03-15 04:00:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
        ])
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': combined_ohlcv,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': status_data
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test states around the transition
        transition_times = [
            pd.Timestamp('2024-03-15 09:29:30', tz='US/Eastern'),  # 30 seconds before
            pd.Timestamp('2024-03-15 09:29:50', tz='US/Eastern'),  # 10 seconds before
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),  # Exact transition
            pd.Timestamp('2024-03-15 09:30:10', tz='US/Eastern'),  # 10 seconds after
            pd.Timestamp('2024-03-15 09:30:30', tz='US/Eastern'),  # 30 seconds after
        ]
        
        states = []
        for ts in transition_times:
            state = simulator.get_market_state(ts)
            states.append(state)
            
            # All states should be valid
            assert state.timestamp == ts
            assert state.last_price > 0
            assert not state.is_halted
        
        # Volume should increase after market open
        pre_open_volume = states[0].volume
        post_open_volume = states[-1].volume
        assert post_open_volume > pre_open_volume * 5  # Significant volume increase
    
    def test_get_historical_bars(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test getting historical bars up to a timestamp (for FeatureExtractor)."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test getting bars up to a specific timestamp
        query_time = sample_ohlcv_data.index[10]  # 10th timestamp
        historical_bars = simulator.get_historical_bars(query_time)
        
        # Should get all bars up to and including query_time
        assert not historical_bars.empty
        assert (historical_bars.index <= query_time).all()
        assert len(historical_bars) == 11  # 0 to 10 inclusive
        
        # Test with lookback window
        historical_bars_windowed = simulator.get_historical_bars(query_time, lookback_minutes=5)
        
        # Should be smaller dataset
        assert len(historical_bars_windowed) <= len(historical_bars)
        assert (historical_bars_windowed.index <= query_time).all()
        
        # Verify no future data leakage
        assert query_time in historical_bars.index or len(historical_bars) == 0
        future_data = sample_ohlcv_data[sample_ohlcv_data.index > query_time]
        if not future_data.empty:
            # Make sure no future timestamps are included
            assert not any(ts in historical_bars.index for ts in future_data.index)
    
    def test_get_current_bars(self, simulator, mock_data_manager, sample_ohlcv_data):
        """Test getting current bars up to simulator's current timestamp."""
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': sample_ohlcv_data,
            'quotes': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Set current timestamp by querying market state
        current_time = sample_ohlcv_data.index[15]
        simulator.get_market_state(current_time)  # This sets current_timestamp
        
        # Get current bars
        current_bars = simulator.get_current_bars(lookback_minutes=60)
        
        # Should get bars up to current timestamp
        assert not current_bars.empty
        assert (current_bars.index <= current_time).all()
        assert simulator.current_timestamp == current_time
        
        # Test with different lookback
        recent_bars = simulator.get_current_bars(lookback_minutes=5)
        assert len(recent_bars) <= len(current_bars)
    
    def test_historical_data_methods_prevent_future_leakage(self, simulator, mock_data_manager):
        """Test that all historical data methods prevent future data leakage."""
        # Create data with clear temporal ordering
        timestamps = pd.date_range(
            pd.Timestamp('2024-03-15 09:30:00', tz='US/Eastern'),
            pd.Timestamp('2024-03-15 09:35:00', tz='US/Eastern'),
            freq='30s'
        )
        
        ohlcv_data = pd.DataFrame({
            'open': range(100, 100 + len(timestamps)),
            'high': range(101, 101 + len(timestamps)),
            'low': range(99, 99 + len(timestamps)),
            'close': range(100, 100 + len(timestamps)),
            'volume': range(1000, 1000 + len(timestamps))
        }, index=timestamps)
        
        quotes_data = pd.DataFrame({
            'bid_price': range(99, 99 + len(timestamps)),
            'ask_price': range(101, 101 + len(timestamps)),
            'bid_size': [500] * len(timestamps),
            'ask_size': [500] * len(timestamps)
        }, index=timestamps)
        
        trades_data = pd.DataFrame({
            'price': range(100, 100 + len(timestamps)),
            'size': range(100, 100 + len(timestamps)),
            'conditions': [[] for _ in range(len(timestamps))]
        }, index=timestamps)
        
        mock_data_manager.get_active_day_data.side_effect = lambda data_type: {
            'bars_1s': ohlcv_data,
            'quotes': quotes_data,
            'trades': trades_data,
            'status': pd.DataFrame()
        }.get(data_type)
        
        simulator.initialize_day('MLGO', datetime(2024, 3, 15))
        
        # Test at middle timestamp
        query_time = timestamps[5]  # Middle of the data
        
        # Test all historical data methods
        historical_bars = simulator.get_historical_bars(query_time)
        historical_quotes = simulator.get_historical_quotes(query_time)
        historical_trades = simulator.get_historical_trades(query_time)
        
        # Verify no future data in any method
        assert (historical_bars.index <= query_time).all()
        assert (historical_quotes.index <= query_time).all()
        assert (historical_trades.index <= query_time).all()
        
        # Verify we get exactly the right amount of data
        assert len(historical_bars) == 6  # timestamps 0-5 inclusive
        assert len(historical_quotes) == 6
        assert len(historical_trades) == 6
        
        # Verify the latest values match the query timestamp
        assert historical_bars.iloc[-1]['close'] == 105  # 100 + 5
        assert historical_quotes.iloc[-1]['bid_price'] == 104  # 99 + 5
        assert historical_trades.iloc[-1]['price'] == 105  # 100 + 5