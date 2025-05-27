"""Test uniform window generation for MarketSimulatorV2.

This module tests the uniform window generation functionality that provides
compatibility with the existing feature extraction system.
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
from unittest.mock import Mock, patch

from simulators.market_simulator_v2 import MarketSimulatorV2
from data.data_manager import DataManager


class TestMarketSimulatorV2Windows:
    """Test uniform window generation functionality."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        return Mock(spec=DataManager)
        
    @pytest.fixture
    def simulator(self, mock_data_manager):
        """Create a MarketSimulatorV2 instance with mock data."""
        sim = MarketSimulatorV2(mock_data_manager)
        
        # Create sample data with different densities
        base_time = pd.Timestamp('2025-01-15 10:00:00')
        
        # Sparse 1s OHLCV data (some seconds missing)
        ohlcv_times = [
            base_time - pd.Timedelta(seconds=60),  # 1 min ago
            base_time - pd.Timedelta(seconds=55),
            base_time - pd.Timedelta(seconds=50),
            base_time - pd.Timedelta(seconds=40),  # Gap here
            base_time - pd.Timedelta(seconds=30),
            base_time - pd.Timedelta(seconds=20),
            base_time - pd.Timedelta(seconds=10),
            base_time - pd.Timedelta(seconds=5),
            base_time
        ]
        
        ohlcv_data = []
        for i, ts in enumerate(ohlcv_times):
            price = 100 + i * 0.1
            ohlcv_data.append({
                'open': price,
                'high': price + 0.05,
                'low': price - 0.05,
                'close': price + 0.02,
                'volume': 1000 + i * 100
            })
            
        sim.ohlcv_1s = pd.DataFrame(ohlcv_data, index=ohlcv_times)
        
        # Build OHLCV index
        for idx, ts in enumerate(ohlcv_times):
            sim.ohlcv_index[ts] = idx
            
        # Sparse quote data
        quote_times = [
            base_time - pd.Timedelta(seconds=58),
            base_time - pd.Timedelta(seconds=45),
            base_time - pd.Timedelta(seconds=25),
            base_time - pd.Timedelta(seconds=15),
            base_time - pd.Timedelta(seconds=3),
            base_time
        ]
        
        quote_data = []
        for i, ts in enumerate(quote_times):
            price = 100 + i * 0.1
            quote_data.append({
                'bid_price': price - 0.01,
                'ask_price': price + 0.01,
                'bid_size': 500,
                'ask_size': 600
            })
            
        sim.quotes = pd.DataFrame(quote_data, index=quote_times)
        
        # Build quote index
        for idx, ts in enumerate(quote_times):
            sim.quote_index[ts] = idx
            
        # Sparse trade data
        trade_times = [
            base_time - pd.Timedelta(seconds=59),
            base_time - pd.Timedelta(seconds=52),
            base_time - pd.Timedelta(seconds=35),
            base_time - pd.Timedelta(seconds=18),
            base_time - pd.Timedelta(seconds=7),
            base_time - pd.Timedelta(seconds=1)
        ]
        
        trade_data = []
        for i, ts in enumerate(trade_times):
            trade_data.append({
                'price': 100 + i * 0.15,
                'size': 200 + i * 50
            })
            
        sim.trades = pd.DataFrame(trade_data, index=trade_times)
        
        # Build trade index
        for idx, ts in enumerate(trade_times):
            sim.trade_index[ts] = idx
            
        # Set current timestamp
        sim.current_timestamp = base_time
        
        return sim
        
    def test_get_current_market_state_basic(self, simulator):
        """Test basic market state generation."""
        state = simulator.get_current_market_state()
        
        # Check structure
        assert 'timestamp_utc' in state
        assert 'market_session' in state
        assert 'current_price' in state
        assert 'hf_data_window' in state
        assert '1m_bars_window' in state
        assert '5m_bars_window' in state
        
        # Check window sizes
        assert len(state['hf_data_window']) == 60  # 60 seconds
        assert len(state['1m_bars_window']) == 60  # 60 1-minute bars
        assert len(state['5m_bars_window']) == 60  # 60 5-minute bars
        
    def test_hf_window_generation_with_gaps(self, simulator):
        """Test HF window generation fills gaps correctly."""
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        assert len(window) == 60
        
        # Check all timestamps are present
        start_time = simulator.current_timestamp - pd.Timedelta(seconds=59)
        for i, entry in enumerate(window):
            expected_ts = start_time + pd.Timedelta(seconds=i)
            assert entry['timestamp'] == expected_ts
            
        # Check synthetic vs real bars
        synthetic_count = sum(1 for entry in window if entry['1s_bar']['is_synthetic'])
        real_count = sum(1 for entry in window if not entry['1s_bar']['is_synthetic'])
        
        # We have 9 real data points in 60 seconds (but one might be outside window)
        assert real_count >= 8
        assert synthetic_count <= 52
        
        # Check forward fill works
        # Find a synthetic bar and verify it has forward filled price
        synthetic_idx = None
        for i in range(1, len(window)):
            if window[i]['1s_bar']['is_synthetic'] and not window[i-1]['1s_bar']['is_synthetic']:
                synthetic_idx = i
                break
                
        if synthetic_idx is not None:
            # Price should be forward filled from previous bar
            assert window[synthetic_idx]['1s_bar']['close'] == window[synthetic_idx-1]['1s_bar']['close']
        
    def test_hf_window_trades_aggregation(self, simulator):
        """Test trades are correctly aggregated into 1s windows."""
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        # Count total trades
        total_trades = sum(len(entry['trades']) for entry in window)
        assert total_trades == 6  # We have 6 trades in the test data
        
        # Check specific second with trade
        # Find the entry with the last trade (1s ago from current timestamp)
        found_last_trade = False
        for entry in window[-5:]:  # Check last 5 seconds
            if len(entry['trades']) > 0:
                for trade in entry['trades']:
                    if trade['price'] == 100.75 and trade['size'] == 450:
                        found_last_trade = True
                        break
        assert found_last_trade, "Last trade not found in window"
        
    def test_hf_window_quotes_forward_fill(self, simulator):
        """Test quotes are forward filled when missing."""
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        # Check that all entries have quotes
        for entry in window:
            assert len(entry['quotes']) > 0
            
        # Check forward fill between real quotes
        # We have a quote at 15s ago and next at 3s ago
        for i in range(44, 56):  # 16s to 4s ago
            entry = window[i]
            # Should have forward filled quote
            assert len(entry['quotes']) == 1
            quote = entry['quotes'][0]
            # Should match the quote from 15s ago (with float tolerance)
            assert abs(quote['bid_price'] - 100.29) < 0.01  # Price from 15s ago quote
            assert abs(quote['ask_price'] - 100.31) < 0.01
            
    def test_hf_window_trade_classification(self, simulator):
        """Test trades are classified as buy/sell based on quotes."""
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        # Find entries with trades
        for entry in window:
            if entry['trades']:
                for trade in entry['trades']:
                    # Check if conditions are set when quotes are available
                    if entry['quotes'] and entry['quotes'][0]['bid_price'] and entry['quotes'][0]['ask_price']:
                        if 'conditions' in trade and trade['conditions']:
                            # Trade should be classified as BUY or SELL
                            assert trade['conditions'][0] in ['BUY', 'SELL']
                            
    def test_bar_window_generation_1m(self, simulator):
        """Test 1-minute bar generation."""
        bars = simulator._generate_bar_window(simulator.current_timestamp, 60, '1m')
        
        assert len(bars) == 60
        
        # Check timestamps are aligned to minute boundaries
        for bar in bars:
            assert bar['timestamp'].second == 0
            assert bar['timestamp'].microsecond == 0
            
        # Check aggregation works
        # Last bar should aggregate multiple 1s bars
        last_bar = bars[-1]
        assert not last_bar['is_synthetic']  # Should have real data
        assert last_bar['volume'] > 0
        
    def test_bar_window_generation_5m(self, simulator):
        """Test 5-minute bar generation."""
        bars = simulator._generate_bar_window(simulator.current_timestamp, 12, '5m')
        
        assert len(bars) == 12  # 12 5-minute bars
        
        # Check timestamps are aligned to 5-minute boundaries
        for bar in bars:
            assert bar['timestamp'].minute % 5 == 0
            assert bar['timestamp'].second == 0
            
    def test_no_data_edge_case(self, mock_data_manager):
        """Test when no data is available."""
        simulator = MarketSimulatorV2(mock_data_manager)
        
        # Initialize with empty data
        simulator.ohlcv_1s = pd.DataFrame()
        simulator.quotes = pd.DataFrame()
        simulator.trades = pd.DataFrame()
        simulator.current_timestamp = pd.Timestamp('2025-01-15 10:00:00')
        
        # Should still generate windows with all synthetic data
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        assert len(window) == 60
        # All should be synthetic with 0 price
        for entry in window:
            assert entry['1s_bar']['is_synthetic']
            assert entry['1s_bar']['close'] == 0
            assert entry['1s_bar']['volume'] == 0
            
    def test_partial_data_edge_case(self, simulator):
        """Test when data starts in the middle of requested window."""
        # Request window that extends before available data
        window = simulator._generate_hf_window(
            simulator.current_timestamp + pd.Timedelta(seconds=30),  # 30s in future
            120  # 2 minute window
        )
        
        assert len(window) == 120
        
        # First part should be synthetic (no data)
        # Last part should have real data
        first_half_synthetic = sum(1 for entry in window[:60] if entry['1s_bar']['is_synthetic'])
        second_half_real = sum(1 for entry in window[60:] if not entry['1s_bar']['is_synthetic'])
        
        assert first_half_synthetic > second_half_real
        
    def test_market_session_detection(self, simulator):
        """Test market session detection."""
        # Test different times
        test_cases = [
            (pd.Timestamp('2025-01-15 03:00:00'), 'CLOSED'),
            (pd.Timestamp('2025-01-15 05:00:00'), 'PREMARKET'),
            (pd.Timestamp('2025-01-15 09:00:00'), 'PREMARKET'),
            (pd.Timestamp('2025-01-15 09:30:00'), 'REGULAR'),
            (pd.Timestamp('2025-01-15 12:00:00'), 'REGULAR'),
            (pd.Timestamp('2025-01-15 16:00:00'), 'POSTMARKET'),
            (pd.Timestamp('2025-01-15 19:00:00'), 'POSTMARKET'),
            (pd.Timestamp('2025-01-15 20:00:00'), 'CLOSED'),
        ]
        
        for timestamp, expected_session in test_cases:
            simulator.current_timestamp = timestamp
            session = simulator._get_market_session(timestamp)
            assert session == expected_session
            
    def test_intraday_highs_lows(self, simulator):
        """Test intraday high/low calculation."""
        high, low = simulator._get_intraday_highs_lows(simulator.current_timestamp)
        
        # Should find highest high and lowest low from available data
        assert high == simulator.ohlcv_1s['high'].max()
        assert low == simulator.ohlcv_1s['low'].min()
        
    def test_custom_window_config(self, simulator):
        """Test custom window configuration."""
        custom_config = {
            'hf_window_seconds': 30,
            'mf_1m_window_bars': 20,
            'mf_5m_window_bars': 10
        }
        
        state = simulator.get_current_market_state(custom_config)
        
        assert len(state['hf_data_window']) == 30
        assert len(state['1m_bars_window']) == 20
        assert len(state['5m_bars_window']) == 10
        
    def test_quote_size_preservation(self, simulator):
        """Test that quote sizes are preserved and forward filled correctly."""
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        # Check that sizes are integers
        for entry in window:
            for quote in entry['quotes']:
                assert isinstance(quote['bid_size'], int)
                assert isinstance(quote['ask_size'], int)
                
    def test_bar_aggregation_accuracy(self, simulator):
        """Test that bar aggregation is accurate."""
        # Add more dense data for a specific minute
        base_time = simulator.current_timestamp - pd.Timedelta(minutes=1)
        
        # Create 60 1s bars for one minute
        new_times = [base_time + pd.Timedelta(seconds=i) for i in range(60)]
        new_data = []
        for i, ts in enumerate(new_times):
            new_data.append({
                'open': 101 if i == 0 else 101 + i * 0.01,
                'high': 101 + i * 0.01 + 0.005,
                'low': 101 + i * 0.01 - 0.005,
                'close': 101 + i * 0.01,
                'volume': 100
            })
            
        new_df = pd.DataFrame(new_data, index=new_times)
        simulator.ohlcv_1s = pd.concat([simulator.ohlcv_1s, new_df]).sort_index()
        
        # Rebuild index
        simulator.ohlcv_index.clear()
        for idx, ts in enumerate(simulator.ohlcv_1s.index):
            simulator.ohlcv_index[ts] = idx
            
        # Generate 1m bars
        bars = simulator._generate_bar_window(simulator.current_timestamp, 5, '1m')
        
        # Find the bar that should contain our dense minute
        target_bar = None
        for bar in bars:
            if bar['timestamp'] == base_time.floor('1min'):
                target_bar = bar
                break
                
        assert target_bar is not None
        # Check that bar aggregation worked
        assert not target_bar['is_synthetic']  # Should have real data
        # Volume should be sum of all 1s bars in that minute
        # We added 60 bars with 100 volume each
        assert target_bar['volume'] >= 6000  # At least our added data
        
    def test_empty_window_handling(self, simulator):
        """Test handling of completely empty time periods."""
        # Clear all data except one old bar
        old_time = simulator.current_timestamp - pd.Timedelta(hours=2)
        simulator.ohlcv_1s = pd.DataFrame([{
            'open': 90, 'high': 91, 'low': 89, 'close': 90.5, 'volume': 1000
        }], index=[old_time])
        
        simulator.ohlcv_index = {old_time: 0}
        simulator.quotes = pd.DataFrame()
        simulator.trades = pd.DataFrame()
        
        # Generate window
        window = simulator._generate_hf_window(simulator.current_timestamp, 60)
        
        # Should forward fill from the old bar
        for entry in window:
            assert entry['1s_bar']['close'] == 90.5
            assert entry['1s_bar']['is_synthetic']
            assert len(entry['trades']) == 0
            # Should have synthetic quotes
            assert len(entry['quotes']) == 1
            
    def test_boundary_conditions(self, simulator):
        """Test edge cases around time boundaries."""
        # Test at exact market open
        market_open = pd.Timestamp('2025-01-15 09:30:00')
        simulator.current_timestamp = market_open
        
        # Should handle pre-market to regular transition
        state = simulator.get_current_market_state()
        assert state['market_session'] == 'REGULAR'
        
        # Test at day boundary
        day_end = pd.Timestamp('2025-01-15 20:00:00')
        simulator.current_timestamp = day_end
        state = simulator.get_current_market_state()
        assert state['market_session'] == 'CLOSED'