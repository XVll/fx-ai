"""
Tests for TradingEnvironment.setup_session method.

Covers all essential functionality including validation, error handling, and state management.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd
import logging
import numpy as np

# Add v2 to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock the config import to avoid dependency issues
class MockConfig:
    def __init__(self):
        self.model = Mock()
        self.model.hf_seq_len = 60
        self.model.hf_feat_dim = 10
        self.model.mf_seq_len = 20
        self.model.mf_feat_dim = 15
        self.model.lf_seq_len = 10
        self.model.lf_feat_dim = 20
        self.model.portfolio_seq_len = 5
        self.model.portfolio_feat_dim = 8

with patch.dict('sys.modules', {
    'v2.config': Mock(Config=MockConfig),
    'v2.config.Config': MockConfig
}):
    from v2.envs.trading_environment import TradingEnvironment

# Import mocks from the correct path
sys.path.insert(0, str(Path(__file__).parent))
from mocks import (
    MockMarketSimulator,
    MockPortfolioSimulator,
    MockExecutionSimulator,
    MockRewardCalculator,
    MockActionMask,
    MockDataManager,
)


class TestSetupSession:
    """Test suite for TradingEnvironment.setup_session method."""

    def setup_method(self):
        """Setup test fixtures before each test."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.model = Mock()
        self.mock_config.model.hf_seq_len = 60
        self.mock_config.model.hf_feat_dim = 10
        self.mock_config.model.mf_seq_len = 20
        self.mock_config.model.mf_feat_dim = 15
        self.mock_config.model.lf_seq_len = 10
        self.mock_config.model.lf_feat_dim = 20
        self.mock_config.model.portfolio_seq_len = 5
        self.mock_config.model.portfolio_feat_dim = 8

        # Create mock components
        self.market_sim = MockMarketSimulator()
        self.portfolio_sim = MockPortfolioSimulator()
        self.execution_sim = MockExecutionSimulator()
        self.reward_calc = MockRewardCalculator()
        self.action_mask = MockActionMask()
        self.data_manager = MockDataManager()

        # Create mock logger to verify log calls
        self.mock_logger = Mock(spec=logging.Logger)

        # Create TradingEnvironment instance with REAL implementation
        self.env = TradingEnvironment(
            config=self.mock_config,
            market_simulator=self.market_sim,
            portfolio_simulator=self.portfolio_sim,
            execution_simulator=self.execution_sim,
            reward_calculator=self.reward_calc,
            action_mask=self.action_mask,
            data_manager=self.data_manager,
            logger=self.mock_logger,
        )

    def test_setup_session_valid_string_date(self):
        """Test setup_session with valid symbol and string date."""
        symbol = "AAPL"
        date_str = "2024-01-15"
        expected_date = datetime(2024, 1, 15)

        # Mock data manager to return reset points
        mock_reset_points = [
            {"timestamp": expected_date, "activity_score": 0.8},
            {"timestamp": expected_date, "activity_score": 0.6},
        ]
        self.data_manager.get_reset_points = Mock(return_value=mock_reset_points)

        # Execute
        self.env.setup_session(symbol, date_str)

        # Verify state
        assert self.env.current_symbol == symbol
        assert self.env.current_date == expected_date
        assert self.env.reset_points == mock_reset_points
        assert self.env.current_reset_idx == 0

        # Verify market simulator initialization
        assert self.market_sim.is_initialized
        assert self.market_sim.current_time.date() == expected_date.date()

        # Verify data manager was called correctly
        self.data_manager.get_reset_points.assert_called_once_with(symbol, expected_date)

        # Verify logging
        assert self.mock_logger.info.call_count >= 2

    def test_setup_session_valid_datetime_date(self):
        """Test setup_session with valid symbol and datetime object."""
        symbol = "TSLA"
        date_obj = datetime(2024, 2, 20, 10, 30, 45)
        expected_date = date_obj

        # Mock data manager
        mock_reset_points = [{"timestamp": expected_date, "activity_score": 0.7}]
        self.data_manager.get_reset_points = Mock(return_value=mock_reset_points)

        # Execute
        self.env.setup_session(symbol, date_obj)

        # Verify state
        assert self.env.current_symbol == symbol
        assert self.env.current_date == expected_date
        assert self.env.reset_points == mock_reset_points

        # Verify market simulator called with exact datetime
        assert self.market_sim.is_initialized
        assert self.market_sim.current_time == datetime(2024, 2, 20, 9, 30, 0)  # Mock resets to 9:30

    def test_setup_session_invalid_symbol_none(self):
        """Test setup_session raises ValueError for None symbol."""
        with pytest.raises(ValueError, match="A valid symbol \\(string\\) must be provided"):
            self.env.setup_session(None, "2024-01-15")

    def test_setup_session_invalid_symbol_empty_string(self):
        """Test setup_session raises ValueError for empty string symbol."""
        with pytest.raises(ValueError, match="A valid symbol \\(string\\) must be provided"):
            self.env.setup_session("", "2024-01-15")

    def test_setup_session_invalid_symbol_non_string(self):
        """Test setup_session raises ValueError for non-string symbol."""
        with pytest.raises(ValueError, match="A valid symbol \\(string\\) must be provided"):
            self.env.setup_session(123, "2024-01-15")

        with pytest.raises(ValueError, match="A valid symbol \\(string\\) must be provided"):
            self.env.setup_session(["AAPL"], "2024-01-15")

    def test_setup_session_invalid_date_string(self):
        """Test setup_session with invalid date string."""
        symbol = "AAPL"
        invalid_date = "invalid-date"

        # This should raise an exception during date parsing
        with pytest.raises((ValueError, pd.errors.ParserError)):
            self.env.setup_session(symbol, invalid_date)

    def test_setup_session_market_simulator_initialization_failure(self):
        """Test setup_session when market simulator fails to initialize."""
        symbol = "AAPL"
        date_str = "2024-01-15"

        # Make market simulator initialization fail
        self.market_sim.initialize_day = Mock(return_value=False)

        # Execute and expect ValueError
        with pytest.raises(ValueError, match="Failed to initialize market simulator"):
            self.env.setup_session(symbol, date_str)

        # Verify market simulator was called
        expected_date = pd.Timestamp(date_str).to_pydatetime()
        self.market_sim.initialize_day.assert_called_once_with(expected_date)

    def test_setup_session_empty_reset_points(self):
        """Test setup_session with empty reset points list."""
        symbol = "AAPL"
        date_str = "2024-01-15"

        # Mock data manager to return empty list  
        self.data_manager.get_reset_points = Mock(return_value=[])

        # Execute (should not raise exception)
        self.env.setup_session(symbol, date_str)

        # Verify state
        assert self.env.current_symbol == symbol
        assert self.env.reset_points == []
        assert self.env.current_reset_idx == 0

        # Verify logging mentions 0 reset points
        log_calls = [call.args[0] for call in self.mock_logger.info.call_args_list]
        assert any("0 reset points available" in call for call in log_calls)

    def test_setup_session_many_reset_points(self):
        """Test setup_session with many reset points."""
        symbol = "NVDA"
        date_str = "2024-01-15"
        expected_date = pd.Timestamp(date_str).to_pydatetime()

        # Create many reset points
        mock_reset_points = []
        for i in range(50):
            mock_reset_points.append({
                "timestamp": expected_date,
                "activity_score": 0.5 + (i * 0.01),
                "reset_type": "momentum"
            })

        self.data_manager.get_reset_points = Mock(return_value=mock_reset_points)

        # Execute
        self.env.setup_session(symbol, date_str)

        # Verify all reset points loaded
        assert len(self.env.reset_points) == 50
        assert self.env.reset_points == mock_reset_points

        # Verify logging mentions correct count
        log_calls = [call.args[0] for call in self.mock_logger.info.call_args_list]
        assert any("50 reset points available" in call for call in log_calls)

    def test_setup_session_data_manager_exception(self):
        """Test setup_session when data manager raises exception."""
        symbol = "AAPL"
        date_str = "2024-01-15"

        # Mock data manager to raise exception
        self.data_manager.get_reset_points = Mock(side_effect=Exception("Data loading failed"))

        # Execute and expect exception to propagate
        with pytest.raises(Exception, match="Data loading failed"):
            self.env.setup_session(symbol, date_str)

    def test_setup_session_multiple_calls_overwrites_state(self):
        """Test that multiple setup_session calls overwrite previous state."""
        # First setup
        symbol1 = "AAPL"
        date1 = "2024-01-15"
        reset_points1 = [{"timestamp": pd.Timestamp(date1).to_pydatetime(), "activity_score": 0.8}]
        
        self.data_manager.get_reset_points = Mock(return_value=reset_points1)
        self.env.setup_session(symbol1, date1)
        
        # Verify first setup
        assert self.env.current_symbol == symbol1
        assert self.env.current_date == pd.Timestamp(date1).to_pydatetime()
        assert self.env.reset_points == reset_points1

        # Second setup with different data
        symbol2 = "TSLA"
        date2 = "2024-02-20"
        reset_points2 = [
            {"timestamp": pd.Timestamp(date2).to_pydatetime(), "activity_score": 0.6},
            {"timestamp": pd.Timestamp(date2).to_pydatetime(), "activity_score": 0.9}
        ]
        
        self.data_manager.get_reset_points = Mock(return_value=reset_points2)
        self.env.setup_session(symbol2, date2)

        # Verify second setup overwrote first
        assert self.env.current_symbol == symbol2
        assert self.env.current_date == pd.Timestamp(date2).to_pydatetime()
        assert self.env.reset_points == reset_points2
        assert self.env.current_reset_idx == 0  # Should reset to 0

    @pytest.mark.parametrize("invalid_symbol", [
        None, "", "   ", 123, [], {}, 0.5, True,
    ])
    def test_setup_session_invalid_symbols(self, invalid_symbol):
        """Test various invalid symbol types."""
        with pytest.raises(ValueError, match="A valid symbol \\(string\\) must be provided"):
            self.env.setup_session(invalid_symbol, "2024-01-15")