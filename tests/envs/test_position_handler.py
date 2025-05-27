import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any

from envs.position_handler import (
    PositionHandler,
    PositionState,
    PositionAction,
    PositionResult,
    PositionInfo,
    ForceExitReason,
    PositionHandlerConfig
)


class TestPositionHandler:
    """Test suite for position handling at episode boundaries."""
    
    @pytest.fixture
    def handler_config(self):
        """Configuration for position handler."""
        return PositionHandlerConfig(
            force_exit_at_market_close=True,
            force_exit_at_max_loss=True,
            max_position_duration=18000,  # 5 hours
            allow_position_inheritance=True,
            exit_slippage_bps=10,  # 0.1%
            market_close_buffer_seconds=300,  # 5 minutes before close
            log_position_transitions=True
        )
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def position_handler(self, handler_config, mock_logger):
        """Create position handler instance."""
        return PositionHandler(handler_config, mock_logger)
    
    @pytest.fixture
    def sample_portfolio_state(self):
        """Sample portfolio state with position."""
        return {
            'cash': 95000,
            'positions': {
                'MLGO': {
                    'symbol': 'MLGO',
                    'quantity': 500,
                    'side': 'long',
                    'avg_price': 10.0,
                    'current_price': 10.2,
                    'entry_time': datetime(2025, 1, 15, 9, 35),
                    'unrealized_pnl': 100,
                    'realized_pnl': 0,
                    'max_price': 10.3,
                    'min_price': 9.95,
                    'volume_traded': 500
                }
            },
            'total_value': 100100,
            'unrealized_pnl': 100,
            'realized_pnl': 0
        }
    
    @pytest.fixture
    def sample_market_state(self):
        """Sample market state for testing."""
        return {
            'timestamp': datetime(2025, 1, 15, 14, 0),
            'bid': 10.18,
            'ask': 10.20,
            'last': 10.19,
            'bid_size': 1000,
            'ask_size': 1200,
            'volume': 1500000,
            'spread_bps': 10
        }
    
    def test_position_handler_initialization(self, position_handler, handler_config):
        """Test position handler initialization."""
        assert position_handler.config == handler_config
        assert position_handler.position_history == []
        assert position_handler.force_exit_stats == {
            'market_close': 0,
            'max_loss': 0,
            'bankruptcy': 0,
            'max_duration': 0
        }
    
    def test_handle_episode_end_no_position(self, position_handler):
        """Test handling episode end with no position."""
        portfolio_state = {
            'cash': 100000,
            'positions': {},
            'total_value': 100000
        }
        
        result = position_handler.handle_episode_end(
            portfolio_state=portfolio_state,
            termination_reason="MAX_DURATION",
            market_state={'bid': 10.0, 'ask': 10.01}
        )
        
        assert result['had_position'] is False
        assert result['forced_exit'] is False
        assert 'exit_details' not in result
    
    def test_handle_market_close_termination(self, position_handler, sample_portfolio_state, sample_market_state):
        """Test forced position closure at market close."""
        # Set market state to closing time
        market_state = sample_market_state.copy()
        market_state['timestamp'] = datetime(2025, 1, 15, 20, 0)
        
        result = position_handler.handle_episode_end(
            portfolio_state=sample_portfolio_state,
            termination_reason="MARKET_CLOSE",
            market_state=market_state
        )
        
        assert result['had_position'] is True
        assert result['forced_exit'] is True
        assert result['exit_reason'] == ForceExitReason.MARKET_CLOSE
        assert result['exit_price'] == market_state['bid']  # Exit at bid
        assert result['realized_pnl'] == 500 * (10.18 - 10.0)  # 500 shares * price diff
        assert position_handler.force_exit_stats['market_close'] == 1
    
    def test_handle_max_loss_termination(self, position_handler, sample_market_state):
        """Test forced position closure due to max loss."""
        # Create portfolio with losing position
        portfolio_state = {
            'cash': 95000,
            'positions': {
                'MLGO': {
                    'symbol': 'MLGO',
                    'quantity': 500,
                    'side': 'long',
                    'avg_price': 10.5,
                    'current_price': 10.0,
                    'entry_time': datetime(2025, 1, 15, 9, 35),
                    'unrealized_pnl': -250,
                }
            },
            'total_value': 95000
        }
        
        # Market state with lower bid
        market_state = sample_market_state.copy()
        market_state['bid'] = 9.95
        market_state['ask'] = 9.97
        
        result = position_handler.handle_episode_end(
            portfolio_state=portfolio_state,
            termination_reason="MAX_LOSS",
            market_state=market_state
        )
        
        assert result['had_position'] is True
        assert result['forced_exit'] is True
        assert result['exit_reason'] == ForceExitReason.MAX_LOSS
        assert result['exit_price'] == 9.95
        assert result['realized_pnl'] == 500 * (9.95 - 10.5)
        assert result['exit_slippage'] > 0
        assert position_handler.force_exit_stats['max_loss'] == 1
    
    def test_handle_max_duration_continuation(self, position_handler, sample_portfolio_state, sample_market_state):
        """Test position continuation when episode ends due to max duration."""
        result = position_handler.handle_episode_end(
            portfolio_state=sample_portfolio_state,
            termination_reason="MAX_DURATION",
            market_state=sample_market_state
        )
        
        assert result['had_position'] is True
        assert result['forced_exit'] is False  # Not forced to exit
        assert result['position_continues'] is True
        assert result['unrealized_pnl'] == 100
        assert result['hold_duration_seconds'] > 0
        
        # Position info for next episode
        assert 'continuation_info' in result
        assert result['continuation_info']['entry_price'] == 10.0
        assert result['continuation_info']['entry_time'] == datetime(2025, 1, 15, 9, 35)
    
    def test_handle_short_position(self, position_handler, sample_market_state):
        """Test handling of short positions."""
        portfolio_state = {
            'cash': 105000,
            'positions': {
                'MLGO': {
                    'symbol': 'MLGO',
                    'quantity': -500,  # Short position
                    'side': 'short',
                    'avg_price': 10.2,
                    'current_price': 10.0,
                    'entry_time': datetime(2025, 1, 15, 10, 0),
                    'unrealized_pnl': 100,  # Profit on short
                }
            },
            'total_value': 100100
        }
        
        result = position_handler.handle_episode_end(
            portfolio_state=portfolio_state,
            termination_reason="MARKET_CLOSE",
            market_state=sample_market_state
        )
        
        assert result['had_position'] is True
        assert result['forced_exit'] is True
        assert result['exit_price'] == sample_market_state['ask']  # Exit short at ask
        assert result['realized_pnl'] == 500 * (10.2 - 10.20)  # Small loss due to spread
    
    def test_calculate_exit_metrics(self, position_handler):
        """Test exit metrics calculation."""
        position = PositionInfo(
            symbol='MLGO',
            quantity=1000,
            side='long',
            avg_price=10.0,
            entry_time=datetime(2025, 1, 15, 9, 30),
            max_price=10.5,
            min_price=9.8,
            volume_traded=1000
        )
        
        exit_time = datetime(2025, 1, 15, 10, 30)
        exit_price = 10.3
        
        metrics = position_handler._calculate_exit_metrics(
            position, exit_price, exit_time
        )
        
        assert metrics['hold_duration_seconds'] == 3600
        assert metrics['realized_pnl'] == 300  # 1000 * (10.3 - 10.0)
        assert metrics['pnl_percent'] == 0.03  # 3%
        assert metrics['max_profit'] == 500  # 1000 * (10.5 - 10.0)
        assert metrics['max_loss'] == -200  # 1000 * (9.8 - 10.0)
        assert metrics['efficiency'] == 0.6  # 300/500
    
    def test_apply_slippage(self, position_handler):
        """Test slippage application on forced exits."""
        # Long position slippage
        bid_price = 10.0
        slippage_bps = 10
        
        exit_price = position_handler._apply_slippage(
            bid_price, 'long', slippage_bps
        )
        
        assert exit_price == 10.0 * (1 - 0.001)  # 10 bps lower
        
        # Short position slippage
        ask_price = 10.0
        exit_price = position_handler._apply_slippage(
            ask_price, 'short', slippage_bps
        )
        
        assert exit_price == 10.0 * (1 + 0.001)  # 10 bps higher
    
    def test_position_history_tracking(self, position_handler, sample_portfolio_state, sample_market_state):
        """Test position history tracking."""
        # Handle multiple position exits
        for i in range(3):
            result = position_handler.handle_episode_end(
                portfolio_state=sample_portfolio_state,
                termination_reason="MARKET_CLOSE",
                market_state=sample_market_state
            )
        
        assert len(position_handler.position_history) == 3
        
        # Check history entry
        history_entry = position_handler.position_history[0]
        assert 'timestamp' in history_entry
        assert 'position_info' in history_entry
        assert 'exit_details' in history_entry
        assert history_entry['termination_reason'] == "MARKET_CLOSE"
    
    def test_multiple_positions_handling(self, position_handler, sample_market_state):
        """Test handling multiple positions (if supported)."""
        portfolio_state = {
            'cash': 80000,
            'positions': {
                'MLGO': {
                    'symbol': 'MLGO',
                    'quantity': 500,
                    'side': 'long',
                    'avg_price': 10.0,
                    'current_price': 10.2,
                    'unrealized_pnl': 100
                },
                'NVDA': {
                    'symbol': 'NVDA',
                    'quantity': 100,
                    'side': 'long',
                    'avg_price': 100.0,
                    'current_price': 102.0,
                    'unrealized_pnl': 200
                }
            },
            'total_value': 100300
        }
        
        result = position_handler.handle_episode_end(
            portfolio_state=portfolio_state,
            termination_reason="MARKET_CLOSE",
            market_state=sample_market_state
        )
        
        assert result['had_position'] is True
        assert len(result['positions_closed']) == 2
        assert result['total_realized_pnl'] > 0
    
    def test_edge_case_zero_quantity(self, position_handler, sample_market_state):
        """Test handling edge case of zero quantity position."""
        portfolio_state = {
            'cash': 100000,
            'positions': {
                'MLGO': {
                    'symbol': 'MLGO',
                    'quantity': 0,  # Flat position
                    'side': 'flat',
                    'avg_price': 0,
                    'current_price': 10.0,
                    'unrealized_pnl': 0
                }
            },
            'total_value': 100000
        }
        
        result = position_handler.handle_episode_end(
            portfolio_state=portfolio_state,
            termination_reason="MAX_DURATION",
            market_state=sample_market_state
        )
        
        assert result['had_position'] is False
    
    def test_bankruptcy_handling(self, position_handler, sample_market_state):
        """Test bankruptcy position handling."""
        portfolio_state = {
            'cash': 100,
            'positions': {
                'MLGO': {
                    'symbol': 'MLGO',
                    'quantity': 10,
                    'side': 'long',
                    'avg_price': 100.0,
                    'current_price': 10.0,
                    'unrealized_pnl': -900
                }
            },
            'total_value': 200  # Nearly bankrupt
        }
        
        result = position_handler.handle_episode_end(
            portfolio_state=portfolio_state,
            termination_reason="BANKRUPTCY",
            market_state=sample_market_state
        )
        
        assert result['had_position'] is True
        assert result['forced_exit'] is True
        assert result['exit_reason'] == ForceExitReason.BANKRUPTCY
        assert position_handler.force_exit_stats['bankruptcy'] == 1
    
    def test_position_age_based_exit(self, position_handler, sample_portfolio_state, sample_market_state):
        """Test position exit based on age/duration."""
        # Make position very old
        sample_portfolio_state['positions']['MLGO']['entry_time'] = \
            datetime(2025, 1, 15, 4, 0)  # 10 hours old
        
        # Current time is 14:00
        position_age = (sample_market_state['timestamp'] - 
                       sample_portfolio_state['positions']['MLGO']['entry_time']).seconds
        
        result = position_handler.handle_episode_end(
            portfolio_state=sample_portfolio_state,
            termination_reason="MAX_DURATION",
            market_state=sample_market_state
        )
        
        # Check if position age exceeds max duration
        if position_age > position_handler.config.max_position_duration:
            assert result['forced_exit'] is True
            assert result['exit_reason'] == ForceExitReason.MAX_POSITION_AGE
            assert position_handler.force_exit_stats['max_duration'] == 1
    
    def test_market_close_buffer(self, position_handler, sample_portfolio_state):
        """Test market close buffer handling."""
        # Time is 5 minutes before market close
        market_state = {
            'timestamp': datetime(2025, 1, 15, 19, 55),
            'bid': 10.15,
            'ask': 10.17,
            'last': 10.16
        }
        
        result = position_handler.handle_episode_end(
            portfolio_state=sample_portfolio_state,
            termination_reason="MAX_DURATION",  # Not market close termination
            market_state=market_state
        )
        
        # Should still force exit due to buffer
        assert result['forced_exit'] is True
        assert result['exit_reason'] == ForceExitReason.MARKET_CLOSE_BUFFER
    
    def test_position_continuity_info(self, position_handler, sample_portfolio_state, sample_market_state):
        """Test position continuity information for next episode."""
        result = position_handler.handle_episode_end(
            portfolio_state=sample_portfolio_state,
            termination_reason="MAX_DURATION",
            market_state=sample_market_state
        )
        
        continuation_info = result['continuation_info']
        
        assert continuation_info['symbol'] == 'MLGO'
        assert continuation_info['quantity'] == 500
        assert continuation_info['side'] == 'long'
        assert continuation_info['entry_price'] == 10.0
        assert continuation_info['entry_time'] == datetime(2025, 1, 15, 9, 35)
        assert continuation_info['current_price'] == 10.2
        assert continuation_info['unrealized_pnl'] == 100
        assert continuation_info['age_seconds'] > 0
    
    def test_logging_and_stats(self, position_handler, sample_portfolio_state, sample_market_state, mock_logger):
        """Test logging and statistics tracking."""
        # Force multiple exits
        for reason in ["MARKET_CLOSE", "MAX_LOSS", "BANKRUPTCY"]:
            position_handler.handle_episode_end(
                portfolio_state=sample_portfolio_state,
                termination_reason=reason,
                market_state=sample_market_state
            )
        
        # Check stats
        assert position_handler.force_exit_stats['market_close'] == 1
        assert position_handler.force_exit_stats['max_loss'] == 1
        assert position_handler.force_exit_stats['bankruptcy'] == 1
        
        # Check logging was called
        assert mock_logger.info.called
        assert mock_logger.warning.called  # For forced exits
    
    def test_get_position_summary(self, position_handler, sample_portfolio_state, sample_market_state):
        """Test position summary generation."""
        # Handle some positions
        for _ in range(5):
            position_handler.handle_episode_end(
                portfolio_state=sample_portfolio_state,
                termination_reason="MARKET_CLOSE",
                market_state=sample_market_state
            )
        
        summary = position_handler.get_position_summary()
        
        assert summary['total_positions_handled'] == 5
        assert summary['forced_exits']['total'] == 5
        assert summary['forced_exits']['market_close'] == 5
        assert 'avg_hold_duration' in summary
        assert 'total_realized_pnl' in summary
        assert 'position_history_length' in summary


class TestPositionDataStructures:
    """Test position-related data structures."""
    
    def test_position_info_creation(self):
        """Test PositionInfo data class."""
        position = PositionInfo(
            symbol='MLGO',
            quantity=500,
            side='long',
            avg_price=10.0,
            entry_time=datetime(2025, 1, 15, 9, 30),
            max_price=10.5,
            min_price=9.8,
            volume_traded=500,
            fees_paid=5.0
        )
        
        assert position.symbol == 'MLGO'
        assert position.quantity == 500
        assert position.side == 'long'
        assert position.is_long is True
        assert position.is_short is False
        assert position.is_flat is False
    
    def test_position_state_enum(self):
        """Test PositionState enum."""
        assert PositionState.FLAT.value == "flat"
        assert PositionState.LONG.value == "long"
        assert PositionState.SHORT.value == "short"
        
    def test_position_action_enum(self):
        """Test PositionAction enum."""
        actions = [
            PositionAction.HOLD,
            PositionAction.FORCE_CLOSE,
            PositionAction.CONTINUE
        ]
        
        for action in actions:
            assert isinstance(action.value, str)
    
    def test_force_exit_reason_enum(self):
        """Test ForceExitReason enum."""
        reasons = [
            ForceExitReason.MARKET_CLOSE,
            ForceExitReason.MAX_LOSS,
            ForceExitReason.BANKRUPTCY,
            ForceExitReason.MAX_POSITION_AGE,
            ForceExitReason.MARKET_CLOSE_BUFFER,
            ForceExitReason.INVALID_STATE
        ]
        
        for reason in reasons:
            assert isinstance(reason.value, str)
    
    def test_position_result_structure(self):
        """Test PositionResult structure."""
        result = PositionResult(
            had_position=True,
            forced_exit=True,
            exit_reason=ForceExitReason.MARKET_CLOSE,
            exit_price=10.18,
            realized_pnl=90,
            exit_slippage=0.001,
            hold_duration_seconds=3600,
            position_continues=False
        )
        
        assert result.had_position is True
        assert result.forced_exit is True
        assert result.exit_reason == ForceExitReason.MARKET_CLOSE
        assert result.realized_pnl == 90