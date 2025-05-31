"""
Comprehensive tests for PortfolioSimulator based on input/output behavior.
Tests all functionality and edge cases without implementation details.
"""

import pytest
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
import numpy as np
from collections import deque

from simulators.portfolio_simulator import (
    PortfolioSimulator, Position, FillDetails, TradeRecord, PortfolioState,
    PortfolioObservation, OrderTypeEnum, OrderSideEnum, PositionSideEnum
)
from config.schemas import EnvConfig, SimulationConfig, ModelConfig


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


@pytest.fixture
def env_config():
    """Create a test environment config."""
    return EnvConfig(
        initial_capital=25000.0,
        symbol="MLGO"
    )


@pytest.fixture
def simulation_config():
    """Create a test simulation config."""
    return SimulationConfig(
        max_position_value_ratio=1.0,
        allow_shorting=False,
        max_position_holding_seconds=3600,  # 1 hour
        initial_cash=25000.0
    )


@pytest.fixture
def model_config():
    """Create a test model config."""
    return ModelConfig(
        portfolio_seq_len=5,
        portfolio_feat_dim=5
    )


@pytest.fixture
def portfolio_simulator(logger, env_config, simulation_config, model_config):
    """Create a portfolio simulator for testing."""
    return PortfolioSimulator(
        logger=logger,
        env_config=env_config,
        simulation_config=simulation_config,
        model_config=model_config,
        tradable_assets=["MLGO"]
    )


@pytest.fixture
def sample_fill_buy():
    """Create a sample buy fill."""
    return FillDetails(
        asset_id="MLGO",
        fill_timestamp=datetime.now(timezone.utc),
        order_type=OrderTypeEnum.MARKET,
        order_side=OrderSideEnum.BUY,
        requested_quantity=100.0,
        executed_quantity=100.0,
        executed_price=100.50,
        commission=5.0,
        fees=1.0,
        slippage_cost_total=2.5
    )


@pytest.fixture
def sample_fill_sell():
    """Create a sample sell fill."""
    return FillDetails(
        asset_id="MLGO",
        fill_timestamp=datetime.now(timezone.utc),
        order_type=OrderTypeEnum.MARKET,
        order_side=OrderSideEnum.SELL,
        requested_quantity=50.0,
        executed_quantity=50.0,
        executed_price=101.00,
        commission=2.5,
        fees=0.5,
        slippage_cost_total=1.25
    )


class TestPortfolioInitialization:
    """Test portfolio initialization and reset functionality."""
    
    def test_initial_state(self, portfolio_simulator):
        """Test initial portfolio state."""
        assert portfolio_simulator.cash == 25000.0
        assert portfolio_simulator.initial_capital == 25000.0
        assert len(portfolio_simulator.positions) == 1
        assert "MLGO" in portfolio_simulator.positions
        assert portfolio_simulator.positions["MLGO"].is_flat()
        assert portfolio_simulator.session_realized_pnl == 0.0
        assert len(portfolio_simulator.open_trades) == 0
        assert len(portfolio_simulator.completed_trades) == 0
    
    def test_reset_functionality(self, portfolio_simulator):
        """Test portfolio reset functionality."""
        # Modify state
        portfolio_simulator.cash = 20000.0
        portfolio_simulator.session_realized_pnl = 1000.0
        portfolio_simulator.session_commission = 100.0
        
        # Reset
        reset_time = datetime.now(timezone.utc)
        portfolio_simulator.reset(reset_time)
        
        # Check reset state
        assert portfolio_simulator.cash == 25000.0
        assert portfolio_simulator.session_realized_pnl == 0.0
        assert portfolio_simulator.session_commission == 0.0
        assert portfolio_simulator.session_start_time == reset_time
        assert len(portfolio_simulator.feature_history) > 0
    
    def test_position_initialization(self, portfolio_simulator):
        """Test that positions are properly initialized."""
        position = portfolio_simulator.positions["MLGO"]
        
        assert position.side == PositionSideEnum.FLAT
        assert position.quantity == 0.0
        assert position.avg_price == 0.0
        assert position.entry_value == 0.0
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.is_flat() is True


class TestFillProcessing:
    """Test fill processing functionality."""
    
    def test_process_buy_fill_opens_position(self, portfolio_simulator, sample_fill_buy):
        """Test that a buy fill opens a long position."""
        initial_cash = portfolio_simulator.cash
        
        enriched_fill = portfolio_simulator.process_fill(sample_fill_buy)
        
        # Check position
        position = portfolio_simulator.positions["MLGO"]
        assert position.side == PositionSideEnum.LONG
        assert position.quantity == 100.0
        assert position.avg_price == 100.50
        assert position.entry_value == 10050.0
        
        # Check cash adjustment
        expected_cash = initial_cash - 10050.0 - 5.0 - 1.0  # trade value - commission - fees
        assert abs(portfolio_simulator.cash - expected_cash) < 0.01
        
        # Check session metrics
        assert portfolio_simulator.session_commission == 5.0
        assert portfolio_simulator.session_fees == 1.0
        assert portfolio_simulator.session_slippage == 2.5
        assert portfolio_simulator.session_volume == 100.0
        assert portfolio_simulator.session_turnover == 10050.0
        
        # Check trade creation
        assert len(portfolio_simulator.open_trades) == 1
        
        # Check enriched fill
        assert enriched_fill.closes_position is False
        assert enriched_fill.realized_pnl is None
    
    def test_process_sell_fill_closes_position(self, portfolio_simulator, sample_fill_buy, sample_fill_sell):
        """Test that a sell fill closes/reduces a position."""
        # First open a position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Now sell part of it
        enriched_fill = portfolio_simulator.process_fill(sample_fill_sell)
        
        # Check position
        position = portfolio_simulator.positions["MLGO"]
        assert position.side == PositionSideEnum.LONG
        assert position.quantity == 50.0  # 100 - 50
        
        # Check realized P&L (101.00 - 100.50) * 50 = 25.0
        expected_pnl = (101.00 - 100.50) * 50.0
        assert abs(portfolio_simulator.session_realized_pnl - expected_pnl) < 0.01
        
        # Check enriched fill
        assert enriched_fill.closes_position is False  # Still has 50 shares
        assert enriched_fill.realized_pnl is not None
        assert abs(enriched_fill.realized_pnl - expected_pnl) < 0.01
        assert enriched_fill.holding_time_minutes is not None
    
    def test_process_sell_all_closes_position_completely(self, portfolio_simulator, sample_fill_buy):
        """Test that selling entire position closes it completely."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Create sell fill for entire position
        sell_all_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=5),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=102.00,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        enriched_fill = portfolio_simulator.process_fill(sell_all_fill)
        
        # Check position is flat
        position = portfolio_simulator.positions["MLGO"]
        assert position.is_flat()
        assert position.quantity == 0.0
        
        # Check realized P&L (102.00 - 100.50) * 100 = 150.0
        expected_pnl = (102.00 - 100.50) * 100.0
        assert abs(portfolio_simulator.session_realized_pnl - expected_pnl) < 0.01
        
        # Check trade completion
        assert len(portfolio_simulator.open_trades) == 0
        assert len(portfolio_simulator.completed_trades) == 1
        
        # Check enriched fill
        assert enriched_fill.closes_position is True
    
    def test_add_to_existing_position(self, portfolio_simulator, sample_fill_buy):
        """Test adding to an existing position updates average price."""
        # Open initial position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Add to position at different price
        add_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=50.0,
            executed_quantity=50.0,
            executed_price=102.00,  # Higher price
            commission=2.5,
            fees=0.5,
            slippage_cost_total=1.25
        )
        
        portfolio_simulator.process_fill(add_fill)
        
        # Check position
        position = portfolio_simulator.positions["MLGO"]
        assert position.quantity == 150.0  # 100 + 50
        
        # Calculate expected weighted average price
        # (100 * 100.50 + 50 * 102.00) / 150 = (10050 + 5100) / 150 = 101.0
        expected_avg_price = (10050.0 + 5100.0) / 150.0
        assert abs(position.avg_price - expected_avg_price) < 0.01
        
        # Trade should still be open and updated
        assert len(portfolio_simulator.open_trades) == 1
        trade = list(portfolio_simulator.open_trades.values())[0]
        assert trade['entry_quantity'] == 150.0
        assert abs(trade['avg_entry_price'] - expected_avg_price) < 0.01
    
    def test_sell_without_position_fails(self, portfolio_simulator):
        """Test that selling without a position fails gracefully."""
        sell_fill = FillDetails(
            asset_id="UNKNOWN",  # Unknown asset
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=100.50,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        # Should handle gracefully without crashing
        result = portfolio_simulator.process_fill(sell_fill)
        assert result is None  # Should return None for unknown assets


class TestMarketValueUpdates:
    """Test market value and unrealized P&L updates."""
    
    def test_update_market_values_long_position(self, portfolio_simulator, sample_fill_buy):
        """Test market value updates for long position."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Update market values
        current_time = datetime.now(timezone.utc)
        market_prices = {"MLGO": 105.00}  # Price went up
        
        portfolio_simulator.update_market_values(market_prices, current_time)
        
        # Check position values
        position = portfolio_simulator.positions["MLGO"]
        assert position.market_value == 100.0 * 105.00  # quantity * current_price
        expected_unrealized_pnl = (105.00 - 100.50) * 100.0  # (current - avg) * quantity
        assert abs(position.unrealized_pnl - expected_unrealized_pnl) < 0.01
        
        # Check equity history updated
        assert len(portfolio_simulator.equity_history) > 1
    
    def test_update_market_values_flat_position(self, portfolio_simulator):
        """Test market value updates with flat position."""
        current_time = datetime.now(timezone.utc)
        market_prices = {"MLGO": 105.00}
        
        portfolio_simulator.update_market_values(market_prices, current_time)
        
        # Check position values remain zero
        position = portfolio_simulator.positions["MLGO"]
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0
    
    def test_peak_equity_and_drawdown_tracking(self, portfolio_simulator, sample_fill_buy):
        """Test peak equity and drawdown tracking."""
        # Open profitable position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Update to profitable price
        current_time = datetime.now(timezone.utc)
        portfolio_simulator.update_market_values({"MLGO": 110.00}, current_time)
        
        # Peak should increase
        total_equity = portfolio_simulator.cash + portfolio_simulator.positions["MLGO"].market_value
        assert portfolio_simulator.peak_equity >= total_equity
        
        # Update to loss
        portfolio_simulator.update_market_values({"MLGO": 90.00}, current_time)
        
        # Drawdown should be recorded
        assert portfolio_simulator.max_drawdown > 0
    
    def test_missing_market_price_logs_warning(self, portfolio_simulator, sample_fill_buy):
        """Test that missing market prices are handled gracefully."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Update with missing price
        current_time = datetime.now(timezone.utc)
        market_prices = {}  # No price for MLGO
        
        # Should not crash
        portfolio_simulator.update_market_values(market_prices, current_time)
        
        # Position values should remain unchanged
        position = portfolio_simulator.positions["MLGO"]
        assert position.market_value == 0.0  # Should remain at default
        assert position.unrealized_pnl == 0.0


class TestFeatureCalculation:
    """Test portfolio feature calculation for model observations."""
    
    def test_flat_position_features(self, portfolio_simulator):
        """Test features with flat position."""
        current_time = datetime.now(timezone.utc)
        features = portfolio_simulator._calculate_portfolio_features(current_time)
        
        assert len(features) == 5  # portfolio_feat_dim
        assert features[0] == 0.0  # position size
        assert features[1] == 0.0  # unrealized PnL
        assert features[2] == 0.0  # time in position
        assert features[3] > 0.0   # cash ratio should be positive
        assert features[4] == 0.0  # session PnL
    
    def test_long_position_features(self, portfolio_simulator, sample_fill_buy):
        """Test features with long position."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Update market value
        current_time = datetime.now(timezone.utc)
        portfolio_simulator.update_market_values({"MLGO": 105.00}, current_time)
        
        features = portfolio_simulator._calculate_portfolio_features(current_time)
        
        assert features[0] > 0.0   # positive position size
        assert features[1] > 0.0   # positive unrealized PnL
        assert features[2] >= 0.0  # time in position
        assert features[3] > 0.0   # cash ratio
    
    def test_feature_normalization_bounds(self, portfolio_simulator, sample_fill_buy):
        """Test that features are properly normalized and bounded."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Update market value to extreme profit
        current_time = datetime.now(timezone.utc)
        portfolio_simulator.update_market_values({"MLGO": 1000.00}, current_time)  # 10x price
        
        features = portfolio_simulator._calculate_portfolio_features(current_time)
        
        # Check bounds
        assert -2.0 <= features[0] <= 2.0   # position size
        assert -2.0 <= features[1] <= 2.0   # unrealized PnL (should be clipped)
        assert 0.0 <= features[2] <= 2.0    # time in position
        assert 0.0 <= features[3] <= 2.0    # cash ratio
        assert -1.0 <= features[4] <= 1.0   # session PnL
    
    def test_portfolio_observation_creation(self, portfolio_simulator):
        """Test portfolio observation creation."""
        obs = portfolio_simulator.get_portfolio_observation()
        
        assert 'features' in obs
        assert obs['features'].shape[0] == 5  # seq_len
        assert obs['features'].shape[1] == 5  # feat_dim
        assert obs['features'].dtype == np.float32
    
    def test_portfolio_observation_with_insufficient_history(self, portfolio_simulator):
        """Test observation creation with insufficient history."""
        # Clear feature history
        portfolio_simulator.feature_history.clear()
        
        obs = portfolio_simulator.get_portfolio_observation()
        
        # Should still work and pad appropriately
        assert obs['features'].shape[0] == 5  # seq_len
        assert obs['features'].shape[1] == 5  # feat_dim


class TestTradeTracking:
    """Test trade tracking and completion."""
    
    def test_trade_creation_on_position_open(self, portfolio_simulator, sample_fill_buy):
        """Test that opening a position creates a trade record."""
        portfolio_simulator.process_fill(sample_fill_buy)
        
        assert len(portfolio_simulator.open_trades) == 1
        
        trade = list(portfolio_simulator.open_trades.values())[0]
        assert trade['asset_id'] == "MLGO"
        assert trade['side'] == PositionSideEnum.LONG
        assert trade['entry_quantity'] == 100.0
        assert trade['avg_entry_price'] == 100.50
        assert trade['realized_pnl'] is None
        assert trade['exit_timestamp'] is None
        assert len(trade['entry_fills']) == 1
        assert len(trade['exit_fills']) == 0
    
    def test_trade_completion_on_position_close(self, portfolio_simulator, sample_fill_buy):
        """Test that closing a position completes the trade record."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Close position
        close_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=30),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=105.00,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        portfolio_simulator.process_fill(close_fill)
        
        # Check trade completion
        assert len(portfolio_simulator.open_trades) == 0
        assert len(portfolio_simulator.completed_trades) == 1
        
        trade = portfolio_simulator.completed_trades[0]
        assert trade['exit_timestamp'] is not None
        assert trade['realized_pnl'] is not None
        assert trade['realized_pnl'] > 0  # Should be profitable
        assert trade['holding_period_seconds'] is not None
        assert trade['holding_period_seconds'] > 0
        assert len(trade['exit_fills']) == 1
    
    def test_partial_trade_closure(self, portfolio_simulator, sample_fill_buy, sample_fill_sell):
        """Test partial trade closure updates trade record."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Partially close
        portfolio_simulator.process_fill(sample_fill_sell)
        
        # Trade should still be open but updated
        assert len(portfolio_simulator.open_trades) == 1
        assert len(portfolio_simulator.completed_trades) == 0
        
        trade = list(portfolio_simulator.open_trades.values())[0]
        assert trade['exit_quantity'] == 50.0  # Partial exit
        assert trade['realized_pnl'] is not None
        assert trade['realized_pnl'] > 0
        assert len(trade['exit_fills']) == 1
    
    def test_trade_callback_on_completion(self, logger, env_config, simulation_config, model_config):
        """Test that trade callback is called on completion."""
        callback_calls = []
        
        def mock_callback(trade):
            callback_calls.append(trade)
        
        portfolio_simulator = PortfolioSimulator(
            logger=logger,
            env_config=env_config,
            simulation_config=simulation_config,
            model_config=model_config,
            tradable_assets=["MLGO"],
            trade_callback=mock_callback
        )
        
        # Open and close position
        buy_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=100.50,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        sell_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=10),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=105.00,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        portfolio_simulator.process_fill(buy_fill)
        portfolio_simulator.process_fill(sell_fill)
        
        # Callback should have been called
        assert len(callback_calls) == 1
        assert callback_calls[0]['asset_id'] == "MLGO"


class TestPortfolioState:
    """Test portfolio state retrieval."""
    
    def test_get_portfolio_state_flat(self, portfolio_simulator):
        """Test portfolio state with flat position."""
        current_time = datetime.now(timezone.utc)
        state = portfolio_simulator.get_portfolio_state(current_time)
        
        assert state['timestamp'] == current_time
        assert state['cash'] == 25000.0
        assert state['total_equity'] == 25000.0
        assert state['unrealized_pnl'] == 0.0
        assert state['realized_pnl_session'] == 0.0
        assert 'MLGO' in state['positions']
        assert state['position_side'] is None
        assert state['position_value'] == 0.0
        assert state['current_drawdown_pct'] == 0.0
    
    def test_get_portfolio_state_with_position(self, portfolio_simulator, sample_fill_buy):
        """Test portfolio state with active position."""
        # Open position
        portfolio_simulator.process_fill(sample_fill_buy)
        
        # Update market values
        current_time = datetime.now(timezone.utc)
        portfolio_simulator.update_market_values({"MLGO": 105.00}, current_time)
        
        state = portfolio_simulator.get_portfolio_state(current_time)
        
        assert state['position_side'] == "LONG"
        assert state['position_value'] > 0.0
        assert state['avg_entry_price'] == 100.50
        assert 'MLGO' in state['positions']
        
        mlgo_position = state['positions']['MLGO']
        assert mlgo_position['quantity'] == 100.0
        assert mlgo_position['current_side'] == PositionSideEnum.LONG
        assert mlgo_position['market_value'] > 0.0
        assert mlgo_position['unrealized_pnl'] > 0.0


class TestTradingMetrics:
    """Test trading performance metrics."""
    
    def test_empty_metrics(self, portfolio_simulator):
        """Test metrics with no completed trades."""
        metrics = portfolio_simulator.get_trading_metrics()
        
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_pnl_per_trade'] == 0.0
        assert metrics['total_realized_pnl'] == 0.0
    
    def test_metrics_with_completed_trades(self, portfolio_simulator):
        """Test metrics calculation with completed trades."""
        # Execute profitable trade
        buy_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=100.00,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        sell_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=15),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=105.00,
            commission=5.0,
            fees=1.0,
            slippage_cost_total=2.5
        )
        
        portfolio_simulator.process_fill(buy_fill)
        portfolio_simulator.process_fill(sell_fill)
        
        metrics = portfolio_simulator.get_trading_metrics()
        
        assert metrics['total_trades'] == 1
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 0
        assert metrics['win_rate'] == 100.0
        assert metrics['avg_pnl_per_trade'] > 0
        assert metrics['total_realized_pnl'] > 0
        assert metrics['avg_holding_time_seconds'] > 0
        assert metrics['profit_factor'] == 0  # No losing trades


class TestUtilityMethods:
    """Test utility and helper methods."""
    
    def test_get_current_position(self, portfolio_simulator, sample_fill_buy):
        """Test getting current position for an asset."""
        # Initially flat
        position = portfolio_simulator.get_current_position("MLGO")
        assert position is not None
        assert position.is_flat()
        
        # After opening position
        portfolio_simulator.process_fill(sample_fill_buy)
        position = portfolio_simulator.get_current_position("MLGO")
        assert not position.is_flat()
        assert position.quantity == 100.0
        
        # Unknown asset
        position = portfolio_simulator.get_current_position("UNKNOWN")
        assert position is None
    
    def test_has_open_positions(self, portfolio_simulator, sample_fill_buy):
        """Test checking for open positions."""
        # Initially no positions
        assert not portfolio_simulator.has_open_positions()
        
        # After opening position
        portfolio_simulator.process_fill(sample_fill_buy)
        assert portfolio_simulator.has_open_positions()
    
    def test_get_open_trade_count(self, portfolio_simulator, sample_fill_buy):
        """Test getting open trade count."""
        # Initially no trades
        assert portfolio_simulator.get_open_trade_count() == 0
        
        # After opening position
        portfolio_simulator.process_fill(sample_fill_buy)
        assert portfolio_simulator.get_open_trade_count() == 1
    
    def test_string_representation(self, portfolio_simulator, sample_fill_buy):
        """Test string representation."""
        repr_str = repr(portfolio_simulator)
        assert "PortfolioSimulator" in repr_str
        assert "equity=" in repr_str
        assert "cash=" in repr_str
        assert "positions=" in repr_str
        assert "trades=" in repr_str


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_quantities(self, portfolio_simulator):
        """Test handling of very small quantities."""
        tiny_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=0.001,
            executed_quantity=0.001,
            executed_price=100.50,
            commission=1.0,
            fees=0.1,
            slippage_cost_total=0.05
        )
        
        portfolio_simulator.process_fill(tiny_fill)
        
        position = portfolio_simulator.positions["MLGO"]
        assert position.quantity == 0.001
        assert not position.is_flat()  # Even tiny positions are tracked
    
    def test_zero_prices(self, portfolio_simulator):
        """Test handling of zero prices."""
        zero_price_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=0.0,  # Zero price
            commission=1.0,
            fees=0.1,
            slippage_cost_total=0.05
        )
        
        portfolio_simulator.process_fill(zero_price_fill)
        
        position = portfolio_simulator.positions["MLGO"]
        assert position.quantity == 100.0
        assert position.avg_price == 0.0
        assert position.entry_value == 0.0
    
    def test_negative_prices(self, portfolio_simulator):
        """Test handling of negative prices."""
        negative_price_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=-10.0,  # Negative price
            commission=1.0,
            fees=0.1,
            slippage_cost_total=0.05
        )
        
        portfolio_simulator.process_fill(negative_price_fill)
        
        position = portfolio_simulator.positions["MLGO"]
        assert position.quantity == 100.0
        assert position.avg_price == -10.0
        # Cash should reflect negative price (gain cash on buy)
    
    def test_extreme_timestamps(self, portfolio_simulator):
        """Test handling of extreme timestamps."""
        future_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime(2030, 1, 1, tzinfo=timezone.utc),  # Far future
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=100.50,
            commission=1.0,
            fees=0.1,
            slippage_cost_total=0.05
        )
        
        portfolio_simulator.process_fill(future_fill)
        
        # Should handle gracefully
        position = portfolio_simulator.positions["MLGO"]
        assert not position.is_flat()
    
    def test_large_numbers(self, portfolio_simulator):
        """Test handling of very large numbers."""
        large_fill = FillDetails(
            asset_id="MLGO",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=1e10,  # 10 billion shares
            executed_quantity=1e10,
            executed_price=1e6,  # $1 million per share
            commission=1e6,
            fees=1e5,
            slippage_cost_total=1e5
        )
        
        portfolio_simulator.process_fill(large_fill)
        
        position = portfolio_simulator.positions["MLGO"]
        assert position.quantity == 1e10
        assert position.avg_price == 1e6
        # Should handle large numbers without overflow
    
    def test_feature_history_maxlen_behavior(self, portfolio_simulator):
        """Test that feature history respects maxlen."""
        # Fill feature history beyond maxlen
        for i in range(20):  # More than maxlen of 5
            current_time = datetime.now(timezone.utc) + timedelta(seconds=i)
            portfolio_simulator.update_market_values({"MLGO": 100.0}, current_time)
        
        # Should only keep last 5
        assert len(portfolio_simulator.feature_history) <= 5
    
    def test_concurrent_timestamp_fills(self, portfolio_simulator):
        """Test fills with identical timestamps."""
        timestamp = datetime.now(timezone.utc)
        
        fill1 = FillDetails(
            asset_id="MLGO",
            fill_timestamp=timestamp,
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=100.00,
            commission=1.0,
            fees=0.1,
            slippage_cost_total=0.05
        )
        
        fill2 = FillDetails(
            asset_id="MLGO",
            fill_timestamp=timestamp,  # Same timestamp
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=50.0,
            executed_quantity=50.0,
            executed_price=101.00,
            commission=1.0,
            fees=0.1,
            slippage_cost_total=0.05
        )
        
        portfolio_simulator.process_fill(fill1)
        portfolio_simulator.process_fill(fill2)
        
        # Should handle gracefully and calculate correct averages
        position = portfolio_simulator.positions["MLGO"]
        assert position.quantity == 150.0
        expected_avg = (100.0 * 100.00 + 50.0 * 101.00) / 150.0
        assert abs(position.avg_price - expected_avg) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])