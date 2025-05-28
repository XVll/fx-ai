"""
Comprehensive tests for PortfolioSimulator.

Tests are designed based on expected input/output behavior and edge cases,
without looking at implementation details.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import numpy as np
from unittest.mock import Mock, MagicMock

from simulators.portfolio_simulator import (
    PortfolioSimulator,
    FillDetails,
    OrderSideEnum,
    OrderTypeEnum,
    PositionSideEnum,
    Position,
    PortfolioState,
    TradeRecord
)
from config.schemas import EnvConfig, SimulationConfig, ModelConfig


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def env_config():
    """Create test environment configuration."""
    return EnvConfig(
        initial_capital=10000.0,
        max_steps=1000,
        invalid_action_limit=10
    )


@pytest.fixture
def simulation_config():
    """Create test simulation configuration."""
    return SimulationConfig(
        max_position_value_ratio=0.95,
        allow_shorting=True,
        max_position_holding_seconds=3600
    )


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        portfolio_seq_len=10,
        portfolio_feat_dim=5,
        hf_seq_len=60,
        hf_feat_dim=20,
        mf_seq_len=30,
        mf_feat_dim=15,
        lf_seq_len=10,
        lf_feat_dim=10,
        static_feat_dim=8,
        action_dim=[3, 4],
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )


@pytest.fixture
def portfolio_simulator(mock_logger, env_config, simulation_config, model_config):
    """Create a PortfolioSimulator instance for testing."""
    tradable_assets = ["AAPL", "GOOGL", "MSFT"]
    return PortfolioSimulator(
        logger=mock_logger,
        env_config=env_config,
        simulation_config=simulation_config,
        model_config=model_config,
        tradable_assets=tradable_assets,
        trade_callback=None
    )


@pytest.fixture
def sample_fill():
    """Create a sample fill for testing."""
    return FillDetails(
        asset_id="AAPL",
        fill_timestamp=datetime.now(timezone.utc),
        order_type=OrderTypeEnum.MARKET,
        order_side=OrderSideEnum.BUY,
        requested_quantity=100.0,
        executed_quantity=100.0,
        executed_price=150.0,
        commission=0.5,
        fees=0.0,
        slippage_cost_total=0.1,
        closes_position=False,
        realized_pnl=None,
        holding_time_minutes=None,
        price=150.0,
        quantity=100.0,
        slippage_cost=0.1
    )


class TestPortfolioSimulatorInitialization:
    """Test portfolio simulator initialization and configuration."""
    
    def test_initialization_with_default_values(self, mock_logger, env_config, simulation_config, model_config):
        """Test that portfolio simulator initializes with correct default values."""
        tradable_assets = ["AAPL", "GOOGL"]
        simulator = PortfolioSimulator(
            logger=mock_logger,
            env_config=env_config,
            simulation_config=simulation_config,
            model_config=model_config,
            tradable_assets=tradable_assets,
            trade_callback=None
        )
        
        assert simulator.initial_capital == 10000.0
        assert simulator.cash == 10000.0
        assert simulator.tradable_assets == tradable_assets
        assert len(simulator.positions) == 2
        assert all(pos.is_flat() for pos in simulator.positions.values())
        assert simulator.session_realized_pnl == 0.0
        assert simulator.peak_equity == 10000.0
        
    def test_initialization_with_trade_callback(self, mock_logger, env_config, simulation_config, model_config):
        """Test initialization with a trade callback function."""
        callback = Mock()
        simulator = PortfolioSimulator(
            logger=mock_logger,
            env_config=env_config,
            simulation_config=simulation_config,
            model_config=model_config,
            tradable_assets=["AAPL"],
            trade_callback=callback
        )
        
        assert simulator.trade_callback == callback
        
    def test_reset_functionality(self, portfolio_simulator):
        """Test that reset properly resets all state."""
        # Modify some state
        portfolio_simulator.cash = 5000.0
        portfolio_simulator.session_realized_pnl = 1000.0
        portfolio_simulator.session_commission = 50.0
        
        # Reset
        reset_time = datetime.now(timezone.utc)
        portfolio_simulator.reset(reset_time)
        
        # Verify reset
        assert portfolio_simulator.cash == 10000.0
        assert portfolio_simulator.session_realized_pnl == 0.0
        assert portfolio_simulator.session_commission == 0.0
        assert portfolio_simulator.session_start_time == reset_time
        assert all(pos.is_flat() for pos in portfolio_simulator.positions.values())
        assert len(portfolio_simulator.open_trades) == 0
        assert len(portfolio_simulator.completed_trades) == 0


class TestPositionManagement:
    """Test position opening, closing, and updating."""
    
    def test_open_long_position(self, portfolio_simulator):
        """Test opening a new long position."""
        fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        initial_cash = portfolio_simulator.cash
        enriched_fill = portfolio_simulator.process_fill(fill)
        
        # Check position
        position = portfolio_simulator.positions["AAPL"]
        assert position.side == PositionSideEnum.LONG
        assert position.quantity == 100.0
        assert position.avg_price == 150.0
        assert position.entry_value == 15000.0
        
        # Check cash
        expected_cash = initial_cash - 15000.0 - 0.5 - 0.0  # price*qty - commission - fees
        assert portfolio_simulator.cash == expected_cash
        
        # Check open trades
        assert len(portfolio_simulator.open_trades) == 1
        
        # Check enriched fill
        assert enriched_fill['closes_position'] == False
        
    def test_open_short_position(self, portfolio_simulator):
        """Test opening a new short position."""
        fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        initial_cash = portfolio_simulator.cash
        portfolio_simulator.process_fill(fill)
        
        # Check position
        position = portfolio_simulator.positions["AAPL"]
        assert position.side == PositionSideEnum.SHORT
        assert position.quantity == 100.0
        assert position.avg_price == 150.0
        
        # Check cash (should increase for short sale)
        expected_cash = initial_cash + 15000.0 - 0.5 - 0.0
        assert portfolio_simulator.cash == expected_cash
        
    def test_add_to_long_position(self, portfolio_simulator):
        """Test adding to an existing long position (averaging up/down)."""
        # Open initial position
        fill1 = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(fill1)
        
        # Add to position at different price
        fill2 = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=160.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=160.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(fill2)
        
        # Check updated position
        position = portfolio_simulator.positions["AAPL"]
        assert position.side == PositionSideEnum.LONG
        assert position.quantity == 200.0
        expected_avg_price = (100 * 150 + 100 * 160) / 200  # 155.0
        assert position.avg_price == expected_avg_price
        
    def test_close_long_position_with_profit(self, portfolio_simulator):
        """Test closing a long position with profit."""
        # Open position
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Close position at higher price
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=30),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=155.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=True,
            realized_pnl=None,
            holding_time_minutes=None,
            price=155.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        enriched_fill = portfolio_simulator.process_fill(sell_fill)
        
        # Check position is flat
        position = portfolio_simulator.positions["AAPL"]
        assert position.is_flat()
        assert position.quantity == 0.0
        
        # Check realized P&L
        expected_pnl = (155.0 - 150.0) * 100.0  # $500 profit
        assert portfolio_simulator.session_realized_pnl == expected_pnl
        
        # Check trade is completed
        assert len(portfolio_simulator.open_trades) == 0
        assert len(portfolio_simulator.completed_trades) == 1
        
        # Check enriched fill
        assert enriched_fill['closes_position'] == True
        assert enriched_fill['realized_pnl'] == expected_pnl
        assert enriched_fill['holding_time_minutes'] == pytest.approx(30.0, rel=0.1)
        
    def test_close_short_position_with_loss(self, portfolio_simulator):
        """Test closing a short position with loss."""
        # Open short position
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(sell_fill)
        
        # Close at higher price (loss for short)
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=15),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=155.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=True,
            realized_pnl=None,
            holding_time_minutes=None,
            price=155.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        portfolio_simulator.process_fill(buy_fill)
        
        # Check realized P&L (loss)
        expected_pnl = (150.0 - 155.0) * 100.0  # -$500 loss
        assert portfolio_simulator.session_realized_pnl == expected_pnl
        
    def test_partial_position_close(self, portfolio_simulator):
        """Test partially closing a position."""
        # Open position
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=200.0,
            executed_quantity=200.0,
            executed_price=150.0,
            commission=1.0,
            fees=0.0,
            slippage_cost_total=0.2,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=200.0,
            slippage_cost=0.2
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Partially close
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=155.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=155.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        enriched_fill = portfolio_simulator.process_fill(sell_fill)
        
        # Check position
        position = portfolio_simulator.positions["AAPL"]
        assert position.side == PositionSideEnum.LONG
        assert position.quantity == 100.0  # 200 - 100
        assert position.avg_price == 150.0  # Unchanged
        
        # Check realized P&L for partial close
        expected_pnl = (155.0 - 150.0) * 100.0  # $500 on 100 shares
        assert portfolio_simulator.session_realized_pnl == expected_pnl
        
        # Trade should still be open
        assert len(portfolio_simulator.open_trades) == 1
        assert enriched_fill['closes_position'] == False


class TestMarketValueAndPnL:
    """Test market value updates and P&L calculations."""
    
    def test_update_market_values_long_position(self, portfolio_simulator):
        """Test updating market values for long positions."""
        # Open position
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Update market values with higher price
        market_prices = {"AAPL": 155.0, "GOOGL": 2800.0, "MSFT": 350.0}
        current_time = datetime.now(timezone.utc)
        portfolio_simulator.update_market_values(market_prices, current_time)
        
        # Check position values
        position = portfolio_simulator.positions["AAPL"]
        assert position.market_value == 15500.0  # 100 * 155
        assert position.unrealized_pnl == 500.0  # (155 - 150) * 100
        
        # Check equity tracking
        expected_equity = portfolio_simulator.cash + 15500.0
        assert len(portfolio_simulator.equity_history) >= 2
        assert portfolio_simulator.equity_history[-1][1] == expected_equity
        
    def test_update_market_values_short_position(self, portfolio_simulator):
        """Test updating market values for short positions."""
        # Open short position
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(sell_fill)
        
        # Update with lower price (profit for short)
        market_prices = {"AAPL": 145.0, "GOOGL": 2800.0, "MSFT": 350.0}
        current_time = datetime.now(timezone.utc)
        portfolio_simulator.update_market_values(market_prices, current_time)
        
        # Check position values
        position = portfolio_simulator.positions["AAPL"]
        assert position.market_value == -14500.0  # -100 * 145 (negative for short)
        assert position.unrealized_pnl == 500.0  # (150 - 145) * 100
        
    def test_drawdown_calculation(self, portfolio_simulator):
        """Test peak equity and drawdown tracking."""
        initial_equity = portfolio_simulator.cash
        
        # Make profitable trade
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Update with higher price (new peak)
        market_prices = {"AAPL": 160.0, "GOOGL": 2800.0, "MSFT": 350.0}
        portfolio_simulator.update_market_values(market_prices, datetime.now(timezone.utc))
        
        peak_equity = portfolio_simulator.peak_equity
        assert peak_equity > initial_equity
        
        # Update with lower price (drawdown)
        market_prices["AAPL"] = 140.0
        portfolio_simulator.update_market_values(market_prices, datetime.now(timezone.utc))
        
        # Check drawdown
        assert portfolio_simulator.max_drawdown > 0
        current_equity = portfolio_simulator.cash + portfolio_simulator.positions["AAPL"].market_value
        expected_drawdown = (peak_equity - current_equity) / peak_equity
        assert portfolio_simulator.max_drawdown == pytest.approx(expected_drawdown, rel=0.001)


class TestPortfolioObservations:
    """Test portfolio observation generation for model input."""
    
    def test_portfolio_observation_shape(self, portfolio_simulator):
        """Test that portfolio observations have correct shape."""
        obs = portfolio_simulator.get_portfolio_observation()
        
        assert 'features' in obs
        assert isinstance(obs['features'], np.ndarray)
        assert obs['features'].shape == (10, 5)  # portfolio_seq_len x portfolio_feat_dim
        assert obs['features'].dtype == np.float32
        
    def test_portfolio_features_with_position(self, portfolio_simulator):
        """Test portfolio features calculation with open position."""
        # Open position with smaller size to avoid negative cash
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=50.0,
            executed_quantity=50.0,
            executed_price=150.0,
            commission=0.25,
            fees=0.0,
            slippage_cost_total=0.05,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=50.0,
            slippage_cost=0.05
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Update market values
        market_prices = {"AAPL": 155.0, "GOOGL": 2800.0, "MSFT": 350.0}
        portfolio_simulator.update_market_values(market_prices, datetime.now(timezone.utc))
        
        # Get observation
        obs = portfolio_simulator.get_portfolio_observation()
        features = obs['features']
        
        # Check latest features (last row)
        latest_features = features[-1]
        
        # Feature 0: Position size should be positive (long position)
        assert latest_features[0] > 0
        
        # Feature 1: Unrealized P&L should be positive
        assert latest_features[1] > 0
        
        # Feature 3: Cash ratio should be between 0 and 1
        # With 50 shares at $150 = $7500, cash should be ~$2500/$10250 ≈ 0.24
        assert 0 < latest_features[3] < 1


class TestPortfolioState:
    """Test portfolio state retrieval."""
    
    def test_get_portfolio_state_empty(self, portfolio_simulator):
        """Test getting portfolio state with no positions."""
        timestamp = datetime.now(timezone.utc)
        state = portfolio_simulator.get_portfolio_state(timestamp)
        
        assert state['timestamp'] == timestamp
        assert state['cash'] == 10000.0
        assert state['total_equity'] == 10000.0
        assert state['unrealized_pnl'] == 0.0
        assert state['realized_pnl_session'] == 0.0
        assert len(state['positions']) == 3  # 3 tradable assets
        assert state['position_side'] is None
        assert state['position_value'] == 0.0
        
    def test_get_portfolio_state_with_positions(self, portfolio_simulator):
        """Test getting portfolio state with open positions."""
        # Open position
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Update market values
        market_prices = {"AAPL": 155.0, "GOOGL": 2800.0, "MSFT": 350.0}
        portfolio_simulator.update_market_values(market_prices, datetime.now(timezone.utc))
        
        # Get state
        timestamp = datetime.now(timezone.utc)
        state = portfolio_simulator.get_portfolio_state(timestamp)
        
        assert state['position_side'] == 'LONG'
        assert state['position_value'] == 15500.0
        assert state['avg_entry_price'] == 150.0
        assert state['unrealized_pnl'] == 500.0
        
        # Check position details
        aapl_position = state['positions']['AAPL']
        assert aapl_position['quantity'] == 100.0
        assert aapl_position['avg_entry_price'] == 150.0
        assert aapl_position['unrealized_pnl'] == 500.0


class TestTradingMetrics:
    """Test trading metrics calculation."""
    
    def test_trading_metrics_no_trades(self, portfolio_simulator):
        """Test metrics with no completed trades."""
        metrics = portfolio_simulator.get_trading_metrics()
        
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_pnl_per_trade'] == 0.0
        assert metrics['total_realized_pnl'] == 0.0
        
    def test_trading_metrics_with_trades(self, portfolio_simulator):
        """Test metrics with completed trades."""
        # Complete a winning trade
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=30),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=155.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=True,
            realized_pnl=None,
            holding_time_minutes=None,
            price=155.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(sell_fill)
        
        # Complete a losing trade
        buy_fill2 = FillDetails(
            asset_id="GOOGL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=10.0,
            executed_quantity=10.0,
            executed_price=2800.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=2800.0,
            quantity=10.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill2)
        
        sell_fill2 = FillDetails(
            asset_id="GOOGL",
            fill_timestamp=datetime.now(timezone.utc) + timedelta(minutes=15),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=10.0,
            executed_quantity=10.0,
            executed_price=2790.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=True,
            realized_pnl=None,
            holding_time_minutes=None,
            price=2790.0,
            quantity=10.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(sell_fill2)
        
        # Get metrics
        metrics = portfolio_simulator.get_trading_metrics()
        
        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == 50.0
        assert metrics['total_realized_pnl'] == 500.0 - 100.0  # 500 win - 100 loss
        assert metrics['avg_pnl_per_trade'] == 200.0
        assert metrics['avg_winning_trade'] == 500.0
        assert metrics['avg_losing_trade'] == -100.0
        assert metrics['profit_factor'] == 5.0  # 500 / 100


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_process_fill_unknown_asset(self, portfolio_simulator):
        """Test processing fill for unknown asset."""
        fill = FillDetails(
            asset_id="UNKNOWN",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        # Should log error and return None
        result = portfolio_simulator.process_fill(fill)
        assert result is None
        # Mock logger should have been called
        
    def test_update_market_values_missing_price(self, portfolio_simulator):
        """Test updating market values with missing price."""
        # Open position
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        # Update without AAPL price
        market_prices = {"GOOGL": 2800.0, "MSFT": 350.0}
        portfolio_simulator.update_market_values(market_prices, datetime.now(timezone.utc))
        
        # Mock logger should have been called
        
    def test_zero_quantity_position(self, portfolio_simulator):
        """Test handling positions with zero quantity."""
        # Create position and then close it
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=True,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(sell_fill)
        
        # Position should be flat
        position = portfolio_simulator.positions["AAPL"]
        assert position.is_flat()
        assert position.quantity == 0.0
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0
        
    def test_trade_callback_exception(self, portfolio_simulator):
        """Test that exceptions in trade callback are handled."""
        def failing_callback(trade):
            raise Exception("Callback failed")
        
        portfolio_simulator.trade_callback = failing_callback
        
        # Complete a trade
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        sell_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=True,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        
        # Should not raise exception
        portfolio_simulator.process_fill(sell_fill)
        # Mock logger should have been called


class TestHelperMethods:
    """Test helper methods and utility functions."""
    
    def test_has_open_positions(self, portfolio_simulator):
        """Test checking for open positions."""
        assert not portfolio_simulator.has_open_positions()
        
        # Open position
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        assert portfolio_simulator.has_open_positions()
        
    def test_get_current_position(self, portfolio_simulator):
        """Test getting current position for an asset."""
        position = portfolio_simulator.get_current_position("AAPL")
        assert position is not None
        assert position.is_flat()
        
        # Non-existent asset
        position = portfolio_simulator.get_current_position("UNKNOWN")
        assert position is None
        
    def test_get_open_trade_count(self, portfolio_simulator):
        """Test counting open trades."""
        assert portfolio_simulator.get_open_trade_count() == 0
        
        # Open trade
        buy_fill = FillDetails(
            asset_id="AAPL",
            fill_timestamp=datetime.now(timezone.utc),
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            requested_quantity=100.0,
            executed_quantity=100.0,
            executed_price=150.0,
            commission=0.5,
            fees=0.0,
            slippage_cost_total=0.1,
            closes_position=False,
            realized_pnl=None,
            holding_time_minutes=None,
            price=150.0,
            quantity=100.0,
            slippage_cost=0.1
        )
        portfolio_simulator.process_fill(buy_fill)
        
        assert portfolio_simulator.get_open_trade_count() == 1
        
    def test_string_representation(self, portfolio_simulator):
        """Test string representation of portfolio simulator."""
        repr_str = repr(portfolio_simulator)
        assert "PortfolioSimulator" in repr_str
        assert "equity=$10000.00" in repr_str
        assert "cash=$10000.00" in repr_str
        assert "positions=0" in repr_str
        assert "trades=0" in repr_str