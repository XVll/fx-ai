"""
Comprehensive tests for ExecutionSimulator based on input/output behavior.
Tests all functionality and edge cases without implementation details.
"""

import pytest
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import numpy as np

from simulators.execution_simulator import (
    ExecutionSimulator, RejectionReason, ActionDecodeResult, OrderRequest, 
    ExecutionContext, ExecutionResult
)
from simulators.portfolio_simulator import OrderTypeEnum, OrderSideEnum, PositionSideEnum, FillDetails
from config.schemas import SimulationConfig, ModelConfig, EnvConfig


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


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
def np_random():
    """Create a numpy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_market_simulator():
    """Create a mock market simulator."""
    return Mock()


@pytest.fixture
def execution_simulator(logger, simulation_config, np_random, mock_market_simulator):
    """Create an execution simulator for testing."""
    return ExecutionSimulator(
        logger=logger,
        simulation_config=simulation_config,
        np_random=np_random,
        market_simulator=mock_market_simulator,
        metrics_integrator=None
    )


@pytest.fixture
def base_market_state():
    """Create a base market state for testing."""
    return {
        'best_ask_price': 100.50,
        'best_bid_price': 100.00,
        'current_price': 100.25,
        'timestamp': datetime.now(timezone.utc),
        'timestamp_utc': datetime.now(timezone.utc)
    }


@pytest.fixture
def base_portfolio_state():
    """Create a base portfolio state for testing."""
    return {
        'cash': 25000.0,
        'total_equity': 25000.0,
        'positions': {
            'MLGO': {
                'quantity': 0.0,
                'current_side': PositionSideEnum.FLAT,
                'avg_entry_price': 0.0,
                'market_value': 0.0,
                'unrealized_pnl': 0.0
            }
        }
    }


class TestActionDecoding:
    """Test action decoding functionality."""
    
    def test_decode_valid_tuple_action(self, execution_simulator):
        """Test decoding a valid tuple action."""
        result = execution_simulator.decode_action((1, 2))  # BUY, 75%
        
        assert result.action_type == "BUY"
        assert result.size_float == 0.75
        assert result.raw_action == [1, 2]
        assert result.is_valid is True
        assert result.rejection_reason is None
    
    def test_decode_valid_list_action(self, execution_simulator):
        """Test decoding a valid list action."""
        result = execution_simulator.decode_action([2, 0])  # SELL, 25%
        
        assert result.action_type == "SELL"
        assert result.size_float == 0.25
        assert result.raw_action == [2, 0]
        assert result.is_valid is True
    
    def test_decode_numpy_array_action(self, execution_simulator):
        """Test decoding a numpy array action."""
        action = np.array([0, 3])  # HOLD, 100%
        result = execution_simulator.decode_action(action)
        
        assert result.action_type == "HOLD"
        assert result.size_float == 1.0
        assert result.is_valid is True
    
    def test_decode_invalid_indices_wraps_around(self, execution_simulator):
        """Test that invalid indices wrap around modulo action space size."""
        result = execution_simulator.decode_action([5, 7])  # 5%3=2 (SELL), 7%4=3 (100%)
        
        assert result.action_type == "SELL"
        assert result.size_float == 1.0
        assert result.is_valid is True
    
    def test_decode_action_during_market_closed(self, execution_simulator):
        """Test action decoding when market is closed."""
        # Mock market closed time (before 4 AM or after 8 PM UTC)
        closed_time = datetime(2025, 1, 1, 22, 0, tzinfo=timezone.utc)  # 10 PM UTC
        
        result = execution_simulator.decode_action([1, 0], closed_time)
        
        assert result.action_type == "BUY"
        assert result.size_float == 0.25
        assert result.is_valid is False
        assert result.rejection_reason == RejectionReason.MARKET_CLOSED
    
    def test_decode_action_during_market_open(self, execution_simulator):
        """Test action decoding when market is open."""
        # Mock market open time (between 4 AM and 8 PM UTC)
        open_time = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)  # 10 AM UTC
        
        result = execution_simulator.decode_action([1, 0], open_time)
        
        assert result.action_type == "BUY"
        assert result.size_float == 0.25
        assert result.is_valid is True
        assert result.rejection_reason is None
    
    def test_decode_invalid_action_type(self, execution_simulator):
        """Test decoding with an invalid action type."""
        result = execution_simulator.decode_action("invalid")
        
        assert result.action_type == "HOLD"
        assert result.size_float == 0.25  # Default size for invalid input (0 index)
        assert result.raw_action == [0, 0]
        assert result.is_valid is True  # This gets handled gracefully, not as an error
        assert result.rejection_reason is None


class TestOrderValidationAndCreation:
    """Test order validation and creation functionality."""
    
    def test_hold_action_returns_none(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test that HOLD actions return no order."""
        action_result = ActionDecodeResult(
            action_type="HOLD", size_float=0.0, raw_action=[0, 0], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
    
    def test_invalid_action_returns_none(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test that invalid actions return no order."""
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=False,
            rejection_reason=RejectionReason.MARKET_CLOSED
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
    
    def test_valid_buy_order_creation(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test creating a valid buy order."""
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is not None
        assert order.asset_id == "MLGO"
        assert order.order_type == OrderTypeEnum.MARKET
        assert order.order_side == OrderSideEnum.BUY
        assert order.quantity > 0
        assert order.ideal_ask_price == 100.50
        assert order.ideal_bid_price == 100.00
    
    def test_valid_sell_order_creation(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test creating a valid sell order when position exists."""
        # Create portfolio state with long position
        portfolio_state = base_portfolio_state.copy()
        portfolio_state['positions']['MLGO']['quantity'] = 100.0
        portfolio_state['positions']['MLGO']['current_side'] = PositionSideEnum.LONG
        
        action_result = ActionDecodeResult(
            action_type="SELL", size_float=0.5, raw_action=[2, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, portfolio_state, "MLGO", None
        )
        
        assert order is not None
        assert order.order_side == OrderSideEnum.SELL
        assert order.quantity == 50.0  # 50% of 100 shares
    
    def test_sell_without_position_fails(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test that selling without a position fails."""
        action_result = ActionDecodeResult(
            action_type="SELL", size_float=0.5, raw_action=[2, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.NO_POSITION_TO_SELL
    
    def test_missing_market_prices_uses_current_price(self, execution_simulator, base_portfolio_state):
        """Test that missing market prices fall back to current price."""
        market_state = {
            'current_price': 100.25,
            'timestamp': datetime.now(timezone.utc)
        }
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is not None
        # Should have calculated spread around current price
        assert order.ideal_ask_price > market_state['current_price']
        assert order.ideal_bid_price < market_state['current_price']
    
    def test_missing_all_prices_fails(self, execution_simulator, base_portfolio_state):
        """Test that missing all prices fails validation."""
        market_state = {'timestamp': datetime.now(timezone.utc)}
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INVALID_PRICES
    
    def test_invalid_price_spread_fails(self, execution_simulator, base_portfolio_state):
        """Test that invalid price spreads fail validation."""
        market_state = {
            'best_ask_price': 99.00,  # Ask lower than bid
            'best_bid_price': 100.00,
            'timestamp': datetime.now(timezone.utc)
        }
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INVALID_PRICES
    
    def test_insufficient_cash_fails(self, execution_simulator, base_market_state):
        """Test that insufficient cash fails validation."""
        portfolio_state = {
            'cash': 5.0,  # Very low cash
            'total_equity': 5.0,
            'positions': {
                'MLGO': {
                    'quantity': 0.0,
                    'current_side': PositionSideEnum.FLAT,
                    'avg_entry_price': 0.0,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0
                }
            }
        }
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=1.0, raw_action=[1, 3], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INSUFFICIENT_CASH
    
    def test_quantity_too_small_fails(self, execution_simulator, base_market_state):
        """Test that very small quantities fail validation."""
        portfolio_state = {
            'cash': 50.0,  # Small amount that results in <1 share
            'total_equity': 50.0,
            'positions': {
                'MLGO': {
                    'quantity': 0.0,
                    'current_side': PositionSideEnum.FLAT,
                    'avg_entry_price': 0.0,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0
                }
            }
        }
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.25, raw_action=[1, 0], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.QUANTITY_TOO_SMALL
    
    def test_unknown_asset_fails(self, execution_simulator, base_market_state):
        """Test that unknown assets fail validation."""
        portfolio_state = {
            'cash': 25000.0,
            'total_equity': 25000.0,
            'positions': {}  # No position data for asset
        }
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, portfolio_state, "UNKNOWN", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INVALID_SYMBOL


class TestOrderExecution:
    """Test order execution functionality."""
    
    def test_successful_order_execution(self, execution_simulator, base_market_state):
        """Test successful order execution."""
        order = OrderRequest(
            asset_id="MLGO",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.50,
            ideal_bid_price=100.00,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        context = ExecutionContext(
            market_state=base_market_state,
            portfolio_state={},
            session_volume=0.0,
            session_turnover=0.0,
            time_of_day=0.5
        )
        
        fill = execution_simulator.execute_order(order, context)
        
        assert fill is not None
        assert fill.asset_id == "MLGO"
        assert fill.order_side == OrderSideEnum.BUY
        assert fill.executed_quantity == 100.0
        assert fill.executed_price > order.ideal_ask_price  # Should have slippage
        assert fill.commission > 0
        assert fill.fees > 0
        assert fill.slippage_cost_total > 0
    
    def test_execution_updates_session_stats(self, execution_simulator, base_market_state):
        """Test that execution updates session statistics."""
        initial_fills = execution_simulator.session_fills
        initial_volume = execution_simulator.session_volume
        
        order = OrderRequest(
            asset_id="MLGO",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.50,
            ideal_bid_price=100.00,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        context = ExecutionContext(
            market_state=base_market_state,
            portfolio_state={},
            session_volume=0.0,
            session_turnover=0.0,
            time_of_day=0.5
        )
        
        execution_simulator.execute_order(order, context)
        
        assert execution_simulator.session_fills == initial_fills + 1
        assert execution_simulator.session_volume == initial_volume + 100.0
        assert execution_simulator.total_orders_attempted > 0
        assert execution_simulator.total_orders_filled > 0
    
    def test_commission_calculation(self, execution_simulator):
        """Test commission calculation with min/max limits."""
        # Test commission calculation as implemented (min then max cap)
        commission1 = execution_simulator._calculate_commission(1.0, 100.0)
        # The implementation applies min then max, so for small trade values:
        # base: 0.005, min: 1.0, max: 0.5% of 100 = 0.5
        # Result: min(max(0.005, 1.0), 0.5) = min(1.0, 0.5) = 0.5
        trade_value = 1.0 * 100.0
        base_commission = 1.0 * execution_simulator.commission_per_share
        after_min = max(base_commission, execution_simulator.min_commission)
        max_allowed = trade_value * (execution_simulator.max_commission_pct / 100.0)
        expected = min(after_min, max_allowed)
        assert abs(commission1 - expected) < 0.01
        
        # Test normal commission (large trade should get per-share rate)
        commission2 = execution_simulator._calculate_commission(1000.0, 100.0)
        trade_value2 = 1000.0 * 100.0
        base_commission2 = 1000.0 * execution_simulator.commission_per_share
        after_min2 = max(base_commission2, execution_simulator.min_commission)
        max_allowed2 = trade_value2 * (execution_simulator.max_commission_pct / 100.0)
        expected2 = min(after_min2, max_allowed2)
        assert abs(commission2 - expected2) < 0.01
        
        # Test that the commission respects the percentage cap
        very_expensive_price = 1000000.0
        commission3 = execution_simulator._calculate_commission(1.0, very_expensive_price)
        max_allowed3 = very_expensive_price * (execution_simulator.max_commission_pct / 100.0)
        assert commission3 <= max_allowed3
    
    def test_slippage_calculation_components(self, execution_simulator, base_market_state):
        """Test that slippage calculation includes all components."""
        order = OrderRequest(
            asset_id="MLGO",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.50,
            ideal_bid_price=100.00,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        # Test at market open (should have higher slippage)
        context_open = ExecutionContext(
            market_state=base_market_state,
            portfolio_state={},
            session_volume=0.0,
            session_turnover=10000.0,  # Some turnover for volume impact
            time_of_day=0.05  # Near market open
        )
        
        price_open, slippage_open = execution_simulator._calculate_execution_price(order, context_open)
        
        # Test at market mid-day (should have lower slippage)
        context_mid = ExecutionContext(
            market_state=base_market_state,
            portfolio_state={},
            session_volume=0.0,
            session_turnover=10000.0,
            time_of_day=0.5  # Mid-day
        )
        
        price_mid, slippage_mid = execution_simulator._calculate_execution_price(order, context_mid)
        
        # Open should have higher slippage than mid-day
        assert slippage_open > slippage_mid
        assert price_open > price_mid  # For buy orders
    
    def test_latency_simulation(self, execution_simulator):
        """Test latency simulation produces reasonable values."""
        latencies = [execution_simulator._simulate_latency() for _ in range(100)]
        
        # All latencies should be positive
        assert all(lat > 0 for lat in latencies)
        
        # Should be centered around mean
        mean_latency = np.mean(latencies)
        assert abs(mean_latency - execution_simulator.base_latency_ms) < 20  # Within reasonable range


class TestCompleteActionExecution:
    """Test the complete action execution pipeline."""
    
    def test_successful_buy_action_pipeline(self, execution_simulator, base_portfolio_state):
        """Test complete successful buy action execution."""
        # Create market state with open market hours
        open_market_state = {
            'best_ask_price': 100.50,
            'best_bid_price': 100.00,
            'current_price': 100.25,
            'timestamp': datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),  # 10 AM UTC (market open)
            'timestamp_utc': datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        }
        
        result = execution_simulator.execute_action(
            raw_action=[1, 1],  # BUY, 50%
            market_state=open_market_state,
            portfolio_state=base_portfolio_state,
            primary_asset="MLGO",
            portfolio_manager=None
        )
        
        assert result.action_decode_result.action_type == "BUY"
        assert result.action_decode_result.size_float == 0.50
        assert result.action_decode_result.is_valid is True
        assert result.order_request is not None
        assert result.fill_details is not None
        assert result.execution_stats['total_attempted'] > 0
        assert result.execution_stats['total_filled'] > 0
        assert result.execution_stats['fill_rate'] > 0
    
    def test_hold_action_pipeline(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test complete hold action execution."""
        result = execution_simulator.execute_action(
            raw_action=[0, 0],  # HOLD
            market_state=base_market_state,
            portfolio_state=base_portfolio_state,
            primary_asset="MLGO",
            portfolio_manager=None
        )
        
        assert result.action_decode_result.action_type == "HOLD"
        assert result.order_request is None
        assert result.fill_details is None
        assert result.execution_stats['total_attempted'] == 0  # No order attempted
    
    def test_invalid_action_pipeline(self, execution_simulator, base_portfolio_state):
        """Test pipeline with invalid action."""
        # Market state with invalid prices
        invalid_market_state = {
            'timestamp': datetime.now(timezone.utc),
            'timestamp_utc': datetime.now(timezone.utc)
        }
        
        result = execution_simulator.execute_action(
            raw_action=[1, 1],  # BUY, 50%
            market_state=invalid_market_state,
            portfolio_state=base_portfolio_state,
            primary_asset="MLGO",
            portfolio_manager=None
        )
        
        assert result.action_decode_result.action_type == "BUY"
        assert result.action_decode_result.is_valid is False
        assert result.order_request is None
        assert result.fill_details is None
        assert result.execution_stats['total_rejected'] > 0
    
    def test_execution_stats_tracking(self, execution_simulator, base_portfolio_state):
        """Test that execution stats are properly tracked."""
        # Create market state with open market hours
        open_market_state = {
            'best_ask_price': 100.50,
            'best_bid_price': 100.00,
            'current_price': 100.25,
            'timestamp': datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),  # 10 AM UTC (market open)
            'timestamp_utc': datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        }
        
        # Execute multiple actions
        for i in range(5):
            execution_simulator.execute_action(
                raw_action=[1, 1],  # BUY, 50%
                market_state=open_market_state,
                portfolio_state=base_portfolio_state,
                primary_asset="MLGO",
                portfolio_manager=None
            )
        
        stats = execution_simulator.get_session_stats()
        
        assert stats['session_fills'] == 5
        assert stats['total_orders_attempted'] == 5
        assert stats['total_orders_filled'] == 5
        assert stats['total_orders_rejected'] == 0
        assert stats['fill_rate'] == 1.0
        assert stats['session_volume'] > 0
        assert stats['session_turnover'] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_negative_prices(self, execution_simulator, base_portfolio_state):
        """Test handling of negative prices."""
        market_state = {
            'best_ask_price': -100.50,
            'best_bid_price': -100.00,
            'timestamp': datetime.now(timezone.utc)
        }
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.5, raw_action=[1, 1], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INVALID_PRICES
    
    def test_zero_quantities(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test handling of zero quantities."""
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=0.0, raw_action=[1, 0], is_valid=True
        )
        
        # This should fail due to quantity too small
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, base_portfolio_state, "MLGO", None
        )
        
        assert order is None
        assert action_result.is_valid is False
    
    def test_very_large_quantities(self, execution_simulator, base_market_state, base_portfolio_state):
        """Test handling of very large quantities."""
        # Modify portfolio to have very high equity
        portfolio_state = base_portfolio_state.copy()
        portfolio_state['cash'] = 1e10  # 10 billion
        portfolio_state['total_equity'] = 1e10
        
        # Modify execution simulator's default position value
        execution_simulator.default_position_value = 1e8  # 100 million default position
        
        action_result = ActionDecodeResult(
            action_type="BUY", size_float=1.0, raw_action=[1, 3], is_valid=True
        )
        
        order = execution_simulator.validate_and_create_order(
            action_result, base_market_state, portfolio_state, "MLGO", None
        )
        
        # Should still create order, just very large
        assert order is not None
        assert order.quantity > 1000  # Should be very large
    
    def test_reset_functionality(self, execution_simulator):
        """Test reset functionality."""
        # Create some session state
        execution_simulator.session_fills = 10
        execution_simulator.session_volume = 1000.0
        execution_simulator.total_orders_attempted = 15
        
        # Reset
        execution_simulator.reset()
        
        # Should be back to initial state
        assert execution_simulator.session_fills == 0
        assert execution_simulator.session_volume == 0.0
        assert execution_simulator.total_orders_attempted == 0
        assert execution_simulator.total_orders_filled == 0
        assert execution_simulator.total_orders_rejected == 0
    
    def test_time_of_day_calculation(self, execution_simulator):
        """Test time of day calculation for various times."""
        # Test market open (4 AM UTC)
        open_time = datetime(2025, 1, 1, 4, 0, tzinfo=timezone.utc)
        time_fraction = execution_simulator._get_time_of_day(open_time)
        assert time_fraction == 0.0
        
        # Test market close (8 PM UTC)
        close_time = datetime(2025, 1, 1, 20, 0, tzinfo=timezone.utc)
        time_fraction = execution_simulator._get_time_of_day(close_time)
        assert time_fraction == 1.0
        
        # Test mid-day (12 PM UTC)
        mid_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        time_fraction = execution_simulator._get_time_of_day(mid_time)
        assert 0.4 < time_fraction < 0.6  # Should be around middle
        
        # Test before market open
        before_time = datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc)
        time_fraction = execution_simulator._get_time_of_day(before_time)
        assert time_fraction == 0.0
        
        # Test after market close
        after_time = datetime(2025, 1, 1, 22, 0, tzinfo=timezone.utc)
        time_fraction = execution_simulator._get_time_of_day(after_time)
        assert time_fraction == 1.0
    
    def test_market_hours_check(self, execution_simulator):
        """Test market hours checking."""
        # Test during market hours
        market_time = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        assert not execution_simulator._is_market_closed(market_time)
        
        # Test before market open
        before_time = datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc)
        assert execution_simulator._is_market_closed(before_time)
        
        # Test after market close
        after_time = datetime(2025, 1, 1, 22, 0, tzinfo=timezone.utc)
        assert execution_simulator._is_market_closed(after_time)


class TestMetricsIntegration:
    """Test metrics integration functionality."""
    
    def test_metrics_recording(self, logger, simulation_config, np_random, mock_market_simulator):
        """Test that metrics are recorded when integrator is provided."""
        mock_metrics = Mock()
        
        execution_simulator = ExecutionSimulator(
            logger=logger,
            simulation_config=simulation_config,
            np_random=np_random,
            market_simulator=mock_market_simulator,
            metrics_integrator=mock_metrics
        )
        
        order = OrderRequest(
            asset_id="MLGO",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.50,
            ideal_bid_price=100.00,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        context = ExecutionContext(
            market_state={'current_price': 100.25},
            portfolio_state={},
            session_volume=0.0,
            session_turnover=0.0,
            time_of_day=0.5
        )
        
        execution_simulator.execute_order(order, context)
        
        # Should have called metrics integrator
        mock_metrics.record_execution.assert_called_once()
    
    def test_no_metrics_when_none_provided(self, execution_simulator, base_market_state):
        """Test that no metrics recording happens when no integrator provided."""
        # This test just ensures no errors occur when metrics_integrator is None
        order = OrderRequest(
            asset_id="MLGO",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.50,
            ideal_bid_price=100.00,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        context = ExecutionContext(
            market_state=base_market_state,
            portfolio_state={},
            session_volume=0.0,
            session_turnover=0.0,
            time_of_day=0.5
        )
        
        # Should not raise any errors
        fill = execution_simulator.execute_order(order, context)
        assert fill is not None


if __name__ == "__main__":
    pytest.main([__file__])