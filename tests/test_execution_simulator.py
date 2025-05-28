"""
Comprehensive tests for ExecutionSimulator.

These tests verify all aspects of the execution simulator including:
- Action decoding
- Order validation and creation
- Order execution with realistic market conditions
- Commission and fee calculations
- Session statistics tracking
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
from unittest.mock import Mock, MagicMock

from simulators.execution_simulator import (
    ExecutionSimulator, ActionDecodeResult, OrderRequest, ExecutionContext,
    ExecutionResult, RejectionReason
)
from simulators.portfolio_simulator import (
    OrderTypeEnum, OrderSideEnum, PositionSideEnum, FillDetails
)
from simulators.market_simulator import MarketSimulator
from config.schemas import SimulationConfig


class TestActionDecoding:
    """Test action decoding functionality."""
    
    @pytest.fixture
    def basic_simulator(self):
        """Create basic simulator for testing."""
        config = SimulationConfig()
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        market_sim = Mock(spec=MarketSimulator)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=market_sim,
            metrics_integrator=None
        )
    
    def test_decode_action_valid_tuple(self, basic_simulator):
        """Test decoding valid tuple action."""
        # Test all valid action combinations
        test_cases = [
            ((0, 0), "HOLD", 0.25),
            ((1, 0), "BUY", 0.25),
            ((2, 0), "SELL", 0.25),
            ((1, 1), "BUY", 0.50),
            ((1, 2), "BUY", 0.75),
            ((1, 3), "BUY", 1.0),
            ((2, 3), "SELL", 1.0),
        ]
        
        for action, expected_type, expected_size in test_cases:
            result = basic_simulator.decode_action(action)
            assert result.action_type == expected_type
            assert result.size_float == expected_size
            assert result.is_valid is True
            assert result.rejection_reason is None
    
    def test_decode_action_valid_list(self, basic_simulator):
        """Test decoding valid list action."""
        action = [1, 2]  # BUY with 75% size
        result = basic_simulator.decode_action(action)
        
        assert result.action_type == "BUY"
        assert result.size_float == 0.75
        assert result.is_valid is True
        assert result.raw_action == [1, 2]
    
    def test_decode_action_numpy_array(self, basic_simulator):
        """Test decoding numpy array action."""
        action = np.array([2, 1])  # SELL with 50% size
        result = basic_simulator.decode_action(action)
        
        assert result.action_type == "SELL"
        assert result.size_float == 0.50
        assert result.is_valid is True
        assert result.raw_action == [2, 1]
    
    def test_decode_action_with_overflow_indices(self, basic_simulator):
        """Test action decoding handles overflow indices correctly."""
        # Indices larger than array size should wrap around
        action = (5, 7)  # 5 % 3 = 2 (SELL), 7 % 4 = 3 (1.0)
        result = basic_simulator.decode_action(action)
        
        assert result.action_type == "SELL"
        assert result.size_float == 1.0
        assert result.is_valid is True
    
    def test_decode_action_market_closed(self, basic_simulator):
        """Test action decoding when market is closed."""
        # Create time outside market hours (e.g., 9 PM)
        closed_time = datetime(2024, 1, 1, 21, 0, 0, tzinfo=timezone.utc)
        
        action = (1, 2)  # BUY action
        result = basic_simulator.decode_action(action, closed_time)
        
        assert result.action_type == "BUY"
        assert result.size_float == 0.75
        assert result.is_valid is False
        assert result.rejection_reason == RejectionReason.MARKET_CLOSED
        assert "Market is closed" in result.rejection_details
    
    def test_decode_action_invalid_type(self, basic_simulator):
        """Test handling of invalid action types."""
        # Test with unexpected type
        action = "invalid_action"
        result = basic_simulator.decode_action(action)
        
        assert result.action_type == "HOLD"
        assert result.size_float == 0.25  # Default size is 0.25
        assert result.is_valid is True  # Still valid, just uses defaults
        assert result.rejection_reason is None
        assert result.raw_action == [0, 0]


class TestOrderValidationAndCreation:
    """Test order validation and creation logic."""
    
    @pytest.fixture
    def simulator_with_mocks(self):
        """Create simulator with necessary mocks."""
        config = SimulationConfig(
            default_position_value=10000.0,
            max_position_value_ratio=0.5,
            allow_shorting=False
        )
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        market_sim = Mock(spec=MarketSimulator)
        
        sim = ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=market_sim,
            metrics_integrator=None
        )
        
        return sim
    
    def test_validate_hold_action(self, simulator_with_mocks):
        """Test that HOLD actions return None order."""
        action_result = ActionDecodeResult(
            action_type="HOLD",
            size_float=0.0,
            raw_action=[0, 0],
            is_valid=True
        )
        
        market_state = {"best_ask_price": 100.0, "best_bid_price": 99.9}
        portfolio_state = {"cash": 50000.0, "positions": {}}
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is None
    
    def test_validate_buy_order_valid(self, simulator_with_mocks):
        """Test creating valid buy order."""
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=0.5,  # 50% position
            raw_action=[1, 1],
            is_valid=True
        )
        
        timestamp = datetime.now(timezone.utc)
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.9,
            "timestamp_utc": timestamp
        }
        
        portfolio_state = {
            "cash": 50000.0,
            "total_equity": 100000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is not None
        assert order.asset_id == "TEST"
        assert order.order_type == OrderTypeEnum.MARKET
        assert order.order_side == OrderSideEnum.BUY
        assert order.quantity == 50.0  # (10000 * 0.5) / 100 = 50 shares
        assert order.ideal_ask_price == 100.0
        assert order.ideal_bid_price == 99.9
        assert order.decision_timestamp == timestamp
    
    def test_validate_buy_order_insufficient_cash(self, simulator_with_mocks):
        """Test buy order rejection with insufficient cash."""
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=1.0,  # 100% position
            raw_action=[1, 3],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.9
        }
        
        portfolio_state = {
            "cash": 5.0,  # Only $5 available
            "total_equity": 100000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INSUFFICIENT_CASH
        assert "Insufficient buying power" in action_result.rejection_details
    
    def test_validate_sell_order_valid(self, simulator_with_mocks):
        """Test creating valid sell order."""
        action_result = ActionDecodeResult(
            action_type="SELL",
            size_float=0.25,  # Sell 25% of position
            raw_action=[2, 0],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.9
        }
        
        portfolio_state = {
            "cash": 10000.0,
            "positions": {
                "TEST": {
                    "quantity": 200.0,  # Own 200 shares
                    "current_side": PositionSideEnum.LONG
                }
            }
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is not None
        assert order.order_side == OrderSideEnum.SELL
        assert order.quantity == 50.0  # 200 * 0.25 = 50 shares
    
    def test_validate_sell_order_no_position(self, simulator_with_mocks):
        """Test sell order rejection when no position exists."""
        action_result = ActionDecodeResult(
            action_type="SELL",
            size_float=0.5,
            raw_action=[2, 1],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.9
        }
        
        portfolio_state = {
            "cash": 50000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.NO_POSITION_TO_SELL
    
    def test_validate_order_invalid_prices(self, simulator_with_mocks):
        """Test order rejection with invalid market prices."""
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=0.5,
            raw_action=[1, 1],
            is_valid=True
        )
        
        # Test various invalid price scenarios
        invalid_price_scenarios = [
            {"best_ask_price": None, "best_bid_price": 99.9},
            {"best_ask_price": 100.0, "best_bid_price": None},
            {"best_ask_price": 0.0, "best_bid_price": 99.9},
            {"best_ask_price": 99.9, "best_bid_price": 100.0},  # Inverted spread
        ]
        
        portfolio_state = {
            "cash": 50000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        for market_state in invalid_price_scenarios:
            action_result.is_valid = True  # Reset for each test
            order = simulator_with_mocks.validate_and_create_order(
                action_result, market_state, portfolio_state, "TEST", None
            )
            
            assert order is None
            assert action_result.is_valid is False
            assert action_result.rejection_reason == RejectionReason.INVALID_PRICES
    
    def test_validate_order_missing_symbol(self, simulator_with_mocks):
        """Test order rejection when symbol has no position data."""
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=0.5,
            raw_action=[1, 1],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.9
        }
        
        portfolio_state = {
            "cash": 50000.0,
            "positions": {}  # No position data for TEST
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.INVALID_SYMBOL
    
    def test_validate_order_quantity_too_small(self, simulator_with_mocks):
        """Test order rejection when quantity is too small."""
        action_result = ActionDecodeResult(
            action_type="SELL",
            size_float=0.25,
            raw_action=[2, 0],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.9
        }
        
        portfolio_state = {
            "cash": 50000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.5,  # Very small position
                    "current_side": PositionSideEnum.LONG
                }
            }
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is None
        assert action_result.is_valid is False
        assert action_result.rejection_reason == RejectionReason.QUANTITY_TOO_SMALL
    
    def test_validate_order_uses_current_price_fallback(self, simulator_with_mocks):
        """Test order creation using current_price when BBO is missing."""
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=0.5,
            raw_action=[1, 1],
            is_valid=True
        )
        
        market_state = {
            "current_price": 100.0,  # Only current price available
            "best_ask_price": None,
            "best_bid_price": None
        }
        
        portfolio_state = {
            "cash": 50000.0,
            "total_equity": 100000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        order = simulator_with_mocks.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is not None
        # Should calculate spread from current price
        assert order.ideal_ask_price == pytest.approx(100.02, rel=1e-3)
        assert order.ideal_bid_price == pytest.approx(99.98, rel=1e-3)


class TestOrderExecution:
    """Test order execution with market dynamics."""
    
    @pytest.fixture
    def execution_simulator(self):
        """Create simulator with specific execution parameters."""
        config = SimulationConfig(
            mean_latency_ms=10.0,
            latency_std_dev_ms=2.0,
            base_slippage_bps=5.0,
            max_total_slippage_bps=50.0,
            size_impact_slippage_bps_per_unit=10.0,
            market_impact_coefficient=0.0001,
            market_impact_model="linear",
            commission_per_share=0.005,
            fee_per_share=0.002,
            min_commission_per_order=1.0,
            max_commission_pct_of_value=0.5
        )
        
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        market_sim = Mock(spec=MarketSimulator)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=market_sim,
            metrics_integrator=None
        )
    
    def test_execute_order_buy_success(self, execution_simulator):
        """Test successful buy order execution."""
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=50.0,
            ideal_bid_price=49.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={"timestamp_utc": timestamp},
            portfolio_state={"cash": 10000.0},
            session_volume=5000.0,
            session_turnover=250000.0,
            time_of_day=0.5  # Mid-day
        )
        
        fill = execution_simulator.execute_order(order, context)
        
        assert fill is not None
        assert fill['asset_id'] == "TEST"
        assert fill['order_type'] == OrderTypeEnum.MARKET
        assert fill['order_side'] == OrderSideEnum.BUY
        assert fill['requested_quantity'] == 100.0
        assert fill['executed_quantity'] == 100.0
        assert fill['executed_price'] > 50.0  # Should have slippage
        assert fill['commission'] > 0
        assert fill['fees'] > 0
        assert fill['slippage_cost_total'] > 0
        
    def test_execute_order_sell_success(self, execution_simulator):
        """Test successful sell order execution."""
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL,
            quantity=50.0,
            ideal_ask_price=50.0,
            ideal_bid_price=49.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={"timestamp_utc": timestamp},
            portfolio_state={"cash": 1000.0},
            session_volume=1000.0,
            session_turnover=50000.0,
            time_of_day=0.3
        )
        
        fill = execution_simulator.execute_order(order, context)
        
        assert fill is not None
        assert fill['order_side'] == OrderSideEnum.SELL
        assert fill['executed_price'] < 49.95  # Should have negative slippage for sell
    
    def test_slippage_calculation_basic(self, execution_simulator):
        """Test basic slippage calculation."""
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.0,
            ideal_bid_price=99.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=10000.0,
            session_turnover=1000000.0,
            time_of_day=0.5
        )
        
        executed_price, slippage_bps = execution_simulator._calculate_execution_price(
            order, context
        )
        
        # Base slippage is 5 bps
        assert slippage_bps >= 5.0
        assert executed_price > 100.0
        
    def test_slippage_at_market_open(self, execution_simulator):
        """Test higher slippage at market open."""
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=100.0,
            ideal_bid_price=99.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=100.0,
            session_turnover=10000.0,
            time_of_day=0.05  # Near market open
        )
        
        executed_price, slippage_bps = execution_simulator._calculate_execution_price(
            order, context
        )
        
        # Should have additional 5 bps for market open
        assert slippage_bps >= 10.0
    
    def test_slippage_with_volume_impact(self, execution_simulator):
        """Test slippage with volume impact."""
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=1000.0,  # Large order
            ideal_ask_price=100.0,
            ideal_bid_price=99.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=5000.0,
            session_turnover=50000.0,  # Small market
            time_of_day=0.5
        )
        
        executed_price, slippage_bps = execution_simulator._calculate_execution_price(
            order, context
        )
        
        # Large order relative to market should have high slippage
        assert slippage_bps > 15.0
    
    def test_slippage_max_cap(self, execution_simulator):
        """Test that slippage is capped at maximum."""
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=10000.0,  # Huge order
            ideal_ask_price=100.0,
            ideal_bid_price=99.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=100.0,
            session_turnover=1000.0,  # Tiny market
            time_of_day=0.95  # Near close
        )
        
        executed_price, slippage_bps = execution_simulator._calculate_execution_price(
            order, context
        )
        
        # Should be capped at max_total_slippage_bps = 50
        assert slippage_bps == 50.0
    
    def test_latency_simulation(self, execution_simulator):
        """Test latency simulation."""
        latencies = []
        for _ in range(100):
            latency = execution_simulator._simulate_latency()
            latencies.append(latency)
            assert latency >= 1.0  # Minimum 1ms
        
        # Check mean is close to configured value
        assert np.mean(latencies) == pytest.approx(10.0, abs=1.0)
        # Check some variation exists
        assert np.std(latencies) > 0


class TestCommissionAndFees:
    """Test commission and fee calculations."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator with specific fee structure."""
        config = SimulationConfig(
            commission_per_share=0.005,
            fee_per_share=0.002,
            min_commission_per_order=1.0,
            max_commission_pct_of_value=0.5
        )
        
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=Mock(),
            metrics_integrator=None
        )
    
    def test_commission_basic(self, simulator):
        """Test basic commission calculation."""
        commission = simulator._calculate_commission(100.0, 50.0)
        # 100 shares * $0.005 = $0.50, but min is $1.0
        assert commission == 1.0
    
    def test_commission_above_minimum(self, simulator):
        """Test commission above minimum."""
        commission = simulator._calculate_commission(500.0, 50.0)
        # 500 shares * $0.005 = $2.50
        assert commission == 2.5
    
    def test_commission_max_percentage(self, simulator):
        """Test commission capped at max percentage."""
        commission = simulator._calculate_commission(10.0, 10.0)
        # Trade value = $100, max commission = 0.5% = $0.50
        # Per share would be 10 * 0.005 = $0.05, but min is $1.0
        # So it would want $1.0, but max is $0.50
        assert commission == 0.5
    
    def test_fees_calculation(self, simulator):
        """Test regulatory fee calculation."""
        fees = simulator._calculate_fees(100.0)
        assert fees == 0.2  # 100 * 0.002
        
        fees = simulator._calculate_fees(1000.0)
        assert fees == 2.0  # 1000 * 0.002


class TestSessionStatistics:
    """Test session statistics tracking."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator for testing."""
        config = SimulationConfig()
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=Mock(),
            metrics_integrator=None
        )
    
    def test_initial_session_stats(self, simulator):
        """Test initial session statistics."""
        stats = simulator.get_session_stats()
        
        assert stats['session_fills'] == 0
        assert stats['session_volume'] == 0.0
        assert stats['session_turnover'] == 0.0
        assert stats['session_commission'] == 0.0
        assert stats['session_slippage'] == 0.0
        assert stats['total_orders_attempted'] == 0
        assert stats['total_orders_filled'] == 0
        assert stats['total_orders_rejected'] == 0
        assert stats['fill_rate'] == 0.0
        assert stats['avg_latency_ms'] == 0.0
        assert stats['avg_slippage_bps'] == 0.0
        assert stats['avg_trade_size'] == 0.0
    
    def test_session_stats_after_fills(self, simulator):
        """Test session statistics after executing fills."""
        # Simulate some fills
        fill1 = {
            'executed_quantity': 100.0,
            'executed_price': 50.0,
            'commission': 1.0,
            'slippage_cost_total': 5.0
        }
        
        fill2 = {
            'executed_quantity': 200.0,
            'executed_price': 51.0,
            'commission': 1.5,
            'slippage_cost_total': 10.0
        }
        
        simulator._update_session_stats(fill1, 10.5, 10.0)
        simulator._update_session_stats(fill2, 12.0, 15.0)
        
        simulator.total_orders_attempted = 3
        simulator.total_orders_filled = 2
        simulator.total_orders_rejected = 1
        
        stats = simulator.get_session_stats()
        
        assert stats['session_fills'] == 2
        assert stats['session_volume'] == 300.0
        assert stats['session_turnover'] == 15200.0  # 100*50 + 200*51
        assert stats['session_commission'] == 2.5
        assert stats['session_slippage'] == 15.0
        assert stats['fill_rate'] == pytest.approx(2/3)
        assert stats['avg_latency_ms'] == pytest.approx(11.25)
        assert stats['avg_slippage_bps'] == pytest.approx(12.5)
        assert stats['avg_trade_size'] == 150.0
    
    def test_rejection_tracking(self, simulator):
        """Test rejection reason tracking."""
        # Track some rejections
        simulator.rejection_counts[RejectionReason.INSUFFICIENT_CASH] = 3
        simulator.rejection_counts[RejectionReason.NO_POSITION_TO_SELL] = 2
        simulator.rejection_counts[RejectionReason.INVALID_PRICES] = 1
        
        stats = simulator.get_session_stats()
        
        rejection_counts = stats['rejection_counts']
        assert rejection_counts[RejectionReason.INSUFFICIENT_CASH] == 3
        assert rejection_counts[RejectionReason.NO_POSITION_TO_SELL] == 2
        assert rejection_counts[RejectionReason.INVALID_PRICES] == 1
    
    def test_reset_functionality(self, simulator):
        """Test reset clears all statistics."""
        # Add some data
        simulator.session_fills = 5
        simulator.session_volume = 1000.0
        simulator.total_orders_attempted = 10
        simulator.rejection_counts[RejectionReason.MARKET_CLOSED] = 2
        simulator.fill_latencies = [10.0, 12.0, 15.0]
        
        # Reset
        simulator.reset()
        
        # Verify everything is cleared
        assert simulator.session_fills == 0
        assert simulator.session_volume == 0.0
        assert simulator.total_orders_attempted == 0
        assert all(count == 0 for count in simulator.rejection_counts.values())
        assert len(simulator.fill_latencies) == 0


class TestMarketHours:
    """Test market hours validation."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator for testing."""
        config = SimulationConfig()
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=Mock(),
            metrics_integrator=None
        )
    
    def test_market_hours_check(self, simulator):
        """Test market hours checking."""
        # Test various times
        test_cases = [
            (datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc), True),   # 3 AM UTC - closed
            (datetime(2024, 1, 1, 4, 0, tzinfo=timezone.utc), False),  # 4 AM UTC - open
            (datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc), False), # Noon UTC - open
            (datetime(2024, 1, 1, 19, 59, tzinfo=timezone.utc), False), # 7:59 PM UTC - open
            (datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc), True),  # 8 PM UTC - closed
            (datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc), True),  # 11 PM UTC - closed
        ]
        
        for time, expected_closed in test_cases:
            assert simulator._is_market_closed(time) == expected_closed
    
    def test_time_of_day_calculation(self, simulator):
        """Test time of day fraction calculation."""
        # Test various times
        test_cases = [
            (datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc), 0.0),    # Before open
            (datetime(2024, 1, 1, 4, 0, tzinfo=timezone.utc), 0.0),    # Market open
            (datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc), 0.5),   # Mid-day
            (datetime(2024, 1, 1, 16, 0, tzinfo=timezone.utc), 0.75),  # 3/4 through day
            (datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc), 1.0),   # Market close
            (datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc), 1.0),   # After close
        ]
        
        for time, expected_fraction in test_cases:
            assert simulator._get_time_of_day(time) == pytest.approx(expected_fraction)


class TestCompleteExecutionFlow:
    """Test complete execution flow integration."""
    
    @pytest.fixture
    def full_simulator(self):
        """Create fully configured simulator."""
        config = SimulationConfig(
            default_position_value=10000.0,
            mean_latency_ms=10.0,
            latency_std_dev_ms=2.0,
            base_slippage_bps=5.0,
            commission_per_share=0.005,
            fee_per_share=0.002,
            min_commission_per_order=1.0
        )
        
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        market_sim = Mock(spec=MarketSimulator)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=market_sim,
            metrics_integrator=None
        )
    
    def test_complete_buy_execution(self, full_simulator):
        """Test complete buy order flow from action to fill."""
        # Use a timestamp during market hours (e.g., 2 PM UTC = 10 AM ET)
        timestamp = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        raw_action = (1, 2)  # BUY with 75% size
        
        market_state = {
            "best_ask_price": 50.0,
            "best_bid_price": 49.95,
            "timestamp_utc": timestamp
        }
        
        portfolio_state = {
            "cash": 20000.0,
            "total_equity": 50000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        result = full_simulator.execute_action(
            raw_action, market_state, portfolio_state, "TEST", None
        )
        
        # Verify complete result
        assert result.fill_details is not None
        assert result.action_decode_result.action_type == "BUY"
        assert result.action_decode_result.size_float == 0.75
        assert result.action_decode_result.is_valid is True
        assert result.order_request is not None
        assert result.order_request.quantity == 150.0  # (10000 * 0.75) / 50 = 150
        assert result.execution_stats['total_filled'] == 1
        assert result.execution_stats['fill_rate'] == 1.0
    
    def test_complete_sell_execution(self, full_simulator):
        """Test complete sell order flow."""
        raw_action = [2, 3]  # SELL with 100% size
        
        market_state = {
            "best_ask_price": 55.0,
            "best_bid_price": 54.95,
            "timestamp_utc": datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)  # Market hours
        }
        
        portfolio_state = {
            "cash": 5000.0,
            "positions": {
                "TEST": {
                    "quantity": 200.0,
                    "current_side": PositionSideEnum.LONG
                }
            }
        }
        
        result = full_simulator.execute_action(
            raw_action, market_state, portfolio_state, "TEST", None
        )
        
        assert result.fill_details is not None
        assert result.action_decode_result.action_type == "SELL"
        assert result.order_request.quantity == 200.0  # Sell all
        assert result.fill_details['order_side'] == OrderSideEnum.SELL
        assert result.fill_details['executed_price'] < 54.95  # Slippage
    
    def test_hold_action_flow(self, full_simulator):
        """Test HOLD action produces no order."""
        raw_action = (0, 0)  # HOLD
        
        market_state = {"best_ask_price": 50.0, "best_bid_price": 49.95}
        portfolio_state = {"cash": 10000.0, "positions": {}}
        
        result = full_simulator.execute_action(
            raw_action, market_state, portfolio_state, "TEST", None
        )
        
        assert result.fill_details is None
        assert result.action_decode_result.action_type == "HOLD"
        assert result.order_request is None
        assert result.execution_stats['total_attempted'] == 0
    
    def test_rejected_action_flow(self, full_simulator):
        """Test flow with rejected action."""
        raw_action = (2, 1)  # SELL when no position
        
        market_state = {"best_ask_price": 50.0, "best_bid_price": 49.95}
        portfolio_state = {
            "cash": 10000.0,
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        result = full_simulator.execute_action(
            raw_action, market_state, portfolio_state, "TEST", None
        )
        
        assert result.fill_details is None
        assert result.action_decode_result.is_valid is False
        assert result.action_decode_result.rejection_reason == RejectionReason.NO_POSITION_TO_SELL
        assert result.order_request is None
        assert result.execution_stats['total_rejected'] == 1
    
    def test_metrics_integration(self, full_simulator):
        """Test metrics recording when integrator is present."""
        # Add mock metrics integrator
        metrics_integrator = Mock()
        full_simulator.metrics_integrator = metrics_integrator
        
        # Execute a successful order
        timestamp = datetime.now(timezone.utc)
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=50.0,
            ideal_bid_price=49.95,
            decision_timestamp=timestamp
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=1000.0,
            session_turnover=50000.0,
            time_of_day=0.5
        )
        
        fill = full_simulator.execute_order(order, context)
        
        # Verify metrics were recorded
        assert metrics_integrator.record_execution.called
        call_args = metrics_integrator.record_execution.call_args[0][0]
        assert 'quantity' in call_args
        assert 'price' in call_args
        assert 'commission' in call_args
        assert 'slippage_bps' in call_args
        assert 'latency_ms' in call_args


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator for edge case testing."""
        config = SimulationConfig()
        logger = logging.getLogger("test")
        np_random = np.random.default_rng(42)
        
        return ExecutionSimulator(
            logger=logger,
            simulation_config=config,
            np_random=np_random,
            market_simulator=Mock(),
            metrics_integrator=None
        )
    
    def test_action_decode_with_exception(self, simulator):
        """Test action decoding handles exceptions gracefully."""
        # Mock to raise exception
        simulator.logger.warning = Mock(side_effect=Exception("Test error"))
        
        result = simulator.decode_action("invalid")
        
        assert result.action_type == "HOLD"
        assert result.is_valid is False
        assert result.rejection_reason == RejectionReason.SYSTEM_ERROR
    
    def test_order_execution_with_exception(self, simulator):
        """Test order execution handles exceptions gracefully."""
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=100.0,
            ideal_ask_price=50.0,
            ideal_bid_price=49.95,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=1000.0,
            session_turnover=50000.0,
            time_of_day=0.5
        )
        
        # Mock to raise exception
        simulator._simulate_latency = Mock(side_effect=Exception("Latency error"))
        
        fill = simulator.execute_order(order, context)
        
        assert fill is None
        assert simulator.total_orders_rejected == 1
    
    def test_zero_latency_std_dev(self, simulator):
        """Test latency simulation with zero standard deviation."""
        simulator.latency_std_ms = 0.0
        
        for _ in range(10):
            latency = simulator._simulate_latency()
            assert latency == simulator.base_latency_ms
    
    def test_negative_latency_protection(self, simulator):
        """Test that negative latency is prevented."""
        simulator.base_latency_ms = 5.0
        simulator.latency_std_ms = 10.0  # High std dev
        
        for _ in range(100):
            latency = simulator._simulate_latency()
            assert latency >= 1.0  # Minimum 1ms
    
    def test_square_root_market_impact(self, simulator):
        """Test square root market impact model."""
        simulator.simulation_config.market_impact_model = "square_root"
        simulator.market_impact_coefficient = 0.001
        
        order = OrderRequest(
            asset_id="TEST",
            order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY,
            quantity=1000.0,
            ideal_ask_price=100.0,
            ideal_bid_price=99.95,
            decision_timestamp=datetime.now(timezone.utc)
        )
        
        context = ExecutionContext(
            market_state={},
            portfolio_state={},
            session_volume=10000.0,
            session_turnover=1000000.0,
            time_of_day=0.5
        )
        
        executed_price, slippage_bps = simulator._calculate_execution_price(
            order, context
        )
        
        # Should use square root of order value
        order_value = 1000.0 * 100.0  # $100,000
        expected_impact = (order_value ** 0.5) * 0.001
        
        assert slippage_bps > simulator.base_slippage_bps
    
    def test_buy_to_cover_short_position(self, simulator):
        """Test buying to cover a short position."""
        simulator.simulation_config.allow_shorting = True
        
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=0.5,
            raw_action=[1, 1],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 50.0,
            "best_bid_price": 49.95
        }
        
        portfolio_state = {
            "cash": 10000.0,
            "total_equity": 20000.0,
            "positions": {
                "TEST": {
                    "quantity": 100.0,  # Short 100 shares
                    "current_side": PositionSideEnum.SHORT
                }
            }
        }
        
        order = simulator.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is not None
        # Should buy to cover 100 shares + new position
        assert order.quantity > 100.0
    
    def test_position_size_limit(self, simulator):
        """Test position size is limited by max ratio."""
        simulator.simulation_config.max_position_value_ratio = 0.3
        
        action_result = ActionDecodeResult(
            action_type="BUY",
            size_float=1.0,  # Want full position
            raw_action=[1, 3],
            is_valid=True
        )
        
        market_state = {
            "best_ask_price": 100.0,
            "best_bid_price": 99.95
        }
        
        portfolio_state = {
            "cash": 50000.0,
            "total_equity": 100000.0,  # Max position = 30k
            "positions": {
                "TEST": {
                    "quantity": 0.0,
                    "current_side": PositionSideEnum.FLAT
                }
            }
        }
        
        order = simulator.validate_and_create_order(
            action_result, market_state, portfolio_state, "TEST", None
        )
        
        assert order is not None
        # default_position_value * size_float = 10000 * 1.0 = $10,000
        # This is less than the max allowed (30k), so we get $10k / $100 = 100 shares
        assert order.quantity == pytest.approx(100.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])