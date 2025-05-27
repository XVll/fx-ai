"""Test suite for order management and execution handlers.

This focuses on:
- Order lifecycle management
- Order state transitions
- Pending order handling
- Order modification and cancellation
- Fill tracking and aggregation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
from enum import Enum

from simulators.execution_simulator import (
    OrderManager,
    OrderState,
    OrderLifecycle,
    FillTracker,
    ExecutionHandler
)
from simulators.market_simulator_v2 import (
    OrderRequest,
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionResult
)


class TestOrderLifecycle:
    """Test order lifecycle and state management."""
    
    @pytest.fixture
    def order_manager(self):
        """Create OrderManager instance."""
        return OrderManager(logger=Mock())
    
    def test_order_creation_and_tracking(self, order_manager):
        """Test order creation and initial tracking."""
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.MARKET
        )
        
        # Submit order
        order_id = order_manager.submit_order(order)
        
        assert order_id is not None
        assert order_manager.has_order(order_id)
        
        # Check order state
        state = order_manager.get_order_state(order_id)
        assert state.status == OrderStatus.PENDING
        assert state.original_quantity == 1000
        assert state.filled_quantity == 0
        assert state.remaining_quantity == 1000
    
    def test_order_state_transitions(self, order_manager):
        """Test valid order state transitions."""
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        order_id = order_manager.submit_order(order)
        
        # Pending -> Working
        order_manager.update_order_state(order_id, OrderStatus.WORKING)
        state = order_manager.get_order_state(order_id)
        assert state.status == OrderStatus.WORKING
        
        # Working -> Partial
        fill = ExecutionResult(
            order_id=order_id,
            executed_quantity=300,
            executed_price=10.05,
            status=OrderStatus.PARTIAL
        )
        order_manager.process_fill(order_id, fill)
        
        state = order_manager.get_order_state(order_id)
        assert state.status == OrderStatus.PARTIAL
        assert state.filled_quantity == 300
        assert state.remaining_quantity == 700
        
        # Partial -> Filled
        fill2 = ExecutionResult(
            order_id=order_id,
            executed_quantity=700,
            executed_price=10.06,
            status=OrderStatus.FILLED
        )
        order_manager.process_fill(order_id, fill2)
        
        state = order_manager.get_order_state(order_id)
        assert state.status == OrderStatus.FILLED
        assert state.filled_quantity == 1000
        assert state.remaining_quantity == 0
    
    def test_invalid_state_transitions(self, order_manager):
        """Test prevention of invalid state transitions."""
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        order_id = order_manager.submit_order(order)
        
        # Cancel order
        order_manager.cancel_order(order_id)
        
        # Try to fill cancelled order
        with pytest.raises(ValueError, match="Cannot fill cancelled order"):
            fill = ExecutionResult(
                order_id=order_id,
                executed_quantity=1000,
                executed_price=10.0
            )
            order_manager.process_fill(order_id, fill)
    
    def test_order_modification(self, order_manager):
        """Test order modification capabilities."""
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.LIMIT,
            limit_price=10.00
        )
        order_id = order_manager.submit_order(order)
        
        # Modify quantity
        success = order_manager.modify_order(
            order_id,
            new_quantity=1500
        )
        assert success is True
        
        state = order_manager.get_order_state(order_id)
        assert state.original_quantity == 1500
        assert state.remaining_quantity == 1500
        
        # Modify price
        success = order_manager.modify_order(
            order_id,
            new_price=10.05
        )
        assert success is True
        
        # Cannot modify after partial fill
        fill = ExecutionResult(
            order_id=order_id,
            executed_quantity=500,
            status=OrderStatus.PARTIAL
        )
        order_manager.process_fill(order_id, fill)
        
        success = order_manager.modify_order(
            order_id,
            new_quantity=2000
        )
        assert success is False  # Can't increase quantity after fill
    
    def test_order_cancellation(self, order_manager):
        """Test order cancellation logic."""
        # Test 1: Cancel pending order
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        order_id = order_manager.submit_order(order)
        
        success = order_manager.cancel_order(order_id)
        assert success is True
        
        state = order_manager.get_order_state(order_id)
        assert state.status == OrderStatus.CANCELLED
        
        # Test 2: Cancel partial fill
        order2 = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        order_id2 = order_manager.submit_order(order2)
        
        fill = ExecutionResult(
            order_id=order_id2,
            executed_quantity=300,
            status=OrderStatus.PARTIAL
        )
        order_manager.process_fill(order_id2, fill)
        
        success = order_manager.cancel_order(order_id2)
        assert success is True
        
        state = order_manager.get_order_state(order_id2)
        assert state.status == OrderStatus.CANCELLED
        assert state.filled_quantity == 300  # Keeps filled portion
        
        # Test 3: Cannot cancel filled order
        order3 = OrderRequest('MLGO', OrderSide.BUY, 100, OrderType.MARKET)
        order_id3 = order_manager.submit_order(order3)
        
        fill = ExecutionResult(
            order_id=order_id3,
            executed_quantity=100,
            status=OrderStatus.FILLED
        )
        order_manager.process_fill(order_id3, fill)
        
        success = order_manager.cancel_order(order_id3)
        assert success is False
    
    def test_order_expiration(self, order_manager):
        """Test order expiration handling."""
        # Day order
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.LIMIT,
            limit_price=10.00,
            time_in_force='DAY'
        )
        order_id = order_manager.submit_order(order)
        
        # Expire at end of day
        order_manager.expire_day_orders()
        
        state = order_manager.get_order_state(order_id)
        assert state.status == OrderStatus.EXPIRED
        
        # GTC order should not expire
        order_gtc = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.LIMIT,
            limit_price=10.00,
            time_in_force='GTC'
        )
        order_id_gtc = order_manager.submit_order(order_gtc)
        
        order_manager.expire_day_orders()
        
        state_gtc = order_manager.get_order_state(order_id_gtc)
        assert state_gtc.status == OrderStatus.PENDING  # Still active


class TestFillTracking:
    """Test fill tracking and aggregation."""
    
    @pytest.fixture
    def fill_tracker(self):
        """Create FillTracker instance."""
        return FillTracker()
    
    def test_single_fill_tracking(self, fill_tracker):
        """Test tracking of single fill."""
        fill = ExecutionResult(
            order_id='ORD001',
            symbol='MLGO',
            timestamp=pd.Timestamp.now(),
            side=OrderSide.BUY,
            executed_quantity=1000,
            executed_price=10.05,
            commission=5.0,
            fees=0.10,
            slippage_dollars=25.0
        )
        
        fill_tracker.add_fill(fill)
        
        # Get fill details
        details = fill_tracker.get_fill_details('ORD001')
        assert details['total_quantity'] == 1000
        assert details['average_price'] == 10.05
        assert details['total_commission'] == 5.0
        assert details['total_fees'] == 0.10
        assert details['total_slippage'] == 25.0
        assert details['fill_count'] == 1
    
    def test_multiple_fill_aggregation(self, fill_tracker):
        """Test aggregation of multiple fills."""
        fills = [
            ExecutionResult(
                order_id='ORD002',
                symbol='MLGO',
                timestamp=pd.Timestamp.now(),
                side=OrderSide.BUY,
                executed_quantity=500,
                executed_price=10.00,
                commission=2.50,
                fees=0.05
            ),
            ExecutionResult(
                order_id='ORD002',
                symbol='MLGO',
                timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=1),
                side=OrderSide.BUY,
                executed_quantity=300,
                executed_price=10.10,
                commission=1.50,
                fees=0.03
            ),
            ExecutionResult(
                order_id='ORD002',
                symbol='MLGO',
                timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=2),
                side=OrderSide.BUY,
                executed_quantity=200,
                executed_price=10.20,
                commission=1.00,
                fees=0.02
            )
        ]
        
        for fill in fills:
            fill_tracker.add_fill(fill)
        
        details = fill_tracker.get_fill_details('ORD002')
        
        # Check aggregation
        assert details['total_quantity'] == 1000
        assert details['fill_count'] == 3
        
        # Weighted average price
        expected_avg = (500*10.00 + 300*10.10 + 200*10.20) / 1000
        assert abs(details['average_price'] - expected_avg) < 0.001
        
        # Total costs
        assert details['total_commission'] == 5.0
        assert details['total_fees'] == 0.10
    
    def test_fill_timeline(self, fill_tracker):
        """Test fill timeline tracking."""
        base_time = pd.Timestamp.now()
        
        fills = []
        for i in range(5):
            fill = ExecutionResult(
                order_id='ORD003',
                symbol='MLGO',
                timestamp=base_time + pd.Timedelta(seconds=i*10),
                side=OrderSide.BUY,
                executed_quantity=200,
                executed_price=10.00 + i*0.01
            )
            fills.append(fill)
            fill_tracker.add_fill(fill)
        
        # Get timeline
        timeline = fill_tracker.get_fill_timeline('ORD003')
        
        assert len(timeline) == 5
        assert timeline[0]['timestamp'] == base_time
        assert timeline[-1]['timestamp'] == base_time + pd.Timedelta(seconds=40)
        
        # Check cumulative quantities
        assert timeline[0]['cumulative_quantity'] == 200
        assert timeline[2]['cumulative_quantity'] == 600
        assert timeline[4]['cumulative_quantity'] == 1000
    
    def test_fill_statistics(self, fill_tracker):
        """Test fill statistics calculation."""
        # Add various fills
        orders = ['ORD004', 'ORD005', 'ORD006']
        
        for order_id in orders:
            for i in range(3):
                fill = ExecutionResult(
                    order_id=order_id,
                    symbol='MLGO',
                    timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=i),
                    side=OrderSide.BUY,
                    executed_quantity=100 + i*50,
                    executed_price=10.00 + i*0.05,
                    slippage_bps=5 + i*2
                )
                fill_tracker.add_fill(fill)
        
        # Get statistics
        stats = fill_tracker.get_statistics()
        
        assert stats['total_orders'] == 3
        assert stats['total_fills'] == 9
        assert stats['average_fills_per_order'] == 3
        assert stats['total_volume'] > 0
        assert 'average_slippage_bps' in stats
        assert 'fill_rate_distribution' in stats


class TestExecutionHandler:
    """Test high-level execution handler."""
    
    @pytest.fixture
    def execution_handler(self):
        """Create ExecutionHandler with dependencies."""
        return ExecutionHandler(
            order_manager=Mock(spec=OrderManager),
            fill_tracker=Mock(spec=FillTracker),
            market_simulator=Mock(),
            portfolio_simulator=Mock(),
            logger=Mock()
        )
    
    def test_action_to_order_conversion(self, execution_handler):
        """Test converting actions to orders."""
        portfolio_state = Mock(
            buying_power=100000,
            positions={'MLGO': Mock(quantity=1000, side='long')}
        )
        
        market_state = Mock(
            last_price=10.0,
            bid_price=9.99,
            ask_price=10.01
        )
        
        # Buy action
        order = execution_handler.create_order_from_action(
            action_type='buy',
            position_size_fraction=0.25,
            symbol='MLGO',
            portfolio_state=portfolio_state,
            market_state=market_state
        )
        
        assert order.side == OrderSide.BUY
        assert order.quantity == 2500  # 25% of $100k / $10
        assert order.order_type == OrderType.MARKET
        
        # Sell action (close position)
        order = execution_handler.create_order_from_action(
            action_type='sell',
            position_size_fraction=1.0,
            symbol='MLGO',
            portfolio_state=portfolio_state,
            market_state=market_state
        )
        
        assert order.side == OrderSide.SELL
        assert order.quantity == 1000  # Full position
        
        # Hold action
        order = execution_handler.create_order_from_action(
            action_type='hold',
            position_size_fraction=0.5,
            symbol='MLGO',
            portfolio_state=portfolio_state,
            market_state=market_state
        )
        
        assert order is None  # No order for hold
    
    def test_position_sizing_logic(self, execution_handler):
        """Test position sizing calculations."""
        portfolio_state = Mock(
            buying_power=50000,
            total_value=100000,
            max_position_pct=0.3  # 30% max per position
        )
        
        market_state = Mock(last_price=20.0)
        
        # Test different size fractions with constraints
        test_cases = [
            (0.25, 625),   # 25% of buying power
            (0.50, 1250),  # 50% of buying power
            (1.00, 1500),  # Would be 2500 but capped at 30% of portfolio
        ]
        
        for fraction, expected_shares in test_cases:
            shares = execution_handler.calculate_position_size(
                size_fraction=fraction,
                symbol='MLGO',
                portfolio_state=portfolio_state,
                market_state=market_state
            )
            assert shares == expected_shares
    
    def test_order_urgency_classification(self, execution_handler):
        """Test order urgency classification."""
        # Momentum breakout - urgent
        context = Mock(
            pattern='breakout',
            phase='acceleration',
            momentum_score=0.9
        )
        
        urgency = execution_handler.classify_order_urgency(
            order_type='buy',
            market_context=context
        )
        
        assert urgency == 'high'
        assert execution_handler.get_order_parameters(urgency)['aggressive'] is True
        
        # Consolidation - not urgent
        context.pattern = 'consolidation'
        context.momentum_score = 0.3
        
        urgency = execution_handler.classify_order_urgency(
            order_type='buy',
            market_context=context
        )
        
        assert urgency == 'low'
        assert execution_handler.get_order_parameters(urgency)['aggressive'] is False
    
    def test_execution_retry_logic(self, execution_handler):
        """Test execution retry on rejection."""
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        
        # First attempt fails
        execution_handler.order_manager.submit_order.return_value = 'ORD001'
        execution_handler.market_simulator.execute_order.return_value = ExecutionResult(
            order_id='ORD001',
            status=OrderStatus.REJECTED,
            rejection_reason='INSUFFICIENT_LIQUIDITY'
        )
        
        # Execute with retry
        result = execution_handler.execute_with_retry(
            order,
            max_retries=3,
            retry_delay_ms=100
        )
        
        # Should retry with modified parameters
        assert execution_handler.market_simulator.execute_order.call_count <= 3
        
        # Check if order was split or modified
        retry_calls = execution_handler.market_simulator.execute_order.call_args_list
        if len(retry_calls) > 1:
            # Later attempts should have smaller size
            assert retry_calls[1][0][0].quantity < order.quantity
    
    def test_batch_order_execution(self, execution_handler):
        """Test batch order execution."""
        orders = [
            OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET),
            OrderRequest('AAPL', OrderSide.BUY, 500, OrderType.MARKET),
            OrderRequest('TSLA', OrderSide.SELL, 200, OrderType.MARKET),
        ]
        
        # Execute batch
        results = execution_handler.execute_batch(
            orders,
            parallel=True,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all('order_id' in r for r in results)
        assert all('status' in r for r in results)
        
        # Check execution order for parallel execution
        if results[0]['status'] == OrderStatus.FILLED:
            # Some orders may have executed simultaneously
            timestamps = [r['timestamp'] for r in results]
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            assert any(diff.total_seconds() < 0.1 for diff in time_diffs)  # Some concurrent
    
    def test_conditional_order_execution(self, execution_handler):
        """Test conditional order execution."""
        # Stop loss order
        stop_order = ConditionalOrder(
            base_order=OrderRequest('MLGO', OrderSide.SELL, 1000, OrderType.MARKET),
            condition_type='stop_loss',
            trigger_price=9.50,
            symbol='MLGO'
        )
        
        execution_handler.add_conditional_order(stop_order)
        
        # Price doesn't hit stop
        market_state = Mock(last_price=10.00)
        triggered = execution_handler.check_conditional_orders(market_state)
        assert len(triggered) == 0
        
        # Price hits stop
        market_state.last_price = 9.45
        triggered = execution_handler.check_conditional_orders(market_state)
        assert len(triggered) == 1
        assert triggered[0] == stop_order
        
        # Order should be removed after triggering
        assert not execution_handler.has_conditional_order(stop_order.id)
    
    def test_execution_performance_metrics(self, execution_handler):
        """Test execution performance tracking."""
        # Simulate various executions
        executions = [
            {'slippage_bps': 5, 'latency_ms': 50, 'fill_rate': 1.0},
            {'slippage_bps': 10, 'latency_ms': 100, 'fill_rate': 0.8},
            {'slippage_bps': 3, 'latency_ms': 75, 'fill_rate': 1.0},
            {'slippage_bps': 15, 'latency_ms': 150, 'fill_rate': 0.6},
        ]
        
        for exec_data in executions:
            execution_handler.record_execution_metrics(exec_data)
        
        # Get performance report
        report = execution_handler.get_performance_report()
        
        assert report['average_slippage_bps'] == np.mean([e['slippage_bps'] for e in executions])
        assert report['average_latency_ms'] == np.mean([e['latency_ms'] for e in executions])
        assert report['average_fill_rate'] == np.mean([e['fill_rate'] for e in executions])
        assert 'slippage_distribution' in report
        assert 'latency_percentiles' in report
        assert report['total_executions'] == 4