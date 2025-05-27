"""Comprehensive test suite for ExecutionSimulatorV2 integrated with MarketSimulatorV2.

This test suite focuses on:
- Integration with MarketSimulatorV2's execution capabilities
- Leveraging market simulator's slippage and latency models
- Order routing and validation
- Execution results and market impact
- Commission and fee calculations
- Market state awareness
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from simulators.execution_simulator import ExecutionSimulatorV2
from simulators.market_simulator_v2 import (
    MarketSimulatorV2, 
    MarketState, 
    ExecutionResult,
    OrderRequest,
    OrderType,
    OrderSide,
    OrderStatus
)
from simulators.portfolio_simulator import PortfolioSimulator, PortfolioState


class TestExecutionSimulatorV2:
    """Test ExecutionSimulatorV2 with MarketSimulatorV2 integration."""
    
    @pytest.fixture
    def execution_config(self):
        """Configuration for execution simulator."""
        return {
            'execution': {
                # Commission structure
                'commission': {
                    'per_share': 0.005,
                    'minimum': 1.0,
                    'maximum_pct': 0.5,  # Max 0.5% of trade value
                    'maker_rebate': -0.002,  # Rebate for providing liquidity
                    'tiered_rates': {
                        0: 0.005,      # First 1000 shares
                        1000: 0.004,   # Next 4000 shares  
                        5000: 0.003,   # Next 5000 shares
                        10000: 0.002   # Above 10000 shares
                    }
                },
                
                # Order validation
                'validation': {
                    'min_order_size': 100,
                    'max_order_size': 50000,
                    'max_position_size': 100000,
                    'reject_on_halt': True,
                    'reject_on_circuit_breaker': True,
                    'max_order_value': 500000,
                    'min_price': 0.01,
                    'max_price': 10000
                },
                
                # Risk checks
                'risk': {
                    'max_daily_loss': 5000,
                    'max_position_value': 250000,
                    'max_leverage': 2.0,
                    'concentration_limit': 0.4,  # Max 40% of portfolio
                    'fat_finger_threshold': 0.1  # 10% from last price
                },
                
                # Smart order routing
                'routing': {
                    'enabled': True,
                    'venue_preferences': ['NASDAQ', 'NYSE', 'ARCA'],
                    'dark_pool_enabled': False,
                    'price_improvement_seek': True
                }
            }
        }
    
    @pytest.fixture
    def mock_market_simulator(self):
        """Mock MarketSimulatorV2 with execution capabilities."""
        market_sim = Mock(spec=MarketSimulatorV2)
        
        # Default market state
        market_sim.get_market_state.return_value = MarketState(
            timestamp=pd.Timestamp('2025-03-27 09:30:00'),
            bid_price=10.00,
            ask_price=10.02,
            bid_size=5000,
            ask_size=5000,
            last_price=10.01,
            last_size=100,
            volume=50000,
            vwap=10.00,
            high=10.05,
            low=9.95,
            open=9.98,
            close=None,
            spread=0.02,
            spread_bps=20,
            imbalance_ratio=0.0,
            is_halted=False,
            is_auction=False
        )
        
        # Default execution result
        market_sim.execute_order.return_value = ExecutionResult(
            order_id='TEST001',
            symbol='MLGO',
            timestamp=pd.Timestamp('2025-03-27 09:30:00.100'),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_quantity=1000,
            executed_quantity=1000,
            requested_price=None,
            executed_price=10.025,
            slippage_bps=25,
            slippage_dollars=0.025,
            commission=5.0,
            fees=0.0,
            total_cost=10030.0,
            latency_ms=100,
            venue='NASDAQ',
            liquidity_flag='TAKER',
            status=OrderStatus.FILLED,
            rejection_reason=None,
            market_impact_bps=5,
            fill_rate=1.0
        )
        
        return market_sim
    
    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio simulator."""
        portfolio = Mock(spec=PortfolioSimulator)
        portfolio.get_state.return_value = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=100000,
            positions={},
            total_value=100000,
            buying_power=200000,  # With 2x margin
            margin_used=0,
            realized_pnl=0,
            unrealized_pnl=0,
            daily_pnl=0,
            total_commission=0,
            total_fees=0
        )
        return portfolio
    
    @pytest.fixture
    def execution_simulator(self, execution_config, mock_market_simulator, mock_portfolio):
        """Create ExecutionSimulatorV2 instance."""
        return ExecutionSimulatorV2(
            config=execution_config,
            market_simulator=mock_market_simulator,
            portfolio_simulator=mock_portfolio,
            logger=Mock()
        )
    
    def test_order_validation_basic(self, execution_simulator):
        """Test basic order validation rules."""
        portfolio_state = Mock(
            cash=100000,
            buying_power=200000,
            positions={}
        )
        
        market_state = Mock(
            ask_price=10.02,
            bid_price=10.00,
            is_halted=False,
            last_price=10.01
        )
        
        # Valid buy order
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is True
        assert reason is None
        
        # Order too small
        order.quantity = 50
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'minimum size' in reason.lower()
        
        # Order too large
        order.quantity = 60000
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'maximum size' in reason.lower()
        
        # Market halted
        market_state.is_halted = True
        order.quantity = 1000
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'halted' in reason.lower()
    
    def test_buying_power_validation(self, execution_simulator):
        """Test buying power and margin validation."""
        portfolio_state = Mock(
            cash=10000,
            buying_power=20000,  # 2x leverage
            positions={},
            margin_used=0
        )
        
        market_state = Mock(
            ask_price=10.02,
            is_halted=False
        )
        
        # Order within buying power
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1500,  # $15,030 cost
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is True
        
        # Order exceeds buying power
        order.quantity = 2500  # $25,050 cost
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'buying power' in reason.lower()
    
    def test_position_limit_validation(self, execution_simulator):
        """Test position size limits."""
        portfolio_state = Mock(
            cash=100000,
            buying_power=200000,
            positions={
                'MLGO': Mock(quantity=80000, side='long')
            }
        )
        
        market_state = Mock(
            ask_price=10.02,
            is_halted=False
        )
        
        # Would exceed position limit
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=30000,  # Total would be 110k
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'position limit' in reason.lower()
        
        # Within limit
        order.quantity = 15000  # Total would be 95k
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is True
    
    def test_risk_check_validation(self, execution_simulator):
        """Test risk management validation."""
        # Test 1: Daily loss limit
        portfolio_state = Mock(
            cash=95000,
            buying_power=190000,
            positions={},
            daily_pnl=-4500  # Already down $4500
        )
        
        market_state = Mock(
            bid_price=10.00,
            is_halted=False
        )
        
        # Selling position that could realize more loss
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.SELL,
            quantity=1000,
            order_type=OrderType.MARKET
        )
        
        portfolio_state.positions = {
            'MLGO': Mock(quantity=1000, avg_price=11.0)  # $1000 loss if sold
        }
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'daily loss limit' in reason.lower()
        
        # Test 2: Concentration limit
        portfolio_state = Mock(
            cash=60000,
            buying_power=120000,
            total_value=100000,
            positions={'OTHER': Mock(value=40000)},
            daily_pnl=0
        )
        
        market_state.ask_price = 10.02
        
        # Would create 45% concentration
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=4500,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'concentration' in reason.lower()
        
        # Test 3: Fat finger check
        market_state = Mock(
            last_price=10.00,
            ask_price=11.50,  # 15% above last
            is_halted=False
        )
        
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.LIMIT,
            limit_price=11.50
        )
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        assert is_valid is False
        assert 'fat finger' in reason.lower()
    
    def test_commission_calculation_basic(self, execution_simulator):
        """Test basic commission calculation."""
        # Simple per-share commission
        commission = execution_simulator.calculate_commission(
            executed_quantity=1000,
            executed_price=10.0,
            liquidity_flag='TAKER'
        )
        
        expected = max(1000 * 0.005, 1.0)  # $5.00
        assert commission == expected
        
        # Small order - minimum commission
        commission = execution_simulator.calculate_commission(
            executed_quantity=100,
            executed_price=10.0,
            liquidity_flag='TAKER'
        )
        
        assert commission == 1.0  # Minimum
        
        # Maker rebate
        commission = execution_simulator.calculate_commission(
            executed_quantity=1000,
            executed_price=10.0,
            liquidity_flag='MAKER'
        )
        
        assert commission == -2.0  # Rebate
    
    def test_commission_calculation_tiered(self, execution_simulator):
        """Test tiered commission structure."""
        # Large order spanning multiple tiers
        commission = execution_simulator.calculate_commission(
            executed_quantity=12000,
            executed_price=10.0,
            liquidity_flag='TAKER',
            use_tiered=True
        )
        
        # First 1000 @ 0.005 = $5
        # Next 4000 @ 0.004 = $16
        # Next 5000 @ 0.003 = $15
        # Last 2000 @ 0.002 = $4
        # Total = $40
        expected = 5 + 16 + 15 + 4
        assert abs(commission - expected) < 0.01
        
        # Check maximum percentage cap
        commission = execution_simulator.calculate_commission(
            executed_quantity=10,
            executed_price=1000.0,  # $10k trade value
            liquidity_flag='TAKER'
        )
        
        # Would be $0.05 but capped at 0.5% of $10k = $50
        # Since $0.05 < $50 and < $1 minimum, use $1
        assert commission == 1.0
    
    def test_order_routing_logic(self, execution_simulator):
        """Test smart order routing decisions."""
        market_state = Mock(
            bid_price=10.00,
            ask_price=10.02,
            bid_size=1000,
            ask_size=1000,
            venue_quotes={
                'NASDAQ': {'bid': 10.00, 'ask': 10.02, 'bid_size': 500, 'ask_size': 500},
                'NYSE': {'bid': 9.99, 'ask': 10.01, 'bid_size': 1000, 'ask_size': 1000},
                'ARCA': {'bid': 10.00, 'ask': 10.03, 'bid_size': 2000, 'ask_size': 2000}
            }
        )
        
        # Buy order - should route to best ask (NYSE)
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=800,
            order_type=OrderType.MARKET
        )
        
        venue = execution_simulator.determine_routing(order, market_state)
        assert venue == 'NYSE'  # Best ask price
        
        # Large buy order - consider liquidity
        order.quantity = 1500
        venue = execution_simulator.determine_routing(order, market_state)
        assert venue == 'ARCA'  # More liquidity despite worse price
        
        # Sell order - should route to best bid
        order.side = OrderSide.SELL
        order.quantity = 800
        venue = execution_simulator.determine_routing(order, market_state)
        assert venue in ['NASDAQ', 'ARCA']  # Both have 10.00 bid
    
    def test_execution_flow_market_order(self, execution_simulator, mock_market_simulator):
        """Test complete market order execution flow."""
        portfolio_state = Mock(
            cash=100000,
            buying_power=200000,
            positions={},
            daily_pnl=0
        )
        
        # Execute buy order
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.MARKET
        )
        
        result = execution_simulator.execute_order(order, portfolio_state)
        
        # Verify market simulator was called
        mock_market_simulator.execute_order.assert_called_once_with(order)
        
        # Check result
        assert result.status == OrderStatus.FILLED
        assert result.executed_quantity == 1000
        assert result.executed_price == 10.025
        assert result.commission == 5.0
        assert result.slippage_bps == 25
        
        # Check metrics recorded
        assert execution_simulator.get_execution_metrics()['total_orders'] == 1
        assert execution_simulator.get_execution_metrics()['filled_orders'] == 1
    
    def test_execution_flow_with_rejection(self, execution_simulator, mock_market_simulator):
        """Test order rejection handling."""
        # Configure rejection
        mock_market_simulator.execute_order.return_value = ExecutionResult(
            order_id='TEST002',
            symbol='MLGO',
            timestamp=pd.Timestamp.now(),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_quantity=5000,
            executed_quantity=0,
            executed_price=None,
            status=OrderStatus.REJECTED,
            rejection_reason='INSUFFICIENT_LIQUIDITY'
        )
        
        portfolio_state = Mock(buying_power=200000, positions={})
        
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=5000,
            order_type=OrderType.MARKET
        )
        
        result = execution_simulator.execute_order(order, portfolio_state)
        
        assert result.status == OrderStatus.REJECTED
        assert result.rejection_reason == 'INSUFFICIENT_LIQUIDITY'
        assert result.executed_quantity == 0
        
        # Check metrics
        metrics = execution_simulator.get_execution_metrics()
        assert metrics['rejected_orders'] == 1
        assert metrics['rejection_reasons']['INSUFFICIENT_LIQUIDITY'] == 1
    
    def test_execution_flow_partial_fill(self, execution_simulator, mock_market_simulator):
        """Test partial fill handling."""
        # Configure partial fill
        mock_market_simulator.execute_order.return_value = ExecutionResult(
            order_id='TEST003',
            symbol='MLGO',
            timestamp=pd.Timestamp.now(),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_quantity=5000,
            executed_quantity=3000,  # Partial
            executed_price=10.025,
            status=OrderStatus.PARTIAL,
            fill_rate=0.6
        )
        
        portfolio_state = Mock(buying_power=200000, positions={})
        
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=5000,
            order_type=OrderType.MARKET
        )
        
        result = execution_simulator.execute_order(order, portfolio_state)
        
        assert result.status == OrderStatus.PARTIAL
        assert result.executed_quantity == 3000
        assert result.fill_rate == 0.6
        
        # Check if remainder is tracked
        assert execution_simulator.has_pending_order('MLGO')
        pending = execution_simulator.get_pending_order('MLGO')
        assert pending.remaining_quantity == 2000
    
    def test_market_impact_awareness(self, execution_simulator, mock_market_simulator):
        """Test market impact calculation and awareness."""
        # Small order - minimal impact
        market_state = Mock(
            volume=100000,
            avg_trade_size=500,
            bid_size=5000,
            ask_size=5000
        )
        
        impact = execution_simulator.estimate_market_impact(
            order_size=200,
            side=OrderSide.BUY,
            market_state=market_state
        )
        
        assert impact['expected_impact_bps'] < 5  # Less than 5 bps
        assert impact['impact_category'] == 'minimal'
        
        # Large order - significant impact
        impact = execution_simulator.estimate_market_impact(
            order_size=10000,
            side=OrderSide.BUY,
            market_state=market_state
        )
        
        assert impact['expected_impact_bps'] > 20  # More than 20 bps
        assert impact['impact_category'] == 'significant'
        assert 'split_recommendation' in impact  # Should recommend splitting
    
    def test_execution_analytics(self, execution_simulator, mock_market_simulator):
        """Test execution analytics and reporting."""
        # Execute multiple orders
        orders = [
            OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET),
            OrderRequest('MLGO', OrderSide.BUY, 500, OrderType.MARKET),
            OrderRequest('MLGO', OrderSide.SELL, 1500, OrderType.MARKET),
        ]
        
        portfolio_state = Mock(
            buying_power=200000,
            positions={'MLGO': Mock(quantity=1500)}
        )
        
        for order in orders:
            execution_simulator.execute_order(order, portfolio_state)
        
        # Get analytics
        analytics = execution_simulator.get_execution_analytics()
        
        assert analytics['total_volume'] == 3000  # 1000 + 500 + 1500
        assert analytics['buy_volume'] == 1500
        assert analytics['sell_volume'] == 1500
        assert analytics['avg_slippage_bps'] > 0
        assert analytics['total_commission'] > 0
        assert analytics['fill_rate'] == 1.0
        assert 'venue_distribution' in analytics
        assert 'hourly_volume' in analytics
    
    def test_latency_tracking(self, execution_simulator, mock_market_simulator):
        """Test execution latency tracking."""
        # Configure variable latency
        latencies = [50, 100, 150, 200, 75]
        execution_results = []
        
        for i, latency in enumerate(latencies):
            result = ExecutionResult(
                order_id=f'TEST{i:03d}',
                symbol='MLGO',
                timestamp=pd.Timestamp.now(),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                requested_quantity=100,
                executed_quantity=100,
                executed_price=10.0,
                status=OrderStatus.FILLED,
                latency_ms=latency
            )
            execution_results.append(result)
        
        mock_market_simulator.execute_order.side_effect = execution_results
        
        # Execute orders
        portfolio_state = Mock(buying_power=200000, positions={})
        for i in range(5):
            order = OrderRequest('MLGO', OrderSide.BUY, 100, OrderType.MARKET)
            execution_simulator.execute_order(order, portfolio_state)
        
        # Check latency statistics
        latency_stats = execution_simulator.get_latency_statistics()
        
        assert latency_stats['mean_latency_ms'] == np.mean(latencies)
        assert latency_stats['median_latency_ms'] == np.median(latencies)
        assert latency_stats['p95_latency_ms'] == np.percentile(latencies, 95)
        assert latency_stats['max_latency_ms'] == 200
        assert latency_stats['min_latency_ms'] == 50
    
    def test_execution_history(self, execution_simulator):
        """Test execution history tracking."""
        # Execute several orders
        portfolio_state = Mock(buying_power=200000, positions={})
        
        orders = [
            OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET),
            OrderRequest('MLGO', OrderSide.SELL, 500, OrderType.MARKET),
            OrderRequest('MLGO', OrderSide.BUY, 2000, OrderType.MARKET),
        ]
        
        for order in orders:
            execution_simulator.execute_order(order, portfolio_state)
        
        # Get history
        history = execution_simulator.get_execution_history()
        
        assert len(history) == 3
        assert all('timestamp' in h for h in history)
        assert all('order_id' in h for h in history)
        assert all('execution_result' in h for h in history)
        
        # Filter history
        buy_history = execution_simulator.get_execution_history(
            side=OrderSide.BUY
        )
        assert len(buy_history) == 2
        
        # Get history for time range
        recent_history = execution_simulator.get_execution_history(
            start_time=pd.Timestamp.now() - pd.Timedelta(minutes=1)
        )
        assert len(recent_history) == 3
    
    def test_multi_symbol_execution(self, execution_simulator, mock_market_simulator):
        """Test execution across multiple symbols."""
        symbols = ['MLGO', 'AAPL', 'TSLA']
        portfolio_state = Mock(buying_power=500000, positions={})
        
        # Execute orders for different symbols
        for symbol in symbols:
            order = OrderRequest(symbol, OrderSide.BUY, 1000, OrderType.MARKET)
            execution_simulator.execute_order(order, portfolio_state)
        
        # Check per-symbol metrics
        metrics = execution_simulator.get_execution_metrics()
        
        assert metrics['symbols_traded'] == 3
        for symbol in symbols:
            assert symbol in metrics['per_symbol']
            assert metrics['per_symbol'][symbol]['volume'] == 1000
    
    def test_execution_callbacks(self, execution_simulator):
        """Test execution callback mechanisms."""
        callbacks_received = []
        
        def on_execution(result: ExecutionResult):
            callbacks_received.append(result)
        
        execution_simulator.register_callback(on_execution)
        
        # Execute order
        portfolio_state = Mock(buying_power=100000, positions={})
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        
        result = execution_simulator.execute_order(order, portfolio_state)
        
        # Check callback was triggered
        assert len(callbacks_received) == 1
        assert callbacks_received[0] == result
    
    def test_execution_state_persistence(self, execution_simulator):
        """Test saving and loading execution state."""
        # Execute some orders
        portfolio_state = Mock(buying_power=200000, positions={})
        
        for i in range(5):
            order = OrderRequest('MLGO', OrderSide.BUY, 100 * (i + 1), OrderType.MARKET)
            execution_simulator.execute_order(order, portfolio_state)
        
        # Save state
        state = execution_simulator.get_state()
        
        assert 'execution_history' in state
        assert 'metrics' in state
        assert 'pending_orders' in state
        assert len(state['execution_history']) == 5
        
        # Create new simulator and restore state
        new_simulator = ExecutionSimulatorV2(
            config=execution_simulator.config,
            market_simulator=execution_simulator.market_simulator,
            portfolio_simulator=execution_simulator.portfolio_simulator
        )
        
        new_simulator.restore_state(state)
        
        # Verify state restored
        assert len(new_simulator.get_execution_history()) == 5
        assert new_simulator.get_execution_metrics()['total_orders'] == 5


class TestExecutionEdgeCases:
    """Test edge cases and error handling in execution."""
    
    @pytest.fixture
    def execution_simulator(self):
        """Create execution simulator with mocked dependencies."""
        config = {'execution': {
            'commission': {'per_share': 0.005, 'minimum': 1.0},
            'validation': {'min_order_size': 100, 'reject_on_halt': True}
        }}
        
        return ExecutionSimulatorV2(
            config=config,
            market_simulator=Mock(spec=MarketSimulatorV2),
            portfolio_simulator=Mock(spec=PortfolioSimulator),
            logger=Mock()
        )
    
    def test_zero_quantity_order(self, execution_simulator):
        """Test handling of zero quantity orders."""
        order = OrderRequest('MLGO', OrderSide.BUY, 0, OrderType.MARKET)
        portfolio_state = Mock()
        market_state = Mock()
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        
        assert is_valid is False
        assert 'zero quantity' in reason.lower()
    
    def test_negative_price_order(self, execution_simulator):
        """Test handling of negative price orders."""
        order = OrderRequest(
            symbol='MLGO',
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=-10.0
        )
        
        portfolio_state = Mock()
        market_state = Mock()
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        
        assert is_valid is False
        assert 'negative price' in reason.lower()
    
    def test_market_closed_execution(self, execution_simulator):
        """Test execution attempts when market is closed."""
        execution_simulator.market_simulator.is_market_open.return_value = False
        
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        portfolio_state = Mock(buying_power=100000)
        
        result = execution_simulator.execute_order(order, portfolio_state)
        
        assert result.status == OrderStatus.REJECTED
        assert 'market closed' in result.rejection_reason.lower()
    
    def test_concurrent_order_handling(self, execution_simulator):
        """Test handling of concurrent orders for same symbol."""
        # Set pending order
        execution_simulator._pending_orders['MLGO'] = Mock(
            remaining_quantity=1000,
            side=OrderSide.BUY
        )
        
        # Try to place opposite side order
        order = OrderRequest('MLGO', OrderSide.SELL, 500, OrderType.MARKET)
        portfolio_state = Mock()
        market_state = Mock()
        
        is_valid, reason = execution_simulator.validate_order(
            order, portfolio_state, market_state
        )
        
        assert is_valid is False
        assert 'pending order' in reason.lower()
    
    def test_execution_during_circuit_breaker(self, execution_simulator):
        """Test execution during circuit breaker."""
        market_state = Mock(
            is_halted=False,
            circuit_breaker_triggered=True,
            circuit_breaker_level=1  # Level 1 halt
        )
        
        execution_simulator.market_simulator.get_market_state.return_value = market_state
        
        order = OrderRequest('MLGO', OrderSide.BUY, 1000, OrderType.MARKET)
        portfolio_state = Mock(buying_power=100000)
        
        result = execution_simulator.execute_order(order, portfolio_state)
        
        assert result.status == OrderStatus.REJECTED
        assert 'circuit breaker' in result.rejection_reason.lower()