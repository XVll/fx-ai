"""Test suite for environment integration with execution and portfolio simulators.

This covers:
- Action decoding and execution delegation
- Portfolio state management
- Position validation
- Trade execution flow
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Optional

from envs.environment_simulator import (
    EnvironmentSimulator,
    ActionDecoder,
    ExecutionHandler
)
from simulators.execution_simulator import (
    ExecutionSimulator,
    ExecutionResult,
    OrderRequest,
    OrderType,
    OrderSide
)
from simulators.portfolio_simulator import (
    PortfolioSimulator,
    PortfolioState,
    Position,
    Trade
)
from simulators.market_simulator_v2 import MarketState


class TestExecutionIntegration:
    """Test integration between environment and execution simulator."""
    
    @pytest.fixture
    def execution_config(self):
        """Execution configuration."""
        return {
            'execution': {
                'default_latency_ms': 100,
                'latency_variance_ms': 20,
                'commission_per_share': 0.005,
                'min_commission': 1.0,
                'maker_rebate': -0.002,  # Negative commission for liquidity provision
                'slippage_model': 'linear',
                'market_impact_model': 'square_root',
                'reject_on_halt': True,
                'max_position_size': 10000,
                'min_order_size': 100
            },
            'portfolio': {
                'initial_capital': 100000,
                'margin_multiplier': 2.0,
                'max_position_value': 50000,
                'risk_check_enabled': True
            }
        }
    
    @pytest.fixture
    def mock_market_state(self):
        """Create mock market state."""
        return MarketState(
            timestamp=pd.Timestamp.now(),
            bid_price=10.00,
            ask_price=10.02,
            bid_size=5000,
            ask_size=5000,
            last_price=10.01,
            last_size=100,
            volume=100000,
            is_halted=False,
            spread=0.02
        )
    
    @pytest.fixture
    def mock_portfolio_state(self):
        """Create mock portfolio state."""
        return PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=100000,
            positions={},
            total_value=100000,
            buying_power=100000,
            margin_used=0,
            realized_pnl=0,
            unrealized_pnl=0,
            total_pnl=0,
            trade_count=0,
            win_count=0,
            loss_count=0
        )
    
    @pytest.fixture
    def execution_handler(self, execution_config):
        """Create ExecutionHandler."""
        return ExecutionHandler(
            config=execution_config,
            execution_simulator=Mock(spec=ExecutionSimulator),
            portfolio_simulator=Mock(spec=PortfolioSimulator),
            logger=Mock()
        )
    
    def test_action_to_order_conversion(self, execution_handler):
        """Test converting environment actions to order requests."""
        # Test different action combinations
        test_cases = [
            # (action_type, position_size, current_position, expected_order)
            ('buy', 0.25, None, OrderRequest('buy', 2500, OrderType.MARKET)),
            ('buy', 0.50, None, OrderRequest('buy', 5000, OrderType.MARKET)),
            ('buy', 1.00, None, OrderRequest('buy', 10000, OrderType.MARKET)),
            ('sell', 0.25, Position('long', 10000), OrderRequest('sell', 2500, OrderType.MARKET)),
            ('sell', 1.00, Position('long', 10000), OrderRequest('sell', 10000, OrderType.MARKET)),
            ('hold', 0.50, None, None),  # No order for hold
        ]
        
        for action_type, size_frac, position, expected in test_cases:
            market_state = Mock(last_price=10.0)
            portfolio_state = Mock(
                buying_power=100000,
                positions={'MLGO': position} if position else {}
            )
            
            order = execution_handler.create_order_request(
                action_type=action_type,
                position_size_fraction=size_frac,
                symbol='MLGO',
                market_state=market_state,
                portfolio_state=portfolio_state
            )
            
            if expected is None:
                assert order is None
            else:
                assert order.side == expected.side
                assert abs(order.quantity - expected.quantity) < 1
                assert order.order_type == expected.order_type
    
    def test_position_size_calculation(self, execution_handler):
        """Test calculation of position sizes."""
        portfolio_state = Mock(
            buying_power=100000,
            total_value=100000,
            positions={}
        )
        
        market_state = Mock(
            last_price=10.0,
            ask_price=10.02
        )
        
        # Test different size fractions
        test_cases = [
            (0.25, 2450),  # ~25% of buying power / price (with buffer)
            (0.50, 4950),  # ~50%
            (0.75, 7425),  # ~75%
            (1.00, 9900),  # ~100% (with safety margin)
        ]
        
        for size_fraction, expected_shares in test_cases:
            shares = execution_handler.calculate_position_size(
                size_fraction=size_fraction,
                portfolio_state=portfolio_state,
                market_state=market_state,
                side='buy'
            )
            
            # Should be close to expected (within 5%)
            assert abs(shares - expected_shares) / expected_shares < 0.05
    
    def test_pre_trade_validation(self, execution_handler):
        """Test pre-trade validation checks."""
        # Test 1: Insufficient buying power
        order = OrderRequest('buy', 15000, OrderType.MARKET, symbol='MLGO')
        portfolio_state = Mock(buying_power=100000)
        market_state = Mock(ask_price=10.0, is_halted=False)
        
        is_valid, reason = execution_handler.validate_order(
            order, portfolio_state, market_state
        )
        
        assert is_valid is False
        assert 'insufficient buying power' in reason.lower()
        
        # Test 2: Market halted
        order = OrderRequest('buy', 1000, OrderType.MARKET, symbol='MLGO')
        portfolio_state = Mock(buying_power=100000)
        market_state = Mock(ask_price=10.0, is_halted=True)
        
        is_valid, reason = execution_handler.validate_order(
            order, portfolio_state, market_state
        )
        
        assert is_valid is False
        assert 'halted' in reason.lower()
        
        # Test 3: Position limit exceeded
        order = OrderRequest('buy', 5000, OrderType.MARKET, symbol='MLGO')
        portfolio_state = Mock(
            buying_power=100000,
            positions={'MLGO': Position('long', 8000)}  # Already have 8k shares
        )
        market_state = Mock(ask_price=10.0, is_halted=False)
        
        execution_handler.config['execution']['max_position_size'] = 10000
        
        is_valid, reason = execution_handler.validate_order(
            order, portfolio_state, market_state
        )
        
        assert is_valid is False
        assert 'position limit' in reason.lower()
    
    def test_execution_flow(self, execution_handler):
        """Test complete execution flow from action to portfolio update."""
        # Setup mocks
        execution_handler.execution_simulator.simulate_execution.return_value = ExecutionResult(
            order_id='TEST_001',
            timestamp=pd.Timestamp.now(),
            symbol='MLGO',
            side='buy',
            requested_price=10.02,
            executed_price=10.025,
            requested_size=1000,
            executed_size=1000,
            slippage=0.005,
            commission=5.0,
            latency_ms=95,
            rejection_reason=None
        )
        
        # Execute trade
        result = execution_handler.execute_action(
            action_type='buy',
            position_size_fraction=0.25,
            symbol='MLGO',
            market_state=Mock(ask_price=10.02, is_halted=False),
            portfolio_state=Mock(buying_power=100000, positions={})
        )
        
        # Check execution simulator was called
        execution_handler.execution_simulator.simulate_execution.assert_called_once()
        
        # Check portfolio update was triggered
        execution_handler.portfolio_simulator.process_execution.assert_called_once_with(
            result
        )
        
        # Result should contain execution details
        assert result['executed'] is True
        assert result['execution_result'].executed_size == 1000
        assert result['execution_result'].commission == 5.0
    
    def test_slippage_calculation(self, execution_handler):
        """Test slippage calculation for different order sizes."""
        market_state = Mock(
            bid_price=10.00,
            ask_price=10.02,
            bid_size=5000,
            ask_size=5000,
            volume=100000
        )
        
        # Small order - minimal slippage
        slippage_small = execution_handler.calculate_expected_slippage(
            order_size=100,
            side='buy',
            market_state=market_state
        )
        
        # Large order - more slippage
        slippage_large = execution_handler.calculate_expected_slippage(
            order_size=5000,
            side='buy',
            market_state=market_state
        )
        
        # Very large order - significant slippage
        slippage_xlarge = execution_handler.calculate_expected_slippage(
            order_size=10000,
            side='buy',
            market_state=market_state
        )
        
        # Slippage should increase with order size
        assert slippage_small < slippage_large < slippage_xlarge
        
        # Small order should have minimal slippage
        assert slippage_small < 0.001  # Less than 0.1%
        
        # Large order exceeding liquidity should have significant slippage
        assert slippage_xlarge > 0.002  # More than 0.2%
    
    def test_market_impact_estimation(self, execution_handler):
        """Test market impact estimation."""
        market_state = Mock(
            volume=100000,
            bid_size=5000,
            ask_size=5000
        )
        
        # Test square root model
        impact = execution_handler.estimate_market_impact(
            order_size=5000,
            average_volume=100000,
            model='square_root'
        )
        
        # Impact should be proportional to sqrt(size/volume)
        expected_ratio = np.sqrt(5000 / 100000)
        assert abs(impact - expected_ratio * 0.001) < 0.0001
        
        # Test linear model
        execution_handler.config['execution']['market_impact_model'] = 'linear'
        
        impact_linear = execution_handler.estimate_market_impact(
            order_size=5000,
            average_volume=100000,
            model='linear'
        )
        
        # Linear impact should be different
        assert impact_linear != impact
    
    def test_commission_calculation(self, execution_handler):
        """Test commission calculation including maker/taker fees."""
        # Regular commission
        commission = execution_handler.calculate_commission(
            executed_size=1000,
            executed_price=10.0,
            is_liquidity_providing=False
        )
        
        expected = max(1000 * 0.005, 1.0)  # Per share commission with minimum
        assert commission == expected
        
        # Maker rebate (providing liquidity)
        rebate = execution_handler.calculate_commission(
            executed_size=1000,
            executed_price=10.0,
            is_liquidity_providing=True
        )
        
        # Should get rebate (negative commission)
        assert rebate == 1000 * (-0.002)
        
        # Large order with tiered commission
        execution_handler.config['execution']['tiered_commission'] = {
            0: 0.005,
            1000: 0.004,
            5000: 0.003,
            10000: 0.002
        }
        
        tiered_commission = execution_handler.calculate_tiered_commission(
            executed_size=7000,
            executed_price=10.0
        )
        
        # Should apply different rates to different tiers
        expected_tiered = (1000 * 0.005 + 4000 * 0.004 + 2000 * 0.003)
        assert abs(tiered_commission - expected_tiered) < 0.01
    
    def test_partial_fill_handling(self, execution_handler):
        """Test handling of partial fills."""
        # Request 5000 shares but only 3000 filled
        execution_result = ExecutionResult(
            order_id='TEST_002',
            timestamp=pd.Timestamp.now(),
            symbol='MLGO',
            side='buy',
            requested_price=10.02,
            executed_price=10.025,
            requested_size=5000,
            executed_size=3000,  # Partial fill
            slippage=0.005,
            commission=15.0,
            latency_ms=100,
            rejection_reason=None
        )
        
        # Process partial fill
        result = execution_handler.process_execution_result(execution_result)
        
        assert result['partial_fill'] is True
        assert result['fill_ratio'] == 0.6  # 3000/5000
        assert result['unfilled_size'] == 2000
        
        # Should track unfilled portion for potential retry
        assert execution_handler.has_pending_order('MLGO')
        pending = execution_handler.get_pending_order('MLGO')
        assert pending['remaining_size'] == 2000
    
    def test_rejection_handling(self, execution_handler):
        """Test handling of order rejections."""
        rejection_reasons = [
            'INSUFFICIENT_LIQUIDITY',
            'MARKET_HALTED',
            'POSITION_LIMIT_EXCEEDED',
            'RISK_CHECK_FAILED',
            'INVALID_PRICE'
        ]
        
        for reason in rejection_reasons:
            execution_result = ExecutionResult(
                order_id=f'TEST_{reason}',
                timestamp=pd.Timestamp.now(),
                symbol='MLGO',
                side='buy',
                requested_price=10.02,
                executed_price=0,
                requested_size=1000,
                executed_size=0,
                slippage=0,
                commission=0,
                latency_ms=50,
                rejection_reason=reason
            )
            
            result = execution_handler.process_execution_result(execution_result)
            
            assert result['executed'] is False
            assert result['rejection_reason'] == reason
            
            # Should categorize rejection type
            assert 'rejection_category' in result
            if 'HALT' in reason:
                assert result['rejection_category'] == 'market_condition'
            elif 'LIMIT' in reason or 'RISK' in reason:
                assert result['rejection_category'] == 'risk_management'
    
    def test_latency_simulation(self, execution_handler):
        """Test realistic latency simulation."""
        latencies = []
        
        # Collect multiple latency samples
        for _ in range(100):
            latency = execution_handler.simulate_latency(
                base_latency_ms=100,
                variance_ms=20,
                order_size=1000,
                market_volatility=0.02
            )
            latencies.append(latency)
        
        # Check distribution
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Mean should be close to base
        assert abs(mean_latency - 100) < 5
        
        # Should have some variance
        assert 10 < std_latency < 30
        
        # Large orders should have higher latency
        large_order_latency = execution_handler.simulate_latency(
            base_latency_ms=100,
            variance_ms=20,
            order_size=10000,  # 10x larger
            market_volatility=0.02
        )
        
        assert large_order_latency > mean_latency


class TestPortfolioIntegration:
    """Test integration with portfolio management."""
    
    @pytest.fixture
    def portfolio_handler(self, execution_config):
        """Create portfolio integration handler."""
        portfolio_sim = Mock(spec=PortfolioSimulator)
        portfolio_sim.get_state.return_value = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=100000,
            positions={},
            total_value=100000,
            buying_power=100000,
            margin_used=0,
            realized_pnl=0,
            unrealized_pnl=0,
            total_pnl=0,
            trade_count=0,
            win_count=0,
            loss_count=0
        )
        
        return PortfolioHandler(
            config=execution_config,
            portfolio_simulator=portfolio_sim,
            logger=Mock()
        )
    
    def test_position_validation(self, portfolio_handler):
        """Test position validation before trades."""
        # Test 1: Can't sell without position
        is_valid, reason = portfolio_handler.validate_action(
            action_type='sell',
            symbol='MLGO',
            portfolio_state=Mock(positions={})
        )
        
        assert is_valid is False
        assert 'no position' in reason.lower()
        
        # Test 2: Can't buy when already long (if restricted)
        portfolio_handler.config['portfolio']['allow_multiple_entries'] = False
        
        is_valid, reason = portfolio_handler.validate_action(
            action_type='buy',
            symbol='MLGO',
            portfolio_state=Mock(positions={'MLGO': Position('long', 1000)})
        )
        
        assert is_valid is False
        assert 'already have position' in reason.lower()
        
        # Test 3: Valid sell with position
        is_valid, reason = portfolio_handler.validate_action(
            action_type='sell',
            symbol='MLGO',
            portfolio_state=Mock(positions={'MLGO': Position('long', 1000)})
        )
        
        assert is_valid is True
    
    def test_risk_checks(self, portfolio_handler):
        """Test portfolio risk checks."""
        # Test 1: Position concentration limit
        portfolio_state = Mock(
            total_value=100000,
            positions={'MLGO': Position('long', 4000, avg_price=10.0)},
            buying_power=60000
        )
        
        market_state = Mock(last_price=10.0)
        
        # Try to buy more when already at 40% concentration
        portfolio_handler.config['portfolio']['max_position_concentration'] = 0.5
        
        is_valid = portfolio_handler.check_concentration_limit(
            symbol='MLGO',
            additional_shares=2000,
            portfolio_state=portfolio_state,
            market_state=market_state
        )
        
        assert is_valid is False  # Would exceed 50% limit
        
        # Test 2: Total exposure limit
        portfolio_state = Mock(
            total_value=100000,
            margin_used=80000,
            buying_power=20000
        )
        
        is_valid = portfolio_handler.check_exposure_limit(
            additional_exposure=30000,
            portfolio_state=portfolio_state
        )
        
        assert is_valid is False  # Would exceed buying power
    
    def test_pnl_tracking(self, portfolio_handler):
        """Test P&L tracking during trades."""
        # Initial state
        initial_state = PortfolioState(
            timestamp=pd.Timestamp.now(),
            cash=100000,
            positions={},
            total_value=100000,
            realized_pnl=0,
            unrealized_pnl=0
        )
        
        # Buy execution
        buy_trade = Trade(
            timestamp=pd.Timestamp.now(),
            symbol='MLGO',
            side='buy',
            quantity=1000,
            price=10.0,
            commission=5.0
        )
        
        # Update portfolio
        new_state = portfolio_handler.update_after_trade(
            initial_state,
            buy_trade
        )
        
        assert new_state.cash == 89995  # 100000 - (1000 * 10) - 5
        assert 'MLGO' in new_state.positions
        assert new_state.positions['MLGO'].quantity == 1000
        assert new_state.positions['MLGO'].avg_price == 10.0
        
        # Price moves up
        new_state = portfolio_handler.update_unrealized_pnl(
            new_state,
            market_prices={'MLGO': 10.5}
        )
        
        assert new_state.unrealized_pnl == 500  # 1000 * 0.5
        assert new_state.total_value == 100495  # 89995 + 10500
        
        # Sell execution
        sell_trade = Trade(
            timestamp=pd.Timestamp.now(),
            symbol='MLGO',
            side='sell',
            quantity=1000,
            price=10.5,
            commission=5.0
        )
        
        final_state = portfolio_handler.update_after_trade(
            new_state,
            sell_trade
        )
        
        assert final_state.cash == 100485  # 89995 + 10500 - 5
        assert final_state.positions == {}  # Position closed
        assert final_state.realized_pnl == 490  # 500 - 10 commission
        assert final_state.unrealized_pnl == 0
    
    def test_margin_calculations(self, portfolio_handler):
        """Test margin and buying power calculations."""
        portfolio_state = PortfolioState(
            cash=100000,
            positions={
                'MLGO': Position('long', 5000, avg_price=10.0)
            },
            margin_multiplier=2.0
        )
        
        # Calculate margin used
        margin_used = portfolio_handler.calculate_margin_used(
            portfolio_state,
            market_prices={'MLGO': 10.0}
        )
        
        assert margin_used == 25000  # 50000 position / 2.0 multiplier
        
        # Calculate buying power
        buying_power = portfolio_handler.calculate_buying_power(
            portfolio_state,
            margin_used=margin_used
        )
        
        assert buying_power == 175000  # (100000 - 25000) * 2.0 + cash
    
    def test_position_metrics(self, portfolio_handler):
        """Test calculation of position-level metrics."""
        position = Position(
            symbol='MLGO',
            side='long',
            quantity=1000,
            avg_price=10.0,
            entry_time=pd.Timestamp.now() - pd.Timedelta(minutes=30)
        )
        
        market_price = 10.5
        
        metrics = portfolio_handler.calculate_position_metrics(
            position,
            market_price
        )
        
        assert metrics['unrealized_pnl'] == 500
        assert metrics['unrealized_return'] == 0.05  # 5%
        assert metrics['position_value'] == 10500
        assert metrics['hold_duration_minutes'] == 30
        assert metrics['pnl_per_minute'] == 500 / 30
    
    def test_trade_statistics(self, portfolio_handler):
        """Test tracking of trade statistics."""
        trades = [
            Trade('MLGO', 'buy', 1000, 10.0, pd.Timestamp.now()),
            Trade('MLGO', 'sell', 1000, 10.5, pd.Timestamp.now()),
            Trade('MLGO', 'buy', 500, 11.0, pd.Timestamp.now()),
            Trade('MLGO', 'sell', 500, 10.8, pd.Timestamp.now()),
            Trade('MLGO', 'buy', 2000, 9.5, pd.Timestamp.now()),
            Trade('MLGO', 'sell', 2000, 9.7, pd.Timestamp.now()),
        ]
        
        stats = portfolio_handler.calculate_trade_statistics(trades)
        
        assert stats['total_trades'] == 3  # 3 round trips
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 1
        assert stats['win_rate'] == 2/3
        assert stats['average_win'] == 350  # (500 + 200) / 2
        assert stats['average_loss'] == 100
        assert stats['profit_factor'] == 7.0  # 700 / 100
        assert stats['total_pnl'] == 600