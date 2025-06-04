#!/usr/bin/env python3
"""
Simple validation script to test P&L calculation flow.
This script creates a minimal trade scenario to verify correct P&L calculation and dashboard transmission.
"""

import logging
from datetime import datetime, timezone
from simulators.portfolio_simulator import (
    PortfolioSimulator,
    FillDetails,
    OrderTypeEnum,
    OrderSideEnum,
)
from config.schemas import EnvironmentConfig, SimulationConfig, ModelConfig
from dashboard.event_stream import event_stream, EventType, TradingEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_configs():
    """Create test configurations."""
    env_config = EnvironmentConfig()

    simulation_config = SimulationConfig(
        max_position_value_ratio=1.0,
        allow_shorting=False,
        max_position_holding_seconds=3600,
        initial_capital=25000.0,
    )

    model_config = ModelConfig()

    return env_config, simulation_config, model_config


def create_test_fill(
    side: OrderSideEnum, quantity: float, price: float, timestamp: datetime = None
) -> FillDetails:
    """Create a test fill."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return FillDetails(
        asset_id="MLGO",
        fill_timestamp=timestamp,
        order_type=OrderTypeEnum.MARKET,
        order_side=side,
        requested_quantity=quantity,
        executed_quantity=quantity,
        executed_price=price,
        commission=0.005 * quantity,  # $0.005 per share
        fees=0.001 * quantity,  # $0.001 per share
        slippage_cost_total=0.001 * quantity * price,  # Small slippage
    )


def capture_dashboard_events():
    """Capture events emitted to dashboard."""
    captured_events = []

    def event_handler(event: TradingEvent):
        captured_events.append(event)
        logger.info(f"Captured event: {event.event_type.value} - {event.data}")

    event_stream.subscribe(event_handler)
    return captured_events, event_handler


def validate_basic_trade_flow():
    """Test basic trade flow: buy -> sell with P&L calculation."""
    logger.info("=== Testing Basic Trade Flow ===")

    # Create configurations
    env_config, simulation_config, model_config = create_test_configs()

    # Create portfolio simulator
    portfolio = PortfolioSimulator(
        logger=logger,
        env_config=env_config,
        simulation_config=simulation_config,
        model_config=model_config,
        tradable_assets=["MLGO"],
    )

    # Capture dashboard events
    captured_events, event_handler = capture_dashboard_events()

    # Reset portfolio
    start_time = datetime.now(timezone.utc)
    portfolio.reset(start_time)

    logger.info(f"Initial cash: ${portfolio.cash:,.2f}")

    # === BUY TRADE ===
    buy_fill = create_test_fill(OrderSideEnum.BUY, 1000, 3.00, start_time)
    logger.info(
        f"Processing BUY fill: {buy_fill.executed_quantity} @ ${buy_fill.executed_price}"
    )

    enriched_buy = portfolio.process_fill(buy_fill)
    logger.info(
        f"Enriched BUY - closes_position: {enriched_buy.closes_position}, realized_pnl: {enriched_buy.realized_pnl}"
    )

    # Update market values
    portfolio.update_market_values({"MLGO": 3.05}, start_time + timedelta(seconds=1))

    # Get position state
    position = portfolio.get_current_position("MLGO")
    logger.info(
        f"Position after BUY: side={position.side.value}, qty={position.quantity}, avg_price=${position.avg_price:.4f}"
    )
    logger.info(f"Position unrealized P&L: ${position.unrealized_pnl:.2f}")

    # === SELL TRADE ===
    sell_time = start_time + timedelta(minutes=5)
    sell_fill = create_test_fill(OrderSideEnum.SELL, 1000, 3.10, sell_time)
    logger.info(
        f"Processing SELL fill: {sell_fill.executed_quantity} @ ${sell_fill.executed_price}"
    )

    enriched_sell = portfolio.process_fill(sell_fill)
    logger.info(
        f"Enriched SELL - closes_position: {enriched_sell.closes_position}, realized_pnl: {enriched_sell.realized_pnl}"
    )

    # Update market values after sell
    portfolio.update_market_values({"MLGO": 3.10}, sell_time)

    # Get final position state
    final_position = portfolio.get_current_position("MLGO")
    logger.info(
        f"Position after SELL: side={final_position.side.value}, qty={final_position.quantity}"
    )

    # Get portfolio state
    portfolio_state = portfolio.get_portfolio_state(sell_time)
    logger.info(f"Final cash: ${portfolio_state['cash']:,.2f}")
    logger.info(f"Session realized P&L: ${portfolio_state['realized_pnl_session']:.2f}")

    # === CALCULATE EXPECTED P&L ===
    expected_gross_pnl = (3.10 - 3.00) * 1000  # $100
    expected_commission = (0.005 * 1000) + (0.005 * 1000)  # $10 total
    expected_fees = (0.001 * 1000) + (0.001 * 1000)  # $2 total
    expected_net_pnl = expected_gross_pnl - expected_commission - expected_fees  # $88

    logger.info(f"Expected gross P&L: ${expected_gross_pnl:.2f}")
    logger.info(f"Expected commission: ${expected_commission:.2f}")
    logger.info(f"Expected fees: ${expected_fees:.2f}")
    logger.info(f"Expected net P&L: ${expected_net_pnl:.2f}")

    # === VALIDATE ===
    actual_realized_pnl = portfolio_state["realized_pnl_session"]

    if (
        abs(actual_realized_pnl - expected_gross_pnl) < 0.01
    ):  # Check against gross P&L (commission/fees handled separately)
        logger.info("✅ P&L calculation CORRECT")
    else:
        logger.error(
            f"❌ P&L calculation INCORRECT - Expected: ${expected_gross_pnl:.2f}, Actual: ${actual_realized_pnl:.2f}"
        )

    # Check dashboard events
    logger.info(f"Captured {len(captured_events)} dashboard events")
    for event in captured_events:
        if event.event_type == EventType.TRADE_EXECUTION:
            logger.info(f"Trade event: {event.data}")
        elif event.event_type == EventType.POSITION_UPDATE:
            logger.info(f"Position event: {event.data}")

    # Cleanup
    event_stream.unsubscribe(event_handler)

    return actual_realized_pnl, expected_gross_pnl


if __name__ == "__main__":
    # Import required modules at runtime to avoid import issues
    from datetime import timedelta

    try:
        validate_basic_trade_flow()
        logger.info("✅ P&L validation completed")
    except Exception as e:
        logger.error(f"❌ P&L validation failed: {e}")
        import traceback

        traceback.print_exc()
