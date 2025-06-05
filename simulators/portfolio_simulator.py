"""
Portfolio Simulator - Redesigned for new architecture

Handles portfolio state management, position tracking, and P&L calculation.
Works with ExecutionSimulator and TradingEnvironment in the new data flow.

Key features:
- Portfolio state tracking with proper metrics integration
- Position management with real-time P&L calculation
- Trade logging with comprehensive analytics
- Feature extraction for model observations
- Configurable through schemas with proper defaults
- Testable design with clear interfaces
"""

import logging
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass

import numpy as np

from config.schemas import EnvironmentConfig, SimulationConfig, ModelConfig
from dashboard.event_stream import event_stream


class OrderTypeEnum(Enum):
    """Order types for fills."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSideEnum(Enum):
    """Order sides for fills."""

    BUY = "BUY"
    SELL = "SELL"


class PositionSideEnum(Enum):
    """Position sides for tracking."""

    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Represents a position in the portfolio."""

    side: PositionSideEnum
    quantity: float
    avg_price: float
    entry_value: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    entry_timestamp: Optional[datetime] = None
    last_update_timestamp: Optional[datetime] = None
    trade_id: Optional[str] = None  # Add trade_id as proper dataclass field

    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side == PositionSideEnum.FLAT or abs(self.quantity) < 1e-6


@dataclass
class FillDetails:
    """Details of an executed trade fill."""

    asset_id: str
    fill_timestamp: datetime
    order_type: OrderTypeEnum
    order_side: OrderSideEnum
    requested_quantity: float
    executed_quantity: float
    executed_price: float
    commission: float
    fees: float
    slippage_cost_total: float
    closes_position: bool = False
    realized_pnl: Optional[float] = None
    holding_time_minutes: Optional[float] = None

    @property
    def price(self) -> float:
        """Alias for executed_price"""
        return self.executed_price

    @property
    def quantity(self) -> float:
        """Alias for executed_quantity"""
        return self.executed_quantity

    @property
    def slippage_cost(self) -> float:
        """Alias for slippage_cost_total"""
        return self.slippage_cost_total


class TradeRecord(TypedDict):
    """Complete trade record from entry to exit."""

    trade_id: str
    asset_id: str
    side: PositionSideEnum
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    entry_quantity: float
    exit_quantity: float
    avg_entry_price: float
    avg_exit_price: Optional[float]
    realized_pnl: Optional[float]
    total_commission: float
    total_fees: float
    total_slippage: float
    holding_period_seconds: Optional[float]
    max_favorable_excursion: float
    max_adverse_excursion: float
    exit_reason: Optional[str]
    entry_fills: List[FillDetails]
    exit_fills: List[FillDetails]


class PortfolioState(TypedDict):
    """Complete portfolio state snapshot."""

    timestamp: datetime
    cash: float
    total_equity: float
    unrealized_pnl: float
    realized_pnl_session: float
    positions: Dict[str, Dict[str, Any]]
    session_metrics: Dict[str, float]

    # Additional fields for reward system compatibility
    position_side: Optional[str]  # Primary position side as string
    position_value: float  # Total value of positions
    total_value: float  # Alias for total_equity
    current_drawdown_pct: float  # Current drawdown percentage
    avg_entry_price: Optional[float]  # Average entry price of primary position


class PortfolioObservation(TypedDict):
    """Portfolio observation for model input."""

    features: np.ndarray


class PortfolioSimulator:
    """
    Portfolio management with enhanced tracking and metrics.

    Designed to work with:
    - ExecutionSimulator: Receives FillDetails from order execution
    - TradingEnvironment: Provides portfolio state and observations
    - MarketSimulator: Uses market prices for valuation
    - Metrics systems: Reports portfolio performance
    """

    def __init__(
        self,
        logger: logging.Logger,
        env_config: EnvironmentConfig,
        simulation_config: SimulationConfig,
        model_config: ModelConfig,
        tradable_assets: List[str],
        trade_callback: Optional[Callable[[TradeRecord], None]] = None,
    ):
        """
        Initialize portfolio simulator.

        Args:
            logger: Logger instance
            env_config: Environment configuration
            simulation_config: Simulation configuration
            model_config: Model configuration for observations
            tradable_assets: List of tradable symbols
            trade_callback: Optional callback for completed trades
        """
        self.logger = logger
        self.env_config = env_config
        self.simulation_config = simulation_config
        self.model_config = model_config
        self.tradable_assets = tradable_assets
        self.trade_callback = trade_callback

        # Portfolio configuration
        self.initial_capital = simulation_config.initial_capital
        self.max_position_ratio = simulation_config.max_position_value_ratio
        self.allow_shorting = simulation_config.allow_shorting
        self.max_holding_seconds = simulation_config.max_position_holding_seconds

        # Model dimensions for observations
        self.portfolio_seq_len = model_config.portfolio_seq_len
        self.portfolio_feat_dim = model_config.portfolio_feat_dim

        # Portfolio state
        self.cash: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.open_trades: Dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        self.completed_trades: List[TradeRecord] = []

        # Session tracking
        self.session_start_time: Optional[datetime] = None
        self.session_realized_pnl: float = 0.0
        self.session_commission: float = 0.0
        self.session_fees: float = 0.0
        self.session_slippage: float = 0.0
        self.session_volume: float = 0.0
        self.session_turnover: float = 0.0

        # Performance tracking
        self.equity_history: List[Tuple[datetime, float]] = []
        self.feature_history: deque = deque(maxlen=max(1, self.portfolio_seq_len))
        self.peak_equity: float = 0.0
        self.max_drawdown: float = 0.0

        # Trade ID generation
        self.trade_counter: int = 0

        # Initialize
        self.reset(datetime.now(timezone.utc))

    def reset(self, session_start: datetime) -> None:
        """Reset portfolio for new session/episode."""
        self.session_start_time = session_start
        self.cash = self.initial_capital
        self.peak_equity = self.initial_capital

        # Reset positions
        self.positions = {
            asset: Position(
                side=PositionSideEnum.FLAT,
                quantity=0.0,
                avg_price=0.0,
                entry_value=0.0,
                entry_timestamp=session_start,
                trade_id=None,  # Initialize trade_id properly
            )
            for asset in self.tradable_assets
        }

        # Clear trade tracking
        self.open_trades.clear()
        self.completed_trades.clear()

        # Reset session metrics
        self.session_realized_pnl = 0.0
        self.session_commission = 0.0
        self.session_fees = 0.0
        self.session_slippage = 0.0
        self.session_volume = 0.0
        self.session_turnover = 0.0
        self.max_drawdown = 0.0

        # Reset history
        self.equity_history = [(session_start, self.initial_capital)]
        self.feature_history.clear()

        # Initialize feature history with initial features
        # self.logger.debug(f"DEBUG: About to calculate initial portfolio features")
        initial_features = self._calculate_portfolio_features(session_start)
        # self.logger.debug(f"DEBUG: Initial features calculated, shape: {initial_features.shape}")
        # self.logger.debug(f"DEBUG: About to append features to history, seq_len: {self.portfolio_seq_len}")
        append_count = max(1, self.portfolio_seq_len)
        # self.logger.debug(f"DEBUG: Will append {append_count} times to deque with maxlen {self.feature_history.maxlen}")

        for i in range(append_count):
            # self.logger.debug(f"DEBUG: Appending feature {i+1}/{append_count}")
            self.feature_history.append(initial_features)
            # self.logger.debug(f"DEBUG: Successfully appended, current length: {len(self.feature_history)}")

        # self.logger.debug(f"DEBUG: Feature history initialized with {len(self.feature_history)} entries")

        self.logger.debug(f"📊 Portfolio reset - Capital: ${self.initial_capital:,.2f}")

    def process_fill(self, fill: FillDetails) -> FillDetails:
        """
        Process a trade fill from ExecutionSimulator.

        This is the main integration point with the execution system.
        """
        asset_id = fill.asset_id
        if asset_id not in self.positions:
            self.logger.error(f"Received fill for unknown asset: {asset_id}")
            return

        position = self.positions[asset_id]
        fill_qty = fill.executed_quantity
        fill_price = fill.executed_price
        fill_value = fill_qty * fill_price
        order_side = fill.order_side
        commission = fill.commission
        fees = fill.fees
        slippage = fill.slippage_cost_total

        # Update session metrics
        self.session_commission += commission
        self.session_fees += fees
        self.session_slippage += slippage
        self.session_volume += abs(fill_qty)
        self.session_turnover += abs(fill_value)

        # Adjust cash for transaction costs
        self.cash -= commission + fees

        # Determine if this opens/adds or closes/reduces position
        is_buy = order_side == OrderSideEnum.BUY
        position_side = position.side

        # Store state before processing
        was_flat = position.is_flat()
        old_quantity = position.quantity
        old_avg_price = position.avg_price

        # Calculate holding time BEFORE processing (as open_trades might change)
        holding_time_minutes = None
        for trade_id, trade in self.open_trades.items():
            if trade["asset_id"] == asset_id:
                if trade.get("entry_timestamp"):
                    entry_time = trade["entry_timestamp"]
                    fill_time = fill.fill_timestamp
                    holding_time_minutes = (
                        fill_time - entry_time
                    ).total_seconds() / 60.0
                break

        # Calculate potential realized PnL before processing
        realized_pnl = None
        if not was_flat and (
            (position_side == PositionSideEnum.LONG and not is_buy)
            or (position_side == PositionSideEnum.SHORT and is_buy)
        ):
            # This will close/reduce position
            if position_side == PositionSideEnum.LONG:
                pnl_per_share = fill_price - old_avg_price
            else:  # SHORT
                pnl_per_share = old_avg_price - fill_price

            close_qty = min(fill_qty, old_quantity)
            realized_pnl = pnl_per_share * close_qty

        if was_flat:
            # Opening new position
            self._open_new_position(position, fill, is_buy)
        elif (position_side == PositionSideEnum.LONG and is_buy) or (
            position_side == PositionSideEnum.SHORT and not is_buy
        ):
            # Adding to existing position
            self._add_to_position(position, fill, is_buy)
        else:
            # Closing/reducing position
            self._reduce_position(position, fill, is_buy)

        # Update position timestamp
        position.last_update_timestamp = fill.fill_timestamp

        # Determine if this fill closes the position
        closes_position = position.is_flat()

        # Create enriched fill details with additional fields for reward system
        enriched_fill = FillDetails(
            asset_id=fill.asset_id,
            fill_timestamp=fill.fill_timestamp,
            order_type=fill.order_type,
            order_side=fill.order_side,
            requested_quantity=fill.requested_quantity,
            executed_quantity=fill.executed_quantity,
            executed_price=fill.executed_price,
            commission=fill.commission,
            fees=fill.fees,
            slippage_cost_total=fill.slippage_cost_total,
            closes_position=closes_position,
            realized_pnl=realized_pnl,
            holding_time_minutes=holding_time_minutes,
        )

        self.logger.debug(
            f"📈 Fill processed: {asset_id} {order_side.value} {fill_qty:.2f}@${fill_price:.4f}"
        )

        return enriched_fill

    def _open_new_position(
        self, position: Position, fill: FillDetails, is_buy: bool
    ) -> None:
        """Open a new position."""
        fill_qty = fill.executed_quantity
        fill_price = fill.executed_price
        fill_value = fill_qty * fill_price

        # Set position details
        position.side = PositionSideEnum.LONG if is_buy else PositionSideEnum.SHORT
        position.quantity = fill_qty
        position.avg_price = fill_price
        position.entry_value = fill_value
        position.entry_timestamp = fill.fill_timestamp

        # Update cash
        if is_buy:
            self.cash -= fill_value
        else:  # Short sale
            self.cash += fill_value

        # Create trade record
        trade_id = self._generate_trade_id()
        self.open_trades[trade_id] = TradeRecord(
            trade_id=trade_id,
            asset_id=fill.asset_id,
            side=position.side,
            entry_timestamp=fill.fill_timestamp,
            exit_timestamp=None,
            entry_quantity=fill_qty,
            exit_quantity=0.0,
            avg_entry_price=fill_price,
            avg_exit_price=None,
            realized_pnl=None,
            total_commission=fill.commission,
            total_fees=fill.fees,
            total_slippage=fill.slippage_cost_total,
            holding_period_seconds=None,
            max_favorable_excursion=0.0,
            max_adverse_excursion=0.0,
            exit_reason=None,
            entry_fills=[fill],
            exit_fills=[],
        )

        # Store trade_id as proper field
        position.trade_id = trade_id

    def _add_to_position(
        self, position: Position, fill: FillDetails, is_buy: bool
    ) -> None:
        """Add to existing position (average in)."""
        fill_qty = fill.executed_quantity
        fill_price = fill.executed_price
        fill_value = fill_qty * fill_price

        # Calculate new weighted average
        old_value = position.quantity * position.avg_price
        new_total_qty = position.quantity + fill_qty
        new_total_value = old_value + fill_value
        new_avg_price = new_total_value / new_total_qty if new_total_qty > 0 else 0

        # Update position
        position.quantity = new_total_qty
        position.avg_price = new_avg_price
        position.entry_value = new_total_value

        # Update cash
        if is_buy:
            self.cash -= fill_value
        else:  # Adding to short
            self.cash += fill_value

        # Update trade record
        if position.trade_id and position.trade_id in self.open_trades:
            trade = self.open_trades[position.trade_id]
            trade["entry_quantity"] = new_total_qty
            trade["avg_entry_price"] = new_avg_price
            trade["total_commission"] += fill.commission
            trade["total_fees"] += fill.fees
            trade["total_slippage"] += fill.slippage_cost_total
            trade["entry_fills"].append(fill)

    def _reduce_position(
        self, position: Position, fill: FillDetails, is_buy: bool
    ) -> None:
        """Reduce or close position."""
        fill_qty = fill.executed_quantity
        fill_price = fill.executed_price
        fill_value = fill_qty * fill_price

        # Calculate realized P&L for the portion closed
        if position.side == PositionSideEnum.LONG:
            pnl_per_share = fill_price - position.avg_price
        else:  # SHORT
            pnl_per_share = position.avg_price - fill_price

        realized_pnl = pnl_per_share * fill_qty
        self.session_realized_pnl += realized_pnl

        # Update cash
        if is_buy:
            self.cash -= fill_value  # Covering short
        else:
            self.cash += fill_value  # Selling long

        # Update position quantity
        new_qty = position.quantity - fill_qty

        # Update trade record
        if position.trade_id and position.trade_id in self.open_trades:
            trade = self.open_trades[position.trade_id]
            trade["exit_quantity"] += fill_qty
            trade["total_commission"] += fill.commission
            trade["total_fees"] += fill.fees
            trade["total_slippage"] += fill.slippage_cost_total
            trade["exit_fills"].append(fill)

            # Update average exit price
            if trade["avg_exit_price"] is None:
                trade["avg_exit_price"] = fill_price
            else:
                # Weighted average of exit prices
                prev_exit_qty = trade["exit_quantity"] - fill_qty
                if prev_exit_qty > 0:
                    total_exit_value = (trade["avg_exit_price"] * prev_exit_qty) + (
                        fill_price * fill_qty
                    )
                    trade["avg_exit_price"] = total_exit_value / trade["exit_quantity"]

            # Add to realized P&L
            if trade["realized_pnl"] is None:
                trade["realized_pnl"] = realized_pnl
            else:
                trade["realized_pnl"] += realized_pnl

            # Check if position fully closed
            if new_qty < 1e-6:
                self._close_position(position, trade, fill.fill_timestamp)
            else:
                position.quantity = new_qty

    def _close_position(
        self, position: Position, trade: TradeRecord, exit_time: datetime
    ) -> None:
        """Complete position closure."""
        # Reset position
        position.side = PositionSideEnum.FLAT
        position.quantity = 0.0
        position.avg_price = 0.0
        position.entry_value = 0.0
        position.market_value = 0.0
        position.unrealized_pnl = 0.0

        # Emit position update event to notify dashboard of FLAT position
        event_stream.emit_position_update(
            side=position.side.name,  # "FLAT"
            quantity=0,
            avg_price=0.0,
            current_price=0.0,
            unrealized_pnl=0.0,
            realized_pnl=self.session_realized_pnl,
            market_value=0.0,
            entry_timestamp=None,
        )

        # Complete trade record
        trade["exit_timestamp"] = exit_time
        trade["holding_period_seconds"] = (
            exit_time - trade["entry_timestamp"]
        ).total_seconds()

        # Move to completed trades
        self.completed_trades.append(trade)
        del self.open_trades[trade["trade_id"]]

        # Notify callback
        if self.trade_callback:
            try:
                self.trade_callback(trade)
            except Exception as e:
                self.logger.warning(f"Trade callback error: {e}")

        # Remove trade link from position
        if hasattr(position, "trade_id"):
            delattr(position, "trade_id")

    def update_market_values(
        self, market_prices: Dict[str, float], current_time: datetime
    ) -> None:
        """Update market values and unrealized P&L."""
        total_unrealized_pnl = 0.0
        total_market_value = 0.0

        for asset_id, position in self.positions.items():
            if position.is_flat():
                position.market_value = 0.0
                position.unrealized_pnl = 0.0
                continue

            if asset_id not in market_prices:
                self.logger.warning(f"No market price for {asset_id}")
                continue

            current_price = market_prices[asset_id]
            position_value = position.quantity * current_price

            # Calculate market value (signed for short positions)
            if position.side == PositionSideEnum.LONG:
                position.market_value = position_value
                unrealized_pnl = (
                    current_price - position.avg_price
                ) * position.quantity
            else:  # SHORT
                position.market_value = -position_value
                unrealized_pnl = (
                    position.avg_price - current_price
                ) * position.quantity

            position.unrealized_pnl = unrealized_pnl
            total_unrealized_pnl += unrealized_pnl
            total_market_value += position.market_value

            # Emit position update event for dashboard
            event_stream.emit_position_update(
                side=position.side.name,
                quantity=int(position.quantity),
                avg_price=position.avg_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.session_realized_pnl,
                market_value=position.market_value,
                entry_timestamp=position.entry_timestamp,
            )

            # Update MFE/MAE for open trades
            if position.trade_id and position.trade_id in self.open_trades:
                self._update_trade_excursions(position, current_price)

        # Update equity tracking
        total_equity = self.cash + total_market_value
        self.equity_history.append((current_time, total_equity))

        # Track peak equity and drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        current_drawdown = (
            (self.peak_equity - total_equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Update feature history
        features = self._calculate_portfolio_features(current_time)
        self.feature_history.append(features)

    def _update_trade_excursions(
        self, position: Position, current_price: float
    ) -> None:
        """Update max favorable/adverse excursion for open trade."""
        trade = self.open_trades[position.trade_id]
        entry_price = trade["avg_entry_price"]

        if position.side == PositionSideEnum.LONG:
            excursion = (current_price - entry_price) * position.quantity
        else:  # SHORT
            excursion = (entry_price - current_price) * position.quantity

        trade["max_favorable_excursion"] = max(
            trade["max_favorable_excursion"], excursion
        )
        trade["max_adverse_excursion"] = min(trade["max_adverse_excursion"], excursion)

    def _calculate_portfolio_features(self, timestamp: datetime) -> np.ndarray:
        """Calculate enhanced portfolio features for model observation with MFE/MAE tracking."""
        features = np.zeros(self.portfolio_feat_dim, dtype=np.float32)

        if not self.tradable_assets:
            return features

        # Get primary asset position (assuming single asset for now)
        try:
            primary_asset = self.tradable_assets[0]
            position = self.positions.get(primary_asset)
            if position is None:
                return features  # No position data available
        except (IndexError, KeyError):
            return features  # No assets or position data

        # Validate timestamp to prevent negative time calculations
        if timestamp is None or (
            position.entry_timestamp and timestamp < position.entry_timestamp
        ):
            return features

        total_equity = self.cash + sum(
            pos.market_value for pos in self.positions.values()
        )

        # Feature 0: Normalized position size (-1 to 1)
        if total_equity > 0 and not position.is_flat():
            max_position_value = total_equity * self.max_position_ratio
            position_ratio = (
                abs(position.market_value) / max_position_value
                if max_position_value > 0
                else 0
            )

            if position.side == PositionSideEnum.LONG:
                features[0] = min(1.0, position_ratio)
            elif position.side == PositionSideEnum.SHORT:
                features[0] = -min(1.0, position_ratio)

        # Feature 1: Normalized unrealized P&L (-2 to 2, as % of position entry value)
        if (
            not position.is_flat() and abs(position.entry_value) > 1e-6
        ):  # Avoid division by zero
            features[1] = np.clip(
                position.unrealized_pnl / abs(position.entry_value), -2.0, 2.0
            )

        # Feature 2: Time in position (0 to 2, normalized by max holding time)
        if (
            not position.is_flat()
            and position.entry_timestamp
            and self.max_holding_seconds
            and self.max_holding_seconds > 0
        ):
            time_held = (timestamp - position.entry_timestamp).total_seconds()
            if time_held >= 0:  # Ensure non-negative time held
                features[2] = np.clip(time_held / self.max_holding_seconds, 0.0, 2.0)

        # Feature 3: Cash ratio (0 to 2, clipped)
        if total_equity > 0:
            # Handle negative cash gracefully
            cash_ratio = max(0.0, self.cash) / total_equity
            features[3] = np.clip(cash_ratio, 0.0, 2.0)

        # Feature 4: Session P&L percentage (-1 to 1, as % of initial capital)
        if self.initial_capital > 1e-6:  # Add epsilon check for initial capital
            session_pnl_pct = self.session_realized_pnl / self.initial_capital
            features[4] = np.clip(session_pnl_pct, -1.0, 1.0)

        # Feature 5: Maximum Favorable Excursion (MFE) normalized (-2 to 2)
        # Shows best profit achieved during current trade as % of entry value
        if not position.is_flat() and abs(position.entry_value) > 1e-6:
            trade_id = position.trade_id
            if trade_id and trade_id in self.open_trades:
                mfe = self.open_trades[trade_id]["max_favorable_excursion"]
                features[5] = np.clip(mfe / abs(position.entry_value), -2.0, 2.0)

        # Feature 6: Maximum Adverse Excursion (MAE) normalized (-2 to 2)
        # Shows worst loss during current trade as % of entry value
        if not position.is_flat() and abs(position.entry_value) > 1e-6:
            trade_id = position.trade_id
            if trade_id and trade_id in self.open_trades:
                mae = self.open_trades[trade_id]["max_adverse_excursion"]
                features[6] = np.clip(mae / abs(position.entry_value), -2.0, 2.0)

        # Feature 7: Profit giveback ratio (-1 to 1)
        # Shows how much profit has been given back from peak (MFE)
        if not position.is_flat() and abs(position.entry_value) > 1e-6:
            trade_id = position.trade_id
            if trade_id and trade_id in self.open_trades:
                mfe = self.open_trades[trade_id]["max_favorable_excursion"]
                current_pnl = position.unrealized_pnl
                if (
                    abs(mfe) > 1e-6
                ):  # Only meaningful if we had significant profits/losses
                    giveback_ratio = (mfe - current_pnl) / mfe
                    features[7] = np.clip(giveback_ratio, -1.0, 1.0)

        # Feature 8: Recovery ratio (-1 to 1)
        # Shows recovery from worst loss (MAE)
        if not position.is_flat() and abs(position.entry_value) > 1e-6:
            trade_id = position.trade_id
            if trade_id and trade_id in self.open_trades:
                mae = self.open_trades[trade_id]["max_adverse_excursion"]
                current_pnl = position.unrealized_pnl
                if abs(mae) > 1e-6:  # Only meaningful if we had significant losses
                    recovery_ratio = (current_pnl - mae) / abs(mae)
                    features[8] = np.clip(recovery_ratio, -1.0, 1.0)

        # Feature 9: Trade quality score (-1 to 1)
        # Combined metric: positive if current P&L is closer to MFE than MAE
        if not position.is_flat() and abs(position.entry_value) > 1e-6:
            trade_id = position.trade_id
            if trade_id and trade_id in self.open_trades:
                mfe = self.open_trades[trade_id]["max_favorable_excursion"]
                mae = self.open_trades[trade_id]["max_adverse_excursion"]
                current_pnl = position.unrealized_pnl

                # Calculate trade quality: where are we relative to MFE/MAE range?
                mfe_mae_diff = mfe - mae
                if abs(mfe_mae_diff) > 1e-6:  # Avoid division by zero/near-zero
                    trade_quality = (current_pnl - mae) / mfe_mae_diff
                    features[9] = np.clip(trade_quality, -1.0, 1.0)

        return features

    def get_portfolio_observation(self) -> PortfolioObservation:
        """Get portfolio observation for model input."""
        hist_len = len(self.feature_history)
        lookback = max(1, self.portfolio_seq_len)

        if hist_len == 0:
            # Return empty observation
            obs_array = np.zeros((lookback, self.portfolio_feat_dim), dtype=np.float32)
        elif hist_len < lookback:
            # Pad with first observation
            padding = [self.feature_history[0]] * (lookback - hist_len)
            obs_list = padding + list(self.feature_history)
            obs_array = np.array(obs_list, dtype=np.float32)
        else:
            # Use recent history
            obs_array = np.array(list(self.feature_history), dtype=np.float32)

        # Ensure correct shape
        if obs_array.ndim == 1:
            if self.portfolio_feat_dim == 1:
                obs_array = obs_array.reshape(-1, 1)
            elif lookback == 1:
                obs_array = obs_array.reshape(1, -1)

        return PortfolioObservation(features=obs_array)

    def get_portfolio_state(self, timestamp: datetime) -> PortfolioState:
        """Get complete portfolio state."""
        # Convert positions to serializable format
        positions_dict = {}
        for asset_id, position in self.positions.items():
            positions_dict[asset_id] = {
                "quantity": position.quantity,
                "avg_entry_price": position.avg_price,
                "current_side": position.side,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
                "entry_value_total": position.entry_value,
                "last_update_timestamp": position.last_update_timestamp or timestamp,
            }

        total_equity = self.cash + sum(
            pos.market_value for pos in self.positions.values()
        )
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        total_position_value = sum(
            abs(pos.market_value) for pos in self.positions.values()
        )

        # Get primary position info
        primary_position = None
        primary_position_side = None
        primary_avg_entry_price = None

        for position in self.positions.values():
            if not position.is_flat():
                primary_position = position
                primary_position_side = position.side.value if position.side else None
                primary_avg_entry_price = position.avg_price
                break

        # Calculate current drawdown percentage
        current_drawdown_pct = 0.0
        if self.peak_equity > 0:
            current_drawdown_pct = max(
                0, (self.peak_equity - total_equity) / self.peak_equity
            )

        return PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            total_equity=total_equity,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl_session=self.session_realized_pnl,
            positions=positions_dict,
            session_metrics={
                "total_commissions_session": self.session_commission,
                "total_fees_session": self.session_fees,
                "total_slippage_cost_session": self.session_slippage,
                "total_volume_traded_session": self.session_volume,
                "total_turnover_session": self.session_turnover,
                "max_drawdown": self.max_drawdown,
                "peak_equity": self.peak_equity,
            },
            # Additional fields for reward system compatibility
            position_side=primary_position_side,
            position_value=total_position_value,
            total_value=total_equity,  # Alias for total_equity
            current_drawdown_pct=current_drawdown_pct,
            avg_entry_price=primary_avg_entry_price,
        )

    def get_trading_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trading performance metrics."""
        total_trades = len(self.completed_trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl_per_trade": 0.0,
                "total_realized_pnl": self.session_realized_pnl,
                "total_commission": self.session_commission,
                "total_fees": self.session_fees,
                "total_slippage": self.session_slippage,
                "volume_traded": self.session_volume,
                "turnover": self.session_turnover,
            }

        # Analyze completed trades
        winning_trades = [
            t
            for t in self.completed_trades
            if t["realized_pnl"] and t["realized_pnl"] > 0
        ]
        losing_trades = [
            t
            for t in self.completed_trades
            if t["realized_pnl"] and t["realized_pnl"] <= 0
        ]

        win_rate = len(winning_trades) / total_trades * 100
        total_realized = sum(
            t["realized_pnl"] for t in self.completed_trades if t["realized_pnl"]
        )
        avg_pnl = total_realized / total_trades if total_trades > 0 else 0

        avg_win = (
            np.mean([t["realized_pnl"] for t in winning_trades])
            if winning_trades
            else 0
        )
        avg_loss = (
            np.mean([t["realized_pnl"] for t in losing_trades]) if losing_trades else 0
        )

        # Holding periods
        holding_periods = [
            t["holding_period_seconds"]
            for t in self.completed_trades
            if t["holding_period_seconds"] is not None
        ]
        avg_holding_time = np.mean(holding_periods) if holding_periods else 0

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_pnl_per_trade": avg_pnl,
            "avg_winning_trade": avg_win,
            "avg_losing_trade": avg_loss,
            "total_realized_pnl": total_realized,
            "total_commission": self.session_commission,
            "total_fees": self.session_fees,
            "total_slippage": self.session_slippage,
            "volume_traded": self.session_volume,
            "turnover": self.session_turnover,
            "avg_holding_time_seconds": avg_holding_time,
            "max_drawdown": self.max_drawdown * 100,  # As percentage
            "profit_factor": abs(avg_win / avg_loss)
            if avg_loss < 0
            else (float("inf") if avg_win > 0 else 0),
        }

    def get_current_position(self, asset_id: str) -> Optional[Position]:
        """Get current position for an asset."""
        return self.positions.get(asset_id)

    def has_open_positions(self) -> bool:
        """Check if any positions are open."""
        return any(not pos.is_flat() for pos in self.positions.values())

    def get_open_trade_count(self) -> int:
        """Get number of open trades."""
        return len(self.open_trades)

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"T{timestamp}_{self.trade_counter}"

    def __repr__(self) -> str:
        """String representation for debugging."""
        total_equity = self.cash + sum(
            pos.market_value for pos in self.positions.values()
        )
        open_positions = sum(1 for pos in self.positions.values() if not pos.is_flat())

        return (
            f"PortfolioSimulator(equity=${total_equity:.2f}, cash=${self.cash:.2f}, "
            f"positions={open_positions}, trades={len(self.completed_trades)})"
        )
