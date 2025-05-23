# simulators/portfolio_simulator.py - FIXED: Correct position tracking and average entry price calculation

import logging
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from config.config import Config


class OrderTypeEnum(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSideEnum(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSideEnum(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class FillDetails(TypedDict):
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


class TradeLogEntry(TypedDict):
    trade_id: str
    asset_id: str
    side: PositionSideEnum
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    entry_quantity_total: float
    avg_entry_price: float
    exit_quantity_total: float
    avg_exit_price: Optional[float]
    commission_total: float
    fees_total: float
    slippage_total_trade_usd: float
    realized_pnl: Optional[float]
    max_favorable_excursion_usd: float
    max_adverse_excursion_usd: float
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    reason_for_exit: Optional[str]
    holding_period_seconds: Optional[float]
    entry_fills: List[FillDetails]
    exit_fills: List[FillDetails]


class PortfolioObservationFeatures(TypedDict):
    features: np.ndarray


class PortfolioState(TypedDict):
    timestamp: datetime
    cash: float
    total_equity: float
    unrealized_pnl: float
    realized_pnl_session: float
    positions: Dict[str, Dict[str, Any]]
    total_commissions_session: float
    total_fees_session: float
    total_slippage_cost_session: float
    total_volume_traded_session: float
    total_turnover_session: float


class PortfolioManager:
    def __init__(self, logger: logging.Logger, config: Config, tradable_assets: List[str]):
        self.logger = logger

        self.env_config = config.env
        self.portfolio_config = config.simulation.portfolio_config
        self.model_config = config.model
        self.execution_config = config.simulation.execution_config

        self.default_position_value = self.portfolio_config.default_position_value
        self.initial_capital: float = self.portfolio_config.initial_cash
        self.tradable_assets: List[str] = tradable_assets
        self.portfolio_seq_len: int = self.model_config.portfolio_seq_len
        self.portfolio_feat_dim: int = self.model_config.portfolio_feat_dim
        self.max_position_value_ratio: float = self.portfolio_config.max_position_value_ratio
        self.allow_shorting: bool = self.execution_config.allow_shorting
        self.max_position_holding_seconds = self.portfolio_config.max_position_holding_seconds

        # State variables
        self.cash: float = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.open_trades: Dict[str, TradeLogEntry] = {}
        self.trade_log: List[TradeLogEntry] = []
        self.realized_pnl_session: float = 0.0
        self.total_commissions_session: float = 0.0
        self.total_fees_session: float = 0.0
        self.total_slippage_cost_session: float = 0.0
        self.total_volume_traded_session: float = 0.0
        self.total_turnover_session: float = 0.0
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.portfolio_feature_history: deque = deque(maxlen=max(1, self.portfolio_seq_len))
        self.current_total_equity: float = 0.0
        self.current_unrealized_pnl: float = 0.0
        self.trade_id_counter: int = 0

        # FIXED: Enhanced validation and debugging
        self._last_fill_details = None
        self._position_validation_enabled = True

        self.reset(datetime.now(timezone.utc))

    def _generate_trade_id(self) -> str:
        self.trade_id_counter += 1
        try:
            ts_str = pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S%f')
        except NameError:
            ts_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        return f"T{ts_str}_{self.trade_id_counter}"

    def reset(self, episode_start_timestamp: datetime):
        self.cash = self.initial_capital
        self.current_total_equity = self.initial_capital
        self.current_unrealized_pnl = 0.0
        self.positions = {
            asset: {
                'quantity': 0.0,
                'avg_entry_price': 0.0,
                'current_side': PositionSideEnum.FLAT,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'entry_value_total': 0.0,
                'open_trade_id': None,
                # FIXED: Add comprehensive tracking fields
                'total_entry_cost': 0.0,  # Total cost including fees
                'weighted_entry_price': 0.0,  # Price weighted by quantity
                'last_update_timestamp': episode_start_timestamp
            } for asset in self.tradable_assets
        }
        self.open_trades.clear()
        self.trade_log.clear()
        self.realized_pnl_session = 0.0
        self.total_commissions_session = 0.0
        self.total_fees_session = 0.0
        self.total_slippage_cost_session = 0.0
        self.total_volume_traded_session = 0.0
        self.total_turnover_session = 0.0
        self.portfolio_value_history = [(episode_start_timestamp, self.initial_capital)]
        self.portfolio_feature_history.clear()
        initial_features = self._calculate_current_portfolio_features(episode_start_timestamp)
        for _ in range(max(1, self.portfolio_seq_len)):
            self.portfolio_feature_history.append(initial_features)

        self.logger.info(f"🔄 PortfolioManager reset. Initial Capital: ${self.initial_capital:.2f}")

    def _validate_position_data(self, asset_id: str, context: str = ""):
        """FIXED: Comprehensive position validation with detailed logging"""
        if not self._position_validation_enabled:
            return

        pos_data = self.positions.get(asset_id, {})
        qty = pos_data.get('quantity', 0.0)
        avg_entry = pos_data.get('avg_entry_price', 0.0)
        side = pos_data.get('current_side', PositionSideEnum.FLAT)
        entry_value = pos_data.get('entry_value_total', 0.0)

        issues = []

        # Check for inconsistent side and quantity
        if qty == 0.0 and side != PositionSideEnum.FLAT:
            issues.append(f"Zero quantity but side is {side.value}")
        elif qty > 0.0 and side == PositionSideEnum.FLAT:
            issues.append(f"Non-zero quantity ({qty:.4f}) but side is FLAT")
        elif qty < 0.0:
            issues.append(f"Negative quantity: {qty:.4f}")

        # FIXED: More reasonable check for average entry price
        if qty > 0.0 and (avg_entry <= 0.0 or avg_entry > 10000.0):  # Reasonable price range
            issues.append(f"Unreasonable avg_entry_price: ${avg_entry:.4f} with qty: {qty:.4f}")

        # Check for inconsistent entry value
        if qty > 0.0 and avg_entry > 0.0:
            expected_entry_value = qty * avg_entry
            if abs(entry_value - expected_entry_value) > 0.01:
                issues.append(f"Entry value mismatch: stored={entry_value:.4f}, calculated={expected_entry_value:.4f}")

        if issues:
            self.logger.error(f"🚨 POSITION VALIDATION FAILED [{context}] for {asset_id}:")
            for issue in issues:
                self.logger.error(f"   - {issue}")
            self.logger.error(f"   Position data: {pos_data}")
            if self._last_fill_details:
                self.logger.error(f"   Last fill: {self._last_fill_details}")

    def update_fill(self, fill: FillDetails):
        """FIXED: Comprehensive fill processing with correct weighted average calculation"""
        asset_id = fill['asset_id']
        pos_data = self.positions[asset_id]

        # Store fill details for debugging
        self._last_fill_details = fill.copy()

        # FIXED: Validate fill details first
        if fill['executed_quantity'] <= 0:
            self.logger.error(f"❌ Invalid fill quantity: {fill['executed_quantity']}")
            return

        if fill['executed_price'] <= 0:
            self.logger.error(f"❌ Invalid fill price: {fill['executed_price']}")
            return

        # Track session totals IMMEDIATELY
        commission = fill['commission']
        fees = fill['fees']
        slippage = fill['slippage_cost_total']

        self.cash -= (commission + fees)
        self.total_commissions_session += commission
        self.total_fees_session += fees
        self.total_slippage_cost_session += slippage
        self.total_volume_traded_session += abs(fill['executed_quantity'])
        self.total_turnover_session += abs(fill['executed_quantity']) * fill['executed_price']

        qty_change = fill['executed_quantity']
        fill_price = fill['executed_price']
        fill_value = qty_change * fill_price
        order_side = fill['order_side']

        # FIXED: Determine the position side this fill is creating/affecting
        fill_creates_long = (order_side == OrderSideEnum.BUY)
        fill_creates_short = (order_side == OrderSideEnum.SELL and not fill_creates_long)

        # DETAILED LOGGING for debugging
        self.logger.info(
            f"🔄 PROCESSING FILL: {asset_id} {order_side.value} {qty_change:.4f} @ ${fill_price:.4f} "
            f"(Value: ${fill_value:.2f}) | Costs: C:${commission:.4f} F:${fees:.4f} S:${slippage:.4f}"
        )

        current_qty = pos_data['quantity']
        current_side = pos_data['current_side']

        self.logger.info(
            f"📊 BEFORE FILL: Qty={current_qty:.4f}, Side={current_side.value}, AvgEntry=${pos_data['avg_entry_price']:.4f}")

        # FIXED: Determine if this is opening/adding vs closing/reducing
        is_opening_or_adding = (
                (current_side == PositionSideEnum.FLAT) or
                (current_side == PositionSideEnum.LONG and fill_creates_long) or
                (current_side == PositionSideEnum.SHORT and fill_creates_short)
        )

        is_closing_or_reducing = (
                (current_side == PositionSideEnum.LONG and not fill_creates_long) or
                (current_side == PositionSideEnum.SHORT and fill_creates_long)
        )

        # --- Handle Opening/Adding Positions ---
        if is_opening_or_adding:
            # Update cash for the trade
            if order_side == OrderSideEnum.BUY:
                self.cash -= fill_value
            else:  # SELL (shorting)
                self.cash += fill_value

            if current_side == PositionSideEnum.FLAT:
                # NEW POSITION
                new_side = PositionSideEnum.LONG if fill_creates_long else PositionSideEnum.SHORT
                pos_data['current_side'] = new_side
                pos_data['avg_entry_price'] = fill_price
                pos_data['quantity'] = abs(qty_change)  # Always store positive quantity
                pos_data['entry_value_total'] = abs(fill_value)
                pos_data['total_entry_cost'] = abs(fill_value) + commission + fees

                # Create new trade record
                trade_id = self._generate_trade_id()
                pos_data['open_trade_id'] = trade_id
                new_trade = TradeLogEntry(
                    trade_id=trade_id, asset_id=asset_id, side=new_side,
                    entry_timestamp=fill['fill_timestamp'], exit_timestamp=None,
                    entry_quantity_total=abs(qty_change), avg_entry_price=fill_price,
                    exit_quantity_total=0.0, avg_exit_price=None,
                    commission_total=commission, fees_total=fees,
                    slippage_total_trade_usd=slippage,
                    realized_pnl=None,
                    max_favorable_excursion_usd=0.0, max_adverse_excursion_usd=0.0,
                    max_favorable_excursion_pct=0.0, max_adverse_excursion_pct=0.0,
                    reason_for_exit=None, holding_period_seconds=None,
                    entry_fills=[fill], exit_fills=[]
                )
                self.open_trades[trade_id] = new_trade

                self.logger.info(
                    f"✅ NEW POSITION: {asset_id} {new_side.value} {abs(qty_change):.4f} @ ${fill_price:.4f} "
                    f"(Total Value: ${abs(fill_value):.2f})")

            else:
                # ADDING TO EXISTING POSITION
                if pos_data['open_trade_id'] and pos_data['open_trade_id'] in self.open_trades:
                    open_trade = self.open_trades[pos_data['open_trade_id']]

                    # FIXED: Correct weighted average calculation using price * quantity
                    old_qty = pos_data['quantity']
                    old_avg_price = pos_data['avg_entry_price']
                    new_fill_qty = abs(qty_change)
                    new_fill_price = fill_price

                    # Calculate weighted average: (old_qty * old_price + new_qty * new_price) / (old_qty + new_qty)
                    total_cost = (old_qty * old_avg_price) + (new_fill_qty * new_fill_price)
                    new_total_qty = old_qty + new_fill_qty
                    new_avg_price = total_cost / new_total_qty if new_total_qty > 0 else 0

                    # Update position
                    pos_data['avg_entry_price'] = new_avg_price
                    pos_data['quantity'] = new_total_qty
                    pos_data['entry_value_total'] = new_total_qty * new_avg_price  # Recalculate based on new average
                    pos_data['total_entry_cost'] += (abs(fill_value) + commission + fees)

                    # Update trade record
                    open_trade['entry_quantity_total'] = new_total_qty
                    open_trade['avg_entry_price'] = new_avg_price
                    open_trade['commission_total'] += commission
                    open_trade['fees_total'] += fees
                    open_trade['slippage_total_trade_usd'] += slippage
                    open_trade['entry_fills'].append(fill)

                    self.logger.info(
                        f"📈 ADDED TO POSITION: {asset_id} | Old: {old_qty:.4f}@${old_avg_price:.4f} + New: {new_fill_qty:.4f}@${new_fill_price:.4f} = Total: {new_total_qty:.4f}@${new_avg_price:.4f}")

        # --- Handle Closing/Reducing Positions ---
        elif is_closing_or_reducing:
            # Update cash for the trade
            if order_side == OrderSideEnum.SELL:
                self.cash += fill_value
            else:  # BUY (covering short)
                self.cash -= fill_value

            fill_qty = abs(qty_change)
            current_avg_entry = pos_data['avg_entry_price']

            # FIXED: Calculate realized PnL correctly based on position side
            realized_pnl_for_this_fill = 0.0

            if current_side == PositionSideEnum.LONG:
                # Selling long: (sell_price - entry_price) * quantity
                realized_pnl_for_this_fill = (fill_price - current_avg_entry) * fill_qty
            elif current_side == PositionSideEnum.SHORT:
                # Covering short: (entry_price - cover_price) * quantity
                realized_pnl_for_this_fill = (current_avg_entry - fill_price) * fill_qty

            # Deduct costs from realized PnL
            realized_pnl_for_this_fill -= (commission + fees)

            # CRITICAL: Update session realized PnL immediately
            self.realized_pnl_session += realized_pnl_for_this_fill

            self.logger.info(
                f"💰 REALIZED PnL: ${realized_pnl_for_this_fill:.4f} | "
                f"Entry: ${current_avg_entry:.4f} | Exit: ${fill_price:.4f} | "
                f"Qty: {fill_qty:.4f} | SESSION TOTAL: ${self.realized_pnl_session:.4f}"
            )

            # Update trade record
            if pos_data['open_trade_id'] and pos_data['open_trade_id'] in self.open_trades:
                open_trade = self.open_trades[pos_data['open_trade_id']]
                open_trade['exit_fills'].append(fill)
                open_trade['commission_total'] += commission
                open_trade['fees_total'] += fees
                open_trade['slippage_total_trade_usd'] += slippage

                # Update average exit price
                current_exit_qty = open_trade.get('exit_quantity_total', 0.0)
                if open_trade.get('avg_exit_price') is None or current_exit_qty < 1e-9:
                    open_trade['avg_exit_price'] = fill_price
                else:
                    # Weighted average for exit price
                    total_exit_cost = (open_trade['avg_exit_price'] * current_exit_qty) + (fill_price * fill_qty)
                    open_trade['avg_exit_price'] = total_exit_cost / (current_exit_qty + fill_qty)

                open_trade['exit_quantity_total'] = current_exit_qty + fill_qty

                # Update the trade's realized PnL
                if 'realized_pnl' not in open_trade or open_trade['realized_pnl'] is None:
                    open_trade['realized_pnl'] = realized_pnl_for_this_fill
                else:
                    open_trade['realized_pnl'] += realized_pnl_for_this_fill

                # FIXED: Reduce position quantity correctly
                new_qty = max(0.0, pos_data['quantity'] - fill_qty)
                pos_data['quantity'] = new_qty

                # Check if position is fully closed
                if new_qty < 1e-6:  # Position fully closed
                    pos_data['quantity'] = 0.0
                    pos_data['current_side'] = PositionSideEnum.FLAT
                    pos_data['avg_entry_price'] = 0.0
                    pos_data['open_trade_id'] = None
                    pos_data['entry_value_total'] = 0.0
                    pos_data['total_entry_cost'] = 0.0

                    open_trade['exit_timestamp'] = fill['fill_timestamp']
                    open_trade['holding_period_seconds'] = (
                            open_trade['exit_timestamp'] - open_trade['entry_timestamp']).total_seconds()

                    # MFE/MAE calculations
                    initial_trade_value = open_trade['avg_entry_price'] * open_trade['entry_quantity_total']
                    if abs(initial_trade_value) > 1e-9:
                        open_trade['max_favorable_excursion_pct'] = open_trade[
                                                                        'max_favorable_excursion_usd'] / initial_trade_value
                        open_trade['max_adverse_excursion_pct'] = open_trade[
                                                                      'max_adverse_excursion_usd'] / initial_trade_value
                    else:
                        open_trade['max_favorable_excursion_pct'] = 0.0
                        open_trade['max_adverse_excursion_pct'] = 0.0

                    self.trade_log.append(open_trade)
                    del self.open_trades[open_trade['trade_id']]

                    self.logger.info(
                        f"🏁 POSITION CLOSED: {asset_id} | Trade PnL: ${open_trade['realized_pnl']:.4f} | "
                        f"Session Total: ${self.realized_pnl_session:.4f}"
                    )
                else:
                    self.logger.info(f"📉 POSITION REDUCED: {asset_id} qty reduced to {new_qty:.4f}")

        # FIXED: Update timestamp
        pos_data['last_update_timestamp'] = fill['fill_timestamp']

        # FIXED: Validate position after fill processing
        self._validate_position_data(asset_id, f"after fill {order_side.value}")

        self.logger.info(
            f"📊 AFTER FILL: Qty={pos_data['quantity']:.4f}, Side={pos_data['current_side'].value}, AvgEntry=${pos_data['avg_entry_price']:.4f}")
        self.logger.info(
            f"💰 Cash after fill: ${self.cash:.2f} | Session costs: C:${self.total_commissions_session:.4f} F:${self.total_fees_session:.4f}")

    def update_market_value(self, asset_market_prices: Dict[str, float], current_timestamp: datetime):
        """FIXED: Enhanced market value updates with comprehensive P&L calculations"""
        self.current_unrealized_pnl = 0.0
        total_positions_value = 0.0

        for asset_id, pos_data in self.positions.items():
            if pos_data['current_side'] != PositionSideEnum.FLAT and asset_id in asset_market_prices:
                current_price = asset_market_prices[asset_id]
                pos_qty = pos_data['quantity']
                avg_entry = pos_data['avg_entry_price']
                pos_side = pos_data['current_side']

                # FIXED: Calculate market value based on position side
                if pos_side == PositionSideEnum.LONG:
                    pos_data['market_value'] = pos_qty * current_price
                elif pos_side == PositionSideEnum.SHORT:
                    # For short positions, market value is negative (liability)
                    pos_data['market_value'] = -(pos_qty * current_price)
                else:
                    pos_data['market_value'] = 0.0

                total_positions_value += pos_data['market_value']

                # FIXED: Calculate unrealized P&L correctly
                if pos_qty > 0.0 and avg_entry > 0.0:
                    if pos_side == PositionSideEnum.LONG:
                        unreal_pnl_asset = (current_price - avg_entry) * pos_qty
                    elif pos_side == PositionSideEnum.SHORT:
                        unreal_pnl_asset = (avg_entry - current_price) * pos_qty
                    else:
                        unreal_pnl_asset = 0.0

                    pos_data['unrealized_pnl'] = unreal_pnl_asset
                    self.current_unrealized_pnl += unreal_pnl_asset
                else:
                    pos_data['unrealized_pnl'] = 0.0

                # Update MFE/MAE for open trades
                open_trade_id = pos_data.get('open_trade_id')
                if open_trade_id and open_trade_id in self.open_trades:
                    trade = self.open_trades[open_trade_id]
                    trade_avg_entry_price = trade['avg_entry_price']
                    trade_entry_qty = trade['entry_quantity_total']

                    # Calculate current excursion
                    if trade['side'] == PositionSideEnum.LONG:
                        current_excursion_usd = (current_price - trade_avg_entry_price) * trade_entry_qty
                    elif trade['side'] == PositionSideEnum.SHORT:
                        current_excursion_usd = (trade_avg_entry_price - current_price) * trade_entry_qty
                    else:
                        current_excursion_usd = 0.0

                    # Update MFE and MAE
                    trade['max_favorable_excursion_usd'] = max(trade.get('max_favorable_excursion_usd', -np.inf),
                                                               current_excursion_usd)
                    trade['max_adverse_excursion_usd'] = min(trade.get('max_adverse_excursion_usd', np.inf),
                                                             current_excursion_usd)
            else:
                # No position or no market price
                pos_data['market_value'] = 0.0
                pos_data['unrealized_pnl'] = 0.0

        # FIXED: Calculate total equity correctly
        self.current_total_equity = self.cash + total_positions_value

        # Store portfolio value history
        self.portfolio_value_history.append((current_timestamp, self.current_total_equity))
        self.portfolio_feature_history.append(self._calculate_current_portfolio_features(current_timestamp))

        # FIXED: Enhanced logging for debugging
        if len(self.portfolio_value_history) % 100 == 0:  # Log every 100 updates
            self.logger.debug(f"💼 Portfolio Update: Cash=${self.cash:.2f}, Positions=${total_positions_value:.2f}, "
                              f"Total=${self.current_total_equity:.2f}, Unrealized=${self.current_unrealized_pnl:.2f}")

    def _calculate_current_portfolio_features(self, timestamp: datetime) -> np.ndarray:
        """FIXED: Enhanced portfolio features calculation with better normalization"""
        if not self.tradable_assets:
            return np.zeros(self.portfolio_feat_dim if hasattr(self, 'portfolio_feat_dim') else 5, dtype=np.float32)

        asset_id = self.tradable_assets[0]
        pos_data = self.positions.get(asset_id, {})

        # 1. Normalized Position Size (-1 to 1, where -1 = max short, 0 = flat, 1 = max long)
        norm_pos_size = 0.0
        current_pos_market_value = abs(pos_data.get('market_value', 0.0))

        if self.current_total_equity > 1e-9:
            max_theoretical_value_for_pos = self.current_total_equity * self.max_position_value_ratio
            if max_theoretical_value_for_pos > 1e-9:
                position_ratio = current_pos_market_value / max_theoretical_value_for_pos

                if pos_data.get('current_side') == PositionSideEnum.LONG:
                    norm_pos_size = min(1.0, position_ratio)
                elif pos_data.get('current_side') == PositionSideEnum.SHORT:
                    norm_pos_size = -min(1.0, position_ratio)
                else:
                    norm_pos_size = 0.0

        # 2. Normalized Unrealized P&L (relative to position cost)
        norm_unreal_pnl = 0.0
        unreal_pnl_asset = pos_data.get('unrealized_pnl', 0.0)
        entry_value_total_asset = pos_data.get('entry_value_total', 0.0)

        if abs(entry_value_total_asset) > 1e-9:
            norm_unreal_pnl = unreal_pnl_asset / abs(entry_value_total_asset)
        elif self.initial_capital > 1e-9 and abs(unreal_pnl_asset) > 1e-9:
            norm_unreal_pnl = unreal_pnl_asset / self.initial_capital

        norm_unreal_pnl = np.clip(norm_unreal_pnl, -2.0, 2.0)  # Allow up to 200% gains/losses

        # 3. Normalized MAE (Max Adverse Excursion as percentage)
        norm_mae_pct = 0.0
        open_trade_id = pos_data.get('open_trade_id')
        if open_trade_id and open_trade_id in self.open_trades:
            current_trade = self.open_trades[open_trade_id]
            mae_usd = current_trade.get('max_adverse_excursion_usd', 0.0)
            entry_value = current_trade.get('avg_entry_price', 0.0) * current_trade.get('entry_quantity_total', 0.0)

            if abs(entry_value) > 1e-9:
                norm_mae_pct = mae_usd / abs(entry_value)
            norm_mae_pct = np.clip(norm_mae_pct, -1.0, 0.0)  # MAE should be negative or zero

        # 4. Time in trade normalized (0 to 1)
        time_in_trade_normalized = 0.0
        if open_trade_id and open_trade_id in self.open_trades:
            current_trade = self.open_trades[open_trade_id]
            entry_time = current_trade['entry_timestamp']
            duration_seconds = (timestamp - entry_time).total_seconds()

            if self.max_position_holding_seconds > 0:
                time_in_trade_normalized = duration_seconds / self.max_position_holding_seconds
            time_in_trade_normalized = np.clip(time_in_trade_normalized, 0.0, 2.0)  # Allow up to 2x max time

        # 5. Normalized cash percentage (0 to 1+)
        norm_cash_pct = 0.0
        if self.current_total_equity > 1e-9:
            norm_cash_pct = self.cash / self.current_total_equity
        norm_cash_pct = np.clip(norm_cash_pct, 0.0, 2.0)  # Allow up to 200% cash (from shorting)

        features = np.array([
            norm_pos_size,
            norm_unreal_pnl,
            norm_mae_pct,
            time_in_trade_normalized,
            norm_cash_pct
        ], dtype=np.float32)

        # FIXED: Ensure correct feature dimension
        if hasattr(self, 'portfolio_feat_dim') and len(features) != self.portfolio_feat_dim:
            final_features = np.zeros(self.portfolio_feat_dim, dtype=np.float32)
            common_len = min(len(features), self.portfolio_feat_dim)
            final_features[:common_len] = features[:common_len]
            return final_features

        return features

    def get_portfolio_observation(self) -> PortfolioObservationFeatures:
        """Get portfolio observation features with proper shape handling"""
        hist_len = len(self.portfolio_feature_history)
        lookback = max(1, self.portfolio_seq_len)

        if hist_len == 0:
            empty_features = np.zeros((lookback, self.portfolio_feat_dim), dtype=np.float32)
            return PortfolioObservationFeatures(features=empty_features)

        if hist_len < lookback:
            padding = [self.portfolio_feature_history[0]] * (lookback - hist_len)
            obs_list = padding + list(self.portfolio_feature_history)
        else:
            obs_list = list(self.portfolio_feature_history)

        obs_array = np.array(obs_list, dtype=np.float32)
        if obs_array.ndim == 1 and self.portfolio_feat_dim == 1:
            obs_array = obs_array.reshape(-1, 1)
        elif obs_array.ndim == 1 and self.portfolio_feat_dim > 1 and lookback == 1:
            obs_array = obs_array.reshape(1, -1)

        return PortfolioObservationFeatures(features=obs_array)

    def get_portfolio_state(self, current_timestamp: datetime) -> PortfolioState:
        """Get comprehensive portfolio state"""
        return PortfolioState(
            timestamp=current_timestamp,
            cash=self.cash,
            total_equity=self.current_total_equity,
            unrealized_pnl=self.current_unrealized_pnl,
            realized_pnl_session=self.realized_pnl_session,
            positions=self.positions.copy(),
            total_commissions_session=self.total_commissions_session,
            total_fees_session=self.total_fees_session,
            total_slippage_cost_session=self.total_slippage_cost_session,
            total_volume_traded_session=self.total_volume_traded_session,
            total_turnover_session=self.total_turnover_session
        )

    def _calculate_max_drawdown(self, equity_series_values: np.ndarray) -> Tuple[float, float]:
        """Calculate maximum drawdown in absolute and percentage terms"""
        if len(equity_series_values) < 2:
            return 0.0, 0.0

        equity_series = equity_series_values[np.isfinite(equity_series_values)]
        if len(equity_series) < 2:
            return 0.0, 0.0

        high_water_mark = np.maximum.accumulate(equity_series)
        drawdowns_abs = high_water_mark - equity_series
        max_dd_abs = np.max(drawdowns_abs) if len(drawdowns_abs) > 0 else 0.0

        drawdowns_pct = np.zeros_like(equity_series, dtype=float)
        valid_hwm_indices = high_water_mark > 1e-9
        if np.any(valid_hwm_indices):
            drawdowns_pct[valid_hwm_indices] = drawdowns_abs[valid_hwm_indices] / high_water_mark[valid_hwm_indices]
        max_dd_pct = np.max(drawdowns_pct) if len(drawdowns_pct) > 0 else 0.0

        return float(max_dd_abs), float(max_dd_pct * 100)

    def get_trader_vue_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trading performance metrics"""
        metrics: Dict[str, Any] = {
            "num_total_trades": 0, "num_winning_trades": 0, "num_losing_trades": 0, "num_breakeven_trades": 0,
            "total_net_profit_closed_trades": 0.0,
            "avg_net_pnl_per_trade": 0.0,
            "avg_net_winning_trade_pnl": 0.0, "avg_net_losing_trade_pnl": 0.0,
            "largest_net_winning_trade": 0.0, "largest_net_losing_trade": 0.0,
            "total_gross_profit_all_trades": 0.0,
            "avg_gross_pnl_per_trade": 0.0,
            "win_rate_pct": 0.0, "loss_rate_pct": 0.0,
            "profit_factor_gross": 0.0,
            "reward_risk_ratio_net_avg": 0.0,
            "avg_trade_duration_seconds": 0.0,
            "avg_winner_duration_seconds": 0.0, "avg_loser_duration_seconds": 0.0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "max_portfolio_drawdown_abs": 0.0, "max_portfolio_drawdown_pct": 0.0,
            "total_commissions_tradelog": 0.0, "total_fees_tradelog": 0.0, "total_slippage_tradelog": 0.0,
            "avg_commission_per_trade": 0.0, "avg_fees_per_trade": 0.0, "avg_slippage_per_trade_usd": 0.0,
            "avg_mfe_usd_per_trade": 0.0, "avg_mae_usd_per_trade": 0.0,
            "avg_mfe_pct_per_trade": 0.0, "avg_mae_pct_per_trade": 0.0,
            "avg_mfe_usd_winners": 0.0, "avg_mae_usd_winners": 0.0,
            "avg_mfe_usd_losers": 0.0, "avg_mae_usd_losers": 0.0,
            "total_volume_traded_tradelog": 0.0,
            "total_turnover_tradelog": 0.0,
        }

        if not self.trade_log:
            if self.portfolio_value_history:
                equity_values = np.array([pv[1] for pv in self.portfolio_value_history])
                metrics["max_portfolio_drawdown_abs"], metrics[
                    "max_portfolio_drawdown_pct"] = self._calculate_max_drawdown(equity_values)
            return metrics

        metrics["num_total_trades"] = len(self.trade_log)
        net_wins, net_losses, gross_wins_val, gross_losses_val_abs = [], [], [], []
        durations, winner_durations, loser_durations = [], [], []
        mfes_usd, maes_usd, mfes_pct, maes_pct = [], [], [], []
        mfes_usd_winners, maes_usd_winners, mfes_usd_losers, maes_usd_losers = [], [], [], []

        current_win_streak, max_win_streak = 0, 0
        current_loss_streak, max_loss_streak = 0, 0

        for trade in self.trade_log:
            net_pnl = trade['realized_pnl'] or 0.0
            metrics["total_net_profit_closed_trades"] += net_pnl
            metrics["total_commissions_tradelog"] += trade['commission_total']
            metrics["total_fees_tradelog"] += trade['fees_total']
            metrics["total_slippage_tradelog"] += trade['slippage_total_trade_usd']
            metrics["total_volume_traded_tradelog"] += trade['entry_quantity_total']
            for fill in trade['entry_fills'] + trade['exit_fills']:
                metrics["total_turnover_tradelog"] += abs(fill['executed_quantity']) * fill['executed_price']

            gross_pnl_trade = net_pnl + trade['commission_total'] + trade['fees_total']
            metrics["total_gross_profit_all_trades"] += gross_pnl_trade

            if net_pnl > 1e-9:
                metrics["num_winning_trades"] += 1
                net_wins.append(net_pnl)
                if gross_pnl_trade > 0:
                    gross_wins_val.append(gross_pnl_trade)
                else:
                    gross_losses_val_abs.append(abs(gross_pnl_trade))
                if trade.get('holding_period_seconds') is not None:
                    winner_durations.append(trade['holding_period_seconds'])
                current_win_streak += 1
                current_loss_streak = 0
                mfes_usd_winners.append(trade['max_favorable_excursion_usd'])
                maes_usd_winners.append(trade['max_adverse_excursion_usd'])
            elif net_pnl < -1e-9:
                metrics["num_losing_trades"] += 1
                net_losses.append(net_pnl)
                if gross_pnl_trade < 0:
                    gross_losses_val_abs.append(abs(gross_pnl_trade))
                else:
                    gross_wins_val.append(gross_pnl_trade)
                if trade.get('holding_period_seconds') is not None:
                    loser_durations.append(trade['holding_period_seconds'])
                current_loss_streak += 1
                current_win_streak = 0
                mfes_usd_losers.append(trade['max_favorable_excursion_usd'])
                maes_usd_losers.append(trade['max_adverse_excursion_usd'])
            else:
                metrics["num_breakeven_trades"] += 1
                current_win_streak = 0
                current_loss_streak = 0
                if gross_pnl_trade > 0:
                    gross_wins_val.append(gross_pnl_trade)
                elif gross_pnl_trade < 0:
                    gross_losses_val_abs.append(abs(gross_pnl_trade))

            max_win_streak = max(max_win_streak, current_win_streak)
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            if trade.get('holding_period_seconds') is not None:
                durations.append(trade['holding_period_seconds'])

            mfes_usd.append(trade['max_favorable_excursion_usd'])
            maes_usd.append(trade['max_adverse_excursion_usd'])
            mfes_pct.append(trade['max_favorable_excursion_pct'])
            maes_pct.append(trade['max_adverse_excursion_pct'])

        metrics["max_consecutive_wins"] = max_win_streak
        metrics["max_consecutive_losses"] = max_loss_streak

        if metrics["num_total_trades"] > 0:
            metrics["avg_net_pnl_per_trade"] = metrics["total_net_profit_closed_trades"] / metrics["num_total_trades"]
            metrics["avg_gross_pnl_per_trade"] = metrics["total_gross_profit_all_trades"] / metrics["num_total_trades"]
            metrics["win_rate_pct"] = (metrics["num_winning_trades"] / metrics["num_total_trades"]) * 100
            metrics["loss_rate_pct"] = (metrics["num_losing_trades"] / metrics["num_total_trades"]) * 100
            metrics["avg_commission_per_trade"] = metrics["total_commissions_tradelog"] / metrics["num_total_trades"]
            metrics["avg_fees_per_trade"] = metrics["total_fees_tradelog"] / metrics["num_total_trades"]
            metrics["avg_slippage_per_trade_usd"] = metrics["total_slippage_tradelog"] / metrics["num_total_trades"]
            if durations:
                metrics["avg_trade_duration_seconds"] = np.mean(durations)

        if net_wins:
            metrics["avg_net_winning_trade_pnl"] = np.mean(net_wins)
            metrics["largest_net_winning_trade"] = np.max(net_wins)
            if winner_durations:
                metrics["avg_winner_duration_seconds"] = np.mean(winner_durations)
            if mfes_usd_winners:
                metrics["avg_mfe_usd_winners"] = np.mean(mfes_usd_winners)
            if maes_usd_winners:
                metrics["avg_mae_usd_winners"] = np.mean(maes_usd_winners)

        if net_losses:
            metrics["avg_net_losing_trade_pnl"] = np.mean(net_losses)
            metrics["largest_net_losing_trade"] = np.min(net_losses)
            if loser_durations:
                metrics["avg_loser_duration_seconds"] = np.mean(loser_durations)
            if mfes_usd_losers:
                metrics["avg_mfe_usd_losers"] = np.mean(mfes_usd_losers)
            if maes_usd_losers:
                metrics["avg_mae_usd_losers"] = np.mean(maes_usd_losers)

        sum_gross_wins = sum(gross_wins_val)
        sum_gross_losses_abs = sum(gross_losses_val_abs)
        if sum_gross_losses_abs > 1e-9:
            metrics["profit_factor_gross"] = sum_gross_wins / sum_gross_losses_abs
        elif sum_gross_wins > 0:
            metrics["profit_factor_gross"] = np.inf
        else:
            metrics["profit_factor_gross"] = 0.0

        avg_net_win = metrics.get("avg_net_winning_trade_pnl", 0.0)
        avg_net_loss_abs = abs(metrics.get("avg_net_losing_trade_pnl", 0.0))
        if avg_net_loss_abs > 1e-9:
            metrics["reward_risk_ratio_net_avg"] = avg_net_win / avg_net_loss_abs if avg_net_win is not None else 0.0
        elif avg_net_win is not None and avg_net_win > 0:
            metrics["reward_risk_ratio_net_avg"] = np.inf
        else:
            metrics["reward_risk_ratio_net_avg"] = 0.0

        if mfes_usd:
            metrics["avg_mfe_usd_per_trade"] = np.mean(mfes_usd)
        if maes_usd:
            metrics["avg_mae_usd_per_trade"] = np.mean(maes_usd)
        if mfes_pct:
            metrics["avg_mfe_pct_per_trade"] = np.mean(mfes_pct) * 100
        if maes_pct:
            metrics["avg_mae_pct_per_trade"] = np.mean(maes_pct) * 100

        if self.portfolio_value_history:
            equity_values = np.array([pv[1] for pv in self.portfolio_value_history])
            metrics["max_portfolio_drawdown_abs"], metrics["max_portfolio_drawdown_pct"] = self._calculate_max_drawdown(
                equity_values)

        self.logger.info(
            f"📈 TraderVue Metrics: Net Profit=${metrics['total_net_profit_closed_trades']:.2f}, "
            f"Win Rate={metrics['win_rate_pct']:.1f}%, Trades={metrics['num_total_trades']}")
        return metrics