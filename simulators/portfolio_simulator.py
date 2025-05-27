# simulators/portfolio_simulator.py - CLEAN: Streamlined portfolio management

import logging
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from config.schemas import EnvConfig, SimulationConfig, ModelConfig
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a position in the portfolio."""
    side: str  # 'long' or 'short'
    quantity: float
    avg_price: float = 0.0
    
    def __init__(self, side: str, quantity: float, avg_price: float = 0.0):
        self.side = side
        self.quantity = quantity
        self.avg_price = avg_price


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: pd.Timestamp


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


class PortfolioSimulator:
    """Clean portfolio management with essential logging only"""

    def __init__(self, logger: logging.Logger, env_config: EnvConfig, tradable_assets: List[str], 
                 simulation_config: Optional[SimulationConfig] = None, 
                 model_config: Optional[ModelConfig] = None, trade_callback=None):
        self.logger = logger
        self.env_config: EnvConfig = env_config
        self.simulation_config = simulation_config
        self.model_config = model_config
        self.trade_callback = trade_callback  # Callback for completed trades

        # Core configuration
        self.initial_capital: float = env_config.initial_capital
        self.tradable_assets: List[str] = tradable_assets

        
        # Model configuration (with defaults)
        if model_config:
            self.portfolio_seq_len: int = model_config.portfolio_seq_len
            self.portfolio_feat_dim: int = model_config.portfolio_feat_dim
        else:
            self.portfolio_seq_len: int = 10
            self.portfolio_feat_dim: int = 10
        
        # Simulation configuration (with defaults)
        if simulation_config:
            self.max_position_value_ratio: float = simulation_config.max_position_value_ratio
            self.allow_shorting: bool = simulation_config.allow_shorting
            self.max_position_holding_seconds = simulation_config.max_position_holding_seconds
        else:
            self.max_position_value_ratio: float = 1.0
            self.allow_shorting: bool = False
            self.max_position_holding_seconds = None


        # State variables
        self.cash: float = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.open_trades: Dict[str, TradeLogEntry] = {}
        self.trade_log: List[TradeLogEntry] = []

        # Session totals
        self.realized_pnl_session: float = 0.0
        self.total_commissions_session: float = 0.0
        self.total_fees_session: float = 0.0
        self.total_slippage_cost_session: float = 0.0
        self.total_volume_traded_session: float = 0.0
        self.total_turnover_session: float = 0.0

        # Current state
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.portfolio_feature_history: deque = deque(maxlen=max(1, self.portfolio_seq_len))
        self.current_total_equity: float = 0.0
        self.current_unrealized_pnl: float = 0.0
        self.trade_id_counter: int = 0

        self.reset(datetime.now(timezone.utc))

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_id_counter += 1
        ts_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        return f"T{ts_str}_{self.trade_id_counter}"

    def reset(self, episode_start_timestamp: datetime):
        """Reset portfolio for new episode"""
        self.cash = self.initial_capital
        self.current_total_equity = self.initial_capital
        self.current_unrealized_pnl = 0.0

        # Initialize positions
        self.positions = {
            asset: {
                'quantity': 0.0,
                'avg_entry_price': 0.0,
                'current_side': PositionSideEnum.FLAT,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'entry_value_total': 0.0,
                'open_trade_id': None,
                'total_entry_cost': 0.0,
                'weighted_entry_price': 0.0,
                'last_update_timestamp': episode_start_timestamp
            } for asset in self.tradable_assets
        }

        # Clear session data
        self.open_trades.clear()
        self.trade_log.clear()
        self.realized_pnl_session = 0.0
        self.total_commissions_session = 0.0
        self.total_fees_session = 0.0
        self.total_slippage_cost_session = 0.0
        self.total_volume_traded_session = 0.0
        self.total_turnover_session = 0.0

        # Reset history
        self.portfolio_value_history = [(episode_start_timestamp, self.initial_capital)]
        self.portfolio_feature_history.clear()

        # Initialize feature history
        initial_features = self._calculate_current_portfolio_features(episode_start_timestamp)
        for _ in range(max(1, self.portfolio_seq_len)):
            self.portfolio_feature_history.append(initial_features)

    def update_fill(self, fill: FillDetails):
        """Process a fill execution"""
        asset_id = fill['asset_id']
        pos_data = self.positions[asset_id]

        # Update session totals
        commission = fill['commission']
        fees = fill['fees']
        slippage = fill['slippage_cost_total']

        self.cash -= (commission + fees)
        self.total_commissions_session += commission
        self.total_fees_session += fees
        self.total_slippage_cost_session += slippage
        self.total_volume_traded_session += abs(fill['executed_quantity'])
        self.total_turnover_session += abs(fill['executed_quantity']) * fill['executed_price']

        # Process fill
        qty_change = fill['executed_quantity']
        fill_price = fill['executed_price']
        fill_value = qty_change * fill_price
        order_side = fill['order_side']

        current_qty = pos_data['quantity']
        current_side = pos_data['current_side']

        # Determine if opening/adding or closing/reducing
        fill_creates_long = (order_side == OrderSideEnum.BUY)

        is_opening_or_adding = (
                (current_side == PositionSideEnum.FLAT) or
                (current_side == PositionSideEnum.LONG and fill_creates_long) or
                (current_side == PositionSideEnum.SHORT and not fill_creates_long)
        )

        is_closing_or_reducing = (
                (current_side == PositionSideEnum.LONG and not fill_creates_long) or
                (current_side == PositionSideEnum.SHORT and fill_creates_long)
        )

        if is_opening_or_adding:
            # Update cash
            if order_side == OrderSideEnum.BUY:
                self.cash -= fill_value
            else:  # SELL (shorting)
                self.cash += fill_value

            if current_side == PositionSideEnum.FLAT:
                # New position
                new_side = PositionSideEnum.LONG if fill_creates_long else PositionSideEnum.SHORT
                pos_data['current_side'] = new_side
                pos_data['avg_entry_price'] = fill_price
                pos_data['quantity'] = abs(qty_change)
                pos_data['entry_value_total'] = abs(fill_value)
                pos_data['total_entry_cost'] = abs(fill_value) + commission + fees

                # Create trade record
                trade_id = self._generate_trade_id()
                pos_data['open_trade_id'] = trade_id
                self.open_trades[trade_id] = TradeLogEntry(
                    trade_id=trade_id, asset_id=asset_id, side=new_side,
                    entry_timestamp=fill['fill_timestamp'], exit_timestamp=None,
                    entry_quantity_total=abs(qty_change), avg_entry_price=fill_price,
                    exit_quantity_total=0.0, avg_exit_price=None,
                    commission_total=commission, fees_total=fees,
                    slippage_total_trade_usd=slippage, realized_pnl=None,
                    max_favorable_excursion_usd=0.0, max_adverse_excursion_usd=0.0,
                    max_favorable_excursion_pct=0.0, max_adverse_excursion_pct=0.0,
                    reason_for_exit=None, holding_period_seconds=None,
                    entry_fills=[fill], exit_fills=[]
                )
            else:
                # Adding to position
                old_qty = pos_data['quantity']
                old_avg_price = pos_data['avg_entry_price']
                new_fill_qty = abs(qty_change)

                # Weighted average calculation
                total_cost = (old_qty * old_avg_price) + (new_fill_qty * fill_price)
                new_total_qty = old_qty + new_fill_qty
                new_avg_price = total_cost / new_total_qty if new_total_qty > 0 else 0

                pos_data['avg_entry_price'] = new_avg_price
                pos_data['quantity'] = new_total_qty
                pos_data['entry_value_total'] = new_total_qty * new_avg_price
                pos_data['total_entry_cost'] += (abs(fill_value) + commission + fees)

                # Update trade record
                if pos_data['open_trade_id'] in self.open_trades:
                    trade = self.open_trades[pos_data['open_trade_id']]
                    trade['entry_quantity_total'] = new_total_qty
                    trade['avg_entry_price'] = new_avg_price
                    trade['commission_total'] += commission
                    trade['fees_total'] += fees
                    trade['slippage_total_trade_usd'] += slippage
                    trade['entry_fills'].append(fill)

        elif is_closing_or_reducing:
            # Update cash
            if order_side == OrderSideEnum.SELL:
                self.cash += fill_value
            else:  # BUY (covering short)
                self.cash -= fill_value

            fill_qty = abs(qty_change)
            current_avg_entry = pos_data['avg_entry_price']

            # Calculate realized PnL
            if current_side == PositionSideEnum.LONG:
                realized_pnl_for_fill = (fill_price - current_avg_entry) * fill_qty
            elif current_side == PositionSideEnum.SHORT:
                realized_pnl_for_fill = (current_avg_entry - fill_price) * fill_qty
            else:
                realized_pnl_for_fill = 0.0

            # Deduct costs
            realized_pnl_for_fill -= (commission + fees)
            self.realized_pnl_session += realized_pnl_for_fill

            # Update trade record
            if pos_data['open_trade_id'] in self.open_trades:
                trade = self.open_trades[pos_data['open_trade_id']]
                trade['exit_fills'].append(fill)
                trade['commission_total'] += commission
                trade['fees_total'] += fees
                trade['slippage_total_trade_usd'] += slippage

                # Update exit info
                current_exit_qty = trade.get('exit_quantity_total', 0.0)
                if trade.get('avg_exit_price') is None or current_exit_qty < 1e-9:
                    trade['avg_exit_price'] = fill_price
                else:
                    total_exit_cost = (trade['avg_exit_price'] * current_exit_qty) + (fill_price * fill_qty)
                    trade['avg_exit_price'] = total_exit_cost / (current_exit_qty + fill_qty)

                trade['exit_quantity_total'] = current_exit_qty + fill_qty

                if 'realized_pnl' not in trade or trade['realized_pnl'] is None:
                    trade['realized_pnl'] = realized_pnl_for_fill
                else:
                    trade['realized_pnl'] += realized_pnl_for_fill

                # Update position quantity
                new_qty = max(0.0, pos_data['quantity'] - fill_qty)
                pos_data['quantity'] = new_qty

                # Check if position fully closed
                if new_qty < 1e-6:
                    pos_data['quantity'] = 0.0
                    pos_data['current_side'] = PositionSideEnum.FLAT
                    pos_data['avg_entry_price'] = 0.0
                    pos_data['open_trade_id'] = None
                    pos_data['entry_value_total'] = 0.0
                    pos_data['total_entry_cost'] = 0.0

                    # Complete trade
                    trade['exit_timestamp'] = fill['fill_timestamp']
                    trade['holding_period_seconds'] = (
                            trade['exit_timestamp'] - trade['entry_timestamp']).total_seconds()

                    # Calculate MFE/MAE percentages
                    initial_value = trade['avg_entry_price'] * trade['entry_quantity_total']
                    if abs(initial_value) > 1e-9:
                        trade['max_favorable_excursion_pct'] = trade['max_favorable_excursion_usd'] / initial_value
                        trade['max_adverse_excursion_pct'] = trade['max_adverse_excursion_usd'] / initial_value

                    self.trade_log.append(trade)
                    del self.open_trades[trade['trade_id']]
                    
                    # Notify callback about completed trade
                    if self.trade_callback:
                        self.trade_callback(trade)

        # Update timestamp
        pos_data['last_update_timestamp'] = fill['fill_timestamp']

    def update_market_value(self, asset_market_prices: Dict[str, float], current_timestamp: datetime):
        """Update market values and unrealized PnL"""
        self.current_unrealized_pnl = 0.0
        total_positions_value = 0.0

        for asset_id, pos_data in self.positions.items():
            if pos_data['current_side'] != PositionSideEnum.FLAT and asset_id in asset_market_prices:
                current_price = asset_market_prices[asset_id]
                pos_qty = pos_data['quantity']
                avg_entry = pos_data['avg_entry_price']
                pos_side = pos_data['current_side']

                # Calculate market value
                if pos_side == PositionSideEnum.LONG:
                    pos_data['market_value'] = pos_qty * current_price
                elif pos_side == PositionSideEnum.SHORT:
                    pos_data['market_value'] = -(pos_qty * current_price)
                else:
                    pos_data['market_value'] = 0.0

                total_positions_value += pos_data['market_value']

                # Calculate unrealized PnL
                if pos_qty > 0.0 and avg_entry > 0.0:
                    if pos_side == PositionSideEnum.LONG:
                        unreal_pnl = (current_price - avg_entry) * pos_qty
                    elif pos_side == PositionSideEnum.SHORT:
                        unreal_pnl = (avg_entry - current_price) * pos_qty
                    else:
                        unreal_pnl = 0.0

                    pos_data['unrealized_pnl'] = unreal_pnl
                    self.current_unrealized_pnl += unreal_pnl

                    # Update MFE/MAE for open trades
                    if pos_data.get('open_trade_id') in self.open_trades:
                        trade = self.open_trades[pos_data['open_trade_id']]
                        entry_price = trade['avg_entry_price']
                        entry_qty = trade['entry_quantity_total']

                        if trade['side'] == PositionSideEnum.LONG:
                            excursion = (current_price - entry_price) * entry_qty
                        else:
                            excursion = (entry_price - current_price) * entry_qty

                        trade['max_favorable_excursion_usd'] = max(
                            trade.get('max_favorable_excursion_usd', -np.inf), excursion)
                        trade['max_adverse_excursion_usd'] = min(
                            trade.get('max_adverse_excursion_usd', np.inf), excursion)
                else:
                    pos_data['unrealized_pnl'] = 0.0
            else:
                pos_data['market_value'] = 0.0
                pos_data['unrealized_pnl'] = 0.0

        # Update total equity
        self.current_total_equity = self.cash + total_positions_value

        # Store history
        self.portfolio_value_history.append((current_timestamp, self.current_total_equity))
        self.portfolio_feature_history.append(self._calculate_current_portfolio_features(current_timestamp))

    def _calculate_current_portfolio_features(self, timestamp: datetime) -> np.ndarray:
        """Calculate portfolio features for observation"""
        if not self.tradable_assets:
            return np.zeros(self.portfolio_feat_dim, dtype=np.float32)

        asset_id = self.tradable_assets[0]
        pos_data = self.positions.get(asset_id, {})

        # Normalized position size (-1 to 1)
        norm_pos_size = 0.0
        current_pos_value = abs(pos_data.get('market_value', 0.0))

        if self.current_total_equity > 1e-9:
            max_pos_value = self.current_total_equity * self.max_position_value_ratio
            if max_pos_value > 1e-9:
                position_ratio = current_pos_value / max_pos_value
                if pos_data.get('current_side') == PositionSideEnum.LONG:
                    norm_pos_size = min(1.0, position_ratio)
                elif pos_data.get('current_side') == PositionSideEnum.SHORT:
                    norm_pos_size = -min(1.0, position_ratio)

        # Normalized unrealized PnL
        norm_unreal_pnl = 0.0
        unreal_pnl = pos_data.get('unrealized_pnl', 0.0)
        entry_value = pos_data.get('entry_value_total', 0.0)

        if abs(entry_value) > 1e-9:
            norm_unreal_pnl = unreal_pnl / abs(entry_value)
        elif self.initial_capital > 1e-9 and abs(unreal_pnl) > 1e-9:
            norm_unreal_pnl = unreal_pnl / self.initial_capital

        norm_unreal_pnl = np.clip(norm_unreal_pnl, -2.0, 2.0)

        # Time in trade
        time_in_trade_norm = 0.0
        if pos_data.get('open_trade_id') in self.open_trades:
            trade = self.open_trades[pos_data['open_trade_id']]
            duration = (timestamp - trade['entry_timestamp']).total_seconds()
            if self.max_position_holding_seconds > 0:
                time_in_trade_norm = duration / self.max_position_holding_seconds
            time_in_trade_norm = np.clip(time_in_trade_norm, 0.0, 2.0)

        # Cash percentage
        norm_cash_pct = 0.0
        if self.current_total_equity > 1e-9:
            norm_cash_pct = self.cash / self.current_total_equity
        norm_cash_pct = np.clip(norm_cash_pct, 0.0, 2.0)

        # Session PnL percentage
        session_pnl_pct = 0.0
        if self.initial_capital > 1e-9:
            session_pnl_pct = self.realized_pnl_session / self.initial_capital
        session_pnl_pct = np.clip(session_pnl_pct, -1.0, 1.0)

        features = np.array([
            norm_pos_size,
            norm_unreal_pnl,
            time_in_trade_norm,
            norm_cash_pct,
            session_pnl_pct
        ], dtype=np.float32)

        # Ensure correct dimension
        if len(features) != self.portfolio_feat_dim:
            final_features = np.zeros(self.portfolio_feat_dim, dtype=np.float32)
            common_len = min(len(features), self.portfolio_feat_dim)
            final_features[:common_len] = features[:common_len]
            return final_features

        return features

    def get_portfolio_observation(self) -> PortfolioObservationFeatures:
        """Get portfolio observation features"""
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

    def get_trader_vue_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trading performance metrics"""
        metrics: Dict[str, Any] = {
            "num_total_trades": len(self.trade_log),
            "num_winning_trades": 0,
            "num_losing_trades": 0,
            "total_net_profit_closed_trades": 0.0,
            "avg_net_pnl_per_trade": 0.0,
            "win_rate_pct": 0.0,
            "total_commissions_tradelog": self.total_commissions_session,
            "total_fees_tradelog": self.total_fees_session,
            "total_slippage_tradelog": self.total_slippage_cost_session,
            "total_volume_traded_tradelog": self.total_volume_traded_session,
        }

        if not self.trade_log:
            return metrics

        winning_trades = []
        losing_trades = []

        for trade in self.trade_log:
            pnl = trade.get('realized_pnl', 0.0)
            metrics["total_net_profit_closed_trades"] += pnl

            if pnl > 0:
                metrics["num_winning_trades"] += 1
                winning_trades.append(pnl)
            else:
                metrics["num_losing_trades"] += 1
                losing_trades.append(pnl)

        if metrics["num_total_trades"] > 0:
            metrics["avg_net_pnl_per_trade"] = metrics["total_net_profit_closed_trades"] / metrics["num_total_trades"]
            metrics["win_rate_pct"] = (metrics["num_winning_trades"] / metrics["num_total_trades"]) * 100

        if winning_trades:
            metrics["avg_winning_trade"] = np.mean(winning_trades)
            metrics["largest_win"] = max(winning_trades)

        if losing_trades:
            metrics["avg_losing_trade"] = np.mean(losing_trades)
            metrics["largest_loss"] = min(losing_trades)

        # Calculate drawdown
        if self.portfolio_value_history:
            equity_values = np.array([pv[1] for pv in self.portfolio_value_history])
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = (running_max - equity_values) / running_max
            metrics["max_portfolio_drawdown_pct"] = np.max(drawdowns) * 100

        return metrics