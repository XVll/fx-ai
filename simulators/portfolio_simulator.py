# simulators/portfolio_simulator.py - CRITICAL FIXES for PnL tracking
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
                'quantity': 0.0, 'avg_entry_price': 0.0, 'current_side': PositionSideEnum.FLAT,
                'market_value': 0.0, 'unrealized_pnl': 0.0,
                'entry_value_total': 0.0,
                'open_trade_id': None
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
        self.logger.info(f"PortfolioManager reset. Initial Capital: ${self.initial_capital:.2f}")

    def update_fill(self, fill: FillDetails):
        """FIXED: Properly track costs and realized PnL"""
        asset_id = fill['asset_id']
        pos_data = self.positions[asset_id]

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
        trade_side_for_fill = PositionSideEnum.LONG if fill[
                                                           'order_side'] == OrderSideEnum.BUY else PositionSideEnum.SHORT

        # DETAILED LOGGING for debugging
        self.logger.info(
            f"FILL PROCESSED: {asset_id} {fill['order_side'].value} {qty_change:.2f} @ ${fill_price:.4f} "
            f"| Commission: ${commission:.4f} | Fees: ${fees:.4f} | Slippage: ${slippage:.4f}"
        )

        # --- Handle Opening/Scaling In ---
        if (fill['order_side'] == OrderSideEnum.BUY and pos_data['current_side'] != PositionSideEnum.SHORT) or \
                (fill['order_side'] == OrderSideEnum.SELL and pos_data[
                    'current_side'] != PositionSideEnum.LONG and self.allow_shorting):

            self.cash -= fill_value if fill['order_side'] == OrderSideEnum.BUY else -fill_value

            if pos_data['current_side'] == PositionSideEnum.FLAT:  # New position
                pos_data['current_side'] = trade_side_for_fill
                pos_data['avg_entry_price'] = fill_price
                pos_data['quantity'] = qty_change
                pos_data['entry_value_total'] = fill_value

                trade_id = self._generate_trade_id()
                pos_data['open_trade_id'] = trade_id
                new_trade = TradeLogEntry(
                    trade_id=trade_id, asset_id=asset_id, side=trade_side_for_fill,
                    entry_timestamp=fill['fill_timestamp'], exit_timestamp=None,
                    entry_quantity_total=qty_change, avg_entry_price=fill_price,
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
                    f"NEW POSITION OPENED: {asset_id} {trade_side_for_fill.value} {qty_change:.2f} @ ${fill_price:.4f}")

            else:  # Scaling into existing position
                if pos_data['current_side'] == trade_side_for_fill and pos_data['open_trade_id'] in self.open_trades:
                    open_trade = self.open_trades[pos_data['open_trade_id']]
                    new_total_qty = pos_data['quantity'] + qty_change
                    new_entry_value_total = pos_data['entry_value_total'] + fill_value

                    pos_data['avg_entry_price'] = new_entry_value_total / new_total_qty if new_total_qty > 1e-9 else 0
                    pos_data['quantity'] = new_total_qty
                    pos_data['entry_value_total'] = new_entry_value_total

                    open_trade['entry_quantity_total'] = new_total_qty
                    open_trade['avg_entry_price'] = pos_data['avg_entry_price']
                    open_trade['commission_total'] += commission
                    open_trade['fees_total'] += fees
                    open_trade['slippage_total_trade_usd'] += slippage
                    open_trade['entry_fills'].append(fill)

                    self.logger.info(f"SCALED INTO POSITION: {asset_id} total qty now {new_total_qty:.2f}")

        # --- Handle Closing/Reducing ---
        elif (fill['order_side'] == OrderSideEnum.SELL and pos_data['current_side'] == PositionSideEnum.LONG) or \
                (fill['order_side'] == OrderSideEnum.BUY and pos_data[
                    'current_side'] == PositionSideEnum.SHORT and self.allow_shorting):

            self.cash += fill_value if fill['order_side'] == OrderSideEnum.SELL else -fill_value

            fill_qty = fill['executed_quantity']
            entry_price = pos_data['avg_entry_price']

            # Calculate realized PnL for this specific fill
            realized_pnl_for_this_fill = 0.0

            if pos_data['current_side'] == PositionSideEnum.LONG:
                # For long positions: (sell_price - entry_price) * quantity
                realized_pnl_for_this_fill = (fill_price - entry_price) * fill_qty
            else:  # SHORT
                # For short positions: (entry_price - cover_price) * quantity
                realized_pnl_for_this_fill = (entry_price - fill_price) * fill_qty

            # Deduct costs from realized PnL
            realized_pnl_for_this_fill -= (commission + fees)

            # CRITICAL: Update session realized PnL immediately
            self.realized_pnl_session += realized_pnl_for_this_fill

            self.logger.info(
                f"REALIZED PnL: ${realized_pnl_for_this_fill:.4f} | Entry: ${entry_price:.4f} | Exit: ${fill_price:.4f} | "
                f"Qty: {fill_qty:.2f} | SESSION TOTAL: ${self.realized_pnl_session:.4f}"
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
                    total_exit_value = (open_trade['avg_exit_price'] * current_exit_qty) + (fill_price * fill_qty)
                    open_trade['avg_exit_price'] = total_exit_value / (current_exit_qty + fill_qty)

                open_trade['exit_quantity_total'] = current_exit_qty + fill_qty

                # Update the trade's realized PnL
                if 'realized_pnl' not in open_trade or open_trade['realized_pnl'] is None:
                    open_trade['realized_pnl'] = realized_pnl_for_this_fill
                else:
                    open_trade['realized_pnl'] += realized_pnl_for_this_fill

                # Reduce position quantity
                pos_data['quantity'] -= fill_qty

                if pos_data['quantity'] < 1e-9:  # Position fully closed
                    pos_data['quantity'] = 0.0
                    pos_data['current_side'] = PositionSideEnum.FLAT
                    pos_data['open_trade_id'] = None
                    pos_data['entry_value_total'] = 0.0

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
                        f"POSITION CLOSED: {asset_id} | Trade PnL: ${open_trade['realized_pnl']:.4f} | "
                        f"Session Total: ${self.realized_pnl_session:.4f}"
                    )

        self.logger.debug(
            f"Cash after fill: ${self.cash:.2f} | Session costs: C:${self.total_commissions_session:.4f} F:${self.total_fees_session:.4f}")

    def update_market_value(self, asset_market_prices: Dict[str, float], current_timestamp: datetime):
        self.current_unrealized_pnl = 0.0
        total_positions_value = 0.0

        for asset_id, pos_data in self.positions.items():
            if pos_data['current_side'] != PositionSideEnum.FLAT and asset_id in asset_market_prices:
                current_price = asset_market_prices[asset_id]
                pos_qty = pos_data['quantity']
                avg_entry = pos_data['avg_entry_price']

                pos_data['market_value'] = pos_qty * current_price
                total_positions_value += pos_data['market_value']

                unreal_pnl_asset = 0.0
                if pos_data['current_side'] == PositionSideEnum.LONG:
                    unreal_pnl_asset = (current_price - avg_entry) * pos_qty
                elif pos_data['current_side'] == PositionSideEnum.SHORT and self.allow_shorting:
                    unreal_pnl_asset = (avg_entry - current_price) * pos_qty
                pos_data['unrealized_pnl'] = unreal_pnl_asset
                self.current_unrealized_pnl += unreal_pnl_asset

                # Update MFE/MAE for open trades
                open_trade_id = pos_data.get('open_trade_id')
                if open_trade_id and open_trade_id in self.open_trades:
                    trade = self.open_trades[open_trade_id]
                    trade_avg_entry_price = trade['avg_entry_price']
                    trade_entry_qty = trade['entry_quantity_total']
                    current_excursion_usd = 0.0

                    if trade['side'] == PositionSideEnum.LONG:
                        current_excursion_usd = (current_price - trade_avg_entry_price) * trade_entry_qty
                    elif trade['side'] == PositionSideEnum.SHORT:
                        current_excursion_usd = (trade_avg_entry_price - current_price) * trade_entry_qty

                    trade['max_favorable_excursion_usd'] = max(trade.get('max_favorable_excursion_usd', -np.inf),
                                                               current_excursion_usd)
                    trade['max_adverse_excursion_usd'] = min(trade.get('max_adverse_excursion_usd', np.inf),
                                                             current_excursion_usd)
            else:
                pos_data['market_value'] = 0.0
                pos_data['unrealized_pnl'] = 0.0

        self.current_total_equity = self.cash + total_positions_value
        self.portfolio_value_history.append((current_timestamp, self.current_total_equity))
        self.portfolio_feature_history.append(self._calculate_current_portfolio_features(current_timestamp))

    def _calculate_current_portfolio_features(self, timestamp: datetime) -> np.ndarray:
        if not self.tradable_assets:
            return np.zeros(self.portfolio_feat_dim if hasattr(self, 'portfolio_feat_dim') else 5, dtype=np.float32)

        asset_id = self.tradable_assets[0]
        pos_data = self.positions.get(asset_id, {})

        # 1. Normalized Position Size
        norm_pos_size = 0.0
        current_pos_market_value = pos_data.get('market_value', 0.0)
        if self.current_total_equity > 1e-9:
            max_theoretical_value_for_pos = self.current_total_equity * self.max_position_value_ratio
            if max_theoretical_value_for_pos > 1e-9:
                if pos_data.get('current_side') == PositionSideEnum.LONG:
                    norm_pos_size = current_pos_market_value / max_theoretical_value_for_pos
                elif pos_data.get('current_side') == PositionSideEnum.SHORT:
                    norm_pos_size = -abs(current_pos_market_value) / max_theoretical_value_for_pos
        norm_pos_size = np.clip(norm_pos_size, -1.0, 1.0)

        # 2. Normalized Unrealized P&L
        unreal_pnl_asset = pos_data.get('unrealized_pnl', 0.0)
        norm_unreal_pnl = 0.0
        entry_value_total_asset = pos_data.get('entry_value_total', 0.0)
        if abs(entry_value_total_asset) > 1e-9:
            norm_unreal_pnl = unreal_pnl_asset / abs(entry_value_total_asset)
        elif self.initial_capital > 1e-9:
            norm_unreal_pnl = unreal_pnl_asset / self.initial_capital
        norm_unreal_pnl = np.clip(norm_unreal_pnl, -1.0, 1.0)

        # 3. Normalized MAE
        norm_mae_pct = 0.0
        open_trade_id = pos_data.get('open_trade_id')
        if open_trade_id and open_trade_id in self.open_trades:
            current_trade = self.open_trades[open_trade_id]
            norm_mae_pct = current_trade.get('max_adverse_excursion_pct', 0.0)
            norm_mae_pct = np.clip(norm_mae_pct, -1.0, 0.0)

        # 4. Time in trade normalized
        time_in_trade_normalized = 0.0
        if open_trade_id and open_trade_id in self.open_trades:
            current_trade = self.open_trades[open_trade_id]
            entry_time = current_trade['entry_timestamp']
            duration_seconds = (timestamp - entry_time).total_seconds()

            if self.max_position_holding_seconds > 0:
                time_in_trade_normalized = duration_seconds / self.max_position_holding_seconds
            time_in_trade_normalized = np.clip(time_in_trade_normalized, 0.0, 1.0)

        # 5. Normalized cash percentage
        norm_cash_pct = 0.0
        if self.current_total_equity > 1e-9:
            norm_cash_pct = self.cash / self.current_total_equity
        norm_cash_pct = np.clip(norm_cash_pct, 0.0, 1.0)

        features = np.array([
            norm_pos_size,
            norm_unreal_pnl,
            norm_mae_pct,
            time_in_trade_normalized,
            norm_cash_pct
        ], dtype=np.float32)

        if hasattr(self, 'portfolio_feat_dim') and len(features) != self.portfolio_feat_dim:
            final_features = np.zeros(self.portfolio_feat_dim, dtype=np.float32)
            common_len = min(len(features), self.portfolio_feat_dim)
            final_features[:common_len] = features[:common_len]
            return final_features

        return features

    def get_portfolio_observation(self) -> PortfolioObservationFeatures:
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
        return PortfolioState(
            timestamp=current_timestamp, cash=self.cash, total_equity=self.current_total_equity,
            unrealized_pnl=self.current_unrealized_pnl, realized_pnl_session=self.realized_pnl_session,
            positions=self.positions.copy(),
            total_commissions_session=self.total_commissions_session,
            total_fees_session=self.total_fees_session,
            total_slippage_cost_session=self.total_slippage_cost_session,
            total_volume_traded_session=self.total_volume_traded_session,
            total_turnover_session=self.total_turnover_session,
            default_position_value=self.default_position_value
        )

    def _calculate_max_drawdown(self, equity_series_values: np.ndarray) -> Tuple[float, float]:
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
            f"TraderVue Metrics Calculated. Net Profit (Closed): ${metrics['total_net_profit_closed_trades']:.2f}")
        return metrics