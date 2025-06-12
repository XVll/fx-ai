"""
Portfolio metrics callback for WandB integration.

Tracks portfolio state including positions, PnL, drawdowns,
and risk metrics throughout training.
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None

from callbacks.core.base import BaseCallback

logger = logging.getLogger(__name__)


class PortfolioMetricsCallback(BaseCallback):
    """
    Specialized callback for portfolio metrics with step-level tracking.
    
    Tracks:
    - Position metrics: size, direction, utilization
    - PnL metrics: realized, unrealized, total
    - Risk metrics: drawdowns, exposure, margin
    - Performance metrics: returns, volatility, ratios
    """
    
    def __init__(self, buffer_size: int = 1000, log_frequency: int = 10, enabled: bool = True):
        """
        Initialize portfolio metrics callback.
        
        Args:
            buffer_size: Size of local buffers for rolling calculations
            log_frequency: How often to log step-level metrics (every N steps)
            enabled: Whether callback is active
        """
        super().__init__(name="PortfolioMetrics", enabled=enabled)
        
        self.buffer_size = buffer_size
        self.log_frequency = log_frequency
        
        # Position tracking buffers
        self.positions = deque(maxlen=buffer_size)
        self.position_values = deque(maxlen=buffer_size)
        self.position_directions = deque(maxlen=buffer_size)  # 1 for long, -1 for short, 0 for flat
        
        # PnL tracking buffers
        self.unrealized_pnls = deque(maxlen=buffer_size)
        self.realized_pnls = deque(maxlen=buffer_size)
        self.total_pnls = deque(maxlen=buffer_size)
        self.cumulative_pnls = deque(maxlen=buffer_size)
        
        # Risk metrics buffers
        self.drawdowns = deque(maxlen=buffer_size)
        self.exposures = deque(maxlen=buffer_size)
        self.margin_used = deque(maxlen=buffer_size)
        self.leverage_ratios = deque(maxlen=buffer_size)
        
        # Portfolio state tracking
        self.portfolio_values = deque(maxlen=buffer_size)
        self.cash_balances = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.steps_logged = 0
        self.portfolio_updates_logged = 0
        self.max_portfolio_value = 0
        self.min_portfolio_value = float('inf')
        
        if wandb is None:
            self.logger.warning("wandb not installed - portfolio metrics will not be logged")
        
        self.logger.info(f"💼 Portfolio metrics callback initialized (buffer_size={buffer_size}, log_freq={log_frequency})")
    
    def on_step_end(self, context: Dict[str, Any]) -> None:
        """Collect portfolio state at each step."""
        if not wandb or not wandb.run:
            return
        
        # Extract portfolio data from context
        position = context.get('position', 0)
        position_value = context.get('position_value', 0)
        unrealized_pnl = context.get('unrealized_pnl', 0)
        portfolio_value = context.get('portfolio_value', 0)
        cash_balance = context.get('cash_balance', 0)
        margin_used = context.get('margin_used', 0)
        
        # Calculate derived metrics
        position_direction = 1 if position > 0 else (-1 if position < 0 else 0)
        exposure = abs(position_value)
        leverage_ratio = exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Add to buffers
        self.positions.append(position)
        self.position_values.append(position_value)
        self.position_directions.append(position_direction)
        self.unrealized_pnls.append(unrealized_pnl)
        self.portfolio_values.append(portfolio_value)
        self.cash_balances.append(cash_balance)
        self.exposures.append(exposure)
        self.margin_used.append(margin_used)
        self.leverage_ratios.append(leverage_ratio)
        
        # Update portfolio extremes
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        if portfolio_value < self.min_portfolio_value:
            self.min_portfolio_value = portfolio_value
        
        # Calculate drawdown
        if len(self.portfolio_values) > 1:
            peak_value = max(self.portfolio_values)
            current_drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0
            self.drawdowns.append(current_drawdown)
        else:
            self.drawdowns.append(0)
        
        # Log at specified frequency to avoid spam
        if self.steps_seen % self.log_frequency == 0:
            metrics = {
                # Current portfolio state
                'portfolio/position': position,
                'portfolio/position_value': position_value,
                'portfolio/position_direction': position_direction,
                'portfolio/unrealized_pnl': unrealized_pnl,
                'portfolio/portfolio_value': portfolio_value,
                'portfolio/cash_balance': cash_balance,
                'portfolio/exposure': exposure,
                'portfolio/margin_used': margin_used,
                'portfolio/leverage_ratio': leverage_ratio,
                'portfolio/current_drawdown': current_drawdown,
                'portfolio/step_number': self.steps_seen
            }
            
            # Add rolling metrics
            self._add_rolling_portfolio_metrics(metrics)
            
            # Add risk metrics
            self._add_risk_metrics(metrics)
            
            # Log to WandB
            wandb.log(metrics)
            
            self.steps_logged += 1
    
    def on_custom_event(self, event_name: str, context: Dict[str, Any]) -> None:
        """Handle portfolio-related custom events."""
        if not wandb or not wandb.run:
            return
        
        if event_name == 'portfolio_update':
            self._handle_portfolio_update(context)
        elif event_name == 'position_change':
            self._handle_position_change(context)
        elif event_name == 'pnl_realization':
            self._handle_pnl_realization(context)
        elif event_name == 'margin_call':
            self._handle_margin_call(context)
    
    def _handle_portfolio_update(self, context: Dict[str, Any]) -> None:
        """Handle detailed portfolio update events."""
        raw = context.get('raw', {})
        
        # Extract detailed portfolio state
        total_equity = raw.get('total_equity', 0)
        buying_power = raw.get('buying_power', 0)
        maintenance_margin = raw.get('maintenance_margin', 0)
        initial_margin = raw.get('initial_margin', 0)
        
        metrics = {
            'portfolio_detail/total_equity': total_equity,
            'portfolio_detail/buying_power': buying_power,
            'portfolio_detail/maintenance_margin': maintenance_margin,
            'portfolio_detail/initial_margin': initial_margin,
            'portfolio_detail/margin_ratio': maintenance_margin / total_equity if total_equity > 0 else 0
        }
        
        wandb.log(metrics)
        self.portfolio_updates_logged += 1
    
    def _handle_position_change(self, context: Dict[str, Any]) -> None:
        """Handle position change events."""
        raw = context.get('raw', {})
        
        old_position = raw.get('old_position', 0)
        new_position = raw.get('new_position', 0)
        position_delta = new_position - old_position
        trade_size = abs(position_delta)
        
        # Determine trade type
        if old_position == 0 and new_position != 0:
            trade_type = 'open'
        elif old_position != 0 and new_position == 0:
            trade_type = 'close'
        elif (old_position > 0 and new_position < 0) or (old_position < 0 and new_position > 0):
            trade_type = 'reverse'
        else:
            trade_type = 'scale'
        
        metrics = {
            'position/old_position': old_position,
            'position/new_position': new_position,
            'position/position_delta': position_delta,
            'position/trade_size': trade_size,
            'position/trade_type': trade_type
        }
        
        wandb.log(metrics)
    
    def _handle_pnl_realization(self, context: Dict[str, Any]) -> None:
        """Handle PnL realization events."""
        raw = context.get('raw', {})
        
        realized_pnl = raw.get('realized_pnl', 0)
        trade_return = raw.get('trade_return', 0)
        hold_duration = raw.get('hold_duration', 0)
        
        self.realized_pnls.append(realized_pnl)
        
        metrics = {
            'pnl/realized_pnl': realized_pnl,
            'pnl/trade_return': trade_return,
            'pnl/hold_duration': hold_duration
        }
        
        # Add cumulative realized PnL
        if len(self.realized_pnls) > 0:
            metrics['pnl/cumulative_realized_pnl'] = sum(self.realized_pnls)
        
        wandb.log(metrics)
    
    def _handle_margin_call(self, context: Dict[str, Any]) -> None:
        """Handle margin call events."""
        raw = context.get('raw', {})
        
        margin_call_amount = raw.get('margin_call_amount', 0)
        current_equity = raw.get('current_equity', 0)
        required_equity = raw.get('required_equity', 0)
        
        metrics = {
            'risk/margin_call': 1,
            'risk/margin_call_amount': margin_call_amount,
            'risk/current_equity': current_equity,
            'risk/required_equity': required_equity,
            'risk/equity_deficit': required_equity - current_equity
        }
        
        wandb.log(metrics)
        self.logger.warning(f"📢 Margin call logged: ${margin_call_amount:.2f}")
    
    def _add_rolling_portfolio_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling portfolio metrics to the metrics dict."""
        # 50-step rolling metrics
        if len(self.portfolio_values) >= 50:
            recent_50_values = list(self.portfolio_values)[-50:]
            recent_50_positions = list(self.positions)[-50:]
            recent_50_exposures = list(self.exposures)[-50:]
            
            metrics.update({
                'rolling_50/avg_portfolio_value': np.mean(recent_50_values),
                'rolling_50/portfolio_volatility': np.std(recent_50_values),
                'rolling_50/avg_position': np.mean(recent_50_positions),
                'rolling_50/avg_exposure': np.mean(recent_50_exposures),
                'rolling_50/position_utilization': np.mean([abs(p) for p in recent_50_positions])
            })
        
        # 200-step rolling metrics
        if len(self.portfolio_values) >= 200:
            recent_200_values = list(self.portfolio_values)[-200:]
            recent_200_unrealized = list(self.unrealized_pnls)[-200:]
            recent_200_drawdowns = list(self.drawdowns)[-200:]
            recent_200_leverage = list(self.leverage_ratios)[-200:]
            
            # Calculate returns for performance metrics
            returns = [(recent_200_values[i] - recent_200_values[i-1]) / recent_200_values[i-1] 
                      for i in range(1, len(recent_200_values)) if recent_200_values[i-1] != 0]
            
            metrics.update({
                'rolling_200/avg_portfolio_value': np.mean(recent_200_values),
                'rolling_200/max_portfolio_value': np.max(recent_200_values),
                'rolling_200/min_portfolio_value': np.min(recent_200_values),
                'rolling_200/avg_unrealized_pnl': np.mean(recent_200_unrealized),
                'rolling_200/max_drawdown': np.min(recent_200_drawdowns),
                'rolling_200/avg_leverage': np.mean(recent_200_leverage),
                'rolling_200/max_leverage': np.max(recent_200_leverage)
            })
            
            # Add return-based metrics
            if returns:
                metrics.update({
                    'rolling_200/avg_return': np.mean(returns),
                    'rolling_200/return_volatility': np.std(returns),
                    'rolling_200/sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8)
                })
    
    def _add_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add risk management metrics."""
        if len(self.drawdowns) >= 20:
            recent_drawdowns = list(self.drawdowns)[-20:]
            
            # Risk metrics
            current_drawdown = recent_drawdowns[-1]
            max_drawdown_20 = np.min(recent_drawdowns)
            avg_drawdown_20 = np.mean(recent_drawdowns)
            
            metrics.update({
                'risk/current_drawdown_pct': current_drawdown * 100,
                'risk/max_drawdown_20_pct': max_drawdown_20 * 100,
                'risk/avg_drawdown_20_pct': avg_drawdown_20 * 100
            })
        
        if len(self.leverage_ratios) >= 50:
            recent_leverage = list(self.leverage_ratios)[-50:]
            
            metrics.update({
                'risk/avg_leverage_50': np.mean(recent_leverage),
                'risk/max_leverage_50': np.max(recent_leverage),
                'risk/leverage_p95': np.percentile(recent_leverage, 95)
            })
        
        # Position concentration metrics
        if len(self.position_directions) >= 100:
            recent_directions = list(self.position_directions)[-100:]
            
            long_pct = np.mean([d == 1 for d in recent_directions]) * 100
            short_pct = np.mean([d == -1 for d in recent_directions]) * 100
            flat_pct = np.mean([d == 0 for d in recent_directions]) * 100
            
            metrics.update({
                'position/long_percentage_100': long_pct,
                'position/short_percentage_100': short_pct,
                'position/flat_percentage_100': flat_pct
            })
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Log episode-level portfolio summary."""
        if not wandb or not wandb.run or len(self.portfolio_values) == 0:
            return
        
        # Calculate episode portfolio performance
        episode_start_value = context.get('episode_start_portfolio_value', 
                                         self.portfolio_values[0] if self.portfolio_values else 0)
        episode_end_value = context.get('episode_end_portfolio_value',
                                       self.portfolio_values[-1] if self.portfolio_values else 0)
        
        episode_return = ((episode_end_value - episode_start_value) / episode_start_value 
                         if episode_start_value != 0 else 0)
        
        episode_metrics = {
            'episode_portfolio/start_value': episode_start_value,
            'episode_portfolio/end_value': episode_end_value,
            'episode_portfolio/episode_return': episode_return,
            'episode_portfolio/episode_return_pct': episode_return * 100,
            'episode_portfolio/max_position': max(self.positions) if self.positions else 0,
            'episode_portfolio/min_position': min(self.positions) if self.positions else 0,
            'episode_portfolio/avg_leverage': np.mean(list(self.leverage_ratios)) if self.leverage_ratios else 0
        }
        
        # Add episode drawdown stats
        if self.drawdowns:
            episode_max_dd = min(self.drawdowns)
            episode_metrics.update({
                'episode_portfolio/max_drawdown': episode_max_dd,
                'episode_portfolio/max_drawdown_pct': episode_max_dd * 100
            })
        
        wandb.log(episode_metrics)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics."""
        return {
            'buffer_size': self.buffer_size,
            'log_frequency': self.log_frequency,
            'steps_logged': self.steps_logged,
            'portfolio_updates_logged': self.portfolio_updates_logged,
            'max_portfolio_value': self.max_portfolio_value,
            'min_portfolio_value': self.min_portfolio_value,
            'current_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 0,
            'portfolio_values_buffer_size': len(self.portfolio_values),
            'positions_buffer_size': len(self.positions),
            'current_position': self.positions[-1] if self.positions else 0,
            'current_drawdown': self.drawdowns[-1] if self.drawdowns else 0
        }