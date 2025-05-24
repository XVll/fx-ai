from typing import Dict, Any, Optional
import logging
from .live_dashboard_v2 import LiveTradingDashboard

logger = logging.getLogger(__name__)


class DashboardMetricsCollector:
    """Integrates the live dashboard with the metrics system"""
    
    def __init__(self, dashboard: Optional[LiveTradingDashboard] = None):
        self.dashboard = dashboard or LiveTradingDashboard()
        self.is_started = False
        self.current_episode_num = 0
        
    def start(self, open_browser: bool = True):
        """Start the dashboard"""
        if not self.is_started:
            self.dashboard.start(open_browser=open_browser)
            self.is_started = True
            logger.info("Dashboard metrics collector started")
    
    def stop(self):
        """Stop the dashboard"""
        if self.is_started:
            self.dashboard.stop()
            self.is_started = False
    
    def on_step(self, step_data: Dict[str, Any]):
        """Handle environment step update"""
        if not self.is_started:
            return
        
        # Update market data
        market_update = {
            'symbol': step_data.get('symbol', 'N/A'),
            'price': step_data.get('price', 0),
            'bid': step_data.get('bid', step_data.get('price', 0) * 0.9999),
            'ask': step_data.get('ask', step_data.get('price', 0) * 1.0001),
            'volume': step_data.get('volume', 0),
        }
        self.dashboard.update_market(market_update)
        
        # Update position
        position_update = {
            'symbol': step_data.get('symbol', 'N/A'),
            'quantity': step_data.get('position', 0),
            'avg_entry_price': step_data.get('avg_entry_price', step_data.get('price', 0)),
        }
        self.dashboard.update_position(position_update)
        
        # Update portfolio
        portfolio_update = {
            'equity': step_data.get('equity', 25000),
            'cash': step_data.get('cash', 25000),
            'realized_pnl': step_data.get('realized_pnl', 0),
            'unrealized_pnl': step_data.get('unrealized_pnl', 0),
        }
        self.dashboard.update_portfolio(portfolio_update)
        
        # Update action
        if 'action' in step_data and 'reward' in step_data:
            self.dashboard.update_action(
                step=step_data.get('step', 0),
                action_type=step_data.get('action', 'HOLD').upper(),
                size=step_data.get('size', 1.0),
                reward=step_data.get('reward', 0.0)
            )
    
    def on_trade(self, trade_data: Dict[str, Any]):
        """Handle trade execution"""
        if not self.is_started:
            return
        
        trade_update = {
            'side': trade_data.get('action', 'UNKNOWN').upper(),
            'quantity': abs(trade_data.get('quantity', 0)),
            'symbol': trade_data.get('symbol', 'N/A'),
            'entry_price': trade_data.get('price', 0),
            'exit_price': trade_data.get('exit_price'),
            'pnl': trade_data.get('pnl', 0),
            'fees': trade_data.get('fees', 0)
        }
        self.dashboard.update_trade(trade_update)
    
    def on_episode_start(self, episode_num: int):
        """Handle episode start"""
        if not self.is_started:
            return
        
        self.current_episode_num = episode_num
        self.dashboard.start_episode(episode_num)
    
    def on_episode_end(self, episode_data: Dict[str, Any]):
        """Handle episode end"""
        if not self.is_started:
            return
        
        reason = episode_data.get('termination_reason', 'Completed')
        self.dashboard.end_episode(reason)
    
    def on_training_update(self, training_data: Dict[str, Any]):
        """Handle training progress updates"""
        if not self.is_started:
            return
        
        self.dashboard.update_training_progress(training_data)
    
    def on_ppo_metrics(self, ppo_data: Dict[str, Any]):
        """Handle PPO metrics updates"""
        if not self.is_started:
            return
        
        self.dashboard.update_ppo_metrics(ppo_data)
    
    def set_model_info(self, model_name: str):
        """Set model information"""
        if not self.is_started:
            return
        
        self.dashboard.set_model_info(model_name)