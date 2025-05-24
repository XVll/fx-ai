from typing import Dict, Any, Optional
import logging
from .live_dashboard import LiveTradingDashboard

logger = logging.getLogger(__name__)


class DashboardMetricsCollector:
    """Integrates the live dashboard with the metrics system"""
    
    def __init__(self, dashboard: Optional[LiveTradingDashboard] = None):
        self.dashboard = dashboard or LiveTradingDashboard()
        self.is_started = False
        
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
            
        # Extract relevant data for dashboard
        update_data = {
            'step': step_data.get('step', 0),
            'price': step_data.get('price', 0),
            'volume': step_data.get('volume', 0),
            'position': step_data.get('position', 0),
            'reward': step_data.get('reward', 0),
            'equity': step_data.get('equity', 25000),
            'action': step_data.get('action', 'HOLD')
        }
        
        self.dashboard.update_step(update_data)
    
    def on_trade(self, trade_data: Dict[str, Any]):
        """Handle trade execution"""
        if not self.is_started:
            return
            
        self.dashboard.update_trade(trade_data)
    
    def on_episode_end(self, episode_data: Dict[str, Any]):
        """Handle episode end"""
        if not self.is_started:
            return
            
        self.dashboard.update_episode(episode_data)
    
    def on_features(self, feature_data: Dict[str, Any]):
        """Handle feature updates"""
        if not self.is_started:
            return
            
        # Select important features for visualization
        selected_features = {}
        feature_names = ['momentum', 'rsi', 'volatility', 'volume_ratio', 
                        'price_velocity', 'bid_ask_spread', 'order_imbalance']
        
        for name in feature_names:
            for key, value in feature_data.items():
                if name in key.lower():
                    selected_features[key] = value
        
        if selected_features:
            self.dashboard.update_features(selected_features)