from typing import Dict, Any

from metrics.collectors.trading_metrics import TradeMetricsCollector
from visualizations.episode_visualizer import EpisodeVisualizer
import logging

logger = logging.getLogger(__name__)


class VisualizationMetrics(TradeMetricsCollector):
    """Collects and generates episode visualizations for W&B."""
    
    def __init__(self):
        super().__init__()
        self.visualizer = EpisodeVisualizer()
        self.episode_buffer = []
        self.is_episode_active = False
        self.current_episode_num = 0
        
    def start_episode(self, episode_num: int, symbol: str, date: str):
        """Start collecting data for a new episode."""
        self.is_episode_active = True
        self.current_episode_num = episode_num
        self.episode_buffer = []
        
        episode_info = {
            'episode_num': episode_num,
            'symbol': symbol,
            'date': date
        }
        self.visualizer.start_episode(episode_info)
        
    def collect_step(self, step_data: Dict[str, Any]):
        """Collect data for a single environment step."""
        if not self.is_episode_active:
            return
            
        # Extract relevant data for visualization
        viz_data = {
            'price': step_data.get('price'),
            'volume': step_data.get('volume'),
            'position': step_data.get('position'),
            'reward': step_data.get('reward'),
            'action': step_data.get('action'),
            'vwap': step_data.get('vwap'),
            'rsi': step_data.get('rsi'),
            'volatility': step_data.get('volatility'),
            'sma_fast': step_data.get('sma_fast'),
            'sma_slow': step_data.get('sma_slow'),
        }
        
        # Add any additional features
        for key, value in step_data.items():
            if key.startswith(('momentum', 'volume_', 'feature_')):
                viz_data[key] = value
                
        self.visualizer.add_step(viz_data)
        
    def collect_trade(self, trade_data: Dict[str, Any]):
        """Record a trade event."""
        if not self.is_episode_active:
            return
            
        trade_info = {
            'step': len(self.visualizer.episode_data) - 1,  # Current step
            'action': trade_data.get('action'),
            'price': trade_data.get('price'),
            'quantity': trade_data.get('quantity', 1),
            'pnl': trade_data.get('pnl', 0),
            'fees': trade_data.get('fees', 0)
        }
        self.visualizer.add_trade(trade_info)
        
    def end_episode(self) -> Dict[str, Any]:
        """Generate visualizations at the end of an episode."""
        if not self.is_episode_active:
            return {}
            
        self.is_episode_active = False
        metrics = {}
        
        try:
            # Create main episode chart
            episode_chart = self.visualizer.create_episode_chart()
            if episode_chart:
                metrics[f'episode_{self.current_episode_num}_chart'] = episode_chart
                
            # Create debug charts
            debug_charts = self.visualizer.create_debug_charts()
            for i, chart in enumerate(debug_charts):
                metrics[f'episode_{self.current_episode_num}_debug_{i}'] = chart
                
        except Exception as e:
            logger.error(f"Error generating episode visualizations: {e}")
            
        return metrics
        
    def collect(self, **kwargs) -> Dict[str, Any]:
        """Standard collect method for compatibility."""
        # This collector works through event-based methods
        return {}
        
    def reset(self):
        """Reset the collector state."""
        self.is_episode_active = False
        self.episode_buffer = []
        self.visualizer = EpisodeVisualizer()