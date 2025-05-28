"""
Dashboard integration for momentum-based trading system.
Connects the momentum dashboard with the metrics and training systems.
"""

from typing import Dict, Any, Optional
import logging
from .dashboard import MomentumDashboard
from .dashboard_data import MomentumDay
from datetime import date

logger = logging.getLogger(__name__)


class DashboardMetricsCollector:
    """Integrates the momentum dashboard with the metrics system"""
    
    def __init__(self, dashboard: Optional[MomentumDashboard] = None, port: int = 8050):
        self.dashboard = dashboard or MomentumDashboard(port=port)
        self.is_started = False
        self.current_episode_num = 0
        
    def start(self, open_browser: bool = True):
        """Start the dashboard"""
        if not self.is_started:
            self.dashboard.start(open_browser=open_browser)
            self.is_started = True
            logger.info("🚀 Momentum dashboard metrics collector started")
    
    def stop(self):
        """Stop the dashboard"""
        if self.is_started:
            self.dashboard.stop()
            self.is_started = False
            logger.info("🛑 Dashboard metrics collector stopped")
    
    def set_model_info(self, model_name: str):
        """Set model information for the dashboard"""
        if self.is_started:
            try:
                # Update the dashboard with model information
                self.dashboard.update_training_state({
                    'model_name': model_name,
                    'model_type': 'PPO_Transformer'
                })
                logger.info(f"Dashboard model info set to: {model_name}")
            except Exception as e:
                logger.error(f"Error setting model info in dashboard: {e}")
    
    def on_training_update(self, training_data: Dict[str, Any]):
        """Handle training update from PPO agent"""
        try:
            if self.is_started:
                self.dashboard.update_training_state(training_data)
        except Exception as e:
            logger.error(f"Error updating dashboard with training data: {e}")
    
    def on_ppo_metrics(self, ppo_data: Dict[str, Any]):
        """Handle PPO-specific metrics"""
        try:
            if self.is_started:
                # PPO metrics are included in training updates
                self.dashboard.update_training_state(ppo_data)
        except Exception as e:
            logger.error(f"Error updating dashboard with PPO metrics: {e}")
    
    def on_episode_end(self, episode_data: Dict[str, Any]):
        """Handle episode completion"""
        try:
            if self.is_started:
                self.current_episode_num += 1
                episode_data['episode_num'] = self.current_episode_num
                self.dashboard.state.update_episode_data(episode_data)
        except Exception as e:
            logger.error(f"Error updating dashboard with episode data: {e}")
    
    def on_momentum_day_change(self, momentum_day_data: Dict[str, Any]):
        """Handle momentum day change"""
        try:
            if self.is_started and momentum_day_data:
                # Convert to MomentumDay object
                momentum_day = MomentumDay(
                    date=momentum_day_data.get('date'),
                    symbol=momentum_day_data.get('symbol', 'UNKNOWN'),
                    activity_score=momentum_day_data.get('activity_score', 0.0),
                    max_intraday_move=momentum_day_data.get('max_intraday_move', 0.0),
                    volume_multiplier=momentum_day_data.get('volume_multiplier', 0.0),
                    reset_points=momentum_day_data.get('reset_points', []),
                    is_front_side=momentum_day_data.get('is_front_side', False),
                    is_back_side=momentum_day_data.get('is_back_side', False),
                    halt_count=momentum_day_data.get('halt_count', 0)
                )
                
                self.dashboard.update_momentum_day(momentum_day)
                logger.info(f"📅 Dashboard updated with momentum day: {momentum_day.date}")
                
        except Exception as e:
            logger.error(f"Error updating dashboard with momentum day: {e}")
    
    def on_curriculum_progress(self, progress: float, strategy: str = None):
        """Handle curriculum learning progress update"""
        try:
            if self.is_started:
                self.dashboard.update_curriculum_progress(progress, strategy)
        except Exception as e:
            logger.error(f"Error updating dashboard with curriculum progress: {e}")
    
    def on_reward_components(self, components: Dict[str, float]):
        """Handle reward components update"""
        try:
            if self.is_started:
                self.dashboard.state.update_reward_components(components)
        except Exception as e:
            logger.error(f"Error updating dashboard with reward components: {e}")
    
    def on_reset_point_performance(self, reset_point_idx: int, performance_data: Dict[str, float]):
        """Handle reset point performance update"""
        try:
            if self.is_started:
                self.dashboard.state.update_reset_point_performance(reset_point_idx, performance_data)
        except Exception as e:
            logger.error(f"Error updating dashboard with reset point performance: {e}")
    
    def on_step(self, step_data: Dict[str, Any]):
        """Handle step update from environment"""
        try:
            if self.is_started:
                # Update dashboard with step-level information
                self.dashboard.state.update_step_data(step_data)
        except Exception as e:
            logger.error(f"Error updating dashboard with step data: {e}")
    
    def on_trade(self, trade_data: Dict[str, Any]):
        """Handle trade execution update"""
        try:
            if self.is_started:
                # Update dashboard with trade information
                self.dashboard.state.update_trade_data(trade_data)
        except Exception as e:
            logger.error(f"Error updating dashboard with trade data: {e}")
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL"""
        return f"http://127.0.0.1:{self.dashboard.port}"
    
    def is_dashboard_running(self) -> bool:
        """Check if dashboard is running"""
        return self.is_started and self.dashboard.is_dashboard_running()


class MockDashboardCollector:
    """Mock dashboard collector for when dashboard is disabled"""
    
    def __init__(self):
        self.is_started = False
    
    def start(self, open_browser: bool = True):
        """Mock start"""
        self.is_started = True
        logger.info("📊 Mock dashboard collector started (dashboard disabled)")
    
    def stop(self):
        """Mock stop"""
        self.is_started = False
    
    def on_training_update(self, training_data: Dict[str, Any]):
        """Mock training update"""
        pass
    
    def on_ppo_metrics(self, ppo_data: Dict[str, Any]):
        """Mock PPO metrics"""
        pass
    
    def on_episode_end(self, episode_data: Dict[str, Any]):
        """Mock episode end"""
        pass
    
    def on_momentum_day_change(self, momentum_day_data: Dict[str, Any]):
        """Mock momentum day change"""
        pass
    
    def on_curriculum_progress(self, progress: float, strategy: str = None):
        """Mock curriculum progress"""
        pass
    
    def on_reward_components(self, components: Dict[str, float]):
        """Mock reward components"""
        pass
    
    def on_reset_point_performance(self, reset_point_idx: int, performance_data: Dict[str, float]):
        """Mock reset point performance"""
        pass
    
    def on_step(self, step_data: Dict[str, Any]):
        """Mock step update"""
        pass
    
    def on_trade(self, trade_data: Dict[str, Any]):
        """Mock trade update"""
        pass
    
    def get_dashboard_url(self) -> str:
        """Mock URL"""
        return "http://localhost:8050 (disabled)"
    
    def is_dashboard_running(self) -> bool:
        """Mock running check"""
        return False