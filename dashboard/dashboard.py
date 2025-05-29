# dashboard/dashboard.py - Simple dashboard for training metrics

import threading
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DashboardState:
    """Simple state container for dashboard data"""
    
    # Training state
    mode: str = "Idle"
    stage: str = "Not Started"
    stage_status: str = ""
    overall_progress: float = 0.0
    stage_progress: float = 0.0
    
    # Counters
    updates: int = 0
    global_steps: int = 0
    total_episodes: int = 0
    
    # Performance metrics
    steps_per_second: float = 0.0
    episodes_per_hour: float = 0.0
    time_per_update: float = 0.0
    time_per_episode: float = 0.0
    
    # PPO metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    explained_variance: float = 0.0
    learning_rate: float = 0.0
    
    # Trading metrics
    mean_episode_reward: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Episode tracking
    episode_rewards: list = field(default_factory=list)
    episode_data: list = field(default_factory=list)
    
    # Reward components
    reward_components: Dict[str, float] = field(default_factory=dict)
    
    # Trade data
    trades: list = field(default_factory=list)
    
    def update_training_state(self, data: Dict[str, Any]):
        """Update training state from metrics"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def update_episode_data(self, episode_data: Dict[str, Any]):
        """Update episode tracking"""
        if 'episode_reward' in episode_data:
            self.episode_rewards.append(episode_data['episode_reward'])
            # Keep only last 100 episodes
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
                
        self.episode_data.append({
            'timestamp': datetime.now().isoformat(),
            **episode_data
        })
        # Keep only last 50 episodes
        if len(self.episode_data) > 50:
            self.episode_data.pop(0)
            
    def update_reward_components(self, components: Dict[str, float]):
        """Update reward component tracking"""
        self.reward_components.update(components)
        
    def update_trade_data(self, trade_data: Dict[str, Any]):
        """Update trade tracking"""
        self.trades.append({
            'timestamp': datetime.now().isoformat(),
            **trade_data
        })
        # Keep only last 100 trades
        if len(self.trades) > 100:
            self.trades.pop(0)
            
    def update_reset_point_performance(self, reset_idx: int, performance: Dict[str, float]):
        """Update reset point performance (placeholder for now)"""
        pass


@dataclass 
class MomentumDay:
    """Simple momentum day data structure"""
    date: Any
    symbol: str
    activity_score: float
    max_intraday_move: float = 0.0
    volume_multiplier: float = 0.0
    reset_points: list = field(default_factory=list)
    is_front_side: bool = False
    is_back_side: bool = False
    halt_count: int = 0


class MomentumDashboard:
    """Simple dashboard for training metrics - placeholder implementation"""
    
    def __init__(self, port: int = 8050, update_interval: int = 1000):
        self.port = port
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        
        # State
        self.state = DashboardState()
        self.is_running = False
        self._momentum_day: Optional[MomentumDay] = None
        self._curriculum_progress: float = 0.0
        self._curriculum_strategy: Optional[str] = None
        
    def start(self, open_browser: bool = True):
        """Start the dashboard (simplified implementation)"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info(f"📊 Simple dashboard started on port {self.port}")
        
        if open_browser:
            self.logger.info(f"🌐 Dashboard would be available at http://localhost:{self.port}")
            self.logger.info("📝 This is a simplified dashboard implementation")
            
    def stop(self):
        """Stop the dashboard"""
        if self.is_running:
            self.is_running = False
            self.logger.info("📊 Dashboard stopped")
            
    def update_training_state(self, data: Dict[str, Any]):
        """Update training state"""
        if self.is_running:
            self.state.update_training_state(data)
            
    def update_momentum_day(self, momentum_day: MomentumDay):
        """Update momentum day"""
        if self.is_running:
            self._momentum_day = momentum_day
            self.logger.info(f"📅 Dashboard: Updated momentum day to {momentum_day.date}")
            
    def update_curriculum_progress(self, progress: float, strategy: Optional[str] = None):
        """Update curriculum progress"""
        if self.is_running:
            self._curriculum_progress = progress
            self._curriculum_strategy = strategy
            self.logger.info(f"📚 Dashboard: Curriculum progress {progress:.1%}")
            
    def is_dashboard_running(self) -> bool:
        """Check if dashboard is running"""
        return self.is_running
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for debugging"""
        return {
            'mode': self.state.mode,
            'stage': self.state.stage,
            'updates': self.state.updates,
            'global_steps': self.state.global_steps,
            'total_episodes': self.state.total_episodes,
            'mean_reward': self.state.mean_episode_reward,
            'momentum_day': self._momentum_day.symbol if self._momentum_day else None,
            'curriculum_progress': self._curriculum_progress,
            'is_running': self.is_running
        }