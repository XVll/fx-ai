"""
Dashboard data structures and state management for momentum-based trading.
Optimized for momentum training with curriculum learning and multi-day episodes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from collections import deque, defaultdict
from datetime import datetime, date
import time
import pandas as pd
import numpy as np


@dataclass
class MomentumDay:
    """Represents a momentum trading day"""
    date: date
    symbol: str
    activity_score: float
    max_intraday_move: float
    volume_multiplier: float
    reset_points: List[Dict[str, Any]] = field(default_factory=list)
    is_front_side: bool = False
    is_back_side: bool = False
    halt_count: int = 0
    episodes_trained: int = 0
    best_reward: float = float('-inf')
    avg_reward: float = 0.0
    
    @property
    def difficulty_level(self) -> str:
        """Get difficulty level based on activity score"""
        if self.activity_score >= 0.8:
            return "Expert"
        elif self.activity_score >= 0.6:
            return "Hard"
        elif self.activity_score >= 0.4:
            return "Medium"
        else:
            return "Easy"
    
    @property
    def momentum_type(self) -> str:
        """Get momentum type"""
        if self.is_front_side:
            return "Bullish"
        elif self.is_back_side:
            return "Bearish"
        else:
            return "Mixed"


@dataclass
class TrainingState:
    """Current training state"""
    mode: str = "Stopped"  # Training/Evaluating/Stopped
    stage: str = "Initialization"
    updates: int = 0
    global_steps: int = 0
    total_episodes: int = 0
    overall_progress: float = 0.0
    stage_progress: float = 0.0
    stage_status: str = ""
    
    # Performance metrics
    steps_per_second: float = 0.0
    episodes_per_hour: float = 0.0
    time_per_update: float = 0.0
    time_per_episode: float = 0.0
    eta_hours: float = 0.0
    
    # PPO metrics
    lr: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    explained_variance: float = 0.0
    
    # Momentum-specific
    current_momentum_day: Optional[MomentumDay] = None
    curriculum_progress: float = 0.0
    curriculum_strategy: str = "quality_based"
    used_reset_point_indices: set = field(default_factory=set)
    momentum_day_switches: int = 0
    
    # Session info
    session_start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None


@dataclass
class RewardComponents:
    """Reward system components tracking"""
    components: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0
    positive_contributions: Dict[str, float] = field(default_factory=dict)
    negative_contributions: Dict[str, float] = field(default_factory=dict)
    component_history: Dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    
    def update_component(self, name: str, value: float):
        """Update a reward component"""
        self.components[name] = value
        self.component_history[name].append(value)
        
        if value > 0:
            self.positive_contributions[name] = value
        else:
            self.negative_contributions[name] = value
    
    def get_component_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a component"""
        history = list(self.component_history.get(name, []))
        if not history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": np.mean(history),
            "std": np.std(history),
            "min": np.min(history),
            "max": np.max(history),
            "latest": history[-1] if history else 0.0
        }


class DashboardState:
    """Central dashboard state management for momentum trading"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Core state
        self.training_state = TrainingState()
        self.reward_components = RewardComponents()
        
        # Historical data
        self.training_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.episode_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.curriculum_history: Deque[float] = deque(maxlen=max_history)
        
        # Momentum-specific tracking
        self.momentum_days: Dict[str, MomentumDay] = {}
        self.reset_point_performance: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.day_performance_stats: Dict[str, Dict[str, Any]] = {}
        
        # Real-time metrics
        self.recent_rewards: Deque[float] = deque(maxlen=50)
        self.recent_episode_lengths: Deque[int] = deque(maxlen=50)
        self.recent_losses: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=50))
        
        # Dashboard metadata
        self.last_update_time = time.time()
        self.update_count = 0
    
    def update_training_state(self, data: Dict[str, Any]):
        """Update training state with new data"""
        try:
            # Update basic training metrics
            for key, value in data.items():
                if hasattr(self.training_state, key):
                    setattr(self.training_state, key, value)
            
            # Handle momentum-specific updates
            if 'current_momentum_day' in data:
                momentum_day_data = data['current_momentum_day']
                if momentum_day_data:
                    self.training_state.current_momentum_day = MomentumDay(**momentum_day_data)
            
            # Update recent metrics
            if 'mean_reward' in data:
                self.recent_rewards.append(data['mean_reward'])
            
            # Store in history
            timestamp_data = {**data, 'timestamp': time.time()}
            self.training_history.append(timestamp_data)
            
            self.last_update_time = time.time()
            self.update_count += 1
            
        except Exception as e:
            print(f"Error updating training state: {e}")
    
    def update_episode_data(self, episode_data: Dict[str, Any]):
        """Update episode-specific data"""
        try:
            # Add timestamp
            episode_data['timestamp'] = time.time()
            self.episode_history.append(episode_data)
            
            # Update recent metrics
            if 'reward' in episode_data:
                self.recent_rewards.append(episode_data['reward'])
            
            if 'length' in episode_data:
                self.recent_episode_lengths.append(episode_data['length'])
                
        except Exception as e:
            print(f"Error updating episode data: {e}")
    
    def update_momentum_day(self, momentum_day: MomentumDay):
        """Update momentum day information"""
        try:
            date_key = momentum_day.date.strftime('%Y-%m-%d')
            self.momentum_days[date_key] = momentum_day
            self.training_state.current_momentum_day = momentum_day
            
        except Exception as e:
            print(f"Error updating momentum day: {e}")
    
    def update_curriculum_progress(self, progress: float, strategy: str = None):
        """Update curriculum learning progress"""
        try:
            self.training_state.curriculum_progress = progress
            if strategy:
                self.training_state.curriculum_strategy = strategy
            
            self.curriculum_history.append(progress)
            
        except Exception as e:
            print(f"Error updating curriculum progress: {e}")
    
    def update_reward_components(self, components: Dict[str, float]):
        """Update reward components"""
        try:
            total_reward = sum(components.values())
            self.reward_components.total_reward = total_reward
            
            for name, value in components.items():
                self.reward_components.update_component(name, value)
                
        except Exception as e:
            print(f"Error updating reward components: {e}")
    
    def update_reset_point_performance(self, reset_point_idx: int, performance_data: Dict[str, float]):
        """Update performance data for a specific reset point"""
        try:
            self.reset_point_performance[reset_point_idx].update(performance_data)
            
        except Exception as e:
            print(f"Error updating reset point performance: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current dashboard state as dictionary"""
        try:
            state = {
                # Training state
                'mode': self.training_state.mode,
                'stage': self.training_state.stage,
                'updates': self.training_state.updates,
                'global_steps': self.training_state.global_steps,
                'total_episodes': self.training_state.total_episodes,
                'overall_progress': self.training_state.overall_progress,
                'stage_progress': self.training_state.stage_progress,
                'stage_status': self.training_state.stage_status,
                
                # Performance
                'steps_per_second': self.training_state.steps_per_second,
                'episodes_per_hour': self.training_state.episodes_per_hour,
                'time_per_update': self.training_state.time_per_update,
                'time_per_episode': self.training_state.time_per_episode,
                'eta_hours': self.training_state.eta_hours,
                
                # PPO metrics
                'lr': self.training_state.lr,
                'mean_reward': self.training_state.mean_reward,
                'reward_std': self.training_state.reward_std,
                'policy_loss': self.training_state.policy_loss,
                'value_loss': self.training_state.value_loss,
                'entropy': self.training_state.entropy,
                'clip_fraction': self.training_state.clip_fraction,
                'approx_kl': self.training_state.approx_kl,
                'explained_variance': self.training_state.explained_variance,
                
                # Momentum-specific
                'current_momentum_day': self._serialize_momentum_day(self.training_state.current_momentum_day),
                'curriculum_progress': self.training_state.curriculum_progress,
                'curriculum_strategy': self.training_state.curriculum_strategy,
                'used_reset_point_indices': self.training_state.used_reset_point_indices,
                'momentum_day_switches': self.training_state.momentum_day_switches,
                
                # Recent metrics
                'recent_mean_reward': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0,
                'recent_reward_std': np.std(list(self.recent_rewards)) if len(self.recent_rewards) > 1 else 0.0,
                'recent_episode_length': np.mean(list(self.recent_episode_lengths)) if self.recent_episode_lengths else 0.0,
                
                # Metadata
                'last_update_time': self.last_update_time,
                'update_count': self.update_count,
                'data_points': len(self.training_history),
            }
            
            return state
            
        except Exception as e:
            print(f"Error getting current state: {e}")
            return {}
    
    def _serialize_momentum_day(self, momentum_day: Optional[MomentumDay]) -> Optional[Dict[str, Any]]:
        """Serialize momentum day for JSON compatibility"""
        if not momentum_day:
            return None
        
        try:
            return {
                'date': momentum_day.date.strftime('%Y-%m-%d'),
                'symbol': momentum_day.symbol,
                'activity_score': momentum_day.activity_score,
                'max_intraday_move': momentum_day.max_intraday_move,
                'volume_multiplier': momentum_day.volume_multiplier,
                'reset_points': momentum_day.reset_points,
                'is_front_side': momentum_day.is_front_side,
                'is_back_side': momentum_day.is_back_side,
                'halt_count': momentum_day.halt_count,
                'episodes_trained': momentum_day.episodes_trained,
                'best_reward': momentum_day.best_reward,
                'avg_reward': momentum_day.avg_reward,
                'difficulty_level': momentum_day.difficulty_level,
                'momentum_type': momentum_day.momentum_type,
            }
        except Exception as e:
            print(f"Error serializing momentum day: {e}")
            return None
    
    def get_training_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get training history"""
        try:
            history = list(self.training_history)
            if last_n:
                history = history[-last_n:]
            return history
        except Exception as e:
            print(f"Error getting training history: {e}")
            return []
    
    def get_episode_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get episode history"""
        try:
            history = list(self.episode_history)
            if last_n:
                history = history[-last_n:]
            return history
        except Exception as e:
            print(f"Error getting episode history: {e}")
            return []
    
    def get_curriculum_history(self) -> List[float]:
        """Get curriculum learning history"""
        try:
            return list(self.curriculum_history)
        except Exception as e:
            print(f"Error getting curriculum history: {e}")
            return []
    
    def get_momentum_days_summary(self) -> Dict[str, Any]:
        """Get summary of all momentum days"""
        try:
            if not self.momentum_days:
                return {}
            
            days = list(self.momentum_days.values())
            
            return {
                'total_days': len(days),
                'avg_activity_score': np.mean([d.activity_score for d in days]),
                'avg_max_move': np.mean([d.max_intraday_move for d in days]),
                'avg_volume_mult': np.mean([d.volume_multiplier for d in days]),
                'front_side_days': sum(1 for d in days if d.is_front_side),
                'back_side_days': sum(1 for d in days if d.is_back_side),
                'expert_days': sum(1 for d in days if d.difficulty_level == 'Expert'),
                'hard_days': sum(1 for d in days if d.difficulty_level == 'Hard'),
                'medium_days': sum(1 for d in days if d.difficulty_level == 'Medium'),
                'easy_days': sum(1 for d in days if d.difficulty_level == 'Easy'),
            }
        except Exception as e:
            print(f"Error getting momentum days summary: {e}")
            return {}
    
    def get_reward_components_summary(self) -> Dict[str, Any]:
        """Get reward components summary"""
        try:
            if not self.reward_components.components:
                return {}
            
            return {
                'total_reward': self.reward_components.total_reward,
                'components': dict(self.reward_components.components),
                'positive_components': dict(self.reward_components.positive_contributions),
                'negative_components': dict(self.reward_components.negative_contributions),
                'component_stats': {
                    name: self.reward_components.get_component_stats(name)
                    for name in self.reward_components.components.keys()
                }
            }
        except Exception as e:
            print(f"Error getting reward components summary: {e}")
            return {}
    
    def clear_history(self):
        """Clear all historical data"""
        try:
            self.training_history.clear()
            self.episode_history.clear()
            self.curriculum_history.clear()
            self.recent_rewards.clear()
            self.recent_episode_lengths.clear()
            self.recent_losses.clear()
            self.reset_point_performance.clear()
            self.day_performance_stats.clear()
            
            self.update_count = 0
            
        except Exception as e:
            print(f"Error clearing history: {e}")
    
    def update_step_data(self, step_data: Dict[str, Any]):
        """Update step-level data (placeholder for dashboard integration)"""
        # This method is called by dashboard integration but doesn't need to store step data
        # Step data is typically too granular for dashboard display
        pass
    
    def update_trade_data(self, trade_data: Dict[str, Any]):
        """Update trade execution data (placeholder for dashboard integration)"""
        # This method is called by dashboard integration but doesn't need to store trade data  
        # Trade data is typically handled through episode summaries
        pass