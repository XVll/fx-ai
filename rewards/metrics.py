# rewards/metrics.py - Comprehensive metrics tracking for reward components

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import time


@dataclass
class ComponentMetrics:
    """Tracks detailed metrics for a single reward component"""
    name: str
    type: str
    
    # Basic statistics
    total_value: float = 0.0
    count_triggered: int = 0
    count_positive: int = 0
    count_negative: int = 0
    count_zero: int = 0
    
    # Value statistics
    min_value: float = float('inf')
    max_value: float = float('-inf')
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Timing
    last_triggered_step: Optional[int] = None
    trigger_intervals: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Correlations tracking
    when_action_buy: deque = field(default_factory=lambda: deque(maxlen=100))
    when_action_sell: deque = field(default_factory=lambda: deque(maxlen=100))
    when_action_hold: deque = field(default_factory=lambda: deque(maxlen=100))
    when_profitable: deque = field(default_factory=lambda: deque(maxlen=100))
    when_losing: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, value: float, step: int, action: str, is_profitable: bool):
        """Update metrics with new value"""
        self.total_value += value
        self.count_triggered += 1
        
        # Update counts
        if value > 0:
            self.count_positive += 1
        elif value < 0:
            self.count_negative += 1
        else:
            self.count_zero += 1
            
        # Update min/max
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        
        # Store value
        self.values.append(value)
        
        # Update timing
        if self.last_triggered_step is not None:
            interval = step - self.last_triggered_step
            self.trigger_intervals.append(interval)
        self.last_triggered_step = step
        
        # Update correlations
        if action.upper() == 'BUY':
            self.when_action_buy.append(value)
        elif action.upper() == 'SELL':
            self.when_action_sell.append(value)
        else:
            self.when_action_hold.append(value)
            
        if is_profitable:
            self.when_profitable.append(value)
        else:
            self.when_losing.append(value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if not self.values:
            return {
                'name': self.name,
                'type': self.type,
                'triggered': False
            }
            
        values_array = np.array(self.values)
        
        stats = {
            'name': self.name,
            'type': self.type,
            'triggered': True,
            
            # Basic stats
            'total_value': self.total_value,
            'count_triggered': self.count_triggered,
            'trigger_rate': self.count_triggered,  # Will be normalized later
            
            # Value distribution
            'mean_value': np.mean(values_array),
            'std_value': np.std(values_array),
            'min_value': self.min_value,
            'max_value': self.max_value,
            'median_value': np.median(values_array),
            'percentile_25': np.percentile(values_array, 25),
            'percentile_75': np.percentile(values_array, 75),
            
            # Sign distribution
            'positive_rate': self.count_positive / self.count_triggered if self.count_triggered > 0 else 0,
            'negative_rate': self.count_negative / self.count_triggered if self.count_triggered > 0 else 0,
            'zero_rate': self.count_zero / self.count_triggered if self.count_triggered > 0 else 0,
            
            # Magnitude analysis
            'mean_positive': np.mean([v for v in values_array if v > 0]) if self.count_positive > 0 else 0,
            'mean_negative': np.mean([v for v in values_array if v < 0]) if self.count_negative > 0 else 0,
            
            # Timing
            'mean_trigger_interval': np.mean(self.trigger_intervals) if self.trigger_intervals else None,
            
            # Action correlations
            'mean_when_buy': np.mean(self.when_action_buy) if self.when_action_buy else None,
            'mean_when_sell': np.mean(self.when_action_sell) if self.when_action_sell else None,
            'mean_when_hold': np.mean(self.when_action_hold) if self.when_action_hold else None,
            
            # Profitability correlations
            'mean_when_profitable': np.mean(self.when_profitable) if self.when_profitable else None,
            'mean_when_losing': np.mean(self.when_losing) if self.when_losing else None,
        }
        
        return stats


class RewardMetricsTracker:
    """Comprehensive metrics tracking for the entire reward system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.episode_metrics = []
        self.current_episode_steps = 0
        
        # Global tracking
        self.total_episodes = 0
        self.total_steps = 0
        
        # Episode-level tracking
        self.episode_total_reward = 0.0
        self.episode_component_totals = defaultdict(float)
        self.episode_action_counts = defaultdict(int)
        self.episode_trades = 0
        self.episode_profitable_trades = 0
        
    def register_component(self, name: str, component_type: str):
        """Register a reward component for tracking"""
        if name not in self.component_metrics:
            self.component_metrics[name] = ComponentMetrics(name=name, type=component_type)
            
    def update_component(self, name: str, value: float, diagnostics: Dict[str, Any], 
                        step: int, action: str, is_profitable: bool):
        """Update metrics for a component"""
        if name not in self.component_metrics:
            self.logger.warning(f"Component {name} not registered, skipping metrics update")
            return
            
        self.component_metrics[name].update(value, step, action, is_profitable)
        self.episode_component_totals[name] += value
        
    def update_step(self, total_reward: float, action: str, has_trade: bool, 
                   is_profitable_trade: Optional[bool] = None):
        """Update step-level metrics"""
        self.current_episode_steps += 1
        self.total_steps += 1
        self.episode_total_reward += total_reward
        self.episode_action_counts[action] += 1
        
        if has_trade:
            self.episode_trades += 1
            if is_profitable_trade:
                self.episode_profitable_trades += 1
                
    def end_episode(self, final_portfolio_metrics: Dict[str, Any]):
        """Finalize episode metrics"""
        self.total_episodes += 1
        
        # Calculate episode summary
        episode_summary = {
            'episode': self.total_episodes,
            'steps': self.current_episode_steps,
            'total_reward': self.episode_total_reward,
            'mean_reward_per_step': self.episode_total_reward / max(1, self.current_episode_steps),
            
            # Component breakdown
            'component_totals': dict(self.episode_component_totals),
            'component_means': {
                name: total / max(1, self.current_episode_steps) 
                for name, total in self.episode_component_totals.items()
            },
            
            # Trading metrics
            'total_trades': self.episode_trades,
            'profitable_trades': self.episode_profitable_trades,
            'win_rate': self.episode_profitable_trades / max(1, self.episode_trades),
            
            # Action distribution
            'action_distribution': dict(self.episode_action_counts),
            
            # Portfolio metrics
            'final_equity': final_portfolio_metrics.get('total_equity', 0),
            'total_pnl': final_portfolio_metrics.get('realized_pnl_session', 0),
            'sharpe_ratio': final_portfolio_metrics.get('sharpe_ratio', 0),
        }
        
        # Get component statistics
        component_stats = {}
        for name, metrics in self.component_metrics.items():
            stats = metrics.get_statistics()
            if self.current_episode_steps > 0 and stats.get('triggered', False):
                stats['trigger_rate'] = stats['count_triggered'] / self.current_episode_steps
            else:
                stats['trigger_rate'] = 0.0
            component_stats[name] = stats
            
        episode_summary['component_statistics'] = component_stats
        
        # Identify dominant components
        dominant_positive = max(self.episode_component_totals.items(), 
                              key=lambda x: x[1], default=(None, 0))
        dominant_negative = min(self.episode_component_totals.items(), 
                              key=lambda x: x[1], default=(None, 0))
        
        episode_summary['dominant_positive_component'] = {
            'name': dominant_positive[0],
            'total': dominant_positive[1]
        } if dominant_positive[0] else None
        
        episode_summary['dominant_negative_component'] = {
            'name': dominant_negative[0],
            'total': dominant_negative[1]
        } if dominant_negative[0] else None
        
        self.episode_metrics.append(episode_summary)
        
        # Reset episode tracking
        self.current_episode_steps = 0
        self.episode_total_reward = 0.0
        self.episode_component_totals.clear()
        self.episode_action_counts.clear()
        self.episode_trades = 0
        self.episode_profitable_trades = 0
        
        return episode_summary
    
    def get_component_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of component behavior"""
        if not self.episode_metrics:
            return {}
            
        analysis = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
        }
        
        # Aggregate component statistics across episodes
        component_aggregates = defaultdict(lambda: {
            'total_value': 0.0,
            'mean_values': [],
            'trigger_rates': [],
            'positive_rates': [],
            'negative_rates': []
        })
        
        for episode in self.episode_metrics:
            for comp_name, comp_stats in episode['component_statistics'].items():
                if comp_stats['triggered']:
                    agg = component_aggregates[comp_name]
                    agg['total_value'] += episode['component_totals'].get(comp_name, 0)
                    agg['mean_values'].append(comp_stats['mean_value'])
                    agg['trigger_rates'].append(comp_stats['trigger_rate'])
                    agg['positive_rates'].append(comp_stats['positive_rate'])
                    agg['negative_rates'].append(comp_stats['negative_rate'])
                    
        # Calculate final statistics
        component_analysis = {}
        for comp_name, agg in component_aggregates.items():
            if agg['mean_values']:
                component_analysis[comp_name] = {
                    'global_total_value': agg['total_value'],
                    'global_mean_value': np.mean(agg['mean_values']),
                    'global_std_value': np.std(agg['mean_values']),
                    'mean_trigger_rate': np.mean(agg['trigger_rates']),
                    'mean_positive_rate': np.mean(agg['positive_rates']),
                    'mean_negative_rate': np.mean(agg['negative_rates']),
                    'consistency': 1.0 - np.std(agg['mean_values']) / (np.mean(agg['mean_values']) + 1e-8)
                }
                
        analysis['components'] = component_analysis
        
        # Identify problematic components
        problems = []
        for comp_name, comp_analysis in component_analysis.items():
            # Check for dominant components
            if abs(comp_analysis['global_total_value']) > abs(analysis.get('total_reward', 0) * 0.5):
                problems.append({
                    'component': comp_name,
                    'issue': 'dominant',
                    'severity': 'high',
                    'description': f'Component contributes >50% of total reward'
                })
                
            # Check for always-negative components
            if comp_analysis['mean_positive_rate'] < 0.1:
                problems.append({
                    'component': comp_name,
                    'issue': 'always_negative',
                    'severity': 'medium',
                    'description': f'Component is negative >90% of the time'
                })
                
            # Check for rarely-triggered components
            if comp_analysis['mean_trigger_rate'] < 0.01:
                problems.append({
                    'component': comp_name,
                    'issue': 'rarely_triggered',
                    'severity': 'low',
                    'description': f'Component triggers <1% of the time'
                })
                
        analysis['potential_issues'] = problems
        
        return analysis
    
    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Analyze correlations between components and behaviors"""
        if not self.episode_metrics:
            return {}
            
        # Extract time series data
        episode_rewards = [ep['total_reward'] for ep in self.episode_metrics]
        episode_trades = [ep['total_trades'] for ep in self.episode_metrics]
        episode_win_rates = [ep['win_rate'] for ep in self.episode_metrics]
        
        correlations = {
            'reward_vs_trades': np.corrcoef(episode_rewards, episode_trades)[0, 1] if len(episode_rewards) > 1 else None,
            'reward_vs_win_rate': np.corrcoef(episode_rewards, episode_win_rates)[0, 1] if len(episode_rewards) > 1 else None,
        }
        
        # Component correlations
        component_correlations = {}
        for comp_name in self.component_metrics.keys():
            comp_totals = [ep['component_totals'].get(comp_name, 0) for ep in self.episode_metrics]
            if len(comp_totals) > 1 and any(comp_totals):
                component_correlations[comp_name] = {
                    'vs_total_reward': np.corrcoef(comp_totals, episode_rewards)[0, 1],
                    'vs_trades': np.corrcoef(comp_totals, episode_trades)[0, 1],
                    'vs_win_rate': np.corrcoef(comp_totals, episode_win_rates)[0, 1],
                }
                
        correlations['component_correlations'] = component_correlations
        
        return correlations