# metrics/collectors/reward_metrics.py - Reward system metrics collector

import logging
from typing import Dict, Optional, Any, List
from collections import defaultdict
import numpy as np

from ..core import MetricCollector, MetricValue, MetricCategory, MetricType, MetricMetadata


class RewardMetricsCollector(MetricCollector):
    """Collector for reward system metrics"""
    
    def __init__(self):
        super().__init__("reward", MetricCategory.TRAINING)
        self.logger = logging.getLogger(__name__)
        
        # Component tracking
        self.component_data = defaultdict(lambda: {
            'values': [],
            'count': 0,
            'total': 0.0,
            'last_value': 0.0
        })
        
        # Episode tracking
        self.episode_total_reward = 0.0
        self.step_rewards = []
        
        # Register base metrics
        self._register_metrics()
        
    def _register_metrics(self):
        """Register reward metrics"""
        
        # Total reward metrics
        self.register_metric("total_reward", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Total reward for current step",
            unit="reward",
            frequency="step"
        ))
        
        self.register_metric("episode_reward", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Cumulative reward for episode",
            unit="reward",
            frequency="step"
        ))
        
        self.register_metric("mean_step_reward", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Mean reward per step in episode",
            unit="reward",
            frequency="step"
        ))
        
    def register_component(self, component_name: str, component_type: str):
        """Register a reward component for tracking"""
        # Register component-specific metrics
        self.register_metric(f"component.{component_name}.value", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description=f"Current value of {component_name} component",
            unit="reward",
            frequency="step"
        ))
        
        self.register_metric(f"component.{component_name}.mean", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description=f"Mean value of {component_name} component",
            unit="reward",
            frequency="step"
        ))
        
        self.register_metric(f"component.{component_name}.trigger_rate", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.PERCENTAGE,
            description=f"Trigger rate of {component_name} component",
            unit="%",
            frequency="step"
        ))
        
    def collect(self) -> Dict[str, MetricValue]:
        """Collect reward metrics"""
        metrics = {}
        
        try:
            # Episode metrics
            metrics[f"{self.category.value}.{self.name}.episode_reward"] = MetricValue(self.episode_total_reward)
            
            if self.step_rewards:
                mean_reward = np.mean(self.step_rewards)
                metrics[f"{self.category.value}.{self.name}.mean_step_reward"] = MetricValue(mean_reward)
                
            # Component metrics
            for comp_name, comp_data in self.component_data.items():
                # Current value
                metrics[f"{self.category.value}.{self.name}.component.{comp_name}.value"] = MetricValue(comp_data['last_value'])
                
                # Mean value
                if comp_data['values']:
                    mean_val = np.mean(comp_data['values'][-100:])  # Last 100 values
                    metrics[f"{self.category.value}.{self.name}.component.{comp_name}.mean"] = MetricValue(mean_val)
                    
                # Trigger rate
                total_steps = len(self.step_rewards) if self.step_rewards else 1
                trigger_rate = (comp_data['count'] / total_steps) * 100
                metrics[f"{self.category.value}.{self.name}.component.{comp_name}.trigger_rate"] = MetricValue(trigger_rate)
                
        except Exception as e:
            self.logger.debug(f"Error collecting reward metrics: {e}")
            
        return metrics
    
    def update_reward(self, total_reward: float, component_rewards: Dict[str, float]):
        """Update reward data"""
        # Update total reward
        self.episode_total_reward += total_reward
        self.step_rewards.append(total_reward)
        
        # Store last total reward
        self._last_total_reward = total_reward
        
        # Update component data
        for comp_name, comp_value in component_rewards.items():
            comp_data = self.component_data[comp_name]
            comp_data['values'].append(comp_value)
            comp_data['last_value'] = comp_value
            comp_data['total'] += comp_value
            if comp_value != 0:
                comp_data['count'] += 1
                
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed component statistics"""
        stats = {}
        
        for comp_name, comp_data in self.component_data.items():
            if comp_data['values']:
                values = np.array(comp_data['values'])
                stats[comp_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': comp_data['total'],
                    'count': comp_data['count'],
                    'trigger_rate': comp_data['count'] / max(1, len(self.step_rewards))
                }
                
        return stats
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_total_reward = 0.0
        self.step_rewards.clear()
        
        # Keep component data but reset counts
        for comp_data in self.component_data.values():
            comp_data['values'].clear()
            comp_data['count'] = 0
            comp_data['total'] = 0.0
            comp_data['last_value'] = 0.0
            
    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)