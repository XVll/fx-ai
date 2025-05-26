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
        self.episode_component_totals = defaultdict(float)
        
        # Correlation tracking
        self.correlation_window = []  # Store recent (reward, outcome) pairs
        
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
        
        # Reward sparsity metrics
        self.register_metric("reward_sparsity", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of non-zero rewards",
            unit="%",
            frequency="episode"
        ))
        
        # Positive vs negative reward metrics
        self.register_metric("positive_reward_magnitude", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Average magnitude of positive rewards",
            unit="reward",
            frequency="episode"
        ))
        
        self.register_metric("negative_reward_magnitude", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description="Average magnitude of negative rewards",
            unit="reward",
            frequency="episode"
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
        
        self.register_metric(f"component.{component_name}.contribution_pct", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.PERCENTAGE,
            description=f"Percentage contribution of {component_name} to total reward",
            unit="%",
            frequency="episode"
        ))
        
        self.register_metric(f"component.{component_name}.cumulative", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description=f"Cumulative value of {component_name} in episode",
            unit="reward",
            frequency="episode"
        ))
        
        self.register_metric(f"component.{component_name}.volatility", MetricMetadata(
            category=MetricCategory.TRAINING,
            metric_type=MetricType.GAUGE,
            description=f"Standard deviation of {component_name} values",
            unit="reward",
            frequency="episode"
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
                
                # Reward sparsity
                non_zero_rewards = sum(1 for r in self.step_rewards if r != 0)
                sparsity = (non_zero_rewards / len(self.step_rewards)) * 100
                metrics[f"{self.category.value}.{self.name}.reward_sparsity"] = MetricValue(sparsity)
                
                # Positive vs negative magnitudes
                positive_rewards = [r for r in self.step_rewards if r > 0]
                negative_rewards = [r for r in self.step_rewards if r < 0]
                
                if positive_rewards:
                    metrics[f"{self.category.value}.{self.name}.positive_reward_magnitude"] = MetricValue(np.mean(positive_rewards))
                if negative_rewards:
                    metrics[f"{self.category.value}.{self.name}.negative_reward_magnitude"] = MetricValue(abs(np.mean(negative_rewards)))
                
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
                
                # Contribution percentage
                if self.episode_total_reward != 0:
                    contribution_pct = (self.episode_component_totals[comp_name] / abs(self.episode_total_reward)) * 100
                    metrics[f"{self.category.value}.{self.name}.component.{comp_name}.contribution_pct"] = MetricValue(contribution_pct)
                
                # Cumulative value
                metrics[f"{self.category.value}.{self.name}.component.{comp_name}.cumulative"] = MetricValue(self.episode_component_totals[comp_name])
                
                # Volatility
                if len(comp_data['values']) > 1:
                    volatility = np.std(comp_data['values'])
                    metrics[f"{self.category.value}.{self.name}.component.{comp_name}.volatility"] = MetricValue(volatility)
                
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
            
            # Track episode totals
            self.episode_component_totals[comp_name] += comp_value
                
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
        self.episode_component_totals.clear()
        
        # Keep component data but reset counts
        for comp_data in self.component_data.values():
            comp_data['values'].clear()
            comp_data['count'] = 0
            comp_data['total'] = 0.0
            comp_data['last_value'] = 0.0
            
    def calculate_component_correlations(self, outcomes: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between reward components and trading outcomes"""
        correlations = {}
        
        for comp_name, comp_data in self.component_data.items():
            if len(comp_data['values']) < 10:  # Need sufficient data
                continue
                
            correlations[comp_name] = {}
            comp_values = np.array(comp_data['values'])
            
            for outcome_name, outcome_values in outcomes.items():
                if len(outcome_values) == len(comp_values):
                    # Calculate Pearson correlation
                    if np.std(comp_values) > 0 and np.std(outcome_values) > 0:
                        corr = np.corrcoef(comp_values, outcome_values)[0, 1]
                        correlations[comp_name][outcome_name] = corr
                        
        return correlations
    
    def _get_metadata(self, metric_name: str) -> MetricMetadata:
        """Get metadata for a metric by name"""
        full_name = f"{self.category.value}.{self.name}.{metric_name}"
        return self._metrics.get(full_name)