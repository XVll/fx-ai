"""
Simple evaluation metrics callback for WandB integration.

Logs basic evaluation results without complex dependencies.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None

from callbacks.core.base import BaseCallback

logger = logging.getLogger(__name__)


class EvaluationMetricsCallback(BaseCallback):
    """Simple callback to log evaluation metrics to WandB."""
    
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.EvaluationMetrics")
        
        if self.enabled:
            self.logger.info("📊 Evaluation metrics callback initialized")
    
    def on_evaluation_end(self, context: Dict[str, Any]) -> None:
        """Log evaluation results to WandB."""
        if not self.enabled or not wandb or not wandb.run:
            return
        
        try:
            # Extract basic evaluation results
            evaluation_result = context.get('evaluation_result', {})
            
            # Basic metrics - only log what's available
            metrics = {}
            
            # Mean reward (most important metric)
            if 'mean_reward' in evaluation_result:
                metrics['eval/mean_reward'] = evaluation_result['mean_reward']
            
            # Reward statistics
            if 'reward_std' in evaluation_result:
                metrics['eval/reward_std'] = evaluation_result['reward_std']
            if 'min_reward' in evaluation_result:
                metrics['eval/min_reward'] = evaluation_result['min_reward']
            if 'max_reward' in evaluation_result:
                metrics['eval/max_reward'] = evaluation_result['max_reward']
            
            # Episode statistics
            if 'mean_length' in evaluation_result:
                metrics['eval/mean_episode_length'] = evaluation_result['mean_length']
            if 'total_episodes' in evaluation_result:
                metrics['eval/total_episodes'] = evaluation_result['total_episodes']
            
            # Training context
            training_state = context.get('training_state', {})
            if hasattr(training_state, 'global_episodes'):
                metrics['eval/training_episodes'] = training_state.global_episodes
            if hasattr(training_state, 'global_updates'):
                metrics['eval/training_updates'] = training_state.global_updates
            
            # Log to WandB if we have any metrics
            if metrics:
                wandb.log(metrics)
                self.logger.debug(f"Logged {len(metrics)} evaluation metrics to WandB")
            else:
                self.logger.warning("No evaluation metrics found to log")
                
        except Exception as e:
            self.logger.warning(f"Failed to log evaluation metrics: {e}")
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Log basic episode metrics."""
        if not self.enabled or not wandb or not wandb.run:
            return
        
        try:
            # Simple episode metrics
            metrics = {}
            
            # Basic episode data
            if 'total_reward' in context:
                metrics['episode/reward'] = context['total_reward']
            if 'episode_length' in context:
                metrics['episode/length'] = context['episode_length']
            
            # Training progress
            if 'episode_number' in context:
                metrics['episode/number'] = context['episode_number']
            if 'update_number' in context:
                metrics['episode/update_number'] = context['update_number']
            
            # Log if we have data
            if metrics:
                wandb.log(metrics)
                
        except Exception as e:
            self.logger.warning(f"Failed to log episode metrics: {e}")