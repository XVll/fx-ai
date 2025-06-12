"""
PPO and training metrics callback for WandB integration.

Tracks episode-level metrics (rewards, lengths, performance) and
training-level metrics (losses, learning rates, gradients).
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None

from callbacks.core.base import BaseCallback

logger = logging.getLogger(__name__)


class PPOMetricsCallback(BaseCallback):
    """
    Specialized callback for PPO/training metrics with minimal local buffering.
    
    Tracks:
    - Episode metrics: rewards, lengths, win rates
    - Training metrics: policy/value losses, learning rates
    - Rolling calculations: moving averages, Sharpe ratios, streaks
    """
    
    def __init__(self, buffer_size: int = 1000, enabled: bool = True):
        """
        Initialize PPO metrics callback.
        
        Args:
            buffer_size: Size of local buffers for rolling calculations
            enabled: Whether callback is active
        """
        super().__init__(name="PPOMetrics", enabled=enabled)
        
        self.buffer_size = buffer_size
        
        # Episode-level buffers
        self.episode_rewards = deque(maxlen=buffer_size)
        self.episode_lengths = deque(maxlen=buffer_size)
        self.episode_pnls = deque(maxlen=buffer_size)
        self.episode_trades = deque(maxlen=buffer_size)
        
        # Training-level buffers
        self.policy_losses = deque(maxlen=200)  # Updates are less frequent
        self.value_losses = deque(maxlen=200)
        self.entropy_losses = deque(maxlen=200)
        self.learning_rates = deque(maxlen=200)
        self.kl_divergences = deque(maxlen=200)
        
        # Reward component buffers (for component analysis)
        self.pnl_components = deque(maxlen=buffer_size)
        self.holding_penalties = deque(maxlen=buffer_size)
        self.action_efficiencies = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.episodes_logged = 0
        self.updates_logged = 0
        
        if wandb is None:
            self.logger.warning("wandb not installed - metrics will not be logged")
        
        self.logger.info(f"🚀 PPO metrics callback initialized (buffer_size={buffer_size})")
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize training session metrics."""
        if wandb and wandb.run:
            # Log training configuration
            config_data = {}
            if 'config' in context:
                config_data.update(context['config'])
            
            # Add our callback info
            config_data.update({
                'callbacks/ppo_metrics_buffer_size': self.buffer_size,
                'callbacks/ppo_metrics_enabled': True
            })
            
            wandb.config.update(config_data)
            
            self.logger.info("🎯 PPO metrics tracking started")
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Collect and log episode-level metrics."""
        if not wandb or not wandb.run:
            return
        
        # Extract episode data
        episode_reward = context.get('total_reward', 0)
        episode_length = context.get('episode_length', 0)
        num_trades = context.get('num_trades', 0)
        
        # Add to local buffers
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_trades.append(num_trades)
        
        # Extract reward components if available
        reward_breakdown = context.get('reward_breakdown', {})
        pnl_component = reward_breakdown.get('pnl', 0)
        holding_penalty = reward_breakdown.get('holding_penalty', 0)
        action_efficiency = reward_breakdown.get('action_efficiency', 0)
        
        self.episode_pnls.append(pnl_component)
        self.pnl_components.append(pnl_component)
        self.holding_penalties.append(holding_penalty)
        self.action_efficiencies.append(action_efficiency)
        
        # Prepare metrics to log
        metrics = {
            # Raw episode metrics
            'episode/reward': episode_reward,
            'episode/length': episode_length,
            'episode/num_trades': num_trades,
            'episode/episode_number': self.episodes_seen,
            
            # Reward components
            'episode/pnl': pnl_component,
            'episode/holding_penalty': holding_penalty,
            'episode/action_efficiency': action_efficiency,
        }
        
        # Add rolling calculations if enough data
        self._add_rolling_metrics(metrics)
        
        # Add performance metrics
        self._add_performance_metrics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.episodes_logged += 1
        
        if self.episodes_logged % 50 == 0:
            self.logger.info(f"📊 Logged {self.episodes_logged} episodes to WandB")
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Collect and log training update metrics."""
        if not wandb or not wandb.run:
            return
        
        # Extract training metrics
        policy_loss = context.get('policy_loss', 0)
        value_loss = context.get('value_loss', 0)
        entropy_loss = context.get('entropy_loss', 0)
        total_loss = context.get('loss', 0)
        learning_rate = context.get('learning_rate', 0)
        kl_divergence = context.get('kl_divergence', 0)
        clip_fraction = context.get('clip_fraction', 0)
        grad_norm = context.get('grad_norm', 0)
        explained_variance = context.get('explained_variance', 0)
        
        # Add to local buffers
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)
        self.learning_rates.append(learning_rate)
        self.kl_divergences.append(kl_divergence)
        
        # Prepare training metrics
        metrics = {
            # Raw training metrics
            'training/policy_loss': policy_loss,
            'training/value_loss': value_loss,
            'training/entropy_loss': entropy_loss,
            'training/total_loss': total_loss,
            'training/learning_rate': learning_rate,
            'training/kl_divergence': kl_divergence,
            'training/clip_fraction': clip_fraction,
            'training/grad_norm': grad_norm,
            'training/explained_variance': explained_variance,
            'training/update_number': self.updates_seen,
        }
        
        # Add rolling training metrics
        self._add_rolling_training_metrics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.updates_logged += 1
    
    def _add_rolling_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling episode metrics to the metrics dict."""
        # 10-episode rolling metrics
        if len(self.episode_rewards) >= 10:
            recent_10_rewards = list(self.episode_rewards)[-10:]
            recent_10_pnls = list(self.episode_pnls)[-10:]
            
            metrics.update({
                'rolling_10/reward_mean': np.mean(recent_10_rewards),
                'rolling_10/reward_std': np.std(recent_10_rewards),
                'rolling_10/pnl_mean': np.mean(recent_10_pnls),
                'rolling_10/win_rate': np.mean([r > 0 for r in recent_10_rewards]),
                'rolling_10/avg_episode_length': np.mean(list(self.episode_lengths)[-10:])
            })
        
        # 50-episode rolling metrics
        if len(self.episode_rewards) >= 50:
            recent_50_rewards = list(self.episode_rewards)[-50:]
            recent_50_pnls = list(self.episode_pnls)[-50:]
            
            metrics.update({
                'rolling_50/reward_mean': np.mean(recent_50_rewards),
                'rolling_50/reward_std': np.std(recent_50_rewards),
                'rolling_50/pnl_mean': np.mean(recent_50_pnls),
                'rolling_50/pnl_sum': np.sum(recent_50_pnls),
                'rolling_50/win_rate': np.mean([r > 0 for r in recent_50_rewards]),
                'rolling_50/sharpe_ratio': self._calculate_sharpe(recent_50_rewards)
            })
        
        # 100-episode rolling metrics
        if len(self.episode_rewards) >= 100:
            recent_100_rewards = list(self.episode_rewards)[-100:]
            recent_100_pnls = list(self.episode_pnls)[-100:]
            
            metrics.update({
                'rolling_100/reward_mean': np.mean(recent_100_rewards),
                'rolling_100/reward_std': np.std(recent_100_rewards),
                'rolling_100/pnl_mean': np.mean(recent_100_pnls),
                'rolling_100/pnl_sum': np.sum(recent_100_pnls),
                'rolling_100/win_rate': np.mean([r > 0 for r in recent_100_rewards]),
                'rolling_100/sharpe_ratio': self._calculate_sharpe(recent_100_rewards),
                'rolling_100/max_drawdown': self._calculate_max_drawdown(recent_100_rewards)
            })
    
    def _add_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add advanced performance metrics."""
        if len(self.episode_rewards) >= 20:
            recent_rewards = list(self.episode_rewards)[-20:]
            
            # Streak calculation
            current_streak = self._calculate_current_streak()
            
            # Percentiles
            p25 = np.percentile(recent_rewards, 25)
            p75 = np.percentile(recent_rewards, 75)
            p95 = np.percentile(recent_rewards, 95)
            
            metrics.update({
                'performance/current_streak': current_streak,
                'performance/reward_p25': p25,
                'performance/reward_p75': p75,
                'performance/reward_p95': p95,
                'performance/volatility_20': np.std(recent_rewards)
            })
        
        # Reward component analysis
        if len(self.pnl_components) >= 50:
            recent_50_pnl = list(self.pnl_components)[-50:]
            recent_50_penalty = list(self.holding_penalties)[-50:]
            recent_50_efficiency = list(self.action_efficiencies)[-50:]
            
            total_reward_50 = np.sum(recent_50_pnl) + np.sum(recent_50_penalty) + np.sum(recent_50_efficiency)
            
            if total_reward_50 != 0:
                metrics.update({
                    'components/pnl_contribution_pct': (np.sum(recent_50_pnl) / total_reward_50) * 100,
                    'components/penalty_contribution_pct': (np.sum(recent_50_penalty) / total_reward_50) * 100,
                    'components/efficiency_contribution_pct': (np.sum(recent_50_efficiency) / total_reward_50) * 100
                })
    
    def _add_rolling_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling training metrics."""
        if len(self.policy_losses) >= 10:
            recent_10_policy = list(self.policy_losses)[-10:]
            recent_10_value = list(self.value_losses)[-10:]
            recent_10_entropy = list(self.entropy_losses)[-10:]
            
            metrics.update({
                'training/avg_policy_loss_10': np.mean(recent_10_policy),
                'training/avg_value_loss_10': np.mean(recent_10_value),
                'training/avg_entropy_loss_10': np.mean(recent_10_entropy),
                'training/policy_loss_std_10': np.std(recent_10_policy)
            })
        
        if len(self.learning_rates) >= 20:
            metrics.update({
                'training/avg_learning_rate_20': np.mean(list(self.learning_rates)[-20:])
            })
    
    def _calculate_sharpe(self, returns: list) -> float:
        """Calculate Sharpe ratio for given returns."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        return (mean_return / std_return) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: list) -> float:
        """Calculate maximum drawdown from returns."""
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        
        return float(np.min(drawdowns))
    
    def _calculate_current_streak(self) -> int:
        """Calculate current winning/losing streak."""
        if len(self.episode_rewards) < 2:
            return 0
        
        rewards = list(self.episode_rewards)
        last_positive = rewards[-1] > 0
        
        streak = 0
        for reward in reversed(rewards):
            if (reward > 0) == last_positive:
                streak += 1
            else:
                break
        
        return streak if last_positive else -streak
    
    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics."""
        return {
            'buffer_size': self.buffer_size,
            'episodes_logged': self.episodes_logged,
            'updates_logged': self.updates_logged,
            'episode_rewards_buffer_size': len(self.episode_rewards),
            'policy_losses_buffer_size': len(self.policy_losses),
            'current_episode_count': self.episodes_seen,
            'current_update_count': self.updates_seen
        }