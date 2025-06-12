"""
Session metrics callback for WandB integration.

Tracks overall training session metrics including runtime, totals,
performance summaries, and system resource usage.
"""

import logging
import time
import psutil
import numpy as np
from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import wandb
except ImportError:
    wandb = None

from callbacks.core.base import BaseCallback

logger = logging.getLogger(__name__)


class SessionMetricsCallback(BaseCallback):
    """
    Specialized callback for session-level metrics and performance tracking.
    
    Tracks:
    - Session timing: runtime, intervals, milestones
    - Training totals: episodes, steps, updates
    - Performance summaries: rates, throughput
    - System resources: CPU, memory, GPU usage
    - Progress tracking: completion rates, estimates
    """
    
    def __init__(self, log_frequency: int = 50, track_system_resources: bool = True, enabled: bool = True):
        """
        Initialize session metrics callback.
        
        Args:
            log_frequency: How often to log session metrics (every N episodes)
            track_system_resources: Whether to track CPU/memory/GPU usage
            enabled: Whether callback is active
        """
        super().__init__(name="SessionMetrics", enabled=enabled)
        
        self.log_frequency = log_frequency
        self.track_system_resources = track_system_resources
        
        # Session timing
        self.session_start_time = time.time()
        self.last_log_time = self.session_start_time
        
        # Training totals
        self.total_episodes = 0
        self.total_steps = 0
        self.total_updates = 0
        self.total_trades = 0
        self.total_volume = 0
        
        # Performance tracking
        self.episode_times = deque(maxlen=100)  # Track episode durations
        self.step_times = deque(maxlen=1000)    # Track step durations
        self.update_times = deque(maxlen=100)   # Track update durations
        
        # Throughput tracking
        self.episodes_per_hour = deque(maxlen=24)  # Hourly throughput
        self.steps_per_minute = deque(maxlen=60)   # Minute throughput
        
        # Resource tracking
        self.cpu_usage = deque(maxlen=60) if track_system_resources else None
        self.memory_usage = deque(maxlen=60) if track_system_resources else None
        self.gpu_usage = deque(maxlen=60) if track_system_resources else None
        
        # Milestone tracking
        self.milestones = {
            'episodes': [100, 500, 1000, 5000, 10000],
            'steps': [10000, 50000, 100000, 500000, 1000000],
            'hours': [1, 6, 12, 24, 48, 168]  # 1h, 6h, 12h, 1d, 2d, 1w
        }
        self.milestones_reached = set()
        
        # Training targets (if known)
        self.target_episodes = None
        self.target_runtime_hours = None
        
        # Performance tracking
        self.session_logs_count = 0
        
        if wandb is None:
            self.logger.warning("wandb not installed - session metrics will not be logged")
        
        self.logger.info(f"📊 Session metrics callback initialized (log_freq={log_frequency}, resources={track_system_resources})")
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize session tracking."""
        if wandb and wandb.run:
            # Extract training targets if available
            config = context.get('config', {})
            self.target_episodes = config.get('total_episodes')
            self.target_runtime_hours = config.get('max_runtime_hours')
            
            # Log session start
            session_info = {
                'session/start_time': datetime.now().isoformat(),
                'session/target_episodes': self.target_episodes or 0,
                'session/target_runtime_hours': self.target_runtime_hours or 0,
                'session/log_frequency': self.log_frequency,
                'session/track_resources': self.track_system_resources
            }
            
            wandb.log(session_info)
            self.logger.info(f"📊 Session tracking started - Target: {self.target_episodes} episodes")
    
    def on_episode_start(self, context: Dict[str, Any]) -> None:
        """Track episode start timing."""
        self.episode_start_time = time.time()
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Track episode completion and log session metrics."""
        if not wandb or not wandb.run:
            return
        
        # Calculate episode duration
        episode_duration = time.time() - getattr(self, 'episode_start_time', time.time())
        self.episode_times.append(episode_duration)
        
        # Update totals
        self.total_episodes += 1
        episode_steps = context.get('episode_length', 0)
        episode_trades = context.get('num_trades', 0)
        episode_volume = context.get('episode_volume', 0)
        
        self.total_steps += episode_steps
        self.total_trades += episode_trades
        self.total_volume += episode_volume
        
        # Log session metrics at specified frequency
        if self.total_episodes % self.log_frequency == 0:
            self._log_session_metrics()
        
        # Check for milestones
        self._check_milestones()
    
    def on_step_end(self, context: Dict[str, Any]) -> None:
        """Track step timing (sampled to avoid overhead)."""
        # Only track every 10th step to reduce overhead
        if self.steps_seen % 10 == 0:
            step_time = time.time()
            if hasattr(self, 'last_step_time'):
                step_duration = step_time - self.last_step_time
                self.step_times.append(step_duration * 10)  # Approximate for 10 steps
            self.last_step_time = step_time
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Track update timing and counts."""
        update_time = time.time()
        if hasattr(self, 'last_update_time'):
            update_duration = update_time - self.last_update_time
            self.update_times.append(update_duration)
        self.last_update_time = update_time
        
        self.total_updates += 1
    
    def _log_session_metrics(self) -> None:
        """Log comprehensive session metrics to WandB."""
        current_time = time.time()
        elapsed_time = current_time - self.session_start_time
        elapsed_hours = elapsed_time / 3600
        elapsed_minutes = elapsed_time / 60
        
        # Calculate rates and throughput
        episodes_per_hour = self.total_episodes / elapsed_hours if elapsed_hours > 0 else 0
        steps_per_minute = self.total_steps / elapsed_minutes if elapsed_minutes > 0 else 0
        updates_per_hour = self.total_updates / elapsed_hours if elapsed_hours > 0 else 0
        
        # Prepare base session metrics
        metrics = {
            # Session timing
            'session/elapsed_hours': elapsed_hours,
            'session/elapsed_minutes': elapsed_minutes,
            'session/elapsed_seconds': elapsed_time,
            
            # Training totals
            'session/total_episodes': self.total_episodes,
            'session/total_steps': self.total_steps,
            'session/total_updates': self.total_updates,
            'session/total_trades': self.total_trades,
            'session/total_volume': self.total_volume,
            
            # Throughput metrics
            'session/episodes_per_hour': episodes_per_hour,
            'session/steps_per_minute': steps_per_minute,
            'session/updates_per_hour': updates_per_hour,
            'session/trades_per_hour': self.total_trades / elapsed_hours if elapsed_hours > 0 else 0,
        }
        
        # Add performance metrics
        self._add_performance_metrics(metrics)
        
        # Add progress tracking
        self._add_progress_metrics(metrics, elapsed_hours)
        
        # Add system resource metrics
        if self.track_system_resources:
            self._add_resource_metrics(metrics)
        
        # Add timing statistics
        self._add_timing_statistics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.session_logs_count += 1
        self.last_log_time = current_time
        
        # Log summary message
        self.logger.info(
            f"📊 Session: {self.total_episodes} episodes, "
            f"{elapsed_hours:.1f}h runtime, "
            f"{episodes_per_hour:.1f} ep/h, "
            f"{steps_per_minute:.0f} steps/min"
        )
    
    def _add_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add performance and efficiency metrics."""
        # Episode timing statistics
        if len(self.episode_times) >= 10:
            recent_episode_times = list(self.episode_times)[-10:]
            metrics.update({
                'performance/avg_episode_duration': np.mean(recent_episode_times),
                'performance/episode_duration_std': np.std(recent_episode_times),
                'performance/min_episode_duration': np.min(recent_episode_times),
                'performance/max_episode_duration': np.max(recent_episode_times)
            })
        
        # Step timing statistics
        if len(self.step_times) >= 50:
            recent_step_times = list(self.step_times)[-50:]
            metrics.update({
                'performance/avg_step_duration': np.mean(recent_step_times),
                'performance/step_duration_std': np.std(recent_step_times),
                'performance/steps_per_second': 1 / np.mean(recent_step_times) if np.mean(recent_step_times) > 0 else 0
            })
        
        # Update timing statistics
        if len(self.update_times) >= 10:
            recent_update_times = list(self.update_times)[-10:]
            metrics.update({
                'performance/avg_update_duration': np.mean(recent_update_times),
                'performance/update_duration_std': np.std(recent_update_times),
                'performance/updates_per_second': 1 / np.mean(recent_update_times) if np.mean(recent_update_times) > 0 else 0
            })
    
    def _add_progress_metrics(self, metrics: Dict[str, Any], elapsed_hours: float) -> None:
        """Add progress tracking and ETA estimates."""
        # Progress percentages
        if self.target_episodes:
            episode_progress = (self.total_episodes / self.target_episodes) * 100
            metrics['progress/episode_completion_pct'] = episode_progress
            
            # ETA calculation
            if self.total_episodes > 0:
                avg_episodes_per_hour = self.total_episodes / elapsed_hours
                remaining_episodes = self.target_episodes - self.total_episodes
                eta_hours = remaining_episodes / avg_episodes_per_hour if avg_episodes_per_hour > 0 else 0
                metrics['progress/eta_hours'] = eta_hours
                metrics['progress/eta_days'] = eta_hours / 24
        
        if self.target_runtime_hours:
            runtime_progress = (elapsed_hours / self.target_runtime_hours) * 100
            metrics['progress/runtime_completion_pct'] = runtime_progress
            metrics['progress/remaining_runtime_hours'] = max(0, self.target_runtime_hours - elapsed_hours)
        
        # Recent throughput trends
        current_time = time.time()
        time_since_last_log = current_time - self.last_log_time
        
        if time_since_last_log > 0:
            recent_episodes_per_hour = (self.log_frequency / (time_since_last_log / 3600))
            self.episodes_per_hour.append(recent_episodes_per_hour)
            
            if len(self.episodes_per_hour) >= 5:
                metrics['progress/recent_episodes_per_hour'] = np.mean(list(self.episodes_per_hour)[-5:])
                metrics['progress/throughput_trend'] = np.mean(list(self.episodes_per_hour)[-3:]) - np.mean(list(self.episodes_per_hour)[-5:-2])
    
    def _add_resource_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add system resource usage metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)
            metrics['resources/cpu_usage_pct'] = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024**3)
            self.memory_usage.append(memory_percent)
            metrics.update({
                'resources/memory_usage_pct': memory_percent,
                'resources/memory_used_gb': memory_gb,
                'resources/memory_available_gb': memory.available / (1024**3)
            })
            
            # Disk usage for current directory
            disk = psutil.disk_usage('.')
            metrics.update({
                'resources/disk_usage_pct': (disk.used / disk.total) * 100,
                'resources/disk_free_gb': disk.free / (1024**3)
            })
            
            # Rolling averages
            if len(self.cpu_usage) >= 10:
                metrics['resources/avg_cpu_usage_10'] = np.mean(list(self.cpu_usage)[-10:])
                metrics['resources/avg_memory_usage_10'] = np.mean(list(self.memory_usage)[-10:])
            
            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage_pct = gpu.load * 100
                    self.gpu_usage.append(gpu_usage_pct)
                    metrics.update({
                        'resources/gpu_usage_pct': gpu_usage_pct,
                        'resources/gpu_memory_usage_pct': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'resources/gpu_memory_used_gb': gpu.memoryUsed / 1024,
                        'resources/gpu_temperature': gpu.temperature
                    })
                    
                    if len(self.gpu_usage) >= 10:
                        metrics['resources/avg_gpu_usage_10'] = np.mean(list(self.gpu_usage)[-10:])
            except ImportError:
                pass  # GPUtil not available
                
        except Exception as e:
            self.logger.warning(f"Failed to collect resource metrics: {e}")
    
    def _add_timing_statistics(self, metrics: Dict[str, Any]) -> None:
        """Add detailed timing statistics."""
        current_time = time.time()
        
        # Session timing breakdown
        metrics.update({
            'timing/session_start_timestamp': self.session_start_time,
            'timing/current_timestamp': current_time,
            'timing/last_log_interval_seconds': current_time - self.last_log_time
        })
        
        # Calculate time distribution
        if self.total_episodes > 0 and self.total_steps > 0:
            avg_steps_per_episode = self.total_steps / self.total_episodes
            avg_time_per_step = (current_time - self.session_start_time) / self.total_steps
            
            metrics.update({
                'timing/avg_steps_per_episode': avg_steps_per_episode,
                'timing/avg_time_per_step_seconds': avg_time_per_step,
                'timing/estimated_time_per_episode': avg_time_per_step * avg_steps_per_episode
            })
    
    def _check_milestones(self) -> None:
        """Check and log milestone achievements."""
        current_time = time.time()
        elapsed_hours = (current_time - self.session_start_time) / 3600
        
        # Episode milestones
        for milestone in self.milestones['episodes']:
            milestone_key = f"episodes_{milestone}"
            if (self.total_episodes >= milestone and 
                milestone_key not in self.milestones_reached):
                
                self.milestones_reached.add(milestone_key)
                wandb.log({
                    'milestones/episodes_milestone': milestone,
                    'milestones/episodes_reached_at_hour': elapsed_hours
                })
                self.logger.info(f"🎯 Milestone reached: {milestone} episodes after {elapsed_hours:.1f} hours")
        
        # Step milestones
        for milestone in self.milestones['steps']:
            milestone_key = f"steps_{milestone}"
            if (self.total_steps >= milestone and 
                milestone_key not in self.milestones_reached):
                
                self.milestones_reached.add(milestone_key)
                wandb.log({
                    'milestones/steps_milestone': milestone,
                    'milestones/steps_reached_at_hour': elapsed_hours
                })
                self.logger.info(f"🎯 Milestone reached: {milestone} steps after {elapsed_hours:.1f} hours")
        
        # Time milestones
        for milestone in self.milestones['hours']:
            milestone_key = f"hours_{milestone}"
            if (elapsed_hours >= milestone and 
                milestone_key not in self.milestones_reached):
                
                self.milestones_reached.add(milestone_key)
                wandb.log({
                    'milestones/runtime_milestone_hours': milestone,
                    'milestones/episodes_at_hour_milestone': self.total_episodes
                })
                self.logger.info(f"🎯 Milestone reached: {milestone} hours with {self.total_episodes} episodes")
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Log final session summary."""
        if not wandb or not wandb.run:
            return
        
        final_time = time.time()
        total_runtime = final_time - self.session_start_time
        total_hours = total_runtime / 3600
        
        # Create comprehensive session summary
        final_summary = {
            'session_final/total_runtime_hours': total_hours,
            'session_final/total_runtime_minutes': total_runtime / 60,
            'session_final/total_episodes': self.total_episodes,
            'session_final/total_steps': self.total_steps,
            'session_final/total_updates': self.total_updates,
            'session_final/total_trades': self.total_trades,
            'session_final/total_volume': self.total_volume,
            'session_final/final_episodes_per_hour': self.total_episodes / total_hours if total_hours > 0 else 0,
            'session_final/final_steps_per_minute': self.total_steps / (total_runtime / 60) if total_runtime > 0 else 0,
            'session_final/session_logs_count': self.session_logs_count,
            'session_final/milestones_reached': len(self.milestones_reached),
            'session_final/end_time': datetime.now().isoformat()
        }
        
        # Add performance summary
        if self.episode_times:
            final_summary.update({
                'session_final/avg_episode_duration': np.mean(list(self.episode_times)),
                'session_final/total_episode_time': np.sum(list(self.episode_times))
            })
        
        # Add efficiency metrics
        if self.total_episodes > 0:
            final_summary['session_final/efficiency_score'] = (self.total_episodes * self.total_steps) / total_runtime
        
        wandb.log(final_summary)
        
        # Log final message
        self.logger.info(
            f"📊 Session completed: {self.total_episodes} episodes, "
            f"{self.total_steps:,} steps, {total_hours:.2f} hours, "
            f"{self.total_episodes/total_hours:.1f} ep/h"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.session_start_time
        
        return {
            'log_frequency': self.log_frequency,
            'track_system_resources': self.track_system_resources,
            'session_logs_count': self.session_logs_count,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'elapsed_hours': elapsed_time / 3600,
            'episodes_per_hour': self.total_episodes / (elapsed_time / 3600) if elapsed_time > 0 else 0,
            'milestones_reached': len(self.milestones_reached),
            'target_episodes': self.target_episodes,
            'target_runtime_hours': self.target_runtime_hours
        }