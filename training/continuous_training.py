"""
Continuous Training - Model Management + Advisory System
Handles model versioning, evaluation, and provides training recommendations.
"""

import os
import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from utils.model_manager import ModelManager


class AdaptationStrategy(Enum):
    """Data difficulty adaptation strategies"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceAnalysis:
    """Performance trend analysis results"""
    trend: str  # "excelling", "stable", "struggling", "plateau", "degrading"
    confidence: float  # 0.0 to 1.0
    recent_performance: float
    baseline_performance: float
    improvement_rate: float
    stability_score: float


@dataclass
class ModelEvaluationResult:
    """Model evaluation results"""
    mean_reward: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    consistency_score: float
    evaluation_episodes: int
    timestamp: datetime


class PerformanceAnalyzer:
    """Analyzes training performance and detects trends"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_history: List[float] = []
        self.evaluation_history: List[ModelEvaluationResult] = []
        
        # Trend detection parameters
        self.excellence_threshold = 0.15  # 15% improvement = excellent
        self.struggle_threshold = -0.10   # 10% degradation = struggling
        self.plateau_patience = 25        # Updates without improvement = plateau
        self.stability_window = 20        # Window for stability calculation
        
    def add_performance(self, performance: float):
        """Add new performance data point"""
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size * 2:
            self.performance_history = self.performance_history[-self.window_size:]
    
    def add_evaluation(self, evaluation: ModelEvaluationResult):
        """Add new evaluation result"""
        self.evaluation_history.append(evaluation)
        
        # Keep only recent evaluations
        if len(self.evaluation_history) > 20:
            self.evaluation_history = self.evaluation_history[-20:]
    
    def analyze_performance(self) -> PerformanceAnalysis:
        """Analyze current performance trend"""
        if len(self.performance_history) < 20:
            return PerformanceAnalysis(
                trend="insufficient_data",
                confidence=0.0,
                recent_performance=0.0,
                baseline_performance=0.0,
                improvement_rate=0.0,
                stability_score=0.0
            )
        
        # Calculate recent vs baseline performance
        recent_window = min(10, len(self.performance_history) // 4)
        baseline_window = min(20, len(self.performance_history) // 2)
        
        recent_performance = np.mean(self.performance_history[-recent_window:])
        baseline_performance = np.mean(
            self.performance_history[-baseline_window:-recent_window]
        ) if len(self.performance_history) > recent_window else recent_performance
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if baseline_performance != 0:
            improvement_rate = (recent_performance - baseline_performance) / abs(baseline_performance)
        
        # Calculate stability score
        recent_std = np.std(self.performance_history[-recent_window:])
        stability_score = 1.0 / (1.0 + recent_std) if recent_std > 0 else 1.0
        
        # Determine trend
        trend = "stable"
        confidence = 0.5
        
        if improvement_rate > self.excellence_threshold:
            trend = "excelling"
            confidence = min(0.9, 0.5 + improvement_rate)
        elif improvement_rate < self.struggle_threshold:
            trend = "struggling"
            confidence = min(0.9, 0.5 + abs(improvement_rate))
        elif self._detect_plateau():
            trend = "plateau"
            confidence = 0.8
        elif self._detect_degradation():
            trend = "degrading"
            confidence = 0.9
        
        return PerformanceAnalysis(
            trend=trend,
            confidence=confidence,
            recent_performance=recent_performance,
            baseline_performance=baseline_performance,
            improvement_rate=improvement_rate,
            stability_score=stability_score
        )
    
    def _detect_plateau(self) -> bool:
        """Detect if performance has plateaued"""
        if len(self.performance_history) < self.plateau_patience:
            return False
        
        recent_performance = self.performance_history[-self.plateau_patience:]
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Plateau if trend is very small
        return abs(trend) < 0.001
    
    def _detect_degradation(self) -> bool:
        """Detect if performance is degrading"""
        if len(self.performance_history) < 30:
            return False
        
        recent_performance = np.mean(self.performance_history[-10:])
        older_performance = np.mean(self.performance_history[-30:-20])
        
        if older_performance > 0:
            degradation = (recent_performance - older_performance) / older_performance
            return degradation < -0.05  # 5% degradation
        
        return False


class DataDifficultyManager:
    """Manages adaptive data difficulty based on performance"""
    
    def __init__(self, initial_quality_range: List[float] = [0.7, 1.0]):
        self.current_quality_range = initial_quality_range.copy()
        self.base_quality_range = initial_quality_range.copy()
        self.adaptation_step = 0.05
        self.min_quality = 0.3
        self.max_quality = 1.0
        
        self.adaptation_history: List[Dict[str, Any]] = []
        
    def adapt_difficulty(self, analysis: PerformanceAnalysis) -> Optional[Dict[str, Any]]:
        """Adapt data difficulty based on performance analysis"""
        if analysis.confidence < 0.6:
            return None  # Not confident enough to adapt
        
        old_range = self.current_quality_range.copy()
        adaptation_made = False
        reason = analysis.trend
        
        if analysis.trend == "excelling":
            # Increase difficulty (lower quality threshold)
            new_min = max(self.min_quality, self.current_quality_range[0] - self.adaptation_step)
            if new_min != self.current_quality_range[0]:
                self.current_quality_range[0] = new_min
                adaptation_made = True
                
        elif analysis.trend == "struggling":
            # Decrease difficulty (higher quality threshold)
            new_min = min(self.max_quality - 0.1, self.current_quality_range[0] + self.adaptation_step)
            if new_min != self.current_quality_range[0]:
                self.current_quality_range[0] = new_min
                adaptation_made = True
        
        if adaptation_made:
            adaptation_record = {
                'timestamp': datetime.now().isoformat(),
                'old_range': old_range,
                'new_range': self.current_quality_range.copy(),
                'reason': reason,
                'confidence': analysis.confidence,
                'improvement_rate': analysis.improvement_rate
            }
            
            self.adaptation_history.append(adaptation_record)
            
            return {
                'quality_range': self.current_quality_range.copy(),
                'adaptation_record': adaptation_record
            }
        
        return None


class ContinuousTraining:
    """
    Continuous Training - Model Management + Advisory System
    
    Responsibilities:
    - Model versioning, loading, and saving
    - Performance analysis and trend detection
    - Adaptive recommendations for training improvements
    - Model evaluation coordination
    """
    
    def __init__(self, config: Dict[str, Any], mode: str = "production", enabled: bool = True):
        self.config = config
        self.mode = mode
        self.enabled = enabled
        self.bypass_all_features = config.get('bypass_all_features', False)
        self.logger = logging.getLogger(__name__)
        
        # If bypassing all features (optuna mode), initialize minimal state
        if self.bypass_all_features:
            self.model_manager = None
            self.best_reward = 0.0
            self.best_model_path = None
            self.performance_analyzer = None
            self.difficulty_manager = None
            return
        
        # Model management
        self.model_manager = ModelManager()
        self.best_reward = float('-inf')
        self.best_model_path: Optional[str] = None
        
        # Performance analysis
        self.performance_analyzer = PerformanceAnalyzer(
            window_size=config.get('performance_window', 50)
        )
        
        # Data difficulty management
        initial_quality_range = config.get('initial_quality_range', [0.7, 1.0])
        self.difficulty_manager = DataDifficultyManager(initial_quality_range)
        
        # Recommendation settings
        self.recommendation_frequency = config.get('recommendation_frequency', 10)
        self.checkpoint_frequency = config.get('checkpoint_frequency', 25)
        
        # Evaluation settings - check nested structure first, then fallback to flat
        evaluation_config = config.get('evaluation', {})
        if isinstance(evaluation_config, dict):
            self.evaluation_frequency = evaluation_config.get('frequency', 50)
            self.evaluation_episodes = evaluation_config.get('episodes', 10)
            self.logger.info(f"📊 Evaluation config from nested structure: frequency={self.evaluation_frequency}, episodes={self.evaluation_episodes}")
        else:
            # Fallback to flat structure for backward compatibility
            self.evaluation_frequency = config.get('evaluation_frequency', 50)
            self.evaluation_episodes = config.get('evaluation_episodes', 10)
            self.logger.info(f"📊 Evaluation config from flat structure: frequency={self.evaluation_frequency}, episodes={self.evaluation_episodes}")
        
        # State tracking
        self.session_start_time: Optional[float] = None
        self.last_checkpoint_update = 0
        self.last_evaluation_update = 0
        self.updates_since_recommendation = 0
        
        # Model loading on initialization
        self.load_metadata = self._load_best_model_if_exists()
        
        self.logger.info(f"🔄 ContinuousTraining initialized in {mode} mode (enabled: {enabled})")
    
    def _load_best_model_if_exists(self) -> Dict[str, Any]:
        """Load existing best model if available"""
        try:
            best_model_info = self.model_manager.get_best_model_info()
            if best_model_info and 'reward' in best_model_info:
                self.best_reward = best_model_info['reward']
                self.best_model_path = best_model_info.get('path')
                self.logger.info(f"📊 Loaded best model: reward={self.best_reward:.4f}")
                return best_model_info
        except Exception as e:
            self.logger.warning(f"Could not load existing best model: {e}")
        
        return {}
    
    def initialize(self, trainer):
        """Initialize continuous training session"""
        self.session_start_time = time.time()
        
    
    def get_recommendations(self, training_state, performance_metrics: Dict[str, Any]):
        """
        Get training recommendations based on evaluation results (NOT training metrics)
        Returns TrainingRecommendations object
        
        NOTE: This method now only provides recommendations when evaluation data is available.
        Training metrics are ignored - only evaluation results drive recommendations.
        """
        from training.training_manager import TrainingRecommendations
        
        if not self.enabled or self.bypass_all_features:
            return TrainingRecommendations.no_changes()
        
        recommendations = TrainingRecommendations()
        
        # Check if it's time for recommendations
        self.updates_since_recommendation += 1
        if self.updates_since_recommendation < self.recommendation_frequency:
            return recommendations
        
        self.updates_since_recommendation = 0
        
        # Check for evaluation data - if insufficient, still allow training-based recommendations
        has_sufficient_evaluation_data = (
            self.performance_analyzer.evaluation_history and 
            len(self.performance_analyzer.evaluation_history) >= 2
        )
        
        if not has_sufficient_evaluation_data:
            self.logger.debug("🔄 No sufficient evaluation data for evaluation-based recommendations")
            # Request evaluation if we don't have recent data
            recommendations.evaluation_request = True
            # Continue to allow training-based recommendations below
        
        # Evaluation-based recommendations (only if we have sufficient data)
        if has_sufficient_evaluation_data:
            # Analyze evaluation performance (NOT training performance)
            analysis = self._analyze_evaluation_performance()
            
            # Data difficulty recommendations (based on evaluation results)
            if self.mode == "production":  # Only adapt in production mode
                difficulty_change = self.difficulty_manager.adapt_difficulty(analysis)
                if difficulty_change:
                    recommendations.data_difficulty_change = difficulty_change
                    self.logger.info(f"📊 Recommending difficulty change based on evaluation: {difficulty_change['adaptation_record']['reason']}")
            
            # Evaluation-based checkpoint recommendations
            if self._should_request_checkpoint_from_evaluation():
                recommendations.checkpoint_request = True
            
            # Termination recommendations (based on evaluation trends only)
            if self.mode == "production":
                termination_reason = self._analyze_termination_from_evaluation(analysis, training_state)
                if termination_reason:
                    recommendations.termination_suggestion = termination_reason
                    self.logger.info(f"🛑 Recommending termination based on evaluation: {termination_reason.value}")
        
        # Training-based recommendations (always available)
        if self._should_request_checkpoint_from_training(training_state):
            recommendations.checkpoint_request = True
        
        # Evaluation recommendations (schedule next evaluation)
        if self._should_request_evaluation(training_state):
            recommendations.evaluation_request = True
        
        return recommendations
    
    def _analyze_evaluation_performance(self) -> PerformanceAnalysis:
        """Analyze evaluation performance instead of training performance"""
        if not self.performance_analyzer.evaluation_history:
            return PerformanceAnalysis(
                trend="unknown",
                confidence=0.0,
                recent_performance=0.0,
                baseline_performance=0.0,
                improvement_rate=0.0,
                stability_score=0.0
            )
        
        # Get evaluation rewards
        eval_rewards = [result.mean_reward for result in self.performance_analyzer.evaluation_history]
        
        # Use the performance analyzer but with evaluation data
        temp_analyzer = PerformanceAnalyzer(window_size=len(eval_rewards))
        for reward in eval_rewards:
            temp_analyzer.add_performance(reward)
        
        return temp_analyzer.analyze_performance()
    
    def _should_request_checkpoint_from_evaluation(self) -> bool:
        """Determine if checkpoint should be requested based on evaluation results"""
        if not self.performance_analyzer.evaluation_history:
            return False
        
        # Get recent evaluation performance
        recent_eval = self.performance_analyzer.evaluation_history[-1]
        
        # Check if recent evaluation shows improvement
        if len(self.performance_analyzer.evaluation_history) >= 2:
            previous_eval = self.performance_analyzer.evaluation_history[-2]
            if recent_eval.mean_reward > previous_eval.mean_reward:
                self.logger.info(f"📈 Evaluation improvement detected: {previous_eval.mean_reward:.4f} → {recent_eval.mean_reward:.4f}")
                return True
        
        # Check if this is a new best evaluation result
        best_eval_reward = max(result.mean_reward for result in self.performance_analyzer.evaluation_history)
        if recent_eval.mean_reward >= best_eval_reward:
            self.logger.info(f"🏆 New best evaluation result: {recent_eval.mean_reward:.4f}")
            return True
        
        return False
    
    def _should_request_checkpoint_from_training(self, training_state) -> bool:
        """Determine if checkpoint should be requested based on training progress"""
        updates_elapsed = training_state.updates - self.last_checkpoint_update
        
        # Request checkpoint based on training frequency
        return updates_elapsed >= self.checkpoint_frequency
    
    def _analyze_termination_from_evaluation(self, analysis: PerformanceAnalysis, training_state) -> Optional:
        """Analyze termination conditions based on evaluation trends"""
        from training.training_manager import TerminationReason
        
        if not self.performance_analyzer.evaluation_history or len(self.performance_analyzer.evaluation_history) < 3:
            return None
        
        # Check for evaluation plateau
        if analysis.trend == "plateau":
            self.logger.info("🔍 Evaluation plateau detected")
            return TerminationReason.PERFORMANCE_PLATEAU
        
        # Check for evaluation degradation
        if analysis.trend == "degrading" and analysis.confidence > 0.7:
            self.logger.info("🔍 Evaluation degradation detected")
            return TerminationReason.PERFORMANCE_DEGRADATION
        
        return None
    
    def _should_request_checkpoint(self, training_state, performance_metrics: Dict[str, Any]) -> bool:
        """Determine if a checkpoint should be requested"""
        current_performance = performance_metrics.get('mean_reward', float('-inf'))
        updates_elapsed = training_state.updates - self.last_checkpoint_update
        
        # Checkpoint if performance improved significantly or periodic interval
        performance_improved = current_performance > self.best_reward
        periodic_checkpoint = updates_elapsed >= self.checkpoint_frequency
        
        return performance_improved or periodic_checkpoint
    
    def _should_request_evaluation(self, training_state) -> bool:
        """Determine if an evaluation should be requested"""
        updates_elapsed = training_state.updates - self.last_evaluation_update
        return updates_elapsed >= self.evaluation_frequency
    
    def _analyze_termination_conditions(self, analysis: PerformanceAnalysis, training_state) -> Optional:
        """Analyze if training should be terminated"""
        from training.training_manager import TerminationReason
        
        # Suggest termination for plateau with high confidence
        if analysis.trend == "plateau" and analysis.confidence > 0.8:
            return TerminationReason.PERFORMANCE_PLATEAU
        
        # Suggest termination for significant degradation
        if analysis.trend == "degrading" and analysis.confidence > 0.9:
            return TerminationReason.PERFORMANCE_DEGRADATION
        
        return None
    
    def handle_checkpoint_request(self, trainer):
        """Handle checkpoint request from training manager"""
        current_update = getattr(trainer, 'global_update_counter', 0)
        current_episode = getattr(trainer, 'global_episode_counter', 0)
        current_steps = getattr(trainer, 'global_step_counter', 0)
        
        # Save current model
        checkpoint_path = os.path.join(
            trainer.model_dir,
            f"checkpoint_iter{current_update}.pt"
        )
        trainer.save_model(checkpoint_path)
        
        # Get comprehensive current metrics from trainer
        current_performance = getattr(trainer, 'mean_episode_reward', 0.0)
        
        # Get additional training metrics
        total_episodes = getattr(trainer, 'global_episode_counter', 0)
        recent_rewards = getattr(trainer, 'recent_episode_rewards', [])
        
        # Calculate more robust performance metric
        if len(recent_rewards) > 0:
            recent_performance = float(np.mean(recent_rewards[-10:]))  # Last 10 episodes
        else:
            recent_performance = current_performance
            
        # Use the better of the two metrics
        effective_performance = max(current_performance, recent_performance)
        
        # Always save checkpoints with meaningful training progress (not just reward improvement)
        should_save_as_best = (
            effective_performance > self.best_reward or 
            (current_update > 0 and total_episodes > 0) or  # Has actual training progress
            self.best_model_path is None  # First checkpoint
        )
        
        if should_save_as_best:
            # Update best tracking
            if effective_performance > self.best_reward:
                self.best_reward = effective_performance
            self.best_model_path = checkpoint_path
            
            # Create comprehensive metrics
            metrics = {
                'mean_reward': effective_performance,
                'recent_reward': recent_performance,
                'update_iter': current_update,
                'episode_count': total_episodes,
                'global_steps': current_steps,
                'timestamp': time.time(),
                'mode': self.mode,
                'recent_episodes_count': len(recent_rewards),
                'training_active': True
            }
            
            self.model_manager.save_best_model(checkpoint_path, metrics, effective_performance)
            self.logger.info(f"💾 Model checkpoint saved: reward={effective_performance:.4f}, episodes={total_episodes}, updates={current_update}")
        
        self.last_checkpoint_update = current_update
    
    def process_evaluation_results(self, eval_results: Dict[str, Any]):
        """Process evaluation results from trainer"""
        evaluation = ModelEvaluationResult(
            mean_reward=eval_results.get('mean_reward', 0.0),
            sharpe_ratio=eval_results.get('sharpe_ratio', 0.0),
            win_rate=eval_results.get('win_rate', 0.0),
            max_drawdown=eval_results.get('max_drawdown', 0.0),
            consistency_score=eval_results.get('consistency_score', 0.0),
            evaluation_episodes=eval_results.get('n_episodes', 0),
            timestamp=datetime.now()
        )
        
        self.performance_analyzer.add_evaluation(evaluation)
        self.last_evaluation_update = eval_results.get('update_iter', 0)
        
        self.logger.info(f"📈 Evaluation processed: reward={evaluation.mean_reward:.4f}")
    
    def on_training_termination(self, reason):
        """Handle training termination notification"""
        self.logger.info(f"🏁 Training termination received: {reason.value}")
    
    def finalize_training(self, final_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize training session and return additional statistics"""
        if self.bypass_all_features:
            return {}  # Return empty dict for optuna mode
            
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        # Create final statistics
        continuous_stats = {
            'continuous_training_session_duration': session_duration,
            'continuous_training_best_reward': self.best_reward,
            'continuous_training_adaptations': len(self.difficulty_manager.adaptation_history),
            'continuous_training_evaluations': len(self.performance_analyzer.evaluation_history),
            'continuous_training_mode': self.mode
        }
        
        # Save session summary
        self._save_session_summary(final_stats, continuous_stats)
        
        self.logger.info(
            f"🎯 Continuous training finalized: "
            f"duration={session_duration:.1f}s, best_reward={self.best_reward:.4f}"
        )
        
        return continuous_stats
    
    def _save_session_summary(self, final_stats: Dict[str, Any], continuous_stats: Dict[str, Any]):
        """Save detailed session summary"""
        summary = {
            **final_stats,
            **continuous_stats,
            'adaptation_history': self.difficulty_manager.adaptation_history,
            'evaluation_history': [
                {
                    'mean_reward': eval_result.mean_reward,
                    'timestamp': eval_result.timestamp.isoformat(),
                    'evaluation_episodes': eval_result.evaluation_episodes
                }
                for eval_result in self.performance_analyzer.evaluation_history
            ],
            'final_quality_range': self.difficulty_manager.current_quality_range,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        summary_path = os.path.join(
            self.model_manager.base_dir, 
            f"continuous_training_summary_{int(time.time())}.json"
        )
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"📄 Session summary saved: {summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get current best model information"""
        return {
            'best_reward': self.best_reward,
            'best_model_path': self.best_model_path,
            'current_quality_range': self.difficulty_manager.current_quality_range,
            'total_adaptations': len(self.difficulty_manager.adaptation_history),
            'mode': self.mode
        }