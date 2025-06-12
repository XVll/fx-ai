"""
Continuous training callback that manages best models and checkpoints based on evaluation results.

This callback integrates with the evaluation system to track best models,
save checkpoints, and manage the continuous training workflow.
"""

import logging
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from callbacks.core.base import BaseCallback
from config.callbacks.callback_config import ContinuousCallbackConfig
from core.model_manager import ModelManager


class ContinuousTrainingCallback(BaseCallback):
    """
    Manages continuous training workflow with evaluation-based model management.
    
    Features:
    - Best model tracking based on evaluation results
    - Periodic checkpoint saving
    - Training session management
    - Model versioning and metadata tracking
    - Integration with ModelManager
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config: ContinuousCallbackConfig,
    ):
        """
        Initialize continuous training callback.
        
        Args:
            model_manager: ModelManager instance for model storage
        """
        super().__init__(name="ContinuousTrainingCallback", enabled=config.enabled, config=config)
        
        self.model_manager = model_manager
        self.metric_name = config.metric_name
        self.metric_mode = config.metric_mode
        self.checkpoint_frequency = config.checkpoint_frequency
        self.checkpoint_time_interval = config.checkpoint_time_interval
        self.save_initial_model = config.save_initial_model
        
        # Best model tracking
        self.best_metric_value = float('-inf') if config.metric_mode == 'max' else float('inf')
        self.best_model_path: Optional[str] = None
        self.best_evaluation_result: Optional[EvaluationResult] = None
        
        # Checkpoint tracking
        self.last_checkpoint_update = 0
        self.last_checkpoint_time = time.time()
        self.session_start_time: Optional[float] = None
        
        # Model saving tracking
        self.models_saved = 0
        self.initial_model_saved = False
        
        # Load previous best if continuing training
        self._load_previous_best()
        
        self.logger.info(
            f"🏭 ContinuousTrainingCallback initialized "
            f"(metric={config.metric_name}, mode={config.metric_mode}, freq={config.checkpoint_frequency})"
        )
    
    def _load_previous_best(self) -> None:
        """Load previous best model information if continuing training."""
        try:
            best_model_info = self.model_manager.get_best_model_info()
            if best_model_info:
                self.best_metric_value = best_model_info.get('reward', self.best_metric_value)
                self.best_model_path = best_model_info.get('path')
                
                self.logger.info(
                    f"📁 Continuing from previous best model: "
                    f"{self.metric_name}={self.best_metric_value:.4f} "
                    f"(path: {self.best_model_path})"
                )
        except Exception as e:
            self.logger.debug(f"No previous best model found or failed to load: {e}")
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize training session state."""
        self.session_start_time = time.time()
        self.last_checkpoint_time = self.session_start_time
        self.last_checkpoint_update = 0
        self.models_saved = 0
        self.initial_model_saved = False
        
        self.logger.info("🏭 Continuous training session started")
        
        # Save initial model if configured
        if self.save_initial_model and not self.initial_model_saved:
            self._save_initial_model(context)
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Check for periodic checkpoint saving after each update."""
        current_update = context.get('global_updates', 0)
        
        # Check if periodic checkpoint is needed
        if self._should_save_periodic_checkpoint(current_update):
            self._save_periodic_checkpoint(context)
    
    def on_evaluation_complete(self, context: Dict[str, Any]) -> None:
        """
        Handle evaluation results - check for best model and save if needed.
        
        This is the main integration point with the evaluation system.
        """
        evaluation_result = context.get('evaluation_result')
        if not evaluation_result:
            self.logger.warning("No evaluation result in context")
            return
        
        # Extract metric value
        metric_value = self._extract_metric_value(evaluation_result)
        if metric_value is None:
            self.logger.warning(f"Could not extract {self.metric_name} from evaluation result")
            return
        
        # Check if this is a new best model
        is_best = self._is_best_model(metric_value)
        
        if is_best:
            self._save_best_model(context, evaluation_result, metric_value)
        else:
            self.logger.debug(
                f"📊 Model not improved: {self.metric_name}={metric_value:.4f} "
                f"(best: {self.best_metric_value:.4f})"
            )
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Save final model and session summary."""
        if not self.session_start_time:
            return
        
        session_duration = time.time() - self.session_start_time
        
        # Save final model
        self._save_final_model(context, session_duration)
        
        # Log session summary
        self.logger.info(
            f"🏁 Training session completed: "
            f"Duration: {session_duration:.1f}s, "
            f"Models saved: {self.models_saved}, "
            f"Best {self.metric_name}: {self.best_metric_value:.4f}"
        )
        
        # Save session summary to file
        self._save_session_summary(context, session_duration)
    
    def _extract_metric_value(self, evaluation_result: EvaluationResult) -> Optional[float]:
        """Extract the target metric value from evaluation result."""
        try:
            # Try to get from evaluation metrics first
            metrics = getattr(evaluation_result, 'evaluation_metrics', {})
            if self.metric_name in metrics:
                return float(metrics[self.metric_name])
            
            # Try direct attribute access
            if hasattr(evaluation_result, self.metric_name):
                return float(getattr(evaluation_result, self.metric_name))
            
            # Default to mean_reward
            return float(evaluation_result.mean_reward)
            
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Failed to extract metric {self.metric_name}: {e}")
            return None
    
    def _is_best_model(self, metric_value: float) -> bool:
        """Check if the current metric value represents a new best model."""
        if self.metric_mode == 'max':
            return metric_value > self.best_metric_value
        else:  # 'min'
            return metric_value < self.best_metric_value
    
    def _save_initial_model(self, context: Dict[str, Any]) -> None:
        """Save the initial model as a baseline."""
        if not self.trainer:
            self.logger.warning("No trainer available for initial model save")
            return
        
        try:
            # Create temporary checkpoint path
            temp_path = self._create_temp_checkpoint_path(context, "initial")
            
            # Save model
            self.trainer.save_model(temp_path)
            
            # Create metadata
            metadata = self._create_model_metadata(context, "initial")
            
            # Save via model manager
            saved_path = self.model_manager.save_best_model(
                model_path=temp_path,
                metrics=metadata,
                target_reward=self.best_metric_value
            )
            
            self.initial_model_saved = True
            self.models_saved += 1
            
            self.logger.info(f"💾 Initial model saved: {saved_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save initial model: {e}")
    
    def _save_best_model(
        self, 
        context: Dict[str, Any], 
        evaluation_result: EvaluationResult, 
        metric_value: float
    ) -> None:
        """Save a new best model."""
        if not self.trainer:
            self.logger.warning("No trainer available for best model save")
            return
        
        try:
            # Update best tracking
            self.best_metric_value = metric_value
            self.best_evaluation_result = evaluation_result
            
            # Create temporary checkpoint path
            temp_path = self._create_temp_checkpoint_path(context, "best")
            
            # Save model
            self.trainer.save_model(temp_path)
            
            # Create comprehensive metadata
            metadata = self._create_model_metadata(context, "best", evaluation_result)
            
            # Save via model manager
            saved_path = self.model_manager.save_best_model(
                model_path=temp_path,
                metrics=metadata,
                target_reward=metric_value
            )
            
            self.best_model_path = saved_path
            self.models_saved += 1
            
            self.logger.info(
                f"🏆 New best model saved: {self.metric_name}={metric_value:.4f} "
                f"(path: {saved_path})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save best model: {e}")
    
    def _should_save_periodic_checkpoint(self, current_update: int) -> bool:
        """Check if a periodic checkpoint should be saved."""
        # Check update frequency
        updates_since_last = current_update - self.last_checkpoint_update
        if updates_since_last < self.checkpoint_frequency:
            return False
        
        # Check time interval
        time_since_last = time.time() - self.last_checkpoint_time
        if time_since_last < self.checkpoint_time_interval:
            return False
        
        return True
    
    def _save_periodic_checkpoint(self, context: Dict[str, Any]) -> None:
        """Save a periodic checkpoint."""
        if not self.trainer:
            return
        
        try:
            current_update = context.get('global_updates', 0)
            
            # Create checkpoint path
            temp_path = self._create_temp_checkpoint_path(context, "periodic")
            
            # Save model
            self.trainer.save_model(temp_path)
            
            # Create metadata
            metadata = self._create_model_metadata(context, "periodic")
            
            # Save via model manager (with current best reward)
            saved_path = self.model_manager.save_best_model(
                model_path=temp_path,
                metrics=metadata,
                target_reward=self.best_metric_value
            )
            
            # Update tracking
            self.last_checkpoint_update = current_update
            self.last_checkpoint_time = time.time()
            self.models_saved += 1
            
            self.logger.debug(f"💾 Periodic checkpoint saved: {saved_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save periodic checkpoint: {e}")
    
    def _save_final_model(self, context: Dict[str, Any], session_duration: float) -> None:
        """Save the final model at the end of training."""
        if not self.trainer:
            return
        
        try:
            # Create final checkpoint path
            temp_path = self._create_temp_checkpoint_path(context, "final")
            
            # Save model
            self.trainer.save_model(temp_path)
            
            # Create final metadata
            metadata = self._create_model_metadata(context, "final")
            metadata.update({
                'session_duration': session_duration,
                'models_saved_this_session': self.models_saved,
                'is_final': True
            })
            
            # Save via model manager
            saved_path = self.model_manager.save_best_model(
                model_path=temp_path,
                metrics=metadata,
                target_reward=self.best_metric_value
            )
            
            self.models_saved += 1
            
            self.logger.info(f"🏁 Final model saved: {saved_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final model: {e}")
    
    def _create_temp_checkpoint_path(self, context: Dict[str, Any], checkpoint_type: str) -> str:
        """Create a temporary checkpoint path for saving."""
        current_update = context.get('global_updates', 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use trainer's model directory if available
        if hasattr(self.trainer, 'model_dir') and self.trainer.model_dir:
            base_dir = Path(self.trainer.model_dir)
        else:
            base_dir = Path("cache/temp")
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{checkpoint_type}_checkpoint_u{current_update}_{timestamp}.pt"
        return str(base_dir / filename)
    
    def _create_model_metadata(
        self, 
        context: Dict[str, Any], 
        model_type: str,
        evaluation_result: Optional[EvaluationResult] = None
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for saved models."""
        metadata = {
            'model_type': model_type,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'global_updates': context.get('global_updates', 0),
            'total_episodes': context.get('total_episodes', self.episodes_seen),
            'metric_name': self.metric_name,
            'metric_mode': self.metric_mode,
            'best_metric_value': self.best_metric_value,
            'session_models_saved': self.models_saved,
        }
        
        # Add evaluation result data if available
        if evaluation_result:
            metadata.update({
                'evaluation_mean_reward': evaluation_result.mean_reward,
                'evaluation_std_reward': evaluation_result.std_reward,
                'evaluation_min_reward': evaluation_result.min_reward,
                'evaluation_max_reward': evaluation_result.max_reward,
                'evaluation_episodes': evaluation_result.total_episodes,
            })
            
            # Add extended metrics if available
            for attr in ['success_rate', 'sharpe_ratio', 'max_drawdown', 'total_return']:
                if hasattr(evaluation_result, attr):
                    metadata[f'evaluation_{attr}'] = getattr(evaluation_result, attr)
        
        # Add training context
        if hasattr(self.trainer, 'config'):
            metadata['trainer_config'] = str(self.trainer.config)
        
        return metadata
    
    def _save_session_summary(self, context: Dict[str, Any], session_duration: float) -> None:
        """Save a summary of the training session."""
        try:
            summary = {
                'session_start': self.session_start_time,
                'session_duration': session_duration,
                'session_end': time.time(),
                'models_saved': self.models_saved,
                'best_metric_name': self.metric_name,
                'best_metric_value': self.best_metric_value,
                'best_model_path': self.best_model_path,
                'final_global_updates': context.get('global_updates', 0),
                'final_total_episodes': context.get('total_episodes', self.episodes_seen),
                'checkpoint_frequency': self.checkpoint_frequency,
                'checkpoint_time_interval': self.checkpoint_time_interval,
            }
            
            # Add best evaluation result if available
            if self.best_evaluation_result:
                summary['best_evaluation'] = {
                    'mean_reward': self.best_evaluation_result.mean_reward,
                    'std_reward': self.best_evaluation_result.std_reward,
                    'total_episodes': self.best_evaluation_result.total_episodes,
                    'timestamp': getattr(self.best_evaluation_result, 'timestamp', None)
                }
            
            # Save to file
            if hasattr(self.trainer, 'output_dir') and self.trainer.output_dir:
                summary_path = Path(self.trainer.output_dir) / "continuous_training_summary.json"
            else:
                summary_path = Path("outputs") / "continuous_training_summary.json"
            
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.debug(f"Session summary saved to: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
    
    # Public API
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """Get information about the current best model."""
        return {
            'metric_name': self.metric_name,
            'metric_value': self.best_metric_value,
            'metric_mode': self.metric_mode,
            'model_path': self.best_model_path,
            'models_saved': self.models_saved,
            'evaluation_result': self.best_evaluation_result,
        }
    
    def force_checkpoint(self, context: Dict[str, Any]) -> Optional[str]:
        """Force an immediate checkpoint save."""
        self.logger.info("🔄 Forcing checkpoint save")
        try:
            self._save_periodic_checkpoint(context)
            return self.best_model_path
        except Exception as e:
            self.logger.error(f"Failed to force checkpoint: {e}")
            return None
    
    def update_checkpoint_frequency(self, new_frequency: int) -> None:
        """Update the checkpoint frequency."""
        if new_frequency > 0:
            self.checkpoint_frequency = new_frequency
            self.logger.info(f"Updated checkpoint frequency to {new_frequency}")