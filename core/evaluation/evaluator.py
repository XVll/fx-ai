"""
Core Evaluator class for model performance assessment.

This module provides clean separation between training and evaluation logic,
with proper state management and deterministic behavior.
"""

import logging
import statistics
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import random
import numpy as np
import torch

from config.evaluation.evaluation_config import EvaluationConfig
from core.evaluation import EvaluationResult, EvaluationEpisodeResult
from agent.ppo_agent import PPOTrainer
from envs import TradingEnvironment
from data.data_manager import DataManager
from training.episode_manager import EpisodeManager


class EvaluationState:
    """Container for saved state during evaluation."""
    
    def __init__(self):
        self.trainer_mode: bool = True
        self.random_state: Any = None
        self.numpy_state: Any = None 
        self.torch_state: Any = None
        self.episode_manager_state: Any = None


class Evaluator:
    """
    Pure evaluation class with clean separation of concerns.
    
    Responsibilities:
    - Run deterministic model evaluation episodes
    - Manage evaluation state isolation
    - Calculate aggregate evaluation metrics
    - Support both periodic and standalone evaluation
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.Evaluator")
        
        # State management
        self._saved_state: Optional[EvaluationState] = None
        
        self.logger.info(f"🔍 Evaluator initialized (episodes={config.episodes}, seed={config.seed})")
    
    def evaluate_model(
        self,
        trainer: PPOTrainer,
        environment: TradingEnvironment,
        data_manager: DataManager,
        episode_manager: Optional[EpisodeManager] = None
    ) -> Optional[EvaluationResult]:
        """
        Run complete model evaluation and return aggregated results.
        
        Args:
            trainer: PPO trainer with model to evaluate
            environment: Trading environment (will be reset for each episode)
            data_manager: Data manager for episode selection
            episode_manager: Optional episode manager for episode selection
            
        Returns:
            EvaluationResult with aggregated metrics, or None if evaluation failed
        """
        if not self.config.enabled:
            self.logger.debug("Evaluation disabled in config")
            return None
            
        try:
            self.logger.info(f"🔍 Starting model evaluation ({self.config.episodes} episodes)")
            
            # Save current state for restoration
            self._save_state(trainer, episode_manager)
            
            try:
                # Enter deterministic evaluation mode
                self._enter_evaluation_mode()
                
                # Get evaluation episodes
                episodes = self._get_evaluation_episodes(data_manager, episode_manager)
                if not episodes:
                    self.logger.warning("No evaluation episodes available")
                    return None
                
                # Run evaluation episodes
                episode_results = self._run_evaluation_episodes(
                    trainer, environment, episodes
                )
                
                # Calculate aggregate metrics
                result = self._calculate_evaluation_metrics(episode_results)
                
                self.logger.info(f"✅ Evaluation complete: mean_reward={result.mean_reward:.4f}, episodes={len(episode_results)}")
                return result
                
            finally:
                # Always restore state
                self._restore_state(trainer, episode_manager)
                
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
            return None
    
    def evaluate_single_episode(
        self,
        trainer: PPOTrainer,
        environment: TradingEnvironment,
        episode_context: Any,
        max_steps: int = 1000
    ) -> Optional[float]:
        """
        Evaluate a single episode and return the total reward.
        
        Args:
            trainer: PPO trainer with model to evaluate
            environment: Trading environment
            episode_context: Episode context for environment setup
            max_steps: Maximum steps per episode (safety limit)
            
        Returns:
            Total reward for the episode, or None if failed
        """
        try:
            # Setup environment for this episode
            setup_success, initial_obs = self._setup_episode(environment, episode_context)
            if not setup_success:
                self.logger.warning(f"Failed to setup episode for evaluation")
                return None
            
            # Run episode with trainer's evaluate method
            total_reward = trainer.evaluate(
                environment=environment,
                initial_obs=initial_obs,
                deterministic=self.config.deterministic_actions,
                max_steps=max_steps
            )
            
            return total_reward
            
        except Exception as e:
            self.logger.warning(f"Single episode evaluation failed: {e}")
            return None
    
    def _save_state(self, trainer: PPOTrainer, episode_manager: Optional[Any]) -> None:
        """Save current state for restoration after evaluation."""
        self._saved_state = EvaluationState()
        
        # Save model training mode
        self._saved_state.trainer_mode = getattr(trainer.model, 'training', True)
        
        # Save RNG states
        self._saved_state.random_state = random.getstate()
        self._saved_state.numpy_state = np.random.get_state()
        self._saved_state.torch_state = torch.get_rng_state()
        
        # Save episode manager state if available
        if episode_manager and hasattr(episode_manager, 'get_current_state'):
            self._saved_state.episode_manager_state = episode_manager.get_current_state()
        
        self.logger.debug("🔄 State saved for evaluation")
    
    def _restore_state(self, trainer: PPOTrainer, episode_manager: Optional[Any]) -> None:
        """Restore state after evaluation."""
        if not self._saved_state:
            self.logger.warning("No saved state to restore")
            return
            
        try:
            # Restore RNG states
            random.setstate(self._saved_state.random_state)
            np.random.set_state(self._saved_state.numpy_state)
            torch.set_rng_state(self._saved_state.torch_state)
            
            # Restore model training mode
            if self._saved_state.trainer_mode:
                trainer.model.train()
            else:
                trainer.model.eval()
            
            # Restore episode manager state if available
            if (self._saved_state.episode_manager_state and 
                episode_manager and 
                hasattr(episode_manager, 'restore_state')):
                episode_manager.restore_state(self._saved_state.episode_manager_state)
            
            self.logger.debug("🔄 State restored after evaluation")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to restore some state: {e}")
        finally:
            self._saved_state = None
    
    def _enter_evaluation_mode(self) -> None:
        """Enter deterministic evaluation mode with fixed seeds."""
        # Set deterministic seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        self.logger.debug(f"🎯 Entered evaluation mode (seed={self.config.seed})")
    
    def _get_evaluation_episodes(
        self, 
        data_manager: DataManager, 
        episode_manager: Optional[Any]
    ) -> List[Any]:
        """Get episodes to use for evaluation."""
        episodes = []
        
        if episode_manager and hasattr(episode_manager, 'get_evaluation_episodes'):
            # Use episode manager's evaluation episode selection
            episodes = episode_manager.get_evaluation_episodes(self.config)
        else:
            # Fallback: get episodes from episode manager or data manager
            if episode_manager:
                for _ in range(self.config.episodes):
                    try:
                        episode = episode_manager.get_next_episode()
                        episodes.append(episode)
                    except Exception as e:
                        self.logger.debug(f"Could not get episode {len(episodes)}: {e}")
                        break
            else:
                # Direct data manager usage (for standalone evaluation)
                # This would need to be implemented based on your data manager interface
                self.logger.warning("No episode manager provided - limited episode selection")
        
        self.logger.debug(f"Selected {len(episodes)} episodes for evaluation")
        return episodes
    
    def _run_evaluation_episodes(
        self,
        trainer: PPOTrainer,
        environment: TradingEnvironment,
        episodes: List[Any]
    ) -> List[EvaluationEpisodeResult]:
        """Run evaluation episodes and collect results."""
        episode_results = []
        
        for i, episode_context in enumerate(episodes):
            total_reward = self.evaluate_single_episode(
                trainer, environment, episode_context, max_steps=1000
            )
            
            if total_reward is not None:
                episode_results.append(EvaluationEpisodeResult(
                    episode_num=i,
                    reward=total_reward
                ))
                self.logger.debug(f"Eval episode {i}: reward={total_reward:.4f}")
            else:
                self.logger.warning(f"Evaluation episode {i} failed")
        
        return episode_results
    
    def _setup_episode(
        self, 
        environment: TradingEnvironment, 
        episode_context: Any
    ) -> Tuple[bool, Optional[Dict]]:
        """Setup environment for an evaluation episode."""
        try:
            # Extract episode information
            symbol = episode_context.symbol
            date = episode_context.date
            reset_point = episode_context.reset_point
            
            self.logger.debug(f"Setting up eval episode: {symbol} {date} at {reset_point.timestamp}")
            
            # Reset environment with evaluation seed
            obs, info = environment.reset(
                symbol=symbol,
                date=date,
                reset_point=reset_point,
                seed=self.config.seed  # Pass evaluation seed for determinism
            )
            
            return True, obs
            
        except Exception as e:
            self.logger.warning(f"Failed to setup evaluation episode: {e}")
            return False, None
    
    def _calculate_evaluation_metrics(
        self, 
        episode_results: List[EvaluationEpisodeResult]
    ) -> EvaluationResult:
        """Calculate aggregate metrics from episode results."""
        if not episode_results:
            self.logger.warning("No evaluation episodes completed")
            rewards = [0.0]
        else:
            rewards = [ep.reward for ep in episode_results]
        
        return EvaluationResult(
            timestamp=datetime.now(),
            model_version=None,  # TODO: Get from model manager if needed
            config=self.config,
            episodes=episode_results,
            mean_reward=statistics.mean(rewards),
            std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            min_reward=min(rewards),
            max_reward=max(rewards),
            total_episodes=len(episode_results)
        )