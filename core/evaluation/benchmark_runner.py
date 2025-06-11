"""
BenchmarkRunner for standalone model benchmarking.

This module provides a complete benchmarking system that can be used
independently of the training loop for model performance assessment.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

from config.evaluation.evaluation_config import EvaluationConfig
from core.evaluation.evaluator import Evaluator
from core.evaluation import EvaluationResult
from core.model_manager import ModelManager
from agent.ppo_agent import PPOTrainer
from envs import TradingEnvironment
from data.data_manager import DataManager


class BenchmarkResult:
    """Extended result for benchmark runs with additional metadata."""
    
    def __init__(
        self,
        evaluation_result: EvaluationResult,
        model_path: str,
        benchmark_config: Dict[str, Any],
        duration_seconds: float
    ):
        self.evaluation_result = evaluation_result
        self.model_path = model_path
        self.benchmark_config = benchmark_config
        self.duration_seconds = duration_seconds
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_path": self.model_path,
            "duration_seconds": self.duration_seconds,
            "benchmark_config": self.benchmark_config,
            "evaluation": {
                "mean_reward": self.evaluation_result.mean_reward,
                "std_reward": self.evaluation_result.std_reward,
                "min_reward": self.evaluation_result.min_reward,
                "max_reward": self.evaluation_result.max_reward,
                "total_episodes": self.evaluation_result.total_episodes,
                "episodes": [
                    {"episode_num": ep.episode_num, "reward": ep.reward}
                    for ep in self.evaluation_result.episodes
                ]
            }
        }


class BenchmarkRunner:
    """
    Standalone benchmark runner for model performance assessment.
    
    Can be used to:
    - Benchmark specific saved models
    - Compare multiple models
    - Run extensive evaluation suites
    - Generate benchmark reports
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize benchmark runner.
        
        Args:
            config: Evaluation configuration for benchmarking
        """
        self.config = config
        self.evaluator = Evaluator(config)
        self.logger = logging.getLogger(f"{__name__}.BenchmarkRunner")
        
        self.logger.info(f"🎯 BenchmarkRunner initialized")
    
    def benchmark_model(
        self,
        model_path: str,
        trainer: PPOTrainer,
        environment: TradingEnvironment,
        data_manager: DataManager,
        output_dir: Optional[str] = None
    ) -> Optional[BenchmarkResult]:
        """
        Run benchmark on a specific model.
        
        Args:
            model_path: Path to saved model to benchmark
            trainer: PPO trainer (model will be loaded into this)
            environment: Trading environment for evaluation
            data_manager: Data manager for episode selection
            output_dir: Optional directory to save detailed results
            
        Returns:
            BenchmarkResult with evaluation metrics and metadata
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🎯 Starting benchmark for model: {model_path}")
            
            # Load the model
            if not self._load_model(model_path, trainer):
                return None
            
            # Run evaluation
            evaluation_result = self.evaluator.evaluate_model(
                trainer=trainer,
                environment=environment,
                data_manager=data_manager,
                episode_manager=None  # Standalone mode
            )
            
            if not evaluation_result:
                self.logger.error(f"Evaluation failed for model: {model_path}")
                return None
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                evaluation_result=evaluation_result,
                model_path=model_path,
                benchmark_config=self._get_benchmark_config(),
                duration_seconds=duration
            )
            
            # Save results if output directory specified
            if output_dir:
                self._save_benchmark_result(benchmark_result, output_dir)
            
            self.logger.info(
                f"✅ Benchmark complete for {model_path}: "
                f"mean_reward={evaluation_result.mean_reward:.4f}, "
                f"duration={duration:.1f}s"
            )
            
            return benchmark_result
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark failed for {model_path}: {e}", exc_info=True)
            return None
    
    def benchmark_best_model(
        self,
        model_manager: ModelManager,
        trainer: PPOTrainer,
        environment: TradingEnvironment,
        data_manager: DataManager,
        output_dir: Optional[str] = None
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark the best available model.
        
        Args:
            model_manager: Model manager to find best model
            trainer: PPO trainer for model loading
            environment: Trading environment
            data_manager: Data manager
            output_dir: Optional output directory
            
        Returns:
            BenchmarkResult for the best model
        """
        try:
            # Find best model
            best_model_info = model_manager.find_best_model()
            if not best_model_info:
                self.logger.error("No best model found")
                return None
            
            model_path = best_model_info["path"]
            self.logger.info(f"Found best model: {model_path}")
            
            return self.benchmark_model(
                model_path=model_path,
                trainer=trainer,
                environment=environment,
                data_manager=data_manager,
                output_dir=output_dir
            )
            
        except Exception as e:
            self.logger.error(f"❌ Best model benchmark failed: {e}", exc_info=True)
            return None
    
    def compare_models(
        self,
        model_paths: List[str],
        trainer: PPOTrainer,
        environment: TradingEnvironment,
        data_manager: DataManager,
        output_dir: Optional[str] = None
    ) -> List[BenchmarkResult]:
        """
        Compare multiple models using the same evaluation setup.
        
        Args:
            model_paths: List of model paths to compare
            trainer: PPO trainer for model loading
            environment: Trading environment
            data_manager: Data manager
            output_dir: Optional output directory
            
        Returns:
            List of BenchmarkResults for comparison
        """
        results = []
        
        self.logger.info(f"🎯 Comparing {len(model_paths)} models")
        
        for i, model_path in enumerate(model_paths):
            self.logger.info(f"Benchmarking model {i+1}/{len(model_paths)}: {model_path}")
            
            result = self.benchmark_model(
                model_path=model_path,
                trainer=trainer,
                environment=environment,
                data_manager=data_manager,
                output_dir=output_dir
            )
            
            if result:
                results.append(result)
        
        # Log comparison summary
        if results:
            self._log_comparison_summary(results)
            
            # Save comparison report
            if output_dir:
                self._save_comparison_report(results, output_dir)
        
        return results
    
    def _load_model(self, model_path: str, trainer: PPOTrainer) -> bool:
        """Load model from path into trainer."""
        try:
            # This would use ModelManager's load_model method
            # For now, simplified loading
            model_manager = ModelManager()
            model, model_state = model_manager.load_model(
                model=trainer.model,
                optimizer=trainer.optimizer,
                model_path=model_path
            )
            
            self.logger.debug(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def _get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmark configuration for metadata."""
        return {
            "evaluation_config": {
                "episodes": self.config.episodes,
                "seed": self.config.seed,
                "deterministic_actions": self.config.deterministic_actions,
                "episode_selection": self.config.episode_selection
            },
            "benchmark_timestamp": datetime.now().isoformat()
        }
    
    def _save_benchmark_result(self, result: BenchmarkResult, output_dir: str) -> None:
        """Save benchmark result to file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            model_name = Path(result.model_path).stem
            filename = f"benchmark_{model_name}_{timestamp}.json"
            
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            self.logger.info(f"Benchmark result saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save benchmark result: {e}")
    
    def _save_comparison_report(self, results: List[BenchmarkResult], output_dir: str) -> None:
        """Save model comparison report."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
            filepath = output_path / filename
            
            comparison_data = {
                "comparison_timestamp": datetime.now().isoformat(),
                "total_models": len(results),
                "models": [result.to_dict() for result in results],
                "summary": {
                    "best_model": max(results, key=lambda r: r.evaluation_result.mean_reward).model_path,
                    "best_reward": max(r.evaluation_result.mean_reward for r in results),
                    "worst_reward": min(r.evaluation_result.mean_reward for r in results),
                    "mean_reward": sum(r.evaluation_result.mean_reward for r in results) / len(results)
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            
            self.logger.info(f"Comparison report saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save comparison report: {e}")
    
    def _log_comparison_summary(self, results: List[BenchmarkResult]) -> None:
        """Log summary of model comparison."""
        if not results:
            return
        
        self.logger.info("📊 Model Comparison Summary:")
        
        # Sort by reward descending
        sorted_results = sorted(results, key=lambda r: r.evaluation_result.mean_reward, reverse=True)
        
        for i, result in enumerate(sorted_results):
            rank = i + 1
            model_name = Path(result.model_path).stem
            reward = result.evaluation_result.mean_reward
            std = result.evaluation_result.std_reward
            
            self.logger.info(f"  {rank:2d}. {model_name}: {reward:.4f} ± {std:.4f}")
        
        best_result = sorted_results[0]
        self.logger.info(f"🏆 Best model: {Path(best_result.model_path).stem} ({best_result.evaluation_result.mean_reward:.4f})")