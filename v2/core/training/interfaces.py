"""
Training interfaces for different training modes and workflows.

These interfaces enable flexible training strategies including
standard training, continuous learning, hyperparameter optimization,
and benchmarking.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable, Callable
from datetime import datetime
from pathlib import Path
import pandas as pd

from ..types.common import (
    RunMode, ModelVersion, EpisodeMetrics, Symbol,
    Configurable, Resettable, Serializable
)
from ..agent.interfaces import ITrainableAgent, IAgentCallback
from ..environment.interfaces import ITradingEnvironment


@runtime_checkable
class ITrainingMode(Protocol):
    """Base interface for all training modes.
    
    Design principles:
    - Each mode encapsulates a complete workflow
    - Modes are composable and reusable
    - Clear lifecycle management
    - Support for interruption and resumption
    """
    
    @property
    def mode_type(self) -> RunMode:
        """Type of training mode.
        
        Returns:
            Mode identifier
        """
        ...
    
    @property
    def is_running(self) -> bool:
        """Whether mode is currently active.
        
        Returns:
            True if running
        """
        ...
    
    def initialize(
        self,
        agent: ITrainableAgent,
        environment: ITradingEnvironment,
        config: dict[str, Any]
    ) -> None:
        """Initialize mode with components.
        
        Args:
            agent: Training agent
            environment: Trading environment
            config: Mode configuration
            
        Design notes:
        - Validate configuration
        - Set up mode-specific resources
        - Register callbacks
        """
        ...
    
    def run(
        self,
        callbacks: Optional[list[IAgentCallback]] = None
    ) -> dict[str, Any]:
        """Execute training mode.
        
        Args:
            callbacks: Optional callbacks
            
        Returns:
            Mode results/metrics
            
        Design notes:
        - Main execution loop
        - Handle interruptions gracefully
        - Return comprehensive results
        """
        ...
    
    def pause(self) -> None:
        """Pause execution.
        
        Design notes:
        - Save state for resumption
        - Release resources if needed
        """
        ...
    
    def resume(self) -> None:
        """Resume execution.
        
        Design notes:
        - Restore state
        - Continue from pause point
        """
        ...
    
    def stop(self) -> None:
        """Stop execution and cleanup.
        
        Design notes:
        - Final cleanup
        - Save final state
        """
        ...


class IStandardTrainingMode(ITrainingMode):
    """Interface for standard RL training mode.
    
    Design principles:
    - Traditional episode-based training
    - Fixed number of episodes/steps
    - Single model output
    """
    
    @abstractmethod
    def set_training_schedule(
        self,
        total_episodes: Optional[int] = None,
        total_steps: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> None:
        """Set training duration.
        
        Args:
            total_episodes: Number of episodes
            total_steps: Number of environment steps
            time_limit: Time limit in hours
            
        Design notes:
        - At least one limit required
        - Support multiple stopping criteria
        """
        ...
    
    @abstractmethod
    def get_training_progress(self) -> dict[str, float]:
        """Get current training progress.
        
        Returns:
            Dict with:
            - episodes_completed
            - steps_completed
            - time_elapsed
            - estimated_time_remaining
        """
        ...


class IContinuousTrainingMode(ITrainingMode):
    """Interface for continuous improvement mode.
    
    Design principles:
    - Never-ending improvement loop
    - Automatic model versioning
    - Performance-based model selection
    - Curriculum learning support
    """
    
    @abstractmethod
    def set_improvement_criteria(
        self,
        metric: str = "average_reward",
        improvement_threshold: float = 0.01,
        patience: int = 10
    ) -> None:
        """Set criteria for model improvement.
        
        Args:
            metric: Metric to optimize
            improvement_threshold: Minimum improvement
            patience: Episodes without improvement
            
        Design notes:
        - Define what constitutes progress
        - Balance exploration vs exploitation
        """
        ...
    
    @abstractmethod
    def set_curriculum(
        self,
        curriculum: list[dict[str, Any]]
    ) -> None:
        """Set training curriculum.
        
        Args:
            curriculum: List of training stages
            
        Design notes:
        - Each stage has symbols and difficulty
        - Progress based on performance
        """
        ...
    
    @abstractmethod
    def get_model_history(self) -> pd.DataFrame:
        """Get history of model versions.
        
        Returns:
            DataFrame with model version info
        """
        ...


class IOptunaMode(ITrainingMode):
    """Interface for hyperparameter optimization mode.
    
    Design principles:
    - Systematic hyperparameter search
    - Multiple optimization algorithms
    - Parallel trial execution
    - Early stopping of bad trials
    """
    
    @abstractmethod
    def set_search_space(
        self,
        parameter_specs: dict[str, dict[str, Any]]
    ) -> None:
        """Define hyperparameter search space.
        
        Args:
            parameter_specs: Dict mapping param name to spec:
                - type: "float", "int", "categorical"
                - low/high: For numeric types
                - choices: For categorical
                - log: Whether to use log scale
                
        Design notes:
        - Support nested parameters
        - Enable conditional parameters
        """
        ...
    
    @abstractmethod
    def set_optimization_config(
        self,
        n_trials: int,
        n_jobs: int = 1,
        sampler: str = "TPE",
        pruner: Optional[str] = "MedianPruner"
    ) -> None:
        """Configure optimization process.
        
        Args:
            n_trials: Number of trials
            n_jobs: Parallel jobs
            sampler: Sampling algorithm
            pruner: Pruning algorithm
            
        Design notes:
        - Balance exploration/exploitation
        - Consider resource constraints
        """
        ...
    
    @abstractmethod
    def get_best_params(self) -> dict[str, Any]:
        """Get best parameters found.
        
        Returns:
            Best parameter configuration
        """
        ...
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """Get trial history.
        
        Returns:
            DataFrame with trial results
        """
        ...


class IBenchmarkMode(ITrainingMode):
    """Interface for benchmarking mode.
    
    Design principles:
    - Standardized performance evaluation
    - Multiple evaluation metrics
    - Statistical significance testing
    - Comparison across models/configs
    """
    
    @abstractmethod
    def set_benchmark_suite(
        self,
        test_episodes: list[dict[str, Any]]
    ) -> None:
        """Define benchmark test suite.
        
        Args:
            test_episodes: List of test configurations:
                - symbol
                - date
                - reset_points
                - expected_difficulty
                
        Design notes:
        - Cover diverse scenarios
        - Include edge cases
        """
        ...
    
    @abstractmethod
    def add_baseline(
        self,
        name: str,
        model_path: Path
    ) -> None:
        """Add baseline model for comparison.
        
        Args:
            name: Baseline name
            model_path: Path to model
        """
        ...
    
    @abstractmethod
    def get_benchmark_results(self) -> pd.DataFrame:
        """Get comprehensive benchmark results.
        
        Returns:
            DataFrame with detailed metrics
        """
        ...
    
    @abstractmethod
    def generate_report(
        self,
        output_path: Path
    ) -> None:
        """Generate benchmark report.
        
        Args:
            output_path: Where to save report
            
        Design notes:
        - Include visualizations
        - Statistical analysis
        - Recommendations
        """
        ...


class ITrainingManager(Configurable):
    """High-level training orchestration interface.
    
    Design principles:
    - Coordinate different training modes
    - Handle mode transitions
    - Manage resources and scheduling
    - Provide unified interface
    """
    
    @abstractmethod
    def register_mode(
        self,
        mode: ITrainingMode
    ) -> None:
        """Register a training mode.
        
        Args:
            mode: Training mode instance
        """
        ...
    
    @abstractmethod
    def start_mode(
        self,
        mode_type: RunMode,
        config: dict[str, Any],
        background: bool = False
    ) -> None:
        """Start a training mode.
        
        Args:
            mode_type: Type of mode
            config: Mode configuration
            background: Run in background
            
        Design notes:
        - Handle mode conflicts
        - Set up monitoring
        """
        ...
    
    @abstractmethod
    def switch_mode(
        self,
        to_mode: RunMode,
        save_state: bool = True
    ) -> None:
        """Switch between modes.
        
        Args:
            to_mode: Target mode
            save_state: Save current state
            
        Design notes:
        - Graceful transitions
        - Preserve progress
        """
        ...
    
    @abstractmethod
    def get_active_modes(self) -> list[RunMode]:
        """Get currently active modes.
        
        Returns:
            List of active mode types
        """
        ...
    
    @abstractmethod
    def schedule_mode_sequence(
        self,
        sequence: list[tuple[RunMode, dict[str, Any]]]
    ) -> None:
        """Schedule sequence of modes.
        
        Args:
            sequence: List of (mode, config) tuples
            
        Design notes:
        - Enable workflows
        - Handle failures
        """
        ...


class ITrainingMonitor(Protocol):
    """Interface for training monitoring.
    
    Design principles:
    - Real-time monitoring
    - Multiple visualization backends
    - Alerting capabilities
    """
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        mode: RunMode
    ) -> None:
        """Log training metrics.
        
        Args:
            metrics: Metric values
            step: Current step
            mode: Active mode
        """
        ...
    
    def log_episode(
        self,
        episode_metrics: EpisodeMetrics,
        mode: RunMode
    ) -> None:
        """Log episode results.
        
        Args:
            episode_metrics: Episode performance
            mode: Active mode
        """
        ...
    
    def set_alert(
        self,
        condition: Callable[[dict[str, float]], bool],
        message: str
    ) -> None:
        """Set metric alert.
        
        Args:
            condition: Alert condition
            message: Alert message
        """
        ...
    
    def get_summary(
        self,
        mode: RunMode,
        last_n: Optional[int] = None
    ) -> dict[str, Any]:
        """Get training summary.
        
        Args:
            mode: Training mode
            last_n: Last N episodes
            
        Returns:
            Summary statistics
        """
        ...
