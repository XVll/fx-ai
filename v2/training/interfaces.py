"""
Unified Training Interfaces for Mode-Based Training System.

This module provides comprehensive interfaces for different training modes:
- Standard RL training with fixed schedules
- Continuous training with model versioning and curriculum learning  
- Hyperparameter optimization with Optuna integration
- Benchmarking and performance evaluation

Design principles:
- Each mode is a complete, self-contained workflow
- Modes are configurable, composable, and interruptible
- Clear separation of concerns between mode logic and training orchestration
- Support for mode transitions and workflow scheduling
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable, Callable, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import pandas as pd

from ..core.common import (
    RunMode, ModelVersion, EpisodeMetrics, Symbol, TerminationReason,
    Configurable, Resettable, Serializable
)


class TrainingPhase(Enum):
    """Training execution phases."""
    INITIALIZATION = "INITIALIZATION"
    WARMUP = "WARMUP"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    CHECKPOINT = "CHECKPOINT"
    TERMINATION = "TERMINATION"


class ModeState(Enum):
    """Training mode states."""
    INACTIVE = "INACTIVE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@runtime_checkable
class ITrainingMode(Protocol):
    """Base interface for all training modes.
    
    Implementation Guide:
    - Store mode_type as class attribute
    - Track state changes through ModeState enum
    - Implement proper resource cleanup in stop()
    - Support interruption at any phase
    - Return comprehensive metrics from run()
    
    Interaction Pattern:
    1. TrainingManager calls initialize() with components
    2. TrainingManager calls run() which executes until completion
    3. Mode can be paused/resumed during execution
    4. Mode must handle termination gracefully
    """
    
    @property
    def mode_type(self) -> RunMode:
        """Type of training mode (immutable)."""
        ...
    
    @property
    def current_state(self) -> ModeState:
        """Current execution state."""
        ...
    
    @property
    def current_phase(self) -> TrainingPhase:
        """Current execution phase within the mode."""
        ...
    
    @property
    def is_running(self) -> bool:
        """Whether mode is actively executing."""
        ...
    
    def initialize(
        self,
        trainer: Any,  # Will be the trainer/agent object
        environment: Any,  # Trading environment
        config: Dict[str, Any]
    ) -> None:
        """Initialize mode with required components.
        
        Implementation Guide:
        - Validate all required config parameters
        - Set up mode-specific resources (loggers, callbacks, etc.)
        - Initialize termination criteria
        - Register any required event handlers
        - Store references to trainer and environment
        - Set state to INITIALIZING, then INACTIVE when ready
        
        Args:
            trainer: Training agent/trainer instance
            environment: Trading environment instance  
            config: Mode-specific configuration
        """
        ...
    
    def run(
        self,
        callbacks: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Execute the training mode workflow.
        
        Implementation Guide:
        - Set state to RUNNING
        - Execute mode-specific training loop
        - Handle interruptions gracefully (check self.current_state)
        - Update current_phase as execution progresses
        - Call callbacks at appropriate points
        - Return comprehensive results including:
          * Final performance metrics
          * Training statistics
          * Resource usage
          * Termination reason
        
        Args:
            callbacks: Optional callback functions
            
        Returns:
            Dictionary with mode execution results
        """
        ...
    
    def pause(self) -> None:
        """Pause execution while preserving state.
        
        Implementation Guide:
        - Set state to PAUSED
        - Save current progress to enable resumption
        - Release non-essential resources
        - Ensure safe checkpoint is created
        """
        ...
    
    def resume(self) -> None:
        """Resume execution from paused state.
        
        Implementation Guide:
        - Restore state from PAUSED to RUNNING
        - Reload any released resources
        - Continue from last checkpoint
        """
        ...
    
    def stop(self) -> None:
        """Terminate execution and cleanup resources.
        
        Implementation Guide:
        - Set state to STOPPING, then COMPLETED
        - Save final checkpoint if needed
        - Clean up all resources (files, connections, etc.)
        - Finalize any pending operations
        """
        ...
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current execution progress.
        
        Returns:
            Progress information including completion percentage,
            current metrics, time elapsed, etc.
        """
        ...


class IStandardTrainingMode(ITrainingMode):
    """Standard reinforcement learning training mode.
    
    Implementation Guide:
    - Traditional episode-based training loop
    - Fixed termination criteria (episodes, steps, or time)
    - Single model output with checkpoints
    - Progress tracking and logging
    - Support for evaluation episodes during training
    
    Use Cases:
    - Initial model training
    - Baseline model development
    - Simple training scenarios
    - Development and testing
    """
    
    @abstractmethod
    def set_training_schedule(
        self,
        total_episodes: Optional[int] = None,
        total_steps: Optional[int] = None,
        time_limit: Optional[float] = None,
        evaluation_frequency: int = 100
    ) -> None:
        """Configure training duration and evaluation schedule.
        
        Implementation Guide:
        - At least one termination criterion must be specified
        - Store criteria for use in training loop
        - Set up evaluation scheduling
        - Configure progress tracking intervals
        
        Args:
            total_episodes: Maximum episodes to train
            total_steps: Maximum environment steps
            time_limit: Maximum training time in hours
            evaluation_frequency: Episodes between evaluations
        """
        ...
    
    @abstractmethod
    def get_training_progress(self) -> Dict[str, float]:
        """Get detailed training progress information.
        
        Implementation Guide:
        - Calculate completion percentage for each criterion
        - Estimate remaining time based on current rate
        - Include current performance metrics
        
        Returns:
            Dictionary containing:
            - episodes_completed: Current episode count
            - steps_completed: Current step count  
            - time_elapsed: Hours of training
            - completion_percentage: Overall progress (0-1)
            - estimated_time_remaining: Hours estimated
            - current_performance: Latest metrics
        """
        ...


class IContinuousTrainingMode(ITrainingMode):
    """Continuous improvement training mode with model versioning.
    
    Implementation Guide:
    - Never-ending training loop with intelligent termination
    - Automatic model versioning (v1, v2, v3...)
    - Curriculum learning with progressive difficulty
    - Performance-based model selection and rollback
    - Adaptive learning rate and hyperparameter adjustment
    
    Use Cases:
    - Production model improvement
    - Long-term model evolution
    - Curriculum-based training
    - Adaptive learning scenarios
    
    Interaction with DataLifecycleManager:
    - Requests next training data configuration
    - Adapts to data quality and availability
    - Handles data exhaustion gracefully
    """
    
    @abstractmethod
    def set_improvement_criteria(
        self,
        metric: str = "average_reward",
        improvement_threshold: float = 0.01,
        patience: int = 10,
        rolling_window: int = 20
    ) -> None:
        """Configure what constitutes model improvement.
        
        Implementation Guide:
        - Track specified metric over rolling window
        - Define minimum improvement to create new version
        - Set patience for reverting to previous version
        - Consider multiple metrics for robust evaluation
        
        Args:
            metric: Primary optimization metric
            improvement_threshold: Minimum relative improvement
            patience: Episodes without improvement before rollback
            rolling_window: Episodes to average for evaluation
        """
        ...
    
    @abstractmethod
    def set_curriculum_config(
        self,
        enable_curriculum: bool = True,
        difficulty_progression: str = "adaptive",
        stage_requirements: Optional[Dict[str, Any]] = None
    ) -> None:
        """Configure curriculum learning behavior.
        
        Implementation Guide:
        - Enable/disable curriculum-based training
        - Set difficulty progression strategy (linear, adaptive, manual)
        - Define requirements for advancing curriculum stages
        - Integrate with DataLifecycleManager for data selection
        
        Args:
            enable_curriculum: Whether to use curriculum learning
            difficulty_progression: How to advance difficulty
            stage_requirements: Custom stage advancement criteria
        """
        ...
    
    @abstractmethod
    def get_model_history(self) -> pd.DataFrame:
        """Get complete history of model versions.
        
        Implementation Guide:
        - Track all model versions with metadata
        - Include performance metrics for each version
        - Show version transitions and rollbacks
        - Enable model comparison and analysis
        
        Returns:
            DataFrame with columns:
            - version: Model version (v1, v2, etc.)
            - timestamp: Creation time
            - episodes_trained: Episodes for this version
            - performance_metrics: Key metrics
            - config_changes: Configuration differences
            - is_active: Whether currently in use
        """
        ...
    
    @abstractmethod
    def should_create_checkpoint(self) -> bool:
        """Determine if new model checkpoint should be created.
        
        Implementation Guide:
        - Evaluate current performance vs best
        - Consider improvement criteria
        - Account for training stability
        - Prevent excessive checkpointing
        
        Returns:
            True if new checkpoint should be created
        """
        ...


class IOptunaMode(ITrainingMode):
    """Hyperparameter optimization training mode using Optuna.
    
    Implementation Guide:
    - Systematic search through hyperparameter space
    - Multiple optimization algorithms (TPE, CMA-ES, Random)
    - Parallel trial execution support
    - Early stopping of poor-performing trials
    - Integration with Optuna study management
    
    Use Cases:
    - Model hyperparameter tuning
    - Architecture search
    - Training configuration optimization
    - Reward system parameter tuning
    
    Deterministic Setup:
    - Fixed random seeds for reproducibility
    - Standardized evaluation methodology
    - Consistent data splits and episodes
    """
    
    @abstractmethod
    def set_search_space(
        self,
        parameter_specs: Dict[str, Dict[str, Any]]
    ) -> None:
        """Define the hyperparameter search space.
        
        Implementation Guide:
        - Support all Optuna parameter types
        - Enable nested and conditional parameters
        - Validate parameter specifications
        - Create Optuna distributions for each parameter
        
        Args:
            parameter_specs: Dictionary mapping parameter names to specs:
                - type: "float", "int", "categorical", "uniform", "loguniform"
                - low/high: Bounds for numeric parameters
                - choices: Options for categorical parameters
                - log: Whether to use log scale for numeric
        """
        ...
    
    @abstractmethod
    def set_optimization_config(
        self,
        n_trials: int,
        n_jobs: int = 1,
        sampler: str = "TPE",
        pruner: Optional[str] = "MedianPruner",
        study_name: Optional[str] = None
    ) -> None:
        """Configure the optimization process.
        
        Implementation Guide:
        - Set up Optuna study with specified sampler
        - Configure pruning for early stopping
        - Enable parallel execution if n_jobs > 1
        - Set up study persistence and resumption
        
        Args:
            n_trials: Total number of trials to run
            n_jobs: Number of parallel processes
            sampler: Optuna sampler algorithm
            pruner: Optuna pruner for early stopping
            study_name: Name for study persistence
        """
        ...
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameter configuration found.
        
        Implementation Guide:
        - Return parameters from best trial
        - Include confidence intervals if available
        - Provide parameter importance rankings
        
        Returns:
            Best parameter configuration
        """
        ...
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """Get complete trial history and results.
        
        Implementation Guide:
        - Include all trial parameters and results
        - Show pruning decisions and reasons
        - Calculate parameter correlations
        - Enable analysis and visualization
        
        Returns:
            DataFrame with trial information
        """
        ...


class IBenchmarkMode(ITrainingMode):
    """Benchmarking and evaluation training mode.
    
    Implementation Guide:
    - Standardized performance evaluation across models
    - Statistical significance testing
    - Multiple evaluation metrics and scenarios
    - Comparison with baseline models
    - Deterministic evaluation for reproducibility
    
    Use Cases:
    - Model performance evaluation
    - A/B testing between models
    - Production model validation
    - Research comparison studies
    
    Deterministic Setup:
    - Fixed evaluation episodes and data
    - Consistent random seeds
    - Standardized metrics calculation
    """
    
    @abstractmethod
    def set_benchmark_suite(
        self,
        test_episodes: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> None:
        """Define the benchmark test suite.
        
        Implementation Guide:
        - Create diverse, representative test scenarios
        - Include edge cases and stress tests
        - Ensure reproducible evaluation conditions
        - Define comprehensive metrics collection
        
        Args:
            test_episodes: List of test configurations with:
                - symbol: Trading symbol
                - date: Trading date
                - reset_points: Episode start times
                - expected_difficulty: Difficulty rating
            metrics: Custom metrics to collect
        """
        ...
    
    @abstractmethod
    def add_baseline(
        self,
        name: str,
        model_path: Path,
        description: Optional[str] = None
    ) -> None:
        """Add baseline model for comparison.
        
        Implementation Guide:
        - Load and validate baseline model
        - Store model metadata and description
        - Ensure compatibility with evaluation suite
        
        Args:
            name: Unique baseline identifier
            model_path: Path to baseline model
            description: Optional description
        """
        ...
    
    @abstractmethod
    def get_benchmark_results(self) -> pd.DataFrame:
        """Get comprehensive benchmark results.
        
        Implementation Guide:
        - Include all models and baselines
        - Calculate statistical significance
        - Provide confidence intervals
        - Show per-scenario breakdown
        
        Returns:
            DataFrame with detailed evaluation results
        """
        ...
    
    @abstractmethod
    def generate_report(
        self,
        output_path: Path,
        include_visualizations: bool = True
    ) -> None:
        """Generate comprehensive benchmark report.
        
        Implementation Guide:
        - Create detailed analysis report
        - Include statistical tests and visualizations
        - Provide actionable recommendations
        - Export in multiple formats (HTML, PDF)
        
        Args:
            output_path: Where to save the report
            include_visualizations: Whether to include charts
        """
        ...


class ITrainingManager(Configurable):
    """High-level training orchestration and mode management.
    
    Implementation Guide:
    - Central coordinator for all training activities
    - Manages mode lifecycles and transitions
    - Handles resource allocation and scheduling
    - Provides unified interface for different training workflows
    - Integrates with monitoring and logging systems
    
    Responsibilities:
    - Mode registration and lifecycle management
    - Configuration validation and distribution
    - Resource conflict resolution
    - Workflow scheduling and execution
    - Progress monitoring and reporting
    
    Integration Points:
    - DataLifecycleManager for data orchestration
    - ContinuousTraining for model management
    - Callback system for monitoring
    - Graceful shutdown management
    """
    
    @abstractmethod
    def register_mode(
        self,
        mode: ITrainingMode
    ) -> None:
        """Register a training mode for use.
        
        Implementation Guide:
        - Validate mode interface compliance
        - Check for mode type conflicts
        - Store mode reference for later use
        - Set up mode-specific monitoring
        
        Args:
            mode: Training mode instance to register
        """
        ...
    
    @abstractmethod
    def start_mode(
        self,
        mode_type: RunMode,
        config: Dict[str, Any],
        trainer: Any,
        environment: Any,
        background: bool = False
    ) -> Dict[str, Any]:
        """Start execution of a training mode.
        
        Implementation Guide:
        - Find registered mode for mode_type
        - Validate configuration
        - Initialize mode with components
        - Execute mode.run() synchronously or asynchronously
        - Handle mode failures and cleanup
        
        Args:
            mode_type: Type of mode to start
            config: Mode configuration
            trainer: Training agent/trainer
            environment: Trading environment
            background: Whether to run asynchronously
            
        Returns:
            Mode execution results
        """
        ...
    
    @abstractmethod
    def switch_mode(
        self,
        to_mode: RunMode,
        config: Dict[str, Any],
        save_state: bool = True
    ) -> None:
        """Switch from current mode to new mode.
        
        Implementation Guide:
        - Pause/stop current mode gracefully
        - Save state if requested
        - Initialize and start new mode
        - Handle transition failures
        
        Args:
            to_mode: Target mode type
            config: Configuration for new mode
            save_state: Whether to save current state
        """
        ...
    
    @abstractmethod
    def get_active_modes(self) -> List[RunMode]:
        """Get list of currently active modes.
        
        Implementation Guide:
        - Check state of all registered modes
        - Return list of modes in RUNNING state
        
        Returns:
            List of active mode types
        """
        ...
    
    @abstractmethod
    def schedule_mode_sequence(
        self,
        sequence: List[Tuple[RunMode, Dict[str, Any]]],
        on_failure: str = "stop"
    ) -> None:
        """Schedule a sequence of training modes.
        
        Implementation Guide:
        - Queue modes for sequential execution
        - Handle dependencies between modes
        - Manage failure scenarios (stop, continue, retry)
        - Provide progress tracking for sequence
        
        Args:
            sequence: List of (mode_type, config) tuples
            on_failure: Failure handling strategy
        """
        ...
    
    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status.
        
        Implementation Guide:
        - Collect status from all active modes
        - Include resource usage information
        - Provide progress estimates
        - Show recent performance metrics
        
        Returns:
            Status information including:
            - active_modes: Currently running modes
            - progress: Training progress by mode
            - metrics: Recent performance metrics
            - resource_usage: CPU, memory, GPU usage
            - time_remaining: Estimated completion times
        """
        ...
    
    @abstractmethod
    def request_termination(
        self,
        reason: TerminationReason,
        mode_type: Optional[RunMode] = None
    ) -> None:
        """Request termination of training modes.
        
        Implementation Guide:
        - Send termination signal to specified mode or all modes
        - Allow modes to finish current operations gracefully
        - Set termination reason for logging
        
        Args:
            reason: Reason for termination
            mode_type: Specific mode to terminate (None for all)
        """
        ...


class ITrainingMonitor(Protocol):
    """Training monitoring and metrics collection interface.
    
    Implementation Guide:
    - Real-time metrics collection and logging
    - Support multiple backends (W&B, TensorBoard, files)
    - Configurable alerting and notifications
    - Comprehensive visualization and reporting
    
    Integration:
    - Called by training modes during execution
    - Provides real-time dashboard updates
    - Exports data for analysis and reporting
    """
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        mode: RunMode,
        episode_id: Optional[str] = None
    ) -> None:
        """Log training metrics at specific step.
        
        Args:
            metrics: Metric name -> value mapping
            step: Current training step
            mode: Active training mode
            episode_id: Optional episode identifier
        """
        ...
    
    def log_episode(
        self,
        episode_metrics: EpisodeMetrics,
        mode: RunMode
    ) -> None:
        """Log complete episode results.
        
        Args:
            episode_metrics: Episode performance data
            mode: Active training mode
        """
        ...
    
    def set_alert(
        self,
        condition: Callable[[Dict[str, float]], bool],
        message: str,
        severity: str = "warning"
    ) -> None:
        """Configure metric-based alert.
        
        Args:
            condition: Function that returns True when alert should fire
            message: Alert message template
            severity: Alert severity level
        """
        ...
    
    def get_summary(
        self,
        mode: RunMode,
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get training summary statistics.
        
        Args:
            mode: Training mode to summarize
            last_n: Number of recent episodes to include
            
        Returns:
            Summary statistics and metrics
        """
        ...