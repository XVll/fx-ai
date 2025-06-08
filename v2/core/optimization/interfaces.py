"""
Optimization interfaces for hyperparameter tuning.

Defines contracts for hyperparameter optimization systems
including Optuna integration and custom search strategies.
"""

from typing import Protocol, Optional, Any, Callable, runtime_checkable
from datetime import datetime
import pandas as pd
from abc import abstractmethod
from enum import Enum

from ..types import ObservationArray, ActionArray, Metrics


class OptimizationState(Enum):
    """Optimization study state."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED" 
    PRUNED = "PRUNED"
    FAILED = "FAILED"
    WAITING = "WAITING"


class TrialState(Enum):
    """Individual trial state."""
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"


@runtime_checkable
class IHyperparameterSpace(Protocol):
    """Defines hyperparameter search space.
    
    Implementation requirements:
    - Define parameter ranges and types
    - Support conditional parameters
    - Validate parameter combinations
    - Provide sampling methods
    """
    
    @abstractmethod
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> float:
        """Suggest float parameter.
        
        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            step: Discretization step
            log: Use log scale
            
        Returns:
            Suggested value
        """
        ...
    
    @abstractmethod
    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False
    ) -> int:
        """Suggest integer parameter.
        
        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            step: Step size
            log: Use log scale
            
        Returns:
            Suggested value
        """
        ...
    
    @abstractmethod
    def suggest_categorical(
        self,
        name: str,
        choices: list[Any]
    ) -> Any:
        """Suggest categorical parameter.
        
        Args:
            name: Parameter name
            choices: List of choices
            
        Returns:
            Selected choice
        """
        ...
    
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get all suggested parameters.
        
        Returns:
            Dictionary of parameters
        """
        ...


@runtime_checkable
class ITrial(Protocol):
    """Individual optimization trial.
    
    Implementation requirements:
    - Track trial progress
    - Support intermediate reporting
    - Handle pruning decisions
    - Store trial metadata
    """
    
    @property
    @abstractmethod
    def number(self) -> int:
        """Trial number."""
        ...
    
    @property
    @abstractmethod
    def state(self) -> TrialState:
        """Current trial state."""
        ...
    
    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Trial parameters."""
        ...
    
    @abstractmethod
    def report(
        self,
        value: float,
        step: int
    ) -> None:
        """Report intermediate value.
        
        Args:
            value: Objective value
            step: Current step
        """
        ...
    
    @abstractmethod
    def should_prune(self) -> bool:
        """Check if trial should be pruned.
        
        Returns:
            True if should prune
        """
        ...
    
    @abstractmethod
    def set_user_attr(
        self,
        key: str,
        value: Any
    ) -> None:
        """Set user attribute.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        ...


@runtime_checkable
class IObjective(Protocol):
    """Optimization objective function.
    
    Implementation requirements:
    - Define objective computation
    - Support multi-objective optimization
    - Handle constraint violations
    - Report intermediate results
    """
    
    @abstractmethod
    def __call__(
        self,
        trial: ITrial
    ) -> float:
        """Evaluate objective.
        
        Args:
            trial: Current trial
            
        Returns:
            Objective value (minimize)
        """
        ...
    
    @abstractmethod
    def get_constraints(self) -> Optional[list[Callable]]:
        """Get constraint functions.
        
        Returns:
            List of constraint functions or None
        """
        ...


@runtime_checkable
class ISampler(Protocol):
    """Hyperparameter sampling strategy.
    
    Implementation requirements:
    - Define sampling algorithm
    - Support warm starting
    - Handle categorical variables
    - Implement acquisition functions
    """
    
    @abstractmethod
    def sample_independent(
        self,
        study: 'IStudy',
        trial: ITrial,
        param_name: str,
        param_distribution: Any
    ) -> Any:
        """Sample single parameter.
        
        Args:
            study: Optimization study
            trial: Current trial
            param_name: Parameter name
            param_distribution: Parameter distribution
            
        Returns:
            Sampled value
        """
        ...
    
    @abstractmethod
    def sample_relative(
        self,
        study: 'IStudy',
        trial: ITrial,
        search_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample relative parameters.
        
        Args:
            study: Optimization study
            trial: Current trial
            search_space: Parameter search space
            
        Returns:
            Sampled parameters
        """
        ...


@runtime_checkable
class IPruner(Protocol):
    """Trial pruning strategy.
    
    Implementation requirements:
    - Define pruning criteria
    - Support custom pruning logic
    - Handle multi-objective pruning
    - Maintain pruning statistics
    """
    
    @abstractmethod
    def prune(
        self,
        study: 'IStudy',
        trial: ITrial
    ) -> bool:
        """Decide whether to prune trial.
        
        Args:
            study: Optimization study
            trial: Current trial
            
        Returns:
            True if should prune
        """
        ...


@runtime_checkable
class IStudy(Protocol):
    """Optimization study.
    
    Implementation requirements:
    - Manage trial execution
    - Track optimization history
    - Support parallel execution
    - Provide result analysis
    """
    
    @property
    @abstractmethod
    def study_name(self) -> str:
        """Study name."""
        ...
    
    @property
    @abstractmethod
    def direction(self) -> str:
        """Optimization direction (minimize/maximize)."""
        ...
    
    @property
    @abstractmethod
    def best_trial(self) -> ITrial:
        """Best trial so far."""
        ...
    
    @property
    @abstractmethod
    def best_value(self) -> float:
        """Best objective value."""
        ...
    
    @property
    @abstractmethod
    def best_params(self) -> dict[str, Any]:
        """Best parameters."""
        ...
    
    @abstractmethod
    def optimize(
        self,
        func: IObjective,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        callbacks: Optional[list[Callable]] = None
    ) -> None:
        """Run optimization.
        
        Args:
            func: Objective function
            n_trials: Number of trials
            timeout: Time limit in seconds
            n_jobs: Parallel jobs
            callbacks: Study callbacks
        """
        ...
    
    @abstractmethod
    def get_trials(
        self,
        states: Optional[list[TrialState]] = None
    ) -> list[ITrial]:
        """Get trials.
        
        Args:
            states: Filter by states
            
        Returns:
            List of trials
        """
        ...
    
    @abstractmethod
    def trials_dataframe(
        self,
        attrs: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Get trials as DataFrame.
        
        Args:
            attrs: Attributes to include
            
        Returns:
            Trials DataFrame
        """
        ...


@runtime_checkable
class IOptunaIntegration(Protocol):
    """Optuna integration for hyperparameter optimization.
    
    Implementation requirements:
    - Create and manage studies
    - Define search spaces
    - Implement objectives
    - Handle result persistence
    """
    
    @abstractmethod
    def create_study(
        self,
        study_name: str,
        direction: str = "minimize",
        sampler: Optional[ISampler] = None,
        pruner: Optional[IPruner] = None,
        storage: Optional[str] = None
    ) -> IStudy:
        """Create optimization study.
        
        Args:
            study_name: Study identifier
            direction: minimize or maximize
            sampler: Sampling algorithm
            pruner: Pruning algorithm
            storage: Storage backend
            
        Returns:
            Created study
        """
        ...
    
    @abstractmethod
    def load_study(
        self,
        study_name: str,
        storage: str
    ) -> IStudy:
        """Load existing study.
        
        Args:
            study_name: Study identifier
            storage: Storage backend
            
        Returns:
            Loaded study
        """
        ...
    
    @abstractmethod
    def delete_study(
        self,
        study_name: str,
        storage: str
    ) -> None:
        """Delete study.
        
        Args:
            study_name: Study identifier
            storage: Storage backend
        """
        ...
    
    @abstractmethod
    def get_all_study_names(
        self,
        storage: str
    ) -> list[str]:
        """Get all study names.
        
        Args:
            storage: Storage backend
            
        Returns:
            List of study names
        """
        ...
    
    @abstractmethod
    def visualize_optimization_history(
        self,
        study: IStudy,
        target: Optional[Callable] = None
    ) -> Any:
        """Visualize optimization history.
        
        Args:
            study: Study to visualize
            target: Custom target function
            
        Returns:
            Visualization object
        """
        ...
    
    @abstractmethod
    def visualize_param_importances(
        self,
        study: IStudy,
        evaluator: Optional[Any] = None
    ) -> Any:
        """Visualize parameter importances.
        
        Args:
            study: Study to analyze
            evaluator: Importance evaluator
            
        Returns:
            Visualization object
        """
        ...


@runtime_checkable
class IHyperparameterOptimizer(Protocol):
    """High-level hyperparameter optimizer.
    
    Implementation requirements:
    - Orchestrate optimization workflow
    - Handle multiple objectives
    - Support distributed optimization
    - Provide result analysis
    """
    
    @abstractmethod
    def optimize_agent(
        self,
        agent_type: str,
        env_config: dict[str, Any],
        n_trials: int,
        n_episodes_per_trial: int,
        optimization_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimize agent hyperparameters.
        
        Args:
            agent_type: Type of agent
            env_config: Environment configuration
            n_trials: Number of trials
            n_episodes_per_trial: Episodes per trial
            optimization_config: Optimization settings
            
        Returns:
            Best hyperparameters
        """
        ...
    
    @abstractmethod
    def optimize_reward_system(
        self,
        base_config: dict[str, Any],
        n_trials: int,
        evaluation_episodes: int
    ) -> dict[str, Any]:
        """Optimize reward system.
        
        Args:
            base_config: Base configuration
            n_trials: Number of trials
            evaluation_episodes: Episodes for evaluation
            
        Returns:
            Optimized reward configuration
        """
        ...
    
    @abstractmethod
    def run_parallel_optimization(
        self,
        objectives: list[IObjective],
        n_trials: int,
        n_jobs: int,
        study_name: str
    ) -> pd.DataFrame:
        """Run parallel optimization.
        
        Args:
            objectives: List of objectives
            n_trials: Trials per objective
            n_jobs: Parallel jobs
            study_name: Base study name
            
        Returns:
            Results DataFrame
        """
        ...
    
    @abstractmethod
    def analyze_results(
        self,
        study: IStudy,
        top_k: int = 10
    ) -> dict[str, Any]:
        """Analyze optimization results.
        
        Args:
            study: Completed study
            top_k: Number of top trials
            
        Returns:
            Analysis results
        """
        ...