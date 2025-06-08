"""
Improved Training interfaces with additional training modes and better separation.

Key improvements:
- Added missing training modes (Real-time, Evaluation, Transfer Learning, Multi-Asset)
- Clear curriculum management separation
- Better orchestration interfaces
- Mode-specific responsibilities
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable, Callable, Generator
from datetime import datetime
from pathlib import Path
import pandas as pd
from enum import Enum

from ..types.common import (
    RunMode, ModelVersion, EpisodeMetrics, Symbol,
    Configurable, Resettable, Serializable
)
from ..agent.interfaces_improved import ILearningAgent, IAgentCallback
from ..environment.interfaces_improved import ITradingEnvironment


class TrainingPhase(Enum):
    """Training phase indicators."""
    WARMUP = "WARMUP"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    COOLDOWN = "COOLDOWN"


class CurriculumStage(Enum):
    """Curriculum difficulty stages."""
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    EXPERT = "EXPERT"


@runtime_checkable
class ICurriculumManager(Protocol):
    """Manages training curriculum progression - MOVED OUT OF ENVIRONMENT.
    
    Responsibility: Control what data to train on and when
    """
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        ...
    
    def select_next_session(
        self,
        exclude_sessions: Optional[list] = None
    ) -> tuple[str, datetime]:
        """Select next (symbol, date) for training.
        
        Args:
            exclude_sessions: Sessions to avoid
            
        Returns:
            Tuple of (symbol, date)
        """
        ...
    
    def should_advance_stage(
        self,
        performance_metrics: dict[str, float]
    ) -> bool:
        """Determine if should advance to next stage."""
        ...
    
    def advance_stage(self) -> CurriculumStage:
        """Advance to next curriculum stage."""
        ...
    
    def get_stage_requirements(
        self,
        stage: CurriculumStage
    ) -> dict[str, Any]:
        """Get requirements for specific stage."""
        ...


@runtime_checkable
class ITrainingMode(Protocol):
    """Base interface for all training modes."""
    
    @property
    def mode_type(self) -> RunMode:
        """Type of training mode."""
        ...
    
    @property
    def is_running(self) -> bool:
        """Whether mode is currently active."""
        ...
    
    @property
    def current_phase(self) -> TrainingPhase:
        """Current training phase."""
        ...
    
    def initialize(
        self,
        agent: ILearningAgent,
        environment: ITradingEnvironment,
        config: dict[str, Any]
    ) -> None:
        """Initialize mode with components."""
        ...
    
    def run(
        self,
        callbacks: Optional[list[IAgentCallback]] = None
    ) -> dict[str, Any]:
        """Execute training mode."""
        ...
    
    def pause(self) -> None:
        """Pause execution."""
        ...
    
    def resume(self) -> None:
        """Resume execution."""
        ...
    
    def stop(self) -> None:
        """Stop execution and cleanup."""
        ...


class IStandardTrainingMode(ITrainingMode):
    """Standard episode-based training mode."""
    
    @abstractmethod
    def set_training_schedule(
        self,
        total_episodes: Optional[int] = None,
        total_steps: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> None:
        """Set training duration."""
        ...
    
    @abstractmethod
    def get_training_progress(self) -> dict[str, float]:
        """Get current training progress."""
        ...


class IContinuousTrainingMode(ITrainingMode):
    """Continuous improvement mode with model versioning."""
    
    @abstractmethod
    def set_improvement_criteria(
        self,
        metric: str = "average_reward",
        improvement_threshold: float = 0.01,
        patience: int = 10
    ) -> None:
        """Set criteria for model improvement."""
        ...
    
    @abstractmethod
    def set_curriculum_manager(
        self,
        curriculum: ICurriculumManager
    ) -> None:
        """Set curriculum management."""
        ...
    
    @abstractmethod
    def get_model_history(self) -> pd.DataFrame:
        """Get history of model versions."""
        ...
    
    @abstractmethod
    def should_create_checkpoint(self) -> bool:
        """Determine if should create new model checkpoint."""
        ...


class IOptunaMode(ITrainingMode):
    """Hyperparameter optimization mode."""
    
    @abstractmethod
    def set_search_space(
        self,
        parameter_specs: dict[str, dict[str, Any]]
    ) -> None:
        """Define hyperparameter search space."""
        ...
    
    @abstractmethod
    def set_optimization_config(
        self,
        n_trials: int,
        n_jobs: int = 1,
        sampler: str = "TPE",
        pruner: Optional[str] = "MedianPruner"
    ) -> None:
        """Configure optimization process."""
        ...
    
    @abstractmethod
    def get_best_params(self) -> dict[str, Any]:
        """Get best parameters found."""
        ...
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """Get trial history."""
        ...


class IBenchmarkMode(ITrainingMode):
    """Benchmarking and evaluation mode."""
    
    @abstractmethod
    def set_benchmark_suite(
        self,
        test_episodes: list[dict[str, Any]]
    ) -> None:
        """Define benchmark test suite."""
        ...
    
    @abstractmethod
    def add_baseline(
        self,
        name: str,
        model_path: Path
    ) -> None:
        """Add baseline model for comparison."""
        ...
    
    @abstractmethod
    def get_benchmark_results(self) -> pd.DataFrame:
        """Get comprehensive benchmark results."""
        ...
    
    @abstractmethod
    def generate_report(
        self,
        output_path: Path
    ) -> None:
        """Generate benchmark report."""
        ...


# === NEW TRAINING MODES ===

class IRealTimeTradingMode(ITrainingMode):
    """Real-time trading mode for live market interaction.
    
    Key features:
    - Live data streams
    - Real order execution
    - Risk management
    - Performance monitoring
    """
    
    @abstractmethod
    def connect_to_broker(
        self,
        broker_config: dict[str, Any]
    ) -> None:
        """Connect to live trading broker."""
        ...
    
    @abstractmethod
    def set_risk_limits(
        self,
        max_position_size: float,
        max_daily_loss: float,
        max_drawdown: float
    ) -> None:
        """Set risk management limits."""
        ...
    
    @abstractmethod
    def start_live_trading(
        self,
        symbols: list[str],
        market_hours_only: bool = True
    ) -> None:
        """Start live trading."""
        ...
    
    @abstractmethod
    def get_live_positions(self) -> dict[str, Any]:
        """Get current live positions."""
        ...
    
    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop with position liquidation."""
        ...


class IEvaluationMode(ITrainingMode):
    """Model evaluation and testing mode.
    
    Key features:
    - Out-of-sample testing
    - Performance metrics
    - Statistical analysis
    - Model comparison
    """
    
    @abstractmethod
    def set_evaluation_dataset(
        self,
        test_sessions: list[tuple[str, datetime]],
        stratified: bool = True
    ) -> None:
        """Set evaluation dataset."""
        ...
    
    @abstractmethod
    def evaluate_model(
        self,
        model_path: Path,
        deterministic: bool = True
    ) -> dict[str, float]:
        """Evaluate single model."""
        ...
    
    @abstractmethod
    def compare_models(
        self,
        model_paths: list[Path],
        significance_test: bool = True
    ) -> pd.DataFrame:
        """Compare multiple models."""
        ...
    
    @abstractmethod
    def generate_evaluation_report(
        self,
        output_path: Path,
        include_visualizations: bool = True
    ) -> None:
        """Generate comprehensive evaluation report."""
        ...


class ITransferLearningMode(ITrainingMode):
    """Transfer learning mode for adapting to new markets/assets.
    
    Key features:
    - Pre-trained model adaptation
    - Domain adaptation techniques
    - Feature transfer
    - Fine-tuning strategies
    """
    
    @abstractmethod
    def load_pretrained_model(
        self,
        model_path: Path,
        freeze_layers: Optional[list[str]] = None
    ) -> None:
        """Load pre-trained model for transfer."""
        ...
    
    @abstractmethod
    def set_target_domain(
        self,
        target_symbols: list[str],
        adaptation_strategy: str = "fine_tune"
    ) -> None:
        """Set target domain for transfer."""
        ...
    
    @abstractmethod
    def configure_adaptation(
        self,
        learning_rate_multiplier: float = 0.1,
        adaptation_episodes: int = 100
    ) -> None:
        """Configure adaptation parameters."""
        ...
    
    @abstractmethod
    def get_transfer_metrics(self) -> dict[str, float]:
        """Get transfer learning performance metrics."""
        ...


class IMultiAssetTrainingMode(ITrainingMode):
    """Multi-asset training mode for portfolio management.
    
    Key features:
    - Multiple asset training
    - Correlation-aware learning
    - Portfolio optimization
    - Risk diversification
    """
    
    @abstractmethod
    def set_asset_universe(
        self,
        assets: list[str],
        correlation_threshold: float = 0.8
    ) -> None:
        """Set asset universe for training."""
        ...
    
    @abstractmethod
    def set_portfolio_constraints(
        self,
        max_position_per_asset: float = 0.2,
        max_sector_allocation: Optional[dict[str, float]] = None
    ) -> None:
        """Set portfolio constraints."""
        ...
    
    @abstractmethod
    def enable_correlation_features(
        self,
        lookback_window: int = 60
    ) -> None:
        """Enable cross-asset correlation features."""
        ...
    
    @abstractmethod
    def get_portfolio_metrics(self) -> dict[str, float]:
        """Get portfolio-level performance metrics."""
        ...


# === TRAINING ORCHESTRATION ===

class ITrainingManager(Configurable):
    """High-level training orchestration - CURRICULUM LOGIC MOVED HERE."""
    
    @abstractmethod
    def register_mode(
        self,
        mode: ITrainingMode
    ) -> None:
        """Register a training mode."""
        ...
    
    @abstractmethod
    def set_curriculum_manager(
        self,
        curriculum: ICurriculumManager
    ) -> None:
        """Set curriculum management."""
        ...
    
    @abstractmethod
    def start_mode(
        self,
        mode_type: RunMode,
        config: dict[str, Any],
        background: bool = False
    ) -> None:
        """Start a training mode."""
        ...
    
    @abstractmethod
    def switch_mode(
        self,
        to_mode: RunMode,
        save_state: bool = True
    ) -> None:
        """Switch between modes."""
        ...
    
    @abstractmethod
    def get_active_modes(self) -> list[RunMode]:
        """Get currently active modes."""
        ...
    
    @abstractmethod
    def schedule_mode_sequence(
        self,
        sequence: list[tuple[RunMode, dict[str, Any]]]
    ) -> None:
        """Schedule sequence of modes."""
        ...
    
    # CURRICULUM ORCHESTRATION - MOVED FROM ENVIRONMENT
    @abstractmethod
    def select_next_training_session(self) -> tuple[str, datetime]:
        """Select next training session using curriculum."""
        ...
    
    @abstractmethod
    def evaluate_training_progress(
        self,
        metrics: dict[str, float]
    ) -> bool:
        """Evaluate if training should continue."""
        ...


class ITrainingMonitor(Protocol):
    """Interface for training monitoring."""
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        mode: RunMode
    ) -> None:
        """Log training metrics."""
        ...
    
    def log_episode(
        self,
        episode_metrics: EpisodeMetrics,
        mode: RunMode
    ) -> None:
        """Log episode results."""
        ...
    
    def set_alert(
        self,
        condition: Callable[[dict[str, float]], bool],
        message: str
    ) -> None:
        """Set metric alert."""
        ...
    
    def get_summary(
        self,
        mode: RunMode,
        last_n: Optional[int] = None
    ) -> dict[str, Any]:
        """Get training summary."""
        ...


# === TRAINING WORKFLOWS ===

class ITrainingWorkflow(Protocol):
    """Interface for complex training workflows."""
    
    def define_workflow(
        self,
        stages: list[dict[str, Any]]
    ) -> None:
        """Define multi-stage training workflow."""
        ...
    
    def execute_workflow(
        self,
        start_stage: Optional[str] = None
    ) -> dict[str, Any]:
        """Execute the defined workflow."""
        ...
    
    def get_workflow_progress(self) -> dict[str, float]:
        """Get workflow execution progress."""
        ...