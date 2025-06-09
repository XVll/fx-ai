"""
V2 Training Interfaces - Callback-Driven Training System.

This module provides the core training interface for a callback-driven architecture:
- TrainingManager as central training engine
- Callback-based feature implementation (checkpointing, evaluation, etc.)
- Configuration-driven mode behavior

Design principles:
- TrainingManager controls core training loop and data management
- Callbacks handle feature-specific behavior (enabled/disabled per mode)
- Configuration overrides control mode behavior, not separate mode classes
- Simple, testable architecture focused on training execution
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable, Callable, Dict, List
from datetime import datetime
from enum import Enum

from ..core.types import (
    TerminationReason
)


class ITrainingManager():
    """Core training engine with callback-driven behavior.
    
    Implementation Guide:
    - Central authority for training loop execution
    - Manages data selection and episode advancement  
    - Controls termination conditions and training state
    - Coordinates with callbacks for feature-specific behavior
    - Integrates with data lifecycle and graceful shutdown
    
    Responsibilities:
    - Execute main training loop (episodes, updates, termination)
    - Manage training state (counters, metrics, progress)
    - Control data lifecycle (day selection, reset point cycling)
    - Coordinate with callback manager for features
    - Handle graceful termination and cleanup
    
    Integration Points:
    - Data providers for episode configuration
    - Callback manager for feature coordination
    - Trainer for episode execution
    - Environment for trading simulation
    - Graceful shutdown system
    """

    @abstractmethod
    def start_training(
        self,
        trainer: Any,
        environment: Any,
        data_manager: Any,
        callback_manager: Any
    ) -> Dict[str, Any]:
        """Start the main training loop.
        
        Implementation Guide:
        - Initialize training state and data lifecycle
        - Load best model if continuing training (via callbacks)
        - Execute main training loop until termination
        - Coordinate with callbacks at key lifecycle points
        - Handle graceful termination and cleanup
        - Return comprehensive training statistics
        
        Args:
            trainer: PPO agent/trainer instance
            environment: Trading environment
            data_manager: Data provider for episodes
            callback_manager: Callback manager for features
            
        Returns:
            Final training results and statistics
        """
        ...

    @abstractmethod
    def should_terminate(self) -> bool:
        """Check if training should terminate.
        
        Implementation Guide:
        - Check hard termination limits (episodes, updates, cycles)
        - Check external termination requests
        - Check data lifecycle termination
        - Callbacks can request termination via request_termination()
        
        Returns:
            True if training should stop
        """
        ...

    @abstractmethod
    def get_current_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring and callbacks.
        
        Implementation Guide:
        - Include episode/update/cycle counters
        - Include current performance metrics
        - Include data lifecycle status (current day, reset point)
        - Include training progress and timing information
        
        Returns:
            Current training state dictionary
        """
        ...

    @abstractmethod
    def get_current_episode_config(self) -> Optional[Dict[str, Any]]:
        """Get current episode configuration.
        
        Implementation Guide:
        - Get current day and reset point from data lifecycle
        - Return configuration needed for trainer.run_episode()
        - Handle case where no data is available
        
        Returns:
            Episode configuration or None if no data available
        """
        ...

    @abstractmethod
    def request_termination(self, reason: TerminationReason) -> None:
        """Request graceful training termination.
        
        Implementation Guide:
        - Set termination flag and reason
        - Allow current episode to complete
        - Trigger cleanup in next training loop iteration
        - Log termination reason
        
        Args:
            reason: Reason for termination request
        """
        ...

    @abstractmethod
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics.
        
        Implementation Guide:
        - Include final performance metrics
        - Include training duration and resource usage
        - Include data lifecycle statistics
        - Include termination reason and status
        
        Returns:
            Complete training statistics dictionary
        """
        ...


class ITrainingMonitor(Protocol):
    """Training monitoring and metrics collection interface.
    
    Implementation Guide:
    - Real-time metrics collection and logging
    - Support multiple backends (W&B, TensorBoard, files)
    - Integration with callback system for event-driven monitoring
    - Comprehensive visualization and reporting
    
    Integration:
    - Called by training manager during execution
    - Receives updates via callback system
    - Provides real-time dashboard updates
    - Exports data for analysis and reporting
    """
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        episode_id: Optional[str] = None
    ) -> None:
        """Log training metrics at specific step.
        
        Args:
            metrics: Metric name -> value mapping
            step: Current training step
            episode_id: Optional episode identifier
        """
        ...
    
    def log_episode_completion(
        self,
        episode_metrics: Dict[str, Any]
    ) -> None:
        """Log complete episode results.
        
        Args:
            episode_metrics: Episode performance data
        """
        ...
    
    def log_training_state(
        self,
        training_state: Dict[str, Any]
    ) -> None:
        """Log current training state.
        
        Args:
            training_state: Current training state information
        """
        ...
    
    def get_summary(
        self,
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get training summary statistics.
        
        Args:
            last_n: Number of recent episodes to include
            
        Returns:
            Summary statistics and metrics
        """
        ...