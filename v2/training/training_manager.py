"""
TrainingManager Implementation - V2 Mode-Based Training System

Central orchestrator for all training modes with pluggable strategies.
Handles mode lifecycle, resource management, and configuration distribution.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .interfaces import (
    ITrainingManager, ITrainingMode, RunMode, ModeState, 
    IContinuousTrainingMode, IOptunaMode, IBenchmarkMode
)
from ..core.types import TerminationReason
from ..core.shutdown import IShutdownHandler, ShutdownReason


logger = logging.getLogger(__name__)


class TrainingManager(ITrainingManager, IShutdownHandler):
    """
    Concrete implementation of ITrainingManager with graceful shutdown support.
    
    Responsibilities:
    - Mode registration and lifecycle management
    - Resource coordination (trainer, environment, config)
    - Mode execution and monitoring
    - Graceful shutdown and cleanup
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TrainingManager with configuration.
        
        Args:
            config: Base configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TrainingManager")
        
        # Mode registry
        self._registered_modes: Dict[RunMode, ITrainingMode] = {}
        self._active_modes: List[RunMode] = []
        self._current_mode: Optional[ITrainingMode] = None
        
        # Resource management
        self._trainer = None
        self._environment = None
        self._initialized = False
        
        # State tracking
        self._start_time: Optional[datetime] = None
        self._termination_requested = False
        self._termination_reason: Optional[TerminationReason] = None
        
        # Register with shutdown manager
        from ..core.shutdown import get_global_shutdown_manager
        shutdown_manager = get_global_shutdown_manager()
        shutdown_manager.register_component(
            component=self,
            timeout=120.0  # 2 minutes for training manager cleanup
        )
        
        self.logger.info("🔧 TrainingManager initialized")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the training manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self.logger.info("� TrainingManager configuration updated")
    
    def reset(self) -> None:
        """Reset training manager state."""
        self._active_modes.clear()
        self._current_mode = None
        self._trainer = None
        self._environment = None
        self._initialized = False
        self._start_time = None
        self._termination_requested = False
        self._termination_reason = None
        self.logger.info("= TrainingManager reset")
    
    def register_mode(self, mode: ITrainingMode) -> None:
        """Register a training mode for use.
        
        Implementation Guide:
        - Validate mode interface compliance
        - Check for mode type conflicts
        - Store mode reference for later use
        - Set up mode-specific monitoring
        
        Args:
            mode: Training mode instance to register
        """
        if not isinstance(mode, ITrainingMode):
            raise ValueError(f"Mode must implement ITrainingMode interface: {type(mode)}")
        
        mode_type = mode.mode_type
        
        if mode_type in self._registered_modes:
            self.logger.warning(f"� Overriding existing mode: {mode_type}")
        
        self._registered_modes[mode_type] = mode
        self.logger.info(f"=� Registered training mode: {mode_type}")
    
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
        if mode_type not in self._registered_modes:
            raise ValueError(f"Mode not registered: {mode_type}")
        
        if mode_type in self._active_modes:
            raise RuntimeError(f"Mode already active: {mode_type}")
        
        # Get mode instance
        mode = self._registered_modes[mode_type]
        
        try:
            # Store resources
            self._trainer = trainer
            self._environment = environment
            
            # Initialize mode
            self.logger.info(f"=� Initializing mode: {mode_type}")
            mode.initialize(trainer, environment, config)
            
            # Track active mode
            self._active_modes.append(mode_type)
            self._current_mode = mode
            self._start_time = datetime.now()
            
            # Execute mode
            if background:
                # TODO: Implement async execution
                raise NotImplementedError("Background execution not yet implemented")
            else:
                self.logger.info(f"� Starting mode execution: {mode_type}")
                results = mode.run()
                
                # Clean up
                self._active_modes.remove(mode_type)
                if self._current_mode == mode:
                    self._current_mode = None
                
                self.logger.info(f" Mode completed: {mode_type}")
                return results
                
        except Exception as e:
            # Clean up on failure
            if mode_type in self._active_modes:
                self._active_modes.remove(mode_type)
            if self._current_mode == mode:
                self._current_mode = None
            
            self.logger.error(f"L Mode execution failed: {mode_type} - {e}")
            raise
    
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
        # TODO: Implement mode switching
        raise NotImplementedError("Mode switching not yet implemented")
    
    def get_active_modes(self) -> List[RunMode]:
        """Get list of currently active modes.
        
        Implementation Guide:
        - Check state of all registered modes
        - Return list of modes in RUNNING state
        
        Returns:
            List of active mode types
        """
        return self._active_modes.copy()
    
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
        # TODO: Implement mode sequencing
        raise NotImplementedError("Mode sequencing not yet implemented")
    
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
        status = {
            "active_modes": self._active_modes,
            "current_mode": self._current_mode.mode_type if self._current_mode else None,
            "registered_modes": list(self._registered_modes.keys()),
            "start_time": self._start_time,
            "termination_requested": self._termination_requested,
            "termination_reason": self._termination_reason
        }
        
        # Add current mode progress if available
        if self._current_mode:
            try:
                progress = self._current_mode.get_progress()
                status["current_progress"] = progress
            except Exception as e:
                self.logger.warning(f"� Failed to get progress from current mode: {e}")
                status["current_progress"] = None
        
        return status
    
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
        self._termination_requested = True
        self._termination_reason = reason
        
        if mode_type:
            self.logger.info(f"=� Termination requested for mode {mode_type}: {reason}")
            # Terminate specific mode
            if mode_type in self._registered_modes:
                mode = self._registered_modes[mode_type]
                try:
                    mode.stop()
                except Exception as e:
                    self.logger.error(f"L Error stopping mode {mode_type}: {e}")
        else:
            self.logger.info(f"=� Termination requested for all modes: {reason}")
            # Terminate all active modes
            for mode_type in self._active_modes.copy():
                try:
                    mode = self._registered_modes[mode_type]
                    mode.stop()
                except Exception as e:
                    self.logger.error(f"L Error stopping mode {mode_type}: {e}")


    # IShutdownHandler implementation
    
    def shutdown(self) -> None:
        """Perform graceful shutdown - stop training and cleanup resources."""
        self.logger.info("🛑 Shutting down TrainingManager")
        
        try:
            # Request termination of all active modes
            self.request_termination(TerminationReason.MANUAL, mode_type=None)
            
            # Stop all active modes
            for mode_type in self._active_modes.copy():
                try:
                    mode = self._registered_modes[mode_type]
                    mode.stop()
                    self.logger.info(f"✅ Stopped mode: {mode_type}")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping mode {mode_type}: {e}")
            
            # Clear state
            self._active_modes.clear()
            self._current_mode = None
            self._trainer = None
            self._environment = None
            
        except Exception as e:
            self.logger.error(f"❌ Error during TrainingManager shutdown: {e}")


def create_training_manager(config: Dict[str, Any]) -> TrainingManager:
    """Factory function to create a configured TrainingManager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TrainingManager instance
    """
    manager = TrainingManager(config)
    
    # Register default modes (when they're implemented)
    # TODO: Register concrete mode implementations
    # manager.register_mode(StandardTrainingMode())
    # manager.register_mode(ContinuousTrainingMode())
    # manager.register_mode(OptunaMode())
    # manager.register_mode(BenchmarkMode())
    
    return manager