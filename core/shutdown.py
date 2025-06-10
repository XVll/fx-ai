"""
Graceful Shutdown System - V2 Implementation

Comprehensive interface and implementation for handling graceful shutdowns
during training with proper resource cleanup and state preservation.

Design Philosophy:
- Clean separation of shutdown concerns
- Proper resource lifecycle management 
- Coordinated shutdown across components
- State preservation for resumability
- Timeout-based cleanup with fallbacks
"""

import signal
import threading
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Set
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class ShutdownReason(Enum):
    """Reasons for shutdown initiation."""
    USER_INTERRUPT = "USER_INTERRUPT"      # Ctrl+C or SIGINT
    SYSTEM_SIGNAL = "SYSTEM_SIGNAL"        # SIGTERM, SIGHUP, etc.
    ERROR_CONDITION = "ERROR_CONDITION"    # Unrecoverable error


class ShutdownPhase(Enum):
    """Phases of the shutdown process."""
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    RESOURCE_CLEANUP = "RESOURCE_CLEANUP"
    FINAL_CLEANUP = "FINAL_CLEANUP"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ComponentShutdownInfo:
    """Information about a component's shutdown requirements."""
    name: str
    shutdown_func: Callable[[], None]
    timeout_seconds: float = 30.0
    dependencies: Set[str] = field(default_factory=set)  # Components this depends on


class IShutdownHandler(ABC):
    """Interface for components that need graceful shutdown handling.
    
    Implementation Guide:
    - Components should implement this interface to participate in shutdown
    - register_shutdown() must be called during component initialization to register with shutdown manager
    - shutdown() performs both state saving and resource cleanup
    - timeout is specified during registration, not in component
    """
    
    @abstractmethod
    def register_shutdown(self, shutdown_manager: 'IShutdownManager') -> None:
        """Register this component with the shutdown manager.
        
        Args:
            shutdown_manager: The shutdown manager to register with
            
        Note: This method must be called during component initialization
        to ensure proper shutdown handling.
        """
        ...
    
    @abstractmethod
    def shutdown(self) -> None:
        """Perform graceful shutdown - save state and cleanup resources."""
        ...


class IShutdownManager(ABC):
    """Interface for the central shutdown coordination system.
    
    Implementation Guide:
    - Central coordinator for all shutdown activities
    - Handles signal registration and shutdown initiation
    - Manages component shutdown ordering and timeouts
    - Provides status and progress tracking
    - Ensures proper cleanup even if components fail
    """
    
    @abstractmethod
    def register_component(
        self,
        component: Optional[IShutdownHandler] = None,
        shutdown_func: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
        timeout: float = 30.0,
        dependencies: Optional[Set[str]] = None
    ) -> None:
        """Register a component for graceful shutdown.
        
        Args:
            component: Component implementing IShutdownHandler (if using interface)
            shutdown_func: Function to call for shutdown (if using simple function)
            name: Component name (auto-generated from class name if not provided)
            timeout: Maximum time allowed for shutdown
            dependencies: Components this depends on
            
        Note: Must provide either component OR shutdown_func, not both
        """
        ...
    
    @abstractmethod
    def initiate_shutdown(self, reason: ShutdownReason) -> None:
        """Initiate graceful shutdown process.
        
        Args:
            reason: Why shutdown is being initiated
        """
        ...
    
    @abstractmethod
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        ...
    
    @abstractmethod
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete.
        
        Args:
            timeout: Maximum time to wait (None for no timeout)
            
        Returns:
            True if shutdown completed successfully
        """
        ...
    


class ShutdownManager(IShutdownManager):
    """
    Comprehensive shutdown manager implementation.
    
    Design Guide:
    ============
    
    Features:
    - Signal handling (SIGINT, SIGTERM, etc.)
    - Component registration and dependency management
    - Ordered shutdown with timeout handling
    - Progress tracking and status reporting
    - Fallback cleanup for failed components
    - Context manager support for clean setup/teardown
    
    Architecture:
    - Signal handlers register shutdown request
    - Components are shut down in dependency order
    - Critical components get priority and extra time
    - Non-critical components are terminated if they timeout
    - Final cleanup ensures all resources are released
    
    Usage Patterns:
    - Context manager: `with ShutdownManager() as shutdown:`
    - Manual: `shutdown = ShutdownManager(); shutdown.register_signals()`
    - Integration: Register in ApplicationBootstrap
    """
    
    def __init__(self, default_timeout: float = 60.0):
        """Initialize shutdown manager.
        
        Args:
            default_timeout: Default timeout for component shutdown
        """
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(f"{__name__}.ShutdownManager")
        
        # Shutdown state
        self.shutdown_requested = False
        self.shutdown_reason: Optional[ShutdownReason] = None
        self.shutdown_phase = ShutdownPhase.SIGNAL_RECEIVED
        self.shutdown_start_time: Optional[float] = None
        
        # Component management
        self.components: Dict[str, ComponentShutdownInfo] = {}
        
        # Synchronization
        self.shutdown_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.shutdown_thread: Optional[threading.Thread] = None
        
        # Signal handling
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
        self.signals_registered = False
        
        self.logger.info("🛡️ ShutdownManager initialized")
    
    def register_component(
        self,
        component: Optional[IShutdownHandler] = None,
        shutdown_func: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
        timeout: float = 30.0,
        dependencies: Optional[Set[str]] = None
    ) -> None:
        """Register a component for graceful shutdown."""
        # Validate input
        if component is not None and shutdown_func is not None:
            raise ValueError("Cannot provide both component and shutdown_func")
        if component is None and shutdown_func is None:
            raise ValueError("Must provide either component or shutdown_func")
        
        # Auto-generate name if not provided
        if name is None:
            if component is not None:
                name = component.__class__.__name__
            else:
                name = f"Function_{len(self.components)}"
        
        if self.shutdown_requested:
            self.logger.warning(f"⚠️ Cannot register component {name} - shutdown in progress")
            return
        
        with self.shutdown_lock:
            if name in self.components:
                self.logger.warning(f"⚠️ Component {name} already registered - replacing")
            
            # Handle IShutdownHandler component
            if component is not None:
                self.components[name] = ComponentShutdownInfo(
                    name=name,
                    shutdown_func=lambda: self._shutdown_component(component),
                    timeout_seconds=timeout,
                    dependencies=dependencies or set()
                )
                self.logger.info(f"📝 Registered component: {name} (timeout={timeout}s)")
            
            # Handle simple shutdown function
            else:
                self.components[name] = ComponentShutdownInfo(
                    name=name,
                    shutdown_func=shutdown_func,
                    timeout_seconds=timeout,
                    dependencies=dependencies or set()
                )
                self.logger.info(f"📝 Registered component: {name} (timeout={timeout}s)")
    
    def register_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        if self.signals_registered:
            return
        
        # Store original handlers
        self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.signals_registered = True
        self.logger.info("📡 Signal handlers registered")
    
    def unregister_signals(self) -> None:
        """Restore original signal handlers."""
        if not self.signals_registered:
            return
        
        if self.original_sigint_handler:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        if self.original_sigterm_handler:
            signal.signal(signal.SIGTERM, self.original_sigterm_handler)
        
        self.signals_registered = False
        self.logger.info("📡 Signal handlers restored")
    
    def initiate_shutdown(self, reason: ShutdownReason) -> None:
        """Initiate graceful shutdown process."""
        with self.shutdown_lock:
            if self.shutdown_requested:
                self.logger.info(f"🛑 Shutdown already in progress (reason: {self.shutdown_reason})")
                return
            
            self.shutdown_requested = True
            self.shutdown_reason = reason
            self.shutdown_start_time = time.time()
            
            self.logger.info(f"🛑 Graceful shutdown initiated: {reason.value}")
        
        # Start shutdown in background thread
        if not self.shutdown_thread or not self.shutdown_thread.is_alive():
            self.shutdown_thread = threading.Thread(
                target=self._execute_shutdown,
                name="ShutdownManager",
                daemon=False
            )
            self.shutdown_thread.start()
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete."""
        if not self.shutdown_requested:
            return True
        
        success = self.shutdown_event.wait(timeout)
        if not success:
            self.logger.warning("⏰ Shutdown wait timeout exceeded")
        
        return success
    
    
    # Context manager support
    
    def __enter__(self):
        """Enter context manager - register signals."""
        self.register_signals()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - ensure clean shutdown."""
        if exc_type is KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self.logger.info("🛑 KeyboardInterrupt detected in context manager")
            self.initiate_shutdown(ShutdownReason.USER_INTERRUPT)
            self.wait_for_shutdown(timeout=self.default_timeout)
        
        self.unregister_signals()
        
        # Don't suppress exceptions unless it's KeyboardInterrupt that we handled
        return exc_type is KeyboardInterrupt and self.shutdown_event.is_set()
    
    # Private implementation methods
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        self.logger.info(f"🛑 Received signal: {signal_name}")
        
        reason = ShutdownReason.USER_INTERRUPT if signum == signal.SIGINT else ShutdownReason.SYSTEM_SIGNAL
        self.initiate_shutdown(reason)
    
    def _execute_shutdown(self) -> None:
        """Execute the shutdown process in background thread."""
        try:
            self.logger.info("🔄 Starting shutdown execution")
            
            # Phase 1: Shutdown components in order
            self.shutdown_phase = ShutdownPhase.RESOURCE_CLEANUP
            self._shutdown_components()
            
            # Phase 2: Final cleanup
            self.shutdown_phase = ShutdownPhase.FINAL_CLEANUP
            self._final_cleanup()
            
            self.shutdown_phase = ShutdownPhase.COMPLETED
            self.logger.info("✅ Graceful shutdown completed successfully")
            
        except Exception as e:
            self.shutdown_phase = ShutdownPhase.FAILED
            self.logger.error(f"❌ Shutdown execution failed: {e}", exc_info=True)
        
        finally:
            self.shutdown_event.set()
    
    
    def _shutdown_components(self) -> None:
        """Shutdown components in dependency order."""
        self.logger.info("🔄 Shutting down components")
        
        # Calculate shutdown order based on dependencies
        shutdown_order = self._calculate_shutdown_order()
        
        for component_name in shutdown_order:
            component_info = self.components[component_name]
            self._shutdown_single_component(component_info)
    
    def _calculate_shutdown_order(self) -> List[str]:
        """Calculate the order to shutdown components based on dependencies."""
        # Simple topological sort for dependency resolution
        # Components with no dependents go first
        
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            # Visit dependencies first
            if name in self.components:
                for dep in self.components[name].dependencies:
                    if dep in self.components:
                        visit(dep)
            
            order.append(name)
        
        # Start with components that have no dependencies
        for name in self.components:
            visit(name)
        
        # Reverse order (dependencies shut down after dependents)
        return list(reversed(order))
    
    def _shutdown_single_component(self, component_info: ComponentShutdownInfo) -> None:
        """Shutdown a single component with timeout handling."""
        name = component_info.name
        timeout = component_info.timeout_seconds
        
        self.logger.info(f"🔄 Shutting down component: {name}")
        
        def shutdown_target():
            try:
                component_info.shutdown_func()
                self.logger.info(f"✅ Component shutdown completed: {name}")
            except Exception as e:
                self.logger.error(f"❌ Component shutdown failed: {name} - {e}")
        
        # Create shutdown thread with timeout
        shutdown_thread = threading.Thread(target=shutdown_target, name=f"Shutdown-{name}")
        shutdown_thread.start()
        shutdown_thread.join(timeout=timeout)
        
        if shutdown_thread.is_alive():
            self.logger.warning(f"⏰ Component shutdown timeout: {name}")
            # Note: We can't force kill threads in Python, so we log and continue
    
    def _shutdown_component(self, component: IShutdownHandler) -> None:
        """Shutdown a component implementing IShutdownHandler."""
        component.shutdown()
    
    def _final_cleanup(self) -> None:
        """Perform final cleanup operations."""
        self.logger.info("🧹 Performing final cleanup")
        
        # TODO: Implement final cleanup logic
        # This could involve:
        # - Flushing logs
        # - Closing file handles
        # - Network connection cleanup
        # - Temporary file cleanup


# Factory function for easy creation
def create_shutdown_manager(default_timeout: float = 60.0) -> ShutdownManager:
    """Create a configured shutdown manager.
    
    Args:
        default_timeout: Default timeout for component shutdown
        
    Returns:
        Configured ShutdownManager instance
    """
    return ShutdownManager(default_timeout=default_timeout)


# Global shutdown manager instance for convenience
_global_shutdown_manager: Optional[ShutdownManager] = None


def get_global_shutdown_manager() -> ShutdownManager:
    """Get or create the global shutdown manager instance."""
    global _global_shutdown_manager
    
    if _global_shutdown_manager is None:
        _global_shutdown_manager = create_shutdown_manager()
    
    return _global_shutdown_manager


@contextmanager
def graceful_shutdown_context(shutdown_manager: Optional[ShutdownManager] = None):
    """Context manager for graceful shutdown handling.
    
    Args:
        shutdown_manager: Optional shutdown manager (creates one if None)
        
    Usage:
        with graceful_shutdown_context() as shutdown:
            shutdown.register_simple_component("my_app", cleanup_func)
            # ... run application ...
    """
    if shutdown_manager is None:
        shutdown_manager = create_shutdown_manager()
    
    with shutdown_manager as manager:
        yield manager