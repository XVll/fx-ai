"""
Graceful Shutdown Manager
Handles proper cleanup of all components during termination.
"""

import signal
import logging
import threading
from typing import List, Callable, Optional
from dataclasses import dataclass
import time


@dataclass
class ShutdownComponent:
    """A component that needs graceful shutdown"""
    name: str
    shutdown_func: Callable
    timeout: float = 30.0  # Max time to wait for shutdown
    critical: bool = False  # Whether to wait for this component regardless


class GracefulShutdownManager:
    """
    Manages graceful shutdown of all training components
    
    Features:
    - Proper WandB finish with status checking
    - Dashboard cleanup
    - Model saving completion
    - SHAP analysis completion
    - Signal handling (SIGINT, SIGTERM)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.components: List[ShutdownComponent] = []
        self.shutdown_requested = False
        self.shutdown_reason = "Unknown"
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        
        # WandB status tracking
        self._wandb_run = None
        self._wandb_finished = False
        
    def register_component(self, name: str, shutdown_func: Callable, 
                          timeout: float = 30.0, critical: bool = False):
        """Register a component for graceful shutdown"""
        component = ShutdownComponent(name, shutdown_func, timeout, critical)
        self.components.append(component)
        self.logger.debug(f"📝 Registered shutdown component: {name}")
    
    def register_wandb_run(self, wandb_run):
        """Register WandB run for proper finishing"""
        self._wandb_run = wandb_run
        self.logger.debug("📝 Registered WandB run for graceful shutdown")
    
    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
        self.logger.info("🛡️ Installed graceful shutdown signal handlers")
    
    def restore_signal_handlers(self):
        """Restore original signal handlers"""
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.info(f"🛑 Received {signal_name}, initiating graceful shutdown...")
        
        self.shutdown_reason = f"Signal {signal_name}"
        self.request_shutdown()
        
        # Set global training_interrupted flag for backward compatibility
        try:
            import __main__
            __main__.training_interrupted = True
        except:
            pass
    
    def request_shutdown(self, reason: str = "Manual request"):
        """Request graceful shutdown"""
        if self.shutdown_requested:
            self.logger.warning("⚠️ Shutdown already requested, ignoring duplicate request")
            return
        
        self.shutdown_requested = True
        self.shutdown_reason = reason
        self.logger.info(f"🛑 Graceful shutdown requested: {reason}")
        
        # Set global training_interrupted flag for backward compatibility
        try:
            import __main__
            __main__.training_interrupted = True
        except:
            pass
        
        # Immediately trigger stop on all registered components that have stop_training
        for component in self.components:
            if "trainer" in component.name.lower():
                try:
                    component.shutdown_func()
                except:
                    pass
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self.shutdown_requested
    
    def perform_shutdown(self) -> bool:
        """Perform graceful shutdown of all components"""
        if not self.shutdown_requested:
            self.logger.warning("⚠️ Shutdown not requested, but performing anyway")
        
        self.logger.info(f"🔄 Starting graceful shutdown: {self.shutdown_reason}")
        success = True
        
        # Shutdown components in reverse order (LIFO)
        for component in reversed(self.components):
            component_success = self._shutdown_component(component)
            if not component_success and component.critical:
                success = False
        
        # Handle WandB specially
        wandb_success = self._shutdown_wandb()
        if not wandb_success:
            success = False
        
        if success:
            self.logger.info("✅ Graceful shutdown completed successfully")
        else:
            self.logger.warning("⚠️ Graceful shutdown completed with warnings")
        
        return success
    
    def _shutdown_component(self, component: ShutdownComponent) -> bool:
        """Shutdown a single component"""
        self.logger.info(f"🔄 Shutting down component: {component.name}")
        
        try:
            # Run shutdown function with timeout
            def target():
                component.shutdown_func()
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=component.timeout)
            
            if thread.is_alive():
                self.logger.warning(
                    f"⚠️ Component {component.name} shutdown timed out after {component.timeout}s"
                )
                return False
            else:
                self.logger.info(f"✅ Component {component.name} shutdown completed")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error shutting down component {component.name}: {e}")
            return False
    
    def _shutdown_wandb(self) -> bool:
        """Shutdown WandB with proper status checking"""
        if not self._wandb_run:
            self.logger.debug("📝 No WandB run to shutdown")
            return True
        
        self.logger.info("🔄 Shutting down WandB...")
        
        try:
            # Check if WandB is currently syncing
            max_wait_time = 60  # Max 60 seconds
            check_interval = 1  # Check every 1 second
            waited_time = 0
            
            while waited_time < max_wait_time:
                try:
                    # Check if WandB has pending sync operations
                    if hasattr(self._wandb_run, '_internal_sync_queue'):
                        queue_size = getattr(self._wandb_run._internal_sync_queue, 'qsize', lambda: 0)()
                        if queue_size == 0:
                            break
                        self.logger.debug(f"⏳ WandB sync queue size: {queue_size}")
                    else:
                        # If we can't check queue, just wait a bit
                        break
                    
                    time.sleep(check_interval)
                    waited_time += check_interval
                    
                except (AttributeError, Exception):
                    # If we can't check status, proceed with finish
                    break
            
            if waited_time >= max_wait_time:
                self.logger.warning(f"⚠️ WandB sync wait timed out after {max_wait_time}s")
            
            # Finish WandB run
            self.logger.info("📤 Finishing WandB run...")
            self._wandb_run.finish(exit_code=0)
            self._wandb_finished = True
            
            self.logger.info("✅ WandB shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error shutting down WandB: {e}")
            try:
                # Force finish if normal finish failed
                self._wandb_run.finish(exit_code=1)
                self._wandb_finished = True
                self.logger.warning("⚠️ WandB force-finished after error")
            except Exception as finish_error:
                self.logger.error(f"❌ Failed to force-finish WandB: {finish_error}")
                return False
            
            return True  # Consider it successful if we managed to force-finish
    
    def wait_for_shutdown_signal(self, check_interval: float = 0.1):
        """Wait for shutdown signal in a loop"""
        while not self.shutdown_requested:
            time.sleep(check_interval)
    
    def __enter__(self):
        """Context manager entry"""
        self.install_signal_handlers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic shutdown"""
        if exc_type == KeyboardInterrupt:
            self.request_shutdown("KeyboardInterrupt")
        elif exc_type is not None:
            self.request_shutdown(f"Exception: {exc_type.__name__}")
        
        self.perform_shutdown()
        self.restore_signal_handlers()


# Global instance for easy access
_global_shutdown_manager: Optional[GracefulShutdownManager] = None


def get_shutdown_manager() -> GracefulShutdownManager:
    """Get the global shutdown manager instance"""
    global _global_shutdown_manager
    if _global_shutdown_manager is None:
        _global_shutdown_manager = GracefulShutdownManager()
    return _global_shutdown_manager


def register_for_shutdown(name: str, shutdown_func: Callable, 
                         timeout: float = 30.0, critical: bool = False):
    """Convenience function to register component for shutdown"""
    get_shutdown_manager().register_component(name, shutdown_func, timeout, critical)


def register_wandb_for_shutdown(wandb_run):
    """Convenience function to register WandB run"""
    get_shutdown_manager().register_wandb_run(wandb_run)


def request_graceful_shutdown(reason: str = "Manual request"):
    """Convenience function to request shutdown"""
    get_shutdown_manager().request_shutdown(reason)


def is_shutdown_requested() -> bool:
    """Convenience function to check shutdown status"""
    return get_shutdown_manager().is_shutdown_requested()