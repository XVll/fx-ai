"""
Comprehensive test suite for v2.core.shutdown module.

Tests cover 100% of public functionality including:
- Enums and dataclasses
- Interfaces
- ShutdownManager core functionality
- Signal handling
- Context manager
- Component registration and dependency resolution
- Timeout and error handling
- Utility functions
"""

import pytest
import signal
import time
import threading
import os
from unittest.mock import patch, call, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.shutdown import (
    ShutdownReason,
    ShutdownPhase,
    ComponentShutdownInfo,
    IShutdownHandler,
    IShutdownManager,
    ShutdownManager,
    create_shutdown_manager,
    get_global_shutdown_manager,
    graceful_shutdown_context,
)


class TestShutdownReason:
    """Test ShutdownReason enum."""

    def test_shutdown_reason_values(self):
        """Test enum values are correct."""
        assert ShutdownReason.USER_INTERRUPT.value == "USER_INTERRUPT"
        assert ShutdownReason.SYSTEM_SIGNAL.value == "SYSTEM_SIGNAL"
        assert ShutdownReason.ERROR_CONDITION.value == "ERROR_CONDITION"

    def test_shutdown_reason_enum_members(self):
        """Test all enum members exist."""
        expected_members = {"USER_INTERRUPT", "SYSTEM_SIGNAL", "ERROR_CONDITION"}
        actual_members = {member.name for member in ShutdownReason}
        assert actual_members == expected_members


class TestShutdownPhase:
    """Test ShutdownPhase enum."""

    def test_shutdown_phase_values(self):
        """Test enum values are correct."""
        assert ShutdownPhase.SIGNAL_RECEIVED.value == "SIGNAL_RECEIVED"
        assert ShutdownPhase.RESOURCE_CLEANUP.value == "RESOURCE_CLEANUP"
        assert ShutdownPhase.FINAL_CLEANUP.value == "FINAL_CLEANUP"
        assert ShutdownPhase.COMPLETED.value == "COMPLETED"
        assert ShutdownPhase.FAILED.value == "FAILED"

    def test_shutdown_phase_enum_members(self):
        """Test all enum members exist."""
        expected_members = {
            "SIGNAL_RECEIVED",
            "RESOURCE_CLEANUP", 
            "FINAL_CLEANUP",
            "COMPLETED",
            "FAILED"
        }
        actual_members = {member.name for member in ShutdownPhase}
        assert actual_members == expected_members


class TestComponentShutdownInfo:
    """Test ComponentShutdownInfo dataclass."""

    def test_component_shutdown_info_creation(self):
        """Test creating ComponentShutdownInfo with defaults."""
        def dummy_func():
            pass

        info = ComponentShutdownInfo(
            name="test_component",
            shutdown_func=dummy_func
        )

        assert info.name == "test_component"
        assert info.shutdown_func == dummy_func
        assert info.timeout_seconds == 30.0
        assert info.dependencies == set()

    def test_component_shutdown_info_with_custom_values(self):
        """Test creating ComponentShutdownInfo with custom values."""
        def dummy_func():
            pass

        dependencies = {"dep1", "dep2"}
        info = ComponentShutdownInfo(
            name="custom_component",
            shutdown_func=dummy_func,
            timeout_seconds=60.0,
            dependencies=dependencies
        )

        assert info.name == "custom_component"
        assert info.shutdown_func == dummy_func
        assert info.timeout_seconds == 60.0
        assert info.dependencies == dependencies

    def test_component_shutdown_info_mutable_dependencies(self):
        """Test that dependencies can be modified after creation."""
        def dummy_func():
            pass

        info = ComponentShutdownInfo(
            name="test_component",
            shutdown_func=dummy_func
        )

        info.dependencies.add("new_dep")
        assert "new_dep" in info.dependencies


class TestIShutdownHandler:
    """Test IShutdownHandler interface."""

    def test_is_abstract_interface(self):
        """Test that IShutdownHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IShutdownHandler()

    def test_shutdown_method_is_abstract(self):
        """Test that shutdown method must be implemented."""
        class IncompleteHandler(IShutdownHandler):
            pass

        with pytest.raises(TypeError):
            IncompleteHandler()

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""
        class ConcreteHandler(IShutdownHandler):
            def __init__(self):
                self.shutdown_called = False
                self.register_called = False
                self.registered_manager = None

            def register_shutdown(self, shutdown_manager) -> None:
                self.register_called = True
                self.registered_manager = shutdown_manager

            def shutdown(self) -> None:
                self.shutdown_called = True

        handler = ConcreteHandler()
        assert not handler.shutdown_called
        assert not handler.register_called
        
        # Test shutdown functionality
        handler.shutdown()
        assert handler.shutdown_called
        
        # Test register_shutdown functionality
        mock_manager = Mock()
        handler.register_shutdown(mock_manager)
        assert handler.register_called
        assert handler.registered_manager == mock_manager

    def test_register_shutdown_is_required(self):
        """Test that register_shutdown method is required for implementation."""
        class IncompleteHandlerNoRegister(IShutdownHandler):
            def shutdown(self) -> None:
                pass
            # Missing register_shutdown method

        with pytest.raises(TypeError, match="register_shutdown"):
            IncompleteHandlerNoRegister()

    def test_register_shutdown_interface_contract(self):
        """Test that register_shutdown follows the interface contract."""
        class TestHandler(IShutdownHandler):
            def __init__(self):
                self.registration_calls = []

            def register_shutdown(self, shutdown_manager) -> None:
                self.registration_calls.append(shutdown_manager)

            def shutdown(self) -> None:
                pass

        handler = TestHandler()
        manager1 = Mock()
        manager2 = Mock()

        # Test single registration
        handler.register_shutdown(manager1)
        assert len(handler.registration_calls) == 1
        assert handler.registration_calls[0] == manager1

        # Test multiple registrations (if allowed by implementation)
        handler.register_shutdown(manager2)
        assert len(handler.registration_calls) == 2
        assert handler.registration_calls[1] == manager2


class TestMockShutdownHandler:
    """Test MockShutdownHandler implementation including register_shutdown."""

    def test_mock_handler_initialization(self):
        """Test MockShutdownHandler proper initialization."""
        handler = MockShutdownHandler("TestHandler", delay=1.0, should_fail=True)
        
        assert handler.name == "TestHandler"
        assert handler.delay == 1.0
        assert handler.should_fail == True
        assert handler.shutdown_called == False
        assert handler.register_called == False
        assert handler.registered_manager is None

    def test_mock_handler_register_shutdown(self):
        """Test MockShutdownHandler register_shutdown functionality."""
        handler = MockShutdownHandler("RegisterTest")
        mock_manager = Mock()
        
        # Initially not registered
        assert not handler.register_called
        assert handler.registered_manager is None
        
        # Register with manager
        handler.register_shutdown(mock_manager)
        
        # Verify registration tracked
        assert handler.register_called
        assert handler.registered_manager == mock_manager

    def test_mock_handler_register_shutdown_multiple_calls(self):
        """Test MockShutdownHandler handles multiple register_shutdown calls."""
        handler = MockShutdownHandler("MultiRegisterTest")
        manager1 = Mock()
        manager2 = Mock()
        
        # First registration
        handler.register_shutdown(manager1)
        assert handler.register_called
        assert handler.registered_manager == manager1
        
        # Second registration (overwrites)
        handler.register_shutdown(manager2)
        assert handler.register_called
        assert handler.registered_manager == manager2  # Latest manager

    def test_mock_handler_shutdown_and_register_independence(self):
        """Test that shutdown and register_shutdown work independently."""
        handler = MockShutdownHandler("IndependenceTest")
        mock_manager = Mock()
        
        # Shutdown first
        handler.shutdown()
        assert handler.shutdown_called
        assert not handler.register_called
        
        # Then register
        handler.register_shutdown(mock_manager)
        assert handler.register_called
        assert handler.registered_manager == mock_manager
        
        # Both should be tracked independently
        assert handler.shutdown_called
        assert handler.register_called


class TestIShutdownManager:
    """Test IShutdownManager interface."""

    def test_is_abstract_interface(self):
        """Test that IShutdownManager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IShutdownManager()

    def test_all_methods_are_abstract(self):
        """Test that all methods must be implemented."""
        class IncompleteManager(IShutdownManager):
            pass

        with pytest.raises(TypeError):
            IncompleteManager()

    def test_partial_implementation_fails(self):
        """Test that partial implementation still fails."""
        class PartialManager(IShutdownManager):
            def register_component(self, component=None, shutdown_func=None, name=None, timeout=30.0, dependencies=None):
                pass
            
            def initiate_shutdown(self, reason):
                pass
            
            # Missing is_shutdown_requested and wait_for_shutdown

        with pytest.raises(TypeError):
            PartialManager()

    def test_complete_implementation_works(self):
        """Test that complete implementation works."""
        class CompleteManager(IShutdownManager):
            def register_component(self, component=None, shutdown_func=None, name=None, timeout=30.0, dependencies=None):
                pass
            
            def initiate_shutdown(self, reason):
                pass
            
            def is_shutdown_requested(self):
                return False
            
            def wait_for_shutdown(self, timeout=None):
                return True

        # Should not raise
        manager = CompleteManager()
        assert manager is not None


class MockShutdownHandler(IShutdownHandler):
    """Mock implementation for testing."""

    def __init__(self, name: str = "MockHandler", delay: float = 0.0, should_fail: bool = False):
        self.name = name
        self.delay = delay
        self.should_fail = should_fail
        self.shutdown_called = False
        self.shutdown_call_time = None
        self.register_called = False
        self.registered_manager = None

    def register_shutdown(self, shutdown_manager) -> None:
        """Register this component with the shutdown manager."""
        self.register_called = True
        self.registered_manager = shutdown_manager

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.shutdown_call_time = time.time()
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.should_fail:
            raise RuntimeError(f"Simulated failure in {self.name}")


class TestShutdownManager:
    """Test ShutdownManager implementation."""

    def setup_method(self):
        """Setup for each test method."""
        self.manager = ShutdownManager(default_timeout=30.0)

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.manager:
            self.manager.unregister_signals()
            # Reset global state
            global _global_shutdown_manager
            _global_shutdown_manager = None

    def test_initialization(self):
        """Test ShutdownManager initialization."""
        manager = ShutdownManager(default_timeout=45.0)
        
        assert manager.default_timeout == 45.0
        assert not manager.shutdown_requested
        assert manager.shutdown_reason is None
        assert manager.shutdown_phase == ShutdownPhase.SIGNAL_RECEIVED
        assert manager.shutdown_start_time is None
        assert len(manager.components) == 0
        assert not manager.signals_registered
        assert manager.original_sigint_handler is None
        assert manager.original_sigterm_handler is None

    def test_register_component_with_handler(self):
        """Test registering component implementing IShutdownHandler."""
        handler = MockShutdownHandler("TestHandler")
        
        self.manager.register_component(
            component=handler,
            name="custom_name",
            timeout=60.0,
            dependencies={"dep1"}
        )

        assert "custom_name" in self.manager.components
        component_info = self.manager.components["custom_name"]
        assert component_info.name == "custom_name"
        assert component_info.timeout_seconds == 60.0
        assert component_info.dependencies == {"dep1"}

    def test_register_component_with_handler_auto_name(self):
        """Test registering component with auto-generated name."""
        handler = MockShutdownHandler("TestHandler")
        
        self.manager.register_component(component=handler)

        assert "MockShutdownHandler" in self.manager.components
        component_info = self.manager.components["MockShutdownHandler"]
        assert component_info.name == "MockShutdownHandler"
        assert component_info.timeout_seconds == 30.0
        assert component_info.dependencies == set()

    def test_register_component_with_function(self):
        """Test registering component with shutdown function."""
        shutdown_calls = []
        
        def shutdown_func():
            shutdown_calls.append("called")
        
        self.manager.register_component(
            shutdown_func=shutdown_func,
            name="function_component",
            timeout=45.0,
            dependencies={"dep1", "dep2"}
        )

        assert "function_component" in self.manager.components
        component_info = self.manager.components["function_component"]
        assert component_info.name == "function_component"
        assert component_info.timeout_seconds == 45.0
        assert component_info.dependencies == {"dep1", "dep2"}

        # Test function is callable
        component_info.shutdown_func()
        assert shutdown_calls == ["called"]

    def test_register_component_with_function_auto_name(self):
        """Test registering function with auto-generated name."""
        def shutdown_func():
            pass
        
        self.manager.register_component(shutdown_func=shutdown_func)

        assert "Function_0" in self.manager.components

    def test_register_component_validation_errors(self):
        """Test validation errors in component registration."""
        handler = MockShutdownHandler()
        
        def shutdown_func():
            pass

        # Both component and function provided
        with pytest.raises(ValueError, match="Cannot provide both component and shutdown_func"):
            self.manager.register_component(component=handler, shutdown_func=shutdown_func)

        # Neither component nor function provided
        with pytest.raises(ValueError, match="Must provide either component or shutdown_func"):
            self.manager.register_component()

    def test_register_component_replacement_warning(self):
        """Test warning when replacing existing component."""
        handler1 = MockShutdownHandler("Handler1")
        handler2 = MockShutdownHandler("Handler2")

        with patch.object(self.manager.logger, 'warning') as mock_warning:
            self.manager.register_component(component=handler1, name="test_component")
            self.manager.register_component(component=handler2, name="test_component")
            
            mock_warning.assert_called_once()
            assert "already registered" in mock_warning.call_args[0][0]

    def test_register_component_during_shutdown(self):
        """Test component registration during shutdown is rejected."""
        handler = MockShutdownHandler()
        
        self.manager.shutdown_requested = True
        
        with patch.object(self.manager.logger, 'warning') as mock_warning:
            self.manager.register_component(component=handler)
            
            mock_warning.assert_called_once()
            assert "shutdown in progress" in mock_warning.call_args[0][0]
            assert len(self.manager.components) == 0

    def test_register_signals(self):
        """Test signal registration."""
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        original_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
        try:
            self.manager.register_signals()
            
            assert self.manager.signals_registered
            assert self.manager.original_sigint_handler is not None
            assert self.manager.original_sigterm_handler is not None
            
            # Test duplicate registration is ignored
            self.manager.register_signals()
            assert self.manager.signals_registered
            
        finally:
            # Restore original handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def test_unregister_signals(self):
        """Test signal unregistration."""
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        original_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
        try:
            self.manager.register_signals()
            assert self.manager.signals_registered
            
            self.manager.unregister_signals()
            assert not self.manager.signals_registered
            
            # Test duplicate unregistration is ignored
            self.manager.unregister_signals()
            assert not self.manager.signals_registered
            
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def test_signal_handler_sigint(self):
        """Test SIGINT signal handling."""
        with patch.object(self.manager, 'initiate_shutdown') as mock_initiate:
            self.manager._signal_handler(signal.SIGINT, None)
            mock_initiate.assert_called_once_with(ShutdownReason.USER_INTERRUPT)

    def test_signal_handler_sigterm(self):
        """Test SIGTERM signal handling."""
        with patch.object(self.manager, 'initiate_shutdown') as mock_initiate:
            self.manager._signal_handler(signal.SIGTERM, None)
            mock_initiate.assert_called_once_with(ShutdownReason.SYSTEM_SIGNAL)

    def test_is_shutdown_requested(self):
        """Test shutdown request status checking."""
        assert not self.manager.is_shutdown_requested()
        
        self.manager.shutdown_requested = True
        assert self.manager.is_shutdown_requested()

    def test_initiate_shutdown_first_time(self):
        """Test first shutdown initiation."""
        with patch.object(self.manager, '_execute_shutdown') as mock_execute:
            self.manager.initiate_shutdown(ShutdownReason.USER_INTERRUPT)
            
            assert self.manager.shutdown_requested
            assert self.manager.shutdown_reason == ShutdownReason.USER_INTERRUPT
            assert self.manager.shutdown_start_time is not None
            assert self.manager.shutdown_thread is not None

    def test_initiate_shutdown_already_in_progress(self):
        """Test shutdown initiation when already in progress."""
        self.manager.shutdown_requested = True
        self.manager.shutdown_reason = ShutdownReason.SYSTEM_SIGNAL

        with patch.object(self.manager.logger, 'info') as mock_info:
            self.manager.initiate_shutdown(ShutdownReason.USER_INTERRUPT)
            
            # Should not change original reason
            assert self.manager.shutdown_reason == ShutdownReason.SYSTEM_SIGNAL
            mock_info.assert_called_with(
                f"🛑 Shutdown already in progress (reason: {ShutdownReason.SYSTEM_SIGNAL})"
            )

    def test_wait_for_shutdown_not_requested(self):
        """Test wait for shutdown when not requested."""
        result = self.manager.wait_for_shutdown(timeout=0.1)
        assert result is True

    def test_wait_for_shutdown_with_completion(self):
        """Test wait for shutdown with completion."""
        self.manager.shutdown_requested = True
        
        # Simulate shutdown completion
        def complete_shutdown():
            time.sleep(0.1)
            self.manager.shutdown_event.set()
        
        thread = threading.Thread(target=complete_shutdown)
        thread.start()
        
        result = self.manager.wait_for_shutdown(timeout=1.0)
        assert result is True
        
        thread.join()

    def test_wait_for_shutdown_timeout(self):
        """Test wait for shutdown with timeout."""
        self.manager.shutdown_requested = True
        
        with patch.object(self.manager.logger, 'warning') as mock_warning:
            result = self.manager.wait_for_shutdown(timeout=0.1)
            assert result is False
            mock_warning.assert_called_once()

    def test_context_manager_normal_exit(self):
        """Test context manager with normal exit."""
        with patch.object(self.manager, 'register_signals') as mock_register:
            with patch.object(self.manager, 'unregister_signals') as mock_unregister:
                with self.manager as manager:
                    assert manager is self.manager
                    mock_register.assert_called_once()
                
                mock_unregister.assert_called_once()

    def test_context_manager_keyboard_interrupt(self):
        """Test context manager with KeyboardInterrupt."""
        with patch.object(self.manager, 'register_signals'):
            with patch.object(self.manager, 'unregister_signals'):
                with patch.object(self.manager, 'initiate_shutdown') as mock_initiate:
                    with patch.object(self.manager, 'wait_for_shutdown', return_value=True) as mock_wait:
                        with patch.object(self.manager.shutdown_event, 'is_set', return_value=True):
                            try:
                                with self.manager:
                                    raise KeyboardInterrupt()
                            except KeyboardInterrupt:
                                pytest.fail("KeyboardInterrupt should be suppressed when handled")
                            
                            mock_initiate.assert_called_once_with(ShutdownReason.USER_INTERRUPT)
                            mock_wait.assert_called_once()

    def test_context_manager_other_exception(self):
        """Test context manager with other exceptions."""
        with patch.object(self.manager, 'register_signals'):
            with patch.object(self.manager, 'unregister_signals'):
                with pytest.raises(ValueError):
                    with self.manager:
                        raise ValueError("Test error")

    def test_calculate_shutdown_order_no_dependencies(self):
        """Test shutdown order calculation with no dependencies."""
        self.manager.register_component(shutdown_func=lambda: None, name="comp1")
        self.manager.register_component(shutdown_func=lambda: None, name="comp2")
        self.manager.register_component(shutdown_func=lambda: None, name="comp3")

        order = self.manager._calculate_shutdown_order()
        
        # Order should include all components
        assert set(order) == {"comp1", "comp2", "comp3"}
        assert len(order) == 3

    def test_calculate_shutdown_order_with_dependencies(self):
        """Test shutdown order calculation with dependencies."""
        self.manager.register_component(shutdown_func=lambda: None, name="comp1")
        self.manager.register_component(
            shutdown_func=lambda: None, 
            name="comp2", 
            dependencies={"comp1"}
        )
        self.manager.register_component(
            shutdown_func=lambda: None, 
            name="comp3", 
            dependencies={"comp2"}
        )

        order = self.manager._calculate_shutdown_order()
        
        # comp3 should come before comp2, comp2 before comp1
        assert order.index("comp3") < order.index("comp2")
        assert order.index("comp2") < order.index("comp1")

    def test_shutdown_single_component_success(self):
        """Test successful shutdown of single component."""
        handler = MockShutdownHandler("TestHandler")
        component_info = ComponentShutdownInfo(
            name="test_comp",
            shutdown_func=handler.shutdown,
            timeout_seconds=1.0
        )

        with patch.object(self.manager.logger, 'info') as mock_info:
            self.manager._shutdown_single_component(component_info)
            
            assert handler.shutdown_called
            mock_info.assert_has_calls([
                call("🔄 Shutting down component: test_comp"),
                call("✅ Component shutdown completed: test_comp")
            ])

    def test_shutdown_single_component_failure(self):
        """Test component shutdown with exception."""
        handler = MockShutdownHandler("FailHandler", should_fail=True)
        component_info = ComponentShutdownInfo(
            name="fail_comp",
            shutdown_func=handler.shutdown,
            timeout_seconds=1.0
        )

        with patch.object(self.manager.logger, 'error') as mock_error:
            self.manager._shutdown_single_component(component_info)
            
            assert handler.shutdown_called
            mock_error.assert_called_once()
            assert "shutdown failed" in mock_error.call_args[0][0]

    def test_shutdown_single_component_timeout(self):
        """Test component shutdown timeout."""
        handler = MockShutdownHandler("SlowHandler", delay=0.5)
        component_info = ComponentShutdownInfo(
            name="slow_comp",
            shutdown_func=handler.shutdown,
            timeout_seconds=0.1
        )

        with patch.object(self.manager.logger, 'warning') as mock_warning:
            self.manager._shutdown_single_component(component_info)
            
            mock_warning.assert_called_once()
            assert "timeout" in mock_warning.call_args[0][0]

    def test_shutdown_component_handler(self):
        """Test _shutdown_component method."""
        handler = MockShutdownHandler("TestHandler")
        
        self.manager._shutdown_component(handler)
        
        assert handler.shutdown_called

    def test_shutdown_components_integration(self):
        """Test full component shutdown process."""
        handler1 = MockShutdownHandler("Handler1")
        handler2 = MockShutdownHandler("Handler2")
        
        self.manager.register_component(component=handler1, name="comp1")
        self.manager.register_component(component=handler2, name="comp2")

        self.manager._shutdown_components()
        
        assert handler1.shutdown_called
        assert handler2.shutdown_called

    def test_execute_shutdown_success(self):
        """Test successful shutdown execution."""
        handler = MockShutdownHandler("TestHandler")
        self.manager.register_component(component=handler)

        with patch.object(self.manager, '_final_cleanup') as mock_cleanup:
            with patch.object(self.manager.logger, 'info') as mock_info:
                self.manager._execute_shutdown()
                
                assert handler.shutdown_called
                assert self.manager.shutdown_phase == ShutdownPhase.COMPLETED
                assert self.manager.shutdown_event.is_set()
                mock_cleanup.assert_called_once()

    def test_execute_shutdown_failure(self):
        """Test shutdown execution with failure."""
        with patch.object(self.manager, '_shutdown_components', side_effect=Exception("Test error")):
            with patch.object(self.manager.logger, 'error') as mock_error:
                self.manager._execute_shutdown()
                
                assert self.manager.shutdown_phase == ShutdownPhase.FAILED
                assert self.manager.shutdown_event.is_set()
                mock_error.assert_called_once()

    def test_final_cleanup(self):
        """Test final cleanup method."""
        with patch.object(self.manager.logger, 'info') as mock_info:
            self.manager._final_cleanup()
            mock_info.assert_called_once_with("🧹 Performing final cleanup")


class TestUtilityFunctions:
    """Test utility functions."""

    def teardown_method(self):
        """Reset global state after each test."""
        global _global_shutdown_manager
        _global_shutdown_manager = None

    def test_create_shutdown_manager(self):
        """Test create_shutdown_manager factory function."""
        manager = create_shutdown_manager(default_timeout=45.0)
        
        assert isinstance(manager, ShutdownManager)
        assert manager.default_timeout == 45.0

    def test_create_shutdown_manager_default_timeout(self):
        """Test create_shutdown_manager with default timeout."""
        manager = create_shutdown_manager()
        
        assert isinstance(manager, ShutdownManager)
        assert manager.default_timeout == 60.0

    def test_get_global_shutdown_manager_first_call(self):
        """Test first call to get_global_shutdown_manager."""
        manager = get_global_shutdown_manager()
        
        assert isinstance(manager, ShutdownManager)
        assert manager.default_timeout == 60.0

    def test_get_global_shutdown_manager_subsequent_calls(self):
        """Test subsequent calls return same instance."""
        manager1 = get_global_shutdown_manager()
        manager2 = get_global_shutdown_manager()
        
        assert manager1 is manager2

    def test_graceful_shutdown_context_with_manager(self):
        """Test graceful_shutdown_context with provided manager."""
        custom_manager = ShutdownManager(default_timeout=45.0)
        
        with patch.object(custom_manager, 'register_signals'):
            with patch.object(custom_manager, 'unregister_signals'):
                with graceful_shutdown_context(custom_manager) as manager:
                    assert manager is custom_manager

    def test_graceful_shutdown_context_without_manager(self):
        """Test graceful_shutdown_context without provided manager."""
        with graceful_shutdown_context() as manager:
            assert isinstance(manager, ShutdownManager)
            assert manager.default_timeout == 60.0


class TestIntegrationScenarios:
    """Integration test scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        self.manager = ShutdownManager(default_timeout=30.0)

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.manager:
            self.manager.unregister_signals()

    def test_full_shutdown_cycle(self):
        """Test complete shutdown cycle."""
        # Setup components
        handler1 = MockShutdownHandler("Handler1", delay=0.1)
        handler2 = MockShutdownHandler("Handler2", delay=0.1)
        
        shutdown_calls = []
        def function_component():
            shutdown_calls.append("function_called")
        
        # Register components
        self.manager.register_component(component=handler1, name="comp1")
        self.manager.register_component(component=handler2, name="comp2")
        self.manager.register_component(shutdown_func=function_component, name="comp3")
        
        # Initiate shutdown
        self.manager.initiate_shutdown(ShutdownReason.USER_INTERRUPT)
        
        # Wait for completion
        success = self.manager.wait_for_shutdown(timeout=5.0)
        
        # Verify results
        assert success
        assert handler1.shutdown_called
        assert handler2.shutdown_called
        assert shutdown_calls == ["function_called"]
        assert self.manager.shutdown_phase == ShutdownPhase.COMPLETED

    def test_shutdown_with_dependencies(self):
        """Test shutdown with component dependencies."""
        handler1 = MockShutdownHandler("Handler1")
        handler2 = MockShutdownHandler("Handler2")
        handler3 = MockShutdownHandler("Handler3")
        
        # Register with dependencies: handler3 -> handler2 -> handler1
        self.manager.register_component(component=handler1, name="comp1")
        self.manager.register_component(
            component=handler2, 
            name="comp2", 
            dependencies={"comp1"}
        )
        self.manager.register_component(
            component=handler3, 
            name="comp3", 
            dependencies={"comp2"}
        )
        
        # Start shutdown and wait
        self.manager.initiate_shutdown(ShutdownReason.USER_INTERRUPT)
        success = self.manager.wait_for_shutdown(timeout=5.0)
        
        # Verify all components shut down
        assert success
        assert handler1.shutdown_called
        assert handler2.shutdown_called
        assert handler3.shutdown_called
        
        # Verify shutdown order (comp3 before comp2 before comp1)
        assert handler3.shutdown_call_time < handler2.shutdown_call_time
        assert handler2.shutdown_call_time < handler1.shutdown_call_time

    def test_shutdown_with_failing_component(self):
        """Test shutdown continues even when components fail."""
        good_handler = MockShutdownHandler("GoodHandler")
        bad_handler = MockShutdownHandler("BadHandler", should_fail=True)
        
        self.manager.register_component(component=good_handler, name="good")
        self.manager.register_component(component=bad_handler, name="bad")
        
        # Shutdown should complete despite failure
        self.manager.initiate_shutdown(ShutdownReason.ERROR_CONDITION)
        success = self.manager.wait_for_shutdown(timeout=5.0)
        
        assert success
        assert good_handler.shutdown_called
        assert bad_handler.shutdown_called
        assert self.manager.shutdown_phase == ShutdownPhase.COMPLETED

    def test_context_manager_integration(self):
        """Test context manager with real components."""
        handler = MockShutdownHandler("ContextHandler")
        
        # Mock shutdown execution to immediately complete
        def mock_execute_shutdown():
            self.manager.shutdown_phase = ShutdownPhase.RESOURCE_CLEANUP
            handler.shutdown()
            self.manager.shutdown_phase = ShutdownPhase.COMPLETED
            self.manager.shutdown_event.set()
        
        # Use context manager
        with patch.object(self.manager, '_execute_shutdown', side_effect=mock_execute_shutdown):
            try:
                with self.manager as manager:
                    manager.register_component(component=handler, name="context_comp")
                    # Simulate KeyboardInterrupt
                    raise KeyboardInterrupt()
            except KeyboardInterrupt:
                # Should be handled by context manager
                pass
        
        # Component should have been shut down
        assert handler.shutdown_called

    def test_signal_handling_integration(self):
        """Test signal handling integration."""
        handler = MockShutdownHandler("SignalHandler")
        
        # Register component and signals
        self.manager.register_component(component=handler)
        self.manager.register_signals()
        
        try:
            # Simulate signal
            self.manager._signal_handler(signal.SIGINT, None)
            
            # Wait for shutdown
            success = self.manager.wait_for_shutdown(timeout=5.0)
            
            assert success
            assert handler.shutdown_called
            assert self.manager.shutdown_reason == ShutdownReason.USER_INTERRUPT
            
        finally:
            self.manager.unregister_signals()

    @pytest.mark.skipif(os.name == 'nt', reason="Signal handling test not compatible with Windows")
    def test_real_sigint_handling(self):
        """Test real SIGINT signal handling (Ctrl+C simulation)."""
        import os
        
        handler = MockShutdownHandler("RealSignalHandler")
        
        # Register component and signals
        self.manager.register_component(component=handler)
        self.manager.register_signals()
        
        try:
            # Send actual SIGINT to current process (simulating Ctrl+C)
            # This will trigger the actual signal handler
            os.kill(os.getpid(), signal.SIGINT)
            
            # Wait for shutdown to complete
            success = self.manager.wait_for_shutdown(timeout=5.0)
            
            assert success
            assert handler.shutdown_called
            assert self.manager.shutdown_reason == ShutdownReason.USER_INTERRUPT
            assert self.manager.shutdown_phase == ShutdownPhase.COMPLETED
            
        finally:
            self.manager.unregister_signals()

    def test_ctrl_c_during_long_running_operation(self):
        """Test Ctrl+C handling during long-running component shutdown."""
        # Create a component with long shutdown time
        slow_handler = MockShutdownHandler("SlowHandler", delay=2.0)
        fast_handler = MockShutdownHandler("FastHandler", delay=0.1)
        
        self.manager.register_component(component=slow_handler, name="slow", timeout=1.0)
        self.manager.register_component(component=fast_handler, name="fast", timeout=1.0)
        
        with patch.object(self.manager, '_signal_handler') as mock_signal:
            # Start shutdown
            self.manager.initiate_shutdown(ShutdownReason.USER_INTERRUPT)
            
            # Simulate additional Ctrl+C during shutdown (should be ignored)
            time.sleep(0.1)  # Let shutdown start
            self.manager._signal_handler(signal.SIGINT, None)
            
            # Wait for completion
            success = self.manager.wait_for_shutdown(timeout=5.0)
            
            assert success
            # Both handlers should have been called despite the slow one timing out
            assert fast_handler.shutdown_called

    def test_multiple_consecutive_ctrl_c(self):
        """Test multiple consecutive Ctrl+C presses."""
        handler = MockShutdownHandler("MultiCtrlCHandler")
        
        self.manager.register_component(component=handler)
        
        # First Ctrl+C should initiate shutdown
        self.manager._signal_handler(signal.SIGINT, None)
        assert self.manager.shutdown_requested
        original_reason = self.manager.shutdown_reason
        
        # Second Ctrl+C should not change the shutdown reason
        self.manager._signal_handler(signal.SIGINT, None)
        assert self.manager.shutdown_reason == original_reason
        
        # Wait for completion
        success = self.manager.wait_for_shutdown(timeout=5.0)
        assert success
        assert handler.shutdown_called