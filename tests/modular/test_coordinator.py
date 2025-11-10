"""
Tests for Module Coordinator

Validates module registration, discovery, message routing, context management,
and error handling in the module coordination layer.
"""

from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timezone

from agentcore.a2a_protocol.models.jsonrpc import (
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    create_success_response,
)
from agentcore.modular.coordinator import (
    ModuleCoordinator,
    ModuleCapability,
    ModuleMessage,
    CoordinationContext,
)
from agentcore.modular.models import ModuleType


class TestModuleCapability:
    """Test ModuleCapability model."""

    def test_module_capability_creation(self) -> None:
        """Test creating a module capability."""
        cap = ModuleCapability(
            module_type=ModuleType.PLANNER,
            methods=["plan.create", "plan.refine"],
            max_concurrent=5,
            timeout_seconds=60.0,
        )

        assert cap.module_type == ModuleType.PLANNER
        assert len(cap.methods) == 2
        assert "plan.create" in cap.methods
        assert cap.max_concurrent == 5
        assert cap.timeout_seconds == 60.0
        assert cap.module_id is not None

    def test_module_capability_defaults(self) -> None:
        """Test module capability default values."""
        cap = ModuleCapability(module_type=ModuleType.EXECUTOR)

        assert cap.methods == []
        assert cap.max_concurrent == 10
        assert cap.timeout_seconds == 30.0
        assert cap.metadata == {}


class TestModuleMessage:
    """Test ModuleMessage model."""

    def test_module_message_creation(self) -> None:
        """Test creating a module message."""
        request = JsonRpcRequest(
            method="test.method",
            params={"key": "value"},
            id="test-123",
        )

        message = ModuleMessage(
            from_module="module-1",
            to_module="module-2",
            request=request,
            timeout_seconds=45.0,
        )

        assert message.from_module == "module-1"
        assert message.to_module == "module-2"
        assert message.request.method == "test.method"
        assert message.timeout_seconds == 45.0
        assert message.message_id is not None
        assert message.created_at is not None


class TestCoordinationContext:
    """Test CoordinationContext model."""

    def test_coordination_context_creation(self) -> None:
        """Test creating coordination context."""
        context = CoordinationContext(
            execution_id="exec-123",
            plan_id="plan-456",
            session_id="session-789",
            iteration=3,
        )

        assert context.execution_id == "exec-123"
        assert context.plan_id == "plan-456"
        assert context.session_id == "session-789"
        assert context.iteration == 3
        assert context.trace_id is not None
        assert context.metadata == {}


class TestModuleCoordinator:
    """Test ModuleCoordinator class."""

    @pytest.fixture
    def coordinator(self) -> ModuleCoordinator:
        """Create a module coordinator instance."""
        return ModuleCoordinator()

    @pytest.fixture
    def planner_capability(self) -> ModuleCapability:
        """Create a planner module capability."""
        return ModuleCapability(
            module_type=ModuleType.PLANNER,
            methods=["plan.create", "plan.refine", "plan.get"],
            max_concurrent=5,
            timeout_seconds=30.0,
        )

    @pytest.fixture
    def executor_capability(self) -> ModuleCapability:
        """Create an executor module capability."""
        return ModuleCapability(
            module_type=ModuleType.EXECUTOR,
            methods=["execute.step", "execute.batch"],
            max_concurrent=10,
            timeout_seconds=60.0,
        )

    # ========================================================================
    # Module Registration Tests
    # ========================================================================

    def test_register_module(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test registering a module."""
        coordinator.register_module(planner_capability)

        # Check module is registered
        modules = coordinator.discover_modules()
        assert len(modules) == 1
        assert modules[0].module_type == ModuleType.PLANNER

        # Check methods are registered
        assert coordinator.find_module_for_method("plan.create") == planner_capability.module_id
        assert coordinator.find_module_for_method("plan.refine") == planner_capability.module_id

    def test_register_duplicate_module_fails(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test that registering duplicate module fails."""
        coordinator.register_module(planner_capability)

        with pytest.raises(ValueError, match="already registered"):
            coordinator.register_module(planner_capability)

    def test_register_multiple_modules(
        self,
        coordinator: ModuleCoordinator,
        planner_capability: ModuleCapability,
        executor_capability: ModuleCapability,
    ) -> None:
        """Test registering multiple modules."""
        coordinator.register_module(planner_capability)
        coordinator.register_module(executor_capability)

        modules = coordinator.discover_modules()
        assert len(modules) == 2

        # Check both module types present
        module_types = {m.module_type for m in modules}
        assert ModuleType.PLANNER in module_types
        assert ModuleType.EXECUTOR in module_types

    def test_unregister_module(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test unregistering a module."""
        coordinator.register_module(planner_capability)
        module_id = planner_capability.module_id

        # Unregister
        coordinator.unregister_module(module_id)

        # Check module removed
        modules = coordinator.discover_modules()
        assert len(modules) == 0

        # Check methods removed
        assert coordinator.find_module_for_method("plan.create") is None

    def test_unregister_nonexistent_module_fails(
        self, coordinator: ModuleCoordinator
    ) -> None:
        """Test unregistering non-existent module fails."""
        with pytest.raises(ValueError, match="not registered"):
            coordinator.unregister_module("nonexistent-module")

    # ========================================================================
    # Module Discovery Tests
    # ========================================================================

    def test_discover_all_modules(
        self,
        coordinator: ModuleCoordinator,
        planner_capability: ModuleCapability,
        executor_capability: ModuleCapability,
    ) -> None:
        """Test discovering all modules."""
        coordinator.register_module(planner_capability)
        coordinator.register_module(executor_capability)

        modules = coordinator.discover_modules()
        assert len(modules) == 2

    def test_discover_modules_by_type(
        self,
        coordinator: ModuleCoordinator,
        planner_capability: ModuleCapability,
        executor_capability: ModuleCapability,
    ) -> None:
        """Test discovering modules filtered by type."""
        coordinator.register_module(planner_capability)
        coordinator.register_module(executor_capability)

        # Find planners
        planners = coordinator.discover_modules(ModuleType.PLANNER)
        assert len(planners) == 1
        assert planners[0].module_type == ModuleType.PLANNER

        # Find executors
        executors = coordinator.discover_modules(ModuleType.EXECUTOR)
        assert len(executors) == 1
        assert executors[0].module_type == ModuleType.EXECUTOR

        # Find verifiers (none registered)
        verifiers = coordinator.discover_modules(ModuleType.VERIFIER)
        assert len(verifiers) == 0

    def test_find_module_for_method(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test finding module for a method."""
        coordinator.register_module(planner_capability)

        # Find existing methods
        assert coordinator.find_module_for_method("plan.create") == planner_capability.module_id
        assert coordinator.find_module_for_method("plan.refine") == planner_capability.module_id
        assert coordinator.find_module_for_method("plan.get") == planner_capability.module_id

        # Non-existent method
        assert coordinator.find_module_for_method("nonexistent.method") is None

    # ========================================================================
    # Context Management Tests
    # ========================================================================

    def test_set_and_get_context(self, coordinator: ModuleCoordinator) -> None:
        """Test setting and getting coordination context."""
        context = CoordinationContext(
            execution_id="exec-123",
            plan_id="plan-456",
            session_id="session-789",
        )

        coordinator.set_context(context)

        retrieved = coordinator.get_context()
        assert retrieved is not None
        assert retrieved.execution_id == "exec-123"
        assert retrieved.plan_id == "plan-456"
        assert retrieved.session_id == "session-789"

    def test_clear_context(self, coordinator: ModuleCoordinator) -> None:
        """Test clearing coordination context."""
        context = CoordinationContext(
            execution_id="exec-123",
            plan_id="plan-456",
        )

        coordinator.set_context(context)
        assert coordinator.get_context() is not None

        coordinator.clear_context()
        assert coordinator.get_context() is None

    # ========================================================================
    # Message Routing Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_send_request_with_explicit_target(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test sending request with explicit target module."""
        coordinator.register_module(planner_capability)

        # Send request (will timeout since no actual handler)
        response = await coordinator.send_request(
            from_module="test-module",
            method="plan.create",
            params={"query": "test query"},
            to_module=planner_capability.module_id,
            timeout=0.1,  # Short timeout for test
        )

        # Should get timeout error
        assert response.error is not None
        assert response.error.code == JsonRpcErrorCode.INTERNAL_ERROR
        assert "timeout" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_send_request_with_auto_discovery(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test sending request with automatic module discovery."""
        coordinator.register_module(planner_capability)

        # Send request without specifying target (auto-discover)
        response = await coordinator.send_request(
            from_module="test-module",
            method="plan.create",
            params={"query": "test query"},
            timeout=0.1,
        )

        # Should get timeout error (but successfully discovered target)
        assert response.error is not None
        assert response.error.code == JsonRpcErrorCode.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_send_request_method_not_found(
        self, coordinator: ModuleCoordinator
    ) -> None:
        """Test sending request for non-existent method."""
        response = await coordinator.send_request(
            from_module="test-module",
            method="nonexistent.method",
            params={},
        )

        assert response.error is not None
        assert response.error.code == JsonRpcErrorCode.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_send_request_module_not_registered(
        self, coordinator: ModuleCoordinator
    ) -> None:
        """Test sending request to unregistered module."""
        response = await coordinator.send_request(
            from_module="test-module",
            method="test.method",
            params={},
            to_module="nonexistent-module",
        )

        assert response.error is not None
        assert response.error.code == JsonRpcErrorCode.INTERNAL_ERROR
        assert "not registered" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_send_notification(
        self, coordinator: ModuleCoordinator, planner_capability: ModuleCapability
    ) -> None:
        """Test sending notification (no response expected)."""
        coordinator.register_module(planner_capability)

        # Should not raise exception
        await coordinator.send_notification(
            from_module="test-module",
            method="plan.status",
            params={"status": "completed"},
            to_module=planner_capability.module_id,
        )

    @pytest.mark.asyncio
    async def test_send_broadcast_notification(
        self, coordinator: ModuleCoordinator
    ) -> None:
        """Test sending broadcast notification."""
        # Should not raise exception
        await coordinator.send_notification(
            from_module="test-module",
            method="system.shutdown",
            params={},
            to_module=None,  # Broadcast
        )

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_handle_error(self, coordinator: ModuleCoordinator) -> None:
        """Test error handling."""
        # Create pending request
        future: asyncio.Future[JsonRpcResponse] = asyncio.Future()
        message_id = "test-message-123"
        coordinator._pending_requests[message_id] = future

        # Handle error
        error = ValueError("Test error")
        coordinator.handle_error(message_id, error, request_id="req-123")

        # Check future was resolved with error response
        assert future.done()
        response = future.result()
        assert response.error is not None
        assert response.error.code == JsonRpcErrorCode.INTERNAL_ERROR
        assert "Test error" in response.error.message

    def test_receive_response(self, coordinator: ModuleCoordinator) -> None:
        """Test receiving response for pending request."""
        # Create pending request
        future: asyncio.Future[JsonRpcResponse] = asyncio.Future()
        message_id = "test-message-123"
        coordinator._pending_requests[message_id] = future

        # Receive response
        response = create_success_response(request_id="req-123", result={"data": "test"})
        coordinator.receive_response(message_id, response)

        # Check future was resolved
        assert future.done()
        result = future.result()
        assert result.result == {"data": "test"}

    # ========================================================================
    # Status & Monitoring Tests
    # ========================================================================

    def test_get_status_empty(self, coordinator: ModuleCoordinator) -> None:
        """Test getting status with no modules registered."""
        status = coordinator.get_status()

        assert status["registered_modules"] == 0
        assert status["registered_methods"] == 0
        assert status["pending_requests"] == 0
        assert status["has_context"] is False
        assert status["modules"] == []

    def test_get_status_with_modules(
        self,
        coordinator: ModuleCoordinator,
        planner_capability: ModuleCapability,
        executor_capability: ModuleCapability,
    ) -> None:
        """Test getting status with registered modules."""
        coordinator.register_module(planner_capability)
        coordinator.register_module(executor_capability)

        status = coordinator.get_status()

        assert status["registered_modules"] == 2
        assert status["registered_methods"] == 5  # 3 planner + 2 executor methods
        assert len(status["modules"]) == 2

        # Check module details
        module_types = {m["module_type"] for m in status["modules"]}
        assert ModuleType.PLANNER in module_types
        assert ModuleType.EXECUTOR in module_types

    def test_get_status_with_context(self, coordinator: ModuleCoordinator) -> None:
        """Test getting status with coordination context set."""
        context = CoordinationContext(
            execution_id="exec-123",
            plan_id="plan-456",
        )
        coordinator.set_context(context)

        status = coordinator.get_status()
        assert status["has_context"] is True
