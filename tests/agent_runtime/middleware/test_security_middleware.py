"""Tests for security middleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.agent_runtime.middleware.security_middleware import SecurityMiddleware
from agentcore.agent_runtime.models.sandbox import (
    AuditEventType,
    AuditLogEntry,
    ExecutionLimits,
    ResourceLimitExceededError,
    SandboxConfig,
    SandboxPermission,
    SecurityViolationError,
)
from agentcore.agent_runtime.services.audit_logger import AuditLogger
from agentcore.agent_runtime.services.sandbox_service import SandboxService


@pytest.fixture
def mock_sandbox_service() -> SandboxService:
    """Create mock sandbox service."""
    service = MagicMock(spec=SandboxService)
    service.check_permission = AsyncMock(return_value=True)
    service.enforce_limits = AsyncMock()
    service._sandboxes = {
        "sandbox-001": SandboxConfig(
            agent_id="test-agent-001",
            sandbox_id="sandbox-001",
            allow_network=False,
            execution_limits=ExecutionLimits(
                max_memory_mb=512,
                max_cpu_percent=50.0,
                max_execution_time_seconds=60,
                max_processes=100,
            ),
        )
    }
    return service


@pytest.fixture
def mock_audit_logger() -> AuditLogger:
    """Create mock audit logger."""
    logger = MagicMock(spec=AuditLogger)
    logger.log_event = AsyncMock()
    return logger


@pytest.fixture
def security_middleware(mock_sandbox_service: SandboxService, mock_audit_logger: AuditLogger) -> SecurityMiddleware:
    """Create security middleware with mocked dependencies."""
    return SecurityMiddleware(
        sandbox_service=mock_sandbox_service,
        audit_logger=mock_audit_logger,
    )


@pytest.mark.asyncio
class TestSecurityMiddleware:
    """Test security middleware functionality."""

    async def test_require_permission_granted(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test permission decorator when permission is granted."""

        @security_middleware.require_permission(SandboxPermission.WRITE, "file_path")
        async def write_file(sandbox_id: str, file_path: str, content: str) -> str:
            return f"Written: {content}"

        result = await write_file(
            sandbox_id="sandbox-001",
            file_path="/tmp/test.txt",
            content="test content",
        )

        assert result == "Written: test content"
        mock_sandbox_service.check_permission.assert_called_once_with(
            sandbox_id="sandbox-001",
            permission=SandboxPermission.WRITE,
            resource="/tmp/test.txt",
        )
        mock_audit_logger.log_event.assert_called_once()

    async def test_require_permission_denied(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
    ) -> None:
        """Test permission decorator when permission is denied."""
        mock_sandbox_service.check_permission = AsyncMock(return_value=False)

        @security_middleware.require_permission(SandboxPermission.WRITE, "file_path")
        async def write_file(sandbox_id: str, file_path: str, content: str) -> str:
            return f"Written: {content}"

        with pytest.raises(SecurityViolationError, match="Permission write denied"):
            await write_file(
                sandbox_id="sandbox-001",
                file_path="/tmp/test.txt",
                content="test content",
            )

    async def test_require_permission_no_sandbox_id(
        self,
        security_middleware: SecurityMiddleware,
    ) -> None:
        """Test permission decorator without sandbox_id."""

        @security_middleware.require_permission(SandboxPermission.READ)
        async def read_file(file_path: str) -> str:
            return "content"

        with pytest.raises(ValueError, match="sandbox_id required"):
            await read_file(file_path="/tmp/test.txt")

    async def test_require_permission_with_positional_args(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
    ) -> None:
        """Test permission decorator with positional arguments."""

        @security_middleware.require_permission(SandboxPermission.READ, "file_path")
        async def read_file(sandbox_id: str, file_path: str) -> str:
            return "content"

        result = await read_file("sandbox-001", "/tmp/test.txt")

        assert result == "content"
        mock_sandbox_service.check_permission.assert_called_once()

    async def test_require_permission_exception_handling(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test permission decorator with function exception."""

        @security_middleware.require_permission(SandboxPermission.EXECUTE, "command")
        async def execute_command(sandbox_id: str, command: str) -> None:
            raise RuntimeError("Execution failed")

        with pytest.raises(RuntimeError, match="Execution failed"):
            await execute_command(sandbox_id="sandbox-001", command="test")

        # Should log the error
        mock_audit_logger.log_event.assert_called()
        call_args = mock_audit_logger.log_event.call_args[0][0]
        assert call_args.event_type == AuditEventType.EXECUTION_ERROR

    async def test_enforce_execution_limits_success(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
    ) -> None:
        """Test execution limits decorator with no violations."""

        @security_middleware.enforce_execution_limits()
        async def execute_task(sandbox_id: str, task: str) -> str:
            return f"Executed: {task}"

        result = await execute_task(sandbox_id="sandbox-001", task="test_task")

        assert result == "Executed: test_task"
        mock_sandbox_service.enforce_limits.assert_called_once()

    async def test_enforce_execution_limits_exceeded(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test execution limits decorator when limits exceeded."""
        mock_sandbox_service.enforce_limits = AsyncMock(
            side_effect=ResourceLimitExceededError(
                message="Memory limit exceeded",
                limit_type="memory",
                current_value=600,
                max_value=512,
            )
        )

        @security_middleware.enforce_execution_limits()
        async def execute_task(sandbox_id: str, task: str) -> str:
            return f"Executed: {task}"

        with pytest.raises(ResourceLimitExceededError):
            await execute_task(sandbox_id="sandbox-001", task="test_task")

        # Should log the limit violation
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[0][0]
        assert call_args.event_type == AuditEventType.LIMIT_EXCEEDED

    async def test_enforce_execution_limits_no_sandbox_id(
        self,
        security_middleware: SecurityMiddleware,
    ) -> None:
        """Test execution limits decorator without sandbox_id."""

        @security_middleware.enforce_execution_limits()
        async def execute_task(task: str) -> str:
            return f"Executed: {task}"

        with pytest.raises(ValueError, match="sandbox_id required"):
            await execute_task(task="test_task")

    async def test_enforce_execution_limits_sandbox_not_found(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
    ) -> None:
        """Test execution limits decorator with non-existent sandbox."""
        mock_sandbox_service._sandboxes = {}

        @security_middleware.enforce_execution_limits()
        async def execute_task(sandbox_id: str, task: str) -> str:
            return f"Executed: {task}"

        with pytest.raises(KeyError, match="Sandbox .* not found"):
            await execute_task(sandbox_id="nonexistent", task="test_task")

    async def test_audit_operation_success(
        self,
        security_middleware: SecurityMiddleware,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test audit operation decorator on success."""

        @security_middleware.audit_operation(AuditEventType.EXECUTION_START)
        async def start_agent(sandbox_id: str) -> str:
            return "started"

        result = await start_agent(sandbox_id="sandbox-001")

        assert result == "started"
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[0][0]
        assert call_args.event_type == AuditEventType.EXECUTION_START
        assert call_args.result is True

    async def test_audit_operation_failure(
        self,
        security_middleware: SecurityMiddleware,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test audit operation decorator on failure."""

        @security_middleware.audit_operation(AuditEventType.EXECUTION_START)
        async def start_agent(sandbox_id: str) -> None:
            raise RuntimeError("Start failed")

        with pytest.raises(RuntimeError, match="Start failed"):
            await start_agent(sandbox_id="sandbox-001")

        # Should log the failure
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[0][0]
        assert call_args.event_type == AuditEventType.EXECUTION_ERROR
        assert call_args.result is False

    async def test_audit_operation_without_sandbox(
        self,
        security_middleware: SecurityMiddleware,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test audit operation decorator without sandbox_id."""

        @security_middleware.audit_operation()
        async def global_operation() -> str:
            return "done"

        result = await global_operation()

        assert result == "done"
        # Should not log if no sandbox_id
        mock_audit_logger.log_event.assert_not_called()

    async def test_validate_resource_access_granted(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test resource access validation when granted."""
        granted = await security_middleware.validate_resource_access(
            sandbox_id="sandbox-001",
            operation="read_file",
            resource="/tmp/test.txt",
            required_permission=SandboxPermission.READ,
        )

        assert granted is True
        mock_sandbox_service.check_permission.assert_called_once()
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[0][0]
        assert call_args.event_type == AuditEventType.PERMISSION_GRANTED

    async def test_validate_resource_access_denied(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
        mock_audit_logger: AuditLogger,
    ) -> None:
        """Test resource access validation when denied."""
        mock_sandbox_service.check_permission = AsyncMock(return_value=False)

        granted = await security_middleware.validate_resource_access(
            sandbox_id="sandbox-001",
            operation="write_file",
            resource="/etc/passwd",
            required_permission=SandboxPermission.WRITE,
        )

        assert granted is False
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[0][0]
        assert call_args.event_type == AuditEventType.PERMISSION_DENIED

    async def test_validate_resource_access_security_violation(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
    ) -> None:
        """Test resource access validation with security violation."""
        mock_sandbox_service.check_permission = AsyncMock(
            side_effect=SecurityViolationError(
                message="Security violation",
                permission=SandboxPermission.NETWORK,
                resource="https://evil.com",
            )
        )

        with pytest.raises(SecurityViolationError):
            await security_middleware.validate_resource_access(
                sandbox_id="sandbox-001",
                operation="network_request",
                resource="https://evil.com",
                required_permission=SandboxPermission.NETWORK,
            )

    async def test_validate_resource_access_exception(
        self,
        security_middleware: SecurityMiddleware,
        mock_sandbox_service: SandboxService,
    ) -> None:
        """Test resource access validation with unexpected exception."""
        mock_sandbox_service.check_permission = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        granted = await security_middleware.validate_resource_access(
            sandbox_id="sandbox-001",
            operation="test_operation",
            resource="test_resource",
            required_permission=SandboxPermission.READ,
        )

        # Should return False on exception
        assert granted is False
