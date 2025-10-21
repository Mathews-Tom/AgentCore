"""Security middleware for sandbox permission checks and audit logging."""

import functools
from collections.abc import Callable
from typing import Any

import structlog

from ..models.sandbox import (
    AuditEventType,
    AuditLogEntry,
    ResourceLimitExceededError,
    SandboxPermission,
    SecurityViolationError,
)
from ..services.audit_logger import AuditLogger
from ..services.sandbox_service import SandboxService

logger = structlog.get_logger()


class SecurityMiddleware:
    """Middleware for enforcing sandbox security policies."""

    def __init__(
        self,
        sandbox_service: SandboxService,
        audit_logger: AuditLogger,
    ) -> None:
        """
        Initialize security middleware.

        Args:
            sandbox_service: Sandbox service for permission checks
            audit_logger: Audit logger for security events
        """
        self._sandbox_service = sandbox_service
        self._audit_logger = audit_logger

    def require_permission(
        self,
        permission: SandboxPermission,
        resource_param: str = "resource",
    ) -> Callable[[Any], Any]:
        """
        Decorator to require permission before function execution.

        Args:
            permission: Required permission
            resource_param: Parameter name containing resource identifier

        Returns:
            Decorated function with permission check

        Example:
            @middleware.require_permission(SandboxPermission.WRITE, "file_path")
            async def write_file(sandbox_id: str, file_path: str, content: str):
                pass
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract sandbox_id and resource from arguments
                sandbox_id = kwargs.get("sandbox_id")
                if not sandbox_id and args:
                    # Try to get from first positional arg
                    sandbox_id = args[0]

                resource = kwargs.get(resource_param, "")

                if not sandbox_id:
                    raise ValueError("sandbox_id required for permission check")

                # Check permission
                try:
                    granted = await self._sandbox_service.check_permission(
                        sandbox_id=sandbox_id,
                        permission=permission,
                        resource=resource,
                    )

                    if not granted:
                        raise SecurityViolationError(
                            message=f"Permission {permission.value} denied for {resource}",
                            permission=permission,
                            resource=resource,
                        )

                    # Execute function
                    result = await func(*args, **kwargs)

                    # Log successful operation
                    config = self._sandbox_service._sandboxes.get(sandbox_id)
                    if config:
                        await self._audit_logger.log_event(
                            AuditLogEntry(
                                event_type=AuditEventType.RESOURCE_ACCESS,
                                sandbox_id=sandbox_id,
                                agent_id=config.agent_id,
                                operation=func.__name__,
                                resource=resource,
                                permission=permission,
                                result=True,
                            )
                        )

                    return result

                except SecurityViolationError:
                    # Already logged by check_permission
                    raise
                except Exception as e:
                    # Log unexpected errors
                    config = self._sandbox_service._sandboxes.get(sandbox_id)
                    if config:
                        await self._audit_logger.log_event(
                            AuditLogEntry(
                                event_type=AuditEventType.EXECUTION_ERROR,
                                sandbox_id=sandbox_id,
                                agent_id=config.agent_id,
                                operation=func.__name__,
                                resource=resource,
                                permission=permission,
                                result=False,
                                reason=str(e),
                                metadata={"error_type": type(e).__name__},
                            )
                        )
                    raise

            return wrapper

        return decorator

    def enforce_execution_limits(
        self,
    ) -> Callable[[Any], Any]:
        """
        Decorator to enforce execution limits before function execution.

        Returns:
            Decorated function with limit enforcement

        Example:
            @middleware.enforce_execution_limits()
            async def execute_task(sandbox_id: str, task: Task):
                pass
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract sandbox_id from arguments
                sandbox_id = kwargs.get("sandbox_id")
                if not sandbox_id and args:
                    sandbox_id = args[0]

                if not sandbox_id:
                    raise ValueError("sandbox_id required for limit enforcement")

                # Get sandbox config
                config = self._sandbox_service._sandboxes.get(sandbox_id)
                if not config:
                    raise KeyError(f"Sandbox {sandbox_id} not found")

                # Enforce limits
                try:
                    await self._sandbox_service.enforce_limits(
                        sandbox_id=sandbox_id,
                        limits=config.execution_limits,
                    )

                    # Execute function
                    result = await func(*args, **kwargs)

                    return result

                except ResourceLimitExceededError as e:
                    # Log limit violation
                    await self._audit_logger.log_event(
                        AuditLogEntry(
                            event_type=AuditEventType.LIMIT_EXCEEDED,
                            sandbox_id=sandbox_id,
                            agent_id=config.agent_id,
                            operation=func.__name__,
                            result=False,
                            reason=str(e),
                            metadata={
                                "limit_type": e.limit_type,
                                "current_value": e.current_value,
                                "max_value": e.max_value,
                            },
                        )
                    )
                    raise

            return wrapper

        return decorator

    def audit_operation(
        self,
        event_type: AuditEventType = AuditEventType.RESOURCE_ACCESS,
    ) -> Callable[[Any], Any]:
        """
        Decorator to automatically audit function execution.

        Args:
            event_type: Type of audit event

        Returns:
            Decorated function with automatic audit logging

        Example:
            @middleware.audit_operation(AuditEventType.EXECUTION_START)
            async def start_agent(sandbox_id: str):
                pass
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract sandbox_id from arguments
                sandbox_id = kwargs.get("sandbox_id")
                if not sandbox_id and args:
                    sandbox_id = args[0]

                config = (
                    self._sandbox_service._sandboxes.get(sandbox_id)
                    if sandbox_id
                    else None
                )

                try:
                    # Execute function
                    result = await func(*args, **kwargs)

                    # Log successful execution
                    if config:
                        await self._audit_logger.log_event(
                            AuditLogEntry(
                                event_type=event_type,
                                sandbox_id=sandbox_id,
                                agent_id=config.agent_id,
                                operation=func.__name__,
                                result=True,
                                metadata={
                                    "args_count": len(args),
                                    "kwargs_count": len(kwargs),
                                },
                            )
                        )

                    return result

                except Exception as e:
                    # Log failed execution
                    if config:
                        await self._audit_logger.log_event(
                            AuditLogEntry(
                                event_type=AuditEventType.EXECUTION_ERROR,
                                sandbox_id=sandbox_id,
                                agent_id=config.agent_id,
                                operation=func.__name__,
                                result=False,
                                reason=str(e),
                                metadata={"error_type": type(e).__name__},
                            )
                        )
                    raise

            return wrapper

        return decorator

    async def validate_resource_access(
        self,
        sandbox_id: str,
        operation: str,
        resource: str,
        required_permission: SandboxPermission,
    ) -> bool:
        """
        Validate resource access and log audit event.

        Args:
            sandbox_id: Sandbox identifier
            operation: Operation being performed
            resource: Resource being accessed
            required_permission: Permission required

        Returns:
            True if access granted

        Raises:
            SecurityViolationError: If access denied in strict mode
        """
        try:
            granted = await self._sandbox_service.check_permission(
                sandbox_id=sandbox_id,
                permission=required_permission,
                resource=resource,
            )

            # Log the access attempt
            config = self._sandbox_service._sandboxes.get(sandbox_id)
            if config:
                await self._audit_logger.log_event(
                    AuditLogEntry(
                        event_type=(
                            AuditEventType.PERMISSION_GRANTED
                            if granted
                            else AuditEventType.PERMISSION_DENIED
                        ),
                        sandbox_id=sandbox_id,
                        agent_id=config.agent_id,
                        operation=operation,
                        resource=resource,
                        permission=required_permission,
                        result=granted,
                    )
                )

            return granted

        except SecurityViolationError:
            raise
        except Exception as e:
            logger.error(
                "resource_access_validation_failed",
                sandbox_id=sandbox_id,
                operation=operation,
                resource=resource,
                error=str(e),
            )
            return False
