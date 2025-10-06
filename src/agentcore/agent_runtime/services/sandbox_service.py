"""Sandbox security service for isolated agent code execution."""

import asyncio
import fnmatch
import resource
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr


def guarded_inplacevar(op, x, y):
    """Guard for augmented assignment operations (+=, -=, etc)."""
    return op(x, y)


from ..models.sandbox import (
    AuditEventType,
    AuditLogEntry,
    ExecutionLimits,
    ResourceLimitExceededError,
    ResourcePolicy,
    SandboxConfig,
    SandboxPermission,
    SandboxStats,
    SecurityViolationError,
)
from .audit_logger import AuditLogger

logger = structlog.get_logger()


class SandboxService:
    """Service for managing isolated sandbox environments."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        workspace_root: Path,
    ) -> None:
        """
        Initialize sandbox service.

        Args:
            audit_logger: Audit logger for security events
            workspace_root: Root directory for sandbox workspaces
        """
        self._audit_logger = audit_logger
        self._workspace_root = workspace_root
        self._sandboxes: dict[str, SandboxConfig] = {}
        self._stats: dict[str, SandboxStats] = {}
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}

        # Ensure workspace root exists
        self._workspace_root.mkdir(parents=True, exist_ok=True)

    async def create_sandbox(self, config: SandboxConfig) -> str:
        """
        Create a new isolated sandbox environment.

        Args:
            config: Sandbox configuration

        Returns:
            Sandbox ID

        Raises:
            ValueError: If sandbox already exists
        """
        if config.sandbox_id in self._sandboxes:
            raise ValueError(f"Sandbox {config.sandbox_id} already exists")

        # Store configuration
        self._sandboxes[config.sandbox_id] = config

        # Initialize stats
        self._stats[config.sandbox_id] = SandboxStats(
            sandbox_id=config.sandbox_id,
        )

        # Create workspace directory
        workspace_path = self._get_workspace_path(config.sandbox_id)
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create writable paths
        for writable_path in config.writable_paths:
            path = workspace_path / writable_path.lstrip("/")
            path.mkdir(parents=True, exist_ok=True)

        # Log creation
        await self._log_audit_event(
            AuditLogEntry(
                event_type=AuditEventType.SANDBOX_CREATED,
                sandbox_id=config.sandbox_id,
                agent_id=config.agent_id,
                operation="create_sandbox",
                result=True,
                metadata={
                    "permissions": [p.value for p in config.permissions],
                    "strict_mode": config.strict_mode,
                },
            )
        )

        logger.info(
            "sandbox_created",
            sandbox_id=config.sandbox_id,
            agent_id=config.agent_id,
            strict_mode=config.strict_mode,
        )

        return config.sandbox_id

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """
        Destroy a sandbox and cleanup resources.

        Args:
            sandbox_id: Sandbox identifier

        Raises:
            KeyError: If sandbox not found
        """
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox {sandbox_id} not found")

        config = self._sandboxes[sandbox_id]

        # Cancel any running tasks
        if sandbox_id in self._running_tasks:
            task = self._running_tasks[sandbox_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._running_tasks[sandbox_id]

        # Cleanup workspace (optional - could keep for forensics)
        # workspace_path = self._get_workspace_path(sandbox_id)
        # if workspace_path.exists():
        #     shutil.rmtree(workspace_path)

        # Remove from tracking
        del self._sandboxes[sandbox_id]
        if sandbox_id in self._stats:
            del self._stats[sandbox_id]

        # Log destruction
        await self._log_audit_event(
            AuditLogEntry(
                event_type=AuditEventType.SANDBOX_DESTROYED,
                sandbox_id=sandbox_id,
                agent_id=config.agent_id,
                operation="destroy_sandbox",
                result=True,
            )
        )

        logger.info(
            "sandbox_destroyed",
            sandbox_id=sandbox_id,
            agent_id=config.agent_id,
        )

    async def execute_in_sandbox(
        self,
        sandbox_id: str,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute code in isolated sandbox environment.

        Args:
            sandbox_id: Sandbox identifier
            code: Python code to execute
            context: Execution context variables

        Returns:
            Execution result

        Raises:
            KeyError: If sandbox not found
            SecurityViolationError: If code violates security policy
            ResourceLimitExceededError: If resource limit exceeded
        """
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox {sandbox_id} not found")

        config = self._sandboxes[sandbox_id]
        stats = self._stats[sandbox_id]
        context = context or {}

        # Log execution start
        await self._log_audit_event(
            AuditLogEntry(
                event_type=AuditEventType.EXECUTION_START,
                sandbox_id=sandbox_id,
                agent_id=config.agent_id,
                operation="execute_code",
                result=True,
                metadata={"code_length": len(code)},
            )
        )

        # Mark as running
        stats.is_running = True
        start_time = time.time()

        try:
            # Execute with timeout and resource limits
            result = await asyncio.wait_for(
                self._execute_restricted_code(
                    sandbox_id=sandbox_id,
                    code=code,
                    context=context,
                ),
                timeout=config.execution_limits.max_execution_time_seconds,
            )

            # Update stats
            execution_time = time.time() - start_time
            stats.execution_time_seconds = execution_time
            stats.is_running = False

            # Log success
            await self._log_audit_event(
                AuditLogEntry(
                    event_type=AuditEventType.EXECUTION_COMPLETE,
                    sandbox_id=sandbox_id,
                    agent_id=config.agent_id,
                    operation="execute_code",
                    result=True,
                    metadata={
                        "execution_time": execution_time,
                        "result_type": type(result).__name__,
                    },
                )
            )

            return result

        except asyncio.TimeoutError as e:
            stats.is_running = False

            # Log timeout
            await self._log_audit_event(
                AuditLogEntry(
                    event_type=AuditEventType.EXECUTION_TIMEOUT,
                    sandbox_id=sandbox_id,
                    agent_id=config.agent_id,
                    operation="execute_code",
                    result=False,
                    reason=f"Exceeded {config.execution_limits.max_execution_time_seconds}s limit",
                )
            )

            raise ResourceLimitExceededError(
                message="Execution time limit exceeded",
                limit_type="execution_time",
                current_value=time.time() - start_time,
                max_value=config.execution_limits.max_execution_time_seconds,
            ) from e

        except Exception as e:
            stats.is_running = False

            # Log error
            await self._log_audit_event(
                AuditLogEntry(
                    event_type=AuditEventType.EXECUTION_ERROR,
                    sandbox_id=sandbox_id,
                    agent_id=config.agent_id,
                    operation="execute_code",
                    result=False,
                    reason=str(e),
                    metadata={"error_type": type(e).__name__},
                )
            )

            raise

    async def check_permission(
        self,
        sandbox_id: str,
        permission: SandboxPermission,
        resource: str = "",
    ) -> bool:
        """
        Check if sandbox has permission for operation on resource.

        Args:
            sandbox_id: Sandbox identifier
            permission: Permission to check
            resource: Resource being accessed

        Returns:
            True if permission granted, False otherwise

        Raises:
            KeyError: If sandbox not found
            SecurityViolationError: If permission denied in strict mode
        """
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox {sandbox_id} not found")

        config = self._sandboxes[sandbox_id]

        # Check resource policies first (more specific)
        for policy in config.resource_policies:
            if self._matches_pattern(resource, policy.resource_pattern):
                # Check explicit denials
                if permission in policy.denied_permissions:
                    result = False
                    reason = f"Explicitly denied by policy: {policy.description}"
                # Check explicit grants
                elif permission in policy.allowed_permissions:
                    result = True
                    reason = f"Granted by policy: {policy.description}"
                else:
                    # Not mentioned in policy, fall through to global check
                    continue

                # Log the decision
                await self._log_audit_event(
                    AuditLogEntry(
                        event_type=(
                            AuditEventType.PERMISSION_GRANTED
                            if result
                            else AuditEventType.PERMISSION_DENIED
                        ),
                        sandbox_id=sandbox_id,
                        agent_id=config.agent_id,
                        operation="check_permission",
                        resource=resource,
                        permission=permission,
                        result=result,
                        reason=reason,
                    )
                )

                if not result and config.strict_mode:
                    raise SecurityViolationError(
                        message=reason,
                        permission=permission,
                        resource=resource,
                    )

                return result

        # Check global permissions
        result = permission in config.permissions

        # Special checks
        if permission == SandboxPermission.NETWORK and not config.allow_network:
            result = False
            reason = "Network access disabled"
        elif (
            permission == SandboxPermission.NETWORK
            and config.allowed_hosts
            and resource not in config.allowed_hosts
        ):
            result = False
            reason = f"Host {resource} not in allowed hosts"
        else:
            reason = (
                "Granted by global permissions"
                if result
                else "Not in global permissions"
            )

        # Log the decision
        await self._log_audit_event(
            AuditLogEntry(
                event_type=(
                    AuditEventType.PERMISSION_GRANTED
                    if result
                    else AuditEventType.PERMISSION_DENIED
                ),
                sandbox_id=sandbox_id,
                agent_id=config.agent_id,
                operation="check_permission",
                resource=resource,
                permission=permission,
                result=result,
                reason=reason,
            )
        )

        if not result and config.strict_mode:
            raise SecurityViolationError(
                message=reason,
                permission=permission,
                resource=resource,
            )

        return result

    async def enforce_limits(
        self,
        sandbox_id: str,
        limits: ExecutionLimits,
    ) -> None:
        """
        Enforce resource limits on sandbox.

        Args:
            sandbox_id: Sandbox identifier
            limits: Execution limits to enforce

        Raises:
            KeyError: If sandbox not found
        """
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox {sandbox_id} not found")

        # Set process resource limits using resource module
        try:
            # Memory limit (RSS)
            memory_bytes = limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_RSS, (memory_bytes, memory_bytes))

            # CPU time limit
            cpu_time = limits.max_execution_time_seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time, cpu_time))

            # File descriptor limit
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (limits.max_file_descriptors, limits.max_file_descriptors),
            )

            # Process limit
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (limits.max_processes, limits.max_processes),
            )

            logger.debug(
                "resource_limits_enforced",
                sandbox_id=sandbox_id,
                memory_mb=limits.max_memory_mb,
                cpu_time=limits.max_execution_time_seconds,
            )

        except Exception as e:
            logger.error(
                "resource_limit_enforcement_failed",
                sandbox_id=sandbox_id,
                error=str(e),
            )

    def get_stats(self, sandbox_id: str) -> SandboxStats:
        """
        Get real-time statistics for sandbox.

        Args:
            sandbox_id: Sandbox identifier

        Returns:
            Sandbox statistics

        Raises:
            KeyError: If sandbox not found
        """
        if sandbox_id not in self._stats:
            raise KeyError(f"Sandbox {sandbox_id} not found")

        return self._stats[sandbox_id]

    async def _execute_restricted_code(
        self,
        sandbox_id: str,
        code: str,
        context: dict[str, Any],
    ) -> Any:
        """
        Execute code with RestrictedPython.

        Args:
            sandbox_id: Sandbox identifier
            code: Python code to execute
            context: Execution context

        Returns:
            Execution result
        """
        config = self._sandboxes[sandbox_id]

        # Compile with RestrictedPython
        try:
            byte_code = compile_restricted(
                code,
                filename=f"<sandbox-{sandbox_id}>",
                mode="exec",
            )
        except SyntaxError as e:
            raise SecurityViolationError(
                message=f"Code compilation failed: {e}",
            ) from e

        # Build safe execution environment
        safe_context = {
            **safe_globals,
            "_getiter_": iter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_getattr_": safer_getattr,
            "_inplacevar_": guarded_inplacevar,
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            },
            **context,
        }

        # Execute in isolated environment
        result_namespace: dict[str, Any] = {}

        try:
            exec(byte_code, safe_context, result_namespace)
        except Exception as e:
            raise SecurityViolationError(
                message=f"Code execution failed: {e}",
            ) from e

        # Return result (if any)
        return result_namespace.get("result", None)

    async def _log_audit_event(self, entry: AuditLogEntry) -> None:
        """Log audit event to audit logger."""
        await self._audit_logger.log_event(entry)

    def _get_workspace_path(self, sandbox_id: str) -> Path:
        """Get workspace path for sandbox."""
        return self._workspace_root / f"sandbox-{sandbox_id}"

    @staticmethod
    def _matches_pattern(resource: str, pattern: str) -> bool:
        """
        Check if resource matches glob pattern.

        Args:
            resource: Resource path/name
            pattern: Glob pattern

        Returns:
            True if matches
        """
        return fnmatch.fnmatch(resource, pattern)
