"""
Modular Agent Core Security Module

Provides authentication, authorization (RBAC), and audit logging for
inter-module communication following NFR-4 requirements:
- NFR-4.1: All inter-module communication uses authenticated channels
- NFR-4.2: Module access controlled through RBAC policies
- NFR-4.3: All module interactions auditable with trace IDs
- NFR-4.4: Sensitive data not logged in module communication
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import deque
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.security import (
    Permission,
    Role,
    TokenPayload,
    TokenType,
)
from agentcore.a2a_protocol.services.security_service import security_service
from agentcore.modular.models import ModuleType

logger = structlog.get_logger()


# ============================================================================
# Modular-Specific Permissions
# ============================================================================


class ModularPermission(str, Enum):
    """Permissions specific to modular agent operations."""

    # Execution permissions
    MODULAR_EXECUTE = "modular:execute"  # Execute modular.solve
    MODULAR_READ = "modular:read"  # Read execution results
    MODULAR_CANCEL = "modular:cancel"  # Cancel running executions

    # Module permissions
    MODULE_PLANNER = "module:planner"  # Access planner module
    MODULE_EXECUTOR = "module:executor"  # Access executor module
    MODULE_VERIFIER = "module:verifier"  # Access verifier module
    MODULE_GENERATOR = "module:generator"  # Access generator module

    # Internal module-to-module communication
    MODULE_INTERNAL = "module:internal"  # Internal module communication

    # Admin permissions
    MODULAR_ADMIN = "modular:admin"  # Full modular system access


# Role-based modular permission mapping
MODULAR_ROLE_PERMISSIONS: dict[Role, list[ModularPermission]] = {
    Role.AGENT: [
        ModularPermission.MODULAR_EXECUTE,
        ModularPermission.MODULAR_READ,
    ],
    Role.SERVICE: [
        ModularPermission.MODULAR_EXECUTE,
        ModularPermission.MODULAR_READ,
        ModularPermission.MODULAR_CANCEL,
        ModularPermission.MODULE_PLANNER,
        ModularPermission.MODULE_EXECUTOR,
        ModularPermission.MODULE_VERIFIER,
        ModularPermission.MODULE_GENERATOR,
        ModularPermission.MODULE_INTERNAL,
    ],
    Role.ADMIN: [
        ModularPermission.MODULAR_ADMIN,
    ],
}

# Module type to permission mapping
MODULE_TYPE_PERMISSIONS: dict[ModuleType, ModularPermission] = {
    ModuleType.PLANNER: ModularPermission.MODULE_PLANNER,
    ModuleType.EXECUTOR: ModularPermission.MODULE_EXECUTOR,
    ModuleType.VERIFIER: ModularPermission.MODULE_VERIFIER,
    ModuleType.GENERATOR: ModularPermission.MODULE_GENERATOR,
}


# ============================================================================
# Audit Log Models
# ============================================================================


class AuditAction(str, Enum):
    """Types of auditable actions."""

    # Execution lifecycle
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_CANCELLED = "execution.cancelled"

    # Module transitions
    MODULE_INVOKED = "module.invoked"
    MODULE_COMPLETED = "module.completed"
    MODULE_FAILED = "module.failed"

    # Security events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    PERMISSION_DENIED = "permission.denied"
    RATE_LIMITED = "rate.limited"

    # Plan lifecycle
    PLAN_CREATED = "plan.created"
    PLAN_REFINED = "plan.refined"

    # Verification
    VERIFICATION_PASSED = "verification.passed"
    VERIFICATION_FAILED = "verification.failed"


class AuditLogEntry(BaseModel):
    """Audit log entry for module interactions."""

    entry_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique entry ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Entry timestamp"
    )
    trace_id: str = Field(..., description="Distributed trace ID")
    execution_id: str | None = Field(None, description="Execution ID if applicable")
    action: AuditAction = Field(..., description="Audit action type")
    actor: str = Field(..., description="Actor (agent/module/user ID)")
    target: str | None = Field(None, description="Target (module/resource)")
    module_type: ModuleType | None = Field(None, description="Module type if applicable")
    success: bool = Field(True, description="Action success status")
    duration_ms: int | None = Field(None, description="Action duration in ms")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (sanitized)"
    )
    error: str | None = Field(None, description="Error message if failed")


class AuditLogConfig(BaseModel):
    """Configuration for audit logging."""

    enabled: bool = Field(default=True, description="Enable audit logging")
    max_entries: int = Field(
        default=10000, description="Maximum in-memory entries before rotation"
    )
    retention_hours: int = Field(
        default=168, description="Hours to retain logs (default: 7 days)"
    )
    log_sensitive_actions: bool = Field(
        default=True, description="Log sensitive actions (auth failures, etc.)"
    )
    redact_patterns: list[str] = Field(
        default_factory=lambda: [
            r"api[_-]?key",
            r"secret",
            r"password",
            r"token",
            r"auth",
            r"credential",
            r"bearer",
            r"jwt",
        ],
        description="Regex patterns for fields to redact",
    )


# ============================================================================
# Security Context
# ============================================================================


class ModularSecurityContext(BaseModel):
    """Security context for modular execution."""

    trace_id: str = Field(..., description="Distributed trace ID")
    execution_id: str | None = Field(None, description="Execution ID")
    authenticated: bool = Field(False, description="Authentication status")
    actor_id: str | None = Field(None, description="Authenticated actor ID")
    role: Role | None = Field(None, description="Actor role")
    permissions: list[ModularPermission] = Field(
        default_factory=list, description="Granted permissions"
    )
    token_payload: TokenPayload | None = Field(None, description="JWT token payload")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Context creation time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )

    def has_permission(self, permission: ModularPermission) -> bool:
        """Check if context has a specific permission."""
        # Admin has all permissions
        if ModularPermission.MODULAR_ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def has_module_access(self, module_type: ModuleType) -> bool:
        """Check if context has access to a specific module type."""
        required_perm = MODULE_TYPE_PERMISSIONS.get(module_type)
        if not required_perm:
            return False
        return self.has_permission(required_perm) or self.has_permission(
            ModularPermission.MODULE_INTERNAL
        )


# ============================================================================
# Modular Security Service
# ============================================================================


class ModularSecurityService:
    """
    Security service for modular agent core.

    Provides:
    - Authentication for modular execution requests
    - RBAC-based authorization for module access
    - Audit logging for all module interactions
    - Sensitive data redaction in logs
    """

    def __init__(self, config: AuditLogConfig | None = None) -> None:
        """Initialize the modular security service."""
        self._config = config or AuditLogConfig()
        self._audit_log: deque[AuditLogEntry] = deque(maxlen=self._config.max_entries)
        self._redact_patterns = [
            re.compile(p, re.IGNORECASE) for p in self._config.redact_patterns
        ]
        self._security_stats = {
            "authentications_success": 0,
            "authentications_failed": 0,
            "authorizations_success": 0,
            "authorizations_denied": 0,
            "audit_entries_logged": 0,
            "sensitive_data_redacted": 0,
        }
        self._lock = asyncio.Lock()
        logger.info(
            "ModularSecurityService initialized",
            audit_enabled=self._config.enabled,
            max_entries=self._config.max_entries,
        )

    # ========================================================================
    # Authentication
    # ========================================================================

    async def authenticate(
        self,
        token: str | None,
        trace_id: str,
        execution_id: str | None = None,
    ) -> ModularSecurityContext:
        """
        Authenticate a request and create security context.

        Args:
            token: JWT token (None for anonymous)
            trace_id: Distributed trace ID
            execution_id: Optional execution ID

        Returns:
            Security context with authentication status and permissions
        """
        context = ModularSecurityContext(
            trace_id=trace_id,
            execution_id=execution_id,
        )

        if not token:
            # Anonymous access - limited permissions
            logger.warning(
                "anonymous_access_attempt",
                trace_id=trace_id,
                execution_id=execution_id,
            )
            await self._log_audit(
                AuditLogEntry(
                    trace_id=trace_id,
                    execution_id=execution_id,
                    action=AuditAction.AUTH_FAILURE,
                    actor="anonymous",
                    success=False,
                    error="No authentication token provided",
                )
            )
            self._security_stats["authentications_failed"] += 1
            return context

        # Validate JWT token using existing security service
        payload = security_service.validate_token(token)

        if not payload:
            logger.warning(
                "authentication_failed",
                trace_id=trace_id,
                execution_id=execution_id,
                reason="invalid_token",
            )
            await self._log_audit(
                AuditLogEntry(
                    trace_id=trace_id,
                    execution_id=execution_id,
                    action=AuditAction.AUTH_FAILURE,
                    actor="unknown",
                    success=False,
                    error="Invalid or expired token",
                )
            )
            self._security_stats["authentications_failed"] += 1
            return context

        # Build authenticated context
        context.authenticated = True
        context.actor_id = payload.sub
        context.role = payload.role
        context.token_payload = payload

        # Get modular permissions based on role
        role_perms = MODULAR_ROLE_PERMISSIONS.get(payload.role, [])
        context.permissions = list(role_perms)

        logger.info(
            "authentication_success",
            trace_id=trace_id,
            execution_id=execution_id,
            actor_id=payload.sub,
            role=payload.role.value,
            permissions_count=len(context.permissions),
        )

        await self._log_audit(
            AuditLogEntry(
                trace_id=trace_id,
                execution_id=execution_id,
                action=AuditAction.AUTH_SUCCESS,
                actor=payload.sub,
                success=True,
                metadata={"role": payload.role.value},
            )
        )

        self._security_stats["authentications_success"] += 1
        return context

    # ========================================================================
    # Authorization (RBAC)
    # ========================================================================

    async def authorize_execution(
        self,
        context: ModularSecurityContext,
    ) -> bool:
        """
        Authorize modular execution request.

        Args:
            context: Security context

        Returns:
            True if authorized, False otherwise
        """
        if not context.authenticated:
            logger.warning(
                "authorization_denied_unauthenticated",
                trace_id=context.trace_id,
                execution_id=context.execution_id,
            )
            await self._log_audit(
                AuditLogEntry(
                    trace_id=context.trace_id,
                    execution_id=context.execution_id,
                    action=AuditAction.PERMISSION_DENIED,
                    actor=context.actor_id or "anonymous",
                    target="modular.solve",
                    success=False,
                    error="Not authenticated",
                )
            )
            self._security_stats["authorizations_denied"] += 1
            return False

        if not context.has_permission(ModularPermission.MODULAR_EXECUTE):
            logger.warning(
                "authorization_denied_no_permission",
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                actor_id=context.actor_id,
                required_permission=ModularPermission.MODULAR_EXECUTE.value,
            )
            await self._log_audit(
                AuditLogEntry(
                    trace_id=context.trace_id,
                    execution_id=context.execution_id,
                    action=AuditAction.PERMISSION_DENIED,
                    actor=context.actor_id or "unknown",
                    target="modular.solve",
                    success=False,
                    error=f"Missing permission: {ModularPermission.MODULAR_EXECUTE.value}",
                )
            )
            self._security_stats["authorizations_denied"] += 1
            return False

        logger.debug(
            "authorization_success",
            trace_id=context.trace_id,
            execution_id=context.execution_id,
            actor_id=context.actor_id,
        )
        self._security_stats["authorizations_success"] += 1
        return True

    async def authorize_module_access(
        self,
        context: ModularSecurityContext,
        module_type: ModuleType,
    ) -> bool:
        """
        Authorize access to a specific module.

        Args:
            context: Security context
            module_type: Target module type

        Returns:
            True if authorized, False otherwise
        """
        if not context.authenticated:
            logger.warning(
                "module_access_denied_unauthenticated",
                trace_id=context.trace_id,
                module_type=module_type.value,
            )
            await self._log_audit(
                AuditLogEntry(
                    trace_id=context.trace_id,
                    execution_id=context.execution_id,
                    action=AuditAction.PERMISSION_DENIED,
                    actor=context.actor_id or "anonymous",
                    target=module_type.value,
                    module_type=module_type,
                    success=False,
                    error="Not authenticated",
                )
            )
            self._security_stats["authorizations_denied"] += 1
            return False

        if not context.has_module_access(module_type):
            required_perm = MODULE_TYPE_PERMISSIONS.get(module_type)
            logger.warning(
                "module_access_denied_no_permission",
                trace_id=context.trace_id,
                module_type=module_type.value,
                actor_id=context.actor_id,
                required_permission=required_perm.value if required_perm else "unknown",
            )
            await self._log_audit(
                AuditLogEntry(
                    trace_id=context.trace_id,
                    execution_id=context.execution_id,
                    action=AuditAction.PERMISSION_DENIED,
                    actor=context.actor_id or "unknown",
                    target=module_type.value,
                    module_type=module_type,
                    success=False,
                    error=f"Missing module permission: {required_perm.value if required_perm else 'unknown'}",
                )
            )
            self._security_stats["authorizations_denied"] += 1
            return False

        logger.debug(
            "module_access_authorized",
            trace_id=context.trace_id,
            module_type=module_type.value,
            actor_id=context.actor_id,
        )
        self._security_stats["authorizations_success"] += 1
        return True

    # ========================================================================
    # Audit Logging
    # ========================================================================

    async def _log_audit(self, entry: AuditLogEntry) -> None:
        """
        Log an audit entry (internal method).

        Args:
            entry: Audit log entry
        """
        if not self._config.enabled:
            return

        # Redact sensitive data from metadata
        entry.metadata = self._redact_sensitive_data(entry.metadata)

        async with self._lock:
            self._audit_log.append(entry)
            self._security_stats["audit_entries_logged"] += 1

    async def log_execution_started(
        self,
        context: ModularSecurityContext,
        query_hash: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Log execution start.

        Args:
            context: Security context
            query_hash: Hash of query (not the query itself for privacy)
            config: Execution config (sanitized)
        """
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=AuditAction.EXECUTION_STARTED,
                actor=context.actor_id or "anonymous",
                success=True,
                metadata={
                    "query_hash": query_hash,
                    "config": self._redact_sensitive_data(config or {}),
                },
            )
        )

    async def log_execution_completed(
        self,
        context: ModularSecurityContext,
        duration_ms: int,
        iterations: int,
        success: bool,
        error: str | None = None,
    ) -> None:
        """
        Log execution completion.

        Args:
            context: Security context
            duration_ms: Execution duration
            iterations: Number of iterations
            success: Success status
            error: Error message if failed
        """
        action = (
            AuditAction.EXECUTION_COMPLETED if success else AuditAction.EXECUTION_FAILED
        )
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=action,
                actor=context.actor_id or "anonymous",
                success=success,
                duration_ms=duration_ms,
                error=error,
                metadata={
                    "iterations": iterations,
                },
            )
        )

    async def log_module_invoked(
        self,
        context: ModularSecurityContext,
        module_type: ModuleType,
        method: str,
        iteration: int,
    ) -> None:
        """
        Log module invocation.

        Args:
            context: Security context
            module_type: Module being invoked
            method: Method being called
            iteration: Current iteration
        """
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=AuditAction.MODULE_INVOKED,
                actor=context.actor_id or "system",
                target=method,
                module_type=module_type,
                success=True,
                metadata={
                    "iteration": iteration,
                },
            )
        )

    async def log_module_completed(
        self,
        context: ModularSecurityContext,
        module_type: ModuleType,
        duration_ms: int,
        success: bool,
        error: str | None = None,
    ) -> None:
        """
        Log module completion.

        Args:
            context: Security context
            module_type: Module that completed
            duration_ms: Module execution duration
            success: Success status
            error: Error message if failed
        """
        action = (
            AuditAction.MODULE_COMPLETED if success else AuditAction.MODULE_FAILED
        )
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=action,
                actor=context.actor_id or "system",
                module_type=module_type,
                success=success,
                duration_ms=duration_ms,
                error=error,
            )
        )

    async def log_verification_result(
        self,
        context: ModularSecurityContext,
        passed: bool,
        confidence: float,
        iteration: int,
    ) -> None:
        """
        Log verification result.

        Args:
            context: Security context
            passed: Verification passed
            confidence: Confidence score
            iteration: Current iteration
        """
        action = (
            AuditAction.VERIFICATION_PASSED if passed else AuditAction.VERIFICATION_FAILED
        )
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=action,
                actor=context.actor_id or "system",
                module_type=ModuleType.VERIFIER,
                success=passed,
                metadata={
                    "confidence": confidence,
                    "iteration": iteration,
                },
            )
        )

    async def log_plan_created(
        self,
        context: ModularSecurityContext,
        plan_id: str,
        step_count: int,
    ) -> None:
        """
        Log plan creation.

        Args:
            context: Security context
            plan_id: Plan identifier
            step_count: Number of steps in plan
        """
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=AuditAction.PLAN_CREATED,
                actor=context.actor_id or "system",
                module_type=ModuleType.PLANNER,
                success=True,
                metadata={
                    "plan_id": plan_id,
                    "step_count": step_count,
                },
            )
        )

    async def log_plan_refined(
        self,
        context: ModularSecurityContext,
        plan_id: str,
        iteration: int,
        step_count: int,
    ) -> None:
        """
        Log plan refinement.

        Args:
            context: Security context
            plan_id: Plan identifier
            iteration: Refinement iteration
            step_count: Number of steps in refined plan
        """
        await self._log_audit(
            AuditLogEntry(
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                action=AuditAction.PLAN_REFINED,
                actor=context.actor_id or "system",
                module_type=ModuleType.PLANNER,
                success=True,
                metadata={
                    "plan_id": plan_id,
                    "iteration": iteration,
                    "step_count": step_count,
                },
            )
        )

    # ========================================================================
    # Sensitive Data Handling (NFR-4.4)
    # ========================================================================

    def _redact_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Redact sensitive data from dictionary.

        Args:
            data: Dictionary to redact

        Returns:
            Dictionary with sensitive values redacted
        """
        if not data:
            return data

        redacted = {}
        for key, value in data.items():
            # Check if key matches sensitive pattern
            if any(pattern.search(key) for pattern in self._redact_patterns):
                redacted[key] = "[REDACTED]"
                self._security_stats["sensitive_data_redacted"] += 1
            elif isinstance(value, dict):
                # Recursively redact nested dicts
                redacted[key] = self._redact_sensitive_data(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings (potential data leakage)
                redacted[key] = value[:50] + "...[TRUNCATED]"
            else:
                redacted[key] = value

        return redacted

    @staticmethod
    def hash_query(query: str) -> str:
        """
        Create hash of query for audit logging (privacy-preserving).

        Args:
            query: Original query

        Returns:
            SHA-256 hash of query
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    # ========================================================================
    # Audit Log Queries
    # ========================================================================

    async def get_audit_logs(
        self,
        trace_id: str | None = None,
        execution_id: str | None = None,
        action: AuditAction | None = None,
        actor: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """
        Query audit logs with filters.

        Args:
            trace_id: Filter by trace ID
            execution_id: Filter by execution ID
            action: Filter by action type
            actor: Filter by actor
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum entries to return

        Returns:
            List of matching audit entries
        """
        async with self._lock:
            results = []
            for entry in reversed(self._audit_log):
                # Apply filters
                if trace_id and entry.trace_id != trace_id:
                    continue
                if execution_id and entry.execution_id != execution_id:
                    continue
                if action and entry.action != action:
                    continue
                if actor and entry.actor != actor:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue

                results.append(entry)
                if len(results) >= limit:
                    break

            return results

    async def get_audit_log_by_trace(self, trace_id: str) -> list[AuditLogEntry]:
        """
        Get all audit entries for a trace ID.

        Args:
            trace_id: Distributed trace ID

        Returns:
            List of audit entries for the trace
        """
        return await self.get_audit_logs(trace_id=trace_id, limit=1000)

    async def get_security_events(
        self,
        start_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """
        Get security-related events (auth failures, permission denied).

        Args:
            start_time: Filter by start time
            limit: Maximum entries to return

        Returns:
            List of security events
        """
        security_actions = {
            AuditAction.AUTH_FAILURE,
            AuditAction.PERMISSION_DENIED,
            AuditAction.RATE_LIMITED,
        }

        async with self._lock:
            results = []
            for entry in reversed(self._audit_log):
                if entry.action in security_actions:
                    if start_time and entry.timestamp < start_time:
                        continue
                    results.append(entry)
                    if len(results) >= limit:
                        break

            return results

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    def get_security_stats(self) -> dict[str, Any]:
        """Get security statistics."""
        return {
            **self._security_stats,
            "audit_log_size": len(self._audit_log),
            "audit_log_max_size": self._config.max_entries,
        }

    async def cleanup_expired_logs(self) -> int:
        """
        Remove audit logs older than retention period.

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now(UTC) - timedelta(hours=self._config.retention_hours)
        removed = 0

        async with self._lock:
            while self._audit_log and self._audit_log[0].timestamp < cutoff:
                self._audit_log.popleft()
                removed += 1

        if removed > 0:
            logger.info(
                "audit_logs_cleaned_up",
                removed_count=removed,
                retention_hours=self._config.retention_hours,
            )

        return removed


# Global modular security service instance
modular_security_service = ModularSecurityService()
