"""
Tests for Modular Agent Core Security Module

Tests authentication, authorization (RBAC), and audit logging following NFR-4:
- NFR-4.1: All inter-module communication uses authenticated channels
- NFR-4.2: Module access controlled through RBAC policies
- NFR-4.3: All module interactions auditable with trace IDs
- NFR-4.4: Sensitive data not logged in module communication
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.security import (
    Permission,
    Role,
    TokenPayload,
    TokenType,
)
from agentcore.modular.models import ModuleType
from agentcore.modular.security import (
    AuditAction,
    AuditLogConfig,
    AuditLogEntry,
    ModularPermission,
    ModularSecurityContext,
    ModularSecurityService,
    MODULAR_ROLE_PERMISSIONS,
    MODULE_TYPE_PERMISSIONS,
    modular_security_service,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def security_service() -> ModularSecurityService:
    """Create a fresh security service for testing."""
    return ModularSecurityService(
        config=AuditLogConfig(
            enabled=True,
            max_entries=100,
            retention_hours=24,
        )
    )


@pytest.fixture
def valid_token_payload() -> TokenPayload:
    """Create a valid token payload for testing."""
    return TokenPayload.create(
        subject="test-agent-001",
        role=Role.SERVICE,
        token_type=TokenType.ACCESS,
        expiration_hours=24,
        agent_id="test-agent-001",
    )


@pytest.fixture
def agent_token_payload() -> TokenPayload:
    """Create an agent role token payload."""
    return TokenPayload.create(
        subject="test-agent-002",
        role=Role.AGENT,
        token_type=TokenType.ACCESS,
        expiration_hours=24,
        agent_id="test-agent-002",
    )


@pytest.fixture
def admin_token_payload() -> TokenPayload:
    """Create an admin role token payload."""
    return TokenPayload.create(
        subject="admin-001",
        role=Role.ADMIN,
        token_type=TokenType.ACCESS,
        expiration_hours=24,
        agent_id="admin-001",
    )


@pytest.fixture
def security_context() -> ModularSecurityContext:
    """Create an authenticated security context."""
    return ModularSecurityContext(
        trace_id=str(uuid4()),
        execution_id=str(uuid4()),
        authenticated=True,
        actor_id="test-agent-001",
        role=Role.SERVICE,
        permissions=[
            ModularPermission.MODULAR_EXECUTE,
            ModularPermission.MODULAR_READ,
            ModularPermission.MODULE_PLANNER,
            ModularPermission.MODULE_EXECUTOR,
            ModularPermission.MODULE_VERIFIER,
            ModularPermission.MODULE_GENERATOR,
            ModularPermission.MODULE_INTERNAL,
        ],
    )


@pytest.fixture
def unauthenticated_context() -> ModularSecurityContext:
    """Create an unauthenticated security context."""
    return ModularSecurityContext(
        trace_id=str(uuid4()),
        execution_id=str(uuid4()),
        authenticated=False,
    )


# ============================================================================
# Test Modular Permissions
# ============================================================================


class TestModularPermissions:
    """Tests for modular-specific permissions."""

    def test_modular_permission_enum_values(self) -> None:
        """Test that all modular permissions have expected values."""
        assert ModularPermission.MODULAR_EXECUTE == "modular:execute"
        assert ModularPermission.MODULAR_READ == "modular:read"
        assert ModularPermission.MODULAR_CANCEL == "modular:cancel"
        assert ModularPermission.MODULE_PLANNER == "module:planner"
        assert ModularPermission.MODULE_EXECUTOR == "module:executor"
        assert ModularPermission.MODULE_VERIFIER == "module:verifier"
        assert ModularPermission.MODULE_GENERATOR == "module:generator"
        assert ModularPermission.MODULE_INTERNAL == "module:internal"
        assert ModularPermission.MODULAR_ADMIN == "modular:admin"

    def test_role_permissions_mapping(self) -> None:
        """Test role to permissions mapping."""
        # Agent role should have limited permissions
        agent_perms = MODULAR_ROLE_PERMISSIONS[Role.AGENT]
        assert ModularPermission.MODULAR_EXECUTE in agent_perms
        assert ModularPermission.MODULAR_READ in agent_perms
        assert ModularPermission.MODULE_INTERNAL not in agent_perms

        # Service role should have module access
        service_perms = MODULAR_ROLE_PERMISSIONS[Role.SERVICE]
        assert ModularPermission.MODULAR_EXECUTE in service_perms
        assert ModularPermission.MODULE_PLANNER in service_perms
        assert ModularPermission.MODULE_EXECUTOR in service_perms
        assert ModularPermission.MODULE_VERIFIER in service_perms
        assert ModularPermission.MODULE_GENERATOR in service_perms
        assert ModularPermission.MODULE_INTERNAL in service_perms

        # Admin role should have admin permission
        admin_perms = MODULAR_ROLE_PERMISSIONS[Role.ADMIN]
        assert ModularPermission.MODULAR_ADMIN in admin_perms

    def test_module_type_permissions_mapping(self) -> None:
        """Test module type to permission mapping."""
        assert MODULE_TYPE_PERMISSIONS[ModuleType.PLANNER] == ModularPermission.MODULE_PLANNER
        assert MODULE_TYPE_PERMISSIONS[ModuleType.EXECUTOR] == ModularPermission.MODULE_EXECUTOR
        assert MODULE_TYPE_PERMISSIONS[ModuleType.VERIFIER] == ModularPermission.MODULE_VERIFIER
        assert MODULE_TYPE_PERMISSIONS[ModuleType.GENERATOR] == ModularPermission.MODULE_GENERATOR


# ============================================================================
# Test Security Context
# ============================================================================


class TestModularSecurityContext:
    """Tests for ModularSecurityContext."""

    def test_context_creation(self) -> None:
        """Test security context creation."""
        context = ModularSecurityContext(
            trace_id="trace-123",
            execution_id="exec-456",
        )
        assert context.trace_id == "trace-123"
        assert context.execution_id == "exec-456"
        assert context.authenticated is False
        assert context.actor_id is None
        assert context.permissions == []

    def test_has_permission_with_permission(
        self, security_context: ModularSecurityContext
    ) -> None:
        """Test has_permission returns True when permission exists."""
        assert security_context.has_permission(ModularPermission.MODULAR_EXECUTE) is True
        assert security_context.has_permission(ModularPermission.MODULE_PLANNER) is True

    def test_has_permission_without_permission(
        self, security_context: ModularSecurityContext
    ) -> None:
        """Test has_permission returns False when permission missing."""
        # Remove all permissions except one
        context = ModularSecurityContext(
            trace_id=str(uuid4()),
            authenticated=True,
            permissions=[ModularPermission.MODULAR_READ],
        )
        assert context.has_permission(ModularPermission.MODULAR_EXECUTE) is False
        assert context.has_permission(ModularPermission.MODULE_PLANNER) is False

    def test_has_permission_admin_has_all(self) -> None:
        """Test admin permission grants access to everything."""
        context = ModularSecurityContext(
            trace_id=str(uuid4()),
            authenticated=True,
            permissions=[ModularPermission.MODULAR_ADMIN],
        )
        assert context.has_permission(ModularPermission.MODULAR_EXECUTE) is True
        assert context.has_permission(ModularPermission.MODULE_PLANNER) is True
        assert context.has_permission(ModularPermission.MODULE_INTERNAL) is True

    def test_has_module_access_with_permission(
        self, security_context: ModularSecurityContext
    ) -> None:
        """Test has_module_access returns True with correct permission."""
        assert security_context.has_module_access(ModuleType.PLANNER) is True
        assert security_context.has_module_access(ModuleType.EXECUTOR) is True
        assert security_context.has_module_access(ModuleType.VERIFIER) is True
        assert security_context.has_module_access(ModuleType.GENERATOR) is True

    def test_has_module_access_without_permission(self) -> None:
        """Test has_module_access returns False without permission."""
        context = ModularSecurityContext(
            trace_id=str(uuid4()),
            authenticated=True,
            permissions=[ModularPermission.MODULAR_EXECUTE],
        )
        assert context.has_module_access(ModuleType.PLANNER) is False

    def test_has_module_access_with_internal_permission(self) -> None:
        """Test MODULE_INTERNAL grants access to all modules."""
        context = ModularSecurityContext(
            trace_id=str(uuid4()),
            authenticated=True,
            permissions=[ModularPermission.MODULE_INTERNAL],
        )
        assert context.has_module_access(ModuleType.PLANNER) is True
        assert context.has_module_access(ModuleType.EXECUTOR) is True
        assert context.has_module_access(ModuleType.VERIFIER) is True
        assert context.has_module_access(ModuleType.GENERATOR) is True


# ============================================================================
# Test Authentication
# ============================================================================


class TestAuthentication:
    """Tests for authentication functionality."""

    @pytest.mark.asyncio
    async def test_authenticate_without_token(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test authentication fails without token."""
        context = await security_service.authenticate(
            token=None,
            trace_id="trace-123",
        )
        assert context.authenticated is False
        assert context.actor_id is None
        assert context.permissions == []

    @pytest.mark.asyncio
    async def test_authenticate_with_invalid_token(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test authentication fails with invalid token."""
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=None,
        ):
            context = await security_service.authenticate(
                token="invalid-token",
                trace_id="trace-123",
            )
            assert context.authenticated is False

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_token(
        self,
        security_service: ModularSecurityService,
        valid_token_payload: TokenPayload,
    ) -> None:
        """Test authentication succeeds with valid token."""
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=valid_token_payload,
        ):
            context = await security_service.authenticate(
                token="valid-token",
                trace_id="trace-123",
                execution_id="exec-456",
            )
            assert context.authenticated is True
            assert context.actor_id == valid_token_payload.sub
            assert context.role == valid_token_payload.role
            assert context.trace_id == "trace-123"
            assert context.execution_id == "exec-456"
            assert len(context.permissions) > 0

    @pytest.mark.asyncio
    async def test_authenticate_agent_role_permissions(
        self,
        security_service: ModularSecurityService,
        agent_token_payload: TokenPayload,
    ) -> None:
        """Test agent role gets correct permissions."""
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=agent_token_payload,
        ):
            context = await security_service.authenticate(
                token="agent-token",
                trace_id="trace-123",
            )
            assert ModularPermission.MODULAR_EXECUTE in context.permissions
            assert ModularPermission.MODULAR_READ in context.permissions
            assert ModularPermission.MODULE_INTERNAL not in context.permissions

    @pytest.mark.asyncio
    async def test_authenticate_admin_role_permissions(
        self,
        security_service: ModularSecurityService,
        admin_token_payload: TokenPayload,
    ) -> None:
        """Test admin role gets admin permission."""
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=admin_token_payload,
        ):
            context = await security_service.authenticate(
                token="admin-token",
                trace_id="trace-123",
            )
            assert ModularPermission.MODULAR_ADMIN in context.permissions


# ============================================================================
# Test Authorization (RBAC)
# ============================================================================


class TestAuthorization:
    """Tests for authorization functionality."""

    @pytest.mark.asyncio
    async def test_authorize_execution_authenticated(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test execution authorization for authenticated user."""
        authorized = await security_service.authorize_execution(security_context)
        assert authorized is True

    @pytest.mark.asyncio
    async def test_authorize_execution_unauthenticated(
        self,
        security_service: ModularSecurityService,
        unauthenticated_context: ModularSecurityContext,
    ) -> None:
        """Test execution authorization denied for unauthenticated user."""
        authorized = await security_service.authorize_execution(unauthenticated_context)
        assert authorized is False

    @pytest.mark.asyncio
    async def test_authorize_execution_without_permission(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test execution authorization denied without MODULAR_EXECUTE permission."""
        context = ModularSecurityContext(
            trace_id=str(uuid4()),
            authenticated=True,
            actor_id="test-user",
            permissions=[ModularPermission.MODULAR_READ],  # No execute permission
        )
        authorized = await security_service.authorize_execution(context)
        assert authorized is False

    @pytest.mark.asyncio
    async def test_authorize_module_access_planner(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test module access authorization for planner."""
        authorized = await security_service.authorize_module_access(
            security_context, ModuleType.PLANNER
        )
        assert authorized is True

    @pytest.mark.asyncio
    async def test_authorize_module_access_executor(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test module access authorization for executor."""
        authorized = await security_service.authorize_module_access(
            security_context, ModuleType.EXECUTOR
        )
        assert authorized is True

    @pytest.mark.asyncio
    async def test_authorize_module_access_unauthenticated(
        self,
        security_service: ModularSecurityService,
        unauthenticated_context: ModularSecurityContext,
    ) -> None:
        """Test module access denied for unauthenticated user."""
        authorized = await security_service.authorize_module_access(
            unauthenticated_context, ModuleType.PLANNER
        )
        assert authorized is False

    @pytest.mark.asyncio
    async def test_authorize_module_access_without_permission(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test module access denied without specific module permission."""
        context = ModularSecurityContext(
            trace_id=str(uuid4()),
            authenticated=True,
            actor_id="test-user",
            permissions=[ModularPermission.MODULAR_EXECUTE],  # No module permission
        )
        authorized = await security_service.authorize_module_access(
            context, ModuleType.PLANNER
        )
        assert authorized is False


# ============================================================================
# Test Audit Logging
# ============================================================================


class TestAuditLogging:
    """Tests for audit logging functionality."""

    @pytest.mark.asyncio
    async def test_log_execution_started(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging execution start."""
        await security_service.log_execution_started(
            context=security_context,
            query_hash="abc123",
            config={"max_iterations": 5},
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id, limit=10
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.EXECUTION_STARTED
        assert logs[0].trace_id == security_context.trace_id

    @pytest.mark.asyncio
    async def test_log_execution_completed(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging execution completion."""
        await security_service.log_execution_completed(
            context=security_context,
            duration_ms=5000,
            iterations=3,
            success=True,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.EXECUTION_COMPLETED
        assert logs[0].duration_ms == 5000
        assert logs[0].success is True

    @pytest.mark.asyncio
    async def test_log_execution_failed(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging execution failure."""
        await security_service.log_execution_completed(
            context=security_context,
            duration_ms=1000,
            iterations=1,
            success=False,
            error="Timeout error",
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.EXECUTION_FAILED
        assert logs[0].success is False
        assert logs[0].error == "Timeout error"

    @pytest.mark.asyncio
    async def test_log_module_invoked(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging module invocation."""
        await security_service.log_module_invoked(
            context=security_context,
            module_type=ModuleType.PLANNER,
            method="create_plan",
            iteration=1,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.MODULE_INVOKED
        assert logs[0].module_type == ModuleType.PLANNER
        assert logs[0].target == "create_plan"

    @pytest.mark.asyncio
    async def test_log_module_completed(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging module completion."""
        await security_service.log_module_completed(
            context=security_context,
            module_type=ModuleType.EXECUTOR,
            duration_ms=2000,
            success=True,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.MODULE_COMPLETED
        assert logs[0].module_type == ModuleType.EXECUTOR
        assert logs[0].duration_ms == 2000

    @pytest.mark.asyncio
    async def test_log_verification_passed(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging verification passed."""
        await security_service.log_verification_result(
            context=security_context,
            passed=True,
            confidence=0.95,
            iteration=2,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.VERIFICATION_PASSED
        assert logs[0].metadata["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_log_verification_failed(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging verification failed."""
        await security_service.log_verification_result(
            context=security_context,
            passed=False,
            confidence=0.45,
            iteration=1,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.VERIFICATION_FAILED

    @pytest.mark.asyncio
    async def test_log_plan_created(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging plan creation."""
        await security_service.log_plan_created(
            context=security_context,
            plan_id="plan-123",
            step_count=5,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.PLAN_CREATED
        assert logs[0].metadata["plan_id"] == "plan-123"
        assert logs[0].metadata["step_count"] == 5

    @pytest.mark.asyncio
    async def test_log_plan_refined(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test logging plan refinement."""
        await security_service.log_plan_refined(
            context=security_context,
            plan_id="plan-456",
            iteration=3,
            step_count=7,
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].action == AuditAction.PLAN_REFINED
        assert logs[0].metadata["iteration"] == 3

    @pytest.mark.asyncio
    async def test_audit_log_by_trace(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test getting all audit logs for a trace."""
        # Log multiple events
        await security_service.log_execution_started(
            context=security_context, query_hash="hash1"
        )
        await security_service.log_module_invoked(
            context=security_context,
            module_type=ModuleType.PLANNER,
            method="create_plan",
            iteration=1,
        )
        await security_service.log_module_completed(
            context=security_context,
            module_type=ModuleType.PLANNER,
            duration_ms=1000,
            success=True,
        )

        logs = await security_service.get_audit_log_by_trace(security_context.trace_id)
        assert len(logs) == 3

    @pytest.mark.asyncio
    async def test_security_events_filtering(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test filtering security events."""
        # Create some security events via authentication
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=None,
        ):
            await security_service.authenticate(
                token="invalid", trace_id="trace-1"
            )
            await security_service.authenticate(
                token="invalid", trace_id="trace-2"
            )

        events = await security_service.get_security_events(limit=10)
        assert len(events) >= 2
        for event in events:
            assert event.action in {
                AuditAction.AUTH_FAILURE,
                AuditAction.PERMISSION_DENIED,
                AuditAction.RATE_LIMITED,
            }


# ============================================================================
# Test Sensitive Data Handling (NFR-4.4)
# ============================================================================


class TestSensitiveDataHandling:
    """Tests for sensitive data redaction."""

    def test_redact_api_key(self, security_service: ModularSecurityService) -> None:
        """Test API key redaction."""
        data = {"api_key": "sk-secret-123", "name": "test"}
        redacted = security_service._redact_sensitive_data(data)
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["name"] == "test"

    def test_redact_password(self, security_service: ModularSecurityService) -> None:
        """Test password redaction."""
        data = {"password": "supersecret", "username": "admin"}
        redacted = security_service._redact_sensitive_data(data)
        assert redacted["password"] == "[REDACTED]"
        assert redacted["username"] == "admin"

    def test_redact_token(self, security_service: ModularSecurityService) -> None:
        """Test token redaction."""
        data = {"auth_token": "eyJhbGciOiJIUzI1NiIs...", "status": "active"}
        redacted = security_service._redact_sensitive_data(data)
        assert redacted["auth_token"] == "[REDACTED]"
        assert redacted["status"] == "active"

    def test_redact_secret(self, security_service: ModularSecurityService) -> None:
        """Test secret redaction."""
        data = {"client_secret": "abc123", "client_id": "app-001"}
        redacted = security_service._redact_sensitive_data(data)
        assert redacted["client_secret"] == "[REDACTED]"
        assert redacted["client_id"] == "app-001"

    def test_redact_credential(self, security_service: ModularSecurityService) -> None:
        """Test credential redaction."""
        data = {"credentials": {"key": "value"}, "type": "oauth"}
        redacted = security_service._redact_sensitive_data(data)
        assert redacted["credentials"] == "[REDACTED]"

    def test_redact_nested_data(self, security_service: ModularSecurityService) -> None:
        """Test nested data redaction."""
        data = {
            "config": {
                "api_key": "secret",
                "timeout": 30,
            },
            "name": "test",
        }
        redacted = security_service._redact_sensitive_data(data)
        assert redacted["config"]["api_key"] == "[REDACTED]"
        assert redacted["config"]["timeout"] == 30
        assert redacted["name"] == "test"

    def test_truncate_long_strings(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test long string truncation."""
        long_string = "a" * 200
        data = {"content": long_string, "short": "ok"}
        redacted = security_service._redact_sensitive_data(data)
        assert "[TRUNCATED]" in redacted["content"]
        assert len(redacted["content"]) < 200
        assert redacted["short"] == "ok"

    def test_hash_query(self) -> None:
        """Test query hashing for privacy."""
        query1 = "What is the capital of France?"
        query2 = "What is the capital of France?"
        query3 = "Different query"

        hash1 = ModularSecurityService.hash_query(query1)
        hash2 = ModularSecurityService.hash_query(query2)
        hash3 = ModularSecurityService.hash_query(query3)

        # Same query should produce same hash
        assert hash1 == hash2
        # Different query should produce different hash
        assert hash1 != hash3
        # Hash should be fixed length (16 chars)
        assert len(hash1) == 16

    @pytest.mark.asyncio
    async def test_sensitive_data_not_in_logs(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test sensitive data is not in audit logs."""
        await security_service.log_execution_started(
            context=security_context,
            query_hash="hash123",
            config={
                "api_key": "sk-secret-key",
                "max_iterations": 5,
            },
        )

        logs = await security_service.get_audit_logs(
            trace_id=security_context.trace_id
        )
        assert len(logs) == 1
        assert logs[0].metadata["config"]["api_key"] == "[REDACTED]"
        assert logs[0].metadata["config"]["max_iterations"] == 5


# ============================================================================
# Test Statistics & Cleanup
# ============================================================================


class TestStatisticsAndCleanup:
    """Tests for statistics and cleanup functionality."""

    @pytest.mark.asyncio
    async def test_security_stats(
        self, security_service: ModularSecurityService
    ) -> None:
        """Test security statistics tracking."""
        # Perform some operations
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=None,
        ):
            await security_service.authenticate(token="invalid", trace_id="t1")

        stats = security_service.get_security_stats()
        assert "authentications_failed" in stats
        assert "audit_log_size" in stats
        assert "audit_entries_logged" in stats

    @pytest.mark.asyncio
    async def test_cleanup_expired_logs(self) -> None:
        """Test cleanup of expired audit logs."""
        # Create service with short retention
        config = AuditLogConfig(
            enabled=True,
            max_entries=100,
            retention_hours=0,  # Expire immediately
        )
        service = ModularSecurityService(config=config)

        # Add some logs
        context = ModularSecurityContext(
            trace_id="test", authenticated=True, actor_id="user"
        )
        await service.log_execution_started(context=context, query_hash="hash")

        # Force timestamp to be old
        if service._audit_log:
            old_entry = service._audit_log[0]
            old_entry.timestamp = datetime.now(UTC) - timedelta(hours=1)

        # Cleanup
        removed = await service.cleanup_expired_logs()
        assert removed >= 1

    def test_audit_log_max_entries(self) -> None:
        """Test audit log respects max entries limit."""
        config = AuditLogConfig(
            enabled=True,
            max_entries=5,
        )
        service = ModularSecurityService(config=config)

        # Log should be capped at max_entries
        assert service._audit_log.maxlen == 5


# ============================================================================
# Test Global Instance
# ============================================================================


class TestGlobalInstance:
    """Tests for global security service instance."""

    def test_global_instance_exists(self) -> None:
        """Test global modular_security_service instance exists."""
        assert modular_security_service is not None
        assert isinstance(modular_security_service, ModularSecurityService)

    def test_global_instance_initialized(self) -> None:
        """Test global instance is properly initialized."""
        stats = modular_security_service.get_security_stats()
        assert "authentications_success" in stats
        assert "audit_log_size" in stats


# ============================================================================
# Integration Tests
# ============================================================================


class TestSecurityIntegration:
    """Integration tests for security workflow."""

    @pytest.mark.asyncio
    async def test_full_authentication_authorization_flow(
        self,
        security_service: ModularSecurityService,
        valid_token_payload: TokenPayload,
    ) -> None:
        """Test complete auth flow from token to module access."""
        trace_id = str(uuid4())

        # Step 1: Authenticate
        with patch(
            "agentcore.modular.security.security_service.validate_token",
            return_value=valid_token_payload,
        ):
            context = await security_service.authenticate(
                token="valid-token",
                trace_id=trace_id,
                execution_id="exec-001",
            )

        assert context.authenticated is True

        # Step 2: Authorize execution
        authorized = await security_service.authorize_execution(context)
        assert authorized is True

        # Step 3: Authorize module access
        for module_type in ModuleType:
            authorized = await security_service.authorize_module_access(
                context, module_type
            )
            assert authorized is True

        # Step 4: Check audit trail
        logs = await security_service.get_audit_log_by_trace(trace_id)
        assert len(logs) >= 1  # At least auth success logged

    @pytest.mark.asyncio
    async def test_execution_lifecycle_audit_trail(
        self,
        security_service: ModularSecurityService,
        security_context: ModularSecurityContext,
    ) -> None:
        """Test complete execution audit trail."""
        # Simulate execution lifecycle
        await security_service.log_execution_started(
            context=security_context,
            query_hash="hash123",
        )
        await security_service.log_plan_created(
            context=security_context,
            plan_id="plan-001",
            step_count=3,
        )
        await security_service.log_module_invoked(
            context=security_context,
            module_type=ModuleType.PLANNER,
            method="create_plan",
            iteration=1,
        )
        await security_service.log_module_completed(
            context=security_context,
            module_type=ModuleType.PLANNER,
            duration_ms=500,
            success=True,
        )
        await security_service.log_module_invoked(
            context=security_context,
            module_type=ModuleType.EXECUTOR,
            method="execute_plan",
            iteration=1,
        )
        await security_service.log_module_completed(
            context=security_context,
            module_type=ModuleType.EXECUTOR,
            duration_ms=2000,
            success=True,
        )
        await security_service.log_verification_result(
            context=security_context,
            passed=True,
            confidence=0.92,
            iteration=1,
        )
        await security_service.log_execution_completed(
            context=security_context,
            duration_ms=5000,
            iterations=1,
            success=True,
        )

        # Verify complete audit trail
        logs = await security_service.get_audit_log_by_trace(
            security_context.trace_id
        )
        assert len(logs) == 8

        # Verify action sequence
        actions = [log.action for log in reversed(logs)]  # Oldest first
        assert AuditAction.EXECUTION_STARTED in actions
        assert AuditAction.PLAN_CREATED in actions
        assert AuditAction.MODULE_INVOKED in actions
        assert AuditAction.MODULE_COMPLETED in actions
        assert AuditAction.VERIFICATION_PASSED in actions
        assert AuditAction.EXECUTION_COMPLETED in actions
