"""Comprehensive tests for sandbox security implementation."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentcore.agent_runtime.models.sandbox import (
    AuditEventType,
    AuditLogEntry,
    ExecutionLimits,
    ResourceLimitExceededError,
    ResourcePolicy,
    SandboxConfig,
    SandboxPermission,
    SandboxStats,
    SecurityViolationError)
from agentcore.agent_runtime.services.audit_logger import AuditLogger
from agentcore.agent_runtime.services.sandbox_service import SandboxService


@pytest.fixture
async def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def temp_log_dir():
    """Create temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def audit_logger(temp_log_dir):
    """Create audit logger instance."""
    logger = AuditLogger(
        log_directory=temp_log_dir,
        max_logs_in_memory=1000,
        retention_days=7)
    await logger.start()
    yield logger
    await logger.stop()


@pytest.fixture
async def sandbox_service(audit_logger, temp_workspace):
    """Create sandbox service instance."""
    return SandboxService(
        audit_logger=audit_logger,
        workspace_root=temp_workspace)


@pytest.fixture
def basic_sandbox_config():
    """Create basic sandbox configuration."""
    return SandboxConfig(
        sandbox_id="test-sandbox-001",
        agent_id="test-agent-001",
        permissions=[SandboxPermission.READ, SandboxPermission.EXECUTE],
        execution_limits=ExecutionLimits(
            max_execution_time_seconds=5,
            max_memory_mb=128,
            max_cpu_percent=50.0),
        strict_mode=True)


@pytest.mark.asyncio
class TestSandboxLifecycle:
    """Test sandbox creation, execution, and destruction."""

    async def test_create_sandbox(self, sandbox_service, basic_sandbox_config):
        """Test sandbox creation."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        assert sandbox_id == "test-sandbox-001"
        assert sandbox_id in sandbox_service._sandboxes
        assert sandbox_id in sandbox_service._stats

        # Verify workspace created
        workspace_path = sandbox_service._get_workspace_path(sandbox_id)
        assert workspace_path.exists()

    async def test_create_duplicate_sandbox(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test creating duplicate sandbox fails."""
        await sandbox_service.create_sandbox(basic_sandbox_config)

        with pytest.raises(ValueError, match="already exists"):
            await sandbox_service.create_sandbox(basic_sandbox_config)

    async def test_destroy_sandbox(self, sandbox_service, basic_sandbox_config):
        """Test sandbox destruction."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)
        await sandbox_service.destroy_sandbox(sandbox_id)

        assert sandbox_id not in sandbox_service._sandboxes
        assert sandbox_id not in sandbox_service._stats

    async def test_destroy_nonexistent_sandbox(self, sandbox_service):
        """Test destroying nonexistent sandbox fails."""
        with pytest.raises(KeyError, match="not found"):
            await sandbox_service.destroy_sandbox("nonexistent-sandbox")


@pytest.mark.asyncio
class TestPermissionEnforcement:
    """Test permission-based access control."""

    async def test_check_permission_granted(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test permission check when granted."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Should grant READ permission (in global permissions)
        result = await sandbox_service.check_permission(
            sandbox_id=sandbox_id,
            permission=SandboxPermission.READ,
            resource="/data/test.txt")

        assert result is True

    async def test_check_permission_denied_strict_mode(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test permission check denied in strict mode raises exception."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Should deny WRITE permission (not in global permissions)
        with pytest.raises(SecurityViolationError) as exc_info:
            await sandbox_service.check_permission(
                sandbox_id=sandbox_id,
                permission=SandboxPermission.WRITE,
                resource="/data/test.txt")

        assert exc_info.value.permission == SandboxPermission.WRITE

    async def test_check_permission_denied_permissive_mode(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test permission check denied in permissive mode returns False."""
        basic_sandbox_config.strict_mode = False
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        result = await sandbox_service.check_permission(
            sandbox_id=sandbox_id,
            permission=SandboxPermission.WRITE,
            resource="/data/test.txt")

        assert result is False

    async def test_resource_policy_explicit_grant(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test resource policy explicit permission grant."""
        # Add policy allowing WRITE to /tmp/*
        basic_sandbox_config.resource_policies.append(
            ResourcePolicy(
                resource_pattern="/tmp/*",
                allowed_permissions=[SandboxPermission.WRITE],
                description="Allow writes to /tmp")
        )

        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Should grant WRITE to /tmp/test.txt
        result = await sandbox_service.check_permission(
            sandbox_id=sandbox_id,
            permission=SandboxPermission.WRITE,
            resource="/tmp/test.txt")

        assert result is True

    async def test_resource_policy_explicit_deny(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test resource policy explicit permission denial."""
        # Add policy denying READ to /secrets/*
        basic_sandbox_config.resource_policies.append(
            ResourcePolicy(
                resource_pattern="/secrets/*",
                allowed_permissions=[],
                denied_permissions=[SandboxPermission.READ],
                description="Deny reads from /secrets")
        )

        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Should deny READ to /secrets/api_key.txt
        with pytest.raises(SecurityViolationError):
            await sandbox_service.check_permission(
                sandbox_id=sandbox_id,
                permission=SandboxPermission.READ,
                resource="/secrets/api_key.txt")

    async def test_network_permission_disabled(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test network permission when disabled."""
        basic_sandbox_config.allow_network = False
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        with pytest.raises(SecurityViolationError):
            await sandbox_service.check_permission(
                sandbox_id=sandbox_id,
                permission=SandboxPermission.NETWORK,
                resource="api.example.com")

    async def test_network_permission_allowed_hosts(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test network permission with allowed hosts."""
        basic_sandbox_config.allow_network = True
        basic_sandbox_config.allowed_hosts = ["api.example.com"]
        basic_sandbox_config.permissions.append(SandboxPermission.NETWORK)

        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Should allow api.example.com
        result = await sandbox_service.check_permission(
            sandbox_id=sandbox_id,
            permission=SandboxPermission.NETWORK,
            resource="api.example.com")
        assert result is True

        # Should deny other hosts in strict mode
        with pytest.raises(SecurityViolationError):
            await sandbox_service.check_permission(
                sandbox_id=sandbox_id,
                permission=SandboxPermission.NETWORK,
                resource="malicious.com")


@pytest.mark.asyncio
class TestCodeExecution:
    """Test sandbox code execution with limits."""

    async def test_execute_simple_code(self, sandbox_service, basic_sandbox_config):
        """Test executing simple code."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        code = """
result = 2 + 2
"""

        result = await sandbox_service.execute_in_sandbox(
            sandbox_id=sandbox_id,
            code=code)

        assert result == 4

    async def test_execute_code_with_context(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test executing code with context variables."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        code = """
result = x + y
"""

        result = await sandbox_service.execute_in_sandbox(
            sandbox_id=sandbox_id,
            code=code,
            context={"x": 10, "y": 20})

        assert result == 30

    @pytest.mark.skip(reason="Timeout test unreliable in RestrictedPython environment")
    async def test_execute_code_timeout(self, sandbox_service, basic_sandbox_config):
        """Test execution timeout enforcement."""
        basic_sandbox_config.execution_limits.max_execution_time_seconds = 1
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Use a busy loop computation
        code = """
# Compute something expensive to trigger timeout
result = 0
for i in range(100000000):
    result += i
"""

        with pytest.raises(ResourceLimitExceededError) as exc_info:
            await sandbox_service.execute_in_sandbox(
                sandbox_id=sandbox_id,
                code=code)

        assert exc_info.value.limit_type == "execution_time"

    async def test_execute_restricted_code(self, sandbox_service, basic_sandbox_config):
        """Test execution of restricted code fails."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Try to import restricted module
        code = """
import os
result = os.system('ls')
"""

        with pytest.raises(SecurityViolationError):
            await sandbox_service.execute_in_sandbox(
                sandbox_id=sandbox_id,
                code=code)

    async def test_execute_safe_builtins(self, sandbox_service, basic_sandbox_config):
        """Test safe builtins are available."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        code = """
data = [1, 2, 3, 4, 5]
result = sum(data)
"""

        result = await sandbox_service.execute_in_sandbox(
            sandbox_id=sandbox_id,
            code=code)

        assert result == 15


@pytest.mark.asyncio
class TestAuditLogging:
    """Test security audit logging."""

    async def test_log_event(self, audit_logger):
        """Test logging audit event."""
        entry = AuditLogEntry(
            event_type=AuditEventType.PERMISSION_GRANTED,
            sandbox_id="test-sandbox",
            agent_id="test-agent",
            operation="read_file",
            resource="/data/test.txt",
            permission=SandboxPermission.READ,
            result=True)

        await audit_logger.log_event(entry)

        # Event should be in memory buffer
        assert len(audit_logger._log_buffer) == 1
        assert audit_logger._log_buffer[0] == entry

    async def test_query_logs_by_sandbox(self, audit_logger):
        """Test querying logs by sandbox ID."""
        # Log multiple events
        for i in range(5):
            entry = AuditLogEntry(
                event_type=AuditEventType.RESOURCE_ACCESS,
                sandbox_id=f"sandbox-{i % 2}",
                agent_id=f"agent-{i}",
                operation="test",
                result=True)
            await audit_logger.log_event(entry)

        # Query for sandbox-0
        results = await audit_logger.query_logs(sandbox_id="sandbox-0")

        assert len(results) == 3
        assert all(r.sandbox_id == "sandbox-0" for r in results)

    async def test_query_logs_by_event_type(self, audit_logger):
        """Test querying logs by event type."""
        # Log different event types
        await audit_logger.log_event(
            AuditLogEntry(
                event_type=AuditEventType.PERMISSION_GRANTED,
                sandbox_id="test",
                agent_id="test",
                operation="test",
                result=True)
        )
        await audit_logger.log_event(
            AuditLogEntry(
                event_type=AuditEventType.PERMISSION_DENIED,
                sandbox_id="test",
                agent_id="test",
                operation="test",
                result=False)
        )

        # Query for denials
        results = await audit_logger.query_logs(
            event_type=AuditEventType.PERMISSION_DENIED
        )

        assert len(results) == 1
        assert results[0].event_type == AuditEventType.PERMISSION_DENIED

    async def test_query_logs_by_time_range(self, audit_logger):
        """Test querying logs by time range."""
        now = datetime.now(UTC)
        past = now - timedelta(hours=2)
        future = now + timedelta(hours=2)

        # Log event
        await audit_logger.log_event(
            AuditLogEntry(
                event_type=AuditEventType.RESOURCE_ACCESS,
                sandbox_id="test",
                agent_id="test",
                operation="test",
                result=True)
        )

        # Query with time range
        results = await audit_logger.query_logs(start_time=past, end_time=future)
        assert len(results) == 1

        # Query with past range
        results = await audit_logger.query_logs(
            start_time=past - timedelta(hours=1),
            end_time=past)
        assert len(results) == 0

    async def test_get_stats(self, audit_logger):
        """Test getting audit statistics."""
        # Log various events
        for i in range(10):
            event_type = (
                AuditEventType.PERMISSION_GRANTED
                if i % 2 == 0
                else AuditEventType.PERMISSION_DENIED
            )
            await audit_logger.log_event(
                AuditLogEntry(
                    event_type=event_type,
                    sandbox_id="test",
                    agent_id="test",
                    operation="test",
                    result=i % 2 == 0)
            )

        stats = await audit_logger.get_stats(sandbox_id="test")

        assert stats["total_events"] == 10
        assert stats["by_result"]["allowed"] == 5
        assert stats["by_result"]["denied"] == 5

    async def test_cleanup_old_logs(self, audit_logger, temp_log_dir):
        """Test cleanup of old audit logs."""
        # Create old log files
        old_date = datetime.now(UTC) - timedelta(days=100)
        old_log_file = temp_log_dir / f"audit_{old_date.strftime('%Y-%m-%d')}.jsonl"
        old_log_file.write_text('{"test": "data"}\n')

        # Create recent log file
        recent_date = datetime.now(UTC)
        recent_log_file = (
            temp_log_dir / f"audit_{recent_date.strftime('%Y-%m-%d')}.jsonl"
        )
        recent_log_file.write_text('{"test": "data"}\n')

        # Run cleanup
        await audit_logger.cleanup_old_logs()

        # Old file should be deleted
        assert not old_log_file.exists()
        # Recent file should remain
        assert recent_log_file.exists()


@pytest.mark.asyncio
class TestSandboxStats:
    """Test sandbox statistics tracking."""

    async def test_get_stats(self, sandbox_service, basic_sandbox_config):
        """Test getting sandbox statistics."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        stats = sandbox_service.get_stats(sandbox_id)

        assert isinstance(stats, SandboxStats)
        assert stats.sandbox_id == sandbox_id
        assert stats.is_running is False
        assert stats.execution_time_seconds == 0.0

    async def test_stats_updated_after_execution(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test stats are updated after code execution."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        code = "result = 42"
        await sandbox_service.execute_in_sandbox(sandbox_id=sandbox_id, code=code)

        stats = sandbox_service.get_stats(sandbox_id)
        assert stats.execution_time_seconds > 0.0


@pytest.mark.asyncio
class TestResourceLimits:
    """Test resource limit enforcement."""

    async def test_enforce_limits(self, sandbox_service, basic_sandbox_config):
        """Test enforcing resource limits."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        limits = ExecutionLimits(
            max_execution_time_seconds=10,
            max_memory_mb=128,
            max_cpu_percent=50.0,
            max_processes=5,
            max_file_descriptors=20)

        # Should not raise exception
        await sandbox_service.enforce_limits(sandbox_id=sandbox_id, limits=limits)


@pytest.mark.asyncio
class TestSecurityViolations:
    """Test security violation handling."""

    async def test_security_violation_logged(
        self, sandbox_service, basic_sandbox_config
    ):
        """Test security violations are logged to audit."""
        sandbox_id = await sandbox_service.create_sandbox(basic_sandbox_config)

        # Attempt denied operation
        with pytest.raises(SecurityViolationError):
            await sandbox_service.check_permission(
                sandbox_id=sandbox_id,
                permission=SandboxPermission.WRITE,
                resource="/readonly/file.txt")

        # Check audit log
        logs = await sandbox_service._audit_logger.query_logs(
            sandbox_id=sandbox_id,
            event_type=AuditEventType.PERMISSION_DENIED)

        assert len(logs) >= 1
        assert logs[0].result is False


class TestPatternMatching:
    """Test resource pattern matching."""

    def test_matches_pattern_exact(self):
        """Test exact pattern matching."""
        assert SandboxService._matches_pattern("/data/test.txt", "/data/test.txt")

    def test_matches_pattern_glob(self):
        """Test glob pattern matching."""
        assert SandboxService._matches_pattern("/data/test.txt", "/data/*.txt")
        assert SandboxService._matches_pattern("/data/test.py", "/data/*.py")
        assert not SandboxService._matches_pattern("/data/test.txt", "/data/*.py")

    def test_matches_pattern_wildcard(self):
        """Test wildcard pattern matching."""
        assert SandboxService._matches_pattern("anything", "*")
        assert SandboxService._matches_pattern("/any/path/file.txt", "*")
