"""
Tests for audit trail system
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentcore.dspy_optimization.security.audit import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
)


class TestAuditConfig:
    """Tests for AuditConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = AuditConfig()
        assert config.enable_audit is True
        assert config.log_to_file is True
        assert config.rotation_days == 90
        assert config.retention_days == 365
        assert config.log_sensitive_data is False


class TestAuditEvent:
    """Tests for AuditEvent"""

    def test_create_audit_event(self):
        """Test creating audit event"""
        event = AuditEvent(
            event_type=AuditEventType.MODEL_CREATE,
            action="Created new model",
            user_id="user1",
        )

        assert event.event_id is not None
        assert event.event_type == AuditEventType.MODEL_CREATE
        assert event.action == "Created new model"
        assert event.user_id == "user1"
        assert event.success is True
        assert event.severity == AuditSeverity.INFO

    def test_event_with_details(self):
        """Test event with additional details"""
        details = {"model_id": "123", "size": "1.5MB"}
        event = AuditEvent(
            event_type=AuditEventType.MODEL_CREATE,
            action="Created model",
            details=details,
        )

        assert event.details == details


class TestAuditLogger:
    """Tests for AuditLogger"""

    @pytest.fixture
    def logger(self, tmp_path: Path) -> AuditLogger:
        """Create audit logger with temp directory"""
        config = AuditConfig(log_directory=tmp_path / "audit_logs")
        return AuditLogger(config)

    def test_initialization(self, logger: AuditLogger):
        """Test logger initialization"""
        assert logger.config.enable_audit is True
        assert len(logger._audit_trail) == 0

    def test_log_event(self, logger: AuditLogger):
        """Test logging event"""
        event = logger.log_event(
            event_type=AuditEventType.MODEL_CREATE,
            action="Created new model",
            user_id="user1",
        )

        assert event is not None
        assert event.event_type == AuditEventType.MODEL_CREATE
        assert len(logger._audit_trail) == 1

    def test_log_event_with_details(self, logger: AuditLogger):
        """Test logging event with details"""
        details = {"model_id": "123", "version": "1.0"}
        event = logger.log_event(
            event_type=AuditEventType.MODEL_CREATE,
            action="Created model",
            user_id="user1",
            details=details,
        )

        assert event.details == details

    def test_log_failed_operation(self, logger: AuditLogger):
        """Test logging failed operation"""
        event = logger.log_event(
            event_type=AuditEventType.MODEL_CREATE,
            action="Failed to create model",
            user_id="user1",
            success=False,
            error_message="Insufficient permissions",
            severity=AuditSeverity.ERROR,
        )

        assert event.success is False
        assert event.error_message == "Insufficient permissions"
        assert event.severity == AuditSeverity.ERROR

    def test_filter_sensitive_data(self, logger: AuditLogger):
        """Test sensitive data filtering"""
        details = {
            "model_id": "123",
            "api_key": "secret123",
            "password": "pass123",
        }

        event = logger.log_event(
            event_type=AuditEventType.SYSTEM_CONFIG_CHANGE,
            action="Updated config",
            details=details,
        )

        assert event.details["model_id"] == "123"
        assert event.details["api_key"] == "[REDACTED]"
        assert event.details["password"] == "[REDACTED]"

    def test_get_events_no_filter(self, logger: AuditLogger):
        """Test getting all events"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create 1", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_UPDATE, action="Update 1", user_id="user2"
        )

        events = logger.get_events()
        assert len(events) == 2

    def test_get_events_by_type(self, logger: AuditLogger):
        """Test filtering events by type"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_UPDATE, action="Update", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create 2", user_id="user2"
        )

        events = logger.get_events(event_type=AuditEventType.MODEL_CREATE)
        assert len(events) == 2
        assert all(e.event_type == AuditEventType.MODEL_CREATE for e in events)

    def test_get_events_by_user(self, logger: AuditLogger):
        """Test filtering events by user"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_UPDATE, action="Update", user_id="user2"
        )

        events = logger.get_events(user_id="user1")
        assert len(events) == 1
        assert events[0].user_id == "user1"

    def test_get_events_by_time_range(self, logger: AuditLogger):
        """Test filtering events by time range"""
        start_time = datetime.now(UTC)

        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )

        end_time = datetime.now(UTC)

        events = logger.get_events(start_time=start_time, end_time=end_time)
        assert len(events) >= 1

    def test_get_events_with_limit(self, logger: AuditLogger):
        """Test event limit"""
        for i in range(10):
            logger.log_event(
                event_type=AuditEventType.MODEL_CREATE,
                action=f"Create {i}",
                user_id="user1",
            )

        events = logger.get_events(limit=5)
        assert len(events) == 5

    def test_get_user_activity(self, logger: AuditLogger):
        """Test getting user activity"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_UPDATE, action="Update", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_DELETE, action="Delete", user_id="user2"
        )

        activity = logger.get_user_activity("user1")
        assert len(activity) == 2
        assert all(e.user_id == "user1" for e in activity)

    def test_get_security_events(self, logger: AuditLogger):
        """Test getting security events"""
        logger.log_event(
            event_type=AuditEventType.AUTH_LOGIN, action="Login", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.PERMISSION_GRANTED,
            action="Grant permission",
            user_id="admin",
        )

        security_events = logger.get_security_events()
        assert len(security_events) == 2
        assert all(
            e.event_type.value.startswith(("security.", "auth."))
            for e in security_events
        )

    def test_get_failed_operations(self, logger: AuditLogger):
        """Test getting failed operations"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE,
            action="Create",
            user_id="user1",
            success=True,
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_DELETE,
            action="Delete",
            user_id="user1",
            success=False,
        )

        failed = logger.get_failed_operations()
        assert len(failed) == 1
        assert failed[0].success is False

    def test_generate_compliance_report(self, logger: AuditLogger):
        """Test compliance report generation"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.AUTH_FAILED,
            action="Failed login",
            user_id="user2",
            success=False,
        )
        logger.log_event(
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            action="Violation",
            user_id="user1",
            severity=AuditSeverity.CRITICAL,
        )

        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        report = logger.generate_compliance_report(start_time, end_time)

        assert "report_period" in report
        assert "total_events" in report
        assert "events_by_type" in report
        assert "unique_users" in report
        assert "failed_operations" in report
        assert "security_events" in report
        assert "critical_events" in report
        assert "compliance_violations" in report

    def test_write_to_file(self, logger: AuditLogger, tmp_path: Path):
        """Test writing events to file"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )

        # Check log file was created
        log_files = list(logger.config.log_directory.glob("audit_*.jsonl"))
        assert len(log_files) > 0

    def test_cleanup_old_logs(self, logger: AuditLogger, tmp_path: Path):
        """Test cleaning up old log files"""
        # Create old log file
        old_date = (datetime.now(UTC) - timedelta(days=400)).strftime("%Y-%m-%d")
        old_log = logger.config.log_directory / f"audit_{old_date}.jsonl"
        old_log.parent.mkdir(parents=True, exist_ok=True)
        old_log.write_text('{"test": "data"}\n')

        # Create recent log file
        recent_date = datetime.now(UTC).strftime("%Y-%m-%d")
        recent_log = logger.config.log_directory / f"audit_{recent_date}.jsonl"
        recent_log.write_text('{"test": "data"}\n')

        # Cleanup (retention is 365 days by default)
        deleted = logger.cleanup_old_logs()

        assert deleted >= 1
        assert not old_log.exists()
        assert recent_log.exists()

    def test_get_audit_stats(self, logger: AuditLogger):
        """Test getting audit statistics"""
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_UPDATE,
            action="Update",
            user_id="user1",
            success=False,
        )

        stats = logger.get_audit_stats()

        assert stats["total_events"] == 2
        assert "events_by_type" in stats
        assert "failed_operations" in stats
        assert "memory_events" in stats

    def test_memory_limit(self, logger: AuditLogger):
        """Test audit trail memory limit"""
        logger._max_memory_events = 10

        # Log more events than limit
        for i in range(15):
            logger.log_event(
                event_type=AuditEventType.MODEL_CREATE,
                action=f"Create {i}",
                user_id="user1",
            )

        # Should only keep last 10
        assert len(logger._audit_trail) == 10

    def test_disabled_audit(self):
        """Test with audit disabled"""
        config = AuditConfig(enable_audit=False)
        logger = AuditLogger(config)

        event = logger.log_event(
            event_type=AuditEventType.MODEL_CREATE, action="Create", user_id="user1"
        )

        assert event is None
        assert len(logger._audit_trail) == 0


class TestAuditIntegration:
    """Integration tests for audit system"""

    def test_complete_audit_trail(self, tmp_path: Path):
        """Test complete audit trail workflow"""
        config = AuditConfig(log_directory=tmp_path / "audit")
        logger = AuditLogger(config)

        # Log various events
        logger.log_event(
            event_type=AuditEventType.AUTH_LOGIN, action="User login", user_id="user1"
        )
        logger.log_event(
            event_type=AuditEventType.MODEL_CREATE,
            action="Create model",
            user_id="user1",
            resource_id="model_123",
        )
        logger.log_event(
            event_type=AuditEventType.OPTIMIZATION_START,
            action="Start optimization",
            user_id="user1",
            resource_id="opt_456",
        )
        logger.log_event(
            event_type=AuditEventType.AUTH_LOGOUT, action="User logout", user_id="user1"
        )

        # Query events
        all_events = logger.get_events()
        assert len(all_events) == 4

        user_activity = logger.get_user_activity("user1")
        assert len(user_activity) == 4

        # Generate report
        start = datetime.now(UTC) - timedelta(hours=1)
        end = datetime.now(UTC)
        report = logger.generate_compliance_report(start, end)

        assert report["total_events"] == 4
        assert report["unique_users"] == 1

    def test_audit_with_all_event_types(self, tmp_path: Path):
        """Test logging all event types"""
        config = AuditConfig(log_directory=tmp_path / "audit")
        logger = AuditLogger(config)

        # Log one of each event type
        for event_type in AuditEventType:
            logger.log_event(
                event_type=event_type, action=f"Test {event_type.value}", user_id="user1"
            )

        events = logger.get_events()
        assert len(events) == len(AuditEventType)

        event_types = {e.event_type for e in events}
        assert event_types == set(AuditEventType)
