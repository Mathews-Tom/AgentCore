"""Unit tests for audit logger.

Tests audit event logging, tamper-proof chain, and event querying.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.integration.security.audit_logger import (
    AuditAction,
    AuditEvent,
    AuditLogger,
    AuditOutcome)


class TestAuditLogger:
    """Test audit logger functionality."""

    def test_initialization(self) -> None:
        """Test audit logger initialization."""
        logger = AuditLogger()
        assert logger is not None

    def test_log_event(self) -> None:
        """Test logging an audit event."""
        logger = AuditLogger()

        event = logger.log_event(
            user_id="user-123",
            service_name="test-service",
            action=AuditAction.CREDENTIAL_ACCESSED,
            resource="cred-001",
            outcome=AuditOutcome.SUCCESS,
            ip_address="192.168.1.1",
            details={"purpose": "test"})

        assert event.user_id == "user-123"
        assert event.service_name == "test-service"
        assert event.action == AuditAction.CREDENTIAL_ACCESSED
        assert event.resource == "cred-001"
        assert event.outcome == AuditOutcome.SUCCESS
        assert event.ip_address == "192.168.1.1"
        assert event.details == {"purpose": "test"}
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_hash_computation(self) -> None:
        """Test tamper-proof hash computation."""
        logger = AuditLogger()

        event = logger.log_event(
            user_id="user-123",
            service_name="test",
            action=AuditAction.DATA_READ,
            resource="data-001",
            outcome=AuditOutcome.SUCCESS)

        # Event should have hash
        assert event.event_hash is not None
        assert len(event.event_hash) == 64  # SHA256 hex

        # Recompute and verify
        recomputed = event.compute_hash(event.previous_hash)
        assert recomputed == event.event_hash

    def test_audit_chain_linking(self) -> None:
        """Test that events are linked in tamper-proof chain."""
        logger = AuditLogger()

        # Log first event
        event1 = logger.log_event(
            user_id="user-1",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)

        # Log second event
        event2 = logger.log_event(
            user_id="user-2",
            service_name="service",
            action=AuditAction.DATA_WRITE,
            resource="resource-2",
            outcome=AuditOutcome.SUCCESS)

        # First event should have no previous hash
        assert event1.previous_hash is None

        # Second event should link to first
        assert event2.previous_hash == event1.event_hash

    def test_verify_chain_valid(self) -> None:
        """Test chain verification with valid chain."""
        logger = AuditLogger()

        # Log multiple events
        for i in range(5):
            logger.log_event(
                user_id=f"user-{i}",
                service_name="service",
                action=AuditAction.DATA_READ,
                resource=f"resource-{i}",
                outcome=AuditOutcome.SUCCESS)

        # Verify chain
        assert logger.verify_chain()

    def test_verify_chain_tampered(self) -> None:
        """Test chain verification detects tampering."""
        logger = AuditLogger()

        # Log events
        event1 = logger.log_event(
            user_id="user-1",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)

        event2 = logger.log_event(
            user_id="user-2",
            service_name="service",
            action=AuditAction.DATA_WRITE,
            resource="resource-2",
            outcome=AuditOutcome.SUCCESS)

        # Tamper with event
        event1.user_id = "tampered-user"

        # Chain verification should fail
        assert not logger.verify_chain()

    def test_get_events_all(self) -> None:
        """Test retrieving all events."""
        logger = AuditLogger()

        # Log multiple events
        for i in range(3):
            logger.log_event(
                user_id=f"user-{i}",
                service_name="service",
                action=AuditAction.DATA_READ,
                resource=f"resource-{i}",
                outcome=AuditOutcome.SUCCESS)

        events = logger.get_events()
        assert len(events) == 3

    def test_get_events_filtered_by_user(self) -> None:
        """Test filtering events by user."""
        logger = AuditLogger()

        # Log events for different users
        logger.log_event(
            user_id="user-1",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)
        logger.log_event(
            user_id="user-2",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-2",
            outcome=AuditOutcome.SUCCESS)
        logger.log_event(
            user_id="user-1",
            service_name="service",
            action=AuditAction.DATA_WRITE,
            resource="resource-3",
            outcome=AuditOutcome.SUCCESS)

        # Filter by user
        events = logger.get_events(user_id="user-1")
        assert len(events) == 2
        assert all(e.user_id == "user-1" for e in events)

    def test_get_events_filtered_by_service(self) -> None:
        """Test filtering events by service."""
        logger = AuditLogger()

        # Log events for different services
        logger.log_event(
            user_id="user",
            service_name="service-1",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)
        logger.log_event(
            user_id="user",
            service_name="service-2",
            action=AuditAction.DATA_READ,
            resource="resource-2",
            outcome=AuditOutcome.SUCCESS)

        # Filter by service
        events = logger.get_events(service_name="service-1")
        assert len(events) == 1
        assert events[0].service_name == "service-1"

    def test_get_events_filtered_by_action(self) -> None:
        """Test filtering events by action."""
        logger = AuditLogger()

        # Log different actions
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_WRITE,
            resource="resource-2",
            outcome=AuditOutcome.SUCCESS)

        # Filter by action
        events = logger.get_events(action=AuditAction.DATA_READ)
        assert len(events) == 1
        assert events[0].action == AuditAction.DATA_READ

    def test_get_events_filtered_by_outcome(self) -> None:
        """Test filtering events by outcome."""
        logger = AuditLogger()

        # Log different outcomes
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-2",
            outcome=AuditOutcome.FAILURE)

        # Filter by outcome
        events = logger.get_events(outcome=AuditOutcome.FAILURE)
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.FAILURE

    def test_get_events_filtered_by_time_range(self) -> None:
        """Test filtering events by time range."""
        logger = AuditLogger()

        now = datetime.now(UTC)

        # Log event in the past
        past_event = logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="past",
            outcome=AuditOutcome.SUCCESS)
        past_event.timestamp = now - timedelta(hours=2)

        # Log current event
        current_event = logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="current",
            outcome=AuditOutcome.SUCCESS)

        # Filter by time range
        events = logger.get_events(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1))

        # Should only get current event
        assert len(events) == 1
        assert events[0].resource == "current"

    def test_get_events_limited(self) -> None:
        """Test event limit."""
        logger = AuditLogger()

        # Log many events
        for i in range(10):
            logger.log_event(
                user_id="user",
                service_name="service",
                action=AuditAction.DATA_READ,
                resource=f"resource-{i}",
                outcome=AuditOutcome.SUCCESS)

        # Get with limit
        events = logger.get_events(limit=5)
        assert len(events) == 5

    def test_get_events_sorted_newest_first(self) -> None:
        """Test events are sorted newest first."""
        logger = AuditLogger()

        # Log events with timestamps
        for i in range(3):
            logger.log_event(
                user_id="user",
                service_name="service",
                action=AuditAction.DATA_READ,
                resource=f"resource-{i}",
                outcome=AuditOutcome.SUCCESS)

        events = logger.get_events()

        # Check sorted newest first
        for i in range(len(events) - 1):
            assert events[i].timestamp >= events[i + 1].timestamp

    def test_get_security_events(self) -> None:
        """Test retrieving security-related events."""
        logger = AuditLogger()

        # Log security events
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.AUTHENTICATION_FAILURE,
            resource="login",
            outcome=AuditOutcome.FAILURE)
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.AUTHORIZATION_FAILURE,
            resource="admin",
            outcome=AuditOutcome.DENIED)

        # Log non-security event
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="data",
            outcome=AuditOutcome.SUCCESS)

        # Get security events
        security_events = logger.get_security_events()
        assert len(security_events) == 2
        assert all(
            e.action
            in {
                AuditAction.AUTHENTICATION_FAILURE,
                AuditAction.AUTHORIZATION_FAILURE,
                AuditAction.SECURITY_VIOLATION,
            }
            for e in security_events
        )

    def test_get_user_activity(self) -> None:
        """Test retrieving user activity."""
        logger = AuditLogger()

        # Log activity for user
        logger.log_event(
            user_id="target-user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-1",
            outcome=AuditOutcome.SUCCESS)
        logger.log_event(
            user_id="other-user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource-2",
            outcome=AuditOutcome.SUCCESS)

        # Get user activity
        activity = logger.get_user_activity("target-user")
        assert len(activity) == 1
        assert activity[0].user_id == "target-user"

    def test_export_events_json(self) -> None:
        """Test exporting events as JSON."""
        logger = AuditLogger()

        # Log events
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource",
            outcome=AuditOutcome.SUCCESS)

        # Export
        json_str = logger.export_events()
        assert json_str is not None
        assert "user" in json_str
        assert "service" in json_str

    def test_cleanup_old_events(self) -> None:
        """Test cleanup of old events."""
        logger = AuditLogger(retention_days=30)

        now = datetime.now(UTC)

        # Log old event
        old_event = logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="old",
            outcome=AuditOutcome.SUCCESS)
        old_event.timestamp = now - timedelta(days=31)

        # Log recent event
        logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="recent",
            outcome=AuditOutcome.SUCCESS)

        # Cleanup
        removed = logger.cleanup_old_events()

        # Should remove old event
        assert removed == 1

        # Should only have recent event
        events = logger.get_events()
        assert len(events) == 1
        assert events[0].resource == "recent"

    def test_siem_integration_flag(self) -> None:
        """Test SIEM integration configuration."""
        logger = AuditLogger(
            enable_siem=True,
            siem_endpoint="https://siem.example.com/events")

        # Log event (should forward to SIEM)
        event = logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource",
            outcome=AuditOutcome.SUCCESS)

        # Event should be logged
        assert event is not None

    def test_event_with_error_message(self) -> None:
        """Test logging event with error message."""
        logger = AuditLogger()

        event = logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource",
            outcome=AuditOutcome.ERROR,
            error_message="Database connection failed")

        assert event.error_message == "Database connection failed"

    def test_event_with_user_agent(self) -> None:
        """Test logging event with user agent."""
        logger = AuditLogger()

        event = logger.log_event(
            user_id="user",
            service_name="service",
            action=AuditAction.DATA_READ,
            resource="resource",
            outcome=AuditOutcome.SUCCESS,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

        assert event.user_agent == "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
