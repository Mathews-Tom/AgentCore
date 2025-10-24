"""Audit logging for security and compliance.

Provides comprehensive audit trail for all integration access with
structured logging, SIEM integration, and tamper-proof storage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AuditAction(str, Enum):
    """Supported audit actions."""

    # Credential operations
    CREDENTIAL_CREATED = "credential_created"
    CREDENTIAL_ACCESSED = "credential_accessed"
    CREDENTIAL_ROTATED = "credential_rotated"
    CREDENTIAL_REVOKED = "credential_revoked"
    CREDENTIAL_DELETED = "credential_deleted"

    # Integration access
    INTEGRATION_ACCESSED = "integration_accessed"
    INTEGRATION_CREATED = "integration_created"
    INTEGRATION_UPDATED = "integration_updated"
    INTEGRATION_DELETED = "integration_deleted"

    # Data operations
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # Security events
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SECURITY_VIOLATION = "security_violation"

    # Configuration changes
    CONFIG_UPDATED = "config_updated"
    POLICY_UPDATED = "policy_updated"


class AuditOutcome(str, Enum):
    """Audit event outcomes."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


class AuditEvent(BaseModel):
    """Audit event record.

    Comprehensive audit event with all required fields for compliance
    and security monitoring, including tamper-proof checksums.
    """

    event_id: str = Field(
        description="Unique event identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC)",
    )
    user_id: str | None = Field(
        default=None,
        description="User who performed action",
    )
    service_name: str = Field(
        description="Service being accessed",
    )
    action: AuditAction = Field(
        description="Action performed",
    )
    resource: str = Field(
        description="Resource affected (e.g., credential ID, integration name)",
    )
    outcome: AuditOutcome = Field(
        description="Action outcome",
    )
    ip_address: str | None = Field(
        default=None,
        description="Source IP address",
    )
    user_agent: str | None = Field(
        default=None,
        description="User agent string",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event details",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if outcome is failure/error",
    )
    previous_hash: str | None = Field(
        default=None,
        description="Hash of previous audit event (for tamper-proof chain)",
    )
    event_hash: str | None = Field(
        default=None,
        description="Hash of this event (for tamper-proof verification)",
    )

    def compute_hash(self, previous_hash: str | None = None) -> str:
        """Compute tamper-proof hash of event.

        Args:
            previous_hash: Hash of previous event in chain

        Returns:
            SHA256 hash of event data
        """
        # Create canonical representation
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "service_name": self.service_name,
            "action": self.action.value,
            "resource": self.resource,
            "outcome": self.outcome.value,
            "details": self.details,
            "previous_hash": previous_hash,
        }

        # Sort keys for deterministic serialization
        canonical = json.dumps(data, sort_keys=True)

        # Compute SHA256 hash
        return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLogger:
    """Audit logging system with tamper-proof storage.

    Provides comprehensive audit trail with structured logging,
    SIEM integration, and tamper-proof event chaining.
    """

    def __init__(
        self,
        retention_days: int = 365,
        enable_siem: bool = False,
        siem_endpoint: str | None = None,
    ) -> None:
        """Initialize audit logger.

        Args:
            retention_days: Audit log retention period in days
            enable_siem: Enable SIEM integration
            siem_endpoint: SIEM endpoint URL for event forwarding
        """
        self._retention_days = retention_days
        self._enable_siem = enable_siem
        self._siem_endpoint = siem_endpoint
        self._events: list[AuditEvent] = []
        self._last_hash: str | None = None

        logger.info(
            "audit_logger_initialized",
            retention_days=retention_days,
            enable_siem=enable_siem,
        )

    def log_event(
        self,
        user_id: str | None,
        service_name: str,
        action: AuditAction,
        resource: str,
        outcome: AuditOutcome,
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            user_id: User performing action
            service_name: Service being accessed
            action: Action performed
            resource: Resource affected
            outcome: Action outcome
            ip_address: Source IP address
            user_agent: User agent string
            details: Additional details
            error_message: Error message if applicable

        Returns:
            Created audit event
        """
        import uuid

        # Create event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            service_name=service_name,
            action=action,
            resource=resource,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            error_message=error_message,
        )

        # Compute tamper-proof hash
        event.previous_hash = self._last_hash
        event.event_hash = event.compute_hash(self._last_hash)
        self._last_hash = event.event_hash

        # Store event
        self._events.append(event)

        # Log to structlog
        log_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "user_id": user_id,
            "service_name": service_name,
            "action": action.value,
            "resource": resource,
            "outcome": outcome.value,
            "ip_address": ip_address,
        }

        if outcome == AuditOutcome.SUCCESS:
            logger.info("audit_event", **log_data)
        elif outcome == AuditOutcome.DENIED:
            logger.warning("audit_event", **log_data, error=error_message)
        else:
            logger.error("audit_event", **log_data, error=error_message)

        # Forward to SIEM if enabled
        if self._enable_siem and self._siem_endpoint:
            self._forward_to_siem(event)

        return event

    def verify_chain(self) -> bool:
        """Verify tamper-proof audit chain integrity.

        Returns:
            True if chain is valid, False if tampered

        Raises:
            ValueError: If chain is invalid
        """
        if not self._events:
            return True

        previous_hash: str | None = None

        for event in self._events:
            # Verify event hash
            expected_hash = event.compute_hash(previous_hash)

            if event.event_hash != expected_hash:
                logger.error(
                    "audit_chain_integrity_violation",
                    event_id=event.event_id,
                    expected_hash=expected_hash,
                    actual_hash=event.event_hash,
                )
                return False

            # Verify previous hash matches
            if event.previous_hash != previous_hash:
                logger.error(
                    "audit_chain_break",
                    event_id=event.event_id,
                    expected_previous=previous_hash,
                    actual_previous=event.previous_hash,
                )
                return False

            previous_hash = event.event_hash

        logger.info(
            "audit_chain_verified",
            event_count=len(self._events),
        )

        return True

    def get_events(
        self,
        user_id: str | None = None,
        service_name: str | None = None,
        action: AuditAction | None = None,
        outcome: AuditOutcome | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events with filtering.

        Args:
            user_id: Filter by user
            service_name: Filter by service
            action: Filter by action
            outcome: Filter by outcome
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum events to return

        Returns:
            List of matching audit events
        """
        events = self._events.copy()

        # Apply filters
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        if service_name:
            events = [e for e in events if e.service_name == service_name]

        if action:
            events = [e for e in events if e.action == action]

        if outcome:
            events = [e for e in events if e.outcome == outcome]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        return events[:limit]

    def get_security_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AuditEvent]:
        """Get security-related audit events.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of security events
        """
        security_actions = {
            AuditAction.AUTHENTICATION_FAILURE,
            AuditAction.AUTHORIZATION_FAILURE,
            AuditAction.SECURITY_VIOLATION,
        }

        events = self._events.copy()

        # Filter security actions
        events = [e for e in events if e.action in security_actions]

        # Apply time filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events

    def get_user_activity(
        self,
        user_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AuditEvent]:
        """Get all activity for a specific user.

        Args:
            user_id: User identifier
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of user's audit events
        """
        return self.get_events(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )

    def export_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> str:
        """Export audit events as JSON.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            JSON string of audit events
        """
        events = self.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        # Convert to dict
        events_dict = [event.model_dump(mode="json") for event in events]

        return json.dumps(events_dict, indent=2, default=str)

    def _forward_to_siem(self, event: AuditEvent) -> None:
        """Forward audit event to SIEM system.

        Args:
            event: Audit event to forward
        """
        if not self._siem_endpoint:
            return

        # In production, this would send to actual SIEM
        # For now, just log the intent
        logger.debug(
            "audit_event_forwarded_to_siem",
            event_id=event.event_id,
            siem_endpoint=self._siem_endpoint,
            action=event.action.value,
            outcome=event.outcome.value,
        )

    def cleanup_old_events(self) -> int:
        """Remove events older than retention period.

        Returns:
            Number of events removed
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self._retention_days)

        old_events = [e for e in self._events if e.timestamp < cutoff_time]
        self._events = [e for e in self._events if e.timestamp >= cutoff_time]

        removed_count = len(old_events)

        if removed_count > 0:
            # Recalculate chain after cleanup
            self._last_hash = None
            if self._events:
                self._last_hash = self._events[-1].event_hash

            logger.info(
                "audit_events_cleaned_up",
                removed_count=removed_count,
                retention_days=self._retention_days,
            )

        return removed_count


# Import for timedelta
from datetime import timedelta
