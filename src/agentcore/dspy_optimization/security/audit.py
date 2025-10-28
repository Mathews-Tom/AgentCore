"""
Audit Trail System

Provides comprehensive operation logging, change tracking, and compliance reporting.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class AuditEventType(str, Enum):
    """Types of audit events"""

    # Authentication
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"

    # Model operations
    MODEL_CREATE = "model.create"
    MODEL_READ = "model.read"
    MODEL_UPDATE = "model.update"
    MODEL_DELETE = "model.delete"
    MODEL_ENCRYPT = "model.encrypt"
    MODEL_DECRYPT = "model.decrypt"

    # Optimization operations
    OPTIMIZATION_START = "optimization.start"
    OPTIMIZATION_COMPLETE = "optimization.complete"
    OPTIMIZATION_FAILED = "optimization.failed"
    OPTIMIZATION_CANCEL = "optimization.cancel"

    # Data operations
    DATA_ACCESS = "data.access"
    DATA_MODIFY = "data.modify"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"

    # Security operations
    PERMISSION_GRANTED = "security.permission_granted"
    PERMISSION_REVOKED = "security.permission_revoked"
    ROLE_ASSIGNED = "security.role_assigned"
    KEY_ROTATED = "security.key_rotated"

    # Compliance
    COMPLIANCE_CHECK = "compliance.check"
    COMPLIANCE_VIOLATION = "compliance.violation"

    # System
    SYSTEM_CONFIG_CHANGE = "system.config_change"
    SYSTEM_ERROR = "system.error"


class AuditSeverity(str, Enum):
    """Audit event severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Audit event record"""

    event_id: str = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_id: str | None = None
    session_token: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


class AuditConfig(BaseModel):
    """Configuration for audit logging"""

    enable_audit: bool = Field(default=True, description="Enable audit logging")
    log_to_file: bool = Field(default=True, description="Log to file")
    log_directory: Path = Field(
        default=Path("logs/audit"), description="Audit log directory"
    )
    rotation_days: int = Field(default=90, description="Log rotation period in days")
    retention_days: int = Field(default=365, description="Log retention period in days")
    log_sensitive_data: bool = Field(
        default=False, description="Log sensitive data (not recommended)"
    )


class AuditLogger:
    """
    Audit logging service.

    Provides comprehensive operation logging and change tracking.
    """

    def __init__(self, config: AuditConfig | None = None):
        self.config = config or AuditConfig()
        self.logger = structlog.get_logger()

        # In-memory audit trail (limited to recent events)
        self._audit_trail: list[AuditEvent] = []
        self._max_memory_events = 10000

        # Statistics
        self._audit_stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "failed_operations": 0,
            "security_events": 0,
        }

        # Initialize log directory
        if self.config.log_to_file:
            self.config.log_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "audit_logger_initialized",
            log_to_file=self.config.log_to_file,
            log_directory=str(self.config.log_directory),
        )

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: str | None = None,
        session_token: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditEvent:
        """
        Log audit event.

        Args:
            event_type: Type of event
            action: Action description
            user_id: User identifier
            session_token: Session token
            resource_type: Type of resource
            resource_id: Resource identifier
            severity: Event severity
            details: Additional details
            success: Whether operation succeeded
            error_message: Error message if failed
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created AuditEvent
        """
        if not self.config.enable_audit:
            return None

        # Filter sensitive data if not allowed
        if not self.config.log_sensitive_data and details:
            details = self._filter_sensitive_data(details)

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_token=session_token,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Add to in-memory trail
        self._audit_trail.append(event)
        if len(self._audit_trail) > self._max_memory_events:
            self._audit_trail.pop(0)

        # Update statistics
        self._audit_stats["total_events"] += 1
        self._audit_stats["events_by_type"][event_type.value] = (
            self._audit_stats["events_by_type"].get(event_type.value, 0) + 1
        )
        self._audit_stats["events_by_severity"][severity.value] = (
            self._audit_stats["events_by_severity"].get(severity.value, 0) + 1
        )

        if not success:
            self._audit_stats["failed_operations"] += 1

        if event_type.value.startswith("security."):
            self._audit_stats["security_events"] += 1

        # Write to file
        if self.config.log_to_file:
            self._write_to_file(event)

        # Log to structlog
        self.logger.info(
            "audit_event",
            event_id=event.event_id,
            event_type=event_type.value,
            action=action,
            success=success,
        )

        return event

    def _filter_sensitive_data(self, details: dict[str, Any]) -> dict[str, Any]:
        """Filter sensitive data from details"""
        sensitive_keys = {"password", "token", "secret", "key", "api_key"}
        filtered = {}

        for key, value in details.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value

        return filtered

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write audit event to file"""
        try:
            # Create daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.config.log_directory / f"audit_{date_str}.jsonl"

            # Write as JSON line
            with open(log_file, "a") as f:
                f.write(event.model_dump_json() + "\n")

        except Exception as e:
            self.logger.error("audit_write_failed", error=str(e))

    def get_events(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """
        Query audit events.

        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            resource_id: Filter by resource ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of events to return

        Returns:
            List of matching audit events
        """
        filtered = self._audit_trail

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if resource_id:
            filtered = [e for e in filtered if e.resource_id == resource_id]

        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        # Sort by timestamp descending
        filtered = sorted(filtered, key=lambda e: e.timestamp, reverse=True)

        return filtered[:limit]

    def get_user_activity(
        self, user_id: str, days: int = 30
    ) -> list[AuditEvent]:
        """
        Get user activity for specified period.

        Args:
            user_id: User identifier
            days: Number of days to look back

        Returns:
            List of user audit events
        """
        from datetime import timedelta

        start_time = datetime.now(UTC) - timedelta(days=days)
        return self.get_events(user_id=user_id, start_time=start_time)

    def get_security_events(self, days: int = 7) -> list[AuditEvent]:
        """
        Get security-related events.

        Args:
            days: Number of days to look back

        Returns:
            List of security audit events
        """
        from datetime import timedelta

        start_time = datetime.now(UTC) - timedelta(days=days)
        security_events = [
            e
            for e in self._audit_trail
            if e.event_type.value.startswith("security.")
            or e.event_type.value.startswith("auth.")
        ]

        return [e for e in security_events if e.timestamp >= start_time]

    def get_failed_operations(self, days: int = 7) -> list[AuditEvent]:
        """
        Get failed operations.

        Args:
            days: Number of days to look back

        Returns:
            List of failed operation events
        """
        from datetime import timedelta

        start_time = datetime.now(UTC) - timedelta(days=days)
        failed = [e for e in self._audit_trail if not e.success]

        return [e for e in failed if e.timestamp >= start_time]

    def generate_compliance_report(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, Any]:
        """
        Generate compliance report for time period.

        Args:
            start_time: Report start time
            end_time: Report end time

        Returns:
            Compliance report with statistics and events
        """
        events = self.get_events(start_time=start_time, end_time=end_time, limit=100000)

        return {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "total_events": len(events),
            "events_by_type": self._count_by_field(events, "event_type"),
            "events_by_severity": self._count_by_field(events, "severity"),
            "unique_users": len({e.user_id for e in events if e.user_id}),
            "failed_operations": len([e for e in events if not e.success]),
            "security_events": len([e for e in events if e.event_type.value.startswith("security.") or e.event_type.value.startswith("auth.")]),
            "critical_events": [
                e.model_dump()
                for e in events
                if e.severity == AuditSeverity.CRITICAL
            ],
            "compliance_violations": [
                e.model_dump()
                for e in events
                if e.event_type == AuditEventType.COMPLIANCE_VIOLATION
            ],
        }

    def _count_by_field(
        self, events: list[AuditEvent], field: str
    ) -> dict[str, int]:
        """Count events by field"""
        counts: dict[str, int] = {}
        for event in events:
            value = getattr(event, field)
            key = value.value if hasattr(value, "value") else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def cleanup_old_logs(self) -> int:
        """
        Clean up old audit logs.

        Returns:
            Number of log files deleted
        """
        if not self.config.log_to_file:
            return 0

        from datetime import timedelta

        cutoff_date = datetime.now(UTC) - timedelta(days=self.config.retention_days)
        deleted_count = 0

        for log_file in self.config.log_directory.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)

                if file_date < cutoff_date:
                    log_file.unlink()
                    deleted_count += 1

            except Exception as e:
                self.logger.error(
                    "log_cleanup_failed", file=str(log_file), error=str(e)
                )

        if deleted_count > 0:
            self.logger.info("old_logs_cleaned", count=deleted_count)

        return deleted_count

    def get_audit_stats(self) -> dict[str, Any]:
        """Get audit statistics"""
        return {
            **self._audit_stats,
            "memory_events": len(self._audit_trail),
            "max_memory_events": self._max_memory_events,
        }
