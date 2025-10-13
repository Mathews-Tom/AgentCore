"""
Alerting and notification service for agent runtime.

This module provides comprehensive alerting capabilities including threshold-based
alerts, anomaly detection, notification channels, and alert management.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(str, Enum):
    """Alert states."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""

    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


class Alert:
    """Represents a single alert instance."""

    def __init__(
        self,
        alert_id: str,
        rule_name: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize alert.

        Args:
            alert_id: Unique alert identifier
            rule_name: Name of the alerting rule
            severity: Alert severity level
            title: Alert title
            description: Alert description
            labels: Alert labels for grouping/routing
            annotations: Additional alert annotations
        """
        self.alert_id = alert_id
        self.rule_name = rule_name
        self.severity = severity
        self.title = title
        self.description = description
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.state = AlertState.ACTIVE
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.acknowledged_at: datetime | None = None
        self.resolved_at: datetime | None = None
        self.acknowledged_by: str | None = None
        self.resolution_note: str | None = None

    def acknowledge(self, acknowledged_by: str) -> None:
        """
        Acknowledge alert.

        Args:
            acknowledged_by: User/system that acknowledged
        """
        self.state = AlertState.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = acknowledged_by
        self.updated_at = datetime.now()

    def resolve(self, resolution_note: str | None = None) -> None:
        """
        Resolve alert.

        Args:
            resolution_note: Note about resolution
        """
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.now()
        self.resolution_note = resolution_note
        self.updated_at = datetime.now()

    def suppress(self) -> None:
        """Suppress alert notifications."""
        self.state = AlertState.SUPPRESSED
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert alert to dictionary.

        Returns:
            Alert data as dictionary
        """
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "labels": self.labels,
            "annotations": self.annotations,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolution_note": self.resolution_note,
        }


class AlertRule:
    """Defines conditions for triggering alerts."""

    def __init__(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        severity: AlertSeverity,
        title_template: str,
        description_template: str,
        labels: dict[str, str] | None = None,
        cooldown_seconds: int = 300,
        auto_resolve: bool = False,
    ) -> None:
        """
        Initialize alert rule.

        Args:
            name: Rule name
            condition: Function to evaluate alert condition
            severity: Alert severity
            title_template: Alert title template
            description_template: Alert description template
            labels: Default labels for alerts
            cooldown_seconds: Minimum time between alerts
            auto_resolve: Auto-resolve when condition clears
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.title_template = title_template
        self.description_template = description_template
        self.labels = labels or {}
        self.cooldown_seconds = cooldown_seconds
        self.auto_resolve = auto_resolve
        self.last_triggered: datetime | None = None
        self.active_alert_id: str | None = None

    def can_trigger(self) -> bool:
        """
        Check if rule can trigger (cooldown elapsed).

        Returns:
            True if rule can trigger
        """
        if not self.last_triggered:
            return True

        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds

    def evaluate(self, context: dict[str, Any]) -> tuple[bool, str, str]:
        """
        Evaluate rule condition.

        Args:
            context: Context data for evaluation

        Returns:
            Tuple of (should_fire, title, description)
        """
        should_fire = self.condition(context)

        if should_fire:
            title = self.title_template.format(**context)
            description = self.description_template.format(**context)
            return True, title, description

        return False, "", ""


class NotificationHandler:
    """Handles notification delivery through various channels."""

    def __init__(self) -> None:
        """Initialize notification handler."""
        self._handlers: dict[NotificationChannel, Callable[[Alert], None]] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Setup default notification handlers."""
        # Log handler (always available)
        self._handlers[NotificationChannel.LOG] = self._log_handler

    def _log_handler(self, alert: Alert) -> None:
        """
        Log notification handler.

        Args:
            alert: Alert to log
        """
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(alert.severity, logger.info)

        log_method(
            "alert_notification",
            alert_id=alert.alert_id,
            rule=alert.rule_name,
            severity=alert.severity.value,
            title=alert.title,
            description=alert.description,
        )

    def register_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Register custom notification handler.

        Args:
            channel: Notification channel
            handler: Handler function
        """
        self._handlers[channel] = handler
        logger.info("notification_handler_registered", channel=channel.value)

    async def send_notification(
        self,
        alert: Alert,
        channels: list[NotificationChannel],
    ) -> dict[str, bool]:
        """
        Send alert notification through specified channels.

        Args:
            alert: Alert to send
            channels: List of channels to use

        Returns:
            Dictionary of channel -> success status
        """
        results = {}

        for channel in channels:
            handler = self._handlers.get(channel)
            if not handler:
                logger.warning(
                    "notification_handler_not_found",
                    channel=channel.value,
                    alert_id=alert.alert_id,
                )
                results[channel.value] = False
                continue

            try:
                # Call handler (async or sync)
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)

                results[channel.value] = True
                logger.debug(
                    "notification_sent",
                    channel=channel.value,
                    alert_id=alert.alert_id,
                )

            except Exception as e:
                logger.error(
                    "notification_send_failed",
                    channel=channel.value,
                    alert_id=alert.alert_id,
                    error=str(e),
                )
                results[channel.value] = False

        return results


class AlertingService:
    """Main alerting service for agent runtime."""

    def __init__(
        self,
        enable_notifications: bool = True,
        default_channels: list[NotificationChannel] | None = None,
    ) -> None:
        """
        Initialize alerting service.

        Args:
            enable_notifications: Enable notification sending
            default_channels: Default notification channels
        """
        self._enable_notifications = enable_notifications
        self._default_channels = default_channels or [NotificationChannel.LOG]

        # Alert storage
        self._alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._max_history = 10000

        # Alert rules
        self._rules: dict[str, AlertRule] = {}

        # Notification handling
        self._notification_handler = NotificationHandler()

        # Alert grouping
        self._alert_groups: dict[str, list[str]] = defaultdict(list)

        # Background task for rule evaluation
        self._evaluation_task: asyncio.Task[None] | None = None
        self._evaluation_interval = 30  # seconds

        # Statistics
        self._stats = {
            "total_alerts": 0,
            "active_alerts": 0,
            "acknowledged_alerts": 0,
            "resolved_alerts": 0,
            "suppressed_alerts": 0,
            "notifications_sent": 0,
            "notifications_failed": 0,
        }

        logger.info(
            "alerting_service_initialized",
            notifications_enabled=enable_notifications,
        )

    def register_rule(self, rule: AlertRule) -> None:
        """
        Register alert rule.

        Args:
            rule: Alert rule to register
        """
        self._rules[rule.name] = rule
        logger.info("alert_rule_registered", rule_name=rule.name)

    def create_threshold_rule(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        comparison: str,
        severity: AlertSeverity,
        cooldown_seconds: int = 300,
    ) -> AlertRule:
        """
        Create threshold-based alert rule.

        Args:
            name: Rule name
            metric_name: Metric to monitor
            threshold: Threshold value
            comparison: Comparison operator (gt, lt, gte, lte, eq)
            severity: Alert severity
            cooldown_seconds: Cooldown period

        Returns:
            Created alert rule
        """
        operators = {
            "gt": lambda x, y: x > y,
            "lt": lambda x, y: x < y,
            "gte": lambda x, y: x >= y,
            "lte": lambda x, y: x <= y,
            "eq": lambda x, y: x == y,
        }

        op = operators.get(comparison)
        if not op:
            raise ValueError(f"Invalid comparison operator: {comparison}")

        def condition(context: dict[str, Any]) -> bool:
            value = context.get(metric_name)
            if value is None:
                return False
            return op(value, threshold)

        rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            title_template=f"{metric_name} threshold exceeded",
            description_template=f"{metric_name} is {{{metric_name}}} (threshold: {threshold})",
            labels={"type": "threshold", "metric": metric_name},
            cooldown_seconds=cooldown_seconds,
        )

        self.register_rule(rule)
        return rule

    async def trigger_alert(
        self,
        rule_name: str,
        title: str,
        description: str,
        labels: dict[str, str] | None = None,
        channels: list[NotificationChannel] | None = None,
    ) -> Alert:
        """
        Manually trigger an alert.

        Args:
            rule_name: Name of the rule
            title: Alert title
            description: Alert description
            labels: Alert labels
            channels: Notification channels

        Returns:
            Created alert
        """
        rule = self._rules.get(rule_name)
        if not rule:
            raise ValueError(f"Unknown alert rule: {rule_name}")

        alert_id = str(uuid4())
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule_name,
            severity=rule.severity,
            title=title,
            description=description,
            labels=labels or rule.labels,
        )

        # Store alert
        self._alerts[alert_id] = alert
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history.pop(0)

        # Update statistics
        self._stats["total_alerts"] += 1
        self._stats["active_alerts"] += 1

        # Update rule state
        rule.last_triggered = datetime.now()
        rule.active_alert_id = alert_id

        # Add to groups
        for label_key, label_value in alert.labels.items():
            group_key = f"{label_key}:{label_value}"
            self._alert_groups[group_key].append(alert_id)

        logger.info(
            "alert_triggered",
            alert_id=alert_id,
            rule=rule_name,
            severity=alert.severity.value,
            title=title,
        )

        # Send notifications
        if self._enable_notifications:
            channels = channels or self._default_channels
            results = await self._notification_handler.send_notification(
                alert, channels
            )

            success_count = sum(1 for success in results.values() if success)
            failed_count = len(results) - success_count

            self._stats["notifications_sent"] += success_count
            self._stats["notifications_failed"] += failed_count

        return alert

    async def evaluate_rules(self, context: dict[str, Any]) -> list[Alert]:
        """
        Evaluate all alert rules.

        Args:
            context: Context data for evaluation

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self._rules.values():
            # Check if rule can trigger (cooldown)
            if not rule.can_trigger():
                continue

            # Evaluate condition
            should_fire, title, description = rule.evaluate(context)

            if should_fire:
                alert = await self.trigger_alert(
                    rule_name=rule.name,
                    title=title,
                    description=description,
                )
                triggered_alerts.append(alert)

            elif rule.auto_resolve and rule.active_alert_id:
                # Auto-resolve if condition cleared
                await self.resolve_alert(
                    rule.active_alert_id,
                    "Auto-resolved: condition cleared",
                )

        return triggered_alerts

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User/system acknowledging

        Returns:
            True if acknowledged successfully
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        if alert.state != AlertState.ACTIVE:
            return False

        alert.acknowledge(acknowledged_by)
        self._stats["active_alerts"] -= 1
        self._stats["acknowledged_alerts"] += 1

        logger.info(
            "alert_acknowledged",
            alert_id=alert_id,
            acknowledged_by=acknowledged_by,
        )

        return True

    async def resolve_alert(
        self,
        alert_id: str,
        resolution_note: str | None = None,
    ) -> bool:
        """
        Resolve alert.

        Args:
            alert_id: Alert identifier
            resolution_note: Resolution note

        Returns:
            True if resolved successfully
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        if alert.state == AlertState.RESOLVED:
            return False

        # Update stats
        if alert.state == AlertState.ACTIVE:
            self._stats["active_alerts"] -= 1
        elif alert.state == AlertState.ACKNOWLEDGED:
            self._stats["acknowledged_alerts"] -= 1

        alert.resolve(resolution_note)
        self._stats["resolved_alerts"] += 1

        # Clear rule association
        rule = self._rules.get(alert.rule_name)
        if rule and rule.active_alert_id == alert_id:
            rule.active_alert_id = None

        logger.info(
            "alert_resolved",
            alert_id=alert_id,
            resolution_note=resolution_note,
        )

        return True

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        label_filters: dict[str, str] | None = None,
    ) -> list[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            severity: Filter by severity
            label_filters: Filter by labels

        Returns:
            List of matching alerts
        """
        alerts = [
            alert
            for alert in self._alerts.values()
            if alert.state in (AlertState.ACTIVE, AlertState.ACKNOWLEDGED)
        ]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if label_filters:
            alerts = [
                a
                for a in alerts
                if all(a.labels.get(k) == v for k, v in label_filters.items())
            ]

        return alerts

    def get_alert_history(
        self,
        hours: int = 24,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """
        Get alert history.

        Args:
            hours: Number of hours to look back
            severity: Filter by severity

        Returns:
            List of historical alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self._alert_history if a.created_at >= cutoff]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_statistics(self) -> dict[str, Any]:
        """
        Get alerting statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "registered_rules": len(self._rules),
            "total_stored_alerts": len(self._alerts),
        }

    def register_notification_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Register custom notification handler.

        Args:
            channel: Notification channel
            handler: Handler function
        """
        self._notification_handler.register_handler(channel, handler)

    async def start(self) -> None:
        """Start background alert evaluation."""
        if self._evaluation_task is None:
            self._evaluation_task = asyncio.create_task(self._evaluation_loop())
            logger.info("alerting_service_started")

    async def stop(self) -> None:
        """Stop background alert evaluation."""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
            self._evaluation_task = None
            logger.info("alerting_service_stopped")

    async def _evaluation_loop(self) -> None:
        """Background loop for rule evaluation."""
        try:
            while True:
                await asyncio.sleep(self._evaluation_interval)

                # This would integrate with metrics collector
                # For now, just log that evaluation occurred
                logger.debug("alert_rules_evaluated", rule_count=len(self._rules))

        except asyncio.CancelledError:
            logger.info("alert_evaluation_loop_cancelled")
            raise


# Global alerting service instance
_global_alerting_service: AlertingService | None = None


def get_alerting_service() -> AlertingService:
    """Get global alerting service instance."""
    global _global_alerting_service
    if _global_alerting_service is None:
        _global_alerting_service = AlertingService()
    return _global_alerting_service
