"""
Resource management service for agent runtime.

This module provides advanced resource allocation, monitoring, alerting,
and dynamic scaling for agent execution with intelligent resource limits
and cleanup automation.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from ..models.agent_config import AgentConfig, AgentPhilosophy
from .performance_optimizer import get_performance_optimizer

logger = structlog.get_logger()


class ResourceType(str, Enum):
    """Types of resources managed by the system."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    FILE_DESCRIPTORS = "file_descriptors"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ResourceAlert:
    """Resource usage alert."""

    def __init__(
        self,
        alert_id: str,
        resource_type: ResourceType,
        severity: AlertSeverity,
        message: str,
        agent_id: str | None = None,
        threshold: float | None = None,
        current_value: float | None = None,
    ) -> None:
        """
        Initialize resource alert.

        Args:
            alert_id: Unique alert identifier
            resource_type: Type of resource
            severity: Alert severity level
            message: Alert description
            agent_id: Agent causing the alert (if applicable)
            threshold: Resource threshold that was exceeded
            current_value: Current resource usage value
        """
        self.alert_id = alert_id
        self.resource_type = resource_type
        self.severity = severity
        self.message = message
        self.agent_id = agent_id
        self.threshold = threshold
        self.current_value = current_value
        self.timestamp = datetime.now(UTC)
        self.acknowledged = False


@dataclass
class ResourceLimits:
    """Resource limits configuration."""

    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_disk_io_mbps: float = 100.0
    max_network_io_mbps: float = 100.0
    max_file_descriptors: int = 1000
    alert_threshold_percent: float = 75.0


@dataclass
class ResourceUsage:
    """Current resource usage metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_io_mbps: float = 0.0
    network_io_mbps: float = 0.0
    file_descriptors: int = 0
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


class ResourceMonitor:
    """Monitors resource usage and generates alerts."""

    def __init__(
        self,
        limits: ResourceLimits,
        check_interval_seconds: int = 5,
    ) -> None:
        """
        Initialize resource monitor.

        Args:
            limits: Resource limits configuration
            check_interval_seconds: Monitoring interval
        """
        self._limits = limits
        self._check_interval = check_interval_seconds
        self._alerts: list[ResourceAlert] = []
        self._alert_handlers: list[Callable[[ResourceAlert], None]] = []
        self._monitoring_task: asyncio.Task[None] | None = None
        self._usage_history: dict[str, list[ResourceUsage]] = defaultdict(list)
        self._history_max_size = 100

    async def start(self, usage_provider: Callable[[], ResourceUsage]) -> None:
        """
        Start resource monitoring.

        Args:
            usage_provider: Async function to get current resource usage
        """
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(
                self._monitor_loop(usage_provider)
            )
            logger.info(
                "resource_monitor_started", interval_seconds=self._check_interval
            )

    async def stop(self) -> None:
        """Stop resource monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("resource_monitor_stopped")

    def register_alert_handler(
        self,
        handler: Callable[[ResourceAlert], None],
    ) -> None:
        """
        Register alert handler function.

        Args:
            handler: Function to call when alert is triggered
        """
        self._alert_handlers.append(handler)

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[ResourceAlert]:
        """
        Get active alerts.

        Args:
            severity: Filter by severity level

        Returns:
            List of active alerts
        """
        alerts = [a for a in self._alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if alert was acknowledged, False if not found
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("alert_acknowledged", alert_id=alert_id)
                return True
        return False

    def get_usage_history(
        self,
        resource_type: str,
        duration_minutes: int = 60,
    ) -> list[ResourceUsage]:
        """
        Get resource usage history.

        Args:
            resource_type: Type of resource (or "system" for overall)
            duration_minutes: History duration to return

        Returns:
            List of resource usage snapshots
        """
        history = self._usage_history.get(resource_type, [])
        cutoff_time = datetime.now(UTC) - timedelta(minutes=duration_minutes)

        return [
            usage
            for usage in history
            if usage.timestamp and usage.timestamp >= cutoff_time
        ]

    async def _monitor_loop(
        self,
        usage_provider: Callable[[], ResourceUsage],
    ) -> None:
        """
        Background monitoring loop.

        Args:
            usage_provider: Function to get current usage
        """
        try:
            while True:
                # Get current resource usage
                usage = await usage_provider()

                # Record in history
                self._record_usage("system", usage)

                # Check thresholds and generate alerts
                self._check_thresholds(usage)

                # Cleanup old history
                self._cleanup_history()

                # Wait for next check
                await asyncio.sleep(self._check_interval)

        except asyncio.CancelledError:
            logger.info("resource_monitor_loop_cancelled")
            raise
        except Exception as e:
            logger.error("resource_monitor_loop_failed", error=str(e))
            raise

    def _record_usage(self, resource_key: str, usage: ResourceUsage) -> None:
        """
        Record usage in history.

        Args:
            resource_key: Resource identifier
            usage: Resource usage snapshot
        """
        history = self._usage_history[resource_key]
        history.append(usage)

        # Keep only recent history
        if len(history) > self._history_max_size:
            history.pop(0)

    def _check_thresholds(self, usage: ResourceUsage) -> None:
        """
        Check resource usage against thresholds.

        Args:
            usage: Current resource usage
        """
        # CPU check
        if usage.cpu_percent >= self._limits.max_cpu_percent:
            self._trigger_alert(
                ResourceType.CPU,
                AlertSeverity.CRITICAL,
                f"CPU usage at {usage.cpu_percent:.1f}% exceeds limit of {self._limits.max_cpu_percent}%",
                threshold=self._limits.max_cpu_percent,
                current_value=usage.cpu_percent,
            )
        elif usage.cpu_percent >= self._limits.alert_threshold_percent:
            self._trigger_alert(
                ResourceType.CPU,
                AlertSeverity.WARNING,
                f"CPU usage at {usage.cpu_percent:.1f}% approaching limit",
                threshold=self._limits.alert_threshold_percent,
                current_value=usage.cpu_percent,
            )

        # Memory check
        if usage.memory_percent >= self._limits.max_memory_percent:
            self._trigger_alert(
                ResourceType.MEMORY,
                AlertSeverity.CRITICAL,
                f"Memory usage at {usage.memory_percent:.1f}% exceeds limit of {self._limits.max_memory_percent}%",
                threshold=self._limits.max_memory_percent,
                current_value=usage.memory_percent,
            )
        elif usage.memory_percent >= self._limits.alert_threshold_percent:
            self._trigger_alert(
                ResourceType.MEMORY,
                AlertSeverity.WARNING,
                f"Memory usage at {usage.memory_percent:.1f}% approaching limit",
                threshold=self._limits.alert_threshold_percent,
                current_value=usage.memory_percent,
            )

        # Disk I/O check
        if usage.disk_io_mbps >= self._limits.max_disk_io_mbps:
            self._trigger_alert(
                ResourceType.DISK_IO,
                AlertSeverity.WARNING,
                f"Disk I/O at {usage.disk_io_mbps:.1f} MB/s exceeds limit",
                threshold=self._limits.max_disk_io_mbps,
                current_value=usage.disk_io_mbps,
            )

        # Network I/O check
        if usage.network_io_mbps >= self._limits.max_network_io_mbps:
            self._trigger_alert(
                ResourceType.NETWORK_IO,
                AlertSeverity.WARNING,
                f"Network I/O at {usage.network_io_mbps:.1f} MB/s exceeds limit",
                threshold=self._limits.max_network_io_mbps,
                current_value=usage.network_io_mbps,
            )

        # File descriptors check
        if usage.file_descriptors >= self._limits.max_file_descriptors:
            self._trigger_alert(
                ResourceType.FILE_DESCRIPTORS,
                AlertSeverity.CRITICAL,
                f"File descriptors at {usage.file_descriptors} exceeds limit of {self._limits.max_file_descriptors}",
                threshold=float(self._limits.max_file_descriptors),
                current_value=float(usage.file_descriptors),
            )

    def _trigger_alert(
        self,
        resource_type: ResourceType,
        severity: AlertSeverity,
        message: str,
        agent_id: str | None = None,
        threshold: float | None = None,
        current_value: float | None = None,
    ) -> None:
        """
        Trigger a new alert.

        Args:
            resource_type: Type of resource
            severity: Alert severity
            message: Alert message
            agent_id: Agent ID if applicable
            threshold: Threshold value
            current_value: Current value
        """
        alert_id = f"{resource_type.value}_{datetime.now(UTC).timestamp()}"

        alert = ResourceAlert(
            alert_id=alert_id,
            resource_type=resource_type,
            severity=severity,
            message=message,
            agent_id=agent_id,
            threshold=threshold,
            current_value=current_value,
        )

        self._alerts.append(alert)

        # Call registered handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(
                    "alert_handler_failed",
                    alert_id=alert_id,
                    error=str(e),
                )

        logger.warning(
            "resource_alert_triggered",
            alert_id=alert_id,
            resource_type=resource_type.value,
            severity=severity.value,
            message=message,
        )

    def _cleanup_history(self) -> None:
        """Clean up old history entries."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=1)

        for resource_key in list(self._usage_history.keys()):
            history = self._usage_history[resource_key]
            self._usage_history[resource_key] = [
                usage
                for usage in history
                if usage.timestamp and usage.timestamp >= cutoff_time
            ]


class DynamicScaler:
    """Manages dynamic resource scaling based on usage patterns."""

    def __init__(
        self,
        scale_up_threshold: float = 75.0,
        scale_down_threshold: float = 25.0,
        cooldown_seconds: int = 60,
    ) -> None:
        """
        Initialize dynamic scaler.

        Args:
            scale_up_threshold: Usage percent to trigger scale up
            scale_down_threshold: Usage percent to trigger scale down
            cooldown_seconds: Cooldown period between scaling actions
        """
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._cooldown_seconds = cooldown_seconds
        self._last_scale_action: dict[str, datetime] = {}
        self._current_scale: dict[str, int] = defaultdict(lambda: 1)

    def should_scale_up(
        self,
        philosophy: AgentPhilosophy,
        current_usage_percent: float,
    ) -> bool:
        """
        Check if scaling up is needed.

        Args:
            philosophy: Agent philosophy type
            current_usage_percent: Current resource usage percentage

        Returns:
            True if scale up is recommended
        """
        if current_usage_percent < self._scale_up_threshold:
            return False

        # Check cooldown
        last_action = self._last_scale_action.get(philosophy.value)
        if last_action:
            cooldown_elapsed = (datetime.now(UTC) - last_action).total_seconds()
            if cooldown_elapsed < self._cooldown_seconds:
                return False

        return True

    def should_scale_down(
        self,
        philosophy: AgentPhilosophy,
        current_usage_percent: float,
    ) -> bool:
        """
        Check if scaling down is possible.

        Args:
            philosophy: Agent philosophy type
            current_usage_percent: Current resource usage percentage

        Returns:
            True if scale down is recommended
        """
        if current_usage_percent > self._scale_down_threshold:
            return False

        # Don't scale below 1
        if self._current_scale[philosophy.value] <= 1:
            return False

        # Check cooldown
        last_action = self._last_scale_action.get(philosophy.value)
        if last_action:
            cooldown_elapsed = (datetime.now(UTC) - last_action).total_seconds()
            if cooldown_elapsed < self._cooldown_seconds:
                return False

        return True

    def record_scale_action(
        self,
        philosophy: AgentPhilosophy,
        new_scale: int,
    ) -> None:
        """
        Record a scaling action.

        Args:
            philosophy: Agent philosophy type
            new_scale: New scale value
        """
        self._last_scale_action[philosophy.value] = datetime.now(UTC)
        self._current_scale[philosophy.value] = new_scale

        logger.info(
            "scaling_action_recorded",
            philosophy=philosophy.value,
            new_scale=new_scale,
        )

    def get_current_scale(self, philosophy: AgentPhilosophy) -> int:
        """
        Get current scale for philosophy.

        Args:
            philosophy: Agent philosophy type

        Returns:
            Current scale value
        """
        return self._current_scale[philosophy.value]


class ResourceManager:
    """Main resource management service."""

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        enable_monitoring: bool = True,
        enable_dynamic_scaling: bool = True,
    ) -> None:
        """
        Initialize resource manager.

        Args:
            limits: Resource limits configuration
            enable_monitoring: Enable resource monitoring
            enable_dynamic_scaling: Enable dynamic scaling
        """
        self._limits = limits or ResourceLimits()
        self._enable_monitoring = enable_monitoring
        self._enable_dynamic_scaling = enable_dynamic_scaling

        # Initialize components
        self._monitor = ResourceMonitor(self._limits) if enable_monitoring else None
        self._scaler = DynamicScaler() if enable_dynamic_scaling else None
        self._optimizer = get_performance_optimizer()

        # Per-agent resource tracking
        self._agent_resources: dict[str, ResourceUsage] = {}

        logger.info(
            "resource_manager_initialized",
            monitoring_enabled=enable_monitoring,
            dynamic_scaling_enabled=enable_dynamic_scaling,
        )

    async def start(self) -> None:
        """Start resource management services."""
        if self._monitor:
            await self._monitor.start(self._get_system_usage)
        logger.info("resource_manager_started")

    async def stop(self) -> None:
        """Stop resource management services."""
        if self._monitor:
            await self._monitor.stop()
        logger.info("resource_manager_stopped")

    def allocate_resources(
        self,
        config: AgentConfig,
    ) -> dict[str, Any]:
        """
        Allocate resources for agent based on config and predictions.

        Args:
            config: Agent configuration

        Returns:
            Resource allocation recommendations
        """
        # Get predicted requirements from optimizer
        predicted = self._optimizer.predict_resource_requirements(config)

        # Adjust based on current system usage
        system_usage = self._get_system_usage_sync()

        # Calculate available resources
        available_cpu = 100.0 - system_usage.cpu_percent
        available_memory = 100.0 - system_usage.memory_percent

        # Determine allocation
        allocated_cpu = min(predicted["cpu_percent"], available_cpu * 0.8)
        allocated_memory = min(predicted["memory_mb"], available_memory * 0.8)

        allocation = {
            "cpu_percent": allocated_cpu,
            "memory_mb": allocated_memory,
            "execution_time_seconds": predicted["execution_time_seconds"],
            "available_cpu_percent": available_cpu,
            "available_memory_percent": available_memory,
        }

        logger.info(
            "resources_allocated",
            agent_id=config.agent_id,
            allocated_cpu=allocated_cpu,
            allocated_memory=allocated_memory,
        )

        return allocation

    def track_agent_usage(
        self,
        agent_id: str,
        usage: ResourceUsage,
    ) -> None:
        """
        Track resource usage for specific agent.

        Args:
            agent_id: Agent identifier
            usage: Resource usage metrics
        """
        self._agent_resources[agent_id] = usage

    def cleanup_agent_resources(self, agent_id: str) -> None:
        """
        Cleanup tracked resources for terminated agent.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agent_resources:
            del self._agent_resources[agent_id]
            logger.info("agent_resources_cleaned", agent_id=agent_id)

    def get_agent_usage(self, agent_id: str) -> ResourceUsage | None:
        """
        Get resource usage for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Resource usage or None if not found
        """
        return self._agent_resources.get(agent_id)

    def get_system_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive system resource metrics.

        Returns:
            System metrics dictionary
        """
        usage = self._get_system_usage_sync()

        metrics = {
            "current_usage": {
                "cpu_percent": usage.cpu_percent,
                "memory_percent": usage.memory_percent,
                "memory_mb": usage.memory_mb,
                "disk_io_mbps": usage.disk_io_mbps,
                "network_io_mbps": usage.network_io_mbps,
                "file_descriptors": usage.file_descriptors,
            },
            "limits": {
                "max_cpu_percent": self._limits.max_cpu_percent,
                "max_memory_percent": self._limits.max_memory_percent,
                "max_disk_io_mbps": self._limits.max_disk_io_mbps,
                "max_network_io_mbps": self._limits.max_network_io_mbps,
                "max_file_descriptors": self._limits.max_file_descriptors,
            },
            "active_agents": len(self._agent_resources),
        }

        if self._monitor:
            metrics["active_alerts"] = len(self._monitor.get_active_alerts())

        return metrics

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[ResourceAlert]:
        """
        Get active resource alerts.

        Args:
            severity: Filter by severity level

        Returns:
            List of active alerts
        """
        if self._monitor:
            return self._monitor.get_active_alerts(severity)
        return []

    def register_alert_handler(
        self,
        handler: Callable[[ResourceAlert], None],
    ) -> None:
        """
        Register alert handler.

        Args:
            handler: Function to call when alert is triggered
        """
        if self._monitor:
            self._monitor.register_alert_handler(handler)

    async def _get_system_usage(self) -> ResourceUsage:
        """
        Get current system resource usage.

        Returns:
            System resource usage
        """
        return self._get_system_usage_sync()

    def _get_system_usage_sync(self) -> ResourceUsage:
        """
        Get current system resource usage (synchronous).

        Returns:
            System resource usage
        """
        import psutil

        # Get system-wide metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()

        # Calculate I/O rates (simplified - would need history for accurate rates)
        disk_io_mbps = 0.0
        if disk_io:
            disk_io_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)

        network_io_mbps = 0.0
        if net_io:
            network_io_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)

        # Get file descriptor count
        process = psutil.Process()
        fd_count = process.num_fds() if hasattr(process, "num_fds") else 0

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            disk_io_mbps=disk_io_mbps,
            network_io_mbps=network_io_mbps,
            file_descriptors=fd_count,
        )


# Global resource manager instance
_global_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ResourceManager()
    return _global_manager
