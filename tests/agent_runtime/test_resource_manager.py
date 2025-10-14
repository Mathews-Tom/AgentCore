"""Tests for resource management service."""

import asyncio
from datetime import UTC, datetime

import pytest

from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.models.agent_config import (
    ResourceLimits as ConfigResourceLimits,
)
from agentcore.agent_runtime.models.agent_config import SecurityProfile
from agentcore.agent_runtime.services.resource_manager import (
    AlertSeverity,
    DynamicScaler,
    ResourceAlert,
    ResourceLimits,
    ResourceManager,
    ResourceMonitor,
    ResourceType,
    ResourceUsage,
)


class TestResourceUsage:
    """Test resource usage data class."""

    def test_resource_usage_creation(self) -> None:
        """Test resource usage is created with correct defaults."""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_mb=1024.0,
        )

        assert usage.cpu_percent == 50.0
        assert usage.memory_percent == 60.0
        assert usage.memory_mb == 1024.0
        assert usage.timestamp is not None

    def test_resource_usage_with_timestamp(self) -> None:
        """Test resource usage with custom timestamp."""
        timestamp = datetime.now(UTC)
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            timestamp=timestamp,
        )

        assert usage.timestamp == timestamp


class TestResourceAlert:
    """Test resource alert creation."""

    def test_alert_creation(self) -> None:
        """Test alert is created with correct attributes."""
        alert = ResourceAlert(
            alert_id="test-alert-1",
            resource_type=ResourceType.CPU,
            severity=AlertSeverity.WARNING,
            message="CPU usage high",
            agent_id="test-agent",
            threshold=75.0,
            current_value=80.0,
        )

        assert alert.alert_id == "test-alert-1"
        assert alert.resource_type == ResourceType.CPU
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "CPU usage high"
        assert alert.agent_id == "test-agent"
        assert alert.threshold == 75.0
        assert alert.current_value == 80.0
        assert not alert.acknowledged
        assert alert.timestamp is not None


class TestResourceMonitor:
    """Test resource monitoring."""

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self) -> None:
        """Test monitor can be started and stopped."""
        limits = ResourceLimits(max_cpu_percent=80.0)
        monitor = ResourceMonitor(limits)

        async def mock_usage_provider() -> ResourceUsage:
            return ResourceUsage(cpu_percent=50.0, memory_percent=50.0)

        await monitor.start(mock_usage_provider)
        assert monitor._monitoring_task is not None

        await monitor.stop()
        assert monitor._monitoring_task is None

    @pytest.mark.asyncio
    async def test_monitor_registers_handler(self) -> None:
        """Test alert handler registration."""
        limits = ResourceLimits()
        monitor = ResourceMonitor(limits)

        alerts_received = []

        def alert_handler(alert: ResourceAlert) -> None:
            alerts_received.append(alert)

        monitor.register_alert_handler(alert_handler)
        assert len(monitor._alert_handlers) == 1

    @pytest.mark.asyncio
    async def test_monitor_threshold_checks(self) -> None:
        """Test monitor checks resource thresholds."""
        limits = ResourceLimits(max_cpu_percent=80.0, max_memory_percent=80.0)
        monitor = ResourceMonitor(limits, check_interval_seconds=1)

        alerts_triggered = []

        def alert_handler(alert: ResourceAlert) -> None:
            alerts_triggered.append(alert)

        monitor.register_alert_handler(alert_handler)

        async def high_usage_provider() -> ResourceUsage:
            return ResourceUsage(cpu_percent=85.0, memory_percent=85.0)

        # Start monitoring
        await monitor.start(high_usage_provider)

        # Wait for at least one check
        await asyncio.sleep(1.5)

        await monitor.stop()

        # Should have triggered alerts for high CPU and memory
        assert len(alerts_triggered) >= 2

    def test_monitor_get_active_alerts(self) -> None:
        """Test getting active alerts."""
        limits = ResourceLimits()
        monitor = ResourceMonitor(limits)

        # Manually trigger an alert
        monitor._trigger_alert(
            ResourceType.CPU,
            AlertSeverity.WARNING,
            "Test alert",
            threshold=75.0,
            current_value=80.0,
        )

        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 1

        # Filter by severity
        warning_alerts = monitor.get_active_alerts(severity=AlertSeverity.WARNING)
        assert len(warning_alerts) == 1

        critical_alerts = monitor.get_active_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 0

    def test_monitor_acknowledge_alert(self) -> None:
        """Test acknowledging alerts."""
        limits = ResourceLimits()
        monitor = ResourceMonitor(limits)

        # Trigger an alert
        monitor._trigger_alert(
            ResourceType.CPU,
            AlertSeverity.WARNING,
            "Test alert",
        )

        alerts = monitor.get_active_alerts()
        assert len(alerts) == 1

        alert_id = alerts[0].alert_id

        # Acknowledge the alert
        result = monitor.acknowledge_alert(alert_id)
        assert result is True

        # Should no longer be in active alerts
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 0

    def test_monitor_usage_history(self) -> None:
        """Test usage history tracking."""
        limits = ResourceLimits()
        monitor = ResourceMonitor(limits)

        # Record some usage
        for i in range(5):
            usage = ResourceUsage(cpu_percent=float(i * 10))
            monitor._record_usage("test_agent", usage)

        history = monitor.get_usage_history("test_agent", duration_minutes=60)
        assert len(history) == 5


class TestDynamicScaler:
    """Test dynamic scaling logic."""

    def test_scaler_creation(self) -> None:
        """Test scaler is created with correct thresholds."""
        scaler = DynamicScaler(
            scale_up_threshold=75.0,
            scale_down_threshold=25.0,
            cooldown_seconds=60,
        )

        assert scaler._scale_up_threshold == 75.0
        assert scaler._scale_down_threshold == 25.0
        assert scaler._cooldown_seconds == 60

    def test_scaler_should_scale_up(self) -> None:
        """Test scale up decision logic."""
        scaler = DynamicScaler(scale_up_threshold=75.0, cooldown_seconds=1)

        # Below threshold - no scale up
        assert not scaler.should_scale_up(AgentPhilosophy.REACT, 50.0)

        # Above threshold - should scale up
        assert scaler.should_scale_up(AgentPhilosophy.REACT, 80.0)

        # Record scale action
        scaler.record_scale_action(AgentPhilosophy.REACT, 2)

        # During cooldown - no scale up
        assert not scaler.should_scale_up(AgentPhilosophy.REACT, 80.0)

    def test_scaler_should_scale_down(self) -> None:
        """Test scale down decision logic."""
        scaler = DynamicScaler(scale_down_threshold=25.0, cooldown_seconds=1)

        # Set initial scale > 1
        scaler._current_scale[AgentPhilosophy.REACT.value] = 3

        # Above threshold - no scale down
        assert not scaler.should_scale_down(AgentPhilosophy.REACT, 50.0)

        # Below threshold - should scale down
        assert scaler.should_scale_down(AgentPhilosophy.REACT, 20.0)

        # At scale = 1, should not scale down
        scaler._current_scale[AgentPhilosophy.REACT.value] = 1
        assert not scaler.should_scale_down(AgentPhilosophy.REACT, 20.0)

    def test_scaler_record_action(self) -> None:
        """Test recording scaling actions."""
        scaler = DynamicScaler()

        scaler.record_scale_action(AgentPhilosophy.REACT, 2)

        assert scaler.get_current_scale(AgentPhilosophy.REACT) == 2
        assert AgentPhilosophy.REACT.value in scaler._last_scale_action

    def test_scaler_get_current_scale(self) -> None:
        """Test getting current scale value."""
        scaler = DynamicScaler()

        # Default scale is 1
        assert scaler.get_current_scale(AgentPhilosophy.REACT) == 1

        # After recording action
        scaler.record_scale_action(AgentPhilosophy.REACT, 3)
        assert scaler.get_current_scale(AgentPhilosophy.REACT) == 3


class TestResourceManager:
    """Test main resource manager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self) -> None:
        """Test manager initializes correctly."""
        manager = ResourceManager(
            limits=ResourceLimits(max_cpu_percent=80.0),
            enable_monitoring=True,
            enable_dynamic_scaling=True,
        )

        assert manager._limits.max_cpu_percent == 80.0
        assert manager._monitor is not None
        assert manager._scaler is not None

    @pytest.mark.asyncio
    async def test_manager_start_stop(self) -> None:
        """Test manager starts and stops services."""
        manager = ResourceManager(enable_monitoring=True)

        await manager.start()
        assert manager._monitor is not None
        assert manager._monitor._monitoring_task is not None

        await manager.stop()
        assert manager._monitor._monitoring_task is None

    def test_manager_allocate_resources(self) -> None:
        """Test resource allocation for agents."""
        manager = ResourceManager()

        config = AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            resource_limits=ConfigResourceLimits(
                max_memory_mb=512,
                max_cpu_cores=1.0,
            ),
            security_profile=SecurityProfile(),
        )

        allocation = manager.allocate_resources(config)

        assert "cpu_percent" in allocation
        assert "memory_mb" in allocation
        assert "execution_time_seconds" in allocation
        assert "available_cpu_percent" in allocation
        assert "available_memory_percent" in allocation

    def test_manager_track_agent_usage(self) -> None:
        """Test tracking agent resource usage."""
        manager = ResourceManager()

        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_mb=512.0,
        )

        manager.track_agent_usage("test-agent", usage)

        tracked_usage = manager.get_agent_usage("test-agent")
        assert tracked_usage is not None
        assert tracked_usage.cpu_percent == 50.0

    def test_manager_cleanup_agent_resources(self) -> None:
        """Test cleaning up agent resources."""
        manager = ResourceManager()

        usage = ResourceUsage(cpu_percent=50.0)
        manager.track_agent_usage("test-agent", usage)

        assert manager.get_agent_usage("test-agent") is not None

        manager.cleanup_agent_resources("test-agent")

        assert manager.get_agent_usage("test-agent") is None

    def test_manager_get_system_metrics(self) -> None:
        """Test getting system resource metrics."""
        manager = ResourceManager()

        metrics = manager.get_system_metrics()

        assert "current_usage" in metrics
        assert "limits" in metrics
        assert "active_agents" in metrics

        assert "cpu_percent" in metrics["current_usage"]
        assert "memory_percent" in metrics["current_usage"]

    def test_manager_get_alerts(self) -> None:
        """Test getting resource alerts."""
        manager = ResourceManager(enable_monitoring=True)

        # Initially no alerts
        alerts = manager.get_alerts()
        assert len(alerts) == 0

        # Trigger an alert manually
        if manager._monitor:
            manager._monitor._trigger_alert(
                ResourceType.CPU,
                AlertSeverity.WARNING,
                "Test alert",
            )

        alerts = manager.get_alerts()
        assert len(alerts) == 1

        # Filter by severity
        warning_alerts = manager.get_alerts(severity=AlertSeverity.WARNING)
        assert len(warning_alerts) == 1

    def test_manager_register_alert_handler(self) -> None:
        """Test registering alert handlers."""
        manager = ResourceManager(enable_monitoring=True)

        alerts_received = []

        def handler(alert: ResourceAlert) -> None:
            alerts_received.append(alert)

        manager.register_alert_handler(handler)

        # Trigger an alert
        if manager._monitor:
            manager._monitor._trigger_alert(
                ResourceType.MEMORY,
                AlertSeverity.CRITICAL,
                "Test memory alert",
            )

        # Handler should have been called
        assert len(alerts_received) == 1
        assert alerts_received[0].resource_type == ResourceType.MEMORY

    def test_manager_without_monitoring(self) -> None:
        """Test manager works without monitoring enabled."""
        manager = ResourceManager(enable_monitoring=False)

        assert manager._monitor is None

        # Should still work for basic operations
        config = AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            resource_limits=ConfigResourceLimits(),
            security_profile=SecurityProfile(),
        )

        allocation = manager.allocate_resources(config)
        assert "cpu_percent" in allocation

    def test_manager_without_scaling(self) -> None:
        """Test manager works without dynamic scaling."""
        manager = ResourceManager(enable_dynamic_scaling=False)

        assert manager._scaler is None

        # Should still allocate resources
        config = AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            resource_limits=ConfigResourceLimits(),
            security_profile=SecurityProfile(),
        )

        allocation = manager.allocate_resources(config)
        assert "cpu_percent" in allocation


class TestResourceLimits:
    """Test resource limits configuration."""

    def test_limits_default_values(self) -> None:
        """Test default resource limits."""
        limits = ResourceLimits()

        assert limits.max_cpu_percent == 80.0
        assert limits.max_memory_percent == 80.0
        assert limits.max_disk_io_mbps == 100.0
        assert limits.max_network_io_mbps == 100.0
        assert limits.max_file_descriptors == 1000
        assert limits.alert_threshold_percent == 75.0

    def test_limits_custom_values(self) -> None:
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_cpu_percent=90.0,
            max_memory_percent=85.0,
            alert_threshold_percent=80.0,
        )

        assert limits.max_cpu_percent == 90.0
        assert limits.max_memory_percent == 85.0
        assert limits.alert_threshold_percent == 80.0
