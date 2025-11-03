# Resource Management System

This guide covers the comprehensive resource management system implemented in AgentCore Runtime for intelligent resource allocation, monitoring, and dynamic scaling.

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Resource Types](#resource-types)
4. [Resource Monitoring](#resource-monitoring)
5. [Alert System](#alert-system)
6. [Dynamic Scaling](#dynamic-scaling)
7. [Resource Allocation](#resource-allocation)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [Testing](#testing)

## Overview

AgentCore Runtime provides advanced resource management with automated monitoring, intelligent allocation, and dynamic scaling:

- **Resource Monitoring**: Real-time tracking of CPU, memory, disk I/O, network I/O, and file descriptors
- **Alert System**: Threshold-based alerting with configurable severity levels
- **Dynamic Scaling**: Automatic scaling decisions based on resource utilization patterns
- **Resource Allocation**: Intelligent allocation with system-aware recommendations
- **History Tracking**: Historical resource usage data for pattern analysis
- **psutil Integration**: Direct system metrics collection via psutil

```
┌─────────────────────────────────────────────────────┐
│  ResourceManager (Main Service)                     │
│  ├─ Resource allocation for agents                 │
│  ├─ Per-agent resource tracking                    │
│  ├─ System metrics aggregation                     │
│  └─ Alert handler registration                     │
└─────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ ResourceMonitor │  │ DynamicScaler   │  │ Performance      │
│                 │  │                 │  │ Optimizer        │
│ • Threshold     │  │ • Scale up/down │  │                  │
│   checking      │  │   decisions     │  │ • Requirement    │
│ • Alert         │  │ • Cooldown      │  │   prediction     │
│   triggering    │  │   management    │  │ • Optimization   │
│ • Usage history │  │ • Scale state   │  │   recommendations│
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

## Components

### ResourceManager

Main service coordinating all resource management operations.

```python
from agentcore.agent_runtime.services.resource_manager import (
    ResourceManager,
    ResourceLimits,
    get_resource_manager
)

# Create manager with custom limits
manager = ResourceManager(
    limits=ResourceLimits(
        max_cpu_percent=80.0,
        max_memory_percent=80.0,
        max_disk_io_mbps=100.0,
        max_network_io_mbps=100.0,
        max_file_descriptors=1000,
        alert_threshold_percent=75.0
    ),
    enable_monitoring=True,
    enable_dynamic_scaling=True
)

# Or use global instance
manager = get_resource_manager()
```

### ResourceMonitor

Monitors resource usage and generates alerts when thresholds are exceeded.

```python
from agentcore.agent_runtime.services.resource_manager import (
    ResourceMonitor,
    ResourceLimits,
    ResourceUsage
)

# Create monitor
limits = ResourceLimits(max_cpu_percent=80.0)
monitor = ResourceMonitor(
    limits=limits,
    check_interval_seconds=5  # Check every 5 seconds
)

# Provide usage data
async def get_usage() -> ResourceUsage:
    return ResourceUsage(
        cpu_percent=50.0,
        memory_percent=60.0,
        memory_mb=1024.0
    )

# Start monitoring
await monitor.start(get_usage)

# Stop monitoring
await monitor.stop()
```

### DynamicScaler

Manages dynamic resource scaling decisions based on utilization patterns.

```python
from agentcore.agent_runtime.services.resource_manager import DynamicScaler
from agentcore.agent_runtime.models.agent_config import AgentPhilosophy

# Create scaler
scaler = DynamicScaler(
    scale_up_threshold=75.0,    # Scale up at 75% utilization
    scale_down_threshold=25.0,  # Scale down at 25% utilization
    cooldown_seconds=60        # 60s cooldown between actions
)

# Check scaling decisions
if scaler.should_scale_up(AgentPhilosophy.REACT, current_usage=80.0):
    # Perform scale up
    new_scale = scaler.get_current_scale(AgentPhilosophy.REACT) + 1
    scaler.record_scale_action(AgentPhilosophy.REACT, new_scale)

if scaler.should_scale_down(AgentPhilosophy.REACT, current_usage=20.0):
    # Perform scale down
    new_scale = scaler.get_current_scale(AgentPhilosophy.REACT) - 1
    scaler.record_scale_action(AgentPhilosophy.REACT, new_scale)
```

## Resource Types

AgentCore monitors five resource types:

### CPU
Processor utilization as percentage (0-100%)

```python
from agentcore.agent_runtime.services.resource_manager import ResourceType

resource_type = ResourceType.CPU
```

### MEMORY
Memory utilization as percentage and absolute megabytes

```python
resource_type = ResourceType.MEMORY
```

### DISK_IO
Disk I/O throughput in megabytes per second

```python
resource_type = ResourceType.DISK_IO
```

### NETWORK_IO
Network I/O throughput in megabytes per second

```python
resource_type = ResourceType.NETWORK_IO
```

### FILE_DESCRIPTORS
Number of open file descriptors

```python
resource_type = ResourceType.FILE_DESCRIPTORS
```

## Resource Monitoring

### Starting Monitoring

```python
manager = ResourceManager(enable_monitoring=True)

# Start background monitoring
await manager.start()

# Monitoring runs every check_interval_seconds (default: 5s)
# Automatically checks thresholds and triggers alerts
```

### Usage Tracking

```python
from agentcore.agent_runtime.services.resource_manager import ResourceUsage

# Create usage snapshot
usage = ResourceUsage(
    cpu_percent=50.0,
    memory_percent=60.0,
    memory_mb=1024.0,
    disk_io_mbps=10.0,
    network_io_mbps=5.0,
    file_descriptors=100
)

# Track per-agent usage
manager.track_agent_usage("agent-123", usage)

# Retrieve agent usage
agent_usage = manager.get_agent_usage("agent-123")

# Cleanup when agent terminates
manager.cleanup_agent_resources("agent-123")
```

### Usage History

```python
# Get historical usage (last 60 minutes)
history = monitor.get_usage_history(
    resource_type="system",  # or agent_id
    duration_minutes=60
)

for usage in history:
    print(f"{usage.timestamp}: CPU {usage.cpu_percent}%")
```

### System Metrics

```python
# Get comprehensive system metrics
metrics = manager.get_system_metrics()

print(f"CPU: {metrics['current_usage']['cpu_percent']}%")
print(f"Memory: {metrics['current_usage']['memory_percent']}%")
print(f"Active agents: {metrics['active_agents']}")
print(f"Active alerts: {metrics.get('active_alerts', 0)}")
```

## Alert System

### Alert Severity Levels

Four severity levels for resource alerts:

#### INFO
Informational alerts (lowest priority)

```python
from agentcore.agent_runtime.services.resource_manager import AlertSeverity

severity = AlertSeverity.INFO
```

#### WARNING
Warning alerts for threshold approaching

```python
# Triggered at alert_threshold_percent (default: 75%)
severity = AlertSeverity.WARNING
```

#### CRITICAL
Critical alerts for threshold exceeded

```python
# Triggered at max_*_percent limits
severity = AlertSeverity.CRITICAL
```

#### EMERGENCY
Emergency alerts for severe resource issues

```python
severity = AlertSeverity.EMERGENCY
```

### Alert Handling

```python
from agentcore.agent_runtime.services.resource_manager import ResourceAlert

# Define alert handler
def handle_alert(alert: ResourceAlert):
    print(f"ALERT [{alert.severity.value}]: {alert.message}")
    print(f"Resource: {alert.resource_type.value}")
    print(f"Threshold: {alert.threshold}, Current: {alert.current_value}")

    if alert.severity == AlertSeverity.CRITICAL:
        # Take immediate action
        notify_ops_team(alert)

# Register handler
manager.register_alert_handler(handle_alert)

# Handlers are called automatically when alerts trigger
```

### Managing Alerts

```python
# Get all active alerts
alerts = manager.get_alerts()

# Filter by severity
critical_alerts = manager.get_alerts(severity=AlertSeverity.CRITICAL)
warning_alerts = manager.get_alerts(severity=AlertSeverity.WARNING)

# Acknowledge alert
for alert in alerts:
    if alert.severity == AlertSeverity.WARNING:
        monitor.acknowledge_alert(alert.alert_id)

# Acknowledged alerts no longer appear in active list
active_alerts = manager.get_alerts()  # Excludes acknowledged
```

## Dynamic Scaling

### Scaling Thresholds

```python
scaler = DynamicScaler(
    scale_up_threshold=75.0,    # Scale up when utilization > 75%
    scale_down_threshold=25.0,  # Scale down when utilization < 25%
    cooldown_seconds=60        # Wait 60s between scaling actions
)
```

### Scale Up Logic

```python
from agentcore.agent_runtime.models.agent_config import AgentPhilosophy

philosophy = AgentPhilosophy.REACT
current_usage = 80.0  # 80% CPU utilization

if scaler.should_scale_up(philosophy, current_usage):
    # Conditions met for scale up:
    # 1. Usage >= scale_up_threshold
    # 2. Cooldown period elapsed
    # 3. Not in cooldown from previous action

    current = scaler.get_current_scale(philosophy)  # e.g., 2
    new_scale = current + 1  # Scale to 3

    # Perform scaling operation
    provision_additional_resources(philosophy, new_scale)

    # Record action (starts cooldown)
    scaler.record_scale_action(philosophy, new_scale)
```

### Scale Down Logic

```python
philosophy = AgentPhilosophy.REACT
current_usage = 20.0  # 20% CPU utilization

if scaler.should_scale_down(philosophy, current_usage):
    # Conditions met for scale down:
    # 1. Usage <= scale_down_threshold
    # 2. Current scale > 1 (won't scale below 1)
    # 3. Cooldown period elapsed

    current = scaler.get_current_scale(philosophy)  # e.g., 3
    new_scale = current - 1  # Scale to 2

    # Perform scaling operation
    reduce_allocated_resources(philosophy, new_scale)

    # Record action (starts cooldown)
    scaler.record_scale_action(philosophy, new_scale)
```

### Cooldown Management

Cooldown prevents rapid scaling oscillations:

```python
# Cooldown is automatic
scaler = DynamicScaler(cooldown_seconds=60)

# First scale action
scaler.record_scale_action(philosophy, 3)

# Immediate second scale attempt - blocked by cooldown
assert not scaler.should_scale_up(philosophy, 80.0)  # False (in cooldown)

# After 60 seconds - allowed
time.sleep(60)
assert scaler.should_scale_up(philosophy, 80.0)  # True (cooldown elapsed)
```

## Resource Allocation

### Intelligent Allocation

ResourceManager provides intelligent allocation recommendations:

```python
from agentcore.agent_runtime.models.agent_config import (
    AgentConfig,
    AgentPhilosophy,
    ResourceLimits as ConfigResourceLimits,
    SecurityProfile
)

# Create agent config
config = AgentConfig(
    agent_id="my-agent",
    philosophy=AgentPhilosophy.REACT,
    resource_limits=ConfigResourceLimits(
        max_memory_mb=512,
        max_cpu_cores=1.0
    ),
    security_profile=SecurityProfile()
)

# Get allocation recommendation
allocation = manager.allocate_resources(config)

print(f"Allocated CPU: {allocation['cpu_percent']}%")
print(f"Allocated Memory: {allocation['memory_mb']} MB")
print(f"Estimated execution time: {allocation['execution_time_seconds']}s")
print(f"Available CPU: {allocation['available_cpu_percent']}%")
print(f"Available Memory: {allocation['available_memory_percent']}%")
```

### Allocation Algorithm

```python
# ResourceManager allocates based on:

# 1. Get predicted requirements from performance optimizer
predicted = optimizer.predict_resource_requirements(config)

# 2. Get current system usage
system_usage = get_system_usage()
available_cpu = 100.0 - system_usage.cpu_percent
available_memory = 100.0 - system_usage.memory_percent

# 3. Allocate up to 80% of available resources
allocated_cpu = min(predicted["cpu_percent"], available_cpu * 0.8)
allocated_memory = min(predicted["memory_mb"], available_memory * 0.8)

# 4. Return allocation recommendation
return {
    "cpu_percent": allocated_cpu,
    "memory_mb": allocated_memory,
    "execution_time_seconds": predicted["execution_time_seconds"],
    "available_cpu_percent": available_cpu,
    "available_memory_percent": available_memory
}
```

## Usage Examples

### Complete Monitoring Setup

```python
from agentcore.agent_runtime.services.resource_manager import (
    ResourceManager,
    ResourceLimits,
    ResourceAlert,
    AlertSeverity
)

# 1. Create manager with custom limits
manager = ResourceManager(
    limits=ResourceLimits(
        max_cpu_percent=80.0,
        max_memory_percent=80.0,
        alert_threshold_percent=70.0
    ),
    enable_monitoring=True,
    enable_dynamic_scaling=True
)

# 2. Register alert handler
def alert_handler(alert: ResourceAlert):
    if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
        # Send notification
        notify_ops_team(f"Resource alert: {alert.message}")

    if alert.resource_type == ResourceType.MEMORY:
        # Trigger memory cleanup
        trigger_garbage_collection()

manager.register_alert_handler(alert_handler)

# 3. Start monitoring
await manager.start()

# 4. Monitor runs in background, alerts automatically triggered
```

### Agent Lifecycle Integration

```python
# When agent starts
async def start_agent(config: AgentConfig):
    # Allocate resources
    allocation = manager.allocate_resources(config)

    # Create agent with allocated resources
    agent = create_agent(config, allocation)

    # Track initial usage
    manager.track_agent_usage(
        config.agent_id,
        ResourceUsage(cpu_percent=0.0, memory_percent=0.0)
    )

    return agent

# During agent execution
async def update_agent_metrics(agent_id: str):
    # Get current usage from agent
    usage = get_agent_resource_usage(agent_id)

    # Track usage
    manager.track_agent_usage(agent_id, usage)

# When agent stops
async def stop_agent(agent_id: str):
    # Cleanup tracked resources
    manager.cleanup_agent_resources(agent_id)
```

### Dynamic Scaling Integration

```python
from agentcore.agent_runtime.models.agent_config import AgentPhilosophy

# Periodic scaling check
async def check_scaling():
    metrics = manager.get_system_metrics()
    cpu_usage = metrics['current_usage']['cpu_percent']

    if manager._scaler:
        # Check scale up
        if manager._scaler.should_scale_up(AgentPhilosophy.REACT, cpu_usage):
            current = manager._scaler.get_current_scale(AgentPhilosophy.REACT)
            new_scale = current + 1

            # Provision resources
            await provision_react_agents(new_scale)

            # Record action
            manager._scaler.record_scale_action(AgentPhilosophy.REACT, new_scale)
            logger.info(f"Scaled REACT agents to {new_scale}")

        # Check scale down
        elif manager._scaler.should_scale_down(AgentPhilosophy.REACT, cpu_usage):
            current = manager._scaler.get_current_scale(AgentPhilosophy.REACT)
            new_scale = current - 1

            # Reduce resources
            await reduce_react_agents(new_scale)

            # Record action
            manager._scaler.record_scale_action(AgentPhilosophy.REACT, new_scale)
            logger.info(f"Scaled REACT agents to {new_scale}")
```

### Historical Analysis

```python
# Analyze resource trends
async def analyze_resource_trends(agent_id: str):
    # Get last hour of usage
    history = monitor.get_usage_history(agent_id, duration_minutes=60)

    if not history:
        return

    # Calculate averages
    avg_cpu = sum(u.cpu_percent for u in history) / len(history)
    avg_memory = sum(u.memory_percent for u in history) / len(history)

    # Identify peaks
    peak_cpu = max(u.cpu_percent for u in history)
    peak_memory = max(u.memory_percent for u in history)

    # Detect trends
    if avg_cpu > 70.0:
        logger.warning(f"Agent {agent_id} high average CPU: {avg_cpu}%")

    if peak_memory > 90.0:
        logger.warning(f"Agent {agent_id} memory spike: {peak_memory}%")

    return {
        "avg_cpu": avg_cpu,
        "avg_memory": avg_memory,
        "peak_cpu": peak_cpu,
        "peak_memory": peak_memory
    }
```

## Best Practices

### 1. Set Appropriate Thresholds

```python
# Production: Conservative limits
production_limits = ResourceLimits(
    max_cpu_percent=70.0,        # Leave 30% buffer
    max_memory_percent=75.0,     # Leave 25% buffer
    alert_threshold_percent=60.0  # Early warning at 60%
)

# Development: Relaxed limits
dev_limits = ResourceLimits(
    max_cpu_percent=90.0,
    max_memory_percent=90.0,
    alert_threshold_percent=80.0
)
```

### 2. Implement Alert Handlers

```python
def production_alert_handler(alert: ResourceAlert):
    """Production-ready alert handler."""

    # Always log
    logger.warning(
        "resource_alert",
        alert_id=alert.alert_id,
        resource_type=alert.resource_type.value,
        severity=alert.severity.value,
        message=alert.message
    )

    # Critical alerts - immediate action
    if alert.severity == AlertSeverity.CRITICAL:
        # Send to monitoring system
        send_to_datadog(alert)

        # Page on-call engineer
        if alert.resource_type == ResourceType.MEMORY:
            page_oncall("Memory critical", alert.message)

    # Emergency alerts - escalate
    if alert.severity == AlertSeverity.EMERGENCY:
        page_oncall("Resource emergency", alert.message)
        trigger_emergency_runbook(alert)
```

### 3. Monitor Continuously

```python
# Start monitoring during application startup
async def startup():
    manager = get_resource_manager()
    await manager.start()
    logger.info("Resource monitoring started")

# Stop during shutdown
async def shutdown():
    manager = get_resource_manager()
    await manager.stop()
    logger.info("Resource monitoring stopped")
```

### 4. Track Per-Agent Usage

```python
# Always track agent resource usage
async def agent_lifecycle():
    agent_id = "agent-123"

    try:
        # Start tracking
        manager.track_agent_usage(
            agent_id,
            ResourceUsage(cpu_percent=0.0, memory_percent=0.0)
        )

        # Update periodically
        while agent_running:
            usage = get_current_usage()
            manager.track_agent_usage(agent_id, usage)
            await asyncio.sleep(5)

    finally:
        # Always cleanup
        manager.cleanup_agent_resources(agent_id)
```

### 5. Use Cooldown for Stability

```python
# Appropriate cooldown prevents oscillation
scaler = DynamicScaler(
    scale_up_threshold=75.0,
    scale_down_threshold=25.0,
    cooldown_seconds=120  # 2 minutes - stable scaling
)

# Too short cooldown - rapid oscillation
bad_scaler = DynamicScaler(
    cooldown_seconds=5  # Too short - will oscillate
)
```

### 6. Configure Check Intervals

```python
# Production: Frequent checks
monitor = ResourceMonitor(
    limits=ResourceLimits(),
    check_interval_seconds=5  # Check every 5s
)

# Development: Less frequent
monitor = ResourceMonitor(
    limits=ResourceLimits(),
    check_interval_seconds=30  # Check every 30s
)
```

### 7. Acknowledge Alerts Appropriately

```python
async def process_alerts():
    """Process and acknowledge alerts."""
    alerts = manager.get_alerts()

    for alert in alerts:
        # Handle based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            # Immediate action required - don't acknowledge yet
            handle_critical_alert(alert)

        elif alert.severity == AlertSeverity.WARNING:
            # Warning handled - acknowledge
            logger.info(f"Handled warning alert: {alert.alert_id}")
            monitor.acknowledge_alert(alert.alert_id)
```

### 8. Monitor System Metrics Regularly

```python
# Regular metrics collection
async def collect_metrics():
    """Collect and export metrics."""
    while True:
        metrics = manager.get_system_metrics()

        # Export to monitoring system
        export_metric("system.cpu_percent", metrics['current_usage']['cpu_percent'])
        export_metric("system.memory_percent", metrics['current_usage']['memory_percent'])
        export_metric("system.active_agents", metrics['active_agents'])
        export_metric("system.active_alerts", metrics.get('active_alerts', 0))

        await asyncio.sleep(10)
```

## Testing

AgentCore includes comprehensive tests for resource management:

### Running Tests (26 scenarios)

```bash
# Run all resource manager tests
uv run pytest tests/agent_runtime/test_resource_manager.py -v
```

**Test Coverage**:
- ResourceUsage dataclass (2 tests)
- ResourceAlert creation (1 test)
- ResourceMonitor functionality (6 tests):
  - Start/stop monitoring
  - Alert handler registration
  - Threshold checking
  - Active alert management
  - Alert acknowledgment
  - Usage history tracking
- DynamicScaler logic (5 tests):
  - Scale up/down decisions
  - Cooldown enforcement
  - Scale state tracking
- ResourceManager integration (12 tests):
  - Service lifecycle
  - Resource allocation
  - Agent usage tracking
  - System metrics
  - Alert management
  - Monitoring disabled scenarios
  - Scaling disabled scenarios
- ResourceLimits configuration (2 tests)

### Test Results

```
✅ 26/26 tests passing
📊 91% coverage on resource_manager.py
```

### Example Test Patterns

```python
import pytest
from agentcore.agent_runtime.services.resource_manager import (
    ResourceManager,
    ResourceLimits,
    ResourceUsage,
    AlertSeverity
)

@pytest.mark.asyncio
async def test_resource_allocation():
    """Test resource allocation for agents."""
    manager = ResourceManager()

    config = AgentConfig(
        agent_id="test-agent",
        philosophy=AgentPhilosophy.REACT,
        resource_limits=ConfigResourceLimits(
            max_memory_mb=512,
            max_cpu_cores=1.0
        )
    )

    allocation = manager.allocate_resources(config)

    assert "cpu_percent" in allocation
    assert "memory_mb" in allocation
    assert allocation["cpu_percent"] > 0
```

## Additional Resources

- [Resource Manager Implementation](../../src/agentcore/agent_runtime/services/resource_manager.py)
- [Resource Manager Tests](../../tests/agent_runtime/test_resource_manager.py)
- [Performance Optimizer](../../src/agentcore/agent_runtime/services/performance_optimizer.py)
- [Agent Configuration](../../src/agentcore/agent_runtime/models/agent_config.py)

## Support

For resource management questions or issues:
- Review test suite for usage patterns
- Check system metrics for resource utilization
- Monitor alerts for threshold violations
- Analyze usage history for patterns
- Adjust limits based on workload characteristics
