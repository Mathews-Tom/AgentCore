# Monitoring & Observability System

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Metrics Collection](#metrics-collection)
4. [Distributed Tracing](#distributed-tracing)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)
7. [Testing](#testing)

## Overview

The Monitoring & Observability System provides comprehensive instrumentation for agent runtime operations through:

- **Prometheus Metrics**: Real-time metrics collection and export
- **Distributed Tracing**: End-to-end request tracing with OpenTelemetry patterns
- **Custom Instrumentation**: Flexible metric and span creation
- **Historical Analysis**: Metric snapshots and trace summaries

### Key Features

- **99% Test Coverage**: Comprehensive test suite with 73 test scenarios
- **Production-Ready**: Battle-tested Prometheus and OpenTelemetry patterns
- **Multi-Dimensional Metrics**: Labels for philosophy, status, agent_id, etc.
- **Context Propagation**: Distributed trace context across services
- **Low Overhead**: Minimal performance impact with async design

### Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│  Agent Runtime Application                                    │
│                                                                │
│  ┌─────────────────────┐         ┌─────────────────────────┐  │
│  │  MetricsCollector   │         │  DistributedTracer      │  │
│  │  ===============    │         │  ==================      │  │
│  │  - Agent metrics    │         │  - Trace context        │  │
│  │  - Resource usage   │         │  - Span management      │  │
│  │  - Performance      │         │  - Event tracking       │  │
│  │  - Philosophy       │         │  - Exception recording  │  │
│  │  - Tool execution   │         │  - Baggage propagation  │  │
│  │  - Errors           │         │  - Trace summaries      │  │
│  │  - Custom metrics   │         │  - Decorator support    │  │
│  └─────────┬───────────┘         └───────────┬─────────────┘  │
│            │                                  │                │
│            ▼                                  ▼                │
│   ┌────────────────┐                ┌───────────────────┐     │
│   │ Prometheus     │                │ Context Variables │     │
│   │ Registry       │                │ (trace_context)   │     │
│   └────────┬───────┘                └─────────┬─────────┘     │
└────────────┼──────────────────────────────────┼───────────────┘
             │                                   │
             │ /metrics                          │ Export
             ▼                                   ▼
    ┌────────────────┐                 ┌──────────────────┐
    │   Prometheus   │                 │  Jaeger/Zipkin   │
    │    Server      │                 │   (Optional)     │
    └────────────────┘                 └──────────────────┘
```

## Architecture

### Design Principles

1. **Separation of Concerns**: Metrics and tracing are independent but complementary
2. **Global Instances**: Single collector/tracer per application via singletons
3. **Label-Based Dimensions**: Prometheus labels for multi-dimensional analysis
4. **Context Propagation**: Trace context flows through async operations
5. **Zero Dependencies**: Works without external backends (Prometheus/Jaeger optional)

### Component Interaction

```python
# Initialize components
from agentcore.agent_runtime.services.metrics_collector import get_metrics_collector
from agentcore.agent_runtime.services.distributed_tracing import get_distributed_tracer

collector = get_metrics_collector()
tracer = get_distributed_tracer()

# Use together
tracer.start_trace(trace_id="request-123")
span = tracer.start_span("create_agent")

collector.record_agent_created(AgentPhilosophy.REACT)
# ... agent creation logic ...
collector.record_agent_initialization(AgentPhilosophy.REACT, duration_seconds=1.5)

tracer.finish_span(span)
```

## Metrics Collection

### Overview

MetricsCollector provides Prometheus-compatible metrics for:
- Agent lifecycle (creation, execution, state transitions)
- Resource usage (CPU, memory, containers)
- Performance optimization (cache, pool, GC)
- Philosophy-specific behavior (ReAct iterations, CoT steps, etc.)
- Tool execution
- Errors and failures

### Metric Types

**Counter**: Monotonically increasing value (total agent creations, errors)
**Gauge**: Current value that can go up/down (active agents, CPU usage)
**Histogram**: Distribution of values (execution duration, initialization time)
**Summary**: Similar to histogram with quantiles
**Info**: Textual information (runtime version, platform)

### Core Metrics

#### Agent Lifecycle Metrics

```python
# Agent creation
collector.record_agent_created(
    philosophy=AgentPhilosophy.REACT,
    status="initializing"
)

# Agent completion
collector.record_agent_completed(
    philosophy=AgentPhilosophy.REACT,
    status="completed"  # or "failed", "terminated"
)

# Initialization duration
collector.record_agent_initialization(
    philosophy=AgentPhilosophy.REACT,
    duration_seconds=1.5
)

# Execution duration
collector.record_agent_execution(
    philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT,
    duration_seconds=30.5
)

# State transitions
collector.record_state_transition(
    from_state="initializing",
    to_state="running",
    philosophy=AgentPhilosophy.REACT
)
```

**Prometheus Metrics Generated**:
- `agentcore_agents_total{philosophy, status}` (Counter)
- `agentcore_agents_active{philosophy}` (Gauge)
- `agentcore_agent_initialization_seconds{philosophy}` (Histogram)
- `agentcore_agent_execution_seconds{philosophy}` (Histogram)
- `agentcore_agent_state_transitions_total{from_state, to_state, philosophy}` (Counter)

#### Resource Metrics

```python
# Agent resource usage
collector.update_resource_usage(
    agent_id="agent-123",
    philosophy=AgentPhilosophy.REACT,
    cpu_percent=45.5,
    memory_mb=256.0
)

# Container creation
collector.record_container_creation(
    philosophy=AgentPhilosophy.REACT,
    duration_seconds=0.05,
    warm_start=True
)

# System resources
collector.update_system_resources(
    cpu_percent=65.5,
    memory_percent=72.3,
    memory_available_mb=4096.0
)
```

**Prometheus Metrics Generated**:
- `agentcore_agent_cpu_percent{agent_id, philosophy}` (Gauge)
- `agentcore_agent_memory_mb{agent_id, philosophy}` (Gauge)
- `agentcore_container_creation_seconds{philosophy, warm_start}` (Histogram)
- `agentcore_system_cpu_percent` (Gauge)
- `agentcore_system_memory_percent` (Gauge)
- `agentcore_system_memory_available_mb` (Gauge)

#### Performance Metrics

```python
# Cache access
collector.record_cache_access(
    cache_type="tool_metadata",
    hit=True,  # or False for miss
    current_size=150
)

# Garbage collection
collector.record_gc_collection(memory_released_mb=25.5)
```

**Prometheus Metrics Generated**:
- `agentcore_cache_hits_total{cache_type}` (Counter)
- `agentcore_cache_misses_total{cache_type}` (Counter)
- `agentcore_cache_size{cache_type}` (Gauge)
- `agentcore_container_pool_size{philosophy}` (Gauge)
- `agentcore_warm_starts_total{philosophy}` (Counter)
- `agentcore_cold_starts_total{philosophy}` (Counter)
- `agentcore_gc_collections_total` (Counter)
- `agentcore_memory_released_mb_total` (Counter)

#### Philosophy-Specific Metrics

```python
# ReAct iterations
collector.record_react_iterations(agent_id="agent-123", iterations=5)

# Chain-of-Thought steps
collector.record_cot_steps(agent_id="agent-456", steps=7)

# Multi-agent communication
collector.record_multi_agent_message(message_type="direct")
collector.record_consensus(result="reached")

# Autonomous agent behavior
collector.record_autonomous_goal(status="active", priority="high")
collector.record_autonomous_decision(agent_id="autonomous-agent-1")
```

**Prometheus Metrics Generated**:
- `agentcore_react_iterations{agent_id}` (Histogram)
- `agentcore_cot_steps{agent_id}` (Histogram)
- `agentcore_multi_agent_messages_total{message_type}` (Counter)
- `agentcore_multi_agent_consensus_total{result}` (Counter)
- `agentcore_autonomous_goals_total{status, priority}` (Counter)
- `agentcore_autonomous_decisions_total{agent_id}` (Counter)

#### Tool Execution Metrics

```python
# Tool execution
collector.record_tool_execution(
    tool_id="calculator",
    duration_seconds=0.25,
    status="success"
)

# Tool execution with error
collector.record_tool_execution(
    tool_id="web_search",
    duration_seconds=1.5,
    status="failed",
    error_type="TimeoutError"
)
```

**Prometheus Metrics Generated**:
- `agentcore_tool_executions_total{tool_id, status}` (Counter)
- `agentcore_tool_execution_seconds{tool_id}` (Histogram)
- `agentcore_tool_errors_total{tool_id, error_type}` (Counter)

#### Error Metrics

```python
# General errors
collector.record_error(
    error_type="ValidationError",
    component="sandbox"
)

# Agent failures
collector.record_agent_failure(
    philosophy=AgentPhilosophy.REACT,
    failure_reason="execution_timeout"
)

# Runtime info
collector.set_runtime_info({
    "version": "1.0.0",
    "environment": "production",
    "platform": "linux"
})
```

**Prometheus Metrics Generated**:
- `agentcore_errors_total{error_type, component}` (Counter)
- `agentcore_agent_failures_total{philosophy, failure_reason}` (Counter)
- `agentcore_runtime_info{version, environment, platform}` (Info)

### Custom Metrics

Create application-specific metrics:

```python
# Create custom counter
custom_counter = collector.create_custom_metric(
    name="user_actions",
    metric_type=MetricType.COUNTER,
    description="User actions performed",
    labels=["action_type", "user_role"]
)

# Use custom metric
custom_counter.labels(action_type="agent_create", user_role="admin").inc()

# Create custom histogram
custom_histogram = collector.create_custom_metric(
    name="request_latency",
    metric_type=MetricType.HISTOGRAM,
    description="API request latency",
    labels=["endpoint"]
)

# Record value
custom_histogram.labels(endpoint="/api/agents").observe(0.123)

# Retrieve custom metric
metric = collector.get_custom_metric("user_actions")
```

### Metric Snapshots

Capture point-in-time metric values for analysis:

```python
# Create snapshot
snapshot = collector.snapshot_metrics()
# Returns:
# {
#     "timestamp": "2025-01-15T12:34:56.789Z",
#     "metrics": {
#         "agents": {"active": 15},
#         "system": {
#             "cpu_percent": 65.5,
#             "memory_percent": 72.3,
#             "memory_available_mb": 4096.0
#         }
#     }
# }

# Get metric history
history = collector.get_metric_history(limit=10)
# Returns last 10 snapshots

# Get specific metric history
tool_history = collector.get_metric_history(metric_name="tool_executions")
```

### Prometheus Integration

Export metrics for Prometheus scraping:

```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

collector = get_metrics_collector()

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    registry = collector.get_registry()
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )
```

**Prometheus Configuration** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'agentcore'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

## Distributed Tracing

### Overview

DistributedTracer provides OpenTelemetry-style distributed tracing with:
- Trace context management and propagation
- Span creation with parent-child relationships
- Event and link tracking
- Exception recording
- Decorator-based instrumentation

### Core Concepts

**Trace**: End-to-end request flow (unique trace_id)
**Span**: Individual operation within trace (unique span_id, parent_span_id)
**Context**: Trace metadata (trace_id, baggage, active span)
**Event**: Timestamped annotation within span
**Link**: Reference to another span (for fan-out scenarios)

### Trace Lifecycle

```python
from agentcore.agent_runtime.services.distributed_tracing import (
    get_distributed_tracer,
    SpanKind,
    SpanStatus
)

tracer = get_distributed_tracer()

# 1. Start trace
context = tracer.start_trace(
    trace_id="request-123",  # Optional, auto-generated if None
    baggage={"user_id": "12345", "tenant": "acme"}
)

# 2. Create spans
span = tracer.start_span(
    operation_name="create_agent",
    kind=SpanKind.INTERNAL,
    attributes={"philosophy": "react", "priority": "high"}
)

# 3. Add events
span.add_event("agent_initialized", attributes={"container_id": "cont-456"})
span.add_event("agent_started")

# 4. Set attributes
span.set_attribute("agent_id", "agent-789")
span.set_attribute("execution_mode", "async")

# 5. Handle errors
try:
    # ... operation ...
    pass
except Exception as e:
    tracer.record_exception(span, e)
    tracer.finish_span(span, status=SpanStatus.ERROR, status_message=str(e))
    raise

# 6. Finish span
tracer.finish_span(span, status=SpanStatus.OK)
```

### Span Management

**Span Kinds**:
- `INTERNAL`: Internal operation
- `SERVER`: Server-side RPC
- `CLIENT`: Client-side RPC
- `PRODUCER`: Message producer
- `CONSUMER`: Message consumer

**Span Status**:
- `UNSET`: Not yet determined
- `OK`: Successful completion
- `ERROR`: Failed execution

**Creating Nested Spans**:
```python
# Parent span
parent_span = tracer.start_span("agent_lifecycle")

# Child spans automatically use parent context
child_span1 = tracer.start_span("initialize_container")
tracer.finish_span(child_span1)

child_span2 = tracer.start_span("execute_task")
tracer.finish_span(child_span2)

tracer.finish_span(parent_span)
```

### Context Propagation

**Baggage** - Contextual key-value pairs propagated across spans:

```python
# Set baggage
context = tracer.start_trace()
context.set_baggage("user_id", "12345")
context.set_baggage("request_id", "req-789")

# Get baggage
user_id = context.get_baggage("user_id")

# Serialize context for cross-service calls
context_dict = context.to_dict()
# Send to another service...

# Deserialize on receiver side
received_context = TraceContext.from_dict(context_dict)
```

### Events and Links

**Events** - Annotations within span:
```python
span.add_event("cache_miss", attributes={"key": "tool:calculator"})
span.add_event("retry_attempt", attributes={"attempt": 2, "delay_ms": 1000})
```

**Links** - References to other spans:
```python
# Link to related operation in different trace
span.add_link(
    trace_id="trace-999",
    span_id="span-888",
    attributes={"relationship": "caused_by"}
)
```

### Decorator-Based Tracing

Automatically trace function execution:

```python
from agentcore.agent_runtime.services.distributed_tracing import trace_operation

@trace_operation(
    operation_name="create_agent",
    kind=SpanKind.INTERNAL,
    attributes={"component": "agent_lifecycle"}
)
async def create_agent(config: AgentConfig) -> Agent:
    # Function automatically traced
    # Exceptions automatically recorded
    agent = Agent(config)
    return agent

# Usage
agent = await create_agent(config)
```

**Works with sync and async functions**:
```python
@trace_operation(operation_name="calculate_metrics")
def calculate_metrics(data: list) -> dict:
    # Sync function automatically traced
    return {"count": len(data), "sum": sum(data)}
```

### Trace Analysis

**Get Trace Spans**:
```python
spans = tracer.get_trace_spans("trace-123")
for span in spans:
    print(f"{span.operation_name}: {span.duration_ms()}ms")
```

**Get Span by ID**:
```python
span = tracer.get_span_by_id("span-456")
if span:
    print(f"Status: {span.status}, Duration: {span.duration_ms()}ms")
```

**Trace Summary**:
```python
summary = tracer.get_trace_summary("trace-123")
# Returns:
# {
#     "trace_id": "trace-123",
#     "span_count": 15,
#     "total_duration_ms": 1234.5,
#     "error_count": 2,
#     "start_time": 1705312496.123,
#     "end_time": 1705312497.357,
#     "operations": ["create_agent", "initialize_container", ...]
# }
```

**Tracing Metrics**:
```python
metrics = tracer.get_metrics()
# Returns:
# {
#     "total_spans": 150,
#     "error_spans": 5,
#     "error_rate_percent": 3.33,
#     "unique_traces": 25,
#     "export_enabled": True
# }
```

### Export Integration

**Jaeger Export** (future enhancement):
```python
tracer = DistributedTracer(
    service_name="agent-runtime",
    enable_export=True
)

# Spans automatically exported to Jaeger/Zipkin
# Configure via environment variables:
# JAEGER_AGENT_HOST=localhost
# JAEGER_AGENT_PORT=6831
```

## Integration Examples

### Complete Agent Lifecycle

```python
from agentcore.agent_runtime.services.metrics_collector import get_metrics_collector
from agentcore.agent_runtime.services.distributed_tracing import get_distributed_tracer

collector = get_metrics_collector()
tracer = get_distributed_tracer()

async def create_and_run_agent(config: AgentConfig):
    """Create and run agent with full observability."""

    # Start trace
    tracer.start_trace(trace_id=f"agent-{config.agent_id}")

    # Create agent span
    create_span = tracer.start_span(
        "create_agent",
        attributes={"philosophy": config.philosophy.value}
    )

    try:
        # Track creation
        start_time = time.time()
        collector.record_agent_created(config.philosophy)

        # Create container with metrics
        collector.record_container_creation(
            philosophy=config.philosophy,
            duration_seconds=0.05,
            warm_start=True
        )

        # Record initialization
        init_duration = time.time() - start_time
        collector.record_agent_initialization(
            config.philosophy,
            init_duration
        )

        create_span.add_event("agent_initialized")
        tracer.finish_span(create_span, SpanStatus.OK)

        # Execute agent
        exec_span = tracer.start_span("execute_agent")
        exec_start = time.time()

        # Track resource usage
        collector.update_resource_usage(
            agent_id=config.agent_id,
            philosophy=config.philosophy,
            cpu_percent=45.5,
            memory_mb=256.0
        )

        # ... agent execution ...

        exec_duration = time.time() - exec_start
        collector.record_agent_execution(
            config.philosophy,
            exec_duration
        )

        tracer.finish_span(exec_span, SpanStatus.OK)

        # Complete
        collector.record_agent_completed(config.philosophy, "completed")

    except Exception as e:
        tracer.record_exception(create_span, e)
        tracer.finish_span(create_span, SpanStatus.ERROR, str(e))
        collector.record_agent_failure(config.philosophy, str(type(e).__name__))
        raise
```

### Tool Execution with Observability

```python
@trace_operation(operation_name="execute_tool", kind=SpanKind.CLIENT)
async def execute_tool(tool_id: str, args: dict):
    """Execute tool with metrics and tracing."""
    collector = get_metrics_collector()
    tracer = get_distributed_tracer()

    start_time = time.time()

    try:
        # Check cache
        cached = get_cached_result(tool_id, args)
        if cached:
            collector.record_cache_access("tool_results", hit=True, current_size=100)
            return cached

        collector.record_cache_access("tool_results", hit=False, current_size=100)

        # Execute tool
        result = await tool_executor.execute(tool_id, args)

        # Record success
        duration = time.time() - start_time
        collector.record_tool_execution(
            tool_id=tool_id,
            duration_seconds=duration,
            status="success"
        )

        return result

    except Exception as e:
        # Record failure
        duration = time.time() - start_time
        collector.record_tool_execution(
            tool_id=tool_id,
            duration_seconds=duration,
            status="failed",
            error_type=type(e).__name__
        )

        collector.record_error(
            error_type=type(e).__name__,
            component="tool_executor"
        )

        raise
```

### System Resource Monitoring

```python
import asyncio
import psutil

async def monitor_system_resources():
    """Continuously monitor and report system resources."""
    collector = get_metrics_collector()

    while True:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Update metrics
        collector.update_system_resources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024)
        )

        # Create snapshot
        snapshot = collector.snapshot_metrics()

        # Log if resources high
        if cpu_percent > 80 or memory.percent > 80:
            logger.warning(
                "high_resource_usage",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent
            )

        await asyncio.sleep(30)  # Check every 30 seconds
```

## Best Practices

### 1. Use Global Instances

**Recommended**:
```python
collector = get_metrics_collector()
tracer = get_distributed_tracer()
```

**Reason**: Ensures consistent metrics registry and trace context across application.

### 2. Label Metrics Appropriately

**Good**:
```python
collector.record_tool_execution(
    tool_id="calculator",  # Specific tool
    status="success"       # Specific status
)
```

**Bad**:
```python
collector.record_tool_execution(
    tool_id="all_tools",   # Too generic
    status="done"          # Vague status
)
```

**Reason**: Specific labels enable precise querying and alerting.

### 3. Create Meaningful Spans

**Good**:
```python
span = tracer.start_span(
    "execute_reasoning_chain",
    attributes={
        "chain_type": "react",
        "max_iterations": 10,
        "timeout_seconds": 300
    }
)
```

**Bad**:
```python
span = tracer.start_span("process")  # Too vague
```

**Reason**: Descriptive spans and attributes enable effective trace analysis.

### 4. Always Finish Spans

**Recommended**:
```python
span = tracer.start_span("operation")
try:
    # ... operation ...
    tracer.finish_span(span, SpanStatus.OK)
except Exception as e:
    tracer.record_exception(span, e)
    tracer.finish_span(span, SpanStatus.ERROR, str(e))
    raise
```

**Reason**: Unfinished spans leak memory and break trace visualization.

### 5. Use Decorators for Common Operations

**Recommended**:
```python
@trace_operation(operation_name="create_agent")
async def create_agent(config: AgentConfig):
    # Automatically traced
    pass
```

**Manual Alternative** (more verbose):
```python
async def create_agent(config: AgentConfig):
    span = tracer.start_span("create_agent")
    try:
        # ... operation ...
        tracer.finish_span(span)
    except Exception as e:
        tracer.record_exception(span, e)
        tracer.finish_span(span, SpanStatus.ERROR)
        raise
```

**Reason**: Decorators reduce boilerplate and prevent missing exception handling.

### 6. Monitor Error Rates

```python
metrics = collector.get_metrics()
error_rate = metrics["error_spans"] / metrics["total_spans"] * 100

if error_rate > 5:  # 5% error threshold
    alert_ops_team(f"High error rate: {error_rate}%")
```

### 7. Create Custom Metrics for Business Logic

```python
# Track business-specific events
deployment_counter = collector.create_custom_metric(
    name="agent_deployments",
    metric_type=MetricType.COUNTER,
    description="Agent deployments by environment",
    labels=["environment", "philosophy"]
)

deployment_counter.labels(
    environment="production",
    philosophy="react"
).inc()
```

### 8. Regular Snapshot Creation

```python
async def snapshot_task():
    """Create regular metric snapshots."""
    collector = get_metrics_collector()

    while True:
        snapshot = collector.snapshot_metrics()
        # Optionally persist to database for long-term analysis
        await save_snapshot_to_db(snapshot)
        await asyncio.sleep(300)  # Every 5 minutes
```

## Testing

### Test Coverage

**Overall**: 73 test scenarios, 99% coverage

**Breakdown**:

**MetricsCollector (36 tests)**:
- Collector initialization (3 tests)
- Agent metrics (5 tests)
- Resource metrics (3 tests)
- Performance metrics (3 tests)
- Philosophy metrics (6 tests)
- Tool metrics (2 tests)
- Error metrics (3 tests)
- Custom metrics (6 tests)
- Metric snapshots (4 tests)
- Registry access (1 test)

**DistributedTracer (37 tests)**:
- Span creation/management (9 tests)
- Trace context (6 tests)
- Distributed tracer (12 tests)
- Decorator tracing (5 tests)
- Global instance (1 test)
- Integration (4 tests)

### Running Tests

```bash
# Run all monitoring tests
uv run pytest tests/agent_runtime/test_metrics_collector.py tests/agent_runtime/test_distributed_tracing.py -v

# Run with coverage
uv run pytest tests/agent_runtime/test_metrics_collector.py tests/agent_runtime/test_distributed_tracing.py \
    --cov=src/agentcore/agent_runtime/services/metrics_collector \
    --cov=src/agentcore/agent_runtime/services/distributed_tracing \
    --cov-report=term-missing

# Run specific test class
uv run pytest tests/agent_runtime/test_metrics_collector.py::TestAgentMetrics -v

# Run single test
uv run pytest tests/agent_runtime/test_distributed_tracing.py::TestSpan::test_span_duration -v
```

### Test Results

```
============ 73 passed in 6.35s ============

Coverage:
- metrics_collector.py: 99% (1 line uncovered)
- distributed_tracing.py: 99% (2 lines uncovered)
```

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [Jaeger Tracing](https://www.jaegertracing.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

## Support

For monitoring and observability questions:
- Review this documentation
- Check test coverage in `tests/agent_runtime/test_metrics_collector.py` and `test_distributed_tracing.py`
- Monitor Prometheus metrics at `/metrics` endpoint
- Analyze traces using trace summaries
- Consult Prometheus and OpenTelemetry documentation
