# OpenTelemetry Distributed Tracing Integration

## Overview

The modular agent core now includes comprehensive OpenTelemetry distributed tracing to track execution flow across all modules (Planner, Executor, Verifier, Generator) with automatic span creation, context propagation, and trace export.

## Features

- **Automatic Span Creation**: Spans automatically created for each module operation
- **Tool Execution Tracing**: Tool invocations linked to parent module spans
- **A2A Context Propagation**: Trace context propagated via A2A protocol
- **OTLP Export**: Compatible with Jaeger, Zipkin, and other OTLP collectors
- **Decorator Support**: `@trace_module_operation` for automatic instrumentation
- **Error Tracking**: Exceptions and errors recorded in spans with status codes
- **Sensitive Data Sanitization**: Automatic redaction of secrets in span attributes

## Quick Start

### 1. Initialize Tracing

```python
from agentcore.modular.tracing import TracingConfig, initialize_tracing

# Configure tracing
config = TracingConfig(
    enabled=True,
    service_name="my-modular-agent",
    otlp_endpoint="http://localhost:4317",  # Jaeger/Zipkin collector
    console_export=False,  # Set True for debugging
    sample_rate=1.0,  # 100% sampling
)

# Initialize global tracer
tracer = initialize_tracing(config)
```

### 2. Create Module Spans

```python
from agentcore.modular.tracing import get_tracer
from agentcore.modular.models import ModuleType

tracer = get_tracer()

# Create span for module operation
span = tracer.create_module_span(
    module_type=ModuleType.PLANNER,
    operation="analyze_query",
    attributes={
        "query": "Calculate fibonacci(10)",
        "max_iterations": 5,
    }
)

try:
    # Perform work...
    result = await planner.analyze_query(query)

    # Mark success
    tracer.set_span_success(span)

except Exception as e:
    # Record exception
    tracer.record_exception(span, e)
    raise

finally:
    # Always end span
    span.end()
```

### 3. Trace Tool Execution

```python
# Link tool execution to parent span
parent_span = tracer.create_module_span(
    module_type=ModuleType.EXECUTOR,
    operation="execute_step",
)

tool_span = tracer.create_tool_span(
    tool_name="calculator",
    parameters={"operation": "add", "a": 2, "b": 3},
    parent_span=parent_span,
)

# Execute tool...
result = await tool.execute()

tool_span.end()
parent_span.end()
```

### 4. Automatic Tracing with Decorator

```python
from agentcore.modular.tracing import trace_module_operation

class MyPlanner:
    @trace_module_operation(ModuleType.PLANNER, "analyze_query")
    async def analyze_query(self, query: str) -> ExecutionPlan:
        # Span automatically created and managed
        return await self._do_analysis(query)
```

### 5. A2A Context Propagation

```python
# Build A2A context with trace headers
span = tracer.create_module_span(
    module_type=ModuleType.PLANNER,
    operation="analyze_query",
)

a2a_context = tracer.build_a2a_context_carrier({
    "source_agent": "planner",
    "target_agent": "executor",
    "session_id": "session-123",
})

# a2a_context now contains:
# {
#   "source_agent": "planner",
#   "target_agent": "executor",
#   "session_id": "session-123",
#   "trace_headers": {"traceparent": "..."},
#   "trace_id": "abc123..."
# }

# Send to next module with trace context
await send_to_executor(request, a2a_context)
```

### 6. Add Events and Annotations

```python
span = tracer.create_module_span(
    module_type=ModuleType.VERIFIER,
    operation="validate_results",
)

# Add event
tracer.add_event(
    span,
    "validation_started",
    attributes={"result_count": 3}
)

# Perform validation...
if validation_passed:
    tracer.add_event(span, "validation_passed")
else:
    tracer.add_event(
        span,
        "validation_failed",
        attributes={"errors": ["incomplete_data"]}
    )

span.end()
```

## Configuration Options

```python
class TracingConfig:
    enabled: bool = True                    # Enable/disable tracing
    service_name: str = "agentcore-modular" # Service name in traces
    otlp_endpoint: str | None = None        # OTLP collector URL
    console_export: bool = False            # Export to console (debug)
    sample_rate: float = 1.0                # Sampling rate (0.0-1.0)
    batch_export: bool = True               # Batch vs simple export
    max_export_batch_size: int = 512        # Max batch size
    export_timeout_ms: int = 30000          # Export timeout
```

## Deployment with Jaeger

### 1. Start Jaeger (Docker)

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

### 2. Configure AgentCore

```python
config = TracingConfig(
    enabled=True,
    service_name="agentcore-modular",
    otlp_endpoint="http://localhost:4317",
)
initialize_tracing(config)
```

### 3. View Traces

Open http://localhost:16686 in browser to view Jaeger UI.

## Trace Visualization

When viewing traces in Jaeger/Zipkin, you'll see:

```
modular.solve [500ms]
  ├─ planner.analyze_query [100ms]
  │   └─ planner.create_steps [50ms]
  ├─ executor.execute_plan [250ms]
  │   ├─ tool.calculator [50ms]
  │   ├─ tool.search_engine [150ms]
  │   └─ tool.data_processor [40ms]
  ├─ verifier.validate_results [80ms]
  │   └─ verifier.check_consistency [30ms]
  └─ generator.synthesize_response [70ms]
```

Each span includes:
- Module type and operation name
- Timing information (start, duration)
- Attributes (query, parameters, counts)
- Events (validation_passed, error_occurred)
- Status (OK, ERROR)
- Parent-child relationships
- Trace ID for correlation

## Best Practices

### 1. Always End Spans

Use try/finally to ensure spans are ended:

```python
span = tracer.create_module_span(...)
try:
    # Work...
finally:
    span.end()
```

### 2. Meaningful Attributes

Add attributes that help debug issues:

```python
span = tracer.create_module_span(
    module_type=ModuleType.EXECUTOR,
    operation="execute_step",
    attributes={
        "step_id": "step-1",
        "action": "search",
        "retry_count": 2,
        "estimated_cost": 0.05,
    }
)
```

### 3. Record Exceptions

Always record exceptions in spans:

```python
try:
    result = await execute()
except Exception as e:
    tracer.record_exception(span, e)
    raise
```

### 4. Use Decorator for Consistency

Prefer `@trace_module_operation` for automatic instrumentation:

```python
@trace_module_operation(ModuleType.PLANNER)
async def analyze_query(self, query: str) -> Plan:
    # Automatic span management
    pass
```

### 5. Sanitize Sensitive Data

The tracer automatically sanitizes common secrets, but be careful with custom attributes:

```python
# Automatically sanitized
tool_span = tracer.create_tool_span(
    tool_name="api_client",
    parameters={
        "api_key": "secret",  # Becomes "***REDACTED***"
        "data": {"name": "test"}  # Not sanitized
    }
)
```

## Integration with Coordinator

The tracing module integrates seamlessly with the coordination loop:

```python
from agentcore.modular.coordinator import ModuleCoordinator
from agentcore.modular.tracing import initialize_tracing, get_tracer

# Initialize tracing
initialize_tracing(TracingConfig(
    enabled=True,
    otlp_endpoint="http://localhost:4317",
))

# Create coordinator
coordinator = ModuleCoordinator()

# Execute with automatic tracing
result = await coordinator.execute_with_refinement(
    query="Calculate fibonacci(10)",
    planner=planner,
    executor=executor,
    verifier=verifier,
    generator=generator,
)

# Trace shows full execution flow across all modules
```

## Environment Variables

Configure tracing via environment:

```bash
# Enable/disable tracing
export AGENTCORE_TRACING_ENABLED=true

# OTLP endpoint
export AGENTCORE_OTLP_ENDPOINT=http://localhost:4317

# Service name
export AGENTCORE_SERVICE_NAME=my-agent

# Sample rate (0.0-1.0)
export AGENTCORE_TRACE_SAMPLE_RATE=0.5
```

## Troubleshooting

### No Traces Appearing

1. Check tracer initialization:
```python
from agentcore.modular.tracing import get_tracer
tracer = get_tracer()  # Should not raise exception
```

2. Verify OTLP endpoint is reachable:
```bash
curl http://localhost:4317
```

3. Enable console export for debugging:
```python
config = TracingConfig(console_export=True)
initialize_tracing(config)
```

### High Memory Usage

Reduce batch size or increase export frequency:

```python
config = TracingConfig(
    max_export_batch_size=128,  # Reduce from 512
    export_timeout_ms=5000,      # Export more frequently
)
```

### Missing Parent-Child Links

Ensure spans are created with parent context:

```python
parent_span = tracer.create_module_span(...)

# Pass parent explicitly
child_span = tracer.create_tool_span(
    tool_name="calculator",
    parent_span=parent_span  # Important!
)
```

## Performance Considerations

- **Sampling**: Use `sample_rate < 1.0` in production for high-throughput systems
- **Batch Export**: Keep `batch_export=True` for better performance
- **Attribute Size**: Limit attribute values to <1KB to avoid overhead
- **Span Count**: Each request should have <100 spans to avoid memory issues

## References

- [OpenTelemetry Python SDK](https://opentelemetry-python.readthedocs.io/)
- [OTLP Specification](https://opentelemetry.io/docs/reference/specification/protocol/otlp/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
