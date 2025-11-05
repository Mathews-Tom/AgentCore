# Coordination Service

The Coordination Service implements the Ripple Effect Protocol (REP) for intelligent agent coordination and load balancing in distributed agentic systems.

## Overview

The service enables agents to register sensitivity signals indicating their current state (load, capacity, quality, cost) and provides optimal agent selection for routing decisions based on multi-dimensional scoring.

### Key Features

- **Multi-dimensional Signal Types:** Load, Capacity, Quality, Cost
- **Predictive Overload Detection:** Forecast agent overload before it occurs
- **Configurable Routing Weights:** Customize optimization priorities
- **Signal TTL Management:** Automatic cleanup of expired signals
- **Prometheus Metrics:** Production-ready observability
- **Sub-millisecond Performance:** Ultra-low latency coordination

## Quick Start

### Register Agent Signals

```python
from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import coordination_service

# Register agent load signal
signal = SensitivitySignal(
    agent_id="agent-001",
    signal_type=SignalType.LOAD,
    value=0.65,  # 65% loaded
    ttl_seconds=60
)
coordination_service.register_signal(signal)
```

### Select Optimal Agent

```python
# Select best agent from candidates
candidates = ["agent-001", "agent-002", "agent-003"]
best_agent = coordination_service.select_optimal_agent(candidates)
```

### Predict Overload

```python
# Predict if agent will overload in next 60 seconds
will_overload, probability = coordination_service.predict_overload(
    agent_id="agent-001",
    forecast_seconds=60,
    threshold=0.8
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MessageRouter                          │
│  (RIPPLE_COORDINATION strategy)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              CoordinationService                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Signal Registration  │  Agent Selection             │  │
│  │  - Multi-signal types │  - Weighted scoring          │  │
│  │  - TTL management     │  - Optimal candidate         │  │
│  │  - State tracking     │  - Fallback handling         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Overload Prediction  │  Metrics Integration         │  │
│  │  - Time-series trend  │  - Prometheus counters       │  │
│  │  - Linear regression  │  - Latency histograms        │  │
│  │  - Configurable threshold                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Signal Types

| Type | Description | Value Range | Example Use Case |
|------|-------------|-------------|------------------|
| **LOAD** | Current processing load | 0.0-1.0 | CPU/memory usage |
| **CAPACITY** | Available processing capacity | 0.0-1.0 | Queue space available |
| **QUALITY** | Service quality score | 0.0-1.0 | Success rate, accuracy |
| **COST** | Processing cost estimate | 0.0-1.0 | Monetary or resource cost |

## Routing Score Calculation

The coordination service computes a multi-dimensional routing score for each agent:

```
routing_score = (w_load × (1 - load_signal)) +
                (w_capacity × capacity_signal) +
                (w_quality × quality_signal) +
                (w_cost × (1 - cost_signal))
```

**Default Weights:**
- Load: 0.4
- Capacity: 0.3
- Quality: 0.2
- Cost: 0.1

Higher scores indicate better routing candidates.

## Configuration

### Optimization Weights

Customize routing priorities by adjusting signal weights:

```python
# Prioritize quality over load
coordination_service.set_optimization_weights(
    load=0.2,
    capacity=0.2,
    quality=0.5,
    cost=0.1
)
```

### Signal TTL

Set appropriate TTL values based on signal volatility:

- **Load signals:** 30-60 seconds (high volatility)
- **Capacity signals:** 60-120 seconds (moderate volatility)
- **Quality signals:** 120-300 seconds (low volatility)
- **Cost signals:** 300+ seconds (stable)

## JSON-RPC API

The coordination service exposes four JSON-RPC methods:

### `coordination.signal`

Register agent sensitivity signal.

**Parameters:**
```json
{
  "agent_id": "agent-001",
  "signal_type": "LOAD",
  "value": 0.65,
  "ttl_seconds": 60
}
```

**Returns:**
```json
{
  "success": true,
  "agent_id": "agent-001",
  "signals_registered": 5
}
```

### `coordination.state`

Retrieve agent coordination state.

**Parameters:**
```json
{
  "agent_id": "agent-001"
}
```

**Returns:**
```json
{
  "agent_id": "agent-001",
  "signals": [...],
  "routing_score": 0.75,
  "last_updated": "2025-11-05T06:57:00Z"
}
```

### `coordination.metrics`

Get coordination metrics snapshot.

**Returns:**
```json
{
  "agents_tracked": 150,
  "total_signals": 520,
  "total_selections": 1420,
  "signals_by_type": {
    "LOAD": 200,
    "CAPACITY": 180,
    "QUALITY": 100,
    "COST": 40
  }
}
```

### `coordination.predict_overload`

Predict agent overload probability.

**Parameters:**
```json
{
  "agent_id": "agent-001",
  "forecast_seconds": 60,
  "threshold": 0.8
}
```

**Returns:**
```json
{
  "agent_id": "agent-001",
  "will_overload": true,
  "probability": 0.92,
  "forecast_seconds": 60
}
```

## Performance

**SLO Compliance (p95 latency):**
- Signal registration: <5ms (measured: 0.012ms)
- Routing score retrieval: <2ms (measured: 0.004ms)
- Agent selection (100 candidates): <10ms (measured: 0.403ms)

See [Performance Report](../coordination-performance-report.md) for detailed benchmarks.

## Monitoring

### Prometheus Metrics

**Counters:**
- `coordination_signals_total{agent_id, signal_type}` - Total signals registered
- `coordination_routing_selections_total{strategy}` - Total routing selections
- `coordination_overload_predictions_total{agent_id, predicted}` - Overload predictions

**Gauges:**
- `coordination_agents_total` - Active agents tracked

**Histograms:**
- `coordination_signal_registration_duration_seconds{signal_type}` - Registration latency
- `coordination_agent_selection_duration_seconds{strategy}` - Selection latency

### Example Queries

```promql
# p95 signal registration latency
histogram_quantile(0.95,
  rate(coordination_signal_registration_duration_seconds_bucket[5m]))

# Signals registered per second
rate(coordination_signals_total[1m])

# Active agents over time
coordination_agents_total
```

## Integration with MessageRouter

The coordination service integrates with MessageRouter via the `RIPPLE_COORDINATION` routing strategy:

```python
from agentcore.a2a_protocol.services.message_router import MessageRouter
from agentcore.a2a_protocol.models.message_router import RoutingStrategy

router = MessageRouter()

# Use RIPPLE_COORDINATION for intelligent routing
selected_agent = await router.route_message(
    message=msg,
    agents=available_agents,
    strategy=RoutingStrategy.RIPPLE_COORDINATION
)
```

## Troubleshooting

### Issue: Agent selection always returns None

**Cause:** No coordination state exists for candidate agents.

**Solution:** Ensure agents register signals before selection:
```python
# Register signals for all candidates
for agent_id in candidates:
    signal = SensitivitySignal(agent_id=agent_id, signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60)
    coordination_service.register_signal(signal)

# Now selection will work
selected = coordination_service.select_optimal_agent(candidates)
```

### Issue: Overload prediction returns low probability despite high load

**Cause:** Insufficient historical data (need 3+ data points).

**Solution:** Register multiple signals over time to build trend:
```python
# Register signals periodically
import time
for i in range(5):
    signal = SensitivitySignal(agent_id="agent-001", signal_type=SignalType.LOAD, value=0.7 + (i * 0.05), ttl_seconds=300)
    coordination_service.register_signal(signal)
    time.sleep(5)

# Now prediction has enough data
will_overload, prob = coordination_service.predict_overload("agent-001", forecast_seconds=60, threshold=0.9)
```

### Issue: Signal TTL cleanup not working

**Cause:** Cleanup service not running.

**Solution:** Start signal cleanup service:
```python
from agentcore.a2a_protocol.services.signal_cleanup import signal_cleanup_service

# Start cleanup (runs every 30 seconds by default)
await signal_cleanup_service.start()
```

## References

- **Ripple Effect Protocol (REP):** [Research Paper](#) (sensitivity propagation theory)
- **A2A Protocol:** Google's Agent-to-Agent communication specification
- **MessageRouter:** [MessageRouter Documentation](../message-router.md)
- **Performance Benchmarks:** [Performance Report](../coordination-performance-report.md)

## Migration Guide

### From RANDOM Routing

Replace `RoutingStrategy.RANDOM` with `RoutingStrategy.RIPPLE_COORDINATION`:

```python
# Before (random routing)
selected = await router.route_message(msg, agents, RoutingStrategy.RANDOM)

# After (coordination-based routing)
# 1. Register agent signals
for agent_id in agents:
    signal = SensitivitySignal(agent_id=agent_id, signal_type=SignalType.LOAD, value=get_agent_load(agent_id), ttl_seconds=60)
    coordination_service.register_signal(signal)

# 2. Use RIPPLE_COORDINATION strategy
selected = await router.route_message(msg, agents, RoutingStrategy.RIPPLE_COORDINATION)
```

### From LEAST_LOADED Routing

RIPPLE_COORDINATION is a superset of LEAST_LOADED with multi-dimensional optimization:

```python
# Before (single dimension: load)
selected = get_least_loaded_agent(agents)

# After (multi-dimensional: load + capacity + quality + cost)
# Signals provide richer agent state
for agent_id in agents:
    coordination_service.register_signal(SensitivitySignal(agent_id=agent_id, signal_type=SignalType.LOAD, value=get_load(agent_id), ttl_seconds=60))
    coordination_service.register_signal(SensitivitySignal(agent_id=agent_id, signal_type=SignalType.CAPACITY, value=get_capacity(agent_id), ttl_seconds=60))

selected = coordination_service.select_optimal_agent(agents)
```

## Example: Complete Workflow

```python
from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import coordination_service
from agentcore.a2a_protocol.services.message_router import MessageRouter
from agentcore.a2a_protocol.models.message_router import RoutingStrategy

# 1. Agents register their state periodically
def register_agent_state(agent_id: str) -> None:
    # Get current agent metrics
    load = get_current_load(agent_id)
    capacity = get_available_capacity(agent_id)
    quality = get_service_quality(agent_id)

    # Register signals
    coordination_service.register_signal(SensitivitySignal(agent_id=agent_id, signal_type=SignalType.LOAD, value=load, ttl_seconds=60))
    coordination_service.register_signal(SensitivitySignal(agent_id=agent_id, signal_type=SignalType.CAPACITY, value=capacity, ttl_seconds=60))
    coordination_service.register_signal(SensitivitySignal(agent_id=agent_id, signal_type=SignalType.QUALITY, value=quality, ttl_seconds=120))

# 2. Route incoming message using coordination
async def handle_incoming_message(msg) -> None:
    router = MessageRouter()
    candidates = get_available_agents()

    # Check for overload prediction
    for agent_id in candidates:
        will_overload, prob = coordination_service.predict_overload(agent_id, forecast_seconds=60, threshold=0.8)
        if will_overload:
            print(f"Warning: {agent_id} predicted to overload (probability: {prob:.2f})")

    # Select optimal agent
    selected_agent = await router.route_message(msg, candidates, RoutingStrategy.RIPPLE_COORDINATION)

    # Dispatch message
    await dispatch_to_agent(selected_agent, msg)

# 3. Monitor coordination metrics
def monitor_coordination() -> None:
    metrics = coordination_service.metrics
    print(f"Active agents: {metrics.agents_tracked}")
    print(f"Total signals: {metrics.total_signals}")
    print(f"Total selections: {metrics.total_selections}")
```

## License

Apache 2.0 - See LICENSE file for details.
