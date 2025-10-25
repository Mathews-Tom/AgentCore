# Coordination Service Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-25
**Specification:** `docs/specs/coordination-service/spec.md`
**Component ID:** COORD
**Priority:** P1 (MEDIUM - Phase 3)

---

## 📖 Context & Documentation

### Traceability Chain

**Specification → Research Context → This Plan**

1. **Formal Specification:** `docs/specs/coordination-service/spec.md`
   - REP (Ripple Effect Protocol)-inspired coordination
   - Functional requirements (FR-1 through FR-6)
   - Multi-objective optimization for agent selection
   - Preemptive overload prediction
   - Success metrics: 41-100% accuracy improvement

2. **Related Research:** `docs/research/modular-agent-architecture.md`
   - Modular agent coordination patterns
   - Specialized module communication (Planner, Executor, Verifier, Generator)
   - Message-passing coordination mechanisms
   - State management best practices

### Related Documentation

**System Context:**

- Architecture: `docs/agentcore-architecture-and-development-plan.md`
  - Core Layer integration (Communication Hub, Coordination)
  - Existing infrastructure: PostgreSQL, Redis, FastAPI
  - Service mesh pattern

**Project Guide:**

- `CLAUDE.md` - AgentCore patterns and conventions
- Tech Stack: Python 3.12+, FastAPI, asyncio, Pydantic

**Existing Code Patterns:**

- `src/agentcore/a2a_protocol/services/message_router.py` - Message routing infrastructure
- `src/agentcore/a2a_protocol/services/agent_manager.py` - Agent lifecycle and health
- `src/agentcore/a2a_protocol/services/jsonrpc_handler.py` - JSON-RPC method registration

**Cross-Component Dependencies:**

- MessageRouter: Primary consumer for coordination-based routing
- AgentManager: Agents register and report health signals
- JSON-RPC Handler: Expose coordination methods

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable intelligent multi-agent coordination through sensitivity signals to optimize task distribution, prevent overload, and improve system-wide performance.

**Value Proposition:**

- **41-100% Coordination Accuracy Improvement**: REP-validated agent selection
- **Preemptive Load Balancing**: Detect and prevent overload before it occurs
- **Dynamic Capability Routing**: Real-time awareness of agent capacity and quality
- **Reduced Failure Rates**: Predict and avoid agent failures
- **Optimal Resource Utilization**: Distribute tasks based on multi-objective optimization

**Target Users:**

- **MessageRouter**: Intelligent agent selection based on real-time signals
- **Agents**: Self-report capacity, load, and quality for coordination
- **Operators**: Monitor coordination health and load patterns
- **System**: Autonomous load balancing without manual intervention

### Technical Approach

**Architecture Pattern:** In-Memory Signal Aggregation with Async Event Processing

- **Core Service**: CoordinationService maintaining agent state
- **Signal Exchange**: Agents broadcast sensitivity signals (load, capacity, quality, cost, availability)
- **Multi-Objective Optimization**: Weighted scoring for optimal agent selection
- **Preemptive Prediction**: Linear regression on signal history for overload forecasting
- **MessageRouter Integration**: New RIPPLE_COORDINATION routing strategy

**Technology Stack:**

- Python 3.12+ (asyncio, type hints)
- Pydantic for signal validation
- In-memory state (dict-based, future Redis for distributed)
- Prometheus for metrics
- structlog for structured logging

**Implementation Strategy:**

- Phase 1 (Week 1): Core CoordinationService with signal aggregation
- Phase 2 (Week 2): MessageRouter integration, JSON-RPC, and validation

### Key Success Metrics

**Service Level Objectives (SLOs):**

- Availability: 99.9% (coordination operations non-blocking)
- Response Time: <10ms (p95 signal aggregation + selection)
- Throughput: 10,000 signals/second
- Error Rate: <0.1%

**Key Performance Indicators (KPIs):**

| Metric | Baseline | Target | REP Validation |
|--------|----------|--------|----------------|
| Coordination Accuracy | Random (50%) | 71-100% | ✅ REP paper: 41-100% improvement |
| Load Distribution Evenness | 60% | 90%+ | Gini coefficient <0.1 |
| Overload Prediction Accuracy | N/A | 80%+ | Precision & Recall ≥0.8 |
| Signal Freshness | N/A | 95% <60s | 95% routing uses signals <60s old |
| Routing Overhead | N/A | <10ms | p95 latency overhead |

**Effectiveness Validation (REP Paper Benchmarks):**

- Test against random routing baseline
- Measure task completion rate improvement
- Validate load distribution evenness (Gini coefficient)
- Track overload incidents before/after

---

## 💻 Code Examples & Patterns

### Signal Registration Pattern

**From Specification:**

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

class SignalType(str, Enum):
    """Types of sensitivity signals (REP)."""
    LOAD = "load"
    CAPACITY = "capacity"
    QUALITY = "quality"
    COST = "cost"
    LATENCY = "latency"
    AVAILABILITY = "availability"

class SensitivitySignal(BaseModel):
    """REP sensitivity signal from agent."""

    signal_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    signal_type: SignalType
    value: float = Field(ge=0.0, le=1.0, description="Normalized 0.0-1.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = Field(default=60, ge=10, le=600)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trace_id: str | None = None  # For distributed tracing

    def is_expired(self, current_time: datetime) -> bool:
        """Check if signal has expired based on TTL."""
        age = (current_time - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    def compute_decayed_value(self, current_time: datetime) -> float:
        """Apply temporal decay to signal value."""
        age = (current_time - self.timestamp).total_seconds()
        decay_factor = max(0.0, 1.0 - (age / self.ttl_seconds))
        return self.value * decay_factor
```

### Agent Coordination State

```python
class AgentCoordinationState(BaseModel):
    """Aggregated coordination state per agent."""

    agent_id: str
    signals: dict[SignalType, SensitivitySignal] = Field(default_factory=dict)
    signal_history: list[SensitivitySignal] = Field(default_factory=list, max_length=100)

    # Individual scores (0.0-1.0)
    load_score: float = 0.5  # Lower is better (inverted)
    capacity_score: float = 0.5  # Higher is better
    quality_score: float = 0.5  # Higher is better
    cost_score: float = 0.5  # Lower is better (inverted)
    availability_score: float = 0.5  # Higher is better

    # Composite routing score (weighted average)
    routing_score: float = 0.5

    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def get_active_signals(self, current_time: datetime) -> dict[SignalType, SensitivitySignal]:
        """Return only non-expired signals."""
        return {
            sig_type: signal
            for sig_type, signal in self.signals.items()
            if not signal.is_expired(current_time)
        }

    def add_signal_to_history(self, signal: SensitivitySignal) -> None:
        """Add signal to history with size limit."""
        self.signal_history.append(signal)
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]  # Keep last 100
```

### Multi-Objective Optimization

```python
class OptimizationWeights(BaseModel):
    """Configurable weights for multi-objective optimization."""

    load: float = 0.25
    capacity: float = 0.25
    quality: float = 0.20
    cost: float = 0.15
    availability: float = 0.15

    @model_validator(mode='after')
    def validate_sum(self) -> 'OptimizationWeights':
        """Ensure weights sum to 1.0."""
        total = self.load + self.capacity + self.quality + self.cost + self.availability
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self

def compute_routing_score(
    state: AgentCoordinationState,
    weights: OptimizationWeights
) -> float:
    """
    Compute composite routing score using multi-objective optimization.

    Higher score = better agent for routing.
    Load and cost are inverted (lower is better).
    """
    # Invert load (high load = low score)
    inverted_load = 1.0 - state.load_score

    # Invert cost (high cost = low score)
    inverted_cost = 1.0 - state.cost_score

    # Weighted combination
    score = (
        weights.load * inverted_load +
        weights.capacity * state.capacity_score +
        weights.quality * state.quality_score +
        weights.cost * inverted_cost +
        weights.availability * state.availability_score
    )

    return score
```

### Overload Prediction

```python
import statistics
from datetime import timedelta

def predict_overload(
    state: AgentCoordinationState,
    forecast_seconds: int = 60,
    threshold: float = 0.8
) -> tuple[bool, float]:
    """
    Predict if agent will overload within forecast window.

    Uses simple linear regression on recent load signals.

    Returns:
        (will_overload: bool, probability: float)
    """
    # Extract recent load signals
    recent_loads = [
        sig for sig in state.signal_history[-10:]
        if sig.signal_type == SignalType.LOAD
    ]

    if len(recent_loads) < 3:
        # Not enough data for prediction
        return (False, 0.0)

    # Simple linear regression: y = mx + b
    # x = time offset from first signal, y = load value
    first_time = recent_loads[0].timestamp
    x_values = [(sig.timestamp - first_time).total_seconds() for sig in recent_loads]
    y_values = [sig.value for sig in recent_loads]

    n = len(recent_loads)
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    # Calculate slope (m)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        # No trend
        return (recent_loads[-1].value > threshold, recent_loads[-1].value)

    slope = numerator / denominator

    # Predict load at forecast time
    forecast_x = x_values[-1] + forecast_seconds
    predicted_load = slope * (forecast_x - x_mean) + y_mean

    # Clamp to 0.0-1.0
    predicted_load = max(0.0, min(1.0, predicted_load))

    will_overload = predicted_load > threshold
    probability = predicted_load if slope > 0 else 0.0

    return (will_overload, probability)
```

### MessageRouter Integration

```python
from enum import Enum

class RoutingStrategy(str, Enum):
    """Routing strategies for message routing."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    RIPPLE_COORDINATION = "ripple_coordination"  # NEW

async def _ripple_coordination_select(
    self,
    candidates: list[str],
    weights: OptimizationWeights | None = None
) -> str:
    """
    Select optimal agent using REP coordination signals.

    Fallback to random if coordination unavailable.
    """
    if not candidates:
        raise ValueError("No candidate agents available")

    # Get coordination service
    coordination_service = get_coordination_service()

    try:
        # Retrieve routing scores for all candidates
        scores: dict[str, float] = {}
        for agent_id in candidates:
            state = coordination_service.get_agent_state(agent_id)
            if state:
                score = compute_routing_score(state, weights or OptimizationWeights())
                scores[agent_id] = score
            else:
                # No signals, use default score
                scores[agent_id] = 0.5

        # Select agent with highest score
        selected = max(scores, key=scores.get)

        # Log selection rationale
        logger.info(
            "RIPPLE_COORDINATION selection",
            selected_agent=selected,
            score=scores[selected],
            all_scores=scores
        )

        return selected

    except Exception as e:
        logger.warning("RIPPLE_COORDINATION failed, fallback to random", error=str(e))
        return random.choice(candidates)
```

### Key Patterns from AgentCore

**Pattern: Async Service with Global Instance**

```python
# Global coordination service instance
_coordination_service: CoordinationService | None = None

def get_coordination_service() -> CoordinationService:
    """Get global coordination service instance."""
    global _coordination_service
    if _coordination_service is None:
        _coordination_service = CoordinationService()
    return _coordination_service

# Usage in startup
@app.on_event("startup")
async def startup_event():
    """Initialize coordination service on startup."""
    coordination_service = get_coordination_service()
    await coordination_service.start_cleanup_task()
```

**Pattern: JSON-RPC Method Registration**

```python
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

@register_jsonrpc_method("coordination.signal")
async def handle_coordination_signal(request: JsonRpcRequest) -> dict[str, Any]:
    """Register sensitivity signal from agent."""
    signal = SensitivitySignal(**request.params)
    coordination_service = get_coordination_service()
    coordination_service.register_signal(signal)

    return {
        "registered": True,
        "signal_id": str(signal.signal_id),
        "agent_id": signal.agent_id
    }

@register_jsonrpc_method("coordination.state")
async def handle_coordination_state(request: JsonRpcRequest) -> dict[str, Any]:
    """Get coordination state for agent."""
    agent_id = request.params.get("agent_id")
    coordination_service = get_coordination_service()
    state = coordination_service.get_agent_state(agent_id)

    if state:
        return state.model_dump()
    else:
        return {"agent_id": agent_id, "routing_score": 0.5, "signals": {}}
```

### Anti-Patterns to Avoid

**From AgentCore Conventions:**

- ❌ Do not use synchronous blocking operations (use asyncio)
- ❌ Do not hardcode configuration values (use config.py)
- ❌ Do not ignore signal expiry (implement TTL correctly)
- ❌ Do not use mutable default arguments in functions
- ❌ Do not skip type hints (Python 3.12+ typing required)
- ❌ Do not use `typing.List/Dict/Optional` (use built-in `list`/`dict`/`|` unions)

---

## 🔧 Technology Stack

### Recommended Stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | AgentCore standard, modern type hints |
| Framework | FastAPI | Latest | Existing AgentCore framework |
| Data Validation | Pydantic | 2.0+ | Existing AgentCore pattern |
| Concurrency | asyncio | Built-in | Non-blocking signal processing |
| Logging | structlog | Latest | Structured logging for signal traces |
| Metrics | prometheus-client | Latest | Existing AgentCore metrics |
| State Storage | In-memory (dict) | N/A | Phase 1, Redis future for distributed |

### Key Technology Decisions

**Decision 1: In-Memory State (Phase 1)**

- **Rationale**: Simplicity and performance for initial implementation
- **Trade-off**: Single-instance limitation vs fast access (<2ms)
- **Future**: Redis for distributed state (Phase 2)

**Decision 2: Pydantic for Signal Validation**

- **Rationale**: Automatic validation, type safety, serialization
- **Alignment**: Consistent with AgentCore patterns
- **Benefits**: Runtime validation of signal ranges (0.0-1.0)

**Decision 3: Async Event Processing**

- **Rationale**: Non-blocking signal registration and score updates
- **Pattern**: Existing AgentCore async services
- **Benefits**: Handle 10K signals/sec without blocking

**Decision 4: Simple Linear Regression for Prediction**

- **Rationale**: Fast, interpretable, no ML dependencies
- **Trade-off**: Simple model vs complex time-series forecasting
- **Performance**: <1ms prediction latency

### Alignment with Existing System

**From AgentCore Tech Stack:**

- ✅ Python 3.12+ with modern typing (`list[str]`, `dict[str, float]`, `str | None`)
- ✅ FastAPI for service infrastructure
- ✅ Pydantic 2.0+ for data validation
- ✅ Async-first architecture (asyncio)
- ✅ Prometheus metrics (existing patterns)
- ✅ structlog for structured logging

**New Additions:**

- None - all technologies already in use by AgentCore

**Future Considerations:**

- Redis for distributed coordination state (Phase 2)
- TimescaleDB for signal history (if advanced analytics needed)

---

## 🏗️ Architecture Design

### System Context

**AgentCore Architecture (6 Layers):**

```
┌─────────────────────────────────────────────────────────────┐
│ Intelligence Layer (Future ACE)                             │
├─────────────────────────────────────────────────────────────┤
│ Experience Layer (Memory Service - Future)                  │
├─────────────────────────────────────────────────────────────┤
│ Enterprise Ops Layer (Monitoring, Metrics)                  │
├─────────────────────────────────────────────────────────────┤
│ Core Layer                                                  │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│  │ Task Mgmt    │  │ Message Router  │  │ Agent Mgmt   │   │
│  └──────┬───────┘  └────────┬────────┘  └──────┬───────┘   │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             │                               │
│                  ┌──────────▼──────────┐                    │
│                  │ Coordination Service│ ◄── THIS COMPONENT │
│                  │   (REP Signals)     │                    │
│                  └─────────────────────┘                    │
├─────────────────────────────────────────────────────────────┤
│ Infrastructure Layer (PostgreSQL, Redis)                    │
├─────────────────────────────────────────────────────────────┤
│ Runtime Layer (FastAPI, uvicorn)                            │
└─────────────────────────────────────────────────────────────┘
```

**Integration Points:**

- **MessageRouter**: Primary consumer for RIPPLE_COORDINATION strategy
- **AgentManager**: Agents report health signals during lifecycle
- **JSON-RPC Handler**: Expose coordination methods to agents
- **Prometheus**: Metrics for coordination effectiveness

### Component Architecture

**Architecture Pattern:** Service Layer with In-Memory State

```
┌────────────────────────────────────────────────────────────────┐
│                    CoordinationService                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Signal Registration & Validation                         │ │
│  │  - Validate signal format (Pydantic)                     │ │
│  │  - Normalize values to 0.0-1.0                           │ │
│  │  - Apply TTL and timestamps                              │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Agent Coordination State (In-Memory Dict)                │ │
│  │  - agent_states: dict[str, AgentCoordinationState]       │ │
│  │  - Signal history (last 100 per agent)                   │ │
│  │  - Active signals with TTL                               │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Score Aggregation & Computation                          │ │
│  │  - Individual scores (load, capacity, quality, etc.)     │ │
│  │  - Composite routing score (weighted avg)                │ │
│  │  - Temporal decay for aging signals                      │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Multi-Objective Optimization                             │ │
│  │  - Apply optimization weights                            │ │
│  │  - Select optimal agent from candidates                  │ │
│  │  - Log selection rationale                               │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Overload Prediction                                      │ │
│  │  - Load trend analysis (linear regression)               │ │
│  │  - Forecast future load                                  │ │
│  │  - Return overload probability                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Background Cleanup Task (async)                          │ │
│  │  - Remove expired signals every 5 minutes                │ │
│  │  - Delete agent states with no active signals            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Architecture Decisions

**Decision 1: In-Memory State for Phase 1**

- **Choice**: dict[str, AgentCoordinationState] in-memory
- **Rationale**: Fast access (<2ms), simple implementation, sufficient for single instance
- **Implementation**: CoordinationService maintains state dict with RLock for thread safety
- **Trade-offs**:
  - ✅ Pros: Fast, simple, no external dependencies
  - ❌ Cons: Single instance, state loss on restart (acceptable for signals with 60s TTL)
- **Future**: Migrate to Redis for distributed state (Phase 2)

**Decision 2: Async Signal Processing**

- **Choice**: Async methods for all signal operations
- **Rationale**: Non-blocking, aligns with AgentCore patterns, handles high throughput
- **Implementation**: `async def register_signal()`, `async def get_optimal_agent()`
- **Trade-offs**:
  - ✅ Pros: 10K signals/sec, non-blocking
  - ❌ Cons: Slightly more complex than sync

**Decision 3: Simple Linear Regression for Prediction**

- **Choice**: Basic linear regression on recent load signals
- **Rationale**: Fast (<1ms), interpretable, no ML dependencies
- **Implementation**: Last 10 load signals, predict load at forecast_seconds
- **Trade-offs**:
  - ✅ Pros: Fast, simple, transparent
  - ❌ Cons: Less accurate than ML models (acceptable for REP use case)

**Decision 4: Integration via MessageRouter**

- **Choice**: New RIPPLE_COORDINATION routing strategy in MessageRouter
- **Rationale**: Clean integration point, preserves existing strategies
- **Implementation**: Add strategy enum, implement `_ripple_coordination_select()`
- **Trade-offs**:
  - ✅ Pros: Non-invasive, optional, easy A/B testing
  - ❌ Cons: Requires MessageRouter changes (acceptable)

### Component Breakdown

**Core Components:**

**1. CoordinationService** (`src/agentcore/a2a_protocol/services/coordination_service.py`)

- **Purpose**: Central coordination state manager
- **Responsibilities**:
  - Register and validate sensitivity signals
  - Maintain agent coordination states
  - Compute routing scores
  - Select optimal agents
  - Predict overload
  - Cleanup expired signals
- **Interfaces**:
  - `register_signal(signal: SensitivitySignal) -> None`
  - `get_agent_state(agent_id: str) -> AgentCoordinationState | None`
  - `get_optimal_agent(candidates: list[str], weights: OptimizationWeights | None) -> str`
  - `predict_overload(agent_id: str, forecast_seconds: int) -> tuple[bool, float]`
  - `cleanup_expired_signals() -> int`
- **Dependencies**: Pydantic (validation), asyncio, structlog

**2. Models** (`src/agentcore/a2a_protocol/models/coordination.py`)

- **Purpose**: Data models for coordination
- **Models**:
  - `SignalType` (enum)
  - `SensitivitySignal` (Pydantic model)
  - `AgentCoordinationState` (Pydantic model)
  - `OptimizationWeights` (Pydantic model)
  - `CoordinationMetrics` (Pydantic model)
- **Validation**:
  - Signal value range (0.0-1.0)
  - TTL range (10-600 seconds)
  - Weights sum to 1.0

**3. JSON-RPC Methods** (`src/agentcore/a2a_protocol/services/coordination_jsonrpc.py`)

- **Purpose**: Expose coordination via JSON-RPC
- **Methods**:
  - `coordination.signal`: Register sensitivity signal
  - `coordination.state`: Get agent coordination state
  - `coordination.metrics`: Get coordination effectiveness metrics
  - `coordination.predict_overload`: Predict agent overload
- **Dependencies**: jsonrpc_handler, CoordinationService

**4. MessageRouter Integration** (`src/agentcore/a2a_protocol/services/message_router.py`)

- **Purpose**: RIPPLE_COORDINATION routing strategy
- **Changes**:
  - Add `RIPPLE_COORDINATION` to RoutingStrategy enum
  - Implement `_ripple_coordination_select()` method
  - Route through CoordinationService for optimal agent selection
- **Fallback**: Random selection if coordination unavailable

**5. Background Cleanup Task**

- **Purpose**: Periodic cleanup of expired signals
- **Implementation**: asyncio background task
- **Interval**: 5 minutes (configurable)
- **Actions**:
  - Recompute scores (auto-removes expired)
  - Delete agent states with no active signals
  - Log cleanup statistics
  - Update Prometheus metrics

### Data Flow & Boundaries

**Signal Registration Flow:**

```
Agent → JSON-RPC → coordination.signal
                        ↓
                  Validate signal (Pydantic)
                        ↓
                  CoordinationService.register_signal()
                        ↓
                  Update AgentCoordinationState
                        ↓
                  Add to signal history
                        ↓
                  Recompute routing score
                        ↓
                  Update Prometheus metrics
                        ↓
                  Return success
```

**Agent Selection Flow:**

```
MessageRouter → RIPPLE_COORDINATION strategy
                        ↓
                Get candidate agents (capability match)
                        ↓
                CoordinationService.get_optimal_agent(candidates)
                        ↓
                Retrieve coordination states
                        ↓
                Compute routing scores (multi-objective)
                        ↓
                Select agent with highest score
                        ↓
                Log selection rationale
                        ↓
                Return selected agent_id
                        ↓
                Route message to agent
```

**Component Boundaries:**

- **Public Interface**: JSON-RPC methods, get_coordination_service() global
- **Internal Implementation**: AgentCoordinationState dict, score computation
- **Cross-Component Contracts**:
  - MessageRouter expects `get_optimal_agent(list[str]) -> str`
  - Agents publish via `coordination.signal` JSON-RPC
  - JSON-RPC handler registers methods on import

---

## 🔧 Technical Specification

### Data Model

**Entities:**

1. **SensitivitySignal**
   - signal_id: UUID (auto-generated)
   - agent_id: str (required)
   - signal_type: SignalType enum (required)
   - value: float (0.0-1.0, required)
   - timestamp: datetime (auto-generated)
   - ttl_seconds: int (10-600, default 60)
   - confidence: float (0.0-1.0, default 1.0)
   - trace_id: str | None (optional, for distributed tracing)

2. **AgentCoordinationState**
   - agent_id: str (required)
   - signals: dict[SignalType, SensitivitySignal] (active signals only)
   - signal_history: list[SensitivitySignal] (last 100, for prediction)
   - load_score: float (0.0-1.0)
   - capacity_score: float (0.0-1.0)
   - quality_score: float (0.0-1.0)
   - cost_score: float (0.0-1.0)
   - availability_score: float (0.0-1.0)
   - routing_score: float (0.0-1.0, composite)
   - last_updated: datetime

3. **OptimizationWeights**
   - load: float (default 0.25)
   - capacity: float (default 0.25)
   - quality: float (default 0.20)
   - cost: float (default 0.15)
   - availability: float (default 0.15)
   - Constraint: Sum must equal 1.0

**Validation Rules:**

- Signal values: 0.0 ≤ value ≤ 1.0
- TTL: 10 ≤ ttl_seconds ≤ 600
- Confidence: 0.0 ≤ confidence ≤ 1.0
- Optimization weights: Sum = 1.0 (±0.001 tolerance)
- Signal history: Max 100 entries per agent

**Indexing Strategy:**

- In-memory dict keyed by agent_id (O(1) lookup)
- Signal history: Append-only list with size limit
- Future (Redis): Hash per agent with TTL

**State Management:**

- In-memory: dict[str, AgentCoordinationState]
- Thread safety: asyncio (single-threaded event loop)
- Persistence: None (signals ephemeral with 60s TTL)
- Future: Redis hash with automatic TTL expiry

### API Design

**Top 6 Critical Methods:**

**1. coordination.signal** (Register Sensitivity Signal)

```
POST /api/v1/jsonrpc

Request:
{
  "jsonrpc": "2.0",
  "method": "coordination.signal",
  "params": {
    "agent_id": "agent-123",
    "signal_type": "load",
    "value": 0.75,
    "ttl_seconds": 60,
    "confidence": 0.9
  },
  "id": "req-1"
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "registered": true,
    "signal_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_id": "agent-123",
    "routing_score": 0.42
  },
  "id": "req-1"
}

Errors:
- -32602: Invalid params (value out of range, invalid signal_type)
```

**2. coordination.state** (Get Agent Coordination State)

```
POST /api/v1/jsonrpc

Request:
{
  "jsonrpc": "2.0",
  "method": "coordination.state",
  "params": {
    "agent_id": "agent-123"
  },
  "id": "req-2"
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "agent_id": "agent-123",
    "load_score": 0.25,
    "capacity_score": 0.80,
    "quality_score": 0.90,
    "cost_score": 0.50,
    "availability_score": 1.0,
    "routing_score": 0.68,
    "signals": {
      "load": {
        "signal_type": "load",
        "value": 0.75,
        "timestamp": "2025-10-25T10:30:00Z",
        "ttl_seconds": 60
      }
    },
    "last_updated": "2025-10-25T10:30:00Z"
  },
  "id": "req-2"
}
```

**3. coordination.select** (Select Optimal Agent)

```
POST /api/v1/jsonrpc

Request:
{
  "jsonrpc": "2.0",
  "method": "coordination.select",
  "params": {
    "candidates": ["agent-1", "agent-2", "agent-3"],
    "weights": {
      "load": 0.3,
      "capacity": 0.3,
      "quality": 0.2,
      "cost": 0.1,
      "availability": 0.1
    }
  },
  "id": "req-3"
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "selected_agent": "agent-2",
    "score": 0.78,
    "all_scores": {
      "agent-1": 0.55,
      "agent-2": 0.78,
      "agent-3": 0.42
    },
    "rationale": "agent-2 selected: highest composite score (0.78)"
  },
  "id": "req-3"
}
```

**4. coordination.predict_overload** (Predict Agent Overload)

```
POST /api/v1/jsonrpc

Request:
{
  "jsonrpc": "2.0",
  "method": "coordination.predict_overload",
  "params": {
    "agent_id": "agent-123",
    "forecast_seconds": 60
  },
  "id": "req-4"
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "will_overload": true,
    "probability": 0.85,
    "current_load": 0.70,
    "predicted_load": 0.85,
    "forecast_seconds": 60,
    "warning": "Agent predicted to overload in 60s"
  },
  "id": "req-4"
}
```

**5. coordination.metrics** (Get Coordination Effectiveness Metrics)

```
POST /api/v1/jsonrpc

Request:
{
  "jsonrpc": "2.0",
  "method": "coordination.metrics",
  "params": {},
  "id": "req-5"
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "total_signals": 15423,
    "active_agents": 42,
    "signals_per_second": 125.3,
    "avg_routing_score": 0.65,
    "coordination_accuracy": 0.82,
    "load_balance_gini": 0.08,
    "overload_predictions": 3,
    "signal_freshness_pct": 97.5
  },
  "id": "req-5"
}
```

**6. coordination.cleanup** (Manual Cleanup Trigger)

```
POST /api/v1/jsonrpc

Request:
{
  "jsonrpc": "2.0",
  "method": "coordination.cleanup",
  "params": {},
  "id": "req-6"
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "signals_removed": 234,
    "agents_removed": 5,
    "duration_ms": 45
  },
  "id": "req-6"
}
```

### Error Handling

**Error Codes:**

- `-32600`: Invalid Request (malformed JSON-RPC)
- `-32601`: Method not found
- `-32602`: Invalid params (validation failed)
- `-32603`: Internal error (unexpected exception)

**Graceful Degradation:**

- Missing agent state → Use default score 0.5
- No signals → Default routing score 0.5
- Insufficient history for prediction → Return (False, 0.0)
- Coordination service unavailable → Fallback to random routing

### Security

**Input Validation:**

- All signals validated via Pydantic (type + range checks)
- agent_id: Non-empty string, max 256 chars
- value: 0.0 ≤ value ≤ 1.0
- ttl_seconds: 10 ≤ ttl ≤ 600

**Rate Limiting:**

- Not implemented in Phase 1 (in-memory state has implicit limit)
- Future: Redis-based rate limiting per agent

**Access Control:**

- JSON-RPC methods: No authentication (internal service)
- Future: JWT validation for agent identity

---

## 📋 Implementation Roadmap

### Phase 1: Foundation (Week 1, Days 1-5)

**Week 1, Day 1-2: Core Service Implementation**

**Goals:**
- CoordinationService skeleton with signal registration
- In-memory state management
- Basic signal validation

**Tasks:**
1. Create `src/agentcore/a2a_protocol/models/coordination.py`
   - Define SignalType enum
   - Implement SensitivitySignal model with validation
   - Implement AgentCoordinationState model
   - Implement OptimizationWeights model
2. Create `src/agentcore/a2a_protocol/services/coordination_service.py`
   - CoordinationService class with in-memory dict
   - `register_signal()` method with validation
   - `get_agent_state()` method
   - Global `get_coordination_service()` function
3. Add configuration to `config.py`
   - COORDINATION_ENABLE_REP: bool
   - COORDINATION_SIGNAL_TTL: int
   - Optimization weights

**Deliverables:**
- `models/coordination.py` with 4 Pydantic models
- `services/coordination_service.py` with basic registration
- `config.py` additions

**Week 1, Day 3: Score Aggregation**

**Goals:**
- Implement score computation logic
- Temporal decay for aging signals
- Composite routing score calculation

**Tasks:**
1. Implement `_compute_individual_scores()` method
   - Load score (inverted)
   - Capacity score
   - Quality score
   - Cost score (inverted)
   - Availability score
2. Implement `_compute_routing_score()` with weighted average
3. Implement `_apply_temporal_decay()` for aging signals
4. Update scores automatically on signal registration

**Deliverables:**
- Score computation methods in CoordinationService
- Temporal decay implementation
- Routing score calculation

**Week 1, Day 4: Signal History & TTL**

**Goals:**
- Signal history management
- TTL-based expiry
- Background cleanup preparation

**Tasks:**
1. Implement `add_signal_to_history()` with 100-entry limit
2. Implement `is_expired()` and `compute_decayed_value()` on SensitivitySignal
3. Implement `get_active_signals()` on AgentCoordinationState
4. Implement `cleanup_expired_signals()` method
5. Unit tests for TTL and expiry

**Deliverables:**
- Signal history management (100-entry circular buffer)
- TTL expiry logic
- Cleanup method

**Week 1, Day 5: Multi-Objective Optimization**

**Goals:**
- Optimal agent selection logic
- Configurable optimization weights
- Selection logging

**Tasks:**
1. Implement `get_optimal_agent(candidates, weights)` method
2. Apply multi-objective optimization formula
3. Sort agents by composite score
4. Log selection rationale with structlog
5. Handle agents with no signals (default 0.5)
6. Unit tests for optimization logic

**Deliverables:**
- `get_optimal_agent()` implementation
- Multi-objective optimization with configurable weights
- Selection logging

### Phase 2: Integration & Validation (Week 2, Days 6-10)

**Week 2, Day 6-7: MessageRouter Integration**

**Goals:**
- RIPPLE_COORDINATION routing strategy
- Integration with existing routing
- Fallback handling

**Tasks:**
1. Update `src/agentcore/a2a_protocol/services/message_router.py`
   - Add RIPPLE_COORDINATION to RoutingStrategy enum
   - Implement `_ripple_coordination_select()` method
   - Integrate with CoordinationService
   - Add fallback to random on errors
2. Update route() method to support new strategy
3. Integration tests with multiple agents

**Deliverables:**
- RIPPLE_COORDINATION strategy in MessageRouter
- Integration with CoordinationService
- Fallback handling

**Week 2, Day 8: JSON-RPC Methods**

**Goals:**
- Expose coordination via JSON-RPC
- Register methods with handler
- Error handling

**Tasks:**
1. Create `src/agentcore/a2a_protocol/services/coordination_jsonrpc.py`
   - Implement `coordination.signal` handler
   - Implement `coordination.state` handler
   - Implement `coordination.select` handler
   - Implement `coordination.predict_overload` handler
   - Implement `coordination.metrics` handler
2. Register methods with `@register_jsonrpc_method` decorator
3. Import module in `main.py` for auto-registration
4. Error handling with JSON-RPC error codes

**Deliverables:**
- `services/coordination_jsonrpc.py` with 5 JSON-RPC methods
- Auto-registration on import
- Error handling

**Week 2, Day 9: Overload Prediction**

**Goals:**
- Implement linear regression prediction
- Forecast future load
- Warning logging

**Tasks:**
1. Implement `predict_overload(agent_id, forecast_seconds)` method
2. Extract recent load signals from history
3. Apply simple linear regression
4. Predict future load at forecast time
5. Return (will_overload, probability)
6. Log warnings for predicted overloads
7. Unit tests for prediction accuracy

**Deliverables:**
- `predict_overload()` implementation
- Linear regression logic
- Warning logging

**Week 2, Day 10: Testing & Validation**

**Goals:**
- Unit test suite (90%+ coverage)
- Integration tests with multi-agent scenarios
- Performance benchmarks
- Effectiveness validation vs baseline

**Tasks:**
1. Unit tests for all CoordinationService methods
2. Unit tests for signal validation and expiry
3. Unit tests for score computation
4. Integration tests:
   - Multi-agent signal exchange
   - End-to-end routing with RIPPLE_COORDINATION
   - Signal cleanup and state management
5. Performance benchmarks:
   - Signal registration latency (target <5ms p95)
   - Optimal agent selection latency (target <10ms p95)
   - Load test with 1000 agents and 10K signals/sec
6. Effectiveness testing:
   - Compare RIPPLE_COORDINATION vs random routing
   - Measure load distribution evenness (Gini coefficient)
   - Validate 41-100% accuracy improvement claim
7. Documentation and examples

**Deliverables:**
- Unit test suite (90%+ coverage)
- Integration test suite
- Performance benchmarks
- Effectiveness validation report
- Documentation with examples

### Background Tasks (Ongoing)

**Cleanup Task:**

- Start on application startup (`app.on_event("startup")`)
- Run every 5 minutes (configurable)
- Cleanup expired signals and remove empty agent states
- Log cleanup statistics
- Update Prometheus metrics

**Prometheus Metrics:**

- `coordination_signals_total`: Total signals registered (counter)
- `coordination_signals_expired`: Expired signals removed (counter)
- `coordination_active_agents`: Active agents with signals (gauge)
- `coordination_routing_score_avg`: Average routing score (gauge)
- `coordination_selection_duration_seconds`: Agent selection latency (histogram)
- `coordination_signal_freshness_pct`: % signals <60s old (gauge)

### Timeline Summary

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| Week 1, Day 1-2 | 2 days | Core service | CoordinationService, models, config |
| Week 1, Day 3 | 1 day | Score aggregation | Individual scores, composite score |
| Week 1, Day 4 | 1 day | History & TTL | Signal history, expiry, cleanup |
| Week 1, Day 5 | 1 day | Optimization | Multi-objective agent selection |
| Week 2, Day 6-7 | 2 days | MessageRouter | RIPPLE_COORDINATION strategy |
| Week 2, Day 8 | 1 day | JSON-RPC | Coordination methods |
| Week 2, Day 9 | 1 day | Prediction | Overload prediction |
| Week 2, Day 10 | 1 day | Testing & Validation | Tests, benchmarks, docs |

**Total Duration:** 10 days (2 weeks)

**Critical Path:**
Models → CoordinationService → Score Aggregation → History/TTL → Optimization → MessageRouter → JSON-RPC → Testing

---

## 📊 Quality Assurance

### Testing Strategy

**Unit Testing (90%+ Coverage):**

1. **Signal Validation Tests** (`tests/unit/test_coordination_models.py`)
   - Valid signal ranges (0.0-1.0)
   - TTL ranges (10-600)
   - Signal expiry logic
   - Temporal decay calculation
   - Weight validation (sum to 1.0)

2. **Score Computation Tests** (`tests/unit/test_coordination_service.py`)
   - Individual score calculation (load, capacity, quality, cost, availability)
   - Load and cost inversion (high load = low score)
   - Composite routing score (weighted average)
   - Default score for agents without signals (0.5)
   - Temporal decay application

3. **Optimization Tests** (`tests/unit/test_multi_objective_optimization.py`)
   - Agent selection with various weights
   - Handling empty candidate list
   - Handling agents with no signals
   - Sorting by composite score
   - Edge cases (all agents equal score)

4. **Prediction Tests** (`tests/unit/test_overload_prediction.py`)
   - Linear regression on load signals
   - Future load prediction accuracy
   - Handling insufficient history (<3 signals)
   - Threshold detection (0.8 default)
   - Edge cases (flat load, negative slope)

5. **TTL and Cleanup Tests** (`tests/unit/test_signal_cleanup.py`)
   - Signal expiry after TTL
   - Cleanup of expired signals
   - Removal of empty agent states
   - Signal history size limit (100)

**Integration Testing:**

1. **Multi-Agent Signal Exchange** (`tests/integration/test_coordination_flow.py`)
   - Multiple agents register signals
   - Coordination state updated correctly
   - Routing scores computed
   - Agent selection based on real signals

2. **MessageRouter Integration** (`tests/integration/test_ripple_routing.py`)
   - RIPPLE_COORDINATION strategy selects optimal agent
   - Fallback to random on coordination failure
   - Integration with capability matching
   - End-to-end message routing

3. **JSON-RPC Methods** (`tests/integration/test_coordination_jsonrpc.py`)
   - Signal registration via JSON-RPC
   - State retrieval via JSON-RPC
   - Selection via JSON-RPC
   - Prediction via JSON-RPC
   - Error handling (invalid params)

4. **Overload Prediction Workflow** (`tests/integration/test_overload_detection.py`)
   - Agent reports increasing load signals
   - Prediction detects overload trend
   - Warning logged
   - MessageRouter avoids overloaded agent

**Performance Testing:**

1. **Latency Benchmarks** (`tests/performance/test_coordination_latency.py`)
   - Signal registration: <5ms (p95)
   - Routing score retrieval: <2ms (p95)
   - Optimal agent selection (100 candidates): <10ms (p95)
   - Cleanup task: <100ms

2. **Throughput Benchmarks** (`tests/performance/test_coordination_throughput.py`)
   - Handle 10,000 signals/second
   - Handle 1,000 concurrent agent states
   - Memory usage with 1,000 agents * 100 history = 100K signals

3. **Scalability Testing** (`tests/performance/test_coordination_scale.py`)
   - Linear scaling up to 1,000 agents
   - No degradation with full signal history (100 per agent)

**Effectiveness Testing (REP Validation):**

1. **Coordination Accuracy** (`tests/effectiveness/test_coordination_accuracy.py`)
   - Baseline: Random routing (50% optimal selection)
   - RIPPLE_COORDINATION: 71-100% optimal selection
   - Target: 41-100% improvement (REP paper benchmark)
   - Metric: % tasks routed to best agent (by actual outcome)

2. **Load Distribution** (`tests/effectiveness/test_load_balance.py`)
   - Baseline: Random routing (Gini coefficient ~0.3)
   - RIPPLE_COORDINATION: Gini coefficient <0.1
   - Target: 90%+ even distribution
   - Metric: Gini coefficient of load across agents

3. **Overload Prevention** (`tests/effectiveness/test_overload_prevention.py`)
   - Baseline: No prediction, agents overload
   - RIPPLE_COORDINATION: Predict and avoid
   - Target: 80%+ overload prediction accuracy
   - Metric: Precision and recall for overload events

4. **Signal Freshness** (`tests/effectiveness/test_signal_freshness.py`)
   - Target: 95%+ routing decisions use signals <60s old
   - Metric: % routing using fresh signals

### Code Quality Gates

**Pre-Commit Checks:**

- Type checking: mypy --strict
- Linting: ruff check (no errors)
- Formatting: ruff format --check
- Unit tests: pytest tests/unit/ (90%+ coverage)

**CI/CD Pipeline:**

1. **Build Stage**:
   - Install dependencies
   - Type checking (mypy)
   - Linting (ruff)

2. **Test Stage**:
   - Unit tests (90%+ coverage required)
   - Integration tests
   - Performance benchmarks (pass/fail thresholds)

3. **Validation Stage**:
   - Effectiveness testing (41%+ improvement vs baseline)
   - Load distribution (Gini <0.1)
   - Overload prediction (80%+ accuracy)

**Deployment Verification:**

- [ ] All unit tests pass (90%+ coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks within targets
- [ ] Effectiveness validation passes (41%+ improvement)
- [ ] MessageRouter integration functional
- [ ] JSON-RPC methods registered
- [ ] Prometheus metrics instrumented
- [ ] Documentation complete

---

## ⚠️ Risk Management

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **In-memory state loss on restart** | M | H | Signals have 60s TTL, transient state acceptable. Future: Redis persistence. |
| **Signal staleness (>60s)** | M | M | Temporal decay reduces old signal influence. Cleanup every 5min removes expired. |
| **Agents not reporting signals** | H | M | Default score 0.5 for agents without signals. Graceful degradation to random routing. |
| **Linear regression prediction inaccuracy** | M | M | Simple model acceptable for REP. Monitor prediction accuracy. Upgrade to ARIMA if needed. |
| **Coordination service single point of failure** | H | L | Fallback to random routing if unavailable. Future: Distributed state with Redis. |
| **Signal spam from malicious agents** | M | L | Signal history limited to 100 per agent. Future: Rate limiting. |
| **Load imbalance despite coordination** | M | M | Monitor Gini coefficient. Tune optimization weights if needed. |
| **REP effectiveness claims not met** | H | L | Effectiveness tests validate 41%+ improvement. A/B testing in production. |
| **Async concurrency bugs** | M | M | Thorough unit tests for race conditions. Use asyncio patterns correctly. |
| **Memory leak in signal history** | M | L | Size limit (100 per agent). Background cleanup removes old states. Monitor memory. |

---

## 📚 References & Traceability

### Source Documentation

**Specification:**
- `docs/specs/coordination-service/spec.md`
  - REP (Ripple Effect Protocol) coordination
  - Functional requirements (FR-1 through FR-6)
  - Multi-objective optimization
  - Preemptive overload prediction
  - Success metrics: 41-100% improvement

**Research Context:**
- `docs/research/modular-agent-architecture.md`
  - Modular coordination patterns
  - Message-passing between specialized modules
  - State management best practices

**System Context:**
- `docs/agentcore-architecture-and-development-plan.md`
  - 6-layer architecture
  - Core Layer integration
  - Service mesh pattern

**Project Guide:**
- `CLAUDE.md`
  - AgentCore patterns and conventions
  - Python 3.12+ typing standards
  - Async-first architecture

### Technology Evaluation

**Ripple Effect Protocol (REP):**
- Source: Research paper (referenced in spec)
- Key Finding: 41-100% improvement in coordination accuracy
- Approach: Sensitivity signals for agent selection
- Validation: Multi-agent benchmarks

**Multi-Objective Optimization:**
- Technique: Weighted average of individual scores
- Weights: Configurable per workload (load, capacity, quality, cost, availability)
- Default: Balanced (0.25, 0.25, 0.20, 0.15, 0.15)

**Overload Prediction:**
- Method: Simple linear regression on recent load signals
- Forecast: 60 seconds (configurable)
- Threshold: 0.8 load (80%)
- Accuracy Target: 80%+ (precision & recall)

### Related Components

**Dependencies:**
- MessageRouter: `src/agentcore/a2a_protocol/services/message_router.py`
  - RIPPLE_COORDINATION strategy integration
  - Fallback to random routing

- JSON-RPC Handler: `src/agentcore/a2a_protocol/services/jsonrpc_handler.py`
  - Method registration
  - Error handling

- AgentManager: `src/agentcore/a2a_protocol/services/agent_manager.py`
  - Agent lifecycle and health
  - Optional signal reporting integration

**Dependents:**
- None (new service, optional routing strategy)

---

**Plan Status:** ✅ Complete and Ready for Implementation

**Next Steps:**

1. Review plan with engineering team
2. Create tickets from plan (use `/sage.tasks coordination-service`)
3. Set up development environment
4. Begin Phase 1, Week 1, Day 1-2 (Core Service Implementation)
5. Run effectiveness tests after Week 2, Day 10 to validate REP claims

**Estimated Effort:** 10 days (2 weeks, 1 senior engineer full-time)

**Risk Level:** LOW-MEDIUM
- Technology proven (REP paper validation)
- Dependencies minimal (MessageRouter, JSON-RPC)
- Clear acceptance criteria
- Fallback strategy (random routing) ensures system stability
