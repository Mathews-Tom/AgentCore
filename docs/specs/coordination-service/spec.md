# Coordination Service Specification

**Component ID:** COORD
**Status:** Ready for Implementation
**Priority:** P1 (MEDIUM - Phase 3)
**Effort:** 2 weeks
**Owner:** Backend Team
**Source:** `.docs/specs/SPEC_COORDINATION_SERVICE.md`

---

## 1. Overview

### Purpose and Business Value

Implement Ripple Effect Protocol (REP)-inspired coordination for AgentCore to optimize multi-agent task distribution and load balancing through sensitivity signals. This service enables:

- Intelligent agent selection based on capacity, quality, and cost signals
- Preemptive load balancing before agent overload
- Dynamic capability-based routing with real-time awareness
- Failure prediction and graceful degradation
- 41-100% improvement in coordination accuracy (per REP paper benchmarks)

### Success Metrics

- **Coordination Accuracy**: 41-100% improvement vs baseline routing
- **Load Balance**: 90%+ even distribution across available agents
- **Prediction Accuracy**: 80%+ accuracy for overload prediction
- **Latency**: <10ms overhead for signal aggregation
- **Signal Freshness**: 95%+ of routing decisions use signals <60s old

### Target Users

- **MessageRouter**: Primary consumer for intelligent agent selection
- **Agents**: Signal producers reporting capacity, load, and quality
- **Operators**: Monitor coordination health and signal patterns
- **System**: Enables autonomous load balancing

---

## 2. Functional Requirements

### FR-1: Sensitivity Signal Management

**FR-1.1** The system SHALL accept sensitivity signals from agents (load, capacity, quality, cost, latency, availability)
**FR-1.2** The system SHALL validate and normalize signal values to 0.0-1.0 range
**FR-1.3** The system SHALL store signals with timestamps and TTL
**FR-1.4** The system SHALL apply temporal decay to aging signals
**FR-1.5** The system SHALL automatically expire signals after TTL

### FR-2: Signal Aggregation

**FR-2.1** The system SHALL maintain coordination state per agent
**FR-2.2** The system SHALL compute individual scores (load, capacity, quality, cost, availability)
**FR-2.3** The system SHALL compute composite routing score with configurable weights
**FR-2.4** The system SHALL update scores immediately upon signal receipt
**FR-2.5** The system SHALL provide batch retrieval of routing scores

### FR-3: Agent Selection Optimization

**FR-3.1** The system SHALL select optimal agent from candidates using multi-objective optimization
**FR-3.2** The system SHALL support custom optimization weight profiles
**FR-3.3** The system SHALL apply default weights (load: 0.25, capacity: 0.25, quality: 0.20, cost: 0.15, availability: 0.15)
**FR-3.4** The system SHALL handle agents with no signals (default score 0.5)
**FR-3.5** The system SHALL log selection rationale for debugging

### FR-4: Preemptive Overload Prediction

**FR-4.1** The system SHALL track signal history per agent (100 signals default)
**FR-4.2** The system SHALL analyze load trends using historical data
**FR-4.3** The system SHALL predict agent overload within forecast window (60s default)
**FR-4.4** The system SHALL return overload probability with prediction
**FR-4.5** The system SHALL warn operators of predicted overloads

### FR-5: MessageRouter Integration

**FR-5.1** The system SHALL provide RIPPLE_COORDINATION routing strategy
**FR-5.2** The system SHALL integrate with existing routing strategies seamlessly
**FR-5.3** The system SHALL track coordination routing statistics
**FR-5.4** The system SHALL support fallback to random routing on errors

### FR-6: Signal Cleanup

**FR-6.1** The system SHALL periodically cleanup expired signals (5min interval default)
**FR-6.2** The system SHALL remove agent states with no active signals
**FR-6.3** The system SHALL log cleanup operations for monitoring

### User Stories

**US-1**: As an agent, I want to broadcast my current load so that I don't receive tasks when overloaded.

**US-2**: As a message router, I want to select the optimal agent considering load, capacity, and quality so that tasks are distributed efficiently.

**US-3**: As an operator, I want to monitor coordination signals so that I can identify load patterns and bottlenecks.

**US-4**: As a system administrator, I want preemptive overload warnings so that I can scale resources proactively.

**US-5**: As a developer, I want configurable optimization weights so that routing can be tuned per workload.

### Business Rules

- **BR-1**: All signals must have TTL and expire automatically
- **BR-2**: Signal values must be normalized to 0.0-1.0 range
- **BR-3**: Agents without signals receive default routing score (0.5)
- **BR-4**: Load scores are inverted (high load = low score)
- **BR-5**: Routing scores computed as weighted average of individual scores

---

## 3. Non-Functional Requirements

### Performance

- **NFR-P1**: Signal registration SHALL complete in <5ms (p95)
- **NFR-P2**: Routing score retrieval SHALL complete in <2ms (p95)
- **NFR-P3**: Optimal agent selection SHALL complete in <10ms for 100 candidates (p95)
- **NFR-P4**: Signal cleanup SHALL complete in <100ms (background task)

### Scalability

- **NFR-S1**: SHALL support 1,000 agents with coordination states
- **NFR-S2**: SHALL handle 10,000 signals per second
- **NFR-S3**: SHALL maintain signal history (100 per agent) without memory issues
- **NFR-S4**: SHALL scale horizontally with shared state in Redis (future)

### Reliability

- **NFR-R1**: SHALL achieve 99.9% uptime for coordination operations
- **NFR-R2**: SHALL handle signal processing failures gracefully (log and continue)
- **NFR-R3**: SHALL provide fallback routing when coordination unavailable
- **NFR-R4**: SHALL recover from state corruption automatically

### Observability

- **NFR-O1**: SHALL emit Prometheus metrics for all operations
- **NFR-O2**: SHALL log signal registrations with structured logging
- **NFR-O3**: SHALL track coordination effectiveness vs baseline
- **NFR-O4**: SHALL expose agent coordination health status

---

## 4. Features & Flows

### Feature 1: Signal Registration (Priority: P0)

**Description**: Agents register sensitivity signals for coordination.

**Key Flow**:

1. Agent computes local state (load, capacity, quality)
2. Agent publishes signals via JSON-RPC or direct API call
3. CoordinationService validates signal format and TTL
4. AgentCoordinationState updated with new signal
5. Routing score recomputed immediately
6. Signal stored in history for trend analysis
7. Metrics updated (total signals, signal types)

**Input**: `SensitivitySignal(agent_id, signal_type, value, confidence, ttl_seconds)`
**Output**: Signal registered confirmation

### Feature 2: Optimal Agent Selection (Priority: P0)

**Description**: Select best agent from candidates using multi-objective optimization.

**Key Flow**:

1. MessageRouter provides list of candidate agents
2. CoordinationService retrieves coordination state for each
3. Scores computed per agent (or default 0.5)
4. Multi-objective optimization applies weighted scoring
5. Agents sorted by composite score (descending)
6. Top agent selected and returned
7. Selection logged with scores for analysis

**Input**: `candidate_agents` (list[str]), `optimization_weights` (optional dict)
**Output**: Selected agent_id (str)

### Feature 3: Overload Prediction (Priority: P1)

**Description**: Predict agent overload before it occurs.

**Key Flow**:

1. CoordinationService retrieves signal history for agent
2. Recent load signals extracted (last 10)
3. Load trend calculated (simple linear regression)
4. Future load predicted for forecast window
5. Overload threshold checked (0.8 default)
6. Prediction returned with probability
7. Warning logged if overload predicted

**Input**: `agent_id` (str), `forecast_seconds` (int, default 60)
**Output**: `(will_overload: bool, probability: float)`

### Feature 4: RIPPLE_COORDINATION Routing (Priority: P0)

**Description**: New routing strategy in MessageRouter using coordination signals.

**Key Flow**:

1. MessageRouter invoked with RIPPLE_COORDINATION strategy
2. Capability matching identifies candidates
3. `_ripple_coordination_select(candidates)` called
4. CoordinationService selects optimal agent
5. Selection logged and metrics updated
6. Agent returned to caller

**Input**: Message envelope, routing strategy
**Output**: Selected agent_id

### Feature 5: Signal Cleanup (Priority: P1)

**Description**: Automatic cleanup of expired signals.

**Background Task**:

1. Periodic task runs every 5 minutes
2. Iterate all agent coordination states
3. Recompute scores (automatically removes expired)
4. Remove agent states with no active signals
5. Log cleanup statistics
6. Update metrics

**Trigger**: Scheduled background task
**Output**: Cleanup count logged

---

## 5. Acceptance Criteria

### Definition of Done

- [ ] CoordinationService implemented with signal aggregation
- [ ] SensitivitySignal models defined and validated
- [ ] RIPPLE_COORDINATION routing strategy added to MessageRouter
- [ ] MessageRouter integration completed and tested
- [ ] Signal expiry and cleanup mechanisms operational
- [ ] Multi-objective optimization logic functional
- [ ] Preemptive overload prediction working
- [ ] JSON-RPC methods registered (coordination.signal, coordination.state, coordination.metrics)
- [ ] Unit tests achieving 90%+ coverage
- [ ] Integration tests with multiple agents
- [ ] Prometheus metrics instrumented
- [ ] Documentation complete with examples
- [ ] Performance benchmarks vs baseline routing completed

### Validation Approach

**Unit Testing**:

- Test signal registration and validation
- Test score computation and aggregation
- Test optimal agent selection logic
- Test signal expiry and decay
- Test overload prediction algorithm

**Integration Testing**:

- Multi-agent signal exchange
- End-to-end routing with coordination
- Signal cleanup and state management
- Overload prediction accuracy

**Performance Testing**:

- Benchmark signal registration latency
- Benchmark optimal agent selection latency
- Load test with 1000 agents and 10K signals/sec
- Measure coordination overhead

**Effectiveness Testing**:

- Compare coordination routing vs random routing
- Measure load distribution evenness
- Validate 41-100% accuracy improvement claim

---

## 6. Dependencies

### Technical Stack

- **Core**: Python 3.12+, FastAPI, Pydantic, asyncio
- **Data Structures**: collections.defaultdict for signal history
- **Logging**: structlog
- **Metrics**: prometheus-client

### Related Components

- **MessageRouter**: Primary consumer of coordination service
- **EventManager**: Can broadcast signals via SSE/WebSocket (optional)
- **AgentManager**: Agents register signals as part of lifecycle
- **JSON-RPC Handler**: Coordination methods exposed via RPC

### Technical Assumptions

- Agents can compute and report their own load/capacity/quality
- Signal updates occur at reasonable frequency (every 10-30 seconds)
- Network latency for signal transmission <50ms
- In-memory coordination state sufficient (no distributed state initially)

---

## 7. Implementation Notes

### Component Structure

```
src/agentcore/a2a_protocol/
├── models/
│   └── coordination.py            # SensitivitySignal, AgentCoordinationState, CoordinationMetrics
├── services/
│   ├── coordination_service.py    # CoordinationService implementation
│   ├── coordination_jsonrpc.py    # JSON-RPC methods
│   └── message_router.py          # RIPPLE_COORDINATION strategy integration
```

### Configuration

```python
# config.py additions
COORDINATION_ENABLE_REP: bool = True
COORDINATION_SIGNAL_TTL: int = 60  # seconds
COORDINATION_MAX_HISTORY_SIZE: int = 100
COORDINATION_CLEANUP_INTERVAL: int = 300  # 5 minutes

# Routing Optimization Weights
ROUTING_WEIGHT_LOAD: float = 0.25
ROUTING_WEIGHT_CAPACITY: float = 0.25
ROUTING_WEIGHT_QUALITY: float = 0.20
ROUTING_WEIGHT_COST: float = 0.15
ROUTING_WEIGHT_AVAILABILITY: float = 0.15
```

### Data Models

**SensitivitySignal**:

- signal_id: UUID
- agent_id: str
- signal_type: SignalType enum
- value: float (0.0-1.0)
- timestamp: datetime
- ttl_seconds: int
- confidence: float
- trace_id: optional str

**AgentCoordinationState**:

- agent_id: str
- signals: dict[SignalType, SensitivitySignal]
- load_score: float
- capacity_score: float
- quality_score: float
- cost_score: float
- availability_score: float
- routing_score: float (composite)
- last_updated: datetime

### Timeline

**Week 1: Core Implementation**

- CoordinationService implementation (Days 1-2)
- Signal models and aggregation logic (Day 3)
- Unit tests (Day 4)
- Configuration and settings (Day 5)

**Week 2: Integration and Validation**

- MessageRouter integration (Days 6-7)
- JSON-RPC methods (Day 8)
- Integration tests (Day 9)
- Performance benchmarks and documentation (Day 10)

---

## 8. References

- Source specification: `.docs/specs/SPEC_COORDINATION_SERVICE.md`
- Ripple Effect Protocol: Research paper on sensitivity-based coordination
- AgentCore MessageRouter: `src/agentcore/a2a_protocol/services/message_router.py`
