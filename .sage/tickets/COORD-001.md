# COORD-001: Coordination Service with REP Implementation

**State:** UNPROCESSED
**Priority:** P1
**Type:** Epic
**Component:** coordination-service
**Effort:** 10 days (2 weeks)
**Phase:** 3

## Description

Implement Ripple Effect Protocol (REP)-inspired coordination for AgentCore to optimize multi-agent task distribution and load balancing through sensitivity signals. Enables intelligent agent selection based on real-time capacity, quality, and cost signals.

### Key Features

- **Sensitivity Signal Management**: Accept and normalize agent signals (load, capacity, quality, cost, latency, availability)
- **Signal Aggregation**: Maintain real-time coordination state per agent with computed routing scores
- **Multi-Objective Optimization**: Select optimal agents using weighted scoring (load, capacity, quality, cost, availability)
- **Preemptive Overload Prediction**: Predict agent overload before it occurs using trend analysis
- **RIPPLE_COORDINATION Routing**: New routing strategy in MessageRouter
- **Signal Cleanup**: Automatic expiry and cleanup of stale signals

### Business Value

- 41-100% improvement in coordination accuracy (per REP paper benchmarks)
- Preemptive load balancing before agent overload
- Quality-aware and cost-optimized agent selection
- Failure prediction and graceful degradation

## Acceptance Criteria

- [ ] CoordinationService implemented with signal aggregation
- [ ] SensitivitySignal models defined and validated
- [ ] RIPPLE_COORDINATION routing strategy added to MessageRouter
- [ ] MessageRouter integration completed and tested
- [ ] Signal expiry and cleanup mechanisms operational
- [ ] Multi-objective optimization logic functional
- [ ] Preemptive overload prediction working (80%+ accuracy)
- [ ] JSON-RPC methods registered (coordination.signal, coordination.state, coordination.metrics)
- [ ] Unit tests achieving 90%+ coverage
- [ ] Integration tests with multiple agents
- [ ] Prometheus metrics instrumented
- [ ] Documentation complete with examples
- [ ] Performance benchmarks vs baseline routing completed
- [ ] 41-100% coordination accuracy improvement validated

## Dependencies

### Component Dependencies
- **MessageRouter** (existing): Primary consumer of coordination service
  - Relationship: Integration point for RIPPLE_COORDINATION routing strategy
  - Impact: New routing strategy added to existing RoutingStrategy enum
- **EventManager** (existing, optional): Can broadcast signals via SSE/WebSocket
  - Relationship: Optional signal distribution mechanism
  - Impact: None (optional integration)
- **AgentManager** (existing): Agents register signals as part of lifecycle
  - Relationship: Agents use coordination.signal JSON-RPC method
  - Impact: None (optional agent participation)
- **JSON-RPC Handler** (existing): For method registration
  - Relationship: Required for exposing coordination methods
  - Impact: None (standard pattern)

### External Dependencies
- None (all dependencies internal to AgentCore)

### Technical Assumptions
- Agents can compute and report their own load/capacity/quality metrics
- Signal updates occur at reasonable frequency (every 10-30 seconds)
- Network latency for signal transmission <50ms
- In-memory coordination state sufficient (no distributed state Phase 1)
- Future: Redis for distributed state (Phase 2)

## Context

**Specification:** `docs/specs/coordination-service/spec.md`
**Source Research:** `.docs/specs/SPEC_COORDINATION_SERVICE.md`

### Component Structure

```
src/agentcore/a2a_protocol/
├── models/
│   └── coordination.py            # SensitivitySignal, AgentCoordinationState, CoordinationMetrics
├── services/
│   ├── coordination_service.py    # CoordinationService implementation
│   ├── coordination_jsonrpc.py    # JSON-RPC methods
│   └── message_router.py          # RIPPLE_COORDINATION strategy integration (modify existing)
```

### Configuration Required

```bash
# Coordination Service
COORDINATION_ENABLE_REP=true
COORDINATION_SIGNAL_TTL=60                  # seconds
COORDINATION_MAX_HISTORY_SIZE=100
COORDINATION_CLEANUP_INTERVAL=300           # 5 minutes

# Routing Optimization Weights
ROUTING_WEIGHT_LOAD=0.25
ROUTING_WEIGHT_CAPACITY=0.25
ROUTING_WEIGHT_QUALITY=0.20
ROUTING_WEIGHT_COST=0.15
ROUTING_WEIGHT_AVAILABILITY=0.15
```

### Data Models

**SensitivitySignal**:
- signal_id: UUID
- agent_id: str
- signal_type: SignalType enum (load, capacity, quality, cost, latency, availability)
- value: float (0.0-1.0 normalized)
- timestamp: datetime
- ttl_seconds: int
- confidence: float
- trace_id: optional str

**AgentCoordinationState**:
- agent_id: str
- signals: dict[SignalType, SensitivitySignal]
- Individual scores: load_score, capacity_score, quality_score, cost_score, availability_score
- routing_score: float (weighted composite)
- last_updated: datetime

## Timeline

**Week 1: Core Implementation** (5 days)
- CoordinationService implementation
- Signal models and aggregation logic
- Unit tests
- Configuration and settings

**Week 2: Integration and Validation** (5 days)
- MessageRouter integration
- JSON-RPC methods
- Integration tests
- Performance benchmarks and documentation

## Progress

**Status:** Ready for planning with `/sage.plan`
**Depends On:** None (can proceed independently)

**Next Steps:**
1. Run `/sage.plan` to generate detailed implementation plan
2. Run `/sage.tasks` to break down into SMART tasks
3. Execute implementation via `/sage.implement`

## Architecture

**Pattern:** Service Layer with In-Memory State Management

**Core Components:**
1. **CoordinationService**: Central state manager with signal aggregation
2. **SensitivitySignal Models**: Pydantic models for validation
3. **Multi-Objective Optimization**: Weighted scoring for agent selection
4. **Linear Regression Prediction**: Simple overload forecasting
5. **Background Cleanup Task**: Periodic signal expiry (5min interval)

**Integration Pattern:**
- MessageRouter: New RIPPLE_COORDINATION routing strategy
- JSON-RPC: Expose coordination.* methods
- Async: Non-blocking signal processing

**State Management:**
- Phase 1: In-memory dict[str, AgentCoordinationState]
- Phase 2 (Future): Redis for distributed state

**Key Design Decisions:**
- In-memory state for fast access (<2ms retrieval)
- Simple linear regression (no ML dependencies)
- Async-first for 10K signals/sec throughput
- Graceful degradation (fallback to random routing)

## Technology Stack

- **Runtime**: Python 3.12+ (asyncio, modern type hints)
- **Validation**: Pydantic 2.0+ (signal range validation)
- **Logging**: structlog (structured signal traces)
- **Metrics**: prometheus-client (coordination effectiveness)
- **State**: In-memory dict (Phase 1), Redis (Phase 2)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| In-memory state loss on restart | M | Signals have 60s TTL, transient state acceptable |
| Signal staleness (>60s) | M | Temporal decay, cleanup every 5min |
| Agents not reporting signals | H | Default score 0.5, fallback to random routing |
| Prediction inaccuracy | M | Monitor accuracy, upgrade to ARIMA if needed |
| Coordination service failure | H | Fallback to random routing in MessageRouter |

## Notes

Phase 3 component that enhances existing MessageRouter with intelligent coordination. Can be implemented independently but provides most value after LLM-001 and memory service are operational. Priority P1 as enhancement rather than foundational requirement.

**Performance Targets**:
- Signal registration: <5ms (p95)
- Routing score retrieval: <2ms (p95)
- Optimal agent selection: <10ms for 100 candidates (p95)
- Support 1,000 agents with coordination states
- Handle 10,000 signals per second

**Plan Reference**: `docs/specs/coordination-service/plan.md`
