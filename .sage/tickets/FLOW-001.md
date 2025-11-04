# FLOW-001: Flow-Based Optimization Engine Implementation

**State:** UNPROCESSED
**Priority:** P0
**Type:** Epic
**Component:** Flow-Based Optimization (FLOW)
**Estimated Effort:** 8 weeks
**Created:** 2025-10-15

---

## Description

Implement the Flow-Based Optimization Engine to enable agent performance improvement through reinforcement learning. This epic covers the complete GRPO (Group Refined Policy Optimization) training infrastructure, including trajectory collection, reward computation, policy updates, and training job management.

**Business Value:**
- **+15-25%** task success rate improvement for trained agents
- **20-40x ROI** within 3 months through reduced support costs and improved agent efficiency
- Automated agent improvement reduces manual prompt engineering overhead
- Competitive differentiation through production-grade agent training capabilities

**Technical Scope:**
- Trajectory collection system with parallel execution (8 concurrent trajectories)
- Reward computation with outcome-based and shaped reward functions
- GRPO trainer implementing policy gradient updates
- Training job management API with status tracking and budget enforcement
- Checkpoint management for training continuity
- Evaluation framework for measuring performance improvements
- Multi-step credit assignment for finer-grained learning

---

## Acceptance Criteria

### Functional Requirements

- ✅ **FR-1: Trajectory Collection**
  - Generate 8 parallel trajectories for a given query-agent pair
  - Capture complete execution state (states, actions, results) at each step
  - Handle failures gracefully without stopping collection
  - Complete within 2x baseline execution time

- ✅ **FR-2: Reward Computation**
  - Compute outcome-based rewards (task success/failure)
  - Apply shaped rewards (tool usage, verification, length penalty)
  - Normalize rewards using group statistics (mean, std)
  - Support custom reward functions per agent type

- ✅ **FR-3: Policy Gradient Updates**
  - Calculate policy gradients: `loss = -log_prob * advantage`
  - Update policies for trajectories with positive advantage
  - Track training loss and convergence metrics
  - Implement gradient clipping for stability

- ✅ **FR-4: Training Job Management**
  - Create jobs via `training.start_grpo` JSON-RPC API
  - Track job status (queued, running, completed, failed, cancelled)
  - Enforce token budget limits (default: $100/month)
  - Provide real-time status via `training.get_status` API
  - Support job cancellation and pause/resume

- ✅ **FR-5: Checkpoint Management**
  - Save checkpoints every N iterations (default: 10)
  - Store policy parameters, optimizer state, metrics
  - Enable resume from latest checkpoint after interruption
  - Retain best 5 checkpoints, cleanup older ones

- ✅ **FR-6: Evaluation Framework**
  - Evaluate on held-out queries (20% of training data)
  - Compute metrics: success_rate, avg_reward, tool_accuracy
  - Compare trained agent vs baseline
  - Run evaluation every 10 iterations

### Performance Requirements

- ✅ Trajectory generation latency: <2x baseline execution time
- ✅ Parallel trajectory throughput: 8 trajectories in <30s
- ✅ Policy update latency: <5s per batch
- ✅ Training job throughput: 100+ concurrent jobs
- ✅ Database write performance: >100 trajectory writes/sec
- ✅ API response time: `training.get_status` <200ms (p95)

### Success Metrics

- ✅ **Task Success Rate**: +15% improvement over baseline (validated via A/B testing)
- ✅ **Sample Efficiency**: Convergence with <10,000 training trajectories
- ✅ **Training Stability**: Monotonic improvement in validation metrics
- ✅ **Cost Efficiency**: Training cost <$25 per 1000 queries
- ✅ **ROI**: 20-40x return within 3 months

---

## Dependencies

### Technical Dependencies

- **Agent Runtime** (`src/agentcore/agent_runtime/`) - Executes agents to generate trajectories
- **Task Manager** (`src/agentcore/a2a_protocol/services/task_manager.py`) - Tracks task execution
- **Database Layer** (`src/agentcore/a2a_protocol/database/`) - PostgreSQL persistence
- **JSON-RPC Handler** (`src/agentcore/a2a_protocol/services/jsonrpc_handler.py`) - API endpoints
- **Portkey/LLM API** - Agent inference and policy updates
- **Redis** - Job queue and caching

### Component Dependencies

- **Modular Agent Architecture** (AGENT-001) - Provides trajectory data from Planner→Executor→Verifier→Generator
- **ACE Integration** (ACE-001) - Future dual optimization with context evolution
- **Bounded Context Reasoning** (BCR-001) - Context-aware trajectory generation
- **Memory System** (MEM-001) - Trajectory data informs memory evolution

### Potential Conflicts

- **DSPy Optimization**: May conflict with GRPO policy updates
  - **Mitigation**: Coordinate optimization schedules, use feature flags
- **Resource Contention**: Training consumes LLM tokens competing with production
  - **Mitigation**: Token budget limits, priority queuing, separate endpoints

---

## Context

### Specifications
- **Primary Spec**: `docs/specs/flow-based-optimization/spec.md`
- **Implementation Plan**: `docs/specs/flow-based-optimization/plan.md` (PRP format)
- **Research Source**: `docs/research/flow-based-optimization.md`

### Related Research
- `docs/research/modular-agent-architecture.md` - Integration with module execution flows
- `docs/research/ace-integration-analysis.md` - Dual optimization strategies

### Architecture Integration
- **Location**: `src/agentcore/training/` (new module)
- **Database**: New tables: `training_jobs`, `trajectories`, `policy_checkpoints`
- **API**: JSON-RPC methods: `training.start_grpo`, `training.get_status`, `training.evaluate`, `training.export_trajectories`

---

## Architecture

### System Layer
**Intelligence Layer** (alongside DSPy Optimization)
- Training infrastructure sits in Intelligence Layer of 6-layer AgentCore architecture
- Integrates with Agent Runtime Layer (trajectory execution)
- Uses Enhanced Core Layer (Task Management, JSON-RPC API)
- Depends on Infrastructure Layer (PostgreSQL, Redis, Portkey AI, FastAPI)

### Architecture Pattern
**Background Worker + Job Queue (Asynchronous Task Processing)**

**Components:**
1. **Training API Handler** - JSON-RPC endpoints (FastAPI)
2. **Training Job Manager** - Job lifecycle management
3. **Training Worker Pool** - Background execution (Redis queue)
4. **Trajectory Collector** - Parallel agent execution (8 concurrent)
5. **GRPO Trainer** - Policy gradient algorithm
6. **Reward Engine** - Outcome-based + shaped rewards
7. **Policy Updater** - Prompt/context evolution
8. **Evaluation Framework** - Held-out query testing
9. **Checkpoint Manager** - Save/restore with versioning
10. **Budget Enforcer** - Cost tracking and limits

**Data Flow:**
1. User → Training API → Job Manager → Redis Queue
2. Worker → Trajectory Collector → Agent Runtime → Portkey AI
3. Worker → GRPO Trainer → Policy Updater → Agent Update
4. Worker → Checkpoint Manager → PostgreSQL
5. Worker → Evaluation Framework → Metrics

**Key Decisions:**
- Training isolation: Separate workers, not inline with agent execution
- Trajectory collection: Wrapper pattern around existing agent runtime
- Policy updates: Prompt-based (Phase 1-2), neural networks (Phase 3)
- Checkpoint storage: Hybrid (PostgreSQL metadata + S3 for large weights)
- Budget enforcement: Pre-flight + real-time monitoring via Portkey

---

## Technology Stack

**Runtime:** Python 3.12+ with asyncio (existing)
- **Rationale:** Native async for parallel trajectory generation (8 concurrent target)

**Framework:** FastAPI 0.104+ (existing)
- **Rationale:** JSON-RPC pattern consistency with A2A protocol layer

**Database:** PostgreSQL 15+ with asyncpg (existing)
- **Rationale:** JSONB for flexible trajectory storage; handles 100GB-1TB growth

**Job Queue:** Redis 7.0+ (existing)
- **Rationale:** Distributed job coordination; lighter than Celery

**LLM Gateway:** Portkey AI 1.15+ (existing)
- **Rationale:** Built-in cost tracking enables budget enforcement

**Optimization:** NumPy + custom policy gradient
- **Rationale:** Start simple; defer PyTorch to Phase 3 if needed

**Testing:** pytest 8.4+ with pytest-asyncio 1.2+ (existing)
- **Rationale:** 90% coverage requirement; async test support

**Monitoring:** Prometheus + structlog (existing)
- **Rationale:** Metrics export; structured logging for training events

**New Dependencies:**
- NumPy (reward normalization, advantage computation)
- Optional (Phase 3): PyTorch (neural policy updates)

---

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-4)
**Story Tickets**: FLOW-002 through FLOW-007

- FLOW-002: Trajectory collection system
- FLOW-003: Reward computation engine
- FLOW-004: GRPO trainer implementation
- FLOW-005: Database schema and models
- FLOW-006: Training job API endpoints
- FLOW-007: Integration with agent runtime

### Phase 2: Evaluation & Monitoring (Weeks 5-6)
**Story Tickets**: FLOW-008 through FLOW-012

- FLOW-008: Evaluation framework
- FLOW-009: Budget tracking and enforcement
- FLOW-010: Checkpoint management system
- FLOW-011: Metrics export and monitoring
- FLOW-012: Data export API

### Phase 3: Advanced Features (Weeks 7-8)
**Story Tickets**: FLOW-013 through FLOW-016

- FLOW-013: Multi-step credit assignment
- FLOW-014: Training job scheduling and queuing
- FLOW-015: Advanced reward shaping
- FLOW-016: Documentation and user guides

---

## Progress

**Current State**: UNPROCESSED - Ready for planning phase

**Next Steps**:
1. Run `/sage.plan FLOW-001` to generate implementation plan with architecture details
2. Run `/sage.tasks FLOW-001` to break down into story tickets (FLOW-002+)
3. Begin implementation with `/sage.implement FLOW-002` (first story ticket)

**Notes**:
- Generated from `/sage.specify flow-based-optimization` command
- Linked to comprehensive specification document
- Ready for detailed implementation planning
- Estimated 8 weeks total effort (2 engineers)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Training cost overruns | High | Medium | Strict budget enforcement, cost alerts, cheaper models |
| Performance degradation | Medium | High | Separate training/production, A/B testing, rollback plan |
| Database scaling issues | Medium | Medium | Archival strategy, sharding, monitor growth |
| Training convergence failure | Medium | Medium | Hyperparameter tuning, early stopping, validation monitoring |
| Integration complexity | High | Medium | Phased rollout, feature flags, comprehensive testing |

---

## Success Criteria Summary

This epic is complete when:
1. ✅ All Phase 1-3 story tickets marked COMPLETE
2. ✅ Integration tests pass (end-to-end training job with real agent)
3. ✅ Performance tests pass (100 concurrent jobs, <2x latency overhead)
4. ✅ Acceptance test shows +15% task success rate improvement
5. ✅ Token costs validated at <$25 per 1000 queries
6. ✅ Documentation published and reviewed
7. ✅ Production deployment completed with monitoring

**Target Completion Date**: Week 8 from start
**Stakeholders**: Platform Team, Agent Developers, Product Management

## Implementation Started
**Started:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Status:** IN_PROGRESS
**Type:** Epic Ticket (Parent of FLOW-002 through FLOW-020)

### Epic Progress
- **Phase 1 (Core Infrastructure)**: ✅ 100% Complete (FLOW-002 through FLOW-007)
- **Phase 2 (Evaluation & Monitoring)**: ✅ 100% Complete (FLOW-008 through FLOW-012)
- **Phase 3 (Advanced Features)**: ⏳ 60% Complete (FLOW-013 through FLOW-020)
  - ✅ FLOW-013: Multi-Step Credit Assignment - COMPLETED
  - ✅ FLOW-014: Data Export API - COMPLETED
  - ✅ FLOW-015: Multi-Step Credit Assignment - COMPLETED
  - 🔄 FLOW-016: Training Job Scheduling - IMPLEMENTED (needs state update)
  - ❌ FLOW-017: Advanced Reward Shaping - UNPROCESSED
  - ❌ FLOW-018: Documentation & Guides - UNPROCESSED
  - ❌ FLOW-019: Integration Tests Phase 2 - UNPROCESSED
  - ❌ FLOW-020: Performance & Load Testing - UNPROCESSED

**Current Status**: 14/19 child tickets completed (73%)
**Remaining Work**: 5 child tickets to complete before marking Epic COMPLETED

### Implementation Notes
- FLOW-016 was implemented in commits 48b3892 and ac8aeb4 but ticket state not updated
- Need to process remaining child tickets: FLOW-016, FLOW-017, FLOW-018, FLOW-019, FLOW-020
- Epic will be marked COMPLETED when all child tickets reach COMPLETED state
