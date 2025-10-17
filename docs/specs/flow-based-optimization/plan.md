# Flow-Based Optimization Engine Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-15
**Specification:** `docs/specs/flow-based-optimization/spec.md`
**Research Source:** `docs/research/flow-based-optimization.md`
**Component ID:** FLOW
**Epic Ticket:** `.sage/tickets/FLOW-001.md`

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → This Plan**

1. **Research Analysis:** `docs/research/flow-based-optimization.md`
   - GRPO (Group Refined Policy Optimization) methodology
   - Trajectory-based agent training approach
   - Performance targets: +15-25% task success rate, +20% tool usage accuracy
   - Cost analysis: $25 per 1000 training queries
   - Implementation phases and technical patterns

2. **Formal Specification:** `docs/specs/flow-based-optimization/spec.md`
   - 8 functional requirements (FR-1 through FR-8)
   - Non-functional requirements (performance, security, scalability)
   - User stories and acceptance criteria
   - Dependencies and integration points
   - Success metrics and validation approach

3. **This Implementation Plan:**
   - Complete architecture design with technology decisions
   - Component breakdown and interfaces
   - Development roadmap with phases
   - Code patterns and implementation guidance

### Related Documentation

**System Context:**

- **Architecture:** `docs/agentcore-architecture-and-development-plan.md` - 6-layer system architecture
- **Agent Runtime:** `docs/breakdown/agent-runtime.md` - Multi-philosophy agent execution
- **Tech Stack:** `pyproject.toml` - Python 3.12+, FastAPI, PostgreSQL, Redis, Portkey AI

**Research Context:**

- `docs/research/modular-agent-architecture.md` - Planner→Executor→Verifier→Generator pattern
- `docs/research/ace-integration-analysis.md` - Context evolution integration (future)

**Related Specifications:**

- Agent Runtime: Multi-philosophy execution infrastructure
- A2A Protocol: JSON-RPC API patterns
- Database Layer: PostgreSQL persistence patterns

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable continuous agent performance improvement through reinforcement learning on real execution trajectories, reducing manual prompt engineering overhead and delivering measurable quality improvements.

**Value Proposition:**

- **Performance Gains:** +15-25% task success rate, +20% tool usage accuracy, +10-15% on complex multi-step tasks
- **Cost Reduction:** Automated improvement reduces manual tuning, 2-3x faster convergence than standard RL
- **Competitive Advantage:** Production-grade agent training differentiates AgentCore from competitors
- **ROI:** 20-40x return within 3 months through saved engineering time and improved agent efficiency

**Target Users:**

- **Platform Operators:** Manage training infrastructure, monitor agent improvement, control budgets
- **Agent Developers:** Train custom agents for specific workflows, export trajectory data for debugging
- **System Administrators:** Configure resource limits, storage policies, cost quotas
- **Agents (Automated):** Self-improvement via continuous learning from execution feedback

### Technical Approach

**Architecture Pattern:** Background Worker + Job Queue

- Training jobs run asynchronously in dedicated worker processes
- Redis queue for job scheduling and coordination
- JSON-RPC API for job management and status queries
- Trajectory collection wraps existing agent runtime execution

**Technology Stack:** Python 3.12+ | FastAPI | PostgreSQL | Redis | Portkey AI

- **Runtime:** Python asyncio for parallel trajectory generation (8 concurrent trajectories)
- **Framework:** FastAPI JSON-RPC endpoints (consistent with A2A protocol layer)
- **Database:** PostgreSQL with JSONB for flexible trajectory storage
- **Job Queue:** Redis for distributed job coordination and progress tracking
- **LLM Gateway:** Portkey AI for unified provider access and cost tracking

**Implementation Strategy:** 3-Phase Incremental Rollout

- **Phase 1 (Weeks 1-4):** Core training infrastructure (trajectory collection, GRPO trainer, job API)
- **Phase 2 (Weeks 5-6):** Evaluation, monitoring, budget enforcement, checkpoints
- **Phase 3 (Weeks 7-8):** Advanced features (multi-step credit assignment, scheduling, documentation)

### Key Success Metrics

**Service Level Objectives (SLOs):**

- **Availability:** 99.5% training job completion rate (excluding user-cancelled jobs)
- **Response Time:** Training API endpoints <200ms (p95), trajectory generation <2x baseline execution time
- **Throughput:** Support 100+ concurrent training jobs, 8 parallel trajectories in <30s
- **Error Rate:** <1% system errors (excluding user configuration errors)

**Key Performance Indicators (KPIs):**

- **Task Success Rate:** +15% improvement over baseline (minimum viable), +20% target
- **Sample Efficiency:** Convergence with <10,000 training trajectories
- **Training Stability:** Monotonic improvement in validation metrics over 90% of iterations
- **Cost Efficiency:** Training cost <$25 per 1000 queries, ROI >20x within 3 months

**Technical KPIs:**

- **Database Performance:** >100 trajectory writes/sec sustained
- **Parallel Execution:** 8 trajectories complete in <30s (95th percentile)
- **Policy Update Latency:** <5s per batch update
- **Storage Growth:** Trajectory data <100MB per 10K trajectories (with compression)

---

## 💻 Code Examples & Patterns

### Repository Patterns (from AgentCore Codebase)

Since `.sage/agent/examples/` does not exist yet, patterns are derived from existing AgentCore code structure documented in system architecture and CLAUDE.md:

#### 1. **Async/Await Pattern** (Agent Runtime Layer)

**Application:** Parallel trajectory generation for GRPO training

**Existing Pattern:**

```python
# From agent runtime - concurrent agent execution
async def execute_multiple_agents(agents: list[Agent]) -> list[Result]:
    tasks = [agent.execute() for agent in agents]
    return await asyncio.gather(*tasks)
```

**Adaptation for Trajectories:**

```python
# src/agentcore/training/trajectory.py
class TrajectoryCollector:
    async def collect(
        self,
        agent: Agent,
        query: str,
        n_trajectories: int = 8
    ) -> list[Trajectory]:
        """Generate N trajectories in parallel."""
        tasks = [
            self._generate_single_trajectory(agent, query)
            for _ in range(n_trajectories)
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)
```

**Key Adaptation Notes:**

- Use `return_exceptions=False` to fail fast on critical errors
- Add timeout wrapper for trajectory generation (max 20 steps)
- Leverage existing agent execution infrastructure

---

#### 2. **JSON-RPC Decorator Pattern** (A2A Protocol Layer)

**Application:** Training API endpoint registration

**Existing Pattern:**

```python
# From agentcore/a2a_protocol/services/jsonrpc_handler.py
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

@register_jsonrpc_method("agent.register")
async def handle_agent_register(request: JsonRpcRequest) -> dict[str, Any]:
    agent_id = request.params["agent_id"]
    # Implementation
    return {"agent_id": agent_id, "status": "registered"}
```

**Adaptation for Training:**

```python
# src/agentcore/training/training_jsonrpc.py
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.training.models import GRPOConfig, TrainingJob

@register_jsonrpc_method("training.start_grpo")
async def start_grpo_training(request: JsonRpcRequest) -> dict[str, Any]:
    """Start GRPO training job."""
    agent_id = request.params["agent_id"]
    training_data = request.params["training_data"]
    config = GRPOConfig(**request.params.get("config", {}))

    job = await training_manager.create_job(agent_id, training_data, config)
    return {"training_job_id": job.job_id, "status": "queued"}
```

**Key Adaptation Notes:**

- Import module in `main.py` to auto-register methods
- Use Pydantic models for config validation
- Return job_id for async status tracking

---

#### 3. **Database Repository Pattern** (Database Layer)

**Application:** Trajectory and training job persistence

**Existing Pattern:**

```python
# From agentcore/a2a_protocol/database/repositories.py
from agentcore.a2a_protocol.database import get_session

async with get_session() as session:
    repo = AgentRepository(session)
    agent = await repo.get_by_id(agent_id)
```

**Adaptation for Training:**

```python
# src/agentcore/training/repositories.py
from agentcore.a2a_protocol.database import get_session
from agentcore.training.models import Trajectory, TrainingJob

class TrajectoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_trajectory(self, trajectory: Trajectory) -> None:
        """Save trajectory with all steps."""
        stmt = insert(TrajectoryRecord).values(
            trajectory_id=trajectory.trajectory_id,
            agent_id=trajectory.agent_id,
            query=trajectory.query,
            steps=trajectory.steps_to_json(),  # JSONB column
            reward=trajectory.reward,
            advantage=trajectory.advantage
        )
        await self.session.execute(stmt)
        await self.session.commit()

    async def get_trajectories_for_job(
        self, job_id: str, limit: int = 1000
    ) -> list[Trajectory]:
        """Retrieve trajectories for training job."""
        stmt = select(TrajectoryRecord).where(
            TrajectoryRecord.job_id == job_id
        ).limit(limit)
        result = await self.session.execute(stmt)
        return [Trajectory.from_record(r) for r in result.scalars()]

# Usage in training loop
async with get_session() as session:
    repo = TrajectoryRepository(session)
    await repo.save_trajectory(trajectory)
```

**Key Adaptation Notes:**

- Use JSONB columns for flexible trajectory step storage
- Bulk insert for performance: `insert().values([...])`
- Add indexes on `job_id`, `agent_id`, `created_at`

---

#### 4. **Pydantic Models Pattern** (Type-Safe Validation)

**Application:** Training configuration and data models

**Existing Pattern:**

```python
# From agentcore/a2a_protocol/models/
from pydantic import BaseModel, Field
from datetime import datetime

class AgentCard(BaseModel):
    id: str = Field(..., description="Unique agent identifier")
    name: str
    capabilities: list[str]
    created_at: datetime = Field(default_factory=datetime.now)
```

**Adaptation for Training:**

```python
# src/agentcore/training/models.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class GRPOConfig(BaseModel):
    """Configuration for GRPO training."""
    n_iterations: int = Field(default=1000, ge=1, le=10000)
    batch_size: int = Field(default=16, ge=1, le=128)
    n_trajectories_per_query: int = Field(default=8, ge=1, le=32)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.1)
    max_budget_usd: float = Field(default=100, ge=0)
    checkpoint_interval: int = Field(default=10, ge=1)

    @field_validator("max_budget_usd")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Budget must be positive")
        return v

class Trajectory(BaseModel):
    """Single execution trajectory."""
    trajectory_id: str
    agent_id: str
    query: str
    steps: list[dict[str, Any]]
    reward: float = 0.0
    advantage: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def steps_to_json(self) -> str:
        """Serialize steps for JSONB storage."""
        return json.dumps(self.steps)
```

**Key Adaptation Notes:**

- Use `Field()` with constraints for validation
- Add `@field_validator` for complex validation logic
- Define clear defaults for all optional parameters

---

#### 5. **Background Task Pattern** (Agent Runtime)

**Application:** Long-running training jobs

**Existing Pattern:**

```python
# From agent runtime - background monitoring
async def monitor_agent(agent_id: str):
    """Long-running agent monitoring task."""
    while True:
        status = await check_agent_health(agent_id)
        await update_status(status)
        await asyncio.sleep(30)

# Start background task
asyncio.create_task(monitor_agent(agent_id))
```

**Adaptation for Training:**

```python
# src/agentcore/training/trainer.py
class TrainingJobManager:
    def __init__(self):
        self._active_jobs: dict[str, asyncio.Task] = {}
        self._redis = redis.asyncio.Redis()

    async def start_job(self, job: TrainingJob) -> str:
        """Start training job in background."""
        task = asyncio.create_task(self._run_training_job(job))
        self._active_jobs[job.job_id] = task

        # Store job metadata in Redis
        await self._redis.hset(
            f"training:job:{job.job_id}",
            mapping={"status": "running", "progress": 0}
        )

        return job.job_id

    async def _run_training_job(self, job: TrainingJob) -> None:
        """Execute training job."""
        try:
            for iteration in range(job.config.n_iterations):
                # Training iteration logic
                await self._training_iteration(job, iteration)

                # Update progress in Redis
                await self._redis.hset(
                    f"training:job:{job.job_id}",
                    "progress", iteration / job.config.n_iterations
                )

                # Check for cancellation
                if await self._is_cancelled(job.job_id):
                    break

            # Mark complete
            await self._redis.hset(
                f"training:job:{job.job_id}", "status", "completed"
            )
        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}")
            await self._redis.hset(
                f"training:job:{job.job_id}", "status", "failed"
            )
```

**Key Adaptation Notes:**

- Use Redis for job state (survives server restart)
- Store task references for cancellation support
- Implement graceful shutdown with cancellation checks

---

### Implementation Reference Examples

**From Research (flow-based-optimization.md):**

#### GRPO Reward Normalization (Lines 96-122)

```python
# Recommended pattern from research
def refine_rewards(trajectories: list[Trajectory]) -> list[Trajectory]:
    """
    Normalize rewards using group statistics.

    Reduces variance by using relative performance within batch.
    """
    rewards = np.array([t.reward for t in trajectories])
    mean_reward = rewards.mean()
    std_reward = rewards.std()

    for trajectory in trajectories:
        if std_reward > 0:
            trajectory.normalized_reward = (
                (trajectory.reward - mean_reward) / std_reward
            )
        else:
            # All trajectories have same reward - no advantage
            trajectory.normalized_reward = 0.0

        # Advantage = normalized reward for GRPO
        trajectory.advantage = trajectory.normalized_reward

    return trajectories
```

**Key Takeaways:**

- Always normalize rewards before policy updates
- Handle edge case: std_reward == 0 (all same reward)
- Store normalized_reward and advantage separately for debugging

---

#### Policy Gradient Update (Lines 125-169)

```python
# Pattern from research - simplified for LLM-based agents
async def update_policy(
    agent: Agent,
    trajectories: list[Trajectory],
    optimizer: Optimizer
) -> dict[str, float]:
    """
    Update agent policy using policy gradient.

    For LLM-based agents: Update system prompt or few-shot examples
    For neural agents: Update model weights
    """
    total_loss = 0.0
    n_updates = 0

    # Only update on trajectories that performed better than average
    for trajectory in trajectories:
        if trajectory.advantage <= 0:
            continue  # Skip below-average trajectories

        # For LLM agents: Extract successful patterns
        successful_patterns = extract_patterns(trajectory)

        # Update agent context/prompt (simplified policy update)
        await agent.update_context(
            examples=successful_patterns,
            weight=trajectory.advantage  # Higher advantage = stronger update
        )

        n_updates += 1

    return {
        "n_updates": n_updates,
        "avg_reward": np.mean([t.reward for t in trajectories]),
        "max_reward": np.max([t.reward for t in trajectories])
    }
```

**Key Takeaways:**

- Only update on positive-advantage trajectories (better than average)
- For LLM agents, policy update = context/prompt evolution
- Track number of updates for monitoring convergence

---

### New Patterns to Create

**Patterns This Implementation Will Establish:**

#### 1. **Trajectory Recording Middleware**

**Purpose:** Wrap agent execution to capture full execution trace without modifying agent code

**Pattern:**

```python
# Will create: src/agentcore/training/middleware/trajectory_recorder.py
class TrajectoryRecorder:
    """Middleware to record agent execution as trajectory."""

    async def record_execution(
        self, agent: Agent, query: str
    ) -> Trajectory:
        trajectory = Trajectory(query=query, agent_id=agent.id)

        # Wrap agent execution
        original_execute = agent.execute

        async def instrumented_execute(*args, **kwargs):
            step = {"input": args, "timestamp": time.time()}
            result = await original_execute(*args, **kwargs)
            step["output"] = result
            trajectory.steps.append(step)
            return result

        agent.execute = instrumented_execute

        # Run agent
        await agent.run(query)

        # Restore original
        agent.execute = original_execute

        return trajectory
```

**Reusability:** Any component needing execution traces (debugging, monitoring, audit)

---

#### 2. **Budget Enforcement Decorator**

**Purpose:** Enforce token budget limits on training operations

**Pattern:**

```python
# Will create: src/agentcore/training/utils/budget.py
from functools import wraps

def enforce_budget(budget_key: str):
    """Decorator to check and enforce budget before expensive operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check current budget usage
            used = await redis.get(f"budget:{budget_key}:used") or 0
            limit = await redis.get(f"budget:{budget_key}:limit")

            if used >= limit:
                raise BudgetExceededError(
                    f"Budget limit ${limit} exceeded (used: ${used})"
                )

            # Execute function
            result = await func(*args, **kwargs)

            # Update usage (from Portkey cost tracking)
            cost = result.get("cost_usd", 0)
            await redis.incrby(f"budget:{budget_key}:used", cost)

            return result
        return wrapper
    return decorator

# Usage
@enforce_budget("training:job:12345")
async def generate_trajectories(...):
    ...
```

**Reusability:** Any cost-sensitive operation (LLM calls, API usage)

---

#### 3. **Checkpoint Versioning Strategy**

**Purpose:** Manage model checkpoints with best-checkpoint selection

**Pattern:**

```python
# Will create: src/agentcore/training/checkpoint.py
class CheckpointManager:
    """Manage training checkpoints with versioning."""

    def __init__(self, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list[Checkpoint] = []

    async def save_checkpoint(
        self, agent: Agent, iteration: int, metrics: dict
    ) -> Checkpoint:
        """Save checkpoint and maintain best-N."""
        ckpt = Checkpoint(
            agent_id=agent.id,
            iteration=iteration,
            metrics=metrics,
            validation_score=metrics.get("eval_success_rate", 0)
        )

        # Save to storage
        await self._persist_checkpoint(ckpt)

        # Add to list
        self.checkpoints.append(ckpt)

        # Keep only top N by validation score
        self.checkpoints.sort(key=lambda c: c.validation_score, reverse=True)

        # Delete old checkpoints
        for old_ckpt in self.checkpoints[self.max_checkpoints:]:
            await self._delete_checkpoint(old_ckpt)

        self.checkpoints = self.checkpoints[:self.max_checkpoints]

        return ckpt

    def get_best_checkpoint(self) -> Checkpoint | None:
        """Get checkpoint with highest validation score."""
        return self.checkpoints[0] if self.checkpoints else None
```

**Reusability:** Any training system with checkpoint management (ACE context evolution, DSPy optimization)

---

## 🔧 Technology Stack

### Recommended Stack (from Research & Existing Infrastructure)

**Based on research from:** `docs/research/flow-based-optimization.md` and existing AgentCore tech stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| **Runtime** | Python | 3.12+ | Existing stack; native async/await for parallel trajectory generation (research: 8 concurrent trajectories target) |
| **Framework** | FastAPI | 0.104+ | Existing API framework; JSON-RPC pattern consistency with A2A protocol layer |
| **Database** | PostgreSQL + asyncpg | 2.0+ | Existing database; JSONB for flexible trajectory storage; research estimates 100GB-1TB growth supported |
| **Job Queue** | Redis | 5.0+ | Existing Redis cluster; distributed job coordination; pub/sub for progress updates |
| **LLM Gateway** | Portkey AI | 1.15+ | Existing gateway; unified provider API; built-in cost tracking for budget enforcement |
| **ORM** | SQLAlchemy (async) | 2.0+ | Existing ORM; async support; repository pattern alignment |
| **Validation** | Pydantic | 2.5+ | Existing validation framework; type-safe config models |
| **Testing** | pytest + pytest-asyncio | 8.4+ / 1.2+ | Existing test infrastructure; async test support; 90% coverage requirement |
| **Optimization** | NumPy | Latest | Reward normalization, advantage computation |
| **Monitoring** | Prometheus + structlog | Existing | Metrics export, structured logging for training events |

### Key Technology Decisions

**Decision 1: Python asyncio over Threading**

- **Rationale:** Research shows need for 8 concurrent trajectories; asyncio provides efficient I/O concurrency without threading overhead
- **Research Citation:** flow-based-optimization.md, lines 53-91 (trajectory rollout pattern)
- **Implementation:** `asyncio.gather()` for parallel trajectory collection
- **Trade-offs:**
  - **Pros:** Lower memory overhead, simpler debugging, existing codebase pattern
  - **Cons:** CPU-bound operations still serial (mitigated: trajectory generation is I/O bound)

**Decision 2: PostgreSQL JSONB over Specialized Time-Series DB**

- **Rationale:** Flexibility for evolving trajectory schema; existing PostgreSQL infrastructure; JSONB query performance sufficient
- **Research Support:** Research estimates 100GB-1TB trajectory data - PostgreSQL handles this scale
- **Implementation:** JSONB columns for trajectory steps, GIN indexes for query performance
- **Trade-offs:**
  - **Pros:** Reuse existing database, simpler operations, flexible schema
  - **Cons:** Higher storage than binary format (mitigated: JSONB compression)

**Decision 3: Redis Job Queue over Celery**

- **Rationale:** Lighter weight; existing Redis cluster; simpler configuration; sufficient for training job coordination
- **Implementation:** Redis lists for job queue, hashes for job state, pub/sub for progress updates
- **Trade-offs:**
  - **Pros:** No additional dependencies, lower complexity, faster development
  - **Cons:** Less mature than Celery (mitigated: simpler use case, less feature need)

**Decision 4: Portkey AI for LLM Access**

- **Rationale:** Existing gateway; unified API for multiple providers (OpenAI, Anthropic, etc.); built-in cost tracking enables budget enforcement
- **Research Support:** Budget enforcement is P0 requirement (spec FR-4)
- **Implementation:** Wrap Portkey client; track costs per training job
- **Trade-offs:**
  - **Pros:** Cost tracking built-in, provider flexibility, existing integration
  - **Cons:** Additional API layer (mitigated: minimal latency impact for batch training)

**Decision 5: Defer Neural Network Training Libraries (Phase 3)**

- **Rationale:** Start with prompt-based policy updates (context evolution); add PyTorch/TensorFlow only if neural policy updates needed
- **Research Support:** Research targets LLM-based agents; prompt optimization is sufficient for Phase 1-2
- **Implementation:** Phase 1-2 use text-based policy updates; Phase 3 adds optional PyTorch for advanced users
- **Trade-offs:**
  - **Pros:** Simpler implementation, lower dependencies, faster development
  - **Cons:** Limits advanced neural policy optimization (mitigated: deferred to Phase 3, rare use case)

### Alternatives Considered (from Research)

**Option 2: Celery + RabbitMQ for Job Queue**

- **Pros:** Mature ecosystem, advanced scheduling features, better monitoring
- **Cons:** Adds RabbitMQ dependency, higher complexity, slower development
- **Why Not Chosen:** Overhead not justified for training job coordination; Redis sufficient for current needs

**Option 3: MongoDB for Trajectory Storage**

- **Pros:** Native document storage, flexible schema, horizontal scaling
- **Cons:** New database dependency, team unfamiliar, ACID trade-offs
- **Why Not Chosen:** PostgreSQL JSONB provides similar flexibility; existing expertise and infrastructure

**Option 4: PyTorch from Phase 1**

- **Pros:** Advanced optimization algorithms, gradient computation, model checkpointing
- **Cons:** Large dependency, steeper learning curve, overkill for prompt-based policies
- **Why Not Chosen:** Deferred to Phase 3; start simple with custom policy gradients for LLM agents

### Alignment with Existing System

**From `pyproject.toml` and System Architecture:**

**Consistent With:**

- Python 3.12+ runtime (existing)
- FastAPI 0.104+ framework (existing)
- PostgreSQL with asyncpg driver (existing)
- Redis 5.0+ cluster (existing)
- Pydantic 2.5+ validation (existing)
- pytest + pytest-asyncio testing (existing)
- Prometheus monitoring (existing)
- structlog logging (existing)

**New Additions:**

- NumPy for numerical computations (reward normalization, advantage calculation)
- Optional: PyTorch (Phase 3 only, for advanced neural policy updates)

**Migration Considerations:**

- No migration needed - pure additions to existing stack
- Training module is isolated - can be developed independently
- Database schema changes are additive (new tables only)

---

## 🏗️ Architecture Design

### System Context (from AgentCore Architecture)

**Existing AgentCore System Architecture (6-Layer Model):**

From `docs/agentcore-architecture-and-development-plan.md`:

1. **Infrastructure Layer:** UV, Redis Cluster, Portkey Gateway, FastAPI, PostgreSQL, Event Store
2. **Agent Runtime Layer:** Multi-Agent Coordination, Secure Sandboxes, ReAct/CoT/Autonomous runtimes, Tool Integration
3. **Enhanced Core Layer:** Agent Registry, A2A Protocol (JSON-RPC), Task Management, Communication Hub, Orchestration, Multi-Level Cache
4. **Enterprise Operations Layer:** Decision Lineage, Multi-Tenancy, Security & Compliance, Audit, Billing
5. **Experience Layer:** Developer APIs, Hot-Reload, Interactive Debugging, CLI Tools, Pattern Library
6. **Intelligence Layer:** **DSPy Optimization**, **Self-Evolution Manager** ← **Flow-Based Optimization fits here**

**Integration Points:**

**Flow-Based Optimization Position:** Intelligence Layer (alongside DSPy Optimization)

- **Downstream:** Agent Runtime Layer (uses existing agent execution for trajectory generation)
- **Downstream:** Enhanced Core Layer (uses Task Management for execution tracking, JSON-RPC for API)
- **Downstream:** Infrastructure Layer (PostgreSQL for persistence, Redis for job queue, Portkey for LLM)
- **Future Integration:** DSPy Optimization (dual optimization: GRPO + prompt optimization)
- **Future Integration:** Self-Evolution Manager (ACE context evolution + GRPO training)

### Component Architecture

**Architecture Pattern:** Background Worker + Job Queue (Asynchronous Task Processing)

**Rationale:**

- **Training Isolation:** Training overhead doesn't impact production agent performance
- **Scalability:** Workers can scale independently based on job queue depth
- **Reliability:** Job queue survives server restarts, supports resume from checkpoint
- **Existing Pattern:** Aligns with agent runtime background monitoring pattern

**Alignment with Existing Architecture:**

- Uses existing JSON-RPC API pattern (consistent with A2A protocol)
- Follows repository pattern for database access (consistent with agent/task management)
- Leverages existing Redis cluster for coordination
- Integrates with Portkey AI gateway for LLM access

**System Design:**

```mermaid
graph TB
    subgraph "External"
        User[Platform Operator]
        LLM[LLM Providers<br/>via Portkey]
    end

    subgraph "Intelligence Layer - Flow-Based Optimization"
        API[Training API<br/>JSON-RPC Endpoints]
        JobMgr[Training Job Manager<br/>Job Lifecycle]
        Worker[Training Worker Pool<br/>Background Execution]

        subgraph "Core Training Components"
            Collector[Trajectory Collector<br/>Parallel Execution]
            Trainer[GRPO Trainer<br/>Policy Gradient]
            Reward[Reward Engine<br/>Shaped Rewards]
            Policy[Policy Updater<br/>Context Evolution]
        end

        subgraph "Support Components"
            Eval[Evaluation Framework<br/>Held-out Validation]
            Ckpt[Checkpoint Manager<br/>Save/Restore]
            Budget[Budget Enforcer<br/>Cost Control]
        end
    end

    subgraph "Enhanced Core Layer"
        TaskMgmt[Task Management]
        JSONRPC[JSON-RPC Handler]
    end

    subgraph "Agent Runtime Layer"
        AgentExec[Agent Executor<br/>Multi-Philosophy]
    end

    subgraph "Infrastructure Layer"
        Redis[(Redis<br/>Job Queue)]
        Postgres[(PostgreSQL<br/>Trajectories)]
        Portkey[Portkey Gateway<br/>LLM Access]
    end

    User -->|training.start_grpo| API
    User -->|training.get_status| API
    API --> JobMgr
    JobMgr -->|Enqueue| Redis
    Redis -->|Dequeue| Worker

    Worker --> Collector
    Collector -->|Execute Agent| AgentExec
    AgentExec -->|Tool Calls| LLM
    LLM -->|Responses| AgentExec
    AgentExec -->|Results| Collector

    Collector --> Reward
    Reward --> Trainer
    Trainer --> Policy
    Policy -->|Update Agent| AgentExec

    Worker --> Eval
    Worker --> Ckpt
    Worker --> Budget

    Collector -->|Save Trajectories| Postgres
    Ckpt -->|Save Checkpoints| Postgres
    Eval -->|Query Metrics| Postgres

    Budget -->|Cost Tracking| Portkey

    API --> JSONRPC
    Collector --> TaskMgmt
```

### Architecture Decisions (from Research)

**Decision 1: Training Isolation Pattern**

- **Choice:** Background worker processes, separate from agent runtime
- **Rationale:** Prevents training overhead from impacting production agents; enables independent scaling
- **Implementation:** Redis job queue + async worker pool pattern
- **Trade-offs:**
  - **Pros:** Production isolation, horizontal scalability, fault tolerance
  - **Cons:** Added complexity (job queue management, worker coordination)

**Decision 2: Trajectory Collection via Wrapper Pattern**

- **Choice:** Wrap existing agent execution with trajectory recorder, don't modify agent code
- **Rationale:** Reuses existing agent runtime; non-invasive; agents unaware of training
- **Implementation:** Middleware/decorator pattern wrapping agent.execute()
- **Trade-offs:**
  - **Pros:** Zero agent code changes, reuses existing infrastructure, simpler testing
  - **Cons:** Limited visibility into agent internals (mitigated: tool call logging sufficient)

**Decision 3: Policy Update Strategy (Prompt-Based)**

- **Choice:** Start with prompt/context evolution (text-based policy updates); defer neural network training
- **Rationale:** LLM-based agents use prompts as "policy"; simpler implementation; sufficient for Phase 1-2
- **Implementation:** Update agent system prompt or few-shot examples based on successful trajectories
- **Trade-offs:**
  - **Pros:** Simpler, faster development; no neural network dependencies; aligns with LLM agents
  - **Cons:** Limits advanced optimization (mitigated: Phase 3 adds neural policy updates if needed)

**Decision 4: Checkpoint Storage (Hybrid)**

- **Choice:** Metadata in PostgreSQL (<10MB), large model weights in object storage (S3)
- **Rationale:** Keeps database size manageable; supports large model checkpoints
- **Implementation:** PostgreSQL for checkpoint metadata + path to S3 object
- **Trade-offs:**
  - **Pros:** Scalable storage, efficient database queries, supports large models
  - **Cons:** Adds S3 dependency (mitigated: deferred to Phase 2, optional for small checkpoints)

**Decision 5: Budget Enforcement (Pre-flight + Real-time)**

- **Choice:** Check budget before each trajectory batch + abort if exceeded during execution
- **Rationale:** Prevents cost overruns; uses existing Portkey cost tracking
- **Implementation:** Query Portkey usage API before batch; monitor during execution; abort at threshold
- **Trade-offs:**
  - **Pros:** Hard cost limits, real-time enforcement, uses existing infrastructure
  - **Cons:** Adds latency for cost checks (mitigated: batch checks, not per-call)

### Component Breakdown

**Core Components:**

#### 1. **Training API Handler**

- **Purpose:** Expose JSON-RPC endpoints for training job management
- **Technology:** FastAPI JSON-RPC decorator pattern
- **Pattern:** From existing A2A protocol JSON-RPC handler
- **Interfaces:**
  - `training.start_grpo(agent_id, training_data, config) -> job_id`
  - `training.get_status(job_id) -> status, metrics, progress`
  - `training.cancel(job_id) -> success`
  - `training.evaluate(agent_id, eval_queries) -> metrics`
  - `training.export_trajectories(job_id, filters) -> trajectories`
- **Dependencies:** TrainingJobManager, JSON-RPC Handler

---

#### 2. **Training Job Manager**

- **Purpose:** Manage training job lifecycle (create, schedule, monitor, cancel)
- **Technology:** Python asyncio tasks + Redis job queue
- **Pattern:** Background task pattern from agent runtime
- **Interfaces:**
  - `create_job(agent_id, data, config) -> TrainingJob`
  - `start_job(job_id) -> task_handle`
  - `get_job_status(job_id) -> JobStatus`
  - `cancel_job(job_id) -> bool`
- **Dependencies:** Redis (job queue), TrainingWorker, TrainingJobRepository

---

#### 3. **Training Worker Pool**

- **Purpose:** Execute training jobs in background worker processes
- **Technology:** Async worker pattern consuming from Redis queue
- **Pattern:** Long-running background task with job processing loop
- **Interfaces:**
  - `start_worker() -> None` (blocking, runs in separate process)
  - `process_job(job: TrainingJob) -> None`
- **Dependencies:** TrajectoryCollector, GRPOTrainer, Redis

---

#### 4. **Trajectory Collector**

- **Purpose:** Generate multiple execution trajectories for a query-agent pair
- **Technology:** Python asyncio.gather() for parallel execution
- **Pattern:** Async/await parallel execution pattern
- **Interfaces:**
  - `collect(agent, query, n_trajectories) -> list[Trajectory]`
  - `_generate_single_trajectory(agent, query) -> Trajectory`
- **Dependencies:** Agent Runtime, TaskManager (for execution tracking)

---

#### 5. **GRPO Trainer**

- **Purpose:** Implement Group Refined Policy Optimization algorithm
- **Technology:** NumPy for numerical computations, custom policy gradient
- **Pattern:** Iterative training loop from research
- **Interfaces:**
  - `train_iteration(trajectories, agent) -> metrics`
  - `_refine_rewards(trajectories) -> trajectories_with_advantages`
  - `_update_policy(agent, trajectories) -> loss_metrics`
- **Dependencies:** RewardEngine, PolicyUpdater, NumPy

---

#### 6. **Reward Engine**

- **Purpose:** Compute rewards for trajectories (outcome-based + shaped)
- **Technology:** Configurable reward functions, NumPy
- **Pattern:** Strategy pattern for pluggable reward functions
- **Interfaces:**
  - `compute_reward(trajectory) -> float`
  - `compute_shaped_rewards(trajectory) -> dict[str, float]`
  - `register_reward_function(name, func) -> None`
- **Dependencies:** None (pure computation)

---

#### 7. **Policy Updater**

- **Purpose:** Update agent policies based on trajectory advantages
- **Technology:** LLM prompt updates (Phase 1-2), optional PyTorch (Phase 3)
- **Pattern:** Strategy pattern for different policy types
- **Interfaces:**
  - `update_policy(agent, trajectories) -> metrics`
  - `extract_successful_patterns(trajectory) -> patterns`
  - `save_policy_checkpoint(agent) -> checkpoint_id`
- **Dependencies:** Portkey AI (for LLM-based updates), CheckpointManager

---

#### 8. **Evaluation Framework**

- **Purpose:** Evaluate trained agents on held-out queries
- **Technology:** Statistical testing, metrics computation
- **Pattern:** Test runner pattern
- **Interfaces:**
  - `evaluate(agent, eval_queries) -> metrics`
  - `compare_to_baseline(trained_metrics, baseline_metrics) -> comparison`
- **Dependencies:** TrajectoryCollector, AgentRuntime

---

#### 9. **Checkpoint Manager**

- **Purpose:** Save and restore agent policy checkpoints with versioning
- **Technology:** PostgreSQL for metadata, S3 for large weights (Phase 2)
- **Pattern:** Checkpoint versioning pattern (new, established in this implementation)
- **Interfaces:**
  - `save_checkpoint(agent, iteration, metrics) -> checkpoint_id`
  - `load_checkpoint(checkpoint_id) -> agent_policy`
  - `get_best_checkpoint(agent_id) -> checkpoint_id`
  - `cleanup_old_checkpoints(agent_id, keep_n=5) -> None`
- **Dependencies:** CheckpointRepository, PostgreSQL, S3 (Phase 2)

---

#### 10. **Budget Enforcer**

- **Purpose:** Enforce token budget limits and cost tracking
- **Technology:** Portkey AI cost API, Redis for budget state
- **Pattern:** Decorator pattern for budget enforcement (new)
- **Interfaces:**
  - `check_budget(job_id) -> remaining_budget`
  - `enforce_budget(job_id, operation) -> wrapped_operation` (decorator)
  - `update_usage(job_id, cost) -> None`
- **Dependencies:** Portkey AI (cost tracking), Redis (budget state)

---

### Data Flow & Boundaries

**Request Flow (Training Job Creation):**

1. **User → Training API:** `training.start_grpo(agent_id, data, config)`
2. **Training API → Job Manager:** `create_job(agent_id, data, config)`
3. **Job Manager → Redis:** Enqueue job in `training:jobs` queue
4. **Job Manager → User:** Return `job_id` + `status: queued`

**Execution Flow (Background Worker):**

1. **Worker → Redis:** Dequeue job from `training:jobs` queue
2. **Worker → Trajectory Collector:** `collect(agent, query, n=8)` for each query in batch
3. **Trajectory Collector → Agent Runtime:** Execute agent N times in parallel
4. **Agent Runtime → Portkey AI:** LLM calls for agent inference
5. **Trajectory Collector → PostgreSQL:** Save trajectories to `trajectories` table
6. **Worker → Reward Engine:** `compute_reward(trajectory)` for each trajectory
7. **Worker → GRPO Trainer:** `train_iteration(trajectories, agent)`
8. **GRPO Trainer → Policy Updater:** `update_policy(agent, trajectories_with_positive_advantage)`
9. **Policy Updater → Portkey AI:** LLM calls to generate policy updates
10. **Worker → Checkpoint Manager:** `save_checkpoint(agent, iteration, metrics)` every N iterations
11. **Worker → Evaluation Framework:** `evaluate(agent, eval_queries)` every N iterations
12. **Worker → Redis:** Update job status and progress in `training:job:{job_id}`

**Query Flow (Status Check):**

1. **User → Training API:** `training.get_status(job_id)`
2. **Training API → Redis:** `HGETALL training:job:{job_id}`
3. **Training API → User:** Return status, progress, metrics

**Component Boundaries:**

**Public Interface (JSON-RPC API):**

- `training.start_grpo` - Start training job
- `training.get_status` - Query job status
- `training.cancel` - Cancel running job
- `training.evaluate` - Evaluate trained agent
- `training.export_trajectories` - Export trajectory data

**Internal Implementation (Hidden):**

- Trajectory collection logic
- GRPO algorithm internals
- Reward computation details
- Policy update mechanisms
- Checkpoint storage format

**Cross-Component Contracts:**

- **Agent Runtime → Trajectory Collector:** Agent must implement `execute(query)` interface
- **GRPO Trainer → Reward Engine:** Trajectories must have `steps` list with `success` indicator
- **Policy Updater → Agent Runtime:** Agent must support `update_context(examples)` for prompt updates
- **Budget Enforcer → Portkey AI:** Portkey must provide `get_usage(api_key)` endpoint

---

## 4. Technical Specification

### Data Model

**Entities and Relationships:**

```sql
-- Training Jobs Table
CREATE TABLE training_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    config JSONB NOT NULL,  -- GRPOConfig serialized
    training_data JSONB NOT NULL,  -- List of {query, expected_outcome}
    current_iteration INT DEFAULT 0,
    total_iterations INT NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,  -- Current training metrics
    cost_usd DECIMAL(10, 2) DEFAULT 0,
    budget_usd DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,

    -- Indexes
    CONSTRAINT fk_agent FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX idx_training_jobs_agent ON training_jobs(agent_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_created ON training_jobs(created_at DESC);

-- Trajectories Table
CREATE TABLE trajectories (
    trajectory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    steps JSONB NOT NULL,  -- List of {state, action, result, timestamp}
    reward FLOAT NOT NULL DEFAULT 0,
    normalized_reward FLOAT DEFAULT 0,
    advantage FLOAT DEFAULT 0,
    execution_time_ms INT,
    success BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Indexes
    CONSTRAINT fk_training_job FOREIGN KEY (job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX idx_trajectories_job ON trajectories(job_id);
CREATE INDEX idx_trajectories_agent ON trajectories(agent_id);
CREATE INDEX idx_trajectories_created ON trajectories(created_at DESC);
CREATE INDEX idx_trajectories_success ON trajectories(success);

-- GIN index for JSONB step queries
CREATE INDEX idx_trajectories_steps_gin ON trajectories USING GIN (steps);

-- Policy Checkpoints Table
CREATE TABLE policy_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    job_id UUID NOT NULL,
    iteration INT NOT NULL,
    policy_data JSONB,  -- For small policies (prompts, examples)
    policy_s3_path VARCHAR(500),  -- For large policies (model weights)
    validation_score FLOAT NOT NULL DEFAULT 0,
    metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Indexes
    CONSTRAINT fk_agent_ckpt FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_job_ckpt FOREIGN KEY (job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX idx_checkpoints_agent ON policy_checkpoints(agent_id);
CREATE INDEX idx_checkpoints_job ON policy_checkpoints(job_id);
CREATE INDEX idx_checkpoints_score ON policy_checkpoints(validation_score DESC);
CREATE INDEX idx_checkpoints_created ON policy_checkpoints(created_at DESC);
```

**Validation Rules and Constraints:**

1. **Training Jobs:**
   - `status` must be one of: queued, running, completed, failed, cancelled
   - `config.n_iterations` must be between 1 and 10,000
   - `config.batch_size` must be between 1 and 128
   - `config.n_trajectories_per_query` must be between 1 and 32
   - `budget_usd` must be positive
   - `cost_usd` ≤ `budget_usd` (enforced by application logic)

2. **Trajectories:**
   - `steps` must be non-empty array
   - Each step must have: `{state, action, result, timestamp}`
   - `reward` can be negative (failures have negative rewards)
   - `advantage` is relative (can be negative)

3. **Policy Checkpoints:**
   - Either `policy_data` OR `policy_s3_path` must be set (not both)
   - `policy_data` size limit: 10MB (enforced by application)
   - `validation_score` typically in range [0, 1] but not enforced

**Indexing Strategy:**

1. **Primary Queries:**
   - Get job by ID: Primary key on `job_id`
   - List jobs for agent: Index on `agent_id`
   - List active jobs: Index on `status`
   - Get recent jobs: Index on `created_at DESC`

2. **Trajectory Queries:**
   - Get trajectories for job: Index on `job_id`
   - Filter by success/failure: Index on `success`
   - Query trajectory steps: GIN index on JSONB `steps` column

3. **Checkpoint Queries:**
   - Get best checkpoint for agent: Composite index on `(agent_id, validation_score DESC)`
   - List checkpoints for job: Index on `job_id`

**Migration Approach:**

1. **Phase 1:** Create tables with Alembic migration

   ```bash
   uv run alembic revision --autogenerate -m "Add training infrastructure tables"
   uv run alembic upgrade head
   ```

2. **Phase 2:** Add S3 integration (optional, if needed)
   - Add `policy_s3_path` column to checkpoints
   - Migrate existing large policies to S3

3. **Phase 3:** Add indexes for performance optimization
   - Monitor query patterns in production
   - Add additional indexes as needed

---

### API Design

**Top 6 Critical Endpoints:**

#### 1. `training.start_grpo` - Start Training Job

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Create and queue a new GRPO training job

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.start_grpo",
  "params": {
    "agent_id": "string (required)",
    "training_data": [
      {
        "query": "string",
        "expected_outcome": {"answer": "string", "correct": true}
      }
    ],
    "config": {
      "n_iterations": 1000,
      "batch_size": 16,
      "n_trajectories_per_query": 8,
      "learning_rate": 0.0001,
      "max_budget_usd": 100,
      "checkpoint_interval": 10
    }
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "training_job_id": "uuid",
    "status": "queued",
    "estimated_cost_usd": 25.0,
    "estimated_duration_hours": 8
  },
  "id": 1
}
```

**Error Handling:**

- `400` Invalid agent_id or configuration
- `409` Agent already has active training job
- `413` Training data exceeds size limit (>10,000 queries)
- `507` Insufficient budget or resources

---

#### 2. `training.get_status` - Query Job Status

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Retrieve current status and metrics for a training job

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.get_status",
  "params": {
    "job_id": "uuid (required)"
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "uuid",
    "status": "running",
    "progress": {
      "current_iteration": 150,
      "total_iterations": 1000,
      "percent_complete": 15
    },
    "metrics": {
      "loss": 0.234,
      "avg_reward": 0.87,
      "std_reward": 0.12,
      "eval_success_rate": 0.78
    },
    "cost": {
      "tokens_consumed": 5000000,
      "cost_usd": 12.50,
      "budget_remaining_usd": 87.50
    },
    "best_checkpoint": "checkpoint-job456-iter140",
    "estimated_completion_time": "2025-10-16T08:00:00Z"
  },
  "id": 1
}
```

**Error Handling:**

- `404` Job not found
- `403` User not authorized to view job

---

#### 3. `training.cancel` - Cancel Training Job

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Cancel a running or queued training job

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.cancel",
  "params": {
    "job_id": "uuid (required)"
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "uuid",
    "status": "cancelled",
    "cancelled_at": "2025-10-15T12:30:00Z",
    "checkpoint_saved": true
  },
  "id": 1
}
```

**Error Handling:**

- `404` Job not found
- `409` Job already completed or cancelled
- `403` User not authorized to cancel job

---

#### 4. `training.evaluate` - Evaluate Trained Agent

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Run evaluation on trained agent with held-out queries

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.evaluate",
  "params": {
    "agent_id": "string (required)",
    "eval_queries": ["query1", "query2", ...],
    "checkpoint_id": "uuid (optional, default: best)"
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "metrics": {
      "success_rate": 0.82,
      "avg_reward": 0.91,
      "avg_steps": 4.2,
      "tool_usage_accuracy": 0.89
    },
    "baseline_comparison": {
      "baseline_success_rate": 0.70,
      "improvement_percent": 17.1,
      "p_value": 0.003,
      "statistically_significant": true
    },
    "evaluation_time_seconds": 45
  },
  "id": 1
}
```

**Error Handling:**

- `404` Agent or checkpoint not found
- `400` Invalid eval queries format

---

#### 5. `training.export_trajectories` - Export Trajectory Data

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Export trajectory data for debugging and analysis

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.export_trajectories",
  "params": {
    "job_id": "uuid (required)",
    "filters": {
      "success_only": false,
      "min_reward": 0.5,
      "limit": 1000
    }
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "trajectories": [
      {
        "trajectory_id": "uuid",
        "query": "string",
        "steps": [
          {"state": {}, "action": {}, "result": {}, "timestamp": "..."}
        ],
        "reward": 0.85,
        "advantage": 0.12,
        "success": true
      }
    ],
    "total_count": 500,
    "exported_count": 500
  },
  "id": 1
}
```

**Error Handling:**

- `404` Job not found
- `403` User lacks data export permission
- `413` Export size limit exceeded (>10,000 trajectories)

---

#### 6. `training.list_jobs` - List Training Jobs

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** List training jobs with filtering and pagination

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.list_jobs",
  "params": {
    "agent_id": "string (optional)",
    "status": "running (optional)",
    "limit": 50,
    "offset": 0
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "jobs": [
      {
        "job_id": "uuid",
        "agent_id": "string",
        "status": "running",
        "progress_percent": 45,
        "created_at": "2025-10-15T10:00:00Z",
        "cost_usd": 15.75
      }
    ],
    "total_count": 127,
    "limit": 50,
    "offset": 0
  },
  "id": 1
}
```

**Error Handling:**

- `400` Invalid filter parameters
- `403` User not authorized to list jobs

---

### Security (from Research)

**Based on:** `docs/research/flow-based-optimization.md` + AgentCore security patterns

#### Authentication/Authorization

**Approach:** JWT-based authentication (existing AgentCore pattern)

- **Implementation:** Reuse existing `agentcore.a2a_protocol.security_service` JWT validation
- **Pattern:** Middleware validates JWT on all training API endpoints
- **Standards:** OWASP authentication best practices, JWT RS256 signing

**Authorization Rules:**

- **training:start** - Platform operators only (role: `operator` or `admin`)
- **training:view** - Agent developers + operators (role: `developer`, `operator`, `admin`)
- **training:cancel** - Job owner or operators
- **training:export** - Users with `data_export` permission

**Implementation Example:**

```python
from agentcore.a2a_protocol.services.security_service import require_permission

@register_jsonrpc_method("training.start_grpo")
@require_permission("training:start")
async def start_grpo_training(request: JsonRpcRequest) -> dict[str, Any]:
    # Only users with training:start permission can call
    ...
```

#### Secrets Management

**Strategy:** Environment variables + Portkey API key rotation

- **Pattern:** Use `pydantic-settings` to load secrets from environment (existing pattern)
- **Rotation:** Portkey API keys rotated every 90 days (automated via Portkey dashboard)
- **Storage:** Secrets stored in Kubernetes secrets (production), `.env` file (development)

**Configuration:**

```python
# src/agentcore/training/config.py
from pydantic_settings import BaseSettings

class TrainingConfig(BaseSettings):
    portkey_api_key: str = Field(..., env="PORTKEY_API_KEY")
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(..., env="REDIS_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

#### Data Protection

**Encryption in Transit:** TLS/SSL for all external communication

- **Implementation:** Existing FastAPI server with TLS termination at load balancer
- **Standard:** TLS 1.2+ with modern cipher suites

**Encryption at Rest:** PostgreSQL transparent data encryption

- **Implementation:** Use PostgreSQL `pgcrypto` extension for sensitive trajectory data
- **Scope:** Encrypt trajectory `steps` JSONB column (may contain PII)

**PII Handling:** Data sanitization before storage

- **Approach:** Strip PII from trajectories before persistence
- **Implementation:** Configurable PII detection + redaction (email, phone, SSN patterns)
- **Compliance:** GDPR right to erasure (delete trajectories containing user PII)

**Example:**

```python
async def sanitize_trajectory(trajectory: Trajectory) -> Trajectory:
    """Remove PII from trajectory before storage."""
    for step in trajectory.steps:
        # Redact email addresses
        step["action"] = re.sub(r'\S+@\S+', '[EMAIL]', str(step["action"]))
        step["result"] = re.sub(r'\S+@\S+', '[EMAIL]', str(step["result"]))
    return trajectory
```

#### Security Testing

**Approach:** SAST + penetration testing

- **SAST:** Bandit security linter (existing in pyproject.toml)
- **Dependency Scanning:** `safety` for known vulnerabilities
- **Penetration Testing:** Manual testing of budget bypass, injection attacks

**Tools:**

- Bandit (existing) - Python security linter
- safety (existing) - Dependency vulnerability scanner
- OWASP ZAP - Web application security testing (Phase 2)

#### Compliance

**GDPR Requirements:**

- **Right to Erasure:** Delete trajectories containing user data on request
- **Data Minimization:** Only store necessary trajectory data
- **Retention Policy:** Auto-delete trajectories after 30 days (configurable)

**SOC2 Requirements:**

- **Audit Logging:** Log all training job creations, cancellations, exports
- **Access Control:** RBAC for training operations
- **Data Integrity:** Checksum verification for checkpoints

---

### Performance (from Research)

**Based on:** `docs/research/flow-based-optimization.md` performance section

#### Performance Targets (from Research)

| Metric | Target | Rationale from Research |
|--------|--------|------------------------|
| **Response Time** | <200ms (p95) | Training API status queries; research: non-blocking job submission |
| **Throughput** | >100 concurrent jobs | Research: multi-user platform with parallel training |
| **Trajectory Generation** | <2x baseline execution time | Research: parallel execution overhead; 8 trajectories in <30s target |
| **Policy Update Latency** | <5s per batch | Research: batch updates enable efficient training |
| **Resource Usage** | <512MB per worker process | Research: 4-8 GPUs for efficient training (deferred to Phase 3) |

#### Caching Strategy

**Approach:** Multi-level caching (Redis for job state, in-memory for rewards)

**Redis Caching:**

- **Job Status:** Cache in Redis with 30-second TTL
- **Budget State:** Cache in Redis, updated on each LLM call
- **Pattern:** Read-through cache for frequently accessed data

**In-Memory Caching:**

- **Reward Functions:** Cache compiled reward functions (avoid re-parsing)
- **Agent Metadata:** Cache agent configuration during training iteration

**Example:**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_reward_function(reward_type: str) -> Callable:
    """Cache reward functions to avoid re-compilation."""
    if reward_type == "outcome":
        return outcome_based_reward
    elif reward_type == "shaped":
        return shaped_reward_function
    ...
```

**TTL Strategy:**

- **Job Status:** 30 seconds (balance freshness vs load)
- **Budget State:** Real-time (no TTL, update on each cost increment)
- **Reward Functions:** Indefinite (immutable after creation)

**Invalidation:**

- **Event-Based:** Invalidate job status cache on job state change
- **Time-Based:** TTL-based expiration for less critical data

#### Database Optimization

**Indexing Strategy:**

- **Covered Indexes:** Create indexes covering common query patterns
- **GIN Index:** JSONB trajectory steps for flexible querying
- **Composite Index:** `(agent_id, validation_score DESC)` for best checkpoint queries

**Query Patterns:**

- **N+1 Prevention:** Eager load trajectories with job metadata
- **Batching:** Bulk insert trajectories (100 per batch) for performance

**Connection Pooling:**

```python
# SQLAlchemy async engine with connection pool
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,  # Max 20 connections
    max_overflow=10,  # Allow 10 temporary connections
    pool_pre_ping=True,  # Verify connections before use
    echo=False  # Disable SQL logging in production
)
```

**Partitioning:**

- **Phase 2:** Partition `trajectories` table by `created_at` (monthly partitions)
- **Rationale:** Research estimates 100GB-1TB growth; partitioning improves query performance

#### Scaling Strategy

**Horizontal Scaling:**

- **Workers:** Scale worker pool based on Redis queue depth
- **Load Balancing:** Round-robin distribution of API requests across multiple servers
- **Auto-scaling Triggers:**
  - Queue depth >100 jobs → scale up workers
  - Queue depth <10 jobs for 5 minutes → scale down workers

**Vertical Scaling:**

- **Worker Resources:** 2 vCPU, 4GB RAM per worker (Phase 1)
- **Scale Up:** 4 vCPU, 8GB RAM if trajectory generation becomes CPU-bound

**Auto-scaling Configuration (Kubernetes HPA):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: training-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: training-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: redis_queue_depth
        selector:
          matchLabels:
            queue: training_jobs
      target:
        type: AverageValue
        averageValue: "50"  # Target: 50 jobs per worker
```

**Performance Monitoring:**

- **Metrics:** Prometheus metrics for trajectory generation time, policy update latency, queue depth
- **Dashboards:** Grafana dashboard for real-time performance tracking
- **Alerts:** Alert on p95 latency >2x target, queue depth >500 jobs

---

## 5. Development Setup

### Required Tools and Versions

**Core Requirements:**

- **Python:** 3.12+ (existing stack requirement)
- **UV:** Latest (existing package manager)
- **Docker:** 24.0+ (existing for containerization)
- **PostgreSQL:** 15+ (existing database)
- **Redis:** 7.0+ (existing job queue)
- **Git:** 2.40+

**Development Tools:**

- **pytest:** 8.4+ with pytest-asyncio 1.2+ (existing test framework)
- **Ruff:** Latest (existing linter)
- **mypy:** 1.7+ (existing type checker)
- **Alembic:** 1.13+ (existing database migrations)
- **httpx:** 0.28+ (existing HTTP client for testing)

### Local Environment Setup

**1. Clone and Install Dependencies:**

```bash
# Clone repository
git clone <repo-url>
cd agentcore

# Install dependencies with UV
uv sync

# Install development dependencies
uv sync --group dev
```

**2. Docker Compose Infrastructure:**

```yaml
# docker-compose.dev.yml (extend existing)
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: agentcore_dev
      POSTGRES_USER: agentcore
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  # Training worker (new)
  training-worker:
    build: .
    command: uv run python -m agentcore.training.worker
    environment:
      DATABASE_URL: postgresql+asyncpg://agentcore:dev_password@postgres:5432/agentcore_dev
      REDIS_URL: redis://redis:6379/0
      PORTKEY_API_KEY: ${PORTKEY_API_KEY}
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
  redis_data:
```

**3. Environment Variables:**

```bash
# .env.development
DATABASE_URL=postgresql+asyncpg://agentcore:dev_password@localhost:5432/agentcore_dev
REDIS_URL=redis://localhost:6379/0
PORTKEY_API_KEY=<your-portkey-key>

# Training Configuration
TRAINING_MAX_CONCURRENT_JOBS=10
TRAINING_DEFAULT_BUDGET_USD=100
TRAINING_CHECKPOINT_INTERVAL=10
TRAINING_TRAJECTORY_RETENTION_DAYS=30
```

**4. Database Migrations:**

```bash
# Create training tables
uv run alembic revision --autogenerate -m "Add training infrastructure"
uv run alembic upgrade head

# Verify tables created
psql $DATABASE_URL -c "\dt"
```

**5. Start Development Environment:**

```bash
# Start infrastructure
docker compose -f docker-compose.dev.yml up -d postgres redis

# Start AgentCore API server (terminal 1)
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# Start training worker (terminal 2)
uv run python -m agentcore.training.worker

# Verify services
curl http://localhost:8001/health
```

### CI/CD Pipeline Requirements

**GitHub Actions Workflow:**

```yaml
name: Training Module CI

on:
  push:
    paths:
      - 'src/agentcore/training/**'
      - 'tests/training/**'
  pull_request:
    paths:
      - 'src/agentcore/training/**'
      - 'tests/training/**'

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        ports:
          - 5432:5432

      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --group dev

      - name: Run migrations
        run: uv run alembic upgrade head
        env:
          DATABASE_URL: postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db

      - name: Run tests
        run: uv run pytest tests/training/ --cov=src/agentcore/training --cov-fail-under=90

      - name: Type check
        run: uv run mypy src/agentcore/training

      - name: Lint
        run: uv run ruff check src/agentcore/training tests/training

      - name: Security scan
        run: uv run bandit -r src/agentcore/training
```

### Testing Framework and Coverage Targets

**Test Structure:**

```
tests/
├── training/
│   ├── unit/
│   │   ├── test_trajectory_collector.py
│   │   ├── test_grpo_trainer.py
│   │   ├── test_reward_engine.py
│   │   └── test_policy_updater.py
│   ├── integration/
│   │   ├── test_training_api.py
│   │   ├── test_job_manager.py
│   │   └── test_end_to_end_training.py
│   └── performance/
│       ├── test_parallel_trajectories.py
│       └── test_database_performance.py
```

**Coverage Targets:**

- **Overall:** 90% (existing AgentCore requirement)
- **Critical Paths:** 100% (GRPO algorithm, budget enforcement, policy updates)
- **Integration Tests:** 80% (API endpoints, job lifecycle)

**Test Configuration:**

```toml
# pyproject.toml (extend existing)
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
asyncio_mode = "auto"
addopts = "--cov=src --cov-report=term-missing --cov-fail-under=90"

# Training-specific markers
markers = [
    "training: Training module tests",
    "slow: Slow tests (>5 seconds)",
    "integration: Integration tests requiring external services"
]
```

**Example Test:**

```python
# tests/training/unit/test_trajectory_collector.py
import pytest
from agentcore.training.trajectory import TrajectoryCollector

@pytest.mark.asyncio
async def test_collect_multiple_trajectories(mock_agent):
    """Test parallel trajectory collection."""
    collector = TrajectoryCollector()

    trajectories = await collector.collect(
        agent=mock_agent,
        query="What is 2+2?",
        n_trajectories=8
    )

    assert len(trajectories) == 8
    assert all(t.query == "What is 2+2?" for t in trajectories)
    assert all(len(t.steps) > 0 for t in trajectories)
```

---

## 6. Risk Management

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Training cost overruns** | High | High | Strict budget enforcement via Portkey cost tracking; pre-flight cost estimation; real-time monitoring with abort at threshold; budget alerts at 75%, 90% |
| **Performance degradation of production agents** | High | Medium | Training isolation (separate worker processes); A/B testing before production deployment; explicit checkpoint promotion step; rollback capability |
| **Database scaling issues (trajectory storage growth)** | Medium | Medium | Implement auto-archival to S3 after 30 days; JSONB compression; table partitioning by month (Phase 2); monitor storage growth weekly |
| **Training convergence failure** | Medium | Medium | Hyperparameter tuning guide; early stopping on validation plateau; expert consultation for advanced cases; fallback to baseline agent |
| **Integration complexity with agent runtime** | High | Medium | Phased rollout (start with ReAct agents only); comprehensive integration tests; feature flags for gradual enablement; extensive documentation |
| **Policy update corrupts agent behavior** | High | Low | Checkpoint versioning (keep best 5); validation before promotion; A/B testing against baseline; automated rollback on metrics regression |
| **Redis job queue failure** | Medium | Low | Redis cluster with replication; job state persisted to PostgreSQL; worker restarts resume from last checkpoint; monitoring + alerts |
| **Portkey API rate limits or downtime** | Medium | Low | Multiple LLM provider support via Portkey fallback; local caching of recent responses; graceful degradation (pause training, resume later) |
| **Security breach via trajectory data** | High | Low | PII sanitization before storage; encryption at rest for sensitive data; RBAC for data export; audit logging for all access |
| **Worker process crashes** | Medium | Medium | Kubernetes health checks and auto-restart; checkpoint saves every N iterations; job resumption from last checkpoint; monitoring + alerts |

**Risk Mitigation Priorities:**

1. **P0:** Budget enforcement, training isolation, checkpoint versioning
2. **P1:** Database scaling, integration complexity, security
3. **P2:** Convergence failure, Redis failure, Portkey downtime

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal:** Core training infrastructure with basic GRPO implementation

**Deliverables:**

**Week 1-2: Infrastructure Setup**

- Database schema and migrations (`training_jobs`, `trajectories`, `policy_checkpoints`)
- Redis job queue setup and worker skeleton
- Pydantic models (`GRPOConfig`, `Trajectory`, `TrainingJob`)
- Repository pattern implementation (`TrajectoryRepository`, `TrainingJobRepository`)
- Development environment setup (Docker Compose, `.env` configuration)

**Week 3: Trajectory Collection**

- `TrajectoryCollector` implementation with async/await parallel execution
- Trajectory recorder middleware (wrap agent execution)
- Integration with existing agent runtime
- Unit tests for trajectory collection (target: 95% coverage)

**Week 4: GRPO Trainer**

- `GRPOTrainer` implementation (reward refinement, policy gradients)
- `RewardEngine` with outcome-based + shaped rewards
- `PolicyUpdater` for prompt-based policy updates
- Training iteration loop
- Unit tests for GRPO algorithm (target: 100% coverage)

**Success Criteria:**

- ✅ Can generate 8 trajectories in parallel for test query
- ✅ GRPO trainer computes valid advantages and updates policies
- ✅ Training job can run end-to-end (1 iteration)
- ✅ Unit tests pass with 90%+ coverage

**Risks & Mitigation:**

- **Risk:** Agent runtime integration complexity
- **Mitigation:** Start with simple ReAct agents; use wrapper pattern to avoid modifying agent code

---

### Phase 2: Core Features (Weeks 5-6)

**Goal:** Complete training job management with evaluation and monitoring

**Deliverables:**

**Week 5: Training Job API**

- JSON-RPC endpoints (`training.start_grpo`, `training.get_status`, `training.cancel`)
- `TrainingJobManager` for job lifecycle management
- Background worker pool consuming from Redis queue
- Job status tracking in Redis
- Integration tests for API endpoints

**Week 6: Evaluation & Budget**

- `EvaluationFramework` for held-out query testing
- Budget enforcement with Portkey cost tracking
- `CheckpointManager` for save/restore with versioning
- Prometheus metrics export
- End-to-end integration test (full training job)

**Success Criteria:**

- ✅ Training jobs run in background, queryable via API
- ✅ Budget enforcement prevents cost overruns (tested)
- ✅ Evaluation shows statistically significant improvement (p<0.05)
- ✅ Checkpoints save/restore successfully
- ✅ Metrics exported to Prometheus

**Risks & Mitigation:**

- **Risk:** Budget enforcement bypassed
- **Mitigation:** Unit tests for budget decorator; integration test with mock Portkey API hitting limit

---

### Phase 3: Hardening (Weeks 7-8)

**Goal:** Production readiness with advanced features and monitoring

**Deliverables:**

**Week 7: Advanced Features**

- Multi-step credit assignment (temporal difference)
- Training job scheduling and queuing
- Advanced reward shaping strategies
- `training.evaluate` and `training.export_trajectories` endpoints
- Performance tests (100 concurrent jobs, parallel trajectory latency)

**Week 8: Production Readiness**

- Comprehensive documentation (API docs, developer guide, operational runbook)
- Security testing (budget bypass attempts, PII leakage, injection attacks)
- Load testing validation (100+ concurrent jobs, database performance)
- Monitoring dashboards (Grafana) and alerting setup
- Deployment to staging environment

**Success Criteria:**

- ✅ Multi-step credit assignment improves convergence speed (validated)
- ✅ Job queue handles 100+ concurrent jobs without degradation
- ✅ Documentation enables developers to start training independently
- ✅ Security tests pass (no critical vulnerabilities)
- ✅ Load tests meet performance targets (p95 <2x latency)

**Risks & Mitigation:**

- **Risk:** Performance degradation under load
- **Mitigation:** Load testing early in Phase 3; horizontal scaling via Kubernetes HPA

---

### Phase 4: Launch (Week 9+)

**Goal:** Production deployment and post-launch support

**Deliverables:**

**Week 9: Deployment**

- Production deployment (Kubernetes with HPA)
- Monitoring and alerting configured (Prometheus + Grafana)
- Rollback procedures tested
- A/B testing setup (trained agents vs baseline)

**Week 10+: Post-Launch**

- Monitor training job success rate, cost, performance
- Collect user feedback on API usability
- Optimize based on production metrics
- Plan Phase 2 enhancements (ACE integration, neural policy updates)

**Success Criteria:**

- ✅ Production deployment successful (zero downtime)
- ✅ First 10 training jobs complete successfully
- ✅ User feedback positive (3+ developers successfully train agents)
- ✅ Cost and performance within targets

**Risks & Mitigation:**

- **Risk:** Production incidents during deployment
- **Mitigation:** Blue-green deployment; feature flags for gradual rollout; 24/7 on-call rotation

---

## 8. Quality Assurance

### Testing Strategy

**Unit Tests (Target: 95% coverage):**

**Critical Paths:**

- Trajectory collection (parallel execution, timeout handling, error recovery)
- GRPO algorithm (reward normalization, advantage computation, policy gradients)
- Reward engine (outcome-based, shaped rewards, custom reward functions)
- Policy updater (prompt updates, checkpoint creation)
- Budget enforcement (pre-flight checks, real-time monitoring, abort logic)

**Test Framework:** pytest with pytest-asyncio for async tests

**Example Critical Path Test:**

```python
@pytest.mark.asyncio
async def test_grpo_reward_refinement():
    """Test GRPO reward normalization."""
    trajectories = [
        Trajectory(reward=0.5),
        Trajectory(reward=0.8),
        Trajectory(reward=0.2)
    ]

    refined = refine_rewards(trajectories)

    # Check normalization: mean=0, std=1
    advantages = [t.advantage for t in refined]
    assert abs(np.mean(advantages)) < 0.01
    assert abs(np.std(advantages) - 1.0) < 0.01

    # Check ordering preserved
    assert refined[1].advantage > refined[0].advantage > refined[2].advantage
```

---

**Integration Tests:**

**Scenarios:**

1. **End-to-End Training Job:** Start job → Worker picks up → Trajectories generated → Policy updated → Checkpoint saved → Job completes
2. **Budget Enforcement:** Start job with $10 budget → Generate expensive trajectories → Abort when budget exceeded → Checkpoint saved
3. **API Endpoints:** Call all JSON-RPC endpoints with valid/invalid inputs → Verify responses
4. **Job Cancellation:** Start job → Cancel mid-execution → Verify graceful shutdown → Checkpoint saved
5. **Checkpoint Resume:** Start job → Crash worker → Restart worker → Resume from checkpoint → Job completes

**Test Framework:** pytest with testcontainers for Redis

**Example Integration Test:**

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_training_job(test_agent, test_db, redis_container):
    """Test complete training job lifecycle."""
    # Create training job
    job = await training_manager.create_job(
        agent_id=test_agent.id,
        training_data=[{"query": "2+2?", "expected": "4"}],
        config=GRPOConfig(n_iterations=10, batch_size=1)
    )

    # Start worker
    worker_task = asyncio.create_task(training_worker.run())

    # Wait for completion (with timeout)
    await asyncio.wait_for(wait_for_job_completion(job.job_id), timeout=60)

    # Verify job completed
    final_status = await training_manager.get_status(job.job_id)
    assert final_status["status"] == "completed"
    assert final_status["progress"]["percent_complete"] == 100

    # Verify checkpoint saved
    checkpoints = await checkpoint_manager.list_checkpoints(test_agent.id)
    assert len(checkpoints) > 0

    # Cleanup
    worker_task.cancel()
```

---

**End-to-End Tests:**

**User Flows:**

1. **Platform Operator Training Flow:** Login → Create training job via API → Monitor progress → Job completes → Evaluate trained agent → Deploy to production
2. **Agent Developer Debugging Flow:** Training job fails → Export trajectories → Analyze failure patterns → Adjust reward function → Retry training
3. **Budget Management Flow:** Set budget limit → Start training → Receive budget alert → Training auto-stops → Review cost breakdown

**Test Framework:** pytest with real AgentCore instance

---

**Performance Tests:**

**Load Scenarios:**

- 100+ concurrent training jobs
- 1000 trajectories generated in parallel
- Database write performance (>100 trajectories/sec)
- API endpoint response time (<200ms p95)

**SLA Targets:**

- Training job throughput: 100+ concurrent jobs
- Trajectory generation: <2x baseline execution time
- Policy update latency: <5s per batch
- Database write performance: >100 trajectories/sec

**Test Framework:** pytest-benchmark, locust for load testing

**Example Performance Test:**

```python
@pytest.mark.performance
def test_parallel_trajectory_generation(benchmark, test_agent):
    """Benchmark parallel trajectory collection."""
    collector = TrajectoryCollector()

    result = benchmark(
        lambda: asyncio.run(collector.collect(test_agent, "test query", n_trajectories=8))
    )

    # Verify performance target: <30s for 8 trajectories
    assert result < 30.0
```

---

### Code Quality Gates

**Pre-Commit Hooks:**

```yaml
# .pre-commit-config.yaml (extend existing)
repos:
  - repo: local
    hooks:
      - id: ruff-training
        name: Ruff linter (training module)
        entry: uv run ruff check src/agentcore/training tests/training
        language: system
        pass_filenames: false

      - id: mypy-training
        name: Type check (training module)
        entry: uv run mypy src/agentcore/training
        language: system
        pass_filenames: false

      - id: pytest-training
        name: Unit tests (training module)
        entry: uv run pytest tests/training/unit --cov-fail-under=90
        language: system
        pass_filenames: false
```

**CI/CD Quality Gates:**

1. **Lint:** Ruff passes with zero errors
2. **Type Check:** mypy strict mode passes with zero errors
3. **Unit Tests:** 90%+ coverage, all tests pass
4. **Security Scan:** Bandit passes (no high/critical findings)
5. **Integration Tests:** All integration tests pass
6. **Performance Tests:** P95 latency meets targets

**Pull Request Requirements:**

- All quality gates pass
- Code review approval from 1+ maintainer
- Documentation updated (if API changes)
- CHANGELOG entry added

---

### Deployment Verification Checklist

**Pre-Deployment:**

- [ ] All tests pass (unit, integration, performance)
- [ ] Database migrations tested in staging
- [ ] Configuration reviewed (budget limits, worker count, etc.)
- [ ] Monitoring dashboards configured
- [ ] Alerting rules configured
- [ ] Rollback procedure documented

**Deployment:**

- [ ] Blue-green deployment initiated
- [ ] Health checks pass (API, worker, database)
- [ ] Smoke tests pass (create job, query status)
- [ ] Metrics flowing to Prometheus
- [ ] No error spikes in logs

**Post-Deployment:**

- [ ] Monitor training job success rate (target: >99%)
- [ ] Monitor API latency (target: p95 <200ms)
- [ ] Monitor database performance (target: >100 writes/sec)
- [ ] Monitor cost (target: stay within budget)
- [ ] Collect user feedback (target: 3+ successful users in week 1)

---

### Monitoring and Alerting Setup

**Prometheus Metrics:**

```python
# src/agentcore/training/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Job lifecycle metrics
training_jobs_created = Counter(
    "training_jobs_created_total",
    "Total training jobs created"
)

training_jobs_completed = Counter(
    "training_jobs_completed_total",
    "Total training jobs completed",
    ["status"]  # completed, failed, cancelled
)

# Performance metrics
trajectory_generation_duration = Histogram(
    "trajectory_generation_seconds",
    "Time to generate single trajectory",
    buckets=[1, 5, 10, 30, 60, 120]
)

policy_update_duration = Histogram(
    "policy_update_seconds",
    "Time to update policy for batch",
    buckets=[0.5, 1, 2, 5, 10]
)

# Resource metrics
active_training_jobs = Gauge(
    "active_training_jobs",
    "Number of currently running training jobs"
)

training_budget_usage = Gauge(
    "training_budget_usage_usd",
    "Training budget usage in USD",
    ["job_id"]
)
```

**Grafana Dashboard:**

**Panel 1: Job Throughput**

- Metric: `rate(training_jobs_completed_total[5m])`
- Alert: Job completion rate <1/minute for 10 minutes

**Panel 2: Performance**

- Metric: `histogram_quantile(0.95, trajectory_generation_seconds)`
- Alert: p95 trajectory generation time >60s for 5 minutes

**Panel 3: Budget Usage**

- Metric: `sum(training_budget_usage_usd) by (job_id)`
- Alert: Job exceeds budget (should never happen - enforced in code)

**Panel 4: Error Rate**

- Metric: `rate(training_jobs_completed_total{status="failed"}[5m])`
- Alert: Failure rate >5% for 10 minutes

**Alerting Rules:**

```yaml
# prometheus_alerts.yml
groups:
  - name: training
    interval: 30s
    rules:
      - alert: TrainingJobHighFailureRate
        expr: rate(training_jobs_completed_total{status="failed"}[5m]) > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Training job failure rate >5% for 10 minutes"

      - alert: TrainingPerformanceDegradation
        expr: histogram_quantile(0.95, trajectory_generation_seconds) > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 trajectory generation time >60s"

      - alert: TrainingWorkerDown
        expr: up{job="training-worker"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Training worker process down"
```

---

## ⚠️ Error Handling & Edge Cases

**From:** Feature request technical considerations + Research findings

### Error Scenarios (from Research)

#### 1. **Budget Exceeded During Training**

**Cause:** Training job consumes more tokens than allocated budget

**Impact:** Cost overrun if not caught; user charged beyond limit

**Handling (from Research - Budget Enforcement):**

```python
@enforce_budget("training:job:{job_id}")
async def generate_trajectory_batch(...):
    # Pre-flight check
    remaining = await budget_enforcer.check_budget(job_id)
    if remaining <= 0:
        raise BudgetExceededError(f"Budget exhausted for job {job_id}")

    # Execute batch
    trajectories = await collector.collect(...)

    # Post-execution: update usage (handled by decorator)
    return trajectories
```

**Recovery:** Save checkpoint immediately; mark job as `budget_exceeded`; notify user

**User Experience:** Clear error message with cost breakdown, link to increase budget

---

#### 2. **Agent Execution Timeout**

**Cause:** Agent gets stuck in infinite loop or extremely long trajectory

**Impact:** Training stalls; resources wasted

**Handling:**

```python
async def generate_single_trajectory(agent, query):
    try:
        async with asyncio.timeout(300):  # 5 minute timeout
            trajectory = await agent.execute(query)
            return trajectory
    except asyncio.TimeoutError:
        logger.warning(f"Trajectory generation timeout for {query}")
        # Return partial trajectory with negative reward
        return Trajectory(
            query=query,
            steps=[],
            reward=-1.0,  # Penalize timeout
            success=False,
            error="timeout"
        )
```

**Recovery:** Continue with other trajectories; timeout treated as failure (negative reward)

**User Experience:** Timeout counted in metrics; user can adjust timeout in config

---

#### 3. **Database Connection Failure**

**Cause:** PostgreSQL temporarily unavailable or connection pool exhausted

**Impact:** Cannot save trajectories; training progress lost

**Handling:**

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3)
)
async def save_trajectories(trajectories: list[Trajectory]):
    """Save trajectories with retry on failure."""
    async with get_session() as session:
        repo = TrajectoryRepository(session)
        await repo.bulk_insert(trajectories)
```

**Recovery:** Retry with exponential backoff (3 attempts); if all fail, cache trajectories in Redis temporarily

**User Experience:** Training continues; temporary failures transparent to user

---

#### 4. **Worker Process Crash**

**Cause:** Out-of-memory, unhandled exception, Kubernetes node failure

**Impact:** Training job interrupted; progress lost if no checkpoint

**Handling:**

- Checkpoints saved every N iterations (default: 10)
- Job state persisted in Redis (survives worker restart)
- Kubernetes restarts worker automatically
- Worker resumes from last checkpoint on startup

**Recovery:**

```python
async def resume_training_job(job_id: str):
    """Resume training job from last checkpoint."""
    job = await job_repo.get_by_id(job_id)

    # Find last checkpoint
    checkpoint = await checkpoint_manager.get_latest(job.agent_id)

    if checkpoint:
        logger.info(f"Resuming job {job_id} from iteration {checkpoint.iteration}")
        agent = await load_agent_from_checkpoint(checkpoint)
        start_iteration = checkpoint.iteration
    else:
        logger.info(f"Starting job {job_id} from beginning (no checkpoint)")
        agent = await load_agent(job.agent_id)
        start_iteration = 0

    # Continue training
    await run_training_iterations(job, agent, start_iteration)
```

**User Experience:** Training resumes automatically; progress preserved; small delay for recovery

---

### Edge Cases (from Feature Request & Research)

**Identified in Research:**

#### Edge Case 1: All Trajectories Have Same Reward

**Scenario:** Agent performs identically on all queries in batch (e.g., all succeed or all fail)

**Detection:** `std_reward == 0` in reward refinement

**Handling:**

```python
def refine_rewards(trajectories: list[Trajectory]) -> list[Trajectory]:
    std_reward = np.std([t.reward for t in trajectories])

    if std_reward == 0:
        logger.warning("All trajectories have same reward - no learning signal")
        # Set all advantages to 0 (no updates)
        for t in trajectories:
            t.advantage = 0.0
        return trajectories

    # Normal normalization
    ...
```

**Impact:** No policy updates for this iteration (skip update step)

**Testing Approach:** Unit test with identical rewards; verify no policy updates

---

#### Edge Case 2: Single Training Query

**Scenario:** User provides only 1 training query (batch_size=1)

**Detection:** `len(training_data) == 1`

**Handling:**

- Allow (valid use case for debugging)
- Warning: "Single query training may not converge - recommend ≥100 queries"
- Use same query for all iterations with different trajectory samples

**Testing Approach:** Integration test with single query; verify training completes

---

#### Edge Case 3: Zero Budget Allocated

**Scenario:** User sets `max_budget_usd=0` (free trial or testing)

**Detection:** `config.max_budget_usd == 0`

**Handling:**

- Reject job creation with error: "Budget must be positive (minimum $1)"
- Alternative: Allow with warning "Zero budget - training will stop after first LLM call"

**Testing Approach:** API test with zero budget; verify rejection

---

#### Edge Case 4: Agent Not Found

**Scenario:** Training job references non-existent agent_id

**Detection:** `await agent_repo.get_by_id(agent_id)` returns None

**Handling:**

```python
@register_jsonrpc_method("training.start_grpo")
async def start_grpo_training(request: JsonRpcRequest):
    agent_id = request.params["agent_id"]

    # Validate agent exists
    async with get_session() as session:
        agent = await AgentRepository(session).get_by_id(agent_id)
        if not agent:
            raise JsonRpcError(
                code=404,
                message=f"Agent {agent_id} not found"
            )

    # Continue with job creation
    ...
```

**Testing Approach:** API test with invalid agent_id; verify 404 error

---

### Input Validation

**Validation Rules:**

**Training Configuration:**

- `n_iterations`: 1 ≤ value ≤ 10,000 (enforce via Pydantic)
- `batch_size`: 1 ≤ value ≤ 128
- `n_trajectories_per_query`: 1 ≤ value ≤ 32
- `learning_rate`: 0 < value ≤ 0.1
- `max_budget_usd`: value > 0

**Training Data:**

- Must be non-empty list
- Each item must have `query` (non-empty string)
- Maximum 10,000 queries per job (prevent abuse)

**Pydantic Validation:**

```python
class GRPOConfig(BaseModel):
    n_iterations: int = Field(default=1000, ge=1, le=10000)
    batch_size: int = Field(default=16, ge=1, le=128)
    n_trajectories_per_query: int = Field(default=8, ge=1, le=32)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.1)
    max_budget_usd: float = Field(default=100, gt=0)

    @field_validator("max_budget_usd")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Budget must be positive (minimum $1)")
        return v
```

**Sanitization:**

**XSS Prevention:** Escape HTML in trajectory data before displaying in UI
**SQL Injection Prevention:** Use parameterized queries (SQLAlchemy handles this)
**Input Normalization:** Trim whitespace from queries, normalize line endings

---

### Graceful Degradation

**Fallback Strategies:**

**Portkey LLM Gateway Unavailable:**

- **Degraded Mode:** Pause training jobs; queue for retry when service restored
- **Fallback:** None (LLM access is critical for training)
- **User Experience:** Job status shows "paused - waiting for LLM service"

**Redis Job Queue Unavailable:**

- **Degraded Mode:** Accept job creation requests; store in PostgreSQL `pending_jobs` table
- **Fallback:** Poll PostgreSQL for pending jobs (less efficient than Redis queue)
- **User Experience:** Higher latency for job pickup; otherwise transparent

**PostgreSQL Database Unavailable:**

- **Degraded Mode:** Cache trajectories in Redis (temporary storage)
- **Fallback:** Flush to PostgreSQL when database restored
- **User Experience:** Training continues; warning "trajectories not persisted yet"
- **Risk:** Data loss if Redis also fails (acceptable for training use case)

**Partial Trajectory Failure:**

- **Scenario:** 7/8 trajectories succeed, 1 fails
- **Handling:** Continue with successful trajectories; compute rewards on subset
- **Impact:** Slightly reduced batch size; training continues

---

### Monitoring & Alerting

**Error Tracking:**

**Tool:** structlog for structured logging + Prometheus for error counters

**Metrics:**

```python
training_errors = Counter(
    "training_errors_total",
    "Total training errors",
    ["error_type", "component"]
)

# Usage
training_errors.labels(error_type="timeout", component="trajectory_collector").inc()
```

**Threshold Alerting:**

- **Budget Exceeded:** Alert immediately (should never happen due to enforcement)
- **High Failure Rate:** Alert if >5% of training jobs fail (10 minute window)
- **Worker Crashes:** Alert if worker restarts >3 times in 5 minutes
- **Database Errors:** Alert if database connection failures >10 in 5 minutes

**Response Procedures:**

1. **Budget Alert:** Investigate budget enforcement logic; check for bypass
2. **Failure Rate Alert:** Review recent failed jobs; identify common failure pattern
3. **Worker Crash Alert:** Check logs for OOM, unhandled exceptions; adjust resource limits
4. **Database Alert:** Check PostgreSQL health; scale connection pool if needed

**Incident Response Plan:**

- **Severity 1 (Critical):** All training jobs failing - on-call responds within 15 minutes
- **Severity 2 (High):** High failure rate or performance degradation - respond within 1 hour
- **Severity 3 (Medium):** Individual job failures - review during business hours

---

## 📚 References & Traceability

### Source Documentation

**Research Analysis:**

- `docs/research/flow-based-optimization.md`
  - GRPO (Group Refined Policy Optimization) methodology
  - Technical implementation patterns (trajectory rollout, reward refinement, policy updates)
  - Performance targets (+15-25% task success rate, +20% tool accuracy)
  - Cost analysis ($25 per 1000 queries, 20-40x ROI)
  - Implementation phases (4 weeks core, 2 weeks evaluation, 2 weeks hardening)

**Specification:**

- `docs/specs/flow-based-optimization/spec.md`
  - 8 functional requirements (FR-1 through FR-8)
  - Non-functional requirements (performance, security, scalability, reliability)
  - User stories (5 stories covering training workflows)
  - Acceptance criteria (definition of done, validation approach)
  - Dependencies (technical stack, integration points, related components)

**Epic Ticket:**

- `.sage/tickets/FLOW-001.md`
  - Epic ticket for Flow-Based Optimization Engine
  - 3 implementation phases with story ticket breakdown
  - Dependencies on Agent Runtime, Task Manager, Database Layer
  - Success criteria and risk assessment

### System Context

**Architecture & Patterns:**

- `docs/agentcore-architecture-and-development-plan.md` - 6-layer system architecture
  - Intelligence Layer: Flow-Based Optimization fits here (alongside DSPy Optimization)
  - Agent Runtime Layer: Multi-philosophy execution (ReAct, CoT, Multi-Agent, Autonomous)
  - Enhanced Core Layer: A2A Protocol (JSON-RPC), Task Management, Orchestration
  - Infrastructure Layer: PostgreSQL, Redis, Portkey, FastAPI

- `docs/breakdown/agent-runtime.md` - Agent execution infrastructure
  - Multi-philosophy agent support (ReAct, CoT, Multi-Agent, Autonomous)
  - Container-based security sandboxing with Docker
  - Tool integration framework with permission-based access control
  - Agent lifecycle management with state persistence

**Tech Stack:**

- `pyproject.toml` - Existing technology stack
  - Python 3.12+ with asyncio
  - FastAPI 0.104+ (JSON-RPC API framework)
  - PostgreSQL with asyncpg (async database driver)
  - Redis 5.0+ (job queue and caching)
  - Pydantic 2.5+ (validation framework)
  - pytest + pytest-asyncio (testing framework, 90% coverage requirement)
  - Portkey AI (LLM gateway with cost tracking)

### Research Citations

**Best Practices Research (from flow-based-optimization.md):**

1. **GRPO Algorithm** (Lines 41-211)
   - Trajectory-based reinforcement learning for agent training
   - Group-based reward normalization reduces variance
   - Policy gradient updates favor high-reward trajectories
   - Supports 8 concurrent trajectories for sample efficiency

2. **Performance Benchmarks** (Lines 317-344)
   - +15-25% task success rate improvement (validated in production)
   - +20% tool usage accuracy improvement
   - +10-15% improvement on complex multi-step tasks
   - 2-3x faster convergence than standard RL

3. **Cost Analysis** (Lines 346-379)
   - Training cost: $25 for 1000 queries (full training run)
   - Monthly savings: $500-1000 through improved agent efficiency
   - ROI: 20-40x within first month of deployment

4. **Implementation Phases** (Lines 432-654)
   - Phase 1: Training Infrastructure (Weeks 1-2) - TrajectoryCollector, GRPO algorithm
   - Phase 2: GRPO Implementation (Week 3) - Reward refinement, policy updates
   - Phase 3: Agent Training API (Week 4) - JSON-RPC endpoints, job management
   - Phase 4: Evaluation & Monitoring (Week 5) - Continuous evaluation, metrics

**Technology Evaluation (from Existing Stack):**

1. **Python asyncio** - Native async/await for parallel trajectory generation
   - Source: Existing AgentCore pattern, `pyproject.toml`
   - Rationale: Efficient I/O concurrency without threading overhead

2. **PostgreSQL with JSONB** - Flexible trajectory storage with query performance
   - Source: Existing AgentCore database, `pyproject.toml`
   - Rationale: Handles 100GB-1TB trajectory data growth (research estimate)

3. **Redis for Job Queue** - Distributed job coordination with pub/sub
   - Source: Existing AgentCore Redis cluster, `pyproject.toml`
   - Rationale: Lighter weight than Celery; sufficient for training coordination

4. **Portkey AI** - Unified LLM gateway with built-in cost tracking
   - Source: Existing AgentCore LLM integration, `pyproject.toml`
   - Rationale: Enables budget enforcement requirement (FR-4 from spec)

**Security Standards (from AgentCore):**

1. **JWT Authentication** - Existing AgentCore security pattern
   - Source: `agentcore.a2a_protocol.security_service`
   - Standard: OWASP authentication best practices

2. **RBAC Authorization** - Role-based access control for training operations
   - Source: Existing AgentCore authorization framework
   - Roles: `operator` (training:start), `developer` (training:view), `admin` (all)

3. **PII Protection** - Data sanitization before trajectory storage
   - Source: GDPR compliance requirements
   - Implementation: Regex-based PII detection + redaction

**Performance Benchmarks (from Research):**

1. **Trajectory Generation** - <2x baseline execution time (research target)
   - Source: flow-based-optimization.md, parallel execution pattern
   - Target: 8 parallel trajectories in <30 seconds

2. **Policy Update** - <5s per batch update (research estimate)
   - Source: flow-based-optimization.md, policy gradient algorithm
   - Implementation: Batch updates for efficiency

3. **Database Performance** - >100 trajectory writes/sec (research requirement)
   - Source: Research estimates 10K trajectories per training run
   - Implementation: Bulk insert with SQLAlchemy, JSONB compression

### Related Components

**Dependencies:**

1. **Agent Runtime** (docs/breakdown/agent-runtime.md)
   - **Relationship:** Flow-Based Optimization uses agent runtime for trajectory execution
   - **Integration:** Trajectory collector wraps existing agent execution
   - **Impact:** Changes to agent execution interface may affect trajectory collection

2. **Task Manager** (agentcore/a2a_protocol/services/task_manager.py)
   - **Relationship:** Tracks individual trajectory executions as tasks
   - **Integration:** Create task record for each trajectory generation
   - **Impact:** Task management overhead adds <10ms per trajectory

3. **Database Layer** (agentcore/a2a_protocol/database/)
   - **Relationship:** Extends database with training-specific tables
   - **Integration:** New tables (training_jobs, trajectories, policy_checkpoints)
   - **Impact:** Database migration required; storage growth 100GB-1TB

4. **JSON-RPC Handler** (agentcore/a2a_protocol/services/jsonrpc_handler.py)
   - **Relationship:** Training API uses JSON-RPC pattern
   - **Integration:** Register training endpoints with decorator
   - **Impact:** No changes to existing endpoints; isolated namespace

**Future Integrations:**

1. **Modular Agent Architecture** (docs/research/modular-agent-architecture.md)
   - **Relationship:** Train modular agents (Planner→Executor→Verifier→Generator)
   - **Integration:** Trajectories capture multi-module execution flows
   - **Timeline:** Phase 2 (after core training infrastructure stable)

2. **ACE Integration** (docs/research/ace-integration-analysis.md)
   - **Relationship:** Dual optimization (GRPO + ACE context evolution)
   - **Integration:** Coordinate GRPO policy updates with ACE context refinement
   - **Timeline:** Phase 3 (advanced feature, deferred pending research validation)

3. **Bounded Context Reasoning** (future spec)
   - **Relationship:** Context-aware trajectory generation
   - **Integration:** Trajectories include context metadata
   - **Timeline:** Future work (after bounded context reasoning implemented)

4. **Memory System** (future spec)
   - **Relationship:** Trajectory data informs memory evolution
   - **Integration:** Extract successful patterns from trajectories for memory storage
   - **Timeline:** Future work (after memory system design complete)

**Potential Conflicts:**

1. **DSPy Optimization** (existing intelligence layer component)
   - **Conflict:** Both modify agent prompts/policies simultaneously
   - **Impact:** Policy updates may interfere; unpredictable behavior
   - **Mitigation:** Coordinate optimization schedules; use feature flags to enable one at a time
   - **Resolution:** Run DSPy prompt optimization first; then GRPO policy training; avoid concurrent updates

2. **Resource Contention** (LLM token limits)
   - **Conflict:** Training jobs consume tokens competing with production workloads
   - **Impact:** Production agents may experience rate limiting
   - **Mitigation:** Token budget limits per training job; separate Portkey API keys for training vs production
   - **Resolution:** Priority queuing (production > training); monitoring + alerts on token usage

---

**Document Status:** Implementation blueprint complete. Ready for `/sage.tasks` breakdown into story tickets.

**Next Steps:**

1. Review plan with engineering team (estimate: 1 hour review session)
2. Update FLOW-001 epic ticket with architecture and dependencies
3. Run `/sage.tasks FLOW-001` to generate story tickets (FLOW-002+)
4. Begin Phase 1 implementation with `/ sage.implement FLOW-002`

**Approval:** Pending technical review and stakeholder sign-off.
