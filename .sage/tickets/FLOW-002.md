# FLOW-002: Database Schema & Models

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** Flow-Based Optimization (FLOW)
**Sprint:** 1
**Estimated Effort:** 5 story points (3-5 days)

---

## Description

Create PostgreSQL database schema for training infrastructure with Alembic migrations. Implement Pydantic models for type-safe configuration and data validation. Create repository pattern for database access.

**Technical Scope:**
- Three new tables: training_jobs, trajectories, policy_checkpoints
- Pydantic models: GRPOConfig, Trajectory, TrainingJob, PolicyCheckpoint
- Repository classes: TrainingJobRepository, TrajectoryRepository, CheckpointRepository
- Database indexes for performance optimization

---

## Acceptance Criteria

- [x] Alembic migration created and runs successfully (`uv run alembic upgrade head`)
- [x] Pydantic models validate correctly (GRPOConfig, Trajectory, TrainingJob)
- [x] Repository classes implement CRUD operations (create, read, update, delete)
- [x] Database indexes created on job_id, agent_id, created_at columns
- [x] JSONB columns for flexible trajectory storage
- [x] Unit tests achieve 95% coverage

---

## Dependencies

- **Parent**: #FLOW-001 (Flow-Based Optimization Engine epic)
- **Blocks**: #FLOW-003, #FLOW-007, #FLOW-014

---

## Context

**Specs:** `docs/specs/flow-based-optimization/spec.md`
**Plans:** `docs/specs/flow-based-optimization/plan.md`
**Tasks:** `docs/specs/flow-based-optimization/tasks.md`

**Database Schema (from plan):**
```sql
CREATE TABLE training_jobs (
    job_id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    ...
);

CREATE TABLE trajectories (
    trajectory_id UUID PRIMARY KEY,
    job_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    steps JSONB NOT NULL,
    ...
);

CREATE TABLE policy_checkpoints (
    checkpoint_id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    policy_data JSONB,
    ...
);
```

---

## Implementation Notes

**Files to Create:**
- `alembic/versions/xxx_add_training_tables.py`
- `src/agentcore/training/models.py`
- `src/agentcore/training/repositories.py`
- `tests/training/unit/test_models.py`
- `tests/training/unit/test_repositories.py`

**Key Patterns:**
- Use async SQLAlchemy with `asyncpg` driver
- Follow existing repository pattern from `agentcore/a2a_protocol/database/repositories.py`
- Use Pydantic `Field()` with validation constraints

---

## Progress

**Owner:** Backend Engineer 1
**Status:** COMPLETED
**Completed:** 2025-10-17
**Notes:**
- Created Alembic migration `e9ba324940b6_add_training_tables.py` with 3 tables
- Implemented Pydantic models in `src/agentcore/training/models.py`
- Implemented SQLAlchemy ORM models in `src/agentcore/training/database_models.py`
- Implemented repository classes in `src/agentcore/training/repositories.py`
- Created comprehensive test suite: 21 unit tests + 4 integration tests (all passing)
- All database indexes created (standard + GIN for JSONB)
- Foreign key cascade delete validated through tests
