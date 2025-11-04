# FLOW-008: Training API Endpoints

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Sprint:** 2
**Effort:** 5 SP

## Dependencies

**Parent:** #FLOW-001
**Requires:**
- #FLOW-007

**Blocks:**
- #FLOW-009

## Context

Specs: `docs/specs/flow-based-optimization/spec.md`
Tasks: `docs/specs/flow-based-optimization/tasks.md` (see FLOW-008 section)

## Owner

Eng-1

## Status

✅ **COMPLETED**

## Implementation Details

**Implemented:** 2025-10-17T05:06:38Z (commit ad47673)
**Verified:** 2025-10-17T08:30:00Z
**Branch:** feature/flow-008
**Tests:** 19/19 passed (100%)

### Deliverables

- ✅ `training.start_grpo` endpoint - Create and start training jobs
- ✅ `training.get_status` endpoint - Get job status and metrics
- ✅ `training.cancel` endpoint - Cancel running training jobs
- ✅ `training.list_jobs` endpoint - List jobs with optional agent_id filter
- ✅ Request/response validation with Pydantic models
- ✅ Comprehensive parameter validation (agent_id, job_id, training_queries)
- ✅ Minimum 100 training queries requirement enforced
- ✅ UUID validation for job_id parameters
- ✅ 19 comprehensive integration tests

### Implementation Approach

**JSON-RPC 2.0 Method Registration:**
- Uses `@register_jsonrpc_method` decorator for automatic registration
- Global `TrainingJobManager` instance with `get_job_manager()` factory
- Error handling with descriptive ValueError messages

**API Features:**
- Accepts GRPOConfig via config parameter (n_iterations, learning_rate, max_budget_usd, etc.)
- Returns job_id, status, progress, metrics, cost information
- Supports filtering jobs by agent_id
- Real-time status tracking

**Files Created:**
- `src/agentcore/training/training_jsonrpc.py` (240 lines)
- `tests/training/integration/test_training_api.py` (19 tests)

### Test Results

```
tests/training/integration/test_training_api.py::test_start_grpo_success PASSED
tests/training/integration/test_training_api.py::test_start_grpo_with_defaults PASSED
tests/training/integration/test_training_api.py::test_start_grpo_missing_agent_id PASSED
tests/training/integration/test_training_api.py::test_start_grpo_missing_training_queries PASSED
tests/training/integration/test_training_api.py::test_start_grpo_insufficient_queries PASSED
tests/training/integration/test_training_api.py::test_get_status_success PASSED
tests/training/integration/test_training_api.py::test_cancel_success PASSED
tests/training/integration/test_training_api.py::test_list_jobs_all PASSED
tests/training/integration/test_training_api.py::test_end_to_end_job_lifecycle PASSED
... (19/19 tests PASSED)
```

### Notes

- JWT authentication handled by existing JSON-RPC infrastructure
- RBAC authorization integration point available for future implementation
- Endpoints automatically registered on module import
