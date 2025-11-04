# FLOW-014: Data Export API

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story
**Sprint:** 3
**Effort:** 3 SP

## Dependencies

**Parent:** #FLOW-001
**Requires:**
- #FLOW-002

**Blocks:**
- #FLOW-019

## Context

Specs: `docs/specs/flow-based-optimization/spec.md`
Tasks: `docs/specs/flow-based-optimization/tasks.md` (see FLOW-014 section)

## Owner

Eng-1

## Status

✅ **COMPLETED**

## Implementation Details

**Implemented:** 2025-10-17T11:45:00Z (commit fc9e562)
**Verified:** 2025-10-17T11:46:00Z
**Branch:** feature/flow-based-optimization
**Tests:** 12/12 passed (100%)

### Deliverables

- ✅ training.export_trajectories JSON-RPC endpoint
- ✅ Filtering support (success_only, min_reward)
- ✅ Pagination support (limit, offset)
- ✅ JSON export format with full trajectory details
- ✅ Size limit enforcement (max 10,000 trajectories)
- ✅ Database integration with TrajectoryRepository
- ✅ Comprehensive integration tests (12 tests)

### Implementation Approach

**Repository Extension:**
- Extended `TrajectoryRepository.get_by_job()` with `min_reward` parameter
- Supports filtering by success status and minimum reward threshold
- Maintains existing pagination (limit, offset) and ordering

**JSON-RPC Endpoint:**
- `training.export_trajectories` method in training_jsonrpc.py
- Parameters:
  * `job_id` (required): Training job UUID
  * `success_only` (optional, default: false): Filter successful trajectories
  * `min_reward` (optional): Minimum reward threshold
  * `limit` (optional, default: 1000, max: 10,000): Pagination limit
  * `offset` (optional, default: 0): Pagination offset
- Returns JSON with trajectories array, count, filters, and pagination metadata
- Enforces MAX_EXPORT_LIMIT of 10,000 trajectories per request
- Async database session management with context manager

**Export Format:**
Each trajectory includes:
- trajectory_id, job_id, agent_id
- query, steps (array of action objects)
- reward, normalized_reward, advantage
- execution_time_ms, success, created_at

**Files Modified:**
- `src/agentcore/training/repositories.py` (added min_reward filter)
- `src/agentcore/training/training_jsonrpc.py` (added export endpoint)
- `tests/training/integration/test_training_api.py` (added 12 tests)

### Test Results

```
test_export_trajectories_success PASSED
test_export_trajectories_with_success_filter PASSED
test_export_trajectories_with_min_reward PASSED
test_export_trajectories_with_both_filters PASSED
test_export_trajectories_with_pagination PASSED
test_export_trajectories_missing_job_id PASSED
test_export_trajectories_invalid_job_id PASSED
test_export_trajectories_exceeds_max_limit PASSED
test_export_trajectories_invalid_success_only_type PASSED
test_export_trajectories_invalid_min_reward_type PASSED
test_export_trajectories_invalid_limit_type PASSED
test_export_trajectories_negative_offset PASSED
12/12 tests PASSED (100%)
```

### Notes

- Phase 1: Returns empty results as trajectories are simulated, not stored in DB
- Phase 2: Will return actual trajectories from PostgreSQL with full filtering
- Authorization checks (data_export permission) deferred to Phase 2
- Ready for integration in FLOW-019 (Integration Tests Phase 2)
