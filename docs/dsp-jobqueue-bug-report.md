# JobQueue Bug Report: Hanging Tests Investigation

**Date:** 2025-10-29
**Component:** DSP (DSPy Optimization) - JobQueue
**Severity:** HIGH - Production Blocker
**Status:** IDENTIFIED - Awaiting Fix

---

## Executive Summary

Performance tests in `test_concurrent_optimizations.py` hang indefinitely when waiting for job results. Investigation revealed critical bugs in the `JobQueue` implementation (`src/agentcore/dspy_optimization/scalability/job_queue.py`) that prevent jobs from completing properly.

**Immediate Action Taken:**
Added `asyncio.wait_for()` timeouts to all affected test methods to prevent indefinite hangs (commit `a1532b9`).

**Required Action:**
Fix JobQueue implementation bugs before DSP component can be considered production-ready.

---

## Problem Description

### Symptoms

1. **Test Hangs:** Tests hang at "Started job queue with X workers" message
2. **Indefinite Wait:** `get_job_result()` method enters infinite busy-wait loop
3. **No Job Completion:** Workers start but jobs never reach COMPLETED status
4. **Queue Error:** "task_done() called too many times" error in worker threads

### Affected Tests

All tests in `tests/performance/dspy_optimization/test_concurrent_optimizations.py`:
- `test_100_concurrent_optimizations` ⚠️
- `test_500_concurrent_optimizations` ⚠️
- `test_1000_concurrent_optimizations` ⚠️
- `test_priority_scheduling` ⚠️
- `test_concurrent_with_failures` ⚠️
- `test_throughput_scaling` ⚠️
- `test_queue_utilization_metrics` ⚠️
- `test_rate_limiting` ⚠️

---

## Root Cause Analysis

### Bug Location

File: `src/agentcore/dspy_optimization/scalability/job_queue.py`

### Issue 1: Queue Synchronization Bug

**Error Message:** "task_done() called too many times"

**Analysis:**
- `task_done()` is called in multiple places without proper synchronization
- Line 293: Called when skipping cancelled jobs
- Line 342: Called in finally block after job execution
- This suggests jobs may be put into/taken from queue incorrectly

**Code Path:**
```python
# Line 292-294: Cancelled job handling
if job.status == JobStatus.CANCELLED:
    self._queue.task_done()  # ← Called here
    continue

# Line 340-342: Finally block
finally:
    self._active_jobs.discard(job.job_id)
    self._queue.task_done()  # ← Also called here
```

### Issue 2: Busy-Wait Loop Without Escape

**File:** `job_queue.py:213-214`

```python
async def get_job_result(self, job_id: str) -> Any:
    # ...
    while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        await asyncio.sleep(0.1)  # ← Infinite loop if status never changes
```

**Problem:**
If job status never changes to COMPLETED/FAILED/CANCELLED, this loop runs forever.

### Issue 3: Worker Processing Issue

**Hypothesis:**
Workers may not be properly picking up jobs from the queue or updating job status due to the `task_done()` synchronization issue.

**Evidence:**
- Simple test with 5 jobs, 2 workers hangs
- No "Handling job" output seen (jobs never execute)
- Error occurs early in worker processing

---

## Investigation Process

### 1. Initial Fix Attempt

**Action:** Added `asyncio.wait_for()` timeouts to test methods
**Result:** Timeouts prevent indefinite hangs but don't fix root cause
**Files Changed:** `tests/performance/dspy_optimization/test_concurrent_optimizations.py`

### 2. Code Review

**Reviewed:**
- `JobQueue.__init__()` - Initialization looks correct
- `JobQueue.start()` - Workers created properly
- `JobQueue.submit_job()` - Jobs queued correctly
- `JobQueue._worker()` - Found synchronization issues
- `JobQueue.get_job_result()` - Found busy-wait issue

### 3. Isolation Test

**Created:** `/tmp/test_jobqueue_simple.py` - Minimal reproduction case
**Config:** 5 jobs, 2 workers, no rate limiting
**Result:** Hangs with "task_done() called too many times" error
**Conclusion:** Bug is in JobQueue implementation, not test configuration

---

## Timeout Fixes Applied

### Commit: `a1532b9`

Added timeouts to prevent indefinite test hangs:

| Test Method | Timeout | Jobs | Rationale |
|-------------|---------|------|-----------|
| `test_100_concurrent_optimizations` | 30s | 100 | ~10 jobs/sec target |
| `test_500_concurrent_optimizations` | 60s | 500 | ~15 jobs/sec target |
| `test_1000_concurrent_optimizations` | 120s | 1000 | ~15 jobs/sec target |
| `test_priority_scheduling` | 10s | 3 | Sequential execution |
| `test_concurrent_with_failures` | 30s | 100 | With 10% failures |
| `test_throughput_scaling` | 30s | 100 | Per worker config |
| `test_queue_utilization_metrics` | 30s | 100 | With monitoring |
| `test_rate_limiting` | 30s | 50 | With 10/sec limit |

**Impact:**
- Tests now fail with `asyncio.TimeoutError` instead of hanging indefinitely
- Failure clearly indicates JobQueue malfunction
- CI/CD pipelines won't hang forever

---

## Required Fixes

### Priority 1: Fix task_done() Synchronization

**Location:** `job_queue.py:_worker()`

**Issue:**
- `task_done()` called multiple times per job
- Causes ValueError: "task_done() called too many times"

**Suggested Fix:**
```python
# Ensure task_done() called exactly once per get()
# Review all code paths that call task_done()
# Consider using try/finally more carefully
```

### Priority 2: Add Escape Condition to get_job_result()

**Location:** `job_queue.py:213-214`

**Current Code:**
```python
while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
    await asyncio.sleep(0.1)
```

**Suggested Fix:**
```python
# Option 1: Add timeout
timeout = 30.0  # Configurable
start_time = time.time()
while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
    if time.time() - start_time > timeout:
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
    await asyncio.sleep(0.1)

# Option 2: Use Event for notification
# Replace busy-wait with asyncio.Event that workers signal on completion
```

### Priority 3: Add Worker Error Handling

**Location:** `job_queue.py:_worker()`

**Suggestion:**
- Add comprehensive logging at each worker processing step
- Log when jobs are picked from queue
- Log when jobs start execution
- Log when jobs complete
- Log all exceptions with full context

### Priority 4: Add Queue Health Checks

**Suggestion:**
- Add method to check queue health
- Detect stalled workers
- Detect stuck jobs
- Expose metrics for monitoring

---

## Testing Strategy

### Unit Tests

1. **Test task_done() synchronization:**
   - Submit and cancel jobs
   - Verify task_done() called correctly
   - Check for ValueError

2. **Test worker lifecycle:**
   - Start/stop workers cleanly
   - Verify all workers process jobs
   - Check worker error handling

3. **Test job status transitions:**
   - QUEUED → RUNNING → COMPLETED
   - QUEUED → RUNNING → FAILED
   - QUEUED → CANCELLED
   - Verify status updates propagate

### Integration Tests

1. **Test concurrent job processing:**
   - Start with small numbers (5, 10 jobs)
   - Gradually scale up (100, 500, 1000)
   - Verify all jobs complete

2. **Test timeout behavior:**
   - Verify get_job_result() times out properly
   - Test graceful degradation

3. **Test error recovery:**
   - Jobs that fail
   - Jobs that retry
   - Workers that crash

---

## Impact Assessment

### Current State

- ❌ **Performance Tests:** 8/8 affected tests hang indefinitely
- ⚠️ **Unit Tests:** 17/17 passing (don't test concurrent scenarios)
- ✅ **Other DSP Tests:** Unaffected (don't use JobQueue)

### Production Risk

**HIGH RISK - DO NOT DEPLOY**

- JobQueue is core infrastructure for DSP-012 (Scalability & Performance)
- Used for 1000+ concurrent optimization target
- Bug prevents ANY concurrent job processing
- Silent failures possible (jobs hang without error)

### Acceptance Criteria Status

From DSP-016 Performance Testing:

- ❌ **Concurrent Optimizations:** 1000+ jobs target - BLOCKED by JobQueue bugs
- ✅ **Optimization Cycle Time:** <2h target - Tests pass (don't use JobQueue)
- ✅ **GPU Acceleration:** Benchmarks complete - Tests pass (don't use JobQueue)
- ⚠️ **Load Testing:** 1 test fails (spike pattern) - Separate issue

**DSP Component Status:** ⚠️ **NOT PRODUCTION READY** until JobQueue fixed

---

## Recommendations

### Immediate Actions

1. **Isolate JobQueue:** Mark as experimental/unstable
2. **Document Limitation:** Update docs to note JobQueue issues
3. **Alternative Approach:** Consider using established libraries (Celery, Dramatiq, etc.)
4. **Skip Tests:** Mark concurrent tests as `@pytest.mark.skip(reason="JobQueue bugs")`

### Short-Term Actions (1-2 days)

1. **Fix Bugs:** Address task_done() and busy-wait issues
2. **Add Logging:** Comprehensive debug logging
3. **Improve Tests:** Add unit tests for edge cases
4. **Code Review:** Have second developer review JobQueue implementation

### Long-Term Actions (1-2 weeks)

1. **Redesign:** Consider using asyncio.Queue + asyncio.Semaphore patterns
2. **Battle Test:** Run extended load tests (hours, not seconds)
3. **Production Monitor:** Add health checks and metrics
4. **Documentation:** Operational runbook for JobQueue issues

---

## Related Files

### Modified Files (Commit a1532b9)
- `tests/performance/dspy_optimization/test_concurrent_optimizations.py`

### Files Requiring Fixes
- `src/agentcore/dspy_optimization/scalability/job_queue.py`

### Test Files
- `/tmp/test_jobqueue_simple.py` (minimal reproduction)
- `tests/unit/dspy_optimization/test_scalability_job_queue.py` (17 unit tests)

---

## Appendix: Error Output

### Task Done Error

```
Worker 1 error: task_done() called too many times
```

### Test Hang Example

```
2025-10-29 20:25:46 [INFO] Started job queue with 20 workers (max concurrent: 100)
<hangs indefinitely - no further output>
```

---

**Report Generated:** 2025-10-29
**Investigated By:** Claude Code
**Branch:** feature/dspy-optimization
**Commits:** e6d6dff (rename fix), a1532b9 (timeout fixes)
**Status:** Awaiting JobQueue implementation fixes
