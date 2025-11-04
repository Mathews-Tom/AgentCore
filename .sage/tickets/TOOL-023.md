# TOOL-023: Rate Limiting with Redis

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Implement RateLimiter class using Redis with token bucket algorithm for per-tool, per-user rate limits with fail-closed strategy

## Acceptance Criteria
- [ ] RateLimiter class with `check_and_consume(tool_id, user_id, tokens=1)` method
- [ ] Token bucket algorithm using Redis INCR + EXPIRE (atomic operations)
- [ ] Per-tool, per-user rate limits configurable in ToolMetadata
- [ ] Fail-closed strategy: Return 503 if Redis unavailable
- [ ] Rate limit exceeded returns 429 with retry-after header
- [ ] Redis key format: `rate_limit:{tool_id}:{user_id}`
- [ ] TTL of 60 seconds for sliding window
- [ ] Unit tests with mocked Redis
- [ ] Integration tests with real Redis via testcontainers
- [ ] Concurrency tests validate atomic operations

## Dependencies
#TOOL-017

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 8
**Estimated Duration:** 8 days
**Sprint:** 4

## Implementation Details
**Owner:** Backend Engineer + DevOps
**Files:** src/agentcore/tools/rate_limiter.py
