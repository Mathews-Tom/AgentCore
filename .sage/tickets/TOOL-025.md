# TOOL-025: Quota Management

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story

## Description
Implement quota tracking for daily/monthly limits with `tools.get_rate_limit_status` endpoint for quota checking

## Acceptance Criteria
- [ ] Daily and monthly quota tracking in Redis
- [ ] Quota configuration per tool in ToolMetadata
- [ ] Quota exceeded returns 429 with quota reset time
- [ ] `tools.get_rate_limit_status` JSON-RPC method
- [ ] Returns limit, remaining, reset_at for tool and user
- [ ] Unit tests for quota logic
- [ ] Integration tests for quota enforcement

## Dependencies
#TOOL-023

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 3
**Estimated Duration:** 3 days
**Sprint:** 4

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/quota.py
