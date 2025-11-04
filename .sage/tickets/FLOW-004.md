# FLOW-004: Reward Engine

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** Flow-Based Optimization (FLOW)
**Sprint:** 1
**Estimated Effort:** 5 story points (3-5 days)

---

## Description

Implement reward computation with outcome-based and shaped reward functions. Support reward normalization using group statistics to reduce variance.

---

## Acceptance Criteria

- [ ] Outcome-based rewards computed correctly (success/failure)
- [ ] Shaped rewards applied (tool usage +0.1, verification +0.05, length -0.01)
- [ ] Reward normalization using group statistics (mean, std)
- [ ] Custom reward function registry working
- [ ] Edge case handled: std_reward == 0
- [ ] Unit tests achieve 100% coverage (critical path)

---

## Dependencies

- **Parent**: #FLOW-001 (Flow-Based Optimization Engine epic)
- #FLOW-003
**Blocks:**
- #FLOW-005
- #FLOW-017

---

## Context

**Specs:** `docs/specs/flow-based-optimization/spec.md`
**Plans:** `docs/specs/flow-based-optimization/plan.md`
**Tasks:** `docs/specs/flow-based-optimization/tasks.md`

---

## Implementation Notes

**Files to Create/Modify:**
- `src/agentcore/training/rewards.py`
- `tests/training/unit/test_reward_engine.py`

**Owner:** Backend Engineer 1

---

## Progress

**Status:** Ready for implementation
**Notes:** Part of Phase 1 (Core Infrastructure)
