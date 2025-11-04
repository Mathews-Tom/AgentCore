# COORD-005: Signal Aggregation Logic

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** coordination-service
**Effort:** 5 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** COORD-001

## Description

Implement score computation and aggregation for routing decisions

## Acceptance Criteria

- [ ] compute_individual_scores(agent_id) computes load, capacity, quality, cost, availability scores
- [ ] Scores derived from most recent signal of each type
- [ ] Load score inverted (high load = low score)
- [ ] compute_routing_score(agent_id) computes weighted composite score
- [ ] Weights applied from configuration
- [ ] Agents without signals receive default score 0.5
- [ ] Scores cached in AgentCoordinationState
- [ ] Unit tests for score computation (100% coverage)

## Dependencies

**Blocks:** Dependent tasks

**Requires:** COORD-004

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_score_aggregation.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 5 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
