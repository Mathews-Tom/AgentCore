# LLM-CLIENT-017: Provider SDK Version Pinning

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story
**Component:** llm-client-service
**Effort:** 1 SP
**Sprint:** Sprint 1
**Phase:** Risk Mitigation
**Parent:** LLM-001

## Description

Pin exact versions of provider SDKs in pyproject.toml and document compatibility matrix to prevent breaking changes.

Risk mitigation against provider API changes.

## Acceptance Criteria

- [ ] pyproject.toml updated with exact versions: openai==1.54.0, anthropic==0.40.0, google-genai==0.2.0
- [ ] Dependency compatibility matrix documented in DEPENDENCIES.md
- [ ] CI pipeline validates pinned versions
- [ ] Upgrade procedure documented
- [ ] Known issues with specific versions documented

## Dependencies

**Requires:** LLM-CLIENT-004 (configuration)

## Technical Notes

**File Location:** `pyproject.toml`, `DEPENDENCIES.md`

**Pinned Versions:**
```toml
[project.dependencies]
openai = "==1.54.0"
anthropic = "==0.40.0"
google-genai = "==0.2.0"
```

**Risk Mitigation Strategy:**
- Exact pinning prevents unexpected breaking changes
- Monthly review cycle for upgrades
- Integration tests validate compatibility

## Estimated Time

- **Story Points:** 1 SP
- **Time:** 0.5-1 day (Backend Engineer 2)
- **Sprint:** Sprint 1, Day 7

## Owner

Backend Engineer 2

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
