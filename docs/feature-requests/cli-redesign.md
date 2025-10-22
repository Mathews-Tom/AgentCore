# Feature Request: CLI Layer Redesign

**Date:** 2025-10-22
**Status:** Approved
**Priority:** Critical
**Type:** Architecture Redesign
**Component:** cli-layer

---

## Problem Statement

The CLI Layer v1.0 implementation has a critical protocol violation that prevents all commands from working.

**Current:** Parameters sent as flat dictionary
**Required:** Parameters wrapped in `params` object per JSON-RPC 2.0

**Impact:** All CLI commands fail with protocol errors.

---

## Proposed Solution

4-layer architecture redesign:
- Transport Layer (HTTP)
- Protocol Layer (JSON-RPC 2.0)
- Service Layer (Business Logic)
- CLI Layer (User Interface)

---

## Implementation

**Timeline:** 2 weeks
**Story Points:** 42
**Tickets:** CLI-R001 through CLI-R013

See: docs/specs/cli-layer/spec.md and plan.md for details.

---

## Success Criteria

- JSON-RPC 2.0 compliance
- All tests pass (95+)
- 90%+ coverage
- No performance regression
