# LLM-CLIENT-003: Abstract Base LLMClient Interface

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 2 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** LLM-001

## Description

Define abstract base class establishing contract for all provider implementations. Include complete() and stream() method signatures with full type hints.

This establishes the implementation pattern that all three provider clients will follow.

## Acceptance Criteria

- [x] Abstract LLMClient class in llm_client_base.py
- [x] complete() abstract method: `async def complete(request: LLMRequest) -> LLMResponse`
- [x] stream() abstract method: `async def stream(request: LLMRequest) -> AsyncIterator[str]`
- [x] _normalize_response() abstract helper method
- [x] Error handling contract documented in docstrings
- [x] Type hints complete (mypy validation passing)
- [x] Example implementation skeleton provided in docstring

## Dependencies

**Blocks:** LLM-CLIENT-005, LLM-CLIENT-006, LLM-CLIENT-007

**Requires:** LLM-CLIENT-002 (needs LLMRequest, LLMResponse models)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/services/llm_client_base.py`

**Pattern:** Abstract Base Class with Protocol

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse

class LLMClient(ABC):
    """Abstract base class for all LLM provider implementations."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute completion request and return normalized response."""
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Execute streaming completion and yield tokens."""
        pass

    @abstractmethod
    def _normalize_response(self, raw_response) -> LLMResponse:
        """Normalize provider-specific response to LLMResponse."""
        pass
```

## Estimated Time

- **Story Points:** 2 SP
- **Time:** 1-2 days (Backend Engineer 1)
- **Sprint:** Sprint 1, Day 2

## Owner

Backend Engineer 1

## Progress

**Status:** COMPLETED
**Created:** 2025-10-25
**Updated:** 2025-10-26
**Completed:** 2025-10-26

**Completion Summary:**
1. ✅ Created abstract base class in src/agentcore/a2a_protocol/services/llm_client_base.py
2. ✅ Defined complete() method signature with full type hints
3. ✅ Defined stream() method signature with AsyncIterator[str] return type
4. ✅ Defined _normalize_response() helper method
5. ✅ Added comprehensive docstrings with error handling contracts
6. ✅ Included example implementation skeleton in class docstring
7. ✅ Validated with mypy (strict mode, 100% type coverage)
8. ✅ Created 14 test cases in tests/unit/services/test_llm_client_base.py
9. ✅ All tests passing

**Files Created:**
- src/agentcore/a2a_protocol/services/llm_client_base.py (280 lines)
- tests/unit/services/test_llm_client_base.py (410 lines)

**Commit:** aeb0a7bd677df321c50e945260fa35cf6e6e2f6d
