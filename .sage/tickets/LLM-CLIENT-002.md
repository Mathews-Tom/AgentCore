# LLM-CLIENT-002: Data Models and Enums

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 3 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** LLM-001

## Description

Create Pydantic models for LLMRequest, LLMResponse, and custom exceptions. Define Provider and ModelTier enums. Establish type-safe foundation for all LLM operations.

This is the critical path blocker for all other LLM client tasks. Must be completed first to unblock parallel provider implementation work.

## Acceptance Criteria

- [x] LLMRequest model with all fields (model, messages, temperature, max_tokens, stream, trace_id, source_agent, session_id)
- [x] LLMResponse model with usage tracking (prompt_tokens, completion_tokens, total_tokens)
- [x] Custom exceptions defined (ModelNotAllowedError, ProviderError, ProviderTimeoutError)
- [x] Provider enum (OPENAI, ANTHROPIC, GEMINI)
- [x] ModelTier enum (FAST, BALANCED, PREMIUM)
- [x] All models have 100% type coverage (mypy strict)
- [x] Pydantic validators for value ranges (temperature 0-2, max_tokens >0)

## Dependencies

**Blocks:** LLM-CLIENT-003, LLM-CLIENT-005, LLM-CLIENT-006, LLM-CLIENT-007

**Requires:** None (starting point)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/models/llm.py`

**Type Annotations:** Use built-in generics per CLAUDE.md
```python
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class ModelTier(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"

class LLMRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float = Field(ge=0, le=2, default=0.7)
    max_tokens: int | None = Field(gt=0, default=None)
    stream: bool = False
    trace_id: str | None = None
    source_agent: str | None = None
    session_id: str | None = None
```

## Estimated Time

- **Story Points:** 3 SP
- **Time:** 2 days (Backend Engineer 1)
- **Sprint:** Sprint 1, Days 1-2

## Owner

Backend Engineer 1

## Progress

**Status:** COMPLETED
**Created:** 2025-10-25
**Updated:** 2025-10-25
**Completed:** 2025-10-25

**Implementation Summary:**
1. Created `src/agentcore/a2a_protocol/models/llm.py` with all data models and enums
2. Defined LLMRequest, LLMResponse, LLMUsage with full type hints
3. Implemented custom exceptions (ModelNotAllowedError, ProviderError, ProviderTimeoutError)
4. Created Provider and ModelTier enums
5. Added Pydantic validators for temperature (0-2) and max_tokens (>0)
6. Achieved 100% mypy strict type coverage
7. Created comprehensive test suite (36 unit tests, 100% pass rate)
8. Used built-in generics (list[], dict[], int | None) per CLAUDE.md
9. All acceptance criteria met
