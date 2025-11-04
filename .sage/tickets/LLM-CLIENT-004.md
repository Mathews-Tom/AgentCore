# LLM-CLIENT-004: Configuration Management

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 2 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** LLM-001

## Description

Add LLM service configuration to config.py using Pydantic Settings. Support environment variable loading for API keys and operational parameters.

This enables testing setup and provider configuration without waiting for provider implementations.

## Acceptance Criteria

- [x] ALLOWED_MODELS list in config.py (gpt-4.1-mini, gpt-5-mini, claude-3-5-haiku-20241022, gemini-1.5-flash)
- [x] LLM_DEFAULT_MODEL setting (default: gpt-4.1-mini)
- [x] Provider API key settings (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY) as Optional[str]
- [x] LLM_REQUEST_TIMEOUT float setting (default: 60.0)
- [x] LLM_MAX_RETRIES int setting (default: 3)
- [x] LLM_RETRY_EXPONENTIAL_BASE float (default: 2.0)
- [x] All settings loadable from .env file
- [x] Settings validation (timeout >0, max_retries >=0)
- [x] Example .env.example updated

## Dependencies

**Blocks:** LLM-CLIENT-017 (SDK version pinning)

**Requires:** None (independent configuration work)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/config.py` (extend existing)

**CLAUDE.md Compliance:**
- Configuration-only model references (no hardcoded models)
- Only allow GPT-4.1, GPT-4.1-mini, GPT-5, GPT-5-mini per CLAUDE.md

```python
class Settings(BaseSettings):
    # Existing settings...

    # LLM Service Configuration
    ALLOWED_MODELS: list[str] = Field(
        default=[
            "gpt-4.1-mini",
            "gpt-5-mini",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-flash"
        ]
    )
    LLM_DEFAULT_MODEL: str = "gpt-4.1-mini"

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    LLM_REQUEST_TIMEOUT: float = Field(default=60.0, gt=0)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0)
    LLM_RETRY_EXPONENTIAL_BASE: float = Field(default=2.0, gt=1)
```

## Estimated Time

- **Story Points:** 2 SP
- **Time:** 1 day (Backend Engineer 2)
- **Sprint:** Sprint 1, Day 1

## Owner

Backend Engineer 2

## Progress

**Status:** COMPLETED
**Created:** 2025-10-25
**Updated:** 2025-10-26
**Completed:** 2025-10-26

**Implementation Summary:**
1. ✓ Updated config.py with LLM service settings (lines 120-154)
2. ✓ Added Pydantic Field validators for timeout, max_retries, exponential_base
3. ✓ Updated .env.example with LLM Client Service section
4. ✓ Created comprehensive test suite (tests/config/test_llm_config.py, 13 tests)
5. ✓ All tests passing, mypy type checking successful
6. ✓ Settings validated to load correctly from environment

**Files Changed:**
- src/agentcore/a2a_protocol/config.py (added LLM configuration fields)
- .env.example (added LLM Client Service Configuration section)
- tests/config/test_llm_config.py (created with 13 test cases)

**Commit:** feat(llm-client): #LLM-CLIENT-004 configuration management
