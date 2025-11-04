# LLM-CLIENT-010: Unit Test Suite

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 5 SP
**Sprint:** Sprint 1
**Phase:** Testing
**Parent:** LLM-001

## Description

Comprehensive unit tests covering all core logic with mocked provider SDKs for fast execution.

Quality gate ensuring 95%+ code coverage before integration testing.

## Acceptance Criteria

- [ ] Tests for all provider implementations (OpenAI, Anthropic, Gemini)
- [ ] Tests for provider selection logic
- [ ] Tests for model governance enforcement
- [ ] Tests for response normalization (each provider format)
- [ ] Tests for A2A context propagation
- [ ] Tests for retry logic and error handling
- [ ] Tests for timeout handling
- [ ] Mock all provider SDKs using pytest-mock
- [ ] 95%+ code coverage for core services
- [ ] All tests run in <10 seconds
- [ ] CI pipeline integration

## Dependencies

**Requires:** LLM-CLIENT-009 (LLMService facade)

## Technical Notes

**File Location:** `tests/unit/test_llm_service.py`, `tests/unit/test_llm_clients.py`

**Tools:**
- pytest-asyncio for async tests
- pytest-mock for SDK mocking
- pytest-cov for coverage

**Example Test:**
```python
@pytest.mark.asyncio
async def test_openai_complete_with_trace_id(mock_openai_client):
    client = LLMClientOpenAI(api_key="test")
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "test"}],
        trace_id="trace-123"
    )
    response = await client.complete(request)
    assert response.trace_id == "trace-123"
```

## Estimated Time

- **Story Points:** 5 SP
- **Time:** 2-3 days (Backend Engineer 2)
- **Sprint:** Sprint 1, Days 8-10

## Owner

Backend Engineer 2

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
