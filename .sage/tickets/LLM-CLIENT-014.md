# LLM-CLIENT-014: Integration Tests

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 8 SP
**Sprint:** Sprint 2
**Phase:** Testing
**Parent:** LLM-001

## Description

End-to-end integration tests with real provider APIs validating complete workflows.

Quality gate for production readiness. **Critical path task.**

## Acceptance Criteria

- [ ] Integration test suite in tests/integration/test_llm_integration.py
- [ ] Test all 3 providers with real API calls (OpenAI, Anthropic, Gemini)
- [ ] Test streaming functionality end-to-end for each provider
- [ ] Test A2A context propagation (verify trace_id in responses)
- [ ] Test error handling (invalid models, timeout, network errors)
- [ ] Test retry logic with transient failures
- [ ] Test concurrent requests (100 concurrent minimum)
- [ ] Test rate limit handling (if test environment allows)
- [ ] Requires API keys in test environment (.env.test)
- [ ] CI pipeline integration (run on staging environment)
- [ ] All tests pass consistently (>95% success rate)

## Dependencies

**Blocks:** LLM-CLIENT-015 (performance benchmarks - critical path)

**Requires:** LLM-CLIENT-013 (JSON-RPC methods)

**Parallel Work:** LLM-CLIENT-020 (security audit) can run in parallel

## Technical Notes

**File Location:** `tests/integration/test_llm_integration.py`

**Environment Setup:**
- Requires `.env.test` with real API keys
- CI: Run in staging environment with rate limit consideration

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_complete_e2e():
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say 'test'"}],
        trace_id="test-trace-123"
    )
    response = await llm_service.complete(request)
    assert response.content
    assert response.trace_id == "test-trace-123"
```

**Critical Path:** This task is on the critical path.

## Estimated Time

- **Story Points:** 8 SP
- **Time:** 3-4 days (Backend Engineer 2 + QA Engineer)
- **Sprint:** Sprint 2, Days 16-19

## Owner

Backend Engineer 2 + QA Engineer

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
