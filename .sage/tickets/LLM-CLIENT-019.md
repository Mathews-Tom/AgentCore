# LLM-CLIENT-019: Rate Limit Handling

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 3 SP
**Sprint:** Sprint 2
**Phase:** Risk Mitigation
**Parent:** LLM-001

## Description

Implement robust rate limit detection and handling with exponential backoff and request queuing.

Production requirement for handling provider rate limits gracefully.

## Acceptance Criteria

- [ ] Rate limit error detection for all providers (OpenAI 429, Anthropic 429, Gemini RESOURCE_EXHAUSTED)
- [ ] Exponential backoff retry logic (base 2, max 5 retries, max delay 32s)
- [ ] Retry-After header respect for providers that support it
- [ ] Request queuing when rate limited (max queue size: 100)
- [ ] Rate limit metrics (llm_rate_limit_errors_total, llm_rate_limit_retry_delay_seconds)
- [ ] Configurable retry behavior (max retries, base delay)
- [ ] Unit tests for rate limit scenarios
- [ ] Integration test simulating rate limits
- [ ] Documentation for production rate limit configuration

## Dependencies

**Requires:** LLM-CLIENT-009 (LLMService facade)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/services/llm_client_base.py` (retry decorator)

**Retry Strategy:**
```python
async def complete_with_retry(self, request: LLMRequest) -> LLMResponse:
    max_retries = settings.LLM_MAX_RETRIES
    base_delay = settings.LLM_RETRY_EXPONENTIAL_BASE

    for attempt in range(max_retries):
        try:
            return await self.complete(request)
        except RateLimitError as e:
            if attempt >= max_retries - 1:
                raise
            delay = base_delay ** attempt
            logger.warning(f"Rate limited, retrying in {delay}s")
            await asyncio.sleep(delay)
```

**Provider-Specific Handling:**
- OpenAI: 429 with Retry-After header
- Anthropic: 429 with rate limit details
- Gemini: RESOURCE_EXHAUSTED status

## Estimated Time

- **Story Points:** 3 SP
- **Time:** 1-2 days (Backend Engineer 2)
- **Sprint:** Sprint 2, Days 14-15

## Owner

Backend Engineer 2

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
