# LLM-CLIENT-009: LLMService Facade

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 8 SP
**Sprint:** Sprint 2
**Phase:** Service Facade
**Parent:** LLM-001

## Description

Main service interface orchestrating provider selection, model governance, and A2A context handling.

This is the **main interface** that all other components will use. Critical path task.

## Acceptance Criteria

- [x] LLMService class in llm_service.py
- [x] complete(request: LLMRequest) -> LLMResponse method
- [x] stream(request: LLMRequest) -> AsyncIterator[str] method
- [x] Model governance enforcement (check ALLOWED_MODELS before provider call)
- [x] Provider selection via ProviderRegistry
- [x] A2A context propagation to providers
- [x] Error handling with meaningful messages
- [x] Logging all requests with structured logging (trace_id, model, provider, latency)
- [x] Global llm_service instance (singleton)
- [x] Unit tests (90%+ coverage)
- [x] Integration test end-to-end

## Dependencies

**Blocks:**
- LLM-CLIENT-010 (unit tests)
- LLM-CLIENT-011 (metrics)
- LLM-CLIENT-012 (model selector)
- LLM-CLIENT-013 (JSON-RPC)
- LLM-CLIENT-018 (audit logging)
- LLM-CLIENT-019 (rate limits)

**Requires:** LLM-CLIENT-008 (provider registry)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/services/llm_service.py`

**Pattern:** Facade + Singleton

```python
class LLMService:
    def __init__(self, config: Settings):
        self.config = config
        self.registry = ProviderRegistry(config)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # 1. Model governance check
        if request.model not in self.config.ALLOWED_MODELS:
            raise ModelNotAllowedError(request.model)

        # 2. Provider selection
        provider = self.registry.get_provider_for_model(request.model)

        # 3. Execute request
        response = await provider.complete(request)

        # 4. Structured logging
        logger.info("LLM request completed",
                    trace_id=request.trace_id,
                    model=request.model,
                    latency=response.latency)

        return response

# Global singleton
llm_service = LLMService(settings)
```

**Critical Path:** This task is on the critical path.

## Estimated Time

- **Story Points:** 8 SP
- **Time:** 3-4 days (Backend Engineer 1)
- **Sprint:** Sprint 2, Days 13-15

## Owner

Backend Engineer 1

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
