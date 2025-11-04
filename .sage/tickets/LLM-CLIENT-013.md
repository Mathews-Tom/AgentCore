# LLM-CLIENT-013: JSON-RPC Methods

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 3 SP
**Sprint:** Sprint 2
**Phase:** Integration
**Parent:** LLM-001

## Description

Expose LLM service via JSON-RPC 2.0 protocol for A2A integration.

Critical for A2A protocol integration. **Critical path task.**

## Acceptance Criteria

- [x] llm.complete handler in llm_jsonrpc.py
- [x] llm.stream handler (returns helpful message for SSE/WebSocket)
- [x] llm.models handler (list available models)
- [x] llm.metrics handler (return current metrics snapshot)
- [x] All methods registered with jsonrpc_processor
- [x] A2A context extraction from JsonRpcRequest
- [x] Error mapping to JsonRpcErrorCode
- [x] Request/response validation with Pydantic
- [x] Unit tests for all handlers
- [x] Integration test via JSON-RPC endpoint (unit tests with mocks)

## Dependencies

**Blocks:** LLM-CLIENT-014 (integration tests - critical path)

**Requires:**
- LLM-CLIENT-009 (LLMService facade)
- LLM-CLIENT-011 (metrics)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/services/llm_jsonrpc.py`

**Pattern:** Follow existing `task_jsonrpc.py` pattern

```python
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

@register_jsonrpc_method("llm.complete")
async def handle_llm_complete(request: JsonRpcRequest) -> dict[str, Any]:
    llm_request = LLMRequest(**request.params)
    response = await llm_service.complete(llm_request)
    return response.model_dump()

@register_jsonrpc_method("llm.models")
async def handle_llm_models(request: JsonRpcRequest) -> dict[str, Any]:
    return {"models": settings.ALLOWED_MODELS}
```

**Critical Path:** This task is on the critical path.

## Estimated Time

- **Story Points:** 3 SP
- **Time:** 1-2 days (Backend Engineer 1)
- **Sprint:** Sprint 2, Day 18

## Owner

Backend Engineer 1

## Progress

**Status:** COMPLETED
**Created:** 2025-10-25
**Updated:** 2025-10-26
**Completed:** 2025-10-26

## Implementation Summary

Successfully implemented JSON-RPC methods for LLM service:

**Files Created:**
- `src/agentcore/a2a_protocol/services/llm_jsonrpc.py` - 4 JSON-RPC handlers
- `tests/unit/services/test_llm_jsonrpc.py` - 10 unit tests (all passing)

**Files Modified:**
- `src/agentcore/a2a_protocol/main.py` - Import llm_jsonrpc for auto-registration

**JSON-RPC Methods:**
1. `llm.complete` - Non-streaming LLM completion with A2A context
2. `llm.stream` - Returns helpful error (JSON-RPC doesn't support streaming)
3. `llm.models` - Lists allowed models and default model
4. `llm.metrics` - Returns Prometheus metrics snapshot

**Features:**
- A2A context extraction (trace_id, source_agent, session_id)
- Error mapping (ModelNotAllowedError → ValueError, ProviderError → RuntimeError)
- Pydantic validation for all requests/responses
- Comprehensive error handling with structured logging
- Prometheus metrics collection via REGISTRY

**Testing:**
- 10 unit tests covering all handlers
- Success cases, error cases, A2A context extraction
- Mock-based testing (no real API calls)
- All tests passing

**Notes:**
- JSON-RPC 2.0 doesn't natively support streaming responses
- llm.stream returns helpful message directing to WebSocket/SSE endpoints
- Integration with existing jsonrpc_processor infrastructure
- Follows established patterns from task_jsonrpc.py and agent_jsonrpc.py
