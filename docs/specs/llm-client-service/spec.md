# LLM Client Service Specification

**Component ID:** LLM-CLIENT
**Status:** Ready for Implementation
**Priority:** P0 (HIGH - Phase 1)
**Effort:** 2-3 weeks
**Owner:** Backend Team
**Source:** `.docs/specs/SPEC_MULTI_PROVIDER_LLM_CLIENT.md`

---

## 1. Overview

### Purpose and Business Value

Build production-grade multi-provider LLM client supporting OpenAI, Anthropic Claude, and Google Gemini with unified interface, comprehensive metrics, and runtime model switching. This service enables AgentCore to:

- Reduce vendor lock-in through provider abstraction
- Optimize costs by selecting appropriate models per task
- Improve resilience through provider failover
- Enable A2A protocol-native LLM operations with full traceability

### Success Metrics

- **Provider Coverage**: 3 providers (OpenAI, Anthropic, Gemini) operational
- **Performance**: <5ms abstraction overhead per request
- **Reliability**: 99.9% uptime with automatic failover
- **Cost Optimization**: 30% cost reduction through intelligent model selection
- **Model Governance**: 100% enforcement of ALLOWED_MODELS policy

### Target Users

- **Internal**: AgentCore services (SessionManager, AgentManager, TaskManager)
- **External**: Agent implementations requiring LLM capabilities
- **Operators**: Platform engineers managing model costs and performance

---

## 2. Functional Requirements

### FR-1: Multi-Provider Support

**FR-1.1** The system SHALL support OpenAI GPT models (gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5-mini)
**FR-1.2** The system SHALL support Anthropic Claude models (claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus)
**FR-1.3** The system SHALL support Google Gemini models (gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash)
**FR-1.4** The system SHALL automatically select the appropriate provider based on the requested model

### FR-2: Unified Interface

**FR-2.1** The system SHALL provide a single `LLMRequest` model for all providers
**FR-2.2** The system SHALL provide a single `LLMResponse` model normalizing all provider responses
**FR-2.3** The system SHALL support both streaming and non-streaming completions
**FR-2.4** The system SHALL normalize provider-specific message formats automatically

### FR-3: Model Governance

**FR-3.1** The system SHALL enforce ALLOWED_MODELS configuration per CLAUDE.md
**FR-3.2** The system SHALL reject requests for non-allowed models with `ModelNotAllowedError`
**FR-3.3** The system SHALL log all model governance violations
**FR-3.4** The system SHALL support runtime updates to allowed models list

### FR-4: A2A Protocol Integration

**FR-4.1** The system SHALL propagate `trace_id` through all provider requests
**FR-4.2** The system SHALL propagate `source_agent` and `session_id` metadata
**FR-4.3** The system SHALL include A2A context in provider request headers where supported
**FR-4.4** The system SHALL return A2A context in all responses

### FR-5: Runtime Model Selection

**FR-5.1** The system SHALL support model tier configuration (fast, balanced, premium)
**FR-5.2** The system SHALL select models based on task complexity hints
**FR-5.3** The system SHALL support provider preference configuration
**FR-5.4** The system SHALL implement fallback providers for resilience

### User Stories

**US-1**: As an agent developer, I want to call any LLM provider through a single interface so that I don't need provider-specific code.

**US-2**: As a platform engineer, I want to enforce which models can be used so that costs remain predictable.

**US-3**: As an operations team member, I want to monitor LLM usage per provider so that I can optimize costs.

**US-4**: As a service owner, I want automatic provider failover so that LLM operations remain available during provider outages.

**US-5**: As a developer, I want A2A trace propagation so that I can debug multi-agent LLM interactions.

### Business Rules

- **BR-1**: Only models in ALLOWED_MODELS configuration may be used
- **BR-2**: Model-to-provider mapping must be unambiguous
- **BR-3**: All LLM operations must be async-first
- **BR-4**: Provider API keys must be loaded from environment only
- **BR-5**: Streaming must be supported for all providers

---

## 3. Non-Functional Requirements

### Performance

- **NFR-P1**: Abstraction layer overhead SHALL be <5ms per request
- **NFR-P2**: Provider selection SHALL complete in <1ms
- **NFR-P3**: Time to first token SHALL be <500ms for streaming (95th percentile)
- **NFR-P4**: Total request latency SHALL match native SDK performance (±5%)

### Scalability

- **NFR-S1**: SHALL support 1000 concurrent requests per provider
- **NFR-S2**: SHALL support horizontal scaling with no shared state
- **NFR-S3**: SHALL handle provider rate limits gracefully with exponential backoff
- **NFR-S4**: SHALL support request queuing during peak load

### Security

- **NFR-SEC1**: API keys SHALL be loaded from environment variables only
- **NFR-SEC2**: API keys SHALL NOT be logged or exposed in responses
- **NFR-SEC3**: All provider communication SHALL use TLS 1.2+
- **NFR-SEC4**: Request/response data SHALL respect provider security policies

### Reliability

- **NFR-R1**: SHALL achieve 99.9% uptime with proper provider configuration
- **NFR-R2**: SHALL implement retry logic with exponential backoff (3 retries default)
- **NFR-R3**: SHALL handle provider timeouts gracefully (60s default)
- **NFR-R4**: SHALL provide meaningful error messages for all failure modes

### Observability

- **NFR-O1**: SHALL emit Prometheus metrics for all operations
- **NFR-O2**: SHALL log all requests with structured logging (trace_id, model, provider)
- **NFR-O3**: SHALL track token usage per provider and model
- **NFR-O4**: SHALL expose provider health status

---

## 4. Features & Flows

### Feature 1: Provider-Agnostic Completion (Priority: P0)

**Description**: Generate LLM completions through unified interface with automatic provider selection.

**Key Flow**:

1. Client creates `LLMRequest` with model and messages
2. LLMService validates model against ALLOWED_MODELS
3. ProviderRegistry resolves provider from model
4. Provider client executes request with native SDK
5. Response normalizer converts to unified `LLMResponse`
6. Metrics recorded (latency, tokens, cost estimate)

**Input**: `LLMRequest(model, messages, temperature, max_tokens, trace_id, source_agent)`
**Output**: `LLMResponse(content, usage, latency_ms, provider, trace_id)`

### Feature 2: Streaming Completion (Priority: P0)

**Description**: Stream LLM tokens as they're generated with provider-agnostic interface.

**Key Flow**:

1. Client creates `LLMRequest` with `stream=True`
2. Provider selection and validation same as non-streaming
3. Provider client returns `AsyncIterator[str]`
4. Client consumes tokens via async iteration
5. Final metrics recorded after stream completes

**Input**: `LLMRequest(stream=True, ...)`
**Output**: `AsyncIterator[str]` yielding content tokens

### Feature 3: Runtime Model Selection (Priority: P1)

**Description**: Select models dynamically based on task requirements and configuration.

**Key Flow**:

1. Client specifies tier (fast/balanced/premium) or task complexity
2. ModelSelector resolves to concrete model based on config
3. Request proceeds with selected model
4. Selection rationale logged for analysis

**Input**: `ModelTier.BALANCED` or `task_complexity="high"`
**Output**: Concrete model string (e.g., "gpt-4.1-mini")

### Feature 4: Comprehensive Metrics (Priority: P0)

**Description**: Track all LLM operations with Prometheus metrics.

**Metrics Exposed**:

- `llm_requests_total{provider, model, status}` - Total requests
- `llm_requests_duration_seconds{provider, model}` - Request latency histogram
- `llm_tokens_total{provider, model, token_type}` - Token usage
- `llm_errors_total{provider, model, error_type}` - Error counts
- `llm_active_requests{provider}` - Active request gauge

### Feature 5: Model Governance (Priority: P0)

**Description**: Enforce allowed models policy across all LLM operations.

**Key Flow**:

1. Every request validated against ALLOWED_MODELS config
2. Rejected requests return `ModelNotAllowedError`
3. Violations logged with trace_id and source_agent
4. Metrics track governance violations

---

## 5. Acceptance Criteria

### Definition of Done

- [ ] All three providers (OpenAI, Anthropic, Gemini) implemented and tested
- [ ] Unified `LLMRequest`/`LLMResponse` models in production
- [ ] Automatic provider selection operational
- [ ] Model governance enforcing ALLOWED_MODELS
- [ ] A2A context propagation working (trace_id, source_agent, session_id)
- [ ] Streaming support functional for all providers
- [ ] Prometheus metrics instrumented and validated
- [ ] Runtime model selection via ModelSelector
- [ ] JSON-RPC methods registered (llm.complete, llm.models, llm.metrics)
- [ ] Unit tests achieving 90%+ coverage
- [ ] Integration tests passing with real provider APIs
- [ ] Abstraction overhead <5ms validated
- [ ] Documentation complete with examples
- [ ] All providers tested in staging environment

### Validation Approach

**Unit Testing**:

- Mock provider SDKs for fast unit tests
- Test model governance enforcement
- Test provider selection logic
- Test response normalization

**Integration Testing**:

- Real API calls to all three providers
- Test streaming functionality
- Test A2A context propagation
- Test error handling and retries

**Performance Testing**:

- Benchmark abstraction overhead
- Load test with 1000 concurrent requests
- Measure time to first token (streaming)

**Security Testing**:

- Verify API keys not logged
- Test TLS configuration
- Validate input sanitization

---

## 6. Dependencies

### Technical Stack

- **Core**: Python 3.12+, FastAPI, Pydantic, asyncio
- **Provider SDKs**: openai ^1.0.0, anthropic ^0.40.0, google-genai ^0.2.0
- **HTTP**: httpx ^0.27.0
- **Metrics**: prometheus-client ^0.21.0
- **Logging**: structlog

### External Integrations

- **OpenAI API**: Requires `OPENAI_API_KEY` environment variable
- **Anthropic API**: Requires `ANTHROPIC_API_KEY` environment variable
- **Google Gemini API**: Requires `GOOGLE_API_KEY` environment variable

### Related Components

- **AgentCore Services**: SessionManager, AgentManager, TaskManager will consume LLMService
- **JSON-RPC Handler**: LLM methods registered via jsonrpc_handler
- **Event Manager**: Metrics published via event streams
- **Configuration**: Settings loaded from config.py and .env

### Technical Assumptions

- Provider API keys are available in environment
- Network connectivity to all three providers
- Provider APIs maintain backward compatibility
- Token usage tracking available from all providers
- Streaming supported by all provider SDKs

---

## 7. Implementation Notes

### Component Structure

```
src/agentcore/a2a_protocol/
├── models/
│   └── llm.py                    # LLMRequest, LLMResponse, errors
├── services/
│   ├── llm_client_base.py        # Abstract LLMClient interface
│   ├── llm_client_openai.py      # OpenAI implementation
│   ├── llm_client_anthropic.py   # Anthropic implementation
│   ├── llm_client_gemini.py      # Gemini implementation
│   ├── llm_service.py            # LLMService facade + ProviderRegistry
│   ├── model_selector.py         # Runtime model selection
│   └── llm_jsonrpc.py            # JSON-RPC methods
├── metrics/
│   └── llm_metrics.py            # Prometheus metrics
```

### Configuration

```python
# config.py additions
OPENAI_API_KEY: str | None = None
ANTHROPIC_API_KEY: str | None = None
GOOGLE_API_KEY: str | None = None

ALLOWED_MODELS: list[str] = [
    "gpt-4.1-mini", "gpt-5-mini",
    "claude-3-5-haiku-20241022",
    "gemini-1.5-flash"
]

LLM_DEFAULT_MODEL: str = "gpt-4.1-mini"
LLM_REQUEST_TIMEOUT: float = 60.0
LLM_MAX_RETRIES: int = 3
```

### Timeline

**Week 1: Core Infrastructure**

- Data models and abstract interface (Days 1-2)
- OpenAI client implementation (Days 3-4)
- Anthropic client implementation (Day 5)

**Week 2: Multi-Provider Completion**

- Gemini client implementation (Day 6)
- Provider registry (Day 7)
- LLM service facade (Day 8)
- Metrics instrumentation (Days 9-10)

**Week 3: Advanced Features**

- Runtime model selection (Day 11)
- JSON-RPC integration (Day 12)
- Integration testing (Days 13-14)
- Documentation and deployment (Day 15)

---

## 8. References

- Source specification: `.docs/specs/SPEC_MULTI_PROVIDER_LLM_CLIENT.md`
- OpenAI SDK: <https://github.com/openai/openai-python>
- Anthropic SDK: <https://github.com/anthropics/anthropic-sdk-python>
- Google GenAI SDK: <https://github.com/google/generative-ai-python>
- AgentCore CLAUDE.md: Model governance rules
