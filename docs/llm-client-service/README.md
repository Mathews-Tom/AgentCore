# LLM Client Service

Multi-provider LLM orchestration service for AgentCore implementing unified interface across OpenAI, Anthropic, and Google Gemini providers with model governance, A2A context propagation, and comprehensive observability.

## Overview

The LLM Client Service provides a production-ready abstraction layer for multi-provider LLM operations in AgentCore's A2A protocol implementation. It solves the core challenge of provider fragmentation by exposing a single, unified interface while maintaining provider-specific optimizations and error handling.

**Key Features:**

- **Multi-Provider Support:** OpenAI (GPT-4.1, GPT-5), Anthropic (Claude 3.5), Google Gemini (2.0, 1.5)
- **Model Governance:** Configuration-driven allowed model lists prevent cost overruns
- **A2A Protocol Integration:** Full trace_id, source_agent, session_id propagation
- **Streaming Support:** Non-blocking token streaming for real-time responses
- **Retry Logic:** Exponential backoff with configurable retry limits (default 3)
- **Timeout Handling:** Configurable timeouts (default 60s) per provider
- **Prometheus Metrics:** Request counts, latency percentiles, token usage, error tracking
- **JSON-RPC 2.0 API:** Standard protocol for remote LLM operations
- **Structured Logging:** Complete request/response logging with A2A context

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Service                             │
│  (Facade Pattern - Unified Interface)                           │
│                                                                  │
│  - Model Governance (ALLOWED_MODELS validation)                 │
│  - Provider Selection (ProviderRegistry)                        │
│  - A2A Context Propagation                                      │
│  - Metrics Collection                                           │
│  - Structured Logging                                           │
└──────────────┬──────────────┬──────────────┬───────────────────┘
               │              │              │
        ┌──────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
        │  OpenAI    │ │ Anthropic  │ │  Gemini    │
        │  Client    │ │  Client    │ │  Client    │
        └──────┬─────┘ └─────┬──────┘ └─────┬──────┘
               │              │              │
        ┌──────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
        │ OpenAI API │ │Anthropic   │ │ Google     │
        │            │ │  API       │ │ Gemini API │
        └────────────┘ └────────────┘ └────────────┘
```

**Component Breakdown:**

1. **LLMService** (`llm_service.py`): Main facade providing `complete()` and `stream()` methods
2. **ProviderRegistry** (`llm_service.py`): Model-to-provider mapping and singleton instance management
3. **LLMClient Implementations**: Provider-specific clients implementing abstract base class
   - `LLMClientOpenAI` - OpenAI GPT models
   - `LLMClientAnthropic` - Anthropic Claude models
   - `LLMClientGemini` - Google Gemini models
4. **JSON-RPC Methods** (`llm_jsonrpc.py`): Remote procedure calls for LLM operations
5. **Prometheus Metrics** (`llm_metrics.py`): Observability instrumentation

## Quick Start

### Installation

The LLM Client Service is included in the AgentCore installation. No additional packages required beyond core dependencies.

```bash
# Install AgentCore with LLM client dependencies
uv add agentcore[llm]

# Or install from source
uv add -e .
```

### Configuration

Set up API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."

# Model governance (optional - defaults to all models)
export ALLOWED_MODELS='["gpt-4.1-mini","claude-3-5-haiku-20241022","gemini-2.0-flash-exp"]'

# Timeout and retry configuration (optional)
export LLM_REQUEST_TIMEOUT=60.0
export LLM_MAX_RETRIES=3
```

### Basic Usage

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

# Non-streaming completion
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    temperature=0.7,
    max_tokens=200,
    trace_id="trace-abc-123",  # A2A context
)

response = await llm_service.complete(request)
print(response.content)
print(f"Tokens: {response.usage.total_tokens}")
print(f"Latency: {response.latency_ms}ms")
```

### Streaming Usage

```python
# Streaming completion
request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True,
    trace_id="trace-xyz-789",
)

async for token in llm_service.stream(request):
    print(token, end="", flush=True)
```

### JSON-RPC Usage

```bash
# Non-streaming completion via JSON-RPC
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "llm.complete",
    "params": {
      "model": "gpt-4.1-mini",
      "messages": [{"role": "user", "content": "Hello"}],
      "temperature": 0.7
    },
    "id": 1
  }'

# List allowed models
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "llm.models",
    "params": {},
    "id": 2
  }'

# Get metrics snapshot
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "llm.metrics",
    "params": {},
    "id": 3
  }'
```

## Supported Models

### OpenAI Models
- `gpt-4.1` - Latest GPT-4 Turbo (premium tier)
- `gpt-4.1-mini` - Cost-effective GPT-4 (fast tier)
- `gpt-5` - Cutting-edge reasoning (premium tier)
- `gpt-5-mini` - Balanced performance (balanced tier)

### Anthropic Models
- `claude-3-5-sonnet` - Highest capability (premium tier)
- `claude-3-5-haiku-20241022` - Fast and efficient (balanced tier)
- `claude-3-opus` - Maximum intelligence (premium tier)

### Google Gemini Models
- `gemini-2.0-flash-exp` - Experimental flash model (fast tier)
- `gemini-1.5-pro` - Advanced reasoning (premium tier)
- `gemini-2.0-flash-exp` - Fast responses (fast tier)

**Model Governance:**

Only models listed in `ALLOWED_MODELS` configuration can be used. Requests for non-allowed models raise `ModelNotAllowedError` and are tracked as governance violations in Prometheus metrics.

## Documentation Structure

- **[API Reference](./api-reference.md)** - Complete API documentation with method signatures
- **[Usage Examples](./usage-examples.md)** - Code examples for common use cases
- **[Configuration Guide](./configuration-guide.md)** - Environment setup and model governance
- **[Troubleshooting Guide](./troubleshooting-guide.md)** - Common errors and solutions
- **[Metrics Reference](./metrics-reference.md)** - Prometheus metrics documentation
- **[Migration Guide](./migration-guide.md)** - Migrating from direct SDK usage

## Performance Characteristics

**Latency Targets:**
- Fast tier models: <1s for simple queries
- Balanced tier models: 1-3s for typical queries
- Premium tier models: 2-10s for complex reasoning

**Overhead:**
- Abstraction layer adds <5ms compared to direct SDK usage
- Retry logic may extend total latency on transient errors
- Streaming reduces perceived latency (time to first token)

**Token Limits:**
- OpenAI: Up to 128K context, 4K default output
- Anthropic: Up to 200K context, 4K default output
- Gemini: Up to 1M context, variable output limits

## Security Best Practices

1. **API Key Management:**
   - Store API keys in environment variables (never hardcode)
   - Use secret management systems in production (AWS Secrets Manager, HashiCorp Vault)
   - Rotate keys regularly per provider policies

2. **Model Governance:**
   - Define `ALLOWED_MODELS` to prevent unauthorized model usage
   - Monitor `llm_governance_violations_total` metric for policy violations
   - Use cost-effective models by default (fast/balanced tiers)

3. **Request Validation:**
   - All requests validated with Pydantic models
   - Temperature clamped to 0.0-2.0 range
   - Max tokens validated as positive integers

4. **A2A Context:**
   - Always include `trace_id` for distributed tracing
   - Use `source_agent` for accountability and auditing
   - Propagate `session_id` for conversation tracking

5. **Error Handling:**
   - Never expose API keys in error messages
   - Log errors with structured context (no PII in logs)
   - Handle provider errors gracefully (retry transient, fail fast on terminal)

## Monitoring and Observability

**Prometheus Metrics:**
- `llm_requests_total` - Request counts by provider, model, status
- `llm_requests_duration_seconds` - Latency percentiles
- `llm_tokens_total` - Token usage by provider, model, type
- `llm_errors_total` - Error counts by provider, model, error type
- `llm_active_requests` - Current active requests by provider
- `llm_governance_violations_total` - Model governance violations

**Structured Logging:**
All operations logged with:
- `trace_id` - A2A distributed trace ID
- `source_agent` - Requesting agent
- `session_id` - Conversation session
- `model` - Model used
- `provider` - Provider used
- `latency_ms` - Request latency
- `prompt_tokens` - Input tokens
- `completion_tokens` - Output tokens
- `total_tokens` - Total tokens

**Grafana Dashboards:**
See `docs/grafana/llm-client-dashboard.json` for pre-built dashboards.

## Development

### Running Tests

```bash
# Unit tests
uv run pytest tests/unit/services/test_llm_*.py

# Integration tests (requires API keys)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
uv run pytest tests/integration/services/test_llm_*.py

# Test coverage
uv run pytest tests/ --cov=src/agentcore/a2a_protocol/services --cov-report=html
```

### Adding a New Provider

1. Create provider client in `src/agentcore/a2a_protocol/services/llm_client_<provider>.py`
2. Extend `LLMClient` abstract base class
3. Implement `complete()`, `stream()`, and `_normalize_response()` methods
4. Add provider to `Provider` enum in `models/llm.py`
5. Update `MODEL_PROVIDER_MAP` in `llm_service.py`
6. Add API key configuration to `config.py`
7. Update `ProviderRegistry._create_provider()` method
8. Write unit and integration tests
9. Update documentation

### Type Safety

The LLM Client Service enforces strict type safety:
- All code passes `mypy --strict`
- Use built-in generics (`list[str]`, `dict[str, Any]`, `int | None`)
- No `typing.Any` unless necessary for dynamic values
- Full Pydantic validation on all models

## Support and Contributing

- **Issues:** Report bugs or request features at [GitHub Issues](https://github.com/yourusername/agentcore/issues)
- **Documentation:** Improvements welcome via pull requests
- **Testing:** Add tests for all new features
- **Code Quality:** Follow CLAUDE.md guidelines and run `uv run ruff check` before committing

## License

Copyright (c) 2025 AetherForge. Licensed under the MIT License.
