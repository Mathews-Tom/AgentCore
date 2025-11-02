# LLM Client Service

**Unified multi-provider LLM client with model governance, metrics, and A2A integration**

## Overview

The LLM Client Service provides a unified abstraction layer for interacting with multiple Language Model providers (OpenAI, Anthropic, Google Gemini) through a consistent API. It implements model governance, comprehensive metrics, retry logic, and seamless integration with the A2A protocol.

### Key Benefits

- **Multi-Provider Support**: Single API for OpenAI, Anthropic, and Gemini
- **Model Governance**: Enforce allowed models via configuration
- **Production-Ready**: Built-in retry logic, rate limiting, circuit breakers
- **Observability**: Comprehensive Prometheus metrics and structured logging
- **Type-Safe**: Full type hints and Pydantic validation
- **A2A Integration**: Native trace_id, source_agent, session_id propagation

## Quick Start

### Basic Usage

```python
from agentcore.a2a_protocol.models.llm import LLMRequest
from agentcore.a2a_protocol.services.llm_service import LLMService

# Create service instance
llm_service = LLMService(timeout=90.0, max_retries=3)

# Create request
request = LLMRequest(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello, world!"}],
    temperature=0.7,
    max_tokens=100,
    trace_id="my-trace-001",
)

# Get completion
response = await llm_service.complete(request)
print(response.content)
print(response.usage.total_tokens)
```

## Documentation

- [Configuration Guide](./CONFIGURATION.md)
- [API Reference](./API_REFERENCE.md)
- [Usage Examples](./EXAMPLES.md)
- [Performance Guide](./PERFORMANCE.md)
- [Security Guide](./SECURITY.md)
- [Troubleshooting](./TROUBLESHOOTING.md)

**Version:** 1.0.0
**Last Updated:** 2025-11-03
