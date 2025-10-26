# Migration Guide

Guide for migrating from direct provider SDK usage to the LLM Client Service.

## Table of Contents

- [Why Migrate?](#why-migrate)
- [Migration Overview](#migration-overview)
- [OpenAI Migration](#openai-migration)
- [Anthropic Migration](#anthropic-migration)
- [Google Gemini Migration](#google-gemini-migration)
- [Multi-Provider Applications](#multi-provider-applications)
- [Testing Migration](#testing-migration)
- [Rollback Strategy](#rollback-strategy)
- [Migration Checklist](#migration-checklist)

---

## Why Migrate?

### Benefits of LLM Client Service

1. **Multi-Provider Support:** Switch between OpenAI, Anthropic, Gemini without code changes
2. **Model Governance:** Centralized control over allowed models (prevent cost overruns)
3. **A2A Protocol Integration:** Built-in distributed tracing and context propagation
4. **Unified Error Handling:** Consistent exceptions across all providers
5. **Observability:** Automatic Prometheus metrics and structured logging
6. **Retry Logic:** Built-in exponential backoff for transient errors
7. **Provider Abstraction:** Unified interface hides provider-specific details
8. **Configuration Management:** Environment-based configuration (no hardcoded keys)

### When to Migrate

**Migrate if you:**
- Use multiple LLM providers
- Need model governance and cost control
- Want observability and metrics
- Require A2A protocol integration
- Need consistent error handling
- Want automatic retry logic

**Consider NOT migrating if you:**
- Only use one provider with no plans to change
- Need provider-specific features not exposed by LLM Client
- Have extremely tight latency requirements (<5ms overhead acceptable)
- Prefer direct SDK control

---

## Migration Overview

### Step-by-Step Migration Process

1. **Install Dependencies:** Add LLM Client Service to project
2. **Configure Environment:** Set API keys and allowed models
3. **Update Imports:** Replace SDK imports with LLM Client imports
4. **Convert Request Format:** Adapt to unified LLMRequest model
5. **Update Error Handling:** Use unified exceptions
6. **Test Thoroughly:** Verify functionality with all providers
7. **Deploy Gradually:** Phased rollout with monitoring
8. **Remove Old Dependencies:** Clean up unused SDK packages

### Compatibility Matrix

| Provider SDK | LLM Client Version | Status |
|-------------|-------------------|--------|
| OpenAI >= 1.0 | All versions | Fully compatible |
| Anthropic >= 0.18 | All versions | Fully compatible |
| Google GenAI >= 0.3 | All versions | Fully compatible |

---

## OpenAI Migration

### Before (Direct SDK)

```python
from openai import AsyncOpenAI

# Initialize client
client = AsyncOpenAI(api_key="sk-...")

# Non-streaming completion
response = await client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    temperature=0.7,
    max_tokens=200,
)

content = response.choices[0].message.content
tokens = response.usage.total_tokens

# Streaming completion
stream = await client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
)

async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Error handling
from openai import APIError, APITimeoutError, RateLimitError

try:
    response = await client.chat.completions.create(...)
except APITimeoutError:
    print("Timeout")
except RateLimitError:
    print("Rate limit")
except APIError as e:
    print(f"API error: {e}")
```

### After (LLM Client Service)

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

# No client initialization needed - use global instance

# Non-streaming completion
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    temperature=0.7,
    max_tokens=200,
)

response = await llm_service.complete(request)
content = response.content
tokens = response.usage.total_tokens

# Streaming completion
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
)

async for token in llm_service.stream(request):
    print(token, end="")

# Error handling
from agentcore.a2a_protocol.models.llm import (
    ModelNotAllowedError,
    ProviderError,
    ProviderTimeoutError,
)

try:
    response = await llm_service.complete(request)
except ProviderTimeoutError as e:
    print(f"Timeout: {e.timeout_seconds}s")
except ModelNotAllowedError as e:
    print(f"Model not allowed: {e.model}")
except ProviderError as e:
    print(f"Provider error: {e.provider}")
```

### Migration Mapping

| OpenAI SDK | LLM Client Service |
|-----------|-------------------|
| `AsyncOpenAI(api_key=...)` | `llm_service` (global instance) |
| `client.chat.completions.create(...)` | `llm_service.complete(request)` |
| `stream=True` | `llm_service.stream(request)` |
| `response.choices[0].message.content` | `response.content` |
| `response.usage.total_tokens` | `response.usage.total_tokens` |
| `APIError` | `ProviderError` |
| `APITimeoutError` | `ProviderTimeoutError` |
| `RateLimitError` | `ProviderError` (with retry) |

---

## Anthropic Migration

### Before (Direct SDK)

```python
from anthropic import AsyncAnthropic

# Initialize client
client = AsyncAnthropic(api_key="sk-ant-...")

# Non-streaming completion
response = await client.messages.create(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Hello"}],
    system="You are helpful",
    temperature=0.7,
    max_tokens=200,
)

content = response.content[0].text
tokens = response.usage.input_tokens + response.usage.output_tokens

# Streaming completion
async with client.messages.stream(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Count to 5"}],
    max_tokens=100,
) as stream:
    async for text in stream.text_stream:
        print(text, end="")

# Error handling
from anthropic import APIError, APITimeoutError, RateLimitError

try:
    response = await client.messages.create(...)
except APITimeoutError:
    print("Timeout")
except RateLimitError:
    print("Rate limit")
except APIError as e:
    print(f"API error: {e}")
```

### After (LLM Client Service)

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

# No client initialization needed

# Non-streaming completion (system message handled automatically)
request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "system", "content": "You are helpful"},  # Automatically extracted
        {"role": "user", "content": "Hello"}
    ],
    temperature=0.7,
    max_tokens=200,
)

response = await llm_service.complete(request)
content = response.content
tokens = response.usage.total_tokens

# Streaming completion
request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Count to 5"}],
    max_tokens=100,
    stream=True,
)

async for token in llm_service.stream(request):
    print(token, end="")

# Error handling (same as OpenAI)
from agentcore.a2a_protocol.models.llm import (
    ProviderError,
    ProviderTimeoutError,
)

try:
    response = await llm_service.complete(request)
except ProviderTimeoutError as e:
    print(f"Timeout: {e.timeout_seconds}s")
except ProviderError as e:
    print(f"Provider error: {e.provider}")
```

### Key Differences

**System Messages:**
- **Before:** Separate `system` parameter
- **After:** Include in `messages` array with `"role": "system"` (automatically extracted)

**Token Usage:**
- **Before:** `input_tokens` + `output_tokens`
- **After:** `total_tokens` (calculated automatically)

**Response Content:**
- **Before:** `response.content[0].text`
- **After:** `response.content`

---

## Google Gemini Migration

### Before (Direct SDK)

```python
import google.generativeai as genai

# Initialize client
genai.configure(api_key="...")
model = genai.GenerativeModel('gemini-1.5-flash')

# Non-streaming completion
response = await model.generate_content_async("Hello")
content = response.text
# No token usage in basic SDK

# Streaming completion
response = await model.generate_content_async(
    "Count to 5",
    stream=True,
)

async for chunk in response:
    print(chunk.text, end="")

# Error handling
try:
    response = await model.generate_content_async("Hello")
except Exception as e:
    print(f"Error: {e}")
```

### After (LLM Client Service)

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

# No client initialization needed

# Non-streaming completion
request = LLMRequest(
    model="gemini-1.5-flash",
    messages=[{"role": "user", "content": "Hello"}],
)

response = await llm_service.complete(request)
content = response.content
tokens = response.usage.total_tokens  # Now available!

# Streaming completion
request = LLMRequest(
    model="gemini-1.5-flash",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
)

async for token in llm_service.stream(request):
    print(token, end="")

# Error handling (unified)
from agentcore.a2a_protocol.models.llm import (
    ProviderError,
    ProviderTimeoutError,
)

try:
    response = await llm_service.complete(request)
except ProviderTimeoutError as e:
    print(f"Timeout: {e.timeout_seconds}s")
except ProviderError as e:
    print(f"Provider error: {e.provider}")
```

### Key Improvements

**Token Usage:** Now automatically tracked and reported
**Error Handling:** Unified exceptions across all providers
**Message Format:** Standard OpenAI-style message array

---

## Multi-Provider Applications

### Before (Provider-Specific Code)

```python
# Different code for each provider
async def get_completion(provider: str, prompt: str) -> str:
    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_KEY)
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=ANTHROPIC_KEY)
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.content[0].text

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        return response.text

    raise ValueError(f"Unknown provider: {provider}")
```

### After (Unified Interface)

```python
# Single unified interface
async def get_completion(model: str, prompt: str) -> str:
    request = LLMRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    response = await llm_service.complete(request)
    return response.content

# Use with any provider
result1 = await get_completion("gpt-4.1-mini", "Hello")
result2 = await get_completion("claude-3-5-haiku-20241022", "Hello")
result3 = await get_completion("gemini-1.5-flash", "Hello")
```

---

## Testing Migration

### Unit Test Migration

**Before:**
```python
from unittest.mock import Mock, AsyncMock
import pytest

@pytest.fixture
def mock_openai_client():
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=Mock(
            choices=[Mock(message=Mock(content="Test response"))],
            usage=Mock(total_tokens=100),
        )
    )
    return client

async def test_completion(mock_openai_client):
    response = await mock_openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Test"}],
    )
    assert response.choices[0].message.content == "Test response"
```

**After:**
```python
from unittest.mock import Mock, AsyncMock
from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage
import pytest

@pytest.fixture
def mock_llm_service():
    service = Mock()
    service.complete = AsyncMock(
        return_value=LLMResponse(
            content="Test response",
            usage=LLMUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            latency_ms=100,
            provider="mock",
            model="mock-model",
        )
    )
    return service

async def test_completion(mock_llm_service):
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Test"}],
    )
    response = await mock_llm_service.complete(request)
    assert response.content == "Test response"
```

### Integration Test Migration

**Before:**
```python
@pytest.mark.integration
async def test_openai_integration():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say 'test'"}],
    )

    assert "test" in response.choices[0].message.content.lower()
```

**After:**
```python
@pytest.mark.integration
async def test_llm_integration():
    from agentcore.a2a_protocol.services.llm_service import llm_service

    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say 'test'"}],
    )

    response = await llm_service.complete(request)

    assert "test" in response.content.lower()
    assert response.provider == "openai"
    assert response.usage.total_tokens > 0
```

---

## Rollback Strategy

### Gradual Migration with Feature Flag

```python
from agentcore.a2a_protocol.config import settings

USE_LLM_CLIENT = settings.USE_LLM_CLIENT  # Feature flag

async def get_completion(prompt: str) -> str:
    if USE_LLM_CLIENT:
        # New LLM Client Service
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        response = await llm_service.complete(request)
        return response.content
    else:
        # Old direct SDK
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_KEY)
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

### Canary Deployment

```python
import random

async def get_completion(prompt: str) -> str:
    # Route 10% of traffic to new implementation
    if random.random() < 0.1:
        # LLM Client Service
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        response = await llm_service.complete(request)
        return response.content
    else:
        # Direct SDK (current implementation)
        # ... existing code
        pass
```

---

## Migration Checklist

### Pre-Migration

- [ ] Review LLM Client Service documentation
- [ ] Identify all LLM SDK usage in codebase
- [ ] Set up environment variables (API keys, ALLOWED_MODELS)
- [ ] Configure Prometheus metrics collection
- [ ] Plan rollback strategy

### Migration

- [ ] Install LLM Client Service dependencies
- [ ] Update imports from SDKs to LLM Client Service
- [ ] Convert request format to LLMRequest model
- [ ] Update error handling to use unified exceptions
- [ ] Update tests to use LLM Client mocks
- [ ] Add A2A context (trace_id, source_agent) where applicable

### Testing

- [ ] Unit tests pass with new implementation
- [ ] Integration tests pass with all providers
- [ ] Performance tests show acceptable overhead (<5ms)
- [ ] Error handling works correctly
- [ ] Streaming functionality verified
- [ ] Metrics collection verified in Prometheus

### Deployment

- [ ] Deploy to staging environment
- [ ] Monitor metrics and logs
- [ ] Gradual rollout (canary/feature flag)
- [ ] Monitor error rates and latency
- [ ] Full production deployment
- [ ] Remove old SDK dependencies

### Post-Migration

- [ ] Verify metrics in production
- [ ] Set up Grafana dashboards
- [ ] Configure alerting rules
- [ ] Update documentation
- [ ] Train team on new interface
- [ ] Remove old code and dependencies

---

## Common Migration Issues

### Issue: API Keys Not Loading

**Solution:** Verify environment variables are set correctly. Check `.env` file is loaded.

### Issue: Model Not Allowed

**Solution:** Add model to `ALLOWED_MODELS` configuration.

### Issue: Different Response Format

**Solution:** Use `response.content` instead of provider-specific fields.

### Issue: Streaming Not Working

**Solution:** Use `llm_service.stream()` instead of `llm_service.complete()`.

### Issue: Tests Failing

**Solution:** Update mocks to return `LLMResponse` objects instead of provider-specific responses.

---

## Support

For migration assistance:
- Review [API Reference](./api-reference.md)
- Check [Troubleshooting Guide](./troubleshooting-guide.md)
- Create issue at [GitHub Issues](https://github.com/yourusername/agentcore/issues)
