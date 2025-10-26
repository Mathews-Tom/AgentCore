# Usage Examples

Executable code examples for common LLM Client Service use cases.

## Table of Contents

- [Basic Completion](#basic-completion)
- [Streaming Responses](#streaming-responses)
- [Multi-Provider Usage](#multi-provider-usage)
- [Error Handling](#error-handling)
- [A2A Context Propagation](#a2a-context-propagation)
- [Model Selection](#model-selection)
- [JSON-RPC Integration](#json-rpc-integration)
- [Batch Processing](#batch-processing)
- [Custom Timeout and Retry](#custom-timeout-and-retry)
- [Testing and Mocking](#testing-and-mocking)

---

## Basic Completion

### Simple Question-Answer

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

# Create request
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)

# Execute completion
response = await llm_service.complete(request)

print(response.content)  # "The capital of France is Paris."
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Latency: {response.latency_ms}ms")
print(f"Provider: {response.provider}")
```

### Multi-Turn Conversation

```python
# Maintain conversation history
conversation = [
    {"role": "system", "content": "You are a helpful Python programming assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
]

request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=conversation,
    temperature=0.7,
)

response = await llm_service.complete(request)
print(response.content)

# Continue conversation
conversation.append({"role": "assistant", "content": response.content})
conversation.append({"role": "user", "content": "Can you show me an async example?"})

request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=conversation,
)

response = await llm_service.complete(request)
print(response.content)
```

### Temperature Control

```python
# Low temperature (more deterministic)
creative_request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Write a creative story about a robot."}],
    temperature=0.2,  # More focused, less random
)

# High temperature (more creative)
factual_request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "List the planets in our solar system."}],
    temperature=0.9,  # More creative, more variation
)

creative_response = await llm_service.complete(creative_request)
factual_response = await llm_service.complete(factual_request)
```

### Max Tokens Limit

```python
# Limit response length
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Explain quantum computing in detail."}],
    max_tokens=100,  # Limit to roughly 100 tokens
)

response = await llm_service.complete(request)
print(f"Completion tokens: {response.usage.completion_tokens}")  # Will be ≤ 100
```

---

## Streaming Responses

### Basic Streaming

```python
from agentcore.a2a_protocol.models.llm import LLMRequest

request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Write a short poem about coding."}],
    stream=True,
)

# Stream tokens as they arrive
async for token in llm_service.stream(request):
    print(token, end="", flush=True)

print()  # New line after streaming completes
```

### Streaming with Accumulation

```python
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Explain async/await in Python."}],
    stream=True,
)

# Accumulate full response while streaming
full_response = ""
async for token in llm_service.stream(request):
    print(token, end="", flush=True)
    full_response += token

print(f"\n\nFull response length: {len(full_response)} characters")
```

### Streaming with Progress Callback

```python
import time

request = LLMRequest(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Write a detailed explanation of recursion."}],
    stream=True,
)

start_time = time.time()
token_count = 0

async for token in llm_service.stream(request):
    token_count += 1
    elapsed = time.time() - start_time

    # Progress indicator every 10 tokens
    if token_count % 10 == 0:
        tokens_per_sec = token_count / elapsed
        print(f"\n[{token_count} tokens, {tokens_per_sec:.1f} tokens/sec]", end="")

    print(token, end="", flush=True)

print(f"\n\nTotal tokens: {token_count}")
print(f"Total time: {elapsed:.2f}s")
```

---

## Multi-Provider Usage

### Provider Comparison

```python
# Same prompt to different providers
prompt = "Explain the difference between lists and tuples in Python."

providers = [
    ("gpt-4.1-mini", "OpenAI"),
    ("claude-3-5-haiku-20241022", "Anthropic"),
    ("gemini-1.5-flash", "Gemini"),
]

for model, provider_name in providers:
    request = LLMRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    response = await llm_service.complete(request)

    print(f"\n{'='*60}")
    print(f"Provider: {provider_name} ({model})")
    print(f"Latency: {response.latency_ms}ms")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"{'='*60}")
    print(response.content)
```

### Automatic Provider Selection by Model Tier

```python
from agentcore.a2a_protocol.models.llm import ModelTier

# Model tier mapping (simplified example)
TIER_MODELS = {
    ModelTier.FAST: "gpt-4.1-mini",
    ModelTier.BALANCED: "claude-3-5-haiku-20241022",
    ModelTier.PREMIUM: "gpt-5",
}

async def complete_with_tier(prompt: str, tier: ModelTier) -> str:
    """Complete request using model from specified tier."""
    model = TIER_MODELS[tier]

    request = LLMRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    response = await llm_service.complete(request)
    return response.content

# Use fast model for simple queries
answer = await complete_with_tier(
    "What is 2+2?",
    ModelTier.FAST
)

# Use premium model for complex reasoning
analysis = await complete_with_tier(
    "Analyze the time complexity of quicksort and explain when it degrades to O(n²).",
    ModelTier.PREMIUM
)
```

---

## Error Handling

### Handling Model Governance Violations

```python
from agentcore.a2a_protocol.models.llm import ModelNotAllowedError

try:
    request = LLMRequest(
        model="gpt-3.5-turbo",  # Not in ALLOWED_MODELS
        messages=[{"role": "user", "content": "Hello"}],
    )

    response = await llm_service.complete(request)

except ModelNotAllowedError as e:
    print(f"Model '{e.model}' is not allowed")
    print(f"Allowed models: {e.allowed}")

    # Fallback to allowed model
    fallback_request = LLMRequest(
        model=e.allowed[0],  # Use first allowed model
        messages=[{"role": "user", "content": "Hello"}],
    )
    response = await llm_service.complete(fallback_request)
```

### Handling Provider Errors

```python
from agentcore.a2a_protocol.models.llm import ProviderError, ProviderTimeoutError

try:
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )

    response = await llm_service.complete(request)

except ProviderTimeoutError as e:
    print(f"Request to {e.provider} timed out after {e.timeout_seconds}s")
    # Retry with different provider or increased timeout

except ProviderError as e:
    print(f"Provider {e.provider} error: {e.original_error}")
    # Log error, try different provider, or notify user

except RuntimeError as e:
    print(f"Configuration error: {e}")
    # Check API key configuration
```

### Retry with Exponential Backoff

```python
import asyncio

async def complete_with_retry(
    request: LLMRequest,
    max_attempts: int = 3,
    base_delay: float = 1.0
) -> LLMResponse:
    """Complete request with custom retry logic."""

    for attempt in range(max_attempts):
        try:
            return await llm_service.complete(request)

        except (ProviderError, ProviderTimeoutError) as e:
            if attempt == max_attempts - 1:
                raise  # Last attempt failed

            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            await asyncio.sleep(delay)

    raise RuntimeError("Should never reach here")

# Use custom retry logic
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

response = await complete_with_retry(request)
```

---

## A2A Context Propagation

### Distributed Tracing

```python
import uuid

# Generate trace ID for request tracking
trace_id = f"trace-{uuid.uuid4()}"

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Explain microservices"}],
    trace_id=trace_id,
    source_agent="agent-001",
    session_id="session-abc-123",
)

response = await llm_service.complete(request)

# Trace ID propagated to response
assert response.trace_id == trace_id

# Check logs for trace_id to track request flow
# All logs will include: trace_id, source_agent, session_id
```

### Agent-to-Agent Communication

```python
# Agent A sends request to LLM
agent_a_request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Analyze this code"}],
    trace_id="trace-multi-agent-001",
    source_agent="agent-analyzer",
    session_id="session-code-review",
)

response_a = await llm_service.complete(agent_a_request)

# Agent B processes result and sends follow-up
agent_b_request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": f"Summarize this analysis: {response_a.content}"}
    ],
    trace_id="trace-multi-agent-001",  # Same trace ID
    source_agent="agent-summarizer",
    session_id="session-code-review",  # Same session
)

response_b = await llm_service.complete(agent_b_request)

# Both requests tracked under same trace_id in logs and metrics
```

---

## Model Selection

### List Available Models

```python
from agentcore.a2a_protocol.services.llm_service import llm_service

# Get all allowed models
models = llm_service.registry.list_available_models()
print("Available models:")
for model in models:
    print(f"  - {model}")
```

### Dynamic Model Selection

```python
def select_model_for_task(task_complexity: str) -> str:
    """Select appropriate model based on task complexity."""

    models = llm_service.registry.list_available_models()

    if task_complexity == "simple":
        # Use fastest available model
        for model in ["gpt-4.1-mini", "gemini-1.5-flash", "claude-3-5-haiku-20241022"]:
            if model in models:
                return model

    elif task_complexity == "moderate":
        # Use balanced model
        for model in ["claude-3-5-haiku-20241022", "gpt-5-mini"]:
            if model in models:
                return model

    else:  # complex
        # Use premium model
        for model in ["gpt-5", "claude-3-5-sonnet", "gemini-1.5-pro"]:
            if model in models:
                return model

    # Fallback to first available
    return models[0]

# Use dynamic selection
model = select_model_for_task("complex")
request = LLMRequest(
    model=model,
    messages=[{"role": "user", "content": "Solve this complex problem..."}],
)

response = await llm_service.complete(request)
```

---

## JSON-RPC Integration

### HTTP Client Example

```python
import httpx

async def llm_complete_jsonrpc(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> dict:
    """Call LLM completion via JSON-RPC."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "llm.complete",
                "params": {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                },
                "id": 1,
            },
        )

        result = response.json()

        if "error" in result:
            raise RuntimeError(f"JSON-RPC error: {result['error']}")

        return result["result"]

# Use JSON-RPC client
result = await llm_complete_jsonrpc(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

print(result["content"])
print(f"Tokens: {result['usage']['total_tokens']}")
```

### List Models via JSON-RPC

```python
import httpx

async def list_models_jsonrpc() -> list[str]:
    """List allowed models via JSON-RPC."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "llm.models",
                "params": {},
                "id": 2,
            },
        )

        result = response.json()
        return result["result"]["allowed_models"]

models = await list_models_jsonrpc()
print(f"Available models: {models}")
```

---

## Batch Processing

### Process Multiple Prompts

```python
import asyncio

prompts = [
    "Explain async/await",
    "What are Python decorators?",
    "How do generators work?",
    "Explain list comprehensions",
]

async def process_batch(prompts: list[str], model: str) -> list[str]:
    """Process multiple prompts concurrently."""

    async def process_single(prompt: str) -> str:
        request = LLMRequest(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        response = await llm_service.complete(request)
        return response.content

    # Process all prompts concurrently
    tasks = [process_single(p) for p in prompts]
    return await asyncio.gather(*tasks)

# Process batch
results = await process_batch(prompts, "gpt-4.1-mini")

for prompt, result in zip(prompts, results):
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result[:100]}...")  # First 100 chars
```

### Rate-Limited Batch Processing

```python
import asyncio
from asyncio import Semaphore

async def process_batch_rate_limited(
    prompts: list[str],
    model: str,
    max_concurrent: int = 5,
) -> list[str]:
    """Process prompts with concurrency limit."""

    semaphore = Semaphore(max_concurrent)

    async def process_single(prompt: str) -> str:
        async with semaphore:  # Limit concurrent requests
            request = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            response = await llm_service.complete(request)
            return response.content

    tasks = [process_single(p) for p in prompts]
    return await asyncio.gather(*tasks)

# Process with rate limiting (max 5 concurrent)
results = await process_batch_rate_limited(prompts, "gpt-4.1-mini", max_concurrent=5)
```

---

## Custom Timeout and Retry

### Custom Service Instance

```python
from agentcore.a2a_protocol.services.llm_service import LLMService

# Create custom service with different settings
custom_service = LLMService(
    timeout=120.0,  # 2 minutes timeout
    max_retries=5,  # 5 retry attempts
)

request = LLMRequest(
    model="gpt-5",
    messages=[{"role": "user", "content": "Complex reasoning task..."}],
)

# Use custom service (longer timeout, more retries)
response = await custom_service.complete(request)
```

### Per-Request Timeout

```python
import asyncio

async def complete_with_timeout(
    request: LLMRequest,
    timeout_seconds: float = 30.0
) -> LLMResponse:
    """Complete request with custom timeout."""

    try:
        return await asyncio.wait_for(
            llm_service.complete(request),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise ProviderTimeoutError("client", timeout_seconds)

# Use per-request timeout
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

response = await complete_with_timeout(request, timeout_seconds=10.0)
```

---

## Testing and Mocking

### Mock LLM Service for Tests

```python
from unittest.mock import AsyncMock, Mock
import pytest

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""

    service = Mock()

    # Mock complete method
    service.complete = AsyncMock(
        return_value=LLMResponse(
            content="Mocked response",
            usage=LLMUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            latency_ms=100,
            provider="mock",
            model="mock-model",
            trace_id="mock-trace",
        )
    )

    # Mock stream method
    async def mock_stream(request):
        for token in ["Mock ", "streaming ", "response"]:
            yield token

    service.stream = Mock(side_effect=mock_stream)

    return service

# Use in tests
async def test_my_function(mock_llm_service):
    # Inject mock service
    result = await my_function_using_llm(mock_llm_service)

    assert result == "expected output"
    mock_llm_service.complete.assert_called_once()
```

### Integration Test Example

```python
import pytest
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

@pytest.mark.asyncio
@pytest.mark.integration
async def test_llm_completion_integration():
    """Integration test for LLM completion (requires API keys)."""

    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
        temperature=0.0,  # Deterministic
        max_tokens=10,
    )

    response = await llm_service.complete(request)

    # Verify response structure
    assert isinstance(response.content, str)
    assert response.usage.total_tokens > 0
    assert response.latency_ms > 0
    assert response.provider == "openai"
    assert response.model == "gpt-4.1-mini"

    # Verify content (should be close to "test")
    assert "test" in response.content.lower()
```

---

## Advanced Patterns

### Caching Responses

```python
from functools import lru_cache
import hashlib

class LLMCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self):
        self._cache: dict[str, LLMResponse] = {}

    def _get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key from request."""
        key_data = f"{request.model}:{request.messages}:{request.temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete with caching."""
        cache_key = self._get_cache_key(request)

        if cache_key in self._cache:
            return self._cache[cache_key]

        response = await llm_service.complete(request)
        self._cache[cache_key] = response
        return response

# Use cached service
cache = LLMCache()

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)

# First call - hits LLM
response1 = await cache.complete(request)

# Second call - cached
response2 = await cache.complete(request)

assert response1.content == response2.content
```

### Response Validation

```python
from pydantic import BaseModel, ValidationError

class ValidatedResponse(BaseModel):
    """Expected response format."""
    answer: str
    confidence: float

async def complete_with_validation(
    request: LLMRequest,
    response_model: type[BaseModel]
) -> BaseModel:
    """Complete and validate response format."""

    response = await llm_service.complete(request)

    try:
        # Parse and validate response
        return response_model.model_validate_json(response.content)
    except ValidationError as e:
        raise ValueError(f"Invalid response format: {e}")

# Use with validation
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{
        "role": "user",
        "content": "Respond in JSON: {\"answer\": \"Paris\", \"confidence\": 0.99}"
    }],
)

validated = await complete_with_validation(request, ValidatedResponse)
print(f"Answer: {validated.answer}, Confidence: {validated.confidence}")
```

---

All examples are production-ready and follow AgentCore coding standards. Adapt as needed for your specific use case.
