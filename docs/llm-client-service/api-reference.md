# API Reference

Complete API documentation for the LLM Client Service including Python classes, methods, and JSON-RPC endpoints.

## Table of Contents

- [Python API](#python-api)
  - [LLMService](#llmservice)
  - [ProviderRegistry](#providerregistry)
  - [LLMClient (Abstract Base)](#llmclient-abstract-base)
  - [Provider Implementations](#provider-implementations)
  - [Data Models](#data-models)
  - [Exceptions](#exceptions)
- [JSON-RPC API](#json-rpc-api)
  - [llm.complete](#llmcomplete)
  - [llm.stream](#llmstream)
  - [llm.models](#llmmodels)
  - [llm.metrics](#llmmetrics)

---

## Python API

### LLMService

**Module:** `agentcore.a2a_protocol.services.llm_service`

Main facade for multi-provider LLM operations. Provides unified interface for completions with model governance, A2A context propagation, and metrics collection.

#### Class Definition

```python
class LLMService:
    def __init__(
        self,
        timeout: float | None = None,
        max_retries: int | None = None
    ) -> None:
        """Initialize LLM service with optional timeout and retry configuration.

        Args:
            timeout: Request timeout in seconds (default from settings.LLM_REQUEST_TIMEOUT)
            max_retries: Maximum retry attempts (default from settings.LLM_MAX_RETRIES)
        """
```

#### Methods

##### complete()

Execute non-streaming LLM completion with model governance and A2A context.

```python
async def complete(self, request: LLMRequest) -> LLMResponse:
    """Execute non-streaming LLM completion.

    Args:
        request: Unified LLM request with model, messages, temperature, max_tokens,
            and A2A context (trace_id, source_agent, session_id).

    Returns:
        Normalized LLM response with content, usage statistics, latency,
        provider information, and propagated A2A context.

    Raises:
        ModelNotAllowedError: When request.model is not in ALLOWED_MODELS
        ProviderError: When provider API returns an error
        ProviderTimeoutError: When request exceeds timeout limit
        RuntimeError: When provider API key is not configured
    """
```

**Example:**

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Explain async/await"}],
    temperature=0.7,
    max_tokens=200,
    trace_id="trace-abc-123",
)

response = await llm_service.complete(request)
# response.content: str - Generated text
# response.usage: LLMUsage - Token statistics
# response.latency_ms: int - Request latency
# response.provider: str - Provider name
# response.model: str - Model used
# response.trace_id: str | None - Propagated trace ID
```

##### stream()

Execute streaming LLM completion with model governance and A2A context.

```python
async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
    """Execute streaming LLM completion.

    Args:
        request: Unified LLM request with model, messages, temperature, max_tokens,
            and A2A context (trace_id, source_agent, session_id).

    Yields:
        Content tokens as strings. Each yield represents a chunk of generated text.

    Raises:
        ModelNotAllowedError: When request.model is not in ALLOWED_MODELS
        ProviderError: When provider API returns an error during streaming
        ProviderTimeoutError: When stream does not produce tokens within timeout
        RuntimeError: When provider API key is not configured
    """
```

**Example:**

```python
request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True,
    trace_id="trace-xyz-789",
)

async for token in llm_service.stream(request):
    print(token, end="", flush=True)
```

#### Global Instance

```python
# Pre-instantiated global singleton
from agentcore.a2a_protocol.services.llm_service import llm_service

# Use directly without instantiation
response = await llm_service.complete(request)
```

---

### ProviderRegistry

**Module:** `agentcore.a2a_protocol.services.llm_service`

Provider registry managing model-to-provider mapping and provider instances using singleton pattern with lazy initialization.

#### Class Definition

```python
class ProviderRegistry:
    def __init__(self, timeout: float = 60.0, max_retries: int = 3) -> None:
        """Initialize provider registry with timeout and retry configuration.

        Args:
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum number of retry attempts (default 3)
        """
```

#### Methods

##### get_provider_for_model()

Get provider client for the specified model.

```python
def get_provider_for_model(self, model: str) -> LLMClient:
    """Get provider client for the specified model.

    Args:
        model: Model identifier (e.g., "gpt-4.1-mini", "claude-3-5-haiku-20241022")

    Returns:
        LLMClient instance for the provider that handles this model

    Raises:
        ModelNotAllowedError: When model is not in ALLOWED_MODELS
        ValueError: When model is unknown (not in MODEL_PROVIDER_MAP)
        RuntimeError: When API key is not configured for the provider
    """
```

##### list_available_models()

List all available models based on ALLOWED_MODELS configuration.

```python
def list_available_models(self) -> list[str]:
    """List all available models.

    Returns intersection of MODEL_PROVIDER_MAP keys and ALLOWED_MODELS.

    Returns:
        List of available model identifiers sorted alphabetically
    """
```

**Example:**

```python
from agentcore.a2a_protocol.services.llm_service import ProviderRegistry

registry = ProviderRegistry(timeout=60.0, max_retries=3)

# Get provider for specific model
client = registry.get_provider_for_model("gpt-4.1-mini")

# List all available models
models = registry.list_available_models()
print(models)  # ["claude-3-5-haiku-20241022", "gpt-4.1-mini", ...]
```

---

### LLMClient (Abstract Base)

**Module:** `agentcore.a2a_protocol.services.llm_client_base`

Abstract base class defining the contract for all provider implementations.

#### Class Definition

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

class LLMClient(ABC):
    """Abstract base class for all LLM provider implementations."""
```

#### Abstract Methods

##### complete()

```python
@abstractmethod
async def complete(self, request: LLMRequest) -> LLMResponse:
    """Execute non-streaming completion request and return normalized response.

    Args:
        request: Unified LLM request with model, messages, and parameters

    Returns:
        Normalized LLM response with content, usage, and latency

    Raises:
        ProviderError: For general provider API errors
        ProviderTimeoutError: When request exceeds timeout
        ModelNotAllowedError: When request.model is not in ALLOWED_MODELS
    """
```

##### stream()

```python
@abstractmethod
async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
    """Execute streaming completion and yield tokens as they are generated.

    Args:
        request: Unified LLM request with model, messages, and parameters

    Yields:
        Content tokens as strings in order of generation

    Raises:
        ProviderError: For general provider API errors during streaming
        ProviderTimeoutError: When stream does not produce tokens within timeout
        ModelNotAllowedError: When request.model is not in ALLOWED_MODELS
    """
```

##### _normalize_response()

```python
@abstractmethod
def _normalize_response(
    self, raw_response: object, request: LLMRequest
) -> LLMResponse:
    """Normalize provider-specific response to unified LLMResponse format.

    Args:
        raw_response: Provider-specific response object
        request: Original LLM request for context propagation

    Returns:
        Normalized LLM response in unified format

    Raises:
        ValueError: When raw_response is malformed or missing required fields
    """
```

---

### Provider Implementations

#### LLMClientOpenAI

**Module:** `agentcore.a2a_protocol.services.llm_client_openai`

OpenAI provider implementation supporting GPT-4.1, GPT-5 models.

```python
class LLMClientOpenAI(LLMClient):
    def __init__(
        self, api_key: str, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key for authentication
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum retry attempts on transient errors (default 3)
        """
```

**Supported Models:**
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-5`
- `gpt-5-mini`

**Retry Behavior:**
- Transient errors (rate limits, connection issues): Exponential backoff (1s, 2s, 4s)
- Terminal errors (authentication, bad request): No retry

#### LLMClientAnthropic

**Module:** `agentcore.a2a_protocol.services.llm_client_anthropic`

Anthropic Claude provider implementation supporting Claude 3.5 models.

```python
class LLMClientAnthropic(LLMClient):
    def __init__(
        self, api_key: str, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key for authentication
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum retry attempts on transient errors (default 3)
        """
```

**Supported Models:**
- `claude-3-5-sonnet`
- `claude-3-5-haiku-20241022`
- `claude-3-opus`

**Message Format Conversion:**
- System messages extracted to separate `system` parameter
- Only "user" and "assistant" roles in messages array

#### LLMClientGemini

**Module:** `agentcore.a2a_protocol.services.llm_client_gemini`

Google Gemini provider implementation supporting Gemini 2.0, 1.5 models.

```python
class LLMClientGemini(LLMClient):
    def __init__(
        self, api_key: str, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        """Initialize Gemini client.

        Args:
            api_key: Google API key for authentication
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum retry attempts on transient errors (default 3)
        """
```

**Supported Models:**
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

---

### Data Models

#### LLMRequest

**Module:** `agentcore.a2a_protocol.models.llm`

Unified LLM request model for all providers.

```python
class LLMRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7  # Range: 0.0-2.0
    max_tokens: int | None = None  # Must be positive
    stream: bool = False

    # A2A Context fields
    trace_id: str | None = None
    source_agent: str | None = None
    session_id: str | None = None
```

**Field Validation:**
- `temperature`: Clamped to 0.0-2.0 range
- `max_tokens`: Must be positive integer if provided
- `messages`: Non-empty list required

#### LLMResponse

**Module:** `agentcore.a2a_protocol.models.llm`

Unified LLM response model from all providers.

```python
class LLMResponse(BaseModel):
    content: str
    usage: LLMUsage
    latency_ms: int
    provider: str
    model: str

    # A2A Context propagation
    trace_id: str | None = None
```

#### LLMUsage

**Module:** `agentcore.a2a_protocol.models.llm`

Token usage statistics from LLM response.

```python
class LLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

#### Provider

**Module:** `agentcore.a2a_protocol.models.llm`

LLM provider enumeration.

```python
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
```

#### ModelTier

**Module:** `agentcore.a2a_protocol.models.llm`

Model tier classification for runtime selection.

```python
class ModelTier(str, Enum):
    FAST = "fast"        # Low latency, cost-effective
    BALANCED = "balanced"  # Balance of quality and cost
    PREMIUM = "premium"   # Highest quality
```

---

### Exceptions

#### ModelNotAllowedError

Raised when requested model is not in ALLOWED_MODELS configuration.

```python
class ModelNotAllowedError(Exception):
    def __init__(self, model: str, allowed: list[str]) -> None:
        """Initialize model not allowed error.

        Attributes:
            model: The requested model that was rejected
            allowed: List of allowed models from configuration
        """
```

#### ProviderError

Raised when provider API returns an error.

```python
class ProviderError(Exception):
    def __init__(self, provider: str, original_error: Exception) -> None:
        """Initialize provider error.

        Attributes:
            provider: The provider that raised the error
            original_error: The original exception from the provider SDK
        """
```

#### ProviderTimeoutError

Raised when provider request exceeds timeout limit.

```python
class ProviderTimeoutError(Exception):
    def __init__(self, provider: str, timeout_seconds: float) -> None:
        """Initialize provider timeout error.

        Attributes:
            provider: The provider that timed out
            timeout_seconds: The timeout value that was exceeded
        """
```

---

## JSON-RPC API

All JSON-RPC methods follow the JSON-RPC 2.0 specification. Endpoint: `POST /api/v1/jsonrpc`

### llm.complete

Execute non-streaming LLM completion.

**Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "llm.complete",
  "params": {
    "model": "gpt-4.1-mini",
    "messages": [
      {"role": "user", "content": "Explain async/await"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  },
  "id": 1
}
```

**Parameters:**
- `model` (string, required): Model identifier (must be in ALLOWED_MODELS)
- `messages` (array, required): Conversation messages
- `temperature` (number, optional, default 0.7): Sampling temperature (0.0-2.0)
- `max_tokens` (integer, optional): Maximum tokens to generate
- `stream` (boolean, optional, default false): Enable streaming (not supported via JSON-RPC)

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": "Async/await is a syntax for writing...",
    "usage": {
      "prompt_tokens": 12,
      "completion_tokens": 45,
      "total_tokens": 57
    },
    "latency_ms": 1234,
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "trace_id": "trace-abc-123"
  },
  "id": 1
}
```

**Errors:**
- `-32602` (Invalid params): Model not allowed or validation failed
- `-32603` (Internal error): Provider error or timeout

---

### llm.stream

Execute streaming LLM completion (not directly supported via JSON-RPC).

**Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "llm.stream",
  "params": {
    "model": "claude-3-5-haiku-20241022",
    "messages": [{"role": "user", "content": "Write a story"}]
  },
  "id": 2
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "error": "Streaming not supported via JSON-RPC",
    "message": "Use WebSocket or Server-Sent Events (SSE) endpoints for streaming completions",
    "alternatives": [
      "WebSocket: /ws/llm/stream",
      "SSE: /api/v1/llm/stream (HTTP streaming)"
    ],
    "note": "For non-streaming completions, use llm.complete method"
  },
  "id": 2
}
```

---

### llm.models

List allowed LLM models.

**Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "llm.models",
  "params": {},
  "id": 3
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "allowed_models": [
      "claude-3-5-haiku-20241022",
      "gemini-1.5-flash",
      "gpt-4.1-mini"
    ],
    "default_model": "gpt-4.1-mini",
    "count": 3
  },
  "id": 3
}
```

---

### llm.metrics

Get current LLM metrics snapshot.

**Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "llm.metrics",
  "params": {},
  "id": 4
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "total_requests": 1234,
    "total_tokens": 567890,
    "by_provider": {
      "openai": {
        "requests": 800,
        "tokens": 400000
      },
      "anthropic": {
        "requests": 300,
        "tokens": 120000
      },
      "gemini": {
        "requests": 134,
        "tokens": 47890
      }
    },
    "governance_violations": 5
  },
  "id": 4
}
```

---

## Type Annotations

All Python code uses strict type annotations compatible with `mypy --strict`:

```python
# Built-in generics (preferred per CLAUDE.md)
list[str]
dict[str, Any]
int | None
tuple[int, str]

# Collections.abc for protocols
from collections.abc import AsyncIterator, Callable

# Pydantic models for validation
from pydantic import BaseModel, Field
```

## Thread Safety

- **LLMService**: Thread-safe for concurrent async operations
- **ProviderRegistry**: Singleton pattern with lazy initialization (thread-safe)
- **Provider clients**: Individual instances are thread-safe for async usage
- **Metrics**: Prometheus client handles concurrent updates

## Performance Notes

- Abstraction overhead: <5ms compared to direct SDK usage
- Retry logic: May extend total latency on transient errors (exponential backoff)
- Streaming: Tokens yielded immediately (no buffering)
- Metrics collection: Minimal overhead (<1ms per request)
