# Troubleshooting Guide

Common errors, solutions, and debugging strategies for the LLM Client Service.

## Table of Contents

- [Configuration Errors](#configuration-errors)
- [API Key Issues](#api-key-issues)
- [Model Governance Errors](#model-governance-errors)
- [Provider Errors](#provider-errors)
- [Timeout Issues](#timeout-issues)
- [Rate Limiting](#rate-limiting)
- [Network and Connectivity](#network-and-connectivity)
- [Streaming Issues](#streaming-issues)
- [Performance Problems](#performance-problems)
- [Debugging Techniques](#debugging-techniques)

---

## Configuration Errors

### Error: "OpenAI API key not configured"

**Symptom:**
```
RuntimeError: OpenAI API key not configured. Set OPENAI_API_KEY environment variable.
```

**Cause:** `OPENAI_API_KEY` environment variable is not set.

**Solution:**

1. **Check environment variable:**
```bash
echo $OPENAI_API_KEY
```

2. **Set environment variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

3. **Add to .env file:**
```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

4. **Verify loading:**
```python
from agentcore.a2a_protocol.config import settings
print(f"OpenAI key configured: {bool(settings.OPENAI_API_KEY)}")
```

---

### Error: "ALLOWED_MODELS is empty"

**Symptom:**
```
ValueError: ALLOWED_MODELS configuration is empty or invalid
```

**Cause:** `ALLOWED_MODELS` environment variable is not set or invalid JSON.

**Solution:**

1. **Set ALLOWED_MODELS:**
```bash
export ALLOWED_MODELS='["gpt-4.1-mini","claude-3-5-haiku-20241022"]'
```

2. **Verify JSON syntax:**
```python
import json
import os

allowed_str = os.environ.get('ALLOWED_MODELS', '[]')
try:
    models = json.loads(allowed_str)
    print(f"Valid JSON: {models}")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

3. **Check configuration loading:**
```python
from agentcore.a2a_protocol.config import settings
print(f"Allowed models: {settings.ALLOWED_MODELS}")
```

---

## API Key Issues

### Error: "Incorrect API key provided"

**Symptom:**
```
ProviderError: Provider 'openai' error: AuthenticationError(...)
```

**Cause:** API key is invalid, expired, or incorrectly formatted.

**Solution:**

1. **Verify API key format:**
   - OpenAI: Starts with `sk-`
   - Anthropic: Starts with `sk-ant-`
   - Google: No specific prefix

2. **Check for whitespace:**
```bash
# Remove any whitespace
export OPENAI_API_KEY=$(echo "sk-..." | tr -d ' \n')
```

3. **Test API key directly:**
```python
import openai

client = openai.OpenAI(api_key="sk-...")
try:
    client.models.list()
    print("API key valid")
except openai.AuthenticationError:
    print("API key invalid")
```

4. **Generate new API key:**
   - Go to provider console
   - Revoke old key
   - Create new key
   - Update environment variable

---

### Error: "You exceeded your current quota"

**Symptom:**
```
ProviderError: Provider 'openai' error: RateLimitError(You exceeded your current quota...)
```

**Cause:** Account has insufficient credits or has reached spending limit.

**Solution:**

1. **Check account balance:**
   - OpenAI: [platform.openai.com/usage](https://platform.openai.com/usage)
   - Anthropic: [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing)

2. **Add credits or update payment method**

3. **Switch to different provider temporarily:**
```python
# Instead of exhausted provider
request = LLMRequest(
    model="gpt-4.1-mini",  # OpenAI quota exhausted
    messages=[...],
)

# Use alternative provider
request = LLMRequest(
    model="claude-3-5-haiku-20241022",  # Switch to Anthropic
    messages=[...],
)
```

---

## Model Governance Errors

### Error: "Model not allowed"

**Symptom:**
```
ModelNotAllowedError: Model 'gpt-3.5-turbo' not allowed. Allowed models: ['gpt-4.1-mini', ...]
```

**Cause:** Requested model is not in `ALLOWED_MODELS` configuration.

**Solution:**

1. **Check current allowed models:**
```python
from agentcore.a2a_protocol.services.llm_service import llm_service

models = llm_service.registry.list_available_models()
print(f"Allowed models: {models}")
```

2. **Add model to ALLOWED_MODELS:**
```bash
export ALLOWED_MODELS='["gpt-4.1-mini","gpt-3.5-turbo"]'
```

3. **Use allowed model instead:**
```python
# Check if model is allowed before using
from agentcore.a2a_protocol.config import settings

model = "gpt-3.5-turbo"
if model not in settings.ALLOWED_MODELS:
    # Fallback to first allowed model
    model = settings.ALLOWED_MODELS[0]

request = LLMRequest(model=model, messages=[...])
```

4. **Monitor governance violations:**
```bash
# Query Prometheus
curl http://localhost:9090/api/v1/query?query=llm_governance_violations_total
```

---

## Provider Errors

### Error: "Provider 'openai' error: BadRequestError"

**Symptom:**
```
ProviderError: Provider 'openai' error: BadRequestError(...)
```

**Common Causes:**

1. **Invalid parameters:**
   - Temperature outside 0.0-2.0 range
   - Negative max_tokens
   - Empty messages array

2. **Model doesn't exist:**
   - Typo in model name
   - Model deprecated or removed

3. **Message format issues:**
   - Missing required fields
   - Invalid role names

**Solutions:**

**Validate request parameters:**
```python
from pydantic import ValidationError
from agentcore.a2a_protocol.models.llm import LLMRequest

try:
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,  # Valid: 0.0-2.0
        max_tokens=100,   # Valid: positive integer
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

**Check model name:**
```python
from agentcore.a2a_protocol.services.llm_service import MODEL_PROVIDER_MAP

model = "gpt-4.1-mini"
if model in MODEL_PROVIDER_MAP:
    print(f"Valid model: {model}")
else:
    print(f"Unknown model: {model}")
    print(f"Available: {list(MODEL_PROVIDER_MAP.keys())}")
```

---

### Error: "Provider 'anthropic' error: APIError"

**Symptom:**
```
ProviderError: Provider 'anthropic' error: APIError(...)
```

**Common Anthropic-Specific Issues:**

1. **Missing max_tokens:**
   Anthropic requires `max_tokens` parameter.

**Solution:**
```python
request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,  # Required for Anthropic
)
```

2. **System message format:**
   Anthropic requires system messages in separate parameter.

**Solution:**
The LLM Client automatically handles this conversion. Ensure system messages use "system" role:
```python
request = LLMRequest(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "system", "content": "You are helpful"},  # Automatically extracted
        {"role": "user", "content": "Hello"},
    ],
)
```

---

## Timeout Issues

### Error: "Request timed out after 60s"

**Symptom:**
```
ProviderTimeoutError: Provider 'openai' request timed out after 60.0s
```

**Cause:** LLM provider did not respond within timeout limit.

**Solutions:**

1. **Increase timeout:**
```bash
export LLM_REQUEST_TIMEOUT=120.0  # 2 minutes
```

2. **Use faster model:**
```python
# Instead of slow premium model
request = LLMRequest(
    model="gpt-5",  # Premium (slow)
    messages=[...],
)

# Use fast tier model
request = LLMRequest(
    model="gpt-4.1-mini",  # Fast tier
    messages=[...],
)
```

3. **Reduce prompt length:**
```python
# Shorten long prompts
long_prompt = "..." * 10000
short_prompt = long_prompt[:5000]  # Truncate

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": short_prompt}],
)
```

4. **Custom timeout per request:**
```python
from agentcore.a2a_protocol.services.llm_service import LLMService

custom_service = LLMService(timeout=180.0)  # 3 minutes
response = await custom_service.complete(request)
```

---

## Rate Limiting

### Error: "Rate limit reached for requests"

**Symptom:**
```
ProviderError: Provider 'openai' error: RateLimitError(Rate limit reached...)
```

**Cause:** Exceeded provider's rate limits (requests per minute, tokens per minute).

**Solutions:**

1. **Implement rate limiting:**
```python
from asyncio import Semaphore
import asyncio

# Limit to 10 concurrent requests
semaphore = Semaphore(10)

async def rate_limited_complete(request: LLMRequest):
    async with semaphore:
        return await llm_service.complete(request)

# Use rate-limited function
response = await rate_limited_complete(request)
```

2. **Add delay between requests:**
```python
import asyncio

for prompt in prompts:
    request = LLMRequest(model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}])
    response = await llm_service.complete(request)

    # Wait before next request
    await asyncio.sleep(1.0)  # 1 second delay
```

3. **Batch processing with delay:**
```python
from asyncio import Semaphore, sleep

async def process_batch(prompts: list[str], batch_size: int = 5, delay: float = 1.0):
    """Process prompts in batches with delay."""

    semaphore = Semaphore(batch_size)

    async def process_single(prompt: str):
        async with semaphore:
            request = LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            response = await llm_service.complete(request)
            await sleep(delay)  # Delay after each request
            return response

    tasks = [process_single(p) for p in prompts]
    return await asyncio.gather(*tasks)

responses = await process_batch(prompts, batch_size=5, delay=0.5)
```

4. **Upgrade rate limits:**
   - Contact provider to increase limits
   - Upgrade to higher tier plan

---

## Network and Connectivity

### Error: "Connection error"

**Symptom:**
```
ProviderError: Provider 'openai' error: APIConnectionError(Connection error...)
```

**Causes:**
- Network connectivity issues
- Firewall blocking requests
- Proxy configuration problems
- Provider API temporarily down

**Solutions:**

1. **Check network connectivity:**
```bash
# Test connectivity to provider
ping api.openai.com
ping api.anthropic.com

# Test HTTPS access
curl -I https://api.openai.com/v1/models
```

2. **Check firewall rules:**
```bash
# Allow outbound HTTPS
sudo ufw allow out 443/tcp
```

3. **Configure proxy:**
```python
import os

# Set proxy for requests
os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
```

4. **Retry on connection errors:**
The LLM Client automatically retries connection errors with exponential backoff (3 attempts by default).

5. **Check provider status:**
   - OpenAI: [status.openai.com](https://status.openai.com)
   - Anthropic: [status.anthropic.com](https://status.anthropic.com)
   - Google: [status.cloud.google.com](https://status.cloud.google.com)

---

## Streaming Issues

### Error: "Streaming not producing tokens"

**Symptom:** Stream iterator doesn't yield any tokens or stops mid-stream.

**Causes:**
- Provider streaming API issues
- Network interruption
- Model generating empty response

**Solutions:**

1. **Add timeout to streaming:**
```python
import asyncio

async def stream_with_timeout(request: LLMRequest, timeout: float = 60.0):
    """Stream with timeout per token."""

    async for token in llm_service.stream(request):
        try:
            yield await asyncio.wait_for(asyncio.coroutine(lambda: token)(), timeout=timeout)
        except asyncio.TimeoutError:
            print("Token timeout - ending stream")
            break
```

2. **Handle stream interruptions:**
```python
try:
    full_response = ""
    async for token in llm_service.stream(request):
        full_response += token
        print(token, end="", flush=True)
except (ProviderError, ProviderTimeoutError) as e:
    print(f"\nStream interrupted: {e}")
    print(f"Partial response: {full_response}")
```

3. **Use non-streaming as fallback:**
```python
try:
    async for token in llm_service.stream(request):
        print(token, end="", flush=True)
except Exception as e:
    print(f"\nStreaming failed, falling back to non-streaming: {e}")
    response = await llm_service.complete(request)
    print(response.content)
```

---

## Performance Problems

### Issue: "High latency"

**Symptoms:**
- Requests taking longer than expected
- Timeout errors
- Slow response times

**Diagnosis:**

1. **Check Prometheus metrics:**
```promql
# Average latency by provider
rate(llm_requests_duration_seconds_sum[5m]) / rate(llm_requests_duration_seconds_count[5m])

# P95 latency
histogram_quantile(0.95, rate(llm_requests_duration_seconds_bucket[5m]))
```

2. **Enable debug logging:**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('agentcore.a2a_protocol.services.llm_service')
logger.setLevel(logging.DEBUG)
```

**Solutions:**

1. **Use faster models:**
```python
# Instead of premium models
model = "gpt-5"  # Premium (slower)

# Use fast tier
model = "gpt-4.1-mini"  # Fast tier (much faster)
```

2. **Reduce max_tokens:**
```python
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[...],
    max_tokens=200,  # Limit response length
)
```

3. **Implement caching:**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
async def cached_complete(prompt_hash: str, model: str):
    """Cache LLM responses."""
    # Implementation omitted for brevity
    pass
```

4. **Use streaming for perceived performance:**
```python
# Streaming reduces time to first token
async for token in llm_service.stream(request):
    print(token, end="", flush=True)
```

---

## Debugging Techniques

### Enable Debug Logging

```python
import logging
import structlog

# Configure structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)

# Enable DEBUG level
logging.basicConfig(level=logging.DEBUG)

# All LLM operations will now log detailed information
```

### Inspect Request/Response

```python
from agentcore.a2a_protocol.models.llm import LLMRequest

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello"}],
    trace_id="debug-trace-123",
)

# Log request
print("Request:")
print(f"  Model: {request.model}")
print(f"  Messages: {request.messages}")
print(f"  Temperature: {request.temperature}")
print(f"  Trace ID: {request.trace_id}")

try:
    response = await llm_service.complete(request)

    # Log response
    print("\nResponse:")
    print(f"  Content: {response.content}")
    print(f"  Provider: {response.provider}")
    print(f"  Latency: {response.latency_ms}ms")
    print(f"  Tokens: {response.usage.total_tokens}")

except Exception as e:
    print(f"\nError: {type(e).__name__}")
    print(f"  Message: {e}")
    import traceback
    traceback.print_exc()
```

### Check Prometheus Metrics

```bash
# Query metrics directly
curl http://localhost:9090/api/v1/query?query=llm_requests_total

# Check error rate
curl http://localhost:9090/api/v1/query?query=rate(llm_errors_total[5m])
```

### Trace Request Flow

```python
import uuid

# Generate unique trace ID
trace_id = f"trace-{uuid.uuid4()}"

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello"}],
    trace_id=trace_id,
    source_agent="debug-agent",
)

# All logs will include trace_id - search logs:
# grep "trace-<uuid>" /var/log/agentcore.log
```

### Test Individual Providers

```python
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI

# Test OpenAI directly
client = LLMClientOpenAI(api_key="sk-...", timeout=60.0, max_retries=3)

request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Test"}],
)

try:
    response = await client.complete(request)
    print("OpenAI client working")
except Exception as e:
    print(f"OpenAI client error: {e}")
```

---

## Getting Help

If issues persist after trying these solutions:

1. **Check logs:** Review application logs for detailed error messages
2. **Check metrics:** Review Prometheus metrics for patterns
3. **Provider status:** Check provider status pages
4. **GitHub Issues:** Search or create issue at [github.com/yourusername/agentcore/issues](https://github.com/yourusername/agentcore/issues)
5. **Documentation:** Review [API Reference](./api-reference.md) and [Configuration Guide](./configuration-guide.md)

**When reporting issues, include:**
- Error message and full traceback
- Request parameters (redact API keys)
- Environment configuration (model, timeout, retries)
- Provider status at time of error
- Relevant logs with trace_id
