# LLM Integration Tests

Comprehensive end-to-end integration tests for the LLM client service. These tests validate production readiness by testing real API calls across all three providers (OpenAI, Anthropic, Gemini).

## Overview

The integration test suite (`test_llm_integration.py`) validates:

1. **All 3 Providers** - Real API calls to OpenAI, Anthropic, and Gemini
2. **Streaming Functionality** - End-to-end streaming for each provider
3. **A2A Context Propagation** - Trace ID verification across requests
4. **Error Handling** - Invalid models, timeouts, network errors
5. **Retry Logic** - Transient failure recovery
6. **Concurrent Requests** - 100+ concurrent request handling
7. **Rate Limit Handling** - Graceful degradation under rate limits
8. **Performance Metrics** - Latency and token usage tracking

**Quality Gate:** >95% test success rate required for production release.

## Quick Start

### 1. Setup API Keys

Copy the environment template:

```bash
cp .env.test.template .env.test
```

Edit `.env.test` and add your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

**Important:** Never commit `.env.test` to version control!

### 2. Run All Integration Tests

```bash
# Run all integration tests
uv run pytest tests/integration/test_llm_integration.py -v -m integration

# Run with detailed output
uv run pytest tests/integration/test_llm_integration.py -v -s -m integration

# Run specific test class
uv run pytest tests/integration/test_llm_integration.py::TestProviderIntegrationOpenAI -v
```

### 3. Run Tests Without Coverage

Integration tests can be run without coverage requirements:

```bash
# Skip coverage for integration tests
uv run pytest tests/integration/test_llm_integration.py --no-cov -v
```

## Test Organization

### Test Classes

#### `TestProviderIntegrationOpenAI`
- `test_openai_complete_basic` - Basic completion with real API
- `test_openai_stream_basic` - Basic streaming with real API
- `test_openai_multi_turn_conversation` - Multi-turn conversation handling

#### `TestProviderIntegrationAnthropic`
- `test_anthropic_complete_basic` - Basic completion with real API
- `test_anthropic_stream_basic` - Basic streaming with real API

#### `TestProviderIntegrationGemini`
- `test_gemini_complete_basic` - Basic completion with real API
- `test_gemini_stream_basic` - Basic streaming with real API

#### `TestA2AContextPropagation`
- `test_trace_id_propagation_openai` - Trace ID through OpenAI
- `test_trace_id_propagation_anthropic` - Trace ID through Anthropic
- `test_trace_id_propagation_gemini` - Trace ID through Gemini
- `test_trace_id_without_context` - Requests without A2A context

#### `TestErrorHandling`
- `test_invalid_model_error` - Invalid model rejection
- `test_timeout_error_handling` - Timeout error handling
- `test_invalid_temperature_error` - Invalid parameter validation
- `test_invalid_max_tokens_error` - Invalid parameter validation

#### `TestRetryLogic`
- `test_retry_configuration` - Retry configuration validation
- `test_successful_request_no_retry` - No retry on success

#### `TestConcurrentRequests`
- `test_concurrent_requests_100_openai` - 100 concurrent OpenAI requests
- `test_concurrent_requests_multi_provider` - Concurrent multi-provider requests

#### `TestRateLimitHandling`
- `test_rate_limit_graceful_handling` - Graceful rate limit handling

#### `TestMultiProviderE2E`
- `test_all_providers_complete_workflow` - Complete workflow all providers
- `test_all_providers_streaming_workflow` - Streaming workflow all providers
- `test_global_singleton_instance` - Global singleton validation

#### `TestPerformanceMetrics`
- `test_latency_tracking_accuracy` - Latency tracking validation
- `test_token_usage_accuracy` - Token usage validation

## Skipped Tests

Tests are automatically skipped if API keys are not configured:

```
SKIPPED [OpenAI API key not configured]
SKIPPED [Anthropic API key not configured]
SKIPPED [Google API key not configured]
```

This allows tests to run in CI/CD without failing when keys are not available.

## Running Specific Provider Tests

Test only OpenAI:
```bash
uv run pytest tests/integration/test_llm_integration.py::TestProviderIntegrationOpenAI -v
```

Test only Anthropic:
```bash
uv run pytest tests/integration/test_llm_integration.py::TestProviderIntegrationAnthropic -v
```

Test only Gemini:
```bash
uv run pytest tests/integration/test_llm_integration.py::TestProviderIntegrationGemini -v
```

## Concurrent Request Tests

The concurrent request tests validate production-grade concurrency:

```bash
# Test 100 concurrent OpenAI requests
uv run pytest tests/integration/test_llm_integration.py::TestConcurrentRequests::test_concurrent_requests_100_openai -v -s

# Test multi-provider concurrency
uv run pytest tests/integration/test_llm_integration.py::TestConcurrentRequests::test_concurrent_requests_multi_provider -v -s
```

Expected output:
```
Concurrent test results:
  Total requests: 100
  Successful: 98
  Failed: 2
  Success rate: 98.00%
  Total time: 12.34s
```

## Error Handling Tests

Test error scenarios:

```bash
# Test all error handling
uv run pytest tests/integration/test_llm_integration.py::TestErrorHandling -v

# Test specific error
uv run pytest tests/integration/test_llm_integration.py::TestErrorHandling::test_invalid_model_error -v
```

## Rate Limits

Integration tests may hit provider rate limits. Consider these limits:

### OpenAI Rate Limits
- **Tier 1:** 500 RPM, 30,000 TPM
- **Tier 2:** 5,000 RPM, 450,000 TPM
- See: https://platform.openai.com/account/rate-limits

### Anthropic Rate Limits
- **Tier 1:** 50 RPM, 40,000 TPM
- **Tier 2:** 1,000 RPM, 80,000 TPM
- See: https://console.anthropic.com/settings/limits

### Google Gemini Rate Limits
- **Free tier:** 15 RPM, 1M TPM
- **Paid tier:** 1,000 RPM, 4M TPM
- See: https://ai.google.dev/pricing

### Handling Rate Limits

If you hit rate limits during testing:

1. **Reduce concurrency:** Lower the number in concurrent tests
2. **Add delays:** Use `asyncio.sleep()` between batches
3. **Use test accounts:** Create separate test accounts with limits
4. **Upgrade tier:** Consider higher tier plans for testing

## CI/CD Integration

### GitHub Actions Example

```yaml
name: LLM Integration Tests

on:
  push:
    branches: [main, staging]
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: |
          uv run pytest tests/integration/test_llm_integration.py -v -m integration --no-cov
```

### Required Secrets

Configure these as repository secrets:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Use separate test accounts with usage limits to prevent cost overruns.

## Troubleshooting

### Tests Skip Due to Missing API Keys

```
SKIPPED [OpenAI API key not configured]
```

**Solution:** Set environment variables in `.env.test`

### Rate Limit Errors

```
ProviderError: Provider 'openai' error: Rate limit exceeded
```

**Solution:**
1. Wait a few minutes before retrying
2. Reduce concurrent request count
3. Upgrade provider tier
4. Use test account with higher limits

### Timeout Errors

```
ProviderTimeoutError: Provider 'openai' request timed out after 30.0s
```

**Solution:**
1. Increase timeout in test: `LLMService(timeout=60.0)`
2. Check network connectivity
3. Verify provider API status

### Network Errors

```
ProviderError: Provider 'openai' error: Connection error
```

**Solution:**
1. Check internet connectivity
2. Verify provider API is accessible
3. Check firewall/proxy settings

## Cost Management

Integration tests make real API calls and incur costs:

### Estimated Costs (per test run)

- **OpenAI (gpt-4.1-mini):** ~$0.50 per full test run
- **Anthropic (claude-3-5-haiku):** ~$0.30 per full test run
- **Gemini (gemini-1.5-flash):** ~$0.10 per full test run

**Total:** ~$0.90 per complete integration test run

### Cost Reduction Tips

1. **Test selectively:** Run only changed provider tests
2. **Use cheaper models:** Test with mini/flash models
3. **Reduce max_tokens:** Use small token limits (10-50)
4. **Cache results:** Skip tests if code hasn't changed
5. **Schedule wisely:** Run integration tests only on critical branches

## Success Criteria

For production release, integration tests must meet:

✅ **Success Rate:** >95% of all tests pass
✅ **Coverage:** All 3 providers tested
✅ **Streaming:** All providers stream correctly
✅ **A2A Context:** Trace IDs propagate correctly
✅ **Concurrency:** 100+ concurrent requests succeed
✅ **Error Handling:** Errors handled gracefully
✅ **Performance:** Latency tracking accurate

## Additional Resources

- [LLM Service Documentation](../../src/agentcore/a2a_protocol/services/llm_service.py)
- [Provider Implementation Guides](../../src/agentcore/a2a_protocol/services/)
- [A2A Protocol Specification](../../docs/architecture/a2a-protocol.md)
- [CLAUDE.md - Development Guidelines](../../CLAUDE.md)

## Support

For issues or questions:
1. Check existing integration test logs
2. Review provider API status pages
3. Consult provider documentation
4. Open GitHub issue with test output

---

**Last Updated:** 2025-10-26
**Ticket:** LLM-CLIENT-014 (Integration Tests)
**Status:** Production Ready
