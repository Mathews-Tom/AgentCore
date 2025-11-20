# External API Integration Tests

## Overview

Tests marked with `@pytest.mark.external_api` depend on external third-party services and may be unreliable due to service availability, rate limiting, or network issues.

## Current External API Tests

### REST API Tool Tests (`test_rest_api_tool_integration.py`)
- **External Service**: httpbin.org
- **Purpose**: Validate REST API tool with real HTTP endpoints
- **Known Issues**: httpbin.org is intermittently unavailable (returns 503 errors)

## Test Runner Behavior

The test runner (`scripts/test_runner.py`) handles external_api tests intelligently:

1. **Separate Section**: External API tests run in their own "external_api" section
2. **Failure Handling**: If external_api tests fail, they're marked as "Flaky" (⚠️) instead of "Failed" (✗)
3. **No Build Failure**: Flaky external_api tests don't cause the overall build to fail
4. **Full Visibility**: You can still see which external tests failed and investigate if needed

This approach ensures:
- ✅ External API tests are always attempted
- ✅ Successful external tests are reported as passed
- ✅ Failed external tests don't block the build
- ✅ You get full visibility into external service health

### To run external_api tests explicitly:

```bash
# Run all external_api tests
uv run pytest -m external_api

# Run specific external_api test file
uv run pytest tests/integration/test_rest_api_tool_integration.py -m "integration and external_api"

# Run with retries for flaky tests
uv run pytest -m external_api --reruns 3 --reruns-delay 5
```

### To exclude external_api tests (default):

```bash
# Run all tests except external_api
uv run pytest -m "not external_api"

# Run integration tests except external_api
uv run pytest tests/integration/ -m "integration and not external_api"
```

## Test Configuration

- **pytest.ini**: Default config excludes `external_api` tests
- **scripts/test_runner.py**: Test runner excludes `external_api` tests by default
- **Skip Logic**: Tests check service availability at module import and skip if unavailable

## Race Condition Warning

Even with availability checks, there's a race condition:
1. Module import checks service availability (passes)
2. Tests start executing
3. Service becomes unavailable mid-run (503 errors)
4. Tests fail unexpectedly

This is inherent to external service dependencies and cannot be fully eliminated.

## Recommendations for CI/CD

1. **Separate Pipeline**: Run external_api tests in a separate, optional pipeline
2. **Retry Logic**: Use pytest-rerunfailures plugin with `--reruns 3`
3. **Mock Services**: Consider using mock servers (e.g., WireMock, httpretty) for stable tests
4. **Monitor Service**: Track external service uptime and skip tests when down
5. **Failure Tolerance**: Don't block deployments on external_api test failures

## Adding New External API Tests

When adding tests that depend on external services:

1. Mark the test module with `pytestmark = pytest.mark.external_api`
2. Add availability check at module import:
   ```python
   try:
       _response = httpx.get("https://service.com/health", timeout=5.0)
       if _response.status_code != 200:
           pytest.skip("service.com unavailable", allow_module_level=True)
   except Exception as e:
       pytest.skip(f"service.com unavailable: {e}", allow_module_level=True)
   ```
3. Document the external dependency in this README
4. Consider adding retry decorators for individual flaky tests

## Alternative: Mock External Services

For more reliable integration tests, consider:

```python
import httpx
import respx

@respx.mock
async def test_rest_api_mock():
    respx.get("https://httpbin.org/get").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    # Test code here
```

This provides deterministic test behavior without external dependencies.
