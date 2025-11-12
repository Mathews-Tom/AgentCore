"""Unit and integration tests for Google Search Tool (TOOL-009).

Tests cover:
- Parameter validation
- Mock results fallback (no API key)
- Real API integration with mocked responses
- Error handling (HTTP errors, timeouts)
- Result parsing and formatting
"""

import pytest
import respx
from httpx import Response

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.search_tools import GoogleSearchTool


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create execution context for testing."""
    return ExecutionContext(
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace-123",
        session_id="test-session",
    )


@pytest.fixture
def google_tool_no_api() -> GoogleSearchTool:
    """Create Google Search Tool without API credentials (uses mocks)."""
    return GoogleSearchTool(api_key="", cse_id="")


@pytest.fixture
def google_tool_with_api() -> GoogleSearchTool:
    """Create Google Search Tool with API credentials."""
    return GoogleSearchTool(api_key="test-api-key", cse_id="test-cse-id")


# Parameter Validation Tests


@pytest.mark.asyncio
async def test_google_search_missing_query(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that missing required query parameter fails validation."""
    result = await google_tool_no_api.execute(
        parameters={},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "query" in result.error.lower()
    assert result.error_type == "ValidationError"


@pytest.mark.asyncio
async def test_google_search_empty_query(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that empty query string fails validation."""
    result = await google_tool_no_api.execute(
        parameters={"query": ""},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_google_search_invalid_num_results(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that invalid num_results fails validation."""
    result = await google_tool_no_api.execute(
        parameters={"query": "test", "num_results": 15},  # max is 10
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "num_results" in result.error.lower()


@pytest.mark.asyncio
async def test_google_search_query_too_long(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that query exceeding max length fails validation."""
    long_query = "x" * 501  # max is 500
    result = await google_tool_no_api.execute(
        parameters={"query": long_query},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


# Mock Results Tests (No API Key)


@pytest.mark.asyncio
async def test_google_search_mock_success(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test successful search with mock results when no API key configured."""
    result = await google_tool_no_api.execute(
        parameters={"query": "Python programming", "num_results": 5},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.result is not None

    result_data = result.result
    assert result_data["query"] == "Python programming"
    assert result_data["total_results"] == 5
    assert len(result_data["results"]) == 5
    assert result_data["provider"] == "Google Custom Search API (Mock)"
    assert result.metadata["using_mock"] is True

    # Verify result structure
    first_result = result_data["results"][0]
    assert "title" in first_result
    assert "url" in first_result
    assert "snippet" in first_result
    assert "Python programming" in first_result["title"]


@pytest.mark.asyncio
async def test_google_search_mock_default_num_results(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that default num_results (10) is used when not specified."""
    result = await google_tool_no_api.execute(
        parameters={"query": "test query"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["total_results"] == 10


# Real API Integration Tests (Mocked HTTP)


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_success(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test successful search with real API (mocked HTTP response)."""
    # Mock Google Custom Search API response
    mock_api_response = {
        "items": [
            {
                "title": "Python Official Documentation",
                "link": "https://docs.python.org",
                "snippet": "Official Python documentation and tutorials",
            },
            {
                "title": "Python on Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "snippet": "Python is a high-level programming language",
            },
            {
                "title": "Real Python",
                "link": "https://realpython.com",
                "snippet": "Python tutorials and resources",
            },
        ]
    }

    respx.get("https://www.googleapis.com/customsearch/v1").mock(
        return_value=Response(200, json=mock_api_response)
    )

    result = await google_tool_with_api.execute(
        parameters={"query": "Python programming", "num_results": 3},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.result is not None

    result_data = result.result
    assert result_data["query"] == "Python programming"
    assert result_data["total_results"] == 3
    assert len(result_data["results"]) == 3
    assert result_data["provider"] == "Google Custom Search API"
    assert result.metadata["using_mock"] is False

    # Verify result structure matches API response
    first_result = result_data["results"][0]
    assert first_result["title"] == "Python Official Documentation"
    assert first_result["url"] == "https://docs.python.org"
    assert first_result["snippet"] == "Official Python documentation and tutorials"


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_empty_results(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test search with no results from API."""
    mock_api_response = {"items": []}

    respx.get("https://www.googleapis.com/customsearch/v1").mock(
        return_value=Response(200, json=mock_api_response)
    )

    result = await google_tool_with_api.execute(
        parameters={"query": "xyzabc123nonexistent"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["total_results"] == 0
    assert len(result_data["results"]) == 0


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_http_error_401(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of HTTP 401 Unauthorized error."""
    respx.get("https://www.googleapis.com/customsearch/v1").mock(
        return_value=Response(401, text="Unauthorized: Invalid API key")
    )

    result = await google_tool_with_api.execute(
        parameters={"query": "test query"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "401" in result.error
    assert result.error_type == "HttpError"


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_http_error_429(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of HTTP 429 Rate Limit Exceeded error."""
    respx.get("https://www.googleapis.com/customsearch/v1").mock(
        return_value=Response(429, text="Rate limit exceeded")
    )

    result = await google_tool_with_api.execute(
        parameters={"query": "test query"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "429" in result.error
    assert result.error_type == "HttpError"


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_timeout(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of API timeout."""
    import httpx

    respx.get("https://www.googleapis.com/customsearch/v1").mock(
        side_effect=httpx.TimeoutException("Request timeout")
    )

    result = await google_tool_with_api.execute(
        parameters={"query": "test query"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.error is not None
    assert "timeout" in result.error.lower()
    assert result.error_type == "TimeoutError"


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_malformed_response(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of malformed API response."""
    # Response with missing expected fields
    mock_api_response = {"unexpected_field": "value"}

    respx.get("https://www.googleapis.com/customsearch/v1").mock(
        return_value=Response(200, json=mock_api_response)
    )

    result = await google_tool_with_api.execute(
        parameters={"query": "test query"},
        context=execution_context,
    )

    # Should succeed but return empty results (graceful handling)
    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["total_results"] == 0


# API Parameter Tests


@pytest.mark.asyncio
@respx.mock
async def test_google_search_api_parameters_sent_correctly(
    google_tool_with_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that correct parameters are sent to Google API."""
    mock_api_response = {"items": []}

    route = respx.get("https://www.googleapis.com/customsearch/v1").mock(
        return_value=Response(200, json=mock_api_response)
    )

    await google_tool_with_api.execute(
        parameters={"query": "test query", "num_results": 7},
        context=execution_context,
    )

    # Verify the request was made with correct parameters
    assert route.called
    request = route.calls[0].request
    params = dict(request.url.params)

    assert params["key"] == "test-api-key"
    assert params["cx"] == "test-cse-id"
    assert params["q"] == "test query"
    assert params["num"] == "7"


# Tool Metadata Tests


def test_google_search_tool_metadata(google_tool_with_api: GoogleSearchTool):
    """Test that tool metadata is correctly configured."""
    metadata = google_tool_with_api.metadata

    assert metadata.tool_id == "google_search"
    assert metadata.name == "Google Search"
    assert metadata.version == "1.0.0"
    assert metadata.is_retryable is True
    assert metadata.max_retries == 3
    assert metadata.timeout_seconds == 30
    assert metadata.is_idempotent is True

    # Verify parameters
    assert "query" in metadata.parameters
    assert "num_results" in metadata.parameters
    assert metadata.parameters["query"].required is True
    assert metadata.parameters["num_results"].required is False
    assert metadata.parameters["num_results"].default == 10

    # Verify rate limits
    assert metadata.rate_limits is not None
    assert metadata.rate_limits["calls_per_minute"] == 100


# Edge Cases


@pytest.mark.asyncio
async def test_google_search_special_characters_in_query(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test search with special characters in query."""
    result = await google_tool_no_api.execute(
        parameters={"query": "Python 3.12+ features: @decorators & *args"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "Python 3.12+ features: @decorators & *args" in result_data["query"]


@pytest.mark.asyncio
async def test_google_search_unicode_query(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test search with Unicode characters."""
    result = await google_tool_no_api.execute(
        parameters={"query": "Python プログラミング 编程"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "Python プログラミング 编程" in result_data["query"]


@pytest.mark.asyncio
async def test_google_search_execution_metadata(
    google_tool_no_api: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that execution metadata is correctly populated."""
    result = await google_tool_no_api.execute(
        parameters={"query": "test"},
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "google_search"
    assert result.execution_time_ms > 0
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
