"""Integration tests for GoogleSearchTool (TOOL-015).

These tests use real Google Custom Search API to validate the Google Search Tool
implementation with actual network calls. Tests cover:
- Real Google API integration
- Result parsing and formatting
- Error handling (API errors, timeouts)
- Mock fallback behavior when API not configured

Note: These tests require GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.
If not set, tests will validate mock fallback behavior.
"""

import os

import pytest

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.search_tools import GoogleSearchTool


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create execution context for testing."""
    return ExecutionContext(
        user_id="integration-test-user",
        agent_id="integration-test-agent",
        trace_id="integration-trace-google-123",
        session_id="integration-session-google",
    )


@pytest.fixture
def google_search_tool() -> GoogleSearchTool:
    """Create Google Search Tool."""
    return GoogleSearchTool()


@pytest.fixture
def has_google_credentials() -> bool:
    """Check if Google API credentials are available."""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    return bool(api_key and cse_id)


# Real API Integration Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_real_api(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
    has_google_credentials: bool,
):
    """Test Google search with real API (if credentials available)."""
    if not has_google_credentials:
        pytest.skip("Google API credentials not configured (GOOGLE_API_KEY, GOOGLE_CSE_ID)")

    result = await google_search_tool.execute(
        parameters={
            "query": "Python programming language",
            "num_results": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result

    # Validate result structure
    assert "query" in result_data
    assert result_data["query"] == "Python programming language"
    assert "total_results" in result_data
    assert "results" in result_data
    assert "provider" in result_data

    # Should get results from real API
    assert result_data["provider"] == "Google Custom Search API"
    assert result_data["total_results"] > 0
    assert len(result_data["results"]) > 0

    # Validate result format
    for search_result in result_data["results"]:
        assert "title" in search_result
        assert "url" in search_result
        assert "snippet" in search_result
        assert isinstance(search_result["title"], str)
        assert isinstance(search_result["url"], str)
        assert isinstance(search_result["snippet"], str)
        assert search_result["url"].startswith("http")

    # Validate metadata
    assert result.metadata["using_mock"] is False
    assert result.metadata["trace_id"] == execution_context.trace_id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_num_results(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
    has_google_credentials: bool,
):
    """Test Google search with custom num_results parameter."""
    if not has_google_credentials:
        pytest.skip("Google API credentials not configured")

    result = await google_search_tool.execute(
        parameters={
            "query": "machine learning",
            "num_results": 3,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # Should return requested number of results (or fewer if not available)
    assert len(result_data["results"]) <= 3
    assert result_data["total_results"] <= 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_complex_query(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
    has_google_credentials: bool,
):
    """Test Google search with complex query (operators, quotes)."""
    if not has_google_credentials:
        pytest.skip("Google API credentials not configured")

    result = await google_search_tool.execute(
        parameters={
            "query": '"artificial intelligence" AND "neural networks"',
            "num_results": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # Should handle complex query
    assert result_data["total_results"] > 0


# Mock Fallback Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_mock_fallback(
    execution_context: ExecutionContext,
):
    """Test Google search falls back to mock when API not configured."""
    # Create tool without credentials
    google_search_tool = GoogleSearchTool(api_key="", cse_id="")

    result = await google_search_tool.execute(
        parameters={
            "query": "test query",
            "num_results": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # Should use mock provider
    assert result_data["provider"] == "Google Custom Search API (Mock)"
    assert result.metadata["using_mock"] is True

    # Mock results should have correct structure
    assert result_data["total_results"] == 5
    assert len(result_data["results"]) == 5

    for i, search_result in enumerate(result_data["results"]):
        assert search_result["title"] == f"Result {i+1} for test query"
        assert search_result["url"] == f"https://example.com/result{i+1}"
        assert "test query" in search_result["snippet"]


# Error Handling Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_invalid_parameters(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test Google search with invalid parameters."""
    # Empty query
    result = await google_search_tool.execute(
        parameters={
            "query": "",
            "num_results": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert result.error_type == "ValidationError"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_num_results_too_large(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test Google search with num_results exceeding maximum."""
    result = await google_search_tool.execute(
        parameters={
            "query": "test",
            "num_results": 100,  # Exceeds max of 10
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "num_results" in result.error


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_invalid_credentials(
    execution_context: ExecutionContext,
):
    """Test Google search with invalid API credentials."""
    # Create tool with invalid credentials
    google_search_tool = GoogleSearchTool(api_key="invalid-key", cse_id="invalid-cse-id")

    result = await google_search_tool.execute(
        parameters={
            "query": "test query",
            "num_results": 5,
        },
        context=execution_context,
    )

    # Should fail with HTTP error (401 or 400)
    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert result.error_type == "HttpError"


# Performance and Metadata Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_execution_time(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
    has_google_credentials: bool,
):
    """Test that execution time is tracked correctly."""
    result = await google_search_tool.execute(
        parameters={
            "query": "AgentCore",
            "num_results": 3,
        },
        context=execution_context,
    )

    # Should track execution time
    assert result.execution_time_ms > 0
    # Real API call should take some time (unless using mock)
    if has_google_credentials and not result.metadata.get("using_mock"):
        assert result.execution_time_ms > 10  # At least 10ms for network call


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_metadata_populated(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
):
    """Test that result metadata is correctly populated."""
    result = await google_search_tool.execute(
        parameters={
            "query": "test",
            "num_results": 5,
        },
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "google_search"
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
    assert "using_mock" in result.metadata


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_special_characters(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
    has_google_credentials: bool,
):
    """Test Google search with special characters in query."""
    if not has_google_credentials:
        pytest.skip("Google API credentials not configured")

    result = await google_search_tool.execute(
        parameters={
            "query": "C++ programming & data structures",
            "num_results": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["total_results"] > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_google_search_unicode(
    google_search_tool: GoogleSearchTool,
    execution_context: ExecutionContext,
    has_google_credentials: bool,
):
    """Test Google search with unicode characters."""
    if not has_google_credentials:
        pytest.skip("Google API credentials not configured")

    result = await google_search_tool.execute(
        parameters={
            "query": "Python プログラミング",  # Japanese characters
            "num_results": 3,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    # Should handle unicode gracefully
    assert "query" in result_data
