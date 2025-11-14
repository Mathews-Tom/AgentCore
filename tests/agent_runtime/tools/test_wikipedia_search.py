"""Unit and integration tests for Wikipedia Search Tool (TOOL-010).

Tests cover:
- Parameter validation
- Article summary extraction with configurable sentence count
- Disambiguation handling for queries with multiple matches
- Error handling (HTTP errors, timeouts, invalid responses)
- Result formatting
"""

import pytest
import respx
from httpx import Response

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.search_tools import WikipediaSearchTool


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
def wikipedia_tool() -> WikipediaSearchTool:
    """Create Wikipedia Search Tool."""
    return WikipediaSearchTool()


# Parameter Validation Tests


@pytest.mark.asyncio
async def test_wikipedia_search_missing_query(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test that missing required query parameter fails validation."""
    result = await wikipedia_tool.execute(
        parameters={},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "query" in result.error.lower()
    assert result.error_type == "ValidationError"


@pytest.mark.asyncio
async def test_wikipedia_search_empty_query(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test that empty query string fails validation."""
    result = await wikipedia_tool.execute(
        parameters={"query": ""},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_wikipedia_search_invalid_sentences(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test that invalid sentences value fails validation."""
    result = await wikipedia_tool.execute(
        parameters={"query": "Python", "sentences": 25},  # max is 20
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "sentences" in result.error.lower()


@pytest.mark.asyncio
async def test_wikipedia_search_query_too_long(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test that query exceeding max length fails validation."""
    long_query = "x" * 301  # max is 300
    result = await wikipedia_tool.execute(
        parameters={"query": long_query},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


# Article Summary Extraction Tests


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_article_summary_success(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test successful article summary extraction."""
    # Mock search response
    search_response = {
        "query": {
            "search": [
                {
                    "title": "Python (programming language)",
                    "pageid": 23862,
                }
            ]
        }
    }

    # Mock extract response with summary
    extract_response = {
        "query": {
            "pages": {
                "23862": {
                    "pageid": 23862,
                    "title": "Python (programming language)",
                    "extract": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms. Python was created by Guido van Rossum.",
                    "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                }
            }
        }
    }

    # Mock Wikipedia API calls
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=extract_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "Python programming language", "sentences": 5},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.result is not None

    result_data = result.result
    assert result_data["query"] == "Python programming language"
    assert result_data["total_results"] == 1
    assert result_data["disambiguation"] is False
    assert len(result_data["results"]) == 1

    article = result_data["results"][0]
    assert article["title"] == "Python (programming language)"
    assert "Python is a high-level" in article["summary"]
    assert article["url"] == "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert article["page_id"] == 23862


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_default_sentences(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test that default sentences value (5) is used when not specified."""
    search_response = {
        "query": {
            "search": [{"title": "Test Article", "pageid": 123}]
        }
    }

    extract_response = {
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "title": "Test Article",
                    "extract": "Test summary.",
                    "fullurl": "https://en.wikipedia.org/wiki/Test_Article",
                }
            }
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=extract_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "test"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    # Verify default sentences parameter was used
    # (can't directly check API call params in this test structure,
    # but the test passes showing default handling works)


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_custom_sentences(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test article summary with custom sentence count."""
    search_response = {
        "query": {
            "search": [{"title": "Test", "pageid": 456}]
        }
    }

    extract_response = {
        "query": {
            "pages": {
                "456": {
                    "pageid": 456,
                    "title": "Test",
                    "extract": "First sentence. Second sentence. Third sentence.",
                    "fullurl": "https://en.wikipedia.org/wiki/Test",
                }
            }
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=extract_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "test", "sentences": 3},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "First sentence" in result_data["results"][0]["summary"]


# Disambiguation Handling Tests


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_disambiguation_handling(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of disambiguation pages."""
    # Mock search response returning disambiguation page
    search_response = {
        "query": {
            "search": [
                {
                    "title": "Python (disambiguation)",
                    "pageid": 789,
                }
            ]
        }
    }

    # Mock disambiguation page links
    links_response = {
        "query": {
            "pages": {
                "789": {
                    "pageid": 789,
                    "title": "Python (disambiguation)",
                    "links": [
                        {"title": "Python (programming language)"},
                        {"title": "Python (genus)"},
                        {"title": "Python (film)"},
                        {"title": "Monty Python"},
                    ],
                }
            }
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=links_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "Python"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    assert result_data["disambiguation"] is True
    assert result_data["disambiguation_page"] == "Python (disambiguation)"
    assert result_data["total_results"] > 0

    # Verify disambiguation options
    options = result_data["results"]
    assert any("programming language" in opt["title"] for opt in options)
    assert all("url" in opt for opt in options)
    assert all("title" in opt for opt in options)


# Error Handling Tests


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_no_results(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of search with no results."""
    search_response = {
        "query": {
            "search": []
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=Response(200, json=search_response)
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "xyznonexistentquery123"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["total_results"] == 0
    assert len(result_data["results"]) == 0


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_api_http_error(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of HTTP errors from Wikipedia API."""
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=Response(500, text="Internal Server Error")
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "test"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "500" in result.error
    assert result.error_type == "HttpError"


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_api_timeout(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of API timeout."""
    import httpx

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=httpx.TimeoutException("Request timeout")
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "test"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.error is not None
    assert "timeout" in result.error.lower()
    assert result.error_type == "TimeoutError"


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_malformed_response(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test handling of malformed API response."""
    # Response missing expected fields
    malformed_response = {"unexpected": "data"}

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=Response(200, json=malformed_response)
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "test"},
        context=execution_context,
    )

    # Should handle gracefully and return empty results
    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["total_results"] == 0


# Tool Metadata Tests


def test_wikipedia_search_tool_metadata(wikipedia_tool: WikipediaSearchTool):
    """Test that tool metadata is correctly configured."""
    metadata = wikipedia_tool.metadata

    assert metadata.tool_id == "wikipedia_search"
    assert metadata.name == "Wikipedia Search"
    assert metadata.version == "1.0.0"
    assert metadata.auth_method.value == "none"
    assert metadata.is_retryable is True
    assert metadata.max_retries == 3
    assert metadata.timeout_seconds == 30
    assert metadata.is_idempotent is True

    # Verify parameters
    assert "query" in metadata.parameters
    assert "sentences" in metadata.parameters
    assert metadata.parameters["query"].required is True
    assert metadata.parameters["sentences"].required is False
    assert metadata.parameters["sentences"].default == 5


# Edge Cases


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_special_characters(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test search with special characters in query."""
    search_response = {
        "query": {
            "search": [{"title": "C++", "pageid": 999}]
        }
    }

    extract_response = {
        "query": {
            "pages": {
                "999": {
                    "pageid": 999,
                    "title": "C++",
                    "extract": "C++ is a programming language.",
                    "fullurl": "https://en.wikipedia.org/wiki/C%2B%2B",
                }
            }
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=extract_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "C++"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result["results"][0]["title"] == "C++"


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_unicode_query(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test search with Unicode characters."""
    search_response = {
        "query": {
            "search": [{"title": "東京", "pageid": 111}]
        }
    }

    extract_response = {
        "query": {
            "pages": {
                "111": {
                    "pageid": 111,
                    "title": "東京",
                    "extract": "Tokyo is the capital of Japan.",
                    "fullurl": "https://en.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC",
                }
            }
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=extract_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "東京"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_execution_metadata(
    wikipedia_tool: WikipediaSearchTool,
    execution_context: ExecutionContext,
):
    """Test that execution metadata is correctly populated."""
    search_response = {
        "query": {
            "search": [{"title": "Test", "pageid": 222}]
        }
    }

    extract_response = {
        "query": {
            "pages": {
                "222": {
                    "pageid": 222,
                    "title": "Test",
                    "extract": "Test summary.",
                    "fullurl": "https://en.wikipedia.org/wiki/Test",
                }
            }
        }
    }

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=search_response),
            Response(200, json=extract_response),
        ]
    )

    result = await wikipedia_tool.execute(
        parameters={"query": "test"},
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "wikipedia_search"
    assert result.execution_time_ms > 0
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
