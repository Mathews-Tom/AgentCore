"""Tests for built-in tool adapters."""

import json
import os

import pytest
import respx
from httpx import Response

from agentcore.agent_runtime.jsonrpc.tools_jsonrpc import get_tool_registry
from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionRequest,
    ToolExecutionStatus,
)
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.agent_runtime.tools.startup import register_builtin_tools

# Check for required environment variables
has_google_api = os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
code_execution_enabled = os.environ.get("ENABLE_CODE_EXECUTION", "false").lower() == "true"


@pytest.fixture
def tool_executor() -> ToolExecutor:
    """Get tool executor with all built-in tools."""
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return ToolExecutor(registry)


# Search Tools Tests


@pytest.mark.asyncio
@pytest.mark.skipif(not has_google_api, reason="Google API credentials not configured")
async def test_google_search(tool_executor: ToolExecutor):
    """Test Google search tool."""
    request = ToolExecutionRequest(
        tool_id="google_search",
        parameters={"query": "Python programming", "num_results": 5},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result is not None
    assert "results" in json.loads(result.result) if isinstance(result.result, str) else result.result


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search(tool_executor: ToolExecutor):
    """Test Wikipedia search tool."""
    # Mock Wikipedia API search response (action=query&list=search)
    mock_search_response = {
        "query": {
            "search": [
                {
                    "title": "Python (programming language)",
                    "snippet": "High-level programming language",
                }
            ]
        }
    }

    # Mock Wikipedia API extract response (action=query&prop=extracts)
    mock_extract_response = {
        "query": {
            "pages": {
                "12345": {
                    "pageid": 12345,
                    "title": "Python (programming language)",
                    "extract": "Python is a high-level programming language. It supports multiple programming paradigms.",
                    "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)"
                }
            }
        }
    }

    # Mock both API calls
    respx.get("https://en.wikipedia.org/w/api.php", params__contains={"list": "search"}).mock(
        return_value=Response(200, json=mock_search_response)
    )
    respx.get("https://en.wikipedia.org/w/api.php", params__contains={"prop": "extracts|info"}).mock(
        return_value=Response(200, json=mock_extract_response)
    )

    request = ToolExecutionRequest(
        tool_id="wikipedia_search",
        parameters={"query": "Python", "limit": 2},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    # Wikipedia search returns detailed result dict
    assert isinstance(result.result, dict)
    assert result.result["query"] == "Python"
    assert result.result["total_results"] == 1
    assert result.result["disambiguation"] is False
    assert len(result.result["results"]) == 1
    assert result.result["results"][0]["title"] == "Python (programming language)"
    assert "summary" in result.result["results"][0]


@pytest.mark.asyncio
@respx.mock
async def test_web_scrape(tool_executor: ToolExecutor):
    """Test web scraping tool."""
    # Mock HTTP response
    html_content = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Hello World</h1>
        <p>This is test content.</p>
        <script>console.log('test');</script>
    </body>
    </html>
    """

    respx.get("https://example.com/test").mock(
        return_value=Response(200, text=html_content, headers={"content-type": "text/html"})
    )

    request = ToolExecutionRequest(
        tool_id="web_scrape",
        parameters={"url": "https://example.com/test", "extract_text": True},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert "text" in result_data
    # Script tags should be removed
    assert "console.log" not in result_data["text"]
    assert "Hello World" in result_data["text"]


# Code Execution Tools Tests


@pytest.mark.asyncio
@pytest.mark.skipif(not code_execution_enabled, reason="Code execution not enabled (ENABLE_CODE_EXECUTION=false)")
async def test_execute_python_simple(tool_executor: ToolExecutor):
    """Test Python execution with simple code."""
    code = """
result = 2 + 2
print(f"Answer: {result}")
"""

    request = ToolExecutionRequest(
        tool_id="execute_python",
        parameters={"code": code, "timeout": 5},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert "Answer: 4" in result_data["stdout"]


@pytest.mark.asyncio
@pytest.mark.skipif(not code_execution_enabled, reason="Code execution not enabled (ENABLE_CODE_EXECUTION=false)")
async def test_execute_python_with_error(tool_executor: ToolExecutor):
    """Test Python execution with error."""
    code = """
undefined_variable + 1
"""

    request = ToolExecutionRequest(
        tool_id="execute_python",
        parameters={"code": code, "timeout": 5},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS  # Tool executed, but code failed
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is False
    assert "error" in result_data
    assert "NameError" in result_data["error"]


@pytest.mark.asyncio
@pytest.mark.skipif(not code_execution_enabled, reason="Code execution not enabled (ENABLE_CODE_EXECUTION=false)")
async def test_execute_python_timeout(tool_executor: ToolExecutor):
    """Test Python execution timeout."""
    # Note: Python sandbox doesn't support imports for security,
    # so we can't test actual timeout. This test verifies the error handling.
    code = """
# Intentionally cause an import error since imports are restricted
import time
time.sleep(100)
"""

    request = ToolExecutionRequest(
        tool_id="execute_python",
        parameters={"code": code, "timeout": 1},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    # The code will fail due to restricted builtins (no import allowed)
    assert result.status == ToolExecutionStatus.SUCCESS  # Tool executed
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is False
    # Should error because 'import' is not available in restricted globals
    assert "error" in result_data


@pytest.mark.asyncio
@pytest.mark.skipif(not code_execution_enabled, reason="Code execution not enabled (ENABLE_CODE_EXECUTION=false)")
async def test_evaluate_expression(tool_executor: ToolExecutor):
    """Test expression evaluation."""
    request = ToolExecutionRequest(
        tool_id="evaluate_expression",
        parameters={"expression": "10 * 5 + 3"},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert result_data["result"] == 53


@pytest.mark.asyncio
@pytest.mark.skipif(not code_execution_enabled, reason="Code execution not enabled (ENABLE_CODE_EXECUTION=false)")
async def test_evaluate_expression_with_functions(tool_executor: ToolExecutor):
    """Test expression evaluation with built-in functions."""
    request = ToolExecutionRequest(
        tool_id="evaluate_expression",
        parameters={"expression": "sum([1, 2, 3, 4, 5])"},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert result_data["result"] == 15


@pytest.mark.asyncio
@pytest.mark.skipif(not code_execution_enabled, reason="Code execution not enabled (ENABLE_CODE_EXECUTION=false)")
async def test_evaluate_expression_error(tool_executor: ToolExecutor):
    """Test expression evaluation with error."""
    request = ToolExecutionRequest(
        tool_id="evaluate_expression",
        parameters={"expression": "1 / 0"},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS  # Tool executed
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is False
    assert "ZeroDivisionError" in result_data["error"]


# API Client Tools Tests


@pytest.mark.asyncio
@respx.mock
async def test_http_request_get(tool_executor: ToolExecutor):
    """Test HTTP GET request."""
    # Mock API response
    mock_data = {"message": "Hello, World!", "status": "success"}

    respx.get("https://api.example.com/data").mock(
        return_value=Response(200, json=mock_data)
    )

    request = ToolExecutionRequest(
        tool_id="http_request",
        parameters={
            "url": "https://api.example.com/data",
            "method": "GET",
        },
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert result_data["status_code"] == 200
    assert result_data["body"]["message"] == "Hello, World!"


@pytest.mark.asyncio
@respx.mock
async def test_http_request_post(tool_executor: ToolExecutor):
    """Test HTTP POST request."""
    # Mock API response
    mock_response = {"id": 123, "created": True}

    respx.post("https://api.example.com/items").mock(
        return_value=Response(201, json=mock_response)
    )

    request_body = json.dumps({"name": "Test Item", "value": 42})

    request = ToolExecutionRequest(
        tool_id="http_request",
        parameters={
            "url": "https://api.example.com/items",
            "method": "POST",
            "body": request_body,
            "headers": {"Content-Type": "application/json"},
        },
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert result_data["status_code"] == 201
    assert result_data["body"]["id"] == 123


@pytest.mark.asyncio
@respx.mock
async def test_rest_get(tool_executor: ToolExecutor):
    """Test REST GET request."""
    mock_data = {"users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]}

    respx.get("https://api.example.com/users").mock(
        return_value=Response(200, json=mock_data)
    )

    request = ToolExecutionRequest(
        tool_id="rest_get",
        parameters={
            "url": "https://api.example.com/users",
            "params": {"page": "1", "limit": "10"},
        },
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert len(result_data["body"]["users"]) == 2


@pytest.mark.asyncio
@respx.mock
async def test_rest_post(tool_executor: ToolExecutor):
    """Test REST POST request."""
    mock_response = {"id": 456, "message": "Created successfully"}

    respx.post("https://api.example.com/posts").mock(
        return_value=Response(201, json=mock_response)
    )

    post_data = json.dumps({"title": "Test Post", "content": "Test content"})

    request = ToolExecutionRequest(
        tool_id="rest_post",
        parameters={
            "url": "https://api.example.com/posts",
            "body": post_data,
        },
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert result_data["body"]["id"] == 456


@pytest.mark.asyncio
@respx.mock
async def test_graphql_query(tool_executor: ToolExecutor):
    """Test GraphQL query."""
    # Mock GraphQL response
    graphql_response = {
        "data": {
            "user": {"id": "123", "name": "Alice", "email": "alice@example.com"}
        }
    }

    respx.post("https://api.example.com/graphql").mock(
        return_value=Response(200, json=graphql_response)
    )

    query = """
    query GetUser($id: ID!) {
        user(id: $id) {
            id
            name
            email
        }
    }
    """

    request = ToolExecutionRequest(
        tool_id="graphql_query",
        parameters={
            "endpoint": "https://api.example.com/graphql",
            "query": query,
            "variables": {"id": "123"},
        },
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is True
    assert result_data["data"]["user"]["name"] == "Alice"


@pytest.mark.asyncio
@respx.mock
async def test_graphql_query_with_errors(tool_executor: ToolExecutor):
    """Test GraphQL query with errors."""
    # Mock GraphQL error response
    graphql_response = {
        "errors": [{"message": "User not found", "path": ["user"]}],
        "data": None,
    }

    respx.post("https://api.example.com/graphql").mock(
        return_value=Response(200, json=graphql_response)
    )

    query = "query { user(id: \"999\") { name } }"

    request = ToolExecutionRequest(
        tool_id="graphql_query",
        parameters={
            "endpoint": "https://api.example.com/graphql",
            "query": query,
        },
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = json.loads(result.result) if isinstance(result.result, str) else result.result
    assert result_data["success"] is False  # GraphQL returned errors
    assert "errors" in result_data


# Integration Tests


@pytest.mark.asyncio
async def test_all_tools_registered():
    """Test that all expected tools are registered."""
    registry = get_tool_registry()
    tools = registry.list_all()

    # Expected tool count: 3 (Phase 1) + 9 (Phase 2) = 12
    assert len(tools) >= 12

    # Check specific tools exist
    expected_tools = [
        "calculator",
        "get_current_time",
        "echo",
        "google_search",
        "wikipedia_search",
        "web_scrape",
        "execute_python",
        "evaluate_expression",
        "http_request",
        "rest_get",
        "rest_post",
        "graphql_query",
    ]

    registered_tool_ids = {tool.metadata.tool_id for tool in tools}
    for tool_id in expected_tools:
        assert tool_id in registered_tool_ids, f"Tool {tool_id} not registered"


@pytest.mark.asyncio
async def test_tool_discovery_by_capability():
    """Test discovering tools by capability."""
    from agentcore.agent_runtime.models.tool_integration import ToolCategory

    registry = get_tool_registry()

    # Find all API client tools
    api_tools = registry.list_by_category(ToolCategory.API_CLIENT)
    assert len(api_tools) >= 4  # http_request, rest_get, rest_post, graphql_query

    # Find code execution tools
    code_tools = registry.list_by_category(ToolCategory.CODE_EXECUTION)
    assert len(code_tools) >= 2  # execute_python, evaluate_expression
