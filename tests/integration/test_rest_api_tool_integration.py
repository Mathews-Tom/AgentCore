"""Integration tests for REST API Tool (TOOL-012).

These tests use real HTTP endpoints (httpbin.org) to validate the REST API Tool
implementation with actual network calls. Tests cover:
- All HTTP methods (GET, POST, PUT, DELETE)
- Real authentication scenarios
- Response parsing with real APIs
- Network error handling
- Timeout scenarios

Note: These tests require internet connectivity to httpbin.org
"""

import pytest

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.api_tools import RESTAPITool


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create execution context for testing."""
    return ExecutionContext(
        user_id="integration-test-user",
        agent_id="integration-test-agent",
        trace_id="integration-trace-123",
        session_id="integration-session",
    )


@pytest.fixture
def rest_api_tool() -> RESTAPITool:
    """Create REST API Tool."""
    return RESTAPITool()


# HTTP Methods Integration Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_get_request(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test GET request with real API (httpbin.org)."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/get",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result
    assert result_data["success"] is True
    assert result_data["status_code"] == 200
    assert "application/json" in result_data["content_type"]

    # httpbin.org returns request details
    body = result_data["body"]
    assert "url" in body
    assert "https://httpbin.org/get" in body["url"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_post_request(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test POST request with JSON body (httpbin.org)."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/post",
            "method": "POST",
            "body": '{"test": "data", "number": 123}',
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200

    # httpbin.org echoes back the JSON we sent
    body = result_data["body"]
    assert "json" in body
    assert body["json"]["test"] == "data"
    assert body["json"]["number"] == 123


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_put_request(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test PUT request with JSON body (httpbin.org)."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/put",
            "method": "PUT",
            "body": '{"update": "successful"}',
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200

    body = result_data["body"]
    assert body["json"]["update"] == "successful"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_delete_request(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test DELETE request (httpbin.org)."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/delete",
            "method": "DELETE",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200


# Authentication Integration Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_bearer_auth(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test Bearer token authentication with real API."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/bearer",
            "method": "GET",
            "auth_type": "bearer_token",
            "auth_token": "test-bearer-token-12345",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200

    # httpbin.org returns authentication info
    body = result_data["body"]
    assert "authenticated" in body
    assert body["authenticated"] is True
    assert body["token"] == "test-bearer-token-12345"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_custom_headers(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test custom headers with real API."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/headers",
            "method": "GET",
            "headers": {
                "X-Custom-Header": "integration-test-value",
                "User-Agent": "AgentCore-Integration-Test/1.0",
            },
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200

    # httpbin.org returns all request headers
    body = result_data["body"]
    assert "headers" in body
    headers = body["headers"]
    assert headers["X-Custom-Header"] == "integration-test-value"
    assert headers["User-Agent"] == "AgentCore-Integration-Test/1.0"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_api_key_auth(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test API key authentication with custom header."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/headers",
            "method": "GET",
            "auth_type": "api_key",
            "auth_token": "test-api-key-xyz",
            "auth_header": "X-API-Key",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # Verify API key was sent in header
    body = result_data["body"]
    headers = body["headers"]
    assert headers["X-Api-Key"] == "test-api-key-xyz"  # httpbin lowercases header names


# Response Parsing Integration Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_json_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test JSON response parsing with real API."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/json",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "application/json" in result_data["content_type"]

    # httpbin.org/json returns sample JSON data
    body = result_data["body"]
    assert isinstance(body, dict)
    assert "slideshow" in body  # Known structure from httpbin


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_html_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test HTML/text response parsing with real API."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/html",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "text/html" in result_data["content_type"]

    # Should return HTML as text
    body = result_data["body"]
    assert isinstance(body, str)
    assert "<html>" in body.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_binary_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test binary response parsing (returns base64) with real API."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/image/png",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "image/png" in result_data["content_type"]

    # Binary content should be base64 encoded
    assert result_data.get("is_binary") is True
    body = result_data["body"]
    assert isinstance(body, str)  # base64 string

    # Verify it's valid base64
    import base64
    decoded = base64.b64decode(body)
    # PNG files start with specific magic bytes
    assert decoded[:4] == b'\x89PNG'


# Error Handling Integration Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_http_404_error(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of real 404 error."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/status/404",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "404" in result.error
    result_data = result.result
    assert result_data["status_code"] == 404
    assert result_data["success"] is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_http_500_error(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of real 500 error."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/status/500",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "500" in result.error
    result_data = result.result
    assert result_data["status_code"] == 500


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_http_401_unauthorized(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of real 401 Unauthorized error."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/status/401",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "401" in result.error
    result_data = result.result
    assert result_data["status_code"] == 401


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_timeout(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test timeout handling with real API."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/delay/10",  # Delays response by 10 seconds
            "method": "GET",
            "timeout": 1,  # But we timeout after 1 second
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert result.error_type == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_invalid_domain(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of network connection error (invalid domain)."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://this-domain-does-not-exist-12345.invalid",
            "method": "GET",
            "timeout": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert result.error_type == "HttpError"


# Query Parameters Integration Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_query_parameters(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test GET request with query parameters in URL."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/get?param1=value1&param2=value2",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # httpbin.org returns query parameters
    body = result_data["body"]
    assert "args" in body
    assert body["args"]["param1"] == "value1"
    assert body["args"]["param2"] == "value2"


# Content Type Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_utf8_content(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of UTF-8 encoded content."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/encoding/utf8",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # Should handle UTF-8 content properly
    body = result_data["body"]
    assert isinstance(body, str)
    # httpbin returns HTML with UTF-8 characters


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_gzip_compression(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of gzip-compressed responses."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/gzip",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    # httpx automatically handles gzip decompression
    body = result_data["body"]
    assert "gzipped" in body
    assert body["gzipped"] is True


# Redirect Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_redirect_following(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that HTTP redirects are followed automatically."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/absolute-redirect/1",
            "method": "GET",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200

    # httpx automatically follows redirects, should end up at /get
    body = result_data["body"]
    assert "url" in body


# Performance Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_execution_time_tracking(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that execution time is tracked correctly."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/delay/1",  # 1 second delay
            "method": "GET",
            "timeout": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    # Execution time should be at least 1000ms (1 second delay)
    assert result.execution_time_ms >= 1000
    # But not too much more (allowing for network latency)
    assert result.execution_time_ms < 5000


# Metadata Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_metadata_populated(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that result metadata is correctly populated."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://httpbin.org/get",
            "method": "GET",
            "auth_type": "bearer_token",
            "auth_token": "test-token",
        },
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "rest_api"
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
    assert result.metadata["auth_type"] == "bearer_token"
