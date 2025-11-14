"""Unit and integration tests for REST API Tool (TOOL-012).

Tests cover:
- Parameter validation (url, method, auth_type, auth_token)
- HTTP methods: GET, POST, PUT, DELETE
- Authentication: none, api_key, bearer_token, oauth
- Response parsing: JSON, text, binary (base64)
- Error handling: HTTP errors, timeouts, network errors
- Custom authentication headers
"""

import base64

import pytest
import respx
from httpx import Response

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.api_tools import RESTAPITool


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
def rest_api_tool() -> RESTAPITool:
    """Create REST API Tool."""
    return RESTAPITool()


# Parameter Validation Tests


@pytest.mark.asyncio
async def test_rest_api_missing_url(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that missing required url parameter fails validation."""
    result = await rest_api_tool.execute(
        parameters={},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "url" in result.error.lower()
    assert result.error_type == "ValidationError"


@pytest.mark.asyncio
async def test_rest_api_invalid_method(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that invalid HTTP method fails validation."""
    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/test", "method": "INVALID"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_rest_api_invalid_auth_type(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that invalid auth_type fails validation."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/test",
            "auth_type": "invalid_auth",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_rest_api_missing_auth_token(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that auth_token is required when auth_type is not 'none'."""
    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/test",
            "auth_type": "api_key",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "auth_token" in result.error.lower()
    assert result.error_type == "ValidationError"


# HTTP Methods Tests


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_get_success(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test successful GET request."""
    mock_response = {"data": "test data", "status": "success"}

    respx.get("https://api.example.com/test").mock(
        return_value=Response(200, json=mock_response)
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/test", "method": "GET"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.result is not None

    result_data = result.result
    assert result_data["success"] is True
    assert result_data["status_code"] == 200
    assert result_data["body"] == mock_response
    assert "application/json" in result_data["content_type"]


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_post_success(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test successful POST request with JSON body."""
    request_body = {"name": "test", "value": 123}
    response_data = {"id": 1, "created": True}

    respx.post("https://api.example.com/create").mock(
        return_value=Response(201, json=response_data)
    )

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/create",
            "method": "POST",
            "body": '{"name": "test", "value": 123}',
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 201
    assert result_data["body"] == response_data


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_put_success(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test successful PUT request."""
    response_data = {"id": 1, "updated": True}

    respx.put("https://api.example.com/update/1").mock(
        return_value=Response(200, json=response_data)
    )

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/update/1",
            "method": "PUT",
            "body": '{"name": "updated"}',
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200
    assert result_data["body"]["updated"] is True


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_delete_success(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test successful DELETE request."""
    response_data = {"deleted": True}

    respx.delete("https://api.example.com/delete/1").mock(
        return_value=Response(200, json=response_data)
    )

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/delete/1",
            "method": "DELETE",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 200
    assert result_data["body"]["deleted"] is True


# Authentication Tests


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_no_auth(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test request with no authentication."""
    respx.get("https://api.example.com/public").mock(
        return_value=Response(200, json={"public": True})
    )

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/public",
            "auth_type": "none",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.metadata["auth_type"] == "none"


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_api_key_auth(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test API key authentication."""
    def check_api_key(request):
        assert request.headers.get("X-API-Key") == "test-api-key-123"
        return Response(200, json={"authenticated": True})

    respx.get("https://api.example.com/protected").mock(side_effect=check_api_key)

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/protected",
            "auth_type": "api_key",
            "auth_token": "test-api-key-123",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.metadata["auth_type"] == "api_key"


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_custom_api_key_header(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test API key authentication with custom header name."""
    def check_custom_header(request):
        assert request.headers.get("X-Custom-Auth") == "custom-key-456"
        return Response(200, json={"authenticated": True})

    respx.get("https://api.example.com/custom").mock(side_effect=check_custom_header)

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/custom",
            "auth_type": "api_key",
            "auth_token": "custom-key-456",
            "auth_header": "X-Custom-Auth",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_bearer_token_auth(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test Bearer token authentication."""
    def check_bearer_token(request):
        assert request.headers.get("Authorization") == "Bearer test-bearer-token"
        return Response(200, json={"authenticated": True})

    respx.get("https://api.example.com/bearer").mock(side_effect=check_bearer_token)

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/bearer",
            "auth_type": "bearer_token",
            "auth_token": "test-bearer-token",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.metadata["auth_type"] == "bearer_token"


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_oauth_auth(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test OAuth authentication (uses Bearer token format)."""
    def check_oauth_token(request):
        assert request.headers.get("Authorization") == "Bearer oauth-access-token"
        return Response(200, json={"authenticated": True})

    respx.get("https://api.example.com/oauth").mock(side_effect=check_oauth_token)

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/oauth",
            "auth_type": "oauth",
            "auth_token": "oauth-access-token",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.metadata["auth_type"] == "oauth"


# Response Parsing Tests


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_json_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test JSON response parsing."""
    json_data = {"key": "value", "number": 42, "nested": {"field": "data"}}

    respx.get("https://api.example.com/json").mock(
        return_value=Response(200, json=json_data)
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/json"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "application/json" in result_data["content_type"]
    assert result_data["body"] == json_data


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_text_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test plain text response parsing."""
    text_content = "This is plain text content"

    respx.get("https://api.example.com/text").mock(
        return_value=Response(
            200,
            text=text_content,
            headers={"content-type": "text/plain"},
        )
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/text"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "text/plain" in result_data["content_type"]
    assert result_data["body"] == text_content


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_binary_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test binary response parsing (returns base64 encoded)."""
    binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00"  # PNG header

    respx.get("https://api.example.com/image").mock(
        return_value=Response(
            200,
            content=binary_content,
            headers={"content-type": "image/png"},
        )
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/image"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "image/png" in result_data["content_type"]
    assert result_data["is_binary"] is True
    assert result_data["body"] == base64.b64encode(binary_content).decode("utf-8")


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_invalid_json_response(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of invalid JSON in response."""
    invalid_json = "{ invalid json }"

    respx.get("https://api.example.com/badjson").mock(
        return_value=Response(
            200,
            text=invalid_json,
            headers={"content-type": "application/json"},
        )
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/badjson"},
        context=execution_context,
    )

    # Should still succeed, but return text instead of parsed JSON
    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["body"] == invalid_json


# Error Handling Tests


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_http_error_404(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of 404 Not Found error."""
    respx.get("https://api.example.com/notfound").mock(
        return_value=Response(404, json={"error": "Not found"})
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/notfound"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "404" in result.error
    result_data = result.result
    assert result_data["status_code"] == 404
    assert result_data["success"] is False


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_http_error_500(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of 500 Internal Server Error."""
    respx.get("https://api.example.com/error").mock(
        return_value=Response(500, text="Internal Server Error")
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/error"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "500" in result.error
    result_data = result.result
    assert result_data["status_code"] == 500


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_timeout(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of request timeout."""
    import httpx

    respx.get("https://api.example.com/slow").mock(
        side_effect=httpx.TimeoutException("Request timeout")
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/slow", "timeout": 1},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert result.error_type == "TimeoutError"


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_network_error(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of network connection errors."""
    import httpx

    respx.get("https://api.example.com/unreachable").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    result = await rest_api_tool.execute(
        parameters={"url": "https://api.example.com/unreachable"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert result.error_type == "HttpError"


# Custom Headers Tests


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_custom_headers(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test passing custom headers."""
    def check_headers(request):
        assert request.headers.get("X-Custom-Header") == "custom-value"
        assert request.headers.get("User-Agent") == "TestAgent/1.0"
        return Response(200, json={"received": True})

    respx.get("https://api.example.com/headers").mock(side_effect=check_headers)

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/headers",
            "headers": {
                "X-Custom-Header": "custom-value",
                "User-Agent": "TestAgent/1.0",
            },
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS


# Tool Metadata Tests


def test_rest_api_tool_metadata(rest_api_tool: RESTAPITool):
    """Test that tool metadata is correctly configured."""
    metadata = rest_api_tool.metadata

    assert metadata.tool_id == "rest_api"
    assert metadata.name == "REST API"
    assert metadata.version == "2.0.0"
    assert metadata.auth_method.value == "none"
    assert metadata.is_retryable is True
    assert metadata.max_retries == 3
    assert metadata.timeout_seconds == 305
    assert metadata.is_idempotent is False

    # Verify parameters
    assert "url" in metadata.parameters
    assert "method" in metadata.parameters
    assert "auth_type" in metadata.parameters
    assert "auth_token" in metadata.parameters
    assert "auth_header" in metadata.parameters

    assert metadata.parameters["url"].required is True
    assert metadata.parameters["method"].default == "GET"
    assert metadata.parameters["auth_type"].default == "none"
    assert metadata.parameters["auth_header"].default == "X-API-Key"

    # Verify method enum
    assert set(metadata.parameters["method"].enum) == {"GET", "POST", "PUT", "DELETE"}

    # Verify auth_type enum
    assert set(metadata.parameters["auth_type"].enum) == {
        "none",
        "api_key",
        "bearer_token",
        "oauth",
    }

    # Verify capabilities
    assert "authenticated_api" in metadata.capabilities
    assert "rest_api" in metadata.capabilities


# Edge Cases


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_empty_response_body(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test handling of empty response body."""
    respx.delete("https://api.example.com/delete/1").mock(
        return_value=Response(204, text="")  # 204 No Content
    )

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/delete/1",
            "method": "DELETE",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["status_code"] == 204


@pytest.mark.asyncio
@respx.mock
async def test_rest_api_execution_metadata(
    rest_api_tool: RESTAPITool,
    execution_context: ExecutionContext,
):
    """Test that execution metadata is correctly populated."""
    respx.get("https://api.example.com/test").mock(
        return_value=Response(200, json={"test": True})
    )

    result = await rest_api_tool.execute(
        parameters={
            "url": "https://api.example.com/test",
            "auth_type": "bearer_token",
            "auth_token": "test-token",
        },
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "rest_api"
    assert result.execution_time_ms > 0
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
    assert result.metadata["auth_type"] == "bearer_token"
