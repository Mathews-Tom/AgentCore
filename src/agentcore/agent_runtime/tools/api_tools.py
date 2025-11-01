"""API client tool adapters for REST and GraphQL APIs."""

import json
from typing import Any

import httpx
import structlog

from ..models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from ..services.tool_registry import ToolRegistry

logger = structlog.get_logger()


async def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Make an HTTP request to a REST API.

    Args:
        url: Request URL
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        headers: Request headers
        body: Request body (JSON string or raw text)
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data
    """
    logger.info(
        "http_request_called",
        url=url,
        method=method,
        timeout=timeout,
    )

    result = {
        "success": False,
        "status_code": None,
        "headers": {},
        "body": None,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Prepare request
            request_kwargs: dict[str, Any] = {
                "method": method.upper(),
                "url": url,
            }

            if headers:
                request_kwargs["headers"] = headers

            if body:
                # Try to parse as JSON first
                try:
                    request_kwargs["json"] = json.loads(body)
                except json.JSONDecodeError:
                    # Use as raw text
                    request_kwargs["content"] = body

            # Make request
            response = await client.request(**request_kwargs)

            # Parse response
            result["success"] = response.is_success
            result["status_code"] = response.status_code
            result["headers"] = dict(response.headers)

            # Try to parse response as JSON
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    result["body"] = response.json()
                except json.JSONDecodeError:
                    result["body"] = response.text
            else:
                result["body"] = response.text

            if not response.is_success:
                result["error"] = f"HTTP {response.status_code}: {response.reason_phrase}"

    except httpx.TimeoutException:
        result["error"] = f"Request timed out after {timeout} seconds"
        logger.warning("http_request_timeout", url=url, timeout=timeout)
    except httpx.HTTPError as e:
        result["error"] = f"HTTP error: {str(e)}"
        logger.error("http_request_error", url=url, error=str(e))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error("http_request_unexpected_error", url=url, error=str(e))

    return result


async def rest_get(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Make a GET request to a REST API.

    Args:
        url: Request URL
        headers: Request headers
        params: Query parameters

    Returns:
        Dictionary with response data
    """
    logger.info("rest_get_called", url=url)

    # Build URL with query parameters
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query_string}" if "?" not in url else f"{url}&{query_string}"

    return await http_request(url=url, method="GET", headers=headers)


async def rest_post(
    url: str,
    body: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Make a POST request to a REST API.

    Args:
        url: Request URL
        body: Request body (JSON string)
        headers: Request headers

    Returns:
        Dictionary with response data
    """
    logger.info("rest_post_called", url=url)

    # Set default content-type if not provided
    if headers is None:
        headers = {}
    if "content-type" not in {k.lower() for k in headers.keys()}:
        headers["Content-Type"] = "application/json"

    return await http_request(url=url, method="POST", headers=headers, body=body)


async def graphql_query(
    endpoint: str,
    query: str,
    variables: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Execute a GraphQL query.

    Args:
        endpoint: GraphQL endpoint URL
        query: GraphQL query string
        variables: Query variables
        headers: Request headers

    Returns:
        Dictionary with query result
    """
    logger.info("graphql_query_called", endpoint=endpoint)

    # Build GraphQL request body
    request_body = {"query": query}
    if variables:
        request_body["variables"] = variables

    # Set content-type
    if headers is None:
        headers = {}
    headers["Content-Type"] = "application/json"

    # Make POST request to GraphQL endpoint
    result = await http_request(
        url=endpoint,
        method="POST",
        headers=headers,
        body=json.dumps(request_body),
    )

    # Extract GraphQL-specific data
    if result["success"] and isinstance(result.get("body"), dict):
        graphql_result = {
            "success": "errors" not in result["body"],
            "data": result["body"].get("data"),
            "errors": result["body"].get("errors"),
        }
        return graphql_result

    return result


def register_api_tools(registry: ToolRegistry) -> None:
    """
    Register API client tools with the tool registry.

    Args:
        registry: ToolRegistry instance
    """
    # HTTP Request tool
    http_request_def = ToolDefinition(
        tool_id="http_request",
        name="http_request",
        description="Make an HTTP request to any REST API",
        version="1.0.0",
        category=ToolCategory.API_CLIENT,
        parameters={
            "url": ToolParameter(
                name="url",
                type="string",
                description="Request URL",
                required=True,
                min_length=10,
                max_length=2000,
            ),
            "method": ToolParameter(
                name="method",
                type="string",
                description="HTTP method",
                required=False,
                default="GET",
                enum=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
            "headers": ToolParameter(
                name="headers",
                type="object",
                description="Request headers as JSON object",
                required=False,
            ),
            "body": ToolParameter(
                name="body",
                type="string",
                description="Request body (JSON string or raw text)",
                required=False,
            ),
            "timeout": ToolParameter(
                name="timeout",
                type="number",
                description="Request timeout in seconds",
                required=False,
                default=30,
                min_value=1,
                max_value=300,
            ),
        },
        timeout_seconds=305,
        is_retryable=True,
        is_idempotent=False,  # Depends on HTTP method
        capabilities=["http_client", "rest_api", "external_api"],
        tags=["http", "rest", "api", "web"],
        metadata={
            "supported_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        },
    )
    registry.register_tool(http_request_def, http_request)

    # REST GET tool
    rest_get_def = ToolDefinition(
        tool_id="rest_get",
        name="rest_get",
        description="Make a GET request to a REST API with query parameters",
        version="1.0.0",
        category=ToolCategory.API_CLIENT,
        parameters={
            "url": ToolParameter(
                name="url",
                type="string",
                description="Request URL",
                required=True,
                min_length=10,
                max_length=2000,
            ),
            "headers": ToolParameter(
                name="headers",
                type="object",
                description="Request headers as JSON object",
                required=False,
            ),
            "params": ToolParameter(
                name="params",
                type="object",
                description="Query parameters as JSON object",
                required=False,
            ),
        },
        timeout_seconds=35,
        is_retryable=True,
        is_idempotent=True,  # GET is idempotent
        capabilities=["http_client", "rest_api", "external_api"],
        tags=["http", "rest", "api", "get"],
    )
    registry.register_tool(rest_get_def, rest_get)

    # REST POST tool
    rest_post_def = ToolDefinition(
        tool_id="rest_post",
        name="rest_post",
        description="Make a POST request to a REST API with JSON body",
        version="1.0.0",
        category=ToolCategory.API_CLIENT,
        parameters={
            "url": ToolParameter(
                name="url",
                type="string",
                description="Request URL",
                required=True,
                min_length=10,
                max_length=2000,
            ),
            "body": ToolParameter(
                name="body",
                type="string",
                description="Request body as JSON string",
                required=True,
                min_length=1,
                max_length=100000,
            ),
            "headers": ToolParameter(
                name="headers",
                type="object",
                description="Request headers as JSON object",
                required=False,
            ),
        },
        timeout_seconds=35,
        is_retryable=False,  # POST may not be idempotent
        is_idempotent=False,
        capabilities=["http_client", "rest_api", "external_api"],
        tags=["http", "rest", "api", "post"],
    )
    registry.register_tool(rest_post_def, rest_post)

    # GraphQL Query tool
    graphql_query_def = ToolDefinition(
        tool_id="graphql_query",
        name="graphql_query",
        description="Execute a GraphQL query against a GraphQL endpoint",
        version="1.0.0",
        category=ToolCategory.API_CLIENT,
        parameters={
            "endpoint": ToolParameter(
                name="endpoint",
                type="string",
                description="GraphQL endpoint URL",
                required=True,
                min_length=10,
                max_length=2000,
            ),
            "query": ToolParameter(
                name="query",
                type="string",
                description="GraphQL query string",
                required=True,
                min_length=1,
                max_length=50000,
            ),
            "variables": ToolParameter(
                name="variables",
                type="object",
                description="Query variables as JSON object",
                required=False,
            ),
            "headers": ToolParameter(
                name="headers",
                type="object",
                description="Request headers as JSON object",
                required=False,
            ),
        },
        timeout_seconds=35,
        is_retryable=True,
        is_idempotent=True,  # GraphQL queries are typically idempotent
        capabilities=["graphql_client", "external_api"],
        tags=["graphql", "api", "query"],
        metadata={
            "protocol": "GraphQL",
        },
    )
    registry.register_tool(graphql_query_def, graphql_query)

    logger.info(
        "api_tools_registered",
        tools=["http_request", "rest_get", "rest_post", "graphql_query"],
    )
