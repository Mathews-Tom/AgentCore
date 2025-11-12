"""Native Tool ABC implementations for API client tools.

This module provides Tool ABC implementations for REST and GraphQL API interactions.
These are native implementations that directly inherit from Tool ABC (not legacy
function-based tools).

Migration from: agent_runtime/tools/api_tools.py
Status: Stage 3 - Native Migration
"""

import json
import time
from typing import Any

import httpx

from ...models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from ..base import ExecutionContext, Tool


class HttpRequestTool(Tool):
    """Generic HTTP request tool for REST API interactions.

    Supports all standard HTTP methods with custom headers and body.
    """

    def __init__(self):
        """Initialize HTTP request tool with metadata."""
        metadata = ToolDefinition(
            tool_id="http_request",
            name="HTTP Request",
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
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=305,
            is_idempotent=False,  # Depends on HTTP method
            capabilities=["http_client", "rest_api", "external_api"],
            tags=["http", "rest", "api", "web"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute HTTP request.

        Args:
            parameters: Dictionary with keys:
                - url: str - Request URL
                - method: str - HTTP method (default: "GET")
                - headers: dict - Request headers (optional)
                - body: str - Request body (optional)
                - timeout: int - Request timeout in seconds (default: 30)
            context: Execution context

        Returns:
            ToolResult with HTTP response data
        """
        start_time = time.time()

        try:
            url = parameters["url"]
            method = parameters.get("method", "GET").upper()
            headers = parameters.get("headers")
            body = parameters.get("body")
            timeout = int(parameters.get("timeout", 30))

            # Validate parameters
            if not url or not url.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="URL cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            self.logger.info(
                "http_request_executing",
                url=url,
                method=method,
                timeout=timeout,
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Prepare request
                request_kwargs: dict[str, Any] = {
                    "method": method,
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
                result_data = {
                    "success": response.is_success,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": None,
                    "error": None,
                }

                # Try to parse response as JSON
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        result_data["body"] = response.json()
                    except json.JSONDecodeError:
                        result_data["body"] = response.text
                else:
                    result_data["body"] = response.text

                if not response.is_success:
                    result_data["error"] = f"HTTP {response.status_code}: {response.reason_phrase}"

                execution_time_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    "http_request_completed",
                    url=url,
                    status_code=response.status_code,
                    success=response.is_success,
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS if response.is_success else ToolExecutionStatus.FAILED,
                    result=result_data,
                    error=result_data["error"],
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                    },
                )

        except httpx.TimeoutException:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.warning("http_request_timeout", url=url, timeout=timeout)

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=f"Request timed out after {timeout} seconds",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

        except httpx.HTTPError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error("http_request_error", url=url, error=str(e))

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=f"HTTP error: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "http_request_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )


class RestGetTool(Tool):
    """REST GET request tool for fetching resources.

    Specialized tool for GET requests with query parameters.
    """

    def __init__(self):
        """Initialize REST GET tool with metadata."""
        metadata = ToolDefinition(
            tool_id="rest_get",
            name="REST GET",
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
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=35,
            is_idempotent=True,  # GET is idempotent
            capabilities=["http_client", "rest_api", "external_api"],
            tags=["http", "rest", "api", "get"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute REST GET request.

        Args:
            parameters: Dictionary with keys:
                - url: str - Request URL
                - headers: dict - Request headers (optional)
                - params: dict - Query parameters (optional)
            context: Execution context

        Returns:
            ToolResult with HTTP response data
        """
        start_time = time.time()

        try:
            url = parameters["url"]
            headers = parameters.get("headers")
            params = parameters.get("params")

            # Validate URL
            if not url or not url.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="URL cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Build URL with query parameters
            if params:
                query_string = "&".join(f"{k}={v}" for k, v in params.items())
                url = f"{url}?{query_string}" if "?" not in url else f"{url}&{query_string}"

            self.logger.info("rest_get_executing", url=url)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)

                # Parse response
                result_data = {
                    "success": response.is_success,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": None,
                    "error": None,
                }

                # Try to parse response as JSON
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        result_data["body"] = response.json()
                    except json.JSONDecodeError:
                        result_data["body"] = response.text
                else:
                    result_data["body"] = response.text

                if not response.is_success:
                    result_data["error"] = f"HTTP {response.status_code}: {response.reason_phrase}"

                execution_time_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    "rest_get_completed",
                    url=url,
                    status_code=response.status_code,
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS if response.is_success else ToolExecutionStatus.FAILED,
                    result=result_data,
                    error=result_data["error"],
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                    },
                )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "rest_get_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )


class RestPostTool(Tool):
    """REST POST request tool for creating resources.

    Specialized tool for POST requests with JSON body.
    """

    def __init__(self):
        """Initialize REST POST tool with metadata."""
        metadata = ToolDefinition(
            tool_id="rest_post",
            name="REST POST",
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
            auth_method=AuthMethod.NONE,
            is_retryable=False,  # POST may not be idempotent
            max_retries=1,
            timeout_seconds=35,
            is_idempotent=False,
            capabilities=["http_client", "rest_api", "external_api"],
            tags=["http", "rest", "api", "post"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute REST POST request.

        Args:
            parameters: Dictionary with keys:
                - url: str - Request URL
                - body: str - Request body as JSON string
                - headers: dict - Request headers (optional)
            context: Execution context

        Returns:
            ToolResult with HTTP response data
        """
        start_time = time.time()

        try:
            url = parameters["url"]
            body = parameters["body"]
            headers = parameters.get("headers", {})

            # Validate parameters
            if not url or not url.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="URL cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            if not body or not body.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Body cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Set default content-type if not provided
            if "content-type" not in {k.lower() for k in headers.keys()}:
                headers["Content-Type"] = "application/json"

            self.logger.info("rest_post_executing", url=url)

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to parse body as JSON
                try:
                    json_body = json.loads(body)
                    response = await client.post(url, json=json_body, headers=headers)
                except json.JSONDecodeError:
                    # Use as raw content
                    response = await client.post(url, content=body, headers=headers)

                # Parse response
                result_data = {
                    "success": response.is_success,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": None,
                    "error": None,
                }

                # Try to parse response as JSON
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        result_data["body"] = response.json()
                    except json.JSONDecodeError:
                        result_data["body"] = response.text
                else:
                    result_data["body"] = response.text

                if not response.is_success:
                    result_data["error"] = f"HTTP {response.status_code}: {response.reason_phrase}"

                execution_time_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    "rest_post_completed",
                    url=url,
                    status_code=response.status_code,
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS if response.is_success else ToolExecutionStatus.FAILED,
                    result=result_data,
                    error=result_data["error"],
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                    },
                )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "rest_post_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )


class GraphQLQueryTool(Tool):
    """GraphQL query tool for executing GraphQL queries.

    Executes GraphQL queries against a GraphQL endpoint with variable support.
    """

    def __init__(self):
        """Initialize GraphQL query tool with metadata."""
        metadata = ToolDefinition(
            tool_id="graphql_query",
            name="GraphQL Query",
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
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=35,
            is_idempotent=True,  # GraphQL queries are typically idempotent
            capabilities=["graphql_client", "external_api"],
            tags=["graphql", "api", "query"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute GraphQL query.

        Args:
            parameters: Dictionary with keys:
                - endpoint: str - GraphQL endpoint URL
                - query: str - GraphQL query string
                - variables: dict - Query variables (optional)
                - headers: dict - Request headers (optional)
            context: Execution context

        Returns:
            ToolResult with GraphQL response data
        """
        start_time = time.time()

        try:
            endpoint = parameters["endpoint"]
            query = parameters["query"]
            variables = parameters.get("variables")
            headers = parameters.get("headers", {})

            # Validate parameters
            if not endpoint or not endpoint.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Endpoint cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            if not query or not query.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Query cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Build GraphQL request body
            request_body = {"query": query}
            if variables:
                request_body["variables"] = variables

            # Set content-type
            headers["Content-Type"] = "application/json"

            self.logger.info("graphql_query_executing", endpoint=endpoint)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    json=request_body,
                    headers=headers,
                )
                response.raise_for_status()

                # Parse GraphQL response
                response_data = response.json()

                graphql_result = {
                    "success": "errors" not in response_data,
                    "data": response_data.get("data"),
                    "errors": response_data.get("errors"),
                }

                execution_time_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    "graphql_query_completed",
                    endpoint=endpoint,
                    success=graphql_result["success"],
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS if graphql_result["success"] else ToolExecutionStatus.FAILED,
                    result=graphql_result,
                    error=str(graphql_result["errors"]) if graphql_result["errors"] else None,
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                    },
                )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "graphql_query_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )
