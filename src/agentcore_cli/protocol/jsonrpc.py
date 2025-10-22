"""JSON-RPC 2.0 client implementation.

This module provides a JSON-RPC 2.0 compliant client that enforces protocol
specification and properly wraps parameters in the `params` object.

This is the CRITICAL fix for the A2A protocol violation in v1.0.
"""

from __future__ import annotations

from typing import Any
import uuid

from agentcore_cli.protocol.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    A2AContext,
)
from agentcore_cli.protocol.exceptions import (
    JsonRpcProtocolError,
    InvalidRequestError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    ParseError,
)
from agentcore_cli.transport.http import HttpTransport


class JsonRpcClient:
    """JSON-RPC 2.0 client with A2A protocol support.

    This client enforces JSON-RPC 2.0 specification compliance by:
    1. Wrapping all parameters in `params` object (CRITICAL FIX)
    2. Validating requests/responses with Pydantic models
    3. Handling A2A context for distributed tracing
    4. Supporting batch requests
    5. Translating protocol errors to typed exceptions

    Args:
        transport: HTTP transport instance
        auth_token: Optional JWT token for authentication
        endpoint: API endpoint path (default: "/api/v1/jsonrpc")

    Attributes:
        transport: HTTP transport for network communication
        auth_token: Authentication token
        endpoint: API endpoint path
        request_id: Auto-incrementing request ID counter

    Example:
        >>> transport = HttpTransport("http://localhost:8001")
        >>> client = JsonRpcClient(transport, auth_token="jwt-token")
        >>> result = client.call("agent.register", {"name": "test-agent"})
        >>> print(result["agent_id"])
        'agent-001'
    """

    def __init__(
        self,
        transport: HttpTransport,
        auth_token: str | None = None,
        endpoint: str = "/api/v1/jsonrpc",
    ) -> None:
        """Initialize JSON-RPC client.

        Args:
            transport: HTTP transport instance
            auth_token: Optional authentication token
            endpoint: API endpoint path
        """
        self.transport = transport
        self.auth_token = auth_token
        self.endpoint = endpoint
        self.request_id = 0

    def _next_id(self) -> int:
        """Generate next request ID.

        Returns:
            Auto-incremented request ID

        Example:
            >>> client = JsonRpcClient(transport)
            >>> id1 = client._next_id()
            >>> id2 = client._next_id()
            >>> assert id2 == id1 + 1
        """
        self.request_id += 1
        return self.request_id

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for request.

        Includes authentication token if configured.

        Returns:
            Dictionary of HTTP headers

        Example:
            >>> client = JsonRpcClient(transport, auth_token="jwt-123")
            >>> headers = client._build_headers()
            >>> print(headers["Authorization"])
            'Bearer jwt-123'
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        a2a_context: A2AContext | None = None,
    ) -> dict[str, Any]:
        """Execute JSON-RPC method call.

        CRITICAL: This method properly wraps parameters in `params` object,
        fixing the protocol violation from v1.0.

        Args:
            method: Method name to invoke (e.g., "agent.register")
            params: Method parameters (will be wrapped in params object)
            a2a_context: Optional A2A context for distributed tracing

        Returns:
            Method result as dictionary

        Raises:
            InvalidRequestError: Request format is invalid
            MethodNotFoundError: Method does not exist
            InvalidParamsError: Parameters are invalid
            InternalError: Server internal error
            JsonRpcProtocolError: Other protocol errors

        Example:
            >>> client = JsonRpcClient(transport)
            >>> result = client.call(
            ...     "agent.register",
            ...     {"name": "test-agent", "capabilities": ["python"]},
            ... )
            >>> print(result["agent_id"])
            'agent-001'
        """
        # Build request with Pydantic validation
        request = JsonRpcRequest(
            method=method,
            params=params or {},  # CRITICAL: params wrapper
            id=self._next_id(),
        )

        # Inject A2A context if provided
        if a2a_context:
            request.params["a2a_context"] = a2a_context.model_dump(exclude_none=True)

        # Validate and serialize request
        try:
            request_data = request.model_dump(exclude_none=True)
        except Exception as e:
            raise InvalidRequestError(f"Request validation failed: {str(e)}")

        # Send via transport
        response_data = self.transport.post(
            self.endpoint,
            request_data,
            headers=self._build_headers(),
        )

        # Ensure response is dict (not list for single call)
        if not isinstance(response_data, dict):
            raise ParseError("Expected JSON object response, got array")

        # Parse and validate response
        try:
            response = JsonRpcResponse(**response_data)
        except Exception as e:
            raise ParseError(f"Response validation failed: {str(e)}")

        # Handle errors
        if response.error:
            self._raise_error(response.error)

        # Return result
        return response.result or {}

    def batch_call(
        self,
        calls: list[tuple[str, dict[str, Any] | None]],
    ) -> list[dict[str, Any]]:
        """Execute batch JSON-RPC calls.

        Send multiple requests in a single HTTP call for efficiency.

        Args:
            calls: List of (method, params) tuples

        Returns:
            List of results in same order as requests

        Raises:
            InvalidRequestError: Batch request format is invalid
            JsonRpcProtocolError: Protocol error in batch

        Example:
            >>> client = JsonRpcClient(transport)
            >>> results = client.batch_call([
            ...     ("agent.list", {"limit": 10}),
            ...     ("task.list", {"limit": 10}),
            ... ])
            >>> len(results)
            2
        """
        # Build batch request
        requests: list[dict[str, Any]] = []
        for method, params in calls:
            request = JsonRpcRequest(
                method=method,
                params=params or {},
                id=self._next_id(),
            )
            requests.append(request.model_dump(exclude_none=True))

        # Send batch via transport
        response_data = self.transport.post(
            self.endpoint,
            requests,
            headers=self._build_headers(),
        )

        # Parse responses
        if not isinstance(response_data, list):
            raise InvalidRequestError("Batch response must be array")

        results: list[dict[str, Any]] = []
        for item in response_data:
            try:
                response = JsonRpcResponse(**item)
            except Exception as e:
                raise ParseError(f"Response validation failed: {str(e)}")

            if response.error:
                self._raise_error(response.error)

            results.append(response.result or {})

        return results

    def notify(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send JSON-RPC notification (no response expected).

        Notifications are requests without an `id` field. The server will
        not send a response.

        Args:
            method: Method name to invoke
            params: Method parameters

        Example:
            >>> client = JsonRpcClient(transport)
            >>> client.notify("agent.heartbeat", {"agent_id": "agent-001"})
        """
        # Build notification (no id field)
        request = JsonRpcRequest(
            method=method,
            params=params or {},
            id=None,  # Notifications have no ID
        )

        # Validate and serialize
        request_data = request.model_dump(exclude_none=True)
        # Remove id field completely for notifications
        if "id" in request_data:
            del request_data["id"]

        # Send via transport (no response expected)
        self.transport.post(
            self.endpoint,
            request_data,
            headers=self._build_headers(),
        )

    def _raise_error(self, error: JsonRpcError) -> None:
        """Raise appropriate exception based on error code.

        Translates JSON-RPC error codes to typed exceptions for better
        error handling.

        Args:
            error: JSON-RPC error object

        Raises:
            MethodNotFoundError: Method not found (-32601)
            InvalidParamsError: Invalid params (-32602)
            InternalError: Internal error (-32603)
            ParseError: Parse error (-32700)
            InvalidRequestError: Invalid request (-32600)
            JsonRpcProtocolError: Other errors
        """
        # Create exception with appropriate arguments based on error code
        if error.code == -32601:
            # MethodNotFoundError expects method name
            method = error.data.get("method", "unknown") if error.data else "unknown"
            raise MethodNotFoundError(method, data=error.data)
        elif error.code == -32602:
            # InvalidParamsError
            raise InvalidParamsError(error.message, data=error.data)
        elif error.code == -32603:
            # InternalError
            raise InternalError(error.message, data=error.data)
        elif error.code == -32700:
            # ParseError
            raise ParseError(error.message, data=error.data)
        elif error.code == -32600:
            # InvalidRequestError
            raise InvalidRequestError(error.message, data=error.data)
        else:
            # Generic protocol error
            raise JsonRpcProtocolError(error.message, code=error.code, data=error.data)
