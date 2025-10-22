"""JSON-RPC 2.0 protocol models.

This module defines Pydantic models for JSON-RPC 2.0 specification compliance.
These models ensure proper request/response formatting and validation.

References:
    JSON-RPC 2.0 Specification: https://www.jsonrpc.org/specification
    A2A Protocol: docs/specs/a2a-protocol/spec.md
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object.

    Represents an error that occurred during request processing.

    Attributes:
        code: Error code (integer)
        message: Human-readable error message
        data: Additional error information (optional)

    Example:
        >>> error = JsonRpcError(code=-32600, message="Invalid Request")
        >>> print(error.code)
        -32600
    """

    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: dict[str, Any] | None = Field(default=None, description="Additional error data")


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request object.

    Represents a request to invoke a method on the server.

    Attributes:
        jsonrpc: Protocol version (always "2.0")
        method: Method name to invoke
        params: Method parameters wrapped in object (CRITICAL for A2A compliance)
        id: Request identifier (required for request, omitted for notification)

    Example:
        >>> request = JsonRpcRequest(
        ...     method="agent.register",
        ...     params={"name": "test-agent", "capabilities": ["python"]},
        ...     id=1
        ... )
        >>> print(request.params)
        {'name': 'test-agent', 'capabilities': ['python']}
    """

    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC version")
    method: str = Field(..., description="Method name to invoke")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Method parameters (MUST be object, not array)"
    )
    id: int | str | None = Field(default=None, description="Request ID")

    model_config = ConfigDict(frozen=False)


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response object.

    Represents a response from the server after method invocation.

    Attributes:
        jsonrpc: Protocol version (always "2.0")
        result: Method result (present on success, mutually exclusive with error)
        error: Error object (present on failure, mutually exclusive with result)
        id: Request identifier (echoed from request)

    Example:
        >>> response = JsonRpcResponse(
        ...     result={"agent_id": "agent-001"},
        ...     id=1
        ... )
        >>> print(response.result)
        {'agent_id': 'agent-001'}
    """

    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC version")
    result: dict[str, Any] | None = Field(default=None, description="Method result")
    error: JsonRpcError | None = Field(default=None, description="Error object")
    id: int | str | None = Field(default=None, description="Request ID")

    model_config = ConfigDict(frozen=False)


class A2AContext(BaseModel):
    """A2A protocol context for distributed tracing.

    Optional context that can be included in JSON-RPC requests to support
    agent-to-agent communication tracking.

    Attributes:
        trace_id: Unique identifier for the entire trace
        source_agent: ID of the agent making the request
        target_agent: ID of the target agent (optional)
        session_id: Session identifier for multi-turn interactions
        timestamp: Request timestamp in ISO 8601 format

    Example:
        >>> context = A2AContext(
        ...     trace_id="trace-123",
        ...     source_agent="agent-001",
        ...     session_id="session-456"
        ... )
        >>> print(context.trace_id)
        'trace-123'
    """

    trace_id: str = Field(..., description="Unique trace identifier")
    source_agent: str = Field(..., description="Source agent ID")
    target_agent: str | None = Field(default=None, description="Target agent ID")
    session_id: str | None = Field(default=None, description="Session identifier")
    timestamp: str | None = Field(default=None, description="Request timestamp")

    model_config = ConfigDict(frozen=False)
