"""
JSON-RPC 2.0 Protocol Models

Implements JSON-RPC 2.0 specification compliant data models for request/response handling.
Includes A2A protocol extensions for agent communication.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class JsonRpcVersion(str, Enum):
    """JSON-RPC version identifier."""
    V2_0 = "2.0"


class JsonRpcErrorCode(int, Enum):
    """Standard JSON-RPC 2.0 error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Server error range (-32000 to -32099)
    SERVER_ERROR_MIN = -32099
    SERVER_ERROR_MAX = -32000

    # A2A Protocol specific errors (-32100 to -32199)
    AGENT_NOT_FOUND = -32100
    AGENT_UNAVAILABLE = -32101
    TASK_CREATION_FAILED = -32102
    AUTHENTICATION_FAILED = -32103
    AUTHORIZATION_FAILED = -32104
    CAPABILITY_MISMATCH = -32105
    TIMEOUT_ERROR = -32106


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error data")

    @field_validator('code')
    @classmethod
    def validate_error_code(cls, v: int) -> int:
        """Validate error code is within valid ranges."""
        if v in [e.value for e in JsonRpcErrorCode]:
            return v
        # Allow server-specific errors in the -32000 to -32099 range
        # Note: SERVER_ERROR_MIN is -32099, SERVER_ERROR_MAX is -32000
        if JsonRpcErrorCode.SERVER_ERROR_MIN <= v <= JsonRpcErrorCode.SERVER_ERROR_MAX:
            return v
        # Allow A2A specific errors in the -32100 to -32199 range
        if -32199 <= v <= -32100:
            return v
        raise ValueError(f"Invalid JSON-RPC error code: {v}")


class A2AContext(BaseModel):
    """A2A protocol context for agent communication."""
    source_agent: str = Field(..., description="Source agent identifier")
    target_agent: Optional[str] = Field(None, description="Target agent identifier")
    trace_id: str = Field(default_factory=lambda: str(uuid4()), description="Request trace ID")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request object with A2A extensions."""
    jsonrpc: JsonRpcVersion = Field(default=JsonRpcVersion.V2_0, description="JSON-RPC version")
    method: str = Field(..., description="Method name to invoke")
    params: Optional[Union[Dict[str, Any], List[Any]]] = Field(None, description="Method parameters")
    id: Optional[Union[str, int]] = Field(None, description="Request identifier (null for notifications)")

    # A2A Protocol extensions
    a2a_context: Optional[A2AContext] = Field(None, description="A2A protocol context")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Method name must be a non-empty string")
        # Allow rpc.* methods for built-in functionality
        # Only reject methods that are not explicitly defined
        return v

    @property
    def is_notification(self) -> bool:
        """Check if this is a notification (no response expected)."""
        return self.id is None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response object."""
    jsonrpc: JsonRpcVersion = Field(default=JsonRpcVersion.V2_0, description="JSON-RPC version")
    result: Optional[Any] = Field(None, description="Method result (present on success)")
    error: Optional[JsonRpcError] = Field(None, description="Error object (present on error)")
    id: Optional[Union[str, int]] = Field(..., description="Request identifier")

    def model_post_init(self, __context) -> None:
        """Validate that either result or error is present, but not both."""
        if self.result is not None and self.error is not None:
            raise ValueError("Response cannot have both result and error")
        if self.result is None and self.error is None:
            raise ValueError("Response must have either result or error")


class JsonRpcBatchRequest(BaseModel):
    """JSON-RPC 2.0 batch request."""
    requests: List[JsonRpcRequest] = Field(..., min_length=1, description="List of requests")

    @field_validator('requests')
    @classmethod
    def validate_non_empty_batch(cls, v: List[JsonRpcRequest]) -> List[JsonRpcRequest]:
        """Ensure batch is not empty."""
        if not v:
            raise ValueError("Batch request cannot be empty")
        return v


class JsonRpcBatchResponse(BaseModel):
    """JSON-RPC 2.0 batch response."""
    responses: List[JsonRpcResponse] = Field(..., description="List of responses")


class MessageEnvelope(BaseModel):
    """Message envelope for routing and metadata."""
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique message identifier")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    source: str = Field(..., description="Source identifier")
    destination: Optional[str] = Field(None, description="Destination identifier")
    content_type: str = Field(default="application/json", description="Content type")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    payload: Union[JsonRpcRequest, JsonRpcResponse, JsonRpcBatchRequest, JsonRpcBatchResponse] = Field(
        ..., description="JSON-RPC payload"
    )


def create_error_response(
    request_id: Optional[Union[str, int]],
    error_code: JsonRpcErrorCode,
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> JsonRpcResponse:
    """Create a standard JSON-RPC error response."""
    error_messages = {
        JsonRpcErrorCode.PARSE_ERROR: "Parse error",
        JsonRpcErrorCode.INVALID_REQUEST: "Invalid Request",
        JsonRpcErrorCode.METHOD_NOT_FOUND: "Method not found",
        JsonRpcErrorCode.INVALID_PARAMS: "Invalid params",
        JsonRpcErrorCode.INTERNAL_ERROR: "Internal error",
        JsonRpcErrorCode.AGENT_NOT_FOUND: "Agent not found",
        JsonRpcErrorCode.AGENT_UNAVAILABLE: "Agent unavailable",
        JsonRpcErrorCode.TASK_CREATION_FAILED: "Task creation failed",
        JsonRpcErrorCode.AUTHENTICATION_FAILED: "Authentication failed",
        JsonRpcErrorCode.AUTHORIZATION_FAILED: "Authorization failed",
        JsonRpcErrorCode.CAPABILITY_MISMATCH: "Capability mismatch",
        JsonRpcErrorCode.TIMEOUT_ERROR: "Timeout error",
    }

    error_message = message or error_messages.get(error_code, "Server error")

    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(
            code=error_code.value,
            message=error_message,
            data=data
        )
    )


def create_success_response(
    request_id: Optional[Union[str, int]],
    result: Any
) -> JsonRpcResponse:
    """Create a successful JSON-RPC response."""
    return JsonRpcResponse(
        id=request_id,
        result=result
    )