"""Protocol layer for AgentCore CLI.

This module handles JSON-RPC 2.0 protocol enforcement with no knowledge of
HTTP transport or business logic. It is responsible for:
- JSON-RPC 2.0 specification compliance
- Request/response Pydantic validation
- Proper params wrapper (CRITICAL FIX)
- A2A context management
- Batch request handling
- Protocol error translation
"""

from agentcore_cli.protocol.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    A2AContext,
)
from agentcore_cli.protocol.exceptions import (
    ProtocolError,
    JsonRpcProtocolError,
    InvalidRequestError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    ParseError,
)
from agentcore_cli.protocol.jsonrpc import JsonRpcClient

__all__ = [
    # Models
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "A2AContext",
    # Exceptions
    "ProtocolError",
    "JsonRpcProtocolError",
    "InvalidRequestError",
    "MethodNotFoundError",
    "InvalidParamsError",
    "InternalError",
    "ParseError",
    # Client
    "JsonRpcClient",
]
