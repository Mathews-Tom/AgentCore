"""
JSON-RPC 2.0 API Router

FastAPI router for handling JSON-RPC 2.0 requests over HTTP and WebSocket.
Integrates with the JSON-RPC processor service for request handling.
"""

from typing import Any, Dict, List, Union

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
import structlog

from agentcore.a2a_protocol.services.jsonrpc_handler import jsonrpc_processor
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcErrorCode

router = APIRouter()
logger = structlog.get_logger()


@router.post("/jsonrpc", summary="JSON-RPC 2.0 Endpoint")
async def jsonrpc_endpoint(
    request: Request,
    raw_data: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> JSONResponse:
    """
    Main JSON-RPC 2.0 endpoint for processing requests.

    Accepts both single requests and batch requests according to the JSON-RPC 2.0 specification.
    Supports A2A protocol extensions for agent communication.

    Args:
        request: FastAPI request object
        raw_data: JSON-RPC request data (single or batch)

    Returns:
        JSON-RPC response or HTTP error
    """
    try:
        # Add request metadata for tracing
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        logger.info(
            "Received JSON-RPC request",
            client_ip=client_ip,
            user_agent=user_agent,
            content_type=request.headers.get("content-type"),
            is_batch=isinstance(raw_data, list)
        )

        # Process the request through JSON-RPC handler
        response = await jsonrpc_processor.process_message(raw_data)

        if response is None:
            # Notification - no response expected
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # Return JSON response
        response_data = response.model_dump(exclude_none=True)
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error("JSON-RPC endpoint error", error=str(e), exc_info=True)

        # Return internal server error
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": JsonRpcErrorCode.INTERNAL_ERROR.value,
                "message": "Internal server error",
                "data": {"details": str(e)}
            },
            "id": None
        }

        return JSONResponse(
            content=error_response,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/jsonrpc/methods", summary="List JSON-RPC Methods")
async def list_jsonrpc_methods() -> Dict[str, List[str]]:
    """
    List all registered JSON-RPC methods.

    Useful for service discovery and debugging.

    Returns:
        Dictionary containing list of available methods
    """
    return {
        "methods": list(jsonrpc_processor.methods.keys())
    }


@router.post("/jsonrpc/ping", summary="JSON-RPC Ping Test")
async def jsonrpc_ping() -> Dict[str, Any]:
    """
    Simple ping endpoint for testing JSON-RPC connectivity.

    Returns:
        Ping response with timestamp
    """
    from datetime import datetime

    return {
        "pong": True,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "agentcore-a2a-protocol"
    }