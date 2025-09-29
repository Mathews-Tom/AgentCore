"""
JSON-RPC 2.0 Request Handler Service

Core service for processing JSON-RPC 2.0 requests with A2A protocol extensions.
Handles request parsing, validation, method routing, and response generation.
"""

import json
import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import structlog
from pydantic import ValidationError

from agentcore.a2a_protocol.models.jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcBatchRequest,
    JsonRpcBatchResponse,
    JsonRpcError,
    JsonRpcErrorCode,
    MessageEnvelope,
    create_error_response,
    create_success_response,
)

logger = structlog.get_logger()

# Type alias for JSON-RPC method handlers
JsonRpcHandler = Callable[[JsonRpcRequest], Any]


class JsonRpcProcessor:
    """
    JSON-RPC 2.0 request processor with A2A protocol support.

    Handles the complete request/response lifecycle including:
    - Request parsing and validation
    - Method routing and execution
    - Error handling and response generation
    - Batch request processing
    - A2A context management
    """

    def __init__(self) -> None:
        """Initialize the JSON-RPC processor."""
        self.methods: Dict[str, JsonRpcHandler] = {}
        self._middleware_stack: List[Callable] = []

    def register_method(self, name: str, handler: JsonRpcHandler) -> None:
        """
        Register a JSON-RPC method handler.

        Args:
            name: Method name (e.g., 'agent.register', 'task.create')
            handler: Async callable that processes the request
        """
        if name in self.methods:
            logger.warning("Overriding existing JSON-RPC method", method=name)

        self.methods[name] = handler
        logger.debug("Registered JSON-RPC method", method=name)

    def unregister_method(self, name: str) -> None:
        """Unregister a JSON-RPC method handler."""
        if name in self.methods:
            del self.methods[name]
            logger.debug("Unregistered JSON-RPC method", method=name)

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the processing stack."""
        self._middleware_stack.append(middleware)

    async def process_raw_message(self, raw_data: Union[str, bytes]) -> Optional[str]:
        """
        Process a raw JSON-RPC message.

        Args:
            raw_data: Raw JSON string or bytes

        Returns:
            JSON response string, or None for notifications
        """
        try:
            # Parse JSON
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode('utf-8')

            parsed_data = json.loads(raw_data)

            # Process the parsed data
            response = await self.process_message(parsed_data)

            # Return JSON response if not None
            if response is not None:
                return json.dumps(response.model_dump(exclude_none=True))

            return None

        except json.JSONDecodeError as e:
            logger.error("JSON parse error", error=str(e))
            error_response = create_error_response(
                request_id=None,
                error_code=JsonRpcErrorCode.PARSE_ERROR,
                data={"details": str(e)}
            )
            return json.dumps(error_response.model_dump(exclude_none=True))

        except Exception as e:
            logger.error("Unexpected error processing message", error=str(e), exc_info=True)
            error_response = create_error_response(
                request_id=None,
                error_code=JsonRpcErrorCode.INTERNAL_ERROR,
                data={"details": str(e)}
            )
            return json.dumps(error_response.model_dump(exclude_none=True))

    async def process_message(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Optional[Union[JsonRpcResponse, JsonRpcBatchResponse]]:
        """
        Process a parsed JSON-RPC message.

        Args:
            data: Parsed JSON data (single request or batch)

        Returns:
            Response object or None for notifications
        """
        try:
            # Handle batch requests
            if isinstance(data, list):
                return await self._process_batch(data)

            # Handle single request
            return await self._process_single_request(data)

        except Exception as e:
            logger.error("Error processing message", error=str(e), exc_info=True)
            return create_error_response(
                request_id=data.get('id') if isinstance(data, dict) else None,
                error_code=JsonRpcErrorCode.INTERNAL_ERROR
            )

    async def _process_batch(self, batch_data: List[Dict[str, Any]]) -> JsonRpcBatchResponse:
        """Process a batch of JSON-RPC requests."""
        if not batch_data:
            # Empty batch is an error
            return JsonRpcBatchResponse(responses=[
                create_error_response(None, JsonRpcErrorCode.INVALID_REQUEST)
            ])

        # Process all requests concurrently
        tasks = []
        for request_data in batch_data:
            task = asyncio.create_task(self._process_single_request(request_data))
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None responses (notifications) and handle exceptions
        responses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Batch request failed", error=str(result))
                responses.append(
                    create_error_response(None, JsonRpcErrorCode.INTERNAL_ERROR)
                )
            elif result is not None:
                responses.append(result)

        return JsonRpcBatchResponse(responses=responses)

    async def _process_single_request(self, request_data: Dict[str, Any]) -> Optional[JsonRpcResponse]:
        """Process a single JSON-RPC request."""
        request_id = request_data.get('id')

        try:
            # Validate request structure
            request = JsonRpcRequest(**request_data)

            # Log request
            logger.info(
                "Processing JSON-RPC request",
                method=request.method,
                request_id=request_id,
                is_notification=request.is_notification,
                trace_id=request.a2a_context.trace_id if request.a2a_context else None
            )

            # Handle notifications (no response expected)
            if request.is_notification:
                await self._execute_method(request)
                return None

            # Execute method and return response
            result = await self._execute_method(request)
            return create_success_response(request_id, result)

        except ValidationError as e:
            logger.error("Request validation failed", error=str(e), request_id=request_id)
            return create_error_response(
                request_id,
                JsonRpcErrorCode.INVALID_REQUEST,
                data={"validation_errors": e.errors()}
            )

        except Exception as e:
            logger.error("Request processing failed", error=str(e), request_id=request_id, exc_info=True)
            return create_error_response(
                request_id,
                JsonRpcErrorCode.INTERNAL_ERROR,
                data={"details": str(e)}
            )

    async def _execute_method(self, request: JsonRpcRequest) -> Any:
        """
        Execute a JSON-RPC method.

        Args:
            request: Validated JSON-RPC request

        Returns:
            Method result

        Raises:
            Exception: If method execution fails
        """
        method_name = request.method

        # Check if method is registered
        if method_name not in self.methods:
            raise Exception(f"Method '{method_name}' not found")

        handler = self.methods[method_name]

        try:
            # Apply middleware stack
            for middleware in self._middleware_stack:
                await middleware(request)

            # Execute the method handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(request)
            else:
                result = handler(request)

            return result

        except Exception as e:
            logger.error(
                "Method execution failed",
                method=method_name,
                error=str(e),
                request_id=request.id,
                exc_info=True
            )
            raise


# Global JSON-RPC processor instance
jsonrpc_processor = JsonRpcProcessor()


def register_jsonrpc_method(name: str):
    """
    Decorator to register a JSON-RPC method handler.

    Usage:
        @register_jsonrpc_method("agent.register")
        async def handle_agent_register(request: JsonRpcRequest) -> Dict[str, Any]:
            # Handle the request
            return {"status": "registered"}
    """
    def decorator(func: JsonRpcHandler):
        jsonrpc_processor.register_method(name, func)
        return func
    return decorator


# Built-in methods
@register_jsonrpc_method("rpc.ping")
async def handle_ping(request: JsonRpcRequest) -> Dict[str, str]:
    """Built-in ping method for testing connectivity."""
    return {
        "pong": True,
        "timestamp": datetime.utcnow().isoformat(),
        "trace_id": request.a2a_context.trace_id if request.a2a_context else None
    }


@register_jsonrpc_method("rpc.methods")
async def handle_list_methods(request: JsonRpcRequest) -> Dict[str, List[str]]:
    """List all registered JSON-RPC methods."""
    return {
        "methods": list(jsonrpc_processor.methods.keys())
    }


@register_jsonrpc_method("rpc.version")
async def handle_version(request: JsonRpcRequest) -> Dict[str, str]:
    """Return A2A protocol and JSON-RPC version information."""
    from agentcore import __version__
    from agentcore.a2a_protocol.config import settings

    return {
        "agentcore_version": __version__,
        "jsonrpc_version": "2.0",
        "a2a_protocol_version": settings.A2A_PROTOCOL_VERSION,
    }