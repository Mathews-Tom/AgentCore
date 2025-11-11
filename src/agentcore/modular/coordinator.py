"""
Module Coordinator for Modular Agent Core

Manages module-to-module communication using JSON-RPC 2.0, handles message
routing based on module capabilities, and maintains execution context.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.jsonrpc import (
    A2AContext,
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    create_error_response,
    create_success_response,
)
from agentcore.modular.models import ModuleType

logger = structlog.get_logger()


# ============================================================================
# Coordination Models
# ============================================================================


class ModuleCapability(BaseModel):
    """Capability declaration for a module."""

    module_type: ModuleType = Field(..., description="Type of module")
    module_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique module instance ID"
    )
    methods: list[str] = Field(
        default_factory=list, description="JSON-RPC methods this module exposes"
    )
    max_concurrent: int = Field(
        default=10, description="Maximum concurrent requests"
    )
    timeout_seconds: float = Field(
        default=30.0, description="Default timeout for requests"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional module metadata"
    )


class ModuleMessage(BaseModel):
    """Message sent between modules."""

    message_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique message ID"
    )
    from_module: str = Field(..., description="Source module ID")
    to_module: str | None = Field(None, description="Target module ID (None for broadcast)")
    request: JsonRpcRequest = Field(..., description="JSON-RPC request")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Message creation timestamp",
    )
    timeout_seconds: float = Field(
        default=30.0, description="Timeout for this message"
    )


class CoordinationContext(BaseModel):
    """Context maintained during module coordination."""

    execution_id: str = Field(..., description="Execution identifier")
    plan_id: str | None = Field(None, description="Plan identifier")
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Trace ID for distributed tracing"
    )
    session_id: str | None = Field(None, description="Session identifier")
    iteration: int = Field(default=0, description="Current iteration number")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )


# ============================================================================
# Module Coordinator
# ============================================================================


class ModuleCoordinator:
    """
    Coordinates communication between Planner, Executor, Verifier, and Generator modules.

    Provides:
    - Module registration and discovery
    - JSON-RPC message routing based on capabilities
    - Execution context management
    - Error handling and timeout management
    - Async message passing between modules
    """

    def __init__(self) -> None:
        """Initialize the module coordinator."""
        self._modules: dict[str, ModuleCapability] = {}
        self._method_registry: dict[str, str] = {}  # method_name -> module_id
        self._pending_requests: dict[str, asyncio.Future[JsonRpcResponse]] = {}
        self._context: CoordinationContext | None = None
        logger.info("ModuleCoordinator initialized")

    # ========================================================================
    # Module Registration & Discovery
    # ========================================================================

    def register_module(self, capability: ModuleCapability) -> None:
        """
        Register a module with its capabilities.

        Args:
            capability: Module capability declaration

        Raises:
            ValueError: If module_id already registered
        """
        module_id = capability.module_id

        if module_id in self._modules:
            raise ValueError(f"Module {module_id} already registered")

        # Register module
        self._modules[module_id] = capability

        # Register method mappings
        for method in capability.methods:
            if method in self._method_registry:
                logger.warning(
                    "Method already registered, overriding",
                    method=method,
                    old_module=self._method_registry[method],
                    new_module=module_id,
                )
            self._method_registry[method] = module_id

        logger.info(
            "Module registered",
            module_id=module_id,
            module_type=capability.module_type,
            methods=capability.methods,
        )

    def unregister_module(self, module_id: str) -> None:
        """
        Unregister a module.

        Args:
            module_id: Module identifier

        Raises:
            ValueError: If module not registered
        """
        if module_id not in self._modules:
            raise ValueError(f"Module {module_id} not registered")

        capability = self._modules[module_id]

        # Unregister method mappings
        for method in capability.methods:
            if self._method_registry.get(method) == module_id:
                del self._method_registry[method]

        # Unregister module
        del self._modules[module_id]

        logger.info(
            "Module unregistered",
            module_id=module_id,
            module_type=capability.module_type,
        )

    def discover_modules(
        self, module_type: ModuleType | None = None
    ) -> list[ModuleCapability]:
        """
        Discover registered modules.

        Args:
            module_type: Filter by module type (None for all modules)

        Returns:
            List of module capabilities
        """
        if module_type is None:
            return list(self._modules.values())

        return [
            cap for cap in self._modules.values() if cap.module_type == module_type
        ]

    def find_module_for_method(self, method: str) -> str | None:
        """
        Find module that handles a specific method.

        Args:
            method: JSON-RPC method name

        Returns:
            Module ID or None if not found
        """
        return self._method_registry.get(method)

    # ========================================================================
    # Context Management
    # ========================================================================

    def set_context(self, context: CoordinationContext) -> None:
        """
        Set the coordination context.

        Args:
            context: Coordination context
        """
        self._context = context
        logger.debug(
            "Coordination context set",
            execution_id=context.execution_id,
            trace_id=context.trace_id,
        )

    def get_context(self) -> CoordinationContext | None:
        """
        Get the current coordination context.

        Returns:
            Current context or None
        """
        return self._context

    def clear_context(self) -> None:
        """Clear the coordination context."""
        self._context = None
        logger.debug("Coordination context cleared")

    def _build_a2a_context(self, from_module: str, to_module: str | None) -> A2AContext:
        """
        Build A2A context for a request.

        Args:
            from_module: Source module ID
            to_module: Target module ID

        Returns:
            A2A context
        """
        if self._context:
            return A2AContext(
                source_agent=from_module,
                target_agent=to_module,
                trace_id=self._context.trace_id,
                session_id=self._context.session_id,
                conversation_id=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return A2AContext(
            source_agent=from_module,
            target_agent=to_module,
            trace_id=str(uuid4()),
            session_id=None,
            conversation_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ========================================================================
    # Message Routing & Delivery
    # ========================================================================

    async def send_request(
        self,
        from_module: str,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        to_module: str | None = None,
        timeout: float | None = None,
    ) -> JsonRpcResponse:
        """
        Send a JSON-RPC request to another module.

        Args:
            from_module: Source module ID
            method: JSON-RPC method name
            params: Method parameters
            to_module: Target module ID (None for auto-discovery)
            timeout: Request timeout in seconds

        Returns:
            JSON-RPC response

        Raises:
            ValueError: If target module not found
            TimeoutError: If request times out
        """
        # Discover target module if not specified
        if to_module is None:
            to_module = self.find_module_for_method(method)
            if to_module is None:
                logger.error(
                    "No module found for method",
                    method=method,
                    from_module=from_module,
                )
                return create_error_response(
                    request_id=str(uuid4()),
                    error_code=JsonRpcErrorCode.METHOD_NOT_FOUND,
                    message=f"No module registered for method: {method}",
                )

        # Get module capability for timeout
        module_cap = self._modules.get(to_module)
        if module_cap is None:
            logger.error(
                "Target module not registered",
                to_module=to_module,
                method=method,
            )
            return create_error_response(
                request_id=str(uuid4()),
                error_code=JsonRpcErrorCode.INTERNAL_ERROR,
                message=f"Target module not registered: {to_module}",
            )

        # Determine timeout
        if timeout is None:
            timeout = module_cap.timeout_seconds

        # Build request
        request_id = str(uuid4())
        a2a_context = self._build_a2a_context(from_module, to_module)

        request = JsonRpcRequest(
            method=method,
            params=params,
            id=request_id,
            a2a_context=a2a_context,
        )

        message = ModuleMessage(
            from_module=from_module,
            to_module=to_module,
            request=request,
            timeout_seconds=timeout,
        )

        # Create future for response
        future: asyncio.Future[JsonRpcResponse] = asyncio.Future()
        self._pending_requests[message.message_id] = future

        logger.info(
            "Sending request to module",
            from_module=from_module,
            to_module=to_module,
            method=method,
            request_id=request_id,
            message_id=message.message_id,
        )

        try:
            # Simulate async message delivery (in real implementation, this would route to actual module)
            # For now, we'll timeout waiting for a response that won't come
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            logger.error(
                "Request timeout",
                message_id=message.message_id,
                method=method,
                timeout=timeout,
            )
            return create_error_response(
                request_id=request_id,
                error_code=JsonRpcErrorCode.INTERNAL_ERROR,
                message=f"Request timeout after {timeout} seconds",
                data={"method": method, "timeout": timeout},
            )

        finally:
            # Clean up pending request
            self._pending_requests.pop(message.message_id, None)

    async def send_notification(
        self,
        from_module: str,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        to_module: str | None = None,
    ) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            from_module: Source module ID
            method: JSON-RPC method name
            params: Method parameters
            to_module: Target module ID (None for broadcast)
        """
        # Build notification request (id=None)
        a2a_context = self._build_a2a_context(from_module, to_module)

        request = JsonRpcRequest(
            method=method,
            params=params,
            id=None,  # Notification
            a2a_context=a2a_context,
        )

        message = ModuleMessage(
            from_module=from_module,
            to_module=to_module,
            request=request,
        )

        logger.info(
            "Sending notification",
            from_module=from_module,
            to_module=to_module or "broadcast",
            method=method,
            message_id=message.message_id,
        )

        # In real implementation, this would route the notification to target module(s)
        # For now, we just log it

    def receive_response(self, message_id: str, response: JsonRpcResponse) -> None:
        """
        Receive a response for a pending request.

        Args:
            message_id: Message ID
            response: JSON-RPC response
        """
        future = self._pending_requests.get(message_id)
        if future and not future.done():
            future.set_result(response)
            logger.debug(
                "Response delivered",
                message_id=message_id,
                request_id=response.id,
            )
        else:
            logger.warning(
                "No pending request for response",
                message_id=message_id,
            )

    # ========================================================================
    # Error Handling
    # ========================================================================

    def handle_error(
        self,
        message_id: str,
        error: Exception,
        request_id: str | int | None = None,
    ) -> None:
        """
        Handle an error during message processing.

        Args:
            message_id: Message ID
            error: Exception that occurred
            request_id: JSON-RPC request ID
        """
        logger.error(
            "Error during message processing",
            message_id=message_id,
            error=str(error),
            error_type=type(error).__name__,
        )

        future = self._pending_requests.get(message_id)
        if future and not future.done():
            # Create error response
            error_response = create_error_response(
                request_id=request_id,
                error_code=JsonRpcErrorCode.INTERNAL_ERROR,
                message=str(error),
                data={"error_type": type(error).__name__},
            )
            future.set_result(error_response)

    # ========================================================================
    # Status & Monitoring
    # ========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get coordinator status.

        Returns:
            Status dictionary
        """
        return {
            "registered_modules": len(self._modules),
            "registered_methods": len(self._method_registry),
            "pending_requests": len(self._pending_requests),
            "has_context": self._context is not None,
            "modules": [
                {
                    "module_id": module_id,
                    "module_type": cap.module_type,
                    "methods": cap.methods,
                    "max_concurrent": cap.max_concurrent,
                }
                for module_id, cap in self._modules.items()
            ],
        }
