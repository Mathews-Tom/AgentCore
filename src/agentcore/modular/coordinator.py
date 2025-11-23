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
from agentcore.modular.models import ModuleType, ModuleTransition
from agentcore.modular.interfaces import (
    PlannerQuery,
    PlanRefinement,
    VerificationRequest,
    GenerationRequest,
    ExecutionResult,
    VerificationResult,
    GeneratedResponse,
)

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
    # Coordination Loop with Refinement
    # ========================================================================

    async def execute_with_refinement(
        self,
        query: str,
        planner: Any,
        executor: Any,
        verifier: Any,
        generator: Any,
        max_iterations: int = 5,
        timeout_seconds: float = 300.0,
        confidence_threshold: float = 0.7,
        output_format: str = "text",
        include_reasoning: bool = False,
    ) -> dict[str, Any]:
        """
        Execute full PEVG workflow with iterative refinement.

        Flow:
        1. Plan (iteration 1)
        2. Execute plan
        3. Verify results
        4. If verification fails AND iterations < max:
           - Refine plan with verifier feedback
           - Goto step 2 (execute refined plan)
        5. Generate final response

        Args:
            query: User query to solve
            planner: Planner module instance
            executor: Executor module instance
            verifier: Verifier module instance
            generator: Generator module instance
            max_iterations: Maximum refinement iterations (default: 5)
            timeout_seconds: Overall execution timeout (default: 300)
            confidence_threshold: Minimum confidence for success (default: 0.7)
            output_format: Output format for generator (default: "text")
            include_reasoning: Include reasoning trace (default: False)

        Returns:
            Dictionary with answer, execution_trace, reasoning, sources

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails after max iterations
        """
        import time

        start_time = time.time()
        iteration = 0
        plan = None
        execution_results: list[ExecutionResult] = []
        verification_result: VerificationResult | None = None
        transitions: list[ModuleTransition] = []
        modules_invoked: list[str] = []

        logger.info(
            "coordination_loop_started",
            query=query,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            trace_id=self._context.trace_id if self._context else None,
        )

        try:
            # Wrap entire coordination loop with timeout
            async def _execute_loop() -> dict[str, Any]:
                nonlocal iteration, plan, execution_results, verification_result

                # Iteration loop
                while iteration < max_iterations:
                    iteration += 1
                    iteration_start = time.time()

                    logger.info(
                        "iteration_started",
                        iteration=iteration,
                        max_iterations=max_iterations,
                    )

                    # =======================================================
                    # Step 1: Planning (or Refinement)
                    # =======================================================
                    if iteration == 1:
                        # Initial planning
                        logger.info("step_planning", iteration=iteration)
                        modules_invoked.append("planner")

                        # Emit transition event
                        transition = ModuleTransition(
                            plan_id=str(uuid4()),
                            iteration=iteration,
                            from_module=ModuleType.PLANNER,
                            to_module=ModuleType.EXECUTOR,
                            reason="initial_planning_complete",
                            trigger="query_received",
                        )
                        transitions.append(transition)

                        planner_query = PlannerQuery(
                            query=query,
                            context={},
                            constraints={"max_iterations": max_iterations},
                        )

                        plan = await planner.analyze_query(planner_query)
                        logger.info(
                            "planning_complete",
                            plan_id=plan.plan_id,
                            step_count=len(plan.steps),
                            iteration=iteration,
                        )

                    else:
                        # Plan refinement
                        logger.info(
                            "step_plan_refinement",
                            iteration=iteration,
                            feedback_available=verification_result is not None,
                        )
                        modules_invoked.append("planner")

                        # Extract feedback from verification result
                        feedback = "No specific feedback"
                        if verification_result:
                            if verification_result.feedback:
                                feedback = verification_result.feedback
                            elif verification_result.errors:
                                feedback = f"Errors: {', '.join(verification_result.errors)}"

                        # Emit transition event
                        transition = ModuleTransition(
                            plan_id=plan.plan_id if plan else str(uuid4()),
                            iteration=iteration,
                            from_module=ModuleType.PLANNER,
                            to_module=ModuleType.EXECUTOR,
                            reason="plan_refinement_complete",
                            trigger="verification_failed",
                            data={"feedback": feedback},
                        )
                        transitions.append(transition)

                        # Refine plan with feedback
                        refinement_request = PlanRefinement(
                            plan_id=plan.plan_id if plan else str(uuid4()),
                            feedback=feedback,
                            constraints={
                                "existing_plan": plan.model_dump() if plan else None,
                                "original_query": query,
                                "max_iterations": max_iterations,
                                "verification_errors": verification_result.errors if verification_result else [],
                            },
                        )

                        plan = await planner.refine_plan(refinement_request)
                        logger.info(
                            "plan_refined",
                            new_plan_id=plan.plan_id,
                            step_count=len(plan.steps),
                            iteration=iteration,
                        )

                    # Persist state after planning
                    if self._context:
                        self._context.plan_id = plan.plan_id
                        self._context.iteration = iteration

                    # =======================================================
                    # Step 2: Execution
                    # =======================================================
                    logger.info("step_execution", iteration=iteration, plan_id=plan.plan_id)
                    modules_invoked.append("executor")

                    # Emit transition event
                    transition = ModuleTransition(
                        plan_id=plan.plan_id,
                        iteration=iteration,
                        from_module=ModuleType.EXECUTOR,
                        to_module=ModuleType.VERIFIER,
                        reason="execution_complete",
                        trigger="plan_received",
                    )
                    transitions.append(transition)

                    execution_results = await executor.execute_plan(plan)
                    successful_steps = sum(1 for r in execution_results if r.success)
                    failed_steps = sum(1 for r in execution_results if not r.success)

                    logger.info(
                        "execution_complete",
                        plan_id=plan.plan_id,
                        total_steps=len(execution_results),
                        successful=successful_steps,
                        failed=failed_steps,
                        iteration=iteration,
                    )

                    # =======================================================
                    # Step 3: Verification
                    # =======================================================
                    logger.info("step_verification", iteration=iteration)
                    modules_invoked.append("verifier")

                    # Emit transition event
                    transition = ModuleTransition(
                        plan_id=plan.plan_id,
                        iteration=iteration,
                        from_module=ModuleType.VERIFIER,
                        to_module=ModuleType.GENERATOR,
                        reason="verification_complete",
                        trigger="execution_received",
                    )
                    transitions.append(transition)

                    verification_request = VerificationRequest(
                        results=execution_results,
                        expected_json_schema=None,
                        consistency_rules=["no_null_results"],
                    )

                    verification_result = await verifier.validate_results(verification_request)
                    logger.info(
                        "verification_complete",
                        valid=verification_result.valid,
                        confidence=verification_result.confidence,
                        errors_count=len(verification_result.errors),
                        iteration=iteration,
                    )

                    # Persist state after verification
                    if self._context:
                        self._context.metadata["last_verification"] = {
                            "valid": verification_result.valid,
                            "confidence": verification_result.confidence,
                            "iteration": iteration,
                        }

                    # =======================================================
                    # Step 4: Check if refinement needed
                    # =======================================================
                    # Success criteria: verification passed AND confidence >= threshold
                    verification_passed = (
                        verification_result.valid
                        and verification_result.confidence >= confidence_threshold
                    )

                    if verification_passed:
                        logger.info(
                            "verification_success_early_exit",
                            iteration=iteration,
                            confidence=verification_result.confidence,
                        )
                        break  # Exit refinement loop

                    # Check if we have more iterations
                    if iteration >= max_iterations:
                        logger.warning(
                            "max_iterations_reached",
                            iteration=iteration,
                            max_iterations=max_iterations,
                            last_confidence=verification_result.confidence,
                        )
                        break  # Exit with partial results

                    # Continue to next iteration for refinement
                    logger.info(
                        "refinement_needed",
                        iteration=iteration,
                        confidence=verification_result.confidence,
                        threshold=confidence_threshold,
                        errors=verification_result.errors,
                    )

                # End of iteration loop

                # =======================================================
                # Step 5: Generation (after verification success or max iterations)
                # =======================================================
                logger.info("step_generation", final_iteration=iteration)
                modules_invoked.append("generator")

                # Emit final transition event
                transition = ModuleTransition(
                    plan_id=plan.plan_id if plan else str(uuid4()),
                    iteration=iteration,
                    from_module=ModuleType.GENERATOR,
                    to_module=ModuleType.GENERATOR,  # Terminal state
                    reason="generation_complete",
                    trigger="verification_complete_or_max_iterations",
                )
                transitions.append(transition)

                generation_request = GenerationRequest(
                    verified_results=execution_results,
                    format=output_format,
                    include_reasoning=include_reasoning,
                    max_length=None,
                )

                generated_response = await generator.synthesize_response(generation_request)
                logger.info(
                    "generation_complete",
                    content_length=len(generated_response.content),
                    has_reasoning=generated_response.reasoning is not None,
                    sources_count=len(generated_response.sources),
                )

                # Build execution trace
                total_duration_ms = int((time.time() - start_time) * 1000)

                execution_trace = {
                    "plan_id": plan.plan_id if plan else "unknown",
                    "iterations": iteration,
                    "modules_invoked": modules_invoked,
                    "total_duration_ms": total_duration_ms,
                    "verification_passed": verification_result.valid if verification_result else False,
                    "step_count": len(execution_results),
                    "successful_steps": sum(1 for r in execution_results if r.success),
                    "failed_steps": sum(1 for r in execution_results if not r.success),
                    "confidence_score": verification_result.confidence if verification_result else 0.0,
                    "transitions": [t.model_dump() for t in transitions],
                    "refinement_history": [
                        {
                            "iteration": i + 1,
                            "module": modules_invoked[i] if i < len(modules_invoked) else "unknown"
                        }
                        for i in range(iteration)
                    ],
                }

                logger.info(
                    "coordination_loop_complete",
                    iterations=iteration,
                    duration_ms=total_duration_ms,
                    verification_passed=verification_result.valid if verification_result else False,
                )

                return {
                    "answer": generated_response.content,
                    "execution_trace": execution_trace,
                    "reasoning": generated_response.reasoning,
                    "sources": generated_response.sources,
                }

            # Execute loop with timeout
            result = await asyncio.wait_for(_execute_loop(), timeout=timeout_seconds)
            return result

        except asyncio.TimeoutError:
            logger.error(
                "coordination_loop_timeout",
                timeout_seconds=timeout_seconds,
                iterations_completed=iteration,
                duration_ms=int((time.time() - start_time) * 1000),
            )

            # Return partial results if available
            if execution_results:
                return {
                    "answer": f"Execution timed out after {timeout_seconds}s (partial results available)",
                    "execution_trace": {
                        "plan_id": plan.plan_id if plan else "unknown",
                        "iterations": iteration,
                        "modules_invoked": modules_invoked,
                        "total_duration_ms": int((time.time() - start_time) * 1000),
                        "verification_passed": False,
                        "step_count": len(execution_results),
                        "successful_steps": sum(1 for r in execution_results if r.success),
                        "failed_steps": sum(1 for r in execution_results if not r.success),
                        "confidence_score": verification_result.confidence if verification_result else 0.0,
                        "timeout": True,
                    },
                    "reasoning": None,
                    "sources": [],
                }

            raise

        except Exception as e:
            logger.error(
                "coordination_loop_error",
                error=str(e),
                error_type=type(e).__name__,
                iterations_completed=iteration,
                exc_info=True,
            )
            raise

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
