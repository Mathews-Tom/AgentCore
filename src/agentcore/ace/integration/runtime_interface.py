"""
Runtime Interface for COMPASS Meta-Thinker (COMPASS ACE-2 - ACE-019)

Handles intervention commands from ACE and manages execution via Agent Runtime.
Provides bidirectional integration for strategic intervention and outcome feedback.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, ValidationError

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.ace.models.ace_models import (
    InterventionState,
    InterventionType,
    RuntimeIntervention,
)

logger = structlog.get_logger()


# Request/Response Models


class ExecuteInterventionRequest(BaseModel):
    """Request to execute intervention via runtime."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    task_id: str = Field(..., description="Task identifier (UUID)")
    intervention_type: str = Field(..., description="Intervention type")
    context: dict[str, Any] = Field(default_factory=dict, description="Intervention context")


class ExecuteInterventionResponse(BaseModel):
    """Response from intervention execution."""

    status: str = Field(..., description="Execution status (success/failed/partial)")
    duration_ms: int = Field(..., description="Execution duration in milliseconds")
    message: str = Field(..., description="Response message")
    outcome: dict[str, Any] = Field(default_factory=dict, description="Execution outcome")


# RuntimeInterface


class RuntimeInterface:
    """
    Agent Runtime interface for intervention command support (COMPASS ACE-2 - ACE-019).

    Manages intervention state and execution, providing the server-side handler
    for intervention commands from the COMPASS Meta-Thinker (ACE).

    Features:
    - Intervention state tracking (PENDING → IN_PROGRESS → COMPLETED/FAILED)
    - Non-blocking async execution
    - Type-specific intervention handlers
    - Outcome reporting back to ACE
    - Structured logging with structlog

    Performance target: <500ms execution (p95)
    """

    def __init__(self) -> None:
        """Initialize RuntimeInterface with in-memory state store."""
        # In-memory intervention store (dict-based for MVP)
        # Key: intervention_id -> RuntimeIntervention
        self._interventions: dict[UUID, RuntimeIntervention] = {}

        logger.info("RuntimeInterface initialized")

    async def handle_intervention(
        self,
        agent_id: str,
        task_id: UUID,
        intervention_type: InterventionType,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle intervention command from ACE.

        This is the main entry point for intervention execution. Creates a
        RuntimeIntervention, executes the intervention based on type, tracks
        state transitions, and returns outcome.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            intervention_type: Type of intervention to execute
            context: Intervention context data

        Returns:
            Outcome dict with:
                - status: "success", "failed", or "partial"
                - duration_ms: Execution duration in milliseconds
                - message: Response message
                - outcome: Type-specific outcome data

        Raises:
            ValueError: If parameters are invalid
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id cannot be empty")

        start_time = time.perf_counter()

        logger.info(
            "Handling intervention",
            agent_id=agent_id,
            task_id=str(task_id),
            intervention_type=intervention_type.value,
        )

        # Create intervention record with PENDING state
        intervention = RuntimeIntervention(
            agent_id=agent_id,
            task_id=task_id,
            intervention_type=intervention_type,
            context=context,
            state=InterventionState.PENDING,
        )

        # Store intervention
        self._interventions[intervention.intervention_id] = intervention

        # Update to IN_PROGRESS
        intervention.state = InterventionState.IN_PROGRESS
        intervention.updated_at = datetime.now(UTC)

        try:
            # Execute intervention based on type
            outcome = await self._execute_intervention(
                agent_id=agent_id,
                task_id=task_id,
                intervention_type=intervention_type,
                context=context,
            )

            # Update to COMPLETED
            intervention.state = InterventionState.COMPLETED
            intervention.outcome = outcome
            intervention.updated_at = datetime.now(UTC)

            # Calculate duration (ensure at least 1ms)
            duration_ms = max(1, int((time.perf_counter() - start_time) * 1000))

            logger.info(
                "Intervention completed successfully",
                agent_id=agent_id,
                intervention_type=intervention_type.value,
                duration_ms=duration_ms,
            )

            return {
                "status": "success",
                "duration_ms": duration_ms,
                "message": f"{intervention_type.value} executed successfully",
                "outcome": outcome,
            }

        except Exception as e:
            # Update to FAILED
            intervention.state = InterventionState.FAILED
            intervention.outcome = {"error": str(e)}
            intervention.updated_at = datetime.now(UTC)

            # Calculate duration (ensure at least 1ms)
            duration_ms = max(1, int((time.perf_counter() - start_time) * 1000))

            logger.error(
                "Intervention execution failed",
                agent_id=agent_id,
                intervention_type=intervention_type.value,
                error=str(e),
                duration_ms=duration_ms,
            )

            return {
                "status": "failed",
                "duration_ms": duration_ms,
                "message": f"Intervention failed: {str(e)}",
                "outcome": {"error": str(e)},
            }

    async def _execute_intervention(
        self,
        agent_id: str,
        task_id: UUID,
        intervention_type: InterventionType,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute intervention based on type.

        Dispatches to type-specific handler methods. Each handler implements
        a realistic placeholder execution for MVP.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            intervention_type: Type of intervention
            context: Intervention context

        Returns:
            Type-specific outcome dict

        Raises:
            ValueError: If intervention type is unknown
        """
        if intervention_type == InterventionType.CONTEXT_REFRESH:
            return await self._execute_context_refresh(agent_id, task_id, context)
        elif intervention_type == InterventionType.REPLAN:
            return await self._execute_replan(agent_id, task_id, context)
        elif intervention_type == InterventionType.REFLECT:
            return await self._execute_reflect(agent_id, task_id, context)
        elif intervention_type == InterventionType.CAPABILITY_SWITCH:
            return await self._execute_capability_switch(agent_id, task_id, context)
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

    async def _execute_context_refresh(
        self,
        agent_id: str,
        task_id: UUID,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute CONTEXT_REFRESH intervention.

        Clears stale context from memory, fetches fresh relevant context,
        and updates agent's working memory.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            context: Intervention context

        Returns:
            Outcome with refreshed_facts and cleared_items counts
        """
        logger.info(
            "Executing context refresh",
            agent_id=agent_id,
            task_id=str(task_id),
        )

        # Placeholder implementation for MVP
        # In production, this would:
        # 1. Clear stale context from agent memory
        # 2. Fetch fresh context based on current task state
        # 3. Update agent's working memory

        refreshed_facts = 42  # Realistic placeholder value
        cleared_items = 15  # Realistic placeholder value

        return {
            "refreshed_facts": refreshed_facts,
            "cleared_items": cleared_items,
        }

    async def _execute_replan(
        self,
        agent_id: str,
        task_id: UUID,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute REPLAN intervention.

        Snapshots current task state, triggers replanning algorithm,
        and updates task plan.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            context: Intervention context

        Returns:
            Outcome with new_plan_steps and changes_made counts
        """
        logger.info(
            "Executing replan",
            agent_id=agent_id,
            task_id=str(task_id),
        )

        # Placeholder implementation for MVP
        # In production, this would:
        # 1. Snapshot current task state
        # 2. Trigger replanning algorithm with current context
        # 3. Update task plan with new steps
        # 4. Notify agent of plan changes

        new_plan_steps = 8  # Realistic placeholder value
        changes_made = 5  # Realistic placeholder value

        return {
            "new_plan_steps": new_plan_steps,
            "changes_made": changes_made,
        }

    async def _execute_reflect(
        self,
        agent_id: str,
        task_id: UUID,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute REFLECT intervention.

        Collects error history, analyzes failure patterns, and generates
        improvement recommendations.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            context: Intervention context

        Returns:
            Outcome with errors_analyzed and insights
        """
        logger.info(
            "Executing reflect",
            agent_id=agent_id,
            task_id=str(task_id),
        )

        # Placeholder implementation for MVP
        # In production, this would:
        # 1. Collect error history from task execution
        # 2. Analyze failure patterns
        # 3. Generate improvement recommendations
        # 4. Update agent's self-improvement log

        errors_analyzed = 12  # Realistic placeholder value
        insights = [
            "API timeout pattern detected in 3 consecutive calls",
            "Memory retrieval returning stale results in 40% of queries",
            "Task breakdown insufficient for complex multi-step operations",
        ]

        return {
            "errors_analyzed": errors_analyzed,
            "insights": insights,
        }

    async def _execute_capability_switch(
        self,
        agent_id: str,
        task_id: UUID,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute CAPABILITY_SWITCH intervention.

        Evaluates capability fitness, selects optimal capability set,
        and updates agent capabilities.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            context: Intervention context

        Returns:
            Outcome with capabilities_changed and new_capabilities
        """
        logger.info(
            "Executing capability switch",
            agent_id=agent_id,
            task_id=str(task_id),
        )

        # Placeholder implementation for MVP
        # In production, this would:
        # 1. Evaluate current capability fitness for task
        # 2. Select optimal capability set based on task requirements
        # 3. Update agent's active capabilities
        # 4. Notify agent of capability changes

        capabilities_changed = 3  # Realistic placeholder value
        new_capabilities = [
            "data_transformation",
            "advanced_search",
            "parallel_execution",
        ]

        return {
            "capabilities_changed": capabilities_changed,
            "new_capabilities": new_capabilities,
        }

    def get_intervention_state(self, intervention_id: UUID) -> RuntimeIntervention | None:
        """
        Get intervention by ID.

        Args:
            intervention_id: Intervention identifier

        Returns:
            RuntimeIntervention if found, None otherwise
        """
        return self._interventions.get(intervention_id)

    def list_interventions(
        self,
        agent_id: str | None = None,
        task_id: UUID | None = None,
        state: InterventionState | None = None,
    ) -> list[RuntimeIntervention]:
        """
        List interventions with optional filtering.

        Args:
            agent_id: Filter by agent ID (optional)
            task_id: Filter by task ID (optional)
            state: Filter by state (optional)

        Returns:
            List of matching RuntimeIntervention objects
        """
        interventions = list(self._interventions.values())

        if agent_id:
            interventions = [i for i in interventions if i.agent_id == agent_id]

        if task_id:
            interventions = [i for i in interventions if i.task_id == task_id]

        if state:
            interventions = [i for i in interventions if i.state == state]

        return interventions


# Global instance
runtime_interface = RuntimeInterface()


# JSON-RPC Method Handler


@register_jsonrpc_method("runtime.execute_intervention")
async def execute_intervention_handler(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute intervention via Agent Runtime.

    Method: runtime.execute_intervention
    Params:
        - agent_id: string
        - task_id: string (UUID)
        - intervention_type: string (context_refresh, replan, reflect, capability_switch)
        - context: object (intervention context data)

    Returns:
        - status: string (success/failed/partial)
        - duration_ms: integer
        - message: string
        - outcome: object (type-specific outcome data)
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: agent_id, task_id, intervention_type, context"
            )

        # Validate request params
        req = ExecuteInterventionRequest(**request.params)

        # Convert task_id to UUID
        task_id = UUID(req.task_id)

        # Parse intervention type
        try:
            intervention_type = InterventionType(req.intervention_type)
        except ValueError:
            raise ValueError(
                f"Invalid intervention_type: {req.intervention_type}. "
                f"Must be one of: {[t.value for t in InterventionType]}"
            )

        # Execute via RuntimeInterface
        result = await runtime_interface.handle_intervention(
            agent_id=req.agent_id,
            task_id=task_id,
            intervention_type=intervention_type,
            context=req.context,
        )

        logger.info(
            "Intervention executed via JSON-RPC",
            agent_id=req.agent_id,
            task_id=str(task_id),
            intervention_type=intervention_type.value,
            status=result["status"],
            method="runtime.execute_intervention",
        )

        return result

    except ValidationError as e:
        logger.error("Intervention execution validation failed", error=str(e))
        raise ValueError(f"Request validation failed: {e}")
    except ValueError as e:
        logger.error("Intervention execution failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Intervention execution failed", error=str(e))
        raise
