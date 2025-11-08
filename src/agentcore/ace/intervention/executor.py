"""
Intervention Execution (COMPASS ACE-2 - ACE-018)

Execution component that sends intervention commands to the Agent Runtime.
Receives InterventionDecision outputs and executes them via the Agent Runtime API.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import structlog

from agentcore.ace.models.ace_models import (
    ExecutionStatus,
    InterventionDecision,
    InterventionRecord,
    InterventionType,
    TriggerType,
)

logger = structlog.get_logger()

# Default Agent Runtime configuration
DEFAULT_RUNTIME_URL = "http://localhost:8001"
DEFAULT_TIMEOUT = 30.0  # seconds


class AgentRuntimeClient:
    """
    Client for Agent Runtime intervention commands.

    Simple HTTP client for sending intervention commands to the Agent Runtime.
    Will be enhanced in ACE-019 with advanced routing and resilience.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_RUNTIME_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize Agent Runtime client.

        Args:
            base_url: Base URL for Agent Runtime API
            timeout: Request timeout in seconds

        Raises:
            ValueError: If parameters are invalid
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info(
            "AgentRuntimeClient initialized",
            base_url=base_url,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def send_intervention(
        self,
        agent_id: str,
        task_id: UUID,
        intervention_type: InterventionType,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send intervention command to Agent Runtime.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            intervention_type: Type of intervention to execute
            context: Intervention context data

        Returns:
            Response dict with:
                - status: "success" or "failed"
                - duration_ms: Execution duration
                - message: Response message

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is invalid
        """
        start_time = time.perf_counter()

        logger.info(
            "Sending intervention to runtime",
            agent_id=agent_id,
            task_id=str(task_id),
            intervention_type=intervention_type.value,
        )

        # Build request payload
        payload = {
            "agent_id": agent_id,
            "task_id": str(task_id),
            "intervention_type": intervention_type.value,
            "context": context,
        }

        # Send POST request to runtime
        endpoint = f"{self.base_url}/api/v1/runtime/interventions"
        try:
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Validate response structure
            if "status" not in data:
                raise ValueError("Response missing 'status' field")

            # Add duration to response if not present (ensure at least 1ms)
            if "duration_ms" not in data:
                duration_ms = max(1, int((time.perf_counter() - start_time) * 1000))
                data["duration_ms"] = duration_ms

            logger.info(
                "Intervention sent successfully",
                agent_id=agent_id,
                status=data["status"],
                duration_ms=data["duration_ms"],
            )

            return data

        except httpx.HTTPError as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "Failed to send intervention to runtime",
                agent_id=agent_id,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise


class InterventionExecutor:
    """
    Intervention executor for COMPASS Meta-Thinker (COMPASS ACE-2 - ACE-018).

    Features:
    - Non-blocking execution (async/await)
    - Agent Runtime integration
    - Execution duration tracking
    - Status tracking (PENDING → IN_PROGRESS → COMPLETED/FAILED)
    - Graceful error handling
    - Structured logging

    Performance target: <500ms execution (p95)
    """

    def __init__(
        self,
        runtime_client: AgentRuntimeClient | None = None,
    ) -> None:
        """
        Initialize InterventionExecutor.

        Args:
            runtime_client: Agent Runtime client (if None, creates default)
        """
        if runtime_client is None:
            self.runtime_client = AgentRuntimeClient()
        else:
            self.runtime_client = runtime_client

        logger.info("InterventionExecutor initialized")

    async def execute_intervention(
        self,
        agent_id: str,
        task_id: UUID,
        decision: InterventionDecision,
        trigger_type: TriggerType,
        trigger_signals: list[str],
    ) -> InterventionRecord:
        """
        Execute intervention based on decision.

        This is the main entry point for intervention execution. Creates an
        InterventionRecord, executes the intervention via Agent Runtime, and
        tracks execution status and duration.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            decision: Intervention decision from DecisionMaker
            trigger_type: Type of trigger that initiated intervention
            trigger_signals: Specific trigger signals

        Returns:
            InterventionRecord with execution details

        Raises:
            ValueError: If parameters are invalid
            httpx.HTTPError: If runtime request fails
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not trigger_signals:
            raise ValueError("trigger_signals cannot be empty")

        execution_start = time.perf_counter()

        logger.info(
            "Executing intervention",
            agent_id=agent_id,
            task_id=str(task_id),
            intervention_type=decision.intervention_type.value,
            trigger_type=trigger_type.value,
        )

        # Create intervention record with PENDING status
        record = InterventionRecord(
            task_id=task_id,
            agent_id=agent_id,
            trigger_type=trigger_type,
            trigger_signals=trigger_signals,
            intervention_type=decision.intervention_type,
            intervention_rationale=decision.rationale,
            decision_confidence=decision.confidence,
            execution_status=ExecutionStatus.PENDING,
        )

        # Update to IN_PROGRESS
        # Note: In production, this would persist to database
        # For ACE-018, we're focusing on execution logic

        try:
            # Build intervention context from decision
            context = {
                "rationale": decision.rationale,
                "expected_impact": decision.expected_impact,
                "confidence": decision.confidence,
                "alternative_interventions": decision.alternative_interventions,
                "metadata": decision.metadata,
            }

            # Send intervention to Agent Runtime
            response = await self.runtime_client.send_intervention(
                agent_id=agent_id,
                task_id=task_id,
                intervention_type=decision.intervention_type,
                context=context,
            )

            # Calculate execution duration (round up to ensure at least 1ms)
            execution_duration_ms = max(1, int((time.perf_counter() - execution_start) * 1000))

            # Determine execution status from response
            if response["status"] == "success":
                execution_status = ExecutionStatus.SUCCESS
                execution_error = None
            elif response["status"] == "partial":
                execution_status = ExecutionStatus.PARTIAL
                execution_error = response.get("message", "Partial execution")
            else:
                execution_status = ExecutionStatus.FAILURE
                execution_error = response.get("message", "Execution failed")

            # Update record with execution results
            record.execution_status = execution_status
            record.execution_duration_ms = execution_duration_ms
            record.execution_error = execution_error
            record.executed_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)

            logger.info(
                "Intervention executed",
                agent_id=agent_id,
                intervention_type=decision.intervention_type.value,
                status=execution_status.value,
                duration_ms=execution_duration_ms,
            )

            return record

        except httpx.HTTPError as e:
            # Calculate execution duration even on failure (round up to ensure at least 1ms)
            execution_duration_ms = max(1, int((time.perf_counter() - execution_start) * 1000))

            # Update record with failure details
            record.execution_status = ExecutionStatus.FAILURE
            record.execution_duration_ms = execution_duration_ms
            record.execution_error = f"Runtime communication error: {str(e)}"
            record.executed_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)

            logger.error(
                "Intervention execution failed",
                agent_id=agent_id,
                intervention_type=decision.intervention_type.value,
                error=str(e),
                duration_ms=execution_duration_ms,
            )

            # Return the record with failure status
            # In production, caller can decide whether to raise or handle gracefully
            return record

        except Exception as e:
            # Handle unexpected errors (round up to ensure at least 1ms)
            execution_duration_ms = max(1, int((time.perf_counter() - execution_start) * 1000))

            record.execution_status = ExecutionStatus.FAILURE
            record.execution_duration_ms = execution_duration_ms
            record.execution_error = f"Unexpected error: {str(e)}"
            record.executed_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)

            logger.error(
                "Unexpected error during intervention execution",
                agent_id=agent_id,
                intervention_type=decision.intervention_type.value,
                error=str(e),
                duration_ms=execution_duration_ms,
            )

            # Re-raise unexpected errors to fail fast
            raise

    async def close(self) -> None:
        """Close runtime client."""
        await self.runtime_client.close()
