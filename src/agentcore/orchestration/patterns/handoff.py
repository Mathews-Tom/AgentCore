"""
Handoff Pattern Implementation

Sequential task handoff with context preservation, quality gates, and rollback capabilities.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.streams.models import OrchestrationEvent
from agentcore.orchestration.streams.producer import StreamProducer


class HandoffStatus(str, Enum):
    """Status of a handoff operation."""

    PENDING = "pending"
    VALIDATING = "validating"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ValidationResult(BaseModel):
    """Result of quality gate validation."""

    valid: bool = Field(description="Whether validation passed")
    gate_name: str = Field(description="Name of the quality gate")
    message: str | None = Field(default=None, description="Validation message or error")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional validation data"
    )


class HandoffContext(BaseModel):
    """
    Context transferred during handoff.

    Contains all task data, metadata, and history needed for continuation.
    """

    handoff_id: UUID = Field(
        default_factory=uuid4, description="Unique handoff identifier"
    )
    task_id: UUID = Field(description="Task being handed off")
    task_type: str = Field(description="Type of task")
    task_data: dict[str, Any] = Field(
        default_factory=dict, description="Current task data and state"
    )

    source_agent_id: str = Field(description="Agent initiating handoff")
    target_agent_id: str = Field(description="Agent receiving handoff")

    handoff_chain: list[str] = Field(
        default_factory=list, description="History of agents that handled this task"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Handoff creation time"
    )
    completed_at: datetime | None = Field(
        default=None, description="Handoff completion time"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional handoff metadata"
    )

    # Snapshot of state before handoff for rollback
    previous_state: dict[str, Any] | None = Field(
        default=None, description="State snapshot before handoff"
    )

    model_config = {"frozen": False}


class HandoffGate(ABC):
    """
    Abstract quality gate interface.

    Quality gates validate task readiness before handoff acceptance.
    """

    def __init__(self, gate_name: str) -> None:
        """
        Initialize quality gate.

        Args:
            gate_name: Name of this quality gate
        """
        self.gate_name = gate_name

    @abstractmethod
    async def validate(self, context: HandoffContext) -> ValidationResult:
        """
        Validate handoff context against quality criteria.

        Args:
            context: Handoff context to validate

        Returns:
            Validation result
        """
        pass


class InputValidationGate(HandoffGate):
    """
    Input validation gate.

    Validates that required input data is present and well-formed.
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        field_validators: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize input validation gate.

        Args:
            required_fields: List of required field names
            field_validators: Dictionary of field name to validator function
        """
        super().__init__("input_validation")
        self.required_fields = required_fields or []
        self.field_validators = field_validators or {}

    async def validate(self, context: HandoffContext) -> ValidationResult:
        """Validate input data is complete and valid."""
        task_data = context.task_data

        # Check required fields
        for field in self.required_fields:
            if field not in task_data:
                return ValidationResult(
                    valid=False,
                    gate_name=self.gate_name,
                    message=f"Missing required field: {field}",
                )

        # Run field validators
        for field, validator in self.field_validators.items():
            if field in task_data:
                try:
                    if not validator(task_data[field]):
                        return ValidationResult(
                            valid=False,
                            gate_name=self.gate_name,
                            message=f"Validation failed for field: {field}",
                        )
                except Exception as e:
                    return ValidationResult(
                        valid=False,
                        gate_name=self.gate_name,
                        message=f"Validator error for field {field}: {e!s}",
                    )

        return ValidationResult(
            valid=True,
            gate_name=self.gate_name,
            message="All input validations passed",
        )


class OutputValidationGate(HandoffGate):
    """
    Output validation gate.

    Validates task output before handoff to ensure quality standards.
    """

    def __init__(
        self,
        output_key: str = "output",
        min_completeness: float = 0.8,
    ) -> None:
        """
        Initialize output validation gate.

        Args:
            output_key: Key in task_data containing output
            min_completeness: Minimum completeness threshold (0.0-1.0)
        """
        super().__init__("output_validation")
        self.output_key = output_key
        self.min_completeness = min_completeness

    async def validate(self, context: HandoffContext) -> ValidationResult:
        """Validate output meets quality standards."""
        task_data = context.task_data

        if self.output_key not in task_data:
            return ValidationResult(
                valid=False,
                gate_name=self.gate_name,
                message=f"Missing output field: {self.output_key}",
            )

        output = task_data[self.output_key]

        # Calculate completeness (example heuristic)
        completeness = self._calculate_completeness(output)

        if completeness < self.min_completeness:
            return ValidationResult(
                valid=False,
                gate_name=self.gate_name,
                message=f"Output completeness {completeness:.2f} below threshold {self.min_completeness:.2f}",
                metadata={"completeness": completeness},
            )

        return ValidationResult(
            valid=True,
            gate_name=self.gate_name,
            message="Output validation passed",
            metadata={"completeness": completeness},
        )

    def _calculate_completeness(self, output: Any) -> float:
        """
        Calculate output completeness heuristic.

        Args:
            output: Output data

        Returns:
            Completeness score (0.0-1.0)
        """
        if output is None or output == "":
            return 0.0

        if isinstance(output, dict):
            # Count non-null values
            total_fields = len(output)
            filled_fields = sum(1 for v in output.values() if v not in (None, "", []))
            return filled_fields / total_fields if total_fields > 0 else 0.0

        if isinstance(output, list):
            return 1.0 if len(output) > 0 else 0.0

        # For other types, assume complete if present
        return 1.0


class CapabilityGate(HandoffGate):
    """
    Capability validation gate.

    Validates target agent has required capabilities.
    """

    def __init__(self, required_capabilities: list[str]) -> None:
        """
        Initialize capability gate.

        Args:
            required_capabilities: List of required capability names
        """
        super().__init__("capability_validation")
        self.required_capabilities = required_capabilities

    async def validate(self, context: HandoffContext) -> ValidationResult:
        """Validate target agent capabilities."""
        # This would typically check against a capability registry
        # For now, we check metadata
        target_capabilities = context.metadata.get("target_capabilities", [])

        missing_capabilities = [
            cap for cap in self.required_capabilities if cap not in target_capabilities
        ]

        if missing_capabilities:
            return ValidationResult(
                valid=False,
                gate_name=self.gate_name,
                message=f"Target agent missing capabilities: {missing_capabilities}",
                metadata={"missing": missing_capabilities},
            )

        return ValidationResult(
            valid=True,
            gate_name=self.gate_name,
            message="All required capabilities present",
        )


class HandoffRecord(BaseModel):
    """Record of a handoff operation."""

    handoff_id: UUID = Field(description="Unique handoff identifier")
    context: HandoffContext = Field(description="Handoff context")
    status: HandoffStatus = Field(description="Current handoff status")
    validation_results: list[ValidationResult] = Field(
        default_factory=list, description="Results from quality gates"
    )
    initiated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Initiation timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Completion timestamp"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )

    model_config = {"frozen": False}


class HandoffConfig(BaseModel):
    """Configuration for handoff pattern."""

    enable_quality_gates: bool = Field(
        default=True, description="Enable quality gate validation"
    )
    enable_rollback: bool = Field(
        default=True, description="Enable rollback on failure"
    )
    validation_timeout_seconds: int = Field(
        default=30, description="Timeout for validation"
    )
    handoff_timeout_seconds: int = Field(
        default=60, description="Timeout for complete handoff"
    )
    preserve_history: bool = Field(
        default=True, description="Preserve handoff chain history"
    )


class HandoffCoordinator:
    """
    Handoff pattern coordinator.

    Implements sequential task handoff with:
    - Context preservation during transfers
    - Quality gates and validation
    - Rollback capabilities
    - Handoff history tracking
    """

    def __init__(
        self,
        coordinator_id: str,
        config: HandoffConfig | None = None,
        event_producer: StreamProducer | None = None,
    ) -> None:
        """
        Initialize handoff coordinator.

        Args:
            coordinator_id: Unique coordinator identifier
            config: Handoff configuration
            event_producer: Event stream producer
        """
        self.coordinator_id = coordinator_id
        self.config = config or HandoffConfig()
        self.event_producer = event_producer

        # Handoff tracking
        self._active_handoffs: dict[UUID, HandoffRecord] = {}
        self._completed_handoffs: dict[UUID, HandoffRecord] = {}
        self._quality_gates: list[HandoffGate] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def register_quality_gate(self, gate: HandoffGate) -> None:
        """
        Register a quality gate.

        Args:
            gate: Quality gate to register
        """
        self._quality_gates.append(gate)

    def unregister_quality_gate(self, gate_name: str) -> None:
        """
        Unregister a quality gate by name.

        Args:
            gate_name: Name of gate to remove
        """
        self._quality_gates = [
            g for g in self._quality_gates if g.gate_name != gate_name
        ]

    async def initiate_handoff(
        self,
        task_id: UUID,
        task_type: str,
        source_agent_id: str,
        target_agent_id: str,
        task_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """
        Initiate a task handoff.

        Args:
            task_id: Task identifier
            task_type: Type of task
            source_agent_id: Source agent
            target_agent_id: Target agent
            task_data: Task data to transfer
            metadata: Additional metadata

        Returns:
            Handoff identifier

        Raises:
            ValueError: If handoff validation fails
        """
        async with self._lock:
            # Create handoff context
            context = HandoffContext(
                task_id=task_id,
                task_type=task_type,
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                task_data=task_data.copy(),
                metadata=metadata or {},
                previous_state=task_data.copy()
                if self.config.enable_rollback
                else None,
            )

            # Build handoff chain
            if self.config.preserve_history:
                context.handoff_chain = [source_agent_id]

            # Create handoff record
            record = HandoffRecord(
                handoff_id=context.handoff_id,
                context=context,
                status=HandoffStatus.PENDING,
            )

            self._active_handoffs[context.handoff_id] = record

        # Publish handoff initiated event
        if self.event_producer:
            await self._publish_handoff_event(
                "handoff_initiated",
                context.handoff_id,
                {
                    "task_id": str(task_id),
                    "source_agent_id": source_agent_id,
                    "target_agent_id": target_agent_id,
                },
            )

        return context.handoff_id

    async def execute_handoff(self, handoff_id: UUID) -> bool:
        """
        Execute handoff with validation and transfer.

        Args:
            handoff_id: Handoff identifier

        Returns:
            True if handoff completed successfully, False otherwise
        """
        async with self._lock:
            if handoff_id not in self._active_handoffs:
                raise ValueError(f"Handoff not found: {handoff_id}")

            record = self._active_handoffs[handoff_id]
            context = record.context

        try:
            # Validate using quality gates
            if self.config.enable_quality_gates:
                record.status = HandoffStatus.VALIDATING

                validation_passed = await self._validate_handoff(record)

                if not validation_passed:
                    record.status = HandoffStatus.FAILED
                    record.error_message = "Quality gate validation failed"

                    # Publish failure event
                    if self.event_producer:
                        await self._publish_handoff_event(
                            "handoff_failed",
                            handoff_id,
                            {
                                "reason": "validation_failed",
                                "validation_results": [
                                    vr.model_dump() for vr in record.validation_results
                                ],
                            },
                        )

                    return False

            # Transfer context
            record.status = HandoffStatus.TRANSFERRING

            # Update handoff chain
            if self.config.preserve_history:
                context.handoff_chain.append(context.target_agent_id)

            # Mark as completed
            record.status = HandoffStatus.COMPLETED
            record.completed_at = datetime.now(UTC)
            context.completed_at = datetime.now(UTC)

            # Move to completed
            async with self._lock:
                self._completed_handoffs[handoff_id] = record
                del self._active_handoffs[handoff_id]

            # Publish completion event
            if self.event_producer:
                await self._publish_handoff_event(
                    "handoff_completed",
                    handoff_id,
                    {
                        "task_id": str(context.task_id),
                        "source_agent_id": context.source_agent_id,
                        "target_agent_id": context.target_agent_id,
                        "handoff_chain": context.handoff_chain,
                    },
                )

            return True

        except Exception as e:
            record.status = HandoffStatus.FAILED
            record.error_message = str(e)

            # Publish failure event
            if self.event_producer:
                await self._publish_handoff_event(
                    "handoff_failed",
                    handoff_id,
                    {"reason": "exception", "error": str(e)},
                )

            return False

    async def rollback_handoff(
        self,
        handoff_id: UUID,
        reason: str,
    ) -> bool:
        """
        Rollback a handoff to previous state.

        Args:
            handoff_id: Handoff identifier
            reason: Reason for rollback

        Returns:
            True if rollback successful, False otherwise
        """
        if not self.config.enable_rollback:
            raise ValueError("Rollback is disabled in configuration")

        async with self._lock:
            # Check active handoffs first
            if handoff_id in self._active_handoffs:
                record = self._active_handoffs[handoff_id]
            elif handoff_id in self._completed_handoffs:
                record = self._completed_handoffs[handoff_id]
            else:
                raise ValueError(f"Handoff not found: {handoff_id}")

            context = record.context

            if context.previous_state is None:
                raise ValueError("No previous state available for rollback")

            # Restore previous state
            context.task_data = context.previous_state.copy()

            # Revert handoff chain
            if self.config.preserve_history and len(context.handoff_chain) > 0:
                context.handoff_chain.pop()

            # Update status
            record.status = HandoffStatus.ROLLED_BACK
            record.error_message = f"Rolled back: {reason}"

            # Move to completed if was active
            if handoff_id in self._active_handoffs:
                self._completed_handoffs[handoff_id] = record
                del self._active_handoffs[handoff_id]

        # Publish rollback event
        if self.event_producer:
            await self._publish_handoff_event(
                "handoff_rolled_back",
                handoff_id,
                {
                    "task_id": str(context.task_id),
                    "reason": reason,
                    "restored_to_agent": context.source_agent_id,
                },
            )

        return True

    async def get_handoff_context(self, handoff_id: UUID) -> HandoffContext:
        """
        Get handoff context.

        Args:
            handoff_id: Handoff identifier

        Returns:
            Handoff context

        Raises:
            ValueError: If handoff not found
        """
        async with self._lock:
            if handoff_id in self._active_handoffs:
                return self._active_handoffs[handoff_id].context
            elif handoff_id in self._completed_handoffs:
                return self._completed_handoffs[handoff_id].context
            else:
                raise ValueError(f"Handoff not found: {handoff_id}")

    async def get_handoff_status(self, handoff_id: UUID) -> HandoffStatus:
        """
        Get handoff status.

        Args:
            handoff_id: Handoff identifier

        Returns:
            Current handoff status

        Raises:
            ValueError: If handoff not found
        """
        async with self._lock:
            if handoff_id in self._active_handoffs:
                return self._active_handoffs[handoff_id].status
            elif handoff_id in self._completed_handoffs:
                return self._completed_handoffs[handoff_id].status
            else:
                raise ValueError(f"Handoff not found: {handoff_id}")

    async def get_handoff_chain(self, task_id: UUID) -> list[str]:
        """
        Get handoff chain for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of agent IDs in handoff chain
        """
        async with self._lock:
            # Search active handoffs
            for record in self._active_handoffs.values():
                if record.context.task_id == task_id:
                    return record.context.handoff_chain.copy()

            # Search completed handoffs
            for record in self._completed_handoffs.values():
                if record.context.task_id == task_id:
                    return record.context.handoff_chain.copy()

        return []

    async def _validate_handoff(self, record: HandoffRecord) -> bool:
        """
        Validate handoff using registered quality gates.

        Args:
            record: Handoff record to validate

        Returns:
            True if all validations pass, False otherwise
        """
        record.validation_results.clear()

        for gate in self._quality_gates:
            try:
                # Apply timeout to validation
                result = await asyncio.wait_for(
                    gate.validate(record.context),
                    timeout=self.config.validation_timeout_seconds,
                )

                record.validation_results.append(result)

                if not result.valid:
                    return False

            except asyncio.TimeoutError:
                result = ValidationResult(
                    valid=False,
                    gate_name=gate.gate_name,
                    message="Validation timeout",
                )
                record.validation_results.append(result)
                return False

            except Exception as e:
                result = ValidationResult(
                    valid=False,
                    gate_name=gate.gate_name,
                    message=f"Validation error: {e!s}",
                )
                record.validation_results.append(result)
                return False

        return True

    async def _publish_handoff_event(
        self,
        event_type: str,
        handoff_id: UUID,
        data: dict[str, Any],
    ) -> None:
        """
        Publish handoff event to stream.

        Args:
            event_type: Type of event
            handoff_id: Handoff identifier
            data: Event data
        """
        if not self.event_producer:
            return

        event_data = {
            "event_type": event_type,
            "handoff_id": str(handoff_id),
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }

        await self.event_producer.publish(event_data)

    async def get_coordinator_status(self) -> dict[str, Any]:
        """
        Get current coordinator status.

        Returns:
            Status dictionary
        """
        async with self._lock:
            return {
                "coordinator_id": self.coordinator_id,
                "active_handoffs": len(self._active_handoffs),
                "completed_handoffs": len(self._completed_handoffs),
                "registered_gates": len(self._quality_gates),
                "quality_gates": [gate.gate_name for gate in self._quality_gates],
                "config": self.config.model_dump(),
            }
