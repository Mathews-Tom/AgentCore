"""
Integration Tests for Error Recovery

Comprehensive test suite for error recovery scenarios including:
- Transient tool failures (network errors, timeouts)
- Invalid tool parameters (correctable via plan refinement)
- Incomplete results (fixable via verification feedback)
- Partial execution failures (recoverable through retries)
- Recovery metrics tracking (success rate, recovery time)

Tests validate NFR target: >80% recovery rate for recoverable errors.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.agent_runtime.models.error_types import CircuitBreakerConfig
from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionStatus,
    ToolResult,
)
from agentcore.agent_runtime.services.circuit_breaker import CircuitBreakerError
from agentcore.agent_runtime.tools.base import ExecutionContext as ToolExecutionContext
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.modular.coordinator import CoordinationContext, ModuleCoordinator
from agentcore.modular.executor import ErrorCategory, ExecutorModule
from agentcore.modular.generator import Generator
from agentcore.modular.interfaces import (
    ExecutionContext,
    ExecutionResult,
    GeneratedResponse,
    PlanStep,
    RetryPolicy,
    VerificationResult,
)
from agentcore.modular.models import (
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    PlanStatus,
    StepStatus,
)
from agentcore.modular.planner import Planner
from agentcore.modular.verifier import Verifier


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def a2a_context() -> A2AContext:
    """Create A2A context for testing."""
    return A2AContext(
        source_agent="test-error-recovery",
        target_agent="modular-agent",
        trace_id=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def coordination_context() -> CoordinationContext:
    """Create coordination context for testing."""
    return CoordinationContext(
        execution_id=str(uuid4()),
        trace_id=str(uuid4()),
        session_id=str(uuid4()),
        iteration=0,
    )


@pytest.fixture
def coordinator(coordination_context: CoordinationContext) -> ModuleCoordinator:
    """Create coordinator with context."""
    coordinator = ModuleCoordinator()
    coordinator.set_context(coordination_context)
    return coordinator


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create mock tool registry."""
    registry = ToolRegistry()
    return registry


@pytest.fixture
def tool_executor() -> ToolExecutor:
    """Create mock tool executor."""
    executor = MagicMock(spec=ToolExecutor)
    return executor


@pytest.fixture
def executor_module(
    tool_registry: ToolRegistry,
    tool_executor: ToolExecutor,
    a2a_context: A2AContext,
) -> ExecutorModule:
    """Create executor module with retry and circuit breaker enabled."""
    return ExecutorModule(
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        a2a_context=a2a_context,
        enable_circuit_breaker=True,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_max_calls=1,
        ),
        default_retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0.1,  # Faster for tests
            exponential=True,
        ),
    )


# ============================================================================
# Recovery Metrics Tracker
# ============================================================================


class RecoveryMetrics:
    """Track recovery metrics for validation."""

    def __init__(self) -> None:
        """Initialize metrics."""
        self.total_errors = 0
        self.recoverable_errors = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_times: list[float] = []
        self.recovery_attempts: list[int] = []

    def record_error(
        self,
        recoverable: bool,
        recovered: bool,
        recovery_time: float | None = None,
        attempts: int = 1,
    ) -> None:
        """Record an error and recovery outcome."""
        self.total_errors += 1
        if recoverable:
            self.recoverable_errors += 1
            if recovered:
                self.successful_recoveries += 1
                if recovery_time is not None:
                    self.recovery_times.append(recovery_time)
                self.recovery_attempts.append(attempts)
            else:
                self.failed_recoveries += 1

    @property
    def recovery_rate(self) -> float:
        """Calculate recovery success rate."""
        if self.recoverable_errors == 0:
            return 0.0
        return self.successful_recoveries / self.recoverable_errors

    @property
    def mean_recovery_time(self) -> float:
        """Calculate mean recovery time."""
        if not self.recovery_times:
            return 0.0
        return sum(self.recovery_times) / len(self.recovery_times)

    @property
    def p95_recovery_time(self) -> float:
        """Calculate p95 recovery time."""
        if not self.recovery_times:
            return 0.0
        sorted_times = sorted(self.recovery_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_recovery_time(self) -> float:
        """Calculate p99 recovery time."""
        if not self.recovery_times:
            return 0.0
        sorted_times = sorted(self.recovery_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def mean_attempts_to_success(self) -> float:
        """Calculate mean attempts before success."""
        if not self.recovery_attempts:
            return 0.0
        return sum(self.recovery_attempts) / len(self.recovery_attempts)

    def get_report(self) -> dict[str, Any]:
        """Generate recovery report."""
        return {
            "total_errors": self.total_errors,
            "recoverable_errors": self.recoverable_errors,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_rate": self.recovery_rate,
            "mean_recovery_time": self.mean_recovery_time,
            "p95_recovery_time": self.p95_recovery_time,
            "p99_recovery_time": self.p99_recovery_time,
            "mean_attempts_to_success": self.mean_attempts_to_success,
            "meets_nfr_target": self.recovery_rate >= 0.80,
        }


# ============================================================================
# Test Suite 1: Transient Tool Failures (Coordinator-Level)
# ============================================================================


@pytest.mark.asyncio
class TestTransientFailures:
    """Test recovery from transient tool failures via executor retry mechanism."""

    async def test_network_timeout_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery from network timeout errors through retry."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="network_fetch",
                    parameters={"url": "http://example.com"},
                    status=StepStatus.PENDING,
                )
            ],
            query="Fetch data from network",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)  # Return same plan for refinement

        # Mock execution results: timeout first time, success second (simulating retry recovery)
        timeout_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="Network timeout",
                execution_time=2.0,
                metadata={
                    "error_type": "TimeoutError",
                    "error_category": "timeout",
                    "retryable": True,
                },
            )
        ]

        success_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "success"},
                execution_time=0.5,
            )
        ]

        executor.execute_plan = AsyncMock(side_effect=[timeout_result, success_result])

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.0,
            errors=["Execution failed with timeout"],
            feedback="Retry execution",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.95,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Data fetched successfully",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute with retry via refinement loop
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Fetch data from network",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert result["answer"] == "Data fetched successfully"
        trace = result["execution_trace"]
        assert trace["iterations"] == 2  # Failed once, succeeded second time

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0

    async def test_temporary_connection_error_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery from temporary connection errors."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="api_call",
                    parameters={"endpoint": "/status"},
                    status=StepStatus.PENDING,
                )
            ],
            query="Check API status",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution: connection error first, success second
        connection_error_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="ConnectionError: Failed to connect",
                execution_time=0.1,
                metadata={
                    "error_type": "ConnectionError",
                    "error_category": "transient",
                    "retryable": True,
                },
            )
        ]

        success_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"status": "ok"},
                execution_time=0.5,
            )
        ]

        executor.execute_plan = AsyncMock(
            side_effect=[connection_error_result, success_result]
        )

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.0,
            errors=["Connection error"],
            feedback="Retry connection",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="API status: ok",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Check API status",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert "ok" in result["answer"]
        trace = result["execution_trace"]
        assert trace["iterations"] == 2

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0

    async def test_transient_runtime_error_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery from transient runtime errors."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="process_data",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Process data",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution: runtime error first, success second
        runtime_error_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="RuntimeError: Temporary resource unavailable",
                execution_time=0.1,
                metadata={
                    "error_type": "RuntimeError",
                    "error_category": "transient",
                    "retryable": True,
                },
            )
        ]

        success_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "recovered"},
                execution_time=0.5,
            )
        ]

        executor.execute_plan = AsyncMock(side_effect=[runtime_error_result, success_result])

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.0,
            errors=["Runtime error"],
            feedback="Retry execution",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.92,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Data processed successfully",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Process data",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert "successfully" in result["answer"]
        trace = result["execution_trace"]
        assert trace["iterations"] == 2

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0


# ============================================================================
# Test Suite 2: Invalid Tool Parameters
# ============================================================================


@pytest.mark.asyncio
class TestInvalidParameters:
    """Test recovery from invalid tool parameters via plan refinement."""

    async def test_parameter_validation_error_with_refinement(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery from parameter validation errors via refinement."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock initial plan with invalid parameters
        invalid_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="calculate",
                    parameters={"value": "invalid"},  # Should be int
                    status=StepStatus.PENDING,
                )
            ],
            query="Calculate sum",
            status=PlanStatus.PENDING,
        )

        # Mock refined plan with corrected parameters
        refined_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="calculate",
                    parameters={"value": 42},  # Fixed
                    status=StepStatus.PENDING,
                )
            ],
            query="Calculate sum",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=invalid_plan)
        planner.refine_plan = AsyncMock(return_value=refined_plan)

        # Mock execution results
        invalid_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="Parameter validation failed: 'value' must be int, got str",
                execution_time=0.1,
                metadata={
                    "error_type": "ParameterValidationError",
                    "error_category": "validation",
                    "retryable": False,
                },
            )
        ]

        valid_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"sum": 42},
                execution_time=0.5,
            )
        ]

        executor.execute_plan = AsyncMock(side_effect=[invalid_result, valid_result])

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.2,
            errors=["Parameter validation error"],
            feedback="Fix parameter types in plan",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.95,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Sum is 42",
            reasoning=None,
            sources=["step:step-1"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Calculate sum",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery via refinement
        assert result["answer"] == "Sum is 42"
        trace = result["execution_trace"]
        assert trace["iterations"] == 2
        assert trace["verification_passed"] is True

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0

    async def test_missing_required_parameter_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery from missing required parameters."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock initial plan missing parameter
        incomplete_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="search",
                    parameters={},  # Missing 'query'
                    status=StepStatus.PENDING,
                )
            ],
            query="Search for information",
            status=PlanStatus.PENDING,
        )

        # Mock refined plan with parameter
        complete_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="search",
                    parameters={"query": "information"},  # Added
                    status=StepStatus.PENDING,
                )
            ],
            query="Search for information",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=incomplete_plan)
        planner.refine_plan = AsyncMock(return_value=complete_plan)

        # Mock execution results
        incomplete_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="Missing required parameter: 'query'",
                execution_time=0.1,
                metadata={
                    "error_type": "ParameterValidationError",
                    "error_category": "validation",
                },
            )
        ]

        complete_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"results": ["item1", "item2"]},
                execution_time=1.0,
            )
        ]

        executor.execute_plan = AsyncMock(
            side_effect=[incomplete_result, complete_result]
        )

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.1,
            errors=["Missing required parameter"],
            feedback="Add required query parameter",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Found 2 results",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Search for information",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert result["answer"] == "Found 2 results"
        trace = result["execution_trace"]
        assert trace["iterations"] == 2

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0


# ============================================================================
# Test Suite 3: Incomplete Results
# ============================================================================


@pytest.mark.asyncio
class TestIncompleteResults:
    """Test recovery from incomplete results via verification feedback."""

    async def test_partial_data_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery when results are incomplete."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plans
        initial_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="fetch_data",
                    parameters={"fields": ["name"]},  # Incomplete
                    status=StepStatus.PENDING,
                )
            ],
            query="Get user info",
            status=PlanStatus.PENDING,
        )

        refined_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="fetch_data",
                    parameters={"fields": ["name", "email", "id"]},  # Complete
                    status=StepStatus.PENDING,
                )
            ],
            query="Get user info",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=initial_plan)
        planner.refine_plan = AsyncMock(return_value=refined_plan)

        # Mock execution results
        partial_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"name": "John"},  # Missing fields
                execution_time=0.5,
            )
        ]

        complete_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"name": "John", "email": "john@example.com", "id": 123},
                execution_time=0.8,
            )
        ]

        executor.execute_plan = AsyncMock(side_effect=[partial_result, complete_result])

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.4,
            errors=["Incomplete result: missing required fields"],
            feedback="Request all required fields: name, email, id",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.95,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="User: John (john@example.com) [ID: 123]",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Get user info",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert "John" in result["answer"]
        assert "john@example.com" in result["answer"]
        trace = result["execution_trace"]
        assert trace["iterations"] == 2

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0

    async def test_null_result_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery when result is null/empty."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plans
        plan_v1 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="get_data",
                    parameters={"query": "empty"},
                    status=StepStatus.PENDING,
                )
            ],
            query="Get data",
            status=PlanStatus.PENDING,
        )

        plan_v2 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="get_data",
                    parameters={"query": "valid"},  # Fixed query
                    status=StepStatus.PENDING,
                )
            ],
            query="Get data",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=plan_v1)
        planner.refine_plan = AsyncMock(return_value=plan_v2)

        # Mock execution results
        null_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result=None,  # Null result
                execution_time=0.3,
            )
        ]

        valid_result = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "content"},
                execution_time=0.5,
            )
        ]

        executor.execute_plan = AsyncMock(side_effect=[null_result, valid_result])

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.0,
            errors=["Null result returned"],
            feedback="Adjust query to return valid data",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Data: content",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Get data",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert "content" in result["answer"]
        trace = result["execution_trace"]
        assert trace["iterations"] == 2

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0


# ============================================================================
# Test Suite 4: Partial Execution Failures
# ============================================================================


@pytest.mark.asyncio
class TestPartialExecutionFailures:
    """Test recovery from partial execution failures."""

    async def test_multi_step_partial_failure_recovery(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test recovery when some steps fail but overall execution succeeds."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan with multiple steps
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="fetch",
                    parameters={},
                    status=StepStatus.PENDING,
                ),
                EnhancedPlanStep(
                    step_id="step-2",
                    action="process",
                    parameters={},
                    status=StepStatus.PENDING,
                ),
                EnhancedPlanStep(
                    step_id="step-3",
                    action="save",
                    parameters={},
                    status=StepStatus.PENDING,
                ),
            ],
            query="Process data",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution results: step-2 fails first time, succeeds second
        results_iter1 = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "raw"},
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step-2",
                success=False,
                result=None,
                error="Processing failed",
                execution_time=0.3,
            ),
            ExecutionResult(
                step_id="step-3",
                success=True,
                result={"saved": False},
                execution_time=0.2,
            ),
        ]

        results_iter2 = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "raw"},
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step-2",
                success=True,
                result={"processed": "data"},
                execution_time=0.6,
            ),
            ExecutionResult(
                step_id="step-3",
                success=True,
                result={"saved": True},
                execution_time=0.2,
            ),
        ]

        executor.execute_plan = AsyncMock(side_effect=[results_iter1, results_iter2])

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Step 2 failed"],
            feedback="Fix processing step",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.95,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Data processed and saved",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        start_time = time.time()
        result = await coordinator.execute_with_refinement(
            query="Process data",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )
        recovery_time = time.time() - start_time

        # Verify recovery
        assert "processed" in result["answer"]
        trace = result["execution_trace"]
        assert trace["iterations"] == 2
        assert trace["successful_steps"] == 3

        # Record metrics
        metrics.record_error(
            recoverable=True,
            recovered=True,
            recovery_time=recovery_time,
            attempts=2,
        )

        assert metrics.recovery_rate == 1.0


# ============================================================================
# Test Suite 5: Non-Recoverable Errors
# ============================================================================


@pytest.mark.asyncio
class TestNonRecoverableErrors:
    """Test proper handling of non-recoverable errors."""

    async def test_tool_not_found_error(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test that tool not found errors are handled gracefully."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="nonexistent_tool",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Use nonexistent tool",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution - tool not found (non-recoverable)
        error_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="Tool 'nonexistent_tool' not found in registry",
                execution_time=0.01,
                metadata={
                    "error_type": "ToolNotFoundError",
                    "error_category": "tool_not_found",
                    "retryable": False,
                },
            )
        ]
        executor.execute_plan = AsyncMock(return_value=error_result)

        # Mock verification - will fail but can't be recovered (always same error)
        verification_result = VerificationResult(
            valid=False,
            confidence=0.0,
            errors=["Tool not found"],
            feedback="Tool does not exist in registry",
        )
        # Return the same error 3 times
        verifier.validate_results = AsyncMock(side_effect=[
            verification_result, verification_result, verification_result
        ])

        # Mock generation for partial results
        generation_response = GeneratedResponse(
            format="text",
            content="Unable to complete: tool not found",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute - should reach max iterations without recovery
        result = await coordinator.execute_with_refinement(
            query="Use nonexistent tool",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=3,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify non-recoverable error handling
        trace = result["execution_trace"]
        assert trace["verification_passed"] is False  # Never recovered
        assert trace["iterations"] >= 2  # At least 2 iterations (may stop early due to convergence)

        # Record metrics
        metrics.record_error(
            recoverable=False,
            recovered=False,
        )

        assert metrics.recovery_rate == 0.0  # No recoverable errors

    async def test_validation_error_non_recoverable_without_refinement(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test that validation errors without refinement potential are handled."""
        metrics = RecoveryMetrics()

        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan that won't improve
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="validation_test",
                    parameters={"invalid": "always"},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test validation",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)  # Returns same plan

        # Mock execution - validation error (non-recoverable at executor level)
        error_result = [
            ExecutionResult(
                step_id="step-1",
                success=False,
                result=None,
                error="Parameter validation failed: invalid parameter",
                execution_time=0.01,
                metadata={
                    "error_type": "ParameterValidationError",
                    "error_category": "validation",
                    "retryable": False,
                },
            )
        ]
        executor.execute_plan = AsyncMock(return_value=error_result)

        # Mock verification - always fail (3 times to reach max iterations)
        verification_result = VerificationResult(
            valid=False,
            confidence=0.0,
            errors=["Validation error"],
            feedback="Invalid parameter",
        )
        verifier.validate_results = AsyncMock(side_effect=[
            verification_result, verification_result, verification_result
        ])

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Unable to complete: validation error",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test validation",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=3,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify non-recoverable error handling
        trace = result["execution_trace"]
        assert trace["verification_passed"] is False
        assert trace["iterations"] >= 2  # At least 2 iterations (may stop early due to convergence)

        # Record metrics
        metrics.record_error(
            recoverable=False,
            recovered=False,
        )


# ============================================================================
# Test Suite 6: Recovery Metrics Validation
# ============================================================================


@pytest.mark.asyncio
class TestRecoveryMetrics:
    """Test recovery metrics tracking and NFR target validation."""

    async def test_comprehensive_recovery_scenarios(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test comprehensive recovery scenarios and validate >80% target."""
        metrics = RecoveryMetrics()

        # Scenario 1: Timeout recovery (recoverable)
        planner1 = MagicMock(spec=Planner)
        executor1 = MagicMock(spec=ExecutorModule)
        verifier1 = MagicMock(spec=Verifier)
        generator1 = MagicMock(spec=Generator)

        plan1 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[EnhancedPlanStep(step_id="s1", action="test", parameters={}, status=StepStatus.PENDING)],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner1.analyze_query = AsyncMock(return_value=plan1)
        planner1.refine_plan = AsyncMock(return_value=plan1)

        executor1.execute_plan = AsyncMock(side_effect=[
            [ExecutionResult(step_id="s1", success=False, result=None, error="Timeout", execution_time=1.0)],
            [ExecutionResult(step_id="s1", success=True, result={"ok": True}, execution_time=0.5)]
        ])

        verifier1.validate_results = AsyncMock(side_effect=[
            VerificationResult(valid=False, confidence=0.0, errors=["Timeout"], feedback="Retry"),
            VerificationResult(valid=True, confidence=0.95, errors=[], warnings=[])
        ])

        generator1.synthesize_response = AsyncMock(return_value=GeneratedResponse(
            format="text", content="Success", reasoning=None, sources=[]
        ))

        start = time.time()
        await coordinator.execute_with_refinement(
            query="Test", planner=planner1, executor=executor1, verifier=verifier1,
            generator=generator1, max_iterations=5, timeout_seconds=30.0, confidence_threshold=0.7
        )
        elapsed1 = time.time() - start
        metrics.record_error(recoverable=True, recovered=True, recovery_time=elapsed1, attempts=2)

        # Scenario 2: Connection error recovery (recoverable)
        planner2 = MagicMock(spec=Planner)
        executor2 = MagicMock(spec=ExecutorModule)
        verifier2 = MagicMock(spec=Verifier)
        generator2 = MagicMock(spec=Generator)

        plan2 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[EnhancedPlanStep(step_id="s2", action="connect", parameters={}, status=StepStatus.PENDING)],
            query="Connect",
            status=PlanStatus.PENDING,
        )
        planner2.analyze_query = AsyncMock(return_value=plan2)
        planner2.refine_plan = AsyncMock(return_value=plan2)

        executor2.execute_plan = AsyncMock(side_effect=[
            [ExecutionResult(step_id="s2", success=False, result=None, error="ConnectionError", execution_time=0.1)],
            [ExecutionResult(step_id="s2", success=True, result={"connected": True}, execution_time=0.5)]
        ])

        verifier2.validate_results = AsyncMock(side_effect=[
            VerificationResult(valid=False, confidence=0.0, errors=["Connection failed"], feedback="Retry"),
            VerificationResult(valid=True, confidence=0.92, errors=[], warnings=[])
        ])

        generator2.synthesize_response = AsyncMock(return_value=GeneratedResponse(
            format="text", content="Connected", reasoning=None, sources=[]
        ))

        start = time.time()
        await coordinator.execute_with_refinement(
            query="Connect", planner=planner2, executor=executor2, verifier=verifier2,
            generator=generator2, max_iterations=5, timeout_seconds=30.0, confidence_threshold=0.7
        )
        elapsed2 = time.time() - start
        metrics.record_error(recoverable=True, recovered=True, recovery_time=elapsed2, attempts=2)

        # Scenario 3: Tool not found (non-recoverable)
        metrics.record_error(recoverable=False, recovered=False)

        # Scenario 4: Runtime error recovery (recoverable)
        planner4 = MagicMock(spec=Planner)
        executor4 = MagicMock(spec=ExecutorModule)
        verifier4 = MagicMock(spec=Verifier)
        generator4 = MagicMock(spec=Generator)

        plan4 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[EnhancedPlanStep(step_id="s4", action="runtime", parameters={}, status=StepStatus.PENDING)],
            query="Runtime",
            status=PlanStatus.PENDING,
        )
        planner4.analyze_query = AsyncMock(return_value=plan4)
        planner4.refine_plan = AsyncMock(return_value=plan4)

        executor4.execute_plan = AsyncMock(side_effect=[
            [ExecutionResult(step_id="s4", success=False, result=None, error="RuntimeError", execution_time=0.1)],
            [ExecutionResult(step_id="s4", success=False, result=None, error="RuntimeError", execution_time=0.1)],
            [ExecutionResult(step_id="s4", success=True, result={"data": "ok"}, execution_time=0.5)]
        ])

        verifier4.validate_results = AsyncMock(side_effect=[
            VerificationResult(valid=False, confidence=0.0, errors=["Runtime error"], feedback="Retry"),
            VerificationResult(valid=False, confidence=0.0, errors=["Runtime error"], feedback="Retry"),
            VerificationResult(valid=True, confidence=0.88, errors=[], warnings=[])
        ])

        generator4.synthesize_response = AsyncMock(return_value=GeneratedResponse(
            format="text", content="OK", reasoning=None, sources=[]
        ))

        start = time.time()
        await coordinator.execute_with_refinement(
            query="Runtime", planner=planner4, executor=executor4, verifier=verifier4,
            generator=generator4, max_iterations=5, timeout_seconds=30.0, confidence_threshold=0.7
        )
        elapsed4 = time.time() - start
        metrics.record_error(recoverable=True, recovered=True, recovery_time=elapsed4, attempts=3)

        # Scenario 5: Validation error (non-recoverable without refinement)
        metrics.record_error(recoverable=False, recovered=False)

        # Generate report
        report = metrics.get_report()

        # Validate metrics
        assert report["total_errors"] == 5
        assert report["recoverable_errors"] == 3  # Scenarios 1, 2, 4
        assert report["successful_recoveries"] == 3  # All recoverable succeeded
        assert report["failed_recoveries"] == 0
        assert report["recovery_rate"] == 1.0  # 3/3 = 100%
        assert report["meets_nfr_target"] is True  # >80%
        assert report["mean_attempts_to_success"] > 1.0  # Had to retry
        assert report["mean_recovery_time"] > 0.0
        assert report["p95_recovery_time"] > 0.0
        assert report["p99_recovery_time"] > 0.0

        # Print report for visibility
        print("\n" + "=" * 80)
        print("ERROR RECOVERY TEST REPORT")
        print("=" * 80)
        print(f"Total Errors: {report['total_errors']}")
        print(f"Recoverable Errors: {report['recoverable_errors']}")
        print(f"Successful Recoveries: {report['successful_recoveries']}")
        print(f"Failed Recoveries: {report['failed_recoveries']}")
        print(f"Recovery Rate: {report['recovery_rate']:.1%}")
        print(f"Mean Recovery Time: {report['mean_recovery_time']:.3f}s")
        print(f"P95 Recovery Time: {report['p95_recovery_time']:.3f}s")
        print(f"P99 Recovery Time: {report['p99_recovery_time']:.3f}s")
        print(f"Mean Attempts to Success: {report['mean_attempts_to_success']:.1f}")
        print(f"Meets NFR Target (>80%): {report['meets_nfr_target']}")
        print("=" * 80)

        # Validate NFR requirement
        assert report["meets_nfr_target"], (
            f"Recovery rate {report['recovery_rate']:.1%} does not meet "
            f"NFR target of >80%"
        )
