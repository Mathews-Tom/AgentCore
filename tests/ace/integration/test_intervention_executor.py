"""
Integration tests for InterventionExecutor (COMPASS ACE-2 - ACE-018).

Tests intervention execution with mocked Agent Runtime client.
Validates all 4 intervention types, execution flow, and error handling.

Coverage target: 95%+
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest

from agentcore.ace.intervention.executor import (
    AgentRuntimeClient,
    InterventionExecutor,
)
from agentcore.ace.models.ace_models import (
    ExecutionStatus,
    InterventionDecision,
    InterventionType,
    TriggerType,
)


class TestAgentRuntimeClientInit:
    """Test AgentRuntimeClient initialization."""

    def test_init_success_default_params(self):
        """Test successful initialization with default parameters."""
        client = AgentRuntimeClient()

        assert client.base_url == "http://localhost:8001"
        assert client.timeout == 30.0
        assert client.client is not None

    def test_init_success_custom_params(self):
        """Test successful initialization with custom parameters."""
        client = AgentRuntimeClient(
            base_url="http://runtime.example.com:9000",
            timeout=60.0,
        )

        assert client.base_url == "http://runtime.example.com:9000"
        assert client.timeout == 60.0

    def test_init_strips_trailing_slash(self):
        """Test initialization strips trailing slash from base_url."""
        client = AgentRuntimeClient(base_url="http://localhost:8001/")

        assert client.base_url == "http://localhost:8001"

    def test_init_invalid_timeout_zero(self):
        """Test initialization with invalid timeout (zero)."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            AgentRuntimeClient(timeout=0)

    def test_init_invalid_timeout_negative(self):
        """Test initialization with invalid timeout (negative)."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            AgentRuntimeClient(timeout=-5.0)

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the HTTP client."""
        client = AgentRuntimeClient()
        await client.close()
        # Verify client is closed (would raise if used after close)
        assert client.client.is_closed


class TestAgentRuntimeClientSendIntervention:
    """Test AgentRuntimeClient send_intervention method."""

    @pytest.fixture
    async def mock_client(self):
        """Create client with mocked httpx client."""
        client = AgentRuntimeClient()
        client.client = AsyncMock(spec=httpx.AsyncClient)
        yield client
        await client.close()

    @pytest.fixture
    def test_params(self):
        """Test parameters for send_intervention."""
        return {
            "agent_id": "agent-001",
            "task_id": uuid4(),
            "intervention_type": InterventionType.CONTEXT_REFRESH,
            "context": {"rationale": "Test intervention", "confidence": 0.85},
        }

    @pytest.mark.asyncio
    async def test_send_intervention_success(self, mock_client, test_params):
        """Test successful intervention send."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "duration_ms": 150,
            "message": "Intervention executed successfully",
        }
        mock_client.client.post = AsyncMock(return_value=mock_response)

        result = await mock_client.send_intervention(**test_params)

        assert result["status"] == "success"
        assert result["duration_ms"] == 150
        assert "message" in result

        # Verify request was made
        mock_client.client.post.assert_called_once()
        call_args = mock_client.client.post.call_args
        assert "/api/v1/runtime/interventions" in call_args[0][0]

        # Verify payload structure
        payload = call_args[1]["json"]
        assert payload["agent_id"] == test_params["agent_id"]
        assert payload["task_id"] == str(test_params["task_id"])
        assert payload["intervention_type"] == test_params["intervention_type"].value
        assert payload["context"] == test_params["context"]

    @pytest.mark.asyncio
    async def test_send_intervention_adds_duration_if_missing(self, mock_client, test_params):
        """Test that duration_ms is added if not in response."""
        # Mock response without duration_ms
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "message": "Done",
        }
        mock_client.client.post = AsyncMock(return_value=mock_response)

        result = await mock_client.send_intervention(**test_params)

        assert "duration_ms" in result
        assert result["duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_send_intervention_http_error(self, mock_client, test_params):
        """Test intervention send with HTTP error."""
        # Mock HTTP error
        mock_client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            await mock_client.send_intervention(**test_params)

    @pytest.mark.asyncio
    async def test_send_intervention_timeout_error(self, mock_client, test_params):
        """Test intervention send with timeout error."""
        # Mock timeout error
        mock_client.client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        with pytest.raises(httpx.TimeoutException):
            await mock_client.send_intervention(**test_params)

    @pytest.mark.asyncio
    async def test_send_intervention_connection_error(self, mock_client, test_params):
        """Test intervention send with connection error."""
        # Mock connection error
        mock_client.client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with pytest.raises(httpx.ConnectError):
            await mock_client.send_intervention(**test_params)

    @pytest.mark.asyncio
    async def test_send_intervention_invalid_response(self, mock_client, test_params):
        """Test intervention send with invalid response (missing status)."""
        # Mock response without status field
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": "Missing status field",
        }
        mock_client.client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Response missing 'status' field"):
            await mock_client.send_intervention(**test_params)


class TestInterventionExecutorInit:
    """Test InterventionExecutor initialization."""

    def test_init_success_default_client(self):
        """Test successful initialization with default client."""
        executor = InterventionExecutor()

        assert executor.runtime_client is not None
        assert isinstance(executor.runtime_client, AgentRuntimeClient)

    def test_init_success_custom_client(self):
        """Test successful initialization with custom client."""
        custom_client = AgentRuntimeClient(base_url="http://custom:8000")
        executor = InterventionExecutor(runtime_client=custom_client)

        assert executor.runtime_client is custom_client


class TestInterventionExecutorContextRefresh:
    """Test execution of CONTEXT_REFRESH interventions."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """CONTEXT_REFRESH decision."""
        return InterventionDecision(
            intervention_type=InterventionType.CONTEXT_REFRESH,
            rationale="Context is stale with low retrieval relevance",
            confidence=0.85,
            expected_impact="Context health should improve to >0.8",
            alternative_interventions=["replan"],
            metadata={"trigger_confidence": 0.78},
        )

    @pytest.mark.asyncio
    async def test_execute_context_refresh_success(self, executor, decision):
        """Test successful CONTEXT_REFRESH execution."""
        # Mock successful runtime response
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "success",
                "duration_ms": 120,
                "message": "Context refreshed successfully",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-001",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.CONTEXT_STALENESS,
            trigger_signals=["context_age_exceeded", "low_retrieval_relevance"],
        )

        # Verify record structure
        assert record.task_id == task_id
        assert record.agent_id == "agent-001"
        assert record.intervention_type == InterventionType.CONTEXT_REFRESH
        assert record.trigger_type == TriggerType.CONTEXT_STALENESS
        assert record.trigger_signals == ["context_age_exceeded", "low_retrieval_relevance"]
        assert record.intervention_rationale == decision.rationale
        assert record.decision_confidence == decision.confidence
        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.execution_duration_ms > 0
        assert record.execution_error is None
        assert record.executed_at is not None

        # Verify runtime was called
        executor.runtime_client.send_intervention.assert_called_once()
        call_args = executor.runtime_client.send_intervention.call_args[1]
        assert call_args["agent_id"] == "agent-001"
        assert call_args["task_id"] == task_id
        assert call_args["intervention_type"] == InterventionType.CONTEXT_REFRESH


class TestInterventionExecutorReplan:
    """Test execution of REPLAN interventions."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """REPLAN decision."""
        return InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Velocity dropped 60% with high error rate",
            confidence=0.88,
            expected_impact="Velocity should return to baseline within 2-3 stages",
            alternative_interventions=["reflect", "context_refresh"],
            metadata={"trigger_confidence": 0.92},
        )

    @pytest.mark.asyncio
    async def test_execute_replan_success(self, executor, decision):
        """Test successful REPLAN execution."""
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "success",
                "duration_ms": 200,
                "message": "Replanning initiated",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-002",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["velocity_drop_below_threshold"],
        )

        assert record.intervention_type == InterventionType.REPLAN
        assert record.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.execution_duration_ms > 0


class TestInterventionExecutorReflect:
    """Test execution of REFLECT interventions."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """REFLECT decision."""
        return InterventionDecision(
            intervention_type=InterventionType.REFLECT,
            rationale="Error patterns show agent not learning from failures",
            confidence=0.82,
            expected_impact="Agent will identify failure patterns and adapt",
            alternative_interventions=["replan"],
            metadata={"trigger_confidence": 0.85},
        )

    @pytest.mark.asyncio
    async def test_execute_reflect_success(self, executor, decision):
        """Test successful REFLECT execution."""
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "success",
                "duration_ms": 180,
                "message": "Reflection triggered",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-003",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.ERROR_ACCUMULATION,
            trigger_signals=["high_error_count_in_stage"],
        )

        assert record.intervention_type == InterventionType.REFLECT
        assert record.trigger_type == TriggerType.ERROR_ACCUMULATION
        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.execution_duration_ms > 0


class TestInterventionExecutorCapabilitySwitch:
    """Test execution of CAPABILITY_SWITCH interventions."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """CAPABILITY_SWITCH decision."""
        return InterventionDecision(
            intervention_type=InterventionType.CAPABILITY_SWITCH,
            rationale="Low capability coverage with high action failure rate",
            confidence=0.92,
            expected_impact="Capability coverage should reach >90%",
            alternative_interventions=["replan"],
            metadata={"trigger_confidence": 0.88},
        )

    @pytest.mark.asyncio
    async def test_execute_capability_switch_success(self, executor, decision):
        """Test successful CAPABILITY_SWITCH execution."""
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "success",
                "duration_ms": 250,
                "message": "Capability switch completed",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-004",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.CAPABILITY_MISMATCH,
            trigger_signals=["low_capability_coverage", "high_action_failure_rate"],
        )

        assert record.intervention_type == InterventionType.CAPABILITY_SWITCH
        assert record.trigger_type == TriggerType.CAPABILITY_MISMATCH
        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.execution_duration_ms > 0


class TestInterventionExecutorFailureScenarios:
    """Test execution failure scenarios."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """Test decision."""
        return InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Test failure scenario",
            confidence=0.75,
            expected_impact="Test expected impact",
            alternative_interventions=[],
        )

    @pytest.mark.asyncio
    async def test_execute_runtime_failure_response(self, executor, decision):
        """Test execution with runtime failure response."""
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "failed",
                "duration_ms": 100,
                "message": "Runtime rejected intervention",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-005",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        assert record.execution_status == ExecutionStatus.FAILURE
        assert record.execution_error == "Runtime rejected intervention"
        assert record.execution_duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_runtime_partial_response(self, executor, decision):
        """Test execution with runtime partial response."""
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "partial",
                "duration_ms": 150,
                "message": "Partial execution completed",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-006",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        assert record.execution_status == ExecutionStatus.PARTIAL
        assert record.execution_error == "Partial execution completed"
        assert record.execution_duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_runtime_timeout(self, executor, decision):
        """Test execution with runtime timeout."""
        executor.runtime_client.send_intervention = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-007",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        assert record.execution_status == ExecutionStatus.FAILURE
        assert "Runtime communication error" in record.execution_error
        assert record.execution_duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_runtime_connection_error(self, executor, decision):
        """Test execution with runtime connection error."""
        executor.runtime_client.send_intervention = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-008",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        assert record.execution_status == ExecutionStatus.FAILURE
        assert "Runtime communication error" in record.execution_error
        assert record.execution_duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_runtime_http_status_error(self, executor, decision):
        """Test execution with runtime HTTP status error."""
        executor.runtime_client.send_intervention = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-009",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        assert record.execution_status == ExecutionStatus.FAILURE
        assert "Runtime communication error" in record.execution_error
        assert record.execution_duration_ms > 0


class TestInterventionExecutorUnexpectedErrors:
    """Test handling of unexpected errors."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """Test decision."""
        return InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Test unexpected error",
            confidence=0.75,
            expected_impact="Should handle unexpected error",
        )

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, executor, decision):
        """Test execution with unexpected error (not httpx.HTTPError)."""
        # Mock an unexpected error (e.g., AttributeError, KeyError, etc.)
        executor.runtime_client.send_intervention = AsyncMock(
            side_effect=RuntimeError("Unexpected runtime error")
        )

        task_id = uuid4()
        with pytest.raises(RuntimeError, match="Unexpected runtime error"):
            await executor.execute_intervention(
                agent_id="agent-013",
                task_id=task_id,
                decision=decision,
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                trigger_signals=["test"],
            )


class TestInterventionExecutorValidation:
    """Test input validation."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """Test decision."""
        return InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Test validation",
            confidence=0.75,
            expected_impact="Test expected impact",
        )

    @pytest.mark.asyncio
    async def test_execute_empty_agent_id(self, executor, decision):
        """Test execution with empty agent_id."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            await executor.execute_intervention(
                agent_id="",
                task_id=uuid4(),
                decision=decision,
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                trigger_signals=["test"],
            )

    @pytest.mark.asyncio
    async def test_execute_empty_trigger_signals(self, executor, decision):
        """Test execution with empty trigger_signals."""
        with pytest.raises(ValueError, match="trigger_signals cannot be empty"):
            await executor.execute_intervention(
                agent_id="agent-010",
                task_id=uuid4(),
                decision=decision,
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                trigger_signals=[],
            )


class TestInterventionExecutorPerformance:
    """Test execution performance and duration tracking."""

    @pytest.fixture
    def executor(self):
        """Executor with mocked runtime client."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        return InterventionExecutor(runtime_client=mock_runtime)

    @pytest.fixture
    def decision(self):
        """Test decision."""
        return InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Performance test",
            confidence=0.80,
            expected_impact="Fast execution",
        )

    @pytest.mark.asyncio
    async def test_execution_duration_tracking(self, executor, decision):
        """Test that execution duration is tracked accurately."""
        executor.runtime_client.send_intervention = AsyncMock(
            return_value={
                "status": "success",
                "duration_ms": 100,
                "message": "Done",
            }
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-011",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test"],
        )

        # Verify duration is positive and reasonable
        assert record.execution_duration_ms > 0
        assert record.execution_duration_ms < 1000  # Should be fast in tests

    @pytest.mark.asyncio
    async def test_execution_duration_on_failure(self, executor, decision):
        """Test that execution duration is tracked even on failure."""
        executor.runtime_client.send_intervention = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        task_id = uuid4()
        record = await executor.execute_intervention(
            agent_id="agent-012",
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test"],
        )

        # Verify duration is tracked despite failure
        assert record.execution_duration_ms > 0
        assert record.execution_status == ExecutionStatus.FAILURE


class TestInterventionExecutorClose:
    """Test executor cleanup."""

    @pytest.mark.asyncio
    async def test_close_executor(self):
        """Test closing the executor."""
        mock_runtime = MagicMock(spec=AgentRuntimeClient)
        mock_runtime.close = AsyncMock()
        executor = InterventionExecutor(runtime_client=mock_runtime)

        await executor.close()

        mock_runtime.close.assert_called_once()
