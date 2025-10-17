"""Unit tests for TrajectoryRecorder middleware."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.agent_runtime.engines.base import PhilosophyEngine
from agentcore.agent_runtime.models.agent_config import AgentConfig
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.training.middleware.trajectory_recorder import TrajectoryRecorder
from agentcore.training.models import TrajectoryStep


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-agent-456",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"},
    )


@pytest.fixture
def mock_engine(agent_config):
    """Create mock philosophy engine."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    async def mock_execute(input_data, state):
        return {
            "final_answer": "42",
            "iterations": 3,
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "thought",
                    "content": "I need to calculate",
                },
                {
                    "step_number": 2,
                    "step_type": "action",
                    "content": "calculator(a=40, b=2)",
                },
                {
                    "step_number": 3,
                    "step_type": "final_answer",
                    "content": "42",
                },
            ],
            "completed": True,
        }

    engine.execute = AsyncMock(side_effect=mock_execute)
    return engine


@pytest.fixture
def recorder(mock_engine):
    """Create trajectory recorder."""
    return TrajectoryRecorder(engine=mock_engine)


@pytest.mark.asyncio
async def test_recorder_initialization(mock_engine):
    """Test trajectory recorder initialization."""
    recorder = TrajectoryRecorder(engine=mock_engine)

    assert recorder.engine == mock_engine
    assert recorder.steps == []
    assert recorder.start_time is None
    assert recorder.recording is False


@pytest.mark.asyncio
async def test_execute_with_recording(recorder):
    """Test executing with recording enabled."""
    input_data = {"goal": "What is 40 + 2?", "max_iterations": 10}
    state = AgentExecutionState(
        agent_id="test-agent-456",
        status="running",
    )

    result, steps = await recorder.execute_with_recording(input_data, state)

    # Check result
    assert result["final_answer"] == "42"
    assert result["completed"] is True
    assert result["iterations"] == 3

    # Check steps
    assert len(steps) == 3
    assert all(isinstance(step, TrajectoryStep) for step in steps)

    # Check recording state after execution
    assert recorder.recording is False
    assert len(recorder.steps) == 3


@pytest.mark.asyncio
async def test_recorded_steps_structure(recorder):
    """Test structure of recorded steps."""
    input_data = {"goal": "Test task", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    result, steps = await recorder.execute_with_recording(input_data, state)

    # Check first step
    step1 = steps[0]
    assert step1.state["iteration"] == 1
    assert step1.state["agent_id"] == "test-agent-456"
    assert step1.action["step_type"] == "thought"
    assert "calculate" in step1.action["content"].lower()
    assert step1.result["content"] == "I need to calculate"
    assert step1.result["success"] is True
    assert isinstance(step1.timestamp, datetime)
    assert step1.duration_ms == 100  # Approximation

    # Check second step
    step2 = steps[1]
    assert step2.state["iteration"] == 2
    assert step2.action["step_type"] == "action"

    # Check third step
    step3 = steps[2]
    assert step3.state["iteration"] == 3
    assert step3.action["step_type"] == "final_answer"


@pytest.mark.asyncio
async def test_execute_with_recording_error(agent_config):
    """Test recording when engine execution fails."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    async def mock_execute_failure(input_data, state):
        raise ValueError("Execution failed")

    engine.execute = AsyncMock(side_effect=mock_execute_failure)

    recorder = TrajectoryRecorder(engine=engine)

    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    result, steps = await recorder.execute_with_recording(input_data, state)

    # Should return error result
    assert "error" in result
    assert result["error"] == "Execution failed"
    assert result["completed"] is False

    # Steps should be empty on error
    assert len(steps) == 0

    # Recording should be stopped
    assert recorder.recording is False


@pytest.mark.asyncio
async def test_execute_with_recording_partial_steps(agent_config):
    """Test recording with partial steps before error."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    async def mock_execute_with_steps(input_data, state):
        # Return result with steps before potential error
        return {
            "final_answer": "Incomplete",
            "iterations": 2,
            "steps": [
                {"step_number": 1, "step_type": "thought", "content": "Starting"},
                {"step_number": 2, "step_type": "action", "content": "Doing something"},
            ],
            "completed": False,
        }

    engine.execute = AsyncMock(side_effect=mock_execute_with_steps)

    recorder = TrajectoryRecorder(engine=engine)

    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    result, steps = await recorder.execute_with_recording(input_data, state)

    # Should have recorded the steps
    assert len(steps) == 2
    assert steps[0].action["step_type"] == "thought"
    assert steps[1].action["step_type"] == "action"


@pytest.mark.asyncio
async def test_get_recorded_steps(recorder):
    """Test getting recorded steps."""
    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    await recorder.execute_with_recording(input_data, state)

    steps = recorder.get_recorded_steps()

    assert len(steps) == 3
    # Should be a copy
    assert steps is not recorder.steps
    assert steps == recorder.steps


@pytest.mark.asyncio
async def test_clear_recording(recorder):
    """Test clearing recording."""
    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    await recorder.execute_with_recording(input_data, state)

    assert len(recorder.steps) == 3
    assert recorder.start_time is not None

    recorder.clear_recording()

    assert len(recorder.steps) == 0
    assert recorder.start_time is None
    assert recorder.recording is False


@pytest.mark.asyncio
async def test_multiple_recordings(recorder):
    """Test multiple sequential recordings."""
    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    # First recording
    result1, steps1 = await recorder.execute_with_recording(input_data, state)
    assert len(steps1) == 3

    # Second recording (should clear previous)
    result2, steps2 = await recorder.execute_with_recording(input_data, state)
    assert len(steps2) == 3

    # Steps should be from second recording only
    assert recorder.get_recorded_steps() == steps2


@pytest.mark.asyncio
async def test_content_truncation(agent_config):
    """Test that long content is truncated in action field."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    # Create step with very long content
    long_content = "x" * 500

    async def mock_execute_long_content(input_data, state):
        return {
            "final_answer": "Done",
            "iterations": 1,
            "steps": [
                {"step_number": 1, "step_type": "thought", "content": long_content},
            ],
            "completed": True,
        }

    engine.execute = AsyncMock(side_effect=mock_execute_long_content)

    recorder = TrajectoryRecorder(engine=engine)

    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    result, steps = await recorder.execute_with_recording(input_data, state)

    # Action content should be truncated to 200 chars
    assert len(steps[0].action["content"]) == 200
    # Result content should not be truncated
    assert len(steps[0].result["content"]) == 500


@pytest.mark.asyncio
async def test_recording_with_empty_steps(agent_config):
    """Test recording when engine returns no steps."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    async def mock_execute_no_steps(input_data, state):
        return {
            "final_answer": "Quick answer",
            "iterations": 1,
            "steps": [],  # No steps
            "completed": True,
        }

    engine.execute = AsyncMock(side_effect=mock_execute_no_steps)

    recorder = TrajectoryRecorder(engine=engine)

    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    result, steps = await recorder.execute_with_recording(input_data, state)

    assert len(steps) == 0
    assert result["completed"] is True


@pytest.mark.asyncio
async def test_timestamp_recording(recorder):
    """Test that timestamps are recorded for each step."""
    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    result, steps = await recorder.execute_with_recording(input_data, state)

    for step in steps:
        assert isinstance(step.timestamp, datetime)
        # Should be timezone-aware
        assert step.timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_recording_state_management(recorder):
    """Test recording state flag management."""
    assert recorder.recording is False

    input_data = {"goal": "Test", "max_iterations": 5}
    state = AgentExecutionState(agent_id="test-agent-456", status="running")

    # Start recording
    task = recorder.execute_with_recording(input_data, state)

    # During execution, recording should be True
    # (Note: This is racy, but works for testing)
    await task

    # After execution, recording should be False
    assert recorder.recording is False
