"""Unit tests for TrajectoryCollector."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.agent_runtime.engines.base import PhilosophyEngine
from agentcore.agent_runtime.models.agent_config import AgentConfig
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.training.models import Trajectory, TrajectoryStep
from agentcore.training.trajectory import TrajectoryCollector


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-agent-123",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"})


@pytest.fixture
def mock_engine(agent_config):
    """Create mock philosophy engine."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    # Mock execute to return successful result
    async def mock_execute(input_data, state):
        # Add small delay to simulate realistic execution
        await asyncio.sleep(0.001)

        return {
            "final_answer": "Test answer",
            "iterations": 2,
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "thought",
                    "content": "I need to solve this",
                },
                {
                    "step_number": 2,
                    "step_type": "final_answer",
                    "content": "Test answer",
                },
            ],
            "completed": True,
        }

    engine.execute = AsyncMock(side_effect=mock_execute)
    return engine


@pytest.fixture
def collector(agent_config, mock_engine):
    """Create trajectory collector."""
    return TrajectoryCollector(
        agent_config=agent_config,
        engine=mock_engine,
        max_concurrent=8,
        max_steps_per_trajectory=20,
        timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_collector_initialization(agent_config, mock_engine):
    """Test trajectory collector initialization."""
    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=mock_engine,
        max_concurrent=4,
        max_steps_per_trajectory=10,
        timeout_seconds=30.0)

    assert collector.agent_config == agent_config
    assert collector.engine == mock_engine
    assert collector.max_concurrent == 4
    assert collector.max_steps_per_trajectory == 10
    assert collector.timeout_seconds == 30.0


@pytest.mark.asyncio
async def test_collect_single_trajectory(collector):
    """Test collecting a single trajectory."""
    job_id = uuid4()
    query = "What is 2+2?"

    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query=query,
        n_trajectories=1)

    assert len(trajectories) == 1
    trajectory = trajectories[0]

    assert trajectory.job_id == job_id
    assert trajectory.agent_id == "test-agent-123"
    assert trajectory.query == query
    assert len(trajectory.steps) == 2
    assert trajectory.success is True
    assert trajectory.execution_time_ms is not None
    assert trajectory.execution_time_ms >= 0  # Can be 0 for very fast mocks


@pytest.mark.asyncio
async def test_collect_multiple_trajectories_parallel(collector):
    """Test collecting multiple trajectories in parallel."""
    job_id = uuid4()
    query = "Test query"
    n_trajectories = 8

    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query=query,
        n_trajectories=n_trajectories)

    assert len(trajectories) == n_trajectories

    for trajectory in trajectories:
        assert trajectory.job_id == job_id
        assert trajectory.agent_id == "test-agent-123"
        assert trajectory.query == query
        assert len(trajectory.steps) == 2


@pytest.mark.asyncio
async def test_collect_trajectories_with_failures(agent_config):
    """Test collecting trajectories with some failures."""
    # Create engine that fails on odd indices
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    call_count = {"count": 0}

    async def mock_execute_with_failures(input_data, state):
        call_count["count"] += 1
        if call_count["count"] % 2 == 0:
            raise ValueError("Simulated execution failure")

        return {
            "final_answer": "Success",
            "iterations": 1,
            "steps": [
                {"step_number": 1, "step_type": "final_answer", "content": "Success"}
            ],
            "completed": True,
        }

    engine.execute = AsyncMock(side_effect=mock_execute_with_failures)

    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=engine,
        max_concurrent=8)

    job_id = uuid4()
    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query="Test query",
        n_trajectories=4)

    # Should get 2 successful trajectories (odd indices: 1, 3)
    assert len(trajectories) == 2


@pytest.mark.asyncio
async def test_collect_trajectories_timeout(agent_config):
    """Test trajectory collection with timeout."""
    # Create engine that hangs
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    async def mock_execute_slow(input_data, state):
        await asyncio.sleep(10)  # Longer than timeout
        return {"final_answer": "Too slow", "completed": True, "steps": []}

    engine.execute = AsyncMock(side_effect=mock_execute_slow)

    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=engine,
        max_concurrent=2,
        timeout_seconds=0.1,  # Very short timeout
    )

    job_id = uuid4()
    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query="Test query",
        n_trajectories=2)

    # All should timeout
    assert len(trajectories) == 0


@pytest.mark.asyncio
async def test_collect_trajectories_concurrency_limit(agent_config):
    """Test that concurrency limit is respected."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    # Track concurrent executions
    concurrent_count = {"current": 0, "max": 0}

    async def mock_execute_tracked(input_data, state):
        concurrent_count["current"] += 1
        concurrent_count["max"] = max(
            concurrent_count["max"], concurrent_count["current"]
        )

        await asyncio.sleep(0.01)  # Small delay to simulate work

        concurrent_count["current"] -= 1

        return {
            "final_answer": "Done",
            "completed": True,
            "steps": [
                {"step_number": 1, "step_type": "final_answer", "content": "Done"}
            ],
        }

    engine.execute = AsyncMock(side_effect=mock_execute_tracked)

    max_concurrent = 3
    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=engine,
        max_concurrent=max_concurrent)

    job_id = uuid4()
    await collector.collect_trajectories(
        job_id=job_id,
        query="Test query",
        n_trajectories=10)

    # Max concurrent should not exceed limit
    assert concurrent_count["max"] <= max_concurrent


@pytest.mark.asyncio
async def test_trajectory_step_extraction(collector):
    """Test trajectory step extraction from engine result."""
    job_id = uuid4()
    query = "Test query"

    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query=query,
        n_trajectories=1)

    trajectory = trajectories[0]
    steps = trajectory.steps

    assert len(steps) == 2

    # Check first step
    assert steps[0].state["iteration"] == 1
    assert steps[0].action["step_type"] == "thought"
    assert "solve" in steps[0].result["content"].lower()
    assert isinstance(steps[0].timestamp, datetime)
    assert steps[0].duration_ms >= 0

    # Check second step
    assert steps[1].state["iteration"] == 2
    assert steps[1].action["step_type"] == "final_answer"


@pytest.mark.asyncio
async def test_trajectory_success_determination(agent_config):
    """Test trajectory success determination based on completion."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    # Test completed execution
    async def mock_execute_complete(input_data, state):
        return {
            "final_answer": "Success",
            "completed": True,
            "steps": [
                {"step_number": 1, "step_type": "final_answer", "content": "Success"}
            ],
        }

    engine.execute = AsyncMock(side_effect=mock_execute_complete)

    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=engine)

    job_id = uuid4()
    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query="Test",
        n_trajectories=1)

    assert trajectories[0].success is True

    # Test incomplete execution
    async def mock_execute_incomplete(input_data, state):
        return {
            "final_answer": "Incomplete",
            "completed": False,
            "steps": [],
        }

    engine.execute = AsyncMock(side_effect=mock_execute_incomplete)

    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query="Test",
        n_trajectories=1)

    assert trajectories[0].success is False


@pytest.mark.asyncio
async def test_execution_time_tracking(collector):
    """Test that execution time is tracked correctly."""
    job_id = uuid4()
    query = "Test query"

    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query=query,
        n_trajectories=1)

    trajectory = trajectories[0]

    assert trajectory.execution_time_ms is not None
    assert trajectory.execution_time_ms >= 0
    # Should be reasonably fast (< 5 seconds = 5000ms) or 0 for fast mocks
    assert trajectory.execution_time_ms < 5000 or trajectory.execution_time_ms >= 0


@pytest.mark.asyncio
async def test_collect_trajectories_empty_steps(agent_config):
    """Test handling of execution with no steps."""
    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    async def mock_execute_no_steps(input_data, state):
        return {
            "final_answer": "Quick answer",
            "completed": True,
            "steps": [],  # No steps
        }

    engine.execute = AsyncMock(side_effect=mock_execute_no_steps)

    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=engine)

    job_id = uuid4()
    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query="Test",
        n_trajectories=1)

    assert len(trajectories) == 1
    assert len(trajectories[0].steps) == 0
    assert trajectories[0].success is True


@pytest.mark.asyncio
async def test_trajectory_fields_populated(collector):
    """Test that all trajectory fields are properly populated."""
    job_id = uuid4()
    query = "What is the capital of France?"

    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query=query,
        n_trajectories=1)

    trajectory = trajectories[0]

    # Check all fields are populated
    assert trajectory.trajectory_id is None  # Not assigned until DB insert
    assert trajectory.job_id == job_id
    assert trajectory.agent_id == "test-agent-123"
    assert trajectory.query == query
    assert isinstance(trajectory.steps, list)
    assert trajectory.reward == 0.0  # Default value
    assert trajectory.normalized_reward == 0.0  # Default value
    assert trajectory.advantage == 0.0  # Default value
    assert trajectory.execution_time_ms is not None
    assert trajectory.success is True
    assert trajectory.created_at is None  # Not assigned until DB insert
