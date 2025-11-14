"""Integration tests for agent execution with trajectory collection."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.database.connection import get_session
from agentcore.agent_runtime.engines.react_engine import ReActEngine
from agentcore.agent_runtime.models.agent_config import AgentConfig
from agentcore.training.middleware.trajectory_recorder import TrajectoryRecorder
from agentcore.training.models import GRPOConfig, TrainingJob, TrainingQuery
from agentcore.training.repositories import TrainingJobRepository, TrajectoryRepository
from agentcore.training.trajectory import TrajectoryCollector


@pytest.mark.asyncio
async def test_trajectory_collection_with_real_engine(init_test_db):
    """Test trajectory collection with real ReAct engine."""
    # Create agent config
    agent_config = AgentConfig(
        agent_id="test-agent-integration",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"})

    # Create ReAct engine
    engine = ReActEngine(config=agent_config)
    await engine.initialize()

    try:
        # Create trajectory collector
        collector = TrajectoryCollector(
            agent_config=agent_config,
            engine=engine,
            max_concurrent=4,
            max_steps_per_trajectory=10,
            timeout_seconds=10.0)

        # Collect trajectories
        job_id = uuid4()
        query = "What is 5 + 3?"

        trajectories = await collector.collect_trajectories(
            job_id=job_id,
            query=query,
            n_trajectories=4)

        # Verify trajectories collected
        assert len(trajectories) == 4

        for trajectory in trajectories:
            assert trajectory.job_id == job_id
            assert trajectory.agent_id == agent_config.agent_id
            assert trajectory.query == query
            assert len(trajectory.steps) > 0
            assert trajectory.execution_time_ms is not None
            assert trajectory.execution_time_ms >= 0  # Can be 0 for very fast execution
            # ReAct engine should complete successfully
            assert trajectory.success is True

    finally:
        await engine.cleanup()


@pytest.mark.asyncio
async def test_trajectory_recorder_with_real_engine(init_test_db):
    """Test trajectory recorder middleware with real ReAct engine."""
    # Create agent config
    agent_config = AgentConfig(
        agent_id="test-agent-recorder",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"})

    # Create ReAct engine
    engine = ReActEngine(config=agent_config)
    await engine.initialize()

    try:
        # Create trajectory recorder
        recorder = TrajectoryRecorder(engine=engine)

        # Execute with recording
        from agentcore.agent_runtime.models.agent_state import AgentExecutionState

        input_data = {"goal": "Calculate 10 + 5", "max_iterations": 5}
        state = AgentExecutionState(
            agent_id=agent_config.agent_id,
            status="running")

        result, steps = await recorder.execute_with_recording(input_data, state)

        # Verify execution
        assert result["completed"] is True
        assert "final_answer" in result

        # Verify steps recorded
        assert len(steps) > 0

        for step in steps:
            assert "iteration" in step.state
            assert "step_type" in step.action
            assert "content" in step.result
            assert isinstance(step.timestamp, datetime)

    finally:
        await engine.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_trajectory_persistence(init_test_db):
    """Test end-to-end trajectory collection and persistence."""
    # Create training job
    config = GRPOConfig()
    queries = [
        TrainingQuery(query=f"Q{i}", expected_outcome={"answer": f"A{i}"})
        for i in range(100)
    ]

    from agentcore.training.models import TrainingJob

    job = TrainingJob(
        agent_id="test-agent-e2e",
        config=config,
        training_data=queries,
        total_iterations=100,
        budget_usd=Decimal("50.00"))

    async with get_session() as session:
        job_db = await TrainingJobRepository.create(session, job)
        await session.commit()
        job_id = job_db.job_id

    try:
        # Create agent and collector
        agent_config = AgentConfig(
            agent_id="test-agent-e2e",
            philosophy="react",
            model_config={"provider": "test", "model": "test-model"})

        engine = ReActEngine(config=agent_config)
        await engine.initialize()

        try:
            collector = TrajectoryCollector(
                agent_config=agent_config,
                engine=engine,
                max_concurrent=2)

            # Collect trajectories
            query = "What is 7 + 8?"
            trajectories = await collector.collect_trajectories(
                job_id=job_id,
                query=query,
                n_trajectories=2)

            assert len(trajectories) == 2

            # Persist trajectories to database
            async with get_session() as session:
                for trajectory in trajectories:
                    traj_db = await TrajectoryRepository.create(session, trajectory)
                    assert traj_db.trajectory_id is not None

                await session.commit()

            # Retrieve trajectories from database
            async with get_session() as session:
                retrieved = await TrajectoryRepository.get_by_job(session, job_id)
                assert len(retrieved) == 2

                for traj_db in retrieved:
                    assert traj_db.job_id == job_id
                    assert traj_db.agent_id == "test-agent-e2e"
                    assert traj_db.query == query
                    assert len(traj_db.steps) > 0

        finally:
            await engine.cleanup()

    finally:
        # Cleanup
        async with get_session() as session:
            await TrainingJobRepository.delete(session, job_id)
            await session.commit()


@pytest.mark.asyncio
async def test_parallel_trajectory_generation_performance(init_test_db):
    """Test performance of parallel trajectory generation."""
    import time

    # Create agent config
    agent_config = AgentConfig(
        agent_id="test-agent-perf",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"})

    # Create ReAct engine
    engine = ReActEngine(config=agent_config)
    await engine.initialize()

    try:
        collector = TrajectoryCollector(
            agent_config=agent_config,
            engine=engine,
            max_concurrent=8,
            max_steps_per_trajectory=5)

        job_id = uuid4()
        query = "Test query for performance"

        # Measure baseline (single trajectory)
        baseline_start = time.time()
        baseline_trajectories = await collector.collect_trajectories(
            job_id=job_id,
            query=query,
            n_trajectories=1)
        baseline_time = time.time() - baseline_start

        assert len(baseline_trajectories) == 1

        # Measure parallel (8 trajectories)
        parallel_start = time.time()
        parallel_trajectories = await collector.collect_trajectories(
            job_id=job_id,
            query=query,
            n_trajectories=8)
        parallel_time = time.time() - parallel_start

        assert len(parallel_trajectories) == 8

        # Parallel execution should be less than 3x baseline time
        # (ideally ~8x faster with 8 concurrent, but allow generous margin for fast operations)
        # Use max of 3x baseline or 0.01s minimum to avoid timing noise
        max_acceptable_time = max(baseline_time * 3, 0.01)

        assert (
            parallel_time < max_acceptable_time
        ), f"Parallel time {parallel_time:.2f}, s exceeds 3x baseline {baseline_time:.2f}, s"

        # Log performance metrics
        print(f"\nPerformance Test Results:")
        print(f"  Baseline (1 trajectory): {baseline_time:.3f}, s")
        print(f"  Parallel (8 trajectories): {parallel_time:.3f}, s")
        print(f"  Speedup: {(baseline_time * 8) / parallel_time:.2f}, x")
        print(f"  Target: < {max_acceptable_time:.3f}, s (3x baseline or 10ms min)")

    finally:
        await engine.cleanup()


@pytest.mark.asyncio
async def test_trajectory_collection_with_timeouts(init_test_db):
    """Test trajectory collection handles timeouts gracefully."""
    from unittest.mock import AsyncMock, MagicMock
    import asyncio

    agent_config = AgentConfig(
        agent_id="test-agent-timeout",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"})

    # Create mock engine that sometimes times out
    from agentcore.agent_runtime.engines.base import PhilosophyEngine

    engine = MagicMock(spec=PhilosophyEngine)
    engine.agent_id = agent_config.agent_id
    engine.config = agent_config

    call_count = {"count": 0}

    async def mock_execute_with_timeout(input_data, state):
        call_count["count"] += 1
        if call_count["count"] % 2 == 0:
            await asyncio.sleep(10)  # Timeout

        return {
            "final_answer": "Success",
            "completed": True,
            "steps": [
                {"step_number": 1, "step_type": "final_answer", "content": "Success"}
            ],
        }

    engine.execute = AsyncMock(side_effect=mock_execute_with_timeout)

    collector = TrajectoryCollector(
        agent_config=agent_config,
        engine=engine,
        max_concurrent=4,
        timeout_seconds=0.5,  # Short timeout
    )

    job_id = uuid4()
    trajectories = await collector.collect_trajectories(
        job_id=job_id,
        query="Test query",
        n_trajectories=4)

    # Should get ~2 successful trajectories (odd indices)
    assert len(trajectories) >= 1
    assert len(trajectories) <= 2


@pytest.mark.asyncio
async def test_trajectory_steps_detail_capture(init_test_db):
    """Test that trajectory steps capture detailed execution information."""
    agent_config = AgentConfig(
        agent_id="test-agent-detail",
        philosophy="react",
        model_config={"provider": "test", "model": "test-model"})

    engine = ReActEngine(config=agent_config)
    await engine.initialize()

    try:
        collector = TrajectoryCollector(
            agent_config=agent_config,
            engine=engine)

        job_id = uuid4()
        trajectories = await collector.collect_trajectories(
            job_id=job_id,
            query="Calculate 3 + 4",
            n_trajectories=1)

        assert len(trajectories) == 1
        trajectory = trajectories[0]

        # Verify step details
        assert len(trajectory.steps) > 0

        for step in trajectory.steps:
            # State should contain iteration info
            assert "iteration" in step.state
            assert step.state["iteration"] > 0

            # Action should contain step type
            assert "step_type" in step.action
            assert step.action["step_type"] in [
                "thought",
                "action",
                "observation",
                "final_answer",
            ]

            # Result should contain content
            assert "content" in step.result

            # Timestamp should be timezone-aware
            assert isinstance(step.timestamp, datetime)
            assert step.timestamp.tzinfo is not None

            # Duration should be non-negative
            assert step.duration_ms >= 0

    finally:
        await engine.cleanup()
