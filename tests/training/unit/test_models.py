"""Unit tests for training Pydantic models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentcore.training.models import (
    GRPOConfig,
    PolicyCheckpoint,
    TrainingJob,
    TrainingJobStatus,
    TrainingQuery,
    Trajectory,
    TrajectoryStep)


class TestGRPOConfig:
    """Test suite for GRPOConfig model."""

    def test_default_values(self) -> None:
        """Test GRPOConfig initializes with correct defaults."""
        config = GRPOConfig()
        assert config.n_iterations == 1000
        assert config.batch_size == 16
        assert config.n_trajectories_per_query == 8
        assert config.learning_rate == 0.0001
        assert config.max_budget_usd == Decimal("100.00")
        assert config.checkpoint_interval == 10
        assert config.max_steps_per_trajectory == 20
        assert config.gamma == 0.99

    def test_custom_values(self) -> None:
        """Test GRPOConfig accepts custom values."""
        config = GRPOConfig(
            n_iterations=500,
            batch_size=32,
            n_trajectories_per_query=4,
            learning_rate=0.001,
            max_budget_usd=Decimal("50.00"),
            checkpoint_interval=5,
            max_steps_per_trajectory=15,
            gamma=0.95)
        assert config.n_iterations == 500
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.gamma == 0.95

    def test_budget_conversion_from_float(self) -> None:
        """Test budget converts from float to Decimal."""
        config = GRPOConfig(max_budget_usd=99.99)
        assert isinstance(config.max_budget_usd, Decimal)
        assert config.max_budget_usd == Decimal("99.99")

    def test_budget_conversion_from_string(self) -> None:
        """Test budget converts from string to Decimal."""
        config = GRPOConfig(max_budget_usd="150.50")
        assert isinstance(config.max_budget_usd, Decimal)
        assert config.max_budget_usd == Decimal("150.50")

    def test_validation_n_iterations_min(self) -> None:
        """Test validation fails for n_iterations < 1."""
        with pytest.raises(ValidationError) as exc_info:
            GRPOConfig(n_iterations=0)
        assert "n_iterations" in str(exc_info.value)

    def test_validation_learning_rate_negative(self) -> None:
        """Test validation fails for negative learning rate."""
        with pytest.raises(ValidationError) as exc_info:
            GRPOConfig(learning_rate=-0.001)
        assert "learning_rate" in str(exc_info.value)


class TestTrainingQuery:
    """Test suite for TrainingQuery model."""

    def test_valid_query(self) -> None:
        """Test TrainingQuery creates successfully with valid data."""
        query = TrainingQuery(
            query="What is the capital of France?",
            expected_outcome={"answer": "Paris", "correct": True})
        assert query.query == "What is the capital of France?"
        assert query.expected_outcome["answer"] == "Paris"

    def test_empty_query_fails(self) -> None:
        """Test validation fails for empty query string."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingQuery(
                query="",
                expected_outcome={"answer": "Paris", "correct": True})
        assert "query" in str(exc_info.value)

    def test_complex_expected_outcome(self) -> None:
        """Test TrainingQuery handles complex expected outcomes."""
        query = TrainingQuery(
            query="Analyze sentiment",
            expected_outcome={
                "sentiment": "positive",
                "confidence": 0.95,
                "entities": ["example", "test"],
                "metadata": {"source": "test"},
            })
        assert query.expected_outcome["confidence"] == 0.95
        assert len(query.expected_outcome["entities"]) == 2


class TestTrajectoryStep:
    """Test suite for TrajectoryStep model."""

    def test_valid_step(self) -> None:
        """Test TrajectoryStep creates successfully."""
        step = TrajectoryStep(
            state={"context": "test context", "memory": []},
            action={"type": "tool_call", "tool": "calculator", "params": {"x": 5}},
            result={"output": 10, "success": True},
            timestamp=datetime.now(),
            duration_ms=150)
        assert step.state["context"] == "test context"
        assert step.action["tool"] == "calculator"
        assert step.duration_ms == 150

    def test_negative_duration_fails(self) -> None:
        """Test validation fails for negative duration."""
        with pytest.raises(ValidationError) as exc_info:
            TrajectoryStep(
                state={},
                action={},
                result={},
                timestamp=datetime.now(),
                duration_ms=-100)
        assert "duration_ms" in str(exc_info.value)


class TestTrajectory:
    """Test suite for Trajectory model."""

    def test_trajectory_defaults(self) -> None:
        """Test Trajectory initializes with defaults."""
        job_id = uuid4()
        traj = Trajectory(
            job_id=job_id,
            agent_id="agent-123",
            query="Test query")
        assert traj.trajectory_id is None
        assert traj.job_id == job_id
        assert traj.agent_id == "agent-123"
        assert traj.reward == 0.0
        assert traj.normalized_reward == 0.0
        assert traj.advantage == 0.0
        assert len(traj.steps) == 0

    def test_trajectory_with_steps(self) -> None:
        """Test Trajectory with execution steps."""
        job_id = uuid4()
        step = TrajectoryStep(
            state={"x": 1},
            action={"y": 2},
            result={"z": 3},
            timestamp=datetime.now(),
            duration_ms=100)
        traj = Trajectory(
            job_id=job_id,
            agent_id="agent-123",
            query="Test query",
            steps=[step],
            reward=0.85,
            normalized_reward=0.5,
            advantage=0.3)
        assert len(traj.steps) == 1
        assert traj.reward == 0.85

    def test_trajectory_exceeds_max_steps(self) -> None:
        """Test validation fails when trajectory exceeds 100 steps."""
        job_id = uuid4()
        steps = [
            TrajectoryStep(
                state={},
                action={},
                result={},
                timestamp=datetime.now(),
                duration_ms=10)
            for _ in range(101)
        ]
        with pytest.raises(ValidationError) as exc_info:
            Trajectory(
                job_id=job_id,
                agent_id="agent-123",
                query="Test",
                steps=steps)
        assert "steps" in str(exc_info.value)
        assert "100 steps" in str(exc_info.value)


class TestTrainingJob:
    """Test suite for TrainingJob model."""

    def test_training_job_defaults(self) -> None:
        """Test TrainingJob initializes with defaults."""
        config = GRPOConfig()
        queries = [
            TrainingQuery(
                query=f"Query {i}",
                expected_outcome={"answer": f"Answer {i}", "correct": True})
            for i in range(100)
        ]
        job = TrainingJob(
            agent_id="agent-123",
            config=config,
            training_data=queries,
            total_iterations=1000,
            budget_usd=Decimal("100.00"))
        assert job.job_id is None
        assert job.agent_id == "agent-123"
        assert job.status == TrainingJobStatus.QUEUED
        assert job.current_iteration == 0
        assert job.cost_usd == Decimal("0.00")
        assert len(job.training_data) == 100

    def test_training_job_insufficient_data(self) -> None:
        """Test validation fails with < 100 training queries."""
        config = GRPOConfig()
        queries = [
            TrainingQuery(
                query=f"Query {i}",
                expected_outcome={"answer": "test", "correct": True})
            for i in range(50)
        ]
        with pytest.raises(ValidationError) as exc_info:
            TrainingJob(
                agent_id="agent-123",
                config=config,
                training_data=queries,
                total_iterations=1000,
                budget_usd=Decimal("100.00"))
        assert "training_data" in str(exc_info.value)

    def test_training_job_cost_conversion(self) -> None:
        """Test cost converts to Decimal correctly."""
        config = GRPOConfig()
        queries = [
            TrainingQuery(query=f"Q{i}", expected_outcome={"a": True})
            for i in range(100)
        ]
        job = TrainingJob(
            agent_id="agent-123",
            config=config,
            training_data=queries,
            total_iterations=1000,
            budget_usd="200.00",
            cost_usd=25.50)
        assert isinstance(job.cost_usd, Decimal)
        assert job.cost_usd == Decimal("25.50")
        assert isinstance(job.budget_usd, Decimal)
        assert job.budget_usd == Decimal("200.00")

    def test_training_job_with_all_fields(self) -> None:
        """Test TrainingJob with all optional fields populated."""
        config = GRPOConfig()
        queries = [
            TrainingQuery(query=f"Q{i}", expected_outcome={"a": True})
            for i in range(100)
        ]
        job_id = uuid4()
        checkpoint_id = uuid4()
        created = datetime.now()

        job = TrainingJob(
            job_id=job_id,
            agent_id="agent-123",
            status=TrainingJobStatus.RUNNING,
            config=config,
            training_data=queries,
            current_iteration=50,
            total_iterations=1000,
            metrics={"loss": 0.5, "reward": 0.8},
            cost_usd=Decimal("15.00"),
            budget_usd=Decimal("100.00"),
            best_checkpoint_id=checkpoint_id,
            created_at=created)
        assert job.job_id == job_id
        assert job.status == TrainingJobStatus.RUNNING
        assert job.current_iteration == 50
        assert job.best_checkpoint_id == checkpoint_id


class TestPolicyCheckpoint:
    """Test suite for PolicyCheckpoint model."""

    def test_checkpoint_with_policy_data(self) -> None:
        """Test PolicyCheckpoint with in-database policy data."""
        job_id = uuid4()
        checkpoint = PolicyCheckpoint(
            agent_id="agent-123",
            job_id=job_id,
            iteration=10,
            policy_data={
                "prompt": "You are a helpful assistant",
                "examples": ["example1", "example2"],
            },
            validation_score=0.85,
            metrics={"accuracy": 0.9, "loss": 0.2})
        assert checkpoint.agent_id == "agent-123"
        assert checkpoint.iteration == 10
        assert checkpoint.policy_data["prompt"] == "You are a helpful assistant"
        assert checkpoint.policy_s3_path is None

    def test_checkpoint_with_s3_path(self) -> None:
        """Test PolicyCheckpoint with S3 storage path."""
        job_id = uuid4()
        checkpoint = PolicyCheckpoint(
            agent_id="agent-123",
            job_id=job_id,
            iteration=20,
            policy_s3_path="s3://bucket/checkpoints/agent-123/iter-20.pkl",
            validation_score=0.90)
        assert checkpoint.policy_s3_path.startswith("s3://")
        assert checkpoint.policy_data is None

    def test_checkpoint_defaults(self) -> None:
        """Test PolicyCheckpoint defaults."""
        job_id = uuid4()
        checkpoint = PolicyCheckpoint(
            agent_id="agent-123",
            job_id=job_id,
            iteration=0)
        assert checkpoint.checkpoint_id is None
        assert checkpoint.validation_score == 0.0
        assert len(checkpoint.metrics) == 0
        assert checkpoint.created_at is None
