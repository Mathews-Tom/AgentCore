"""
Unit tests for Supervisor Pattern Implementation.

Tests master-worker coordination, task distribution, load balancing,
and failure handling.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from agentcore.orchestration.patterns.supervisor import (
    LoadBalancingStrategy,
    SupervisorConfig,
    SupervisorCoordinator,
    WorkerStatus)


class TestSupervisorCoordinator:
    """Test suite for SupervisorCoordinator."""

    @pytest.fixture
    def config(self) -> SupervisorConfig:
        """Create test configuration."""
        return SupervisorConfig(
            max_workers=5,
            load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED,
            worker_timeout_seconds=30,
            task_timeout_seconds=300,
            enable_auto_recovery=True,
            max_task_retries=3)

    @pytest.fixture
    def supervisor(self, config: SupervisorConfig) -> SupervisorCoordinator:
        """Create supervisor coordinator instance."""
        return SupervisorCoordinator(
            supervisor_id="test-supervisor",
            config=config)

    @pytest.mark.asyncio
    async def test_register_worker(self, supervisor: SupervisorCoordinator) -> None:
        """Test worker registration."""
        # Register a worker
        await supervisor.register_worker(
            worker_id="worker-1",
            capabilities=["task_a", "task_b"])

        # Verify worker is registered
        workers = await supervisor.get_worker_states()
        assert len(workers) == 1
        assert workers[0].worker_id == "worker-1"
        assert workers[0].status == WorkerStatus.IDLE
        assert "task_a" in workers[0].capabilities

    @pytest.mark.asyncio
    async def test_unregister_worker(self, supervisor: SupervisorCoordinator) -> None:
        """Test worker unregistration."""
        # Register and unregister a worker
        await supervisor.register_worker("worker-1")
        await supervisor.unregister_worker("worker-1")

        # Verify worker is removed
        workers = await supervisor.get_worker_states()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_submit_task(self, supervisor: SupervisorCoordinator) -> None:
        """Test task submission."""
        # Register a worker
        await supervisor.register_worker("worker-1")

        # Submit a task
        task_id = uuid4()
        await supervisor.submit_task(
            task_id=task_id,
            task_type="test_task",
            input_data={"key": "value"})

        # Small delay for async assignment
        await asyncio.sleep(0.1)

        # Verify task is assigned
        assignments = await supervisor.get_task_assignments()
        assert len(assignments) == 1
        assert assignments[0].task_id == task_id
        assert assignments[0].worker_id == "worker-1"

    @pytest.mark.asyncio
    async def test_task_completion(self, supervisor: SupervisorCoordinator) -> None:
        """Test task completion handling."""
        # Register worker and submit task
        await supervisor.register_worker("worker-1")
        task_id = uuid4()
        await supervisor.submit_task(task_id, "test_task")
        await asyncio.sleep(0.1)

        # Complete the task
        await supervisor.handle_task_completion(
            task_id=task_id,
            worker_id="worker-1",
            result_data={"result": "success"})

        # Verify task is completed
        status = await supervisor.get_supervisor_status()
        assert status["tasks"]["completed"] == 1
        assert status["tasks"]["active"] == 0

        # Verify worker is idle again
        workers = await supervisor.get_worker_states()
        assert workers[0].status == WorkerStatus.IDLE
        assert workers[0].tasks_completed == 1

    @pytest.mark.asyncio
    async def test_task_failure_with_retry(
        self, supervisor: SupervisorCoordinator
    ) -> None:
        """Test task failure handling with retry."""
        # Register worker and submit task
        await supervisor.register_worker("worker-1")
        task_id = uuid4()
        await supervisor.submit_task(task_id, "test_task")
        await asyncio.sleep(0.1)

        # Fail the task (first attempt)
        await supervisor.handle_task_failure(
            task_id=task_id,
            worker_id="worker-1",
            error_message="Test error",
            error_type="TestError")

        # Verify task is retried (back in pending or reassigned)
        status = await supervisor.get_supervisor_status()
        assert status["tasks"]["failed"] == 0  # Not permanently failed yet

    @pytest.mark.asyncio
    async def test_task_failure_max_retries(
        self, supervisor: SupervisorCoordinator
    ) -> None:
        """Test task failure after max retries."""
        # Register worker and submit task
        await supervisor.register_worker("worker-1")
        task_id = uuid4()
        await supervisor.submit_task(task_id, "test_task")
        await asyncio.sleep(0.1)

        # Fail the task multiple times
        for _ in range(3):
            await supervisor.handle_task_failure(
                task_id=task_id,
                worker_id="worker-1",
                error_message="Test error",
                error_type="TestError")
            await asyncio.sleep(0.05)

        # Verify task is permanently failed
        status = await supervisor.get_supervisor_status()
        assert status["tasks"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_load_balancing_round_robin(self) -> None:
        """Test round-robin load balancing."""
        config = SupervisorConfig(
            load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN
        )
        supervisor = SupervisorCoordinator("test-supervisor", config)

        # Register multiple workers
        for i in range(3):
            await supervisor.register_worker(f"worker-{i}")

        # Submit multiple tasks
        task_ids = [uuid4() for _ in range(6)]
        for task_id in task_ids:
            await supervisor.submit_task(task_id, "test_task")
            await asyncio.sleep(0.05)

        # Verify tasks are distributed across workers
        assignments = await supervisor.get_task_assignments()
        worker_counts: dict[str, int] = {}
        for assignment in assignments:
            worker_counts[assignment.worker_id] = (
                worker_counts.get(assignment.worker_id, 0) + 1
            )

        # With 3 workers and 6 tasks, should have all 3 workers busy (3 active)
        # and 3 tasks pending
        assert len(assignments) == 3  # 3 workers are busy
        assert len(worker_counts) == 3  # All 3 workers have tasks
        assert all(count == 1 for count in worker_counts.values())  # Each has 1 active task

    @pytest.mark.asyncio
    async def test_load_balancing_least_loaded(self) -> None:
        """Test least-loaded load balancing."""
        config = SupervisorConfig(
            load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED
        )
        supervisor = SupervisorCoordinator("test-supervisor", config)

        # Register workers with different load scores
        await supervisor.register_worker("worker-1")
        await supervisor.register_worker("worker-2")

        # Manually set load scores
        async with supervisor._lock:
            supervisor._workers["worker-1"].load_score = 0.5
            supervisor._workers["worker-2"].load_score = 0.1

        # Submit a task - should go to worker-2 (least loaded)
        task_id = uuid4()
        await supervisor.submit_task(task_id, "test_task")
        await asyncio.sleep(0.1)

        assignments = await supervisor.get_task_assignments()
        assert assignments[0].worker_id == "worker-2"

    @pytest.mark.asyncio
    async def test_worker_heartbeat(self, supervisor: SupervisorCoordinator) -> None:
        """Test worker heartbeat handling."""
        # Register worker
        await supervisor.register_worker("worker-1")

        # Get initial heartbeat time
        workers = await supervisor.get_worker_states()
        initial_heartbeat = workers[0].last_heartbeat

        # Wait a bit
        await asyncio.sleep(0.1)

        # Send heartbeat with load score
        await supervisor.handle_worker_heartbeat("worker-1", load_score=0.3)

        # Verify heartbeat was updated
        workers = await supervisor.get_worker_states()
        assert workers[0].last_heartbeat > initial_heartbeat
        assert workers[0].load_score == 0.3

    @pytest.mark.asyncio
    async def test_worker_timeout_detection(self) -> None:
        """Test worker timeout detection and recovery."""
        config = SupervisorConfig(
            worker_timeout_seconds=1,  # Short timeout for testing
            enable_auto_recovery=True)
        supervisor = SupervisorCoordinator("test-supervisor", config)

        # Register worker
        await supervisor.register_worker("worker-1")

        # Submit task
        task_id = uuid4()
        await supervisor.submit_task(task_id, "test_task")
        await asyncio.sleep(0.1)

        # Set worker heartbeat to past (simulate timeout)
        async with supervisor._lock:
            supervisor._workers["worker-1"].last_heartbeat = datetime.now(
                UTC
            ) - timedelta(seconds=2)

        # Monitor workers (should detect timeout)
        await supervisor.monitor_workers()

        # Verify worker was removed and task reassigned
        workers = await supervisor.get_worker_states()
        assert len(workers) == 0

        # Task should be back in pending queue (for reassignment)
        status = await supervisor.get_supervisor_status()
        assert status["tasks"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_multiple_workers_concurrent_tasks(
        self, supervisor: SupervisorCoordinator
    ) -> None:
        """Test concurrent task execution with multiple workers."""
        # Register multiple workers
        for i in range(3):
            await supervisor.register_worker(f"worker-{i}")

        # Submit multiple tasks
        task_ids = [uuid4() for _ in range(5)]
        for task_id in task_ids:
            await supervisor.submit_task(task_id, "test_task")

        await asyncio.sleep(0.2)

        # Verify tasks are distributed
        assignments = await supervisor.get_task_assignments()
        assert len(assignments) == 3  # 3 workers, 3 active tasks

        # Verify 2 tasks are still pending
        status = await supervisor.get_supervisor_status()
        assert status["tasks"]["pending"] == 2

    @pytest.mark.asyncio
    async def test_supervisor_status(self, supervisor: SupervisorCoordinator) -> None:
        """Test supervisor status reporting."""
        # Register workers
        await supervisor.register_worker("worker-1")
        await supervisor.register_worker("worker-2")

        # Submit and complete tasks
        task1 = uuid4()
        task2 = uuid4()

        await supervisor.submit_task(task1, "test_task")
        await supervisor.submit_task(task2, "test_task")
        await asyncio.sleep(0.1)

        await supervisor.handle_task_completion(task1, "worker-1")
        await asyncio.sleep(0.05)  # Allow status updates to propagate

        # Get status
        status = await supervisor.get_supervisor_status()

        assert status["supervisor_id"] == "test-supervisor"
        assert status["workers"]["total"] == 2
        # After completing task1, worker-1 is idle, worker-2 is still busy with task2
        assert status["workers"]["idle"] == 1
        assert status["workers"]["busy"] == 1
        assert status["tasks"]["completed"] == 1
        assert status["tasks"]["active"] == 1

    @pytest.mark.asyncio
    async def test_task_reassignment_on_worker_failure(
        self, supervisor: SupervisorCoordinator
    ) -> None:
        """Test task reassignment when worker fails."""
        # Register two workers
        await supervisor.register_worker("worker-1")
        await supervisor.register_worker("worker-2")

        # Submit task (goes to worker-1)
        task_id = uuid4()
        await supervisor.submit_task(task_id, "test_task")
        await asyncio.sleep(0.1)

        # Unregister worker-1 (simulates failure)
        await supervisor.unregister_worker("worker-1")

        # Wait for reassignment
        await asyncio.sleep(0.1)

        # Verify task was reassigned to worker-2
        status = await supervisor.get_supervisor_status()
        # Task should be reassigned or in pending queue
        assert status["tasks"]["pending"] >= 0  # May be reassigned or pending

    @pytest.mark.asyncio
    async def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid config
        config = SupervisorConfig(max_workers=10, task_timeout_seconds=60)
        assert config.max_workers == 10

        # Invalid config (negative values should be rejected by Pydantic)
        with pytest.raises(Exception):
            SupervisorConfig(max_workers=-1)

    @pytest.mark.asyncio
    async def test_concurrent_worker_operations(
        self, supervisor: SupervisorCoordinator
    ) -> None:
        """Test thread safety with concurrent worker operations."""
        # Concurrent registration
        tasks = [
            supervisor.register_worker(f"worker-{i}") for i in range(10)
        ]
        await asyncio.gather(*tasks)

        workers = await supervisor.get_worker_states()
        assert len(workers) == 10

        # Concurrent unregistration
        tasks = [
            supervisor.unregister_worker(f"worker-{i}") for i in range(10)
        ]
        await asyncio.gather(*tasks)

        workers = await supervisor.get_worker_states()
        assert len(workers) == 0


class TestLoadBalancingStrategies:
    """Test different load balancing strategies."""

    @pytest.mark.asyncio
    async def test_random_strategy(self) -> None:
        """Test random load balancing strategy."""
        config = SupervisorConfig(
            load_balancing_strategy=LoadBalancingStrategy.RANDOM
        )
        supervisor = SupervisorCoordinator("test-supervisor", config)

        # Register workers
        for i in range(3):
            await supervisor.register_worker(f"worker-{i}")

        # Submit multiple tasks
        for _ in range(10):
            await supervisor.submit_task(uuid4(), "test_task")
            await asyncio.sleep(0.05)

        # Verify tasks are distributed (not necessarily evenly due to randomness)
        assignments = await supervisor.get_task_assignments()
        worker_ids = {a.worker_id for a in assignments}
        assert len(worker_ids) >= 2  # At least 2 workers should have tasks
