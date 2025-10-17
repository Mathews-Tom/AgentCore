"""
Trajectory collection system for training infrastructure.

Implements async parallel trajectory generation with middleware integration.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

from agentcore.agent_runtime.engines.base import PhilosophyEngine
from agentcore.agent_runtime.models.agent_config import AgentConfig
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.training.models import Trajectory, TrajectoryStep

logger = structlog.get_logger()


class TrajectoryCollector:
    """
    Collects agent execution trajectories for training.

    Generates multiple trajectories in parallel for a given query-agent pair,
    capturing complete execution state at each step.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        engine: PhilosophyEngine,
        max_concurrent: int = 8,
        max_steps_per_trajectory: int = 20,
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        Initialize trajectory collector.

        Args:
            agent_config: Agent configuration
            engine: Philosophy engine for agent execution
            max_concurrent: Maximum concurrent trajectory generations
            max_steps_per_trajectory: Maximum steps per trajectory
            timeout_seconds: Timeout for single trajectory execution
        """
        self.agent_config = agent_config
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.timeout_seconds = timeout_seconds

        logger.info(
            "trajectory_collector_initialized",
            agent_id=agent_config.agent_id,
            max_concurrent=max_concurrent,
        )

    async def collect_trajectories(
        self,
        job_id: UUID,
        query: str,
        n_trajectories: int = 8,
    ) -> list[Trajectory]:
        """
        Generate multiple trajectories in parallel for a query.

        Args:
            job_id: Training job identifier
            query: Query to execute
            n_trajectories: Number of trajectories to generate

        Returns:
            List of collected trajectories
        """
        logger.info(
            "collect_trajectories_start",
            agent_id=self.agent_config.agent_id,
            job_id=str(job_id),
            query=query,
            n_trajectories=n_trajectories,
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create tasks for parallel execution
        tasks = [
            self._collect_single_trajectory(
                job_id=job_id,
                query=query,
                trajectory_index=i,
                semaphore=semaphore,
            )
            for i in range(n_trajectories)
        ]

        # Execute in parallel
        start_time = datetime.now(UTC)
        trajectories = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now(UTC)

        # Filter out exceptions and log errors
        valid_trajectories: list[Trajectory] = []
        failed_count = 0

        for i, result in enumerate(trajectories):
            if isinstance(result, Exception):
                logger.error(
                    "trajectory_collection_failed",
                    agent_id=self.agent_config.agent_id,
                    trajectory_index=i,
                    error=str(result),
                )
                failed_count += 1
            elif isinstance(result, Trajectory):
                valid_trajectories.append(result)

        total_time = (end_time - start_time).total_seconds()

        logger.info(
            "collect_trajectories_complete",
            agent_id=self.agent_config.agent_id,
            job_id=str(job_id),
            total_trajectories=len(valid_trajectories),
            failed_trajectories=failed_count,
            total_time_seconds=total_time,
        )

        return valid_trajectories

    async def _collect_single_trajectory(
        self,
        job_id: UUID,
        query: str,
        trajectory_index: int,
        semaphore: asyncio.Semaphore,
    ) -> Trajectory:
        """
        Collect a single trajectory with timeout and error handling.

        Args:
            job_id: Training job identifier
            query: Query to execute
            trajectory_index: Index for logging
            semaphore: Semaphore for concurrency control

        Returns:
            Collected trajectory

        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: If execution fails
        """
        async with semaphore:
            logger.info(
                "trajectory_collection_start",
                agent_id=self.agent_config.agent_id,
                job_id=str(job_id),
                trajectory_index=trajectory_index,
            )

            start_time = datetime.now(UTC)

            try:
                # Execute with timeout
                trajectory = await asyncio.wait_for(
                    self._execute_trajectory(job_id, query),
                    timeout=self.timeout_seconds,
                )

                end_time = datetime.now(UTC)
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

                trajectory.execution_time_ms = execution_time_ms

                logger.info(
                    "trajectory_collection_success",
                    agent_id=self.agent_config.agent_id,
                    trajectory_index=trajectory_index,
                    execution_time_ms=execution_time_ms,
                    steps=len(trajectory.steps),
                )

                return trajectory

            except asyncio.TimeoutError:
                logger.warning(
                    "trajectory_collection_timeout",
                    agent_id=self.agent_config.agent_id,
                    trajectory_index=trajectory_index,
                    timeout_seconds=self.timeout_seconds,
                )
                raise TimeoutError(
                    f"Trajectory collection timed out after {self.timeout_seconds}s"
                )

            except Exception as e:
                logger.error(
                    "trajectory_collection_error",
                    agent_id=self.agent_config.agent_id,
                    trajectory_index=trajectory_index,
                    error=str(e),
                )
                raise

    async def _execute_trajectory(
        self,
        job_id: UUID,
        query: str,
    ) -> Trajectory:
        """
        Execute agent and capture trajectory.

        Args:
            job_id: Training job identifier
            query: Query to execute

        Returns:
            Captured trajectory with execution steps
        """
        # Create execution state
        state = AgentExecutionState(
            agent_id=self.agent_config.agent_id,
            status="running",
        )

        # Prepare input
        input_data = {
            "goal": query,
            "max_iterations": self.max_steps_per_trajectory,
        }

        # Execute engine
        result = await self.engine.execute(input_data, state)

        # Extract trajectory steps
        steps: list[TrajectoryStep] = []
        raw_steps = result.get("steps", [])

        for raw_step in raw_steps:
            # Convert engine step to trajectory step
            step = TrajectoryStep(
                state={"iteration": raw_step.get("step_number", 0)},
                action={"step_type": raw_step.get("step_type", "unknown")},
                result={"content": raw_step.get("content", "")},
                timestamp=datetime.now(UTC),
                duration_ms=0,  # Engine doesn't track per-step duration yet
            )
            steps.append(step)

        # Determine success based on completion
        success = result.get("completed", False)

        # Create trajectory
        trajectory = Trajectory(
            job_id=job_id,
            agent_id=self.agent_config.agent_id,
            query=query,
            steps=steps,
            success=success,
        )

        return trajectory
