"""
Middleware for recording agent execution trajectories.

Wraps agent engine execution to capture detailed trajectory information.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from agentcore.agent_runtime.engines.base import PhilosophyEngine
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.training.models import TrajectoryStep

logger = structlog.get_logger()


class TrajectoryRecorder:
    """
    Middleware wrapper for recording agent execution trajectories.

    Wraps philosophy engine execution to capture detailed step-by-step
    execution information for training purposes.
    """

    def __init__(self, engine: PhilosophyEngine) -> None:
        """
        Initialize trajectory recorder.

        Args:
            engine: Philosophy engine to wrap
        """
        self.engine = engine
        self.steps: list[TrajectoryStep] = []
        self.start_time: datetime | None = None
        self.recording = False

        logger.info(
            "trajectory_recorder_initialized",
            agent_id=engine.agent_id,
            engine_type=type(engine).__name__,
        )

    async def execute_with_recording(
        self,
        input_data: dict[str, Any],
        state: AgentExecutionState,
    ) -> tuple[dict[str, Any], list[TrajectoryStep]]:
        """
        Execute agent with trajectory recording.

        Args:
            input_data: Input data for execution
            state: Agent execution state

        Returns:
            Tuple of (execution result, recorded steps)
        """
        self.steps = []
        self.start_time = datetime.now(UTC)
        self.recording = True

        logger.info(
            "trajectory_recording_start",
            agent_id=self.engine.agent_id,
            goal=input_data.get("goal", ""),
        )

        try:
            # Execute engine
            result = await self.engine.execute(input_data, state)

            # Extract steps from result
            self._extract_steps_from_result(result)

            logger.info(
                "trajectory_recording_complete",
                agent_id=self.engine.agent_id,
                steps_recorded=len(self.steps),
            )

            return result, self.steps

        except Exception as e:
            logger.error(
                "trajectory_recording_error",
                agent_id=self.engine.agent_id,
                error=str(e),
            )
            # Still return partial steps if available
            return {"error": str(e), "completed": False}, self.steps

        finally:
            self.recording = False

    def _extract_steps_from_result(self, result: dict[str, Any]) -> None:
        """
        Extract trajectory steps from engine execution result.

        Args:
            result: Engine execution result
        """
        raw_steps = result.get("steps", [])

        for i, raw_step in enumerate(raw_steps):
            # Calculate timestamp (approximate based on order)
            if self.start_time:
                # Assume even distribution of time across steps
                step_offset_ms = i * 100  # 100ms per step (approximation)
                timestamp = self.start_time
            else:
                timestamp = datetime.now(UTC)

            # Create trajectory step
            step = TrajectoryStep(
                state={
                    "iteration": raw_step.get("step_number", i + 1),
                    "agent_id": self.engine.agent_id,
                },
                action={
                    "step_type": raw_step.get("step_type", "unknown"),
                    "content": raw_step.get("content", "")[
                        :200
                    ],  # Truncate for storage
                },
                result={
                    "content": raw_step.get("content", ""),
                    "success": True,  # Steps in result are considered successful
                },
                timestamp=timestamp,
                duration_ms=100,  # Approximation
            )

            self.steps.append(step)

    def get_recorded_steps(self) -> list[TrajectoryStep]:
        """
        Get recorded trajectory steps.

        Returns:
            List of recorded steps
        """
        return self.steps.copy()

    def clear_recording(self) -> None:
        """Clear recorded steps."""
        self.steps = []
        self.start_time = None
        self.recording = False
