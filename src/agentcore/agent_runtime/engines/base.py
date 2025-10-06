"""Base philosophy engine interface."""

from abc import ABC, abstractmethod
from typing import Any

from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState


class PhilosophyEngine(ABC):
    """Base class for philosophy-specific execution engines."""

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize philosophy engine.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        state: AgentExecutionState,
    ) -> dict[str, Any]:
        """
        Execute agent with philosophy-specific logic.

        Args:
            input_data: Input data for execution
            state: Current agent execution state

        Returns:
            Execution result
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize engine resources."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup engine resources."""
