"""
Context Chain Utility

Multi-stage workflow orchestration with context accumulation and lineage tracking.
Implements A2A-018: Context Engineering Patterns.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContextStage(BaseModel):
    """Single stage in a context chain."""
    stage_id: str = Field(..., description="Unique stage identifier")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    agent_id: Optional[str] = Field(None, description="Agent that executed this stage")
    input_context: Dict[str, Any] = Field(default_factory=dict, description="Input context for this stage")
    output_context: Dict[str, Any] = Field(default_factory=dict, description="Output context from this stage")
    summary: Optional[str] = Field(None, description="Summary of this stage's transformation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Stage execution timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContextChain:
    """
    Multi-stage workflow orchestration with context accumulation.

    Tracks context transformations across multiple stages,
    maintains lineage, and supports input transformations.

    Example:
        ```python
        chain = ContextChain(initial_context={"user_query": "What's the weather?"})

        # Add stages
        chain.add_stage(
            stage_id="parse",
            output_context={"intent": "weather", "location": "current"},
            summary="Parsed user intent"
        )

        chain.add_stage(
            stage_id="fetch_data",
            output_context={"temperature": 72, "condition": "sunny"},
            summary="Fetched weather data"
        )

        # Get accumulated context
        context = chain.get_accumulated_context()
        # {"user_query": "What's the weather?", "intent": "weather", "location": "current", "temperature": 72, "condition": "sunny"}

        # Get lineage
        lineage = chain.get_lineage()
        # ["parse", "fetch_data"]
        ```
    """

    def __init__(self, initial_context: Optional[Dict[str, Any]] = None, chain_id: Optional[str] = None):
        """
        Initialize context chain.

        Args:
            initial_context: Initial context dictionary
            chain_id: Optional chain identifier
        """
        self.chain_id = chain_id or f"chain_{datetime.utcnow().timestamp()}"
        self.stages: List[ContextStage] = []
        self.accumulated_context = initial_context or {}
        self.input_transforms: Dict[str, Callable] = {}

    def add_stage(
        self,
        stage_id: str,
        output_context: Dict[str, Any],
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        summary: Optional[str] = None,
        input_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ContextChain":
        """
        Add a stage to the context chain.

        Args:
            stage_id: Unique identifier for this stage
            output_context: Output context produced by this stage
            task_id: Optional associated task ID
            agent_id: Optional agent that executed this stage
            summary: Optional summary of the stage's transformation
            input_transform: Optional function to transform accumulated context before this stage
            metadata: Optional additional metadata

        Returns:
            Self for method chaining

        Raises:
            ValueError: If stage_id already exists
        """
        if any(s.stage_id == stage_id for s in self.stages):
            raise ValueError(f"Stage with id '{stage_id}' already exists")

        # Apply input transform if provided
        input_context = self.accumulated_context.copy()
        if input_transform:
            try:
                input_context = input_transform(input_context)
                self.input_transforms[stage_id] = input_transform
            except Exception as e:
                logger.error(f"Input transform failed for stage '{stage_id}': {e}")
                raise ValueError(f"Input transform failed: {e}")

        # Create stage
        stage = ContextStage(
            stage_id=stage_id,
            task_id=task_id,
            agent_id=agent_id,
            input_context=input_context,
            output_context=output_context,
            summary=summary,
            metadata=metadata or {},
        )

        self.stages.append(stage)

        # Accumulate context (merge output into accumulated)
        self.accumulated_context.update(output_context)

        logger.info(f"Added stage '{stage_id}' to chain '{self.chain_id}'")
        return self

    def get_accumulated_context(self) -> Dict[str, Any]:
        """
        Get the accumulated context from all stages.

        Returns:
            Dictionary containing all accumulated context
        """
        return self.accumulated_context.copy()

    def get_stage(self, stage_id: str) -> Optional[ContextStage]:
        """
        Get a specific stage by ID.

        Args:
            stage_id: Stage identifier

        Returns:
            ContextStage if found, None otherwise
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_lineage(self) -> List[str]:
        """
        Get the lineage of context transformations (ordered list of stage IDs).

        Returns:
            List of stage IDs in execution order
        """
        return [stage.stage_id for stage in self.stages]

    def get_task_lineage(self) -> List[str]:
        """
        Get the lineage of task IDs (for TaskArtifact.context_lineage).

        Returns:
            List of task IDs in execution order (excluding None values)
        """
        return [stage.task_id for stage in self.stages if stage.task_id]

    def generate_summary(self) -> str:
        """
        Generate a summary of the entire context chain.

        Returns:
            Human-readable summary of all transformations
        """
        if not self.stages:
            return "Empty context chain"

        summaries = []
        for i, stage in enumerate(self.stages, 1):
            stage_summary = stage.summary or f"Stage {stage.stage_id}"
            summaries.append(f"{i}. {stage_summary}")

        return "Context Chain:\n" + "\n".join(summaries)

    def export_for_artifact(self) -> Dict[str, Any]:
        """
        Export context chain data for TaskArtifact.

        Returns:
            Dictionary with context_lineage and context_summary
        """
        return {
            "context_lineage": self.get_task_lineage(),
            "context_summary": self.generate_summary(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Export full context chain to dictionary.

        Returns:
            Dictionary representation of the context chain
        """
        return {
            "chain_id": self.chain_id,
            "stages": [stage.model_dump() for stage in self.stages],
            "accumulated_context": self.accumulated_context,
            "lineage": self.get_lineage(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextChain":
        """
        Create context chain from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ContextChain instance
        """
        chain = cls(chain_id=data.get("chain_id"))
        chain.accumulated_context = data.get("accumulated_context", {})

        for stage_data in data.get("stages", []):
            stage = ContextStage(**stage_data)
            chain.stages.append(stage)

        return chain

    def __len__(self) -> int:
        """Get number of stages in the chain."""
        return len(self.stages)

    def __repr__(self) -> str:
        """String representation."""
        return f"ContextChain(chain_id='{self.chain_id}', stages={len(self.stages)})"


# Example patterns for common use cases

def calendar_analysis_pattern(initial_query: str) -> ContextChain:
    """
    Example pattern for calendar analysis workflow.

    Args:
        initial_query: User's calendar query

    Returns:
        Pre-configured ContextChain for calendar analysis
    """
    return ContextChain(
        initial_context={"user_query": initial_query, "domain": "calendar"},
        chain_id="calendar_analysis"
    )


def research_synthesis_pattern(research_topic: str) -> ContextChain:
    """
    Example pattern for research synthesis workflow.

    Args:
        research_topic: Topic to research

    Returns:
        Pre-configured ContextChain for research synthesis
    """
    return ContextChain(
        initial_context={"topic": research_topic, "sources": [], "findings": []},
        chain_id="research_synthesis"
    )


def multi_step_reasoning_pattern(problem: str) -> ContextChain:
    """
    Example pattern for multi-step reasoning workflow.

    Args:
        problem: Problem to solve

    Returns:
        Pre-configured ContextChain for multi-step reasoning
    """
    return ContextChain(
        initial_context={"problem": problem, "steps": [], "solution": None},
        chain_id="multi_step_reasoning"
    )
