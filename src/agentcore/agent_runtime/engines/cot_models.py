"""Data models for Chain-of-Thought (CoT) philosophy engine."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CoTStepType(str, Enum):
    """Types of steps in CoT reasoning chain."""

    STEP = "step"
    VERIFICATION = "verification"
    REFINEMENT = "refinement"
    CONCLUSION = "conclusion"


class CoTStep(BaseModel):
    """Single step in Chain-of-Thought reasoning."""

    step_number: int = Field(description="Step sequence number")
    step_type: CoTStepType = Field(description="Type of reasoning step")
    content: str = Field(description="Step content/reasoning")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this step",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Step timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class CoTExecutionContext(BaseModel):
    """Execution context for Chain-of-Thought agent."""

    agent_id: str = Field(description="Agent identifier")
    goal: str = Field(description="Agent goal or task")
    max_steps: int = Field(
        default=10,
        description="Maximum reasoning steps",
    )
    current_step: int = Field(
        default=0,
        description="Current step number",
    )
    steps: list[CoTStep] = Field(
        default_factory=list,
        description="Reasoning steps history",
    )
    context_window: list[str] = Field(
        default_factory=list,
        description="Recent context for memory management",
    )
    completed: bool = Field(
        default=False,
        description="Whether execution is complete",
    )
    final_conclusion: str | None = Field(
        default=None,
        description="Final conclusion if completed",
    )
    verification_enabled: bool = Field(
        default=True,
        description="Whether to verify reasoning steps",
    )


class CoTPromptTemplate(BaseModel):
    """Prompt templates for Chain-of-Thought reasoning."""

    system_prompt: str = Field(
        default="""You are a helpful AI assistant using Chain-of-Thought (CoT) reasoning.

Break down complex problems into clear, logical steps:
1. STEP: Articulate each reasoning step clearly
2. VERIFICATION: Check the validity of your reasoning
3. REFINEMENT: Adjust reasoning if needed
4. CONCLUSION: Provide the final answer

Think step-by-step, showing your work at each stage.""",
        description="System prompt for CoT agent",
    )

    step_prompt: str = Field(
        default="""Goal: {goal}

Previous steps:
{history}

Current context:
{context}

What is the next logical step in your reasoning?""",
        description="Prompt template for each step",
    )

    verification_prompt: str = Field(
        default="""Review the following reasoning step:
{step}

Is this reasoning valid and consistent with previous steps?
{previous_steps}

Provide verification or suggest refinements.""",
        description="Prompt for step verification",
    )


class LLMRequest(BaseModel):
    """Request to LLM provider."""

    prompt: str = Field(description="Prompt text")
    system_prompt: str = Field(description="System prompt")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum tokens in response",
    )
    model: str = Field(
        default="gpt-5",
        description="LLM model identifier",
    )


class LLMResponse(BaseModel):
    """Response from LLM provider."""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model used")
    tokens_used: int = Field(description="Total tokens consumed")
    finish_reason: str = Field(description="Reason for completion")
