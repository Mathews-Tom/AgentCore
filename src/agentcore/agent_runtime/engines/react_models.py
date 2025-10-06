"""Data models for ReAct philosophy engine."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReActStepType(str, Enum):
    """Types of steps in ReAct cycle."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class ReActStep(BaseModel):
    """Single step in ReAct reasoning cycle."""

    step_number: int = Field(description="Step sequence number")
    step_type: ReActStepType = Field(description="Type of step")
    content: str = Field(description="Step content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Step timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class ToolCall(BaseModel):
    """Tool invocation in action step."""

    tool_name: str = Field(description="Name of tool to call")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters",
    )
    call_id: str = Field(description="Unique call identifier")


class ToolResult(BaseModel):
    """Result from tool execution."""

    call_id: str = Field(description="Corresponding call identifier")
    success: bool = Field(description="Whether tool execution succeeded")
    result: Any = Field(default=None, description="Tool result data")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(description="Execution time in milliseconds")


class ReActExecutionContext(BaseModel):
    """Execution context for ReAct agent."""

    agent_id: str = Field(description="Agent identifier")
    goal: str = Field(description="Agent goal or task")
    max_iterations: int = Field(
        default=10,
        description="Maximum reasoning iterations",
    )
    current_iteration: int = Field(
        default=0,
        description="Current iteration number",
    )
    steps: list[ReActStep] = Field(
        default_factory=list,
        description="Execution steps history",
    )
    available_tools: list[str] = Field(
        default_factory=list,
        description="Available tool names",
    )
    completed: bool = Field(
        default=False,
        description="Whether execution is complete",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final answer if completed",
    )


class ReActPromptTemplate(BaseModel):
    """Prompt templates for ReAct reasoning."""

    system_prompt: str = Field(
        default="""You are a helpful AI assistant using the ReAct (Reasoning and Acting) framework.

For each step, you should:
1. THOUGHT: Reason about what to do next
2. ACTION: Choose a tool to use and specify parameters
3. OBSERVATION: Analyze the tool's result
4. Repeat until you have enough information to provide a FINAL_ANSWER

Available tools:
{tool_descriptions}

Format your responses as:
THOUGHT: <your reasoning>
ACTION: <tool_name>(<parameters>)
or
FINAL_ANSWER: <your conclusive answer>""",
        description="System prompt for ReAct agent",
    )

    step_prompt: str = Field(
        default="""Goal: {goal}

Previous steps:
{history}

What should you do next?""",
        description="Prompt template for each step",
    )
