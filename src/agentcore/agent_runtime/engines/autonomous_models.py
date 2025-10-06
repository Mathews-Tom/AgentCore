"""Data models for Autonomous Agent philosophy engine."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GoalStatus(str, Enum):
    """Status of agent goals."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class GoalPriority(str, Enum):
    """Priority levels for goals."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Goal(BaseModel):
    """Agent goal definition."""

    goal_id: str = Field(description="Unique goal identifier")
    description: str = Field(description="Goal description")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM)
    status: GoalStatus = Field(default=GoalStatus.PENDING)
    success_criteria: dict[str, Any] = Field(
        default_factory=dict,
        description="Criteria for goal completion",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    parent_goal_id: str | None = Field(
        default=None,
        description="Parent goal for hierarchical goals",
    )
    sub_goals: list[str] = Field(
        default_factory=list,
        description="List of sub-goal IDs",
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Goal completion progress",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """Agent decision record."""

    decision_id: str = Field(description="Unique decision identifier")
    goal_id: str = Field(description="Associated goal ID")
    description: str = Field(description="Decision description")
    rationale: str = Field(description="Decision rationale")
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Alternative options considered",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Decision confidence score",
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    outcome: str | None = Field(default=None, description="Decision outcome")
    impact_assessment: dict[str, Any] = Field(
        default_factory=dict,
        description="Impact assessment",
    )


class MemoryType(str, Enum):
    """Types of agent memory."""

    EPISODIC = "episodic"  # Specific events
    SEMANTIC = "semantic"  # General knowledge
    PROCEDURAL = "procedural"  # Skills/procedures
    WORKING = "working"  # Short-term context


class Memory(BaseModel):
    """Agent memory entry."""

    memory_id: str = Field(description="Unique memory identifier")
    memory_type: MemoryType = Field(description="Type of memory")
    content: dict[str, Any] = Field(description="Memory content")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Memory importance score",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0, description="Number of times accessed")
    tags: list[str] = Field(default_factory=list, description="Memory tags")
    related_goals: list[str] = Field(
        default_factory=list,
        description="Related goal IDs",
    )


class LearningExperience(BaseModel):
    """Learning experience for self-directed learning."""

    experience_id: str = Field(description="Unique experience identifier")
    goal_id: str = Field(description="Associated goal ID")
    action_taken: str = Field(description="Action performed")
    outcome: str = Field(description="Action outcome")
    success: bool = Field(description="Whether action was successful")
    lesson_learned: str = Field(description="Lesson learned from experience")
    timestamp: datetime = Field(default_factory=datetime.now)
    impact_score: float = Field(
        default=0.0,
        description="Impact score for learning",
    )


class TaskExecutionPlan(BaseModel):
    """Plan for executing a goal-oriented task."""

    plan_id: str = Field(description="Unique plan identifier")
    goal_id: str = Field(description="Associated goal ID")
    steps: list[dict[str, Any]] = Field(description="Execution steps")
    estimated_duration: int = Field(description="Estimated duration in seconds")
    resources_required: list[str] = Field(
        default_factory=list,
        description="Required resources",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AutonomousExecutionContext(BaseModel):
    """Execution context for autonomous agent."""

    agent_id: str = Field(description="Agent identifier")
    primary_goal: Goal = Field(description="Primary goal")
    active_goals: list[Goal] = Field(
        default_factory=list,
        description="Currently active goals",
    )
    completed_goals: list[Goal] = Field(
        default_factory=list,
        description="Completed goals",
    )
    decision_lineage: list[Decision] = Field(
        default_factory=list,
        description="History of decisions made",
    )
    long_term_memory: list[Memory] = Field(
        default_factory=list,
        description="Long-term memory store",
    )
    working_memory: list[Memory] = Field(
        default_factory=list,
        description="Working memory (short-term)",
    )
    learning_experiences: list[LearningExperience] = Field(
        default_factory=list,
        description="Learning history",
    )
    current_plan: TaskExecutionPlan | None = Field(
        default=None,
        description="Current execution plan",
    )
    context_retention_limit: int = Field(
        default=100,
        description="Maximum context items to retain",
    )


class AutonomousPromptTemplate(BaseModel):
    """Prompt templates for autonomous agent."""

    system_prompt: str = Field(
        default="""You are an autonomous AI agent capable of goal-oriented task execution.

You have the ability to:
1. Break down complex goals into manageable sub-goals
2. Make decisions based on available information
3. Learn from experiences and adjust your approach
4. Maintain long-term context and memory
5. Self-direct your learning and improvement

Always explain your reasoning and track your decision lineage.""",
        description="System prompt",
    )

    goal_planning_prompt: str = Field(
        default="""Primary Goal: {goal}

Current Context:
{context}

Available Resources:
{resources}

Create a detailed execution plan with specific steps.""",
        description="Goal planning prompt",
    )

    decision_prompt: str = Field(
        default="""Current Goal: {goal}

Current Situation:
{situation}

Available Options:
{options}

Previous Decisions:
{previous_decisions}

Make a decision and explain your rationale.""",
        description="Decision-making prompt",
    )

    learning_prompt: str = Field(
        default="""Experience: {experience}

Outcome: {outcome}

Success: {success}

What lesson can be learned from this experience?""",
        description="Learning prompt",
    )
