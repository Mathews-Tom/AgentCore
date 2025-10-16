"""
Pydantic models for the Context Reasoning framework.

Implements data models for:
- ReasoningRequest: Input parameters for reasoning requests
- ReasoningResult: Standardized output from reasoning strategies
- ReasoningMetrics: Performance tracking and compute efficiency metrics
- BoundedContextConfig: Configuration for bounded context strategy
- IterationMetrics: Per-iteration tracking for multi-step reasoning
- CarryoverContent: Compressed summary structure for bounded context
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ReasoningRequest(BaseModel):
    """Input parameters for reasoning API requests."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The problem or question to solve via reasoning"
    )
    strategy: str | None = Field(
        default=None,
        pattern="^[a-z_]+$",
        description="Reasoning strategy to use (e.g., 'bounded_context', 'chain_of_thought', 'react')"
    )
    strategy_config: dict[str, Any] | None = Field(
        default=None,
        description="Strategy-specific configuration parameters"
    )


class ReasoningMetrics(BaseModel):
    """Performance and compute efficiency metrics for reasoning execution."""

    total_tokens: int = Field(
        ...,
        ge=0,
        description="Total tokens processed during reasoning"
    )
    execution_time_ms: int = Field(
        ...,
        ge=0,
        description="Total execution time in milliseconds"
    )
    strategy_specific: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific metrics (e.g., iterations, compute savings)"
    )


class ReasoningResult(BaseModel):
    """Standardized output from all reasoning strategies."""

    answer: str = Field(
        ...,
        description="Final answer produced by reasoning process"
    )
    strategy_used: str = Field(
        ...,
        description="Name of the strategy that generated this result"
    )
    metrics: ReasoningMetrics = Field(
        ...,
        description="Performance and efficiency metrics"
    )
    trace: list[dict[str, Any]] | None = Field(
        default=None,
        description="Execution trace/iterations (optional, for debugging/analysis)"
    )


class BoundedContextConfig(BaseModel):
    """Configuration parameters for bounded context reasoning strategy."""

    chunk_size: int = Field(
        default=8192,
        ge=1024,
        le=32768,
        description="Maximum tokens per reasoning iteration"
    )
    carryover_size: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Maximum tokens to carry forward between iterations"
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of reasoning iterations allowed"
    )

    @model_validator(mode='after')
    def validate_carryover_less_than_chunk(self) -> BoundedContextConfig:
        """Validate that carryover_size is less than chunk_size."""
        if self.carryover_size >= self.chunk_size:
            raise ValueError(
                f"carryover_size ({self.carryover_size}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class IterationMetrics(BaseModel):
    """Metrics for a single reasoning iteration in bounded context strategy."""

    iteration: int = Field(
        ...,
        ge=0,
        description="Iteration number (0-indexed)"
    )
    tokens: int = Field(
        ...,
        ge=0,
        description="Tokens consumed in this iteration"
    )
    has_answer: bool = Field(
        ...,
        description="Whether an answer was found in this iteration"
    )
    carryover_generated: bool = Field(
        default=False,
        description="Whether carryover was generated after this iteration"
    )
    execution_time_ms: int | None = Field(
        default=None,
        ge=0,
        description="Execution time for this iteration in milliseconds"
    )


class CarryoverContent(BaseModel):
    """Structured carryover content for bounded context reasoning."""

    current_strategy: str = Field(
        ...,
        description="High-level approach being used for reasoning"
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Important insights discovered so far"
    )
    progress: str = Field(
        ...,
        description="What has been accomplished in reasoning so far"
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="What needs to be done next"
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Open questions or challenges that remain"
    )

    def to_text(self) -> str:
        """Convert structured carryover to text format for LLM prompt."""
        lines = [
            f"Current Strategy: {self.current_strategy}",
            "",
            "Key Findings:",
        ]
        for finding in self.key_findings:
            lines.append(f"- {finding}")

        lines.extend([
            "",
            f"Progress: {self.progress}",
            "",
            "Next Steps:",
        ])
        for step in self.next_steps:
            lines.append(f"- {step}")

        if self.unresolved:
            lines.extend([
                "",
                "Unresolved:",
            ])
            for issue in self.unresolved:
                lines.append(f"- {issue}")

        return "\n".join(lines)


class BoundedContextIterationResult(BaseModel):
    """Result from a single bounded context reasoning iteration."""

    content: str = Field(
        ...,
        description="Generated reasoning content for this iteration"
    )
    has_answer: bool = Field(
        ...,
        description="Whether the answer was found in this iteration"
    )
    answer: str | None = Field(
        default=None,
        description="Extracted answer if has_answer=True"
    )
    carryover: CarryoverContent | None = Field(
        default=None,
        description="Generated carryover for next iteration (if continuing)"
    )
    metrics: IterationMetrics = Field(
        ...,
        description="Metrics for this iteration"
    )


class BoundedContextResult(BaseModel):
    """Complete result from bounded context reasoning execution."""

    answer: str = Field(
        ...,
        description="Final answer found via bounded context reasoning"
    )
    iterations: list[BoundedContextIterationResult] = Field(
        ...,
        min_length=1,
        description="All iterations executed during reasoning"
    )
    total_tokens: int = Field(
        ...,
        ge=0,
        description="Total tokens consumed across all iterations"
    )
    total_iterations: int = Field(
        ...,
        ge=1,
        description="Number of iterations executed"
    )
    compute_savings_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of compute saved vs traditional reasoning"
    )
    carryover_compressions: int = Field(
        default=0,
        ge=0,
        description="Number of carryover compressions performed"
    )
    execution_time_ms: int = Field(
        ...,
        ge=0,
        description="Total execution time in milliseconds"
    )

    @field_validator('total_iterations')
    @classmethod
    def validate_iterations_match(cls, v: int, info) -> int:
        """Validate total_iterations matches length of iterations list."""
        if 'iterations' in info.data and len(info.data['iterations']) != v:
            raise ValueError(
                f"total_iterations ({v}) must match iterations list length "
                f"({len(info.data['iterations'])})"
            )
        return v
