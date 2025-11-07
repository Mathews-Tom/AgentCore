"""
Delta Generation Service

Analyzes execution traces and generates context improvement suggestions.
Core component of ACE's self-supervised learning system.
"""

import json
from typing import Any
from uuid import UUID

import structlog

from agentcore.ace.models.ace_models import ContextDelta, ContextPlaybook, ExecutionTrace
from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse
from agentcore.a2a_protocol.services.llm_client_base import LLMClient

logger = structlog.get_logger()


class DeltaGenerator:
    """
    Context Delta generation service.

    Analyzes execution traces and generates improvement suggestions for agent playbooks.
    Uses cost-effective LLM models for self-supervised learning.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Initialize delta generator.

        Args:
            llm_client: LLM client for generating deltas
        """
        self.llm_client = llm_client
        logger.info("Delta generator initialized")

    async def generate_deltas(
        self,
        execution_trace: ExecutionTrace,
        playbook: ContextPlaybook,
        model: str = "gpt-4.1-mini",
        max_deltas: int = 5,
    ) -> list[ContextDelta]:
        """
        Generate context improvement deltas from execution trace.

        Args:
            execution_trace: Execution performance trace to analyze
            playbook: Current agent playbook
            model: LLM model to use (default: gpt-4.1-mini for cost)
            max_deltas: Maximum number of deltas to generate

        Returns:
            List of context delta suggestions with confidence scores

        Raises:
            ValueError: If trace or playbook is invalid
        """
        logger.info(
            "Generating deltas from trace",
            agent_id=execution_trace.agent_id,
            trace_id=str(execution_trace.trace_id),
            success=execution_trace.success,
            model=model,
        )

        # Build analysis prompt
        prompt = self._build_analysis_prompt(execution_trace, playbook, max_deltas)

        # Call LLM for delta generation
        llm_request = LLMRequest(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,  # Lower temperature for focused analysis
            max_tokens=2000,
            trace_id=str(execution_trace.trace_id),
        )

        try:
            response = await self.llm_client.complete(llm_request)

            # Parse LLM response into deltas
            deltas = self._parse_llm_response(
                response, playbook.playbook_id, execution_trace
            )

            logger.info(
                "Deltas generated successfully",
                agent_id=execution_trace.agent_id,
                trace_id=str(execution_trace.trace_id),
                delta_count=len(deltas),
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return deltas

        except Exception as e:
            logger.error(
                "Delta generation failed",
                agent_id=execution_trace.agent_id,
                trace_id=str(execution_trace.trace_id),
                error=str(e),
            )
            raise

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for delta generation.

        Returns:
            System prompt string
        """
        return """You are an expert AI system analyzer specializing in agent context optimization.

Your task is to analyze agent execution traces and suggest context improvements that will enhance future performance.

For each execution trace, identify patterns, learnings, and opportunities for improvement. Generate specific, actionable context deltas with:
1. **Changes**: Precise modifications to agent context (key-value pairs)
2. **Confidence**: Your confidence in the improvement (0.0-1.0)
3. **Reasoning**: Clear explanation of why this change helps

Focus on:
- Learning from failures (error patterns, edge cases)
- Optimizing strategies (successful approaches)
- Improving patterns (recurring behaviors)
- Refining preferences (model parameters, approaches)

Return your analysis as a JSON array of delta objects with this structure:
```json
[
  {
    "changes": {"key.subkey": "value"},
    "confidence": 0.85,
    "reasoning": "Explanation of improvement..."
  }
]
```

Be concise but specific. Prioritize high-confidence improvements that directly address observed performance issues."""

    def _build_analysis_prompt(
        self,
        execution_trace: ExecutionTrace,
        playbook: ContextPlaybook,
        max_deltas: int,
    ) -> str:
        """
        Build analysis prompt for LLM.

        Args:
            execution_trace: Execution trace to analyze
            playbook: Current playbook context
            max_deltas: Maximum deltas to generate

        Returns:
            Formatted prompt string
        """
        # Extract key trace information
        trace_info = {
            "success": execution_trace.success,
            "execution_time": execution_trace.execution_time,
            "output_quality": execution_trace.output_quality,
            "error_message": execution_trace.error_message,
            "metadata": execution_trace.metadata,
        }

        # Extract relevant context sections
        context_summary = self._summarize_context(playbook.context)

        prompt = f"""## Execution Trace Analysis

**Agent ID:** {execution_trace.agent_id}
**Task ID:** {execution_trace.task_id or "N/A"}
**Execution Result:** {"SUCCESS" if execution_trace.success else "FAILURE"}
**Execution Time:** {execution_trace.execution_time:.2f}s
**Output Quality:** {execution_trace.output_quality if execution_trace.output_quality is not None else "N/A"}

**Error Details:**
{execution_trace.error_message or "No errors"}

**Trace Metadata:**
```json
{json.dumps(execution_trace.metadata, indent=2)}
```

## Current Agent Context

**Playbook Version:** {playbook.version}
**Context Summary:**
```json
{json.dumps(context_summary, indent=2)}
```

## Task

Analyze this execution trace and generate up to {max_deltas} context improvement deltas.

Consider:
1. What went wrong (if failure)?
2. What could be optimized (even if success)?
3. What patterns emerge from the trace?
4. What context changes would prevent similar issues?

Return a JSON array of delta objects (max {max_deltas}).
"""

        return prompt

    def _summarize_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Summarize context for prompt (avoid token overflow).

        Args:
            context: Full playbook context

        Returns:
            Summarized context dict
        """
        summary: dict[str, Any] = {}

        for key, value in context.items():
            if isinstance(value, dict):
                # Include first 3 items for dict values
                summary[key] = dict(list(value.items())[:3])
            elif isinstance(value, list):
                # Include first 5 items for list values
                summary[key] = value[:5]
            elif isinstance(value, str) and len(value) > 200:
                # Truncate long strings
                summary[key] = value[:200] + "..."
            else:
                summary[key] = value

        return summary

    def _parse_llm_response(
        self,
        response: LLMResponse,
        playbook_id: UUID,
        execution_trace: ExecutionTrace,
    ) -> list[ContextDelta]:
        """
        Parse LLM response into ContextDelta objects.

        Args:
            response: LLM response containing delta suggestions
            playbook_id: Target playbook ID
            execution_trace: Original execution trace

        Returns:
            List of parsed ContextDelta objects
        """
        deltas: list[ContextDelta] = []

        try:
            # Extract JSON from response content
            content = response.content.strip()

            # Handle markdown code blocks - extract content between ```json and ```
            if "```json" in content:
                # Find the start and end of the JSON code block
                start_marker = "```json"
                end_marker = "```"

                start_idx = content.find(start_marker)
                if start_idx != -1:
                    # Move past the start marker
                    start_idx += len(start_marker)
                    # Find the end marker after the start
                    end_idx = content.find(end_marker, start_idx)
                    if end_idx != -1:
                        content = content[start_idx:end_idx]
            elif "```" in content:
                # Handle generic code blocks without json specifier
                start_idx = content.find("```")
                if start_idx != -1:
                    start_idx += 3
                    end_idx = content.find("```", start_idx)
                    if end_idx != -1:
                        content = content[start_idx:end_idx]

            content = content.strip()

            # Parse JSON array
            delta_dicts = json.loads(content)

            if not isinstance(delta_dicts, list):
                logger.warning(
                    "LLM response is not a list, wrapping in array",
                    trace_id=str(execution_trace.trace_id),
                )
                delta_dicts = [delta_dicts]

            # Create ContextDelta objects
            for delta_dict in delta_dicts:
                try:
                    delta = ContextDelta(
                        playbook_id=playbook_id,
                        changes=delta_dict.get("changes", {}),
                        confidence=float(delta_dict.get("confidence", 0.5)),
                        reasoning=delta_dict.get("reasoning", "LLM-generated improvement"),
                    )
                    deltas.append(delta)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to create delta from response item",
                        error=str(e),
                        delta_dict=delta_dict,
                    )
                    continue

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM response as JSON",
                trace_id=str(execution_trace.trace_id),
                error=str(e),
                response_content=response.content[:500],
            )
            # Return empty list rather than raising
            return []

        return deltas
