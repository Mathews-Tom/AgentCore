"""
Compression Prompts for COMPASS Memory Service

Optimized prompts for context compression, critical fact extraction,
and quality validation. Designed for use with gpt-4.1-mini (test-time scaling).

Component ID: MEM-012
"""


def build_fact_extraction_prompt(
    content: str, context_type: str | None = None
) -> str:
    """
    Build prompt for extracting critical facts from content.

    Args:
        content: Content to extract facts from
        context_type: Optional context type (stage, task, etc.)

    Returns:
        Formatted prompt for LLM
    """
    context_hint = ""
    if context_type == "stage":
        context_hint = "Focus on key decisions, constraints, and action outcomes from this reasoning stage."
    elif context_type == "task":
        context_hint = "Focus on critical constraints, goals, and progress milestones."
    else:
        context_hint = "Focus on essential information that must not be lost."

    return f"""Extract critical facts from the following content that MUST be preserved during compression.

{context_hint}

Critical facts include:
- Hard constraints and requirements
- Key decisions and rationale
- Important numerical values or thresholds
- Error conditions and recovery actions
- Dependencies and relationships

Content to analyze:
---
{content}
---

Return ONLY a numbered list of critical facts, one per line. Be concise but complete.
Example format:
1. User requires JWT authentication
2. Redis cache must have 1-hour TTL
3. Error rate threshold is 5%

Critical Facts:"""


def build_compression_prompt(
    content: str,
    target_ratio: float,
    context_type: str,
    stage_type: str | None = None,
    task_goal: str | None = None,
    critical_facts: list[str] | None = None,
) -> str:
    """
    Build prompt for compressing content with target ratio.

    Args:
        content: Content to compress
        target_ratio: Target compression ratio (e.g., 10.0 for 10:1)
        context_type: Context type (stage or task)
        stage_type: Optional stage type (planning, execution, etc.)
        task_goal: Optional task goal for context
        critical_facts: Optional list of critical facts to preserve

    Returns:
        Formatted prompt for LLM
    """
    # Build context-specific instructions
    context_instructions = ""
    if context_type == "stage":
        stage_hint = f" ({stage_type} stage)" if stage_type else ""
        context_instructions = f"""You are compressing a reasoning stage{stage_hint}.

Compression guidelines for stage summary:
- Preserve key decisions and their rationale
- Include action outcomes (success/failure)
- Retain error conditions and recovery steps
- Keep critical constraints and requirements
- Summarize observations and insights
- Target length: ~{int(len(content) / target_ratio)} characters (compression ratio: {target_ratio}:1)"""

    elif context_type == "task":
        goal_hint = f"\nTask goal: {task_goal}" if task_goal else ""
        context_instructions = f"""You are compressing multiple stage summaries into a task progress summary.{goal_hint}

Compression guidelines for task progress:
- Synthesize progress across all stages
- Preserve critical constraints from any stage
- Highlight key milestones and blockers
- Include error patterns and learnings
- Retain actionable insights
- Target length: ~{int(len(content) / target_ratio)} characters (compression ratio: {target_ratio}:1)"""

    # Build critical facts section
    critical_facts_section = ""
    if critical_facts:
        facts_list = "\n".join([f"- {fact}" for fact in critical_facts])
        critical_facts_section = f"""

CRITICAL FACTS (MUST preserve):
{facts_list}"""

    return f"""{context_instructions}{critical_facts_section}

Content to compress:
---
{content}
---

Compressed output (preserve all critical facts, achieve ~{target_ratio}:1 ratio):"""


def build_quality_validation_prompt(
    original_content: str,
    compressed_content: str,
    critical_facts: list[str],
) -> str:
    """
    Build prompt for validating compression quality.

    Args:
        original_content: Original uncompressed content
        compressed_content: Compressed content
        critical_facts: List of critical facts to validate

    Returns:
        Formatted prompt for LLM
    """
    facts_list = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(critical_facts)])

    return f"""Validate compression quality by checking if critical facts are preserved.

Critical facts from original content:
{facts_list}

Compressed content:
---
{compressed_content}
---

For each critical fact, determine if it is preserved (fully, partially, or missing) in the compressed content.

Calculate the fact retention score as:
- Fully preserved fact = 1.0 point
- Partially preserved fact = 0.5 points
- Missing fact = 0.0 points

Return ONLY the final quality score as a decimal between 0.0 and 1.0.
Example: "Quality Score: 0.95" (meaning 95% of critical facts preserved)

Quality Score:"""


__all__ = [
    "build_fact_extraction_prompt",
    "build_compression_prompt",
    "build_quality_validation_prompt",
]
