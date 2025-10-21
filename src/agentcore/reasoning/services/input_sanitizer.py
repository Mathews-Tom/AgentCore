"""
Input sanitization for prompt injection prevention.

Provides utilities to detect and prevent prompt injection attacks in
reasoning queries and system prompts.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import structlog

logger = structlog.get_logger()

# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    # System prompt manipulation
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|commands)",
    r"forget\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|commands)",
    r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|commands)",
    r"system\s*:\s*.{0,50}(?:ignore|forget|disregard)",
    # Role manipulation
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an)\s+\w+",
    r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an)\s+\w+",
    r"roleplay\s+as\s+(?:a|an)\s+\w+",
    # Instruction injection
    r"new\s+(?:instruction|directive|command)\s*:",
    r"updated\s+(?:instruction|directive|command)\s*:",
    r"revised\s+(?:instruction|directive|command)\s*:",
    # Delimiter manipulation (trying to break out of context)
    r"</?\s*(?:system|user|assistant)\s*>",
    r"\[\s*/?\s*(?:SYSTEM|USER|ASSISTANT)\s*\]",
    r"<\|\s*(?:im_start|im_end)\s*\|>",
    # Command execution attempts
    r"(?:execute|run|eval)\s+(?:code|command|script)",
    r"import\s+os\s*;",
    r"subprocess\.(?:run|call|Popen)",
    r"__import__\s*\(",
    # Data exfiltration attempts
    r"print\s+(?:api_key|secret|password|token|credentials)",
    r"reveal\s+(?:api_key|secret|password|token|credentials)",
    r"show\s+(?:api_key|secret|password|credentials|token)",
]

# Compile patterns for efficient matching
COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
]


class SanitizationResult(NamedTuple):
    """Result of input sanitization."""

    is_safe: bool
    reason: str | None
    detected_patterns: list[str]


def sanitize_input(
    text: str,
    max_length: int = 100000,
    allow_special_chars: bool = True,
) -> SanitizationResult:
    """
    Sanitize input text for prompt injection prevention.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed text length
        allow_special_chars: Allow special characters (brackets, tags, etc.)

    Returns:
        SanitizationResult indicating if input is safe
    """
    detected_patterns = []

    # Check length
    if len(text) > max_length:
        return SanitizationResult(
            is_safe=False,
            reason=f"Input exceeds maximum length ({len(text)} > {max_length})",
            detected_patterns=[],
        )

    # Check for null bytes (potential injection)
    if "\x00" in text:
        return SanitizationResult(
            is_safe=False,
            reason="Null byte detected in input",
            detected_patterns=["null_byte"],
        )

    # Check for excessive special characters (unless allowed)
    if not allow_special_chars:
        special_char_ratio = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        ) / max(1, len(text))
        if special_char_ratio > 0.3:  # More than 30% special chars
            return SanitizationResult(
                is_safe=False,
                reason=f"Excessive special characters ({special_char_ratio:.1%})",
                detected_patterns=["excessive_special_chars"],
            )

    # Check for known injection patterns
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            detected_patterns.append(match.group(0))

    if detected_patterns:
        logger.warning(
            "prompt_injection_detected",
            patterns=detected_patterns[:5],  # Log first 5 patterns
            text_preview=text[:100],
        )
        return SanitizationResult(
            is_safe=False,
            reason="Potential prompt injection detected",
            detected_patterns=detected_patterns,
        )

    return SanitizationResult(
        is_safe=True,
        reason=None,
        detected_patterns=[],
    )


def sanitize_reasoning_request(
    query: str,
    system_prompt: str | None = None,
) -> tuple[bool, str | None]:
    """
    Validate reasoning request for security issues.

    Args:
        query: User query to validate
        system_prompt: Optional system prompt to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Sanitize query
    query_result = sanitize_input(query, max_length=100000)
    if not query_result.is_safe:
        return False, f"Query validation failed: {query_result.reason}"

    # Sanitize system prompt if provided
    if system_prompt:
        prompt_result = sanitize_input(system_prompt, max_length=10000)
        if not prompt_result.is_safe:
            return False, f"System prompt validation failed: {prompt_result.reason}"

    return True, None
