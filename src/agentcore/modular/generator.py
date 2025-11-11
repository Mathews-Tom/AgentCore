"""
Generator Module Implementation

This module provides the Generator implementation that synthesizes final responses
from verified execution results with support for multiple output formats.

Key Features:
- Response synthesis from verified execution results
- Multiple output formats: text, JSON, markdown, HTML
- Reasoning trace generation and inclusion
- Evidence source tracking from execution metadata
- Response quality validation
- Output length constraints
"""

from __future__ import annotations

import json
from typing import Any

from agentcore.modular.base import BaseGenerator
from agentcore.modular.interfaces import (
    ExecutionResult,
    GeneratedResponse,
    GenerationRequest,
    OutputFormat,
)


class Generator(BaseGenerator):
    """
    Generator module that synthesizes final responses from verified execution
    results with support for multiple output formats and reasoning traces.

    Implements the GeneratorInterface protocol with comprehensive response
    synthesis logic for text, JSON, markdown, and HTML outputs.

    Example:
        >>> generator = Generator()
        >>> request = GenerationRequest(
        ...     verified_results=[execution_result],
        ...     format="markdown",
        ...     include_reasoning=True
        ... )
        >>> response = await generator.synthesize_response(request)
        >>> print(response.content)
    """

    def __init__(
        self,
        a2a_context: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        """
        Initialize Generator module.

        Args:
            a2a_context: A2A context for distributed tracing
            logger: Logger instance for structured logging
        """
        super().__init__(a2a_context, logger)

    async def health_check(self) -> dict[str, Any]:
        """
        Check Generator module health.

        Returns:
            Health status with module information
        """
        return {
            "status": "healthy",
            "module": "Generator",
            "execution_id": self.state.execution_id,
        }

    # ========================================================================
    # Core Generation Methods (implements GeneratorInterface)
    # ========================================================================

    async def _synthesize_response_impl(
        self, request: GenerationRequest
    ) -> GeneratedResponse:
        """
        Implementation-specific response synthesis.

        Synthesizes a coherent response from verified execution results with
        support for multiple output formats and reasoning inclusion.

        Args:
            request: Generation request with results and format

        Returns:
            Generated response with content and metadata

        Raises:
            ValueError: If request is invalid
        """
        if not request.verified_results:
            raise ValueError("GenerationRequest must contain at least one result")

        self.logger.info(
            "synthesizing_response",
            results_count=len(request.verified_results),
            format=request.format,
            include_reasoning=request.include_reasoning,
        )

        # Extract key information from results
        synthesis_data = self._extract_synthesis_data(request.verified_results)

        # Build content based on format
        content = await self._build_content(
            synthesis_data,
            request.format,
            request.max_length,
        )

        # Extract sources from execution metadata
        sources = self._extract_sources(request.verified_results)

        # Build reasoning trace if requested
        reasoning = None
        if request.include_reasoning:
            reasoning = self._build_reasoning_trace(request.verified_results)

        # Create response
        response = GeneratedResponse(
            content=content,
            format=request.format,
            reasoning=reasoning,
            sources=sources,
            metadata={
                "results_count": len(request.verified_results),
                "successful_count": sum(
                    1 for r in request.verified_results if r.success
                ),
                "failed_count": sum(
                    1 for r in request.verified_results if not r.success
                ),
            },
        )

        self.logger.info(
            "synthesis_complete",
            content_length=len(content),
            sources_count=len(sources),
            has_reasoning=reasoning is not None,
        )

        return response

    async def _format_output_impl(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Implementation-specific output formatting.

        Formats content according to the specified output format with optional
        JSON schema validation and template rendering.

        Args:
            content: Content to format
            format_spec: Output format specification

        Returns:
            Formatted content string

        Raises:
            ValueError: If format specification is invalid
        """
        if not content:
            raise ValueError("Content cannot be empty")

        if not format_spec.type:
            raise ValueError("OutputFormat must specify a type")

        self.logger.info("formatting_output", format_type=format_spec.type)

        # Dispatch to format-specific handler
        format_handlers = {
            "text": self._format_as_text,
            "json": self._format_as_json,
            "markdown": self._format_as_markdown,
            "html": self._format_as_html,
        }

        handler = format_handlers.get(format_spec.type.lower())
        if not handler:
            available_formats = ", ".join(format_handlers.keys())
            raise ValueError(
                f"Unknown format type: {format_spec.type}. "
                f"Available formats: {available_formats}"
            )

        formatted = await handler(content, format_spec)

        self.logger.info(
            "formatting_complete",
            format_type=format_spec.type,
            output_length=len(formatted),
        )

        return formatted

    async def _include_reasoning_impl(
        self, response: GeneratedResponse, reasoning: str
    ) -> GeneratedResponse:
        """
        Implementation-specific reasoning inclusion.

        Adds reasoning chain to generated response, preserving all existing
        metadata and sources.

        Args:
            response: Generated response to enhance
            reasoning: Reasoning chain to include

        Returns:
            Enhanced response with reasoning

        Raises:
            ValueError: If response or reasoning is invalid
        """
        if not reasoning:
            raise ValueError("Reasoning cannot be empty")

        self.logger.info("including_reasoning", reasoning_length=len(reasoning))

        # Create enhanced response with reasoning
        enhanced = GeneratedResponse(
            content=response.content,
            format=response.format,
            reasoning=reasoning,
            sources=response.sources,
            metadata={
                **response.metadata,
                "reasoning_included": True,
                "reasoning_length": len(reasoning),
            },
        )

        return enhanced

    # ========================================================================
    # Synthesis Helper Methods
    # ========================================================================

    def _extract_synthesis_data(
        self, results: list[ExecutionResult]
    ) -> dict[str, Any]:
        """
        Extract key information from execution results for synthesis.

        Args:
            results: Execution results to process

        Returns:
            Dictionary with synthesis data
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        return {
            "successful_results": successful_results,
            "failed_results": failed_results,
            "total_count": len(results),
            "success_count": len(successful_results),
            "failure_count": len(failed_results),
            "total_execution_time": sum(r.execution_time for r in results),
        }

    async def _build_content(
        self,
        synthesis_data: dict[str, Any],
        format_type: str,
        max_length: int | None,
    ) -> str:
        """
        Build response content from synthesis data.

        Args:
            synthesis_data: Extracted synthesis data
            format_type: Output format type
            max_length: Maximum content length (optional)

        Returns:
            Synthesized content string
        """
        content_parts: list[str] = []

        # Build success summary
        if synthesis_data["successful_results"]:
            content_parts.append(
                self._build_success_summary(synthesis_data["successful_results"])
            )

        # Build failure summary (if any)
        if synthesis_data["failed_results"]:
            content_parts.append(
                self._build_failure_summary(synthesis_data["failed_results"])
            )

        # Join parts based on format
        if format_type == "markdown":
            content = "\n\n".join(content_parts)
        elif format_type == "html":
            content = "<br><br>".join(content_parts)
        elif format_type == "json":
            # For JSON, return structured data
            content = json.dumps(
                {
                    "success": synthesis_data["success_count"],
                    "failures": synthesis_data["failure_count"],
                    "results": [
                        self._result_to_dict(r)
                        for r in synthesis_data["successful_results"]
                    ],
                    "errors": [
                        self._result_to_dict(r)
                        for r in synthesis_data["failed_results"]
                    ],
                },
                indent=2,
            )
        else:
            content = "\n\n".join(content_parts)

        # Apply max length constraint
        if max_length and len(content) > max_length:
            content = content[:max_length] + "..."

        return content

    def _build_success_summary(
        self, successful_results: list[ExecutionResult]
    ) -> str:
        """
        Build summary of successful results.

        Args:
            successful_results: List of successful execution results

        Returns:
            Summary string
        """
        if len(successful_results) == 1:
            result = successful_results[0]
            return self._format_single_result(result)

        # Multiple results - build aggregate summary
        summary_parts = [
            f"Successfully completed {len(successful_results)} steps:"
        ]

        for result in successful_results:
            summary_parts.append(f"- {result.step_id}: {self._format_result_data(result.result)}")

        return "\n".join(summary_parts)

    def _build_failure_summary(
        self, failed_results: list[ExecutionResult]
    ) -> str:
        """
        Build summary of failed results.

        Args:
            failed_results: List of failed execution results

        Returns:
            Summary string
        """
        summary_parts = [
            f"Encountered {len(failed_results)} failures:"
        ]

        for result in failed_results:
            error_msg = result.error or "Unknown error"
            summary_parts.append(f"- {result.step_id}: {error_msg}")

        return "\n".join(summary_parts)

    def _format_single_result(self, result: ExecutionResult) -> str:
        """
        Format a single execution result.

        Args:
            result: Execution result to format

        Returns:
            Formatted result string
        """
        result_data = self._format_result_data(result.result)
        return f"Result: {result_data}"

    def _format_result_data(self, data: Any) -> str:
        """
        Format result data for display.

        Args:
            data: Result data to format

        Returns:
            Formatted data string
        """
        if data is None:
            return "null"
        elif isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)
        else:
            return str(data)

    def _result_to_dict(self, result: ExecutionResult) -> dict[str, Any]:
        """
        Convert execution result to dictionary.

        Args:
            result: Execution result to convert

        Returns:
            Dictionary representation
        """
        return {
            "step_id": result.step_id,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
            "metadata": result.metadata,
        }

    def _extract_sources(self, results: list[ExecutionResult]) -> list[str]:
        """
        Extract source references from execution results.

        Args:
            results: Execution results to process

        Returns:
            List of unique source identifiers
        """
        sources: set[str] = set()

        for result in results:
            # Extract sources from metadata
            if result.metadata and "sources" in result.metadata:
                result_sources = result.metadata["sources"]
                if isinstance(result_sources, list):
                    sources.update(result_sources)
                elif isinstance(result_sources, str):
                    sources.add(result_sources)

            # Add step_id as implicit source
            sources.add(f"step:{result.step_id}")

        return sorted(sources)

    def _build_reasoning_trace(
        self, results: list[ExecutionResult]
    ) -> str:
        """
        Build reasoning trace from execution results.

        Args:
            results: Execution results to process

        Returns:
            Reasoning trace string
        """
        trace_parts = ["Execution Reasoning Trace:", ""]

        for i, result in enumerate(results, 1):
            trace_parts.append(f"Step {i}: {result.step_id}")
            trace_parts.append(f"  Status: {'Success' if result.success else 'Failed'}")
            trace_parts.append(f"  Execution Time: {result.execution_time:.3f}s")

            if result.success:
                trace_parts.append(f"  Result: {self._format_result_data(result.result)}")
            else:
                trace_parts.append(f"  Error: {result.error or 'Unknown error'}")

            trace_parts.append("")  # Blank line between steps

        # Add summary
        total_time = sum(r.execution_time for r in results)
        success_count = sum(1 for r in results if r.success)
        trace_parts.append("Summary:")
        trace_parts.append(f"  Total Steps: {len(results)}")
        trace_parts.append(f"  Successful: {success_count}")
        trace_parts.append(f"  Failed: {len(results) - success_count}")
        trace_parts.append(f"  Total Time: {total_time:.3f}s")

        return "\n".join(trace_parts)

    # ========================================================================
    # Format-Specific Handlers
    # ========================================================================

    async def _format_as_text(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Format content as plain text.

        Args:
            content: Content to format
            format_spec: Format specification

        Returns:
            Plain text content
        """
        # Apply template if provided
        if format_spec.template:
            return format_spec.template.format(content=content)

        return content

    async def _format_as_json(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Format content as JSON.

        Args:
            content: Content to format
            format_spec: Format specification

        Returns:
            JSON formatted content

        Raises:
            ValueError: If content is not valid JSON
        """
        # Validate JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Content is not valid JSON: {e}") from e

        # Validate against schema if provided
        if format_spec.json_schema:
            self._validate_json_schema(data, format_spec.json_schema)

        # Return formatted JSON
        return json.dumps(data, indent=2)

    async def _format_as_markdown(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Format content as Markdown.

        Args:
            content: Content to format
            format_spec: Format specification

        Returns:
            Markdown formatted content
        """
        # Apply template if provided
        if format_spec.template:
            return format_spec.template.format(content=content)

        # Add markdown formatting if content doesn't have it
        if not content.startswith("#"):
            return f"# Response\n\n{content}"

        return content

    async def _format_as_html(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Format content as HTML.

        Args:
            content: Content to format
            format_spec: Format specification

        Returns:
            HTML formatted content
        """
        # Apply template if provided
        if format_spec.template:
            return format_spec.template.format(content=content)

        # Wrap in basic HTML structure
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Response</title>
</head>
<body>
    <div class="response">
        {self._text_to_html(content)}
    </div>
</body>
</html>"""

        return html

    def _text_to_html(self, text: str) -> str:
        """
        Convert plain text to HTML paragraphs.

        Args:
            text: Plain text content

        Returns:
            HTML formatted text
        """
        # Split by double newlines for paragraphs
        paragraphs = text.split("\n\n")
        html_parts = [f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs]
        return "\n".join(html_parts)

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def _validate_json_schema(
        self, data: Any, schema: dict[str, Any]
    ) -> None:
        """
        Validate data against JSON schema.

        Basic schema validation for type checking.
        For production use, consider using jsonschema library.

        Args:
            data: Data to validate
            schema: JSON schema definition

        Raises:
            ValueError: If validation fails
        """
        # Basic type validation
        expected_type = schema.get("type")
        if expected_type:
            actual_type = self._get_json_type(data)
            if actual_type != expected_type:
                raise ValueError(
                    f"Type mismatch: expected {expected_type}, got {actual_type}"
                )

        # Properties validation (for objects)
        if expected_type == "object" and isinstance(data, dict):
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    raise ValueError(f"Missing required property: {prop}")

        # Array validation
        if expected_type == "array" and isinstance(data, list):
            min_items = schema.get("minItems")
            if min_items is not None and len(data) < min_items:
                raise ValueError(
                    f"Array has {len(data)} items, minimum is {min_items}"
                )

    def _get_json_type(self, value: Any) -> str:
        """
        Get JSON schema type name for Python value.

        Args:
            value: Python value

        Returns:
            JSON schema type string
        """
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
