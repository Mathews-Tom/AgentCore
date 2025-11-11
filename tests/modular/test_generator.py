"""
Tests for Generator Module Implementation

Validates response synthesis, output formatting, and reasoning inclusion.
"""

from __future__ import annotations

import json

import pytest

from agentcore.modular.generator import Generator
from agentcore.modular.interfaces import (
    ExecutionResult,
    GeneratedResponse,
    GenerationRequest,
    OutputFormat,
)


class TestGeneratorInitialization:
    """Test Generator module initialization."""

    def test_generator_default_initialization(self) -> None:
        """Test Generator with default configuration."""
        generator = Generator()

        assert generator.module_name == "Generator"
        assert generator.state is not None

    @pytest.mark.asyncio
    async def test_generator_health_check(self) -> None:
        """Test Generator health check."""
        generator = Generator()
        health = await generator.health_check()

        assert health["status"] == "healthy"
        assert health["module"] == "Generator"
        assert "execution_id" in health


class TestSynthesizeResponse:
    """Test response synthesis functionality."""

    @pytest.mark.asyncio
    async def test_synthesize_empty_results_raises_error(self) -> None:
        """Test that empty results list raises ValueError."""
        generator = Generator()
        request = GenerationRequest(verified_results=[])

        with pytest.raises(ValueError, match="must contain at least one result"):
            await generator.synthesize_response(request)

    @pytest.mark.asyncio
    async def test_synthesize_single_successful_result(self) -> None:
        """Test synthesis with single successful result."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = GenerationRequest(verified_results=[result])

        response = await generator.synthesize_response(request)

        assert response.content != ""
        assert response.format == "text"
        assert "42" in response.content
        assert len(response.sources) > 0
        assert "step:step_1" in response.sources

    @pytest.mark.asyncio
    async def test_synthesize_multiple_successful_results(self) -> None:
        """Test synthesis with multiple successful results."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.6,
            ),
        ]
        request = GenerationRequest(verified_results=results)

        response = await generator.synthesize_response(request)

        assert response.content != ""
        assert "step_1" in response.content
        assert "step_2" in response.content
        assert response.metadata["results_count"] == 2
        assert response.metadata["successful_count"] == 2

    @pytest.mark.asyncio
    async def test_synthesize_with_failed_results(self) -> None:
        """Test synthesis with failed results."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=False,
                result=None,
                error="Connection timeout",
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.6,
            ),
        ]
        request = GenerationRequest(verified_results=results)

        response = await generator.synthesize_response(request)

        assert response.content != ""
        assert "Connection timeout" in response.content
        assert response.metadata["failed_count"] == 1
        assert response.metadata["successful_count"] == 1

    @pytest.mark.asyncio
    async def test_synthesize_with_reasoning(self) -> None:
        """Test synthesis with reasoning trace included."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = GenerationRequest(
            verified_results=[result],
            include_reasoning=True,
        )

        response = await generator.synthesize_response(request)

        assert response.reasoning is not None
        assert "Execution Reasoning Trace" in response.reasoning
        assert "step_1" in response.reasoning
        assert "Success" in response.reasoning
        assert "0.500s" in response.reasoning

    @pytest.mark.asyncio
    async def test_synthesize_json_format(self) -> None:
        """Test synthesis with JSON format."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = GenerationRequest(
            verified_results=[result],
            format="json",
        )

        response = await generator.synthesize_response(request)

        assert response.format == "json"
        # Content should be valid JSON
        data = json.loads(response.content)
        assert "success" in data
        assert "results" in data

    @pytest.mark.asyncio
    async def test_synthesize_markdown_format(self) -> None:
        """Test synthesis with markdown format."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = GenerationRequest(
            verified_results=[result],
            format="markdown",
        )

        response = await generator.synthesize_response(request)

        assert response.format == "markdown"

    @pytest.mark.asyncio
    async def test_synthesize_with_max_length(self) -> None:
        """Test synthesis with max length constraint."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": "very long result " * 100},
            execution_time=0.5,
        )
        request = GenerationRequest(
            verified_results=[result],
            max_length=100,
        )

        response = await generator.synthesize_response(request)

        assert len(response.content) <= 103  # 100 + "..."

    @pytest.mark.asyncio
    async def test_synthesize_with_sources_from_metadata(self) -> None:
        """Test synthesis extracts sources from result metadata."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
            metadata={
                "sources": ["database", "api"],
            },
        )
        request = GenerationRequest(verified_results=[result])

        response = await generator.synthesize_response(request)

        assert "database" in response.sources
        assert "api" in response.sources


class TestFormatOutput:
    """Test output formatting functionality."""

    @pytest.mark.asyncio
    async def test_format_output_empty_content_raises_error(self) -> None:
        """Test that empty content raises ValueError."""
        generator = Generator()
        format_spec = OutputFormat(type="text")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            await generator.format_output("", format_spec)

    @pytest.mark.asyncio
    async def test_format_output_empty_type_raises_error(self) -> None:
        """Test that empty type raises ValueError."""
        generator = Generator()
        format_spec = OutputFormat(type="")

        with pytest.raises(ValueError, match="must specify a type"):
            await generator.format_output("content", format_spec)

    @pytest.mark.asyncio
    async def test_format_output_unknown_type_raises_error(self) -> None:
        """Test that unknown format type raises ValueError."""
        generator = Generator()
        format_spec = OutputFormat(type="unknown_format")

        with pytest.raises(ValueError, match="Unknown format type"):
            await generator.format_output("content", format_spec)

    @pytest.mark.asyncio
    async def test_format_as_text(self) -> None:
        """Test plain text formatting."""
        generator = Generator()
        content = "This is plain text content"
        format_spec = OutputFormat(type="text")

        formatted = await generator.format_output(content, format_spec)

        assert formatted == content

    @pytest.mark.asyncio
    async def test_format_as_text_with_template(self) -> None:
        """Test plain text formatting with template."""
        generator = Generator()
        content = "This is content"
        format_spec = OutputFormat(
            type="text",
            template="Result: {content}",
        )

        formatted = await generator.format_output(content, format_spec)

        assert formatted == "Result: This is content"

    @pytest.mark.asyncio
    async def test_format_as_json_valid(self) -> None:
        """Test JSON formatting with valid JSON content."""
        generator = Generator()
        content = json.dumps({"key": "value"})
        format_spec = OutputFormat(type="json")

        formatted = await generator.format_output(content, format_spec)

        # Should be valid JSON
        data = json.loads(formatted)
        assert data["key"] == "value"

    @pytest.mark.asyncio
    async def test_format_as_json_invalid_raises_error(self) -> None:
        """Test JSON formatting with invalid JSON content."""
        generator = Generator()
        content = "not valid json"
        format_spec = OutputFormat(type="json")

        with pytest.raises(ValueError, match="not valid JSON"):
            await generator.format_output(content, format_spec)

    @pytest.mark.asyncio
    async def test_format_as_json_with_schema_valid(self) -> None:
        """Test JSON formatting with schema validation (valid)."""
        generator = Generator()
        content = json.dumps({"name": "test", "age": 25})
        format_spec = OutputFormat(
            type="json",
            json_schema={
                "type": "object",
                "required": ["name", "age"],
            },
        )

        formatted = await generator.format_output(content, format_spec)

        data = json.loads(formatted)
        assert data["name"] == "test"
        assert data["age"] == 25

    @pytest.mark.asyncio
    async def test_format_as_json_with_schema_invalid(self) -> None:
        """Test JSON formatting with schema validation (invalid)."""
        generator = Generator()
        content = json.dumps({"name": "test"})  # Missing "age"
        format_spec = OutputFormat(
            type="json",
            json_schema={
                "type": "object",
                "required": ["name", "age"],
            },
        )

        with pytest.raises(ValueError, match="Missing required property: age"):
            await generator.format_output(content, format_spec)

    @pytest.mark.asyncio
    async def test_format_as_markdown(self) -> None:
        """Test markdown formatting."""
        generator = Generator()
        content = "This is content"
        format_spec = OutputFormat(type="markdown")

        formatted = await generator.format_output(content, format_spec)

        assert "# Response" in formatted
        assert "This is content" in formatted

    @pytest.mark.asyncio
    async def test_format_as_markdown_with_existing_header(self) -> None:
        """Test markdown formatting with existing header."""
        generator = Generator()
        content = "# Existing Header\n\nContent"
        format_spec = OutputFormat(type="markdown")

        formatted = await generator.format_output(content, format_spec)

        # Should not add another header
        assert formatted == content

    @pytest.mark.asyncio
    async def test_format_as_markdown_with_template(self) -> None:
        """Test markdown formatting with template."""
        generator = Generator()
        content = "This is content"
        format_spec = OutputFormat(
            type="markdown",
            template="## Result\n\n{content}",
        )

        formatted = await generator.format_output(content, format_spec)

        assert formatted == "## Result\n\nThis is content"

    @pytest.mark.asyncio
    async def test_format_as_html(self) -> None:
        """Test HTML formatting."""
        generator = Generator()
        content = "This is content"
        format_spec = OutputFormat(type="html")

        formatted = await generator.format_output(content, format_spec)

        assert "<!DOCTYPE html>" in formatted
        assert "<html>" in formatted
        assert "<body>" in formatted
        assert "This is content" in formatted

    @pytest.mark.asyncio
    async def test_format_as_html_with_template(self) -> None:
        """Test HTML formatting with template."""
        generator = Generator()
        content = "This is content"
        format_spec = OutputFormat(
            type="html",
            template="<div>{content}</div>",
        )

        formatted = await generator.format_output(content, format_spec)

        assert formatted == "<div>This is content</div>"


class TestIncludeReasoning:
    """Test reasoning inclusion functionality."""

    @pytest.mark.asyncio
    async def test_include_reasoning_empty_raises_error(self) -> None:
        """Test that empty reasoning raises ValueError."""
        generator = Generator()
        response = GeneratedResponse(
            content="content",
            format="text",
            sources=[],
        )

        with pytest.raises(ValueError, match="Reasoning cannot be empty"):
            await generator.include_reasoning(response, "")

    @pytest.mark.asyncio
    async def test_include_reasoning_adds_to_response(self) -> None:
        """Test that reasoning is added to response."""
        generator = Generator()
        response = GeneratedResponse(
            content="content",
            format="text",
            sources=["source1"],
        )
        reasoning = "Step 1: Analyzed query\nStep 2: Executed plan"

        enhanced = await generator.include_reasoning(response, reasoning)

        assert enhanced.reasoning == reasoning
        assert enhanced.content == response.content
        assert enhanced.sources == response.sources
        assert enhanced.metadata["reasoning_included"] is True
        assert enhanced.metadata["reasoning_length"] == len(reasoning)

    @pytest.mark.asyncio
    async def test_include_reasoning_preserves_metadata(self) -> None:
        """Test that reasoning inclusion preserves existing metadata."""
        generator = Generator()
        response = GeneratedResponse(
            content="content",
            format="text",
            sources=["source1"],
            metadata={"custom_key": "custom_value"},
        )
        reasoning = "Reasoning trace"

        enhanced = await generator.include_reasoning(response, reasoning)

        assert enhanced.metadata["custom_key"] == "custom_value"
        assert enhanced.metadata["reasoning_included"] is True


class TestReasoningTraceGeneration:
    """Test reasoning trace generation."""

    @pytest.mark.asyncio
    async def test_reasoning_trace_single_result(self) -> None:
        """Test reasoning trace with single result."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = GenerationRequest(
            verified_results=[result],
            include_reasoning=True,
        )

        response = await generator.synthesize_response(request)

        assert response.reasoning is not None
        assert "Step 1: step_1" in response.reasoning
        assert "Status: Success" in response.reasoning
        assert "Result:" in response.reasoning
        assert "Summary:" in response.reasoning
        assert "Total Steps: 1" in response.reasoning

    @pytest.mark.asyncio
    async def test_reasoning_trace_multiple_results(self) -> None:
        """Test reasoning trace with multiple results."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=False,
                result=None,
                error="Failed to connect",
                execution_time=0.3,
            ),
        ]
        request = GenerationRequest(
            verified_results=results,
            include_reasoning=True,
        )

        response = await generator.synthesize_response(request)

        assert response.reasoning is not None
        assert "Step 1: step_1" in response.reasoning
        assert "Step 2: step_2" in response.reasoning
        assert "Status: Success" in response.reasoning
        assert "Status: Failed" in response.reasoning
        assert "Error: Failed to connect" in response.reasoning
        assert "Total Steps: 2" in response.reasoning
        assert "Successful: 1" in response.reasoning
        assert "Failed: 1" in response.reasoning


class TestSourceExtraction:
    """Test source extraction from results."""

    @pytest.mark.asyncio
    async def test_extract_sources_from_metadata_list(self) -> None:
        """Test extracting sources from metadata list."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
            metadata={"sources": ["database", "api"]},
        )
        request = GenerationRequest(verified_results=[result])

        response = await generator.synthesize_response(request)

        assert "database" in response.sources
        assert "api" in response.sources

    @pytest.mark.asyncio
    async def test_extract_sources_from_metadata_string(self) -> None:
        """Test extracting sources from metadata string."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
            metadata={"sources": "database"},
        )
        request = GenerationRequest(verified_results=[result])

        response = await generator.synthesize_response(request)

        assert "database" in response.sources

    @pytest.mark.asyncio
    async def test_extract_sources_implicit_step_id(self) -> None:
        """Test that step_id is always included as implicit source."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = GenerationRequest(verified_results=[result])

        response = await generator.synthesize_response(request)

        assert "step:step_1" in response.sources

    @pytest.mark.asyncio
    async def test_extract_sources_deduplication(self) -> None:
        """Test that duplicate sources are deduplicated."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=0.5,
                metadata={"sources": ["database"]},
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.6,
                metadata={"sources": ["database"]},
            ),
        ]
        request = GenerationRequest(verified_results=results)

        response = await generator.synthesize_response(request)

        # Should only have one "database" source
        assert response.sources.count("database") == 1


class TestContentBuilding:
    """Test content building from synthesis data."""

    @pytest.mark.asyncio
    async def test_content_single_successful_result(self) -> None:
        """Test content for single successful result."""
        generator = Generator()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=42,
            execution_time=0.5,
        )
        request = GenerationRequest(verified_results=[result])

        response = await generator.synthesize_response(request)

        assert "Result: 42" in response.content

    @pytest.mark.asyncio
    async def test_content_multiple_successful_results(self) -> None:
        """Test content for multiple successful results."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result=42,
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result=43,
                execution_time=0.6,
            ),
        ]
        request = GenerationRequest(verified_results=results)

        response = await generator.synthesize_response(request)

        assert "Successfully completed 2 steps" in response.content
        assert "step_1" in response.content
        assert "step_2" in response.content

    @pytest.mark.asyncio
    async def test_content_with_failures(self) -> None:
        """Test content includes failure information."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=False,
                result=None,
                error="Connection failed",
                execution_time=0.5,
            ),
        ]
        request = GenerationRequest(verified_results=results)

        response = await generator.synthesize_response(request)

        assert "Encountered 1 failures" in response.content
        assert "Connection failed" in response.content


class TestJsonSchemaValidation:
    """Test JSON schema validation."""

    def test_validate_json_schema_type_mismatch(self) -> None:
        """Test schema validation with type mismatch."""
        generator = Generator()
        data = "string"
        schema = {"type": "number"}

        with pytest.raises(ValueError, match="Type mismatch"):
            generator._validate_json_schema(data, schema)

    def test_validate_json_schema_missing_required(self) -> None:
        """Test schema validation with missing required property."""
        generator = Generator()
        data = {"name": "test"}
        schema = {"type": "object", "required": ["name", "age"]}

        with pytest.raises(ValueError, match="Missing required property: age"):
            generator._validate_json_schema(data, schema)

    def test_validate_json_schema_array_min_items(self) -> None:
        """Test schema validation with array minItems."""
        generator = Generator()
        data = [1, 2]
        schema = {"type": "array", "minItems": 5}

        with pytest.raises(ValueError, match="minimum is 5"):
            generator._validate_json_schema(data, schema)

    def test_validate_json_schema_valid(self) -> None:
        """Test schema validation with valid data."""
        generator = Generator()
        data = {"name": "test", "age": 25}
        schema = {"type": "object", "required": ["name", "age"]}

        # Should not raise
        generator._validate_json_schema(data, schema)


class TestJsonTypeDetection:
    """Test JSON type detection helpers."""

    def test_get_json_type_null(self) -> None:
        """Test JSON type detection for null."""
        generator = Generator()
        assert generator._get_json_type(None) == "null"

    def test_get_json_type_boolean(self) -> None:
        """Test JSON type detection for boolean."""
        generator = Generator()
        assert generator._get_json_type(True) == "boolean"
        assert generator._get_json_type(False) == "boolean"

    def test_get_json_type_integer(self) -> None:
        """Test JSON type detection for integer."""
        generator = Generator()
        assert generator._get_json_type(42) == "integer"
        assert generator._get_json_type(0) == "integer"

    def test_get_json_type_number(self) -> None:
        """Test JSON type detection for number (float)."""
        generator = Generator()
        assert generator._get_json_type(3.14) == "number"

    def test_get_json_type_string(self) -> None:
        """Test JSON type detection for string."""
        generator = Generator()
        assert generator._get_json_type("hello") == "string"

    def test_get_json_type_array(self) -> None:
        """Test JSON type detection for array."""
        generator = Generator()
        assert generator._get_json_type([1, 2, 3]) == "array"

    def test_get_json_type_object(self) -> None:
        """Test JSON type detection for object."""
        generator = Generator()
        assert generator._get_json_type({"key": "value"}) == "object"

    def test_get_json_type_unknown(self) -> None:
        """Test JSON type detection for unknown types."""
        generator = Generator()
        assert generator._get_json_type(object()) == "unknown"


class TestHtmlFormatting:
    """Test HTML-specific formatting."""

    @pytest.mark.asyncio
    async def test_html_with_paragraphs(self) -> None:
        """Test HTML formatting with paragraph breaks."""
        generator = Generator()
        content = "Paragraph 1\n\nParagraph 2"
        format_spec = OutputFormat(type="html")

        formatted = await generator.format_output(content, format_spec)

        assert "<p>" in formatted
        assert "Paragraph 1" in formatted
        assert "Paragraph 2" in formatted

    @pytest.mark.asyncio
    async def test_html_with_line_breaks(self) -> None:
        """Test HTML formatting with line breaks."""
        generator = Generator()
        content = "Line 1\nLine 2"
        format_spec = OutputFormat(type="html")

        formatted = await generator.format_output(content, format_spec)

        assert "<br>" in formatted


class TestMetadata:
    """Test metadata generation."""

    @pytest.mark.asyncio
    async def test_metadata_includes_counts(self) -> None:
        """Test that metadata includes result counts."""
        generator = Generator()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result=42,
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=False,
                result=None,
                error="Failed",
                execution_time=0.3,
            ),
        ]
        request = GenerationRequest(verified_results=results)

        response = await generator.synthesize_response(request)

        assert response.metadata["results_count"] == 2
        assert response.metadata["successful_count"] == 1
        assert response.metadata["failed_count"] == 1
