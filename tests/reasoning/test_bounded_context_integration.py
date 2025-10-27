"""
Integration tests for Bounded Context Reasoning Engine.

Tests the complete reasoning flow from query to answer, including:
- Multi-iteration reasoning
- Carryover generation
- Answer detection
- Metrics calculation
- Error handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agentcore.reasoning.engines.bounded_context_engine import BoundedContextEngine
from src.agentcore.reasoning.models.reasoning_models import BoundedContextConfig
from src.agentcore.reasoning.services.llm_client import (
    GenerationResult,
    LLMClient,
    LLMClientConfig)


@pytest.fixture
def llm_config() -> LLMClientConfig:
    """Create test LLM client configuration."""
    return LLMClientConfig(
        api_key="test-key",
        base_url="https://api.test.com/v1",
        timeout_seconds=30)


@pytest.fixture
def bounded_config() -> BoundedContextConfig:
    """Create test bounded context configuration."""
    return BoundedContextConfig(
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=5)


@pytest.fixture
def mock_llm_client(llm_config: LLMClientConfig) -> LLMClient:
    """Create mock LLM client for testing."""
    client = LLMClient(llm_config)

    # Mock the generate method
    client.generate = AsyncMock()

    # Mock token counting
    client.count_tokens = MagicMock(return_value=100)

    return client


@pytest.mark.asyncio
async def test_single_iteration_answer(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test reasoning with answer found in first iteration."""
    # Setup mock to return answer immediately
    mock_llm_client.generate.return_value = GenerationResult(
        content="Let me solve this. <answer>42</answer>",
        tokens_used=500,
        finish_reason="stop",
        model="gpt-4.1",
        stop_sequence_found="<answer>")

    # Create engine
    engine = BoundedContextEngine(mock_llm_client, bounded_config)

    # Execute reasoning
    result = await engine.reason(query="What is 2+2?")

    # Assertions
    assert result.answer == "42"
    assert result.total_iterations == 1
    assert result.iterations[0].has_answer is True
    assert result.iterations[0].answer == "42"
    assert result.carryover_compressions == 0  # No carryover needed
    assert mock_llm_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_multi_iteration_reasoning(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test reasoning across multiple iterations."""
    # Setup mock to require multiple iterations
    responses = [
        # Iteration 0: No answer, continue
        GenerationResult(
            content="Step 1: Breaking down the problem... <continue>",
            tokens_used=600,
            finish_reason="stop",
            model="gpt-4.1",
            stop_sequence_found="<continue>"),
        # Iteration 1: Carryover generation (for carryover)
        GenerationResult(
            content='{"current_strategy": "Solve step by step", "key_findings": ["Problem decomposed"], "progress": "Started analysis", "next_steps": ["Calculate"], "unresolved": []}',
            tokens_used=200,
            finish_reason="stop",
            model="gpt-4.1"),
        # Iteration 2: Still working
        GenerationResult(
            content="Step 2: Calculating intermediate results... <continue>",
            tokens_used=700,
            finish_reason="stop",
            model="gpt-4.1",
            stop_sequence_found="<continue>"),
        # Iteration 3: Carryover generation
        GenerationResult(
            content='{"current_strategy": "Continue calculation", "key_findings": ["Intermediate result found"], "progress": "Halfway done", "next_steps": ["Final calculation"], "unresolved": []}',
            tokens_used=200,
            finish_reason="stop",
            model="gpt-4.1"),
        # Iteration 4: Answer found
        GenerationResult(
            content="Step 3: Final calculation complete. <answer>The answer is 42</answer>",
            tokens_used=500,
            finish_reason="stop",
            model="gpt-4.1",
            stop_sequence_found="<answer>"),
    ]

    mock_llm_client.generate.side_effect = responses

    # Create engine
    engine = BoundedContextEngine(mock_llm_client, bounded_config)

    # Execute reasoning
    result = await engine.reason(query="Complex problem")

    # Assertions
    assert result.answer == "The answer is 42"
    assert result.total_iterations == 3  # 3 reasoning iterations (0, 2, 4)
    assert result.iterations[2].has_answer is True
    assert result.carryover_compressions == 2  # 2 carryovers generated
    assert mock_llm_client.generate.call_count == 5  # 3 reasoning + 2 carryover


@pytest.mark.asyncio
async def test_max_iterations_reached(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test reasoning when max iterations reached without finding answer."""
    # Setup mock to never find answer
    def generate_no_answer(*args, **kwargs):
        return GenerationResult(
            content="Still thinking... <continue>",
            tokens_used=600,
            finish_reason="stop",
            model="gpt-4.1",
            stop_sequence_found="<continue>")

    mock_llm_client.generate.side_effect = [
        generate_no_answer(),  # Iteration 0
        GenerationResult(content='{"current_strategy": "Think", "key_findings": [], "progress": "Thinking", "next_steps": [], "unresolved": []}', tokens_used=200, finish_reason="stop", model="gpt-4.1"),
        generate_no_answer(),  # Iteration 1
        GenerationResult(content='{"current_strategy": "Think", "key_findings": [], "progress": "Thinking", "next_steps": [], "unresolved": []}', tokens_used=200, finish_reason="stop", model="gpt-4.1"),
        generate_no_answer(),  # Iteration 2
        GenerationResult(content='{"current_strategy": "Think", "key_findings": [], "progress": "Thinking", "next_steps": [], "unresolved": []}', tokens_used=200, finish_reason="stop", model="gpt-4.1"),
        generate_no_answer(),  # Iteration 3
        GenerationResult(content='{"current_strategy": "Think", "key_findings": [], "progress": "Thinking", "next_steps": [], "unresolved": []}', tokens_used=200, finish_reason="stop", model="gpt-4.1"),
        generate_no_answer(),  # Iteration 4 (last)
    ]

    # Create engine
    engine = BoundedContextEngine(mock_llm_client, bounded_config)

    # Execute reasoning
    result = await engine.reason(query="Unsolvable problem")

    # Assertions
    assert result.total_iterations == 5  # Hit max iterations
    assert all(not it.has_answer for it in result.iterations)
    assert result.carryover_compressions == 4  # Generated carryover except last
    assert "Still thinking..." in result.answer  # Uses last iteration content


@pytest.mark.asyncio
async def test_answer_extraction(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test answer extraction from different formats."""
    # Test with closing tag
    mock_llm_client.generate.return_value = GenerationResult(
        content="Here's the answer: <answer>42</answer> and some trailing text",
        tokens_used=500,
        finish_reason="stop",
        model="gpt-4.1",
        stop_sequence_found="<answer>")

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test query")

    assert result.answer == "42"

    # Test without closing tag
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>The answer is forty-two",
        tokens_used=500,
        finish_reason="stop",
        model="gpt-4.1",
        stop_sequence_found="<answer>")

    result = await engine.reason(query="Test query 2")
    assert result.answer == "The answer is forty-two"


@pytest.mark.asyncio
async def test_metrics_calculation(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test that compute savings metrics are calculated."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>Result</answer>",
        tokens_used=1000,
        finish_reason="stop",
        model="gpt-4.1",
        stop_sequence_found="<answer>")

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test")

    # Check metrics are populated
    assert result.total_tokens > 0
    assert result.compute_savings_pct >= 0.0
    assert result.execution_time_ms >= 0  # May be 0 in tests with mocked async
    assert isinstance(result.compute_savings_pct, float)


@pytest.mark.asyncio
async def test_empty_query_handling(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test handling of empty query."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>Empty query</answer>",
        tokens_used=100,
        finish_reason="stop",
        model="gpt-4.1")

    engine = BoundedContextEngine(mock_llm_client, bounded_config)

    # Should not raise error, engine handles it
    result = await engine.reason(query="")
    assert result.answer is not None


@pytest.mark.asyncio
async def test_llm_failure_handling(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test handling of LLM generation failures."""
    # Mock LLM to raise exception
    mock_llm_client.generate.side_effect = RuntimeError("LLM service unavailable")

    engine = BoundedContextEngine(mock_llm_client, bounded_config)

    # Should raise RuntimeError with iteration info
    with pytest.raises(RuntimeError, match="Iteration 0 failed"):
        await engine.reason(query="Test")


@pytest.mark.asyncio
async def test_iteration_metrics_tracking(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test that iteration metrics are properly tracked."""
    responses = [
        GenerationResult(content="Thinking... <continue>", tokens_used=500, finish_reason="stop", model="gpt-4.1"),
        GenerationResult(content='{"current_strategy": "Plan", "key_findings": [], "progress": "Working", "next_steps": [], "unresolved": []}', tokens_used=200, finish_reason="stop", model="gpt-4.1"),
        GenerationResult(content="<answer>Done</answer>", tokens_used=600, finish_reason="stop", model="gpt-4.1"),
    ]

    mock_llm_client.generate.side_effect = responses

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test")

    # Check iteration metrics
    assert len(result.iterations) == 2

    iter_0 = result.iterations[0]
    assert iter_0.metrics.iteration == 0
    assert iter_0.metrics.tokens == 500
    assert iter_0.metrics.has_answer is False
    assert iter_0.metrics.carryover_generated is True
    assert iter_0.metrics.execution_time_ms >= 0  # May be 0 in tests with mocked async

    iter_1 = result.iterations[1]
    assert iter_1.metrics.iteration == 1
    assert iter_1.metrics.tokens == 600
    assert iter_1.metrics.has_answer is True
    assert iter_1.metrics.carryover_generated is False


@pytest.mark.asyncio
async def test_carryover_parse_failure_fallback(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test fallback when carryover JSON parsing fails."""
    responses = [
        # Iteration 0: No answer, continue
        GenerationResult(
            content="Step 1: Working on it... <continue>",
            tokens_used=600,
            finish_reason="stop",
            model="gpt-4.1"),
        # Carryover generation returns invalid JSON
        GenerationResult(
            content="This is not valid JSON at all!",
            tokens_used=200,
            finish_reason="stop",
            model="gpt-4.1"),
        # Iteration 1: Answer found
        GenerationResult(
            content="<answer>Final answer</answer>",
            tokens_used=500,
            finish_reason="stop",
            model="gpt-4.1"),
    ]

    mock_llm_client.generate.side_effect = responses

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test query")

    # Should complete successfully with fallback carryover
    assert result.answer == "Final answer"
    assert result.total_iterations == 2
    # Carryover should still be generated (fallback)
    assert result.iterations[0].carryover is not None


@pytest.mark.asyncio
async def test_carryover_missing_fields_fallback(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test fallback when carryover JSON is missing required fields."""
    responses = [
        GenerationResult(
            content="Reasoning... <continue>",
            tokens_used=600,
            finish_reason="stop",
            model="gpt-4.1"),
        # Carryover with incomplete fields
        GenerationResult(
            content='{"current_strategy": "Plan", "key_findings": []}',  # Missing fields
            tokens_used=200,
            finish_reason="stop",
            model="gpt-4.1"),
        GenerationResult(
            content="<answer>Done</answer>",
            tokens_used=500,
            finish_reason="stop",
            model="gpt-4.1"),
    ]

    mock_llm_client.generate.side_effect = responses

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test")

    # Should use fallback carryover
    assert result.answer == "Done"
    assert result.iterations[0].carryover is not None


@pytest.mark.asyncio
async def test_carryover_exceeds_token_limit(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test that oversized carryover is trimmed."""
    responses = [
        GenerationResult(
            content="Step 1... <continue>",
            tokens_used=600,
            finish_reason="stop",
            model="gpt-4.1"),
        # Very large carryover
        GenerationResult(
            content='{"current_strategy": "Plan", "key_findings": ["F1", "F2", "F3", "F4", "F5"], "progress": "Progress", "next_steps": ["N1", "N2", "N3", "N4", "N5"], "unresolved": []}',
            tokens_used=200,
            finish_reason="stop",
            model="gpt-4.1"),
        GenerationResult(
            content="<answer>Result</answer>",
            tokens_used=500,
            finish_reason="stop",
            model="gpt-4.1"),
    ]

    # Mock count_tokens to simulate oversized carryover that needs trimming
    call_counts = [0]

    def mock_count(text: str) -> int:
        call_counts[0] += 1
        # First few calls return large value, then it decreases as trimming happens
        if "F1" in text and "F5" in text:
            return 10000  # Too large
        elif "F1" in text and "F4" in text:
            return 9000  # Still too large
        elif "F1" in text and "F3" in text:
            return 8000  # Still too large
        elif "F1" in text and "F2" in text:
            return 4000  # Within limit after trimming
        return 100

    mock_llm_client.count_tokens = MagicMock(side_effect=mock_count)
    mock_llm_client.generate.side_effect = responses

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test")

    # Should still complete successfully with trimmed carryover
    assert result.answer == "Result"
    assert result.iterations[0].carryover is not None
    # Carryover should have been trimmed
    assert len(result.iterations[0].carryover.key_findings) < 5


@pytest.mark.asyncio
async def test_carryover_generation_exception(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test fallback when carryover generation raises exception."""
    call_count = [0]

    def side_effect_generator(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: iteration 0
            return GenerationResult(
                content="Working... <continue>",
                tokens_used=600,
                finish_reason="stop",
                model="gpt-4.1")
        elif call_count[0] == 2:
            # Second call: carryover generation - fails
            raise RuntimeError("Carryover generation failed")
        else:
            # Third call: iteration 1
            return GenerationResult(
                content="<answer>Answer</answer>",
                tokens_used=500,
                finish_reason="stop",
                model="gpt-4.1")

    mock_llm_client.generate.side_effect = side_effect_generator

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test")

    # Should use fallback carryover
    assert result.answer == "Answer"
    assert result.iterations[0].carryover is not None


@pytest.mark.asyncio
async def test_metrics_calculator_edge_cases(
    mock_llm_client: LLMClient,
    bounded_config: BoundedContextConfig) -> None:
    """Test metrics calculation with edge cases."""
    # Test with zero iterations (edge case, shouldn't happen in practice)
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>Quick</answer>",
        tokens_used=100,
        finish_reason="stop",
        model="gpt-4.1")

    engine = BoundedContextEngine(mock_llm_client, bounded_config)
    result = await engine.reason(query="Test")

    # Single iteration should have valid metrics
    assert result.total_iterations == 1
    assert result.compute_savings_pct >= 0.0
    assert result.total_tokens > 0
