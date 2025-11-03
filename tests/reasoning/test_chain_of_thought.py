"""
Tests for Chain of Thought reasoning strategy.

Validates CoT strategy implementation including:
- Single-pass reasoning execution
- Answer extraction from reasoning traces
- Configuration handling
- Error scenarios
- Metrics calculation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.reasoning.engines.chain_of_thought_engine import ChainOfThoughtEngine
from agentcore.reasoning.models.reasoning_models import ChainOfThoughtConfig
from agentcore.reasoning.services.llm_client import GenerationResult


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def cot_config():
    """Create default CoT configuration."""
    return ChainOfThoughtConfig(
        max_tokens=4096,
        temperature=0.7,
        show_reasoning=True,
    )


@pytest.fixture
def cot_engine(mock_llm_client, cot_config):
    """Create CoT engine with mocked dependencies."""
    return ChainOfThoughtEngine(llm_client=mock_llm_client, config=cot_config)


@pytest.mark.asyncio
async def test_cot_basic_execution(cot_engine, mock_llm_client):
    """Test basic CoT execution with answer extraction."""
    # Setup mock response
    mock_llm_client.generate.return_value = GenerationResult(
        content="""Let me think step by step:
Step 1: Understanding the problem - We need to add 2 and 2
Step 2: Performing addition - 2 + 2 = 4
Step 3: Verification - 4 is correct

<answer>4</answer>""",
        tokens_used=150, finish_reason="stop", model="gpt-5",
    )

    # Execute
    result = await cot_engine.execute(query="What is 2+2?")

    # Verify
    assert result.answer == "4"
    assert result.strategy_used == "chain_of_thought"
    assert result.metrics.total_tokens == 150
    assert result.metrics.execution_time_ms >= 0  # Can be 0 for fast mocked calls
    assert result.trace is not None
    assert len(result.trace) == 2  # reasoning + answer
    assert result.trace[0]["type"] == "reasoning"
    assert result.trace[1]["type"] == "answer"


@pytest.mark.asyncio
async def test_cot_answer_extraction_with_prefix(cot_engine, mock_llm_client):
    """Test answer extraction when using 'Answer:' prefix."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="""Step 1: Analyze...
Step 2: Calculate...
Final Answer: The result is 42""",
        tokens_used=130, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="Test query")

    assert result.answer == "The result is 42"
    assert result.strategy_used == "chain_of_thought"


@pytest.mark.asyncio
async def test_cot_answer_extraction_fallback(cot_engine, mock_llm_client):
    """Test answer extraction fallback to last line."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="""Some reasoning here
More reasoning
Final line is the answer""",
        tokens_used=110, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="Test query")

    assert result.answer == "Final line is the answer"


@pytest.mark.asyncio
async def test_cot_with_custom_temperature(cot_engine, mock_llm_client):
    """Test CoT with custom temperature parameter."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>result</answer>",
        tokens_used=80, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="Test query", temperature=0.9)

    # Verify temperature was passed to LLM
    mock_llm_client.generate.assert_called_once()
    call_kwargs = mock_llm_client.generate.call_args.kwargs
    assert call_kwargs["temperature"] == 0.9
    assert result.metrics.strategy_specific["temperature"] == 0.9


@pytest.mark.asyncio
async def test_cot_with_custom_max_tokens(cot_engine, mock_llm_client):
    """Test CoT with custom max_tokens parameter."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>result</answer>",
        tokens_used=80, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="Test query", max_tokens=8192)

    # Verify max_tokens was passed to LLM
    call_kwargs = mock_llm_client.generate.call_args.kwargs
    assert call_kwargs["max_tokens"] == 8192
    assert result.metrics.strategy_specific["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_cot_without_reasoning_trace(cot_engine, mock_llm_client):
    """Test CoT execution without including reasoning trace."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="Reasoning...<answer>final answer</answer>",
        tokens_used=120, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="Test query", show_reasoning=False)

    assert result.answer == "final answer"
    assert result.trace is None


@pytest.mark.asyncio
async def test_cot_with_custom_system_prompt(cot_engine, mock_llm_client):
    """Test CoT with custom system prompt."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>answer</answer>",
        tokens_used=100, finish_reason="stop", model="gpt-5",
    )

    custom_prompt = "You are a math expert. Solve problems step by step."
    result = await cot_engine.execute(
        query="What is 5+5?",
        system_prompt=custom_prompt,
    )

    # Verify custom prompt was used
    call_args = mock_llm_client.generate.call_args
    assert custom_prompt in call_args.kwargs["prompt"]
    assert result.answer == "answer"


@pytest.mark.asyncio
async def test_cot_llm_failure_handling(cot_engine, mock_llm_client):
    """Test error handling when LLM fails."""
    mock_llm_client.generate.side_effect = RuntimeError("LLM API error")

    with pytest.raises(RuntimeError, match="Chain of Thought reasoning failed"):
        await cot_engine.execute(query="Test query")


@pytest.mark.asyncio
async def test_cot_metrics_calculation(cot_engine, mock_llm_client):
    """Test that metrics are correctly calculated."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>test</answer>",
        tokens_used=300, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="Test query")

    assert result.metrics.total_tokens == 300
    assert result.metrics.execution_time_ms >= 0  # Can be 0 for fast mocked calls
    assert result.metrics.strategy_specific["finish_reason"] == "stop"
    assert result.metrics.strategy_specific["model"] == "gpt-5"


def test_cot_strategy_metadata(cot_engine):
    """Test strategy metadata properties."""
    assert cot_engine.name == "chain_of_thought"
    assert cot_engine.version == "1.0.0"

    capabilities = cot_engine.get_capabilities()
    assert "reasoning.strategy.chain_of_thought" in capabilities

    schema = cot_engine.get_config_schema()
    assert schema["type"] == "object"
    assert "max_tokens" in schema["properties"]
    assert "temperature" in schema["properties"]


def test_cot_config_validation():
    """Test configuration validation."""
    # Valid config
    config = ChainOfThoughtConfig(max_tokens=2048, temperature=0.5)
    assert config.max_tokens == 2048
    assert config.temperature == 0.5

    # Test defaults
    config_default = ChainOfThoughtConfig()
    assert config_default.max_tokens == 4096
    assert config_default.temperature == 0.7
    assert config_default.show_reasoning is True


def test_cot_answer_extraction_edge_cases(cot_engine):
    """Test answer extraction with various edge cases."""
    # Empty content
    assert cot_engine._extract_answer("") == ""

    # Nested answer tags
    content = "<answer>outer <answer>inner</answer> end</answer>"
    answer = cot_engine._extract_answer(content)
    assert "inner" in answer or "outer" in answer

    # Multiple answer tags (should extract first)
    content = "<answer>first</answer> some text <answer>second</answer>"
    assert cot_engine._extract_answer(content) == "first"

    # Case insensitive tags
    content = "<ANSWER>caps answer</ANSWER>"
    assert cot_engine._extract_answer(content) == "caps answer"


@pytest.mark.asyncio
async def test_cot_long_query_handling(cot_engine, mock_llm_client):
    """Test handling of very long queries."""
    long_query = "x" * 50000

    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>processed</answer>",
        tokens_used=10100, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query=long_query)

    assert result.answer == "processed"
    assert result.metrics.total_tokens == 10100


@pytest.mark.asyncio
async def test_cot_empty_query_handling(cot_engine, mock_llm_client):
    """Test handling of empty query."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="<answer>No query provided</answer>",
        tokens_used=50, finish_reason="stop", model="gpt-5",
    )

    result = await cot_engine.execute(query="")

    assert result.answer is not None
    assert result.metrics.total_tokens > 0
