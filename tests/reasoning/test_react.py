"""
Tests for ReAct (Reasoning + Acting) reasoning strategy.

Validates ReAct strategy implementation including:
- Multi-iteration thought-action-observation cycles
- Action execution and observation handling
- Answer extraction from iterations
- Configuration handling
- Error scenarios
- Metrics calculation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.reasoning.engines.react_engine import ReActEngine
from agentcore.reasoning.models.reasoning_models import ReActConfig
from agentcore.reasoning.services.llm_client import GenerationResult


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def react_config():
    """Create default ReAct configuration."""
    return ReActConfig(
        max_iterations=10,
        max_tokens_per_step=2048,
        temperature=0.7,
        allow_tool_use=False,
    )


@pytest.fixture
def react_engine(mock_llm_client, react_config):
    """Create ReAct engine with mocked dependencies."""
    return ReActEngine(llm_client=mock_llm_client, config=react_config)


@pytest.mark.asyncio
async def test_react_basic_execution(react_engine, mock_llm_client):
    """Test basic ReAct execution with answer extraction."""
    # Setup mock responses for iterations
    mock_llm_client.generate.side_effect = [
        # Iteration 1: Initial thought and action
        GenerationResult(
            content="""Thought: I need to calculate 2+2
Action: Calculate
Observation: """,
            tokens_used=80, finish_reason="stop", model="gpt-5",
        ),
        # Iteration 2: Final answer
        GenerationResult(
            content="""Thought: I now have enough information to answer.
Action: Answer
Observation: <answer>4</answer>""",
            tokens_used=120, finish_reason="stop", model="gpt-5",
        ),
    ]

    # Execute
    result = await react_engine.execute(query="What is 2+2?")

    # Verify
    assert result.answer == "4"
    assert result.strategy_used == "react"
    assert result.metrics.total_tokens == 200  # 80 + 120
    assert result.metrics.execution_time_ms > 0
    assert len(result.trace) == 2  # 2 iterations
    assert result.trace[0]["iteration"] == 0
    assert result.trace[1]["iteration"] == 1
    assert result.trace[1]["answer_found"] is True


@pytest.mark.asyncio
async def test_react_single_iteration_answer(react_engine, mock_llm_client):
    """Test ReAct with answer in first iteration."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="""Thought: This is a simple question.
Action: Answer
Observation: <answer>42</answer>""",
        tokens_used=70, finish_reason="stop", model="gpt-5",
    )

    result = await react_engine.execute(query="What is the answer?")

    assert result.answer == "42"
    assert len(result.trace) == 1
    assert result.metrics.strategy_specific["total_iterations"] == 1
    assert result.metrics.strategy_specific["answer_found_at_iteration"] == 0


@pytest.mark.asyncio
async def test_react_max_iterations_reached(react_engine, mock_llm_client):
    """Test ReAct when max iterations is reached."""
    # Return responses without answer for all iterations
    mock_llm_client.generate.return_value = GenerationResult(
        content="""Thought: Still thinking...
Action: Search
Observation: No result yet""",
        tokens_used=80, finish_reason="stop", model="gpt-5",
    )

    result = await react_engine.execute(query="Complex question", max_iterations=3)

    # Should use last observation as answer
    assert result.answer is not None
    assert len(result.trace) == 3
    assert result.metrics.strategy_specific["total_iterations"] == 3
    assert result.metrics.strategy_specific["answer_found_at_iteration"] is None


@pytest.mark.asyncio
async def test_react_action_execution(react_engine, mock_llm_client):
    """Test that actions are executed and observations generated."""
    mock_llm_client.generate.side_effect = [
        GenerationResult(
            content="""Thought: Need to search for information
Action: Search for Python documentation
Observation: """,
            tokens_used=70, finish_reason="stop", model="gpt-5",
        ),
        GenerationResult(
            content="""Thought: Got the information
Action: Answer
Observation: <answer>Python is a programming language</answer>""",
            tokens_used=110, finish_reason="stop", model="gpt-5",
        ),
    ]

    result = await react_engine.execute(query="What is Python?")

    assert result.answer == "Python is a programming language"
    assert len(result.trace) == 2
    # First iteration should have simulated action result
    assert "Simulated" in result.trace[0]["observation"]


@pytest.mark.asyncio
async def test_react_with_custom_parameters(react_engine, mock_llm_client):
    """Test ReAct with custom parameters."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="""Thought: Quick answer
Action: Answer
Observation: <answer>result</answer>""",
        tokens_used=90, finish_reason="stop", model="gpt-5",
    )

    result = await react_engine.execute(
        query="Test query",
        max_iterations=5,
        temperature=0.9,
        max_tokens_per_step=1024,
    )

    assert result.answer == "result"
    # Verify parameters were used
    call_kwargs = mock_llm_client.generate.call_args.kwargs
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["max_tokens"] == 1024
    assert result.metrics.strategy_specific["max_iterations"] == 5


@pytest.mark.asyncio
async def test_react_step_parsing(react_engine):
    """Test parsing of ReAct steps."""
    # Test complete step
    content = """Thought: I need to think about this
Action: Do something
Observation: Got a result"""

    step = react_engine._parse_step(content)

    assert step["thought"] == "I need to think about this"
    assert step["action"] == "Do something"
    assert step["observation"] == "Got a result"

    # Test partial step (missing observation)
    content = """Thought: Thinking
Action: Acting"""

    step = react_engine._parse_step(content)

    assert step["thought"] == "Thinking"
    assert step["action"] == "Acting"
    assert step["observation"] == ""


@pytest.mark.asyncio
async def test_react_llm_failure_handling(react_engine, mock_llm_client):
    """Test error handling when LLM fails."""
    mock_llm_client.generate.side_effect = RuntimeError("LLM API error")

    with pytest.raises(RuntimeError, match="ReAct iteration 0 failed"):
        await react_engine.execute(query="Test query")


@pytest.mark.asyncio
async def test_react_metrics_calculation(react_engine, mock_llm_client):
    """Test that metrics are correctly calculated."""
    mock_llm_client.generate.side_effect = [
        GenerationResult(
            content="Thought: Step 1\nAction: Search\nObservation: ",
            tokens_used=90, finish_reason="stop", model="gpt-5",
        ),
        GenerationResult(
            content="Thought: Step 2\nAction: Answer\nObservation: <answer>done</answer>",
            tokens_used=140, finish_reason="stop", model="gpt-5",
        ),
    ]

    result = await react_engine.execute(query="Test query")

    assert result.metrics.total_tokens == 230  # 90 + 140
    assert result.metrics.execution_time_ms >= 0  # Can be 0 for fast mocked calls
    assert result.metrics.strategy_specific["total_iterations"] == 2
    assert result.metrics.strategy_specific["answer_found_at_iteration"] == 1


def test_react_strategy_metadata(react_engine):
    """Test strategy metadata properties."""
    assert react_engine.name == "react"
    assert react_engine.version == "1.0.0"

    capabilities = react_engine.get_capabilities()
    assert "reasoning.strategy.react" in capabilities

    schema = react_engine.get_config_schema()
    assert schema["type"] == "object"
    assert "max_iterations" in schema["properties"]
    assert "temperature" in schema["properties"]
    assert "allow_tool_use" in schema["properties"]


def test_react_config_validation():
    """Test configuration validation."""
    # Valid config
    config = ReActConfig(
        max_iterations=5,
        max_tokens_per_step=1024,
        temperature=0.5,
    )
    assert config.max_iterations == 5
    assert config.max_tokens_per_step == 1024
    assert config.temperature == 0.5

    # Test defaults
    config_default = ReActConfig()
    assert config_default.max_iterations == 10
    assert config_default.max_tokens_per_step == 2048
    assert config_default.temperature == 0.7
    assert config_default.allow_tool_use is False


def test_react_tool_use_capability(react_config):
    """Test tool use capability flag."""
    # Without tool use
    engine_no_tools = ReActEngine(llm_client=MagicMock(), config=react_config)
    capabilities_no_tools = engine_no_tools.get_capabilities()
    assert "reasoning.strategy.react" in capabilities_no_tools
    assert "reasoning.action.tool_use" not in capabilities_no_tools

    # With tool use
    config_with_tools = ReActConfig(allow_tool_use=True)
    engine_with_tools = ReActEngine(llm_client=MagicMock(), config=config_with_tools)
    capabilities_with_tools = engine_with_tools.get_capabilities()
    assert "reasoning.strategy.react" in capabilities_with_tools
    assert "reasoning.action.tool_use" in capabilities_with_tools


def test_react_answer_extraction(react_engine):
    """Test answer extraction from various formats."""
    # Standard answer tags
    content = "Some text <answer>the answer</answer> more text"
    assert react_engine._extract_answer(content) == "the answer"

    # Case insensitive
    content = "<ANSWER>caps</ANSWER>"
    assert react_engine._extract_answer(content) == "caps"

    # No answer
    content = "No answer here"
    assert react_engine._extract_answer(content) is None


def test_react_action_execution_types(react_engine):
    """Test different action types generate appropriate observations."""
    # Search action
    obs = react_engine._execute_action("Search for Python docs")
    assert "search" in obs.lower() or "Simulated" in obs

    # Calculate action
    obs = react_engine._execute_action("Calculate 5+5")
    assert "calculation" in obs.lower() or "Simulated" in obs

    # Answer action
    obs = react_engine._execute_action("Answer: final result")
    assert "Answer ready" in obs

    # Generic action
    obs = react_engine._execute_action("Do something custom")
    assert "Simulated" in obs


@pytest.mark.asyncio
async def test_react_multi_iteration_trace(react_engine, mock_llm_client):
    """Test that trace captures all iterations correctly."""
    mock_llm_client.generate.side_effect = [
        GenerationResult(
            content="Thought: T1\nAction: A1\nObservation: O1",
            tokens_used=70, finish_reason="stop", model="gpt-5",
        ),
        GenerationResult(
            content="Thought: T2\nAction: A2\nObservation: O2",
            tokens_used=100, finish_reason="stop", model="gpt-5",
        ),
        GenerationResult(
            content="Thought: T3\nAction: Answer\nObservation: <answer>done</answer>",
            tokens_used=140, finish_reason="stop", model="gpt-5",
        ),
    ]

    result = await react_engine.execute(query="Test query")

    assert len(result.trace) == 3
    assert result.trace[0]["thought"] == "T1"
    assert result.trace[1]["thought"] == "T2"
    assert result.trace[2]["thought"] == "T3"
    assert result.trace[2]["answer_found"] is True


@pytest.mark.asyncio
async def test_react_with_custom_system_prompt(react_engine, mock_llm_client):
    """Test ReAct with custom system prompt."""
    mock_llm_client.generate.return_value = GenerationResult(
        content="Thought: Quick\nAction: Answer\nObservation: <answer>done</answer>",
        tokens_used=100, finish_reason="stop", model="gpt-5",
    )

    custom_prompt = "You are an expert system. Use ReAct framework."
    result = await react_engine.execute(
        query="Test query",
        system_prompt=custom_prompt,
    )

    # Verify custom prompt was used
    call_args = mock_llm_client.generate.call_args
    assert custom_prompt in call_args.kwargs["prompt"]
    assert result.answer == "done"
