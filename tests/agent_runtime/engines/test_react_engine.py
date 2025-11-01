"""Tests for ReAct philosophy engine."""

import pytest

from agentcore.agent_runtime.engines.react_engine import ReActEngine
from agentcore.agent_runtime.engines.react_models import ReActStepType
from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from agentcore.agent_runtime.services.tool_registry import ToolRegistry


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create tool registry with test tools."""
    registry = ToolRegistry()

    # Register calculator tool
    async def calculator(operation: str, a: float, b: float) -> float:
        ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y}
        return ops[operation](a, b)

    calculator_def = ToolDefinition(
        tool_id="calculator",
        name="calculator",
        description="Basic calculator",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "operation": ToolParameter(
                name="operation",
                type="string",
                description="Operation to perform",
                required=True,
            ),
            "a": ToolParameter(
                name="a",
                type="number",
                description="First number",
                required=True,
            ),
            "b": ToolParameter(
                name="b",
                type="number",
                description="Second number",
                required=True,
            ),
        },
        auth_method=AuthMethod.NONE,
    )
    registry.register_tool(calculator_def, calculator)

    return registry


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="react-test-agent",
        philosophy=AgentPhilosophy.REACT)


@pytest.fixture
def agent_state() -> AgentExecutionState:
    """Create test agent state."""
    return AgentExecutionState(
        agent_id="react-test-agent",
        status="running")


@pytest.fixture
def react_engine(agent_config: AgentConfig, tool_registry: ToolRegistry) -> ReActEngine:
    """Create ReAct engine instance."""
    return ReActEngine(agent_config, tool_registry)


@pytest.mark.asyncio
async def test_react_engine_initialization(react_engine: ReActEngine) -> None:
    """Test ReAct engine initialization."""
    await react_engine.initialize()
    assert react_engine.agent_id == "react-test-agent"
    assert react_engine.tool_registry is not None


@pytest.mark.asyncio
async def test_react_engine_cleanup(react_engine: ReActEngine) -> None:
    """Test ReAct engine cleanup."""
    await react_engine.initialize()
    await react_engine.cleanup()
    # Should not raise any errors


@pytest.mark.asyncio
async def test_react_execution_simple_goal(
    react_engine: ReActEngine,
    agent_state: AgentExecutionState) -> None:
    """Test ReAct execution with simple goal."""
    await react_engine.initialize()

    input_data = {
        "goal": "Calculate 5 + 3",
        "max_iterations": 5,
    }

    result = await react_engine.execute(input_data, agent_state)

    assert result["completed"] is True
    assert result["final_answer"] is not None
    assert result["iterations"] > 0
    assert len(result["steps"]) > 0


@pytest.mark.asyncio
async def test_react_execution_steps(
    react_engine: ReActEngine,
    agent_state: AgentExecutionState) -> None:
    """Test ReAct execution generates correct step types."""
    await react_engine.initialize()

    input_data = {
        "goal": "Calculate 10 + 5",
        "max_iterations": 5,
    }

    result = await react_engine.execute(input_data, agent_state)

    # Check step types
    steps = result["steps"]
    step_types = [step["step_type"] for step in steps]

    # Should have at least: THOUGHT, ACTION, OBSERVATION
    assert ReActStepType.THOUGHT.value in step_types
    assert ReActStepType.ACTION.value in step_types or ReActStepType.FINAL_ANSWER.value in step_types


@pytest.mark.asyncio
async def test_react_max_iterations(
    react_engine: ReActEngine,
    agent_state: AgentExecutionState) -> None:
    """Test ReAct respects max iterations."""
    await react_engine.initialize()

    input_data = {
        "goal": "Complex task that won't complete",
        "max_iterations": 2,
    }

    result = await react_engine.execute(input_data, agent_state)

    assert result["iterations"] <= 2


@pytest.mark.asyncio
async def test_react_parse_action(react_engine: ReActEngine) -> None:
    """Test action parsing from thought."""
    thought = "THOUGHT: I need to calculate\nACTION: calculator(operation='+', a=5, b=3)"

    action = react_engine._parse_action(thought)

    assert action is not None
    assert action.tool_name == "calculator"
    assert action.parameters["operation"] == "+"
    assert action.parameters["a"] == 5
    assert action.parameters["b"] == 3


@pytest.mark.asyncio
async def test_react_parse_parameters(react_engine: ReActEngine) -> None:
    """Test parameter parsing."""
    params_str = "operation='+', a=10, b=20"

    parameters = react_engine._parse_parameters(params_str)

    assert parameters["operation"] == "+"
    assert parameters["a"] == 10
    assert parameters["b"] == 20


@pytest.mark.asyncio
async def test_react_is_final_answer(react_engine: ReActEngine) -> None:
    """Test final answer detection."""
    thought_with_answer = "THOUGHT: I have the answer\nFINAL_ANSWER: The result is 42"
    thought_without_answer = "THOUGHT: I need to think more\nACTION: calculator()"

    assert react_engine._is_final_answer(thought_with_answer) is True
    assert react_engine._is_final_answer(thought_without_answer) is False


@pytest.mark.asyncio
async def test_react_extract_final_answer(react_engine: ReActEngine) -> None:
    """Test final answer extraction."""
    thought = "THOUGHT: Done\nFINAL_ANSWER: The answer is 42"

    answer = react_engine._extract_final_answer(thought)

    assert "42" in answer


@pytest.mark.asyncio
async def test_react_execution_with_time_tool(
    react_engine: ReActEngine,
    agent_state: AgentExecutionState) -> None:
    """Test ReAct execution with time tool."""
    await react_engine.initialize()

    input_data = {
        "goal": "What time is it?",
        "max_iterations": 5,
    }

    result = await react_engine.execute(input_data, agent_state)

    assert result["completed"] is True
    assert result["final_answer"] is not None
