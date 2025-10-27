"""Tests for Chain-of-Thought (CoT) engine."""

import pytest

from agentcore.agent_runtime.engines.cot_engine import CoTEngine
from agentcore.agent_runtime.engines.cot_models import CoTStepType
from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.models.agent_state import AgentExecutionState


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-cot-agent",
        philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT)


@pytest.fixture
def agent_state() -> AgentExecutionState:
    """Create test agent state."""
    return AgentExecutionState(
        agent_id="test-cot-agent",
        status="running")


@pytest.fixture
async def cot_engine(agent_config: AgentConfig) -> CoTEngine:
    """Create CoT engine instance."""
    # Use simulated LLM for testing
    engine = CoTEngine(agent_config, use_real_llm=False)
    await engine.initialize()
    return engine


@pytest.mark.asyncio
class TestCoTEngine:
    """Test suite for CoT engine."""

    async def test_engine_initialization(self, agent_config: AgentConfig) -> None:
        """Test engine initialization."""
        engine = CoTEngine(agent_config, use_real_llm=False)
        await engine.initialize()

        assert engine.agent_id == "test-cot-agent"
        assert engine.config == agent_config
        assert engine.prompt_template is not None
        assert engine.context is None
        assert engine.use_real_llm is False

        await engine.cleanup()

    async def test_simple_execution(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test simple goal execution."""
        result = await cot_engine.execute(
            input_data={
                "goal": "Calculate the sum of 5 and 3",
                "max_steps": 5,
            },
            state=agent_state)

        assert result["completed"] is True
        assert result["final_conclusion"] is not None
        assert result["steps"] > 0
        assert len(result["reasoning_chain"]) > 0

        await cot_engine.cleanup()

    async def test_execution_with_verification(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test execution with verification enabled."""
        result = await cot_engine.execute(
            input_data={
                "goal": "Analyze the benefits of cloud computing",
                "max_steps": 5,
                "verification_enabled": True,
            },
            state=agent_state)

        assert result["completed"] is True
        assert result["verification_enabled"] is True
        assert len(result["reasoning_chain"]) > 0

        # Check for verification steps
        chain = result["reasoning_chain"]
        verification_steps = [
            step for step in chain if step["step_type"] == CoTStepType.VERIFICATION.value
        ]
        assert len(verification_steps) > 0

        await cot_engine.cleanup()

    async def test_execution_without_verification(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test execution without verification."""
        result = await cot_engine.execute(
            input_data={
                "goal": "Simple task",
                "max_steps": 3,
                "verification_enabled": False,
            },
            state=agent_state)

        assert result["completed"] is True
        assert result["verification_enabled"] is False

        # No verification steps should exist
        chain = result["reasoning_chain"]
        verification_steps = [
            step for step in chain if step["step_type"] == CoTStepType.VERIFICATION.value
        ]
        assert len(verification_steps) == 0

        await cot_engine.cleanup()

    async def test_max_steps_limit(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test max steps limit."""
        result = await cot_engine.execute(
            input_data={
                "goal": "Very complex goal that requires many steps",
                "max_steps": 2,  # Very low limit
                "verification_enabled": False,  # Disable to test step limit directly
            },
            state=agent_state)

        assert result["steps"] <= 2
        # May not be completed due to step limit
        assert result["final_conclusion"] is not None

        await cot_engine.cleanup()

    async def test_reasoning_chain_structure(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test reasoning chain structure."""
        result = await cot_engine.execute(
            input_data={
                "goal": "Test goal",
                "max_steps": 5,
                "verification_enabled": True,
            },
            state=agent_state)

        chain = result["reasoning_chain"]
        assert len(chain) > 0

        # Check each step has required fields
        for step in chain:
            assert "step_number" in step
            assert "step_type" in step
            assert "content" in step
            assert "confidence" in step
            assert "timestamp" in step

            # Check step types are valid
            assert step["step_type"] in [
                CoTStepType.STEP.value,
                CoTStepType.VERIFICATION.value,
                CoTStepType.REFINEMENT.value,
                CoTStepType.CONCLUSION.value,
            ]

        # Last step should be conclusion
        assert chain[-1]["step_type"] == CoTStepType.CONCLUSION.value

        await cot_engine.cleanup()

    async def test_context_window_management(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test context window management."""
        result = await cot_engine.execute(
            input_data={
                "goal": "Goal with multiple steps",
                "max_steps": 10,
            },
            state=agent_state)

        # Context window should be populated during execution
        assert cot_engine.context is not None
        assert len(cot_engine.context.context_window) <= 5  # Max window size

        await cot_engine.cleanup()

    async def test_conclusion_detection(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test conclusion detection and extraction."""
        # Test with various conclusion formats
        test_conclusions = [
            "CONCLUSION: This is the final answer",
            "FINAL ANSWER: Here is the result",
            "In conclusion, the solution is X",
            "Therefore, we can determine that Y",
        ]

        for conclusion_text in test_conclusions:
            assert cot_engine._is_conclusion(conclusion_text)

        # Test extraction
        extracted = cot_engine._extract_conclusion("CONCLUSION: The answer is 42")
        assert "42" in extracted
        assert "CONCLUSION:" not in extracted

        await cot_engine.cleanup()

    async def test_error_handling(
        self,
        agent_config: AgentConfig,
        agent_state: AgentExecutionState) -> None:
        """Test error handling."""
        engine = CoTEngine(agent_config, use_real_llm=False)
        await engine.initialize()

        # Test execution without context initialization should fail
        with pytest.raises(RuntimeError, match="Execution context not initialized"):
            await engine._generate_step()

        await engine.cleanup()

    async def test_history_formatting(
        self,
        cot_engine: CoTEngine,
        agent_state: AgentExecutionState) -> None:
        """Test history formatting."""
        # Execute to populate history
        await cot_engine.execute(
            input_data={"goal": "Test goal", "max_steps": 3},
            state=agent_state)

        history = cot_engine._format_history()
        assert isinstance(history, str)
        assert len(history) > 0

        await cot_engine.cleanup()
