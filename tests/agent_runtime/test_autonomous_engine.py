"""Tests for Autonomous Agent engine."""

import pytest

from agentcore.agent_runtime.engines.autonomous_engine import AutonomousEngine
from agentcore.agent_runtime.engines.autonomous_models import (
    GoalPriority,
    GoalStatus,
    MemoryType)
from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.models.agent_state import AgentExecutionState


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-autonomous-agent",
        philosophy=AgentPhilosophy.AUTONOMOUS)


@pytest.fixture
def agent_state() -> AgentExecutionState:
    """Create test agent state."""
    return AgentExecutionState(
        agent_id="test-autonomous-agent",
        status="running")


@pytest.fixture
async def autonomous_engine(agent_config: AgentConfig) -> AutonomousEngine:
    """Create Autonomous engine instance."""
    # Use simulated LLM for testing
    engine = AutonomousEngine(agent_config, use_real_llm=False)
    await engine.initialize()
    return engine


@pytest.mark.asyncio
class TestAutonomousEngine:
    """Test suite for Autonomous engine."""

    async def test_engine_initialization(self, agent_config: AgentConfig) -> None:
        """Test engine initialization."""
        engine = AutonomousEngine(agent_config, use_real_llm=False)
        await engine.initialize()

        assert engine.agent_id == "test-autonomous-agent"
        assert engine.config == agent_config
        assert engine.prompt_template is not None
        assert engine.context is None
        assert engine.use_real_llm is False

        await engine.cleanup()

    async def test_simple_goal_execution(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test simple goal execution."""
        result = await autonomous_engine.execute(
            input_data={
                "goal": "Complete a simple task",
                "priority": "medium",
            },
            state=agent_state)

        assert result["completed"] is True
        assert result["goal_status"] == GoalStatus.COMPLETED.value
        assert result["goal_progress"] == 1.0
        assert len(result["decisions_made"]) > 0
        assert result["memories_created"] > 0

        await autonomous_engine.cleanup()

    async def test_high_priority_goal(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test high priority goal execution."""
        result = await autonomous_engine.execute(
            input_data={
                "goal": "Critical task",
                "priority": "critical",
            },
            state=agent_state)

        assert result["completed"] is True
        assert autonomous_engine.context is not None
        assert autonomous_engine.context.primary_goal.priority == GoalPriority.CRITICAL

        await autonomous_engine.cleanup()

    async def test_goal_with_success_criteria(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test goal with success criteria."""
        result = await autonomous_engine.execute(
            input_data={
                "goal": "Task with criteria",
                "success_criteria": {
                    "metric": "accuracy",
                    "threshold": 0.95,
                },
            },
            state=agent_state)

        assert result["completed"] is True
        assert autonomous_engine.context is not None
        assert autonomous_engine.context.primary_goal.success_criteria is not None

        await autonomous_engine.cleanup()

    async def test_complex_goal_decomposition(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test complex goal decomposition into sub-goals."""
        result = await autonomous_engine.execute(
            input_data={
                "goal": "Build a complex system and deploy it and test it thoroughly",
                # Long goal with "and" - should trigger decomposition
            },
            state=agent_state)

        assert result["completed"] is True
        assert autonomous_engine.context is not None

        # Check that sub-goals were created
        primary_goal = autonomous_engine.context.primary_goal
        assert len(primary_goal.sub_goals) > 0

        await autonomous_engine.cleanup()

    async def test_decision_lineage_tracking(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test decision lineage tracking."""
        result = await autonomous_engine.execute(
            input_data={"goal": "Make multiple decisions"},
            state=agent_state)

        decisions = result["decisions_made"]
        assert len(decisions) > 0

        # Check decision structure
        for decision in decisions:
            assert "decision_id" in decision
            assert "goal_id" in decision
            assert "description" in decision
            assert "rationale" in decision
            assert "confidence" in decision
            assert "timestamp" in decision

        await autonomous_engine.cleanup()

    async def test_memory_creation(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test memory creation and storage."""
        result = await autonomous_engine.execute(
            input_data={"goal": "Create memories"},
            state=agent_state)

        assert autonomous_engine.context is not None
        assert len(autonomous_engine.context.long_term_memory) > 0

        # Check memory types
        memories = autonomous_engine.context.long_term_memory
        memory_types = {mem.memory_type for mem in memories}
        assert MemoryType.EPISODIC in memory_types or MemoryType.SEMANTIC in memory_types

        await autonomous_engine.cleanup()

    async def test_learning_experiences(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test learning experience recording."""
        result = await autonomous_engine.execute(
            input_data={"goal": "Learn from experience"},
            state=agent_state)

        experiences = result["learning_experiences"]
        assert len(experiences) > 0

        # Check learning experience structure
        for exp in experiences:
            assert "experience_id" in exp
            assert "goal_id" in exp
            assert "action_taken" in exp
            assert "outcome" in exp
            assert "success" in exp
            assert "lesson_learned" in exp

        await autonomous_engine.cleanup()

    async def test_execution_plan_creation(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test execution plan creation."""
        await autonomous_engine.execute(
            input_data={"goal": "Test planning"},
            state=agent_state)

        assert autonomous_engine.context is not None
        assert autonomous_engine.context.current_plan is not None

        plan = autonomous_engine.context.current_plan
        assert plan.goal_id == autonomous_engine.context.primary_goal.goal_id
        assert len(plan.steps) > 0
        assert plan.estimated_duration > 0

        await autonomous_engine.cleanup()

    async def test_goal_progress_tracking(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test goal progress tracking."""
        result = await autonomous_engine.execute(
            input_data={"goal": "Track progress"},
            state=agent_state)

        assert result["goal_progress"] >= 0.0
        assert result["goal_progress"] <= 1.0

        if result["completed"]:
            assert result["goal_progress"] == 1.0

        await autonomous_engine.cleanup()

    async def test_working_memory_limit(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test working memory size limit."""
        await autonomous_engine.execute(
            input_data={"goal": "Test memory limits"},
            state=agent_state)

        assert autonomous_engine.context is not None

        # Working memory should be limited
        working_memory = autonomous_engine.context.working_memory
        assert len(working_memory) <= 10  # Max size defined in engine

        await autonomous_engine.cleanup()

    async def test_goal_status_transitions(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test goal status transitions."""
        await autonomous_engine.execute(
            input_data={"goal": "Test status transitions"},
            state=agent_state)

        assert autonomous_engine.context is not None
        primary_goal = autonomous_engine.context.primary_goal

        # Goal should have transitioned through states
        assert primary_goal.started_at is not None
        assert primary_goal.status in [GoalStatus.COMPLETED, GoalStatus.FAILED]

        if primary_goal.status == GoalStatus.COMPLETED:
            assert primary_goal.completed_at is not None

        await autonomous_engine.cleanup()

    async def test_error_handling(
        self,
        agent_config: AgentConfig,
        agent_state: AgentExecutionState) -> None:
        """Test error handling."""
        engine = AutonomousEngine(agent_config, use_real_llm=False)
        await engine.initialize()

        # Test method calls without context should fail
        with pytest.raises(RuntimeError, match="Execution context not initialized"):
            await engine._plan_goal_execution(None)  # type: ignore

        await engine.cleanup()

    async def test_goal_decomposition_logic(
        self,
        autonomous_engine: AutonomousEngine) -> None:
        """Test goal decomposition logic."""
        from agentcore.agent_runtime.engines.autonomous_models import Goal

        # Simple goal should not be decomposed
        simple_goal = Goal(
            goal_id="simple",
            description="Simple task")
        is_complex = await autonomous_engine._is_complex_goal(simple_goal)
        assert is_complex is False

        # Complex goal should be decomposed
        complex_goal = Goal(
            goal_id="complex",
            description="This is a very long and complex goal that requires multiple steps and careful planning")
        is_complex = await autonomous_engine._is_complex_goal(complex_goal)
        assert is_complex is True

        # Goal with "and" should be decomposed
        compound_goal = Goal(
            goal_id="compound",
            description="Do task A and task B")
        is_complex = await autonomous_engine._is_complex_goal(compound_goal)
        assert is_complex is True

        await autonomous_engine.cleanup()

    async def test_memory_access_tracking(
        self,
        autonomous_engine: AutonomousEngine,
        agent_state: AgentExecutionState) -> None:
        """Test memory access counting."""
        await autonomous_engine.execute(
            input_data={"goal": "Test memory access"},
            state=agent_state)

        assert autonomous_engine.context is not None

        # Memories should have access counts
        for memory in autonomous_engine.context.long_term_memory:
            assert memory.access_count > 0
            assert memory.last_accessed is not None

        await autonomous_engine.cleanup()
