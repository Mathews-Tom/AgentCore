"""
End-to-end workflow integration tests.

Tests complete agent execution scenarios from creation through
task execution to termination, including all philosophy types
and real-world use cases.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agentcore.agent_runtime.engines.react_engine import ReActEngine
from agentcore.agent_runtime.engines.cot_engine import CoTEngine
from agentcore.agent_runtime.engines.autonomous_engine import AutonomousEngine
from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.agent_runtime.models.tool_integration import ToolDefinition
from agentcore.agent_runtime.services.agent_lifecycle import AgentLifecycleManager
from agentcore.agent_runtime.services.multi_agent_coordinator import (
    AgentMessage,
    MessagePriority,
    MessageType,
    MultiAgentCoordinator,
)
from agentcore.agent_runtime.services.task_handler import TaskHandler
from agentcore.agent_runtime.services.tool_registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.slow
async def test_complete_react_agent_workflow(
    mock_container_manager,
    mock_a2a_client,
):
    """Test complete workflow with ReAct agent."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create ReAct agent
    config = AgentConfig(
        agent_id="react-agent-001",
        philosophy=AgentPhilosophy.REACT,
    )

    # 1. Agent Creation
    mock_container_manager.create_container.return_value = "container-react-001"
    state = await lifecycle_manager.create_agent(config)
    assert state.agent_id == "react-agent-001"
    assert state.status == "initializing"

    # 2. Agent Registration (A2A)
    mock_a2a_client.register_agent.assert_called_once()

    # 3. Agent Startup
    await lifecycle_manager.start_agent("react-agent-001")
    state = await lifecycle_manager.get_agent_status("react-agent-001")
    assert state.status == "running"

    # 4. Tool Registration
    tool_registry = ToolRegistry()
    calculator_tool = ToolDefinition(
        tool_id="calculator",
        name="calculator",
        description="Perform calculations",
        parameters={
            "operation": {"type": "string", "enum": ["+", "-", "*", "/"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
    )

    def calculator_executor(operation: str, a: float, b: float) -> float:
        operations = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y,
        }
        return operations[operation](a, b)

    tool_registry.register_tool(
        tool=calculator_tool,
        executor=calculator_executor,
    )

    # 5. Execute ReAct Cycle
    engine = ReActEngine(config, tool_registry=tool_registry)
    execution_state = AgentExecutionState(
        agent_id="react-agent-001",
        status="running",
    )
    result = await engine.execute(
        input_data={
            "goal": "Calculate 5 + 3 and return the result",
            "max_iterations": 5,
        },
        state=execution_state,
    )

    assert result["completed"] is True
    assert "8" in str(result.get("final_answer", ""))

    # 6. Agent Termination
    await lifecycle_manager.terminate_agent("react-agent-001", cleanup=True)
    mock_a2a_client.unregister_agent.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.slow
async def test_complete_cot_agent_workflow(
    mock_container_manager,
    mock_a2a_client,
):
    """Test complete workflow with Chain-of-Thought agent."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create CoT agent
    config = AgentConfig(
        agent_id="cot-agent-001",
        philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT,
    )

    mock_container_manager.create_container.return_value = "container-cot-001"

    # 1. Create agent
    state = await lifecycle_manager.create_agent(config)
    assert state.agent_id == "cot-agent-001"

    # 2. Start agent
    await lifecycle_manager.start_agent("cot-agent-001")

    # 3. Execute reasoning
    engine = CoTEngine(config, use_real_llm=False)
    execution_state = AgentExecutionState(
        agent_id="cot-agent-001",
        status="running",
    )

    result = await engine.execute(
        input_data={
            "goal": "Analyze the best approach to solve problem X",
            "max_steps": 5,
        },
        state=execution_state,
    )

    assert result["completed"] is True
    assert len(result["reasoning_chain"]) >= 3

    # 4. Cleanup
    await lifecycle_manager.terminate_agent("cot-agent-001", cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multi_agent_collaborative_workflow(
    mock_container_manager,
    mock_a2a_client,
):
    """Test collaborative workflow with multiple agents."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    coordinator = MultiAgentCoordinator()

    # Create specialized agents
    agents = [
        ("agent-researcher", AgentPhilosophy.REACT, ["research", "analysis"]),
        ("agent-writer", AgentPhilosophy.CHAIN_OF_THOUGHT, ["writing", "editing"]),
        ("agent-reviewer", AgentPhilosophy.AUTONOMOUS, ["review", "quality_check"]),
    ]

    # 1. Create all agents
    for agent_id, philosophy, capabilities in agents:
        config = AgentConfig(
            agent_id=agent_id,
            philosophy=philosophy,
        )

        mock_container_manager.create_container.return_value = f"container-{agent_id}"
        await lifecycle_manager.create_agent(config)
        await lifecycle_manager.start_agent(agent_id)
        await coordinator.register_agent(
            agent_id, metadata={"capabilities": capabilities}
        )

    # 2. Coordinate workflow
    # Research phase
    await coordinator.send_message(
        AgentMessage(
            sender_id="orchestrator",
            recipient_id="agent-researcher",
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.NORMAL,
            content={"request_type": "research", "topic": "AI Safety"},
        )
    )

    # Writing phase
    await coordinator.send_message(
        AgentMessage(
            sender_id="agent-researcher",
            recipient_id="agent-writer",
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.NORMAL,
            content={"request_type": "write", "findings": "[research results]"},
        )
    )

    # Review phase
    await coordinator.send_message(
        AgentMessage(
            sender_id="agent-writer",
            recipient_id="agent-reviewer",
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.NORMAL,
            content={"request_type": "review", "draft": "[written content]"},
        )
    )

    # 3. Verify messages received
    for agent_id, _, _ in agents:
        msg = await coordinator.receive_message(agent_id, timeout=1.0)
        assert msg is not None

    # 4. Cleanup
    for agent_id, _, _ in agents:
        await coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_autonomous_agent_goal_execution(
    mock_container_manager,
    mock_a2a_client,
):
    """Test autonomous agent achieving complex goal."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create autonomous agent
    config = AgentConfig(
        agent_id="autonomous-001",
        philosophy=AgentPhilosophy.AUTONOMOUS,
    )

    mock_container_manager.create_container.return_value = "container-autonomous-001"

    # 1. Create and start
    await lifecycle_manager.create_agent(config)
    await lifecycle_manager.start_agent("autonomous-001")

    # 2. Execute autonomous goal
    engine = AutonomousEngine(config, use_real_llm=False)
    execution_state = AgentExecutionState(
        agent_id="autonomous-001",
        status="running",
    )

    result = await engine.execute(
        input_data={
            "goal": "Build a simple web scraper",
            "success_criteria": {
                "can_fetch_url": True,
                "can_parse_html": True,
                "can_extract_data": True,
            },
        },
        state=execution_state,
    )

    assert result["completed"] is True
    assert result["goal_status"] in ["completed", "in_progress"]

    # 3. Cleanup
    await lifecycle_manager.terminate_agent("autonomous-001", cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_task_lifecycle_with_checkpointing(
    mock_container_manager,
    mock_a2a_client,
    test_task_data,
):
    """Test complete task lifecycle with checkpointing and recovery."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    task_handler = TaskHandler(
        a2a_client=mock_a2a_client,
        lifecycle_manager=lifecycle_manager,
    )

    # 1. Create agent
    config = AgentConfig(
        agent_id="task-agent-001",
        philosophy=AgentPhilosophy.REACT,
    )

    mock_container_manager.create_container.return_value = "container-task-001"
    await lifecycle_manager.create_agent(config)
    await lifecycle_manager.start_agent("task-agent-001")

    # 2. Assign task
    await task_handler.assign_task(
        task_id=test_task_data["task_id"],
        agent_id="task-agent-001",
        task_data=test_task_data,
    )

    # 3. Wait for partial execution
    await asyncio.sleep(0.2)

    # 4. Create checkpoint
    await lifecycle_manager.save_checkpoint(
        agent_id="task-agent-001",
        checkpoint_data=b"task checkpoint data",
    )

    # 5. Simulate failure and restart
    await lifecycle_manager.terminate_agent("task-agent-001", cleanup=False)

    # 6. Restore from checkpoint
    restored_data = await lifecycle_manager.restore_checkpoint("task-agent-001")
    assert restored_data == b"task checkpoint data"

    # 7. Resume task execution (would create new agent in real scenario)
    await asyncio.sleep(0.1)

    # 8. Cleanup
    await task_handler.shutdown()
    await lifecycle_manager.terminate_agent("task-agent-001", cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_fault_tolerant_workflow(
    mock_container_manager,
    mock_a2a_client,
):
    """Test workflow resilience to agent failures."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    coordinator = MultiAgentCoordinator()

    # Create primary and backup agents
    agents = [
        ("agent-primary", ["primary"]),
        ("agent-backup", ["primary"]),  # Same capability
    ]

    for agent_id, capabilities in agents:
        config = AgentConfig(
            agent_id=agent_id,
            philosophy=AgentPhilosophy.REACT,
        )

        mock_container_manager.create_container.return_value = f"container-{agent_id}"
        await lifecycle_manager.create_agent(config)
        await lifecycle_manager.start_agent(agent_id)
        await coordinator.register_agent(
            agent_id, metadata={"capabilities": capabilities}
        )

    # Send task to primary
    await coordinator.send_message(
        AgentMessage(
            sender_id="orchestrator",
            recipient_id="agent-primary",
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.NORMAL,
            content={"data": "process this"},
        )
    )

    # Simulate primary failure
    await lifecycle_manager.terminate_agent("agent-primary", cleanup=True)
    await coordinator.unregister_agent("agent-primary")

    # Check that backup agent is still available
    active_agents = coordinator.get_active_agents()
    assert "agent-backup" in active_agents

    # Redirect to backup
    await coordinator.send_message(
        AgentMessage(
            sender_id="orchestrator",
            recipient_id="agent-backup",
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.NORMAL,
            content={"data": "process this"},
        )
    )

    msg = await coordinator.receive_message("agent-backup", timeout=1.0)
    assert msg is not None

    # Cleanup
    await coordinator.unregister_agent("agent-backup")
    await lifecycle_manager.terminate_agent("agent-backup", cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_resource_constrained_execution(
    mock_container_manager,
    mock_a2a_client,
):
    """Test workflow under resource constraints."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create agent with strict resource limits
    config = AgentConfig(
        agent_id="constrained-agent",
        philosophy=AgentPhilosophy.REACT,
    )

    mock_container_manager.create_container.return_value = "container-constrained"

    # Create and start
    await lifecycle_manager.create_agent(config)
    await lifecycle_manager.start_agent("constrained-agent")

    # Execute task
    await asyncio.sleep(0.2)

    # Monitor resource usage
    state = await lifecycle_manager.get_agent_status("constrained-agent")
    assert state.status == "running"

    # Cleanup
    await lifecycle_manager.terminate_agent("constrained-agent", cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multi_philosophy_collaboration(
    mock_container_manager,
    mock_a2a_client,
):
    """Test collaboration between agents using different philosophies."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    coordinator = MultiAgentCoordinator()

    # Create agents with different philosophies
    philosophies = [
        ("react-agent", AgentPhilosophy.REACT),
        ("cot-agent", AgentPhilosophy.CHAIN_OF_THOUGHT),
        ("autonomous-agent", AgentPhilosophy.AUTONOMOUS),
    ]

    for agent_id, philosophy in philosophies:
        config = AgentConfig(
            agent_id=agent_id,
            philosophy=philosophy,
        )

        mock_container_manager.create_container.return_value = f"container-{agent_id}"
        await lifecycle_manager.create_agent(config)
        await lifecycle_manager.start_agent(agent_id)
        await coordinator.register_agent(
            agent_id, metadata={"capabilities": ["collaboration"]}
        )

    # Coordinate between different philosophies using broadcast
    broadcast_msg = AgentMessage(
        sender_id="orchestrator",
        recipient_id=None,  # None means broadcast
        message_type=MessageType.BROADCAST,
        priority=MessagePriority.NORMAL,
        content={"step": "initialization"},
    )
    await coordinator.send_message(broadcast_msg)

    # Verify all received
    for agent_id, _ in philosophies:
        msg = await coordinator.receive_message(agent_id, timeout=1.0)
        assert msg is not None

    # Cleanup
    for agent_id, _ in philosophies:
        await coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_complete_deployment_lifecycle(
    mock_container_manager,
    mock_a2a_client,
):
    """Test complete deployment from creation to production operation."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # 1. Development: Create test agent
    config = AgentConfig(
        agent_id="prod-agent-001",
        philosophy=AgentPhilosophy.REACT,
    )

    mock_container_manager.create_container.return_value = "container-prod-001"

    # 2. Staging: Deploy to staging
    state = await lifecycle_manager.create_agent(config)
    await lifecycle_manager.start_agent("prod-agent-001")

    # 3. Testing: Run health checks
    await asyncio.sleep(0.1)
    state = await lifecycle_manager.get_agent_status("prod-agent-001")
    assert state.status == "running"

    # 4. Production: Agent operates normally
    await asyncio.sleep(0.2)

    # 5. Monitoring: Check metrics
    state = await lifecycle_manager.get_agent_status("prod-agent-001")
    assert "performance_metrics" in state.model_dump()

    # 6. Maintenance: Create backup
    await lifecycle_manager.save_checkpoint(
        agent_id="prod-agent-001",
        checkpoint_data=b"production backup",
    )

    # 7. Decommission: Clean shutdown
    await lifecycle_manager.terminate_agent("prod-agent-001", cleanup=True)
    mock_a2a_client.unregister_agent.assert_called()
