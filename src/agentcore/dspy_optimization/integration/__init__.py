"""
DSPy Optimization Agent Integration

Connects DSPy optimization pipeline with AgentCore runtime for real-time
agent performance optimization and feedback loops.
"""

from agentcore.dspy_optimization.integration.agent_connector import AgentRuntimeConnector
from agentcore.dspy_optimization.integration.feedback_loop import AgentPerformanceFeedbackLoop
from agentcore.dspy_optimization.integration.monitoring_hooks import OptimizationMonitor
from agentcore.dspy_optimization.integration.target_spec import AgentOptimizationTarget

__all__ = [
    "AgentRuntimeConnector",
    "AgentOptimizationTarget",
    "OptimizationMonitor",
    "AgentPerformanceFeedbackLoop",
]
