"""
Context Reasoning Framework for AgentCore.

This module provides a pluggable architecture for multiple reasoning strategies
including Chain of Thought, Bounded Context, ReAct, and Tree of Thought.

Key Components:
- ReasoningStrategy: Protocol/interface for all reasoning strategies
- ReasoningStrategyRegistry: Strategy registration and discovery
- BoundedContextEngine: Fixed-window reasoning with linear complexity scaling
- CarryoverGenerator: Compressed summary generation between iterations
- MetricsCalculator: Performance and efficiency metrics tracking
- JSON-RPC Handler: Unified reasoning.execute API endpoint

The framework supports:
- Multiple reasoning strategies running concurrently
- Configuration at system, agent, and request levels
- Optional deployment (zero strategies enabled is valid)
- Compute optimization for long-form reasoning tasks
"""

__version__ = "0.1.0"
