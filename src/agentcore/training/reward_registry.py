"""
Custom reward function registry for agent training.

Enables domain-specific reward strategies with validation and per-agent-type configuration.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import structlog

from agentcore.training.models import Trajectory

logger = structlog.get_logger()


class RewardFunction(Protocol):
    """Protocol for custom reward functions."""

    def __call__(self, trajectory: Trajectory) -> float:
        """
        Compute reward for trajectory.

        Args:
            trajectory: Trajectory to evaluate

        Returns:
            Reward value in range [0, 1]
        """
        ...


class RewardValidationError(Exception):
    """Raised when reward function validation fails."""

    pass


class RewardRegistry:
    """
    Registry for managing custom reward functions with validation and agent-type strategies.

    Supports:
    - Custom reward function registration with validation
    - Per-agent-type reward strategies
    - Default fallback strategies
    - Runtime reward function validation (output range [0, 1])
    """

    def __init__(self) -> None:
        """Initialize reward registry."""
        self._functions: dict[str, RewardFunction] = {}
        self._agent_strategies: dict[str, str] = {}
        self._default_strategy: str | None = None

        logger.info("reward_registry_initialized")

    def register(
        self,
        name: str,
        func: RewardFunction,
        validate: bool = True,
    ) -> None:
        """
        Register custom reward function with optional validation.

        Args:
            name: Unique identifier for reward function
            func: Callable that takes Trajectory and returns float in [0, 1]
            validate: Whether to validate function signature and test output

        Raises:
            RewardValidationError: If validation fails
            ValueError: If name already registered
        """
        if name in self._functions:
            raise ValueError(f"Reward function '{name}' already registered")

        # Validate function signature
        if not callable(func):
            raise RewardValidationError(f"Reward function '{name}' must be callable")

        # Validate function behavior if requested
        if validate:
            self._validate_reward_function(name, func)

        self._functions[name] = func

        logger.info("reward_function_registered", name=name, validated=validate)

    def unregister(self, name: str) -> None:
        """
        Remove reward function from registry.

        Args:
            name: Function identifier to remove

        Raises:
            KeyError: If function not found
        """
        if name not in self._functions:
            raise KeyError(f"Reward function '{name}' not found")

        del self._functions[name]

        # Remove from agent strategies if used
        strategies_to_remove = [
            agent_type
            for agent_type, strategy in self._agent_strategies.items()
            if strategy == name
        ]
        for agent_type in strategies_to_remove:
            del self._agent_strategies[agent_type]

        # Clear default if it was this function
        if self._default_strategy == name:
            self._default_strategy = None

        logger.info("reward_function_unregistered", name=name)

    def get(self, name: str) -> RewardFunction:
        """
        Retrieve reward function by name.

        Args:
            name: Function identifier

        Returns:
            Reward function

        Raises:
            KeyError: If function not found
        """
        if name not in self._functions:
            raise KeyError(f"Reward function '{name}' not found")

        return self._functions[name]

    def list_functions(self) -> list[str]:
        """
        Get list of registered function names.

        Returns:
            List of function identifiers
        """
        return list(self._functions.keys())

    def set_agent_strategy(self, agent_type: str, function_name: str) -> None:
        """
        Configure reward strategy for specific agent type.

        Args:
            agent_type: Agent type identifier (e.g., "code_generator", "qa_agent")
            function_name: Reward function name to use

        Raises:
            KeyError: If function not registered
        """
        if function_name not in self._functions:
            raise KeyError(f"Reward function '{function_name}' not registered")

        self._agent_strategies[agent_type] = function_name

        logger.info(
            "agent_strategy_configured",
            agent_type=agent_type,
            function=function_name,
        )

    def get_agent_strategy(self, agent_type: str) -> str | None:
        """
        Get reward function name for agent type.

        Args:
            agent_type: Agent type identifier

        Returns:
            Function name or None if not configured
        """
        return self._agent_strategies.get(agent_type)

    def set_default_strategy(self, function_name: str) -> None:
        """
        Set default reward function for agents without specific strategy.

        Args:
            function_name: Reward function name

        Raises:
            KeyError: If function not registered
        """
        if function_name not in self._functions:
            raise KeyError(f"Reward function '{function_name}' not registered")

        self._default_strategy = function_name

        logger.info("default_strategy_set", function=function_name)

    def get_default_strategy(self) -> str | None:
        """
        Get default reward function name.

        Returns:
            Function name or None if not set
        """
        return self._default_strategy

    def resolve_strategy(self, agent_type: str | None = None) -> str | None:
        """
        Resolve reward function name for agent type with fallback to default.

        Args:
            agent_type: Agent type identifier (optional)

        Returns:
            Function name or None if no strategy configured
        """
        if agent_type and agent_type in self._agent_strategies:
            return self._agent_strategies[agent_type]

        return self._default_strategy

    def compute_reward(
        self,
        trajectory: Trajectory,
        function_name: str | None = None,
        agent_type: str | None = None,
    ) -> float:
        """
        Compute reward using specified or resolved function.

        Args:
            trajectory: Trajectory to evaluate
            function_name: Explicit function name (overrides agent_type)
            agent_type: Agent type for strategy resolution

        Returns:
            Reward value in [0, 1]

        Raises:
            KeyError: If function not found
            RewardValidationError: If reward out of valid range
        """
        # Resolve function name
        if function_name is None:
            function_name = self.resolve_strategy(agent_type)

        if function_name is None:
            raise ValueError("No reward function specified or resolved")

        # Get function
        func = self.get(function_name)

        # Compute reward
        reward = func(trajectory)

        # Validate output
        if not isinstance(reward, (int, float)):
            raise RewardValidationError(
                f"Reward function '{function_name}' returned non-numeric value: {type(reward)}"
            )

        if not (0.0 <= reward <= 1.0):
            raise RewardValidationError(
                f"Reward function '{function_name}' returned value outside [0, 1]: {reward}"
            )

        logger.debug(
            "reward_computed_from_registry",
            function=function_name,
            agent_type=agent_type,
            reward=reward,
            trajectory_id=str(trajectory.trajectory_id)
            if trajectory.trajectory_id
            else "none",
        )

        return reward

    def _validate_reward_function(self, name: str, func: RewardFunction) -> None:
        """
        Validate reward function behavior with test trajectory.

        Args:
            name: Function identifier
            func: Function to validate

        Raises:
            RewardValidationError: If validation fails
        """
        # Import uuid4 here to avoid circular imports
        from uuid import uuid4

        # Create minimal test trajectory
        test_trajectory = Trajectory(
            job_id=uuid4(),
            agent_id="test_agent",
            query="test query",
            steps=[],
            success=True,
        )

        try:
            reward = func(test_trajectory)
        except Exception as e:
            raise RewardValidationError(
                f"Reward function '{name}' raised exception during validation: {e}"
            ) from e

        # Validate return type
        if not isinstance(reward, (int, float)):
            raise RewardValidationError(
                f"Reward function '{name}' must return numeric value, got {type(reward)}"
            )

        # Validate range
        if not (0.0 <= reward <= 1.0):
            raise RewardValidationError(
                f"Reward function '{name}' must return value in [0, 1], got {reward}"
            )

        logger.debug("reward_function_validated", name=name, test_reward=reward)


# Example domain-specific reward functions


def code_quality_reward(trajectory: Trajectory) -> float:
    """
    Reward function for code generation agents based on code quality indicators.

    Evaluates:
    - Successful execution (0.5 base)
    - Number of syntax errors (penalty)
    - Test coverage (bonus)
    - Code complexity (penalty for high complexity)

    Args:
        trajectory: Code generation trajectory

    Returns:
        Reward in [0, 1]
    """
    reward = 0.0

    # Base reward for success
    if trajectory.success:
        reward += 0.5

    # Analyze steps for code quality indicators
    syntax_errors = 0
    test_coverage = 0.0
    complexity_penalty = 0.0

    for step in trajectory.steps:
        result = step.result

        # Check for syntax errors
        if isinstance(result, dict):
            if result.get("error_type") == "syntax":
                syntax_errors += 1

            # Extract test coverage if available
            if "test_coverage" in result:
                test_coverage = max(test_coverage, float(result["test_coverage"]))

            # Extract complexity metrics
            if "complexity" in result:
                complexity = float(result["complexity"])
                if complexity > 10:  # High complexity threshold
                    complexity_penalty += 0.1

    # Apply penalties and bonuses
    reward -= syntax_errors * 0.1  # -0.1 per syntax error
    reward += test_coverage * 0.3  # Up to +0.3 for 100% coverage
    reward -= min(complexity_penalty, 0.2)  # Max -0.2 complexity penalty

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reward))


def response_accuracy_reward(trajectory: Trajectory) -> float:
    """
    Reward function for QA/response agents based on answer accuracy.

    Evaluates:
    - Task success (0.6 base)
    - Response confidence (bonus)
    - Number of clarification steps (slight penalty)
    - Source citation (bonus)

    Args:
        trajectory: QA agent trajectory

    Returns:
        Reward in [0, 1]
    """
    reward = 0.0

    # Base reward for success
    if trajectory.success:
        reward += 0.6

    # Analyze response quality
    has_sources = False
    confidence_sum = 0.0
    confidence_count = 0
    clarification_steps = 0

    for step in trajectory.steps:
        result = step.result

        if isinstance(result, dict):
            # Check for source citations
            if result.get("sources") or result.get("citations"):
                has_sources = True

            # Aggregate confidence scores
            if "confidence" in result:
                confidence_sum += float(result["confidence"])
                confidence_count += 1

            # Count clarification requests
            if result.get("needs_clarification"):
                clarification_steps += 1

    # Apply bonuses and penalties
    if has_sources:
        reward += 0.15  # Bonus for citing sources

    if confidence_count > 0:
        avg_confidence = confidence_sum / confidence_count
        reward += avg_confidence * 0.2  # Up to +0.2 for high confidence

    # Small penalty for excessive clarifications
    reward -= min(clarification_steps * 0.05, 0.15)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reward))


def task_efficiency_reward(trajectory: Trajectory) -> float:
    """
    Reward function based on task completion efficiency.

    Evaluates:
    - Task success (0.5 base)
    - Number of steps (penalty for more steps)
    - Execution time (penalty for longer time)
    - Resource usage (penalty for high resource consumption)

    Args:
        trajectory: Task execution trajectory

    Returns:
        Reward in [0, 1]
    """
    reward = 0.0

    # Base reward for success
    if trajectory.success:
        reward += 0.5

    # Step efficiency (fewer steps = higher reward)
    num_steps = len(trajectory.steps)
    if num_steps <= 3:
        reward += 0.3  # Very efficient
    elif num_steps <= 5:
        reward += 0.2  # Moderately efficient
    elif num_steps <= 10:
        reward += 0.1  # Acceptable efficiency
    # No bonus for > 10 steps

    # Time efficiency (check if duration metadata available)
    if hasattr(trajectory, "duration_seconds"):
        duration = trajectory.duration_seconds
        if duration < 10:
            reward += 0.1  # Fast execution
        elif duration > 60:
            reward -= 0.1  # Slow execution penalty

    # Resource usage (if available in trajectory)
    total_resource_penalty = 0.0
    for step in trajectory.steps:
        if isinstance(step.result, dict):
            if "resource_usage" in step.result:
                usage = float(step.result["resource_usage"])
                if usage > 0.8:  # High resource usage
                    total_resource_penalty += 0.05

    reward -= min(total_resource_penalty, 0.2)  # Max -0.2 penalty

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reward))


# Global registry instance
_global_registry: RewardRegistry | None = None


def get_global_registry() -> RewardRegistry:
    """
    Get or create global reward registry instance.

    Returns:
        Global RewardRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = RewardRegistry()

        # Register example functions
        _global_registry.register("code_quality", code_quality_reward)
        _global_registry.register("response_accuracy", response_accuracy_reward)
        _global_registry.register("task_efficiency", task_efficiency_reward)

        logger.info("global_registry_initialized", functions=3)

    return _global_registry


def reset_global_registry() -> None:
    """Reset global registry (primarily for testing)."""
    global _global_registry
    _global_registry = None
    logger.info("global_registry_reset")
