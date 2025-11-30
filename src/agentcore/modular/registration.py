"""
Module Registration for A2A Protocol

Registers all four PEVG modules (Planner, Executor, Verifier, Generator) as
discoverable A2A agents with AgentCard specifications and capability advertisements.

This module provides:
- AgentCard creation for each module with capabilities
- Module registration with AgentManager on startup
- Discovery via /.well-known/agents/{module-id}
- Health check endpoints for each module
- Capability advertisement with version, cost, and latency estimates
"""

from __future__ import annotations

from typing import Any

import structlog

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCapability,
    AgentCard,
    AgentEndpoint,
    AgentMetadata,
    AgentRegistrationRequest,
    AgentStatus,
    AuthenticationType,
    EndpointType,
)
from agentcore.a2a_protocol.services.agent_manager import agent_manager

logger = structlog.get_logger()

# Module configuration constants
BASE_URL = "http://localhost:8001"
MODULE_VERSION = "1.0.0"
A2A_SCHEMA_VERSION = "0.2"


def create_planner_agent_card() -> AgentCard:
    """
    Create AgentCard for Planner Module.

    Returns:
        AgentCard with planner capabilities and configuration
    """
    return AgentCard(
        schema_version=A2A_SCHEMA_VERSION,
        agent_id="agentcore.modular.planner",
        agent_name="Modular Agent - Planner",
        agent_version=MODULE_VERSION,
        status=AgentStatus.ACTIVE,
        description="Analyzes requests and creates structured execution plans with task decomposition and workflow design",
        endpoints=[
            AgentEndpoint(
                url=f"{BASE_URL}/api/v1/jsonrpc",
                type=EndpointType.HTTPS,
                protocols=["jsonrpc-2.0"],
                description="JSON-RPC endpoint for planner operations",
                health_check_path="/.well-known/health",
            )
        ],
        capabilities=[
            AgentCapability(
                name="planning",
                version=MODULE_VERSION,
                description="Generate execution plans from user queries with step-by-step decomposition",
                cost_per_request=0.001,
                avg_latency_ms=150.0,
                quality_score=0.95,
                parameters={
                    "max_steps": 20,
                    "enable_parallel": False,
                    "supported_task_types": [
                        "information_retrieval",
                        "computation",
                        "multi_step",
                        "simple",
                    ],
                },
            ),
            AgentCapability(
                name="task-decomposition",
                version=MODULE_VERSION,
                description="Decompose complex tasks into manageable sequential steps with dependencies",
                cost_per_request=0.001,
                avg_latency_ms=120.0,
                quality_score=0.92,
                parameters={
                    "decomposition_strategies": [
                        "information_retrieval",
                        "computation",
                        "multi_step",
                    ],
                },
            ),
            AgentCapability(
                name="workflow-design",
                version=MODULE_VERSION,
                description="Design workflow patterns with step dependencies and success criteria",
                cost_per_request=0.002,
                avg_latency_ms=200.0,
                quality_score=0.90,
                parameters={
                    "supports_dependencies": True,
                    "supports_parallel_execution": False,
                    "max_refinement_iterations": 5,
                },
            ),
            AgentCapability(
                name="plan-refinement",
                version=MODULE_VERSION,
                description="Refine execution plans based on verification feedback and failures",
                cost_per_request=0.0015,
                avg_latency_ms=180.0,
                quality_score=0.88,
                parameters={
                    "refinement_strategies": [
                        "add_error_handling",
                        "adjust_parameters",
                        "add_validation",
                        "reorder_steps",
                    ],
                },
            ),
        ],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE,
            required=False,
            config={},
        ),
        metadata=AgentMetadata(
            tags=["planning", "task-decomposition", "workflow", "pevg"],
            category="modular-agent",
            documentation_url="https://github.com/YourOrg/AgentCore/blob/main/docs/modular-agent-core.md",
        ),
    )


def create_executor_agent_card() -> AgentCard:
    """
    Create AgentCard for Executor Module.

    Returns:
        AgentCard with executor capabilities and configuration
    """
    return AgentCard(
        schema_version=A2A_SCHEMA_VERSION,
        agent_id="agentcore.modular.executor",
        agent_name="Modular Agent - Executor",
        agent_version=MODULE_VERSION,
        status=AgentStatus.ACTIVE,
        description="Executes plan steps by invoking tools and resources with retry logic and circuit breaker protection",
        endpoints=[
            AgentEndpoint(
                url=f"{BASE_URL}/api/v1/jsonrpc",
                type=EndpointType.HTTPS,
                protocols=["jsonrpc-2.0"],
                description="JSON-RPC endpoint for executor operations",
                health_check_path="/.well-known/health",
            )
        ],
        capabilities=[
            AgentCapability(
                name="tool-execution",
                version=MODULE_VERSION,
                description="Execute plan steps using tool integration framework with parameter formatting and validation",
                cost_per_request=0.005,
                avg_latency_ms=500.0,
                quality_score=0.93,
                parameters={
                    "max_parallel_steps": 5,
                    "default_timeout_seconds": 30,
                    "supports_tool_registry": True,
                },
            ),
            AgentCapability(
                name="step-execution",
                version=MODULE_VERSION,
                description="Execute individual plan steps with dependency resolution and parameter substitution",
                cost_per_request=0.003,
                avg_latency_ms=400.0,
                quality_score=0.91,
                parameters={
                    "supports_dependencies": True,
                    "supports_parallel": True,
                    "parameter_substitution": True,
                },
            ),
            AgentCapability(
                name="retry-logic",
                version=MODULE_VERSION,
                description="Configurable retry strategies with exponential backoff, jitter, and error categorization",
                cost_per_request=0.002,
                avg_latency_ms=300.0,
                quality_score=0.89,
                parameters={
                    "max_retries": 3,
                    "exponential_backoff": True,
                    "backoff_base_seconds": 1.0,
                    "error_categories": [
                        "transient",
                        "permanent",
                        "timeout",
                        "validation",
                        "tool_not_found",
                        "circuit_open",
                    ],
                },
            ),
            AgentCapability(
                name="circuit-breaker",
                version=MODULE_VERSION,
                description="Circuit breaker pattern for fault tolerance with per-tool state management",
                cost_per_request=0.001,
                avg_latency_ms=100.0,
                quality_score=0.90,
                parameters={
                    "enabled": True,
                    "failure_threshold": 5,
                    "timeout_seconds": 60,
                    "half_open_max_calls": 1,
                },
            ),
        ],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE,
            required=False,
            config={},
        ),
        metadata=AgentMetadata(
            tags=["execution", "tools", "retry", "circuit-breaker", "pevg"],
            category="modular-agent",
            documentation_url="https://github.com/YourOrg/AgentCore/blob/main/docs/modular-agent-core.md",
        ),
    )


def create_verifier_agent_card() -> AgentCard:
    """
    Create AgentCard for Verifier Module.

    Returns:
        AgentCard with verifier capabilities and configuration
    """
    return AgentCard(
        schema_version=A2A_SCHEMA_VERSION,
        agent_id="agentcore.modular.verifier",
        agent_name="Modular Agent - Verifier",
        agent_version=MODULE_VERSION,
        status=AgentStatus.ACTIVE,
        description="Validates execution results against success criteria with rule-based and LLM-based verification",
        endpoints=[
            AgentEndpoint(
                url=f"{BASE_URL}/api/v1/jsonrpc",
                type=EndpointType.HTTPS,
                protocols=["jsonrpc-2.0"],
                description="JSON-RPC endpoint for verifier operations",
                health_check_path="/.well-known/health",
            )
        ],
        capabilities=[
            AgentCapability(
                name="result-validation",
                version=MODULE_VERSION,
                description="Validate execution results using rule-based and LLM-based verification strategies",
                cost_per_request=0.002,
                avg_latency_ms=250.0,
                quality_score=0.94,
                parameters={
                    "validation_types": [
                        "schema",
                        "format",
                        "completeness",
                        "consistency",
                    ],
                    "confidence_threshold": 0.7,
                },
            ),
            AgentCapability(
                name="quality-assurance",
                version=MODULE_VERSION,
                description="Ensure result quality through confidence scoring and threshold-based approval",
                cost_per_request=0.0015,
                avg_latency_ms=200.0,
                quality_score=0.92,
                parameters={
                    "confidence_threshold": 0.7,
                    "quality_metrics": ["accuracy", "completeness", "consistency"],
                },
            ),
            AgentCapability(
                name="feedback-generation",
                version=MODULE_VERSION,
                description="Generate structured feedback for plan refinement based on validation results",
                cost_per_request=0.001,
                avg_latency_ms=150.0,
                quality_score=0.90,
                parameters={
                    "feedback_types": ["failures", "performance", "completeness"],
                    "include_recommendations": True,
                },
            ),
            AgentCapability(
                name="consistency-checking",
                version=MODULE_VERSION,
                description="Check logical consistency between multiple execution results",
                cost_per_request=0.002,
                avg_latency_ms=220.0,
                quality_score=0.88,
                parameters={
                    "consistency_rules": [
                        "values_match",
                        "types_match",
                        "ranges_valid",
                        "no_contradictions",
                    ],
                },
            ),
        ],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE,
            required=False,
            config={},
        ),
        metadata=AgentMetadata(
            tags=["verification", "validation", "quality-assurance", "pevg"],
            category="modular-agent",
            documentation_url="https://github.com/YourOrg/AgentCore/blob/main/docs/modular-agent-core.md",
        ),
    )


def create_generator_agent_card() -> AgentCard:
    """
    Create AgentCard for Generator Module.

    Returns:
        AgentCard with generator capabilities and configuration
    """
    return AgentCard(
        schema_version=A2A_SCHEMA_VERSION,
        agent_id="agentcore.modular.generator",
        agent_name="Modular Agent - Generator",
        agent_version=MODULE_VERSION,
        status=AgentStatus.ACTIVE,
        description="Synthesizes final responses from verified results with multi-format output support",
        endpoints=[
            AgentEndpoint(
                url=f"{BASE_URL}/api/v1/jsonrpc",
                type=EndpointType.HTTPS,
                protocols=["jsonrpc-2.0"],
                description="JSON-RPC endpoint for generator operations",
                health_check_path="/.well-known/health",
            )
        ],
        capabilities=[
            AgentCapability(
                name="response-synthesis",
                version=MODULE_VERSION,
                description="Synthesize coherent responses from verified execution results",
                cost_per_request=0.003,
                avg_latency_ms=300.0,
                quality_score=0.93,
                parameters={
                    "supported_formats": ["text", "json", "markdown", "html"],
                    "max_length_default": None,
                },
            ),
            AgentCapability(
                name="output-formatting",
                version=MODULE_VERSION,
                description="Format output in multiple formats with template support and schema validation",
                cost_per_request=0.002,
                avg_latency_ms=200.0,
                quality_score=0.91,
                parameters={
                    "formats": ["text", "json", "markdown", "html"],
                    "supports_templates": True,
                    "supports_json_schema": True,
                },
            ),
            AgentCapability(
                name="explanation-generation",
                version=MODULE_VERSION,
                description="Generate reasoning traces and explanations from execution results",
                cost_per_request=0.0025,
                avg_latency_ms=250.0,
                quality_score=0.89,
                parameters={
                    "includes_execution_trace": True,
                    "includes_timing": True,
                    "includes_sources": True,
                },
            ),
            AgentCapability(
                name="source-tracking",
                version=MODULE_VERSION,
                description="Track and include evidence sources from execution metadata",
                cost_per_request=0.001,
                avg_latency_ms=100.0,
                quality_score=0.92,
                parameters={
                    "source_types": ["step_ids", "metadata", "tool_outputs"],
                },
            ),
        ],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE,
            required=False,
            config={},
        ),
        metadata=AgentMetadata(
            tags=["generation", "synthesis", "formatting", "pevg"],
            category="modular-agent",
            documentation_url="https://github.com/YourOrg/AgentCore/blob/main/docs/modular-agent-core.md",
        ),
    )


async def register_module_agents() -> dict[str, str]:
    """
    Register all four PEVG modules as A2A agents.

    Registers Planner, Executor, Verifier, and Generator modules with the
    AgentManager, making them discoverable via the A2A protocol.

    Returns:
        Dictionary mapping module names to their discovery URLs

    Raises:
        RuntimeError: If registration fails for any module
    """
    logger.info("Registering PEVG modules as A2A agents")

    modules = {
        "planner": create_planner_agent_card(),
        "executor": create_executor_agent_card(),
        "verifier": create_verifier_agent_card(),
        "generator": create_generator_agent_card(),
    }

    discovery_urls: dict[str, str] = {}
    registration_errors: list[str] = []

    for module_name, agent_card in modules.items():
        try:
            # Create registration request
            registration_request = AgentRegistrationRequest(
                agent_card=agent_card,
                override_existing=True,  # Allow re-registration on restart
            )

            # Register with AgentManager
            response = await agent_manager.register_agent(registration_request)

            discovery_urls[module_name] = response.discovery_url

            logger.info(
                "Module registered as A2A agent",
                module=module_name,
                agent_id=agent_card.agent_id,
                discovery_url=response.discovery_url,
                capabilities_count=len(agent_card.capabilities),
            )

        except Exception as e:
            error_msg = f"Failed to register {module_name} module: {str(e)}"
            logger.error(
                "Module registration failed",
                module=module_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            registration_errors.append(error_msg)

    # Check if any registrations failed
    if registration_errors:
        error_summary = "; ".join(registration_errors)
        raise RuntimeError(f"Module registration failed: {error_summary}")

    logger.info(
        "All PEVG modules registered successfully",
        registered_count=len(discovery_urls),
        discovery_urls=discovery_urls,
    )

    return discovery_urls


async def unregister_module_agents() -> None:
    """
    Unregister all PEVG modules from AgentManager.

    Called during application shutdown to clean up module registrations.
    """
    logger.info("Unregistering PEVG modules")

    module_ids = [
        "agentcore.modular.planner",
        "agentcore.modular.executor",
        "agentcore.modular.verifier",
        "agentcore.modular.generator",
    ]

    for agent_id in module_ids:
        try:
            success = await agent_manager.unregister_agent(agent_id)
            if success:
                logger.info("Module unregistered", agent_id=agent_id)
            else:
                logger.warning("Module not found during unregistration", agent_id=agent_id)
        except Exception as e:
            logger.error(
                "Module unregistration failed",
                agent_id=agent_id,
                error=str(e),
                error_type=type(e).__name__,
            )

    logger.info("PEVG modules unregistration complete")


async def get_module_health_status() -> dict[str, Any]:
    """
    Get health status for all registered PEVG modules.

    Returns:
        Dictionary with health status for each module
    """
    module_ids = {
        "planner": "agentcore.modular.planner",
        "executor": "agentcore.modular.executor",
        "verifier": "agentcore.modular.verifier",
        "generator": "agentcore.modular.generator",
    }

    health_status: dict[str, Any] = {
        "overall_status": "healthy",
        "modules": {},
        "registered_count": 0,
        "total_count": len(module_ids),
    }

    for module_name, agent_id in module_ids.items():
        try:
            agent = await agent_manager.get_agent(agent_id)
            if agent:
                health_status["modules"][module_name] = {
                    "status": agent.status.value,
                    "agent_id": agent_id,
                    "is_active": agent.is_active(),
                    "capabilities_count": len(agent.capabilities),
                    "last_seen": agent.last_seen.isoformat() if agent.last_seen else None,
                }
                health_status["registered_count"] += 1
            else:
                health_status["modules"][module_name] = {
                    "status": "not_registered",
                    "agent_id": agent_id,
                }
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["modules"][module_name] = {
                "status": "error",
                "agent_id": agent_id,
                "error": str(e),
            }
            health_status["overall_status"] = "unhealthy"

    return health_status
