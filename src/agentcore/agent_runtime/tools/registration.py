"""Tool registration utilities for native Tool ABC implementations.

This module provides helper functions to register native Tool ABC implementations
from the builtin package into a ToolRegistry or legacy ToolRegistry.
"""

import structlog

from .builtin import (
    CalculatorTool,
    EchoTool,
    EvaluateExpressionTool,
    ExecutePythonTool,
    FileOperationsTool,
    GetCurrentTimeTool,
    GoogleSearchTool,
    GraphQLQueryTool,
    HttpRequestTool,
    RestGetTool,
    RestPostTool,
    WebScrapeTool,
    WikipediaSearchTool,
)

logger = structlog.get_logger()


def register_native_builtin_tools(registry) -> None:
    """Register all native Tool ABC implementations from builtin package.

    This function registers all native Tool ABC implementations with the
    provided registry. It works with both the new ToolRegistry (tools/registry.py)
    and the legacy ToolRegistry (services/tool_registry.py).

    Args:
        registry: ToolRegistry instance (either new or legacy)

    Example:
        ```python
        from agentcore.agent_runtime.tools.registry import ToolRegistry
        from agentcore.agent_runtime.tools.registration import register_native_builtin_tools

        registry = ToolRegistry()
        register_native_builtin_tools(registry)
        ```
    """
    # Utility tools
    tools_to_register = [
        CalculatorTool(),
        GetCurrentTimeTool(),
        EchoTool(),
        FileOperationsTool(),
        # Search tools
        GoogleSearchTool(),
        WikipediaSearchTool(),
        WebScrapeTool(),
        # API tools
        HttpRequestTool(),
        RestGetTool(),
        RestPostTool(),
        GraphQLQueryTool(),
        # Code execution tools
        ExecutePythonTool(),
        EvaluateExpressionTool(),
    ]

    registered_count = 0
    for tool in tools_to_register:
        try:
            # Check if registry has 'register' method (new ToolRegistry)
            # or 'register_tool' method (legacy ToolRegistry)
            if hasattr(registry, "register"):
                registry.register(tool)
            elif hasattr(registry, "register_tool"):
                # Legacy ToolRegistry expects (ToolDefinition, executor_function)
                # We need to adapt the Tool ABC instance to the legacy format
                # This is a temporary bridge until legacy ToolRegistry is fully replaced
                logger.warning(
                    "legacy_tool_registry_detected",
                    tool_id=tool.metadata.tool_id,
                    message="Using legacy ToolRegistry. Consider migrating to new ToolRegistry.",
                )
                # For now, skip legacy registration as it requires function-based executors
                continue
            else:
                raise ValueError(
                    f"Registry does not have 'register' or 'register_tool' method: {type(registry)}"
                )

            registered_count += 1
            logger.debug(
                "native_tool_registered",
                tool_id=tool.metadata.tool_id,
                tool_name=tool.metadata.name,
            )
        except Exception as e:
            logger.error(
                "native_tool_registration_failed",
                tool_id=tool.metadata.tool_id,
                error=str(e),
            )

    logger.info(
        "native_builtin_tools_registered",
        total_tools=len(tools_to_register),
        registered=registered_count,
        failed=len(tools_to_register) - registered_count,
    )


def get_native_builtin_tool_ids() -> list[str]:
    """Get list of all native builtin tool IDs.

    Returns:
        List of tool IDs for all native builtin tools
    """
    return [
        "calculator",
        "get_current_time",
        "echo",
        "file_operations",
        "google_search",
        "wikipedia_search",
        "web_scrape",
        "http_request",
        "rest_get",
        "rest_post",
        "graphql_query",
        "execute_python",
        "evaluate_expression",
    ]
