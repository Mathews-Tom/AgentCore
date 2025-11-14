"""Tool system startup and configuration.

This module handles tool system initialization during application startup,
including conditional registration based on environment variables and
configuration validation.

TOOL-014: Tool Registration on Startup
"""

import os
from typing import Any

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
from .registry import ToolRegistry

logger = structlog.get_logger()


class ToolStartupConfig:
    """Configuration for tool system startup.

    Reads environment variables to determine which tools should be registered
    and with what configuration.
    """

    def __init__(self):
        """Initialize startup configuration from environment variables."""
        # Search tool configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        # Code execution configuration
        self.enable_code_execution = os.getenv("ENABLE_CODE_EXECUTION", "false").lower() == "true"
        self.code_execution_timeout = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))

        # API tool configuration
        self.enable_http_tools = os.getenv("ENABLE_HTTP_TOOLS", "true").lower() == "true"

        # File operations configuration
        self.enable_file_operations = os.getenv("ENABLE_FILE_OPERATIONS", "true").lower() == "true"
        self.file_allowed_directories = os.getenv("FILE_ALLOWED_DIRECTORIES", "").split(",") if os.getenv("FILE_ALLOWED_DIRECTORIES") else None

        # General configuration
        self.enable_all_tools = os.getenv("ENABLE_ALL_TOOLS", "true").lower() == "true"

    def should_register_google_search(self) -> bool:
        """Check if Google Search tool should be registered."""
        return self.enable_all_tools and self.google_api_key is not None and self.google_search_engine_id is not None

    def should_register_wikipedia_search(self) -> bool:
        """Check if Wikipedia Search tool should be registered."""
        return self.enable_all_tools

    def should_register_code_execution(self) -> bool:
        """Check if code execution tools should be registered."""
        return self.enable_all_tools and self.enable_code_execution

    def should_register_http_tools(self) -> bool:
        """Check if HTTP/API tools should be registered."""
        return self.enable_all_tools and self.enable_http_tools

    def should_register_file_operations(self) -> bool:
        """Check if file operations tool should be registered."""
        return self.enable_all_tools and self.enable_file_operations

    def should_register_utility_tools(self) -> bool:
        """Check if utility tools should be registered."""
        return self.enable_all_tools


def register_builtin_tools(registry: ToolRegistry, config: ToolStartupConfig | None = None) -> dict[str, Any]:
    """Register all built-in tools based on configuration.

    This is the main entry point for tool registration during application startup.
    It conditionally registers tools based on environment variables and configuration.

    Args:
        registry: ToolRegistry instance to register tools with
        config: Optional startup configuration. If None, creates from environment.

    Returns:
        Dictionary with registration statistics

    Example:
        ```python
        registry = ToolRegistry()
        stats = register_builtin_tools(registry)
        logger.info("Tools registered", **stats)
        ```
    """
    if config is None:
        config = ToolStartupConfig()

    registered_tools: list[str] = []
    skipped_tools: list[dict[str, str]] = []
    failed_tools: list[dict[str, str]] = []

    # Utility tools (always available, no dependencies)
    if config.should_register_utility_tools():
        utility_tools = [
            ("calculator", CalculatorTool()),
            ("get_current_time", GetCurrentTimeTool()),
            ("echo", EchoTool()),
        ]

        for tool_id, tool in utility_tools:
            try:
                registry.register(tool)
                registered_tools.append(tool_id)
                logger.debug("tool_registered", tool_id=tool_id, category="utility")
            except Exception as e:
                failed_tools.append({"tool_id": tool_id, "error": str(e)})
                logger.error("tool_registration_failed", tool_id=tool_id, error=str(e))
    else:
        skipped_tools.append({"tool_id": "utility_tools", "reason": "ENABLE_ALL_TOOLS=false"})

    # File operations tool
    if config.should_register_file_operations():
        try:
            if config.file_allowed_directories:
                tool = FileOperationsTool(allowed_directories=config.file_allowed_directories)
                logger.debug("file_operations_custom_dirs", directories=config.file_allowed_directories)
            else:
                tool = FileOperationsTool()
            registry.register(tool)
            registered_tools.append("file_operations")
            logger.debug("tool_registered", tool_id="file_operations", category="utility")
        except Exception as e:
            failed_tools.append({"tool_id": "file_operations", "error": str(e)})
            logger.error("tool_registration_failed", tool_id="file_operations", error=str(e))
    else:
        skipped_tools.append({"tool_id": "file_operations", "reason": "ENABLE_FILE_OPERATIONS=false"})

    # Search tools
    if config.should_register_google_search():
        try:
            tool = GoogleSearchTool()
            registry.register(tool)
            registered_tools.append("google_search")
            logger.debug("tool_registered", tool_id="google_search", category="search")
        except Exception as e:
            failed_tools.append({"tool_id": "google_search", "error": str(e)})
            logger.error("tool_registration_failed", tool_id="google_search", error=str(e))
    else:
        skipped_tools.append(
            {
                "tool_id": "google_search",
                "reason": "Missing GOOGLE_API_KEY or GOOGLE_SEARCH_ENGINE_ID",
            }
        )

    if config.should_register_wikipedia_search():
        try:
            tool = WikipediaSearchTool()
            registry.register(tool)
            registered_tools.append("wikipedia_search")
            logger.debug("tool_registered", tool_id="wikipedia_search", category="search")
        except Exception as e:
            failed_tools.append({"tool_id": "wikipedia_search", "error": str(e)})
            logger.error("tool_registration_failed", tool_id="wikipedia_search", error=str(e))

        try:
            tool = WebScrapeTool()
            registry.register(tool)
            registered_tools.append("web_scrape")
            logger.debug("tool_registered", tool_id="web_scrape", category="search")
        except Exception as e:
            failed_tools.append({"tool_id": "web_scrape", "error": str(e)})
            logger.error("tool_registration_failed", tool_id="web_scrape", error=str(e))

    # HTTP/API tools
    if config.should_register_http_tools():
        http_tools = [
            ("http_request", HttpRequestTool()),
            ("rest_get", RestGetTool()),
            ("rest_post", RestPostTool()),
            ("graphql_query", GraphQLQueryTool()),
        ]

        for tool_id, tool in http_tools:
            try:
                registry.register(tool)
                registered_tools.append(tool_id)
                logger.debug("tool_registered", tool_id=tool_id, category="api")
            except Exception as e:
                failed_tools.append({"tool_id": tool_id, "error": str(e)})
                logger.error("tool_registration_failed", tool_id=tool_id, error=str(e))
    else:
        skipped_tools.append({"tool_id": "http_tools", "reason": "ENABLE_HTTP_TOOLS=false"})

    # Code execution tools
    if config.should_register_code_execution():
        code_tools = [
            ("execute_python", ExecutePythonTool()),
            ("evaluate_expression", EvaluateExpressionTool()),
        ]

        for tool_id, tool in code_tools:
            try:
                registry.register(tool)
                registered_tools.append(tool_id)
                logger.debug("tool_registered", tool_id=tool_id, category="code_execution")
            except Exception as e:
                failed_tools.append({"tool_id": tool_id, "error": str(e)})
                logger.error("tool_registration_failed", tool_id=tool_id, error=str(e))
    else:
        skipped_tools.append({"tool_id": "code_execution_tools", "reason": "ENABLE_CODE_EXECUTION=false"})

    # Log summary
    stats = {
        "total_registered": len(registered_tools),
        "total_skipped": len(skipped_tools),
        "total_failed": len(failed_tools),
        "registered_tools": registered_tools,
        "skipped_tools": skipped_tools,
        "failed_tools": failed_tools,
    }

    logger.info(
        "builtin_tools_registration_complete",
        registered=len(registered_tools),
        skipped=len(skipped_tools),
        failed=len(failed_tools),
        tools=registered_tools,
    )

    if skipped_tools:
        logger.info(
            "tools_skipped_due_to_config",
            count=len(skipped_tools),
            details=skipped_tools,
        )

    if failed_tools:
        logger.warning(
            "tool_registration_failures",
            count=len(failed_tools),
            details=failed_tools,
        )

    return stats


async def initialize_tool_system(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Initialize the tool system during application startup.

    This function should be called during FastAPI lifespan startup.
    It creates the registry, registers all configured tools, and returns
    the initialized registry.

    Args:
        registry: Optional existing registry. If None, creates a new one.

    Returns:
        Initialized ToolRegistry with registered tools

    Example:
        ```python
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            tool_registry = await initialize_tool_system()
            app.state.tool_registry = tool_registry
            yield
            # Shutdown
            pass
        ```
    """
    logger.info("initializing_tool_system")

    if registry is None:
        registry = ToolRegistry()

    # Register built-in tools with configuration
    stats = register_builtin_tools(registry)

    logger.info(
        "tool_system_initialized",
        total_tools=stats["total_registered"],
        registry_size=len(registry.list_all()),
    )

    return registry
