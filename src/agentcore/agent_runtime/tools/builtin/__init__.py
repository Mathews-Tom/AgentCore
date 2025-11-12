"""Built-in native Tool ABC implementations.

This package contains native Tool ABC implementations for all built-in tools.
These replace the legacy function-based tools during Stage 3 migration.
"""

from .api_tools import GraphQLQueryTool, HttpRequestTool, RestGetTool, RestPostTool
from .code_execution_tools import EvaluateExpressionTool, ExecutePythonTool
from .search_tools import GoogleSearchTool, WebScrapeTool, WikipediaSearchTool
from .utility_tools import CalculatorTool, EchoTool, GetCurrentTimeTool

__all__ = [
    # Utility tools
    "CalculatorTool",
    "GetCurrentTimeTool",
    "EchoTool",
    # Search tools
    "GoogleSearchTool",
    "WikipediaSearchTool",
    "WebScrapeTool",
    # API tools
    "HttpRequestTool",
    "RestGetTool",
    "RestPostTool",
    "GraphQLQueryTool",
    # Code execution tools
    "ExecutePythonTool",
    "EvaluateExpressionTool",
]
