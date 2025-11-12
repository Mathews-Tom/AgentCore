"""Tests for tool registry implementation."""

import time

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool
from agentcore.agent_runtime.tools.registry import ToolRegistry


class MockSearchTool(Tool):
    """Mock search tool for testing."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="google_search",
            name="Google Search",
            description="Search the web using Google",
            version="1.0.0",
            category=ToolCategory.SEARCH,
            parameters={
                "query": ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                )
            },
            auth_method=AuthMethod.API_KEY,
            capabilities=["web_search", "real_time"],
        )
        super().__init__(metadata)

    async def execute(self, parameters, context):
        pass


class MockCodeTool(Tool):
    """Mock code execution tool for testing."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="python_repl",
            name="Python REPL",
            description="Execute Python code in a sandboxed environment",
            version="2.0.0",
            category=ToolCategory.CODE_EXECUTION,
            parameters={
                "code": ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True,
                )
            },
            auth_method=AuthMethod.NONE,
            capabilities=["python", "sandbox"],
        )
        super().__init__(metadata)

    async def execute(self, parameters, context):
        pass


class MockApiTool(Tool):
    """Mock API tool for testing."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="weather_api",
            name="Weather API",
            description="Get current weather information",
            version="1.5.0",
            category=ToolCategory.API_CLIENT,
            parameters={
                "location": ToolParameter(
                    name="location",
                    type="string",
                    description="Location for weather",
                    required=True,
                )
            },
            auth_method=AuthMethod.API_KEY,
            capabilities=["weather", "real_time"],
        )
        super().__init__(metadata)

    async def execute(self, parameters, context):
        pass


class TestToolRegistryBasics:
    """Test basic registry operations."""

    def test_registry_initialization(self):
        """Test registry initializes with empty state."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.list_all() == []
        stats = registry.get_stats()
        assert stats["total_tools"] == 0

    def test_registry_repr(self):
        """Test registry string representation."""
        registry = ToolRegistry()
        assert "ToolRegistry" in repr(registry)
        assert "tools=0" in repr(registry)


class TestToolRegistration:
    """Test tool registration functionality."""

    def test_register_tool_success(self):
        """Test successful tool registration."""
        registry = ToolRegistry()
        tool = MockSearchTool()

        registry.register(tool)

        assert len(registry) == 1
        assert "google_search" in registry
        assert registry.get("google_search") == tool

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        tool1 = MockSearchTool()
        tool2 = MockCodeTool()
        tool3 = MockApiTool()

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        assert len(registry) == 3
        assert "google_search" in registry
        assert "python_repl" in registry
        assert "weather_api" in registry

    def test_register_duplicate_tool_id_replaces(self):
        """Test that registering duplicate tool_id replaces the old one."""
        registry = ToolRegistry()
        tool1 = MockSearchTool()

        registry.register(tool1)
        assert len(registry) == 1

        # Create new tool with same ID but different version
        tool2 = MockSearchTool()
        tool2.metadata.version = "2.0.0"

        registry.register(tool2)
        assert len(registry) == 1  # Still just one tool
        assert registry.get("google_search").metadata.version == "2.0.0"

    def test_register_none_tool_raises_error(self):
        """Test that registering None raises ValueError."""
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="Tool cannot be None"):
            registry.register(None)

    def test_register_tool_with_none_metadata_raises_error(self):
        """Test that registering tool with None metadata raises ValueError."""
        registry = ToolRegistry()
        tool = MockSearchTool()
        tool.metadata = None

        with pytest.raises(ValueError, match="Tool metadata cannot be None"):
            registry.register(tool)

    def test_register_tool_with_empty_id_raises_error(self):
        """Test that registering tool with empty ID raises ValueError."""
        registry = ToolRegistry()
        tool = MockSearchTool()
        tool.metadata.tool_id = ""

        with pytest.raises(ValueError, match="Tool ID cannot be empty"):
            registry.register(tool)

        tool.metadata.tool_id = "   "  # Whitespace only
        with pytest.raises(ValueError, match="Tool ID cannot be empty"):
            registry.register(tool)


class TestToolLookup:
    """Test tool lookup functionality."""

    def test_get_existing_tool(self):
        """Test retrieving an existing tool by ID."""
        registry = ToolRegistry()
        tool = MockSearchTool()
        registry.register(tool)

        retrieved = registry.get("google_search")

        assert retrieved is not None
        assert retrieved == tool
        assert retrieved.metadata.tool_id == "google_search"

    def test_get_nonexistent_tool_returns_none(self):
        """Test that getting nonexistent tool returns None."""
        registry = ToolRegistry()

        result = registry.get("nonexistent_tool")

        assert result is None

    def test_get_lookup_performance(self):
        """Test that lookup is fast (<10ms) even with many tools.

        Implements NFR-1.2: <10ms lookup for 1000+ tools.
        """
        registry = ToolRegistry()

        # Register 1000 tools
        for i in range(1000):
            tool = MockSearchTool()
            tool.metadata.tool_id = f"tool_{i}"
            tool.metadata.name = f"Tool {i}"
            registry.register(tool)

        # Measure lookup time for 100 lookups
        lookup_times = []
        for i in range(100):
            start = time.perf_counter()
            registry.get(f"tool_{i}")
            end = time.perf_counter()
            lookup_times.append((end - start) * 1000)  # Convert to ms

        avg_lookup_time = sum(lookup_times) / len(lookup_times)
        max_lookup_time = max(lookup_times)

        # Assert performance requirement
        assert avg_lookup_time < 10, f"Average lookup time {avg_lookup_time:.2f}ms exceeds 10ms"
        assert max_lookup_time < 10, f"Max lookup time {max_lookup_time:.2f}ms exceeds 10ms"

    def test_contains_operator(self):
        """Test __contains__ operator for checking tool existence."""
        registry = ToolRegistry()
        tool = MockSearchTool()
        registry.register(tool)

        assert "google_search" in registry
        assert "nonexistent" not in registry


class TestToolListing:
    """Test tool listing functionality."""

    def test_list_all_empty_registry(self):
        """Test listing all tools in empty registry."""
        registry = ToolRegistry()

        tools = registry.list_all()

        assert tools == []

    def test_list_all_with_tools(self):
        """Test listing all registered tools."""
        registry = ToolRegistry()
        tool1 = MockSearchTool()
        tool2 = MockCodeTool()
        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_all()

        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_list_by_category_empty(self):
        """Test listing tools by category when none exist."""
        registry = ToolRegistry()

        tools = registry.list_by_category(ToolCategory.SEARCH)

        assert tools == []

    def test_list_by_category_with_tools(self):
        """Test listing tools by category."""
        registry = ToolRegistry()
        search_tool = MockSearchTool()
        code_tool = MockCodeTool()
        api_tool = MockApiTool()
        registry.register(search_tool)
        registry.register(code_tool)
        registry.register(api_tool)

        search_tools = registry.list_by_category(ToolCategory.SEARCH)
        code_tools = registry.list_by_category(ToolCategory.CODE_EXECUTION)
        api_tools = registry.list_by_category(ToolCategory.API_CLIENT)

        assert len(search_tools) == 1
        assert search_tool in search_tools
        assert len(code_tools) == 1
        assert code_tool in code_tools
        assert len(api_tools) == 1
        assert api_tool in api_tools


class TestToolSearch:
    """Test tool search functionality."""

    def test_search_no_filters_returns_all(self):
        """Test search with no filters returns all tools."""
        registry = ToolRegistry()
        tool1 = MockSearchTool()
        tool2 = MockCodeTool()
        registry.register(tool1)
        registry.register(tool2)

        results = registry.search()

        assert len(results) == 2

    def test_search_by_name(self):
        """Test searching tools by name."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())
        registry.register(MockApiTool())

        results = registry.search("Google")

        assert len(results) == 1
        assert results[0].metadata.tool_id == "google_search"

    def test_search_by_description(self):
        """Test searching tools by description."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())

        results = registry.search("sandboxed")

        assert len(results) == 1
        assert results[0].metadata.tool_id == "python_repl"

    def test_search_by_capabilities(self):
        """Test searching tools by capabilities."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())
        registry.register(MockApiTool())

        results = registry.search("real_time")

        assert len(results) == 2  # google_search and weather_api
        tool_ids = {tool.metadata.tool_id for tool in results}
        assert "google_search" in tool_ids
        assert "weather_api" in tool_ids

    def test_search_case_insensitive(self):
        """Test that search is case-insensitive."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())

        results_lower = registry.search("google")
        results_upper = registry.search("GOOGLE")
        results_mixed = registry.search("GoOgLe")

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert len(results_mixed) == 1

    def test_search_by_category_only(self):
        """Test searching by category filter only."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())
        registry.register(MockApiTool())

        results = registry.search(category=ToolCategory.SEARCH)

        assert len(results) == 1
        assert results[0].metadata.category == ToolCategory.SEARCH

    def test_search_by_query_and_category(self):
        """Test searching with both query and category filters."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())
        registry.register(MockApiTool())

        # Search for "real_time" in SEARCH category only
        results = registry.search("real_time", category=ToolCategory.SEARCH)

        assert len(results) == 1
        assert results[0].metadata.tool_id == "google_search"

    def test_search_no_matches(self):
        """Test search with no matches returns empty list."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())

        results = registry.search("nonexistent_keyword")

        assert results == []

    def test_search_empty_query_with_category(self):
        """Test that empty query with category returns all tools in category."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())

        results = registry.search("", category=ToolCategory.SEARCH)

        assert len(results) == 1
        assert results[0].metadata.tool_id == "google_search"


class TestToolUnregistration:
    """Test tool unregistration functionality."""

    def test_unregister_existing_tool(self):
        """Test unregistering an existing tool."""
        registry = ToolRegistry()
        tool = MockSearchTool()
        registry.register(tool)

        result = registry.unregister("google_search")

        assert result is True
        assert len(registry) == 0
        assert "google_search" not in registry
        assert registry.get("google_search") is None

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool returns False."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent_tool")

        assert result is False

    def test_unregister_removes_from_category_index(self):
        """Test that unregistration removes tool from category index."""
        registry = ToolRegistry()
        tool = MockSearchTool()
        registry.register(tool)

        registry.unregister("google_search")

        search_tools = registry.list_by_category(ToolCategory.SEARCH)
        assert len(search_tools) == 0


class TestRegistryStats:
    """Test registry statistics functionality."""

    def test_get_stats_empty_registry(self):
        """Test stats for empty registry."""
        registry = ToolRegistry()

        stats = registry.get_stats()

        assert stats["total_tools"] == 0
        assert stats["tools_by_category"] == {}

    def test_get_stats_with_tools(self):
        """Test stats with registered tools."""
        registry = ToolRegistry()
        registry.register(MockSearchTool())
        registry.register(MockCodeTool())
        registry.register(MockApiTool())

        stats = registry.get_stats()

        assert stats["total_tools"] == 3
        assert stats["tools_by_category"]["search"] == 1
        assert stats["tools_by_category"]["code_execution"] == 1
        assert stats["tools_by_category"]["api_client"] == 1

    def test_get_stats_multiple_tools_same_category(self):
        """Test stats with multiple tools in same category."""
        registry = ToolRegistry()
        tool1 = MockSearchTool()
        tool2 = MockSearchTool()
        tool2.metadata.tool_id = "bing_search"
        registry.register(tool1)
        registry.register(tool2)

        stats = registry.get_stats()

        assert stats["total_tools"] == 2
        assert stats["tools_by_category"]["search"] == 2

    def test_len_operator(self):
        """Test __len__ operator."""
        registry = ToolRegistry()

        assert len(registry) == 0

        registry.register(MockSearchTool())
        assert len(registry) == 1

        registry.register(MockCodeTool())
        assert len(registry) == 2
