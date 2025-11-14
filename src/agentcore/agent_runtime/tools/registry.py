"""Tool registry for centralized tool management and discovery.

This module implements the ToolRegistry class that manages tool registration, lookup,
and search functionality. Following FR-1 specification from docs/specs/tool-integration/spec.md.

The registry provides:
- Fast tool lookup (<10ms for 1000+ tools)
- Category-based indexing for efficient filtering
- Fuzzy search by name, description, and capabilities
- Tool versioning support
"""

from collections import defaultdict
from typing import Any

import structlog

from ..models.tool_integration import ToolCategory
from .base import Tool

logger = structlog.get_logger()


class ToolRegistry:
    """Centralized registry for tool registration and discovery.

    Implements FR-1: Tool Registry specification with in-memory storage,
    category indexing, and search capabilities. Optimized for <10ms lookups
    (NFR-1.2) and supports 10,000+ tools (NFR-3.1).

    The registry maintains two data structures:
    - _tools: Dict mapping tool_id to Tool instance for O(1) lookup
    - _categories: Dict mapping category to list of tool_ids for fast filtering

    Thread-safety: This implementation is not thread-safe. In production,
    use appropriate locking mechanisms for concurrent access.

    Attributes:
        _tools: Dictionary mapping tool_id to Tool instances
        _categories: Dictionary mapping ToolCategory to lists of tool_ids

    Example:
        ```python
        registry = ToolRegistry()

        # Register a tool
        registry.register(my_tool)

        # Lookup by ID
        tool = registry.get("google_search")

        # Search by query
        results = registry.search("search", category=ToolCategory.SEARCH)

        # List by category
        search_tools = registry.list_by_category(ToolCategory.SEARCH)
        ```
    """

    def __init__(self):
        """Initialize empty tool registry with indexes."""
        self._tools: dict[str, Tool] = {}
        self._categories: dict[ToolCategory, list[str]] = defaultdict(list)
        self.logger = logger.bind(component="tool_registry")
        self.logger.info("tool_registry_initialized")

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Validates tool metadata and stores it for discovery. If a tool with the
        same ID already exists, it will be replaced (useful for version updates).

        Implements FR-1.1 and FR-1.2: Tool registration with metadata validation.

        Args:
            tool: Tool instance to register (must have valid metadata)

        Raises:
            ValueError: If tool or tool.metadata is None
            ValueError: If tool_id is empty or invalid

        Example:
            ```python
            registry = ToolRegistry()
            registry.register(GoogleSearchTool())
            ```
        """
        if tool is None:
            raise ValueError("Tool cannot be None")

        if tool.metadata is None:
            raise ValueError("Tool metadata cannot be None")

        if not tool.metadata.tool_id or not tool.metadata.tool_id.strip():
            raise ValueError("Tool ID cannot be empty")

        tool_id = tool.metadata.tool_id
        category = tool.metadata.category

        # Remove old registration if exists
        if tool_id in self._tools:
            old_tool = self._tools[tool_id]
            old_category = old_tool.metadata.category
            if tool_id in self._categories[old_category]:
                self._categories[old_category].remove(tool_id)
            self.logger.info(
                "tool_updated",
                tool_id=tool_id,
                old_version=old_tool.metadata.version,
                new_version=tool.metadata.version,
            )

        # Register new tool
        self._tools[tool_id] = tool
        self._categories[category].append(tool_id)

        self.logger.info(
            "tool_registered",
            tool_id=tool_id,
            name=tool.metadata.name,
            category=category.value,
            version=tool.metadata.version,
        )

    def get(self, tool_id: str) -> Tool | None:
        """Get a tool by its unique identifier.

        Provides O(1) lookup performance as required by NFR-1.2 (<10ms for 1000+ tools).

        Implements FR-1.1: Tool discovery by ID.

        Args:
            tool_id: Unique identifier of the tool to retrieve

        Returns:
            Tool instance if found, None otherwise

        Example:
            ```python
            tool = registry.get("google_search")
            if tool:
                result = await tool.execute(params, context)
            ```
        """
        return self._tools.get(tool_id)

    def search(
        self,
        query: str | None = None,
        category: ToolCategory | None = None,
    ) -> list[Tool]:
        """Search for tools by query and/or category.

        Implements FR-1.3: Tool search by name, description, and capabilities.
        Implements FR-1.4: Tool filtering by category.

        The search performs fuzzy matching (case-insensitive substring match) across:
        - Tool name
        - Tool description
        - Tool capabilities (if present)

        If both query and category are provided, tools must match both criteria.
        If neither is provided, returns all registered tools.

        Args:
            query: Optional search string to match against name, description, capabilities
            category: Optional category filter

        Returns:
            List of Tool instances matching the search criteria

        Example:
            ```python
            # Search for search tools
            tools = registry.search(category=ToolCategory.SEARCH)

            # Search by keyword
            tools = registry.search("google")

            # Combined search
            tools = registry.search("advanced", category=ToolCategory.SEARCH)
            ```
        """
        # Start with tools filtered by category if specified
        if category is not None:
            candidate_ids = self._categories.get(category, [])
            candidates = [self._tools[tool_id] for tool_id in candidate_ids]
        else:
            candidates = list(self._tools.values())

        # If no query, return all candidates
        if query is None or not query.strip():
            return candidates

        # Fuzzy search across name, description, and capabilities
        query_lower = query.lower()
        results = []

        for tool in candidates:
            # Check name
            if query_lower in tool.metadata.name.lower():
                results.append(tool)
                continue

            # Check description
            if query_lower in tool.metadata.description.lower():
                results.append(tool)
                continue

            # Check capabilities if present
            if hasattr(tool.metadata, "capabilities") and tool.metadata.capabilities:
                capabilities_str = " ".join(tool.metadata.capabilities).lower()
                if query_lower in capabilities_str:
                    results.append(tool)
                    continue

        self.logger.debug(
            "tool_search_completed",
            query=query,
            category=category.value if category else None,
            result_count=len(results),
        )

        return results

    def list_by_category(self, category: ToolCategory) -> list[Tool]:
        """List all tools in a specific category.

        Provides efficient category-based filtering using pre-built indexes.

        Implements FR-1.4: Tool listing by category.

        Args:
            category: Category to filter by

        Returns:
            List of Tool instances in the specified category

        Example:
            ```python
            search_tools = registry.list_by_category(ToolCategory.SEARCH)
            for tool in search_tools:
                print(f"{tool.metadata.name}: {tool.metadata.description}")
            ```
        """
        tool_ids = self._categories.get(category, [])
        return [self._tools[tool_id] for tool_id in tool_ids]

    def list_all(self) -> list[Tool]:
        """List all registered tools.

        Returns:
            List of all Tool instances in the registry

        Example:
            ```python
            all_tools = registry.list_all()
            print(f"Total tools: {len(all_tools)}")
            ```
        """
        return list(self._tools.values())

    def unregister(self, tool_id: str) -> bool:
        """Unregister a tool from the registry.

        Removes the tool from both the main registry and category index.

        Args:
            tool_id: Unique identifier of the tool to remove

        Returns:
            True if tool was removed, False if tool was not found

        Example:
            ```python
            if registry.unregister("deprecated_tool"):
                print("Tool removed successfully")
            ```
        """
        if tool_id not in self._tools:
            return False

        tool = self._tools[tool_id]
        category = tool.metadata.category

        # Remove from main registry
        del self._tools[tool_id]

        # Remove from category index
        if tool_id in self._categories[category]:
            self._categories[category].remove(tool_id)

        self.logger.info("tool_unregistered", tool_id=tool_id)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary containing registry metrics:
            - total_tools: Total number of registered tools
            - tools_by_category: Tool count per category

        Example:
            ```python
            stats = registry.get_stats()
            print(f"Total tools: {stats['total_tools']}")
            for category, count in stats['tools_by_category'].items():
                print(f"  {category}: {count}")
            ```
        """
        return {
            "total_tools": len(self._tools),
            "tools_by_category": {
                category.value: len(tool_ids)
                for category, tool_ids in self._categories.items()
            },
        }

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        return tool_id in self._tools

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"<ToolRegistry(tools={len(self._tools)})>"
