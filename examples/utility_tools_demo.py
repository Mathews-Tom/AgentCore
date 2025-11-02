"""Demo script showing utility tools in action.

This script demonstrates the three new utility tools:
- calculator: Basic arithmetic operations
- get_current_time: Get current time with formatting
- echo: Echo messages with transformations
"""

import asyncio

from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionRequest,
    ToolExecutionStatus,
)
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_registry import ToolRegistry
from agentcore.agent_runtime.tools import (
    register_api_tools,
    register_code_execution_tools,
    register_search_tools,
    register_utility_tools,
)


async def demo_calculator():
    """Demo calculator tool."""
    print("\n" + "=" * 60)
    print("CALCULATOR TOOL DEMO")
    print("=" * 60)

    registry = ToolRegistry()
    register_utility_tools(registry)
    executor = ToolExecutor(registry=registry, enable_metrics=False)

    operations = [
        ("+", 15, 25, "Addition"),
        ("-", 100, 42, "Subtraction"),
        ("*", 7, 8, "Multiplication"),
        ("/", 144, 12, "Division"),
        ("%", 17, 5, "Modulo"),
        ("**", 2, 10, "Power"),
    ]

    for op, a, b, name in operations:
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": op, "a": a, "b": b},
            agent_id="demo-agent",
        )

        result = await executor.execute(request)

        if result.status == ToolExecutionStatus.SUCCESS:
            print(f"\n{name}:")
            print(f"  Expression: {result.result['expression']}")
            print(f"  Result: {result.result['result']}")
        else:
            print(f"\n{name} FAILED: {result.error}")


async def demo_get_current_time():
    """Demo get_current_time tool."""
    print("\n" + "=" * 60)
    print("GET CURRENT TIME TOOL DEMO")
    print("=" * 60)

    registry = ToolRegistry()
    register_utility_tools(registry)
    executor = ToolExecutor(registry=registry, enable_metrics=False)

    formats = ["iso", "unix", "human"]

    for fmt in formats:
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={"timezone": "UTC", "format": fmt},
            agent_id="demo-agent",
        )

        result = await executor.execute(request)

        if result.status == ToolExecutionStatus.SUCCESS:
            print(f"\nFormat: {fmt}")
            print(f"  Current Time: {result.result['current_time']}")
            print(f"  Timezone: {result.result['timezone']}")
        else:
            print(f"\nFormat {fmt} FAILED: {result.error}")


async def demo_echo():
    """Demo echo tool."""
    print("\n" + "=" * 60)
    print("ECHO TOOL DEMO")
    print("=" * 60)

    registry = ToolRegistry()
    register_utility_tools(registry)
    executor = ToolExecutor(registry=registry, enable_metrics=False)

    messages = [
        ("Hello, World!", False, "Basic echo"),
        ("hello, world!", True, "Uppercase echo"),
        ("The quick brown fox jumps", False, "Word count demo"),
    ]

    for message, uppercase, description in messages:
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": message, "uppercase": uppercase},
            agent_id="demo-agent",
        )

        result = await executor.execute(request)

        if result.status == ToolExecutionStatus.SUCCESS:
            print(f"\n{description}:")
            print(f"  Input: {result.result['original']}")
            print(f"  Output: {result.result['echo']}")
            print(f"  Length: {result.result['length']} characters")
            print(f"  Words: {result.result['word_count']}")
        else:
            print(f"\n{description} FAILED: {result.error}")


async def demo_all_tools():
    """Show all registered tools."""
    print("\n" + "=" * 60)
    print("ALL REGISTERED TOOLS")
    print("=" * 60)

    registry = ToolRegistry()
    register_utility_tools(registry)
    register_search_tools(registry)
    register_code_execution_tools(registry)
    register_api_tools(registry)

    tools = registry.list_tools()

    by_category = {}
    for tool in tools:
        category = tool.category.value
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(tool)

    for category, category_tools in sorted(by_category.items()):
        print(f"\n{category.upper()}:")
        for tool in sorted(category_tools, key=lambda t: t.tool_id):
            print(f"  - {tool.tool_id}: {tool.name}")

    print(f"\nTotal: {len(tools)} tools registered")


async def main():
    """Run all demos."""
    await demo_calculator()
    await demo_get_current_time()
    await demo_echo()
    await demo_all_tools()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
