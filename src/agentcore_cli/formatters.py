"""Output formatters for AgentCore CLI.

Provides multiple output formats:
- JSON: Machine-readable format for scripting
- Table: Human-readable tabular format (default)
- Tree: Hierarchical visualization

Features:
- Color auto-detection (disabled for piped output)
- Timestamp formatting options
- Column selection for tables
- Pagination support
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.tree import Tree


def _should_use_color() -> bool:
    """Determine if color output should be used.

    Color is disabled when:
    - Output is piped (not a TTY)
    - NO_COLOR environment variable is set
    - TERM is set to "dumb"

    Returns:
        True if color should be used, False otherwise
    """
    import os

    # Check NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return False

    # Check TERM environment variable
    if os.environ.get("TERM") == "dumb":
        return False

    # Check if stdout is a TTY
    if not sys.stdout.isatty():
        return False

    return True


def _get_console(force_color: bool | None = None) -> Console:
    """Get a Rich Console with appropriate color settings.

    Args:
        force_color: If True, force color output. If False, disable color.
                    If None, auto-detect based on environment.

    Returns:
        Configured Console instance
    """
    if force_color is None:
        force_color = _should_use_color()

    return Console(
        force_terminal=force_color,
        no_color=not force_color,
        legacy_windows=False,
    )


def _format_timestamp(
    timestamp: str | datetime | None,
    include_time: bool = False,
) -> str:
    """Format a timestamp for display.

    Args:
        timestamp: Timestamp as ISO string, datetime object, or None
        include_time: Whether to include time portion (default: False)

    Returns:
        Formatted timestamp string or "N/A" if None
    """
    if timestamp is None:
        return "N/A"

    # Parse timestamp if it's a string
    if isinstance(timestamp, str):
        try:
            # Try parsing ISO format
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            # Return as-is if parsing fails
            return timestamp
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)

    # Format based on include_time flag
    if include_time:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d")


def format_json(data: Any, pretty: bool = True) -> str:
    """Format data as JSON.

    Args:
        data: Data to format
        pretty: Whether to pretty-print (default: True)

    Returns:
        JSON-formatted string
    """
    if pretty:
        return json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def format_table(
    data: list[dict[str, Any]],
    columns: list[str] | None = None,
    title: str | None = None,
    timestamps: bool = False,
    limit: int | None = None,
    force_color: bool | None = None,
) -> str:
    """Format data as a rich table.

    Args:
        data: List of dictionaries to display as rows
        columns: Optional list of column keys to display (default: all keys)
        title: Optional table title
        timestamps: Whether to include time in timestamp fields (default: False)
        limit: Maximum number of rows to display (default: None for all)
        force_color: Force color output (None for auto-detect)

    Returns:
        Formatted table string
    """
    if not data:
        return "[dim]No data to display[/dim]"

    # Apply limit if specified
    if limit is not None and limit > 0:
        data = data[:limit]

    # Auto-detect columns if not specified
    if columns is None:
        columns = list(data[0].keys()) if data else []
    else:
        # Filter to only columns that exist in data
        first_row_keys = set(data[0].keys()) if data else set()
        columns = [c for c in columns if c in first_row_keys]

    # Create table
    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    for col in columns:
        # Convert snake_case to Title Case
        col_title = col.replace("_", " ").title()
        table.add_column(col_title, overflow="fold")

    # Add rows
    for row in data:
        formatted_row = []
        for col in columns:
            value = row.get(col, "")
            # Format timestamps if column name suggests it's a timestamp
            if timestamps and isinstance(value, str) and any(
                time_field in col.lower()
                for time_field in ["_at", "timestamp", "time", "date"]
            ):
                value = _format_timestamp(value, include_time=timestamps)
            formatted_row.append(_format_value(value))
        table.add_row(*formatted_row)

    # Capture output with appropriate console
    console = _get_console(force_color)
    with console.capture() as capture:
        console.print(table)

    return capture.get()


def format_tree(
    data: dict[str, Any],
    label: str = "Root",
    timestamps: bool = False,
    force_color: bool | None = None,
) -> str:
    """Format data as a rich tree.

    Args:
        data: Dictionary to display as tree
        label: Root node label
        timestamps: Whether to include time in timestamp fields (default: False)
        force_color: Force color output (None for auto-detect)

    Returns:
        Formatted tree string
    """
    tree = Tree(label)
    _build_tree(tree, data, timestamps=timestamps)

    # Capture output with appropriate console
    console = _get_console(force_color)
    with console.capture() as capture:
        console.print(tree)

    return capture.get()


def format_agent_info(
    agent_data: dict[str, Any],
    timestamps: bool = False,
    force_color: bool | None = None,
) -> str:
    """Format agent information in a detailed view.

    Args:
        agent_data: Agent data dictionary
        timestamps: Whether to include time in timestamp fields (default: False)
        force_color: Force color output (None for auto-detect)

    Returns:
        Formatted agent info
    """
    console = _get_console(force_color)
    output_lines = []

    # Basic info
    output_lines.append(f"[bold]Agent ID:[/bold] {agent_data.get('agent_id', 'N/A')}")
    output_lines.append(f"[bold]Name:[/bold] {agent_data.get('name', 'N/A')}")
    output_lines.append(f"[bold]Status:[/bold] {_format_status(agent_data.get('status', 'unknown'))}")

    # Capabilities
    capabilities = agent_data.get("capabilities", [])
    if capabilities:
        caps_str = ", ".join(capabilities)
        output_lines.append(f"[bold]Capabilities:[/bold] {caps_str}")

    # Requirements
    requirements = agent_data.get("requirements", {})
    if requirements:
        output_lines.append("[bold]Requirements:[/bold]")
        for key, value in requirements.items():
            output_lines.append(f"  • {key}: {value}")

    # Cost
    cost = agent_data.get("cost_per_request")
    if cost is not None:
        output_lines.append(f"[bold]Cost per Request:[/bold] ${cost:.4f}")

    # Timestamps
    registered = agent_data.get("registered_at") or agent_data.get("created_at")
    if registered:
        formatted_registered = _format_timestamp(registered, include_time=timestamps)
        output_lines.append(f"[bold]Registered:[/bold] {formatted_registered}")

    updated = agent_data.get("updated_at")
    if updated:
        formatted_updated = _format_timestamp(updated, include_time=timestamps)
        output_lines.append(f"[bold]Last Updated:[/bold] {formatted_updated}")

    # Health status
    health = agent_data.get("health_status")
    if health:
        output_lines.append(f"[bold]Health:[/bold] {health}")

    # Active tasks
    active_tasks = agent_data.get("active_tasks")
    if active_tasks is not None:
        output_lines.append(f"[bold]Active Tasks:[/bold] {active_tasks}")

    return "\n".join(output_lines)


def _build_tree(
    tree: Tree,
    data: Any,
    max_depth: int = 5,
    current_depth: int = 0,
    timestamps: bool = False,
    parent_key: str = "",
) -> None:
    """Recursively build a tree from data.

    Args:
        tree: Tree node to add to
        data: Data to add
        max_depth: Maximum tree depth
        current_depth: Current depth level
        timestamps: Whether to include time in timestamp fields
        parent_key: Parent key name for timestamp detection
    """
    if current_depth >= max_depth:
        tree.add("[dim]...[/dim]")
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                branch = tree.add(f"[bold]{key}[/bold]")
                _build_tree(
                    branch,
                    value,
                    max_depth,
                    current_depth + 1,
                    timestamps,
                    parent_key=key,
                )
            else:
                # Format timestamp values if enabled
                formatted_value = value
                if timestamps and isinstance(value, str) and any(
                    time_field in key.lower()
                    for time_field in ["_at", "timestamp", "time", "date"]
                ):
                    formatted_value = _format_timestamp(value, include_time=timestamps)
                tree.add(f"[bold]{key}:[/bold] {_format_value(formatted_value)}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                branch = tree.add(f"[bold][{i}][/bold]")
                _build_tree(
                    branch,
                    item,
                    max_depth,
                    current_depth + 1,
                    timestamps,
                    parent_key=parent_key,
                )
            else:
                tree.add(f"[{i}] {_format_value(item)}")
    else:
        tree.add(str(data))


def _format_value(value: Any) -> str:
    """Format a value for display.

    Args:
        value: Value to format

    Returns:
        Formatted string
    """
    if value is None:
        return "[dim]N/A[/dim]"
    if isinstance(value, bool):
        return "[green]Yes[/green]" if value else "[red]No[/red]"
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[dim]None[/dim]"
        if all(isinstance(x, str) for x in value):
            return ", ".join(str(x) for x in value)
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_status(status: str) -> str:
    """Format status with color.

    Args:
        status: Status string

    Returns:
        Formatted status with color markup
    """
    status_lower = status.lower()
    if status_lower in ("active", "online", "running", "healthy"):
        return f"[green]{status}[/green]"
    if status_lower in ("inactive", "offline", "stopped", "paused"):
        return f"[yellow]{status}[/yellow]"
    if status_lower in ("failed", "error", "unhealthy"):
        return f"[red]{status}[/red]"
    return status


def format_success(message: str) -> str:
    """Format a success message.

    Args:
        message: Success message

    Returns:
        Formatted success message
    """
    return f"[green]✓[/green] {message}"


def format_error(message: str) -> str:
    """Format an error message.

    Args:
        message: Error message

    Returns:
        Formatted error message
    """
    return f"[red]✗[/red] {message}"


def format_warning(message: str) -> str:
    """Format a warning message.

    Args:
        message: Warning message

    Returns:
        Formatted warning message
    """
    return f"[yellow]⚠[/yellow] {message}"
