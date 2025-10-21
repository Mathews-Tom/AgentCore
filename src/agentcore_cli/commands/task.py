"""Task lifecycle management commands for AgentCore CLI."""

from __future__ import annotations

import json
import sys
import time
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from agentcore_cli.client import AgentCoreClient
from agentcore_cli.config import Config
from agentcore_cli.exceptions import (
    AgentCoreError,
    AuthenticationError,
    ConnectionError as CliConnectionError,
)
from agentcore_cli.formatters import (
    format_error,
    format_json,
    format_success,
    format_table,
    format_warning,
)

app = typer.Typer(
    name="task",
    help="Create, monitor, and manage tasks",
    no_args_is_help=True,
)

console = Console()


@app.command()
def create(
    task_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Task type (required)"),
    ],
    input_data: Annotated[
        Optional[str],
        typer.Option("--input", "-i", help="Task input data (string or JSON)"),
    ] = None,
    requirements: Annotated[
        Optional[str],
        typer.Option("--requirements", "-r", help="JSON string of requirements"),
    ] = None,
    priority: Annotated[
        Optional[str],
        typer.Option("--priority", "-p", help="Task priority (low, medium, high, critical)"),
    ] = None,
    timeout: Annotated[
        Optional[int],
        typer.Option("--timeout", help="Task timeout in seconds"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Create a new task.

    Example:
        agentcore task create --type "code-review" --input "src/**/*.py"

    With requirements:
        agentcore task create \\
            --type "code-review" \\
            --input "src/**/*.py" \\
            --requirements '{"language": "python"}' \\
            --priority high
    """
    try:
        # Load configuration
        config = Config.load()

        # Parse requirements if provided
        req_dict = {}
        if requirements:
            try:
                req_dict = json.loads(requirements)
                if not isinstance(req_dict, dict):
                    console.print(format_error("Requirements must be a JSON object"))
                    sys.exit(2)
            except json.JSONDecodeError as e:
                console.print(format_error(f"Invalid JSON in requirements: {e}"))
                sys.exit(2)

        # Build task parameters
        params: dict[str, object] = {
            "type": task_type,
        }

        if input_data:
            params["input"] = input_data

        if requirements:
            params["requirements"] = req_dict
        elif config.defaults.task.requirements:
            params["requirements"] = config.defaults.task.requirements

        if priority:
            if priority not in ["low", "medium", "high", "critical"]:
                console.print(format_error(
                    f"Invalid priority: {priority}. Must be one of: low, medium, high, critical"
                ))
                sys.exit(2)
            params["priority"] = priority
        else:
            params["priority"] = config.defaults.task.priority

        if timeout:
            params["timeout"] = timeout
        elif config.defaults.task.timeout:
            params["timeout"] = config.defaults.task.timeout

        # Create client
        auth_token = config.auth.token if config.auth.type == "jwt" else None
        client = AgentCoreClient(
            api_url=config.api.url,
            timeout=config.api.timeout,
            retries=config.api.retries,
            verify_ssl=config.api.verify_ssl,
            auth_token=auth_token,
        )

        # Call API with progress indicator
        if not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Creating task...", total=None)
                result = client.call("task.create", params)
        else:
            result = client.call("task.create", params)

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            task_id = result.get("task_id", "N/A")
            console.print(format_success(f"Task created: {task_id}"))
            console.print(f"  Type: {task_type}")
            console.print(f"  Priority: {params.get('priority', 'N/A')}")
            console.print(f"\n  Check status: agentcore task status {task_id}")

    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except CliConnectionError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(3)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
    except Exception as e:
        console.print(format_error(f"Unexpected error: {e}"))
        raise typer.Exit(1)


@app.command()
def status(
    task_id: Annotated[
        str,
        typer.Argument(help="Task ID to check status for"),
    ],
    watch: Annotated[
        bool,
        typer.Option("--watch", "-w", help="Watch for real-time status updates"),
    ] = False,
    interval: Annotated[
        int,
        typer.Option("--interval", help="Watch interval in seconds"),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Get task status.

    Example:
        agentcore task status task-12345

    Watch mode (real-time updates):
        agentcore task status task-12345 --watch

    Press Ctrl+C to stop watching (task will continue running)
    """
    try:
        # Load configuration
        config = Config.load()

        # Create client
        auth_token = config.auth.token if config.auth.type == "jwt" else None
        client = AgentCoreClient(
            api_url=config.api.url,
            timeout=config.api.timeout,
            retries=config.api.retries,
            verify_ssl=config.api.verify_ssl,
            auth_token=auth_token,
        )

        if watch and not json_output:
            # Watch mode with live updates
            _watch_task_status(client, task_id, interval)
        else:
            # Single status check
            result = client.call("task.status", {"task_id": task_id})

            if json_output:
                console.print(format_json(result))
            else:
                _display_task_status(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching (task continues running)[/yellow]")
        sys.exit(130)
    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except CliConnectionError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(3)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
    except Exception as e:
        console.print(format_error(f"Unexpected error: {e}"))
        raise typer.Exit(1)


@app.command("list")
def list_tasks(
    status_filter: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Filter by status (pending, running, completed, failed)"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of tasks to display"),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """List tasks.

    Example:
        agentcore task list

    Filter by status:
        agentcore task list --status running

    JSON output:
        agentcore task list --json
    """
    try:
        # Load configuration
        config = Config.load()

        # Create client
        auth_token = config.auth.token if config.auth.type == "jwt" else None
        client = AgentCoreClient(
            api_url=config.api.url,
            timeout=config.api.timeout,
            retries=config.api.retries,
            verify_ssl=config.api.verify_ssl,
            auth_token=auth_token,
        )

        # Build parameters
        params = {"limit": limit}
        if status_filter:
            params["status"] = status_filter

        # Call API
        result = client.call("task.list", params)

        # Extract tasks list
        tasks = result.get("tasks", [])

        # Output result
        if json_output:
            console.print(format_json(tasks))
        else:
            if not tasks:
                console.print("[dim]No tasks found[/dim]")
                return

            # Display as table
            columns = ["task_id", "type", "status", "priority", "created_at"]
            table_output = format_table(tasks, columns=columns, title="Tasks")
            console.print(table_output)

            # Show count
            console.print(f"\n[dim]Total: {len(tasks)} task(s)[/dim]")

    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except CliConnectionError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(3)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
    except Exception as e:
        console.print(format_error(f"Unexpected error: {e}"))
        raise typer.Exit(1)


@app.command()
def cancel(
    task_id: Annotated[
        str,
        typer.Argument(help="Task ID to cancel"),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Cancel a running task.

    Example:
        agentcore task cancel task-12345

    Skip confirmation:
        agentcore task cancel task-12345 --force
    """
    try:
        # Load configuration
        config = Config.load()

        # Create client
        auth_token = config.auth.token if config.auth.type == "jwt" else None
        client = AgentCoreClient(
            api_url=config.api.url,
            timeout=config.api.timeout,
            retries=config.api.retries,
            verify_ssl=config.api.verify_ssl,
            auth_token=auth_token,
        )

        # Get task info for confirmation
        if not force and not json_output:
            try:
                task_info = client.call("task.status", {"task_id": task_id})
                task_type = task_info.get("type", task_id)
                task_status = task_info.get("status", "unknown")

                console.print(format_warning(
                    f"Cancel task '{task_type}' ({task_id})?\n"
                    f"   Current status: {task_status}\n"
                    "   This action cannot be undone."
                ))

                confirm = typer.confirm("Continue?", default=False)
                if not confirm:
                    console.print("[yellow]Operation cancelled[/yellow]")
                    sys.exit(0)
            except typer.Exit:
                raise
            except AgentCoreError:
                # Task might not exist, proceed anyway
                pass

        # Call API
        result = client.call("task.cancel", {"task_id": task_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            console.print(format_success(f"Task cancelled: {task_id}"))

    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except CliConnectionError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(3)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
    except Exception as e:
        console.print(format_error(f"Unexpected error: {e}"))
        raise typer.Exit(1)


@app.command()
def result(
    task_id: Annotated[
        str,
        typer.Argument(help="Task ID to retrieve result for"),
    ],
    output_file: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Save result to file"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Get task result and artifacts.

    Example:
        agentcore task result task-12345

    Save to file:
        agentcore task result task-12345 --output result.json

    JSON output:
        agentcore task result task-12345 --json
    """
    try:
        # Load configuration
        config = Config.load()

        # Create client
        auth_token = config.auth.token if config.auth.type == "jwt" else None
        client = AgentCoreClient(
            api_url=config.api.url,
            timeout=config.api.timeout,
            retries=config.api.retries,
            verify_ssl=config.api.verify_ssl,
            auth_token=auth_token,
        )

        # Call API
        result = client.call("task.result", {"task_id": task_id})

        # Save to file if requested
        if output_file:
            import pathlib
            output_path = pathlib.Path(output_file)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            console.print(format_success(f"Result saved to: {output_path.absolute()}"))
            return

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            _display_task_result(result)

    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except CliConnectionError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(3)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
    except Exception as e:
        console.print(format_error(f"Unexpected error: {e}"))
        raise typer.Exit(1)


@app.command()
def retry(
    task_id: Annotated[
        str,
        typer.Argument(help="Task ID to retry"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Retry a failed task.

    Example:
        agentcore task retry task-12345
    """
    try:
        # Load configuration
        config = Config.load()

        # Create client
        auth_token = config.auth.token if config.auth.type == "jwt" else None
        client = AgentCoreClient(
            api_url=config.api.url,
            timeout=config.api.timeout,
            retries=config.api.retries,
            verify_ssl=config.api.verify_ssl,
            auth_token=auth_token,
        )

        # Call API with progress indicator
        if not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Retrying task...", total=None)
                result = client.call("task.retry", {"task_id": task_id})
        else:
            result = client.call("task.retry", {"task_id": task_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            new_task_id = result.get("task_id", task_id)
            console.print(format_success(f"Task retried: {new_task_id}"))
            console.print(f"  Original task: {task_id}")
            console.print(f"\n  Check status: agentcore task status {new_task_id}")

    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except CliConnectionError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(3)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
    except Exception as e:
        console.print(format_error(f"Unexpected error: {e}"))
        raise typer.Exit(1)


# Helper functions

def _watch_task_status(client: AgentCoreClient, task_id: str, interval: int) -> None:
    """Watch task status with live updates.

    Args:
        client: AgentCore client
        task_id: Task ID to watch
        interval: Update interval in seconds
    """
    console.print(f"[bold]Watching task: {task_id}[/bold]")
    console.print("[dim]Press Ctrl+C to stop watching[/dim]\n")

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                # Get current status
                result = client.call("task.status", {"task_id": task_id})

                # Create status table
                status_table = _create_status_table(result)
                live.update(status_table)

                # Check if task is in terminal state
                task_status = result.get("status", "").lower()
                if task_status in ["completed", "failed", "cancelled"]:
                    break

                # Wait before next update
                time.sleep(interval)

            except AgentCoreError as e:
                live.update(format_error(str(e)))
                break

    # Final status message
    task_status = result.get("status", "unknown")
    if task_status.lower() == "completed":
        console.print(f"\n{format_success('Task completed successfully')}")
    elif task_status.lower() == "failed":
        console.print(f"\n{format_error('Task failed')}")
    elif task_status.lower() == "cancelled":
        console.print(f"\n{format_warning('Task was cancelled')}")


def _create_status_table(task_data: dict[str, object]) -> Table:
    """Create a rich table for task status display.

    Args:
        task_data: Task data dictionary

    Returns:
        Rich Table object
    """
    table = Table(title=f"Task: {task_data.get('task_id', 'N/A')}", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    # Add rows
    table.add_row("Status", _format_status_value(str(task_data.get("status", "unknown"))))
    table.add_row("Type", str(task_data.get("type", "N/A")))
    table.add_row("Priority", str(task_data.get("priority", "N/A")))

    # Agent info
    agent_id = task_data.get("agent_id")
    if agent_id:
        table.add_row("Agent", str(agent_id))

    # Progress
    progress = task_data.get("progress")
    if progress is not None:
        progress_bar = _create_progress_bar(float(progress))
        table.add_row("Progress", progress_bar)

    # Timestamps
    created_at = task_data.get("created_at")
    if created_at:
        table.add_row("Created", str(created_at))

    started_at = task_data.get("started_at")
    if started_at:
        table.add_row("Started", str(started_at))

    completed_at = task_data.get("completed_at")
    if completed_at:
        table.add_row("Completed", str(completed_at))

    # Error info
    error = task_data.get("error")
    if error:
        table.add_row("Error", f"[red]{error}[/red]")

    return table


def _format_status_value(status: str) -> str:
    """Format status with color.

    Args:
        status: Status string

    Returns:
        Formatted status with color markup
    """
    status_lower = status.lower()
    if status_lower in ("completed", "success"):
        return f"[green]{status}[/green]"
    if status_lower in ("running", "pending"):
        return f"[yellow]{status}[/yellow]"
    if status_lower in ("failed", "error"):
        return f"[red]{status}[/red]"
    if status_lower == "cancelled":
        return f"[dim]{status}[/dim]"
    return status


def _create_progress_bar(progress: float) -> str:
    """Create a text-based progress bar.

    Args:
        progress: Progress value (0-100)

    Returns:
        Progress bar string
    """
    width = 30
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {progress:.1f}%"


def _display_task_status(task_data: dict[str, object]) -> None:
    """Display task status in detailed view.

    Args:
        task_data: Task data dictionary
    """
    console.print(f"[bold]Task ID:[/bold] {task_data.get('task_id', 'N/A')}")
    console.print(f"[bold]Type:[/bold] {task_data.get('type', 'N/A')}")
    console.print(f"[bold]Status:[/bold] {_format_status_value(str(task_data.get('status', 'unknown')))}")
    console.print(f"[bold]Priority:[/bold] {task_data.get('priority', 'N/A')}")

    # Agent
    agent_id = task_data.get("agent_id")
    if agent_id:
        console.print(f"[bold]Agent:[/bold] {agent_id}")

    # Progress
    progress = task_data.get("progress")
    if progress is not None:
        progress_bar = _create_progress_bar(float(progress))
        console.print(f"[bold]Progress:[/bold] {progress_bar}")

    # Timestamps
    created_at = task_data.get("created_at")
    if created_at:
        console.print(f"[bold]Created:[/bold] {created_at}")

    started_at = task_data.get("started_at")
    if started_at:
        console.print(f"[bold]Started:[/bold] {started_at}")

    completed_at = task_data.get("completed_at")
    if completed_at:
        console.print(f"[bold]Completed:[/bold] {completed_at}")

    # Error
    error = task_data.get("error")
    if error:
        console.print(f"[bold red]Error:[/bold red] {error}")


def _display_task_result(result_data: dict[str, object]) -> None:
    """Display task result in detailed view.

    Args:
        result_data: Result data dictionary
    """
    console.print(f"[bold]Task ID:[/bold] {result_data.get('task_id', 'N/A')}")
    console.print(f"[bold]Status:[/bold] {_format_status_value(str(result_data.get('status', 'unknown')))}")

    # Result output
    output = result_data.get("output")
    if output:
        console.print("\n[bold]Output:[/bold]")
        if isinstance(output, dict):
            console.print(format_json(output))
        else:
            console.print(str(output))

    # Artifacts
    artifacts = result_data.get("artifacts", [])
    if artifacts:
        console.print("\n[bold]Artifacts:[/bold]")
        for i, artifact in enumerate(artifacts, 1):
            artifact_name = artifact.get("name", f"artifact-{i}")
            artifact_type = artifact.get("type", "unknown")
            artifact_size = artifact.get("size", "N/A")
            console.print(f"  {i}. {artifact_name} ({artifact_type}, {artifact_size} bytes)")

    # Metadata
    metadata = result_data.get("metadata")
    if metadata:
        console.print("\n[bold]Metadata:[/bold]")
        console.print(format_json(metadata))
