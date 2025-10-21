"""Workflow management commands for AgentCore CLI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

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
    name="workflow",
    help="Define and execute multi-step workflows",
    no_args_is_help=True,
)

console = Console()


# Workflow definition models for validation
class WorkflowTask(BaseModel):
    """Workflow task definition."""

    name: str = Field(..., description="Task name")
    type: str = Field(..., description="Task type")
    input: str | dict[str, object] | None = Field(None, description="Task input data")
    requirements: dict[str, object] | None = Field(None, description="Task requirements")
    depends_on: list[str] | None = Field(None, description="Task dependencies")
    priority: str | None = Field(None, description="Task priority")
    timeout: int | None = Field(None, description="Task timeout in seconds")


class WorkflowDefinition(BaseModel):
    """Workflow definition schema."""

    name: str = Field(..., description="Workflow name")
    description: str | None = Field(None, description="Workflow description")
    version: str | None = Field("1.0", description="Workflow version")
    tasks: list[WorkflowTask] = Field(..., description="List of tasks")
    max_retries: int | None = Field(None, description="Maximum retries per task")
    timeout: int | None = Field(None, description="Overall workflow timeout")
    metadata: dict[str, object] | None = Field(None, description="Custom metadata")


@app.command()
def create(
    file: Annotated[
        str,
        typer.Option("--file", "-f", help="YAML workflow definition file (required)"),
    ],
    validate_only: Annotated[
        bool,
        typer.Option("--validate-only", help="Only validate without creating"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Create workflow from YAML definition file.

    Example workflow.yaml:
        name: "code-review-workflow"
        description: "Automated code review process"
        tasks:
          - name: "lint-check"
            type: "code-linting"
            requirements:
              language: "python"
          - name: "security-scan"
            type: "security-analysis"
            depends_on: ["lint-check"]
          - name: "unit-tests"
            type: "test-execution"
            depends_on: ["lint-check"]

    Example:
        agentcore workflow create --file workflow.yaml
    """
    try:
        # Load and validate YAML file
        workflow_path = Path(file)
        if not workflow_path.exists():
            console.print(format_error(f"Workflow file not found: {file}"))
            sys.exit(2)

        # Parse YAML
        try:
            with open(workflow_path) as f:
                workflow_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            console.print(format_error(f"Invalid YAML syntax: {e}"))
            sys.exit(2)

        # Validate workflow definition
        try:
            workflow_def = WorkflowDefinition(**workflow_data)
        except ValidationError as e:
            console.print(format_error("Workflow validation failed:"))
            for error in e.errors():
                field = " -> ".join(str(x) for x in error["loc"])
                console.print(f"  • {field}: {error['msg']}")
            sys.exit(2)

        # If validation only, exit here
        if validate_only:
            console.print(format_success(f"Workflow definition is valid: {workflow_def.name}"))
            console.print(f"  Tasks: {len(workflow_def.tasks)}")
            console.print(f"  Version: {workflow_def.version}")
            return

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

        # Build workflow parameters
        params = {
            "name": workflow_def.name,
            "description": workflow_def.description,
            "version": workflow_def.version,
            "tasks": [task.model_dump(exclude_none=True) for task in workflow_def.tasks],
        }

        if workflow_def.max_retries is not None:
            params["max_retries"] = workflow_def.max_retries
        if workflow_def.timeout is not None:
            params["timeout"] = workflow_def.timeout
        if workflow_def.metadata:
            params["metadata"] = workflow_def.metadata

        # Call API with progress indicator
        if not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Creating workflow...", total=None)
                result = client.call("workflow.create", params)
        else:
            result = client.call("workflow.create", params)

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            workflow_id = result.get("workflow_id", "N/A")
            console.print(format_success(f"Workflow created: {workflow_id}"))
            console.print(f"  Name: {workflow_def.name}")
            console.print(f"  Tasks: {len(workflow_def.tasks)}")
            console.print(f"\n  Execute: agentcore workflow execute {workflow_id}")
            console.print(f"  Visualize: agentcore workflow visualize {workflow_id}")

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
def execute(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow ID to execute"),
    ],
    watch: Annotated[
        bool,
        typer.Option("--watch", "-w", help="Watch for real-time execution updates"),
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
    """Execute a workflow.

    Example:
        agentcore workflow execute workflow-12345

    Watch mode (real-time updates):
        agentcore workflow execute workflow-12345 --watch

    Press Ctrl+C to stop watching (workflow will continue)
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

        # Call API to start execution
        if not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Starting workflow...", total=None)
                result = client.call("workflow.execute", {"workflow_id": workflow_id})
        else:
            result = client.call("workflow.execute", {"workflow_id": workflow_id})

        # If watch mode, monitor execution
        if watch and not json_output:
            console.print(format_success("Workflow execution started"))
            _watch_workflow_execution(client, workflow_id, interval)
        else:
            # Output result
            if json_output:
                console.print(format_json(result))
            else:
                console.print(format_success(f"Workflow execution started: {workflow_id}"))
                console.print(f"\n  Check status: agentcore workflow status {workflow_id}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching (workflow continues running)[/yellow]")
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


@app.command()
def status(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow ID to check status for"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Get workflow status showing task progress.

    Example:
        agentcore workflow status workflow-12345
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
        result = client.call("workflow.status", {"workflow_id": workflow_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            _display_workflow_status(result)

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
def list_workflows(
    status_filter: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Filter by status (pending, running, completed, failed)"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of workflows to display"),
    ] = 100,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """List workflows.

    Example:
        agentcore workflow list

    Filter by status:
        agentcore workflow list --status running

    JSON output:
        agentcore workflow list --json
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
        result = client.call("workflow.list", params)

        # Extract workflows list
        workflows = result.get("workflows", [])

        # Output result
        if json_output:
            console.print(format_json(workflows))
        else:
            if not workflows:
                console.print("[dim]No workflows found[/dim]")
                return

            # Display as table
            columns = ["workflow_id", "name", "status", "tasks_total", "tasks_completed", "created_at"]
            table_output = format_table(workflows, columns=columns, title="Workflows")
            console.print(table_output)

            # Show count
            console.print(f"\n[dim]Total: {len(workflows)} workflow(s)[/dim]")

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
def visualize(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow ID to visualize"),
    ],
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Save visualization to file"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Visualize workflow as ASCII graph.

    Example:
        agentcore workflow visualize workflow-12345

    Save to file:
        agentcore workflow visualize workflow-12345 --output graph.txt
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
        result = client.call("workflow.status", {"workflow_id": workflow_id})

        # Generate ASCII graph
        if json_output:
            console.print(format_json(result))
        else:
            graph = _create_workflow_graph(result)

            # Output or save
            if output:
                output_path = Path(output)
                with open(output_path, "w") as f:
                    # Capture without color codes for file output
                    from rich.console import Console as PlainConsole
                    plain_console = PlainConsole(file=f, force_terminal=False, no_color=True)
                    plain_console.print(graph)
                console.print(format_success(f"Workflow graph saved to: {output_path.absolute()}"))
            else:
                console.print(graph)

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
def pause(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow ID to pause"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Pause a running workflow.

    Example:
        agentcore workflow pause workflow-12345
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
        result = client.call("workflow.pause", {"workflow_id": workflow_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            console.print(format_success(f"Workflow paused: {workflow_id}"))
            console.print(f"  Resume with: agentcore workflow resume {workflow_id}")

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
def resume(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow ID to resume"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output in JSON format"),
    ] = False,
) -> None:
    """Resume a paused workflow.

    Example:
        agentcore workflow resume workflow-12345
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
        result = client.call("workflow.resume", {"workflow_id": workflow_id})

        # Output result
        if json_output:
            console.print(format_json(result))
        else:
            console.print(format_success(f"Workflow resumed: {workflow_id}"))
            console.print(f"  Check status: agentcore workflow status {workflow_id}")

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

def _watch_workflow_execution(client: AgentCoreClient, workflow_id: str, interval: int) -> None:
    """Watch workflow execution with live updates.

    Args:
        client: AgentCore client
        workflow_id: Workflow ID to watch
        interval: Update interval in seconds
    """
    console.print(f"[bold]Watching workflow: {workflow_id}[/bold]")
    console.print("[dim]Press Ctrl+C to stop watching[/dim]\n")

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                # Get current status
                result = client.call("workflow.status", {"workflow_id": workflow_id})

                # Create status display
                status_display = _create_workflow_status_display(result)
                live.update(status_display)

                # Check if workflow is in terminal state
                workflow_status = result.get("status", "").lower()
                if workflow_status in ["completed", "failed", "cancelled"]:
                    break

                # Wait before next update
                time.sleep(interval)

            except AgentCoreError as e:
                live.update(format_error(str(e)))
                break

    # Final status message
    workflow_status = result.get("status", "unknown")
    if workflow_status.lower() == "completed":
        console.print(f"\n{format_success('Workflow completed successfully')}")
    elif workflow_status.lower() == "failed":
        console.print(f"\n{format_error('Workflow failed')}")
    elif workflow_status.lower() == "cancelled":
        console.print(f"\n{format_warning('Workflow was cancelled')}")


def _create_workflow_status_display(workflow_data: dict[str, object]) -> str:
    """Create a formatted display for workflow status.

    Args:
        workflow_data: Workflow data dictionary

    Returns:
        Formatted status string
    """
    from rich.table import Table

    # Create main info table
    table = Table(title=f"Workflow: {workflow_data.get('workflow_id', 'N/A')}", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    # Add basic info
    table.add_row("Name", str(workflow_data.get("name", "N/A")))
    table.add_row("Status", _format_workflow_status(str(workflow_data.get("status", "unknown"))))

    # Task progress
    tasks_total = workflow_data.get("tasks_total", 0)
    tasks_completed = workflow_data.get("tasks_completed", 0)
    tasks_failed = workflow_data.get("tasks_failed", 0)
    tasks_running = workflow_data.get("tasks_running", 0)

    progress_str = f"{tasks_completed}/{tasks_total} completed"
    if tasks_running:
        progress_str += f", {tasks_running} running"
    if tasks_failed:
        progress_str += f", {tasks_failed} failed"

    table.add_row("Progress", progress_str)

    # Overall progress bar
    if tasks_total > 0:
        progress_pct = (tasks_completed / tasks_total) * 100
        progress_bar = _create_progress_bar(progress_pct)
        table.add_row("Completion", progress_bar)

    # Timestamps
    created_at = workflow_data.get("created_at")
    if created_at:
        table.add_row("Created", str(created_at))

    started_at = workflow_data.get("started_at")
    if started_at:
        table.add_row("Started", str(started_at))

    # Capture to string
    from rich.console import Console as TempConsole
    temp_console = TempConsole()
    with temp_console.capture() as capture:
        temp_console.print(table)

    return capture.get()


def _display_workflow_status(workflow_data: dict[str, object]) -> None:
    """Display workflow status in detailed view.

    Args:
        workflow_data: Workflow data dictionary
    """
    console.print(f"[bold]Workflow ID:[/bold] {workflow_data.get('workflow_id', 'N/A')}")
    console.print(f"[bold]Name:[/bold] {workflow_data.get('name', 'N/A')}")
    console.print(f"[bold]Status:[/bold] {_format_workflow_status(str(workflow_data.get('status', 'unknown')))}")

    # Description
    description = workflow_data.get("description")
    if description:
        console.print(f"[bold]Description:[/bold] {description}")

    # Task progress
    tasks_total = workflow_data.get("tasks_total", 0)
    tasks_completed = workflow_data.get("tasks_completed", 0)
    tasks_failed = workflow_data.get("tasks_failed", 0)
    tasks_running = workflow_data.get("tasks_running", 0)
    tasks_pending = workflow_data.get("tasks_pending", 0)

    console.print("\n[bold]Task Progress:[/bold]")
    console.print(f"  Total: {tasks_total}")
    console.print(f"  Completed: [green]{tasks_completed}[/green]")
    if tasks_running:
        console.print(f"  Running: [yellow]{tasks_running}[/yellow]")
    if tasks_pending:
        console.print(f"  Pending: [dim]{tasks_pending}[/dim]")
    if tasks_failed:
        console.print(f"  Failed: [red]{tasks_failed}[/red]")

    # Overall progress
    if tasks_total > 0:
        progress_pct = (tasks_completed / tasks_total) * 100
        progress_bar = _create_progress_bar(progress_pct)
        console.print(f"\n[bold]Overall Progress:[/bold] {progress_bar}")

    # Timestamps
    created_at = workflow_data.get("created_at")
    if created_at:
        console.print(f"\n[bold]Created:[/bold] {created_at}")

    started_at = workflow_data.get("started_at")
    if started_at:
        console.print(f"[bold]Started:[/bold] {started_at}")

    completed_at = workflow_data.get("completed_at")
    if completed_at:
        console.print(f"[bold]Completed:[/bold] {completed_at}")

    # Task details
    tasks = workflow_data.get("tasks", [])
    if tasks:
        console.print("\n[bold]Tasks:[/bold]")
        for task in tasks:
            task_name = task.get("name", "N/A")
            task_status = task.get("status", "unknown")
            task_status_colored = _format_task_status(task_status)
            console.print(f"  • {task_name}: {task_status_colored}")


def _create_workflow_graph(workflow_data: dict[str, object]) -> Tree:
    """Create an ASCII graph visualization of the workflow.

    Args:
        workflow_data: Workflow data dictionary

    Returns:
        Rich Tree object representing the workflow
    """
    workflow_name = workflow_data.get("name", "Workflow")
    workflow_status = workflow_data.get("status", "unknown")

    # Create root tree
    tree = Tree(f"[bold]{workflow_name}[/bold] ({_format_workflow_status(workflow_status)})")

    # Add tasks
    tasks = workflow_data.get("tasks", [])
    if not tasks:
        tree.add("[dim]No tasks[/dim]")
        return tree

    # Build dependency graph
    task_map = {task.get("name"): task for task in tasks}
    task_nodes = {}

    # Add tasks without dependencies first
    for task in tasks:
        task_name = task.get("name", "N/A")
        task_status = task.get("status", "unknown")
        task_type = task.get("type", "N/A")
        depends_on = task.get("depends_on", [])

        if not depends_on:
            # Root task
            task_label = f"{task_name} ({task_type}): {_format_task_status(task_status)}"
            task_nodes[task_name] = tree.add(task_label)

    # Add dependent tasks
    for task in tasks:
        task_name = task.get("name", "N/A")
        task_status = task.get("status", "unknown")
        task_type = task.get("type", "N/A")
        depends_on = task.get("depends_on", [])

        if depends_on:
            # Find parent node
            parent_name = depends_on[0] if depends_on else None
            parent_node = task_nodes.get(parent_name, tree)

            task_label = f"{task_name} ({task_type}): {_format_task_status(task_status)}"
            task_nodes[task_name] = parent_node.add(task_label)

    return tree


def _format_workflow_status(status: str) -> str:
    """Format workflow status with color.

    Args:
        status: Status string

    Returns:
        Formatted status with color markup
    """
    status_lower = status.lower()
    if status_lower in ("completed", "success"):
        return f"[green]{status}[/green]"
    if status_lower in ("running", "executing"):
        return f"[yellow]{status}[/yellow]"
    if status_lower in ("pending", "created"):
        return f"[blue]{status}[/blue]"
    if status_lower in ("failed", "error"):
        return f"[red]{status}[/red]"
    if status_lower in ("paused", "suspended"):
        return f"[dim]{status}[/dim]"
    return status


def _format_task_status(status: str) -> str:
    """Format task status with color.

    Args:
        status: Status string

    Returns:
        Formatted status with color markup
    """
    status_lower = status.lower()
    if status_lower in ("completed", "success"):
        return f"[green]✓ {status}[/green]"
    if status_lower in ("running", "executing"):
        return f"[yellow]⟳ {status}[/yellow]"
    if status_lower in ("pending", "waiting"):
        return f"[blue]○ {status}[/blue]"
    if status_lower in ("failed", "error"):
        return f"[red]✗ {status}[/red]"
    if status_lower == "skipped":
        return f"[dim]− {status}[/dim]"
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
