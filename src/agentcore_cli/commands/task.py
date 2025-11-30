"""Task management commands.

This module provides CLI commands for task lifecycle management:
- create: Create a new task
- list: List tasks with optional status filter
- info: Get detailed task information
- cancel: Cancel a running task
- logs: Get task execution logs

All commands use the service layer for business logic and abstract
JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
import json

from agentcore_cli.container import get_task_service
from agentcore_cli.services.exceptions import (
    ValidationError,
    TaskNotFoundError,
    OperationError,
    ServiceError,
)

app = typer.Typer(
    name="task",
    help="Manage task lifecycle and execution",
    no_args_is_help=True,
)

console = Console()


@app.command()
def create(
    description: Annotated[str, typer.Option("--description", "-d", help="Task description")],
    agent_id: Annotated[
        str | None,
        typer.Option(
            "--agent-id",
            "-a",
            help="Agent ID to assign task to (optional)",
        ),
    ] = None,
    priority: Annotated[
        str,
        typer.Option(
            "--priority",
            "-p",
            help="Task priority (low, normal, high, critical)",
        ),
    ] = "normal",
    parameters: Annotated[
        str | None,
        typer.Option(
            "--parameters",
            "-m",
            help="Task parameters as JSON string (optional)",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Create a new task.

    This command creates a new task with the specified description and optional
    parameters. The task can be assigned to a specific agent or left unassigned
    for automatic routing.

    Examples:
        # Create a simple task
        agentcore task create --description "Analyze code repository"

        # Create with agent assignment
        agentcore task create -d "Run tests" -a agent-001

        # Create with priority
        agentcore task create -d "Fix critical bug" -p critical

        # Create with parameters (JSON)
        agentcore task create -d "Process data" -m '{"repo": "foo/bar"}'

        # Get JSON output
        agentcore task create -d "Test" --json
    """
    # Parse parameters if provided (do this before try block)
    params_dict = None
    if parameters:
        try:
            params_dict = json.loads(parameters)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in parameters:[/red] {str(e)}")
            raise typer.Exit(2)

    try:

        # Get service from DI container
        service = get_task_service()

        # Call service method
        task_id = service.create(
            description=description,
            agent_id=agent_id,
            priority=priority,
            parameters=params_dict,
        )

        # Format output
        if json_output:
            result = {
                "task_id": task_id,
                "description": description,
                "agent_id": agent_id,
                "priority": priority,
                "parameters": params_dict,
            }
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Task created successfully")
            console.print(f"[bold]Task ID:[/bold] {task_id}")
            console.print(f"[bold]Description:[/bold] {description}")
            if agent_id:
                console.print(f"[bold]Agent ID:[/bold] {agent_id}")
            console.print(f"[bold]Priority:[/bold] {priority}")

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        if e.details:
            console.print(f"[dim]Details: {e.details}[/dim]")
        raise typer.Exit(1)
    except ServiceError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def list(
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by status (pending, running, completed, failed, cancelled)",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of tasks to return",
        ),
    ] = 100,
    offset: Annotated[
        int,
        typer.Option(
            "--offset",
            "-o",
            help="Number of tasks to skip",
        ),
    ] = 0,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """List tasks with optional filtering.

    Examples:
        # List all tasks
        agentcore task list

        # Filter by status
        agentcore task list --status running

        # Limit results
        agentcore task list --limit 10

        # Pagination
        agentcore task list --limit 10 --offset 20

        # Get JSON output
        agentcore task list --json
    """
    try:
        # Get service from DI container
        service = get_task_service()

        # Call service method
        tasks = service.list_tasks(status=status, limit=limit, offset=offset)

        # Format output
        if json_output:
            print(json.dumps(tasks, indent=2))
        else:
            if not tasks:
                console.print("[yellow]No tasks found[/yellow]")
                return

            # Create table
            table = Table(title=f"Tasks ({len(tasks)})")
            table.add_column("Task ID", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Priority", style="magenta")
            table.add_column("Agent ID", style="blue")

            for task in tasks:
                task_id = task.get("task_id", "N/A")
                description = task.get("description", "N/A")
                status_val = task.get("status", "N/A")
                priority = task.get("priority", "N/A")
                agent_id = task.get("agent_id", "N/A")

                # Truncate description if too long
                if len(description) > 50:
                    description = description[:47] + "..."

                table.add_row(task_id, description, status_val, priority, agent_id)

            console.print(table)

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
    except ServiceError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def info(
    task_id: Annotated[str, typer.Argument(help="Task identifier")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Get detailed information about a task.

    Examples:
        # Get task info
        agentcore task info task-001

        # Get JSON output
        agentcore task info task-001 --json
    """
    try:
        # Get service from DI container
        service = get_task_service()

        # Call service method
        task = service.get(task_id)

        # Format output
        if json_output:
            print(json.dumps(task, indent=2))
        else:
            console.print(f"[bold]Task Information[/bold]")
            console.print(f"[bold]ID:[/bold] {task.get('task_id', 'N/A')}")
            console.print(f"[bold]Description:[/bold] {task.get('description', 'N/A')}")
            console.print(f"[bold]Status:[/bold] {task.get('status', 'N/A')}")
            console.print(f"[bold]Priority:[/bold] {task.get('priority', 'N/A')}")
            console.print(f"[bold]Agent ID:[/bold] {task.get('agent_id', 'N/A')}")

            # Show parameters if present
            params = task.get("parameters")
            if params:
                console.print(f"[bold]Parameters:[/bold]")
                print(json.dumps(params, indent=2))

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except TaskNotFoundError as e:
        console.print(f"[red]Task not found:[/red] {e.message}")
        raise typer.Exit(1)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
    except ServiceError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def cancel(
    task_id: Annotated[str, typer.Argument(help="Task identifier")],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force cancellation even if task is running",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Cancel a task.

    Examples:
        # Cancel a task
        agentcore task cancel task-001

        # Force cancellation
        agentcore task cancel task-001 --force

        # Get JSON output
        agentcore task cancel task-001 --json
    """
    try:
        # Get service from DI container
        service = get_task_service()

        # Call service method
        success = service.cancel(task_id, force=force)

        # Format output
        if json_output:
            result = {"success": success, "task_id": task_id}
            print(json.dumps(result, indent=2))
        else:
            if success:
                console.print(f"[green]✓[/green] Task cancelled successfully")
                console.print(f"[bold]Task ID:[/bold] {task_id}")
            else:
                console.print(f"[red]Failed to cancel task {task_id}[/red]")
                raise typer.Exit(1)

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except TaskNotFoundError as e:
        console.print(f"[red]Task not found:[/red] {e.message}")
        raise typer.Exit(1)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
    except ServiceError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def logs(
    task_id: Annotated[str, typer.Argument(help="Task identifier")],
    follow: Annotated[
        bool,
        typer.Option(
            "--follow",
            "-f",
            help="Follow logs in real-time",
        ),
    ] = False,
    lines: Annotated[
        int | None,
        typer.Option(
            "--lines",
            "-n",
            help="Number of lines to retrieve (optional, default: all)",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Get task execution logs.

    Examples:
        # Get all logs
        agentcore task logs task-001

        # Get last 100 lines
        agentcore task logs task-001 --lines 100

        # Follow logs in real-time
        agentcore task logs task-001 --follow

        # Get JSON output
        agentcore task logs task-001 --json
    """
    try:
        # Get service from DI container
        service = get_task_service()

        # Call service method
        log_lines = service.logs(task_id, follow=follow, lines=lines)

        # Format output
        if json_output:
            result = {"task_id": task_id, "logs": log_lines}
            print(json.dumps(result, indent=2))
        else:
            if not log_lines:
                console.print("[yellow]No logs available[/yellow]")
                return

            console.print(f"[bold]Logs for Task {task_id}[/bold]")
            console.print("─" * 80)
            for line in log_lines:
                console.print(line)

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except TaskNotFoundError as e:
        console.print(f"[red]Task not found:[/red] {e.message}")
        raise typer.Exit(1)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
    except ServiceError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
